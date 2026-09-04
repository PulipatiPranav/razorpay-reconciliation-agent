"""Layer 2: fuzzy recovery of links Layer 1 could not make exactly.

Every tolerance below was **measured on the dev split**, never guessed and
never taken from the generator's source.  ``recon calibrate`` prints the
distributions that justify them:

* invoice issue-to-capture lag: 0-12 days on dev (p99 = 12) -> a 14-day window
* batch amount residual after the best of the two credit reconstructions:
  max 5 paise on dev -> a 10-paise tolerance
* bank value date minus settlement date: exactly 0 on dev -> a +/-2 day window
  is kept anyway, because posted-date drift and real T+ variance exist and the
  window costs nothing while the amount must still tie

Rules are tried in descending order of reliability and the first one that
yields exactly one unclaimed candidate wins.  A rule that finds several
candidates does not guess; it falls through, and if nothing resolves the
subject becomes Layer 3's residue.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

from recon.matcher.confidence import (
    L2_BATCH_AMOUNT_DEDUPED,
    L2_BATCH_AMOUNT_RECONSTRUCTION,
    L2_INVOICE_AMOUNT_WINDOW,
    L2_NARRATION_UTR_EXACT,
    L2_NARRATION_UTR_TRUNCATED,
    L2_ORDER_ID_FUZZY,
    L2_ORDER_ID_NORMALISED,
    L2_RECEIPT_INVOICE_HINT,
    L2_UTR_FUZZY,
    confidence_for,
)
from recon.matcher.resolution import Resolution
from recon.matcher.text import (
    INVOICE_PATTERN,
    damerau_levenshtein,
    extract_utrs,
    find_id_like,
    is_truncation_of,
    normalise_id,
)
from recon.matcher.types import MatchLayer
from recon.models import BankRow, InvoiceRow, PaymentView, SettlementBatch
from recon.money import format_rupees

#: Measured on dev: worst residual across all batches was 5 paise.
BATCH_AMOUNT_TOLERANCE_PAISE = 10
#: Measured on dev: bank value date always equalled the settlement date.
BATCH_DATE_WINDOW_DAYS = 2
#: Measured on dev: p99 of the invoice-issue-to-capture lag was 12 days.
INVOICE_ISSUE_WINDOW_DAYS = 14
#: Adjacent transposition scores 1 under Damerau-Levenshtein; 2 starts matching
#: unrelated identifiers, so the threshold stays at 1.
MAX_ID_EDIT_DISTANCE = 1


def _single(candidates: list[str]) -> str | None:
    return candidates[0] if len(candidates) == 1 else None


# ---------------------------------------------------------------------------
# bank leg
# ---------------------------------------------------------------------------
def _bank_narration_exact(
    batch: SettlementBatch, rows: list[BankRow]
) -> tuple[list[str], list[str]]:
    hits = [r for r in rows if batch.utr and batch.utr in extract_utrs(r.narration)]
    return [r.txn_id for r in hits], [
        f"UTR {batch.utr} recovered verbatim from the narration text",
        *[f"narration: {r.narration}" for r in hits],
    ]


def _bank_narration_truncated(
    batch: SettlementBatch, rows: list[BankRow]
) -> tuple[list[str], list[str]]:
    hits = []
    notes = []
    for row in rows:
        for candidate in extract_utrs(row.narration):
            if batch.utr and is_truncation_of(candidate, batch.utr):
                hits.append(row)
                notes.append(f"narration carries {candidate}, a prefix of {batch.utr}")
                break
    return [r.txn_id for r in hits], notes


def _bank_utr_fuzzy(batch: SettlementBatch, rows: list[BankRow]) -> tuple[list[str], list[str]]:
    hits = []
    notes = []
    for row in rows:
        candidates = list(extract_utrs(row.narration))
        if row.utr:
            candidates.append(row.utr)
        for candidate in candidates:
            if not batch.utr or len(candidate) != len(batch.utr):
                continue
            distance = damerau_levenshtein(candidate, batch.utr, max_distance=MAX_ID_EDIT_DISTANCE)
            if 0 < distance <= MAX_ID_EDIT_DISTANCE:
                hits.append(row)
                notes.append(
                    f"statement shows {candidate}, one transposition from {batch.utr}"
                )
                break
    return [r.txn_id for r in hits], notes


def _bank_amount_window(
    batch: SettlementBatch, rows: list[BankRow], *, deduped: bool
) -> tuple[list[str], list[str]]:
    expected = batch.expected_credit_dedup_paise if deduped else batch.expected_credit_paise
    if batch.settled_date is None:
        return [], []
    hits = []
    for row in rows:
        if abs(row.credit_paise - expected) > BATCH_AMOUNT_TOLERANCE_PAISE:
            continue
        if abs((row.value_date - batch.settled_date).days) > BATCH_DATE_WINDOW_DAYS:
            continue
        hits.append(row)
    label = "with duplicate refund rows collapsed" if deduped else "as reported"
    notes = [
        f"batch credits less debits {label} = {format_rupees(expected)}",
        f"sought within {BATCH_AMOUNT_TOLERANCE_PAISE} paise of a credit "
        f"dated {batch.settled_date} +/-{BATCH_DATE_WINDOW_DAYS}d",
        *[
            f"bank credit {r.txn_id} of {format_rupees(r.credit_paise)} on {r.value_date} "
            f"(residual {r.credit_paise - expected} paise)"
            for r in hits
        ],
    ]
    if deduped and batch.has_duplicate_refund_rows:
        notes.insert(1, "report contains a refund row duplicated for the same payment and amount")
    return [r.txn_id for r in hits], notes


BankRule = Callable[[SettlementBatch, list[BankRow]], tuple[list[str], list[str]]]

BANK_RULES: list[tuple[str, BankRule]] = [
    (L2_NARRATION_UTR_EXACT, _bank_narration_exact),
    (L2_NARRATION_UTR_TRUNCATED, _bank_narration_truncated),
    (L2_UTR_FUZZY, _bank_utr_fuzzy),
    (L2_BATCH_AMOUNT_RECONSTRUCTION, lambda b, r: _bank_amount_window(b, r, deduped=False)),
    (L2_BATCH_AMOUNT_DEDUPED, lambda b, r: _bank_amount_window(b, r, deduped=True)),
]


def resolve_batches_fuzzy(
    batches: list[SettlementBatch],
    bank_rows: list[BankRow],
    claimed: set[str] | None = None,
) -> dict[str, Resolution]:
    """Recover batch-to-credit links the exact UTR join could not make."""
    claimed = set(claimed or set())
    out: dict[str, Resolution] = {}
    for batch in batches:
        available = [r for r in bank_rows if r.txn_id not in claimed]
        for rule, fn in BANK_RULES:
            candidates, notes = fn(batch, available)
            winner = _single(candidates)
            if winner is None:
                continue
            out[batch.settlement_id] = Resolution(
                subject_id=batch.settlement_id,
                counterpart_ids=[winner],
                layer=MatchLayer.L2_FUZZY,
                rule=rule,
                confidence=confidence_for(rule),
                evidence=[*notes, f"batch covers {len(batch.payment_ids)} payment(s)"],
            )
            claimed.add(winner)
            break
    return out


# ---------------------------------------------------------------------------
# invoice leg
# ---------------------------------------------------------------------------
def _invoice_receipt_hint(
    payment: PaymentView, invoices: list[InvoiceRow]
) -> tuple[list[str], list[str]]:
    hinted = set(find_id_like(payment.order_receipt, INVOICE_PATTERN))
    if not hinted:
        return [], []
    hits = [i for i in invoices if i.invoice_id in hinted]
    return [i.invoice_id for i in hits], [
        f"settlement receipt field carries {', '.join(sorted(hinted))}",
        "that invoice number exists in the ledger",
    ]


def _invoice_order_normalised(
    payment: PaymentView, invoices: list[InvoiceRow]
) -> tuple[list[str], list[str]]:
    if payment.order_id is None:
        return [], []
    target = normalise_id(payment.order_id)
    hits = [i for i in invoices if i.order_id and normalise_id(i.order_id) == target]
    return [i.invoice_id for i in hits], [
        f"ERP order_id matches {payment.order_id} once case and separators are folded",
        *[f"ledger shows {i.order_id}" for i in hits],
    ]


def _invoice_order_fuzzy(
    payment: PaymentView, invoices: list[InvoiceRow]
) -> tuple[list[str], list[str]]:
    if payment.order_id is None:
        return [], []
    hits = []
    notes = []
    for invoice in invoices:
        if not invoice.order_id or len(invoice.order_id) != len(payment.order_id):
            continue
        distance = damerau_levenshtein(
            invoice.order_id, payment.order_id, max_distance=MAX_ID_EDIT_DISTANCE
        )
        if 0 < distance <= MAX_ID_EDIT_DISTANCE:
            hits.append(invoice)
            notes.append(f"ledger shows {invoice.order_id}, one transposition from the gateway's")
    return [i.invoice_id for i in hits], notes


def _invoice_amount_window(
    payment: PaymentView, invoices: list[InvoiceRow]
) -> tuple[list[str], list[str]]:
    if payment.captured_at is None:
        return [], []
    captured = payment.captured_at.date()
    window_start = captured - timedelta(days=INVOICE_ISSUE_WINDOW_DAYS)
    hits = [
        i
        for i in invoices
        if i.invoice_amount_paise == payment.gross_paise
        and window_start <= i.issue_date <= captured
    ]
    return [i.invoice_id for i in hits], [
        f"gross {format_rupees(payment.gross_paise)} sought as an exact invoice amount",
        f"issued within {INVOICE_ISSUE_WINDOW_DAYS} days before capture on {captured}",
        f"{len(hits)} invoice(s) in that window carry the amount",
    ]


InvoiceRule = Callable[[PaymentView, list[InvoiceRow]], tuple[list[str], list[str]]]

INVOICE_RULES: list[tuple[str, InvoiceRule]] = [
    (L2_RECEIPT_INVOICE_HINT, _invoice_receipt_hint),
    (L2_ORDER_ID_NORMALISED, _invoice_order_normalised),
    (L2_ORDER_ID_FUZZY, _invoice_order_fuzzy),
    (L2_INVOICE_AMOUNT_WINDOW, _invoice_amount_window),
]


def resolve_invoices_fuzzy(
    payments: list[PaymentView],
    invoices: list[InvoiceRow],
    claimed: set[str] | None = None,
) -> dict[str, Resolution]:
    """Recover payment-to-invoice links the exact order_id join could not make."""
    claimed = set(claimed or set())
    out: dict[str, Resolution] = {}
    for payment in payments:
        available = [i for i in invoices if i.invoice_id not in claimed]
        for rule, fn in INVOICE_RULES:
            candidates, notes = fn(payment, available)
            winner = _single(candidates)
            if winner is None:
                continue
            out[payment.payment_id] = Resolution(
                subject_id=payment.payment_id,
                counterpart_ids=[winner],
                layer=MatchLayer.L2_FUZZY,
                rule=rule,
                confidence=confidence_for(rule),
                evidence=notes,
            )
            claimed.add(winner)
            break
    return out
