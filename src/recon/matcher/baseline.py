"""Phase 2 deterministic baseline: exact matching, zero parameters.

This is the number the layered matcher has to beat.  It is deliberately naive
and it has nothing to tune -- no tolerances, no windows, no thresholds -- which
is why running it on the held-out set is not a form of peeking.

Two variants are provided.  ``exact_amount_and_date`` is the rule the brief
specifies.  ``exact_amount_only`` drops the date predicate, which is the
stronger strawman: it is closer to what a finance team actually does in a
spreadsheet, and beating it is a more convincing claim than beating the weaker
one.  Both are reported.

The rule in both cases is *unique* exact match: zero candidates is an
exception, and so is more than one.  Picking arbitrarily among equal candidates
would inflate the baseline's match rate with coin flips and destroy its
precision, which would flatter the layered matcher for the wrong reason.
"""

from __future__ import annotations

from recon.matcher.types import (
    ExceptionReason,
    LinkType,
    Match,
    MatchLayer,
    ReconException,
    ReconResult,
    SubjectType,
)
from recon.models import BankRow, InvoiceRow, PaymentView, SourceBundle
from recon.money import format_rupees

RULE_AMOUNT_AND_DATE = "exact_amount_and_date"
RULE_AMOUNT_ONLY = "exact_amount_only"


def _bank_candidates(
    payment: PaymentView, bank_rows: list[BankRow], *, use_date: bool
) -> list[BankRow]:
    """Bank credits whose amount (and optionally value date) match exactly."""
    out = []
    for row in bank_rows:
        if row.credit_paise != payment.net_paise:
            continue
        if use_date and row.value_date not in payment.settled_dates:
            continue
        out.append(row)
    return out


def _invoice_candidates(
    payment: PaymentView, invoices: list[InvoiceRow], *, use_date: bool
) -> list[InvoiceRow]:
    """Invoices whose amount (and optionally issue date) match exactly."""
    captured = payment.captured_at.date() if payment.captured_at else None
    out = []
    for invoice in invoices:
        if invoice.invoice_amount_paise != payment.gross_paise:
            continue
        if use_date and (captured is None or invoice.issue_date != captured):
            continue
        out.append(invoice)
    return out


def _resolve(
    payment: PaymentView,
    link_type: LinkType,
    candidate_ids: list[str],
    rule: str,
    evidence: list[str],
) -> tuple[Match | None, ReconException | None]:
    """Unique candidate wins; zero or many becomes a typed exception."""
    if len(candidate_ids) == 1:
        return (
            Match.build(
                link_type=link_type,
                payment_id=payment.payment_id,
                counterpart_ids=candidate_ids,
                layer=MatchLayer.BASELINE,
                rule=rule,
                confidence=1.0,
                evidence=evidence,
                source_records={"gateway": [payment.payment_id], "counterpart": candidate_ids},
            ),
            None,
        )
    if not candidate_ids:
        reason, detail = ExceptionReason.NO_CANDIDATE, "no record matched exactly"
    else:
        reason = ExceptionReason.AMBIGUOUS_CANDIDATES
        detail = f"{len(candidate_ids)} records matched equally well"
    return (
        None,
        ReconException.build(
            subject_type=SubjectType.PAYMENT,
            subject_id=payment.payment_id,
            link_type=link_type,
            reason=reason,
            detail=detail,
            layer_reached=MatchLayer.BASELINE,
            candidates_considered=len(candidate_ids),
            evidence=evidence,
        ),
    )


def run_baseline(sources: SourceBundle, split: str, *, use_date: bool = True) -> ReconResult:
    """Match every payment to a bank credit and an invoice by exact equality."""
    rule = RULE_AMOUNT_AND_DATE if use_date else RULE_AMOUNT_ONLY
    matches: list[Match] = []
    exceptions: list[ReconException] = []
    matched_bank: set[str] = set()
    matched_invoices: set[str] = set()

    for payment in sources.payments:
        bank_hits = _bank_candidates(payment, sources.bank_rows, use_date=use_date)
        dates = ", ".join(d.isoformat() for d in payment.settled_dates) or "none"
        evidence = [
            f"net settled {format_rupees(payment.net_paise)} sought as an exact bank credit",
            f"settlement date(s) {dates}" if use_date else "date predicate not applied",
            f"{len(bank_hits)} exact candidate(s) in the statement",
        ]
        match, exc = _resolve(
            payment, LinkType.PAYMENT_TO_BANK, [r.txn_id for r in bank_hits], rule, evidence
        )
        if match:
            matches.append(match)
            matched_bank.update(match.counterpart_ids)
        if exc:
            exceptions.append(exc)

        invoice_hits = _invoice_candidates(payment, sources.invoices, use_date=use_date)
        captured = payment.captured_at.date().isoformat() if payment.captured_at else "unknown"
        evidence = [
            f"gross {format_rupees(payment.gross_paise)} sought as an exact invoice amount",
            f"capture date {captured}" if use_date else "date predicate not applied",
            f"{len(invoice_hits)} exact candidate(s) in the ledger",
        ]
        match, exc = _resolve(
            payment,
            LinkType.PAYMENT_TO_INVOICE,
            [i.invoice_id for i in invoice_hits],
            rule,
            evidence,
        )
        if match:
            matches.append(match)
            matched_invoices.update(match.counterpart_ids)
        if exc:
            exceptions.append(exc)

    # Counterparts nothing claimed.  Some of these are genuinely orphaned and
    # belong here; the scorer is what decides whether they were right to be.
    for row in sources.bank_rows:
        if row.txn_id not in matched_bank:
            exceptions.append(
                ReconException.build(
                    subject_type=SubjectType.BANK_TXN,
                    subject_id=row.txn_id,
                    link_type=LinkType.PAYMENT_TO_BANK,
                    reason=ExceptionReason.UNMATCHED_COUNTERPART,
                    detail="bank credit claimed by no payment",
                    layer_reached=MatchLayer.BASELINE,
                    evidence=[
                        f"credit {format_rupees(row.credit_paise)} on {row.value_date}",
                        f"narration: {row.narration}",
                    ],
                )
            )
    for invoice in sources.invoices:
        if invoice.invoice_id not in matched_invoices:
            exceptions.append(
                ReconException.build(
                    subject_type=SubjectType.INVOICE,
                    subject_id=invoice.invoice_id,
                    link_type=LinkType.PAYMENT_TO_INVOICE,
                    reason=ExceptionReason.UNMATCHED_COUNTERPART,
                    detail="invoice claimed by no payment",
                    layer_reached=MatchLayer.BASELINE,
                    evidence=[
                        f"{invoice.invoice_id} for {format_rupees(invoice.invoice_amount_paise)}",
                        f"issued {invoice.issue_date}, customer {invoice.customer_name}",
                    ],
                )
            )

    return ReconResult(
        matcher=f"baseline:{rule}",
        split=split,
        matches=matches,
        exceptions=exceptions,
        stats={"payments": float(len(sources.payments))},
    )


RULE_UTR_JOIN = "exact_utr_join"
RULE_ORDER_JOIN = "exact_order_id_join"


def run_id_join_baseline(sources: SourceBundle, split: str) -> ReconResult:
    """A third, stronger baseline: join on the identifiers that are meant to join.

    Amount-and-date matching cannot resolve the bank leg at all -- one credit
    covers 4 to 60 payments, so no single payment's net ever equals a credit.
    Reporting only that would set up a strawman.  The honest thing a competent
    finance team does first is join the settlement UTR to the bank statement's
    UTR column, and the ERP's ``order_id`` to the gateway's.  This baseline does
    exactly that, deterministically and with no tolerances, and it is the number
    the layered matcher genuinely has to beat.

    It fails precisely where the identifiers are broken: corrupted narrations
    (27% of batches) and broken ERP links (9% of payments).
    """
    matches: list[Match] = []
    exceptions: list[ReconException] = []
    matched_bank: set[str] = set()
    matched_invoices: set[str] = set()

    by_utr: dict[str, list[BankRow]] = {}
    for row in sources.bank_rows:
        if row.utr:
            by_utr.setdefault(row.utr, []).append(row)
    by_order: dict[str, list[InvoiceRow]] = {}
    for invoice in sources.invoices:
        if invoice.order_id:
            by_order.setdefault(invoice.order_id, []).append(invoice)

    for payment in sources.payments:
        txn_ids: list[str] = []
        misses: list[str] = []
        for utr in payment.settlement_utrs:
            bank_hits = by_utr.get(utr, [])
            if len(bank_hits) == 1:
                txn_ids.append(bank_hits[0].txn_id)
            else:
                misses.append(f"UTR {utr}: {len(bank_hits)} candidate(s) in the statement")
        evidence = [
            f"settlement UTR(s) {', '.join(payment.settlement_utrs) or 'none'}",
            f"resolved {len(txn_ids)} of {len(payment.settlement_utrs)} by exact UTR column join",
            *misses,
        ]
        if txn_ids and not misses:
            match = Match.build(
                link_type=LinkType.PAYMENT_TO_BANK,
                payment_id=payment.payment_id,
                counterpart_ids=txn_ids,
                layer=MatchLayer.BASELINE,
                rule=RULE_UTR_JOIN,
                confidence=1.0,
                evidence=evidence,
                source_records={"gateway": [payment.payment_id], "bank": txn_ids},
            )
            matches.append(match)
            matched_bank.update(txn_ids)
        else:
            exceptions.append(
                ReconException.build(
                    subject_type=SubjectType.PAYMENT,
                    subject_id=payment.payment_id,
                    link_type=LinkType.PAYMENT_TO_BANK,
                    reason=ExceptionReason.NO_CANDIDATE
                    if not txn_ids
                    else ExceptionReason.AMBIGUOUS_CANDIDATES,
                    detail="UTR did not resolve to exactly one bank credit",
                    layer_reached=MatchLayer.BASELINE,
                    candidates_considered=len(txn_ids),
                    evidence=evidence,
                )
            )

        if payment.order_id is None:
            exceptions.append(
                ReconException.build(
                    subject_type=SubjectType.PAYMENT,
                    subject_id=payment.payment_id,
                    link_type=LinkType.PAYMENT_TO_INVOICE,
                    reason=ExceptionReason.NO_ORDER_ID,
                    detail="gateway row carries no order_id to join on",
                    layer_reached=MatchLayer.BASELINE,
                    evidence=["no order_id on the settlement row"],
                )
            )
            continue

        invoice_hits = by_order.get(payment.order_id, [])
        invoice_evidence = [
            f"order_id {payment.order_id} sought in the ERP ledger",
            f"{len(invoice_hits)} invoice(s) carry that order_id",
        ]
        invoice_match, invoice_exc = _resolve(
            payment,
            LinkType.PAYMENT_TO_INVOICE,
            [i.invoice_id for i in invoice_hits],
            RULE_ORDER_JOIN,
            invoice_evidence,
        )
        if invoice_match:
            matches.append(invoice_match)
            matched_invoices.update(invoice_match.counterpart_ids)
        if invoice_exc:
            exceptions.append(invoice_exc)

    for row in sources.bank_rows:
        if row.txn_id not in matched_bank:
            exceptions.append(
                ReconException.build(
                    subject_type=SubjectType.BANK_TXN,
                    subject_id=row.txn_id,
                    link_type=LinkType.PAYMENT_TO_BANK,
                    reason=ExceptionReason.UNMATCHED_COUNTERPART,
                    detail="bank credit claimed by no payment",
                    layer_reached=MatchLayer.BASELINE,
                    evidence=[
                        f"credit {format_rupees(row.credit_paise)} on {row.value_date}",
                        f"narration: {row.narration}",
                    ],
                )
            )
    for invoice in sources.invoices:
        if invoice.invoice_id not in matched_invoices:
            exceptions.append(
                ReconException.build(
                    subject_type=SubjectType.INVOICE,
                    subject_id=invoice.invoice_id,
                    link_type=LinkType.PAYMENT_TO_INVOICE,
                    reason=ExceptionReason.UNMATCHED_COUNTERPART,
                    detail="invoice claimed by no payment",
                    layer_reached=MatchLayer.BASELINE,
                    evidence=[
                        f"{invoice.invoice_id} for {format_rupees(invoice.invoice_amount_paise)}",
                        f"issued {invoice.issue_date}, customer {invoice.customer_name}",
                    ],
                )
            )

    return ReconResult(
        matcher="baseline:exact_id_join",
        split=split,
        matches=matches,
        exceptions=exceptions,
        stats={"payments": float(len(sources.payments))},
    )
