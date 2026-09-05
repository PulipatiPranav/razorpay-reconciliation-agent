"""Layer 3: Claude on the residue only.

This layer never sees a record that Layers 1-2 could resolve.  By the time it
runs, what is left is genuinely hard: settlement batches whose bank credit
carries no reference of any kind *and* whose total does not tie because of an
unitemised deduction, and payments whose ERP link is broken in a way no string
rule recovers.

Three rules govern it, all of them about not being talked into a match:

1. **Declining is a correct answer.**  About 5% of the corpus has no
   counterpart.  The prompt says so explicitly and the schema makes ``null``
   a first-class value rather than an error path.
2. **Its confidence is capped.**  Whatever the model reports is multiplied by
   :data:`LLM_CONFIDENCE_CEILING`, so an LLM proposal can never outrank a
   deterministic rule whose precision was actually measured.
3. **It gets candidates, not the corpus.**  Only unclaimed counterparts in a
   plausible window are offered, so the model is choosing among a short list
   rather than searching.

The module itself is pure: it takes an :class:`LLMClient` and returns
resolutions.  No file access, no client construction.
"""

from __future__ import annotations

from datetime import timedelta

from recon.llm.client import LLMClient
from recon.llm.schemas import BANK_DECISION_SCHEMA, INVOICE_DECISION_SCHEMA, LinkDecision
from recon.matcher.confidence import (
    L3_LLM_BANK,
    L3_LLM_INVOICE,
    LLM_CONFIDENCE_CEILING,
)
from recon.matcher.resolution import Resolution
from recon.matcher.types import MatchLayer
from recon.models import BankRow, InvoiceRow, PaymentView, SettlementBatch
from recon.money import format_rupees
from recon.obs.logging import CallRecord

#: How far from the settlement date a credit may sit and still be offered.
CANDIDATE_WINDOW_DAYS = 4
#: How far from the capture date an invoice may be issued and still be offered.
INVOICE_CANDIDATE_WINDOW_DAYS = 21
#: A credit more than this far from the batch total is not plausibly the same
#: payout even allowing for an unitemised deduction.
MAX_PLAUSIBLE_SHORTFALL_PCT = 5.0

SYSTEM_PROMPT = """\
You are a reconciliation analyst for an Indian payment gateway. You match \
settlement batches to bank credits and payments to ERP invoices.

Facts about this domain that you should rely on:
- One bank credit covers a whole batch of payments, not a single payment.
- The credit equals the batch's gross less gateway fee, 18% GST on that fee, \
and sometimes 1% TDS. Those are already itemised in the figures you are given.
- A credit can be short of the batch total by an amount nothing in the report \
explains: a rolling reserve, a platform fee, or an FX adjustment. These are \
typically 0.2% to 1% of the batch. A shortfall in that range is evidence for \
a match, not against it.
- Bank value dates lag settlement by 0 to 2 days.
- Paise-level differences of a few units are rounding, not evidence.

Rules you must follow:
- Answering null is correct and expected. Roughly one record in twenty has no \
counterpart in the other system at all. Asserting a match that does not exist \
silently closes a real discrepancy, which is worse than leaving it open.
- If two or more candidates fit comparably well, answer null. Do not pick the \
closest one.
- Your confidence must reflect the evidence you can actually cite. Reserve \
values above 0.8 for cases where one candidate fits and no other comes close.
- Your reasoning must cite the specific figures that drove the decision, in \
one or two sentences. An auditor will read it.
"""


def _bank_prompt(
    batch: SettlementBatch, candidates: list[BankRow], competing: list[SettlementBatch]
) -> str:
    lines = [
        "Settlement batch to identify in the bank statement:",
        f"  settlement_id: {batch.settlement_id}",
        f"  settled on: {batch.settled_date}",
        f"  payments in batch: {len(batch.payment_ids)}",
        f"  credits less debits as reported: {format_rupees(batch.expected_credit_paise)}",
    ]
    if batch.expected_credit_dedup_paise != batch.expected_credit_paise:
        lines.append(
            "  same figure with duplicate refund rows collapsed: "
            f"{format_rupees(batch.expected_credit_dedup_paise)}"
        )
        lines.append(
            "  (the report shows a refund twice for the same payment and amount; "
            "the bank may have moved it once)"
        )
    if batch.refund_entity_ids:
        lines.append(f"  refund rows in batch: {len(batch.refund_entity_ids)}")
    if batch.adjustment_entity_ids:
        lines.append(f"  adjustment rows in batch: {len(batch.adjustment_entity_ids)}")
    lines.append("  the bank statement carries no usable UTR for this batch")

    lines.append("")
    lines.append("Unclaimed bank credits near that date:")
    for row in candidates:
        shortfall = batch.expected_credit_dedup_paise - row.credit_paise
        expected = batch.expected_credit_dedup_paise
        pct = 100 * shortfall / expected if expected else 0
        lines.append(
            f"  - txn_id {row.txn_id} | {format_rupees(row.credit_paise)} "
            f"| value date {row.value_date} "
            f"| short of batch total by {format_rupees(shortfall)} ({pct:.2f}%)"
        )
        lines.append(f"    narration: {row.narration}")

    if competing:
        lines.append("")
        lines.append("Other unresolved batches settling nearby, which may claim the same credits:")
        for other in competing:
            lines.append(
                f"  - {other.settlement_id} | settled {other.settled_date} "
                f"| total {format_rupees(other.expected_credit_dedup_paise)}"
            )

    lines.append("")
    lines.append(
        "Return the txn_id of the credit that is this batch's payout, or null. "
        "If you attribute a shortfall to an unitemised deduction, report it in paise."
    )
    return "\n".join(lines)


def _invoice_prompt(payment: PaymentView, candidates: list[InvoiceRow]) -> str:
    captured = payment.captured_at.date() if payment.captured_at else None
    lines = [
        "Payment to match to an ERP invoice:",
        f"  payment_id: {payment.payment_id}",
        f"  gross amount: {format_rupees(payment.gross_paise)}",
        f"  captured: {captured}",
        f"  order_id on the settlement row: {payment.order_id or 'absent'}",
        f"  receipt field: {payment.order_receipt or 'empty'}",
        f"  method: {payment.method.value if payment.method else 'unknown'}",
        "",
        "Unclaimed invoices with a compatible amount or date:",
    ]
    for invoice in candidates:
        delta = invoice.invoice_amount_paise - payment.gross_paise
        lines.append(
            f"  - {invoice.invoice_id} | {format_rupees(invoice.invoice_amount_paise)} "
            f"(differs from gross by {format_rupees(delta)}) "
            f"| issued {invoice.issue_date} | status {invoice.status}"
        )
        lines.append(
            f"    customer: {invoice.customer_name} | order_id: {invoice.order_id or 'absent'}"
        )
    if not candidates:
        lines.append("  (none)")
    lines.append("")
    lines.append(
        "Return the invoice_id this payment settles, or null if none of them is "
        "justified. Two invoices for the same customer and amount are common and "
        "are not distinguishable on amount alone."
    )
    return "\n".join(lines)


def _evidence(record: CallRecord, reasoning: str, raw_confidence: float, rule: str) -> list[str]:
    return [
        f"Layer 3 ({record.model}) was asked to choose among the remaining candidates",
        f"model reasoning: {reasoning}",
        f"model confidence {raw_confidence:.2f}, capped to "
        f"{raw_confidence * LLM_CONFIDENCE_CEILING:.2f} by rule {rule}",
        f"call {record.call_id}"
        + (f", request {record.request_id}" if record.request_id else "")
        + (" (replayed from transcript)" if record.replayed else ""),
    ]


def _decline_evidence(
    record: CallRecord, decision: LinkDecision, offered: set[str], n_candidates: int
) -> list[str]:
    """Record a refusal as carefully as a match.

    Two different refusals land here and they are not the same thing, so the
    evidence says which: the model declining outright, and the model naming an
    identifier that was never offered -- a hallucination the pipeline downgrades
    to a decline rather than trusting.
    """
    lines = [
        f"Layer 3 ({record.model}) considered {n_candidates} candidate(s) and declined",
        f"model reasoning: {decision.reasoning}",
    ]
    if decision.chosen_id is not None and decision.chosen_id not in offered:
        lines.append(
            f"the model named {decision.chosen_id}, which was not among the candidates "
            "offered -- treated as a decline rather than trusted"
        )
    lines.append(
        f"model confidence {decision.confidence:.2f}"
        + (f", call {record.call_id}" if record.call_id else "")
        + (" (replayed from transcript)" if record.replayed else "")
    )
    return lines


class LLMResolver:
    """Layer 3 as the pipeline sees it: batches in, resolutions out."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.declined: list[str] = []
        self.schema_failures: list[str] = []
        #: Why the model refused, keyed by subject id.  A decline is a decision,
        #: and an audit trail that records the matches but not the refusals is
        #: only half a trail -- the refusals are where a reviewer most wants to
        #: know what was considered.
        self.decline_evidence: dict[str, list[str]] = {}

    def resolve_batches(
        self, batches: list[SettlementBatch], bank_rows: list[BankRow]
    ) -> dict[str, Resolution]:
        out: dict[str, Resolution] = {}
        claimed: set[str] = set()
        for batch in batches:
            if batch.settled_date is None:
                continue
            candidates = [
                row
                for row in bank_rows
                if row.txn_id not in claimed
                and abs((row.value_date - batch.settled_date).days) <= CANDIDATE_WINDOW_DAYS
                and _within_plausible_shortfall(batch, row)
            ]
            if not candidates:
                continue
            competing = [
                other
                for other in batches
                if other.settlement_id != batch.settlement_id
                and other.settled_date is not None
                and abs((other.settled_date - batch.settled_date).days) <= CANDIDATE_WINDOW_DAYS
            ]
            decision, record = self.client.decide(
                purpose=f"bank:{batch.settlement_id}",
                system=SYSTEM_PROMPT,
                user=_bank_prompt(batch, candidates, competing),
                schema=BANK_DECISION_SCHEMA,
            )
            if decision is None:
                self.schema_failures.append(batch.settlement_id)
                continue
            valid_ids = {row.txn_id for row in candidates}
            if decision.chosen_id is None or decision.chosen_id not in valid_ids:
                # A chosen id outside the offered list is a hallucination, and
                # is treated exactly like a decline rather than trusted.
                self.declined.append(batch.settlement_id)
                self.decline_evidence[batch.settlement_id] = _decline_evidence(
                    record, decision, valid_ids, len(candidates)
                )
                continue
            evidence = _evidence(record, decision.reasoning, decision.confidence, L3_LLM_BANK)
            if decision.inferred_deduction_paise:
                evidence.insert(
                    2,
                    "model attributed a shortfall of "
                    f"{format_rupees(decision.inferred_deduction_paise)} to an "
                    "unitemised deduction",
                )
            out[batch.settlement_id] = Resolution(
                subject_id=batch.settlement_id,
                counterpart_ids=[decision.chosen_id],
                layer=MatchLayer.L3_LLM,
                rule=L3_LLM_BANK,
                confidence=decision.confidence * LLM_CONFIDENCE_CEILING,
                evidence=evidence,
            )
            claimed.add(decision.chosen_id)
        return out

    def resolve_invoices(
        self, payments: list[PaymentView], invoices: list[InvoiceRow]
    ) -> dict[str, Resolution]:
        out: dict[str, Resolution] = {}
        claimed: set[str] = set()
        for payment in payments:
            if payment.captured_at is None:
                continue
            captured = payment.captured_at.date()
            window = timedelta(days=INVOICE_CANDIDATE_WINDOW_DAYS)
            candidates = [
                invoice
                for invoice in invoices
                if invoice.invoice_id not in claimed
                and abs(invoice.issue_date - captured) <= window
                and abs(invoice.invoice_amount_paise - payment.gross_paise)
                <= payment.gross_paise * 0.15
            ]
            if not candidates:
                continue
            decision, record = self.client.decide(
                purpose=f"invoice:{payment.payment_id}",
                system=SYSTEM_PROMPT,
                user=_invoice_prompt(payment, candidates[:12]),
                schema=INVOICE_DECISION_SCHEMA,
            )
            if decision is None:
                self.schema_failures.append(payment.payment_id)
                continue
            valid_ids = {invoice.invoice_id for invoice in candidates[:12]}
            if decision.chosen_id is None or decision.chosen_id not in valid_ids:
                self.declined.append(payment.payment_id)
                self.decline_evidence[payment.payment_id] = _decline_evidence(
                    record, decision, valid_ids, len(candidates[:12])
                )
                continue
            out[payment.payment_id] = Resolution(
                subject_id=payment.payment_id,
                counterpart_ids=[decision.chosen_id],
                layer=MatchLayer.L3_LLM,
                rule=L3_LLM_INVOICE,
                confidence=decision.confidence * LLM_CONFIDENCE_CEILING,
                evidence=_evidence(record, decision.reasoning, decision.confidence, L3_LLM_INVOICE),
            )
            claimed.add(decision.chosen_id)
        return out


def _within_plausible_shortfall(batch: SettlementBatch, row: BankRow) -> bool:
    expected = batch.expected_credit_dedup_paise
    if expected <= 0:
        return False
    shortfall = expected - row.credit_paise
    if shortfall < -BATCH_OVERSHOOT_TOLERANCE_PAISE:
        return False  # the bank paid more than the batch is worth
    return (100 * shortfall / expected) <= MAX_PLAUSIBLE_SHORTFALL_PCT


#: A credit may exceed the batch total only by rounding.
BATCH_OVERSHOOT_TOLERANCE_PAISE = 10
