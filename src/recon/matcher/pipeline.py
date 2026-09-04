"""The layered matcher: exact, then fuzzy, then the LLM on what is left.

The pipeline itself is pure.  Layer 3 arrives as an injected resolver object,
so the whole thing can be exercised without a network call -- with no resolver,
with a stub, or with a recorded transcript.  That is what makes the layered
matcher unit-testable rather than only end-to-end testable.

Two structural decisions worth naming:

**The bank leg is resolved at batch level, then propagated.**  A bank credit
corresponds to a settlement batch, and every payment in the batch inherits the
answer.  Resolving per payment would ask the same many-to-one question 4 to 60
times over and invite the layers to disagree with themselves.

**A counterpart can only be claimed once.**  A bank credit belongs to exactly
one batch and an invoice to one payment, so each is removed from the pool as
soon as it is taken.  Letting two subjects claim the same record would buy a
recall point at the cost of a precision loss on both.
"""

from __future__ import annotations

from typing import Protocol

from recon.matcher.confidence import DEFAULT_CONFIDENCE_THRESHOLD
from recon.matcher.layer1 import resolve_batches_exact, resolve_invoices_exact
from recon.matcher.layer2 import resolve_batches_fuzzy, resolve_invoices_fuzzy
from recon.matcher.resolution import Resolution
from recon.matcher.types import (
    ExceptionReason,
    LinkType,
    Match,
    MatchLayer,
    ReconException,
    ReconResult,
    SubjectType,
)
from recon.models import BankRow, InvoiceRow, PaymentView, SettlementBatch, SourceBundle
from recon.money import format_rupees


class Layer3Resolver(Protocol):
    """Whatever resolves the residue Layers 1-2 refused to guess at."""

    def resolve_batches(
        self, batches: list[SettlementBatch], bank_rows: list[BankRow]
    ) -> dict[str, Resolution]: ...

    def resolve_invoices(
        self, payments: list[PaymentView], invoices: list[InvoiceRow]
    ) -> dict[str, Resolution]: ...


def _claimed(resolutions: dict[str, Resolution]) -> set[str]:
    return {cid for res in resolutions.values() for cid in res.counterpart_ids}


def resolve_bank_leg(
    sources: SourceBundle, resolver: Layer3Resolver | None
) -> dict[str, Resolution]:
    """Batch -> bank credit, tried exactly, then fuzzily, then by the LLM."""
    resolved = resolve_batches_exact(sources.batches, sources.bank_rows)

    residue = [b for b in sources.batches if b.settlement_id not in resolved]
    resolved |= resolve_batches_fuzzy(residue, sources.bank_rows, claimed=_claimed(resolved))

    if resolver is not None:
        residue = [b for b in sources.batches if b.settlement_id not in resolved]
        if residue:
            available = [r for r in sources.bank_rows if r.txn_id not in _claimed(resolved)]
            for settlement_id, res in resolver.resolve_batches(residue, available).items():
                if settlement_id not in resolved:
                    resolved[settlement_id] = res
    return resolved


def resolve_invoice_leg(
    sources: SourceBundle, resolver: Layer3Resolver | None
) -> dict[str, Resolution]:
    """Payment -> invoice, tried exactly, then fuzzily, then by the LLM."""
    resolved = resolve_invoices_exact(sources.payments, sources.invoices)

    residue = [p for p in sources.payments if p.payment_id not in resolved]
    resolved |= resolve_invoices_fuzzy(residue, sources.invoices, claimed=_claimed(resolved))

    if resolver is not None:
        residue = [p for p in sources.payments if p.payment_id not in resolved]
        if residue:
            available = [i for i in sources.invoices if i.invoice_id not in _claimed(resolved)]
            for payment_id, res in resolver.resolve_invoices(residue, available).items():
                if payment_id not in resolved:
                    resolved[payment_id] = res
    return resolved


def _exception_from(
    subject_id: str,
    link_type: LinkType,
    resolution: Resolution | None,
    threshold: float,
    detail: str,
) -> ReconException:
    if resolution is None:
        return ReconException.build(
            subject_type=SubjectType.PAYMENT,
            subject_id=subject_id,
            link_type=link_type,
            reason=ExceptionReason.NO_CANDIDATE,
            detail=detail,
            layer_reached=MatchLayer.L3_LLM,
            evidence=[detail],
        )
    return ReconException.build(
        subject_type=SubjectType.PAYMENT,
        subject_id=subject_id,
        link_type=link_type,
        reason=ExceptionReason.BELOW_CONFIDENCE,
        detail=(
            f"{resolution.rule} proposed a link at {resolution.confidence:.2f}, "
            f"below the {threshold:.2f} threshold"
        ),
        layer_reached=resolution.layer,
        candidates_considered=len(resolution.counterpart_ids),
        evidence=resolution.evidence,
    )


def run_layered(
    sources: SourceBundle,
    split: str,
    *,
    resolver: Layer3Resolver | None = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ReconResult:
    """Run all available layers and turn their verdicts into matches and exceptions."""
    bank = resolve_bank_leg(sources, resolver)
    invoice = resolve_invoice_leg(sources, resolver)

    matches: list[Match] = []
    exceptions: list[ReconException] = []
    claimed_bank: set[str] = set()
    claimed_invoices: set[str] = set()
    layer_counts: dict[str, int] = {}

    for payment in sources.payments:
        # --- bank leg: inherit whatever the payment's batches resolved to ---
        parts = [bank.get(sid) for sid in payment.settlement_ids]
        confident = [p for p in parts if p is not None and p.confidence >= threshold]
        txn_ids = [cid for p in confident for cid in p.counterpart_ids]
        if txn_ids:
            worst = min(confident, key=lambda p: p.confidence)
            evidence = [line for p in confident for line in p.evidence]
            if len(confident) < len(parts):
                evidence.append(
                    f"only {len(confident)} of {len(parts)} settlement batches for this "
                    "payment could be tied; the remainder is listed as an exception"
                )
            matches.append(
                Match.build(
                    link_type=LinkType.PAYMENT_TO_BANK,
                    payment_id=payment.payment_id,
                    counterpart_ids=txn_ids,
                    layer=worst.layer,
                    rule="+".join(sorted({p.rule for p in confident})),
                    confidence=worst.confidence,
                    evidence=evidence,
                    source_records={
                        "gateway": [payment.payment_id],
                        "settlement": payment.settlement_ids,
                        "bank": txn_ids,
                    },
                )
            )
            claimed_bank.update(txn_ids)
            for resolved_part in confident:
                layer_counts[resolved_part.rule] = layer_counts.get(resolved_part.rule, 0) + 1
        for settlement_id, part in zip(payment.settlement_ids, parts, strict=True):
            if part is None or part.confidence < threshold:
                exceptions.append(
                    _exception_from(
                        payment.payment_id,
                        LinkType.PAYMENT_TO_BANK,
                        part,
                        threshold,
                        f"settlement {settlement_id} could not be tied to a bank credit",
                    )
                )

        # --- invoice leg ---
        invoice_part = invoice.get(payment.payment_id)
        if invoice_part is not None and invoice_part.confidence >= threshold:
            matches.append(
                Match.build(
                    link_type=LinkType.PAYMENT_TO_INVOICE,
                    payment_id=payment.payment_id,
                    counterpart_ids=invoice_part.counterpart_ids,
                    layer=invoice_part.layer,
                    rule=invoice_part.rule,
                    confidence=invoice_part.confidence,
                    evidence=invoice_part.evidence,
                    source_records={
                        "gateway": [payment.payment_id],
                        "erp": invoice_part.counterpart_ids,
                    },
                )
            )
            claimed_invoices.update(invoice_part.counterpart_ids)
            layer_counts[invoice_part.rule] = layer_counts.get(invoice_part.rule, 0) + 1
        else:
            exceptions.append(
                _exception_from(
                    payment.payment_id,
                    LinkType.PAYMENT_TO_INVOICE,
                    invoice_part,
                    threshold,
                    "no invoice could be tied to this payment",
                )
            )

    for row in sources.bank_rows:
        if row.txn_id not in claimed_bank:
            exceptions.append(
                ReconException.build(
                    subject_type=SubjectType.BANK_TXN,
                    subject_id=row.txn_id,
                    link_type=LinkType.PAYMENT_TO_BANK,
                    reason=ExceptionReason.UNMATCHED_COUNTERPART,
                    detail="bank credit claimed by no settlement batch",
                    layer_reached=MatchLayer.L3_LLM if resolver else MatchLayer.L2_FUZZY,
                    evidence=[
                        f"credit {format_rupees(row.credit_paise)} on {row.value_date}",
                        f"narration: {row.narration}",
                    ],
                )
            )
    for inv in sources.invoices:
        if inv.invoice_id not in claimed_invoices:
            exceptions.append(
                ReconException.build(
                    subject_type=SubjectType.INVOICE,
                    subject_id=inv.invoice_id,
                    link_type=LinkType.PAYMENT_TO_INVOICE,
                    reason=ExceptionReason.UNMATCHED_COUNTERPART,
                    detail="invoice claimed by no payment",
                    layer_reached=MatchLayer.L3_LLM if resolver else MatchLayer.L2_FUZZY,
                    evidence=[
                        f"{inv.invoice_id} for {format_rupees(inv.invoice_amount_paise)}",
                        f"issued {inv.issue_date}, customer {inv.customer_name}",
                    ],
                )
            )

    return ReconResult(
        matcher="layered" + ("+llm" if resolver else " (L1+L2)"),
        split=split,
        matches=matches,
        exceptions=exceptions,
        stats={"payments": float(len(sources.payments)), "threshold": threshold}
        | {f"rule:{k}": float(v) for k, v in sorted(layer_counts.items())},
    )
