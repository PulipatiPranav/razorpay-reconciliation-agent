"""Layer 1: exact deterministic joins on the identifiers that are meant to join.

Pure functions.  Both take the already-claimed counterparts so that a credit or
an invoice can only ever be spent once -- a bank credit belongs to exactly one
batch, and letting two batches claim it would trade a recall point for a
precision loss on both.
"""

from __future__ import annotations

from collections import defaultdict

from recon.matcher.confidence import (
    L1_ORDER_ID_JOIN,
    L1_UTR_COLUMN_JOIN,
    confidence_for,
)
from recon.matcher.resolution import Resolution
from recon.matcher.types import MatchLayer
from recon.models import BankRow, InvoiceRow, PaymentView, SettlementBatch


def resolve_batches_exact(
    batches: list[SettlementBatch],
    bank_rows: list[BankRow],
    claimed: set[str] | None = None,
) -> dict[str, Resolution]:
    """Join each batch's settlement UTR to the bank statement's UTR column."""
    claimed = claimed or set()
    by_utr: dict[str, list[BankRow]] = defaultdict(list)
    for row in bank_rows:
        if row.utr and row.txn_id not in claimed:
            by_utr[row.utr].append(row)

    out: dict[str, Resolution] = {}
    for batch in batches:
        if batch.utr is None:
            continue
        hits = by_utr.get(batch.utr, [])
        if len(hits) != 1:
            continue
        row = hits[0]
        out[batch.settlement_id] = Resolution(
            subject_id=batch.settlement_id,
            counterpart_ids=[row.txn_id],
            layer=MatchLayer.L1_EXACT,
            rule=L1_UTR_COLUMN_JOIN,
            confidence=confidence_for(L1_UTR_COLUMN_JOIN),
            evidence=[
                f"settlement UTR {batch.utr} equals the statement's UTR column exactly",
                f"bank credit {row.txn_id} dated {row.value_date}",
                f"batch covers {len(batch.payment_ids)} payment(s)",
            ],
        )
    return out


def resolve_invoices_exact(
    payments: list[PaymentView],
    invoices: list[InvoiceRow],
    claimed: set[str] | None = None,
) -> dict[str, Resolution]:
    """Join the gateway's ``order_id`` to the ERP ledger's."""
    claimed = claimed or set()
    by_order: dict[str, list[InvoiceRow]] = defaultdict(list)
    for invoice in invoices:
        if invoice.order_id and invoice.invoice_id not in claimed:
            by_order[invoice.order_id].append(invoice)

    out: dict[str, Resolution] = {}
    for payment in payments:
        if payment.order_id is None:
            continue
        hits = by_order.get(payment.order_id, [])
        if len(hits) != 1:
            continue
        invoice = hits[0]
        out[payment.payment_id] = Resolution(
            subject_id=payment.payment_id,
            counterpart_ids=[invoice.invoice_id],
            layer=MatchLayer.L1_EXACT,
            rule=L1_ORDER_ID_JOIN,
            confidence=confidence_for(L1_ORDER_ID_JOIN),
            evidence=[
                f"order_id {payment.order_id} appears on exactly one invoice",
                f"invoice {invoice.invoice_id} for {invoice.customer_name}",
            ],
        )
    return out
