"""Parse the three CSV sources into typed records.

This is the only module that reads the source files.  Everything downstream
takes parsed records as arguments, which is what keeps the matching layers pure
and what makes the held-out guard enforceable.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from recon.models import (
    BankRow,
    EntityType,
    GatewayRow,
    InvoiceRow,
    PaymentMethod,
    PaymentView,
    RefundView,
    SourceBundle,
)
from recon.money import Paise, parse_rupees


def _opt(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _money(value: str) -> Paise:
    """Blank money columns mean zero here; the CSV writer always emits 0.00."""
    stripped = value.strip()
    return parse_rupees(stripped) if stripped else 0


def _ts(value: str) -> datetime | None:
    stripped = value.strip()
    return datetime.fromisoformat(stripped) if stripped else None


def _require_ts(value: str, *, row: str) -> datetime:
    """``created_at`` is mandatory on every gateway row; a blank is corrupt input."""
    parsed = _ts(value)
    if parsed is None:
        raise ValueError(f"gateway row {row} has no created_at")
    return parsed


def read_gateway(path: Path) -> list[GatewayRow]:
    rows: list[GatewayRow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            method = _opt(raw["method"])
            rows.append(
                GatewayRow(
                    entity_type=EntityType(raw["entity_type"]),
                    entity_id=raw["entity_id"],
                    payment_id=_opt(raw["payment_id"]),
                    order_id=_opt(raw["order_id"]),
                    order_receipt=raw["order_receipt"],
                    method=PaymentMethod(method) if method else None,
                    card_network=_opt(raw["card_network"]),
                    amount_paise=_money(raw["amount"]),
                    currency=raw["currency"],
                    fee_paise=_money(raw["fee"]),
                    tax_paise=_money(raw["tax"]),
                    tds_paise=_money(raw["tds"]),
                    credit_paise=_money(raw["credit"]),
                    debit_paise=_money(raw["debit"]),
                    settlement_id=_opt(raw["settlement_id"]),
                    settlement_utr=_opt(raw["settlement_utr"]),
                    created_at=_require_ts(raw["created_at"], row=raw["entity_id"]),
                    captured_at=_ts(raw["captured_at"]),
                    settled_at=_ts(raw["settled_at"]),
                    description=raw["description"],
                )
            )
    return rows


def read_bank(path: Path) -> list[BankRow]:
    rows: list[BankRow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                BankRow(
                    txn_id=raw["txn_id"],
                    value_date=date.fromisoformat(raw["value_date"]),
                    posted_date=date.fromisoformat(raw["posted_date"]),
                    narration=raw["narration"],
                    utr=_opt(raw["utr"]),
                    credit_paise=_money(raw["credit_amount"]),
                    debit_paise=_money(raw["debit_amount"]),
                    balance_paise=_money(raw["balance"]),
                    ref_no=raw["ref_no"],
                )
            )
    return rows


def read_invoices(path: Path) -> list[InvoiceRow]:
    rows: list[InvoiceRow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                InvoiceRow(
                    invoice_id=raw["invoice_id"],
                    customer_id=raw["customer_id"],
                    customer_name=raw["customer_name"],
                    invoice_amount_paise=_money(raw["invoice_amount"]),
                    tax_amount_paise=_money(raw["tax_amount"]),
                    currency=raw["currency"],
                    issue_date=date.fromisoformat(raw["issue_date"]),
                    due_date=date.fromisoformat(raw["due_date"]),
                    order_id=_opt(raw["order_id"]),
                    po_number=raw["po_number"],
                    status=raw["status"],
                    notes=raw["notes"],
                )
            )
    return rows


def build_payment_views(gateway: list[GatewayRow]) -> list[PaymentView]:
    """Collapse gateway payment rows into one view per payment.

    Fee, tax and TDS are taken from the capture row only -- the deferred
    balance row of a split settlement carries no fee of its own, so summing
    them would be wrong.  ``net_paise`` sums every credit, so it is the total
    the merchant was actually paid across all batches.
    """
    grouped: dict[str, list[GatewayRow]] = defaultdict(list)
    for row in gateway:
        if row.entity_type is not EntityType.PAYMENT or row.payment_id is None:
            continue
        grouped[row.payment_id].append(row)

    views: list[PaymentView] = []
    for payment_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda r: (r.settled_at or r.created_at))
        head = max(ordered, key=lambda r: r.amount_paise)  # the capture row
        views.append(
            PaymentView(
                payment_id=payment_id,
                order_id=head.order_id,
                order_receipt=head.order_receipt,
                method=head.method,
                gross_paise=head.amount_paise,
                fee_paise=head.fee_paise,
                tax_paise=head.tax_paise,
                tds_paise=head.tds_paise,
                net_paise=sum(r.credit_paise for r in ordered),
                settlement_ids=[r.settlement_id for r in ordered if r.settlement_id],
                settlement_utrs=[r.settlement_utr for r in ordered if r.settlement_utr],
                captured_at=head.captured_at,
                settled_dates=[r.settled_at.date() for r in ordered if r.settled_at],
                row_count=len(ordered),
            )
        )
    views.sort(key=lambda v: v.payment_id)
    return views


def build_refund_views(gateway: list[GatewayRow]) -> list[RefundView]:
    return sorted(
        (
            RefundView(
                entity_id=row.entity_id,
                entity_type=row.entity_type,
                payment_id=row.payment_id,
                amount_paise=row.debit_paise,
                settlement_id=row.settlement_id,
                settlement_utr=row.settlement_utr,
                created_at=row.created_at,
            )
            for row in gateway
            if row.entity_type in (EntityType.REFUND, EntityType.ADJUSTMENT)
        ),
        key=lambda r: r.entity_id,
    )


def load_split(directory: Path) -> SourceBundle:
    """Read one split directory into the parsed form the matchers consume."""
    gateway = read_gateway(directory / "gateway_settlements.csv")
    return SourceBundle(
        payments=build_payment_views(gateway),
        refunds=build_refund_views(gateway),
        bank_rows=read_bank(directory / "bank_statement.csv"),
        invoices=read_invoices(directory / "erp_invoices.csv"),
    )
