"""Small hand-built fixtures for matcher and scorer unit tests.

Deliberately tiny and explicit: the matching layers are pure functions, so they
are tested against records constructed in the test rather than against the
generated corpus.  A test that depends on 600 generated records tells you a
score changed, not which rule broke.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from recon.models import (
    BankRow,
    GroundTruth,
    GroundTruthBundle,
    GroundTruthLink,
    InvoiceRow,
    Manifest,
    PaymentMethod,
    PaymentView,
    SourceBundle,
)

IST = timezone(timedelta(hours=5, minutes=30))


def make_payment(
    payment_id: str = "pay_1",
    *,
    order_id: str | None = "order_1",
    gross: int = 100_000,
    net: int = 97_000,
    utrs: list[str] | None = None,
    settled: list[date] | None = None,
    captured: date = date(2026, 2, 3),
    rows: int = 1,
) -> PaymentView:
    return PaymentView(
        payment_id=payment_id,
        order_id=order_id,
        order_receipt="rcpt_x",
        method=PaymentMethod.UPI,
        gross_paise=gross,
        fee_paise=2_000,
        tax_paise=360,
        tds_paise=640,
        net_paise=net,
        settlement_ids=["setl_1"],
        settlement_utrs=utrs if utrs is not None else ["HDFC000000000001"],
        captured_at=datetime(captured.year, captured.month, captured.day, 12, 0, tzinfo=IST),
        settled_dates=settled if settled is not None else [date(2026, 2, 5)],
        row_count=rows,
    )


def make_bank(
    txn_id: str = "bank_1",
    *,
    credit: int = 97_000,
    value_date: date = date(2026, 2, 5),
    utr: str | None = "HDFC000000000001",
) -> BankRow:
    return BankRow(
        txn_id=txn_id,
        value_date=value_date,
        posted_date=value_date,
        narration=f"NEFT-{utr or 'UNKNOWN'}-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT",
        utr=utr,
        credit_paise=credit,
        debit_paise=0,
        balance_paise=10_00_000,
        ref_no="000000001",
    )


def make_invoice(
    invoice_id: str = "INV-2026-00001",
    *,
    amount: int = 100_000,
    issue: date = date(2026, 2, 3),
    order_id: str | None = "order_1",
) -> InvoiceRow:
    return InvoiceRow(
        invoice_id=invoice_id,
        customer_id="cust_0001",
        customer_name="Anvaya Retail Pvt Ltd",
        invoice_amount_paise=amount,
        tax_amount_paise=amount - int(round(amount / 1.18)),
        issue_date=issue,
        due_date=issue + timedelta(days=30),
        order_id=order_id,
        po_number="PO/1000/26",
        status="paid",
        notes="",
    )


def make_sources(
    payments: list[PaymentView] | None = None,
    bank: list[BankRow] | None = None,
    invoices: list[InvoiceRow] | None = None,
) -> SourceBundle:
    return SourceBundle(
        payments=payments if payments is not None else [make_payment()],
        refunds=[],
        bank_rows=bank if bank is not None else [make_bank()],
        invoices=invoices if invoices is not None else [make_invoice()],
    )


def make_truth(links: list[GroundTruthLink], *, orphan_bank=(), orphan_invoices=()) -> GroundTruth:
    manifest = Manifest(
        split="test",
        seed=0,
        generator_version="test",
        config_hash="test",
        generated_at=datetime(2026, 1, 1, tzinfo=IST),
        n_payments=len(links),
        n_bundles=1,
        n_gateway_rows=len(links),
        n_bank_rows=1,
        n_invoice_rows=1,
        defect_counts={},
        bundle_defect_counts={},
        payments_affected_by_bundle_defect={},
    )
    return GroundTruth(
        manifest=manifest,
        bundles=[
            GroundTruthBundle(
                utr="HDFC000000000001",
                settlement_id="setl_1",
                bank_txn_id="bank_1",
                expected_credit_paise=97_000,
                payment_ids=[link.payment_id for link in links],
                refund_entity_ids=[],
                adjustment_entity_ids=[],
            )
        ],
        links=links,
        orphan_invoice_ids=list(orphan_invoices),
        orphan_bank_txn_ids=list(orphan_bank),
    )


def make_link(
    payment_id: str = "pay_1",
    *,
    invoice_id: str | None = "INV-2026-00001",
    bank_txn_ids: list[str] | None = None,
    unresolvable: str | None = None,
) -> GroundTruthLink:
    return GroundTruthLink(
        payment_id=payment_id,
        order_id="order_1",
        invoice_id=invoice_id,
        settlement_ids=["setl_1"],
        utrs=["HDFC000000000001"],
        bank_txn_ids=bank_txn_ids if bank_txn_ids is not None else ["bank_1"],
        gross_paise=100_000,
        net_paise=97_000,
        unresolvable_reason=unresolvable,
    )


@pytest.fixture
def sources() -> SourceBundle:
    return make_sources()
