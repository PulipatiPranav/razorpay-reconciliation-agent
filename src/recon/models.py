"""Typed record schemas for the three sources and the ground-truth file.

Amounts live as integer paise on the models.  Rupee-decimal rendering happens
only in :mod:`recon.generator.writers` and parsing only in :mod:`recon.io`.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from recon.money import Paise


class DefectTag(StrEnum):
    """Every injected imperfection carries a tag, recorded in ground truth.

    Phase 4's per-mess-type breakdown is only possible because these are
    attached at injection time, so they are part of the data contract.
    """

    BUNDLED = "bundled_payout"
    TDS = "tds_deducted"
    WEEKEND_DRIFT = "weekend_holiday_drift"
    SPLIT_SETTLEMENT = "split_settlement"
    PAISE_DRIFT_ROW = "paise_drift_row"
    PAISE_DRIFT_BUNDLE = "paise_drift_bundle"
    REFUND_FULL = "refund_full"
    REFUND_PARTIAL = "refund_partial"
    DUPLICATE_REFUND = "duplicate_refund"
    NARRATION_CORRUPT = "narration_corrupt"
    ERP_LINK_BROKEN = "erp_link_broken"
    INVOICE_AMOUNT_MISMATCH = "invoice_amount_mismatch"
    CHARGEBACK_ADJUSTMENT = "chargeback_adjustment"
    # --- classes that resist mechanical inversion (see README, Phase 3) ---
    NARRATION_OPAQUE = "narration_opaque"
    UNEXPLAINED_DEDUCTION = "unexplained_deduction"
    DUPLICATE_CUSTOMER_INVOICE = "duplicate_customer_invoice"
    TZ_BOUNDARY = "tz_boundary"
    NO_INVOICE = "unresolvable_no_invoice"
    NO_BANK_CREDIT = "unresolvable_no_bank_credit"
    ORPHAN_ORDER_ID = "unresolvable_orphan_order_id"


class EntityType(StrEnum):
    PAYMENT = "payment"
    REFUND = "refund"
    ADJUSTMENT = "adjustment"


class PaymentMethod(StrEnum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class GatewayRow(BaseModel):
    """One line of a Razorpay-style settlement report."""

    model_config = ConfigDict(frozen=True)

    entity_type: EntityType
    entity_id: str
    payment_id: str | None
    order_id: str | None
    order_receipt: str
    method: PaymentMethod | None
    card_network: str | None
    amount_paise: Paise
    currency: str = "INR"
    fee_paise: Paise
    tax_paise: Paise
    tds_paise: Paise
    credit_paise: Paise
    debit_paise: Paise
    settlement_id: str | None
    settlement_utr: str | None
    created_at: datetime
    captured_at: datetime | None
    settled_at: datetime | None
    description: str


class BankRow(BaseModel):
    """One line of a bank statement -- bundled credits only, no order detail."""

    model_config = ConfigDict(frozen=True)

    txn_id: str
    value_date: date
    posted_date: date
    narration: str
    utr: str | None
    credit_paise: Paise
    debit_paise: Paise
    balance_paise: Paise
    ref_no: str


class InvoiceRow(BaseModel):
    """One line of the invoice / ERP ledger."""

    model_config = ConfigDict(frozen=True)

    invoice_id: str
    customer_id: str
    customer_name: str
    invoice_amount_paise: Paise
    tax_amount_paise: Paise
    currency: str = "INR"
    issue_date: date
    due_date: date
    order_id: str | None
    po_number: str
    status: str
    notes: str


class GroundTruthLink(BaseModel):
    """The correct answer for one payment.

    ``invoice_id`` / ``bank_txn_id`` are ``None`` when no counterpart exists.
    A matcher that proposes a link where ground truth has ``None`` scores a
    false positive; correctly routing it to exceptions scores a true negative.
    """

    payment_id: str
    order_id: str | None
    invoice_id: str | None
    settlement_ids: list[str]
    utrs: list[str]
    bank_txn_ids: list[str]
    gross_paise: Paise
    net_paise: Paise
    # Defects belonging to this payment's own record.
    defect_tags: list[DefectTag] = Field(default_factory=list)
    # Defects belonging to the settlement batch it rides in.  Kept separate
    # because a batch-level defect (a phantom duplicate refund, a chargeback,
    # a corrupted narration) breaks the batch tie-out for every payment in it.
    # Folding the two together made "22% of payments have a duplicate refund"
    # look like a payment-level prevalence, which it is not.
    bundle_defect_tags: list[DefectTag] = Field(default_factory=list)
    unresolvable_reason: str | None = None

    @property
    def resolvable(self) -> bool:
        return self.unresolvable_reason is None


class GroundTruthBundle(BaseModel):
    """The correct payment set behind one bank credit."""

    utr: str
    settlement_id: str
    bank_txn_id: str | None
    expected_credit_paise: Paise
    payment_ids: list[str]
    refund_entity_ids: list[str]
    adjustment_entity_ids: list[str]
    defect_tags: list[DefectTag] = Field(default_factory=list)


class Manifest(BaseModel):
    """Provenance for a generated split.

    ``config_hash`` is the SHA-256 of the serialized :class:`MessConfig`.  If
    the held-out set were ever regenerated with tuned parameters, this hash
    would change and the change would be visible in git history.  That is the
    auditable form of "I did not tune against the held-out set".
    """

    split: str
    seed: int
    generator_version: str
    config_hash: str
    generated_at: datetime
    n_payments: int
    n_bundles: int
    n_gateway_rows: int
    n_bank_rows: int
    n_invoice_rows: int
    # payment-level prevalence (denominator: payments)
    defect_counts: dict[str, int]
    # batch-level prevalence (denominator: bundles)
    bundle_defect_counts: dict[str, int]
    # payments riding a batch with each batch-level defect
    payments_affected_by_bundle_defect: dict[str, int]


class GroundTruth(BaseModel):
    manifest: Manifest
    bundles: list[GroundTruthBundle]
    links: list[GroundTruthLink]
    orphan_invoice_ids: list[str] = Field(default_factory=list)
    orphan_bank_txn_ids: list[str] = Field(default_factory=list)


class PaymentView(BaseModel):
    """A payment assembled from its gateway rows.

    A split settlement occupies two rows in the report (the capture and the
    deferred-balance release), so the payment-level view has to aggregate.
    Both the baseline and the layered matcher consume this same view, which is
    what keeps the comparison between them like-for-like: any difference in
    score comes from matching logic, not from one of them reading the file
    more cleverly than the other.
    """

    model_config = ConfigDict(frozen=True)

    payment_id: str
    order_id: str | None
    order_receipt: str
    method: PaymentMethod | None
    gross_paise: Paise
    fee_paise: Paise
    tax_paise: Paise
    tds_paise: Paise
    net_paise: Paise
    settlement_ids: list[str]
    settlement_utrs: list[str]
    captured_at: datetime | None
    settled_dates: list[date]
    row_count: int


class RefundView(BaseModel):
    """A refund or adjustment row, keyed to the batch it was deducted from."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    entity_type: EntityType
    payment_id: str | None
    amount_paise: Paise
    settlement_id: str | None
    settlement_utr: str | None
    created_at: datetime


class SettlementBatch(BaseModel):
    """One payout batch, reconstructed from the gateway report.

    The bank leg is a batch-level problem, not a payment-level one: a bank
    credit corresponds to a *batch*, and every payment in that batch inherits
    the answer.  Resolving batches and then propagating is both how a human
    does it and what makes many-to-one matching tractable.
    """

    model_config = ConfigDict(frozen=True)

    settlement_id: str
    utr: str | None
    payment_ids: list[str]
    settled_date: date | None
    # sum(credits) - sum(debits) across the batch's rows
    expected_credit_paise: Paise
    # the same, after collapsing refund rows that duplicate an identical
    # (payment_id, amount) pair.  A phantom duplicate appears twice in the
    # report while the bank moved the money once, so the naive figure is short
    # by exactly one refund; carrying both lets the matcher try each.
    expected_credit_dedup_paise: Paise
    refund_entity_ids: list[str]
    adjustment_entity_ids: list[str]
    has_duplicate_refund_rows: bool


class SourceBundle(BaseModel):
    """The three parsed sources for one split, as the matchers see them."""

    model_config = ConfigDict(frozen=True)

    payments: list[PaymentView]
    refunds: list[RefundView]
    bank_rows: list[BankRow]
    invoices: list[InvoiceRow]
    batches: list[SettlementBatch] = Field(default_factory=list)
