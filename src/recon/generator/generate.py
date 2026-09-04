"""Synthetic multi-source reconciliation universe.

The generator builds one internally-consistent universe of payments,
settlements, bank credits and invoices, injects each defect class
independently, records the exact answer for every payment, and then splits the
result into ``dev`` and ``holdout`` **at the settlement-bundle level**.

Why bundle-level and not payment-level: a bundle that straddled the split would
leak held-out structure into dev and would make many-to-one matching
untestable, because the dev side would hold a partial view of a held-out bank
credit.  The cost is that the 400/200 split is approximate -- the manifest
records the counts actually produced.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

from recon.calendar_in import add_business_days, is_bank_holiday
from recon.generator import ids, names
from recon.generator.config import FEE_PCT_BY_METHOD, METHOD_WEIGHTS, MessConfig
from recon.models import (
    BankRow,
    DefectTag,
    InvoiceRow,
    PaymentMethod,
)
from recon.money import Paise, net_of, pct_of

IST = timezone(timedelta(hours=5, minutes=30), "IST")
GENERATOR_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# internal working records (mutable; the pydantic models are the output form)
# --------------------------------------------------------------------------
@dataclass
class _Payment:
    payment_id: str
    true_order_id: str
    reported_order_id: str | None
    customer_idx: int
    method: PaymentMethod
    gross: Paise
    fee: Paise
    tax: Paise
    tds: Paise
    credit: Paise
    created_at: datetime
    captured_at: datetime
    receipt: str
    description: str
    invoice_id: str | None = None
    tags: set[DefectTag] = field(default_factory=set)
    unresolvable_reason: str | None = None
    # (bundle_id, paise) -- more than one entry means a split settlement
    portions: list[tuple[str, Paise]] = field(default_factory=list)


@dataclass
class _Refund:
    refund_id: str
    payment_id: str
    order_id: str | None
    amount: Paise
    created_at: datetime
    bundle_id: str
    is_phantom_duplicate: bool = False


@dataclass
class _Adjustment:
    adjustment_id: str
    bundle_id: str
    amount: Paise
    description: str
    created_at: datetime


@dataclass
class _Bundle:
    bundle_id: str  # doubles as settlement_id
    utr: str
    payment_ids: list[str]
    last_capture: date
    settled_date: date = date(2026, 1, 1)
    credit_total: Paise = 0
    bank_txn_id: str | None = None
    tags: set[DefectTag] = field(default_factory=set)
    has_bank_row: bool = True


# Defects that belong to a settlement batch rather than to one payment.
BUNDLE_LEVEL_TAGS: frozenset[DefectTag] = frozenset(
    {
        DefectTag.BUNDLED,
        DefectTag.WEEKEND_DRIFT,
        DefectTag.NARRATION_CORRUPT,
        DefectTag.PAISE_DRIFT_BUNDLE,
        DefectTag.CHARGEBACK_ADJUSTMENT,
        DefectTag.DUPLICATE_REFUND,
        DefectTag.SPLIT_SETTLEMENT,
        DefectTag.NO_BANK_CREDIT,
    }
)

# Of those, the ones a co-riding payment genuinely suffers from.
PROPAGATED_BUNDLE_TAGS: frozenset[DefectTag] = BUNDLE_LEVEL_TAGS - {
    DefectTag.NO_BANK_CREDIT,
    DefectTag.SPLIT_SETTLEMENT,
}


@dataclass
class Universe:
    """Everything the generator produced, before splitting."""

    config: MessConfig
    payments: list[_Payment]
    refunds: list[_Refund]
    adjustments: list[_Adjustment]
    bundles: list[_Bundle]
    invoices: list[InvoiceRow]
    bank_rows: dict[str, BankRow]  # bundle_id -> row
    noise_bank_rows: list[BankRow]
    orphan_invoice_ids: list[str]
    customers: list[tuple[str, str]]


def _weighted_method(rng: random.Random) -> PaymentMethod:
    methods = list(METHOD_WEIGHTS)
    weights = [METHOD_WEIGHTS[m] for m in methods]
    return rng.choices(methods, weights=weights, k=1)[0]


def _business_datetime(rng: random.Random, cfg: MessConfig) -> datetime:
    """A capture timestamp inside the window, weighted toward business days."""
    span = (cfg.window_end - cfg.window_start).days
    while True:
        day = cfg.window_start + timedelta(days=rng.randint(0, span))
        # payments still happen at weekends, just less often
        if is_bank_holiday(day) and rng.random() > 0.35:
            continue
        break
    hour = min(23, max(6, int(rng.gauss(14, 4))))
    return datetime(
        day.year, day.month, day.day, hour, rng.randint(0, 59), rng.randint(0, 59), tzinfo=IST
    )


# Repeated list prices.  Real books are full of identical amounts on the same
# day, which is precisely what defeats amount-only matching -- so the ambiguity
# is put in deliberately rather than left to chance.
PRICE_POINTS: list[Paise] = [49_900, 99_900, 1_49_900, 2_49_900, 4_99_900, 9_99_900, 19_99_900]


def _ticket_amount(rng: random.Random, cfg: MessConfig) -> Paise:
    """Ticket sizes: a log-normal body plus a spike of repeated price points."""
    if rng.random() < cfg.price_point_share:
        return rng.choice(PRICE_POINTS)
    lo, hi = cfg.min_ticket_paise, cfg.max_ticket_paise
    value = int(rng.lognormvariate(12.6, 1.0))
    value = max(lo, min(hi, value))
    if rng.random() < 0.75:
        value = round(value / 100) * 100  # whole rupees
    return value


def _transpose(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    i = rng.randint(1, len(text) - 2)
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


# --------------------------------------------------------------------------
# stage 1: payments
# --------------------------------------------------------------------------
def _build_customers(rng: random.Random, cfg: MessConfig) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i in range(cfg.n_customers):
        name = (
            f"{rng.choice(names.PREFIXES)} {rng.choice(names.MIDDLES)} "
            f"{rng.choice(names.SUFFIXES)}"
        )
        out.append((f"cust_{i:04d}", name))
    return out


def _build_payments(rng: random.Random, cfg: MessConfig) -> list[_Payment]:
    payments: list[_Payment] = []
    for _ in range(cfg.n_payments):
        method = _weighted_method(rng)
        gross = _ticket_amount(rng, cfg)
        captured = _business_datetime(rng, cfg)

        tags: set[DefectTag] = set()
        if rng.random() < cfg.rate_for(DefectTag.TZ_BOUNDARY):
            captured = captured.replace(hour=23, minute=rng.randint(5, 59))
            tags.add(DefectTag.TZ_BOUNDARY)

        # Rounding-mode drift: some rows are priced with banker's rounding.  The
        # row stays internally consistent, so the bank still ties -- but a
        # matcher that assumes HALF_UP mis-derives gross from net by a paisa.
        rounding = ROUND_HALF_UP
        if rng.random() < cfg.rate_for(DefectTag.PAISE_DRIFT_ROW):
            rounding = ROUND_HALF_EVEN
            tags.add(DefectTag.PAISE_DRIFT_ROW)

        fee_pct = FEE_PCT_BY_METHOD[method]
        fee = pct_of(gross, fee_pct, rounding=rounding)
        tax = pct_of(fee, cfg.gst_on_fee_pct, rounding=rounding)
        tds = 0
        if rng.random() < cfg.rate_for(DefectTag.TDS):
            tds = pct_of(gross, cfg.tds_pct, rounding=rounding)
            tags.add(DefectTag.TDS)

        order = ids.order_id(rng)
        payments.append(
            _Payment(
                payment_id=ids.payment_id(rng),
                true_order_id=order,
                reported_order_id=order,
                customer_idx=rng.randrange(cfg.n_customers),
                method=method,
                gross=gross,
                fee=fee,
                tax=tax,
                tds=tds,
                credit=net_of(gross, fee, tax, tds),
                created_at=captured - timedelta(seconds=rng.randint(20, 900)),
                captured_at=captured,
                receipt=ids.receipt_code(rng),
                description=rng.choice(names.DESCRIPTIONS),
                tags=tags,
            )
        )
    payments.sort(key=lambda p: p.captured_at)
    return payments


# --------------------------------------------------------------------------
# stage 2: bundling into settlements
# --------------------------------------------------------------------------
def _large_bundle_draw_prob(cfg: MessConfig) -> float:
    """Draw probability that yields the configured *payment* share in large bundles.

    ``large_bundle_payment_share`` is expressed as a share of payments, not a
    share of bundles -- large bundles hold far more payments each, so drawing
    "large" 60% of the time would put ~86% of payments in them.  Solve for the
    draw probability p such that p*E[large] / (p*E[large] + (1-p)*E[small])
    equals the configured share.
    """
    e_large = sum(cfg.large_bundle_size) / 2
    e_small = sum(cfg.small_bundle_size) / 2
    share = cfg.large_bundle_payment_share
    if share <= 0:
        return 0.0
    if share >= 1:
        return 1.0
    p = (share * e_small) / (e_large * (1 - share) + share * e_small)
    return min(1.0, max(0.0, p))


def _build_bundles(rng: random.Random, cfg: MessConfig, payments: list[_Payment]) -> list[_Bundle]:
    """Group payments into settlement batches by capture order.

    Bundle sizes are drawn from a two-component mixture: large bundles in the
    brief's 20-60 range carry ~60% of payments, the rest sit in small 4-15
    bundles.  Pure 20-60 would yield only ~15 bundles across 600 payments,
    leaving ~5 in the held-out set -- too few for a bundle-level metric to say
    anything.  The mixture roughly doubles the bundle count while keeping the
    large-bundle regime the brief asks for as the dominant case.
    """
    bundles: list[_Bundle] = []
    cursor = 0
    n = len(payments)
    draw_prob = _large_bundle_draw_prob(cfg)
    while cursor < n:
        if rng.random() < draw_prob:
            size = rng.randint(*cfg.large_bundle_size)
        else:
            size = rng.randint(*cfg.small_bundle_size)
        chunk = payments[cursor : cursor + size]
        if not chunk:
            break
        # avoid a runt tail bundle
        if n - (cursor + len(chunk)) < cfg.small_bundle_size[0]:
            chunk = payments[cursor:]
        bundle = _Bundle(
            bundle_id=ids.settlement_id(rng),
            utr=ids.utr(rng),
            payment_ids=[p.payment_id for p in chunk],
            last_capture=max(p.captured_at.date() for p in chunk),
        )
        bundle.settled_date = add_business_days(
            bundle.last_capture, cfg.settlement_lag_business_days
        )
        if (bundle.settled_date - bundle.last_capture).days > cfg.settlement_lag_business_days:
            bundle.tags.add(DefectTag.WEEKEND_DRIFT)
        bundle.tags.add(DefectTag.BUNDLED)
        bundles.append(bundle)
        cursor += len(chunk)
    return bundles


def _apply_splits(
    rng: random.Random,
    cfg: MessConfig,
    payments: dict[str, _Payment],
    bundles: list[_Bundle],
) -> None:
    """Assign each payment's credit to bundles, splitting a small fraction."""
    index_of = {b.bundle_id: i for i, b in enumerate(bundles)}
    for bundle in bundles:
        for pid in list(bundle.payment_ids):
            pay = payments[pid]
            if pay.portions:  # already placed (it was the tail of a split)
                continue
            rate = cfg.rate_for(DefectTag.SPLIT_SETTLEMENT)
            later_idx = index_of[bundle.bundle_id] + rng.randint(1, 2)
            if rng.random() < rate and later_idx < len(bundles):
                first = (pay.credit * 6) // 10
                second = pay.credit - first
                later = bundles[later_idx]
                pay.portions = [(bundle.bundle_id, first), (later.bundle_id, second)]
                pay.tags.add(DefectTag.SPLIT_SETTLEMENT)
                later.payment_ids.append(pid)
                later.tags.add(DefectTag.SPLIT_SETTLEMENT)
                bundle.tags.add(DefectTag.SPLIT_SETTLEMENT)
            else:
                pay.portions = [(bundle.bundle_id, pay.credit)]


# --------------------------------------------------------------------------
# stage 3: refunds, chargebacks
# --------------------------------------------------------------------------
def _build_refunds(
    rng: random.Random,
    cfg: MessConfig,
    payments: list[_Payment],
    bundles: list[_Bundle],
) -> list[_Refund]:
    refunds: list[_Refund] = []
    bundle_of_payment = {pid: b for b in bundles for pid in b.payment_ids}
    index_of = {b.bundle_id: i for i, b in enumerate(bundles)}
    for pay in payments:
        if rng.random() >= cfg.rate_for(DefectTag.REFUND_FULL):
            continue
        partial = rng.random() < cfg.refund_partial_share
        if partial:
            amount = max(100, (pay.gross * rng.randint(20, 70)) // 100)
            pay.tags.add(DefectTag.REFUND_PARTIAL)
        else:
            amount = pay.gross
            pay.tags.add(DefectTag.REFUND_FULL)

        home = bundle_of_payment[pay.payment_id]
        # a refund raised after the settlement cycle lands on a later payout
        idx = index_of[home.bundle_id] + (0 if rng.random() < 0.45 else rng.randint(1, 2))
        target = bundles[min(idx, len(bundles) - 1)]
        refunds.append(
            _Refund(
                refund_id=ids.refund_id(rng),
                payment_id=pay.payment_id,
                order_id=pay.reported_order_id,
                amount=amount,
                created_at=pay.captured_at
                + timedelta(days=rng.randint(1, 6), minutes=rng.randint(-300, 300)),
                bundle_id=target.bundle_id,
            )
        )
        # A phantom duplicate: the report shows the refund twice, the bank moved
        # the money once.  Naive summation double-counts and the bundle no
        # longer ties -- this is the classic double-count trap.
        duplicate_odds = cfg.rate_for(DefectTag.DUPLICATE_REFUND) / max(
            cfg.rate_for(DefectTag.REFUND_FULL), 1e-9
        )
        if rng.random() < duplicate_odds:
            refunds.append(
                _Refund(
                    refund_id=ids.refund_id(rng),
                    payment_id=pay.payment_id,
                    order_id=pay.reported_order_id,
                    amount=amount,
                    created_at=refunds[-1].created_at + timedelta(minutes=rng.randint(2, 90)),
                    bundle_id=target.bundle_id,
                    is_phantom_duplicate=True,
                )
            )
            pay.tags.add(DefectTag.DUPLICATE_REFUND)
            target.tags.add(DefectTag.DUPLICATE_REFUND)
    return refunds


def _build_adjustments(
    rng: random.Random, cfg: MessConfig, bundles: list[_Bundle]
) -> list[_Adjustment]:
    out: list[_Adjustment] = []
    for bundle in bundles:
        if rng.random() >= cfg.rate_for(DefectTag.CHARGEBACK_ADJUSTMENT):
            continue
        amount = rng.randint(50_000, 4_00_000)
        out.append(
            _Adjustment(
                adjustment_id=ids.adjustment_id(rng),
                bundle_id=bundle.bundle_id,
                amount=amount,
                description=rng.choice(
                    ["Chargeback debit", "Dispute hold", "Platform fee adjustment"]
                ),
                created_at=datetime.combine(bundle.settled_date, datetime.min.time(), tzinfo=IST),
            )
        )
        bundle.tags.add(DefectTag.CHARGEBACK_ADJUSTMENT)
    return out


# --------------------------------------------------------------------------
# stage 4: bank statement
# --------------------------------------------------------------------------
NARRATION_TEMPLATES = [
    "NEFT-{utr}-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT",
    "IMPS/{utr}/RAZORPAYSOFT/PAYOUT",
    "NEFT CR-{utr}-RAZORPAY SOFTWARE PRIVATE LIMITED-MERCHANT SETTLEMENT",
    "RTGS-{utr}-RZPY SETTLEMENT-BATCH",
]

NOISE_NARRATIONS = [
    "NEFT-{utr}-{name}-VENDOR REFUND",
    "UPI/{utr}/{name}/DIRECT COLLECTION",
    "INT.PD:{utr} SAVINGS INTEREST CREDIT",
    "NEFT-{utr}-{name}-ADVANCE",
]


def _compute_bundle_credits(
    cfg: MessConfig,
    rng: random.Random,
    payments: dict[str, _Payment],
    bundles: list[_Bundle],
    refunds: list[_Refund],
    adjustments: list[_Adjustment],
) -> None:
    """Money the bank actually moved for each settlement batch."""
    for bundle in bundles:
        total = 0
        for pid in bundle.payment_ids:
            for bid, portion in payments[pid].portions:
                if bid == bundle.bundle_id:
                    total += portion
        for refund in refunds:
            if refund.bundle_id == bundle.bundle_id and not refund.is_phantom_duplicate:
                total -= refund.amount
        for adj in adjustments:
            if adj.bundle_id == bundle.bundle_id:
                total -= adj.amount
        # Bank-side truncation: the credit lands a few paise off the sum of the
        # report rows.  Forces the matcher to carry an explicit tolerance
        # instead of relying on exact equality.
        if rng.random() < cfg.rate_for(DefectTag.PAISE_DRIFT_BUNDLE):
            total += rng.choice([-5, -3, -2, -1, 1, 2, 3, 5])
            bundle.tags.add(DefectTag.PAISE_DRIFT_BUNDLE)
        bundle.credit_total = total


def _build_bank_rows(
    rng: random.Random, cfg: MessConfig, bundles: list[_Bundle]
) -> dict[str, BankRow]:
    rows: dict[str, BankRow] = {}
    for bundle in bundles:
        if not bundle.has_bank_row:
            continue
        txn_id = ids.bank_txn_id(rng)
        bundle.bank_txn_id = txn_id
        utr_in_narration = bundle.utr
        utr_column: str | None = bundle.utr

        if rng.random() < cfg.rate_for(DefectTag.NARRATION_CORRUPT):
            mode = rng.choice(["column_null", "truncated", "conflicting"])
            if mode == "column_null":
                # UTR recoverable only by parsing the narration string
                utr_column = None
            elif mode == "truncated":
                utr_column = None
                utr_in_narration = bundle.utr[: rng.randint(9, 13)]
            else:
                # structured column disagrees with the narration
                utr_column = _transpose(bundle.utr, rng)
            bundle.tags.add(DefectTag.NARRATION_CORRUPT)

        posted = bundle.settled_date
        if rng.random() < 0.12:
            posted = bundle.settled_date + timedelta(days=1)

        rows[bundle.bundle_id] = BankRow(
            txn_id=txn_id,
            value_date=bundle.settled_date,
            posted_date=posted,
            narration=rng.choice(NARRATION_TEMPLATES).format(utr=utr_in_narration),
            utr=utr_column,
            credit_paise=max(0, bundle.credit_total),
            debit_paise=0 if bundle.credit_total >= 0 else -bundle.credit_total,
            balance_paise=0,  # filled in once rows are ordered
            ref_no=ids.bank_ref_no(rng),
        )
    return rows


def _build_noise_bank_rows(
    rng: random.Random, cfg: MessConfig, customers: list[tuple[str, str]]
) -> list[BankRow]:
    """Credits that are not settlements at all and must be rejected, not matched."""
    out: list[BankRow] = []
    span = (cfg.window_end - cfg.window_start).days + 6
    for _ in range(cfg.n_noise_bank_credits):
        day = cfg.window_start + timedelta(days=rng.randint(2, span))
        name = rng.choice(customers)[1].upper()[:24]
        fake_utr = ids.utr(rng)
        out.append(
            BankRow(
                txn_id=ids.bank_txn_id(rng),
                value_date=day,
                posted_date=day,
                narration=rng.choice(NOISE_NARRATIONS).format(utr=fake_utr, name=name),
                utr=fake_utr if rng.random() < 0.6 else None,
                credit_paise=rng.randint(20_000, 9_00_000),
                debit_paise=0,
                balance_paise=0,
                ref_no=ids.bank_ref_no(rng),
            )
        )
    return out


# --------------------------------------------------------------------------
# stage 5: ERP invoices
# --------------------------------------------------------------------------
def _build_invoices(
    rng: random.Random,
    cfg: MessConfig,
    payments: list[_Payment],
    customers: list[tuple[str, str]],
) -> tuple[list[InvoiceRow], list[str]]:
    invoices: list[InvoiceRow] = []
    seq = 1
    for pay in payments:
        if pay.invoice_id is None and DefectTag.NO_INVOICE in pay.tags:
            continue
        if DefectTag.ORPHAN_ORDER_ID in pay.tags:
            continue
        invoice_id = ids.invoice_id(rng, seq)
        seq += 1
        pay.invoice_id = invoice_id

        amount = pay.gross
        if rng.random() < cfg.rate_for(DefectTag.INVOICE_AMOUNT_MISMATCH):
            # credit note or short payment: ERP and gateway legitimately differ
            delta = max(100, (pay.gross * rng.randint(2, 12)) // 100)
            amount = pay.gross + (delta if rng.random() < 0.4 else -delta)
            pay.tags.add(DefectTag.INVOICE_AMOUNT_MISMATCH)

        reported_order: str | None = pay.true_order_id
        if rng.random() < cfg.rate_for(DefectTag.ERP_LINK_BROKEN):
            mode = rng.choice(["null", "typo", "case"])
            if mode == "null":
                reported_order = None
            elif mode == "typo":
                reported_order = _transpose(pay.true_order_id, rng)
            else:
                reported_order = pay.true_order_id.upper()
            pay.tags.add(DefectTag.ERP_LINK_BROKEN)

        cust_id, cust_name = customers[pay.customer_idx]
        issue = pay.captured_at.date() - timedelta(days=rng.randint(0, 12))
        invoices.append(
            InvoiceRow(
                invoice_id=invoice_id,
                customer_id=cust_id,
                customer_name=cust_name,
                invoice_amount_paise=amount,
                # gross is GST-inclusive; the ERP records the tax component
                tax_amount_paise=amount - int(round(amount / Decimal("1.18"))),
                issue_date=issue,
                due_date=issue + timedelta(days=rng.choice([15, 30, 30, 45, 60])),
                order_id=reported_order,
                po_number=ids.po_number(rng),
                status="paid",
                notes="",
            )
        )
        # 35% of gateway receipts carry the invoice number -- a text signal the
        # LLM layer can exploit when the structured order_id link is broken.
        if rng.random() < 0.35:
            pay.receipt = invoice_id

    orphan_ids: list[str] = []
    for _ in range(cfg.n_orphan_invoices):
        invoice_id = ids.invoice_id(rng, seq)
        seq += 1
        orphan_ids.append(invoice_id)
        cust_id, cust_name = customers[rng.randrange(len(customers))]
        issue = cfg.window_start + timedelta(days=rng.randint(0, 60))
        amount = _ticket_amount(rng, cfg)
        invoices.append(
            InvoiceRow(
                invoice_id=invoice_id,
                customer_id=cust_id,
                customer_name=cust_name,
                invoice_amount_paise=amount,
                tax_amount_paise=amount - int(round(amount / Decimal("1.18"))),
                issue_date=issue,
                due_date=issue + timedelta(days=30),
                order_id=ids.order_id(rng),  # points at an order that never existed
                po_number=ids.po_number(rng),
                status="open",
                notes="Awaiting payment",
            )
        )
    return invoices, orphan_ids


# --------------------------------------------------------------------------
# stage 6: genuinely unresolvable records
# --------------------------------------------------------------------------
def _apply_unresolvable(
    rng: random.Random,
    cfg: MessConfig,
    payments: list[_Payment],
    bundles: list[_Bundle],
) -> None:
    """Inject records that *should* land in exceptions.

    These are not generator bugs.  A reconciliation agent that resolves them
    anyway is hallucinating, and the evaluation treats a confident match here
    as a false positive.
    """
    pool = list(payments)
    rng.shuffle(pool)
    cursor = 0

    n_no_invoice = int(round(cfg.rate_for(DefectTag.NO_INVOICE) * len(payments)))
    for pay in pool[cursor : cursor + n_no_invoice]:
        pay.tags.add(DefectTag.NO_INVOICE)
        pay.unresolvable_reason = "no_erp_counterpart"
    cursor += n_no_invoice

    n_orphan = int(round(cfg.rate_for(DefectTag.ORPHAN_ORDER_ID) * len(payments)))
    for pay in pool[cursor : cursor + n_orphan]:
        pay.tags.add(DefectTag.ORPHAN_ORDER_ID)
        pay.reported_order_id = None
        pay.receipt = ids.receipt_code(rng)  # no invoice hint either
        pay.unresolvable_reason = "no_order_id_and_no_erp_counterpart"
    cursor += n_orphan

    # A missing bank credit is a batch-level event, so it is applied by
    # dropping whole (small) settlement batches until the target share of
    # payments is covered.  Doing it per payment would be physically incoherent.
    target = int(round(cfg.rate_for(DefectTag.NO_BANK_CREDIT) * len(payments)))
    if target:
        candidates = sorted(bundles, key=lambda b: len(b.payment_ids))
        covered = 0
        by_id = {p.payment_id: p for p in payments}
        for bundle in candidates:
            if covered >= target:
                break
            if DefectTag.SPLIT_SETTLEMENT in bundle.tags:
                continue  # keep split payments coherent
            bundle.has_bank_row = False
            bundle.tags.add(DefectTag.NO_BANK_CREDIT)
            for pid in bundle.payment_ids:
                pay = by_id[pid]
                pay.tags.add(DefectTag.NO_BANK_CREDIT)
                if pay.unresolvable_reason is None:
                    pay.unresolvable_reason = "settlement_never_credited"
            covered += len(bundle.payment_ids)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def generate_universe(cfg: MessConfig) -> Universe:
    """Build one complete, internally consistent universe.  Pure in ``cfg``."""
    rng = random.Random(cfg.seed)
    customers = _build_customers(rng, cfg)
    payments = _build_payments(rng, cfg)
    bundles = _build_bundles(rng, cfg, payments)
    _apply_unresolvable(rng, cfg, payments, bundles)
    by_id = {p.payment_id: p for p in payments}
    _apply_splits(rng, cfg, by_id, bundles)
    refunds = _build_refunds(rng, cfg, payments, bundles)
    adjustments = _build_adjustments(rng, cfg, bundles)
    _compute_bundle_credits(cfg, rng, by_id, bundles, refunds, adjustments)
    bank_rows = _build_bank_rows(rng, cfg, bundles)
    noise_rows = _build_noise_bank_rows(rng, cfg, customers)
    invoices, orphan_invoice_ids = _build_invoices(rng, cfg, payments, customers)
    return Universe(
        config=cfg,
        payments=payments,
        refunds=refunds,
        adjustments=adjustments,
        bundles=bundles,
        invoices=invoices,
        bank_rows=bank_rows,
        noise_bank_rows=noise_rows,
        orphan_invoice_ids=orphan_invoice_ids,
        customers=customers,
    )


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def _bundle_components(universe: Universe) -> list[list[str]]:
    """Group bundles that share a split payment; they must not be separated."""
    parent: dict[str, str] = {b.bundle_id: b.bundle_id for b in universe.bundles}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pay in universe.payments:
        bids = [bid for bid, _ in pay.portions]
        for other in bids[1:]:
            union(bids[0], other)
    groups: dict[str, list[str]] = {}
    for bid in parent:
        groups.setdefault(find(bid), []).append(bid)
    return list(groups.values())


def split_universe(universe: Universe) -> dict[str, set[str]]:
    """Assign bundle components to ``dev`` / ``holdout``, stratified by defect mix.

    Balancing payment count alone produced a held-out set with 28% of payments
    riding a chargeback batch against dev's 9%, and zero corrupted narrations
    in dev at all.  A held-out score under a materially different defect mix is
    not a fair comparison, so the split balances every batch-level defect class
    alongside the payment count, assigning each component to whichever split it
    leaves least over-quota on its worst dimension.

    Components (bundles joined by a split settlement) are never broken up, so
    the realised split is close to -- not exactly -- 400/200.
    """
    cfg = universe.config
    rng = random.Random(cfg.seed + 9_973)
    by_id = {b.bundle_id: b for b in universe.bundles}
    sizes = {b.bundle_id: len(set(b.payment_ids)) for b in universe.bundles}
    components = _bundle_components(universe)
    rng.shuffle(components)

    def tag_count(tag: DefectTag) -> int:
        return sum(1 for b in universe.bundles if tag in b.tags)

    # Only stratify on defect classes that carry information: a tag on almost
    # every bundle (BUNDLED, and SPLIT_SETTLEMENT, which chains through most
    # batches) discriminates nothing, and a tag on fewer than six bundles
    # cannot be balanced without wrecking the payment split.
    n_bundles = len(universe.bundles)
    strat_tags = [
        t
        for t in sorted(BUNDLE_LEVEL_TAGS, key=str)
        if 6 <= tag_count(t) <= int(0.75 * n_bundles)
    ]
    dims = ["payments"] + [str(t) for t in strat_tags]

    def weight_vector(comp: list[str]) -> dict[str, float]:
        vec = {"payments": float(sum(sizes[bid] for bid in comp))}
        for tag in strat_tags:
            vec[str(tag)] = float(sum(1 for bid in comp if tag in by_id[bid].tags))
        return vec

    vectors = {id(comp): weight_vector(comp) for comp in components}
    totals = {d: sum(vectors[id(c)][d] for c in components) for d in dims}

    share = {
        "dev": cfg.n_dev_payments / cfg.n_payments,
        "holdout": 1.0 - cfg.n_dev_payments / cfg.n_payments,
    }
    quota = {k: {d: totals[d] * share[k] for d in dims} for k in share}
    filled = {k: dict.fromkeys(dims, 0.0) for k in share}
    assigned: dict[str, set[str]] = {k: set() for k in share}

    # Squared over-quota deviation, with the payment count weighted heavily.
    # A max-over-dimensions rule let one rare defect class saturate a split and
    # pushed the realised split to 500/100; this keeps payments dominant and
    # uses the defect dimensions to break ties.
    payment_weight = 8.0

    def cost(split: str, vec: dict[str, float]) -> float:
        total = 0.0
        for d in dims:
            if quota[split][d] <= 0:
                continue
            ratio = (filled[split][d] + vec[d]) / quota[split][d]
            total += (payment_weight if d == "payments" else 1.0) * ratio * ratio
        return total

    for comp in sorted(components, key=lambda c: -vectors[id(c)]["payments"]):
        vec = vectors[id(comp)]
        target = min(share, key=lambda k: (cost(k, vec), k))
        assigned[target].update(comp)
        for d in dims:
            filled[target][d] += vec[d]
    return assigned
