"""Generator configuration: volumes, rates, and per-defect toggles.

Every defect class is (a) independently toggleable via ``disabled`` and (b)
independently proportioned via its own ``rate_*`` field.  Nothing in the
generator reads a rate directly -- everything goes through :meth:`rate_for`,
so disabling a defect is guaranteed to zero it out everywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal

from recon.models import DefectTag, PaymentMethod

# Gateway fee by method, in percent of gross.  Loosely modelled on published
# Indian PG pricing: UPI cheapest, cards dearest.
FEE_PCT_BY_METHOD: dict[PaymentMethod, Decimal] = {
    PaymentMethod.UPI: Decimal("0.90"),
    PaymentMethod.CARD: Decimal("2.36"),
    PaymentMethod.NETBANKING: Decimal("1.80"),
    PaymentMethod.WALLET: Decimal("2.00"),
}

METHOD_WEIGHTS: dict[PaymentMethod, float] = {
    PaymentMethod.UPI: 0.55,
    PaymentMethod.CARD: 0.25,
    PaymentMethod.NETBANKING: 0.14,
    PaymentMethod.WALLET: 0.06,
}


@dataclass(frozen=True)
class MessConfig:
    """Volumes, economics and defect proportions for one generated universe."""

    # --- volumes ------------------------------------------------------------
    n_payments: int = 600
    n_dev_payments: int = 400  # target; the bundle-level split lands near it
    n_customers: int = 120
    window_start: date = date(2026, 1, 5)
    window_end: date = date(2026, 3, 27)
    seed: int = 42

    # --- economics ----------------------------------------------------------
    gst_on_fee_pct: Decimal = Decimal("18")
    tds_pct: Decimal = Decimal("1")
    settlement_lag_business_days: int = 2
    min_ticket_paise: int = 45_000          # Rs 450
    max_ticket_paise: int = 2_50_00_000     # Rs 2,50,000
    price_point_share: float = 0.20         # share drawn from repeated list prices

    # --- bundle shape -------------------------------------------------------
    large_bundle_payment_share: float = 0.70
    large_bundle_size: tuple[int, int] = (20, 60)
    small_bundle_size: tuple[int, int] = (4, 15)

    # --- defect proportions (fraction of the eligible population) -----------
    rate_tds: float = 0.15
    rate_split_settlement: float = 0.06
    rate_paise_drift_row: float = 0.12
    rate_paise_drift_bundle: float = 0.20   # per bundle
    rate_refund: float = 0.08
    refund_partial_share: float = 0.40
    rate_duplicate_refund: float = 0.015
    rate_narration_corrupt: float = 0.25   # per bank credit
    rate_erp_link_broken: float = 0.08
    rate_invoice_amount_mismatch: float = 0.04
    rate_chargeback_bundle: float = 0.12   # per bundle; 2% gave <1 case in 37 bundles

    # --- defects that cannot be undone by a deterministic rule --------------
    # Without these the corpus is mechanically invertible: every corruption has
    # an exact inverse, Layers 1-2 recover everything, and Layer 3 is left with
    # nothing to do.  See README, "why the corpus was hardened".
    rate_narration_opaque: float = 0.18       # per bank credit: no UTR anywhere
    rate_unexplained_deduction: float = 0.10  # per bundle: credit short, unitemised
    # Opacity and unitemised deductions share a root cause -- a payout that went
    # out through a manual or cross-border path rather than the normal automated
    # one.  Modelling them as independent made the genuinely hard case (no
    # reference *and* no tie-out) occur about 4% of the time, which is too rare
    # to measure.  Conditioning one on the other reproduces the real correlation
    # without inflating either marginal rate.
    unexplained_deduction_given_opaque: float = 0.65
    rate_duplicate_customer_invoice: float = 0.06  # per payment: twin open invoice
    rate_tz_boundary: float = 0.03

    # --- genuinely unresolvable (5% of payments, split across reasons) ------
    rate_unresolvable_no_invoice: float = 0.025
    rate_unresolvable_orphan_order: float = 0.015
    rate_unresolvable_no_bank_credit: float = 0.010

    # --- standalone noise ---------------------------------------------------
    n_noise_bank_credits: int = 10
    n_orphan_invoices: int = 6

    # --- toggles ------------------------------------------------------------
    disabled: frozenset[DefectTag] = field(default_factory=frozenset)

    _RATE_BY_TAG = {
        DefectTag.TDS: "rate_tds",
        DefectTag.SPLIT_SETTLEMENT: "rate_split_settlement",
        DefectTag.PAISE_DRIFT_ROW: "rate_paise_drift_row",
        DefectTag.PAISE_DRIFT_BUNDLE: "rate_paise_drift_bundle",
        DefectTag.REFUND_FULL: "rate_refund",
        DefectTag.DUPLICATE_REFUND: "rate_duplicate_refund",
        DefectTag.NARRATION_CORRUPT: "rate_narration_corrupt",
        DefectTag.ERP_LINK_BROKEN: "rate_erp_link_broken",
        DefectTag.INVOICE_AMOUNT_MISMATCH: "rate_invoice_amount_mismatch",
        DefectTag.CHARGEBACK_ADJUSTMENT: "rate_chargeback_bundle",
        DefectTag.TZ_BOUNDARY: "rate_tz_boundary",
        DefectTag.NO_INVOICE: "rate_unresolvable_no_invoice",
        DefectTag.ORPHAN_ORDER_ID: "rate_unresolvable_orphan_order",
        DefectTag.NO_BANK_CREDIT: "rate_unresolvable_no_bank_credit",
        DefectTag.NARRATION_OPAQUE: "rate_narration_opaque",
        DefectTag.UNEXPLAINED_DEDUCTION: "rate_unexplained_deduction",
        DefectTag.DUPLICATE_CUSTOMER_INVOICE: "rate_duplicate_customer_invoice",
    }

    def rate_for(self, tag: DefectTag) -> float:
        """Effective rate for ``tag`` -- zero when the defect is disabled."""
        if tag in self.disabled:
            return 0.0
        attr = self._RATE_BY_TAG.get(tag)
        if attr is None:
            return 0.0
        return float(getattr(self, attr))

    def enabled(self, tag: DefectTag) -> bool:
        return tag not in self.disabled

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["disabled"] = sorted(str(t) for t in self.disabled)
        raw["window_start"] = self.window_start.isoformat()
        raw["window_end"] = self.window_end.isoformat()
        raw["gst_on_fee_pct"] = str(self.gst_on_fee_pct)
        raw["tds_pct"] = str(self.tds_pct)
        raw["large_bundle_size"] = list(self.large_bundle_size)
        raw["small_bundle_size"] = list(self.small_bundle_size)
        return raw

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
