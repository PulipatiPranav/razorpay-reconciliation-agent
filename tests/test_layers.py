"""Layer 1 and Layer 2 rules, tested as the pure functions they are.

Each test builds the minimum records needed to make one rule fire or refuse.
Testing them against the 600-record corpus would tell you a score moved; this
tells you which rule broke.
"""

from __future__ import annotations

from datetime import date

from conftest import make_bank, make_batch, make_invoice, make_payment

from recon.matcher.confidence import (
    L1_ORDER_ID_JOIN,
    L1_UTR_COLUMN_JOIN,
    L2_BATCH_AMOUNT_DEDUPED,
    L2_BATCH_AMOUNT_RECONSTRUCTION,
    L2_INVOICE_AMOUNT_WINDOW,
    L2_NARRATION_UTR_EXACT,
    L2_NARRATION_UTR_TRUNCATED,
    L2_ORDER_ID_FUZZY,
    L2_ORDER_ID_NORMALISED,
    L2_RECEIPT_INVOICE_HINT,
    L2_UTR_FUZZY,
)
from recon.matcher.layer1 import resolve_batches_exact, resolve_invoices_exact
from recon.matcher.layer2 import (
    BATCH_AMOUNT_TOLERANCE_PAISE,
    resolve_batches_fuzzy,
    resolve_invoices_fuzzy,
)

UTR = "HDFC000000000001"


# --- Layer 1 ---------------------------------------------------------------
def test_l1_joins_the_utr_column() -> None:
    out = resolve_batches_exact([make_batch()], [make_bank()])
    assert out["setl_1"].counterpart_ids == ["bank_1"]
    assert out["setl_1"].rule == L1_UTR_COLUMN_JOIN


def test_l1_refuses_when_two_credits_share_a_utr() -> None:
    rows = [make_bank("bank_1"), make_bank("bank_2")]
    assert resolve_batches_exact([make_batch()], rows) == {}


def test_l1_will_not_take_an_already_claimed_credit() -> None:
    out = resolve_batches_exact([make_batch()], [make_bank()], claimed={"bank_1"})
    assert out == {}


def test_l1_joins_order_id() -> None:
    out = resolve_invoices_exact([make_payment()], [make_invoice()])
    assert out["pay_1"].counterpart_ids == ["INV-2026-00001"]
    assert out["pay_1"].rule == L1_ORDER_ID_JOIN


def test_l1_skips_payments_with_no_order_id() -> None:
    assert resolve_invoices_exact([make_payment(order_id=None)], [make_invoice()]) == {}


# --- Layer 2, bank leg -----------------------------------------------------
def test_l2_recovers_a_utr_that_only_exists_in_the_narration() -> None:
    row = make_bank(utr=None).model_copy(
        update={"narration": f"NEFT-{UTR}-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT"}
    )
    out = resolve_batches_fuzzy([make_batch()], [row])
    assert out["setl_1"].rule == L2_NARRATION_UTR_EXACT


def test_l2_recovers_a_truncated_utr() -> None:
    row = make_bank(utr=None).model_copy(update={"narration": f"RTGS-{UTR[:12]}-RZPY BATCH"})
    out = resolve_batches_fuzzy([make_batch()], [row])
    assert out["setl_1"].rule == L2_NARRATION_UTR_TRUNCATED


def test_l2_recovers_a_transposed_utr() -> None:
    transposed = UTR[:-2] + UTR[-1] + UTR[-2]
    row = make_bank(utr=transposed).model_copy(update={"narration": "BULK CR RAZORPAY"})
    out = resolve_batches_fuzzy([make_batch()], [row])
    assert out["setl_1"].rule == L2_UTR_FUZZY


def test_l2_will_not_fuzzy_match_an_unrelated_utr() -> None:
    row = make_bank(utr="AXIS999999999999").model_copy(update={"narration": "BULK CR RAZORPAY"})
    # amounts also differ, so no rule should fire at all
    batch = make_batch(expected=12_345)
    assert resolve_batches_fuzzy([batch], [row]) == {}


def test_l2_reconstructs_the_batch_total_when_no_reference_survives() -> None:
    row = make_bank(utr=None, credit=97_000).model_copy(
        update={"narration": "BULK CR RAZORPAY SOFTWARE PRIVATE LIMITED"}
    )
    out = resolve_batches_fuzzy([make_batch(expected=97_000)], [row])
    assert out["setl_1"].rule == L2_BATCH_AMOUNT_RECONSTRUCTION


def test_l2_tolerates_bank_truncation_but_not_more() -> None:
    opaque = {"narration": "BULK CR RAZORPAY SOFTWARE PRIVATE LIMITED"}
    inside = make_bank(utr=None, credit=97_000 - BATCH_AMOUNT_TOLERANCE_PAISE).model_copy(
        update=opaque
    )
    outside = make_bank(utr=None, credit=97_000 - BATCH_AMOUNT_TOLERANCE_PAISE - 1).model_copy(
        update=opaque
    )
    assert resolve_batches_fuzzy([make_batch(expected=97_000)], [inside])
    assert resolve_batches_fuzzy([make_batch(expected=97_000)], [outside]) == {}


def test_l2_falls_back_to_the_deduplicated_total_for_a_phantom_refund() -> None:
    # The report shows a refund twice; the bank moved it once, so only the
    # deduplicated figure ties.
    row = make_bank(utr=None, credit=90_000).model_copy(
        update={"narration": "BULK CR RAZORPAY SOFTWARE PRIVATE LIMITED"}
    )
    batch = make_batch(expected=83_000, expected_dedup=90_000, duplicates=True)
    out = resolve_batches_fuzzy([batch], [row])
    assert out["setl_1"].rule == L2_BATCH_AMOUNT_DEDUPED
    assert any("duplicated" in line for line in out["setl_1"].evidence)


def test_l2_refuses_when_two_credits_both_tie() -> None:
    opaque = {"narration": "BULK CR RAZORPAY SOFTWARE PRIVATE LIMITED"}
    rows = [
        make_bank("bank_1", utr=None, credit=97_000).model_copy(update=opaque),
        make_bank("bank_2", utr=None, credit=97_000).model_copy(update=opaque),
    ]
    assert resolve_batches_fuzzy([make_batch(expected=97_000)], rows) == {}


def test_l2_respects_the_date_window() -> None:
    row = make_bank(utr=None, credit=97_000, value_date=date(2026, 3, 20)).model_copy(
        update={"narration": "BULK CR RAZORPAY SOFTWARE PRIVATE LIMITED"}
    )
    assert resolve_batches_fuzzy([make_batch(expected=97_000)], [row]) == {}


# --- Layer 2, invoice leg --------------------------------------------------
def test_l2_uses_the_invoice_number_in_the_receipt_field() -> None:
    payment = make_payment(order_id="order_missing")
    payment = payment.model_copy(update={"order_receipt": "INV-2026-00001"})
    out = resolve_invoices_fuzzy([payment], [make_invoice(order_id=None)])
    assert out["pay_1"].rule == L2_RECEIPT_INVOICE_HINT


def test_l2_folds_case_on_the_order_id() -> None:
    out = resolve_invoices_fuzzy([make_payment()], [make_invoice(order_id="ORDER_1")])
    assert out["pay_1"].rule == L2_ORDER_ID_NORMALISED


def test_l2_recovers_a_transposed_order_id() -> None:
    out = resolve_invoices_fuzzy(
        [make_payment(order_id="order_abcd")], [make_invoice(order_id="order_abdc")]
    )
    assert out["pay_1"].rule == L2_ORDER_ID_FUZZY


def test_l2_matches_on_amount_inside_the_issue_window() -> None:
    invoice = make_invoice(order_id=None, issue=date(2026, 1, 28))
    out = resolve_invoices_fuzzy([make_payment()], [invoice])
    assert out["pay_1"].rule == L2_INVOICE_AMOUNT_WINDOW


def test_l2_refuses_two_twin_invoices_on_amount_alone() -> None:
    twins = [
        make_invoice("INV-2026-00001", order_id=None, issue=date(2026, 1, 30)),
        make_invoice("INV-2026-00002", order_id=None, issue=date(2026, 2, 1)),
    ]
    assert resolve_invoices_fuzzy([make_payment(order_id=None)], twins) == {}


def test_l2_ignores_an_invoice_issued_after_capture() -> None:
    invoice = make_invoice(order_id=None, issue=date(2026, 2, 10))
    assert resolve_invoices_fuzzy([make_payment()], [invoice]) == {}


def test_l2_rules_are_tried_in_reliability_order() -> None:
    # Both the receipt hint and the amount window would fire; the hint wins.
    payment = make_payment(order_id=None).model_copy(update={"order_receipt": "INV-2026-00001"})
    invoices = [make_invoice("INV-2026-00001", order_id=None, issue=date(2026, 1, 30))]
    out = resolve_invoices_fuzzy([payment], invoices)
    assert out["pay_1"].rule == L2_RECEIPT_INVOICE_HINT


def test_l2_never_spends_a_counterpart_twice() -> None:
    payments = [make_payment("pay_1", order_id=None), make_payment("pay_2", order_id=None)]
    invoice = make_invoice(order_id=None, issue=date(2026, 1, 30))
    out = resolve_invoices_fuzzy(payments, [invoice])
    claimed = [cid for res in out.values() for cid in res.counterpart_ids]
    assert len(claimed) == len(set(claimed))
