"""The scorer is the thing every claim in the README rests on, so it is tested
against hand-built cases where the right answer is obvious by inspection."""

from __future__ import annotations

import pytest
from conftest import make_link, make_truth

from recon.eval.scoring import score, wilson_interval
from recon.matcher.types import (
    ExceptionReason,
    LinkType,
    Match,
    MatchLayer,
    ReconException,
    ReconResult,
    SubjectType,
)


def _result(matches=(), exceptions=()) -> ReconResult:
    return ReconResult(
        matcher="test", split="test", matches=list(matches), exceptions=list(exceptions)
    )


def _match(payment_id: str, counterparts: list[str], link_type: LinkType, confidence=1.0) -> Match:
    return Match.build(
        link_type=link_type,
        payment_id=payment_id,
        counterpart_ids=counterparts,
        layer=MatchLayer.L1_EXACT,
        rule="test",
        confidence=confidence,
        evidence=["because"],
    )


def _exception(payment_id: str, link_type: LinkType) -> ReconException:
    return ReconException.build(
        subject_type=SubjectType.PAYMENT,
        subject_id=payment_id,
        link_type=link_type,
        reason=ExceptionReason.NO_CANDIDATE,
        detail="none",
        layer_reached=MatchLayer.L1_EXACT,
    )


# --- Wilson intervals ------------------------------------------------------
def test_wilson_is_empty_for_no_trials() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_stays_inside_the_unit_interval_at_the_extremes() -> None:
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0.0 < high < 0.35
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and 0.65 < low < 1.0


def test_wilson_narrows_as_the_sample_grows() -> None:
    small = wilson_interval(5, 10)
    large = wilson_interval(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


# --- perfect and empty matchers -------------------------------------------
def test_a_perfect_matcher_scores_one_everywhere() -> None:
    truth = make_truth([make_link()])
    result = _result(
        [
            _match("pay_1", ["bank_1"], LinkType.PAYMENT_TO_BANK),
            _match("pay_1", ["INV-2026-00001"], LinkType.PAYMENT_TO_INVOICE),
        ]
    )
    card = score(result, truth)
    assert card.fully_reconciled_rate == 1.0
    for link_type in LinkType:
        metrics = card.link(link_type)
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.exact_resolution_rate == 1.0


def test_a_matcher_that_asserts_nothing_scores_zero_recall_not_zero_precision() -> None:
    truth = make_truth([make_link()])
    card = score(_result(), truth)
    metrics = card.link(LinkType.PAYMENT_TO_BANK)
    assert metrics.recall == 0.0
    assert metrics.precision == 0.0  # no predictions at all
    assert metrics.predicted_edges == 0


# --- split settlements get partial credit on edges, none on exactness ------
def test_half_a_split_settlement_is_half_recall_and_zero_exact() -> None:
    truth = make_truth([make_link(bank_txn_ids=["bank_1", "bank_2"])])
    result = _result([_match("pay_1", ["bank_1"], LinkType.PAYMENT_TO_BANK)])
    metrics = score(result, truth).link(LinkType.PAYMENT_TO_BANK)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.precision == 1.0
    assert metrics.exact_resolution_rate == 0.0


def test_both_legs_of_a_split_scores_exact() -> None:
    truth = make_truth([make_link(bank_txn_ids=["bank_1", "bank_2"])])
    result = _result([_match("pay_1", ["bank_1", "bank_2"], LinkType.PAYMENT_TO_BANK)])
    metrics = score(result, truth).link(LinkType.PAYMENT_TO_BANK)
    assert metrics.recall == 1.0
    assert metrics.exact_resolution_rate == 1.0


def test_a_wrong_extra_counterpart_costs_precision() -> None:
    truth = make_truth([make_link()])
    result = _result([_match("pay_1", ["bank_1", "bank_9"], LinkType.PAYMENT_TO_BANK)])
    metrics = score(result, truth).link(LinkType.PAYMENT_TO_BANK)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == 1.0
    assert metrics.exact_resolution_rate == 0.0


# --- honesty ---------------------------------------------------------------
def test_matching_an_unresolvable_record_is_counted_as_a_hallucination() -> None:
    truth = make_truth([make_link(invoice_id=None, unresolvable="no_erp_counterpart")])
    result = _result([_match("pay_1", ["INV-2026-00001"], LinkType.PAYMENT_TO_INVOICE)])
    card = score(result, truth)
    honesty = next(h for h in card.honesty if h.link_type == LinkType.PAYMENT_TO_INVOICE.value)
    assert honesty.unresolvable_payments == 1
    assert honesty.falsely_matched == 1
    assert honesty.hallucination_rate == 1.0
    assert card.fully_reconciled_rate == 0.0


def test_correctly_excepting_an_unresolvable_record_counts_as_reconciled() -> None:
    truth = make_truth([make_link(invoice_id=None, unresolvable="no_erp_counterpart")])
    result = _result(
        [_match("pay_1", ["bank_1"], LinkType.PAYMENT_TO_BANK)],
        [_exception("pay_1", LinkType.PAYMENT_TO_INVOICE)],
    )
    card = score(result, truth)
    honesty = next(h for h in card.honesty if h.link_type == LinkType.PAYMENT_TO_INVOICE.value)
    assert honesty.falsely_matched == 0
    assert honesty.routed_to_exceptions == 1
    assert card.fully_reconciled_rate == 1.0


def test_unresolvable_records_are_excluded_from_recall_denominators() -> None:
    truth = make_truth(
        [make_link("pay_1"), make_link("pay_2", bank_txn_ids=[], unresolvable="never_credited")]
    )
    result = _result([_match("pay_1", ["bank_1"], LinkType.PAYMENT_TO_BANK)])
    metrics = score(result, truth).link(LinkType.PAYMENT_TO_BANK)
    assert metrics.resolvable_payments == 1
    assert metrics.recall == 1.0


# --- confidence threshold --------------------------------------------------
def test_matches_below_the_threshold_are_not_counted_as_predictions() -> None:
    truth = make_truth([make_link()])
    result = _result([_match("pay_1", ["bank_1"], LinkType.PAYMENT_TO_BANK, confidence=0.4)])
    assert score(result, truth, confidence_threshold=0.7).link(
        LinkType.PAYMENT_TO_BANK
    ).predicted_edges == 0
    assert score(result, truth, confidence_threshold=0.3).link(
        LinkType.PAYMENT_TO_BANK
    ).predicted_edges == 1


# --- orphan counterparts ---------------------------------------------------
def test_orphan_counterpart_metrics() -> None:
    truth = make_truth([make_link()], orphan_bank=["bank_ghost"])
    flagged = ReconException.build(
        subject_type=SubjectType.BANK_TXN,
        subject_id="bank_ghost",
        link_type=LinkType.PAYMENT_TO_BANK,
        reason=ExceptionReason.UNMATCHED_COUNTERPART,
        detail="nothing claimed it",
        layer_reached=MatchLayer.L1_EXACT,
    )
    card = score(_result([], [flagged]), truth)
    bank = next(c for c in card.counterparts if c.subject_type == SubjectType.BANK_TXN.value)
    assert bank.orphans_in_ground_truth == 1
    assert bank.correctly_flagged == 1
    assert bank.precision == 1.0
    assert bank.recall == 1.0
