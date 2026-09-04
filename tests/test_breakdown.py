"""Per-defect, per-layer and per-exception analysis."""

from __future__ import annotations

from conftest import make_link, make_truth

from recon.eval.breakdown import (
    UNINFORMATIVE_TAGS,
    cost_breakdown,
    defect_breakdown,
    exception_breakdown,
    informative_tags,
    layer_breakdown,
)
from recon.matcher.types import (
    ExceptionReason,
    LinkType,
    Match,
    MatchLayer,
    ReconException,
    ReconResult,
    SubjectType,
)
from recon.models import DefectTag
from recon.obs.logging import CallLog, CallRecord

TDS = DefectTag.TDS
OPAQUE = DefectTag.NARRATION_OPAQUE
BUNDLED = DefectTag.BUNDLED


def _result(matches=(), exceptions=()) -> ReconResult:
    return ReconResult(
        matcher="t", split="t", matches=list(matches), exceptions=list(exceptions)
    )


def _match(payment_id, counterparts, link_type, rule="r", layer=MatchLayer.L1_EXACT):
    return Match.build(
        link_type=link_type,
        payment_id=payment_id,
        counterpart_ids=counterparts,
        layer=layer,
        rule=rule,
        confidence=1.0,
        evidence=["e"],
    )


# --- tag hygiene ------------------------------------------------------------
def test_the_universal_tag_is_excluded_from_isolation() -> None:
    """`bundled_payout` is on every payment, so counting it isolates nothing."""
    assert BUNDLED in UNINFORMATIVE_TAGS
    link = make_link(defect_tags=[TDS], bundle_defect_tags=[BUNDLED])
    assert informative_tags(link) == {str(TDS)}


def test_own_and_batch_tags_are_pooled_for_analysis() -> None:
    link = make_link(defect_tags=[TDS], bundle_defect_tags=[OPAQUE])
    assert informative_tags(link) == {str(TDS), str(OPAQUE)}


# --- marginal vs isolated ---------------------------------------------------
def test_marginal_counts_everything_carrying_the_tag() -> None:
    truth = make_truth(
        [
            make_link("pay_1", defect_tags=[TDS]),
            make_link("pay_2", defect_tags=[TDS, OPAQUE]),
        ]
    )
    result = _result([_match("pay_1", ["bank_1"], LinkType.PAYMENT_TO_BANK)])
    rows, _ = defect_breakdown(result, truth)
    tds_row = next(r for r in rows if r.tag == str(TDS) and r.link_type == "payment_to_bank")
    assert tds_row.marginal_n == 2
    assert tds_row.marginal_resolved == 1


def test_isolated_counts_only_the_sole_defect() -> None:
    truth = make_truth(
        [
            make_link("pay_1", defect_tags=[TDS]),
            make_link("pay_2", defect_tags=[TDS, OPAQUE]),
        ]
    )
    rows, _ = defect_breakdown(_result(), truth)
    tds_row = next(r for r in rows if r.tag == str(TDS) and r.link_type == "payment_to_bank")
    assert tds_row.isolated_n == 1


def test_lift_is_measured_against_defect_free_payments() -> None:
    truth = make_truth(
        [
            make_link("pay_1"),                      # clean, resolved
            make_link("pay_2", defect_tags=[TDS]),   # defective, not resolved
        ]
    )
    result = _result([_match("pay_1", ["bank_1"], LinkType.PAYMENT_TO_BANK)])
    rows, clean = defect_breakdown(result, truth)
    assert clean["payment_to_bank"] == 1.0
    tds_row = next(r for r in rows if r.tag == str(TDS) and r.link_type == "payment_to_bank")
    assert tds_row.isolated_rate == 0.0
    assert tds_row.isolated_lift == -100.0


def test_lift_is_none_when_nothing_carries_the_tag_alone() -> None:
    truth = make_truth([make_link("pay_1", defect_tags=[TDS, OPAQUE])])
    rows, _ = defect_breakdown(_result(), truth)
    row = next(r for r in rows if r.tag == str(TDS) and r.link_type == "payment_to_bank")
    assert row.isolated_n == 0 and row.isolated_lift is None


def test_unresolvable_payments_are_left_out_of_the_denominators() -> None:
    truth = make_truth(
        [make_link("pay_1", bank_txn_ids=[], defect_tags=[TDS], unresolvable="none")]
    )
    rows, _ = defect_breakdown(_result(), truth)
    assert [r for r in rows if r.link_type == "payment_to_bank"] == []


# --- layer attribution ------------------------------------------------------
def test_layer_precision_is_measured_against_truth() -> None:
    truth = make_truth([make_link("pay_1"), make_link("pay_2")])
    result = _result(
        [
            _match("pay_1", ["bank_1"], LinkType.PAYMENT_TO_BANK, rule="good"),
            _match("pay_2", ["bank_wrong"], LinkType.PAYMENT_TO_BANK, rule="bad"),
        ]
    )
    rows = {r.rule: r for r in layer_breakdown(result, truth)}
    assert rows["good"].precision == 1.0
    assert rows["bad"].precision == 0.0


def test_a_rule_is_credited_only_for_edges_in_ground_truth() -> None:
    truth = make_truth([make_link("pay_1", bank_txn_ids=["bank_1"])])
    result = _result(
        [_match("pay_1", ["bank_1", "bank_extra"], LinkType.PAYMENT_TO_BANK, rule="loose")]
    )
    row = layer_breakdown(result, truth)[0]
    assert row.matches == 2 and row.correct == 1 and row.precision == 0.5


# --- exception correctness --------------------------------------------------
def _exception(subject_type, subject_id, reason, link_type=LinkType.PAYMENT_TO_INVOICE):
    return ReconException.build(
        subject_type=subject_type,
        subject_id=subject_id,
        link_type=link_type,
        reason=reason,
        detail="d",
        layer_reached=MatchLayer.L2_FUZZY,
    )


def test_an_exception_on_a_truly_unresolvable_payment_counts_as_correct() -> None:
    truth = make_truth([make_link("pay_1", invoice_id=None, unresolvable="no_erp")])
    result = _result(
        [], [_exception(SubjectType.PAYMENT, "pay_1", ExceptionReason.NO_CANDIDATE)]
    )
    row = exception_breakdown(result, truth)[0]
    assert row.correctly_raised == 1 and row.missed == 0


def test_an_exception_hiding_a_findable_answer_counts_as_a_miss() -> None:
    truth = make_truth([make_link("pay_1")])
    result = _result(
        [], [_exception(SubjectType.PAYMENT, "pay_1", ExceptionReason.NO_CANDIDATE)]
    )
    row = exception_breakdown(result, truth)[0]
    assert row.correctly_raised == 0 and row.missed == 1


def test_orphan_counterparts_are_judged_against_the_orphan_lists() -> None:
    truth = make_truth([make_link()], orphan_bank=["bank_ghost"])
    result = _result(
        [],
        [
            _exception(
                SubjectType.BANK_TXN,
                "bank_ghost",
                ExceptionReason.UNMATCHED_COUNTERPART,
                LinkType.PAYMENT_TO_BANK,
            ),
            _exception(
                SubjectType.BANK_TXN,
                "bank_real",
                ExceptionReason.UNMATCHED_COUNTERPART,
                LinkType.PAYMENT_TO_BANK,
            ),
        ],
    )
    row = exception_breakdown(result, truth)[0]
    assert row.total == 2 and row.correctly_raised == 1 and row.missed == 1


# --- cost -------------------------------------------------------------------
def test_cost_and_latency_are_normalised_to_one_hundred_records(tmp_path) -> None:
    log = CallLog(tmp_path / "c.jsonl")
    for i in range(4):
        log.append(
            CallRecord(
                call_id=f"c{i}",
                purpose="p",
                model="claude-opus-5",
                prompt_hash="h",
                system="s",
                user="u",
                response_text="{}",
                input_tokens=1000,
                output_tokens=100,
                latency_ms=2000,
            )
        )
    row = cost_breakdown(log, n_records=200, layer3_records=51)
    assert row.calls_per_100_records == 2.0
    assert row.latency_s_per_100_records == 4.0
    assert row.cost_usd_per_100_records == round(row.cost_usd / 2, 6)
    assert row.records_reaching_layer3 == 51


def test_cost_is_zero_when_no_calls_were_made(tmp_path) -> None:
    row = cost_breakdown(CallLog(tmp_path / "c.jsonl"), n_records=100)
    assert row.llm_calls == 0 and row.cost_usd == 0.0


def test_transcript_misses_are_not_counted_as_work_done(tmp_path) -> None:
    """A replay run where every prompt missed has made no decisions at all."""
    log = CallLog(tmp_path / "c.jsonl")
    log.append(
        CallRecord(
            call_id="c1",
            purpose="p",
            model="claude-opus-5",
            prompt_hash="h",
            system="s",
            user="u",
            response_text="",
            error="no recorded response for this prompt",
            replayed=True,
        )
    )
    log.append(
        CallRecord(
            call_id="c2",
            purpose="p",
            model="claude-opus-5",
            prompt_hash="h2",
            system="s",
            user="u2",
            response_text="{}",
            input_tokens=500,
            output_tokens=50,
            replayed=True,
        )
    )
    row = cost_breakdown(log, n_records=100)
    assert row.llm_calls == 2
    assert row.answered == 1
    assert row.transcript_misses == 1
