"""The deterministic baselines are pure functions, tested in isolation."""

from __future__ import annotations

from datetime import date

from conftest import make_bank, make_invoice, make_payment, make_sources

from recon.matcher.baseline import run_baseline, run_id_join_baseline
from recon.matcher.types import ExceptionReason, LinkType, MatchLayer, SubjectType


def _bank_matches(result):
    return result.matches_for(LinkType.PAYMENT_TO_BANK)


def _invoice_matches(result):
    return result.matches_for(LinkType.PAYMENT_TO_INVOICE)


def _payment_exceptions(result, link_type):
    return [
        x
        for x in result.exceptions_for(link_type)
        if x.subject_type is SubjectType.PAYMENT
    ]


# --- exact amount + date ---------------------------------------------------
def test_unique_exact_match_is_asserted_with_full_confidence() -> None:
    result = run_baseline(make_sources(), "test")
    matches = _bank_matches(result)
    assert len(matches) == 1
    assert matches[0].counterpart_ids == ["bank_1"]
    assert matches[0].confidence == 1.0
    assert matches[0].layer is MatchLayer.BASELINE
    assert matches[0].evidence


def test_no_candidate_becomes_a_typed_exception() -> None:
    sources = make_sources(bank=[make_bank(credit=55_555)])
    result = run_baseline(sources, "test")
    assert _bank_matches(result) == []
    exceptions = _payment_exceptions(result, LinkType.PAYMENT_TO_BANK)
    assert [x.reason for x in exceptions] == [ExceptionReason.NO_CANDIDATE]


def test_two_equal_candidates_are_refused_not_guessed() -> None:
    sources = make_sources(bank=[make_bank("bank_1"), make_bank("bank_2")])
    result = run_baseline(sources, "test")
    assert _bank_matches(result) == []
    exceptions = _payment_exceptions(result, LinkType.PAYMENT_TO_BANK)
    assert exceptions[0].reason is ExceptionReason.AMBIGUOUS_CANDIDATES
    assert exceptions[0].candidates_considered == 2


def test_the_date_predicate_actually_excludes() -> None:
    sources = make_sources(bank=[make_bank(value_date=date(2026, 2, 9))])
    with_date = run_baseline(sources, "test", use_date=True)
    without_date = run_baseline(sources, "test", use_date=False)
    assert _bank_matches(with_date) == []
    assert len(_bank_matches(without_date)) == 1


def test_invoice_leg_matches_gross_not_net() -> None:
    result = run_baseline(make_sources(), "test")
    matches = _invoice_matches(result)
    assert len(matches) == 1
    assert matches[0].counterpart_ids == ["INV-2026-00001"]


def test_unclaimed_counterparts_are_reported() -> None:
    sources = make_sources(
        bank=[make_bank("bank_1"), make_bank("bank_ghost", credit=12_345, utr=None)],
        invoices=[make_invoice(), make_invoice("INV-2026-09999", amount=9_999)],
    )
    result = run_baseline(sources, "test")
    unmatched = {
        x.subject_id
        for x in result.exceptions
        if x.reason is ExceptionReason.UNMATCHED_COUNTERPART
    }
    assert unmatched == {"bank_ghost", "INV-2026-09999"}


# --- identifier join -------------------------------------------------------
def test_identifier_join_resolves_by_utr_column() -> None:
    result = run_id_join_baseline(make_sources(), "test")
    assert _bank_matches(result)[0].counterpart_ids == ["bank_1"]


def test_identifier_join_fails_when_the_utr_column_is_null() -> None:
    sources = make_sources(bank=[make_bank(utr=None)])
    result = run_id_join_baseline(sources, "test")
    assert _bank_matches(result) == []
    assert _payment_exceptions(result, LinkType.PAYMENT_TO_BANK)[0].reason is (
        ExceptionReason.NO_CANDIDATE
    )


def test_identifier_join_returns_both_legs_of_a_split_settlement() -> None:
    payment = make_payment(
        utrs=["HDFC000000000001", "HDFC000000000002"],
        settled=[date(2026, 2, 5), date(2026, 2, 12)],
        rows=2,
    )
    sources = make_sources(
        payments=[payment],
        bank=[
            make_bank("bank_1", utr="HDFC000000000001"),
            make_bank("bank_2", utr="HDFC000000000002", value_date=date(2026, 2, 12)),
        ],
    )
    result = run_id_join_baseline(sources, "test")
    assert _bank_matches(result)[0].counterpart_ids == ["bank_1", "bank_2"]


def test_a_half_resolved_split_is_refused_rather_than_half_asserted() -> None:
    payment = make_payment(utrs=["HDFC000000000001", "HDFC000000000002"], rows=2)
    sources = make_sources(payments=[payment], bank=[make_bank("bank_1")])
    result = run_id_join_baseline(sources, "test")
    assert _bank_matches(result) == []


def test_missing_order_id_is_its_own_exception_reason() -> None:
    sources = make_sources(payments=[make_payment(order_id=None)])
    result = run_id_join_baseline(sources, "test")
    exceptions = _payment_exceptions(result, LinkType.PAYMENT_TO_INVOICE)
    assert exceptions[0].reason is ExceptionReason.NO_ORDER_ID


def test_broken_erp_link_is_not_silently_guessed() -> None:
    sources = make_sources(invoices=[make_invoice(order_id="order_TYPO")])
    result = run_id_join_baseline(sources, "test")
    assert _invoice_matches(result) == []


# --- properties ------------------------------------------------------------
def test_baselines_are_deterministic() -> None:
    sources = make_sources()
    for runner in (run_baseline, run_id_join_baseline):
        first = runner(sources, "test")
        second = runner(sources, "test")
        assert first.model_dump_json() == second.model_dump_json()


def test_every_payment_gets_a_verdict_on_both_legs() -> None:
    payments = [make_payment(f"pay_{i}", order_id=f"order_{i}") for i in range(5)]
    sources = make_sources(payments=payments)
    for result in (run_baseline(sources, "test"), run_id_join_baseline(sources, "test")):
        for link_type in LinkType:
            decided = {m.payment_id for m in result.matches_for(link_type)}
            decided |= {x.subject_id for x in _payment_exceptions(result, link_type)}
            assert decided == {p.payment_id for p in payments}
