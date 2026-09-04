"""The layered pipeline: ordering, thresholds, propagation, and honesty."""

from __future__ import annotations

from datetime import date

from conftest import make_bank, make_batch, make_invoice, make_payment, make_sources

from recon.llm.client import StubClient
from recon.llm.schemas import LinkDecision
from recon.matcher.layer3 import LLMResolver
from recon.matcher.pipeline import run_layered
from recon.matcher.types import ExceptionReason, LinkType, MatchLayer, SubjectType

OPAQUE = {"narration": "BULK CR RAZORPAY SOFTWARE PRIVATE LIMITED"}


def _bank_match(result):
    return result.matches_for(LinkType.PAYMENT_TO_BANK)


def test_layer_one_wins_when_it_can_resolve() -> None:
    result = run_layered(make_sources(), "test")
    assert _bank_match(result)[0].layer is MatchLayer.L1_EXACT


def test_layer_two_only_sees_what_layer_one_could_not_do() -> None:
    row = make_bank(utr=None).model_copy(
        update={"narration": "NEFT-HDFC000000000001-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT"}
    )
    result = run_layered(make_sources(bank=[row]), "test")
    assert _bank_match(result)[0].layer is MatchLayer.L2_FUZZY


def test_layer_three_is_never_called_when_layers_one_and_two_succeed() -> None:
    stub = StubClient({})
    run_layered(make_sources(), "test", resolver=LLMResolver(stub))
    assert stub.calls == []


def test_layer_three_runs_on_the_residue_and_is_labelled_as_such() -> None:
    row = make_bank(utr=None, credit=96_500).model_copy(update=OPAQUE)
    stub = StubClient(
        {"bank:setl_1": LinkDecision(chosen_id="bank_1", confidence=0.9, reasoning="0.5% short")}
    )
    result = run_layered(
        make_sources(bank=[row], batches=[make_batch(expected=97_000)]),
        "test",
        resolver=LLMResolver(stub),
    )
    match = _bank_match(result)[0]
    assert match.layer is MatchLayer.L3_LLM
    assert stub.calls  # it was actually consulted


def test_a_batch_answer_propagates_to_every_payment_in_it() -> None:
    payments = [make_payment(f"pay_{i}", order_id=f"order_{i}") for i in range(4)]
    batch = make_batch(payment_ids=[p.payment_id for p in payments])
    result = run_layered(make_sources(payments=payments, batches=[batch]), "test")
    assert {m.payment_id for m in _bank_match(result)} == {p.payment_id for p in payments}
    assert all(m.counterpart_ids == ["bank_1"] for m in _bank_match(result))


def test_a_split_settlement_collects_both_legs() -> None:
    payment = make_payment(utrs=["HDFC000000000001", "HDFC000000000002"], rows=2)
    payment = payment.model_copy(update={"settlement_ids": ["setl_1", "setl_2"]})
    sources = make_sources(
        payments=[payment],
        bank=[
            make_bank("bank_1", utr="HDFC000000000001"),
            make_bank("bank_2", utr="HDFC000000000002", value_date=date(2026, 2, 12)),
        ],
        batches=[
            make_batch("setl_1", utr="HDFC000000000001"),
            make_batch("setl_2", utr="HDFC000000000002", settled=date(2026, 2, 12)),
        ],
    )
    result = run_layered(sources, "test")
    assert _bank_match(result)[0].counterpart_ids == ["bank_1", "bank_2"]


def test_a_half_resolved_split_asserts_what_it_knows_and_flags_the_rest() -> None:
    """Partial credit is honest: state the leg you tied, queue the one you did not."""
    payment = make_payment(utrs=["HDFC000000000001", "HDFC000000000002"], rows=2)
    payment = payment.model_copy(update={"settlement_ids": ["setl_1", "setl_2"]})
    sources = make_sources(
        payments=[payment],
        bank=[make_bank("bank_1", utr="HDFC000000000001")],
        batches=[
            make_batch("setl_1", utr="HDFC000000000001"),
            make_batch("setl_2", utr="HDFC000000000002", settled=date(2026, 2, 12), expected=1),
        ],
    )
    result = run_layered(sources, "test")
    assert _bank_match(result)[0].counterpart_ids == ["bank_1"]
    reasons = {x.reason for x in result.exceptions_for(LinkType.PAYMENT_TO_BANK)}
    assert ExceptionReason.NO_CANDIDATE in reasons
    assert any("only 1 of 2" in line for m in _bank_match(result) for line in m.evidence)


def test_a_proposal_below_the_threshold_is_routed_to_exceptions() -> None:
    row = make_bank(utr=None, credit=96_500).model_copy(update=OPAQUE)
    stub = StubClient(
        {"bank:setl_1": LinkDecision(chosen_id="bank_1", confidence=0.3, reasoning="unsure")}
    )
    result = run_layered(
        make_sources(bank=[row], batches=[make_batch(expected=97_000)]),
        "test",
        resolver=LLMResolver(stub),
        threshold=0.7,
    )
    assert _bank_match(result) == []
    exception = next(
        x
        for x in result.exceptions_for(LinkType.PAYMENT_TO_BANK)
        if x.subject_type is SubjectType.PAYMENT
    )
    assert exception.reason is ExceptionReason.BELOW_CONFIDENCE
    assert exception.evidence  # the rejected reasoning is still auditable


def test_lowering_the_threshold_admits_the_same_proposal() -> None:
    row = make_bank(utr=None, credit=96_500).model_copy(update=OPAQUE)
    stub = StubClient(
        {"bank:setl_1": LinkDecision(chosen_id="bank_1", confidence=0.9, reasoning="fits")}
    )
    sources = make_sources(bank=[row], batches=[make_batch(expected=97_000)])
    strict = run_layered(sources, "t", resolver=LLMResolver(stub), threshold=0.9)
    lax = run_layered(sources, "t", resolver=LLMResolver(stub), threshold=0.5)
    assert _bank_match(strict) == []
    assert _bank_match(lax)


def test_every_match_carries_a_full_audit_trail() -> None:
    result = run_layered(make_sources(), "test")
    for match in result.matches:
        assert match.match_id.startswith("m_")
        assert match.rule and match.evidence
        assert 0.0 <= match.confidence <= 1.0
        assert match.source_records["gateway"] == [match.payment_id]


def test_unclaimed_counterparts_are_listed_as_exceptions() -> None:
    sources = make_sources(
        bank=[make_bank("bank_1"), make_bank("bank_ghost", utr=None, credit=1)],
        invoices=[make_invoice(), make_invoice("INV-2026-09999", amount=7, order_id=None)],
    )
    result = run_layered(sources, "test")
    unmatched = {
        x.subject_id
        for x in result.exceptions
        if x.reason is ExceptionReason.UNMATCHED_COUNTERPART
    }
    assert unmatched == {"bank_ghost", "INV-2026-09999"}


def test_the_pipeline_is_deterministic() -> None:
    sources = make_sources()
    first = run_layered(sources, "t").model_dump_json()
    assert first == run_layered(sources, "t").model_dump_json()
