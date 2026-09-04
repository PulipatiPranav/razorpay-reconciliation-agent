"""Layer 3, driven by a stub client -- no network, no key, no cost.

The tests that matter here are the refusals. A layer that resolves everything
it is asked about is worse than useless on a corpus where 5% of records have no
answer, so most of this file is about the model being ignored when it should be.
"""

from __future__ import annotations

from datetime import date

from conftest import make_bank, make_batch, make_invoice, make_payment

from recon.llm.client import StubClient
from recon.llm.schemas import LinkDecision
from recon.matcher.confidence import LLM_CONFIDENCE_CEILING
from recon.matcher.layer3 import LLMResolver, _bank_prompt, _invoice_prompt

OPAQUE = {"narration": "BULK CR RAZORPAY SOFTWARE PRIVATE LIMITED"}


def _opaque_bank(txn_id="bank_1", credit=96_500, value_date=date(2026, 2, 5)):
    return make_bank(txn_id, utr=None, credit=credit, value_date=value_date).model_copy(
        update=OPAQUE
    )


def test_a_confident_choice_becomes_a_resolution() -> None:
    stub = StubClient(
        {"bank:setl_1": LinkDecision(chosen_id="bank_1", confidence=0.9, reasoning="0.5% short")}
    )
    out = LLMResolver(stub).resolve_batches([make_batch()], [_opaque_bank()])
    assert out["setl_1"].counterpart_ids == ["bank_1"]
    assert any("model reasoning" in line for line in out["setl_1"].evidence)


def test_model_confidence_is_capped_never_raised() -> None:
    stub = StubClient(
        {"bank:setl_1": LinkDecision(chosen_id="bank_1", confidence=1.0, reasoning="certain")}
    )
    out = LLMResolver(stub).resolve_batches([make_batch()], [_opaque_bank()])
    assert out["setl_1"].confidence == LLM_CONFIDENCE_CEILING
    assert out["setl_1"].confidence < 1.0


def test_declining_produces_no_resolution() -> None:
    stub = StubClient(
        {"bank:setl_1": LinkDecision(chosen_id=None, confidence=0.0, reasoning="two fit equally")}
    )
    resolver = LLMResolver(stub)
    assert resolver.resolve_batches([make_batch()], [_opaque_bank()]) == {}
    assert resolver.declined == ["setl_1"]


def test_an_invented_identifier_is_treated_as_a_decline() -> None:
    """The model must choose from the offered list; anything else is a hallucination."""
    stub = StubClient(
        {
            "bank:setl_1": LinkDecision(
                chosen_id="bank_does_not_exist", confidence=0.99, reasoning=""
            )
        }
    )
    resolver = LLMResolver(stub)
    assert resolver.resolve_batches([make_batch()], [_opaque_bank()]) == {}
    assert resolver.declined == ["setl_1"]


def test_a_schema_failure_is_recorded_not_raised() -> None:
    resolver = LLMResolver(StubClient({}))  # returns None, as a failed parse does
    assert resolver.resolve_batches([make_batch()], [_opaque_bank()]) == {}
    assert resolver.schema_failures == ["setl_1"]


def test_no_call_is_made_when_there_are_no_candidates() -> None:
    stub = StubClient({})
    far_away = _opaque_bank(value_date=date(2026, 6, 1))
    LLMResolver(stub).resolve_batches([make_batch()], [far_away])
    assert stub.calls == []


def test_a_credit_larger_than_the_batch_is_not_offered() -> None:
    stub = StubClient({})
    LLMResolver(stub).resolve_batches([make_batch(expected=97_000)], [_opaque_bank(credit=200_000)])
    assert stub.calls == []


def test_an_implausibly_short_credit_is_not_offered() -> None:
    stub = StubClient({})
    # 20% short is not a platform fee; it is a different payout.
    LLMResolver(stub).resolve_batches([make_batch(expected=97_000)], [_opaque_bank(credit=77_000)])
    assert stub.calls == []


def test_the_llm_never_claims_one_credit_for_two_batches() -> None:
    decision = LinkDecision(chosen_id="bank_1", confidence=0.9, reasoning="fits")
    stub = StubClient({"bank:setl_1": decision, "bank:setl_2": decision})
    batches = [make_batch("setl_1"), make_batch("setl_2")]
    out = LLMResolver(stub).resolve_batches(batches, [_opaque_bank()])
    assert len(out) == 1


def test_invoice_leg_resolves_and_caps_confidence() -> None:
    stub = StubClient(
        {
            "invoice:pay_1": LinkDecision(
                chosen_id="INV-2026-00001", confidence=0.8, reasoning="only one in window"
            )
        }
    )
    out = LLMResolver(stub).resolve_invoices([make_payment(order_id=None)], [make_invoice()])
    assert out["pay_1"].counterpart_ids == ["INV-2026-00001"]
    assert out["pay_1"].confidence == 0.8 * LLM_CONFIDENCE_CEILING


def test_invoice_candidates_outside_the_amount_band_are_not_offered() -> None:
    stub = StubClient({})
    LLMResolver(stub).resolve_invoices(
        [make_payment(gross=100_000)], [make_invoice(amount=500_000)]
    )
    assert stub.calls == []


# --- prompt content --------------------------------------------------------
def test_the_bank_prompt_states_the_shortfall_and_the_competition() -> None:
    prompt = _bank_prompt(
        make_batch(expected=97_000),
        [_opaque_bank(credit=96_500)],
        [make_batch("setl_2", settled=date(2026, 2, 6), expected=50_000)],
    )
    assert "96,500" in prompt.replace(",", "96,500") or "965.00" in prompt
    assert "0.52%" in prompt          # the shortfall, computed for the model
    assert "setl_2" in prompt          # the competing batch it must weigh
    assert "no usable UTR" in prompt


def test_the_bank_prompt_flags_a_phantom_duplicate_refund() -> None:
    batch = make_batch(expected=83_000, expected_dedup=90_000, duplicates=True)
    prompt = _bank_prompt(batch, [_opaque_bank(credit=90_000)], [])
    assert "duplicate refund rows collapsed" in prompt


def test_the_invoice_prompt_warns_against_choosing_on_amount_alone() -> None:
    prompt = _invoice_prompt(make_payment(), [make_invoice()])
    assert "not distinguishable on amount alone" in prompt
