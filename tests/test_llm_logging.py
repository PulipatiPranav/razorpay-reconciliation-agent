"""Call logging, cost accounting, and the replay path.

Replay is what lets `make eval` reproduce the README's numbers without a key
and without spending money, so its miss behaviour matters as much as its hits:
a stale transcript must announce itself, never quietly answer the wrong prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

from recon.llm.client import ReplayClient, _validate
from recon.llm.schemas import BANK_DECISION_SCHEMA, INVOICE_DECISION_SCHEMA
from recon.obs.logging import CallLog, CallRecord, load_transcript, prompt_hash


def _record(**kwargs) -> CallRecord:
    base = dict(
        call_id="c1",
        purpose="bank:setl_1",
        model="claude-opus-5",
        prompt_hash="h",
        system="sys",
        user="usr",
        response_text="{}",
    )
    return CallRecord(**(base | kwargs))


# --- cost -------------------------------------------------------------------
def test_cost_uses_published_per_million_rates() -> None:
    record = _record(input_tokens=1_000_000, output_tokens=0)
    assert record.cost_usd == 5.00
    record = _record(input_tokens=0, output_tokens=1_000_000)
    assert record.cost_usd == 25.00


def test_cached_input_is_billed_at_a_tenth() -> None:
    record = _record(input_tokens=0, cache_read_tokens=1_000_000)
    assert record.cost_usd == 0.50


def test_an_unknown_model_costs_zero_rather_than_guessing() -> None:
    assert _record(model="stub", input_tokens=10_000).cost_usd == 0.0


# --- prompt hashing ---------------------------------------------------------
def test_prompt_hash_is_stable_and_sensitive() -> None:
    a = prompt_hash("system", "user", "claude-opus-5")
    assert a == prompt_hash("system", "user", "claude-opus-5")
    assert a != prompt_hash("system", "user!", "claude-opus-5")
    assert a != prompt_hash("system", "user", "claude-sonnet-5")


def test_hash_distinguishes_a_boundary_shift() -> None:
    """"ab"+"c" and "a"+"bc" must not collide."""
    assert prompt_hash("ab", "c", "m") != prompt_hash("a", "bc", "m")


# --- logging ----------------------------------------------------------------
def test_every_call_is_appended_as_one_json_line(tmp_path: Path) -> None:
    log = CallLog(tmp_path / "calls.jsonl")
    log.append(_record(input_tokens=100, output_tokens=50, latency_ms=1234.5))
    log.append(_record(call_id="c2", error="boom"))
    lines = (tmp_path / "calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["latency_ms"] == 1234.5
    assert first["cost_usd"] > 0


def test_the_summary_reports_what_the_readme_needs(tmp_path: Path) -> None:
    log = CallLog(tmp_path / "calls.jsonl")
    log.append(_record(input_tokens=1000, output_tokens=100, latency_ms=1000))
    log.append(_record(call_id="c2", input_tokens=3000, output_tokens=200, latency_ms=3000))
    summary = log.summary()
    assert summary["calls"] == 2
    assert summary["input_tokens"] == 4000
    assert summary["latency_ms_mean"] == 2000.0


# --- schema validation ------------------------------------------------------
def test_valid_json_parses_to_a_decision() -> None:
    decision, problem = _validate('{"chosen_id": "bank_1", "confidence": 0.8, "reasoning": "x"}')
    assert problem is None
    assert decision is not None and decision.chosen_id == "bank_1"


def test_a_null_choice_is_valid_not_an_error() -> None:
    decision, problem = _validate('{"chosen_id": null, "confidence": 0.0, "reasoning": "none fit"}')
    assert problem is None
    assert decision is not None and decision.chosen_id is None


def test_malformed_json_is_reported_not_raised() -> None:
    decision, problem = _validate("this is not json")
    assert decision is None and problem is not None


def test_out_of_range_confidence_is_rejected() -> None:
    decision, problem = _validate('{"chosen_id": "b", "confidence": 5, "reasoning": "x"}')
    assert decision is None and "schema violation" in (problem or "")


def test_unexpected_fields_are_rejected() -> None:
    decision, problem = _validate(
        '{"chosen_id": "b", "confidence": 0.5, "reasoning": "x", "sneaky": 1}'
    )
    assert decision is None and problem is not None


def test_the_api_schemas_forbid_extra_properties() -> None:
    for schema in (BANK_DECISION_SCHEMA, INVOICE_DECISION_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
    assert "inferred_deduction_paise" in BANK_DECISION_SCHEMA["properties"]
    assert "inferred_deduction_paise" not in INVOICE_DECISION_SCHEMA["properties"]


# --- replay -----------------------------------------------------------------
def _transcript(tmp_path: Path, system: str, user: str, response: str) -> Path:
    path = tmp_path / "transcript.jsonl"
    record = _record(
        prompt_hash=prompt_hash(system, user, "claude-opus-5"),
        system=system,
        user=user,
        response_text=response,
        input_tokens=900,
        output_tokens=120,
        latency_ms=2500.0,
    )
    path.write_text(record.to_json() + "\n")
    return path


def test_replay_answers_a_recorded_prompt_exactly(tmp_path: Path) -> None:
    body = '{"chosen_id": "bank_1", "confidence": 0.9, "reasoning": "0.5% short"}'
    path = _transcript(tmp_path, "sys", "usr", body)
    client = ReplayClient(path, CallLog(tmp_path / "out.jsonl"))
    decision, record = client.decide(purpose="p", system="sys", user="usr", schema={})
    assert decision is not None and decision.chosen_id == "bank_1"
    assert record.replayed and record.input_tokens == 900
    assert client.misses == []


def test_replay_announces_a_miss_instead_of_answering_wrongly(tmp_path: Path) -> None:
    body = '{"chosen_id": null, "confidence": 0, "reasoning": ""}'
    path = _transcript(tmp_path, "sys", "usr", body)
    client = ReplayClient(path, CallLog(tmp_path / "out.jsonl"))
    decision, record = client.decide(purpose="p", system="sys", user="CHANGED", schema={})
    assert decision is None
    assert record.error and client.misses


def test_errored_calls_are_not_replayable(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(_record(error="rate limited").to_json() + "\n")
    assert load_transcript(path) == {}


def test_a_missing_transcript_is_empty_not_fatal(tmp_path: Path) -> None:
    assert load_transcript(tmp_path / "nope.jsonl") == {}


# --- replay must not depend on which model answered -------------------------
def test_replay_finds_a_transcript_recorded_by_a_different_model(tmp_path: Path) -> None:
    """Keying replay on the model was a real bug: a Gemini-recorded transcript
    missed every lookup from a client defaulting to the Anthropic model id,
    silently, so Layer 3 looked as though it had declined everything."""
    body = '{"chosen_id": "bank_1", "confidence": 0.9, "reasoning": "fits"}'
    path = tmp_path / "t.jsonl"
    record = _record(
        model="gemini-3.6-flash",
        prompt_hash=prompt_hash("sys", "usr", "gemini-3.6-flash"),
        system="sys",
        user="usr",
        response_text=body,
    )
    path.write_text(record.to_json() + "\n")

    # A replay client using a completely different model id must still hit.
    client = ReplayClient(path, CallLog(tmp_path / "o.jsonl"), model="claude-opus-5")
    decision, replayed = client.decide(purpose="p", system="sys", user="usr", schema={})
    assert decision is not None and decision.chosen_id == "bank_1"
    assert client.misses == []


def test_the_prompt_still_decides_the_key(tmp_path: Path) -> None:
    body = '{"chosen_id": null, "confidence": 0.0, "reasoning": "none"}'
    path = tmp_path / "t.jsonl"
    path.write_text(_record(system="sys", user="usr", response_text=body).to_json() + "\n")
    client = ReplayClient(path, CallLog(tmp_path / "o.jsonl"))
    client.decide(purpose="p", system="sys", user="DIFFERENT", schema={})
    assert client.misses


def test_a_later_successful_record_supersedes_an_earlier_one(tmp_path: Path) -> None:
    """Re-recording a prompt after a quota failure must win."""
    path = tmp_path / "t.jsonl"
    old_body = '{"chosen_id": "old", "confidence": 0.5, "reasoning": "x"}'
    new_body = '{"chosen_id": "new", "confidence": 0.9, "reasoning": "y"}'
    old = _record(system="s", user="u", response_text=old_body)
    new = _record(system="s", user="u", response_text=new_body)
    path.write_text(old.to_json() + "\n" + new.to_json() + "\n")
    client = ReplayClient(path, CallLog(tmp_path / "o.jsonl"))
    decision, _ = client.decide(purpose="p", system="s", user="u", schema={})
    assert decision is not None and decision.chosen_id == "new"


# --- resumable recording ----------------------------------------------------
def test_recording_resumes_instead_of_re_spending_quota(tmp_path: Path) -> None:
    """A free-tier quota can die mid-run; re-running must not re-ask what it knows."""
    from recon.llm.client import ResumingClient, StubClient
    from recon.llm.schemas import LinkDecision

    body = '{"chosen_id": "bank_1", "confidence": 0.9, "reasoning": "fits"}'
    path = tmp_path / "t.jsonl"
    path.write_text(_record(system="s1", user="u1", response_text=body).to_json() + "\n")

    live = StubClient({"p2": LinkDecision(chosen_id="bank_2", confidence=0.7, reasoning="ok")})
    client = ResumingClient(live, path, CallLog(tmp_path / "o.jsonl"))

    reused, _ = client.decide(purpose="p1", system="s1", user="u1", schema={})
    fresh, _ = client.decide(purpose="p2", system="s2", user="u2", schema={})

    assert reused is not None and reused.chosen_id == "bank_1"
    assert fresh is not None and fresh.chosen_id == "bank_2"
    assert client.reused == 1 and client.called == 1
    assert len(live.calls) == 1  # the already-recorded prompt was never re-sent


def test_a_failed_call_is_retried_not_cached_as_a_refusal(tmp_path: Path) -> None:
    from recon.llm.client import ResumingClient, StubClient
    from recon.llm.schemas import LinkDecision

    path = tmp_path / "t.jsonl"
    path.write_text(_record(system="s", user="u", error="429 quota").to_json() + "\n")
    live = StubClient({"p": LinkDecision(chosen_id="bank_1", confidence=0.8, reasoning="ok")})
    client = ResumingClient(live, path, CallLog(tmp_path / "o.jsonl"))
    decision, _ = client.decide(purpose="p", system="s", user="u", schema={})
    assert decision is not None
    assert client.called == 1 and client.reused == 0
