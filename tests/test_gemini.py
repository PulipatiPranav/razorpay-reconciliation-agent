"""The Gemini Layer 3 client, exercised against a fake SDK.

Provider parity is the point: the same prompts, the same schema, the same
transcript format, so a run recorded through either provider replays the same
way. These tests pin the request shape and the response handling.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from recon.llm.schemas import BANK_DECISION_SCHEMA
from recon.obs.logging import CallLog, CallRecord


class _Usage:
    total_input_tokens = 900
    total_output_tokens = 120
    total_thought_tokens = 300
    total_cached_tokens = 0


class _Interaction:
    id = "int_fake123"
    status = "completed"
    usage = _Usage()
    output_text = json.dumps(
        {
            "chosen_id": "bank_1",
            "confidence": 0.82,
            "reasoning": "0.5% shortfall consistent with a platform fee",
            "inferred_deduction_paise": 24152,
        }
    )


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a stand-in google.genai and capture the request."""
    captured: dict = {}

    class _Interactions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return captured.get("_response", _Interaction())

    class _Client:
        def __init__(self, *a, **k):
            self.interactions = _Interactions()

    module = types.ModuleType("google.genai")
    module.Client = _Client
    google = types.ModuleType("google")
    google.genai = module
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", module)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return captured


def _client(tmp_path: Path, log: CallLog | None = None):
    from recon.llm.gemini import GeminiClient

    return GeminiClient(log or CallLog(tmp_path / "calls.jsonl"))


def test_missing_credentials_fail_with_a_useful_message(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from recon.llm.gemini import GeminiClient

    with pytest.raises(RuntimeError, match="aistudio.google.com"):
        GeminiClient(CallLog(tmp_path / "c.jsonl"))


def test_the_request_carries_the_schema_and_the_system_prompt(fake_sdk, tmp_path) -> None:
    client = _client(tmp_path)
    client.decide(purpose="bank:setl_1", system="SYS", user="USER", schema=BANK_DECISION_SCHEMA)
    from recon.llm.gemini import DEFAULT_GEMINI_MODEL

    assert fake_sdk["model"] == DEFAULT_GEMINI_MODEL
    assert fake_sdk["input"] == "USER"
    assert fake_sdk["system_instruction"] == "SYS"
    assert fake_sdk["response_format"]["mime_type"] == "application/json"
    assert fake_sdk["response_format"]["schema"] is BANK_DECISION_SCHEMA
    assert fake_sdk["generation_config"]["max_output_tokens"] == 4096


def test_a_valid_reply_is_parsed_and_logged(fake_sdk, tmp_path) -> None:
    log = CallLog(tmp_path / "calls.jsonl")
    decision, record = _client(tmp_path, log).decide(
        purpose="bank:setl_1", system="SYS", user="USER", schema=BANK_DECISION_SCHEMA
    )
    assert decision is not None
    assert decision.chosen_id == "bank_1" and decision.confidence == 0.82
    assert record.schema_valid is True
    assert record.request_id == "int_fake123"
    assert len((tmp_path / "calls.jsonl").read_text().strip().splitlines()) == 1


def test_thinking_tokens_are_billed_as_output(fake_sdk, tmp_path) -> None:
    """Keeps the cost column comparable with the Anthropic path."""
    _, record = _client(tmp_path).decide(
        purpose="p", system="s", user="u", schema=BANK_DECISION_SCHEMA
    )
    assert record.input_tokens == 900
    assert record.output_tokens == 120 + 300
    assert record.cost_usd > 0


def test_an_incomplete_interaction_becomes_an_error_not_a_match(fake_sdk, tmp_path) -> None:
    class _Truncated(_Interaction):
        status = "incomplete"

    fake_sdk["_response"] = _Truncated()
    decision, record = _client(tmp_path).decide(
        purpose="p", system="s", user="u", schema=BANK_DECISION_SCHEMA
    )
    assert decision is None
    assert record.error is not None and "incomplete" in record.error


def test_malformed_output_is_a_schema_failure_not_a_crash(fake_sdk, tmp_path) -> None:
    class _Garbage(_Interaction):
        output_text = "I think it's probably bank_1?"

    fake_sdk["_response"] = _Garbage()
    decision, record = _client(tmp_path).decide(
        purpose="p", system="s", user="u", schema=BANK_DECISION_SCHEMA
    )
    assert decision is None and record.schema_valid is False


def test_a_transport_failure_is_recorded_rather_than_raised(
    fake_sdk, tmp_path, monkeypatch
) -> None:
    import google.genai as genai

    class _Boom:
        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    monkeypatch.setattr(genai.Client, "__init__", lambda self, *a, **k: None)
    client = _client(tmp_path)
    client._client = types.SimpleNamespace(interactions=_Boom())
    decision, record = client.decide(
        purpose="p", system="s", user="u", schema=BANK_DECISION_SCHEMA
    )
    assert decision is None
    assert record.error is not None and "connection reset" in record.error


def test_a_gemini_transcript_replays_like_any_other(fake_sdk, tmp_path) -> None:
    """Provider parity: replay is keyed on the prompt, not the vendor."""
    from recon.llm.client import ReplayClient
    from recon.obs.logging import prompt_hash

    transcript = tmp_path / "t.jsonl"
    record = CallRecord(
        call_id="c1",
        purpose="bank:setl_1",
        model="gemini-3.8-flash",
        prompt_hash=prompt_hash("SYS", "USER", "gemini-3.8-flash"),
        system="SYS",
        user="USER",
        response_text=_Interaction.output_text,
        input_tokens=900,
        output_tokens=420,
    )
    transcript.write_text(record.to_json() + "\n")

    replay = ReplayClient(transcript, CallLog(tmp_path / "out.jsonl"), model="gemini-3.8-flash")
    decision, replayed = replay.decide(purpose="p", system="SYS", user="USER", schema={})
    assert decision is not None and decision.chosen_id == "bank_1"
    assert replayed.replayed and replay.misses == []


# --- free-tier rate limiting ------------------------------------------------
class _RateLimitError(Exception):
    def __str__(self) -> str:
        return (
            "Error code: 429 - Quota exceeded for metric: "
            "generate_content_free_tier_requests, limit: 5. Please retry in 42.3s"
        )


def test_a_quota_bounce_is_retried_not_dropped(fake_sdk, tmp_path) -> None:
    """Losing a record to a 429 would silently shrink Layer 3's coverage."""
    from recon.llm.gemini import GeminiClient

    calls = {"n": 0}

    class _Flaky:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RateLimitError()
            return _Interaction()

    slept: list[float] = []
    client = GeminiClient(
        CallLog(tmp_path / "c.jsonl"), sleep=slept.append, requests_per_minute=0
    )
    client._client = types.SimpleNamespace(interactions=_Flaky())
    decision, record = client.decide(
        purpose="p", system="s", user="u", schema=BANK_DECISION_SCHEMA
    )
    assert decision is not None and decision.chosen_id == "bank_1"
    assert record.error is None
    assert record.metadata["attempts"] == 2
    assert slept and 42 < slept[0] <= 44  # honoured the server's retry hint


def test_the_server_hint_is_preferred_over_the_fallback(fake_sdk, tmp_path) -> None:
    from recon.llm.gemini import FALLBACK_BACKOFF_S, _retry_after_seconds

    assert 42 < _retry_after_seconds(_RateLimitError()) <= 44
    assert _retry_after_seconds(Exception("429 no hint here")) == FALLBACK_BACKOFF_S


def test_persistent_quota_failure_becomes_an_error_record(fake_sdk, tmp_path) -> None:
    from recon.llm.gemini import GeminiClient

    class _AlwaysLimited:
        def create(self, **kwargs):
            raise _RateLimitError()

    client = GeminiClient(
        CallLog(tmp_path / "c.jsonl"), sleep=lambda _: None, requests_per_minute=0
    )
    client._client = types.SimpleNamespace(interactions=_AlwaysLimited())
    decision, record = client.decide(
        purpose="p", system="s", user="u", schema=BANK_DECISION_SCHEMA
    )
    assert decision is None
    assert record.error is not None and "429" in record.error


def test_a_non_quota_error_is_not_retried(fake_sdk, tmp_path) -> None:
    from recon.llm.gemini import GeminiClient

    calls = {"n": 0}

    class _Broken:
        def create(self, **kwargs):
            calls["n"] += 1
            raise ValueError("bad request")

    client = GeminiClient(
        CallLog(tmp_path / "c.jsonl"), sleep=lambda _: None, requests_per_minute=0
    )
    client._client = types.SimpleNamespace(interactions=_Broken())
    _, record = client.decide(purpose="p", system="s", user="u", schema=BANK_DECISION_SCHEMA)
    assert calls["n"] == 1
    assert record.error is not None and "bad request" in record.error


def test_calls_are_paced_below_the_free_tier_limit(fake_sdk, tmp_path) -> None:
    from recon.llm.gemini import GeminiClient

    slept: list[float] = []
    client = GeminiClient(
        CallLog(tmp_path / "c.jsonl"), sleep=slept.append, requests_per_minute=5
    )
    for _ in range(3):
        client.decide(purpose="p", system="s", user="u", schema=BANK_DECISION_SCHEMA)
    # first call is free; the next two wait out the 12s interval
    assert len(slept) == 2
    assert all(10 < s <= 12 for s in slept)
