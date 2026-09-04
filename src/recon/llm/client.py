"""Claude client for Layer 3, plus a replay client that needs no network.

Three implementations behind one protocol:

``AnthropicClient``  live calls to the Claude API, every one logged to JSONL.
``ReplayClient``     answers from a committed transcript, keyed by prompt hash.
``StubClient``       a canned answer, for unit tests.

The replay client is what makes ``make eval`` honest.  A live model is
non-deterministic and costs money, so a README claiming reproducible numbers
while calling an API on every run would be claiming something untrue.  Replay
reruns the identical pipeline against the recorded transcript: same matches,
same evidence, same score, no key required.  Change a prompt and the hash
changes, the transcript misses, and the run tells you so rather than quietly
using a stale answer.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import ValidationError

from recon.llm.schemas import LinkDecision
from recon.obs.logging import CallLog, CallRecord, load_transcript, prompt_hash, timed

if TYPE_CHECKING:
    from anthropic.types import MessageParam, OutputConfigParam

DEFAULT_MODEL = "claude-opus-5"
# Thinking tokens are billed and counted as output on this model, so the budget
# has to cover the reasoning *and* the JSON. At 2048 a long deliberation could
# crowd out the answer and truncate it into a schema failure; 4096 leaves room.
# It also bounds the worst case: ~20 calls x 4096 output is well under $2.
DEFAULT_MAX_TOKENS = 4096
DEFAULT_EFFORT = "high"


class LLMClient(Protocol):
    """Anything that can turn a prompt into a validated :class:`LinkDecision`."""

    def decide(
        self, *, purpose: str, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[LinkDecision | None, CallRecord]: ...


def _validate(text: str) -> tuple[LinkDecision | None, str | None]:
    """Parse and validate model output; never raise into the pipeline."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON: {exc}"
    try:
        return LinkDecision.model_validate(payload), None
    except ValidationError as exc:
        return None, f"schema violation: {exc.errors()[:2]}"


class AnthropicClient:
    """Live Claude calls with structured output and full call logging."""

    def __init__(
        self,
        log: CallLog,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        import anthropic  # imported lazily so offline runs never need the package

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # An `ant auth login` profile also works, so this is a warning path
            # rather than a hard failure -- the SDK resolves credentials itself.
            pass
        self._client = anthropic.Anthropic()
        self._log = log
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def decide(
        self, *, purpose: str, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[LinkDecision | None, CallRecord]:
        key = prompt_hash(system, user, self.model)
        record = CallRecord(
            call_id=f"call_{uuid.uuid4().hex[:12]}",
            purpose=purpose,
            model=self.model,
            prompt_hash=key,
            system=system,
            user=user,
            response_text="",
        )
        with timed() as elapsed:
            try:
                messages = [cast("MessageParam", {"role": "user", "content": user})]
                output_config = cast(
                    "OutputConfigParam",
                    {
                        "format": {"type": "json_schema", "schema": schema},
                        "effort": self.effort,
                    },
                )
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=messages,
                    output_config=output_config,
                )
            except Exception as exc:  # noqa: BLE001 - any failure becomes an exception row
                record.error = f"{type(exc).__name__}: {exc}"
                record.latency_ms = elapsed[0]
                self._log.append(record)
                return None, record

        record.latency_ms = elapsed[0]
        record.request_id = getattr(response, "_request_id", None)
        record.stop_reason = response.stop_reason
        usage = response.usage
        record.input_tokens = usage.input_tokens
        record.output_tokens = usage.output_tokens
        record.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        record.cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

        if response.stop_reason == "refusal":
            record.error = "refusal"
            self._log.append(record)
            return None, record

        text = next((b.text for b in response.content if b.type == "text"), "")
        record.response_text = text
        decision, problem = _validate(text)
        record.schema_valid = problem is None
        if problem:
            record.error = problem
        self._log.append(record)
        return decision, record


class ReplayClient:
    """Answer from a recorded transcript.  No network, no key, no cost."""

    def __init__(self, transcript_path: Path, log: CallLog, *, model: str = DEFAULT_MODEL) -> None:
        self._transcript = load_transcript(transcript_path)
        self._log = log
        self.model = model
        self.misses: list[str] = []

    def decide(
        self, *, purpose: str, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[LinkDecision | None, CallRecord]:
        key = prompt_hash(system, user, self.model)
        hit = self._transcript.get(key)
        if hit is None:
            self.misses.append(f"{purpose}:{key}")
            record = CallRecord(
                call_id=f"replay_{key}",
                purpose=purpose,
                model=self.model,
                prompt_hash=key,
                system=system,
                user=user,
                response_text="",
                error="no recorded response for this prompt",
                replayed=True,
            )
            self._log.append(record)
            return None, record

        record = CallRecord(
            call_id=hit.get("call_id", f"replay_{key}"),
            purpose=purpose,
            model=hit.get("model", self.model),
            prompt_hash=key,
            system=system,
            user=user,
            response_text=hit.get("response_text", ""),
            input_tokens=hit.get("input_tokens", 0),
            output_tokens=hit.get("output_tokens", 0),
            cache_read_tokens=hit.get("cache_read_tokens", 0),
            cache_write_tokens=hit.get("cache_write_tokens", 0),
            latency_ms=hit.get("latency_ms", 0.0),
            stop_reason=hit.get("stop_reason"),
            request_id=hit.get("request_id"),
            replayed=True,
        )
        decision, problem = _validate(record.response_text)
        record.schema_valid = problem is None
        if problem:
            record.error = problem
        self._log.append(record)
        return decision, record


class StubClient:
    """Canned decisions for unit tests, keyed by purpose."""

    def __init__(self, decisions: dict[str, LinkDecision | None]) -> None:
        self.decisions = decisions
        self.calls: list[tuple[str, str]] = []

    def decide(
        self, *, purpose: str, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[LinkDecision | None, CallRecord]:
        self.calls.append((purpose, user))
        record = CallRecord(
            call_id=f"stub_{len(self.calls)}",
            purpose=purpose,
            model="stub",
            prompt_hash=prompt_hash(system, user, "stub"),
            system=system,
            user=user,
            response_text="",
            schema_valid=True,
        )
        return self.decisions.get(purpose), record
