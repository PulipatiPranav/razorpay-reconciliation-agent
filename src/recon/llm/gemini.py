"""Gemini implementation of the Layer 3 client.

Provider-swappable by design: :class:`recon.llm.client.LLMClient` is a protocol
with a single method, and :mod:`recon.matcher.layer3` is pure with respect to
it, so a second provider is a new class rather than a change to the matcher.

Why this exists alongside the Anthropic client: Gemini's Flash models have a
free tier that needs no payment details, which is the difference between Layer 3
being demonstrable and being theoretical for someone without a billing account.
The prompts, the schema, the confidence cap, the decline handling and the
transcript format are all identical -- only the transport differs, so results
recorded through either provider are comparable and replay the same way.

Structured output is requested through ``response_format`` with the same JSON
schema handed to Claude, and the reply is still re-validated locally with
pydantic.  A provider that ignores or mangles the schema therefore produces a
typed ``llm_schema_invalid`` exception, exactly as before.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any

from recon.llm.client import _validate
from recon.llm.schemas import LinkDecision
from recon.obs.logging import CallLog, CallRecord, prompt_hash, timed

#: Overridable with GEMINI_MODEL.  The default is a workhorse Flash model with a
#: usable free-tier quota; the newest premium models are metered far tighter
#: (gemini-3.8-flash allows twenty free requests, fewer than one recording run).
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
DEFAULT_MAX_TOKENS = 4096

#: Gemini's free tier allows five requests per minute on the Flash models.  A
#: recording run issues roughly twenty, so without pacing it trips the quota
#: about a third of the way in and silently loses those records.  Self-pacing
#: below the limit avoids most 429s; the retry loop handles the rest.
DEFAULT_REQUESTS_PER_MINUTE = 5.0
MAX_ATTEMPTS = 6
#: Fallback wait when the server does not tell us how long to hold off.
FALLBACK_BACKOFF_S = 20.0

_RETRY_HINT = re.compile(r"retry(?:Delay|\s+in)?[\"':\s]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "RateLimit" in text


def _retry_after_seconds(exc: Exception) -> float:
    """How long the server asked us to wait, if it said."""
    match = _RETRY_HINT.search(str(exc))
    if match:
        try:
            return min(float(match.group(1)) + 1.0, 120.0)
        except ValueError:  # pragma: no cover - defensive
            pass
    return FALLBACK_BACKOFF_S

#: The SDK reads either name; checked only to fail with a useful message.
API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


class GeminiClient:
    """Live Gemini calls with structured output and full call logging."""

    def __init__(
        self,
        log: CallLog,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE,
        sleep: Any = time.sleep,
    ) -> None:
        from google import genai  # imported lazily so offline runs need no SDK

        if not any(os.environ.get(name) for name in API_KEY_VARS):
            raise RuntimeError(
                "No Gemini credentials found. Set GEMINI_API_KEY (get a free key at "
                "https://aistudio.google.com/apikey) and try again."
            )
        self._client = genai.Client()
        self._log = log
        self.model = model
        self.max_tokens = max_tokens
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._sleep = sleep
        self._last_call_at = 0.0

    def _pace(self) -> None:
        """Hold off so the free tier's per-minute quota is never the thing that fails."""
        if self._min_interval <= 0 or self._last_call_at == 0.0:
            return
        waited = time.monotonic() - self._last_call_at
        if waited < self._min_interval:
            self._sleep(self._min_interval - waited)

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
        interaction = None
        attempts = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            self._pace()
            with timed() as elapsed:
                try:
                    interaction = self._client.interactions.create(
                        model=self.model,
                        input=user,
                        system_instruction=system,
                        response_format={
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": schema,
                        },
                        generation_config={"max_output_tokens": self.max_tokens},
                    )
                except Exception as exc:  # noqa: BLE001 - failures become exception rows
                    self._last_call_at = time.monotonic()
                    if _is_rate_limit(exc) and attempt < MAX_ATTEMPTS:
                        # A quota bounce is not a failed decision -- waiting and
                        # retrying is. Dropping the record instead would silently
                        # shrink Layer 3's coverage and flatter the layers below it.
                        self._sleep(_retry_after_seconds(exc))
                        continue
                    record.error = f"{type(exc).__name__}: {exc}"
                    record.latency_ms = elapsed[0]
                    record.metadata["attempts"] = attempt
                    self._log.append(record)
                    return None, record
            self._last_call_at = time.monotonic()
            break

        record.latency_ms = elapsed[0]
        record.metadata["attempts"] = attempts
        if interaction is None:  # pragma: no cover - guarded by the loop above
            record.error = "no response after retries"
            self._log.append(record)
            return None, record
        record.request_id = getattr(interaction, "id", None)
        status = str(getattr(interaction, "status", "") or "")
        record.stop_reason = status

        usage = getattr(interaction, "usage", None)
        if usage is not None:
            record.input_tokens = getattr(usage, "total_input_tokens", 0) or 0
            # Thinking tokens are generated and billed as output; folding them in
            # keeps the cost column comparable with the Anthropic path, where the
            # same tokens arrive inside output_tokens.
            record.output_tokens = (getattr(usage, "total_output_tokens", 0) or 0) + (
                getattr(usage, "total_thought_tokens", 0) or 0
            )
            record.cache_read_tokens = getattr(usage, "total_cached_tokens", 0) or 0

        if status and status != "completed":
            record.error = f"interaction did not complete: {status}"
            self._log.append(record)
            return None, record

        text = getattr(interaction, "output_text", None) or ""
        record.response_text = text
        decision, problem = _validate(text)
        record.schema_valid = problem is None
        if problem:
            record.error = problem
        self._log.append(record)
        return decision, record
