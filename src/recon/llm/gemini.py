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
import uuid
from typing import Any

from recon.llm.client import _validate
from recon.llm.schemas import LinkDecision
from recon.obs.logging import CallLog, CallRecord, prompt_hash, timed

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_MAX_TOKENS = 4096

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
            except Exception as exc:  # noqa: BLE001 - any failure becomes an exception row
                record.error = f"{type(exc).__name__}: {exc}"
                record.latency_ms = elapsed[0]
                self._log.append(record)
                return None, record

        record.latency_ms = elapsed[0]
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
