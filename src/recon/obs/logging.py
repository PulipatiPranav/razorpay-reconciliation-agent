"""Structured JSONL logging for every LLM call.

Lives outside ``matcher/`` and ``llm/`` on purpose: those packages are held to a
no-filesystem rule by ``tests/test_holdout_guard.py``, which is what guarantees
a matching layer cannot reach for the held-out data.  Observability is a
separate concern, so it gets a separate home.

The log is not only a record.  It is the replay source: ``make eval`` reruns
the whole pipeline against a committed transcript, so every number in the
README reproduces exactly without spending a token or needing a key.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path("logs/llm_calls.jsonl")

# List pricing, USD per million tokens (input, output).  Cached 2026-09-05;
# update alongside the model id if either changes.  Gemini Flash models also
# have a free tier, on which the billed cost is zero -- the figures here are the
# paid-tier rates, so the cost column states what the run *would* cost rather
# than pretending free work is worthless.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-2.5-flash": (0.30, 2.50),
}
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class CallRecord:
    """Everything worth knowing about one model call."""

    call_id: str
    purpose: str
    model: str
    prompt_hash: str
    system: str
    user: str
    response_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None
    request_id: str | None = None
    error: str | None = None
    schema_valid: bool | None = None
    replayed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        rates = PRICE_PER_MTOK.get(self.model)
        if rates is None:
            return 0.0
        input_rate, output_rate = rates
        billable_input = (
            self.input_tokens
            + self.cache_read_tokens * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * CACHE_WRITE_MULTIPLIER
        )
        return (billable_input * input_rate + self.output_tokens * output_rate) / 1_000_000

    def to_json(self) -> str:
        payload = asdict(self)
        payload["cost_usd"] = round(self.cost_usd, 8)
        return json.dumps(payload, sort_keys=True)


def prompt_hash(system: str, user: str, model: str) -> str:
    """Stable key for replay.  Any prompt change invalidates the transcript."""
    digest = hashlib.sha256("\x00".join([model, system, user]).encode()).hexdigest()
    return digest[:20]


class CallLog:
    """Append-only JSONL sink."""

    def __init__(self, path: Path = DEFAULT_LOG_PATH) -> None:
        self.path = path
        self.records: list[CallRecord] = []

    def append(self, record: CallRecord) -> None:
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    # --- aggregates used by the evaluation report --------------------------
    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def summary(self) -> dict[str, float]:
        calls = len(self.records)
        return {
            "calls": float(calls),
            "input_tokens": float(sum(r.input_tokens for r in self.records)),
            "output_tokens": float(sum(r.output_tokens for r in self.records)),
            "cost_usd": round(self.total_cost_usd, 6),
            "latency_ms_total": round(self.total_latency_ms, 1),
            "latency_ms_mean": round(self.total_latency_ms / calls, 1) if calls else 0.0,
            "schema_failures": float(sum(1 for r in self.records if r.schema_valid is False)),
            "errors": float(sum(1 for r in self.records if r.error)),
        }


def load_transcript(path: Path) -> dict[str, dict[str, Any]]:
    """Read a JSONL call log into a prompt-hash -> record map for replay."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("error"):
            out[record["prompt_hash"]] = record
    return out


@contextmanager
def timed() -> Iterator[list[float]]:
    """Wall-clock milliseconds for one call, captured even when it raises."""
    holder: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = (time.perf_counter() - start) * 1000
