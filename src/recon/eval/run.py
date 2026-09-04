"""Run matchers over a split and render the comparison.

This module is the only place allowed to name the held-out split
(``tests/test_holdout_guard.py`` enforces that), and it is what ``make eval``
drives so that every number in the README is reproducible from scratch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from recon.eval.scoring import ScoreCard, score
from recon.io import load_split
from recon.matcher.baseline import run_baseline, run_id_join_baseline
from recon.matcher.pipeline import Layer3Resolver, run_layered
from recon.matcher.types import LinkType, MatchLayer, ReconResult
from recon.models import GroundTruth, Manifest, SourceBundle
from recon.obs.logging import DEFAULT_LOG_PATH, CallLog

MatcherFn = Callable[[SourceBundle, str], ReconResult]

BASELINES: dict[str, MatcherFn] = {
    "amount+date": lambda sources, split: run_baseline(sources, split, use_date=True),
    "amount only": lambda sources, split: run_baseline(sources, split, use_date=False),
    "identifier join": run_id_join_baseline,
}


def build_resolver(
    mode: str, log: CallLog, transcript: Path = DEFAULT_LOG_PATH
) -> Layer3Resolver | None:
    """Construct Layer 3 in one of three modes.

    ``off``     Layers 1-2 only.
    ``replay``  answers from a committed transcript -- deterministic, free, and
                what ``make eval`` uses so the README numbers reproduce.
    ``live``    real Claude calls; use it to *record* a transcript, not to
                report numbers, since a live model is non-deterministic.
    """
    if mode == "off":
        return None
    from recon.llm.client import AnthropicClient, ReplayClient
    from recon.matcher.layer3 import LLMResolver

    if mode == "replay":
        return LLMResolver(ReplayClient(transcript, log))
    if mode == "live":
        return LLMResolver(AnthropicClient(log))
    raise ValueError(f"unknown llm mode: {mode!r}")


def matchers_for(resolver: Layer3Resolver | None, threshold: float) -> dict[str, MatcherFn]:
    """The full comparison set: three baselines plus the layered matcher."""
    label = "layered L1+L2" if resolver is None else "layered +LLM"
    return dict(BASELINES) | {
        label: lambda sources, split: run_layered(
            sources, split, resolver=resolver, threshold=threshold
        )
    }


def load_truth(directory: Path) -> GroundTruth:
    return GroundTruth.model_validate_json((directory / "ground_truth.json").read_text())


def run_matchers(
    data_root: Path, split: str, matchers: dict[str, MatcherFn]
) -> list[tuple[str, ReconResult, ScoreCard]]:
    directory = data_root / split
    sources = load_split(directory)
    truth = load_truth(directory)
    out = []
    for name, fn in matchers.items():
        result = fn(sources, split)
        out.append((name, result, score(result, truth)))
    return out


def _pct(value: float) -> str:
    """Percentages render without the sign; the column header carries it."""
    return f"{value * 100:.1f}"


def _ci(bounds: tuple[float, float]) -> str:
    return f"[{bounds[0] * 100:.0f}-{bounds[1] * 100:.0f}]"


def render_comparison(console: Console, split: str, scored: list[tuple[str, ScoreCard]]) -> None:
    table = Table(title=f"{split} — matcher comparison")
    table.add_column("matcher", width=15, no_wrap=True)
    table.add_column("bank P %", justify="right", width=8)
    table.add_column("bank R %", justify="right", width=8)
    table.add_column("inv P %", justify="right", width=8)
    table.add_column("inv R %", justify="right", width=8)
    table.add_column("reconciled % [CI]", justify="right", width=19)
    # False positives asserted on payments that have no correct answer.
    table.add_column("FP", justify="right", width=5, no_wrap=True)
    for name, card in scored:
        bank = card.link(LinkType.PAYMENT_TO_BANK)
        invoice = card.link(LinkType.PAYMENT_TO_INVOICE)
        hallucinations = sum(h.falsely_matched for h in card.honesty)
        table.add_row(
            name,
            _pct(bank.precision),
            _pct(bank.recall),
            _pct(invoice.precision),
            _pct(invoice.recall),
            f"{_pct(card.fully_reconciled_rate)} {_ci(card.fully_reconciled_ci)}",
            str(hallucinations),
        )
    console.print(table)


def render_exceptions(console: Console, name: str, card: ScoreCard) -> None:
    table = Table(title=f"{name} — exceptions by reason ({card.split})")
    table.add_column("reason")
    table.add_column("count", justify="right")
    for reason, count in card.exception_reasons.items():
        table.add_row(reason, str(count))
    console.print(table)


def write_reports(out_dir: Path, split: str, scored: list[tuple[str, ScoreCard]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {name: json.loads(card.model_dump_json()) for name, card in scored}
    (out_dir / f"scorecards_{split}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def load_manifest(directory: Path) -> Manifest:
    data = json.loads((directory / "manifest.json").read_text())
    return Manifest.model_validate(data["manifest"])


def layer3_payment_count(result: ReconResult) -> int:
    """Payments whose answer came from the model rather than a rule."""
    return len({m.payment_id for m in result.matches if m.layer is MatchLayer.L3_LLM})
