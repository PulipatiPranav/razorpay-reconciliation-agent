"""Generate every results table and splice them into the README.

The brief requires that ``make eval`` reproduce every number in the README from
scratch.  Hand-copying figures out of a terminal cannot satisfy that -- the
first time a number moves and the prose does not, the README becomes a claim
nobody checked.  So the numeric sections of the README live inside a generated
block, and ``recon eval --check`` fails when the checked-in block differs from
what a fresh run produces.  Prose outside the markers is hand-written and must
not restate figures; it refers to the tables instead.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from recon.eval.ablation import AblationRow
from recon.eval.breakdown import Breakdown
from recon.eval.scoring import ScoreCard
from recon.matcher.types import LinkType
from recon.models import Manifest

BEGIN = "<!-- BEGIN GENERATED: results -->"
END = "<!-- END GENERATED: results -->"


@dataclass
class ReportContext:
    """Everything the report needs, gathered by the CLI."""

    manifests: dict[str, Manifest]
    scorecards: dict[str, list[tuple[str, ScoreCard]]]
    breakdowns: dict[str, Breakdown]
    ablation: list[AblationRow]
    ablation_seeds: int
    llm_mode: str
    threshold: float


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ci(bounds: tuple[float, float]) -> str:
    return f"{bounds[0] * 100:.0f}–{bounds[1] * 100:.0f}%"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _corpus_section(manifests: dict[str, Manifest]) -> list[str]:
    lines = [
        "### Corpus",
        "",
        "| split | payments | batches | gateway rows | bank rows | invoices |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, manifest in manifests.items():
        lines.append(
            f"| {name} | {manifest.n_payments} | {manifest.n_bundles} | "
            f"{manifest.n_gateway_rows} | {manifest.n_bank_rows} | {manifest.n_invoice_rows} |"
        )
    any_manifest = next(iter(manifests.values()))
    lines += [
        "",
        f"Seed `{any_manifest.seed}`, generator `{any_manifest.generator_version}`, "
        f"config hash `{any_manifest.config_hash}`.",
        "",
    ]
    return lines


def _defect_mix_section(manifests: dict[str, Manifest]) -> list[str]:
    splits = list(manifests)
    lines = [
        "### Defect mix",
        "",
        "Payment-level, as a share of that split's payments:",
        "",
        "| defect | " + " | ".join(splits) + " |",
        "|---" + "|---:" * len(splits) + "|",
    ]
    tags = sorted({t for m in manifests.values() for t in m.defect_counts})
    for tag in tags:
        cells = []
        for split in splits:
            manifest = manifests[split]
            count = manifest.defect_counts.get(tag, 0)
            cells.append(f"{count} ({count / manifest.n_payments:.1%})")
        lines.append(f"| `{tag}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Batch-level, as batches carrying the defect and the payments riding them:",
        "",
        "| defect | " + " | ".join(splits) + " |",
        "|---" + "|---:" * len(splits) + "|",
    ]
    btags = sorted({t for m in manifests.values() for t in m.bundle_defect_counts})
    for tag in btags:
        cells = []
        for split in splits:
            manifest = manifests[split]
            batches = manifest.bundle_defect_counts.get(tag, 0)
            riding = manifest.payments_affected_by_bundle_defect.get(tag, 0)
            cells.append(
                f"{batches}/{manifest.n_bundles} ({riding / manifest.n_payments:.0%} of payments)"
            )
        lines.append(f"| `{tag}` | " + " | ".join(cells) + " |")

    lines += ["", *_distribution_gap(manifests), ""]
    return lines


def _distribution_gap(manifests: dict[str, Manifest]) -> list[str]:
    """State the largest defect-mix gap between splits rather than let it hide.

    The split is stratified on batch *counts*, which does not guarantee equal
    shares of *payments* -- a batch-level defect landing on larger batches
    weighs more.  Whichever way the gap falls, saying so is what lets a reader
    judge the held-out number.
    """
    if len(manifests) != 2:
        return []
    (name_a, a), (name_b, b) = list(manifests.items())
    worst_tag, worst_gap, shares = "", 0.0, (0.0, 0.0)
    for tag in set(a.payments_affected_by_bundle_defect) | set(
        b.payments_affected_by_bundle_defect
    ):
        share_a = a.payments_affected_by_bundle_defect.get(tag, 0) / a.n_payments
        share_b = b.payments_affected_by_bundle_defect.get(tag, 0) / b.n_payments
        if abs(share_a - share_b) > worst_gap:
            worst_tag, worst_gap, shares = tag, abs(share_a - share_b), (share_a, share_b)
    if not worst_tag:
        return []
    harder = name_a if shares[0] > shares[1] else name_b
    return [
        f"Largest batch-level mix gap: `{worst_tag}` rides {shares[0]:.0%} of {name_a} "
        f"payments against {shares[1]:.0%} of {name_b}, making **{harder} the harder "
        "split** on that defect. The split is stratified on batch counts, which does "
        "not equalise payment shares when a defect lands on larger batches.",
    ]


def _comparison_section(scorecards: dict[str, list[tuple[str, ScoreCard]]]) -> list[str]:
    lines = ["### Baseline vs. layered", ""]
    for split, cards in scorecards.items():
        n = cards[0][1].n_payments if cards else 0
        lines += [
            f"**{split}** ({n} payments):",
            "",
            "| matcher | bank P | bank R | invoice P | invoice R "
            "| **fully reconciled** | 95% CI | false matches |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, card in cards:
            bank = card.link(LinkType.PAYMENT_TO_BANK)
            invoice = card.link(LinkType.PAYMENT_TO_INVOICE)
            false_matches = sum(h.falsely_matched for h in card.honesty)
            lines.append(
                f"| {name} | {_pct(bank.precision)} | {_pct(bank.recall)} "
                f"| {_pct(invoice.precision)} | {_pct(invoice.recall)} "
                f"| **{_pct(card.fully_reconciled_rate)}** "
                f"| {_ci(card.fully_reconciled_ci)} | {false_matches} |"
            )
        lines.append("")
    lines += [
        "*Fully reconciled* means both legs simultaneously correct, counting a "
        "correctly empty prediction on an unresolvable payment as correct. "
        "*False matches* are links asserted on payments that have no counterpart "
        "at all — the metric that matters most, and the one precision alone hides.",
        "",
    ]
    return lines


def _layer_section(breakdown: Breakdown) -> list[str]:
    lines = [
        f"### Which layer earned its keep ({breakdown.split})",
        "",
        "| layer | rule | leg | edges asserted | correct | precision |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in breakdown.layers:
        lines.append(
            f"| {row.layer} | `{row.rule}` | {row.link_type} | {row.matches} "
            f"| {row.correct} | {_pct(row.precision)} |"
        )
    lines.append("")
    return lines


def _defect_section(breakdown: Breakdown) -> list[str]:
    lines = [
        f"### Per-defect resolution ({breakdown.split}, observational)",
        "",
        "Resolution rate among payments carrying each defect. This view is "
        "**confounded**: batch-level defects attach to every payment in their "
        "batch, so classes co-occur constantly. The isolated column counts only "
        "payments where the defect is the sole one present, and is empty for most "
        "classes for exactly that reason. The ablation below is the causal answer.",
        "",
        "| defect | leg | carrying it | resolved | rate | isolated n | isolated rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(breakdown.defects, key=lambda r: (r.link_type, r.marginal_rate)):
        isolated = f"{_pct(row.isolated_rate)}" if row.isolated_n else "—"
        lines.append(
            f"| `{row.tag}` | {row.link_type.replace('payment_to_', '')} | {row.marginal_n} "
            f"| {row.marginal_resolved} | {_pct(row.marginal_rate)} "
            f"| {row.isolated_n} | {isolated} |"
        )
    clean = ", ".join(
        f"{leg.replace('payment_to_', '')} {_pct(rate)}"
        for leg, rate in breakdown.clean_rate.items()
    )
    lines += ["", f"Payments carrying no defect at all resolve at: {clean}.", ""]
    return lines


def _ablation_section(rows: list[AblationRow], seeds: int) -> list[str]:
    lines = [
        "### Which defect class actually costs us (causal ablation)",
        "",
        f"Each class is switched off in the generator, the corpus is regenerated, "
        "and the matcher re-run. Paired per seed against an all-defects-on run of "
        f"the same seed, averaged over {seeds} seeds, on the **dev** distribution "
        "only. A positive delta means switching the defect off recovers that many "
        "points, so a larger number means the defect costs more.",
        "",
        "| defect switched off | fully reconciled | delta (points) | ± std. error "
        "| distinguishable from noise |",
        "|---|---:|---:|---:|:--:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.tag}` | {_pct(row.ablated_rate)} | {row.delta_points:+.2f} "
            f"| {row.delta_stderr:.2f} | {'**yes**' if row.above_noise else 'no'} |"
        )
    baseline = rows[0].baseline_rate if rows else 0.0
    lines += [
        "",
        f"All-defects-on baseline across the same seeds: {_pct(baseline)}.",
        "",
    ]
    return lines


def _exception_section(breakdown: Breakdown) -> list[str]:
    lines = [
        f"### The exception list, and whether it was right ({breakdown.split})",
        "",
        "An exception is *correct* when the subject genuinely has no counterpart, "
        "and a *miss* when one existed and was not found. Both belong in the same "
        "table: a queue that hides misses among genuine orphans is not an honest one.",
        "",
        "| reason | subject | raised | correctly raised | misses | precision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in breakdown.exceptions:
        lines.append(
            f"| `{row.reason}` | {row.subject_type} | {row.total} "
            f"| {row.correctly_raised} | {row.missed} | {_pct(row.precision)} |"
        )
    lines.append("")
    return lines


def _honesty_section(scorecards: dict[str, list[tuple[str, ScoreCard]]]) -> list[str]:
    lines = [
        "### Behaviour on records that have no answer",
        "",
        "| split | matcher | leg | unresolvable | correctly excepted | **falsely matched** |",
        "|---|---|---|---:|---:|---:|",
    ]
    for split, cards in scorecards.items():
        name, card = cards[-1]
        for honesty in card.honesty:
            lines.append(
                f"| {split} | {name} | {honesty.link_type.replace('payment_to_', '')} "
                f"| {honesty.unresolvable_payments} | {honesty.routed_to_exceptions} "
                f"| **{honesty.falsely_matched}** |"
            )
    lines.append("")
    return lines


def _cost_section(breakdowns: dict[str, Breakdown], llm_mode: str) -> list[str]:
    lines = ["### Cost and latency per 100 records", ""]
    if llm_mode == "off":
        lines += [
            "Layer 3 was not run (`--llm off`), so no model calls were made and the "
            "marginal cost of the reported numbers is zero. Layers 1–2 process the "
            "whole corpus in well under a second.",
            "",
        ]
        return lines

    costs = [b.cost for b in breakdowns.values() if b.cost is not None]
    answered = sum(c.answered for c in costs)
    attempted = sum(c.llm_calls for c in costs)
    misses = sum(c.transcript_misses for c in costs)

    if attempted and not answered:
        lines += [
            f"**Layer 3 answered nothing.** The pipeline issued {attempted} prompts "
            f"and all {misses} missed the transcript, because no transcript has been "
            "recorded yet — this environment has no API credentials. Every number in "
            "this report is therefore Layers 1–2 only, and the rows below are what "
            "the pipeline *attempted*, not work that was done. Run `make record-llm` "
            "with a key set, then `make eval`, and these become measured figures.",
            "",
        ]

    lines += [
        "| split | records | prompts | answered | misses | tokens in/out "
        "| cost USD | cost/100 | latency/100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, breakdown in breakdowns.items():
        cost = breakdown.cost
        if cost is None:
            continue
        lines.append(
            f"| {split} | {cost.records} | {cost.llm_calls} | {cost.answered} "
            f"| {cost.transcript_misses} "
            f"| {cost.input_tokens}/{cost.output_tokens} "
            f"| ${cost.cost_usd:.4f} | ${cost.cost_usd_per_100_records:.4f} "
            f"| {cost.latency_s_per_100_records}s |"
        )
    if answered:
        lines += [
            "",
            "Token counts, latency and cost are the figures **recorded during the "
            "live run that produced the transcript**, replayed here. The replay "
            "itself makes no calls and costs nothing.",
        ]
    lines.append("")
    return lines


def build_report(context: ReportContext) -> str:
    """The whole generated block, ready to splice between the markers."""
    holdout = context.breakdowns.get("holdout")
    parts: list[str] = [
        "## Results",
        "",
        f"Everything below is generated by `make eval` — commit `{_git_commit()}`, "
        f"confidence threshold {context.threshold:.2f}, Layer 3 mode `{context.llm_mode}`. "
        "Do not edit by hand; `recon eval --check` fails when this block is stale.",
        "",
    ]
    parts += _corpus_section(context.manifests)
    parts += _defect_mix_section(context.manifests)
    parts += _comparison_section(context.scorecards)
    if holdout:
        parts += _layer_section(holdout)
        parts += _defect_section(holdout)
    parts += _ablation_section(context.ablation, context.ablation_seeds)
    if holdout:
        parts += _exception_section(holdout)
    parts += _honesty_section(context.scorecards)
    parts += _cost_section(context.breakdowns, context.llm_mode)
    return "\n".join(parts).rstrip() + "\n"


def splice(readme: Path, block: str) -> tuple[str, bool]:
    """Replace the generated block in the README; report whether it changed."""
    text = readme.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise ValueError(f"{readme} has no generated-results markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    updated = f"{head}{BEGIN}\n\n{block}\n{END}{tail}"
    return updated, updated != text
