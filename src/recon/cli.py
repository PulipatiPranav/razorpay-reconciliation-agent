"""Command line entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from recon.generator.config import MessConfig
from recon.generator.generate import generate_universe
from recon.generator.validate import (
    check_defect_coverage,
    check_split_files,
    check_universe,
)
from recon.generator.writers import write_universe
from recon.models import DefectTag

app = typer.Typer(add_completion=False, help="Multi-source reconciliation agent.")
console = Console()


@app.command()
def generate(
    out: Annotated[Path, typer.Option(help="Root output directory.")] = Path("data"),
    seed: Annotated[int, typer.Option(help="RNG seed; fixes the whole universe.")] = 42,
    n: Annotated[int, typer.Option(help="Number of payments to generate.")] = 600,
    dev: Annotated[int, typer.Option(help="Target payments in the dev split.")] = 400,
    disable: Annotated[
        list[str] | None,
        typer.Option(help="Defect tag to switch off; repeatable."),
    ] = None,
) -> None:
    """Generate the dev and held-out datasets plus their ground truth."""
    disabled = frozenset(DefectTag(tag) for tag in (disable or []))
    cfg = MessConfig(n_payments=n, n_dev_payments=dev, seed=seed, disabled=disabled)

    universe = generate_universe(cfg)
    problems = check_universe(universe)
    if problems:
        for line in problems[:20]:
            console.print(f"[red]invariant violated:[/red] {line}")
        raise typer.Exit(1)

    manifests = write_universe(cfg, universe, out)
    problems = check_split_files(out)
    enabled = {t for t in DefectTag if cfg.rate_for(t) > 0}
    problems += check_defect_coverage(out, enabled - cfg.disabled)
    if problems:
        for line in problems[:20]:
            console.print(f"[red]post-write check failed:[/red] {line}")
        raise typer.Exit(1)

    table = Table(title=f"generated  seed={seed}  config={cfg.config_hash()}")
    for column in ("split", "payments", "bundles", "gateway rows", "bank rows", "invoices"):
        table.add_column(column, justify="right" if column != "split" else "left")
    for name, manifest in manifests.items():
        table.add_row(
            name,
            str(manifest.n_payments),
            str(manifest.n_bundles),
            str(manifest.n_gateway_rows),
            str(manifest.n_bank_rows),
            str(manifest.n_invoice_rows),
        )
    console.print(table)
    console.print("[green]all invariants hold[/green]")


@app.command()
def baseline(
    data_root: Annotated[Path, typer.Option(help="Root of the generated data.")] = Path("data"),
    split: Annotated[str, typer.Option(help="dev, holdout, or both.")] = "both",
    reports: Annotated[Path, typer.Option(help="Where to write JSON scorecards.")] = Path(
        "reports"
    ),
) -> None:
    """Run the Phase 2 deterministic baselines and score them against truth."""
    from recon.eval.run import (
        BASELINES,
        render_comparison,
        render_exceptions,
        run_matchers,
        write_reports,
    )

    splits = ["dev", "holdout"] if split == "both" else [split]
    for name in splits:
        scored = run_matchers(data_root, name, BASELINES)
        cards = [(label, card) for label, _, card in scored]
        render_comparison(console, name, cards)
        write_reports(reports, name, cards)
    console.print(f"[green]scorecards written to {reports}/[/green]")
    strongest = scored[-1]
    render_exceptions(console, strongest[0], strongest[2])


@app.command()
def reconcile(
    data_root: Annotated[Path, typer.Option(help="Root of the generated data.")] = Path("data"),
    split: Annotated[str, typer.Option(help="dev, holdout, or both.")] = "both",
    llm: Annotated[str, typer.Option(help="Layer 3 mode: off, replay, or live.")] = "off",
    threshold: Annotated[float, typer.Option(help="Confidence threshold.")] = 0.70,
    transcript: Annotated[Path, typer.Option(help="LLM call transcript.")] = Path(
        "logs/llm_calls.jsonl"
    ),
    reports: Annotated[Path, typer.Option(help="Where to write scorecards.")] = Path("reports"),
) -> None:
    """Run the layered matcher alongside the baselines and score everything."""
    from recon.eval.run import (
        build_resolver,
        matchers_for,
        render_comparison,
        render_exceptions,
        run_matchers,
        write_reports,
    )
    from recon.eval.scoring import ScoreCard
    from recon.matcher.types import ReconResult
    from recon.obs.logging import CallLog

    log = CallLog(transcript if llm == "live" else Path("logs/replay.jsonl"))
    resolver = build_resolver(llm, log, transcript)

    scored: list[tuple[str, ReconResult, ScoreCard]] = []
    for name in ["dev", "holdout"] if split == "both" else [split]:
        scored = run_matchers(data_root, name, matchers_for(resolver, threshold))
        cards = [(label, card) for label, _, card in scored]
        render_comparison(console, name, cards)
        write_reports(reports, name, cards)

    if log.records:
        summary = log.summary()
        console.print(
            f"[cyan]LLM:[/cyan] {int(summary['calls'])} calls, "
            f"{int(summary['input_tokens'])} in / {int(summary['output_tokens'])} out tokens, "
            f"${summary['cost_usd']:.4f}, mean {summary['latency_ms_mean']:.0f} ms, "
            f"{int(summary['schema_failures'])} schema failures, "
            f"{int(summary['errors'])} errors"
        )
    if scored:
        render_exceptions(console, scored[-1][0], scored[-1][2])


@app.command("data-report")
def data_report(
    directory: Annotated[Path, typer.Argument(help="A split directory.")] = Path("data/dev"),
) -> None:
    """Print the defect mix of a generated split."""
    manifest = json.loads((directory / "manifest.json").read_text())["manifest"]
    n, n_bundles = manifest["n_payments"], manifest["n_bundles"]

    payment_table = Table(title=f"{directory} -- payment-level defects (n={n})")
    payment_table.add_column("defect")
    payment_table.add_column("payments", justify="right")
    payment_table.add_column("share", justify="right")
    for key, value in manifest["defect_counts"].items():
        payment_table.add_row(key, str(value), f"{value / n:.1%}")
    console.print(payment_table)

    bundle_table = Table(title=f"{directory} -- batch-level defects (n={n_bundles} batches)")
    bundle_table.add_column("defect")
    bundle_table.add_column("batches", justify="right")
    bundle_table.add_column("payments riding them", justify="right")
    for key, value in manifest["bundle_defect_counts"].items():
        affected = manifest["payments_affected_by_bundle_defect"].get(key, 0)
        bundle_table.add_row(key, f"{value}/{n_bundles}", f"{affected} ({affected / n:.1%})")
    console.print(bundle_table)


if __name__ == "__main__":  # pragma: no cover
    app()
