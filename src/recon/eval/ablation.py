"""Causal ablation: how much does each defect class actually cost us?

The per-defect table in :mod:`recon.eval.breakdown` is observational.  It can
only report the resolution rate among payments carrying a tag, and on this
corpus that is badly confounded: batch-level defects attach to every payment in
their batch, so ``weekend_holiday_drift`` and ``narration_opaque`` co-occur
constantly and the isolated view is left with fewer than five payments for most
classes.

This module answers the question the observational table cannot.  Every defect
class is independently toggleable by construction (that is what the Phase 1
config is for), so for each class we regenerate the corpus with **only that
class switched off**, re-run the matcher, and measure how far the
reconciliation rate moves.  That difference is the defect's causal cost.

Two honest caveats, both handled rather than hidden:

* Switching a defect off perturbs the RNG stream, so the whole corpus differs,
  not only that defect.  Results are therefore **paired per seed** -- each
  ablation is compared against the all-defects-on run of the *same* seed -- and
  averaged over several seeds, with the spread reported so a delta smaller than
  the noise is visible as such.
* Ablations run on the **dev** distribution only.  Regenerating variants of the
  held-out corpus and reading the results would be a form of peeking, however
  indirect.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import replace

from pydantic import BaseModel

from recon.eval.scoring import score
from recon.generator.config import MessConfig
from recon.generator.generate import generate_universe, split_universe
from recon.generator.writers import allocate_extras, materialise_split
from recon.io import build_payment_views, build_refund_views, build_settlement_batches
from recon.matcher.types import ReconResult
from recon.models import DefectTag, GroundTruth, SourceBundle

#: Seeds used for the paired comparison.  Eight keeps the run under a few
#: seconds while making a one-point difference distinguishable from noise.
DEFAULT_SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404, 505, 606, 707)

MatcherFn = Callable[[SourceBundle, str], ReconResult]


class AblationRow(BaseModel):
    tag: str
    seeds: int
    baseline_rate: float
    ablated_rate: float
    #: Percentage points recovered by switching this defect off.  Larger means
    #: the defect costs more.
    delta_points: float
    #: Standard error of the mean across seeds.  Reported rather than the raw
    #: spread because the spread is a property of the sample size, and the
    #: question is whether the mean itself is distinguishable from zero.
    delta_stderr: float
    #: True when the mean is more than two standard errors from zero.  Anything
    #: below that is reported as indistinguishable from reshuffling noise.
    above_noise: bool


def _materialise_dev(cfg: MessConfig) -> tuple[SourceBundle, GroundTruth]:
    """Build the dev split of one config entirely in memory."""
    universe = generate_universe(cfg)
    splits = split_universe(universe)
    extras = allocate_extras(universe, splits)
    gateway, bank, invoices, truth = materialise_split(
        universe, "dev", splits["dev"], extras["dev"].bank, extras["dev"].invoices
    )
    sources = SourceBundle(
        payments=build_payment_views(gateway),
        refunds=build_refund_views(gateway),
        bank_rows=bank,
        invoices=invoices,
        batches=build_settlement_batches(gateway),
    )
    return sources, truth


def _rate(cfg: MessConfig, matcher: MatcherFn, threshold: float) -> float:
    sources, truth = _materialise_dev(cfg)
    card = score(matcher(sources, "dev"), truth, confidence_threshold=threshold)
    return card.fully_reconciled_rate


def run_ablation(
    matcher: MatcherFn,
    *,
    base: MessConfig | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    threshold: float = 0.0,
    tags: list[DefectTag] | None = None,
) -> list[AblationRow]:
    """One row per toggleable defect class, sorted by how much it costs."""
    base = base or MessConfig()
    candidates = tags or [
        tag for tag in DefectTag if MessConfig().rate_for(tag) > 0
    ]

    baselines: dict[int, float] = {}
    for seed in seeds:
        baselines[seed] = _rate(replace(base, seed=seed), matcher, threshold)

    rows: list[AblationRow] = []
    for tag in candidates:
        deltas: list[float] = []
        ablated: list[float] = []
        for seed in seeds:
            cfg = replace(base, seed=seed, disabled=frozenset({tag}))
            rate = _rate(cfg, matcher, threshold)
            ablated.append(rate)
            deltas.append((rate - baselines[seed]) * 100)
        mean_delta = statistics.fmean(deltas)
        stderr = (
            statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        )
        rows.append(
            AblationRow(
                tag=str(tag),
                seeds=len(seeds),
                baseline_rate=statistics.fmean(baselines.values()),
                ablated_rate=statistics.fmean(ablated),
                delta_points=round(mean_delta, 2),
                delta_stderr=round(stderr, 2),
                above_noise=abs(mean_delta) > 2 * stderr,
            )
        )
    rows.sort(key=lambda r: -r.delta_points)
    return rows
