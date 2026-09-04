"""Per-defect, per-layer, per-exception analysis of a matcher run.

The scorecard says *how well* a matcher did.  This module says *where* it did
badly and *what earned its keep*, which is the part that tells you what to
build next.

Three deliberate choices:

**Marginal and isolated views are both reported.**  Defects compound -- a
payment can carry five tags at once -- so "payments carrying tag X" mixes in
the effect of everything co-occurring with X.  The isolated view counts only
payments where X is the *sole* defect, which is the closest this design gets to
a causal read.  Marginal alone is confounded; isolated alone throws away most
of the corpus.  Reporting one without the other would be misleading in
opposite directions, so both appear with their own denominators.

**``bundled_payout`` is excluded from the tag set when deciding isolation.**
It is on 100% of payments, so counting it would leave no payment isolated at
all and the whole view would read as empty.

**Layer precision is measured against ground truth, not asserted.**  A layer
that fires often but is wrong half the time is worse than one that never fires,
and the table has to be able to say so.
"""

from __future__ import annotations

from pydantic import BaseModel

from recon.eval.scoring import wilson_interval
from recon.matcher.types import (
    LinkType,
    ReconResult,
    SubjectType,
)
from recon.models import DefectTag, GroundTruth, GroundTruthLink
from recon.obs.logging import CallLog

#: On every payment, so it cannot discriminate anything.
UNINFORMATIVE_TAGS = frozenset({DefectTag.BUNDLED})


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def informative_tags(link: GroundTruthLink) -> frozenset[str]:
    tags = set(link.defect_tags) | set(link.bundle_defect_tags)
    return frozenset(str(t) for t in tags - UNINFORMATIVE_TAGS)


def _expected(link: GroundTruthLink, link_type: LinkType) -> set[str]:
    if link_type is LinkType.PAYMENT_TO_BANK:
        return set(link.bank_txn_ids)
    return {link.invoice_id} if link.invoice_id else set()


def _predicted(result: ReconResult, link_type: LinkType, threshold: float) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for match in result.matches_for(link_type):
        if match.confidence < threshold:
            continue
        out.setdefault(match.payment_id, set()).update(match.counterpart_ids)
    return out


class DefectRow(BaseModel):
    tag: str
    link_type: str
    marginal_n: int
    marginal_resolved: int
    marginal_rate: float
    isolated_n: int
    isolated_resolved: int
    isolated_rate: float
    isolated_ci: tuple[float, float]
    #: Percentage points lost against payments carrying no defect at all.
    #: ``None`` when no payment carries this tag alone.
    isolated_lift: float | None


class LayerRow(BaseModel):
    layer: str
    rule: str
    link_type: str
    matches: int
    correct: int
    precision: float
    precision_ci: tuple[float, float]


class ExceptionRow(BaseModel):
    """One exception category, split by whether raising it was right."""

    reason: str
    subject_type: str
    total: int
    #: The subject genuinely had no counterpart -- an exception was the correct answer.
    correctly_raised: int
    #: A counterpart existed and the matcher failed to find it.
    missed: int
    precision: float


class CostRow(BaseModel):
    records: int
    #: Prompts the pipeline issued.
    llm_calls: int
    #: Prompts that actually came back with a validated answer.  In replay mode
    #: this is calls minus transcript misses, and the gap matters: a run where
    #: every prompt missed has made no decisions at all, however many "calls"
    #: it counted.
    answered: int
    transcript_misses: int
    calls_per_100_records: float
    records_reaching_layer3: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_usd_per_100_records: float
    latency_ms_mean: float
    latency_s_per_100_records: float
    schema_failures: int
    errors: int
    replayed: bool


class Breakdown(BaseModel):
    matcher: str
    split: str
    clean_rate: dict[str, float]
    defects: list[DefectRow]
    layers: list[LayerRow]
    exceptions: list[ExceptionRow]
    cost: CostRow | None = None


def defect_breakdown(
    result: ReconResult, truth: GroundTruth, *, threshold: float = 0.0
) -> tuple[list[DefectRow], dict[str, float]]:
    rows: list[DefectRow] = []
    clean_rates: dict[str, float] = {}

    for link_type in LinkType:
        predicted = _predicted(result, link_type, threshold)
        resolvable = [
            link for link in truth.links if _expected(link, link_type)
        ]

        def resolved(
            link: GroundTruthLink,
            *,
            predicted: dict[str, set[str]] = predicted,
            link_type: LinkType = link_type,
        ) -> bool:
            # Loop variables bound as defaults: the closure is used only inside
            # this iteration, but leaving them late-bound is a latent bug.
            return predicted.get(link.payment_id, set()) == _expected(link, link_type)

        clean = [link for link in resolvable if not informative_tags(link)]
        clean_rate = _safe_div(sum(1 for link in clean if resolved(link)), len(clean))
        clean_rates[link_type.value] = clean_rate

        all_tags = sorted({tag for link in resolvable for tag in informative_tags(link)})
        for tag in all_tags:
            marginal = [link for link in resolvable if tag in informative_tags(link)]
            isolated = [link for link in resolvable if informative_tags(link) == {tag}]
            isolated_ok = sum(1 for link in isolated if resolved(link))
            isolated_rate = _safe_div(isolated_ok, len(isolated))
            rows.append(
                DefectRow(
                    tag=tag,
                    link_type=link_type.value,
                    marginal_n=len(marginal),
                    marginal_resolved=sum(1 for link in marginal if resolved(link)),
                    marginal_rate=_safe_div(
                        sum(1 for link in marginal if resolved(link)), len(marginal)
                    ),
                    isolated_n=len(isolated),
                    isolated_resolved=isolated_ok,
                    isolated_rate=isolated_rate,
                    isolated_ci=wilson_interval(isolated_ok, len(isolated)),
                    isolated_lift=(
                        (isolated_rate - clean_rate) * 100 if isolated else None
                    ),
                )
            )
    return rows, clean_rates


def layer_breakdown(
    result: ReconResult, truth: GroundTruth, *, threshold: float = 0.0
) -> list[LayerRow]:
    """Precision of every rule that fired, measured against ground truth."""
    expected_by_link = {
        link_type: {link.payment_id: _expected(link, link_type) for link in truth.links}
        for link_type in LinkType
    }
    tally: dict[tuple[str, str, str], list[int]] = {}
    for match in result.matches:
        if match.confidence < threshold:
            continue
        want = expected_by_link[match.link_type].get(match.payment_id, set())
        # A rule is credited only for edges that are actually in ground truth.
        correct = len(set(match.counterpart_ids) & want)
        key = (match.layer.value, match.rule, match.link_type.value)
        bucket = tally.setdefault(key, [0, 0])
        bucket[0] += len(match.counterpart_ids)
        bucket[1] += correct

    rows = [
        LayerRow(
            layer=layer,
            rule=rule,
            link_type=link_type,
            matches=total,
            correct=correct,
            precision=_safe_div(correct, total),
            precision_ci=wilson_interval(correct, total),
        )
        for (layer, rule, link_type), (total, correct) in tally.items()
    ]
    rows.sort(key=lambda r: (r.layer, -r.matches))
    return rows


def exception_breakdown(result: ReconResult, truth: GroundTruth) -> list[ExceptionRow]:
    """Was each exception the right call, or a miss dressed up as one?"""
    by_payment = {link.payment_id: link for link in truth.links}
    orphan_bank = set(truth.orphan_bank_txn_ids)
    orphan_invoices = set(truth.orphan_invoice_ids)

    tally: dict[tuple[str, str], list[int]] = {}
    for exception in result.exceptions:
        key = (exception.reason.value, exception.subject_type.value)
        bucket = tally.setdefault(key, [0, 0])
        bucket[0] += 1

        if exception.subject_type is SubjectType.PAYMENT:
            link = by_payment.get(exception.subject_id)
            if link is None:
                continue
            link_type = exception.link_type or LinkType.PAYMENT_TO_BANK
            if not _expected(link, link_type):
                bucket[1] += 1
        elif exception.subject_type is SubjectType.BANK_TXN:
            if exception.subject_id in orphan_bank:
                bucket[1] += 1
        elif exception.subject_id in orphan_invoices:
            bucket[1] += 1

    rows = [
        ExceptionRow(
            reason=reason,
            subject_type=subject_type,
            total=total,
            correctly_raised=correct,
            missed=total - correct,
            precision=_safe_div(correct, total),
        )
        for (reason, subject_type), (total, correct) in tally.items()
    ]
    rows.sort(key=lambda r: -r.total)
    return rows


def cost_breakdown(log: CallLog, n_records: int, *, layer3_records: int = 0) -> CostRow:
    """Cost and latency normalised to 100 records, as the brief asks for."""
    summary = log.summary()
    scale = 100 / n_records if n_records else 0.0
    misses = sum(
        1 for r in log.records if r.error == "no recorded response for this prompt"
    )
    return CostRow(
        records=n_records,
        llm_calls=int(summary["calls"]),
        answered=int(summary["calls"]) - misses,
        transcript_misses=misses,
        calls_per_100_records=round(summary["calls"] * scale, 2),
        records_reaching_layer3=layer3_records,
        input_tokens=int(summary["input_tokens"]),
        output_tokens=int(summary["output_tokens"]),
        cost_usd=round(summary["cost_usd"], 6),
        cost_usd_per_100_records=round(summary["cost_usd"] * scale, 6),
        latency_ms_mean=summary["latency_ms_mean"],
        latency_s_per_100_records=round(summary["latency_ms_total"] * scale / 1000, 2),
        schema_failures=int(summary["schema_failures"]),
        errors=int(summary["errors"]),
        replayed=all(r.replayed for r in log.records) if log.records else False,
    )


def build_breakdown(
    result: ReconResult,
    truth: GroundTruth,
    *,
    threshold: float = 0.0,
    log: CallLog | None = None,
    layer3_records: int = 0,
) -> Breakdown:
    defects, clean = defect_breakdown(result, truth, threshold=threshold)
    return Breakdown(
        matcher=result.matcher,
        split=result.split,
        clean_rate=clean,
        defects=defects,
        layers=layer_breakdown(result, truth, threshold=threshold),
        exceptions=exception_breakdown(result, truth),
        cost=cost_breakdown(log, len(truth.links), layer3_records=layer3_records)
        if log is not None
        else None,
    )
