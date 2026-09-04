"""Scoring against ground truth.

Metric choices, and why:

**Edges, not payments, are the unit of precision and recall.** One payment can
legitimately belong to two bank credits (a split settlement), so scoring at
payment level would force an all-or-nothing verdict on 5.7% of the corpus and
hide partial credit.  An edge is one ``(payment, counterpart)`` pair; precision
and recall over edges are the standard, granular, defensible numbers.

**Payment-level exact resolution is reported alongside it.** A finance
controller does not care about half a payment being reconciled, so the
all-or-nothing rate is the number that matters operationally.  Reporting only
the edge metric would flatter the system; reporting only the exact rate would
make it look noisier than it is.

**Hallucination is measured separately.** Roughly 5% of payments have no
counterpart at all.  Confidently matching one of those is not a small precision
loss -- it is the single worst thing a reconciliation agent can do, because it
silently closes a real discrepancy.  It gets its own metric.

**Every rate carries a Wilson interval.**  The held-out set is 230 payments and
15 batches.  Point estimates at that size are not precise, and quoting them
bare would overstate what the evaluation can support.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from recon.matcher.types import (
    ExceptionReason,
    LinkType,
    ReconResult,
    SubjectType,
)
from recon.models import GroundTruth, GroundTruthLink


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because it stays inside [0, 1] and
    stays sane at the extremes -- which matters here, since several rates sit
    at or near 0 and 1 on a small held-out set.
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


class LinkMetrics(BaseModel):
    link_type: str
    gt_edges: int
    predicted_edges: int
    true_positives: int
    precision: float
    precision_ci: tuple[float, float]
    recall: float
    recall_ci: tuple[float, float]
    f1: float
    resolvable_payments: int
    payments_exactly_resolved: int
    exact_resolution_rate: float
    exact_resolution_ci: tuple[float, float]
    match_rate: float


class HonestyMetrics(BaseModel):
    """How the matcher behaves on records that have no correct answer."""

    link_type: str
    unresolvable_payments: int
    routed_to_exceptions: int
    falsely_matched: int
    hallucination_rate: float
    hallucination_ci: tuple[float, float]


class CounterpartMetrics(BaseModel):
    """Bank credits and invoices that no payment should ever claim."""

    subject_type: str
    orphans_in_ground_truth: int
    flagged_unmatched: int
    correctly_flagged: int
    precision: float
    recall: float


class ScoreCard(BaseModel):
    matcher: str
    split: str
    n_payments: int
    # Headline: both legs simultaneously correct, counting a correctly-empty
    # prediction on an unresolvable payment as correct.  This is what a finance
    # controller means by "reconciled" -- half a payment is not reconciled, and
    # correctly refusing to match an unresolvable record is a right answer, not
    # a gap in coverage.
    fully_reconciled: int
    fully_reconciled_rate: float
    fully_reconciled_ci: tuple[float, float]
    links: list[LinkMetrics]
    honesty: list[HonestyMetrics]
    counterparts: list[CounterpartMetrics]
    exception_reasons: dict[str, int]

    def link(self, link_type: LinkType) -> LinkMetrics:
        return next(m for m in self.links if m.link_type == link_type.value)


def _gt_edges(links: list[GroundTruthLink], link_type: LinkType) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for link in links:
        if link_type is LinkType.PAYMENT_TO_BANK:
            edges.update((link.payment_id, txn) for txn in link.bank_txn_ids)
        elif link.invoice_id:
            edges.add((link.payment_id, link.invoice_id))
    return edges


def _gt_sets(links: list[GroundTruthLink], link_type: LinkType) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for link in links:
        if link_type is LinkType.PAYMENT_TO_BANK:
            out[link.payment_id] = set(link.bank_txn_ids)
        else:
            out[link.payment_id] = {link.invoice_id} if link.invoice_id else set()
    return out


def score(
    result: ReconResult, truth: GroundTruth, *, confidence_threshold: float = 0.0
) -> ScoreCard:
    """Score one matcher run against ground truth for the same split."""
    links = truth.links
    n_payments = len(links)
    link_metrics: list[LinkMetrics] = []
    honesty_metrics: list[HonestyMetrics] = []

    for link_type in LinkType:
        gt_edges = _gt_edges(links, link_type)
        gt_sets = _gt_sets(links, link_type)

        predicted_sets: dict[str, set[str]] = {}
        for match in result.matches_for(link_type):
            if match.confidence < confidence_threshold:
                continue
            predicted_sets.setdefault(match.payment_id, set()).update(match.counterpart_ids)
        predicted_edges = {
            (payment_id, counterpart)
            for payment_id, counterparts in predicted_sets.items()
            for counterpart in counterparts
        }

        true_positives = len(predicted_edges & gt_edges)
        precision = _safe_div(true_positives, len(predicted_edges))
        recall = _safe_div(true_positives, len(gt_edges))

        resolvable = [p for p, expected in gt_sets.items() if expected]
        exact = sum(1 for p in resolvable if predicted_sets.get(p, set()) == gt_sets[p])
        matched_any = sum(1 for p in resolvable if predicted_sets.get(p))

        link_metrics.append(
            LinkMetrics(
                link_type=link_type.value,
                gt_edges=len(gt_edges),
                predicted_edges=len(predicted_edges),
                true_positives=true_positives,
                precision=precision,
                precision_ci=wilson_interval(true_positives, len(predicted_edges)),
                recall=recall,
                recall_ci=wilson_interval(true_positives, len(gt_edges)),
                f1=_safe_div(2 * precision * recall, precision + recall),
                resolvable_payments=len(resolvable),
                payments_exactly_resolved=exact,
                exact_resolution_rate=_safe_div(exact, len(resolvable)),
                exact_resolution_ci=wilson_interval(exact, len(resolvable)),
                match_rate=_safe_div(matched_any, len(resolvable)),
            )
        )

        unresolvable = [p for p, expected in gt_sets.items() if not expected]
        excepted_subjects = {
            x.subject_id
            for x in result.exceptions_for(link_type)
            if x.subject_type is SubjectType.PAYMENT
        }
        falsely_matched = sum(1 for p in unresolvable if predicted_sets.get(p))
        honesty_metrics.append(
            HonestyMetrics(
                link_type=link_type.value,
                unresolvable_payments=len(unresolvable),
                routed_to_exceptions=sum(1 for p in unresolvable if p in excepted_subjects),
                falsely_matched=falsely_matched,
                hallucination_rate=_safe_div(falsely_matched, len(unresolvable)),
                hallucination_ci=wilson_interval(falsely_matched, len(unresolvable)),
            )
        )

    counterpart_metrics: list[CounterpartMetrics] = []
    for subject_type, orphan_ids in (
        (SubjectType.BANK_TXN, set(truth.orphan_bank_txn_ids)),
        (SubjectType.INVOICE, set(truth.orphan_invoice_ids)),
    ):
        flagged = {
            x.subject_id
            for x in result.exceptions
            if x.subject_type is subject_type
            and x.reason is ExceptionReason.UNMATCHED_COUNTERPART
        }
        correct = len(flagged & orphan_ids)
        counterpart_metrics.append(
            CounterpartMetrics(
                subject_type=subject_type.value,
                orphans_in_ground_truth=len(orphan_ids),
                flagged_unmatched=len(flagged),
                correctly_flagged=correct,
                precision=_safe_div(correct, len(flagged)),
                recall=_safe_div(correct, len(orphan_ids)),
            )
        )

    per_link_predictions: dict[str, dict[str, set[str]]] = {}
    for link_type in LinkType:
        predicted: dict[str, set[str]] = {}
        for match in result.matches_for(link_type):
            if match.confidence < confidence_threshold:
                continue
            predicted.setdefault(match.payment_id, set()).update(match.counterpart_ids)
        per_link_predictions[link_type.value] = predicted

    fully_reconciled = 0
    for link in links:
        expected_bank = set(link.bank_txn_ids)
        expected_invoice = {link.invoice_id} if link.invoice_id else set()
        got_bank = per_link_predictions[LinkType.PAYMENT_TO_BANK.value].get(link.payment_id, set())
        got_invoice = per_link_predictions[LinkType.PAYMENT_TO_INVOICE.value].get(
            link.payment_id, set()
        )
        if got_bank == expected_bank and got_invoice == expected_invoice:
            fully_reconciled += 1

    reasons: dict[str, int] = {}
    for exception in result.exceptions:
        reasons[exception.reason.value] = reasons.get(exception.reason.value, 0) + 1

    return ScoreCard(
        matcher=result.matcher,
        split=result.split,
        n_payments=n_payments,
        fully_reconciled=fully_reconciled,
        fully_reconciled_rate=_safe_div(fully_reconciled, n_payments),
        fully_reconciled_ci=wilson_interval(fully_reconciled, n_payments),
        links=link_metrics,
        honesty=honesty_metrics,
        counterparts=counterpart_metrics,
        exception_reasons=dict(sorted(reasons.items())),
    )
