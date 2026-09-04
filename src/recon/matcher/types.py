"""The output contract shared by every matcher.

Both the Phase 2 baseline and the Phase 3 layered matcher emit exactly these
types.  That is deliberate: if the baseline produced a different shape of
output, the comparison between them would be an argument about formats rather
than about matching quality.  It also means the audit view and the scorer are
written once.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LinkType(StrEnum):
    """The two independent questions asked of every payment."""

    PAYMENT_TO_BANK = "payment_to_bank"
    PAYMENT_TO_INVOICE = "payment_to_invoice"


class MatchLayer(StrEnum):
    BASELINE = "baseline"
    L1_EXACT = "l1_exact"
    L2_FUZZY = "l2_fuzzy"
    L3_LLM = "l3_llm"


class SubjectType(StrEnum):
    PAYMENT = "payment"
    BANK_TXN = "bank_txn"
    INVOICE = "invoice"


class ExceptionReason(StrEnum):
    NO_CANDIDATE = "no_candidate"
    AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
    BELOW_CONFIDENCE = "below_confidence"
    NO_ORDER_ID = "no_order_id"
    UNMATCHED_COUNTERPART = "unmatched_counterpart"
    LLM_DECLINED = "llm_declined"
    LLM_SCHEMA_INVALID = "llm_schema_invalid"


def make_match_id(
    link_type: LinkType, payment_id: str, counterpart_ids: list[str], rule: str
) -> str:
    """Deterministic id, so two runs of the same matcher diff cleanly."""
    payload = "|".join([link_type, payment_id, ",".join(sorted(counterpart_ids)), rule])
    return "m_" + hashlib.sha1(payload.encode()).hexdigest()[:12]


class Match(BaseModel):
    """One asserted link, with everything needed to defend it to an auditor."""

    model_config = ConfigDict(frozen=True)

    match_id: str
    link_type: LinkType
    payment_id: str
    counterpart_ids: list[str]
    layer: MatchLayer
    rule: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    source_records: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        link_type: LinkType,
        payment_id: str,
        counterpart_ids: list[str],
        layer: MatchLayer,
        rule: str,
        confidence: float,
        evidence: list[str],
        source_records: dict[str, list[str]] | None = None,
    ) -> Match:
        return cls(
            match_id=make_match_id(link_type, payment_id, counterpart_ids, rule),
            link_type=link_type,
            payment_id=payment_id,
            counterpart_ids=sorted(counterpart_ids),
            layer=layer,
            rule=rule,
            confidence=confidence,
            evidence=evidence,
            source_records=source_records or {},
        )


class ReconException(BaseModel):
    """A record the matcher declined to resolve, and why.

    Exceptions are first-class output, not leftovers: roughly 5% of the corpus
    is genuinely unresolvable and landing those here is the correct answer.
    """

    model_config = ConfigDict(frozen=True)

    exception_id: str
    subject_type: SubjectType
    subject_id: str
    link_type: LinkType | None
    reason: ExceptionReason
    detail: str
    layer_reached: MatchLayer
    candidates_considered: int = 0
    evidence: list[str] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        subject_type: SubjectType,
        subject_id: str,
        link_type: LinkType | None,
        reason: ExceptionReason,
        detail: str,
        layer_reached: MatchLayer,
        candidates_considered: int = 0,
        evidence: list[str] | None = None,
    ) -> ReconException:
        payload = "|".join([subject_type, subject_id, str(link_type), reason])
        return cls(
            exception_id="x_" + hashlib.sha1(payload.encode()).hexdigest()[:12],
            subject_type=subject_type,
            subject_id=subject_id,
            link_type=link_type,
            reason=reason,
            detail=detail,
            layer_reached=layer_reached,
            candidates_considered=candidates_considered,
            evidence=evidence or [],
        )


class ReconResult(BaseModel):
    """Everything one matcher produced for one split."""

    matcher: str
    split: str
    matches: list[Match]
    exceptions: list[ReconException]
    stats: dict[str, float] = Field(default_factory=dict)

    def matches_for(self, link_type: LinkType) -> list[Match]:
        return [m for m in self.matches if m.link_type is link_type]

    def exceptions_for(self, link_type: LinkType) -> list[ReconException]:
        return [x for x in self.exceptions if x.link_type is link_type]
