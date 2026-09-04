"""One layer's verdict on one subject, before it becomes a Match or an exception."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from recon.matcher.types import MatchLayer


class Resolution(BaseModel):
    """A proposed link produced by one layer.

    ``subject_id`` is a settlement id on the bank leg and a payment id on the
    invoice leg.  The pipeline turns these into :class:`Match` objects once the
    confidence threshold has been applied, so a layer never decides on its own
    whether its answer is good enough.
    """

    model_config = ConfigDict(frozen=True)

    subject_id: str
    counterpart_ids: list[str]
    layer: MatchLayer
    rule: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
