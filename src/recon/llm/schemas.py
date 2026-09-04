"""Strict response schemas for the Layer 3 model calls.

Two halves, both explicit on purpose:

* ``JSON_SCHEMA`` is handed to the API as ``output_config.format`` so the model
  is constrained at generation time.
* the pydantic model re-validates the returned text independently.

Belt and braces is not redundant here.  The API constraint stops most
malformed output; the local validation is what turns anything that still slips
through into a typed exception (``llm_schema_invalid``) instead of a crash or,
worse, a silently mis-parsed match.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LinkDecision(BaseModel):
    """The model's verdict on one ambiguous link."""

    model_config = ConfigDict(extra="forbid")

    #: ``None`` means "I could not justify any of these", which is a valid and
    #: expected answer.  Roughly 5% of the corpus has no correct counterpart.
    chosen_id: str | None = Field(
        description="Identifier of the chosen counterpart, or null to decline."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=1200)
    #: Populated on the bank leg when the model believes the shortfall between
    #: the batch total and the credit is an unitemised deduction.
    inferred_deduction_paise: int | None = None


def _schema(*, include_deduction: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "chosen_id": {
            "type": ["string", "null"],
            "description": "Identifier of the chosen counterpart, or null to decline.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string", "maxLength": 1200},
    }
    required = ["chosen_id", "confidence", "reasoning"]
    if include_deduction:
        properties["inferred_deduction_paise"] = {
            "type": ["integer", "null"],
            "description": "Shortfall attributed to an unitemised deduction, in paise.",
        }
        required.append("inferred_deduction_paise")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


BANK_DECISION_SCHEMA = _schema(include_deduction=True)
INVOICE_DECISION_SCHEMA = _schema(include_deduction=False)
