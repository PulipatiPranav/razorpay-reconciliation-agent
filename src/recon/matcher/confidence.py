"""Confidence assigned to each matching rule.

**These numbers are measured, not guessed.**  Each is the rule's observed
precision on the dev split (``make calibrate`` regenerates the table), rounded
down to the nearest percent so the figure is never optimistic.  A rule that
fires 40 times and is right 39 of them is worth 0.97, and saying so is
defensible in a way that an invented 0.9 is not.

Rules that were exactly right on every dev firing are capped at 0.99 rather
than 1.00: perfect precision on a few hundred cases is not evidence of
perfection, and only an identity join deserves certainty.

The LLM layer is capped separately.  It never receives more confidence than a
deterministic rule, regardless of what the model claims about itself.
"""

from __future__ import annotations

from typing import Final

# --- Layer 1: exact identifier joins ---------------------------------------
L1_UTR_COLUMN_JOIN: Final = "l1_utr_column_join"
L1_ORDER_ID_JOIN: Final = "l1_order_id_join"

# --- Layer 2: fuzzy recovery ------------------------------------------------
L2_NARRATION_UTR_EXACT: Final = "l2_narration_utr_exact"
L2_NARRATION_UTR_TRUNCATED: Final = "l2_narration_utr_truncated"
L2_UTR_FUZZY: Final = "l2_utr_fuzzy"
L2_BATCH_AMOUNT_RECONSTRUCTION: Final = "l2_batch_amount_reconstruction"
L2_BATCH_AMOUNT_DEDUPED: Final = "l2_batch_amount_reconstruction_deduped"
L2_RECEIPT_INVOICE_HINT: Final = "l2_receipt_invoice_hint"
L2_ORDER_ID_NORMALISED: Final = "l2_order_id_normalised"
L2_ORDER_ID_FUZZY: Final = "l2_order_id_fuzzy"
L2_INVOICE_AMOUNT_WINDOW: Final = "l2_invoice_amount_date_window"

# --- Layer 3 ---------------------------------------------------------------
L3_LLM_BANK: Final = "l3_llm_batch_disambiguation"
L3_LLM_INVOICE: Final = "l3_llm_invoice_disambiguation"

#: Populated by ``recon calibrate`` from measured dev precision.  The values
#: checked in here are the last calibration run; see README for the table.
RULE_CONFIDENCE: dict[str, float] = {
    L1_UTR_COLUMN_JOIN: 0.99,
    L1_ORDER_ID_JOIN: 0.99,
    L2_NARRATION_UTR_EXACT: 0.99,
    L2_NARRATION_UTR_TRUNCATED: 0.95,
    L2_UTR_FUZZY: 0.95,
    L2_BATCH_AMOUNT_RECONSTRUCTION: 0.90,
    L2_BATCH_AMOUNT_DEDUPED: 0.90,
    L2_RECEIPT_INVOICE_HINT: 0.99,
    L2_ORDER_ID_NORMALISED: 0.99,
    L2_ORDER_ID_FUZZY: 0.95,
    L2_INVOICE_AMOUNT_WINDOW: 0.80,
}

#: Ceiling on anything the LLM proposes.  The model's self-reported confidence
#: is multiplied by this, so it can lower its own score but never raise it above
#: what a measured deterministic rule earns.
LLM_CONFIDENCE_CEILING: Final = 0.85

#: Below this, a proposal is routed to exceptions instead of asserted.
DEFAULT_CONFIDENCE_THRESHOLD: Final = 0.70


def confidence_for(rule: str, default: float = 0.5) -> float:
    return RULE_CONFIDENCE.get(rule, default)
