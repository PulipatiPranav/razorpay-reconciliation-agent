"""Identifier-recovery helpers are pure and tested independently."""

from __future__ import annotations

import pytest

from recon.matcher.text import (
    INVOICE_PATTERN,
    damerau_levenshtein,
    extract_utrs,
    find_id_like,
    is_truncation_of,
    normalise_id,
)


@pytest.mark.parametrize(
    ("narration", "expected"),
    [
        ("NEFT-HDFC393427186692-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT", ["HDFC393427186692"]),
        ("IMPS/AXIS999266536197/RAZORPAYSOFT/PAYOUT", ["AXIS999266536197"]),
        ("RTGS-KKBK6539276-RZPY SETTLEMENT-BATCH", ["KKBK6539276"]),
        ("INT.PD SAVINGS INTEREST CREDIT", []),
    ],
)
def test_extract_utrs(narration: str, expected: list[str]) -> None:
    assert extract_utrs(narration) == expected


def test_extract_returns_longest_first() -> None:
    narration = "NEFT-HDFC393427186692-REF-ICIC5792116-END"
    assert extract_utrs(narration)[0] == "HDFC393427186692"


def test_truncation_requires_a_meaningful_prefix() -> None:
    full = "HDFC393427186692"
    assert is_truncation_of("HDFC393427186", full)
    assert not is_truncation_of("HDFC39", full)       # too short to be safe
    assert not is_truncation_of(full, full)           # identical is not truncation
    assert not is_truncation_of("AXIS393427186", full)


def test_adjacent_transposition_is_distance_one() -> None:
    # This is exactly the corruption the generator injects.
    assert damerau_levenshtein("HDFC566880316647", "HDFC566880316467") == 1


def test_unrelated_utrs_exceed_the_cap() -> None:
    assert damerau_levenshtein("HDFC566880316647", "AXIS111111111111") == 3


def test_distance_is_symmetric_and_zero_on_equality() -> None:
    a, b = "HDFC566880316647", "HDFC566880316457"
    assert damerau_levenshtein(a, a) == 0
    assert damerau_levenshtein(a, b) == damerau_levenshtein(b, a)


def test_length_gap_beyond_the_cap_short_circuits() -> None:
    assert damerau_levenshtein("HDFC1", "HDFC123456789012") == 3


def test_normalise_id_folds_case_and_separators() -> None:
    assert normalise_id("ORDER_KEW7H75AXG1GH6") == normalise_id("order_kew7h75axg1gh6")
    assert normalise_id("order_abc") == "orderabc"


def test_invoice_ids_are_recovered_from_free_text() -> None:
    assert find_id_like("INV-2026-00417", INVOICE_PATTERN) == ["INV-2026-00417"]
    assert find_id_like("rcpt_cdvp48ne", INVOICE_PATTERN) == []
