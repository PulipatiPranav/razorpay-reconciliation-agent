"""String utilities for recovering identifiers from messy bank narrations.

Pure functions, no I/O, unit-tested independently.  Everything here exists to
attack one defect class: ``narration_corrupt``, which accounts for 61 of the 62
bank-leg misses the deterministic baseline makes.
"""

from __future__ import annotations

import re

# A UTR is four letters of bank code followed by twelve digits.  The relaxed
# digit count catches truncated narrations, which is the point.
UTR_PATTERN = re.compile(r"\b([A-Z]{4})(\d{6,12})\b")

UTR_FULL_DIGITS = 12


def extract_utrs(narration: str) -> list[str]:
    """Every UTR-shaped token in a narration, longest first.

    Longest first matters: when a narration contains both a full and a
    truncated form, the full one should be tried as an exact join before
    anything fuzzy is attempted.
    """
    found = [match.group(0) for match in UTR_PATTERN.finditer(narration.upper())]
    unique = list(dict.fromkeys(found))
    return sorted(unique, key=len, reverse=True)


def is_truncation_of(candidate: str, full: str) -> bool:
    """True when ``candidate`` is ``full`` cut short (and not trivially so).

    Requires at least ten characters so that a four-letter bank code plus a
    couple of digits cannot masquerade as a match -- with only ~40 batches in a
    split, a six-character prefix would collide by chance.
    """
    if len(candidate) < 10 or len(candidate) >= len(full):
        return False
    return full.startswith(candidate)


def damerau_levenshtein(left: str, right: str, *, max_distance: int = 2) -> int:
    """Restricted Damerau-Levenshtein distance, capped for speed.

    Damerau rather than plain Levenshtein because the corruption injected into
    the data is an *adjacent transposition*, which Levenshtein scores as two
    substitutions and Damerau scores as one.  Using Levenshtein would force a
    distance-2 threshold, which is loose enough to start matching unrelated
    UTRs.  Returns ``max_distance + 1`` once the cap is exceeded.
    """
    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1

    previous_previous: list[int] = []
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current[j] = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and left_char == right[j - 2]
                and left[i - 2] == right_char
            ):
                current[j] = min(current[j], previous_previous[j - 2] + cost)
        if min(current) > max_distance:
            return max_distance + 1
        previous_previous, previous = previous, current
    return previous[-1]


def normalise_id(value: str) -> str:
    """Case- and separator-insensitive form of an identifier.

    The ERP's ``order_id`` is sometimes upper-cased on export; comparing the
    normalised forms recovers that case without loosening anything else.
    """
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_id_like(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Identifiers of a given shape embedded in free text (e.g. a receipt field)."""
    return list(dict.fromkeys(pattern.findall(text)))


INVOICE_PATTERN = re.compile(r"INV-\d{4}-\d{4,6}")
