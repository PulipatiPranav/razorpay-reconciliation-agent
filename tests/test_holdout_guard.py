"""Structural guard against tuning on the held-out set.

The held-out set is only defensible if nothing but the evaluation harness ever
reads it.  This test enforces that as a build rule rather than a promise: the
matching layers cannot reference the held-out directory, and the generator is
the only thing allowed to write it.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "recon"

# Modules permitted to name the held-out split at all.
ALLOWED = {
    "cli.py",
    "generator/generate.py",
    "generator/writers.py",
    "generator/validate.py",
}
ALLOWED_PREFIXES = ("eval/",)

HOLDOUT = re.compile(r"holdout", re.IGNORECASE)


def _relative_modules() -> list[tuple[str, str]]:
    return [
        (str(path.relative_to(SRC)), path.read_text(encoding="utf-8"))
        for path in sorted(SRC.rglob("*.py"))
    ]


def test_only_the_harness_and_generator_reference_the_holdout_split() -> None:
    offenders = []
    for name, text in _relative_modules():
        if name in ALLOWED or name.startswith(ALLOWED_PREFIXES):
            continue
        if HOLDOUT.search(text):
            offenders.append(name)
    assert offenders == [], (
        f"{offenders} reference the held-out split; only the evaluation harness may"
    )


def test_matching_layers_never_read_data_paths_directly() -> None:
    """Matchers take parsed records as arguments; they do not open files.

    Keeping them file-free is what makes them pure and unit-testable, and it
    removes any route by which a matcher could peek at held-out data.
    """
    offenders = []
    for name, text in _relative_modules():
        if not (name.startswith("matcher/") or name.startswith("llm/")):
            continue
        if re.search(r"open\(|read_text\(|Path\(\s*[\"']data", text):
            offenders.append(name)
    assert offenders == [], f"{offenders} touch the filesystem; matchers must stay pure"
