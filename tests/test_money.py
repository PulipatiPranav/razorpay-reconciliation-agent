"""Money arithmetic is pure and independently unit-tested."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

import pytest

from recon.money import (
    MoneyParseError,
    format_rupees,
    net_of,
    parse_rupees,
    pct_of,
    reconstruct_gross,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1234.56", 123456),
        ("1,234.56", 123456),
        (" 1234.5 ", 123450),
        ("0.01", 1),
        ("(12.00)", -1200),
        ("₹99.99", 9999),
    ],
)
def test_parse_rupees(text: str, expected: int) -> None:
    assert parse_rupees(text) == expected


def test_parse_rejects_blank() -> None:
    with pytest.raises(MoneyParseError):
        parse_rupees("   ")


@pytest.mark.parametrize("paise", [0, 1, 99, 100, 123456, 99_99_99_999])
def test_format_parse_roundtrip(paise: int) -> None:
    assert parse_rupees(format_rupees(paise)) == paise


def test_no_float_drift_on_a_known_trap() -> None:
    # 0.1 + 0.2 in float is 0.30000000000000004; in paise it is exactly 30.
    assert parse_rupees("0.10") + parse_rupees("0.20") == parse_rupees("0.30")


def test_pct_rounding_modes_differ_where_it_matters() -> None:
    # 2.5 paise: half-up goes to 3, banker's rounding goes to 2.  This one-paisa
    # disagreement is exactly the drift the generator injects.
    assert pct_of(500, Decimal("0.5"), rounding=ROUND_HALF_UP) == 3
    assert pct_of(500, Decimal("0.5"), rounding=ROUND_HALF_EVEN) == 2


def test_net_of_is_plain_subtraction() -> None:
    assert net_of(100_000, 2_000, 360, 1_000) == 96_640


@pytest.mark.parametrize("gross", [45_000, 99_900, 123_457, 2_50_00_000, 1])
@pytest.mark.parametrize(
    ("fee_pct", "tds_pct"), [("0.90", "0"), ("2.36", "1"), ("1.80", "1"), ("2.00", "0")]
)
def test_reconstruct_gross_is_exact(gross: int, fee_pct: str, tds_pct: str) -> None:
    fee = pct_of(gross, Decimal(fee_pct))
    tax = pct_of(fee, Decimal("18"))
    tds = pct_of(gross, Decimal(tds_pct))
    net = net_of(gross, fee, tax, tds)
    assert reconstruct_gross(net, Decimal(fee_pct), Decimal("18"), Decimal(tds_pct)) == gross


def test_reconstruct_gross_rejects_impossible_rates() -> None:
    with pytest.raises(ValueError):
        reconstruct_gross(1000, Decimal("120"), Decimal("18"), Decimal("0"))
