"""Money arithmetic.

Every monetary value inside this codebase is an ``int`` number of paise.  Rupee
decimals only ever exist at the CSV boundary.  This is deliberate: the whole
point of the project is to measure paise-level drift, and float arithmetic
manufactures exactly the kind of sub-paise error we are trying to detect.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Final

Paise = int

_HUNDRED: Final = Decimal(100)
_QUANT: Final = Decimal("0.01")


class MoneyParseError(ValueError):
    """Raised when a rupee string cannot be interpreted as an amount."""


def parse_rupees(value: str | float | int | Decimal) -> Paise:
    """Parse a rupee-denominated CSV value into integer paise.

    Accepts ``"1,234.56"``, ``"1234.56"``, ``"(12.00)"`` (accounting negative),
    ``""``/``None``-ish blanks are rejected -- callers must decide what a blank
    means in their column, since blank-is-zero is a real source of silent
    reconciliation bugs.
    """
    if isinstance(value, str):
        raw = value.strip().replace(",", "").replace("₹", "")
        if not raw:
            raise MoneyParseError("empty amount string")
        negative = raw.startswith("(") and raw.endswith(")")
        if negative:
            raw = raw[1:-1]
        try:
            dec = Decimal(raw)
        except InvalidOperation as exc:  # pragma: no cover - defensive
            raise MoneyParseError(f"cannot parse amount: {value!r}") from exc
        if negative:
            dec = -dec
    elif isinstance(value, Decimal):
        dec = value
    elif isinstance(value, int):
        return value * 100
    else:
        # float input is accepted but routed through str() so that 0.1 + 0.2
        # style representation error is truncated at the decimal boundary.
        dec = Decimal(str(value))
    return int((dec * _HUNDRED).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def format_rupees(paise: Paise) -> str:
    """Render integer paise as a plain rupee decimal string (no separators)."""
    dec = (Decimal(paise) / _HUNDRED).quantize(_QUANT, rounding=ROUND_HALF_UP)
    return f"{dec:.2f}"


def pct_of(paise: Paise, pct: Decimal, *, rounding: str = ROUND_HALF_UP) -> Paise:
    """Percentage of an amount, rounded to whole paise.

    ``pct`` is expressed in percent (``Decimal("2.0")`` means 2%).  Rounding mode
    is a parameter because gateways and ERPs genuinely disagree about it, and
    that disagreement is one of the drift sources we inject.
    """
    result = (Decimal(paise) * pct / _HUNDRED).quantize(Decimal(1), rounding=rounding)
    return int(result)


def net_of(gross: Paise, fee: Paise, tax: Paise, tds: Paise) -> Paise:
    """Merchant credit for a payment: gross less fee, GST on fee, and TDS."""
    return gross - fee - tax - tds


def reconstruct_gross(
    net: Paise, fee_pct: Decimal, gst_pct: Decimal, tds_pct: Decimal
) -> Paise:
    """Invert :func:`net_of` -- recover gross from a net-settled amount.

    net = g - round(g*f) - round(round(g*f)*t) - round(g*d)

    The nested rounding makes this non-invertible in closed form, so we solve
    the continuous approximation and then search a small integer neighbourhood
    for an exact fixed point.  Returns the closest candidate; callers in the
    matching layer must still verify by re-deriving net.
    """
    f = fee_pct / _HUNDRED
    t = gst_pct / _HUNDRED
    d = tds_pct / _HUNDRED
    denom = Decimal(1) - f - (f * t) - d
    if denom <= 0:
        raise ValueError("deduction rates sum to >= 100%")
    approx = int((Decimal(net) / denom).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    best = approx
    best_err = None
    for candidate in range(approx - 4, approx + 5):
        fee = pct_of(candidate, fee_pct)
        tax = pct_of(fee, gst_pct)
        tds = pct_of(candidate, tds_pct)
        err = abs(net_of(candidate, fee, tax, tds) - net)
        if best_err is None or err < best_err:
            best, best_err = candidate, err
        if err == 0:
            break
    return best
