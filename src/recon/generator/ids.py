"""Razorpay-shaped identifier generation.  Seeded; no global RNG use."""

from __future__ import annotations

import random
import string

_ALNUM = string.ascii_lowercase + string.digits
_BANK_CODES = ["HDFC", "ICIC", "AXIS", "SBIN", "KKBK"]


def _token(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_ALNUM) for _ in range(n))


def payment_id(rng: random.Random) -> str:
    return f"pay_{_token(rng, 14)}"


def order_id(rng: random.Random) -> str:
    return f"order_{_token(rng, 14)}"


def refund_id(rng: random.Random) -> str:
    return f"rfnd_{_token(rng, 14)}"


def settlement_id(rng: random.Random) -> str:
    return f"setl_{_token(rng, 14)}"


def adjustment_id(rng: random.Random) -> str:
    return f"adj_{_token(rng, 14)}"


def utr(rng: random.Random) -> str:
    """16-character UTR: 4-letter bank code + 12 digits, as seen on NEFT credits."""
    code = rng.choice(_BANK_CODES)
    digits = "".join(rng.choice(string.digits) for _ in range(12))
    return f"{code}{digits}"


def bank_txn_id(rng: random.Random) -> str:
    return f"bank_{_token(rng, 12)}"


def bank_ref_no(rng: random.Random) -> str:
    """The bank's own reference column -- deliberately unrelated to the UTR."""
    return "".join(rng.choice(string.digits) for _ in range(9))


def invoice_id(rng: random.Random, seq: int) -> str:
    return f"INV-2026-{seq:05d}"


def po_number(rng: random.Random) -> str:
    return f"PO/{rng.randint(1000, 9999)}/{rng.randint(20, 26)}"


def receipt_code(rng: random.Random) -> str:
    return f"rcpt_{_token(rng, 8)}"
