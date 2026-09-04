"""Settlement-calendar rules are pure functions and tested in isolation."""

from __future__ import annotations

from datetime import date

import pytest

from recon.calendar_in import (
    add_business_days,
    is_bank_holiday,
    is_business_day,
    is_second_or_fourth_saturday,
    next_business_day,
)


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 1, 3), False),   # 1st Saturday - banks open
        (date(2026, 1, 10), True),   # 2nd Saturday
        (date(2026, 1, 17), False),  # 3rd Saturday
        (date(2026, 1, 24), True),   # 4th Saturday
        (date(2026, 1, 31), False),  # 5th Saturday
        (date(2026, 1, 26), False),  # a Monday, not a Saturday at all
    ],
)
def test_second_and_fourth_saturday(day: date, expected: bool) -> None:
    assert is_second_or_fourth_saturday(day) is expected


def test_sundays_and_gazetted_holidays_are_closed() -> None:
    assert is_bank_holiday(date(2026, 1, 25))  # Sunday
    assert is_bank_holiday(date(2026, 1, 26))  # Republic Day
    assert is_business_day(date(2026, 1, 27))


def test_t_plus_two_across_a_long_weekend() -> None:
    # Friday 23 Jan -> Sat(4th)/Sun/Republic Day are all closed, so T+2 lands
    # on Wednesday 28 Jan: five calendar days of legitimate drift.
    settled = add_business_days(date(2026, 1, 23), 2)
    assert settled == date(2026, 1, 28)
    assert (settled - date(2026, 1, 23)).days == 5


def test_t_plus_two_on_a_clear_week() -> None:
    assert add_business_days(date(2026, 2, 3), 2) == date(2026, 2, 5)


def test_next_business_day_is_strictly_after() -> None:
    assert next_business_day(date(2026, 2, 3)) == date(2026, 2, 4)
    assert next_business_day(date(2026, 3, 3)) == date(2026, 3, 5)  # Holi on the 4th


def test_add_zero_business_days_is_identity() -> None:
    assert add_business_days(date(2026, 1, 25), 0) == date(2026, 1, 25)


def test_negative_count_rejected() -> None:
    with pytest.raises(ValueError):
        add_business_days(date(2026, 1, 5), -1)
