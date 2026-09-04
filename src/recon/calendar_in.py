"""Indian bank-settlement calendar.

Settlement drift is not random: it is the deterministic consequence of the RBI
working-day rule.  Banks are closed on Sundays, on the **second and fourth
Saturday** of every month, and on gazetted holidays.  A T+2 settlement captured
on a Thursday before a long weekend lands four calendar days later, and that is
exactly the kind of legitimate disagreement the matcher has to tolerate.

The 2026 holiday list below is illustrative rather than authoritative -- state
holidays vary by region and lunar dates shift.  It is fixed and seeded so that
generated data is reproducible; the README says so plainly.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Final

# Gazetted / RBI holidays used by the generator.  Illustrative for 2026.
BANK_HOLIDAYS_2026: Final[frozenset[date]] = frozenset(
    {
        date(2026, 1, 26),  # Republic Day
        date(2026, 3, 4),   # Holi
        date(2026, 3, 20),  # Id-ul-Fitr
        date(2026, 4, 1),   # Bank annual closing
        date(2026, 4, 3),   # Good Friday
        date(2026, 4, 14),  # Ambedkar Jayanti
        date(2026, 5, 1),   # Maharashtra Day
        date(2026, 5, 27),  # Bakrid
        date(2026, 8, 15),  # Independence Day
        date(2026, 10, 2),  # Gandhi Jayanti
        date(2026, 11, 8),  # Diwali
        date(2026, 12, 25),  # Christmas
    }
)


def is_second_or_fourth_saturday(day: date) -> bool:
    """RBI rule: banks close on the 2nd and 4th Saturday of each month."""
    if day.weekday() != 5:  # 5 == Saturday
        return False
    occurrence = (day.day - 1) // 7 + 1
    return occurrence in (2, 4)


def is_bank_holiday(day: date) -> bool:
    """True when banks do not settle on ``day``."""
    if day.weekday() == 6:  # Sunday
        return True
    if is_second_or_fourth_saturday(day):
        return True
    return day in BANK_HOLIDAYS_2026


def is_business_day(day: date) -> bool:
    return not is_bank_holiday(day)


def next_business_day(day: date) -> date:
    """First settlement day strictly after ``day``."""
    cursor = day + timedelta(days=1)
    while is_bank_holiday(cursor):
        cursor += timedelta(days=1)
    return cursor


def add_business_days(day: date, count: int) -> date:
    """Add ``count`` settlement days to ``day`` (count must be >= 0)."""
    if count < 0:
        raise ValueError("count must be non-negative")
    cursor = day
    for _ in range(count):
        cursor = next_business_day(cursor)
    return cursor


def drift_days(captured: date, settled: date) -> int:
    """Calendar days between capture and settlement -- the observable offset."""
    return (settled - captured).days
