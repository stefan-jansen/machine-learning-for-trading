"""``absent_calendar_sessions`` sees the direction a stray-print filter cannot.

Every panel in the repo is checked one way: each date it carries is asked whether the exchange
held a market, and the dates that fail are dropped. A session the exchange held that the archive
never printed leaves no row to ask about, so that check passes over it in silence - nothing
raises, every query succeeds, and one day's rows are gone. `us_equities_panel`'s single missing
session, 2017-11-08, was found only because two counts of an unrelated quantity came out one
apart (agent-workspace #1050).

The dates here are written down rather than read from the archive, so each test states the shape
it is about - a complete week, a hole in one, a holiday, a stray print - and does not depend on
which data checkout is present.
"""

from __future__ import annotations

import datetime as dt

from utils.data_quality import absent_calendar_sessions

# Monday 2017-11-06 to Friday 2017-11-10. Every one is an NYSE session.
WEEK = [
    dt.date(2017, 11, 6),
    dt.date(2017, 11, 7),
    dt.date(2017, 11, 8),
    dt.date(2017, 11, 9),
    dt.date(2017, 11, 10),
]
WEDNESDAY = dt.date(2017, 11, 8)


def test_a_complete_week_is_absent_nothing() -> None:
    assert absent_calendar_sessions(WEEK, calendar="NYSE") == []


def test_a_session_the_panel_never_printed_is_returned() -> None:
    without = [d for d in WEEK if d != WEDNESDAY]
    assert absent_calendar_sessions(without, calendar="NYSE") == [WEDNESDAY]


def test_a_declared_absence_is_not_returned() -> None:
    without = [d for d in WEEK if d != WEDNESDAY]
    assert absent_calendar_sessions(without, calendar="NYSE", known_absent=[WEDNESDAY]) == []


def test_declaring_one_session_does_not_hide_another() -> None:
    without_two = [d for d in WEEK if d not in (WEDNESDAY, dt.date(2017, 11, 9))]
    assert absent_calendar_sessions(without_two, calendar="NYSE", known_absent=[WEDNESDAY]) == [
        dt.date(2017, 11, 9)
    ]


def test_a_weekend_or_holiday_is_not_a_missing_session() -> None:
    # Thanksgiving 2017 is the 23rd and the exchange was shut; the 24th is a half day it held.
    # Neither the holiday nor the weekend either side of the span may be reported.
    thanksgiving_week = [dt.date(2017, 11, 22), dt.date(2017, 11, 24), dt.date(2017, 11, 27)]
    assert absent_calendar_sessions(thanksgiving_week, calendar="NYSE") == []


def test_only_the_span_between_the_first_and_last_date_is_checked() -> None:
    # Sessions before the panel opens and after it closes are not its to carry. A panel of one
    # day can be missing nothing.
    assert absent_calendar_sessions([WEDNESDAY], calendar="NYSE") == []


def test_an_empty_panel_reports_nothing_rather_than_raising() -> None:
    assert absent_calendar_sessions([], calendar="NYSE") == []


def test_order_and_duplicates_do_not_change_the_answer() -> None:
    without = [d for d in WEEK if d != WEDNESDAY]
    shuffled = list(reversed(without)) + without
    assert absent_calendar_sessions(shuffled, calendar="NYSE") == [WEDNESDAY]


def test_datetimes_are_accepted_as_well_as_dates() -> None:
    # 02_labels carries `timestamp` as a Date and the raw archive as a Datetime; a caller
    # handing over either must get the same answer rather than every session reported missing.
    stamps = [dt.datetime.combine(d, dt.time(0, 0)) for d in WEEK if d != WEDNESDAY]
    assert absent_calendar_sessions(stamps, calendar="NYSE") == [WEDNESDAY]
    assert (
        absent_calendar_sessions(stamps, calendar="NYSE", known_absent=[dt.datetime(2017, 11, 8)])
        == []
    )
