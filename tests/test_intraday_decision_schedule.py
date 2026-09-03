"""A declared intraday cadence builds the decision schedule; it does not assume one.

`resolve_rebalance_timestamps` used to return an intraday panel's timestamps unchanged, on
the comment that the data was already at the right granularity. That is a precondition and
nothing checked it. nasdaq100_microstructure did not meet it: its prediction panel carries
every minute, so a declared fifteen-minute cadence produced a decision every minute and the
backtest traded fifteen times more often than its config said.

ml4t/agent-workspace#187.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

# The feature panel starts after the warmup hour the features pay, not at the open, and an
# early close is shorter. Neither length is a multiple of 15 or 60 minutes, which is what
# makes anchoring per session decisive rather than cosmetic.
REGULAR_FIRST, REGULAR_SLOTS = "10:31", 329
EARLY_SLOTS = 149


def _minutes(day: str, first: str, n: int) -> list[datetime]:
    t0 = datetime.fromisoformat(f"{day}T{first}:00")
    return [t0 + timedelta(minutes=i) for i in range(n)]


def _panel() -> pl.Series:
    return pl.Series(
        "ts",
        _minutes("2024-07-01", REGULAR_FIRST, REGULAR_SLOTS)
        + _minutes("2024-07-03", REGULAR_FIRST, EARLY_SLOTS)
        + _minutes("2024-07-05", REGULAR_FIRST, REGULAR_SLOTS),
    )


def test_a_minute_panel_under_a_fifteen_minute_cadence_decides_every_fifteen_minutes() -> None:
    """The declared cadence decides, not the panel's own resolution."""
    schedule = resolve_rebalance_timestamps(_panel(), "15_minute")
    gaps = schedule.diff().drop_nulls().dt.total_seconds().unique().sort().to_list()
    within_session = [g for g in gaps if g < 24 * 3600]
    assert within_session == [900], f"expected only 15-minute gaps inside a session: {gaps}"


def test_the_panel_is_not_the_schedule() -> None:
    """The reading the old branch produced, so a revert to `return ts` fails here."""
    panel = _panel()
    assert resolve_rebalance_timestamps(panel, "15_minute").len() < panel.len() / 10


def test_decisions_land_on_the_clock_in_every_session() -> None:
    """A fifteen-minute cadence decides on the quarter hour, whatever time the panel starts.

    This is the grid `03_financial_features.py` already calls the decision grid
    (`minute % DECISION_MINUTES == 0`). Anchoring on the panel's first row instead would put
    them at 10:31, 10:46, 11:01 for this panel and agree with nothing else in the case study.
    """
    for cadence, seconds in (("5_minute", 300), ("15_minute", 900), ("60_minute", 3600)):
        schedule = resolve_rebalance_timestamps(_panel(), cadence)
        assert schedule.len() > 0, cadence
        offsets = {t.hour * 3600 + t.minute * 60 + t.second for t in schedule.dt.time().to_list()}
        assert all(o % seconds == 0 for o in offsets), cadence


def test_an_early_close_simply_holds_fewer_decisions() -> None:
    """The grid does not move when a session is short; the session just runs out of it.

    A count anchored on the first row would instead put a different time of day at the head
    of every session whose length is not a multiple of the cadence - and neither 329 nor 149
    minutes is a multiple of 5, 15 or 60.
    """
    schedule = resolve_rebalance_timestamps(_panel(), "60_minute")
    per_session = (
        pl.DataFrame({"ts": schedule})
        .with_columns(d=pl.col("ts").dt.date())
        .group_by("d")
        .agg(pl.col("ts").min().dt.time().alias("first"), pl.len())
        .sort("d")
    )
    # The panel opens at 10:31, so the first hour mark is 11:00 in every session.
    assert per_session.get_column("first").cast(pl.String).to_list() == ["11:00:00"] * 3
    # 11:00 through 15:00 in a full session; 11:00 and 12:00 before a 13:00 early close.
    assert per_session.get_column("len").to_list() == [5, 2, 5]


def test_the_sweep_arms_differ_only_in_the_token() -> None:
    """Ch18 sweeps 15/30/60/240 minutes over one panel; each arm must get its own schedule."""
    panel = _panel()
    counts = {
        c: resolve_rebalance_timestamps(panel, c).len()
        for c in ("15_minute", "30_minute", "1_hour", "4_hour")
    }
    assert counts["15_minute"] > counts["30_minute"] > counts["1_hour"] > counts["4_hour"]


def test_a_daily_or_coarser_cadence_is_untouched() -> None:
    """Tokens that name no intraday interval keep the behaviour every other case study has."""
    daily = pl.Series("ts", [datetime(2024, 7, 1, 16, 0) + timedelta(days=i) for i in range(30)])
    for cadence in ("daily_close", "daily_ny_close", "8_hour_funding_aligned"):
        out = resolve_rebalance_timestamps(daily, cadence)
        assert out.len() == daily.len(), cadence


def test_an_eight_hourly_panel_under_its_own_cadence_is_unchanged() -> None:
    """crypto_perps_funding is done, and construction must not move a slot of it.

    Its token parses as an intraday interval, so it now takes the constructed branch. That
    branch is a subset, and on a panel already at the declared spacing the subset is the
    whole panel - including across a gap, since each date anchors on its own first slot.
    """
    stamps = [
        datetime(2024, 7, 1) + timedelta(hours=8 * i)
        for i in range(30)
        # a venue outage removes one funding time
        if i != 11
    ]
    panel = pl.Series("ts", stamps)
    out = resolve_rebalance_timestamps(panel, "8_hour_funding_aligned")
    assert out.to_list() == panel.to_list()


def test_a_stepped_intraday_schedule_does_not_move_when_a_slot_is_missing() -> None:
    """crypto_perps_funding trades every third funding time; an outage must not re-phase it.

    `gather_every(3)` counts rows, so removing one funding time shifts every decision after
    it to a different time of day and leaves it there - which funding time the strategy
    trades then depends on the gaps in the data. Folding the step into the interval resolves
    24 hours on the clock instead, so the outage costs the decisions inside it and nothing
    else. RoboRev job #17879.
    """
    from case_studies.utils.backtest_loaders import resolve_decision_schedule

    complete = [datetime(2024, 7, 1) + timedelta(hours=8 * i) for i in range(30)]
    with_outage = pl.Series("ts", [t for i, t in enumerate(complete) if i != 11])

    chosen = resolve_decision_schedule(with_outage, "8_hour_funding_aligned", 3)
    assert {t.isoformat() for t in chosen.dt.time().to_list()} == {"00:00:00"}

    unaffected = resolve_decision_schedule(pl.Series("ts", complete), "8_hour_funding_aligned", 3)
    assert chosen.to_list() == unaffected.to_list()


def test_a_daily_cadence_still_counts_sessions() -> None:
    """A slot on a daily-or-coarser grid is a session, which is what the step means there."""
    from case_studies.utils.backtest_loaders import resolve_decision_schedule

    daily = pl.Series("ts", [datetime(2024, 7, 1, 16, 0) + timedelta(days=i) for i in range(30)])
    assert (
        resolve_decision_schedule(daily, "daily_close", 3).to_list()
        == daily.gather_every(3).to_list()
    )
