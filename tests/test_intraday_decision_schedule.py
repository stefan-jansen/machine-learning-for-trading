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


def test_every_session_decides_first_at_its_own_first_slot() -> None:
    """Anchored on the session, so the first decision is not a different lag each day."""
    for cadence in ("5_minute", "15_minute", "60_minute"):
        schedule = resolve_rebalance_timestamps(_panel(), cadence)
        firsts = (
            pl.DataFrame({"ts": schedule})
            .with_columns(d=pl.col("ts").dt.date())
            .group_by("d")
            .agg(pl.col("ts").min())
            .get_column("ts")
            .dt.time()
            .unique()
            .to_list()
        )
        assert [t.isoformat() for t in firsts] == [f"{REGULAR_FIRST}:00"], cadence


def test_an_hourly_cadence_does_not_walk_across_the_close() -> None:
    """329 and 149 are multiples of no cadence here, which is what a global count would break."""
    schedule = resolve_rebalance_timestamps(_panel(), "60_minute")
    per_session = (
        pl.DataFrame({"ts": schedule})
        .with_columns(d=pl.col("ts").dt.date())
        .group_by("d")
        .len()
        .sort("d")
        .get_column("len")
        .to_list()
    )
    # ceil(329/60) = 6 decisions in a regular session, ceil(149/60) = 3 in the early close.
    assert per_session == [6, 3, 6]


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
