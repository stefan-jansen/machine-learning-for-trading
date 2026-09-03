"""The rebalance step counts slots inside a session, not across the overnight boundary.

`labels.rebalance_step` composes with the declared cadence to decide which slots are traded.
Applied as a single `gather_every` over the whole schedule it counts across sessions, so a
session whose length is not a multiple of the step shifts the phase of every session after it
and the shift accumulates. These tests pin the reading the anchored form produces AND the
reading the unanchored form produces, so a revert fails rather than passing quietly.

ml4t/agent-workspace#187, #1005.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from case_studies.utils.backtest_loaders import apply_rebalance_step


def _session(day: str, minutes: int) -> list[datetime]:
    open_at = datetime.fromisoformat(f"{day}T09:30:00")
    return [open_at + timedelta(minutes=i) for i in range(minutes)]


# A regular NASDAQ session is 390 minutes; an early close is 210. Neither divides by 60.
REGULAR, EARLY = 390, 210


@pytest.fixture
def intraday_schedule() -> pl.Series:
    stamps = _session("2024-07-01", REGULAR) + _session("2024-07-03", EARLY)
    stamps += _session("2024-07-05", REGULAR)
    return pl.Series("ts", stamps)


def test_every_session_decides_at_its_own_open(intraday_schedule: pl.Series) -> None:
    """Each session's first decision is its first slot, on every session, at any step."""
    for step in (5, 15, 60):
        kept = apply_rebalance_step(intraday_schedule, step)
        firsts = (
            pl.DataFrame({"ts": kept})
            .with_columns(d=pl.col("ts").dt.date())
            .group_by("d")
            .agg(pl.col("ts").min())
            .get_column("ts")
            .dt.time()
            .unique()
            .to_list()
        )
        assert firsts == [datetime.fromisoformat("2024-07-01T09:30:00").time()], (
            f"step {step} did not anchor every session at its open: {firsts}"
        )


def test_an_hourly_step_would_drift_without_anchoring(intraday_schedule: pl.Series) -> None:
    """The reading the unanchored form produces, so a revert to gather_every fails here.

    390 % 60 == 30, so a global gather enters the second session 30 minutes after its open.
    The third session opens on time again, because 390 + 210 is a whole number of hours - the
    phase error cancels here and would not on a different sequence of session lengths. That is
    the reason this is worth a test rather than an inspection: the defect is present on every
    early close and visible only on some of them.
    """
    unanchored = intraday_schedule.gather_every(60)
    drifted = (
        pl.DataFrame({"ts": unanchored})
        .with_columns(d=pl.col("ts").dt.date())
        .group_by("d")
        .agg(pl.col("ts").min())
        .sort("ts")
        .get_column("ts")
        .dt.time()
        .to_list()
    )
    assert [t.isoformat() for t in drifted] == ["09:30:00", "10:00:00", "09:30:00"]

    anchored = apply_rebalance_step(intraday_schedule, 60)
    assert anchored.len() > unanchored.len()
    assert set(unanchored.to_list()) - set(anchored.to_list())


def test_five_and_fifteen_minute_steps_are_whole_sessions(intraday_schedule: pl.Series) -> None:
    """Both divide 390 and 210, so anchoring keeps the count a reader would compute by hand."""
    assert apply_rebalance_step(intraday_schedule, 5).len() == (
        REGULAR // 5 + EARLY // 5 + REGULAR // 5
    )
    assert apply_rebalance_step(intraday_schedule, 15).len() == (
        REGULAR // 15 + EARLY // 15 + REGULAR // 15
    )


def test_a_daily_schedule_keeps_the_identity_it_already_has() -> None:
    """At most one slot per date is not intraday: the step counts sessions, as it always did."""
    daily = pl.Series(
        "ts", [datetime.fromisoformat("2024-07-01T16:00:00") + timedelta(days=i) for i in range(40)]
    )
    for step in (1, 2, 3, 5):
        assert apply_rebalance_step(daily, step).to_list() == daily.gather_every(step).to_list()


def test_step_one_and_an_empty_schedule_are_returned_unchanged(
    intraday_schedule: pl.Series,
) -> None:
    assert apply_rebalance_step(intraday_schedule, 1).to_list() == intraday_schedule.to_list()
    empty = pl.Series("ts", [], dtype=pl.Datetime("us"))
    assert apply_rebalance_step(empty, 15).is_empty()


# ---------------------------------------------------------------------------
# The step is part of the identity it decides (ml4t/agent-workspace#1005)
# ---------------------------------------------------------------------------


def test_a_declared_step_reaches_the_registered_spec() -> None:
    """A run at a different step must hash differently, or the corrected run is skipped.

    The step decides which slots are traded and therefore every metric recorded, so two runs
    of one configuration at different steps are two different runs. Before the step entered
    `strategy.rebalance` they hashed identically and the second was dropped as already
    registered, keeping the first run's numbers under a spec that did not name the parameter
    that produced them.
    """
    from case_studies.utils.backtest_loaders import (
        declared_rebalance_step,
        get_backtest_config,
        load_backtest_prices,
    )
    from case_studies.utils.backtest_presets import build_backtest_spec

    case_study, label = "etfs", "fwd_ret_21d"
    step = declared_rebalance_step(case_study, label)
    assert step is not None, f"{case_study}/{label} declares no step; pick one that does"

    bt = get_backtest_config(case_study)
    prices = load_backtest_prices(case_study, max_symbols=2)
    spec = build_backtest_spec(
        case_study,
        bt,
        signal={"method": "equal_weight_top_k", "top_k": 10, "long_short": False},
        prices=prices,
        prediction_hash="pred123",
        initial_cash=1_000_000.0,
        label=label,
    )
    assert spec["strategy"]["rebalance"]["step"] == step


def test_an_undeclared_step_leaves_the_spec_byte_identical() -> None:
    """A case study that declares no step for a label keeps the identity it already has.

    The same rule `cadence_for` follows: the key appears only where the parameter is
    load-bearing, so nothing already registered is orphaned by this change.
    """
    from case_studies.utils.backtest_loaders import declared_rebalance_step

    assert declared_rebalance_step("etfs", "a_label_no_case_study_declares") is None
