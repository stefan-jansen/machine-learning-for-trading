"""Option-analytics windows count sessions, and a security can miss one.

`sp500_equity_option_analytics/03_financial_features` puts every trailing window on the
sessions a security traded. A security that skipped a session would have its row-counted
windows span one more session than their name, so each window is taken over the security and
its `contiguous_run` on the panel's session counter. On the shipped S&P 500 bars this checks
that such a window always covers exactly its length in sessions, and that the construction
over the security alone does not.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.utils.coverage import absent_calendar_sessions
from case_studies.utils.feature_engineering import contiguous_run

WINDOW = 21


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    from data import load_sp500_daily_bars

    try:
        frame = load_sp500_daily_bars()
    except Exception as exc:
        pytest.skip(f"no S&P 500 daily bars: {exc}")
    sessions = (
        frame.select("timestamp")
        .unique()
        .sort("timestamp")
        .with_row_index("session")
        .with_columns(pl.col("session").cast(pl.Int64))
    )
    return (
        frame.select("sec_id", "timestamp")
        .join(sessions, on="timestamp")
        .sort("sec_id", "timestamp")
    )


def _span(frame: pl.DataFrame, over: list[str]) -> pl.Series:
    """Sessions covered by each full WINDOW-row window, first to last inclusive."""
    first = pl.col("session").shift(WINDOW - 1).over(over)
    return frame.select((pl.col("session") - first + 1).alias("span"))["span"].drop_nulls()


def test_the_panel_dates_are_the_exchange_sessions(bars: pl.DataFrame) -> None:
    assert absent_calendar_sessions(bars["timestamp"].unique().to_list(), calendar="NYSE") == []


def test_a_window_over_the_run_spans_exactly_its_length(bars: pl.DataFrame) -> None:
    runs = bars.with_columns(contiguous_run("session", "sec_id").alias("_run"))
    if runs.select("sec_id", "_run").n_unique() == bars["sec_id"].n_unique():
        pytest.skip("no security in these bars misses a session")

    assert (_span(bars, ["sec_id"]) > WINDOW).any(), "a missed session no longer shows"
    assert (_span(runs, ["sec_id", "_run"]) == WINDOW).all()
