"""A feature lag in `90_ic_diagnostic` is a number of sessions, not a number of rows.

The options panel is sparse per symbol: a name carries a row only on sessions its selected
contract quoted, and the validation windows the lag sweep reads are separated by training
spans. A shift over a symbol's rows therefore reaches back to whatever row came before, which
is `lag` sessions earlier only when the name quoted on every session in between. The figure
labels its axis in sessions, so the lagged value has to come from exactly that many sessions
back, or be missing.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from case_studies.sp500_options._ic_diagnostics import lagged_by_sessions
from case_studies.utils.coverage import absent_calendar_sessions
from utils.paths import get_case_study_dir

SESSIONS = pl.Series("timestamp", [date(2024, 1, 1) + timedelta(days=d) for d in range(10)])


def _panel() -> pl.DataFrame:
    # "A" quotes every session; "B" skips sessions 2-4, so its row 5 follows its row 1.
    a = [("A", SESSIONS[i], float(i)) for i in range(10)]
    b = [("B", SESSIONS[i], 100.0 + i) for i in (0, 1, 5, 6, 7, 8, 9)]
    return pl.DataFrame(a + b, schema=["symbol", "timestamp", "x"], orient="row")


def _value(frame: pl.DataFrame, symbol: str, session: int) -> float | None:
    return frame.filter((pl.col("symbol") == symbol) & (pl.col("timestamp") == SESSIONS[session]))[
        "lagged"
    ].item()


def test_a_complete_series_is_lagged_by_its_rows() -> None:
    lagged = lagged_by_sessions(_panel(), "x", 2, alias="lagged", sessions=SESSIONS)
    assert _value(lagged, "A", 5) == 3.0
    assert _value(lagged, "A", 1) is None


def test_a_lag_reaching_into_a_gap_is_missing_not_the_row_before_it() -> None:
    lagged = lagged_by_sessions(_panel(), "x", 1, alias="lagged", sessions=SESSIONS)
    # Row 5 of B follows row 1 of B, but session 4 carried no quote for B.
    assert _value(lagged, "B", 5) is None
    assert _value(lagged, "B", 6) == 105.0


def test_a_lag_spanning_a_gap_reads_the_session_that_many_back() -> None:
    lagged = lagged_by_sessions(_panel(), "x", 5, alias="lagged", sessions=SESSIONS)
    # Five sessions before session 6 is session 1, which B quoted: two rows back, not five.
    assert _value(lagged, "B", 6) == 101.0
    assert _value(lagged, "B", 7) is None


def test_the_input_rows_are_kept_one_for_one() -> None:
    panel = _panel()
    lagged = lagged_by_sessions(panel, "x", 3, alias="lagged", sessions=SESSIONS)
    assert lagged.height == panel.height
    assert (
        lagged.select("symbol", "timestamp", "x")
        .sort("symbol", "timestamp")
        .equals(panel.sort("symbol", "timestamp"))
    )


def _feature_artifact() -> Path | None:
    """The checkout's own artifact, else the maintainer's artifact store."""
    relative = Path("features") / "financial.parquet"
    for root in (
        get_case_study_dir("sp500_options", create=False),
        Path(os.environ.get("ML4T_ARTIFACT_ROOT", Path.home() / "Dropbox/ml4t/case-studies"))
        / "sp500_options",
    ):
        if (root / relative).exists():
            return root / relative
    return None


def test_real_options_panel_lags_land_exactly_that_many_sessions_back() -> None:
    path = _feature_artifact()
    if path is None:
        pytest.skip("no sp500_options feature artifact")
    panel = (
        pl.read_parquet(path, columns=["symbol", "timestamp"])
        .unique(["symbol", "timestamp"])
        .with_columns(pl.col("timestamp").alias("source"))
    )
    sessions = panel["timestamp"].unique().sort()
    # The session grid is the panel's own dates, so it has to be the exchange's.
    assert absent_calendar_sessions(sessions.to_list(), calendar="NYSE") == []
    number = sessions.to_frame().with_row_index("n")
    for lag in (5, 63):
        lagged = (
            lagged_by_sessions(panel, "source", lag, alias="lagged", sessions=sessions)
            .drop_nulls("lagged")
            .join(number, on="timestamp")
            .join(number.rename({"timestamp": "lagged", "n": "n_lagged"}), on="lagged")
        )
        assert lagged.height > 0
        off = lagged.filter(pl.col("n") - pl.col("n_lagged") != lag).height
        assert off == 0, f"lag {lag}: {off} of {lagged.height} values are not {lag} sessions back"
