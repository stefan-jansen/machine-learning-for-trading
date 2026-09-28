"""US equities features count sessions, and a symbol's rows are not always consecutive sessions.

The archive has no security identifier, and a ticker the exchange reassigned continues under the
same symbol after a gap of months or years (WPP, ENTR). A shift over the symbol then returns a
"one-day" return spanning both companies. `03_financial_features` and `04_model_based_features`
take every per-symbol shift and window over the symbol and its `contiguous_run` on the session
counter; this checks on the archive that the one-session return built that way is the return
joined from the previous session, and that the construction over the symbol alone is not.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from case_studies.utils.feature_engineering import contiguous_run

WINDOW = 252


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    from ml4t.diagnostic.splitters.calendar import TradingCalendar

    from data import load_us_equities

    try:
        raw = load_us_equities(start_date="1990-01-01", end_date="2018-03-31")
    except Exception as exc:  # licensed data, absent from most checkouts
        pytest.skip(f"no US equities archive: {exc}")
    raw = raw.select("symbol", pl.col("timestamp").cast(pl.Date), "adj_close")
    dates = raw.select("timestamp").unique().sort("timestamp")
    settling = pl.Series(
        TradingCalendar("NYSE")
        .get_sessions(pd.DatetimeIndex(dates["timestamp"].to_list(), tz="UTC"))
        .to_numpy()
    ).cast(pl.Date)
    sessions = (
        dates.filter(settling == pl.col("timestamp"))
        .with_row_index("session")
        .with_columns(pl.col("session").cast(pl.Int64))
    )
    return (
        raw.join(sessions, on="timestamp")
        .with_columns(contiguous_run("session", "symbol").alias("_run"))
        .sort("symbol", "timestamp")
    )


def _has_gaps(panel: pl.DataFrame) -> None:
    if panel.select("symbol", "_run").n_unique() == panel["symbol"].n_unique():
        pytest.skip("no symbol in this extract misses a session")


def test_one_session_returns_are_one_session_apart(panel: pl.DataFrame) -> None:
    _has_gaps(panel)
    previous = panel.select(
        "symbol", (pl.col("session") + 1).alias("session"), pl.col("adj_close").alias("_prev")
    )
    expected = panel.join(previous, on=["symbol", "session"], how="left").select(
        "symbol", "timestamp", (pl.col("adj_close") / pl.col("_prev") - 1).alias("expected")
    )

    def disagreements(over: list[str]) -> int:
        got = panel.select(
            "symbol",
            "timestamp",
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over(over) - 1).alias("got"),
        ).join(expected, on=["symbol", "timestamp"], validate="1:1")
        return got.filter(
            ~pl.col("got").eq_missing(pl.col("expected"))
            & ~((pl.col("got") - pl.col("expected")).abs() < 1e-12).fill_null(False)
        ).height

    assert disagreements(["symbol"]) > 0, "no symbol's rows skip a session any more"
    assert disagreements(["symbol", "_run"]) == 0


def test_a_year_window_over_the_run_spans_a_year_of_sessions(panel: pl.DataFrame) -> None:
    _has_gaps(panel)

    def spans(over: list[str]) -> pl.Series:
        first = pl.col("session").shift(WINDOW - 1).over(over)
        return panel.select((pl.col("session") - first + 1).alias("s"))["s"].drop_nulls()

    assert (spans(["symbol"]) > WINDOW).any()
    assert (spans(["symbol", "_run"]) == WINDOW).all()
