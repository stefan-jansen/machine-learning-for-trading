"""Crypto perpetual features count settlements, and the 8-hour bars have gaps.

The exchange published no bars for some contracts through two outages in 2022, so the row
before a row is sometimes days back. `03_financial_features` takes every per-symbol window over
``["symbol", "_run"]``, where ``_run`` is `contiguous_run` on the bar clock; this checks on the
real bars that the one-bar log return built that way is exactly the return joined from the bar
one bar length earlier, and that the construction over ``symbol`` alone is not.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from case_studies.utils.feature_engineering import contiguous_run

BAR = timedelta(hours=8)


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    from data import load_crypto_perps

    try:
        frame = load_crypto_perps(frequency="8h")
    except Exception as exc:  # the dataset is licensed and absent from CI checkouts
        pytest.skip(f"no crypto perpetual bars: {exc}")
    if frame.is_empty():
        pytest.skip("no crypto perpetual bars")
    return frame.select("symbol", "timestamp", "close").sort("symbol", "timestamp")


def _one_bar_back(bars: pl.DataFrame) -> pl.DataFrame:
    """The log return joined on the timestamp one bar earlier: null where that bar is absent."""
    previous = bars.select(
        "symbol", (pl.col("timestamp") + BAR).alias("timestamp"), pl.col("close").alias("_prev")
    )
    return bars.join(previous, on=["symbol", "timestamp"], how="left").select(
        "symbol", "timestamp", (pl.col("close") / pl.col("_prev")).log().alias("expected")
    )


def _disagreements(built: pl.DataFrame, expected: pl.DataFrame) -> int:
    both = built.join(expected, on=["symbol", "timestamp"], validate="1:1")
    return both.filter(
        ~pl.col("got").eq_missing(pl.col("expected"))
        & ~((pl.col("got") - pl.col("expected")).abs() < 1e-12).fill_null(False)
    ).height


def test_returns_over_the_run_are_one_bar_returns(bars: pl.DataFrame) -> None:
    runs = bars.with_columns(contiguous_run("timestamp", "symbol", size=BAR).alias("_run"))
    if runs.select("symbol", "_run").n_unique() == bars["symbol"].n_unique():
        pytest.skip("these bars carry no gap, so there is nothing for the run to separate")
    expected = _one_bar_back(bars)
    over_symbol = bars.with_columns(pl.col("close").log().diff().over("symbol").alias("got"))
    over_run = runs.with_columns(pl.col("close").log().diff().over("symbol", "_run").alias("got"))

    assert _disagreements(over_symbol, expected) > 0, "a gap no longer shows in the row diff"
    assert _disagreements(over_run, expected) == 0
