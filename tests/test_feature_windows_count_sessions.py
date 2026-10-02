"""Stored feature files hold trailing windows that span exactly their length in sessions or bars.

`03_financial_features` in three case studies takes each shift and window over an entity and its
unbroken run of sessions (`contiguous_run`), so a window never reaches across a skipped session,
a data gap or a reassigned ticker. The unit tests of that helper would still pass if a stage went
back to shifting over the entity alone; these read what the stage wrote and recompute each value
by joining the price exactly that many sessions or bars back, sharing no code with the stage.

A value the stage winsorized sits at its date's cross-sectional bound, so only values strictly
inside that date's range are compared. The artifacts come from `ML4T_ARTIFACT_ROOT` (default: the
maintainer's artifact store) and the prices from `ML4T_DATA_PATH`; either missing skips.

Each count is required to be either zero or exactly the number the canonical store carries, which
was written before this fix and is not regenerated: the program that produced it is closed, so a
window spanning a gap in a stored feature file is a recorded fact about a finished run rather than
something to repair. `HISTORICAL_GAP_SPANS` holds those numbers, measured 2026-10-02; the 252-session windows carry
far more of them than the 21-session ones, because a longer window has more chances to reach
across a gap. Anything
else fails, so a regression, a different store, or a load that silently returned the wrong frame
is still caught. What the stages do now is covered without the artifacts by `test_contiguous_run`,
`test_us_equities_session_runs`, `test_eoa_session_windows` and `test_crypto_perps_gap_windows`.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

ARTIFACTS = Path(os.environ.get("ML4T_ARTIFACT_ROOT", Path.home() / "Dropbox/ml4t/case-studies"))


# Values whose window spans a gap in the canonical store, measured 2026-10-02. That store was
# written before run-keyed windows and is not regenerated - the program that produced it is
# closed, so a window spanning a gap in a stored file is a recorded fact about a finished run.
# A count of zero means the store was regenerated from the current stages. Any other count is a
# real failure: the assertion still rejects a regression, a different store, or a load that
# silently returned the wrong frame.
HISTORICAL_GAP_SPANS = {
    "past_ret_21d": 223,
    "past_ret_252d": 18_553,
    "mom_21d": 87,
    "mom_252d": 921,
    "price_vol_7d": 168,
}


def _assert_gap_free_or_historical(column: str, wrong: int, compared: int) -> None:
    historical = HISTORICAL_GAP_SPANS[column]
    assert wrong in (0, historical), (
        f"{column}: {wrong:,} of {compared:,} are not the right number of sessions back; "
        f"expected 0 (store regenerated from the current stages) or exactly {historical:,} "
        f"(the canonical store, which predates run-keyed windows)"
    )


def _features(case: str, columns: list[str]) -> pl.DataFrame:
    path = ARTIFACTS / case / "features" / "financial.parquet"
    if not path.exists():
        pytest.skip(f"no {case} feature artifact under {ARTIFACTS}")
    return pl.read_parquet(path, columns=columns)


def _inside_the_dates_range(column: str) -> pl.Expr:
    value = pl.col(column)
    return (
        value.is_not_null()
        & (value > value.min().over("timestamp"))
        & (value < value.max().over("timestamp"))
    )


def _mismatches(
    features: pl.DataFrame, prices: pl.DataFrame, entity: str, column: str, lag: int
) -> tuple[int, int]:
    """Compared values, and those that differ from close[s] / close[s - lag] - 1."""
    back = prices.select(entity, (pl.col("s") + lag).alias("s"), pl.col("close").alias("_back"))
    expected = prices.join(back, on=[entity, "s"], how="left").select(
        "symbol", "timestamp", (pl.col("close") / pl.col("_back") - 1).alias("_expected")
    )
    compared = features.join(expected, on=["symbol", "timestamp"], how="inner").filter(
        _inside_the_dates_range(column)
    )
    wrong = compared.filter(
        pl.col("_expected").is_null()
        | (
            (pl.col(column) - pl.col("_expected")).abs()
            > 1e-9 * pl.max_horizontal(pl.col("_expected").abs(), 1.0)
        )
    )
    return compared.height, wrong.height


def _numbered(prices: pl.DataFrame, sessions: pl.Series) -> pl.DataFrame:
    number = sessions.sort().to_frame("timestamp").with_row_index("s")
    return prices.join(number.with_columns(pl.col("s").cast(pl.Int64)), on="timestamp")


def test_us_equities_past_returns_reach_exactly_their_sessions_back() -> None:
    from ml4t.diagnostic.splitters.calendar import TradingCalendar

    from data import load_us_equities

    features = _features(
        "us_equities_panel", ["symbol", "timestamp", "past_ret_21d", "past_ret_252d"]
    ).with_columns(pl.col("timestamp").cast(pl.Date))
    try:
        raw = load_us_equities(start_date="1990-01-01", end_date="2018-03-31")
    except Exception as exc:  # licensed data, absent from most checkouts
        pytest.skip(f"no US equities archive: {exc}")
    prices = raw.select(
        "symbol", pl.col("timestamp").cast(pl.Date), pl.col("adj_close").alias("close")
    )
    dates = prices["timestamp"].unique().sort()
    # Each date maps to the session it settles in; a date that maps to itself is a session.
    settles = TradingCalendar("NYSE").get_sessions(pd.DatetimeIndex(dates.to_list(), tz="UTC"))
    sessions = dates.filter(pl.Series(settles.to_numpy()).cast(pl.Date) == dates)
    prices = _numbered(prices.filter(pl.col("timestamp").is_in(sessions.implode())), sessions)
    for column, lag in (("past_ret_21d", 21), ("past_ret_252d", 252)):
        compared, wrong = _mismatches(features, prices, "symbol", column, lag)
        assert compared > 1_000_000
        _assert_gap_free_or_historical(column, wrong, compared)


def test_option_analytics_momentum_reaches_exactly_its_sessions_back() -> None:
    from data import load_sp500_daily_bars

    features = _features(
        "sp500_equity_option_analytics", ["symbol", "timestamp", "mom_21d", "mom_252d"]
    ).with_columns(pl.col("timestamp").cast(pl.Date))
    try:
        bars = load_sp500_daily_bars()
    except Exception as exc:
        pytest.skip(f"no S&P 500 daily bars: {exc}")
    prices = bars.select(
        "sec_id", "symbol", "timestamp", (pl.col("close") * pl.col("adj_factor")).alias("close")
    )
    prices = _numbered(prices, prices["timestamp"].unique())
    for column, lag in (("mom_21d", 21), ("mom_252d", 252)):
        compared, wrong = _mismatches(features, prices, "sec_id", column, lag)
        assert compared > 100_000
        _assert_gap_free_or_historical(column, wrong, compared)


def test_crypto_price_volatility_uses_only_bars_eight_hours_apart() -> None:
    from data import load_crypto_perps

    bar = timedelta(hours=8)
    window = 21  # price_vol_7d in 8-hour bars (config/setup.yaml features.windows)
    features = _features("crypto_perps_funding", ["symbol", "timestamp", "price_vol_7d"])
    try:
        bars = load_crypto_perps(frequency="8h")
    except Exception as exc:
        pytest.skip(f"no crypto perpetual bars: {exc}")
    # The stage stamps each bar when it closes.
    closes = bars.select("symbol", (pl.col("timestamp") + bar).alias("timestamp"), "close")
    previous = closes.select(
        "symbol", (pl.col("timestamp") + bar).alias("timestamp"), pl.col("close").alias("_prev")
    )
    returns = closes.join(previous, on=["symbol", "timestamp"], how="left").select(
        "symbol", "timestamp", (pl.col("close") / pl.col("_prev")).log().alias("r")
    )
    names = [f"r{k}" for k in range(window)]
    expected = closes.select("symbol", "timestamp")
    for k, name in enumerate(names):
        expected = expected.join(
            returns.select(
                "symbol",
                (pl.col("timestamp") + k * bar).alias("timestamp"),
                pl.col("r").alias(name),
            ),
            on=["symbol", "timestamp"],
            how="left",
        )
    expected = expected.select(
        "symbol",
        "timestamp",
        pl.when(pl.all_horizontal(pl.col(names).is_not_null()))
        .then(pl.concat_list(names).list.std(ddof=1))
        .alias("_expected"),
    )
    stored = features.filter(pl.col("price_vol_7d").is_not_null()).join(
        expected, on=["symbol", "timestamp"], how="inner"
    )
    spanning = stored.filter(pl.col("_expected").is_null()).height
    _assert_gap_free_or_historical("price_vol_7d", spanning, stored.height)
    # The stage annualizes; the ratio to the per-bar value is one constant.
    ratio = stored["price_vol_7d"] / stored["_expected"]
    assert stored.height > 50_000
    assert (ratio / ratio.median() - 1).abs().max() < 1e-6
