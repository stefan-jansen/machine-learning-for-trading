"""Focused invariants for the S&P 500 options feasibility RV diagnostic."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from case_studies.sp500_options._underlying_returns import reconcile_underlying_log_returns


def _realized_volatility(prices: pl.DataFrame, window: int = 3) -> pl.DataFrame:
    return reconcile_underlying_log_returns(prices).with_columns(
        pl.col("clean_log_return")
        .rolling_std(window, min_samples=window)
        .over(["symbol", "sec_id"])
        .alias("rv")
    )


def _panel() -> pl.DataFrame:
    start = date(2020, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(10)],
            "symbol": ["AAPL"] * 10,
            "sec_id": [1] * 5 + [2] * 5,
            "close": [100.0, 101.0, 102.0, 25.75, 25.50, 130.0, 131.0, 132.0, 133.0, 134.0],
            "adj_factor": [1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def _paired_round_trip_costs(quotes: pl.DataFrame) -> pl.DataFrame:
    pair_keys = ["timestamp", "symbol", "strike", "expiration"]
    return (
        quotes.with_columns(
            ((pl.col("ask") + pl.col("bid")) / 2).alias("mid"),
            (pl.col("ask") - pl.col("bid")).alias("full_spread"),
        )
        .group_by(pair_keys)
        .agg(
            pl.len().alias("n_legs"),
            pl.col("call_put").n_unique().alias("n_leg_types"),
            pl.col("full_spread").sum().alias("round_trip_dollars"),
            pl.col("mid").sum().alias("straddle_mid"),
        )
        .filter((pl.col("n_legs") == 2) & (pl.col("n_leg_types") == 2))
        .with_columns(
            (pl.col("round_trip_dollars") / pl.col("straddle_mid") * 100).alias(
                "round_trip_pct_of_mid"
            )
        )
    )


def test_identity_boundary_restarts_realized_volatility_warmup() -> None:
    result = _realized_volatility(_panel())
    new_security = result.filter(pl.col("sec_id") == 2)

    assert new_security["clean_log_return"][0] is None
    assert new_security.head(3)["rv"].null_count() == 3
    assert new_security["rv"][3] is not None


def test_true_split_remains_continuous_within_security() -> None:
    result = _realized_volatility(_panel()).filter(pl.col("sec_id") == 1)
    split_row = result.filter(pl.col("timestamp") == date(2020, 1, 4))

    assert abs(split_row["clean_log_return"].item()) < 0.02
    assert split_row["rv"].item() is not None


def test_segment_scale_does_not_change_returns_or_volatility() -> None:
    baseline = _realized_volatility(_panel())
    scaled = _realized_volatility(
        _panel().with_columns(
            pl.when(pl.col("sec_id") == 2)
            .then(pl.col("close") * 11)
            .otherwise(pl.col("close"))
            .alias("close")
        )
    )

    for column in ("clean_log_return", "rv"):
        left = baseline.filter(pl.col("sec_id") == 2)[column].drop_nulls()
        right = scaled.filter(pl.col("sec_id") == 2)[column].drop_nulls()
        assert np.allclose(left, right, rtol=0, atol=1e-12)


def test_future_perturbation_does_not_change_realized_volatility_prefix() -> None:
    prices = _panel()
    cutoff = date(2020, 1, 8)
    baseline = _realized_volatility(prices)
    perturbed = _realized_volatility(
        prices.with_columns(
            pl.when(pl.col("timestamp") > cutoff)
            .then(pl.col("close") * pl.lit(7.0))
            .otherwise(pl.col("close"))
            .alias("close")
        )
    )
    columns = ["timestamp", "symbol", "sec_id", "clean_log_return", "rv"]

    assert (
        baseline.filter(pl.col("timestamp") <= cutoff)
        .select(columns)
        .equals(perturbed.filter(pl.col("timestamp") <= cutoff).select(columns))
    )


def test_round_trip_cost_uses_complete_paired_call_put_quotes() -> None:
    quotes = pl.DataFrame(
        {
            "timestamp": [date(2020, 1, 2)] * 5,
            "symbol": ["PAIR", "PAIR", "INCOMPLETE", "DUPLICATE", "DUPLICATE"],
            "strike": [100.0] * 5,
            "expiration": [date(2020, 2, 21)] * 5,
            "call_put": ["C", "P", "C", "C", "C"],
            "bid": [9.0, 4.0, 2.0, 5.0, 5.0],
            "ask": [11.0, 6.0, 3.0, 7.0, 7.0],
        }
    )

    costs = _paired_round_trip_costs(quotes)
    call_half_spread = (
        quotes.head(1)
        .select(
            ((pl.col("ask") - pl.col("bid")) / (pl.col("ask") + pl.col("bid")) * 100).alias(
                "half_spread_pct"
            )
        )["half_spread_pct"]
        .item()
    )

    assert call_half_spread == 10.0
    assert costs.height == 1
    assert costs["round_trip_dollars"].item() == 4.0
    assert costs["straddle_mid"].item() == 15.0
    assert costs["round_trip_pct_of_mid"].item() == 100 * 4 / 15
