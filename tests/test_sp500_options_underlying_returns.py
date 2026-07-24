"""Tests for security-identity-aware S&P 500 returns."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from case_studies.sp500_options._underlying_returns import (
    reconcile_underlying_log_returns,
    validate_reconciled_returns,
)
from data import load_sp500_daily_bars, load_sp500_options_straddles
from data.exceptions import DataNotFoundError

TRUE_SPLITS = {
    "FAST": (64.28, 31.34, 2.668525, 5.337051),
    "AAPL": (499.23, 129.04, 8.100549, 32.402197),
    "NVDA": (751.19, 186.12, 1.632315, 6.529261),
}
IDENTITY_CHANGES = {
    "DIS": (132.04, 132.47, 1.193281, 1.0),
    "APA": (19.52, 19.50, 1.216634, 1.0),
    "STX": (101.17, 99.56, 1.769281, 1.0),
    "DD": (83.93, 76.10, 1.508851, 0.736900),
    "ARNC": (16.06, 7.30, 0.538575, 1.0),
    "DLPH": (104.30, 56.50, 1.074981, 1.0),
    "IR": (129.04, 32.80, 1.578133, 1.0),
}


def _event_panel() -> pl.DataFrame:
    rows = []
    for index, (symbol, values) in enumerate({**TRUE_SPLITS, **IDENTITY_CHANGES}.items()):
        previous_close, close, previous_factor, factor = values
        security = 10_000 + index
        rows.extend(
            [
                {
                    "timestamp": date(2020, 1, 2),
                    "symbol": symbol,
                    "sec_id": security,
                    "close": previous_close,
                    "adj_factor": previous_factor,
                },
                {
                    "timestamp": date(2020, 1, 3),
                    "symbol": symbol,
                    "sec_id": security + (symbol in IDENTITY_CHANGES) * 1_000,
                    "close": close,
                    "adj_factor": factor,
                },
            ]
        )
    return pl.DataFrame(rows)


def test_true_splits_stay_continuous_and_identity_changes_are_null() -> None:
    events = reconcile_underlying_log_returns(_event_panel()).filter(
        pl.col("timestamp") == pl.date(2020, 1, 3)
    )
    splits = events.filter(pl.col("symbol").is_in(TRUE_SPLITS))
    boundaries = events.filter(pl.col("symbol").is_in(IDENTITY_CHANGES))

    assert splits["clean_log_return"].null_count() == 0
    assert splits["clean_log_return"].abs().max() < 0.04
    assert boundaries["identity_boundary"].all()
    assert boundaries["clean_log_return"].null_count() == len(IDENTITY_CHANGES)


def test_real_same_security_shocks_are_retained() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [date(2020, 3, 6), date(2020, 3, 9), date(2019, 8, 8), date(2019, 8, 9)],
            "symbol": ["OXY", "OXY", "DXC", "DXC"],
            "sec_id": [40709, 40709, 35237, 35237],
            "close": [26.86, 12.51, 51.65, 35.91],
            "adj_factor": [1.575002, 1.622729, 2.754422, 2.754422],
        }
    )
    returns = reconcile_underlying_log_returns(prices).filter(
        pl.col("clean_log_return").is_not_null()
    )

    assert returns.filter(pl.col("symbol") == "OXY")["clean_log_return"].item() == pytest.approx(
        -0.734257, abs=1e-6
    )
    assert returns.filter(pl.col("symbol") == "DXC")["clean_log_return"].item() == pytest.approx(
        -0.363474, abs=1e-6
    )


def test_future_and_segment_scale_perturbations_preserve_prior_returns() -> None:
    start = date(2020, 1, 1)
    prices = pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(8)],
            "symbol": ["AAPL"] * 8,
            "sec_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "close": [100.0, 101.0, 25.75, 25.50, 130.0, 131.0, 132.0, 133.0],
            "adj_factor": [1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    baseline = reconcile_underlying_log_returns(prices)
    scaled = reconcile_underlying_log_returns(
        prices.with_columns(
            pl.when(pl.col("sec_id") == 2)
            .then(pl.col("close") * 11)
            .otherwise(pl.col("close"))
            .alias("close"),
        )
    )
    future_perturbed = reconcile_underlying_log_returns(
        prices.with_columns(
            pl.when(pl.col("timestamp") > start + timedelta(days=5))
            .then(pl.col("adj_factor") * 7)
            .otherwise(pl.col("adj_factor"))
            .alias("adj_factor"),
        )
    )
    prefix = pl.col("timestamp") <= start + timedelta(days=5)
    columns = ["timestamp", "symbol", "clean_log_return", "identity_boundary"]

    assert (
        baseline.filter(prefix)
        .select(columns)
        .equals(future_perturbed.filter(prefix).select(columns))
    )
    segment_two = scaled.filter(pl.col("sec_id") == 2)["clean_log_return"].drop_nulls()
    assert np.allclose(
        segment_two,
        baseline.filter(pl.col("sec_id") == 2)["clean_log_return"].drop_nulls(),
    )


def test_fails_loudly_on_invalid_input_or_boundary_return() -> None:
    prices = _event_panel()
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_underlying_log_returns(pl.concat([prices, prices.head(1)]))
    with pytest.raises(ValueError, match="nonpositive"):
        reconcile_underlying_log_returns(prices.with_columns(pl.lit(0.0).alias("adj_factor")))

    reconciled = reconcile_underlying_log_returns(prices).with_columns(
        pl.when(pl.col("identity_boundary"))
        .then(pl.lit(0.01))
        .otherwise(pl.col("clean_log_return"))
        .alias("clean_log_return")
    )
    with pytest.raises(ValueError, match="crosses a security identity"):
        validate_reconciled_returns(reconciled)


def test_real_option_universe_nulls_all_identity_boundaries() -> None:
    try:
        symbols = load_sp500_options_straddles()["symbol"].unique().to_list()
        bars = load_sp500_daily_bars(symbols=symbols)
    except DataNotFoundError:
        pytest.skip("Licensed S&P 500 data is unavailable")

    reconciled = reconcile_underlying_log_returns(bars)
    boundaries = reconciled.filter(pl.col("identity_boundary"))
    known_splits = reconciled.filter(
        ((pl.col("symbol") == "FAST") & (pl.col("timestamp") == pl.date(2019, 5, 23)))
        | ((pl.col("symbol") == "AAPL") & (pl.col("timestamp") == pl.date(2020, 8, 31)))
        | ((pl.col("symbol") == "NVDA") & (pl.col("timestamp") == pl.date(2021, 7, 20)))
    )
    real_shocks = reconciled.filter(
        ((pl.col("symbol") == "OXY") & (pl.col("timestamp") == pl.date(2020, 3, 9)))
        | ((pl.col("symbol") == "DXC") & (pl.col("timestamp") == pl.date(2019, 8, 9)))
    )

    assert boundaries.height == 15
    assert boundaries["clean_log_return"].null_count() == 15
    assert known_splits["clean_log_return"].null_count() == 0
    assert known_splits["clean_log_return"].abs().max() < 0.04
    assert real_shocks["clean_log_return"].null_count() == 0
    assert real_shocks["clean_log_return"].abs().min() > 0.30
