from __future__ import annotations

from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices,
)
from utils.modeling import load_modeling_dataset


def test_us_equities_pilot_helpers_preserve_current_outputs() -> None:
    bt = get_backtest_config("us_equities_panel")
    prices = load_backtest_prices("us_equities_panel", max_symbols=2)
    mds = load_modeling_dataset("us_equities_panel", "fwd_ret_1d", max_symbols=2)

    assert bt.primary_label == "fwd_ret_1d"
    assert bt.label_buffer == "1D"
    assert bt.calendar == "NYSE"
    assert bt.cadence == "daily_close"

    assert prices.columns == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert prices["symbol"].n_unique() == 2

    assert mds.label_col == "fwd_ret_1d"
    assert mds.date_col == "timestamp"
    assert mds.entity_cols == ["symbol"]
    assert mds.join_cols == ["symbol", "timestamp"]
    assert len(mds.feature_names) == 72
    assert len(mds.splits) == 16
    assert mds.label_buffer == "1D"
    assert mds.task_type == "regression"


def test_microstructure_pilot_helpers_preserve_current_outputs() -> None:
    bt = get_backtest_config("nasdaq100_microstructure")
    prices = load_backtest_prices("nasdaq100_microstructure", max_symbols=2)
    mds = load_modeling_dataset("nasdaq100_microstructure", "fwd_ret_15m", max_symbols=2)

    assert bt.primary_label == "fwd_ret_15m"
    assert bt.label_buffer == "15min"
    assert bt.calendar == "NYSE"
    assert bt.cadence == "15_minute"

    # Microstructure carries OHLCV + bid/ask OHLC so the backtest engine can
    # cost spread-aware fills.
    required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert all(c in prices.columns for c in required_cols)
    assert "bid_close" in prices.columns and "ask_close" in prices.columns
    assert prices["symbol"].n_unique() == 2

    assert mds.label_col == "fwd_ret_15m"
    assert mds.date_col == "timestamp"
    assert mds.entity_cols == ["symbol"]
    assert mds.join_cols == ["symbol", "timestamp"]
    assert len(mds.feature_names) == 88
    assert len(mds.splits) == 2
    assert mds.label_buffer == "15min"
    assert mds.task_type == "regression"
