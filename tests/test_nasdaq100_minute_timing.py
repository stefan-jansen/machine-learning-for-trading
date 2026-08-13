from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import yaml

from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices
from case_studies.utils.backtest_presets import build_backtest_spec
from case_studies.utils.backtest_runner import _run_engine

CASE_STUDY = "nasdaq100_microstructure"


def _minutes(token: str) -> int:
    values = {
        "1m": 1,
        "15m": 15,
        "1_minute": 1,
        "15_minute": 15,
    }
    return values[token]


def _minute_prices() -> pl.DataFrame:
    start = datetime(2020, 7, 2, 9, 30)
    timestamps = [start + timedelta(minutes=offset) for offset in range(7)]
    rows = []
    for timestamp in timestamps:
        for symbol, base in (("AAPL", 100.0), ("MSFT", 200.0)):
            price = base
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000,
                    "bid_open": price - 0.01,
                    "ask_open": price + 0.01,
                }
            )
    return pl.DataFrame(rows)


def _minute_predictions(prices: pl.DataFrame) -> pl.DataFrame:
    return prices.select("timestamp", "symbol").with_columns(
        y_score=pl.when(pl.col("symbol") == "AAPL").then(1.0).otherwise(0.0),
        y_true=pl.lit(0.0),
        fold_id=pl.lit(0),
        model_id=pl.lit("timing"),
        source=pl.lit("timing"),
    )


def test_nasdaq_backtest_loader_defaults_to_minute_bars(monkeypatch) -> None:
    import data.equities.loader as loader

    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return _minute_prices()

    monkeypatch.setattr(loader, "load_nasdaq100_bars", fake_loader)

    loaded = load_backtest_prices(CASE_STUDY, max_symbols=2)

    assert captured["frequency"] == "1m"
    assert loaded["timestamp"].n_unique() == 7


def test_production_nasdaq_cadence_matches_declared_rebalance_steps() -> None:
    config_dir = Path("case_studies") / CASE_STUDY / "config"
    setup = yaml.safe_load((config_dir / "setup.yaml").read_text())
    backtest = yaml.safe_load((config_dir / "backtest" / "base.yaml").read_text())

    cadence_minutes = _minutes(setup["decision"]["bar_frequency"])
    assert cadence_minutes == _minutes(backtest["calendar"]["data_frequency"])
    assert cadence_minutes == 1
    assert setup["execution"]["allocator_lookback"] == 7_800
    assert setup["backtest"]["sweep"]["cadence_sweep"][0] == "1_minute"

    position_controls = setup["backtest"]["sweep"]["risk_controls"]["position"]
    time_exits = {
        control["name"]: control["bars"]
        for control in position_controls
        if control["type"] == "time_exit"
    }
    assert time_exits == {"time_exit_10": 150, "time_exit_20": 300, "time_exit_40": 600}

    for label, step in setup["labels"]["rebalance_step"].items():
        match = re.search(r"_(\d+)m$", label)
        assert match is not None, label
        horizon_minutes = int(match.group(1))
        expected_step = max(1, math.ceil((horizon_minutes - 1) / cadence_minutes))
        assert step == expected_step, label


def test_minute_engine_fills_next_bar_and_replacement_at_label_exit() -> None:
    prices = _minute_prices()
    predictions = _minute_predictions(prices)
    case_config = get_backtest_config(CASE_STUDY)
    spec = build_backtest_spec(
        CASE_STUDY,
        case_config,
        prices=prices,
        prediction_hash="timing-test",
        initial_cash=1_000_000,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="engine",
        min_weight_change=0.0,
        min_trade_value=0.0,
    )
    spec["strategy"]["rebalance"]["cadence"] = "1_minute"
    spec["backtest_config"]["calendar"]["data_frequency"] = "1m"
    spec["backtest_config"]["metadata"]["cadence"] = "1_minute"
    start = prices["timestamp"].min()
    assert isinstance(start, datetime)
    weights = pl.DataFrame(
        {
            "timestamp": [start, start, start + timedelta(minutes=4), start + timedelta(minutes=4)],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "weight": [1.0, 0.0, 0.0, 1.0],
        }
    )

    result = _run_engine(
        weights=weights,
        prices=prices,
        predictions=predictions,
        strategy_spec=spec,
        rebalance_spec=spec["strategy"]["rebalance"],
        risk_spec={},
        allow_short=False,
        initial_cash=1_000_000,
        calendar="NYSE",
    )

    fills = result["fills_df"]
    assert fills is not None
    aapl = fills.filter(pl.col("asset") == "AAPL").sort("timestamp")
    assert aapl["timestamp"].to_list() == [
        start + timedelta(minutes=1),
        start + timedelta(minutes=5),
    ]
    msft = fills.filter(pl.col("asset") == "MSFT").select("timestamp", "side")
    assert msft.to_dicts() == [
        {"timestamp": start + timedelta(minutes=5), "side": "buy"},
    ]
