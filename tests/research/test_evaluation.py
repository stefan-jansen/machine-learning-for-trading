"""Focused walk-forward and metric tests for the MNQ evaluator."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from research.mnq_strategy.config import StrategyConfig
from research.mnq_strategy.evaluation import WalkForwardWindow, walk_forward_evaluate
from research.mnq_strategy.fixtures import make_confirmed_signal_fixture
from research.mnq_strategy.risk import CostModel

NEW_YORK = "America/New_York"


def _row(
    timestamp: datetime,
    *,
    open_: float = 100.0,
    high: float = 100.5,
    low: float = 99.5,
    close: float = 100.0,
    signal: bool = False,
    direction: str | None = None,
    setup: str | None = None,
) -> dict:
    return {
        "timestamp_ny": timestamp,
        "session_date": timestamp.date(),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "bar_closed": True,
        "signal": signal,
        "direction": direction,
        "signal_type": setup,
    }


def _bars(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp_ny").cast(pl.Datetime(time_zone=NEW_YORK)),
        pl.col("session_date").cast(pl.Date),
        pl.col("bar_closed").cast(pl.Boolean),
        pl.col("signal").cast(pl.Boolean),
        pl.col("direction").cast(pl.Utf8),
        pl.col("signal_type").cast(pl.Utf8),
    )


def _metric_fixture() -> pl.DataFrame:
    rows = [_row(datetime(2024, 6, 28, 15, 55))]
    outcomes = ("target", "stop", "target")
    for index, outcome in enumerate(outcomes):
        signal_time = datetime(2024, 7, 1 + index, 10, 0)
        rows.append(_row(signal_time, signal=True, direction="long", setup="10am"))
        rows.append(
            _row(
                signal_time + timedelta(minutes=5),
                open_=100.0,
                high=120.0 if outcome == "target" else 100.5,
                low=90.0 if outcome == "stop" else 99.5,
                close=100.0,
            )
        )
    return _bars(rows)


def _config() -> StrategyConfig:
    return StrategyConfig()


def test_walk_forward_keeps_test_window_out_of_parameter_selection():
    windows = [
        WalkForwardWindow(
            train_start="2024-06-01",
            train_end="2024-06-30",
            test_start="2024-07-01",
            test_end="2024-07-31",
        )
    ]

    report = walk_forward_evaluate(_metric_fixture(), windows=windows, config=_config())

    assert report["test_results"]
    assert report["chronology_check"] is True
    assert report["signal_provenance_check"] == "not_available"
    assert report["selection_policy"]["threshold_selection"] == "fixed_config_only"
    assert report["selection_policy"]["test_data_used_for_selection"] is False


def test_walk_forward_supports_multiple_chronological_windows():
    windows = [
        WalkForwardWindow("2024-01-01", "2024-06-30", "2024-07-01", "2024-07-31"),
        WalkForwardWindow("2024-08-01", "2024-08-31", "2024-09-01", "2024-09-30"),
    ]

    report = walk_forward_evaluate(_metric_fixture(), windows, _config())

    assert report["selection_policy"]["window_count"] == 2
    assert len(report["windows"]) == 2
    assert report["chronology_check"] is True


def test_walk_forward_normalizes_utc_timestamp_before_date_window_masking():
    bars = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 6, 30, 23, 30),
                datetime(2024, 6, 30, 23, 35),
                datetime(2024, 7, 1, 14, 0),
            ],
            "open": [100.0, 100.0, 100.0],
            "high": [100.5, 100.5, 100.5],
            "low": [99.5, 99.5, 99.5],
            "close": [100.0, 100.0, 100.0],
            "volume": [100, 100, 100],
            "bar_closed": [True, True, True],
            "signal": [False, False, False],
            "direction": [None, None, None],
            "signal_type": [None, None, None],
            "session_date": [date(2024, 6, 30), date(2024, 7, 1), date(2024, 7, 1)],
        }
    ).with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_zone="UTC")),
        pl.col("session_date").cast(pl.Date),
        pl.col("bar_closed").cast(pl.Boolean),
        pl.col("signal").cast(pl.Boolean),
        pl.col("direction").cast(pl.Utf8),
        pl.col("signal_type").cast(pl.Utf8),
    )
    window = WalkForwardWindow("2024-06-01", "2024-06-30", "2024-07-01", "2024-07-01")

    report = walk_forward_evaluate(bars, [window], _config())

    assert report["windows"][0]["train_rows"] == 2
    assert report["windows"][0]["test_rows"] == 1
    assert report["chronology_check"] is True
    assert report["test_results"] == []


def test_walk_forward_rejects_irregular_cadence():
    bars = _metric_fixture().with_columns(
        pl.when(pl.arange(0, pl.len()) == 2)
        .then(pl.col("timestamp_ny") + timedelta(minutes=37))
        .otherwise(pl.col("timestamp_ny"))
        .alias("timestamp_ny")
    )
    window = WalkForwardWindow("2024-06-01", "2024-06-30", "2024-07-01", "2024-07-31")

    with pytest.raises(ValueError, match="5-minute|cadence"):
        walk_forward_evaluate(bars, [window], _config())


def test_walk_forward_report_calculates_trade_metrics_without_division_by_zero():
    window = WalkForwardWindow(
        train_start=date(2024, 6, 1),
        train_end=date(2024, 6, 30),
        test_start=date(2024, 7, 1),
        test_end=date(2024, 7, 31),
    )

    report = walk_forward_evaluate(_metric_fixture(), [window], _config())

    assert report["trade_count"] == 3
    assert report["net_pnl"] == pytest.approx(405.0)
    assert report["expectancy"] == pytest.approx(135.0)
    assert report["win_rate"] == pytest.approx(2.0 / 3.0)
    assert report["profit_factor"] == pytest.approx(2.8)
    assert report["max_drawdown"] == pytest.approx(225.0)
    assert report["daily_loss_breaches"] == 0
    assert report["consecutive_loss_breaches"] == 0
    assert report["average_r"] == pytest.approx(1.0)
    assert report["cost_share"] == pytest.approx(0.25)
    assert report["per_setup"]["10am"]["trade_count"] == 3
    assert report["config_hash"] == _config().config_hash


def test_walk_forward_resets_loss_streak_at_each_session_date():
    rows = []
    for index, session_day in enumerate((date(2024, 7, 1), date(2024, 7, 2))):
        signal_time = datetime.combine(session_day, datetime.min.time()).replace(hour=10)
        rows.append(_row(signal_time, signal=True, direction="long", setup="10am"))
        rows.append(
            _row(
                signal_time + timedelta(minutes=5),
                open_=100.0,
                low=89.0,
                close=95.0,
            )
        )
    bars = _bars(rows)
    window = WalkForwardWindow("2024-06-01", "2024-06-30", "2024-07-01", "2024-07-31")

    report = walk_forward_evaluate(bars, [window], _config())

    assert report["trade_count"] == 2
    assert report["consecutive_loss_breaches"] == 0
    assert {row["session_date"] for row in report["test_results"]} == {
        date(2024, 7, 1),
        date(2024, 7, 2),
    }


def test_walk_forward_rejects_overlapping_or_reversed_windows():
    with pytest.raises(ValueError, match="chronological|overlap"):
        walk_forward_evaluate(
            _metric_fixture(),
            [
                WalkForwardWindow(
                    train_start="2024-06-01",
                    train_end="2024-07-02",
                    test_start="2024-07-01",
                    test_end="2024-07-31",
                )
            ],
            _config(),
        )

    with pytest.raises(ValueError, match="chronological|overlap"):
        WalkForwardWindow("2024-07-01", "2024-07-31", "2024-07-01", "2024-08-01")

    with pytest.raises(ValueError, match="chronological|overlap"):
        walk_forward_evaluate(
            _metric_fixture(),
            [
                WalkForwardWindow(
                    train_start="2024-07-01",
                    train_end="2024-07-31",
                    test_start="2024-06-01",
                    test_end="2024-06-30",
                )
            ],
            _config(),
        )


def test_walk_forward_returns_zero_metrics_for_empty_or_signal_free_input():
    window = WalkForwardWindow("2024-01-01", "2024-06-30", "2024-07-01", "2024-09-30")
    empty = make_confirmed_signal_fixture().head(0)
    signal_free = make_confirmed_signal_fixture().with_columns(pl.lit(False).alias("signal"))

    for bars in (empty, signal_free):
        report = walk_forward_evaluate(bars, [window], _config())
        assert report["test_results"] == []
        assert report["trade_count"] == 0
        assert report["net_pnl"] == 0.0
        assert report["expectancy"] == 0.0
        assert report["win_rate"] == 0.0
        assert report["max_drawdown"] == 0.0
        assert report["average_r"] == 0.0
        assert report["cost_share"] == 0.0


def test_walk_forward_rejects_non_chronological_input_rows():
    window = WalkForwardWindow("2024-01-01", "2024-06-30", "2024-07-01", "2024-09-30")
    bars = _metric_fixture().select(pl.all().gather(pl.Series([2, 1, 0, 3, 4, 5, 6])))

    with pytest.raises(ValueError, match="chronological"):
        walk_forward_evaluate(bars, [window], _config())


def test_walk_forward_validates_fixed_contract_before_empty_report():
    window = WalkForwardWindow("2024-01-01", "2024-06-30", "2024-07-01", "2024-09-30")
    config = StrategyConfig(rejection_wick_ratio=2.1)

    with pytest.raises(ValueError, match="fixed signal contract"):
        walk_forward_evaluate(make_confirmed_signal_fixture().head(0), [window], config)
