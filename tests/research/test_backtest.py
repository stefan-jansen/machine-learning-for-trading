"""Focused event-ordering and risk tests for the MNQ backtest."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from research.mnq_strategy.backtest import run_backtest
from research.mnq_strategy.config import StrategyConfig
from research.mnq_strategy.fixtures import (
    make_confirmed_signal_fixture,
    make_cost_fixture,
    make_daily_guard_fixture,
    make_overlapping_signals_fixture,
    make_stop_target_fixture,
)

NEW_YORK = "America/New_York"


def _bars(rows: list[dict]) -> pl.DataFrame:
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.col("timestamp_ny").cast(pl.Datetime(time_zone=NEW_YORK)),
        pl.col("session_date").cast(pl.Date),
        pl.col("bar_closed").cast(pl.Boolean),
        pl.col("signal").cast(pl.Boolean),
        pl.col("direction").cast(pl.Utf8),
        pl.col("signal_type").cast(pl.Utf8),
    )


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
    bar_closed: bool = True,
) -> dict:
    return {
        "timestamp_ny": timestamp,
        "session_date": date.fromisoformat(timestamp.date().isoformat()),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "bar_closed": bar_closed,
        "signal": signal,
        "direction": direction,
        "signal_type": setup,
    }


def test_entry_is_first_price_after_confirmation_close():
    result = run_backtest(make_confirmed_signal_fixture(), StrategyConfig())

    assert result.height == 1
    assert result["entry_time"][0] > result["signal_time"][0]
    assert result["entry_price"][0] == 100.25


def test_backtest_rejects_irregular_five_minute_cadence_before_execution():
    bars = _bars(
        [
            _row(datetime(2024, 1, 8, 10, 0), signal=True, direction="long", setup="10am"),
            _row(datetime(2024, 1, 8, 10, 37), open_=100.0, close=101.0),
        ]
    )

    with pytest.raises(ValueError, match="5-minute|cadence"):
        run_backtest(bars, StrategyConfig())


def test_momentum_signal_is_rejected_without_execution():
    bars = _bars(
        [
            _row(datetime(2024, 1, 8, 10, 0), signal=True, direction="long", setup="momentum"),
            _row(datetime(2024, 1, 8, 10, 5), open_=100.0, close=101.0),
        ]
    )

    result = run_backtest(bars, StrategyConfig())

    assert result.filter(pl.col("status") == "closed").is_empty()
    assert result["rejection_reason"].to_list() == ["unsupported_signal_type"]


def test_only_closed_signal_rows_are_consumed():
    bars = make_confirmed_signal_fixture().with_columns(
        pl.when(pl.col("signal")).then(False).otherwise(pl.col("bar_closed")).alias("bar_closed")
    )

    result = run_backtest(bars, StrategyConfig())

    assert result.is_empty()


def test_one_position_at_a_time_rejects_overlapping_entry_window():
    bars = make_overlapping_signals_fixture()
    result = run_backtest(bars.with_columns(pl.lit(True).alias("bar_closed")), StrategyConfig())

    assert result.filter(pl.col("status") == "closed").height == 1
    rejected = result.filter(pl.col("rejection_reason") == "position_already_open")
    assert rejected.height == 1


def test_overlapping_fixture_declares_actual_next_bar_entries():
    bars = make_overlapping_signals_fixture()
    signals = bars.filter(pl.col("signal")).sort("timestamp_ny")

    for signal in signals.to_dicts():
        next_bar = (
            bars.filter(pl.col("timestamp_ny") > signal["timestamp_ny"])
            .sort("timestamp_ny")
            .row(0, named=True)
        )
        assert signal["entry_time"] == next_bar["timestamp_ny"]


def test_long_trade_uses_direction_aware_pnl():
    signal_time = datetime(2024, 1, 8, 10, 0)
    bars = _bars(
        [
            _row(signal_time, signal=True, direction="long", setup="10am"),
            _row(signal_time + timedelta(minutes=5), open_=100.0, close=101.0),
        ]
    )

    result = run_backtest(bars, StrategyConfig())

    assert result["exit_reason"][0] == "end_of_data"
    assert result["gross_pnl"][0] == pytest.approx(18.0)
    assert result["net_pnl"][0] == pytest.approx(-27.0)


def test_short_trade_uses_direction_aware_pnl():
    signal_time = datetime(2024, 1, 8, 10, 0)
    bars = _bars(
        [
            _row(signal_time, signal=True, direction="short", setup="10am"),
            _row(signal_time + timedelta(minutes=5), open_=100.0, close=99.0),
        ]
    )

    result = run_backtest(bars, StrategyConfig())

    assert result["gross_pnl"][0] == pytest.approx(18.0)
    assert result["stop"][0] == pytest.approx(110.0)
    assert result["target"][0] == pytest.approx(80.0)


@pytest.mark.parametrize(
    ("fixture", "direction", "expected_exit"),
    [
        (make_stop_target_fixture, "long", 90.25),
    ],
)
def test_both_stop_and_target_touched_resolves_stop_first(fixture, direction, expected_exit):
    result = run_backtest(fixture(), StrategyConfig())

    assert result["direction"][0] == direction
    assert result["exit_price"][0] == pytest.approx(expected_exit)
    assert result["exit_reason"][0] == "stop_loss"


def test_short_both_stop_and_target_touched_resolves_stop_first():
    signal_time = datetime(2024, 1, 8, 10, 0)
    bars = _bars(
        [
            _row(signal_time, signal=True, direction="short", setup="10am"),
            _row(
                signal_time + timedelta(minutes=5),
                open_=100.0,
                high=120.0,
                low=70.0,
                close=100.0,
            ),
        ]
    )

    result = run_backtest(bars, StrategyConfig())

    assert result["exit_price"][0] == pytest.approx(110.0)
    assert result["exit_reason"][0] == "stop_loss"


def test_costs_are_reported_separately_from_raw_entry_price():
    result = run_backtest(
        make_cost_fixture(),
        StrategyConfig(),
    )

    assert result["entry_price"][0] == 100.25
    assert result["adjusted_entry_price"][0] == 100.75
    assert result["total_costs"][0] == pytest.approx(45.0)
    assert result["gross_pnl"][0] == pytest.approx(9.0)
    assert result["net_pnl"][0] == pytest.approx(-36.0)


def test_unclosed_next_bar_cannot_be_used_as_entry():
    bars = make_confirmed_signal_fixture().with_columns(
        pl.when(pl.col("timestamp_ny") == datetime(2024, 1, 8, 10, 5, tzinfo=ZoneInfo(NEW_YORK)))
        .then(False)
        .otherwise(pl.col("bar_closed"))
        .alias("bar_closed")
    )

    result = run_backtest(bars, StrategyConfig())

    assert result.height == 1
    assert result["status"][0] == "rejected"
    assert result["rejection_reason"][0] == "no_eligible_entry_bar"
    assert result["entry_time"][0] is None


def test_unclosed_bar_cannot_trigger_an_ohlc_exit_or_end_of_data_fill():
    signal_time = datetime(2024, 1, 8, 10, 0)
    bars = _bars(
        [
            _row(signal_time, signal=True, direction="long", setup="10am"),
            _row(signal_time + timedelta(minutes=5), open_=100.0, close=100.0),
            _row(
                signal_time + timedelta(minutes=10),
                high=100.5,
                low=80.0,
                close=80.0,
                bar_closed=False,
            ),
            _row(signal_time + timedelta(minutes=15), open_=100.0, close=101.0),
        ]
    )

    result = run_backtest(bars, StrategyConfig())

    assert result["exit_reason"][0] == "end_of_data"
    assert result["exit_time"][0].hour == 5
    assert result["exit_time"][0].minute == 15
    assert result["exit_price"][0] == pytest.approx(101.0)


def test_non_mnq_point_value_is_rejected_before_execution():
    config = StrategyConfig(point_value=1.0)

    with pytest.raises(ValueError, match="point_value"):
        run_backtest(make_confirmed_signal_fixture(), config)


def test_non_fixed_stop_distance_is_rejected_before_execution():
    config = StrategyConfig(stop_points=31.25, target_points=40.0)

    with pytest.raises(ValueError, match="stop_points"):
        run_backtest(make_confirmed_signal_fixture(), config)


def test_daily_guard_latches_after_two_realized_losses():
    base = datetime(2024, 1, 8, 10, 0)
    rows: list[dict] = []
    for index in range(3):
        signal_time = base + timedelta(minutes=index * 10)
        entry_time = signal_time + timedelta(minutes=5)
        rows.append(_row(signal_time, signal=True, direction="long", setup="10am"))
        rows.append(
            _row(
                entry_time,
                open_=100.0,
                high=100.5,
                low=89.0,
                close=95.0,
            )
        )

    result = run_backtest(
        _bars(rows),
        StrategyConfig(),
    )

    assert result.filter(pl.col("status") == "closed").height == 2
    assert result.filter(pl.col("rejection_reason") == "daily_loss_limit").height == 1


def test_daily_guard_fixture_realizes_two_losses_before_rejecting_third_signal():
    result = run_backtest(make_daily_guard_fixture(), StrategyConfig())

    closed = result.filter(pl.col("status") == "closed")
    assert closed.height == 2
    assert closed["net_pnl"].to_list() == [-225.0, -225.0]
    assert result.filter(pl.col("rejection_reason") == "daily_loss_limit").height == 1


def test_empty_and_signal_free_inputs_return_stable_empty_frame():
    empty = make_confirmed_signal_fixture().head(0)
    signal_free = make_confirmed_signal_fixture().with_columns(pl.lit(False).alias("signal"))

    empty_result = run_backtest(empty, StrategyConfig())
    signal_free_result = run_backtest(signal_free, StrategyConfig())

    assert empty_result.is_empty()
    assert signal_free_result.is_empty()
    assert empty_result.schema == signal_free_result.schema
    assert empty_result.schema["contracts"] == pl.Int64
    assert empty_result.schema["signal_time"] == pl.Datetime(time_zone=NEW_YORK)


def test_signal_on_last_bar_is_rejected_without_an_entry_observation():
    bars = (
        make_confirmed_signal_fixture()
        .tail(1)
        .with_columns(
            pl.lit(True).alias("signal"),
            pl.lit("long").alias("direction"),
            pl.lit("10am").alias("signal_type"),
        )
    )

    result = run_backtest(bars, StrategyConfig())

    assert result["status"][0] == "rejected"
    assert result["rejection_reason"][0] == "no_eligible_entry_bar"


def test_fixed_signal_contract_is_validated_before_execution():
    config = StrategyConfig(rejection_wick_ratio=2.1)

    with pytest.raises(ValueError, match="fixed signal contract"):
        run_backtest(make_confirmed_signal_fixture(), config)
