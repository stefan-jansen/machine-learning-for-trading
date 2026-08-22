from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from case_studies.crypto_perps_funding.funding_backtest import FundingSettlementLedger
from case_studies.utils import backtest_runner


def test_funding_changes_broker_cash_before_same_timestamp_sizing() -> None:
    first = datetime(2024, 1, 1, tzinfo=UTC)
    settlement = datetime(2024, 1, 1, 8, tzinfo=UTC)
    last = datetime(2024, 1, 1, 16, tzinfo=UTC)
    funding = pl.DataFrame(
        {
            "timestamp": [settlement],
            "symbol": ["BTCUSDT"],
            "funding_rate": [0.01],
        }
    )

    position = SimpleNamespace(quantity=1.0, current_price=90.0, multiplier=1.0)
    marks = {"BTCUSDT": 90.0}
    broker = SimpleNamespace(
        cash=1_000.0,
        positions={"BTCUSDT": position},
        get_mark_price=lambda symbol, **_kwargs: marks[symbol],
    )

    def update_time(timestamp, prices, *_args, **_kwargs):
        marks.update(prices)

    broker._update_time = update_time
    ledger = FundingSettlementLedger(funding)
    ledger.install(broker)

    broker._update_time(settlement, {"BTCUSDT": 100.0})
    target_quantity = broker.cash / 100.0

    assert broker.cash == 999.0
    assert target_quantity == 9.99
    assert ledger.metrics() == {
        "funding_pnl": -1.0,
        "funding_events": 1.0,
        "funding_settlements": 1.0,
    }

    broker._update_time(settlement, {"BTCUSDT": 100.0})
    assert broker.cash == 999.0

    broker._update_time(last, {"BTCUSDT": 100.0})
    assert ledger.metrics()["funding_pnl"] == -1.0


def test_longs_pay_and_shorts_receive_the_same_funding_rate() -> None:
    timestamp = datetime(2024, 1, 1, 8, tzinfo=UTC)
    funding = pl.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "symbol": ["LONG", "SHORT"],
            "funding_rate": [0.01, 0.01],
        }
    )
    positions = {
        "LONG": SimpleNamespace(quantity=2.0, current_price=100.0, multiplier=1.0),
        "SHORT": SimpleNamespace(quantity=-1.0, current_price=100.0, multiplier=1.0),
    }
    broker = SimpleNamespace(
        cash=1_000.0,
        positions=positions,
        get_mark_price=lambda symbol, **_kwargs: positions[symbol].current_price,
    )
    ledger = FundingSettlementLedger(funding)

    ledger.settle(timestamp, broker)

    assert broker.cash == 999.0
    assert ledger.metrics() == {
        "funding_pnl": -1.0,
        "funding_events": 1.0,
        "funding_settlements": 2.0,
    }


def test_funding_rejects_duplicate_settlement_keys() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    funding = pl.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "funding_rate": [0.01, 0.01],
        }
    )
    try:
        FundingSettlementLedger(funding)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate funding settlements were accepted")


def test_funding_rejects_incomplete_engine_timeline_coverage() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    ledger = FundingSettlementLedger(
        pl.DataFrame(
            {
                "timestamp": [timestamp],
                "symbol": ["BTCUSDT"],
                "funding_rate": [0.01],
            }
        )
    )

    with pytest.raises(RuntimeError, match="0/1"):
        ledger.metrics()


def _backtest_frames(timestamp: datetime) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    prices = pl.DataFrame(
        {
            "timestamp": [timestamp],
            "symbol": ["BTCUSDT"],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1_000.0],
        }
    )
    predictions = pl.DataFrame(
        {
            "timestamp": [timestamp],
            "symbol": ["BTCUSDT"],
            "y_score": [0.1],
            "y_true": [0.01],
        }
    )
    weights = pl.DataFrame(
        {
            "timestamp": [timestamp],
            "symbol": ["BTCUSDT"],
            "weight": [-1.0],
        }
    )
    return prices, predictions, weights


def _strategy_spec(mode: str) -> dict:
    return {
        "signal": {"method": "precomputed_positions"},
        "execution": {
            "mode": mode,
            "cadence": "8_hour_funding",
            "fill_timing": "AT_FUNDING_TIMESTAMP",
            "min_weight_change": 0.0,
            "min_trade_value": 0.0,
        },
        "costs": {"model": "percentage", "commission_bps": 0.0, "slippage_bps": 0.0},
    }


def test_negative_precomputed_weights_enable_shorting_in_resolved_engine_config(
    monkeypatch,
) -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    prices, predictions, weights = _backtest_frames(timestamp)
    prices = prices.with_columns(pl.lit("A").alias("symbol"))
    predictions = predictions.with_columns(pl.lit("A").alias("symbol"))
    weights = weights.with_columns(pl.lit("A").alias("symbol"))
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {
            "metrics": {},
            "daily_returns": pl.DataFrame({"timestamp": [timestamp], "daily_return": [0.0]}),
        }

    monkeypatch.setattr(backtest_runner, "_run_engine", fake_engine)

    backtest_runner.run_backtest(
        "etfs",
        "prediction-a",
        _strategy_spec("engine"),
        prices=prices,
        predictions=predictions,
        precomputed_weights=weights,
        register=False,
    )

    assert captured["allow_short"] is True
    assert captured["strategy_spec"]["backtest_config"]["account"]["allow_short_selling"] is True


def test_vectorized_backtest_rejects_funding_cashflows(monkeypatch) -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    prices, predictions, weights = _backtest_frames(timestamp)
    funding = pl.DataFrame(
        {
            "timestamp": [timestamp],
            "symbol": ["BTCUSDT"],
            "funding_rate": [0.01],
        }
    )
    monkeypatch.setattr(
        backtest_runner,
        "_run_vectorized",
        lambda **kwargs: pytest.fail("vectorized path ran with funding rates"),
    )

    with pytest.raises(ValueError, match="vectorized.*funding"):
        backtest_runner.run_backtest(
            "crypto_perps_funding",
            "prediction-a",
            _strategy_spec("vectorized"),
            prices=prices,
            predictions=predictions,
            precomputed_weights=weights,
            funding_rates=funding,
            register=False,
        )


def test_live_funding_changes_the_next_engine_target_order() -> None:
    timestamps = [datetime(2024, 1, 1, hour=hour, tzinfo=UTC) for hour in (0, 8, 16)]
    prices = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 3,
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
            "volume": [1_000_000.0] * 3,
        }
    )
    predictions = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 3,
            "y_score": [0.1] * 3,
            "y_true": [0.0] * 3,
        }
    )
    weights = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 3,
            "weight": [0.5] * 3,
        }
    )
    funding = pl.DataFrame(
        {
            "timestamp": [timestamps[1]],
            "symbol": ["BTCUSDT"],
            "funding_rate": [0.1],
        }
    )

    unfunded = backtest_runner.run_backtest(
        "crypto_perps_funding",
        "prediction-a",
        _strategy_spec("engine"),
        prices=prices,
        predictions=predictions,
        precomputed_weights=weights,
        register=False,
    )
    funded = backtest_runner.run_backtest(
        "crypto_perps_funding",
        "prediction-a",
        _strategy_spec("engine"),
        prices=prices,
        predictions=predictions,
        precomputed_weights=weights,
        funding_rates=funding,
        register=False,
    )

    assert funded.metrics["funding_pnl"] < 0
    assert len(funded.engine_result.fills) > len(unfunded.engine_result.fills)
    assert any(fill.timestamp == timestamps[1] for fill in funded.engine_result.fills)


def test_timezone_naive_engine_targets_match_timezone_naive_feed() -> None:
    timestamps = [datetime(2024, 1, 1, hour=hour) for hour in (0, 8)]
    prices = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 2,
            "open": [100.0] * 2,
            "high": [100.0] * 2,
            "low": [100.0] * 2,
            "close": [100.0] * 2,
            "volume": [1_000_000.0] * 2,
        }
    )
    predictions = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 2,
            "y_score": [0.1] * 2,
            "y_true": [0.0] * 2,
        }
    )
    weights = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 2,
            "weight": [0.5] * 2,
        }
    )

    result = backtest_runner.run_backtest(
        "crypto_perps_funding",
        "prediction-a",
        _strategy_spec("engine"),
        prices=prices,
        predictions=predictions,
        precomputed_weights=weights,
        register=False,
    )

    assert result.engine_result.fills


def test_date_engine_targets_match_date_feed() -> None:
    timestamps = [datetime(2024, 1, day).date() for day in (2, 3)]
    prices = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["A"] * 2,
            "open": [100.0] * 2,
            "high": [100.0] * 2,
            "low": [100.0] * 2,
            "close": [100.0] * 2,
            "volume": [1_000_000.0] * 2,
        }
    )
    predictions = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["A"] * 2,
            "y_score": [0.1] * 2,
            "y_true": [0.0] * 2,
        }
    )
    weights = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["A"] * 2,
            "weight": [0.5] * 2,
        }
    )

    result = backtest_runner.run_backtest(
        "etfs",
        "prediction-a",
        _strategy_spec("engine"),
        prices=prices,
        predictions=predictions,
        precomputed_weights=weights,
        register=False,
    )

    assert result.engine_result.fills


@pytest.mark.parametrize(
    ("timestamps", "expected_avg_turnover"),
    [
        ([datetime(2024, 1, 1, hour=hour, tzinfo=UTC) for hour in (0, 8)], 1.5),
        ([datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2)], 0.75),
    ],
    ids=["intraday-events-summed", "daily-events-averaged"],
)
def test_contiguous_state_transition_flattens_then_reenters_at_boundary(
    timestamps: list[datetime], expected_avg_turnover: float
) -> None:
    prices = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 2,
            "open": [100.0] * 2,
            "high": [100.0] * 2,
            "low": [100.0] * 2,
            "close": [100.0] * 2,
            "volume": [1_000_000.0] * 2,
        }
    )
    predictions = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 2,
            "y_score": [0.1] * 2,
            "y_true": [0.0] * 2,
        }
    )
    weights = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 2,
            "weight": [0.5, 0.5],
            "_state_transition": [False, True],
        }
    )

    result = backtest_runner.run_backtest(
        "crypto_perps_funding",
        "prediction-a",
        _strategy_spec("engine"),
        prices=prices,
        predictions=predictions,
        precomputed_weights=weights,
        register=False,
    )

    assert [fill.side.value for fill in result.engine_result.fills] == ["buy", "sell", "buy"]
    assert result.engine_result.fills[1].timestamp == timestamps[1]
    assert result.engine_result.fills[2].timestamp == timestamps[1]
    assert result.metrics["avg_turnover"] == pytest.approx(expected_avg_turnover)


def test_state_transition_turnover_includes_symbol_missing_from_new_target() -> None:
    timestamps = [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2)]
    price_keys = [(timestamp, symbol) for timestamp in timestamps for symbol in ("BTC", "ETH")]
    prices = pl.DataFrame(
        {
            "timestamp": [timestamp for timestamp, _ in price_keys],
            "symbol": [symbol for _, symbol in price_keys],
            "open": [100.0] * 4,
            "high": [100.0] * 4,
            "low": [100.0] * 4,
            "close": [100.0] * 4,
            "volume": [1_000_000.0] * 4,
        }
    )
    predictions = pl.DataFrame(
        {
            "timestamp": [timestamp for timestamp, _ in price_keys],
            "symbol": [symbol for _, symbol in price_keys],
            "y_score": [0.1] * 4,
            "y_true": [0.0] * 4,
        }
    )
    weights = pl.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[0], timestamps[1]],
            "symbol": ["BTC", "ETH", "BTC"],
            "weight": [0.5, 0.5, 0.5],
            "_state_transition": [False, False, True],
        }
    )

    result = backtest_runner.run_backtest(
        "crypto_perps_funding",
        "prediction-a",
        _strategy_spec("engine"),
        prices=prices,
        predictions=predictions,
        precomputed_weights=weights,
        register=False,
    )

    assert [fill.side.value for fill in result.engine_result.fills] == [
        "buy",
        "buy",
        "sell",
        "sell",
        "buy",
    ]
    assert result.metrics["avg_turnover"] == pytest.approx(1.25)
