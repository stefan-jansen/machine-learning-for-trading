from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import polars as pl

from case_studies.crypto_perps_funding.funding_backtest import apply_funding_settlements


def test_funding_is_position_signed_before_same_timestamp_fills() -> None:
    first = datetime(2024, 1, 1, tzinfo=UTC)
    settlement = datetime(2024, 1, 1, 8, tzinfo=UTC)
    last = datetime(2024, 1, 1, 16, tzinfo=UTC)
    buy = SimpleNamespace(
        timestamp=first,
        quantity=1.0,
        side=SimpleNamespace(value="buy"),
        price=100.0,
        commission=0.0,
        asset="BTCUSDT",
    )
    sell = SimpleNamespace(
        timestamp=settlement,
        quantity=1.0,
        side=SimpleNamespace(value="sell"),
        price=100.0,
        commission=0.0,
        asset="BTCUSDT",
    )
    engine = SimpleNamespace(
        fills=[buy, sell],
        equity_curve=[(first, 1_000.0), (settlement, 1_000.0), (last, 1_000.0)],
        portfolio_state=[
            (first, 1_000.0, 900.0, 100.0, 100.0, 1),
            (settlement, 1_000.0, 1_000.0, 0.0, 0.0, 0),
            (last, 1_000.0, 1_000.0, 0.0, 0.0, 0),
        ],
    )
    prices = pl.DataFrame(
        {
            "timestamp": [first, settlement, last],
            "symbol": ["BTCUSDT"] * 3,
            "close": [100.0] * 3,
        }
    )
    funding = pl.DataFrame(
        {
            "timestamp": [settlement],
            "symbol": ["BTCUSDT"],
            "funding_rate": [0.01],
        }
    )

    metrics = apply_funding_settlements(
        engine,
        prices=prices,
        funding_rates=funding,
        initial_cash=1_000.0,
    )

    assert metrics == {
        "funding_pnl": -1.0,
        "funding_events": 1.0,
        "funding_settlements": 1.0,
        "funding_reconstruction_error": 0.0,
    }
    assert engine.equity_curve == [(first, 1_000.0), (settlement, 999.0), (last, 999.0)]
    assert engine.portfolio_state[1][1:3] == (999.0, 999.0)


def test_funding_rejects_duplicate_settlement_keys() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    funding = pl.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "funding_rate": [0.01, 0.01],
        }
    )
    engine = SimpleNamespace(fills=[], equity_curve=[], portfolio_state=[])

    try:
        apply_funding_settlements(
            engine,
            prices=pl.DataFrame(
                {"timestamp": [timestamp], "symbol": ["BTCUSDT"], "close": [100.0]}
            ),
            funding_rates=funding,
            initial_cash=1_000.0,
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate funding settlements were accepted")
