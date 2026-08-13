from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from case_studies.utils.backtest_loaders import load_backtest_prices

CASE_STUDY = "nasdaq100_microstructure"


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
