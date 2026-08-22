"""Funding-settlement accounting for perpetual-futures engine backtests."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from functools import wraps
from typing import Any

import polars as pl


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class FundingSettlementLedger:
    """Apply position-signed funding during the engine's bar-time update."""

    def __init__(self, funding_rates: pl.DataFrame) -> None:
        required = {"symbol", "timestamp", "funding_rate"}
        missing = required - set(funding_rates.columns)
        if missing:
            raise ValueError(f"funding rates are missing columns: {sorted(missing)}")
        selected = funding_rates.select("symbol", "timestamp", "funding_rate")
        if selected.null_count().row(0) != (0, 0, 0):
            raise ValueError("funding settlements cannot contain null keys or rates")
        if selected.n_unique(["symbol", "timestamp"]) != selected.height:
            raise ValueError("funding settlement keys must be unique")

        self._rates: dict[datetime, dict[str, float]] = {}
        for row in selected.sort("timestamp", "symbol").iter_rows(named=True):
            rate = float(row["funding_rate"])
            if not math.isfinite(rate):
                raise ValueError("funding rates must be finite")
            self._rates.setdefault(_as_utc(row["timestamp"]), {})[str(row["symbol"])] = rate
        self._rate_count = selected.height
        self._settled_timestamps: set[datetime] = set()
        self._funding_pnl = 0.0
        self._funding_events = 0
        self._funding_settlements = 0
        self._installed = False

    def settle(self, timestamp: datetime, broker: Any) -> float:
        """Settle one timestamp exactly once against positions marked on that bar."""
        normalized = _as_utc(timestamp)
        if normalized in self._settled_timestamps:
            return 0.0
        rates = self._rates.get(normalized)
        if rates is None:
            return 0.0
        self._settled_timestamps.add(normalized)
        self._funding_settlements += len(rates)

        event_cash = 0.0
        for symbol, rate in rates.items():
            position = broker.positions.get(symbol)
            if position is None or float(position.quantity) == 0.0:
                continue
            mark = broker.get_mark_price(symbol, quantity=position.quantity)
            if mark is None:
                raise RuntimeError(f"funding settlement has no current mark for {symbol!r}")
            event_cash -= (
                float(position.quantity)
                * float(mark)
                * float(getattr(position, "multiplier", 1.0))
                * rate
            )
        if event_cash:
            broker.cash = float(broker.cash) + event_cash
            self._funding_pnl += event_cash
            self._funding_events += 1
        return event_cash

    def install(self, broker: Any) -> None:
        """Install settlement immediately after each engine mark update."""
        if self._installed:
            raise RuntimeError("funding settlement ledger is already installed")
        original_update_time = broker._update_time

        @wraps(original_update_time)
        def update_time_with_funding(timestamp, *args, **kwargs):
            result = original_update_time(timestamp, *args, **kwargs)
            self.settle(timestamp, broker)
            return result

        broker._update_time = update_time_with_funding
        self._installed = True

    def metrics(self) -> dict[str, float]:
        """Return cashflows actually presented to the engine timeline."""
        if self._funding_settlements != self._rate_count:
            raise RuntimeError(
                "funding settlement coverage is incomplete: "
                f"{self._funding_settlements}/{self._rate_count} keys reached the engine timeline"
            )
        return {
            "funding_pnl": self._funding_pnl,
            "funding_events": float(self._funding_events),
            "funding_settlements": float(self._funding_settlements),
        }
