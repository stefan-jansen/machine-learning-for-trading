"""Funding-settlement accounting for perpetual-futures engine backtests."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import polars as pl


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def apply_funding_settlements(
    engine_result: Any,
    *,
    prices: pl.DataFrame,
    funding_rates: pl.DataFrame,
    initial_cash: float,
) -> dict[str, float]:
    """Add position-signed funding cash before same-timestamp fills and update engine state."""
    required = {"symbol", "timestamp", "funding_rate"}
    missing = required - set(funding_rates.columns)
    if missing:
        raise ValueError(f"funding rates are missing columns: {sorted(missing)}")
    if funding_rates.n_unique(["symbol", "timestamp"]) != funding_rates.height:
        raise ValueError("funding settlement keys must be unique")

    fills = defaultdict(list)
    for fill in engine_result.fills:
        fills[_as_utc(fill.timestamp)].append(fill)
    rates = defaultdict(dict)
    for row in funding_rates.select(*sorted(required)).iter_rows(named=True):
        rates[_as_utc(row["timestamp"])][row["symbol"]] = float(row["funding_rate"])
    marks = defaultdict(dict)
    for row in prices.select("timestamp", "symbol", "close").iter_rows(named=True):
        marks[_as_utc(row["timestamp"])][row["symbol"]] = float(row["close"])
    engine_equity = {_as_utc(ts): float(value) for ts, value in engine_result.equity_curve}

    cash = float(initial_cash)
    positions = defaultdict(float)
    last_marks: dict[str, float] = {}
    cumulative_funding = 0.0
    events = 0
    adjusted = []
    reconstructed = []
    funding_by_equity_time: dict[datetime, float] = {}
    for timestamp in sorted(set(engine_equity) | set(rates)):
        last_marks.update(marks[timestamp])
        event_cash = sum(
            -(positions.get(symbol, 0.0) * last_marks[symbol]) * rate
            for symbol, rate in rates.get(timestamp, {}).items()
            if positions.get(symbol, 0.0) != 0 and symbol in last_marks
        )
        if event_cash:
            cash += event_cash
            cumulative_funding += event_cash
            events += 1
        for fill in fills.get(timestamp, []):
            quantity = float(fill.quantity) if fill.side.value == "buy" else -float(fill.quantity)
            cash -= quantity * float(fill.price) + float(fill.commission)
            last_marks.setdefault(fill.asset, float(fill.price))
            positions[fill.asset] += quantity
            if abs(positions[fill.asset]) < 1e-12:
                del positions[fill.asset]
        if timestamp in engine_equity:
            marked = sum(quantity * last_marks[symbol] for symbol, quantity in positions.items())
            adjusted.append((timestamp, cash + marked))
            reconstructed.append((timestamp, cash - cumulative_funding + marked))
            funding_by_equity_time[timestamp] = cumulative_funding

    reconstruction_error = max(
        (abs(value - engine_equity[timestamp]) for timestamp, value in reconstructed),
        default=0.0,
    )
    tolerance = max(1e-6, abs(initial_cash) * 1e-10)
    if reconstruction_error > tolerance:
        raise RuntimeError(
            f"funding ledger cannot reconstruct engine equity: {reconstruction_error:.12g}"
        )
    if len(rates) and not adjusted:
        raise RuntimeError("funding settlements did not overlap the engine equity timeline")

    engine_result.equity_curve = adjusted
    engine_result.portfolio_state = [
        (
            timestamp,
            equity + funding_by_equity_time[_as_utc(timestamp)],
            cash_value + funding_by_equity_time[_as_utc(timestamp)],
            gross,
            net,
            count,
        )
        for timestamp, equity, cash_value, gross, net, count in engine_result.portfolio_state
    ]
    return {
        "funding_pnl": cumulative_funding,
        "funding_events": float(events),
        "funding_settlements": float(funding_rates.height),
        "funding_reconstruction_error": reconstruction_error,
    }
