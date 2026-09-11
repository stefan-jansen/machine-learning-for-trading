"""Event-driven, cost-aware backtest for the objective MNQ strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import polars as pl

from .config import StrategyConfig
from .risk import DailyRiskGuard, RiskDecision, calculate_position_size

RESULT_COLUMNS = (
    "signal_time",
    "entry_time",
    "exit_time",
    "setup",
    "direction",
    "session_date",
    "contracts",
    "stop",
    "target",
    "entry_price",
    "adjusted_entry_price",
    "exit_price",
    "adjusted_exit_price",
    "gross_pnl",
    "total_costs",
    "net_pnl",
    "r_multiple",
    "status",
    "exit_reason",
    "rejection_reason",
)

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "bar_closed", "signal", "direction"}


@dataclass
class _Position:
    signal_time: datetime
    entry_index: int
    entry_time: datetime
    setup: str
    direction: str
    contracts: int
    stop: float
    target: float
    entry_price: float
    adjusted_entry_price: float
    risk_decision: RiskDecision


def _timestamp_dtype(frame: pl.DataFrame) -> pl.DataType:
    dtype = frame.schema["timestamp_ny"]
    if dtype == pl.Datetime:
        return dtype
    if isinstance(dtype, pl.Datetime):
        return dtype
    raise ValueError("timestamp_ny must be a datetime column")


def _coerce_timestamp_column(frame: pl.DataFrame, config: StrategyConfig) -> pl.DataFrame:
    """Ensure a timezone-aware New York timestamp column is available."""
    result = frame
    if "timestamp_ny" not in result.columns:
        if "timestamp" not in result.columns:
            raise ValueError("required input column missing: timestamp_ny")
        result = result.with_columns(pl.col("timestamp").alias("timestamp_ny"))

    timestamp_dtype = result.schema["timestamp_ny"]
    if timestamp_dtype == pl.Date:
        result = result.with_columns(pl.col("timestamp_ny").cast(pl.Datetime))
        timestamp_dtype = result.schema["timestamp_ny"]
    if timestamp_dtype == pl.Utf8:
        try:
            result = result.with_columns(
                pl.col("timestamp_ny").str.to_datetime(strict=True).alias("timestamp_ny")
            )
        except (pl.exceptions.ComputeError, ValueError) as exc:
            raise ValueError("timestamp_ny contains invalid datetime values") from exc
        timestamp_dtype = result.schema["timestamp_ny"]
    if not isinstance(timestamp_dtype, pl.Datetime):
        try:
            result = result.with_columns(
                pl.col("timestamp_ny").cast(pl.Datetime, strict=True).alias("timestamp_ny")
            )
        except (pl.exceptions.ComputeError, ValueError) as exc:
            raise ValueError("timestamp_ny must contain datetime values") from exc
        timestamp_dtype = result.schema["timestamp_ny"]

    if timestamp_dtype.time_zone is None:
        result = result.with_columns(
            pl.col("timestamp_ny").dt.replace_time_zone(config.timezone).alias("timestamp_ny")
        )
    elif timestamp_dtype.time_zone != config.timezone:
        result = result.with_columns(
            pl.col("timestamp_ny").dt.convert_time_zone(config.timezone).alias("timestamp_ny")
        )
    return result


def _prepare_bars(bars: pl.DataFrame, config: StrategyConfig) -> pl.DataFrame:
    """Validate and chronologically order a backtest input frame."""
    if not isinstance(bars, pl.DataFrame):
        raise TypeError("bars must be a polars.DataFrame")
    result = _coerce_timestamp_column(bars.clone(), config)
    missing = sorted(_REQUIRED_COLUMNS.difference(result.columns))
    if missing:
        raise ValueError(f"required backtest columns missing: {', '.join(missing)}")
    if result.schema["bar_closed"] != pl.Boolean or result["bar_closed"].null_count():
        raise ValueError("bar_closed must contain only explicit boolean values")
    if result.schema["signal"] != pl.Boolean or result["signal"].null_count():
        raise ValueError("signal must contain only explicit boolean values")
    for column in ("timestamp_ny", "open", "high", "low", "close"):
        if result[column].null_count():
            raise ValueError(f"{column} must not contain null values")
    if "signal_type" not in result.columns:
        if "setup" in result.columns:
            result = result.with_columns(pl.col("setup").alias("signal_type"))
        else:
            result = result.with_columns(pl.lit(None, dtype=pl.Utf8).alias("signal_type"))
    if "session_date" not in result.columns:
        result = result.with_columns(pl.col("timestamp_ny").dt.date().alias("session_date"))
    else:
        result = result.with_columns(pl.col("session_date").cast(pl.Date, strict=True))

    result = result.sort("timestamp_ny")
    timestamps = result["timestamp_ny"].to_list()
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("backtest timestamps must be unique")
    raw_rows = result.to_dicts()
    blocked_entries = {
        row["timestamp_ny"]
        for index, row in enumerate(raw_rows[:-1])
        if row["signal"] and row["bar_closed"] and not raw_rows[index + 1]["bar_closed"]
    }
    result = result.filter(pl.col("bar_closed"))
    return result.with_columns(
        pl.col("timestamp_ny")
        .is_in(list(blocked_entries))
        .fill_null(False)
        .alias("_entry_blocked_by_unclosed")
        if blocked_entries
        else pl.lit(False).alias("_entry_blocked_by_unclosed")
    )


def _result_schema(timestamp_dtype: pl.DataType) -> dict[str, pl.DataType]:
    return {
        "signal_time": timestamp_dtype,
        "entry_time": timestamp_dtype,
        "exit_time": timestamp_dtype,
        "setup": pl.Utf8,
        "direction": pl.Utf8,
        "session_date": pl.Date,
        "contracts": pl.Int64,
        "stop": pl.Float64,
        "target": pl.Float64,
        "entry_price": pl.Float64,
        "adjusted_entry_price": pl.Float64,
        "exit_price": pl.Float64,
        "adjusted_exit_price": pl.Float64,
        "gross_pnl": pl.Float64,
        "total_costs": pl.Float64,
        "net_pnl": pl.Float64,
        "r_multiple": pl.Float64,
        "status": pl.Utf8,
        "exit_reason": pl.Utf8,
        "rejection_reason": pl.Utf8,
    }


def _frame_from_rows(rows: list[dict[str, Any]], timestamp_dtype: pl.DataType) -> pl.DataFrame:
    schema = _result_schema(timestamp_dtype)
    return pl.DataFrame(
        {
            column: pl.Series(
                column,
                [row.get(column) for row in rows],
                dtype=dtype,
            )
            for column, dtype in schema.items()
        }
    ).select(RESULT_COLUMNS)


def _empty_results(timestamp_dtype: pl.DataType) -> pl.DataFrame:
    return _frame_from_rows([], timestamp_dtype)


def _session_date(row: dict[str, Any]) -> date:
    value = row.get("session_date")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return row["timestamp_ny"].date()


def _directional_price(
    entry_price: float, direction: str, distance: float, is_target: bool
) -> float:
    if direction == "long":
        return entry_price + distance if is_target else entry_price - distance
    return entry_price - distance if is_target else entry_price + distance


def _exit_for_bar(row: dict[str, Any], position: _Position) -> tuple[float, str] | None:
    low = float(row["low"])
    high = float(row["high"])
    if position.direction == "long":
        stop_hit = low <= position.stop
        target_hit = high >= position.target
    else:
        stop_hit = high >= position.stop
        target_hit = low <= position.target

    if stop_hit:
        return position.stop, "stop_loss"
    if target_hit:
        return position.target, "target"
    return None


def _closed_row(
    position: _Position,
    exit_time: datetime,
    session_date: date,
    exit_price: float,
    exit_reason: str,
    config: StrategyConfig,
) -> dict[str, Any]:
    multiplier = 1.0 if position.direction == "long" else -1.0
    gross_pnl = (
        (exit_price - position.entry_price) * multiplier * config.point_value * position.contracts
    )
    total_costs = position.risk_decision.estimated_costs
    net_pnl = gross_pnl - total_costs
    risk = position.risk_decision.gross_risk
    r_multiple = gross_pnl / risk if risk else 0.0
    adjusted_exit = exit_price - config.cost_model.slippage_points * multiplier
    return {
        "signal_time": position.signal_time,
        "entry_time": position.entry_time,
        "exit_time": exit_time,
        "setup": position.setup,
        "direction": position.direction,
        "session_date": session_date,
        "contracts": position.contracts,
        "stop": position.stop,
        "target": position.target,
        "entry_price": position.entry_price,
        "adjusted_entry_price": position.adjusted_entry_price,
        "exit_price": exit_price,
        "adjusted_exit_price": adjusted_exit,
        "gross_pnl": gross_pnl,
        "total_costs": total_costs,
        "net_pnl": net_pnl,
        "r_multiple": r_multiple,
        "status": "closed",
        "exit_reason": exit_reason,
        "rejection_reason": None,
    }


def _rejected_row(
    signal: dict[str, Any],
    entry_time: datetime | None,
    setup: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "signal_time": signal["timestamp_ny"],
        "entry_time": entry_time,
        "exit_time": None,
        "setup": setup,
        "direction": signal.get("direction"),
        "session_date": _session_date(signal),
        "contracts": 0,
        "stop": None,
        "target": None,
        "entry_price": None,
        "adjusted_entry_price": None,
        "exit_price": None,
        "adjusted_exit_price": None,
        "gross_pnl": 0.0,
        "total_costs": 0.0,
        "net_pnl": 0.0,
        "r_multiple": 0.0,
        "status": "rejected",
        "exit_reason": None,
        "rejection_reason": reason,
    }


def _guard_rejection_reason(
    daily_pnl: float, consecutive_losses: int, config: StrategyConfig
) -> str:
    if daily_pnl <= -config.daily_stop:
        return "daily_loss_limit"
    if consecutive_losses >= config.max_consecutive_losses:
        return "consecutive_loss_limit"
    return "daily_guard_blocked"


def _risk_decision(config: StrategyConfig) -> RiskDecision:
    decision = calculate_position_size(
        stop_points=config.stop_points,
        requested_contracts=config.max_contracts,
        costs=config.cost_model,
    )
    if decision.accepted and decision.total_risk > config.max_trade_risk:
        return RiskDecision(
            accepted=False,
            contracts=0,
            gross_risk=decision.gross_risk,
            estimated_costs=decision.estimated_costs,
            total_risk=decision.total_risk,
            reason=(
                f"total risk {decision.total_risk:.2f} exceeds configured "
                f"{config.max_trade_risk:.2f} USD ceiling"
            ),
        )
    if decision.accepted and decision.contracts < config.min_contracts:
        return RiskDecision(
            accepted=False,
            contracts=0,
            gross_risk=decision.gross_risk,
            estimated_costs=decision.estimated_costs,
            total_risk=decision.total_risk,
            reason="sized contracts below configured minimum",
        )
    return decision


def run_backtest(bars: pl.DataFrame, config: StrategyConfig) -> pl.DataFrame:
    """Run a chronological, one-position-at-a-time MNQ backtest.

    Signal rows describe a confirmation at their own close. The next
    chronological bar supplies the raw entry observation. OHLC exits are
    evaluated from that bar onward, with the stop taking precedence if both
    configured levels are touched in one bar.
    """
    config.validate_fixed_contract()
    prepared = _prepare_bars(bars, config)
    timestamp_dtype = _timestamp_dtype(prepared)
    if prepared.height == 0:
        return _empty_results(timestamp_dtype)

    rows = prepared.to_dicts()
    guard = DailyRiskGuard(
        max_daily_loss=config.daily_stop,
        max_consecutive_losses=config.max_consecutive_losses,
    )
    results: list[dict[str, Any]] = []
    position: _Position | None = None
    guard_date: date | None = None
    daily_pnl = 0.0
    consecutive_losses = 0
    risk_decision = _risk_decision(config)

    for index, row in enumerate(rows):
        row_date = _session_date(row)
        if guard_date is None:
            guard_date = row_date
        elif row_date != guard_date:
            guard.reset_day()
            guard_date = row_date
            daily_pnl = 0.0
            consecutive_losses = 0

        if position is not None and index >= position.entry_index:
            exit_details = _exit_for_bar(row, position)
            if exit_details is not None:
                exit_price, exit_reason = exit_details
                closed = _closed_row(
                    position,
                    row["timestamp_ny"],
                    row_date,
                    exit_price,
                    exit_reason,
                    config,
                )
                results.append(closed)
                daily_pnl += closed["net_pnl"]
                if closed["net_pnl"] < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                guard.record_trade(closed["net_pnl"])
                position = None

        if not row["signal"] or not row["bar_closed"]:
            continue

        setup = row.get("signal_type") or "unknown"
        next_index = index + 1
        next_time = rows[next_index]["timestamp_ny"] if next_index < len(rows) else None
        if position is not None:
            results.append(_rejected_row(row, next_time, setup, "position_already_open"))
            continue
        if row.get("_entry_blocked_by_unclosed") or next_index >= len(rows):
            results.append(_rejected_row(row, None, setup, "no_eligible_entry_bar"))
            continue
        if not isinstance(row.get("direction"), str) or row["direction"] not in {"long", "short"}:
            results.append(_rejected_row(row, next_time, setup, "invalid_direction"))
            continue
        if not risk_decision.accepted:
            results.append(_rejected_row(row, next_time, setup, "position_size_rejected"))
            continue
        if not guard.can_trade():
            reason = _guard_rejection_reason(daily_pnl, consecutive_losses, config)
            results.append(_rejected_row(row, next_time, setup, reason))
            continue

        entry_price = float(rows[next_index]["open"])
        direction = row["direction"]
        multiplier = 1.0 if direction == "long" else -1.0
        adjusted_entry_price = entry_price + config.cost_model.slippage_points * multiplier
        position = _Position(
            signal_time=row["timestamp_ny"],
            entry_index=next_index,
            entry_time=next_time,
            setup=str(setup),
            direction=direction,
            contracts=risk_decision.contracts,
            stop=_directional_price(
                entry_price,
                direction,
                config.stop_points,
                is_target=False,
            ),
            target=_directional_price(
                entry_price,
                direction,
                config.target_points,
                is_target=True,
            ),
            entry_price=entry_price,
            adjusted_entry_price=adjusted_entry_price,
            risk_decision=risk_decision,
        )

    if position is not None:
        last_closed_row = next(
            (candidate for candidate in reversed(rows) if candidate["bar_closed"]),
            None,
        )
        if last_closed_row is not None:
            closed = _closed_row(
                position,
                last_closed_row["timestamp_ny"],
                _session_date(last_closed_row),
                float(last_closed_row["close"]),
                "end_of_data",
                config,
            )
            results.append(closed)

    results.sort(key=lambda result: result["signal_time"])
    return _frame_from_rows(results, timestamp_dtype)


__all__ = ["RESULT_COLUMNS", "run_backtest"]
