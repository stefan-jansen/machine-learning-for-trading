"""Chronological walk-forward evaluation and metric reporting for MNQ."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

import polars as pl

from .backtest import run_backtest
from .config import StrategyConfig


@dataclass(frozen=True)
class WalkForwardWindow:
    """Non-overlapping train and test date/datetime boundaries."""

    train_start: date | datetime | str
    train_end: date | datetime | str
    test_start: date | datetime | str
    test_end: date | datetime | str

    def __post_init__(self) -> None:
        values = {
            "train_start": _boundary(self.train_start),
            "train_end": _boundary(self.train_end),
            "test_start": _boundary(self.test_start),
            "test_end": _boundary(self.test_end),
        }
        if _compare_boundaries(values["train_start"], values["train_end"], right_is_end=True) > 0:
            raise ValueError("train_start must be before or equal to train_end")
        if _compare_boundaries(values["test_start"], values["test_end"], right_is_end=True) > 0:
            raise ValueError("test_start must be before or equal to test_end")
        if _compare_boundaries(values["train_end"], values["test_start"], left_is_end=True) >= 0:
            raise ValueError("train and test windows must be chronologically non-overlapping")
        object.__setattr__(self, "train_start", values["train_start"])
        object.__setattr__(self, "train_end", values["train_end"])
        object.__setattr__(self, "test_start", values["test_start"])
        object.__setattr__(self, "test_end", values["test_end"])

    def to_dict(self) -> dict[str, str]:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def _boundary(value: date | datetime | str) -> date | datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("walk-forward boundaries must be date, datetime, or ISO strings")
    if len(value) == 10:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"invalid walk-forward boundary: {value!r}") from exc
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"invalid walk-forward boundary: {value!r}") from exc


def _boundary_datetime(value: date | datetime, *, is_end: bool = False) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time.max if is_end else time.min)


def _compare_boundaries(
    left: date | datetime,
    right: date | datetime,
    *,
    left_is_end: bool = False,
    right_is_end: bool = False,
) -> int:
    left_dt = _boundary_datetime(left, is_end=left_is_end)
    right_dt = _boundary_datetime(right, is_end=right_is_end)
    if (left_dt.tzinfo is None) != (right_dt.tzinfo is None):
        timezone = right_dt.tzinfo if left_dt.tzinfo is None else left_dt.tzinfo
        if left_dt.tzinfo is None:
            left_dt = left_dt.replace(tzinfo=timezone)
        else:
            right_dt = right_dt.replace(tzinfo=timezone)
    elif left_dt.tzinfo is not None and right_dt.tzinfo is not None:
        right_dt = right_dt.astimezone(left_dt.tzinfo)
    return (left_dt > right_dt) - (left_dt < right_dt)


def _as_comparable(
    value: date | datetime,
    reference: datetime,
    *,
    is_end: bool = False,
) -> datetime:
    boundary = _boundary_datetime(value, is_end=is_end)
    if reference.tzinfo is None:
        return boundary.replace(tzinfo=None)
    if boundary.tzinfo is None:
        return boundary.replace(tzinfo=reference.tzinfo)
    return boundary.astimezone(reference.tzinfo)


def _within_boundaries(
    timestamps: list[datetime], start: date | datetime, end: date | datetime
) -> list[bool]:
    return [
        _as_comparable(start, timestamp) <= timestamp <= _as_comparable(end, timestamp, is_end=True)
        for timestamp in timestamps
    ]


def _validate_windows(windows: list[WalkForwardWindow]) -> None:
    if not windows:
        raise ValueError("at least one walk-forward window is required")
    for window in windows:
        if (
            _compare_boundaries(
                window.train_start,
                window.train_end,
                right_is_end=True,
            )
            > 0
            or _compare_boundaries(
                window.test_start,
                window.test_end,
                right_is_end=True,
            )
            > 0
        ):
            raise ValueError("walk-forward boundaries must be chronological")
        if (
            _compare_boundaries(
                window.train_end,
                window.test_start,
                left_is_end=True,
            )
            >= 0
        ):
            raise ValueError("train and test windows must be chronologically non-overlapping")
    for previous, current in zip(windows, windows[1:]):
        if (
            _compare_boundaries(
                previous.test_end,
                current.train_start,
                left_is_end=True,
            )
            >= 0
        ):
            raise ValueError("walk-forward windows must be chronologically non-overlapping")


def _closed_trades(results: pl.DataFrame) -> pl.DataFrame:
    return results.filter(pl.col("status") == "closed")


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _breach_counts(trades: pl.DataFrame, config: StrategyConfig) -> tuple[int, int]:
    if trades.is_empty():
        return 0, 0
    daily_pnl: dict[date, float] = {}
    daily_breaches: set[date] = set()
    consecutive_losses = 0
    consecutive_breaches = 0
    for trade in trades.to_dicts():
        trade_date = trade["entry_time"].date()
        daily_pnl[trade_date] = daily_pnl.get(trade_date, 0.0) + float(trade["net_pnl"])
        if daily_pnl[trade_date] <= -config.daily_stop:
            daily_breaches.add(trade_date)
        if trade["net_pnl"] < 0:
            consecutive_losses += 1
            if consecutive_losses >= config.max_consecutive_losses:
                consecutive_breaches += 1
        else:
            consecutive_losses = 0
    return len(daily_breaches), consecutive_breaches


def _per_setup(trades: pl.DataFrame) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    if trades.is_empty():
        return output
    for setup, group in trades.partition_by("setup", as_dict=True).items():
        name = setup[0] if isinstance(setup, tuple) else setup
        pnls = [float(value) for value in group["net_pnl"]]
        output[str(name)] = {
            "trade_count": len(pnls),
            "net_pnl": sum(pnls),
            "win_rate": sum(value > 0 for value in pnls) / len(pnls),
            "average_r": float(group["r_multiple"].mean() or 0.0),
        }
    return output


def _metrics(results: pl.DataFrame, config: StrategyConfig) -> dict[str, Any]:
    trades = _closed_trades(results)
    if trades.is_empty():
        return {
            "trade_count": 0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "daily_loss_breaches": 0,
            "consecutive_loss_breaches": 0,
            "average_r": 0.0,
            "cost_share": 0.0,
            "per_setup": {},
        }

    pnls = [float(value) for value in trades["net_pnl"]]
    gross_pnls = [float(value) for value in trades["gross_pnl"]]
    costs = sum(float(value) for value in trades["total_costs"])
    wins = sum(value > 0 for value in pnls)
    positive = sum(value for value in pnls if value > 0)
    negative = -sum(value for value in pnls if value < 0)
    profit_factor = positive / negative if negative else (math.inf if positive else 0.0)
    gross_total = sum(gross_pnls)
    cost_share = costs / abs(gross_total) if gross_total else 0.0
    daily_breaches, consecutive_breaches = _breach_counts(trades, config)
    return {
        "trade_count": len(pnls),
        "net_pnl": sum(pnls),
        "expectancy": sum(pnls) / len(pnls),
        "win_rate": wins / len(pnls),
        "profit_factor": profit_factor,
        "max_drawdown": _max_drawdown(pnls),
        "daily_loss_breaches": daily_breaches,
        "consecutive_loss_breaches": consecutive_breaches,
        "average_r": float(trades["r_multiple"].mean() or 0.0),
        "cost_share": cost_share,
        "per_setup": _per_setup(trades),
    }


def _empty_results_from_bars(bars: pl.DataFrame, config: StrategyConfig) -> pl.DataFrame:
    return run_backtest(bars.head(0), config)


def walk_forward_evaluate(
    bars: pl.DataFrame,
    windows: list[WalkForwardWindow],
    config: StrategyConfig,
) -> dict[str, Any]:
    """Evaluate fixed-contract signals on chronological test windows.

    Training and validation rows are used only to document the selected
    window and policy. No threshold or parameter is selected from test data.
    """
    config.validate_fixed_contract()
    _validate_windows(windows)
    if not isinstance(bars, pl.DataFrame):
        raise TypeError("bars must be a polars.DataFrame")

    timestamp_column = "timestamp_ny" if "timestamp_ny" in bars.columns else "timestamp"
    if timestamp_column not in bars.columns:
        raise ValueError("required input column missing: timestamp_ny")
    raw_timestamps = bars[timestamp_column].to_list()
    if raw_timestamps and not isinstance(raw_timestamps[0], datetime):
        raise ValueError("walk-forward input timestamps must be datetime values")
    chronological = sorted(raw_timestamps)
    if raw_timestamps != chronological:
        raise ValueError("walk-forward input must be chronological")

    test_frames: list[pl.DataFrame] = []
    window_reports: list[dict[str, Any]] = []
    for window in windows:
        timestamps = bars[timestamp_column].to_list()
        train_mask = _within_boundaries(timestamps, window.train_start, window.train_end)
        test_mask = _within_boundaries(timestamps, window.test_start, window.test_end)
        train_rows = bars.filter(pl.Series("train", train_mask, dtype=pl.Boolean))
        test_rows = bars.filter(pl.Series("test", test_mask, dtype=pl.Boolean))
        if train_rows.height and test_rows.height:
            if max(train_rows[timestamp_column].to_list()) >= min(
                test_rows[timestamp_column].to_list()
            ):
                raise ValueError("test rows must occur after train rows")
        test_results = run_backtest(test_rows, config)
        test_frames.append(test_results)
        window_reports.append(
            {
                **window.to_dict(),
                "train_rows": train_rows.height,
                "test_rows": test_rows.height,
                "test_trade_count": int(test_results.filter(pl.col("status") == "closed").height),
                "train_last_timestamp": (
                    max(train_rows[timestamp_column].to_list()) if train_rows.height else None
                ),
                "test_first_timestamp": (
                    min(test_rows[timestamp_column].to_list()) if test_rows.height else None
                ),
            }
        )

    if test_frames:
        non_empty = [frame for frame in test_frames if frame.height]
        all_results = (
            pl.concat(non_empty, how="vertical")
            if non_empty
            else _empty_results_from_bars(bars, config)
        )
    else:  # pragma: no cover - _validate_windows prevents this
        all_results = _empty_results_from_bars(bars, config)
    all_results = all_results.sort("signal_time")
    metrics = _metrics(all_results, config)
    lookahead_checks: list[bool] = []
    for window in windows:
        lookahead_checks.append(
            _compare_boundaries(
                window.train_end,
                window.test_start,
                left_is_end=True,
            )
            < 0
        )
    lookahead_check = (
        config.validate_fixed_contract()
        and all(lookahead_checks)
        and all(
            report["train_last_timestamp"] is None
            or report["test_first_timestamp"] is None
            or report["train_last_timestamp"] < report["test_first_timestamp"]
            for report in window_reports
        )
    )
    return {
        "test_results": all_results.to_dicts(),
        **metrics,
        "config_hash": config.config_hash,
        "lookahead_check": bool(lookahead_check),
        "windows": window_reports,
        "selection_policy": {
            "threshold_selection": "fixed_config_only",
            "train_validation_usage": "reporting_and_sensitivity_metadata_only",
            "test_data_used_for_selection": False,
            "window_count": len(windows),
        },
    }


__all__ = ["WalkForwardWindow", "walk_forward_evaluate"]
