"""Objective signal definitions for the MNQ intraday strategy."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

import polars as pl

REJECTION_WICK_RATIO = 2.0
REJECTION_CLOSE_PCT = 0.30
MOMENTUM_BODY_RANGE = 0.60
MOMENTUM_CLOSE_PCT = 0.80
MOMENTUM_LOOKBACK = 6
MOMENTUM_MULT = 1.5


def _validate_bars(bars: pl.DataFrame) -> None:
    required = {
        "timestamp_ny",
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "bar_closed",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"required signal columns missing: {', '.join(missing)}")
    if bars.schema["bar_closed"] != pl.Boolean or bars.get_column("bar_closed").null_count():
        raise ValueError("bar_closed must contain only explicit boolean values")


def _output(bars: pl.DataFrame, signal_type: str) -> pl.DataFrame:
    height = bars.height
    timestamp_dtype = bars.schema["timestamp_ny"]
    return bars.with_columns(
        pl.Series("signal", [False] * height, dtype=pl.Boolean),
        pl.Series("direction", [None] * height, dtype=pl.Utf8),
        pl.Series("entry_time", [None] * height, dtype=timestamp_dtype),
        pl.Series("signal_type", [signal_type] * height, dtype=pl.Utf8),
    )


def _close_direction(current: float, previous: float) -> str | None:
    if current > previous:
        return "long"
    if current < previous:
        return "short"
    return None


def _mnq_direction(
    row: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    index: int | None,
) -> str | None:
    if rows is None or index is None or index == 0:
        return None
    return _close_direction(float(row["close"]), float(rows[index - 1]["close"]))


def _es_coherent(
    row: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    index: int | None = None,
) -> bool:
    mnq_direction = _mnq_direction(row, rows, index)
    if "es_direction" in row:
        return mnq_direction is not None and row["es_direction"] == mnq_direction

    if "es_close" not in row:
        return True
    if mnq_direction is None:
        return False

    previous = rows[index - 1]
    previous_es_close = row.get("es_prev_close", previous.get("es_close"))
    if previous_es_close is None:
        return False
    es_direction = _close_direction(float(row["es_close"]), float(previous_es_close))
    return es_direction == mnq_direction


def _candle_metrics(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    body = abs(float(row["close"]) - float(row["open"]))
    candle_range = float(row["high"]) - float(row["low"])
    lower_wick = min(float(row["open"]), float(row["close"])) - float(row["low"])
    upper_wick = float(row["high"]) - max(float(row["open"]), float(row["close"]))
    close_pct = (
        (float(row["close"]) - float(row["low"])) / candle_range if candle_range > 0 else 0.0
    )
    return body, candle_range, lower_wick, upper_wick, close_pct


def _momentum_direction(rows: list[dict[str, Any]], index: int) -> str | None:
    row = rows[index]
    if not row["bar_closed"]:
        return None
    body, candle_range, _, _, close_pct = _candle_metrics(row)
    if candle_range <= 0 or body / candle_range < MOMENTUM_BODY_RANGE:
        return None
    if close_pct < MOMENTUM_CLOSE_PCT and close_pct > (1.0 - MOMENTUM_CLOSE_PCT):
        return None
    prior = [
        abs(float(candidate["close"]) - float(candidate["open"]))
        for candidate in rows[max(0, index - MOMENTUM_LOOKBACK) : index]
        if candidate["bar_closed"]
    ]
    if len(prior) != MOMENTUM_LOOKBACK:
        return None
    if body < MOMENTUM_MULT * (sum(prior) / len(prior)):
        return None
    if close_pct >= MOMENTUM_CLOSE_PCT:
        return "long"
    if close_pct <= (1.0 - MOMENTUM_CLOSE_PCT):
        return "short"
    return None


def detect_midnight_rejection(bars: pl.DataFrame) -> pl.DataFrame:
    """Detect the first closed-bar rejection of the exact 00:00 New York open."""
    _validate_bars(bars)
    result = _output(bars, "rejection")
    rows = bars.to_dicts()
    levels: dict[Any, tuple[datetime, float]] = {}
    for row in rows:
        if row["timestamp_ny"].time() == time(0, 0):
            levels.setdefault(row["session_date"], (row["timestamp_ny"], float(row["open"])))

    emitted: set[Any] = set()
    for index, row in enumerate(rows):
        level = levels.get(row["session_date"])
        if (
            not row["bar_closed"]
            or level is None
            or row["timestamp_ny"] <= level[0]
            or row["session_date"] in emitted
        ):
            continue
        body, candle_range, lower_wick, upper_wick, close_pct = _candle_metrics(row)
        if candle_range <= 0 or body <= 0:
            continue
        midnight = level[1]
        max_wick = max(lower_wick, upper_wick)
        bullish = (
            max_wick / body >= REJECTION_WICK_RATIO
            and lower_wick > 0
            and float(row["low"]) <= midnight
            and float(row["close"]) > midnight
            and close_pct >= (1.0 - REJECTION_CLOSE_PCT)
        )
        bearish = (
            max_wick / body >= REJECTION_WICK_RATIO
            and upper_wick > 0
            and float(row["high"]) >= midnight
            and float(row["close"]) < midnight
            and close_pct <= REJECTION_CLOSE_PCT
        )
        direction = "long" if bullish else "short" if bearish else None
        if direction is None or not _es_coherent(row, rows, index):
            continue
        result[index, "signal"] = True
        result[index, "direction"] = direction
        result[index, "entry_time"] = row["timestamp_ny"]
        emitted.add(row["session_date"])
    return result


def detect_momentum(bars: pl.DataFrame) -> pl.DataFrame:
    """Return momentum helper signals; this is not an allowed standalone setup."""
    _validate_bars(bars)
    result = _output(bars, "momentum")
    rows = bars.to_dicts()
    session_start: dict[Any, int] = {}
    for index, row in enumerate(rows):
        start = session_start.setdefault(row["session_date"], index)
        direction = _momentum_direction(rows, index) if index - start >= MOMENTUM_LOOKBACK else None
        if direction is not None and _es_coherent(row, rows, index):
            result[index, "signal"] = True
            result[index, "direction"] = direction
            result[index, "entry_time"] = row["timestamp_ny"]
    return result


def _zone_touch(row: dict[str, Any], zone: dict[str, Any], direction: str) -> bool:
    low = float(row["low"])
    high = float(row["high"])
    if direction == "long":
        edge = float(zone["high"])
        return low <= edge <= high
    edge = float(zone["low"])
    return low <= edge <= high


def _lvn_signal_for_zone(
    rows: list[dict[str, Any]], zone: dict[str, Any]
) -> tuple[int, str] | None:
    state = "unbroken"
    direction: str | None = None
    break_index: int | None = None
    zone_low = float(zone["low"])
    zone_high = float(zone["high"])
    for index, row in enumerate(rows):
        if not row["bar_closed"]:
            continue
        close = float(row["close"])
        if state == "unbroken":
            if close > zone_high:
                state, direction, break_index = "broken", "long", index
            elif close < zone_low:
                state, direction, break_index = "broken", "short", index
        elif state == "broken" and direction is not None:
            if (
                direction == "long"
                and close < zone_low
                or direction == "short"
                and close > zone_high
            ):
                state, direction, break_index = "unbroken", None, None
            elif (
                break_index is not None
                and index > break_index
                and _zone_touch(row, zone, direction)
            ):
                state = "retested"
        elif state == "retested" and direction is not None:
            if (
                direction == "long"
                and close < zone_low
                or direction == "short"
                and close > zone_high
            ):
                state, direction = "unbroken", None
            elif (
                direction == "long"
                and close > zone_high
                or direction == "short"
                and close < zone_low
            ):
                return index, direction
    return None


def detect_lvn_break_retest(bars: pl.DataFrame) -> pl.DataFrame:
    """Detect a closed LVN break, later edge retest, and second confirmation."""
    _validate_bars(bars)
    result = _output(bars, "lvn_break_retest")
    rows = bars.to_dicts()
    by_session: dict[Any, list[dict[str, Any]]] = {}
    positions: dict[Any, list[int]] = {}
    for index, row in enumerate(rows):
        by_session.setdefault(row["session_date"], []).append(row)
        positions.setdefault(row["session_date"], []).append(index)

    for session_date, session_rows in by_session.items():
        zone_values = [
            row["previous_lvn_zones"]
            for row in session_rows
            if row.get("previous_lvn_zones") is not None
        ]
        if not zone_values:
            continue
        raw_zones = zone_values[0]
        zones = [raw_zones] if isinstance(raw_zones, dict) else list(raw_zones)
        if not zones:
            continue
        candidate = None
        for zone in zones:
            found = _lvn_signal_for_zone(session_rows, zone)
            if found is not None and (candidate is None or found[0] < candidate[0]):
                candidate = found
        if candidate is None:
            continue
        local_index, direction = candidate
        output_index = positions[session_date][local_index]
        row = session_rows[local_index]
        if not _es_coherent(row, session_rows, local_index):
            continue
        result[output_index, "signal"] = True
        result[output_index, "direction"] = direction
        result[output_index, "entry_time"] = row["timestamp_ny"]
    return result


def detect_10am_confirmation(bars: pl.DataFrame) -> pl.DataFrame:
    """Evaluate the closed 10:00-10:05 bar and emit its 10:05 entry time."""
    _validate_bars(bars)
    result = _output(bars, "10am")
    rows = bars.to_dicts()
    session_start: dict[Any, int] = {}
    for index, row in enumerate(rows):
        if row["timestamp_ny"].time() != time(10, 0) or not row["bar_closed"]:
            session_start.setdefault(row["session_date"], index)
            continue
        start = session_start.setdefault(row["session_date"], index)
        direction = _momentum_direction(rows, index) if index - start >= MOMENTUM_LOOKBACK else None
        if direction is None or not _es_coherent(row, rows, index):
            continue
        result[index, "signal"] = True
        result[index, "direction"] = direction
        result[index, "entry_time"] = row["timestamp_ny"] + timedelta(minutes=5)
    return result


def combine_allowed_signals(bars: pl.DataFrame) -> pl.DataFrame:
    """Combine only Midnight rejection, LVN, and 10:00 with fixed precedence."""
    _validate_bars(bars)
    rejection = detect_midnight_rejection(bars)
    lvn = detect_lvn_break_retest(bars)
    ten_am = detect_10am_confirmation(bars)
    result = _output(bars, "allowed")
    for index in range(bars.height):
        for source in (rejection, lvn, ten_am):
            if not source[index, "signal"]:
                continue
            result[index, "signal"] = True
            result[index, "direction"] = source[index, "direction"]
            result[index, "entry_time"] = source[index, "entry_time"]
            result[index, "signal_type"] = source[index, "signal_type"]
            break
    return result
