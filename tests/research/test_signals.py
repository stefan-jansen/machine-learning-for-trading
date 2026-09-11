"""Focused tests for the approved MNQ objective signal contracts."""

from datetime import UTC, datetime, timedelta

import polars as pl

from research.mnq_strategy.signals import (
    combine_allowed_signals,
    detect_10am_confirmation,
    detect_lvn_break_retest,
    detect_midnight_rejection,
    detect_momentum,
)

UTC = UTC


def _frame(rows, *, zones=None, es_direction=None, es_close=None):
    timestamps_ny = [row["timestamp_ny"] for row in rows]
    timestamps = [
        value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        for value in timestamps_ny
    ]
    zone_values = zones if zones is not None else [None] * len(rows)
    data = {
        "timestamp": timestamps,
        "timestamp_ny": timestamps_ny,
        "session_date": [row["session_date"] for row in rows],
        "session_type": [row.get("session_type", "rth") for row in rows],
        "open": [row["open"] for row in rows],
        "high": [row["high"] for row in rows],
        "low": [row["low"] for row in rows],
        "close": [row["close"] for row in rows],
        "volume": [row.get("volume", 100) for row in rows],
        "bar_closed": [row.get("bar_closed", True) for row in rows],
        "previous_lvn_zones": zone_values,
    }
    if es_direction is not None:
        data["es_direction"] = es_direction
    if es_close is not None:
        data["es_close"] = es_close
    return pl.DataFrame(data)


def _row(
    ny, open_, high, low, close, *, session_date=None, session_type="rth", closed=True, volume=100
):
    return {
        "timestamp_ny": ny,
        "session_date": session_date or ny.date(),
        "session_type": session_type,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "bar_closed": closed,
        "volume": volume,
    }


def _midnight_rejection_rows(*, bullish=True, closed=True):
    level = _row(datetime(2026, 1, 5, 0, 0), 100.0, 100.5, 99.5, 100.25)
    rejection = (
        _row(datetime(2026, 1, 5, 9, 55), 103.0, 103.5, 99.5, 103.5, closed=closed)
        if bullish
        else _row(datetime(2026, 1, 5, 9, 55), 97.0, 100.5, 96.5, 96.5, closed=closed)
    )
    return [level, rejection]


def _momentum_rows(*, closed=True):
    rows = []
    for index in range(6):
        offset = index * 0.05
        rows.append(
            _row(
                datetime(2026, 1, 5, 9, 30) + timedelta(minutes=5 * index),
                100.0 + offset,
                100.25 + offset,
                99.75 + offset,
                100.1 + offset,
                volume=50,
            )
        )
    rows.append(
        _row(
            datetime(2026, 1, 5, 10, 0),
            100.0,
            102.0,
            99.8,
            101.8,
            closed=closed,
            volume=300,
        )
    )
    return rows


def _lvn_rows(direction="long", *, closed=True):
    if direction == "long":
        values = [
            (9, 30, 104.0, 106.0, 103.5, 106.0),
            (9, 35, 106.0, 106.5, 105.8, 106.2),
            (9, 40, 106.0, 106.5, 105.5, 105.5),
            (9, 45, 106.0, 107.0, 105.8, 106.8),
        ]
        zone = {"low": 105.25, "high": 105.75}
    else:
        values = [
            (9, 30, 106.0, 106.5, 104.0, 104.0),
            (9, 35, 104.0, 105.0, 103.5, 103.8),
            (9, 40, 104.0, 104.8, 104.5, 104.5),
            (9, 45, 104.0, 104.5, 103.0, 103.2),
        ]
        zone = {"low": 104.25, "high": 104.75}
    rows = [
        _row(datetime(2026, 1, 6, hour, minute), open_, high, low, close, closed=closed)
        for hour, minute, open_, high, low, close in values
    ]
    return rows, [zone] * len(rows)


def _ten_am_rows(*, closed=True):
    rows = _momentum_rows(closed=True)
    rows[-1]["timestamp_ny"] = datetime(2026, 1, 5, 10, 0)
    rows[-1]["bar_closed"] = closed
    return rows


def test_midnight_rejection_requires_closed_bar_and_exact_level():
    signal = detect_midnight_rejection(_frame(_midnight_rejection_rows()))
    rows = signal.filter(pl.col("signal"))
    assert rows.height == 1
    assert rows["direction"][0] == "long"
    assert rows["entry_time"][0] == datetime(2026, 1, 5, 9, 55)

    unclosed = detect_midnight_rejection(_frame(_midnight_rejection_rows(closed=False)))
    assert unclosed.filter(pl.col("signal")).height == 0

    no_exact_level = _frame(
        [
            _row(datetime(2026, 1, 5, 0, 5), 100.0, 100.5, 99.5, 100.25),
            _midnight_rejection_rows()[1],
        ]
    )
    assert detect_midnight_rejection(no_exact_level).filter(pl.col("signal")).height == 0


def test_midnight_rejection_requires_wick_to_cross_level():
    rows = [
        _row(datetime(2026, 1, 5, 0, 0), 100.0, 100.5, 99.5, 100.25),
        _row(datetime(2026, 1, 5, 9, 55), 103.0, 103.5, 100.5, 103.5),
    ]
    assert detect_midnight_rejection(_frame(rows)).filter(pl.col("signal")).height == 0


def test_midnight_rejection_is_bearish_when_upper_wick_crosses_level():
    result = detect_midnight_rejection(_frame(_midnight_rejection_rows(bullish=False)))
    rows = result.filter(pl.col("signal"))
    assert rows.height == 1
    assert rows["direction"][0] == "short"


def test_only_first_midnight_rejection_per_session_is_emitted():
    rows = _midnight_rejection_rows()
    rows.append(_row(datetime(2026, 1, 5, 10, 0), 104.0, 104.5, 100.0, 104.5))
    result = detect_midnight_rejection(_frame(rows))
    assert result.filter(pl.col("signal")).height == 1


def test_momentum_requires_six_prior_closed_bars():
    result = detect_momentum(_frame(_momentum_rows()))
    rows = result.filter(pl.col("signal"))
    assert rows.height == 1
    assert rows["direction"][0] == "long"

    short_history = _frame(_momentum_rows()[-1:])
    assert detect_momentum(short_history).filter(pl.col("signal")).height == 0
    assert (
        detect_momentum(_frame(_momentum_rows(closed=False))).filter(pl.col("signal")).height == 0
    )


def test_lvn_requires_break_retest_and_second_close():
    rows, zones = _lvn_rows()
    result = detect_lvn_break_retest(_frame(rows, zones=zones))
    rows_out = result.filter(pl.col("signal"))
    assert rows_out.height == 1
    assert rows_out["direction"][0] == "long"
    assert rows_out["entry_time"][0] == datetime(2026, 1, 6, 9, 45)


def test_lvn_supports_bearish_direction_and_rejects_intrabar_confirmation():
    rows, zones = _lvn_rows("short")
    result = detect_lvn_break_retest(_frame(rows, zones=zones))
    assert result.filter(pl.col("signal"))["direction"][0] == "short"

    rows[-1]["bar_closed"] = False
    assert detect_lvn_break_retest(_frame(rows, zones=zones)).filter(pl.col("signal")).height == 0


def test_lvn_retest_must_touch_zone_and_opposite_close_invalidates():
    rows, zones = _lvn_rows()
    rows[2]["low"] = 106.0
    assert detect_lvn_break_retest(_frame(rows, zones=zones)).filter(pl.col("signal")).height == 0

    rows, zones = _lvn_rows()
    rows[2]["close"] = 104.0
    assert detect_lvn_break_retest(_frame(rows, zones=zones)).filter(pl.col("signal")).height == 0


def test_lvn_retest_must_be_later_than_break():
    rows, zones = _lvn_rows()
    rows[0]["open"] = 105.0
    rows[0]["high"] = 105.0
    rows[0]["low"] = 104.5
    rows[0]["close"] = 105.0
    rows[1]["low"] = 105.75
    rows[2]["low"] = 106.0
    rows[2]["close"] = 106.8
    assert detect_lvn_break_retest(_frame(rows, zones=zones)).filter(pl.col("signal")).height == 0


def test_10am_uses_the_closed_10_00_bar_and_emits_at_10_05():
    result = detect_10am_confirmation(_frame(_ten_am_rows()))
    rows = result.filter(pl.col("signal"))
    assert rows.height == 1
    assert rows["timestamp_ny"][0] == datetime(2026, 1, 5, 10, 0)
    assert rows["entry_time"][0] == datetime(2026, 1, 5, 10, 5)
    assert rows["direction"][0] == "long"

    assert (
        detect_10am_confirmation(_frame(_ten_am_rows(closed=False))).filter(pl.col("signal")).height
        == 0
    )


def test_10am_requires_momentum_not_a_midnight_level_break():
    rows = [
        _row(datetime(2026, 1, 5, 0, 0), 200.0, 201.0, 199.0, 200.5),
        _row(datetime(2026, 1, 5, 10, 0), 201.0, 202.0, 200.5, 201.5),
    ]
    assert detect_10am_confirmation(_frame(rows)).filter(pl.col("signal")).height == 0


def test_es_direction_filter_blocks_divergence():
    rows = _midnight_rejection_rows()
    frame = _frame(rows, es_direction=["short", "short"])
    assert detect_midnight_rejection(frame).filter(pl.col("signal")).height == 0

    frame = _frame(rows, es_direction=["short", "long"])
    assert detect_midnight_rejection(frame).filter(pl.col("signal")).height == 1


def test_combine_allows_only_approved_setups_and_uses_precedence():
    rejection = detect_midnight_rejection(_frame(_midnight_rejection_rows()))
    combined = combine_allowed_signals(_frame(_midnight_rejection_rows()))
    rows = combined.filter(pl.col("signal"))
    assert rows.height == 1
    assert rows["signal_type"][0] == "rejection"
    assert rejection.filter(pl.col("signal")).height == 1

    momentum_only = _frame(_momentum_rows())
    combined_momentum = combine_allowed_signals(momentum_only)
    momentum_signals = combined_momentum.filter(pl.col("signal"))
    assert momentum_signals.height == 1
    assert momentum_signals["signal_type"][0] == "10am"


def test_es_close_coherence_filter_uses_the_same_bar_direction():
    rows = _midnight_rejection_rows()
    coherent = _frame(rows, es_close=[100.0, 101.0])
    assert detect_midnight_rejection(coherent).filter(pl.col("signal")).height == 1

    divergent = _frame(rows, es_close=[100.0, 99.0])
    assert detect_midnight_rejection(divergent).filter(pl.col("signal")).height == 0


def test_10am_does_not_evaluate_a_10_05_bar_as_the_observation_bar():
    rows = _momentum_rows()
    rows[-1]["timestamp_ny"] = datetime(2026, 1, 5, 10, 5)
    assert detect_10am_confirmation(_frame(rows)).filter(pl.col("signal")).height == 0


def test_lvn_emits_only_the_first_confirmation_per_session():
    rows, zones = _lvn_rows()
    rows.append(_row(datetime(2026, 1, 6, 9, 50), 107.0, 107.5, 106.0, 107.5))
    zones.append(zones[-1])
    result = detect_lvn_break_retest(_frame(rows, zones=zones))
    assert result.filter(pl.col("signal")).height == 1
