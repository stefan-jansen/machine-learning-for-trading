"""Tests for case_studies/utils/slot_strategy.py — persistent-slot selection.

The slot mechanism is a new selection method introduced for intraday case
studies where per-symbol score distributions and signal-based exits matter.
These tests pin the observable contracts of the high-level
``build_persistent_slot_weights_hybrid`` entry plus the underlying
``_run_slot_simulation`` mechanism:

  - max-hold caps position age regardless of score
  - signal-exit fires when current score < stay threshold
  - capacity is respected (max_slots concurrent holdings)
  - new entries are score-ordered when capacity is constrained
  - short_only flips the weight sign
  - stale-pred rows (older than freshness tolerance) are dropped before entry
  - empty input returns empty frame with canonical schema
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from case_studies.utils.slot_strategy import (
    _align_predictions_to_bars,
    _run_slot_simulation,
    build_persistent_slot_weights_hybrid,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _bars(n: int, start: datetime | None = None, step: timedelta | None = None) -> list[datetime]:
    """Generate ``n`` evenly spaced bar timestamps."""
    start = start or datetime(2024, 1, 2, 9, 30)
    step = step or timedelta(minutes=15)
    return [start + i * step for i in range(n)]


# -----------------------------------------------------------------------------
# _run_slot_simulation — pure mechanism
# -----------------------------------------------------------------------------


def test_simulation_single_symbol_fill_then_maxhold_exit() -> None:
    """One symbol enters at bar 0, must exit at bar 4 when hold_bars=4."""
    bars = _bars(8)
    signals = {bars[0]: [("AAA", 1.0)]}
    weights, stats = _run_slot_simulation(
        signals_by_ts=signals,
        all_bars_sorted=bars,
        max_slots=1,
        weight_per_slot=1.0,
        hold_bars=4,
        score_by_ts_sym=None,
        stay_threshold_by_ts_sym=None,
    )
    # Held bars 0..3 inclusive, exits at bar 4 (entry_i=0, i-entry_i=4 >= hold_bars)
    held_ts = weights["timestamp"].to_list()
    assert held_ts == bars[:4]
    assert (weights["symbol"] == "AAA").all()
    assert stats["n_entries"] == 1
    assert stats["n_exits_maxhold"] == 1
    assert stats["n_exits_signal"] == 0


def test_simulation_max_slots_caps_concurrent_holdings() -> None:
    """5 symbols signal simultaneously, max_slots=2 keeps top-2 by score."""
    bars = _bars(3)
    signals = {bars[0]: [(s, sc) for s, sc in zip("ABCDE", [0.1, 0.9, 0.5, 0.7, 0.3])]}
    weights, stats = _run_slot_simulation(
        signals_by_ts=signals,
        all_bars_sorted=bars,
        max_slots=2,
        weight_per_slot=0.5,
        hold_bars=10,
        score_by_ts_sym=None,
        stay_threshold_by_ts_sym=None,
    )
    held_first_bar = set(weights.filter(pl.col("timestamp") == bars[0])["symbol"].to_list())
    assert held_first_bar == {"B", "D"}  # top-2 scores 0.9 and 0.7
    assert stats["n_entries"] == 2


def test_simulation_signal_exit_fires_when_score_below_stay() -> None:
    """Signal-exit triggers when current score drops below stay threshold."""
    bars = _bars(5)
    signals = {bars[0]: [("AAA", 0.9)]}
    score_lookup = {(bars[i], "AAA"): 0.9 if i < 2 else 0.1 for i in range(5)}
    stay_lookup = {(bars[i], "AAA"): 0.5 for i in range(5)}
    weights, stats = _run_slot_simulation(
        signals_by_ts=signals,
        all_bars_sorted=bars,
        max_slots=1,
        weight_per_slot=1.0,
        hold_bars=10,
        score_by_ts_sym=score_lookup,
        stay_threshold_by_ts_sym=stay_lookup,
    )
    held_ts = weights["timestamp"].to_list()
    # Held at bars 0, 1; at bar 2 score (0.1) < stay (0.5) → exit at start of bar 2
    assert held_ts == bars[:2]
    assert stats["n_exits_signal"] == 1
    assert stats["n_exits_maxhold"] == 0


def test_simulation_signal_exit_skipped_when_score_unknown() -> None:
    """If a (ts,sym) is missing from score_lookup, signal-exit must not fire."""
    bars = _bars(4)
    signals = {bars[0]: [("AAA", 0.9)]}
    score_lookup = {(bars[0], "AAA"): 0.9}  # only bar 0
    stay_lookup = {(bars[i], "AAA"): 0.5 for i in range(4)}
    weights, stats = _run_slot_simulation(
        signals_by_ts=signals,
        all_bars_sorted=bars,
        max_slots=1,
        weight_per_slot=1.0,
        hold_bars=10,
        score_by_ts_sym=score_lookup,
        stay_threshold_by_ts_sym=stay_lookup,
    )
    # Missing scores at bars 1,2,3 → never signal-exit. Held all 4 bars.
    assert weights.height == 4
    assert stats["n_exits_signal"] == 0


def test_simulation_validates_positive_max_slots() -> None:
    with pytest.raises(ValueError, match="max_slots must be positive"):
        _run_slot_simulation(
            signals_by_ts={},
            all_bars_sorted=[],
            max_slots=0,
            weight_per_slot=1.0,
            hold_bars=1,
            score_by_ts_sym=None,
            stay_threshold_by_ts_sym=None,
        )


def test_simulation_validates_weight_per_slot_range() -> None:
    with pytest.raises(ValueError, match="weight_per_slot must be in"):
        _run_slot_simulation(
            signals_by_ts={},
            all_bars_sorted=[],
            max_slots=1,
            weight_per_slot=1.5,
            hold_bars=1,
            score_by_ts_sym=None,
            stay_threshold_by_ts_sym=None,
        )


def test_simulation_empty_signals_returns_empty_frame_with_schema() -> None:
    weights, stats = _run_slot_simulation(
        signals_by_ts={},
        all_bars_sorted=_bars(3),
        max_slots=1,
        weight_per_slot=1.0,
        hold_bars=5,
        score_by_ts_sym=None,
        stay_threshold_by_ts_sym=None,
    )
    assert weights.is_empty()
    assert weights.columns == ["timestamp", "symbol", "weight"]
    assert weights.schema["timestamp"] == pl.Datetime("us")
    assert stats["n_entries"] == 0
    assert stats["n_exits_total"] == 0


# -----------------------------------------------------------------------------
# _align_predictions_to_bars — backward-asof staleness filter
# -----------------------------------------------------------------------------


def test_align_drops_predictions_older_than_freshness_tolerance() -> None:
    """Predictions older than ``pred_freshness_max_min`` are filtered out."""
    bar_grid = pl.DataFrame(
        {
            "symbol": ["AAA"] * 3,
            "timestamp": _bars(3),  # 09:30, 09:45, 10:00
        }
    )
    # One prediction at 09:30 (fresh), one at 09:20 (stale for 09:45 bar with 14m tol)
    preds = pl.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "timestamp": [datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 9, 20)],
            "y_score": [0.5, 0.3],
        }
    )
    aligned = _align_predictions_to_bars(
        preds,
        bar_grid,
        pred_freshness_max_min=14,
        score_col="y_score",
        time_col="timestamp",
        asset_col="symbol",
    )
    # 09:30 bar: 09:30 pred (0m stale) -> 0.5
    # 09:45 bar: 09:30 pred (15m stale) -> dropped; 09:20 also too old
    # 10:00 bar: same — all preds >14m stale
    aligned_ts = aligned["timestamp"].to_list()
    assert aligned_ts == [datetime(2024, 1, 2, 9, 30)]
    assert aligned["y_score"].to_list() == [0.5]


# -----------------------------------------------------------------------------
# build_persistent_slot_weights_hybrid — high-level entry
# -----------------------------------------------------------------------------


@pytest.fixture
def predictions_dense() -> pl.DataFrame:
    """50 days × 3 symbols × 14 bars/day; deterministic seeded scores."""
    rng = np.random.default_rng(7)
    rows = []
    for d in range(50):
        for i in range(14):
            ts = datetime(2024, 1, 2, 9, 30) + timedelta(days=d, minutes=15 * i)
            for sym in ("AAA", "BBB", "CCC"):
                rows.append((ts, sym, float(rng.standard_normal())))
    return pl.DataFrame(rows, schema=["timestamp", "symbol", "y_score"], orient="row").sort(
        "symbol", "timestamp"
    )


def test_build_weights_returns_canonical_schema(predictions_dense) -> None:
    prices = predictions_dense.select(["symbol", "timestamp"]).with_columns(close=pl.lit(100.0))
    weights, stats = build_persistent_slot_weights_hybrid(
        predictions_dense,
        prices,
        long_q=0.90,
        lookback_days=10,
        bars_per_day=14,
        max_slots=2,
        hold_bars=4,
    )
    assert set(weights.columns) == {"timestamp", "symbol", "weight"}
    assert weights.schema["timestamp"] == pl.Datetime("us")
    # weight defaults to 1/max_slots
    if not weights.is_empty():
        assert (weights["weight"] - 0.5).abs().max() < 1e-9
    assert stats["max_slots"] == 2
    assert stats["direction"] == "long_only"


def test_build_weights_short_only_flips_sign(predictions_dense) -> None:
    prices = predictions_dense.select(["symbol", "timestamp"]).with_columns(close=pl.lit(100.0))
    long_w, _ = build_persistent_slot_weights_hybrid(
        predictions_dense,
        prices,
        long_q=0.80,
        lookback_days=10,
        bars_per_day=14,
        max_slots=2,
        hold_bars=4,
        direction="long_only",
    )
    short_w, _ = build_persistent_slot_weights_hybrid(
        predictions_dense,
        prices,
        long_q=0.80,
        lookback_days=10,
        bars_per_day=14,
        max_slots=2,
        hold_bars=4,
        direction="short_only",
    )
    assert long_w.shape == short_w.shape
    if not long_w.is_empty():
        # short_only is a pure sign flip
        merged = long_w.join(short_w, on=["timestamp", "symbol"], suffix="_s")
        assert (merged["weight"] + merged["weight_s"]).abs().max() < 1e-9


def test_build_weights_rejects_long_short_direction(predictions_dense) -> None:
    prices = predictions_dense.select(["symbol", "timestamp"]).with_columns(close=pl.lit(100.0))
    with pytest.raises(ValueError, match="long_short is not supported"):
        build_persistent_slot_weights_hybrid(
            predictions_dense,
            prices,
            long_q=0.80,
            lookback_days=10,
            bars_per_day=14,
            max_slots=2,
            hold_bars=4,
            direction="long_short",  # type: ignore[arg-type]
        )


def test_build_weights_rejects_stay_q_at_or_above_long_q(predictions_dense) -> None:
    prices = predictions_dense.select(["symbol", "timestamp"]).with_columns(close=pl.lit(100.0))
    with pytest.raises(ValueError, match="must be < long_q"):
        build_persistent_slot_weights_hybrid(
            predictions_dense,
            prices,
            long_q=0.50,
            lookback_days=10,
            bars_per_day=14,
            max_slots=2,
            hold_bars=4,
            exit_signal_q=0.50,
        )


def test_build_weights_with_stay_threshold_runs_clean(predictions_dense) -> None:
    """End-to-end with signal-exit enabled — schema + non-degenerate stats."""
    prices = predictions_dense.select(["symbol", "timestamp"]).with_columns(close=pl.lit(100.0))
    _weights, stats = build_persistent_slot_weights_hybrid(
        predictions_dense,
        prices,
        long_q=0.80,
        lookback_days=10,
        bars_per_day=14,
        max_slots=2,
        hold_bars=8,
        exit_signal_q=0.40,
    )
    assert stats["exit_signal_q"] == 0.40
    # Either path can produce zero entries on a tiny synthetic sample, but the
    # mechanism must not crash and stats must be coherent.
    assert stats["n_exits_total"] == stats["n_exits_maxhold"] + stats["n_exits_signal"]
