"""Correctness tests for case_studies/utils/sequence_dataset.py.

These tests encode the methodology property that every DL case study
depends on: the first validation sequence must predict the target at
val_start, using an input window that may extend back into train (this
is legal because features at times ≤ val_start are already known at
val_start; only labels after val_start are held out).

A test failure here means validation sequences have a warmup-drop bug
where the first `lookback` trading days of each val fold are silently
discarded — this inflates DL Sharpe on adversarial sample-period
exclusions and diverges from how the model would be deployed in
production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_fold_df(
    *,
    n_symbols: int = 3,
    train_start: str = "2020-01-01",
    train_end: str = "2020-12-31",
    val_start: str = "2021-01-01",
    val_end: str = "2021-06-30",
    freq: str = "B",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    """Build a synthetic panel: N symbols × business days train+val.

    Returns (df, train_mask, val_mask, val_start_ts, val_end_ts).
    """
    all_dates = pd.date_range(train_start, val_end, freq=freq)
    rows = []
    for i, sym in enumerate([f"S{j}" for j in range(n_symbols)]):
        for dt in all_dates:
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": dt,
                    "feat0": float(i) + dt.toordinal() / 1e6,
                    "feat1": float(i) * 2 + dt.toordinal() / 1e6,
                    "y": float(i) + np.sin(dt.toordinal() / 10.0),
                }
            )
    df = pd.DataFrame(rows)

    ts_start = pd.Timestamp(train_start)
    ts_train_end = pd.Timestamp(train_end)
    ts_val_start = pd.Timestamp(val_start)
    ts_val_end = pd.Timestamp(val_end)

    train_mask = (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_train_end)
    val_mask = (df["timestamp"] >= ts_val_start) & (df["timestamp"] <= ts_val_end)
    return df, train_mask, val_mask, ts_val_start, ts_val_end


def test_val_sequence_starts_at_val_start():
    """Every symbol's first val sequence should have target == val_start.

    This is the core correctness property: in production, on val_start
    we have all pre-val features available and must emit a prediction
    for val_start. The prior (buggy) implementation discards the first
    `lookback` rows of each val fold.
    """
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    df, train_mask, val_mask, val_start_ts, _ = _synthetic_fold_df()
    lookback = 20

    _, val_store, fold_info = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0", "feat1"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=lookback,
        val_start=val_start_ts,
    )

    assert fold_info["val_sequences"] > 0, "No val sequences generated"

    # For each symbol, find the first sequence's target timestamp
    for symbol_id in range(val_store.n_symbols):
        end_positions = val_store.end_idx[val_store.symbol_idx == symbol_id]
        if len(end_positions) == 0:
            continue
        first_end = end_positions.min()
        first_target_ts = val_store.timestamps[symbol_id][first_end]
        assert pd.Timestamp(first_target_ts) == val_start_ts, (
            f"Symbol {val_store.entities[symbol_id]!r}: first val sequence "
            f"predicts {first_target_ts}, expected {val_start_ts}. "
            f"This indicates the warmup-drop bug — the first {lookback} "
            f"trading days of val are being silently skipped."
        )


def test_val_sequence_count_matches_val_calendar_days():
    """Number of val sequences per symbol == number of val-period rows."""
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    df, train_mask, val_mask, val_start_ts, val_end_ts = _synthetic_fold_df()
    lookback = 20

    _, val_store, fold_info = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0", "feat1"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=lookback,
        val_start=val_start_ts,
    )

    expected_per_symbol = int(
        df[(df["timestamp"] >= val_start_ts) & (df["timestamp"] <= val_end_ts)]
        .groupby("symbol")
        .size()
        .iloc[0]
    )
    actual_per_symbol = fold_info["val_sequences"] // val_store.n_symbols
    assert actual_per_symbol == expected_per_symbol, (
        f"Each symbol should have {expected_per_symbol} val sequences "
        f"(one per val trading day); got {actual_per_symbol}. "
        f"Shortfall indicates warmup drop."
    )


def test_val_sequence_targets_never_include_train_period():
    """No val sequence should have a target timestamp < val_start.

    Train-tail rows are used for priming input features only; their
    labels must not appear as val targets (that would be leakage).
    """
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    df, train_mask, val_mask, val_start_ts, _ = _synthetic_fold_df()
    lookback = 20

    _, val_store, _ = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0", "feat1"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=lookback,
        val_start=val_start_ts,
    )

    for symbol_id in range(val_store.n_symbols):
        end_positions = val_store.end_idx[val_store.symbol_idx == symbol_id]
        for pos in end_positions:
            target_ts = val_store.timestamps[symbol_id][pos]
            assert pd.Timestamp(target_ts) >= val_start_ts, (
                f"Val sequence target {target_ts} predates val_start "
                f"{val_start_ts} — train-tail priming is leaking into "
                f"predictions."
            )


def test_backwards_compatible_without_val_start():
    """Omitting val_start should preserve the legacy behavior exactly.

    This ensures existing callers that don't pass val_start get the
    same (buggy, but known) output — the fix is opt-in via val_start.
    The legacy path may be removed in a later commit.
    """
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    df, train_mask, val_mask, _, _ = _synthetic_fold_df()
    lookback = 20

    _, val_store, fold_info = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0", "feat1"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=lookback,
        # val_start intentionally omitted — legacy behavior
    )

    # In legacy mode, first val sequence should be at position `lookback`
    # within the val slice (the bug we're documenting).
    for symbol_id in range(val_store.n_symbols):
        end_positions = val_store.end_idx[val_store.symbol_idx == symbol_id]
        if len(end_positions) == 0:
            continue
        assert int(end_positions.min()) == lookback, (
            "Legacy path should start sequences at position=lookback"
        )
