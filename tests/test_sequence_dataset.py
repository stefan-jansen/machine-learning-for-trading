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

from case_studies.utils.sequence_dataset import (
    materialize_store_metadata,
    prepare_fold_sequence_stores,
    sequence_validation_keys,
)


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


def test_sequence_store_carries_fitted_training_preprocessing():
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    df, train_mask, val_mask, val_start_ts, _ = _synthetic_fold_df()
    train_store, val_store, _ = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0", "feat1"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=20,
        val_start=val_start_ts,
    )

    assert train_store.feature_mean is not None
    assert train_store.feature_scale is not None
    np.testing.assert_array_equal(val_store.feature_mean, train_store.feature_mean)
    np.testing.assert_array_equal(val_store.feature_scale, train_store.feature_scale)


@pytest.mark.parametrize("missing_validation_row", [False, True])
def test_declared_validation_keys_equal_sequence_store(missing_validation_row):
    df, train_mask, val_mask, val_start_ts, val_end_ts = _synthetic_fold_df()
    if missing_validation_row:
        df = df.loc[
            ~((df["symbol"] == "S1") & (df["timestamp"] == pd.Timestamp("2021-02-01")))
        ].reset_index(drop=True)
        train_mask = df["timestamp"] <= pd.Timestamp("2020-12-31")
        val_mask = df["timestamp"].between(val_start_ts, val_end_ts, inclusive="both")
    df.loc[(df["symbol"] == "S2") & (df["timestamp"] == pd.Timestamp("2021-03-01")), "y"] = np.nan
    split = {
        "fold": 3,
        "train_start": pd.Timestamp("2020-01-01"),
        "train_end": pd.Timestamp("2020-12-31"),
        "val_start": val_start_ts,
        "val_end": val_end_ts,
    }

    _, val_store, _ = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0", "feat1"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=20,
        val_start=val_start_ts,
    )
    _, timestamps, symbols = materialize_store_metadata(val_store)
    actual = {
        (str(symbol), pd.Timestamp(timestamp), 3)
        for symbol, timestamp in zip(symbols, timestamps, strict=True)
    }
    declared = sequence_validation_keys(
        df,
        [split],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=20,
    )

    assert set(declared.iter_rows()) == actual


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


def test_sequence_windows_do_not_span_missing_entity_periods():
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    df, train_mask, val_mask, val_start_ts, _ = _synthetic_fold_df(
        train_end="2021-03-31",
        val_start="2021-04-01",
        val_end="2021-06-30",
    )
    missing_date = pd.Timestamp("2020-10-15")
    keep = ~((df["symbol"] == "S0") & (df["timestamp"] == missing_date))
    df = df.loc[keep].reset_index(drop=True)
    train_mask = train_mask.loc[keep].reset_index(drop=True)
    val_mask = val_mask.loc[keep].reset_index(drop=True)
    lookback = 20

    train_store, _, _ = prepare_fold_sequence_stores(
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

    calendar = pd.Index(sorted(df["timestamp"].unique()))
    for symbol_id, end_idx in zip(train_store.symbol_idx, train_store.end_idx, strict=True):
        timestamps = train_store.timestamps[int(symbol_id)]
        window = timestamps[int(end_idx) - lookback : int(end_idx) + 1]
        positions = calendar.get_indexer(window)
        assert np.all(np.diff(positions) == 1)


def test_fixed_cadence_windows_do_not_span_missing_panel_periods():
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    dates = pd.date_range("2021-01-01", periods=100, freq="8h", tz="UTC")
    missing_date = dates[40]
    rows = [
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "feat0": float(i),
            "y": float(i),
        }
        for symbol in ("S0", "S1")
        for i, timestamp in enumerate(dates)
        if timestamp != missing_date
    ]
    df = pd.DataFrame(rows)
    train_mask = df["timestamp"] < dates[75]
    val_mask = df["timestamp"] >= dates[75]
    lookback = 12

    train_store, _, _ = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=lookback,
        val_start=dates[75],
    )

    cadence = np.timedelta64(8, "h")
    for symbol_id, end_idx in zip(train_store.symbol_idx, train_store.end_idx, strict=True):
        timestamps = train_store.timestamps[int(symbol_id)]
        window = timestamps[int(end_idx) - lookback : int(end_idx) + 1]
        assert np.all(np.diff(window) == cadence)


def test_weekday_intraday_windows_reject_a_panel_wide_missing_bar():
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    session_days = pd.date_range("2021-01-04", periods=4, freq="B")
    dates = pd.DatetimeIndex(
        [
            day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * slot)
            for day in session_days
            for slot in range(26)
        ]
    )
    missing_date = session_days[1] + pd.Timedelta(hours=10)
    rows = [
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "feat0": float(i),
            "y": float(i),
        }
        for symbol in ("S0", "S1")
        for i, timestamp in enumerate(dates)
        if timestamp != missing_date
    ]
    df = pd.DataFrame(rows)
    train_mask = df["timestamp"].dt.normalize() < session_days[3]
    val_mask = df["timestamp"].dt.normalize() == session_days[3]
    lookback = 2

    train_store, _, _ = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=lookback,
        val_start=session_days[3] + pd.Timedelta(hours=9, minutes=30),
        calendar_id="NYSE",
    )

    complete_positions = dates.get_indexer
    crossed_session = False
    for symbol_id, end_idx in zip(train_store.symbol_idx, train_store.end_idx, strict=True):
        timestamps = train_store.timestamps[int(symbol_id)]
        window = timestamps[int(end_idx) - lookback : int(end_idx) + 1]
        assert np.all(np.diff(complete_positions(window)) == 1)
        crossed_session |= len(pd.DatetimeIndex(window).normalize().unique()) > 1
    assert crossed_session


def test_monthly_period_numbers_preserve_gaps_at_millisecond_resolution():
    from case_studies.utils.sequence_dataset import _sequence_period_numbers

    timestamps = pd.Series(
        np.asarray(["2021-01-31", "2021-02-28", "2021-04-30"], dtype="datetime64[ms]")
    )

    periods = _sequence_period_numbers(timestamps)

    assert np.diff(periods).tolist() == [1, 2]


def test_daily_period_numbers_use_the_declared_market_calendar():
    from case_studies.utils.sequence_dataset import _sequence_period_numbers

    around_holiday = pd.Series(
        pd.to_datetime(["2022-06-29", "2022-06-30", "2022-07-01", "2022-07-05", "2022-07-06"])
    )
    missing_session = pd.Series(
        pd.to_datetime(["2022-06-29", "2022-06-30", "2022-07-01", "2022-07-06", "2022-07-07"])
    )

    observed = _sequence_period_numbers(around_holiday, calendar_id="NYSE")
    missing = _sequence_period_numbers(missing_session, calendar_id="NYSE")

    assert np.diff(observed).tolist() == [1, 1, 1, 1]
    assert np.diff(missing).tolist() == [1, 1, 2, 1]


def test_sequence_period_cache_is_recomputed_for_a_declared_calendar():
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    dates = pd.to_datetime(["2022-06-30", "2022-07-01", "2022-07-06", "2022-07-07"])
    df = pd.DataFrame(
        [
            {"symbol": symbol, "timestamp": timestamp, "feat0": float(i), "y": float(i)}
            for symbol in ("S0", "S1")
            for i, timestamp in enumerate(dates)
        ]
    )
    train_mask = df["timestamp"] <= dates[2]
    val_mask = df["timestamp"] == dates[3]
    kwargs = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "feature_names": ["feat0"],
        "label_col": "y",
        "date_col": "timestamp",
        "entity_col": "symbol",
        "lookback": 1,
        "val_start": dates[3],
    }

    fallback_train, _, _ = prepare_fold_sequence_stores(df, **kwargs)
    calendar_train, _, _ = prepare_fold_sequence_stores(df, calendar_id="NYSE", **kwargs)

    assert fallback_train.n_sequences == 4
    assert calendar_train.n_sequences == 2


def test_priming_includes_label_buffer_gap_rows():
    from case_studies.utils.sequence_dataset import prepare_fold_sequence_stores

    train_end = pd.Timestamp("2020-12-30")
    df, train_mask, val_mask, val_start_ts, _ = _synthetic_fold_df(
        train_end=str(train_end.date()),
        val_start="2021-01-04",
    )
    gap_mask = (df["timestamp"] > train_end) & (df["timestamp"] < val_start_ts) & ~train_mask
    assert gap_mask.any()
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
        entity = val_store.entities[symbol_id]
        end_positions = val_store.end_idx[val_store.symbol_idx == symbol_id]
        first_end = int(end_positions.min())
        last_context = pd.Timestamp(val_store.timestamps[symbol_id][first_end - 1])
        expected_context = pd.Timestamp(
            df.loc[(df["symbol"] == entity) & (df["timestamp"] < val_start_ts), "timestamp"].max()
        )
        assert last_context == expected_context
        assert last_context > train_end
