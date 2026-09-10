"""Shared lazy sequence datasets for PyTorch deep-learning case studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset

from utils.modeling import RANDOM_SEED

_SEQUENCE_PERIOD_COL = "__sequence_period__"
_SEQUENCE_PERIOD_CACHE_ATTR = "ml4t_sequence_period_cache"

# Sequence windows are built on the panel's expected observation calendar, not
# on the symbol's own row order. A symbol that is not quoted on a session keeps
# its cell in the window; the cell carries no feature values and is marked
# unobserved. Dropping the cell instead would present a window of `lookback`
# observations spanning more calendar periods than `lookback` as if it were
# contiguous, which silently rescales every horizon the model learns - a
# five-day-ahead target predicted from what the model reads as five days but
# which actually covers eight.
OBSERVED_FEATURE = "__observed__"
STALENESS_FEATURE = "__periods_since_observation__"
GAP_MASK_FEATURES: tuple[str, str] = (OBSERVED_FEATURE, STALENESS_FEATURE)

# Eligibility bounds, measured rather than assumed. Across the nine panels the
# observed-cell share of a 60-period window sits at 100% for the median symbol
# in eight of them, with a thin left tail; 0.90 is the tenth percentile of the
# four densest panels and keeps 93-100% of their windows. The consecutive bound
# separates six scattered absences from one six-period outage, which the
# fraction alone cannot: at a lookback of 60 a 0.90 floor already caps a run at
# six, so 5 removes only the single-outage case. Distributions and the script
# that produced them: work/2026-09-08-sequence-gap-policy/ in the agents repo.
DEFAULT_MIN_OBSERVED_FRACTION = 0.90
DEFAULT_MAX_CONSECUTIVE_GAP = 5
GAP_POLICY_ID = "calendar_grid_observation_mask/min_observed=0.90,max_gap=5/v1"


@dataclass(slots=True)
class SequenceStore:
    """Per-fold sequence store backed by normalized per-symbol arrays."""

    features: list[np.ndarray]
    targets: list[np.ndarray]
    timestamps: list[np.ndarray]
    entities: list[str]
    symbol_idx: np.ndarray
    end_idx: np.ndarray
    lookback: int
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    @property
    def n_sequences(self) -> int:
        return int(len(self.symbol_idx))

    @property
    def n_symbols(self) -> int:
        return int(len(self.entities))


class FoldSequenceDataset(Dataset):
    """Lazy map-style dataset yielding lookback windows on demand."""

    def __init__(self, store: SequenceStore, *, include_metadata: bool = False) -> None:
        self.store = store
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return self.store.n_sequences

    def __getitem__(self, idx: int):
        symbol_id = int(self.store.symbol_idx[idx])
        end_idx = int(self.store.end_idx[idx])
        features = self.store.features[symbol_id]
        window = torch.from_numpy(features[end_idx - self.store.lookback : end_idx])
        target = torch.tensor(self.store.targets[symbol_id][end_idx], dtype=torch.float32)
        if not self.include_metadata:
            return window, target
        timestamp = self.store.timestamps[symbol_id][end_idx]
        entity = self.store.entities[symbol_id]
        return window, target, timestamp, entity


def collate_with_metadata(batch):
    """Collate evaluation batches while preserving timestamps/entities."""

    X = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    timestamps = np.asarray([item[2] for item in batch])
    entities = np.asarray([item[3] for item in batch], dtype="U64")
    return X, y, timestamps, entities


def materialize_store_metadata(store: SequenceStore) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return targets, timestamps, and entities in dataset order."""

    y_parts: list[np.ndarray] = []
    ts_parts: list[np.ndarray] = []
    entity_parts: list[np.ndarray] = []
    for symbol_id, end_idx in zip(store.symbol_idx, store.end_idx, strict=True):
        y_parts.append(np.asarray([store.targets[int(symbol_id)][int(end_idx)]], dtype=np.float32))
        ts_parts.append(np.asarray([store.timestamps[int(symbol_id)][int(end_idx)]]))
        entity_parts.append(np.asarray([store.entities[int(symbol_id)]], dtype="U64"))

    if not y_parts:
        empty_f = np.array([], dtype=np.float32)
        empty_u = np.array([], dtype="U64")
        empty_t = np.array([], dtype="datetime64[ns]")
        return empty_f, empty_t, empty_u

    return np.concatenate(y_parts), np.concatenate(ts_parts), np.concatenate(entity_parts)


def materialize_sequences(
    store: SequenceStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize a SequenceStore into contiguous numpy arrays.

    Returns (X, y, timestamps, entities) where:
    - X: shape (n_sequences, lookback, n_features), float32
    - y: shape (n_sequences,), float32
    - timestamps: shape (n_sequences,)
    - entities: shape (n_sequences,), dtype U64
    """
    n_seq = store.n_sequences
    if n_seq == 0:
        n_feat = store.features[0].shape[1] if store.features else 0
        return (
            np.empty((0, store.lookback, n_feat), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            np.empty(0, dtype="U64"),
        )

    n_features = store.features[0].shape[1]
    X = np.empty((n_seq, store.lookback, n_features), dtype=np.float32)
    y = np.empty(n_seq, dtype=np.float32)
    ts_dtype = store.timestamps[0].dtype if store.timestamps else "datetime64[ns]"
    timestamps = np.empty(n_seq, dtype=ts_dtype)
    entities = np.empty(n_seq, dtype="U64")

    for i, (sid, eidx) in enumerate(zip(store.symbol_idx, store.end_idx, strict=True)):
        sid, eidx = int(sid), int(eidx)
        X[i] = store.features[sid][eidx - store.lookback : eidx]
        y[i] = store.targets[sid][eidx]
        timestamps[i] = store.timestamps[sid][eidx]
        entities[i] = store.entities[sid]

    return X, y, timestamps, entities


def _sample_sequence_positions(
    counts: np.ndarray,
    max_sequences: int,
    stride: int = 0,
) -> list[np.ndarray | None]:
    """Sample sequence endpoints while preserving full symbol coverage.

    ``stride`` spaces the endpoints and lets the count follow, which is what
    ``modeling.dl.train_sequence_stride_horizons`` declares. It is applied per symbol and
    counts valid endpoints of that symbol, not panel rows: a symbol that is missing bars has
    no window ending in the gap, so spacing its endpoints is the closest thing to spacing them
    in time that a per-symbol index supports. Every symbol with any valid endpoint keeps at
    least its first, so striding never drops a symbol from the universe.

    ``max_sequences`` is the other form. It fixes the total and derives the spacing from it, so
    two folds of different length get different spacing; that is the price of a fixed budget.
    """

    sampled_positions: list[np.ndarray | None] = [None] * len(counts)
    if stride > 1:
        for idx, n_seq in enumerate(counts):
            if n_seq > stride:
                sampled_positions[idx] = np.arange(0, int(n_seq), stride, dtype=np.int64)
            elif n_seq > 1:
                sampled_positions[idx] = np.zeros(1, dtype=np.int64)
        return sampled_positions
    if max_sequences <= 0 or int(counts.sum()) <= max_sequences:
        return sampled_positions

    active_symbols = np.flatnonzero(counts > 0)
    n_active_symbols = len(active_symbols)
    if max_sequences < n_active_symbols:
        raise ValueError(
            f"max_sequences={max_sequences:,} is smaller than the symbol count "
            f"({n_active_symbols:,}); cannot preserve full universe coverage"
        )

    alloc = np.zeros(len(counts), dtype=np.int64)
    alloc[active_symbols] = 1
    remaining = max_sequences - n_active_symbols
    extra_capacity = np.maximum(counts - alloc, 0)

    if remaining > 0 and extra_capacity.sum() > 0:
        weighted_extra = np.floor(extra_capacity / extra_capacity.sum() * remaining).astype(
            np.int64
        )
        weighted_extra = np.minimum(weighted_extra, extra_capacity)
        alloc += weighted_extra
        remaining -= int(weighted_extra.sum())

    while remaining > 0:
        spare = counts - alloc
        available = np.flatnonzero(spare > 0)
        if len(available) == 0:
            break
        step = min(remaining, len(available))
        alloc[available[:step]] += 1
        remaining -= step

    for idx, n_seq in enumerate(counts):
        take = int(min(alloc[idx], n_seq))
        if take >= n_seq:
            continue
        offsets = (np.arange(take, dtype=np.int64) * n_seq) // take
        sampled_positions[idx] = offsets

    return sampled_positions


def _period_timestamp_grid(sorted_df: pd.DataFrame, *, date_col: str) -> tuple[int, np.ndarray]:
    """Return ``(first_period, timestamps)`` covering every period in the panel.

    Indexing is ``timestamps[period - first_period]``. Periods no symbol has a
    row for still get a cell, because for a fixed-cadence panel the period
    numbers are generated from the cadence rather than from the observed rows,
    so a bar the whole panel is missing leaves a hole in the numbering that a
    per-symbol window can span. Their timestamp is reconstructed from the
    panel's own period spacing; a lookup that fell through to the next present
    period instead would put the same timestamp on two adjacent cells and make
    a window's own time axis non-monotonic.
    """

    pairs = sorted_df[[_SEQUENCE_PERIOD_COL, date_col]].drop_duplicates(
        subset=[_SEQUENCE_PERIOD_COL]
    )
    pairs = pairs.sort_values(_SEQUENCE_PERIOD_COL, kind="stable")
    periods = pairs[_SEQUENCE_PERIOD_COL].to_numpy(dtype=np.int64)
    stamps = pairs[date_col].to_numpy(dtype="datetime64[ns]").astype("int64")
    first = int(periods[0])
    span = int(periods[-1]) - first + 1
    grid = np.full(span, np.iinfo(np.int64).min, dtype=np.int64)
    grid[periods - first] = stamps
    missing = grid == np.iinfo(np.int64).min
    if missing.any():
        step_periods = np.diff(periods)
        step_time = np.diff(stamps)
        cadence = int(np.median(step_time[step_periods == 1])) if (step_periods == 1).any() else 0
        if cadence <= 0:
            cadence = int(np.median(step_time // np.maximum(step_periods, 1)))
        positions = np.arange(span, dtype=np.int64)
        anchor = np.maximum.accumulate(np.where(~missing, positions, -1))
        grid[missing] = grid[anchor[missing]] + (positions[missing] - anchor[missing]) * cadence
    return first, grid.astype("datetime64[ns]")


def _build_symbol_arrays(
    fold_df: pd.DataFrame,
    *,
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    lookback: int,
    min_observed_fraction: float = DEFAULT_MIN_OBSERVED_FRACTION,
    max_consecutive_gap: int = DEFAULT_MAX_CONSECUTIVE_GAP,
    emit_gap_mask: bool = True,
    min_end_timestamp: pd.Timestamp | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str], list[np.ndarray]]:
    """Convert a fold dataframe into per-symbol calendar-grid arrays and endpoints.

    Each symbol is laid out on the panel's expected observation periods between
    its first and last row, so index arithmetic on the arrays is arithmetic in
    calendar periods. Periods the symbol has no row for are present as cells
    with missing features, an ``observed`` flag of zero, and the number of
    periods since its last real observation. A window is eligible when it ends
    on a real observation with a finite target, at least
    ``min_observed_fraction`` of its cells are real, and no run of consecutive
    missing cells exceeds ``max_consecutive_gap``.

    ``min_end_timestamp`` bounds which targets a window may predict. The
    validation frame carries priming rows from before the fold boundary so the
    first validation window has a full input history; without this bound those
    priming rows would themselves be predicted, emitting validation predictions
    for training-period targets. The boundary is enforced here rather than left
    to the size of the priming tail, which is sized for calendar coverage and
    is not a statement about where predictions begin.
    """

    if fold_df.empty:
        return [], [], [], [], []

    features_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    timestamps_list: list[np.ndarray] = []
    entities: list[str] = []
    valid_positions_list: list[np.ndarray] = []

    sorted_df = fold_df.sort_values([entity_col, date_col], kind="stable")
    # Coerce the date column to tz-naive datetime64[ns] once. tz-aware pandas
    # datetimes (e.g., crypto's Datetime[ms, UTC]) survive `.to_numpy()` as an
    # object array of pd.Timestamp; that lands in polars as Object dtype,
    # which the row-encoding path (group_by/sort/join keys) cannot handle and
    # panics with "Unsupported in row encoding". The IC and downstream
    # ranking ops only need unique date keys; tz is informational.
    date_col_dtype = sorted_df[date_col].dtype
    if hasattr(date_col_dtype, "tz") and date_col_dtype.tz is not None:
        sorted_df = sorted_df.assign(**{date_col: sorted_df[date_col].dt.tz_convert(None)})

    # Inserted cells still need a timestamp, because a store's timestamp array
    # is indexed by grid position. Take it from the panel's own period/timestamp
    # pairing rather than interpolating, so an inserted cell carries the session
    # it stands for and never a date the panel does not trade on.
    panel_first_period, panel_timestamps = _period_timestamp_grid(sorted_df, date_col=date_col)

    n_features = len(feature_names)
    n_mask = len(GAP_MASK_FEATURES) if emit_gap_mask else 0

    for symbol, sym_df in sorted_df.groupby(entity_col, sort=False):
        periods = sym_df[_SEQUENCE_PERIOD_COL].to_numpy(dtype=np.int64)
        if len(periods) == 0:
            continue
        first_period = int(periods[0])
        span = int(periods[-1]) - first_period + 1
        if span < lookback + 1:
            continue
        grid_index = periods - first_period

        observed = np.zeros(span, dtype=bool)
        observed[grid_index] = True

        feats = np.full((span, n_features + n_mask), np.nan, dtype=np.float32)
        if n_features:
            raw = sym_df[feature_names].to_numpy(dtype=np.float32, copy=True)
            # Keep missing values missing until after normalization. Filling here
            # would put a raw 0.0 into _compute_feature_stats, which is not the
            # feature's mean on its own scale - for a strictly positive feature
            # like a conditional volatility it sits below the observed minimum, so
            # a symbol with no model-based estimate is presented to the model as
            # the calmest name in the panel rather than a neutral one. Infinities
            # are treated as missing for the same reason: np.nan_to_num leaves
            # posinf on its default, the float32 maximum, which would destroy the
            # feature's mean and standard deviation for every other symbol.
            raw[~np.isfinite(raw)] = np.nan
            feats[grid_index, :n_features] = raw

        # Periods since the last real observation. The grid starts on the
        # symbol's first observation, so this is defined at every cell.
        positions = np.arange(span, dtype=np.int64)
        last_observed = np.maximum.accumulate(np.where(observed, positions, -1))
        staleness = positions - last_observed
        if n_mask:
            # Clipped at the eligibility bound: a longer run never appears inside
            # an accepted window, and leaving one in would let a symbol that
            # stopped quoting for five thousand sessions set the scale of this
            # channel for every other symbol in _compute_feature_stats.
            feats[:, n_features] = observed.astype(np.float32)
            feats[:, n_features + 1] = np.minimum(staleness, max_consecutive_gap).astype(np.float32)

        targets = np.full(span, np.nan, dtype=np.float32)
        targets[grid_index] = sym_df[label_col].to_numpy(dtype=np.float32)

        # Cast to datetime64[ns] explicitly so concat/np.asarray downstream
        # never falls back to object dtype.
        timestamps = panel_timestamps[first_period - panel_first_period + positions]

        gap_bound = min(max_consecutive_gap, lookback)
        candidate_positions = np.arange(lookback, span, dtype=np.int32)
        # The window a model reads is [end - lookback, end); the target is at end.
        observed_prefix = np.concatenate(([0], np.cumsum(observed, dtype=np.int64)))
        observed_in_window = (
            observed_prefix[candidate_positions] - observed_prefix[candidate_positions - lookback]
        )
        enough_observed = observed_in_window >= np.ceil(min_observed_fraction * lookback)

        # A run longer than the bound exists in [s, e) exactly when some cell in
        # [s + max_consecutive_gap, e) closes a run of at least
        # max_consecutive_gap + 1 missing cells.
        run_too_long = staleness >= gap_bound + 1
        run_prefix = np.concatenate(([0], np.cumsum(run_too_long, dtype=np.int64)))
        gap_within_bound = (
            run_prefix[candidate_positions] - run_prefix[candidate_positions - lookback + gap_bound]
        ) == 0

        ends_on_observation = observed[candidate_positions]
        target_is_finite = np.isfinite(targets[candidate_positions])
        eligible = ends_on_observation & target_is_finite & enough_observed & gap_within_bound
        if min_end_timestamp is not None:
            boundary = pd.Timestamp(min_end_timestamp)
            if boundary.tz is not None:
                boundary = boundary.tz_localize(None)
            eligible &= timestamps[candidate_positions] >= boundary.to_datetime64()
        valid_positions = candidate_positions[eligible]

        features_list.append(feats)
        targets_list.append(targets)
        timestamps_list.append(timestamps)
        entities.append(str(symbol))
        valid_positions_list.append(valid_positions)

    return (
        features_list,
        targets_list,
        timestamps_list,
        entities,
        valid_positions_list,
    )


def _compute_feature_stats(
    features_list: list[np.ndarray], *, n_passthrough: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std across raw training rows without concatenating arrays.

    The last ``n_passthrough`` channels are left on their own scale, reported as
    a mean of zero and a scale of one. They are the observation mask and the
    staleness count, which already occupy fixed ranges. Standardizing them would
    tie the value that means "this cell was never observed" to how much of the
    fold happened to be missing, so the same absence would reach the model as a
    different number in a dense fold than in a sparse one.
    """

    if not features_list:
        raise ValueError("No feature arrays available to compute scaling statistics")

    n_features = features_list[0].shape[1]
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)
    n_rows = np.zeros(n_features, dtype=np.int64)

    for feats in features_list:
        observed = np.isfinite(feats)
        values = np.where(observed, feats, 0.0).astype(np.float64)
        sum_x += values.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(values).sum(axis=0, dtype=np.float64)
        n_rows += observed.sum(axis=0, dtype=np.int64)

    denominator = np.maximum(n_rows, 1)
    means = sum_x / denominator
    variances = np.maximum(sum_x2 / denominator - np.square(means), 0.0)
    stds = np.sqrt(variances)
    stds[stds == 0] = 1.0
    # A feature observed nowhere in training normalizes to zero everywhere,
    # which is what the post-normalization fill would give it anyway.
    means[n_rows == 0] = 0.0
    if n_passthrough:
        means[-n_passthrough:] = 0.0
        stds[-n_passthrough:] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def _normalize_feature_arrays(
    features_list: list[np.ndarray],
    means: np.ndarray,
    stds: np.ndarray,
) -> None:
    """Normalize feature arrays in place."""

    for feats in features_list:
        feats -= means
        feats /= stds
        # Now that the arrays are on the training scale, 0.0 is the training
        # mean of the observed rows, so a row with no observation reads as
        # average rather than as an extreme. The linear and tabular families
        # reach the same place by a different route - SimpleImputer fills the
        # training median and StandardScaler is then fitted on the filled
        # data - so the normalized values differ; what is shared is that the
        # fill is a central value of the feature rather than a raw zero.
        np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _build_sequence_index(
    valid_positions_list: list[np.ndarray],
    entities: list[str],
    max_sequences: int,
    stride: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat symbol/end-position indices for a sequence store."""

    counts = np.asarray([len(positions) for positions in valid_positions_list], dtype=np.int64)
    sampled_offsets = _sample_sequence_positions(counts, max_sequences, stride)
    symbol_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []

    for symbol_id, (entity, valid_positions, offsets) in enumerate(
        zip(entities, valid_positions_list, sampled_offsets, strict=True)
    ):
        del entity  # entity order is already captured by the symbol_id list
        positions = valid_positions
        if offsets is not None:
            positions = positions[offsets]
        positions = positions.astype(np.int32, copy=False)
        symbol_parts.append(np.full(len(positions), symbol_id, dtype=np.int32))
        end_parts.append(positions)

    symbol_idx = np.concatenate(symbol_parts) if symbol_parts else np.array([], dtype=np.int32)
    end_idx = np.concatenate(end_parts) if end_parts else np.array([], dtype=np.int32)
    return symbol_idx, end_idx


def _sequence_period_numbers(
    timestamps: pd.Series,
    *,
    calendar_id: str | None = None,
) -> np.ndarray:
    """Map timestamps to consecutive expected observation periods."""

    values = pd.DatetimeIndex(pd.to_datetime(timestamps)).as_unit("ns")
    unique = values.dropna().unique().sort_values()
    if len(unique) < 2:
        return np.zeros(len(values), dtype=np.int64)

    diffs = np.diff(unique.asi8)
    median_days = float(np.median(diffs)) / float(pd.Timedelta(days=1).value)
    intraday = unique.normalize().nunique() < len(unique)
    trades_on_weekends = bool(np.any(unique.dayofweek >= 5))

    if intraday:
        positive_diffs = diffs[diffs > 0]
        cadence_ns = int(pd.Series(positive_diffs).mode().iloc[0])
        if trades_on_weekends:
            return ((values.asi8 - unique[0].value) // cadence_ns).astype(np.int64)

        if calendar_id:
            import pandas_market_calendars as mcal

            calendar = mcal.get_calendar(calendar_id)
            schedule = calendar.schedule(
                start_date=(unique.min() - pd.Timedelta(days=7)).date(),
                end_date=(unique.max() + pd.Timedelta(days=7)).date(),
            )
            cadence = pd.Timedelta(cadence_ns, unit="ns")
            if values.tz is None:
                instant_candidates = (
                    values.tz_localize(calendar.tz).tz_convert("UTC"),
                    values.tz_localize("UTC"),
                )
            else:
                instant_candidates = (values.tz_convert("UTC"),)

            best_positions = np.full(len(values), -1, dtype=np.int64)
            for instants in instant_candidates:
                for closed, force_close in (("left", False), ("right", True)):
                    expected = mcal.date_range(
                        schedule,
                        frequency=cadence,
                        closed=closed,
                        force_close=force_close,
                    )
                    positions = expected.get_indexer(instants)
                    if np.count_nonzero(positions >= 0) > np.count_nonzero(best_positions >= 0):
                        best_positions = positions
            if np.all(best_positions >= 0):
                return best_positions.astype(np.int64)

        normalized = values.normalize()
        slot = (values.asi8 - normalized.asi8) // cadence_ns
        first_slot = int(slot.min())
        slots_per_session = int(slot.max() - first_slot + 1)
        session_number = np.busday_count(
            np.datetime64("1970-01-01", "D"),
            normalized.to_numpy(dtype="datetime64[D]"),
        )
        return (session_number * slots_per_session + slot - first_slot).astype(np.int64)

    if median_days >= 20:
        return (values.year * 12 + values.month).astype(np.int64)

    if median_days >= 4:
        return (values.to_period("W").asi8 - values.to_period("W").asi8.min()).astype(np.int64)

    if calendar_id:
        import pandas_market_calendars as mcal

        calendar = mcal.get_calendar(calendar_id)
        sessions = calendar.valid_days(
            start_date=unique.min().normalize(),
            end_date=unique.max().normalize(),
        ).tz_localize(None)
        positions = sessions.get_indexer(values.normalize().tz_localize(None))
        if np.all(positions >= 0):
            return positions.astype(np.int64)

    return unique.get_indexer(values).astype(np.int64)


def _build_val_df_with_priming(
    full_val_source: pd.DataFrame,
    *,
    entity_col: str,
    date_col: str,
    val_start: pd.Timestamp,
    lookback: int,
    min_observed_fraction: float = DEFAULT_MIN_OBSERVED_FRACTION,
) -> pd.DataFrame:
    """Per-symbol, keep the pre-validation priming rows plus all validation rows.

    Pre-validation rows provide the input window for the first validation
    target prediction; their labels are not emitted as val targets because
    sequence positions start at index `lookback` within each symbol's grid,
    and the priming tail is sized so the first target falls at val_start.

    The tail is counted in observations but the window is counted in calendar
    periods, and a sparsely quoted symbol needs more of the former to fill the
    latter. Taking `lookback` observations always spans at least `lookback`
    periods, so the first window exists either way - but it can be mostly
    inserted cells and fail the observed-fraction test that the same symbol
    passes in production, where more history is available. Sizing the tail by
    the fraction removes that difference between the first validation window
    and every later one.
    """
    priming_rows = int(np.ceil(lookback / max(min_observed_fraction, 1e-9)))
    pieces: list[pd.DataFrame] = []
    for _, sym_df in full_val_source.groupby(entity_col, sort=False):
        sym_df = sym_df.sort_values(date_col, kind="stable")
        is_val = sym_df[date_col] >= val_start
        context_tail = sym_df.loc[~is_val].tail(priming_rows)
        val_part = sym_df.loc[is_val]
        if context_tail.empty and val_part.empty:
            continue
        pieces.append(pd.concat([context_tail, val_part], ignore_index=True))
    if not pieces:
        return full_val_source.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _ensure_sequence_periods(
    dataset_pd: pd.DataFrame,
    *,
    date_col: str,
    calendar_id: str | None,
) -> None:
    cache_key = (
        date_col,
        calendar_id,
        len(dataset_pd),
        dataset_pd[date_col].iloc[0] if len(dataset_pd) else None,
        dataset_pd[date_col].iloc[-1] if len(dataset_pd) else None,
    )
    if (
        _SEQUENCE_PERIOD_COL not in dataset_pd.columns
        or dataset_pd.attrs.get(_SEQUENCE_PERIOD_CACHE_ATTR) != cache_key
    ):
        dataset_pd[_SEQUENCE_PERIOD_COL] = _sequence_period_numbers(
            dataset_pd[date_col],
            calendar_id=calendar_id,
        )
        dataset_pd.attrs[_SEQUENCE_PERIOD_CACHE_ATTR] = cache_key


def _coerce_boundary(value: pd.Timestamp | str, series: pd.Series) -> pd.Timestamp:
    boundary = pd.Timestamp(value)
    col_tz = getattr(series.dtype, "tz", None)
    if col_tz is not None and boundary.tz is None:
        return boundary.tz_localize(col_tz)
    if col_tz is None and boundary.tz is not None:
        return boundary.tz_localize(None)
    return boundary


def sequence_validation_keys(
    dataset_pd: pd.DataFrame,
    splits: list[dict],
    *,
    label_col: str,
    date_col: str,
    entity_col: str,
    lookback: int,
    calendar_id: str | None = None,
    max_predict_sequences: int = 0,
    min_observed_fraction: float = DEFAULT_MIN_OBSERVED_FRACTION,
    max_consecutive_gap: int = DEFAULT_MAX_CONSECUTIVE_GAP,
) -> pl.DataFrame:
    """Return the exact validation keys a sequence model may predict.

    ``max_predict_sequences`` caps the windows drawn per fold, using the same
    even-spacing rule and full-symbol-coverage guarantee that
    ``max_train_sequences`` applies to training. It must match what
    :func:`prepare_fold_sequence_stores` is given for the same fold, because these
    keys are the contract the published predictions are checked against.
    """
    _ensure_sequence_periods(dataset_pd, date_col=date_col, calendar_id=calendar_id)
    use_cols = [date_col, entity_col, label_col, _SEQUENCE_PERIOD_COL]
    frames: list[pl.DataFrame] = []
    for split in splits:
        val_start = _coerce_boundary(split["val_start"], dataset_pd[date_col])
        val_end = _coerce_boundary(split["val_end"], dataset_pd[date_col])
        val_mask = dataset_pd[date_col].between(val_start, val_end, inclusive="both")
        source = dataset_pd.loc[(dataset_pd[date_col] < val_start) | val_mask, use_cols]
        val_df = _build_val_df_with_priming(
            source,
            entity_col=entity_col,
            date_col=date_col,
            val_start=val_start,
            lookback=lookback,
            min_observed_fraction=min_observed_fraction,
        )
        _, _, timestamps, entities, valid_positions = _build_symbol_arrays(
            val_df,
            feature_names=[],
            label_col=label_col,
            date_col=date_col,
            entity_col=entity_col,
            lookback=lookback,
            min_observed_fraction=min_observed_fraction,
            max_consecutive_gap=max_consecutive_gap,
            emit_gap_mask=False,
            min_end_timestamp=val_start,
        )
        sampled = _sample_sequence_positions(
            np.asarray([len(positions) for positions in valid_positions], dtype=np.int64),
            max_predict_sequences,
        )
        rows = [
            {
                "symbol": entities[symbol_id],
                "timestamp": pd.Timestamp(timestamps[symbol_id][position]),
                "fold": int(split["fold"]),
            }
            for symbol_id, positions in enumerate(valid_positions)
            for position in (
                positions if sampled[symbol_id] is None else positions[sampled[symbol_id]]
            )
        ]
        if rows:
            frames.append(pl.from_dicts(rows))
    if not frames:
        return pl.DataFrame(
            schema={"symbol": pl.String, "timestamp": pl.Datetime("ns"), "fold": pl.Int64}
        )
    expected = pl.concat(frames).sort("symbol", "timestamp", "fold")
    if expected.n_unique(["symbol", "timestamp", "fold"]) != expected.height:
        raise ValueError("sequence request produced duplicate expected prediction keys")
    return expected


def prepare_fold_sequence_stores(
    dataset_pd: pd.DataFrame,
    *,
    train_mask: pd.Series,
    val_mask: pd.Series,
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    lookback: int,
    max_train_sequences: int = 0,
    max_predict_sequences: int = 0,
    train_sequence_stride: int = 0,
    temporal_by_fold=None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    fold_id: int | None = None,
    val_start: pd.Timestamp | str | None = None,
    calendar_id: str | None = None,
    min_observed_fraction: float = DEFAULT_MIN_OBSERVED_FRACTION,
    max_consecutive_gap: int = DEFAULT_MAX_CONSECUTIVE_GAP,
) -> tuple[SequenceStore, SequenceStore, dict[str, int]]:
    """Build normalized train/validation sequence stores for a fold.

    When ``val_start`` is provided, validation sequences use each symbol's
    last ``lookback`` observable rows before validation plus its validation
    rows. This includes a label-buffer period after the training endpoint.
    Windows that cross a missing expected observation period are excluded.

    When ``val_start`` is None, val sequences start at position
    ``lookback`` within the val slice, which discards the first
    ``lookback`` trading days of each val fold. Callers should pass
    ``val_start`` so the val window aligns with production.
    """

    _ensure_sequence_periods(dataset_pd, date_col=date_col, calendar_id=calendar_id)
    feature_names = [name for name in feature_names if name not in GAP_MASK_FEATURES]
    use_cols = [date_col, entity_col, label_col, _SEQUENCE_PERIOD_COL, *feature_names]
    val_start_ts: pd.Timestamp | None
    if val_start is None:
        val_start_ts = None
    else:
        val_start_ts = _coerce_boundary(val_start, dataset_pd[date_col])

    if (
        temporal_by_fold is not None
        and temporal_keys
        and temporal_feature_names
        and fold_id is not None
    ):
        from utils.modeling import replace_temporal_columns

        train_df = replace_temporal_columns(
            dataset_pd,
            train_mask,
            temporal_by_fold,
            temporal_keys,
            temporal_feature_names,
            fold_id,
        )[use_cols].copy()

        if val_start_ts is not None:
            context_mask = dataset_pd[date_col] < val_start_ts
            full_val_source = replace_temporal_columns(
                dataset_pd,
                context_mask | val_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )[use_cols]
            val_df = _build_val_df_with_priming(
                full_val_source,
                entity_col=entity_col,
                date_col=date_col,
                val_start=val_start_ts,
                lookback=lookback,
                min_observed_fraction=min_observed_fraction,
            ).copy()
        else:
            val_df = replace_temporal_columns(
                dataset_pd,
                val_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )[use_cols].copy()
    else:
        train_df = dataset_pd.loc[train_mask, use_cols].copy()

        if val_start_ts is not None:
            context_mask = dataset_pd[date_col] < val_start_ts
            full_val_source = dataset_pd.loc[context_mask | val_mask, use_cols]
            val_df = _build_val_df_with_priming(
                full_val_source,
                entity_col=entity_col,
                date_col=date_col,
                val_start=val_start_ts,
                lookback=lookback,
            ).copy()
        else:
            val_df = dataset_pd.loc[val_mask, use_cols].copy()

    train_features, train_targets, train_timestamps, train_entities, train_positions = (
        _build_symbol_arrays(
            train_df,
            feature_names=feature_names,
            label_col=label_col,
            date_col=date_col,
            entity_col=entity_col,
            lookback=lookback,
            min_observed_fraction=min_observed_fraction,
            max_consecutive_gap=max_consecutive_gap,
        )
    )
    val_features, val_targets, val_timestamps, val_entities, val_positions = _build_symbol_arrays(
        val_df,
        feature_names=feature_names,
        label_col=label_col,
        date_col=date_col,
        entity_col=entity_col,
        lookback=lookback,
        min_observed_fraction=min_observed_fraction,
        max_consecutive_gap=max_consecutive_gap,
        min_end_timestamp=val_start_ts,
    )

    if not train_features or not val_features:
        empty = SequenceStore(
            [], [], [], [], np.array([], dtype=np.int32), np.array([], dtype=np.int32), lookback
        )
        return (
            empty,
            empty,
            {
                "train_symbols": len(train_entities),
                "val_symbols": len(val_entities),
                "train_sequences": 0,
                "val_sequences": 0,
            },
        )

    means, stds = _compute_feature_stats(train_features, n_passthrough=len(GAP_MASK_FEATURES))
    _normalize_feature_arrays(train_features, means, stds)
    _normalize_feature_arrays(val_features, means, stds)

    train_symbol_idx, train_end_idx = _build_sequence_index(
        train_positions, train_entities, max_train_sequences, train_sequence_stride
    )
    val_symbol_idx, val_end_idx = _build_sequence_index(
        val_positions, val_entities, max_predict_sequences
    )

    train_store = SequenceStore(
        features=train_features,
        targets=train_targets,
        timestamps=train_timestamps,
        entities=train_entities,
        symbol_idx=train_symbol_idx,
        end_idx=train_end_idx,
        lookback=lookback,
        feature_mean=means.copy(),
        feature_scale=stds.copy(),
    )
    val_store = SequenceStore(
        features=val_features,
        targets=val_targets,
        timestamps=val_timestamps,
        entities=val_entities,
        symbol_idx=val_symbol_idx,
        end_idx=val_end_idx,
        lookback=lookback,
        feature_mean=means.copy(),
        feature_scale=stds.copy(),
    )

    return (
        train_store,
        val_store,
        {
            "train_symbols": train_store.n_symbols,
            "val_symbols": val_store.n_symbols,
            "train_sequences": train_store.n_sequences,
            "val_sequences": val_store.n_sequences,
        },
    )
