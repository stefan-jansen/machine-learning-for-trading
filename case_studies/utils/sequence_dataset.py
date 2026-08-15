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
) -> list[np.ndarray | None]:
    """Sample sequence endpoints while preserving full symbol coverage."""

    sampled_positions: list[np.ndarray | None] = [None] * len(counts)
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


def _build_symbol_arrays(
    fold_df: pd.DataFrame,
    *,
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    lookback: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str], list[np.ndarray]]:
    """Convert a fold dataframe into per-symbol arrays and valid endpoints."""

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
    for symbol, sym_df in sorted_df.groupby(entity_col, sort=False):
        n_rows = len(sym_df)
        if n_rows < lookback + 1:
            continue
        feats = np.nan_to_num(sym_df[feature_names].to_numpy(dtype=np.float32), nan=0.0)
        targets = sym_df[label_col].to_numpy(dtype=np.float32)
        # Cast to datetime64[ns] explicitly so concat/np.asarray downstream
        # never falls back to object dtype.
        timestamps = sym_df[date_col].to_numpy(dtype="datetime64[ns]")
        periods = sym_df[_SEQUENCE_PERIOD_COL].to_numpy(dtype=np.int64)
        candidate_positions = np.arange(lookback, n_rows, dtype=np.int32)
        steps = np.diff(periods)
        bad_step_prefix = np.concatenate(([0], np.cumsum(steps != 1, dtype=np.int64)))
        gap_free = (
            bad_step_prefix[candidate_positions] == bad_step_prefix[candidate_positions - lookback]
        )
        target_is_finite = np.isfinite(targets[candidate_positions])
        valid_positions = candidate_positions[gap_free & target_is_finite]
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


def _compute_feature_stats(features_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std across raw training rows without concatenating arrays."""

    if not features_list:
        raise ValueError("No feature arrays available to compute scaling statistics")

    n_features = features_list[0].shape[1]
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)
    n_rows = 0

    for feats in features_list:
        sum_x += feats.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(feats, dtype=np.float64).sum(axis=0, dtype=np.float64)
        n_rows += feats.shape[0]

    means = sum_x / max(n_rows, 1)
    variances = np.maximum(sum_x2 / max(n_rows, 1) - np.square(means), 0.0)
    stds = np.sqrt(variances)
    stds[stds == 0] = 1.0
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


def _build_sequence_index(
    valid_positions_list: list[np.ndarray],
    entities: list[str],
    max_sequences: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat symbol/end-position indices for a sequence store."""

    counts = np.asarray([len(positions) for positions in valid_positions_list], dtype=np.int64)
    sampled_offsets = _sample_sequence_positions(counts, max_sequences)
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
) -> pd.DataFrame:
    """Per-symbol, keep last `lookback` pre-validation rows plus all validation rows.

    Pre-validation rows provide the input window for the first validation
    target prediction; their labels are not emitted as val targets because
    sequence positions start at index `lookback` within each symbol's
    sorted array, and with exactly `lookback` priming rows the first
    target falls at val_start.
    """
    pieces: list[pd.DataFrame] = []
    for _, sym_df in full_val_source.groupby(entity_col, sort=False):
        sym_df = sym_df.sort_values(date_col, kind="stable")
        is_val = sym_df[date_col] >= val_start
        context_tail = sym_df.loc[~is_val].tail(lookback)
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
) -> pl.DataFrame:
    """Return exact validation keys eligible for gap-free sequence prediction."""
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
        )
        _, _, timestamps, entities, valid_positions = _build_symbol_arrays(
            val_df,
            feature_names=[],
            label_col=label_col,
            date_col=date_col,
            entity_col=entity_col,
            lookback=lookback,
        )
        rows = [
            {
                "symbol": entities[symbol_id],
                "timestamp": pd.Timestamp(timestamps[symbol_id][position]),
                "fold": int(split["fold"]),
            }
            for symbol_id, positions in enumerate(valid_positions)
            for position in positions
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
    temporal_by_fold=None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    fold_id: int | None = None,
    val_start: pd.Timestamp | str | None = None,
    calendar_id: str | None = None,
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
        )
    )
    val_features, val_targets, val_timestamps, val_entities, val_positions = _build_symbol_arrays(
        val_df,
        feature_names=feature_names,
        label_col=label_col,
        date_col=date_col,
        entity_col=entity_col,
        lookback=lookback,
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

    means, stds = _compute_feature_stats(train_features)
    _normalize_feature_arrays(train_features, means, stds)
    _normalize_feature_arrays(val_features, means, stds)

    train_symbol_idx, train_end_idx = _build_sequence_index(
        train_positions, train_entities, max_train_sequences
    )
    val_symbol_idx, val_end_idx = _build_sequence_index(val_positions, val_entities, 0)

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
