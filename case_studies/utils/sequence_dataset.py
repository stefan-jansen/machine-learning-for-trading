"""Shared lazy sequence datasets for PyTorch deep-learning case studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.modeling import RANDOM_SEED


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
    lookback: int,
    max_sequences: int,
) -> list[np.ndarray | None]:
    """Sample sequence endpoints while preserving full symbol coverage."""

    sampled_positions: list[np.ndarray | None] = [None] * len(counts)
    if max_sequences <= 0 or int(counts.sum()) <= max_sequences:
        return sampled_positions

    n_symbols = len(counts)
    if max_sequences < n_symbols:
        raise ValueError(
            f"max_sequences={max_sequences:,} is smaller than the symbol count "
            f"({n_symbols:,}); cannot preserve full universe coverage"
        )

    alloc = np.ones(n_symbols, dtype=np.int64)
    remaining = max_sequences - n_symbols
    extra_capacity = counts - 1

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
        sampled_positions[idx] = offsets + lookback

    return sampled_positions


def _build_symbol_arrays(
    fold_df: pd.DataFrame,
    *,
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    lookback: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str], np.ndarray]:
    """Convert a fold dataframe into per-symbol arrays and sequence counts."""

    if fold_df.empty:
        return [], [], [], [], np.array([], dtype=np.int64)

    features_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    timestamps_list: list[np.ndarray] = []
    entities: list[str] = []
    counts: list[int] = []

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
        features_list.append(feats)
        targets_list.append(targets)
        timestamps_list.append(timestamps)
        entities.append(str(symbol))
        counts.append(n_rows - lookback)

    return (
        features_list,
        targets_list,
        timestamps_list,
        entities,
        np.asarray(counts, dtype=np.int64),
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
    counts: np.ndarray,
    entities: list[str],
    lookback: int,
    max_sequences: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat symbol/end-position indices for a sequence store."""

    sampled_positions = _sample_sequence_positions(counts, lookback, max_sequences)
    symbol_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []

    for symbol_id, (entity, n_seq, positions) in enumerate(
        zip(entities, counts, sampled_positions, strict=True)
    ):
        del entity  # entity order is already captured by the symbol_id list
        if positions is None:
            positions = np.arange(lookback, lookback + n_seq, dtype=np.int32)
        else:
            positions = positions.astype(np.int32, copy=False)
        symbol_parts.append(np.full(len(positions), symbol_id, dtype=np.int32))
        end_parts.append(positions)

    symbol_idx = np.concatenate(symbol_parts) if symbol_parts else np.array([], dtype=np.int32)
    end_idx = np.concatenate(end_parts) if end_parts else np.array([], dtype=np.int32)
    return symbol_idx, end_idx


def _build_val_df_with_priming(
    full_val_source: pd.DataFrame,
    *,
    entity_col: str,
    date_col: str,
    val_start: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    """Per-symbol, keep last `lookback` train-tail rows + all val rows.

    Train-tail rows provide the input window (priming) for the first val
    target prediction; their labels are not emitted as val targets because
    sequence positions start at index `lookback` within each symbol's
    sorted array, and with exactly `lookback` priming rows the first
    target falls at val_start.
    """
    pieces: list[pd.DataFrame] = []
    for _, sym_df in full_val_source.groupby(entity_col, sort=False):
        sym_df = sym_df.sort_values(date_col, kind="stable")
        is_val = sym_df[date_col] >= val_start
        train_tail = sym_df.loc[~is_val].tail(lookback)
        val_part = sym_df.loc[is_val]
        if train_tail.empty and val_part.empty:
            continue
        pieces.append(pd.concat([train_tail, val_part], ignore_index=True))
    if not pieces:
        return full_val_source.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


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
) -> tuple[SequenceStore, SequenceStore, dict[str, int]]:
    """Build normalized train/validation sequence stores for a fold.

    When ``val_start`` is provided, val sequences are built from the
    concatenation of (a) each symbol's last ``lookback`` train-tail rows
    and (b) its val-period rows. This "train-tail priming" ensures the
    first val sequence predicts the target at ``val_start`` — matching
    production behavior and Chapter 13's teaching implementation.

    When ``val_start`` is None, val sequences start at position
    ``lookback`` within the val slice, which discards the first
    ``lookback`` trading days of each val fold. Callers should pass
    ``val_start`` so the val window aligns with production.
    """

    use_cols = [date_col, entity_col, label_col, *feature_names]
    val_start_ts: pd.Timestamp | None
    if val_start is None:
        val_start_ts = None
    else:
        val_start_ts = pd.Timestamp(val_start)
        # Match the date column's timezone — crypto's timestamps are
        # tz-aware (datetime64[ms, UTC]); a tz-naive literal raises
        # `Invalid comparison between dtype=datetime64[ms, UTC] and Timestamp`.
        col_tz = getattr(dataset_pd[date_col].dtype, "tz", None)
        if col_tz is not None and val_start_ts.tz is None:
            val_start_ts = val_start_ts.tz_localize(col_tz)
        elif col_tz is None and val_start_ts.tz is not None:
            val_start_ts = val_start_ts.tz_localize(None)

    if (
        temporal_by_fold is not None
        and temporal_keys
        and temporal_feature_names
        and fold_id is not None
    ):
        from utils.modeling import _replace_temporal_columns

        train_df = (
            _replace_temporal_columns(
                dataset_pd,
                train_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )[use_cols]
            .dropna(subset=[label_col])
            .copy()
        )

        if val_start_ts is not None:
            # Source = train-rows-before-val + val-rows (temporal columns
            # replaced consistently per fold across both halves).
            train_tail_mask = train_mask & (dataset_pd[date_col] < val_start_ts)
            full_val_source = _replace_temporal_columns(
                dataset_pd,
                train_tail_mask | val_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )[use_cols].dropna(subset=[label_col])
            val_df = _build_val_df_with_priming(
                full_val_source,
                entity_col=entity_col,
                date_col=date_col,
                val_start=val_start_ts,
                lookback=lookback,
            ).copy()
        else:
            val_df = (
                _replace_temporal_columns(
                    dataset_pd,
                    val_mask,
                    temporal_by_fold,
                    temporal_keys,
                    temporal_feature_names,
                    fold_id,
                )[use_cols]
                .dropna(subset=[label_col])
                .copy()
            )
    else:
        train_df = dataset_pd.loc[train_mask, use_cols].dropna(subset=[label_col]).copy()

        if val_start_ts is not None:
            train_tail_mask = train_mask & (dataset_pd[date_col] < val_start_ts)
            full_val_source = dataset_pd.loc[train_tail_mask | val_mask, use_cols].dropna(
                subset=[label_col]
            )
            val_df = _build_val_df_with_priming(
                full_val_source,
                entity_col=entity_col,
                date_col=date_col,
                val_start=val_start_ts,
                lookback=lookback,
            ).copy()
        else:
            val_df = dataset_pd.loc[val_mask, use_cols].dropna(subset=[label_col]).copy()

    train_features, train_targets, train_timestamps, train_entities, train_counts = (
        _build_symbol_arrays(
            train_df,
            feature_names=feature_names,
            label_col=label_col,
            date_col=date_col,
            entity_col=entity_col,
            lookback=lookback,
        )
    )
    val_features, val_targets, val_timestamps, val_entities, val_counts = _build_symbol_arrays(
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
        train_counts, train_entities, lookback, max_train_sequences
    )
    val_symbol_idx, val_end_idx = _build_sequence_index(val_counts, val_entities, lookback, 0)

    train_store = SequenceStore(
        features=train_features,
        targets=train_targets,
        timestamps=train_timestamps,
        entities=train_entities,
        symbol_idx=train_symbol_idx,
        end_idx=train_end_idx,
        lookback=lookback,
    )
    val_store = SequenceStore(
        features=val_features,
        targets=val_targets,
        timestamps=val_timestamps,
        entities=val_entities,
        symbol_idx=val_symbol_idx,
        end_idx=val_end_idx,
        lookback=lookback,
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
