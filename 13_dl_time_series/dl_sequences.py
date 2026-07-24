"""Multi-asset DL utilities for Chapter 13 notebooks.

This module provides canonical functions for:
- Loading case study data via load_modeling_dataset() (the shared Ch11+ API)
- Creating sequences for RNN/Transformer/CNN models
- Training models with early stopping (shared across notebooks)
- Creating time-based cross-validation folds
- Standardized prediction output

Usage:
    from dl_sequences import (
        # Data loading (thin wrapper around utils/modeling.py)
        load_dl_dataset,
        # Sequence creation
        create_sequences_multi_asset,
        create_patched_sequences_multi_asset,
        # Training
        train_model,
        # Cross-validation
        create_sequence_folds,
        create_expanding_folds,
        create_train_val_split,
        # Predictions
        make_predictions_df,
        save_predictions,
    )

Dataset IDs:
    Use canonical case study IDs (etfs, crypto_perps_funding) or short aliases (crypto, etf).
    See DATASET_ALIASES for the mapping.
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from utils.modeling import ModelingDataset, load_modeling_dataset
from utils.paths import get_case_study_dir

# =============================================================================
# Dataset Configuration
# =============================================================================

# Canonical case study IDs (match directory names in case_studies/)
CANONICAL_DATASET_IDS = {
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "cme_futures",
    "us_equities_panel",
    "us_firm_characteristics",
    "fx_pairs",
    "sp500_options",
    "sp500_equity_option_analytics",
}

# Short aliases and backward-compatible old IDs → canonical IDs
DATASET_ALIASES = {
    # Short aliases
    "crypto": "crypto_perps_funding",
    "etf": "etfs",
    "algoseek": "nasdaq100_microstructure",
    "futures": "cme_futures",
    "wiki": "us_equities_panel",
    "fx": "fx_pairs",
    # Backward-compatible old IDs
    "crypto_premium": "crypto_perps_funding",
    "etf_momentum": "etfs",
    "nasdaq100_reversal": "nasdaq100_microstructure",
    "futures_carry": "cme_futures",
    "us_factors": "us_equities_panel",
    "fx_momentum": "fx_pairs",
}

# Default primary labels per dataset
DEFAULT_LABELS = {
    "etfs": "fwd_ret_21d",
    "crypto_perps_funding": "fwd_ret_8h",
    "nasdaq100_microstructure": "fwd_ret_15m",
    "cme_futures": "fwd_ret_5d",
    "us_equities_panel": "fwd_ret_1d",
    "us_firm_characteristics": "fwd_ret_1m",
    "fx_pairs": "fwd_ret_1d",
    "sp500_options": "fwd_ret_dh_10d",
    "sp500_equity_option_analytics": "fwd_ret_5d",
}


def resolve_dataset_id(dataset: str) -> str:
    """Resolve a dataset name to its canonical case study ID.

    Args:
        dataset: Canonical ID (e.g., 'crypto_perps_funding'),
                 short alias (e.g., 'crypto'), or
                 old ID (e.g., 'crypto_premium') for backward compatibility.

    Returns:
        Canonical case study ID
    """
    if dataset in CANONICAL_DATASET_IDS:
        return dataset
    if dataset in DATASET_ALIASES:
        return DATASET_ALIASES[dataset]
    raise ValueError(
        f"Unknown dataset: {dataset!r}. "
        f"Valid IDs: {sorted(CANONICAL_DATASET_IDS)}. "
        f"Valid aliases: {sorted(DATASET_ALIASES.keys())}"
    )


# =============================================================================
# Data Loading (delegates to utils/modeling.py)
# =============================================================================


def load_dl_dataset(
    dataset: str,
    label: str | None = None,
    max_symbols: int = 0,
) -> ModelingDataset:
    """Load a modeling dataset for DL notebooks.

    Thin wrapper around load_modeling_dataset() that:
    - Resolves short aliases (e.g., 'crypto' → 'crypto_perps_funding')
    - Defaults to the primary label if none specified

    Args:
        dataset: Dataset name (canonical ID or alias)
        label: Label file stem (e.g., 'fwd_ret_8h'). None = primary label.
        max_symbols: Universe reduction for fast development. 0 = all.

    Returns:
        ModelingDataset with .dataset, .feature_names, .label_col,
        .date_col, .entity_cols, .splits, etc.
    """
    dataset_id = resolve_dataset_id(dataset)
    if label is None:
        label = DEFAULT_LABELS[dataset_id]

    mds = load_modeling_dataset(dataset_id, label, max_symbols=max_symbols)

    n_entities = mds.dataset[mds.entity_cols[0]].n_unique() if mds.entity_cols else 0
    print(
        f"Loaded {dataset_id}: {len(mds.dataset):,} rows, "
        f"{len(mds.feature_names)} features, "
        f"{n_entities} entities, label={mds.label_col}"
    )
    return mds


# =============================================================================
# Sequence Creation (for RNNs, Transformers, etc.)
# =============================================================================


def create_sequences_multi_asset(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences WITH symbol tracking for multi-asset learning.

    Creates sliding window sequences from each symbol independently,
    then pools them together while preserving symbol identity.

    Args:
        df: DataFrame with features, target, canonical time column, and asset
        feature_cols: List of feature column names
        target_col: Name of target column
        lookback: Number of timesteps in each sequence
        timestamp_col: Name of date/timestamp column
        symbol_col: Name of asset column

    Returns:
        Tuple of (X, y, timestamps, symbols):
        - X: np.ndarray of shape (n_samples, lookback, n_features)
        - y: np.ndarray of shape (n_samples,)
        - timestamps: np.ndarray of timestamps for each sample
        - symbols: np.ndarray of symbol names for each sample
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    dates_list: list[Any] = []
    symbols_list: list[str] = []

    symbols = df.select(symbol_col).unique().to_series().to_list()

    for symbol in symbols:
        sym_df = df.filter(pl.col(symbol_col) == symbol).sort(timestamp_col)

        if len(sym_df) < lookback + 1:
            continue

        features = sym_df.select(feature_cols).to_numpy()
        targets = sym_df[target_col].to_numpy()
        timestamps = sym_df[timestamp_col].to_numpy()

        for i in range(lookback, len(features)):
            X_list.append(features[i - lookback : i])
            y_list.append(float(targets[i]))
            dates_list.append(timestamps[i])
            symbols_list.append(symbol)

    if not X_list:
        raise ValueError(f"No sequences created. Check lookback={lookback} vs data size.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    timestamps_arr = np.array(dates_list)
    symbols_arr = np.array(symbols_list)

    return X, y, timestamps_arr, symbols_arr


def create_patched_sequences_multi_asset(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    patch_size: int,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create patched sequences for Transformer models.

    Patching groups consecutive timesteps into tokens for Transformer input.

    Args:
        df: DataFrame with features, target, canonical time column, and asset
        feature_cols: List of feature column names
        target_col: Name of target column
        lookback: Number of timesteps in each sequence
        patch_size: Size of each patch (must divide lookback evenly)
        timestamp_col: Name of date/timestamp column
        symbol_col: Name of asset column

    Returns:
        Tuple of (X, y, timestamps, symbols):
        - X: np.ndarray of shape (n_samples, n_patches, patch_size * n_features)
        - y, timestamps, symbols: as in create_sequences_multi_asset
    """
    if lookback % patch_size != 0:
        raise ValueError(f"lookback ({lookback}) must be divisible by patch_size ({patch_size})")

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    dates_list: list[Any] = []
    symbols_list: list[str] = []

    symbols = df.select(symbol_col).unique().to_series().to_list()
    n_patches = lookback // patch_size

    for symbol in symbols:
        sym_df = df.filter(pl.col(symbol_col) == symbol).sort(timestamp_col)

        if len(sym_df) < lookback + 1:
            continue

        features = sym_df.select(feature_cols).to_numpy()
        targets = sym_df[target_col].to_numpy()
        timestamps = sym_df[timestamp_col].to_numpy()

        for i in range(lookback, len(features)):
            seq = features[i - lookback : i]
            patched = seq.reshape(n_patches, patch_size * len(feature_cols))
            X_list.append(patched)
            y_list.append(float(targets[i]))
            dates_list.append(timestamps[i])
            symbols_list.append(symbol)

    if not X_list:
        raise ValueError(f"No sequences created. Check lookback={lookback} vs data size.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    timestamps_arr = np.array(dates_list)
    symbols_arr = np.array(symbols_list)

    return X, y, timestamps_arr, symbols_arr


# =============================================================================
# Cross-Validation
# =============================================================================


def create_sequence_folds(
    timestamps: np.ndarray,
    mds: ModelingDataset,
) -> list[dict[str, Any]]:
    """Create CV folds for sequence data using setup.yaml splits.

    Maps walk-forward fold boundaries (from mds.splits) to sequence data
    using the timestamps from create_sequences_multi_asset(). Ensures
    Ch13 DL models use the SAME fold boundaries as Ch11/Ch12.

    Args:
        timestamps: Array of timestamps from create_sequences_multi_asset()
        mds: ModelingDataset with .splits containing fold date boundaries

    Returns:
        List of fold dicts with 'fold_id', 'train_indices', 'test_indices'
    """
    import pandas as pd

    seq_timestamps = pd.to_datetime(timestamps)
    if seq_timestamps.tz is not None:
        seq_timestamps = seq_timestamps.tz_localize(None)

    folds = []
    for split in mds.splits:
        fold_id = split["fold"]
        train_end = pd.Timestamp(split["train_end"])
        val_start = pd.Timestamp(split["val_start"])
        val_end = pd.Timestamp(split["val_end"])
        # Normalize timezone awareness to match sequence timestamps
        if train_end.tz is not None:
            train_end = train_end.tz_localize(None)
            val_start = val_start.tz_localize(None)
            val_end = val_end.tz_localize(None)

        train_mask = seq_timestamps <= train_end
        test_mask = (seq_timestamps >= val_start) & (seq_timestamps <= val_end)

        train_indices = np.where(train_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()

        if len(train_indices) < 100 or len(test_indices) < 50:
            continue

        folds.append(
            {
                "fold_id": fold_id,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "train_end": train_end,
                "test_start": val_start,
                "test_end": val_end,
            }
        )

    return folds


def create_expanding_folds(
    n_samples: int,
    n_folds: int = 5,
    min_train_size: int = 100,
) -> list[dict[str, Any]]:
    """Create simple time-based expanding window folds.

    For quick experiments where exact fold matching is not required.
    Use create_sequence_folds() for Ch16-compatible results.

    Args:
        n_samples: Total number of samples
        n_folds: Number of folds to create
        min_train_size: Minimum training set size

    Returns:
        List of fold dicts with 'fold_id', 'train_indices', 'test_indices'
    """
    fold_size = n_samples // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n_samples)

        if train_end < min_train_size or test_end <= test_start:
            continue

        folds.append(
            {
                "fold_id": i,
                "train_indices": list(range(train_end)),
                "test_indices": list(range(test_start, test_end)),
            }
        )

    return folds


def create_train_val_split(
    train_indices: list[int],
    val_ratio: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Split training indices into train/validation (temporal split)."""
    n = len(train_indices)
    val_size = int(n * val_ratio)
    train_end = n - val_size
    return train_indices[:train_end], train_indices[train_end:]


# =============================================================================
# Prediction Output
# =============================================================================


def make_predictions_df(
    timestamps: np.ndarray,
    symbols: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    fold_id: int,
    model_id: str,
    horizon: str,
    dataset: str,
    time_col: str = "timestamp",
    asset_col: str = "symbol",
) -> pl.DataFrame:
    """Create standardized predictions DataFrame."""
    dataset_id = resolve_dataset_id(dataset)

    n = len(timestamps)
    return pl.DataFrame(
        {
            time_col: timestamps,
            asset_col: symbols,
            "y_true": y_true.astype(np.float64),
            "y_score": y_score.astype(np.float64),
            "fold_id": [fold_id] * n,
            "model_id": [model_id] * n,
            "horizon": [horizon] * n,
            "dataset": [dataset_id] * n,
        }
    )


def save_predictions(preds: pl.DataFrame, dataset: str, model_id: str) -> Path:
    """Save predictions to case study models/deep_learning directory.

    Path: case_studies/{dataset_id}/models/deep_learning/{model_id}_predictions.parquet

    Args:
        preds: Predictions DataFrame
        dataset: Dataset name (canonical ID or alias)
        model_id: Model identifier

    Returns:
        Path to saved file
    """
    dataset_id = resolve_dataset_id(dataset)
    case_dir = get_case_study_dir(dataset_id)
    output_dir = case_dir / "models" / "deep_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_id}_predictions.parquet"
    preds.write_parquet(output_path)
    print(f"Saved {len(preds):,} predictions to {output_path}")

    return output_path


def validate_predictions(
    preds: pl.DataFrame,
    require_multi_symbol: bool = True,
    time_col: str = "timestamp",
    asset_col: str = "symbol",
) -> None:
    """Validate prediction DataFrame schema and content."""
    required = {
        time_col,
        asset_col,
        "y_true",
        "y_score",
        "fold_id",
        "model_id",
        "horizon",
        "dataset",
    }
    missing = required - set(preds.columns)
    if missing:
        raise AssertionError(f"Missing columns: {missing}")

    n_symbols = preds[asset_col].n_unique()
    n_folds = preds["fold_id"].n_unique()

    if require_multi_symbol and n_symbols <= 1:
        raise AssertionError(f"Must have multiple symbols, got {n_symbols}")

    if n_folds < 1:
        raise AssertionError("Must have at least one fold")

    null_counts = preds.select(list(required)).null_count()
    total_nulls = null_counts.sum_horizontal().item()
    if total_nulls > 0:
        raise AssertionError(f"Found {total_nulls} null values in required columns")

    datasets = preds["dataset"].unique().to_list()
    for ds in datasets:
        if ds not in CANONICAL_DATASET_IDS:
            print(f"Warning: Non-canonical dataset ID: {ds}")

    print(f"Validated: {len(preds):,} rows, {n_symbols} symbols, {n_folds} folds")


def get_output_path(dataset: str, model_id: str) -> Path:
    """Get canonical output path for predictions."""
    dataset_id = resolve_dataset_id(dataset)
    case_dir = get_case_study_dir(dataset_id, create=False)
    return case_dir / "models" / "deep_learning" / f"{model_id}_predictions.parquet"


def load_predictions(dataset: str, model_id: str) -> pl.DataFrame:
    """Load predictions from canonical location."""
    path = get_output_path(dataset, model_id)
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    return pl.read_parquet(path)


# =============================================================================
# Training
# =============================================================================


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
    weight_decay: float = 0.0,
    patience: int = 5,
    log_interval: int = 5,
) -> dict[str, list[float]]:
    """Train a PyTorch model with early stopping.

    Uses AdamW optimizer (equivalent to Adam when weight_decay=0) with
    gradient clipping. The model is modified in-place: best weights are
    loaded via load_state_dict before returning.

    Args:
        model: PyTorch nn.Module to train
        X_train, y_train: Training arrays (numpy)
        X_val, y_val: Validation arrays (numpy)
        epochs: Maximum training epochs
        lr: Learning rate
        batch_size: Mini-batch size
        device: torch.device for computation
        weight_decay: AdamW weight decay (default 0 = plain Adam behavior)
        patience: Early stopping patience (epochs without improvement)
        log_interval: Print progress every N epochs

    Returns:
        Dict with 'train_loss' and 'val_loss' lists (per-epoch averages)
    """
    import torch
    import torch.nn as nn

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            optimizer.zero_grad()
            preds = model(X_train_t[batch_idx])
            loss = criterion(preds, y_train_t[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        model.eval()
        with torch.no_grad():
            val_sum = 0.0
            val_count = 0
            for i in range(0, len(X_val_t), batch_size):
                xb = X_val_t[i : i + batch_size]
                yb = y_val_t[i : i + batch_size]
                val_preds = model(xb)
                batch_loss = criterion(val_preds, yb).item()
                n_batch = len(xb)
                val_sum += batch_loss * n_batch
                val_count += n_batch
            val_loss = val_sum / max(val_count, 1)
            history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
