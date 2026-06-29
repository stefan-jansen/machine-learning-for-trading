"""Shared deep learning pipeline infrastructure for Ch13 case study templates.

Provides:
- create_model(): Factory for PyTorch DL architectures (nlinear, lstm, patchtst, tcn)
- run_dl_cv(): Walk-forward CV with epoch-checkpoint IC evaluation

Cross-sectional IC computation is delegated to
``ml4t.diagnostic.metrics.cross_sectional_ic`` against polars frames of
(date, symbol, y_true, y_pred).

Usage:
    from case_studies.utils.deep_learning import run_dl_cv
    from utils.modeling import load_configs

    dl_configs = load_configs("etfs", "fwd_ret_21d", "deep_learning")
    result = run_dl_cv(dataset_pd, splits, configs=dl_configs,
                       n_features=44, feature_names=..., label_col=...)
"""

from __future__ import annotations

import gc
import time
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic
from torch.utils.data import DataLoader

from case_studies.utils.registry import compute_fold_metrics_from_predictions
from case_studies.utils.registry.store import (
    _save_parquet,
    flush_fold_predictions,
    flush_fold_training_log,
)
from case_studies.utils.sequence_dataset import (
    FoldSequenceDataset,
    collate_with_metadata,
    materialize_store_metadata,
    prepare_fold_sequence_stores,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from utils.modeling import RANDOM_SEED, seed_everything

# ---------------------------------------------------------------------------
# Model Factory (lazy imports — models package may not be deployed)
# ---------------------------------------------------------------------------


def _get_model_registry() -> dict[str, type[nn.Module]]:
    """Lazy-load model classes from case_studies/config/{model_type}/."""
    from case_studies.config.lstm.lstm import LSTMRegressor
    from case_studies.config.nlinear.nlinear import NLinear
    from case_studies.config.patchtst.patchtst import PatchTST
    from case_studies.config.tcn.tcn import TCNRegressor
    from case_studies.config.tsmixer.tsmixer import TSMixerRegressor

    return {
        "nlinear": NLinear,
        "lstm": LSTMRegressor,
        "patchtst": PatchTST,
        "tcn": TCNRegressor,
        "tsmixer": TSMixerRegressor,
    }


def create_model(
    name: str,
    config: dict[str, Any],
) -> nn.Module:
    """Create a DL model by name.

    Parameters
    ----------
    name : str
        Architecture name: "nlinear", "lstm", "patchtst", "tcn", "tsmixer".
    config : dict
        Architecture-specific kwargs passed to constructor.

    Returns
    -------
    nn.Module
    """
    registry = _get_model_registry()
    if name not in registry:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(registry.keys())}")
    return registry[name](**config)


# ---------------------------------------------------------------------------
# Config → Architecture Kwargs
# ---------------------------------------------------------------------------

# Architecture → constructor dimension params (runtime-injected)
_DIM_INJECT: dict[str, dict[str, str]] = {
    "lstm": {"input_size": "n_features"},
    "nlinear": {"n_features": "n_features", "lookback": "lookback"},
    "patchtst": {"n_features": "n_features", "lookback": "lookback"},
    "tcn": {"n_features": "n_features"},
    "tsmixer": {"n_features": "n_features", "seq_len": "lookback"},
}


def _build_arch_kwargs(cfg: dict[str, Any], n_features: int, lookback: int) -> dict[str, Any]:
    """Extract architecture constructor kwargs from a config dict.

    Takes the YAML ``params`` dict, removes ``architecture`` and ``lookback``
    (metadata/training fields), and injects runtime dimensions (n_features,
    lookback) as the architecture's constructor expects them.
    """
    params = dict(cfg.get("params", {}))
    params.pop("architecture", None)
    params.pop("lookback", None)

    dim_vals = {"n_features": n_features, "lookback": lookback}
    arch = cfg["params"].get("architecture", _resolve_arch_name(cfg["config_name"]))
    for param_name, source_key in _DIM_INJECT.get(arch, {}).items():
        params[param_name] = dim_vals[source_key]

    return params


# ---------------------------------------------------------------------------
# MC Dropout Inference
# ---------------------------------------------------------------------------


def mc_dropout_predict(
    model: nn.Module,
    X: torch.Tensor,
    n_samples: int = 50,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """MC Dropout: keep dropout active during inference, run N forward passes.

    Parameters
    ----------
    model : nn.Module
        Trained model with dropout layers.
    X : torch.Tensor
        Input tensor on the target device.
    n_samples : int
        Number of stochastic forward passes.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    mean_pred : np.ndarray
        Mean prediction across MC samples.
    std_pred : np.ndarray
        Standard deviation (epistemic uncertainty estimate).
    """
    # Enable dropout during inference
    model.train()
    for m in model.modules():
        if not isinstance(m, nn.Dropout):
            m.eval()

    device = next(model.parameters()).device
    all_preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            preds = []
            for start in range(0, len(X), batch_size):
                batch = X[start : start + batch_size].to(device)
                preds.append(model(batch).cpu().numpy())
            all_preds.append(np.concatenate(preds))

    model.eval()
    all_preds_arr = np.stack(all_preds, axis=0)  # (n_samples, n_obs)
    return all_preds_arr.mean(axis=0), all_preds_arr.std(axis=0)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def _train_one_config(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    checkpoint_interval: int,
    device: torch.device,
    checkpoint_callback: Callable[[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray], None]
    | None = None,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[int, float], dict[int, np.ndarray], dict[int, float]]:
    """Train a single model config, storing predictions at ALL checkpoints.

    Trains to completion (no early stopping). Stores predictions at every
    checkpoint so the caller can select the best epoch after all folds finish.

    Returns
    -------
    checkpoint_ics : dict[epoch, ic]
    checkpoint_preds : dict[epoch, np.ndarray]
    epoch_losses : dict[epoch, avg_loss]
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    checkpoint_ics: dict[int, float] = {}
    checkpoint_preds: dict[int, np.ndarray] = {}
    epoch_losses: dict[int, float] = {}

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        # Mini-batch training
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for n_batches, (X_batch, y_batch) in enumerate(train_loader, 1):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_losses[epoch] = avg_loss

        # Evaluate and store predictions at checkpoint epochs
        if epoch % checkpoint_interval == 0 or epoch == n_epochs:
            model.eval()
            y_parts: list[np.ndarray] = []
            pred_parts: list[np.ndarray] = []
            date_parts: list[np.ndarray] = []
            entity_parts: list[np.ndarray] = []
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for X_batch, y_batch, timestamps, entities in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch_dev = y_batch.to(device, non_blocking=True)
                    pred_batch = model(X_batch)
                    pred_parts.append(pred_batch.cpu().numpy())
                    y_parts.append(y_batch.numpy())
                    date_parts.append(timestamps)
                    entity_parts.append(entities)
                    val_loss += criterion(pred_batch, y_batch_dev).item()
                    val_batches += 1

            val_preds = np.concatenate(pred_parts)
            y_val = np.concatenate(y_parts)
            val_dates = np.concatenate(date_parts)
            val_entities = np.concatenate(entity_parts)
            avg_val_loss = val_loss / max(val_batches, 1)

            ic_frame = pl.DataFrame(
                {
                    "date": val_dates,
                    "symbol": val_entities,
                    "y_true": y_val,
                    "y_pred": val_preds,
                }
            )
            ic = cross_sectional_ic(
                ic_frame,
                ic_frame,
                pred_col="y_pred",
                ret_col="y_true",
                date_col="date",
                entity_col="symbol",
                min_obs=5,
            )["ic_mean"]
            checkpoint_ics[epoch] = ic
            checkpoint_preds[epoch] = val_preds.copy()
            if checkpoint_callback is not None:
                checkpoint_callback(checkpoint_preds, y_val, val_dates, val_entities)
            if epoch_callback is not None:
                epoch_callback(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "val_loss": avg_val_loss,
                        "ic": ic,
                        "epoch_s": time.time() - epoch_start,
                    }
                )
            print(
                "      epoch "
                f"{epoch:3d}/{n_epochs}: "
                f"train_loss={avg_loss:.6f}, val_loss={avg_val_loss:.6f}, IC={ic:+.4f}",
                flush=True,
            )
        else:
            if epoch_callback is not None:
                epoch_callback(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "val_loss": None,
                        "ic": None,
                        "epoch_s": time.time() - epoch_start,
                    }
                )
            print(f"      epoch {epoch:3d}/{n_epochs}: train_loss={avg_loss:.6f}", flush=True)

    return checkpoint_ics, checkpoint_preds, epoch_losses


# ---------------------------------------------------------------------------
# Incremental Save Helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config Name → Architecture Resolution
# ---------------------------------------------------------------------------

# Maps grid config names to model registry keys
_CONFIG_ARCH_MAP: dict[str, str] = {
    "nlinear": "nlinear",
    "lstm_h64": "lstm",
    "lstm_h128": "lstm",
    "patchtst": "patchtst",
    "tcn": "tcn",
    "tsmixer": "tsmixer",
}


def _resolve_arch_name(config_name: str) -> str:
    """Map a grid config name (e.g., 'lstm_h64') to a model registry key ('lstm')."""
    if config_name in _CONFIG_ARCH_MAP:
        return _CONFIG_ARCH_MAP[config_name]
    # Fallback: try prefix match
    for prefix in _get_model_registry():
        if config_name.startswith(prefix):
            return prefix
    raise ValueError(f"Cannot resolve architecture for config: {config_name!r}")


# ---------------------------------------------------------------------------
# Registry Integration
# ---------------------------------------------------------------------------


def _register_dl_config(
    *,
    case_study: str,
    label: str,
    config_name: str,
    architecture: str,
    n_epochs: int | None,
    best_epoch: int,
    lookback: int,
    n_folds: int,
    ic_mean: float,
    predictions,
    notebook: str | None = None,
    learning_curves=None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
    prediction_split: str = "validation",
) -> str:
    """Register a single DL config — thin delegate to register_epoch_checkpoint."""
    from case_studies.utils.registry import register_epoch_checkpoint

    return register_epoch_checkpoint(
        case_study,
        family="deep_learning",
        library="pytorch",
        config_name=config_name,
        label=label,
        n_folds=n_folds,
        n_epochs=n_epochs,
        best_epoch=best_epoch,
        ic_mean=ic_mean,
        predictions=predictions,
        extra_params={"architecture": architecture, "lookback": lookback},
        learning_curves=learning_curves,
        entry_point=notebook,
        started_at=started_at,
        elapsed_s=elapsed_s,
        prediction_split=prediction_split,
    )


# ---------------------------------------------------------------------------
# Main CV Pipeline
# ---------------------------------------------------------------------------


def run_dl_cv(
    dataset_pd: pd.DataFrame,
    splits: list[dict[str, Any]],
    *,
    configs: list[dict[str, Any]],
    n_features: int,
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str = "symbol",
    device: str = "cuda",
    save_dir: Path | None = None,
    max_train_sequences: int = 0,
    register: bool = False,
    case_study: str | None = None,
    notebook: str | None = None,
    selected_folds: list[int] | None = None,
    temporal_by_fold=None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    force_retrain: bool = False,
    prediction_split: str = "validation",
) -> dict[str, Any]:
    """Walk-forward DL CV with epoch-checkpoint IC evaluation.

    All training parameters (n_epochs, batch_size, lookback, checkpoint_interval)
    are read from each config dict. Config dicts come from ``load_configs()``.

    Parameters
    ----------
    dataset_pd : pandas DataFrame
        Full dataset with features, label, date, and entity columns.
    splits : list[dict]
        Walk-forward splits from generate_cv_splits().
    configs : list[dict]
        Config dicts from ``load_configs()``. Each must have ``config_name``,
        ``params`` (with ``architecture`` and architecture-specific kwargs),
        and training params: ``n_epochs``, ``batch_size``, ``checkpoint_interval``.
        ``params.lookback`` controls sequence length.
    n_features : int
        Number of input features (for architecture constructor injection).
    feature_names : list[str]
        Column names to use as features.
    label_col : str
        Target column name.
    date_col : str
        Date/timestamp column name.
    entity_col : str
        Entity column name (default "symbol").
    device : str
        "cuda" or "cpu".
    save_dir : Path, optional
        Directory to save model checkpoints and predictions.

    Returns
    -------
    dict with keys:
        grid_results: list[dict] — per-config results ranked by best IC
        best_config_name: str
        best_epoch: int
        best_ic: float
        predictions: pl.DataFrame — OOS predictions from best config
        all_predictions: pl.DataFrame — predictions for ALL configs × epoch checkpoints
        fold_metrics: pl.DataFrame — per-fold cross-sectional IC for best config
        all_learning_curves: pl.DataFrame — IC × epoch × config
    """
    from case_studies.utils.darts_forecasting import run_darts_cv, uses_darts_backend

    if register and save_dir is None:
        raise ValueError(
            "register=True requires save_dir for incremental prediction saves. "
            "Pass save_dir=CASE_DIR / 'run_log' / 'training' / 'deep_learning'"
        )

    # Filter out configs whose training_hash is already complete (unless
    # force_retrain). Fold-major training can't skip individual configs
    # mid-fold, so the filter happens BEFORE the fold loop starts.
    if register and case_study and not force_retrain:
        from case_studies.utils.registry import (
            build_training_spec,
            load_prediction_sets,
            training_hash_from_spec,
            training_run_status,
        )

        pending_configs = []
        for cfg in configs:
            try:
                spec = build_training_spec(
                    cfg["family"],
                    cfg["config_name"],
                    label_col,
                    n_folds=len(splits),
                    n_epochs=cfg.get("n_epochs"),
                )
                status = training_run_status(case_study, spec)
                split_rows = load_prediction_sets(
                    case_study,
                    training_hash=training_hash_from_spec(spec),
                    split=prediction_split,
                )
                split_complete = not split_rows.is_empty()
                if status.complete and split_complete:
                    print(
                        f"  SKIP {cfg['config_name']:25s}  ({status.summary()}, split={prediction_split})"
                    )
                    continue
                if status.complete and not split_complete:
                    print(
                        f"  RETRAIN {cfg['config_name']:25s}  missing {prediction_split} predictions"
                    )
                elif status.partial:
                    print(f"  RETRAIN {cfg['config_name']:25s}  partial state: {status.summary()}")
            except Exception as exc:
                print(f"  WARN: skip-check failed for {cfg['config_name']}: {exc}")
            pending_configs.append(cfg)

        if not pending_configs:
            print("All configs already complete — nothing to do.")
            return {
                "grid_results": [],
                "best_config_name": None,
                "best_epoch": 0,
                "best_ic": float("nan"),
                "predictions": pl.DataFrame(),
                "all_predictions": pl.DataFrame(),
                "fold_metrics": pl.DataFrame(),
                "all_learning_curves": pl.DataFrame(),
                "training_log": pl.DataFrame(),
            }
        configs = pending_configs

    if uses_darts_backend(configs):
        return run_darts_cv(
            dataset_pd,
            splits,
            configs=configs,
            feature_names=feature_names,
            label_col=label_col,
            date_col=date_col,
            entity_col=entity_col,
            device=device,
            save_dir=save_dir,
            max_train_sequences=max_train_sequences,
            register=register,
            case_study=case_study,
            notebook=notebook,
            prediction_split=prediction_split,
        )

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    expected_fold_ids = [int(split["fold"]) for split in splits]
    if selected_folds:
        selected = {int(fold) for fold in selected_folds}
        splits = [split for split in splits if int(split["fold"]) in selected]
        print(f"Selected folds: {sorted(selected)}")
        if not splits:
            raise ValueError(f"No splits matched selected_folds={selected_folds}")

    seed_everything(RANDOM_SEED)

    # Extract lookback from configs (must be uniform for fold-major sequencing)
    lookback = configs[0].get("params", {}).get("lookback", 60)

    dates_series = dataset_pd[date_col]

    # Fold-major grid search: create sequences one fold at a time, train all
    # configs on that fold, then free the data before moving to the next fold.
    # This keeps only ONE fold's tensors in memory at a time — critical for
    # large datasets (e.g. us_equities_panel: 9M+ rows × 16 folds).
    print(f"Fold-major CV: {len(splits)} folds × {len(configs)} configs × {lookback} lookback")

    # Per-config accumulator (lightweight — stores only predictions and ICs)
    config_acc: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        config_acc[cfg["config_name"]] = {
            "fold_checkpoint_ics": {},  # {epoch: [ic_per_fold]}
            "preds": [],
            "elapsed_s": 0.0,
            "started_at": None,
        }

    n_valid_folds = 0

    _has_fold_temporal = temporal_by_fold is not None and temporal_keys and temporal_feature_names

    for split in splits:
        seed_everything(RANDOM_SEED + split["fold"])

        train_mask = (dates_series >= split["train_start"]) & (dates_series <= split["train_end"])
        val_mask = (dates_series >= split["val_start"]) & (dates_series <= split["val_end"])

        print(f"\n  Fold {split['fold']}: creating sequences...")
        train_store, val_store, fold_info = prepare_fold_sequence_stores(
            dataset_pd,
            train_mask=train_mask,
            val_mask=val_mask,
            feature_names=feature_names,
            label_col=label_col,
            date_col=date_col,
            entity_col=entity_col,
            lookback=lookback,
            max_train_sequences=max_train_sequences,
            temporal_by_fold=temporal_by_fold if _has_fold_temporal else None,
            temporal_keys=temporal_keys,
            temporal_feature_names=temporal_feature_names,
            fold_id=split["fold"],
            val_start=split["val_start"],
        )

        if fold_info["train_sequences"] < 100 or fold_info["val_sequences"] < 50:
            print(
                "    Skipped "
                f"(train={fold_info['train_sequences']}, val={fold_info['val_sequences']})"
            )
            continue

        print(
            f"    train={fold_info['train_sequences']:,} seq across "
            f"{fold_info['train_symbols']} symbols"
        )
        print(
            f"    val={fold_info['val_sequences']:,} seq across {fold_info['val_symbols']} symbols"
        )
        print("    creating datasets...")

        train_ds = FoldSequenceDataset(train_store)
        val_ds = FoldSequenceDataset(val_store, include_metadata=True)

        n_train_seq = len(train_ds)
        n_val_seq = len(val_ds)
        n_valid_folds += 1
        print("    datasets ready")

        # Train ALL configs on this fold before freeing fold data
        for cfg in configs:
            config_name = cfg["config_name"]
            cfg_n_epochs = cfg.get("n_epochs", 100)
            cfg_batch_size = cfg.get("batch_size", 2048)
            cfg_checkpoint = cfg.get("checkpoint_interval", 5)
            arch_name = cfg["params"].get("architecture", _resolve_arch_name(config_name))
            arch_kwargs = _build_arch_kwargs(cfg, n_features, lookback)

            if config_acc[config_name]["started_at"] is None:
                config_acc[config_name]["started_at"] = datetime.now(UTC).isoformat()
            t0 = time.perf_counter()
            print(f"    {config_name}:")

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch_device.type == "cuda",
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch_device.type == "cuda",
                collate_fn=collate_with_metadata,
            )

            model = create_model(arch_name, arch_kwargs)

            epoch_rows: list[dict[str, Any]] = []
            incr_dir = save_dir / "_incremental" if save_dir is not None else None
            log_dir = save_dir / "_incremental_logs" if save_dir is not None else None
            y_val_store = val_dates_store = val_entities_store = None
            if incr_dir is not None:
                incr_dir.mkdir(parents=True, exist_ok=True)
                y_val_store, val_dates_store, val_entities_store = materialize_store_metadata(
                    val_store
                )
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)

            def _on_checkpoint(
                checkpoint_preds_so_far: dict[int, np.ndarray],
                y_val_full: np.ndarray,
                val_dates_full: np.ndarray,
                val_entities_full: np.ndarray,
                *,
                _incr_dir: Path | None = incr_dir,
                _config_name: str = config_name,
                _fold: int = split["fold"],
            ) -> None:
                if _incr_dir is None:
                    return
                flush_fold_predictions(
                    _incr_dir,
                    _config_name,
                    _fold,
                    checkpoint_preds_so_far,
                    val_dates_full,
                    val_entities_full,
                    y_val_full,
                    date_col,
                    entity_col,
                )

            def _on_epoch(
                epoch_row: dict[str, Any],
                *,
                _config_name: str = config_name,
                _fold: int = split["fold"],
                _n_train_seq: int = n_train_seq,
                _n_val_seq: int = n_val_seq,
                _epoch_rows: list[dict[str, Any]] = epoch_rows,
                _log_dir: Path | None = log_dir,
            ) -> None:
                row = {
                    "config": _config_name,
                    "fold": _fold,
                    "epoch": int(epoch_row["epoch"]),
                    "train_loss": float(epoch_row["train_loss"]),
                    "val_loss": (
                        float(epoch_row["val_loss"]) if epoch_row["val_loss"] is not None else None
                    ),
                    "ic": float(epoch_row["ic"]) if epoch_row["ic"] is not None else None,
                    "epoch_s": float(epoch_row["epoch_s"]),
                    "n_train": _n_train_seq,
                    "n_val": _n_val_seq,
                }
                _epoch_rows.append(row)
                if _log_dir is not None:
                    flush_fold_training_log(_log_dir, _config_name, _fold, _epoch_rows)

            checkpoint_ics, checkpoint_preds, epoch_losses = _train_one_config(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=cfg_n_epochs,
                checkpoint_interval=cfg_checkpoint,
                device=torch_device,
                checkpoint_callback=_on_checkpoint,
                epoch_callback=_on_epoch,
            )

            elapsed = time.perf_counter() - t0
            acc = config_acc[config_name]
            acc["elapsed_s"] += elapsed

            # Accumulate per-checkpoint ICs across folds
            for ep, ic in checkpoint_ics.items():
                acc["fold_checkpoint_ics"].setdefault(ep, []).append(ic)

            best_ep = max(checkpoint_ics, key=lambda e: checkpoint_ics[e])
            for row in epoch_rows:
                row["elapsed_s"] = round(elapsed, 1)
                row["best_epoch"] = best_ep
                row["best_ic"] = float(checkpoint_ics[best_ep])
            acc.setdefault("training_log", []).extend(epoch_rows)

            del model, checkpoint_preds, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                f"      best_ep={best_ep}, "
                f"IC={checkpoint_ics[best_ep]:+.4f} "
                f"({elapsed:.1f}s, {len(checkpoint_ics)} checkpoints)"
            )

        # Free this fold's data before creating next fold's sequences
        del train_ds, val_ds, train_store, val_store
        gc.collect()

    if n_valid_folds == 0:
        raise ValueError("No valid folds created. Check data size vs lookback.")

    # Reassemble all predictions from incremental saves
    incr_dir = save_dir / "_incremental" if save_dir is not None else None
    if incr_dir is not None and incr_dir.exists():
        parquet_files = sorted(incr_dir.glob("*.parquet"))
        all_predictions = (
            pl.concat(
                [
                    pl.read_parquet(f).cast({"timestamp": pl.Datetime("us")}, strict=False)
                    for f in parquet_files
                ],
                how="diagonal_relaxed",
            )
            if parquet_files
            else pl.DataFrame()
        )
    else:
        all_predictions = pl.DataFrame()

    # --- Aggregate results per config (post-processing) ---
    config_results: list[dict[str, Any]] = []
    all_curves: list[dict] = []
    training_log: list[dict] = []
    complete_prediction_frames: list[pl.DataFrame] = []
    log_dir = save_dir / "_incremental_logs" if save_dir is not None else None
    incremental_logs = pl.DataFrame()
    if log_dir is not None and log_dir.exists():
        log_files = sorted(log_dir.glob("*.parquet"))
        if log_files:
            incremental_logs = pl.concat(
                [pl.read_parquet(f) for f in log_files], how="diagonal_relaxed"
            )

    for cfg in configs:
        config_name = cfg["config_name"]
        acc = config_acc[config_name]
        cfg_preds = (
            all_predictions.filter(pl.col("config") == config_name)
            if all_predictions.height > 0
            else pl.DataFrame()
        )

        epoch_scores: list[tuple[int, float, float]] = []
        if cfg_preds.height > 0:
            for epoch in sorted(cfg_preds["epoch"].unique().to_list()):
                ep_df = cfg_preds.filter(pl.col("epoch") == epoch)
                fold_ids = sorted(ep_df["fold_id"].unique().to_list())
                if fold_ids != expected_fold_ids:
                    continue
                fold_ics = []
                for fold_id in expected_fold_ids:
                    fold_df = ep_df.filter(pl.col("fold_id") == fold_id)
                    _entity = entity_col if entity_col in fold_df.columns else None
                    ic = cross_sectional_ic(
                        fold_df,
                        fold_df,
                        pred_col="y_score",
                        ret_col="y_true",
                        date_col=date_col,
                        entity_col=_entity,
                        method="spearman",
                        min_obs=5,
                    )["ic_mean"]
                    fold_ics.append(ic)
                ic_mean = float(np.nanmean(fold_ics))
                ic_std = float(np.nanstd(fold_ics)) if len(fold_ics) > 1 else 0.0
                all_curves.append(
                    {
                        "config": config_name,
                        "epoch": epoch,
                        "ic_mean": ic_mean,
                        "ic_std": ic_std,
                    }
                )
                complete_prediction_frames.append(ep_df)
                epoch_scores.append((epoch, ic_mean, ic_std))

        if epoch_scores:
            best_cp, best_ic_val, _best_ic_std = max(epoch_scores, key=lambda item: item[1])
        else:
            best_cp = 0
            best_ic_val = float("nan")

        config_results.append(
            {
                "config_name": config_name,
                "best_epoch": best_cp,
                "best_ic": best_ic_val,
                "elapsed_s": acc["elapsed_s"],
                "started_at": acc["started_at"],
            }
        )

        if incremental_logs.height > 0:
            cfg_logs = incremental_logs.filter(pl.col("config") == config_name)
            if cfg_logs.height > 0:
                training_log.extend(cfg_logs.to_dicts())
        else:
            for entry in acc.get("training_log", []):
                training_log.append({"config": config_name, **entry})

        print(
            f"  {config_name}: best_epoch={best_cp}, IC={best_ic_val:+.4f} ({acc['elapsed_s']:.1f}s)"
        )

        # Incremental registration: persist this config as soon as aggregation
        # completes. If the notebook is interrupted here, completed configs are
        # already in the registry. (Fold-major training means ALL configs reach
        # this point together, but registration is still moved out of the old
        # batched block at the end.)
        if register and case_study and epoch_scores:
            cfg_best_preds = None
            for frame in complete_prediction_frames:
                if (
                    frame.height > 0
                    and "config" in frame.columns
                    and frame["config"].unique().to_list() == [config_name]
                ):
                    # Filter to the best epoch for this config
                    bep_df = frame.filter(pl.col("epoch") == best_cp)
                    if bep_df.height > 0:
                        cfg_best_preds = (
                            bep_df
                            if cfg_best_preds is None
                            else pl.concat([cfg_best_preds, bep_df], how="diagonal_relaxed")
                        )
            if cfg_best_preds is not None and cfg_best_preds.height > 0:
                try:
                    arch = _resolve_arch_name(config_name)
                    cfg_curves_df = pl.DataFrame(
                        [c for c in all_curves if c["config"] == config_name]
                    )
                    _register_dl_config(
                        case_study=case_study,
                        label=label_col,
                        config_name=config_name,
                        architecture=arch,
                        n_epochs=cfg.get("n_epochs"),
                        best_epoch=best_cp,
                        lookback=lookback,
                        n_folds=len(splits),
                        ic_mean=best_ic_val,
                        predictions=cfg_best_preds,
                        notebook=notebook,
                        learning_curves=cfg_curves_df if cfg_curves_df.height > 0 else None,
                        started_at=acc.get("started_at"),
                        elapsed_s=acc.get("elapsed_s"),
                        prediction_split=prediction_split,
                    )
                    print(f"    registered {config_name} incrementally")
                except Exception as exc:
                    print(f"    WARN: incremental registration failed for {config_name}: {exc}")

    del config_acc
    gc.collect()

    config_results.sort(
        key=lambda r: r["best_ic"] if not np.isnan(r["best_ic"]) else -999, reverse=True
    )
    best_result = config_results[0]
    best_config_name = best_result["config_name"]
    best_epoch = best_result["best_epoch"]
    best_ic = best_result["best_ic"]

    print(f"\n  Best: {best_config_name} @ epoch {best_epoch} (IC={best_ic:+.4f})")

    complete_predictions = (
        pl.concat(complete_prediction_frames, how="diagonal_relaxed")
        if complete_prediction_frames
        else pl.DataFrame()
    )

    # Extract best-config predictions at best epoch
    if complete_predictions.height > 0:
        best_preds_df = complete_predictions.filter(
            (pl.col("config") == best_config_name) & (pl.col("epoch") == best_epoch)
        )
        predictions = best_preds_df.with_columns(
            pl.lit(best_config_name).alias("model_id"),
        ).drop("config", "epoch")
    else:
        predictions = pl.DataFrame()

    learning_curves = pl.DataFrame(all_curves) if all_curves else pl.DataFrame()
    training_log_df = pl.DataFrame(training_log) if training_log else pl.DataFrame()

    # Save final outputs
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if predictions.height > 0:
            _save_parquet(save_dir / "predictions.parquet", predictions)
        if complete_predictions.height > 0:
            _save_parquet(save_dir / "all_predictions.parquet", complete_predictions)
        if learning_curves.height > 0:
            _save_parquet(save_dir / "learning_curves.parquet", learning_curves)
        if training_log_df.height > 0:
            _save_parquet(save_dir / "training_log.parquet", training_log_df)
        print(f"  Saved to {save_dir}")

    # Note: per-config registration happens incrementally inside the per-config
    # aggregation loop above (right after each config's best_epoch is computed).
    # The old batched registration block was removed to avoid duplicate writes.

    return {
        "grid_results": config_results,
        "best_config_name": best_config_name,
        "best_epoch": best_epoch,
        "best_ic": best_ic,
        "predictions": predictions,
        "all_predictions": complete_predictions,
        "fold_metrics": compute_fold_metrics_from_predictions(
            complete_predictions,
            best_config_name,
            best_epoch,
            date_col=date_col,
        ),
        "all_learning_curves": learning_curves,
        "training_log": training_log_df,
    }
