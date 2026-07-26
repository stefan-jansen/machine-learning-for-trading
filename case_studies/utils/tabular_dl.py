"""Shared tabular deep learning pipeline for Ch12 case study templates.

Provides:
- TabMModel: Rank-1 adapter MLP ensemble (Gorishniy et al., ICLR 2025)
- run_tabm_cv(): Walk-forward CV with epoch-checkpoint IC evaluation

Usage:
    from case_studies.utils.tabular_dl import run_tabm_cv
    from utils.modeling import load_configs

    tabdl_configs = load_configs("etfs", "fwd_ret_21d", "tabular_dl")
    result = run_tabm_cv(dataset_pd, splits, configs=tabdl_configs,
                         n_features=44, feature_names=..., label_col=...)
"""

from __future__ import annotations

import gc
import os
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from case_studies.utils.registry import clear_prediction_sets, compute_fold_metrics_from_predictions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from utils.modeling import RANDOM_SEED, seed_everything


def resolve_torch_device(device: str) -> torch.device:
    """Resolve an explicit Torch device without silently changing execution."""
    normalized = device.lower()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported Torch device: {device!r}")


def tabm_runtime_spec(
    device: str,
    *,
    seed: int = RANDOM_SEED,
    num_threads: int = 8,
) -> dict[str, Any]:
    """Return the execution settings that define a reproducible TabM run."""
    if num_threads < 1:
        raise ValueError("num_threads must be at least 1")
    resolved = resolve_torch_device(device)
    return {
        "device": resolved.type,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "num_threads": num_threads,
        "seed": seed,
    }


def _tabm_checkpoint_epochs(config: dict[str, Any]) -> tuple[int, ...]:
    """Return the exact checkpoint surface implied by one effective config."""
    if str(config["config_name"]).startswith("tabpfn"):
        return (1,)
    n_epochs = int(config.get("n_epochs", 200))
    checkpoint_interval = int(config.get("checkpoint_interval", 25))
    if n_epochs < 1 or checkpoint_interval < 1:
        raise ValueError("n_epochs and checkpoint_interval must be positive")
    checkpoints = list(range(checkpoint_interval, n_epochs + 1, checkpoint_interval))
    if not checkpoints or checkpoints[-1] != n_epochs:
        checkpoints.append(n_epochs)
    return tuple(checkpoints)


def _build_tabm_training_spec(
    config: dict[str, Any],
    *,
    label_col: str,
    n_folds: int,
    feature_names: list[str],
    eval_label_col: str | None,
    task_type: str,
    class_values: list | None,
    runtime_spec: dict[str, Any],
    seed: int,
    splits: list[dict[str, Any]] | None = None,
    input_data_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the single identity used for TabM lookup and registration."""
    params = dict(config.get("params", {}))
    params.update(
        {
            "batch_size": int(config.get("batch_size", 4096)),
            "class_values": list(class_values) if class_values is not None else None,
            "eval_label_col": eval_label_col,
            "feature_names": list(feature_names),
            "runtime": dict(runtime_spec),
            "splits": [
                {
                    key: str(split[key]) if key != "fold" else int(split[key])
                    for key in ("fold", "train_start", "train_end", "val_start", "val_end")
                }
                for split in (splits or [])
            ],
            "task_type": task_type,
        }
    )
    if input_data_spec is not None:
        params["input_data_spec"] = input_data_spec
    return {
        "checkpoint_interval": int(config.get("checkpoint_interval", 25)),
        "config_name": config["config_name"],
        "family": config.get("family", "tabular_dl"),
        "feature_sets": ["financial", "model_based"],
        "label": label_col,
        "library": config.get("library", "tabm"),
        "n_epochs": int(config.get("n_epochs", 200)),
        "n_folds": n_folds,
        "params": params,
        "seed": seed,
    }


def _configure_torch_runtime(runtime_spec: dict[str, Any]) -> torch.device:
    """Apply the strict deterministic settings recorded in a training spec."""
    torch.set_num_threads(int(runtime_spec["num_threads"]))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(int(runtime_spec["seed"]))
    return resolve_torch_device(str(runtime_spec["device"]))


# ---------------------------------------------------------------------------
# TabM Model
# ---------------------------------------------------------------------------


class TabMModel(nn.Module):
    """Rank-1 adapter MLP ensemble for tabular data.

    Shared backbone + M rank-1 scaling vectors = efficient deep ensemble.
    From Gorishniy et al. (ICLR 2025).
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_members: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_members = n_members

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Per-member rank-1 adapters (scaling vectors for last hidden layer)
        self.adapters = nn.Parameter(torch.randn(n_members, hidden_dim) * 0.1)

        # Per-member output heads
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_members)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)  # (batch, hidden)
        outputs = []
        for i in range(self.n_members):
            h_adapted = h * self.adapters[i].unsqueeze(0)  # rank-1 scaling
            outputs.append(self.heads[i](h_adapted))
        return torch.stack(outputs, dim=0).mean(dim=0).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Prediction Helpers
# ---------------------------------------------------------------------------


def _predict_in_chunks(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    chunk_size: int = 32768,
) -> np.ndarray:
    """Predict on large arrays in chunks to avoid GPU OOM."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), chunk_size):
            batch = torch.FloatTensor(X[start : start + chunk_size]).to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def _run_tabpfn_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    max_samples: int,
    n_ensemble: int,
) -> np.ndarray:
    """Run TabPFN on a single fold. Returns predictions or raises."""
    from tabpfn import TabPFNRegressor

    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    model = TabPFNRegressor(n_estimators=n_ensemble)
    model.fit(X_train, y_train)
    return model.predict(X_val)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def _train_tabm_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_eval_val: np.ndarray,
    val_dates: np.ndarray,
    val_entities: np.ndarray | None,
    n_epochs: int,
    batch_size: int,
    checkpoint_interval: int,
    device: torch.device,
) -> tuple[dict[int, float], dict[int, np.ndarray], dict[int, float]]:
    """Train TabM on one fold, storing predictions at ALL checkpoints.

    Trains to completion (no early stopping). Stores predictions at every
    checkpoint so the caller can select the best epoch after all folds finish.

    Returns (checkpoint_ics, checkpoint_predictions, epoch_losses).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_train = len(X_train)

    checkpoint_ics: dict[int, float] = {}
    checkpoint_preds: dict[int, np.ndarray] = {}
    epoch_losses: dict[int, float] = {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = np.random.permutation(n_train)
        for n_batches, start in enumerate(range(0, n_train, batch_size), 1):
            batch_idx = indices[start : start + batch_size]
            X_batch = torch.FloatTensor(X_train[batch_idx]).to(device)
            y_batch = torch.FloatTensor(y_train[batch_idx]).to(device)

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
            val_preds = _predict_in_chunks(model, X_val, device)
            ic_frame = pl.DataFrame(
                {
                    "timestamp": val_dates,
                    "symbol": val_entities,
                    "y_true": y_eval_val,
                    "y_pred": val_preds,
                }
            )
            ic = cross_sectional_ic(
                ic_frame,
                ic_frame,
                pred_col="y_pred",
                ret_col="y_true",
                date_col="timestamp",
                entity_col="symbol",
                min_obs=5,
            )["ic_mean"]
            checkpoint_ics[epoch] = ic
            checkpoint_preds[epoch] = val_preds.copy()
            print(
                f"      epoch {epoch:3d}/{n_epochs}: loss={avg_loss:.6f}, IC={ic:+.4f}",
                flush=True,
            )

    return checkpoint_ics, checkpoint_preds, epoch_losses


# ---------------------------------------------------------------------------
# Incremental Save/Load Helpers
# ---------------------------------------------------------------------------


from case_studies.utils.registry.store import flush_fold_predictions


def _decision_time_checkpoint_metrics(
    frame: pl.DataFrame,
    *,
    date_col: str,
    entity_col: str,
    pred_col: str = "y_score",
    ret_col: str = "y_true",
) -> dict[str, float | int]:
    """Score one pooled checkpoint with equal weight per decision timestamp.

    The counterpart of ``deep_learning._decision_time_checkpoint_metrics``, and
    the same contract: a decision time with 200 names counts once, exactly as a
    decision time with 20 does. Pooling the rows instead would weight the wide
    days, and averaging per-fold ICs would weight the folds.
    """
    stats = cross_sectional_ic(
        frame,
        frame,
        pred_col=pred_col,
        ret_col=ret_col,
        date_col=date_col,
        entity_col=entity_col,
        method="spearman",
        min_obs=5,
    )
    return {
        "ic_mean": float(stats["ic_mean"]),
        "ic_std": float(stats["ic_std"]),
        "ic_n_days": int(stats["n_periods"]),
    }


def _load_incremental_preds_for_config(incr_dir: Path, config_name: str) -> pl.DataFrame:
    """Reassemble one config's predictions from its per-fold incremental saves."""
    parquet_files = sorted(incr_dir.glob(f"{config_name}_fold*.parquet"))
    if not parquet_files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in parquet_files])


def _load_cached_tabm_config(
    *,
    case_study: str,
    training_spec: dict[str, Any],
    config_name: str,
    prediction_split: str,
    date_col: str,
    entity_col: str,
    eval_col: str | None,
    expected_checkpoints: tuple[int, ...],
    expected_keys: pl.DataFrame,
) -> tuple[dict[str, Any], pl.DataFrame, list[dict[str, Any]]]:
    """Reconstruct one completed config from content-addressed registry artifacts."""
    from case_studies.utils.registry import (
        load_prediction_metrics,
        load_prediction_sets,
        prediction_dir,
        training_hash_from_spec,
    )

    training_hash = training_hash_from_spec(training_spec)
    prediction_sets = load_prediction_sets(
        case_study,
        training_hash=training_hash,
        split=prediction_split,
    )
    required_metadata = {"prediction_hash", "checkpoint_value", "checkpoint_kind"}
    missing_metadata = required_metadata - set(prediction_sets.columns)
    if missing_metadata:
        raise ValueError(
            f"Cached {config_name} checkpoint metadata is missing {sorted(missing_metadata)}"
        )
    if prediction_sets.height != len(expected_checkpoints):
        raise ValueError(
            f"Cached {config_name} checkpoints row count {prediction_sets.height} does not match "
            f"expected {len(expected_checkpoints)}"
        )
    if prediction_sets["checkpoint_value"].null_count():
        raise ValueError(f"Cached {config_name} has a null checkpoint value")
    if prediction_sets.filter(pl.col("checkpoint_kind") != "epoch").height:
        raise ValueError(f"Cached {config_name} contains a non-epoch checkpoint")
    checkpoint_values = prediction_sets["checkpoint_value"].to_list()
    observed_checkpoints = tuple(sorted(int(value) for value in checkpoint_values))
    if len(observed_checkpoints) != len(set(observed_checkpoints)):
        raise ValueError(f"Cached {config_name} has duplicate checkpoints")
    if observed_checkpoints != expected_checkpoints:
        raise ValueError(
            f"Cached {config_name} checkpoints {observed_checkpoints} do not match "
            f"expected {expected_checkpoints}"
        )

    key_cols = [date_col, entity_col, "fold_id"]
    expected_sorted = expected_keys.select(key_cols).sort(key_cols)
    frames: list[pl.DataFrame] = []
    curves: list[dict[str, Any]] = []
    for row in prediction_sets.iter_rows(named=True):
        epoch = row["checkpoint_value"]
        if epoch is None:
            continue
        path = prediction_dir(case_study, row["prediction_hash"]) / "predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        predictions = pl.read_parquet(path)
        required_cols = {date_col, entity_col, "fold_id", "y_true", "y_score"}
        if eval_col:
            required_cols.add(eval_col)
        missing_cols = required_cols - set(predictions.columns)
        if missing_cols:
            raise ValueError(
                f"Cached {config_name} checkpoint {epoch} schema is missing {sorted(missing_cols)}"
            )
        if predictions.select(pl.col(list(required_cols)).null_count()).row(0) != (0,) * len(
            required_cols
        ):
            raise ValueError(f"Cached {config_name} checkpoint {epoch} schema contains nulls")
        actual_keys = predictions.select(key_cols)
        if actual_keys.n_unique() != predictions.height:
            raise ValueError(f"Cached {config_name} checkpoint {epoch} has duplicate keys")
        if not actual_keys.sort(key_cols).equals(expected_sorted):
            raise ValueError(
                f"Cached {config_name} checkpoint {epoch} key or fold coverage is incomplete"
            )
        actual_col = eval_col or "y_true"
        metric = cross_sectional_ic(
            predictions,
            predictions,
            pred_col="y_score",
            ret_col=actual_col,
            date_col=date_col,
            entity_col=entity_col,
            method="spearman",
            min_obs=5,
        )
        registry_metrics = load_prediction_metrics(
            case_study, prediction_hash=row["prediction_hash"]
        )
        required_daily_metrics = {"ic_mean_daily", "ic_std_daily"}
        missing_daily_metrics = required_daily_metrics - set(registry_metrics.columns)
        if registry_metrics.height != 1 or missing_daily_metrics:
            raise ValueError(
                f"Cached {config_name} checkpoint {epoch} has invalid daily registry metrics"
            )
        comparisons = {
            "daily mean": (
                registry_metrics["ic_mean_daily"][0],
                float(metric["ic_mean"]),
            ),
            "daily std": (
                registry_metrics["ic_std_daily"][0],
                float(metric.get("ic_std", 0.0)),
            ),
        }
        mismatches = {
            name: values
            for name, values in comparisons.items()
            if values[0] is None
            or not np.isclose(float(values[0]), values[1], atol=1e-12, rtol=0.0)
        }
        if mismatches:
            raise ValueError(
                f"Cached {config_name} checkpoint {epoch} daily metric mismatch: {mismatches}"
            )
        curves.append(
            {
                "config": config_name,
                "epoch": int(epoch),
                "ic_mean": float(metric["ic_mean"]),
                "ic_std": float(metric.get("ic_std", 0.0)),
            }
        )
        frames.append(
            predictions.with_columns(
                pl.lit(config_name).alias("config"),
                pl.lit(int(epoch), dtype=pl.Int32).alias("epoch"),
            )
        )
    if not curves:
        raise ValueError(f"No cached {prediction_split} checkpoints for {config_name}")
    best = max(curves, key=lambda row: row["ic_mean"])
    result = {
        "config_name": config_name,
        "best_epoch": best["epoch"],
        "best_ic": best["ic_mean"],
        "elapsed_s": 0.0,
        "started_at": None,
        "cached": True,
    }
    return result, pl.concat(frames), curves


def _assemble_tabm_results(
    *,
    config_results: list[dict[str, Any]],
    all_predictions: pl.DataFrame,
    curve_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    save_dir: Path | None,
    date_col: str,
    entity_col: str,
    eval_col: str | None,
) -> dict[str, Any]:
    """Select the winner and build the same result for trained or cached configs."""
    if not config_results:
        raise ValueError("No configs completed successfully.")
    ranked = sorted(
        config_results,
        key=lambda row: row["best_ic"] if not np.isnan(row["best_ic"]) else -999,
        reverse=True,
    )
    best = ranked[0]
    best_name = best["config_name"]
    best_epoch = best["best_epoch"]
    best_ic = best["best_ic"]
    print(f"\n  Best: {best_name} @ epoch {best_epoch} (IC={best_ic:+.4f})")

    best_predictions = all_predictions.filter(
        (pl.col("config") == best_name) & (pl.col("epoch") == best_epoch)
    )
    if best_predictions.height:
        best_predictions = best_predictions.with_columns(pl.lit(best_name).alias("model_id")).drop(
            "config", "epoch"
        )
    curves = pl.DataFrame(curve_rows) if curve_rows else pl.DataFrame()
    training_log = pl.DataFrame(training_rows) if training_rows else pl.DataFrame()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if best_predictions.height:
            best_predictions.write_parquet(save_dir / "predictions.parquet")
        if all_predictions.height:
            all_predictions.write_parquet(save_dir / "all_predictions.parquet")
        if curves.height:
            curves.write_parquet(save_dir / "learning_curves.parquet")
        if training_log.height:
            training_log.write_parquet(save_dir / "training_log.parquet")
        print(f"  Saved TabM artifacts for {save_dir.name}")

    return {
        "grid_results": ranked,
        "best_config_name": best_name,
        "best_epoch": best_epoch,
        "best_ic": best_ic,
        "predictions": best_predictions,
        "all_predictions": all_predictions,
        "fold_metrics": compute_fold_metrics_from_predictions(
            all_predictions,
            best_name,
            best_epoch,
            date_col=date_col,
            entity_col=entity_col,
            eval_col=eval_col,
        ),
        "all_learning_curves": curves,
        "training_log": training_log,
    }


# ---------------------------------------------------------------------------
# Registry Integration
# ---------------------------------------------------------------------------


def _register_tabm_config(
    *,
    case_study: str,
    label: str,
    config_name: str,
    n_epochs: int | None,
    best_epoch: int,
    n_folds: int,
    ic_mean: float,
    predictions,
    notebook: str | None = None,
    learning_curves=None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
    prediction_split: str = "validation",
    checkpoint_interval: int | None = None,
    runtime_spec: dict[str, Any] | None = None,
    task_type: str = "regression",
    class_values: list | None = None,
    eval_col: str | None = None,
    training_spec: dict[str, Any] | None = None,
) -> str:
    """Register a single tabm config — thin delegate to register_epoch_checkpoint."""
    from case_studies.utils.registry import register_epoch_checkpoint

    return register_epoch_checkpoint(
        case_study,
        family="tabular_dl",
        library="tabm",
        config_name=config_name,
        label=label,
        n_folds=n_folds,
        n_epochs=n_epochs,
        best_epoch=best_epoch,
        ic_mean=ic_mean,
        predictions=predictions,
        learning_curves=learning_curves,
        entry_point=notebook,
        started_at=started_at,
        elapsed_s=elapsed_s,
        prediction_split=prediction_split,
        checkpoint_interval=checkpoint_interval,
        spec_extra_params={"runtime": runtime_spec} if runtime_spec else None,
        task_type=task_type,
        class_values=class_values,
        eval_col=eval_col,
        training_spec=training_spec,
    )


# ---------------------------------------------------------------------------
# Main CV Pipeline
# ---------------------------------------------------------------------------


def run_tabm_cv(
    dataset_pd: pd.DataFrame,
    splits: list[dict[str, Any]],
    *,
    configs: list[dict[str, Any]],
    n_features: int,
    feature_names: list[str],
    label_col: str,
    eval_label_col: str | None = None,
    task_type: str = "regression",
    class_values: list | None = None,
    date_col: str,
    entity_col: str = "symbol",
    device: str = "cuda",
    save_dir: Path | None = None,
    register: bool = False,
    case_study: str | None = None,
    notebook: str | None = None,
    temporal_by_fold=None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    force_retrain: bool = False,
    prediction_split: str = "validation",
    seed: int = RANDOM_SEED,
    num_threads: int = 8,
    input_data_spec: dict[str, Any] | None = None,
    identity_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Walk-forward tabular DL CV with epoch-checkpoint IC evaluation.

    All training parameters (n_epochs, batch_size, checkpoint_interval) are
    read from each config dict. Config dicts come from ``load_configs()``.

    Parameters
    ----------
    dataset_pd : pandas DataFrame
        Full dataset with features, label, date, and entity columns.
    splits : list[dict]
        Walk-forward splits from generate_cv_splits().
    configs : list[dict]
        Config dicts from ``load_configs()``. Each must have ``config_name``,
        ``params`` (with TabM arch kwargs or TabPFN kwargs), and training
        params: ``n_epochs``, ``batch_size``, ``checkpoint_interval``.
    n_features : int
        Number of input features (injected into TabM constructor).
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
        Directory to save predictions and metrics.

    Returns
    -------
    dict with keys:
        grid_results: list[dict] — per-config results ranked by best IC
        best_config_name: str
        best_epoch: int
        best_ic: float
        predictions: pl.DataFrame — OOS predictions from best config
        all_predictions: pl.DataFrame — predictions for ALL configs
        fold_metrics: pl.DataFrame — per-fold cross-sectional IC for best config
        all_learning_curves: pl.DataFrame — IC × epoch × config
    """
    if task_type not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    if task_type == "classification" and not eval_label_col:
        raise ValueError("classification requires eval_label_col for continuous-return IC")
    if task_type == "classification" and not class_values:
        raise ValueError("classification requires class_values")
    if eval_label_col and eval_label_col not in dataset_pd.columns:
        raise ValueError(f"eval_label_col {eval_label_col!r} is absent from the dataset")
    if n_features != len(feature_names):
        raise ValueError(
            f"n_features={n_features} does not match {len(feature_names)} feature names"
        )
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names contains duplicates")
    if register and save_dir is None:
        raise ValueError(
            "register=True requires save_dir for incremental prediction saves. "
            "Pass save_dir=CASE_DIR / 'run_log' / 'training' / 'tabular_dl'"
        )
    if input_data_spec is not None and identity_params is not None:
        raise ValueError("Pass either input_data_spec or legacy identity_params, not both")

    runtime_spec = tabm_runtime_spec(device, seed=seed, num_threads=num_threads)
    torch_device = _configure_torch_runtime(runtime_spec)
    eval_col = "eval_actual" if eval_label_col else None

    dataset_pd = dataset_pd.sort_values([date_col, entity_col], kind="mergesort").reset_index(
        drop=True
    )
    expected_key_frames = []
    for split in splits:
        val_mask = (dataset_pd[date_col] >= split["val_start"]) & (
            dataset_pd[date_col] <= split["val_end"]
        )
        val_rows = dataset_pd.loc[val_mask]
        valid = val_rows[label_col].notna()
        if eval_label_col:
            valid &= val_rows[eval_label_col].notna()
        keys = pl.from_pandas(val_rows.loc[valid, [date_col, entity_col]])
        expected_key_frames.append(
            keys.with_columns(pl.lit(int(split["fold"]), dtype=pl.Int32).alias("fold_id"))
        )
    expected_keys = (
        pl.concat(expected_key_frames)
        if expected_key_frames
        else pl.DataFrame(
            schema={date_col: pl.Datetime, entity_col: pl.String, "fold_id": pl.Int32}
        )
    )
    if expected_keys.n_unique(subset=[date_col, entity_col, "fold_id"]) != expected_keys.height:
        raise ValueError("validation data contains duplicate timestamp/entity/fold keys")
    if identity_params is not None:
        from case_studies.utils.registry import build_training_spec

        training_specs = {
            cfg["config_name"]: build_training_spec(
                cfg.get("family", "tabular_dl"),
                cfg["config_name"],
                label_col,
                n_folds=len(splits),
                n_epochs=cfg.get("n_epochs"),
                extra_params=identity_params,
            )
            for cfg in configs
        }
    else:
        training_specs = {
            cfg["config_name"]: _build_tabm_training_spec(
                cfg,
                label_col=label_col,
                n_folds=len(splits),
                feature_names=feature_names,
                eval_label_col=eval_label_col,
                task_type=task_type,
                class_values=class_values,
                runtime_spec=runtime_spec,
                seed=seed,
                splits=splits,
                input_data_spec=input_data_spec,
            )
            for cfg in configs
        }
    cached_results: list[dict[str, Any]] = []
    cached_prediction_frames: list[pl.DataFrame] = []
    cached_curves: list[dict[str, Any]] = []

    # Filter out configs whose training_hash is already complete (unless
    # force_retrain). This prevents re-running finished work across the entire
    # sweep — the caller can override with force_retrain=True for debugging.
    if register and case_study and not force_retrain:
        from case_studies.utils.registry import (
            load_prediction_sets,
            training_hash_from_spec,
            training_run_status,
        )

        pending_configs = []
        for cfg in configs:
            try:
                spec = training_specs[cfg["config_name"]]
                status = training_run_status(case_study, spec)
                split_rows = load_prediction_sets(
                    case_study,
                    training_hash=training_hash_from_spec(spec),
                    split=prediction_split,
                )
                split_complete = not split_rows.is_empty()
                if status.complete and split_complete:
                    cached_result, cached_predictions, cached_curve_rows = _load_cached_tabm_config(
                        case_study=case_study,
                        training_spec=spec,
                        config_name=cfg["config_name"],
                        prediction_split=prediction_split,
                        date_col=date_col,
                        entity_col=entity_col,
                        eval_col=eval_col,
                        expected_checkpoints=_tabm_checkpoint_epochs(cfg),
                        expected_keys=expected_keys,
                    )
                    cached_results.append(cached_result)
                    cached_prediction_frames.append(cached_predictions)
                    cached_curves.extend(cached_curve_rows)
                    print(
                        f"  REUSE {cfg['config_name']:24s}  "
                        f"({status.summary()}, split={prediction_split})"
                    )
                    continue
                if status.complete and not split_complete:
                    print(
                        f"  RETRAIN {cfg['config_name']:25s}  missing {prediction_split} predictions"
                    )
                elif status.partial:
                    print(f"  RETRAIN {cfg['config_name']:25s}  partial state: {status.summary()}")
            except Exception as exc:
                print(f"  RETRAIN {cfg['config_name']:25s}  invalid cache: {exc}")
            pending_configs.append(cfg)

        if not pending_configs:
            print("All configs complete; replaying content-addressed predictions.")
            return _assemble_tabm_results(
                config_results=cached_results,
                all_predictions=pl.concat(cached_prediction_frames),
                curve_rows=cached_curves,
                training_rows=[],
                save_dir=save_dir / label_col if save_dir is not None else None,
                date_col=date_col,
                entity_col=entity_col,
                eval_col=eval_col,
            )
        configs = pending_configs

    dates_series = dataset_pd[date_col]

    # Pre-build per-fold data: mask dates → extract numpy → impute + scale
    has_fold_temporal = temporal_by_fold is not None and temporal_keys and temporal_feature_names
    print("Preparing fold data...")
    fold_data = []
    for split in splits:
        train_mask = (dates_series >= split["train_start"]) & (dates_series <= split["train_end"])
        val_mask = (dates_series >= split["val_start"]) & (dates_series <= split["val_end"])

        if has_fold_temporal:
            from utils.modeling import _replace_temporal_columns

            train_df = _replace_temporal_columns(
                dataset_pd,
                train_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                split["fold"],
            )
            val_df = _replace_temporal_columns(
                dataset_pd,
                val_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                split["fold"],
            )
        else:
            train_df = dataset_pd.loc[train_mask]
            val_df = dataset_pd.loc[val_mask]

        # Drop rows without a fit target or declared evaluation target.
        train_valid = train_df[label_col].notna()
        val_valid = val_df[label_col].notna()
        if eval_label_col:
            train_valid &= train_df[eval_label_col].notna()
            val_valid &= val_df[eval_label_col].notna()
        train_df = train_df.loc[train_valid]
        val_df = val_df.loc[val_valid]

        if len(train_df) < 100 or len(val_df) < 50:
            raise ValueError(
                f"Fold {split['fold']} is too small: train={len(train_df)}, val={len(val_df)}"
            )

        X_train = train_df[feature_names].values.astype(np.float32)
        y_train = train_df[label_col].values.astype(np.float32)
        X_val = val_df[feature_names].values.astype(np.float32)
        y_val = val_df[label_col].values.astype(np.float32)
        y_eval_val = (
            val_df[eval_label_col].values.astype(np.float32) if eval_label_col else y_val.copy()
        )
        val_dates = val_df[date_col].values
        val_entities = val_df[entity_col].values

        # Impute + scale per fold
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val = scaler.transform(imputer.transform(X_val))

        fold_data.append(
            {
                "fold": split["fold"],
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "y_eval_val": y_eval_val,
                "val_dates": val_dates,
                "val_entities": val_entities,
                "n_train": len(X_train),
                "n_val": len(X_val),
            }
        )
        print(f"  Fold {split['fold']}: train={len(X_train):,}  val={len(X_val):,}")

    if not fold_data:
        raise ValueError("No valid folds created. Check data size.")

    # Grid search — train each config, evaluate at checkpoints, store ALL predictions.
    # Incremental save: flush predictions to disk after each fold × config.
    config_results: list[dict[str, Any]] = list(cached_results)
    all_curves: list[dict] = list(cached_curves)
    training_log: list[dict] = []

    # Set up incremental save directory
    run_save_dir = save_dir / label_col if save_dir is not None else None
    incr_dir = run_save_dir / "_incremental" if run_save_dir is not None else None
    if incr_dir is not None:
        incr_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        config_name = cfg["config_name"]
        if register and case_study and force_retrain:
            from case_studies.utils.registry import build_training_spec, training_hash_from_spec

            spec = build_training_spec(
                cfg["family"],
                config_name,
                label_col,
                n_folds=len(splits),
                n_epochs=cfg.get("n_epochs"),
            )
            removed = clear_prediction_sets(
                case_study,
                training_hash_from_spec(spec),
                split=prediction_split,
            )
            if removed["prediction_sets"]:
                print(
                    f"  cleared {removed['prediction_sets']} prior {prediction_split} "
                    f"checkpoint(s) for {config_name}"
                )
        cfg_params = dict(cfg.get("params", {}))
        cfg_n_epochs = cfg.get("n_epochs", 200)
        cfg_batch_size = cfg.get("batch_size", 4096)
        cfg_checkpoint = cfg.get("checkpoint_interval", 25)
        is_tabpfn = config_name.startswith("tabpfn")

        config_started_at = datetime.now(UTC).isoformat()
        t0 = time.perf_counter()
        print(f"\n  {config_name}:")

        fold_checkpoint_ics: dict[int, list[float]] = {}
        tabpfn_available = True

        for fd in fold_data:
            fold_t0 = time.perf_counter()
            seed_everything(seed + fd["fold"])

            if is_tabpfn:
                try:
                    preds = _run_tabpfn_fold(
                        fd["X_train"],
                        fd["y_train"],
                        fd["X_val"],
                        max_samples=cfg_params.get("max_samples", 2000),
                        n_ensemble=cfg_params.get("n_ensemble", 4),
                    )
                    ic_frame = pl.DataFrame(
                        {
                            "timestamp": fd["val_dates"],
                            "symbol": fd["val_entities"],
                            "y_true": fd["y_eval_val"],
                            "y_pred": preds,
                        }
                    )
                    ic = cross_sectional_ic(
                        ic_frame,
                        ic_frame,
                        pred_col="y_pred",
                        ret_col="y_true",
                        date_col="timestamp",
                        entity_col="symbol",
                        min_obs=5,
                    )["ic_mean"]
                    fold_checkpoint_ics.setdefault(1, []).append(ic)

                    # Incremental save: flush this fold's predictions to disk
                    if incr_dir is not None:
                        flush_fold_predictions(
                            incr_dir,
                            config_name,
                            fd["fold"],
                            {1: preds},
                            fd["val_dates"],
                            fd["val_entities"],
                            fd["y_val"],
                            date_col,
                            entity_col,
                            eval_actual=fd["y_eval_val"] if eval_col else None,
                            eval_col=eval_col or "eval_actual",
                        )

                    fold_elapsed = time.perf_counter() - fold_t0
                    training_log.append(
                        {
                            "config": config_name,
                            "fold": fd["fold"],
                            "elapsed_s": round(fold_elapsed, 1),
                            "n_train": fd["n_train"],
                            "n_val": fd["n_val"],
                            "best_ic": round(ic, 4),
                            "n_checkpoints": 1,
                        }
                    )
                    print(f"    Fold {fd['fold']}: IC={ic:+.4f} ({fold_elapsed:.1f}s)")
                except ImportError:
                    if fd == fold_data[0]:
                        print("    TabPFN not installed — skipping")
                    tabpfn_available = False
                    break
                except (RuntimeError, ValueError) as e:
                    if fd == fold_data[0]:
                        print(f"    TabPFN failed: {e}")
                    tabpfn_available = False
                    break
            else:
                # TabM: train to completion, store ALL checkpoint predictions
                tabm_kwargs = {"n_features": n_features, **cfg_params}
                model = TabMModel(**tabm_kwargs)
                checkpoint_ics, checkpoint_preds, epoch_losses = _train_tabm_fold(
                    model=model,
                    X_train=fd["X_train"],
                    y_train=fd["y_train"],
                    X_val=fd["X_val"],
                    y_val=fd["y_val"],
                    y_eval_val=fd["y_eval_val"],
                    val_dates=fd["val_dates"],
                    val_entities=fd["val_entities"],
                    n_epochs=cfg_n_epochs,
                    batch_size=cfg_batch_size,
                    checkpoint_interval=cfg_checkpoint,
                    device=torch_device,
                )

                for ep, ic in checkpoint_ics.items():
                    fold_checkpoint_ics.setdefault(ep, []).append(ic)

                # Incremental save: flush ALL checkpoint predictions for this fold
                if incr_dir is not None:
                    flush_fold_predictions(
                        incr_dir,
                        config_name,
                        fd["fold"],
                        checkpoint_preds,
                        fd["val_dates"],
                        fd["val_entities"],
                        fd["y_val"],
                        date_col,
                        entity_col,
                        eval_actual=fd["y_eval_val"] if eval_col else None,
                        eval_col=eval_col or "eval_actual",
                    )

                del model, checkpoint_preds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                best_ep = max(checkpoint_ics, key=lambda e: checkpoint_ics[e])
                fold_elapsed = time.perf_counter() - fold_t0
                # Sample losses at checkpoint epochs for the log
                loss_at_checkpoints = {
                    str(k): round(epoch_losses.get(k, 0.0), 6)
                    for k in sorted(checkpoint_ics.keys())
                }
                training_log.append(
                    {
                        "config": config_name,
                        "fold": fd["fold"],
                        "elapsed_s": round(fold_elapsed, 1),
                        "n_train": fd["n_train"],
                        "n_val": fd["n_val"],
                        "best_ic": round(checkpoint_ics[best_ep], 4),
                        "n_checkpoints": len(checkpoint_ics),
                        "checkpoint_ics": {str(k): round(v, 4) for k, v in checkpoint_ics.items()},
                        "checkpoint_losses": loss_at_checkpoints,
                    }
                )
                print(
                    f"    Fold {fd['fold']}: best_ep={best_ep}, "
                    f"IC={checkpoint_ics[best_ep]:+.4f} ({fold_elapsed:.1f}s)"
                )

        if is_tabpfn and not tabpfn_available:
            continue

        cfg_all_preds = (
            _load_incremental_preds_for_config(incr_dir, config_name)
            if incr_dir is not None
            else pl.DataFrame()
        )
        checkpoint_metrics: dict[int, dict[str, float]] = {}
        if cfg_all_preds.height:
            actual_col = eval_col if eval_col else "y_true"
            for epoch in sorted(cfg_all_preds["epoch"].unique().to_list()):
                epoch_predictions = cfg_all_preds.filter(pl.col("epoch") == epoch)
                checkpoint_metrics[int(epoch)] = _decision_time_checkpoint_metrics(
                    epoch_predictions,
                    date_col=date_col,
                    entity_col=entity_col,
                    ret_col=actual_col,
                )
        elif fold_checkpoint_ics:
            checkpoint_metrics = {
                int(epoch): {
                    "ic_mean": float(np.nanmean(values)),
                    "ic_std": float(np.nanstd(values)) if len(values) > 1 else 0.0,
                }
                for epoch, values in fold_checkpoint_ics.items()
            }

        if checkpoint_metrics:
            best_cp = max(
                checkpoint_metrics, key=lambda epoch: checkpoint_metrics[epoch]["ic_mean"]
            )
            best_ic_val = float(checkpoint_metrics[best_cp]["ic_mean"])
        else:
            best_cp = 0
            best_ic_val = float("nan")

        elapsed = time.perf_counter() - t0
        config_results.append(
            {
                "config_name": config_name,
                "best_epoch": best_cp,
                "best_ic": best_ic_val,
                "elapsed_s": elapsed,
                "started_at": config_started_at,
            }
        )

        cfg_curves_list = []
        for ep, metric in sorted(checkpoint_metrics.items()):
            entry = {
                "config": config_name,
                "epoch": ep,
                "ic_mean": float(metric["ic_mean"]),
                "ic_std": float(metric.get("ic_std", 0.0)),
            }
            all_curves.append(entry)
            cfg_curves_list.append(entry)

        print(f"    → best_epoch={best_cp}, IC={best_ic_val:+.4f} ({elapsed:.1f}s)")

        # Incremental registration: persist this config immediately so a later
        # interruption or re-run doesn't lose work. Safe because config-major
        # loop: this config's folds are all complete at this point.
        # Registers ONE prediction_set per epoch checkpoint (each parquet contains
        # exactly one epoch's predictions). The training_run is registered once on
        # the first epoch slice via _register_tabm_config; subsequent epochs go
        # through register_prediction_set directly so we don't re-register the
        # training_run each time.
        if register and case_study and incr_dir is not None:
            try:
                if cfg_all_preds.height > 0:
                    from case_studies.utils.registry import register_prediction_set

                    cfg_curves_df = pl.DataFrame(cfg_curves_list) if cfg_curves_list else None
                    epoch_ics = {
                        epoch: float(metric["ic_mean"])
                        for epoch, metric in checkpoint_metrics.items()
                    }
                    epochs = sorted(cfg_all_preds["epoch"].unique().to_list())

                    # First epoch registers the training_run + its prediction_set
                    first_ep = best_cp if best_cp in epochs else epochs[0]
                    first_slice = cfg_all_preds.filter(pl.col("epoch") == first_ep).drop(
                        "config", "epoch"
                    )
                    t_hash = _register_tabm_config(
                        case_study=case_study,
                        label=label_col,
                        config_name=config_name,
                        n_epochs=cfg.get("n_epochs"),
                        best_epoch=int(first_ep),
                        n_folds=len(fold_data),
                        ic_mean=epoch_ics.get(first_ep, best_ic_val),
                        predictions=first_slice,
                        notebook=notebook,
                        learning_curves=cfg_curves_df,
                        started_at=config_started_at,
                        elapsed_s=elapsed,
                        prediction_split=prediction_split,
                        checkpoint_interval=cfg.get("checkpoint_interval"),
                        runtime_spec=runtime_spec,
                        task_type=task_type,
                        class_values=class_values,
                        eval_col=eval_col,
                        training_spec=training_specs[config_name],
                    )

                    # Remaining epochs: just register prediction_sets
                    for ep in epochs:
                        if ep == first_ep:
                            continue
                        ep_slice = cfg_all_preds.filter(pl.col("epoch") == ep).drop(
                            "config", "epoch"
                        )
                        register_prediction_set(
                            case_study,
                            training_hash=t_hash,
                            checkpoint_value=int(ep),
                            checkpoint_kind="epoch",
                            split=prediction_split,
                            predictions=ep_slice,
                            metrics={"ic_mean": epoch_ics.get(ep, float("nan"))},
                            task_type=task_type,
                            class_values=class_values,
                            eval_col=eval_col,
                            label=label_col,
                        )
                    print(
                        f"    registered {config_name} incrementally ({len(epochs)} per-epoch slices)"
                    )
            except Exception as exc:
                print(f"    WARN: incremental registration failed for {config_name}: {exc}")

        gc.collect()

    prediction_frames = list(cached_prediction_frames)
    if incr_dir is not None:
        for config_name in [cfg["config_name"] for cfg in configs]:
            frame = _load_incremental_preds_for_config(incr_dir, config_name)
            if frame.height:
                prediction_frames.append(frame)
    all_predictions = pl.concat(prediction_frames) if prediction_frames else pl.DataFrame()
    return _assemble_tabm_results(
        config_results=config_results,
        all_predictions=all_predictions,
        curve_rows=all_curves,
        training_rows=training_log,
        save_dir=run_save_dir,
        date_col=date_col,
        entity_col=entity_col,
        eval_col=eval_col,
    )
