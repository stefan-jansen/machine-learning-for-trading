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
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from case_studies.utils.registry import compute_fold_metrics_from_predictions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from utils.modeling import RANDOM_SEED, seed_everything

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
            print(
                f"      epoch {epoch:3d}/{n_epochs}: loss={avg_loss:.6f}, IC={ic:+.4f}",
                flush=True,
            )

    return checkpoint_ics, checkpoint_preds, epoch_losses


# ---------------------------------------------------------------------------
# Incremental Save/Load Helpers
# ---------------------------------------------------------------------------


from case_studies.utils.registry.store import flush_fold_predictions


def _load_incremental_preds(incr_dir: Path) -> pl.DataFrame:
    """Reassemble all predictions from incremental fold saves."""
    parquet_files = sorted(incr_dir.glob("*.parquet"))
    if not parquet_files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in parquet_files])


def _load_incremental_preds_for_config(incr_dir: Path, config_name: str) -> pl.DataFrame:
    """Reassemble one config's predictions from its per-fold incremental saves."""
    parquet_files = sorted(incr_dir.glob(f"{config_name}_fold*.parquet"))
    if not parquet_files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in parquet_files])


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
    if register and save_dir is None:
        raise ValueError(
            "register=True requires save_dir for incremental prediction saves. "
            "Pass save_dir=CASE_DIR / 'run_log' / 'training' / 'tabular_dl'"
        )

    # Filter out configs whose training_hash is already complete (unless
    # force_retrain). This prevents re-running finished work across the entire
    # sweep — the caller can override with force_retrain=True for debugging.
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

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    seed_everything(RANDOM_SEED)

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

        # Drop NaN labels
        train_valid = train_df[label_col].notna()
        val_valid = val_df[label_col].notna()
        train_df = train_df.loc[train_valid]
        val_df = val_df.loc[val_valid]

        if len(train_df) < 100 or len(val_df) < 50:
            print(f"  Fold {split['fold']}: skipped (train={len(train_df)}, val={len(val_df)})")
            continue

        X_train = train_df[feature_names].values.astype(np.float32)
        y_train = train_df[label_col].values.astype(np.float32)
        X_val = val_df[feature_names].values.astype(np.float32)
        y_val = val_df[label_col].values.astype(np.float32)
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
    config_results: list[dict[str, Any]] = []
    all_curves: list[dict] = []
    training_log: list[dict] = []

    # Set up incremental save directory
    incr_dir = save_dir / "_incremental" if save_dir is not None else None
    if incr_dir is not None:
        incr_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        config_name = cfg["config_name"]
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
            seed_everything(RANDOM_SEED + fd["fold"])

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
                            "date": fd["val_dates"],
                            "symbol": fd["val_entities"],
                            "y_true": fd["y_val"],
                            "y_pred": preds,
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

        # Find best epoch for this config (mean IC across folds)
        if fold_checkpoint_ics:
            best_cp = max(
                fold_checkpoint_ics.keys(),
                key=lambda ep: (
                    np.nanmean(fold_checkpoint_ics[ep]) if fold_checkpoint_ics[ep] else -1
                ),
            )
            best_ic_val = float(np.nanmean(fold_checkpoint_ics[best_cp]))
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
        for ep, ics in sorted(fold_checkpoint_ics.items()):
            entry = {
                "config": config_name,
                "epoch": ep,
                "ic_mean": float(np.nanmean(ics)),
                "ic_std": float(np.nanstd(ics)) if len(ics) > 1 else 0.0,
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
                cfg_all_preds = _load_incremental_preds_for_config(incr_dir, config_name)
                if cfg_all_preds.height > 0:
                    from case_studies.utils.registry import register_prediction_set

                    cfg_curves_df = pl.DataFrame(cfg_curves_list) if cfg_curves_list else None
                    epoch_ics = {
                        ep: float(np.nanmean(ics)) for ep, ics in fold_checkpoint_ics.items()
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
                        )
                    print(
                        f"    registered {config_name} incrementally ({len(epochs)} per-epoch slices)"
                    )
            except Exception as exc:
                print(f"    WARN: incremental registration failed for {config_name}: {exc}")

        gc.collect()

    if not config_results:
        raise ValueError("No configs completed successfully.")

    # Select best config
    config_results.sort(
        key=lambda r: r["best_ic"] if not np.isnan(r["best_ic"]) else -999,
        reverse=True,
    )
    best_result = config_results[0]
    best_config_name = best_result["config_name"]
    best_epoch = best_result["best_epoch"]
    best_ic = best_result["best_ic"]

    print(f"\n  Best: {best_config_name} @ epoch {best_epoch} (IC={best_ic:+.4f})")

    # Reassemble all predictions from incremental saves
    all_predictions = _load_incremental_preds(incr_dir) if incr_dir is not None else pl.DataFrame()

    # Extract best-config predictions at best epoch
    if all_predictions.height > 0:
        best_preds_df = all_predictions.filter(
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
            predictions.write_parquet(save_dir / "predictions.parquet")
        if all_predictions.height > 0:
            all_predictions.write_parquet(save_dir / "all_predictions.parquet")
        if learning_curves.height > 0:
            learning_curves.write_parquet(save_dir / "learning_curves.parquet")
        if training_log_df.height > 0:
            training_log_df.write_parquet(save_dir / "training_log.parquet")
        print(f"  Saved to {save_dir}")

    # Register in unified registry
    # Note: per-config registration happens incrementally inside the training
    # loop above (see the `register` block after each config's best_epoch is
    # computed). The old batched registration block was removed to avoid
    # duplicate writes — each config's training_hash is persisted immediately
    # after its folds complete, protecting against interruption losing work.

    return {
        "grid_results": config_results,
        "best_config_name": best_config_name,
        "best_epoch": best_epoch,
        "best_ic": best_ic,
        "predictions": predictions,
        "all_predictions": all_predictions,
        "fold_metrics": compute_fold_metrics_from_predictions(
            all_predictions,
            best_config_name,
            best_epoch,
            date_col=date_col,
        ),
        "all_learning_curves": learning_curves,
        "training_log": training_log_df,
    }
