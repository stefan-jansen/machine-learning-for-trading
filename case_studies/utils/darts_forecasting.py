"""Darts-backed global forecasting helpers for production DL case studies."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as pl_lightning
import torch
from darts import TimeSeries
from darts.models import NBEATSModel, TSMixerModel
from ml4t.diagnostic.metrics import cross_sectional_ic

from case_studies.utils.registry import compute_fold_metrics_from_predictions
from utils.modeling import RANDOM_SEED, seed_everything

SUPPORTED_DARTS_ARCHITECTURES = {"nbeats", "tsmixer"}
BASE_TARGET_COL = "_darts_target_1d"


def uses_darts_backend(configs: list[dict[str, Any]]) -> bool:
    """Return True when all configs are Darts-backed supported architectures."""
    if not configs:
        return False
    return all(
        cfg.get("library") == "darts"
        and cfg.get("params", {}).get("architecture") in SUPPORTED_DARTS_ARCHITECTURES
        for cfg in configs
    )


@dataclass
class _FoldSeries:
    entity: str
    full_target: TimeSeries
    full_covariates: TimeSeries
    train_target: TimeSeries
    train_covariates: TimeSeries
    prediction_start_pos: int
    val_start_pos: int
    val_end_pos: int
    dates: np.ndarray
    y_true: np.ndarray
    n_train_samples: int


def _metric_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


from case_studies.utils.registry.store import _save_parquet, flush_fold_training_log


def _flush_darts_fold_training_log(
    log_dir: Path,
    config_name: str,
    fold: int,
    epoch_rows: list[dict[str, Any]],
) -> None:
    flush_fold_training_log(log_dir, config_name, fold, epoch_rows)


def _flush_darts_fold_preds(
    incr_dir: Path,
    config_name: str,
    fold: int,
    prediction_frames: list[pl.DataFrame],
) -> None:
    """Flush pre-assembled prediction DataFrames (darts builds these during training)."""
    if not prediction_frames:
        return
    _save_parquet(incr_dir / f"{config_name}_fold{fold}.parquet", pl.concat(prediction_frames))


class _DartsEpochProgressCallback(pl_lightning.callbacks.Callback):
    def __init__(
        self,
        *,
        config_name: str,
        fold: int,
        n_epochs: int,
        n_train: int,
        log_dir: Path | None,
        epoch_rows: list[dict[str, Any]],
    ) -> None:
        self.config_name = config_name
        self.fold = fold
        self.n_epochs = n_epochs
        self.n_train = n_train
        self.log_dir = log_dir
        self.epoch_rows = epoch_rows
        self._fit_start: float | None = None
        self._prev_elapsed_s = 0.0

    def on_train_start(self, trainer, pl_module) -> None:
        if self._fit_start is None:
            self._fit_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._fit_start is None:
            self._fit_start = time.perf_counter()

        epoch = len(self.epoch_rows) + 1
        elapsed_s = time.perf_counter() - self._fit_start
        epoch_s = elapsed_s - self._prev_elapsed_s
        self._prev_elapsed_s = elapsed_s
        eta_s = ((self.n_epochs - epoch) * elapsed_s / epoch) if epoch else None
        train_loss = _metric_to_float(trainer.callback_metrics.get("train_loss"))
        val_loss = _metric_to_float(trainer.callback_metrics.get("val_loss"))

        row = {
            "config": self.config_name,
            "fold": self.fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ic": None,
            "epoch_s": round(epoch_s, 3),
            "elapsed_s": round(elapsed_s, 1),
            "eta_s": round(eta_s, 1) if eta_s is not None else None,
            "n_train": self.n_train,
            "n_val": None,
            "best_epoch": None,
            "best_ic": None,
        }
        self.epoch_rows.append(row)

        if self.log_dir is not None:
            _flush_darts_fold_training_log(
                self.log_dir,
                self.config_name,
                self.fold,
                self.epoch_rows,
            )

        loss_str = f"{train_loss:.6f}" if train_loss is not None else "n/a"
        eta_min = eta_s / 60 if eta_s is not None else float("nan")
        print(
            f"      epoch {epoch:3d}/{self.n_epochs}: "
            f"train_loss={loss_str} "
            f"elapsed={elapsed_s / 60:.1f}m "
            f"eta={eta_min:.1f}m",
            flush=True,
        )


def _trainer_kwargs(device: str) -> dict[str, Any]:
    accelerator = "gpu" if device == "cuda" and torch.cuda.is_available() else "cpu"
    return {
        "accelerator": accelerator,
        "devices": 1,
        "deterministic": True,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "logger": False,
    }


def _parse_label_horizon(label_col: str) -> int:
    match = re.search(r"_(\d+)d$", label_col)
    if match is None:
        raise ValueError(
            f"Darts forecasting currently requires a daily return label ending in '_<H>d': {label_col}"
        )
    return int(match.group(1))


def _recommended_input_chunk_length(horizon: int) -> int:
    if horizon >= 21:
        return 252
    return max(60, 8 * horizon)


def _resolve_chunk_lengths(cfg: dict[str, Any], label_horizon: int) -> tuple[int, int]:
    params = cfg.get("params", {})
    input_chunk_length = int(
        params.get(
            "input_chunk_length",
            params.get("darts_input_chunk_length", _recommended_input_chunk_length(label_horizon)),
        )
    )
    output_chunk_length = int(params.get("darts_output_chunk_length", label_horizon))
    if input_chunk_length <= output_chunk_length:
        raise ValueError(
            f"Darts requires input_chunk_length > output_chunk_length, got "
            f"{input_chunk_length} <= {output_chunk_length} for {cfg['config_name']}"
        )
    return input_chunk_length, output_chunk_length


def _build_darts_model(
    cfg: dict[str, Any],
    device: str,
    fold_seed: int,
    input_chunk_length: int,
    output_chunk_length: int,
    trainer_callbacks: list[Any] | None = None,
):
    params = dict(cfg.get("params", {}))
    arch = params.pop("architecture")
    model_cls = NBEATSModel if arch == "nbeats" else TSMixerModel
    params.pop("lookback", None)
    params.pop("darts_input_chunk_length", None)
    params.pop("darts_output_chunk_length", None)
    params.pop("input_chunk_length", None)
    params.pop("output_chunk_length", None)
    params["input_chunk_length"] = input_chunk_length
    params["output_chunk_length"] = output_chunk_length
    if arch == "nbeats":
        params["num_stacks"] = int(params.pop("num_stacks", 1))
        params["num_blocks"] = int(params.pop("num_blocks", params.pop("n_blocks", 1)))
        params["num_layers"] = int(params.pop("num_layers", params.pop("n_layers", 4)))
        params["layer_widths"] = int(params.pop("layer_widths", params.pop("hidden_size", 256)))
    else:
        hidden_size = int(params.pop("hidden_size", params.pop("hidden_dim", 64)))
        params["hidden_size"] = hidden_size
        params["ff_size"] = int(params.pop("ff_size", hidden_size))
        params["num_blocks"] = int(params.pop("num_blocks", params.pop("n_blocks", 2)))
        params.setdefault("use_static_covariates", False)
    params["n_epochs"] = cfg.get("n_epochs", 100)
    params["batch_size"] = cfg.get("batch_size", 2048)
    params["random_state"] = fold_seed
    params["save_checkpoints"] = False
    params["force_reset"] = True
    trainer = dict(params.pop("pl_trainer_kwargs", {}))
    trainer = {**_trainer_kwargs(device), **trainer}
    callbacks = trainer.get("callbacks")
    callback_list = list(callbacks) if callbacks is not None else []
    if trainer_callbacks:
        callback_list.extend(trainer_callbacks)
    if callback_list:
        trainer["callbacks"] = callback_list
    params["pl_trainer_kwargs"] = trainer
    return model_cls(**params)


def _resolve_sampling(
    fold_series: list[_FoldSeries],
    input_chunk_length: int,
    output_chunk_length: int,
    max_train_sequences: int,
) -> tuple[int, int | None]:
    sample_counts = [
        max(state.n_train_samples - input_chunk_length - output_chunk_length + 1, 0)
        for state in fold_series
    ]
    total_samples = int(sum(sample_counts))
    if max_train_sequences <= 0 or total_samples <= max_train_sequences:
        return 1, None

    stride = max(1, int(np.ceil(total_samples / max_train_sequences)))
    stride_samples = sum(int(np.ceil(n / stride)) for n in sample_counts if n > 0)
    max_samples_per_ts = None
    if stride_samples > max_train_sequences:
        max_samples_per_ts = max(1, int(np.ceil(max_train_sequences / len(fold_series))))
    return stride, max_samples_per_ts


def _load_base_target_frame(
    case_study: str,
    dataset_pd: pd.DataFrame,
    date_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    if case_study == "etfs":
        from data import load_etfs

        target_df = (
            load_etfs()
            .sort(["symbol", "timestamp"])
            .with_columns(
                ((pl.col("close") / pl.col("close").shift(1).over("symbol")).log()).alias(
                    BASE_TARGET_COL
                )
            )
            .select(["timestamp", "symbol", BASE_TARGET_COL])
        )
        return target_df.to_pandas(), [date_col, "symbol"]

    if case_study == "cme_futures":
        from data import load_cme_futures

        target_df = (
            load_cme_futures()
            .rename({"session_date": "timestamp", "tenor": "position"})
            .sort(["product", "position", "timestamp"])
            .with_columns(
                (
                    (
                        pl.col("adj_close")
                        / pl.col("adj_close").shift(1).over(["product", "position"])
                    ).log()
                ).alias(BASE_TARGET_COL)
            )
            .select(["timestamp", "product", "position", BASE_TARGET_COL])
        )
        join_keys = [date_col, "product"]
        if "position" in dataset_pd.columns:
            join_keys.append("position")
        return target_df.to_pandas(), join_keys

    if case_study == "us_equities_panel":
        from data import load_us_equities

        target_df = (
            load_us_equities(start_date="1990-01-01", end_date="2018-03-31")
            .sort(["symbol", "timestamp"])
            .with_columns(
                ((pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol")).log()).alias(
                    BASE_TARGET_COL
                )
            )
            .select(["timestamp", "symbol", BASE_TARGET_COL])
        )
        return target_df.to_pandas(), [date_col, "symbol"]

    raise RuntimeError(
        "Horizon-aware Darts support is currently implemented for etfs, cme_futures, "
        "and us_equities_panel only. "
        f"{case_study} remains blocked because its label is not a single calendar-time return series."
    )


def _attach_base_target(
    dataset_pd: pd.DataFrame,
    case_study: str,
    date_col: str,
) -> pd.DataFrame:
    target_pd, join_keys = _load_base_target_frame(case_study, dataset_pd, date_col)
    dataset_pd = dataset_pd.copy()
    dataset_pd[date_col] = pd.to_datetime(dataset_pd[date_col])
    target_pd[date_col] = pd.to_datetime(target_pd[date_col])
    merged = dataset_pd.merge(target_pd, on=join_keys, how="left", validate="many_to_one")
    if merged[BASE_TARGET_COL].notna().sum() == 0:
        raise RuntimeError(f"Failed to join base Darts target series for {case_study}")
    return merged


def _prepare_fold_series(
    dataset_pd: pd.DataFrame,
    split: dict[str, Any],
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    input_chunk_length: int,
    output_chunk_length: int,
) -> list[_FoldSeries]:
    train_start = pd.Timestamp(split["train_start"]).to_datetime64()
    train_end = pd.Timestamp(split["train_end"]).to_datetime64()
    val_start = pd.Timestamp(split["val_start"]).to_datetime64()
    val_end = pd.Timestamp(split["val_end"]).to_datetime64()
    cols = [date_col, entity_col, label_col, BASE_TARGET_COL, *feature_names]
    fold_mask = (dataset_pd[date_col] >= train_start) & (dataset_pd[date_col] <= val_end)
    fold_df = dataset_pd.loc[fold_mask, cols].copy().dropna(subset=[BASE_TARGET_COL])
    train_df = fold_df.loc[fold_df[date_col] <= train_end].copy()
    if train_df.empty or fold_df.empty:
        return []

    fold_df = fold_df.astype({name: np.float32 for name in feature_names}, copy=False)
    train_df = train_df.astype({name: np.float32 for name in feature_names}, copy=False)
    feature_frame = fold_df[feature_names].astype(np.float32)
    train_features = train_df[feature_names].astype(np.float32)
    means = train_features.mean()
    stds = train_features.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    fold_df.loc[:, feature_names] = ((feature_frame - means) / stds).fillna(0.0).astype(np.float32)

    series: list[_FoldSeries] = []
    for entity, sym_df in fold_df.groupby(entity_col, sort=False):
        sym_df = sym_df.sort_values(date_col).reset_index(drop=True)
        dates = sym_df[date_col].to_numpy()
        train_cut = int((dates <= train_end).sum())
        val_positions = np.flatnonzero((dates >= val_start) & (dates <= val_end))
        if len(val_positions) == 0:
            continue
        if train_cut < input_chunk_length + output_chunk_length:
            continue

        val_start_pos = int(val_positions[0])
        val_end_pos = int(val_positions[-1])
        prediction_start_pos = val_start_pos + 1
        if prediction_start_pos <= 0 or prediction_start_pos >= len(sym_df):
            continue

        t = np.arange(len(sym_df), dtype=np.int32)
        target_df = pd.DataFrame(
            {"t": t, BASE_TARGET_COL: sym_df[BASE_TARGET_COL].to_numpy(np.float32)}
        )
        cov_df = pd.DataFrame(
            {"t": t, **{f: sym_df[f].to_numpy(np.float32) for f in feature_names}}
        )

        full_target = TimeSeries.from_dataframe(target_df, time_col="t", value_cols=BASE_TARGET_COL)
        full_covariates = TimeSeries.from_dataframe(cov_df, time_col="t", value_cols=feature_names)

        series.append(
            _FoldSeries(
                entity=str(entity),
                full_target=full_target,
                full_covariates=full_covariates,
                train_target=full_target[:train_cut],
                train_covariates=full_covariates[:train_cut],
                prediction_start_pos=prediction_start_pos,
                val_start_pos=val_start_pos,
                val_end_pos=val_end_pos,
                dates=dates,
                y_true=sym_df[label_col].to_numpy(np.float32),
                n_train_samples=train_cut,
            )
        )

    return series


def _predict_fold(
    model,
    fold_series: list[_FoldSeries],
    fold_id: int,
    date_col: str,
    entity_col: str,
    output_chunk_length: int,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for state in fold_series:
        forecasts = model.historical_forecasts(
            state.full_target,
            past_covariates=state.full_covariates,
            start=state.prediction_start_pos,
            start_format="position",
            forecast_horizon=output_chunk_length,
            stride=1,
            retrain=False,
            overlap_end=True,
            last_points_only=False,
            verbose=False,
            show_warnings=False,
        )
        if isinstance(forecasts, TimeSeries):
            forecasts = [forecasts]

        rows: list[dict[str, Any]] = []
        for forecast in forecasts:
            start_pos = int(forecast.start_time())
            base_pos = start_pos - 1
            if base_pos < state.val_start_pos or base_pos > state.val_end_pos:
                continue
            if base_pos < 0 or base_pos >= len(state.dates):
                continue
            score_path = forecast.values(copy=False).reshape(-1).astype(np.float64, copy=False)
            rows.append(
                {
                    date_col: pd.Timestamp(state.dates[base_pos]),
                    entity_col: state.entity,
                    "y_true": float(state.y_true[base_pos]),
                    "y_score": float(np.expm1(score_path.sum())),
                    "fold_id": fold_id,
                }
            )

        if rows:
            frames.append(pl.DataFrame(rows))

    return pl.concat(frames) if frames else pl.DataFrame()


def run_darts_cv(
    dataset_pd: pd.DataFrame,
    splits: list[dict[str, Any]],
    *,
    configs: list[dict[str, Any]],
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    device: str,
    save_dir: Path | None,
    max_train_sequences: int,
    register: bool,
    case_study: str | None,
    notebook: str | None,
    prediction_split: str = "validation",
) -> dict[str, Any]:
    """Run Darts-backed global forecasting models and emit standard DL artifacts."""
    if case_study is None:
        raise ValueError(
            "Darts backends require case_study so the base target series can be built."
        )

    if register and save_dir is None:
        raise ValueError("register=True requires save_dir for Darts prediction artifacts.")

    from case_studies.utils.deep_learning import _register_dl_config

    label_horizon = _parse_label_horizon(label_col)
    dataset_pd = _attach_base_target(dataset_pd.copy(), case_study, date_col)

    config_results: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    prediction_frames: list[pl.DataFrame] = []

    for cfg in configs:
        config_name = cfg["config_name"]
        params = cfg.get("params", {})
        input_chunk_length, output_chunk_length = _resolve_chunk_lengths(cfg, label_horizon)
        cfg_seed = int(cfg.get("seed", RANDOM_SEED))
        n_epochs = int(cfg.get("n_epochs", 100))
        checkpoint_interval = int(cfg.get("checkpoint_interval", n_epochs))
        started_at = datetime.now(UTC).isoformat()
        elapsed_total = 0.0
        cfg_prediction_frames: list[pl.DataFrame] = []

        print(
            f"Darts CV: {config_name} ({params['architecture']}) "
            f"{len(splits)} folds × {n_epochs} epochs | "
            f"input={input_chunk_length} output={output_chunk_length}"
        )

        for split in splits:
            fold_seed = cfg_seed + split["fold"]
            seed_everything(fold_seed)
            fold_series = _prepare_fold_series(
                dataset_pd,
                split,
                feature_names,
                label_col,
                date_col,
                entity_col,
                input_chunk_length,
                output_chunk_length,
            )
            if not fold_series:
                print(f"  Fold {split['fold']}: skipped (insufficient series after filtering)")
                continue

            stride, max_samples_per_ts = _resolve_sampling(
                fold_series,
                input_chunk_length,
                output_chunk_length,
                max_train_sequences,
            )
            if max_train_sequences > 0:
                msg = f"  Fold {split['fold']}: {len(fold_series)} series, stride={stride}"
                if max_samples_per_ts is not None:
                    msg += f", max_samples_per_ts={max_samples_per_ts}"
                print(msg)
            else:
                print(f"  Fold {split['fold']}: {len(fold_series)} series")

            train_series = [state.train_target for state in fold_series]
            train_covariates = [state.train_covariates for state in fold_series]
            epoch_rows: list[dict[str, Any]] = []
            checkpoint_frames: list[pl.DataFrame] = []
            checkpoint_ics: dict[int, float] = {}
            n_val_points = 0
            incr_dir = save_dir / "_incremental" if save_dir is not None else None
            log_dir = save_dir / "_incremental_logs" if save_dir is not None else None
            if incr_dir is not None:
                incr_dir.mkdir(parents=True, exist_ok=True)
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            model = _build_darts_model(
                cfg,
                device,
                fold_seed,
                input_chunk_length,
                output_chunk_length,
                trainer_callbacks=[
                    _DartsEpochProgressCallback(
                        config_name=config_name,
                        fold=split["fold"],
                        n_epochs=n_epochs,
                        n_train=int(sum(state.n_train_samples for state in fold_series)),
                        log_dir=log_dir,
                        epoch_rows=epoch_rows,
                    )
                ],
            )
            epochs_trained = 0
            while epochs_trained < n_epochs:
                epochs_to_train = min(checkpoint_interval, n_epochs - epochs_trained)
                model.fit(
                    train_series,
                    past_covariates=train_covariates,
                    verbose=False,
                    epochs=epochs_to_train,
                    stride=stride,
                    max_samples_per_ts=max_samples_per_ts,
                )
                epochs_trained += epochs_to_train

                checkpoint_preds = _predict_fold(
                    model,
                    fold_series,
                    split["fold"],
                    date_col,
                    entity_col,
                    output_chunk_length,
                )
                elapsed = time.perf_counter() - t0
                if checkpoint_preds.height == 0:
                    print(
                        f"        checkpoint {epochs_trained:3d}/{n_epochs}: "
                        f"no validation predictions ({elapsed:.1f}s elapsed)",
                        flush=True,
                    )
                    continue

                n_val_points = checkpoint_preds.height
                checkpoint_preds = checkpoint_preds.with_columns(
                    pl.lit(config_name).alias("config"),
                    pl.lit(epochs_trained).alias("epoch"),
                )
                checkpoint_frames.append(checkpoint_preds)
                cfg_prediction_frames.append(checkpoint_preds)
                prediction_frames.append(checkpoint_preds)
                if incr_dir is not None:
                    _flush_darts_fold_preds(incr_dir, config_name, split["fold"], checkpoint_frames)

                _entity = entity_col if entity_col in checkpoint_preds.columns else None
                ic = cross_sectional_ic(
                    checkpoint_preds,
                    checkpoint_preds,
                    pred_col="y_score",
                    ret_col="y_true",
                    date_col=date_col,
                    entity_col=_entity,
                    method="spearman",
                    min_obs=5,
                )["ic_mean"]
                checkpoint_ics[epochs_trained] = ic
                for row in reversed(epoch_rows):
                    if row["epoch"] == epochs_trained:
                        row["ic"] = round(ic, 4)
                        break
                if log_dir is not None:
                    _flush_darts_fold_training_log(log_dir, config_name, split["fold"], epoch_rows)
                print(
                    f"        checkpoint {epochs_trained:3d}/{n_epochs}: "
                    f"fold IC={ic:+.4f} ({elapsed:.1f}s elapsed)",
                    flush=True,
                )

            elapsed = time.perf_counter() - t0
            elapsed_total += elapsed
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not checkpoint_ics:
                for row in epoch_rows:
                    row["n_val"] = 0
                    row["best_epoch"] = n_epochs
                if log_dir is not None:
                    _flush_darts_fold_training_log(log_dir, config_name, split["fold"], epoch_rows)
                training_log.extend(epoch_rows)
                print(f"    no validation predictions generated ({elapsed:.1f}s)")
                continue

            fold_best_epoch = max(checkpoint_ics, key=checkpoint_ics.get)
            fold_best_ic = checkpoint_ics[fold_best_epoch]
            for row in epoch_rows:
                row["n_val"] = n_val_points
                row["best_epoch"] = fold_best_epoch
                row["best_ic"] = round(fold_best_ic, 4)
            if log_dir is not None:
                _flush_darts_fold_training_log(log_dir, config_name, split["fold"], epoch_rows)
            training_log.extend(epoch_rows)
            print(f"    fold best epoch={fold_best_epoch}, IC={fold_best_ic:+.4f} ({elapsed:.1f}s)")

        epoch_scores: list[tuple[int, float, float]] = []
        if cfg_prediction_frames:
            cfg_all_preds = pl.concat(cfg_prediction_frames)
            for epoch in sorted(cfg_all_preds["epoch"].unique().to_list()):
                ep_df = cfg_all_preds.filter(pl.col("epoch") == epoch)
                fold_ids = sorted(ep_df["fold_id"].unique().to_list())
                fold_epoch_ics = []
                for fold_id in fold_ids:
                    fold_df = ep_df.filter(pl.col("fold_id") == fold_id)
                    _entity = entity_col if entity_col in fold_df.columns else None
                    fold_epoch_ics.append(
                        cross_sectional_ic(
                            fold_df,
                            fold_df,
                            pred_col="y_score",
                            ret_col="y_true",
                            date_col=date_col,
                            entity_col=_entity,
                            method="spearman",
                            min_obs=5,
                        )["ic_mean"]
                    )
                ic_mean = float(np.nanmean(fold_epoch_ics))
                ic_std = float(np.nanstd(fold_epoch_ics)) if len(fold_epoch_ics) > 1 else 0.0
                learning_rows.append(
                    {
                        "config": config_name,
                        "epoch": epoch,
                        "ic_mean": ic_mean,
                        "ic_std": ic_std,
                    }
                )
                epoch_scores.append((epoch, ic_mean, ic_std))

        if epoch_scores:
            best_epoch, best_ic, best_ic_std = max(epoch_scores, key=lambda item: item[1])
        else:
            best_epoch, best_ic, best_ic_std = n_epochs, float("nan"), 0.0
        config_results.append(
            {
                "config_name": config_name,
                "best_epoch": best_epoch,
                "best_ic": best_ic,
                "input_chunk_length": input_chunk_length,
                "elapsed_s": elapsed_total,
                "started_at": started_at,
            }
        )
        print(
            f"  {config_name}: epoch={best_epoch}, IC={best_ic:+.4f} "
            f"(std={best_ic_std:.4f}, {elapsed_total:.1f}s)"
        )

    if not config_results:
        raise RuntimeError("Darts run produced no config results.")

    config_results.sort(
        key=lambda row: row["best_ic"] if not np.isnan(row["best_ic"]) else -999,
        reverse=True,
    )
    best_result = config_results[0]
    best_config_name = best_result["config_name"]
    best_epoch = best_result["best_epoch"]
    best_ic = best_result["best_ic"]

    all_predictions = pl.concat(prediction_frames) if prediction_frames else pl.DataFrame()
    predictions = (
        all_predictions.filter(
            (pl.col("config") == best_config_name) & (pl.col("epoch") == best_epoch)
        )
        .with_columns(pl.lit(best_config_name).alias("model_id"))
        .drop("config", "epoch")
        if all_predictions.height > 0
        else pl.DataFrame()
    )
    learning_curves = pl.DataFrame(learning_rows) if learning_rows else pl.DataFrame()
    training_log_df = pl.DataFrame(training_log) if training_log else pl.DataFrame()

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

    if register and case_study and all_predictions.height > 0:
        for row in config_results:
            cfg_name = row["config_name"]
            cfg = next(c for c in configs if c["config_name"] == cfg_name)
            cfg_preds = all_predictions.filter(
                (pl.col("config") == cfg_name) & (pl.col("epoch") == row["best_epoch"])
            )
            cfg_curves = learning_curves.filter(pl.col("config") == cfg_name)
            _register_dl_config(
                case_study=case_study,
                label=label_col,
                config_name=cfg_name,
                architecture=cfg["params"]["architecture"],
                n_epochs=cfg.get("n_epochs", 100),
                best_epoch=row["best_epoch"],
                lookback=row["input_chunk_length"],
                n_folds=len(splits),
                ic_mean=row["best_ic"],
                predictions=cfg_preds,
                notebook=notebook,
                learning_curves=cfg_curves,
                started_at=row.get("started_at"),
                elapsed_s=row.get("elapsed_s"),
                prediction_split=prediction_split,
            )

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
