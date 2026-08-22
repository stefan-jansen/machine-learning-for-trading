"""Darts-backed global forecasting helpers for production DL case studies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
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

from case_studies.utils.cv_results import assemble_cv_result
from utils.modeling import RANDOM_SEED, seed_everything

SUPPORTED_DARTS_ARCHITECTURES = {"nbeats", "tsmixer"}
BASE_TARGET_COL = "_darts_target_1d"
_DARTS_PERIOD_COL = "_darts_expected_period"
_CME_MAX_SESSION_GAP_DAYS = 5


def darts_checkpoint_path(root: Path, config_name: str, fold: int, checkpoint: int) -> Path:
    return Path(root) / config_name / f"fold_{fold:02d}" / f"epoch_{checkpoint:04d}.pt"


def _darts_checkpoint_files(path: Path) -> tuple[Path, Path, Path]:
    path = Path(path)
    return path, Path(f"{path}.ckpt"), Path(f"{path}.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_darts_checkpoint(
    path: Path,
    *,
    model: Any,
    architecture: str,
    metadata: dict[str, Any],
) -> Path:
    """Persist an immutable Darts model, Lightning weights, and digest record."""
    path = Path(path)
    model_path, weights_path, sidecar_path = _darts_checkpoint_files(path)
    existing = [item for item in (model_path, weights_path, sidecar_path) if item.exists()]
    if existing:
        raise FileExistsError(f"immutable Darts checkpoint conflict: {existing[0]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_model = path.with_name(f".{path.name}.{token}.tmp")
    temporary_weights = Path(f"{temporary_model}.ckpt")
    temporary_sidecar = path.with_name(f".{path.name}.{token}.json.tmp")
    published: list[Path] = []
    try:
        model.save(str(temporary_model), clean=True)
        if not temporary_model.is_file() or not temporary_weights.is_file():
            raise RuntimeError(f"Darts did not persist both checkpoint files for {path}")
        record = {
            "schema_version": 1,
            "architecture": architecture,
            "metadata": metadata,
            "model_sha256": _file_sha256(temporary_model),
            "model_size": temporary_model.stat().st_size,
            "weights_sha256": _file_sha256(temporary_weights),
            "weights_size": temporary_weights.stat().st_size,
        }
        temporary_sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        for source, target in (
            (temporary_model, model_path),
            (temporary_weights, weights_path),
            (temporary_sidecar, sidecar_path),
        ):
            os.replace(source, target)
            published.append(target)
    except Exception:
        for item in published:
            item.unlink(missing_ok=True)
        raise
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_weights.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    return path


def load_darts_checkpoint(path: Path, *, device: str = "cpu"):
    """Load a Darts checkpoint after both persisted files pass their digests."""
    model_path, weights_path, sidecar_path = _darts_checkpoint_files(Path(path))
    if not all(item.is_file() for item in (model_path, weights_path, sidecar_path)):
        raise FileNotFoundError(f"Darts checkpoint population is incomplete: {model_path}")
    record = json.loads(sidecar_path.read_text())
    if record.get("schema_version") != 1:
        raise ValueError(f"unsupported Darts checkpoint schema at {model_path}")
    if record.get("model_sha256") != _file_sha256(model_path) or record.get(
        "weights_sha256"
    ) != _file_sha256(weights_path):
        raise ValueError(f"Darts checkpoint digest does not match its sidecar: {model_path}")
    model_cls = {
        "nbeats": NBEATSModel,
        "tsmixer": TSMixerModel,
    }.get(record.get("architecture"))
    if model_cls is None:
        raise ValueError(f"unsupported Darts architecture at {model_path}")
    model = model_cls.load(
        str(model_path),
        pl_trainer_kwargs=_trainer_kwargs(device),
        weights_only=False,
    )
    return model, record["metadata"]


def validate_darts_checkpoint_population(
    root: Path,
    *,
    config_name: str,
    fold_ids: list[int] | tuple[int, ...],
    checkpoints: list[int] | tuple[int, ...],
    architecture: str,
) -> tuple[Path, ...]:
    """Require the exact Darts fitted-state population declared by one request."""
    expected = tuple(
        darts_checkpoint_path(root, config_name, fold, checkpoint)
        for fold in sorted({int(value) for value in fold_ids})
        for checkpoint in sorted({int(value) for value in checkpoints})
    )
    if not expected:
        raise ValueError("Darts checkpoint validation requires folds and checkpoint values")
    expected_files: set[Path] = set()
    for path in expected:
        expected_files.update(_darts_checkpoint_files(path))
        try:
            _model_path, _weights_path, sidecar_path = _darts_checkpoint_files(path)
            if not all(item.is_file() for item in _darts_checkpoint_files(path)):
                raise FileNotFoundError(path)
            record = json.loads(sidecar_path.read_text())
            if (
                record.get("schema_version") != 1
                or record.get("model_sha256") != _file_sha256(path)
                or record.get("weights_sha256") != _file_sha256(Path(f"{path}.ckpt"))
            ):
                raise ValueError(path)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Darts fitted checkpoint population is incomplete: {path}") from error
        fold = int(path.parent.name.removeprefix("fold_"))
        checkpoint = int(path.stem.removeprefix("epoch_"))
        required_metadata = {
            "config_name": config_name,
            "fold": fold,
            "checkpoint_kind": "epoch",
            "checkpoint_value": checkpoint,
        }
        mismatches = {
            key: (record.get("metadata", {}).get(key), value)
            for key, value in required_metadata.items()
            if record.get("metadata", {}).get(key) != value
        }
        if record.get("architecture") != architecture:
            mismatches["architecture"] = (record.get("architecture"), architecture)
        if mismatches:
            raise ValueError(f"Darts fitted checkpoint metadata mismatch at {path}: {mismatches}")

    config_root = Path(root) / config_name
    actual_files = {path for path in config_root.glob("fold_*/epoch_*.pt*") if path.is_file()}
    extras = actual_files - expected_files
    if extras:
        raise ValueError(
            "Darts fitted checkpoint population contains undeclared artifacts: "
            f"{[str(path) for path in sorted(extras)]}"
        )
    return expected


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
    identity: dict[str, Any]
    full_target: TimeSeries
    full_covariates: TimeSeries
    train_target: TimeSeries | None
    train_covariates: TimeSeries | None
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
            params.get(
                "darts_input_chunk_length",
                params.get("lookback", _recommended_input_chunk_length(label_horizon)),
            ),
        )
    )
    output_chunk_length = int(params.get("darts_output_chunk_length", label_horizon))
    if input_chunk_length <= output_chunk_length:
        raise ValueError(
            f"Darts requires input_chunk_length > output_chunk_length, got "
            f"{input_chunk_length} <= {output_chunk_length} for {cfg['config_name']}"
        )
    return input_chunk_length, output_chunk_length


def darts_training_identity(
    cfg: dict[str, Any],
    label_col: str,
    *,
    case_study: str,
    input_data_spec: dict[str, Any] | None,
    max_train_sequences: int,
) -> dict[str, Any]:
    """Return the runtime parameters that define a Darts training run."""
    input_chunk_length, output_chunk_length = _resolve_chunk_lengths(
        cfg, _parse_label_horizon(label_col)
    )
    target_mode = str(cfg.get("params", {}).get("darts_target", "one_period_return"))
    base_target_data_spec = (
        {
            "delay_periods": output_chunk_length,
            "kind": "lagged_label",
            "label": label_col,
        }
        if target_mode == "lagged_label"
        else darts_base_target_identity(case_study)
    )
    return {
        "batch_size": cfg.get("batch_size", 2048),
        "base_target_data_spec": base_target_data_spec,
        "input_chunk_length": input_chunk_length,
        "input_data_spec": input_data_spec,
        "lookback": cfg.get("params", {}).get("lookback", input_chunk_length),
        "max_train_sequences": max_train_sequences,
        "output_chunk_length": output_chunk_length,
    }


def darts_base_target_identity(case_study: str) -> dict[str, str]:
    """Hash the raw market file from which Darts derives its one-period target."""
    from utils import ML4T_DATA_PATH

    relative_paths = {
        "etfs": Path("etfs/market/etf_universe.parquet"),
        "cme_futures": Path("futures/market/continuous/daily/continuous_daily.parquet"),
        "us_equities_panel": Path("equities/market/us_equities/us_equities.parquet"),
    }
    if case_study not in relative_paths:
        raise ValueError(f"No Darts base-target identity is defined for {case_study}")
    relative_path = relative_paths[case_study]
    path = ML4T_DATA_PATH / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing Darts base-target file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "dataset": relative_path.as_posix(),
        "sha256": f"sha256:{digest.hexdigest()}",
    }


def select_full_coverage_checkpoint(
    curve: list[dict[str, Any]],
) -> tuple[dict[str, Any], float | None, list[int]]:
    """Select maximum daily IC only among maximum-coverage checkpoints."""
    if not curve:
        raise ValueError("Cannot select a checkpoint from an empty curve")
    finite_days = [item["ic_n_days"] for item in curve if np.isfinite(item["ic_n_days"])]
    full_days = max(finite_days) if finite_days else None
    eligible = [
        item for item in curve if np.isfinite(item["ic_n_days"]) and item["ic_n_days"] == full_days
    ] or curve
    partial_epochs = [item["epoch"] for item in curve if item not in eligible]
    return max(eligible, key=lambda item: item["ic_mean"]), full_days, partial_epochs


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
    params.pop("decision_cadence", None)
    params.pop("lookback", None)
    params.pop("darts_input_chunk_length", None)
    params.pop("darts_output_chunk_length", None)
    params.pop("darts_target", None)
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


def _attach_expected_periods(
    dataset_pd: pd.DataFrame,
    *,
    date_col: str,
    calendar_id: str | None,
    case_study: str | None = None,
) -> pd.DataFrame:
    from case_studies.utils.sequence_dataset import _sequence_period_numbers

    result = dataset_pd.copy()
    if case_study == "cme_futures":
        product_dates = (
            result[["product", date_col]]
            .drop_duplicates()
            .sort_values(["product", date_col], kind="mergesort")
        )
        within_product = product_dates.groupby("product", sort=False)
        sequence = within_product.cumcount()
        breaks = within_product[date_col].diff().dt.days.gt(_CME_MAX_SESSION_GAP_DAYS)
        product_dates[_DARTS_PERIOD_COL] = (
            sequence + breaks.groupby(product_dates["product"], sort=False).cumsum()
        )
        return result.merge(
            product_dates,
            on=["product", date_col],
            how="left",
            validate="many_to_one",
            sort=False,
        )
    result[_DARTS_PERIOD_COL] = _sequence_period_numbers(
        result[date_col],
        calendar_id=calendar_id,
    )
    return result


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

        join_keys = [date_col, "product", "position"]
        missing_keys = set(join_keys) - set(dataset_pd.columns)
        if missing_keys:
            raise ValueError(f"CME Darts dataset is missing panel keys: {sorted(missing_keys)}")
        target_df = load_cme_futures().rename({"session_date": "timestamp", "tenor": "position"})
        eligible_keys = pl.from_pandas(dataset_pd[join_keys].drop_duplicates()).with_columns(
            *(pl.col(key).cast(target_df.schema[key]) for key in join_keys)
        )
        target_df = (
            target_df.join(eligible_keys, on=join_keys, how="inner", validate="m:1")
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
        assert isinstance(target_df, pl.DataFrame)
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


def _panel_identity_columns(dataset_pd: pd.DataFrame, entity_col: str) -> list[str]:
    identity_cols = [entity_col]
    if "position" in dataset_pd.columns and entity_col != "position":
        identity_cols.append("position")
    return identity_cols


def _attach_darts_target(
    dataset_pd: pd.DataFrame,
    *,
    case_study: str,
    date_col: str,
    entity_col: str,
    label_col: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    params = config.get("params", {})
    mode = str(params.get("darts_target", "one_period_return"))
    if mode == "one_period_return":
        if params.get("decision_cadence") is not None:
            raise ValueError("cadence-selected Darts runs require an explicit cadence-aware target")
        return _attach_base_target(dataset_pd, case_study, date_col)
    if mode != "lagged_label":
        raise ValueError(f"unsupported Darts target mode {mode!r}")
    _, output_chunk_length = _resolve_chunk_lengths(config, _parse_label_horizon(label_col))
    if output_chunk_length != 2:
        raise ValueError("lagged-label Darts targets require a two-period forecast")
    if _DARTS_PERIOD_COL not in dataset_pd.columns:
        raise ValueError("lagged-label Darts targets require expected-period identity")
    if label_col not in dataset_pd.columns:
        raise ValueError(f"lagged-label Darts target is missing {label_col!r}")

    result = dataset_pd.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    identity_cols = _panel_identity_columns(result, entity_col)
    result = result.sort_values([*identity_cols, date_col], kind="stable")
    segment_col = "_darts_target_segment"
    if segment_col in result.columns:
        raise ValueError(f"Darts dataset contains reserved column {segment_col!r}")
    result[segment_col] = (
        result.groupby(identity_cols, sort=False)[_DARTS_PERIOD_COL]
        .diff()
        .ne(1)
        .groupby([result[column] for column in identity_cols])
        .cumsum()
    )
    target = result.groupby([*identity_cols, segment_col], sort=False)[label_col].shift(
        output_chunk_length
    )
    if (target.dropna() <= -1.0).any():
        raise ValueError("lagged-label Darts targets require returns greater than -1")
    result[BASE_TARGET_COL] = np.log1p(target)
    return result.drop(columns=segment_col)


def darts_validation_keys(
    dataset_pd: pd.DataFrame,
    splits: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    case_study: str,
    calendar_id: str | None = None,
    temporal_by_fold=None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
) -> pl.DataFrame:
    """Return exact finite validation keys eligible for Darts forecasting."""
    input_chunk_length, output_chunk_length = _resolve_chunk_lengths(
        config, _parse_label_horizon(label_col)
    )
    if calendar_id is None:
        from utils.cv_splits import make_walk_forward_config

        calendar_id = make_walk_forward_config(case_study, date_col=date_col).calendar_id
    dataset_pd = _attach_expected_periods(
        dataset_pd.copy(),
        date_col=date_col,
        calendar_id=calendar_id,
        case_study=case_study,
    )
    dataset_pd = _attach_darts_target(
        dataset_pd,
        case_study=case_study,
        date_col=date_col,
        entity_col=entity_col,
        label_col=label_col,
        config=config,
    )
    identity_cols = _panel_identity_columns(dataset_pd, entity_col)
    frames: list[pl.DataFrame] = []
    has_fold_temporal = temporal_by_fold is not None and temporal_keys and temporal_feature_names
    for split in splits:
        fold_dataset = (
            _overlay_fold_temporal_features(
                dataset_pd,
                split,
                date_col,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
            )
            if has_fold_temporal
            else dataset_pd
        )
        states = _prepare_fold_series(
            fold_dataset,
            split,
            feature_names,
            label_col,
            date_col,
            entity_col,
            input_chunk_length,
            output_chunk_length,
        )
        rows = [
            {
                **state.identity,
                "timestamp": pd.Timestamp(state.dates[position]),
                "fold": int(split["fold"]),
            }
            for state in states
            for position in range(state.prediction_start_pos - 1, state.val_end_pos + 1)
            if state.val_start_pos >= 0
            if np.isfinite(state.y_true[position])
        ]
        if rows:
            frames.append(pl.from_dicts(rows))
    if not frames:
        identity_schema = pl.from_pandas(dataset_pd[identity_cols].head(0)).schema
        return pl.DataFrame(
            schema={**identity_schema, "timestamp": pl.Datetime("ns"), "fold": pl.Int64}
        )
    key_cols = [*identity_cols, "timestamp", "fold"]
    expected = pl.concat(frames).sort(key_cols)
    if expected.n_unique(key_cols) != expected.height:
        raise ValueError("Darts request produced duplicate expected prediction keys")
    return expected


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
    if _DARTS_PERIOD_COL not in dataset_pd.columns:
        raise ValueError("Darts dataset is missing expected-period identity")
    identity_cols = _panel_identity_columns(dataset_pd, entity_col)
    cols = [
        date_col,
        *identity_cols,
        label_col,
        BASE_TARGET_COL,
        _DARTS_PERIOD_COL,
        *feature_names,
    ]
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
    identity_grouper: str | list[str] = (
        identity_cols[0] if len(identity_cols) == 1 else identity_cols
    )
    for identity_key, entity_df in fold_df.groupby(identity_grouper, sort=False):
        identity_values = identity_key if isinstance(identity_key, tuple) else (identity_key,)
        identity = dict(zip(identity_cols, identity_values, strict=True))
        entity_df = entity_df.sort_values(date_col).reset_index(drop=True)
        segments = entity_df[_DARTS_PERIOD_COL].diff().ne(1).cumsum()
        for _, sym_df in entity_df.groupby(segments, sort=False):
            sym_df = sym_df.reset_index(drop=True)
            dates = sym_df[date_col].to_numpy()
            train_cut = int((dates <= train_end).sum())
            val_positions = np.flatnonzero((dates >= val_start) & (dates <= val_end))
            can_train = train_cut >= input_chunk_length + output_chunk_length
            val_start_pos = int(val_positions[0]) if len(val_positions) else -1
            val_end_pos = int(val_positions[-1]) if len(val_positions) else -1
            first_prediction_base = max(val_start_pos, input_chunk_length - 1)
            prediction_start_pos = first_prediction_base + 1 if val_start_pos >= 0 else -1
            can_predict = (
                val_start_pos >= 0
                and first_prediction_base <= val_end_pos
                and prediction_start_pos <= len(sym_df)
            )
            if not can_train and not can_predict:
                continue

            t = np.arange(len(sym_df), dtype=np.int32)
            target_df = pd.DataFrame(
                {"t": t, BASE_TARGET_COL: sym_df[BASE_TARGET_COL].to_numpy(np.float32)}
            )
            cov_df = pd.DataFrame(
                {"t": t, **{f: sym_df[f].to_numpy(np.float32) for f in feature_names}}
            )
            full_target = TimeSeries.from_dataframe(
                target_df, time_col="t", value_cols=BASE_TARGET_COL
            )
            full_covariates = TimeSeries.from_dataframe(
                cov_df, time_col="t", value_cols=feature_names
            )
            series.append(
                _FoldSeries(
                    identity=identity,
                    full_target=full_target,
                    full_covariates=full_covariates,
                    train_target=full_target[:train_cut] if can_train else None,
                    train_covariates=full_covariates[:train_cut] if can_train else None,
                    prediction_start_pos=prediction_start_pos if can_predict else -1,
                    val_start_pos=val_start_pos if can_predict else -1,
                    val_end_pos=val_end_pos if can_predict else -1,
                    dates=dates,
                    y_true=sym_df[label_col].to_numpy(np.float32),
                    n_train_samples=train_cut if can_train else 0,
                )
            )

    return series


def _overlay_fold_temporal_features(
    dataset_pd: pd.DataFrame,
    split: dict[str, Any],
    date_col: str,
    temporal_by_fold,
    temporal_keys: list[str] | None,
    temporal_feature_names: list[str] | None,
) -> pd.DataFrame:
    """Return the requested fold with its training-fitted temporal features."""
    if temporal_by_fold is None or not temporal_keys or not temporal_feature_names:
        return dataset_pd
    from utils.modeling import replace_temporal_columns

    fold_mask = (dataset_pd[date_col] >= split["train_start"]) & (
        dataset_pd[date_col] <= split["val_end"]
    )
    return replace_temporal_columns(
        dataset_pd,
        fold_mask,
        temporal_by_fold,
        temporal_keys,
        temporal_feature_names,
        split["fold"],
    )


def _predict_fold(
    model,
    fold_series: list[_FoldSeries],
    fold_id: int,
    date_col: str,
    entity_col: str,
    output_chunk_length: int,
    forecast_reduction: str = "compound_path",
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for state in fold_series:
        if state.val_start_pos < 0:
            continue
        if state.prediction_start_pos == len(state.full_target):
            forecasts = model.predict(
                output_chunk_length,
                series=state.full_target,
                past_covariates=state.full_covariates,
                verbose=False,
            )
        else:
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
            if not np.isfinite(state.y_true[base_pos]):
                continue
            score_path = forecast.values(copy=False).reshape(-1).astype(np.float64, copy=False)
            if forecast_reduction == "terminal":
                y_score = float(np.expm1(score_path[-1]))
            elif forecast_reduction == "compound_path":
                y_score = float(np.expm1(score_path.sum()))
            else:
                raise ValueError(f"unsupported Darts forecast reduction {forecast_reduction!r}")
            rows.append(
                {
                    date_col: pd.Timestamp(state.dates[base_pos]),
                    **state.identity,
                    "y_true": float(state.y_true[base_pos]),
                    "y_score": y_score,
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
    identity_params: dict[str, Any] | None = None,
    input_data_spec: dict[str, Any] | None = None,
    temporal_by_fold=None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    checkpoint_root: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Run Darts-backed global forecasting models and emit standard DL artifacts."""
    if case_study is None:
        raise ValueError(
            "Darts backends require case_study so the base target series can be built."
        )

    if register and save_dir is None:
        raise ValueError("register=True requires save_dir for Darts prediction artifacts.")

    from case_studies.utils.deep_learning import _register_dl_config

    def _config_identity_params(cfg: dict[str, Any]) -> dict[str, Any] | None:
        params = dict(identity_params or {})
        if input_data_spec is not None:
            params.update(
                darts_training_identity(
                    cfg,
                    label_col,
                    case_study=case_study,
                    input_data_spec=input_data_spec,
                    max_train_sequences=max_train_sequences,
                )
            )
        return params or None

    label_horizon = _parse_label_horizon(label_col)
    from utils.cv_splits import make_walk_forward_config

    dataset_pd = _attach_expected_periods(
        dataset_pd.copy(),
        date_col=date_col,
        calendar_id=make_walk_forward_config(case_study, date_col=date_col).calendar_id,
        case_study=case_study,
    )

    config_results: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    prediction_frames: list[pl.DataFrame] = []
    has_fold_temporal = temporal_by_fold is not None and temporal_keys and temporal_feature_names
    one_period_dataset: pd.DataFrame | None = None

    for cfg in configs:
        config_name = cfg["config_name"]
        params = cfg.get("params", {})
        if params.get("darts_target", "one_period_return") == "one_period_return":
            if one_period_dataset is None:
                one_period_dataset = _attach_darts_target(
                    dataset_pd,
                    case_study=case_study,
                    date_col=date_col,
                    entity_col=entity_col,
                    label_col=label_col,
                    config=cfg,
                )
            config_dataset = one_period_dataset
        else:
            config_dataset = _attach_darts_target(
                dataset_pd,
                case_study=case_study,
                date_col=date_col,
                entity_col=entity_col,
                label_col=label_col,
                config=cfg,
            )
        input_chunk_length, output_chunk_length = _resolve_chunk_lengths(cfg, label_horizon)
        cfg_seed = int(cfg.get("seed", RANDOM_SEED))
        n_epochs = int(cfg.get("n_epochs", 100))
        checkpoint_interval = int(cfg.get("checkpoint_interval", n_epochs))
        started_at = datetime.now(UTC).isoformat()
        elapsed_total = 0.0
        cfg_prediction_frames: list[pl.DataFrame] = []
        expected_fold_ids: list[int] = []

        print(
            f"Darts CV: {config_name} ({params['architecture']}) "
            f"{len(splits)} folds × {n_epochs} epochs | "
            f"input={input_chunk_length} output={output_chunk_length}"
        )

        for split in splits:
            fold_seed = cfg_seed + split["fold"]
            seed_everything(fold_seed)
            fold_dataset = (
                _overlay_fold_temporal_features(
                    config_dataset,
                    split,
                    date_col,
                    temporal_by_fold,
                    temporal_keys,
                    temporal_feature_names,
                )
                if has_fold_temporal
                else config_dataset
            )
            fold_series = _prepare_fold_series(
                fold_dataset,
                split,
                feature_names,
                label_col,
                date_col,
                entity_col,
                input_chunk_length,
                output_chunk_length,
            )
            training_states = [state for state in fold_series if state.train_target is not None]
            prediction_states = [state for state in fold_series if state.val_start_pos >= 0]
            if not training_states or not prediction_states:
                if strict:
                    raise ValueError(
                        f"Darts fold {split['fold']} has no trainable or predictable series "
                        "after gap filtering"
                    )
                print(f"  Fold {split['fold']}: skipped (insufficient gap-free series)")
                continue
            expected_fold_ids.append(int(split["fold"]))

            stride, max_samples_per_ts = _resolve_sampling(
                training_states,
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
                print(
                    f"  Fold {split['fold']}: {len(training_states)} training series, "
                    f"{len(prediction_states)} validation series"
                )

            train_series = [state.train_target for state in training_states]
            train_covariates = [state.train_covariates for state in training_states]
            epoch_rows: list[dict[str, Any]] = []
            checkpoint_frames: list[pl.DataFrame] = []
            checkpoint_ics: dict[int, float] = {}
            checkpoint_n_days: dict[int, int] = {}
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
                        n_train=int(sum(state.n_train_samples for state in training_states)),
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
                if checkpoint_root is not None:
                    write_darts_checkpoint(
                        darts_checkpoint_path(
                            checkpoint_root,
                            config_name,
                            int(split["fold"]),
                            epochs_trained,
                        ),
                        model=model,
                        architecture=str(params["architecture"]),
                        metadata={
                            "config_name": config_name,
                            "fold": int(split["fold"]),
                            "checkpoint_kind": "epoch",
                            "checkpoint_value": epochs_trained,
                        },
                    )

                checkpoint_preds = _predict_fold(
                    model,
                    prediction_states,
                    split["fold"],
                    date_col,
                    entity_col,
                    output_chunk_length,
                    forecast_reduction=(
                        "terminal"
                        if params.get("darts_target") == "lagged_label"
                        else "compound_path"
                    ),
                )
                elapsed = time.perf_counter() - t0
                if checkpoint_preds.height == 0:
                    if strict:
                        raise ValueError(
                            f"Darts fold {split['fold']} checkpoint {epochs_trained} "
                            "produced no validation predictions"
                        )
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
                ic_result = cross_sectional_ic(
                    checkpoint_preds,
                    checkpoint_preds,
                    pred_col="y_score",
                    ret_col="y_true",
                    date_col=date_col,
                    entity_col=_entity,
                    method="spearman",
                    min_obs=5,
                )
                ic = float(ic_result["ic_mean"])
                checkpoint_ics[epochs_trained] = ic
                checkpoint_n_days[epochs_trained] = int(ic_result["n_periods"])
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

            full_fold_coverage = max(checkpoint_n_days.values())
            fold_best_epoch = max(
                (
                    epoch
                    for epoch, n_days in checkpoint_n_days.items()
                    if n_days == full_fold_coverage
                ),
                key=lambda epoch: checkpoint_ics[epoch],
            )
            fold_best_ic = checkpoint_ics[fold_best_epoch]
            for row in epoch_rows:
                row["n_val"] = n_val_points
                row["best_epoch"] = fold_best_epoch
                row["best_ic"] = round(fold_best_ic, 4)
            if log_dir is not None:
                _flush_darts_fold_training_log(log_dir, config_name, split["fold"], epoch_rows)
            training_log.extend(epoch_rows)
            print(f"    fold best epoch={fold_best_epoch}, IC={fold_best_ic:+.4f} ({elapsed:.1f}s)")

        epoch_scores: list[tuple[int, float, float, int]] = []
        if cfg_prediction_frames:
            cfg_all_preds = pl.concat(cfg_prediction_frames)
            expected_fold_ids = sorted(cfg_all_preds["fold_id"].unique().to_list())
            for epoch in sorted(cfg_all_preds["epoch"].unique().to_list()):
                ep_df = cfg_all_preds.filter(pl.col("epoch") == epoch)
                _entity = entity_col if entity_col in ep_df.columns else None
                ic_mean = float(
                    cross_sectional_ic(
                        ep_df,
                        ep_df,
                        pred_col="y_score",
                        ret_col="y_true",
                        date_col=date_col,
                        entity_col=_entity,
                        method="spearman",
                        min_obs=5,
                    )["ic_mean"]
                )
                fold_ids = sorted(ep_df["fold_id"].unique().to_list())
                if fold_ids != expected_fold_ids:
                    continue
                fold_epoch_ics = []
                fold_n_days = []
                for fold_id in fold_ids:
                    fold_df = ep_df.filter(pl.col("fold_id") == fold_id)
                    _entity = entity_col if entity_col in fold_df.columns else None
                    ic_result = cross_sectional_ic(
                        fold_df,
                        fold_df,
                        pred_col="y_score",
                        ret_col="y_true",
                        date_col=date_col,
                        entity_col=_entity,
                        method="spearman",
                        min_obs=5,
                    )
                    fold_epoch_ics.append(float(ic_result["ic_mean"]))
                    fold_n_days.append(int(ic_result["n_periods"]))
                ic_std = float(np.nanstd(fold_epoch_ics)) if len(fold_epoch_ics) > 1 else 0.0
                ic_n_days = sum(fold_n_days)
                learning_rows.append(
                    {
                        "config": config_name,
                        "epoch": epoch,
                        "ic_mean": ic_mean,
                        "ic_std": ic_std,
                        "ic_n_days": ic_n_days,
                    }
                )
                epoch_scores.append((epoch, ic_mean, ic_std, ic_n_days))

        if epoch_scores:
            full_coverage = max(item[3] for item in epoch_scores)
            eligible_scores = [item for item in epoch_scores if item[3] == full_coverage]
            best_epoch, best_ic, best_ic_std, best_ic_n_days = max(
                eligible_scores, key=lambda item: item[1]
            )
        else:
            best_epoch, best_ic, best_ic_std = n_epochs, float("nan"), 0.0
            best_ic_n_days = 0
        config_results.append(
            {
                "config_name": config_name,
                "best_epoch": best_epoch,
                "best_ic": best_ic,
                "ic_n_days": best_ic_n_days,
                "input_chunk_length": input_chunk_length,
                "elapsed_s": elapsed_total,
                "started_at": started_at,
            }
        )
        print(
            f"  {config_name}: epoch={best_epoch}, IC={best_ic:+.4f} "
            f"(std={best_ic_std:.4f}, {elapsed_total:.1f}s)"
        )
        if checkpoint_root is not None:
            from case_studies.utils.deep_model_state import declared_epoch_checkpoints

            validate_darts_checkpoint_population(
                checkpoint_root,
                config_name=config_name,
                fold_ids=expected_fold_ids,
                checkpoints=declared_epoch_checkpoints(n_epochs, checkpoint_interval),
                architecture=str(params["architecture"]),
            )

    if not config_results:
        raise RuntimeError("Darts run produced no config results.")

    all_predictions = pl.concat(prediction_frames) if prediction_frames else pl.DataFrame()
    learning_curves = pl.DataFrame(learning_rows) if learning_rows else pl.DataFrame()
    training_log_df = pl.DataFrame(training_log) if training_log else pl.DataFrame()
    result = assemble_cv_result(
        learning_curves,
        all_predictions,
        date_col=date_col,
        entity_col=entity_col,
        metadata={row["config_name"]: row for row in config_results},
        training_log=training_log_df,
    )
    predictions = result["predictions"]

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
        from case_studies.utils.registry import register_prediction_set

        for row in config_results:
            cfg_name = row["config_name"]
            cfg = next(c for c in configs if c["config_name"] == cfg_name)
            cfg_preds = all_predictions.filter(pl.col("config") == cfg_name)
            cfg_curves = learning_curves.filter(pl.col("config") == cfg_name)
            epoch_ics = {
                int(item["epoch"]): float(item["ic_mean"])
                for item in cfg_curves.iter_rows(named=True)
            }
            epochs = sorted(cfg_preds["epoch"].unique().to_list())
            first_epoch = row["best_epoch"] if row["best_epoch"] in epochs else epochs[0]
            first_slice = cfg_preds.filter(pl.col("epoch") == first_epoch).drop("config", "epoch")
            training_hash = _register_dl_config(
                case_study=case_study,
                label=label_col,
                config_name=cfg_name,
                architecture=cfg["params"]["architecture"],
                n_epochs=cfg.get("n_epochs", 100),
                best_epoch=first_epoch,
                lookback=row["input_chunk_length"],
                n_folds=len(splits),
                ic_mean=epoch_ics.get(first_epoch, row["best_ic"]),
                predictions=first_slice,
                notebook=notebook,
                learning_curves=cfg_curves,
                started_at=row.get("started_at"),
                elapsed_s=row.get("elapsed_s"),
                prediction_split=prediction_split,
                identity_params=_config_identity_params(cfg),
            )
            for epoch in epochs:
                if epoch == first_epoch:
                    continue
                epoch_slice = cfg_preds.filter(pl.col("epoch") == epoch).drop("config", "epoch")
                register_prediction_set(
                    case_study,
                    training_hash=training_hash,
                    checkpoint_value=int(epoch),
                    checkpoint_kind="epoch",
                    split=prediction_split,
                    predictions=epoch_slice,
                    metrics={"ic_mean": epoch_ics.get(epoch, float("nan"))},
                )
            print(f"    registered {cfg_name} ({len(epochs)} per-epoch slices)")

    return result
