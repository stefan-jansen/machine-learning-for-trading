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
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import uuid
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from case_studies.research.models import ModelRun
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry import clear_prediction_sets, compute_fold_metrics_from_predictions

if TYPE_CHECKING:
    from case_studies.research.workspace import Study

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from utils.modeling import RANDOM_SEED, seed_everything

_TABM_PREVIEW_FIELDS = {"checkpoint_interval", "folds", "max_symbols", "n_epochs"}
_TABM_IMBALANCE_METHODS = {"balanced", "none"}


@dataclass(frozen=True)
class TabMResearchContext:
    dataset_pd: pd.DataFrame
    splits: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    feature_names: tuple[str, ...]
    label_col: str
    eval_label_col: str | None
    date_col: str
    entity_col: str
    task_type: str
    class_values: tuple[Any, ...]
    class_weights_by_fold: dict[int, tuple[float, ...]]
    temporal_by_fold: pd.DataFrame | None
    temporal_keys: tuple[str, ...]
    temporal_feature_names: tuple[str, ...]
    expected_keys: pl.DataFrame
    runtime_provenance: dict[str, Any]
    prediction_split: str = "validation"
    published_checkpoints: tuple[int, ...] | None = None
    immutable_recovery: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tabm_source_identity() -> dict[str, str]:
    from case_studies.utils import deep_model_state

    deep_model_state_file = deep_model_state.__file__
    if deep_model_state_file is None:
        raise RuntimeError("deep_model_state has no source file")
    deep_model_state_path = Path(deep_model_state_file)
    return {
        Path(__file__).name: _sha256(Path(__file__)),
        deep_model_state_path.name: _sha256(deep_model_state_path),
    }


def _tabm_runtime_identity() -> dict[str, str]:
    return {
        "numpy": importlib.metadata.version("numpy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "torch": importlib.metadata.version("torch"),
    }


def _tabm_runtime_provenance(study: Study) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(study.release_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "entry_point": "case_studies.utils.tabular_dl",
        "packages": _tabm_runtime_identity(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_commit": commit,
    }


def _normalize_splits(splits: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    fields = ("fold", "train_start", "train_end", "val_start", "val_end")
    return tuple(
        {
            key: int(split[key]) if key == "fold" else str(split[key])
            for key in fields
            if split.get(key) is not None
        }
        for split in splits
    )


def _tabm_splits(mds, request: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cv = request.get("cv")
    if cv is None:
        splits = [dict(split) for split in mds.splits]
        normalized = _normalize_splits(splits)
        cv_record = {
            "request": {"source": "case_study_default"},
            "folds": list(normalized),
            "identity": value_digest(pl.DataFrame(list(normalized))),
        }
    else:
        resolved = cv.resolve(mds.dataset.select(mds.date_col).unique(), date_col=mds.date_col)
        splits = [dict(fold) for fold in resolved.normalized_folds]
        cv_record = resolved.as_dict()
    requested_folds = request["preview_reductions"].get("folds")
    if requested_folds is not None:
        selected = {int(fold) for fold in requested_folds}
        splits = [split for split in splits if int(split["fold"]) in selected]
        if {int(split["fold"]) for split in splits} != selected:
            raise ValueError("preview fold reduction refers to an unavailable fold")
        cv_record = {**cv_record, "preview_folds": sorted(selected)}
    if not splits:
        raise ValueError("TabM request resolved no cross-validation folds")
    from utils.modeling import validate_temporal_split_geometry

    validate_temporal_split_geometry(splits, mds.splits, mds.temporal_by_fold)
    return splits, cv_record


def _resolve_tabm_config(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    resolved = {**config, "params": dict(config.get("params") or {})}
    top_level = {"batch_size", "checkpoint_interval", "n_epochs"}
    unknown = (
        set(overrides)
        - top_level
        - set(resolved["params"])
        - {"class_weight", "device", "num_threads"}
    )
    if unknown:
        raise ValueError(f"unsupported TabM overrides: {sorted(unknown)}")
    for key, value in overrides.items():
        if key in top_level:
            resolved[key] = value
        elif key not in {"class_weight", "device", "num_threads"}:
            resolved["params"][key] = value
    if resolved.get("library") != "tabm" or str(resolved["config_name"]).startswith("tabpfn"):
        raise ValueError("the TabM research adapter requires a persisted TabM configuration")
    _tabm_checkpoint_epochs(resolved)
    return resolved


def _balanced_class_weights(values: np.ndarray, class_values: tuple[Any, ...]) -> tuple[float, ...]:
    """Return sklearn-style balanced weights in the declared class order."""
    if not class_values:
        return ()
    counts = np.asarray([(values == value).sum() for value in class_values], dtype=np.float64)
    if np.any(counts == 0):
        missing = [value for value, count in zip(class_values, counts, strict=True) if count == 0]
        raise ValueError(f"classification fold is missing declared classes: {missing}")
    return tuple((len(values) / (len(class_values) * counts)).tolist())


def _tabm_class_weights_by_fold(
    mds,
    splits: list[dict[str, Any]],
    *,
    method: str,
) -> dict[int, tuple[float, ...]]:
    if mds.task_type != "classification":
        return {}
    if method not in _TABM_IMBALANCE_METHODS:
        raise ValueError(
            f"unsupported TabM class_weight {method!r}; expected {sorted(_TABM_IMBALANCE_METHODS)}"
        )
    class_values = tuple(mds.class_values)
    if not class_values:
        raise ValueError("classification requires declared class values")
    weights = {}
    date_dtype = mds.dataset.schema[mds.date_col]
    for split in splits:
        labels = (
            mds.dataset.filter(
                pl.col(mds.date_col).is_between(
                    pl.lit(split["train_start"]).cast(date_dtype, strict=False),
                    pl.lit(split["train_end"]).cast(date_dtype, strict=False),
                    closed="both",
                )
            )
            .drop_nulls(mds.label_col)
            .get_column(mds.label_col)
            .to_numpy()
        )
        weights[int(split["fold"])] = (
            _balanced_class_weights(labels, class_values)
            if method == "balanced"
            else tuple(1.0 for _ in class_values)
        )
    return weights


def _tabm_expected_keys(mds, splits: list[dict[str, Any]]) -> pl.DataFrame:
    entity_col = mds.entity_cols[0]
    frames = []
    for split in splits:
        date_dtype = mds.dataset.schema[mds.date_col]
        frame = mds.dataset.filter(
            pl.col(mds.date_col).is_between(
                pl.lit(split["val_start"]).cast(date_dtype, strict=False),
                pl.lit(split["val_end"]).cast(date_dtype, strict=False),
                closed="both",
            )
        ).drop_nulls([mds.label_col, *([mds.eval_label_col] if mds.eval_label_col else [])])
        frames.append(
            frame.select(
                pl.col(entity_col).alias("symbol"),
                pl.col(mds.date_col).alias("timestamp"),
            ).with_columns(pl.lit(int(split["fold"]), dtype=pl.Int64).alias("fold"))
        )
    expected = pl.concat(frames).sort("symbol", "timestamp", "fold")
    if expected.n_unique(["symbol", "timestamp", "fold"]) != expected.height:
        raise ValueError("TabM request produced duplicate expected prediction keys")
    return expected


def _resolve_model_request_from_materialized(
    study: Study,
    request: dict[str, Any],
    *,
    label_ref,
    mds,
    dataset_pd: pd.DataFrame | None,
    configured_by_name: dict[str, dict[str, Any]],
    runtime_provenance: dict[str, Any],
):
    from case_studies.research.contracts import ExecutionTier
    from case_studies.research.identity import ResolvedSpec

    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    unknown_reductions = set(reductions) - _TABM_PREVIEW_FIELDS
    if unknown_reductions:
        raise ValueError(f"unsupported TabM preview reductions: {sorted(unknown_reductions)}")
    if mds.date_col != "timestamp" or not mds.entity_cols:
        raise ValueError("TabM runner requires timestamp and an entity key")
    entity_col = mds.entity_cols[0]
    if entity_col not in {"product", "symbol"}:
        raise ValueError(f"TabM runner does not support entity key {entity_col!r}")
    splits, cv_record = _tabm_splits(mds, request)
    try:
        configured = configured_by_name[request["config_name"]]
    except KeyError as error:
        raise ValueError(f"unknown TabM configuration {request['config_name']!r}") from error
    config = _resolve_tabm_config(configured, request["overrides"])
    if tier is ExecutionTier.PREVIEW:
        for field in ("checkpoint_interval", "n_epochs"):
            if field in reductions:
                config[field] = int(reductions[field])
        _tabm_checkpoint_epochs(config)
    device = str(request["overrides"].get("device", "cuda"))
    num_threads = int(request["overrides"].get("num_threads", 8))
    runtime = tabm_runtime_spec(device, num_threads=num_threads)
    expected = _tabm_expected_keys(mds, splits)
    input_lineage = mds.input_lineage
    checkpoints = _tabm_checkpoint_epochs(config)
    class_weight_method = str(request["overrides"].get("class_weight", "balanced"))
    class_weights_by_fold = _tabm_class_weights_by_fold(
        mds,
        splits,
        method=class_weight_method,
    )
    task = {
        "type": mds.task_type,
        "class_values": list(mds.class_values),
        "continuous_eval_label": label_ref.definition.continuous_eval_label,
    }
    if mds.task_type == "classification":
        metrics = ["ic", "accuracy", "balanced_accuracy"]
        if len(mds.class_values) == 2:
            metrics[1:1] = ["auc_roc", "log_loss"]
        task.update(
            {
                "metrics": metrics,
                "imbalance": {
                    "method": class_weight_method,
                    "effective_class_weights_by_fold": {
                        str(fold): list(weights)
                        for fold, weights in sorted(class_weights_by_fold.items())
                    },
                },
            }
        )
    computation = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "task": task,
        "cv": cv_record,
        "model": {
            "class": "TabMModel",
            "implementation": "pytorch",
            "objective": "classification" if mds.task_type == "classification" else "regression",
            "params": {
                **config["params"],
                "batch_size": int(config["batch_size"]),
                "checkpoint_interval": int(config["checkpoint_interval"]),
                "n_epochs": int(config["n_epochs"]),
            },
        },
        "preprocessing": {
            "imputer": {"class": "SimpleImputer", "strategy": "median"},
            "scaler": {"class": "StandardScaler", "with_mean": True, "with_std": True},
        },
        "checkpoint_schedule": [
            {"kind": "epoch", "value": checkpoint} for checkpoint in checkpoints
        ],
        "expected_prediction_keys": {
            "digest": value_digest(expected, ("symbol", "timestamp", "fold")),
            "n_rows": expected.height,
            "n_folds": expected["fold"].n_unique(),
        },
        "input_data_spec": input_lineage,
        "sampling": {"max_symbols": int(reductions.get("max_symbols", 0))},
        "numerics": runtime,
        "source_identity": _tabm_source_identity(),
        "runtime_identity": _tabm_runtime_identity(),
    }
    if tier is ExecutionTier.PREVIEW:
        computation["preview_reductions"] = reductions
    spec = ResolvedSpec.create(
        family="tabular_dl",
        label=label_ref.name,
        seed=int(runtime["seed"]),
        computation=computation,
        provenance=runtime_provenance,
        config_name=config["config_name"],
        execution_tier=tier.value,
    ).as_dict()
    context = TabMResearchContext(
        dataset_pd=mds.dataset.to_pandas() if dataset_pd is None else dataset_pd,
        splits=tuple(splits),
        config=config,
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        eval_label_col=mds.eval_label_col,
        date_col=mds.date_col,
        entity_col=entity_col,
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        class_weights_by_fold=class_weights_by_fold,
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=tuple(mds.temporal_keys),
        temporal_feature_names=tuple(mds.temporal_feature_names),
        expected_keys=expected,
        runtime_provenance=runtime_provenance,
    )
    return spec, context


def _materialize_tabm_request_group(study: Study, request: dict[str, Any]):
    from case_studies.research.contracts import ExecutionTier
    from utils.modeling import load_configs, load_modeling_dataset

    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    unknown_reductions = set(reductions) - _TABM_PREVIEW_FIELDS
    if unknown_reductions:
        raise ValueError(f"unsupported TabM preview reductions: {sorted(unknown_reductions)}")
    study.require_writable()
    study.activate(tier)
    label_ref = study.labels.get(request["label"], execution_tier=tier)
    mds = load_modeling_dataset(
        study.case_study,
        label_ref.name,
        max_symbols=int(reductions.get("max_symbols", 0)),
    )
    configured_by_name = {
        config["config_name"]: config
        for config in load_configs(study.case_study, label_ref.name, "tabular_dl")
    }
    return (
        label_ref,
        mds,
        configured_by_name,
        _tabm_runtime_provenance(study),
    )


def resolve_model_request(study: Study, request: dict[str, Any]):
    materialized = _materialize_tabm_request_group(study, request)
    return _resolve_model_request_from_materialized(
        study,
        request,
        label_ref=materialized[0],
        mds=materialized[1],
        dataset_pd=None,
        configured_by_name=materialized[2],
        runtime_provenance=materialized[3],
    )


def reconstruct_locked_request(
    study: Study,
    spec: dict[str, Any],
    *,
    checkpoint_kind: str,
    checkpoint_value: int | None,
):
    """Reconstruct a TabM holdout fit without consulting a mutable preset."""
    from case_studies.research.contracts import ExecutionTier
    from case_studies.research.cv import require_fold_scoped_temporal_compatibility
    from case_studies.research.models import (
        ResolvedModelRequest,
        locked_holdout_split,
        validate_locked_expected_keys,
    )
    from case_studies.utils.registry import training_hash_from_spec
    from utils.modeling import load_modeling_dataset

    if checkpoint_kind != "epoch" or checkpoint_value is None:
        raise ValueError("TabM holdout requires one locked epoch checkpoint")
    study.require_writable()
    study.activate(ExecutionTier.CANONICAL)
    computation = spec["computation"]
    if computation.get("sampling") != {"max_symbols": 0}:
        raise ValueError("locked TabM holdout requires an unreduced canonical dataset")
    label_ref = study.labels.get(spec["label"], execution_tier=ExecutionTier.CANONICAL)
    mds = load_modeling_dataset(study.case_study, label_ref.name, max_symbols=0)
    if mds.date_col != "timestamp" or mds.entity_cols[:1] not in (["symbol"], ["product"]):
        raise ValueError("locked TabM runner requires canonical entity and timestamp keys")
    expected_inputs = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": mds.input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "input_data_spec": mds.input_lineage,
        "source_identity": _tabm_source_identity(),
        "runtime_identity": _tabm_runtime_identity(),
        "preprocessing": {
            "imputer": {"class": "SimpleImputer", "strategy": "median"},
            "scaler": {"class": "StandardScaler", "with_mean": True, "with_std": True},
        },
    }
    for name, expected_value in expected_inputs.items():
        if computation.get(name) != expected_value:
            raise ValueError(f"locked TabM {name} does not match the available computation")
    split = locked_holdout_split(spec, mds.dataset, mds.date_col, study.case_study)
    if mds.temporal_by_fold is not None and mds.temporal_keys and mds.temporal_feature_names:
        require_fold_scoped_temporal_compatibility([split], mds.temporal_artifact_splits)
    expected = _tabm_expected_keys(mds, [split])
    validate_locked_expected_keys(spec, expected)

    model = computation.get("model")
    if (
        not isinstance(model, dict)
        or model.get("class") != "TabMModel"
        or model.get("implementation") != "pytorch"
        or not isinstance(model.get("params"), dict)
    ):
        raise ValueError("locked TabM model specification is unsupported")
    params = dict(model["params"])
    required_training = {
        name: params.pop(name, None)
        for name in (
            "batch_size",
            "checkpoint_interval",
            "n_epochs",
        )
    }
    if any(value is None for value in required_training.values()):
        raise ValueError("locked TabM model omits exact training parameters")
    config = _resolve_tabm_config(
        {
            "family": "tabular_dl",
            "library": "tabm",
            "config_name": str(
                spec.get("config_name") or f"locked-{training_hash_from_spec(spec)}"
            ),
            "params": params,
            **required_training,
        },
        {},
    )
    checkpoints = _tabm_checkpoint_epochs(config)
    schedule = computation.get("checkpoint_schedule")
    expected_schedule = [{"kind": "epoch", "value": value} for value in checkpoints]
    if schedule != expected_schedule or checkpoint_value not in checkpoints:
        raise ValueError("locked TabM checkpoint is absent from its exact schedule")
    task = computation.get("task")
    class_weight_method = "balanced"
    if mds.task_type == "classification":
        if not isinstance(task, dict) or not isinstance(task.get("imbalance"), dict):
            raise ValueError("locked classification TabM task omits imbalance behavior")
        class_weight_method = str(task["imbalance"].get("method"))
    weights = _tabm_class_weights_by_fold(mds, [split], method=class_weight_method)
    expected_task = {
        "type": mds.task_type,
        "class_values": list(mds.class_values),
        "continuous_eval_label": label_ref.definition.continuous_eval_label,
    }
    if mds.task_type == "classification":
        metrics = ["ic", "accuracy", "balanced_accuracy"]
        if len(mds.class_values) == 2:
            metrics[1:1] = ["auc_roc", "log_loss"]
        expected_task.update(
            {
                "metrics": metrics,
                "imbalance": {
                    "method": class_weight_method,
                    "effective_class_weights_by_fold": {
                        str(fold): list(values) for fold, values in sorted(weights.items())
                    },
                },
            }
        )
    if task != expected_task:
        raise ValueError("locked TabM task behavior does not reproduce exactly")
    numerics = computation.get("numerics")
    if not isinstance(numerics, dict):
        raise ValueError("locked TabM numerics are missing")
    reproduced_numerics = tabm_runtime_spec(
        str(numerics.get("device")),
        seed=int(numerics.get("seed")),
        num_threads=int(numerics.get("num_threads")),
    )
    if numerics != reproduced_numerics or int(spec["seed"]) != int(numerics["seed"]):
        raise ValueError("locked TabM numerics cannot be reproduced")
    context = TabMResearchContext(
        dataset_pd=mds.dataset.to_pandas(),
        splits=(split,),
        config=config,
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        eval_label_col=mds.eval_label_col,
        date_col=mds.date_col,
        entity_col=mds.entity_cols[0],
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        class_weights_by_fold=weights,
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=tuple(mds.temporal_keys),
        temporal_feature_names=tuple(mds.temporal_feature_names),
        expected_keys=expected,
        runtime_provenance=_tabm_runtime_provenance(study),
        prediction_split="holdout",
        published_checkpoints=(int(checkpoint_value),),
        immutable_recovery=True,
    )
    return ResolvedModelRequest(study, "tabular_dl", spec, context)


def _cached_research_run(study: Study, spec: dict[str, Any], context: TabMResearchContext):
    from case_studies.research.models import ModelRun
    from case_studies.research.results import PredictionResult, Result, TrainingResult
    from case_studies.utils.deep_model_state import validate_deep_checkpoint_population
    from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec

    training_hash = training_hash_from_spec(spec)
    computation = spec.get("computation", spec)
    checkpoint_values = tuple(int(item["value"]) for item in computation["checkpoint_schedule"])
    published = context.published_checkpoints or checkpoint_values
    try:
        training = Result.open(
            study, training_hash, include_preview=spec["execution_tier"] == "preview"
        )
        predictions = tuple(
            Result.open(
                study,
                prediction_hash_from_parts(
                    training_hash,
                    checkpoint,
                    context.prediction_split,
                    checkpoint_kind="epoch",
                    identity_version=spec["identity_version"],
                ),
                include_preview=spec["execution_tier"] == "preview",
            )
            for checkpoint in published
        )
    except KeyError:
        return None
    if not isinstance(training, TrainingResult) or any(
        not isinstance(result, PredictionResult) or not result.complete for result in predictions
    ):
        return None
    complete_predictions = tuple(
        result for result in predictions if isinstance(result, PredictionResult)
    )
    model_root = training.root / "run_log" / "training" / training.hash / "models"
    try:
        validate_deep_checkpoint_population(
            model_root,
            config_name=training.hash,
            fold_ids=tuple(int(split["fold"]) for split in context.splits),
            checkpoints=checkpoint_values,
            architecture="tabm",
        )
    except ValueError:
        return None
    diagnostics = training.root / "run_log" / "training" / training.hash / "diagnostics"
    required = {
        "all_predictions.parquet",
        "learning_curves.parquet",
        "predictions.parquet",
        "result.json",
        "training_log.parquet",
    }
    if not diagnostics.is_dir() or required - {path.name for path in diagnostics.iterdir()}:
        return None
    try:
        for name in required - {"result.json"}:
            pl.read_parquet(diagnostics / name)
        selected = pl.read_parquet(diagnostics / "predictions.parquet")
        if "model_id" not in selected.columns or {"config", "epoch"} & set(selected.columns):
            return None
        json.loads((diagnostics / "result.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError, pl.exceptions.PolarsError):
        return None
    return ModelRun(training=training, predictions=complete_predictions)


def _publish_tabm_predictions(
    study: Study,
    spec: dict[str, Any],
    context: TabMResearchContext,
    training,
    result: dict[str, Any],
    *,
    candidate_key: str | None = None,
):
    computation = spec.get("computation", spec)
    result_key = candidate_key or context.config["config_name"]
    prediction_results = []
    published = context.published_checkpoints or tuple(
        int(item["value"]) for item in computation["checkpoint_schedule"]
    )
    for checkpoint in published:
        predictions = (
            result["all_predictions"]
            .filter((pl.col("config") == result_key) & (pl.col("epoch") == checkpoint))
            .drop("config", "epoch")
            .rename({"fold_id": "fold", "y_true": "actual", "y_score": "prediction"})
        )
        # The expected keys are the internal contract and always name the entity
        # `symbol`; the runner emits the reader-facing key the case study uses.
        if context.entity_col != "symbol":
            predictions = predictions.rename({context.entity_col: "symbol"})
        predictions = predictions.with_columns(
            pl.col(context.date_col).cast(context.expected_keys.schema[context.date_col]),
            pl.col("symbol").cast(context.expected_keys.schema["symbol"]),
            pl.col("fold").cast(context.expected_keys.schema["fold"]),
        )
        prediction_results.append(
            study.results.publish_predictions(
                training,
                checkpoint_kind="epoch",
                checkpoint_value=int(checkpoint),
                split=context.prediction_split,
                predictions=predictions,
                expected_keys=context.expected_keys,
                task_type=context.task_type,
                class_values=list(context.class_values) or None,
                eval_col="eval_actual" if context.eval_label_col else None,
                label=context.label_col,
            )
        )
    return tuple(prediction_results)


def _record_tabm_runtime(train_dir: Path, result: dict[str, Any], config_name: str) -> None:
    runtime_path = train_dir / "runtime.json"
    if not runtime_path.exists():
        return
    runtime = json.loads(runtime_path.read_text())
    runtime["elapsed_s"] = sum(
        float(row.get("elapsed_s", 0.0))
        for row in result["grid_results"]
        if row["config_name"] == config_name
    )
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")


def _persist_tabm_diagnostics(train_dir: Path, result: dict[str, Any], candidate_key: str) -> None:
    diagnostics_dir = train_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    predictions = result["all_predictions"].filter(pl.col("config") == candidate_key)
    curves = result["all_learning_curves"]
    if "config" in curves.columns:
        curves = curves.filter(pl.col("config") == candidate_key)
    training_log = result["training_log"]
    if "config" in training_log.columns:
        training_log = training_log.filter(pl.col("config") == candidate_key)
    grid_row = next(row for row in result["grid_results"] if row["config_name"] == candidate_key)
    best = (
        predictions.filter(pl.col("epoch") == int(grid_row["best_epoch"]))
        .with_columns(pl.lit(candidate_key).alias("model_id"))
        .drop("config", "epoch")
    )
    predictions.write_parquet(diagnostics_dir / "all_predictions.parquet")
    best.write_parquet(diagnostics_dir / "predictions.parquet")
    curves.write_parquet(diagnostics_dir / "learning_curves.parquet")
    training_log.write_parquet(diagnostics_dir / "training_log.parquet")
    (diagnostics_dir / "result.json").write_text(
        json.dumps(grid_row, indent=2, sort_keys=True) + "\n"
    )


@dataclass
class _TabMRecoveryCandidate:
    spec: dict[str, Any]
    context: TabMResearchContext
    training: Any
    ledger: Any
    attempt: Any
    reused_folds: list[int]
    fitted_folds: list[int]


class _TabMRecovery:
    def __init__(self, candidates: dict[str, _TabMRecoveryCandidate]) -> None:
        self.candidates = candidates

    def model_root(self, candidate_key: str) -> Path:
        candidate = self.candidates[candidate_key]
        return candidate.training.root / "run_log" / "training" / candidate.training.hash / "models"

    def _paths(self, candidate_key: str, fold_id: int) -> tuple[Path, Path]:
        candidate = self.candidates[candidate_key]
        manifest = (
            self.model_root(candidate_key) / candidate_key / f"fold_{fold_id:02d}" / "manifest.json"
        )
        shard = (
            candidate.training.root
            / "run_log"
            / "training"
            / candidate.training.hash
            / "prediction_folds"
            / f"fold_{fold_id}.parquet"
        )
        return manifest, shard

    def _diagnostic_path(self, candidate_key: str, fold_id: int) -> Path:
        _, shard = self._paths(candidate_key, fold_id)
        return shard.with_suffix(".json")

    def _settings(self, candidate_key: str, fold_id: int) -> dict[str, Any]:
        candidate = self.candidates[candidate_key]
        split = next(split for split in candidate.context.splits if int(split["fold"]) == fold_id)
        computation = candidate.spec.get("computation", candidate.spec)
        return {
            "checkpoint_schedule": computation["checkpoint_schedule"],
            "fold": _normalize_splits([split])[0],
            "training_hash": candidate.training.hash,
        }

    def _checkpoint_files(self, candidate_key: str, fold_id: int) -> tuple[Path, ...]:
        from case_studies.utils.deep_model_state import checkpoint_sidecar, deep_checkpoint_path

        candidate = self.candidates[candidate_key]
        computation = candidate.spec.get("computation", candidate.spec)
        checkpoints = tuple(int(item["value"]) for item in computation["checkpoint_schedule"])
        files = []
        for checkpoint in checkpoints:
            path = deep_checkpoint_path(
                self.model_root(candidate_key), candidate_key, fold_id, checkpoint
            )
            files.extend((path, checkpoint_sidecar(path)))
        return tuple(files)

    def _valid_manifest(self, candidate_key: str, fold_id: int, manifest: Path) -> bool:
        if not manifest.is_file():
            return False
        try:
            record = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        files = self._checkpoint_files(candidate_key, fold_id)
        expected = {
            str(path.relative_to(self.model_root(candidate_key))): _sha256(path)
            for path in files
            if path.is_file()
        }
        return len(expected) == len(files) and record == {
            "files": expected,
            "schema_version": 1,
        }

    def _quarantine(self, candidate_key: str, fold_id: int) -> None:
        candidate = self.candidates[candidate_key]
        manifest, shard = self._paths(candidate_key, fold_id)
        diagnostic = self._diagnostic_path(candidate_key, fold_id)
        fold_dir = manifest.parent
        if not fold_dir.exists() and not shard.exists():
            return
        train_dir = candidate.training.root / "run_log" / "training" / candidate.training.hash
        quarantine = train_dir / "invalid_folds" / f"fold_{fold_id}.{uuid.uuid4().hex}"
        quarantine.mkdir(parents=True)
        if fold_dir.exists():
            os.replace(fold_dir, quarantine / "models")
        if shard.exists():
            os.replace(shard, quarantine / shard.name)
        if diagnostic.exists():
            os.replace(diagnostic, quarantine / diagnostic.name)

    def _reject_or_quarantine(
        self,
        candidate_key: str,
        fold_id: int,
        reason: str,
    ) -> None:
        candidate = self.candidates[candidate_key]
        if candidate.context.immutable_recovery:
            raise ValueError(f"locked TabM fold {fold_id} {reason}")
        self._quarantine(candidate_key, fold_id)

    def reuse(self, candidate_key: str, fold_id: int) -> tuple[pl.DataFrame, dict[str, Any]] | None:
        candidate = self.candidates[candidate_key]
        manifest, shard = self._paths(candidate_key, fold_id)
        diagnostic = self._diagnostic_path(candidate_key, fold_id)
        settings = self._settings(candidate_key, fold_id)
        reusable = candidate.ledger.reusable_fold(
            training_hash=candidate.training.hash,
            candidate_identity=candidate.training.hash,
            fold_id=fold_id,
            fitted_state=manifest,
            prediction_shard=shard,
            resolved_settings=settings,
        )
        completed = candidate.ledger.fold_completion_exists(
            training_hash=candidate.training.hash,
            candidate_identity=candidate.training.hash,
            fold_id=fold_id,
        )
        complete_population = all(
            path.is_file()
            for path in (
                manifest,
                shard,
                diagnostic,
                *self._checkpoint_files(candidate_key, fold_id),
            )
        )
        valid_files = (
            self._valid_manifest(candidate_key, fold_id, manifest) and diagnostic.is_file()
        )
        recover_uncommitted = not reusable and not completed and valid_files and shard.is_file()
        if reusable and not valid_files:
            self._reject_or_quarantine(
                candidate_key,
                fold_id,
                "has fitted artifacts that disagree with its manifest",
            )
            return None
        if not reusable and not recover_uncommitted:
            if candidate.context.immutable_recovery and (completed or complete_population):
                raise ValueError(f"locked TabM fold {fold_id} has conflicting persisted artifacts")
            self._quarantine(candidate_key, fold_id)
            return None
        frame = pl.read_parquet(shard)
        checkpoints = {
            int(item["value"])
            for item in candidate.spec.get("computation", candidate.spec)["checkpoint_schedule"]
        }
        required = {
            candidate.context.date_col,
            candidate.context.entity_col,
            "config",
            "epoch",
            "fold_id",
            "y_score",
            "y_true",
        }
        if required - set(frame.columns):
            self._reject_or_quarantine(candidate_key, fold_id, "has an invalid prediction schema")
            return None
        if (
            set(frame["config"].unique().to_list()) != {candidate_key}
            or set(frame["fold_id"].unique().to_list()) != {fold_id}
            or {int(value) for value in frame["epoch"].unique().to_list()} != checkpoints
        ):
            self._reject_or_quarantine(
                candidate_key,
                fold_id,
                "has conflicting prediction identities",
            )
            return None
        keys = [candidate.context.date_col, candidate.context.entity_col, "fold_id", "epoch"]
        if frame.n_unique(keys) != frame.height:
            self._reject_or_quarantine(candidate_key, fold_id, "has duplicate prediction keys")
            return None
        try:
            training_record = json.loads(diagnostic.read_text())
        except (OSError, json.JSONDecodeError):
            self._reject_or_quarantine(candidate_key, fold_id, "has invalid diagnostics")
            return None
        if recover_uncommitted:
            candidate.ledger.complete_fold(
                training_hash=candidate.training.hash,
                candidate_identity=candidate.training.hash,
                fold_id=fold_id,
                fitted_state=manifest,
                prediction_shard=shard,
                resolved_settings=settings,
            )
        candidate.reused_folds.append(fold_id)
        return frame, training_record

    def complete(
        self,
        candidate_key: str,
        fold_id: int,
        frame: pl.DataFrame,
        training_record: dict[str, Any],
    ) -> None:
        candidate = self.candidates[candidate_key]
        manifest, shard = self._paths(candidate_key, fold_id)
        files = self._checkpoint_files(candidate_key, fold_id)
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise ValueError(f"TabM fold checkpoint population is incomplete: {missing}")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_tmp = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.tmp")
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard_tmp = shard.with_name(f".{shard.name}.{uuid.uuid4().hex}.tmp")
        diagnostic = self._diagnostic_path(candidate_key, fold_id)
        diagnostic_tmp = diagnostic.with_name(f".{diagnostic.name}.{uuid.uuid4().hex}.tmp")
        try:
            record = {
                "files": {
                    str(path.relative_to(self.model_root(candidate_key))): _sha256(path)
                    for path in files
                },
                "schema_version": 1,
            }
            manifest_tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            frame.write_parquet(shard_tmp)
            diagnostic_tmp.write_text(json.dumps(training_record, indent=2, sort_keys=True) + "\n")
            os.replace(manifest_tmp, manifest)
            os.replace(shard_tmp, shard)
            os.replace(diagnostic_tmp, diagnostic)
        finally:
            manifest_tmp.unlink(missing_ok=True)
            shard_tmp.unlink(missing_ok=True)
            diagnostic_tmp.unlink(missing_ok=True)
        candidate.ledger.complete_fold(
            training_hash=candidate.training.hash,
            candidate_identity=candidate.training.hash,
            fold_id=fold_id,
            fitted_state=manifest,
            prediction_shard=shard,
            resolved_settings=self._settings(candidate_key, fold_id),
        )
        candidate.fitted_folds.append(fold_id)


def run_resolved_request(study: Study, spec: dict[str, Any], context: TabMResearchContext):
    return _run_tabm_compatible_group(study, [(0, spec, context)])[0]


def _reconstruct_locked_tabm_predictions(
    model_root: Path,
    training_hash: str,
    context: TabMResearchContext,
    checkpoint: int,
    device: torch.device,
) -> pl.DataFrame:
    from case_studies.utils.deep_model_state import deep_checkpoint_path, restore_deep_model

    frames = []
    model_params = dict(context.config["params"])
    output_dim = len(context.class_values) if context.task_type == "classification" else 1
    expected_kwargs = {
        "n_features": len(context.feature_names),
        "output_dim": output_dim,
        **model_params,
    }

    def factory(architecture: str, model_kwargs: Mapping[str, Any]) -> nn.Module:
        if architecture != "tabm" or dict(model_kwargs) != expected_kwargs:
            raise ValueError("locked TabM checkpoint architecture changed")
        return TabMModel(**dict(model_kwargs))

    for split in context.splits:
        prepared = _prepare_tabm_fold(
            context.dataset_pd,
            split,
            feature_names=list(context.feature_names),
            label_col=context.label_col,
            eval_label_col=context.eval_label_col,
            date_col=context.date_col,
            entity_col=context.entity_col,
            temporal_by_fold=context.temporal_by_fold,
            temporal_keys=list(context.temporal_keys),
            temporal_feature_names=list(context.temporal_feature_names),
        )
        fold = int(split["fold"])
        model, preprocessing, metadata = restore_deep_model(
            deep_checkpoint_path(model_root, training_hash, fold, checkpoint),
            factory,
        )
        expected_metadata = {
            "config_name": training_hash,
            "fold": fold,
            "checkpoint_kind": "epoch",
            "checkpoint_value": checkpoint,
        }
        if metadata != expected_metadata:
            raise ValueError("locked TabM checkpoint metadata changed")
        expected_preprocessing = prepared["preprocessing"]
        stored_arrays = {
            name: preprocessing.get(name)
            for name in ("imputer_statistics", "scaler_mean", "scaler_scale")
        }
        arrays_match = True
        for name, stored in stored_arrays.items():
            if stored is None or not np.array_equal(stored, expected_preprocessing[name]):
                arrays_match = False
                break
        if (
            preprocessing.get("feature_names") != expected_preprocessing["feature_names"]
            or not arrays_match
        ):
            raise ValueError("locked TabM preprocessing state changed")
        raw_prediction = _predict_in_chunks(model.to(device), prepared["X_val"], device)
        prediction = (
            _classification_scores(raw_prediction, context.class_values)
            if context.task_type == "classification"
            else raw_prediction
        )
        columns: dict[str, Any] = {
            context.date_col: prepared["val_dates"],
            "symbol": prepared["val_entities"],
            "fold": fold,
            "actual": prepared["y_val"],
            "prediction": prediction,
        }
        if context.eval_label_col:
            columns["eval_actual"] = prepared["y_eval_val"]
        frames.append(pl.DataFrame(columns))
    return pl.concat(frames).with_columns(
        pl.col(context.date_col).cast(context.expected_keys.schema[context.date_col]),
        # expected_keys names the entity `symbol` whatever the reader key is, so the
        # dtype has to be read from that column rather than from the reader name.
        pl.col("symbol").cast(context.expected_keys.schema["symbol"]),
        pl.col("fold").cast(context.expected_keys.schema["fold"]),
    )


def validate_locked_run(
    study: Study,
    spec: dict[str, Any],
    context: TabMResearchContext,
    run: ModelRun,
) -> str:
    """Validate the selected prediction and persisted TabM checkpoint population."""
    from case_studies.utils.registry import training_hash_from_spec
    from case_studies.utils.registry.specs import canonical_json

    if run.training.hash != training_hash_from_spec(spec) or len(run.predictions) != 1:
        raise ValueError("locked TabM run has the wrong training or prediction identity")
    selected = context.published_checkpoints
    prediction = run.predictions[0]
    record = prediction.registry_record()
    if (
        selected is None
        or len(selected) != 1
        or (
            record["split"],
            record["checkpoint_kind"],
            record["checkpoint_value"],
        )
        != (context.prediction_split, "epoch", selected[0])
    ):
        raise ValueError("locked TabM run published the wrong checkpoint")
    # Both sides name the entity `symbol`: publishing renames it, and the reconstruction
    # builds it that way, so they compare directly.
    published = prediction.load().sort("symbol", context.date_col, "fold")
    reopened = _cached_research_run(study, spec, context)
    if reopened is None or reopened.predictions[0].hash != prediction.hash:
        raise ValueError("locked TabM fitted state cannot be reused exactly")
    model_root = run.training.root / "run_log" / "training" / run.training.hash / "models"
    device = _configure_torch_runtime(spec["computation"]["numerics"])
    reconstructed = _reconstruct_locked_tabm_predictions(
        model_root,
        run.training.hash,
        context,
        selected[0],
        device,
    )
    reconstructed = reconstructed.sort("symbol", context.date_col, "fold")
    key_columns = ["symbol", context.date_col, "fold"]
    value_columns = ["prediction", "actual"]
    if context.eval_label_col:
        value_columns.append("eval_actual")
    if not reconstructed.select(key_columns).equals(
        published.select(key_columns)
    ) or not np.allclose(
        reconstructed.select(value_columns).to_numpy(),
        published.select(value_columns).to_numpy(),
        rtol=1e-7,
        atol=1e-7,
        equal_nan=False,
    ):
        raise ValueError("locked TabM fitted state does not reproduce published predictions")
    files = {
        str(path.relative_to(model_root)): _sha256(path)
        for path in sorted(model_root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError("locked TabM run has no fitted state")
    return hashlib.sha256(canonical_json(files).encode()).hexdigest()


def _tabm_materialization_key(request: dict[str, Any]) -> tuple[str, str, int]:
    reductions = dict(request["preview_reductions"])
    return (
        str(request["label"]),
        str(request["execution_tier"]),
        int(reductions.get("max_symbols", 0)),
    )


def _tabm_execution_key(spec: dict[str, Any]) -> str:
    from case_studies.utils.registry.specs import canonical_json

    computation = spec.get("computation", spec)
    shared = {
        "execution_tier": spec["execution_tier"],
        "cv": computation["cv"],
        "feature_artifacts": computation["feature_artifacts"],
        "feature_names": computation["feature_names"],
        "input_data_spec": computation["input_data_spec"],
        "label_artifact": computation["label_artifact"],
        "numerics": computation["numerics"],
        "preprocessing": computation["preprocessing"],
        "sampling": computation["sampling"],
        "task": computation["task"],
    }
    return canonical_json(shared)


def _run_tabm_compatible_group(study: Study, items):
    from case_studies.research.models import ModelRun
    from case_studies.research.recovery import ExecutionLedger

    compatibility_group = hashlib.sha256(_tabm_execution_key(items[0][1]).encode()).hexdigest()[:12]
    completed: dict[int, ModelRun] = {}
    cached_indices = []
    pending = []
    for index, spec, context in items:
        cached = _cached_research_run(study, spec, context)
        if cached is not None:
            completed[index] = ModelRun(
                training=cached.training,
                predictions=cached.predictions,
                diagnostics={
                    "execution_order": "fold_major",
                    "compatibility_group": compatibility_group,
                    "compatibility_group_size": len(items),
                    "group_size": len(items),
                    "reused": True,
                    "base_fold_preparations": 0,
                    "base_fold_preparation_s": 0.0,
                    "candidate_fit_s": 0.0,
                    "preparation_fraction": 0.0,
                    "disk_fold_cache": False,
                },
            )
            cached_indices.append(index)
        else:
            pending.append((index, spec, context))
    if not pending:
        return completed

    registered = []
    recovery_candidates = {}
    primary_index_by_key = {}
    duplicate_indices: dict[str, list[int]] = {}
    for index, spec, context in pending:
        training = study.results.register_training(
            spec,
            execution_tier=spec["execution_tier"],
            runtime_provenance=context.runtime_provenance,
        )
        candidate_key = training.hash
        if candidate_key in recovery_candidates:
            duplicate_indices.setdefault(candidate_key, []).append(index)
            continue
        train_dir = training.root / "run_log" / "training" / training.hash
        ledger = ExecutionLedger(study, training.root)
        candidate = _TabMRecoveryCandidate(
            spec=spec,
            context=context,
            training=training,
            ledger=ledger,
            attempt=ledger.start(training.hash),
            reused_folds=[],
            fitted_folds=[],
        )
        recovery_candidates[candidate_key] = candidate
        primary_index_by_key[candidate_key] = index
        registered.append((index, spec, context, training, train_dir, candidate_key, candidate))

    first_context = pending[0][2]
    first_computation = pending[0][1].get("computation", pending[0][1])
    recovery = _TabMRecovery(recovery_candidates)
    try:
        result = run_tabm_cv(
            first_context.dataset_pd,
            list(first_context.splits),
            configs=[
                {
                    **context.config,
                    "_artifact_name": candidate_key,
                    "_execution_key": candidate_key,
                }
                for _, _, context, _, _, candidate_key, _ in registered
            ],
            n_features=len(first_context.feature_names),
            feature_names=list(first_context.feature_names),
            label_col=first_context.label_col,
            eval_label_col=first_context.eval_label_col,
            task_type=first_context.task_type,
            class_values=list(first_context.class_values) or None,
            date_col=first_context.date_col,
            entity_col=first_context.entity_col,
            device=str(first_computation["numerics"]["device"]),
            save_dir=None,
            register=False,
            temporal_by_fold=first_context.temporal_by_fold,
            temporal_keys=list(first_context.temporal_keys),
            temporal_feature_names=list(first_context.temporal_feature_names),
            class_weights_by_fold=first_context.class_weights_by_fold,
            seed=int(pending[0][1]["seed"]),
            num_threads=int(first_computation["numerics"]["num_threads"]),
            checkpoint_root=None,
            strict=True,
            _recovery=recovery,
        )
        execution_diagnostics = result["execution_diagnostics"]
        preparation_elapsed_s = float(execution_diagnostics["base_fold_preparation_s"])
        fit_elapsed_by_candidate = execution_diagnostics["candidate_fit_s"]
        measured_s = preparation_elapsed_s + sum(
            float(value) for value in fit_elapsed_by_candidate.values()
        )
        group_measurements = {
            "base_fold_preparations": int(execution_diagnostics["base_fold_preparations"]),
            "base_fold_preparation_s": preparation_elapsed_s,
            "preparation_fraction": (preparation_elapsed_s / measured_s if measured_s else 0.0),
        }
        for index in cached_indices:
            completed[index].diagnostics.update(group_measurements)
        for index, spec, context, training, train_dir, candidate_key, candidate in registered:
            failure = result.get("failures", {}).get(candidate_key)
            if failure is not None:
                candidate.attempt.finish(
                    "failed",
                    {
                        "error": str(failure),
                        "error_type": type(failure).__name__,
                        "fitted_folds": candidate.fitted_folds,
                        "reused_folds": candidate.reused_folds,
                    },
                )
                candidate.attempt = None
                continue
            _record_tabm_runtime(train_dir, result, candidate_key)
            _persist_tabm_diagnostics(train_dir, result, candidate_key)
            predictions = _publish_tabm_predictions(
                study,
                spec,
                context,
                training,
                result,
                candidate_key=candidate_key,
            )
            verified = _cached_research_run(study, spec, context)
            if verified is None:
                raise ValueError(
                    f"TabM batch result failed fitted-state validation for {context.config['config_name']}"
                )
            diagnostics = {
                "execution_order": "fold_major",
                "compatibility_group": compatibility_group,
                "compatibility_group_size": len(items),
                "fitted_folds": candidate.fitted_folds,
                "group_size": len(pending),
                "reused": False,
                "reused_folds": candidate.reused_folds,
                **group_measurements,
                "candidate_fit_s": float(fit_elapsed_by_candidate[candidate_key]),
                "disk_fold_cache": False,
            }
            candidate.attempt.finish("completed", diagnostics)
            candidate.attempt = None
            completed[index] = ModelRun(
                training=verified.training,
                predictions=predictions,
                diagnostics=diagnostics,
            )
        failures = list(result.get("failures", {}).values())
        if failures:
            raise failures[0]
        for candidate_key, indices in duplicate_indices.items():
            primary = completed[primary_index_by_key[candidate_key]]
            for index in indices:
                completed[index] = primary
    except Exception as error:
        for candidate in recovery_candidates.values():
            if candidate.attempt is not None:
                candidate.attempt.finish(
                    "failed",
                    {
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "fitted_folds": candidate.fitted_folds,
                        "reused_folds": candidate.reused_folds,
                    },
                )
                candidate.attempt = None
        raise
    return completed


def plan_model_requests(
    study: Study,
    requests: list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[Any, ...]]:
    """Resolve compatible TabM requests once without fitting or result writes."""
    if not requests:
        raise ValueError("TabM batch planner requires at least one request")
    ordered: list[dict[str, Any] | None] = [None] * len(requests)
    resolved_by_index = []
    materialization_groups: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}
    for index, request in enumerate(requests):
        materialization_groups.setdefault(_tabm_materialization_key(request), []).append(
            (index, request)
        )
    for group in materialization_groups.values():
        materialized = _materialize_tabm_request_group(study, group[0][1])
        dataset_pd = None
        for index, request in group:
            spec, context = _resolve_model_request_from_materialized(
                study,
                request,
                label_ref=materialized[0],
                mds=materialized[1],
                dataset_pd=dataset_pd,
                configured_by_name=materialized[2],
                runtime_provenance=materialized[3],
            )
            dataset_pd = context.dataset_pd
            ordered[index] = spec
            resolved_by_index.append((index, spec, context))
    if any(spec is None for spec in ordered):
        raise RuntimeError("TabM batch planner did not resolve every request")
    return tuple(spec for spec in ordered if spec is not None), tuple(resolved_by_index)


def run_model_plan(study: Study, payload: tuple[Any, ...]) -> tuple[Any, ...]:
    execution_groups: dict[str, list[Any]] = {}
    for item in payload:
        execution_groups.setdefault(_tabm_execution_key(item[1]), []).append(item)
    completed = {}
    failures = []
    for compatible in execution_groups.values():
        try:
            completed.update(_run_tabm_compatible_group(study, compatible))
        except Exception as error:
            failures.append(error)
    if failures:
        raise failures[0]
    return tuple(completed[index] for index in range(len(payload)))


def run_model_requests(study: Study, requests: list[dict[str, Any]]):
    if not requests:
        return ()
    resolved_by_index = []
    materialization_groups: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}
    for index, request in enumerate(requests):
        materialization_groups.setdefault(_tabm_materialization_key(request), []).append(
            (index, request)
        )
    for group in materialization_groups.values():
        materialized = _materialize_tabm_request_group(study, group[0][1])
        dataset_pd = None
        for index, request in group:
            spec, context = _resolve_model_request_from_materialized(
                study,
                request,
                label_ref=materialized[0],
                mds=materialized[1],
                dataset_pd=dataset_pd,
                configured_by_name=materialized[2],
                runtime_provenance=materialized[3],
            )
            dataset_pd = context.dataset_pd
            resolved_by_index.append((index, spec, context))

    execution_groups: dict[str, list[Any]] = {}
    for item in resolved_by_index:
        execution_groups.setdefault(_tabm_execution_key(item[1]), []).append(item)
    completed = {}
    for compatible in execution_groups.values():
        completed.update(_run_tabm_compatible_group(study, compatible))
    return tuple(completed[index] for index in range(len(requests)))


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
    from case_studies.utils.deep_model_state import declared_epoch_checkpoints

    n_epochs = int(config.get("n_epochs", 200))
    checkpoint_interval = int(config.get("checkpoint_interval", 25))
    return declared_epoch_checkpoints(n_epochs, checkpoint_interval)


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
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

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
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(n_members)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)  # (batch, hidden)
        outputs = []
        for i in range(len(self.heads)):
            h_adapted = h * self.adapters[i].unsqueeze(0)  # rank-1 scaling
            outputs.append(self.heads[i](h_adapted))
        output = torch.stack(outputs, dim=0).mean(dim=0)
        return output.squeeze(-1) if output.shape[-1] == 1 else output


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


def _classification_scores(logits: np.ndarray, class_values: tuple[Any, ...]) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if len(class_values) == 2:
        return probabilities[:, 1]
    return probabilities @ np.asarray(class_values, dtype=np.float64)


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
    task_type: str = "regression",
    class_values: tuple[Any, ...] = (),
    class_weights: tuple[float, ...] = (),
    state_callback: Callable[[int, nn.Module], None] | None = None,
) -> tuple[dict[int, float], dict[int, np.ndarray], dict[int, float]]:
    """Train TabM on one fold, storing predictions at ALL checkpoints.

    Trains to completion (no early stopping). Stores predictions at every
    checkpoint so the caller can select the best epoch after all folds finish.

    Returns (checkpoint_ics, checkpoint_predictions, epoch_losses).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    is_classification = task_type == "classification"
    if is_classification:
        if not class_values or len(class_weights) != len(class_values):
            raise ValueError("classification training requires class values and aligned weights")
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        class_to_index = {value: index for index, value in enumerate(class_values)}
        try:
            y_train_index = np.asarray([class_to_index[value] for value in y_train], dtype=np.int64)
        except KeyError as error:
            raise ValueError(
                f"training label is outside declared classes: {error.args[0]!r}"
            ) from error
    else:
        criterion = nn.MSELoss()
        y_train_index = None
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
            if is_classification:
                if y_train_index is None:
                    raise RuntimeError("classification target index was not initialized")
                y_batch = torch.as_tensor(y_train_index[batch_idx], dtype=torch.long, device=device)
            else:
                y_batch = torch.as_tensor(y_train[batch_idx], dtype=torch.float32, device=device)

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
            raw_val_preds = _predict_in_chunks(model, X_val, device)
            val_preds = (
                _classification_scores(raw_val_preds, class_values)
                if is_classification
                else raw_val_preds
            )
            if state_callback is not None:
                state_callback(epoch, model)
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


def _n_invalid_scores(frame: pl.DataFrame) -> int:
    """Count null, NaN, and infinite checkpoint scores."""

    if frame.is_empty() or "y_score" not in frame.columns:
        return 0
    return int(
        frame.select(
            (
                pl.col("y_score").is_null().sum()
                + pl.col("y_score").is_nan().sum()
                + pl.col("y_score").is_infinite().sum()
            ).alias("n_invalid")
        ).item()
    )


def _checkpoint_prediction_frame(
    config_name: str,
    fold: int,
    checkpoint_preds: dict[int, np.ndarray],
    val_dates: np.ndarray,
    val_entities: np.ndarray | None,
    y_val: np.ndarray,
    date_col: str,
    entity_col: str,
    *,
    eval_actual: np.ndarray | None = None,
    eval_col: str = "eval_actual",
) -> pl.DataFrame:
    """Build the checkpoint frame used when incremental persistence is disabled."""
    dates = pl.Series(date_col, val_dates)
    if dates.dtype == pl.Object:
        dates = dates.map_elements(str, return_dtype=pl.String).str.to_datetime(strict=False)

    frames = []
    for epoch, scores in checkpoint_preds.items():
        n_rows = len(scores)
        entities = val_entities if val_entities is not None else np.array(["unknown"] * n_rows)
        frame = pl.DataFrame(
            {
                date_col: dates,
                entity_col: entities,
                "y_true": y_val.astype(np.float64),
                "y_score": scores.astype(np.float64),
                "fold_id": np.full(n_rows, fold, dtype=np.int32),
                "config": [config_name] * n_rows,
                "epoch": np.full(n_rows, epoch, dtype=np.int32),
            }
        )
        if eval_actual is not None:
            frame = frame.with_columns(pl.Series(eval_col, eval_actual.astype(np.float64)))
        frames.append(frame)
    return pl.concat(frames) if frames else pl.DataFrame()


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
        n_invalid = _n_invalid_scores(predictions)
        if n_invalid:
            raise ValueError(
                f"Cached {config_name} checkpoint {epoch} has {n_invalid} invalid scores"
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
                "ic_n_days": int(metric["n_periods"]),
                "n_invalid": n_invalid,
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
    full_days = max(int(row["ic_n_days"]) for row in curves)
    eligible = [
        row
        for row in curves
        if np.isfinite(float(row["ic_mean"]))
        and int(row["n_invalid"]) == 0
        and int(row["ic_n_days"]) == full_days
    ]
    if not eligible:
        raise ValueError(f"No selectable cached checkpoint for {config_name}")
    best = max(eligible, key=lambda row: row["ic_mean"])
    result = {
        "config_name": config_name,
        "best_epoch": best["epoch"],
        "best_ic": best["ic_mean"],
        "ic_n_days": best["ic_n_days"],
        "n_invalid": best["n_invalid"],
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

    enriched_results: list[dict[str, Any]] = []
    for result in config_results:
        enriched = dict(result)
        checkpoint_predictions = all_predictions.filter(
            (pl.col("config") == enriched["config_name"])
            & (pl.col("epoch") == enriched["best_epoch"])
        )
        if "ic_n_days" not in enriched and checkpoint_predictions.height:
            actual_col = eval_col or "y_true"
            enriched["ic_n_days"] = int(
                _decision_time_checkpoint_metrics(
                    checkpoint_predictions,
                    date_col=date_col,
                    entity_col=entity_col,
                    ret_col=actual_col,
                )["ic_n_days"]
            )
        enriched.setdefault("ic_n_days", 0)
        enriched.setdefault("n_invalid", _n_invalid_scores(checkpoint_predictions))
        enriched_results.append(enriched)

    positive_coverage = [
        int(row["ic_n_days"]) for row in enriched_results if int(row["ic_n_days"]) > 0
    ]
    full_coverage = max(positive_coverage) if positive_coverage else None
    for row in enriched_results:
        row["selectable"] = bool(
            np.isfinite(float(row["best_ic"]))
            and int(row["n_invalid"]) == 0
            and (full_coverage is None or int(row["ic_n_days"]) == full_coverage)
        )
    ranked = sorted(
        enriched_results,
        key=lambda row: (
            row["selectable"],
            float(row["best_ic"]) if np.isfinite(float(row["best_ic"])) else -999.0,
        ),
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


def _prepare_tabm_fold(
    dataset_pd: pd.DataFrame,
    split: dict[str, Any],
    *,
    feature_names: list[str],
    label_col: str,
    eval_label_col: str | None,
    date_col: str,
    entity_col: str,
    temporal_by_fold,
    temporal_keys: list[str] | None,
    temporal_feature_names: list[str] | None,
) -> dict[str, Any]:
    dates = dataset_pd[date_col]
    train_mask = (dates >= split["train_start"]) & (dates <= split["train_end"])
    val_mask = (dates >= split["val_start"]) & (dates <= split["val_end"])
    if temporal_by_fold is not None and temporal_keys and temporal_feature_names:
        from utils.modeling import replace_temporal_columns

        train_df = replace_temporal_columns(
            dataset_pd,
            train_mask,
            temporal_by_fold,
            temporal_keys,
            temporal_feature_names,
            split["fold"],
        )
        val_df = replace_temporal_columns(
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
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val = scaler.transform(imputer.transform(X_val))
    if scaler.mean_ is None or scaler.scale_ is None:
        raise RuntimeError("fitted standard scaler did not expose its state")
    return {
        "fold": split["fold"],
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "y_eval_val": y_eval_val,
        "val_dates": val_df[date_col].values,
        "val_entities": val_df[entity_col].values,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "preprocessing": {
            "feature_names": list(feature_names),
            "imputer_statistics": imputer.statistics_.astype(np.float32),
            "scaler_mean": np.asarray(scaler.mean_, dtype=np.float32),
            "scaler_scale": np.asarray(scaler.scale_, dtype=np.float32),
        },
    }


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
    class_weights_by_fold: dict[int, tuple[float, ...]] | None = None,
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
    checkpoint_root: Path | None = None,
    strict: bool = False,
    _recovery: _TabMRecovery | None = None,
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
    if task_type == "classification" and class_weights_by_fold is None:
        raise ValueError("classification requires identity-resolved class_weights_by_fold")
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
    if not splits:
        raise ValueError("TabM cross-validation requires at least one fold")

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
                    if checkpoint_root is not None:
                        from case_studies.utils.deep_model_state import (
                            validate_deep_checkpoint_population,
                        )

                        validate_deep_checkpoint_population(
                            checkpoint_root,
                            config_name=cfg["config_name"],
                            fold_ids=tuple(int(split["fold"]) for split in splits),
                            checkpoints=_tabm_checkpoint_epochs(cfg),
                            architecture="tabm",
                        )
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

    # Train every compatible candidate while one prepared fold is resident.
    config_results: list[dict[str, Any]] = list(cached_results)
    all_curves: list[dict] = list(cached_curves)
    training_log: list[dict] = []
    in_memory_prediction_frames: list[pl.DataFrame] = []
    run_save_dir = save_dir / label_col if save_dir is not None else None
    incr_dir = run_save_dir / "_incremental" if run_save_dir is not None else None
    if incr_dir is not None:
        incr_dir.mkdir(parents=True, exist_ok=True)

    states: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        artifact_name = cfg["config_name"]
        candidate_key = str(cfg.get("_execution_key", artifact_name))
        if candidate_key in states:
            raise ValueError(f"TabM execution keys must be unique: {candidate_key}")
        if (checkpoint_root is not None or _recovery is not None) and artifact_name.startswith(
            "tabpfn"
        ):
            raise ValueError("TabPFN fitted-state persistence is not implemented")
        if register and case_study and force_retrain:
            from case_studies.utils.registry import build_training_spec, training_hash_from_spec

            spec = build_training_spec(
                cfg["family"],
                artifact_name,
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
                    f"checkpoint(s) for {artifact_name}"
                )
        states[candidate_key] = {
            "available": True,
            "config": cfg,
            "elapsed_s": 0.0,
            "error": None,
            "fold_checkpoint_ics": {},
            "prediction_frames": [],
            "started_at": datetime.now(UTC).isoformat(),
        }

    preparation_elapsed_s = 0.0
    preparation_count = 0
    print("Preparing and releasing folds...")
    for split in splits:
        fold_id = int(split["fold"])
        pending_states = []
        for candidate_key, state in states.items():
            if not state["available"]:
                continue
            if _recovery is not None:
                reused = _recovery.reuse(candidate_key, fold_id)
                if reused is not None:
                    reused_predictions, reused_training_record = reused
                    state["prediction_frames"].append(reused_predictions)
                    training_log.append(reused_training_record)
                    print(f"  Fold {fold_id}: reused completed fitted state")
                    continue
            pending_states.append((candidate_key, state))
        if not pending_states:
            continue
        preparation_started = time.perf_counter()
        fd = _prepare_tabm_fold(
            dataset_pd,
            split,
            feature_names=feature_names,
            label_col=label_col,
            eval_label_col=eval_label_col,
            date_col=date_col,
            entity_col=entity_col,
            temporal_by_fold=temporal_by_fold,
            temporal_keys=temporal_keys,
            temporal_feature_names=temporal_feature_names,
        )
        preparation_elapsed_s += time.perf_counter() - preparation_started
        preparation_count += 1
        print(f"  Fold {fd['fold']}: train={fd['n_train']:,}  val={fd['n_val']:,}")
        for candidate_key, state in pending_states:
            cfg = state["config"]
            artifact_name = cfg["config_name"]
            checkpoint_artifact_name = str(cfg.get("_artifact_name", artifact_name))
            cfg_params = dict(cfg.get("params", {}))
            cfg_n_epochs = cfg.get("n_epochs", 200)
            cfg_batch_size = cfg.get("batch_size", 4096)
            cfg_checkpoint = cfg.get("checkpoint_interval", 25)
            is_tabpfn = artifact_name.startswith("tabpfn")
            fold_t0 = time.perf_counter()
            seed_everything(seed + fd["fold"])
            fold_prediction_frame = None
            fold_training_record = None
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
                    state["fold_checkpoint_ics"].setdefault(1, []).append(ic)
                    fold_prediction_frame = _checkpoint_prediction_frame(
                        candidate_key,
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
                    if _recovery is not None:
                        state["prediction_frames"].append(fold_prediction_frame)
                    elif incr_dir is not None:
                        flush_fold_predictions(
                            incr_dir,
                            candidate_key,
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
                    else:
                        state["prediction_frames"].append(fold_prediction_frame)

                    fold_elapsed = time.perf_counter() - fold_t0
                    fold_training_record = {
                        "config": candidate_key,
                        "fold": fd["fold"],
                        "elapsed_s": round(fold_elapsed, 1),
                        "n_train": fd["n_train"],
                        "n_val": fd["n_val"],
                        "best_ic": round(ic, 4),
                        "n_checkpoints": 1,
                    }
                    training_log.append(fold_training_record)
                    print(f"    Fold {fd['fold']}: IC={ic:+.4f} ({fold_elapsed:.1f}s)")
                except ImportError:
                    if int(fd["fold"]) == int(splits[0]["fold"]):
                        print("    TabPFN not installed — skipping")
                    state["available"] = False
                except (RuntimeError, ValueError) as e:
                    if int(fd["fold"]) == int(splits[0]["fold"]):
                        print(f"    TabPFN failed: {e}")
                    state["available"] = False
            else:
                output_dim = len(class_values or []) if task_type == "classification" else 1
                tabm_kwargs = {"n_features": n_features, "output_dim": output_dim, **cfg_params}
                model = TabMModel(**tabm_kwargs)
                train_kwargs: dict[str, Any] = {}
                candidate_checkpoint_root = (
                    _recovery.model_root(candidate_key)
                    if _recovery is not None
                    else checkpoint_root
                )
                if candidate_checkpoint_root is not None:
                    from case_studies.utils.deep_model_state import (
                        deep_checkpoint_path,
                        write_deep_checkpoint,
                    )

                    def persist_state(
                        epoch: int,
                        fitted_model: nn.Module,
                        *,
                        _fold: int = int(fd["fold"]),
                        _preprocessing: dict[str, Any] = fd["preprocessing"],
                        _config_name: str = checkpoint_artifact_name,
                        _model_kwargs: dict[str, Any] = tabm_kwargs,
                        _checkpoint_root: Path = candidate_checkpoint_root,
                    ) -> None:
                        write_deep_checkpoint(
                            deep_checkpoint_path(
                                _checkpoint_root,
                                _config_name,
                                _fold,
                                epoch,
                            ),
                            model=fitted_model,
                            architecture="tabm",
                            model_kwargs=_model_kwargs,
                            preprocessing=_preprocessing,
                            metadata={
                                "config_name": _config_name,
                                "fold": _fold,
                                "checkpoint_kind": "epoch",
                                "checkpoint_value": epoch,
                            },
                        )

                    train_kwargs["state_callback"] = persist_state
                try:
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
                        task_type=task_type,
                        class_values=tuple(class_values or ()),
                        class_weights=(class_weights_by_fold or {}).get(int(fd["fold"]), ()),
                        **train_kwargs,
                    )
                except Exception as error:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if _recovery is None and not (register and case_study):
                        raise
                    state["available"] = False
                    state["error"] = error
                    state["prediction_frames"].clear()
                    print(f"    Fold {fd['fold']}: {artifact_name} failed: {error}")
                    continue
                for ep, ic in checkpoint_ics.items():
                    state["fold_checkpoint_ics"].setdefault(ep, []).append(ic)
                fold_prediction_frame = _checkpoint_prediction_frame(
                    candidate_key,
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
                if _recovery is not None:
                    state["prediction_frames"].append(fold_prediction_frame)
                elif incr_dir is not None:
                    flush_fold_predictions(
                        incr_dir,
                        candidate_key,
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
                else:
                    state["prediction_frames"].append(fold_prediction_frame)

                del model, checkpoint_preds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                best_ep = max(checkpoint_ics, key=lambda e: checkpoint_ics[e])
                fold_elapsed = time.perf_counter() - fold_t0
                loss_at_checkpoints = {
                    str(k): round(epoch_losses.get(k, 0.0), 6)
                    for k in sorted(checkpoint_ics.keys())
                }
                fold_training_record = {
                    "config": candidate_key,
                    "fold": fd["fold"],
                    "elapsed_s": round(fold_elapsed, 1),
                    "n_train": fd["n_train"],
                    "n_val": fd["n_val"],
                    "best_ic": round(checkpoint_ics[best_ep], 4),
                    "n_checkpoints": len(checkpoint_ics),
                    "checkpoint_ics": {str(k): round(v, 4) for k, v in checkpoint_ics.items()},
                    "checkpoint_losses": loss_at_checkpoints,
                }
                training_log.append(fold_training_record)
                print(
                    f"    Fold {fd['fold']}: best_ep={best_ep}, "
                    f"IC={checkpoint_ics[best_ep]:+.4f} ({fold_elapsed:.1f}s)"
                )
            if _recovery is not None and fold_prediction_frame is not None:
                try:
                    if fold_training_record is None:
                        raise RuntimeError("TabM fold completed without a training record")
                    _recovery.complete(
                        candidate_key,
                        int(fd["fold"]),
                        fold_prediction_frame,
                        fold_training_record,
                    )
                except Exception as error:
                    state["available"] = False
                    state["error"] = error
                    state["prediction_frames"].clear()
                    print(f"    Fold {fd['fold']}: {artifact_name} persistence failed: {error}")
                    continue
            state["elapsed_s"] += time.perf_counter() - fold_t0
        del fd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    completed_candidate_keys: list[str] = []
    for candidate_key, state in states.items():
        if not state["available"]:
            continue
        cfg = state["config"]
        artifact_name = cfg["config_name"]
        checkpoint_artifact_name = str(cfg.get("_artifact_name", artifact_name))
        completed_candidate_keys.append(candidate_key)
        fold_checkpoint_ics = state["fold_checkpoint_ics"]
        config_prediction_frames = state["prediction_frames"]
        prediction_parts = list(config_prediction_frames)
        if incr_dir is not None:
            incremental = _load_incremental_preds_for_config(incr_dir, candidate_key)
            if incremental.height:
                prediction_parts.append(incremental)
        cfg_all_preds = pl.concat(prediction_parts) if prediction_parts else pl.DataFrame()
        if (incr_dir is None or _recovery is not None) and cfg_all_preds.height:
            in_memory_prediction_frames.append(cfg_all_preds)
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
                checkpoint_metrics[int(epoch)]["n_invalid"] = _n_invalid_scores(epoch_predictions)
        elif fold_checkpoint_ics:
            checkpoint_metrics = {
                int(epoch): {
                    "ic_mean": float(np.nanmean(values)),
                    "ic_std": float(np.nanstd(values)) if len(values) > 1 else 0.0,
                    "ic_n_days": 0,
                    "n_invalid": 0,
                }
                for epoch, values in fold_checkpoint_ics.items()
            }

        if checkpoint_metrics:
            positive_days = [
                int(metric.get("ic_n_days", 0))
                for metric in checkpoint_metrics.values()
                if int(metric.get("ic_n_days", 0)) > 0
            ]
            full_days = max(positive_days) if positive_days else None
            eligible_checkpoints = [
                epoch
                for epoch, metric in checkpoint_metrics.items()
                if np.isfinite(float(metric["ic_mean"]))
                and int(metric.get("n_invalid", 0)) == 0
                and (full_days is None or int(metric.get("ic_n_days", 0)) == full_days)
            ]
            if not eligible_checkpoints:
                raise ValueError(f"No selectable checkpoint completed for {artifact_name}")
            best_cp = max(
                eligible_checkpoints,
                key=lambda epoch: checkpoint_metrics[epoch]["ic_mean"],
            )
            best_ic_val = float(checkpoint_metrics[best_cp]["ic_mean"])
            best_ic_n_days = int(checkpoint_metrics[best_cp].get("ic_n_days", 0))
            best_n_invalid = int(checkpoint_metrics[best_cp].get("n_invalid", 0))
        else:
            best_cp = 0
            best_ic_val = float("nan")
            best_ic_n_days = 0
            best_n_invalid = 0
        elapsed = float(state["elapsed_s"])
        config_started_at = state["started_at"]
        config_results.append(
            {
                "config_name": candidate_key,
                "best_epoch": best_cp,
                "best_ic": best_ic_val,
                "ic_n_days": best_ic_n_days,
                "n_invalid": best_n_invalid,
                "elapsed_s": elapsed,
                "started_at": config_started_at,
            }
        )
        cfg_curves_list = []
        for ep, metric in sorted(checkpoint_metrics.items()):
            entry = {
                "config": candidate_key,
                "epoch": ep,
                "ic_mean": float(metric["ic_mean"]),
                "ic_std": float(metric.get("ic_std", 0.0)),
                "ic_n_days": int(metric.get("ic_n_days", 0)),
                "n_invalid": int(metric.get("n_invalid", 0)),
            }
            all_curves.append(entry)
            cfg_curves_list.append(entry)
        print(f"    → best_epoch={best_cp}, IC={best_ic_val:+.4f} ({elapsed:.1f}s)")
        candidate_checkpoint_root = (
            _recovery.model_root(candidate_key) if _recovery is not None else checkpoint_root
        )
        if candidate_checkpoint_root is not None:
            from case_studies.utils.deep_model_state import validate_deep_checkpoint_population

            validate_deep_checkpoint_population(
                candidate_checkpoint_root,
                config_name=checkpoint_artifact_name,
                fold_ids=tuple(int(split["fold"]) for split in splits),
                checkpoints=_tabm_checkpoint_epochs(cfg),
                architecture="tabm",
            )
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
                        config_name=artifact_name,
                        n_epochs=cfg.get("n_epochs"),
                        best_epoch=int(first_ep),
                        n_folds=len(splits),
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
                        training_spec=training_specs[artifact_name],
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
                        f"    registered {artifact_name} incrementally ({len(epochs)} per-epoch slices)"
                    )
            except Exception as exc:
                if strict:
                    raise
                print(f"    WARN: incremental registration failed for {artifact_name}: {exc}")
        gc.collect()

    failures = {
        candidate_key: state["error"]
        for candidate_key, state in states.items()
        if state["error"] is not None
    }
    direct_failure = next(iter(failures.values()), None) if _recovery is None else None
    prediction_frames = [*cached_prediction_frames, *in_memory_prediction_frames]
    if incr_dir is not None and _recovery is None:
        for candidate_key in completed_candidate_keys:
            frame = _load_incremental_preds_for_config(incr_dir, candidate_key)
            if frame.height:
                prediction_frames.append(frame)
    all_predictions = pl.concat(prediction_frames) if prediction_frames else pl.DataFrame()
    execution_diagnostics = {
        "base_fold_preparation_s": preparation_elapsed_s,
        "base_fold_preparations": preparation_count,
        "candidate_fit_s": {
            candidate_key: float(state["elapsed_s"]) for candidate_key, state in states.items()
        },
    }
    if not config_results:
        if direct_failure is not None:
            raise direct_failure
        return {
            "all_learning_curves": pl.DataFrame(all_curves),
            "all_predictions": all_predictions,
            "failures": failures,
            "fold_metrics": pl.DataFrame(),
            "grid_results": [],
            "predictions": pl.DataFrame(),
            "training_log": pl.DataFrame(training_log),
            "execution_diagnostics": execution_diagnostics,
        }
    assembled = _assemble_tabm_results(
        config_results=config_results,
        all_predictions=all_predictions,
        curve_rows=all_curves,
        training_rows=training_log,
        save_dir=run_save_dir,
        date_col=date_col,
        entity_col=entity_col,
        eval_col=eval_col,
    )
    assembled["failures"] = failures
    assembled["execution_diagnostics"] = execution_diagnostics
    if direct_failure is not None:
        raise direct_failure
    return assembled
