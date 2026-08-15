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
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
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
import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic
from torch.utils.data import DataLoader

from case_studies.research.identity import ResolvedSpec
from case_studies.research.models import ModelRun
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.cv_results import (
    assemble_cv_result,
    combine_cv_results,
    rebuild_cv_result_from_registry,
)
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
    sequence_validation_keys,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from utils.modeling import RANDOM_SEED, seed_everything

if TYPE_CHECKING:
    from case_studies.research.workspace import Study

_SEQUENCE_PREVIEW_FIELDS = {"folds", "max_symbols", "max_train_sequences"}


@dataclass(frozen=True)
class SequenceResearchContext:
    dataset_pd: pd.DataFrame
    splits: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    feature_names: tuple[str, ...]
    label_col: str
    date_col: str
    entity_col: str
    task_type: str
    class_values: tuple[Any, ...]
    temporal_by_fold: pd.DataFrame | None
    temporal_keys: tuple[str, ...]
    temporal_feature_names: tuple[str, ...]
    expected_keys: pl.DataFrame
    max_train_sequences: int
    runtime_provenance: dict[str, Any]
    prediction_split: str = "validation"
    published_checkpoints: tuple[int, ...] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_path(module: Any) -> Path:
    source_file = getattr(module, "__file__", None)
    if not isinstance(source_file, str):
        raise RuntimeError(f"{module.__name__} has no source file")
    return Path(source_file)


def _sequence_source_identity(config: dict[str, Any]) -> dict[str, str]:
    from case_studies.utils import deep_model_state, sequence_dataset

    paths = [Path(__file__), _module_path(deep_model_state), _module_path(sequence_dataset)]
    if config.get("library") == "darts":
        from case_studies.utils import darts_forecasting

        paths.append(_module_path(darts_forecasting))
    else:
        architecture = str(config["params"]["architecture"])
        model_class = _get_model_registry()[architecture]
        module = __import__(model_class.__module__, fromlist=[model_class.__name__])
        paths.append(_module_path(module))
    return {path.name: _sha256(path) for path in paths}


def _sequence_runtime_identity(config: dict[str, Any]) -> dict[str, str]:
    packages = {
        "numpy": importlib.metadata.version("numpy"),
        "torch": importlib.metadata.version("torch"),
    }
    if config.get("library") == "darts":
        packages.update(
            {
                "darts": importlib.metadata.version("darts"),
                "pytorch-lightning": importlib.metadata.version("pytorch-lightning"),
            }
        )
    return packages


def _sequence_runtime_provenance(study: Study, config: dict[str, Any]) -> dict[str, Any]:
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
        "entry_point": "case_studies.utils.deep_learning",
        "packages": _sequence_runtime_identity(config),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_commit": commit,
    }


def _normalize_sequence_splits(
    splits: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    fields = ("fold", "train_start", "train_end", "val_start", "val_end")
    return tuple(
        {
            key: int(split[key]) if key == "fold" else str(split[key])
            for key in fields
            if split.get(key) is not None
        }
        for split in splits
    )


def _select_sequence_observations(
    frame: pl.DataFrame | pd.DataFrame,
    *,
    date_col: str,
    cadence: str | None,
    calendar: str | None,
) -> pl.DataFrame | pd.DataFrame:
    if cadence is None:
        return frame
    from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

    supported = {"monthly_month_end", "weekly", "weekly_friday", "weekly_friday_close"}
    if cadence not in supported:
        raise ValueError(f"unsupported sequence decision cadence {cadence!r}")
    if date_col not in frame.columns:
        raise ValueError(f"decision cadence requires timestamp column {date_col!r}")
    timestamps = (
        frame.get_column(date_col).unique()
        if isinstance(frame, pl.DataFrame)
        else pl.Series(date_col, frame[date_col].unique().to_numpy())
    )
    selected = resolve_rebalance_timestamps(
        timestamps,
        cadence,
        calendar or "NYSE",
    )
    if selected.is_empty():
        raise ValueError(f"decision cadence {cadence!r} selected no observations")
    if isinstance(frame, pl.DataFrame):
        return frame.join(pl.DataFrame({date_col: selected}), on=date_col, how="semi")
    return frame.loc[frame[date_col].isin(selected.to_pandas())].copy()


def _sequence_splits(mds, request: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from case_studies.utils.artifact_digest import value_digest

    cv = request.get("cv")
    if cv is None:
        splits = [dict(split) for split in mds.splits]
        normalized = _normalize_sequence_splits(splits)
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
        raise ValueError("sequence request resolved no cross-validation folds")
    from utils.modeling import validate_temporal_split_geometry

    validate_temporal_split_geometry(splits, mds.splits, mds.temporal_by_fold)
    return splits, cv_record


def _resolve_sequence_config(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {**config, "params": dict(config.get("params") or {})}
    top_level = {"batch_size", "checkpoint_interval", "n_epochs", "seed"}
    runtime_fields = {"device", "num_threads"}
    unknown = set(overrides) - top_level - runtime_fields - set(resolved["params"])
    if unknown:
        raise ValueError(f"unsupported sequence overrides: {sorted(unknown)}")
    for key, value in overrides.items():
        if key in top_level:
            resolved[key] = value
        elif key not in runtime_fields:
            resolved["params"][key] = value
    architecture = str(resolved["params"].get("architecture", ""))
    library = resolved.get("library", "pytorch")
    if library == "darts":
        from case_studies.utils.darts_forecasting import SUPPORTED_DARTS_ARCHITECTURES

        if architecture not in SUPPORTED_DARTS_ARCHITECTURES:
            raise ValueError(f"unsupported Darts sequence architecture {architecture!r}")
    elif library != "pytorch" or architecture not in _get_model_registry():
        raise ValueError(f"unsupported PyTorch sequence architecture {architecture!r}")
    from case_studies.utils.deep_model_state import declared_epoch_checkpoints

    declared_epoch_checkpoints(
        int(resolved.get("n_epochs", 100)),
        int(resolved.get("checkpoint_interval", 5)),
    )
    return resolved


def _sequence_runtime_spec(
    device: str,
    *,
    seed: int,
    num_threads: int,
) -> dict[str, Any]:
    if num_threads < 1:
        raise ValueError("num_threads must be at least 1")
    normalized = device.lower().replace("gpu", "cuda")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for sequence training, but CUDA is unavailable")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported sequence device {device!r}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return {
        "device": normalized,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "num_threads": num_threads,
        "seed": seed,
    }


def _configure_sequence_runtime(spec: dict[str, Any]) -> None:
    torch.set_num_threads(int(spec["num_threads"]))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(int(spec["seed"]))


def resolve_model_request(study: Study, request: dict[str, Any]):
    from case_studies.research.contracts import ExecutionTier
    from case_studies.utils.deep_model_state import declared_epoch_checkpoints
    from utils.cv_splits import make_walk_forward_config
    from utils.modeling import load_configs, load_modeling_dataset

    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    unknown_reductions = set(reductions) - _SEQUENCE_PREVIEW_FIELDS
    if unknown_reductions:
        raise ValueError(f"unsupported sequence preview reductions: {sorted(unknown_reductions)}")
    study.require_writable()
    study.activate(tier)
    label_ref = study.labels.get(request["label"])
    mds = load_modeling_dataset(
        study.case_study,
        label_ref.name,
        max_symbols=int(reductions.get("max_symbols", 0)),
    )
    if mds.date_col != "timestamp" or not mds.entity_cols:
        raise ValueError("sequence runner requires timestamp and an entity key")
    entity_col = mds.entity_cols[0]
    if entity_col not in {"product", "symbol"}:
        raise ValueError(f"sequence runner does not support entity key {entity_col!r}")
    if mds.task_type != "regression":
        raise ValueError("sequence runner currently supports regression labels only")
    cv = request.get("cv")
    splits, cv_record = _sequence_splits(mds, request)
    configs = {
        config["config_name"]: config
        for config in load_configs(study.case_study, label_ref.name, "deep_learning")
    }
    try:
        configured = configs[request["config_name"]]
    except KeyError as error:
        raise ValueError(f"unknown sequence configuration {request['config_name']!r}") from error
    config = _resolve_sequence_config(configured, request["overrides"])
    if config.get("family") != "deep_learning":
        raise ValueError(f"sequence preset belongs to unsupported family {config.get('family')!r}")
    preset_cadence = config["params"].get("decision_cadence")
    requested_cadence = cv.decision_cadence if cv is not None else None
    if preset_cadence and requested_cadence and preset_cadence != requested_cadence:
        raise ValueError(
            f"sequence preset requires decision cadence {preset_cadence!r}, "
            f"not {requested_cadence!r}"
        )
    decision_cadence = requested_cadence or preset_cadence
    if decision_cadence is not None:
        config["params"]["decision_cadence"] = decision_cadence
    dataset = _select_sequence_observations(
        mds.dataset,
        date_col=mds.date_col,
        cadence=decision_cadence,
        calendar=cv.calendar if cv is not None else None,
    )
    seed = int(config.get("seed", RANDOM_SEED))
    runtime = _sequence_runtime_spec(
        str(request["overrides"].get("device", "cuda")),
        seed=seed,
        num_threads=int(request["overrides"].get("num_threads", 8)),
    )
    max_train_sequences = int(reductions.get("max_train_sequences", 0))
    calendar_id = make_walk_forward_config(study.case_study, date_col=mds.date_col).calendar_id
    lookback = int(config["params"].get("lookback", 60))
    dataset_pd = dataset.to_pandas()
    if config.get("library") == "darts":
        from case_studies.utils.darts_forecasting import (
            darts_training_identity,
            darts_validation_keys,
        )

        # darts_validation_keys names its keys after the entity column and adds
        # `position` where the panel has one, unlike the three key builders that
        # emit `symbol`. No product-keyed case study configures a Darts preset,
        # so that combination has never been exercised; refuse it here rather
        # than fail later on a key that is missing from the digest.
        if entity_col != "symbol":
            raise ValueError(f"Darts presets require the symbol entity key, not {entity_col!r}")

        expected = darts_validation_keys(
            dataset_pd,
            splits,
            config=config,
            feature_names=list(mds.feature_names),
            label_col=mds.label_col,
            date_col=mds.date_col,
            entity_col=entity_col,
            case_study=study.case_study,
            calendar_id=calendar_id,
            temporal_by_fold=mds.temporal_by_fold,
            temporal_keys=list(mds.temporal_keys),
            temporal_feature_names=list(mds.temporal_feature_names),
        )
        sequence_identity = darts_training_identity(
            config,
            mds.label_col,
            case_study=study.case_study,
            input_data_spec=mds.input_lineage,
            max_train_sequences=max_train_sequences,
        )
        preprocessing = {
            "class": "fold_train_standardization",
            "base_target": sequence_identity["base_target_data_spec"],
            "calendar_id": calendar_id,
            "decision_cadence": decision_cadence,
            "input_chunk_length": sequence_identity["input_chunk_length"],
            "output_chunk_length": sequence_identity["output_chunk_length"],
        }
    else:
        expected = sequence_validation_keys(
            dataset_pd,
            splits,
            label_col=mds.label_col,
            date_col=mds.date_col,
            entity_col=entity_col,
            lookback=lookback,
            calendar_id=calendar_id,
        )
        sequence_identity = {
            "input_data_spec": mds.input_lineage,
            "lookback": lookback,
            "max_train_sequences": max_train_sequences,
        }
        preprocessing = {
            "class": "fold_train_standardization",
            "calendar_id": calendar_id,
            "gap_policy": "exclude_windows_crossing_missing_expected_periods",
            "lookback": lookback,
        }
    checkpoints = declared_epoch_checkpoints(
        int(config.get("n_epochs", 100)), int(config.get("checkpoint_interval", 5))
    )
    computation = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": mds.input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "task": {
            "type": "regression",
            "class_values": [],
            "continuous_eval_label": label_ref.definition.continuous_eval_label,
        },
        "cv": cv_record,
        "model": {
            "class": config["params"]["architecture"],
            "implementation": config.get("library", "pytorch"),
            "objective": "regression",
            "params": {
                **config["params"],
                "batch_size": int(config["batch_size"]),
                "checkpoint_interval": int(config["checkpoint_interval"]),
                "n_epochs": int(config["n_epochs"]),
            },
        },
        "preprocessing": preprocessing,
        "checkpoint_schedule": [
            {"kind": "epoch", "value": checkpoint} for checkpoint in checkpoints
        ],
        "expected_prediction_keys": {
            "digest": value_digest(expected, ("symbol", "timestamp", "fold")),
            "n_rows": expected.height,
            "n_folds": expected["fold"].n_unique(),
        },
        "input_data_spec": sequence_identity,
        "sampling": {
            "max_symbols": int(reductions.get("max_symbols", 0)),
            "max_train_sequences": max_train_sequences,
        },
        "numerics": runtime,
        "source_identity": _sequence_source_identity(config),
        "runtime_identity": _sequence_runtime_identity(config),
    }
    if tier is ExecutionTier.PREVIEW:
        computation["preview_reductions"] = reductions
    runtime_provenance = _sequence_runtime_provenance(study, config)
    spec = ResolvedSpec.create(
        family="deep_learning",
        label=label_ref.name,
        seed=seed,
        computation=computation,
        provenance=runtime_provenance,
        config_name=config["config_name"],
        execution_tier=tier.value,
    ).as_dict()
    context = SequenceResearchContext(
        dataset_pd=dataset_pd,
        splits=tuple(splits),
        config=config,
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        date_col=mds.date_col,
        entity_col=entity_col,
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=tuple(mds.temporal_keys),
        temporal_feature_names=tuple(mds.temporal_feature_names),
        expected_keys=expected,
        max_train_sequences=max_train_sequences,
        runtime_provenance=runtime_provenance,
    )
    return spec, context


def reconstruct_locked_request(
    study: Study,
    spec: dict[str, Any],
    *,
    checkpoint_kind: str,
    checkpoint_value: int | None,
):
    """Reconstruct a sequence holdout fit without consulting a mutable preset."""
    from case_studies.research.contracts import ExecutionTier
    from case_studies.research.cv import require_fold_scoped_temporal_compatibility
    from case_studies.research.models import (
        ResolvedModelRequest,
        locked_holdout_split,
        validate_locked_expected_keys,
    )
    from case_studies.utils.artifact_digest import value_digest
    from case_studies.utils.deep_model_state import declared_epoch_checkpoints
    from case_studies.utils.registry import training_hash_from_spec
    from utils.modeling import load_modeling_dataset

    if checkpoint_kind != "epoch" or checkpoint_value is None:
        raise ValueError("sequence holdout requires one locked epoch checkpoint")
    study.require_writable()
    study.activate(ExecutionTier.CANONICAL)
    computation = spec["computation"]
    if computation.get("sampling") != {"max_symbols": 0, "max_train_sequences": 0}:
        raise ValueError("locked sequence holdout requires unreduced canonical inputs")
    label_ref = study.labels.get(spec["label"], execution_tier=ExecutionTier.CANONICAL)
    mds = load_modeling_dataset(study.case_study, label_ref.name, max_symbols=0)
    if mds.date_col != "timestamp" or mds.entity_cols[:1] not in (["symbol"], ["product"]):
        raise ValueError("locked sequence runner requires canonical entity and timestamp keys")
    if mds.task_type != "regression":
        raise ValueError("locked sequence runner currently supports regression labels only")
    split = locked_holdout_split(spec, mds.dataset, mds.date_col, study.case_study)
    if mds.temporal_by_fold is not None and mds.temporal_keys and mds.temporal_feature_names:
        require_fold_scoped_temporal_compatibility([split], mds.temporal_artifact_splits)

    model = computation.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("params"), dict):
        raise ValueError("locked sequence model specification is missing")
    model_params = dict(model["params"])
    required_training = {
        name: model_params.pop(name, None)
        for name in ("batch_size", "checkpoint_interval", "n_epochs")
    }
    if any(value is None for value in required_training.values()):
        raise ValueError("locked sequence model omits exact training parameters")
    config = _resolve_sequence_config(
        {
            "family": "deep_learning",
            "library": model.get("implementation"),
            "config_name": str(
                spec.get("config_name") or f"locked-{training_hash_from_spec(spec)}"
            ),
            "params": model_params,
            **required_training,
            "seed": int(spec["seed"]),
        },
        {},
    )
    expected_model = {
        "class": config["params"]["architecture"],
        "implementation": config.get("library", "pytorch"),
        "objective": "regression",
        "params": {
            **config["params"],
            "batch_size": int(config["batch_size"]),
            "checkpoint_interval": int(config["checkpoint_interval"]),
            "n_epochs": int(config["n_epochs"]),
        },
    }
    if model != expected_model:
        raise ValueError("locked sequence model specification does not reproduce exactly")
    checkpoints = declared_epoch_checkpoints(
        int(config["n_epochs"]), int(config["checkpoint_interval"])
    )
    expected_schedule = [{"kind": "epoch", "value": value} for value in checkpoints]
    if (
        computation.get("checkpoint_schedule") != expected_schedule
        or checkpoint_value not in checkpoints
    ):
        raise ValueError("locked sequence checkpoint is absent from its exact schedule")
    numerics = computation.get("numerics")
    if not isinstance(numerics, dict):
        raise ValueError("locked sequence numerics are missing")
    reproduced_numerics = _sequence_runtime_spec(
        str(numerics.get("device")),
        seed=int(numerics.get("seed")),
        num_threads=int(numerics.get("num_threads")),
    )
    if numerics != reproduced_numerics or int(spec["seed"]) != int(numerics["seed"]):
        raise ValueError("locked sequence numerics cannot be reproduced")

    preprocessing = computation.get("preprocessing")
    if not isinstance(preprocessing, dict) or not preprocessing.get("calendar_id"):
        raise ValueError("locked sequence preprocessing omits the exact calendar")
    # resolve_model_request thins the dataset to the decision cadence before it builds
    # any target or window, so reconstruction has to thin it the same way. Otherwise a
    # weekly-cadence lock retrains on every session, which is a different model. The
    # cadence comes from the locked model params, never from the preset.
    decision_cadence = config["params"].get("decision_cadence")
    locked_cv = computation.get("cv") or {}
    locked_cv_request = locked_cv.get("request") or {}
    # Presence, not truth. CVSpec.calendar is None whenever it was left unset, and a
    # resolved CV record still writes the key, so None is an exactly reproducible
    # choice rather than a missing one - rejecting it would refuse a canonical weekly
    # run that simply never named a calendar. Only a block that never mentions the key
    # leaves nothing to reproduce, and defaulting there would silently thin on a
    # different calendar than the fit did once resolve_rebalance_timestamps honors it.
    for block in (locked_cv, locked_cv_request):
        if "calendar" in block:
            cadence_calendar = block["calendar"]
            break
    else:
        if decision_cadence is not None:
            raise ValueError(
                "locked cadence-selected sequence training does not record its calendar, "
                "so its observation selection cannot be reproduced"
            )
        cadence_calendar = None
    dataset_pd = _select_sequence_observations(
        mds.dataset,
        date_col=mds.date_col,
        cadence=decision_cadence,
        calendar=cadence_calendar,
    ).to_pandas()
    lookback = int(config["params"].get("lookback", 60))
    if config.get("library") == "darts":
        from case_studies.utils.darts_forecasting import (
            darts_training_identity,
            darts_validation_keys,
        )

        expected = darts_validation_keys(
            dataset_pd,
            [split],
            config=config,
            feature_names=list(mds.feature_names),
            label_col=mds.label_col,
            date_col=mds.date_col,
            entity_col=mds.entity_cols[0],
            case_study=study.case_study,
            calendar_id=str(preprocessing["calendar_id"]),
            temporal_by_fold=mds.temporal_by_fold,
            temporal_keys=list(mds.temporal_keys),
            temporal_feature_names=list(mds.temporal_feature_names),
        )
        input_data_spec = darts_training_identity(
            config,
            mds.label_col,
            case_study=study.case_study,
            input_data_spec=mds.input_lineage,
            max_train_sequences=0,
        )
        expected_preprocessing = {
            "class": "fold_train_standardization",
            "base_target": input_data_spec["base_target_data_spec"],
            "calendar_id": preprocessing["calendar_id"],
            "decision_cadence": decision_cadence,
            "input_chunk_length": input_data_spec["input_chunk_length"],
            "output_chunk_length": input_data_spec["output_chunk_length"],
        }
    else:
        expected = sequence_validation_keys(
            dataset_pd,
            [split],
            label_col=mds.label_col,
            date_col=mds.date_col,
            entity_col=mds.entity_cols[0],
            lookback=lookback,
            calendar_id=str(preprocessing["calendar_id"]),
        )
        input_data_spec = {
            "input_data_spec": mds.input_lineage,
            "lookback": lookback,
            "max_train_sequences": 0,
        }
        expected_preprocessing = {
            "class": "fold_train_standardization",
            "calendar_id": preprocessing["calendar_id"],
            "gap_policy": "exclude_windows_crossing_missing_expected_periods",
            "lookback": lookback,
        }
    validate_locked_expected_keys(spec, expected)
    expected_inputs = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": mds.input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "task": {
            "type": "regression",
            "class_values": [],
            "continuous_eval_label": label_ref.definition.continuous_eval_label,
        },
        "preprocessing": expected_preprocessing,
        "input_data_spec": input_data_spec,
        "source_identity": _sequence_source_identity(config),
        "runtime_identity": _sequence_runtime_identity(config),
    }
    for name, expected_value in expected_inputs.items():
        if computation.get(name) != expected_value:
            raise ValueError(f"locked sequence {name} does not match the available computation")
    context = SequenceResearchContext(
        dataset_pd=dataset_pd,
        splits=(split,),
        config=config,
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        date_col=mds.date_col,
        entity_col=mds.entity_cols[0],
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=tuple(mds.temporal_keys),
        temporal_feature_names=tuple(mds.temporal_feature_names),
        expected_keys=expected,
        max_train_sequences=0,
        runtime_provenance=_sequence_runtime_provenance(study, config),
        prediction_split="holdout",
        published_checkpoints=(int(checkpoint_value),),
    )
    return ResolvedModelRequest(study, "deep_learning", spec, context)


def _cached_sequence_run(study: Study, spec: dict[str, Any], context: SequenceResearchContext):
    from case_studies.research.models import ModelRun
    from case_studies.research.results import PredictionResult, Result, TrainingResult
    from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec

    training_hash = training_hash_from_spec(spec)
    computation = spec["computation"]
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
    model_root = training.root / "run_log" / "training" / training.hash / "models"
    fold_ids = tuple(int(split["fold"]) for split in context.splits)
    try:
        if context.config.get("library") == "darts":
            from case_studies.utils.darts_forecasting import (
                validate_darts_checkpoint_population,
            )

            validate_darts_checkpoint_population(
                model_root,
                config_name=context.config["config_name"],
                fold_ids=fold_ids,
                checkpoints=checkpoint_values,
                architecture=context.config["params"]["architecture"],
            )
        else:
            from case_studies.utils.deep_model_state import validate_deep_checkpoint_population

            validate_deep_checkpoint_population(
                model_root,
                config_name=context.config["config_name"],
                fold_ids=fold_ids,
                checkpoints=checkpoint_values,
                architecture=context.config["params"]["architecture"],
            )
    except ValueError:
        return None
    complete_predictions = tuple(
        result for result in predictions if isinstance(result, PredictionResult)
    )
    return ModelRun(training=training, predictions=complete_predictions)


def _predict_reconstructed_sequence(
    model: nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.as_tensor(
                sequences[start : start + batch_size], dtype=torch.float32, device=device
            )
            parts.append(model(batch).detach().cpu().numpy())
    if not parts:
        raise ValueError("locked sequence checkpoint produced no predictions")
    return np.concatenate(parts)


def _reconstruct_pytorch_predictions(
    model_root: Path,
    context: SequenceResearchContext,
    computation: dict[str, Any],
) -> pl.DataFrame:
    from case_studies.utils.deep_model_state import deep_checkpoint_path, restore_deep_model
    from case_studies.utils.sequence_dataset import materialize_sequences

    frames = []
    lookback = int(context.config["params"].get("lookback", 60))
    calendar_id = str(computation["preprocessing"]["calendar_id"])
    device = torch.device(str(computation["numerics"]["device"]))

    def model_factory(name: str, config: Mapping[str, Any]) -> nn.Module:
        return create_model(name, dict(config))

    dates = context.dataset_pd[context.date_col]
    for split in context.splits:
        train_mask = (dates >= split["train_start"]) & (dates <= split["train_end"])
        val_mask = (dates >= split["val_start"]) & (dates <= split["val_end"])
        train_store, val_store, _ = prepare_fold_sequence_stores(
            context.dataset_pd,
            train_mask=train_mask,
            val_mask=val_mask,
            feature_names=list(context.feature_names),
            label_col=context.label_col,
            date_col=context.date_col,
            entity_col=context.entity_col,
            lookback=lookback,
            max_train_sequences=0,
            temporal_by_fold=context.temporal_by_fold,
            temporal_keys=list(context.temporal_keys),
            temporal_feature_names=list(context.temporal_feature_names),
            fold_id=int(split["fold"]),
            val_start=split["val_start"],
            calendar_id=calendar_id,
        )
        if not train_store.n_sequences or not val_store.n_sequences:
            raise ValueError(f"locked sequence fold {split['fold']} cannot be reconstructed")
        sequences, actual, timestamps, entities = materialize_sequences(val_store)
        for checkpoint in computation["checkpoint_schedule"]:
            value = int(checkpoint["value"])
            model, preprocessing, metadata = restore_deep_model(
                deep_checkpoint_path(
                    model_root,
                    context.config["config_name"],
                    int(split["fold"]),
                    value,
                ),
                model_factory,
            )
            expected_metadata = {
                "config_name": context.config["config_name"],
                "fold": int(split["fold"]),
                "checkpoint_kind": "epoch",
                "checkpoint_value": value,
            }
            if metadata != expected_metadata:
                raise ValueError("locked sequence checkpoint metadata changed")
            stored_mean = preprocessing.get("mean")
            stored_scale = preprocessing.get("scale")
            if (
                preprocessing.get("feature_names") != list(context.feature_names)
                or int(preprocessing.get("lookback", -1)) != lookback
                or stored_mean is None
                or stored_scale is None
                or train_store.feature_mean is None
                or train_store.feature_scale is None
                or not np.array_equal(stored_mean, train_store.feature_mean)
                or not np.array_equal(stored_scale, train_store.feature_scale)
            ):
                raise ValueError("locked sequence preprocessing state changed")
            model = model.to(device)
            prediction = _predict_reconstructed_sequence(
                model,
                sequences,
                device,
                batch_size=int(context.config["batch_size"]),
            )
            frames.append(
                pl.DataFrame(
                    {
                        context.date_col: timestamps,
                        context.entity_col: entities,
                        "fold_id": int(split["fold"]),
                        "y_score": prediction,
                        "y_true": actual,
                        "config": context.config["config_name"],
                        "epoch": value,
                    }
                )
            )
    if not frames:
        raise ValueError("locked sequence fitted state produced no predictions")
    return pl.concat(frames).sort(context.entity_col, context.date_col, "fold_id", "epoch")


def _reconstruct_darts_predictions(
    model_root: Path,
    context: SequenceResearchContext,
    computation: dict[str, Any],
    case_study: str,
) -> pl.DataFrame:
    from case_studies.utils.darts_forecasting import (
        _attach_darts_target,
        _attach_expected_periods,
        _overlay_fold_temporal_features,
        _parse_label_horizon,
        _predict_fold,
        _prepare_fold_series,
        _resolve_chunk_lengths,
        darts_checkpoint_path,
        load_darts_checkpoint,
    )

    config = context.config
    input_chunk_length, output_chunk_length = _resolve_chunk_lengths(
        config, _parse_label_horizon(context.label_col)
    )
    # Same pipeline and same order as darts_validation_keys: expected periods first,
    # because the lagged-label target is built from them, then _attach_darts_target,
    # which dispatches on params.darts_target. Calling _attach_base_target directly
    # built the one-period target for every config, so replaying a lagged-label
    # checkpoint scored it against a target the fit never saw.
    dataset = _attach_expected_periods(
        context.dataset_pd.copy(),
        date_col=context.date_col,
        calendar_id=str(computation["preprocessing"]["calendar_id"]),
        case_study=case_study,
    )
    dataset = _attach_darts_target(
        dataset,
        case_study=case_study,
        date_col=context.date_col,
        entity_col=context.entity_col,
        label_col=context.label_col,
        config=config,
    )
    frames = []
    has_temporal = bool(
        context.temporal_by_fold is not None
        and context.temporal_keys
        and context.temporal_feature_names
    )
    for split in context.splits:
        fold_dataset = (
            _overlay_fold_temporal_features(
                dataset,
                split,
                context.date_col,
                context.temporal_by_fold,
                list(context.temporal_keys),
                list(context.temporal_feature_names),
            )
            if has_temporal
            else dataset
        )
        fold_series = _prepare_fold_series(
            fold_dataset,
            split,
            list(context.feature_names),
            context.label_col,
            context.date_col,
            context.entity_col,
            input_chunk_length,
            output_chunk_length,
        )
        for checkpoint in computation["checkpoint_schedule"]:
            value = int(checkpoint["value"])
            model, metadata = load_darts_checkpoint(
                darts_checkpoint_path(
                    model_root,
                    config["config_name"],
                    int(split["fold"]),
                    value,
                ),
                device=str(computation["numerics"]["device"]),
            )
            expected_metadata = {
                "config_name": config["config_name"],
                "fold": int(split["fold"]),
                "checkpoint_kind": "epoch",
                "checkpoint_value": value,
            }
            if metadata != expected_metadata:
                raise ValueError("locked Darts checkpoint metadata changed")
            frame = _predict_fold(
                model,
                fold_series,
                int(split["fold"]),
                context.date_col,
                context.entity_col,
                output_chunk_length,
            ).with_columns(
                pl.lit(config["config_name"]).alias("config"),
                pl.lit(value).alias("epoch"),
            )
            frames.append(frame)
    if not frames or any(frame.is_empty() for frame in frames):
        raise ValueError("locked Darts fitted state produced incomplete predictions")
    return pl.concat(frames).sort(context.entity_col, context.date_col, "fold_id", "epoch")


def _reconstruct_sequence_predictions(
    model_root: Path,
    context: SequenceResearchContext,
    computation: dict[str, Any],
    case_study: str,
) -> dict[str, Any]:
    _configure_sequence_runtime(computation["numerics"])
    checkpoints = tuple(int(item["value"]) for item in computation["checkpoint_schedule"])
    fold_ids = tuple(int(split["fold"]) for split in context.splits)
    if context.config.get("library") == "darts":
        from case_studies.utils.darts_forecasting import validate_darts_checkpoint_population

        validate_darts_checkpoint_population(
            model_root,
            config_name=context.config["config_name"],
            fold_ids=fold_ids,
            checkpoints=checkpoints,
            architecture=context.config["params"]["architecture"],
        )
        predictions = _reconstruct_darts_predictions(model_root, context, computation, case_study)
    else:
        from case_studies.utils.deep_model_state import validate_deep_checkpoint_population

        validate_deep_checkpoint_population(
            model_root,
            config_name=context.config["config_name"],
            fold_ids=fold_ids,
            checkpoints=checkpoints,
            architecture=context.config["params"]["architecture"],
        )
        predictions = _reconstruct_pytorch_predictions(model_root, context, computation)
    return {"all_predictions": predictions, "grid_results": []}


def _publish_sequence_predictions(
    study: Study,
    computation: dict[str, Any],
    context: SequenceResearchContext,
    training,
    result: dict[str, Any],
) -> tuple:
    """Register one prediction set per published checkpoint under the expected key names."""
    prediction_results = []
    # A locked holdout run publishes only the checkpoint its lock selected, on the
    # holdout split. Every other run publishes the whole declared schedule as
    # validation, which is what published_checkpoints being None means.
    published = context.published_checkpoints or tuple(
        int(item["value"]) for item in computation["checkpoint_schedule"]
    )
    for checkpoint in published:
        predictions = (
            result["all_predictions"]
            .filter(
                (pl.col("config") == context.config["config_name"])
                & (pl.col("epoch") == checkpoint)
            )
            .drop("config", "epoch")
            .rename({"fold_id": "fold", "y_true": "actual", "y_score": "prediction"})
        )
        # The expected keys are the internal contract and always name the entity
        # `symbol`; the runner emits the reader-facing key the case study uses.
        if context.entity_col != "symbol":
            predictions = predictions.rename({context.entity_col: "symbol"})
        prediction_results.append(
            study.results.publish_predictions(
                training,
                checkpoint_kind="epoch",
                checkpoint_value=int(checkpoint),
                split=context.prediction_split,
                predictions=predictions,
                expected_keys=context.expected_keys,
                task_type="regression",
                class_values=None,
                label=context.label_col,
            )
        )
    return tuple(prediction_results)


def run_resolved_request(
    study: Study,
    spec: dict[str, Any],
    context: SequenceResearchContext,
):
    from case_studies.research.models import ModelRun

    cached = _cached_sequence_run(study, spec, context)
    if cached is not None:
        return cached
    computation = spec["computation"]
    training = study.results.register_training(
        spec,
        execution_tier=spec["execution_tier"],
        runtime_provenance=context.runtime_provenance,
    )
    train_dir = training.root / "run_log" / "training" / training.hash
    model_dir = train_dir / "models"
    reused_fitted_state = model_dir.exists()
    if reused_fitted_state:
        result = _reconstruct_sequence_predictions(
            model_dir, context, computation, study.case_study
        )
    else:
        staging = train_dir / f".models.{uuid.uuid4().hex}.tmp"
        _configure_sequence_runtime(computation["numerics"])
        try:
            result = run_dl_cv(
                context.dataset_pd,
                list(context.splits),
                configs=[context.config],
                n_features=len(context.feature_names),
                feature_names=list(context.feature_names),
                label_col=context.label_col,
                date_col=context.date_col,
                entity_col=context.entity_col,
                device=computation["numerics"]["device"],
                save_dir=train_dir / "diagnostics",
                max_train_sequences=context.max_train_sequences,
                register=False,
                case_study=study.case_study,
                temporal_by_fold=context.temporal_by_fold,
                temporal_keys=list(context.temporal_keys),
                temporal_feature_names=list(context.temporal_feature_names),
                checkpoint_root=staging,
                strict=True,
                seed=int(spec["seed"]),
                num_threads=int(computation["numerics"]["num_threads"]),
            )
            os.replace(staging, model_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    prediction_results = _publish_sequence_predictions(
        study, computation, context, training, result
    )
    runtime_path = train_dir / "runtime.json"
    if runtime_path.exists() and not reused_fitted_state:
        runtime = json.loads(runtime_path.read_text())
        if result["grid_results"]:
            runtime["elapsed_s"] = sum(
                float(row.get("elapsed_s", 0.0)) for row in result["grid_results"]
            )
        runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    return ModelRun(training=training, predictions=prediction_results)


def validate_locked_run(
    study: Study,
    spec: dict[str, Any],
    context: SequenceResearchContext,
    run: ModelRun,
) -> str:
    """Validate selected predictions and the persisted sequence checkpoint population."""
    from case_studies.utils.registry import training_hash_from_spec
    from case_studies.utils.registry.specs import canonical_json

    if run.training.hash != training_hash_from_spec(spec) or len(run.predictions) != 1:
        raise ValueError("locked sequence run has the wrong training or prediction identity")
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
        raise ValueError("locked sequence run published the wrong checkpoint")
    # prediction.load() returns what was published, and publishing renames the entity to
    # `symbol`; the reconstruction still carries the reader key, so bring it to the
    # published contract before comparing rather than sorting a column that is not there.
    published = prediction.load().sort("symbol", "timestamp", "fold")
    reopened = _cached_sequence_run(study, spec, context)
    if reopened is None or reopened.predictions[0].hash != prediction.hash:
        raise ValueError("locked sequence fitted state cannot be reused exactly")
    model_root = run.training.root / "run_log" / "training" / run.training.hash / "models"
    reconstructed_all = _reconstruct_sequence_predictions(
        model_root,
        context,
        spec["computation"],
        study.case_study,
    )["all_predictions"]
    reconstructed = (
        reconstructed_all.filter(
            (pl.col("config") == context.config["config_name"]) & (pl.col("epoch") == selected[0])
        )
        .drop("config", "epoch")
        .rename({"fold_id": "fold", "y_true": "actual", "y_score": "prediction"})
    )
    if context.entity_col != "symbol":
        reconstructed = reconstructed.rename({context.entity_col: "symbol"})
    reconstructed = reconstructed.sort("symbol", "timestamp", "fold")
    key_columns = ["symbol", "timestamp", "fold"]
    value_columns = ["prediction", "actual"]
    if not reconstructed.select(key_columns).equals(
        published.select(key_columns)
    ) or not np.allclose(
        reconstructed.select(value_columns).to_numpy(),
        published.select(value_columns).to_numpy(),
        rtol=1e-7,
        atol=1e-7,
        equal_nan=False,
    ):
        raise ValueError("locked sequence fitted state does not reproduce published predictions")
    files = {
        str(path.relative_to(model_root)): _sha256(path)
        for path in sorted(model_root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError("locked sequence run has no fitted state")
    return hashlib.sha256(canonical_json(files).encode()).hexdigest()


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


def build_arch_kwargs(cfg: dict[str, Any], n_features: int, lookback: int) -> dict[str, Any]:
    """Extract architecture constructor kwargs from a config dict.

    Takes the YAML ``params`` dict, removes ``architecture`` and ``lookback``
    (metadata/training fields), and injects runtime dimensions (n_features,
    lookback) as the architecture's constructor expects them.
    """
    params = dict(cfg.get("params", {}))
    params.pop("architecture", None)
    params.pop("decision_cadence", None)
    params.pop("lookback", None)

    dim_vals = {"n_features": n_features, "lookback": lookback}
    arch = cfg["params"].get("architecture", resolve_arch_name(cfg["config_name"]))
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
    state_callback: Callable[[int, nn.Module], None] | None = None,
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
            if state_callback is not None:
                state_callback(epoch, model)
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


def resolve_arch_name(config_name: str) -> str:
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
    identity_params: dict[str, Any] | None = None,
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
        spec_extra_params=identity_params,
    )


def _decision_time_checkpoint_metrics(
    frame: pl.DataFrame,
    *,
    date_col: str,
    entity_col: str,
) -> dict[str, float | int]:
    """Score one pooled checkpoint with equal weight per decision timestamp."""
    stats = cross_sectional_ic(
        frame,
        frame,
        pred_col="y_score",
        ret_col="y_true",
        date_col=date_col,
        entity_col=entity_col,
        min_obs=5,
    )
    return {
        "ic_mean": float(stats["ic_mean"]),
        "ic_std": float(stats["ic_std"]),
        "ic_n_days": int(stats["n_periods"]),
    }


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
    identity_params: dict[str, Any] | None = None,
    input_data_spec: dict[str, Any] | None = None,
    checkpoint_root: Path | None = None,
    strict: bool = False,
    seed: int = RANDOM_SEED,
    num_threads: int = 8,
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
    from case_studies.utils.darts_forecasting import (
        darts_training_identity,
        run_darts_cv,
        uses_darts_backend,
    )

    _configure_sequence_runtime(
        _sequence_runtime_spec(device, seed=int(seed), num_threads=int(num_threads))
    )

    cadences = {cfg.get("params", {}).get("decision_cadence") for cfg in configs}
    if len(cadences) > 1:
        raise ValueError("sequence configurations with different decision cadences must run apart")
    decision_cadence = next(iter(cadences), None)
    dataset_pd = _select_sequence_observations(
        dataset_pd,
        date_col=date_col,
        cadence=decision_cadence,
        calendar=None,
    )
    assert isinstance(dataset_pd, pd.DataFrame)

    if register and save_dir is None:
        raise ValueError(
            "register=True requires save_dir for incremental prediction saves. "
            "Pass save_dir=CASE_DIR / 'run_log' / 'training' / 'deep_learning'"
        )

    cached_result = None

    def _config_identity_params(cfg: dict[str, Any]) -> dict[str, Any] | None:
        params = dict(identity_params or {})
        if input_data_spec is not None:
            if uses_darts_backend([cfg]):
                if case_study is None:
                    raise ValueError("Darts identity requires case_study")
                params.update(
                    darts_training_identity(
                        cfg,
                        label_col,
                        case_study=case_study,
                        input_data_spec=input_data_spec,
                        max_train_sequences=max_train_sequences,
                    )
                )
            else:
                params.update(
                    {
                        "batch_size": cfg.get("batch_size", 2048),
                        "input_data_spec": input_data_spec,
                        "lookback": cfg.get("params", {}).get("lookback", 60),
                        "max_train_sequences": max_train_sequences,
                    }
                )
        return params or None

    from case_studies.utils.registry import build_training_spec

    training_specs = {
        cfg["config_name"]: build_training_spec(
            cfg["family"],
            cfg["config_name"],
            label_col,
            n_folds=len(splits),
            n_epochs=cfg.get("n_epochs"),
            extra_params=_config_identity_params(cfg),
        )
        for cfg in configs
    }

    # Filter out configs whose training_hash is already complete (unless
    # force_retrain). Fold-major training can't skip individual configs
    # mid-fold, so the filter happens BEFORE the fold loop starts.
    if register and case_study and not force_retrain:
        from case_studies.utils.registry import (
            load_prediction_sets,
            training_hash_from_spec,
            training_run_status,
        )

        pending_configs = []
        cached_configs = []
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
                        from case_studies.utils.deep_model_state import declared_epoch_checkpoints

                        checkpoint_values = declared_epoch_checkpoints(
                            int(cfg.get("n_epochs", 100)),
                            int(cfg.get("checkpoint_interval", 5)),
                        )
                        fold_ids = tuple(int(split["fold"]) for split in splits)
                        if uses_darts_backend([cfg]):
                            from case_studies.utils.darts_forecasting import (
                                validate_darts_checkpoint_population,
                            )

                            validate_darts_checkpoint_population(
                                checkpoint_root,
                                config_name=cfg["config_name"],
                                fold_ids=fold_ids,
                                checkpoints=checkpoint_values,
                                architecture=cfg["params"]["architecture"],
                            )
                        else:
                            from case_studies.utils.deep_model_state import (
                                validate_deep_checkpoint_population,
                            )

                            validate_deep_checkpoint_population(
                                checkpoint_root,
                                config_name=cfg["config_name"],
                                fold_ids=fold_ids,
                                checkpoints=checkpoint_values,
                                architecture=resolve_arch_name(cfg["config_name"]),
                            )
                    print(
                        f"  SKIP {cfg['config_name']:25s}  ({status.summary()}, split={prediction_split})"
                    )
                    cached_configs.append(cfg)
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

        if cached_configs:
            cached_result = rebuild_cv_result_from_registry(
                case_study,
                cached_configs,
                label_col=label_col,
                n_folds=len(splits),
                prediction_split=prediction_split,
                date_col=date_col,
                entity_col=entity_col,
                training_specs=training_specs,
            )
        if not pending_configs:
            print("All configs already complete - rebuilt results from the registry.")
            assert cached_result is not None
            return cached_result
        configs = pending_configs

    if register and case_study and force_retrain:
        from case_studies.utils.registry import (
            build_training_spec,
            clear_prediction_sets,
            training_hash_from_spec,
        )

        for cfg in configs:
            spec = build_training_spec(
                cfg["family"],
                cfg["config_name"],
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
                    f"checkpoint(s) for {cfg['config_name']}"
                )

    if uses_darts_backend(configs):
        fresh_result = run_darts_cv(
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
            identity_params=identity_params,
            input_data_spec=input_data_spec,
            temporal_by_fold=temporal_by_fold,
            temporal_keys=temporal_keys,
            temporal_feature_names=temporal_feature_names,
            checkpoint_root=checkpoint_root,
            strict=strict,
        )
        if cached_result is not None:
            return combine_cv_results(
                [cached_result, fresh_result],
                date_col=date_col,
                entity_col=entity_col,
            )
        return fresh_result

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for deep learning, but CUDA is unavailable")
    torch_device = torch.device(device)
    if selected_folds:
        selected = {int(fold) for fold in selected_folds}
        splits = [split for split in splits if int(split["fold"]) in selected]
        print(f"Selected folds: {sorted(selected)}")
        if not splits:
            raise ValueError(f"No splits matched selected_folds={selected_folds}")

    seed_everything(seed)

    # Extract lookback from configs (must be uniform for fold-major sequencing)
    lookback = configs[0].get("params", {}).get("lookback", 60)
    calendar_id = None
    if case_study:
        from utils.cv_splits import make_walk_forward_config

        calendar_id = make_walk_forward_config(case_study, date_col=date_col).calendar_id

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
    expected_fold_ids: list[int] = []

    _has_fold_temporal = temporal_by_fold is not None and temporal_keys and temporal_feature_names

    for split in splits:
        seed_everything(seed + split["fold"])

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
            calendar_id=calendar_id,
        )

        if fold_info["train_sequences"] < 100 or fold_info["val_sequences"] < 50:
            if strict:
                raise ValueError(
                    f"Fold {split['fold']} is too small: "
                    f"train={fold_info['train_sequences']}, val={fold_info['val_sequences']}"
                )
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
        expected_fold_ids.append(int(split["fold"]))
        print("    datasets ready")

        # Train ALL configs on this fold before freeing fold data
        for cfg in configs:
            config_name = cfg["config_name"]
            cfg_n_epochs = cfg.get("n_epochs", 100)
            cfg_batch_size = cfg.get("batch_size", 2048)
            cfg_checkpoint = cfg.get("checkpoint_interval", 5)
            arch_name = cfg["params"].get("architecture", resolve_arch_name(config_name))
            arch_kwargs = build_arch_kwargs(cfg, n_features, lookback)

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

            train_kwargs: dict[str, Any] = {}
            if checkpoint_root is not None:
                from case_studies.utils.deep_model_state import (
                    deep_checkpoint_path,
                    write_deep_checkpoint,
                )

                if train_store.feature_mean is None or train_store.feature_scale is None:
                    raise ValueError("sequence fold has no fitted preprocessing state")
                preprocessing = {
                    "feature_names": list(feature_names),
                    "mean": train_store.feature_mean.copy(),
                    "scale": train_store.feature_scale.copy(),
                    "lookback": int(lookback),
                }

                def persist_state(
                    epoch: int,
                    fitted_model: nn.Module,
                    *,
                    _fold: int = int(split["fold"]),
                    _config_name: str = config_name,
                    _architecture: str = arch_name,
                    _model_kwargs: dict[str, Any] = arch_kwargs,
                    _preprocessing: dict[str, Any] = preprocessing,
                ) -> None:
                    write_deep_checkpoint(
                        deep_checkpoint_path(checkpoint_root, _config_name, _fold, epoch),
                        model=fitted_model,
                        architecture=_architecture,
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

            checkpoint_ics, checkpoint_preds, epoch_losses = _train_one_config(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=cfg_n_epochs,
                checkpoint_interval=cfg_checkpoint,
                device=torch_device,
                checkpoint_callback=_on_checkpoint,
                epoch_callback=_on_epoch,
                **train_kwargs,
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

        epoch_scores: list[tuple[int, float, float, int]] = []
        if cfg_preds.height > 0:
            for epoch in sorted(cfg_preds["epoch"].unique().to_list()):
                ep_df = cfg_preds.filter(pl.col("epoch") == epoch)
                fold_ids = sorted(ep_df["fold_id"].unique().to_list())
                if fold_ids != expected_fold_ids:
                    continue
                metrics = _decision_time_checkpoint_metrics(
                    ep_df,
                    date_col=date_col,
                    entity_col=entity_col,
                )
                ic_mean = float(metrics["ic_mean"])
                ic_std = float(metrics["ic_std"])
                ic_n_days = int(metrics["ic_n_days"])
                all_curves.append(
                    {
                        "config": config_name,
                        "epoch": epoch,
                        "ic_mean": ic_mean,
                        "ic_std": ic_std,
                        "ic_n_days": ic_n_days,
                    }
                )
                complete_prediction_frames.append(ep_df)
                epoch_scores.append((epoch, ic_mean, ic_std, ic_n_days))

        if epoch_scores:
            full_coverage = max(item[3] for item in epoch_scores)
            eligible_scores = [item for item in epoch_scores if item[3] == full_coverage]
            best_cp, best_ic_val, _best_ic_std, best_ic_n_days = max(
                eligible_scores, key=lambda item: item[1]
            )
        else:
            best_cp = 0
            best_ic_val = float("nan")
            best_ic_n_days = 0

        config_results.append(
            {
                "config_name": config_name,
                "best_epoch": best_cp,
                "best_ic": best_ic_val,
                "ic_n_days": best_ic_n_days,
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

        if checkpoint_root is not None:
            from case_studies.utils.deep_model_state import (
                declared_epoch_checkpoints,
                validate_deep_checkpoint_population,
            )

            validate_deep_checkpoint_population(
                checkpoint_root,
                config_name=config_name,
                fold_ids=tuple(expected_fold_ids),
                checkpoints=declared_epoch_checkpoints(
                    int(cfg.get("n_epochs", 100)),
                    int(cfg.get("checkpoint_interval", 5)),
                ),
                architecture=resolve_arch_name(config_name),
            )

        # Incremental registration: persist every complete checkpoint as soon as
        # aggregation finishes. Registering only the raw-IC peak would prevent a
        # reader from applying the checkpoint-level coverage guard when that peak
        # has undefined daily IC on part of the validation surface.
        if register and case_study and epoch_scores and cfg_preds.height > 0:
            try:
                from case_studies.utils.registry import register_prediction_set

                arch = resolve_arch_name(config_name)
                cfg_curves_df = pl.DataFrame([c for c in all_curves if c["config"] == config_name])
                epoch_ic = {epoch: ic for epoch, ic, _std, _days in epoch_scores}
                epochs = sorted(cfg_preds["epoch"].unique().to_list())
                first_ep = best_cp if best_cp in epochs else epochs[0]
                first_slice = cfg_preds.filter(pl.col("epoch") == first_ep).drop("config", "epoch")
                t_hash = _register_dl_config(
                    case_study=case_study,
                    label=label_col,
                    config_name=config_name,
                    architecture=arch,
                    n_epochs=cfg.get("n_epochs"),
                    best_epoch=int(first_ep),
                    lookback=lookback,
                    n_folds=len(splits),
                    ic_mean=epoch_ic.get(first_ep, best_ic_val),
                    predictions=first_slice,
                    notebook=notebook,
                    learning_curves=cfg_curves_df if cfg_curves_df.height > 0 else None,
                    started_at=acc.get("started_at"),
                    elapsed_s=acc.get("elapsed_s"),
                    prediction_split=prediction_split,
                    identity_params=_config_identity_params(cfg),
                )
                for epoch, _ic, _epoch_std, _days in epoch_scores:
                    if epoch == first_ep:
                        continue
                    epoch_preds = cfg_preds.filter(pl.col("epoch") == epoch).drop("config", "epoch")
                    register_prediction_set(
                        case_study,
                        training_hash=t_hash,
                        checkpoint_value=int(epoch),
                        checkpoint_kind="epoch",
                        split=prediction_split,
                        predictions=epoch_preds,
                        metrics={"ic_mean": epoch_ic[epoch]},
                    )
                print(
                    f"    registered {config_name} incrementally "
                    f"({len(epoch_scores)} per-epoch slices)"
                )
            except Exception as exc:
                if strict:
                    raise
                print(f"    WARN: incremental registration failed for {config_name}: {exc}")

    del config_acc
    gc.collect()

    complete_predictions = (
        pl.concat(complete_prediction_frames, how="diagonal_relaxed")
        if complete_prediction_frames
        else pl.DataFrame()
    )

    learning_curves = pl.DataFrame(all_curves) if all_curves else pl.DataFrame()
    training_log_df = pl.DataFrame(training_log) if training_log else pl.DataFrame()
    fresh_result = assemble_cv_result(
        learning_curves,
        complete_predictions,
        date_col=date_col,
        entity_col=entity_col,
        metadata={row["config_name"]: row for row in config_results},
        training_log=training_log_df,
    )
    predictions = fresh_result["predictions"]

    print(
        f"\n  Best: {fresh_result['best_config_name']} @ epoch "
        f"{fresh_result['best_epoch']} (IC={fresh_result['best_ic']:+.4f})"
    )

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

    if cached_result is not None:
        return combine_cv_results(
            [cached_result, fresh_result],
            date_col=date_col,
            entity_col=entity_col,
        )
    return fresh_result


# Deprecated private aliases. Thirty notebook cells import these names with their leading
# underscore, which is what made them look private in the first place; each is a
# cross-module interface and is now public. The aliases keep those callers working until
# each notebook is re-executed with the public name, and go when the last one moves.
_resolve_arch_name = resolve_arch_name
_build_arch_kwargs = build_arch_kwargs
