from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

from case_studies.research.contracts import ExecutionTier
from case_studies.research.cv import require_fold_scoped_temporal_compatibility
from case_studies.research.identity import ResolvedSpec
from case_studies.research.models import ModelRun
from case_studies.research.recovery import ExecutionAttempt, ExecutionLedger
from case_studies.research.results import PredictionResult, Result, TrainingResult
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec
from case_studies.utils.registry.specs import canonical_json
from utils.modeling import (
    load_modeling_dataset,
    prepare_cv_folds,
    prepare_single_fold,
    resolve_linear_params,
)

if TYPE_CHECKING:
    from case_studies.research.workspace import Study


# Coordinate descent and the BLAS kernels behind these fits reduce in thread-order, so a fit is
# a deterministic function of the pool rather than of the data alone. Measured on
# crypto_perps_funding lasso_f0.08: identical training_hash 51c3b31a83a2 under both arms, with
# prediction digests 37352b2ff14a4b6a uncapped (pools 16/24) against c81ca6b5302e1dc2 capped.
# Fixed rather than derived: a value like -1 varies with the host, so a result would not be
# identity-stable across the readers' hardware.
LINEAR_THREAD_LIMIT = 1

_MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "LogisticRegression": LogisticRegression,
}
_PREVIEW_FIELDS = {"folds", "max_symbols", "train_sample_frac"}


def _module_source(module_name: str) -> Path:
    source = importlib.import_module(module_name).__file__
    if source is None:
        raise RuntimeError(f"module {module_name!r} has no source file")
    return Path(source)


_SOURCE_FILES = (Path(__file__), _module_source("utils.modeling"))


@dataclass(frozen=True)
class LinearContext:
    folds: tuple[dict[str, Any], ...]
    fold_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    label_col: str
    eval_label_col: str | None
    date_col: str
    entity_col: str
    task_type: str
    class_values: tuple[Any, ...]
    expected_keys: pl.DataFrame
    runtime_provenance: dict[str, Any]
    prediction_split: str = "validation"
    checkpoint_kind: str = "final"
    checkpoint_value: int | None = None
    immutable_recovery: bool = False


@dataclass
class _BatchCandidate:
    index: int
    request: dict[str, Any]
    config: dict[str, Any]
    effective_params: dict[str, dict[str, Any]]
    spec: dict[str, Any] | None = None
    context: LinearContext | None = None
    training: TrainingResult | None = None
    ledger: ExecutionLedger | None = None
    attempt: ExecutionAttempt | None = None
    frames: list[pl.DataFrame] = field(default_factory=list)
    reused_folds: list[int] = field(default_factory=list)
    fitted_folds: list[int] = field(default_factory=list)
    fit_elapsed_s: float = 0.0
    started_at_s: float = 0.0
    result: ModelRun | None = None
    error: Exception | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity() -> dict[str, str]:
    return {str(path.name): _sha256(path) for path in _SOURCE_FILES}


def _runtime_identity() -> dict[str, str]:
    return {
        "joblib": importlib.metadata.version("joblib"),
        "numpy": importlib.metadata.version("numpy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
    }


def _runtime_provenance(study: Study) -> dict[str, Any]:
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
        "entry_point": "case_studies.utils.linear",
        "packages": _runtime_identity(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_commit": commit,
    }


def _normalize_folds(splits: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    fields = ("fold", "train_start", "train_end", "val_start", "val_end")
    return tuple(
        {
            key: int(split[key]) if key == "fold" else str(split[key])
            for key in fields
            if split.get(key) is not None
        }
        for split in splits
    )


def _select_splits(
    mds,
    request: dict[str, Any],
    label_timeline: pl.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cv = request.get("cv")
    if cv is None:
        splits = list(mds.splits)
        normalized = _normalize_folds(splits)
        cv_record = {
            "request": {"source": "case_study_default"},
            "folds": list(normalized),
            "identity": value_digest(pl.DataFrame(list(normalized))),
        }
    else:
        resolved = cv.resolve(label_timeline, date_col=mds.date_col)
        splits = [dict(fold) for fold in resolved.normalized_folds]
        cv_record = resolved.as_dict()

    reductions = request["preview_reductions"]
    requested_folds = reductions.get("folds")
    if requested_folds is not None:
        selected = {int(fold) for fold in requested_folds}
        splits = [split for split in splits if int(split["fold"]) in selected]
        if {int(split["fold"]) for split in splits} != selected:
            raise ValueError("preview fold reduction refers to an unavailable fold")
        cv_record = {**cv_record, "preview_folds": sorted(selected)}
    if not splits:
        raise ValueError("model request resolved no cross-validation folds")
    return splits, cv_record


def _expected_keys(folds: list[dict[str, Any]], entity_col: str, date_col: str) -> pl.DataFrame:
    frames = [
        pl.from_pandas(fold["meta"][[entity_col, date_col]])
        .with_columns(pl.lit(int(fold["fold"]), dtype=pl.Int64).alias("fold"))
        .rename({entity_col: "symbol", date_col: "timestamp"})
        .select("symbol", "timestamp", "fold")
        for fold in folds
    ]
    expected = pl.concat(frames).sort("symbol", "timestamp", "fold")
    if expected.n_unique(["symbol", "timestamp", "fold"]) != expected.height:
        raise ValueError("linear request produced duplicate expected prediction keys")
    return expected


def _load_preset(config_name: str) -> dict[str, Any]:
    from case_studies.utils.registry.specs import load_preset

    config = load_preset("linear", config_name)
    if config.get("family") != "linear":
        raise ValueError(f"preset {config_name!r} belongs to {config.get('family')!r}")
    model_class = config.get("model_class")
    if model_class not in _MODEL_CLASSES:
        raise ValueError(
            f"unsupported linear model class {model_class!r}; expected {sorted(_MODEL_CLASSES)}"
        )
    return config


def _effective_params(config: dict[str, Any], overrides: dict[str, Any], folds) -> dict[str, dict]:
    merged = {**dict(config.get("params") or {}), **overrides}
    cls = _MODEL_CLASSES[config["model_class"]]
    supported = cls().get_params(deep=True)
    if "random_state" in supported and "random_state" not in merged:
        merged["random_state"] = 42
    candidate = {**config, "params": merged}
    effective = {}
    for fold in folds:
        resolved = resolve_linear_params(candidate, fold["X_train"], fold["y_train"])
        try:
            model = cls(**resolved)
        except TypeError as exc:
            raise ValueError(
                f"invalid parameters for {config['model_class']}: {sorted(resolved)}"
            ) from exc
        effective[str(int(fold["fold"]))] = model.get_params(deep=True)
    return effective


def _fixed_effective_params(
    config: dict[str, Any],
    overrides: dict[str, Any],
    fold_ids: tuple[int, ...],
) -> dict[str, dict[str, Any]] | None:
    merged = {**dict(config.get("params") or {}), **overrides}
    if "alpha_frac" in merged:
        return None
    cls = _MODEL_CLASSES[config["model_class"]]
    supported = cls().get_params(deep=True)
    if "random_state" in supported and "random_state" not in merged:
        merged["random_state"] = 42
    try:
        resolved = cls(**merged).get_params(deep=True)
    except TypeError as exc:
        raise ValueError(
            f"invalid parameters for {config['model_class']}: {sorted(merged)}"
        ) from exc
    return {str(fold_id): dict(resolved) for fold_id in fold_ids}


def _boundary_literal(dtype: pl.DataType, value: Any) -> pl.Expr:
    raw = str(value)
    if dtype == pl.Date:
        return pl.lit(datetime.fromisoformat(raw[:10]).date())
    if dtype.base_type() == pl.Datetime:
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            parsed = datetime.fromisoformat(f"{raw[:10]}T00:00:00")
        return pl.lit(parsed).cast(dtype)
    return pl.lit(value).cast(dtype)


def _expected_keys_from_dataset(
    dataset: pl.DataFrame,
    splits: list[dict[str, Any]],
    *,
    entity_col: str,
    date_col: str,
    label_col: str,
) -> pl.DataFrame:
    dtype = dataset.schema[date_col]
    label_valid = pl.col(label_col).is_not_null()
    if dataset.schema[label_col] in {pl.Float32, pl.Float64}:
        label_valid &= pl.col(label_col).is_not_nan()
    frames = []
    for split in splits:
        val_start = split.get("val_start", split.get("test_start"))
        val_end = split.get("val_end", split.get("test_end"))
        frame = (
            dataset.filter(
                (pl.col(date_col) >= _boundary_literal(dtype, val_start))
                & (pl.col(date_col) <= _boundary_literal(dtype, val_end))
                & label_valid
            )
            .select(entity_col, date_col)
            .with_columns(pl.lit(int(split["fold"]), dtype=pl.Int64).alias("fold"))
            .rename({entity_col: "symbol", date_col: "timestamp"})
            .select("symbol", "timestamp", "fold")
        )
        if frame.is_empty():
            raise ValueError(f"linear request produced no validation keys for fold {split['fold']}")
        frames.append(frame)
    expected = pl.concat(frames).sort("symbol", "timestamp", "fold")
    if expected.n_unique(["symbol", "timestamp", "fold"]) != expected.height:
        raise ValueError("linear request produced duplicate expected prediction keys")
    return expected


def _build_resolved_request(
    study: Study,
    request: dict[str, Any],
    *,
    label_ref,
    mds,
    splits: list[dict[str, Any]],
    cv_record: dict[str, Any],
    config: dict[str, Any],
    effective: dict[str, dict[str, Any]],
    expected: pl.DataFrame,
    folds: tuple[dict[str, Any], ...],
    runtime_provenance: dict[str, Any],
) -> tuple[dict[str, Any], LinearContext]:
    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    max_symbols = int(reductions.get("max_symbols", 0))
    train_sample_frac = float(reductions.get("train_sample_frac", 1.0))
    entity_col = mds.entity_cols[0]
    input_lineage = mds.input_lineage
    computation = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "task": {
            "type": mds.task_type,
            "class_values": list(mds.class_values),
            "continuous_eval_label": label_ref.definition.continuous_eval_label,
        },
        "cv": cv_record,
        "model": {
            "class": config["model_class"],
            "implementation": "scikit-learn",
            "objective": "classification" if mds.task_type == "classification" else "regression",
            "effective_params_by_fold": effective,
        },
        "preprocessing": {
            "imputer": {"class": "SimpleImputer", "strategy": "median"},
            "scaler": {"class": "StandardScaler", "with_mean": True, "with_std": True},
        },
        "numerics": {
            "thread_limit": LINEAR_THREAD_LIMIT,
            "deterministic_reduction": True,
        },
        "checkpoint_schedule": [{"kind": "final", "value": None}],
        "expected_prediction_keys": {
            "digest": value_digest(expected, tuple(expected.columns)),
            "n_rows": expected.height,
            "n_folds": expected.get_column("fold").n_unique(),
        },
        "input_data_spec": input_lineage,
        "sampling": {"train_sample_frac": train_sample_frac, "max_symbols": max_symbols},
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
    }
    if tier is ExecutionTier.PREVIEW:
        computation["preview_reductions"] = reductions
    spec = ResolvedSpec.create(
        family="linear",
        label=label_ref.name,
        seed=42,
        computation=computation,
        provenance=runtime_provenance,
        config_name=request["config_name"],
        execution_tier=tier.value,
    ).as_dict()
    context = LinearContext(
        folds=folds,
        fold_ids=tuple(int(split["fold"]) for split in splits),
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        eval_label_col=mds.eval_label_col,
        date_col=mds.date_col,
        entity_col=entity_col,
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        expected_keys=expected,
        runtime_provenance=runtime_provenance,
    )
    return spec, context


def _load_batch_base(
    study: Study,
    request: dict[str, Any],
    *,
    inputs: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    unknown_reductions = set(reductions) - _PREVIEW_FIELDS
    if unknown_reductions:
        raise ValueError(f"unsupported linear preview reductions: {sorted(unknown_reductions)}")
    study.require_writable()
    study.activate(tier)
    max_symbols = int(reductions.get("max_symbols", 0))
    train_sample_frac = float(reductions.get("train_sample_frac", 1.0))
    if not 0 < train_sample_frac <= 1:
        raise ValueError("train_sample_frac must be in (0, 1]")
    if inputs is None:
        label_ref = study.labels.get(request["label"], execution_tier=tier)
        mds = load_modeling_dataset(study.case_study, label_ref.name, max_symbols=max_symbols)
    else:
        label_ref, mds = inputs
    if mds.date_col != "timestamp" or not mds.entity_cols:
        raise ValueError("linear runner requires timestamp and an entity key")
    entity_col = mds.entity_cols[0]
    if entity_col not in {"product", "symbol"}:
        raise ValueError(f"linear runner does not support entity key {entity_col!r}")
    label_timeline = label_ref.load().select(mds.date_col).unique()
    splits, cv_record = _select_splits(mds, request, label_timeline)
    if (
        request.get("cv") is not None
        and mds.temporal_by_fold is not None
        and mds.temporal_keys
        and mds.temporal_feature_names
    ):
        require_fold_scoped_temporal_compatibility(splits, mds.temporal_artifact_splits)
    expected = _expected_keys_from_dataset(
        mds.dataset,
        splits,
        entity_col=entity_col,
        date_col=mds.date_col,
        label_col=mds.label_col,
    )
    return {
        "train_sample_frac": train_sample_frac,
        "label_ref": label_ref,
        "mds": mds,
        "splits": splits,
        "cv_record": cv_record,
        "expected": expected,
        "runtime_provenance": _runtime_provenance(study),
    }


def _input_compatibility_key(request: dict[str, Any]) -> tuple[str, str, int]:
    reductions = request["preview_reductions"]
    return (
        request["label"],
        request["execution_tier"],
        int(reductions.get("max_symbols", 0)),
    )


def _compatibility_key(request: dict[str, Any]) -> str:
    cv = request.get("cv")
    cv_value = asdict(cv) if cv is not None else None
    reductions = request["preview_reductions"]
    return canonical_json(
        {
            "label": request["label"],
            "execution_tier": request["execution_tier"],
            "cv": cv_value,
            "folds": reductions.get("folds"),
            "max_symbols": reductions.get("max_symbols", 0),
            "train_sample_frac": reductions.get("train_sample_frac", 1.0),
            "preprocessing": "median-imputer-standard-scaler/v1",
        }
    )


def resolve_model_request(study: Study, request: dict[str, Any]):
    base = _load_batch_base(study, request)
    mds = base["mds"]
    folds = prepare_cv_folds(
        mds.dataset.to_pandas(),
        base["splits"],
        mds.feature_names,
        mds.label_col,
        mds.date_col,
        mds.entity_cols[0],
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=mds.temporal_keys,
        temporal_feature_names=mds.temporal_feature_names,
        train_sample_frac=base["train_sample_frac"],
        eval_label_col=mds.eval_label_col,
    )
    if len(folds) != len(base["splits"]):
        raise ValueError("linear request did not prepare every declared fold")

    config = _load_preset(request["config_name"])
    effective = _effective_params(config, request["overrides"], folds)
    return _build_resolved_request(
        study,
        request,
        label_ref=base["label_ref"],
        mds=mds,
        splits=base["splits"],
        cv_record=base["cv_record"],
        config=config,
        effective=effective,
        expected=base["expected"],
        folds=tuple(folds),
        runtime_provenance=base["runtime_provenance"],
    )


def reconstruct_locked_request(
    study: Study,
    spec: dict[str, Any],
    *,
    checkpoint_kind: str,
    checkpoint_value: int | None,
):
    """Reconstruct a linear holdout fit without consulting a mutable preset."""
    from case_studies.research.models import (
        ResolvedModelRequest,
        locked_holdout_split,
        validate_locked_expected_keys,
    )

    study.require_writable()
    study.activate(ExecutionTier.CANONICAL)
    if spec.get("seed") != 42:
        raise ValueError("locked linear seed cannot be reproduced")
    computation = spec["computation"]
    sampling = computation.get("sampling")
    if sampling != {"train_sample_frac": 1.0, "max_symbols": 0}:
        raise ValueError("locked linear holdout requires an unreduced canonical dataset")
    label_ref = study.labels.get(spec["label"], execution_tier=ExecutionTier.CANONICAL)
    mds = load_modeling_dataset(study.case_study, label_ref.name, max_symbols=0)
    if mds.date_col != "timestamp" or not mds.entity_cols:
        raise ValueError("locked linear runner requires timestamp and an entity key")
    entity_col = mds.entity_cols[0]
    if entity_col not in {"product", "symbol"}:
        raise ValueError(f"locked linear runner does not support entity key {entity_col!r}")

    expected_inputs = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": mds.input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "input_data_spec": mds.input_lineage,
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
        "task": {
            "type": mds.task_type,
            "class_values": list(mds.class_values),
            "continuous_eval_label": label_ref.definition.continuous_eval_label,
        },
        "preprocessing": {
            "imputer": {"class": "SimpleImputer", "strategy": "median"},
            "scaler": {"class": "StandardScaler", "with_mean": True, "with_std": True},
        },
        "checkpoint_schedule": [{"kind": "final", "value": None}],
    }
    for name, expected_value in expected_inputs.items():
        if computation.get(name) != expected_value:
            raise ValueError(f"locked linear {name} does not match the available computation")
    if (checkpoint_kind, checkpoint_value) != ("final", None):
        raise ValueError("linear holdout supports only the final checkpoint")

    split = locked_holdout_split(spec, mds.dataset, mds.date_col, study.case_study)
    if mds.temporal_by_fold is not None and mds.temporal_keys and mds.temporal_feature_names:
        require_fold_scoped_temporal_compatibility([split], mds.temporal_artifact_splits)
    expected = _expected_keys_from_dataset(
        mds.dataset,
        [split],
        entity_col=entity_col,
        date_col=mds.date_col,
        label_col=mds.label_col,
    )
    validate_locked_expected_keys(spec, expected)
    fold = prepare_single_fold(
        mds.dataset,
        split,
        mds.feature_names,
        mds.label_col,
        mds.date_col,
        entity_col,
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=mds.temporal_keys,
        temporal_feature_names=mds.temporal_feature_names,
        train_sample_frac=1.0,
        eval_label_col=mds.eval_label_col,
    )
    if fold is None:
        raise ValueError("locked linear holdout fold could not be prepared")

    model = computation.get("model")
    if not isinstance(model, dict) or model.get("class") not in _MODEL_CLASSES:
        raise ValueError("locked linear model class cannot be reconstructed")
    fold_id = str(split["fold"])
    effective = model.get("effective_params_by_fold")
    if not isinstance(effective, dict) or set(effective) != {fold_id}:
        raise ValueError("locked linear model must declare parameters for the holdout fold")
    params = effective[fold_id]
    try:
        reconstructed = _MODEL_CLASSES[model["class"]](**params).get_params(deep=True)
    except TypeError as exc:
        raise ValueError("locked linear model parameters cannot be reconstructed") from exc
    if reconstructed != params:
        raise ValueError("locked linear model parameters do not reproduce exactly")
    expected_model = {
        "class": model["class"],
        "implementation": "scikit-learn",
        "objective": "classification" if mds.task_type == "classification" else "regression",
        "effective_params_by_fold": effective,
    }
    if model != expected_model:
        raise ValueError("locked linear model specification is unsupported")

    context = LinearContext(
        folds=(fold,),
        fold_ids=(int(split["fold"]),),
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        eval_label_col=mds.eval_label_col,
        date_col=mds.date_col,
        entity_col=entity_col,
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        expected_keys=expected,
        runtime_provenance=_runtime_provenance(study),
        prediction_split="holdout",
        checkpoint_kind=checkpoint_kind,
        checkpoint_value=checkpoint_value,
        immutable_recovery=True,
    )
    return ResolvedModelRequest(study, "linear", spec, context)


def _cached_run(study: Study, spec: dict[str, Any], context: LinearContext) -> ModelRun | None:
    training_hash = training_hash_from_spec(spec)
    prediction_hash = prediction_hash_from_parts(
        training_hash,
        context.checkpoint_value,
        context.prediction_split,
        checkpoint_kind=context.checkpoint_kind,
        identity_version=spec["identity_version"],
    )
    try:
        training = Result.open(
            study,
            training_hash,
            include_preview=spec["execution_tier"] == "preview",
        )
        prediction = Result.open(
            study,
            prediction_hash,
            include_preview=spec["execution_tier"] == "preview",
        )
    except KeyError:
        return None
    if not isinstance(training, TrainingResult) or not isinstance(prediction, PredictionResult):
        return None
    model_dir = training.root / "run_log" / "training" / training.hash / "models"
    if not prediction.complete:
        return None
    manifest = model_dir / "manifest.json"
    expected_files = {f"fold_{fold_id}.joblib" for fold_id in context.fold_ids}
    if not manifest.is_file():
        return None
    record = json.loads(manifest.read_text())
    if set(record.get("files") or {}) != expected_files:
        return None
    if any(_sha256(model_dir / name) != digest for name, digest in record["files"].items()):
        return None
    if context.immutable_recovery:
        ledger = ExecutionLedger(study, training.root)
        shard_dir = training.root / "run_log" / "training" / training.hash / "prediction_folds"
        for fold_id in context.fold_ids:
            params = spec["computation"]["model"]["effective_params_by_fold"][str(fold_id)]
            if not ledger.reusable_fold(
                training_hash=training.hash,
                candidate_identity=training.hash,
                fold_id=fold_id,
                fitted_state=model_dir / f"fold_{fold_id}.joblib",
                prediction_shard=shard_dir / f"fold_{fold_id}.parquet",
                resolved_settings=params,
            ):
                return None
    return ModelRun(
        training=training,
        predictions=(prediction,),
        diagnostics={
            "cache_hit": True,
            "reused_folds": sorted(context.fold_ids),
            "fitted_folds": [],
        },
    )


def _fold_predictions(model: Any, fold: dict[str, Any], context: LinearContext) -> np.ndarray:
    if context.task_type == "classification":
        predict_proba = getattr(model, "predict_proba", None)
        if not callable(predict_proba):
            raise ValueError("classification linear model must expose predict_proba")
        probabilities = predict_proba(fold["X_val"])
        predictions = probabilities @ np.asarray(sorted(context.class_values), dtype=np.float64)
    else:
        predictions = np.asarray(model.predict(fold["X_val"]), dtype=np.float64)
    fold_id = int(fold["fold"])
    if not np.isfinite(predictions).all() or np.nanstd(predictions) <= 1e-15:
        raise ValueError(f"linear fold {fold_id} produced non-finite or constant predictions")
    return predictions


def _prediction_frame(
    fold: dict[str, Any], predictions: np.ndarray, context: LinearContext
) -> pl.DataFrame:
    fold_id = int(fold["fold"])
    metadata = fold.get("meta_pl")
    if metadata is not None:
        frame = metadata.select(context.entity_col, context.date_col)
    else:
        frame = pl.from_pandas(fold["meta"][[context.entity_col, context.date_col]])
    frame = frame.with_columns(
        pl.lit(fold_id, dtype=pl.Int64).alias("fold"),
        pl.Series("prediction", predictions),
        pl.Series("actual", fold["y_val"]),
    )
    if fold.get("y_eval") is not None:
        frame = frame.with_columns(pl.Series("eval_actual", fold["y_eval"]))
    return frame.rename({context.date_col: "timestamp"}).with_columns(
        pl.col("timestamp").cast(context.expected_keys.schema["timestamp"])
    )


def _write_runtime_fields(runtime_path: Path, **fields: float) -> None:
    if not runtime_path.exists():
        return
    runtime = json.loads(runtime_path.read_text())
    runtime.update(fields)
    temporary = runtime_path.with_name(f".{runtime_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, runtime_path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit_or_reuse_fold(
    spec: dict[str, Any],
    context: LinearContext,
    training: TrainingResult,
    ledger: ExecutionLedger,
    fold: dict[str, Any],
) -> tuple[pl.DataFrame, bool, float]:
    fold_id = int(fold["fold"])
    params = spec["computation"]["model"]["effective_params_by_fold"][str(fold_id)]
    model_dir = training.root / "run_log" / "training" / training.hash / "models"
    shard_dir = training.root / "run_log" / "training" / training.hash / "prediction_folds"
    model_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    artifact = model_dir / f"fold_{fold_id}.joblib"
    shard = shard_dir / f"fold_{fold_id}.parquet"
    if ledger.reusable_fold(
        training_hash=training.hash,
        candidate_identity=training.hash,
        fold_id=fold_id,
        fitted_state=artifact,
        prediction_shard=shard,
        resolved_settings=params,
    ):
        return pl.read_parquet(shard), True, 0.0
    completed = ledger.fold_completion_exists(
        training_hash=training.hash,
        candidate_identity=training.hash,
        fold_id=fold_id,
    )
    if context.immutable_recovery and completed:
        raise ValueError(f"locked linear fold {fold_id} has conflicting persisted artifacts")
    if context.immutable_recovery and artifact.is_file() and shard.is_file():
        try:
            persisted = joblib.load(artifact)
            if persisted.get("feature_names") != context.feature_names:
                raise ValueError("feature identity changed")
            recovered = _prediction_frame(
                fold,
                _fold_predictions(persisted["model"], fold, context),
                context,
            )
            if not pl.read_parquet(shard).equals(recovered):
                raise ValueError("prediction shard changed")
            ledger.complete_fold(
                training_hash=training.hash,
                candidate_identity=training.hash,
                fold_id=fold_id,
                fitted_state=artifact,
                prediction_shard=shard,
                resolved_settings=params,
            )
            return recovered, True, 0.0
        except Exception as exc:
            raise ValueError(
                f"locked linear fold {fold_id} has conflicting uncommitted artifacts"
            ) from exc
    if context.immutable_recovery and (artifact.exists() or shard.exists()):
        incomplete = (
            training.root
            / "run_log"
            / "training"
            / training.hash
            / "incomplete_folds"
            / f"fold_{fold_id}.{uuid.uuid4().hex}"
        )
        incomplete.mkdir(parents=True)
        for path in (artifact, shard):
            if path.exists():
                os.replace(path, incomplete / path.name)

    started = time.perf_counter()
    cls = _MODEL_CLASSES[spec["computation"]["model"]["class"]]
    model = cls(**params)
    thread_limit = int(
        spec["computation"].get("numerics", {}).get("thread_limit", LINEAR_THREAD_LIMIT)
    )
    with threadpool_limits(limits=thread_limit):
        model.fit(fold["X_train"], fold["y_train"])
        predictions = _fold_predictions(model, fold, context)
    frame = _prediction_frame(fold, predictions, context)
    artifact_temp = model_dir / f".{artifact.name}.{uuid.uuid4().hex}.tmp"
    shard_temp = shard_dir / f".{shard.name}.{uuid.uuid4().hex}.tmp"
    try:
        joblib.dump(
            {
                "feature_names": context.feature_names,
                "model": model,
                "preprocessor": fold["preprocessor"],
            },
            artifact_temp,
        )
        frame.write_parquet(shard_temp)
        os.replace(artifact_temp, artifact)
        os.replace(shard_temp, shard)
        ledger.complete_fold(
            training_hash=training.hash,
            candidate_identity=training.hash,
            fold_id=fold_id,
            fitted_state=artifact,
            prediction_shard=shard,
            resolved_settings=params,
        )
    finally:
        artifact_temp.unlink(missing_ok=True)
        shard_temp.unlink(missing_ok=True)
    return frame, False, time.perf_counter() - started


def _write_model_manifest(training: TrainingResult, *, immutable: bool = False) -> None:
    model_dir = training.root / "run_log" / "training" / training.hash / "models"
    files = {path.name: _sha256(path) for path in sorted(model_dir.glob("fold_*.joblib"))}
    manifest = model_dir / "manifest.json"
    record = {"files": files, "schema_version": 1}
    if immutable and manifest.exists():
        if json.loads(manifest.read_text()) != record:
            raise ValueError("locked linear fitted-state manifest conflict")
        return
    manifest_temp = model_dir / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        manifest_temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(manifest_temp, manifest)
    finally:
        manifest_temp.unlink(missing_ok=True)


def _fit_or_reuse_predictions(
    spec: dict[str, Any],
    context: LinearContext,
    training: TrainingResult,
    ledger: ExecutionLedger,
) -> tuple[pl.DataFrame, list[int], list[int]]:
    prediction_frames = []
    reused_folds = []
    fitted_folds = []
    for fold in context.folds:
        fold_id = int(fold["fold"])
        frame, reused, _ = _fit_or_reuse_fold(spec, context, training, ledger, fold)
        prediction_frames.append(frame)
        if reused:
            reused_folds.append(fold_id)
        else:
            fitted_folds.append(fold_id)

    _write_model_manifest(training, immutable=context.immutable_recovery)
    predictions = pl.concat(prediction_frames).sort(context.entity_col, "timestamp", "fold")
    return predictions, reused_folds, fitted_folds


def _prepare_batch_fold(base: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    mds = base["mds"]
    fold = prepare_single_fold(
        mds.dataset,
        split,
        mds.feature_names,
        mds.label_col,
        mds.date_col,
        mds.entity_cols[0],
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=mds.temporal_keys,
        temporal_feature_names=mds.temporal_feature_names,
        train_sample_frac=base["train_sample_frac"],
        eval_label_col=mds.eval_label_col,
    )
    if fold is None:
        raise ValueError(f"linear request could not prepare fold {split['fold']}")
    return fold


def _resolve_batch_candidate(
    study: Study,
    candidate: _BatchCandidate,
    base: dict[str, Any],
) -> None:
    placeholder_folds = tuple({"fold": int(split["fold"])} for split in base["splits"])
    spec, context = _build_resolved_request(
        study,
        candidate.request,
        label_ref=base["label_ref"],
        mds=base["mds"],
        splits=base["splits"],
        cv_record=base["cv_record"],
        config=candidate.config,
        effective=candidate.effective_params,
        expected=base["expected"],
        folds=placeholder_folds,
        runtime_provenance=base["runtime_provenance"],
    )
    candidate.spec = spec
    candidate.context = context
    cached = _cached_run(study, spec, context)
    if cached is not None:
        candidate.result = cached
        return
    candidate.started_at_s = time.perf_counter()
    training = study.results.register_training(
        spec,
        execution_tier=spec["execution_tier"],
        runtime_provenance=context.runtime_provenance,
    )
    candidate.training = training
    candidate.ledger = ExecutionLedger(study, training.root)
    candidate.attempt = candidate.ledger.start(training.hash)


def _fail_batch_candidate(candidate: _BatchCandidate, error: Exception) -> None:
    if candidate.error is not None:
        return
    candidate.error = error
    if candidate.attempt is not None:
        candidate.attempt.finish(
            "failed",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "reused_folds": candidate.reused_folds,
                "fitted_folds": candidate.fitted_folds,
            },
        )
        candidate.attempt = None


def _run_batch_candidate_fold(candidate: _BatchCandidate, fold: dict[str, Any]) -> None:
    if candidate.result is not None or candidate.error is not None:
        return
    assert candidate.spec is not None
    assert candidate.context is not None
    assert candidate.training is not None
    assert candidate.ledger is not None
    fold_id = int(fold["fold"])
    if fold_id in candidate.reused_folds or fold_id in candidate.fitted_folds:
        return
    try:
        frame, reused, elapsed = _fit_or_reuse_fold(
            candidate.spec,
            candidate.context,
            candidate.training,
            candidate.ledger,
            fold,
        )
    except Exception as exc:
        _fail_batch_candidate(candidate, exc)
        return
    candidate.frames.append(frame)
    candidate.fit_elapsed_s += elapsed
    (candidate.reused_folds if reused else candidate.fitted_folds).append(fold_id)


def _reuse_batch_candidate_fold(candidate: _BatchCandidate, fold_id: int) -> bool:
    if candidate.result is not None or candidate.error is not None:
        return True
    assert candidate.spec is not None
    assert candidate.training is not None
    assert candidate.ledger is not None
    params = candidate.spec["computation"]["model"]["effective_params_by_fold"][str(fold_id)]
    training_dir = candidate.training.root / "run_log" / "training" / candidate.training.hash
    artifact = training_dir / "models" / f"fold_{fold_id}.joblib"
    shard = training_dir / "prediction_folds" / f"fold_{fold_id}.parquet"
    if not candidate.ledger.reusable_fold(
        training_hash=candidate.training.hash,
        candidate_identity=candidate.training.hash,
        fold_id=fold_id,
        fitted_state=artifact,
        prediction_shard=shard,
        resolved_settings=params,
    ):
        return False
    candidate.frames.append(pl.read_parquet(shard))
    candidate.reused_folds.append(fold_id)
    return True


def _finish_batch_candidate(study: Study, candidate: _BatchCandidate) -> None:
    if candidate.result is not None or candidate.error is not None:
        return
    assert candidate.spec is not None
    assert candidate.context is not None
    assert candidate.training is not None
    assert candidate.attempt is not None
    try:
        if len(candidate.frames) != len(candidate.context.fold_ids):
            raise RuntimeError(
                f"linear candidate produced {len(candidate.frames)} of "
                f"{len(candidate.context.fold_ids)} fold shards"
            )
        _write_model_manifest(candidate.training)
        predictions = pl.concat(candidate.frames).sort(
            candidate.context.entity_col, "timestamp", "fold"
        )
        prediction = study.results.publish_predictions(
            candidate.training,
            checkpoint_kind="final",
            checkpoint_value=None,
            split="validation",
            predictions=predictions,
            expected_keys=candidate.context.expected_keys,
            task_type=candidate.context.task_type,
            class_values=list(candidate.context.class_values) or None,
            eval_col="eval_actual" if candidate.context.eval_label_col else None,
            label=candidate.context.label_col,
        )
        diagnostics = {
            "cache_hit": False,
            "reused_folds": candidate.reused_folds,
            "fitted_folds": candidate.fitted_folds,
        }
        candidate.attempt.finish("completed", diagnostics)
        candidate.attempt = None
        runtime_path = (
            candidate.training.root
            / "run_log"
            / "training"
            / candidate.training.hash
            / "runtime.json"
        )
        _write_runtime_fields(runtime_path, elapsed_s=time.perf_counter() - candidate.started_at_s)
        candidate.result = ModelRun(candidate.training, (prediction,), diagnostics)
    except Exception as exc:
        _fail_batch_candidate(candidate, exc)


def _run_batch_group(
    study: Study,
    indexed_requests: list[tuple[int, dict[str, Any]]],
    compatibility_key: str,
    base: dict[str, Any],
    *,
    report_batch: bool,
    planned_effective: dict[int, dict[str, dict[str, Any]]] | None = None,
) -> list[_BatchCandidate]:
    fold_ids = tuple(int(split["fold"]) for split in base["splits"])
    candidates = []
    dependent = []
    for index, request in indexed_requests:
        config = _load_preset(request["config_name"])
        effective = (
            planned_effective[index]
            if planned_effective is not None
            else _fixed_effective_params(config, request["overrides"], fold_ids)
        )
        candidate = _BatchCandidate(index, request, config, effective or {})
        candidates.append(candidate)
        if effective is None:
            dependent.append(candidate)
            continue
        _resolve_batch_candidate(study, candidate, base)
    dependent_indices = {candidate.index for candidate in dependent}

    preparation_elapsed_s = 0.0
    preparation_count = 0
    first_pass_needed = bool(dependent) or any(
        candidate.result is None and candidate.error is None for candidate in candidates
    )
    if first_pass_needed:
        for split in base["splits"]:
            fold_id = int(split["fold"])
            fixed_pending = [
                candidate
                for candidate in candidates
                if candidate.index not in dependent_indices
                and not _reuse_batch_candidate_fold(candidate, fold_id)
            ]
            if not dependent and not fixed_pending:
                continue
            started = time.perf_counter()
            try:
                fold = _prepare_batch_fold(base, split)
            except Exception as exc:
                for candidate in candidates:
                    _fail_batch_candidate(candidate, exc)
                break
            preparation_elapsed_s += time.perf_counter() - started
            preparation_count += 1
            for candidate in dependent:
                if candidate.error is not None:
                    continue
                try:
                    candidate.effective_params.update(
                        _effective_params(candidate.config, candidate.request["overrides"], [fold])
                    )
                except Exception as exc:
                    _fail_batch_candidate(candidate, exc)
            for candidate in fixed_pending:
                _run_batch_candidate_fold(candidate, fold)
            del fold

    for candidate in candidates:
        if candidate.index not in dependent_indices:
            _finish_batch_candidate(study, candidate)

    active_dependent = []
    for candidate in dependent:
        if candidate.error is not None:
            continue
        if set(candidate.effective_params) != {str(fold_id) for fold_id in fold_ids}:
            _fail_batch_candidate(
                candidate,
                RuntimeError("linear candidate did not resolve parameters for every fold"),
            )
            continue
        try:
            _resolve_batch_candidate(study, candidate, base)
        except Exception as exc:
            _fail_batch_candidate(candidate, exc)
            continue
        if candidate.result is None:
            active_dependent.append(candidate)

    if active_dependent:
        for split in base["splits"]:
            fold_id = int(split["fold"])
            pending = [
                candidate
                for candidate in active_dependent
                if not _reuse_batch_candidate_fold(candidate, fold_id)
            ]
            if not pending:
                continue
            started = time.perf_counter()
            try:
                fold = _prepare_batch_fold(base, split)
            except Exception as exc:
                for candidate in active_dependent:
                    _fail_batch_candidate(candidate, exc)
                break
            preparation_elapsed_s += time.perf_counter() - started
            preparation_count += 1
            for candidate in pending:
                _run_batch_candidate_fold(candidate, fold)
            del fold
        for candidate in active_dependent:
            _finish_batch_candidate(study, candidate)

    group_digest = hashlib.sha256(compatibility_key.encode()).hexdigest()[:12]
    fit_elapsed_s = sum(candidate.fit_elapsed_s for candidate in candidates)
    measured_s = preparation_elapsed_s + fit_elapsed_s
    for candidate in candidates:
        if candidate.result is None or not report_batch:
            continue
        candidate.result.diagnostics.update(
            {
                "execution_order": "fold_major",
                "compatibility_group": group_digest,
                "compatibility_group_size": len(candidates),
                "base_fold_preparations": preparation_count,
                "base_fold_preparation_s": preparation_elapsed_s,
                "candidate_fit_s": candidate.fit_elapsed_s,
                "preparation_fraction": (preparation_elapsed_s / measured_s if measured_s else 0.0),
                "disk_fold_cache": False,
            }
        )
    return candidates


def plan_model_requests(
    study: Study,
    requests: list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[Any, ...]]:
    """Resolve a request batch once without fitting or writing result rows."""
    if not requests:
        raise ValueError("linear batch planner requires at least one request")
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, request in enumerate(requests):
        groups.setdefault(_compatibility_key(request), []).append((index, request))

    ordered: list[dict[str, Any] | None] = [None] * len(requests)
    planned_groups = []
    input_cache: dict[tuple[str, str, int], tuple[Any, Any]] = {}
    for key, indexed_requests in groups.items():
        input_key = _input_compatibility_key(indexed_requests[0][1])
        base = _load_batch_base(
            study,
            indexed_requests[0][1],
            inputs=input_cache.get(input_key),
        )
        input_cache.setdefault(input_key, (base["label_ref"], base["mds"]))
        fold_ids = tuple(int(split["fold"]) for split in base["splits"])
        candidates = []
        dependent = []
        for index, request in indexed_requests:
            config = _load_preset(request["config_name"])
            effective = _fixed_effective_params(config, request["overrides"], fold_ids)
            candidate = _BatchCandidate(index, request, config, effective or {})
            candidates.append(candidate)
            if effective is None:
                dependent.append(candidate)

        if dependent:
            for split in base["splits"]:
                fold = _prepare_batch_fold(base, split)
                try:
                    for candidate in dependent:
                        candidate.effective_params.update(
                            _effective_params(
                                candidate.config,
                                candidate.request["overrides"],
                                [fold],
                            )
                        )
                finally:
                    del fold

        placeholder_folds = tuple({"fold": fold_id} for fold_id in fold_ids)
        for candidate in candidates:
            if set(candidate.effective_params) != {str(fold_id) for fold_id in fold_ids}:
                raise RuntimeError(
                    f"linear planner did not resolve every fold for "
                    f"{candidate.request['config_name']}"
                )
            spec, _ = _build_resolved_request(
                study,
                candidate.request,
                label_ref=base["label_ref"],
                mds=base["mds"],
                splits=base["splits"],
                cv_record=base["cv_record"],
                config=candidate.config,
                effective=candidate.effective_params,
                expected=base["expected"],
                folds=placeholder_folds,
                runtime_provenance=base["runtime_provenance"],
            )
            ordered[candidate.index] = spec
        planned_groups.append(
            (
                key,
                indexed_requests,
                base,
                {candidate.index: candidate.effective_params for candidate in candidates},
            )
        )
    if any(spec is None for spec in ordered):
        raise RuntimeError("linear batch planner did not resolve every request")
    return tuple(spec for spec in ordered if spec is not None), tuple(planned_groups)


def run_model_plan(study: Study, payload: tuple[Any, ...]) -> tuple[ModelRun, ...]:
    ordered: list[ModelRun | None] = [
        None for _ in range(sum(len(indexed) for _, indexed, _, _ in payload))
    ]
    failures = []
    for key, indexed_requests, base, planned_effective in payload:
        try:
            candidates = _run_batch_group(
                study,
                indexed_requests,
                key,
                base,
                report_batch=len(ordered) > 1,
                planned_effective=planned_effective,
            )
        except Exception as error:
            failures.append(error)
            continue
        for candidate in candidates:
            if candidate.error is not None:
                failures.append(candidate.error)
            elif candidate.result is not None:
                ordered[candidate.index] = candidate.result
    if failures:
        raise failures[0]
    if any(result is None for result in ordered):
        raise RuntimeError("linear planned batch did not produce every requested result")
    return tuple(result for result in ordered if result is not None)


def run_model_requests(study: Study, requests: list[dict[str, Any]]) -> tuple[ModelRun, ...]:
    if not requests:
        raise ValueError("linear batch runner requires at least one request")
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, request in enumerate(requests):
        groups.setdefault(_compatibility_key(request), []).append((index, request))

    ordered: list[ModelRun | None] = [None] * len(requests)
    failures: list[Exception] = []
    input_cache: dict[tuple[str, str, int], tuple[Any, Any]] = {}
    for key, indexed_requests in groups.items():
        input_key = _input_compatibility_key(indexed_requests[0][1])
        base = _load_batch_base(
            study,
            indexed_requests[0][1],
            inputs=input_cache.get(input_key),
        )
        input_cache.setdefault(input_key, (base["label_ref"], base["mds"]))
        candidates = _run_batch_group(
            study,
            indexed_requests,
            key,
            base,
            report_batch=len(requests) > 1,
        )
        for candidate in candidates:
            if candidate.error is not None:
                failures.append(candidate.error)
            elif candidate.result is not None:
                ordered[candidate.index] = candidate.result
    if failures:
        raise failures[0]
    if any(result is None for result in ordered):
        raise RuntimeError("linear batch did not produce every requested result")
    return tuple(result for result in ordered if result is not None)


def run_resolved_request(
    study: Study,
    spec: dict[str, Any],
    context: LinearContext,
) -> ModelRun:
    cached = _cached_run(study, spec, context)
    if cached is not None:
        return cached

    started = time.perf_counter()
    training = study.results.register_training(
        spec,
        execution_tier=spec["execution_tier"],
        runtime_provenance=context.runtime_provenance,
    )
    train_dir = training.root / "run_log" / "training" / training.hash
    ledger = ExecutionLedger(study, training.root)
    attempt = ledger.start(training.hash)
    try:
        predictions, reused_folds, fitted_folds = _fit_or_reuse_predictions(
            spec, context, training, ledger
        )
        prediction = study.results.publish_predictions(
            training,
            checkpoint_kind=context.checkpoint_kind,
            checkpoint_value=context.checkpoint_value,
            split=context.prediction_split,
            predictions=predictions,
            expected_keys=context.expected_keys,
            task_type=context.task_type,
            class_values=list(context.class_values) or None,
            eval_col="eval_actual" if context.eval_label_col else None,
            label=context.label_col,
        )
        diagnostics = {
            "cache_hit": False,
            "reused_folds": reused_folds,
            "fitted_folds": fitted_folds,
        }
        attempt.finish("completed", diagnostics)
    except Exception as exc:
        attempt.finish("failed", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    runtime_path = train_dir / "runtime.json"
    _write_runtime_fields(runtime_path, elapsed_s=time.perf_counter() - started)
    return ModelRun(training=training, predictions=(prediction,), diagnostics=diagnostics)


def validate_locked_run(
    study: Study,
    spec: dict[str, Any],
    context: LinearContext,
    run: ModelRun,
) -> str:
    """Validate the persisted fit, prediction shard, and selected prediction."""
    if run.training.hash != training_hash_from_spec(spec) or len(run.predictions) != 1:
        raise ValueError("locked linear run has the wrong training or prediction identity")
    prediction = run.predictions[0]
    record = prediction.registry_record()
    expected_checkpoint = (
        context.prediction_split,
        context.checkpoint_kind,
        context.checkpoint_value,
    )
    if (
        record["split"],
        record["checkpoint_kind"],
        record["checkpoint_value"],
    ) != expected_checkpoint:
        raise ValueError("locked linear run published the wrong checkpoint")
    published = prediction.load().sort(context.entity_col, "timestamp", "fold")

    training_dir = run.training.root / "run_log" / "training" / run.training.hash
    model_dir = training_dir / "models"
    manifest = model_dir / "manifest.json"
    if not manifest.is_file():
        raise ValueError("locked linear run has no fitted-state manifest")
    manifest_record = json.loads(manifest.read_text())
    expected_files = {f"fold_{fold_id}.joblib" for fold_id in context.fold_ids}
    if set(manifest_record.get("files") or {}) != expected_files:
        raise ValueError("locked linear fitted-state manifest has the wrong fold population")
    ledger = ExecutionLedger(study, run.training.root)
    shards = []
    for fold_id in context.fold_ids:
        params = spec["computation"]["model"]["effective_params_by_fold"][str(fold_id)]
        artifact = model_dir / f"fold_{fold_id}.joblib"
        shard = training_dir / "prediction_folds" / f"fold_{fold_id}.parquet"
        if manifest_record["files"][artifact.name] != _sha256(artifact):
            raise ValueError("locked linear fitted-state digest does not validate")
        if not ledger.reusable_fold(
            training_hash=run.training.hash,
            candidate_identity=run.training.hash,
            fold_id=fold_id,
            fitted_state=artifact,
            prediction_shard=shard,
            resolved_settings=params,
        ):
            raise ValueError("locked linear completed-fold record does not validate")
        shards.append(pl.read_parquet(shard))
    reconstructed = pl.concat(shards).sort(context.entity_col, "timestamp", "fold")
    if not reconstructed.equals(published):
        raise ValueError("locked linear fitted state does not reproduce published predictions")
    return hashlib.sha256(canonical_json(manifest_record).encode()).hexdigest()
