from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge

from case_studies.research.contracts import ExecutionTier
from case_studies.research.cv import require_fold_scoped_temporal_compatibility
from case_studies.research.models import ModelRun
from case_studies.research.results import PredictionResult, Result, TrainingResult
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec
from utils.modeling import load_modeling_dataset, prepare_cv_folds, resolve_linear_params

if TYPE_CHECKING:
    from case_studies.research.workspace import Study


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
    feature_names: tuple[str, ...]
    label_col: str
    eval_label_col: str | None
    date_col: str
    entity_col: str
    task_type: str
    class_values: tuple[Any, ...]
    expected_keys: pl.DataFrame
    runtime_provenance: dict[str, Any]


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


def _expected_keys(folds: list[dict[str, Any]], entity_col: str, date_col: str) -> pl.DataFrame:
    frames = []
    for fold in folds:
        frame = pl.from_pandas(fold["meta"][[entity_col, date_col]]).with_columns(
            pl.lit(int(fold["fold"]), dtype=pl.Int64).alias("fold")
        )
        frames.append(
            frame.rename({entity_col: "symbol", date_col: "timestamp"}).select(
                "symbol", "timestamp", "fold"
            )
        )
    expected = pl.concat(frames).sort("symbol", "timestamp", "fold")
    if expected.n_unique(["symbol", "timestamp", "fold"]) != expected.height:
        raise ValueError("linear request produced duplicate expected prediction keys")
    return expected


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


def resolve_model_request(study: Study, request: dict[str, Any]):
    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    unknown_reductions = set(reductions) - _PREVIEW_FIELDS
    if unknown_reductions:
        raise ValueError(f"unsupported linear preview reductions: {sorted(unknown_reductions)}")
    study.require_writable()
    study.activate(tier)

    label_ref = study.labels.get(request["label"])
    max_symbols = int(reductions.get("max_symbols", 0))
    train_sample_frac = float(reductions.get("train_sample_frac", 1.0))
    if not 0 < train_sample_frac <= 1:
        raise ValueError("train_sample_frac must be in (0, 1]")
    mds = load_modeling_dataset(study.case_study, label_ref.name, max_symbols=max_symbols)
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
    folds = prepare_cv_folds(
        mds.dataset.to_pandas(),
        splits,
        mds.feature_names,
        mds.label_col,
        mds.date_col,
        entity_col,
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=mds.temporal_keys,
        temporal_feature_names=mds.temporal_feature_names,
        train_sample_frac=train_sample_frac,
        eval_label_col=mds.eval_label_col,
    )
    if len(folds) != len(splits):
        raise ValueError("linear request did not prepare every declared fold")

    config = _load_preset(request["config_name"])
    effective = _effective_params(config, request["overrides"], folds)
    expected = _expected_keys(folds, entity_col, mds.date_col)
    input_lineage = mds.input_lineage
    spec = {
        "identity_version": 2,
        "execution_tier": tier.value,
        "family": "linear",
        "label": label_ref.name,
        "seed": 42,
        "config_name": request["config_name"],
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
        "checkpoint_schedule": [{"kind": "final", "value": None}],
        "expected_prediction_keys": {
            "digest": value_digest(expected, ("symbol", "timestamp", "fold")),
            "n_rows": expected.height,
            "n_folds": expected.get_column("fold").n_unique(),
        },
        "input_data_spec": input_lineage,
        "sampling": {"train_sample_frac": train_sample_frac, "max_symbols": max_symbols},
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
    }
    if tier is ExecutionTier.PREVIEW:
        spec["preview_reductions"] = reductions
    context = LinearContext(
        folds=tuple(folds),
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        eval_label_col=mds.eval_label_col,
        date_col=mds.date_col,
        entity_col=entity_col,
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        expected_keys=expected,
        runtime_provenance=_runtime_provenance(study),
    )
    return spec, context


def _cached_run(study: Study, spec: dict[str, Any], context: LinearContext) -> ModelRun | None:
    training_hash = training_hash_from_spec(spec)
    prediction_hash = prediction_hash_from_parts(
        training_hash,
        None,
        "validation",
        checkpoint_kind="final",
        identity_version=2,
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
    _validate_fitted_state(model_dir, context)
    return ModelRun(training=training, predictions=(prediction,))


def _validate_fitted_state(model_dir: Path, context: LinearContext) -> None:
    manifest = model_dir / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"partial fitted-state directory requires inspection: {model_dir}")
    record = json.loads(manifest.read_text())
    expected_files = {f"fold_{int(fold['fold'])}.joblib" for fold in context.folds}
    if set(record.get("files") or {}) != expected_files:
        raise ValueError(f"fitted-state manifest does not match resolved folds: {model_dir}")
    if any(_sha256(model_dir / name) != digest for name, digest in record["files"].items()):
        raise ValueError(f"fitted-state digest mismatch: {model_dir}")
    prediction_record = record.get("predictions") or {}
    prediction_path = model_dir / str(prediction_record.get("name", ""))
    if (
        prediction_record.get("name") != "validation_predictions.parquet"
        or not prediction_path.is_file()
        or _sha256(prediction_path) != prediction_record.get("sha256")
    ):
        raise ValueError(f"fitted-state prediction frame mismatch: {model_dir}")


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
    frame = pl.from_pandas(fold["meta"][[context.entity_col, context.date_col]]).with_columns(
        pl.lit(fold_id, dtype=pl.Int64).alias("fold"),
        pl.Series("prediction", predictions),
        pl.Series("actual", fold["y_val"]),
    )
    if fold.get("y_eval") is not None:
        frame = frame.with_columns(pl.Series("eval_actual", fold["y_eval"]))
    return frame.rename({context.entity_col: "symbol", context.date_col: "timestamp"})


def _predict_from_fitted_state(model_dir: Path, context: LinearContext) -> pl.DataFrame:
    _validate_fitted_state(model_dir, context)
    predictions = pl.read_parquet(model_dir / "validation_predictions.parquet")
    required = {"symbol", "timestamp", "fold", "prediction", "actual"}
    if not required <= set(predictions.columns):
        raise ValueError("fitted-state prediction frame is missing required columns")
    keys = predictions.select("symbol", "timestamp", "fold")
    if keys.height != context.expected_keys.height or value_digest(
        keys, ("symbol", "timestamp", "fold")
    ) != value_digest(context.expected_keys, ("symbol", "timestamp", "fold")):
        raise ValueError("fitted-state prediction frame does not match expected coverage")
    return predictions.sort("symbol", "timestamp", "fold")


def _write_runtime_fields(runtime_path: Path, **fields: float) -> None:
    if not runtime_path.exists():
        return
    runtime = json.loads(runtime_path.read_text())
    runtime.update(fields)
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")


def _fit_predictions(spec: dict[str, Any], context: LinearContext, staging: Path) -> pl.DataFrame:
    cls = _MODEL_CLASSES[spec["model"]["class"]]
    prediction_frames = []
    files = {}
    for fold in context.folds:
        fold_id = int(fold["fold"])
        params = spec["model"]["effective_params_by_fold"][str(fold_id)]
        model = cls(**params)
        model.fit(fold["X_train"], fold["y_train"])
        predictions = _fold_predictions(model, fold, context)

        artifact = staging / f"fold_{fold_id}.joblib"
        joblib.dump(
            {
                "feature_names": context.feature_names,
                "model": model,
                "preprocessor": fold["preprocessor"],
            },
            artifact,
        )
        files[artifact.name] = _sha256(artifact)
        prediction_frames.append(_prediction_frame(fold, predictions, context))
    predictions = pl.concat(prediction_frames).sort("symbol", "timestamp", "fold")
    prediction_path = staging / "validation_predictions.parquet"
    predictions.write_parquet(prediction_path)
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "files": files,
                "predictions": {
                    "name": prediction_path.name,
                    "sha256": _sha256(prediction_path),
                },
                "schema_version": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return predictions


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
    model_dir = train_dir / "models"
    runtime_path = train_dir / "runtime.json"
    recovering = model_dir.exists()
    if recovering:
        predictions = _predict_from_fitted_state(model_dir, context)
    else:
        staging = train_dir / f".models.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True)
        try:
            predictions = _fit_predictions(spec, context, staging)
            os.replace(staging, model_dir)
            _write_runtime_fields(runtime_path, fit_elapsed_s=time.perf_counter() - started)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=predictions,
        expected_keys=context.expected_keys,
        task_type=context.task_type,
        class_values=list(context.class_values) or None,
        eval_col="eval_actual" if context.eval_label_col else None,
        label=context.label_col,
    )
    if recovering:
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        _write_runtime_fields(
            runtime_path,
            elapsed_s=float(runtime.get("fit_elapsed_s", 0.0)),
            recovery_elapsed_s=time.perf_counter() - started,
        )
    else:
        _write_runtime_fields(runtime_path, elapsed_s=time.perf_counter() - started)
    return ModelRun(training=training, predictions=(prediction,))
