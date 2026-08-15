from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
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

import numpy as np
import polars as pl
import torch

from case_studies.research.cv import require_fold_scoped_temporal_compatibility
from case_studies.research.identity import ResolvedSpec
from case_studies.research.models import ModelRun
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.latent_factors.cv import (
    _build_prediction_frame,
    _expected_latent_checkpoints,
    _prepare_fold_inputs,
    _save_fold_extras,
    run_latent_factor_cv,
)
from case_studies.utils.latent_factors.library_bridge import (
    predict_latent_fold_from_artifact,
)
from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec
from utils.modeling import RANDOM_SEED

if TYPE_CHECKING:
    from case_studies.research.workspace import Study
    from case_studies.utils.latent_factors.case_study import LatentFactorCaseStudyContext


_LATENT_MODELS = {"cae", "ipca", "pca", "sae", "sdf"}
_PREVIEW_FIELDS = {
    "folds",
    "max_iter",
    "max_symbols",
    "n_epochs",
    "n_epochs_cond",
    "n_epochs_moment",
    "n_epochs_unc",
    "n_factors",
}
_MODEL_PREVIEW_FIELDS = {
    "cae": {"folds", "max_symbols", "n_epochs", "n_factors"},
    "ipca": {"folds", "max_iter", "max_symbols", "n_factors"},
    "pca": {"folds", "max_symbols", "n_factors"},
    "sae": {"folds", "max_symbols", "n_epochs"},
    "sdf": {"folds", "max_symbols", "n_epochs_cond", "n_epochs_moment", "n_epochs_unc"},
}
_INTERFACE_FIELDS = {
    "deterministic_algorithms",
    "device",
    "fold_workers",
    "n_epochs",
    "n_factors",
    "num_threads",
    "use_macro",
}


@dataclass(frozen=True)
class LatentFactorContext:
    case: LatentFactorCaseStudyContext
    model_name: str
    model_kwargs: dict[str, Any]
    n_factors: int
    n_epochs: int
    fold_workers: int
    expected_keys: pl.DataFrame
    runtime_provenance: dict[str, Any]
    prediction_split: str = "validation"
    published_checkpoints: tuple[int, ...] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_identity() -> dict[str, str]:
    root = Path(__file__).parent
    release_root = root.parents[2]
    files = [
        root / "adapter.py",
        root / "cae.py",
        root / "case_study.py",
        root / "common.py",
        root / "cv.py",
        root / "ipca.py",
        root / "library_bridge.py",
        root / "macro_context.py",
        root / "panel.py",
        root / "pca.py",
        root / "sae.py",
        root / "sdf.py",
        release_root / "utils" / "artifact_specs.py",
        release_root / "utils" / "modeling.py",
    ]
    return {path.relative_to(release_root).as_posix(): _sha256(path) for path in files}


def _runtime_identity() -> dict[str, str | None]:
    return {
        "ml4t-models": _package_version("ml4t-models"),
        "numpy": _package_version("numpy"),
        "polars": _package_version("polars"),
        "torch": _package_version("torch"),
    }


def _runtime_provenance(study: Study, device: str) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(study.release_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    lock_path = study.release_root / "uv.lock"
    record: dict[str, Any] = {
        "device": device,
        "entry_point": "case_studies.utils.latent_factors",
        "lock_digest": _sha256(lock_path) if lock_path.is_file() else None,
        "packages": _runtime_identity(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_commit": commit,
    }
    if device == "cuda":
        record["cuda_runtime"] = torch.version.cuda
        record["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return record


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
    case: LatentFactorCaseStudyContext,
    request: dict[str, Any],
    label_timeline: pl.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cv = request.get("cv")
    if cv is None:
        splits = list(case.splits)
        normalized = _normalize_folds(splits)
        cv_record = {
            "request": {"source": "case_study_default"},
            "folds": list(normalized),
            "identity": value_digest(pl.DataFrame(list(normalized))),
        }
    else:
        resolved = cv.resolve(label_timeline, date_col=case.date_col)
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
        raise ValueError("latent-factor request resolved no cross-validation folds")
    return splits, cv_record


def _model_runner(model_name: str):
    from case_studies.utils.latent_factors.cae import run_cae_fold
    from case_studies.utils.latent_factors.ipca import run_ipca_fold
    from case_studies.utils.latent_factors.pca import run_pca_fold
    from case_studies.utils.latent_factors.sae import run_sae_fold
    from case_studies.utils.latent_factors.sdf import run_sdf_fold

    return {
        "cae": run_cae_fold,
        "ipca": run_ipca_fold,
        "pca": run_pca_fold,
        "sae": run_sae_fold,
        "sdf": run_sdf_fold,
    }[model_name]


def _resolve_model_configuration(
    case: LatentFactorCaseStudyContext,
    model_name: str,
    overrides: dict[str, Any],
    reductions: dict[str, Any],
) -> tuple[dict[str, Any], int, int, int, dict[str, int]]:
    unsupported_reductions = set(reductions) - _MODEL_PREVIEW_FIELDS[model_name]
    if unsupported_reductions:
        raise ValueError(
            f"unsupported {model_name} preview reductions: {sorted(unsupported_reductions)}"
        )
    interface = {key: value for key, value in overrides.items() if key in _INTERFACE_FIELDS}
    model_kwargs = dict(case.model_kwargs.get(model_name) or {})
    model_kwargs.update(
        {key: value for key, value in overrides.items() if key not in _INTERFACE_FIELDS}
    )

    n_factors = int(interface.get("n_factors", model_kwargs.pop("n_factors", 5)))
    n_epochs = int(interface.get("n_epochs", model_kwargs.pop("n_epochs", 0)))
    checkpoint_metadata: dict[str, int] = {}
    if model_name in {"pca", "ipca"}:
        checkpoint_interval = int(model_kwargs.pop("checkpoint_interval", 0))
        if checkpoint_interval != 0:
            raise ValueError(f"{model_name} checkpoint_interval must be 0")
        checkpoint_metadata["checkpoint_interval"] = checkpoint_interval
    if "n_factors" in reductions:
        n_factors = int(reductions["n_factors"])
    if "n_epochs" in reductions:
        n_epochs = int(reductions["n_epochs"])
    for field in ("max_iter", "n_epochs_cond", "n_epochs_moment", "n_epochs_unc"):
        if field in reductions:
            model_kwargs[field] = int(reductions[field])
    if model_name == "sdf" and reductions:
        reduced_budget = max(
            int(model_kwargs.get("n_epochs_unc", 256)),
            int(model_kwargs.get("n_epochs_cond", 1024)),
        )
        configured = model_kwargs.get("checkpoint_epochs")
        if configured is not None:
            compatible = sorted(
                {int(epoch) for epoch in configured if 1 <= int(epoch) <= reduced_budget}
            )
            if reduced_budget not in compatible:
                compatible.append(reduced_budget)
            model_kwargs["checkpoint_epochs"] = compatible
    if model_name == "sdf" and model_kwargs.get("expected_return_mapper", "linear") != "linear":
        raise ValueError("SDF expected_return_mapper currently supports only 'linear'")
    if n_factors < 1 or n_epochs < 0:
        raise ValueError("latent-factor n_factors must be positive and n_epochs non-negative")

    runner = _model_runner(model_name)
    positional = {"chars_train", "returns_train", "chars_val", "returns_val"}
    allowed = (
        set(inspect.signature(runner).parameters)
        - positional
        - {
            "artifact_dir",
            "log_fn",
            "n_factors",
            "n_epochs",
        }
    )
    unknown = set(model_kwargs) - allowed
    if unknown:
        raise ValueError(f"unsupported {model_name} parameters: {sorted(unknown)}")
    fold_workers = int(interface.get("fold_workers", 1))
    if fold_workers < 1 or (fold_workers > 1 and model_name != "ipca"):
        raise ValueError("parallel latent folds are supported only for IPCA")
    return model_kwargs, n_factors, n_epochs, fold_workers, checkpoint_metadata


def _resolved_macro_digest(case: LatentFactorCaseStudyContext) -> str:
    digest = hashlib.sha256()
    resolved = 0
    for split in case.splits:
        fold_inputs = _prepare_fold_inputs(
            dataset=case.dataset,
            split=split,
            feature_names=case.feature_names,
            label_col=case.primary_label,
            date_col=case.date_col,
            entity_col=case.entity_col,
            eval_label_col=case.eval_label_col,
            macro_panel=case.macro_panel,
            need_pca_inputs=False,
            need_ragged_inputs=True,
            temporal_by_fold=case.temporal_by_fold,
            temporal_keys=case.temporal_keys,
            temporal_feature_names=case.temporal_feature_names,
        )
        if fold_inputs is None:
            raise ValueError(f"latent fold {split['fold']} has insufficient eligible periods")
        model_input = fold_inputs["ragged"]
        for segment in ("macro_train", "macro_val"):
            values = model_input[segment]
            if values is None:
                raise ValueError("SDF request resolved no macro values")
            array = np.ascontiguousarray(values)
            metadata = json.dumps(
                {
                    "dtype": array.dtype.str,
                    "fold": int(split["fold"]),
                    "segment": segment,
                    "shape": array.shape,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update(metadata.encode())
            digest.update(array.tobytes())
            resolved += array.size
    if resolved == 0:
        raise ValueError("SDF request resolved an empty macro surface")
    return f"sha256:{digest.hexdigest()}"


def _prepare_expected_keys(
    case: LatentFactorCaseStudyContext,
    model_name: str,
) -> pl.DataFrame:
    frames = []
    for split in case.splits:
        fold_inputs = _prepare_fold_inputs(
            dataset=case.dataset,
            split=split,
            feature_names=case.feature_names,
            label_col=case.primary_label,
            date_col=case.date_col,
            entity_col=case.entity_col,
            eval_label_col=case.eval_label_col,
            macro_panel=case.macro_panel,
            need_pca_inputs=model_name == "pca",
            need_ragged_inputs=model_name != "pca",
            temporal_by_fold=case.temporal_by_fold,
            temporal_keys=case.temporal_keys,
            temporal_feature_names=case.temporal_feature_names,
        )
        if fold_inputs is None:
            raise ValueError(f"latent fold {split['fold']} has insufficient eligible periods")
        model_input = fold_inputs["pca"] if model_name == "pca" else fold_inputs["ragged"]
        finite = np.isfinite(model_input["returns_val"])
        if model_input.get("eval_returns_val") is not None:
            finite &= np.isfinite(model_input["eval_returns_val"])
        dummy = np.where(finite, 0.0, np.nan)
        frame = _build_prediction_frame(
            predictions=dummy,
            returns_val=model_input["returns_val"],
            eval_returns_val=model_input.get("eval_returns_val"),
            val_dates=model_input["val_dates"],
            val_entities=model_input["val_entities"],
            fold_id=int(split["fold"]),
            model_name=model_name,
            epoch=0,
        )
        if frame is None:
            raise ValueError(f"latent fold {split['fold']} has no eligible validation keys")
        frames.append(frame.select("symbol", "timestamp", pl.col("fold_id").alias("fold")))
    expected = pl.concat(frames).sort("symbol", "timestamp", "fold")
    if expected.n_unique(["symbol", "timestamp", "fold"]) != expected.height:
        raise ValueError("latent-factor request produced duplicate expected prediction keys")
    return expected


def resolve_model_request(study: Study, request: dict[str, Any]):
    from case_studies.research.contracts import ExecutionTier
    from case_studies.utils.latent_factors.case_study import load_case_study_context

    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    unknown_reductions = set(reductions) - _PREVIEW_FIELDS
    if unknown_reductions:
        raise ValueError(
            f"unsupported latent-factor preview reductions: {sorted(unknown_reductions)}"
        )
    model_name = request["config_name"]
    if model_name not in _LATENT_MODELS:
        raise ValueError(f"unknown latent-factor config {model_name!r}")
    study.require_writable()
    study.activate(tier)
    label_ref = study.labels.get(request["label"], execution_tier=tier)
    max_symbols = int(reductions.get("max_symbols", 0))
    interface = {
        key: value for key, value in request["overrides"].items() if key in _INTERFACE_FIELDS
    }
    use_macro = bool(interface.get("use_macro", model_name == "sdf"))
    if model_name != "sdf" and use_macro:
        raise ValueError("only SDF accepts macro context")
    case = load_case_study_context(
        study.case_study,
        primary_label=label_ref.name,
        max_symbols=max_symbols,
        use_macro=use_macro,
    )
    if case.date_col != "timestamp" or case.entity_col not in {"product", "symbol"}:
        raise ValueError("latent-factor runner requires timestamp and symbol or product")
    if model_name == "pca" and not case.persistent_entities:
        raise ValueError("PCA requires persistent entity IDs")
    splits, cv_record = _select_splits(
        case,
        request,
        label_ref.load().select(case.date_col).unique(),
    )
    if (
        request.get("cv") is not None
        and case.temporal_by_fold is not None
        and case.temporal_keys
        and case.temporal_feature_names
    ):
        require_fold_scoped_temporal_compatibility(splits, case.temporal_artifact_splits)
    case.splits = splits
    (
        model_kwargs,
        n_factors,
        n_epochs,
        fold_workers,
        checkpoint_metadata,
    ) = _resolve_model_configuration(
        case,
        model_name,
        request["overrides"],
        reductions,
    )
    device = str(interface.get("device", case.device)).lower()
    if device == "gpu":
        device = "cuda"
    num_threads = int(interface.get("num_threads", case.num_threads))
    deterministic = bool(interface.get("deterministic_algorithms", case.deterministic_algorithms))
    if device not in {"cpu", "cuda"} or num_threads < 1:
        raise ValueError("invalid latent-factor runtime configuration")
    case.device = device
    case.num_threads = num_threads
    case.deterministic_algorithms = deterministic

    checkpoints = _expected_latent_checkpoints(
        model_name,
        n_epochs=n_epochs,
        model_kwargs=model_kwargs,
    )
    expected = _prepare_expected_keys(case, model_name)
    macro_context = None
    if model_name == "sdf":
        macro_context = dict(case.macro_context_spec or {"policy": "unspecified", "version": "v1"})
        if case.macro_panel is not None:
            macro_context["resolved_fold_digest"] = _resolved_macro_digest(case)
    input_lineage = case.input_data_spec
    computation = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": input_lineage["files"],
        "feature_names": list(case.feature_names),
        "task": {
            "type": case.task_type,
            "class_values": list(case.class_values),
            "continuous_eval_label": label_ref.definition.continuous_eval_label,
        },
        "cv": cv_record,
        "model": {
            "class": model_name,
            "implementation": "ml4t-models",
            "n_epochs": n_epochs,
            "n_factors": n_factors,
            "params": model_kwargs,
            **checkpoint_metadata,
        },
        "checkpoint_schedule": [
            {"kind": "epoch", "value": checkpoint} for checkpoint in checkpoints
        ],
        "expected_prediction_keys": {
            "digest": value_digest(expected, ("symbol", "timestamp", "fold")),
            "n_rows": expected.height,
            "n_folds": expected.get_column("fold").n_unique(),
        },
        "input_data_spec": input_lineage,
        "macro_context": macro_context,
        "runtime": {
            "deterministic_algorithms": deterministic,
            "device": device,
            "fold_workers": fold_workers,
            "num_threads": num_threads,
        },
        "numerical_runtime": {
            "deterministic_algorithms": deterministic,
            "device": device,
            "fold_workers": fold_workers,
            "num_threads": num_threads,
        },
        "sampling": {"max_symbols": max_symbols},
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
    }
    if tier is ExecutionTier.PREVIEW:
        computation["preview_reductions"] = reductions
    runtime_provenance = _runtime_provenance(study, device)
    spec = ResolvedSpec.create(
        family="latent_factors",
        label=label_ref.name,
        seed=RANDOM_SEED,
        computation=computation,
        provenance=runtime_provenance,
        config_name=model_name,
        execution_tier=tier.value,
    ).as_dict()
    context = LatentFactorContext(
        case=case,
        model_name=model_name,
        model_kwargs=model_kwargs,
        n_factors=n_factors,
        n_epochs=n_epochs,
        fold_workers=fold_workers,
        expected_keys=expected,
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
    """Reconstruct a latent-factor holdout fit without loading model defaults."""
    from case_studies.research.contracts import ExecutionTier
    from case_studies.research.models import (
        ResolvedModelRequest,
        locked_holdout_split,
        validate_locked_expected_keys,
    )
    from case_studies.utils.latent_factors.case_study import load_case_study_context

    computation = spec["computation"]
    model = computation.get("model")
    model_name = str(model.get("class", "")) if isinstance(model, dict) else ""
    if model_name not in _LATENT_MODELS:
        raise ValueError(f"unknown locked latent-factor model {model_name!r}")
    if checkpoint_kind != "epoch" or checkpoint_value is None:
        raise ValueError("latent-factor holdout requires one locked epoch checkpoint")
    study.require_writable()
    study.activate(ExecutionTier.CANONICAL)
    if spec.get("seed") != RANDOM_SEED:
        raise ValueError("locked latent-factor seed cannot be reproduced")
    if computation.get("sampling") != {"max_symbols": 0}:
        raise ValueError("locked latent-factor holdout requires an unreduced canonical dataset")
    label_ref = study.labels.get(spec["label"], execution_tier=ExecutionTier.CANONICAL)
    case = load_case_study_context(
        study.case_study,
        primary_label=label_ref.name,
        max_symbols=0,
        use_macro=model_name == "sdf",
    )
    if case.date_col != "timestamp" or case.entity_col not in {"product", "symbol"}:
        raise ValueError("locked latent-factor runner requires timestamp and symbol or product")
    if model_name == "pca" and not case.persistent_entities:
        raise ValueError("locked PCA holdout requires persistent entity IDs")
    expected_inputs = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": case.input_data_spec["files"],
        "feature_names": list(case.feature_names),
        "input_data_spec": case.input_data_spec,
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
        "task": {
            "type": case.task_type,
            "class_values": list(case.class_values),
            "continuous_eval_label": label_ref.definition.continuous_eval_label,
        },
    }
    for name, expected_value in expected_inputs.items():
        if computation.get(name) != expected_value:
            raise ValueError(
                f"locked latent-factor {name} does not match the available computation"
            )
    split = locked_holdout_split(spec, case.dataset, case.date_col, study.case_study)
    if case.temporal_by_fold is not None and case.temporal_keys and case.temporal_feature_names:
        require_fold_scoped_temporal_compatibility([split], case.temporal_artifact_splits)
    case.splits = [split]
    expected = _prepare_expected_keys(case, model_name)
    validate_locked_expected_keys(spec, expected)

    if (
        not isinstance(model, dict)
        or model.get("class") != model_name
        or model.get("implementation") != "ml4t-models"
        or not isinstance(model.get("params"), dict)
    ):
        raise ValueError("locked latent-factor model specification is unsupported")
    n_epochs = int(model.get("n_epochs", -1))
    n_factors = int(model.get("n_factors", 0))
    if n_epochs < 0 or n_factors < 1:
        raise ValueError("locked latent-factor training budget is invalid")
    model_kwargs = dict(model["params"])
    runner = _model_runner(model_name)
    positional = {"chars_train", "returns_train", "chars_val", "returns_val"}
    allowed = (
        set(inspect.signature(runner).parameters)
        - positional
        - {
            "artifact_dir",
            "log_fn",
            "n_factors",
            "n_epochs",
        }
    )
    if set(model_kwargs) - allowed:
        raise ValueError("locked latent-factor model parameters are unsupported")
    checkpoint_metadata = {"checkpoint_interval": 0} if model_name in {"pca", "ipca"} else {}
    expected_model = {
        "class": model_name,
        "implementation": "ml4t-models",
        "n_epochs": n_epochs,
        "n_factors": n_factors,
        "params": model_kwargs,
        **checkpoint_metadata,
    }
    if model != expected_model:
        raise ValueError("locked latent-factor model specification does not reproduce exactly")
    checkpoints = _expected_latent_checkpoints(
        model_name,
        n_epochs=n_epochs,
        model_kwargs=model_kwargs,
    )
    expected_schedule = [{"kind": "epoch", "value": value} for value in checkpoints]
    if (
        computation.get("checkpoint_schedule") != expected_schedule
        or checkpoint_value not in checkpoints
    ):
        raise ValueError("locked latent-factor checkpoint is absent from its exact schedule")
    runtime = computation.get("runtime")
    if not isinstance(runtime, dict) or computation.get("numerical_runtime") != runtime:
        raise ValueError("locked latent-factor runtime declarations disagree")
    required_runtime = {"deterministic_algorithms", "device", "fold_workers", "num_threads"}
    if set(runtime) != required_runtime:
        raise ValueError("locked latent-factor runtime declaration is incomplete")
    device = str(runtime["device"])
    fold_workers = int(runtime["fold_workers"])
    num_threads = int(runtime["num_threads"])
    if (
        device not in {"cpu", "cuda"}
        or num_threads < 1
        or fold_workers < 1
        or (fold_workers > 1 and model_name != "ipca")
    ):
        raise ValueError("locked latent-factor runtime cannot be reproduced")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("locked latent-factor runtime requires unavailable CUDA")
    case.device = device
    case.num_threads = num_threads
    case.deterministic_algorithms = bool(runtime["deterministic_algorithms"])
    macro_context = None
    if model_name == "sdf":
        macro_context = dict(case.macro_context_spec or {"policy": "unspecified", "version": "v1"})
        if case.macro_panel is not None:
            macro_context["resolved_fold_digest"] = _resolved_macro_digest(case)
    if computation.get("macro_context") != macro_context:
        raise ValueError("locked latent-factor macro context does not reproduce exactly")
    context = LatentFactorContext(
        case=case,
        model_name=model_name,
        model_kwargs=model_kwargs,
        n_factors=n_factors,
        n_epochs=n_epochs,
        fold_workers=fold_workers,
        expected_keys=expected,
        runtime_provenance=_runtime_provenance(study, device),
        prediction_split="holdout",
        published_checkpoints=(int(checkpoint_value),),
    )
    return ResolvedModelRequest(study, "latent_factors", spec, context)


def _manifest_files(model_dir: Path) -> dict[str, str]:
    path = model_dir / "manifest.json"
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text()).get("files") or {})


def _valid_model_dir(model_dir: Path, context: LatentFactorContext) -> bool:
    files = _manifest_files(model_dir)
    expected_models = {
        f"{context.model_name}/artifacts/fold_{int(split['fold'])}/model.ml4t"
        for split in context.case.splits
    }
    return (
        bool(files)
        and expected_models <= set(files)
        and all(
            (model_dir / name).is_file() and _sha256(model_dir / name) == digest
            for name, digest in files.items()
        )
    )


def _normalize_prediction_frame(frame: pl.DataFrame) -> pl.DataFrame:
    rename = {
        old: new
        for old, new in {
            "fold_id": "fold",
            "y_score": "prediction",
            "y_true": "actual",
        }.items()
        if old in frame.columns
    }
    columns = ["symbol", "timestamp", "fold", "prediction", "actual"]
    normalized = frame.rename(rename)
    if "eval_actual" in normalized.columns:
        columns.append("eval_actual")
    return normalized.select(columns).sort("symbol", "timestamp", "fold")


def _reconstruct_predictions(
    model_dir: Path,
    context: LatentFactorContext,
) -> dict[int, pl.DataFrame]:
    case = context.case
    frames: dict[int, list[pl.DataFrame]] = {
        int(checkpoint): []
        for checkpoint in _expected_latent_checkpoints(
            context.model_name,
            n_epochs=context.n_epochs,
            model_kwargs=context.model_kwargs,
            include_internal_aliases=True,
        )
    }
    for split in case.splits:
        fold_inputs = _prepare_fold_inputs(
            dataset=case.dataset,
            split=split,
            feature_names=case.feature_names,
            label_col=case.primary_label,
            date_col=case.date_col,
            entity_col=case.entity_col,
            eval_label_col=case.eval_label_col,
            macro_panel=case.macro_panel,
            need_pca_inputs=context.model_name == "pca",
            need_ragged_inputs=context.model_name != "pca",
            temporal_by_fold=case.temporal_by_fold,
            temporal_keys=case.temporal_keys,
            temporal_feature_names=case.temporal_feature_names,
        )
        if fold_inputs is None:
            raise ValueError(f"latent fold {split['fold']} no longer prepares from resolved inputs")
        model_input = fold_inputs["pca"] if context.model_name == "pca" else fold_inputs["ragged"]
        predictions = predict_latent_fold_from_artifact(
            context.model_name,
            artifact_dir=model_dir
            / context.model_name
            / "artifacts"
            / f"fold_{int(split['fold'])}",
            chars_train=model_input["chars_train"],
            returns_train=model_input["returns_train"],
            chars_val=model_input["chars_val"],
            returns_val=model_input["returns_val"],
            factor_returns_train=model_input.get("factor_returns_train"),
            macro_train=model_input.get("macro_train"),
            macro_val=model_input.get("macro_val"),
            output_mode=str(context.model_kwargs.get("output_mode", "beta_network")),
            device=case.device,
        )
        if set(predictions) != set(frames):
            raise ValueError(
                f"persisted {context.model_name} checkpoints {sorted(predictions)} do not match "
                f"resolved request {sorted(frames)}"
            )
        for epoch, values in predictions.items():
            frame = _build_prediction_frame(
                predictions=values,
                returns_val=model_input["returns_val"],
                eval_returns_val=model_input.get("eval_returns_val"),
                val_dates=model_input["val_dates"],
                val_entities=model_input["val_entities"],
                fold_id=int(split["fold"]),
                model_name=context.model_name,
                epoch=epoch,
            )
            if frame is None:
                raise ValueError(f"persisted latent fold {split['fold']} produced no predictions")
            frames[int(epoch)].append(frame)
    return {
        epoch: _normalize_prediction_frame(pl.concat(epoch_frames))
        for epoch, epoch_frames in frames.items()
    }


def _same_predictions(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    if (
        left.select("symbol", "timestamp", "fold").equals(
            right.select("symbol", "timestamp", "fold")
        )
        is False
    ):
        return False
    value_columns = ["prediction", "actual"]
    if "eval_actual" in left.columns or "eval_actual" in right.columns:
        if "eval_actual" not in left.columns or "eval_actual" not in right.columns:
            return False
        value_columns.append("eval_actual")
    return np.allclose(
        left.select(value_columns).to_numpy(),
        right.select(value_columns).to_numpy(),
        atol=1e-7,
        rtol=1e-7,
        equal_nan=False,
    )


def _cached_run(study: Study, spec: dict[str, Any], context: LatentFactorContext):
    from case_studies.research.models import ModelRun
    from case_studies.research.results import PredictionResult, Result, TrainingResult

    include_preview = spec["execution_tier"] == "preview"
    training_hash = training_hash_from_spec(spec)
    published = context.published_checkpoints or tuple(
        int(item["value"]) for item in spec["computation"]["checkpoint_schedule"]
    )
    try:
        training = Result.open(study, training_hash, include_preview=include_preview)
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
                include_preview=include_preview,
            )
            for checkpoint in published
        )
    except KeyError:
        return None
    if not isinstance(training, TrainingResult) or any(
        not isinstance(result, PredictionResult) or not result.complete for result in predictions
    ):
        return None
    model_dir = training.root / "run_log" / "training" / training.hash / "models"
    if not _valid_model_dir(model_dir, context):
        return None
    model_extras = model_dir / "fold_extras.json"
    public_extras = model_dir.parent / "fold_extras.json"
    if (
        not model_extras.is_file()
        or not public_extras.is_file()
        or json.loads(public_extras.read_text()) != json.loads(model_extras.read_text())
    ):
        return None
    reconstructed = _reconstruct_predictions(model_dir, context)
    prediction_results = tuple(
        result for result in predictions if isinstance(result, PredictionResult)
    )
    for result in prediction_results:
        epoch = int(result.registry_record()["checkpoint_value"])
        if not _same_predictions(
            _normalize_prediction_frame(result.load()),
            reconstructed[epoch],
        ):
            raise ValueError(f"persisted latent checkpoint {epoch} disagrees with registered data")
    return ModelRun(training=training, predictions=prediction_results)


def _write_manifest(staging: Path) -> None:
    files = {
        str(path.relative_to(staging)): _sha256(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if not files:
        raise ValueError("latent fit produced no persisted artifacts")
    (staging / "manifest.json").write_text(
        json.dumps({"files": files, "schema_version": 1}, indent=2, sort_keys=True) + "\n"
    )


def _publish_fold_extras(
    model_dir: Path,
    train_dir: Path,
    *,
    immutable: bool = False,
) -> None:
    source = model_dir / "fold_extras.json"
    serialized = source.read_text()
    extras = json.loads(serialized)
    if not isinstance(extras, list):
        raise ValueError(f"invalid latent fold diagnostics: {source}")
    target = train_dir / "fold_extras.json"
    if immutable and target.exists():
        try:
            existing = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("locked latent fold diagnostics are unreadable") from exc
        if existing != extras:
            raise ValueError("locked latent fold diagnostics conflict with fitted state")
        return
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(serialized)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def run_resolved_request(study: Study, spec: dict[str, Any], context: LatentFactorContext):
    from case_studies.research.models import ModelRun

    cached = _cached_run(study, spec, context)
    if cached is not None:
        return cached
    started = time.perf_counter()
    computation = spec["computation"]
    training = study.results.register_training(
        spec,
        execution_tier=spec["execution_tier"],
        runtime_provenance=context.runtime_provenance,
    )
    train_dir = training.root / "run_log" / "training" / training.hash
    model_dir = train_dir / "models"
    if model_dir.exists():
        if not _valid_model_dir(model_dir, context):
            raise ValueError(f"partial fitted-state directory requires inspection: {model_dir}")
        prediction_frames = _reconstruct_predictions(model_dir, context)
    else:
        staging = train_dir / f".models.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True)
        try:
            result = run_latent_factor_cv(
                panel_data=None,
                splits=context.case.splits,
                models=[context.model_name],
                n_factors=context.n_factors,
                n_epochs=context.n_epochs,
                model_kwargs={context.model_name: context.model_kwargs},
                save_dir=staging,
                use_cache=False,
                force_retrain=True,
                random_state=RANDOM_SEED,
                dataset=context.case.dataset,
                feature_names=context.case.feature_names,
                label_col=context.case.primary_label,
                date_col=context.case.date_col,
                entity_col=context.case.entity_col,
                eval_label_col=context.case.eval_label_col,
                task_type=context.case.task_type,
                class_values=context.case.class_values or None,
                macro_panel=context.case.macro_panel,
                macro_context_spec=context.case.macro_context_spec,
                input_data_spec=context.case.input_data_spec,
                persistent_entities=context.case.persistent_entities,
                checkpoint_selection_policy="fixed",
                reporting_epoch=max(
                    int(checkpoint["value"]) for checkpoint in computation["checkpoint_schedule"]
                ),
                device=context.case.device,
                num_threads=context.case.num_threads,
                deterministic_algorithms=context.case.deterministic_algorithms,
                temporal_by_fold=context.case.temporal_by_fold,
                temporal_keys=context.case.temporal_keys,
                temporal_feature_names=context.case.temporal_feature_names,
                fold_workers=context.fold_workers,
                checkpoint_surface="fitted_state",
            )
            _save_fold_extras(
                staging / "fold_extras.json",
                result["fold_extras"][context.model_name],
            )
            model_stage = staging / context.model_name
            shutil.rmtree(model_stage / "_incremental", ignore_errors=True)
            fresh_frames = {
                int(key[0]): _normalize_prediction_frame(frame)
                for key, frame in result["all_predictions"][context.model_name]
                .partition_by("epoch", as_dict=True)
                .items()
            }
            prediction_frames = _reconstruct_predictions(staging, context)
            if set(fresh_frames) != set(prediction_frames) or any(
                not _same_predictions(fresh_frames[epoch], prediction_frames[epoch])
                for epoch in fresh_frames
            ):
                raise ValueError("persisted latent state does not reproduce fresh predictions")
            _write_manifest(staging)
            os.replace(staging, model_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    _publish_fold_extras(
        model_dir,
        train_dir,
        immutable=context.prediction_split == "holdout",
    )

    prediction_results = []
    published = context.published_checkpoints or tuple(
        int(item["value"]) for item in computation["checkpoint_schedule"]
    )
    for value in published:
        prediction_results.append(
            study.results.publish_predictions(
                training,
                checkpoint_kind="epoch",
                checkpoint_value=value,
                split=context.prediction_split,
                predictions=prediction_frames[value],
                expected_keys=context.expected_keys,
                task_type=context.case.task_type,
                class_values=context.case.class_values or None,
                eval_col="eval_actual" if context.case.eval_label_col else None,
                label=context.case.primary_label,
            )
        )
    runtime_path = train_dir / "runtime.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        runtime["elapsed_s"] = time.perf_counter() - started
        temporary = runtime_path.with_name(f".{runtime_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, runtime_path)
    return ModelRun(training=training, predictions=tuple(prediction_results))


def validate_locked_run(
    study: Study,
    spec: dict[str, Any],
    context: LatentFactorContext,
    run: ModelRun,
) -> str:
    """Validate selected predictions against persisted latent fitted state."""
    if run.training.hash != training_hash_from_spec(spec) or len(run.predictions) != 1:
        raise ValueError("locked latent-factor run has the wrong identity")
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
        raise ValueError("locked latent-factor run published the wrong checkpoint")
    prediction.load()
    reopened = _cached_run(study, spec, context)
    if reopened is None or reopened.predictions[0].hash != prediction.hash:
        raise ValueError("locked latent fitted state cannot reproduce the selected prediction")
    model_dir = run.training.root / "run_log" / "training" / run.training.hash / "models"
    manifest = json.loads((model_dir / "manifest.json").read_text())
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
