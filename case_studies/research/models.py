from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any

import polars as pl

from case_studies.utils.registry.specs import (
    IDENTITY_VERSION,
    SUPPORTED_IDENTITY_VERSIONS,
    training_hash_from_spec,
)

from .adapters import get_adapter, registered_adapters
from .contracts import ExecutionTier
from .cv import CVSpec
from .results import PredictionResult, TrainingResult

if TYPE_CHECKING:
    from .workspace import Study


@dataclass(frozen=True)
class ModelRun:
    training: TrainingResult
    predictions: tuple[PredictionResult, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("model run requires at least one prediction result")
        if any(result.study != self.training.study for result in self.predictions):
            raise ValueError("model run results must belong to one study")
        if any(
            result.registry_record()["training_hash"] != self.training.hash
            for result in self.predictions
        ):
            raise ValueError("prediction result does not belong to the training result")


@dataclass(frozen=True)
class ResolvedModelRequest:
    study: Study
    family: str
    spec: dict[str, Any]
    _context: Any

    @property
    def identity(self) -> str:
        return training_hash_from_spec(self.spec)

    def run(self) -> ModelRun:
        module = _family_module(self.family)
        runner = getattr(module, "run_resolved_request", None)
        if runner is None:
            raise NotImplementedError(f"{self.family!r} has no shared model runner")
        result = runner(self.study, self.spec, self._context)
        if not isinstance(result, ModelRun):
            raise TypeError(
                f"{self.family!r} runner returned {type(result).__name__}, not ModelRun"
            )
        return result


def reconstruct_locked_model_request(
    study: Study,
    spec: dict[str, Any],
    *,
    checkpoint_kind: str,
    checkpoint_value: int | None,
) -> ResolvedModelRequest:
    """Reconstruct one canonical holdout request from an immutable training spec."""
    family = str(spec.get("family", ""))
    if spec.get("execution_tier") != ExecutionTier.CANONICAL.value:
        raise ValueError("locked holdout reconstruction requires a canonical training spec")
    if spec.get("identity_version") != IDENTITY_VERSION:
        raise ValueError("locked holdout reconstruction requires the current identity version")
    module = _family_module(family)
    reconstructor = getattr(module, "reconstruct_locked_request", None)
    if not callable(reconstructor):
        raise ValueError(f"{family!r} cannot reconstruct locked holdout computation")
    resolved = reconstructor(
        study,
        spec,
        checkpoint_kind=checkpoint_kind,
        checkpoint_value=checkpoint_value,
    )
    if not isinstance(resolved, ResolvedModelRequest):
        raise TypeError(
            f"{family!r} locked reconstructor returned {type(resolved).__name__}, "
            "not ResolvedModelRequest"
        )
    if resolved.study != study or resolved.family != family or resolved.spec != spec:
        raise ValueError("locked model reconstruction changed the immutable training request")
    if resolved.identity != training_hash_from_spec(spec):
        raise ValueError("locked model reconstruction changed the training identity")
    return resolved


def validate_locked_model_run(
    request: ResolvedModelRequest,
    run: ModelRun,
) -> str:
    module = _family_module(request.family)
    validator = getattr(module, "validate_locked_run", None)
    if not callable(validator):
        raise ValueError(f"{request.family!r} cannot validate locked fitted state")
    digest = validator(request.study, request.spec, request._context, run)
    if not isinstance(digest, str) or not digest:
        raise ValueError("locked model validator returned no fitted-state digest")
    return digest


def locked_holdout_split(
    spec: dict[str, Any], dataset: pl.DataFrame, date_col: str, case_study: str
) -> dict[str, Any]:
    computation = spec.get("computation")
    if not isinstance(computation, dict):
        raise ValueError("locked holdout requires a current resolved training specification")
    cv = computation.get("cv")
    if not isinstance(cv, dict):
        raise ValueError("locked holdout training specification has no resolved CV interval")
    request = cv.get("request") if isinstance(cv.get("request"), dict) else {}
    split_name = cv.get("split", request.get("split", request.get("phase")))
    if split_name != "holdout":
        raise ValueError("locked training CV must explicitly declare the holdout split")
    folds = cv.get("folds")
    if folds is not None:
        if not isinstance(folds, list) or len(folds) != 1 or not isinstance(folds[0], dict):
            raise ValueError("locked holdout CV must contain exactly one resolved fold")
        fold = dict(folds[0])
    else:
        fold = {
            "fold": cv.get("fold", 0),
            "train_start": cv.get("train_start", request.get("train_start")),
            "train_end": cv.get("train_end", request.get("train_end")),
            "val_start": cv.get(
                "evaluation_start", request.get("evaluation_start", request.get("val_start"))
            ),
            "val_end": cv.get(
                "evaluation_end", request.get("evaluation_end", request.get("val_end"))
            ),
        }
    required = {"fold", "train_start", "train_end", "val_start", "val_end"}
    missing = {name for name in required if fold.get(name) is None}
    if missing:
        raise ValueError(f"locked holdout CV is missing exact boundaries: {sorted(missing)}")
    fold["fold"] = int(fold["fold"])
    dtype = dataset.schema[date_col]
    boundaries = {
        name: pl.DataFrame({date_col: [fold[name]]})
        .with_columns(pl.col(date_col).cast(dtype, strict=False))
        .item()
        for name in ("train_start", "train_end", "val_start", "val_end")
    }
    if any(value is None for value in boundaries.values()):
        raise ValueError("locked holdout CV contains an invalid boundary")
    if boundaries["train_start"] > boundaries["train_end"]:
        raise ValueError("locked holdout training interval is empty")
    if boundaries["train_end"] >= boundaries["val_start"]:
        raise ValueError("locked holdout training interval overlaps evaluation")
    if boundaries["val_start"] > boundaries["val_end"]:
        raise ValueError("locked holdout evaluation interval is empty")
    from case_studies.utils.cv_window import canonical_window

    canonical = canonical_window(case_study, str(spec["label"]), split="holdout")
    if canonical is None:
        raise ValueError("case study has no canonical holdout window")
    actual_window = (
        boundaries["val_start"].date()
        if hasattr(boundaries["val_start"], "date")
        else boundaries["val_start"],
        boundaries["val_end"].date()
        if hasattr(boundaries["val_end"], "date")
        else boundaries["val_end"],
    )
    if actual_window != canonical:
        raise ValueError(
            f"locked holdout interval {actual_window!r} does not match canonical {canonical!r}"
        )
    available = dataset.get_column(date_col)
    if not available.is_between(
        boundaries["train_start"], boundaries["train_end"], closed="both"
    ).any():
        raise ValueError("locked holdout training interval has no source rows")
    if not available.is_between(
        boundaries["val_start"], boundaries["val_end"], closed="both"
    ).any():
        raise ValueError("locked holdout evaluation interval has no source rows")
    # Widen a pl.Date boundary to datetime before handing the split to the family
    # adapters. Polars casts either back to the column dtype, but the adapters that
    # build masks over `dataset.to_pandas()` compare against a datetime64 column,
    # and pandas raises TypeError on datetime64 vs datetime.date rather than
    # coercing. The preset CV path never hits this because it passes ISO strings.
    return {
        "fold": fold["fold"],
        **{
            name: datetime.combine(value, time.min)
            if isinstance(value, date) and not isinstance(value, datetime)
            else value
            for name, value in boundaries.items()
        },
    }


def validate_locked_expected_keys(spec: dict[str, Any], expected: pl.DataFrame) -> None:
    from case_studies.utils.artifact_digest import value_digest

    record = spec["computation"].get("expected_prediction_keys")
    if not isinstance(record, dict):
        raise ValueError("locked holdout training specification has no eligibility manifest")
    actual = {
        # The same key tuple every family resolver digests. Deriving it from the frame
        # instead would silently change the recorded identity the moment a builder
        # gained a column, and would stop matching the specs already registered.
        "digest": value_digest(expected, ("symbol", "timestamp", "fold")),
        "n_rows": expected.height,
        "n_folds": expected.get_column("fold").n_unique(),
    }
    if record != actual:
        raise ValueError(f"locked holdout eligibility mismatch: {record!r} != {actual!r}")


@dataclass(frozen=True)
class ModelRequest:
    study: Study
    family: str
    label: str
    config_name: str
    overrides: dict[str, Any]
    cv: CVSpec | None
    execution_tier: ExecutionTier
    preview_reductions: dict[str, Any]

    @classmethod
    def from_request(cls, study: Study, request: dict[str, Any]) -> ModelRequest:
        supported = {
            "family",
            "label",
            "config_name",
            "overrides",
            "cv",
            "execution_tier",
            "preview_reductions",
        }
        unknown = set(request) - supported
        if unknown:
            raise ValueError(f"unsupported model request fields: {sorted(unknown)}")
        missing = {"family", "label", "config_name"} - set(request)
        if missing:
            raise ValueError(f"model request is missing fields: {sorted(missing)}")
        family = str(request["family"])
        available = {binding.name for binding in registered_adapters("model")}
        if family not in available:
            raise ValueError(f"unsupported predictive family {family!r}")
        cv = request.get("cv")
        if cv is not None and not isinstance(cv, CVSpec):
            raise TypeError("model request cv must be a CVSpec")
        tier = ExecutionTier(request.get("execution_tier", ExecutionTier.CANONICAL))
        reductions = dict(request.get("preview_reductions") or {})
        if tier is ExecutionTier.PREVIEW and not reductions:
            raise ValueError("preview model requests must declare every reduction")
        if tier is ExecutionTier.CANONICAL and reductions:
            raise ValueError("canonical model requests cannot declare preview reductions")
        return cls(
            study=study,
            family=family,
            label=str(request["label"]),
            config_name=str(request["config_name"]),
            overrides=dict(request.get("overrides") or {}),
            cv=cv,
            execution_tier=tier,
            preview_reductions=reductions,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "label": self.label,
            "config_name": self.config_name,
            "overrides": dict(self.overrides),
            "cv": self.cv,
            "execution_tier": self.execution_tier.value,
            "preview_reductions": dict(self.preview_reductions),
        }

    def resolve(self) -> ResolvedModelRequest:
        module = _family_module(self.family)
        resolver = getattr(module, "resolve_model_request", None)
        if resolver is None:
            raise NotImplementedError(f"{self.family!r} has no shared model resolver")
        spec, context = resolver(self.study, self.as_dict())
        identity_version = spec.get("identity_version")
        if identity_version not in SUPPORTED_IDENTITY_VERSIONS:
            raise ValueError(
                "family resolver did not produce a supported versioned identity request"
            )
        if (
            identity_version == IDENTITY_VERSION
            and spec.get("resolved_spec_schema") != "ml4t.resolved-spec/v1"
        ):
            raise ValueError("family resolver did not produce the current resolved-spec schema")
        if spec.get("family") != self.family or spec.get("label") != self.label:
            raise ValueError("family resolver changed the requested family or label")
        if spec.get("execution_tier") != self.execution_tier.value:
            raise ValueError("family resolver changed the execution tier")
        return ResolvedModelRequest(self.study, self.family, spec, context)

    def run(self) -> ModelRun:
        from .execution import run_models

        return run_models(self.study, requests=[self]).runs[0]


def _family_module(family: str):
    return get_adapter("model", family)
