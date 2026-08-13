from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from case_studies.utils.registry.specs import IDENTITY_VERSION, training_hash_from_spec

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
        if spec.get("identity_version") != IDENTITY_VERSION:
            raise ValueError(
                f"family resolver did not produce an identity-version-{IDENTITY_VERSION} request"
            )
        if spec.get("resolved_spec_schema") != "ml4t.resolved-spec/v1":
            raise ValueError("family resolver did not produce the current resolved-spec schema")
        if spec.get("family") != self.family or spec.get("label") != self.label:
            raise ValueError("family resolver changed the requested family or label")
        if spec.get("execution_tier") != self.execution_tier.value:
            raise ValueError("family resolver changed the execution tier")
        return ResolvedModelRequest(self.study, self.family, spec, context)

    def run(self) -> ModelRun:
        return self.resolve().run()


def _family_module(family: str):
    return get_adapter("model", family)
