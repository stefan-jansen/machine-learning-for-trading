from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from case_studies.utils.registry.specs import (
    canonical_json,
    prediction_hash_from_parts,
    training_hash_from_spec,
)

from .adapters import get_adapter
from .contracts import ExecutionTier
from .models import ModelRequest

if TYPE_CHECKING:
    from .execution import ModelExecution
    from .population import OfficialPopulation
    from .workspace import Study


@dataclass(frozen=True)
class PlannedModel:
    family: str
    label: str
    config_name: str
    training_hash: str
    checkpoint_kind: str
    checkpoint_value: int | None
    prediction_hash: str
    spec_json: str


@dataclass(frozen=True)
class _FamilyPlan:
    family: str
    indexes: tuple[int, ...]
    payload: Any


@dataclass(frozen=True)
class ModelPlan:
    study: Study
    requests: tuple[ModelRequest, ...]
    members: tuple[PlannedModel, ...]
    execution_tier: ExecutionTier
    _family_plans: tuple[_FamilyPlan, ...] = field(repr=False)

    @property
    def expected_training_hashes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(member.training_hash for member in self.members))

    @property
    def expected_prediction_hashes(self) -> tuple[str, ...]:
        return tuple(member.prediction_hash for member in self.members)

    def create_population(
        self,
        *,
        name: str,
        supersedes: str | None = None,
    ) -> OfficialPopulation:
        """Persist the complete canonical checkpoint population before execution."""
        from .population import OfficialPopulation

        if self.execution_tier is not ExecutionTier.CANONICAL:
            raise ValueError("preview model plans cannot create an official population")
        return OfficialPopulation.create(
            self.study,
            name=name,
            member_kind="prediction",
            members=self.expected_prediction_hashes,
            supersedes=supersedes,
        )

    def run(self) -> ModelExecution:
        from .execution import ModelExecution

        ordered_runs: list[Any | None] = [None] * len(self.requests)
        failures = []
        for family_plan in self._family_plans:
            module = get_adapter("model", family_plan.family)
            batch_runner = getattr(module, "run_model_plan", None)
            if callable(batch_runner):
                try:
                    runs = tuple(batch_runner(self.study, family_plan.payload))
                except Exception as error:
                    failures.append(error)
                    continue
            else:
                completed = []
                for request in family_plan.payload:
                    try:
                        completed.append(request.run())
                    except Exception as error:
                        failures.append(error)
                        completed.append(None)
                runs = tuple(completed)
            if len(runs) != len(family_plan.indexes):
                raise RuntimeError(
                    f"{family_plan.family!r} planned runner returned {len(runs)} results for "
                    f"{len(family_plan.indexes)} requests"
                )
            for index, run in zip(family_plan.indexes, runs, strict=True):
                if run is not None:
                    ordered_runs[index] = run
        if failures:
            raise failures[0]
        if any(run is None for run in ordered_runs):
            raise RuntimeError("planned model execution did not produce every result")
        runs = tuple(run for run in ordered_runs if run is not None)
        actual_predictions = tuple(
            prediction.hash for run in runs for prediction in run.predictions
        )
        actual_training = tuple(run.training.hash for run in runs)
        if actual_training != self.expected_training_hashes:
            raise RuntimeError("model execution changed its planned training population")
        if actual_predictions != self.expected_prediction_hashes:
            raise RuntimeError("model execution changed its planned checkpoint population")
        catalog_rows = self.study.predictions.table(include_preview=True).filter(
            pl.col("prediction_hash").is_in(actual_predictions)
        )
        diagnostics = tuple(
            {"status": "completed", "training_hash": run.training.hash, **run.diagnostics}
            for run in runs
        )
        return ModelExecution(runs, catalog_rows, diagnostics)


def _planned_members(spec: dict[str, Any]) -> tuple[PlannedModel, ...]:
    computation = spec.get("computation", spec)
    schedule = computation.get("checkpoint_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("resolved model request has no checkpoint schedule")
    training_hash = training_hash_from_spec(spec)
    spec_json = canonical_json(spec)
    members = []
    for checkpoint in schedule:
        kind = checkpoint.get("kind")
        value = checkpoint.get("value")
        if not isinstance(kind, str) or not kind:
            raise ValueError("resolved model checkpoint has no kind")
        if value is not None and not isinstance(value, int):
            raise TypeError("resolved model checkpoint value must be an integer or null")
        members.append(
            PlannedModel(
                family=str(spec["family"]),
                label=str(spec["label"]),
                config_name=str(spec["config_name"]),
                training_hash=training_hash,
                checkpoint_kind=kind,
                checkpoint_value=value,
                prediction_hash=prediction_hash_from_parts(
                    training_hash,
                    value,
                    "validation",
                    checkpoint_kind=kind,
                    identity_version=spec["identity_version"],
                ),
                spec_json=spec_json,
            )
        )
    return tuple(members)


def plan_models(
    study: Study, *, requests: tuple[ModelRequest, ...] | list[ModelRequest]
) -> ModelPlan:
    """Resolve every training and checkpoint identity without fitting a model."""
    submitted = tuple(requests)
    if not submitted:
        raise ValueError("plan_models requires at least one request")
    if any(not isinstance(request, ModelRequest) for request in submitted):
        raise TypeError("plan_models requires unresolved ModelRequest objects")
    if any(request.study != study for request in submitted):
        raise ValueError("model request belongs to another study")
    tiers = {request.execution_tier for request in submitted}
    if len(tiers) != 1:
        raise ValueError("one model plan cannot mix canonical and preview requests")

    ordered_specs: list[dict[str, Any] | None] = [None] * len(submitted)
    family_plans = []
    grouped: dict[str, list[tuple[int, ModelRequest]]] = {}
    for index, request in enumerate(submitted):
        grouped.setdefault(request.family, []).append((index, request))
    for family, indexed_requests in grouped.items():
        module = get_adapter("model", family)
        batch_planner = getattr(module, "plan_model_requests", None)
        if callable(batch_planner):
            planned = batch_planner(
                study,
                [request.as_dict() for _, request in indexed_requests],
            )
            if not isinstance(planned, tuple) or len(planned) != 2:
                raise TypeError(f"{family!r} planner must return specifications and a payload")
            specs = tuple(planned[0])
            payload = planned[1]
        else:
            resolved = tuple(request.resolve() for _, request in indexed_requests)
            specs = tuple(request.spec for request in resolved)
            payload = resolved
        if len(specs) != len(indexed_requests):
            raise ValueError(
                f"{family!r} planner returned {len(specs)} specifications for "
                f"{len(indexed_requests)} requests"
            )
        for (index, request), spec in zip(indexed_requests, specs, strict=True):
            if spec.get("family") != request.family or spec.get("label") != request.label:
                raise ValueError("family planner changed the requested family or label")
            if spec.get("execution_tier") != request.execution_tier.value:
                raise ValueError("family planner changed the execution tier")
            ordered_specs[index] = spec
        family_plans.append(
            _FamilyPlan(
                family,
                tuple(index for index, _ in indexed_requests),
                payload,
            )
        )
    if any(spec is None for spec in ordered_specs):
        raise RuntimeError("model planning did not resolve every request")

    members = tuple(
        member for spec in ordered_specs if spec is not None for member in _planned_members(spec)
    )
    training_hashes = [member.training_hash for member in members]
    expected_training_count = len(submitted)
    if len(set(training_hashes)) != expected_training_count:
        raise ValueError("model plan contains duplicate training identities")
    prediction_hashes = [member.prediction_hash for member in members]
    if len(set(prediction_hashes)) != len(prediction_hashes):
        raise ValueError("model plan contains duplicate checkpoint identities")
    return ModelPlan(study, submitted, members, tiers.pop(), tuple(family_plans))
