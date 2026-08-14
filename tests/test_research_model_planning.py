from types import SimpleNamespace
from typing import cast

import pytest

from case_studies.research.contracts import ExecutionTier
from case_studies.research.identity import ResolvedSpec
from case_studies.research.model_planning import plan_models
from case_studies.research.models import ModelRequest
from case_studies.research.workspace import Study
from case_studies.utils import gbm, linear, tabular_dl


def _request(
    study,
    config_name: str,
    *,
    execution_tier: ExecutionTier = ExecutionTier.CANONICAL,
) -> ModelRequest:
    return ModelRequest(
        study=study,
        family="planned_family",
        label="target",
        config_name=config_name,
        overrides={},
        cv=None,
        execution_tier=execution_tier,
        preview_reductions={"folds": [0]} if execution_tier is ExecutionTier.PREVIEW else {},
    )


def _spec(config_name: str) -> dict:
    return ResolvedSpec.create(
        family="planned_family",
        label="target",
        seed=7,
        computation={
            "checkpoint_schedule": [
                {"kind": "epoch", "value": 5},
                {"kind": "epoch", "value": 10},
            ],
            "cv": {"folds": [0, 1]},
            "feature_artifacts": {"features": "digest"},
            "label_artifact": {"name": "target", "digest": "label-digest"},
            "model": {"class": "FixtureModel", "params": {"variant": config_name}},
        },
        provenance={"device": "cpu"},
        config_name=config_name,
        execution_tier="canonical",
    ).as_dict()


def test_plan_models_resolves_one_family_batch_and_declares_every_checkpoint(monkeypatch) -> None:
    study = cast(Study, SimpleNamespace())
    requests = [_request(study, "first"), _request(study, "second")]
    calls = []

    def batch_planner(received_study, request_dicts):
        calls.append((received_study, request_dicts))
        specs = tuple(_spec(request["config_name"]) for request in request_dicts)
        return specs, request_dicts

    monkeypatch.setattr(
        "case_studies.research.model_planning.get_adapter",
        lambda kind, family: SimpleNamespace(plan_model_requests=batch_planner),
    )

    plan = plan_models(study, requests=requests)

    assert len(calls) == 1
    assert calls[0][0] is study
    assert [request["config_name"] for request in calls[0][1]] == ["first", "second"]
    assert len(plan.expected_training_hashes) == 2
    assert len(plan.expected_prediction_hashes) == 4
    assert [(member.config_name, member.checkpoint_value) for member in plan.members] == [
        ("first", 5),
        ("first", 10),
        ("second", 5),
        ("second", 10),
    ]


def test_model_plan_rejects_execution_identity_drift(monkeypatch) -> None:
    study = cast(Study, SimpleNamespace())
    request = _request(study, "first")

    def run_model_plan(received_study, payload):
        return (
            SimpleNamespace(
                training=SimpleNamespace(hash=plan.expected_training_hashes[0]),
                predictions=(SimpleNamespace(hash="wrong-checkpoint"),),
            ),
        )

    monkeypatch.setattr(
        "case_studies.research.model_planning.get_adapter",
        lambda kind, family: SimpleNamespace(
            plan_model_requests=lambda received_study, request_dicts: (
                (_spec("first"),),
                request_dicts,
            ),
            run_model_plan=run_model_plan,
        ),
    )
    plan = plan_models(study, requests=[request])

    with pytest.raises(RuntimeError, match="planned checkpoint population"):
        plan.run()


def test_preview_model_plan_rejects_official_population_before_registry_write(
    monkeypatch,
) -> None:
    study = cast(Study, SimpleNamespace())
    request = _request(study, "first", execution_tier=ExecutionTier.PREVIEW)
    preview_spec = _spec("first")
    preview_spec["execution_tier"] = "preview"
    monkeypatch.setattr(
        "case_studies.research.model_planning.get_adapter",
        lambda kind, family: SimpleNamespace(
            plan_model_requests=lambda received_study, request_dicts: (
                (preview_spec,),
                request_dicts,
            )
        ),
    )

    plan = plan_models(study, requests=[request])

    with pytest.raises(ValueError, match="preview model plans"):
        plan.create_population(name="invalid-preview-population")


@pytest.mark.parametrize(
    ("module", "batch_name"),
    [(linear, "_run_batch_group"), (gbm, "_run_gbm_batch_group")],
)
def test_planned_fold_major_runner_attempts_later_group_after_failure(
    monkeypatch, module, batch_name
) -> None:
    calls = []

    def observed_batch(study, indexed_requests, key, base, **kwargs):
        calls.append(key)
        if key == "first":
            raise RuntimeError("injected compatibility-group failure")
        return [SimpleNamespace(index=1, error=None, result=object())]

    monkeypatch.setattr(module, batch_name, observed_batch)
    payload = (
        ("first", [(0, {})], object(), {}),
        ("second", [(1, {})], object(), {}),
    )

    with pytest.raises(RuntimeError, match="injected compatibility-group failure"):
        module.run_model_plan(SimpleNamespace(), payload)

    assert calls == ["first", "second"]


def test_planned_tabm_runner_attempts_later_group_after_failure(monkeypatch) -> None:
    calls = []

    def observed_batch(study, items):
        key = items[0][1]
        calls.append(key)
        if key == "first":
            raise RuntimeError("injected compatibility-group failure")
        return {1: object()}

    monkeypatch.setattr(tabular_dl, "_tabm_execution_key", lambda spec: spec)
    monkeypatch.setattr(tabular_dl, "_run_tabm_compatible_group", observed_batch)
    payload = ((0, "first", object()), (1, "second", object()))

    with pytest.raises(RuntimeError, match="injected compatibility-group failure"):
        tabular_dl.run_model_plan(cast(Study, SimpleNamespace()), payload)

    assert calls == ["first", "second"]
