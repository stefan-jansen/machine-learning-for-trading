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


def _planned_spec(config_name: str) -> dict:
    """A spec with the blocks `planned_model_plan` reads, shaped like a registered one."""
    return ResolvedSpec.create(
        family="planned_family",
        label="target",
        seed=7,
        computation={
            "checkpoint_schedule": [{"kind": "epoch", "value": 5}],
            "cv": {
                "folds": [
                    {"fold": 0, "val_start": "2021-01-04", "val_end": "2021-12-31"},
                    {"fold": 1, "val_start": "2022-01-03", "val_end": "2022-12-30"},
                ]
            },
            "expected_prediction_keys": {"digest": "abc", "n_folds": 2, "n_rows": 4096},
            "feature_artifacts": {"features": "digest"},
            "feature_names": ["a", "b", "c"],
            "label_artifact": {"name": "target", "digest": "label-digest"},
            "model": {"class": "FixtureModel", "params": {"variant": config_name}},
            "task": {"type": "regression"},
        },
        provenance={"device": "cpu"},
        config_name=config_name,
        execution_tier="canonical",
    ).as_dict()


def _plan_of(study, config_names, monkeypatch):
    def batch_planner(_study, request_dicts):
        specs = tuple(_planned_spec(request["config_name"]) for request in request_dicts)
        return specs, request_dicts

    monkeypatch.setattr(
        "case_studies.research.model_planning.get_adapter",
        lambda kind, family: SimpleNamespace(plan_model_requests=batch_planner),
    )
    return plan_models(study, requests=[_request(study, name) for name in config_names])


def test_planned_model_plan_reads_the_table_out_of_the_plan(monkeypatch) -> None:
    """The notebook's table without resolving, which is what a large panel cannot afford."""
    from case_studies.research.configs import planned_model_plan

    study = cast(Study, SimpleNamespace())
    frame = planned_model_plan(_plan_of(study, ["first", "second"], monkeypatch))

    assert frame.height == 2
    assert frame.get_column("config_name").to_list() == ["first", "second"]
    assert frame.get_column("eligible_rows").to_list() == [4096, 4096]
    assert frame.get_column("folds").to_list() == [2, 2]
    assert frame.get_column("feature_count").to_list() == [3, 3]
    assert frame.get_column("validation_start").to_list() == ["2021-01-04", "2021-01-04"]
    assert frame.get_column("validation_end").to_list() == ["2022-12-30", "2022-12-30"]
    assert frame.get_column("checkpoints").to_list() == [1, 1]
    assert frame.get_column("training_hash").n_unique() == 2
    # An entity count would need the eligibility keys, which is the memory this path avoids.
    assert "eligible_entities" not in frame.columns


def test_run_official_models_given_a_plan_does_not_plan_a_second_time(monkeypatch) -> None:
    """A notebook shows its plan and then runs it. Planning a large panel twice is not free."""
    from case_studies.research import execution

    study = cast(Study, SimpleNamespace())
    plan = _plan_of(study, ["first", "second"], monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("run_official_models planned again instead of using the plan given")

    monkeypatch.setattr(execution, "plan_models", refuse)
    monkeypatch.setattr(
        execution.ModelPlan,
        "create_population",
        lambda self, *, name, supersedes=None: SimpleNamespace(name=name, supersedes=supersedes),
    )
    forwarded = {}

    def capture(_study, requests, **kwargs):
        forwarded["requests"] = tuple(requests)
        forwarded["expected"] = kwargs["expected"]
        return "execution", kwargs["population"]

    monkeypatch.setattr(execution, "run_official_model_subset", capture)

    _, population = execution.run_official_models(
        study, plan, population_name="p", supersedes="0ff1ce"
    )

    assert population.name == "p"
    # The plan path has to carry the lineage too. A refit under a corrected parameter is a changed
    # population under an existing name, and the registry refuses that write without it.
    assert population.supersedes == "0ff1ce"
    assert forwarded["expected"] == plan.expected_prediction_hashes
    # Unresolved, so run_models takes the fold-major batch path instead of holding one prepared
    # fold set per configuration - the failure the planning path exists to prevent.
    assert forwarded["requests"] == plan.requests
    assert all(isinstance(request, ModelRequest) for request in forwarded["requests"])
