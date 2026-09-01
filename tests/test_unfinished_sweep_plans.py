"""A sweep is finished when the plan it recorded is complete, and not by any reading of rows.

Every cheaper test was tried on the live registry first and each accepted an interruption as a
finished sweep. Stage presence: an interrupted allocation sweep is still present at the
allocation stage. Row counts: the count is simply lower, with nothing to compare it against.
Distinct model configurations against ``top_n_predictions``: the sweep runs configuration-major,
so an interruption leaves whole configurations, and one row per configuration reaches the cut
while most of the grid is absent.

What separates them is the list of backtest identities the sweep computed before executing any
of them. These tests hold ``unfinished_sweep_plans`` to reporting a plan that was never written
and a plan whose members are not all registered, because those are the two shapes an unfinished
sweep leaves behind.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import (
    OfficialPopulation,
    advancing_labels,
    unfinished_sweep_plans,
)
from case_studies.research.workspace import Study
from case_studies.utils.registry import register_backtest_run
from tests.test_research_contract_execution import _publish_prediction
from tests.test_research_workspace import _seed_release


@pytest.fixture
def study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def test_a_plan_that_was_never_written_is_reported(study: Study) -> None:
    """A sweep that has not run, and a sweep that ran before plans were recorded.

    Both leave no population, and neither is a sweep whose results may be sealed into an
    immutable set. Reporting them the same way is deliberate.
    """
    unfinished = unfinished_sweep_plans(
        study, plan_names={"fwd_ret_5d allocation": "etfs-allocation-fwd_ret_5d-v1"}
    )
    assert len(unfinished) == 1
    assert unfinished[0].startswith("fwd_ret_5d allocation:")


def test_a_plan_whose_members_are_not_registered_is_reported(study: Study) -> None:
    """The interrupted sweep: the plan names the whole grid, the registry holds part of it."""
    OfficialPopulation.create(
        study,
        name="etfs-allocation-fwd_ret_5d-v1",
        member_kind="backtest",
        members=["aaaa11112222", "bbbb33334444"],
    )
    unfinished = unfinished_sweep_plans(
        study, plan_names={"fwd_ret_5d allocation": "etfs-allocation-fwd_ret_5d-v1"}
    )
    assert len(unfinished) == 1
    assert "aaaa11112222" in unfinished[0] and "bbbb33334444" in unfinished[0]


def test_every_named_plan_is_reported_not_only_the_first(study: Study) -> None:
    """The caller is deciding whether to freeze, so it needs the whole remaining list.

    Stopping at the first would turn one sequential run into as many runs as there are
    unfinished plans, each learning about exactly one more.
    """
    unfinished = unfinished_sweep_plans(
        study,
        plan_names={
            f"{label} {stage}": f"etfs-{key}-{label}-v1"
            for label in ("fwd_ret_5d", "fwd_ret_21d")
            for stage, key in (("allocation", "allocation"), ("risk_overlay", "risk"))
        },
    )
    assert len(unfinished) == 4


def test_a_registry_without_the_table_reports_absence_not_a_crash(tmp_path: Path) -> None:
    """A registry predating official populations has no table to query.

    That is an absent plan, not a broken registry, and the caller declines to freeze either
    way. Letting ``OperationalError`` escape would fail the notebook instead.
    """

    class _Study:
        root = tmp_path

    (tmp_path / "run_log").mkdir(parents=True)
    sqlite3.connect(tmp_path / "run_log" / "registry.db").close()

    unfinished = unfinished_sweep_plans(
        _Study(), plan_names={"fwd_ret_5d allocation": "etfs-allocation-fwd_ret_5d-v1"}
    )
    assert len(unfinished) == 1


def test_no_plans_named_is_nothing_unfinished(study: Study) -> None:
    assert unfinished_sweep_plans(study, plan_names={}) == []


def _complete_backtest(study: Study, *, top_k: int) -> str:
    """One registered, complete backtest, so a plan naming it is genuinely complete."""
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=top_k)
    returns = pl.DataFrame({"timestamp": ["2024-01-05"], "return": [0.01]}).with_columns(
        pl.col("timestamp").str.to_date()
    )
    return register_backtest_run(
        "etfs",
        prediction_hash,
        {
            "identity_version": 3,
            "execution_tier": "canonical",
            "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": top_k}},
        },
        stage="signal",
        returns=returns,
        metrics={"sharpe": 1.0},
        case_dir=study.root,
    )


def test_a_changed_plan_is_read_instead_of_the_complete_generation_it_replaced(
    study: Study,
) -> None:
    """The reason the plan is published before the sweep rather than after it.

    Published after, a sweep that has grown leaves the previous generation in force while it
    runs. That generation is complete, so an interrupted re-run under a widened grid reports as
    finished on the strength of a plan it has already replaced, and the freeze seals a field
    the current sweep never produced. Published before, the live generation always describes
    the sweep in flight, and an interruption is visible as members that are not registered.
    """
    first = _complete_backtest(study, top_k=1)
    name = "etfs-allocation-fwd_ret_5d-v1"
    generation_one = OfficialPopulation.create(
        study, name=name, member_kind="backtest", members=[first]
    )
    assert unfinished_sweep_plans(study, plan_names={"fwd_ret_5d allocation": name}) == []

    # The widened grid, published before it runs: the extra member is not registered yet.
    OfficialPopulation.create(
        study,
        name=name,
        member_kind="backtest",
        members=[first, "cccc55556666"],
        supersedes=generation_one.hash,
    )
    unfinished = unfinished_sweep_plans(study, plan_names={"fwd_ret_5d allocation": name})
    assert len(unfinished) == 1
    assert "cccc55556666" in unfinished[0]


def test_a_label_that_stops_after_its_baseline_does_not_block_the_freeze(study: Study) -> None:
    """One label swept to completion, one deliberately left at its baseline.

    Requiring a plan for every declared label makes the field unfreezable until sweeps nobody
    intended are run. The dropped label is declared instead, and the freeze waits only for what
    is actually being produced.
    """
    swept = _complete_backtest(study, top_k=1)
    plans = {
        "fwd_ret_21d": "etfs-allocation-fwd_ret_21d-v1",
        "fwd_ret_5d": "etfs-allocation-fwd_ret_5d-v1",
    }
    OfficialPopulation.create(
        study, name=plans["fwd_ret_21d"], member_kind="backtest", members=[swept]
    )

    advancing = advancing_labels(
        study, allocation_plans=plans, not_advancing={"fwd_ret_5d": "dominated at baseline"}
    )
    assert advancing == ["fwd_ret_21d"]
    assert unfinished_sweep_plans(study, plan_names={"fwd_ret_21d": plans["fwd_ret_21d"]}) == []


def test_a_declared_drop_that_the_registry_contradicts_is_refused(study: Study) -> None:
    """The label was declared dropped and its sweep ran anyway.

    Honouring the declaration would silently discard configurations that exist, so the
    contradiction is raised rather than resolved in either direction.
    """
    swept = _complete_backtest(study, top_k=1)
    plans = {"fwd_ret_5d": "etfs-allocation-fwd_ret_5d-v1"}
    OfficialPopulation.create(
        study, name=plans["fwd_ret_5d"], member_kind="backtest", members=[swept]
    )

    with pytest.raises(ValueError, match="recorded allocation plan"):
        advancing_labels(
            study, allocation_plans=plans, not_advancing={"fwd_ret_5d": "dominated at baseline"}
        )


def test_a_declared_drop_whose_sweep_was_interrupted_is_refused(study: Study) -> None:
    """The one combination that would seal a partial field into an immutable set.

    Plans are published before their sweep runs, so an interruption leaves an *incomplete*
    plan. Testing the declaration against completeness would read that as "no finished sweep
    here" and honour the drop, excluding the label from the wait while its partial rows stayed
    eligible for the field. Existence is the test, not completeness.
    """
    plans = {"fwd_ret_5d": "etfs-allocation-fwd_ret_5d-v1"}
    OfficialPopulation.create(
        study,
        name=plans["fwd_ret_5d"],
        member_kind="backtest",
        members=[_complete_backtest(study, top_k=1), "dddd77778888"],
    )
    assert unfinished_sweep_plans(study, plan_names=plans) != []

    with pytest.raises(ValueError, match="recorded allocation plan"):
        advancing_labels(
            study, allocation_plans=plans, not_advancing={"fwd_ret_5d": "dominated at baseline"}
        )


def test_a_drop_naming_an_undeclared_label_is_refused(study: Study) -> None:
    """A typo or a leftover from a renamed label would silently exclude a real label."""
    with pytest.raises(ValueError, match="not declared by this case study"):
        advancing_labels(
            study,
            allocation_plans={"fwd_ret_5d": "etfs-allocation-fwd_ret_5d-v1"},
            not_advancing={"fwd_ret_1d": "typo"},
        )
