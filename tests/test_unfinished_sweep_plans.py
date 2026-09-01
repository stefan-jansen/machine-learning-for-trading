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


def test_a_label_that_stopped_at_its_baseline_is_not_waited_for() -> None:
    """The funnel drops a dominated label after its baseline, and nothing waits on it.

    Requiring a plan from every declared label makes the field unfreezable until sweeps nobody
    intended are run. A deliberate drop and a sweep that has not been started are the same
    registry state and want the same answer, which is why this is read off the field rather
    than declared.
    """
    reached = {"fwd_ret_21d": {"signal", "allocation", "risk_overlay"}, "fwd_ret_5d": {"signal"}}
    assert advancing_labels(reached) == ["fwd_ret_21d"]


def test_a_label_whose_sweep_was_interrupted_is_still_waited_for() -> None:
    """The case a declaration-based test would wave through.

    An interrupted sweep has rows past the baseline, so it is waited for, and its plan - which
    it published before executing - is incomplete, so `unfinished_sweep_plans` reports it. This
    is the combination that would otherwise seal a partial field into an immutable set.
    """
    assert advancing_labels({"fwd_ret_5d": {"signal", "allocation"}}) == ["fwd_ret_5d"]


def test_no_label_past_the_baseline_waits_on_nothing() -> None:
    assert advancing_labels({"fwd_ret_5d": {"signal"}, "fwd_ret_21d": {"signal"}}) == []
