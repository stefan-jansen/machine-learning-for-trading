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
    sweep_plan_name,
    unfinished_sweep_plans,
)
from case_studies.research.workspace import Study
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
        study, case_study="etfs", labels=["fwd_ret_5d"], stages=["allocation"]
    )
    assert len(unfinished) == 1
    assert unfinished[0].startswith("fwd_ret_5d allocation (etfs-allocation-fwd_ret_5d-v1):")


def test_a_plan_whose_members_are_not_registered_is_reported(study: Study) -> None:
    """The interrupted sweep: the plan names the whole grid, the registry holds part of it."""
    OfficialPopulation.create(
        study,
        name="etfs-allocation-fwd_ret_5d-v1",
        member_kind="backtest",
        members=["aaaa11112222", "bbbb33334444"],
    )
    unfinished = unfinished_sweep_plans(
        study, case_study="etfs", labels=["fwd_ret_5d"], stages=["allocation"]
    )
    assert len(unfinished) == 1
    assert "aaaa11112222" in unfinished[0] and "bbbb33334444" in unfinished[0]


def test_every_named_plan_is_reported_not_only_the_first(study: Study) -> None:
    """The caller is deciding whether to freeze, so it needs the whole remaining list.

    Stopping at the first would turn one sequential run into as many runs as there are
    unfinished plans, each learning about exactly one more.
    """
    unfinished = unfinished_sweep_plans(
        study, case_study="etfs", labels=["fwd_ret_5d", "fwd_ret_21d"]
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
        _Study(), case_study="etfs", labels=["fwd_ret_5d"], stages=["allocation"]
    )
    assert len(unfinished) == 1


def test_a_label_with_no_rows_at_all_is_still_waited_for(study: Study) -> None:
    """The premature freeze: whichever label runs first sealing the field on the rest.

    An earlier version waited only on labels that already had rows past their baseline, so a
    label whose sweep had not started was read as a label deliberately dropped after its
    baseline - the two leave the registry in the same state. The set is immutable under its
    name, so the first label to finish would have locked every other one out permanently.

    Every declared label is asked instead. Nothing here has run, so both plans of both labels
    are reported and the caller declines to freeze.
    """
    unfinished = unfinished_sweep_plans(
        study, case_study="etfs", labels=["fwd_ret_5d", "fwd_ret_21d"]
    )
    assert {line.split(" (")[0] for line in unfinished} == {
        "fwd_ret_5d allocation",
        "fwd_ret_5d risk_overlay",
        "fwd_ret_21d allocation",
        "fwd_ret_21d risk_overlay",
    }


def test_the_plan_name_is_the_one_the_sweeps_publish_under() -> None:
    """The freeze and the four notebooks that rebuild the field live have to agree on it.

    Spelled out once here rather than at each call site, because a convention written in five
    places is a convention that can differ in one of them. `risk_overlay` is the stage and
    `risk` is the key its population name carries, and reading either for the other was the
    shape of the mistake this closes.
    """
    assert sweep_plan_name("etfs", "fwd_ret_5d", "allocation") == "etfs-allocation-fwd_ret_5d-v1"
    assert sweep_plan_name("etfs", "fwd_ret_5d", "risk_overlay") == "etfs-risk-fwd_ret_5d-v1"
