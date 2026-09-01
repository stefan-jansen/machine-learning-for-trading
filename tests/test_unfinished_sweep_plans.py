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
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import (
    OfficialPopulation,
    predictions_identity,
    sweep_plan_name,
    unfinished_sweep_plans,
)
from case_studies.research.workspace import Study
from case_studies.utils.registry import register_backtest_run
from tests.test_research_workspace import _seed_release


@pytest.fixture
def study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


UNRESTRICTED = predictions_identity(None)


def test_a_plan_that_was_never_written_is_reported(study: Study) -> None:
    """A sweep that has not run, and a sweep that ran before plans were recorded.

    Both leave no population, and neither is a sweep whose results may be sealed into an
    immutable set. Reporting them the same way is deliberate.
    """
    unfinished = unfinished_sweep_plans(
        study, case_study="etfs", labels=["fwd_ret_5d"], stages=["allocation"]
    )
    assert len(unfinished) == 1
    assert unfinished[0].startswith(
        f"fwd_ret_5d allocation (etfs-allocation-fwd_ret_5d-{UNRESTRICTED}):"
    )


def test_a_plan_whose_members_are_not_registered_is_reported(study: Study) -> None:
    """The interrupted sweep: the plan names the whole grid, the registry holds part of it."""
    OfficialPopulation.create(
        study,
        name=f"etfs-allocation-fwd_ret_5d-{UNRESTRICTED}",
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
    assert len(unfinished) == 6


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
        "fwd_ret_5d signal",
        "fwd_ret_5d allocation",
        "fwd_ret_5d risk_overlay",
        "fwd_ret_21d signal",
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
    assert (
        sweep_plan_name("etfs", "fwd_ret_5d", "signal", "abc123")
        == "etfs-baseline-fwd_ret_5d-abc123"
    )
    assert (
        sweep_plan_name("etfs", "fwd_ret_5d", "allocation", "abc123")
        == "etfs-allocation-fwd_ret_5d-abc123"
    )
    assert (
        sweep_plan_name("etfs", "fwd_ret_5d", "risk_overlay", "abc123")
        == "etfs-risk-fwd_ret_5d-abc123"
    )


def _complete_plan(study: Study, *, name: str, alpha: float) -> str:
    """A recorded plan whose one backtest is registered and complete, and its prediction hash.

    Registering it for real is the point: the staleness check runs only after
    `require_complete` passes, so a plan that cannot pass it never reaches the behaviour under
    test.
    """
    training = study.results.register_training(
        {
            "identity_version": 2,
            "family": "linear",
            "label": "fwd_ret_5d",
            "label_artifact": "label-a",
            "feature_artifacts": {"financial": "features-a"},
            "feature_names": ["momentum"],
            "cv": {"folds": [{"fold": 0, "val_start": "2024-01-05"}]},
            "model": {"class": "Ridge", "params": {"alpha": alpha}},
            "numerics": {"seed": 42, "precision": "float64"},
            "execution_tier": "canonical",
            "seed": 42,
        }
    )
    frame = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": ["2024-01-05", "2024-01-05"],
            "fold_id": [0, 0],
            "y_true": [0.01, -0.02],
            "y_score": [0.02 * alpha, -0.01 * alpha],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    backtest_hash = register_backtest_run(
        "etfs",
        prediction.hash,
        {"strategy": {"top_k": 1}, "stage": "allocation", "alpha": alpha},
        returns=pl.DataFrame({"timestamp": [date(2024, 1, 5)], "daily_return": [0.001 * alpha]}),
        metrics={"sharpe": alpha, "sharpe_se_lo": 0.0},
        case_dir=study.root,
    )
    OfficialPopulation.create(study, name=name, member_kind="backtest", members=[backtest_hash])
    return prediction.hash


def _rename_population(study: Study, old: str, new: str) -> None:
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute("UPDATE official_populations SET name = ? WHERE name = ?", (new, old))
        db.commit()


def test_a_sweep_planned_against_other_predictions_is_reported(study: Study) -> None:
    """The premature freeze a refit makes possible, and the one completeness cannot see.

    A plan supersedes only when its own sweep re-runs, so under a fixed name the previous
    generation would still be the plan in force after a refit - and still complete, because its
    members are still registered. Waving it through seals a field holding current baselines and
    none of this label's current allocation rows.

    The plan below is complete. It is simply not the plan for the predictions in force, and the
    name says so, so it is reported the same way a plan that was never written is.
    """
    prediction_hash = _complete_plan(
        study, name=f"etfs-allocation-fwd_ret_5d-{predictions_identity({'p-old'})}", alpha=1.0
    )

    unfinished = unfinished_sweep_plans(
        study,
        case_study="etfs",
        labels=["fwd_ret_5d"],
        stages=["allocation"],
        prediction_hashes={prediction_hash},
    )

    assert len(unfinished) == 1
    assert predictions_identity({prediction_hash}) in unfinished[0]


def test_a_sweep_planned_against_the_predictions_in_force_is_not_reported(study: Study) -> None:
    """The other side of it: the sweep did run against these predictions, so its plan is found."""
    # The plan is recorded under a placeholder, then renamed to the identity of the prediction
    # its own member actually rides - which is only knowable after the member is registered.
    # Naming it from a made-up hash would let this pass while the plan described some other
    # generation, which is the thing it is here to establish.
    prediction_hash = _complete_plan(study, name="etfs-allocation-fwd_ret_5d-pending", alpha=1.0)
    _rename_population(
        study,
        "etfs-allocation-fwd_ret_5d-pending",
        f"etfs-allocation-fwd_ret_5d-{predictions_identity({prediction_hash})}",
    )

    assert (
        unfinished_sweep_plans(
            study,
            case_study="etfs",
            labels=["fwd_ret_5d"],
            stages=["allocation"],
            prediction_hashes={prediction_hash},
        )
        == []
    )


def test_a_refit_that_only_adds_a_prediction_is_still_reported(study: Study) -> None:
    """The direction every inference over a plan's members missed.

    Asking whether the members ride predictions still in force answers for removals only: a
    prediction the refit *added* leaves every existing member riding a current prediction, and
    the backtests that would ride the new one do not exist until the sweep runs again. Both
    versions of that check passed this state. A digest of the whole set moves on an addition
    exactly as it does on a removal.
    """
    kept = "p-kept"
    _complete_plan(
        study, name=f"etfs-allocation-fwd_ret_5d-{predictions_identity({kept})}", alpha=1.0
    )

    unfinished = unfinished_sweep_plans(
        study,
        case_study="etfs",
        labels=["fwd_ret_5d"],
        stages=["allocation"],
        prediction_hashes={kept, "p-added-by-the-refit"},
    )

    assert len(unfinished) == 1


def test_without_populations_in_force_completeness_is_the_whole_check(study: Study) -> None:
    """A case study that declares no prediction populations has no generation to be behind.

    Its plans are named for that state rather than for a digest, so they keep being found.
    """
    _complete_plan(study, name=f"etfs-allocation-fwd_ret_5d-{UNRESTRICTED}", alpha=1.0)

    assert (
        unfinished_sweep_plans(
            study, case_study="etfs", labels=["fwd_ret_5d"], stages=["allocation"]
        )
        == []
    )
