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
    attest_sweep,
    open_sweep_attempt,
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
    prediction_hash, backtest_hash = _registered_member(study, alpha=alpha)
    plan = OfficialPopulation.create(
        study, name=name, member_kind="backtest", members=[backtest_hash]
    )
    attest_sweep(study, plan, open_sweep_attempt(study, plan))
    return prediction_hash


def _registered_member(study: Study, *, alpha: float, label: str = "fwd_ret_5d") -> tuple[str, str]:
    """One complete prediction and the backtest that rides it, with no plan around them."""
    training = study.results.register_training(
        {
            "identity_version": 2,
            "family": "linear",
            "label": label,
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
    return prediction.hash, backtest_hash


def _rename_population(study: Study, old: str, new: str) -> None:
    """Rename a recorded plan, and the attempt and attestation that name it, as one act.

    A real run writes both under its final name; the fixture cannot, because the name carries
    the identity of a prediction that does not exist until the member is registered. Renaming
    only the plan would leave every renamed fixture reporting an absent attestation, which is a
    property of the shim rather than of the behaviour under test.
    """
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute("UPDATE official_populations SET name = ? WHERE name = ?", (new, old))
        # One pattern covers both: an attempt and its attestation are named after the plan's
        # grid, so each is `<plan>-g<generation>-...` and the rename touches only the prefix.
        db.execute(
            "UPDATE official_populations SET name = ? || substr(name, ?) WHERE name LIKE ?",
            (new, len(old) + 1, f"{old}-g%"),
        )
        db.commit()


def test_a_downstream_stage_is_not_charged_against_the_whole_pool(study: Study) -> None:
    """Allocation and risk price what their upstream advanced, not what the pool holds.

    The baseline prices every prediction in force. The stages after it price the leading
    configurations the stage before them advanced - ten of sixty on
    ``sp500_equity_option_analytics``'s direction labels - so charging them against the pool
    reports fifty uncovered members on a chain that is doing exactly what it declares. What
    ties those stages to the current pool is their attestation, whose name folds in the
    upstream plan identities, so a baseline that re-sweeps leaves them unattested.
    """
    advanced, advanced_backtest = _registered_member(study, alpha=1.0)
    not_advanced, not_advanced_backtest = _registered_member(study, alpha=2.0)
    assert not_advanced != advanced
    # The baseline prices both, which is what it is charged with.
    baseline = OfficialPopulation.create(
        study,
        name=f"etfs-baseline-fwd_ret_5d-{predictions_identity({advanced, not_advanced})}",
        member_kind="backtest",
        members=sorted({advanced_backtest, not_advanced_backtest}),
    )
    attest_sweep(study, baseline, open_sweep_attempt(study, baseline))
    allocation = OfficialPopulation.create(
        study,
        name=f"etfs-allocation-fwd_ret_5d-{predictions_identity({advanced, not_advanced})}",
        member_kind="backtest",
        members=[advanced_backtest],
    )
    upstream = (baseline.hash,)
    attest_sweep(study, allocation, open_sweep_attempt(study, allocation, upstream), upstream)

    assert (
        unfinished_sweep_plans(
            study,
            case_study="etfs",
            labels=["fwd_ret_5d"],
            stages=["allocation"],
            prediction_hashes={advanced, not_advanced},
        )
        == []
    )


def test_a_sweep_that_never_priced_a_prediction_in_force_is_reported(study: Study) -> None:
    """The premature freeze a refit makes possible, and the one completeness cannot see.

    A plan supersedes only when its own sweep re-runs, so the previous generation is still the
    plan on record after a refit - and still complete, because its members are still
    registered. Waving it through seals a field holding current baselines and none of this
    label's current allocation rows.

    What says so is the plan's contents, not its name. Its members ride the prediction the
    sweep actually priced, and the prediction now in force is not that one, so the grid has
    never been run against what is being ranked. The report names the member it misses.
    """
    priced, backtest = _registered_member(study, alpha=1.0)
    refitted, _ = _registered_member(study, alpha=2.0)
    assert refitted != priced
    plan = OfficialPopulation.create(
        study,
        name=f"etfs-baseline-fwd_ret_5d-{predictions_identity({priced})}",
        member_kind="backtest",
        members=[backtest],
    )
    attest_sweep(study, plan, open_sweep_attempt(study, plan))

    unfinished = unfinished_sweep_plans(
        study,
        case_study="etfs",
        labels=["fwd_ret_5d"],
        stages=["signal"],
        prediction_hashes={refitted},
    )

    assert len(unfinished) == 1
    assert refitted in unfinished[0]


def test_a_plan_that_priced_a_member_the_pool_dropped_names_that_member(study: Study) -> None:
    """The other direction, and it is a different finding from a sweep that is short.

    A prediction leaving the pool - retired, or dropped by a correction to the cross-sectional
    screen - does not leave this grid short of anything. It moves the ranking the stage after
    it advances from: drop one of ten advancing configurations and the eleventh should now
    advance, while the downstream plan is still complete and still attested, because the
    upstream plan's identity never changed. So the grid has to be re-derived, and ranking the
    members that are left would publish one the rule no longer specifies.

    The digest lookup refused this too, and refused it as "no plan is recorded" - the same
    sentence a sweep that never ran gets. The two need different work and are now reported
    apart, each naming the members it found.
    """
    kept, kept_backtest = _registered_member(study, alpha=1.0)
    dropped, dropped_backtest = _registered_member(study, alpha=2.0)
    plan = OfficialPopulation.create(
        study,
        name=f"etfs-baseline-fwd_ret_5d-{predictions_identity({kept, dropped})}",
        member_kind="backtest",
        members=sorted({kept_backtest, dropped_backtest}),
    )
    attest_sweep(study, plan, open_sweep_attempt(study, plan))

    unfinished = unfinished_sweep_plans(
        study,
        case_study="etfs",
        labels=["fwd_ret_5d"],
        stages=["signal"],
        # The screen dropped one member this plan priced.
        prediction_hashes={kept},
    )

    assert len(unfinished) == 1
    assert dropped in unfinished[0]
    assert "no longer admits" in unfinished[0]
    assert kept not in unfinished[0]


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


def test_a_plan_is_charged_for_its_own_label_and_not_for_the_others(study: Study) -> None:
    """The pool a notebook hands down spans every declared label; a plan is per label.

    ``prediction_members_in_force`` asks the registry what is published and screens it, and
    neither step knows which label the sweep about to run is for. Charging a plan against the
    whole pool charges the ``fwd_ret_5d`` sweep for four labels it was never asked to run: on
    ``sp500_equity_option_analytics`` that reads as 579 uncovered members where one is.
    """
    mine, my_backtest = _registered_member(study, alpha=1.0, label="fwd_ret_5d")
    other, _ = _registered_member(study, alpha=1.0, label="fwd_ret_21d")
    assert other != mine
    plan = OfficialPopulation.create(
        study,
        name=f"etfs-baseline-fwd_ret_5d-{predictions_identity({mine})}",
        member_kind="backtest",
        members=[my_backtest],
    )
    attest_sweep(study, plan, open_sweep_attempt(study, plan))

    assert (
        unfinished_sweep_plans(
            study,
            case_study="etfs",
            labels=["fwd_ret_5d"],
            stages=["signal"],
            prediction_hashes={mine, other},
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
        study, name=f"etfs-baseline-fwd_ret_5d-{predictions_identity({kept})}", alpha=1.0
    )

    unfinished = unfinished_sweep_plans(
        study,
        case_study="etfs",
        labels=["fwd_ret_5d"],
        stages=["signal"],
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


def test_a_complete_plan_says_nothing_about_the_window_its_members_ran_on(study: Study) -> None:
    """Why a sweep that reported failures may not hand its plan to the freeze.

    `require_complete` asks whether each member is registered, of the declared kind, and whole
    on its own terms. It does not and cannot ask whether the member was produced over the
    prediction window in force - that is a comparison between two artifacts, and a registered
    backtest is complete against its own.

    The state this rules out is the one the baseline sweep hits. A member whose registered
    artifact covers a different window is re-run with `force_rebacktest`, the immutability guard
    refuses to overwrite the artifact, and the run fails. The plan is then *still* complete,
    because the stale row is still a registered, whole backtest. Counting that failure and
    carrying on would freeze a field holding it. So the notebook raises on any failure rather
    than relying on the plan to notice, and this is the test that says why it has to.

    Both members below are complete, and their return series cover disjoint windows.
    """
    name = f"etfs-baseline-fwd_ret_5d-{UNRESTRICTED}"
    _complete_plan(study, name=name, alpha=1.0)
    OfficialPopulation.one(study, name=name).require_complete()

    training = study.results.register_training(
        {
            "identity_version": 2,
            "family": "linear",
            "label": "fwd_ret_5d",
            "label_artifact": "label-b",
            "feature_artifacts": {"financial": "features-b"},
            "feature_names": ["momentum"],
            "cv": {"folds": [{"fold": 0, "val_start": "2020-06-01"}]},
            "model": {"class": "Ridge", "params": {"alpha": 2.0}},
            "numerics": {"seed": 42, "precision": "float64"},
            "execution_tier": "canonical",
            "seed": 42,
        }
    )
    frame = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": ["2020-06-01"],
            "fold_id": [0],
            "y_true": [0.01],
            "y_score": [0.02],
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
    # A year away from the first member's single session, which is what a stale artifact looks
    # like: the same identity, a return series over a window nothing in force covers.
    other_window = register_backtest_run(
        "etfs",
        prediction.hash,
        {"strategy": {"top_k": 1}, "stage": "signal", "alpha": 2.0},
        returns=pl.DataFrame({"timestamp": [date(2020, 6, 1)], "daily_return": [0.002]}),
        metrics={"sharpe": 2.0, "sharpe_se_lo": 0.0},
        case_dir=study.root,
    )
    _rename_population(study, name, f"{name}-retired")
    _second = OfficialPopulation.create(
        study,
        name=name,
        member_kind="backtest",
        members=[other_window],
    )
    attest_sweep(study, _second, open_sweep_attempt(study, _second))

    # Complete either way. Nothing in the plan distinguishes the two.
    assert OfficialPopulation.one(study, name=name).require_complete() == (other_window,)


def test_a_plan_superseded_by_a_wider_grid_does_not_inherit_the_old_grid_s_attestation(
    study: Study,
) -> None:
    """A sweep that declares more configurations has not run because its predecessor did.

    The plan name carries the predictions and not the grid, so a sweep whose declared
    configurations change - an allocator added, a schedule withdrawn - supersedes under the
    same name. Numbering attempts by that name alone let the retired grid's successful attempt
    stand as the latest one on record, and the freeze read a grid that had never executed as
    finished. Attempts are named after the grid, so the second generation starts at none.
    """
    name = "etfs-allocation-fwd_ret_5d-pending"
    prediction_hash = _complete_plan(study, name=name, alpha=1.0)
    first = OfficialPopulation.one(study, name=name)
    widened = register_backtest_run(
        "etfs",
        prediction_hash,
        {"strategy": {"top_k": 1}, "stage": "allocation", "alpha": 3.0},
        returns=pl.DataFrame({"timestamp": [date(2024, 1, 5)], "daily_return": [0.003]}),
        metrics={"sharpe": 3.0, "sharpe_se_lo": 0.0},
        case_dir=study.root,
    )
    OfficialPopulation.create(
        study,
        name=name,
        member_kind="backtest",
        members=[*first.members, widened],
        supersedes=first.hash,
    )
    _rename_population(
        study, name, f"etfs-allocation-fwd_ret_5d-{predictions_identity({prediction_hash})}"
    )

    unfinished = unfinished_sweep_plans(
        study,
        case_study="etfs",
        labels=["fwd_ret_5d"],
        stages=["allocation"],
        prediction_hashes={prediction_hash},
    )

    assert len(unfinished) == 1
    assert "no attestation" in unfinished[0]


def test_a_pool_that_returns_to_an_earlier_generation_finds_its_own_plan(study: Study) -> None:
    """A -> B -> A. The plan that covers the pool in force is not the most recent one.

    ``OfficialPopulation.create`` is idempotent on membership, so re-publishing A's grid
    answers with A's existing population and its original ``created_at``. Taking the most
    recently published plan for the stage therefore hands back B's, whose grid prices a
    prediction A's pool does not admit, and the sweep that does cover the pool is reported
    missing. Re-running it cannot clear that: the re-run reuses A's snapshot and its timestamp
    again, so the report survives every remedy it suggests.

    Both plans stay recorded here and the pool is A's. What closes it is asking each recorded
    plan in turn rather than only the newest.
    """
    first, first_backtest = _registered_member(study, alpha=1.0)
    second, second_backtest = _registered_member(study, alpha=2.0)

    pool_a = {first}
    pool_b = {first, second}
    for pool, members in (
        (pool_a, [first_backtest]),
        (pool_b, sorted({first_backtest, second_backtest})),
    ):
        plan = OfficialPopulation.create(
            study,
            name=f"etfs-baseline-fwd_ret_5d-{predictions_identity(pool)}",
            member_kind="backtest",
            members=members,
        )
        attest_sweep(study, plan, open_sweep_attempt(study, plan))

    # B is the most recently published plan, and it prices a prediction A's pool has dropped.
    assert (
        unfinished_sweep_plans(
            study,
            case_study="etfs",
            labels=["fwd_ret_5d"],
            stages=["signal"],
            prediction_hashes=pool_b,
        )
        == []
    )
    assert (
        unfinished_sweep_plans(
            study,
            case_study="etfs",
            labels=["fwd_ret_5d"],
            stages=["signal"],
            prediction_hashes=pool_a,
        )
        == []
    ), "the pool is back on A, whose plan is recorded, complete and attested"


def test_when_no_recorded_plan_covers_the_pool_the_most_recent_one_is_reported(
    study: Study,
) -> None:
    """Trying every recorded plan must not turn a real gap into silence.

    Two plans, neither covering the pool in force, and the report names the most recently
    published one - so the caller is told about the grid it would actually re-run.
    """
    first, first_backtest = _registered_member(study, alpha=1.0)
    second, second_backtest = _registered_member(study, alpha=2.0)
    third, _third_backtest = _registered_member(study, alpha=3.0)

    for pool, members in (
        ({first}, [first_backtest]),
        ({second}, [second_backtest]),
    ):
        plan = OfficialPopulation.create(
            study,
            name=f"etfs-baseline-fwd_ret_5d-{predictions_identity(pool)}",
            member_kind="backtest",
            members=members,
        )
        attest_sweep(study, plan, open_sweep_attempt(study, plan))

    unfinished = unfinished_sweep_plans(
        study,
        case_study="etfs",
        labels=["fwd_ret_5d"],
        stages=["signal"],
        prediction_hashes={first, second, third},
    )
    assert len(unfinished) == 1
    assert third in unfinished[0], unfinished[0]


def test_an_unfinished_latest_sweep_is_not_answered_by_its_predecessor(study: Study) -> None:
    """The reachable path, and the reason the fallback is narrowed to one failure.

    A plan name hashes the whole prediction pool while coverage is checked per label, so
    changing one label's predictions mints a new plan name for every *other* label whose
    predictions did not move. Here `fwd_ret_5d` is untouched and `fwd_ret_21d` gains a
    prediction, which renames `fwd_ret_5d`'s plan. The sweep under the new name is then
    interrupted - it publishes its grid before executing it, which is what the plan is for -
    and its predecessor covers exactly the same predictions.

    Falling past an incomplete plan to that predecessor reports an interrupted sweep as
    finished, which is the one thing this module exists to prevent.
    """
    kept, kept_backtest = _registered_member(study, alpha=1.0)
    other, other_backtest = _registered_member(study, alpha=2.0, label="fwd_ret_21d")
    later, _later_backtest = _registered_member(study, alpha=3.0, label="fwd_ret_21d")

    finished = OfficialPopulation.create(
        study,
        name=f"etfs-baseline-fwd_ret_5d-{predictions_identity({kept, other})}",
        member_kind="backtest",
        members=[kept_backtest],
    )
    attest_sweep(study, finished, open_sweep_attempt(study, finished))

    # The same label, the same predictions, a new name because another label's pool moved -
    # and this sweep publishes its grid and then does not finish it.
    OfficialPopulation.create(
        study,
        name=f"etfs-baseline-fwd_ret_5d-{predictions_identity({kept, other, later})}",
        member_kind="backtest",
        members=sorted({kept_backtest, "cccc55556666"}),
    )

    unfinished = unfinished_sweep_plans(
        study,
        case_study="etfs",
        labels=["fwd_ret_5d"],
        stages=["signal"],
        prediction_hashes={kept, other, later},
    )
    assert len(unfinished) == 1, unfinished
    assert "cccc55556666" in unfinished[0], unfinished[0]
