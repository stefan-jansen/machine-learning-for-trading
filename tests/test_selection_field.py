"""The field a holdout selection is made over is built once and spans every declared label."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from case_studies.research.selection_field import (
    COVERAGE_STAGE,
    FIELD_STAGES,
    label_of,
    open_selection_field,
    predictions_identity,
    resolve_field_members,
    sweep_attempt_name,
    sweep_attestation_name,
)

#: What `resolve_best_backtest_runs` actually returns. The fixture resolver is held to this
#: exactly, because a fixture that hands back `family` and `config_name` would test a frame the
#: production query cannot produce, and every count taken from it would pass here and be zero
#: against a registry.
RESOLVER_COLUMNS = ("backtest_hash", "prediction_hash", "spec_json", "sharpe")


@dataclass
class _Study:
    root: Path
    case_study: str = "fixture"


def _prediction_hash(label: str, config: str) -> str:
    return f"p-{label}-{config}"


def _study_at(
    tmp_path: Path,
    *,
    primary: str,
    variants: list[str],
    configs: tuple[str, ...] = ("c1",),
) -> _Study:
    """A study whose registry knows which configuration every prediction was fitted under.

    The configuration lives on ``training_runs`` in production, so the fixture puts it there
    rather than on the resolver's frame. That is what makes the config count under test the
    same join the notebooks run.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "setup.yaml").write_text(
        "labels:\n"
        f"  primary: {primary}\n"
        "  variants:\n" + "".join(f"    - {name}\n" for name in variants)
    )
    registry_dir = tmp_path / "run_log"
    registry_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(registry_dir / "registry.db")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            training_hash TEXT PRIMARY KEY, label TEXT, family TEXT, config_name TEXT
        );
        CREATE TABLE IF NOT EXISTS prediction_sets (
            prediction_hash TEXT PRIMARY KEY, training_hash TEXT
        );
        CREATE TABLE IF NOT EXISTS backtest_runs (
            backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT
        );
        """
    )
    for label in [primary, *variants]:
        for config in configs:
            training = f"t-{label}-{config}"
            db.execute(
                "INSERT OR REPLACE INTO training_runs VALUES (?, ?, ?, ?)",
                (training, label, "gbm", config),
            )
            db.execute(
                "INSERT OR REPLACE INTO prediction_sets VALUES (?, ?)",
                (_prediction_hash(label, config), training),
            )
    db.commit()
    db.close()
    return _Study(root=tmp_path)


def _rows(
    label: str,
    stage: str,
    *,
    sharpe: float = 1.0,
    configs: tuple[str, ...] = ("c1",),
) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "backtest_hash": [f"{label}-{stage}-{config}" for config in configs],
            "prediction_hash": [_prediction_hash(label, config) for config in configs],
            "spec_json": ["{}"] * len(configs),
            "sharpe": [sharpe] * len(configs),
        }
    )
    assert frame.columns == list(RESOLVER_COLUMNS)
    return frame


def test_field_spans_every_declared_label_and_stage(tmp_path: Path) -> None:
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_10d", "fwd_dir_5d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        assert split == "validation"
        return _rows(label, stage)

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert field.height == 3 * len(FIELD_STAGES)
    assert set(field["backtest_hash"]) == {
        f"{label}-{stage}-c1"
        for label in ("fwd_ret_5d", "fwd_ret_10d", "fwd_dir_5d")
        for stage in FIELD_STAGES
    }


def test_a_dominated_label_may_stop_after_the_baselines(tmp_path: Path) -> None:
    """Dropping a label after the baselines is the point of running them, not an incomplete run.

    Every declared label is backtested equal-weight, which is what makes them comparable. The
    stages after that develop whichever labels the comparison favours, so a label the baselines
    show to be dominated is deliberately not carried into allocation or risk overlay. A rule
    demanding every label in every stage would order backtests whose only purpose is filling a
    matrix, and would refuse to freeze a field that is finished.

    Nothing in the rows records that the drop was a decision, so this is also the shape of a
    sweep that has not started. Neither this function nor any other reading of the registry can
    separate them; what does is the plan each sweep records before it runs, checked by the
    notebook that freezes the field.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_5d" and stage != COVERAGE_STAGE:
            return _rows(label, stage).clear()
        return _rows(label, stage)

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert set(field["backtest_hash"]) == {
        f"fwd_ret_5d-{COVERAGE_STAGE}-c1",
        *(f"fwd_ret_21d-{stage}-c1" for stage in FIELD_STAGES),
    }


def test_a_label_with_no_baseline_refuses_to_freeze(tmp_path: Path) -> None:
    """The sequential-run failure, at the one stage where absence means unfinished.

    A run mid-way through the second label's baseline sweep would otherwise freeze a field that
    excludes it, and nothing can correct that afterwards: the set is immutable under its name,
    and every later run produces the same membership and resolves to it.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_10d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_10d":
            return _rows(label, stage).clear()
        return _rows(label, stage)

    with pytest.raises(RuntimeError, match=r"fwd_ret_10d"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def test_label_comes_from_the_winner_not_the_primary(tmp_path: Path) -> None:
    """What the stages after the selection run under is a property of what won."""
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_risk_adj_5d"])
    db = sqlite3.connect(study.root / "run_log" / "registry.db")
    db.execute(
        "INSERT INTO backtest_runs VALUES (?, ?)",
        ("b1", _prediction_hash("fwd_ret_risk_adj_5d", "c1")),
    )
    db.commit()
    db.close()

    @dataclass
    class _Result:
        hash: str

    assert label_of(study, _Result("b1")) == "fwd_ret_risk_adj_5d"

    with pytest.raises(RuntimeError, match="no label in this registry"):
        label_of(study, _Result("absent"))


def test_an_unrankable_row_does_not_satisfy_coverage(tmp_path: Path) -> None:
    """A null Sharpe cannot be ranked, so it is not a backtest the field can select from.

    Counting it towards coverage freezes a set that ``best_validation_sharpe`` then rejects
    whole, for holding a member it cannot rank - the failure lands after the set is immutable.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        rows = _rows(label, stage)
        if label == "fwd_ret_21d":
            return rows.with_columns(sharpe=pl.lit(None, dtype=pl.Float64))
        return rows

    with pytest.raises(RuntimeError, match="fwd_ret_21d"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def _record_plan(
    study: _Study,
    *,
    name: str,
    members: list[str],
    supersedes: str | None = None,
    attested: bool = True,
    attempt: int = 1,
) -> str:
    """A recorded sweep plan, written straight to the table `OfficialPopulation.one` reads.

    Not through `create`, which requires a writable study and registered members. What is under
    test is the effect of a plan on membership, and that is the same whichever way the row got
    there.

    ``attested`` writes what a finished sweep leaves: an attempt record and the attestation
    for it. It defaults to true; ``attested=False`` writes the attempt alone, which is what a
    run that raised or died part-way leaves, and ``attempt`` selects which attempt number this
    call is writing so a test can put a failure after a success on the same grid.
    """
    population_hash = f"pop-{name}-{len(members)}-{supersedes or 'first'}"
    db = sqlite3.connect(study.root / "run_log" / "registry.db")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS official_populations (
            population_hash TEXT PRIMARY KEY, name TEXT NOT NULL, member_kind TEXT NOT NULL,
            snapshot_json TEXT NOT NULL, supersedes_hash TEXT, created_at TEXT NOT NULL
        )
        """
    )
    db.execute(
        "INSERT OR REPLACE INTO official_populations VALUES (?, ?, ?, ?, ?, ?)",
        (
            population_hash,
            name,
            "backtest",
            json.dumps({"members": members}),
            supersedes,
            "2026-09-01T00:00:00+00:00",
        ),
    )
    records = [sweep_attempt_name(name, attempt)]
    if attested:
        records.append(sweep_attestation_name(name, attempt))
    for record in records:
        db.execute(
            "INSERT OR REPLACE INTO official_populations VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"pop-{record}",
                record,
                "backtest",
                json.dumps({"members": members}),
                None,
                "2026-09-01T00:00:01+00:00",
            ),
        )
    db.commit()
    db.close()
    return population_hash


def test_a_recorded_plan_decides_which_downstream_rows_are_eligible(tmp_path: Path) -> None:
    """A withdrawn allocator's rows keep current predictions, so nothing else excludes them.

    Checking a plan only for completeness left it decorative: the sweep declares a grid, and a
    row from a grid the sweep no longer declares was still eligible to win the selection. The
    plan is what says which rows the published field is made of.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1", "c2"))
    study.results = _Results()
    _record_plan(
        study,
        name=f"fixture-baseline-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-signal-c1", "fwd_ret_5d-signal-c2"],
    )
    _record_plan(
        study,
        name=f"fixture-allocation-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-allocation-c1"],
    )
    _record_plan(
        study,
        name=f"fixture-risk-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-risk_overlay-c1", "fwd_ret_5d-risk_overlay-c2"],
    )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, configs=("c1", "c2"))

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    allocation = {name for name in field["backtest_hash"] if "-allocation-" in name}
    assert allocation == {"fwd_ret_5d-allocation-c1"}
    assert {name for name in field["backtest_hash"] if f"-{COVERAGE_STAGE}-" in name} == {
        "fwd_ret_5d-signal-c1",
        "fwd_ret_5d-signal-c2",
    }


def test_a_superseded_plan_does_not_admit_its_own_members(tmp_path: Path) -> None:
    """Only the generation in force decides membership; the grid it replaced does not."""
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1", "c2"))
    study.results = _Results()
    _record_plan(
        study,
        name=f"fixture-baseline-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-signal-c1", "fwd_ret_5d-signal-c2"],
    )
    _record_plan(
        study,
        name=f"fixture-allocation-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-allocation-c1", "fwd_ret_5d-allocation-c2"],
    )
    first = _record_plan(
        study,
        name=f"fixture-risk-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-risk_overlay-c1"],
    )
    _record_plan(
        study,
        name=f"fixture-risk-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-risk_overlay-c2"],
        supersedes=first,
    )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, configs=("c1", "c2"))

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    overlay = {name for name in field["backtest_hash"] if "-risk_overlay-" in name}
    assert overlay == {"fwd_ret_5d-risk_overlay-c2"}


@dataclass
class _Member:
    """What the field construction needs of a result: an identity, a kind, and whether it is usable."""

    hash: str
    reason: str | None = None
    kind: str = "backtest"

    def completeness(self) -> str | None:
        return self.reason

    @property
    def complete(self) -> bool:
        return self.reason is None


class _Results:
    def __init__(self, incomplete: dict[str, str] | None = None) -> None:
        self._incomplete = incomplete or {}

    def open(self, backtest_hash: str) -> _Member:
        return _Member(backtest_hash, self._incomplete.get(backtest_hash))


def _register_backtests(study: _Study, rows: list[tuple[str, str]]) -> None:
    db = sqlite3.connect(study.root / "run_log" / "registry.db")
    db.executemany("INSERT OR REPLACE INTO backtest_runs VALUES (?, ?)", rows)
    db.commit()
    db.close()


def test_open_selection_field_ranks_live_where_no_set_is_recorded(tmp_path: Path) -> None:
    """The path a reader's clean clone takes, and the one that had no test at all.

    `open_selection_field` returns the frozen set where one exists and rebuilds the same field
    live where none does. Only the frozen branch was ever exercised, so a change to what
    `resolve_field_members` returns broke every notebook that reads the field on a clone while
    every test and every maintainer run stayed green - they all had a recorded set and returned
    before reaching this.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_10d"])
    study.results = _Results()
    _register_backtests(
        study,
        [
            (f"{label}-{stage}-c1", _prediction_hash(label, "c1"))
            for label in ("fwd_ret_5d", "fwd_ret_10d")
            for stage in FIELD_STAGES
        ],
    )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        # The winner is a variant label at the overlay stage, so the label carried forward has
        # to be read off the selection rather than taken from `labels.primary`.
        sharpe = 2.0 if (label, stage) == ("fwd_ret_10d", "risk_overlay") else 1.0
        return _rows(label, stage, sharpe=sharpe)

    field = open_selection_field(
        study,
        case_study="fixture",
        name="fixture:holdout-candidates",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )

    assert field.frozen is False
    assert field.selected.hash == "fwd_ret_10d-risk_overlay-c1"
    assert field.label == "fwd_ret_10d"
    assert len(field.members) == 2 * len(FIELD_STAGES)


def test_open_selection_field_drops_members_it_cannot_open_complete(tmp_path: Path) -> None:
    """`CandidateSet.create` refuses partial members, so the live field has to match.

    Otherwise a reader rebuilding the field would rank over members the frozen set excluded,
    and could select something the published selection never saw.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[])
    study.results = _Results({"fwd_ret_5d-risk_overlay-c1": "no metrics row"})
    _register_backtests(
        study,
        [
            (f"fwd_ret_5d-{stage}-c1", _prediction_hash("fwd_ret_5d", "c1"))
            for stage in FIELD_STAGES
        ],
    )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, sharpe=2.0 if stage == "risk_overlay" else 1.0)

    field = open_selection_field(
        study,
        case_study="fixture",
        name="fixture:holdout-candidates",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert "fwd_ret_5d-risk_overlay-c1" not in field.members
    assert field.selected.hash != "fwd_ret_5d-risk_overlay-c1"


def test_a_live_rebuild_refuses_when_the_predictions_moved_past_the_plans(tmp_path: Path) -> None:
    """The reader's clean clone, rebuilding a field whose sweeps have not caught up.

    `open_selection_field` reads the frozen set where one exists and rebuilds the field live
    where none does, and only the frozen path was gated on the plans. So after a prediction was
    added or replaced, the live path admitted every downstream row it could find - rows from the
    grid the previous predictions implied - and ranked over a membership the freeze would have
    refused. A reader would then see a different selection from the published one, which is the
    divergence this module exists to close.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1", "c2"))
    study.results = _Results()
    _register_backtests(
        study,
        [
            (f"fwd_ret_5d-{stage}-{config}", _prediction_hash("fwd_ret_5d", config))
            for stage in FIELD_STAGES
            for config in ("c1", "c2")
        ],
    )
    # The plans the previous prediction generation implied. Nothing supersedes them - a plan
    # supersedes only when its own sweep re-runs - so they are still the plans on record.
    for key, stage in (("allocation", "allocation"), ("risk", "risk_overlay")):
        _record_plan(
            study,
            name=f"fixture-{key}-fwd_ret_5d-{predictions_identity({'p-before-the-refit'})}",
            members=[f"fwd_ret_5d-{stage}-c1"],
        )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, configs=("c1", "c2"))

    with pytest.raises(RuntimeError, match="has not been run against them"):
        open_selection_field(
            study,
            case_study="fixture",
            name="fixture:holdout-candidates",
            prediction_hashes={"p-after-the-refit"},
            resolve_best_backtest_runs=resolver,
        )


def test_a_live_rebuild_refuses_a_plan_whose_sweep_is_still_running(tmp_path: Path) -> None:
    """The state publishing a plan before its sweep executes deliberately creates.

    Recording the plan afterwards was the alternative, and it is worse: an interrupted sweep
    then records nothing and leaves the previous, smaller, complete plan in force, so a freeze
    reads the interruption as a finished run. Published first, an interruption leaves a current
    plan whose members are not all registered.

    The freeze declines that state through `unfinished_sweep_plans`. A live rebuild has to
    reach the same answer, or a reader who rebuilds the field while the sweep is running ranks
    over the part of the grid that finished and gets a selection the freeze would have refused
    to make.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1", "c2"))
    study.results = _Results({"fwd_ret_5d-allocation-c2": "no metrics row"})
    _record_plan(
        study,
        name=f"fixture-baseline-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-signal-c1", "fwd_ret_5d-signal-c2"],
    )
    # The whole grid is named; only `c1` has finished running.
    _record_plan(
        study,
        name=f"fixture-allocation-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-allocation-c1", "fwd_ret_5d-allocation-c2"],
    )
    _record_plan(
        study,
        name=f"fixture-risk-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-risk_overlay-c1", "fwd_ret_5d-risk_overlay-c2"],
    )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, configs=("c1", "c2"))

    with pytest.raises(ValueError, match="fwd_ret_5d-allocation-c2"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def test_the_baseline_grid_is_planned_like_every_other_stage(tmp_path: Path) -> None:
    """Coverage is a floor, and it was standing in for a completion check it cannot make.

    The baseline was left unplanned on the reading that one rankable row per declared label is
    enough to say it finished. It is a grid - predictions by entry scheme - so an interruption
    leaves the same shape every other interrupted sweep leaves, and a grid that has since
    changed leaves rows that are still rankable and no current plan contains. Both states pass
    coverage, and both are here: `c2`'s baseline is registered, rankable, and not in the plan.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1", "c2"))
    study.results = _Results()
    _record_plan(
        study,
        name=f"fixture-baseline-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-signal-c1"],
    )
    for key, stage in (("allocation", "allocation"), ("risk", "risk_overlay")):
        _record_plan(
            study,
            name=f"fixture-{key}-fwd_ret_5d-{predictions_identity(None)}",
            members=[f"fwd_ret_5d-{stage}-c1", f"fwd_ret_5d-{stage}-c2"],
        )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, configs=("c1", "c2"))

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert {name for name in field["backtest_hash"] if f"-{COVERAGE_STAGE}-" in name} == {
        "fwd_ret_5d-signal-c1"
    }


def test_a_case_study_that_publishes_plans_needs_one_for_its_baseline(tmp_path: Path) -> None:
    """An absent baseline plan is a baseline sweep that has not run against these predictions.

    It reads exactly like the downstream stages, and for the same reason: the rows the stage
    does have belong to another generation, so building the field anyway hands a reader a
    membership the freeze never published.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1",))
    study.results = _Results()
    for key, stage in (("allocation", "allocation"), ("risk", "risk_overlay")):
        _record_plan(
            study,
            name=f"fixture-{key}-fwd_ret_5d-{predictions_identity(None)}",
            members=[f"fwd_ret_5d-{stage}-c1"],
        )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage)

    with pytest.raises(RuntimeError, match="at the 'signal' stage"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def test_a_complete_plan_whose_sweep_failed_does_not_admit_its_members(tmp_path: Path) -> None:
    """The state the raise inside the sweep cannot protect on its own.

    The sweep raises on a failed member, which stops that process. It does not stop the freeze,
    which runs later and usually in another notebook, and it cannot make the plan look
    unfinished: a member that failed because its registered artifact was produced over a
    different prediction window is a registered, internally complete backtest, so
    `require_complete` passes on it. Without a record of the run's own outcome, the next reader
    builds a field from a grid it was told not to trust.

    The attestation is that record. Here the baseline plan is complete and its sweep reported a
    failure, so no attestation was written, and the rebuild refuses rather than admitting the
    grid.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1",))
    study.results = _Results()
    _record_plan(
        study,
        name=f"fixture-baseline-fwd_ret_5d-{predictions_identity(None)}",
        members=["fwd_ret_5d-signal-c1"],
        attested=False,
    )
    for key, stage in (("allocation", "allocation"), ("risk", "risk_overlay")):
        _record_plan(
            study,
            name=f"fixture-{key}-fwd_ret_5d-{predictions_identity(None)}",
            members=[f"fwd_ret_5d-{stage}-c1"],
        )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, configs=("c1",))

    with pytest.raises(ValueError, match="records no attestation"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def test_a_failed_re_run_is_not_covered_by_the_previous_run_s_attestation(tmp_path: Path) -> None:
    """The indelibility a single attestation per plan would have.

    The stale-artifact failure does not change the grid: the planned identities are the same,
    the registered rows are the same, and what differs is that a member's artifact no longer
    matches the prediction window, so the forced re-backtest fails. A sweep that succeeded once
    and then fails that way would, under a name keyed on the plan alone, still be covered by
    the first run's attestation - and the freeze would accept the failed attempt. The record is
    per attempt for that reason, and the reader asks the latest one.

    Attempt 1 succeeded on this exact grid; attempt 2 did not.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1",))
    study.results = _Results()
    name = f"fixture-baseline-fwd_ret_5d-{predictions_identity(None)}"
    _record_plan(study, name=name, members=["fwd_ret_5d-signal-c1"], attempt=1)
    _record_plan(study, name=name, members=["fwd_ret_5d-signal-c1"], attempt=2, attested=False)
    for key, stage in (("allocation", "allocation"), ("risk", "risk_overlay")):
        _record_plan(
            study,
            name=f"fixture-{key}-fwd_ret_5d-{predictions_identity(None)}",
            members=[f"fwd_ret_5d-{stage}-c1"],
        )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, configs=("c1",))

    with pytest.raises(ValueError, match="records no attestation"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def test_a_re_run_that_succeeds_after_a_failure_is_accepted(tmp_path: Path) -> None:
    """The other direction, so the check above is not simply "any failed attempt, ever"."""
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=[], configs=("c1",))
    study.results = _Results()
    for key, stage in (
        ("baseline", "signal"),
        ("allocation", "allocation"),
        ("risk", "risk_overlay"),
    ):
        _record_plan(
            study,
            name=f"fixture-{key}-fwd_ret_5d-{predictions_identity(None)}",
            members=[f"fwd_ret_5d-{stage}-c1"],
            attempt=1,
            attested=False,
        )
        _record_plan(
            study,
            name=f"fixture-{key}-fwd_ret_5d-{predictions_identity(None)}",
            members=[f"fwd_ret_5d-{stage}-c1"],
            attempt=2,
        )

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        return _rows(label, stage, configs=("c1",))

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert set(field["backtest_hash"]) == {
        "fwd_ret_5d-signal-c1",
        "fwd_ret_5d-allocation-c1",
        "fwd_ret_5d-risk_overlay-c1",
    }
