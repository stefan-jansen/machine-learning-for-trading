"""A holdout window carries one evaluation, and a row that is not one has an owner.

The holdout prediction notebooks refuse to register a second evaluation on the window, and
the refusal read `row["refitted"]` - so it only ever saw *refits*. A prediction set
published over the holdout window by a model fitted on the validation folds is not a refit,
was therefore invisible to the refusal, and was equally invisible to the deletion that
follows it. Nothing owned removing one.

That is not a hypothetical shape. `us_firm_characteristics` carried two holdout
generations, one of them a validation-fitted model publishing over the window, and the
stale row was removed by hand on 2026-08-30 after a registry backup. The note written at
the time says plainly that the next case study to reach the stage would accumulate the same
second generation and the same silence.

The three answers are governed by different rules and the middle one is the gap:

* a refit of another configuration is a second holdout evaluation, and replacing it is a
  research decision - deleting the rows does not undo having observed them;
* a run whose CV declares something other than the holdout may not be reported as a holdout
  result, and cannot be deleted unattended either: `20_strategy_synthesis/holdout.py`'s
  `generate_holdout` refits on a holdout fold and then registers the predictions under the
  VALIDATION training identity, so this record covers both a validation-fitted model
  published over the window and a real refit filed under the wrong identity;
* a run that records no CV split establishes neither, and deleting on that would destroy a
  result nothing has shown to be wrong.

Two of the three are therefore refusals rather than deletions. The gap was never that
nothing deleted these rows - it was that nothing SAW them.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.strategy_analysis import (
    holdout_generations_to_retire,
    holdout_refit_status,
    registered_holdout_generations,
    training_run_fitted_for_the_holdout,
)

CASE_STUDY = "fixture_case_study"


def _spec(cv_split: str | None) -> str | None:
    if cv_split is None:
        return None
    return json.dumps({"computation": {"cv": {"split": cv_split}}})


def _registry(path: Path, generations: list[tuple[str, str, int, str, str | None]]) -> None:
    """Each generation is ``(prediction_hash, training_hash, checkpoint, config, cv_split)``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT,
                spec_json TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_kind TEXT, checkpoint_value INTEGER
            );
            """
        )
        for prediction_hash, training_hash, checkpoint, config_name, cv_split in generations:
            db.execute(
                "INSERT OR IGNORE INTO training_runs VALUES (?, 'gbm', ?, 'fwd_ret_5d', ?)",
                (training_hash, config_name, _spec(cv_split)),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'holdout', 'iteration', ?)",
                (prediction_hash, training_hash, checkpoint),
            )


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    return tmp_path / CASE_STUDY


THIS_GENERATION = ("train_current", ("iteration", 500))


def test_a_validation_fitted_row_is_found_rather_than_filtered_out(case_dir: Path) -> None:
    """The gap itself.

    `pred_stale` declares a validation CV and publishes over the holdout window. The
    `refitted` filter this replaces skipped it entirely - it raised no refusal and reached
    nothing after one. It sat beside the real evaluation, readable and quotable, and
    whichever row a downstream resolver reached first became the published number.

    Being seen is the fix. What is then done with it is a refusal, because the same record
    is what `generate_holdout` writes for a genuine refit.
    """
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("pred_stale", "train_stale", 200, "sae", "validation"),
            ("pred_current", "train_current", 500, "leaves_31_huber", "holdout"),
        ],
    )

    retire = holdout_generations_to_retire(case_dir, this_generation=THIS_GENERATION)

    assert [row["prediction_hash"] for row in retire.not_out_of_sample] == ["pred_stale"]
    assert retire.superseded == ()
    assert retire.unattributable == ()


def test_a_refit_of_another_configuration_is_superseded_rather_than_deletable(
    case_dir: Path,
) -> None:
    """A real second evaluation, which is a research decision and not maintenance."""
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("pred_other", "train_other", 300, "default_mae", "holdout"),
            ("pred_current", "train_current", 500, "leaves_31_huber", "holdout"),
        ],
    )

    retire = holdout_generations_to_retire(case_dir, this_generation=THIS_GENERATION)

    assert [row["prediction_hash"] for row in retire.superseded] == ["pred_other"]
    assert retire.not_out_of_sample == ()


def test_a_run_with_no_recorded_cv_split_is_neither_and_is_reported_as_such(
    case_dir: Path,
) -> None:
    """Absence of evidence, kept out of both actionable buckets.

    This is the case that makes the three-way answer necessary rather than tidy. Under the
    two-valued reading a missing specification answers "not refitted", which is the same
    answer a validation-fitted run gives - so a caller that deletes on it deletes a result
    it cannot show is not a holdout evaluation.
    """
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("pred_unknown", "train_unknown", 100, "sae", None),
            ("pred_current", "train_current", 500, "leaves_31_huber", "holdout"),
        ],
    )

    retire = holdout_generations_to_retire(case_dir, this_generation=THIS_GENERATION)

    assert [row["prediction_hash"] for row in retire.unattributable] == ["pred_unknown"]
    assert retire.not_out_of_sample == ()
    assert retire.superseded == ()


def test_the_generation_being_registered_is_in_no_bucket(case_dir: Path) -> None:
    """An idempotent replay finds itself and must not retire itself.

    The checkpoint is part of the identity, so the same training run at another checkpoint
    is a different configuration on the same window rather than this one again.
    """
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("pred_current", "train_current", 500, "leaves_31_huber", "holdout"),
            ("pred_other_checkpoint", "train_current", 200, "leaves_31_huber", "holdout"),
        ],
    )

    retire = holdout_generations_to_retire(case_dir, this_generation=THIS_GENERATION)

    assert [row["prediction_hash"] for row in retire.superseded] == ["pred_other_checkpoint"]
    assert retire.not_out_of_sample == ()
    assert retire.unattributable == ()


@pytest.mark.parametrize(
    ("cv_split", "expected"),
    [("holdout", "refit"), ("validation", "not_out_of_sample"), (None, "unattributable")],
)
def test_the_status_is_read_from_the_specification(cv_split: str | None, expected: str) -> None:
    """And the two-valued predicate stays the same reading of it.

    `training_run_fitted_for_the_holdout` is what the lineage resolver applies, where "not
    shown to be a refit" is the right default. Expressing it in terms of the three-valued
    answer is what stops the two drifting into different notions of a refit.
    """
    spec = _spec(cv_split)
    assert holdout_refit_status(spec) == expected
    assert training_run_fitted_for_the_holdout(spec) is (expected == "refit")


def test_an_empty_registry_answers_with_nothing(case_dir: Path) -> None:
    """A case study whose holdout stage has not run yet is a normal state, not a defect."""
    _registry(case_dir / "run_log" / "registry.db", [])

    assert registered_holdout_generations(case_dir) == []
    retire = holdout_generations_to_retire(case_dir, this_generation=THIS_GENERATION)
    assert (retire.superseded, retire.not_out_of_sample, retire.unattributable) == ((), (), ())


def test_the_two_valued_predicate_cannot_separate_the_two_ways_of_not_being_a_refit() -> None:
    """Why the answer has to be three-valued, stated on the predicate that was there before.

    `training_run_fitted_for_the_holdout` is the whole vocabulary the notebooks had, and it
    answers False for a validation-fitted run and False for a run that records nothing. One
    of those should be deleted on sight and the other must never be, so a filter written on
    this predicate cannot express either decision - which is why it expressed neither and
    the rows accumulated.
    """
    assert training_run_fitted_for_the_holdout(_spec("validation")) is False
    assert training_run_fitted_for_the_holdout(None) is False
    assert holdout_refit_status(_spec("validation")) != holdout_refit_status(None)
