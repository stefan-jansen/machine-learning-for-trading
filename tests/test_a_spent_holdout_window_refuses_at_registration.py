"""One evaluation per holdout window, enforced where the registry is written.

`holdout_generations_to_retire` and `refuse_a_second_look` state the rule. Until now the
rule was applied by whichever notebook had copied it: five of the nine holdout-predictions
notebooks wrote the check out for themselves and four did not, and `fx_pairs` carries two
generations on one window because the notebook that registered the second had nothing to
fire. A guard copied into nine notebooks is a guard that is in five of them.

`register_prediction_set` is the one call every path arrives through, and it knows exactly
what the rule needs: the case directory, the training identity, and the checkpoint.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.registry.registration import register_prediction_set
from case_studies.utils.strategy_analysis import HoldoutWindowSpent

CASE_STUDY = "fixture_case_study"
CARRIER = ("pred_first", "train_first", "ridge_a1000000.0", "holdout")


def _spec(cv_split: str | None) -> str | None:
    if cv_split is None:
        return None
    return json.dumps({"computation": {"cv": {"split": cv_split}}})


def _registry(case_dir: Path, rows: list[tuple[str, str, str, str | None]]) -> None:
    """Each row is ``(prediction_hash, training_hash, config_name, cv_split)``.

    ``identity_version`` is left null so registration stops at the legacy path rather than
    demanding coverage evidence: every test here asserts what happens before that point.
    """
    db_path = case_dir / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT,
                spec_json TEXT, identity_version TEXT, execution_tier TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_kind TEXT, checkpoint_value INTEGER
            );
            """
        )
        for prediction_hash, training_hash, config_name, cv_split in rows:
            db.execute(
                "INSERT OR IGNORE INTO training_runs VALUES (?,'linear',?, 'fwd_ret_5d',?,NULL,NULL)",
                (training_hash, config_name, _spec(cv_split)),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?,?, 'holdout', 'final', NULL)",
                (prediction_hash, training_hash),
            )


def _arriving(case_dir: Path, training_hash: str, config_name: str) -> None:
    """A training run that has not published yet, which is what is about to register."""
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as db:
        db.execute(
            "INSERT OR IGNORE INTO training_runs VALUES (?,'deep_learning',?, 'fwd_ret_5d',?,NULL,NULL)",
            (training_hash, config_name, _spec("holdout")),
        )


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    case_dir = tmp_path / CASE_STUDY
    _registry(case_dir, [CARRIER])
    return case_dir


def _register(case_dir: Path, training_hash: str, *, split: str = "holdout", **kwargs):
    return register_prediction_set(
        CASE_STUDY,
        training_hash,
        split=split,
        checkpoint_kind="final",
        checkpoint_value=None,
        case_dir=case_dir,
        **kwargs,
    )


class TestASecondConfigurationIsRefused:
    def test_a_different_training_identity_on_a_spent_window_refuses(self, case_dir: Path) -> None:
        """The fx_pairs shape: a second configuration arriving at an observed window."""
        _arriving(case_dir, "train_second", "lstm_h64")

        with pytest.raises(HoldoutWindowSpent) as refusal:
            _register(case_dir, "train_second")

        assert "already carries a refit of a different configuration" in str(refusal.value)

    def test_the_refusal_names_both_configurations(self, case_dir: Path) -> None:
        """A reader has to be able to see what was spent and what was about to spend it."""
        _arriving(case_dir, "train_second", "lstm_h64")

        with pytest.raises(HoldoutWindowSpent) as refusal:
            _register(case_dir, "train_second")

        message = str(refusal.value)
        assert "ridge_a1000000.0" in message
        assert "lstm_h64" in message
        assert "pred_first" in message

    def test_naming_the_registered_hash_is_the_way_past_it(self, case_dir: Path) -> None:
        """The override is per generation, so it cannot be set once and left set.

        Getting past the guard is not the same as registering: the call goes on to fail on
        the predictions it was not given. What is asserted is that the refusal is no longer
        the thing stopping it.
        """
        _arriving(case_dir, "train_second", "lstm_h64")

        with pytest.raises(Exception) as outcome:
            _register(case_dir, "train_second", retiring=["pred_first"])

        assert not isinstance(outcome.value, HoldoutWindowSpent)

    def test_an_override_naming_something_absent_is_itself_refused(self, case_dir: Path) -> None:
        """An override that outlives its generation would authorize whatever arrives next."""
        _arriving(case_dir, "train_second", "lstm_h64")

        with pytest.raises(HoldoutWindowSpent, match="which this window does not carry"):
            _register(case_dir, "train_second", retiring=["pred_absent"])


class TestWhatIsNotASecondLook:
    def test_re_registering_this_generation_is_not_refused(self, case_dir: Path) -> None:
        """A holdout notebook re-runs like any other stage.

        The generation about to register is the one already there, so nothing about the
        window has been observed twice. Without this the guard would make every holdout
        notebook single-use, which is a different rule from the one it enforces.
        """
        with pytest.raises(Exception) as outcome:
            _register(case_dir, "train_first")

        assert not isinstance(outcome.value, HoldoutWindowSpent)

    def test_a_validation_registration_is_not_touched(self, case_dir: Path) -> None:
        """The rule is about the holdout window; validation is swept over repeatedly."""
        _arriving(case_dir, "train_second", "lstm_h64")

        with pytest.raises(Exception) as outcome:
            _register(case_dir, "train_second", split="validation")

        assert not isinstance(outcome.value, HoldoutWindowSpent)

    def test_an_empty_window_admits_the_first_evaluation(self, tmp_path: Path) -> None:
        """The control: nothing registered, so nothing to be a second look at."""
        case_dir = tmp_path / CASE_STUDY
        _registry(case_dir, [])
        _arriving(case_dir, "train_only", "lstm_h64")

        with pytest.raises(Exception) as outcome:
            _register(case_dir, "train_only")

        assert not isinstance(outcome.value, HoldoutWindowSpent)


class TestTheOtherTwoBucketsAreNotOverridable:
    def test_a_row_with_no_cv_record_refuses_and_says_why(self, tmp_path: Path) -> None:
        case_dir = tmp_path / CASE_STUDY
        _registry(case_dir, [("pred_mystery", "train_mystery", "unknown_config", None)])
        _arriving(case_dir, "train_second", "lstm_h64")

        with pytest.raises(HoldoutWindowSpent, match="record no CV split"):
            _register(case_dir, "train_second")

    def test_naming_an_unattributable_row_does_not_authorize_it(self, tmp_path: Path) -> None:
        """`retiring` asserts a row may be retired, not that its provenance is known."""
        case_dir = tmp_path / CASE_STUDY
        _registry(case_dir, [("pred_mystery", "train_mystery", "unknown_config", None)])
        _arriving(case_dir, "train_second", "lstm_h64")

        with pytest.raises(HoldoutWindowSpent, match="record no CV split"):
            _register(case_dir, "train_second", retiring=["pred_mystery"])

    def test_a_validation_fitted_row_published_over_the_window_refuses(
        self, tmp_path: Path
    ) -> None:
        case_dir = tmp_path / CASE_STUDY
        _registry(case_dir, [("pred_stale", "train_stale", "sae", "validation")])
        _arriving(case_dir, "train_second", "lstm_h64")

        with pytest.raises(HoldoutWindowSpent, match="declare a CV split other than the holdout"):
            _register(case_dir, "train_second")
