"""What a validation training specification must have done to it before it can fit a holdout.

Three things have to happen, and each is a separate refusal somewhere else in the codebase:
derive the holdout interval, bound its training start at whatever the family's features
actually reach, and recompute the fields the resolver derived per validation fold. A
specification carrying two of the three is not obviously broken - it has a holdout CV and a
manifest, and it fits - which is why ``build_holdout_training_spec`` exists as one call and why
these tests check that all three happened rather than that the function returned something.

The case study is a real one, so the declared holdout window is a declared one. The family is a
stand-in installed at the dispatch point, because what is under test is whether this composes
the family's answers, not what any particular family answers.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from case_studies.research.holdout import build_holdout_training_spec
from tests.test_holdout_cv_derivation import MONTH_ENDS, _validation_spec

CASE_STUDY = "us_firm_characteristics"


def _install_family(
    monkeypatch: pytest.MonkeyPatch,
    *,
    floor: Any,
    rekey: Any,
) -> list[dict[str, Any]]:
    """Install a stand-in family at the hook dispatch point, and record what it was asked.

    ``_holdout_training_floor`` and ``_rekey_holdout_spec`` both resolve the family through
    ``case_studies.research.models._family_module`` at call time, so replacing that replaces
    both hooks at once and leaves the composition under test untouched.
    """
    seen: list[dict[str, Any]] = []

    def holdout_training_floor(study, *, validation_spec):
        seen.append({"hook": "floor", "spec": deepcopy(validation_spec)})
        return floor

    def rekey_holdout_spec(study, spec, *, validation_spec):
        seen.append({"hook": "rekey", "spec": deepcopy(spec)})
        rekey(spec)

    monkeypatch.setattr(
        "case_studies.research.models._family_module",
        lambda family: SimpleNamespace(
            holdout_training_floor=holdout_training_floor,
            rekey_holdout_spec=rekey_holdout_spec,
        ),
    )
    return seen


def _rekey_to_one_fold(spec: dict[str, Any]) -> None:
    """The minimum a family hook must do for ``_require_holdout_keyed_fields`` to accept it."""
    spec["computation"]["expected_prediction_keys"] = {
        "digest": "re-keyed-to-the-holdout-fold",
        "n_rows": 11,
        "n_folds": 1,
    }


def _build(monkeypatch: pytest.MonkeyPatch, *, floor: Any = None) -> tuple[dict, dict, list]:
    validation_spec = _validation_spec()
    validation_spec["computation"]["expected_prediction_keys"] = {
        "digest": "describes-the-validation-folds",
        "n_rows": 33,
        "n_folds": 3,
    }
    seen = _install_family(monkeypatch, floor=floor, rekey=_rekey_to_one_fold)
    holdout_spec = build_holdout_training_spec(
        SimpleNamespace(case_study=CASE_STUDY),
        validation_spec,
        timeline=MONTH_ENDS,
        case_study=CASE_STUDY,
    )
    return validation_spec, holdout_spec, seen


def test_the_families_training_floor_reaches_the_derived_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor is the whole reason the hook exists, and it is invisible in the result shape.

    A composition that derived the interval and then forgot to pass the floor returns a spec
    that looks complete: one fold, correct window, re-keyed manifest. What it describes is a
    fit over a training window the family has no features for - on
    sp500_equity_option_analytics, 482 of 977 dates on null columns. So the assertion is on the
    boundary the floor moved, not on the call having been made.
    """
    _, unclamped, _ = _build(monkeypatch)
    _, clamped, _ = _build(monkeypatch, floor="2010-06-30")

    assert unclamped["computation"]["cv"]["folds"][0]["train_start"].startswith("2003-05-30")
    assert clamped["computation"]["cv"]["folds"][0]["train_start"].startswith("2010-06-30")
    assert clamped["computation"]["cv"]["request"]["train_start_floor"].startswith("2010-06-30")


def test_the_family_is_asked_to_re_key_and_the_result_carries_its_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carrying the validation manifest forward is the failure this rules out.

    It is the one that survives execution: the fit runs, the predictions publish, and the
    eligibility manifest describes three folds that are not the fold that was fitted.
    """
    validation_spec, holdout_spec, seen = _build(monkeypatch)

    assert [entry["hook"] for entry in seen] == ["floor", "rekey"]
    assert holdout_spec["computation"]["expected_prediction_keys"]["n_folds"] == 1
    assert (
        holdout_spec["computation"]["expected_prediction_keys"]["digest"]
        != validation_spec["computation"]["expected_prediction_keys"]["digest"]
    )


def test_the_hook_is_handed_the_holdout_cv_and_not_the_validation_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The family re-keys against the fold it is given, so the order of the two steps is load-bearing."""
    _, _, seen = _build(monkeypatch)

    handed = next(entry for entry in seen if entry["hook"] == "rekey")["spec"]
    assert handed["computation"]["cv"]["split"] == "holdout"
    assert len(handed["computation"]["cv"]["folds"]) == 1


def test_the_validation_specification_is_not_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller still needs it: ``evaluate_holdout`` compares the two, and a notebook prints both."""
    validation_spec, holdout_spec, _ = _build(monkeypatch)

    assert validation_spec["computation"]["cv"]["identity"] == "validation-cv-identity"
    assert len(validation_spec["computation"]["cv"]["folds"]) == 3
    assert validation_spec["computation"]["expected_prediction_keys"]["n_folds"] == 3
    assert holdout_spec is not validation_spec


def test_a_family_that_cannot_re_key_refuses_rather_than_returning_a_validation_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hook that returns without recomputing leaves fields that look present and are wrong."""
    validation_spec = _validation_spec()
    validation_spec["computation"]["expected_prediction_keys"] = {
        "digest": "describes-the-validation-folds",
        "n_rows": 33,
        "n_folds": 3,
    }
    _install_family(monkeypatch, floor=None, rekey=lambda spec: None)

    with pytest.raises(ValueError, match="was not re-keyed to one fold"):
        build_holdout_training_spec(
            SimpleNamespace(case_study=CASE_STUDY),
            validation_spec,
            timeline=MONTH_ENDS,
            case_study=CASE_STUDY,
        )
