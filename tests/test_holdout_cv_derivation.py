"""What the one holdout retraining interval is derived from, and what it refuses.

The holdout is evaluated once, so the interval it is evaluated over cannot be a parameter a
notebook chooses. Every boundary here comes from something already declared: the window from
the case study's own ``evaluation`` block, the training start from the fold set the selection
was made over, and the gap between them from the label's declared buffer.

These tests are written against real case studies rather than a fixture wherever the value
under test is a declared one. A fixture that declares its own holdout window would agree with
a derivation that read the fixture, which is the shape of test that passes on wrong code.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from case_studies.research.holdout import build_holdout_cv
from case_studies.research.lifecycle import _locked_training_spec
from utils.artifact_specs import load_setup_config

# Newest first, which is the order generate_cv_splits returns and the order the trap depends on:
# fold 0 carries the LATEST train_start in the set, so a derivation that reads folds[0] hands the
# holdout retrain the shortest history rather than the longest.
FOLDS = [
    {
        "fold": 0,
        "train_start": "2005-03-31",
        "train_end": "2014-12-31",
        "val_start": "2015-02-27",
        "val_end": "2015-12-31",
    },
    {
        "fold": 1,
        "train_start": "2004-04-30",
        "train_end": "2014-02-28",
        "val_start": "2014-03-31",
        "val_end": "2015-01-30",
    },
    {
        "fold": 2,
        "train_start": "2003-05-30",
        "train_end": "2013-03-28",
        "val_start": "2013-04-30",
        "val_end": "2014-02-28",
    },
]


def _validation_spec(label: str = "fwd_ret_1m", folds: list[dict] | None = None) -> dict[str, Any]:
    return {
        "label": label,
        "family": "linear",
        "config_name": "ridge",
        "execution_tier": "canonical",
        "identity_version": 3,
        "resolved_spec_schema": "ml4t.resolved-spec/v1",
        "seed": 7,
        "computation": {
            "model": {"kind": "ridge"},
            "cv": {
                "identity": "validation-cv-identity",
                "request": {"source": "case_study_default"},
                "folds": deepcopy(folds if folds is not None else FOLDS),
            },
        },
    }


# Real observation grids: a five-session week for the calendar-backed case studies, month ends
# for us_firm_characteristics. The buffer is counted along these, so the grid is not incidental to
# the test - it is the thing that separates a correct seal from a calendar-shortened one.
SESSIONS = pd.bdate_range("2004-01-01", "2026-12-31")
MONTH_ENDS = pd.date_range("2004-01-31", "2026-12-31", freq="ME")


def _fold(cv: dict[str, Any]) -> dict[str, Any]:
    assert len(cv["folds"]) == 1, "a holdout is evaluated once, over one interval"
    return cv["folds"][0]


def test_the_evaluation_interval_is_the_case_studys_own_declared_holdout_window() -> None:
    setup = load_setup_config("us_firm_characteristics")
    declared = setup["evaluation"]

    fold = _fold(
        build_holdout_cv(
            _validation_spec(), case_study="us_firm_characteristics", timeline=MONTH_ENDS
        )
    )

    assert pd.Timestamp(fold["val_start"]) == pd.Timestamp(declared["holdout_start"])
    assert pd.Timestamp(fold["val_end"]) == pd.Timestamp(declared["holdout_end"])


def test_training_starts_at_the_earliest_fold_not_the_first_one_in_the_list() -> None:
    fold = _fold(
        build_holdout_cv(
            _validation_spec(), case_study="us_firm_characteristics", timeline=MONTH_ENDS
        )
    )

    earliest = min(pd.Timestamp(entry["train_start"]) for entry in FOLDS)
    assert pd.Timestamp(fold["train_start"]) == earliest
    # The assertion that carries the test: fold 0's own start is later, and a derivation that
    # read a list position instead of the boundaries would have produced it.
    assert pd.Timestamp(FOLDS[0]["train_start"]) > earliest
    assert pd.Timestamp(fold["train_start"]) != pd.Timestamp(FOLDS[0]["train_start"])


def test_the_buffer_is_counted_in_observations_not_calendar_time() -> None:
    """The defect this function was rewritten to remove, pinned as a property.

    etfs declares a `21D` buffer and trades a five-session week. Subtracting `pd.Timedelta("21
    days")` leaves about fifteen sessions - short, and short in the direction that looks fine,
    which is why `utils/cv_splits.py` already converts D-buffers to trading days and the causal
    resolver already counts observations. This is the third construction of the same seal and it
    has to agree with the other two.
    """
    cv = build_holdout_cv(_validation_spec("fwd_ret_21d"), case_study="etfs", timeline=SESSIONS)
    fold = _fold(cv)

    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    excluded = [d for d in SESSIONS if train_end < d < val_start]

    assert cv["request"]["label_buffer_steps"] == 21
    assert len(excluded) == 21, "the seal must span 21 observations, not 21 calendar days"
    assert train_end < val_start

    # What the calendar subtraction would have produced, stated so the two cannot be confused.
    calendar_end = val_start - pd.Timedelta(load_setup_config("etfs")["labels"]["buffer"])
    assert train_end < calendar_end, (
        "a calendar-day subtraction lands later than the observation count, which is exactly the "
        "under-buffering that puts the last training label inside the holdout"
    )


def test_a_month_buffer_on_a_monthly_panel_is_one_observation() -> None:
    """The mirror failure: per-unit defaults read `1M` as 21 observations.

    `embargo_from_buffer` refuses to divide a month by an observation step, and its other branch
    takes periods_per_year. Falling through to the per-unit default there makes the seal 21 months
    long on us_firm_characteristics, whose panel is monthly and whose own causal notebook resolves
    the same buffer to 1.
    """
    cv = build_holdout_cv(
        _validation_spec(), case_study="us_firm_characteristics", timeline=MONTH_ENDS
    )
    fold = _fold(cv)

    assert cv["request"]["label_buffer_steps"] == 1
    assert cv["request"]["periods_per_year"] == 12
    excluded = [
        d
        for d in MONTH_ENDS
        if pd.Timestamp(fold["train_end"]) < d < pd.Timestamp(fold["val_start"])
    ]
    assert len(excluded) == 1


def test_a_buffer_shorter_than_the_outcome_horizon_is_refused_as_a_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.research import holdout as module

    monkeypatch.setattr(
        "utils.artifact_specs.resolve_label_buffer", lambda *a, **k: "5D", raising=True
    )
    monkeypatch.setattr(
        "utils.artifact_specs.resolve_label_horizon", lambda *a, **k: "21D", raising=True
    )
    assert module.build_holdout_cv is build_holdout_cv

    with pytest.raises(ValueError, match="shorter than the outcome horizon"):
        build_holdout_cv(_validation_spec("fwd_ret_21d"), case_study="etfs", timeline=SESSIONS)


def test_a_case_study_with_no_declared_holdout_window_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "case_studies.utils.cv_window.canonical_window", lambda *a, **k: None, raising=True
    )

    with pytest.raises(ValueError, match="declares no holdout window"):
        build_holdout_cv(
            _validation_spec(), case_study="us_firm_characteristics", timeline=MONTH_ENDS
        )


def test_an_empty_training_interval_is_refused_rather_than_produced() -> None:
    # Every fold starts after the buffered boundary, so there is no history to retrain on.
    late = [
        {**fold, "train_start": "2015-12-30", "train_end": "2015-12-31"} for fold in deepcopy(FOLDS)
    ]

    with pytest.raises(ValueError, match="holdout training interval is empty"):
        build_holdout_cv(
            _validation_spec(folds=late),
            case_study="us_firm_characteristics",
            timeline=MONTH_ENDS,
        )


def test_a_spec_with_no_resolved_folds_is_refused() -> None:
    spec = _validation_spec()
    spec["computation"]["cv"]["folds"] = []

    with pytest.raises(ValueError, match="no resolved validation folds"):
        build_holdout_cv(spec, case_study="us_firm_characteristics", timeline=MONTH_ENDS)


def test_folds_missing_a_boundary_are_named_rather_than_read_as_none() -> None:
    spec = _validation_spec()
    del spec["computation"]["cv"]["folds"][1]["train_start"]

    with pytest.raises(ValueError, match=r"missing boundaries: \['train_start'\]"):
        build_holdout_cv(spec, case_study="us_firm_characteristics", timeline=MONTH_ENDS)


def test_the_derived_interval_satisfies_the_lock_contract_it_will_be_checked_against() -> None:
    """The derivation is only correct if ``lifecycle.lock`` accepts what it produces.

    ``_locked_training_spec`` is the real gate: it requires the holdout spec to differ from the
    selected validation spec in the CV interval and in nothing else, and to carry an explicit,
    distinct interval. Asserting against it rather than against a restatement of the derivation
    is what makes this a test of the contract instead of a mirror of the code.
    """
    validation = _validation_spec()
    holdout = deepcopy(validation)
    holdout["computation"]["cv"] = build_holdout_cv(
        validation, case_study="us_firm_characteristics", timeline=MONTH_ENDS
    )

    locked = _locked_training_spec(validation, holdout)

    assert locked["computation"]["cv"]["split"] == "holdout"
    assert locked["computation"]["cv"]["identity"] != validation["computation"]["cv"]["identity"]


def test_the_lock_refuses_a_holdout_interval_identical_to_the_validation_one() -> None:
    """The failure direction of the test above: an unchanged interval must not pass."""
    validation = _validation_spec()
    unchanged = deepcopy(validation)

    with pytest.raises(ValueError, match="explicit, distinct CV interval"):
        _locked_training_spec(validation, unchanged)
