"""The sequence and TabM holdout hooks must CALL the validation key rule, not restate it.

`validate_locked_expected_keys` compares a spec's eligibility manifest against the rule that
wrote it. So a hook that computes the manifest by its own copy of the rule produces a holdout
that registers, validates, and is wrong exactly where nothing looks - the check and the value
agree because they came from the same mistake.

The equality tests below are therefore the weaker half. The half that matters replaces the
shared key function with one returning a recognisably different frame and requires the hook's
answer to change with it: a hook that had restated the rule would ignore the replacement and
keep returning the right answer, which is the failure this file exists to catch.

`sp500_options` is why these hooks exist. Correcting its label buffer to calendar time moved
the selected configuration from `linear/lasso_f0.7` to `deep_learning/patchtst`, and until
then no case study had ever asked a sequence family for a holdout.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

HOLDOUT_CV = {
    "folds": [
        {
            "fold": "2",
            "train_start": "2005-01-31T00:00:00",
            "train_end": "2015-12-31T00:00:00",
            "val_start": "2016-01-29T00:00:00",
            "val_end": "2016-12-30T00:00:00",
        }
    ],
    "identity": "holdout_identity",
    "split": "holdout",
    "calendar": None,
    "request": {"source": "case_study_holdout"},
}


def _dataset() -> pl.DataFrame:
    timestamps = pl.datetime_range(
        pl.datetime(2005, 1, 31), pl.datetime(2016, 12, 30), interval="1mo", eager=True
    )
    return pl.DataFrame(
        {
            "timestamp": [t for t in timestamps for _ in ("A", "B")],
            "symbol": ["A", "B"] * len(timestamps),
            "feature": [0.1] * (2 * len(timestamps)),
            "fwd_ret_1m": [0.01, -0.01] * len(timestamps),
        }
    )


@pytest.fixture
def study(monkeypatch: pytest.MonkeyPatch) -> Any:
    mds = SimpleNamespace(
        dataset=_dataset(),
        date_col="timestamp",
        entity_cols=["symbol"],
        label_col="fwd_ret_1m",
        feature_names=["feature"],
        task_type="regression",
        class_values=[],
        eval_label_col=None,
        temporal_by_fold=None,
        # Empty rather than None: the Darts key builder is handed `list(mds.temporal_keys)`
        # unconditionally, so None is a shape production never presents it with.
        temporal_keys=[],
        temporal_feature_names=[],
        input_lineage={"artifacts": []},
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *a, **k: mds)
    monkeypatch.setattr(
        "case_studies.utils.cv_window.canonical_window",
        lambda *a, **k: (date(2016, 1, 29), date(2016, 12, 30)),
    )
    return SimpleNamespace(
        case_study="fixture_case_study",
        labels=SimpleNamespace(get=lambda name, **_: SimpleNamespace(name=name, digest="d")),
    )


def _sequence_spec(library: str = "pytorch", architecture: str = "patchtst") -> dict[str, Any]:
    return {
        "family": "deep_learning",
        "label": "fwd_ret_1m",
        "seed": 42,
        "config_name": "patchtst",
        "computation": {
            "cv": HOLDOUT_CV,
            "expected_prediction_keys": {
                "digest": "the_validation_digest",
                "n_rows": 48,
                "n_folds": 2,
            },
            "model": {
                "class": architecture,
                "implementation": library,
                "objective": "regression",
                "params": {
                    "architecture": architecture,
                    "lookback": 3,
                    "batch_size": 32,
                    "checkpoint_interval": 5,
                    "n_epochs": 10,
                },
            },
            "preprocessing": {"class": "fold_train_standardization", "calendar_id": "NYSE"},
        },
    }


def _tabm_spec() -> dict[str, Any]:
    return {
        "family": "tabular_dl",
        "label": "fwd_ret_1m",
        "computation": {
            "cv": HOLDOUT_CV,
            "expected_prediction_keys": {
                "digest": "the_validation_digest",
                "n_rows": 48,
                "n_folds": 2,
            },
            "model": {
                "class": "TabMModel",
                "implementation": "pytorch",
                "objective": "regression",
                "params": {},
            },
        },
    }


def _marker_frame() -> pl.DataFrame:
    """A frame no real derivation would return, so its digest identifies its source."""
    return pl.DataFrame(
        {"symbol": ["Z"], "timestamp": [datetime(2016, 6, 30)], "fold": [2]},
        schema={"symbol": pl.String, "timestamp": pl.Datetime, "fold": pl.Int64},
    )


def test_the_sequence_hook_writes_a_one_fold_manifest(study: Any) -> None:
    from case_studies.utils import deep_learning

    spec = _sequence_spec()
    deep_learning.rekey_holdout_spec(study, spec, validation_spec={"computation": {"cv": {}}})

    manifest = spec["computation"]["expected_prediction_keys"]
    assert manifest["n_folds"] == 1
    assert manifest["digest"] != "the_validation_digest"
    assert manifest["n_rows"] > 0


def test_the_sequence_hook_takes_its_keys_from_sequence_validation_keys(
    study: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace the rule and the hook's answer has to move with it."""
    from case_studies.utils import deep_learning

    marker = _marker_frame()
    monkeypatch.setattr(deep_learning, "sequence_validation_keys", lambda *a, **k: marker)

    spec = _sequence_spec()
    deep_learning.rekey_holdout_spec(study, spec, validation_spec={"computation": {"cv": {}}})

    from case_studies.utils.artifact_digest import value_digest

    assert spec["computation"]["expected_prediction_keys"] == {
        "digest": value_digest(marker, ("symbol", "timestamp", "fold")),
        "n_rows": 1,
        "n_folds": 1,
    }


def test_the_darts_branch_takes_its_keys_from_darts_validation_keys(
    study: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The branch no shipped configuration exercises is held to the same rule."""
    from case_studies.utils import darts_forecasting, deep_learning

    marker = _marker_frame()
    monkeypatch.setattr(darts_forecasting, "darts_validation_keys", lambda *a, **k: marker)

    spec = _sequence_spec(library="darts", architecture="tsmixer")
    deep_learning.rekey_holdout_spec(study, spec, validation_spec={"computation": {"cv": {}}})

    from case_studies.utils.artifact_digest import value_digest

    assert spec["computation"]["expected_prediction_keys"]["digest"] == value_digest(
        marker, ("symbol", "timestamp", "fold")
    )


def test_the_tabm_hook_takes_its_keys_from_tabm_expected_keys(
    study: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The family no case study has selected for a holdout is held to the same rule."""
    from case_studies.utils import tabular_dl

    marker = _marker_frame()
    monkeypatch.setattr(tabular_dl, "_tabm_expected_keys", lambda *a, **k: marker)

    spec = _tabm_spec()
    tabular_dl.rekey_holdout_spec(study, spec, validation_spec={"computation": {"cv": {}}})

    from case_studies.utils.artifact_digest import value_digest

    assert spec["computation"]["expected_prediction_keys"] == {
        "digest": value_digest(marker, ("symbol", "timestamp", "fold")),
        "n_rows": 1,
        "n_folds": 1,
    }
