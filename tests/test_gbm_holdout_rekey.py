"""Re-keying a GBM training spec from the validation folds to the derived holdout fold.

Without this the GBM family cannot produce a holdout prediction at all: the spec that
selection ranked carries an eligibility manifest and a parameter set describing the
validation folds, and `reconstruct_locked_request` checks both against the holdout fold
before it will build a request.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from case_studies.utils import gbm

VALIDATION_CV = {
    "folds": [
        {
            "fold": "0",
            "train_start": "2005-01-31T00:00:00",
            "train_end": "2014-12-31T00:00:00",
            "val_start": "2015-01-30T00:00:00",
            "val_end": "2015-12-31T00:00:00",
        },
        {
            "fold": "1",
            "train_start": "2005-01-31T00:00:00",
            "train_end": "2013-12-31T00:00:00",
            "val_start": "2014-01-31T00:00:00",
            "val_end": "2014-12-31T00:00:00",
        },
    ],
    "identity": "validation_identity",
    "request": {"source": "case_study_default"},
}

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
    "request": {"source": "case_study_holdout"},
}

BASE_PARAMS = {
    "objective": "regression",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "verbosity": -1,
    "metric": "None",
}


def _dataset() -> pl.DataFrame:
    timestamps = pl.datetime_range(
        pl.datetime(2005, 1, 31),
        pl.datetime(2016, 12, 30),
        interval="1mo",
        eager=True,
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
    dataset = _dataset()
    mds = SimpleNamespace(
        dataset=dataset,
        date_col="timestamp",
        entity_cols=["symbol"],
        label_col="fwd_ret_1m",
        feature_names=["feature"],
        task_type="regression",
        class_values=[],
        eval_label_col=None,
        temporal_by_fold=None,
        temporal_keys=None,
        temporal_feature_names=None,
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *a, **k: mds)
    # The canonical window is what binds `locked_holdout_split`; the fixture case study has
    # no setup.yaml, so it is declared here rather than left to fail before re-keying runs.
    monkeypatch.setattr(
        "case_studies.utils.cv_window.canonical_window",
        lambda *a, **k: (date(2016, 1, 29), date(2016, 12, 30)),
    )
    return SimpleNamespace(
        case_study="fixture_case_study",
        labels=SimpleNamespace(get=lambda name, **_: SimpleNamespace(name=name)),
    )


def _spec(params_by_fold: dict[str, dict[str, Any]], **model_extra: Any) -> dict[str, Any]:
    return {
        "label": "fwd_ret_1m",
        "computation": {
            "cv": HOLDOUT_CV,
            "expected_prediction_keys": {
                "digest": "the_validation_digest",
                "n_rows": 48,
                "n_folds": 2,
            },
            "model": {
                "class": "lightgbm.Booster",
                "implementation": "lightgbm",
                "effective_params_by_fold": params_by_fold,
                "huber_alpha_scale": None,
                "max_iterations": 500,
                **model_extra,
            },
        },
    }


def _validation_spec() -> dict[str, Any]:
    return {"computation": {"cv": VALIDATION_CV}}


def test_manifest_and_parameters_are_rekeyed_to_the_holdout_fold(study: Any) -> None:
    spec = _spec({"0": dict(BASE_PARAMS), "1": dict(BASE_PARAMS)})
    gbm.rekey_holdout_spec(study, spec, validation_spec=_validation_spec())

    computation = spec["computation"]
    manifest = computation["expected_prediction_keys"]
    assert manifest["n_folds"] == 1
    assert manifest["digest"] != "the_validation_digest"
    # The holdout window closes 2016-12-30, so the December month-end falls outside it:
    # eleven of the twelve 2016 dates, two symbols.
    assert manifest["n_rows"] == 22
    assert computation["model"]["effective_params_by_fold"] == {"2": BASE_PARAMS}


def test_parameters_that_vary_for_no_known_reason_are_refused(study: Any) -> None:
    """Carrying one fold's values forward would evaluate a configuration nobody ranked."""
    spec = _spec({"0": dict(BASE_PARAMS), "1": {**BASE_PARAMS, "num_leaves": 31}})
    with pytest.raises(ValueError, match=r"differ across the recorded folds in \['num_leaves'\]"):
        gbm.rekey_holdout_spec(study, spec, validation_spec=_validation_spec())


def test_a_varying_huber_delta_needs_the_scale_that_derives_it(study: Any) -> None:
    """`alpha` is the one parameter resolved per fold, and only from a declared scale.

    Without the scale there is no rule that produces the holdout fold's delta, so the spec
    cannot be re-keyed at all - and silently reusing a validation fold's delta would size the
    Huber transition against the wrong training labels.
    """
    varying = {
        "0": {**BASE_PARAMS, "objective": "huber", "alpha": 0.031},
        "1": {**BASE_PARAMS, "objective": "huber", "alpha": 0.029},
    }
    with pytest.raises(ValueError, match="no huber_alpha_scale"):
        gbm.rekey_holdout_spec(study, _spec(varying), validation_spec=_validation_spec())


def test_parameters_keyed_to_folds_the_validation_cv_never_declared_are_refused(
    study: Any,
) -> None:
    """A spec already re-keyed once must not be re-keyed again from its own output."""
    spec = _spec({"7": dict(BASE_PARAMS)})
    with pytest.raises(ValueError, match=r"records parameters for folds \['7'\]"):
        gbm.rekey_holdout_spec(study, spec, validation_spec=_validation_spec())


def test_the_huber_delta_is_rederived_from_the_holdout_folds_own_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`alpha` is a scaled standard deviation of the training labels, so it moves per fold.

    It is read through `training_labels_for_split`, the same selection the request planner
    derives its own thresholds from. Reading the labels through fold preparation instead
    would cast them to LightGBM's float32, and the delta is quantized to twelve significant
    digits - far past where float32 stops carrying information - so the two paths resolve
    different alphas from identical labels and give one declared configuration two
    training identities. The labels below are the case: their float32 standard deviation
    quantizes to something else entirely.
    """
    import numpy as np

    from case_studies.utils.derived_params import quantize_derived

    timestamps = pl.datetime_range(
        pl.datetime(2005, 1, 31), pl.datetime(2016, 12, 30), interval="1mo", eager=True
    )
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.03, size=2 * len(timestamps))
    dataset = pl.DataFrame(
        {
            "timestamp": [t for t in timestamps for _ in ("A", "B")],
            "symbol": ["A", "B"] * len(timestamps),
            "feature": [0.1] * (2 * len(timestamps)),
            "fwd_ret_1m": values,
        }
    )
    mds = SimpleNamespace(
        dataset=dataset,
        date_col="timestamp",
        entity_cols=["symbol"],
        label_col="fwd_ret_1m",
        feature_names=["feature"],
        task_type="regression",
        class_values=[],
        eval_label_col=None,
        temporal_by_fold=None,
        temporal_keys=None,
        temporal_feature_names=None,
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *a, **k: mds)
    monkeypatch.setattr(
        "case_studies.utils.cv_window.canonical_window",
        lambda *a, **k: (date(2016, 1, 29), date(2016, 12, 30)),
    )
    study = SimpleNamespace(
        case_study="fixture_case_study",
        labels=SimpleNamespace(get=lambda name, **_: SimpleNamespace(name=name)),
    )

    huber = {**BASE_PARAMS, "objective": "huber"}
    spec = _spec(
        {"0": {**huber, "alpha": 0.031}, "1": {**huber, "alpha": 0.029}},
        huber_alpha_scale=1.0,
    )
    gbm.rekey_holdout_spec(study, spec, validation_spec=_validation_spec())

    train = dataset.filter(pl.col("timestamp") <= pl.datetime(2015, 12, 31))
    expected = quantize_derived(float(np.nanstd(train["fwd_ret_1m"].to_numpy())))
    resolved = spec["computation"]["model"]["effective_params_by_fold"]["2"]["alpha"]
    assert resolved == expected
    float32_delta = quantize_derived(
        float(np.nanstd(train["fwd_ret_1m"].to_numpy().astype(np.float32)))
    )
    assert resolved != float32_delta, "the two paths must be distinguishable here"


# --- the holdout asks coverage, not fold-boundary compatibility ------------------------
#
# `#676` gave the GBM family a re-key hook, which is what the cases above cover. It did not
# change what the locked reconstruction asks of a fold-scoped stage-04 artifact, and that
# question was still compatibility: does the artifact declare a fold with this geometry. For a
# holdout it never can - the fold is derived after stage 04 ran, and the artifact cannot be
# rebuilt to declare it without changing the sha256 the selection was made under. So every
# gbm-carried case study with fold-scoped model-based features refused its own holdout refit
# with "custom CV is incompatible with fold-scoped temporal features". Measured on etfs, whose
# carrier is `gbm/default_mae`.
#
# `latent_factors/adapter.py` already took the coverage branch here. This asserts the GBM path
# takes it too, by which function the reconstruction calls rather than by re-testing what the
# functions themselves do - `test_holdout_cv_derivation.py` covers that.


class _Reached(Exception):
    """Raised by the stubbed check, so the test ends at the call it is about."""


def _fold_scoped_mds(dataset: pl.DataFrame) -> Any:
    return SimpleNamespace(
        dataset=dataset,
        date_col="timestamp",
        entity_cols=["symbol"],
        label_col="fwd_ret_1m",
        feature_names=["feature"],
        task_type="regression",
        class_values=[],
        eval_label_col=None,
        temporal_by_fold=dataset.select("timestamp", "symbol"),
        temporal_keys=["symbol"],
        temporal_feature_names=["temporal_feature"],
        # Present so that reverting the fix fails on the assertion below rather than on a
        # missing attribute. The artifact declares the validation folds and not the derived
        # holdout one, which is the whole shape of the defect.
        temporal_artifact_splits=VALIDATION_CV["folds"],
    )


def test_the_locked_gbm_holdout_asks_coverage_not_compatibility(
    study: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "case_studies.research.cv.require_fold_scoped_temporal_compatibility",
        lambda *a, **k: calls.append("compatibility"),
    )
    monkeypatch.setattr(
        "case_studies.utils.gbm.require_fold_scoped_temporal_compatibility",
        lambda *a, **k: calls.append("compatibility"),
    )

    def _coverage(*_a: Any, **_k: Any) -> None:
        calls.append("coverage")
        raise _Reached

    monkeypatch.setattr(
        "case_studies.utils.gbm.require_fold_scoped_temporal_holdout_coverage", _coverage
    )
    mds = _fold_scoped_mds(_dataset())
    mds.input_lineage = {"artifacts": {"features": "feat_digest"}}
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *a, **k: mds)
    label_ref = SimpleNamespace(
        name="fwd_ret_1m",
        digest="label_digest",
        definition=SimpleNamespace(continuous_eval_label=None),
    )

    # The prelude is reproduced rather than stubbed: every field below is one the
    # reconstruction refuses on, so building them from the same helpers is what gets the call
    # as far as the branch this test is about.
    interval = 500
    schedule = [
        {"kind": "iteration", "value": value}
        for value in gbm.gbm_checkpoint_iterations(
            {"max_iterations": 500, "checkpoint_interval": interval}
        )
    ]
    spec = _spec({"2": dict(BASE_PARAMS)})
    spec["seed"] = gbm.RANDOM_SEED
    computation = spec["computation"]
    computation["sampling"] = {"train_sample_frac": 1.0, "max_symbols": 0}
    computation["checkpoint_schedule"] = schedule
    computation["label_artifact"] = {"digest": "label_digest", "name": "fwd_ret_1m"}
    computation["feature_artifacts"] = mds.input_lineage["artifacts"]
    computation["feature_names"] = ["feature"]
    computation["input_data_spec"] = mds.input_lineage
    computation["source_identity"] = gbm._gbm_source_identity()
    computation["runtime_identity"] = gbm._gbm_runtime_identity()
    computation["task"] = {"type": "regression", "class_values": [], "continuous_eval_label": None}

    writable_study = SimpleNamespace(
        case_study="fixture_case_study",
        require_writable=lambda: None,
        activate=lambda _tier: None,
        labels=SimpleNamespace(get=lambda name, **_: label_ref),
    )

    with pytest.raises(_Reached):
        gbm.reconstruct_locked_request(
            writable_study, spec, checkpoint_kind="iteration", checkpoint_value=500
        )

    assert calls == ["coverage"]
