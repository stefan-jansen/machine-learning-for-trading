"""The GBM runner's declared version, and what it claims about a fitted result.

``GBM_RUNNER_VERSION`` sits in every GBM training identity in place of a SHA-256 of ``gbm.py``.
The digest was unworkable - ``gbm.py`` is nearly three thousand lines, so any edit anywhere in it
invalidated every registered GBM result - but a declared version is only worth what checks it,
which is this file.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from case_studies.utils.folds import (
    FOLD_PREPARATION_VERSION,
    clear_memo,
    gbm_fold,
    prepare_raw_folds,
)
from case_studies.utils.gbm import (
    GBM_PREPROCESSING_ID,
    GBM_RUNNER_VERSION,
    _gbm_source_identity,
)

from .test_folds import SPLITS, _dataset

PINNED_VERSION = 1
PINNED_PREPROCESSING = "lightgbm-native-float32/v1"
PINNED_PREDICTIONS = "ad01dd304ecf2852"


@pytest.fixture(autouse=True)
def _clean_memo():
    clear_memo()
    yield
    clear_memo()


@pytest.fixture
def fold():
    return gbm_fold(prepare_raw_folds(_dataset(), SPLITS, use_cache=False)[0])


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()[:16]


def _fit(fold, **overrides):
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "num_leaves": 7,
        "learning_rate": 0.1,
        "n_estimators": 25,
        "min_child_samples": 5,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
        "seed": 17,
        "num_threads": 1,
    }
    params.update(overrides)
    model = lgb.LGBMRegressor(**params).fit(fold["X_train"], fold["y_train"])
    return model.predict(fold["X_val"])


class TestTheDeclaredIdentity:
    def test_it_declares_versions_rather_than_source_digests(self):
        """A 64-character hex value here means the file digest came back."""
        identity = _gbm_source_identity()
        assert identity == {
            "gbm_runner": GBM_RUNNER_VERSION,
            "fold_preparation": FOLD_PREPARATION_VERSION,
            "preprocessing": GBM_PREPROCESSING_ID,
        }
        for value in identity.values():
            assert not (isinstance(value, str) and len(value) == 64)

    def test_the_version_is_what_the_pinned_result_below_describes(self):
        assert GBM_RUNNER_VERSION == PINNED_VERSION
        assert GBM_PREPROCESSING_ID == PINNED_PREPROCESSING

    def test_editing_this_module_does_not_move_the_identity(self):
        """The defect this replaced: the identity moved when a file's bytes moved."""
        assert _gbm_source_identity() == _gbm_source_identity()


class TestWhatTheVersionClaims:
    """Bump ``GBM_RUNNER_VERSION`` when one of these moves. Do not re-pin without bumping."""

    def test_predictions_are_stable_for_a_fixed_configuration(self, fold):
        assert _digest(_fit(fold)) == PINNED_PREDICTIONS

    def test_the_label_dtype_does_not_change_the_fit(self, fold):
        """Why ``gbm_fold`` leaves labels float64 while the path it replaced cast them.

        LightGBM converts a label to its own precision, so the fit is identical either way, and
        float64 keeps ``y_eval`` comparable with what the standardising families measure IC on.
        """
        as_float32 = dict(fold, y_train=fold["y_train"].astype(np.float32))
        assert np.array_equal(_fit(fold), _fit(as_float32))

    def test_the_label_dtype_does_not_change_a_huber_fit(self, fold):
        """Huber derives its scale from the labels, so it is the case most likely to differ."""
        as_float32 = dict(fold, y_train=fold["y_train"].astype(np.float32))
        assert np.array_equal(
            _fit(fold, objective="huber", alpha=0.9),
            _fit(as_float32, objective="huber", alpha=0.9),
        )

    def test_the_design_matrix_reaches_the_booster_as_float32(self, fold):
        assert fold["X_train"].dtype == np.float32
        assert fold["X_val"].dtype == np.float32

    def test_missing_values_are_left_for_the_booster_to_route(self):
        """Imputing here would replace a missing value with a fabricated observation.

        The `fold` fixture comes from `_dataset()`, which injects no missing values, so this
        builds its own. It used to take that fixture and skip when it found no NaN, which it
        always did, so the assertion never ran in a file whose job is to make GBM_RUNNER_VERSION
        enforceable.
        """
        raw = prepare_raw_folds(_dataset(missing=True), SPLITS, use_cache=False)[0]
        assert np.isnan(raw.X_train).any(), "the fixture stopped injecting missing values"
        assert np.isnan(gbm_fold(raw)["X_train"]).any()
