"""The linear runner's declared version, and the float noise it is meant to absorb.

``LINEAR_RUNNER_VERSION`` sits in every linear training identity in place of a SHA-256 of
``linear.py``. The digest was unworkable - it made a comment invalidate every registered linear
result - but a declared version is only worth what checks it, which is this file.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from sklearn.linear_model import Lasso, Ridge

from case_studies.utils.folds import clear_memo, prepare_raw_folds, standardized_fold
from case_studies.utils.linear import (
    DERIVED_PARAM_SIGNIFICANT_DIGITS,
    LINEAR_RUNNER_VERSION,
    _quantize_derived,
    _source_identity,
)
from utils.modeling import resolve_linear_params

from .test_folds import SPLITS, _dataset

PINNED_VERSION = 1
PINNED_RIDGE_COEFFICIENTS = "16f01d84576aadd5"
PINNED_LASSO_ALPHA = 0.000197181801558


@pytest.fixture(autouse=True)
def _clean_memo():
    clear_memo()
    yield
    clear_memo()


@pytest.fixture
def fold():
    return standardized_fold(prepare_raw_folds(_dataset(), SPLITS, use_cache=False)[0])


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()[:16]


class TestQuantizingADerivedParameter:
    """The defect this exists for: two paths computed the same alpha and disagreed in the last
    two digits, which moved the training identity while changing nothing about the fit."""

    def test_the_two_values_that_forked_an_identity_now_agree(self) -> None:
        assert _quantize_derived(0.004286799774493045) == _quantize_derived(0.004286799774493049)

    def test_a_difference_the_fit_would_notice_survives(self) -> None:
        assert _quantize_derived(0.004286) != _quantize_derived(0.004287)

    @pytest.mark.parametrize("magnitude", [1e-9, 1e-3, 1.0, 1e3, 1e9])
    def test_the_resolution_is_relative_not_absolute(self, magnitude: float) -> None:
        """A fraction of a fold's degeneracy threshold spans orders of magnitude across case
        studies, so rounding to a fixed number of decimals would erase small alphas entirely."""
        value = 1.234567890123456 * magnitude
        quantized = _quantize_derived(value)

        assert quantized != 0.0
        assert abs(quantized - value) / value < 10.0 ** -(DERIVED_PARAM_SIGNIFICANT_DIGITS - 1)

    def test_values_that_are_not_derived_floats_pass_through(self) -> None:
        assert _quantize_derived(0.0) == 0.0
        assert _quantize_derived(42) == 42
        assert _quantize_derived("auto") == "auto"
        assert _quantize_derived(None) is None

    def test_a_non_finite_value_is_not_rounded(self) -> None:
        assert np.isnan(_quantize_derived(float("nan")))
        assert np.isinf(_quantize_derived(float("inf")))


class TestWhatEntersTheIdentity:
    def test_the_identity_declares_versions_rather_than_source_digests(self) -> None:
        """A 64-character hex string here means the source-hashing scheme came back."""
        identity = _source_identity()

        assert identity["linear_runner"] == LINEAR_RUNNER_VERSION
        assert not any(isinstance(value, str) and len(value) == 64 for value in identity.values())

    def test_it_covers_both_the_runner_and_the_shared_preparation(self) -> None:
        """Preparation lives outside this module but decides the arrays every fit sees."""
        assert {"linear_runner", "fold_preparation", "preprocessing"} <= set(_source_identity())


class TestTheDeclaredVersion:
    """If either pin below moves, the linear runner produces different results than every row
    registered under the current version claims. Bump ``LINEAR_RUNNER_VERSION`` and update the
    pins in the same commit."""

    def test_the_declared_version_matches_what_this_file_pins(self) -> None:
        assert LINEAR_RUNNER_VERSION == PINNED_VERSION

    def test_a_fixed_alpha_fit_reproduces_its_pinned_coefficients(self, fold) -> None:
        model = Ridge(alpha=1.0, random_state=42).fit(fold["X_train"], fold["y_train"])

        assert _digest(model.coef_) == PINNED_RIDGE_COEFFICIENTS, (
            "the linear runner now fits different coefficients; bump LINEAR_RUNNER_VERSION in "
            "case_studies/utils/linear.py and update this pin in the same commit"
        )

    def test_a_derived_alpha_reproduces_its_pinned_value(self, fold) -> None:
        config = {"model_class": "Lasso", "params": {"alpha_frac": 0.5}}
        resolved = resolve_linear_params(config, fold["X_train"], fold["y_train"])

        assert _quantize_derived(resolved["alpha"]) == PINNED_LASSO_ALPHA

    def test_the_quantized_alpha_is_what_the_estimator_is_given(self, fold) -> None:
        """The recorded identity has to describe the model that was fitted, not a neighbour."""
        config = {"model_class": "Lasso", "params": {"alpha_frac": 0.5}}
        resolved = resolve_linear_params(config, fold["X_train"], fold["y_train"])
        quantized = _quantize_derived(resolved["alpha"])

        model = Lasso(alpha=quantized).fit(fold["X_train"], fold["y_train"])

        assert model.get_params()["alpha"] == quantized
