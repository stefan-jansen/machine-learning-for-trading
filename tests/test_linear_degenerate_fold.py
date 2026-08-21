"""What a linear fold is allowed to predict, and what stops the run.

A declared penalty grid exists to be swept past the point where it works. At the strong end a
penalty zeroes every coefficient and the model predicts its intercept on every row, which is a
result the grid is meant to produce and the scoring layer is built to record: a fold with no
cross-sectional variation yields no IC, and ``n_folds_ic`` beside ``n_folds`` is what makes that
visible (``case_studies/utils/registry/metrics.py``).

Aborting the fit on that fold instead cost 28 configurations of the binary label
``fwd_dir_8h`` when one of them went degenerate on fold 0. Non-finite predictions are a different
thing - a numerical failure with nothing to record - and still stop the run.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from case_studies.utils.linear import _fold_predictions


class _Constant:
    """A regression model whose coefficients have all been shrunk away."""

    def __init__(self, value: float = 0.25) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float64)


class _NonFinite:
    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.linspace(0.0, 1.0, len(X))
        out[0] = np.nan
        return out


class _ConstantProba:
    """A classifier that assigns every row the same class probabilities."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.6, 0.4]), (len(X), 1))


@pytest.fixture
def fold() -> dict[str, np.ndarray | int]:
    return {"fold": 0, "X_val": np.arange(24, dtype=np.float64).reshape(8, 3)}


def _context(task_type: str = "regression", class_values: tuple[int, ...] = (0, 1)):
    return SimpleNamespace(task_type=task_type, class_values=class_values)


def test_constant_regression_predictions_are_returned(fold):
    """The degenerate end of a penalty sweep is a result, not a failure."""
    predictions = _fold_predictions(_Constant(), fold, _context())

    assert predictions.shape == (8,)
    assert np.all(predictions == 0.25)
    assert float(np.std(predictions)) == 0.0


def test_constant_classification_predictions_are_returned(fold):
    """The same holds through the probability remap, which is where fwd_dir_8h failed."""
    predictions = _fold_predictions(_ConstantProba(), fold, _context("classification"))

    assert predictions.shape == (8,)
    assert float(np.std(predictions)) == 0.0
    # P(class=1) * 1 + P(class=0) * 0
    assert np.allclose(predictions, 0.4)


def test_non_finite_predictions_still_raise(fold):
    """A NaN has nothing to record downstream, so it stops the run."""
    with pytest.raises(ValueError, match="non-finite"):
        _fold_predictions(_NonFinite(), fold, _context())


def test_varying_predictions_are_unchanged(fold):
    """The ordinary path is untouched: identical inputs, identical outputs."""

    class _Linear:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X[:, 0] * 2.0

    predictions = _fold_predictions(_Linear(), fold, _context())

    assert np.allclose(predictions, fold["X_val"][:, 0] * 2.0)
