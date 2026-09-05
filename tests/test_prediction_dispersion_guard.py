"""The prediction dispersion guard, and what tier it applies at.

`_validate_prediction_dispersion` had no test of its own: its only caller was
`register_prediction_set`, so the bound was exercised through notebook runs and
nowhere else. That is how a threshold calibrated on converged production fits
came to gate a fixture that takes two optimizer steps.
"""

from __future__ import annotations

import json
import logging

import polars as pl
import pytest

from case_studies.utils.registry.registration import (
    MAX_PREDICTION_STD_RATIO,
    _sampling_reduced,
    _validate_prediction_dispersion,
)


def _predictions(score_std: float, target_std: float, n: int = 400, folds: int = 2):
    """Two folds whose score and target dispersions are the ones asked for."""
    import numpy as np

    rng = np.random.default_rng(0)
    frames = []
    for fold in range(folds):
        actual = rng.normal(0.0, target_std, n)
        score = rng.normal(0.0, score_std, n)
        frames.append(
            pl.DataFrame(
                {
                    "fold": [fold] * n,
                    "y_true": actual,
                    "y_score": score,
                }
            )
        )
    return pl.concat(frames)


class TestDispersionBound:
    def test_a_proportionate_score_scale_passes(self):
        _validate_prediction_dispersion(_predictions(score_std=0.02, target_std=0.01))

    def test_a_diverged_fold_is_refused_at_canonical_tier(self):
        with pytest.raises(ValueError, match="diverged fold"):
            _validate_prediction_dispersion(_predictions(score_std=5.0, target_std=0.002))

    def test_refusing_is_the_default(self):
        """No caller may reach the permissive branch by omitting the argument."""
        with pytest.raises(ValueError, match="diverged fold"):
            _validate_prediction_dispersion(
                _predictions(score_std=5.0, target_std=0.002), refuse=True
            )
        with pytest.raises(ValueError, match="diverged fold"):
            _validate_prediction_dispersion(_predictions(score_std=5.0, target_std=0.002))


class TestReducedRun:
    def test_a_reduced_run_reports_the_same_ratio_instead_of_refusing(self, caplog):
        frame = _predictions(score_std=5.0, target_std=0.002)
        with caplog.at_level(logging.WARNING):
            _validate_prediction_dispersion(frame, refuse=False)
        assert "reported rather than refused" in caplog.text
        assert f"{MAX_PREDICTION_STD_RATIO:g}" in caplog.text

    def test_a_reduced_run_within_the_bound_warns_about_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            _validate_prediction_dispersion(
                _predictions(score_std=0.02, target_std=0.01), refuse=False
            )
        assert "dispersion" not in caplog.text

    def test_a_non_finite_score_is_refused_on_a_reduced_run_too(self):
        """Scale depends on how far a fit got. A NaN score does not."""
        frame = _predictions(score_std=0.02, target_std=0.01)
        broken = frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(float("nan"))
            .otherwise(pl.col("y_score"))
            .alias("y_score")
        )
        with pytest.raises(ValueError, match="non-finite"):
            _validate_prediction_dispersion(broken, refuse=False)


class TestSamplingReduced:
    """The five family no-op shapes, read off the specs each family writes."""

    @pytest.mark.parametrize(
        "sampling",
        [
            {"max_symbols": 0, "max_train_sequences": 0},  # deep_learning
            {"max_symbols": 0},  # latent_factors, tabular_dl
            {"train_sample_frac": 1.0, "max_symbols": 0},  # gbm, linear
        ],
    )
    def test_an_unreduced_run_is_not_sampled(self, sampling):
        assert not _sampling_reduced(json.dumps({"computation": {"sampling": sampling}}))

    @pytest.mark.parametrize(
        "sampling",
        [
            {"max_symbols": 5, "max_train_sequences": 2000},
            {"max_symbols": 0, "max_train_sequences": 2000},
            {"train_sample_frac": 0.02, "max_symbols": 5},
            {"train_sample_frac": 1.0, "max_symbols": 5},
        ],
    )
    def test_any_reduced_axis_is_sampled(self, sampling):
        assert _sampling_reduced(json.dumps({"computation": {"sampling": sampling}}))

    def test_a_spec_without_sampling_is_not_sampled(self):
        assert not _sampling_reduced(json.dumps({"computation": {}}))
        assert not _sampling_reduced(json.dumps({}))
        assert not _sampling_reduced(None)
        assert not _sampling_reduced("not json")
