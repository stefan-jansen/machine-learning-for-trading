"""The split-conformal half-width is a calibration score at a rank, or unbounded.

Split conformal prediction's finite-sample guarantee rests on taking the
ceil((n + 1) * coverage)-th smallest calibration score. These tests pin the two
properties that make it that number rather than a nearby one: the value is a
score the calibration set actually contains, at exactly that rank, and there is
no value at all when the rank exceeds the set.

Both were wrong in the notebook this helper replaced, in ways np.quantile makes
easy to reach: its default interpolates between ranks, and method="higher" maps
the level onto n - 1 intervals rather than n ranks and clamps an unattainable
rank to the largest score.
"""

import math

import numpy as np
import pytest

from utils.modeling import conformal_quantile


@pytest.mark.parametrize("n", [10, 37, 100, 999, 1000])
@pytest.mark.parametrize("coverage", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_returns_the_score_at_the_ceiling_rank(n, coverage):
    scores = np.sort(np.random.default_rng(n).random(n))
    rank = math.ceil((n + 1) * coverage)
    expected = float("inf") if rank > n else float(scores[rank - 1])
    assert conformal_quantile(scores, coverage) == expected


def test_the_value_is_one_of_the_calibration_scores():
    scores = np.random.default_rng(0).random(250)
    q = conformal_quantile(scores, 0.9)
    assert q in set(scores.tolist())


def test_input_order_does_not_matter():
    scores = np.random.default_rng(1).random(400)
    assert conformal_quantile(scores, 0.9) == conformal_quantile(np.sort(scores)[::-1], 0.9)


def test_unattainable_coverage_is_unbounded_rather_than_the_largest_score():
    # 5 scores cannot certify 90%: ceil(6 * 0.9) = 6 > 5. Returning max(scores)
    # would assert a guarantee the calibration set does not support.
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert conformal_quantile(scores, 0.9) == float("inf")
    assert conformal_quantile(scores, 0.5) == 0.3


def test_wider_coverage_never_narrows_the_interval():
    scores = np.random.default_rng(2).random(500)
    widths = [conformal_quantile(scores, c) for c in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)]
    assert widths == sorted(widths)


def test_non_finite_scores_are_dropped_rather_than_poisoning_the_rank():
    clean = np.random.default_rng(3).random(200)
    dirty = np.concatenate([clean, [np.nan, np.inf]])
    assert conformal_quantile(dirty, 0.9) == conformal_quantile(clean, 0.9)


def test_no_calibration_score_is_unbounded():
    assert conformal_quantile(np.array([]), 0.9) == float("inf")


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, 1.5])
def test_coverage_outside_the_open_unit_interval_is_refused(coverage):
    with pytest.raises(ValueError, match="coverage"):
        conformal_quantile(np.array([0.1, 0.2, 0.3]), coverage)
