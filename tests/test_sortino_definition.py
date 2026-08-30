"""One Sortino ratio, so an interval describes the number it is printed beside.

`case_studies/utils/uncertainty.py` held three downside deviations - the shortfall over
all periods, the root mean square of the negative returns alone, and their standard
deviation about their own mean. `backtest_metrics.sortino` is written by the engine using
the first; the bootstrap interval around it was computed with the second. On
`us_firm_characteristics`' validation rank-1 - 99 monthly periods, 20 of them negative -
that put the stored 13.876 outside its own CI of [4.22, 9.65], by exactly the
sqrt(n / n_negative) the two definitions differ by, and the §5 forest plot raised rather
than drawing a bar of negative length.
"""

from __future__ import annotations

import numpy as np

from case_studies.utils.uncertainty import (
    _sample_stats,
    _sortino,
    compute_backtest_uncertainty,
)

# mean 0.006; shortfall (0, -0.01, 0, -0.02, 0) has mean square 0.0001, so the downside
# deviation is exactly 0.01 and the ratio is 0.6 * sqrt(12).
SERIES = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
SORTINO_AT_12 = 0.6 * np.sqrt(12.0)


def test_the_downside_deviation_averages_over_every_period() -> None:
    # Averaging the two negative squares over 2 rather than over 5 gives 1.31, which is
    # what this file did before and what the stored metric never agreed with.
    assert _sample_stats(SERIES, 12).sortino == float(SORTINO_AT_12)


def test_the_cohort_leader_uses_the_same_ratio() -> None:
    assert _sortino(SERIES, 12) == _sample_stats(SERIES, 12).sortino


def test_the_bootstrap_interval_contains_the_point_estimate() -> None:
    """The property that broke, stated as the property rather than as the formula.

    A percentile interval from resamples of a series need not contain the full-sample
    estimate in general, but it cannot sit entirely on one side of it when both are the
    same estimator on a well-behaved series - and that is exactly what a different
    downside deviation on each side produced.
    """
    rng = np.random.default_rng(0)
    returns = rng.normal(0.004, 0.02, size=180)
    point = _sample_stats(returns, 12).sortino
    uncertainty = compute_backtest_uncertainty(returns, periods_per_year=12, n_boot=400, seed=1)

    assert uncertainty["sortino_ci95_lo"] <= point <= uncertainty["sortino_ci95_hi"]
    assert uncertainty["sharpe_ci95_lo"] <= _sample_stats(returns, 12).sharpe
