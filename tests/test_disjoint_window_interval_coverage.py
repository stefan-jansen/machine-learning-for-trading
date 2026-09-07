"""What the validation-to-holdout interval is calibrated for, measured rather than argued.

``compute_independent_diff_uncertainty`` bootstraps each window over its own length and
takes the difference of those draws. The rule that used to justify it - the windows are
disjoint, so their Sharpes are independent - is false, and the correction that was
proposed alongside it (ignoring a positive covariance inflates ``Var(A - B)``, so the
interval is conservative) is false in the other direction.

Resampling inside a window conditions on that window's realized returns, so each side's
spread is the sampling noise given those returns. That makes the interval right for the
gap between the two windows and blind to a market regime that lands differently on each
- and the regime is exactly what decides whether one edge carried across both. The tests
below simulate a regime common to both windows, vary how much of it the two share, and
measure coverage of each target.
"""

from __future__ import annotations

import numpy as np
import pytest

from case_studies.utils.uncertainty import compute_independent_diff_uncertainty

PERIODS_PER_YEAR = 252
DAILY_VOL = 0.01
EDGE = 0.0005
WINDOW = 126
# A regime shift four times the edge, so the effect is far larger than the Monte Carlo
# error of 120 replications and the assertions below need no wide tolerance bands.
REGIME_VOL = 0.002
REPLICATIONS = 120
DRAWS = 200


def _coverage(regime_correlation: float, seed: int = 7) -> tuple[float, float]:
    """Coverage of (one edge across both windows, each window's own Sharpe).

    ``regime_correlation`` is how much of the regime shift the two windows share: +1 is
    a regime identical in both, 0 two independent draws, -1 a reversal across the
    boundary. The strategy's edge is the same in every replication, so the first target
    is a difference of zero; the second is the difference the two realized windows
    actually carry.
    """
    rng = np.random.default_rng(seed)
    one_edge = 0
    per_window = 0
    for replication in range(REPLICATIONS):
        shift_c = rng.normal(0.0, REGIME_VOL)
        shift_b = regime_correlation * shift_c + np.sqrt(
            max(0.0, 1.0 - regime_correlation**2)
        ) * rng.normal(0.0, REGIME_VOL)
        challenger = rng.normal(EDGE + shift_c, DAILY_VOL, WINDOW)
        baseline = rng.normal(EDGE + shift_b, DAILY_VOL, WINDOW)

        result = compute_independent_diff_uncertainty(
            challenger,
            baseline,
            periods_per_year=PERIODS_PER_YEAR,
            block_length=1,
            n_boot=DRAWS,
            seed=replication,
        )
        lo = result["sharpe_diff_ci95_lo"]
        hi = result["sharpe_diff_ci95_hi"]
        one_edge += lo <= 0.0 <= hi
        realized = (shift_c - shift_b) / DAILY_VOL * np.sqrt(PERIODS_PER_YEAR)
        per_window += lo <= realized <= hi
    return one_edge / REPLICATIONS, per_window / REPLICATIONS


@pytest.mark.parametrize("regime_correlation", [1.0, -1.0])
def test_the_interval_covers_the_gap_between_the_two_windows(
    regime_correlation: float,
) -> None:
    """The estimand the interval is right for, whatever the windows share.

    Conditioning on each window's realized returns is what makes this hold: the regime
    is inside both samples rather than in either resampling distribution, so it cannot
    push the coverage of the realized gap around.
    """
    _, per_window = _coverage(regime_correlation)
    assert 0.90 <= per_window <= 0.99


def test_a_regime_the_two_windows_share_leaves_nothing_uncovered() -> None:
    """With an identical shift in both windows there is no unshared part to miss.

    This is the premise of the test below. If coverage were short here too, the shortfall
    would be something other than the regime and the diagnosis would be wrong.
    """
    one_edge, _ = _coverage(1.0)
    assert one_edge >= 0.90


def test_a_regime_that_reverses_across_the_boundary_is_not_covered() -> None:
    """The reading the notebooks take, and the direction the error runs.

    "The strategy has one edge; is this gap noise?" is not what the interval answers. A
    regime that moves the two windows apart widens the true spread of the difference and
    the interval does not widen with it, because a resampler that never looks outside
    either window cannot see it.

    The claim this refutes is that ignoring a positive covariance inflates
    ``Var(A - B) = Var A + Var B - 2 Cov(A, B)`` and so makes the interval conservative.
    Asserting that reading here - ``one_edge >= 0.95`` - fails at 0.400. The conditioning
    that drops the covariance term drops the same amount from both marginals, so the
    error runs toward under-coverage and never toward over-coverage.
    """
    one_edge, _ = _coverage(-1.0)
    assert one_edge <= 0.70


def test_the_interval_does_not_widen_for_what_it_cannot_see() -> None:
    """Coverage is lost through the width staying put, not through the target moving.

    Without this, a shortfall above could be read as the interval being correctly
    narrower in one regime than another rather than as the same interval either way.
    """
    widths = {}
    for regime_correlation in (1.0, -1.0):
        rng = np.random.default_rng(7)
        measured = []
        for replication in range(REPLICATIONS):
            shift_c = rng.normal(0.0, REGIME_VOL)
            shift_b = regime_correlation * shift_c + np.sqrt(
                max(0.0, 1.0 - regime_correlation**2)
            ) * rng.normal(0.0, REGIME_VOL)
            result = compute_independent_diff_uncertainty(
                rng.normal(EDGE + shift_c, DAILY_VOL, WINDOW),
                rng.normal(EDGE + shift_b, DAILY_VOL, WINDOW),
                periods_per_year=PERIODS_PER_YEAR,
                block_length=1,
                n_boot=DRAWS,
                seed=replication,
            )
            measured.append(result["sharpe_diff_ci95_hi"] - result["sharpe_diff_ci95_lo"])
        widths[regime_correlation] = float(np.mean(measured))

    assert widths[1.0] == pytest.approx(widths[-1.0], rel=0.05)
