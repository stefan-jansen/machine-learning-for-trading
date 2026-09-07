"""A bankrupt member is not the leader of a cohort, nor the reality check's best.

ml4t/agent-workspace#1079. `_apply_ruin_semantics` registers every ranking metric
null for a ruined run so it cannot sort against a solvent one. `uncertainty.py`
computes its own scores from the return matrix and so never saw that decision:
two code paths deriving a leader from the same run, one taught that ruin is
unrankable and one not.

The disagreement is reachable, not theoretical. The series the engine persists for
a ruined path is -1.0 followed by n-1 zeros, whose per-period Sharpe is
-1/sqrt(n-1) - the single -1.0 inflates the dispersion it is divided by, and every
later zero shrinks the mean without adding any. Annualized at 252 periods that is
-0.50 over 1,000 periods and -0.32 over 2,500, which sits in the middle of a
losing cohort rather than at the bottom of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from case_studies.utils.uncertainty import compute_reality_check


def _ruined(n: int) -> np.ndarray:
    """What `apply_ruin_stop` writes: the wipe-out period, then nothing."""
    arr = np.zeros(n)
    arr[0] = -1.0
    return arr


def _solvent_loser(n: int, per_period: float, seed: int = 0) -> np.ndarray:
    """A book that lost money every period without ever losing its capital."""
    rng = np.random.default_rng(seed)
    return per_period + rng.normal(0.0, 0.01, n)


def test_the_arithmetic_that_makes_this_reachable() -> None:
    """A ruined path scores mid-pack, not last, and it is worse the shorter it is."""
    for n, expected in ((1_000, -0.50), (2_500, -0.32)):
        arr = _ruined(n)
        sharpe = arr.mean() / arr.std(ddof=1) * np.sqrt(252)
        assert sharpe == pytest.approx(expected, abs=0.02)


def test_a_large_loss_prior_gains_absorb_is_not_ruin() -> None:
    """Equity never reaches zero, so the member stays eligible to lead.

    Stated through the public entry point rather than the detector, because what
    matters is that a merely bad year is not mistaken for a wipe-out and demoted.
    """
    n = 600
    survivor = np.zeros(n)
    survivor[0] = 4.0
    survivor[1] = -0.9
    out = compute_reality_check(
        {"survivor": survivor, "steady_loser": _solvent_loser(n, -0.002, seed=4)},
        np.zeros(n),
        n_bootstrap=50,
        seed=0,
    )
    assert out["reality_check_best"] == "survivor"


def test_the_reality_check_names_a_solvent_challenger() -> None:
    """The bankrupt one has the better mean excess return, and must not win."""
    n = 1_000
    bench = np.zeros(n)
    challengers = {
        "bankrupt": _ruined(n),
        "lost_more_slowly": _solvent_loser(n, -0.002, seed=1),
    }
    out = compute_reality_check(challengers, bench, n_bootstrap=50, seed=0)
    assert out["reality_check_best"] == "lost_more_slowly"


def test_an_all_bankrupt_cohort_names_nobody() -> None:
    """There is no solvent leader to report, and naming one is what #920 forbids."""
    n = 500
    out = compute_reality_check(
        {"a": _ruined(n), "b": _ruined(n)}, np.zeros(n), n_bootstrap=50, seed=0
    )
    assert out == {}


def test_an_all_solvent_cohort_is_unchanged() -> None:
    """The common case has to name exactly who it named before."""
    n = 800
    challengers = {
        "worse": _solvent_loser(n, -0.002, seed=2),
        "better": _solvent_loser(n, 0.003, seed=3),
    }
    out = compute_reality_check(challengers, np.zeros(n), n_bootstrap=50, seed=0)
    assert out["reality_check_best"] == "better"
