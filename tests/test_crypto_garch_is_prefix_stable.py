"""The crypto GJR-GARCH emission path must not read settlements after the value it emits.

`04_model_based_features` fits GJR-GARCH on a refit schedule so the parameters behind any
settlement's volatility come from settlements strictly before it. That closes one of the two
channels a fitted feature has into the future. The other is the conditioning set, and it stayed
open until the filter stopped calling `arch_model(...).fix(params)`.

**crypto is the case where that mattered least and was hardest to see.** The notebook fits
`mean="Zero"`, and `ARCHModel.fix` seeds its recursion from `self.resids(self.starting_values())`
- residuals at the ESTIMATED mean - which under a zero mean is just the returns. So the seed
does not move when the sample is extended, and a truncation test over a healthy fit finds
nothing. What still moves is `variance_bounds`: its per-row EWMA is clamped by two whole-sample
quantities, `np.var(resids) / 1e8` below and `1e7 * (1 + max(resids ** 2))` above. Those sit six
orders of magnitude apart, so they reach an emitted value only when the variance reaches a
clamp - which is what a degenerate fit does, and `arch` returns degenerate fits without
complaint.

Every test here is written against the emission path directly rather than against a walk.
`walk_forward_feature` calls `apply(model, X[:emit_end])` and keeps `values[fit_end:emit_end]`,
and `emit_end` comes from `refit_boundaries`, which depends only on `burnin` and `refit_every`.
So a completed block gets a byte-identical array in a truncated run and a full one, and a
walk-level truncation cannot probe this channel at all. That was measured, not assumed: an
earlier draft of this file truncated the walk and reported 420 shared rows with 0 differences
against the very path this conversion removes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from case_studies.utils.temporal import garch11_conditional_volatility

arch_model = pytest.importorskip("arch").arch_model

ARCH_KW = dict(mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="StudentsT")
CUT = 1500


def _settlements(n: int = 2000, seed: int = 5) -> np.ndarray:
    """Quiet settlements, then a liquidation cascade 500 bars past the cut.

    The cascade is the point: it is what moves `np.var(resids)` and `max(resids ** 2)`, the two
    whole-sample quantities inside `variance_bounds`, and therefore the clamp sitting under
    every row that came before it.
    """
    rng = np.random.default_rng(seed)
    quiet = rng.normal(0.0, 0.5, CUT)
    tail = rng.normal(0.0, 0.5, n - CUT)
    tail[250] = 400.0
    return np.concatenate([quiet, tail])


HEALTHY = pd.Series({"omega": 0.02, "alpha[1]": 0.05, "gamma[1]": 0.08, "beta[1]": 0.88, "nu": 5.0})
# alpha + gamma < 0: a down settlement REDUCES the variance, so the recursion is driven onto
# the lower clamp. `arch` returns this shape without raising, and the old path emitted it.
DEGENERATE = pd.Series(
    {"omega": 1e-8, "alpha[1]": 0.02, "gamma[1]": -0.30, "beta[1]": 0.90, "nu": 5.0}
)


def _arch_path(sample: np.ndarray, params: pd.Series) -> np.ndarray:
    """The emission path as it was: arch's own conditional volatility, one step advanced."""
    omega, alpha, gamma, beta = (
        float(params[name]) for name in ("omega", "alpha[1]", "gamma[1]", "beta[1]")
    )
    variance = np.asarray(arch_model(sample, **ARCH_KW).fix(params).conditional_volatility) ** 2
    forecast = omega + (alpha + gamma * (sample < 0)) * sample**2 + beta * variance
    return np.sqrt(forecast)


def _helper_path(sample: np.ndarray, params: pd.Series, seed_from: np.ndarray) -> np.ndarray:
    """The emission path as the notebook now computes it, seeded from a training window."""
    omega, alpha, gamma, beta = (
        float(params[name]) for name in ("omega", "alpha[1]", "gamma[1]", "beta[1]")
    )
    volatility = arch_model(seed_from, **ARCH_KW).fix(params).model.volatility
    bounds = volatility.variance_bounds(seed_from)
    variance = (
        garch11_conditional_volatility(
            sample,
            mu=0.0,
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            backcast=float(volatility.backcast(seed_from)),
            bounds=(float(np.min(bounds[:, 0])), float(np.max(bounds[:, 1]))),
        )
        ** 2
    )
    forecast = omega + (alpha + gamma * (sample < 0)) * sample**2 + beta * variance
    return np.sqrt(forecast)


def test_the_emitted_volatility_does_not_move_when_later_settlements_arrive() -> None:
    """The property, on the fit that breaks the old path."""
    settlements = _settlements()
    training = settlements[:900]  # strictly before the cut, as the schedule guarantees
    short = _helper_path(settlements[:CUT], DEGENERATE, training)
    full = _helper_path(settlements, DEGENERATE, training)[:CUT]
    # assert_array_equal treats nan as equal to nan, so state that there is something to
    # compare before comparing it. The one-step advance can still go negative on this fit -
    # that is the notebook's arithmetic, not the recursion's - and those rows are nan in both.
    assert np.isfinite(short).sum() > 500
    np.testing.assert_array_equal(short, full)


def test_it_does_not_move_on_a_healthy_fit_either() -> None:
    settlements = _settlements()
    training = settlements[:900]
    short = _helper_path(settlements[:CUT], HEALTHY, training)
    full = _helper_path(settlements, HEALTHY, training)[:CUT]
    np.testing.assert_array_equal(short, full)


def test_the_previous_path_moved_which_is_why_this_one_exists() -> None:
    """The negative control. Without it the assertion above is evidence about nothing."""
    settlements = _settlements()
    short = _arch_path(settlements[:CUT], DEGENERATE)
    full = _arch_path(settlements, DEGENERATE)[:CUT]

    assert not np.array_equal(short, full), (
        "arch's fixed-parameter path is prefix-stable on this fit, which contradicts the "
        "measurement the conversion was written for. Re-derive the reason before reverting."
    )
    # Pin the shape, so a crash or an all-nan path cannot satisfy the assertion above. The
    # one-step advance goes negative on rows where the clipped variance is small enough, and
    # sqrt of that is nan in both paths, so the comparison is over the rows both emitted.
    both = np.isfinite(short) & np.isfinite(full)
    assert both.sum() > 500, "too few emitted rows for the comparison to say anything"
    relative = np.abs(short[both] - full[both]) / np.maximum(np.abs(full[both]), 1e-30)
    assert np.isfinite(relative).all()
    assert (relative > 1e-9).sum() > 400
    assert relative.max() > 0.1


def test_the_old_path_was_stable_on_a_healthy_fit_which_is_why_this_was_missed() -> None:
    """Record the reason a spot check would have cleared the old code.

    Under `mean="Zero"` a well-behaved fit never reaches a clamp, so the two paths agree
    bit-for-bit and any test written against a well-behaved series reports no defect. Stating
    it here stops the next reader concluding the conversion was unnecessary from the same
    observation.
    """
    settlements = _settlements()
    np.testing.assert_array_equal(
        _arch_path(settlements[:CUT], HEALTHY), _arch_path(settlements, HEALTHY)[:CUT]
    )


def test_the_helper_refuses_the_degenerate_fit_rather_than_clipping_it() -> None:
    """Clipping is available and is not the default: a nan is not a feature value."""
    settlements = _settlements()
    with pytest.raises(ValueError, match=r"alpha \+ gamma"):
        garch11_conditional_volatility(
            settlements, mu=0.0, omega=1e-8, alpha=0.02, beta=0.90, gamma=-0.30, backcast=0.25
        )
