"""The block-permutation refutation is computed on t-statistics, not on raw effects.

Comparing raw thetas made this test anti-conservative on every run ever recorded
(ml4t/agent-workspace#1120). DML's second stage regresses the residualized outcome on the
residualized treatment, so ``var(T_res)`` is the estimator's whole denominator. Permuting
the treatment also frees it from the controls: the first stage can no longer predict it,
its residual keeps essentially all of its variance, and each placebo divides by a much
larger number than the observed estimate does. Every placebo theta is therefore shrunk
toward zero by arithmetic, the permutation distribution is narrower than the null it
stands for, and the observed effect clears it more often than it should.

These tests fix theta at exactly zero, so any rejection is a false positive that needs no
interpretation, and they assert the structure rather than a p-value: which statistic the
test is computed on, that the two placebo collections stay in step, and the signature of
the defect - a theta placebo distribution far narrower than the estimator's own standard
error, next to a t placebo distribution that is not.
"""

import numpy as np
import pandas as pd
import pytest

from case_studies.utils.causal import (
    empirical_permutation_p,
    run_dml_analysis,
)

N_PLACEBO = 12
THETA_TRUE = 0.0


def _ar1(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


@pytest.fixture(scope="module")
def analysis() -> dict:
    """One DML analysis on a panel whose true treatment effect is exactly zero.

    The controls deliberately explain most of the treatment - that is the condition under
    which the denominator matters, and it is the ordinary case in every case study,
    because a treatment no confounder predicts would not need DML.
    """
    rng = np.random.default_rng(20260910)
    n = 900
    x = np.column_stack([_ar1(n, 0.95, rng) for _ in range(3)])
    t = x @ np.array([0.8, -0.5, 0.3]) + 0.45 * _ar1(n, 0.7, rng)
    y = THETA_TRUE * t + x @ np.array([0.6, 0.4, -0.7]) + 0.9 * _ar1(n, 0.6, rng)

    df = pd.DataFrame(
        {"t": t, "y": y, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2]},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    return run_dml_analysis(
        df,
        treatment_col="t",
        outcome_col="y",
        confounder_cols=["x0", "x1", "x2"],
        n_folds=3,
        embargo=5,
        n_placebo=N_PLACEBO,
        block_size=30,
        seed=7,
    )


def test_the_refutation_declares_which_statistic_it_used(analysis):
    """A reader must never have to infer the scale the p-value was computed on."""
    assert analysis["refutation"]["refutation_statistic"] == "t_stat_hac"


def test_the_p_value_is_computed_on_the_placebo_t_statistics(analysis):
    """Recomputing the p from the stored t draws reproduces it exactly.

    Exactly, not approximately: `empirical_permutation_p` is a count over a stored array,
    so any drift between what is reported and what is stored is a defect, not noise.
    """
    ref = analysis["refutation"]
    observed_t = analysis["dml_result"]["t_stat_hac"]

    assert ref["observed_t_stat"] == pytest.approx(observed_t)
    assert ref["empirical_p"] == empirical_permutation_p(
        np.asarray(ref["placebo_t_stats"], dtype=float), observed_t
    )


def test_the_two_placebo_collections_stay_in_step(analysis):
    """One draw contributes one theta and one t-statistic, or neither.

    A draw whose second stage fails is dropped from both. If the collections could drift
    apart, the histogram a notebook renders would not be the distribution behind the
    verdict printed beside it.
    """
    ref = analysis["refutation"]
    assert len(ref["placebo_t_stats"]) == len(ref["placebo_effects"])
    assert len(ref["placebo_t_stats"]) == ref["n_successful"]
    assert len(ref["placebo_n_obs"]) == ref["n_successful"]
    assert np.isfinite(ref["placebo_t_stats"]).all()


def test_the_placebo_thetas_are_narrower_than_the_estimators_own_standard_error(analysis):
    """The defect's signature, asserted so a revert cannot pass quietly.

    A permutation null on the effect scale should be about as wide as the sampling
    distribution of the effect. It is not: the placebo thetas come in far inside the
    estimator's own HAC standard error, which is exactly the shrinkage the inflated
    denominator produces. The t placebo distribution has no such problem - it is close to
    unit spread, because each draw divides by its own standard error.
    """
    ref = analysis["refutation"]
    se_hac = analysis["dml_result"]["se_hac"]

    theta_spread = float(np.std(ref["placebo_effects"]))
    t_spread = float(np.std(ref["placebo_t_stats"]))

    assert theta_spread < 0.5 * se_hac
    assert 0.4 < t_spread < 2.5


def test_a_true_null_is_not_reported_as_a_refutation_that_passed(analysis):
    """With no effect present, the verdict must not be "Passes".

    This is the whole point of the change. On the raw-effect scale the same draws put the
    observed estimate outside almost the entire placebo distribution and the notebook
    printed a refutation that passed; on the t scale it sits inside it.
    """
    ref = analysis["refutation"]
    assert ref["refutation_class"] != "Passes"
    assert ref["empirical_p"] > 0.05
