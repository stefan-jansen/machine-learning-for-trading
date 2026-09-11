"""`se_hac` is a robust standard error or it is NaN, and `covariance_type` says which.

`manual_dml_timeseries` seeded `se_hac` with the HC0 value and overwrote it only if the
robust covariance succeeded, under `except Exception: pass`. Three things followed:

1. A run that fell back reported an HC0 standard error under the name `se_hac`. HC0 on an
   overlapping panel label is the estimator this chain exists to replace - the module's own
   comment says row-wise treatment "understates risk" and HC0 understates it further - so
   the failure direction is toward significance, with a smaller p-value, under a column
   name that says the number is robust.
2. `covariance_type` was set unconditionally from whether groups were supplied, so the one
   field whose job is to name the estimator reported a successful Driscoll-Kraay on the
   fallback path. `hac_maxlags` likewise reported a bandwidth that was never applied.
3. `except Exception` caught every programming error in those lines, not only a numerical
   failure, so a future defect there degraded silently to HC0 instead of raising.

Nothing published is wrong today: the fallback could not be triggered in eight
configurations here, the one case that does raise (`IndexError` on a constant treatment,
because `add_constant` drops the duplicate column) raises earlier at `cov_HC0[1, 1]`, and a
300-call replication against `15_causal_estimation/04_dml_crypto_regime`'s real group
structure produced zero raises. What makes the fix load-bearing is where a fallback would
be least visible. Because `se_hac` was initialized from `se_iid`, a fallback returned a
DK/IID ratio of exactly 1.0 at full float precision, and under random data that ratio
already sits near 1.0 - so a fallback inside one of the 300 placebo permutations landed in
a population clustered on its own signature and nothing printed could separate them. Those
permutations feed the published placebo t-distribution.

There is a second way to publish a wrong standard error that needs no exception at all. At
one decision time the groupsum HAC returns a variance at the rounding floor - measured
`se_hac = 5.16e-17` - without raising, which `math.isfinite` accepts and registers as an
astronomical t-statistic. The `n_valid >= 50` guard does not bound it: eighty entities
sharing one timestamp satisfies it. The precondition is on `n_periods`, not on how small
the standard error looks, because a magnitude floor would be a number nobody can defend.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from case_studies.utils.causal import manual_dml_timeseries

FAILED = "failed"


def _panel(n_periods: int, n_entities: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_periods), n_entities)
    n = groups.size
    treatment = rng.normal(size=n)
    confounders = rng.normal(size=(n, 2))
    outcome = treatment + rng.normal(size=n)
    return outcome, treatment, confounders, groups


def _fit(outcome, treatment, confounders, groups, n_folds: int = 3, embargo: int = 1):
    return manual_dml_timeseries(
        outcome,
        treatment,
        confounders,
        n_folds=n_folds,
        embargo=embargo,
        model_y=DummyRegressor(),
        model_t=DummyRegressor(),
        groups=groups,
    )


def _assert_invariant(result: dict) -> None:
    """`covariance_type == "failed"` exactly when `se_hac` is not a robust standard error.

    Stated once and applied to every case below, because the defect this file is about was
    precisely a `covariance_type` that did not track `se_hac`.
    """
    failed = result["covariance_type"] == FAILED
    assert failed == bool(np.isnan(result["se_hac"])), result
    if failed:
        assert np.isnan(result["t_stat_hac"]), result
        assert np.isnan(result["p_value_hac"]), result
        assert result["hac_maxlags"] == 0, result


def test_a_successful_panel_fit_names_driscoll_kraay_and_applies_a_bandwidth() -> None:
    result = _fit(*_panel(40, 6))

    assert result["covariance_type"] == "driscoll_kraay"
    assert np.isfinite(result["se_hac"]) and result["se_hac"] > 0
    assert result["hac_maxlags"] >= 1
    _assert_invariant(result)


def test_a_successful_fit_without_groups_names_newey_west() -> None:
    outcome, treatment, confounders, _ = _panel(40, 6)
    result = manual_dml_timeseries(
        outcome,
        treatment,
        confounders,
        n_folds=3,
        embargo=1,
        model_y=DummyRegressor(),
        model_t=DummyRegressor(),
    )

    assert result["covariance_type"] == "newey_west"
    assert np.isfinite(result["se_hac"])
    _assert_invariant(result)


def test_one_decision_time_is_refused_rather_than_returning_a_rounding_floor() -> None:
    """The case that needs no exception, and the one the old code got most wrong.

    `n_folds=1` over two decision times trains on the first and tests on the second, so the
    residuals are valid at exactly one decision time while `n_valid = 60` clears the
    50-row guard. Measured on this fixture against `origin/main`, before the precondition:

        covariance_type = "driscoll_kraay"      hac_maxlags = 1
        se_iid          = 0.1237                se_hac      = 3.69e-16
        t_stat_hac      = 2.28e15               p_value_hac = 2.80e-16

    and `register_causal_run`'s finiteness check accepted it. No exception, no fallback,
    no NaN - a robust-looking standard error sixteen orders of magnitude below the HC0 one
    and a p-value to match.

    Nothing registered can reach it: all twenty rows in `causal_runs` across seven case
    studies were fitted at `n_folds=5`, and `n_periods` is at least the number of test
    folds. The guard is for the configuration, not for a row that exists.
    """
    outcome, treatment, confounders, groups = _panel(2, 60)

    with pytest.warns(RuntimeWarning, match="at least two decision times"):
        result = _fit(outcome, treatment, confounders, groups, n_folds=1, embargo=0)

    assert result["n_obs"] >= 50
    assert result["n_periods"] == 1
    assert result["covariance_type"] == FAILED
    _assert_invariant(result)
    # The HC0 number is not discarded - it is reported under the name that is true for it.
    assert np.isfinite(result["se_iid"]) and result["se_iid"] > 0


def test_the_seeded_fallback_would_have_passed_every_downstream_guard() -> None:
    """Why NaN and not a labelled HC0 number.

    A labelled fallback still puts a float in `se_hac`, and `register_causal_run`'s
    finiteness check, `t_stat_hac` and `p_value_hac` all accept it. The check below is what
    that check does, so the test fails if the failure path ever starts returning a number
    again, whatever it is labelled.
    """
    outcome, treatment, confounders, groups = _panel(2, 60)

    with pytest.warns(RuntimeWarning):
        result = _fit(outcome, treatment, confounders, groups, n_folds=1, embargo=0)

    import math

    assert not all(math.isfinite(float(result[name])) for name in ("theta", "se_hac"))


def test_a_numerical_failure_reports_failed_rather_than_the_hc0_value(monkeypatch) -> None:
    outcome, treatment, confounders, groups = _panel(40, 6)
    seen: dict[str, float] = {}

    import statsmodels.regression.linear_model as smlm

    original = smlm.RegressionResults.get_robustcov_results

    def raising(self, *args, **kwargs):
        seen["hc0"] = float(np.sqrt(self.cov_HC0[1, 1]))
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(smlm.RegressionResults, "get_robustcov_results", raising)
    with pytest.warns(RuntimeWarning, match="robust covariance failed"):
        result = _fit(outcome, treatment, confounders, groups)
    monkeypatch.setattr(smlm.RegressionResults, "get_robustcov_results", original)

    assert result["covariance_type"] == FAILED
    _assert_invariant(result)
    # The exact value the old code would have reported as `se_hac`, bit for bit, which is
    # what made a fallback indistinguishable from a robust result at a DK/IID ratio of 1.0.
    assert result["se_iid"] == pytest.approx(seen["hc0"])


def test_an_unusable_variance_that_does_not_raise_is_still_failed(monkeypatch) -> None:
    """A covariance that comes back non-positive without raising takes the same path.

    Otherwise `covariance_type` would say `driscoll_kraay` while `se_hac` was NaN, and the
    invariant this file rests on would not hold.
    """
    outcome, treatment, confounders, groups = _panel(40, 6)

    import statsmodels.regression.linear_model as smlm

    class _Degenerate:
        def cov_params(self):
            return np.array([[1.0, 0.0], [0.0, -1e-12]])

    monkeypatch.setattr(
        smlm.RegressionResults,
        "get_robustcov_results",
        lambda self, *a, **k: _Degenerate(),
    )
    with pytest.warns(RuntimeWarning, match="non-positive variance"):
        result = _fit(outcome, treatment, confounders, groups)

    assert result["covariance_type"] == FAILED
    _assert_invariant(result)


def test_a_programming_error_propagates_instead_of_degrading_to_hc0(monkeypatch) -> None:
    """The half of the fix that will fire first.

    `except Exception` swallowed a `TypeError` or an `AttributeError` in those lines and
    returned an HC0 number under the robust name. A defect in this function must raise.
    """
    outcome, treatment, confounders, groups = _panel(40, 6)

    import statsmodels.regression.linear_model as smlm

    def broken(self, *args, **kwargs):
        raise TypeError("get_robustcov_results() got an unexpected keyword argument")

    monkeypatch.setattr(smlm.RegressionResults, "get_robustcov_results", broken)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _fit(outcome, treatment, confounders, groups)


def test_too_few_rows_to_fit_does_not_name_an_estimator() -> None:
    """The early return had the same lie: all-NaN, and `covariance_type` named a estimator."""
    outcome, treatment, confounders, groups = _panel(6, 4)

    result = _fit(outcome, treatment, confounders, groups)

    assert result["n_obs"] < 50
    assert result["covariance_type"] == FAILED
    _assert_invariant(result)


def test_a_failed_observed_covariance_withholds_the_refutation_verdict(monkeypatch) -> None:
    """The regression the NaN introduces, and the reason the guard is not optional.

    `empirical_permutation_p` counts placebo draws at least as extreme as the observed
    one. Every `>=` comparison against NaN is False, so a NaN observed t-statistic scores
    zero placebos as extreme and the test returns the smallest p-value it can produce,
    `1 / (n + 1)`. Measured with the guard removed and 24 draws: `empirical_p = 0.04`,
    `observed_t_stat = nan`, `refutation_class = "Passes"` - a published verdict that the
    effect survives permutation, computed against an undefined observed statistic. The
    placebo fits themselves succeed, so nothing else in the run looks wrong.

    The draw count matters to what the failure looks like. `classify_refutation` returns
    "Underpowered" whenever `1 / (n + 1) >= 0.05`, so below 20 successful draws the same
    defect is masked by the underpowered answer. The case studies run 100.

    `run_resolved_causal_request` refuses the fit at its finiteness check before this
    reaches a registry. The chapter-15 notebooks call `run_dml_analysis` directly and do
    not, which is who would have read the verdict.
    """
    import pandas as pd

    from case_studies.utils import causal

    real = causal.manual_dml_timeseries
    calls = {"n": 0}

    def observed_fails_placebos_succeed(*args, **kwargs):
        result = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] > 1:
            return result
        # Only the observed fit loses its covariance; every placebo draw keeps its own.
        return {
            **result,
            "se_hac": np.nan,
            "t_stat_hac": np.nan,
            "p_value_hac": np.nan,
            "covariance_type": FAILED,
            "hac_maxlags": 0,
        }

    monkeypatch.setattr(causal, "manual_dml_timeseries", observed_fails_placebos_succeed)

    rng = np.random.default_rng(7)
    n_periods, n_entities = 120, 8
    timestamps = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        np.repeat(np.arange(n_periods), n_entities), unit="D"
    )
    n = timestamps.size
    treatment = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": np.tile([f"e{i}" for i in range(n_entities)], n_periods),
            "treat": treatment,
            "conf": rng.normal(size=n),
            "outcome": treatment + rng.normal(size=n),
        }
    )

    with pytest.warns(RuntimeWarning, match="observed t-statistic is not finite"):
        results = causal.run_dml_analysis(
            frame,
            "treat",
            "outcome",
            ["conf"],
            n_folds=3,
            embargo=1,
            n_placebo=24,
            block_size=5,
            horizon=1,
            time_col="timestamp",
            entity_col="symbol",
            model_y=DummyRegressor(),
            model_t=DummyRegressor(),
        )

    assert calls["n"] > 1, "the placebo draws must have run, or the test proves nothing"
    assert results["dml_result"]["covariance_type"] == FAILED
    # No verdict at all, rather than the minimum p-value the comparison would have
    # produced against NaN.
    assert results["refutation"] == {}
