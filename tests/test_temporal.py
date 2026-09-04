"""Tests for case_studies/utils/temporal.py.

This module is an extraction, not a new behaviour, so the tests that matter run the
implementations the stage-04 notebooks carry today beside the shared ones and assert the
results are identical. `_notebook_*` below are verbatim copies of what is in the
notebooks; if one of these tests fails, the extraction changed a fitted feature.

Also pinned: the property the forward recursion exists for. `predict_proba` is smoothed,
so the value it gives for time t moves when observations after t arrive; the filtered
value must not.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits

from case_studies.utils.temporal import (
    filtered_state_probs,
    fit_hmm_kmeans_init,
    garch11_conditional_volatility,
    refit_boundaries,
    relabel_states,
    sort_states_by_mean,
    sort_states_by_variance,
    walk_forward_feature,
    write_model_based,
)

# ---------------------------------------------------------------------------
# Verbatim copies of what the notebooks run today.
# ---------------------------------------------------------------------------


def _notebook_compute_filtered_probs(model, X):
    """case_studies/etfs and cme_futures 04_model_based_features, unchanged."""
    framelogprob = model._compute_log_likelihood(X)
    n_samples = X.shape[0]
    n_components = model.n_components
    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)
    fwdlattice = np.zeros((n_samples, n_components))
    fwdlattice[0] = log_startprob + framelogprob[0]
    for t in range(1, n_samples):
        for j in range(n_components):
            fwdlattice[t, j] = framelogprob[t, j] + np.logaddexp.reduce(
                fwdlattice[t - 1] + log_transmat[:, j]
            )
    log_normalizer = np.logaddexp.reduce(fwdlattice, axis=1, keepdims=True)
    return np.exp(fwdlattice - log_normalizer)


def _notebook_sort_states_by_variance(model):
    """case_studies/etfs and crypto_perps_funding, unchanged; fx_pairs inlines it."""
    variances = np.array([np.trace(model.covars_[k]) for k in range(model.n_components)])
    return np.argsort(variances)


def _notebook_sort_states_by_carry(model):
    """case_studies/cme_futures, unchanged."""
    means = np.array([float(model.means_[k][0]) for k in range(model.n_components)])
    return np.argsort(means)


def _notebook_relabel_states(states, probs, order):
    """case_studies/etfs, unchanged."""
    inv_order = np.argsort(order)
    return inv_order[states], probs[:, order]


# ---------------------------------------------------------------------------
# Fixtures: a two-regime series, and a model fitted on it.
# ---------------------------------------------------------------------------


def _series(n: int = 400, n_features: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0, 0.4, (n // 2, n_features))
    stressed = rng.normal(0.6, 2.0, (n - n // 2, n_features))
    return np.vstack([calm, stressed, calm])


@pytest.fixture(scope="module")
def fitted() -> tuple[GaussianHMM, np.ndarray]:
    X = _series()
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=50, random_state=0)
    model.fit(X)
    return model, X


# ---------------------------------------------------------------------------
# filtered_state_probs
# ---------------------------------------------------------------------------


def test_filtered_state_probs_matches_the_notebook_recursion(fitted) -> None:
    model, X = fitted
    np.testing.assert_array_equal(
        filtered_state_probs(model, X), _notebook_compute_filtered_probs(model, X)
    )


def test_filtered_state_probs_rows_are_a_distribution(fitted) -> None:
    model, X = fitted
    probs = filtered_state_probs(model, X)
    assert probs.shape == (X.shape[0], model.n_components)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_filtered_state_probs_does_not_move_when_later_observations_arrive(fitted) -> None:
    """The property the recursion exists for: no value depends on its own future."""
    model, X = fitted
    prefix = filtered_state_probs(model, X[:200])
    full = filtered_state_probs(model, X)
    np.testing.assert_allclose(prefix, full[:200])


def test_predict_proba_does_move_when_later_observations_arrive(fitted) -> None:
    """The reason `predict_proba` may not be used as the feature."""
    model, X = fitted
    prefix = model.predict_proba(X[:200])
    full = model.predict_proba(X)
    assert not np.allclose(prefix, full[:200]), "smoothed and filtered would be the same quantity"


# ---------------------------------------------------------------------------
# State ordering
# ---------------------------------------------------------------------------


def test_sort_states_by_variance_matches_the_notebook_rule(fitted) -> None:
    model, _ = fitted
    np.testing.assert_array_equal(
        sort_states_by_variance(model), _notebook_sort_states_by_variance(model)
    )


def test_sort_states_by_variance_puts_the_calm_state_first(fitted) -> None:
    model, _ = fitted
    order = sort_states_by_variance(model)
    dispersion = [np.trace(model.covars_[k]) for k in order]
    assert dispersion == sorted(dispersion)


def test_sort_states_by_mean_matches_the_notebook_rule(fitted) -> None:
    model, _ = fitted
    np.testing.assert_array_equal(sort_states_by_mean(model), _notebook_sort_states_by_carry(model))


def test_relabel_states_matches_the_notebook_helper(fitted) -> None:
    model, X = fitted
    probs = filtered_state_probs(model, X)
    states = probs.argmax(axis=1)
    order = sort_states_by_variance(model)
    shared = relabel_states(states, probs, order)
    notebook = _notebook_relabel_states(states, probs, order)
    np.testing.assert_array_equal(shared[0], notebook[0])
    np.testing.assert_array_equal(shared[1], notebook[1])


def test_relabel_states_is_the_identity_under_an_ordering_that_changes_nothing(fitted) -> None:
    model, X = fitted
    probs = filtered_state_probs(model, X)
    states = probs.argmax(axis=1)
    relabelled, reordered = relabel_states(states, probs, np.arange(model.n_components))
    np.testing.assert_array_equal(relabelled, states)
    np.testing.assert_array_equal(reordered, probs)


# ---------------------------------------------------------------------------
# fit_hmm_kmeans_init
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_features", [1, 3])
def test_kmeans_covariance_widening_reproduces_both_notebook_expressions(n_features) -> None:
    """The one line the two notebook copies differ on.

    etfs writes ``np.cov(cluster.T)`` and cme_futures writes ``np.cov(cluster.T).reshape(1, 1)``,
    because np.cov of a single-feature cluster returns a 0-d array. ``np.atleast_2d`` is
    each of them in its own case, which is what lets one helper serve both.
    """
    cluster = _series(n=60, n_features=n_features, seed=1)
    shared = np.atleast_2d(np.cov(cluster.T))
    if n_features == 1:
        np.testing.assert_array_equal(shared, np.cov(cluster.T).reshape(1, 1))
    else:
        np.testing.assert_array_equal(shared, np.cov(cluster.T))
    assert shared.shape == (n_features, n_features)


@pytest.mark.parametrize("n_features", [1, 3])
def test_fit_hmm_kmeans_init_returns_a_fitted_model_of_the_right_shape(n_features) -> None:
    X = _series(n_features=n_features)
    model = fit_hmm_kmeans_init(X, n_states=2, random_state=42)
    assert model.means_.shape == (2, n_features)
    assert model.covars_.shape == (2, n_features, n_features)
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0)
    assert filtered_state_probs(model, X).shape == (X.shape[0], 2)


@pytest.mark.parametrize("n_features", [1, 3])
def test_fit_hmm_kmeans_init_survives_a_one_observation_cluster(n_features) -> None:
    """np.cov of a single member divides by zero degrees of freedom and returns NaN.

    The 1e-6 ridge does not repair a NaN, so without a fallback the fit is handed a
    covariance of NaN. A far outlier is the way k-means is made to produce such a cluster.
    """
    X = _series(n=200, n_features=n_features)
    X = np.vstack([X, np.full((1, n_features), 1e6)])
    model = fit_hmm_kmeans_init(X, n_states=2, random_state=42)
    assert np.isfinite(model.covars_).all()
    assert np.isfinite(filtered_state_probs(model, X)).all()


def test_cluster_covariance_falls_back_rather_than_returning_nan() -> None:
    from case_studies.utils.temporal import _cluster_covariance

    pooled = np.array([[2.0]])
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert np.isnan(np.cov(np.array([[1.5]]).T)), "the condition the fallback is for"
    np.testing.assert_array_equal(_cluster_covariance(np.array([[1.5]]), pooled), pooled)
    two = np.array([[1.0], [3.0]])
    np.testing.assert_allclose(_cluster_covariance(two, pooled), np.cov(two.T).reshape(1, 1))


def test_fit_hmm_kmeans_init_is_the_same_model_at_every_thread_count() -> None:
    """A fixed seed did not give a fixed model until the fit was pinned to one thread.

    Floating-point addition is not associative, so a parallel reduction sums in whatever
    order the threads finish. Unpinned, this function returned five different
    log-likelihoods and five different transition matrices across five thread counts, and
    EM carried the fifteenth-digit difference into the fitted model. Three stage-04
    artifacts hashed differently run to run because of it (ml4t/agent-workspace#328).

    Varying the *outer* pool is what makes this a regression test: the limiter inside
    ``fit_hmm_kmeans_init`` has to override it, so removing that limiter makes the two
    fits diverge and this assertion fail.
    """
    X = _series(n=2000, n_features=2)
    models = []
    for outer_threads in (1, 8):
        with threadpool_limits(outer_threads):
            models.append(fit_hmm_kmeans_init(X, n_states=2, random_state=42))

    one, eight = models
    assert one.score(X) == eight.score(X), "log-likelihood moved with the ambient thread count"
    np.testing.assert_array_equal(one.transmat_, eight.transmat_)
    np.testing.assert_array_equal(one.means_, eight.means_)
    np.testing.assert_array_equal(one.covars_, eight.covars_)


def test_fit_hmm_kmeans_init_separates_the_two_regimes() -> None:
    X = _series()
    model = fit_hmm_kmeans_init(X, n_states=2, random_state=42)
    calm, stressed = sort_states_by_variance(model)
    assert np.trace(model.covars_[calm]) < np.trace(model.covars_[stressed])


# --- write_model_based -------------------------------------------------------------------
#
# Each guard gets a frame that violates exactly one thing, so a test that passes because the
# helper rejected the frame for an unrelated reason would still fail on the message.


def _emit_frame(n_symbols: int = 3, n_days: int = 6, folds: tuple[int, ...] = (0, 1)):
    rows = []
    for fold in folds:
        for s in range(n_symbols):
            for d in range(n_days):
                rows.append(
                    {
                        "timestamp": datetime(2020, 1, 1) + timedelta(days=d),
                        "symbol": f"S{s}",
                        "fold": fold,
                        "vol_state": float(d),
                        "garch_sigma": float(d) * 0.5,
                    }
                )
    return pl.DataFrame(rows)


FEATURES = ["vol_state", "garch_sigma"]
WRITE_KW = dict(
    keys=["timestamp", "symbol"],
    feature_columns=FEATURES,
    time_column="timestamp",
    written_by="tests/test_temporal.py",
)


def test_write_model_based_writes_the_artifact_and_its_sidecar(tmp_path: Path) -> None:
    out = tmp_path / "model_based.parquet"
    record = write_model_based(_emit_frame(), out, expected_folds=[0, 1], **WRITE_KW)
    assert out.exists()
    assert record["n_rows"] == 36
    assert pl.read_parquet(out).height == 36


def test_write_model_based_records_where_each_feature_starts(tmp_path: Path) -> None:
    frame = _emit_frame().with_columns(
        pl.when(pl.col("timestamp") < datetime(2020, 1, 3))
        .then(None)
        .otherwise(pl.col("garch_sigma"))
        .alias("garch_sigma")
    )
    record = write_model_based(frame, tmp_path / "m.parquet", **WRITE_KW)
    geometry = {(g["fold"], g["feature"]): g for g in record["feature_geometry"]}
    # The warm-up is visible in the sidecar rather than only in the values, which is the
    # whole point: the defect it stands for left no trace anywhere before this.
    assert geometry[(0, "garch_sigma")]["first_valid"].startswith("2020-01-03")
    assert geometry[(0, "vol_state")]["first_valid"].startswith("2020-01-01")
    assert geometry[(0, "garch_sigma")]["n_null"] == 6


def test_write_model_based_rejects_a_duplicated_row_within_a_fold(tmp_path: Path) -> None:
    frame = _emit_frame()
    frame = pl.concat([frame, frame.head(1)])
    with pytest.raises(ValueError, match="duplicate rows"):
        write_model_based(frame, tmp_path / "m.parquet", **WRITE_KW)


def test_write_model_based_allows_the_same_key_in_two_folds(tmp_path: Path) -> None:
    # The identity is key + fold, not key: every fold re-emits the same panel rows.
    record = write_model_based(_emit_frame(folds=(0, 1, 2)), tmp_path / "m.parquet", **WRITE_KW)
    assert record["n_rows"] == 54


def test_write_model_based_rejects_a_feature_that_is_null_across_a_whole_fold(
    tmp_path: Path,
) -> None:
    frame = _emit_frame().with_columns(
        pl.when(pl.col("fold") == 1).then(None).otherwise(pl.col("vol_state")).alias("vol_state")
    )
    with pytest.raises(ValueError, match="no value at all in a fold"):
        write_model_based(frame, tmp_path / "m.parquet", **WRITE_KW)


def test_write_model_based_rejects_a_fold_that_was_not_resolved(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match the resolved folds"):
        write_model_based(
            _emit_frame(folds=(0, 1, 7)), tmp_path / "m.parquet", expected_folds=[0, 1], **WRITE_KW
        )


def test_write_model_based_rejects_a_null_key(tmp_path: Path) -> None:
    frame = _emit_frame().with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("symbol")).alias("symbol")
    )
    with pytest.raises(ValueError, match="null values in key"):
        write_model_based(frame, tmp_path / "m.parquet", **WRITE_KW)


def test_write_model_based_rejects_a_missing_declared_feature(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing declared columns"):
        write_model_based(_emit_frame().drop("garch_sigma"), tmp_path / "m.parquet", **WRITE_KW)


def test_write_model_based_writes_a_fold_free_frame(tmp_path: Path) -> None:
    # What a walk-forward refit schedule emits: one row per key, no fold column, the same
    # value whichever fold later selects the row.
    frame = _emit_frame(folds=(0,)).drop("fold")
    record = write_model_based(frame, tmp_path / "m.parquet", fold_column=None, **WRITE_KW)
    assert record["n_rows"] == 18
    assert "fold" not in pl.read_parquet(tmp_path / "m.parquet").columns
    assert "fold_digests" not in record


def test_a_fold_free_geometry_record_carries_no_fold(tmp_path: Path) -> None:
    frame = _emit_frame(folds=(0,)).drop("fold")
    record = write_model_based(frame, tmp_path / "m.parquet", fold_column=None, **WRITE_KW)
    geometry = {g["feature"]: g for g in record["feature_geometry"]}
    assert set(geometry) == set(FEATURES)
    assert all(g["fold"] is None for g in geometry.values())
    assert geometry["garch_sigma"]["n_rows"] == 18


def test_a_fold_free_frame_still_rejects_a_repeated_key(tmp_path: Path) -> None:
    # Without a fold column the key alone is the identity, so the second copy of a row is a
    # duplicate rather than the next fold's re-emission.
    frame = _emit_frame(folds=(0,)).drop("fold")
    with pytest.raises(ValueError, match="duplicate rows"):
        write_model_based(
            pl.concat([frame, frame.head(1)]),
            tmp_path / "m.parquet",
            fold_column=None,
            **WRITE_KW,
        )


def test_a_fold_free_frame_refuses_expected_folds(tmp_path: Path) -> None:
    frame = _emit_frame(folds=(0,)).drop("fold")
    with pytest.raises(ValueError, match="without a fold column"):
        write_model_based(
            frame, tmp_path / "m.parquet", fold_column=None, expected_folds=[0], **WRITE_KW
        )


def test_write_model_based_writes_nothing_when_a_guard_fires(tmp_path: Path) -> None:
    out = tmp_path / "m.parquet"
    with pytest.raises(ValueError):
        write_model_based(_emit_frame(folds=(0, 9)), out, expected_folds=[0, 1], **WRITE_KW)
    assert not out.exists()


# ---------------------------------------------------------------------------
# The GARCH(1,1) recursion.
#
# The property is the same one the schedule exists for, one level down: a value must not move
# when observations after it are appended. `arch`'s own result object fails it, which is why
# there is a recursion here at all.
# ---------------------------------------------------------------------------

GARCH_PARAMS = dict(mu=0.05, omega=0.02, alpha=0.08, beta=0.90)


def _garch_returns(n: int = 800, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(0.0, 0.6, size=n // 2), rng.normal(0.0, 2.2, size=n - n // 2)]
    )


def test_the_garch_recursion_does_not_move_when_later_returns_arrive() -> None:
    returns = _garch_returns()
    short = garch11_conditional_volatility(returns[:500], **GARCH_PARAMS)
    full = garch11_conditional_volatility(returns, **GARCH_PARAMS)
    np.testing.assert_array_equal(short, full[:500])


def test_the_garch_recursion_reproduces_its_own_definition() -> None:
    returns = _garch_returns(n=60)
    got = garch11_conditional_volatility(returns, **GARCH_PARAMS)
    mu, omega = GARCH_PARAMS["mu"], GARCH_PARAMS["omega"]
    alpha, beta = GARCH_PARAMS["alpha"], GARCH_PARAMS["beta"]
    resid = returns - mu
    want = np.empty(len(returns))
    want[0] = omega / (1.0 - alpha - beta)
    for t in range(1, len(returns)):
        want[t] = omega + alpha * resid[t - 1] ** 2 + beta * want[t - 1]
    np.testing.assert_allclose(got, np.sqrt(want), rtol=1e-12)


def test_the_garch_seed_survives_a_non_stationary_fit() -> None:
    # alpha + beta >= 1 has no long-run variance, so the seed is clamped rather than infinite.
    out = garch11_conditional_volatility(
        _garch_returns(n=50), mu=0.0, omega=0.02, alpha=0.3, beta=0.75
    )
    assert np.isfinite(out).all()


def test_an_empty_return_series_gives_an_empty_recursion() -> None:
    assert garch11_conditional_volatility(np.empty(0), **GARCH_PARAMS).shape == (0,)


# ---------------------------------------------------------------------------
# The walk-forward refit schedule.
#
# The property under test is the one the fold-frozen design could not state: the value at
# row t must not move when observations after t are deleted. That has to hold through BOTH
# channels - the conditioning set and the parameters - and it is the parameter channel that
# `walk_forward_feature` is here to close.
# ---------------------------------------------------------------------------


def _mean_fit(X: np.ndarray) -> float:
    """A model whose parameter is trivially readable, so a test can say where it came from."""
    return float(X[:, 0].mean())


def _emit_parameter(model: float, X: np.ndarray) -> np.ndarray:
    """One row per input row, every row carrying the parameter that speaks for it."""
    return np.full((len(X), 1), model)


def test_refit_boundaries_never_fits_on_the_rows_it_speaks_for() -> None:
    for fit_end, emit_end in refit_boundaries(100, burnin=10, refit_every=7):
        assert fit_end <= emit_end
        assert fit_end >= 10


def test_refit_boundaries_covers_every_row_past_the_burn_in_exactly_once() -> None:
    covered = [i for a, b in refit_boundaries(100, burnin=10, refit_every=7) for i in range(a, b)]
    assert covered == list(range(10, 100))


def test_refit_boundaries_is_empty_when_the_series_cannot_pay_the_burn_in() -> None:
    assert refit_boundaries(10, burnin=10, refit_every=5) == []
    assert refit_boundaries(3, burnin=10, refit_every=5) == []


def test_refit_boundaries_fits_once_when_only_one_row_follows_the_burn_in() -> None:
    assert refit_boundaries(11, burnin=10, refit_every=5) == [(10, 11)]


@pytest.mark.parametrize("refit_every", [1, 5, 21])
def test_the_burn_in_prefix_carries_no_value(refit_every: int) -> None:
    X = np.arange(60, dtype=float).reshape(-1, 1)
    out = walk_forward_feature(
        X, burnin=20, refit_every=refit_every, fit=_mean_fit, apply=_emit_parameter, n_features=1
    )
    assert np.isnan(out[:20]).all()
    assert np.isfinite(out[20:]).all()


def test_a_value_is_computed_from_parameters_fitted_strictly_before_it() -> None:
    """The whole point. Row 20's parameter is the mean of rows 0-19 and of nothing later."""
    X = np.arange(60, dtype=float).reshape(-1, 1)
    out = walk_forward_feature(
        X, burnin=20, refit_every=10, fit=_mean_fit, apply=_emit_parameter, n_features=1
    )
    assert out[20, 0] == pytest.approx(X[:20, 0].mean())
    assert out[29, 0] == pytest.approx(X[:20, 0].mean())
    # The next block refits on everything up to 30, so row 30 moves and row 29 does not.
    assert out[30, 0] == pytest.approx(X[:30, 0].mean())


def test_deleting_later_observations_does_not_move_an_earlier_value() -> None:
    """Run the same walk over a truncated series; every surviving row must be unchanged.

    This is what a fold-frozen fit fails. Its parameters come from the whole training
    window, so shortening the series moves values at rows the truncation did not touch.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 1))
    full = walk_forward_feature(
        X, burnin=15, refit_every=6, fit=_mean_fit, apply=_emit_parameter, n_features=1
    )
    for cut in (40, 55, 79):
        short = walk_forward_feature(
            X[:cut], burnin=15, refit_every=6, fit=_mean_fit, apply=_emit_parameter, n_features=1
        )
        np.testing.assert_allclose(short, full[:cut], equal_nan=True)


def test_a_rolling_window_forgets_and_an_expanding_one_does_not() -> None:
    X = np.arange(60, dtype=float).reshape(-1, 1)
    kwargs = dict(burnin=20, refit_every=10, fit=_mean_fit, apply=_emit_parameter, n_features=1)
    expanding = walk_forward_feature(X, **kwargs)
    rolling = walk_forward_feature(X, window=20, **kwargs)
    assert expanding[20, 0] == pytest.approx(rolling[20, 0])
    assert rolling[40, 0] == pytest.approx(X[20:40, 0].mean())
    assert expanding[40, 0] == pytest.approx(X[:40, 0].mean())


def test_the_rolling_window_is_still_bounded_by_the_start_of_the_series() -> None:
    X = np.arange(30, dtype=float).reshape(-1, 1)
    out = walk_forward_feature(
        X,
        burnin=5,
        refit_every=5,
        window=100,
        fit=_mean_fit,
        apply=_emit_parameter,
        n_features=1,
    )
    assert out[5, 0] == pytest.approx(X[:5, 0].mean())


def test_a_series_shorter_than_the_burn_in_is_all_null_rather_than_an_error() -> None:
    out = walk_forward_feature(
        np.arange(5, dtype=float).reshape(-1, 1),
        burnin=20,
        refit_every=5,
        fit=_mean_fit,
        apply=_emit_parameter,
        n_features=1,
    )
    assert out.shape == (5, 1)
    assert np.isnan(out).all()


def test_a_failing_fit_raises_by_default() -> None:
    def explode(X: np.ndarray) -> float:
        raise RuntimeError("did not converge")

    with pytest.raises(RuntimeError, match="did not converge"):
        walk_forward_feature(
            np.arange(30, dtype=float).reshape(-1, 1),
            burnin=10,
            refit_every=5,
            fit=explode,
            apply=_emit_parameter,
            n_features=1,
        )


def test_a_skipped_block_is_null_and_the_walk_carries_on() -> None:
    calls: list[int] = []

    def sometimes(X: np.ndarray) -> float:
        calls.append(len(X))
        if len(X) == 15:
            raise RuntimeError("did not converge")
        return _mean_fit(X)

    out = walk_forward_feature(
        np.arange(30, dtype=float).reshape(-1, 1),
        burnin=10,
        refit_every=5,
        fit=sometimes,
        apply=_emit_parameter,
        n_features=1,
        on_fit_error="skip",
    )
    assert np.isfinite(out[10:15]).all()
    assert np.isnan(out[15:20]).all()
    assert np.isfinite(out[20:]).all()
    assert calls == [10, 15, 20, 25]


def test_apply_returning_the_wrong_number_of_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="one row per input row"):
        walk_forward_feature(
            np.arange(30, dtype=float).reshape(-1, 1),
            burnin=10,
            refit_every=5,
            fit=_mean_fit,
            apply=lambda model, X: np.full((len(X) - 1, 1), model),
            n_features=1,
        )


def test_a_multi_column_feature_keeps_its_columns_in_order() -> None:
    out = walk_forward_feature(
        np.arange(40, dtype=float).reshape(-1, 1),
        burnin=10,
        refit_every=5,
        fit=_mean_fit,
        apply=lambda model, X: np.column_stack([np.full(len(X), model), np.arange(len(X))]),
        n_features=2,
    )
    assert out[12, 1] == 12
    assert out[12, 0] == pytest.approx(np.arange(10, dtype=float).mean())


def test_a_fitted_hidden_markov_model_walks_forward_without_reading_its_future() -> None:
    """The real shape: fit_hmm_kmeans_init plus filtered_state_probs under the schedule."""
    rng = np.random.default_rng(7)
    X = np.concatenate([rng.normal(0.0, 0.5, size=(150, 1)), rng.normal(0.0, 2.5, size=(150, 1))])
    kwargs = dict(
        burnin=100,
        refit_every=25,
        fit=lambda train: fit_hmm_kmeans_init(train, n_states=2, random_state=0),
        apply=lambda model, prefix: filtered_state_probs(model, prefix)[
            :, sort_states_by_variance(model)
        ],
        n_features=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = walk_forward_feature(X, **kwargs)
        short = walk_forward_feature(X[:220], **kwargs)
    assert np.isnan(full[:100]).all()
    np.testing.assert_allclose(short, full[:220], rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(full[100:].sum(axis=1), 1.0, rtol=1e-9)


def test_no_parameters_are_estimated_past_the_freeze_point() -> None:
    """The holdout rule: the last estimate before it opens speaks for all of it."""
    fitted_on: list[int] = []

    def record(X: np.ndarray) -> float:
        fitted_on.append(len(X))
        return _mean_fit(X)

    X = np.arange(60, dtype=float).reshape(-1, 1)
    out = walk_forward_feature(
        X,
        burnin=20,
        refit_every=10,
        freeze_after=40,
        fit=record,
        apply=_emit_parameter,
        n_features=1,
    )
    assert fitted_on == [20, 30, 40]
    # Rows 40 onwards all carry the estimate made from the first 40 observations.
    assert out[40, 0] == pytest.approx(X[:40, 0].mean())
    assert out[59, 0] == pytest.approx(X[:40, 0].mean())
    assert np.isfinite(out[20:]).all()


def test_a_freeze_point_inside_the_burn_in_emits_nothing() -> None:
    out = walk_forward_feature(
        np.arange(60, dtype=float).reshape(-1, 1),
        burnin=20,
        refit_every=10,
        freeze_after=5,
        fit=_mean_fit,
        apply=_emit_parameter,
        n_features=1,
    )
    assert np.isnan(out).all()


def test_freezing_does_not_change_the_values_before_the_freeze_point() -> None:
    X = np.arange(60, dtype=float).reshape(-1, 1)
    kwargs = dict(burnin=20, refit_every=10, fit=_mean_fit, apply=_emit_parameter, n_features=1)
    np.testing.assert_allclose(
        walk_forward_feature(X, freeze_after=40, **kwargs)[:41],
        walk_forward_feature(X, **kwargs)[:41],
        equal_nan=True,
    )
