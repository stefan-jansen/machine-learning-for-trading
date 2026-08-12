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

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits

from case_studies.utils.temporal import (
    filtered_state_probs,
    fit_hmm_kmeans_init,
    relabel_states,
    sort_states_by_mean,
    sort_states_by_variance,
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
