"""Fitted-model helpers shared by the stage-04 ``04_model_based_features`` notebooks.

A model-based feature is a function of parameters estimated from bars, so the estimation
window is part of the feature's information set. That makes two things shared rather than
per-notebook: how inference is run forward over a fold without reading the future, and how
the fitted states are put in an order that means the same thing from one fold to the next.

Both were copied into every notebook that needed them. The forward recursion in particular
existed in six near-verbatim copies, each calling a private ``hmmlearn`` method, and only
three of the six said so - an upstream rename would have broken all six and been documented
at half of them.

Nothing here is a new behaviour. Each helper reproduces what the notebooks already ran, and
``tests/test_temporal.py`` pins that by running the notebook implementations beside these
and asserting the results are identical.
"""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

__all__ = [
    "filtered_state_probs",
    "fit_hmm_kmeans_init",
    "relabel_states",
    "sort_states_by_mean",
    "sort_states_by_variance",
]

# Guards the log of a zero transition or start probability. Small enough not to move a
# fitted probability, large enough that log() stays finite.
_LOG_FLOOR = 1e-300


def filtered_state_probs(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    r"""Filtered state probabilities :math:`P(z_t \mid x_{1:t})`, by forward recursion.

    ``hmmlearn``'s ``predict_proba`` returns the *smoothed* posterior
    :math:`P(z_t \mid x_{1:T})`, which conditions on the whole sequence including
    observations after :math:`t`. Used as a feature that is look-ahead: the value at
    :math:`t` moves when data arrives later. The forward pass conditions on the past and
    the present only, which is what a feature computed in production can know.

    ``model._compute_log_likelihood`` is a **private** ``hmmlearn`` API (present through
    0.3.x) and is the one version-fragile call in this module. It is here, once, rather
    than at six call sites.

    What it returns is the per-state emission log-density,
    :math:`\log p(x_t \mid z_t = k)`, as an ``(n_samples, n_components)`` array. If a
    future release removes it, there is no public method that returns that:
    ``score_samples`` gives the sequence log-likelihood and the *smoothed* posterior, and
    ``predict_proba`` gives the smoothed posterior alone - neither is the emission term.
    The replacement is to evaluate the fitted Gaussians directly, column ``k`` being
    ``scipy.stats.multivariate_normal(model.means_[k], model.covars_[k]).logpdf(X)``.

    Parameters
    ----------
    model
        A fitted ``GaussianHMM``.
    X
        Observations, shape ``(n_samples, n_features)``, in time order.

    Returns
    -------
    ndarray
        Shape ``(n_samples, n_components)``, each row summing to one.
    """
    framelogprob = model._compute_log_likelihood(X)
    n_samples = X.shape[0]
    n_components = model.n_components

    log_startprob = np.log(model.startprob_ + _LOG_FLOOR)
    log_transmat = np.log(model.transmat_ + _LOG_FLOOR)

    # Accumulated in the log domain: the joint P(z_t, x_{1:t}) underflows to zero in the
    # linear domain within a few hundred observations, and these run to thousands.
    fwdlattice = np.zeros((n_samples, n_components))
    fwdlattice[0] = log_startprob + framelogprob[0]
    for t in range(1, n_samples):
        for j in range(n_components):
            fwdlattice[t, j] = framelogprob[t, j] + np.logaddexp.reduce(
                fwdlattice[t - 1] + log_transmat[:, j]
            )

    log_normalizer = np.logaddexp.reduce(fwdlattice, axis=1, keepdims=True)
    return np.exp(fwdlattice - log_normalizer)


def sort_states_by_variance(model: GaussianHMM) -> np.ndarray:
    """State indices ordered by fitted variance, ascending, so state 0 is the calm one.

    EM returns the states in an arbitrary order, so the same fitted state can come back as
    state 0 in one fold and state 1 in the next - and a feature named for one of them then
    means different things across folds. The ordering rule has to be the quantity the
    feature name claims.

    The dispersion of a multivariate state is summarized by the trace of its covariance,
    which is the sum of the per-feature variances and reduces to the variance itself when
    there is one feature.
    """
    dispersion = np.array([np.trace(model.covars_[k]) for k in range(model.n_components)])
    return np.argsort(dispersion)


def sort_states_by_mean(model: GaussianHMM, dim: int = 0) -> np.ndarray:
    """State indices ordered by fitted mean of observation *dim*, ascending.

    The companion to :func:`sort_states_by_variance`, for a feature emitted as the
    probability of the high-*level* state rather than the high-dispersion one - a carry
    regime, for instance, where the states differ in where the mean sits and not in how
    far the observation travels.
    """
    means = np.array([float(model.means_[k][dim]) for k in range(model.n_components)])
    return np.argsort(means)


def relabel_states(
    states: np.ndarray, probs: np.ndarray, order: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an ordering from one of the ``sort_states_by_*`` helpers.

    ``order[i]`` is the fitted state that becomes state ``i``, so the probability columns
    are taken in that order and the state labels are mapped through its inverse.

    Returns
    -------
    tuple of ndarray
        The relabelled state sequence and the reordered probability columns.
    """
    inverse = np.argsort(order)
    return inverse[states], probs[:, order]


def fit_hmm_kmeans_init(
    X: np.ndarray,
    n_states: int = 2,
    random_state: int = 42,
    n_iter: int = 200,
) -> GaussianHMM:
    """Fit a ``GaussianHMM`` whose emissions start from a k-means partition of *X*.

    EM on a Gaussian HMM converges to a local optimum, and from a random start the one it
    reaches moves with the seed. Seeding the means and covariances from k-means starts it
    somewhere the data chose, which makes the fit far less sensitive to that draw. Only
    the start and transition probabilities are left to ``hmmlearn`` to initialise
    (``init_params="st"``); the emission parameters are set here and then refit.

    The covariance of each cluster is regularised by ``1e-6`` on the diagonal so a cluster
    whose members are nearly collinear still yields a positive-definite matrix. A cluster
    with a single member has no covariance at all - ``np.cov`` divides by zero degrees of
    freedom and returns NaN, which the ridge does not repair and ``fit`` does not survive
    - so it starts from the covariance of the whole sample instead.

    The fit runs inside ``threadpool_limits(1)``, and without it a fixed seed does not give
    a fixed model. Floating-point addition is not associative, so a parallel reduction sums
    in whatever order the threads finish, and both the k-means partition and the E-step
    likelihoods inherit that. Measured on this function: five thread counts gave five
    different log-likelihoods (``-12263.024967575566`` at one thread through
    ``-12263.024967576186`` at the ambient count) and five different transition matrices.
    The difference is at the fifteenth digit, but EM amplifies it - the etfs, fx_pairs and
    crypto_perps_funding stage-04 artifacts each hashed differently run to run because of
    it, which is a defect in the artifact rather than a rounding curiosity, since the digest
    is what says the notebook reproduces. Pinning the pool costs nothing measurable here:
    these fits are seconds on windows of a few thousand bars.

    Parameters
    ----------
    X
        Observations, shape ``(n_samples, n_features)``, in time order.
    n_states
        Number of hidden states.
    random_state
        Seed passed to both ``KMeans`` and the ``GaussianHMM``.
    n_iter
        EM iteration cap.
    """
    with threadpool_limits(1):
        kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
        kmeans.fit(X)

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            init_params="st",
        )
        model.means_ = kmeans.cluster_centers_
        ridge = np.eye(X.shape[1]) * 1e-6
        pooled = np.atleast_2d(np.cov(X.T))
        model.covars_ = np.array(
            [_cluster_covariance(X[kmeans.labels_ == k], pooled) + ridge for k in range(n_states)]
        )
        model.fit(X)
    return model


def _cluster_covariance(cluster: np.ndarray, pooled: np.ndarray) -> np.ndarray:
    """Covariance of one k-means cluster, widened to a matrix and never NaN.

    ``np.cov`` of a single-feature cluster returns a 0-d array rather than a 1x1 matrix,
    so the result is widened; the univariate case is the common one here, since most of
    these HMMs read one series. With fewer than two members there is nothing to estimate
    from and ``np.cov`` returns NaN, so the sample covariance stands in - a starting point
    for EM, which refits it either way.
    """
    if cluster.shape[0] < 2:
        return pooled.copy()
    return np.atleast_2d(np.cov(cluster.T))
