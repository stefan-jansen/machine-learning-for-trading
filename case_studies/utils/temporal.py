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

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from case_studies.utils.artifact_digest import write_artifact

__all__ = [
    "filtered_state_probs",
    "fit_hmm_kmeans_init",
    "fold_feature_geometry",
    "refit_boundaries",
    "relabel_states",
    "sort_states_by_mean",
    "sort_states_by_variance",
    "walk_forward_feature",
    "write_model_based",
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


def fold_feature_geometry(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    time_column: str,
    fold_column: str = "fold",
) -> list[dict]:
    """Per fold and feature, where the values actually start and stop.

    Returns one record per (fold, feature) with the first and last timestamp carrying a
    non-null value and the null count. This is descriptive, not a check: a fitted feature
    legitimately begins after its estimation window, so a leading gap is only a defect
    relative to the other features and labels on the same fold, which this frame cannot
    see on its own.

    It exists because that comparison was impossible after the fact. A model-based feature
    that started late left no trace in the artifact, the registry or any metric:
    ``sequence_dataset`` turns a null feature into ``0.0``, which after normalization is the
    feature's mean, so the affected rows were fitted as average observations and nothing
    raised. Recording the geometry at write time is what lets a later stage compare a
    variant's start against the primary's instead of discovering it by hand.
    """
    records: list[dict] = []
    for (fold_id,), part in frame.group_by([fold_column], maintain_order=True):
        for col in feature_columns:
            present = part.filter(pl.col(col).is_not_null())
            records.append(
                {
                    "fold": fold_id,
                    "feature": col,
                    "n_rows": part.height,
                    "n_null": part.height - present.height,
                    "first_valid": None if present.is_empty() else present[time_column].min(),
                    "last_valid": None if present.is_empty() else present[time_column].max(),
                }
            )
    return records


def write_model_based(
    frame: pl.DataFrame,
    path: Path | str,
    *,
    keys: Sequence[str],
    feature_columns: Sequence[str],
    time_column: str,
    written_by: str,
    fold_column: str = "fold",
    expected_folds: Sequence[int] | None = None,
    inputs: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict:
    """Write the stage-04 artifact with the guards that were spread across eight notebooks.

    Replaces the ad-hoc write block each ``04_model_based_features`` notebook carried. Those
    blocks agreed on calling :func:`~case_studies.utils.artifact_digest.write_artifact` and
    on nothing else: the duplicate-key assertion was in six of eight, the fold-id check in
    none, and the schema was frozen in none, so a notebook could emit a column of the wrong
    dtype or a fold that did not exist and the artifact would still be written and digested.

    Guards, in order, each raising before anything reaches disk:

    * every key and the fold column is present, and no key value is null
    * ``keys + [fold_column]`` is unique, so a fold cannot carry a row twice
    * every declared feature column is present and not entirely null within any fold
    * the fold ids are exactly ``expected_folds`` when given

    The per-fold feature geometry from :func:`fold_feature_geometry` goes into the sidecar
    metadata under ``fold_feature_geometry``. It is recorded rather than asserted on for the
    reason given there: this frame cannot tell a legitimate estimation warm-up from an
    excess one, and a guard that refused every leading gap would reject the case studies
    where the gap is correct.
    """
    frame_keys = list(keys)
    missing = [c for c in [*frame_keys, fold_column, *feature_columns] if c not in frame.columns]
    if missing:
        raise ValueError(f"model_based frame is missing declared columns: {missing}")

    null_keys = [c for c in [*frame_keys, fold_column] if frame[c].null_count()]
    if null_keys:
        raise ValueError(f"null values in key or fold columns: {null_keys}")

    identity = [*frame_keys, fold_column]
    n_dup = int(frame.select(identity).is_duplicated().sum())
    if n_dup:
        raise ValueError(f"{n_dup:,} duplicate rows on {identity}")

    geometry = fold_feature_geometry(
        frame,
        feature_columns=feature_columns,
        time_column=time_column,
        fold_column=fold_column,
    )
    empty_in_fold = [
        (rec["fold"], rec["feature"]) for rec in geometry if rec["n_null"] == rec["n_rows"]
    ]
    if empty_in_fold:
        raise ValueError(
            "feature columns with no value at all in a fold, which means the fit did not "
            f"run or its output was not joined back: {empty_in_fold}"
        )

    if expected_folds is not None:
        got = sorted({int(f) for f in frame[fold_column].unique()})
        want = sorted(int(f) for f in expected_folds)
        if got != want:
            raise ValueError(f"fold ids {got} do not match the resolved folds {want}")

    merged = dict(metadata or {})
    merged["fold_feature_geometry"] = [
        {
            **rec,
            "first_valid": None if rec["first_valid"] is None else str(rec["first_valid"]),
            "last_valid": None if rec["last_valid"] is None else str(rec["last_valid"]),
        }
        for rec in geometry
    ]
    return write_artifact(
        frame,
        path,
        keys=frame_keys,
        written_by=written_by,
        inputs=inputs,
        metadata=merged,
        fold_column=fold_column,
    )


# ---------------------------------------------------------------------------
# The walk-forward refit schedule
# ---------------------------------------------------------------------------
#
# Two channels carry data into a fitted feature value at time t: the CONDITIONING set (which
# observations the value is computed from) and the PARAMETERS (which observations theta was
# estimated from). A causal feature needs both to end at or before t. The forward-filtering
# helpers above close the first channel. This closes the second.
#
# The design these replace fitted theta once per fold on the fold's whole training window and
# then ran the model forward from the START of that same window. Every training row therefore
# carried parameters estimated from its own future - for etfs' fold 0, the earliest rows carried
# 9.8 years of it - while every validation row carried parameters estimated only from its past.
# The model was then fitted on one version of the column and scored on another. Nothing raised,
# because a fold's own rows are internally consistent and the artifact records no estimation
# window.
#
# `walk_forward_feature` is the design cme_futures' ARIMA already used and this generalizes:
# spend a burn-in, fit, emit until the next refit, refit on everything up to that point, carry
# on. The schedule replaces the fold as the thing that bounds an estimate, so the artifact
# stops needing a fold column at all - one value per (entity, timestamp), the same value
# whichever fold later selects the row.


def refit_boundaries(n_obs: int, burnin: int, refit_every: int) -> list[tuple[int, int]]:
    """``(fit_end, emit_end)`` index pairs for one walk over ``n_obs`` observations.

    ``fit_end`` is exclusive, so a block's parameters are estimated from ``obs[:fit_end]`` and
    then speak for ``obs[fit_end:emit_end]`` - no observation is ever used to fit the model that
    describes it. The first ``burnin`` observations are in no block: they pay for the first
    estimate and carry no feature value.

    Returns an empty list when there is not enough history to fit even once, which is a
    statement about the series and not an error. The caller emits nulls for the whole of it.
    """
    if burnin < 1:
        raise ValueError(f"burnin must be at least 1, got {burnin}")
    if refit_every < 1:
        raise ValueError(f"refit_every must be at least 1, got {refit_every}")
    if n_obs <= burnin:
        return []
    return [
        (fit_end, min(fit_end + refit_every, n_obs))
        for fit_end in range(burnin, n_obs, refit_every)
    ]


def walk_forward_feature(
    X: np.ndarray,
    *,
    burnin: int,
    refit_every: int,
    fit: Callable[[np.ndarray], Any],
    apply: Callable[[Any, np.ndarray], np.ndarray],
    n_features: int,
    window: int | None = None,
    on_fit_error: str = "raise",
) -> np.ndarray:
    """Emit a fitted feature over one series, refitting on a schedule instead of per fold.

    ``fit(X_train)`` returns whatever object ``apply`` needs. ``apply(model, X_prefix)`` runs the
    fitted model forward over ``X_prefix`` and returns one row per input row; only the rows of
    the current block are kept, so ``apply`` may condition on everything up to each row and must
    not read past the end of what it is handed.

    ``window`` is ``None`` for an expanding estimation window - every refit sees the whole
    history, which is what cme_futures' ARIMA does - or an integer for a rolling one of that many
    observations, for a model whose parameters are expected to drift.

    ``on_fit_error="skip"`` leaves a block null and carries on with the previous parameters where
    a single estimate fails to converge; the default raises, because a model that cannot be
    fitted on most of its blocks is not a feature.

    Returns ``(len(X), n_features)`` with ``np.nan`` wherever no parameters were available: the
    burn-in prefix always, and any skipped block.
    """
    if on_fit_error not in ("raise", "skip"):
        raise ValueError(f"on_fit_error must be 'raise' or 'skip', got {on_fit_error!r}")
    out = np.full((len(X), n_features), np.nan, dtype=float)
    for fit_end, emit_end in refit_boundaries(len(X), burnin, refit_every):
        fit_start = 0 if window is None else max(0, fit_end - window)
        try:
            model = fit(X[fit_start:fit_end])
        except Exception:
            if on_fit_error == "raise":
                raise
            continue
        values = np.asarray(apply(model, X[:emit_end]), dtype=float)
        if len(values) != emit_end:
            raise ValueError(
                f"apply returned {len(values)} rows for a {emit_end}-row prefix; it must return "
                "one row per input row so the block slice below lines up"
            )
        out[fit_end:emit_end] = values[fit_end:emit_end].reshape(emit_end - fit_end, n_features)
    return out
