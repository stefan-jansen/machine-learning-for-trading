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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
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
    geometry = {(g["fold"], g["feature"]): g for g in record["fold_feature_geometry"]}
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


def test_write_model_based_writes_nothing_when_a_guard_fires(tmp_path: Path) -> None:
    out = tmp_path / "m.parquet"
    with pytest.raises(ValueError):
        write_model_based(_emit_frame(folds=(0, 9)), out, expected_folds=[0, 1], **WRITE_KW)
    assert not out.exists()


# --- the fold-free artifact a refit schedule produces ------------------------------------


def _fold_free_frame() -> pl.DataFrame:
    return _emit_frame(folds=(0,)).drop("fold")


def test_write_model_based_writes_a_frame_that_carries_no_fold_column(tmp_path: Path) -> None:
    out = tmp_path / "model_based.parquet"
    record = write_model_based(_fold_free_frame(), out, fold_column=None, **WRITE_KW)
    assert "fold" not in pl.read_parquet(out).columns
    assert record["n_rows"] == 18
    # One geometry record per feature, not per fold and feature.
    assert {(g["fold"], g["feature"]) for g in record["fold_feature_geometry"]} == {
        (None, "vol_state"),
        (None, "garch_sigma"),
    }


def test_the_keys_alone_identify_a_row_when_there_is_no_fold(tmp_path: Path) -> None:
    frame = _fold_free_frame()
    frame = pl.concat([frame, frame.head(1)])
    with pytest.raises(ValueError, match="duplicate rows"):
        write_model_based(frame, tmp_path / "m.parquet", fold_column=None, **WRITE_KW)


def test_expected_folds_is_refused_rather_than_ignored_without_a_fold_column(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m.parquet"
    with pytest.raises(ValueError, match="expected_folds"):
        write_model_based(_fold_free_frame(), out, fold_column=None, expected_folds=[0], **WRITE_KW)
    assert not out.exists()


def test_an_all_null_feature_is_still_refused_without_a_fold_column(tmp_path: Path) -> None:
    frame = _fold_free_frame().with_columns(pl.lit(None, dtype=pl.Float64).alias("garch_sigma"))
    with pytest.raises(ValueError, match="no value at all"):
        write_model_based(frame, tmp_path / "m.parquet", fold_column=None, **WRITE_KW)


# ---------------------------------------------------------------------------
# The GARCH(1,1) recursion.
#
# The property is the same one the schedule exists for, one level down: a value must not move
# when observations after it are appended. `arch`'s own result object fails it, which is why
# there is a recursion here at all.
# ---------------------------------------------------------------------------

GARCH_PARAMS = dict(mu=0.05, omega=0.02, alpha=0.08, beta=0.90, backcast=1.5)
GJR_PARAMS = dict(mu=0.05, omega=0.02, alpha=0.04, beta=0.90, gamma=0.07, backcast=1.5)


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


def test_the_gjr_recursion_does_not_move_when_later_returns_arrive() -> None:
    # Same property with the asymmetry term on, because gamma enters through the sign of the
    # PREVIOUS residual and a prefix ends one observation earlier.
    returns = _garch_returns()
    short = garch11_conditional_volatility(returns[:500], **GJR_PARAMS)
    full = garch11_conditional_volatility(returns, **GJR_PARAMS)
    np.testing.assert_array_equal(short, full[:500])


def test_arch_moves_earlier_values_when_the_sample_is_extended() -> None:
    """The negative control: the prefix-stability assertions above are not vacuous.

    A prefix-stability assertion passes trivially against any implementation that happens to
    process observations in order, so on its own it is evidence about nothing. This runs the
    identical assertion against `arch`'s own fixed-parameter result object - the thing the
    helper replaces - and requires it to FAIL.

    The mechanism is one line of `ARCHModel.fix`: it takes `resids = self.resids(
    self.starting_values())`, the residuals at the ESTIMATED mean of whatever array it was
    handed, and seeds the backcast from those. Under `mean="Constant"` the estimated mean moves
    when the sample is extended, so the seed moves with it.

    If `arch` ever stops doing this the test goes red, and the helper's reason for existing has
    to be re-established rather than assumed.
    """
    arch_model = pytest.importorskip("arch").arch_model
    returns = _garch_returns(n=2000)
    params = pd.Series(
        {
            "mu": GARCH_PARAMS["mu"],
            "omega": GARCH_PARAMS["omega"],
            "alpha[1]": GARCH_PARAMS["alpha"],
            "beta[1]": GARCH_PARAMS["beta"],
        }
    )

    def arch_path(sample: np.ndarray) -> np.ndarray:
        spec = arch_model(sample, mean="Constant", vol="GARCH", p=1, q=1, dist="Normal")
        return np.asarray(spec.fix(params).conditional_volatility, dtype=float)

    short = arch_path(returns[:1500])
    full = arch_path(returns)[:1500]

    assert not np.array_equal(short, full), (
        "arch's fixed-parameter path is prefix-stable under an estimated mean, which "
        "contradicts the measurement this helper was written for. Re-derive the reason before "
        "deleting the helper."
    )
    # Pin the shape, so a crash or an all-nan path cannot satisfy the assertion above. The
    # disagreement is a seeding effect: largest at the start, decaying to nothing.
    relative = np.abs(short - full) / np.maximum(np.abs(full), 1e-12)
    assert np.isfinite(relative).all()
    assert relative.max() > 1e-6
    assert relative[0] > relative[-1]


def test_the_arch_seed_is_taken_at_the_estimated_mean_not_the_fixed_one(monkeypatch) -> None:
    """Name the mechanism, so the control above cannot pass for an unrelated reason.

    Without this, `fix` could start disagreeing with itself for some other cause and the
    negative control would keep passing while the stated reason had become false. The
    assertion has to be about what `fix` ITSELF hands to `backcast`, not about what an
    independent recomputation would hand it, so this captures the array in flight.
    """
    arch_model = pytest.importorskip("arch").arch_model
    volatility = pytest.importorskip("arch.univariate.volatility")

    seen: list[np.ndarray] = []
    original = volatility.GARCH.backcast

    def spy(self, resids):
        seen.append(np.asarray(resids, dtype=float).copy())
        return original(self, resids)

    monkeypatch.setattr(volatility.GARCH, "backcast", spy)

    returns = _garch_returns(n=2000)
    params = pd.Series(
        {
            "mu": GARCH_PARAMS["mu"],
            "omega": GARCH_PARAMS["omega"],
            "alpha[1]": GARCH_PARAMS["alpha"],
            "beta[1]": GARCH_PARAMS["beta"],
        }
    )
    for sample in (returns[:1500], returns):
        arch_model(sample, mean="Constant", vol="GARCH", p=1, q=1, dist="Normal").fix(params)

    assert len(seen) == 2, "fix should seed the recursion exactly once per call"
    short_resids, full_resids = seen

    # What fix passed is the sample centred at its OWN estimated mean, not at the mu given.
    np.testing.assert_allclose(short_resids, returns[:1500] - returns[:1500].mean(), rtol=1e-12)
    np.testing.assert_allclose(full_resids, returns - returns.mean(), rtol=1e-12)
    assert not np.allclose(short_resids, returns[:1500] - GARCH_PARAMS["mu"])

    # And that is what makes the seed move: the same residuals taken at the fixed mu would not.
    garch = volatility.GARCH(p=1, q=1)
    assert original(garch, short_resids) != original(garch, full_resids[:1500])
    assert original(garch, returns[:1500] - GARCH_PARAMS["mu"]) == original(
        garch, returns - GARCH_PARAMS["mu"]
    )


def test_arch_is_prefix_stable_under_a_zero_mean_until_the_bounds_bind() -> None:
    """The channel is narrower under `mean="Zero"`, and does not close.

    crypto_perps_funding fits a zero-mean model, where nothing is estimated and the seed
    therefore does not move. What still moves is `variance_bounds`, which clamps every row
    with two WHOLE-SAMPLE quantities - `np.var(resids) / 1e8` below and
    `1e7 * (1 + max(resids**2))` above. They sit six orders of magnitude apart, so they change
    an emitted value only when the variance reaches one - which is what a degenerate fit does.
    """
    arch_model = pytest.importorskip("arch").arch_model
    rng = np.random.default_rng(5)
    # A quiet series, then a crash 500 observations past the cut. The crash is what moves
    # np.var(resids), and with it the lower clamp under every row that came before it.
    quiet = rng.normal(0.0, 0.5, 1500)
    tail = rng.normal(0.0, 0.5, 500)
    tail[250] = 400.0
    returns = np.concatenate([quiet, tail])

    def zero_mean_path(sample: np.ndarray, params: pd.Series) -> np.ndarray:
        spec = arch_model(sample, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="Normal")
        return np.asarray(spec.fix(params).conditional_volatility, dtype=float)

    healthy = pd.Series({"omega": 0.02, "alpha[1]": 0.05, "gamma[1]": 0.08, "beta[1]": 0.88})
    np.testing.assert_array_equal(
        zero_mean_path(returns[:1500], healthy), zero_mean_path(returns, healthy)[:1500]
    )

    # alpha + gamma < 0: a down day REDUCES the variance, so the recursion is driven onto the
    # lower clamp. arch returns this shape without complaint.
    degenerate = pd.Series({"omega": 1e-8, "alpha[1]": 0.02, "gamma[1]": -0.30, "beta[1]": 0.90})
    short = zero_mean_path(returns[:1500], degenerate)
    full = zero_mean_path(returns, degenerate)[:1500]
    relative = np.abs(short - full) / np.maximum(np.abs(full), 1e-30)
    assert (relative > 1e-9).sum() > 1000
    assert relative.max() > 0.1

    # The helper does not clip that fit silently - it refuses it.
    with pytest.raises(ValueError, match=r"alpha \+ gamma"):
        garch11_conditional_volatility(
            returns, mu=0.0, omega=1e-8, alpha=0.02, beta=0.90, gamma=-0.30, backcast=0.25
        )


def test_the_garch_recursion_reproduces_its_own_definition() -> None:
    returns = _garch_returns(n=60)
    got = garch11_conditional_volatility(returns, **GARCH_PARAMS)
    mu, omega = GARCH_PARAMS["mu"], GARCH_PARAMS["omega"]
    alpha, beta = GARCH_PARAMS["alpha"], GARCH_PARAMS["beta"]
    resid = returns - mu
    want = np.empty(len(returns))
    want[0] = omega + (alpha + beta) * GARCH_PARAMS["backcast"]
    for t in range(1, len(returns)):
        want[t] = omega + alpha * resid[t - 1] ** 2 + beta * want[t - 1]
    np.testing.assert_allclose(got, np.sqrt(want), rtol=1e-12)


def test_the_gjr_recursion_reproduces_its_own_definition() -> None:
    returns = _garch_returns(n=60)
    got = garch11_conditional_volatility(returns, **GJR_PARAMS)
    mu, omega = GJR_PARAMS["mu"], GJR_PARAMS["omega"]
    alpha, beta, gamma = GJR_PARAMS["alpha"], GJR_PARAMS["beta"], GJR_PARAMS["gamma"]
    resid = returns - mu
    want = np.empty(len(returns))
    want[0] = omega + (alpha + 0.5 * gamma + beta) * GJR_PARAMS["backcast"]
    for t in range(1, len(returns)):
        shock = alpha + gamma * (resid[t - 1] < 0.0)
        want[t] = omega + shock * resid[t - 1] ** 2 + beta * want[t - 1]
    np.testing.assert_allclose(got, np.sqrt(want), rtol=1e-12)


def test_a_zero_gamma_is_the_symmetric_model() -> None:
    returns = _garch_returns(n=200)
    np.testing.assert_array_equal(
        garch11_conditional_volatility(returns, **GARCH_PARAMS),
        garch11_conditional_volatility(returns, gamma=0.0, **GARCH_PARAMS),
    )


def test_the_asymmetry_raises_variance_only_after_a_negative_return() -> None:
    # gamma has to attach to the sign of the residual it multiplies, not to the current row.
    # A sign error here is invisible to every aggregate and to the definition test above if
    # that test repeats the same mistake, so it is stated directly against constructed input.
    returns = np.array([0.05, 0.05 - 1.0, 0.05 + 1.0, 0.05, 0.05])
    sym = garch11_conditional_volatility(returns, **GARCH_PARAMS)
    asym = garch11_conditional_volatility(returns, **{**GARCH_PARAMS, "gamma": 0.07})
    # Row 2 follows the -1.0 residual and must be lifted; row 3 follows the +1.0 and must not.
    assert asym[2] > sym[2]
    assert asym[3] == pytest.approx(
        sym[3] + GARCH_PARAMS["beta"] * (asym[2] ** 2 - sym[2] ** 2) / (asym[3] + sym[3]), rel=1e-6
    )


def test_a_degenerate_fit_raises_rather_than_emitting_nan() -> None:
    """arch returns alpha + gamma < 0 without complaint, and sqrt of it is nan.

    A nan is not a feature value. Measured on sp500_equity_option_analytics: 19% of first
    blocks come back with a negative net shock coefficient, and the old path emitted them.
    """
    with pytest.raises(ValueError, match=r"alpha \+ gamma"):
        garch11_conditional_volatility(
            _garch_returns(n=400),
            mu=0.0,
            omega=1e-8,
            alpha=0.02,
            beta=0.90,
            gamma=-0.30,
            backcast=1.0,
        )


def test_bounds_clip_the_recursion_instead_of_raising() -> None:
    out = garch11_conditional_volatility(
        _garch_returns(n=400),
        mu=0.0,
        omega=1e-8,
        alpha=0.02,
        beta=0.90,
        gamma=-0.30,
        backcast=1.0,
        bounds=(1e-6, 1e4),
    )
    assert np.isfinite(out).all()
    assert (out >= np.sqrt(1e-6) - 1e-12).all()


def test_an_integrated_fit_no_longer_seeds_from_the_parameters() -> None:
    """Persistence 1.0 has no long-run variance; the seed comes from the data instead.

    Two of 25 sampled sp500_equity_option_analytics securities fitted to persistence 1.0000,
    where omega/(1 - persistence) is infinite, any clamp makes it omega x 1e6, and beta = 1
    means that seed never decays. The backcast bounds the seed whatever the optimizer returns.
    """
    out = garch11_conditional_volatility(
        _garch_returns(n=300), mu=0.0, omega=0.02, alpha=0.10, beta=0.90, backcast=1.5
    )
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(np.sqrt(0.02 + 1.0 * 1.5))


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


def _walk(X: np.ndarray, **kwargs) -> np.ndarray:
    """``walk_forward_feature`` over a series whose rows are one session apart.

    The tests below are about the refit schedule, and every one of them wants the same
    unremarkable time axis. Supplying it here keeps ``timestamps`` written out only in the
    tests that are about ``timestamps``.
    """
    kwargs.setdefault("timestamps", np.arange(len(X)))
    return walk_forward_feature(X, **kwargs)


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
    out = _walk(
        X, burnin=20, refit_every=refit_every, fit=_mean_fit, apply=_emit_parameter, n_features=1
    )
    assert np.isnan(out[:20]).all()
    assert np.isfinite(out[20:]).all()


def test_a_value_is_computed_from_parameters_fitted_strictly_before_it() -> None:
    """The whole point. Row 20's parameter is the mean of rows 0-19 and of nothing later."""
    X = np.arange(60, dtype=float).reshape(-1, 1)
    out = _walk(X, burnin=20, refit_every=10, fit=_mean_fit, apply=_emit_parameter, n_features=1)
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
    full = _walk(X, burnin=15, refit_every=6, fit=_mean_fit, apply=_emit_parameter, n_features=1)
    for cut in (40, 55, 79):
        short = _walk(
            X[:cut], burnin=15, refit_every=6, fit=_mean_fit, apply=_emit_parameter, n_features=1
        )
        np.testing.assert_allclose(short, full[:cut], equal_nan=True)


def test_a_rolling_window_forgets_and_an_expanding_one_does_not() -> None:
    X = np.arange(60, dtype=float).reshape(-1, 1)
    kwargs = dict(burnin=20, refit_every=10, fit=_mean_fit, apply=_emit_parameter, n_features=1)
    expanding = _walk(X, **kwargs)
    rolling = _walk(X, window=20, **kwargs)
    assert expanding[20, 0] == pytest.approx(rolling[20, 0])
    assert rolling[40, 0] == pytest.approx(X[20:40, 0].mean())
    assert expanding[40, 0] == pytest.approx(X[:40, 0].mean())


def test_the_rolling_window_is_still_bounded_by_the_start_of_the_series() -> None:
    X = np.arange(30, dtype=float).reshape(-1, 1)
    out = _walk(
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
    out = _walk(
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
        _walk(
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

    out = _walk(
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
        _walk(
            np.arange(30, dtype=float).reshape(-1, 1),
            burnin=10,
            refit_every=5,
            fit=_mean_fit,
            apply=lambda model, X: np.full((len(X) - 1, 1), model),
            n_features=1,
        )


def test_a_block_scoped_apply_sees_only_the_rows_the_walk_emits() -> None:
    """``apply_scope="block"`` is what makes a per-bar refit affordable, so it has to be exact.

    The prefix form is quadratic in the length of the series: at one refit per bar it asks
    ``apply`` for ``n(n+1)/2`` rows to keep ``n`` of them. A model whose value at a row is a
    function of that row and the parameters needs none of that, and the two forms have to agree
    on every value for the substitution to be a cost change rather than a behaviour change.
    """
    X = np.arange(40, dtype=float).reshape(-1, 1)
    seen: list[int] = []

    def apply_block(model, block):
        seen.append(len(block))
        return np.full((len(block), 1), model)

    out = _walk(
        X,
        burnin=10,
        refit_every=5,
        fit=_mean_fit,
        apply=apply_block,
        n_features=1,
        apply_scope="block",
    )
    prefix = _walk(X, burnin=10, refit_every=5, fit=_mean_fit, apply=_emit_parameter, n_features=1)
    np.testing.assert_array_equal(out, prefix)
    assert seen == [5, 5, 5, 5, 5, 5], "apply was handed something other than the block"


def test_a_block_scoped_apply_returning_the_wrong_number_of_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="one row per input row"):
        _walk(
            np.arange(30, dtype=float).reshape(-1, 1),
            burnin=10,
            refit_every=5,
            fit=_mean_fit,
            apply=lambda model, block: np.full((len(block) + 1, 1), model),
            n_features=1,
            apply_scope="block",
        )


def test_an_unknown_apply_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="apply_scope must be"):
        _walk(
            np.arange(30, dtype=float).reshape(-1, 1),
            burnin=10,
            refit_every=5,
            fit=_mean_fit,
            apply=_emit_parameter,
            n_features=1,
            apply_scope="whole",
        )


def test_a_multi_column_feature_keeps_its_columns_in_order() -> None:
    out = _walk(
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
        full = _walk(X, **kwargs)
        short = _walk(X[:220], **kwargs)
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
    out = _walk(
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
    out = _walk(
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
        _walk(X, freeze_after=40, **kwargs)[:41],
        _walk(X, **kwargs)[:41],
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# The time axis
# ---------------------------------------------------------------------------
#
# Everything above this line is about WHEN the model is refitted. These are about whether the
# rows it walks are in time order at all. Until the guard they were not checked anywhere: the
# function receives a bare float array, so an unsorted frame and two entities concatenated both
# read as one well-formed series, produce a feature fitted on scrambled history, and raise
# nothing. Six case studies' stage-04 notebooks sorted correctly and that discipline was the
# only thing enforcing it.


def _walk_over(timestamps) -> np.ndarray:
    """Walk a 40-row series carrying the given time axis, whatever order it is in."""
    return walk_forward_feature(
        np.arange(len(timestamps), dtype=float).reshape(-1, 1),
        timestamps=timestamps,
        burnin=10,
        refit_every=5,
        fit=_mean_fit,
        apply=_emit_parameter,
        n_features=1,
    )


def test_a_walk_in_time_order_is_accepted() -> None:
    out = _walk_over(pl.date_range(date(2020, 1, 1), date(2020, 2, 9), eager=True))
    assert out.shape == (40, 1)
    assert np.isnan(out[:10]).all()
    assert np.isfinite(out[10:]).all()


def test_an_unsorted_series_is_refused() -> None:
    """A shuffled frame: the rows are all there, and 'the rows before it' means nothing."""
    days = pl.date_range(date(2020, 1, 1), date(2020, 2, 9), eager=True).to_numpy()
    shuffled = days.copy()
    shuffled[7], shuffled[8] = days[8], days[7]
    with pytest.raises(ValueError, match="do not strictly increase"):
        _walk_over(shuffled)


def test_two_entities_concatenated_are_refused() -> None:
    """The failure a missing ``partition_by`` produces: one call, two securities' histories."""
    one = pl.date_range(date(2020, 1, 1), date(2020, 1, 20), eager=True).to_numpy()
    both = np.concatenate([one, one])
    with pytest.raises(ValueError, match="do not strictly increase"):
        _walk_over(both)


def test_a_repeated_timestamp_is_refused() -> None:
    """Two rows the schedule cannot order are two rows it cannot separate fit from emit on."""
    days = pl.date_range(date(2020, 1, 1), date(2020, 2, 9), eager=True).to_numpy()
    days[20] = days[19]
    with pytest.raises(ValueError, match="do not strictly increase"):
        _walk_over(days)


def test_a_time_axis_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="entries for a"):
        walk_forward_feature(
            np.arange(40, dtype=float).reshape(-1, 1),
            timestamps=np.arange(39),
            burnin=10,
            refit_every=5,
            fit=_mean_fit,
            apply=_emit_parameter,
            n_features=1,
        )


def test_the_guard_refuses_before_the_first_fit_runs() -> None:
    """Atomicity: nothing is estimated, so a refused walk leaves no partial state behind."""
    fitted: list[int] = []

    def recording_fit(X: np.ndarray) -> float:
        fitted.append(len(X))
        return _mean_fit(X)

    with pytest.raises(ValueError, match="do not strictly increase"):
        walk_forward_feature(
            np.arange(40, dtype=float).reshape(-1, 1),
            timestamps=np.arange(40)[::-1],
            burnin=10,
            refit_every=5,
            fit=recording_fit,
            apply=_emit_parameter,
            n_features=1,
        )
    assert fitted == []


def test_a_polars_series_and_a_numpy_array_are_read_the_same_way() -> None:
    days = pl.date_range(date(2020, 1, 1), date(2020, 2, 9), eager=True)
    np.testing.assert_allclose(_walk_over(days), _walk_over(days.to_numpy()), equal_nan=True)


def test_a_series_too_short_to_compare_is_accepted() -> None:
    """One row cannot be out of order, and a walk over it is empty rather than an error."""
    out = walk_forward_feature(
        np.zeros((1, 1)),
        timestamps=np.array([date(2020, 1, 1)]),
        burnin=10,
        refit_every=5,
        fit=_mean_fit,
        apply=_emit_parameter,
        n_features=1,
    )
    assert np.isnan(out).all()
