"""`fit_hmm_restarts` is the loop three case studies each wrote for themselves.

The property that matters is that adopting it does not move anybody's numbers. Two case
studies are on the canonical primitive today - `cme_futures` with a single fit and `etfs`
with its own `range(N_RESTARTS)` loop around it - and both must land byte-identical, or the
consolidation costs a regeneration it was supposed to avoid.

These reconstruct the two call shapes from the notebooks rather than describing them, so a
notebook that changes shape fails here instead of silently diverging.
"""

import numpy as np
import pytest

from case_studies.utils.temporal import HmmFit, fit_hmm_kmeans_init, fit_hmm_restarts


@pytest.fixture(scope="module")
def observations() -> np.ndarray:
    """Two columns and two overlapping regimes, which is the shape every HMM user feeds it.

    Overlapping rather than well separated on purpose: with clean separation every k-means
    partition lands in the same basin, every restart finds the same optimum, and a test
    over restarts would pass without exercising them.
    """
    rng = np.random.default_rng(11)
    calm = np.column_stack([rng.normal(0.0004, 0.008, 900), rng.normal(0.011, 0.004, 900)])
    stress = np.column_stack([rng.normal(-0.0012, 0.019, 500), rng.normal(0.024, 0.009, 500)])
    return np.vstack([calm, stress])


def _etfs_loop(X: np.ndarray, n_states: int, n_restarts: int):
    """`etfs/04_model_based_features.py::fit_regime_model`, reduced to its selection rule."""
    best, best_ll = None, -np.inf
    for seed in range(n_restarts):
        candidate = fit_hmm_kmeans_init(X, n_states=n_states, random_state=seed)
        ll = float(candidate.score(X))
        if ll > best_ll:
            best, best_ll = candidate, ll
    return best, best_ll


def _same_model(a, b) -> bool:
    return (
        np.array_equal(a.transmat_, b.transmat_)
        and np.array_equal(a.means_, b.means_)
        and np.array_equal(a.covars_, b.covars_)
        and np.array_equal(a.startprob_, b.startprob_)
    )


def test_one_restart_is_the_single_fit_cme_makes(observations):
    """`cme_futures/04:1405` calls the primitive once at a fixed seed."""
    assert _same_model(
        fit_hmm_kmeans_init(observations, n_states=2, random_state=42),
        fit_hmm_restarts(observations, n_states=2, random_state=42, n_restarts=1).model,
    )


@pytest.mark.parametrize("n_restarts", [2, 4])
def test_the_wrapper_reproduces_the_loop_etfs_wrote(observations, n_restarts):
    """`etfs/04:555-566` loops `range(N_RESTARTS)` and keeps the highest likelihood."""
    expected, expected_ll = _etfs_loop(observations, 2, n_restarts)
    got = fit_hmm_restarts(observations, n_states=2, random_state=0, n_restarts=n_restarts)
    assert _same_model(expected, got.model)
    assert got.log_likelihood == expected_ll


def test_restarts_explore_rather_than_repeat(observations):
    """A restart that lands where the last one did is a restart that bought nothing.

    Measured on etfs' own SPY panel, ten restarts give ten distinct log-likelihoods, so
    the k-means start does not collapse the search on real two-column data. If this ever
    reduces to one value the parameter has become a no-op and the case studies declaring
    ten are paying ten times over for one fit.
    """
    seen = {
        round(
            fit_hmm_restarts(observations, n_states=2, random_state=s, n_restarts=1).log_likelihood,
            9,
        )
        for s in range(6)
    }
    assert len(seen) > 1


class _Monitor:
    def __init__(self, history):
        self.history = history


class _StubModel:
    """Enough of a fitted model for the rejection rule, with a history it controls.

    Real fits on any data these tests can generate all end uphill - measured, six seeds,
    final deltas +1.8e-03 to +9.7e-03 - so a test driving `fit_hmm_kmeans_init` cannot
    reach the rejection branch at all. Asserting `n_rejected >= 0` against real fits is
    the vacuous version of this test: it passes whether or not the rule is implemented.
    """

    def __init__(self, history, score):
        self.monitor_ = _Monitor(history)
        self._score = score

    def score(self, X):  # noqa: ARG002 - signature is the library's
        return self._score


def test_a_downhill_final_step_is_rejected_when_asked(monkeypatch, observations):
    """A restart whose last EM step lowered the likelihood is discarded and counted."""
    histories = [
        [-1000.0, -900.0, -905.0],  # ends downhill by 5 nats against a scale of 900
        [-1000.0, -900.0, -899.0],  # ends uphill
    ]
    calls = iter(histories)
    monkeypatch.setattr(
        "case_studies.utils.temporal.fit_hmm_kmeans_init",
        lambda X, **kw: _StubModel(next(calls), score=-899.0),
    )
    result = fit_hmm_restarts(observations, n_states=2, n_restarts=2, reject_unstable_rel_tol=1e-3)
    assert isinstance(result, HmmFit)
    assert result.n_rejected == 1, "the downhill restart was kept"
    assert result.n_failed == 0
    assert result.model.monitor_.history == histories[1]


def test_the_rejection_rule_is_off_by_default(monkeypatch, observations):
    """A default that moves existing callers' values is a migration wearing a default's
    name, so `cme_futures` and `etfs` keep every restart until they ask not to."""
    monkeypatch.setattr(
        "case_studies.utils.temporal.fit_hmm_kmeans_init",
        lambda X, **kw: _StubModel([-1000.0, -900.0, -905.0], score=-905.0),
    )
    result = fit_hmm_restarts(observations, n_states=2, n_restarts=2)
    assert result.n_rejected == 0


def test_nothing_surviving_names_which_way_it_failed(observations):
    """Failed and rejected call for different fixes, so the message separates them."""
    with pytest.raises(RuntimeError, match="raised"):
        fit_hmm_restarts(observations[:1], n_states=2, random_state=0, n_restarts=2)


def test_a_restart_count_below_one_is_refused(observations):
    with pytest.raises(ValueError, match="at least 1"):
        fit_hmm_restarts(observations, n_states=2, n_restarts=0)
