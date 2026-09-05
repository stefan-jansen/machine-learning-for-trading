"""`us_equities_panel` stage 04 emits a value that cannot move once it is emitted.

The notebook's two fitted transforms - the Wasserstein regime clustering and the per-stock
GARCH volatility - are driven by `walk_forward_feature`, so a value at a session is a
function of that session's own past and of parameters estimated strictly earlier. Neither
property is visible in the artifact: a value fitted on its own future looks exactly like one
that was not, and the file records no estimation window. So it is checked here, on synthetic
series, by the only thing that distinguishes them - whether a value moves when later
observations arrive.

The functions are pulled out of the notebook source rather than reimplemented. A copy of the
logic would pass while the notebook drifted away from it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from case_studies.utils.temporal import (
    garch11_conditional_volatility,
    refit_boundaries,
    walk_forward_feature,
)

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "case_studies"
    / "us_equities_panel"
    / "04_model_based_features.py"
)

# The declared schedule, mirrored from `config/setup.yaml`. A change there is meant to change
# what the notebook fits; it is not meant to change whether the values are causal, which is
# what this module asserts, so the numbers are restated rather than read.
REGIME = dict(window=21, overlap=5, n_clusters=2, burnin=756, refit_every=63)
GARCH = dict(burnin=504, refit_every=63)

DEFINITIONS = (
    "LiftedStream",
    "lift_stream",
    "wasserstein_distance_1d",
    "wasserstein_barycenter_1d",
    "fit_wasserstein_kmeans",
    "fit_regime_centroids",
    "assign_regime_features",
    "garch_walk",
)


def _load(arch_model) -> dict:
    """The notebook's transform definitions, executed against a stub of its globals."""
    body = [
        node
        for node in ast.parse(NOTEBOOK.read_text()).body
        if getattr(node, "name", None) in DEFINITIONS
    ]
    missing = set(DEFINITIONS) - {node.name for node in body}
    assert not missing, f"{NOTEBOOK.name} no longer defines {sorted(missing)}"

    namespace = {
        "np": np,
        "dataclass": dataclass,
        "arch_model": arch_model,
        "walk_forward_feature": walk_forward_feature,
        "garch11_conditional_volatility": garch11_conditional_volatility,
        "FloatArray": np.ndarray,
        "IntArray": np.ndarray,
        "WASSERSTEIN_WINDOW": REGIME["window"],
        "WASSERSTEIN_OVERLAP": REGIME["overlap"],
        "N_CLUSTERS": REGIME["n_clusters"],
        "SEED": 42,
        "GARCH_KW": dict(mean="Constant", vol="GARCH", p=1, q=1, dist="Normal"),
        "GARCH_BURNIN": GARCH["burnin"],
        "GARCH_REFIT_EVERY": GARCH["refit_every"],
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def notebook_functions() -> dict:
    return _load(pytest.importorskip("arch").arch_model)


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    """Two thousand daily returns with a variance break two-thirds of the way through.

    The break is what makes the test able to fail: on a homoskedastic series a later
    observation barely moves a refit, so a leaky implementation and a causal one agree to
    several decimals.
    """
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(0.0005, 0.008, 1200), rng.normal(-0.001, 0.02, 800)])


def _regime_walk(functions: dict, series: np.ndarray, freeze_after: int) -> np.ndarray:
    return walk_forward_feature(
        series.reshape(-1, 1),
        burnin=REGIME["burnin"],
        refit_every=REGIME["refit_every"],
        fit=functions["fit_regime_centroids"],
        apply=functions["assign_regime_features"],
        n_features=5,
        freeze_after=freeze_after,
    )


def test_regime_values_do_not_move_when_later_sessions_arrive(notebook_functions, returns):
    full = _regime_walk(notebook_functions, returns, len(returns))
    short = _regime_walk(notebook_functions, returns[:1500], 1500)

    emitted = ~np.isnan(short[:, 1])
    assert emitted.sum() > 0, "the shorter walk emitted no regime value to compare"
    np.testing.assert_array_equal(
        full[:1500][emitted],
        short[emitted],
        err_msg="a regime value moved when 500 later sessions were appended",
    )


def test_a_regime_value_reads_the_sessions_before_it_and_no_others(notebook_functions, returns):
    centroids = notebook_functions["fit_regime_centroids"](returns[:900].reshape(-1, 1))
    assign = notebook_functions["assign_regime_features"]

    base = assign(centroids, returns[:1000].reshape(-1, 1))
    poked = returns[:1000].copy()
    poked[900] += 0.5
    after = assign(centroids, poked.reshape(-1, 1))

    np.testing.assert_array_equal(
        base[:901], after[:901], err_msg="session 900 reached a value dated at or before it"
    )
    assert not np.allclose(base[901], after[901]), (
        "session 900 did not reach the value at 901, so the window is not the one declared"
    )


def test_volatility_values_do_not_move_when_later_sessions_arrive(notebook_functions, returns):
    walk = notebook_functions["garch_walk"]
    percent = returns * 100

    _, full, fits = walk(("TEST", percent, len(returns)))
    _, short, _ = walk(("TEST", percent[:1500], 1500))

    emitted = ~np.isnan(short)
    assert emitted.sum() > 0, "the shorter walk emitted no volatility to compare"
    np.testing.assert_array_equal(
        full[:1500][emitted],
        short[emitted],
        err_msg="a conditional volatility moved when 500 later sessions were appended",
    )
    assert np.all(full[~np.isnan(full)] > 0), "a conditional volatility left the positive reals"
    assert np.all(np.isnan(full[: GARCH["burnin"]])), "a value was emitted inside the burn-in"
    assert len(fits) == len(
        refit_boundaries(len(returns), GARCH["burnin"], GARCH["refit_every"])
    ), f"{len(fits)} estimations against the declared schedule"


def test_freezing_stops_estimation_without_stopping_emission(notebook_functions, returns):
    walk = notebook_functions["garch_walk"]
    percent = returns * 100

    _, unfrozen, all_fits = walk(("TEST", percent, len(returns)))
    _, frozen, frozen_fits = walk(("TEST", percent, 1500))

    assert len(frozen_fits) < len(all_fits), "freezing stopped no estimation"
    assert max(fit["fit_end"] for fit in frozen_fits) <= 1500, (
        "an estimation read observations past the freeze point"
    )
    assert not np.isnan(frozen[-1]), (
        "freezing stopped emission as well as estimation; the holdout would carry no value"
    )


def test_a_fit_the_optimizer_did_not_converge_on_is_not_an_estimate(returns):
    """`arch` returns a result whatever the search did, and only warns.

    `show_warning=False` swallows the warning, so without an explicit check a parameter
    vector the optimizer never converged on would be filtered forward and written to the
    artifact as a feature value. What must happen instead is that the block is left empty.
    """

    class _Unconverged:
        convergence_flag = 1

        def __getattr__(self, name):  # pragma: no cover - reaching it is the failure
            raise AssertionError(f"a non-converged result was read for {name!r}")

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, **kwargs):
            return _Unconverged()

    functions = _load(_Model)
    _, values, fits = functions["garch_walk"](("TEST", returns * 100, len(returns)))

    assert fits == [], "a non-converged fit was recorded as an estimation"
    assert np.all(np.isnan(values)), "a non-converged fit produced a feature value"
