"""The nasdaq HAR walk emits exactly what the hand-rolled refit loop emitted.

`04_model_based_features` used to carry its own loop over refit boundaries: for every bar,
slice the trailing window, drop the rows with a null in them, least-squares, forecast the bar,
and difference against the forecast made for it one bar earlier. That loop is gone and the
schedule is `walk_forward_feature`'s, which is what criterion 4 of the standardization spec
asks for.

The reason this is a test and not a claim is criterion 7b. `model_based.parquet` is pinned by
whole-file sha256 inside `computation.feature_artifacts`, so every registered training run on
this case study stays valid if and only if the conversion moves no value. The loop below is the
one that was removed, kept here verbatim as the thing the new path has to reproduce - exactly,
not within a tolerance, because both are the same arithmetic on the same bars.

The series are built to exercise the three branches that made the old loop's null pattern what
it is: a leading warm-up where the widest component has no value yet, a stretch too thin to
identify the regression so the refit is skipped, and a constant stretch whose design matrix is
rank deficient.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

from case_studies.utils.temporal import walk_forward_feature

FIT_WINDOW = 120
MIN_TRAIN_OBS = 20


def har_loop(
    rv_5: np.ndarray,
    rv_15: np.ndarray,
    rv_60: np.ndarray,
    fit_window: int = FIT_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The refit loop `04_model_based_features` carried before it adopted the harness."""
    n = len(rv_5)
    har_forecast = np.full(n, np.nan)
    har_residual = np.full(n, np.nan)
    har_betas = np.full((n, 4), np.nan)

    for t in range(fit_window + 1, n):
        start = t - fit_window
        y_train = rv_5[start + 1 : t + 1]
        X_train = np.column_stack(
            [np.ones(fit_window), rv_5[start:t], rv_15[start:t], rv_60[start:t]]
        )

        valid_mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
        if valid_mask.sum() < MIN_TRAIN_OBS:
            continue

        try:
            beta = np.linalg.lstsq(X_train[valid_mask], y_train[valid_mask], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        har_betas[t] = beta

        x_t = np.array([1.0, rv_5[t], rv_15[t], rv_60[t]])
        if np.all(np.isfinite(x_t)):
            har_forecast[t] = x_t @ beta
            if np.isfinite(rv_5[t]):
                har_residual[t] = (
                    rv_5[t] - har_forecast[t - 1] if np.isfinite(har_forecast[t - 1]) else np.nan
                )

    return har_forecast, har_residual, har_betas


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "case_studies/nasdaq100_microstructure/04_model_based_features.py"
)


def _notebook_har_functions() -> tuple[Callable, Callable]:
    """`fit_har_window` and `apply_har` as the notebook defines them.

    Lifted out of the source rather than reimplemented here. A local copy would make this
    file agree with itself while the notebook drifted, which is the one way an equivalence
    test can pass and mean nothing. Importing the module is not an option - it reads a
    3.6 GB panel at import time - so the two function definitions are compiled on their own,
    with the module-level constants they close over supplied from the same `setup.yaml` the
    notebook reads.
    """
    tree = ast.parse(NOTEBOOK.read_text())
    wanted = {"fit_har_window", "apply_har"}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {n.name for n in defs}
    if missing:
        raise AssertionError(f"04_model_based_features no longer defines {sorted(missing)}")

    setup = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "case_studies/nasdaq100_microstructure/config/setup.yaml"
        ).read_text()
    )
    namespace: dict = {
        "np": np,
        "HAR_MIN_TRAIN_OBS": setup["model_based"]["har"]["min_train_obs"],
    }
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["fit_har_window"], namespace["apply_har"]


fit_har_window, apply_har = _notebook_har_functions()


def har_walk(rv_5: np.ndarray, rv_15: np.ndarray, rv_60: np.ndarray):
    """The path the notebook now takes."""
    series = np.column_stack([rv_5, rv_15, rv_60, np.append(rv_5[1:], np.nan)])
    emitted = walk_forward_feature(
        series,
        timestamps=np.arange(len(series)),
        burnin=FIT_WINDOW + 1,
        refit_every=1,
        window=FIT_WINDOW,
        fit=fit_har_window,
        apply=apply_har,
        apply_scope="block",
        n_features=4,
        on_fit_error="skip",
    )
    forecast, betas = emitted[:, 0], emitted[:, 1:]
    previous = np.concatenate([[np.nan], forecast[:-1]])
    residual = np.where(np.isfinite(forecast) & np.isfinite(previous), rv_5 - previous, np.nan)
    return forecast, residual, betas


def _components(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realized-variance components with the null and rank shapes the real panel has."""
    rng = np.random.default_rng(seed)
    r2 = np.abs(rng.standard_normal(n)) ** 2 * 1e-8
    # A constant stretch: the three components stop varying, so the design matrix over a
    # window inside it is rank deficient and lstsq's answer is not the loop's business.
    r2[400:560] = 4e-9
    rv_5 = np.full(n, np.nan)
    rv_15 = np.full(n, np.nan)
    rv_60 = np.full(n, np.nan)
    for t in range(60, n):
        rv_5[t] = r2[t - 5 : t].mean()
        rv_15[t] = r2[t - 15 : t].mean()
        rv_60[t] = r2[t - 60 : t].mean()
    # A stretch the refit cannot use: fewer than MIN_TRAIN_OBS usable rows in the window that
    # ends just past it, so the old loop skipped and left a null.
    rv_15[130:245] = np.nan
    return rv_5, rv_15, rv_60


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_walk_reproduces_the_loop_it_replaced(seed: int) -> None:
    rv_5, rv_15, rv_60 = _components(900, seed)

    want_forecast, want_residual, want_betas = har_loop(rv_5, rv_15, rv_60)
    got_forecast, got_residual, got_betas = har_walk(rv_5, rv_15, rv_60)

    # The loop kept an intercept it never emitted; the walk emits the three slopes the
    # notebook reads. Compare what both produced.
    np.testing.assert_array_equal(got_forecast, want_forecast)
    np.testing.assert_array_equal(got_residual, want_residual)
    np.testing.assert_array_equal(got_betas, want_betas[:, 1:])


def test_the_series_exercises_the_skip_and_the_warmup() -> None:
    """A comparison of two all-null arrays would pass and prove nothing."""
    rv_5, rv_15, rv_60 = _components(900, 0)
    forecast, residual, betas = har_walk(rv_5, rv_15, rv_60)

    assert np.isnan(forecast[: FIT_WINDOW + 1]).all(), "the burn-in has to carry no value"
    assert np.isfinite(forecast).sum() > 400, "most bars past the burn-in should carry a fit"
    assert np.isnan(forecast[FIT_WINDOW + 1 :]).any(), "no refit was ever skipped"
    assert np.isfinite(residual).sum() > 400
    assert np.isfinite(betas).all(axis=1).sum() > 400


def test_the_declared_floor_is_the_one_this_file_exercises() -> None:
    """The notebook's functions close over `HAR_MIN_TRAIN_OBS`, supplied from setup.yaml.

    The equivalence above is against a loop that hard-codes the floor, so the two agree only
    while the declared value is the one the loop was written for. If the declaration moves,
    this fails here rather than silently comparing two different models.
    """
    setup = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "case_studies/nasdaq100_microstructure/config/setup.yaml"
        ).read_text()
    )
    har = setup["model_based"]["har"]
    assert har["min_train_obs"] == MIN_TRAIN_OBS
    assert har["fit_window"] == FIT_WINDOW
    assert har["refit_every"] == 1, "the loop this file pins refits at every bar"
