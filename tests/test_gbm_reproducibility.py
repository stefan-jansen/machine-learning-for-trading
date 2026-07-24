"""Reproducibility guarantees for GBM training.

The pre-existing "determinism" tests (`tests/test_registry_specs.py`) assert that a spec
*dict* hashes the same as a copy of itself. That is bookkeeping. Nothing asserted that
training the same spec twice yields the same model.

It happens to. Every registered GBM number was produced by CPU LightGBM with `max_bin=63`
(recorded inside the saved boosters, which replay every published checkpoint to 0.000000)
— even though seven `setup.yaml` files requested `device: gpu`. `_best_gpu_device()`
returned None for the CPU-only PyPI wheel and `train_gbm_config` fell through to CPU
without a word, carrying the GPU branch's `max_bin` with it.

So the book's numbers are reproducible by accident, not by construction. Three things had
to be true at once, none of them asserted anywhere: the fallback was silent, `max_bin` was
inherited rather than declared, and neither `device` nor `max_bin` entered the training
hash. Any one of them changing moves every published result. A reader running the
documented `ml4t-gpu` image gets CUDA-LightGBM, which trains non-deterministically and
does not reproduce the book at all.

These tests pin the properties that were merely lucky.

See agents `issues/2026-07-08-cme-registry-numbers-not-reproducible.md`.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from case_studies.utils.gbm import GBM_DEFAULT_MAX_BIN, resolve_gbm_device, train_gbm_config
from case_studies.utils.registry.specs import build_training_spec, training_hash_from_spec
from utils.config import REPO_ROOT

N_DATES, N_ENTITIES, N_FEATURES = 60, 20, 6


def _panel(seed: int = 0):
    """Small synthetic cross-sectional panel with a real (noisy) signal."""
    rng = np.random.default_rng(seed)
    n = N_DATES * N_ENTITIES
    x = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    y = (0.4 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=1.0, size=n)).astype(np.float32)
    dates = np.repeat(np.arange(N_DATES), N_ENTITIES)
    entities = np.tile(np.arange(N_ENTITIES), N_DATES).astype(str)
    return x, y, dates, entities


def _fold_data(n_folds: int = 2):
    x, y, dates, entities = _panel()
    folds = []
    cut = N_DATES // (n_folds + 1)
    for f in range(n_folds):
        tr_end = cut * (f + 1)
        va_end = cut * (f + 2)
        tr = dates < tr_end
        va = (dates >= tr_end) & (dates < va_end)
        folds.append(
            {
                "fold": f,
                "X_train": x[tr],
                "y_train": y[tr],
                "y_train_lgb": y[tr],
                "X_val": x[va],
                "y_val": y[va],
                "y_val_lgb": y[va],
                "dates": dates[va],
                "entities": entities[va],
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
            }
        )
    return folds


CONFIG = {
    "config_name": "test_repro",
    "family": "gbm",
    "max_iterations": 40,
    "checkpoint_interval": 20,
    "params": {
        "objective": "regression_l1",
        "num_leaves": 7,
        "learning_rate": 0.1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
    },
}

FEATURES = [f"f{i}" for i in range(N_FEATURES)]


def _train(device: str, max_bin: int) -> tuple[float, tuple[float, ...]]:
    res = train_gbm_config(
        dict(CONFIG),
        _fold_data(),
        feature_names=FEATURES,
        device=device,
        max_bin=max_bin,
        entity_col="symbol",
        date_col="date",
        task_type="regression",
        save_dir=None,
    )
    return res["best_ic"], tuple(c["ic_mean"] for c in res["learning_curves"])


def test_cpu_gbm_training_is_bit_reproducible() -> None:
    """Same config, same seed, same data -> identical model. The claim the book makes."""
    a_ic, a_curve = _train("cpu", GBM_DEFAULT_MAX_BIN)
    b_ic, b_curve = _train("cpu", GBM_DEFAULT_MAX_BIN)
    assert a_ic == b_ic, f"CPU GBM is not reproducible: {a_ic!r} != {b_ic!r}"
    assert a_curve == b_curve, "CPU GBM learning curves diverge across identical runs"


def test_max_bin_changes_the_model() -> None:
    """max_bin is not cosmetic. It must be declared and hashed, never inherited.

    Every published number was produced with max_bin=63 (recorded inside the saved
    boosters). Swapping to LightGBM's 255 default silently moves every result.
    """
    ic_63, _ = _train("cpu", 63)
    ic_255, _ = _train("cpu", 255)
    assert ic_63 != ic_255, "max_bin should change the fitted model; a guard has gone stale"


# --- device resolution -------------------------------------------------------


def test_device_defaults_to_cpu_and_max_bin_to_the_published_value() -> None:
    assert resolve_gbm_device({}) == ("cpu", GBM_DEFAULT_MAX_BIN)
    assert resolve_gbm_device({"modeling": {"gbm": {}}}) == ("cpu", GBM_DEFAULT_MAX_BIN)


def test_max_bin_is_configuration_not_a_function_of_device() -> None:
    """Regression: max_bin was inherited from whichever device branch ran.

    It must come from setup.yaml and survive a device change untouched, so that
    switching device can never silently move a published number.
    """
    assert resolve_gbm_device({"modeling": {"gbm": {"device": "cpu", "max_bin": 63}}}) == (
        "cpu",
        63,
    )
    assert resolve_gbm_device({"modeling": {"gbm": {"device": "cpu", "max_bin": 255}}}) == (
        "cpu",
        255,
    )
    setup = {
        "modeling": {
            "gbm": {"device": "cuda", "max_bin": 63, "allow_nondeterministic_device": True}
        }
    }
    # falling back to CPU must not rewrite max_bin
    assert resolve_gbm_device(setup, cuda_available=False) == ("cpu", 63)
    assert resolve_gbm_device(setup, cuda_available=True) == ("cuda", 63)


@pytest.mark.parametrize("device", ["cuda", "gpu"])
def test_nondeterministic_device_requires_explicit_opt_in(device: str) -> None:
    with pytest.raises(ValueError, match="not reproducible"):
        resolve_gbm_device({"modeling": {"gbm": {"device": device}}})


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown gbm device"):
        resolve_gbm_device({"modeling": {"gbm": {"device": "tpu"}}})


# --- the guard that would have caught this -----------------------------------

SETUPS = sorted((REPO_ROOT / "case_studies").glob("*/config/setup.yaml"))


@pytest.mark.parametrize("setup_path", SETUPS, ids=lambda p: p.parts[-3])
def test_no_case_study_registers_on_a_nondeterministic_device(setup_path) -> None:
    """Registered numbers are published and drive config selection. They must be replayable."""
    setup = yaml.safe_load(setup_path.read_text())
    device, max_bin = resolve_gbm_device(setup, cuda_available=True)
    assert device == "cpu", (
        f"{setup_path.parts[-3]} registers GBM runs on {device!r}, which is not "
        "bit-reproducible. Set modeling.gbm.device: cpu."
    )
    assert max_bin == GBM_DEFAULT_MAX_BIN, (
        f"{setup_path.parts[-3]} declares max_bin={max_bin}; every published number was "
        f"produced with {GBM_DEFAULT_MAX_BIN}. Changing it moves the results."
    )


# --- the device must enter the training hash ---------------------------------
#
# Every parameter that determines the fitted model belongs in the hash. `device`
# determines it (CUDA vs CPU LightGBM fit different trees) and was absent, so a CPU
# run and a CUDA run of one config collided to a single training_hash — and the
# registry's completeness check would accept either as satisfying the other.


def _spec(**kw):
    return build_training_spec("gbm", "leaves_15_mae", "fwd_ret_5d", n_folds=5, max_bin=255, **kw)


def test_device_is_recorded_in_the_training_spec() -> None:
    assert _spec(device="cpu")["device"] == "cpu"
    assert _spec(device="CUDA")["device"] == "cuda", "device must be normalised"


def test_training_hash_differs_when_device_changes() -> None:
    """The regression: cpu and cuda runs of one config must not share a training_hash."""
    cpu = training_hash_from_spec(_spec(device="cpu"))
    cuda = training_hash_from_spec(_spec(device="cuda"))
    assert cpu != cuda, "cpu and cuda runs collide to one training_hash"


def test_training_hash_is_stable_for_the_same_device() -> None:
    assert training_hash_from_spec(_spec(device="cpu")) == training_hash_from_spec(
        _spec(device="cpu")
    )


def test_omitting_device_leaves_legacy_hashes_untouched() -> None:
    """Callers not yet threaded through must not have their hashes silently move."""
    assert "device" not in _spec()
