"""The latent-factor runners' declared versions, and the fitted results they stand for.

``LATENT_RUNNER_VERSION`` and the five per-model versions sit in every latent training identity in
place of a SHA-256 of fourteen files. The digest was unworkable twice over: a comment in
``panel.py`` invalidated every registered latent row, and it coupled the models to each other, so
an SAE change refit IPCA. A declared version is only worth what checks it, which is this file.

PCA, IPCA, CAE and SAE are pinned against a fitted result below. SDF is not: a fit is not cheap
enough for a unit test at the time of writing, so its version is declared and dispatched but its
behaviour is unpinned, and a silent change to it would not fail here.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from case_studies.utils.latent_factors.adapter import LATENT_RUNNER_VERSION, _source_identity
from case_studies.utils.latent_factors.cae import CAE_RUNNER_VERSION, run_cae_fold
from case_studies.utils.latent_factors.cv import (
    _MODEL_RUNNERS,
    _MODEL_VERSIONS,
    latent_model_version,
)
from case_studies.utils.latent_factors.ipca import IPCA_RUNNER_VERSION, run_ipca_fold
from case_studies.utils.latent_factors.pca import PCA_RUNNER_VERSION, run_pca_fold
from case_studies.utils.latent_factors.sae import SAE_RUNNER_VERSION, run_sae_fold
from case_studies.utils.latent_factors.sdf import SDF_RUNNER_VERSION

PINNED_VERSIONS = {
    "latent_runner": 1,
    "pca": 1,
    "ipca": 1,
    "cae": 1,
    "sae": 2,
    "sdf": 1,
}
PINNED_PCA_PREDICTIONS = "4d62f4ceefc58141"
PINNED_IPCA_PREDICTIONS = "0be530a116515880"
PINNED_CAE_PREDICTIONS = "82a8721e8c074459"
PINNED_SAE_PREDICTIONS = "01594687acc5416c"
SAE_BATCH = 64

N_DATES, N_ASSETS, N_CHARS, N_FACTORS = 24, 12, 4, 2


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()[:16]


@pytest.fixture
def panel() -> dict[str, np.ndarray]:
    """A small balanced panel. Its shape matters only in that it stays fixed."""
    rng = np.random.default_rng(0)
    loadings = rng.standard_normal((N_CHARS, N_FACTORS))

    def block(n_dates: int) -> tuple[np.ndarray, np.ndarray]:
        chars = rng.standard_normal((n_dates, N_ASSETS, N_CHARS))
        factors = rng.standard_normal((n_dates, N_FACTORS))
        returns = np.einsum("tnc,cf,tf->tn", chars, loadings, factors)
        return chars, returns + 0.1 * rng.standard_normal((n_dates, N_ASSETS))

    chars_train, returns_train = block(N_DATES)
    chars_val, returns_val = block(N_DATES // 2)
    return {
        "chars_train": chars_train,
        "returns_train": returns_train,
        "chars_val": chars_val,
        "returns_val": returns_val,
    }


class TestWhatEntersTheIdentity:
    def test_the_identity_declares_versions_rather_than_source_digests(self) -> None:
        """A 64-character hex string here means the source-hashing scheme came back."""
        identity = _source_identity("pca")

        assert identity["latent_runner"] == LATENT_RUNNER_VERSION
        assert not any(isinstance(value, str) and len(value) == 64 for value in identity.values())

    def test_it_covers_the_shared_machinery_and_the_one_model_being_fitted(self) -> None:
        assert set(_source_identity("ipca")) == {"latent_runner", "latent_model"}

    def test_every_model_the_runner_dispatches_to_declares_a_version(self) -> None:
        """A model reachable through `_MODEL_RUNNERS` with no declared version would fit under an
        identity that describes nothing about it."""
        for model_name in _MODEL_RUNNERS:
            assert isinstance(latent_model_version(model_name), int)

    def test_an_undeclared_model_is_refused_rather_than_given_a_default(self) -> None:
        with pytest.raises(KeyError):
            latent_model_version("no_such_model")

    def test_bumping_one_models_version_leaves_the_others_untouched(self, monkeypatch) -> None:
        """The defect this replaces: one digest over fourteen files meant an edit to sae.py moved
        every IPCA identity too. Two models sharing a version number is not the point - what has
        to hold is that changing one of them changes only its own identities."""
        before = {name: _source_identity(name) for name in _MODEL_RUNNERS}
        monkeypatch.setitem(_MODEL_VERSIONS, "sae", SAE_RUNNER_VERSION + 1)
        after = {name: _source_identity(name) for name in _MODEL_RUNNERS}

        assert after["sae"] != before["sae"]
        assert {k: v for k, v in after.items() if k != "sae"} == {
            k: v for k, v in before.items() if k != "sae"
        }


class TestTheDeclaredVersions:
    """If a pin below moves, that runner produces different results than every row registered
    under its current version claims. Bump the version constant in the model's own module and
    update the pin in the same commit."""

    def test_the_declared_versions_match_what_this_file_pins(self) -> None:
        assert PINNED_VERSIONS["latent_runner"] == LATENT_RUNNER_VERSION
        assert PINNED_VERSIONS["pca"] == PCA_RUNNER_VERSION
        assert PINNED_VERSIONS["ipca"] == IPCA_RUNNER_VERSION
        assert PINNED_VERSIONS["cae"] == CAE_RUNNER_VERSION
        assert PINNED_VERSIONS["sae"] == SAE_RUNNER_VERSION
        assert PINNED_VERSIONS["sdf"] == SDF_RUNNER_VERSION

    def test_pca_reproduces_its_pinned_predictions(self, panel) -> None:
        predictions, _ = run_pca_fold(
            panel["chars_train"],
            panel["returns_train"],
            panel["chars_val"],
            panel["returns_val"],
            N_FACTORS,
        )

        assert _digest(np.asarray(predictions)) == PINNED_PCA_PREDICTIONS, (
            "the PCA runner now fits a different result; bump PCA_RUNNER_VERSION in "
            "case_studies/utils/latent_factors/pca.py and update this pin in the same commit"
        )

    def test_ipca_reproduces_its_pinned_predictions(self, panel) -> None:
        predictions, _ = run_ipca_fold(
            panel["chars_train"],
            panel["returns_train"],
            panel["chars_val"],
            panel["returns_val"],
            N_FACTORS,
            max_iter=20,
        )

        assert _digest(np.asarray(predictions)) == PINNED_IPCA_PREDICTIONS, (
            "the IPCA runner now fits a different result; bump IPCA_RUNNER_VERSION in "
            "case_studies/utils/latent_factors/ipca.py and update this pin in the same commit"
        )

    def test_cae_reproduces_its_pinned_predictions(self, panel) -> None:
        """The neural runners reach the library through `library_bridge`, so this covers the
        bridge's keyword arguments as well as the model - which is where the SAE lost its
        `batch_size` and trained on the whole panel in one batch."""
        checkpoints, _ = run_cae_fold(
            panel["chars_train"],
            panel["returns_train"],
            panel["chars_val"],
            panel["returns_val"],
            N_FACTORS,
            n_epochs=2,
            checkpoint_interval=1,
            device="cpu",
            seed=42,
        )

        stacked = np.concatenate([np.asarray(checkpoints[e]) for e in sorted(checkpoints)])
        assert _digest(stacked) == PINNED_CAE_PREDICTIONS, (
            "the CAE runner now fits a different result; bump CAE_RUNNER_VERSION in "
            "case_studies/utils/latent_factors/cae.py and update this pin in the same commit"
        )

    def test_sae_reproduces_its_pinned_predictions(self, panel) -> None:
        checkpoints, _ = run_sae_fold(
            panel["chars_train"],
            panel["returns_train"],
            panel["chars_val"],
            panel["returns_val"],
            N_FACTORS,
            n_epochs=2,
            checkpoint_interval=1,
            batch_size=SAE_BATCH,
            device="cpu",
            seed=42,
        )

        stacked = np.concatenate([np.asarray(checkpoints[e]) for e in sorted(checkpoints)])
        assert _digest(stacked) == PINNED_SAE_PREDICTIONS, (
            "the SAE runner now fits a different result; bump SAE_RUNNER_VERSION in "
            "case_studies/utils/latent_factors/sae.py and update this pin in the same commit"
        )

    def test_the_sae_batch_size_reaches_the_library(self, panel) -> None:
        """The defect behind `SAE_RUNNER_VERSION = 2`: `run_sae_fold_with_library` built its
        `SAEConfig` without `batch_size`, so it took the library default of `None` and trained on
        the whole panel in one batch - 21.72 GiB of allocations and an OOM on a 24 GB card at one
        fold. A pin alone would not catch a re-omission, because this panel is smaller than the
        default batch and would train identically either way. Two batch sizes that must disagree
        is what actually holds the keyword in place."""

        def fit(batch_size: int) -> np.ndarray:
            checkpoints, _ = run_sae_fold(
                panel["chars_train"],
                panel["returns_train"],
                panel["chars_val"],
                panel["returns_val"],
                N_FACTORS,
                n_epochs=2,
                checkpoint_interval=1,
                batch_size=batch_size,
                device="cpu",
                seed=42,
            )
            return np.concatenate([np.asarray(checkpoints[e]) for e in sorted(checkpoints)])

        minibatched = fit(SAE_BATCH)
        full_panel = fit(10 * N_DATES * N_ASSETS)

        assert not np.allclose(minibatched, full_panel), (
            "minibatched and full-batch SAE fits agree, so `batch_size` is not reaching the "
            "library and the OOM this guards against is back"
        )
