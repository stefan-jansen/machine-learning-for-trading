"""The TabM runner's declared version, and the fitted result it stands for.

``TABM_RUNNER_VERSION`` sits in every TabM training identity in place of a SHA-256 of
``tabular_dl.py``. The digest was unworkable - it made a comment, a log line, or threading a
provenance field through invalidate all 46 registered TabM rows across six case studies - but a
declared version is only worth what checks it, which is this file.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from case_studies.utils.tabular_dl import (
    DEEP_MODEL_STATE_VERSION,
    TABM_RUNNER_VERSION,
    TabMModel,
    _tabm_source_identity,
    _train_tabm_fold,
)
from utils.modeling import seed_everything

PINNED_RUNNER_VERSION = 1
PINNED_DEEP_MODEL_STATE_VERSION = 1
PINNED_FORWARD = "37bfa3b1fd93489f"
PINNED_CHECKPOINT_PREDICTIONS = "e6a61ccc61c82234"

N_FEATURES = 6
HIDDEN_DIM = 8
N_MEMBERS = 3


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()[:16]


@pytest.fixture
def fold() -> dict[str, np.ndarray]:
    """A small deterministic regression fold. Shape matters only in that it stays fixed."""
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((120, N_FEATURES))
    x_val = rng.standard_normal((40, N_FEATURES))
    weights = rng.standard_normal(N_FEATURES)
    return {
        "X_train": x_train,
        "y_train": x_train @ weights + 0.1 * rng.standard_normal(120),
        "X_val": x_val,
        "y_val": x_val @ weights + 0.1 * rng.standard_normal(40),
        # Two validation dates, so the per-date IC the runner computes is defined.
        "val_dates": np.repeat(np.array([0, 1]), 20),
    }


class TestWhatEntersTheIdentity:
    def test_the_identity_declares_versions_rather_than_source_digests(self) -> None:
        """A 64-character hex string here means the source-hashing scheme came back."""
        identity = _tabm_source_identity()

        assert identity["tabm_runner"] == TABM_RUNNER_VERSION
        assert not any(isinstance(value, str) and len(value) == 64 for value in identity.values())

    def test_it_covers_the_runner_and_the_state_a_checkpoint_restores(self) -> None:
        """A prediction is read back from persisted state, so that state is part of the result."""
        assert {"tabm_runner", "deep_model_state"} == set(_tabm_source_identity())

    def test_it_does_not_depend_on_where_the_file_lives(self) -> None:
        """The digest scheme made the identity a property of the checkout, not of the behaviour."""
        assert _tabm_source_identity() == _tabm_source_identity()


class TestTheDeclaredVersion:
    """If a pin below moves, the TabM runner produces different results than every row registered
    under the current version claims. Bump ``TABM_RUNNER_VERSION`` in
    ``case_studies/utils/tabular_dl.py`` and update the pin in the same commit."""

    def test_the_declared_versions_match_what_this_file_pins(self) -> None:
        assert TABM_RUNNER_VERSION == PINNED_RUNNER_VERSION
        assert DEEP_MODEL_STATE_VERSION == PINNED_DEEP_MODEL_STATE_VERSION

    def test_the_architecture_reproduces_its_pinned_forward_pass(self) -> None:
        """Covers the backbone, the rank-1 adapters and the per-member heads."""
        seed_everything(42)
        model = TabMModel(
            n_features=N_FEATURES, hidden_dim=HIDDEN_DIM, n_members=N_MEMBERS, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.arange(2 * N_FEATURES, dtype=torch.float32).reshape(2, N_FEATURES))

        assert _digest(output.numpy()) == PINNED_FORWARD, (
            "the TabM architecture now computes a different forward pass; bump "
            "TABM_RUNNER_VERSION in case_studies/utils/tabular_dl.py and update this pin"
        )

    def test_a_fixed_fold_reproduces_its_pinned_checkpoint_predictions(self, fold) -> None:
        """Covers the training loop itself - optimizer, loss, and the checkpoint grid."""
        seed_everything(42)
        model = TabMModel(
            n_features=N_FEATURES, hidden_dim=HIDDEN_DIM, n_members=N_MEMBERS, dropout=0.0
        )
        _, predictions, _ = _train_tabm_fold(
            model,
            fold["X_train"],
            fold["y_train"],
            fold["X_val"],
            fold["y_val"],
            fold["y_val"],
            fold["val_dates"],
            None,
            n_epochs=4,
            batch_size=32,
            checkpoint_interval=2,
            device=torch.device("cpu"),
        )

        assert sorted(predictions) == [2, 4]
        stacked = np.concatenate([predictions[epoch] for epoch in sorted(predictions)])
        assert _digest(stacked) == PINNED_CHECKPOINT_PREDICTIONS, (
            "the TabM training loop now fits a different result; bump TABM_RUNNER_VERSION in "
            "case_studies/utils/tabular_dl.py and update this pin in the same commit"
        )
