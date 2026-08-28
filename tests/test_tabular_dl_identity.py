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
    TABM_RUNNER_VERSION,
    TABM_STATE_VERSION,
    TabMModel,
    _tabm_source_identity,
    _train_tabm_fold,
)
from utils.modeling import seed_everything

PINNED_RUNNER_VERSION = 1
PINNED_TABM_STATE_VERSION = 1

# The architecture, as the exact integers a build cannot change: every parameter tensor
# by name and shape, and the total parameter count.
#
# This replaced a SHA-256 of the forward pass. That digest passed locally and failed in
# `test-unit-image`, because the pinned bytes are floating-point output and the two
# environments run different torch builds - a CUDA build here, a CPU-only build in
# `ml4t/ml4t:latest` - which select different kernels for the same seeded init and give
# results that differ in the last bits. A pin that only holds on the machine it was
# taken on does not guard the architecture; it reports the environment.
#
# What the float digest was standing in for is here instead, and none of it is
# build-dependent: an added layer, a changed width, a renamed module or a different
# member count all move these, and a torch rebuild does not.
#
# Written out as literals, not derived from `TabMModel`. Building the model to produce
# the expected value and then comparing it against itself is a tautology that passes
# whatever the architecture becomes, which is the failure this file exists to prevent.
PINNED_PARAMETER_SHAPES = {
    "adapters": (3, 8),
    "backbone.0.weight": (8, 6),
    "backbone.0.bias": (8,),
    "backbone.3.weight": (8, 8),
    "backbone.3.bias": (8,),
    "heads.0.weight": (1, 8),
    "heads.0.bias": (1,),
    "heads.1.weight": (1, 8),
    "heads.1.bias": (1,),
    "heads.2.weight": (1, 8),
    "heads.2.bias": (1,),
}
PINNED_PARAMETER_COUNT = 179

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
        assert {"tabm_runner", "tabm_state"} == set(_tabm_source_identity())

    def test_it_does_not_depend_on_where_the_file_lives(self) -> None:
        """The digest scheme made the identity a property of the checkout, not of the behaviour."""
        assert _tabm_source_identity() == _tabm_source_identity()


class TestTheDeclaredVersion:
    """If a pin below moves, the TabM runner produces different results than every row registered
    under the current version claims. Bump ``TABM_RUNNER_VERSION`` in
    ``case_studies/utils/tabular_dl.py`` and update the pin in the same commit."""

    def test_the_declared_versions_match_what_this_file_pins(self) -> None:
        assert TABM_RUNNER_VERSION == PINNED_RUNNER_VERSION
        assert TABM_STATE_VERSION == PINNED_TABM_STATE_VERSION

    def test_the_architecture_keeps_its_declared_shape(self) -> None:
        """Covers the backbone, the rank-1 adapters and the per-member heads.

        Shapes and a parameter count rather than a digest of the forward pass: the
        digest is floating point and differs between torch builds, so it failed in the
        modelling image while passing here. These are integers and move only when the
        architecture does.
        """
        model = TabMModel(
            n_features=N_FEATURES, hidden_dim=HIDDEN_DIM, n_members=N_MEMBERS, dropout=0.0
        )
        shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}

        assert shapes == PINNED_PARAMETER_SHAPES, (
            "the TabM architecture declares different parameters; bump "
            "TABM_RUNNER_VERSION in case_studies/utils/tabular_dl.py and update this pin"
        )
        assert sum(t.numel() for t in model.state_dict().values()) == PINNED_PARAMETER_COUNT

    def test_the_forward_pass_is_reproducible_under_a_fixed_seed(self) -> None:
        """What the pinned digest could actually promise across environments.

        Two constructions under the same seed, in one process against one build, must
        agree bit for bit. That is the property the runner's declared version stands
        for - a fit is reproducible - and unlike a stored digest it holds wherever the
        test runs. It fails on an unseeded initialisation or a nondeterministic kernel.
        """
        inputs = torch.arange(2 * N_FEATURES, dtype=torch.float32).reshape(2, N_FEATURES)

        def once() -> np.ndarray:
            seed_everything(42)
            model = TabMModel(
                n_features=N_FEATURES, hidden_dim=HIDDEN_DIM, n_members=N_MEMBERS, dropout=0.0
            )
            model.eval()
            with torch.no_grad():
                return model(inputs).numpy()

        assert _digest(once()) == _digest(once()), (
            "two TabM constructions under seed 42 disagree, so the runner's declared "
            "version does not stand for a reproducible fit"
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

        assert sorted(predictions) == [2, 4], (
            "the checkpoint grid moved: n_epochs=4 at checkpoint_interval=2 publishes "
            "epochs 2 and 4, and a change here changes which checkpoints every "
            "registered TabM row carries"
        )
        first, second = (predictions[epoch] for epoch in sorted(predictions))
        assert first.shape == second.shape == (len(fold["y_val"]),)
        assert np.isfinite(first).all() and np.isfinite(second).all()
        assert _digest(first) != _digest(second), (
            "the two checkpoints predict identically, so four epochs of training moved "
            "nothing - the loop is not fitting"
        )
