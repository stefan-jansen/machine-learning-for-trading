"""Semantic implementation versions for model families that used to hash source files."""

# ruff: noqa: E402  # pytest.importorskip must run before the torch-backed imports

from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from case_studies.utils import causal, deep_learning, tabular_dl
from case_studies.utils.latent_factors import adapter as latent_adapter
from case_studies.utils.registry import training_hash_from_spec

PINNED_SEQUENCE_RUNNER = 1
PINNED_SEQUENCE_PREPARATION = 1
PINNED_SEQUENCE_STATE = 1
PINNED_TABM_RUNNER = 1
PINNED_TABM_STATE = 1
PINNED_LATENT_ADAPTER = 1
PINNED_CAUSAL_RUNNER = 1


def _training_spec(source_identity: dict) -> dict:
    return {
        "identity_version": 3,
        "resolved_spec_schema": "ml4t.resolved-spec/v1",
        "execution_tier": "canonical",
        "family": "fixture",
        "label": "target",
        "seed": 42,
        "computation": {"source_identity": source_identity},
        "provenance": {},
    }


def test_sequence_identity_is_declared_and_architecture_scoped() -> None:
    assert deep_learning.SEQUENCE_RUNNER_VERSION == PINNED_SEQUENCE_RUNNER
    assert deep_learning.SEQUENCE_PREPARATION_VERSION == PINNED_SEQUENCE_PREPARATION
    assert deep_learning.SEQUENCE_STATE_VERSION == PINNED_SEQUENCE_STATE

    nlinear = deep_learning._sequence_source_identity(
        {"library": "pytorch", "params": {"architecture": "nlinear"}}
    )
    lstm = deep_learning._sequence_source_identity(
        {"library": "pytorch", "params": {"architecture": "lstm"}}
    )
    darts = deep_learning._sequence_source_identity(
        {"library": "darts", "params": {"architecture": "tsmixer"}}
    )

    assert nlinear == {
        "sequence_runner": 1,
        "sequence_preparation": 1,
        "sequence_state": 1,
        "backend": "pytorch/v1",
        "architecture": "nlinear/v1",
    }
    assert nlinear != lstm
    assert nlinear != darts


def test_sequence_identity_rejects_an_unversioned_architecture() -> None:
    with pytest.raises(ValueError, match="implementation version"):
        deep_learning._sequence_source_identity(
            {"library": "pytorch", "params": {"architecture": "unknown"}}
        )


def test_tabm_identity_is_declared() -> None:
    assert tabular_dl.TABM_RUNNER_VERSION == PINNED_TABM_RUNNER
    assert tabular_dl.TABM_STATE_VERSION == PINNED_TABM_STATE
    assert tabular_dl._tabm_source_identity() == {
        "tabm_runner": 1,
        "tabm_state": 1,
    }


@pytest.mark.parametrize("model", ["pca", "ipca", "cae", "sae", "sdf"])
def test_latent_identity_is_declared_and_model_scoped(model: str) -> None:
    assert latent_adapter.LATENT_ADAPTER_VERSION == PINNED_LATENT_ADAPTER
    assert latent_adapter._source_identity(model) == {
        "latent_adapter": 1,
        "latent_model": f"{model}/v1",
    }


def test_causal_identity_is_declared() -> None:
    assert causal.CAUSAL_RUNNER_VERSION == PINNED_CAUSAL_RUNNER
    assert causal._causal_source_identity() == {"causal_runner": 1}


@pytest.mark.parametrize(
    "identity",
    [
        lambda: deep_learning._sequence_source_identity(
            {"library": "pytorch", "params": {"architecture": "nlinear"}}
        ),
        tabular_dl._tabm_source_identity,
        lambda: latent_adapter._source_identity("pca"),
        causal._causal_source_identity,
    ],
)
def test_a_declared_version_change_moves_the_training_hash(identity) -> None:
    original = identity()
    changed = deepcopy(original)
    first = next(iter(changed))
    changed[first] = int(changed[first]) + 1

    assert training_hash_from_spec(_training_spec(original)) != training_hash_from_spec(
        _training_spec(changed)
    )


def test_active_source_identities_contain_no_file_digests() -> None:
    identities = [
        deep_learning._sequence_source_identity(
            {"library": "pytorch", "params": {"architecture": "nlinear"}}
        ),
        tabular_dl._tabm_source_identity(),
        latent_adapter._source_identity("pca"),
        causal._causal_source_identity(),
    ]
    assert not any(
        isinstance(value, str) and len(value) == 64
        for identity in identities
        for value in identity.values()
    )
