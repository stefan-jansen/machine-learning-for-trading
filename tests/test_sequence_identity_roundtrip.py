"""What a notebook looks a sequence fit up with must equal what ``run_dl_cv`` registers it under.

``run_dl_cv`` builds every registration identity through ``sequence_identity_params``. A notebook
that wants to find an existing fit has to build the same identity, and the only safe way to do
that is to call the same function. Transcribing its fields instead works until one is added -
``darts_forecasting.py`` says so directly: "any field in one and not the other means the lookup
can never find the registration".

That happened. PR #620 made ``device`` identity-bearing, and two etfs notebooks hand-rolled their
lookup spec without it. The failure is not a cache miss costing a refit: the config trains,
registers under the device-qualified hash, and the post-training rebuild still finds nothing and
raises "Training completed but registered checkpoints are incomplete".

These tests pin the builder's contract - which fields are identity-bearing and which are not - so
that changing one fails here rather than in a production notebook run.

They deliberately do **not** assert anything about what ``etfs/09_dl_lstm`` and ``10_dl_tsmixer``
currently pass. A test comparing the builder against a transcription of those call sites written
here would compare this file to itself: it stays green whatever the notebooks do, so it cannot
detect the drift it appears to be about. The check that those call sites reach the registration
belongs in the commit that converts them onto ``sequence_identity_params``, where it goes green
because the notebook changed rather than because the literal did.
"""

from __future__ import annotations

from typing import Any

import pytest

from case_studies.utils.deep_learning import sequence_identity_params

INPUT_DATA_SPEC = {"features": "abc123", "labels": "def456"}

TORCH_CONFIG: dict[str, Any] = {
    "family": "deep_learning",
    "config_name": "lstm_h64",
    "architecture": "lstm",
    "batch_size": 512,
    "params": {"lookback": 60, "hidden_size": 64},
}

DARTS_CONFIG: dict[str, Any] = {
    "family": "deep_learning",
    "config_name": "tsmixer",
    "architecture": "tsmixer",
    "params": {"lookback": 60},
}


def _canonical(config: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    params = sequence_identity_params(
        config,
        identity_params={"n_epochs": 20},
        input_data_spec=INPUT_DATA_SPEC,
        label_col="fwd_ret_21d",
        case_study="etfs",
        max_train_sequences=0,
        device=device,
    )
    assert params is not None
    return params


def test_the_device_is_part_of_the_identity() -> None:
    """A CPU fit and a GPU fit of one configuration are not the same fit.

    Different kernels, different reduction orders, a different nondeterminism profile. If they
    shared a hash, a completed CPU run would satisfy the already-complete check that skips a GPU
    fit and the result would be registered under a device it did not have.
    """
    assert _canonical(TORCH_CONFIG, device="cpu") != _canonical(TORCH_CONFIG, device="cuda")
    assert _canonical(TORCH_CONFIG, device="cpu")["device"] == "cpu"
    assert _canonical(TORCH_CONFIG, device="gpu")["device"] == "cuda"


def test_the_device_index_is_not_part_of_the_identity() -> None:
    """``cuda:0`` and ``cuda:1`` make the same claim about what the numbers came from."""
    assert _canonical(TORCH_CONFIG, device="cuda:0") == _canonical(TORCH_CONFIG, device="cuda:1")


@pytest.mark.parametrize("config", [TORCH_CONFIG, DARTS_CONFIG], ids=["torch", "darts"])
def test_the_builder_is_the_only_thing_that_reproduces_itself(config: dict[str, Any]) -> None:
    """Called twice with the same arguments it agrees with itself, on both backends.

    Without this the tests above could pass on a builder that was simply unstable, which would
    break every lookup rather than only the hand-rolled ones.
    """
    assert _canonical(config) == _canonical(config)
    assert "device" in _canonical(config)
