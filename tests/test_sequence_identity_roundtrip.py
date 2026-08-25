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

These tests pin the drift itself, so the next identity-bearing field fails here rather than in a
production notebook run.
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


def test_hand_rolling_the_torch_lookup_spec_does_not_reach_the_registration() -> None:
    """The exact shape ``etfs/09_dl_lstm`` built, and why it could never match.

    This is not a test that the fields differ - it is a test that a caller assembling the
    documented field list by hand lands on a different identity than the builder does, which is
    what makes transcription unsafe rather than merely repetitive.
    """
    hand_rolled = {
        "n_epochs": 20,
        "batch_size": TORCH_CONFIG["batch_size"],
        "input_data_spec": INPUT_DATA_SPEC,
        "lookback": TORCH_CONFIG["params"]["lookback"],
        "max_train_sequences": 0,
    }

    assert hand_rolled != _canonical(TORCH_CONFIG)
    assert "device" not in hand_rolled


def test_calling_the_darts_sub_builder_directly_does_not_reach_the_registration() -> None:
    """The shape ``etfs/10_dl_tsmixer`` built.

    ``darts_training_identity`` is a component of the identity, not the identity. Calling it
    directly returns a spec the registration never uses.
    """
    from case_studies.utils.darts_forecasting import darts_training_identity

    sub_builder_only = darts_training_identity(
        DARTS_CONFIG,
        "fwd_ret_21d",
        case_study="etfs",
        input_data_spec=INPUT_DATA_SPEC,
        max_train_sequences=0,
    )

    assert sub_builder_only != _canonical(DARTS_CONFIG)


@pytest.mark.parametrize("config", [TORCH_CONFIG, DARTS_CONFIG], ids=["torch", "darts"])
def test_the_builder_is_the_only_thing_that_reproduces_itself(config: dict[str, Any]) -> None:
    """Called twice with the same arguments it agrees with itself, on both backends.

    Without this the tests above could pass on a builder that was simply unstable, which would
    break every lookup rather than only the hand-rolled ones.
    """
    assert _canonical(config) == _canonical(config)
    assert "device" in _canonical(config)
