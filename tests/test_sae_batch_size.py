"""The SAE fits in minibatches because a runner declares the size, not because a library does.

``SAEConfig.batch_size`` defaults to ``None``, and the training loop reads that as a single
batch holding the entire training window. On a daily equity panel that is roughly a quarter of
a million rows in one gradient step, which exhausts a 24 GB card - so the SAE could not be run
at production width at all. It was never a decision about gradient estimation: ``run_cae_fold``
has always carried ``batch_size``, and ``run_sae_fold``, its sibling, was the one member of the
pair that passed nothing.

These pin the value reaching the model rather than the presence of a keyword, because a
parameter that is accepted and dropped on the floor looks identical from the call site.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from case_studies.utils.latent_factors import sae
from case_studies.utils.latent_factors.library_bridge import SAEConfig


def test_the_runner_declares_a_batch_size_rather_than_inheriting_none() -> None:
    """Both members of the pair take the parameter, and neither leaves it to the library."""
    sae_default = inspect.signature(sae.run_sae_fold).parameters["batch_size"].default
    assert isinstance(sae_default, int) and sae_default > 0

    from case_studies.utils.latent_factors import cae

    cae_default = inspect.signature(cae.run_cae_fold).parameters["batch_size"].default
    assert sae_default == cae_default, "the pair should agree unless a case study overrides"

    # The library's own default is the thing being overridden; if it ever stops being None
    # this test should be reconsidered rather than silently kept.
    assert SAEConfig.batch_size is None or SAEConfig().batch_size is None


def test_the_declared_batch_size_reaches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parameter accepted and never forwarded looks the same from the call site."""
    from case_studies.utils.latent_factors import library_bridge

    seen: dict = {}

    class _Spy:
        def __init__(self, config):
            seen["batch_size"] = config.batch_size

    monkeypatch.setattr(library_bridge, "SAEModel", _Spy)
    monkeypatch.setattr(
        library_bridge,
        "_run_checkpointed_signal_pipeline",
        lambda **kw: {"checkpoint_predictions": {}},
    )

    rng = np.random.default_rng(0)
    chars = rng.normal(size=(4, 6, 3)).astype(np.float32)
    rets = rng.normal(size=(4, 6)).astype(np.float32)

    # Through `run_sae_fold`, the runner the adapter calls - not through the bridge directly.
    # The regression this file exists for is the runner failing to forward the value, one hop
    # above the bridge, so a test that starts at the bridge cannot see it.
    sae.run_sae_fold(chars, rets, chars, rets, 5, batch_size=4096)

    assert seen["batch_size"] == 4096
