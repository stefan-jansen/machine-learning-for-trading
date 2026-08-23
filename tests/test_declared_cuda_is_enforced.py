"""The device a spec declares must be the device the fit refuses to run without.

Resolving a model request validates that ``device`` is one of ``cpu`` or ``cuda`` and
deliberately does NOT check availability, so a CPU machine can resolve a portable spec that
declares ``cuda``. ``tests/test_latent_factors_no_leak.py`` pins that half. Device is
identity-bearing - it reaches ``params["runtime"]["device"]`` and moves the training hash -
so the spec has to mean something at execution: a fit under a spec saying ``cuda`` must
refuse on a machine with no CUDA rather than quietly produce CPU numbers under a hash that
claims otherwise.

Three guards enforce that, one per family, and none of them had a test. The intended-behavior
half was covered and the enforcement half was not, which is the shape that lets a guard be
deleted or inverted without anything going red.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report no CUDA regardless of what this machine actually has.

    Patched at ``torch.cuda.is_available`` rather than through ``CUDA_VISIBLE_DEVICES``
    because the developer box has an RTX 3090 and the CI runners have none, so the test must
    assert the same thing on both.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_sequence_runtime_refuses_a_cuda_spec_without_cuda(no_cuda: None) -> None:
    from case_studies.utils.deep_learning import _sequence_runtime_spec

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _sequence_runtime_spec("cuda", seed=42, num_threads=1)

    # "gpu" normalizes to "cuda" before the check, so the alias cannot slip past it.
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _sequence_runtime_spec("gpu", seed=42, num_threads=1)

    # A cpu spec is unaffected: the guard is about a declaration the machine cannot honour.
    assert _sequence_runtime_spec("cpu", seed=42, num_threads=1)["device"] == "cpu"


def test_the_latent_and_deep_learning_guards_are_present_and_ordered(no_cuda: None) -> None:
    """The other two guards sit too deep in their call paths to reach without a fitted study.

    So this pins what can be pinned without one: that each still exists, and that it runs
    before the device is used rather than after. Reaching them properly needs
    ``reconstruct_locked_request`` driven to its runtime block, which needs a writable study,
    a canonical activation and a loaded case-study context - worth building, and not from a
    session that cannot run a fit.
    """
    import inspect

    from case_studies.utils import deep_learning
    from case_studies.utils.latent_factors import adapter

    latent = inspect.getsource(adapter.reconstruct_locked_request)
    guard = 'if device == "cuda" and not torch.cuda.is_available():'
    assert guard in latent, (
        "the locked latent-factor runtime no longer refuses an unavailable CUDA; if the check "
        "moved, move this assertion to wherever it now lives"
    )
    assert latent.index(guard) < latent.index("case.device = device"), (
        "the latent guard now runs after the device is assigned, so a spec declaring cuda "
        "would be applied before being checked"
    )

    source = inspect.getsource(deep_learning)
    assert 'if device.startswith("cuda") and not torch.cuda.is_available():' in source, (
        "the deep-learning fit no longer refuses an unavailable CUDA"
    )
