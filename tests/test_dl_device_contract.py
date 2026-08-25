"""What decides which device the sequence families train on, and what is recorded.

The device is a spec-hash input for a sequence run (``deep_learning.py``'s runtime spec),
so it cannot be inferred from whatever hardware happens to be present: a run that asked
for a GPU and silently retrained on CPU would register under an identity it did not have.
The other half of that rule is that the requirement has to be written down somewhere. A
notebook that defaults to ``gpu`` when its case study declares no ``modeling.dl`` section
imposes a hardware requirement nobody chose, which is how ``cs-nasdaq100_microstructure``
went red on a CPU runner while the config said nothing about a device at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from case_studies.utils.deep_learning import resolve_dl_device
from tests.pm_helpers import get_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]

NASDAQ_DL_NOTEBOOKS = ("08_dl_nlinear", "09_dl_lstm", "10_dl_tcn", "11_dl_patchtst")


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_an_undeclared_device_is_refused_rather_than_defaulted_to_gpu() -> None:
    with pytest.raises(ValueError, match="modeling.dl.device must be declared"):
        resolve_dl_device(None)
    with pytest.raises(ValueError, match="modeling.dl.device must be declared"):
        resolve_dl_device({})


def test_a_declared_gpu_with_no_cuda_present_stops_the_run(no_cuda: None) -> None:
    with pytest.raises(RuntimeError, match="modeling.dl.device requests a GPU"):
        resolve_dl_device({"device": "gpu"})


def test_an_explicit_cpu_request_overrides_the_declared_gpu(no_cuda: None) -> None:
    assert resolve_dl_device({"device": "gpu"}, "cpu") == "cpu"
    assert resolve_dl_device({"device": "cpu"}) == "cpu"


def test_an_override_asking_for_an_absent_gpu_stops_the_run(no_cuda: None) -> None:
    with pytest.raises(RuntimeError, match="the DEVICE override requests a GPU"):
        resolve_dl_device({"device": "cpu"}, "gpu")


def test_an_unrecognised_device_names_where_it_came_from() -> None:
    with pytest.raises(ValueError, match="unsupported sequence device 'mps'"):
        resolve_dl_device({"device": "mps"})
    with pytest.raises(ValueError, match=r"unsupported sequence device 'tpu' \(from the DEVICE"):
        resolve_dl_device({"device": "cpu"}, "tpu")


def _declared_dl_config(case_study: str) -> dict | None:
    setup_path = REPO_ROOT / "case_studies" / case_study / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text()) or {}
    return (setup.get("modeling") or {}).get("dl")


def test_nasdaq_declares_the_gpu_its_production_runs_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No override, so this reads the declaration and nothing else.

    CUDA is forced present because the assertion is about what the case study asks for,
    not about the hardware under the test runner.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_dl_device(_declared_dl_config("nasdaq100_microstructure")) == "cuda"


@pytest.mark.parametrize("notebook", NASDAQ_DL_NOTEBOOKS)
def test_every_nasdaq_dl_notebook_runs_as_the_ci_harness_configures_it(
    notebook: str, no_cuda: None
) -> None:
    """On a CPU runner each notebook either resolves to CPU or declares it needs a GPU.

    ``gpu: true`` makes the harness skip the notebook where no CUDA device is visible, so
    that is the other admissible answer. What is not admissible is a notebook the harness
    runs on a CPU runner that then refuses the device it was never told to want.
    """
    overrides = get_overrides(f"case_studies/nasdaq100_microstructure/{notebook}")
    declared = _declared_dl_config("nasdaq100_microstructure")
    requested = (overrides.get("parameters") or {}).get("DEVICE", "")

    if overrides.get("gpu"):
        pytest.skip(f"{notebook} declares gpu: true and is skipped where CUDA is absent")
    assert resolve_dl_device(declared, requested) == "cpu"
