"""A registered training row has to say how much of the machine it used.

`pre_run_gate.py` fails "the run reports how much of the machine it used" when no
row carries `cores_used`, and `cores_used` appears only when `cpu_s` is passed
alongside `elapsed_s` (`utils/runtime.py:122-124`). Wall time alone cannot
recover it: an hour on one core and an hour on twenty are indistinguishable.

Three adapters recorded `elapsed_s` without `cpu_s`, so tabular_dl, deep_learning
and latent_factors rows could never clear that check while linear and gbm always
did. These tests hold the boundary each family's recording helper writes through.
"""

from __future__ import annotations

import pytest

from case_studies.utils import tabular_dl
from case_studies.utils.runtime import resource_measurement


def test_cores_used_needs_cpu_seconds():
    """The gate's input, stated as a contract rather than assumed."""
    assert "cores_used" not in resource_measurement(elapsed_s=10.0)
    measured = resource_measurement(elapsed_s=10.0, cpu_s=40.0)
    assert measured["cores_used"] == pytest.approx(4.0)


def test_a_zero_length_run_reports_no_cores_rather_than_dividing():
    measured = resource_measurement(elapsed_s=0.0, cpu_s=5.0)
    assert measured["cores_used"] == 0.0


def test_tabm_training_runtime_records_cores(monkeypatch):
    """The TabM helper must hand the registry a core count, not just seconds."""
    captured = {}

    def _capture(case_study, training_hash, *, case_dir, measured):
        captured.update(measured)

    monkeypatch.setattr(
        "case_studies.utils.registry.registration.record_training_runtime", _capture
    )

    class _Training:
        hash = "abc123"
        root = None

    class _Study:
        case_study = "etfs"

    tabular_dl._record_tabm_training_runtime(
        _Study(), _Training(), elapsed_s=100.0, cpu_s=350.0, preparation_s=4.0
    )

    assert captured["elapsed_s"] == 100.0
    assert captured["cores_used"] == pytest.approx(3.5)
    assert captured["fold_preparation_s"] == 4.0
