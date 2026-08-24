"""A registered training row has to say how much of the machine it used.

`pre_run_gate.py` fails "the run reports how much of the machine it used" when no
row carries `cores_used`, and `cores_used` appears only when `cpu_s` is passed
alongside `elapsed_s` (`utils/runtime.py:122-124`). Wall time alone cannot
recover it: an hour on one core and an hour on twenty are indistinguishable.

Three adapters recorded `elapsed_s` without `cpu_s`, so tabular_dl, deep_learning
and latent_factors rows could never clear that check while linear and gbm always
did.

This module holds the contract itself and imports nothing heavier than
`utils.runtime`, which is stdlib-only, so it runs in the torch-free `test-unit`
job. The TabM helper's own test needs `tabular_dl`, and therefore torch, so it
lives in `test_tabular_dl_runtime.py` beside the rest of that adapter's tests.
"""

from __future__ import annotations

import pytest

from case_studies.utils.runtime import resource_measurement


def test_cores_used_needs_cpu_seconds():
    """The gate's input, stated as a contract rather than assumed."""
    assert "cores_used" not in resource_measurement(elapsed_s=10.0)
    measured = resource_measurement(elapsed_s=10.0, cpu_s=40.0)
    assert measured["cores_used"] == pytest.approx(4.0)


def test_a_zero_length_run_reports_no_cores_rather_than_dividing():
    measured = resource_measurement(elapsed_s=0.0, cpu_s=5.0)
    assert measured["cores_used"] == 0.0
