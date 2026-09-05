"""`case_studies.utils.runtime` must import where the POSIX `resource` module does not exist.

Every model runner imports it, so a bare `import resource` made `import case_studies.utils.gbm`
fail outright on native Windows - `ModuleNotFoundError: No module named 'resource'`, found by the
Reader install walk once econml 0.17.0 made `uv sync` complete there.

The two readings it provides are costs, not results: this module states that nothing in it may
influence a fitted result, so measuring a cost differently per platform is not a difference in
the numbers a reader gets.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import time

import pytest

MODULE = "case_studies.utils.runtime"


@pytest.fixture
def runtime_without_resource(monkeypatch):
    """Import the module with `resource` absent, the way a Windows interpreter sees it."""
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "resource":
            raise ModuleNotFoundError("No module named 'resource'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.delitem(sys.modules, "resource", raising=False)
    module = importlib.import_module(MODULE)
    yield module
    # Leave the real module behind for everything else in the session.
    monkeypatch.undo()
    importlib.reload(importlib.import_module(MODULE))


def test_the_module_imports_without_the_posix_resource_module(runtime_without_resource):
    assert runtime_without_resource.resource is None


def test_cpu_seconds_still_reports_process_cpu_time(runtime_without_resource):
    """`time.process_time()` sums user and system CPU for the process, which is what
    `ru_utime + ru_stime` sums. A difference between two readings is still the run."""
    before = runtime_without_resource.cpu_seconds()
    deadline = time.process_time() + 0.02
    while time.process_time() < deadline:
        pass
    after = runtime_without_resource.cpu_seconds()

    assert before >= 0.0
    assert after > before


def test_resource_measurement_still_records_every_field(runtime_without_resource, monkeypatch):
    """`process_peak_rss_bytes` is read through the Windows helper on that platform; here the
    helper cannot run, so it is stubbed to prove the field is still populated and `cores_used`
    is still derived."""
    monkeypatch.setattr(runtime_without_resource, "_windows_peak_working_set", lambda: 4096)

    measured = runtime_without_resource.resource_measurement(elapsed_s=2.0, cpu_s=8.0)

    assert measured["process_peak_rss_bytes"] == 4096
    assert measured["cpu_s"] == 8.0
    assert measured["cores_used"] == 4.0
    assert measured["cpu_count"] == runtime_without_resource.os.cpu_count()


def test_the_posix_reading_is_unchanged_where_resource_exists():
    """The platform this runs on keeps `getrusage`, so the recorded numbers do not move."""
    runtime = importlib.import_module(MODULE)
    if runtime.resource is None:  # pragma: no cover - only on Windows
        pytest.skip("no POSIX resource module on this platform")
    assert runtime.peak_rss_bytes() > 0
    assert runtime.cpu_seconds() > 0.0
