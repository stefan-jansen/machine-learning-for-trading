"""Operational provenance and resource measurement for model runners.

This module is in no family's source identity: it is described by none of the declared versions
that linear and GBM hash, and it is listed in none of the source-file digests that tabular DL,
sequence models, latent factors and causal inference hash. What a run *records about itself* must
be changeable without refitting a family, which is why the declared provenance and the resource
capture live here rather than in the runners - moving any of it into a digested module would make
every future edit to it cost a refit.

Nothing in this module may influence a fitted result.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# POSIX only, and this module is imported by every model runner, so a bare import made
# `import case_studies.utils.gbm` fail outright on native Windows - found by the Reader
# install walk once econml 0.17.0 made `uv sync` complete there. The two readings it
# provides have exact Windows equivalents below; nothing here influences a fitted result,
# so a platform difference in how a cost is measured is not a difference in the result.
try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - Windows
    resource = None

__all__ = [
    "ResourceUsage",
    "capture_resources",
    "cpu_seconds",
    "peak_rss_bytes",
    "resource_measurement",
    "runtime_provenance",
    "source_commit",
]


def source_commit(repository_root: Any) -> str:
    """Return the commit the run was executed from, or ``unknown``."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def runtime_provenance(
    repository_root: Any,
    *,
    entry_point: str,
    packages: dict[str, str],
    **extra: Any,
) -> dict[str, Any]:
    """Return the declared provenance written beside every training row.

    This is the immutable half: what the run was, not what it cost. It is written once when the
    row is registered and is compared byte for byte on re-registration, so measured quantities
    must not be added to it.
    """
    record = {
        "entry_point": entry_point,
        "packages": dict(packages),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_commit": source_commit(repository_root),
    }
    record.update(extra)
    return record


class ResourceUsage(dict):
    """What one fit cost. Populated when the :func:`capture_resources` block exits."""

    @property
    def elapsed_s(self) -> float:
        return float(self.get("elapsed_s", 0.0))


def _windows_peak_working_set() -> int:
    """Peak working set of this process, in bytes - the Windows reading of `ru_maxrss`.

    `PROCESS_MEMORY_COUNTERS` is `cb`, `PageFaultCount`, then eight `SIZE_T` fields, of which
    `PeakWorkingSetSize` is the first. Reported in bytes already, so no scale is applied.
    Returning 0 on failure would be a false measurement, so this raises nothing and the
    caller sees the zero only if the API itself refused.
    """
    import ctypes
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.PeakWorkingSetSize) if ok else 0


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ``ru_maxrss`` is a high-water mark over the life of the process and never falls, so this is
    the peak of the notebook rather than of the run that just finished. That is the number the
    concurrency policy needs - whether this notebook can share the machine - and it is recorded
    under a name that says so.

    ``ru_maxrss`` carries no portable unit: Linux reports kilobytes and macOS reports bytes, so
    the scale is read off the platform rather than assumed. Reading it as kilobytes on macOS
    reports a peak 1024 times too large, and `scripts/pre_run_gate.py` prints that figure in GB
    as a check it passes on.
    """
    if resource is None:  # pragma: no cover - Windows
        return _windows_peak_working_set()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1 if sys.platform == "darwin" else 1024
    return int(usage.ru_maxrss) * scale


def cpu_seconds() -> float:
    """CPU time consumed by this process so far. Differences between two readings are the run."""
    if resource is None:  # pragma: no cover - Windows
        # User plus system CPU for this process, which is what ru_utime + ru_stime sums.
        return float(time.process_time())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def resource_measurement(
    *,
    elapsed_s: float,
    cpu_s: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """What a completed run cost, in the shape the registry records.

    ``cores_used`` is CPU seconds over wall seconds. It is the number that decides how many
    notebooks can share the machine, and it cannot be recovered from wall time alone: an hour
    spent on one core and an hour spent on twenty look identical until this is recorded.
    """
    measured: dict[str, Any] = {
        "elapsed_s": float(elapsed_s),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "cpu_count": os.cpu_count(),
    }
    if cpu_s is not None:
        measured["cpu_s"] = float(cpu_s)
        measured["cores_used"] = (float(cpu_s) / elapsed_s) if elapsed_s > 0 else 0.0
    measured.update({key: value for key, value in extra.items() if value is not None})
    return measured


@contextmanager
def capture_resources(**declared: Any) -> Iterator[ResourceUsage]:
    """Measure wall time, CPU time and peak memory across a block.

    ``cpu_s / elapsed_s`` is how many cores the block actually used, which is what decides
    whether two notebooks can run at once. Both are needed: elapsed alone cannot tell a
    CPU-bound run from one that spent the time waiting.
    """
    measured = ResourceUsage(declared)
    started_wall = time.perf_counter()
    started_cpu = cpu_seconds()
    try:
        yield measured
    finally:
        measured.update(
            resource_measurement(
                elapsed_s=time.perf_counter() - started_wall,
                cpu_s=cpu_seconds() - started_cpu,
            )
        )
