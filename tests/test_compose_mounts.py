"""Phase 2a — deterministic compose-mount contract (regression-lock for #361).

The in-container free-data download bug (#361) was that ``docker-compose.yml``
mounted the reader's data directory **read-only** (``:ro``), so every downloader
failed with ``Read-only file system`` the moment it tried to write to
``/data``. The functional container smoke (Phase 2b, ``container-smoke.yml``)
catches that end-to-end, but a full 12 GB image pull is too heavy to run per PR.

This test pins the exact compose contract behind that fix, statically, in
milliseconds: the data volume is writable and ``ML4T_DATA_PATH`` points at the
mount. No Docker, no network — just parse the compose file. If someone flips the
mount back to ``:ro`` or repoints ``ML4T_DATA_PATH``, this fails on the PR that
does it, not weeks later in a reader's terminal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

# Services a reader actually runs the download workflow in. Benchmark/db-only
# services don't mount /data for writes and are out of scope here.
_WRITE_SERVICES = ("ml4t", "ml4t-gpu")


def _load_compose() -> dict:
    # SafeLoader resolves YAML anchors/aliases and ``<<`` merge keys, so each
    # service dict already carries the merged ``x-common`` volumes/environment.
    return yaml.safe_load(COMPOSE_FILE.read_text())


# Container target ``/data`` with an optional trailing mode. Anchored at the end
# so an interpolated source like ``${ML4T_DATA_PATH:-./data}`` — which itself
# contains ``:`` — can't be mistaken for the target/mode.
_DATA_MOUNT_RE = re.compile(r":(?P<target>/data)(?::(?P<mode>[a-zA-Z]+))?$")


def _data_mounts(service: dict) -> list[tuple[str, str]]:
    """``(volume_string, mode)`` for every mount whose container target is ``/data``.

    ``mode`` is ``""`` when the entry omits an explicit rw/ro suffix.
    """
    mounts = []
    for vol in service.get("volumes", []):
        if not isinstance(vol, str):
            continue  # long-form mounts not used for /data here
        m = _DATA_MOUNT_RE.search(vol)
        if m:
            mounts.append((vol, m.group("mode") or ""))
    return mounts


def test_compose_file_exists():
    assert COMPOSE_FILE.exists(), f"missing {COMPOSE_FILE}"


@pytest.mark.parametrize("service_name", _WRITE_SERVICES)
def test_data_mount_is_writable(service_name):
    """The reader's /data mount must be read-write (the #361 fix)."""
    compose = _load_compose()
    service = compose["services"][service_name]
    mounts = _data_mounts(service)
    assert mounts, f"{service_name}: no /data volume mount found"
    for vol, mode in mounts:
        assert mode != "ro", (
            f"{service_name}: /data is mounted read-only ('{vol}') — this is the "
            f"#361 bug; the in-container download workflow cannot write to /data"
        )
        assert mode == "rw", (
            f"{service_name}: /data mount '{vol}' must be explicitly ':rw' so the "
            f"download workflow can populate it"
        )


@pytest.mark.parametrize("service_name", _WRITE_SERVICES)
def test_data_path_env_points_at_mount(service_name):
    """ML4T_DATA_PATH inside the container must be the /data mount target.

    Every downloader resolves its output root from ML4T_DATA_PATH; if it doesn't
    equal the writable mount, files land somewhere the host mount can't see
    (the ETF wrong-dir class of bug).
    """
    compose = _load_compose()
    env = compose["services"][service_name].get("environment", [])
    # environment is a list of "KEY=VALUE" strings in this compose file.
    env_map = {}
    for item in env:
        if isinstance(item, str) and "=" in item:
            key, _, value = item.partition("=")
            env_map[key] = value
    assert env_map.get("ML4T_DATA_PATH") == "/data", (
        f"{service_name}: ML4T_DATA_PATH is {env_map.get('ML4T_DATA_PATH')!r}, "
        f"expected '/data' (the writable mount target)"
    )


def test_no_service_mounts_data_readonly():
    """Defensive: no service anywhere reintroduces a read-only /data mount."""
    compose = _load_compose()
    offenders = []
    for name, service in compose.get("services", {}).items():
        if not isinstance(service, dict):
            continue
        for vol, mode in _data_mounts(service):
            if mode == "ro":
                offenders.append(f"{name} -> {vol}")
    assert not offenders, f"read-only /data mount(s) found: {offenders}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
