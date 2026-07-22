"""Every published compose port must bind to loopback only.

Docker's ``"8888:8888"`` short form publishes on **0.0.0.0** - every interface,
not just the reader's machine. With Jupyter running token-less, the repo
bind-mounted ``:rw`` and ``.env`` loaded, that made a reader's first documented
command (``docker compose up ml4t``) an unauthenticated remote shell for anyone
on their LAN: office, cafe, hotel, campus. The same applies to the benchmark
databases, which all ship default or empty credentials.

The fix is the explicit host IP - ``"127.0.0.1:8888:8888"``. It is one character
class away from being silently reverted, and a revert is invisible in review, so
pin it here: parse the compose file, assert every published port names a loopback
host IP. No Docker, no network, milliseconds.

Container-to-container traffic (benchmark -> clickhouse et al.) goes over the
compose network by service name and does not use published ports, so it is
unaffected by the host IP.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

_LOOPBACK = ("127.0.0.1", "::1")


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


def _published_ports(service: dict) -> list[tuple[object, str | None]]:
    """``(entry, host_ip)`` for every published port; ``host_ip`` is None if unset.

    Handles both the short string form (``"127.0.0.1:8888:8888"``) and the long
    mapping form (``{published: 8888, host_ip: 127.0.0.1}``).
    """
    published = []
    for entry in service.get("ports", []):
        if isinstance(entry, dict):
            published.append((entry, entry.get("host_ip")))
            continue
        # Short form: [HOST_IP:][HOST_PORT:]CONTAINER_PORT[/PROTOCOL]. An IPv6
        # host IP is bracketed, e.g. "[::1]:8888:8888".
        text = str(entry).split("/")[0]
        if text.startswith("["):
            host_ip = text[1 : text.index("]")]
        else:
            parts = text.split(":")
            # 3 parts means a host IP is present; fewer means it was omitted,
            # which is exactly the 0.0.0.0 default this test exists to catch.
            host_ip = parts[0] if len(parts) == 3 else None
        published.append((entry, host_ip))
    return published


def _services_with_ports() -> list[str]:
    compose = _load_compose()
    return sorted(
        name
        for name, service in compose["services"].items()
        if isinstance(service, dict) and service.get("ports")
    )


def test_compose_file_exists():
    assert COMPOSE_FILE.exists(), f"missing {COMPOSE_FILE}"


def test_some_service_publishes_ports():
    """Guard the guard: if the parse silently finds nothing, the test below is vacuous."""
    assert _services_with_ports(), "no service publishes any port - parser or compose changed"


@pytest.mark.parametrize("service_name", _services_with_ports())
def test_published_ports_bind_loopback(service_name):
    compose = _load_compose()
    service = compose["services"][service_name]
    for entry, host_ip in _published_ports(service):
        assert host_ip is not None, (
            f"{service_name}: port {entry!r} publishes with no host IP, which binds "
            f"0.0.0.0 (every interface) - reachable by anyone on the reader's network. "
            f"Use '127.0.0.1:<host>:<container>'."
        )
        assert host_ip in _LOOPBACK, (
            f"{service_name}: port {entry!r} publishes on {host_ip!r}, which is reachable "
            f"beyond the reader's machine. Published ports must bind loopback "
            f"({' or '.join(_LOOPBACK)})."
        )
