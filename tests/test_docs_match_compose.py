"""Every service, profile, port and notebook path the docs tell a reader to type must exist.

There is no sync tooling between `docs/`, `README.md` and `docker-compose.yml`,
and the install doc has silently drifted before. A mirroring script is the wrong
shape for that: it would keep two copies of the same prose equal without either
one being right. What a reader actually hits is a command that no longer names
anything - a renamed service, a profile that moved, a port the compose file
stopped publishing, a notebook path that was reorganised. Those are checkable
against the repository, so check them.

Scope is deliberately narrow: the commands a reader is told to run, from the
files an onboarding reader reads. It parses `docker compose` invocations out of
fenced code blocks and resolves each name against the compose file and the
working tree. It runs no Docker, opens no socket, and takes milliseconds.

What it does not check: prose. A sentence claiming the image is 12 GB is not
verifiable here, and pretending otherwise is how a check becomes noise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

# The files an onboarding reader is sent to, in the order they meet them.
DOC_FILES = (
    "README.md",
    "docs/installation.md",
    "docs/running-notebooks.md",
    "envs/README.md",
)

# `docker compose` subcommands whose first non-flag argument is a service name.
_SERVICE_SUBCOMMANDS = {"pull", "up", "run", "build", "exec", "restart", "stop", "start", "logs"}

# Flags that swallow the token after them, so it is never a service name.
_FLAGS_WITH_VALUE = {
    "-e",
    "--env",
    "-p",
    "--publish",
    "-w",
    "--workdir",
    "-u",
    "--user",
    "--file",
    "-f",
    "--project-name",
    "-P",
    "--scale",
    "--entrypoint",
}

# Every fence, tagged or not. Matching only the shell tags would mis-pair: an
# untagged or ```powershell fence would not register as an opener, and its
# closing ``` would then open a "block" of ordinary prose.
_FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```", re.DOTALL | re.MULTILINE)
_SHELL_TAGS = {"", "bash", "sh", "shell", "console"}


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


def _compose_services() -> dict[str, dict]:
    return _load_compose()["services"]


def _compose_profiles() -> set[str]:
    return {p for svc in _compose_services().values() for p in svc.get("profiles", [])}


def _published_host_ports() -> set[str]:
    """Host-side ports the compose file publishes, as strings."""
    ports: set[str] = set()
    for svc in _compose_services().values():
        for entry in svc.get("ports", []):
            if isinstance(entry, dict):
                ports.add(str(entry["published"]))
                continue
            text = str(entry).split("/")[0]
            if text.startswith("["):  # bracketed IPv6 host IP
                text = text[text.index("]") + 1 :].lstrip(":")
            parts = text.split(":")
            # HOST_IP:HOST_PORT:CONTAINER  |  HOST_PORT:CONTAINER  |  CONTAINER
            if len(parts) == 3:
                ports.add(parts[1])
            elif len(parts) == 2:
                ports.add(parts[0])
    return ports


def _shell_lines(doc: Path) -> list[str]:
    """Every logical command line inside a fenced shell block, continuations joined."""
    lines: list[str] = []
    for tag, block in _FENCE.findall(doc.read_text()):
        if tag.strip().lower() not in _SHELL_TAGS:
            continue
        joined = block.replace("\\\n", " ")
        for raw in joined.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                lines.append(line)
    return lines


def _compose_invocations() -> list[tuple[str, str]]:
    """``(doc path, command line)`` for every documented `docker compose ...` call."""
    found = []
    for name in DOC_FILES:
        doc = REPO_ROOT / name
        for line in _shell_lines(doc):
            if "docker compose" in line:
                found.append((name, line[line.index("docker compose") :]))
    return found


def _parse(command: str) -> tuple[list[str], str | None, str | None]:
    """``(profiles, subcommand, service)`` for one `docker compose` line."""
    tokens = command.split()[2:]  # drop "docker compose"
    profiles: list[str] = []
    subcommand: str | None = None
    service: str | None = None

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--profile" and index + 1 < len(tokens):
            profiles.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("--profile="):
            profiles.append(token.split("=", 1)[1])
            index += 1
            continue
        if subcommand is None:
            if not token.startswith("-"):
                subcommand = token
            index += 1
            continue
        if token in _FLAGS_WITH_VALUE:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        service = token
        break
    return profiles, subcommand, service


def test_documented_services_exist():
    services = _compose_services()
    unknown = []
    for doc, command in _compose_invocations():
        profiles, subcommand, service = _parse(command)
        if subcommand not in _SERVICE_SUBCOMMANDS or service is None:
            continue
        if service not in services:
            unknown.append(f"{doc}: `{command}` names service {service!r}")

    assert not unknown, "docker-compose.yml has no such service:\n  " + "\n  ".join(unknown)


def test_documented_profiles_exist():
    declared = _compose_profiles()
    unknown = []
    for doc, command in _compose_invocations():
        profiles, _, _ = _parse(command)
        unknown += [
            f"{doc}: `{command}` names profile {p!r}" for p in profiles if p not in declared
        ]

    assert not unknown, "no service declares that profile:\n  " + "\n  ".join(unknown)


def test_documented_profile_matches_the_service_it_targets():
    """A `--profile` that does not cover the named service is a stale doc.

    Compose still runs it - naming a service explicitly activates its profiles -
    so this drift is invisible to a reader who follows the instruction and to
    anyone who tests it. It surfaces the day they adapt the command.
    """
    services = _compose_services()
    mismatched = []
    for doc, command in _compose_invocations():
        profiles, subcommand, service = _parse(command)
        if not profiles or subcommand not in _SERVICE_SUBCOMMANDS or service not in services:
            continue
        declared = set(services[service].get("profiles", []))
        if declared and not declared & set(profiles):
            mismatched.append(
                f"{doc}: `{command}` passes {profiles} but {service!r} declares {sorted(declared)}"
            )

    assert not mismatched, "profile does not cover the service:\n  " + "\n  ".join(mismatched)


def test_no_bare_docker_compose_up():
    """`docker compose up` with no service named starts every profile-less service.

    Naming a profiled service explicitly activates its profile, so a command that
    names one is fine without `--profile`. A bare `up` is the case that behaves
    differently from what the surrounding prose describes.
    """
    services = _compose_services()
    bare_up = []
    for doc, command in _compose_invocations():
        profiles, subcommand, service = _parse(command)
        if subcommand == "up" and service is None and not profiles:
            bare_up.append(f"{doc}: `{command}`")

    assert not bare_up, (
        "`docker compose up` with no service starts every profile-less service at once; "
        "name the service:\n  " + "\n  ".join(bare_up)
    )


def test_documented_localhost_ports_are_published():
    published = _published_host_ports()
    unpublished = []
    for name in DOC_FILES:
        doc = REPO_ROOT / name
        for port in set(re.findall(r"localhost:(\d{2,5})", doc.read_text())):
            if port not in published:
                unpublished.append(f"{name}: localhost:{port}")

    assert not unpublished, "docker-compose.yml publishes no such host port:\n  " + "\n  ".join(
        sorted(unpublished)
    )


@pytest.mark.parametrize("doc", DOC_FILES)
def test_documented_notebook_paths_exist(doc: str):
    """Every `python <chapter>/<stem>.py` a doc tells a reader to run must be a real file."""
    pattern = re.compile(r"\bpython3?\s+((?:case_studies/)?[\w.]+/[\w/]+\.py)\b")
    missing = []
    for line in _shell_lines(REPO_ROOT / doc):
        for path in pattern.findall(line):
            if not (REPO_ROOT / path).exists():
                missing.append(f"{doc}: `{path}`")

    assert not missing, "no such file in the repository:\n  " + "\n  ".join(missing)
