#!/usr/bin/env python3
"""Fail when the environment's installed distributions do not match uv.lock.

`ml4t/ml4t:latest` is the container every chapter and case-study job runs in. It
is built from the lock and drifts from it silently the moment the lock moves: the
image built 2026-07-16 carried `ml4t-models==0.1.0a4` while pyproject moved to
`>=0.1.0a6` on 2026-07-21, so for eleven days public CI ran a dependency set the
repository did not declare. `0.1.0a4`'s IPCA solver was genuinely broken, and the
notebook that failed on it read as a numerical problem, which cost four rounds of
investigation aimed at the notebook.

Nothing compared the two. This does, from inside the environment, with no network:
every installed distribution the lock pins must be at the locked version.

The general trap it closes: when a dev machine and CI disagree on the same code
and data, "different hardware" and "different dependency set" produce identical
symptoms, and only one of them is cheap to check.

## The carve-out is declared, not implicit

`envs/ml4t/Dockerfile` builds the torch stack from the CUDA index rather than from
the lock, so those distributions are a newer build than the lock's by design. They
are named here because an undeclared carve-out is what the LightGBM 4.7 nvlink
failure cost: `lightgbm` was excluded from the lock-derived constraint and then
installed as a floating `>=4.6`, which drifted to a version that could not build.
It is deliberately NOT exempt here - the Dockerfile now pins it to the lock's exact
version, and this check is what holds that.

A distribution `[tool.uv] override-dependencies` names is held to that override's
specifier rather than to the lock, because an override is uv asserting a version
against what the dependency graph asks for and a pip install cannot reproduce it.

Usage:
    python .github/scripts/check_env_matches_lock.py [--lock uv.lock] [--pyproject pyproject.toml]
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]

# Distributions the image deliberately installs off-lock, and why. A name here is
# a statement that the lock's version is not the one this environment is supposed
# to have - see envs/ml4t/Dockerfile, which installs the cu128 build of torch from
# download.pytorch.org and lets that wheel's own strict pins choose the rest.
DECLARED_OFF_LOCK = {
    "torch": "built from the cu128 index, newer than the lock's generic build",
    "torchvision": "pinned by the cu128 torch wheel",
    "torchaudio": "pinned by the cu128 torch wheel",
    "triton": "pinned by the cu128 torch wheel",
    "setuptools": "held below 82 by the cu128 torch wheel",
    "pip": "the environment's own installer, from the interpreter, not resolved by the project",
}
# The CUDA runtime wheels the torch build pins strictly, by prefix.
DECLARED_OFF_LOCK_PREFIXES = ("nvidia-",)


def overridden_dependencies(pyproject_path: Path) -> dict[str, Requirement]:
    """Distributions `[tool.uv] override-dependencies` forces the lock to resolve.

    An override is uv asserting a version against what the dependency graph asks
    for, and pip has no equivalent: `protobuf>=5.0` is overridden here because
    protobuf 4.x has a C-extension metaclass bug on Python 3.14, and the lock then
    resolves protobuf 7.35.0 while `opentelemetry-proto` requires `<7.0`. A pip
    install into the image resolves 6.33.6 and is right to.

    So an overridden distribution is held to the override rather than to the lock.
    It is not exempt from checking: exempting it outright would let protobuf 4.x
    through, which is the version the override exists to keep out.
    """
    if not pyproject_path.is_file():
        return {}
    project = tomllib.loads(pyproject_path.read_text())
    overrides = (project.get("tool", {}).get("uv", {}) or {}).get("override-dependencies", [])
    parsed = [Requirement(spec) for spec in overrides]
    return {canonical(requirement.name): requirement for requirement in parsed}


def canonical(name: str) -> str:
    """PEP 503 normalization: distribution names compare case- and separator-insensitively."""
    return name.lower().replace("_", "-").replace(".", "-")


def locked_versions(lock_path: Path) -> dict[str, str]:
    """{canonical name: version} for every package uv.lock resolves."""
    lock = tomllib.loads(lock_path.read_text())
    return {
        canonical(package["name"]): package["version"]
        for package in lock.get("package", [])
        if "version" in package
    }


def installed_versions() -> dict[str, str]:
    """{canonical name: version} for every distribution importable here."""
    found: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name and dist.version:
            found.setdefault(canonical(name), dist.version)
    return found


def is_declared_off_lock(name: str, overridden: dict[str, Requirement] | None = None) -> bool:
    """Whether this distribution is compared against something other than the lock."""
    return (
        name in DECLARED_OFF_LOCK
        or name in (overridden or {})
        or name.startswith(DECLARED_OFF_LOCK_PREFIXES)
    )


def drift(
    locked: dict[str, str],
    installed: dict[str, str],
    overridden: dict[str, Requirement] | None = None,
) -> list[tuple[str, str, str]]:
    """(name, what is required, installed version) for every distribution that disagrees.

    Only distributions that are both required and installed are compared. The lock
    resolves platform- and extra-specific packages this environment is not meant to
    carry, and an image that deliberately installs less than the lock is a size
    decision rather than drift; installing a *different version* of something that
    is pinned is what makes a CI result unattributable.

    An overridden distribution is compared against the override's specifier instead
    of the lock's version, because that is what the repository actually declares
    for it - see :func:`overridden_dependencies`.
    """
    overridden = overridden or {}
    mismatched = [
        (name, locked[name], version)
        for name, version in installed.items()
        if name in locked and not is_declared_off_lock(name, overridden) and locked[name] != version
    ]
    mismatched += [
        (name, str(requirement.specifier), installed[name])
        for name, requirement in overridden.items()
        if name in installed
        and not requirement.specifier.contains(Version(installed[name]), prereleases=True)
    ]
    return sorted(mismatched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=REPO_ROOT / "uv.lock")
    parser.add_argument("--pyproject", type=Path, default=REPO_ROOT / "pyproject.toml")
    args = parser.parse_args()

    if not args.lock.is_file():
        print(f"no lock file at {args.lock}", file=sys.stderr)
        return 2

    locked = locked_versions(args.lock)
    installed = installed_versions()
    overridden = overridden_dependencies(args.pyproject)
    mismatched = drift(locked, installed, overridden)
    compared = sum(
        1 for name in installed if name in locked and not is_declared_off_lock(name, overridden)
    )

    print(f"{args.lock} resolves {len(locked)} packages; {len(installed)} are installed here")
    print(f"compared {compared}, off-lock by declaration {len(DECLARED_OFF_LOCK)} + nvidia-*")
    for name, requirement in sorted(overridden.items()):
        state = installed.get(name, "not installed")
        print(f"held to pyproject's override instead of the lock: {requirement} ({state})")

    if not mismatched:
        print("every installed distribution the lock pins is at its locked version")
        return 0

    print(f"\n{len(mismatched)} installed distributions do not match what is declared:")
    for name, want, have in mismatched:
        print(f"  {name}: {want} required, installed {have}")
    print(
        "\nThis environment is not the one the repository declares, so a result "
        "measured in it is not evidence about this commit. Rebuild the image, or "
        "declare the difference in DECLARED_OFF_LOCK with the reason."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
