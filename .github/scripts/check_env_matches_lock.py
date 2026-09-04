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

Usage:
    python .github/scripts/check_env_matches_lock.py [--lock uv.lock]
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from importlib.metadata import distributions
from pathlib import Path

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


def overridden_dependencies(pyproject_path: Path) -> dict[str, str]:
    """Distributions `[tool.uv] override-dependencies` forces the lock to resolve.

    An override is uv asserting a version against what the dependency graph asks
    for, and pip has no equivalent: `protobuf>=5.0` is overridden here because
    protobuf 4.x has a C-extension metaclass bug on Python 3.14, and the lock then
    resolves protobuf 7.35.0 while `opentelemetry-proto` requires `<7.0`. A pip
    install into the image resolves 6.33.6 and is right to. Comparing an overridden
    distribution against the lock therefore reports the override, not drift.
    """
    if not pyproject_path.is_file():
        return {}
    project = tomllib.loads(pyproject_path.read_text())
    overrides = (project.get("tool", {}).get("uv", {}) or {}).get("override-dependencies", [])
    return {
        canonical(re.split(r"[<>=!~ \[]", spec, maxsplit=1)[0]): f"overridden to {spec}"
        for spec in overrides
    }


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


def is_declared_off_lock(name: str, overridden: dict[str, str] | None = None) -> bool:
    return (
        name in DECLARED_OFF_LOCK
        or name in (overridden or {})
        or name.startswith(DECLARED_OFF_LOCK_PREFIXES)
    )


def drift(
    locked: dict[str, str],
    installed: dict[str, str],
    overridden: dict[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """(name, locked version, installed version) for every distribution that disagrees.

    Only distributions that are both locked and installed are compared. The lock
    resolves platform- and extra-specific packages this environment is not meant to
    carry, and an image that deliberately installs less than the lock is a size
    decision rather than drift; installing a *different version* of something the
    lock pins is what makes a CI result unattributable.
    """
    return sorted(
        (name, locked[name], version)
        for name, version in installed.items()
        if name in locked and not is_declared_off_lock(name, overridden) and locked[name] != version
    )


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
    print(
        f"compared {compared}, off-lock by declaration {len(DECLARED_OFF_LOCK)} + nvidia-*"
        f"{f', overridden in pyproject: {", ".join(sorted(overridden))}' if overridden else ''}"
    )

    if not mismatched:
        print("every installed distribution the lock pins is at its locked version")
        return 0

    print(f"\n{len(mismatched)} installed distributions do not match {args.lock}:")
    for name, want, have in mismatched:
        print(f"  {name}: lock says {want}, installed {have}")
    print(
        "\nThis environment is not the one the repository declares, so a result "
        "measured in it is not evidence about this commit. Rebuild the image, or "
        "declare the difference in DECLARED_OFF_LOCK with the reason."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
