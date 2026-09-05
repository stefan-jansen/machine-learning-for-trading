"""A CI job's environment is decided by `uv.lock`, not by whatever released today.

Three jobs in `.github/workflows/test.yml` build a small venv with `uv pip install`
rather than syncing the project, deliberately: `guards` and `test-unit` are fast
gates and a full sync would cost more than the checks they run. That is a
reasonable trade as long as the *list* is names and the *versions* still come from
the lock.

They did not. `test-unit` installed `"ml4t-diagnostic>=0.1.2"`, and on 2026-09-05
`ml4t-diagnostic` 0.1.4 shipped with `WalkForwardCV` returning folds oldest-first.
The next run picked it up, `generate_cv_splits` raised on every case study's
configuration, and `test-unit` - a required check - went red on `main` and on every
branch cut from it with no commit in the repository having changed. `uv.lock` said
0.1.2 throughout, and the job that was measuring the repository was not measuring
the environment the repository declares.

That is the same shape as `check_env_matches_lock.py`, which exists because the
Docker image drifted off the lock the same way. The fix is the same one
`envs/ml4t/Dockerfile` already uses: export the lock as a fully-pinned requirements
file and hand it to `uv` as a constraint.

So this asserts the contract rather than the text. Every `uv pip install` in the
workflow constrains against the lock; no package it names carries a version, because
naming one is how the two sources of truth start to disagree; and every package it
names is actually in the lock, because a constraint file that does not mention a
package does not constrain it and the pin would be silently absent.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WORKFLOW = REPO_ROOT / ".github/workflows/test.yml"
LOCK = REPO_ROOT / "uv.lock"

# uv itself, and the project. `uv` is the installer rather than a dependency, and
# `ml4t` is this repository, which the lock does not pin a version of.
NOT_LOCKED = frozenset({"uv", "ml4t"})


def _install_commands() -> list[str]:
    """Each `uv pip install` invocation in the workflow, line continuations joined."""
    text = WORKFLOW.read_text(encoding="utf-8").replace("\\\n", " ")
    return [line.strip() for line in text.splitlines() if "uv pip install" in line]


def _named_packages(command: str) -> list[str]:
    """The package arguments of an install command, options and their values removed."""
    tokens = command.split("uv pip install", 1)[1].split()
    packages: list[str] = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            skip = "=" not in token
            continue
        packages.append(token.strip("\"'"))
    return packages


def _locked_names() -> set[str]:
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    return {p["name"].lower().replace("_", "-") for p in lock["package"]}


def test_the_workflow_still_builds_venvs_by_hand():
    """The premise. If these jobs ever sync the project instead, delete this file."""
    assert _install_commands(), (
        "no `uv pip install` left in test.yml - if the fast gates now sync the "
        "project, this whole contract is moot and the file should go"
    )


def test_every_hand_built_venv_is_constrained_by_the_lock():
    for command in _install_commands():
        assert "--constraint" in command, (
            "an unconstrained `uv pip install` resolves to whatever released today, "
            "so an external publish decides what a required check measures:\n"
            f"  {command.strip()}"
        )


def test_no_installed_package_carries_a_version():
    """A version in the workflow is a second source of truth, and it loses.

    `jupytext==1.19.3` sat beside `"ml4t-diagnostic>=0.1.2"` and both were wrong to
    be there: one duplicated the lock and would silently go stale against it, the
    other was a floor that let anything newer in.
    """
    offenders = [
        pkg
        for command in _install_commands()
        for pkg in _named_packages(command)
        if re.search(r"[=<>~!]", pkg)
    ]

    assert not offenders, (
        f"{offenders} pin or bound a version in the workflow; uv.lock decides these. "
        "Name the package and let the constraint file pin it."
    )


def test_every_installed_package_is_in_the_lock():
    """A constraint file only constrains what it mentions.

    Installing a package the lock has never heard of reintroduces the defect for
    that one package, silently, in a job that now looks pinned.
    """
    locked = _locked_names()
    missing = sorted(
        {
            pkg.lower().replace("_", "-")
            for command in _install_commands()
            for pkg in _named_packages(command)
        }
        - locked
        - NOT_LOCKED
    )

    assert not missing, (
        f"{missing} are installed in CI but absent from uv.lock, so the constraint "
        "file cannot pin them and they resolve freely. Add them to the project's "
        "dependencies, or drop them."
    )
