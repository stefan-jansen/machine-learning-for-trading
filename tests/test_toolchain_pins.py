"""The formatter the project installs and the formatter pre-commit runs are one version.

`uv run ruff format` and the pre-commit hook are two separate installs of ruff, and
nothing but this test makes them the same version. When they differ they format the
same code differently: 0.15.14 rewrote a committed multi-line `assert ... , (...)`
in `21_rl_execution_hedging/07_backtest_with_impact.py` and the 0.15.8 hook rewrote
it back.

That is not cosmetic. `notebooks/TEACHING_WORKER.md` has a worker format the `.py`,
`jupytext --sync`, execute, then commit; jupytext embeds the `.py` source in the
`.ipynb`, so the hook's counter-reformat at commit time makes the freshly executed
notebook a stale render of its own source and the notebook-sync gate rejects it. The
worker then re-syncs, re-executes and re-stamps a notebook nothing was wrong with.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_PRE_COMMIT = "https://github.com/astral-sh/ruff-pre-commit"


def _pyproject_ruff_pin() -> str:
    """The exact version the dev extra pins, e.g. "0.15.14"."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    pins = [
        match.group(1)
        for spec in dev
        if (match := re.fullmatch(r"ruff==([0-9][^\s;]*)", spec.strip()))
    ]

    assert pins, f"the dev extra must pin ruff exactly (`ruff==X.Y.Z`); found {dev}"
    return pins[0]


def _pre_commit_ruff_rev() -> str:
    """The rev of the ruff hook, with the leading `v` stripped."""
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    revs = [repo["rev"] for repo in config["repos"] if repo.get("repo") == RUFF_PRE_COMMIT]

    assert len(revs) == 1, f"expected exactly one {RUFF_PRE_COMMIT} entry, found {revs}"
    return revs[0].lstrip("v")


def test_pre_commit_ruff_matches_the_project_pin() -> None:
    assert _pre_commit_ruff_rev() == _pyproject_ruff_pin(), (
        "the ruff pre-commit hook and the project's ruff pin are different versions, "
        "so `uv run ruff format` and `pre-commit run ruff-format` disagree; "
        "bump both together"
    )


def test_the_lock_resolves_the_pinned_ruff() -> None:
    """A pin the lock has not been regenerated against installs some other version."""
    pin = _pyproject_ruff_pin()
    lock = (REPO_ROOT / "uv.lock").read_text()

    assert f'name = "ruff"\nversion = "{pin}"' in lock, (
        f"uv.lock does not resolve ruff {pin}; run `uv lock` after changing the pin"
    )
