"""The Interactive Brokers client is ib_async, and ib_insync must not be installed beside it.

Both distributions install a top-level ``eventkit`` package to the same
``site-packages/eventkit/`` path. ib_async ships a copy with the module-level
``main_event_loop = get_event_loop()`` removed; the standalone ``eventkit`` that
``ib_insync`` depends on still runs it at import time, and on Python 3.14
``asyncio.get_event_loop()`` raises when no loop is running. Whichever
distribution unpacks last wins, so an environment holding both is a coin flip.

The published ``ml4t/ml4t:latest`` image lost that flip. It builds with
``--extra live``, which pulled ``ib_insync``, so ``import ml4t.live`` - and
therefore every Ch25 notebook, and the documented onboarding gate
``docker compose run --rm ml4t python scripts/verify_installation.py`` - failed
with ``RuntimeError: There is no current event loop in thread 'MainThread'``.
The local ``uv sync`` path installs no extras, got ib_async's copy, and passed,
which is why the break reached a published image unnoticed.

Nothing in this repository imports ``ib_insync``; the notebooks and ``ml4t-live``
use ``ib_async``, its maintained successor.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ib_async_imports():
    """The failure a reader sees: importing the IB client raises at module scope."""
    pytest.importorskip("ib_async", reason="live extra not installed")

    import ib_async

    assert hasattr(ib_async, "IB")


def test_live_extra_declares_ib_async_not_ib_insync():
    live = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["live"]
    names = [spec.split(">")[0].split("=")[0].split("[")[0].strip().lower() for spec in live]

    assert "ib_insync" not in names and "ib-insync" not in names, (
        f"the live extra must not depend on ib_insync; found {live}"
    )
    assert "ib_async" in names or "ib-async" in names, (
        f"the live extra must depend on ib_async; found {live}"
    )


def test_lock_resolves_neither_ib_insync_nor_standalone_eventkit():
    """Static half, so a reintroduction fails its own PR rather than the next image build."""
    lock = (REPO_ROOT / "uv.lock").read_text()

    for package in ("ib-insync", "eventkit"):
        assert f'name = "{package}"' not in lock, (
            f"uv.lock resolves {package}, which overwrites ib_async's eventkit and breaks "
            "`import ib_async` on Python 3.14 (see this module's docstring)"
        )
