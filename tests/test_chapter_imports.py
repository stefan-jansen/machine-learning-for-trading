"""Guard: per-chapter helper modules import from the repo root.

Chapter directories are number-prefixed (``25_live_trading``), so they are not
Python packages and their helper modules (``async_utils`` etc.) are only
importable when the chapter directory is on ``sys.path``. ``sitecustomize.py``
(declared as a top-level py-module in pyproject) arranges that at interpreter
startup in every environment. This test pins that contract: a bare
``import async_utils`` must succeed in a fresh interpreter started from the
repo root, with no chapter directory injected onto the path.

If this fails, ``sitecustomize.py`` or its ``[tool.setuptools] py-modules``
declaration was likely removed, or the package needs reinstalling
(``uv pip install -e .``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_CHAPTER_DIR = re.compile(r"/\d\d_[^/]+/?$")


def test_chapter_helper_imports_from_repo_root() -> None:
    # Inherit the real environment (so PYTHONPATH=/app survives in Docker CI),
    # but strip any pre-injected chapter directory so the test genuinely
    # exercises the sitecustomize hook rather than a path the harness added.
    env = dict(os.environ)
    if pp := env.get("PYTHONPATH"):
        kept = [p for p in pp.split(os.pathsep) if not _CHAPTER_DIR.search(p)]
        env["PYTHONPATH"] = os.pathsep.join(kept)

    # Representative sibling helpers from number-prefixed chapter dirs.
    code = "import async_utils, limit_orderbook; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        "Bare chapter-helper import failed from the repo root — the "
        "sitecustomize path hook is not active.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
