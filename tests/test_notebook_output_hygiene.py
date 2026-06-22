"""Guard: committed notebooks must not leak machine-specific absolute paths.

Notebooks executed on a contributor's machine can bake ``/home/<user>/...``
paths into committed cell outputs and papermill metadata. Readers should never
see those. This test scans every tracked ``.ipynb`` and fails if any survive.

To fix a failure, run::

    uv run python scripts/sanitize_notebook_paths.py

which rewrites repo-internal paths to repo-relative form. See that script and
``utils.paths.display_path`` for the source-side helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sanitize_notebook_paths import _iter_notebooks, sanitize_text  # noqa: E402


def test_no_machine_specific_paths_in_committed_notebooks() -> None:
    offenders: list[str] = []
    for nb in _iter_notebooks():
        raw = nb.read_text(encoding="utf-8")
        _, n = sanitize_text(raw)
        if n:
            offenders.append(f"{nb.relative_to(REPO_ROOT)} ({n})")
    assert not offenders, (
        "Notebooks leak machine-specific absolute paths in their committed "
        "outputs/metadata. Run `uv run python scripts/sanitize_notebook_paths.py` "
        "to fix:\n  " + "\n  ".join(offenders)
    )
