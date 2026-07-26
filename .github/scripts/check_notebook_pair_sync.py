"""Gate: a committed ``.ipynb`` must agree with its jupytext-paired ``.py``.

JupyterLab's jupytext contents manager refuses to open a paired notebook whose
``.ipynb`` is newer than its ``.py`` and disagrees with it. The reader gets a
``File Load Error`` dialog instead of a notebook, and the notebook looks
corrupted even though its content is fine. Public issue #372 was exactly this:
papermill had stamped every cell with an empty ``tags: []`` that the ``.py`` did
not carry, across 41 shipped notebooks.

The specific fossil is cleaned by ``strip_empty_cell_tags.py``. This gate guards
the *invariant* rather than that one cause: whatever makes a pair disagree —
empty tags, a hand-edited ``.ipynb``, a forgotten ``jupytext --sync`` — is
caught here before it reaches a reader.

This complements ``notebook_provenance.py``, which asks whether the ``.ipynb``
is the current ``.py`` *executed*. This asks whether the pair can be *opened*.
A notebook can pass that gate and still fail this one.

Usage::

    uv run python .github/scripts/check_notebook_pair_sync.py          # all paired notebooks
    uv run python .github/scripts/check_notebook_pair_sync.py a.ipynb b.py  # only these (pre-commit)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = {"_reference", ".venv", ".git", ".ipynb_checkpoints"}


def _tracked_notebooks() -> list[Path]:
    out = []
    for p in REPO_ROOT.rglob("*.ipynb"):
        if SKIP_PARTS & set(p.parts):
            continue
        if p.name.startswith(("_executed_", "_lock_")):
            continue
        out.append(p)
    return sorted(out)


def _selected(argv: list[str]) -> list[Path]:
    """Notebooks to check: the paired .ipynb of any given path, else the whole repo."""
    if not argv:
        return _tracked_notebooks()
    seen: dict[Path, None] = {}
    for a in argv:
        nb = Path(a).resolve().with_suffix(".ipynb")
        if nb.exists() and not (SKIP_PARTS & set(nb.parts)):
            seen[nb] = None
    return sorted(seen)


def pair_diff(nb: Path) -> str:
    """jupytext's own diff between the notebook and its .py. Empty == in sync."""
    py = nb.with_suffix(".py")
    if not py.exists():
        return ""
    r = subprocess.run(
        ["jupytext", "--diff", str(py), str(nb)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def main(argv: list[str]) -> int:
    desynced = [nb for nb in _selected(argv) if pair_diff(nb)]
    if not desynced:
        return 0

    print("notebook/py pairs disagree — JupyterLab will refuse to open these:")
    for nb in desynced:
        print(f"  {nb.relative_to(REPO_ROOT)}")
    print(
        "\nReaders see a 'File Load Error' dialog, not a notebook (cf. public #372).\n"
        "Fix: `uv run python .github/scripts/strip_empty_cell_tags.py` if it is empty `tags: []`,\n"
        "otherwise re-sync the pair (`jupytext --sync <nb>.py`) and re-execute if code changed."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
