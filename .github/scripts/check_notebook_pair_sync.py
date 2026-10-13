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

jupytext is resolved by probing ``sys.executable -m jupytext`` first and a
``jupytext`` on ``PATH`` second, and never by assuming either runs. A console
script left behind by another environment points at an interpreter that may no
longer carry the module, and the empty stdout it produces is indistinguishable
from "in sync" - which is how this gate passed two notebooks it never compared
on 2026-10-13, both of which CI then rejected. The gate now measures with
something it has watched run, or reports that it could not measure.

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


class JupytextUnusable(RuntimeError):
    """`jupytext --diff` could not be run, so the pair was never compared."""


def _works(cmd: list[str]) -> bool:
    """True if `cmd --version` actually runs. Probing is the whole point.

    Separate from the diff because `python -m` conflates two exit codes: "no
    module named jupytext" and "jupytext ran and disagreed" are both non-zero,
    so the diff's status alone cannot tell a broken environment from a desynced
    pair.
    """
    try:
        r = subprocess.run([*cmd, "--version"], capture_output=True, text=True)
    except OSError:
        return False
    return r.returncode == 0


def _jupytext_cmd() -> list[str]:
    """The first jupytext invocation that runs, preferring this interpreter.

    `sys.executable -m jupytext` comes first so the gate measures with the
    environment it was started in. The `jupytext` console script is the
    fallback, because the gate also runs under interpreters that do not carry
    the module themselves. Each candidate is probed rather than assumed: a
    script left behind by a deleted environment is on PATH on this machine and
    raises ModuleNotFoundError on startup, and the empty stdout that produces
    reads exactly like "in sync".
    """
    for cmd in ([sys.executable, "-m", "jupytext"], ["jupytext"]):
        if _works(cmd):
            return cmd
    raise JupytextUnusable(
        "no working jupytext: neither "
        f"`{sys.executable} -m jupytext` nor a `jupytext` on PATH would run"
    )


def pair_diff(nb: Path) -> str:
    """jupytext's own diff between the notebook and its .py. Empty == in sync.

    An empty stdout means "in sync" only if jupytext actually ran. A missing or
    broken `jupytext` on PATH also produces empty stdout, and reading that as
    agreement is how this gate passes a pair it never looked at: measured
    2026-10-13, a `~/.local/bin/jupytext` that raised on startup made both this
    gate and its pre-commit hook green on two notebooks papermill had stamped
    with `tags: []` on every cell, which CI then failed on. A tool that cannot
    run is not a passing check.
    """
    py = nb.with_suffix(".py")
    if not py.exists():
        return ""
    try:
        r = subprocess.run(
            [*_jupytext_cmd(), "--diff", str(py), str(nb)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError as e:
        raise JupytextUnusable(f"could not execute jupytext: {e}") from e
    # jupytext --diff exits 0 whether or not it printed a diff, and the command
    # was probed above, so a non-zero exit is the tool itself failing.
    if r.returncode != 0:
        raise JupytextUnusable(
            f"jupytext --diff exited {r.returncode} on {py.name}:\n"
            f"{(r.stderr or r.stdout).strip()[-800:]}"
        )
    return r.stdout.strip()


def main(argv: list[str]) -> int:
    try:
        desynced = [nb for nb in _selected(argv) if pair_diff(nb)]
    except JupytextUnusable as e:
        print(f"cannot check notebook/py pairs: {e}", file=sys.stderr)
        print(
            "\nThis is not a passing check - no pair was compared. Install jupytext into the\n"
            "environment running this gate (`uv run python .github/scripts/...` uses the\n"
            "locked one) and re-run.",
            file=sys.stderr,
        )
        return 2
    if not desynced:
        return 0

    print("notebook/py pairs disagree — JupyterLab will refuse to open these:")
    for nb in desynced:
        print(f"  {nb.relative_to(REPO_ROOT)}")
    print(
        "\nReaders see a 'File Load Error' dialog, not a notebook (cf. public #372).\n"
        "Fix: `uv run python .github/scripts/strip_empty_cell_tags.py` if it is empty `tags: []`.\n"
        "Otherwise bring the .ipynb forward with the smallest command that covers the edit, and\n"
        "re-execute only if code that computes something changed:\n"
        "  markdown only, including adding/deleting/merging/retagging a markdown cell:\n"
        "    notebook_provenance.py sync-prose <nb.py>\n"
        "  figure alt text, alone or together with any of the above:\n"
        "    notebook_provenance.py sync-alt <nb.py>\n"
        "Both keep every output and the original executed_at. A bare `jupytext --sync` is not the\n"
        "route: it brings the pair into agreement without checking that the outputs still belong\n"
        "to the source, which is the claim the provenance stamp makes."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
