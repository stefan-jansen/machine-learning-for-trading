"""Strip empty ``"tags": []`` cell metadata from committed notebooks.

Papermill stamps every cell it executes with an empty ``tags`` list. Our
notebooks are jupytext-paired with ``cell_metadata_filter: "tags,-all"``, so
``tags`` is a *tracked* key: an empty list in the ``.ipynb`` that the ``.py``
does not carry counts as a real difference between the pair.

JupyterLab's jupytext contents manager refuses to open a paired notebook whose
``.ipynb`` is newer than its ``.py`` and disagrees with it, so readers get a
``File Load Error`` dialog instead of a notebook (reported as "corrupted
notebooks" in public issue #372). The content is fine; only the metadata is out
of sync.

Empty tag lists carry no information, so removing them is lossless. Non-empty
tags (``parameters``, ``injected-parameters``) are load-bearing and preserved.

A handful of ``.py`` files carry the same fossil (``# %% tags=[]``) because a
``jupytext --sync`` propagated it. Those pairs *agree*, so they load fine and are
NOT part of this bug; stripping only the ``.ipynb`` side would be what breaks
them. They are skipped: the invariant is that the pair agrees, not that the
fossil is gone. Leaving their ``.py`` untouched also keeps the provenance stamps
of those notebooks valid (the gate hashes the ``.py`` blob), which is why this
tool never edits a ``.py``.

Like ``sanitize_notebook_paths.py`` this edits the raw ``.ipynb`` text rather
than reserializing the JSON: a no-op round-trip is *not* byte-identical for
every notebook in this repo, so reserializing would produce large spurious
diffs. Editing the text keeps the diff to the removed lines.

Idempotent: running twice is a no-op. A companion test
(``tests/test_notebook_output_hygiene.py``) fails CI if any empty tag survives.

Usage:
    uv run python .github/scripts/strip_empty_cell_tags.py            # rewrite in place
    uv run python .github/scripts/strip_empty_cell_tags.py a.ipynb    # selected notebooks
    uv run python .github/scripts/strip_empty_cell_tags.py --check    # report only, exit 1 if dirty
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from sanitize_notebook_paths import REPO_ROOT, _iter_notebooks

# Two serializations occur, depending on whether ``tags`` is the last key of the
# cell's metadata object. Both are removed whole; the comma is part of the match
# so the surrounding JSON stays valid.
#
#   mid-object:   "tags": [],          -> drop the line
#   last key:     ...,\n   "tags": []  -> drop the line and the preceding comma
#
# ``\[\]`` only ever matches an *empty* list, so multi-line non-empty tag blocks
# ("tags": [\n "parameters"\n ]) can never be touched.
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'[ \t]*"tags": \[\],\n'), ""),
    (re.compile(r',\n[ \t]*"tags": \[\]\n'), "\n"),
    (re.compile(r'[ \t]*"tags": \[\]\n'), ""),
]


def strip_text(text: str) -> tuple[str, int]:
    n = 0
    for pat, new in PATTERNS:
        text, k = pat.subn(new, text)
        n += k
    return text, n


def paired_py_has_fossil(nb: Path) -> bool:
    """True if this notebook's paired .py carries `tags=[]` markers of its own.

    Both sides then agree and the notebook loads; stripping one side would break
    the pair. See module docstring.
    """
    py = nb.with_suffix(".py")
    return py.exists() and "tags=[]" in py.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if any found")
    ap.add_argument("paths", nargs="*", type=Path, help="notebooks to inspect (default: all)")
    args = ap.parse_args()

    notebooks = (
        [path.resolve().with_suffix(".ipynb") for path in args.paths]
        if args.paths
        else _iter_notebooks()
    )

    dirty: list[tuple[Path, int]] = []
    skipped = 0
    for nb in notebooks:
        if not nb.exists():
            raise SystemExit(f"notebook not found: {nb}")
        if paired_py_has_fossil(nb):
            skipped += 1
            continue
        raw = nb.read_text(encoding="utf-8")
        new, n = strip_text(raw)
        if n:
            dirty.append((nb.relative_to(REPO_ROOT), n))
            if not args.check:
                nb.write_text(new, encoding="utf-8")

    note = f" ({skipped} skipped: paired .py carries the same markers)" if skipped else ""
    if not dirty:
        print(f"clean: no empty cell tags in any notebook{note}")
        return 0

    verb = "would strip" if args.check else "stripped"
    total = sum(n for _, n in dirty)
    print(f"{verb} {total} empty tag list(s) across {len(dirty)} notebook(s){note}:")
    for rel, n in dirty:
        print(f"  {n:4d}  {rel}")
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
