"""Strip per-cell ``papermill`` execution metadata from committed notebooks.

Papermill records timing on every cell it executes::

    "papermill": {"duration": 0.0077, "end_time": "...", "exception": false,
                  "start_time": "...", "status": "completed"}

Notebooks whose jupytext config does not set a ``cell_metadata_filter`` (173 of
them here — the repo is inconsistent) round-trip that block into the paired
``.py``. The committed ``.py`` does not carry it, so the pair disagrees and
JupyterLab's jupytext contents manager refuses to open the notebook: the reader
gets a ``File Load Error`` dialog instead (public issue #372, cause B).

This is the same class as the empty ``tags: []`` fossil handled by
``strip_empty_cell_tags.py``, but a different key and a different reason for
being tracked. Timing of a past run carries no information for a reader, so
removing it is lossless.

**Notebook-level** ``metadata.papermill`` is NOT touched; only per-cell blocks are
removed. ``notebook_provenance.py`` no longer decides production-vs-TEST from
``metadata.papermill.parameters`` — that key is a fossil, and the executor now
declares the parameters instead — but it does rewrite the key when it stamps, and
rewriting it here as well would mean two tools racing over one value.

Unlike ``strip_empty_cell_tags.py`` this reserializes the JSON rather than
editing raw text, because the block is nested and regex-matching nested objects
is fragile. That is safe only where a no-op round-trip is byte-identical (not
true for every notebook in this repo), so each file is checked first and skipped
with a warning if reserializing would introduce unrelated diff noise.

Only notebooks whose pair **actually disagrees** are touched. Most notebooks here
carry this fossil harmlessly: their ``cell_metadata_filter`` already excludes it
from the round-trip, so they open fine, and rewriting 300+ of them would be a
large diff with no reader benefit. The invariant is that the pair agrees, not
that the fossil is gone.

Idempotent. Guarded by ``tests/test_notebook_output_hygiene.py``.

Usage:
    uv run python .github/scripts/strip_papermill_cell_metadata.py            # rewrite in place
    uv run python .github/scripts/strip_papermill_cell_metadata.py a.ipynb    # selected notebooks
    uv run python .github/scripts/strip_papermill_cell_metadata.py --check    # report only, exit 1 if dirty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from check_notebook_pair_sync import _tracked_notebooks, pair_diff
from sanitize_notebook_paths import REPO_ROOT


def _dump(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def roundtrip_is_lossless(raw: str) -> bool:
    """True if reserializing this notebook changes nothing but what we remove."""
    return _dump(json.loads(raw)) == raw


def strip_cell_papermill(raw: str) -> tuple[str, int]:
    """Remove per-cell papermill blocks. Returns (new_text, cells_changed)."""
    nb = json.loads(raw)
    n = 0
    for cell in nb.get("cells", []):
        if "papermill" in cell.get("metadata", {}):
            del cell["metadata"]["papermill"]
            n += 1
    return (_dump(nb) if n else raw), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if any found")
    ap.add_argument("paths", nargs="*", type=Path, help="notebooks to inspect (default: all)")
    args = ap.parse_args()

    notebooks = (
        [path.resolve().with_suffix(".ipynb") for path in args.paths]
        if args.paths
        else _tracked_notebooks()
    )

    dirty: list[tuple[Path, int]] = []
    unsafe: list[Path] = []
    for nb_path in notebooks:
        if not nb_path.exists():
            raise SystemExit(f"notebook not found: {nb_path}")
        if not pair_diff(nb_path):
            continue  # pair already agrees; the fossil is harmless here
        raw = nb_path.read_text(encoding="utf-8")
        new, n = strip_cell_papermill(raw)
        if not n:
            continue
        if not roundtrip_is_lossless(raw):
            unsafe.append(nb_path.relative_to(REPO_ROOT))
            continue
        dirty.append((nb_path.relative_to(REPO_ROOT), n))
        if not args.check:
            nb_path.write_text(new, encoding="utf-8")

    for rel in unsafe:
        print(f"SKIPPED (reserializing would add unrelated diff noise): {rel}")

    if not dirty:
        print("clean: no per-cell papermill metadata in any notebook")
        return 1 if unsafe else 0

    verb = "would strip" if args.check else "stripped"
    total = sum(n for _, n in dirty)
    print(f"{verb} per-cell papermill from {total} cell(s) across {len(dirty)} notebook(s):")
    for rel, n in dirty:
        print(f"  {n:4d}  {rel}")
    return 1 if (args.check or unsafe) else 0


if __name__ == "__main__":
    sys.exit(main())
