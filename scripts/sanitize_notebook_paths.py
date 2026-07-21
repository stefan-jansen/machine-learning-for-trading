"""Strip machine-specific absolute paths from committed notebook outputs.

Notebooks executed on a contributor's machine bake absolute paths
(``/home/<user>/ml4t/code/...``) into committed cell outputs and into the
``papermill`` execution metadata. Readers should never see those. This tool
rewrites them in place:

* repo-internal paths -> repo-relative (matching ``utils.paths.display_path``)
* anything else under ``~/ml4t`` -> a ``~``-prefixed generic path

It parses the ``.ipynb`` and only traverses notebook metadata, cell metadata,
and cell outputs. Cell source is never rewritten, including intentional Docker
paths such as ``/app``. The canonical one-space JSON format is preserved.

Idempotent: running twice is a no-op. A companion test
(``tests/test_notebook_output_hygiene.py``) fails CI if any leak survives.

Usage:
    uv run python scripts/sanitize_notebook_paths.py            # rewrite in place
    uv run python scripts/sanitize_notebook_paths.py --check    # report only, exit 1 if dirty
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Applied in order; earlier (longer) rules win. The two repo-root prefixes map
# to "" so paths become repo-relative. `third_edition/code` is a stale former
# layout that still lingers in some papermill metadata. The ``/home/<user>/``
# prefix is matched generically (not just one username) so any contributor's
# or CI runner's path (e.g. /home/runner/...) is sanitized and CI-guarded.
REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/home/[^/]+/ml4t/third_edition/code/"), ""),
    (re.compile(r"/home/[^/]+/ml4t/code/"), ""),
    # Docker container repo root: GPU notebooks (e.g. Ch12 02_gbm_comparison run
    # in the ml4t-gpu image) bake the container working dir /app into outputs.
    (re.compile(r"/app/"), ""),
    (re.compile(r"/home/[^/]+/ml4t/"), "~/ml4t/"),
    (re.compile(r"/home/[^/]+/"), "~/"),
]

SKIP_PARTS = {"_reference", ".venv", ".git"}
MACHINE_HOME_PATTERN = re.compile(r"/home/[^/]+/")


def _iter_notebooks() -> list[Path]:
    out = []
    for p in REPO_ROOT.rglob("*.ipynb"):
        if SKIP_PARTS & set(p.parts):
            continue
        if p.name.startswith("_executed_"):
            continue
        out.append(p)
    return sorted(out)


def sanitize_text(text: str) -> tuple[str, int]:
    n = 0
    for pat, new in REPLACEMENTS:
        text, k = pat.subn(new, text)
        n += k
    return text, n


def source_home_path_leaks(text: str) -> list[int]:
    """Return cell indexes containing a machine-specific home path in source."""
    notebook = json.loads(text)
    offenders = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        if MACHINE_HOME_PATTERN.search(source_text):
            offenders.append(index)
    return offenders


def _sanitize_value(value: object) -> tuple[object, int]:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        clean = []
        count = 0
        for item in value:
            new_item, item_count = _sanitize_value(item)
            clean.append(new_item)
            count += item_count
        return clean, count
    if isinstance(value, dict):
        clean = {}
        count = 0
        for key, item in value.items():
            new_item, item_count = _sanitize_value(item)
            clean[key] = new_item
            count += item_count
        return clean, count
    return value, 0


def sanitize_notebook_text(text: str) -> tuple[str, int]:
    """Sanitize notebook outputs and metadata without touching cell source."""
    notebook = json.loads(text)
    count = 0

    notebook["metadata"], item_count = _sanitize_value(notebook.get("metadata", {}))
    count += item_count
    for cell in notebook.get("cells", []):
        if "metadata" in cell:
            cell["metadata"], item_count = _sanitize_value(cell["metadata"])
            count += item_count
        if "outputs" in cell:
            cell["outputs"], item_count = _sanitize_value(cell["outputs"])
            count += item_count

    if not count:
        return text, 0
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if any leak found")
    args = ap.parse_args()

    source_offenders: list[tuple[Path, list[int]]] = []
    for nb in _iter_notebooks():
        indexes = source_home_path_leaks(nb.read_text(encoding="utf-8"))
        if indexes:
            source_offenders.append((nb.relative_to(REPO_ROOT), indexes))

    if source_offenders:
        print("machine-specific home paths require manual source edits:", file=sys.stderr)
        for rel, indexes in source_offenders:
            print(f"  {rel}: cells {indexes}", file=sys.stderr)
        return 2

    dirty: list[tuple[Path, int]] = []
    for nb in _iter_notebooks():
        raw = nb.read_text(encoding="utf-8")
        new, n = sanitize_notebook_text(raw)
        if n:
            dirty.append((nb.relative_to(REPO_ROOT), n))
            if not args.check:
                nb.write_text(new, encoding="utf-8")

    if not dirty:
        print("clean: no /home/<user> paths in any notebook")
        return 0

    verb = "would rewrite" if args.check else "rewrote"
    total = sum(n for _, n in dirty)
    print(f"{verb} {total} occurrence(s) across {len(dirty)} notebook(s):")
    for rel, n in dirty:
        print(f"  {n:4d}  {rel}")
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
