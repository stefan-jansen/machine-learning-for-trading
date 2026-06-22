"""Strip machine-specific absolute paths from committed notebook outputs.

Notebooks executed on a contributor's machine bake absolute paths
(``/home/<user>/ml4t/code/...``) into committed cell outputs and into the
``papermill`` execution metadata. Readers should never see those. This tool
rewrites them in place:

* repo-internal paths -> repo-relative (matching ``utils.paths.display_path``)
* anything else under ``~/ml4t`` -> a ``~``-prefixed generic path

It edits the raw ``.ipynb`` text (no JSON reserialization) so the only diff is
the replaced substrings — formatting, key order and outputs are untouched.

Idempotent: running twice is a no-op. A companion test
(``tests/test_notebook_output_hygiene.py``) fails CI if any leak survives.

Usage:
    uv run python scripts/sanitize_notebook_paths.py            # rewrite in place
    uv run python scripts/sanitize_notebook_paths.py --check    # report only, exit 1 if dirty
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if any leak found")
    args = ap.parse_args()

    dirty: list[tuple[Path, int]] = []
    for nb in _iter_notebooks():
        raw = nb.read_text(encoding="utf-8")
        new, n = sanitize_text(raw)
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
