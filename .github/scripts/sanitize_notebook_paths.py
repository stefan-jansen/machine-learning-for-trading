"""Strip machine-specific absolute paths from committed notebook outputs.

Notebooks executed on a contributor's machine bake absolute paths
(``/home/<user>/ml4t/code/...``) into committed cell outputs and into the
``papermill`` execution metadata. Readers should never see those. This tool
rewrites them in place:

* repo-internal paths -> repo-relative (matching ``utils.paths.display_path``)
* anything else under ``~/ml4t`` -> a ``~``-prefixed generic path

Outputs and metadata only; ``source`` is never touched
------------------------------------------------------

A notebook's ``source`` is code the reader runs, and a path in it can be load
bearing. An earlier version of this script rewrote the raw ``.ipynb`` text and so
rewrote source along with everything else: it turned the real Docker mount path
in ``02_financial_data_universe/16_provider_comparison`` into a relative path.
That is why the companion test was deselected in CI rather than fixed.

So the notebook is parsed, and only strings reachable through notebook metadata,
cell metadata and cell outputs are candidates. Two strings in this repository's
source survive precisely because of that: the mount path above, and a comment in
``12_gradient_boosting/02_gbm_comparison`` that names ``/app/`` to explain it.

The rewrite itself is still done on the raw text, one substring at a time, so the
diff is the replaced paths and nothing else - no JSON reserialization, no
reflowed formatting, no reordered keys. A string that also occurs inside the
notebook's source is skipped and reported instead of replaced, because a raw-text
edit cannot tell the two occurrences apart. Nothing in the repository trips that
today; the check is here so the guarantee holds without being re-verified by hand.

Idempotent: running twice is a no-op. A companion test
(``tests/test_notebook_output_hygiene.py``) fails CI if any leak survives.

Usage:
    uv run python .github/scripts/sanitize_notebook_paths.py            # rewrite in place
    uv run python .github/scripts/sanitize_notebook_paths.py --check    # report only, exit 1 if dirty
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    """Apply the path rules to one string. Says nothing about where it came from."""
    n = 0
    for pat, new in REPLACEMENTS:
        text, k = pat.subn(new, text)
        n += k
    return text, n


def _strings(obj: object, out: list[str]) -> None:
    """Every string reachable from ``obj``."""
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            _strings(item, out)
    elif isinstance(obj, dict):
        for value in obj.values():
            _strings(value, out)


def _partition(nb: dict) -> tuple[list[str], list[str]]:
    """(strings this tool may rewrite, strings it must leave alone).

    The second list is the notebook's ``source``. Everything in the first is
    reachable only through notebook metadata, cell metadata or cell outputs.
    """
    rewritable: list[str] = []
    protected: list[str] = []
    _strings(nb.get("metadata", {}), rewritable)
    for cell in nb.get("cells", []):
        _strings(cell.get("metadata", {}), rewritable)
        _strings(cell.get("outputs", []), rewritable)
        _strings(cell.get("source", []), protected)
    return rewritable, protected


def _encoded(text: str) -> str:
    """``text`` as it appears inside a JSON string literal, without the quotes."""
    return json.dumps(text)[1:-1]


def sanitize_notebook(raw: str) -> tuple[str, int, list[str]]:
    """Rewrite the machine-specific paths in one notebook's outputs and metadata.

    Args:
        raw: the notebook file's text.

    Returns:
        (new text, paths rewritten, strings skipped because the notebook's
        ``source`` contains them too).
    """
    rewritable, protected = _partition(json.loads(raw))
    protected_encoded = [_encoded(s) for s in protected]

    targets = {value: sanitize_text(value) for value in rewritable}
    targets = {value: cleaned for value, (cleaned, n) in targets.items() if n}

    replaced = 0
    skipped: list[str] = []
    # Longest first: one leaked string is often a substring of another, and
    # rewriting the shorter one first would leave the longer one unfindable.
    for original in sorted(targets, key=len, reverse=True):
        encoded = _encoded(original)
        if any(encoded in candidate for candidate in protected_encoded):
            skipped.append(original)
            continue
        occurrences = raw.count(encoded)
        raw = raw.replace(encoded, _encoded(targets[original]))
        replaced += occurrences * sanitize_text(original)[1]
    return raw, replaced, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if any leak found")
    args = ap.parse_args()

    dirty: list[tuple[Path, int]] = []
    blocked: list[tuple[Path, str]] = []
    for nb in _iter_notebooks():
        raw = nb.read_text(encoding="utf-8")
        new, n, skipped = sanitize_notebook(raw)
        blocked += [(nb.relative_to(REPO_ROOT), s) for s in skipped]
        if n:
            dirty.append((nb.relative_to(REPO_ROOT), n))
            if not args.check:
                nb.write_text(new, encoding="utf-8")

    if blocked:
        print("NOT rewritten - the same string appears in the notebook's source.")
        print("Remove it by hand or re-execute the notebook; this tool cannot tell the")
        print("two occurrences apart:")
        for rel, s in blocked:
            print(f"  {rel}\n    {s}")

    if dirty:
        verb = "would rewrite" if args.check else "rewrote"
        total = sum(n for _, n in dirty)
        print(f"{verb} {total} occurrence(s) across {len(dirty)} notebook(s):")
        for rel, n in dirty:
            print(f"  {n:4d}  {rel}")

    if not dirty and not blocked:
        print("clean: no machine-specific paths in any notebook's outputs or metadata")

    return 1 if blocked or (args.check and dirty) else 0


if __name__ == "__main__":
    sys.exit(main())
