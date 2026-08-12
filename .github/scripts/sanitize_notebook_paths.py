"""Strip machine-specific absolute paths from committed notebook outputs.

Notebooks executed on a contributor's machine bake absolute paths
(``/home/<user>/ml4t/code/...``) into committed cell outputs and into the
``papermill`` execution metadata. Readers should never see those. This tool
rewrites them in place:

* repo-internal paths -> repo-relative (matching ``utils.paths.display_path``)
* anything else under ``~/ml4t`` -> a ``~``-prefixed generic path
* a scratch root under ``/tmp`` -> ``~/scratch/``, except the two shapes that are
  not machine-specific (see ``REPLACEMENTS``)

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

Image payloads are outputs and are never rewritten
--------------------------------------------------

A figure is stored in its cell's outputs as base64, and base64's alphabet
includes ``/``, ``a`` and ``p``, so a long enough payload contains ``/app/`` by
chance - roughly once per quarter-megabyte. Treating that as a Docker mount path
and deleting it removes five characters from the encoding, which both corrupts the
image and leaves a length no longer divisible by four, so the payload stops
decoding at all and the rendered page shows a broken figure. Measured on
``case_studies/etfs/05_evaluation``: one 244,684-character PNG, one chance
occurrence, and the result did not decode.

Nothing announces this. The notebook still parses, the cell still has an output,
and the only symptom is an image that fails to draw. So binary payloads are held
out of the candidate set by MIME type, and ``_assert_binary_intact`` re-reads the
rewritten text and compares every one of them byte for byte before it is returned.

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
    # Scratch roots under /tmp. Two of the four shapes found under /tmp are leaks
    # and two are not, so the rules below are anchored and the two exemptions are
    # excluded by name rather than left to ordering:
    #
    # rewritten - an agent session's scratchpad, whose path carries a user id, a
    #   working-directory slug and a session uuid, and a staging notebook or
    #   output directory a maintainer executed from /tmp, whose name is
    #   per-session and so has no stable replacement;
    # left alone - `/tmp/ml4t-test-output...`, which is the documented test
    #   output directory (`AGENTS.md`, "Output isolation"), so a notebook
    #   printing it is showing real configuration, and `/tmp/ipykernel_<pid>/`,
    #   which is how IPython names a cell in every user's kernel rather than one
    #   machine's layout, and which appears inside tracebacks a reader may need
    #   to follow.
    #
    # `(?<![\w.~-])` requires the match to be a filesystem root: without it the
    # `/tmp/` inside an already-rewritten `~/.claude/jobs/<id>/tmp/run.ipynb`
    # would be rewritten a second time, splicing a `~` into the middle of a path.
    # The anchor is not what keeps these rules out of a base64 payload - `+` and
    # `/` are both in that alphabet and both satisfy it. What does that is the
    # MIME filter above, which every rule in this list depends on: `t`, `m` and
    # `p` are as much a part of base64 as `a` and `p` are.
    (re.compile(r"(?<![\w.~-])/tmp/claude-\d+/[^/\s]+/[0-9a-fA-F-]{36}/scratchpad/"), "~/scratch/"),
    (re.compile(r"(?<![\w.~-])/tmp/claude-\d+/"), "~/scratch/"),
    (re.compile(r"(?<![\w.~-])/tmp/(?!ml4t-test-output|ipykernel_)"), "~/scratch/"),
]

SKIP_PARTS = {"_reference", ".venv", ".git"}

# The output MIME types nbformat stores base64-encoded rather than as text. A path
# rule that fires inside one of these has matched the encoding, not a path.
BINARY_MIME = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "application/pdf",
    }
)


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


def _binary_payloads(nb: dict) -> list[str]:
    """Every base64 output payload in the notebook, in cell order."""
    out: list[str] = []
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data") or {}
            for mime in BINARY_MIME & data.keys():
                payload = data[mime]
                out.append(payload if isinstance(payload, str) else "".join(payload))
    return out


def _output_strings(outputs: list, into: list[str]) -> None:
    """Strings from one cell's outputs, minus the base64 image payloads.

    A payload is skipped by MIME rather than by looking at the string, because
    what makes it ineligible is that it is an encoding rather than prose - not
    that it happens to be long or to look like base64.
    """
    for output in outputs:
        for key, value in output.items():
            if key != "data":
                _strings(value, into)
                continue
            for mime, payload in value.items():
                if mime not in BINARY_MIME:
                    _strings(payload, into)


def _partition(nb: dict) -> tuple[list[str], list[str]]:
    """(strings this tool may rewrite, strings it must leave alone).

    The second list is the notebook's ``source``. Everything in the first is
    reachable only through notebook metadata, cell metadata or cell outputs, and
    excludes the base64 image payloads for the reason given at the top of this file.
    """
    rewritable: list[str] = []
    protected: list[str] = []
    _strings(nb.get("metadata", {}), rewritable)
    for cell in nb.get("cells", []):
        _strings(cell.get("metadata", {}), rewritable)
        _output_strings(cell.get("outputs", []), rewritable)
        _strings(cell.get("source", []), protected)
    return rewritable, protected


def _assert_binary_intact(before: dict, raw: str) -> None:
    """Every base64 payload survives the rewrite unchanged, or nothing is written.

    The MIME filter above is what keeps payloads out of the candidate set. This is
    the check that the filter worked, and it runs on the rewritten text rather than
    on the plan, so a future rule that reaches a payload some other way is caught
    here instead of in a rendered page.
    """
    after = _binary_payloads(json.loads(raw))
    expected = _binary_payloads(before)
    if after != expected:
        moved = sum(1 for a, b in zip(expected, after, strict=False) if a != b)
        raise AssertionError(
            f"the rewrite changed {moved} base64 output payload(s): a path rule matched "
            "inside an encoding rather than inside a path, which corrupts the figure"
        )


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
    parsed = json.loads(raw)
    rewritable, protected = _partition(parsed)
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
    _assert_binary_intact(parsed, raw)
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
