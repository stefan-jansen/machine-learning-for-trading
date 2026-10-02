#!/usr/bin/env python3
"""Strip the embedded Plotly figure JSON from committed notebooks.

Plotly's notebook renderer serializes every data point of every trace into the
``.ipynb`` under ``application/vnd.plotly.v1+json``, beside a static ``image/png``
of the same figure. The JSON is what makes a chart interactive in a live kernel;
the PNG is what every static reader shows, GitHub included.

That costs a reader real time. ``04_fundamental_alternative_data/01_academic_characteristics.ipynb``
is 25.3 MB, of which 0.03 MB is source and 26.1 MB (pre-compaction) is Plotly JSON.
Measured 2026-10-01: opening it on github.com left the content pane empty for over
twenty seconds. An Amazon reviewer described the notebooks as impossible to read,
and this is the part of that complaint with something behind it - the median
notebook here is 37 cells and 429 lines of code, which is not large.

Across the repository: 654 Plotly outputs in 203 notebooks, **all 654 carrying an
``image/png`` fallback**, 0.11 GB of JSON in total.

What is lost and what is not
----------------------------
Lost: hover, zoom and legend toggling for a reader browsing the committed notebook
on GitHub without running it. Not lost: the figure itself, which the PNG shows, and
not the interactivity for anyone who runs the notebook, because executing the cell
rebuilds the figure from the data.

So this never strips a Plotly output that has no static fallback - that would delete
the only copy of the figure. Such an output is skipped and reported, and the file is
left alone.

Why this is not a re-run
------------------------
``outputs_digest`` in ``notebook_provenance.py`` hashes the outputs, so a strip reads
as stale. Re-executing is the wrong answer and not merely an expensive one: a re-run
writes the Plotly JSON straight back, so the gate would be demanding the one action
that undoes the fix. That is the argument ``sanitize_notebook_paths.py`` makes for a
machine path in an output, and ``VOLATILE_OUTPUT_KEYS`` makes for figure ``alt``.
Fold a strip in with::

    uv run python .github/scripts/notebook_provenance.py sync-plotly <nb.ipynb>

which re-derives the strip from the committed notebook and refuses anything else.

    strip_plotly_json.py            # rewrite every tracked notebook in place
    strip_plotly_json.py a.ipynb    # selected notebooks
    strip_plotly_json.py --check    # report only, exit 1 if any would change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sanitize_notebook_paths import REPO_ROOT, _iter_notebooks  # noqa: E402

PLOTLY_MIME = "application/vnd.plotly.v1+json"
# A reader of the committed notebook sees one of these instead. SVG counts: it is a
# static rendering of the same figure, not a second interactive copy.
STATIC_FALLBACKS = frozenset({"image/png", "image/jpeg", "image/svg+xml"})


def strip_plotly(raw: str) -> tuple[str, int, int]:
    """Return (rewritten JSON text, outputs stripped, outputs left alone).

    Mirrors ``sanitize_notebook``'s signature so the gate can re-derive the rewrite
    from the committed text the same way. Reserializes rather than editing raw text,
    because the payload is a deeply nested object and regex-matching one is fragile.
    """
    nb = json.loads(raw)
    stripped = skipped = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs") or []:
            data = output.get("data")
            if not isinstance(data, dict) or PLOTLY_MIME not in data:
                continue
            if STATIC_FALLBACKS & set(data):
                del data[PLOTLY_MIME]
                stripped += 1
            else:
                # The only copy of this figure. Removing it would delete the chart.
                skipped += 1
    if not stripped:
        return raw, 0, skipped
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n", stripped, skipped


def _display(path: Path) -> str:
    """Repo-relative where that is meaningful, absolute otherwise.

    A notebook outside the repository is a legitimate argument - a scratch copy, a
    file under review - and must not crash the run just to print its name.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _process(path: Path, *, check: bool) -> tuple[bool, int, int]:
    raw = path.read_text(encoding="utf-8")
    new, stripped, skipped = strip_plotly(raw)
    if stripped and not check:
        path.write_text(new, encoding="utf-8")
    return bool(stripped), stripped, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="*", type=Path)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if any would change")
    args = ap.parse_args()

    # Resolve before anything prints a path. A relative argument - which is how anyone
    # types one - reached `relative_to(REPO_ROOT)` unresolved and raised ValueError, but
    # only once there was something to report, so a run over an already-clean tree hid it.
    paths = [p.resolve() for p in args.notebooks] or _iter_notebooks()
    dirty: list[Path] = []
    total_stripped = total_skipped = 0
    saved = 0
    for path in paths:
        before = path.stat().st_size
        changed, stripped, skipped = _process(path, check=args.check)
        total_stripped += stripped
        total_skipped += skipped
        if skipped:
            print(
                f"{_display(path)}: {skipped} Plotly output(s) with no static fallback, left alone"
            )
        if changed:
            dirty.append(path)
            saved += before - (before if args.check else path.stat().st_size)

    if not dirty:
        print(f"no embedded Plotly JSON to strip ({total_skipped} output(s) had no fallback)")
        return 0

    verb = "would strip" if args.check else "stripped"
    print(f"{verb} {total_stripped} Plotly output(s) from {len(dirty)} notebook(s)")
    if not args.check:
        print(f"{saved / 1e6:.1f} MB removed")
    for path in dirty:
        print(f"  {_display(path)}")
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
