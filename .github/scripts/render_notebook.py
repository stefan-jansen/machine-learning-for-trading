#!/usr/bin/env python
"""Render an executed notebook to standalone HTML, keeping the figures' alt text.

    uv run .github/scripts/render_notebook.py case_studies/etfs/03_financial_features.ipynb \
        --out-dir ~/ml4t/review

`jupyter nbconvert --to html` uses the `lab` template, which reads width, height and
the background hints out of an output's `image/png` metadata and never reads `alt`.
`utils.style.show_with_alt` stores the description exactly where a renderer should
look for it, and `HTMLExporter` then replaces every missing alt with "No description
has been provided for this image". So every figure description written for this book
was accurate, stored correctly, and absent from the page.

This selects the repo's `ml4t` template, which is the `lab` template with those two
image blocks overridden, and reports how many images carry a real description. That
count is the check: a notebook whose figures all went through `show_with_alt` should
report zero missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_BASEDIR = REPO_ROOT / ".jupyter" / "nbconvert_templates"
TEMPLATE_NAME = "ml4t"
PLACEHOLDER = "No description has been provided for this image"


def render(notebook: Path, out_dir: Path, name: str | None) -> Path:
    import nbformat
    from nbconvert import HTMLExporter

    # Both go to the constructor. `extra_template_basedirs` is `affects_environment`,
    # and the template is resolved during __init__, so assigning it afterwards is too
    # late and the repo template is not found.
    exporter = HTMLExporter(
        template_name=TEMPLATE_NAME, extra_template_basedirs=[str(TEMPLATE_BASEDIR)]
    )

    body, _ = exporter.from_notebook_node(
        nbformat.read(notebook, as_version=4),
        resources={"metadata": {"path": str(notebook.parent)}},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name or notebook.stem}.html"
    out_path.write_text(body, encoding="utf-8")
    return out_path


def report_alt_coverage(html: str) -> tuple[int, int]:
    """Return (images carrying a description, images carrying none).

    An empty or whitespace-only ``alt`` counts as none. It reaches the page exactly as
    the placeholder does - a screen reader announces an unlabelled image either way -
    so counting it would let this report call the coverage complete while the figure
    is undescribed.
    """
    from bs4 import BeautifulSoup

    images = BeautifulSoup(html, features="html.parser").select("img")
    missing = sum(1 for img in images if img.attrs.get("alt", "").strip() in ("", PLACEHOLDER))
    return len(images) - missing, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "ml4t" / "review")
    parser.add_argument(
        "--name",
        help="output stem; defaults to the notebook's, so pass <case_study>_<notebook> "
        "when rendering nine notebooks that share a name into one directory",
    )
    args = parser.parse_args()

    out_path = render(args.notebook, args.out_dir.expanduser(), args.name)
    described, missing = report_alt_coverage(out_path.read_text(encoding="utf-8"))

    print(f"{out_path}")
    print(f"{described} image(s) carry a description, {missing} do not")
    if missing:
        print(
            f"The {missing} without one render as '{PLACEHOLDER}'. Every figure a "
            "notebook shows should go through utils.style.show_with_alt.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
