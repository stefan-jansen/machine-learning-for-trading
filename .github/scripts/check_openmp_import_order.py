#!/usr/bin/env python3
"""Fail when a module imports scikit-learn before LightGBM or XGBoost.

    python .github/scripts/check_openmp_import_order.py

scikit-learn, LightGBM, XGBoost and torch each ship their own OpenMP runtime.
The first one loaded wins for the whole process. On macOS ARM64 a reader who
gets scikit-learn's `libomp` first, and then asks LightGBM to fit with more
than one thread, loses the kernel to a segfault inside
`__kmp_suspend_initialize_thread` - no Python traceback, just a dead kernel.
It cost a student the better part of a day (see R2P issues #7 and #10), and it
reproduces on no Linux runner, so nothing in CI saw it.

The ordering is easy to restore and just as easy to lose again: it is invisible
in review, and it is the kind of thing an editor's "organize imports" undoes.
Hence this gate.

It reads the paired `.py` of every notebook, so a notebook is covered by its
own source rather than by parsing JSON. `import` statements sort ahead of
`from ... import` ones under isort, so the fix is usually to merge two import
blocks into one canonical block rather than to add a `# noqa` or a manual
ordering an autoformatter will fight.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The runtime that must be initialized first, and the ones that must not precede it.
GBM = {"lightgbm", "xgboost"}
BLAS_OMP = {"sklearn"}

SKIP_PARTS = {".venv", ".git", ".ipynb_checkpoints", "_reference", "node_modules"}


def first_import_lines(path: Path) -> dict[str, int]:
    """Line number of the first import of each watched package."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return {}

    seen: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in GBM | BLAS_OMP and top not in seen:
                seen[top] = node.lineno
    return seen


def main() -> int:
    offenders: list[tuple[Path, str, int, str, int]] = []
    checked = 0

    for path in sorted(REPO.rglob("*.py")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        seen = first_import_lines(path)
        gbm = {k: v for k, v in seen.items() if k in GBM}
        omp = {k: v for k, v in seen.items() if k in BLAS_OMP}
        if not gbm or not omp:
            continue
        checked += 1
        first_gbm_pkg, first_gbm_line = min(gbm.items(), key=lambda kv: kv[1])
        first_omp_pkg, first_omp_line = min(omp.items(), key=lambda kv: kv[1])
        if first_omp_line < first_gbm_line:
            offenders.append(
                (
                    path.relative_to(REPO),
                    first_omp_pkg,
                    first_omp_line,
                    first_gbm_pkg,
                    first_gbm_line,
                )
            )

    if offenders:
        print("OpenMP import order (macOS ARM64 kernel death):\n")
        for rel, omp_pkg, omp_line, gbm_pkg, gbm_line in offenders:
            print(f"  {rel}")
            print(f"    {omp_pkg} imported at line {omp_line}, before {gbm_pkg} at line {gbm_line}")
        print(
            f"\n{len(offenders)} of {checked} module(s) that use both.\n"
            "Move the lightgbm/xgboost import above the scikit-learn one. Merging the two\n"
            "import blocks into a single isort-canonical block is usually enough, because\n"
            "plain `import x` sorts ahead of `from x import y`; check with\n"
            "`ruff check --select I --diff <file>` that the order is stable."
        )
        return 1

    print(f"OpenMP import order ok ({checked} modules import both a GBM and scikit-learn).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
