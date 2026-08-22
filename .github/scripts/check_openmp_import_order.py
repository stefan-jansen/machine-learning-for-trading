#!/usr/bin/env python3
"""Fail when a module loads scikit-learn before a gradient-boosting library.

    python .github/scripts/check_openmp_import_order.py

scikit-learn, LightGBM, XGBoost, CatBoost and torch each ship their own OpenMP
runtime. The first one loaded wins for the whole process. On macOS ARM64 a
reader who gets scikit-learn's `libomp` first, and then asks LightGBM to fit
with more than one thread, loses the kernel to a segfault inside
`__kmp_suspend_initialize_thread` - no Python traceback, just a dead kernel.
It cost a student the better part of a day (see R2P issues #7 and #10), and it
reproduces on no Linux runner, so nothing in CI saw it.

The ordering is easy to restore and just as easy to lose again: it is invisible
in review, and it is the kind of thing an editor's "organize imports" undoes.
Hence this gate.

Neither side of the comparison is spelled the way it reads. `ml4t.diagnostic`
and `shap` pull scikit-learn in transitively, so a module whose only
scikit-learn contact is `from ml4t.diagnostic.metrics import ...` has the same
defect and shows nothing at the call site; and `case_studies.utils.gbm` and
`case_studies.utils.model_analysis` import lightgbm at module scope, so they
stand in for it on the other side. The first version of this gate watched the
`sklearn` and `lightgbm` names alone and passed eleven affected modules, among
them `case_studies/utils/gbm.py`, which every case study's GBM stage imports.
See BLAS_OMP and GBM_VIA below.

A deferred import does not help either: a function-local `import lightgbm` runs
long after the module's own header has already brought scikit-learn up, which
is why the fix is a module-level import even where nothing at module scope uses
the name.

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

# The libraries that bring their own OpenMP runtime and must initialize first.
GBM = {"lightgbm", "xgboost", "catboost"}

# Repo modules that import one of the above at module scope, and so stand in
# for it. Both are shared helpers on the production path - `case_studies.utils.
# gbm` is imported by every case study's GBM stage - so a notebook can be
# affected without naming a GBM library anywhere. Keep in step with the
# unused module-level lightgbm import in each, which is there for this reason
# and marked with an F401 suppression.
GBM_VIA = ("case_studies.utils.gbm", "case_studies.utils.model_analysis")

# The imports that bring up scikit-learn's OpenMP runtime, and so must not come
# first. `sklearn` is the obvious one; the other two are here because they
# import it transitively, which is invisible at the call site and is what the
# first version of this gate missed. The list is measured, not guessed -
# re-derive it by importing a candidate in a fresh interpreter and checking
# whether an OpenMP runtime appears:
#
#     python -c "import <pkg>, re; \
#         print([l for l in open('/proc/self/maps') if re.search(r'lib(omp|gomp|iomp5)', l)])"
#
# `econml`, `optuna`, `scipy` and `statsmodels` were checked the same way and
# load none at import time, so they are deliberately absent.
BLAS_OMP = ("sklearn", "ml4t.diagnostic", "shap")

SKIP_PARTS = {".venv", ".git", ".ipynb_checkpoints", "_reference", "node_modules"}

# Affected, and blocked on something other than the import order itself.
#
# An earlier revision of this list held eleven entries on one shared excuse: that
# editing a notebook's `.py` obliges a re-execution (notebook_provenance.py
# enforces that), and re-execution restates whatever the `~/ml4t/code` artifacts
# now produce. That is true, and it is not a reason to leave a crash in place.
# Six of the eleven were fixed and re-executed instead. What the re-run moved was
# prose that had hard-coded a third decimal place, and where a notebook's own
# commentary quoted values it no longer printed, the commentary was rewritten to
# state the ordering it was actually arguing for.
#
# So an entry here is not "we would rather not re-run this". It names a blocker
# that a re-execution cannot clear:
#
#   - the *book* quotes the notebook's figures, so restating them is an errata
#     decision about the manuscript, taken deliberately and not as a side effect
#     of an import fix;
#   - or the notebook cannot be executed in this environment at all.
#
# Every entry is printed on every run, and an entry that stops offending fails
# the gate, so the list cannot rot into a place where things are quietly parked.
#
# The copies students actually run carry none of this: the course repo's
# vendored notebooks ship without a paired `.py`, so an import-cell edit fixes
# the crash there and leaves every output alone. All fourteen are already clean
# in that repo.
#
# Revised 2026-08-08.
KNOWN_BLOCKED = {
    "11_ml_pipeline/07_case_study_insights.py": (
        "does not execute at all today: it calls insight_chapter.plot_rolling_daily_ic with a "
        "selected_prediction_hashes= argument that function has never accepted (line 678, "
        "TypeError). Predates this gate; reads registries and fits nothing, so it cannot crash"
    ),
    "12_gradient_boosting/02_gbm_comparison.py": (
        "Section 12.2 quotes its library timings and speedup ratios throughout, and a live "
        "worktree (group/ch11-12) is mid-edit on it; fits a GBM, so it can still crash"
    ),
    "12_gradient_boosting/12_case_study_insights.py": (
        "pins an approved etfs registry SHA-256 that the local registry no longer matches, "
        "so it cannot be executed here; reads registries and fits nothing, so it cannot crash"
    ),
}


def first_import_lines(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Line of the first GBM import and of the first scikit-learn-loading import."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return {}, {}

    gbm: dict[str, int] = {}
    omp: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.split(".")[0] in GBM:
                gbm.setdefault(name.split(".")[0], node.lineno)
            for pkg in GBM_VIA:
                if name == pkg or name.startswith(pkg + "."):
                    gbm.setdefault(pkg, node.lineno)
            for pkg in BLAS_OMP:
                if name == pkg or name.startswith(pkg + "."):
                    omp.setdefault(pkg, node.lineno)
    return gbm, omp


def main() -> int:
    offenders: list[tuple[str, str, int, str, int]] = []
    blocked: list[tuple[str, str, int, str, int]] = []
    checked = 0

    for path in sorted(REPO.rglob("*.py")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        gbm, omp = first_import_lines(path)
        if not gbm or not omp:
            continue
        checked += 1
        first_gbm_pkg, first_gbm_line = min(gbm.items(), key=lambda kv: kv[1])
        first_omp_pkg, first_omp_line = min(omp.items(), key=lambda kv: kv[1])
        if first_omp_line >= first_gbm_line:
            continue
        rel = str(path.relative_to(REPO))
        row = (rel, first_omp_pkg, first_omp_line, first_gbm_pkg, first_gbm_line)
        (blocked if rel in KNOWN_BLOCKED else offenders).append(row)

    # A file that was listed and has since been fixed should not stay listed:
    # a stale entry is how a gate quietly stops covering something.
    still_bad = {row[0] for row in blocked}
    resolved = sorted(set(KNOWN_BLOCKED) - still_bad)

    if blocked:
        print("Known-blocked, reported but not failing:\n")
        for rel, omp_pkg, omp_line, gbm_pkg, gbm_line in blocked:
            print(f"  {rel}")
            print(f"    {omp_pkg} at line {omp_line}, before {gbm_pkg} at line {gbm_line}")
            print(f"    blocked: {KNOWN_BLOCKED[rel]}")
        print()

    if resolved:
        print(
            "These are in KNOWN_BLOCKED but no longer offend. Fix the import order's\n"
            "record by deleting their entries in this file:\n"
            + "".join(f"  {rel}\n" for rel in resolved)
        )
        return 1

    if offenders:
        print("OpenMP import order (macOS ARM64 kernel death):\n")
        for rel, omp_pkg, omp_line, gbm_pkg, gbm_line in offenders:
            print(f"  {rel}")
            print(f"    {omp_pkg} imported at line {omp_line}, before {gbm_pkg} at line {gbm_line}")
        print(
            f"\n{len(offenders)} of {checked} module(s) that use both.\n"
            "Move the GBM import above the one named on the left. Merging the two import\n"
            "blocks into a single isort-canonical block is usually enough, because plain\n"
            "`import x` sorts ahead of `from x import y`; check with\n"
            "`ruff check --select I --diff <file>` that the order is stable.\n"
            "If the GBM is only used inside a function, add a module-level\n"
            "`import lightgbm  # noqa: F401` anyway - a deferred import runs too late."
        )
        return 1

    print(
        f"OpenMP import order ok ({checked} modules import both a GBM and scikit-learn"
        f"; {len(blocked)} known-blocked)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
