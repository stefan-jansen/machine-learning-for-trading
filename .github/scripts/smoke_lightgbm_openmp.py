#!/usr/bin/env python3
"""Fit a multithreaded LightGBM on the real import path, after a real module load.

    python .github/scripts/smoke_lightgbm_openmp.py

`check_openmp_import_order.py` reads source and is therefore only as good as its
list of imports that bring scikit-learn up. This runs the thing instead: it
imports `case_studies.utils.gbm`, which every case study's GBM stage imports and
which pulls in `ml4t.diagnostic` and so scikit-learn, and then asks LightGBM to
fit on more than one thread. That is the exact sequence that killed a reader's
kernel on macOS ARM64 (R2P issues #7 and #10).

On a macOS ARM64 runner a regression here is a segfault, which fails the step
with no explanation. So the check does not rely on the crash: it records which
OpenMP-owning library was imported first and fails with a readable message when
scikit-learn wins. That verdict is correct on Linux too, where the fit itself
would survive - which is the whole reason this class of defect reached a reader
with a green CI behind it.
"""

from __future__ import annotations

import importlib
import importlib.abc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

OWNS_OPENMP = ("lightgbm", "xgboost", "catboost", "sklearn")

_order: list[str] = []


class _Recorder(importlib.abc.MetaPathFinder):
    """Record the order in which OpenMP-owning packages are first imported."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102, ANN001
        top = fullname.split(".")[0]
        if top in OWNS_OPENMP and top not in _order:
            _order.append(top)
        return None  # never claims the import; only observes it


def main() -> int:
    sys.meta_path.insert(0, _Recorder())
    sys.path.insert(0, str(REPO))

    # The production import path, not a synthetic one.
    importlib.import_module("case_studies.utils.gbm")

    if not _order:
        print("no OpenMP-owning library was imported - the check is not exercising anything")
        return 1

    first = _order[0]
    if first == "sklearn":
        print(
            "scikit-learn's OpenMP runtime was initialized first "
            f"(import order: {_order}).\n"
            "On macOS ARM64 the next multithreaded LightGBM fit segfaults in\n"
            "__kmp_suspend_initialize_thread and takes the kernel with it. Import\n"
            "lightgbm above whatever brought scikit-learn up; run\n"
            ".github/scripts/check_openmp_import_order.py to find the file."
        )
        return 1

    # Multithreaded on purpose: the single-threaded path does not touch the
    # suspend/resume machinery that dies.
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, 8))
    y = rng.standard_normal(512)
    lgb.LGBMRegressor(n_estimators=20, n_jobs=4, verbose=-1).fit(X, y)

    print(
        f"OpenMP runtime ok: {first} initialized first (import order: {_order}); "
        "4-thread LightGBM fit completed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
