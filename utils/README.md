# Shared utilities

Library code imported across every chapter and case study. Notebooks run from the
repository root, so `import utils` and `from utils.x import y` resolve with no
installation. This is code you **import**, not run — for command-line tools, see
[`scripts/`](../scripts).

**Configuration and paths.** `config.py` loads and validates the paths and settings
in `.env` (and sorts out CUDA library paths); `paths.py` holds the chapter registry
and resolves chapter, case-study, and output directories so notebooks never hard-code
a location.

**Figures.** `style.py` defines the ML4T color palette and the matplotlib / Plotly
defaults that give every figure in the book one consistent look.

**Data.** `data_quality.py` summarizes coverage, checks OHLC invariants, and subsets
symbols for fast test runs; `downloading.py` is the shared backbone of the `data/`
download scripts (argument parsing, path/YAML resolution, atomic writes);
`artifact_specs.py` loads the per-case-study YAML sidecars that describe market data,
labels, and features.

**Modeling and cross-validation.** `modeling.py` is the workhorse — it loads a
modeling dataset, parses model configs, prepares folds, and detects the schema;
`cv_splits.py` builds the walk-forward splits (calendar-aware, leakage-safe);
`predictions_cache.py` caches long-form prediction frames so the teaching notebooks
don't recompute them.

**Reproducibility.** `reproducibility.py` seeds Python, NumPy, and Torch (CPU + CUDA)
in a single call; `storage_benchmarks.py` provides the synthetic data and timing
harness behind the Chapter 2 storage benchmarks.
