# Reader scripts

The few standalone command-line tools a reader runs, from the repository root
(`uv run python scripts/<name>.py`). These are programs you **execute**; the
importable library code the notebooks build on lives in [`utils/`](../utils).

- **`verify_installation.py`** — run this first, right after installing: it imports
  the core dependencies and reports whether CUDA, matplotlib, and Plotly are working.
- **`download_artifacts.py`** — pulls the pre-computed model predictions and backtest
  results so the Chapter 11–20 case-study notebooks run without retraining from scratch.
- **`sync_notebooks.py`** — regenerates a notebook's `.ipynb` from its Jupytext `.py`
  source (or the reverse); pass `--check` to only report which pairs have drifted.
- **`sanitize_notebook_paths.py`** — strips machine-specific absolute paths out of
  committed notebook outputs and metadata; a CI check fails if any slip through.
- **`notebook_provenance.py`** — the pre-commit gate: it stamps each executed notebook
  with the git hash of its `.py` source and blocks commits of stale or test-mode runs.

> Internal registry-maintenance tooling (backfills, schema migrations, one-off data
> repairs) is intentionally **not** in this repository — it lives in the separate
> maintainer workspace.
