# Reader scripts

The few standalone command-line tools a reader runs, from the repository root
(`uv run python scripts/<name>.py`). These are programs you **execute**; the
importable library code the notebooks build on lives in [`utils/`](../utils).

- **`verify_installation.py`** — run this first, right after installing: it imports
  the core dependencies and reports whether CUDA, matplotlib, and Plotly are working.
- **`download_artifacts.py`** - verifies and installs the released registries, model files,
  predictions, and backtests for cached Chapter 11-20 execution.
- **`create_experiment.py`** - copies an installed read-only run log into a writable,
  `ML4T_OUTPUT_DIR`-isolated experiment.
- **`sync_notebooks.py`** — regenerates a notebook's `.ipynb` from its Jupytext `.py`
  source (or the reverse); pass `--check` to only report which pairs have drifted.

That is the whole directory, which is the point: everything here is something you
run. The checks CI enforces and the tools that maintain the committed notebooks
live in [`.github/scripts/`](../.github/scripts) — you never need to run them, and
you would only read them to see what a pull request has to satisfy.

> Internal registry-maintenance tooling (backfills, schema migrations, one-off data
> repairs) is intentionally **not** in this repository — it lives in the separate
> maintainer workspace.
