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

That is the whole directory, which is the point: everything here is something a
reader runs. The checks CI enforces and the tools that repair committed notebooks
live in [`.github/scripts/`](../.github/scripts). Reading the book needs none of
them; opening a pull request can, and the failure tells you which:

- `notebook_provenance.py stamp <nb.ipynb> --executor <env> --production` — re-stamps
  a notebook. Pass `--parameters '{"MAX_SYMBOLS": 5}'` instead of `--production` when
  the run did use overrides; one of the two is required, because the notebook's own
  `metadata.papermill.parameters` can outlive the run it describes. The pre-commit
  gate fails a stamped notebook whose `.py` source has moved on since, any notebook
  committed from a test-mode run, and any stamp that contradicts the
  `injected-parameters` cell in the committed notebook. An unstamped notebook is
  reported but does not fail, until the backfill is complete and the gate moves to
  `--strict`.
- `strip_empty_cell_tags.py` — run when the pair-sync gate reports a notebook
  whose `.ipynb` carries empty `tags: []` its `.py` does not.
- `sanitize_notebook_paths.py` — strips machine-specific absolute paths out of
  committed notebook outputs. It currently rewrites source as well as outputs, so
  check its diff before committing.

> Internal registry-maintenance tooling (backfills, schema migrations, one-off data
> repairs) is intentionally **not** in this repository — it lives in the separate
> maintainer workspace.
