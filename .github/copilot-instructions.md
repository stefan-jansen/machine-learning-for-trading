# Repository conventions for code review

This is the companion code repository for *Machine Learning for Trading, 3rd Edition*.
The conventions below are deliberate and repository-wide. Please do not flag adherence
to them as defects, and do apply them when reviewing.

## Prose and comment style

- **No em dashes.** The repository standardizes on the ASCII hyphen (`-`) in all prose,
  markdown, docstrings, figure titles, and printed output. Em dashes (`—`) are being
  removed repository-wide; a file that uses a hyphen where a sibling still has an em dash
  is **ahead** of the sweep, not a regression. Do not suggest reintroducing em dashes for
  "consistency" with files that have not been cleaned yet.

## Notebooks

- Every notebook is a **paired `.py` + `.ipynb`** managed by
  [jupytext](https://jupytext.readthedocs.io/). The `.py` is the source of truth; the
  `.ipynb` is generated. Edit the `.py` and run `jupytext --sync`; never hand-edit the
  `.ipynb`.
- A committed `.ipynb` is a **production re-execution** of its paired `.py`, stamped in
  `metadata.ml4t_provenance`. Cell outputs are real, not illustrative.
- Notebook output must carry **no machine-specific paths** (no `/tmp/...`, no home
  directory, no absolute checkout path). Print repo-root-relative paths instead, e.g.
  `output_path.relative_to(get_case_study_dir("etfs").parents[1])`.

## Data schema

- The canonical schema is **`symbol`** (entity identifier) + **`timestamp`** (all
  frequencies, daily and intraday). The `cme_futures` dataset uses **`product`** instead
  of `symbol`. These are intentional; do not suggest renaming to `asset`, `date`,
  `ticker`, or `pair`.

## Results and DataFrames

- The **sole source of truth for model/backtest/strategy results is `run_log/registry.db`**
  in each case study. `results/*.json` files are a deprecated legacy format and must not
  be treated as ground truth.
- **Polars-first.** DataFrame operations use [Polars](https://pola.rs/); pandas appears
  only at visualization boundaries. Do not suggest converting Polars code to pandas.

## Library corrections

- The six `ml4t-*` libraries are the production implementations. When a PR updates numbers
  because a library estimator was corrected (for example a beta/correlation fix), that is a
  **deliberate correctness change**, not an unexplained result drift.
