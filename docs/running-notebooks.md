# Running Notebooks

This guide explains how to execute notebooks, work with case studies, and experiment with your own strategies.

---

## Two Ways to Run

### Option A: Docker (Recommended)

Docker provides a consistent environment across all platforms with pre-built images on Docker Hub. After [installation](installation.md):

```bash
# Pull the image (one time, ~12 GB on x86, ~3 GB on ARM64)
docker compose pull ml4t

# Start Jupyter Lab
docker compose up ml4t
# Open http://localhost:8888

# Run a notebook directly
docker compose run --rm ml4t python 11_ml_pipeline/01_ols_inference.py

# Run with GPU (deep learning chapters)
docker compose --profile gpu run --rm ml4t-gpu python 13_dl_time_series/01_core_architectures.py
```

Docker covers **all** notebooks across all 27 chapters and 9 case studies, though a small
subset requires a non-default profile such as `py312`, `benchmark`, or `rapids`.

### Option B: Local with uv (Advanced)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments automatically. A local setup covers ~90% of notebooks; a few require Docker:

| Docker-Only Notebooks | Reason | Image |
|----------------------|--------|-------|
| Ch05 `03_sigcwgan_signatures` | signatory (no Python 3.14 wheel) | py312 |
| Ch09 `06_path_signatures`, `12_wasserstein_regimes` | signatory, esig (no Python 3.14 wheel) | py312 |
| Ch10 `01_word2vec`, `02_asset_embeddings`, `03_sentiment_evolution` | gensim (no Python 3.14 wheel) | py312 |
| Ch12 `10_shap_nlp_sentiment` | torch CUDA bug on 3.14 + shap | py312 |
| Ch14 `06_conditional_autoencoder` | torch CUDA bug on 3.14 + shap | py312 |
| Ch15 `06_fed_announcement_bsts` | tfcausalimpact (TFP BSTS, isolated `/opt/bsts/bin/python`) | py312 |
| Ch02 `21_storage_benchmark_database` | requires benchmark image + database services | benchmark |
| Ch12 `02_gbm_comparison` (GPU section) | RAPIDS cuML, LightGBM CUDA | rapids |

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
cd machine-learning-for-trading

# Set up environment
cp .env.example .env
# Edit .env to add API keys (see data/README.md)

# Install all dependencies
uv sync

# Run a notebook
uv run python 11_ml_pipeline/01_ols_inference.py

# Or start Jupyter Lab, from the repo root, and open the URL it prints
ML4T_DATA_PATH="${ML4T_DATA_PATH:-$PWD/data}" uv run jupyter lab
```

A local Jupyter Lab generates an access token on each start and prints the address with the token
attached:

```
http://localhost:8888/lab?token=ef2600091d6010aa5e7f044172907ebf893f16f5d1aaa851
```

Open that whole line, not a bare `http://localhost:8888`, which only shows a token prompt. On
Windows the server runs inside WSL2 and no browser opens by itself, so copy the URL into your normal
Windows browser; WSL2 forwards `localhost` for you.

`uv sync` installs Jupyter Lab, so no separate install is needed. The `ML4T_DATA_PATH` prefix gives
the data loaders an absolute path: Jupyter runs each notebook with its own chapter folder as the
working directory, and without it they look for `01_process_is_edge/data/…` and report the data as
missing. The form above keeps a value you already exported and falls back to the repository's own
`data/` only when you have not set one. If you keep the datasets elsewhere, export that path in your
shell profile - setting it in `.env` alone is not enough, because `uv run` does not read `.env`.

**Platform notes for local setup:**
- **Python 3.14+** required
- **GPU**: PyTorch auto-detects CUDA if NVIDIA drivers are installed
- **Apple Silicon**: Most packages have native ARM64 wheels; the py312 notebooks above cannot run on ARM64 — view their pre-executed `.ipynb` files instead

---

## Your First Notebook

Once Jupyter Lab is running - `docker compose up ml4t` on the Docker path, or
`ML4T_DATA_PATH="${ML4T_DATA_PATH:-$PWD/data}" uv run jupyter lab` from the repo
root on the local path - open it in your web browser: **http://localhost:8888**
for Docker, and for the local path the tokenized URL the server printed. The
repository's file tree appears on the left. On Windows, start it inside your WSL2
Ubuntu terminal and open the address in your normal Windows browser.

1. In the file browser (left panel), open a chapter folder — e.g.
   `01_process_is_edge` — and **double-click** a `.ipynb` file to open it.
2. Run the selected cell with **Shift+Enter** (it runs, then moves to the next
   cell), or run the whole notebook from the menu: **Run → Run All Cells**.
3. When a step asks you to run a shell command (for example, downloading data),
   open a terminal *inside* Jupyter Lab: **File → New → Terminal**. That terminal
   is already inside the container, at the repo root, with everything installed.
   It is where Docker readers run the `python …` commands shown in this guide —
   type `python …` directly (drop the `uv run` prefix; there is no `uv` in the
   container).

You do **not** need a separate terminal window for the Docker workflow — the
Jupyter Lab terminal tile is where the commands below run.

---

## Notebook Format

Notebooks use **Jupytext percent format**: the source of truth is the `.py` file, and `.ipynb` is generated from it.

```
11_ml_pipeline/
  01_ols_inference.py      # Source (edit this)
  01_ols_inference.ipynb   # Generated (view in Jupyter)
```

**Viewing**: Open `.ipynb` files in Jupyter Lab, VS Code, or on GitHub (rendered with outputs).

**Running**: Execute the `.py` file from the repo root:

```bash
uv run python 11_ml_pipeline/01_ols_inference.py
# or
docker compose run --rm ml4t python 11_ml_pipeline/01_ols_inference.py
```

**Important**: Run `.py` notebooks from the repository root. The data loaders resolve `data/` relative
to the working directory, so running from a chapter folder reports the datasets as missing even when
they are downloaded. Setting `ML4T_DATA_PATH` to an absolute path removes the constraint, which is
why the Jupyter Lab command above sets it.

---

## Chapter Notebooks

Each chapter directory contains teaching notebooks that demonstrate concepts from the book:

```
07_defining_the_learning_task/
  01_data_preprocessing.py       # Notebook (Jupytext source)
  01_data_preprocessing.ipynb    # Notebook (Jupyter, with outputs)
  02_label_methods.py
  ...
  README.md                      # Chapter overview and notebook guide
```

These notebooks are self-contained. Run them in order within a chapter, or jump to any notebook that interests you — most chapter notebooks only depend on downloaded data, not on other notebooks.

---

## Case Study Notebooks

Each case study applies the **same end-to-end research workflow** to a different market — ETFs, crypto perpetuals, intraday equities, options, FX, futures, and equity factor panels. Each is a **pipeline**: every stage writes artifacts (labels, features, predictions, backtests) that later stages consume. The stages are already programmed and extensible, but they read like a research process — run one straight through, or open any stage, change it, and re-run from there.

### The Workflow

Every case study follows the same sequence of phases, and **each phase maps to a book chapter**. The stage *numbers* differ from one case study to the next — each market gets a different set of model-family stages (more or fewer deep-learning architectures, latent-factor models where they apply) — but the phase order is identical everywhere:

| Phase | Chapter | What it does |
|-------|---------|--------------|
| Feasibility | Ch6 | Universe breadth, point-in-time eligibility, horizon-cost feasibility, walk-forward setup |
| Labels | Ch7 | Forward returns and classification labels with walk-forward splits |
| Financial features | Ch8 | Momentum, volatility, carry, and cross-sectional ranking features |
| Model-based features | Ch9 | ARIMA, GARCH, HMM, and spectral features from walk-forward fits |
| Evaluation | Ch7–9 | Feature–label IC diagnostics across all engineered features |
| Linear | Ch11 | Ridge / LASSO / ElasticNet baseline every later model must beat |
| Gradient boosting | Ch12 | LightGBM with Optuna tuning |
| Tabular DL | Ch12 | TabM / TabPFN tabular deep learning |
| Sequence DL | Ch13 | LSTM, TCN, TSMixer, PatchTST (architectures vary by market) |
| Latent factors | Ch14 | PCA, IPCA, autoencoders, SDF (where applicable) |
| Causal DML | Ch15 | Double ML — does the signal cause returns or reflect confounders? |
| Model analysis | Ch11–15 | Cross-family IC comparison, fold stability, checkpoint sensitivity |
| Backtest | Ch16 | Strategy simulation, falsified against an equal-weight benchmark |
| Portfolio | Ch17 | Score-weighted, risk-parity, inverse-vol, MVO, HRP, conformal allocation |
| Costs | Ch18 | Transaction-cost impact on the edge |
| Risk | Ch19 | Position-level stops, trailing stops, time exits |
| Strategy analysis | Ch20 | End-to-end assessment — IC, Sharpe, cost survival, holdout |

Each case study's own `README.md` lists its exact stage files with this mapping. To see a given case study's stages, list them:

```bash
ls case_studies/etfs/        # 01_feasibility_analysis.py … 18_strategy_analysis.py
```

### Running a Case Study End to End

Run the stages in order from the repo root. Using the ETF case study (stages `01`–`18`):

```bash
# Define, label, engineer, evaluate (Ch6–9)
uv run python case_studies/etfs/01_feasibility_analysis.py
uv run python case_studies/etfs/02_labels.py
uv run python case_studies/etfs/03_financial_features.py
uv run python case_studies/etfs/04_model_based_features.py
uv run python case_studies/etfs/05_evaluation.py

# Train model families — run any or all (Ch11–15)
uv run python case_studies/etfs/06_linear.py
uv run python case_studies/etfs/07_gbm.py
# … 08_tabular_dl, 09_dl_lstm, 10_dl_tsmixer, 11_latent_factors, 12_causal_dml
uv run python case_studies/etfs/13_model_analysis.py

# Build the strategy — backtest, portfolio, costs, risk, synthesis (Ch16–20)
uv run python case_studies/etfs/14_backtest.py
uv run python case_studies/etfs/15_portfolio_management.py
uv run python case_studies/etfs/16_costs.py
uv run python case_studies/etfs/17_risk_management.py
uv run python case_studies/etfs/18_strategy_analysis.py
```

Each stage checks for the artifacts it needs and tells you which earlier stage to run if anything is missing, so you can always pick up partway through.

### How a Case Study Is Configured

Hyperparameter grids and strategy constants are not written into the stage files. Each stage reads
them at runtime from three layers of configuration:

| File | Declares |
|---|---|
| `case_studies/{cs}/config/setup.yaml` | The trading problem - universe, decision cadence, execution defaults, costs, labels, walk-forward splits, and the Ch16-19 sweep grid |
| `case_studies/{cs}/config/training/{label}.yaml` | The training menu - which named model configs run for that label, by family |
| `case_studies/config/{model_type}/{name}.yaml` | The preset - the hyperparameters behind one config name, shared across case studies |

[`case_studies/RUN_LOG.md`](../case_studies/RUN_LOG.md#configuration-flow) documents the layers and
how a configuration becomes a content-addressed hash.
[Experimenting](#experimenting-without-changing-the-release-baseline) below is the practical
counterpart: which file to edit for a given change, and how to run it without touching the release
baseline.

### The Run Log

Every model training run, prediction set, causal-effect estimate, and backtest is recorded in a per-case-study **run log** (`run_log/`). The SQLite catalog `run_log/registry.db` is the single source of truth for all metrics discussed in the book — IC scores, Sharpe ratios, drawdowns, etc.

See [`case_studies/RUN_LOG.md`](../case_studies/RUN_LOG.md) for the schema and querying API.

### Pre-Computed Results (Download Artifacts)

Running a case study end to end can take hours or days. The artifact release provides the complete
registered run logs for all nine case studies. Each is a separate download, so you can take only the
ones you want - start with `etfs` (33 MB), the case study the book follows most closely.

| Case study | Download |
|---|---:|
| `etfs` | 33 MB |
| `cme_futures` | 37 MB |
| `fx_pairs` | 41 MB |
| `sp500_options` | 42 MB |
| `crypto_perps_funding` | 43 MB |
| `sp500_equity_option_analytics` | 44 MB |
| `us_firm_characteristics` | 56 MB |
| `nasdaq100_microstructure` | 1.1 GB |
| `us_equities_panel` | 1.6 GB |

The last two are large because they are wide, high-frequency panels - a single NASDAQ minute-bar
prediction set is about 80 MB on its own. Their artifact counts are in line with the rest.

```bash
# Download all nine case study artifacts (about 3.1 GB total)
uv run python scripts/download_artifacts.py

# Download a single case study
uv run python scripts/download_artifacts.py --cs etfs

# Check what's installed
uv run python scripts/download_artifacts.py --list
```

The downloader verifies the archive and every file inside it before atomically installing
`case_studies/{cs}/run_log/`. An interrupted or corrupt download leaves any existing run log
unchanged. The installed baseline is read-only and contains:

- **`registry.db`** - the accepted metrics and provenance database
- **Training artifacts** - registered specifications, coefficients, boosters, checkpoints, and curves
- **Predictions** - every stored validation and holdout prediction referenced by the registry
- **Backtests** - every registered return, trade, weight, and configuration file that the run produced
- **Release metadata** - source identity, scope records, and per-file checksums

With these artifacts, you can:

1. **Browse results immediately** - analysis notebooks load metrics and stored artifacts directly.
2. **Trace results** - registry hashes resolve to the exact prediction and backtest files used downstream.
3. **Reproduce selectively** - rerun a chosen model in an isolated experiment instead of retraining the sweep.
4. **Compare safely** - start from the released run log without modifying the downloaded baseline.

The separate maintainer archive also preserves historical registry backups and obsolete caches. Those
files are not reader inputs and are therefore excluded from the release bundles.

### Why a Fresh Run May Not Match the Published Numbers Exactly

The released artifacts are a snapshot: they record what the pipeline produced at the time the book
went to press. If you rerun a stage yourself, expect small differences from the stored values.

- **The code keeps improving.** This repository is maintained after publication. Bug fixes and
  refinements land in the notebooks continuously, and a fix made after the snapshot was taken will
  move the numbers a fresh run produces. The released registry is not regenerated every time.
- **Hardware and libraries differ.** GPU training is not bitwise reproducible, and library versions,
  BLAS backends, and CPU-versus-GPU execution all shift results at the margin.
- **Market data is revised.** Vendors restate history. A download today may not be byte-identical to
  the one behind the release.

Differences of this kind are normally small enough to leave the conclusions intact. The book's
arguments rest on the *shape* of the results - which model families work, how much the selection
funnel deflates apparent performance, where costs bite - not on a specific Sharpe ratio to three
decimals. Treat the published numbers as the reference run, not as values a rerun must match.

If a rerun produces a difference large enough to change a conclusion rather than a decimal, that is
worth reporting as an issue.

---

## Experimenting Without Changing the Release Baseline

Create a writable copy of the installed artifacts before changing a configuration or running a
training or backtest stage. The artifact bundle installs the registry (`run_log/`) only, so first
produce the modeling dataset (`features/`, `labels/`) that the model stages consume, then create the
experiment and run your edited stage against it:

```bash
# 1. Produce the modeling dataset the model notebooks need (writes features/ and
#    labels/ into the case study directory; skip any that already exist).
uv run python case_studies/etfs/02_labels.py
uv run python case_studies/etfs/03_financial_features.py
uv run python case_studies/etfs/04_model_based_features.py

# 2. Snapshot the installed artifacts + config into a writable experiment.
uv run python scripts/create_experiment.py \
  --cs etfs \
  --output /tmp/ml4t-etf-experiment

# 3. Edit config in the experiment (see below), then run the stage against it.
ML4T_OUTPUT_DIR=/tmp/ml4t-etf-experiment \
  uv run python case_studies/etfs/07_gbm.py
```

The setup command copies every available generated prerequisite **and the case study's `config/`
tree** into the experiment, changes the release marker to a baseline marker, and makes only the copy
writable. `ML4T_OUTPUT_DIR` routes both config reads and new registry rows into
`/tmp/ml4t-etf-experiment/`, so you change a configuration **inside the experiment** and the
downloaded release stays untouched. You edit config there, not in the model notebook - the notebooks
have no hyperparameter grids inline; they read the config system described below.

### Try Different Model Hyperparameters

The GBM grid is not a `PARAM_GRID` inside `07_gbm.py`. It is the list of preset names in
`config/training/{label}.yaml` under the `gbm:` key; each name resolves to a preset file in
`case_studies/config/lgb/{name}.yaml` that holds the actual LightGBM parameters. To change the grid,
edit these files **in the experiment**:

```bash
# Add or remove configs from the grid (one preset name per line under `gbm:`):
$EDITOR /tmp/ml4t-etf-experiment/etfs/config/training/fwd_ret_21d.yaml

# Change the hyperparameters of a preset, or add a new preset file:
$EDITOR /tmp/ml4t-etf-experiment/config/lgb/leaves_63_mse.yaml

# Run on CPU if you do not have a CUDA-enabled LightGBM build
# (set modeling.gbm.device: cpu in the experiment's setup.yaml):
$EDITOR /tmp/ml4t-etf-experiment/etfs/config/setup.yaml

ML4T_OUTPUT_DIR=/tmp/ml4t-etf-experiment \
  uv run python case_studies/etfs/07_gbm.py
```

Each config that is not already in the copied registry trains and receives a unique hash there; the
analysis notebooks pick up every registry hash automatically.

The same two files drive **every** model family, not just GBM. A stage reads the list under its own
family key and resolves each name against the shared preset directory. Which listed presets actually
run is not uniform, so check the last column before adding one:

| Family key in `config/training/{label}.yaml` | Presets live in | ETF stage that reads it | Which presets it runs |
|---|---|---|---|
| `linear` | `config/{ols,ridge,lasso,elastic_net,logistic}/` | `06_linear.py` | All of them |
| `gbm` | `config/lgb/` | `07_gbm.py` | All of them |
| `tabular_dl` | `config/tabm/` | `08_tabular_dl.py` | All of them |
| `deep_learning` | `config/{lstm,tcn,tsmixer,nlinear,patchtst,nbeats}/` | `09_dl_lstm.py`, `10_dl_tsmixer.py` | Only those whose `params.architecture` matches the stage's own (`lstm`, `tsmixer`); the rest are dropped, so the ETF case study runs nothing for `tcn`, `nlinear`, `patchtst` or `nbeats` |
| `latent_factors` | `config/{pca,ipca,cae,sae,sdf}/` | `11a_pca.py` … `11e_supervised_autoencoder.py` | One per notebook, each asking for a fixed model name. `11_latent_factors.py` is an index that reports registered results, not a training stage |
| `causal_dml` | `config/dml/` | `12_causal_dml.py` | Only the first listed |

**Some stages override the preset they load.** Where a notebook constant wins, edit that constant
rather than the preset:

| Stage | Ignores the preset's | In favor of |
|---|---|---|
| `09_dl_lstm.py`, `10_dl_tsmixer.py` | `n_epochs`, `batch_size`, `params.lookback` | `N_EPOCHS`, `BATCH_SIZE`, `LOOKBACK` in the stage |
| `08_tabular_dl.py` | `n_epochs`, `batch_size` | `N_EPOCHS`, `BATCH_SIZE` in the stage |
| `11c_conditional_autoencoder.py`, `11e_supervised_autoencoder.py` | `n_epochs` | `N_EPOCHS = 50` in the stage |
| `12_causal_dml.py` | `n_folds`, `n_placebo`, `max_samples`, `seed` | the stage's parameter cell |
| `07_gbm.py` on CPU | `params.seed` | the runtime seed, applied in `case_studies/utils/gbm.py` |

The other latent-factor stages are unaffected: PCA and IPCA have no epoch setting, and the SDF preset
declares `n_epochs_unc`, `n_epochs_moment` and `n_epochs_cond`, which the stage passes through.

That last one is applied one layer below the stage file, so reading `07_gbm.py` alone will not reveal
it. The list is what we have found rather than a guarantee of completeness - this precedence is a
known wart, not a design, and it is tracked for a future release. If an edit appears to do nothing,
check the stage for a reassignment of your key after `load_configs`.

**Adding a new preset.** Drop a YAML file into the directory for its model type and list its filename
stem in the menu. Nothing else is needed: `family` and `library` come from the directory name, so a
file in `config/lgb/` is a LightGBM run by construction.

```bash
cp /tmp/ml4t-etf-experiment/config/lgb/leaves_63_mse.yaml \
   /tmp/ml4t-etf-experiment/config/lgb/leaves_127_mse.yaml
$EDITOR /tmp/ml4t-etf-experiment/config/lgb/leaves_127_mse.yaml   # num_leaves: 127
$EDITOR /tmp/ml4t-etf-experiment/etfs/config/training/fwd_ret_21d.yaml   # add: - leaves_127_mse
```

**What earns a new hash.** A run is identified by the hash of its resolved specification, so a config
the registry has not seen trains and registers under a new hash, and re-running an unchanged one
reuses the stored result. That is what lets an experiment accumulate your variants alongside the
released ones and stay comparable. Two caveats: a preset the stage never dispatches produces no hash
at all because nothing runs, and the hash is a cache key rather than a record of what trained, so do
not use "a new hash appeared" to confirm that an override reached the model. The exact hash inputs are
in [`case_studies/RUN_LOG.md`](../case_studies/RUN_LOG.md#configuration-flow).

### Try a Different Backtest Configuration

Transaction costs and selection breadth live in the experiment's `etfs/config/setup.yaml`, not as
variables in `14_backtest.py`, and both are safe to vary against an existing set of predictions:

- `costs.*` - the transaction-cost model (per-share fees, spreads).
- `backtest.sweep.top_n_predictions` - how many model configs advance at each stage.

`decision.cadence` lives there too and takes one extra step. No retraining is needed - the backtest
applies the cadence to your existing predictions and registers the result under a new hash. But
`labels.rebalance_step`, which thins decision dates so holding periods do not overlap, depends on the
cadence and is one of the repository-pinned declarations
[below](#declarations-that-always-come-from-the-repository). Set a compatible value there at the same
time, or it will be wrong for your new cadence.

The `14_backtest.py` parameter cell exposes run-scoping knobs - `TOP_K`, `MAX_SYMBOLS`,
`TOP_N_PREDICTIONS`, `FORCE_REBACKTEST` - which scope what to backtest, not the strategy economics.
This notebook always backtests on the **validation** split (the split the sweep selects on); the
held-out test set is evaluated once on the selected winner in the analysis stage, not from here, so
do not repurpose the `SPLIT` variable to backtest holdout.

```bash
$EDITOR /tmp/ml4t-etf-experiment/etfs/config/setup.yaml   # edit costs / backtest.sweep

ML4T_OUTPUT_DIR=/tmp/ml4t-etf-experiment \
  uv run python case_studies/etfs/14_backtest.py
```

### Compare Your Experiments

Run the analysis notebook with the same output root so it reads the copied registry:

```bash
ML4T_OUTPUT_DIR=/tmp/ml4t-etf-experiment \
  uv run python case_studies/etfs/18_strategy_analysis.py
```

### Declarations That Always Come From the Repository

The four `setup.yaml` entries below are methodology declarations rather than knobs. They are read from
the repository copy even when `ML4T_OUTPUT_DIR` points at an experiment, so editing them in the
experiment does nothing. Leave them alone - that is the intended use. If you do change one in the
repository, start from an empty `run_log/`: none of the four reaches `backtest_hash`, so rows computed
under the old value keep their hash and get reused, quietly mixing two methodologies in one registry.
`config/backtest/base.yaml`, listed last, does not work like them.

- `labels.rebalance_step` - how many schedule slots a trade advances so holding periods do not
  overlap. It follows from the cadence and the label horizon, so it is declared per label rather than
  inferred at runtime.
- `labels.classification_eval_label` - the continuous return substituted for a classification target
  when a backtest needs economic P&L (classification case studies only).
- `universe.cost_feasible` - the frozen, per-split symbol list used by the `cost_feasible` universe
  filter. It is a committed *result* of
  `case_studies/nasdaq100_microstructure/_build_cost_feasible_universe.py`, profiled strictly before
  each window so it carries no look-ahead.
- `backtest.sweep.htm_cost_cascade.liquid_quantile` - the quantile defining the tightest-spread subset
  for the `liquid` universe filter. **Treat this as fixed at 0.20.** Its readers do not agree: the
  shared runtime filter takes the repository value, the Ch18 cascade notebook reads the
  experiment-aware one so an experiment edit changes only what it *reports*, and two `sp500_options`
  stages prefilter at a hardcoded 0.20 first, so no larger configured value can widen the cohort.
  Making it a real knob means routing every reader through one hash-covered value.
- `config/backtest/base.yaml` - the engine-level backtest preset, and the exception to everything
  above. Treat it as read-only. Its engine fields *are* hashed into `backtest_config`, so a repository
  edit produces new hashes rather than silently reusing old rows. And it is only partly pinned: the
  engine uses the repository copy, but the price loader consults the experiment copy to decide whether
  to pull bid/ask columns, so an experiment edit can change which data loads, or fail the load,
  without changing what the engine runs.

---

## Data Requirements

Notebooks require downloaded datasets. See [`data/README.md`](../data/README.md) for the complete data guide.

**Quick start with free data:**

```bash
# Local uv setup:
uv run python data/etfs/market/download.py       # ETF data (Yahoo Finance, no API key)
uv run python data/download_all.py --free-only   # all free datasets
```

**Docker readers:** run the *same* commands in the Jupyter Lab terminal
(**File → New → Terminal**), without the `uv run` prefix — there is no host
Python on the Docker path:

```bash
python data/etfs/market/download.py
python data/download_all.py --free-only
```

Some datasets require API keys (set in `.env`):
- **OANDA** (FX pairs): Free API key from [oanda.com](https://www.oanda.com/)
- **NASDAQ Data Link** (US equities): Free API key from [data.nasdaq.com](https://data.nasdaq.com/)
- **Databento** (CME futures): $125 free signup credit from [databento.com](https://databento.com/)

**AlgoSeek** (NASDAQ-100 minute bars, S&P 500 option chains, NASDAQ-100 TAQ ticks) needs no key and
no account. Download the archives from
[algoseek.com/ml-for-trading](https://algoseek.com/ml-for-trading/); the two large ones convert once
and the ticks only need unzipping — see [AlgoSeek datasets](../data/README.md#algoseek-datasets).
The fourth AlgoSeek dataset the book uses, the S&P 500 daily bars, ships with this repository, so
there is nothing to download or configure for it.

---

## Accelerated Execution with Papermill

Every notebook has a **parameters cell** (`# %% tags=["parameters"]`) with production defaults — the values readers see in the book. [Papermill](https://papermill.readthedocs.io/) can inject override values that reduce data scope, training epochs, or universe size so notebooks complete in minutes instead of hours.

### How It Works

1. The parameters cell defines production values:
   ```python
   # %% tags=["parameters"]
   MAX_SYMBOLS = 0      # 0 = all symbols (production)
   N_EPOCHS = 500
   START_DATE = "2006-01-01"
   ```

2. Papermill creates an *injected* cell after the tagged cell that overrides selected values:
   ```python
   # Injected by Papermill
   MAX_SYMBOLS = 15     # Reduced for fast execution
   N_EPOCHS = 2
   ```

3. The notebook code sees only the final (overridden) values. **Same code path always runs** — there are no `if TEST:` branches.

### Running a Single Notebook with Overrides

```bash
# Run with reduced parameters (output goes to /dev/null)
uv run papermill notebook.ipynb /dev/null \
    --cwd . -k python3 \
    -p MAX_SYMBOLS 15 \
    -p N_EPOCHS 2

# Or save the executed notebook
uv run papermill notebook.ipynb output.ipynb \
    --cwd . -k python3 \
    -p MAX_SYMBOLS 15
```

### Running via pytest (Recommended)

The test suite reads per-notebook overrides from `tests/overrides.yaml` and runs each notebook through Papermill with appropriate parameter reductions:

```bash
# Run all notebooks in a chapter
uv run pytest tests/test_chapter_notebooks.py -v -k "11_ml_pipeline"

# Run a specific notebook
uv run pytest tests/test_chapter_notebooks.py -v -k "01_ols_inference"

# Run all case study notebooks for ETFs
uv run pytest tests/test_chapter_notebooks.py -v -k "etfs"

# Run everything (takes ~2 hours with reduced parameters)
uv run pytest tests/test_chapter_notebooks.py -v
```

### Override Configuration

Test parameter overrides are defined in `tests/overrides.yaml`, keyed by notebook path:

```yaml
# Example entries
11_ml_pipeline/01_ols_inference:
  timeout: 300
  parameters:
    MAX_SYMBOLS: 10
    MAX_TRAIN_ROWS: 5000

case_studies/etfs/07_gbm:
  timeout: 180
  parameters:
    MAX_FOLDS: 2
    MAX_SYMBOLS: 5
```

Papermill injects these values in a cell placed right after the notebook's
`# %% tags=["parameters"]` cell, so each name has to be one the notebook reads
below that point and does not overwrite before reading. Anything else is either
an unused variable or is discarded before it is used.
`tests/test_pm_helpers.py` rejects names that fail either condition, so a
mistyped or renamed parameter turns the build red instead of quietly running the
notebook at full scale.

**To customize for your machine**: copy `tests/overrides.yaml` to `tests/overrides.local.yaml` (gitignored) and adjust timeouts or parameter values. The test runner checks for the local file first.

### Output Isolation

When the environment variable `ML4T_OUTPUT_DIR` is set (which `pytest` does automatically), notebook outputs **and the model/sweep config reads** are redirected to that directory. This prevents test runs from overwriting production artifacts like trained models or backtest results. Because that config is redirected too, the target must contain the case study's config: `pytest` seeds it automatically, and for a manual run `create_experiment.py` builds the isolated copy. To actually run a stage against the isolated directory - including generating the `features/`/`labels/` a model stage needs first - follow the runnable sequence in [Experimenting Without Changing the Release Baseline](#experimenting-without-changing-the-release-baseline) above; setting `ML4T_OUTPUT_DIR` by hand at an empty path will fail because the redirected config (and modeling dataset) are absent.

---

## Headless Execution

For running notebooks without a display (e.g., on a server or in CI):

```bash
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python notebook.py
```

---

## Troubleshooting

### "No module named 'utils'"

You're running from a subdirectory. Always run from the repository root:

```bash
# Wrong
cd 11_ml_pipeline && python 01_ols_inference.py

# Right
uv run python 11_ml_pipeline/01_ols_inference.py
```

### Missing prerequisite files

Case study notebooks check for upstream artifacts. If a file is missing, the notebook tells you which notebook to run first.

### Slow notebooks

Some model training notebooks take several minutes. Notebooks with long runtimes print progress during execution. For faster iteration, reduce the data scope in the parameters cell at the top of each notebook.
