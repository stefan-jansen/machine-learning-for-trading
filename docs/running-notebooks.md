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
| Ch15 `06_fed_announcement_bsts` | tfcausalimpact (TFP BSTS, py<3.13) | py312 |
| Ch21 `05_deep_hedging_pfhedge` | pfhedge (unmaintained, numpy<2) | py312 |
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
```

**Platform notes for local setup:**
- **Python 3.14+** required
- **TA-Lib** must be installed separately ([instructions](https://ta-lib.github.io/ta-lib-python/install.html))
- **GPU**: PyTorch auto-detects CUDA if NVIDIA drivers are installed
- **Apple Silicon**: Most packages have native ARM64 wheels; the py312 notebooks above cannot run on ARM64 — view their pre-executed `.ipynb` files instead

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

**Important**: Always run from the repository root. Running from a subdirectory will fail with `ImportError: No module named 'utils'`.

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

### The Run Log

Every model training run, prediction set, causal-effect estimate, and backtest is recorded in a per-case-study **run log** (`run_log/`). The SQLite catalog `run_log/registry.db` is the single source of truth for all metrics discussed in the book — IC scores, Sharpe ratios, drawdowns, etc.

See [`case_studies/RUN_LOG.md`](../case_studies/RUN_LOG.md) for the schema and querying API.

### Pre-Computed Results (Download Artifacts)

Running all nine case study pipelines end-to-end (training ~50 model configurations, running ~1,000 backtests per case study) takes days of compute. To let you explore results immediately, we provide a curated subset of artifacts as a **GitHub release**:

```bash
# Download all case study artifacts (~1.6 GB total)
uv run python scripts/download_artifacts.py

# Download a single case study
uv run python scripts/download_artifacts.py --cs etfs

# Check what's installed
uv run python scripts/download_artifacts.py --list
```

This populates `case_studies/{cs}/run_log/` with:

- **`registry.db`** — full metrics database (all training runs, predictions, backtests)
- **Best predictions per model family** — validation predictions for cross-model comparison (Ch11-15 insight notebooks)
- **Top-10 predictions by IC** — for backtest analysis (Ch16)
- **Top backtests by stage** — signal, allocation, cost sensitivity (Ch17-19 strategy notebooks)
- **Holdout predictions** — for out-of-sample synthesis (Ch20)

With these artifacts, you can:

1. **Browse results immediately** — the model-analysis and strategy-analysis stages load predictions and metrics from the registry
2. **Reproduce selectively** — run any model notebook to verify or extend results
3. **Experiment** — new runs register automatically alongside the shipped baselines
4. **Compare** — analytical notebooks query whatever is in the registry, so your experiments appear next to the book's results

**What's not included**: The full set of ~1,000 backtest variations per case study (these total ~97 GB). The download provides the ~20 best-performing configurations that the book discusses. You can generate the rest by running the backtest stages yourself.

---

## Experimenting

The case study pipeline is designed for experimentation. Here are common workflows:

### Try Different Model Hyperparameters

Open a model notebook (e.g., `07_gbm.py`), modify the configuration, and run it. The new run registers with a unique hash — your results coexist with the originals.

```python
# In 07_gbm.py, change the parameter grid:
PARAM_GRID = {
    "num_leaves": [31, 63, 127],      # Try more complex trees
    "learning_rate": [0.01, 0.05],     # Different learning rates
    "min_child_samples": [20, 50],
}
```

### Try a Different Backtest Configuration

Modify the signal-to-position mapping, change cost assumptions, or adjust position sizing:

```python
# In 14_backtest.py, change the strategy:
TOP_N = 10              # Hold top 10 instead of top 20
COST_BPS = 15           # Higher transaction costs
REBALANCE_FREQ = "W"    # Weekly instead of monthly
```

### Compare Your Experiments

Open the analysis notebook — it automatically picks up all registry entries:

```bash
uv run python case_studies/etfs/18_strategy_analysis.py
# Shows your new runs alongside the book's baselines
```

---

## Data Requirements

Notebooks require downloaded datasets. See [`data/README.md`](../data/README.md) for the complete data guide.

**Quick start with free data:**

```bash
# Download ETF data (Yahoo Finance, no API key needed)
uv run python data/etfs/market/download.py

# Download all free datasets
uv run python data/download_all.py --free-only
```

Some datasets require API keys (set in `.env`):
- **OANDA** (FX pairs): Free API key from [oanda.com](https://www.oanda.com/)
- **NASDAQ Data Link** (US equities): Free API key from [data.nasdaq.com](https://data.nasdaq.com/)
- **Databento** (CME futures): $125 free signup credit from [databento.com](https://databento.com/)
- **AlgoSeek** (microstructure, options): Requires commercial license

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
  timeout: 180
  parameters:
    MAX_SYMBOLS: 15

case_studies/etfs/07_gbm:
  timeout: 300
  parameters:
    MAX_SYMBOLS: 15
    START_DATE: "2020-01-01"
```

**To customize for your machine**: copy `tests/overrides.yaml` to `tests/overrides.local.yaml` (gitignored) and adjust timeouts or parameter values. The test runner checks for the local file first.

### Output Isolation

When the environment variable `ML4T_OUTPUT_DIR` is set (which `pytest` does automatically), all notebook outputs are redirected to a temporary directory. This prevents test runs from overwriting production artifacts like trained models or backtest results.

```bash
# Manual output isolation
ML4T_OUTPUT_DIR=/tmp/ml4t-test uv run python case_studies/etfs/07_gbm.py
```

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
