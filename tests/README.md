# Notebook Test Suite

Every notebook in this repository is tested via [Papermill](https://papermill.readthedocs.io/) parameter injection. The same code path always runs — only the data scale differs between production and test.

---

## Quick Start

```bash
# Run all environments (ml4t, gpu, py312, benchmark, neo4j)
./scripts/run_all_tests.sh

# Run one environment
./scripts/run_all_tests.sh ml4t

# Rerun everything (ignore already-passed)
./scripts/run_all_tests.sh --force

# Run a specific notebook via pytest
docker compose run --rm ml4t pytest tests/test_chapter_notebooks.py -v -k "01_timegan"

# Run locally (with uv)
uv run pytest tests/test_chapter_notebooks.py -v -k "11_ml_pipeline"
```

---

## How It Works

### Papermill Parameter Injection

Every notebook has a `# %% tags=["parameters"]` cell with **production defaults** — the values readers see in the book:

```python
# %% tags=["parameters"]
MAX_SYMBOLS = 0      # 0 = all symbols (production)
N_EPOCHS = 500
START_DATE = "2006-01-01"
```

During testing, Papermill creates an **injected cell** after the tagged cell that overrides selected values:

```python
# Injected by Papermill
MAX_SYMBOLS = 15     # Reduced for fast execution
N_EPOCHS = 2
```

The notebook code sees only the final (overridden) values. **There are no `if TEST:` branches** — the same code path always runs, just with less data or fewer iterations.

### Output Isolation

Tests set `ML4T_OUTPUT_DIR` to a temp directory. All notebook writes (`get_output_dir()`, `get_case_study_dir()`) redirect there, so production artifacts (trained models, backtest results) are never overwritten.

### Seeded Fixtures

Downstream notebooks (Ch11+) need upstream pipeline outputs (features, labels, model predictions). Rather than running the full pipeline during every test, `tests/fixtures/seed_results.py` creates minimal but schema-correct fixtures:

- Registry databases (`run_log/registry.db`) with realistic training_runs, prediction_sets, and backtest entries
- Feature parquets (350 rows × 15 symbols)
- Label parquets (5 label variants per case study)
- Prediction parquets (200 rows per model)

Fixtures are deterministic and only written if the file doesn't already exist — real upstream results take priority.

---

## Test Data

### Architecture

Tests require two things: **raw market data** (what loaders read) and **pipeline intermediates** (what downstream notebooks consume). Both live in a private repo that CI pulls automatically.

```
~/ml4t/test-data/                  # Local clone of ml4t/third-edition-test-data (~2 GB private GitHub repo)
├── data/                          # Pre-subsampled raw data (553 MB)
│   ├── etfs/                      # 15 most liquid ETFs (full date range)
│   ├── crypto/                    # 5 largest perps
│   ├── futures/                   # 8 CME products
│   ├── fx/                        # 8 major pairs
│   ├── equities/                  # 50 US stocks, 3 NASDAQ-100 (minute bars)
│   │   ├── microstructure/        # Synthetic ITCH/LOB/MBO for Ch03
│   │   └── firm_characteristics/  # 200 most-observed per month
│   ├── factors/                   # Fama-French + AQR
│   └── manifest.json              # Symbol counts per dataset
│
└── intermediates/                 # Pre-computed pipeline outputs (301 MB)
    ├── etfs/                      # registry.db, features, labels, predictions
    ├── crypto_perps_funding/
    ├── ...                        # All 9 case studies
    └── _metadata.json             # Generation timestamp and parameters
```

**Key design decisions:**

- **Same schema, fewer symbols**: Loaders work without code changes. Full date ranges are preserved so cross-validation folds remain valid.
- **Pre-computed intermediates**: Running all 9 case study pipelines from scratch takes ~25 minutes. Pre-computing once and shipping the results lets CI focus on testing individual notebooks.
- **Fixtures as safety net**: If intermediates are missing or a new notebook needs data that wasn't pre-computed, `seed_results.py` generates minimal placeholders at test time.

### Running Tests Locally

**With the test-data repo** (full test coverage):

```bash
# Clone test data once
git clone git@github.com:ml4t/third-edition-test-data.git ~/ml4t/test-data

# Point tests at it
export ML4T_DATA_PATH=~/ml4t/test-data/data
export ML4T_OUTPUT_DIR=/tmp/ml4t-test-output

# Seed intermediates
mkdir -p $ML4T_OUTPUT_DIR
cp -r ~/ml4t/test-data/intermediates/* $ML4T_OUTPUT_DIR/

# Run tests
uv run pytest tests/test_chapter_notebooks.py -v -k "11_ml_pipeline"
```

**Without the test-data repo** (limited coverage):

```bash
# Point at your production data
export ML4T_DATA_PATH=/path/to/your/data
export ML4T_OUTPUT_DIR=/tmp/ml4t-test-output

# Fixtures will be auto-generated for missing intermediates
uv run pytest tests/test_chapter_notebooks.py -v -k "01_ols_inference"
```

### Regenerating Test Data

If the data schema or pipeline logic changes, regenerate the test data:

```bash
# 1. Subsample raw data
uv run python tests/create_test_data.py --output ~/ml4t/test-data

# 2. Generate pipeline intermediates (runs all 9 case studies, ~25 min)
ML4T_DATA_PATH=~/ml4t/test-data/data \
ML4T_OUTPUT_DIR=~/ml4t/test-data/intermediates \
  uv run python tests/_internal/generate_intermediates.py

# 3. Commit and push to test-data repo
cd ~/ml4t/test-data && git add -A && git commit -m "regenerate test data"
```

---

## Environments

Each notebook is assigned to exactly one Docker environment via `docker_env` in `overrides.yaml`.

| Environment | Docker Service | Notebooks | What It Provides |
|-------------|---------------|-----------|-----------------|
| `ml4t` | `ml4t` | ~410 | CPU, all Python packages |
| `gpu` | `ml4t-gpu` | ~31 | NVIDIA GPU (PyTorch CUDA) |
| `py312` | `py312` | ~10 | gensim, signatory, esig, pfhedge, tfcausalimpact (Python 3.12) |
| `benchmark` | `benchmark` + database services | 2 | TimescaleDB, ClickHouse, QuestDB, InfluxDB |
| `neo4j` | `ml4t` + Neo4j service | 7 | Neo4j graph database |

Notebooks default to `ml4t` unless tagged otherwise. Multi-environment tags (e.g., `docker_env: ml4t+neo4j+gpu`) require all listed services.

The test runner (`scripts/run_all_tests.sh`) iterates over environments, running only notebooks tagged for each one.

---

## Override Format

Per-notebook configuration lives in `tests/overrides.yaml`:

```yaml
05_synthetic_data/01_timegan:
  docker_env: gpu          # Runs only in GPU environment
  timeout: 600             # Max seconds before test is killed
  parameters:              # Papermill parameter overrides
    TRAIN_STEPS: 100
    BATCH_SIZE: 32

case_studies/etfs/07_gbm:
  timeout: 300
  parameters:
    MAX_SYMBOLS: 15
    START_DATE: "2020-01-01"

26_mlops_governance/05b_feast_live:
  skip: true               # Never runs
  skip_reason: "Requires Feast feature server"
```

**Override fields:**

| Field | Default | Purpose |
|-------|---------|---------|
| `timeout` | 300 | Max seconds per notebook |
| `docker_env` | `ml4t` | Which Docker environment to use |
| `skip` | false | Skip this notebook entirely |
| `skip_reason` | — | Reason displayed in test output |
| `parameters` | {} | Papermill parameter overrides |

---

## Files

| File | Purpose |
|------|---------|
| `test_chapter_notebooks.py` | Parametrized tests for Ch01-Ch27 teaching notebooks |
| `test_case_studies.py` | Parametrized tests for all 9 case study pipelines |
| `test_backtest_schedule.py` | Backtest-specific integration tests |
| `conftest.py` | Session fixtures: data dirs, output seeding, config patching |
| `pm_helpers.py` | Papermill execution, override loading, Docker env detection |
| `overrides.yaml` | Per-notebook parameter overrides, timeouts, skip/env tags |
| `fixtures/seed_results.py` | Registry DB and parquet fixture generation |
| `_internal/` | Scripts for generating test data and intermediates (not run during tests) |

---

## CI Pipeline

GitHub Actions runs tests on every push and PR:

1. **Checkout** code + test data repo (via deploy key)
2. **Seed** intermediates into `ML4T_OUTPUT_DIR`
3. **Run** pytest inside Docker containers (one job per environment)
4. **Upload** JUnit XML results as artifacts

See `.github/workflows/test.yml` for the full configuration.

---

## Adding a New Notebook

1. Add a `# %% tags=["parameters"]` cell with production defaults
2. Add an entry in `overrides.yaml` with timeout and parameter overrides
3. If GPU-required, add `docker_env: gpu`
4. Run: `./scripts/run_all_tests.sh ml4t` (or the relevant environment)
5. If the notebook depends on upstream pipeline outputs, ensure `seed_results.py` generates the necessary fixtures
