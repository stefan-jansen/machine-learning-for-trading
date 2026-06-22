# Docker Environments

Pre-built images are available on Docker Hub (`docker.io/ml4t/`). Most readers need only the main `ml4t` image.

## Images

| Image | Docker Hub | Python | Platforms | Size |
|-------|-----------|--------|-----------|------|
| **ml4t** | `ml4t/ml4t:latest` | 3.14 | amd64 + arm64 | ~12 GB / ~3 GB |
| **py312** | `ml4t/ml4t-py312:latest` | 3.12 | amd64 only | ~9.6 GB |
| **benchmark** | `ml4t/ml4t-benchmark:latest` | 3.14 | amd64 + arm64 | ~1.7 GB |
| **rapids** | (build locally) | 3.12 | amd64 + NVIDIA GPU | ~15 GB |

### ml4t (Main)

Covers all 27 chapters and 9 case studies. Includes PyTorch with CUDA 12.8 support, LightGBM, scikit-learn, Polars, Plotly, and all ML4T libraries.

```bash
docker compose pull ml4t
docker compose up ml4t                    # Jupyter Lab at http://localhost:8888
docker compose run --rm ml4t python nb.py # Run a notebook directly
```

GPU passthrough (same image, NVIDIA runtime required):

```bash
docker compose --profile gpu run --rm ml4t-gpu python notebook.py
```

### py312 (Python 3.12 Dependencies)

For notebooks requiring libraries without Python 3.14 wheels:

| Notebook | Library |
|----------|---------|
| Ch05 `03_sigcwgan_signatures` | signatory |
| Ch09 `06_path_signatures`, `12_wasserstein_regimes` | signatory, esig |
| Ch10 `01_word2vec`, `02_asset_embeddings`, `03_sentiment_evolution` | gensim |
| Ch15 `06_fed_announcement_bsts` | tfcausalimpact (TFP BSTS) |
| Ch21 `05_deep_hedging_pfhedge` | pfhedge |

```bash
docker compose --profile py312 pull py312
docker compose --profile py312 run --rm py312 python 05_synthetic_data/03_sigcwgan_signatures.py
```

Not available on Apple Silicon — view pre-executed `.ipynb` files instead.

### benchmark (Storage Benchmarks)

For Chapter 2 storage benchmarks comparing file formats and databases (TimescaleDB, ClickHouse, QuestDB, InfluxDB).

```bash
docker compose pull benchmark

# Start database services
docker compose --profile benchmark up -d timescaledb clickhouse questdb influxdb

# Run benchmark
docker compose --profile benchmark run --rm benchmark \
  python 02_financial_data_universe/21_storage_benchmark_database.py

# Stop databases
docker compose --profile benchmark down
```

### rapids (GPU Benchmarks)

For Chapter 12 GBM GPU benchmark with RAPIDS cuML and LightGBM CUDA. Requires NVIDIA GPU. Must be built locally:

```bash
docker compose --profile rapids build rapids
docker compose --profile rapids run --rm rapids python 12_gradient_boosting/02_gbm_comparison.py
```

## Directory Structure

```
envs/
├── README.md                  # This file
├── ml4t/Dockerfile            # Main image (Python 3.14)
├── py312/
│   ├── Dockerfile             # Python 3.12 for signatory/esig/gensim/pfhedge/tfcausalimpact
│   └── pyproject.toml         # py312-specific dependencies
├── benchmark/
│   ├── Dockerfile             # Benchmark image with DB clients
│   └── pyproject.toml         # Benchmark-specific dependencies
├── rapids/
│   └── Dockerfile             # RAPIDS cuML + LightGBM CUDA
└── test_all_imports.py        # Import verification script (63 packages)
```

## Import Verification

Each Docker image has a self-test that verifies all packages needed by its notebooks are importable. Run it after pulling or building to confirm the environment is healthy:

```bash
# ml4t image (baked-in command)
docker compose run --rm ml4t ml4t-test-imports

# Or explicitly
docker compose run --rm ml4t python envs/test_all_imports.py

# py312 image
docker compose --profile py312 run --rm py312 python envs/test_all_imports.py --image py312

# Benchmark image
docker compose --profile benchmark run --rm benchmark python envs/test_all_imports.py --image benchmark

# Test a specific chapter
docker compose run --rm ml4t python envs/test_all_imports.py --chapter 15 --verbose
```

**What's tested per image:**

| Image | Packages | ML4T Libs | Utils Modules | Chapters |
|-------|----------|-----------|---------------|----------|
| **ml4t** | 50 third-party | 6 (data, engineer, models, diagnostic, backtest, live) | 22 (4 repo + 18 case study) | Ch01-Ch26 |
| **py312** | 5 (signatory, esig, gensim, pfhedge, tfcausalimpact) | 1 (diagnostic) | — | Ch05, Ch09, Ch10, Ch12, Ch14, Ch15, Ch21 |
| **benchmark** | 5 (duckdb, tables, DB clients) | 0 | — | Ch02 |

The test groups packages by chapter, so failures map directly to which notebooks are affected. Exit code is 0 (all pass) or 1 (failures). py312-only packages are shown as informational when running the ml4t test.

## Building Locally

If you prefer to build from source instead of pulling from Docker Hub:

```bash
docker compose build ml4t                       # ~45 min on x86, ~15 min on ARM64
docker compose --profile py312 build py312      # ~30 min
docker compose --profile benchmark build benchmark  # ~10 min
```
