# Installation Guide

This guide walks you through setting up Docker to run the ML4T notebooks. Pre-built images on Docker Hub mean you can be running notebooks in minutes.

---

## Platform Support

| Platform                | ml4t | py312  | Benchmark | GPU |
|-------------------------|:----:|:------:|:---------:|:---:|
| **Linux x86_64**        |  ✅  |   ✅   |    ✅     | ✅* |
| **Windows 11 (WSL2)**   |  ✅  |   ✅   |    ✅     | ✅* |
| **macOS Intel**         |  ✅  |   ✅   |    ✅     |  -  |
| **macOS Apple Silicon** |  ✅  |   -    |    ✅     |  -  |

\* Requires NVIDIA GPU + nvidia-container-toolkit

### Which image do I need?

| Image | What it covers | Platforms |
|-------|----------------|-----------|
| **ml4t** | All chapters (Ch01-Ch27) + all 9 case studies | amd64 + arm64 |
| **ml4t-py312** | Ch05 NB01/03/07, Ch09 NB06/12, Ch10 NB01-03, Ch12 NB10, Ch14 NB06, Ch15 NB06, Ch21 deep_hedging (signatory, esig, gensim, pfhedge, tfcausalimpact) | amd64 only |
| **benchmark** | Ch02 storage benchmarks (DuckDB, HDF5, database clients) | amd64 + arm64 |
| **rapids** | Ch12 GBM GPU benchmark (RAPIDS cuML, LightGBM CUDA) | amd64 + NVIDIA GPU |

**Most readers need only `ml4t`.** The other images are for specific notebooks.

**Apple Silicon users**: The notebooks requiring `ml4t-py312` are not runnable on ARM64 because the underlying libraries (signatory, esig) have no ARM64 builds. View the pre-executed `.ipynb` files on GitHub or in Jupyter instead.

---

## Quick Start (All Platforms)

```bash
# 1. Clone the repository
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
cd machine-learning-for-trading

# 2. Copy environment template
cp .env.example .env

# 3. Pull the pre-built image from Docker Hub
docker compose pull ml4t

# 4. Start Jupyter Lab
docker compose up ml4t
# Open http://localhost:8888

# 5. Or run a notebook directly
docker compose run --rm ml4t python 01_process_is_edge/factor_regimes.py
```

**That's it.** No build step needed — Docker pulls the pre-built image (~12 GB on x86, ~3 GB on ARM64).

To build locally instead (if you prefer or need to modify the environment):

```bash
docker compose build ml4t    # ~45 min on x86, ~15 min on ARM64
```

---

## Platform-Specific Setup

### Ubuntu / Linux

```bash
# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# Log out and back in for group membership

# Verify
docker run --rm hello-world
docker compose version
```

If Docker Compose is missing: `sudo apt install docker-compose-plugin`

### Windows 11 (WSL2)

1. **Enable WSL2**: Open PowerShell as Administrator:
   ```powershell
   wsl --install
   # Restart when prompted
   ```

2. **Increase WSL2 memory limit**: WSL2 defaults to 50% of host RAM, which may not be enough for data-heavy notebooks. Create or edit `%USERPROFILE%\.wslconfig`:
   ```ini
   [wsl2]
   memory=12GB
   swap=4GB
   ```
   Then restart WSL: `wsl --shutdown` from PowerShell, then reopen your terminal.

3. **Install Docker Desktop** from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
   - Ensure "Use WSL 2 based engine" is checked in Settings → General
   - In Settings → Resources → WSL Integration, enable your Ubuntu distribution
   - In Settings → Resources, allocate at least 8 GB memory and 60 GB disk

4. **Verify Docker Desktop integration**: Open your WSL Ubuntu terminal and run:
   ```bash
   docker version
   ```
   If this fails with "Cannot connect to the Docker daemon", Docker Desktop's WSL integration is not enabled for your distribution. Check step 3 above.

5. **Clone in WSL** (not on Windows drives — much faster):
   ```bash
   cd ~
   git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
   cd machine-learning-for-trading
   cp .env.example .env
   docker compose pull ml4t
   ```

**Important**: Always run `docker` commands from inside a WSL terminal (Ubuntu), not from Windows PowerShell or Command Prompt. Docker Desktop exposes the Docker socket to WSL distributions, but the Docker CLI in Windows may behave differently.

**Tip**: Keep the repo at `~/machine-learning-for-trading` in WSL, not `/mnt/c/...`. The Windows filesystem (`/mnt/c/`) is dramatically slower due to the 9P protocol bridge. Access WSL files from Windows Explorer via `\\wsl$\Ubuntu\home\<username>\machine-learning-for-trading`.

### macOS (Intel and Apple Silicon)

1. **Install Docker Desktop** from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
   - Choose the correct chip: **Intel** or **Apple chip**
   - Recommended resources: 4+ CPUs, 8+ GB memory, 64+ GB disk

2. **Apple Silicon only**: In Docker Desktop Settings → General, enable **Use Rosetta for x86_64/amd64 emulation** (needed only for the `ml4t-py312` image, which most readers won't use).

3. **Clone and pull**:
   ```bash
   git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
   cd machine-learning-for-trading
   cp .env.example .env
   docker compose pull ml4t
   ```

---

## GPU Support (NVIDIA)

GPU acceleration benefits deep learning chapters (Ch05, Ch10, Ch13, Ch14, Ch21). Requires NVIDIA GPU with CUDA support.

### Requirements

- NVIDIA GPU (GTX 1060 or better)
- NVIDIA Driver 525+ (for CUDA 12.x)
- Linux (native) or Windows 11 (WSL2)
- Not available on macOS

### Ubuntu: Install nvidia-container-toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Windows WSL2

GPU passthrough works automatically with NVIDIA Driver 525+ installed on Windows and Docker Desktop with WSL2 backend.

### Verify and Run

```bash
# Verify GPU is visible
docker compose --profile gpu run --rm ml4t-gpu python -c \
  "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True

# Run with GPU
docker compose --profile gpu run --rm ml4t-gpu python 13_dl_time_series/01_core_architectures.py
```

---

## Storage Benchmarks (Chapter 2)

Chapter 2 includes storage benchmarks comparing file formats and databases.

```bash
# Pull benchmark image
docker compose pull benchmark

# Start database services
docker compose --profile benchmark up -d timescaledb clickhouse questdb influxdb

# Wait for databases to be healthy
docker compose --profile benchmark ps

# Run benchmark
docker compose --profile benchmark run --rm benchmark \
  python 02_financial_data_universe/21_storage_benchmark_database.py

# Stop databases when done
docker compose --profile benchmark down
```

---

## Py312 Image (Specific Notebooks)

A small number of notebooks require Python 3.12 libraries not available on Python 3.14:

| Notebook | Library | Chapter |
|----------|---------|---------|
| `01_timegan`, `03_sigcwgan_signatures`, `07_dp_gan` | signatory, torch CUDA bug on 3.14 | Ch05 |
| `06_path_signatures`, `12_wasserstein_regimes` | signatory, esig | Ch09 |
| `01_word2vec`, `02_asset_embeddings`, `03_sentiment_evolution` | gensim | Ch10 |
| `10_shap_nlp_sentiment` | torch CUDA bug + shap | Ch12 |
| `06_conditional_autoencoder` | torch CUDA bug + shap | Ch14 |
| `06_fed_announcement_bsts` | tfcausalimpact (TFP BSTS) | Ch15 |
| `05_deep_hedging_pfhedge` | pfhedge (unmaintained, numpy<2) | Ch21 |

```bash
# x86 systems only (Linux, Windows WSL2, macOS Intel)
docker compose --profile py312 pull py312
docker compose --profile py312 run --rm py312 python 05_synthetic_data/03_sigcwgan_signatures.py
```

**Apple Silicon**: These notebooks cannot run natively. View the pre-executed `.ipynb` files in Jupyter or on GitHub.

---

## Troubleshooting

### "Cannot connect to Docker daemon"

- **Linux**: `sudo systemctl start docker && sudo systemctl enable docker`
- **Windows/macOS**: Ensure Docker Desktop is running (system tray / menu bar)
- **Windows WSL2**: Make sure you are running from a WSL terminal, not PowerShell. Verify integration: Docker Desktop → Settings → Resources → WSL Integration → enable your distribution

### Out of memory or container killed (WSL2)

WSL2 defaults to 50% of host RAM. Large notebooks (Ch13 deep learning, case study pipelines) may exceed this. Edit `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=4GB
```

Then restart: `wsl --shutdown` from PowerShell and reopen your terminal.

### "Permission denied" on Linux

```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Slow on Apple Silicon

If a container is slow, check if it's running under x86 emulation:
```bash
docker compose run --rm ml4t uname -m
# Should print: aarch64 (native) not x86_64 (emulated)
```

If you see `x86_64`, the image may not have an arm64 variant. The `ml4t` and `benchmark` images both have native arm64 builds.

### "No space left on device"

```bash
docker system prune -a    # Remove unused images/containers
docker system df           # Check space usage
```

### Build fails with network errors

```bash
# Behind a proxy:
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
docker compose pull ml4t
```

---

## Local Setup with uv (Alternative to Docker)

Docker is recommended because it guarantees a consistent environment. But if you prefer a local Python setup — for faster iteration, IDE integration, or GPU access without container overhead — [uv](https://docs.astral.sh/uv/) handles everything from Python installation through dependency resolution.

### What uv Does

`uv` is a fast Python package manager written in Rust. It replaces `pip`, `venv`, `pip-tools`, and `pyenv` in a single tool. When you run `uv sync`, it:

1. Reads `pyproject.toml` for dependency specifications
2. Reads `uv.lock` for exact pinned versions (reproducible across machines)
3. Creates a virtual environment in `.venv/`
4. Installs all packages including PyTorch with CUDA support

### Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the repository
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
cd machine-learning-for-trading

# Install all dependencies (creates .venv/, installs ~300 packages)
uv sync

# Copy environment template and add API keys
cp .env.example .env
# Edit .env — see data/README.md for API key instructions

# Verify
uv run python -c "import polars, torch, lightgbm; print('Ready')"
```

### How pyproject.toml Works

The `pyproject.toml` at the repository root defines all Python dependencies:

- **Core data science**: NumPy, SciPy, Pandas, Polars, PyArrow
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine learning**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP
- **Deep learning**: PyTorch 2.x (with CUDA 12.8 on Linux/Windows)
- **NLP**: Hugging Face Transformers, sentence-transformers, FinBERT
- **ML4T libraries**: ml4t-data, ml4t-engineer, ml4t-models, ml4t-diagnostic, ml4t-backtest, ml4t-live (installed from PyPI)

The lockfile `uv.lock` pins every transitive dependency to exact versions, so `uv sync` produces the same environment regardless of when you install.

### What Local Setup Cannot Run

A few notebooks require Docker because their dependencies have no Python 3.14 wheel or need external services:

| Notebook | Reason | Docker Image |
|----------|--------|-------------|
| Ch05 `03_sigcwgan_signatures` | signatory requires Python 3.12 | py312 |
| Ch09 `06_path_signatures` | esig requires Python 3.12 | py312 |
| Ch10 `01-03` (word2vec, embeddings, sentiment) | gensim requires Python 3.12 | py312 |
| Ch12 `10_shap_nlp_sentiment` | torch CUDA bug on 3.14 + shap | py312 |
| Ch14 `06_conditional_autoencoder` | torch CUDA bug on 3.14 + shap | py312 |
| Ch15 `06_fed_announcement_bsts` | tfcausalimpact requires Python 3.12 | py312 |
| Ch21 `05_deep_hedging_pfhedge` | pfhedge requires numpy<2 | py312 |
| Ch02 `21_storage_benchmark_database` | requires database services | benchmark |

For these, use `docker compose` with the appropriate profile even if your main workflow is local.

### GPU with Local Setup

PyTorch auto-detects NVIDIA GPUs when CUDA drivers are installed. No special configuration needed:

```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

GPU-intensive notebooks (Ch05 GANs, Ch13 deep learning, Ch14 autoencoders, Ch21 RL) benefit from GPU but all include CPU fallback with reduced parameters.

---

## Next Steps

- [Running Notebooks](running-notebooks.md) — How to execute notebooks, Papermill test mode, case study pipelines
- [Data Guide](../data/README.md) — Download required datasets
