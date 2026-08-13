# Installation Guide

This guide walks you through setting up Docker to run the ML4T notebooks. Pre-built images on Docker Hub mean you can be running notebooks in minutes.

---

## Before You Begin

Every command in this guide is typed into a **terminal** on your own computer, not into the
GitHub website. GitHub stores the code; your terminal is where you tell your machine to fetch
and run it.

| Platform | How to open a terminal |
|----------|------------------------|
| **Windows** | Start menu → type `PowerShell` → open *Windows PowerShell*. Some steps below need *Run as administrator* (right-click → Run as administrator). You use PowerShell only to **set up WSL2**; once WSL2 is running, every other command in this guide is typed into the **Ubuntu** terminal it gives you, not into PowerShell. |
| **macOS** | Applications → Utilities → *Terminal* |
| **Linux** | `Ctrl+Alt+T`, or search for *Terminal* |

Commands shown in a block like this are typed at the terminal prompt, one line at a time,
then Enter:

```bash
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
```

Type flags exactly as written. `--install` is two dashes attached to the word with no space
before it. `wsl -- install` is a different command and will not do what you want.

If a command is not found, the tool it belongs to is not installed yet. `git` ships with
[Git for Windows](https://git-scm.com/download/win) and with the Xcode command-line tools on
macOS (`xcode-select --install`).

### What you need before either path

| | Docker path | Local `uv` path |
|---|---|---|
| `git` | yes | yes |
| Docker Desktop or Docker Engine | yes | no |
| **C/C++ compiler and Python headers** | no, the image carries both | **yes** |
| Disk | ~13 GB image + ~4 GB data | ~11 GB environment + ~4 GB data + ~1 GB git history |

The compiler is not optional on the local path and it is the most common way a first install
fails. Twelve locked packages publish no wheel for Python 3.14, so `uv` builds them from
source; six of those are C or C++ (`scikit-learn`, `shap`, `hmmlearn`, `ruptures`, `econml`,
`causalml`). Without a compiler `uv sync` stops with:

```
error: command 'c++' failed: No such file or directory
```

Install one first:

```bash
sudo apt install build-essential python3-dev   # Ubuntu, Debian, and inside WSL2
xcode-select --install                         # macOS
```

**macOS also needs the OpenMP runtime, and this one fails later rather than at `uv sync`.** The
LightGBM wheel links against `libomp`, which macOS does not ship and the compiler tools do not
install. Without it the environment builds cleanly and then Chapter 12 stops at its first import:

```
OSError: dlopen(.../lightgbm/lib/lib_lightgbm.dylib): Library not loaded: @rpath/libomp.dylib
```

The supported way to get it is [Homebrew](https://brew.sh), which a fresh macOS does not have
either. Install it first if `brew` is not already on your `PATH`. Its installer prints PATH
instructions rather than applying them, so the `eval` below is what it asks you to run, given
here for Apple Silicon (`/usr/local/bin/brew` on Intel):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"      # puts brew on PATH in this shell
brew install libomp                            # macOS, both chips
```

`python3-dev` is the second half of the requirement on any distribution whose own `python3` is
already 3.14 or newer, Ubuntu 26.04 being the first. `uv` downloads a managed CPython only when
no installed interpreter satisfies the project's floor; when the system one does, `uv` builds
against it, and Debian and Ubuntu ship that interpreter without its headers. Every source build
then stops with:

```
fatal error: Python.h: No such file or directory
```

On a distribution whose Python is older, such as Ubuntu 24.04, the package changes nothing:
`uv` fetches its own CPython, which carries its headers with it. Installing both is correct in
either case.

**Docker is the one path that avoids this**, because the image ships its own toolchain. WSL2
does not avoid it: a local `uv` environment inside WSL2 is the Linux path, so it needs the same
two packages that native Linux does. What WSL2 avoids is the *Windows* build, which
does not work at all — see the note under [Platform Support](#platform-support).

---

## Platform Support

| Platform                | ml4t | py312  | Benchmark | GPU |
|-------------------------|:----:|:------:|:---------:|:---:|
| **Linux x86_64**        |  ✅  |   ✅   |    ✅     | ✅* |
| **Windows 11 (WSL2)**   |  ✅  |   ✅   |    ✅     | ✅* |
| **macOS Intel**         |  ✅  |   ✅   |    ✅     |  -  |
| **macOS Apple Silicon** |  ✅  |   †    |    ✅     |  -  |

\* Requires NVIDIA GPU + nvidia-container-toolkit
† `ml4t-py312` is amd64 only. It has no native build on Apple Silicon and runs under Rosetta
emulation, which [Py312 Image](#py312-image-specific-notebooks) covers.

The table is about the **Docker images**, which work on all four rows. The local `uv` path is
narrower: it works on Linux, on Apple Silicon, and inside WSL2 (which is the Linux path), and it
does not work on Intel Macs or in native Windows Python.

> **Windows: use WSL2, not PowerShell.** Everything on Windows runs inside WSL2, whether you
> pick Docker or the local `uv` environment. Installing directly into Windows Python is not
> supported and does not work: the dependency set resolves `scikit-learn 1.6.1`, which has no
> Python 3.14 wheel for Windows, and building it from source fails partway through even on a
> machine that already has the Visual Studio Build Tools. Inside WSL2 you are on the Linux path
> above, which is the one that is tested.

> **macOS: which path depends on the chip.** On **Apple Silicon**, use the local `uv`
> environment: it builds natively against the Xcode command-line tools. Docker is worth adding
> there only for the twelve `ml4t-py312` notebooks, which have no arm64 build, and for Chapter
> 2's containerized database benchmarks. On
> an **Intel Mac**, Docker is the only option, because PyTorch stopped publishing macOS x86_64
> wheels and `uv sync` stops immediately with `Distribution torch==2.10.0 ... doesn't have a
> source distribution or wheel for the current platform`. There is nothing to configure around
> it. See [macOS](#macos).

### Which image do I need?

| Image | What it covers | Platforms |
|-------|----------------|-----------|
| **ml4t** | All chapters (Ch01-Ch27) + all 9 case studies | amd64 + arm64 |
| **ml4t-py312** | Ch05 NB01/03/07, Ch09 NB06/12, Ch10 NB01-03, Ch12 NB10, Ch14 NB06, Ch15 NB06 (signatory, esig, gensim, tfcausalimpact) | amd64 only |
| **benchmark** | Ch02 storage benchmarks (DuckDB, HDF5, database clients) | amd64 + arm64 |
| **rapids** | Ch12 GBM GPU benchmark (RAPIDS cuML, LightGBM CUDA) | amd64 + NVIDIA GPU |

**Most readers need only `ml4t`.** The other images are for specific notebooks.

**Apple Silicon users**: `signatory` and `esig` have no ARM64 builds, so the `ml4t-py312`
notebooks do not run natively. They all ship pre-executed, and
[Py312 Image](#py312-image-specific-notebooks) covers both reading them and running them under
Rosetta. Nothing else in the book needs this.

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

## Verify Your Installation

Before opening any notebook, run the one command that confirms every required
library imports and the runtime is wired up correctly:

```bash
# Docker (recommended)
docker compose run --rm ml4t python scripts/verify_installation.py

# Local uv
uv run python scripts/verify_installation.py
```

It prints a `PASS`/`FAIL` line for each component — core libraries, PyTorch and
CUDA, repo-root imports, plotting, and your data path — followed by a summary.
**If every line says PASS, you are ready.** If a line says FAIL, it names the
missing piece; see [Troubleshooting](#troubleshooting) below.

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

Docker Desktop on Windows runs its engine inside WSL2. WSL2 must be working before Docker
Desktop can start, so complete steps 1-3 in order and do not skip the restart.

0. **Check that hardware virtualization is on.** WSL2 cannot run without it, and it is disabled
   by default on some machines. Press `Ctrl+Shift+Esc` → *Performance* tab → *CPU*, and look for
   **Virtualization**.

   - **Enabled**: continue to step 1.
   - **Disabled**: turn it on in your BIOS/UEFI setup screen, where it is called *Intel VT-x*,
     *AMD-V*, or *Virtualization Technology*. The key to enter setup varies by manufacturer
     (commonly `F2`, `F10`, or `Del` during boot). Nothing below will work until this reads Enabled.

1. **Install WSL2 and a Linux distribution.** Open PowerShell **as Administrator**:
   ```powershell
   wsl --install -d Ubuntu
   ```

   Two dashes, no space: `--install`, not `-- install`. Keep the `-d Ubuntu`: without it, some
   Windows builds install the WSL runtime and no Linux distribution at all.

   On a machine that has never had WSL, expect this run to enable the Windows features and
   install the WSL runtime **without** installing Ubuntu. It prints `Changes will not be
   effective until the system is rebooted` and says nothing about a distribution. That is the
   normal path, and step 3 completes it.

2. **Restart your computer.** This is a required step, not a conditional one. `wsl --install`
   enables a Windows feature that does not take effect until you reboot, and Windows does not
   always prompt you. If the command printed `The operation completed successfully`, restart now.

   Nothing opens by itself after the restart.

3. **Run the same command again**, in an Administrator PowerShell:
   ```powershell
   wsl --install -d Ubuntu
   ```
   This is the run that prints `Downloading: Ubuntu`, `Installing: Ubuntu` and `Distribution
   successfully installed`. If the first run already installed Ubuntu, this one reports that it
   is already installed and changes nothing.

   Then open **Ubuntu** from the Start menu. Its first launch asks you to create a username and
   password; the password is not echoed as you type, which is expected.

4. **Verify WSL2 before installing Docker.** In PowerShell:
   ```powershell
   wsl --list --verbose
   ```
   You should see `Ubuntu` with `STATE  Running` (or `Stopped`) and `VERSION  2`. If you get
   `Windows Subsystem for Linux has no installed distributions`, step 3 did not complete, so run
   it again. If `VERSION` reads `1`, run `wsl --set-version Ubuntu 2`.

5. **Increase WSL2 memory limit** *(optional — skip unless a notebook runs out of memory; most
   chapters are fine on the default)*: WSL2 defaults to 50% of host RAM, which may not be enough for
   data-heavy notebooks. The `%USERPROFILE%\.wslconfig` file lives in your **Windows** home folder, so
   create it from **Windows** PowerShell (a regular window, not admin), not from inside Ubuntu. Paste
   this one line to create it with the recommended settings:
   ```powershell
   Set-Content -Path "$env:USERPROFILE\.wslconfig" -Value "[wsl2]`nmemory=12GB`nswap=4GB"
   ```
   That writes:
   ```ini
   [wsl2]
   memory=12GB
   swap=4GB
   ```
   Then apply it by restarting WSL: `wsl --shutdown` from PowerShell, then reopen your terminal.

6. **Install Docker Desktop.** This is a **Windows program you download in your web browser** — not
   a command you type into a terminal. Open [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
   in Edge or Chrome, click **Download for Windows**, and run the downloaded `Docker Desktop
   Installer.exe`. Do **not** type `docker.com/...` into PowerShell or the Ubuntu terminal — that
   address is a web link, not a command.

   Install it only after `wsl --list --verbose` shows a `VERSION 2` distribution. Docker Desktop
   started against a non-working WSL2 backend hangs on "Starting the Docker Engine…" indefinitely.

   - Ensure "Use WSL 2 based engine" is checked in Settings → General
   - In Settings → Resources → WSL Integration, enable your Ubuntu distribution
   - In Settings → Resources, allocate at least 8 GB memory and 60 GB disk

7. **Verify Docker Desktop integration**: Open your WSL Ubuntu terminal and run:
   ```bash
   docker version
   ```
   If this fails with "Cannot connect to the Docker daemon", Docker Desktop's WSL integration is not enabled for your distribution. Check step 6 above.

8. **Clone in WSL** (not on Windows drives — much faster):
   ```bash
   cd ~
   git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
   cd machine-learning-for-trading
   cp .env.example .env
   docker compose pull ml4t
   ```

**Important**: Always run `docker` commands from inside a WSL terminal (Ubuntu), not from Windows PowerShell or Command Prompt. Docker Desktop exposes the Docker socket to WSL distributions, but the Docker CLI in Windows may behave differently.

**Tip**: Keep the repo at `~/machine-learning-for-trading` in WSL, not under `/mnt/c/...`. Windows drives reach WSL through the 9P protocol bridge, which costs roughly 8x on a 512 MB sequential write, 240x on creating two thousand small files, and 470x on reading their metadata. `git clone` and `uv sync` are almost entirely small-file and metadata work, so those last two ratios are the ones a reader pays. Access WSL files from Windows Explorer via `\\wsl$\Ubuntu\home\<username>\machine-learning-for-trading`.

### macOS

**Apple Silicon: use the local `uv` path, not Docker.** Everything in the main environment either
has an arm64 wheel or builds from source against the Xcode command-line tools, the same way it
does on Linux, so Docker would add an image you have no use for. Go to
[Local Setup with uv](#local-setup-with-uv-alternative-to-docker); this is the path walked on
real hardware before every release. Two things still want Docker on that machine: the twelve
`ml4t-py312` notebooks, which have no arm64 build at all and are covered under
[Py312 Image](#py312-image-specific-notebooks), and Chapter 2's storage benchmarks, which compare
databases that run as containers. Everything else is `uv`.

```bash
xcode-select --install                        # compiler, if you do not have it already
brew install libomp                           # OpenMP runtime; LightGBM will not import without it.
                                              # Needs Homebrew - see the prerequisites section above
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env                   # puts uv on PATH in this shell
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
cd machine-learning-for-trading
cp .env.example .env
uv sync
```

**Intel Macs: Docker is the only local option.** PyTorch publishes no macOS x86_64 wheel, so the
`uv` path cannot be made to work on that hardware. The `ml4t` image is amd64 and runs, so:

1. Install Docker Desktop from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/),
   choosing the **Intel chip** download. Give it 4+ CPUs, 8+ GB memory and 64+ GB disk in
   Settings → Resources, and note that the image plus data wants about 17 GB of that disk.
2. Clone and pull:
   ```bash
   git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
   cd machine-learning-for-trading
   cp .env.example .env
   docker compose pull ml4t
   ```

If that machine is tight on memory or disk, a Linux box or a cloud instance is the more
comfortable route, and it is the same Linux path documented above.

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

```bash
# On x86 (Linux, Windows WSL2, Intel Mac) run these as they stand. On Apple Silicon
# prefix each with DOCKER_DEFAULT_PLATFORM=linux/amd64, see below.
docker compose --profile py312 pull py312
docker compose --profile py312 run --rm py312 python 09_model_based_features/06_path_signatures.py
docker compose --profile py312 run --rm py312 \
  /opt/bsts/bin/python 15_causal_estimation/06_fed_announcement_bsts.py

# The six GPU-tagged notebooks, on a machine with an NVIDIA GPU:
docker compose --profile py312-gpu run --rm py312-gpu \
  python 05_synthetic_data/03_sigcwgan_signatures.py
```

Chapter 15 notebook 06 uses the isolated `/opt/bsts` interpreter so its NumPy 1 and
pandas 2.2 constraints do not replace dependencies required by the other py312 notebooks.

The `py312` service reserves no GPU, so it runs anywhere the amd64 image does. Six of the
eleven are GPU-tagged and run faster with one, whether they train or only do inference: Ch05
`01_timegan`, `03_sigcwgan_signatures` and `07_dp_gan`, Ch10 `03_sentiment_evolution`, Ch12
`10_shap_nlp_sentiment` and Ch14 `06_conditional_autoencoder`. The `py312-gpu` service is the
same image with an NVIDIA GPU attached, for those.

**Apple Silicon**: these notebooks have no arm64 build and all of them ship pre-executed, so
reading the `.ipynb` in Jupyter or on GitHub is the intended route, and the local `uv` path
covers everything else in the book. To execute them anyway you need Docker, which the Apple
Silicon setup above does not install:

1. Install Docker Desktop from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/),
   choosing the **Apple chip** download.
2. Enable Settings → General → **Use Rosetta for x86_64/amd64 emulation**.
3. Prefix the **`py312`** commands above with `DOCKER_DEFAULT_PLATFORM=linux/amd64`, for example
   `DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose --profile py312 pull py312`. Skip the
   `py312-gpu` one: it reserves an NVIDIA device, which no Mac has, so all twelve notebooks go
   through the CPU-only service here.

It runs at emulation speed. Apart from Chapter 2's database benchmarks, this is the only thing
on an Apple Silicon Mac that Docker is needed for.

---

## Troubleshooting

### Docker Desktop hangs on "Starting the Docker Engine…" (Windows)

Check the status bar at the bottom of the Docker Desktop window. If it reads `RAM 0.00 GB`
and `CPU 0.00%`, the engine's virtual machine never started, and the cause is the WSL2
backend rather than Docker itself. Work through it in this order:

1. **Virtualization off in firmware.** Task Manager → Performance → CPU → *Virtualization*.
   If it says Disabled, enable Intel VT-x / AMD-V in BIOS/UEFI. See step 0 of the
   [Windows setup](#windows-11-wsl2) above.
2. **Pending reboot.** If you ran `wsl --install` and did not restart, restart now.
3. **No Linux distribution.** Run `wsl --list --verbose` in PowerShell. If it prints
   `has no installed distributions`, run `wsl --install -d Ubuntu` again. On a machine that has
   already rebooted, this second run is what downloads and installs Ubuntu.
4. **WSL2 backend not selected.** Docker Desktop → Settings → General → "Use the WSL 2
   based engine".

Then quit Docker Desktop fully (right-click the tray icon → Quit) and start it again.

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

### "Kaleido requires Google Chrome to be installed"

Notebooks pick their Plotly renderer automatically and no longer ask for static
PNGs where Chrome is missing, so this should not appear. If you are on an older
image, either pull the current one or set the renderer explicitly:

```bash
docker compose pull ml4t
docker compose run --rm -e PLOTLY_RENDERER=json ml4t python case_studies/etfs/05_evaluation.py
```

Chrome installed on your *host* has no bearing on this: the notebook runs inside
the container, which ships without one.

Only static image export (`fig.write_image`, `fig.to_image`) genuinely needs
Chrome. Install it into a *running* container, since `docker compose run --rm`
discards the download when the container exits:

```bash
docker compose up -d ml4t
docker compose exec ml4t plotly_get_chrome -y
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
# Install uv — use this installer, not `pip install uv`. Most current systems either
# ship no `pip` at all or refuse the install with `externally-managed-environment`.
curl -LsSf https://astral.sh/uv/install.sh | sh
# The installer puts uv in ~/.local/bin, which your current shell does not know about
# yet. Load it now rather than opening a new terminal:
source $HOME/.local/bin/env        # sh, bash, zsh;  env.fish for fish
# Windows PowerShell: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and enter the repository (about 0.9 GB of history)
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
cd machine-learning-for-trading

# Install all dependencies (creates .venv/, installs ~300 packages, about 11 GB).
# Twelve of them compile from source, so a C/C++ compiler must already be installed —
# see "What you need before either path" above.
uv sync

# Copy environment template (defaults work as-is; no editing needed to start)
cp .env.example .env
# API keys are optional and only needed for specific datasets later —
# see data/README.md when a chapter asks for one.

# Verify
uv run python -c "import polars, torch, lightgbm; print('Ready')"

# Start Jupyter Lab, from the repo root, and open the URL it prints
ML4T_DATA_PATH="${ML4T_DATA_PATH:-$PWD/data}" uv run jupyter lab
```

`uv sync` installs Jupyter Lab along with everything else. The `ML4T_DATA_PATH` prefix gives the data
loaders an absolute path: Jupyter runs each notebook with its own chapter folder as the working
directory, so they would otherwise search inside that folder and report the datasets as missing. The
form above keeps a value you have already exported and falls back to this repository's `data/`.

Jupyter prints its address with a freshly generated access token attached
(`http://localhost:8888/lab?token=…`). Open that whole line; a bare `http://localhost:8888` only
shows a token prompt. On Windows, run the command in your WSL2 Ubuntu terminal and paste the URL into
your normal Windows browser - no browser opens by itself there, and WSL2 forwards `localhost` for
you.

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
