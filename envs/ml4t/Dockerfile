# ML4T Unified Environment — Multi-Architecture
# Single Dockerfile builds for both linux/amd64 (x86) and linux/arm64 (Apple Silicon).
#
# Architecture strategy (via Docker BuildKit TARGETARCH):
#   amd64: CUDA 12.8 PyTorch wheels (~5.4GB) + signatory/esig + LightGBM CUDA
#   arm64: CPU-only PyTorch (~200MB), skip signatory/esig (no ARM wheels)
#
# Usage:
#   docker compose up ml4t                    # Start Jupyter Lab (CPU)
#   docker compose up ml4t-gpu                # Start with GPU passthrough
#   docker compose run --rm ml4t ml4t-status  # Check environment
#
# See envs/README.md for detailed setup instructions

FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System tooling only — Python 3.14 comes from uv (python-build-standalone),
# not from the OS. This avoids the deadsnakes PPA's intermittent network
# flakiness on GitHub runners.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git libhdf5-dev pkg-config wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN curl -L https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz | tar xz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf ta-lib-0.6.4

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -s /bin/bash ml4t

# Install Python 3.14 via uv (python-build-standalone) and create venv at
# /opt/ml4t. /opt/ml4t is on PATH later, so /opt/ml4t/bin/python becomes the
# default `python` for downstream RUN steps. No system Python needed.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.14 \
    && uv venv --seed --python 3.14 /opt/ml4t \
    && /opt/ml4t/bin/python --version \
    && /opt/ml4t/bin/pip --version

ARG TARGETARCH

# Install PyTorch FIRST (rarely changes - maximize cache reuse)
# amd64: CUDA 12.8 wheels (~5.4GB) bundle CUDA runtime, work on CPU-only machines too
# arm64: CPU-only wheels (~200MB) — no NVIDIA GPU on Apple Silicon
RUN /opt/ml4t/bin/pip install --no-cache-dir pip setuptools wheel && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      /opt/ml4t/bin/pip install --no-cache-dir torch torchvision torchaudio; \
    else \
      /opt/ml4t/bin/pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; \
    fi
RUN /opt/ml4t/bin/pip freeze | grep -E '^(torch|torchaudio|torchvision)==' > /tmp/ml4t-constraints.txt

# Install ml4t libraries from PyPI
# --pre allows pre-release versions
RUN /opt/ml4t/bin/pip install --no-cache-dir --pre \
    "ml4t-data>=0.1.0b11" \
    "ml4t-diagnostic>=0.1.0b19" \
    "ml4t-engineer>=0.1.0b8" \
    "ml4t-live>=0.1.0b3" \
    "ml4t-backtest>=0.1.0b16" \
    "ml4t-models>=0.1.0a1"

# amd64 only: Install CUDA toolkit (nvcc + headers) for building LightGBM with CUDA
# Minimal install: nvcc-12.6 + runtime dev headers (12.4 not packaged for Ubuntu 24.04)
# CUDA is backward-compatible: 12.6 nvcc compiles for 12.4+ runtime
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends gnupg && \
      wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
      dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
      apt-get update && \
      apt-get install -y --no-install-recommends cuda-nvcc-12-6 cuda-cudart-dev-12-6 && \
      rm -rf /var/lib/apt/lists/*; \
    fi
ENV PATH="${PATH:+${PATH}:}/usr/local/cuda-12.6/bin"

# NOW copy pyproject.toml (changes here only rebuild deps, not PyTorch)
# Uses root pyproject.toml — single source of truth for all dependencies.
WORKDIR /build
COPY pyproject.toml /build/

# Remove packages that need special handling from pyproject copy:
# - lightgbm: installed separately with CUDA build on amd64
# - torch: already installed from CUDA wheels above
# Packages stripped here are installed separately or via special build steps
RUN sed -i '/"lightgbm/d; /"torch>=/d' /build/pyproject.toml

# signatory/esig/gensim: moved to py312 image (Python 3.12, x86 only)
# Use: docker compose --profile py312 run --rm py312 python ...

# Install remaining dependencies from pyproject.toml without replacing the
# preinstalled PyTorch stack needed for CUDA/signatory compatibility.
# Use uv for resolution (handles packages that mark Requires-Python <3.14
# but actually work fine, like dowhy and granite-tsfm).
# --override lets uv resolve freely while keeping torch CUDA pins.
# Force cvxpy>=1.5 for numpy 2.x C API compat (riskfolio-lib pins 1.4).
# --prerelease=if-necessary-or-explicit mirrors the [tool.uv] policy used by
# uv.lock, so the image resolves the same STABLE versions we test against rather
# than gratuitous prereleases (e.g. scipy 1.18rc, pydantic 2.14a, matplotlib 3.11rc).
RUN echo 'cvxpy>=1.5' >> /tmp/ml4t-constraints.txt && \
    uv pip install --python /opt/ml4t/bin/python --no-cache \
    --prerelease=if-necessary-or-explicit \
    --override /tmp/ml4t-constraints.txt /build

# LightGBM: two deterministic paths, no fallbacks within a platform
# - amd64: CUDA build from source. Mandatory. Build fails if CUDA doesn't work.
# - arm64 (macOS): pip wheel. No GPU available, no pretense of one.
#
# amd64: build from source with CUDA, verify the CUDA codepath compiled
# NOTE: Full CUDA verification (lgb.train device=cuda) requires a GPU at runtime.
# At build time we only verify the library imports and reports CUDA support.
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      set -e && \
      apt-get update && apt-get install -y --no-install-recommends \
        libnccl-dev libnccl2 gcc-12 g++-12 && \
      rm -rf /var/lib/apt/lists/* && \
      CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
      CUDACXX=/usr/local/cuda/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-12 \
      /opt/ml4t/bin/pip install --no-cache-dir --force-reinstall --no-deps \
        --no-binary lightgbm \
        --config-settings=cmake.define.USE_CUDA=ON "lightgbm>=4.6" && \
      /opt/ml4t/bin/python -c "\
import lightgbm as lgb; \
print(f'LightGBM {lgb.__version__}'); \
print('CUDA build: OK (runtime verification requires GPU)')"; \
    fi
# arm64: standard pip wheel (no GPU on macOS)
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      /opt/ml4t/bin/pip install --no-cache-dir "lightgbm>=4.6"; \
    fi

# Install optional dependency groups (bayesian, advanced, mlops)
# Skip rapids (requires specific CUDA version matching)
RUN /opt/ml4t/bin/pip install --no-cache-dir \
    "pymc>=5.14" "arviz>=0.18" \
    "opacus>=1.4" \
    "feast>=0.61" "mlflow>=2.16" \
    || echo "WARNING: Some optional deps failed"

# Python 3.14 compatibility: protobuf 4.x uses metaclass tp_new (crashes).
# Force protobuf 5+ which uses the pure-Python descriptor pool.
RUN /opt/ml4t/bin/pip install --no-cache-dir "protobuf>=5.0"

WORKDIR /app
ENV PATH="/opt/ml4t/bin:$PATH"
ENV PYTHONPATH="/app"

# For GPU users: CUDA libraries are bundled in PyTorch wheel
# nvidia-container-toolkit passes through /dev/nvidia* and sets these:
# - NVIDIA_VISIBLE_DEVICES
# - LD_LIBRARY_PATH (for libnvidia-ml.so, etc.)

# Jupyter configuration - disable token authentication
ENV JUPYTER_ENABLE_LAB=yes
RUN mkdir -p /etc/jupyter && \
    echo "c.ServerApp.token = ''" >> /etc/jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /etc/jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> /etc/jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.disable_check_xsrf = True" >> /etc/jupyter/jupyter_server_config.py

# NOTE: jupyterlab-plotly extension is required for Plotly widget rendering
# It is installed automatically by plotly pip package - do NOT delete it

# Disable catboost-widget extension (causes "require is not defined" errors in JupyterLab 4.x)
# The widget is bundled with catboost but not compatible with modern JupyterLab
RUN mkdir -p /etc/jupyter/labconfig && \
    echo '{"disabledExtensions": {"catboost-widget": true}}' > /etc/jupyter/labconfig/page_config.json

# Test infrastructure — pytest + helpers so CI doesn't need runtime pip install
RUN /opt/ml4t/bin/pip install --no-cache-dir \
    "pytest>=8.0" "pytest-timeout>=2.3"

# Copy import test script
COPY envs/test_all_imports.py /opt/ml4t/bin/test_imports.py
RUN chmod +x /opt/ml4t/bin/test_imports.py

# Health check script - reports GPU status
RUN echo '#!/bin/bash\n\
echo "============================================"\n\
echo "ML4T Environment Status"\n\
echo "============================================"\n\
python -c "\n\
import torch\n\
print(f\"PyTorch version: {torch.__version__}\")\n\
print(f\"CUDA available: {torch.cuda.is_available()}\")\n\
if torch.cuda.is_available():\n\
    print(f\"CUDA version: {torch.version.cuda}\")\n\
    print(f\"GPU count: {torch.cuda.device_count()}\")\n\
    for i in range(torch.cuda.device_count()):\n\
        print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\")\n\
else:\n\
    print(\"Running in CPU-only mode (this is fine!)\")\n\
"\n\
echo ""\n\
echo "Run: ml4t-test-imports     # Test all chapter imports"\n\
echo "Run: ml4t-test-imports 15  # Test specific chapter"\n\
echo "============================================"' > /usr/local/bin/ml4t-status && chmod +x /usr/local/bin/ml4t-status

# Import test wrapper
RUN echo '#!/bin/bash\npython /opt/ml4t/bin/test_imports.py "$@"' > /usr/local/bin/ml4t-test-imports && \
    chmod +x /usr/local/bin/ml4t-test-imports

# Default command
# Note: --allow-root removed; docker-compose.yml passes host UID:GID
# Token/password disabled via /etc/jupyter/jupyter_server_config.py
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
