# ML4T RAPIDS GBM Benchmark Environment
#
# Purpose: Ch12 02_gbm_comparison — GPU benchmark of gradient boosting libraries.
# Includes RAPIDS cuML (XGBoost GPU), LightGBM CUDA, CatBoost GPU, and CPU baselines.
#
# This is a single-purpose image for one notebook. It is NOT the general ml4t image.
# x86/amd64 only (RAPIDS has no ARM64 support).
#
# Usage:
#   docker compose --profile rapids run --rm rapids python 12_gradient_boosting/02_gbm_comparison.py
#
# Requires: NVIDIA GPU + nvidia-container-toolkit

# RAPIDS base image includes cuML, cuDF, and CUDA runtime
# Pin to specific RAPIDS version for reproducibility
FROM nvcr.io/nvidia/rapidsai/base:25.04-cuda12.8-py3.12

ENV DEBIAN_FRONTEND=noninteractive

# RAPIDS base image may run as non-root; switch to root for installs
USER root

# System deps for LightGBM CUDA build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libnccl-dev \
    libnccl2 \
    && rm -rf /var/lib/apt/lists/*

# Use the RAPIDS conda env's pip for all installs
# RAPIDS images use conda; we install additional packages into that env

# Derive a fully-pinned constraint from uv.lock so this image reproduces the
# locked environment readers get from `uv sync`, the same way envs/ml4t and
# envs/benchmark do. Without it this image free-resolved its ml4t-* pins from
# open lower bounds and drifted badly: the last local build carried ml4t-data
# 0.1.0b7, ml4t-diagnostic 0.1.0b2 and ml4t-engineer 0.1.0b1, all far behind the
# lock. Nothing builds this image in CI, so that drift went unnoticed.
#
# pandas and pyarrow are excluded because the RAPIDS base image caps them below
# the lock and cuDF is the reason this image exists: cudf 25.4.0 declares
# pandas<2.2.4dev0 (base ships 2.2.3, lock pins 2.3.3) and pyarrow<20.0.0a0
# (base ships 19.0.1, lock pins 23.0.1). Pinning either to the lock would pull
# them out from under cuDF. Everything else in the lock is compatible with the
# RAPIDS stack - cudf/cuml/rmm declare only numpy<3.0a0,>=1.23 and scipy>=1.8.0,
# which the locked numpy 2.3.5 and scipy 1.17.1 satisfy.
#
# The constraint reaches only what pip resolves. Everything the RAPIDS conda env
# already provides and pip never touches keeps the base image's version - about
# 28 packages, xgboost (2.1.4 against the lock's 3.2.0) and dask, xarray, fsspec,
# aiohttp, networkx and cuda-bindings among them. Forcing those would mean
# installing over the conda stack cuDF and cuML are built against, which is the
# same conflict as pandas above. Everything the benchmark actually measures is
# pinned: numpy, scipy, scikit-learn, polars, matplotlib, lightgbm, catboost and
# the three ml4t-* libraries all match the lock.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /build/root/
RUN cd /build/root && uv export --frozen --no-emit-project --no-hashes --format requirements-txt \
      | grep -iE '^[a-z0-9._-]+==' \
      | grep -viE '^(pandas|pyarrow)==' \
      > /tmp/locked-constraints.txt && \
    test -s /tmp/locked-constraints.txt && \
    grep -iE '^lightgbm==' /tmp/locked-constraints.txt | head -1 | cut -d';' -f1 | tr -d '[:space:]' \
      > /tmp/lightgbm-pin.txt && \
    test -s /tmp/lightgbm-pin.txt && cat /tmp/lightgbm-pin.txt

# Core packages needed for the benchmark notebook
RUN pip install --no-cache-dir -c /tmp/locked-constraints.txt \
    "polars>=1.0" \
    "matplotlib>=3.8" \
    "scikit-learn>=1.5" \
    "catboost>=1.2" \
    "psutil>=5.9" \
    "jupyterlab>=4.1" \
    "ml4t-data>=0.1.0b15" \
    "ml4t-diagnostic>=0.1.0b25" \
    "ml4t-engineer>=0.1.0b2" \
    "pytest>=8.0" \
    "pyyaml>=6.0" \
    --pre && \
    python -c "import cudf, cuml; print('cudf', cudf.__version__, 'cuml', cuml.__version__)"

# XGBoost: already included in RAPIDS with GPU support — verify
RUN python -c "import xgboost; print(f'XGBoost {xgboost.__version__}')"

# LightGBM: build from source with CUDA (mandatory, no fallback)
# NOTE: GPU verification cannot run during build (no GPU access).
# detect_gpu_capabilities() verifies at runtime.
#
# The version comes from uv.lock, never from a floating ">=4.6". A free resolve
# takes 4.7.0, which device-links a CUDA-13-built libnccl_static.a that nvcc's
# nvlink cannot read, and this image failed to build on 2026-08-10 with
# "nvlink fatal: Input file libnccl_static.a:common.cu.o ABI version '8' is
# incompatible with target ABI version '7' (target: sm_80)". envs/ml4t/Dockerfile
# has read the version from the lock since the same drift hit it on 2026-07-27;
# this image was left behind because nothing builds it in CI.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --no-binary lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON "$(cat /tmp/lightgbm-pin.txt)" && \
    python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__} — CUDA build installed')"

# CatBoost: verify import (GPU test requires runtime GPU access)
RUN python -c "from catboost import CatBoostRegressor; print('CatBoost installed')"

WORKDIR /app
ENV PYTHONPATH="/app"

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
