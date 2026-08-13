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
# RAPIDS is a compiled environment, so its core scientific packages must remain
# at the versions shipped by the base image. For example, cuDF 25.4 requires
# pandas<2.2.4 and pyarrow<20, CuPy requires numpy<2.3, and cudf-polars requires
# polars<1.26. The root lock is authoritative for packages outside that stack.
# Stable ml4t-diagnostic requires scipy>=1.17, while this base image ships
# scipy 1.15.2. SciPy is the one tracked package that moves to the root lock;
# every other tracked package remains at the base version. Verify both sets.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /build/root/
COPY envs/rapids/base_constraints.py /build/base_constraints.py
RUN python /build/base_constraints.py > /tmp/rapids-base-versions.txt && \
    grep -viE '^scipy==' /tmp/rapids-base-versions.txt > /tmp/rapids-base-constraints.txt && \
    test -s /tmp/rapids-base-constraints.txt && \
    cd /build/root && uv export --frozen --no-emit-project --no-hashes --format requirements-txt \
      | grep -iE '^[a-z0-9._-]+==' \
      > /tmp/root-constraints.txt && \
    grep -iE '^scipy==' /tmp/root-constraints.txt | head -1 | cut -d';' -f1 \
      | sed 's/[[:space:]]//g' \
      > /tmp/scipy-pin.txt && \
    test -s /tmp/scipy-pin.txt && \
    python -c 'from pathlib import Path; import re; norm=lambda value: re.sub(r"[-_.]+", "-", value.lower()); base={norm(line.split("==", 1)[0]) for line in Path("/tmp/rapids-base-constraints.txt").read_text().splitlines()}; root=Path("/tmp/root-constraints.txt").read_text().splitlines(); Path("/tmp/locked-constraints.txt").write_text("\n".join(line for line in root if norm(line.split("==", 1)[0]) not in base) + "\n")' && \
    test -s /tmp/locked-constraints.txt && \
    grep -iE '^lightgbm==' /tmp/locked-constraints.txt | head -1 | cut -d';' -f1 | tr -d '[:space:]' \
      > /tmp/lightgbm-pin.txt && \
    test -s /tmp/lightgbm-pin.txt && cat /tmp/lightgbm-pin.txt

# Core packages needed for the benchmark notebook
RUN pip install --no-cache-dir \
    -c /tmp/rapids-base-constraints.txt \
    -c /tmp/locked-constraints.txt \
    "polars>=1.0" \
    "matplotlib>=3.8" \
    "scikit-learn>=1.5" \
    "catboost>=1.2" \
    "psutil>=5.9" \
    "jupyterlab>=4.1" \
    "ml4t-data>=0.1.2" \
    "ml4t-diagnostic>=0.1.1" \
    "ml4t-engineer>=0.1.3" \
    "pytest>=8.0" \
    "pyyaml>=6.0" && \
    python -c "import cudf, cuml; print('cudf', cudf.__version__, 'cuml', cuml.__version__)" && \
    python /build/base_constraints.py > /tmp/rapids-final-versions.txt && \
    grep -viE '^scipy==' /tmp/rapids-final-versions.txt > /tmp/rapids-final-constraints.txt && \
    grep -iE '^scipy==' /tmp/rapids-final-versions.txt > /tmp/scipy-final.txt && \
    diff -u /tmp/rapids-base-constraints.txt /tmp/rapids-final-constraints.txt && \
    diff -u /tmp/scipy-pin.txt /tmp/scipy-final.txt

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
