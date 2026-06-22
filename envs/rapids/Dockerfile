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

# Core packages needed for the benchmark notebook
RUN pip install --no-cache-dir \
    "polars>=1.0" \
    "matplotlib>=3.8" \
    "scikit-learn>=1.5" \
    "catboost>=1.2" \
    "psutil>=5.9" \
    "jupyterlab>=4.1" \
    "ml4t-data>=0.1.0b9" \
    "ml4t-diagnostic>=0.1.0b7" \
    "ml4t-engineer>=0.1.0b2" \
    "pytest>=8.0" \
    "pyyaml>=6.0" \
    --pre

# XGBoost: already included in RAPIDS with GPU support — verify
RUN python -c "import xgboost; print(f'XGBoost {xgboost.__version__}')"

# LightGBM: build from source with CUDA (mandatory, no fallback)
# NOTE: GPU verification cannot run during build (no GPU access).
# detect_gpu_capabilities() verifies at runtime.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --no-binary lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON "lightgbm>=4.6" && \
    python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__} — CUDA build installed')"

# CatBoost: verify import (GPU test requires runtime GPU access)
RUN python -c "from catboost import CatBoostRegressor; print('CatBoost installed')"

WORKDIR /app
ENV PYTHONPATH="/app"

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
