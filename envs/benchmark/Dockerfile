# ML4T Benchmark Environment
# Storage benchmarks: Parquet, DuckDB, HDF5, database clients
#
# Usage:
#   docker compose build benchmark
#   docker compose run --rm benchmark python 02_financial_data_universe/18_storage_benchmark_database.py
#
# Note: ArcticDB excluded (no Linux ARM64 wheels on PyPI)
# Use benchmark-full profile on x86 systems for ArcticDB benchmarks

FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies.
# A modern Rust toolchain is required to build the questdb python client from
# sdist on linux/arm64 (PyPI ships no Linux arm64 wheel for it). Debian's
# packaged `cargo` is too old, so we install rustup so questdb-rs compiles
# successfully both on amd64 (where a wheel exists) and on arm64 (sdist build).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install rustup so we have an up-to-date Rust toolchain for questdb-rs builds.
ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -s /bin/bash ml4t

# Create venv
RUN python -m venv /opt/ml4t

# Copy and install dependencies
WORKDIR /build
COPY envs/benchmark/pyproject.toml /build/
COPY pyproject.toml uv.lock /build/root/

# Derive a fully-pinned constraint from uv.lock so this image reproduces the
# locked environment readers get from `uv sync` (single source of truth:
# uv.lock), the same way envs/ml4t/Dockerfile does. Without it the resolve below
# free-resolves to the latest compatible release at build time and drifts off
# the lock — e.g. it landed pandas 3.0.2 where the lock pins 2.3.3, and pandas 3
# stores the `symbol` column ~8 bytes/row wider in PyTables, moving Ch2's HDF5
# benchmark from 71 MB to 79 MB for the identical panel and making §2.4's number
# unreproducible in the very image the notebook told readers to use.
# Only packages the lock knows about are pinned; this image's own extras (the
# database clients, pykx) are absent from the lock and resolve freely.
# kaleido is excluded: this image deliberately caps it at <1.0 (1.x needs a
# Chrome runtime this image does not carry — the notebooks' fig.write_image call
# is wrapped in try/except for exactly that reason), while the lock pins 1.3.0.
# It only affects static image export, never a benchmark number, so the image's
# cap wins and the constraint stays out of its way.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN cd /build/root && uv export --frozen --no-emit-project --no-hashes --format requirements-txt \
      | grep -iE '^[a-z0-9._-]+==' \
      | grep -viE '^kaleido==' \
      > /tmp/locked-constraints.txt

# Install all dependencies in one go. We deliberately do NOT have a fallback
# that strips questdb on failure — a successful build must include every
# database client declared in pyproject.toml so users never hit a runtime
# `ModuleNotFoundError`.
RUN /opt/ml4t/bin/pip install --no-cache-dir pip setuptools wheel && \
    /opt/ml4t/bin/pip install --no-cache-dir -c /tmp/locked-constraints.txt /build && \
    /opt/ml4t/bin/pip install --no-cache-dir -c /tmp/locked-constraints.txt \
        "pytest>=8.0" "pytest-timeout>=2.3" "papermill>=2.6" "jupytext>=1.16"

WORKDIR /app
ENV PATH="/opt/ml4t/bin:$PATH"
ENV PYTHONPATH="/app"

# Jupyter configuration
ENV JUPYTER_ENABLE_LAB=yes
# Token-less, like envs/ml4t/Dockerfile: safe only because compose binds the
# published port to 127.0.0.1. Origin/XSRF checks stay on - see that file.
RUN mkdir -p /etc/jupyter && \
    echo "c.ServerApp.token = ''" >> /etc/jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /etc/jupyter/jupyter_server_config.py

# Platform detection script
RUN echo '#!/bin/bash\n\
ARCH=$(uname -m)\n\
echo "============================================"\n\
echo "ML4T Benchmark Environment"\n\
echo "============================================"\n\
echo "Platform: $(uname -s) $ARCH"\n\
echo ""\n\
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then\n\
    echo "Running on ARM64 (Apple Silicon compatible)"\n\
    echo "ArcticDB: Not available (use benchmark-full on x86)"\n\
else\n\
    echo "Running on x86_64"\n\
    echo "ArcticDB: Use benchmark-full profile for full benchmarks"\n\
fi\n\
echo ""\n\
echo "Available benchmarks:"\n\
echo "  - Parquet (PyArrow)"\n\
echo "  - DuckDB"\n\
echo "  - HDF5 (PyTables)"\n\
echo "  - ClickHouse"\n\
echo "  - TimescaleDB"\n\
echo "  - QuestDB"\n\
echo "  - InfluxDB"\n\
echo "============================================"' > /usr/local/bin/benchmark-status && chmod +x /usr/local/bin/benchmark-status

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
