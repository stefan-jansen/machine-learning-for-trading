# ML4T Python 3.12 Environment (x86 only)
# For packages without Python 3.14 support: gensim, signatory, esig, pycausalimpact
# Also covers notebooks hitting Python 3.14 torch CUDA import bug
#
# Covers:
#   Ch05 NB01 (timegan)              — torch CUDA bug on 3.14
#   Ch05 NB03 (sigcwgan_signatures)  — signatory, esig
#   Ch05 NB07 (dp_gan)               — torch CUDA bug on 3.14
#   Ch09 NB06 (path_signatures)      — signatory
#   Ch09 NB12 (wasserstein_regimes)  — esig
#   Ch10 NB01 (word2vec_training)    — gensim
#   Ch10 NB02 (asset_embeddings)     — gensim
#   Ch10 NB03 (sentiment_evolution)  — gensim
#   Ch12 NB10 (shap_nlp_sentiment)   — torch CUDA + shap
#   Ch14 NB06 (conditional_autoencoder) — torch CUDA + shap
#   Ch15 NB06 (fed_announcement_bsts) - tfcausalimpact in isolated /opt/bsts
#
# Ch21 deep_hedging_pfhedge is NOT here: pfhedge is a main dependency and 0.22.0
# imports and prices on Python 3.14 under numpy 2.3.5, so that notebook runs in
# the default image and declares `ml4t-gpu`.
#
# Usage:
#   docker compose --profile py312 run --rm py312 python 10_text_feature_engineering/01_word2vec_training.py

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libhdf5-dev \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -m -s /bin/bash ml4t
RUN python -m venv /opt/ml4t
RUN /opt/ml4t/bin/pip install --no-cache-dir pip setuptools wheel

WORKDIR /build
COPY envs/py312/pyproject.toml /build/
COPY pyproject.toml uv.lock /build/root/

# Derive a fully-pinned constraint from uv.lock so this image reproduces the
# locked environment readers get from `uv sync`, the same way envs/ml4t and
# envs/benchmark do. Without it this image free-resolved and drifted off the
# lock: the published image carries numpy 2.4.6 where the lock pins 2.3.5,
# pyarrow 25.0.0 where the lock pins 23.0.1, polars 1.42.1 for 1.41.1, plotly
# 6.9.0 for 6.7.0, and an ml4t-diagnostic older than the one in the lock. A
# reader on the container path and a reader on `uv sync` were running different
# libraries.
#
# One package is held away from the lock: kaleido, capped <1.0 here because 1.x
# needs a Chrome runtime this image does not carry (fig.write_image is wrapped in
# try/except for that reason). Same exemption, same reason, as
# envs/benchmark/Dockerfile.
# Only packages the lock knows about are pinned; this image's 3.12-only extras
# (gensim, signatory, esig, pyrecombine, roughpy) are absent from the lock and
# resolve freely.
RUN cd /build/root && uv export --frozen --no-emit-project --no-hashes --format requirements-txt \
      | grep -iE '^[a-z0-9._-]+==' \
      | grep -viE '^kaleido==' \
      > /tmp/locked-constraints.txt && \
    test -s /tmp/locked-constraints.txt
RUN uv pip install --python /opt/ml4t/bin/python --no-cache \
    --constraint /tmp/locked-constraints.txt /build

# x86-only signature stack (signatory + esig) for Ch05/Ch09 notebooks.
# numpy carries the constraint too, or this step would immediately undo the
# pinning above - it is what pulled numpy to 2.4.6 in the published image.
RUN /opt/ml4t/bin/pip install --no-cache-dir --no-build-isolation "signatory>=1.2" && \
    /opt/ml4t/bin/pip install --no-cache-dir --no-deps "esig>=1.0" && \
    /opt/ml4t/bin/pip install --no-cache-dir -c /tmp/locked-constraints.txt \
        "numpy>=2,<2.5" "pyrecombine>=1.0" "roughpy>=0.2"

# tfcausalimpact requires pandas<=2.2 and NumPy 1.x, while the signature stack
# above requires NumPy 2.x. Keep both valid dependency sets in isolated
# interpreters inside the same reader image.
RUN python -m venv /opt/bsts
COPY envs/py312/bsts-pyproject.toml /build/bsts/pyproject.toml
RUN uv pip install --python /opt/bsts/bin/python --no-cache /build/bsts && \
    /opt/bsts/bin/python -m ipykernel install --prefix=/opt/bsts --name python3 \
        --display-name "Python 3.12 (BSTS)" && \
    /opt/bsts/bin/python -c "import importlib.metadata as m; import causalimpact; assert m.version('tfcausalimpact') == '0.0.18'"
COPY envs/py312/kernels/bsts-quiet/kernel.json /opt/bsts/share/jupyter/kernels/bsts-quiet/kernel.json

# Test infrastructure — pytest + helpers so CI doesn't need runtime pip install
RUN /opt/ml4t/bin/pip install --no-cache-dir -c /tmp/locked-constraints.txt \
    "pytest>=8.0" "pytest-timeout>=2.3"

WORKDIR /app
ENV PATH="/opt/ml4t/bin:$PATH"
ENV PYTHONPATH="/app"
ENV JUPYTER_ENABLE_LAB=yes

# Token-less, like envs/ml4t/Dockerfile: safe only because compose binds the
# published port to 127.0.0.1. Origin/XSRF checks stay on - see that file.
RUN mkdir -p /etc/jupyter && \
    echo "c.ServerApp.token = ''" >> /etc/jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /etc/jupyter/jupyter_server_config.py

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
