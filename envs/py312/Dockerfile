# ML4T Python 3.12 Environment (x86 only)
# For packages without Python 3.14 support: gensim, signatory, esig, pfhedge, pycausalimpact
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
#   Ch15 NB06 (fed_announcement_bsts) — pycausalimpact (ARIMA event-study port; py<3.13)
#   Ch21 deep_hedging_pfhedge        — pfhedge (unmaintained, numpy<2)
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
RUN uv pip install --python /opt/ml4t/bin/python --no-cache /build

# x86-only signature stack (signatory + esig) for Ch05/Ch09 notebooks
RUN /opt/ml4t/bin/pip install --no-cache-dir --no-build-isolation "signatory>=1.2" && \
    /opt/ml4t/bin/pip install --no-cache-dir --no-deps "esig>=1.0" && \
    /opt/ml4t/bin/pip install --no-cache-dir "pyrecombine>=1.0" "roughpy>=0.2"

# Test infrastructure — pytest + helpers so CI doesn't need runtime pip install
RUN /opt/ml4t/bin/pip install --no-cache-dir \
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
