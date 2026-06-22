"""Centralized configuration using python-dotenv.

This module loads all configuration from the .env file in the repository root.
It provides explicit, fail-fast configuration with clear error messages.

Configuration priority:
1. .env file (ONLY source - no fallbacks)
2. Validation ensures paths exist or provides clear instructions

Usage:
    from utils import ML4T_PATH, ML4T_DATA_PATH, REPO_ROOT
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Find repository root (where .env lives)
REPO_ROOT = Path(__file__).parent.parent.resolve()
ENV_FILE = REPO_ROOT / ".env"

# Auto-create .env from .env.example if missing (CI, first-time setup)
if not ENV_FILE.exists():
    example = REPO_ROOT / ".env.example"
    if example.exists():
        import contextlib
        import shutil

        with contextlib.suppress(OSError):
            shutil.copy(example, ENV_FILE)
    else:
        raise FileNotFoundError(
            f".env file not found at {ENV_FILE}\n\n"
            f"Copy .env.example to .env and configure paths:\n"
            f"  cd {REPO_ROOT}\n"
            f"  cp .env.example .env\n"
            f"  # Edit .env with your paths:\n"
            f"  #   ML4T_PATH={REPO_ROOT}\n"
            f"  #   ML4T_DATA_PATH=/path/to/data\n"
        )

# Load environment variables from .env
# override=False means environment variables take precedence over .env file
load_dotenv(ENV_FILE, override=False)

# ============================================================================
# Required Paths
# ============================================================================

# Priority: 1. Environment variable, 2. .env file, 3. Default
ML4T_PATH = Path(os.getenv("ML4T_PATH", str(REPO_ROOT))).expanduser().resolve()
data_path = os.getenv("ML4T_DATA_PATH")
if data_path is None:
    ML4T_DATA_PATH = ML4T_PATH / "data"
else:
    ML4T_DATA_PATH = Path(data_path).expanduser().resolve()

# Case studies directory - centralized artifact store for all strategies
# Default: case_studies/ in repo root (git-tracked configs, gitignored binaries)
case_studies_path = os.getenv("CASE_STUDIES_DIR")
if case_studies_path is None:
    CASE_STUDIES_DIR = REPO_ROOT / "case_studies"
else:
    CASE_STUDIES_DIR = Path(case_studies_path).expanduser().resolve()

# ============================================================================
# Validation
# ============================================================================

# Validate ML4T_PATH exists
if not ML4T_PATH.exists():
    raise FileNotFoundError(
        f"ML4T_PATH does not exist: {ML4T_PATH}\n\n"
        f"Update ML4T_PATH in .env to point to the repository root."
    )

# Validate ML4T_DATA_PATH exists
if not ML4T_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data directory not found: {ML4T_DATA_PATH}\n\n"
        f"Options:\n"
        f"  1. Download data:\n"
        f"       python data/download_all.py\n"
        f"  2. Update ML4T_DATA_PATH in .env to point to existing data\n"
    )

# ============================================================================
# CUDA Library Path (local uv environments)
# ============================================================================

# PyTorch bundles its own CUDA libraries which may be newer than system ones.
# Without this, torch imports fail with "undefined symbol" errors on systems
# where the system libcudart.so is older than what torch expects.
# Docker images don't need this (CUDA is installed system-wide).
if not os.environ.get("LD_LIBRARY_PATH", "").startswith("/usr/local/cuda"):
    try:
        import torch as _torch

        _torch_root = Path(_torch.__file__).parent
        _cuda_paths = [str(_torch_root / "lib")]
        _nvidia_dir = _torch_root.parent / "nvidia"
        if _nvidia_dir.exists():
            _cuda_paths.extend(str(p) for p in _nvidia_dir.glob("*/lib"))
        _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            _cuda_paths + ([_existing_ld] if _existing_ld else [])
        )
        del _torch, _torch_root, _cuda_paths, _nvidia_dir, _existing_ld
    except ImportError:
        pass

# ============================================================================
# API Keys (optional - only needed for data downloads)
# ============================================================================

# These are optional and only needed if you're downloading data
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY", "")
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
