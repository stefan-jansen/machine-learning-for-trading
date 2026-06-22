#!/usr/bin/env python3
"""ML4T Installation Verification Script.

Validates that all libraries required by the ML4T Third Edition notebooks
are importable and prints version information. Also checks runtime
requirements (CUDA, repo-root importability, matplotlib styling, Plotly).

Run from repo root:
    python scripts/verify_installation.py
    # or
    uv run python scripts/verify_installation.py
    # or in Docker
    docker compose run --rm ml4t python scripts/verify_installation.py
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CUDA library path fix (must happen before any torch import)
# ---------------------------------------------------------------------------


def _setup_cuda_library_path():
    """Set LD_LIBRARY_PATH to torch's bundled CUDA libs if needed."""
    if os.environ.get("LD_LIBRARY_PATH", "").startswith("/usr/local/cuda"):
        return
    try:
        import torch

        torch_root = Path(torch.__file__).parent
        cuda_paths = [str(torch_root / "lib")]
        nvidia_dir = torch_root.parent / "nvidia"
        if nvidia_dir.exists():
            cuda_paths.extend(str(p) for p in nvidia_dir.glob("*/lib"))
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_paths + ([existing] if existing else []))
    except ImportError:
        pass


_setup_cuda_library_path()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results: list[tuple[str, str, str, str]] = []  # (category, name, status, detail)


def _version(mod: object) -> str:
    """Best-effort version string from a module object."""
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if v is not None:
            return str(v) if isinstance(v, str) else str(v)
    return "installed (version unknown)"


def check_import(category: str, import_name: str, package_label: str | None = None):
    """Try to import *import_name* and record the result."""
    label = package_label or import_name
    try:
        mod = importlib.import_module(import_name)
        ver = _version(mod)
        results.append((category, label, "PASS", ver))
    except Exception as exc:
        results.append((category, label, "FAIL", str(exc)[:120]))


def check_from_import(category: str, module: str, names: list[str], label: str | None = None):
    """Try 'from module import names' and record the result."""
    display = label or f"{module} ({', '.join(names)})"
    try:
        mod = importlib.import_module(module)
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            results.append((category, display, "FAIL", f"Missing attributes: {missing}"))
        else:
            results.append((category, display, "PASS", _version(mod)))
    except Exception as exc:
        results.append((category, display, "FAIL", str(exc)[:120]))


# ---------------------------------------------------------------------------
# Category 1: Core Data Science
# ---------------------------------------------------------------------------


def check_core_data_science():
    cat = "Core Data Science"
    check_import(cat, "numpy", "numpy")
    check_import(cat, "scipy", "scipy")
    check_import(cat, "pandas", "pandas")
    check_import(cat, "polars", "polars")
    check_import(cat, "pyarrow", "pyarrow")
    check_import(cat, "numba", "numba")
    check_import(cat, "sympy", "sympy")


# ---------------------------------------------------------------------------
# Category 2: Visualization
# ---------------------------------------------------------------------------


def check_visualization():
    cat = "Visualization"
    check_import(cat, "matplotlib", "matplotlib")
    check_import(cat, "matplotlib.pyplot", "matplotlib.pyplot")
    check_import(cat, "seaborn", "seaborn")
    check_import(cat, "plotly", "plotly")
    check_import(cat, "plotly.express", "plotly.express")
    check_import(cat, "plotly.graph_objects", "plotly.graph_objects")
    check_import(cat, "plotly.subplots", "plotly.subplots")
    check_import(cat, "kaleido", "kaleido")


# ---------------------------------------------------------------------------
# Category 3: Machine Learning (sklearn, boosting, optimization)
# ---------------------------------------------------------------------------


def check_ml():
    cat = "Machine Learning"
    check_import(cat, "sklearn", "scikit-learn")
    # Key sklearn submodules actually used
    for sub in [
        "sklearn.linear_model",
        "sklearn.ensemble",
        "sklearn.decomposition",
        "sklearn.manifold",
        "sklearn.cluster",
        "sklearn.metrics",
        "sklearn.model_selection",
        "sklearn.preprocessing",
        "sklearn.pipeline",
        "sklearn.impute",
        "sklearn.inspection",
        "sklearn.calibration",
        "sklearn.covariance",
        "sklearn.neural_network",
        "sklearn.mixture",
        "sklearn.feature_extraction.text",
    ]:
        check_import(cat, sub)

    check_import(cat, "lightgbm", "lightgbm")
    check_import(cat, "xgboost", "xgboost")
    check_import(cat, "catboost", "catboost")
    check_import(cat, "optuna", "optuna")
    check_import(cat, "shap", "shap")
    check_import(cat, "tabpfn", "tabpfn")
    check_import(cat, "joblib", "joblib")


# ---------------------------------------------------------------------------
# Category 4: Deep Learning
# ---------------------------------------------------------------------------


def check_deep_learning():
    cat = "Deep Learning"
    check_import(cat, "torch", "pytorch")
    check_import(cat, "torch.nn", "torch.nn")
    check_import(cat, "torch.optim", "torch.optim")
    check_import(cat, "torch.utils.data", "torch.utils.data")
    check_import(cat, "pytorch_lightning", "pytorch-lightning")
    check_import(cat, "torchdiffeq", "torchdiffeq")
    # torchode: optional, Ch5 NB04 has try/except fallback to torchdiffeq
    # Broken with torch 2.10+ due to torchtyping dependency
    check_import(cat, "einops", "einops")
    check_import(cat, "opacus", "opacus")


# ---------------------------------------------------------------------------
# Category 5: NLP / Transformers
# ---------------------------------------------------------------------------


def check_nlp():
    cat = "NLP / Transformers"
    check_import(cat, "transformers", "transformers")
    check_import(cat, "sentence_transformers", "sentence-transformers")
    check_import(cat, "datasets", "datasets (HuggingFace)")
    check_import(cat, "evaluate", "evaluate (HuggingFace)")


# ---------------------------------------------------------------------------
# Category 6: Time Series
# ---------------------------------------------------------------------------


def check_time_series():
    cat = "Time Series"
    check_import(cat, "statsmodels", "statsmodels")
    check_import(cat, "statsmodels.api", "statsmodels.api")
    check_import(cat, "statsmodels.tsa.stattools", "statsmodels.tsa.stattools")
    check_import(cat, "statsmodels.tsa.arima.model", "statsmodels.tsa.arima.model")
    check_import(cat, "arch", "arch")
    check_import(cat, "pmdarima", "pmdarima")
    check_import(cat, "sktime", "sktime")
    check_import(cat, "darts", "darts")
    check_import(cat, "statsforecast", "statsforecast")
    check_import(cat, "hmmlearn", "hmmlearn")
    check_import(cat, "filterpy", "filterpy")
    check_import(cat, "pykalman", "pykalman")
    check_import(cat, "pywt", "PyWavelets")
    check_import(cat, "ruptures", "ruptures")

    # Granite & Chronos (time series foundation models)
    check_import(cat, "tsfm_public", "granite-tsfm")
    check_import(cat, "chronos", "chronos-forecasting")


# ---------------------------------------------------------------------------
# Category 7: Causal Inference
# ---------------------------------------------------------------------------


def check_causal():
    cat = "Causal Inference"
    check_import(cat, "dowhy", "dowhy")
    check_import(cat, "econml", "econml")
    check_import(cat, "causalml", "causalml")
    check_import(cat, "tigramite", "tigramite")
    check_import(cat, "causalimpact", "pycausalimpact")
    check_import(cat, "causallearn", "causal-learn")
    check_import(cat, "linearmodels", "linearmodels")


# ---------------------------------------------------------------------------
# Category 8: Portfolio / Finance
# ---------------------------------------------------------------------------


def check_finance():
    cat = "Portfolio / Finance"
    check_import(cat, "pypfopt", "PyPortfolioOpt")
    check_import(cat, "riskfolio", "riskfolio-lib")
    check_import(cat, "skfolio", "skfolio")
    check_import(cat, "vectorbt", "vectorbt")
    check_import(cat, "exchange_calendars", "exchange-calendars")
    # pfhedge: moved to py312 image (Python 3.12, numpy<2 constraint)


# ---------------------------------------------------------------------------
# Category 9: RL
# ---------------------------------------------------------------------------


def check_rl():
    cat = "Reinforcement Learning"
    check_import(cat, "gymnasium", "gymnasium")
    check_import(cat, "stable_baselines3", "stable-baselines3")


# ---------------------------------------------------------------------------
# Category 10: Data Sources / Providers
# ---------------------------------------------------------------------------


def check_data_sources():
    cat = "Data Sources"
    check_import(cat, "yfinance", "yfinance")
    check_import(cat, "edgar", "edgartools")
    # sec-edgar-api: removed from deps (no notebook imports it, uses edgartools)
    check_import(cat, "iex_parser", "iex-parser")
    check_import(cat, "bs4", "beautifulsoup4")
    check_import(cat, "databento", "databento")
    check_import(cat, "gdown", "gdown")
    check_import(cat, "oandapyV20", "oandapyV20")


# ---------------------------------------------------------------------------
# Category 11: Technical Analysis
# ---------------------------------------------------------------------------


def check_ta():
    cat = "Technical Analysis"
    check_import(cat, "talib", "TA-Lib")
    check_import(cat, "ta", "ta")


# ---------------------------------------------------------------------------
# Category 12: Synthetic Data
# ---------------------------------------------------------------------------


def check_synthetic():
    cat = "Synthetic Data"
    check_import(cat, "be_great", "be-great (REaLTabFormer)")


# ---------------------------------------------------------------------------
# Category 13: Utilities / Infrastructure
# ---------------------------------------------------------------------------


def check_utilities():
    cat = "Utilities"
    check_import(cat, "dotenv", "python-dotenv")
    check_import(cat, "yaml", "PyYAML")
    check_import(cat, "tqdm", "tqdm")
    check_import(cat, "rich", "rich")
    check_import(cat, "requests", "requests")
    check_import(cat, "nest_asyncio", "nest-asyncio")
    check_import(cat, "psutil", "psutil")
    check_import(cat, "openpyxl", "openpyxl")
    check_import(cat, "neo4j", "neo4j")
    check_import(cat, "thefuzz", "thefuzz")
    check_import(cat, "rapidfuzz", "rapidfuzz")
    check_import(cat, "networkx", "networkx")


# ---------------------------------------------------------------------------
# Category 14: Jupyter / Notebook Tooling
# ---------------------------------------------------------------------------


def check_jupyter():
    cat = "Jupyter"
    check_import(cat, "jupyter", "jupyter")
    check_import(cat, "notebook", "notebook")
    check_import(cat, "jupytext", "jupytext")
    check_import(cat, "jupyterlab", "jupyterlab")
    check_import(cat, "IPython", "IPython")
    check_import(cat, "anywidget", "anywidget")
    check_import(cat, "nbconvert", "nbconvert")
    check_import(cat, "nbformat", "nbformat")
    check_import(cat, "papermill", "papermill")


# ---------------------------------------------------------------------------
# Category 15: File Formats / Storage
# ---------------------------------------------------------------------------


def check_storage():
    cat = "Storage / Formats"
    check_import(cat, "tables", "tables (PyTables/HDF5)")
    check_import(cat, "duckdb", "duckdb")
    check_import(cat, "sqlite3", "sqlite3 (stdlib)")
    check_import(cat, "mlflow", "mlflow")


# ---------------------------------------------------------------------------
# Category 16: ML4T Libraries (PyPI)
# ---------------------------------------------------------------------------


def check_ml4t_libraries():
    cat = "ML4T Libraries"
    check_import(cat, "ml4t.data", "ml4t-data")
    check_import(cat, "ml4t.diagnostic", "ml4t-diagnostic")
    check_import(cat, "ml4t.engineer", "ml4t-engineer")
    check_import(cat, "ml4t.backtest", "ml4t-backtest")
    check_import(cat, "ml4t.live", "ml4t-live")

    # Key submodules that notebooks import directly
    sub = "ML4T Sub-modules"
    check_from_import(sub, "ml4t.data", ["DataManager"], "ml4t.data.DataManager")
    check_from_import(sub, "ml4t.diagnostic.signal", ["analyze_signal"], "ml4t.diagnostic.signal")
    check_from_import(
        sub,
        "ml4t.diagnostic.splitters",
        ["CombinatorialCV", "WalkForwardCV"],
        "ml4t.diagnostic.splitters",
    )
    check_from_import(sub, "ml4t.engineer", ["compute_features"], "ml4t.engineer.compute_features")
    check_from_import(
        sub,
        "ml4t.backtest",
        ["BacktestConfig", "DataFeed", "Engine"],
        "ml4t.backtest core classes",
    )


# ---------------------------------------------------------------------------
# Category 17: Repo-Root Imports (utils, data, case_studies)
# ---------------------------------------------------------------------------


def check_repo_imports():
    cat = "Repo Packages"

    # utils
    try:
        import utils  # noqa: F811

        results.append((cat, "utils", "PASS", "importable"))
    except Exception as exc:
        results.append((cat, "utils", "FAIL", str(exc)[:120]))

    check_from_import(
        cat, "utils", ["DATA_DIR", "ML4T_DATA_PATH", "REPO_ROOT"], "utils (core exports)"
    )
    check_from_import(
        cat,
        "utils.paths",
        ["get_chapter_dir", "get_output_dir", "get_case_study_dir"],
        "utils.paths",
    )
    check_from_import(cat, "utils.style", ["COLORS"], "utils.style")
    check_from_import(
        cat,
        "utils.data_quality",
        ["check_ohlc_invariants", "describe_coverage"],
        "utils.data_quality",
    )
    check_from_import(cat, "utils.modeling", ["load_modeling_dataset"], "utils.modeling")
    check_from_import(cat, "utils.cv_splits", ["generate_cv_splits"], "utils.cv_splits")

    # data package
    try:
        import data  # noqa: F811

        results.append((cat, "data", "PASS", "importable"))
    except Exception as exc:
        results.append((cat, "data", "FAIL", str(exc)[:120]))

    check_from_import(
        cat, "data", ["load_etfs", "load_us_equities", "load_cme_futures"], "data (loaders)"
    )

    # case_studies package
    try:
        import case_studies  # noqa: F811

        results.append((cat, "case_studies", "PASS", "importable"))
    except Exception as exc:
        results.append((cat, "case_studies", "FAIL", str(exc)[:120]))


# ---------------------------------------------------------------------------
# Category 18: Runtime Checks
# ---------------------------------------------------------------------------


def check_runtime():
    cat = "Runtime"

    # 1. PYTHONPATH / repo root on sys.path
    repo_root = Path(__file__).resolve().parent.parent
    on_path = any(Path(p).resolve() == repo_root for p in sys.path if p)
    if on_path:
        results.append((cat, "Repo root on sys.path", "PASS", str(repo_root)))
    else:
        results.append((cat, "Repo root on sys.path", "FAIL", f"{repo_root} not found in sys.path"))

    # 2. CUDA / GPU detection
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            results.append(
                (
                    cat,
                    "CUDA (GPU)",
                    "PASS",
                    f"{gpu_name} ({gpu_mem:.1f} GB) - CUDA {torch.version.cuda}",
                )
            )
        else:
            results.append((cat, "CUDA (GPU)", "SKIP", "No GPU detected (CPU-only mode)"))
    except ImportError:
        results.append(
            (cat, "CUDA (GPU)", "SKIP", "PyTorch not importable (see Deep Learning section)")
        )
    except Exception as exc:
        results.append((cat, "CUDA (GPU)", "FAIL", str(exc)[:120]))

    # 3. matplotlibrc detection
    matplotlibrc_path = repo_root / "matplotlibrc"
    if matplotlibrc_path.exists():
        results.append((cat, "matplotlibrc", "PASS", str(matplotlibrc_path)))
    else:
        results.append((cat, "matplotlibrc", "FAIL", "matplotlibrc not found at repo root"))

    # 4. Matplotlib styling active
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend for script mode
        import matplotlib.pyplot as plt

        # Check that our matplotlibrc settings took effect
        spines_top = plt.rcParams.get("axes.spines.top", True)
        if not spines_top:
            results.append(
                (
                    cat,
                    "Matplotlib styling (matplotlibrc)",
                    "PASS",
                    "axes.spines.top=False (ML4T style active)",
                )
            )
        else:
            results.append(
                (
                    cat,
                    "Matplotlib styling (matplotlibrc)",
                    "FAIL",
                    "axes.spines.top=True (ML4T matplotlibrc not loaded - run from repo root)",
                )
            )
    except Exception as exc:
        results.append((cat, "Matplotlib styling (matplotlibrc)", "FAIL", str(exc)[:120]))

    # 5. Plotly renderer available
    try:
        import plotly.io as pio

        renderer = pio.renderers.default or os.environ.get("PLOTLY_RENDERER", "(not set)")
        results.append((cat, "Plotly renderer", "PASS", f"default={renderer}"))
    except Exception as exc:
        results.append((cat, "Plotly renderer", "FAIL", str(exc)[:120]))

    # 6. ML4T Plotly template registered
    try:
        import plotly.io as pio

        # The template is registered when utils.style is imported
        if "ml4t" not in pio.templates:
            try:
                from utils.style import _register_plotly_template

                _register_plotly_template()
            except Exception:
                pass
        if "ml4t" in pio.templates:
            results.append((cat, "Plotly ML4T template", "PASS", "registered"))
        else:
            results.append(
                (
                    cat,
                    "Plotly ML4T template",
                    "FAIL",
                    "template 'ml4t' not in plotly.io.templates (import utils.style to register)",
                )
            )
    except Exception as exc:
        results.append((cat, "Plotly ML4T template", "FAIL", str(exc)[:120]))

    # 7. ML4T_DATA_PATH set and exists
    data_path = os.environ.get("ML4T_DATA_PATH")
    if data_path:
        p = Path(data_path)
        if p.exists():
            results.append((cat, "ML4T_DATA_PATH", "PASS", str(p)))
        else:
            results.append(
                (cat, "ML4T_DATA_PATH", "FAIL", f"Set to {data_path} but directory does not exist")
            )
    else:
        # Check .env fallback
        env_file = repo_root / ".env"
        if env_file.exists():
            results.append(
                (
                    cat,
                    "ML4T_DATA_PATH",
                    "PASS",
                    "Not in env but .env file found (loaded at runtime)",
                )
            )
        else:
            results.append((cat, "ML4T_DATA_PATH", "FAIL", "Not set and no .env file found"))

    # 8. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    results.append((cat, "Python version", "PASS", py_ver))


# ---------------------------------------------------------------------------
# Category 19: Optional / Docker-Profile Extras
# ---------------------------------------------------------------------------


def check_optional():
    cat = "Optional"
    # gensim: separate docker profile (word2vec)
    check_import(cat, "gensim", "gensim (word2vec Docker profile)")
    # signatory: separate env (signatures)
    check_import(cat, "signatory", "signatory (signatures env)")
    # Bayesian
    check_import(cat, "pymc", "pymc (bayesian extra)")
    check_import(cat, "arviz", "arviz (bayesian extra)")
    # Agents
    check_import(cat, "crewai", "crewai (agents extra)")
    check_import(cat, "langgraph", "langgraph (agents extra)")
    # DB benchmark extras
    check_import(cat, "arcticdb", "arcticdb (db-benchmark extra)")
    check_import(cat, "clickhouse_connect", "clickhouse-connect (db-benchmark extra)")
    check_import(cat, "influxdb_client", "influxdb-client (db-benchmark extra)")
    # Live trading
    check_import(cat, "ib_async", "ib_insync/ib_async (live extra)")
    check_import(cat, "alpaca", "alpaca-py (live extra)")
    # Feast / MLOps
    check_import(cat, "feast", "feast (mlops extra)")
    # COT reports
    check_import(cat, "cot_reports", "cot-reports")
    # secedgar
    check_import(cat, "secedgar", "secedgar")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary():
    # Group by category
    categories: dict[str, list[tuple[str, str, str]]] = {}
    for cat, name, status, detail in results:
        categories.setdefault(cat, []).append((name, status, detail))

    total = len(results)
    passed = sum(1 for *_, s, _ in results if s == "PASS")
    failed = sum(1 for *_, s, _ in results if s == "FAIL")
    skipped = sum(1 for *_, s, _ in results if s == "SKIP")

    print()
    print("=" * 78)
    print(f"{BOLD}ML4T Third Edition - Installation Verification{RESET}")
    print("=" * 78)
    print()

    optional_cats = {"Optional"}
    for cat, items in categories.items():
        cat_passed = sum(1 for _, s, _ in items if s == "PASS")
        cat_total = len(items)
        is_optional = cat in optional_cats
        if cat_passed == cat_total:
            cat_indicator = PASS
        elif is_optional:
            cat_indicator = f"{SKIP} (not required)"
        else:
            cat_indicator = FAIL
        print(f"{BOLD}--- {cat} ({cat_passed}/{cat_total}) {cat_indicator}{RESET}")
        for name, status, detail in items:
            if status == "PASS":
                indicator = PASS
            elif status == "SKIP":
                indicator = SKIP
            else:
                indicator = FAIL
            # Truncate detail for display
            detail_short = detail[:60] + "..." if len(detail) > 63 else detail
            print(f"  {indicator}  {name:<45s} {detail_short}")
        print()

    # Separate required vs optional failures
    required_fails = [(c, n, d) for c, n, s, d in results if s == "FAIL" and c not in optional_cats]
    optional_fails = [(c, n, d) for c, n, s, d in results if s == "FAIL" and c in optional_cats]

    # Final summary
    print("=" * 78)
    if not required_fails and not optional_fails:
        print(f"{BOLD}{PASS}  All {total} checks passed.{RESET}")
    elif not required_fails:
        print(
            f"{BOLD}{PASS}  All required checks passed ({passed - len(optional_fails)}/{total - len(optional_fails)}).{RESET}"
        )
        print(f"     {len(optional_fails)} optional packages not installed (Docker-only extras).")
        print("     These are not needed for standard usage — install with:")
        print("       uv sync --extra db-benchmark --extra bayesian --extra agents --extra mlops")
    else:
        print(f"{BOLD}{FAIL}  {len(required_fails)} REQUIRED package(s) failed:{RESET}")
        print()
        for cat, name, detail in required_fails:
            print(f"  {FAIL} {cat} / {name}")
            print(f"       {detail[:100]}")
        if optional_fails:
            print()
            print(f"  Plus {len(optional_fails)} optional packages (not required).")
    print("=" * 78)

    return failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    start = time.time()

    # Ensure repo root is on sys.path (so utils/data/case_studies are importable)
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Run all checks
    check_core_data_science()
    check_visualization()
    check_ml()
    check_deep_learning()
    check_nlp()
    check_time_series()
    check_causal()
    check_finance()
    check_rl()
    check_data_sources()
    check_ta()
    check_synthetic()
    check_utilities()
    check_jupyter()
    check_storage()
    check_ml4t_libraries()
    check_repo_imports()
    check_runtime()
    check_optional()

    elapsed = time.time() - start
    print_summary()
    print(f"\nCompleted in {elapsed:.1f}s")

    # Exit code: 0 if all required (non-Optional) checks pass
    required_failures = sum(
        1 for cat, _, status, _ in results if status == "FAIL" and cat != "Optional"
    )
    sys.exit(1 if required_failures > 0 else 0)


if __name__ == "__main__":
    main()
