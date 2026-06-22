#!/usr/bin/env python3
"""
Test all imports for ML4T 3rd Edition.

Verifies that every third-party package used by notebooks is importable.
Groups by chapter so failures map directly to affected content.

Usage:
    python envs/test_all_imports.py              # Test all chapters
    python envs/test_all_imports.py --chapter 15 # Test specific chapter
    python envs/test_all_imports.py --image py312   # Test py312-only packages

Baked into Docker images as: ml4t-test-imports
"""

import argparse
import importlib
import sys
from collections import defaultdict

# Package -> import name mapping (where they differ)
IMPORT_MAP = {
    "beautifulsoup4": "bs4",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "pytorch-lightning": "pytorch_lightning",
    "causal-learn": "causallearn",
    "pywavelets": "pywt",
    "riskfolio-lib": "riskfolio",
    "iex-parser": "iex_parser",
    "chronos-t5": "chronos",
    "granite-tsfm": "tsfm_public",
    "psycopg2-binary": "psycopg2",
    "ib-insync": "ib_async",
    "python-dotenv": "dotenv",
    "nest-asyncio": "nest_asyncio",
    "stable-baselines3": "stable_baselines3",
    "sentence-transformers": "sentence_transformers",
    "exchange-calendars": "exchange_calendars",
    "be-great": "be_great",
}

# =========================================================================
# ML4T image (Python 3.14) — packages used by chapter notebooks
# =========================================================================
# Derived from actual imports in each chapter's .py files.
# Only includes third-party packages (not stdlib, not utils/, not data/).

CHAPTER_PACKAGES = {
    1: ["numpy", "polars", "pandas", "matplotlib", "seaborn", "sklearn", "scipy"],
    2: ["polars", "plotly", "ml4t.data", "numpy", "pandas", "scipy", "pyarrow"],
    3: [
        "polars",
        "plotly",
        "ml4t.data",
        "numba",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "seaborn",
        "tqdm",
    ],
    4: ["polars", "plotly", "bs4", "edgar", "ml4t.data", "numpy"],
    5: [
        "polars",
        "plotly",
        "torch",
        "arch",
        "einops",
        "hmmlearn",
        "numpy",
        "opacus",
        "sklearn",
        "statsmodels",
        "torchdiffeq",
        "tqdm",
        "ml4t.data",
    ],
    6: [
        "polars",
        "plotly",
        "exchange_calendars",
        "ml4t.data",
        "numpy",
        "pandas",
        "sklearn",
        "yaml",
    ],
    7: ["polars", "plotly", "ml4t.data", "numpy", "pandas", "scipy", "sklearn"],
    8: ["polars", "plotly", "ml4t.data", "numpy", "scipy", "seaborn"],
    9: [
        "polars",
        "plotly",
        "arch",
        "arviz",
        "filterpy",
        "hmmlearn",
        "lightgbm",
        "ml4t.data",
        "numpy",
        "pandas",
        "pymc",
        "ruptures",
        "scipy",
        "sklearn",
        "statsforecast",
        "statsmodels",
    ],
    10: [
        "polars",
        "plotly",
        "evaluate",
        "numpy",
        "scipy",
        "seaborn",
        "sentence_transformers",
        "sklearn",
        "torch",
        "transformers",
    ],
    11: [
        "polars",
        "plotly",
        "joblib",
        "ml4t.data",
        "numpy",
        "optuna",
        "pandas",
        "scipy",
        "shap",
        "sklearn",
        "statsmodels",
    ],
    12: [
        "polars",
        "plotly",
        "catboost",
        "lightgbm",
        "numpy",
        "optuna",
        "scipy",
        "shap",
        "sklearn",
        "torch",
        "transformers",
        "xgboost",
    ],
    13: ["polars", "plotly", "numpy", "pandas", "scipy", "seaborn", "sklearn", "torch"],
    14: ["polars", "plotly", "numpy", "pandas", "scipy", "seaborn", "shap", "sklearn", "torch"],
    15: [
        "polars",
        "plotly",
        "lightgbm",
        "ml4t.data",
        "networkx",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
    ],
    16: [
        "polars",
        "plotly",
        "ml4t.data",
        "ml4t.backtest",
        "numpy",
        "pandas",
        "scipy",
        "seaborn",
        "yaml",
    ],
    17: [
        "polars",
        "plotly",
        "cvxpy",
        "ml4t.data",
        "numpy",
        "pandas",
        "pypfopt",
        "riskfolio",
        "scipy",
        "skfolio",
        "sklearn",
        "sympy",
        "torch",
    ],
    18: ["polars", "plotly", "ml4t.data", "numpy", "pandas", "seaborn", "sklearn", "yaml"],
    19: [
        "polars",
        "plotly",
        "arch",
        "lightgbm",
        "ml4t.data",
        "numpy",
        "pandas",
        "scipy",
        "shap",
        "sklearn",
        "statsmodels",
        "torch",
    ],
    20: ["polars", "numpy", "seaborn", "yaml"],
    21: [
        "polars",
        "plotly",
        "gymnasium",
        "numpy",
        "pandas",
        "sklearn",
        "stable_baselines3",
        "torch",
    ],
    22: ["polars", "plotly", "numpy", "torch", "transformers"],
    23: ["polars", "plotly", "neo4j", "networkx", "numpy", "scipy", "torch"],
    24: ["polars", "plotly", "numpy", "torch"],
    25: ["polars", "plotly", "ib_async", "ml4t.data", "nest_asyncio", "numpy", "pandas"],
    26: ["polars", "plotly", "feast", "ml4t.data", "numpy", "pandas", "scipy", "seaborn", "yaml"],
}

# =========================================================================
# Py312 image (Python 3.12) — packages NOT in the ml4t image
# =========================================================================
PY312_PACKAGES = {
    5: ["signatory"],
    9: ["esig"],
    10: ["gensim"],
    15: ["causalimpact"],  # tfcausalimpact pip dist; module imports as `causalimpact`
    21: ["pfhedge"],
}

# All py312-only package names (for skip detection in ml4t image)
PY312_ONLY = {"esig", "gensim", "signatory", "pfhedge", "causalimpact"}

# =========================================================================
# Benchmark image — database clients
# =========================================================================
BENCHMARK_PACKAGES = {
    2: ["duckdb", "tables", "clickhouse_connect", "psycopg2", "influxdb_client"],
}

# =========================================================================
# Case study infrastructure — utils modules used by pipeline notebooks
# =========================================================================
REPO_UTILS = [
    "utils.config",
    "utils.paths",
    "utils.style",
    "utils.modeling",
]

CASE_STUDY_UTILS = [
    "case_studies.utils.analytics",
    "case_studies.utils.backtest_explorer",
    "case_studies.utils.backtest_loaders",
    "case_studies.utils.backtest_presets",
    "case_studies.utils.backtest_runner",
    "case_studies.utils.causal",
    "case_studies.utils.deep_learning",
    "case_studies.utils.factor_attribution",
    "case_studies.utils.gbm",
    "case_studies.utils.latent_factors",
    "case_studies.utils.model_analysis",
    "case_studies.utils.notebook_contracts",
    "case_studies.utils.sequence_dataset",
    "case_studies.utils.signals",
    "case_studies.utils.strategy_analysis",
    "case_studies.utils.sweep_config",
    "case_studies.utils.tabular_dl",
]

# ML4T library imports
ML4T_LIBRARIES = {
    "ml4t.data": "Data acquisition and loaders",
    "ml4t.diagnostic": "Statistical validation and metrics",
    "ml4t.engineer": "Feature engineering",
    "ml4t.backtest": "Backtesting engine",
    "ml4t.live": "Live trading",
}

# Known high-risk packages (historically problematic installs)
HIGH_RISK = {
    "pymc",
    "arviz",
    "darts",
    "sktime",
    "vectorbt",
    "riskfolio",
    "skfolio",
    "chronos",
    "tabpfn",
    "cvxpy",
    "gymnasium",
    "stable_baselines3",
    "feast",
    "sentence_transformers",
}


def get_import_name(package: str) -> str:
    """Convert package name to import name."""
    return IMPORT_MAP.get(package, package.replace("-", "_"))


def test_import(package: str) -> tuple[bool, str]:
    """Try to import a package, return (success, error_message)."""
    import_name = get_import_name(package)
    try:
        importlib.import_module(import_name)
        return True, ""
    except ImportError as e:
        return False, f"ImportError: {e}"
    except (FileNotFoundError, NotADirectoryError):
        # Module found but needs data directory — counts as OK
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_chapter(chapter: int, packages_map: dict) -> dict:
    """Test all imports for a chapter."""
    packages = packages_map.get(chapter, [])
    results = {"passed": [], "failed": [], "py312": [], "high_risk_failed": []}

    for pkg in packages:
        if pkg in PY312_ONLY:
            success, error = test_import(pkg)
            if success:
                results["passed"].append(pkg)
            else:
                results["py312"].append(pkg)
            continue

        success, error = test_import(pkg)
        if success:
            results["passed"].append(pkg)
        else:
            results["failed"].append((pkg, error))
            if pkg in HIGH_RISK:
                results["high_risk_failed"].append(pkg)

    return results


def test_modules(modules: list[str], label: str) -> tuple[int, int]:
    """Test a list of module imports. Returns (ok, fail) counts."""
    print(f"\n{'-' * 60}")
    print(label)
    print("-" * 60)
    ok = fail = 0
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
            ok += 1
        except (FileNotFoundError, NotADirectoryError) as e:
            msg = str(e).split("\n")[0][:50]
            print(f"  ⚠️  {module}: {msg} (OK — needs data)")
            ok += 1
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            fail += 1
    return ok, fail


def _run_scan_mode(image: str, verbose: bool) -> int:
    """Run the AST-based scanner and import every package classified for ``image``.

    Unlike the hand-maintained CHAPTER_PACKAGES path, this mode re-discovers
    the dependency set on each invocation, so new imports added to any
    chapter or case study are picked up automatically. Recommended as
    readers' self-test inside a Docker container.
    """
    from envs.scan_imports import classify, scan_repo, try_import

    imports = scan_repo()
    groups = classify(imports)
    expected = sorted(groups.get(image, set()))

    print("=" * 60)
    print(f"ML4T Import Scan — {image} image (auto-discovered)")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Scanned {len(imports)} external imports across the repo")
    print(f"{len(expected)} classified for the {image!r} image")
    print()

    failures = []
    for pkg in expected:
        ok, err = try_import(pkg)
        if ok:
            if verbose:
                print(f"  ✅ {pkg}")
        else:
            print(f"  ❌ {pkg}: {err[:100]}")
            failures.append((pkg, err))

    print()
    if failures:
        print(f"❌ {len(failures)} of {len(expected)} expected imports failed in {image!r}")
        return 1
    print(f"✅ All {len(expected)} imports succeeded in {image!r}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test ML4T imports")
    parser.add_argument("--chapter", type=int, help="Test specific chapter only")
    parser.add_argument(
        "--image",
        choices=["ml4t", "py312", "benchmark", "rapids", "optional"],
        default="ml4t",
        help="Which Docker image packages to test",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help=(
            "Use AST-based scanner (envs/scan_imports.py) to auto-discover "
            "imports from source code instead of the hand-maintained CHAPTER_PACKAGES. "
            "Recommended for reader self-test — catches new deps automatically."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all results")
    args = parser.parse_args()

    if args.scan:
        # Ensure the project root is on sys.path so `from envs.scan_imports` works.
        # This is robust whether invoked as `python envs/test_all_imports.py`
        # or via the Docker entrypoint wrapper.
        from pathlib import Path

        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        return _run_scan_mode(args.image, args.verbose)

    # Select package map based on image
    if args.image == "py312":
        packages_map = PY312_PACKAGES
    elif args.image == "benchmark":
        packages_map = BENCHMARK_PACKAGES
    elif args.image == "ml4t":
        packages_map = CHAPTER_PACKAGES
    else:
        parser.error(
            f"--image {args.image} has no hand-maintained package map; "
            f"use --scan to auto-discover imports for that image."
        )

    print("=" * 60)
    print(f"ML4T Import Test — {args.image} image")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print()

    chapters = [args.chapter] if args.chapter else sorted(packages_map.keys())

    all_passed = set()
    all_failed = defaultdict(list)
    all_py312 = set()
    chapter_status = {}

    for ch in chapters:
        results = test_chapter(ch, packages_map)
        all_passed.update(results["passed"])
        all_py312.update(results["py312"])
        for pkg, err in results["failed"]:
            all_failed[pkg].append((ch, err))

        total = len(results["passed"]) + len(results["failed"]) + len(results["py312"])
        passed = len(results["passed"]) + len(results["py312"])
        status = "✅" if not results["failed"] else "⚠️" if results["passed"] else "❌"
        chapter_status[ch] = (status, passed, total, results["high_risk_failed"])

        if args.verbose or results["failed"]:
            print(f"\nChapter {ch}: {status} ({passed}/{total} packages)")
            if results["py312"]:
                for pkg in results["py312"]:
                    print(f"  ℹ️  {pkg}: py312 image only (ml4t-py312)")
            if results["failed"]:
                for pkg, err in results["failed"]:
                    risk = " [HIGH RISK]" if pkg in HIGH_RISK else ""
                    print(f"  ❌ {pkg}{risk}: {err[:80]}")

    # ML4T libraries (always test these)
    ml4t_ok, ml4t_fail = test_modules(list(ML4T_LIBRARIES.keys()), "ML4T Libraries")

    # Infrastructure modules (only for ml4t image)
    utils_ok = utils_fail = 0
    if args.image == "ml4t":
        u_ok, u_fail = test_modules(REPO_UTILS, "Repository Utils")
        cs_ok, cs_fail = test_modules(CASE_STUDY_UTILS, "Case Study Utils")
        utils_ok = u_ok + cs_ok
        utils_fail = u_fail + cs_fail

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nChapter Status:")
    for ch, (status, passed, total, hr_failed) in sorted(chapter_status.items()):
        hr_note = f" (high-risk: {', '.join(hr_failed)})" if hr_failed else ""
        print(f"  Ch{ch:02d}: {status} {passed}/{total}{hr_note}")

    n_tested = len(all_passed) + len(all_failed) + len(all_py312)
    n_ok = len(all_passed) + len(all_py312)
    print(f"\nPackages: {n_ok}/{n_tested} OK")
    if all_py312:
        print(f"  Py312 image: {', '.join(sorted(all_py312))} (use ml4t-py312)")
    print(f"ML4T libraries: {ml4t_ok}/{ml4t_ok + ml4t_fail}")
    if args.image == "ml4t":
        print(f"Infrastructure modules: {utils_ok}/{utils_ok + utils_fail}")

    if all_failed:
        print("\nFailed packages:")
        for pkg in sorted(all_failed.keys()):
            chapters_affected = [ch for ch, _ in all_failed[pkg]]
            risk = " [HIGH RISK]" if pkg in HIGH_RISK else ""
            print(f"  {pkg}{risk}: Ch {', '.join(map(str, chapters_affected))}")

        high_risk_failed = [p for p in all_failed if p in HIGH_RISK]
        if high_risk_failed:
            print(f"\n⚠️  High-risk failures: {', '.join(sorted(high_risk_failed))}")
    else:
        print("\n✅ All imports successful!")

    return 1 if all_failed else 0


if __name__ == "__main__":
    sys.exit(main())
