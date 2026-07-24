"""Load internal validation helpers from the sibling ml4t-backtest repository."""

from __future__ import annotations

import importlib
import importlib.util
from functools import cache
from pathlib import Path

from ml4t.backtest import __file__ as ml4t_backtest_init


def resolve_backtest_repo() -> Path:
    """Resolve the sibling ml4t-backtest repository root."""
    package_path = Path(ml4t_backtest_init).resolve()
    for parent in package_path.parents:
        if (parent / "validation" / "benchmark_suite.py").exists():
            return parent
    fallback = Path.home() / "ml4t" / "libraries" / "ml4t-backtest"
    if (fallback / "validation" / "benchmark_suite.py").exists():
        return fallback
    raise FileNotFoundError("Could not locate the ml4t-backtest repository root.")


@cache
def load_validation_helper(module_name: str):
    """Load an internal ml4t-backtest validation helper by module name."""
    qualified_name = f"ml4t.backtest._validation.{module_name}"
    try:
        return importlib.import_module(qualified_name)
    except ModuleNotFoundError as exc:
        if exc.name != "ml4t.backtest._validation":
            raise

    backtest_repo = resolve_backtest_repo()
    module_path = backtest_repo / "src" / "ml4t" / "backtest" / "_validation" / f"{module_name}.py"
    if not module_path.exists():
        raise ModuleNotFoundError(
            f"Could not import {qualified_name}. Install ml4t-backtest "
            "(pip install ml4t-backtest) or keep the source at ~/ml4t/libraries/ml4t-backtest."
        )

    spec = importlib.util.spec_from_file_location(
        f"ml4t_backtest_validation_{module_name}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load validation helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
