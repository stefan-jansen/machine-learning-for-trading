"""Regression test for ETF skip-recent momentum definitions."""

import ast
from pathlib import Path

import numpy as np
import polars as pl

NOTEBOOK = Path(__file__).parents[1] / "case_studies" / "etfs" / "03_financial_features.py"


def _load_compute_momentum_features():
    tree = ast.parse(NOTEBOOK.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_momentum_features"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "MOMENTUM_HORIZONS": [5, 10, 21, 42, 63, 126, 189, 252],
        "VOLATILITY_HORIZONS": [21, 63, 126, 252],
        "np": np,
        "pl": pl,
    }
    exec(compile(module, NOTEBOOK, "exec"), namespace)
    return namespace["compute_momentum_features"]


def test_skip_recent_momentum_uses_price_ratio_ending_one_month_ago() -> None:
    compute = _load_compute_momentum_features()
    close = np.exp(np.linspace(0.0, 1.0, 253))
    frame = pl.DataFrame(
        {
            "symbol": ["ETF"] * len(close),
            "timestamp": pl.date_range(
                pl.date(2020, 1, 1),
                pl.date(2020, 9, 9),
                interval="1d",
                eager=True,
            ),
            "close": close,
        }
    )

    final = compute(frame).row(-1, named=True)

    assert np.isclose(final["skip_recent_12_1"], close[-22] / close[0] - 1)
    assert np.isclose(final["skip_recent_6_1"], close[-22] / close[-127] - 1)
