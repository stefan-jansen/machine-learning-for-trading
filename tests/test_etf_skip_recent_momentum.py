"""Regression tests for the ETF momentum definitions."""

import ast
from pathlib import Path

import numpy as np
import polars as pl

from case_studies.utils.feature_engineering import momentum_volatility_block

NOTEBOOK = Path(__file__).parents[1] / "case_studies" / "etfs" / "03_financial_features.py"
MOMENTUM_WINDOWS = [5, 10, 21, 42, 63, 126, 189, 252]


def _price_frame(close: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["ETF"] * len(close),
            "timestamp": pl.date_range(
                pl.date(2018, 1, 1),
                pl.date(2018, 1, 1) + pl.duration(days=len(close) - 1),
                interval="1d",
                eager=True,
            ),
            "close": close,
        }
    )


def _load_momentum_features():
    """Lift ``momentum_features`` out of the notebook without executing the notebook.

    The function reads ``WINDOWS``, ``EPS`` and the shared trailing block from the
    notebook's module scope, so those are injected. ``skip_recent`` is written here
    rather than read from ``config/setup.yaml``: the assertion is that the skipped
    stretch is one month, and reading that value out of the file under test would make
    the test agree with whatever the config happens to say.
    """
    tree = ast.parse(NOTEBOOK.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "momentum_features"
    )
    namespace = {
        "WINDOWS": {
            "momentum": MOMENTUM_WINDOWS,
            "volatility": [21, 63, 126, 252],
            "skip_recent": 21,
        },
        "momentum_volatility_block": momentum_volatility_block,
        "EPS": 1e-12,
        "np": np,
        "pl": pl,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), NOTEBOOK, "exec"), namespace)
    return namespace["momentum_features"]


def test_skip_recent_momentum_uses_price_ratio_ending_one_month_ago() -> None:
    close = np.exp(np.linspace(0.0, 1.0, 253))
    final = _load_momentum_features()(_price_frame(close)).row(-1, named=True)

    assert np.isclose(final["skip_recent_12_1"], close[-22] / close[0] - 1)
    assert np.isclose(final["skip_recent_6_1"], close[-22] / close[-127] - 1)


def test_trailing_sharpe_is_on_the_scale_a_sharpe_ratio_is_read_on() -> None:
    """A one-year window must report an annualized ratio, not one inflated by its length.

    Every case study previously divided the rolling *sum* of log returns by the
    per-period standard deviation and scaled by ``sqrt(252 / window)``. That is a
    mean-over-dispersion ratio multiplied by ``sqrt(window)`` - about 16 at a one-year
    window - which put the shipped ``sharpe_252d`` values past 50.
    """
    rng = np.random.default_rng(0)
    steps = 0.0004 + 0.01 * rng.standard_normal(600)
    out = momentum_volatility_block(
        _price_frame(np.exp(np.cumsum(steps))),
        entity="symbol",
        return_windows=[252],
        volatility_windows=[252],
    )

    window = steps[-252:]
    expected = window.mean() / window.std(ddof=1) * np.sqrt(252)
    assert np.isclose(out["sharpe_252d"].to_numpy()[-1], expected)
    assert abs(expected) < 5.0
