"""Regression coverage for quote-based MBO markouts."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = REPO_ROOT / "03_market_microstructure" / "09_databento_mbo_analysis.py"


def _load_compute_markouts():
    tree = ast.parse(NOTEBOOK_SOURCE.read_text())
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "LATENCY_BARS"
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "compute_markouts")
    ]
    namespace = {"pl": pl}
    exec(compile(ast.Module(body=selected, type_ignores=[]), NOTEBOOK_SOURCE, "exec"), namespace)
    return namespace["compute_markouts"]


def test_midpoint_markout_uses_reconstructed_quote_not_trade_proxy() -> None:
    compute_markouts = _load_compute_markouts()
    bars = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                datetime(2024, 1, 2, 14, 31, tzinfo=UTC),
            ],
            "mid_price": [100.0, 110.0],
            "mid_quote": [200.0, 202.0],
            "best_ask": [201.0, 203.0],
            "best_bid": [199.0, 201.0],
        }
    )

    result = compute_markouts(bars, horizons=[1], latency_bars=0)

    assert result["markout_1"][0] == pytest.approx(0.01)
    assert result["markout_1_adj"][0] == pytest.approx(0.01)
    assert result["markout_1_exec"][0] == pytest.approx(0.0)
