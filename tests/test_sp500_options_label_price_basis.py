"""Regression tests for the price bases used by the S&P 500 options labels."""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from case_studies.sp500_options._underlying_returns import reconcile_underlying_log_returns

NOTEBOOK = Path(__file__).parents[1] / "case_studies" / "sp500_options" / "02_labels.py"


def _assignment_nodes(*targets: str) -> list[ast.stmt]:
    tree = ast.parse(NOTEBOOK.read_text())
    selected = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names.intersection(targets):
            selected.append(node)
    return selected


def _run_assignments(targets: tuple[str, ...], namespace: dict[str, object]) -> None:
    module = ast.Module(body=_assignment_nodes(*targets), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, NOTEBOOK, "exec"), namespace)


def test_vrp_rv_respects_splits_security_boundaries_and_segment_scale() -> None:
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 10), eager=True)
    index = np.arange(len(dates))
    first_segment = index < 35
    adjusted = np.where(
        first_segment,
        np.linspace(100.0, 103.4, len(dates)),
        np.linspace(200.0, 203.4, len(dates)),
    )
    split = index >= 20
    raw_close = np.where(first_segment & split, adjusted / 2, adjusted)
    adj_factor = np.where(first_segment & split, 2.0, 1.0)
    bars = pl.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["SPLT"] * len(dates),
            "sec_id": np.where(first_segment, 1, 2),
            "close": raw_close,
            "adj_factor": adj_factor,
        }
    )
    straddles = pl.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["SPLT"] * len(dates),
            "iv_atm": [0.25] * len(dates),
        }
    )

    def run_baseline(prices: pl.DataFrame) -> dict[str, object]:
        namespace = {
            "np": np,
            "pl": pl,
            "reconcile_underlying_log_returns": reconcile_underlying_log_returns,
            "load_sp500_daily_bars": lambda: prices,
            "load_sp500_options_straddles": lambda: straddles,
        }
        _run_assignments(("straddles", "underlying", "underlying_rv", "vrp_baseline"), namespace)
        return namespace

    baseline = run_baseline(bars)
    scaled = run_baseline(
        bars.with_columns(
            pl.when(pl.col("sec_id") == 2)
            .then(pl.col("close") * 11)
            .otherwise(pl.col("close"))
            .alias("close")
        )
    )
    baseline_rv = baseline["underlying_rv"]
    boundaries = baseline_rv.filter(pl.col("identity_boundary"))
    baseline_vrp = baseline["vrp_baseline"].sort("timestamp")
    scaled_vrp = scaled["vrp_baseline"].sort("timestamp")

    assert boundaries.height == 1
    assert boundaries["clean_log_return"].null_count() == 1
    assert baseline_vrp.select(["timestamp", "symbol"]).equals(
        scaled_vrp.select(["timestamp", "symbol"])
    )
    np.testing.assert_allclose(baseline_vrp["rv_21d"], scaled_vrp["rv_21d"], atol=1e-12)
    assert baseline_vrp["rv_21d"].max() < 0.01


def test_expiry_intrinsic_value_keeps_historical_close_basis() -> None:
    expiration = date(2020, 8, 31)
    bars = pl.DataFrame(
        {
            "timestamp": [expiration],
            "symbol": ["SPLT"],
            "close": [52.0],
            "adj_factor": [4.0],
        }
    )
    contract_returns = pl.DataFrame(
        {
            "feature_date": [date(2020, 8, 3)],
            "expiration": [expiration],
            "symbol": ["SPLT"],
            "strike": [50.0],
            "entry_straddle_mid": [10.0],
        }
    )
    namespace = {
        "pl": pl,
        "INSTRUMENT_ID": "straddle_30d_atm",
        "contract_returns": contract_returns,
        "load_sp500_daily_bars": lambda: bars,
    }

    _run_assignments(("underlying_close", "htm_label"), namespace)
    result = namespace["htm_label"]

    assert result["ret_to_expiry"].item() == 0.8
    assert result["dte_calendar"].item() == 28


def test_notebook_rv_rolls_within_full_security_identity() -> None:
    assignments = _assignment_nodes("underlying_rv")

    assert len(assignments) == 1
    expression = ast.dump(assignments[0])
    assert "reconcile_underlying_log_returns" in expression
    assert "clean_log_return" in expression
    assert "symbol" in expression
    assert "sec_id" in expression


def test_baseline_ic_uses_sorted_horizon_aware_hac() -> None:
    tree = ast.parse(NOTEBOOK.read_text())
    hac_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_ic_hac_stats"
    ]

    assert len(hac_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in hac_calls[0].keywords}
    assert isinstance(keywords["label_horizon"], ast.Constant)
    assert keywords["label_horizon"].value == 21

    sorted_ic_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ic_series" for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "sort"
    ]
    assert len(sorted_ic_assignments) == 1
    sort_call = sorted_ic_assignments[0].value
    assert isinstance(sort_call.args[0], ast.Constant)
    assert sort_call.args[0].value == "timestamp"
