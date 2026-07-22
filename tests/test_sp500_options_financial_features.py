"""Regression tests for S&P 500 options financial features."""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from case_studies.sp500_options._underlying_returns import reconcile_underlying_log_returns

NOTEBOOK = Path("case_studies/sp500_options/03_financial_features.py")


def _load_notebook_functions(*wanted_names: str) -> dict[str, object]:
    tree = ast.parse(NOTEBOOK.read_text())
    wanted = set(wanted_names)
    support = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "STATEFUL_WARMUP_REQUIREMENTS"
            for target in node.targets
        )
    ]
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=support + definitions, type_ignores=[])
    namespace = {
        "np": np,
        "pl": pl,
        "reconcile_underlying_log_returns": reconcile_underlying_log_returns,
    }
    exec(compile(module, NOTEBOOK, "exec"), namespace)
    return {name: namespace[name] for name in wanted_names}


def _identity_panel() -> pl.DataFrame:
    start = date(2020, 1, 1)
    closes = [100.0, 101.0, 102.0, 25.75, 25.50, 25.80, 26.0]
    closes += [130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0]
    factors = [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0] + [1.0] * 7
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(14)],
            "symbol": ["AAPL"] * 14,
            "sec_id": [1] * 7 + [2] * 7,
            "close": closes,
            "adj_factor": factors,
            "volume": [1_000_000 + i for i in range(14)],
        }
    )


def test_horizon_features_use_clean_returns_with_full_segment_warmup() -> None:
    functions = _load_notebook_functions(
        "compute_underlying_features", "audit_underlying_feature_windows"
    )
    compute_features = functions["compute_underlying_features"]
    audit_features = functions["audit_underlying_feature_windows"]
    prices = _identity_panel()
    features = compute_features(prices)
    expected = reconcile_underlying_log_returns(prices).with_columns(
        pl.col("clean_log_return")
        .rolling_sum(5, min_samples=5)
        .over(["symbol", "sec_id"])
        .exp()
        .sub(1)
        .alias("expected_ret_5d")
    )
    comparison = features.join(
        expected.select("timestamp", "symbol", "expected_ret_5d"),
        on=["timestamp", "symbol"],
    )
    boundary = features.filter(pl.col("identity_boundary"))
    positions = features.with_columns(
        pl.col("timestamp").cum_count().over(["symbol", "sec_id"]).alias("position")
    )

    assert np.allclose(
        comparison["ret_5d"].drop_nulls(), comparison["expected_ret_5d"].drop_nulls()
    )
    assert boundary.select("ret_1d", "ret_5d", "rv_5d").null_count().row(0) == (1, 1, 1)
    assert positions.filter(pl.col("position") <= 5)["ret_5d"].null_count() == 10
    assert audit_features(prices, features) == {
        "identity_boundaries": 1,
        "warmup_violations": 0,
        "max_identity_error": 0.0,
    }


def test_segment_audit_rejects_a_boundary_return() -> None:
    functions = _load_notebook_functions(
        "compute_underlying_features", "audit_underlying_feature_windows"
    )
    compute_features = functions["compute_underlying_features"]
    audit_features = functions["audit_underlying_feature_windows"]
    prices = _identity_panel()
    features = compute_features(prices).with_columns(
        pl.when(pl.col("identity_boundary"))
        .then(pl.lit(0.10))
        .otherwise(pl.col("ret_1d"))
        .alias("ret_1d")
    )

    with pytest.raises(ValueError, match="Security-segment return audit failed"):
        audit_features(prices, features)


def test_notebook_pins_chronology_horizons_and_actual_vrp_transforms() -> None:
    source = NOTEBOOK.read_text()

    assert "reconcile_underlying_log_returns(prices_df)" in source
    assert ".rolling_sum(w, min_samples=w)" in source
    assert (
        'return pl.DataFrame(rows, schema={"timestamp": pl.Date, "ic": pl.Float64}).sort(' in source
    )
    assert "PRIMARY_HAC_HORIZON = 21" in source
    assert "SECONDARY_HAC_HORIZON = 10" in source
    assert "VRP percentile (252d)" not in source
    assert "colorscale=ml4t_diverging()" in source
    assert "HAC usually reduces |t-statistics|" in source
    assert 'print(f"Saved features: {features_path}")' not in source
    assert "n_features = len(model_feature_cols)" in source


def test_notebook_code_cells_respect_publication_line_limit() -> None:
    source = NOTEBOOK.read_text()
    code_cells = [
        cell for cell in source.split("# %%")[1:] if not cell.lstrip().startswith("[markdown]")
    ]
    oversized = [
        len(cell.rstrip().splitlines())
        for cell in code_cells
        if len(cell.rstrip().splitlines()) > 40
    ]

    assert oversized == []


def _stateful_panel(segment_lengths: tuple[int, ...] = (300, 300)) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2018, 1, 1)
    day = 0
    for sec_id, length in enumerate(segment_lengths, start=1):
        for position in range(length):
            rows.append(
                {
                    "timestamp": start + timedelta(days=day),
                    "symbol": "SAME",
                    "sec_id": sec_id,
                    "instr_mid": 10.0 + sec_id + position / 100,
                    "instr_theta": -0.2,
                    "instr_vega": 0.5,
                    "instr_dte": 30,
                    "instr_delta": 0.01,
                    "instr_pct_of_S": 0.04 + sec_id / 100 + position / 100_000,
                    "iv_atm": 0.20 + sec_id / 100 + position / 10_000,
                    "vrp_21d": 0.03 + sec_id / 100 + position / 20_000,
                }
            )
            day += 1
    return pl.DataFrame(rows)


def _compute_stateful_features(panel: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    functions = _load_notebook_functions(
        "compute_instrument_features",
        "compute_vrp_normalization",
        "compute_dynamics_features",
        "audit_stateful_feature_warmups",
    )
    result = functions["compute_instrument_features"](panel)
    result = functions["compute_vrp_normalization"](result)
    result = functions["compute_dynamics_features"](result)
    census = functions["audit_stateful_feature_warmups"](result)
    return result, census


STATEFUL_FEATURES = [
    "instr_ret_1d",
    "instr_ret_5d",
    "vrp_21d_median_252",
    "vrp_21d_mean_252",
    "vrp_21d_std_252",
    "vrp_zscore_252",
    "iv_atm_z_63",
    "iv_atm_z_252",
    "iv_mom_5d",
    "iv_mom_10d",
    "iv_mom_21d",
    "vrp_mom_5d",
    "vrp_mom_10d",
    "instr_cost_mom_5d",
]


def test_stateful_instrument_families_restart_at_every_security_identity() -> None:
    result, census = _compute_stateful_features(_stateful_panel())
    second = result.filter(pl.col("sec_id") == 2).with_row_index("position")

    assert census["violations"].sum() == 0
    assert second.filter(pl.col("position") == 0).select(STATEFUL_FEATURES).null_count().row(0) == (
        1,
    ) * len(STATEFUL_FEATURES)
    assert second.filter(pl.col("position") < 5)["instr_ret_5d"].null_count() == 5
    assert second.filter(pl.col("position") < 62)["iv_atm_z_63"].null_count() == 62
    assert second.filter(pl.col("position") < 251)["iv_atm_z_252"].null_count() == 251


def test_prior_security_perturbation_cannot_change_new_security_features() -> None:
    panel = _stateful_panel()
    baseline, _ = _compute_stateful_features(panel)
    perturbed, _ = _compute_stateful_features(
        panel.with_columns(
            pl.when(pl.col("sec_id") == 1)
            .then(pl.col("instr_mid") * 100)
            .otherwise(pl.col("instr_mid"))
            .alias("instr_mid"),
            pl.when(pl.col("sec_id") == 1)
            .then(pl.col("instr_pct_of_S") * 50)
            .otherwise(pl.col("instr_pct_of_S"))
            .alias("instr_pct_of_S"),
            pl.when(pl.col("sec_id") == 1)
            .then(pl.col("iv_atm") + 2)
            .otherwise(pl.col("iv_atm"))
            .alias("iv_atm"),
            pl.when(pl.col("sec_id") == 1)
            .then(pl.col("vrp_21d") - 3)
            .otherwise(pl.col("vrp_21d"))
            .alias("vrp_21d"),
        )
    )

    assert (
        baseline.filter(pl.col("sec_id") == 2)
        .select(STATEFUL_FEATURES)
        .equals(perturbed.filter(pl.col("sec_id") == 2).select(STATEFUL_FEATURES))
    )


def test_appending_future_security_cannot_change_existing_feature_prefix() -> None:
    baseline, _ = _compute_stateful_features(_stateful_panel((300, 300)))
    extended, _ = _compute_stateful_features(_stateful_panel((300, 300, 300)))

    assert baseline.select("timestamp", "symbol", "sec_id", *STATEFUL_FEATURES).equals(
        extended.filter(pl.col("sec_id") <= 2).select(
            "timestamp", "symbol", "sec_id", *STATEFUL_FEATURES
        )
    )


def test_security_identity_join_is_unique_complete_and_fail_loud() -> None:
    attach_identity = _load_notebook_functions("attach_security_identity")[
        "attach_security_identity"
    ]
    straddles = pl.DataFrame(
        {
            "timestamp": [date(2020, 1, 1), date(2020, 1, 2)],
            "symbol": ["SAME", "SAME"],
            "instr_mid": [1.0, 2.0],
        }
    )
    identity = pl.DataFrame(
        {
            "timestamp": [date(2020, 1, 1), date(2020, 1, 2)],
            "symbol": ["SAME", "SAME"],
            "sec_id": [1, 2],
        }
    )

    assert attach_identity(straddles, identity)["sec_id"].to_list() == [1, 2]
    with pytest.raises(ValueError, match="not unique"):
        attach_identity(straddles, pl.concat([identity, identity.head(1)]))
    with pytest.raises(ValueError, match="lack contemporaneous"):
        attach_identity(straddles, identity.head(1))
