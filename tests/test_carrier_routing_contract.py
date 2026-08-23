"""Regression tests for owner-controlled, fail-closed carrier routing."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils import strategy_analysis
from case_studies.utils.carrier_pins import (
    CARRIER_PINS,
    carrier_config_name,
    filter_to_carrier_config,
    prioritize_carrier_hash,
)
from case_studies.utils.cohort_reporting import cohort_metric_attribution, reportable_pbo


def _pin_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, config_name TEXT);
            CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, training_hash TEXT);
            CREATE TABLE backtest_runs (backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT);
            INSERT INTO training_runs VALUES ('train_us', 'owner_config');
            INSERT INTO prediction_sets VALUES ('pred_us', 'train_us');
            INSERT INTO backtest_runs VALUES ('e676e1989e1f_suffix', 'pred_us');
            """
        )


def test_carrier_pins_are_single_sourced_and_owner_value_passes_through() -> None:
    assert strategy_analysis.CARRIER_PINS is CARRIER_PINS
    assert CARRIER_PINS["us_firm_characteristics"] == "e676e1989e1f"

    repo = Path(__file__).parents[1]
    for relative in (
        "20_strategy_synthesis/holdout.py",
        "20_strategy_synthesis/01_aggregate_synthesis.py",
    ):
        tree = ast.parse((repo / relative).read_text())
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "CARRIER_PINS"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ]
        assert assignments == []


def test_owner_pin_resolves_without_copying_config_name(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.db"
    _pin_db(db_path)
    assert carrier_config_name("us_firm_characteristics", db_path) == "owner_config"
    candidates = pl.DataFrame({"config_name": ["other", "owner_config"], "sharpe": [3.0, 2.9]})
    result = filter_to_carrier_config(
        candidates,
        "us_firm_characteristics",
        db_path=db_path,
    )
    assert result["config_name"].to_list() == ["owner_config"]


def test_carrier_application_fails_closed_after_filters_and_on_missing_schema() -> None:
    with pytest.raises(ValueError, match="absent after candidate filters"):
        prioritize_carrier_hash(
            pl.DataFrame({"backtest_hash": ["not-the-pin"], "ic_mean": [0.1]}),
            "sp500_options",
        )
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="backtest_hash"):
        prioritize_carrier_hash(pl.DataFrame({"ic_mean": [0.1]}), "sp500_options")


def test_carrier_row_is_prioritized_only_after_surviving_filters() -> None:
    pin = CARRIER_PINS["sp500_options"]
    candidates = pl.DataFrame(
        {
            "backtest_hash": ["raw_max", f"{pin}_suffix"],
            "ic_mean": [0.2, 0.1],
        }
    )
    filtered = candidates.filter(pl.col("ic_mean") >= 0.1)
    result = prioritize_carrier_hash(filtered, "sp500_options")
    assert result["backtest_hash"].to_list()[0] == f"{pin}_suffix"


def test_cohort_metrics_are_attributed_to_their_leader() -> None:
    attribution = cohort_metric_attribution({"leader_hash": "lasso_hash"}, "ridge_hash")
    assert attribution == {
        "leader_hash": "lasso_hash",
        "carrier_hash": "ridge_hash",
        "applies_to_carrier": False,
        "subject": "family cohort leader lasso_hash",
    }


def test_pbo_with_two_combinations_is_not_reportable() -> None:
    assert reportable_pbo(0.5, 2) == {
        "value": None,
        "status": "insufficient combinations (2 < 10)",
        "n_combinations": 2,
    }
