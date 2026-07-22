"""Regression tests for owner-controlled, fail-closed carrier routing."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.sp500_options.backtest_contract import (
    ACCEPTED_DEEP_PRODUCERS,
    SP500_OPTIONS_EXECUTION_UNIVERSES,
    assert_accepted_deep_baselines,
    assert_accepted_deep_registry,
    assert_complete_allocation_surface,
    assert_complete_baseline_surface,
    validate_accepted_deep_predictions,
)
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


def test_options_sweep_materializes_both_universes() -> None:
    assert SP500_OPTIONS_EXECUTION_UNIVERSES == ("full", "liquid")


def test_options_consumers_pin_producers_and_allocation_contract() -> None:
    repo = Path(__file__).parents[1]
    for notebook in (
        "13_portfolio_management.py",
        "14_costs.py",
        "15_risk_management.py",
        "16_strategy_analysis.py",
    ):
        source = (repo / "case_studies" / "sp500_options" / notebook).read_text()
        assert "assert_accepted_deep_baselines" in source

    allocation = (repo / "case_studies/sp500_options/13_portfolio_management.py").read_text()
    assert "LIQUID_ONLY = True" in allocation
    assert 'if allocation["method"] != "equal_weight"' in allocation
    assert "Allocation sweep failed" in allocation
    assert 'universe_filter="liquid" if LIQUID_ONLY else "full"' in allocation
    assert "assert_complete_baseline_surface" in allocation
    assert "assert_complete_allocation_surface" in allocation
    assert "BUDGET_SECONDS" not in allocation

    costs = (repo / "case_studies/sp500_options/14_costs.py").read_text()
    assert "load_existing_backtest_hashes" in costs
    assert "backtest_hash_from_parts(pred_hash, spec)" in costs
    assert "serializable_backtest_spec" not in costs
    assert "if backtest_hash not in existing_cost_hashes" in costs

    baseline = (repo / "case_studies/sp500_options/12_backtest.py").read_text()
    assert "print_stage_dsr_summary" not in baseline
    assert "cohort_type='stagelabel'" in baseline
    assert "cohort_metric_attribution(_stage_cohort, _baseline_leader_hash)" in baseline
    assert "reportable_pbo(_family_pbo, _family_pbo_n)" in baseline
    assert "does not match the complete" in baseline

    holdout = (repo / "20_strategy_synthesis/holdout.py").read_text()
    assert "resolve_linear_params(config, fold_data" in holdout


def test_options_surface_contracts_require_exact_cartesian_products(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.db"
    with sqlite3.connect(str(db_path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, label TEXT, config_name TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
            );
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, spec_json TEXT, stage TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY, sharpe REAL, cagr REAL, max_drawdown REAL
            );
            INSERT INTO training_runs VALUES ('train', 'linear', 'ret_to_expiry', 'config');
            INSERT INTO prediction_sets VALUES ('pred', 'train', 'validation');
            """
        )
        rows = (
            (
                "signal_full",
                "pred",
                '{"strategy":{"signal":{"top_k":5,"universe_filter":"full"}}}',
                "signal",
            ),
            (
                "signal_liquid",
                "pred",
                '{"strategy":{"signal":{"top_k":5,"universe_filter":"liquid"}}}',
                "signal",
            ),
            (
                "allocation",
                "pred",
                '{"strategy":{"signal":{"universe_filter":"liquid"},'
                '"allocation":{"top_k":5,"method":"score_weighted"}}}',
                "allocation",
            ),
        )
        db.executemany("INSERT INTO backtest_runs VALUES (?, ?, ?, ?)", rows)
        db.executemany(
            "INSERT INTO backtest_metrics VALUES (?, -0.1, -0.2, -0.3)",
            [(row[0],) for row in rows],
        )

    assert_complete_baseline_surface(db_path, expected_predictions=1, top_ks=(5,))
    assert_complete_allocation_surface(
        db_path,
        prediction_hashes={"pred"},
        top_ks=(5,),
        allocators={"score_weighted"},
    )

    with sqlite3.connect(str(db_path)) as db:
        db.execute("DELETE FROM backtest_metrics WHERE backtest_hash='signal_liquid'")
    with pytest.raises(RuntimeError, match="null metrics"):
        assert_complete_baseline_surface(db_path, expected_predictions=1, top_ks=(5,))


def _accepted_deep_registry(path: Path) -> None:
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT,
                config_name TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY,
                training_hash TEXT,
                split TEXT
            );
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY,
                prediction_hash TEXT,
                stage TEXT
            );
            """
        )
        for config_name, (training_hash, prediction_hash) in ACCEPTED_DEEP_PRODUCERS.items():
            db.execute(
                "INSERT INTO training_runs VALUES (?, 'deep_learning', ?)",
                (training_hash, config_name),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation')",
                (prediction_hash, training_hash),
            )


def test_accepted_deep_hashes_are_exact_and_complete(tmp_path: Path) -> None:
    rows = [
        {
            "family": "deep_learning",
            "config_name": config_name,
            "training_hash": training_hash,
            "prediction_hash": prediction_hash,
        }
        for config_name, (training_hash, prediction_hash) in ACCEPTED_DEEP_PRODUCERS.items()
    ]
    frame = pl.DataFrame(rows)
    assert validate_accepted_deep_predictions(frame).equals(frame)

    wrong = frame.with_columns(
        pl.when(pl.col("config_name") == "patchtst")
        .then(pl.lit("obsolete"))
        .otherwise(pl.col("prediction_hash"))
        .alias("prediction_hash")
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate_accepted_deep_predictions(wrong)

    db_path = tmp_path / "registry.db"
    _accepted_deep_registry(db_path)
    assert_accepted_deep_registry(db_path)


def test_notebook16_fails_before_carrier_when_accepted_baselines_are_absent(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "registry.db"
    _accepted_deep_registry(db_path)
    with pytest.raises(RuntimeError, match="no equal-weight baseline backtests"):
        assert_accepted_deep_baselines(db_path)
