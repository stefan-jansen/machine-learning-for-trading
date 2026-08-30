"""A cost curve must describe one strategy, and a prediction does not identify one.

Several configurations share a prediction set, and the registry is immutable, so a
superseded generation stays in it under the same prediction hash as the one that replaced
it. On us_firm_characteristics the retired ``walk_forward_v2`` conformal sweep and its
``walk_forward_v3`` replacement are both ``conformal_weighted`` on prediction
``63060fdcc823``, so a read scoped by prediction draws two generations as one line.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from case_studies.utils.backtest_explorer import BacktestExplorer

# (backtest_hash, calibration_version, cost_bps, sharpe) - one prediction, two generations.
ROWS = [
    ("v3_000", "walk_forward_v3", 0.0, 2.94),
    ("v3_050", "walk_forward_v3", 50.0, 2.65),
    ("v2_000", "walk_forward_v2", 0.0, 3.10),
    ("v2_050", "walk_forward_v2", 50.0, 2.80),
]

PREDICTION = "shared_prediction"


def _build_registry(case_dir: Path) -> None:
    run_log = case_dir / "run_log"
    run_log.mkdir(parents=True)
    with sqlite3.connect(run_log / "registry.db") as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_value REAL
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY, ic_mean REAL, ic_mean_daily REAL,
                ic_ci_lo REAL, ic_ci_hi REAL, ic_n_days REAL
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, spec_json TEXT, stage TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY, sharpe REAL, cagr REAL, max_drawdown REAL,
                total_return REAL, volatility REAL, num_trades REAL
            );
            CREATE TABLE backtest_fold_metrics (
                backtest_hash TEXT, fold_id INTEGER, sharpe REAL
            );
            INSERT INTO training_runs VALUES ('train', 'gbm', 'leaves_63_mse', 'fwd_ret_1m');
            INSERT INTO prediction_sets VALUES ('shared_prediction', 'train', 'validation', 0);
            INSERT INTO prediction_metrics
                VALUES ('shared_prediction', 0.1, 0.1, 0.0, 0.2, 4.0);
            """
        )
        for backtest_hash, version, cost_bps, sharpe in ROWS:
            spec = json.dumps(
                {
                    "version": 2,
                    "strategy": {
                        "allocation": {
                            "method": "conformal_weighted",
                            "calibration_version": version,
                        }
                    },
                    "backtest_config": {
                        "commission": {"model": "percentage", "rate": cost_bps / 2 / 10_000},
                        "slippage": {"rate": cost_bps / 2 / 10_000},
                    },
                }
            )
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, ?, 'cost_sensitivity')",
                (backtest_hash, PREDICTION, spec),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.1, 0.2, 0.1, 10)",
                (backtest_hash, sharpe),
            )


def test_scoping_by_prediction_pools_both_generations(tmp_path: Path) -> None:
    """The behaviour that makes the hash scoping necessary, pinned so it stays visible."""
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    curve = BacktestExplorer("test", case_dir=case_dir).cost_sensitivity(prediction_hash=PREDICTION)
    assert curve.height == 4
    # Two rows at each cost level, under one allocator name: a curve that doubles back.
    assert sorted(curve["cost_bps"].to_list()) == [0.0, 0.0, 50.0, 50.0]
    assert curve["allocator"].unique().to_list() == ["conformal_weighted"]


def test_scoping_by_backtest_hash_returns_one_generation(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    curve = BacktestExplorer("test", case_dir=case_dir).cost_sensitivity(
        backtest_hashes=["v3_000", "v3_050"]
    )
    assert sorted(curve["cost_bps"].to_list()) == [0.0, 50.0]
    assert sorted(curve["sharpe"].to_list()) == [2.65, 2.94]


def test_an_empty_selection_is_an_empty_curve(tmp_path: Path) -> None:
    """A sweep that registered nothing must not fall through to every cost row there is."""
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    assert (
        BacktestExplorer("test", case_dir=case_dir).cost_sensitivity(backtest_hashes=[]).is_empty()
    )
