"""The top table names the concentration each row was run at.

ml4t/agent-workspace#910. Every entry scheme in a baseline sweep uses the same
method, `equal_weight_top_k`, and varies only `top_k`. `best()` returned
`signal_method` and not `top_k`, so the ten-row table every case study's backtest
notebook prints read as one strategy repeated at different Sharpes - it hid
exactly what the sweep was run to measure. `concentration_curve()` looked like
the answer and was not: it hardcoded the allocation stage, so at the signal
stage, where the baseline sweep lives, it returned an empty frame.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from case_studies.utils.backtest_explorer import BacktestExplorer

# (prediction_hash, stage, top_k, sharpe)
ROWS = [
    ("pred_a", "signal", 5, 1.80),
    ("pred_a", "signal", 10, 1.68),
    ("pred_a", "signal", 20, 1.64),
    ("pred_b", "allocation", 5, 1.20),
]


def _build_registry(case_dir) -> None:
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
            """
        )
        for prediction_hash in {row[0] for row in ROWS}:
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, 'gbm', 'leaves_7_mae', 'fwd_ret_5d')",
                (training_hash,),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', 0)",
                (prediction_hash, training_hash),
            )
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, 0.1, 0.1, 0.0, 0.2, 4.0)",
                (prediction_hash,),
            )
        for prediction_hash, stage, top_k, sharpe in ROWS:
            backtest_hash = f"bt_{prediction_hash}_{stage}_{top_k}"
            spec = {
                "version": 2,
                "strategy": {
                    "signal": {"method": "equal_weight_top_k", "top_k": top_k},
                    "allocation": {"method": "equal_weight"},
                },
                "backtest_config": {},
            }
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, ?, ?)",
                (backtest_hash, prediction_hash, json.dumps(spec), stage),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.2, 0.2, 0.1, 100)",
                (backtest_hash, sharpe),
            )


@pytest.fixture
def explorer(tmp_path) -> BacktestExplorer:
    case_dir = tmp_path / "cs"
    _build_registry(case_dir)
    return BacktestExplorer("us_firm_characteristics", case_dir=case_dir)


def test_the_top_table_names_the_concentration(explorer) -> None:
    """The defect: three rows of one method, and nothing saying which is which."""
    best = explorer.best(stage="signal")
    assert best["top_k"].to_list() == [5, 10, 20]
    assert best["signal_method"].unique().to_list() == ["equal_weight_top_k"]


def test_the_column_sits_beside_the_method_it_disambiguates(explorer) -> None:
    columns = explorer.best(stage="signal").columns
    assert columns.index("top_k") == columns.index("signal_method") + 1


def test_the_concentration_curve_reads_the_stage_the_sweep_ran_at(explorer) -> None:
    curve = explorer.concentration_curve("pred_a", stage="signal")
    assert curve["top_k"].to_list() == [5, 10, 20]
    assert curve["sharpe"].to_list() == [1.80, 1.68, 1.64]


def test_the_default_stage_is_unchanged(explorer) -> None:
    """The control: a caller reading the allocation stage still reads it."""
    curve = explorer.concentration_curve("pred_b")
    assert curve["top_k"].to_list() == [5]


def test_asking_the_wrong_stage_says_where_the_rows_are(explorer) -> None:
    """An empty frame cannot say whether the sweep was never run or run elsewhere."""
    with pytest.raises(ValueError) as excinfo:
        explorer.concentration_curve("pred_a")
    message = str(excinfo.value)
    assert "signal: 3" in message
    assert "stage='signal'" in message


def test_a_prediction_with_no_backtests_returns_empty(explorer) -> None:
    """Nothing to point at, so nothing to raise about."""
    assert explorer.concentration_curve("pred_missing", stage="signal").is_empty()
