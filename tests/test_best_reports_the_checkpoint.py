"""The top table names the checkpoint each row was predicted at.

A configuration publishes a prediction set per checkpoint, and those checkpoints
rank separately, so a ten-row table can print one configuration six times at six
Sharpes with nothing on the page saying what separates them. Measured on
`us_firm_characteristics`, `best(stage="signal", top_n=10, label="fwd_ret_1m")`:
six of the ten rows were `gbm/leaves_7_mse` at `top_k` 50, at iterations 200,
250, 300, 350, 400 and 450, Sharpe 2.70 to 2.82.

`checkpoint_kind` and `checkpoint_value` are on `prediction_sets`, which `best`
already joins. This is the same defect `top_k` had one level up: the displayed
columns could not separate rows that genuinely differed.
"""

from __future__ import annotations

import json
import sqlite3

import polars as pl
import pytest

from case_studies.utils.backtest_explorer import (
    BacktestExplorer,
    _drop_rows_a_reader_cannot_tell_apart,
)

# One configuration, one concentration, four checkpoints of the same fit.
CHECKPOINTS = [(200, 2.70), (250, 2.75), (300, 2.82), (350, 2.71)]


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
                checkpoint_value REAL, checkpoint_kind TEXT
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
        # One fit, so one training_hash for every checkpoint below.
        db.execute(
            "INSERT INTO training_runs VALUES ('train', 'gbm', 'leaves_7_mse', 'fwd_ret_1m')"
        )
        spec = {
            "version": 2,
            "strategy": {
                "signal": {"method": "equal_weight_top_k", "top_k": 50},
                "allocation": {"method": "equal_weight"},
            },
            "backtest_config": {},
        }
        for checkpoint, sharpe in CHECKPOINTS:
            prediction_hash = f"pred_{checkpoint}"
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, 'train', 'validation', ?, 'iteration')",
                (prediction_hash, checkpoint),
            )
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, 0.1, 0.1, 0.0, 0.2, 4.0)",
                (prediction_hash,),
            )
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, ?, 'signal')",
                (f"bt_{checkpoint}", prediction_hash, json.dumps(spec)),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.2, 0.2, 0.1, 100)",
                (f"bt_{checkpoint}", sharpe),
            )


@pytest.fixture
def explorer(tmp_path) -> BacktestExplorer:
    case_dir = tmp_path / "cs"
    _build_registry(case_dir)
    return BacktestExplorer("us_firm_characteristics", case_dir=case_dir)


def test_the_top_table_names_the_checkpoint(explorer) -> None:
    top = explorer.best(stage="signal", top_n=10)

    assert top.height == len(CHECKPOINTS)
    # Without the columns this reads as one strategy at four Sharpes.
    assert top["source"].n_unique() == 1
    assert top["top_k"].n_unique() == 1
    assert sorted(top["checkpoint_value"].to_list()) == [c for c, _ in CHECKPOINTS]
    assert set(top["checkpoint_kind"].to_list()) == {"iteration"}


def test_the_checkpoint_sits_beside_the_configuration_it_disambiguates(explorer) -> None:
    columns = explorer.best(stage="signal", top_n=10).columns

    assert columns.index("checkpoint_kind") == columns.index("config_name") + 1
    assert columns.index("checkpoint_value") == columns.index("checkpoint_kind") + 1


def test_the_collapse_does_not_drop_a_row_that_differs_only_by_checkpoint() -> None:
    # `best` drops rows identical in every column a caller receives except
    # `backtest_hash`. Two checkpoints of one fit are two measurements, so the
    # checkpoint has to be inside that comparison: drop it from the subset and
    # the collapse deletes one of them with nothing on the page to say so.
    rows = pl.DataFrame(
        [
            {
                "backtest_hash": "bt_a",
                "config_name": "leaves_7_mse",
                "checkpoint_kind": "iteration",
                "checkpoint_value": 200,
                "sharpe": 2.70,
            },
            {
                "backtest_hash": "bt_b",
                "config_name": "leaves_7_mse",
                "checkpoint_kind": "iteration",
                "checkpoint_value": 250,
                "sharpe": 2.70,
            },
        ]
    )

    kept = _drop_rows_a_reader_cannot_tell_apart(rows)

    assert kept.height == 2
    assert sorted(kept["checkpoint_value"].to_list()) == [200, 250]


def test_an_empty_result_carries_the_checkpoint_columns(explorer) -> None:
    # An empty frame typed from the declared schema, so a caller that selects the
    # columns gets an empty result rather than a SchemaError.
    empty = explorer.best(stage="risk_overlay", top_n=10)

    assert empty.is_empty()
    assert "checkpoint_kind" in empty.columns
    assert "checkpoint_value" in empty.columns
