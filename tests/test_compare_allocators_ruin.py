"""``compare_allocators`` counts the bankrupt runs instead of dropping them.

ml4t/agent-workspace#920, item 2. Measured on `us_firm_characteristics`: five of
twenty-four allocation runs had crossed zero equity, and the table averaged them in
with the rest, reporting `avg_max_dd` of -5.74 and -8.15 where every solvent member
was between -0.10 and -0.41. The engine now stops those paths and registers no
Sharpe for them, so the fix has to keep them visible rather than let the null drop
them out of the population.
"""

from __future__ import annotations

import json
import sqlite3

from case_studies.utils.backtest_explorer import BacktestExplorer

# (prediction_hash, allocator, sharpe, max_drawdown, ruin)
ROWS = [
    ("mvo_a", "mean_variance", 1.0, -0.20, 0.0),
    ("mvo_b", "mean_variance", 0.5, -0.30, 0.0),
    # Stopped at ruin by the engine: no Sharpe to rank, drawdown floored at -1.0.
    ("mvo_ruined", "mean_variance", None, -1.00, 1.0),
    # Every run of this allocator went bankrupt. It must still appear.
    ("sw_ruined_1", "score_weighted", None, -1.00, 1.0),
    ("sw_ruined_2", "score_weighted", None, -1.00, 1.0),
]


def _build_registry(case_dir) -> None:
    run_log = case_dir / "run_log"
    run_log.mkdir(parents=True)
    with sqlite3.connect(run_log / "registry.db") as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT,
                config_name TEXT,
                label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY,
                training_hash TEXT,
                split TEXT,
                checkpoint_value REAL
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY,
                ic_mean REAL,
                ic_mean_daily REAL,
                ic_ci_lo REAL,
                ic_ci_hi REAL,
                ic_n_days REAL
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY,
                prediction_hash TEXT,
                spec_json TEXT,
                stage TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY,
                sharpe REAL,
                cagr REAL,
                max_drawdown REAL,
                total_return REAL,
                volatility REAL,
                num_trades REAL,
                ruin REAL
            );
            CREATE TABLE backtest_fold_metrics (
                backtest_hash TEXT,
                fold_id INTEGER,
                sharpe REAL
            );
            """
        )
        for prediction_hash, allocator, sharpe, max_drawdown, ruin in ROWS:
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, 'gbm', ?, 'fwd_ret_5d')",
                (training_hash, prediction_hash),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', 0)",
                (prediction_hash, training_hash),
            )
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, 0.1, 0.1, 0.0, 0.2, 4.0)",
                (prediction_hash,),
            )
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, ?, 'allocation')",
                (
                    f"bt_{prediction_hash}",
                    prediction_hash,
                    json.dumps({"allocation": {"method": allocator}}),
                ),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, ?, 0.2, 0.1, 1, ?)",
                (f"bt_{prediction_hash}", sharpe, max_drawdown, ruin),
            )


def _compare(tmp_path):
    case_dir = tmp_path / "cs"
    _build_registry(case_dir)
    explorer = BacktestExplorer("us_firm_characteristics", case_dir=case_dir)
    return {row["allocator"]: row for row in explorer.compare_allocators().iter_rows(named=True)}


def test_an_allocator_whose_every_run_went_bankrupt_stays_on_the_table(tmp_path) -> None:
    rows = _compare(tmp_path)
    assert set(rows) == {"mean_variance", "score_weighted"}
    assert rows["score_weighted"]["n"] == 0
    assert rows["score_weighted"]["ruined"] == 2
    assert rows["score_weighted"]["avg_sharpe"] is None


def test_the_reported_statistics_describe_the_solvent_runs(tmp_path) -> None:
    rows = _compare(tmp_path)
    mvo = rows["mean_variance"]
    assert mvo["n"] == 2
    assert mvo["ruined"] == 1
    assert mvo["avg_sharpe"] == 0.75
    # -0.20 and -0.30, not dragged toward -1.0 by the run that ended.
    assert mvo["avg_max_dd"] == -0.25


def test_a_registry_without_the_ruin_column_still_reads_the_drawdown(tmp_path) -> None:
    """An older registry carries the evidence only in `max_drawdown`."""
    case_dir = tmp_path / "cs"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("ALTER TABLE backtest_metrics DROP COLUMN ruin")
    explorer = BacktestExplorer("us_firm_characteristics", case_dir=case_dir)
    rows = {r["allocator"]: r for r in explorer.compare_allocators().iter_rows(named=True)}
    assert rows["mean_variance"]["ruined"] == 1
    assert rows["score_weighted"]["ruined"] == 2
