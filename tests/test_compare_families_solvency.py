"""``compare_families`` can exclude runs whose equity went negative.

A long-short book can lose more than its capital: the short leg's loss is unbounded
and the engine has no margin call, so equity compounds through zero and later periods
are arithmetic on a negative balance. Those runs still carry a Sharpe, and it is large
often enough to top a family maximum and pull a median. ``max_drawdown`` below -100% is
that condition.
"""

from __future__ import annotations

import sqlite3

from case_studies.utils.backtest_explorer import BacktestExplorer

# (prediction_hash, sharpe, max_drawdown)
ROWS = [
    ("solvent_a", 1.0, -0.2),
    ("solvent_b", 0.5, -0.3),
    ("ruined", 9.0, -3.0),
    ("no_drawdown", 8.0, None),
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
                num_trades REAL
            );
            CREATE TABLE backtest_fold_metrics (
                backtest_hash TEXT,
                fold_id INTEGER,
                sharpe REAL
            );
            """
        )
        for prediction_hash, sharpe, max_drawdown in ROWS:
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, 'gbm', ?, 'fwd_ret_5d')",
                (training_hash, prediction_hash),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', 0)",
                (prediction_hash, training_hash),
            )
            # Equal ic_n_days across rows, so coverage filtering keeps all four and
            # solvency is the only thing separating them.
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, 0.1, 0.1, 0.0, 0.2, 4.0)",
                (prediction_hash,),
            )
            db.execute(
                """
                INSERT INTO backtest_runs VALUES (
                    ?, ?, '{"allocation":{"method":"score_weighted"}}', 'signal'
                )
                """,
                (f"bt_{prediction_hash}", prediction_hash),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, ?, 0.2, 0.1, 1)",
                (f"bt_{prediction_hash}", sharpe, max_drawdown),
            )


def test_default_keeps_every_run(tmp_path) -> None:
    """Off by default, so callers that reported on the full population still do."""
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    families = BacktestExplorer("test", case_dir=case_dir).compare_families()

    row = families.filter(family="gbm")
    assert row["n"].item() == 4
    assert row["sharpe_max"].item() == 9.0


def test_exclude_insolvent_drops_negative_equity_and_unrecorded_drawdown(tmp_path) -> None:
    """The insolvent run holds the family maximum until it is excluded.

    ``no_drawdown`` goes with it: a run with no recorded drawdown cannot be shown to
    have stayed solvent, and dropping it matches ``max_drawdown >= -1.0`` in Polars,
    which the notebooks apply to their figures.
    """
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    families = BacktestExplorer("test", case_dir=case_dir).compare_families(exclude_insolvent=True)

    row = families.filter(family="gbm")
    assert row["n"].item() == 2
    assert row["sharpe_max"].item() == 1.0
    assert row["sharpe_median"].item() == 0.75
