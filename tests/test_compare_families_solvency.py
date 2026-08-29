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

# (prediction_hash, family, sharpe, max_drawdown)
ROWS = [
    ("solvent_a", "gbm", 1.0, -0.2),
    ("solvent_b", "gbm", 0.5, -0.3),
    ("ruined", "gbm", 9.0, -3.0),
    # Equity exactly at zero. It cannot earn a later return, so it is ruin and not the
    # edge of solvency - the boundary notebook 12 already applies at the allocation stage.
    ("exactly_zero", "gbm", 7.0, -1.0),
    # No recorded drawdown: cannot be shown to have stayed solvent, so it is counted
    # with the insolvent rather than silently lost from both counts.
    ("no_drawdown", "gbm", 8.0, None),
    # A family with nothing left. It has no Sharpe to report, and must not vanish.
    ("wiped", "tabular_dl", 5.0, -2.0),
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
        for prediction_hash, family, sharpe, max_drawdown in ROWS:
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, ?, ?, 'fwd_ret_5d')",
                (training_hash, family, prediction_hash),
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


def test_default_keeps_every_run_and_its_columns(tmp_path) -> None:
    """Off by default, so callers that reported on the full population still do.

    The column set is part of that: the five other case studies that call this print
    the frame, and an added column would change what their notebooks render.
    """
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    families = BacktestExplorer("test", case_dir=case_dir).compare_families()

    assert families.columns == [
        "family",
        "n",
        "sharpe_median",
        "sharpe_max",
        "sharpe_q75",
        "pct_positive",
    ]
    gbm = families.filter(family="gbm")
    assert gbm["n"].item() == 5
    assert gbm["sharpe_max"].item() == 9.0


def test_exclude_insolvent_reports_the_ruined_rather_than_dropping_them(tmp_path) -> None:
    """The statistics come from the solvent runs; the count of the rest sits beside them.

    Dropping the ruined runs and saying nothing would rank a family by its survivors.
    ``tabular_dl`` is the case that shows it: every run went to zero, so it has no
    Sharpe at all, and it has to remain visible as a family that was wiped out rather
    than disappear from the comparison.
    """
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    families = BacktestExplorer("test", case_dir=case_dir).compare_families(exclude_insolvent=True)

    gbm = families.filter(family="gbm")
    # solvent_a and solvent_b; ruined, exactly_zero and no_drawdown are counted, not lost.
    assert gbm["n"].item() == 2
    assert gbm["insolvent"].item() == 3
    assert gbm["sharpe_max"].item() == 1.0
    assert gbm["sharpe_median"].item() == 0.75
    assert gbm["pct_positive"].item() == 100.0

    wiped = families.filter(family="tabular_dl")
    assert wiped.height == 1
    assert wiped["n"].item() == 0
    assert wiped["insolvent"].item() == 1
    assert wiped["sharpe_median"].item() is None
    assert wiped["pct_positive"].item() is None

    # A family with no solvent run sorts last rather than heading the table on a null.
    assert families["family"].to_list() == ["gbm", "tabular_dl"]


def test_equity_at_exactly_zero_is_ruin(tmp_path) -> None:
    """``max_drawdown`` of exactly -1.0 is the boundary, and it is on the ruined side.

    A drawdown of -100% means the trough reached zero. Nothing is left to earn a return,
    so its Sharpe is no more meaningful than that of a run that went further negative.
    """
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    families = BacktestExplorer("test", case_dir=case_dir).compare_families(exclude_insolvent=True)

    # `exactly_zero` carries Sharpe 7.0, so it would top the family maximum if the
    # boundary were treated as solvent.
    assert families.filter(family="gbm")["sharpe_max"].item() == 1.0
