"""Two selections that tie must be broken the same way, by both of the paths that rank.

Exact Sharpe ties are produced by construction in this data rather than by coincidence.
Two backtests of one model that differ only in a specification field the returns do not
depend on - an overlay that never triggers, a re-run that reproduces its inputs, an
equal-weight allocation over a top-k signal that was already equal-weight - book identical
return series, so their Sharpes agree to the last bit. Measured across the nine production
registries 2026-09-07: 17 of 253 cohorts have a tied leader, and in 12 of them the tie
decides which hash is reported.

Two independent paths pick a rank-1 from that data and neither used to specify the tie:

* `BacktestExplorer.best` ordered by `bm.sharpe DESC` alone, so SQLite returned whichever
  tied row its scan reached first;
* `uncertainty._cohort_bundle` took `np.nanargmax`, which returns the FIRST maximum - and
  "first" is the order the caller's mapping was built in, which is the cohort listing's
  SQLite row order.

Both are properties of one database file rather than of the results in it, so the same
registry content answers differently before and after a rebuild, and two readers with the
same rows and different insert histories disagree. The tie-break is the backtest hash,
which is the only key that belongs to the result.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.uncertainty import compute_cohort_metrics

# Two series that differ, and two that are identical to the last bit. The identical pair is
# the tie; the third series is there so the cohort has a leader to be wrong about.
_PERIODS = 60
_RNG = np.random.default_rng(0)
_STRONG = list(_RNG.normal(0.004, 0.01, _PERIODS))
_WEAK = list(_RNG.normal(0.000, 0.01, _PERIODS))


def _frame(values: list[float]) -> pl.DataFrame:
    start = dt.datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + dt.timedelta(days=i) for i in range(len(values))],
            "ret": values,
        }
    )


def test_the_cohort_leader_does_not_depend_on_the_order_the_mapping_was_built_in() -> None:
    """The same rows, inserted two ways, must name the same leader.

    `aaa...` and `zzz...` carry the same series, so they tie exactly and nothing about the
    data can separate them. What used to separate them was `np.nanargmax` returning the
    first maximum in the mapping's own order, which is the order a cohort listing came back
    from SQLite - so the answer moved when the registry was rebuilt and differed between
    two readers holding the same results.
    """
    tied_a, tied_z = "aaa000000000", "zzz000000000"
    forward = {
        tied_a: _frame(_STRONG),
        tied_z: _frame(_STRONG),
        "mmm000000000": _frame(_WEAK),
    }
    reverse = {
        tied_z: _frame(_STRONG),
        tied_a: _frame(_STRONG),
        "mmm000000000": _frame(_WEAK),
    }

    leader_forward = compute_cohort_metrics(forward, periods_per_year=252)["leader_hash"]
    leader_reverse = compute_cohort_metrics(reverse, periods_per_year=252)["leader_hash"]

    assert leader_forward == leader_reverse, (
        f"the same cohort named {leader_forward} built one way and {leader_reverse} built "
        "the other; the leader is a property of the results, not of how the mapping was "
        "assembled"
    )
    assert leader_forward == tied_a, (
        f"the leader is {leader_forward}; among rows that tie exactly it has to be the "
        "lowest hash, which is the only key that is a property of the result"
    )


def test_the_explorer_returns_tied_rows_in_hash_order(tmp_path: Path) -> None:
    """`ORDER BY bm.sharpe DESC` alone leaves SQLite to choose, and it chooses by storage.

    The higher hash is inserted first here, so a scan in storage order reaches it first and
    a query with no second key returns it as rank-1. Nothing in the registry says it should
    win; it was written first.
    """
    case_dir = tmp_path / "fixture_case_study"
    (case_dir / "run_log").mkdir(parents=True)
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as db:
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
            INSERT INTO training_runs VALUES ('train', 'gbm', 'cfg', 'fwd_ret_5d');
            INSERT INTO prediction_sets VALUES ('pred', 'train', 'validation', 0);
            INSERT INTO prediction_metrics VALUES ('pred', 0.02, 0.02, 0.0, 0.04, 250);
            INSERT INTO fold_metrics VALUES ('pred', 0.02);
            """
        )
        for backtest_hash in ("zzz_written_first", "aaa_written_second"):
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, 'pred', '{\"strategy\": {}}', 'signal')",
                (backtest_hash,),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, 1.25, 0.1, -0.2, 0.3, 0.1, 100)",
                (backtest_hash,),
            )

    ranked = BacktestExplorer("fixture_case_study", case_dir=case_dir).best(top_n=10)

    assert ranked["backtest_hash"].to_list()[0] == "aaa_written_second", (
        f"the explorer ranked {ranked['backtest_hash'].to_list()} first-to-last; the two "
        "carry the same Sharpe, so the order has to come from the hash rather than from "
        "which row was written first"
    )


@pytest.mark.parametrize(
    ("scores", "names", "expected"),
    [
        ([1.0, 3.0, 2.0], ["b", "a", "c"], "a"),
        ([3.0, 1.0, 3.0], ["z", "m", "a"], "a"),
        ([np.nan, 2.0, 2.0], ["a", "z", "m"], "m"),
        ([2.0, np.nan], ["z", "a"], "z"),
    ],
)
def test_the_tie_break_takes_the_lowest_hash_among_the_maxima(
    scores: list[float], names: list[str], expected: str
) -> None:
    """Highest score first, hash second, and NaN never enters the tied set.

    The third case is the one worth stating: `a` sorts lowest and is NaN, so it is not a
    maximum and cannot win by being first alphabetically. A NaN never compares equal to
    anything, itself included, which is what keeps it out.

    Imported inside the test rather than at module scope, so that a build without the
    helper fails the two cases above on their behaviour instead of failing the whole file
    at collection.
    """
    from case_studies.utils.uncertainty import _leader_index

    assert names[_leader_index(np.asarray(scores), names)] == expected
