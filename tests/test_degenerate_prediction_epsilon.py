"""A constant fold whose IC reached the registry as machine epsilon is still degenerate.

`degenerate_prediction_sql` tested `fold_metrics.ic IS NULL`, which is how one code path
records a correlation over a constant vector. Another path reduces the correlation
numerically instead of short-circuiting on zero variance, and stores ~2e-16. Both mean
the fold carried no ranking information; only the first was caught.

Measured 2026-09-14 on `nasdaq100_microstructure`: twelve prediction sets store a fold IC
of 2.0e-16, the registry holds no NULL fold IC at all, and all twelve reached the
published population and 396 backtests. The fleet's smallest real fold IC is -3.2e-07
(`latent_factors/cae` on `sp500_equity_option_analytics`, 525 entities), so the two
populations are separated by five orders of magnitude and the negative control below
sits between them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.notebook_contracts import (
    degenerate_prediction_hashes,
    degenerate_prediction_sql,
)

# (prediction_hash, fold ICs) - one row per fold.
CASES = (
    ("null-fold", [None, -0.012]),
    ("epsilon-fold", [2.0075e-16, 2.0877e-16]),
    ("negative-epsilon-fold", [-2.0075e-16, -1.9e-16]),
    ("small-but-real", [-3.1941e-07, 0.004]),
    ("healthy", [0.0159, -0.0121]),
)

DEGENERATE = {"null-fold", "epsilon-fold", "negative-epsilon-fold"}


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    db_path = tmp_path / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE fold_metrics (prediction_hash TEXT, fold_id INTEGER, ic REAL)")
    for phash, ics in CASES:
        for fold_id, ic in enumerate(ics):
            db.execute("INSERT INTO fold_metrics VALUES (?,?,?)", (phash, fold_id, ic))
    db.commit()
    db.close()
    return tmp_path


def test_hashes_exclude_null_and_epsilon_folds(case_dir: Path) -> None:
    assert degenerate_prediction_hashes(case_dir) == DEGENERATE


def test_a_small_but_real_ic_stays_selectable(case_dir: Path) -> None:
    """The negative control. Without it the rule could exclude everything and still pass."""
    excluded = degenerate_prediction_hashes(case_dir)
    assert "small-but-real" not in excluded
    assert "healthy" not in excluded


def test_sql_clause_applies_the_same_rule(case_dir: Path) -> None:
    db = sqlite3.connect(f"file:{case_dir / 'run_log' / 'registry.db'}?mode=ro", uri=True)
    db.execute("CREATE TEMP VIEW p AS SELECT DISTINCT prediction_hash FROM fold_metrics")
    query = "SELECT p.prediction_hash FROM p WHERE 1=1" + degenerate_prediction_sql()
    kept = {row[0] for row in db.execute(query)}
    db.close()
    assert kept == {phash for phash, _ in CASES} - DEGENERATE
