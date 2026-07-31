"""Tests for the two sampling steps a strategy-analysis notebook depends on.

Both exist because a fixture that is internally consistent can still fail a contract
the production registry satisfies. Step 3c keeps a retained prediction's full
cost/risk grid, which top-N-per-(family, stage) slices across predictions; the
artifact copy places the per-hash files selection later reads. Neither had coverage,
so narrowing the downstream query or dropping the copy would resurface the same
fixture defect only at the next full regeneration.
"""

import sqlite3
from pathlib import Path

import pytest

from tests.sample_registry_for_tests import _copy_backtest_artifacts, _populate_sample_db

# The declared risk grid a strategy-analysis notebook plans in full for its carrier.
RISK_GRID = 14
COST_GRID = 5
# Above TOP_N_PER_GROUP (3), so top-N alone leaves the allocation grid partial too.
# At one row per prediction the grid survived sampling by accident and the
# completeness assertion below held without step 3c doing anything.
ALLOC_GRID = 4

# Descending base Sharpe per prediction. Four of them, so the signal bucket exceeds
# TOP_N_PER_GROUP (3) and the weakest is ranked out of every stage's top-N. That
# weakest one is the cohort leader, so it can only enter the sample through step 3c
# and its complete grid can only come from step 3d. With a leader that survives top-N
# on its own, every assertion below holds with the leader-seeding block deleted.
BASE_SHARPE = {"P1": 1.0, "P2": 0.9, "P3": 0.8, "P_LEADER": 0.1}
LEADER_PRED = "P_LEADER"
LEADER_HASH = f"{LEADER_PRED}_signal_0"

COHORT_SCHEMA = """
    CREATE TABLE cohort_metrics (
        cohort_type TEXT, label TEXT, stage TEXT, leader_hash TEXT
    );
"""


def _build_source_db(path: Path, cohort_schema: str | None = COHORT_SCHEMA) -> None:
    """One family, four predictions, each with a complete cost/risk grid.

    Sharpe is assigned so top-N-per-(family, stage) alone keeps only 3 of each
    prediction's 14 risk_overlay rows - the partial-set shape step 3c must repair -
    and so the cohort leader is ranked out of top-N entirely.

    ``cohort_schema`` is the DDL for cohort_metrics, or None to omit the table, so the
    two branches of the sampler's existence guard can be exercised.
    """
    db = sqlite3.connect(str(path))
    db.executescript("""
        CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, family TEXT);
        CREATE TABLE prediction_sets (
            prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
        );
        CREATE TABLE prediction_metrics (prediction_hash TEXT, ic REAL);
        CREATE TABLE fold_metrics (prediction_hash TEXT, fold INTEGER);
        CREATE TABLE backtest_runs (
            backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT
        );
        CREATE TABLE backtest_metrics (backtest_hash TEXT PRIMARY KEY, sharpe REAL);
        CREATE TABLE backtest_fold_metrics (backtest_hash TEXT, fold INTEGER);
    """)
    if cohort_schema:
        db.executescript(cohort_schema)
    db.execute("INSERT INTO training_runs VALUES ('T1', 'gbm')")
    if cohort_schema:
        # One leader ranked out of top-N in every stage, which only step 3c can pull
        # into the sample, and one whose leader_hash was never a real backtest_run at
        # all, standing in for a leader that sampling dropped - a strategy-analysis
        # notebook's JOIN through leader_hash must drop the latter and keep the former.
        # stage='signal' is part of the leader query's WHERE clause, so a cohort_metrics
        # table without that column makes step 3c raise rather than run.
        cols = cohort_schema.count(",") + 1
        row = ("stagelabel", "labelA", "signal", LEADER_HASH)[-cols:]
        ghost = ("stagelabel", "labelA", "signal", "GHOST_NOT_SAMPLED")[-cols:]
        ph = ",".join(["?"] * cols)
        db.execute(f"INSERT INTO cohort_metrics VALUES ({ph})", row)
        db.execute(f"INSERT INTO cohort_metrics VALUES ({ph})", ghost)
    for pred, base in BASE_SHARPE.items():
        db.execute("INSERT INTO prediction_sets VALUES (?, 'T1', 'validation')", (pred,))
        db.execute("INSERT INTO prediction_metrics VALUES (?, 0.01)", (pred,))
        for stage, n in (
            ("signal", 1),
            ("allocation", ALLOC_GRID),
            ("risk_overlay", RISK_GRID),
            ("cost_sensitivity", COST_GRID),
        ):
            for i in range(n):
                bt = f"{pred}_{stage}_{i}"
                db.execute("INSERT INTO backtest_runs VALUES (?, ?, ?)", (bt, pred, stage))
                db.execute("INSERT INTO backtest_metrics VALUES (?, ?)", (bt, base - i * 0.001))
                db.execute("INSERT INTO backtest_fold_metrics VALUES (?, 0)", (bt,))
    db.commit()
    db.close()


def _sample(tmp_path: Path) -> sqlite3.Connection:
    src_path = tmp_path / "src.db"
    _build_source_db(src_path)
    dst_path = tmp_path / "dst.db"
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        _populate_sample_db(src, dst, dst_path)
        dst.commit()
    finally:
        src.close()
    return dst


@pytest.mark.parametrize(
    ("stage", "grid"), [("risk_overlay", RISK_GRID), ("allocation", ALLOC_GRID)]
)
def test_top_n_alone_would_keep_a_partial_grid(tmp_path: Path, stage: str, grid: int) -> None:
    """Pins the defect's precondition, so this suite fails loudly rather than
    vacuously if the source fixture stops exercising the slicing at all."""
    src_path = tmp_path / "src.db"
    _build_source_db(src_path)
    src = sqlite3.connect(str(src_path))
    try:
        kept = src.execute(
            """
            WITH ranked AS (
                SELECT b.backtest_hash, ROW_NUMBER() OVER (
                    PARTITION BY b.stage, t.family ORDER BY ABS(bm.sharpe) DESC
                ) AS rn
                FROM backtest_runs b
                JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
                JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
                JOIN training_runs t ON p.training_hash = t.training_hash
                WHERE p.split != 'holdout'
            )
            SELECT COUNT(*) FROM ranked WHERE rn <= 3
              AND backtest_hash LIKE ?
            """,
            (f"P1_{stage}_%",),
        ).fetchone()[0]
    finally:
        src.close()
    assert 0 < kept < grid


def test_the_cohort_leader_is_ranked_out_of_top_n(tmp_path: Path) -> None:
    """The other precondition. If the leader ever starts surviving top-N on its own,
    the two assertions below stop testing step 3c and start passing for free."""
    src_path = tmp_path / "src.db"
    _build_source_db(src_path)
    src = sqlite3.connect(str(src_path))
    try:
        kept = src.execute(
            """
            WITH ranked AS (
                SELECT b.backtest_hash, ROW_NUMBER() OVER (
                    PARTITION BY b.stage, t.family ORDER BY ABS(bm.sharpe) DESC
                ) AS rn
                FROM backtest_runs b
                JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
                JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
                JOIN training_runs t ON p.training_hash = t.training_hash
                WHERE p.split != 'holdout'
            )
            SELECT COUNT(*) FROM ranked WHERE rn <= 3 AND backtest_hash LIKE ?
            """,
            (f"{LEADER_PRED}_%",),
        ).fetchone()[0]
    finally:
        src.close()
    assert kept == 0, f"{LEADER_PRED} survives top-N on its own; step 3c is not under test"


def test_a_leader_outside_top_n_is_pulled_in_with_its_complete_grid(tmp_path: Path) -> None:
    """What step 3c exists for: the frozen carrier a notebook pins by hash is not the
    top performer, so top-N drops it and the JOIN through leader_hash finds nothing."""
    dst = _sample(tmp_path)
    try:
        assert dst.execute(
            "SELECT 1 FROM backtest_runs WHERE backtest_hash = ?", (LEADER_HASH,)
        ).fetchone(), f"{LEADER_HASH} was not seeded into the sample"
        for stage, expected in (
            ("allocation", ALLOC_GRID),
            ("risk_overlay", RISK_GRID),
            ("cost_sensitivity", COST_GRID),
        ):
            n = dst.execute(
                "SELECT COUNT(*) FROM backtest_runs WHERE prediction_hash=? AND stage=?",
                (LEADER_PRED, stage),
            ).fetchone()[0]
            assert n == expected, f"{LEADER_PRED}/{stage}: {n} of {expected}"
    finally:
        dst.close()


def test_a_registry_without_cohort_metrics_still_samples(tmp_path: Path) -> None:
    """The one condition the existence guard exists to tolerate: a registry old enough
    to predate the table. It has no leaders to seed, and that is not an error."""
    src_path = tmp_path / "src.db"
    _build_source_db(src_path, cohort_schema=None)
    dst_path = tmp_path / "dst.db"
    src, dst = sqlite3.connect(str(src_path)), sqlite3.connect(str(dst_path))
    try:
        stats = _populate_sample_db(src, dst, dst_path)
        assert stats["status"] == "OK"
        assert stats["backtest_runs_sampled"] > 0
    finally:
        src.close()
        dst.close()


def test_a_cohort_metrics_table_missing_stage_is_not_swallowed(tmp_path: Path) -> None:
    """Everything the guard must NOT tolerate, standing in for schema drift generally.
    Reporting a successful sample with an empty leader set is the failure mode: the
    caller in seed_results.py reads that set to decide which real carrier artifacts to
    keep, and an empty one sends every last of them back through the synthetic rewrite.
    """
    src_path = tmp_path / "src.db"
    _build_source_db(
        src_path,
        cohort_schema="CREATE TABLE cohort_metrics (cohort_type TEXT, label TEXT, leader_hash TEXT);",
    )
    dst_path = tmp_path / "dst.db"
    src, dst = sqlite3.connect(str(src_path)), sqlite3.connect(str(dst_path))
    try:
        with pytest.raises(sqlite3.OperationalError, match="stage"):
            _populate_sample_db(src, dst, dst_path)
    finally:
        src.close()
        dst.close()


def test_every_retained_prediction_keeps_its_complete_grid(tmp_path: Path) -> None:
    dst = _sample(tmp_path)
    try:
        retained = [
            row[0]
            for row in dst.execute("SELECT DISTINCT prediction_hash FROM backtest_runs").fetchall()
        ]
        assert retained, "sampling kept no prediction at all"
        for pred in retained:
            for stage, expected in (
                ("allocation", ALLOC_GRID),
                ("risk_overlay", RISK_GRID),
                ("cost_sensitivity", COST_GRID),
            ):
                n = dst.execute(
                    "SELECT COUNT(*) FROM backtest_runs WHERE prediction_hash=? AND stage=?",
                    (pred, stage),
                ).fetchone()[0]
                assert n == expected, f"{pred}/{stage}: {n} of {expected}"
    finally:
        dst.close()


def test_metrics_follow_every_completed_row(tmp_path: Path) -> None:
    """A completed run row with no metrics row fails downstream, not at the sample."""
    dst = _sample(tmp_path)
    try:
        orphans = dst.execute("""
            SELECT COUNT(*) FROM backtest_runs b
            LEFT JOIN backtest_metrics m ON b.backtest_hash = m.backtest_hash
            WHERE m.backtest_hash IS NULL
        """).fetchone()[0]
        assert orphans == 0
    finally:
        dst.close()


def test_cohort_metrics_keeps_only_sampled_leaders(tmp_path: Path) -> None:
    """A cohort row's leader_hash must survive the JOIN a downstream notebook makes
    against backtest_runs, or resolving the frozen carrier fails outright regardless
    of how complete the rest of the sample is."""
    dst = _sample(tmp_path)
    try:
        leaders = {r[0] for r in dst.execute("SELECT leader_hash FROM cohort_metrics").fetchall()}
        assert leaders == {LEADER_HASH}
    finally:
        dst.close()


def _artifact_tree(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "src_run_log"
    (src / "backtest" / "complete").mkdir(parents=True)
    (src / "backtest" / "complete" / "daily_returns.parquet").write_bytes(b"x")
    (src / "backtest" / "complete" / "spec.json").write_text("{}")
    (src / "backtest" / "no_returns").mkdir(parents=True)
    (src / "backtest" / "no_returns" / "spec.json").write_text("{}")
    return src, tmp_path / "dst_run_log"


def test_artifacts_are_placed_for_each_sampled_hash(tmp_path: Path) -> None:
    src, dst = _artifact_tree(tmp_path)
    result = _copy_backtest_artifacts(src, dst, {"complete"})

    assert result == {"copied": 1, "missing_dir": 0, "missing_returns": 0}
    assert (dst / "backtest" / "complete" / "daily_returns.parquet").is_file()
    assert (dst / "backtest" / "complete" / "spec.json").is_file()


def test_a_gap_in_the_source_is_reported_not_swallowed(tmp_path: Path) -> None:
    src, dst = _artifact_tree(tmp_path)
    result = _copy_backtest_artifacts(src, dst, {"complete", "no_returns", "absent"})

    assert result["missing_dir"] == 1
    assert result["missing_returns"] == 1
    assert not (dst / "backtest" / "absent").exists()


def test_a_source_with_no_backtest_dir_reports_every_hash_missing(tmp_path: Path) -> None:
    result = _copy_backtest_artifacts(tmp_path / "nothing", tmp_path / "dst", {"a", "b"})

    assert result == {"copied": 0, "missing_dir": 2, "missing_returns": 0}
