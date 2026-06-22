"""Tests for case_studies/utils/registry/queries.py cohort_metrics overrides.

Pins three behaviors of ``load_backtest_metrics``:

1. ER values from ``cohort_metrics`` override the legacy ``dsr`` /
   ``dsr_pvalue`` / ``expected_max_sharpe`` / ``min_trl_periods`` columns
   for rows that are the family leader; non-leaders pass through with
   NULLs (legacy backtest_metrics columns were dropped post-Phase-H).
2. The pre-migration fallback — when ``cohort_metrics`` doesn't exist on
   the registry — returns raw ``backtest_metrics`` rows with null
   placeholder columns for the override columns. The fallback is keyed
   on a ``sqlite_master`` probe (``_table_exists``) so the narrow case
   stays narrow.
3. Duplicate-leader_hash defense — if two family cohorts somehow share
   a leader_hash, the join keeps the first row and emits a warning
   instead of fanning out and silently changing row cardinality.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest


def _bootstrap_registry(db_path: Path, *, with_cohort_metrics: bool = True) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        # Minimal backtest_metrics schema — only the columns the override
        # touches plus backtest_hash + a passthrough column.
        conn.execute(
            """
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY,
                sharpe REAL
            )
            """
        )
        conn.executemany(
            "INSERT INTO backtest_metrics(backtest_hash, sharpe) VALUES (?, ?)",
            [("hash_leader", 1.2), ("hash_other", 0.6)],
        )
        if with_cohort_metrics:
            conn.execute(
                """
                CREATE TABLE cohort_metrics (
                    cohort_type TEXT NOT NULL,
                    stage TEXT,
                    label TEXT NOT NULL,
                    family TEXT,
                    leader_hash TEXT NOT NULL,
                    k_variants INTEGER,
                    dsr_er REAL,
                    dsr_er_pvalue REAL,
                    expected_max_sharpe_er REAL,
                    min_trl_periods_er REAL,
                    leader_min_trl REAL,
                    pbo REAL,
                    pbo_n_combinations REAL,
                    pbo_median_oos_rank REAL,
                    pbo_mean_degradation REAL,
                    pbo_n_folds REAL,
                    reality_check_pvalue REAL,
                    reality_check_statistic REAL,
                    reality_check_k REAL
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def _seed_cohort_row(
    db_path: Path,
    *,
    leader_hash: str,
    stage: str = "signal",
    label: str = "fwd_ret_21d",
    family: str = "linear",
    dsr_er: float = 0.85,
    dsr_er_pvalue: float = 0.02,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO cohort_metrics(
                cohort_type, stage, label, family, leader_hash, k_variants,
                dsr_er, dsr_er_pvalue, expected_max_sharpe_er, min_trl_periods_er,
                leader_min_trl, pbo, pbo_n_combinations, pbo_median_oos_rank,
                pbo_mean_degradation, pbo_n_folds, reality_check_pvalue,
                reality_check_statistic, reality_check_k
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "family",
                stage,
                label,
                family,
                leader_hash,
                30,
                dsr_er,
                dsr_er_pvalue,
                1.5,
                40.0,
                40.0,
                0.18,
                20.0,
                3.5,
                0.1,
                6.0,
                0.04,
                0.7,
                30.0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def registry_with_cohort(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case_x"
    db_path = case_dir / "run_log" / "registry.db"
    _bootstrap_registry(db_path, with_cohort_metrics=True)
    _seed_cohort_row(db_path, leader_hash="hash_leader")
    return case_dir


@pytest.fixture
def registry_without_cohort(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case_y"
    db_path = case_dir / "run_log" / "registry.db"
    _bootstrap_registry(db_path, with_cohort_metrics=False)
    return case_dir


def test_cohort_override_applies_to_leader_and_skips_non_leader(
    registry_with_cohort: Path,
) -> None:
    from case_studies.utils.registry.queries import load_backtest_metrics

    df = load_backtest_metrics("case_x", case_dir=registry_with_cohort)
    assert df.height == 2  # row count preserved

    leader = df.filter(pl.col("backtest_hash") == "hash_leader")
    other = df.filter(pl.col("backtest_hash") == "hash_other")

    assert leader["dsr"].item() == pytest.approx(0.85)
    assert leader["dsr_pvalue"].item() == pytest.approx(0.02)
    assert leader["k_variants"].item() == 30
    assert leader["pbo"].item() == pytest.approx(0.18)
    # Non-leader carries NULLs for the override columns.
    assert other["dsr"].item() is None
    assert other["pbo"].item() is None


def test_missing_cohort_table_returns_raw_rows_with_null_overrides(
    registry_without_cohort: Path,
) -> None:
    from case_studies.utils.registry.queries import load_backtest_metrics

    df = load_backtest_metrics("case_y", case_dir=registry_without_cohort)
    assert df.height == 2
    assert {"dsr", "pbo", "reality_check_pvalue"}.issubset(df.columns)
    # All override columns null when cohort_metrics doesn't exist.
    assert df["dsr"].is_null().all()
    assert df["pbo"].is_null().all()


def test_duplicate_leader_hash_keeps_row_cardinality_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two family cohorts with the same leader_hash must not fan out."""
    import logging

    from case_studies.utils.registry.queries import load_backtest_metrics

    case_dir = tmp_path / "case_z"
    db_path = case_dir / "run_log" / "registry.db"
    _bootstrap_registry(db_path, with_cohort_metrics=True)
    _seed_cohort_row(db_path, leader_hash="hash_leader", family="linear", dsr_er=0.85)
    _seed_cohort_row(db_path, leader_hash="hash_leader", family="gbm", dsr_er=0.99)

    with caplog.at_level(logging.WARNING, logger="case_studies.utils.registry.queries"):
        df = load_backtest_metrics("case_z", case_dir=case_dir)
    # Row count preserved; the join didn't fan out.
    assert df.height == 2
    assert any("duplicate family-cohort leader_hash" in r.message for r in caplog.records)
