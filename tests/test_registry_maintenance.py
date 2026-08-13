"""Tests for idempotent semantic backtest maintenance."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from case_studies.utils.registry.maintenance import deduplicate_semantic_backtests
from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL


def _seed_duplicate_registry(path: Path) -> None:
    old_spec = {
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 5, "universe_filter": "full"}
        }
    }
    current_spec = {
        "strategy": {
            "signal": {
                "method": "equal_weight_top_k",
                "top_k": 5,
                "universe_filter": "full",
                "direction": "long_only",
            }
        }
    }
    with closing(sqlite3.connect(str(path))) as db, db:
        db.executescript(REGISTRY_SCHEMA_SQL)
        db.execute(
            "INSERT INTO training_runs "
            "(training_hash, family, label, config_name, created_at) VALUES (?,?,?,?,?)",
            ("train", "linear", "ret_to_expiry", "ridge", "2026-07-21"),
        )
        db.execute(
            "INSERT INTO prediction_sets "
            "(prediction_hash, training_hash, split, created_at) VALUES (?,?,?,?)",
            ("pred", "train", "validation", "2026-07-21"),
        )
        for backtest_hash, spec in (("legacy", old_spec), ("current", current_spec)):
            db.execute(
                "INSERT INTO backtest_runs "
                "(backtest_hash, prediction_hash, spec_json, stage, created_at) VALUES (?,?,?,?,?)",
                (backtest_hash, "pred", json.dumps(spec), "signal", "2026-07-21"),
            )
            db.execute(
                "INSERT INTO backtest_metrics (backtest_hash, computed_at, sharpe) VALUES (?,?,?)",
                (backtest_hash, "2026-07-21", 0.1),
            )


def test_semantic_backtest_deduplication_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.db"
    _seed_duplicate_registry(db_path)

    dry_run = deduplicate_semantic_backtests(db_path)
    assert len(dry_run) == 1
    assert len(dry_run[0].drop_hashes) == 1

    applied = deduplicate_semantic_backtests(db_path, apply=True)
    assert applied == dry_run
    assert deduplicate_semantic_backtests(db_path, apply=True) == []

    with closing(sqlite3.connect(str(db_path))) as db:
        assert db.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM backtest_metrics").fetchone()[0] == 1
