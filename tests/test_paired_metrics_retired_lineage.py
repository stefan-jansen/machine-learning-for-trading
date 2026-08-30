"""Retirement crosses the split by training run and checkpoint, member-wise."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import paired_metrics


def _registry(tmp_path: Path, rows: list[tuple[str, str, str, str, int]]) -> Path:
    """``rows`` are (prediction_hash, training_hash, split, checkpoint_kind, value)."""
    case_dir = tmp_path / "probe"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "CREATE TABLE prediction_sets (prediction_hash TEXT, training_hash TEXT, "
        "split TEXT, checkpoint_kind TEXT, checkpoint_value INTEGER)"
    )
    db.executemany("INSERT INTO prediction_sets VALUES (?,?,?,?,?)", rows)
    db.commit()
    db.close()
    return case_dir


@pytest.fixture(autouse=True)
def _clear_cache():
    paired_metrics._retired_prediction_hashes.cache_clear()
    yield
    paired_metrics._retired_prediction_hashes.cache_clear()


def _install(monkeypatch, case_dir: Path, recorded: set[str]) -> None:
    monkeypatch.setattr(paired_metrics, "get_case_study_dir", lambda cs: case_dir)
    monkeypatch.setattr(
        "case_studies.research.population.superseded_members_at",
        lambda _dir, member_kind="prediction": frozenset(recorded),
    )


def test_a_retired_validation_prediction_retires_its_holdout_sibling(monkeypatch, tmp_path) -> None:
    """The two carry different prediction hashes; the training run and checkpoint connect them."""
    case_dir = _registry(
        tmp_path,
        [
            ("val_old", "train_old", "validation", "epoch", 50),
            ("ho_old", "train_old", "holdout", "epoch", 50),
            ("val_new", "train_new", "validation", "epoch", 50),
            ("ho_new", "train_new", "holdout", "epoch", 50),
        ],
    )
    _install(monkeypatch, case_dir, {"val_old"})

    assert paired_metrics._retired_prediction_hashes("probe") == frozenset({"val_old", "ho_old"})


def test_a_live_sibling_checkpoint_does_not_rescue_the_retired_one(monkeypatch, tmp_path) -> None:
    """Member-wise: moving one checkpoint of a run retires that checkpoint and no other."""
    case_dir = _registry(
        tmp_path,
        [
            ("val_25", "train_one", "validation", "epoch", 25),
            ("ho_25", "train_one", "holdout", "epoch", 25),
            ("val_50", "train_one", "validation", "epoch", 50),
            ("ho_50", "train_one", "holdout", "epoch", 50),
        ],
    )
    _install(monkeypatch, case_dir, {"val_25"})

    retired = paired_metrics._retired_prediction_hashes("probe")

    assert retired == frozenset({"val_25", "ho_25"})
    assert "ho_50" not in retired


def test_no_lineage_retires_nothing(monkeypatch, tmp_path) -> None:
    case_dir = _registry(tmp_path, [("val", "train", "validation", "epoch", 50)])
    _install(monkeypatch, case_dir, set())

    assert paired_metrics._retired_prediction_hashes("probe") == frozenset()
