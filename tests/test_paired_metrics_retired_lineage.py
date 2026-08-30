"""Retirement crosses the split through the training run, not the prediction hash."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import paired_metrics


def _registry(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """``rows`` are (prediction_hash, training_hash, split)."""
    case_dir = tmp_path / "probe"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "CREATE TABLE prediction_sets (prediction_hash TEXT, training_hash TEXT, split TEXT)"
    )
    db.executemany("INSERT INTO prediction_sets VALUES (?,?,?)", rows)
    db.commit()
    db.close()
    return case_dir


@pytest.fixture(autouse=True)
def _clear_cache():
    paired_metrics._retired_training_hashes.cache_clear()
    yield
    paired_metrics._retired_training_hashes.cache_clear()


def _install(monkeypatch, case_dir: Path, retired: set[str]) -> None:
    monkeypatch.setattr(paired_metrics, "get_case_study_dir", lambda cs: case_dir)
    monkeypatch.setattr(
        "case_studies.research.population.superseded_members_at",
        lambda _dir, member_kind="prediction": frozenset(retired),
    )


def test_a_retired_validation_prediction_retires_its_holdout_sibling(monkeypatch, tmp_path) -> None:
    """The two carry different prediction hashes, so only the training run connects them."""
    case_dir = _registry(
        tmp_path,
        [
            ("val_old", "train_old", "validation"),
            ("ho_old", "train_old", "holdout"),
            ("val_new", "train_new", "validation"),
            ("ho_new", "train_new", "holdout"),
        ],
    )
    _install(monkeypatch, case_dir, {"val_old"})

    retired = paired_metrics._retired_training_hashes("probe")

    assert retired == frozenset({"train_old"})


def test_a_run_whose_other_checkpoints_are_live_is_not_retired(monkeypatch, tmp_path) -> None:
    """A population that moved one checkpoint of a run has not replaced the run."""
    case_dir = _registry(
        tmp_path,
        [
            ("ckpt_25", "train_one", "validation"),
            ("ckpt_50", "train_one", "validation"),
            ("ho", "train_one", "holdout"),
        ],
    )
    _install(monkeypatch, case_dir, {"ckpt_25"})

    assert paired_metrics._retired_training_hashes("probe") == frozenset()


def test_no_lineage_retires_nothing(monkeypatch, tmp_path) -> None:
    case_dir = _registry(tmp_path, [("val", "train", "validation")])
    _install(monkeypatch, case_dir, set())

    assert paired_metrics._retired_training_hashes("probe") == frozenset()
