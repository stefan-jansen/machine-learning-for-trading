"""The field a holdout selection is made over is built once and spans every declared label."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from case_studies.research.selection_field import (
    FIELD_STAGES,
    label_of,
    resolve_field_members,
)


@dataclass
class _Study:
    root: Path
    case_study: str = "fixture"


def _study_at(tmp_path: Path, *, primary: str, variants: list[str]) -> _Study:
    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "setup.yaml").write_text(
        "labels:\n"
        f"  primary: {primary}\n"
        "  variants:\n" + "".join(f"    - {name}\n" for name in variants)
    )
    return _Study(root=tmp_path)


def _rows(label: str, stage: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "backtest_hash": [f"{label}-{stage}"],
            "prediction_hash": [f"p-{label}"],
            "sharpe": [1.0],
        }
    )


def test_field_spans_every_declared_label_and_stage(tmp_path: Path) -> None:
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_10d", "fwd_dir_5d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        assert split == "validation"
        return _rows(label, stage)

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert field.height == 3 * len(FIELD_STAGES)
    assert set(field["backtest_hash"]) == {
        f"{label}-{stage}"
        for label in ("fwd_ret_5d", "fwd_ret_10d", "fwd_dir_5d")
        for stage in FIELD_STAGES
    }


def test_a_label_carried_through_only_some_stages_refuses_to_freeze(tmp_path: Path) -> None:
    """The sequential-run failure: baselines land before overlays, and the set is immutable.

    A per-label check that pools the three stages passes as soon as a label has ANY row, so a
    run mid-way through the second label's sweep would freeze a field holding that label's
    baselines and none of its overlays. Nothing can correct it afterwards - the set is immutable
    under its name, and every later run produces the same membership and resolves to it.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_10d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_10d" and stage == "risk_overlay":
            return _rows(label, stage).clear()
        return _rows(label, stage)

    with pytest.raises(RuntimeError, match=r"fwd_ret_10d/risk_overlay"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def test_label_comes_from_the_winner_not_the_primary(tmp_path: Path) -> None:
    """What the stages after the selection run under is a property of what won."""
    registry_dir = tmp_path / "run_log"
    registry_dir.mkdir(parents=True)
    db = sqlite3.connect(registry_dir / "registry.db")
    db.executescript(
        """
        CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT);
        CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, training_hash TEXT);
        CREATE TABLE backtest_runs (backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT);
        INSERT INTO training_runs VALUES ('t1', 'fwd_ret_risk_adj_5d');
        INSERT INTO prediction_sets VALUES ('p1', 't1');
        INSERT INTO backtest_runs VALUES ('b1', 'p1');
        """
    )
    db.commit()
    db.close()

    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_risk_adj_5d"])

    @dataclass
    class _Result:
        hash: str

    assert label_of(study, _Result("b1")) == "fwd_ret_risk_adj_5d"

    with pytest.raises(RuntimeError, match="no label in this registry"):
        label_of(study, _Result("absent"))
