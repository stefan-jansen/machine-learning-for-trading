"""The field a holdout selection is made over is built once and spans every declared label."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from case_studies.research.selection_field import (
    COVERAGE_STAGE,
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


def test_a_dominated_label_may_stop_after_the_baselines(tmp_path: Path) -> None:
    """Dropping a label after the baselines is the point of running them, not an incomplete run.

    Every declared label is backtested equal-weight, which is what makes them comparable. The
    stages after that develop whichever labels the comparison favours, so a label the baselines
    show to be dominated is deliberately not carried into allocation or risk overlay. A rule
    demanding every label in every stage would order backtests whose only purpose is filling a
    matrix, and would refuse to freeze a field that is finished.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_5d" and stage != COVERAGE_STAGE:
            return _rows(label, stage).clear()
        return _rows(label, stage)

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert set(field["backtest_hash"]) == {
        f"fwd_ret_5d-{COVERAGE_STAGE}",
        *(f"fwd_ret_21d-{stage}" for stage in FIELD_STAGES),
    }


def test_a_label_with_no_baseline_refuses_to_freeze(tmp_path: Path) -> None:
    """The sequential-run failure, at the one stage where absence means unfinished.

    A run mid-way through the second label's baseline sweep would otherwise freeze a field that
    excludes it, and nothing can correct that afterwards: the set is immutable under its name,
    and every later run produces the same membership and resolves to it.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_10d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_10d":
            return _rows(label, stage).clear()
        return _rows(label, stage)

    with pytest.raises(RuntimeError, match=r"fwd_ret_10d"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def test_a_stage_no_label_reached_refuses_to_freeze(tmp_path: Path) -> None:
    """One label dropping out of a stage is a decision; every label missing is an unrun stage."""
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if stage == "risk_overlay":
            return _rows(label, stage).clear()
        return _rows(label, stage)

    with pytest.raises(RuntimeError, match="risk_overlay"):
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
