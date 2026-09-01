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


def _rows(label: str, stage: str, *, sharpe: float = 1.0, config: str = "c1") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "backtest_hash": [f"{label}-{stage}"],
            "prediction_hash": [f"p-{label}"],
            "sharpe": [sharpe],
            "family": ["gbm"],
            "config_name": [config],
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

    Which labels the comparison favoured is read off the pooled baseline ranking rather than
    off which labels happen to have allocation rows. Here the cut keeps one configuration, and
    ``fwd_ret_21d`` outranks ``fwd_ret_5d`` at the baseline, so ``fwd_ret_5d`` stopping there is
    the decision the baselines made.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_5d" and stage != COVERAGE_STAGE:
            return _rows(label, stage).clear()
        return _rows(label, stage, sharpe=2.0 if label == "fwd_ret_21d" else 0.5)

    field = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
        advancing_top_n=1,
    )
    assert set(field["backtest_hash"]) == {
        f"fwd_ret_5d-{COVERAGE_STAGE}",
        *(f"fwd_ret_21d-{stage}" for stage in FIELD_STAGES),
    }


def test_a_favoured_label_whose_allocation_never_started_refuses_to_freeze(
    tmp_path: Path,
) -> None:
    """The case reading post-baseline rows alone cannot see.

    ``fwd_ret_21d`` tops the pooled baseline ranking and has no rows in any advancing stage.
    Read off observed rows that is indistinguishable from a label dropped after the baselines;
    read off the ranking it is a sweep that has not started, and freezing now would publish a
    field missing the candidates the comparison actually selected.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_21d" and stage != COVERAGE_STAGE:
            return _rows(label, stage).clear()
        return _rows(label, stage, sharpe=2.0 if label == "fwd_ret_21d" else 0.5)

    with pytest.raises(RuntimeError, match="fwd_ret_21d"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
            advancing_top_n=1,
        )


def test_without_a_cut_every_label_is_required_to_advance(tmp_path: Path) -> None:
    """No cut means the question is unanswerable, and an unanswerable question refuses.

    A caller that cannot supply ``top_n`` cannot say which labels the baselines favoured, so
    every label is treated as favoured. Returning an empty favoured set instead would excuse
    every label from the check and freeze whatever happened to be there.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_5d" and stage != COVERAGE_STAGE:
            return _rows(label, stage).clear()
        return _rows(label, stage, sharpe=2.0 if label == "fwd_ret_21d" else 0.5)

    with pytest.raises(RuntimeError, match="fwd_ret_5d"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


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


def test_a_label_part_way_through_advancing_refuses_to_freeze(tmp_path: Path) -> None:
    """Neither a decision nor coverage: a sweep still running.

    A label that stops at the baseline was dropped there. A label that reached allocation and
    not risk overlay is mid-sweep, and freezing now would permanently exclude the risk
    candidates it is about to produce - the set is immutable under its name.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_21d" and stage == "risk_overlay":
            return _rows(label, stage).clear()
        return _rows(label, stage)

    with pytest.raises(RuntimeError, match="part-way through advancing"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )


def test_an_unrankable_row_does_not_satisfy_coverage(tmp_path: Path) -> None:
    """A null Sharpe cannot be ranked, so it is not a backtest the field can select from.

    Counting it towards coverage freezes a set that ``best_validation_sharpe`` then rejects
    whole, for holding a member it cannot rank - the failure lands after the set is immutable.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        rows = _rows(label, stage)
        if label == "fwd_ret_21d":
            return rows.with_columns(sharpe=pl.lit(None, dtype=pl.Float64))
        return rows

    with pytest.raises(RuntimeError, match="fwd_ret_21d"):
        resolve_field_members(
            study,
            case_study="fixture",
            prediction_hashes=None,
            resolve_best_backtest_runs=resolver,
        )
