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

#: What `resolve_best_backtest_runs` actually returns. The fixture resolver is held to this
#: exactly, because a fixture that hands back `family` and `config_name` would test a frame the
#: production query cannot produce, and every count taken from it would pass here and be zero
#: against a registry.
RESOLVER_COLUMNS = ("backtest_hash", "prediction_hash", "spec_json", "sharpe")


@dataclass
class _Study:
    root: Path
    case_study: str = "fixture"


def _prediction_hash(label: str, config: str) -> str:
    return f"p-{label}-{config}"


def _study_at(
    tmp_path: Path,
    *,
    primary: str,
    variants: list[str],
    configs: tuple[str, ...] = ("c1",),
) -> _Study:
    """A study whose registry knows which configuration every prediction was fitted under.

    The configuration lives on ``training_runs`` in production, so the fixture puts it there
    rather than on the resolver's frame. That is what makes the config count under test the
    same join the notebooks run.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "setup.yaml").write_text(
        "labels:\n"
        f"  primary: {primary}\n"
        "  variants:\n" + "".join(f"    - {name}\n" for name in variants)
    )
    registry_dir = tmp_path / "run_log"
    registry_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(registry_dir / "registry.db")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            training_hash TEXT PRIMARY KEY, label TEXT, family TEXT, config_name TEXT
        );
        CREATE TABLE IF NOT EXISTS prediction_sets (
            prediction_hash TEXT PRIMARY KEY, training_hash TEXT
        );
        CREATE TABLE IF NOT EXISTS backtest_runs (
            backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT
        );
        """
    )
    for label in [primary, *variants]:
        for config in configs:
            training = f"t-{label}-{config}"
            db.execute(
                "INSERT OR REPLACE INTO training_runs VALUES (?, ?, ?, ?)",
                (training, label, "gbm", config),
            )
            db.execute(
                "INSERT OR REPLACE INTO prediction_sets VALUES (?, ?)",
                (_prediction_hash(label, config), training),
            )
    db.commit()
    db.close()
    return _Study(root=tmp_path)


def _rows(
    label: str,
    stage: str,
    *,
    sharpe: float = 1.0,
    configs: tuple[str, ...] = ("c1",),
) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "backtest_hash": [f"{label}-{stage}-{config}" for config in configs],
            "prediction_hash": [_prediction_hash(label, config) for config in configs],
            "spec_json": ["{}"] * len(configs),
            "sharpe": [sharpe] * len(configs),
        }
    )
    assert frame.columns == list(RESOLVER_COLUMNS)
    return frame


def test_field_spans_every_declared_label_and_stage(tmp_path: Path) -> None:
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_10d", "fwd_dir_5d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        assert split == "validation"
        return _rows(label, stage)

    field, _reached = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert field.height == 3 * len(FIELD_STAGES)
    assert set(field["backtest_hash"]) == {
        f"{label}-{stage}-c1"
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

    Nothing in the rows records that the drop was a decision, so this is also the shape of a
    sweep that has not started. Neither this function nor any other reading of the registry can
    separate them; what does is the plan each sweep records before it runs, checked by the
    notebook that freezes the field.
    """
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_21d"])

    def resolver(case_study, label, *, split, stage, top_n, prediction_hashes):
        if label == "fwd_ret_5d" and stage != COVERAGE_STAGE:
            return _rows(label, stage).clear()
        return _rows(label, stage)

    field, _reached = resolve_field_members(
        study,
        case_study="fixture",
        prediction_hashes=None,
        resolve_best_backtest_runs=resolver,
    )
    assert set(field["backtest_hash"]) == {
        f"fwd_ret_5d-{COVERAGE_STAGE}-c1",
        *(f"fwd_ret_21d-{stage}-c1" for stage in FIELD_STAGES),
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


def test_label_comes_from_the_winner_not_the_primary(tmp_path: Path) -> None:
    """What the stages after the selection run under is a property of what won."""
    study = _study_at(tmp_path, primary="fwd_ret_5d", variants=["fwd_ret_risk_adj_5d"])
    db = sqlite3.connect(study.root / "run_log" / "registry.db")
    db.execute(
        "INSERT INTO backtest_runs VALUES (?, ?)",
        ("b1", _prediction_hash("fwd_ret_risk_adj_5d", "c1")),
    )
    db.commit()
    db.close()

    @dataclass
    class _Result:
        hash: str

    assert label_of(study, _Result("b1")) == "fwd_ret_risk_adj_5d"

    with pytest.raises(RuntimeError, match="no label in this registry"):
        label_of(study, _Result("absent"))


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
