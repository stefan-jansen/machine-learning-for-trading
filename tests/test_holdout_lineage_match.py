"""Which holdout backtest belongs to the validation rank-1.

A holdout prediction produced correctly carries a NEW training identity: it is the same
declared configuration refitted on the holdout fold, and the identity covers the CV
interval. Matching on the validation training hash therefore selects only a holdout scored
from the validation-fitted model - the one thing a holdout exists to rule out.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.strategy_analysis import select_holdout_self_backtest

STRATEGY = {"signal": {"method": "equal_weight_top_k", "top_k": 50}}
OTHER_STRATEGY = {"signal": {"method": "equal_weight_top_k", "top_k": 10}}

VALIDATION_CV = {"folds": [{"fold": "0"}, {"fold": "1"}], "identity": "val"}
HOLDOUT_CV = {"folds": [{"fold": "2"}], "identity": "ho", "split": "holdout"}


def _registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT,
                spec_json TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_value INTEGER, checkpoint_kind TEXT
            );
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, spec_json TEXT
            );
            """
        )
        for row in rows:
            db.execute(
                "INSERT OR IGNORE INTO training_runs VALUES (?, 'gbm', 'leaves_63', 'y', ?)",
                (
                    row["training_hash"],
                    json.dumps({"computation": {"cv": row["cv"]}}),
                ),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, ?, 50, 'iteration')",
                (row["prediction_hash"], row["training_hash"], row["split"]),
            )
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, ?)",
                (
                    row["backtest_hash"],
                    row["prediction_hash"],
                    json.dumps({"strategy": row["strategy"]}),
                ),
            )


VALIDATION_ROW = {
    "training_hash": "train_val",
    "prediction_hash": "pred_val",
    "backtest_hash": "bt_val",
    "split": "validation",
    "strategy": STRATEGY,
    "cv": VALIDATION_CV,
}
REFITTED_ROW = {
    "training_hash": "train_holdout",
    "prediction_hash": "pred_holdout",
    "backtest_hash": "bt_holdout",
    "split": "holdout",
    "strategy": STRATEGY,
    "cv": HOLDOUT_CV,
}
# The defect: predictions land on the holdout window, but the model that made them was
# fitted on the validation folds - so its parameters were chosen while looking at the
# period it is being judged against.
VALIDATION_FITTED_ROW = {
    "training_hash": "train_val",
    "prediction_hash": "pred_holdout_defective",
    "backtest_hash": "bt_holdout_defective",
    "split": "holdout",
    "strategy": STRATEGY,
    "cv": VALIDATION_CV,
}


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "utils.paths.get_case_study_dir", lambda case_study, **_: tmp_path / case_study
    )
    return tmp_path / "fixture_case_study"


def test_the_refitted_holdout_is_matched(case_dir: Path) -> None:
    _registry(case_dir / "run_log" / "registry.db", [VALIDATION_ROW, REFITTED_ROW])
    assert select_holdout_self_backtest("fixture_case_study", "bt_val") == "bt_holdout"


def test_a_validation_fitted_holdout_is_not_the_lineage(case_dir: Path) -> None:
    """It is the only thing the old match could find, and it is the defect."""
    _registry(case_dir / "run_log" / "registry.db", [VALIDATION_ROW, VALIDATION_FITTED_ROW])
    assert select_holdout_self_backtest("fixture_case_study", "bt_val") is None


def test_the_refit_wins_where_both_generations_are_registered(case_dir: Path) -> None:
    """The registry is immutable, so a corrected re-run leaves the defective row in place."""
    _registry(
        case_dir / "run_log" / "registry.db",
        [VALIDATION_ROW, VALIDATION_FITTED_ROW, REFITTED_ROW],
    )
    assert select_holdout_self_backtest("fixture_case_study", "bt_val") == "bt_holdout"


def test_a_different_strategy_on_the_same_refit_is_not_matched(case_dir: Path) -> None:
    """A side-channel allocator sharing the holdout prediction must not displace the anchor."""
    side_channel = {
        **REFITTED_ROW,
        "prediction_hash": "pred_holdout_side",
        "backtest_hash": "bt_holdout_side",
        "strategy": OTHER_STRATEGY,
    }
    _registry(case_dir / "run_log" / "registry.db", [VALIDATION_ROW, REFITTED_ROW, side_channel])
    assert select_holdout_self_backtest("fixture_case_study", "bt_val") == "bt_holdout"


def test_two_indistinguishable_refits_raise_rather_than_one_being_picked(case_dir: Path) -> None:
    twin = {
        **REFITTED_ROW,
        "training_hash": "train_holdout_twin",
        "prediction_hash": "pred_holdout_twin",
        "backtest_hash": "bt_holdout_twin",
    }
    _registry(case_dir / "run_log" / "registry.db", [VALIDATION_ROW, REFITTED_ROW, twin])
    with pytest.raises(ValueError, match="ambiguous"):
        select_holdout_self_backtest("fixture_case_study", "bt_val")
