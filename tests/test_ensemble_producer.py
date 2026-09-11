"""Contract tests for the mean-forecast ensemble producer.

``case_studies/utils/ensemble.py`` builds the only ``family='ensemble'`` rows in
the fleet, and three readers depend on them existing:
``14_backtest.py``'s Act 2, ``20_strategy_analysis`` and
``paired_metrics.py``, which pins nasdaq's rank-1 rung on
``family == 'ensemble' AND universe_filter == 'cost_feasible'``.

What is pinned here is the part a wrong answer would not announce: which
configurations are members, which checkpoint each enters at, and that a member
covering different rows than its siblings is refused rather than averaged around.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.ensemble import (
    LIGHTGBM_DEFAULT_NUM_LEAVES,
    load_ensemble_declaration,
    mean_forecast,
    member_num_leaves,
    resolve_members,
)


def _registry(tmp_path: Path, rows: list[tuple[str, str, str, str, int]]) -> Path:
    """A case dir holding only what ``resolve_members`` reads."""
    case_dir = tmp_path / "case"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, family TEXT, "
        "label TEXT, config_name TEXT)"
    )
    db.execute(
        "CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, training_hash TEXT, "
        "checkpoint_value INTEGER, split TEXT)"
    )
    for family, label, config_name, prediction_hash, checkpoint in rows:
        # Keyed on the label as well as the configuration: a real training hash is
        # per (configuration, label), and collapsing the two lets one label's
        # checkpoints answer for another's.
        t_hash = f"t_{label}_{config_name}"
        db.execute(
            "INSERT OR IGNORE INTO training_runs VALUES (?,?,?,?)",
            (t_hash, family, label, config_name),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?,?)",
            (prediction_hash, t_hash, checkpoint, "validation"),
        )
    db.commit()
    db.close()
    return case_dir


def test_default_preset_num_leaves_is_the_library_default():
    # `default_*` declares no num_leaves, and whether it is inside or outside a
    # `<= 31` rule is what decides a 12-member ensemble from a 9-member one.
    assert member_num_leaves("default_mae") == LIGHTGBM_DEFAULT_NUM_LEAVES
    assert member_num_leaves("leaves_7_mae") == 7
    assert member_num_leaves("leaves_63_mae") == 63


def test_members_are_the_last_checkpoint_of_each_qualifying_config(tmp_path):
    case_dir = _registry(
        tmp_path,
        [
            ("gbm", "fwd_ret_60m", "leaves_7_mae", "p_l7_50", 50),
            ("gbm", "fwd_ret_60m", "leaves_7_mae", "p_l7_500", 500),
            ("gbm", "fwd_ret_60m", "leaves_63_mae", "p_l63_500", 500),
            ("gbm", "fwd_ret_60m", "default_mae", "p_def_500", 500),
            ("linear", "fwd_ret_60m", "ols", "p_ols", 1),
            ("gbm", "fwd_ret_15m", "leaves_7_mae", "p_other_label", 500),
        ],
    )
    members = resolve_members(case_dir, label="fwd_ret_60m", max_num_leaves=31)
    assert members["config_name"].to_list() == ["default_mae", "leaves_7_mae"]
    # The 500-tree checkpoint, not the 50-tree one: one row per configuration.
    assert members["prediction_hash"].to_list() == ["p_def_500", "p_l7_500"]
    assert members["num_leaves"].to_list() == [LIGHTGBM_DEFAULT_NUM_LEAVES, 7]

    # Without the rule, the 63-leaf configuration joins.
    unfiltered = resolve_members(case_dir, label="fwd_ret_60m")
    assert "leaves_63_mae" in unfiltered["config_name"].to_list()


def test_no_qualifying_member_raises_rather_than_returning_an_empty_frame(tmp_path):
    case_dir = _registry(tmp_path, [("gbm", "fwd_ret_60m", "leaves_63_mae", "p", 500)])
    with pytest.raises(ValueError, match="more than 31 leaves"):
        resolve_members(case_dir, label="fwd_ret_60m", max_num_leaves=31)
    with pytest.raises(ValueError, match="nothing to average"):
        resolve_members(case_dir, label="no_such_label")


def test_one_member_is_not_an_ensemble(monkeypatch):
    with pytest.raises(ValueError, match="at least two members"):
        mean_forecast("any_case_study", ["only_one"])


def _frame(n: int, score: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fold_id": [0] * n,
            "symbol": ["AAA"] * n,
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 1, 1) + pl.duration(minutes=n - 1),
                interval="1m",
                eager=True,
            ),
            "y_true": [0.01] * n,
            "y_score": [score] * n,
        }
    )


def test_members_that_cover_different_rows_are_refused(monkeypatch):
    frames = {"a": _frame(10, 1.0), "b": _frame(9, 3.0)}
    monkeypatch.setattr("case_studies.utils.registry.read_predictions", lambda _cs, h: frames[h])
    with pytest.raises(ValueError, match="must cover the same rows"):
        mean_forecast("cs", ["a", "b"])


def test_the_forecast_is_the_mean_of_its_members(monkeypatch):
    frames = {"a": _frame(10, 1.0), "b": _frame(10, 3.0), "c": _frame(10, 5.0)}
    monkeypatch.setattr("case_studies.utils.registry.read_predictions", lambda _cs, h: frames[h])
    out = mean_forecast("cs", ["a", "b", "c"])
    assert out.height == 10
    assert out["y_score"].to_list() == [3.0] * 10
    # The truth column is carried, not averaged away, and the schema is the
    # canonical one every other family's prediction frame normalizes into.
    assert out.columns == ["fold_id", "symbol", "timestamp", "y_true", "y_score"]


def test_nasdaq_declares_an_ensemble_and_every_other_case_study_does_not():
    declared = load_ensemble_declaration("nasdaq100_microstructure")
    assert declared is not None
    assert declared["member_family"] == "gbm"
    assert declared["max_num_leaves"] == 31
    # Act 2 filters the carrier on slots == 10 and entry_q == 0.9 literally, so the
    # featured arm's name has to carry both.
    assert "s10" in declared["featured_scheme"]
    assert "lq90" in declared["featured_scheme"]
    assert declared["universe_filter"] == "cost_feasible"
    assert load_ensemble_declaration("sp500_options") is None
