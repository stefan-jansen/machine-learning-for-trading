"""What `select_holdout_self_backtest` will and will not accept as the holdout replay.

Every strategy-analysis notebook reads its holdout number through this lookup, so a candidate
it accepts wrongly is a holdout result reported against a configuration nothing selected.
`config_name` cannot carry that weight on its own: a stale generation and an experimental
variant both register under the same name.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.strategy_analysis import select_holdout_self_backtest

STRATEGY = {"signal": {"method": "equal_weight_top_k", "top_k": 1}}


def _training_spec(split: str, folds: int, *, alpha: float = 0.1, features: str = "feat") -> str:
    return json.dumps(
        {
            "family": "linear",
            "label": "fwd_ret_5d",
            "seed": 42,
            "computation": {
                "cv": {"split": split, "identity": f"cv-{split}"},
                "expected_prediction_keys": {"n_folds": folds},
                "model": {
                    "class": "Lasso",
                    "params": {"alpha": alpha},
                    # Re-keyed by the refit, so it must not be compared for equality.
                    "effective_params_by_fold": {str(i): {"alpha": alpha} for i in range(folds)},
                },
                "feature_artifacts": {"features": features},
                "label_artifact": {"digest": "label-digest"},
            },
        }
    )


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    case_dir = tmp_path / "case_studies" / "fixture_cs"
    (case_dir / "run_log").mkdir(parents=True)
    db_path = case_dir / "run_log" / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, spec_json TEXT, "
            "family TEXT, config_name TEXT, label TEXT)"
        )
        db.execute(
            "CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, training_hash TEXT, "
            "split TEXT, checkpoint_kind TEXT, checkpoint_value TEXT)"
        )
        db.execute(
            "CREATE TABLE backtest_runs (backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, "
            "spec_json TEXT)"
        )
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda _cs: case_dir)
    return db_path


def _seed(db_path: Path, holdout_specs: dict[str, str]) -> None:
    """The validation run and its backtest, plus one holdout candidate per entry."""
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?,?)",
            ("tr-val", _training_spec("validation", 5), "linear", "lasso", "fwd_ret_5d"),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?,?,?)",
            ("pr-val", "tr-val", "validation", "final", None),
        )
        db.execute(
            "INSERT INTO backtest_runs VALUES (?,?,?)",
            ("bt-val", "pr-val", json.dumps({"strategy": STRATEGY})),
        )
        for name, spec in holdout_specs.items():
            db.execute(
                "INSERT INTO training_runs VALUES (?,?,?,?,?)",
                (f"tr-{name}", spec, "linear", "lasso", "fwd_ret_5d"),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?,?,?,?,?)",
                (f"pr-{name}", f"tr-{name}", "holdout", "final", None),
            )
            db.execute(
                "INSERT INTO backtest_runs VALUES (?,?,?)",
                (f"bt-{name}", f"pr-{name}", json.dumps({"strategy": STRATEGY})),
            )


def test_the_refit_of_this_configuration_is_the_replay(registry: Path) -> None:
    """The positive case, and it must survive the checks: only cv and fold-derived fields move."""
    _seed(registry, {"good": _training_spec("holdout", 1)})
    assert select_holdout_self_backtest("fixture_cs", "bt-val") == "bt-good"


def test_a_refit_with_a_different_model_parameter_is_not_the_replay(registry: Path) -> None:
    """Same config_name, genuinely refitted, different alpha - a different question."""
    _seed(registry, {"variant": _training_spec("holdout", 1, alpha=0.9)})
    assert select_holdout_self_backtest("fixture_cs", "bt-val") is None


def test_a_refit_on_a_different_feature_lineage_is_not_the_replay(registry: Path) -> None:
    """The stale-generation case: features regenerated, config_name unchanged."""
    _seed(registry, {"stale": _training_spec("holdout", 1, features="feat-v2")})
    assert select_holdout_self_backtest("fixture_cs", "bt-val") is None


def test_the_variant_does_not_hide_the_real_replay(registry: Path) -> None:
    """Both present: the matching refit is returned and the variant is neither picked nor
    treated as an ambiguity, which is what a `config_name` match alone would have produced."""
    _seed(
        registry,
        {
            "good": _training_spec("holdout", 1),
            "variant": _training_spec("holdout", 1, alpha=0.9),
        },
    )
    assert select_holdout_self_backtest("fixture_cs", "bt-val") == "bt-good"
