from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CandidateSet, Result, Study
from case_studies.utils.registry import (
    backtest_hash_from_parts,
    prediction_hash_from_parts,
    training_hash_from_spec,
)
from case_studies.utils.registry.store import _open_registry
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _training_spec(**changes) -> dict:
    spec = {
        "identity_version": 2,
        "family": "linear",
        "label": "fwd_ret_21d",
        "label_artifact": "label-a",
        "feature_artifacts": {"financial": "features-a"},
        "feature_names": ["momentum", "volatility"],
        "cv": {"folds": [{"fold": 0, "val_start": "2024-01-05"}]},
        "model": {"class": "Ridge", "params": {"alpha": 1.0}},
        "numerics": {"seed": 42, "precision": "float64"},
        "execution_tier": "canonical",
        "seed": 42,
    }
    spec.update(changes)
    return spec


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": ["2024-01-05", "2024-01-05"],
            "fold_id": [0, 0],
            "y_true": [0.01, -0.02],
            "y_score": [0.02, -0.01],
        }
    ).with_columns(pl.col("timestamp").str.to_date())


def test_version_2_identity_is_semantic_and_legacy_hashes_are_pinned() -> None:
    base = _training_spec()
    same = {**base, "display_name": "Reader experiment", "notebook_path": "/tmp/book.ipynb"}
    assert training_hash_from_spec(base) == training_hash_from_spec(same)

    for mutation in (
        {"label_artifact": "label-b"},
        {"feature_artifacts": {"financial": "features-b"}},
        {"cv": {"folds": [{"fold": 0, "val_start": "2024-01-06"}]}},
        {"model": {"class": "Ridge", "params": {"alpha": 2.0}}},
        {"execution_tier": "preview"},
    ):
        assert training_hash_from_spec(base) != training_hash_from_spec({**base, **mutation})

    legacy = {"family": "linear", "label": "fwd_ret_21d", "seed": 42}
    assert training_hash_from_spec(legacy) == "a32ccdc1db2e"
    assert prediction_hash_from_parts("abc", None, "validation") == "f8d8ffe712a3"
    assert backtest_hash_from_parts("pred", {"top_k": 10}) == "921c296be434"


def test_checkpoint_kind_is_part_of_version_2_prediction_identity() -> None:
    a = prediction_hash_from_parts(
        "training", 10, "validation", checkpoint_kind="epoch", identity_version=2
    )
    b = prediction_hash_from_parts(
        "training", 10, "validation", checkpoint_kind="tree_limit", identity_version=2
    )
    assert a != b


def test_legacy_result_reopens_without_being_inferred_complete(tmp_path: Path) -> None:
    study = _study(tmp_path)
    db = _open_registry(study.root)
    try:
        db.execute(
            "INSERT INTO training_runs "
            "(training_hash, family, label, spec_json, created_at) VALUES (?,?,?,?,?)",
            ("legacy-training", "linear", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.execute(
            "INSERT INTO prediction_sets "
            "(prediction_hash, training_hash, split, created_at) VALUES (?,?,?,?)",
            ("legacy-prediction", "legacy-training", "validation", "2024-01-01"),
        )
        db.commit()
    finally:
        db.close()

    reopened = Result.open(study, "legacy-prediction")

    assert reopened.hash == "legacy-prediction"
    assert reopened.identity_version is None
    assert not reopened.complete
    assert reopened.coverage() is None


def test_read_only_legacy_registry_reopens_without_schema_writes(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    db_path = release / "case_studies" / "etfs" / "run_log" / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE training_runs ("
            "training_hash TEXT PRIMARY KEY, family TEXT NOT NULL, label TEXT NOT NULL, "
            "spec_json TEXT, created_at TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE prediction_sets ("
            "prediction_hash TEXT PRIMARY KEY, training_hash TEXT NOT NULL, "
            "split TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?,?)",
            ("legacy-training", "linear", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?,?)",
            ("legacy-prediction", "legacy-training", "validation", "2024-01-01"),
        )
        db.commit()
    before = db_path.read_bytes()

    study = Study.open("etfs", release_root=release)
    reopened = Result.open(study, "legacy-prediction")

    assert reopened.identity_version is None
    assert not reopened.complete
    assert db_path.read_bytes() == before


def test_workspace_open_migrates_copied_legacy_registry(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    db_path = release / "case_studies" / "etfs" / "run_log" / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE training_runs ("
            "training_hash TEXT PRIMARY KEY, family TEXT NOT NULL, label TEXT NOT NULL, "
            "spec_json TEXT, created_at TEXT NOT NULL)"
        )
        db.commit()

    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(training_runs)")}
        coverage_table = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'prediction_coverage'"
        ).fetchone()

    assert {"identity_version", "execution_tier"} <= columns
    assert coverage_table == (1,)


def test_legacy_registry_schema_migrates_additively(tmp_path: Path) -> None:
    case_dir = tmp_path / "legacy" / "etfs"
    db_path = case_dir / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                label TEXT NOT NULL,
                config_name TEXT,
                spec_json TEXT,
                created_at TEXT NOT NULL,
                git_commit TEXT,
                entry_point TEXT
            )
            """
        )
        db.execute(
            "INSERT INTO training_runs "
            "(training_hash, family, label, spec_json, created_at) VALUES (?,?,?,?,?)",
            ("legacy-training", "linear", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.commit()

    migrated = _open_registry(case_dir)
    try:
        columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(training_runs)").fetchall()
        }
        tables = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        row = migrated.execute(
            "SELECT training_hash, identity_version, execution_tier FROM training_runs"
        ).fetchone()
    finally:
        migrated.close()

    assert {"identity_version", "execution_tier"} <= columns
    assert {"prediction_coverage", "candidate_sets", "research_locks"} <= tables
    assert row == ("legacy-training", None, None)


def test_invalid_coverage_registration_is_atomic(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    actual = _predictions().head(1)
    expected = _predictions().select("symbol", "timestamp", "fold_id")

    with pytest.raises(ValueError, match="coverage"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="final",
            checkpoint_value=None,
            split="validation",
            predictions=actual,
            expected_keys=expected,
        )

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM prediction_coverage").fetchone()[0] == 0
    assert not list((study.root / "run_log" / "predictions").glob("*"))


def test_version_2_registration_is_immutable(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec(display_name="first"))
    assert (
        study.results.register_training(_training_spec(display_name="second")).hash == training.hash
    )
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )

    with pytest.raises(ValueError, match="artifact conflict"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="final",
            checkpoint_value=None,
            split="validation",
            predictions=frame.with_columns((pl.col("y_score") + 1).alias("y_score")),
            expected_keys=expected,
        )

    assert prediction.load().get_column("y_score").to_list() == [0.02, -0.01]


def test_checkpoint_prediction_schema_must_match(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=1,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )

    with pytest.raises(ValueError, match="schema"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="epoch",
            checkpoint_value=2,
            split="validation",
            predictions=frame.with_columns(pl.lit("extra").alias("diagnostic")),
            expected_keys=expected,
        )


def test_preview_requires_identity_covered_reductions(tmp_path: Path) -> None:
    study = _study(tmp_path)

    with pytest.raises(ValueError, match="identity-cover"):
        study.results.register_training(
            _training_spec(execution_tier="preview"), execution_tier="preview"
        )

    assert not (study.output_root / ".preview" / "etfs" / "run_log").exists()


def test_candidate_sets_are_immutable_and_validate_protocols(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    first = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    second = study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=2,
        split="validation",
        predictions=frame.with_columns((pl.col("y_score") * 2).alias("y_score")),
        expected_keys=expected,
    )

    original = CandidateSet.create(study, "baseline", [first])
    extended = original.extend("baseline-plus", [second])

    assert original.members == (first.hash,)
    assert extended.members == (first.hash, second.hash)
    assert original.hash != extended.hash
    assert CandidateSet.open(study, original.hash).members == (first.hash,)

    other_training = study.results.register_training(
        _training_spec(cv={"folds": [{"fold": 1, "val_start": "2024-01-06"}]})
    )
    incompatible = study.results.publish_predictions(
        other_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    with pytest.raises(ValueError, match="protocol"):
        CandidateSet.create(study, "invalid", [first, incompatible])

    comparison = {"comparable_fields": ["cv"]}
    ordered = CandidateSet.create(
        study,
        "cv-comparison",
        [first, incompatible],
        comparison_contract=comparison,
    )
    reversed_set = CandidateSet.create(
        study,
        "cv-comparison-reversed",
        [incompatible, first],
        comparison_contract=comparison,
    )
    third = study.results.publish_predictions(
        study.results.register_training(
            _training_spec(cv={"folds": [{"fold": 2, "val_start": "2024-01-07"}]})
        ),
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    extended_comparison = ordered.extend("cv-comparison-plus", [third])

    assert ordered.hash == reversed_set.hash
    assert ordered.comparison_contract["protocol"].get("cv") is None
    assert set(extended_comparison.members) == {first.hash, incompatible.hash, third.hash}


def test_partial_and_preview_results_are_rejected_from_canonical_sets(tmp_path: Path) -> None:
    study = _study(tmp_path)
    canonical_training = study.results.register_training(_training_spec())
    frame = _predictions()
    partial = study.results.publish_predictions(
        canonical_training,
        checkpoint_kind="epoch",
        checkpoint_value=1,
        split="validation",
        predictions=frame.head(1),
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
        allow_partial=True,
    )
    with pytest.raises(ValueError, match="partial"):
        CandidateSet.create(study, "partial", [partial])
    with pytest.raises(ValueError, match="partial"):
        study.strategy(
            prediction=partial,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            execution_mode="vectorized",
        )

    preview_training = study.results.register_training(
        _training_spec(execution_tier="preview", preview_reductions={"folds": [0], "max_rows": 2}),
        execution_tier="preview",
    )
    preview = study.results.publish_predictions(
        preview_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )

    with pytest.raises(ValueError, match="preview"):
        CandidateSet.create(study, "canonical", [preview])
    with pytest.raises(KeyError):
        Result.open(study, preview.hash)
    assert Result.open(study, preview.hash, include_preview=True).hash == preview.hash
