"""Verified reuse of complete training results across identity representation changes."""

from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from case_studies.research import (
    OfficialPopulation,
    PredictionResult,
    Result,
    Study,
    TrainingResult,
)
from case_studies.utils import deep_learning
from case_studies.utils.deep_model_state import deep_checkpoint_path, write_deep_checkpoint
from case_studies.utils.registry import training_hash_from_spec
from case_studies.utils.registry.completeness import skip_training_if_complete
from case_studies.utils.registry.maintenance import migrate_equivalent_training_identity
from case_studies.utils.registry.store import _open_registry
from tests.test_research_registry import _predictions, _study
from tests.test_research_workspace import _seed_release


def _source_spec() -> dict:
    return {
        "identity_version": 3,
        "resolved_spec_schema": "ml4t.resolved-spec/v1",
        "family": "deep_learning",
        "label": "fwd_ret_21d",
        "seed": 42,
        "execution_tier": "canonical",
        "config_name": "nlinear",
        "computation": {
            "checkpoint_schedule": [{"kind": "epoch", "value": 5}],
            "cv": {"folds": [{"fold": 0, "val_start": "2024-01-05"}]},
            "feature_artifacts": {"financial": "features-a"},
            "feature_names": ["momentum", "volatility"],
            "label_artifact": "label-a",
            "model": {
                "class": "nlinear",
                "implementation": "pytorch",
                "params": {"architecture": "nlinear", "width": 32},
            },
            "numerics": {"precision": "float32", "seed": 42},
            "source_identity": {"deep_learning.py": "a" * 64},
        },
        "provenance": {"notebook": "09_dl_nlinear.ipynb"},
    }


def _target_spec(source: dict) -> dict:
    target = copy.deepcopy(source)
    target["computation"]["source_identity"] = {
        "architecture": "nlinear/v1",
        "backend": "pytorch/v1",
        "sequence_preparation": 1,
        "sequence_runner": 1,
        "sequence_state": 1,
    }
    target["provenance"] = {"notebook": "09_dl_nlinear.py", "publication": "current"}
    return target


def _complete_source(study, spec: dict):
    training = study.results.register_training(spec)
    model_dir = training.root / "run_log" / "training" / training.hash / "models"
    checkpoint = deep_checkpoint_path(model_dir, spec["config_name"], 0, 5)
    write_deep_checkpoint(
        checkpoint,
        model=nn.Linear(1, 1),
        architecture="nlinear",
        model_kwargs={"input_dim": 1},
        preprocessing={"mean": [0.0], "scale": [1.0]},
        metadata={
            "checkpoint_kind": "epoch",
            "checkpoint_value": 5,
            "config_name": spec["config_name"],
            "fold": 0,
        },
    )
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=5,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
        metrics={"ic_mean": 0.1},
    )
    assert training.complete and prediction.complete
    return training, prediction


def _registry_rows(db_path: Path) -> dict[str, list[tuple]]:
    with sqlite3.connect(db_path) as db:
        tables = (
            "training_runs",
            "prediction_sets",
            "prediction_coverage",
            "prediction_metrics",
            "fold_metrics",
            "training_identity_migrations",
        )
        return {
            table: db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall() for table in tables
        }


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    run_log = root / "run_log"
    return {
        path.relative_to(run_log).as_posix(): path.read_bytes()
        for path in sorted(run_log.rglob("*"))
        if path.is_file() and path.name != "registry.db" and "registry.db-" not in path.name
    }


def test_equivalent_identity_migration_reuses_complete_training_and_predictions(tmp_path) -> None:
    study = _study(tmp_path)
    source_spec = _source_spec()
    source, source_prediction = _complete_source(study, source_spec)
    target_spec = _target_spec(source_spec)

    migration = migrate_equivalent_training_identity(study, source.hash, target_spec)
    target = Result.open(study, migration.target_training_hash)
    target_prediction = Result.open(study, migration.prediction_map[source_prediction.hash])

    assert isinstance(target, TrainingResult) and target.complete
    assert isinstance(target_prediction, PredictionResult) and target_prediction.complete
    assert target.hash == training_hash_from_spec(target_spec)
    assert target.spec() == target_spec
    assert target_prediction.load().equals(source_prediction.load())
    target_checkpoint = deep_checkpoint_path(
        target.root / "run_log" / "training" / target.hash / "models", "nlinear", 0, 5
    )
    source_checkpoint = deep_checkpoint_path(
        source.root / "run_log" / "training" / source.hash / "models", "nlinear", 0, 5
    )
    assert target_checkpoint.read_bytes() == source_checkpoint.read_bytes()
    assert Result.open(study, source.hash).complete
    assert migration.created
    assert skip_training_if_complete(
        study.case_study,
        target_spec,
        verbose=False,
        case_dir=study.root,
    ).complete
    cached = deep_learning._cached_sequence_run(
        study,
        target_spec,
        SimpleNamespace(
            config={
                "config_name": "nlinear",
                "library": "pytorch",
                "params": {"architecture": "nlinear"},
            },
            prediction_split="validation",
            published_checkpoints=(5,),
            splits=({"fold": 0},),
        ),
    )
    assert cached is not None
    assert cached.training.hash == target.hash
    assert cached.predictions == (target_prediction,)

    population = OfficialPopulation.create(
        study,
        name="equivalent-sequence-results",
        member_kind="prediction",
        members=[target_prediction.hash],
    )
    assert population.require_complete() == (target_prediction.hash,)

    repeated = migrate_equivalent_training_identity(study, source.hash, target_spec)
    assert repeated == migration.with_created(False)

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        proof = db.execute(
            "SELECT proof_json FROM training_identity_migrations WHERE target_training_hash = ?",
            (target.hash,),
        ).fetchone()
    assert proof is not None
    record = json.loads(proof[0])
    assert record["migrated_fields"] == ["computation.source_identity"]
    assert record["source_artifacts"]


def test_semantic_difference_refuses_before_registry_or_artifact_mutation(tmp_path) -> None:
    study = _study(tmp_path)
    source_spec = _source_spec()
    source, _ = _complete_source(study, source_spec)
    target_spec = _target_spec(source_spec)
    target_spec["computation"]["model"]["params"]["width"] = 64
    db_path = study.root / "run_log" / "registry.db"
    before_rows = _registry_rows(db_path)
    before_artifacts = _artifact_bytes(study.root)

    with pytest.raises(ValueError, match="model.params.width"):
        migrate_equivalent_training_identity(study, source.hash, target_spec)

    assert _registry_rows(db_path) == before_rows
    assert _artifact_bytes(study.root) == before_artifacts
    assert not (study.root / "run_log" / "training" / training_hash_from_spec(target_spec)).exists()


def test_version_2_result_remains_readable_after_migration_to_version_3(tmp_path) -> None:
    study = _study(tmp_path)
    version_3 = _source_spec()
    source_spec = {
        "identity_version": 2,
        "family": version_3["family"],
        "label": version_3["label"],
        "seed": version_3["seed"],
        "execution_tier": version_3["execution_tier"],
        "config_name": version_3["config_name"],
        **copy.deepcopy(version_3["computation"]),
    }
    source, _ = _complete_source(study, source_spec)
    target_spec = _target_spec(version_3)

    migration = migrate_equivalent_training_identity(study, source.hash, target_spec)

    assert Result.open(study, source.hash).identity_version == 2
    assert Result.open(study, source.hash).complete
    assert Result.open(study, migration.target_training_hash).identity_version == 3
    assert Result.open(study, migration.target_training_hash).complete


def test_released_result_migration_reconciles_dynamic_metric_columns(tmp_path) -> None:
    release_root = _seed_release(tmp_path)
    release_case = release_root / "case_studies" / "etfs"
    _open_registry(release_case).close()
    source_study = Study(
        case_study="etfs",
        root=release_case,
        release_root=release_root,
        output_root=release_root / "case_studies",
        read_only=False,
        manifest={"schema_version": 1, "case_study": "etfs"},
    )
    source_spec = _source_spec()
    source, source_prediction = _complete_source(source_study, source_spec)
    with sqlite3.connect(release_case / "run_log" / "registry.db") as db:
        db.execute("ALTER TABLE prediction_metrics ADD COLUMN source_only_metric REAL")
        db.execute(
            "UPDATE prediction_metrics SET source_only_metric = 7.5 WHERE prediction_hash = ?",
            (source_prediction.hash,),
        )
        db.commit()

    target_study = Study.open(
        "etfs",
        workspace=tmp_path / "target-workspace",
        release_root=release_root,
    )
    migration = migrate_equivalent_training_identity(
        target_study,
        source.hash,
        _target_spec(source_spec),
    )
    target_prediction_hash = migration.prediction_map[source_prediction.hash]

    with sqlite3.connect(target_study.root / "run_log" / "registry.db") as db:
        row = db.execute(
            "SELECT source_only_metric FROM prediction_metrics WHERE prediction_hash = ?",
            (target_prediction_hash,),
        ).fetchone()
    assert row == (7.5,)
    assert Result.open(target_study, target_prediction_hash).complete


def test_incomplete_source_and_unrecorded_target_are_refused(tmp_path) -> None:
    study = _study(tmp_path)
    source_spec = _source_spec()
    incomplete = study.results.register_training(source_spec)
    target_spec = _target_spec(source_spec)

    with pytest.raises(ValueError, match="complete training result"):
        migrate_equivalent_training_identity(study, incomplete.hash, target_spec)

    complete, _ = _complete_source(study, {**source_spec, "seed": 7})
    occupied_spec = _target_spec({**source_spec, "seed": 7})
    occupied = study.results.register_training(occupied_spec)
    assert not occupied.complete
    with pytest.raises(ValueError, match="already exists without an equivalence record"):
        migrate_equivalent_training_identity(study, complete.hash, occupied_spec)
