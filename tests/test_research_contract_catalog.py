from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import EligibilityManifest, ResolvedSpec, Study
from case_studies.utils.registry import register_prediction_set, register_training_run
from tests.test_research_registry import _predictions
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolved_spec(*, alpha: float = 1.0, host: str = "host-a") -> dict:
    return ResolvedSpec.create(
        family="linear",
        label="fwd_ret_21d",
        seed=42,
        computation={
            "implementation": {"resolver": "linear-v3", "source": "source-a"},
            "model": {"class": "Ridge", "params": {"alpha": alpha, "fit_intercept": True}},
            "backend": {"library": "scikit-learn", "version": "1.7", "precision": "float64"},
            "cv": {"identity": "cv-a", "request": {"folds": [0]}},
        },
        provenance={"host": host, "device_name": "cpu-a", "elapsed_s": 1.0},
        config_name="ridge",
    ).as_dict()


def _publish(
    case_dir: Path,
    *,
    spec: dict,
    score_shift: float = 0.0,
) -> str:
    training_hash = register_training_run("etfs", spec, case_dir=case_dir)
    frame = _predictions().with_columns((pl.col("y_score") + score_shift).alias("y_score"))
    return register_prediction_set(
        "etfs",
        training_hash,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
        case_dir=case_dir,
    )


def test_workspace_overlay_restarts_without_copying_release_run_log(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    release_case = release / "case_studies" / "etfs"
    released_hash = _publish(release_case, spec=_resolved_spec())
    release_digest = _tree_digest(release_case)

    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    workspace_registry = study.root / "run_log" / "registry.db"
    with sqlite3.connect(workspace_registry) as db:
        assert db.execute("SELECT COUNT(*) FROM training_runs").fetchone() == (0,)
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone() == (0,)
    assert not (study.root / "run_log" / "predictions" / released_hash).exists()

    workspace_spec = _resolved_spec(alpha=2.0)
    workspace_prediction = _publish(study.root, spec=workspace_spec, score_shift=0.1)
    reopened = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    catalog = reopened.predictions.table()

    assert set(catalog.get_column("origin")) == {"released", "workspace"}
    assert set(catalog.get_column("prediction_hash")) == {released_hash, workspace_prediction}
    assert _tree_digest(release_case) == release_digest


def test_workspace_overlay_prefers_workspace_row_for_same_identity(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    release_case = release / "case_studies" / "etfs"
    prediction_hash = _publish(release_case, spec=_resolved_spec())
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    with (
        sqlite3.connect(study.root / "run_log" / "registry.db") as workspace_db,
        sqlite3.connect(release_case / "run_log" / "registry.db") as release_db,
    ):
        training = release_db.execute(
            "SELECT * FROM training_runs WHERE training_hash = ("
            "SELECT training_hash FROM prediction_sets WHERE prediction_hash = ?)",
            (prediction_hash,),
        ).fetchone()
        training_columns = [
            row[1] for row in release_db.execute("PRAGMA table_info(training_runs)")
        ]
        workspace_db.execute(
            f"INSERT INTO training_runs ({', '.join(training_columns)}) "
            f"VALUES ({', '.join('?' for _ in training_columns)})",
            training,
        )
        prediction = release_db.execute(
            "SELECT * FROM prediction_sets WHERE prediction_hash = ?", (prediction_hash,)
        ).fetchone()
        prediction_columns = [
            row[1] for row in release_db.execute("PRAGMA table_info(prediction_sets)")
        ]
        workspace_db.execute(
            f"INSERT INTO prediction_sets ({', '.join(prediction_columns)}) "
            f"VALUES ({', '.join('?' for _ in prediction_columns)})",
            prediction,
        )
        coverage = release_db.execute(
            "SELECT * FROM prediction_coverage WHERE prediction_hash = ?", (prediction_hash,)
        ).fetchone()
        coverage_columns = [
            row[1] for row in release_db.execute("PRAGMA table_info(prediction_coverage)")
        ]
        workspace_db.execute(
            f"INSERT INTO prediction_coverage ({', '.join(coverage_columns)}) "
            f"VALUES ({', '.join('?' for _ in coverage_columns)})",
            coverage,
        )
        workspace_db.commit()

    catalog = study.predictions.table()

    assert catalog.filter(pl.col("prediction_hash") == prediction_hash).height == 1
    assert (
        catalog.filter(pl.col("prediction_hash") == prediction_hash).item(0, "origin")
        == "workspace"
    )


def test_two_workspaces_keep_catalogs_and_config_roots_isolated(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    first = Study.open("etfs", workspace=tmp_path / "first", release_root=release)
    first_hash = _publish(first.root, spec=_resolved_spec(alpha=1.0))
    second = Study.open("etfs", workspace=tmp_path / "second", release_root=release)
    second_hash = _publish(second.root, spec=_resolved_spec(alpha=2.0))

    assert first_hash in first.predictions.table().get_column("prediction_hash").to_list()
    assert second_hash not in first.predictions.table().get_column("prediction_hash").to_list()
    assert second_hash in second.predictions.table().get_column("prediction_hash").to_list()
    assert first_hash not in second.predictions.table().get_column("prediction_hash").to_list()


def test_resolved_identity_is_stable_to_provenance_and_sensitive_to_computation() -> None:
    first = ResolvedSpec.from_dict(_resolved_spec(host="host-a"))
    second = ResolvedSpec.from_dict(_resolved_spec(host="host-b"))
    changed = ResolvedSpec.from_dict(_resolved_spec(alpha=2.0, host="host-a"))

    assert first.identity == second.identity
    assert first.identity != changed.identity
    assert first.computation["model"]["params"]["fit_intercept"] is True


@pytest.mark.parametrize(
    "unsupported",
    [{"bad": {1, 2}}, {"bad": float("nan")}, {"bad": {1: "non-string key"}}],
)
def test_resolved_identity_rejects_non_round_trip_parameters(unsupported: dict) -> None:
    with pytest.raises((TypeError, ValueError), match="canonical|finite|string keys"):
        ResolvedSpec.create(
            family="linear",
            label="fwd_ret_21d",
            seed=42,
            computation={"model": {"params": unsupported}},
            provenance={},
        )


def test_eligibility_manifest_records_schema_sources_diagnostics_and_sorted_digest() -> None:
    keys = pl.DataFrame(
        {
            "symbol": ["B", "A", "A"],
            "timestamp": ["2024-01-04", "2024-01-03", "2024-01-04"],
            "fold": [1, 0, 1],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    reordered = keys.reverse()

    first = EligibilityManifest.resolve(
        keys,
        entity_columns=("symbol",),
        source_identity={"labels": "label-a", "features": "features-a"},
        logic_identity={"implementation": "eligibility-v1"},
        diagnostics_by_fold={0: {"dropped": 2}, 1: {"dropped": 1}},
    )
    second = EligibilityManifest.resolve(
        reordered,
        entity_columns=("symbol",),
        source_identity={"features": "features-a", "labels": "label-a"},
        logic_identity={"implementation": "eligibility-v1"},
        diagnostics_by_fold={1: {"dropped": 1}, 0: {"dropped": 2}},
    )

    assert first == second
    assert first.entity_schema == {
        "entity_columns": ["symbol"],
        "timestamp": "timestamp",
        "fold": "fold",
        "dtypes": {"symbol": "String", "timestamp": "Date", "fold": "Int64"},
    }
    assert first.n_eligible == 3
    assert [fold["n_eligible"] for fold in first.folds] == [1, 2]


def test_catalog_schema_and_open_parameter_types_are_stable(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    _publish(study.root, spec=_resolved_spec(alpha=1.0))
    _publish(study.root, spec=_resolved_spec(alpha=2))

    first = study.predictions.table()
    second = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=release
    ).predictions.table()

    required = {
        "catalog_version",
        "origin",
        "family",
        "config_name",
        "label",
        "task",
        "split",
        "checkpoint_kind",
        "checkpoint_value",
        "cv_identity",
        "execution_tier",
        "approval",
        "complete",
        "artifact_available",
        "ic_mean",
        "diagnostic_metrics_json",
        "provenance_json",
        "training_hash",
        "prediction_hash",
        "spec_json",
        "model__params__alpha",
    }
    assert required <= set(first.columns)
    assert first.schema == second.schema
    assert first.schema["model__params__alpha"] == pl.Float64
    assert first.schema["ic_mean"] == pl.Float64


def test_catalog_one_matches_null_checkpoint_and_projects_resolved_fold_parameters(
    tmp_path: Path,
) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    spec = _resolved_spec()
    spec["computation"]["model"] = {
        "class": "Ridge",
        "effective_params_by_fold": {"0": {"alpha": 1.5, "fit_intercept": True}},
    }
    prediction_hash = _publish(study.root, spec=spec)

    row = study.predictions.one(prediction_hash=prediction_hash, checkpoint_value=None)

    assert row["model__effective_params_by_fold__0__alpha"] == 1.5


def test_catalog_reads_complete_v2_artifacts_without_granting_current_completeness(
    tmp_path: Path,
) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    legacy_spec = {
        "identity_version": 2,
        "execution_tier": "canonical",
        "family": "linear",
        "label": "fwd_ret_21d",
        "seed": 42,
        "config_name": "ridge",
        "model": {"class": "Ridge", "params": {"alpha": 1.0}},
    }
    prediction_hash = _publish(study.root, spec=legacy_spec)

    row = (
        study.predictions.table()
        .filter(pl.col("prediction_hash") == prediction_hash)
        .row(0, named=True)
    )

    assert row["artifact_available"] is True
    assert row["identity_status"] == "legacy-v2"
    assert row["complete"] is False


def test_catalog_reads_legacy_rows_without_claiming_current_contract(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    case_dir = release / "case_studies" / "etfs"
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(
            "CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, family TEXT NOT NULL, "
            "label TEXT NOT NULL, spec_json TEXT, created_at TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, training_hash TEXT "
            "NOT NULL, split TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?,?)",
            ("legacy-training", "linear", "label", json.dumps({"family": "linear"}), "now"),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?,?)",
            ("legacy-prediction", "legacy-training", "validation", "now"),
        )
        db.commit()

    row = Study.open("etfs", release_root=release).predictions.table().row(0, named=True)

    assert row["origin"] == "released"
    assert row["prediction_hash"] == "legacy-prediction"
    assert row["identity_status"] == "legacy"
    assert row["complete"] is False


def test_catalog_says_which_sibling_a_regression_auc_was_scored_against(tmp_path: Path) -> None:
    """A regression row's AUC is scored against a declared direction label, not its own classes.

    Without ``direction_label`` the column reads as the classification path's AUC on a row that
    has no classes, and the two are not the same quantity.
    """
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    scored = _publish(study.root, spec=_resolved_spec(alpha=1.0))
    unscored = _publish(study.root, spec=_resolved_spec(alpha=2))

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        # The metric writer adds this column the first time a regression AUC is recorded, the
        # same way it adds any other metric name it has not seen.
        db.execute("ALTER TABLE prediction_metrics ADD COLUMN direction_label TEXT")
        for prediction_hash, direction in ((scored, "fwd_dir_21d"), (unscored, None)):
            db.execute(
                "UPDATE prediction_metrics SET auc_mean_daily = 0.53, direction_label = ? "
                "WHERE prediction_hash = ?",
                (direction, prediction_hash),
            )

    rows = {row["prediction_hash"]: row for row in study.predictions.table().iter_rows(named=True)}
    assert rows[scored]["direction_label"] == "fwd_dir_21d"
    assert rows[unscored]["direction_label"] is None
    assert study.predictions.table().schema["direction_label"] == pl.String
