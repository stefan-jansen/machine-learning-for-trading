"""Auditable, idempotent maintenance for semantically duplicate backtests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .specs import (
    IDENTITY_VERSION,
    SUPPORTED_IDENTITY_VERSIONS,
    backtest_hash_from_parts,
    canonical_json,
    prediction_hash_from_parts,
    project_training_identity,
    training_hash_from_spec,
)
from .store import _git_hash, _open_registry, _prediction_dir, _save_json, _training_dir, _utc_now

if TYPE_CHECKING:
    from case_studies.research.workspace import Study


@dataclass(frozen=True)
class DuplicateBacktest:
    keep_hash: str
    drop_hashes: tuple[str, ...]


@dataclass(frozen=True)
class TrainingIdentityMigration:
    source_training_hash: str
    target_training_hash: str
    prediction_map: dict[str, str]
    migrated_fields: tuple[str, ...]
    created: bool

    def with_created(self, created: bool) -> TrainingIdentityMigration:
        return replace(self, created=created)


_MIGRATABLE_FIELDS = frozenset({"computation.source_identity"})


def _normalized_training_identity(spec: dict[str, Any]) -> dict[str, Any]:
    projection = project_training_identity(spec)
    common = {
        "family": projection["family"],
        "label": projection["label"],
        "seed": projection["seed"],
        "execution_tier": projection["execution_tier"],
    }
    if spec["identity_version"] == IDENTITY_VERSION:
        computation = projection["computation"]
    else:
        computation = {
            key: value
            for key, value in projection.items()
            if key not in {*common, "identity_version", "resolved_spec_schema"}
        }
    return {**common, "computation": copy.deepcopy(computation)}


def _remove_path(record: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    parent: Any = record
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            return
        parent = parent[part]
    if isinstance(parent, dict):
        parent.pop(parts[-1], None)


def _first_difference(left: Any, right: Any, prefix: str = "") -> str:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                return path
            difference = _first_difference(left[key], right[key], path)
            if difference:
                return difference
        return ""
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{prefix}.length"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_item, right_item, f"{prefix}[{index}]")
            if difference:
                return difference
        return ""
    return "" if left == right else prefix


def _artifact_manifest(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _row(db: sqlite3.Connection, table: str, key: str, value: str) -> dict[str, Any] | None:
    cursor = db.execute(f'SELECT * FROM "{table}" WHERE "{key}" = ?', (value,))
    values = cursor.fetchone()
    if values is None:
        return None
    return dict(zip((column[0] for column in cursor.description), values, strict=True))


def _insert_record(db: sqlite3.Connection, table: str, record: dict[str, Any]) -> None:
    columns = tuple(record)
    placeholders = ",".join("?" for _ in columns)
    column_sql = ",".join(f'"{column}"' for column in columns)
    db.execute(
        f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})',
        tuple(record[column] for column in columns),
    )


def _reconcile_table_schema(
    source_db: sqlite3.Connection,
    target_db: sqlite3.Connection,
    table: str,
) -> None:
    source_columns = source_db.execute(f'PRAGMA table_info("{table}")').fetchall()
    target_names = {row[1] for row in target_db.execute(f'PRAGMA table_info("{table}")').fetchall()}
    for column in source_columns:
        name = str(column[1])
        if name in target_names:
            continue
        declared_type = str(column[2] or "TEXT")
        target_db.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {declared_type}')


def _validate_manifest_files(model_root: Path, manifest: Path) -> None:
    try:
        files = json.loads(manifest.read_text()).get("files")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(f"invalid fitted-state manifest: {manifest}") from exc
    if not isinstance(files, dict) or not files:
        raise ValueError(f"empty fitted-state manifest: {manifest}")
    for relative, expected_digest in files.items():
        artifact = model_root / relative
        if (
            not artifact.is_file()
            or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_digest
        ):
            raise ValueError(f"fitted-state manifest mismatch: {artifact}")


def _validate_fitted_state(source: Any, spec: dict[str, Any]) -> None:
    computation = spec.get("computation", spec)
    model = computation.get("model")
    cv = computation.get("cv")
    schedule = computation.get("checkpoint_schedule")
    if not isinstance(model, dict) or not isinstance(cv, dict) or not isinstance(schedule, list):
        raise ValueError("source training spec does not declare its fitted-state population")
    folds = cv.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("source training spec does not declare any folds")
    fold_ids = tuple(int(fold["fold"]) for fold in folds)
    config_name = spec.get("config_name")
    if not isinstance(config_name, str) or not config_name:
        raise ValueError("source training spec has no fitted-state config name")
    model_root = source.root / "run_log" / "training" / source.hash / "models"
    family = str(spec["family"])
    if family == "deep_learning":
        try:
            checkpoints = tuple(int(item["value"]) for item in schedule)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("deep-learning fitted state requires numeric checkpoints") from exc
        architecture = str(model.get("class"))
        if model.get("implementation") == "darts":
            from case_studies.utils.darts_forecasting import validate_darts_checkpoint_population

            validate_darts_checkpoint_population(
                model_root,
                config_name=config_name,
                fold_ids=fold_ids,
                checkpoints=checkpoints,
                architecture=architecture,
            )
        else:
            from case_studies.utils.deep_model_state import validate_deep_checkpoint_population

            validate_deep_checkpoint_population(
                model_root,
                config_name=config_name,
                fold_ids=fold_ids,
                checkpoints=checkpoints,
                architecture=architecture,
            )
        return
    if family == "tabular_dl":
        manifests = tuple(
            model_root / config_name / f"fold_{fold_id:02d}" / "manifest.json"
            for fold_id in fold_ids
        )
    else:
        manifests = (model_root / "manifest.json",)
    for manifest in manifests:
        _validate_manifest_files(model_root, manifest)


def _clone_prediction_records(
    source_db: sqlite3.Connection,
    target_db: sqlite3.Connection,
    source_hash: str,
    target_hash: str,
    target_training_hash: str,
) -> None:
    prediction = _row(source_db, "prediction_sets", "prediction_hash", source_hash)
    assert prediction is not None
    prediction.update(
        prediction_hash=target_hash,
        training_hash=target_training_hash,
        created_at=_utc_now(),
    )
    _insert_record(target_db, "prediction_sets", prediction)
    for table in ("prediction_coverage", "prediction_metrics"):
        record = _row(source_db, table, "prediction_hash", source_hash)
        if record is not None:
            record["prediction_hash"] = target_hash
            _insert_record(target_db, table, record)
    cursor = source_db.execute(
        "SELECT * FROM fold_metrics WHERE prediction_hash = ?", (source_hash,)
    )
    columns = tuple(column[0] for column in cursor.description)
    for values in cursor.fetchall():
        record = dict(zip(columns, values, strict=True))
        record["prediction_hash"] = target_hash
        _insert_record(target_db, "fold_metrics", record)


def _existing_migration(
    study: Study,
    source_training_hash: str,
    target_training_hash: str,
    target_spec: dict[str, Any],
    migrated_fields: tuple[str, ...],
) -> TrainingIdentityMigration | None:
    target_root = study.storage_root(target_spec["execution_tier"])
    with closing(_open_registry(target_root)) as db:
        record = _row(
            db,
            "training_identity_migrations",
            "target_training_hash",
            target_training_hash,
        )
        target_exists = _row(db, "training_runs", "training_hash", target_training_hash)
    if record is None:
        if target_exists is not None:
            raise ValueError(
                f"target training identity {target_training_hash} already exists without an "
                "equivalence record"
            )
        return None
    if (
        record["source_training_hash"] != source_training_hash
        or json.loads(record["target_spec_json"]) != target_spec
    ):
        raise ValueError(f"target training identity {target_training_hash} has another migration")
    proof = json.loads(record["proof_json"])
    if proof.get("migrated_fields") != list(migrated_fields):
        raise ValueError(f"target training identity {target_training_hash} has another proof scope")
    prediction_map = json.loads(record["prediction_map_json"])
    from case_studies.research.results import PredictionResult, Result, TrainingResult

    training = Result.open(study, target_training_hash, include_preview=True)
    if not isinstance(training, TrainingResult) or not training.complete:
        raise ValueError(f"recorded target training identity {target_training_hash} is incomplete")
    for prediction_hash in prediction_map.values():
        prediction = Result.open(study, prediction_hash, include_preview=True)
        if not isinstance(prediction, PredictionResult) or not prediction.complete:
            raise ValueError(f"recorded target prediction {prediction_hash} is incomplete")
    return TrainingIdentityMigration(
        source_training_hash,
        target_training_hash,
        prediction_map,
        migrated_fields,
        False,
    )


def migrate_equivalent_training_identity(
    study: Study,
    source_training_hash: str,
    target_spec: dict[str, Any],
    *,
    migrated_fields: tuple[str, ...] = ("computation.source_identity",),
) -> TrainingIdentityMigration:
    """Materialize a proven equivalent training identity without fitting again."""
    from case_studies.research.results import PredictionResult, Result, TrainingResult

    study.require_writable()
    if target_spec.get("identity_version") not in SUPPORTED_IDENTITY_VERSIONS:
        raise ValueError("target spec must use a supported identity version")
    if not migrated_fields or not set(migrated_fields) <= _MIGRATABLE_FIELDS:
        raise ValueError(f"unsupported migrated fields: {sorted(set(migrated_fields))}")
    target_training_hash = training_hash_from_spec(target_spec)
    if target_training_hash == source_training_hash:
        raise ValueError("source and target already have the same training identity")

    source = Result.open(study, source_training_hash, include_preview=True)
    if not isinstance(source, TrainingResult) or not source.complete:
        raise ValueError(f"source {source_training_hash} is not a complete training result")
    source_spec = source.spec()
    if source_spec.get("config_name") != target_spec.get("config_name"):
        raise ValueError("training artifact config_name differs between source and target")
    source_identity = _normalized_training_identity(source_spec)
    target_identity = _normalized_training_identity(target_spec)
    for path in migrated_fields:
        _remove_path(source_identity, path)
        _remove_path(target_identity, path)
    difference = _first_difference(source_identity, target_identity)
    if difference:
        raise ValueError(f"training computations differ at {difference}")

    existing = _existing_migration(
        study,
        source_training_hash,
        target_training_hash,
        target_spec,
        migrated_fields,
    )
    if existing is not None:
        return existing

    _validate_fitted_state(source, source_spec)
    source_training_dir = _training_dir(source.root, source_training_hash)
    source_manifest = _artifact_manifest(source_training_dir)
    with closing(sqlite3.connect(source.root / "run_log" / "registry.db")) as source_db:
        source_db.row_factory = sqlite3.Row
        training_record = _row(source_db, "training_runs", "training_hash", source_training_hash)
        assert training_record is not None
        prediction_rows = source_db.execute(
            "SELECT prediction_hash, checkpoint_value, checkpoint_kind, split "
            "FROM prediction_sets WHERE training_hash = ? ORDER BY prediction_hash",
            (source_training_hash,),
        ).fetchall()
        if not prediction_rows:
            raise ValueError(f"source {source_training_hash} has no prediction results")
        prediction_map: dict[str, str] = {}
        prediction_manifests: dict[str, dict[str, str]] = {}
        for prediction_row in prediction_rows:
            source_prediction_hash = str(prediction_row[0])
            prediction = Result.open(study, source_prediction_hash, include_preview=True)
            if not isinstance(prediction, PredictionResult) or not prediction.complete:
                raise ValueError(f"source prediction {source_prediction_hash} is incomplete")
            target_prediction_hash = prediction_hash_from_parts(
                target_training_hash,
                prediction_row[1],
                prediction_row[3],
                checkpoint_kind=prediction_row[2],
                identity_version=target_spec["identity_version"],
            )
            if target_prediction_hash in prediction_map.values():
                raise ValueError("source predictions collapse to one target identity")
            prediction_map[source_prediction_hash] = target_prediction_hash
            prediction_manifests[source_prediction_hash] = _artifact_manifest(
                _prediction_dir(prediction.root, source_prediction_hash)
            )

        target_root = study.storage_root(target_spec["execution_tier"])
        target_training_dir = _training_dir(target_root, target_training_hash)
        target_prediction_dirs = {
            source_hash: _prediction_dir(target_root, target_hash)
            for source_hash, target_hash in prediction_map.items()
        }
        collisions = [
            path
            for path in (target_training_dir, *target_prediction_dirs.values())
            if path.exists()
        ]
        if collisions:
            raise ValueError(f"target artifact paths already exist: {collisions}")
        with closing(_open_registry(target_root)) as target_db:
            if any(
                _row(target_db, "prediction_sets", "prediction_hash", target_hash) is not None
                for target_hash in prediction_map.values()
            ):
                raise ValueError("a target prediction identity already exists")

        token = uuid.uuid4().hex
        staged_training = target_training_dir.with_name(f".{target_training_hash}.{token}.tmp")
        staged_predictions = {
            source_hash: target_dir.with_name(f".{target_dir.name}.{token}.tmp")
            for source_hash, target_dir in target_prediction_dirs.items()
        }
        finalized: list[Path] = []
        try:
            shutil.copytree(source_training_dir, staged_training)
            _save_json(staged_training / "spec.json", target_spec)
            for source_hash, staged in staged_predictions.items():
                shutil.copytree(_prediction_dir(source.root, source_hash), staged)
            staged_source_manifest = dict(source_manifest)
            staged_source_manifest.pop("spec.json", None)
            staged_target_manifest = _artifact_manifest(staged_training)
            staged_target_manifest.pop("spec.json", None)
            if staged_target_manifest != staged_source_manifest:
                raise RuntimeError("staged training artifacts differ from the proven source")
            for source_hash, staged in staged_predictions.items():
                if _artifact_manifest(staged) != prediction_manifests[source_hash]:
                    raise RuntimeError(
                        f"staged prediction artifacts differ from source {source_hash}"
                    )
            target_training_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_training, target_training_dir)
            finalized.append(target_training_dir)
            for source_hash, target_dir in target_prediction_dirs.items():
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_predictions[source_hash], target_dir)
                finalized.append(target_dir)

            with closing(_open_registry(target_root)) as target_db:
                target_db.execute("BEGIN IMMEDIATE")
                for table in (
                    "training_runs",
                    "prediction_sets",
                    "prediction_coverage",
                    "prediction_metrics",
                    "fold_metrics",
                ):
                    _reconcile_table_schema(source_db, target_db, table)
                if (
                    _row(target_db, "training_runs", "training_hash", target_training_hash)
                    is not None
                ):
                    raise ValueError(f"target training identity {target_training_hash} appeared")
                migrated_training = dict(training_record)
                migrated_training.update(
                    training_hash=target_training_hash,
                    family=target_spec["family"],
                    label=target_spec["label"],
                    config_name=target_spec.get("config_name"),
                    spec_json=canonical_json(target_spec),
                    created_at=_utc_now(),
                    git_commit=_git_hash(),
                    entry_point=f"identity-migration:{source_training_hash}",
                    identity_version=target_spec["identity_version"],
                    execution_tier=target_spec["execution_tier"],
                )
                _insert_record(target_db, "training_runs", migrated_training)
                for source_hash, target_hash in prediction_map.items():
                    _clone_prediction_records(
                        source_db,
                        target_db,
                        source_hash,
                        target_hash,
                        target_training_hash,
                    )
                proof = {
                    "migrated_fields": list(migrated_fields),
                    "source_artifacts": source_manifest,
                    "source_identity": project_training_identity(source_spec),
                    "source_predictions": prediction_manifests,
                    "target_identity": project_training_identity(target_spec),
                }
                _insert_record(
                    target_db,
                    "training_identity_migrations",
                    {
                        "target_training_hash": target_training_hash,
                        "source_training_hash": source_training_hash,
                        "target_spec_json": canonical_json(target_spec),
                        "prediction_map_json": canonical_json(prediction_map),
                        "proof_json": canonical_json(proof),
                        "created_at": _utc_now(),
                    },
                )
                target_db.commit()
        except Exception:
            for path in reversed(finalized):
                shutil.rmtree(path, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(staged_training, ignore_errors=True)
            for staged in staged_predictions.values():
                shutil.rmtree(staged, ignore_errors=True)

    return TrainingIdentityMigration(
        source_training_hash,
        target_training_hash,
        prediction_map,
        migrated_fields,
        True,
    )


def _referencing_tables(
    db: sqlite3.Connection, parent: str, parent_key: str
) -> list[tuple[str, str]]:
    """Every (table, column) in the schema that points at ``parent(parent_key)``.

    Read from ``PRAGMA foreign_key_list`` rather than written down, so a table added to the
    schema later is covered without anyone remembering to add it here. A hand-maintained list
    is what produced the defect this exists to close: `cohort_metrics.leader_hash` references
    `backtest_runs`, was not on the list, and with foreign keys enabled its absence turns a
    silent orphan into an IntegrityError that aborts the delete.
    """
    found: list[tuple[str, str]] = []
    tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        if table == parent:
            continue
        for fk in db.execute(f"PRAGMA foreign_key_list({table})"):
            # (id, seq, table, from, to, on_update, on_delete, match)
            if fk[2] == parent and (fk[4] or parent_key) == parent_key:
                found.append((table, fk[3]))
    return found


# References the schema declares in prose but not as a foreign key, so
# `PRAGMA foreign_key_list` cannot find them. Each is here with the reason it carries no key.
#
# `backtest_paired_metrics.benchmark_hash` holds a synthetic benchmark as often as a registered
# one - the equal-weight universe a strategy is compared against is not a backtest_runs row - so
# a foreign key would refuse the common case. `challenger_hash` next to it does have one.
#
# `holdout_evaluations` and `holdout_staging` are keyed on the research lock, and their three
# hash columns record what the lock resolved to rather than pointing into those tables.
#
# The pragma and this list are complementary and neither replaces the other: the pragma covers a
# table added later that declares its key, this covers the ones that deliberately do not, and
# `test_delete_prediction_generation` pins both against `REGISTRY_SCHEMA_SQL` so a new
# unenforced reference fails rather than being silently missed.
_UNENFORCED_BACKTEST_REFERENCES = (
    ("backtest_paired_metrics", "benchmark_hash"),
    ("holdout_evaluations", "holdout_backtest_hash"),
    ("holdout_staging", "holdout_backtest_hash"),
)
_UNENFORCED_PREDICTION_REFERENCES = (
    ("holdout_evaluations", "holdout_prediction_hash"),
    ("holdout_staging", "holdout_prediction_hash"),
)


def _existing(db: sqlite3.Connection, refs: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return [(table, column) for table, column in refs if table in tables]


def delete_prediction_generation(db_path: Path, prediction_hash: str) -> dict[str, int]:
    """Delete one prediction set, every backtest on it, and every row that references either.

    Returns the row count deleted per table, so a caller can say what it removed rather than
    asserting it worked.

    The child tables are derived from the schema, not listed. Deleting the two parent rows and
    leaving the children is worse than leaving the generation in place: the parent is gone, so
    nothing reports the orphan, while a query joining from the child side still finds it.

    Foreign keys are enabled on this connection. SQLite leaves them off per connection, so the
    schema's REFERENCES clauses enforce nothing by default and a missed child table succeeds
    silently instead of raising - which is exactly how the first version of this shipped.
    """
    counts: dict[str, int] = {}
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute("PRAGMA foreign_keys = ON")
        backtests = [
            row[0]
            for row in db.execute(
                "SELECT backtest_hash FROM backtest_runs WHERE prediction_hash = ?",
                (prediction_hash,),
            )
        ]
        backtest_refs = _referencing_tables(db, "backtest_runs", "backtest_hash") + _existing(
            db, _UNENFORCED_BACKTEST_REFERENCES
        )
        for table, column in backtest_refs:
            for backtest_hash in backtests:
                cur = db.execute(f"DELETE FROM {table} WHERE {column} = ?", (backtest_hash,))
                counts[table] = counts.get(table, 0) + cur.rowcount
        cur = db.execute("DELETE FROM backtest_runs WHERE prediction_hash = ?", (prediction_hash,))
        counts["backtest_runs"] = cur.rowcount
        prediction_refs = _referencing_tables(db, "prediction_sets", "prediction_hash") + _existing(
            db, _UNENFORCED_PREDICTION_REFERENCES
        )
        for table, column in prediction_refs:
            cur = db.execute(f"DELETE FROM {table} WHERE {column} = ?", (prediction_hash,))
            counts[table] = counts.get(table, 0) + cur.rowcount
        cur = db.execute(
            "DELETE FROM prediction_sets WHERE prediction_hash = ?", (prediction_hash,)
        )
        counts["prediction_sets"] = cur.rowcount
        db.commit()

    run_log = db_path.parent
    for directory in [run_log / "predictions" / prediction_hash] + [
        run_log / "backtest" / backtest_hash for backtest_hash in backtests
    ]:
        if directory.is_dir():
            shutil.rmtree(directory)

    return {table: n for table, n in counts.items() if n}


def find_semantic_backtest_duplicates(db_path: Path) -> list[DuplicateBacktest]:
    """Find rows differing only by normalized, identity-neutral spec defaults."""
    with closing(sqlite3.connect(str(db_path))) as db:
        rows = db.execute(
            "SELECT backtest_hash, prediction_hash, stage, spec_json FROM backtest_runs"
        ).fetchall()

    groups: dict[tuple[str, str | None, str], list[str]] = {}
    for stored_hash, prediction_hash, stage, spec_json in rows:
        if not spec_json:
            continue
        spec = json.loads(spec_json)
        semantic_hash = backtest_hash_from_parts(prediction_hash, spec)
        groups.setdefault((prediction_hash, stage, semantic_hash), []).append(stored_hash)

    duplicates = []
    for hashes in groups.values():
        if len(hashes) < 2:
            continue
        ordered = sorted(hashes)
        duplicates.append(DuplicateBacktest(ordered[0], tuple(ordered[1:])))
    return sorted(duplicates, key=lambda item: item.keep_hash)


def deduplicate_semantic_backtests(
    db_path: Path, *, apply: bool = False
) -> list[DuplicateBacktest]:
    """Remove unreferenced semantic duplicates; dry-run unless ``apply`` is true."""
    duplicates = find_semantic_backtest_duplicates(db_path)
    if not apply or not duplicates:
        return duplicates

    drops = [item for group in duplicates for item in group.drop_hashes]
    with closing(sqlite3.connect(str(db_path))) as db, db:
        db.execute("PRAGMA foreign_keys = ON")
        placeholders = ",".join("?" for _ in drops)
        references = {
            "backtest_paired_metrics.challenger_hash": db.execute(
                f"SELECT challenger_hash FROM backtest_paired_metrics WHERE challenger_hash IN ({placeholders})",
                drops,
            ).fetchall(),
            "backtest_paired_metrics.benchmark_hash": db.execute(
                f"SELECT benchmark_hash FROM backtest_paired_metrics WHERE benchmark_hash IN ({placeholders})",
                drops,
            ).fetchall(),
            "cohort_metrics.leader_hash": db.execute(
                f"SELECT leader_hash FROM cohort_metrics WHERE leader_hash IN ({placeholders})",
                drops,
            ).fetchall(),
        }
        live_refs = {name: rows for name, rows in references.items() if rows}
        if live_refs:
            raise RuntimeError(f"Refusing to delete referenced backtests: {live_refs}")

        db.execute(
            f"DELETE FROM backtest_fold_metrics WHERE backtest_hash IN ({placeholders})", drops
        )
        db.execute(f"DELETE FROM backtest_metrics WHERE backtest_hash IN ({placeholders})", drops)
        db.execute(f"DELETE FROM backtest_runs WHERE backtest_hash IN ({placeholders})", drops)
    return duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    groups = deduplicate_semantic_backtests(args.registry, apply=args.apply)
    mode = "applied" if args.apply else "dry-run"
    print(f"{mode}: {sum(len(group.drop_hashes) for group in groups)} duplicate rows")
    for group in groups:
        print(f"keep={group.keep_hash} drop={','.join(group.drop_hashes)}")


if __name__ == "__main__":
    main()
