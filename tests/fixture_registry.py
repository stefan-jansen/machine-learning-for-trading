"""Keep a fixture's ``run_log`` describing what the fixture actually ships.

The CI fixture registries were copied whole from production while the artifacts beside
them were subsampled, so they pin files the fixture has never held. This module holds the
operations that put a fixture registry back into agreement with its own tree; the callers
are :mod:`tests.generate_intermediates`, which prunes before it registers, and
``tests/reconcile_fixture_registry.py``, which reconciles a fixture already committed.

Imported by ``generate_intermediates.py``, which runs standalone without pytest, so
nothing here may import from ``conftest`` or from pytest.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

#: The directories under a case study whose parquets a training run pins by whole-file
#: sha256. ``utils.modeling`` builds ``input_data_spec.artifacts`` from exactly these:
#: ``financial`` and ``model_based`` from ``features/``, ``label`` and ``eval_label``
#: from ``labels/``.
PINNED_ARTIFACT_DIRS = ("features", "labels")

#: Every table that hangs off a training run, in the order they must be deleted so no
#: statement removes a row another still needs to find its own. Each entry is
#: ``(table, column, key)`` where *key* names which set of hashes the column is matched
#: against: ``training``, ``prediction`` or ``backtest``.
_CASCADE: tuple[tuple[str, str, str], ...] = (
    ("backtest_paired_metrics", "challenger_hash", "backtest"),
    ("backtest_paired_metrics", "benchmark_hash", "backtest"),
    ("cohort_metrics", "leader_hash", "backtest"),
    ("backtest_fold_metrics", "backtest_hash", "backtest"),
    ("backtest_metrics", "backtest_hash", "backtest"),
    ("backtest_runs", "backtest_hash", "backtest"),
    ("fold_metrics", "prediction_hash", "prediction"),
    ("prediction_metrics", "prediction_hash", "prediction"),
    ("prediction_coverage", "prediction_hash", "prediction"),
    ("official_population_members", "member_hash", "prediction"),
    ("candidate_set_members", "member_hash", "prediction"),
    ("holdout_staging", "holdout_prediction_hash", "prediction"),
    ("holdout_evaluations", "holdout_prediction_hash", "prediction"),
    ("prediction_sets", "prediction_hash", "prediction"),
    ("candidate_fold_completions", "training_hash", "training"),
    ("holdout_staging", "holdout_training_hash", "training"),
    ("holdout_evaluations", "holdout_training_hash", "training"),
    ("training_runs", "training_hash", "training"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shipped_artifact_shas(case_dir: Path) -> dict[str, str]:
    """Every sha256 the fixture ships under this case study, to the file carrying it.

    Keyed by digest rather than by artifact name on purpose. A training run pins
    ``label`` to ``labels/<its own label>.parquet`` and ``financial`` to
    ``features/financial.parquet``, so resolving a pin by name means reproducing that
    mapping here and going stale the day it changes. The question this module asks is
    narrower and name-free: does the fixture hold the bytes this run was fitted on.
    """
    shas: dict[str, str] = {}
    for subdir in PINNED_ARTIFACT_DIRS:
        for parquet in sorted((case_dir / subdir).glob("*.parquet")):
            shas[sha256_file(parquet)] = f"{subdir}/{parquet.name}"
    return shas


def pinned_input_shas(spec_json: str | dict | None) -> dict[str, str]:
    """The ``{artifact name: sha256}`` a training run's spec pins, hex and unprefixed.

    Mirrors ``case_studies/utils/registry/registration.py::_input_artifact_shas``, which
    is what the vintage guard compares against, and tolerates the same spec shapes it
    does: a spec that carries no ``computation.input_data_spec.artifacts`` block pins
    nothing and is never stale.
    """
    spec = spec_json
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except (TypeError, ValueError):
            return {}
    if not isinstance(spec, dict):
        return {}
    computation = spec.get("computation")
    if not isinstance(computation, dict):
        return {}
    input_data_spec = computation.get("input_data_spec")
    if not isinstance(input_data_spec, dict):
        return {}
    artifacts = input_data_spec.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}
    return {
        str(name): str(record["sha256"]).removeprefix("sha256:")
        for name, record in sorted(artifacts.items())
        if isinstance(record, dict) and record.get("sha256")
    }


def stale_training_runs(db: sqlite3.Connection, case_dir: Path) -> dict[str, dict[str, str]]:
    """Training runs fitted on input artifacts this fixture does not ship.

    Returns ``{training_hash: {artifact name: pinned sha}}`` for the pins that are
    absent, so a caller can say which file each row wanted rather than only how many
    rows it dropped.
    """
    shipped = set(shipped_artifact_shas(case_dir))
    stale: dict[str, dict[str, str]] = {}
    for training_hash, spec_json in db.execute(
        "SELECT training_hash, spec_json FROM training_runs"
    ):
        absent = {
            name: sha for name, sha in pinned_input_shas(spec_json).items() if sha not in shipped
        }
        if absent:
            stale[str(training_hash)] = absent
    return stale


def _existing_tables(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _in_batches(values: Iterable[str], size: int = 500):
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def delete_training_runs(db: sqlite3.Connection, training_hashes: set[str]) -> dict[str, int]:
    """Remove these training runs and everything that hangs off them.

    SQLite does not enforce the schema's foreign keys unless the connection asks it to,
    so deleting ``training_runs`` alone leaves prediction sets, metrics and backtests
    pointing at a row that is gone - which reads downstream as a corrupt registry rather
    than a pruned one. The cascade is spelled out in :data:`_CASCADE` and applied leaf
    first.
    """
    if not training_hashes:
        return {}
    tables = _existing_tables(db)
    prediction_hashes: set[str] = set()
    backtest_hashes: set[str] = set()
    for batch in _in_batches(training_hashes):
        marks = ",".join("?" * len(batch))
        prediction_hashes.update(
            str(row[0])
            for row in db.execute(
                f"SELECT prediction_hash FROM prediction_sets WHERE training_hash IN ({marks})",
                batch,
            )
        )
    if "backtest_runs" in tables:
        for batch in _in_batches(prediction_hashes):
            marks = ",".join("?" * len(batch))
            backtest_hashes.update(
                str(row[0])
                for row in db.execute(
                    f"SELECT backtest_hash FROM backtest_runs WHERE prediction_hash IN ({marks})",
                    batch,
                )
            )
    keys = {
        "training": training_hashes,
        "prediction": prediction_hashes,
        "backtest": backtest_hashes,
    }
    deleted: dict[str, int] = {}
    for table, column, key in _CASCADE:
        if table not in tables:
            continue
        hashes = keys[key]
        if not hashes:
            continue
        for batch in _in_batches(hashes):
            marks = ",".join("?" * len(batch))
            cursor = db.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", batch)
            deleted[table] = deleted.get(table, 0) + cursor.rowcount
    db.commit()
    return {table: count for table, count in deleted.items() if count}


def prune_stale_training_runs(case_dir: Path) -> dict:
    """Drop this fixture's training runs that were fitted on artifacts it no longer has.

    Called by the fixture generator immediately before its first registering stage. By
    then stages 01-05 have rewritten ``features/`` and ``labels/``, so a row pinning an
    older vintage is describing a population the fixture no longer holds - and the
    vintage guard in ``register_training_run`` refuses to let the run about to start
    join it (ml4t/agent-workspace#1082). Declaring an artifact supersession is the wrong
    override: it would record that the fixture's reduced file deliberately replaces the
    production one inside one continuing population, and the two were never in the same
    population.

    Returns a summary with the rows dropped per table and one example of what was
    pinned, or ``{}`` when the case study has no registry yet.
    """
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.is_file():
        return {}
    db = sqlite3.connect(str(db_path))
    try:
        stale = stale_training_runs(db, case_dir)
        if not stale:
            return {"training_runs_pruned": 0}
        deleted = delete_training_runs(db, set(stale))
    finally:
        db.close()
    example_hash = sorted(stale)[0]
    return {
        "training_runs_pruned": len(stale),
        "deleted": deleted,
        "example": {"training_hash": example_hash, "absent_pins": stale[example_hash]},
    }
