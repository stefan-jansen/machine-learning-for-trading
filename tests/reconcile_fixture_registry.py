#!/usr/bin/env python3
"""Bring a committed fixture registry back into agreement with the artifacts beside it.

The nine registries under ``ml4t/third-edition-test-data`` ``intermediates/*/run_log/``
were copied whole from production while the artifacts beside them were subsampled. Two
consequences are repairable in place, without regenerating the fixture:

* **Metrics that describe a different file.** A shipped ``predictions.parquet`` is a
  subsample; its ``prediction_metrics`` row is production's. ``crypto_perps_funding``
  ``f6bd7cd2d208`` registers 2104 IC days against a parquet holding 66, and
  ``08_signal_method_comparison::validate_prediction_set`` rejects the pair - correctly.
  Recomputing the row from the shipped parquet makes the fixture internally consistent
  at whatever scale it ships (ml4t/agent-workspace#286).

* **Artifact directories nothing can reach.** A ``run_log/predictions/<hash>/`` with no
  ``prediction_sets`` row is addressable by nothing: every reader resolves a hash out of
  the registry. These are what a re-sample leaves behind when it rewrites the registry
  after a generation wrote artifacts (ml4t/agent-workspace#1081).

What it deliberately does not do is invent rows for the 97.7% of registered predictions
whose artifact the fixture has never held. Those rows are the replay surface the
chapter notebooks and the ``research_preview: false`` case-study stages rank over, and
seeding their artifacts is 88 MB per panel. ``--report`` prints how many there are so
the gap is visible rather than silent.

Usage::

    uv run python tests/reconcile_fixture_registry.py --report
    uv run python tests/reconcile_fixture_registry.py --apply
    uv run python tests/reconcile_fixture_registry.py --apply --case-study etfs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tests.fixture_registry import PINNED_ARTIFACT_DIRS
except ModuleNotFoundError:  # standalone, as generate_intermediates.py also runs
    from fixture_registry import PINNED_ARTIFACT_DIRS

DEFAULT_INTERMEDIATES = Path.home() / "ml4t" / "test-data" / "intermediates"

#: The column each role takes in a shipped artifact. Both conventions are in the tree:
#: ``tests/conftest.py::_migrate_predictions_schema`` renames the legacy trio at seed
#: time, so the fixture ships either and a reconciliation that knew only one would skip
#: every legacy artifact silently.
_COLUMNS = {
    "score": ("prediction", "y_score"),
    "target": ("actual", "y_true"),
    "fold": ("fold", "fold_id"),
    "date": ("timestamp", "date"),
    "entity": ("symbol", "product"),
}


def _resolve(columns: list[str], role: str) -> str | None:
    return next((name for name in _COLUMNS[role] if name in columns), None)


def _spec_context(spec_json: str | None) -> dict:
    """Task type, label buffer and eval column, read off the training run's own spec."""
    try:
        spec = json.loads(spec_json or "{}")
    except (TypeError, ValueError):
        return {}
    computation = spec.get("computation") or {}
    input_spec = computation.get("input_data_spec") or {}
    task = computation.get("task") or {}
    return {
        "label": spec.get("label"),
        "task_type": input_spec.get("task_type") or "regression",
        "label_buffer": input_spec.get("label_buffer"),
        "eval_label_col": input_spec.get("eval_label_col"),
        "class_values": task.get("class_values"),
    }


def prediction_dirs(case_dir: Path) -> list[str]:
    """Every ``run_log/predictions/<hash>/`` the fixture ships, parquet or not.

    Empty ones count. ``sp500_equity_option_analytics`` ships 39 directories of which 30
    hold no file at all - the residue of a regeneration whose registry rows a later
    re-sample overwrote - and a check that only looked at directories holding a parquet
    would call the tree clean while 30 dead directories sat in it.
    """
    root = case_dir / "run_log" / "predictions"
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def shipped_predictions(case_dir: Path) -> dict[str, Path]:
    """``{prediction_hash: parquet}`` for every prediction artifact this fixture ships."""
    root = case_dir / "run_log" / "predictions"
    return {
        name: root / name / "predictions.parquet"
        for name in prediction_dirs(case_dir)
        if (root / name / "predictions.parquet").is_file()
    }


def recompute_metrics(parquet: Path, context: dict) -> tuple[dict, dict[int, dict]]:
    """Headline and per-fold metrics as the *shipped* parquet determines them.

    Calls the same ``compute_prediction_fold_metrics`` a registration calls, so the
    reconciled row is produced by the code that writes a real one rather than by a second
    implementation of the same statistics that could drift from it.
    """
    import polars as pl

    from case_studies.utils.registry.metrics import compute_prediction_fold_metrics

    frame = pl.read_parquet(parquet)
    columns = frame.columns
    score = _resolve(columns, "score")
    target = _resolve(columns, "target")
    fold = _resolve(columns, "fold")
    date = _resolve(columns, "date")
    entity = _resolve(columns, "entity")
    if not (score and target and fold and date and entity):
        raise ValueError(
            f"{parquet} carries {columns}, which does not resolve a score, target, fold, "
            f"date and entity column; its metrics cannot be recomputed from it."
        )
    eval_col = "eval_actual" if "eval_actual" in columns else None
    task_type = str(context.get("task_type") or "regression")
    if task_type == "classification" and eval_col is None:
        # IC against a binary label collapses to 2*(AUC-0.5) and is not a rank
        # correlation against returns, which is why compute_prediction_fold_metrics
        # requires the continuous column. Without it in the artifact there is nothing to
        # recompute honestly, so say so rather than register a different statistic under
        # the same column name.
        raise ValueError(
            f"{parquet} is a classification prediction set with no 'eval_actual' column, "
            f"so its IC cannot be recomputed from the shipped artifact."
        )
    return compute_prediction_fold_metrics(
        frame,
        y_true_col=target,
        y_score_col=score,
        fold_col=fold,
        date_col=date,
        entity_col=entity,
        task_type=task_type,
        class_values=context.get("class_values"),
        eval_col=eval_col,
        label=context.get("label"),
        label_buffer=context.get("label_buffer"),
    )


def _write_metrics(
    db: sqlite3.Connection, prediction_hash: str, headline: dict, folds: dict
) -> None:
    columns = {row[1] for row in db.execute("PRAGMA table_info(prediction_metrics)")}
    payload = {
        name: value
        for name, value in headline.items()
        if name in columns and name != "prediction_hash"
    }
    payload["computed_at"] = datetime.now(UTC).isoformat()
    names = ["prediction_hash", *sorted(payload)]
    marks = ",".join("?" * len(names))
    quoted = ",".join(f'"{name}"' for name in names)
    db.execute("DELETE FROM prediction_metrics WHERE prediction_hash = ?", (prediction_hash,))
    db.execute(
        f"INSERT INTO prediction_metrics ({quoted}) VALUES ({marks})",
        [prediction_hash, *(payload[name] for name in sorted(payload))],
    )
    fold_columns = {row[1] for row in db.execute("PRAGMA table_info(fold_metrics)")}
    db.execute("DELETE FROM fold_metrics WHERE prediction_hash = ?", (prediction_hash,))
    for fold_id, metrics in sorted(folds.items()):
        row = {name: value for name, value in metrics.items() if name in fold_columns}
        row["prediction_hash"] = prediction_hash
        row["fold_id"] = int(fold_id)
        row["computed_at"] = payload["computed_at"]
        names = sorted(row)
        marks = ",".join("?" * len(names))
        quoted = ",".join(f'"{name}"' for name in names)
        db.execute(
            f"INSERT INTO fold_metrics ({quoted}) VALUES ({marks})", [row[name] for name in names]
        )


def reconcile(case_dir: Path, *, apply: bool) -> dict:
    """Report, and optionally repair, one case study's registry against its own tree."""
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.is_file():
        return {"status": "no registry"}
    shipped = shipped_predictions(case_dir)
    # Counted before anything is removed. Reporting the tree after the deletions
    # describes the repair rather than what was found, which is the one thing a
    # --report run exists to say.
    dirs = prediction_dirs(case_dir)
    db = sqlite3.connect(str(db_path))
    try:
        registered = {row[0] for row in db.execute("SELECT prediction_hash FROM prediction_sets")}
        orphans = sorted(set(dirs) - registered)
        unbacked = len(registered - set(shipped))
        recomputed, failed = [], {}
        for prediction_hash in sorted(set(shipped) & registered):
            row = db.execute(
                "SELECT tr.spec_json FROM prediction_sets ps "
                "JOIN training_runs tr ON tr.training_hash = ps.training_hash "
                "WHERE ps.prediction_hash = ?",
                (prediction_hash,),
            ).fetchone()
            context = _spec_context(row[0] if row else None)
            try:
                headline, folds = recompute_metrics(shipped[prediction_hash], context)
            except (ValueError, KeyError) as exc:
                failed[prediction_hash] = str(exc)
                continue
            recomputed.append(prediction_hash)
            if apply:
                _write_metrics(db, prediction_hash, headline, folds)
        if apply:
            db.commit()
    finally:
        db.close()
    if apply:
        for prediction_hash in orphans:
            shutil.rmtree(case_dir / "run_log" / "predictions" / prediction_hash)
    return {
        "status": "ok",
        "dirs": len(dirs),
        "shipped": len(shipped),
        "registered": len(registered),
        "recomputed": len(recomputed),
        "orphan_dirs": len(orphans),
        "rows_without_a_shipped_artifact": unbacked,
        "failed": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediates", type=Path, default=DEFAULT_INTERMEDIATES)
    parser.add_argument("--case-study", action="append", dest="case_studies")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="measure and change nothing (default)")
    mode.add_argument("--apply", action="store_true", help="write the reconciled rows")
    args = parser.parse_args()

    root = args.intermediates.expanduser().resolve()
    if root.name != "intermediates":
        parser.error(f"{root} is not a fixture intermediates root")
    names = args.case_studies or sorted(
        entry.name for entry in root.iterdir() if (entry / "run_log" / "registry.db").is_file()
    )
    failures = 0
    for name in names:
        stats = reconcile(root / name, apply=args.apply)
        if stats["status"] != "ok":
            print(f"{name:34s} {stats['status']}")
            continue
        print(
            f"{name:34s} dirs {stats['dirs']:4d}  with a parquet {stats['shipped']:4d}  "
            f"registered {stats['registered']:5d}  recomputed {stats['recomputed']:4d}  "
            f"orphan dirs {stats['orphan_dirs']:3d}  "
            f"rows with no shipped artifact {stats['rows_without_a_shipped_artifact']:5d}"
        )
        for prediction_hash, reason in stats["failed"].items():
            failures += 1
            print(f"    {prediction_hash}: {reason}")
    if not args.apply:
        print("\nNothing was written. Re-run with --apply.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
