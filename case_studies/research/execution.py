from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from case_studies.utils.registry.store import _open_registry, _utc_now

from .models import ModelRequest, ModelRun, ResolvedModelRequest
from .results import BacktestResult, PredictionResult, Result
from .workspace import Study


@dataclass(frozen=True)
class ModelExecution:
    runs: tuple[ModelRun, ...]
    catalog_rows: pl.DataFrame
    diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BacktestExecution:
    results: tuple[BacktestResult, ...]
    catalog_rows: pl.DataFrame
    diagnostics: tuple[dict[str, Any], ...]


def run_models(
    study: Study,
    *,
    requests: Iterable[ModelRequest | ResolvedModelRequest],
) -> ModelExecution:
    resolved: list[ResolvedModelRequest] = []
    for request in requests:
        if request.study != study:
            raise ValueError("model request belongs to another study")
        resolved.append(request.resolve() if isinstance(request, ModelRequest) else request)
    if not resolved:
        raise ValueError("run_models requires at least one request")

    runs = tuple(request.run() for request in resolved)
    hashes = [prediction.hash for run in runs for prediction in run.predictions]
    catalog_rows = study.predictions.table(include_preview=True).filter(
        pl.col("prediction_hash").is_in(hashes)
    )
    diagnostics = tuple(
        {"status": "completed", "training_hash": run.training.hash, **run.diagnostics}
        for run in runs
    )
    return ModelExecution(runs, catalog_rows, diagnostics)


def _copy_row(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    table: str,
    where: str,
    params: tuple[Any, ...],
) -> None:
    def quoted(column: str) -> str:
        return '"' + column.replace('"', '""') + '"'

    source_tables = {
        row[0]
        for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if table not in source_tables:
        return
    source_columns = [row[1] for row in source.execute(f"PRAGMA table_info({table})")]
    destination_columns = {row[1] for row in destination.execute(f"PRAGMA table_info({table})")}
    columns = [column for column in source_columns if column in destination_columns]
    rows = source.execute(
        f"SELECT {', '.join(quoted(column) for column in columns)} FROM {table} WHERE {where}",
        params,
    ).fetchall()
    if not rows:
        return
    destination.executemany(
        f"INSERT OR IGNORE INTO {table} ({', '.join(quoted(column) for column in columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        rows,
    )


def _import_released_prediction(study: Study, prediction: PredictionResult) -> None:
    if prediction.origin != "released":
        return
    source_path = prediction.root / "run_log" / "registry.db"
    prediction_record = prediction.registry_record()
    training_hash = prediction_record["training_hash"]
    with closing(sqlite3.connect(f"file:{source_path}?mode=ro&immutable=1", uri=True)) as source:
        destination = _open_registry(study.root)
        try:
            destination.execute("BEGIN IMMEDIATE")
            _copy_row(source, destination, "training_runs", "training_hash = ?", (training_hash,))
            _copy_row(
                source,
                destination,
                "prediction_sets",
                "prediction_hash = ?",
                (prediction.hash,),
            )
            for table in ("prediction_coverage", "prediction_metrics", "fold_metrics"):
                _copy_row(
                    source,
                    destination,
                    table,
                    "prediction_hash = ?",
                    (prediction.hash,),
                )
            destination.executemany(
                "INSERT OR IGNORE INTO overlay_references "
                "(result_hash, result_kind, source_root, created_at) VALUES (?,?,?,?)",
                [
                    (training_hash, "training", str(prediction.root), _utc_now()),
                    (prediction.hash, "prediction", str(prediction.root), _utc_now()),
                ],
            )
            destination.commit()
        except Exception:
            destination.rollback()
            raise
        finally:
            destination.close()


def _validate_selection(study: Study, predictions: pl.DataFrame) -> list[PredictionResult]:
    required = {"training_hash", "prediction_hash"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"prediction catalog selection is missing required identity columns: {sorted(missing)}"
        )
    if predictions.is_empty():
        raise ValueError("prediction catalog selection is empty")
    if predictions.get_column("prediction_hash").n_unique() != predictions.height:
        raise ValueError("duplicate prediction identities make the selection ambiguous")
    catalog = study.predictions.table(include_preview=True)
    resolved = []
    for row in predictions.select("training_hash", "prediction_hash").iter_rows(named=True):
        match = catalog.filter(pl.col("prediction_hash") == row["prediction_hash"])
        if match.height != 1:
            raise ValueError(
                f"prediction identity {row['prediction_hash']!r} resolved to {match.height} rows"
            )
        authoritative = match.row(0, named=True)
        if authoritative["training_hash"] != row["training_hash"]:
            raise ValueError(
                f"prediction {row['prediction_hash']} has training_hash "
                f"{authoritative['training_hash']}, not {row['training_hash']}"
            )
        if not authoritative["complete"]:
            raise ValueError(f"prediction {row['prediction_hash']} is partial")
        result = Result.open(study, str(row["prediction_hash"]), include_preview=True)
        if not isinstance(result, PredictionResult):
            raise ValueError(f"catalog identity {row['prediction_hash']} is not a prediction")
        resolved.append(result)
    return resolved


def run_backtests(
    study: Study,
    *,
    predictions: pl.DataFrame,
    signal: dict[str, Any],
    prices: pl.DataFrame | None = None,
    allocation: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    costs: dict[str, Any] | None = None,
    chapter: str | None = None,
    execution_mode: str | None = None,
    decision=None,
) -> BacktestExecution:
    study.require_writable()
    if not isinstance(predictions, pl.DataFrame):
        raise TypeError("run_backtests requires a Polars prediction catalog selection")
    resolved = _validate_selection(study, predictions)
    if decision is not None and len(resolved) != 1:
        raise ValueError("one decision artifact requires exactly one selected prediction")
    for prediction in resolved:
        _import_released_prediction(study, prediction)

    results = []
    diagnostics = []
    for prediction in resolved:
        result = study.strategy(
            prediction=prediction,
            signal=signal,
            decision=decision,
            allocation=allocation,
            risk=risk,
            costs=costs,
            chapter=chapter,
            execution_mode=execution_mode,
        ).run(prices=prices)
        results.append(result)
        diagnostics.append(
            {
                "status": "completed",
                "prediction_hash": prediction.hash,
                "backtest_hash": result.hash,
            }
        )
    return BacktestExecution(tuple(results), predictions.clone(), tuple(diagnostics))
