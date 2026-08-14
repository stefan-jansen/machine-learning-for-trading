from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from case_studies.utils.backtest_presets import serializable_backtest_spec
from case_studies.utils.registry.specs import backtest_hash_from_parts, canonical_json, compute_hash
from case_studies.utils.registry.store import _open_registry, _utc_now

from .adapters import get_adapter
from .catalog import _resolve_authoritative_selection
from .contracts import ExecutionTier
from .models import ModelRequest, ModelRun, ResolvedModelRequest
from .population import OfficialPopulation
from .results import BacktestResult, PredictionResult, Result
from .strategy import Strategy
from .workspace import Study

if TYPE_CHECKING:
    from .decisions import DecisionArtifact


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
    population: OfficialPopulation | None

    @property
    def population_hash(self) -> str | None:
        return self.population.hash if self.population is not None else None


@dataclass(frozen=True)
class PlannedBacktest:
    training_hash: str
    prediction_hash: str
    backtest_hash: str
    spec_json: str


@dataclass(frozen=True)
class BacktestPlan:
    members: tuple[PlannedBacktest, ...]
    execution_tier: ExecutionTier

    @property
    def expected_hashes(self) -> tuple[str, ...]:
        return tuple(member.backtest_hash for member in self.members)


@dataclass(frozen=True)
class _ResolvedBacktest:
    member: PlannedBacktest
    prediction: PredictionResult
    strategy: Strategy


def run_models(
    study: Study,
    *,
    requests: Iterable[ModelRequest | ResolvedModelRequest],
) -> ModelExecution:
    submitted = list(requests)
    if not submitted:
        raise ValueError("run_models requires at least one request")
    for request in submitted:
        if request.study != study:
            raise ValueError("model request belongs to another study")

    ordered_runs: list[ModelRun | None] = [None] * len(submitted)
    unresolved: dict[str, list[tuple[int, ModelRequest]]] = {}
    for index, request in enumerate(submitted):
        if isinstance(request, ModelRequest):
            unresolved.setdefault(request.family, []).append((index, request))
            continue
        study.activate(ExecutionTier(request.spec["execution_tier"]))
        ordered_runs[index] = request.run()

    for family, indexed_requests in unresolved.items():
        module = get_adapter("model", family)
        batch_runner = getattr(module, "run_model_requests", None)
        if callable(batch_runner):
            batch_runs = tuple(
                batch_runner(study, [request.as_dict() for _, request in indexed_requests])
            )
            if len(batch_runs) != len(indexed_requests):
                raise ValueError(
                    f"{family!r} batch runner returned {len(batch_runs)} results for "
                    f"{len(indexed_requests)} requests"
                )
            for (index, _), run in zip(indexed_requests, batch_runs, strict=True):
                ordered_runs[index] = run
            continue
        for index, request in indexed_requests:
            resolved = request.resolve()
            study.activate(ExecutionTier(resolved.spec["execution_tier"]))
            ordered_runs[index] = resolved.run()

    if any(run is None for run in ordered_runs):
        raise RuntimeError("model execution did not produce a result for every request")
    runs = tuple(run for run in ordered_runs if run is not None)
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
    resolved = _resolve_authoritative_selection(
        study,
        predictions,
        kind="prediction",
        canonical=False,
    )
    if not all(isinstance(result, PredictionResult) for result in resolved):
        raise RuntimeError("prediction selection resolved a non-prediction result")
    return list(resolved)


def _resolve_backtest_plan(
    study: Study,
    *,
    predictions: pl.DataFrame,
    signal: dict[str, Any],
    prices: pl.DataFrame | None,
    allocation: dict[str, Any] | None,
    risk: dict[str, Any] | None,
    costs: dict[str, Any] | None,
    chapter: str | None,
    execution_mode: str | None,
    decision: DecisionArtifact | None,
) -> tuple[BacktestPlan, tuple[_ResolvedBacktest, ...]]:
    resolved_predictions = _validate_selection(study, predictions)
    if decision is not None and len(resolved_predictions) != 1:
        raise ValueError("one decision artifact requires exactly one selected prediction")
    tiers = {ExecutionTier(prediction.execution_tier) for prediction in resolved_predictions}
    if len(tiers) != 1:
        raise ValueError("one backtest plan cannot mix canonical and preview predictions")
    tier = tiers.pop()
    resolved: list[_ResolvedBacktest] = []
    for prediction in resolved_predictions:
        strategy = study.strategy(
            prediction=prediction,
            signal=signal,
            decision=decision,
            allocation=allocation,
            risk=risk,
            costs=costs,
            chapter=chapter,
            execution_mode=execution_mode,
        )
        spec = strategy.resolve(prices=prices)
        serializable = serializable_backtest_spec(spec)
        backtest_hash = backtest_hash_from_parts(
            prediction.hash,
            serializable,
            identity_version=spec.get("identity_version"),
        )
        member = PlannedBacktest(
            training_hash=str(prediction.registry_record()["training_hash"]),
            prediction_hash=prediction.hash,
            backtest_hash=backtest_hash,
            spec_json=canonical_json(serializable),
        )
        resolved.append(_ResolvedBacktest(member, prediction, strategy))
    plan = BacktestPlan(tuple(item.member for item in resolved), tier)
    return plan, tuple(resolved)


def plan_backtests(
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
    decision: DecisionArtifact | None = None,
) -> BacktestPlan:
    """Resolve every expected backtest identity without executing or writing one."""
    plan, _ = _resolve_backtest_plan(
        study,
        predictions=predictions,
        signal=signal,
        prices=prices,
        allocation=allocation,
        risk=risk,
        costs=costs,
        chapter=chapter,
        execution_mode=execution_mode,
        decision=decision,
    )
    return plan


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
    decision: DecisionArtifact | None = None,
    population_name: str | None = None,
) -> BacktestExecution:
    study.require_writable()
    if not isinstance(predictions, pl.DataFrame):
        raise TypeError("run_backtests requires a Polars prediction catalog selection")
    plan, resolved = _resolve_backtest_plan(
        study,
        predictions=predictions,
        signal=signal,
        prices=prices,
        allocation=allocation,
        risk=risk,
        costs=costs,
        chapter=chapter,
        execution_mode=execution_mode,
        decision=decision,
    )
    population = None
    canonical_decision = decision is None or decision.canonical
    if plan.execution_tier is ExecutionTier.CANONICAL and canonical_decision:
        ordered_hashes = tuple(sorted(plan.expected_hashes))
        if population_name is None:
            suffix = compute_hash(canonical_json({"members": list(ordered_hashes)}))
            population_name = f"backtests-{suffix}"
        population = OfficialPopulation.create(
            study,
            name=population_name,
            member_kind="backtest",
            members=ordered_hashes,
        )
    elif population_name is not None:
        ancestry = "preview" if plan.execution_tier is ExecutionTier.PREVIEW else "exploratory"
        raise ValueError(f"{ancestry} backtests cannot create an official population")
    for item in resolved:
        _import_released_prediction(study, item.prediction)

    results = []
    diagnostics = []
    for item in resolved:
        try:
            existing = Result.open(
                study,
                item.member.backtest_hash,
                include_preview=plan.execution_tier is ExecutionTier.PREVIEW,
            )
        except KeyError:
            existing = None
        if isinstance(existing, BacktestResult) and existing.complete:
            result = existing
            status = "reused"
        else:
            result = item.strategy.run(prices=prices)
            status = "completed"
        if result.hash != item.member.backtest_hash:
            raise RuntimeError(
                "backtest execution returned an identity different from its plan: "
                f"{item.member.backtest_hash} -> {result.hash}"
            )
        results.append(result)
        diagnostics.append(
            {
                "status": status,
                "prediction_hash": item.prediction.hash,
                "backtest_hash": result.hash,
            }
        )
    if population is not None:
        population.require_complete()
    result_hashes = [result.hash for result in results]
    catalog_rows = study.backtests.table(include_preview=True).filter(
        pl.col("backtest_hash").is_in(result_hashes)
    )
    return BacktestExecution(tuple(results), catalog_rows, tuple(diagnostics), population)
