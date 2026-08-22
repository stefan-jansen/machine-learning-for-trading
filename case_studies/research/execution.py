from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from case_studies.utils.backtest_presets import serializable_backtest_spec
from case_studies.utils.registry.specs import (
    backtest_hash_from_parts,
    canonical_json,
    compute_hash,
    prediction_hash_from_parts,
    training_hash_from_spec,
)
from case_studies.utils.registry.store import _open_registry, _utc_now

from .adapters import get_adapter
from .catalog import _resolve_authoritative_selection
from .contracts import ExecutionTier
from .lifecycle import ResearchLock
from .model_planning import ModelPlan, plan_models
from .models import (
    ModelRequest,
    ModelRun,
    ResolvedModelRequest,
    reconstruct_locked_model_request,
    validate_locked_model_run,
)
from .population import OfficialPopulation
from .results import BacktestResult, PredictionResult, Result, TrainingResult
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
class PreviewPopulation:
    """What a preview run declared and then produced, verified and not registered.

    A preview fits a reduced universe in a throwaway workspace, and
    :class:`~case_studies.research.population.OfficialPopulation` refuses such a result as a
    member so that nothing downstream can ever bind to one. The declaration is still worth making
    and checking - it is what catches a run that produced a different set of predictions than it
    said it would - so the preview gets this instead, with the same two fields a notebook reads.
    """

    name: str
    members: tuple[str, ...]


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
class HoldoutExecution:
    lock: ResearchLock
    training: TrainingResult
    prediction: PredictionResult
    backtest: BacktestResult
    fitted_state_digest: str


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


def _locked_checkpoint_is_declared(lock: ResearchLock) -> None:
    checkpoint_kind = lock.record.get("checkpoint_kind")
    if not isinstance(checkpoint_kind, str) or not checkpoint_kind:
        raise ValueError("research lock has no explicit checkpoint kind")
    schedule = (
        lock.record["holdout_training_spec"].get("computation", {}).get("checkpoint_schedule", [])
    )
    expected = (checkpoint_kind, lock.record["checkpoint_value"])
    declared = {(item.get("kind"), item.get("value")) for item in schedule}
    if expected not in declared:
        raise ValueError(f"locked checkpoint {expected!r} is absent from the training schedule")


def _validate_locked_prediction_population(
    training: TrainingResult,
    prediction: PredictionResult,
    expected: tuple[str, str, int | None],
) -> None:
    with closing(sqlite3.connect(training.root / "run_log" / "registry.db")) as db:
        rows = db.execute(
            "SELECT prediction_hash, split, checkpoint_kind, checkpoint_value "
            "FROM prediction_sets WHERE training_hash = ?",
            (training.hash,),
        ).fetchall()
    if rows != [(prediction.hash, *expected)]:
        raise ValueError("locked training lineage contains an unexpected prediction population")


def run_locked_holdout(lock: ResearchLock) -> HoldoutExecution:
    """Produce and atomically finalize the one holdout lineage authorized by ``lock``."""
    reopened = lock.reopen()
    if reopened.state != "LOCKED":
        raise ValueError("holdout execution requires a LOCKED research lock")
    spec = reopened.record["holdout_training_spec"]
    if training_hash_from_spec(spec) != reopened.record["holdout_training_hash"]:
        raise ValueError("research lock contains an invalid holdout training identity")
    _locked_checkpoint_is_declared(reopened)

    from .holdout import prepare_locked_strategy_replay

    strategy_replay = prepare_locked_strategy_replay(reopened)
    request = reconstruct_locked_model_request(
        reopened.study,
        spec,
        checkpoint_kind=reopened.record["checkpoint_kind"],
        checkpoint_value=reopened.record["checkpoint_value"],
    )
    model_run = request.run()
    if model_run.training.hash != reopened.record["holdout_training_hash"]:
        raise ValueError("locked model runner produced the wrong training identity")
    fitted_state_digest = validate_locked_model_run(request, model_run)
    if len(model_run.predictions) != 1:
        raise ValueError("locked model runner must publish only the selected checkpoint")
    prediction = model_run.predictions[0]
    prediction_record = prediction.registry_record()
    expected_prediction = (
        "holdout",
        reopened.record["checkpoint_kind"],
        reopened.record["checkpoint_value"],
    )
    actual_prediction = (
        prediction_record["split"],
        prediction_record["checkpoint_kind"],
        prediction_record["checkpoint_value"],
    )
    if not prediction.complete or actual_prediction != expected_prediction:
        raise ValueError("locked model runner produced the wrong holdout prediction")
    _validate_locked_prediction_population(
        model_run.training,
        prediction,
        expected_prediction,
    )
    if prediction.lineage()["training_spec"] != spec or not fitted_state_digest:
        raise ValueError("locked model fitted state does not validate against the training spec")

    backtest = strategy_replay.run(prediction)
    staged_fitted_state_digest = validate_locked_model_run(request, model_run)
    if staged_fitted_state_digest != fitted_state_digest:
        raise ValueError("locked model fitted state changed during holdout execution")
    reopened.study.lifecycle.stage_holdout(
        reopened.hash,
        holdout_training_hash=model_run.training.hash,
        holdout_prediction_hash=prediction.hash,
        holdout_backtest_hash=backtest.hash,
        fitted_state_digest=staged_fitted_state_digest,
    )
    evaluated = reopened.study.lifecycle.finalize_holdout(reopened.hash)
    return HoldoutExecution(
        evaluated,
        model_run.training,
        prediction,
        backtest,
        staged_fitted_state_digest,
    )


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


def prediction_hashes_from_specs(specs: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Project declared checkpoints to prediction identities, from specifications alone.

    A specification is all this needs. Taking resolved request objects instead forces every one of
    them to exist at the same moment, and a resolved linear request holds the prepared folds it
    was resolved against: on `us_equities_panel` one fold set measures 90 GB, so the sixteen
    declared configurations cannot be resolved together on any machine this program runs on.
    Planning produces the same specifications without retaining the arrays.
    """
    hashes = []
    for spec in specs:
        computation = spec.get("computation", spec)
        identity = training_hash_from_spec(spec)
        for checkpoint in computation["checkpoint_schedule"]:
            hashes.append(
                prediction_hash_from_parts(
                    identity,
                    checkpoint["value"],
                    "validation",
                    checkpoint_kind=checkpoint["kind"],
                    identity_version=spec["identity_version"],
                )
            )
    if len(hashes) != len(set(hashes)):
        raise ValueError("declared request population contains duplicate prediction identities")
    return tuple(hashes)


def expected_prediction_hashes(
    resolved_requests: Iterable[ResolvedModelRequest],
) -> tuple[str, ...]:
    """Project declared checkpoints to the validation prediction identities they will produce.

    Computed from the resolved specification alone, so it can be evaluated *before* anything is
    fitted. That is what lets a canonical run state its whole expected population up front and be
    held to it afterwards.
    """
    return prediction_hashes_from_specs(request.spec for request in resolved_requests)


def snapshot_official_models(
    study: Study,
    resolved_requests: Iterable[ResolvedModelRequest],
    *,
    population_name: str,
    supersedes: str | None = None,
) -> OfficialPopulation:
    """Record every expected canonical prediction identity before any member executes."""
    resolved = tuple(resolved_requests)
    if any(request.spec["execution_tier"] != "canonical" for request in resolved):
        raise ValueError("official model populations require canonical requests")
    return OfficialPopulation.create(
        study,
        name=population_name,
        member_kind="prediction",
        members=expected_prediction_hashes(resolved),
        supersedes=supersedes,
    )


def run_official_model_subset(
    study: Study,
    resolved_requests: Iterable[ResolvedModelRequest | ModelRequest],
    *,
    population: OfficialPopulation | str,
    expected: Iterable[str] | None = None,
    require_population_complete: bool = False,
) -> tuple[ModelExecution, OfficialPopulation]:
    """Execute members a case-wide official population already declared.

    `expected` is the declared prediction set. It is derived from the requests when they are
    resolved, and passed in when they are not - an unresolved request cannot state its identity
    without resolving, which is exactly what a large panel cannot afford to do for every
    configuration at once.
    """
    resolved = tuple(resolved_requests)
    tiers = {
        request.spec["execution_tier"]
        if isinstance(request, ResolvedModelRequest)
        else request.execution_tier.value
        for request in resolved
    }
    if tiers != {"canonical"}:
        raise ValueError("official model subsets require canonical requests")
    if isinstance(population, str):
        population = OfficialPopulation.one(study, name=population)
    elif population.study != study:
        raise ValueError("official model population belongs to another study")
    if expected is None:
        expected = expected_prediction_hashes(resolved)
    expected = tuple(expected)
    undeclared = sorted(set(expected) - set(population.members))
    if undeclared:
        raise ValueError(f"model subset contains undeclared predictions: {undeclared}")
    execution = run_models(study, requests=resolved)
    actual = tuple(prediction.hash for run in execution.runs for prediction in run.predictions)
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"model population mismatch: missing={missing}, extra={extra}")
    if require_population_complete:
        population.require_complete()
    return execution, population


def run_official_models(
    study: Study,
    requests: ModelPlan | Iterable[ModelRequest | ResolvedModelRequest],
    *,
    population_name: str,
    supersedes: str | None = None,
) -> tuple[ModelExecution, OfficialPopulation]:
    """Snapshot, execute and verify one complete canonical model population.

    The population is written before the first fit, so a run that produces a different set of
    predictions than it declared fails rather than quietly publishing the set it happened to
    produce. This is the canonical entry point for a model-execution notebook.

    Unresolved requests are planned rather than resolved, and then handed to the batch runner
    still unresolved. Both halves of that matter on a large panel. Resolving every request up
    front holds every configuration's prepared folds at once - 90 GB per fold set times sixteen
    configurations on `us_equities_panel` - while planning computes the same identities from
    placeholder folds; and the batch runner walks folds on the outside and configurations on the
    inside, so one fold set is live at a time instead of one per configuration. The declaration
    is unchanged: the same identities are written down before the same fits happen, and
    `pre_run_gate.py` checks that the two paths agree on them.

    A notebook that already built a :class:`ModelPlan` to show what it is about to fit passes the
    plan itself. Passing its requests instead would plan a second time, and planning a large panel
    is not free: resolving a data-dependent penalty prepares every fold to do it.
    """
    if isinstance(requests, ModelPlan):
        if requests.study != study:
            raise ValueError("model plan belongs to another study")
        submitted = requests.requests
        plan: ModelPlan | None = requests
    else:
        submitted = tuple(requests)
        plan = None
    unresolved = tuple(request for request in submitted if isinstance(request, ModelRequest))
    if len(unresolved) == len(submitted):
        if plan is None:
            plan = plan_models(study, requests=list(unresolved))
        if plan.execution_tier is not ExecutionTier.CANONICAL:
            raise ValueError("official model populations require canonical requests")
        population = plan.create_population(name=population_name, supersedes=supersedes)
        return run_official_model_subset(
            study,
            submitted,
            population=population,
            expected=plan.expected_prediction_hashes,
            require_population_complete=True,
        )

    resolved = tuple(
        request.resolve() if isinstance(request, ModelRequest) else request for request in submitted
    )
    population = snapshot_official_models(
        study, resolved, population_name=population_name, supersedes=supersedes
    )
    return run_official_model_subset(
        study,
        resolved,
        population=population,
        require_population_complete=True,
    )


def run_model_population(
    study: Study,
    requests: ModelPlan | Iterable[ModelRequest | ResolvedModelRequest],
    *,
    population_name: str,
    supersedes: str | None = None,
) -> tuple[ModelExecution, OfficialPopulation | PreviewPopulation]:
    """Execute one model population in whichever tier its requests declare.

    Both tiers declare the whole expected set of predictions before the first fit and fail if the
    run produces a different one, which is the check most likely to catch a mistake at the end of
    a long canonical run and therefore the one a preview must rehearse.

    They differ in what the declaration is worth afterwards. A canonical run registers an
    immutable population that downstream work binds to. A preview's is a reduced result computed
    in a throwaway workspace, and :class:`OfficialPopulation` refuses such a member by design, so
    the preview gets a declaration that verifies and is then discarded with its workspace.

    Every model-execution notebook calls this rather than branching on the tier itself. It takes
    either the requests or the :class:`ModelPlan` built from them; a notebook that shows its plan
    before running passes the plan, so the panel is planned once rather than twice.

    ``supersedes`` names the population hash this run replaces. A population is the set of
    prediction identities, so anything that moves a training identity - a changed estimator
    parameter as much as a changed configuration menu - produces a different population under the
    same name, and :class:`OfficialPopulation` refuses to write it without being told which
    snapshot it supersedes. Refitting the nine GBM sweeps on a corrected ``max_bin`` is exactly
    that case: the members are the same configurations, the predictions are not, and the lineage
    is the only record of which is which. Canonical tier only.
    """
    if isinstance(requests, ModelPlan):
        if requests.study != study:
            raise ValueError("model plan belongs to another study")
        submitted = requests.requests
    else:
        submitted = tuple(requests)
    if not submitted:
        raise ValueError("run_model_population requires at least one request")
    tiers = {
        ExecutionTier(
            request.spec["execution_tier"]
            if isinstance(request, ResolvedModelRequest)
            else request.execution_tier
        )
        for request in submitted
    }
    if len(tiers) != 1:
        raise ValueError(
            f"one population cannot mix execution tiers: {sorted(t.value for t in tiers)}"
        )
    tier = tiers.pop()

    if tier is ExecutionTier.CANONICAL:
        return run_official_models(
            study,
            requests if isinstance(requests, ModelPlan) else submitted,
            population_name=population_name,
            supersedes=supersedes,
        )

    if supersedes is not None:
        # A preview population is discarded with its workspace, so it has no lineage to extend.
        # Accepting the argument here would let a caller believe a snapshot was superseded when
        # nothing was written down.
        raise ValueError("preview populations cannot supersede a snapshot")
    if study.output_root is None:
        raise ValueError("preview execution requires an isolated workspace")
    for request in submitted:
        if isinstance(request, ResolvedModelRequest):
            # A resolved spec carries its reductions inside ``computation``, where they are part
            # of the identity: a preview and a canonical result must never hash alike.
            reductions = request.spec.get("computation", {}).get("preview_reductions")
        else:
            reductions = request.preview_reductions
        if not reductions:
            raise ValueError("preview execution requires every request to declare its reductions")

    resolved = tuple(
        request.resolve() if isinstance(request, ModelRequest) else request for request in submitted
    )
    declared = expected_prediction_hashes(resolved)
    execution = run_models(study, requests=resolved)
    produced = tuple(prediction.hash for run in execution.runs for prediction in run.predictions)
    if set(produced) != set(declared) or len(produced) != len(declared):
        missing = sorted(set(declared) - set(produced))
        extra = sorted(set(produced) - set(declared))
        raise RuntimeError(f"model population mismatch: missing={missing}, extra={extra}")
    return execution, PreviewPopulation(name=population_name, members=declared)
