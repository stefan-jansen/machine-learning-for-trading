"""Reader-facing S&P 500 option decision and backtest workflow."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from case_studies.research import (
    BacktestResult,
    DecisionArtifact,
    OfficialPopulation,
    PredictionResult,
    ResolvedModelRequest,
    Result,
    Study,
    plan_backtests,
    run_models,
)
from case_studies.research.execution import ModelExecution
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.sp500_options._htm_backtest import (
    _apply_cohort_allocator,
    _load_option_lifecycle,
    _select_cohorts,
    option_contract_source_identity,
    option_data_paths,
)
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.backtest_runner import (
    apply_universe_filter,
    normalize_prediction_columns,
)
from case_studies.utils.registry import prediction_hash_from_parts
from utils.modeling import load_configs
from utils.paths import REPO_ROOT

CASE_STUDY = "sp500_options"
PRIMARY_LABEL = "ret_to_expiry"
ALL_LABELS = (PRIMARY_LABEL,)


@dataclass(frozen=True)
class OptionBacktestExecution:
    results: tuple[BacktestResult, ...]
    catalog_rows: pl.DataFrame
    population: OfficialPopulation | None


def open_study(*, execution_tier: str, workspace: str | Path | None = None) -> Study:
    """Open canonical regeneration or an isolated reader preview."""
    if execution_tier == "canonical":
        return Study.regenerate(CASE_STUDY, release_root=REPO_ROOT)
    if execution_tier != "preview":
        raise ValueError("execution_tier must be canonical or preview")
    if workspace is None:
        raise ValueError("preview execution requires an explicit workspace")
    workspace = Path(workspace).expanduser().resolve()
    generated = tuple(
        REPO_ROOT / "case_studies" / CASE_STUDY / name for name in ("features", "labels", "run_log")
    )
    if all(path.is_symlink() for path in generated):
        workspace.mkdir(parents=True, exist_ok=True)
        shared_config = workspace / "config"
        if not shared_config.exists():
            shared_config.symlink_to(
                REPO_ROOT / "case_studies" / "config", target_is_directory=True
            )
        study = Study(
            case_study=CASE_STUDY,
            root=REPO_ROOT / "case_studies" / CASE_STUDY,
            release_root=REPO_ROOT,
            output_root=workspace,
            read_only=False,
            manifest={
                "schema_version": 1,
                "case_study": CASE_STUDY,
                "baseline_source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip(),
                "preview_only": True,
            },
        )
        study.activate(execution_tier)
        return study
    return Study.open(CASE_STUDY, workspace=workspace, release_root=REPO_ROOT)


def model_request_catalog(
    family: str,
    *,
    labels: Iterable[str] = ALL_LABELS,
    config_names: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Return the declared model population as visible Polars rows."""
    selected = set(config_names) if config_names is not None else None
    rows = []
    for label in labels:
        for config in load_configs(CASE_STUDY, label, family):
            name = str(config["config_name"])
            if selected is None or name in selected:
                rows.append({"family": family, "label": label, "config_name": name})
    if not rows:
        raise ValueError(f"no declared requests for {family!r}")
    return pl.DataFrame(rows).unique(maintain_order=True)


def resolve_model_requests(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
) -> tuple[ResolvedModelRequest, ...]:
    """Resolve visible catalog rows through the shared family boundary."""
    required = {"family", "label", "config_name"}
    missing = required - set(request_catalog.columns)
    if missing:
        raise ValueError(f"model request catalog is missing {sorted(missing)}")
    return tuple(
        study.model(
            **row,
            execution_tier=execution_tier,
            overrides=dict(overrides or {}),
            preview_reductions=dict(preview_reductions or {}),
        ).resolve()
        for row in request_catalog.select(*sorted(required)).iter_rows(named=True)
    )


def run_resolved_model_requests(
    study: Study,
    resolved_requests: Iterable[ResolvedModelRequest],
) -> ModelExecution:
    """Execute already-resolved requests without re-resolving their inputs."""
    resolved = tuple(resolved_requests)
    if not resolved:
        raise ValueError("resolved model requests cannot be empty")
    execution = run_models(study, requests=resolved)
    expected_rows = sum(len(run.predictions) for run in execution.runs)
    if execution.catalog_rows.height != expected_rows:
        raise RuntimeError("model execution did not return every checkpoint catalog row")
    if execution.catalog_rows.filter(~pl.col("complete")).height:
        raise RuntimeError("model execution returned incomplete prediction rows")
    return execution


def expected_prediction_hashes(
    resolved_requests: Iterable[ResolvedModelRequest],
) -> tuple[str, ...]:
    """Project declared checkpoints to immutable validation prediction identities."""
    hashes = []
    for request in resolved_requests:
        computation = request.spec.get("computation", request.spec)
        for checkpoint in computation["checkpoint_schedule"]:
            hashes.append(
                prediction_hash_from_parts(
                    request.identity,
                    checkpoint["value"],
                    "validation",
                    checkpoint_kind=checkpoint["kind"],
                    identity_version=request.spec["identity_version"],
                )
            )
    if len(hashes) != len(set(hashes)):
        raise ValueError("declared request population contains duplicate prediction identities")
    return tuple(hashes)


def run_official_model_catalog(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    population_name: str,
    resolved_requests: Iterable[ResolvedModelRequest] | None = None,
) -> tuple[ModelExecution, OfficialPopulation]:
    """Snapshot and execute one complete canonical model population."""
    resolved = tuple(resolved_requests or ())
    if not resolved:
        resolved = resolve_model_requests(study, request_catalog, execution_tier="canonical")
    population = snapshot_official_model_catalog(
        study,
        request_catalog,
        population_name=population_name,
        resolved_requests=resolved,
    )
    execution, population = run_official_model_subset(
        study,
        resolved,
        population=population,
        require_population_complete=True,
    )
    return execution, population


def snapshot_official_model_catalog(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    population_name: str,
    resolved_requests: Iterable[ResolvedModelRequest] | None = None,
) -> OfficialPopulation:
    """Snapshot every expected canonical prediction before any member executes."""
    resolved = tuple(resolved_requests or ())
    if not resolved:
        resolved = resolve_model_requests(study, request_catalog, execution_tier="canonical")
    if any(request.spec["execution_tier"] != "canonical" for request in resolved):
        raise ValueError("official model populations require canonical requests")
    expected = expected_prediction_hashes(resolved)
    return OfficialPopulation.create(
        study,
        name=population_name,
        member_kind="prediction",
        members=expected,
    )


def run_official_model_subset(
    study: Study,
    resolved_requests: Iterable[ResolvedModelRequest],
    *,
    population: OfficialPopulation | str,
    require_population_complete: bool = False,
) -> tuple[ModelExecution, OfficialPopulation]:
    """Execute members already declared by a case-wide official population."""
    resolved = tuple(resolved_requests)
    if any(request.spec["execution_tier"] != "canonical" for request in resolved):
        raise ValueError("official model subsets require canonical requests")
    if isinstance(population, str):
        population = OfficialPopulation.one(study, name=population)
    elif population.study != study:
        raise ValueError("official model population belongs to another study")
    expected = expected_prediction_hashes(resolved)
    undeclared = sorted(set(expected) - set(population.members))
    if undeclared:
        raise ValueError(f"model subset contains undeclared predictions: {undeclared}")
    execution = run_resolved_model_requests(study, resolved)
    actual = tuple(prediction.hash for run in execution.runs for prediction in run.predictions)
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"model population mismatch: missing={missing}, extra={extra}")
    if require_population_complete:
        population.require_complete()
    return execution, population


def resolved_model_plan(
    resolved_requests: Iterable[ResolvedModelRequest],
) -> pl.DataFrame:
    """Show the data, folds, checkpoints, and eligibility for each request."""
    rows = []
    for request in resolved_requests:
        computation = request.spec.get("computation", request.spec)
        expected = request._context.expected_keys
        entity = next(
            (column for column in ("symbol", "product") if column in expected.columns), None
        )
        fold = next((column for column in ("fold", "fold_id") if column in expected.columns), None)
        if entity is None or fold is None:
            raise ValueError("resolved model eligibility has no entity or fold key")
        timestamps = expected.get_column("timestamp")
        rows.append(
            {
                "family": request.family,
                "label": request.spec["label"],
                "config_name": request.spec.get("config_name"),
                "task": (computation.get("task") or {}).get("type", "regression"),
                "feature_count": len(computation.get("feature_names") or []),
                "eligible_entities": expected.get_column(entity).n_unique(),
                "eligible_rows": expected.height,
                "folds": expected.get_column(fold).n_unique(),
                "validation_start": timestamps.min(),
                "validation_end": timestamps.max(),
                "checkpoints": len(computation["checkpoint_schedule"]),
                "execution_tier": request.spec["execution_tier"],
                "training_hash": request.identity,
            }
        )
    return pl.DataFrame(rows).sort("label", "family", "config_name")


def official_prediction_catalog(
    study: Study,
    population_names: Iterable[str],
) -> pl.DataFrame:
    """Resolve named complete prediction populations to exact Polars catalog rows."""
    members = []
    for name in population_names:
        population = OfficialPopulation.one(study, name=name)
        if population.member_kind != "prediction":
            raise ValueError(f"official population {name!r} does not contain predictions")
        members.extend(population.require_complete())
    if len(members) != len(set(members)):
        raise ValueError("official prediction populations overlap")
    catalog = study.predictions.table().filter(pl.col("prediction_hash").is_in(members))
    if catalog.height != len(members) or catalog.filter(~pl.col("complete")).height:
        raise ValueError("official prediction catalog is incomplete")
    return catalog.sort(
        "label",
        "family",
        "config_name",
        "checkpoint_kind",
        "checkpoint_value",
    )


def selected_prediction(study: Study, catalog_row: dict[str, Any]) -> PredictionResult:
    """Resolve one complete selected prediction row."""
    result = Result.open(
        study,
        str(catalog_row["prediction_hash"]),
        include_preview=catalog_row.get("execution_tier") == "preview",
    )
    if not isinstance(result, PredictionResult) or not result.complete:
        raise ValueError("selected catalog row does not identify a complete prediction")
    return result


def resolve_short_straddle_decisions(
    prediction: PredictionResult,
    *,
    prices: pl.DataFrame,
    signal: dict[str, Any],
    allocation: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Resolve ranked predictions to exact option contracts and cohort weights."""
    if prediction.lineage()["training_spec"]["label"] != PRIMARY_LABEL:
        raise ValueError("short-straddle decisions require ret_to_expiry predictions")
    predictions = normalize_prediction_columns(prediction.load())
    predictions = apply_universe_filter(
        predictions,
        prices,
        CASE_STUDY,
        signal,
        prediction_hash=prediction.hash,
    )
    labels_dir, raw_options_dir = option_data_paths()
    contract_returns = pl.read_parquet(labels_dir / "contract_returns.parquet")
    decisions = _select_cohorts(
        predictions,
        contract_returns,
        method=str(signal.get("method", "equal_weight_top_k")),
        top_k=int(signal.get("top_k", 20)),
        percentile=float(signal.get("percentile", 90.0)),
    )
    if allocation:
        decisions = _apply_cohort_allocator(decisions, raw_options_dir, allocation)
    fold_columns = [column for column in ("fold", "fold_id") if column in predictions.columns]
    if len(fold_columns) != 1:
        raise ValueError("option predictions require exactly one fold column")
    fold = fold_columns[0]
    fold_by_time = (
        predictions.select("timestamp", fold)
        .unique()
        .group_by("timestamp")
        .agg(pl.col(fold).n_unique().alias("n_folds"), pl.col(fold).first().alias("fold"))
    )
    if fold_by_time.filter(pl.col("n_folds") != 1).height:
        raise ValueError("each option decision timestamp must belong to exactly one fold")
    decisions = decisions.with_columns(
        pl.col("timestamp").cast(fold_by_time.schema["timestamp"])
    ).join(fold_by_time.select("timestamp", "fold"), on="timestamp", how="left")
    if decisions.get_column("fold").null_count():
        raise ValueError("option decisions contain timestamps outside prediction eligibility")
    return decisions.drop("y_score").sort("timestamp", "symbol")


def publish_short_straddle_decisions(
    prediction: PredictionResult,
    *,
    prices: pl.DataFrame,
    signal: dict[str, Any],
    allocation: dict[str, Any] | None = None,
    canonical: bool = False,
    clean_replay_digest: str | None = None,
) -> DecisionArtifact:
    """Publish exact short-straddle contracts with settlement and hedge policy."""
    if signal.get("exit_at_max_days") is not None:
        raise ValueError("short-straddle decisions are hold-to-expiry and cannot declare an exit")
    decisions = resolve_short_straddle_decisions(
        prediction,
        prices=prices,
        signal=signal,
        allocation=allocation,
    )
    return _publish_resolved_short_straddle_decisions(
        prediction,
        decisions=decisions,
        prices=prices,
        signal=signal,
        allocation=allocation,
        canonical=canonical,
        clean_replay_digest=clean_replay_digest,
    )


def _publish_resolved_short_straddle_decisions(
    prediction: PredictionResult,
    *,
    decisions: pl.DataFrame,
    prices: pl.DataFrame,
    signal: dict[str, Any],
    allocation: dict[str, Any] | None,
    canonical: bool,
    clean_replay_digest: str | None,
) -> DecisionArtifact:
    if canonical and clean_replay_digest is None:
        raise ValueError("canonical option decisions require a clean-process replay digest")
    labels_dir, _ = option_data_paths()
    contract_identity = option_contract_source_identity(labels_dir)
    source_identity: dict[str, Any] | None = None
    if canonical:
        source_identity = {
            "module": "case_studies.sp500_options.research_workflow",
            "source_digest": hashlib.sha256(
                inspect.getsource(resolve_short_straddle_decisions).encode()
            ).hexdigest(),
            "holdout_replay": {
                "version": 1,
                "function": "resolve_short_straddle_decisions",
            },
            "declared_inputs": {
                "prediction_hashes": [prediction.hash],
                "prices": value_digest(prices),
                "option_contract_returns": contract_identity,
                "signal": signal,
                "allocation": allocation,
            },
            "determinism": {"deterministic": True},
            "clean_replay_digest": clean_replay_digest,
        }
    return DecisionArtifact.publish(
        prediction.study,
        kind="short_straddles",
        decisions=decisions,
        prediction_hashes=[prediction.hash],
        parameters={
            "decision_cadence": "weekly_friday",
            "entry_policy": "next_session_close",
            "exit_policy": "hold_to_expiry",
            "settlement_policy": "cash_intrinsic_at_expiration",
            "hedge_policy": "retained_underlying_delta_with_threshold",
            "signal": signal,
            "allocation": allocation,
        },
        source_identity=source_identity,
        canonical=canonical,
    )


def _clean_replay_digests(
    study: Study,
    requests: list[dict[str, Any]],
) -> dict[str, str]:
    """Replay a complete decision request set in a fresh interpreter."""
    tiers = {str(request["execution_tier"]) for request in requests}
    if len(tiers) != 1:
        raise ValueError("clean option decision replay cannot mix execution tiers")
    payload = {
        "study": {
            "case_study": study.case_study,
            "root": str(study.root),
            "release_root": str(study.release_root),
            "output_root": str(study.output_root) if study.output_root is not None else None,
            "manifest": study.manifest,
        },
        "execution_tier": tiers.pop(),
        "requests": requests,
    }
    completed = subprocess.run(
        [sys.executable, "-m", "case_studies.sp500_options.research_workflow", "--replay"],
        cwd=REPO_ROOT,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "clean option decision replay failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    marker = "OPTION_DECISION_REPLAY="
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError("clean option decision replay did not return one digest set")
    replayed = json.loads(lines[0].removeprefix(marker))
    return {str(row["request_name"]): str(row["decision_digest"]) for row in replayed}


def strategy_request_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build visible option strategy rows while retaining nested request fields."""
    if not rows:
        raise ValueError("strategy request rows cannot be empty")
    nested = ("signal", "allocation", "risk", "costs")
    scalar_rows = [{key: value for key, value in row.items() if key not in nested} for row in rows]
    frame = pl.DataFrame(scalar_rows)
    return frame.with_columns(
        *(pl.Series(name, [row.get(name) for row in rows], dtype=pl.Object) for name in nested)
    )


def run_official_backtest_requests(
    study: Study,
    requests: pl.DataFrame,
    *,
    population_name: str | None,
) -> OptionBacktestExecution:
    """Resolve and execute typed option requests, snapshotting canonical populations."""
    required = {"request_name", "prediction_hash", "label", "signal"}
    missing = required - set(requests.columns)
    if missing:
        raise ValueError(f"strategy requests are missing columns: {sorted(missing)}")
    if (
        requests.is_empty()
        or requests.get_column("request_name").null_count()
        or requests.get_column("request_name").n_unique() != requests.height
        or any(not str(value).strip() for value in requests.get_column("request_name"))
    ):
        raise ValueError("strategy request names must be non-empty and unique")
    request_rows = list(requests.iter_rows(named=True))
    for row in request_rows:
        if row.get("risk") is not None:
            raise ValueError("the specialized option path does not support risk overlays")
        if row.get("costs") is not None:
            raise ValueError(
                "option cost variants use signal.option_spread_fraction; generic costs are unsupported"
            )
        if row["signal"].get("exit_at_max_days") is not None:
            raise ValueError("official short-straddle requests must hold to expiration")
    catalog = study.predictions.table(include_preview=True)
    price_cache: dict[tuple[str, int], pl.DataFrame] = {}
    resolved = []
    replay_requests = []
    execution_tiers = set()
    for row in request_rows:
        selected = catalog.filter(pl.col("prediction_hash") == row["prediction_hash"])
        if selected.height != 1 or not selected.item(0, "complete"):
            raise ValueError(
                f"prediction {row['prediction_hash']!r} is absent, ambiguous, or incomplete"
            )
        if selected.item(0, "label") != row["label"] or row["label"] != PRIMARY_LABEL:
            raise ValueError("option request label does not match its prediction catalog row")
        prediction = selected_prediction(study, selected.row(0, named=True))
        execution_tiers.add(prediction.execution_tier)
        allocation = row.get("allocation")
        warmup = strategy_warmup_periods({"strategy": {"allocation": allocation}})
        cache_key = (str(row["label"]), warmup)
        if cache_key not in price_cache:
            price_cache[cache_key] = load_backtest_prices_for(
                CASE_STUDY,
                str(row["label"]),
                split="validation",
                warmup_periods=warmup,
            )
        prices = price_cache[cache_key]
        decisions = resolve_short_straddle_decisions(
            prediction,
            prices=prices,
            signal=row["signal"],
            allocation=allocation,
        )
        resolved.append((row, prediction, prices, decisions))
        replay_requests.append(
            {
                "request_name": row["request_name"],
                "prediction_hash": prediction.hash,
                "label": row["label"],
                "signal": row["signal"],
                "allocation": allocation,
                "execution_tier": prediction.execution_tier,
            }
        )
    if len(execution_tiers) != 1:
        raise ValueError("one option request population cannot mix execution tiers")
    execution_tier = execution_tiers.pop()
    if execution_tier == "canonical" and population_name is None:
        raise ValueError("canonical option execution requires an official population name")
    if execution_tier == "preview" and population_name is not None:
        raise ValueError("preview option execution cannot create an official population")
    replay_digests = _clean_replay_digests(study, replay_requests)
    local_digests = {
        row["request_name"]: value_digest(decisions)
        for row, _prediction, _prices, decisions in resolved
    }
    if replay_digests != local_digests:
        raise RuntimeError("clean option decision replay differs from the prepared request set")

    prepared = []
    expected = []
    all_decisions = []
    for row, prediction, prices, decisions in resolved:
        allocation = row.get("allocation")
        decision = _publish_resolved_short_straddle_decisions(
            prediction,
            decisions=decisions,
            prices=prices,
            signal=row["signal"],
            allocation=allocation,
            canonical=execution_tier == "canonical",
            clean_replay_digest=replay_digests[row["request_name"]],
        )
        prediction_row = catalog.filter(pl.col("prediction_hash") == prediction.hash)
        if prediction_row.height != 1:
            raise RuntimeError("prepared option prediction no longer resolves uniquely")
        plan = plan_backtests(
            study,
            predictions=prediction_row,
            signal=row["signal"],
            prices=prices,
            allocation=allocation,
            risk=row.get("risk"),
            costs=row.get("costs"),
            chapter=row.get("chapter"),
            decision=decision,
        )
        if len(plan.members) != 1:
            raise RuntimeError("one option strategy request must resolve to one backtest")
        expected_hash = plan.expected_hashes[0]
        expected.append(expected_hash)
        all_decisions.append(decision.load())
        prepared.append((row, prediction, prices, decision, expected_hash))
    if len(expected) != len(set(expected)):
        raise ValueError("strategy requests resolve to duplicate backtest identities")
    population = (
        OfficialPopulation.create(
            study,
            name=population_name,
            member_kind="backtest",
            members=expected,
        )
        if population_name is not None
        else None
    )
    labels_dir, raw_options_dir = option_data_paths()
    del labels_dir
    lifecycle = _load_option_lifecycle(pl.concat(all_decisions), raw_options_dir)
    results = []
    rows = []
    for row, prediction, prices, decision, expected_hash in prepared:
        strategy = study.strategy(
            prediction=prediction,
            signal=row["signal"],
            allocation=row.get("allocation"),
            risk=row.get("risk"),
            costs=row.get("costs"),
            chapter=row.get("chapter"),
            decision=decision,
        )
        result = strategy.run(prices=prices, option_lifecycle=lifecycle)
        if result.hash != expected_hash:
            raise RuntimeError(f"backtest identity changed: {expected_hash} -> {result.hash}")
        results.append(result)
        rows.append(
            {
                "request_name": row["request_name"],
                "label": row["label"],
                "prediction_hash": prediction.hash,
                "decision_hash": decision.hash,
                "backtest_hash": result.hash,
                "complete": result.complete,
            }
        )
    if tuple(result.hash for result in results) != tuple(expected):
        raise RuntimeError("backtest execution did not preserve declared request order")
    if population is not None:
        population.require_complete()
    return OptionBacktestExecution(tuple(results), pl.DataFrame(rows), population)


def _replay_from_stdin() -> None:
    payload = json.load(sys.stdin)
    descriptor = payload["study"]
    study = Study(
        case_study=str(descriptor["case_study"]),
        root=Path(descriptor["root"]),
        release_root=Path(descriptor["release_root"]),
        output_root=(
            Path(descriptor["output_root"]) if descriptor["output_root"] is not None else None
        ),
        read_only=False,
        manifest=descriptor["manifest"],
    )
    study.activate(payload["execution_tier"])
    catalog = study.predictions.table(include_preview=True)
    price_cache: dict[tuple[str, int], pl.DataFrame] = {}
    replayed = []
    for row in payload["requests"]:
        selected = catalog.filter(pl.col("prediction_hash") == row["prediction_hash"])
        if selected.height != 1 or not selected.item(0, "complete"):
            raise ValueError("clean replay prediction is absent, ambiguous, or incomplete")
        prediction = selected_prediction(study, selected.row(0, named=True))
        allocation = row.get("allocation")
        warmup = strategy_warmup_periods({"strategy": {"allocation": allocation}})
        cache_key = (str(row["label"]), warmup)
        if cache_key not in price_cache:
            price_cache[cache_key] = load_backtest_prices_for(
                CASE_STUDY,
                str(row["label"]),
                split="validation",
                warmup_periods=warmup,
            )
        decisions = resolve_short_straddle_decisions(
            prediction,
            prices=price_cache[cache_key],
            signal=row["signal"],
            allocation=allocation,
        )
        replayed.append(
            {"request_name": row["request_name"], "decision_digest": value_digest(decisions)}
        )
    print("OPTION_DECISION_REPLAY=" + json.dumps(replayed, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    arguments = parser.parse_args()
    if not arguments.replay:
        parser.error("--replay is required")
    _replay_from_stdin()
