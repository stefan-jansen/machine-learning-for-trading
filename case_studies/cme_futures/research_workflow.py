"""Reader-facing model and futures-strategy workflow for CME futures."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import yaml

from case_studies.research import (
    BacktestResult,
    CandidateSet,
    DecisionArtifact,
    OfficialPopulation,
    PredictionResult,
    ResolvedModelRequest,
    Result,
    StateTransitionPolicy,
    Study,
    plan_backtests,
    run_backtests,
    run_models,
)
from case_studies.research.execution import ModelExecution
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_runner import precompute_weights
from case_studies.utils.registry import prediction_hash_from_parts
from data import load_cme_futures
from utils.modeling import load_configs
from utils.paths import REPO_ROOT

CASE_STUDY = "cme_futures"
ALL_LABELS = ("fwd_ret_5d", "fwd_ret_21d")
FRONT_CONTRACT_POSITION = 0
ROLL_POLICY = "volume_rolled_multiplicative_back_adjustment"
EXPIRY_POLICY = "continuous_front_contract_rolls_before_delivery"
# The result-protocol fields that differ between the two return horizons, and the axis the
# final cross-horizon comparison therefore spans.
HORIZON_DEPENDENT_PROTOCOL_FIELDS = ("cv", "feature_artifacts", "label_artifact")
# Every name here has to be one a notebook actually registers: `official_prediction_catalog`
# resolves each with `OfficialPopulation.one`, which raises on a name that does not exist, so a
# name declared and never written stops `12_model_analysis` rather than being ignored. The first
# two carry the case-study id, which is what the other eight case studies do; the remaining four
# keep the abbreviation their producers still use, and move when 08, 09, 10a and 10b are next run.
MODEL_POPULATION_NAMES = (
    "cme_futures-linear-validation-v1",
    "cme_futures-gbm-validation-v1",
    "cme-tabular-dl-validation-v1",
    "cme-sequence-validation-v1",
    "cme-pca-validation-v1",
    "cme-sdf-validation-v1",
)


@dataclass(frozen=True)
class FuturesPricePath:
    """Reader-facing prices and the rows that establish their roll lineage."""

    prices: pl.DataFrame
    audit: pl.DataFrame
    roll_transitions: pl.DataFrame
    expiry_rules: pl.DataFrame


@dataclass(frozen=True)
class FuturesBacktestExecution:
    results: tuple[BacktestResult, ...]
    catalog_rows: pl.DataFrame
    population: OfficialPopulation


def open_study(*, execution_tier: str, workspace: str | Path | None = None) -> Study:
    """Open canonical regeneration or an isolated reader preview."""
    if execution_tier == "canonical":
        return Study.regenerate(CASE_STUDY, release_root=REPO_ROOT)
    if execution_tier != "preview":
        raise ValueError("execution_tier must be canonical or preview")
    if workspace is None:
        raise ValueError("preview execution requires an explicit workspace")
    workspace = Path(workspace).expanduser().resolve()
    try:
        return Study.open(CASE_STUDY, workspace=workspace, release_root=REPO_ROOT)
    except ValueError as error:
        generated = tuple(
            REPO_ROOT / "case_studies" / CASE_STUDY / name
            for name in ("features", "labels", "run_log")
        )
        if "artifact bundle" not in str(error) or not all(path.is_symlink() for path in generated):
            raise
        workspace.mkdir(parents=True, exist_ok=True)
        shared_config = workspace / "config"
        if not shared_config.exists():
            shared_config.symlink_to(
                REPO_ROOT / "case_studies" / "config", target_is_directory=True
            )
        return Study(
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


def model_request_catalog(
    family: str,
    *,
    labels: Iterable[str] = ALL_LABELS,
    config_names: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Return the declared model population as visible Polars rows."""
    selected = set(config_names) if config_names is not None else None
    rows = []
    missing_by_label = {}
    for label in labels:
        configs = load_configs(CASE_STUDY, label, family)
        available = {str(config["config_name"]) for config in configs}
        missing = sorted((selected or set()) - available)
        if missing:
            missing_by_label[label] = missing
        for config in configs:
            name = str(config["config_name"])
            if selected is None or name in selected:
                rows.append({"family": family, "label": label, "config_name": name})
    if missing_by_label:
        raise ValueError(f"unknown {family!r} configurations by label: {missing_by_label}")
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


def run_model_catalog(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
) -> ModelExecution:
    """Execute every visible model request and require complete catalog output."""
    resolved = resolve_model_requests(
        study,
        request_catalog,
        execution_tier=execution_tier,
        overrides=overrides,
        preview_reductions=preview_reductions,
    )
    return run_resolved_model_requests(study, resolved)


def run_resolved_model_requests(
    study: Study,
    resolved_requests: Iterable[ResolvedModelRequest],
) -> ModelExecution:
    """Execute already-resolved requests without loading their input artifacts again."""
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


def run_official_model_catalog(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    population_name: str,
    resolved_requests: Iterable[ResolvedModelRequest] | None = None,
    supersedes: str | None = None,
) -> tuple[ModelExecution, OfficialPopulation]:
    """Snapshot and execute one complete canonical model population.

    ``supersedes`` names the population hash this run replaces, and it is a value a person
    sets after reading the registry rather than one the notebook can derive. It belongs in
    the snapshot, so passing it changes the population identity: a re-run that supplies it
    is registering a new population that records what it replaced, not re-registering the
    old one. Where a population already exists under this name and the membership has
    changed, ``OfficialPopulation.create`` refuses without it and names the hash required.
    """
    resolved = tuple(resolved_requests or ())
    if not resolved:
        resolved = resolve_model_requests(study, request_catalog, execution_tier="canonical")
    if any(request.spec["execution_tier"] != "canonical" for request in resolved):
        raise ValueError("official model populations require canonical requests")
    expected = expected_prediction_hashes(resolved)
    population = OfficialPopulation.create(
        study,
        name=population_name,
        member_kind="prediction",
        members=expected,
        supersedes=supersedes,
    )
    execution = run_resolved_model_requests(study, resolved)
    actual = tuple(prediction.hash for run in execution.runs for prediction in run.predictions)
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"model population mismatch: missing={missing}, extra={extra}")
    population.require_complete()
    return execution, population


def resolved_model_plan(resolved_requests: Iterable[ResolvedModelRequest]) -> pl.DataFrame:
    """Show the data, folds, checkpoints, and eligibility each request will use."""
    rows = []
    for request in resolved_requests:
        computation = request.spec.get("computation", request.spec)
        expected = request._context.expected_keys
        entity = next(
            (column for column in ("product", "symbol") if column in expected.columns),
            None,
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


def product_universe_table() -> pl.DataFrame:
    """Return the configured CME product groups with expiry references."""
    setup = yaml.safe_load(
        (REPO_ROOT / "case_studies" / CASE_STUDY / "config" / "setup.yaml").read_text()
    )
    rows = [
        {"sector": sector, "product": product}
        for sector, products in setup["universe"]["product_groups"].items()
        for product in products
    ]
    universe = pl.DataFrame(rows)
    return universe.join(
        _expiry_rules(universe.get_column("product").to_list()),
        on="product",
        how="left",
    ).sort("sector", "product")


def official_prediction_catalog(
    study: Study,
    population_names: Iterable[str],
) -> pl.DataFrame:
    """Return the exact complete catalog rows from declared official populations."""
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
    return catalog.sort("label", "family", "config_name", "checkpoint_kind", "checkpoint_value")


def expected_prediction_hashes(resolved_requests) -> tuple[str, ...]:
    """Project the declared checkpoint population to immutable prediction identities."""
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


def _expiry_rules(products: list[str]) -> pl.DataFrame:
    path = REPO_ROOT / "data" / "futures" / "market" / "futures_specs.yaml"
    configured = yaml.safe_load(path.read_text())["products"]
    missing = sorted(set(products) - set(configured))
    if missing:
        raise ValueError(f"products have no contract specification: {missing}")
    rows = []
    for product in products:
        spec = configured[product]
        if not spec.get("expiry_rule") or not spec.get("contract_months"):
            raise ValueError(f"{product} has an incomplete expiry specification")
        rows.append(
            {
                "product": product,
                "expiry_rule": str(spec["expiry_rule"]),
                "contract_months": ",".join(str(value) for value in spec["contract_months"]),
            }
        )
    return pl.DataFrame(rows).sort("product")


def load_futures_price_path(
    label: str,
    *,
    split: Literal["validation", "holdout"] = "validation",
    max_products: int = 0,
    products: Iterable[str] | None = None,
    warmup_periods: int = 0,
) -> FuturesPricePath:
    """Load front-contract prices while retaining the rows that prove each roll."""
    selected_products = sorted(set(products or ()))
    if max_products and selected_products:
        raise ValueError("select CME prices by max_products or products, not both")
    engine_prices = load_backtest_prices_for(
        CASE_STUDY,
        label,
        split=split,
        max_symbols=max_products,
        warmup_periods=warmup_periods,
    ).rename({"symbol": "product"})
    if selected_products:
        engine_prices = engine_prices.filter(pl.col("product").is_in(selected_products))
        loaded_products = set(engine_prices.get_column("product"))
        missing_products = sorted(set(selected_products) - loaded_products)
        if missing_products:
            raise ValueError(f"selected CME products have no backtest prices: {missing_products}")
    resolved_products = sorted(engine_prices.get_column("product").unique().to_list())
    if not resolved_products:
        raise ValueError(f"CME backtest prices for {label!r} resolved to no products")
    price_keys = engine_prices.select("product", "timestamp").unique()
    start = str(price_keys.get_column("timestamp").min())[:10]
    end = str(price_keys.get_column("timestamp").max())[:10]
    loaded_audit = load_cme_futures(
        products=resolved_products,
        start_date=start,
        end_date=end,
    )
    audit = (
        cast(pl.DataFrame, loaded_audit.collect())
        if isinstance(loaded_audit, pl.LazyFrame)
        else loaded_audit
    )
    audit = audit.rename({"session_date": "timestamp", "tenor": "position"})
    required = {
        "product",
        "position",
        "timestamp",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "raw_close",
        "cum_ratio",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"CME price source is missing columns: {sorted(missing)}")
    audit = audit.filter(pl.col("position") == FRONT_CONTRACT_POSITION).with_columns(
        pl.col("timestamp").cast(price_keys.schema["timestamp"]),
        pl.col("product").cast(price_keys.schema["product"]),
    )
    audit = audit.join(price_keys, on=["product", "timestamp"], how="semi").sort(
        "product", "timestamp"
    )
    if audit.n_unique(["product", "position", "timestamp"]) != audit.height:
        raise ValueError("front-contract roll audit contains duplicate keys")
    missing_audit = price_keys.join(
        audit.select("product", "timestamp"), on=["product", "timestamp"], how="anti"
    )
    if not missing_audit.is_empty():
        raise ValueError("backtest price keys are missing from the front-contract roll audit")
    ratio = pl.col("cum_ratio").cast(pl.Float64)
    invalid_ratio = audit.filter(
        ratio.is_null() | ratio.is_nan() | ratio.is_infinite() | (ratio <= 0)
    )
    if not invalid_ratio.is_empty():
        raise ValueError("front-contract roll ratios must be finite positive values")
    scale = pl.max_horizontal(pl.col("adj_close").abs(), pl.lit(1.0))
    inconsistent = audit.filter(
        (pl.col("adj_close") - pl.col("raw_close") * pl.col("cum_ratio")).abs() > scale * 1e-10
    )
    if not inconsistent.is_empty():
        raise ValueError("roll-adjusted prices do not equal raw prices times the roll ratio")
    previous_ratio = pl.col("cum_ratio").shift(1).over("product")
    audit = audit.with_columns(
        previous_ratio.alias("previous_cum_ratio"),
        (previous_ratio.is_not_null() & (pl.col("cum_ratio") != previous_ratio)).alias(
            "roll_transition"
        ),
    )
    transitions = audit.filter(pl.col("roll_transition")).select(
        "product",
        "timestamp",
        "raw_close",
        "adj_close",
        "previous_cum_ratio",
        "cum_ratio",
        (pl.col("cum_ratio") / pl.col("previous_cum_ratio")).alias("roll_adjustment_factor"),
    )
    return FuturesPricePath(
        prices=engine_prices.sort("timestamp", "product"),
        audit=audit,
        roll_transitions=transitions,
        expiry_rules=_expiry_rules(resolved_products),
    )


def _resolved_signal(signal: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(signal)
    resolved.setdefault("long_short", get_backtest_config(CASE_STUDY).long_short)
    return resolved


def _resolved_allocation(allocation: dict[str, Any] | None) -> dict[str, Any] | None:
    if allocation is None:
        return None
    resolved = dict(allocation)
    resolved.setdefault("long_short", get_backtest_config(CASE_STUDY).long_short)
    return resolved


def resolve_product_weights(
    prediction: PredictionResult,
    *,
    prices: pl.DataFrame,
    signal: dict[str, Any],
    allocation: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    costs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Resolve built-in research decisions and return canonical product weights."""
    if "product" not in prices.columns or "symbol" in prices.columns:
        raise ValueError("reader-facing CME prices require product and cannot contain symbol")
    study = prediction.study
    signal = _resolved_signal(signal)
    allocation = _resolved_allocation(allocation)
    unresolved = study.strategy(
        prediction=prediction,
        signal=signal,
        allocation=allocation,
        risk=risk,
        costs=costs,
    )
    spec = unresolved.resolve(prices=prices)
    engine_prices = prices.rename({"product": "symbol"})
    weights = precompute_weights(
        prediction.load(),
        spec,
        engine_prices,
        label=unresolved.label,
        case_study=CASE_STUDY,
        prediction_hash=prediction.hash,
    ).rename({"symbol": "product"})
    prediction_frame = prediction.load()
    entity_columns = [
        column for column in ("symbol", "product") if column in prediction_frame.columns
    ]
    if len(entity_columns) != 1:
        raise ValueError("CME predictions require exactly one entity key: symbol or product")
    fold_columns = [column for column in ("fold", "fold_id") if column in prediction_frame.columns]
    if len(fold_columns) != 1:
        raise ValueError("CME predictions require exactly one fold key: fold or fold_id")
    entity = entity_columns[0]
    fold = fold_columns[0]
    if entity == "symbol":
        prediction_frame = prediction_frame.rename({"symbol": "product"})
    prediction_frame = prediction_frame.with_columns(
        pl.col("product").cast(weights.schema["product"]),
        pl.col("timestamp").cast(weights.schema["timestamp"]),
    )
    fold_by_time = (
        prediction_frame.select("timestamp", fold)
        .unique()
        .group_by("timestamp")
        .agg(pl.col(fold).n_unique().alias("n_folds"), pl.col(fold).first().alias("fold"))
    )
    if fold_by_time.filter(pl.col("n_folds") != 1).height:
        raise ValueError("each CME decision timestamp must belong to exactly one fold")
    eligible = prediction_frame.select("product", "timestamp").unique()
    missing = weights.select("product", "timestamp").join(
        eligible, on=["product", "timestamp"], how="anti"
    )
    if not missing.is_empty():
        raise ValueError("resolved CME decisions contain keys outside prediction eligibility")
    return (
        weights.join(fold_by_time.select("timestamp", "fold"), on="timestamp", how="left")
        .select("product", "timestamp", "weight", "fold")
        .sort("timestamp", "product")
    )


def publish_product_weights(
    prediction: PredictionResult,
    *,
    prices: pl.DataFrame,
    signal: dict[str, Any],
    allocation: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    costs: dict[str, Any] | None = None,
    canonical: bool = False,
) -> DecisionArtifact:
    """Publish validated CME weights with product, roll, expiry, and fold lineage."""
    signal = _resolved_signal(signal)
    allocation = _resolved_allocation(allocation)
    weights = resolve_product_weights(
        prediction,
        prices=prices,
        signal=signal,
        allocation=allocation,
        risk=risk,
        costs=costs,
    )
    source_identity: dict[str, Any] | None = None
    if canonical:
        replay = resolve_product_weights(
            prediction,
            prices=prices,
            signal=signal,
            allocation=allocation,
            risk=risk,
            costs=costs,
        )
        if value_digest(replay) != value_digest(weights):
            raise RuntimeError("canonical CME decision replay was not deterministic")
        source_identity = {
            "module": "case_studies.cme_futures.research_workflow",
            "source_digest": hashlib.sha256(
                inspect.getsource(resolve_product_weights).encode()
            ).hexdigest(),
            "declared_inputs": {
                "prediction_hashes": [prediction.hash],
                "prices": value_digest(prices),
            },
            "determinism": {"deterministic": True},
            "clean_replay_digest": value_digest(replay),
        }
    return DecisionArtifact.publish(
        prediction.study,
        kind="target_weights",
        decisions=weights,
        prediction_hashes=[prediction.hash],
        parameters={
            "signal": signal,
            "allocation": allocation,
            "risk": risk,
            "costs": costs,
            "cadence": "7d",
            "contract_position": FRONT_CONTRACT_POSITION,
            "roll_policy": ROLL_POLICY,
            "expiry_policy": EXPIRY_POLICY,
        },
        source_identity=source_identity,
        state_transition_policy=StateTransitionPolicy(
            fold_boundary="liquidate",
            temporal_gap="continue",
        ),
        canonical=canonical,
    )


def selected_prediction(study: Study, catalog_row: dict[str, Any]) -> PredictionResult:
    """Resolve one selected catalog row without exposing its hash to notebook control flow."""
    result = Result.open(
        study,
        str(catalog_row["prediction_hash"]),
        include_preview=catalog_row.get("execution_tier") == "preview",
    )
    if not isinstance(result, PredictionResult):
        raise ValueError("selected catalog row does not identify a prediction")
    return result


def strategy_request_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build visible request rows while preserving each nested strategy dictionary."""
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
    population_name: str,
) -> FuturesBacktestExecution:
    """Resolve, snapshot, and execute visible canonical futures strategy requests."""
    required = {"request_name", "prediction_hash", "label", "signal"}
    missing = required - set(requests.columns)
    if missing:
        raise ValueError(f"strategy requests are missing columns: {sorted(missing)}")
    if requests.is_empty() or requests.get_column("request_name").n_unique() != requests.height:
        raise ValueError("strategy request names must be non-empty and unique")
    catalog = study.predictions.table()
    price_cache: dict[tuple[str, int], FuturesPricePath] = {}
    prepared = []
    expected = []
    for row in requests.iter_rows(named=True):
        selected = catalog.filter(pl.col("prediction_hash") == row["prediction_hash"])
        if selected.height != 1 or not selected.item(0, "complete"):
            raise ValueError(
                f"prediction {row['prediction_hash']!r} is absent, ambiguous, or incomplete"
            )
        if selected.item(0, "label") != row["label"]:
            raise ValueError("strategy request label does not match its prediction catalog row")
        prediction = selected_prediction(study, selected.row(0, named=True))
        allocation = _resolved_allocation(row.get("allocation"))
        risk = row.get("risk")
        costs = row.get("costs")
        warmup = strategy_warmup_periods({"strategy": {"allocation": allocation}})
        cache_key = (str(row["label"]), warmup)
        if cache_key not in price_cache:
            price_cache[cache_key] = load_futures_price_path(
                str(row["label"]),
                split="validation",
                warmup_periods=warmup,
            )
        price_path = price_cache[cache_key]
        signal = _resolved_signal(row["signal"])
        decision = publish_product_weights(
            prediction,
            prices=price_path.prices,
            signal=signal,
            allocation=allocation,
            risk=risk,
            costs=costs,
            canonical=True,
        )
        plan = plan_backtests(
            study,
            predictions=selected,
            signal=signal,
            decision=decision,
            prices=price_path.prices,
            allocation=allocation,
            risk=risk,
            costs=costs,
            chapter=row.get("chapter"),
        )
        if len(plan.members) != 1:
            raise RuntimeError("one futures strategy request must resolve to one backtest")
        expected_hash = plan.expected_hashes[0]
        expected.append(expected_hash)
        prepared.append((row, selected, price_path, signal, allocation, decision, expected_hash))
    if len(expected) != len(set(expected)):
        raise ValueError("strategy requests resolve to duplicate backtest identities")
    population = OfficialPopulation.create(
        study,
        name=population_name,
        member_kind="backtest",
        members=expected,
    )
    results = []
    rows = []
    for row, selected, price_path, signal, allocation, decision, expected_hash in prepared:
        result = run_backtests(
            study,
            predictions=selected,
            signal=signal,
            prices=price_path.prices,
            allocation=allocation,
            risk=row.get("risk"),
            costs=row.get("costs"),
            chapter=row.get("chapter"),
            decision=decision,
        ).results[0]
        if result.hash != expected_hash:
            raise RuntimeError(f"backtest identity changed: {expected_hash} -> {result.hash}")
        results.append(result)
        rows.append(
            {
                "request_name": row["request_name"],
                "label": row["label"],
                "prediction_hash": row["prediction_hash"],
                "decision_hash": decision.hash,
                "backtest_hash": result.hash,
                "complete": result.complete,
            }
        )
    if tuple(result.hash for result in results) != tuple(expected):
        raise RuntimeError("backtest execution did not preserve declared request order")
    population.require_complete()
    return FuturesBacktestExecution(tuple(results), pl.DataFrame(rows), population)


def create_label_candidate_sets(
    study: Study,
    execution: FuturesBacktestExecution,
    *,
    name_prefix: str,
) -> dict[str, CandidateSet]:
    """Create one immutable comparable backtest set per label."""
    labels = execution.catalog_rows.get_column("label").unique().sort().to_list()
    output = {}
    for label in labels:
        hashes = execution.catalog_rows.filter(pl.col("label") == label).get_column("backtest_hash")
        members_by_hash = {result.hash: result for result in execution.results}
        members = [members_by_hash[value] for value in hashes]
        output[label] = CandidateSet.create(
            study,
            f"{name_prefix}-{label}-v1",
            members,
        )
    return output


def shortlist_signal_configurations(
    study: Study,
    *,
    label: str,
    limit: int,
) -> tuple[BacktestResult, ...]:
    """Select the strongest signal result for each distinct model configuration."""
    candidates = CandidateSet.one(study, name=f"cme-signal-{label}-v1")
    selected = []
    configurations = set()
    for result in candidates.ranked_validation_sharpe():
        assert isinstance(result, BacktestResult)
        training = result.lineage()["training_spec"]
        key = (training["family"], training.get("config_name"))
        if key in configurations:
            continue
        configurations.add(key)
        selected.append(result)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(
            f"signal population has {len(selected)} distinct configurations, expected {limit}"
        )
    return tuple(selected)


def pre_overlay_candidate_set(study: Study, *, label: str) -> CandidateSet:
    """Return the immutable union of signal and allocation validation results."""
    signal = CandidateSet.one(study, name=f"cme-signal-{label}-v1")
    allocation = CandidateSet.one(study, name=f"cme-allocation-{label}-v1")
    members = [Result.open(study, value) for value in (*signal.members, *allocation.members)]
    return CandidateSet.create(study, f"cme-pre-overlay-{label}-v1", members)


def final_validation_candidate_set(study: Study, *, label: str) -> CandidateSet:
    """Return the selection pool across signal, allocation, and risk-overlay stages."""
    pre_overlay = pre_overlay_candidate_set(study, label=label)
    risk = CandidateSet.one(study, name=f"cme-risk-{label}-v1")
    members = [Result.open(study, value) for value in (*pre_overlay.members, *risk.members)]
    return CandidateSet.create(study, f"cme-final-validation-{label}-v1", members)


def final_selection_candidate_set(study: Study) -> CandidateSet:
    """Return the one immutable pool the case-study configuration is selected from.

    The funnel runs per label through the risk overlay. The object of selection is one
    case-study configuration, so the per-label pools are compared as a single set and the
    label comes from the row that wins rather than being fixed in advance.

    The return horizon is the axis the comparison spans, and naming it is what makes the
    comparison legible: the two pools carry different label artifacts, different derived
    features and a different purge interval. Everything else must match, and the candidate
    set refuses a member whose split or execution tier disagrees.
    """
    members = []
    for label in ALL_LABELS:
        pool = final_validation_candidate_set(study, label=label)
        members.extend(Result.open(study, value) for value in pool.members)
    return CandidateSet.create(
        study,
        "cme-final-selection-v1",
        members,
        comparison_contract={"comparable_fields": list(HORIZON_DEPENDENT_PROTOCOL_FIELDS)},
    )


def selection_catalog(study: Study, candidates: CandidateSet) -> pl.DataFrame:
    """Return the registered catalog rows for one immutable selection pool."""
    table = study.backtests.table().filter(pl.col("backtest_hash").is_in(candidates.members))
    if table.height != len(candidates.members):
        raise ValueError("the backtest catalog does not describe every candidate")
    return table.select(
        "label",
        "family",
        "config_name",
        "checkpoint_kind",
        "checkpoint_value",
        "stage",
        "signal_method",
        "allocation_method",
        "risk_method",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "avg_turnover",
        "complete",
        "training_hash",
        "prediction_hash",
        "backtest_hash",
    ).sort("sharpe", "backtest_hash", descending=[True, False])


def holdout_evidence(study: Study) -> pl.DataFrame:
    """Return the research lock and the one holdout evaluation it authorizes, if any.

    The lock is what makes the holdout usable once: it records the candidate set, the
    selected validation backtest and the retraining contract before any holdout artifact
    exists. Reading it here is how the analysis shows which configuration the holdout ran
    on without being able to choose a different one.
    """
    database = study.root / "run_log" / "registry.db"
    if not database.is_file():
        return pl.DataFrame()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        names = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if "research_locks" not in names:
            return pl.DataFrame()
        rows = db.execute(
            "SELECT l.lock_hash, l.state, l.lock_json, e.holdout_training_hash, "
            "e.holdout_prediction_hash, e.holdout_backtest_hash, e.evaluated_at "
            "FROM research_locks l "
            "LEFT JOIN holdout_evaluations e ON e.lock_hash = l.lock_hash"
        ).fetchall()
    records = []
    for lock_hash, state, lock_json, training, prediction, backtest, evaluated_at in rows:
        lock = json.loads(lock_json)
        records.append(
            {
                "lock_hash": lock_hash,
                "state": state,
                "label": lock.get("label"),
                "checkpoint_kind": lock.get("checkpoint_kind"),
                "checkpoint_value": lock.get("checkpoint_value"),
                "candidate_set_hash": lock.get("candidate_set_hash"),
                "validation_backtest_hash": lock.get("validation_backtest_hash"),
                "holdout_training_hash": training,
                "holdout_prediction_hash": prediction,
                "holdout_backtest_hash": backtest,
                "evaluated_at": evaluated_at,
            }
        )
    return pl.DataFrame(records)
