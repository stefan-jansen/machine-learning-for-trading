from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    load_contract_specs_from_yaml,
    load_futures_market_contract,
)
from case_studies.utils.backtest_presets import (
    build_backtest_spec,
    serializable_backtest_spec,
    strategy_view,
)
from case_studies.utils.backtest_runner import precompute_weights, run_backtest
from case_studies.utils.conformal import (
    compute_holdout_conformal_widths,
    ensure_conformal_calibration_identity,
    holdout_conformal_embargo_steps,
)
from case_studies.utils.registry import backtest_hash_from_parts, canonical_json, compute_hash
from case_studies.utils.sweep_config import get_allocator_lookback

from .contracts import ExecutionTier
from .results import BacktestResult, PredictionResult, Result

if TYPE_CHECKING:
    from .decisions import DecisionArtifact
    from .workspace import Study


_MOMENT_ALLOCATORS = {"inverse_vol", "risk_parity", "hrp", "mvo", "mvo_ledoit_wolf"}


def _cadence_delta(value: str) -> timedelta:
    match = re.fullmatch(r"([1-9][0-9]*)(us|ms|s|m|h|d|w)", value.strip().lower())
    if match is None:
        raise ValueError(f"decision cadence must be a compact duration such as '8h', got {value!r}")
    amount = int(match.group(1))
    unit = match.group(2)
    keyword = {
        "us": "microseconds",
        "ms": "milliseconds",
        "s": "seconds",
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
    }[unit]
    return timedelta(**{keyword: amount})


def _last_at_or_before(timestamps: pl.Series, moment: object) -> object | None:
    """The weight row in force at a moment on the observation grid."""
    earlier = timestamps.filter(timestamps <= moment)
    return earlier[-1] if earlier.len() else None


def apply_state_transition_policy(
    weights: pl.DataFrame,
    *,
    policy: dict[str, str],
    cadence: str | None,
    price_keys: pl.DataFrame | None = None,
    timeline: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Preserve targets while making fold and temporal state resets executable.

    `cadence` describes the timeline the policy is written against - the observation grid, not
    the weight frame. For a decision artifact the two coincide, so `timeline` defaults to the
    weights' own timestamps. For generated weights they do not: `precompute_weights` thins to
    the non-overlapping rebalance schedule, so a label with `rebalance_step` 3 yields weights
    24h apart under a declared 8h cadence and every ordinary rebalance would read as a gap.
    Callers in that position pass the observation timeline explicitly.
    """
    required = {"symbol", "timestamp", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"decision weights are missing columns: {sorted(missing)}")
    if weights.is_empty():
        return weights
    source = weights if timeline is None else timeline
    fold_column = "fold" if "fold" in source.columns else None
    timeline_columns = ["timestamp", *([fold_column] if fold_column else [])]
    timeline = source.select(*timeline_columns).unique().sort("timestamp")
    if fold_column and timeline.n_unique("timestamp") != timeline.height:
        raise ValueError("each decision timestamp must belong to exactly one fold")
    weight_timestamps = weights.get_column("timestamp").unique().sort()

    temporal_action = policy.get("temporal_gap", "continue")
    expected = None
    if temporal_action != "continue":
        if cadence is None:
            raise ValueError("non-continuing temporal-gap policy requires decision cadence")
        expected = _cadence_delta(cadence)

    transition_at: set[object] = set()
    flat_frames: list[pl.DataFrame] = []
    # A fold boundary and a temporal gap can resolve to the same off-grid moment. The on-grid
    # path dedupes through `transition_at`; without this the flat-row path would append the row
    # twice and `_target_weights_by_timestamp` rejects duplicate (symbol, timestamp) pairs, so
    # the reset would abort the backtest instead of executing.
    flat_moments: set[object] = set()
    price_timestamps = (
        set(price_keys.get_column("timestamp").unique().to_list())
        if price_keys is not None
        else set()
    )
    weight_moments = set(weight_timestamps.to_list())

    def _flat_row_at(moment: object, carried_from: object) -> pl.DataFrame:
        return weights.filter(pl.col("timestamp") == carried_from).with_columns(
            pl.lit(moment).cast(weights.schema["timestamp"]).alias("timestamp"),
            pl.lit(0.0).cast(weights.schema["weight"]).alias("weight"),
        )

    def _mark(moment: object, *, previous_moment: object, reason: str) -> None:
        """Mark a declared reset at the moment it is declared for.

        If the moment is on the weight grid, marking the row is enough. If it is not - which
        happens whenever the weight frame is thinned relative to the observation grid - a
        zero-weight row is inserted at that exact moment, carrying the position that was in
        force. Snapping the mark forward to the next weight row instead would carry the old
        state across the boundary and then collapse the liquidation into an ordinary rebalance
        on the same bar, paying a round trip for no change in exposure.
        """
        if moment in weight_moments:
            transition_at.add(moment)
            return
        held_at = _last_at_or_before(weight_timestamps, previous_moment)
        if held_at is None or moment not in price_timestamps:
            raise ValueError(
                f"declared {reason} at {moment} cannot be represented on the weight grid"
            )
        if moment in flat_moments:
            return
        flat_moments.add(moment)
        flat_frames.append(_flat_row_at(moment, held_at))

    rows = timeline.iter_rows(named=True)
    previous = next(rows, None)
    for current in rows:
        assert previous is not None
        fold_changed = fold_column is not None and current[fold_column] != previous[fold_column]
        if fold_changed and policy.get("fold_boundary", "continue") != "continue":
            _mark(current["timestamp"], previous_moment=previous["timestamp"], reason="fold reset")
        if expected is not None and current["timestamp"] - previous["timestamp"] > expected:
            first_missing = previous["timestamp"] + expected
            held_at = _last_at_or_before(weight_timestamps, previous["timestamp"])
            if first_missing in price_timestamps and held_at is not None:
                if first_missing not in flat_moments:
                    flat_moments.add(first_missing)
                    flat_frames.append(_flat_row_at(first_missing, held_at))
            else:
                _mark(
                    current["timestamp"],
                    previous_moment=previous["timestamp"],
                    reason="temporal-gap reset",
                )
        previous = current
    transition_values = pl.Series(
        "_transition_timestamp",
        list(transition_at),
        dtype=weights.schema["timestamp"],
    )
    result = weights.with_columns(
        pl.col("timestamp").is_in(transition_values.implode()).alias("_state_transition")
    )
    if flat_frames:
        result = pl.concat(
            [
                result,
                *[
                    frame.with_columns(pl.lit(True).alias("_state_transition"))
                    for frame in flat_frames
                ],
            ]
        )
    return result.sort("timestamp", "symbol")


def _date_string(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"expected a date-like timestamp, got {type(value).__name__}")


def strategy_warmup_periods(strategy_spec: dict[str, Any]) -> int:
    strategy = strategy_spec.get("strategy", strategy_spec)
    allocation = strategy.get("allocation") or {}
    method = allocation.get("method")
    if method not in _MOMENT_ALLOCATORS:
        return 0
    key = "lookback" if method in {"mvo", "mvo_ledoit_wolf"} else "vol_window"
    value = allocation.get(key, allocation.get("lookback"))
    if value is None:
        raise ValueError(f"rolling allocator {method!r} has no resolved {key}")
    return int(value)


@dataclass(frozen=True)
class Strategy:
    study: Study
    prediction: PredictionResult
    signal: dict[str, Any]
    decision: DecisionArtifact | None = None
    allocation: dict[str, Any] | None = None
    risk: dict[str, Any] | None = None
    costs: dict[str, Any] | None = None
    chapter: str | None = None
    execution_mode: str | None = None
    min_weight_change: float | None = None
    min_trade_value: float | None = None

    def _active_lock_record(self) -> dict[str, Any]:
        db_path = self.study.root / "run_log" / "registry.db"
        with closing(sqlite3.connect(db_path)) as db:
            row = db.execute(
                "SELECT lock_json FROM research_locks WHERE state = 'LOCKED' LIMIT 1"
            ).fetchone()
        if row is None:
            raise ValueError("holdout strategy execution requires a LOCKED research lock")
        return json.loads(row[0])

    def __post_init__(self) -> None:
        if self.prediction.study != self.study:
            raise ValueError("prediction belongs to another study")
        if not self.prediction.complete:
            raise ValueError("partial predictions cannot enter strategy execution")
        if self.decision is not None:
            if self.decision.study != self.study:
                raise ValueError("decision artifact belongs to another study")
            if tuple(self.decision.spec["prediction_hashes"]) != (self.prediction.hash,):
                raise ValueError("decision artifact must declare exactly the selected prediction")
        prediction_record = self.prediction.registry_record()
        split = prediction_record["split"]
        if split == "validation":
            return
        if split != "holdout" or self.prediction.execution_tier != "canonical":
            raise ValueError("strategy execution requires validation or locked holdout predictions")
        lock_record = self._active_lock_record()
        matches_lock = (
            prediction_record["training_hash"] == lock_record.get("holdout_training_hash")
            and prediction_record["checkpoint_kind"] == lock_record.get("checkpoint_kind")
            and prediction_record["checkpoint_value"] == lock_record.get("checkpoint_value")
        )
        if not matches_lock:
            raise ValueError("holdout prediction does not match the locked retraining contract")

    @property
    def split(self) -> Literal["validation", "holdout"]:
        split = self.prediction.registry_record()["split"]
        if split not in {"validation", "holdout"}:
            raise ValueError(f"unsupported prediction split {split!r}")
        return split

    @classmethod
    def from_request(cls, study: Study, request: dict[str, Any]) -> Strategy:
        supported = {
            "prediction",
            "signal",
            "decision",
            "allocation",
            "risk",
            "costs",
            "chapter",
            "execution_mode",
            "min_weight_change",
            "min_trade_value",
        }
        unknown = set(request) - supported
        if unknown:
            raise ValueError(f"unsupported strategy request fields: {sorted(unknown)}")
        prediction = request.get("prediction")
        if not isinstance(prediction, PredictionResult):
            raise TypeError("strategy request requires one PredictionResult")
        decision = request.get("decision")
        if decision is not None:
            from .decisions import DecisionArtifact

            if not isinstance(decision, DecisionArtifact):
                raise TypeError("strategy request decision must be a DecisionArtifact")
        return cls(
            study=study,
            prediction=prediction,
            signal=deepcopy(request.get("signal") or {}),
            decision=decision,
            allocation=deepcopy(request.get("allocation")),
            risk=deepcopy(request.get("risk")),
            costs=deepcopy(request.get("costs")),
            chapter=request.get("chapter"),
            execution_mode=request.get("execution_mode"),
            min_weight_change=request.get("min_weight_change"),
            min_trade_value=request.get("min_trade_value"),
        )

    @property
    def label(self) -> str:
        return str(self.prediction.lineage()["training_spec"]["label"])

    def _resolved_allocation(self) -> dict[str, Any] | None:
        if self.allocation is None:
            return None
        allocation = deepcopy(self.allocation)
        method = allocation.get("method")
        if method in _MOMENT_ALLOCATORS:
            if method in {"mvo", "mvo_ledoit_wolf"}:
                if "lookback" not in allocation:
                    allocation["lookback"] = get_allocator_lookback(self.study.case_study)
            elif not any(key in allocation for key in ("vol_window", "lookback")):
                allocation["vol_window"] = get_allocator_lookback(self.study.case_study)
        return allocation

    def _warmup_periods(self) -> int:
        allocation = self._resolved_allocation()
        return strategy_warmup_periods({"allocation": allocation}) if allocation else 0

    def _engine_prices(
        self,
        prices: pl.DataFrame,
        *,
        reader_supplied: bool,
    ) -> pl.DataFrame:
        if self.study.case_study != "cme_futures":
            return prices
        if "product" in prices.columns and "symbol" in prices.columns:
            raise ValueError("CME prices cannot contain both product and symbol")
        if "product" in prices.columns:
            return prices.rename({"product": "symbol"})
        if reader_supplied or "symbol" not in prices.columns:
            raise ValueError("CME prices require the canonical product entity key")
        return prices

    def resolve(self, *, prices: pl.DataFrame | None = None) -> dict[str, Any]:
        self.study.activate(ExecutionTier(self.prediction.execution_tier))
        if self.split == "holdout" and prices is not None:
            raise ValueError("locked holdout strategy must load canonical holdout prices")
        case_config = get_backtest_config(self.study.case_study)
        reader_supplied = prices is not None
        resolved_prices = prices
        if not reader_supplied:
            resolved_prices = load_backtest_prices_for(
                self.study.case_study,
                self.label,
                split=self.split,
                warmup_periods=self._warmup_periods(),
            )
        assert resolved_prices is not None
        resolved_prices = self._engine_prices(
            resolved_prices,
            reader_supplied=reader_supplied,
        )
        contract_specs = (
            load_contract_specs_from_yaml() if self.study.case_study == "cme_futures" else None
        )
        return self._build_spec(resolved_prices, case_config, contract_specs)

    def _build_spec(self, prices, case_config, contract_specs) -> dict[str, Any]:
        spec = build_backtest_spec(
            self.study.case_study,
            case_config,
            prices=prices,
            prediction_hash=self.prediction.hash,
            initial_cash=case_config.initial_cash,
            signal=self.signal,
            allocation=self._resolved_allocation(),
            risk=self.risk,
            costs=self.costs,
            chapter=self.chapter,
            execution_mode=self.execution_mode,
            min_weight_change=self.min_weight_change,
            min_trade_value=self.min_trade_value,
        )
        spec["identity_version"] = 2
        spec["execution_tier"] = self.prediction.execution_tier
        spec["input_identity"] = {"prices": value_digest(prices)}
        if self.study.case_study == "sp500_options" and self.label == "ret_to_expiry":
            from case_studies.sp500_options._htm_backtest import (
                option_accounting_parameters,
                option_data_paths,
                option_source_identity,
            )

            if self.risk is not None:
                raise ValueError("the specialized option path does not support risk overlays")
            if self.costs is not None:
                raise ValueError(
                    "option cost variants use signal.option_spread_fraction; generic costs are unsupported"
                )

            labels_dir, raw_options_dir = option_data_paths()
            option_inputs = option_source_identity(labels_dir, raw_options_dir)
            spec["input_identity"]["option_contract_returns"] = option_inputs["contract_returns"]
            spec["input_identity"]["option_lifecycle"] = compute_hash(
                canonical_json(option_inputs["raw_lifecycle"])
            )
            accounting = option_accounting_parameters(self.signal)
            if (
                self.decision is not None
                and self.decision.kind == "short_straddles"
                and accounting["exit_at_max_days"] is not None
            ):
                raise ValueError("hold-to-expiry decisions cannot declare an earlier option exit")
            spec["options_market"] = {
                "decision_timestamp": "feature_session_close",
                "entry": "next_session_close",
                "position": "short_atm_call_and_put",
                "marking": "paired_daily_midpoint",
                "settlement": accounting["settlement"],
                "hedge": "retained_underlying_delta_with_threshold",
            }
            spec["options_accounting"] = accounting
        if self.study.case_study == "crypto_perps_funding":
            funding_rates = self._funding_rates(prices)
            assert funding_rates is not None
            spec["input_identity"]["funding_rates"] = value_digest(funding_rates)
            spec["economic_cashflows"] = {"funding": "position_signed_before_same_timestamp_fills"}
        if self.decision is not None:
            spec["decision_artifact"] = {
                "hash": self.decision.hash,
                "kind": self.decision.kind,
                "decision_keys": self.decision.spec["decision_keys"],
                "parameters": self.decision.spec["parameters"],
                "artifact_digest": self.decision.spec["artifact_digest"],
                "canonical": self.decision.canonical,
                "source_identity": self.decision.spec["source_identity"],
                "state_transition_policy": self.decision.spec["state_transition_policy"],
            }
            if self.decision.kind == "short_straddles":
                spec["decision_artifact"]["decision_keys"] = self.decision.spec["decision_keys"]
                spec["decision_artifact"]["parameters"] = self.decision.spec["parameters"]
            decision_weights = self._decision_weights(prices)
            assert decision_weights is not None
            spec["backtest_config"]["account"]["allow_short_selling"] = bool(
                decision_weights.filter(pl.col("weight") < 0).height
            )
        if contract_specs is not None:
            serialized = {
                symbol: asdict(contract_spec) for symbol, contract_spec in contract_specs.items()
            }
            spec["input_identity"]["contract_specs"] = compute_hash(canonical_json(serialized))
            products = prices.get_column("symbol").unique().sort().to_list()
            futures_market = load_futures_market_contract(products)
            spec["input_identity"]["futures_market"] = compute_hash(canonical_json(futures_market))
            spec["futures_market"] = futures_market
            spec["entity_contract"] = {
                "reader_key": "product",
                "engine_key": "symbol",
                "mapping": "one_to_one_at_backtest_boundary",
            }
        resolved = ensure_conformal_calibration_identity(spec)
        if self.split == "holdout":
            from .lifecycle import _locked_strategy_projection

            locked = self._active_lock_record()["strategy_spec"]
            if _locked_strategy_projection(resolved) != _locked_strategy_projection(locked):
                raise ValueError("holdout strategy does not match the locked validation strategy")
        return resolved

    def _funding_rates(self, prices: pl.DataFrame) -> pl.DataFrame | None:
        if self.study.case_study != "crypto_perps_funding":
            return None
        from case_studies.crypto_perps_funding.funding_data import load_funding_rates

        timestamp_dtype = prices.schema["timestamp"]
        price_keys = prices.select("symbol", "timestamp").unique()
        symbols = price_keys.get_column("symbol").unique().to_list()
        start = _date_string(price_keys.get_column("timestamp").min())
        end = _date_string(price_keys.get_column("timestamp").max())
        funding = load_funding_rates(symbols=symbols, start_date=start, end_date=end)
        funding = funding.with_columns(
            pl.col("symbol").cast(price_keys.schema["symbol"]),
            pl.col("timestamp").cast(timestamp_dtype),
        ).join(price_keys, on=["symbol", "timestamp"], how="semi")
        if funding.is_empty():
            raise ValueError("crypto backtest resolved no official funding settlements")
        return funding.sort("timestamp", "symbol")

    def _decision_weights(self, prices: pl.DataFrame) -> pl.DataFrame | None:
        if self.decision is None:
            return None
        decisions = self.decision.load()
        decision_keys = tuple(self.decision.spec.get("decision_keys") or ())
        if len(decision_keys) != 2 or decision_keys[1] != "timestamp":
            raise ValueError("decision artifact has an invalid key contract")
        entity_key = decision_keys[0]
        if entity_key not in {"symbol", "product"}:
            raise ValueError(f"unsupported decision entity key {entity_key!r}")
        if self.study.case_study == "cme_futures" and entity_key != "product":
            raise ValueError("CME futures decisions require the canonical product entity key")
        if self.study.case_study != "cme_futures" and entity_key != "symbol":
            raise ValueError(
                f"{self.study.case_study} decisions require the canonical symbol entity key"
            )
        fold_columns = [column for column in ("fold", "fold_id") if column in decisions.columns]
        if len(fold_columns) > 1:
            raise ValueError("decision artifact cannot contain both fold and fold_id")
        fold_column = fold_columns[0] if fold_columns else None
        selected_fold = [fold_column] if fold_column else []
        if self.decision.kind == "target_weights":
            weights = decisions.select(entity_key, "timestamp", "weight", *selected_fold)
        elif self.decision.kind == "target_positions":
            weights = decisions.select(
                entity_key, "timestamp", "position", *selected_fold
            ).with_columns(pl.col("position").abs().sum().over("timestamp").alias("gross"))
            weights = weights.with_columns(
                pl.when(pl.col("gross") > 0)
                .then(pl.col("position") / pl.col("gross"))
                .otherwise(0.0)
                .alias("weight")
            ).select(entity_key, "timestamp", "weight", *selected_fold)
        elif self.decision.kind == "short_straddles":
            if self.study.case_study != "sp500_options" or self.label != "ret_to_expiry":
                raise ValueError("short-straddle decisions require sp500_options and ret_to_expiry")
            parameters = self.decision.spec.get("parameters") or {}
            expected = {
                "decision_cadence": "weekly_friday",
                "entry_policy": "next_session_close",
                "exit_policy": "hold_to_expiry",
                "settlement_policy": "cash_intrinsic_at_expiration",
                "hedge_policy": "retained_underlying_delta_with_threshold",
            }
            mismatched = {
                key: (parameters.get(key), value)
                for key, value in expected.items()
                if parameters.get(key) != value
            }
            if mismatched:
                raise ValueError(f"short-straddle decision parameters are invalid: {mismatched}")
            weights = decisions
        else:
            raise ValueError("canonical backtesting does not support order decision artifacts")
        if entity_key == "product":
            weights = weights.rename({"product": "symbol"})
        if fold_column == "fold_id":
            weights = weights.rename({"fold_id": "fold"})
        price_keys = prices.select("symbol", "timestamp").unique()
        weights = weights.with_columns(
            pl.col("symbol").cast(price_keys.schema["symbol"]),
            pl.col("timestamp").cast(price_keys.schema["timestamp"]),
        )
        policy = self.decision.spec.get("state_transition_policy")
        if policy is not None:
            weights = apply_state_transition_policy(
                weights,
                policy=policy,
                cadence=self.decision.spec.get("parameters", {}).get("cadence"),
                price_keys=price_keys,
            )
        missing = weights.join(price_keys, on=["symbol", "timestamp"], how="anti")
        if not missing.is_empty():
            raise ValueError("decision artifact contains keys outside the backtest price grid")
        if self.decision.kind == "short_straddles":
            return weights.sort("timestamp", "symbol")
        selected = ["symbol", "timestamp", "weight"]
        if "_state_transition" in weights.columns:
            selected.append("_state_transition")
        return weights.select(*selected)

    def _risk_state_weights(
        self,
        predictions: pl.DataFrame,
        prices: pl.DataFrame,
        strategy_spec: dict[str, Any],
    ) -> pl.DataFrame | None:
        risk = strategy_view(strategy_spec).get("risk") or {}
        policy = risk.get("state_transition_policy")
        if policy is None or self.decision is not None:
            return None
        if not isinstance(policy, dict):
            raise TypeError("risk state-transition policy must be a mapping")
        from .decisions import StateTransitionPolicy

        policy = asdict(StateTransitionPolicy(**policy))
        cadence = risk.get("state_transition_cadence")
        if cadence is None:
            raise ValueError("risk state-transition policy requires an explicit cadence")
        weights = precompute_weights(
            predictions,
            strategy_spec,
            prices,
            label=self.label,
            case_study=self.study.case_study,
            prediction_hash=self.prediction.hash,
        )
        fold_columns = [column for column in ("fold", "fold_id") if column in predictions.columns]
        if len(fold_columns) != 1:
            raise ValueError("stateful strategy predictions require exactly one fold column")
        fold_map = predictions.select("timestamp", fold_columns[0]).unique()
        if fold_map.get_column("timestamp").n_unique() != fold_map.height:
            raise ValueError("each stateful strategy timestamp must belong to exactly one fold")
        if fold_columns[0] == "fold_id":
            fold_map = fold_map.rename({"fold_id": "fold"})
        fold_map = fold_map.with_columns(pl.col("timestamp").cast(weights.schema["timestamp"]))
        weights = weights.join(fold_map, on="timestamp", how="left", validate="m:1")
        if weights.get_column("fold").null_count():
            raise ValueError("stateful strategy weights contain timestamps without a fold")
        return apply_state_transition_policy(
            weights,
            policy=policy,
            cadence=str(cadence),
            price_keys=prices.select("symbol", "timestamp")
            .unique()
            .with_columns(pl.col("timestamp").cast(weights.schema["timestamp"])),
            # The declared cadence describes the observation grid. precompute_weights thins to
            # the rebalance schedule, so for a label with rebalance_step > 1 the weight frame is
            # coarser than the cadence and every ordinary rebalance would read as a gap.
            timeline=fold_map.sort("timestamp"),
        )

    def identity(self, *, prices: pl.DataFrame | None = None) -> str:
        spec = self.resolve(prices=prices)
        return backtest_hash_from_parts(self.prediction.hash, spec, identity_version=2)

    def run(
        self,
        *,
        prices: pl.DataFrame | None = None,
        option_lifecycle: pl.DataFrame | None = None,
    ) -> BacktestResult:
        self.study.require_writable()
        if self.split == "holdout" and prices is not None:
            raise ValueError("locked holdout strategy must load canonical holdout prices")
        predictions = self.prediction.load()
        reader_supplied = prices is not None
        resolved_prices = prices
        self.study.activate(ExecutionTier(self.prediction.execution_tier))
        if not reader_supplied:
            resolved_prices = load_backtest_prices_for(
                self.study.case_study,
                self.label,
                split=self.split,
                warmup_periods=self._warmup_periods(),
            )
        assert resolved_prices is not None
        resolved_prices = self._engine_prices(
            resolved_prices,
            reader_supplied=reader_supplied,
        )
        contract_specs = (
            load_contract_specs_from_yaml() if self.study.case_study == "cme_futures" else None
        )
        case_config = get_backtest_config(self.study.case_study)
        spec = self._build_spec(resolved_prices, case_config, contract_specs)
        allocation = spec.get("strategy", {}).get("allocation", {})
        if self.split == "holdout" and allocation.get("method") == "conformal_weighted":
            lock_record = self._active_lock_record()
            compute_holdout_conformal_widths(
                self.study.case_study,
                lock_record["prediction_hash"],
                self.prediction.hash,
                alpha=float(allocation.get("alpha", 0.2)),
                min_calibration_n=int(allocation["min_calibration_n"]),
                embargo_steps=holdout_conformal_embargo_steps(self.study.case_study, self.label),
                write=True,
            )
        tier = ExecutionTier(self.prediction.execution_tier)
        self.study.activate(tier)
        decision_weights = self._decision_weights(resolved_prices)
        if decision_weights is None:
            decision_weights = self._risk_state_weights(predictions, resolved_prices, spec)
        result = run_backtest(
            self.study.case_study,
            self.prediction.hash,
            spec,
            prices=resolved_prices,
            predictions=predictions,
            precomputed_weights=decision_weights,
            funding_rates=self._funding_rates(resolved_prices),
            label=self.label,
            initial_cash=float(spec["backtest_config"]["cash"]["initial"]),
            calendar=case_config.calendar,
            contract_specs=contract_specs,
            option_lifecycle=option_lifecycle,
        )
        expected_hash = backtest_hash_from_parts(
            self.prediction.hash,
            serializable_backtest_spec(spec),
            identity_version=2,
        )
        if result.backtest_hash != expected_hash:
            raise RuntimeError(
                f"backtest identity changed during execution: {expected_hash} -> "
                f"{result.backtest_hash}"
            )
        reopened = Result.open(
            self.study,
            expected_hash,
            include_preview=tier is ExecutionTier.PREVIEW,
        )
        assert isinstance(reopened, BacktestResult)
        return reopened
