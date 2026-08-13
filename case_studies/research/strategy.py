from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    load_contract_specs_from_yaml,
)
from case_studies.utils.backtest_presets import build_backtest_spec, serializable_backtest_spec
from case_studies.utils.backtest_runner import resolve_cost_feasible_universe, run_backtest
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
    from .workspace import Study


_MOMENT_ALLOCATORS = {"inverse_vol", "risk_parity", "hrp", "mvo", "mvo_ledoit_wolf"}


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
        return cls(
            study=study,
            prediction=prediction,
            signal=deepcopy(request.get("signal") or {}),
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

    def resolve(self, *, prices: pl.DataFrame | None = None) -> dict[str, Any]:
        self.study.activate(ExecutionTier.CANONICAL)
        if self.split == "holdout" and prices is not None:
            raise ValueError("locked holdout strategy must load canonical holdout prices")
        case_config = get_backtest_config(self.study.case_study)
        resolved_prices = prices
        if resolved_prices is None:
            resolved_prices = load_backtest_prices_for(
                self.study.case_study,
                self.label,
                split=self.split,
                warmup_periods=self._warmup_periods(),
            )
        contract_specs = (
            load_contract_specs_from_yaml() if self.study.case_study == "cme_futures" else None
        )
        return self._build_spec(resolved_prices, case_config, contract_specs)

    def _build_spec(self, prices, case_config, contract_specs) -> dict[str, Any]:
        signal = deepcopy(self.signal)
        if signal.get("universe_filter") == "cost_feasible":
            split, symbols = resolve_cost_feasible_universe(
                self.study.case_study,
                self.prediction.hash,
            )
            signal["universe_split"] = split
            signal["universe_symbols"] = symbols
        spec = build_backtest_spec(
            self.study.case_study,
            case_config,
            prices=prices,
            prediction_hash=self.prediction.hash,
            initial_cash=case_config.initial_cash,
            signal=signal,
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
        if signal.get("universe_filter") == "cost_feasible":
            spec["input_identity"]["universe"] = compute_hash(
                canonical_json(
                    {
                        "split": signal["universe_split"],
                        "symbols": signal["universe_symbols"],
                    }
                )
            )
        if contract_specs is not None:
            serialized = {
                symbol: asdict(contract_spec) for symbol, contract_spec in contract_specs.items()
            }
            spec["input_identity"]["contract_specs"] = compute_hash(canonical_json(serialized))
        resolved = ensure_conformal_calibration_identity(spec)
        if self.split == "holdout":
            from .lifecycle import _locked_strategy_projection

            locked = self._active_lock_record()["strategy_spec"]
            if _locked_strategy_projection(resolved) != _locked_strategy_projection(locked):
                raise ValueError("holdout strategy does not match the locked validation strategy")
        return resolved

    def identity(self, *, prices: pl.DataFrame | None = None) -> str:
        spec = self.resolve(prices=prices)
        return backtest_hash_from_parts(self.prediction.hash, spec, identity_version=2)

    def run(self, *, prices: pl.DataFrame | None = None) -> BacktestResult:
        self.study.require_writable()
        if self.split == "holdout" and prices is not None:
            raise ValueError("locked holdout strategy must load canonical holdout prices")
        predictions = self.prediction.load()
        resolved_prices = prices
        self.study.activate(ExecutionTier.CANONICAL)
        if resolved_prices is None:
            resolved_prices = load_backtest_prices_for(
                self.study.case_study,
                self.label,
                split=self.split,
                warmup_periods=self._warmup_periods(),
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
        result = run_backtest(
            self.study.case_study,
            self.prediction.hash,
            spec,
            prices=resolved_prices,
            predictions=predictions,
            label=self.label,
            initial_cash=float(spec["backtest_config"]["cash"]["initial"]),
            calendar=case_config.calendar,
            contract_specs=contract_specs,
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
