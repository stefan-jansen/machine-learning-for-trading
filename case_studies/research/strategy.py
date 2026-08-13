from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    load_contract_specs_from_yaml,
)
from case_studies.utils.backtest_presets import build_backtest_spec, serializable_backtest_spec
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.registry import backtest_hash_from_parts, canonical_json, compute_hash

from .contracts import ExecutionTier
from .results import BacktestResult, PredictionResult, Result

if TYPE_CHECKING:
    from .workspace import Study


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

    def __post_init__(self) -> None:
        if self.prediction.study != self.study:
            raise ValueError("prediction belongs to another study")
        if not self.prediction.complete:
            raise ValueError("partial predictions cannot enter strategy execution")
        if self.prediction.registry_record()["split"] != "validation":
            raise ValueError("development strategy execution requires validation predictions")

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

    def resolve(self, *, prices: pl.DataFrame | None = None) -> dict[str, Any]:
        self.study.activate(ExecutionTier.CANONICAL)
        case_config = get_backtest_config(self.study.case_study)
        resolved_prices = prices
        if resolved_prices is None:
            resolved_prices = load_backtest_prices_for(
                self.study.case_study, self.label, split="validation"
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
            allocation=self.allocation,
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
        if contract_specs is not None:
            serialized = {
                symbol: asdict(contract_spec) for symbol, contract_spec in contract_specs.items()
            }
            spec["input_identity"]["contract_specs"] = compute_hash(canonical_json(serialized))
        return spec

    def identity(self, *, prices: pl.DataFrame | None = None) -> str:
        spec = self.resolve(prices=prices)
        return backtest_hash_from_parts(self.prediction.hash, spec, identity_version=2)

    def run(self, *, prices: pl.DataFrame | None = None) -> BacktestResult:
        self.study.require_writable()
        predictions = self.prediction.load()
        resolved_prices = prices
        self.study.activate(ExecutionTier.CANONICAL)
        if resolved_prices is None:
            resolved_prices = load_backtest_prices_for(
                self.study.case_study, self.label, split="validation"
            )
        contract_specs = (
            load_contract_specs_from_yaml() if self.study.case_study == "cme_futures" else None
        )
        case_config = get_backtest_config(self.study.case_study)
        spec = self._build_spec(resolved_prices, case_config, contract_specs)
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
