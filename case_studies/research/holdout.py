from __future__ import annotations

import hashlib
import importlib
import inspect
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

from .decisions import DecisionArtifact, StateTransitionPolicy
from .lifecycle import ResearchLock
from .results import BacktestResult, PredictionResult


def _source_digest(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


@dataclass(frozen=True)
class _DecisionReplay:
    original: DecisionArtifact
    function: Any
    locked_inputs: dict[str, Any]

    def publish(
        self,
        prediction: PredictionResult,
        prices: pl.DataFrame,
    ) -> DecisionArtifact:
        parameters = dict(self.original.spec["parameters"])
        available = {
            **deepcopy(self.locked_inputs),
            "prediction": prediction,
            "predictions": prediction.load(),
            "prediction_hash": prediction.hash,
            "prediction_hashes": [prediction.hash],
            "prices": prices,
            "study": prediction.study,
        }
        accepted = {
            name
            for name, parameter in inspect.signature(self.function).parameters.items()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        accepts_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(self.function).parameters.values()
        )
        values = {**parameters, **available}
        decisions = self.function(
            **{
                name: values[name]
                for name in (set(values) if accepts_keywords else accepted)
                if name in values
            },
        )
        if not isinstance(decisions, pl.DataFrame):
            raise TypeError("holdout decision replay must return a Polars DataFrame")
        source_identity = deepcopy(self.original.spec["source_identity"])
        from case_studies.utils.artifact_digest import value_digest

        declared_inputs = deepcopy(source_identity["declared_inputs"])
        declared_inputs["prediction_hashes"] = [prediction.hash]
        if "prices" in declared_inputs:
            declared_inputs["prices"] = value_digest(prices)
        source_identity["declared_inputs"] = declared_inputs
        source_identity["clean_replay_digest"] = value_digest(decisions)
        policy = self.original.spec.get("state_transition_policy")
        return DecisionArtifact.publish(
            prediction.study,
            kind=self.original.kind,
            decisions=decisions,
            prediction_hashes=[prediction.hash],
            parameters=parameters,
            source_identity=source_identity,
            state_transition_policy=StateTransitionPolicy(**policy) if policy else None,
            canonical=True,
        )


@dataclass(frozen=True)
class LockedStrategyReplay:
    lock: ResearchLock
    request: dict[str, Any]
    decision_replay: _DecisionReplay | None

    def run(self, prediction: PredictionResult) -> BacktestResult:
        if prediction.study != self.lock.study:
            raise ValueError("holdout prediction belongs to another study")
        from case_studies.utils.artifact_digest import value_digest
        from case_studies.utils.backtest_presets import serializable_backtest_spec
        from case_studies.utils.backtest_runner import resolved_allow_short_selling
        from case_studies.utils.registry import backtest_hash_from_parts

        from . import strategy as strategy_module
        from .contracts import ExecutionTier
        from .lifecycle import _locked_strategy_projection
        from .results import Result

        locked_spec = deepcopy(self.lock.record["strategy_spec"])
        warmup = strategy_module.strategy_warmup_periods(locked_spec)
        prices = strategy_module.load_backtest_prices_for(
            self.lock.study.case_study,
            str(self.lock.record["label"]),
            split="holdout",
            warmup_periods=warmup,
        )
        decision = (
            self.decision_replay.publish(prediction, prices) if self.decision_replay else None
        )
        strategy = self.lock.study.strategy(
            prediction=prediction,
            decision=decision,
            **self.request,
        )
        # The loader keys cme_futures prices on `product`, while the allocator and
        # the engine both select `symbol`. Strategy.run renames before either sees
        # the frame; reuse that owner's implementation rather than repeating it.
        # The digest stays on the reader frame, because that is the one
        # lifecycle._validated_holdout_lineage loads and digests, and the locked
        # strategy projection excludes input_identity.prices from its comparison.
        engine_prices = strategy._engine_prices(prices, reader_supplied=False)
        spec = deepcopy(locked_spec)
        spec.pop("_runtime_backtest_config", None)
        spec["backtest_config"]["metadata"]["prediction_hash"] = prediction.hash
        spec.setdefault("input_identity", {})["prices"] = value_digest(prices)
        if decision is not None:
            decision_record = deepcopy(spec["decision_artifact"])
            decision_record.update(
                {
                    "hash": decision.hash,
                    "kind": decision.kind,
                    "artifact_digest": decision.spec["artifact_digest"],
                    "canonical": decision.canonical,
                    "source_identity": decision.spec["source_identity"],
                    "state_transition_policy": decision.spec["state_transition_policy"],
                }
            )
            for name in ("decision_keys", "parameters"):
                if name in decision_record:
                    decision_record[name] = decision.spec[name]
            spec["decision_artifact"] = decision_record
        elif spec.get("decision_artifact") is not None:
            raise ValueError("locked decision artifact was not transformed for holdout")

        contract_specs = None
        if self.lock.study.case_study == "cme_futures":
            contract_specs = strategy_module.load_contract_specs_from_yaml()
            serialized = {
                symbol: asdict(contract_spec) for symbol, contract_spec in contract_specs.items()
            }
            contract_digest = strategy_module.compute_hash(
                strategy_module.canonical_json(serialized)
            )
            if spec["input_identity"].get("contract_specs") != contract_digest:
                raise ValueError("locked futures contract specifications do not validate")
        funding_rates = strategy._funding_rates(engine_prices)
        if funding_rates is not None:
            spec["input_identity"]["funding_rates"] = value_digest(funding_rates)
            spec["economic_cashflows"] = {"funding": "position_signed_before_same_timestamp_fills"}

        allocation = spec.get("strategy", {}).get("allocation", {})
        if allocation.get("method") == "conformal_weighted":
            strategy_module.compute_holdout_conformal_widths(
                self.lock.study.case_study,
                self.lock.record["prediction_hash"],
                prediction.hash,
                alpha=float(allocation.get("alpha", 0.2)),
                min_calibration_n=int(allocation["min_calibration_n"]),
                embargo_steps=strategy_module.holdout_conformal_embargo_steps(
                    self.lock.study.case_study,
                    strategy.label,
                ),
                write=True,
                immutable=True,
            )
        predictions = prediction.load()
        weights = strategy._decision_weights(engine_prices)
        if weights is None:
            risk_replay = getattr(strategy, "_risk_state_weights", None)
            if callable(risk_replay):
                weights = risk_replay(predictions, engine_prices, spec)
            elif (spec.get("strategy", {}).get("risk") or {}).get(
                "state_transition_policy"
            ) is not None:
                raise ValueError("locked stateful strategy has no executable replay path")
        if weights is not None:
            spec["backtest_config"]["account"]["allow_short_selling"] = (
                resolved_allow_short_selling(spec, weights)
            )
        if _locked_strategy_projection(spec) != _locked_strategy_projection(locked_spec):
            raise ValueError("holdout strategy reconstruction changed the locked computation")

        calendar_block = spec["backtest_config"].get("calendar")
        calendar = (
            calendar_block.get("calendar") if isinstance(calendar_block, dict) else calendar_block
        )
        if not calendar:
            raise ValueError("locked strategy has no resolved trading calendar")
        expected_hash = backtest_hash_from_parts(
            prediction.hash,
            serializable_backtest_spec(spec),
            identity_version=2,
        )
        try:
            cached = Result.open(self.lock.study, expected_hash)
        except KeyError:
            cached = None
        if isinstance(cached, BacktestResult) and cached.complete:
            return cached
        self.lock.study.activate(ExecutionTier.CANONICAL)
        result = strategy_module.run_backtest(
            self.lock.study.case_study,
            prediction.hash,
            spec,
            prices=engine_prices,
            predictions=predictions,
            precomputed_weights=weights,
            funding_rates=funding_rates,
            label=strategy.label,
            initial_cash=float(spec["backtest_config"]["cash"]["initial"]),
            calendar=str(calendar),
            contract_specs=contract_specs,
            resolved_spec_only=True,
            force_rebacktest=cached is not None,
        )
        if result.backtest_hash != expected_hash:
            raise RuntimeError(
                f"locked backtest identity changed during execution: "
                f"{expected_hash} -> {result.backtest_hash}"
            )
        reopened = Result.open(self.lock.study, expected_hash)
        if not isinstance(reopened, BacktestResult) or not reopened.complete:
            raise ValueError("locked strategy did not publish a complete backtest result")
        return reopened


def _prepare_decision_replay(lock: ResearchLock) -> _DecisionReplay | None:
    decision_record = lock.record["strategy_spec"].get("decision_artifact")
    if decision_record is None:
        return None
    if not decision_record.get("canonical"):
        raise ValueError("locked holdout decisions require a canonical validation artifact")
    original = DecisionArtifact.open(lock.study, str(decision_record["hash"]))
    expected = {
        "artifact_digest": original.spec["artifact_digest"],
        "canonical": original.canonical,
        "hash": original.hash,
        "kind": original.kind,
        "source_identity": original.spec["source_identity"],
        "state_transition_policy": original.spec["state_transition_policy"],
    }
    for name in ("decision_keys", "parameters"):
        if name in decision_record:
            expected[name] = original.spec[name]
    if decision_record != expected:
        raise ValueError("locked decision artifact differs from its immutable registry record")
    original.load()
    source_identity = original.spec["source_identity"]
    replay = source_identity.get("holdout_replay")
    if (
        not isinstance(replay, dict)
        or set(replay) != {"version", "function"}
        or replay.get("version") != 1
        or not replay.get("function")
    ):
        raise ValueError("validation decision artifact has no reproducible holdout transformation")
    module = importlib.import_module(str(source_identity["module"]))
    function = getattr(module, str(replay["function"]), None)
    if not callable(function) or _source_digest(function) != source_identity["source_digest"]:
        raise ValueError("holdout decision transformation source identity does not validate")
    signature = inspect.signature(function)
    if not {"prediction", "predictions"} & set(signature.parameters):
        raise ValueError("holdout decision transformation must accept prediction input")
    declared_inputs = source_identity["declared_inputs"]
    if not isinstance(declared_inputs, dict):
        raise ValueError("holdout decision transformation has invalid declared inputs")
    locked_inputs = deepcopy(declared_inputs)
    locked_inputs.pop("prediction_hashes", None)
    locked_inputs.pop("prices", None)
    injected = {
        "prediction",
        "predictions",
        "prediction_hash",
        "prediction_hashes",
        "prices",
        "study",
        *locked_inputs,
    }
    positional_only = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    }
    if positional_only:
        raise ValueError("holdout decision transformation cannot require positional-only inputs")
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    undeclared_parameters = set(locked_inputs) - set(signature.parameters)
    if undeclared_parameters and not accepts_keywords:
        raise ValueError(
            "holdout decision transformation does not accept declared immutable inputs: "
            f"{sorted(undeclared_parameters)}"
        )
    parameters = original.spec["parameters"]
    missing = {
        name
        for name, parameter in signature.parameters.items()
        if name not in injected
        and parameter.default is inspect.Parameter.empty
        and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        and name not in parameters
    }
    if missing:
        raise ValueError(
            f"holdout decision transformation is missing locked parameters: {sorted(missing)}"
        )
    return _DecisionReplay(original, function, locked_inputs)


def prepare_locked_strategy_replay(lock: ResearchLock) -> LockedStrategyReplay:
    """Validate the locked strategy and prepare its exact holdout replay before model writes."""
    spec = lock.record.get("strategy_spec")
    if not isinstance(spec, dict) or spec.get("version") != 2:
        raise ValueError("research lock has no complete canonical strategy specification")
    strategy = spec.get("strategy")
    if not isinstance(strategy, dict) or not isinstance(strategy.get("signal"), dict):
        raise ValueError("research lock has no resolved signal specification")
    rebalance = strategy.get("rebalance") or {}
    request = {
        "signal": deepcopy(strategy["signal"]),
        "allocation": deepcopy(strategy.get("allocation")),
        "risk": deepcopy(strategy.get("risk")),
        "chapter": spec.get("chapter"),
        "execution_mode": rebalance.get("mode"),
        "min_weight_change": rebalance.get("min_weight_change"),
        "min_trade_value": rebalance.get("min_trade_value"),
    }
    return LockedStrategyReplay(lock, request, _prepare_decision_replay(lock))
