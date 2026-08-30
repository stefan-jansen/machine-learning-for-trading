from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import time as dt_time
from typing import Any

import pandas as pd
import polars as pl

from .decisions import DecisionArtifact, StateTransitionPolicy
from .lifecycle import ResearchLock
from .results import BacktestResult, PredictionResult

_FOLD_FIELDS = ("fold", "train_start", "train_end", "val_start", "val_end")


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


def _validation_folds(validation_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    computation = validation_spec.get("computation")
    if not isinstance(computation, dict):
        raise ValueError("holdout CV derivation requires a current resolved training specification")
    cv = computation.get("cv")
    if not isinstance(cv, dict):
        raise ValueError("selected training specification has no resolved CV interval")
    folds = cv.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("selected training specification has no resolved validation folds")
    missing = {name for fold in folds for name in _FOLD_FIELDS if name not in fold}
    if missing:
        raise ValueError(f"validation folds are missing boundaries: {sorted(missing)}")
    return [dict(fold) for fold in folds]


def widest_label_buffer(case_study: str, setup: Mapping[str, Any]) -> tuple[str, str]:
    """Return the widest buffer any of a case study's labels declares, and whose it is.

    The holdout fold is one fold. A fold-scoped temporal artifact carries a single set of
    boundaries per fold id, and a fold-fitted feature's ``train_end`` is what that feature
    knows, so the fold's geometry has to be the same whichever label a model is later fitted
    on. That leaves one question: which buffer.

    It is the widest, and the narrow ones are unsafe rather than merely tighter. A fold built
    on a one-day buffer and handed to a twenty-one-day model gives that model training rows
    whose features were fitted on data twenty sessions past its own ``train_end`` - the leak
    the buffer exists to prevent, arriving through the feature instead of the label. The
    widest buffer costs the shorter-horizon models a longer gap than they need, which is the
    conservative direction and, on the case studies measured, twenty sessions out of thousands.

    Labels are read from ``labels.primary`` and ``labels.variants`` and resolved through
    :func:`utils.artifact_specs.resolve_label_buffer`, so a label carrying its own spec
    artifact still wins over the setup block.
    """
    from utils.artifact_specs import resolve_label_buffer
    from utils.cv_splits import normalize_label_buffer

    labels = setup.get("labels") or {}
    names = [str(labels["primary"])] if labels.get("primary") else []
    names += [str(name) for name in (labels.get("variants") or [])]
    if not names:
        raise ValueError(f"{case_study} declares no labels, so no holdout buffer can be derived")

    widest: tuple[pd.Timedelta, str, str] | None = None
    for name in names:
        buffer = resolve_label_buffer(case_study, name, setup)
        if not buffer:
            continue
        span = pd.Timedelta(normalize_label_buffer(buffer))
        if widest is None or span > widest[0]:
            widest = (span, str(buffer), name)
    if widest is None:
        raise ValueError(
            f"{case_study} declares labels {names} and a buffer for none of them, so the gap "
            "sealing holdout training from the holdout window cannot be derived"
        )
    return widest[1], widest[2]


def _boundary_iso(moment: pd.Timestamp) -> str:
    """Render a fold boundary the way the panel it describes carries its dates.

    A midnight boundary is a date, and writing it as `2023-11-29T00:00:00` says the panel
    has a time of day that it does not. Every daily panel stores its dates as `Date`, and
    Polars reads a full ISO datetime into `Date` as null rather than truncating it, so the
    datetime rendering could not be read back by the consumer it was written for. The time
    is kept when there is one - `crypto_perps_funding` and `nasdaq100_microstructure` are
    intraday and their boundaries are not midnight.
    """
    if moment.time() == dt_time(0, 0):
        return moment.date().isoformat()
    return moment.isoformat()


def _on_panel_clock(moment: pd.Timestamp, zone: Any) -> pd.Timestamp:
    """Read a boundary on the clock the panel keeps its own observations on.

    ``evaluation.holdout_start`` is a date, and a date is not a moment until something says
    which clock it is read on. The fold boundaries are moments, taken from the panel, so an
    intraday panel carries them tz-aware; pandas then refuses to compare the two rather than
    assuming a zone, and the derivation raises ``Cannot compare tz-naive and tz-aware
    timestamps`` before it computes anything. Every daily case study escaped it because its
    panel is tz-naive, which is why this surfaced first on ``crypto_perps_funding``.

    Localizing rather than converting is what keeps the declaration meaning what it says: the
    window is declared in the calendar the case study trades on, so 2024-01-01 is midnight on
    that calendar and not midnight UTC shifted into it.
    """
    if zone is None or moment.tzinfo is not None:
        return moment
    return moment.tz_localize(zone)


def build_holdout_cv(
    validation_spec: Mapping[str, Any],
    *,
    case_study: str,
    timeline: Sequence[Any],
    label: str | None = None,
    train_start_floor: Any | None = None,
) -> dict[str, Any]:
    """Derive the one holdout CV interval that retrains the selected validation configuration.

    The holdout window is not a choice. It is ``evaluation.holdout_start`` and
    ``evaluation.holdout_end`` from the case study's own ``setup.yaml``, read here through
    :func:`case_studies.utils.cv_window.canonical_window` so this derivation and the window a
    backtest is sliced to cannot disagree. ``lifecycle.lock`` re-checks the same window before
    it will accept the spec, and :func:`case_studies.research.models.locked_holdout_split`
    checks it a third time at execution.

    The training interval is the whole history available before that window, which is
    ``min(train_start)`` across the validation folds and never one fold's own start: the fold
    list runs newest first, so ``folds[0]["train_start"]`` is the *latest* start in the set and
    would hand the retrain the shortest window it could have had rather than the longest.
    :func:`utils.cv_splits.earliest_train_start` is that read, and this calls it rather than
    repeating it.

    ``train_start_floor`` bounds that below, and exists because "the whole history available"
    is a claim about the FEATURES, not about the calendar. A configuration fitted on fold-scoped
    model-based features has no history before the fold that produced them: stage 04 emits each
    fold over a rolling window, so on sp500_equity_option_analytics the deriver asks for
    2017-01-05..2020-12-16 and the artifact's holdout fold begins 2019-01-02, leaving 495 of 977
    training dates covered. Fitting the other 482 on null columns is not the configuration that
    was ranked - every validation fold saw a fully populated three-year window - so the holdout
    would evaluate an estimator nobody selected. Clamping to the producer's geometry applies the
    same rule correctly rather than contradicting it: take everything available, where available
    is what the features actually span. Families with no fold-scoped features supply no floor and
    are unaffected. ml4t/agent-workspace#977 has the measurement and the rejected alternative.

    Training ends one label buffer before the holdout opens, using the same buffer the
    validation folds were built with. That gap is what stops the last training label's outcome
    window from reaching into the holdout, which would train the holdout model on the period it
    is meant to be judged against. The buffer is required rather than defaulted: a case study
    that declares none has no basis for a gap, and a zero gap here is a leak rather than a
    conservative choice.
    """
    from case_studies.utils.artifact_digest import value_digest
    from case_studies.utils.causal import embargo_from_buffer, observation_step
    from case_studies.utils.cv_window import canonical_window
    from utils.artifact_specs import (
        load_setup_config,
        resolve_label_buffer,
        resolve_label_horizon,
    )
    from utils.cv_splits import earliest_train_start, normalize_label_buffer

    resolved_label = str(label if label is not None else validation_spec.get("label") or "")
    if not resolved_label:
        raise ValueError("holdout CV derivation requires the label the selection was made on")

    window = canonical_window(case_study, resolved_label, split="holdout")
    if window is None:
        raise ValueError(
            f"{case_study} declares no holdout window for {resolved_label!r}; "
            "evaluation.holdout_start and evaluation.holdout_end must both be set in "
            "config/setup.yaml before a holdout can be locked"
        )
    holdout_start, holdout_end = window

    folds = _validation_folds(validation_spec)
    validation_cv = dict(validation_spec["computation"]["cv"])
    observations = sorted({pd.Timestamp(str(value)) for value in timeline})
    if len(observations) < 2:
        raise ValueError("holdout CV derivation needs at least two observations to measure cadence")
    # Every boundary below is compared against these observations, so they are what decides
    # the clock. Resolved once, here, rather than at each comparison: a fold set and a window
    # that reach this function on different clocks have to be reconciled in one place or the
    # reconciliation is a thing to remember at four call sites.
    panel_zone = observations[0].tz
    train_start = _on_panel_clock(earliest_train_start(folds), panel_zone)
    floor_applied = None
    if train_start_floor is not None:
        floor = _on_panel_clock(pd.Timestamp(train_start_floor), panel_zone)
        if floor > train_start:
            # Recorded, not silent: the clamp changes the interval the lock is taken over, so a
            # reader of the spec has to be able to see that the window is the producer's and why.
            floor_applied = _boundary_iso(floor)
            train_start = floor

    # Both resolvers fall back to setup.yaml's own labels block, and return None without it
    # for every case study whose label carries no separate spec artifact - which is all nine
    # for their primary label. Passing the setup is what makes the buffer resolvable at all.
    setup = load_setup_config(case_study)
    # The case study's widest buffer, not the selected label's own. The fold-scoped temporal
    # artifact carries one holdout fold whose features every label's holdout model is fitted
    # on, so its boundary has to be label-independent, and the widest is the only choice that
    # leaks for no label. `widest_label_buffer` carries the argument. The selected label still
    # supplies the horizon check below, which this now satisfies by construction.
    buffer, buffer_label = widest_label_buffer(case_study, setup)
    holdout_open = _on_panel_clock(pd.Timestamp(holdout_start), panel_zone)
    holdout_close = _on_panel_clock(pd.Timestamp(holdout_end), panel_zone)
    # `evaluation.holdout_end` is a DATE, and a date on an intraday panel means the whole of that
    # day. Parsed, it is that date at midnight, and every window filter downstream is
    # `timestamp <= val_end`, so the final session sorts after the boundary and is dropped from
    # the interval the holdout is evaluated over. `utils/modeling.py::_inclusive_end_of` says the
    # same thing with a nanosecond sentinel; this says it with an observation the panel actually
    # holds, which is what `train_end` already is and what makes the fold readable as a pair of
    # settlements rather than one settlement and a fencepost.
    #
    # A daily panel is untouched by construction: its last observation of that date IS midnight,
    # so the widening condition is false and the rendering does not move. That matters because
    # this value is inside the hashed fold, and `fx_pairs` and `sp500_equity_option_analytics`
    # each hold a research lock derived from it. ml4t/agent-workspace#986.
    within_close = [value for value in observations if value.date() <= holdout_close.date()]
    if within_close and within_close[-1] > holdout_close:
        holdout_close = within_close[-1]

    # Counted in OBSERVATIONS and stepped back along the panel's own dates, never subtracted as
    # calendar time. `utils/cv_splits.py` already carries this bug's epitaph: "21D" as a
    # pd.Timedelta is ~15 trading days, not 21, so a calendar subtraction leaves the last training
    # label resolving inside the holdout - short, silent, and in the direction that looks fine.
    # `generate_cv_splits` converts D-buffers to trading days for exactly this reason, and the
    # causal resolver counts observations for the same one. This is the third construction of the
    # same seal and it must agree with the other two.
    # Measured against the panel's OWN cadence, not a per-unit default. Without observed_step
    # embargo_from_buffer assumes a daily grid, which reads "1M" as 21 observations - correct on a
    # daily panel and 21x too long on us_firm_characteristics' monthly one, where a month IS one
    # observation. AGENTS.md records the mirror of this: "24H as one period on an eight-hour panel".
    cadence = observation_step(pd.DataFrame({"timestamp": observations}))
    # A month has no fixed length, so it cannot be divided by an observation step and
    # embargo_from_buffer refuses. Its other branch takes periods_per_year, which is COUNTED here
    # off the same timeline rather than assumed: falling through to the per-unit defaults is what
    # turns "1M" into 21 observations on a monthly panel where a month is one.
    span_years = (observations[-1] - observations[0]).days / 365.25
    periods_per_year = max(1, round(len(observations) / span_years)) if span_years > 0 else 1
    try:
        buffer_steps = embargo_from_buffer(buffer, observed_step=cadence)
    except ValueError:
        buffer_steps = embargo_from_buffer(buffer, periods_per_year=periods_per_year)
    if buffer_steps < 1:
        raise ValueError(f"label buffer {buffer!r} leaves no gap before the holdout window")

    # A declared zero horizon - us_firm_characteristics dates each row by the month the return was
    # earned, so "0D" - means the outcome is already realised at the observation and there is
    # nothing for the buffer to cover. embargo_from_buffer divides by the value, so it must not be
    # asked about zero rather than being asked and having its answer discarded.
    horizon = resolve_label_horizon(case_study, resolved_label, setup)
    if horizon and pd.Timedelta(normalize_label_buffer(horizon)) > pd.Timedelta(0):
        try:
            horizon_steps = embargo_from_buffer(horizon, observed_step=cadence)
        except ValueError:
            horizon_steps = embargo_from_buffer(horizon, periods_per_year=periods_per_year)
        if buffer_steps < horizon_steps:
            raise ValueError(
                f"label buffer {buffer!r} is {buffer_steps} observations, shorter than the "
                f"outcome horizon {horizon!r} at {horizon_steps}, so the last training label "
                "resolves inside the holdout window"
            )

    pre_holdout = [value for value in observations if value < holdout_open]
    if len(pre_holdout) <= buffer_steps:
        raise ValueError(
            f"{case_study} has {len(pre_holdout)} observations before the holdout opens, which "
            f"cannot absorb a {buffer_steps}-observation buffer"
        )
    # The buffer is the number of observations that must NOT be trained on, so the last retained
    # one sits one step beyond it. `pre_holdout[-buffer_steps]` is the first excluded observation.
    train_end = pre_holdout[-(buffer_steps + 1)]
    if train_end <= train_start:
        raise ValueError(
            f"{case_study} holdout training interval is empty: history starts "
            f"{train_start.date()} and the buffered boundary is {train_end.date()}"
        )

    fold = {
        "fold": max(int(entry["fold"]) for entry in folds) + 1,
        "train_start": _boundary_iso(train_start),
        "train_end": _boundary_iso(train_end),
        "val_start": _boundary_iso(holdout_open),
        "val_end": _boundary_iso(holdout_close),
    }
    identity = value_digest(pl.DataFrame([fold]))
    if identity == validation_cv.get("identity"):
        raise ValueError("derived holdout CV is identical to the selected validation CV")
    return {
        "folds": [fold],
        "identity": identity,
        "split": "holdout",
        "request": {
            "source": "case_study_holdout",
            "label_buffer": str(buffer),
            "label_buffer_label": buffer_label,
            "label_buffer_steps": buffer_steps,
            "observation_cadence": str(cadence),
            "periods_per_year": periods_per_year,
            "holdout_window": [str(holdout_start), str(holdout_end)],
            # Present only when it moved the boundary, so a spec that needed no clamp hashes
            # exactly as it did before this existed and no recorded lock is disturbed.
            **({"train_start_floor": floor_applied} if floor_applied else {}),
        },
    }


def build_holdout_training_spec(
    study: Any,
    validation_spec: Mapping[str, Any],
    *,
    timeline: Sequence[Any],
    case_study: str | None = None,
) -> dict[str, Any]:
    """Re-key one validation training specification onto the derived holdout fold.

    Three steps have to happen together and in this order, and each of them already refuses
    on its own terms: derive the holdout interval from the case study's declared window,
    bound its training start at whatever the family's features actually reach, and recompute
    the fields the resolver derived per validation fold. Doing two of the three produces a
    specification that looks complete and fits the wrong estimator - a manifest describing
    the validation folds, or a training window half of which has no features - so they are
    one call rather than three a caller assembles.

    This takes a ``study`` and a specification, not a lock. A holdout fit is a computation,
    and the question of how many times a case study may run one is a separate question about
    its lifecycle: :func:`evaluate_holdout` is the answer for a case study that wants the
    holdout spent once and calls this to build what it locks, and a case study whose holdout
    notebooks re-run like any other stage calls this directly.

    Returns a new specification; ``validation_spec`` is not modified.
    """
    holdout_spec = deepcopy(dict(validation_spec))
    holdout_spec["computation"]["cv"] = build_holdout_cv(
        validation_spec,
        case_study=str(case_study if case_study is not None else study.case_study),
        timeline=timeline,
        train_start_floor=_holdout_training_floor(study, validation_spec),
    )
    _rekey_holdout_spec(study, holdout_spec, dict(validation_spec))
    return holdout_spec


@dataclass(frozen=True)
class HoldoutOutcome:
    """One case study's holdout evaluation, and whether this call is what produced it."""

    lock: ResearchLock
    evaluated_now: bool

    @property
    def lineage(self) -> dict[str, str]:
        return self.lock.study.lifecycle.holdout_lineage(self.lock.hash)


def evaluate_holdout(
    study: Any,
    *,
    candidate_set_name: str,
    timeline: Sequence[Any],
    case_study: str | None = None,
    selection_evidence: Mapping[str, Any] | None = None,
) -> HoldoutOutcome:
    """Lock one selected configuration and evaluate it on the holdout, at most once ever.

    This is the whole sequence, in one place, because every case study needs the identical one
    and nine notebooks assembling it from five primitives is how nine versions of it appear.
    Each primitive it calls already exists and is already tested; what did not exist was anything
    that called them.

    The holdout is used once. That is not a convention this enforces by asking callers to check
    first - re-running a notebook is normal and must not be able to spend the holdout a second
    time. So an already-evaluated lifecycle returns its recorded lineage and executes nothing,
    and ``evaluated_now`` is how a caller tells the two apart. A notebook re-run therefore reads
    back exactly the numbers it published before, which is also what makes the page reproducible.

    Selection is not a parameter either. ``lifecycle.lock`` refuses any selection that is not the
    candidate set's highest validation backtest Sharpe, so passing the rank-1 member is the only
    thing that can be passed - this reads it from the set rather than accepting it, which removes
    the one place a caller could have disagreed with the documented rule.
    """
    from .comparison import CandidateSet
    from .execution import run_locked_holdout
    from .lifecycle import LifecycleState

    lifecycle = study.lifecycle
    state = lifecycle.state
    if state == LifecycleState.HOLDOUT_EVALUATED.value:
        # Returning the recorded lineage is what makes a notebook re-run safe, but returning it
        # WITHOUT looking at the arguments would make this function answer a question it was not
        # asked: a caller naming a different candidate set, or one that no longer resolves, would
        # silently receive the old holdout as though it had been confirmed against the new
        # selection. So the selection is re-derived and checked against what the lock recorded.
        existing = _sole_lock(study)
        _confirm_recorded_selection(study, existing, candidate_set_name)
        return HoldoutOutcome(existing, evaluated_now=False)

    candidates = CandidateSet.one(study, name=candidate_set_name)
    selected = candidates.best_validation_sharpe()
    selected_record = selected.registry_record()
    prediction = study.results.open(selected_record["prediction_hash"])
    training = study.results.open(prediction.registry_record()["training_hash"])

    validation_spec = training.spec()
    holdout_spec = build_holdout_training_spec(
        study,
        validation_spec,
        timeline=timeline,
        case_study=case_study,
    )

    # selection_evidence is hashed into the lock identity, so anything put here that is already
    # recorded elsewhere gives one fact two sources and makes the lock unreproducible by any
    # caller that words it differently. The candidate set is already in the record under
    # candidate_set_hash; the metric is the only thing this adds, and it is the documented rule.
    evidence = {"metric": "validation_backtest_sharpe", **dict(selection_evidence or {})}
    lock = lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=selected.hash,
        selection_evidence=evidence,
        holdout_training_spec=holdout_spec,
    )
    if lock.reopen().state == LifecycleState.HOLDOUT_EVALUATED.value:
        # The lock already existed and had been spent. lifecycle.lock returns the existing lock
        # rather than raising when the request is identical, so this is reached by a re-run whose
        # selection has not changed - and it must not re-execute.
        return HoldoutOutcome(lock.reopen(), evaluated_now=False)

    execution = run_locked_holdout(lock)
    return HoldoutOutcome(execution.lock, evaluated_now=True)


def _sole_lock(study: Any) -> ResearchLock:
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        rows = db.execute("SELECT lock_hash FROM research_locks").fetchall()
    if len(rows) != 1:
        raise ValueError(f"lifecycle holds {len(rows)} research locks, not one")
    return study.lifecycle.open(rows[0][0])


# Fields the resolver derives PER FOLD, from the data, during a run. They describe the VALIDATION
# fold set, and `validate_locked_model_run` requires them re-keyed to the HOLDOUT fold:
# `validate_locked_expected_keys` raises "no eligibility manifest" when
# `expected_prediction_keys` is absent, and "eligibility mismatch" when it describes a different
# frame. So neither carrying them forward nor dropping them is correct - both produce a lock that
# fails at execution, one silently wrong and one loudly.
#
# WHAT THE FIX IS, so this is a specified task and not a vague blocker.
#
# `case_studies/utils/linear.py:675` computes the manifest at RECONSTRUCTION time with
# `_expected_keys_from_dataset(mds.dataset, [split], ...)`, where `split` comes from
# `locked_holdout_split(spec, ...)`, and then checks it against what the spec recorded. So the
# computation exists; it just runs after the lock, against a value the lock was supposed to carry.
#
# Building the spec correctly means running that same computation BEFORE locking: open the dataset,
# build the holdout split from the derived CV, compute the eligible keys, and record the digest,
# row count and fold count. It is family-specific - `_expected_keys_from_dataset` lives in
# `linear.py` and each family has its own - so it wants a per-family hook resolved through
# `_family_module`, exactly as `reconstruct_locked_request` and `validate_locked_run` already are.
#
# Note also that `CVSpec` is NOT the vehicle. It carries `holdout_start`/`holdout_end`, but
# `resolve()` passes them to `generate_cv_splits` as boundaries to seal VALIDATION against; it
# selects validation folds and cannot emit a holdout fold. Nothing in the resolver produces a
# holdout training fold today, which is why this had to be derived here in the first place.
#
# Until that hook exists, refusing is the only honest option: a lock is the one artifact in the
# pipeline that cannot be revised, so producing one that is known to fail at execution is worse
# than producing none.
_FOLD_DERIVED_FIELDS = (
    ("computation", "expected_prediction_keys"),
    ("model", "effective_params_by_fold"),
    ("macro_context", "resolved_fold_digest"),
)


def _holdout_training_floor(study: Any, validation_spec: Mapping[str, Any]) -> Any | None:
    """Ask the family how far back its features actually reach, or None if nothing bounds it.

    Dispatches through ``_family_module`` exactly as ``_rekey_holdout_spec`` does. Absence is the
    answer for most families and is not an error: a configuration whose features are defined over
    the whole panel has no floor, and returning None leaves the derivation exactly as it was.
    Only a family whose features are fold-scoped can answer, because only it knows which artifact
    holds them.
    """
    from .models import _family_module

    hook = getattr(_family_module(validation_spec.get("family")), "holdout_training_floor", None)
    return None if hook is None else hook(study, validation_spec=validation_spec)


def _rekey_holdout_spec(study: Any, spec: dict[str, Any], validation_spec: dict[str, Any]) -> None:
    """Recompute the fold-derived fields for the holdout fold, or refuse with the family named.

    The fields are family-specific - each family derives them with its own rule, from its own
    training rows - so this dispatches through ``_family_module`` exactly as
    ``reconstruct_locked_request`` and ``validate_locked_run`` already do. A family that has not
    implemented the hook still refuses, but it now refuses for itself rather than on behalf of
    every family at once.
    """
    from .models import _family_module

    family = spec.get("family")
    module = _family_module(family)
    hook = getattr(module, "rekey_holdout_spec", None)
    if hook is None:
        raise NotImplementedError(
            f"the {family!r} family cannot yet re-key a validation training spec to the holdout "
            "fold, so no lock can be taken for it. Implementing it means recomputing this "
            "family's fold-derived fields against the derived holdout fold - the eligibility "
            "manifest from the dataset, and any parameter the family resolves from a fold's own "
            "training rows - by the same rule that produced the recorded validation values. See "
            "`rekey_holdout_spec` in case_studies/utils/linear.py."
        )
    hook(study, spec, validation_spec=validation_spec)
    _require_holdout_keyed_fields(spec)


def _require_holdout_keyed_fields(spec: dict[str, Any]) -> None:
    """Check the re-keyed fields describe the holdout fold, not the validation folds.

    A hook that returned without recomputing, or that recomputed against the wrong split, leaves
    fields that look present and are wrong - and the lock is the one artifact that cannot be
    revised. So the shape is checked here rather than trusted: exactly one fold, and it is the
    fold the derived holdout CV names.
    """
    computation = spec["computation"]
    cv = computation.get("cv")
    if not isinstance(cv, dict):
        raise ValueError("holdout spec has no resolved CV")
    # Both shapes `locked_holdout_split` accepts: an explicit one-fold list, or the flat form
    # where the single fold's boundaries sit on the CV record itself.
    folds = cv.get("folds")
    if folds is None:
        fold_id = str(int(cv.get("fold", 0)))
    elif isinstance(folds, list) and len(folds) == 1:
        fold_id = str(int(folds[0]["fold"]))
    else:
        raise ValueError("holdout spec must carry exactly one resolved fold")

    manifest = computation.get("expected_prediction_keys")
    if not isinstance(manifest, dict) or manifest.get("n_folds") != 1:
        raise ValueError(f"holdout eligibility manifest was not re-keyed to one fold: {manifest!r}")

    model = computation.get("model")
    if isinstance(model, dict) and "effective_params_by_fold" in model:
        keys = set(model["effective_params_by_fold"])
        if keys != {fold_id}:
            raise ValueError(
                f"holdout parameters are keyed to {sorted(keys)}, not the holdout fold "
                f"{fold_id!r}; they still describe the validation folds"
            )

    task = computation.get("task")
    if isinstance(task, dict):
        imbalance = task.get("imbalance")
        if isinstance(imbalance, dict) and "effective_class_weights_by_fold" in imbalance:
            keys = set(imbalance["effective_class_weights_by_fold"])
            if keys != {fold_id}:
                raise ValueError(
                    f"holdout class weights are keyed to {sorted(keys)}, not the holdout fold "
                    f"{fold_id!r}"
                )


def _confirm_recorded_selection(study: Any, lock: ResearchLock, candidate_set_name: str) -> None:
    """Check the spent holdout answers the selection the caller is asking about."""
    from .comparison import CandidateSet

    candidates = CandidateSet.one(study, name=candidate_set_name)
    if candidates.hash != lock.record["candidate_set_hash"]:
        raise ValueError(
            f"holdout was evaluated against candidate set {lock.record['candidate_set_hash']!r}, "
            f"not {candidate_set_name!r} ({candidates.hash!r}); the holdout is used once and "
            "cannot be re-spent against a different selection"
        )
    selected = candidates.best_validation_sharpe()
    if selected.hash != lock.record["validation_backtest_hash"]:
        raise ValueError(
            f"candidate set {candidate_set_name!r} now ranks {selected.hash!r} first, but the "
            f"holdout was evaluated on {lock.record['validation_backtest_hash']!r}"
        )
