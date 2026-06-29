from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from case_studies.utils.backtest_loaders import BacktestConfig as CaseStudyBacktestConfig
from utils.paths import get_case_study_dir

try:
    from ml4t.backtest import BacktestConfig as EngineBacktestConfig
except (ImportError, ModuleNotFoundError):  # pragma: no cover - import depends on env
    EngineBacktestConfig = None


_EXECUTION_MODE_BY_DELAY = {
    "NEXT_BAR_OPEN": "next_bar",
    "MONDAY_OPEN": "next_bar",
    "1_BAR": "next_bar",
    "AT_FUNDING_TIMESTAMP": "same_bar",
}


def resolve_execution_mode(fill_timing: str):
    """Map fill_timing string to ExecutionMode enum.

    Raises ValueError for unknown tokens instead of silently degrading.
    """
    from ml4t.backtest import ExecutionMode

    token = fill_timing.upper().replace(" ", "_")
    mode_str = _EXECUTION_MODE_BY_DELAY.get(token)
    if mode_str is None:
        raise ValueError(
            f"Unknown execution delay '{fill_timing}'. "
            f"Known values: {sorted(_EXECUTION_MODE_BY_DELAY.keys())}"
        )
    return ExecutionMode.NEXT_BAR if mode_str == "next_bar" else ExecutionMode.SAME_BAR


def preset_path(case_study: str) -> Path:
    """Path to the source-controlled backtest preset.

    Always reads from the source repo (never ML4T_OUTPUT_DIR), since
    config/backtest/base.yaml is checked-in source, not runtime data.
    """
    from utils.paths import get_case_study_source_dir

    return get_case_study_source_dir(case_study) / "config" / "backtest" / "base.yaml"


def load_backtest_preset(case_study: str) -> dict[str, Any]:
    path = preset_path(case_study)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Backtest preset at {path} must be a mapping")
    return data


def _infer_data_frequency(cadence: str) -> str:
    token = cadence.lower()
    if "15" in token:
        return "15m"
    if "30" in token:
        return "30m"
    if "1_hour" in token or "hourly" in token or token == "1h":
        return "1h"
    if "8_hour" in token or "funding" in token:
        return "irregular"
    return "daily"


def _build_feed_spec(
    case_study: str,
    prices: pl.DataFrame,
    case_config: CaseStudyBacktestConfig,
) -> dict[str, Any]:
    columns = set(prices.columns)
    feed = {
        "timestamp_col": "timestamp",
        "entity_col": "symbol",
        "open_col": "open" if "open" in columns else None,
        "high_col": "high" if "high" in columns else None,
        "low_col": "low" if "low" in columns else None,
        "close_col": "close" if "close" in columns else None,
        "price_col": "price" if "price" in columns else ("close" if "close" in columns else None),
        "volume_col": "volume" if "volume" in columns else None,
        "bid_col": "bid" if "bid" in columns else None,
        "ask_col": "ask" if "ask" in columns else None,
        "mid_col": "mid" if "mid" in columns else None,
        "calendar": case_config.calendar,
        "data_frequency": _infer_data_frequency(case_config.cadence),
        "timezone": "UTC",
    }
    if case_study in {"sp500_options", "sp500_equity_option_analytics"}:
        feed["bar_type"] = "quote"
    return {k: v for k, v in feed.items() if v is not None}


def build_resolved_backtest_config(
    case_study: str,
    case_config: CaseStudyBacktestConfig,
    strategy_spec: dict[str, Any],
    *,
    prices: pl.DataFrame,
    initial_cash: float,
) -> EngineBacktestConfig:
    if EngineBacktestConfig is None:  # pragma: no cover - import depends on env
        raise ImportError("ml4t-backtest is required for backtest preset resolution")

    preset = deepcopy(load_backtest_preset(case_study))
    preset.setdefault("account", {})
    preset.setdefault("execution", {})
    preset.setdefault("commission", {})
    preset.setdefault("slippage", {})
    preset.setdefault("cash", {})
    preset.setdefault("calendar", {})
    preset.setdefault("position_sizing", {})

    fill_timing = strategy_spec.get("execution", {}).get(
        "fill_timing"
    ) or case_config.execution_delay.upper().replace(" ", "_")
    execution_mode = _EXECUTION_MODE_BY_DELAY.get(fill_timing, "next_bar")
    signal_spec = strategy_spec.get("signal", {})
    signal_direction = str(signal_spec.get("direction", "long_only")).strip().lower()
    allow_short = bool(signal_spec.get("long_short", case_config.long_short)) or (
        signal_direction == "short_only"
    )

    preset["account"]["allow_short_selling"] = allow_short
    preset["execution"]["execution_mode"] = execution_mode
    preset["cash"]["initial"] = float(initial_cash)

    costs = strategy_spec.get("costs", {})
    cost_model = costs.get("model", "percentage")

    if cost_model == "percentage":
        commission_bps = float(costs.get("commission_bps", case_config.commission_bps))
        slippage_bps = float(costs.get("slippage_bps", case_config.slippage_bps))
        preset["commission"]["model"] = "percentage"
        preset["commission"]["rate"] = commission_bps / 10_000.0
        preset["slippage"]["model"] = "percentage"
        preset["slippage"]["rate"] = slippage_bps / 10_000.0
    elif cost_model == "per_share_plus_spread":
        # IB-style realistic equity costs: per-share commission, integer-share
        # sizing, half-spread slippage in dollars per share. Per-asset spreads
        # can be supplied directly (asset_spreads dict in setup.yaml) or via a
        # parquet artifact (asset_spreads_source) measured from quote data.
        preset["commission"]["model"] = "per_share"
        preset["commission"]["per_share"] = float(costs["per_share"])
        preset["commission"]["minimum"] = float(costs.get("minimum", 0.35))

        # When execution_price is quote_side, the fill price already includes
        # the bid/ask half-spread relative to mid (FillEngine returns
        # ask for BUY / bid for SELL via broker.QUOTE_SIDE). Wiring the same
        # measured half-spread into the slippage layer on top of that would
        # charge the spread twice. Use a zero slippage layer in that case;
        # per-share commission still applies independently.
        execution_price = preset.get("execution", {}).get("execution_price")
        if execution_price == "quote_side":
            preset["slippage"]["model"] = "percentage"
            preset["slippage"]["rate"] = 0.0
        else:
            preset["slippage"]["model"] = "spread"
            preset["slippage"]["spread_convention"] = costs.get("spread_convention", "half_spread")

            spread_by_asset: dict[str, float] = {}
            asset_spreads_source = costs.get("asset_spreads_source")
            if asset_spreads_source:
                spreads_path = get_case_study_dir(case_study, create=False) / asset_spreads_source
                spread_col = costs.get("asset_spreads_column", "median_half_spread_usd")
                spreads_df = pl.read_parquet(spreads_path)
                spread_by_asset = dict(
                    zip(
                        spreads_df["symbol"].to_list(),
                        [float(x) for x in spreads_df[spread_col].to_list()],
                    )
                )
            elif "asset_spreads" in costs:
                spread_by_asset = {str(k): float(v) for k, v in costs["asset_spreads"].items()}
            if spread_by_asset:
                preset["slippage"]["spread_by_asset"] = spread_by_asset
            if "default_half_spread_usd" in costs:
                preset["slippage"]["spread"] = float(costs["default_half_spread_usd"])
    else:
        raise ValueError(
            f"Unknown costs.model {cost_model!r}. Supported: 'percentage', 'per_share_plus_spread'."
        )

    # Share-type comes from setup.yaml::execution.share_type via case_config —
    # never hardcoded per-cost-model branch. Falls back to the preset JSON's
    # value (if any) when case_config.share_type is the default placeholder.
    share_type = getattr(case_config, "share_type", None)
    if share_type:
        preset["position_sizing"]["share_type"] = share_type

    preset["calendar"]["calendar"] = case_config.calendar
    preset["calendar"].setdefault("data_frequency", _infer_data_frequency(case_config.cadence))
    derived_feed = _build_feed_spec(case_study, prices, case_config)
    explicit_feed = preset.get("feed", {})
    preset["feed"] = {
        **derived_feed,
        **{k: v for k, v in explicit_feed.items() if v is not None},
    }

    metadata = dict(preset.get("metadata", {}))
    metadata.update(
        {
            "case_study": case_study,
            "chapter": strategy_spec.get("chapter"),
            "cadence": strategy_spec.get("execution", {}).get("cadence", case_config.cadence),
            "fill_timing": fill_timing,
            "preset_path": str(preset_path(case_study)),
            "signal_direction": signal_direction,
        }
    )
    preset["metadata"] = metadata

    return EngineBacktestConfig.from_dict(preset, preset_name=preset_path(case_study).stem)


def _serialize_backtest_config(config: EngineBacktestConfig | dict[str, Any]) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return dict(config)


def runtime_backtest_config(spec: dict[str, Any]) -> EngineBacktestConfig:
    if EngineBacktestConfig is None:  # pragma: no cover - import depends on env
        raise ImportError("ml4t-backtest is required for backtest config resolution")
    runtime = EngineBacktestConfig.from_dict(spec["backtest_config"])
    spec["_runtime_backtest_config"] = runtime
    return runtime


# Calendars that require session enforcement (drop bars outside trading
# sessions, e.g. CME Saturdays). Mirror of the rule in backtest_runner._run_engine;
# applied at spec construction so plan-time hashes match registered hashes.
SESSION_ENFORCED_CALENDARS = frozenset({"CME", "us_futures"})


def apply_calendar_session_enforcement(config: EngineBacktestConfig, calendar: str | None) -> None:
    """Set ``enforce_sessions=True`` on ``config`` when ``calendar`` requires it.

    Without this, ``ensure_backtest_spec`` would produce a plan-time spec
    with ``enforce_sessions=False`` while ``_run_engine`` later mutates the
    same runtime to ``True``, breaking ``_runtime_backtest_config`` hash
    stability.
    """
    if calendar in SESSION_ENFORCED_CALENDARS:
        config.enforce_sessions = True


def serializable_backtest_spec(spec: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(spec)
    clean.pop("_runtime_backtest_config", None)
    if "backtest_config" in clean:
        clean["backtest_config"] = _serialize_backtest_config(clean["backtest_config"])
    return clean


def is_backtest_spec(spec: dict[str, Any]) -> bool:
    """Return True if ``spec`` is in canonical form (has ``strategy`` + ``backtest_config``)."""
    return spec.get("version") == 2 and "strategy" in spec and "backtest_config" in spec


def ensure_backtest_spec(
    case_study: str,
    case_config: CaseStudyBacktestConfig,
    strategy_spec: dict[str, Any],
    *,
    prices: pl.DataFrame,
    prediction_hash: str,
    initial_cash: float,
) -> dict[str, Any]:
    """Normalize ``strategy_spec`` to the canonical backtest spec form.

    Idempotent: if ``strategy_spec`` is already canonical, it is returned with
    a refreshed ``_runtime_backtest_config``. Otherwise, a flat strategy_spec
    (with ``signal`` / ``execution`` / ``costs`` / etc. blocks) is projected
    into the canonical envelope: ``strategy.{signal, rebalance, allocation, risk}``
    plus a resolved ``backtest_config`` block.

    Rebalance thresholds (``min_weight_change``, ``min_trade_value``) are
    always populated in ``strategy.rebalance`` — taken from ``execution.*``
    when present, otherwise from ``case_config`` (which sources them from
    ``setup.yaml::backtest.rebalance.default``).
    """
    if is_backtest_spec(strategy_spec):
        spec = deepcopy(strategy_spec)
        spec.setdefault(
            "chapter", spec.get("backtest_config", {}).get("metadata", {}).get("chapter")
        )
        # Ensure rebalance thresholds are populated; specs that omit them
        # would otherwise raise KeyError on `rebalance_spec["min_weight_change"]`.
        rb = spec.setdefault("strategy", {}).setdefault("rebalance", {})
        rb.setdefault("min_weight_change", float(getattr(case_config, "min_weight_change", 0.005)))
        rb.setdefault("min_trade_value", float(getattr(case_config, "min_trade_value", 100.0)))
        if "backtest_config" in spec:
            # Always overwrite metadata.prediction_hash with the caller's
            # argument — the spec may have been cloned from another run
            # (e.g. Ch20 holdout reuses the validation rank-1 spec with a
            # fresh holdout pred_hash), and downstream split-resolution
            # depends on the metadata reflecting the prediction set actually
            # being backtested.
            metadata = spec["backtest_config"].setdefault("metadata", {})
            metadata["prediction_hash"] = prediction_hash
            runtime = EngineBacktestConfig.from_dict(spec["backtest_config"])
            # Only the session-enforced calendars actually mutate ``runtime``
            # here; for all other calendars the from_dict/to_dict round-trip
            # would be a silent re-serialize that could perturb hashes for
            # any CS whose dict shape differs from the dataclass defaults
            # (None→0 normalization, dropped unknown keys, etc.). Confine the
            # ``backtest_config`` overwrite to the CSes where it is needed.
            if case_config.calendar in SESSION_ENFORCED_CALENDARS:
                apply_calendar_session_enforcement(runtime, case_config.calendar)
                # Re-serialize so the canonical ``backtest_config`` matches
                # the runtime; preserve every caller-supplied metadata key
                # by merging the original metadata back over the dataclass's
                # serialized view (the dataclass typically pins a schema
                # and drops unknown keys).
                original_metadata = dict(metadata)
                rebuilt = runtime.to_dict()
                rebuilt_metadata = dict(rebuilt.get("metadata") or {})
                rebuilt_metadata.update(original_metadata)
                rebuilt["metadata"] = rebuilt_metadata
                spec["backtest_config"] = rebuilt
            spec["_runtime_backtest_config"] = runtime
        return spec

    execution = deepcopy(strategy_spec.get("execution", {}))
    rebalance = {
        "mode": execution.get("mode", "engine"),
        "cadence": execution.get("cadence", case_config.cadence),
        "min_weight_change": float(
            execution["min_weight_change"]
            if "min_weight_change" in execution
            else getattr(case_config, "min_weight_change", 0.005)
        ),
        "min_trade_value": float(
            execution["min_trade_value"]
            if "min_trade_value" in execution
            else getattr(case_config, "min_trade_value", 100.0)
        ),
    }
    strategy = {
        "signal": deepcopy(strategy_spec.get("signal", {})),
        "rebalance": rebalance,
    }
    if "allocation" in strategy_spec:
        strategy["allocation"] = deepcopy(strategy_spec["allocation"])
    if "risk" in strategy_spec:
        strategy["risk"] = deepcopy(strategy_spec["risk"])

    resolved_config = build_resolved_backtest_config(
        case_study,
        case_config,
        strategy_spec,
        prices=prices,
        initial_cash=initial_cash,
    )
    resolved_config.metadata["prediction_hash"] = prediction_hash
    apply_calendar_session_enforcement(resolved_config, case_config.calendar)
    resolved_config_dict = resolved_config.to_dict()

    return {
        "version": 2,
        "chapter": strategy_spec.get("chapter"),
        "preset_id": f"{case_study}:base",
        "strategy": strategy,
        "backtest_config": resolved_config_dict,
        "_runtime_backtest_config": resolved_config,
    }


_COST_PASSTHROUGH_KEYS = (
    "model",
    "per_share",
    "minimum",
    "max_pct",
    "asset_spreads_source",
    "asset_spreads_column",
    "asset_spreads",
    "default_half_spread_usd",
    "spread_convention",
)


def _costs_block_from_case_config(
    case_config: CaseStudyBacktestConfig,
) -> dict[str, Any]:
    """Build the strategy_spec.costs block from the case config.

    For the percentage model (default), emit the bps form. For the
    per_share_plus_spread model, forward the full costs schema from setup.yaml
    so build_resolved_backtest_config can dispatch.
    """
    raw_costs = getattr(case_config, "raw_costs", None) or {}
    cost_model = raw_costs.get("model", "percentage")
    if cost_model == "per_share_plus_spread":
        return {key: deepcopy(raw_costs[key]) for key in _COST_PASSTHROUGH_KEYS if key in raw_costs}
    return {
        "commission_bps": case_config.commission_bps,
        "slippage_bps": case_config.slippage_bps,
    }


def build_backtest_spec(
    case_study: str,
    case_config: CaseStudyBacktestConfig,
    *,
    prices: pl.DataFrame,
    prediction_hash: str,
    initial_cash: float,
    signal: dict[str, Any],
    allocation: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    chapter: str | None = None,
    execution_mode: str | None = None,
    min_weight_change: float | None = None,
    min_trade_value: float | None = None,
) -> dict[str, Any]:
    strategy_spec: dict[str, Any] = {
        "signal": deepcopy(signal),
        "execution": {
            "mode": (
                execution_mode
                if execution_mode is not None
                else "vectorized"
                if case_study in {"us_firm_characteristics", "sp500_options"}
                else "engine"
            ),
            "engine_preset": "realistic",
            "cadence": case_config.cadence,
            "fill_timing": case_config.execution_delay.upper().replace(" ", "_"),
            "min_weight_change": (
                min_weight_change
                if min_weight_change is not None
                else getattr(case_config, "min_weight_change", 0.005)
            ),
            "min_trade_value": (
                min_trade_value
                if min_trade_value is not None
                else getattr(case_config, "min_trade_value", 100.0)
            ),
        },
        "costs": _costs_block_from_case_config(case_config),
    }
    if chapter is not None:
        strategy_spec["chapter"] = chapter
    if allocation is not None:
        strategy_spec["allocation"] = deepcopy(allocation)
    if risk is not None:
        strategy_spec["risk"] = deepcopy(risk)
    return ensure_backtest_spec(
        case_study,
        case_config,
        strategy_spec,
        prices=prices,
        prediction_hash=prediction_hash,
        initial_cash=initial_cash,
    )


def clone_backtest_spec(spec: dict[str, Any]) -> dict[str, Any]:
    cloned = serializable_backtest_spec(spec)
    if is_backtest_spec(cloned):
        cloned["_runtime_backtest_config"] = EngineBacktestConfig.from_dict(
            cloned["backtest_config"]
        )
    return cloned


def set_backtest_costs_bps(
    spec: dict[str, Any],
    *,
    commission_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    if not is_backtest_spec(spec):
        updated = deepcopy(spec)
        updated["costs"] = {
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
        }
        return updated

    updated = deepcopy(spec)
    bt_cfg = updated["backtest_config"]
    bt_cfg["commission"] = {
        "model": "percentage",
        "rate": commission_bps / 10_000.0,
    }
    bt_cfg["slippage"] = {
        "model": "percentage",
        "rate": slippage_bps / 10_000.0,
    }
    updated["_runtime_backtest_config"] = EngineBacktestConfig.from_dict(bt_cfg)
    return updated


def set_backtest_costs_per_share(
    spec: dict[str, Any],
    *,
    per_share: float,
    default_half_spread_usd: float,
    asset_spreads: dict[str, float] | None = None,
    spread_convention: str = "half_spread",
    minimum: float = 0.0,
) -> dict[str, Any]:
    """Mutate spec to use per-share commission + spread slippage.

    Switches the engine commission/slippage models from `percentage` to
    `per_share` / `spread`. Safe to call on a spec that originally used the
    percentage model — the previous rate fields are replaced. Used by the
    cost-sensitivity sweep to walk the per-share+spread regime alongside
    the bps regime for case studies whose dataset supports it (those with
    prices and integer-share semantics).

    `minimum` is the per-order commission floor in dollars and defaults to
    `0.0` (no floor). Pass `minimum=0.35` to match the IBKR Pro per-order
    floor used by `build_resolved_backtest_config`.
    """
    if not is_backtest_spec(spec):
        updated = deepcopy(spec)
        updated["costs"] = {
            "model": "per_share_plus_spread",
            "per_share": float(per_share),
            "default_half_spread_usd": float(default_half_spread_usd),
            "asset_spreads": dict(asset_spreads or {}),
            "spread_convention": spread_convention,
            "minimum": float(minimum),
        }
        return updated

    updated = deepcopy(spec)
    bt_cfg = updated["backtest_config"]
    bt_cfg["commission"] = {
        "model": "per_share",
        "rate": 0.0,
        "per_share": float(per_share),
        "minimum": float(minimum),
        "per_trade": 0.0,
    }
    bt_cfg["slippage"] = {
        "model": "spread",
        "rate": 0.0,
        "spread": float(default_half_spread_usd),
        "spread_convention": spread_convention,
    }
    if asset_spreads:
        bt_cfg["slippage"]["spread_by_asset"] = {str(k): float(v) for k, v in asset_spreads.items()}
    bt_cfg.setdefault("position_sizing", {})["share_type"] = "integer"
    updated["_runtime_backtest_config"] = EngineBacktestConfig.from_dict(bt_cfg)
    return updated


def strategy_view(spec: dict[str, Any]) -> dict[str, Any]:
    return spec["strategy"] if is_backtest_spec(spec) else spec


def cost_view(spec: dict[str, Any]) -> dict[str, Any]:
    if not is_backtest_spec(spec):
        return spec.get("costs", {})
    cfg = spec.get("backtest_config", {})
    commission = cfg.get("commission", {})
    slippage = cfg.get("slippage", {})
    return {
        "commission_bps": round(float(commission.get("rate", 0.0)) * 10_000.0, 10),
        "slippage_bps": round(float(slippage.get("rate", 0.0)) * 10_000.0, 10),
    }
