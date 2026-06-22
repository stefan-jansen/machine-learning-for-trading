from __future__ import annotations

from datetime import datetime

import polars as pl

from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices
from case_studies.utils.backtest_presets import (
    cost_view,
    ensure_backtest_spec,
    is_backtest_spec,
    load_backtest_preset,
    strategy_view,
)
from case_studies.utils.backtest_runner import normalize_prediction_columns
from case_studies.utils.registry.specs import backtest_hash_from_parts
from case_studies.utils.registry.store import _infer_stage


def test_etf_backtest_base_preset_exists() -> None:
    preset = load_backtest_preset("etfs")
    assert preset["calendar"]["calendar"] == "NYSE"
    assert preset["commission"]["rate"] == 0.0006
    assert preset["slippage"]["rate"] == 0.0004


def test_ensure_backtest_spec_builds_composite_spec() -> None:
    bt = get_backtest_config("etfs")
    prices = load_backtest_prices("etfs", max_symbols=2)
    legacy_spec = {
        "chapter": "ch18",
        "signal": {"method": "equal_weight_top_k", "top_k": 10, "long_short": False},
        "execution": {
            "mode": "engine",
            "engine_preset": "realistic",
            "cadence": bt.cadence,
            "fill_timing": bt.execution_delay.upper(),
        },
        "costs": {"commission_bps": 6.0, "slippage_bps": 4.0},
        "allocation": {"method": "risk_parity", "top_k": 10},
    }
    spec = ensure_backtest_spec(
        "etfs",
        bt,
        legacy_spec,
        prices=prices,
        prediction_hash="pred123",
        initial_cash=1_000_000.0,
    )

    assert is_backtest_spec(spec)
    assert spec["strategy"]["signal"]["method"] == "equal_weight_top_k"
    assert spec["strategy"]["allocation"]["method"] == "risk_parity"
    assert spec["strategy"]["rebalance"]["cadence"] == bt.cadence
    assert spec["backtest_config"]["commission"]["rate"] == 0.0006
    assert spec["backtest_config"]["slippage"]["rate"] == 0.0004
    assert spec["backtest_config"]["metadata"]["prediction_hash"] == "pred123"
    assert cost_view(spec) == {"commission_bps": 6.0, "slippage_bps": 4.0}


def test_backtest_hash_changes_with_resolved_config() -> None:
    bt = get_backtest_config("etfs")
    prices = load_backtest_prices("etfs", max_symbols=2)
    base = {
        "signal": {"method": "equal_weight_top_k", "top_k": 10, "long_short": False},
        "execution": {
            "mode": "engine",
            "engine_preset": "realistic",
            "cadence": bt.cadence,
            "fill_timing": bt.execution_delay.upper(),
        },
        "costs": {"commission_bps": 6.0, "slippage_bps": 4.0},
    }
    cheap = ensure_backtest_spec(
        "etfs",
        bt,
        base,
        prices=prices,
        prediction_hash="pred123",
        initial_cash=1_000_000.0,
    )
    expensive = ensure_backtest_spec(
        "etfs",
        bt,
        {**base, "costs": {"commission_bps": 10.0, "slippage_bps": 4.0}},
        prices=prices,
        prediction_hash="pred123",
        initial_cash=1_000_000.0,
    )

    assert backtest_hash_from_parts("pred123", cheap) != backtest_hash_from_parts(
        "pred123", expensive
    )


def test_stage_inference_supports_v2_specs() -> None:
    spec = {
        "version": 2,
        "chapter": "ch19",
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 10},
            "rebalance": {"mode": "engine", "cadence": "monthly_month_end"},
            "risk": {"name": "trailing", "position_rules": [{"type": "trailing_stop"}]},
        },
        "backtest_config": {"commission": {"rate": 0.0006}, "slippage": {"rate": 0.0004}},
    }
    assert _infer_stage(spec) == "risk_overlay"
    assert strategy_view(spec)["risk"]["name"] == "trailing"


def test_microstructure_preset_auto_loads_quote_columns() -> None:
    prices = load_backtest_prices("nasdaq100_microstructure", max_symbols=2)

    assert "bid_open" in prices.columns
    assert "ask_open" in prices.columns


def test_sp500_options_short_only_engine_spec_preserves_explicit_feed() -> None:
    bt = get_backtest_config("sp500_options")
    prices = load_backtest_prices("sp500_options", max_symbols=2)
    legacy_spec = {
        "chapter": "ch16",
        "signal": {
            "method": "score_weighted_top_k",
            "top_k": 10,
            "long_short": False,
            "direction": "short_only",
        },
        "execution": {
            "mode": "engine",
            "engine_preset": "realistic",
            "cadence": bt.cadence,
            "fill_timing": bt.execution_delay.upper(),
        },
        "costs": {"commission_bps": bt.commission_bps, "slippage_bps": bt.slippage_bps},
    }
    spec = ensure_backtest_spec(
        "sp500_options",
        bt,
        legacy_spec,
        prices=prices,
        prediction_hash="pred123",
        initial_cash=1_000_000.0,
    )

    cfg = spec["backtest_config"]
    assert spec["strategy"]["rebalance"]["mode"] == "engine"
    assert cfg["account"]["allow_short_selling"] is True
    assert cfg["execution"]["execution_price"] == "quote_side"
    assert cfg["execution"]["mark_price"] == "price"
    assert cfg["feed"]["price_col"] == "instr_mid"
    assert cfg["feed"]["bid_col"] == "instr_bid"
    assert cfg["feed"]["ask_col"] == "instr_ask"


def test_ensure_backtest_spec_pins_enforce_sessions_for_cme_calendar() -> None:
    """CME-calendar specs must set enforce_sessions=True at construction time.

    Regression: without this, ensure_backtest_spec emits a runtime with
    enforce_sessions=False, but _run_engine later mutates it to True, so the
    plan-time and post-engine hashes diverge (verify fires "0/N in_registry").
    BacktestConfig.to_dict() does NOT serialize enforce_sessions; the hash
    picks it up through ``_runtime_backtest_config`` in the spec (canonical_json
    uses default=str on the dataclass repr), so the runtime is the load-bearing
    surface and what we pin here.

    Also pins that the NYSE projection branch does NOT trip the re-serialize
    path — its original ``backtest_config`` dict and metadata are preserved.
    """
    bt_cme = get_backtest_config("cme_futures")
    prices_cme = load_backtest_prices("cme_futures", max_symbols=2)

    canonical_cme = {
        "version": 2,
        "chapter": "ch18",
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 5, "long_short": True},
            "rebalance": {"mode": "engine", "cadence": bt_cme.cadence},
        },
        "backtest_config": {
            "commission": {"rate": 0.0},
            "slippage": {"rate": 0.0},
            "metadata": {"chapter": "ch18", "extra": "preserved"},
        },
    }
    spec_cme = ensure_backtest_spec(
        "cme_futures",
        bt_cme,
        canonical_cme,
        prices=prices_cme,
        prediction_hash="pred_cme",
        initial_cash=1_000_000.0,
    )
    assert spec_cme["_runtime_backtest_config"].enforce_sessions is True
    # Caller-supplied metadata keys must survive the re-serialization.
    assert spec_cme["backtest_config"]["metadata"]["extra"] == "preserved"
    assert spec_cme["backtest_config"]["metadata"]["prediction_hash"] == "pred_cme"

    # NYSE specs (etfs) must NOT trip the projection re-serialize path —
    # the original backtest_config dict should be preserved untouched.
    bt_etf = get_backtest_config("etfs")
    prices_etf = load_backtest_prices("etfs", max_symbols=2)
    canonical_etf = {
        "version": 2,
        "chapter": "ch18",
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 10, "long_short": False},
            "rebalance": {"mode": "engine", "cadence": bt_etf.cadence},
        },
        "backtest_config": {
            "commission": {"rate": 0.0006},
            "slippage": {"rate": 0.0004},
            "metadata": {"chapter": "ch18", "extra": "preserved"},
        },
    }
    spec_etf = ensure_backtest_spec(
        "etfs",
        bt_etf,
        canonical_etf,
        prices=prices_etf,
        prediction_hash="pred_etf",
        initial_cash=1_000_000.0,
    )
    assert spec_etf["_runtime_backtest_config"].enforce_sessions is False
    # Original dict preserved (we did not re-serialize) — keys the caller
    # passed in are exactly the keys present.
    assert set(spec_etf["backtest_config"].keys()) == {"commission", "slippage", "metadata"}
    assert spec_etf["backtest_config"]["metadata"]["extra"] == "preserved"
    assert spec_etf["backtest_config"]["metadata"]["prediction_hash"] == "pred_etf"


def test_normalize_prediction_columns_maps_causal_and_legacy_fields() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2)],
            "symbol": ["AAPL"],
            "fold": [0],
            "actual": [0.01],
            "prediction": [0.02],
        }
    )

    normalized = normalize_prediction_columns(df)

    assert "y_score" in normalized.columns
    assert "y_true" in normalized.columns
    assert "fold_id" in normalized.columns
    assert normalized["y_score"].to_list() == [0.02]
    assert normalized["y_true"].to_list() == [0.01]
