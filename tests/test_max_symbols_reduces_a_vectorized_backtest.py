"""MAX_SYMBOLS reduces the vectorized backtest, not just the price frame.

ml4t/agent-workspace#911. The vectorized path computes `gross_ret = weight *
y_true` from the predictions frame, and reads `prices` only for the rebalance
calendar - the same set of decision dates whichever symbols are in the panel. So
a preview taken at a reduced `MAX_SYMBOLS` was the production sweep with the
production cost per backtest: measured on us_firm_characteristics/11_backtest,
32 backtests at 300 symbols and at 3,708 agreed in every column to six decimals.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

SPEC = {
    "version": 2,
    "strategy": {
        "signal": {"method": "equal_weight_top_k", "top_k": 2, "long_short": False},
        "rebalance": {"mode": "vectorized", "cadence": "daily", "step": 1},
    },
    "backtest_config": {
        "cash": {"initial": 100_000.0},
        "commission": {"model": "percentage", "rate": 0.0},
        "slippage": {"model": "percentage", "rate": 0.0},
        "account": {"allow_short_selling": False},
    },
}


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)] * 4,
            "symbol": ["A", "B", "C", "D"],
            "y_score": [0.9, 0.8, 0.7, 0.6],
            "y_true": [0.1, 0.2, 0.3, 0.4],
        }
    )


def _prices(symbols: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)] * len(symbols),
            "symbol": symbols,
            "close": [100.0] * len(symbols),
        }
    )


def _run(monkeypatch, prices, **kwargs) -> dict:
    """Call run_backtest and capture what the vectorized path was handed."""
    import case_studies.utils.backtest_runner as br
    import case_studies.utils.conformal as conformal

    captured: dict = {}

    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kw: args[2])
    monkeypatch.setattr(conformal, "ensure_conformal_calibration_identity", lambda spec: spec)
    monkeypatch.setattr(br, "substitute_continuous_return_for_classification", lambda p, *_: p)

    def fake_vectorized(**kw):
        captured.update(kw)
        return {
            "daily_returns": pl.DataFrame(
                {"timestamp": [datetime(2024, 1, 1)], "daily_return": [0.0]}
            ),
            "metrics": {"sharpe": 0.0},
        }

    monkeypatch.setattr(br, "_run_vectorized", fake_vectorized)

    br.run_backtest(
        "us_firm_characteristics",
        "pred1",
        SPEC,
        prices=prices,
        predictions=_predictions(),
        register=False,
        **kwargs,
    )
    return captured


def test_a_reduced_price_panel_reduces_the_backtest(monkeypatch) -> None:
    """The defect: the panel was reduced and the backtest ran over all four names.

    B outranks C, so a top-2 over the full cross-section holds {A, B}. Over the
    reduced panel it holds {A, C} - the selection is made inside the universe the
    panel defines, which is what the parameter reads as and what a preview needs
    it to mean.
    """
    captured = _run(monkeypatch, _prices(["A", "C"]))
    assert captured["predictions"]["symbol"].to_list() == ["A", "C"]
    assert sorted(captured["weights"]["symbol"].to_list()) == ["A", "C"]


def test_the_full_panel_leaves_the_predictions_alone(monkeypatch) -> None:
    """The control: nothing is dropped when the panel carries every predicted name."""
    captured = _run(monkeypatch, _prices(["A", "B", "C", "D", "E"]))
    assert captured["predictions"]["symbol"].to_list() == ["A", "B", "C", "D"]
    assert sorted(captured["weights"]["symbol"].to_list()) == ["A", "B"]


def test_a_precomputed_allocation_is_not_silently_narrowed(monkeypatch) -> None:
    """The Ch19 risk sweep hands in weights an allocator solved for.

    Dropping positions out of that would leave weights summing to something the
    allocator never chose, so the reduction applies where the selection is made.
    """
    weights = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.25, 0.25, 0.25, 0.25],
        }
    )
    captured = _run(monkeypatch, _prices(["A", "C"]), precomputed_weights=weights)
    assert captured["weights"]["symbol"].to_list() == ["A", "B", "C", "D"]
    assert captured["predictions"]["symbol"].to_list() == ["A", "B", "C", "D"]


def test_a_panel_that_prices_nothing_stops_the_run() -> None:
    from case_studies.utils.backtest_runner import restrict_to_priced_universe

    with pytest.raises(ValueError, match="would have no universe"):
        restrict_to_priced_universe(
            _predictions(), _prices(["X", "Y"]), case_study="demo", label="fwd_ret_1m"
        )


def test_an_empty_panel_is_left_to_the_engine() -> None:
    """No panel is not a reduction to zero; the engine's own guards cover it."""
    from case_studies.utils.backtest_runner import restrict_to_priced_universe

    predictions = _predictions()
    restricted = restrict_to_priced_universe(
        predictions, pl.DataFrame(), case_study="demo", label="fwd_ret_1m"
    )
    assert restricted.equals(predictions)


def test_the_htm_option_path_keeps_its_own_universe(monkeypatch) -> None:
    """sp500_options/ret_to_expiry indexes prices and predictions differently."""
    import case_studies.utils.backtest_runner as br
    import case_studies.utils.conformal as conformal

    captured: dict = {}
    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kw: args[2])
    monkeypatch.setattr(conformal, "ensure_conformal_calibration_identity", lambda spec: spec)
    monkeypatch.setattr(br, "substitute_continuous_return_for_classification", lambda p, *_: p)

    def fake_htm(**kw):
        captured.update(kw)
        return {
            "daily_returns": pl.DataFrame(
                {"timestamp": [datetime(2024, 1, 1)], "daily_return": [0.0]}
            ),
            "metrics": {"sharpe": 0.0},
        }

    monkeypatch.setattr(br, "_run_htm_daily_mtm", fake_htm)
    monkeypatch.setattr(
        br,
        "declared_rebalance_step",
        lambda *_: None,
    )
    br.run_backtest(
        "sp500_options",
        "pred1",
        SPEC,
        prices=_prices(["X"]),
        predictions=_predictions(),
        label="ret_to_expiry",
        register=False,
    )
    assert captured["predictions"]["symbol"].to_list() == ["A", "B", "C", "D"]
