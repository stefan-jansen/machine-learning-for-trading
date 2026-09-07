"""A price panel narrower than the predictions is reported, not acted on.

ml4t/agent-workspace#911. The vectorized path computes `gross_ret = weight *
y_true` from the predictions frame and reads `prices` only for the rebalance
calendar - the same set of decision dates whichever symbols are in the panel. So
a preview taken at a reduced `MAX_SYMBOLS` was the production sweep at the
production cost per backtest: measured on us_firm_characteristics/11_backtest,
32 backtests at 300 symbols and at 3,708 agreed in every column to six decimals.

Both ways of acting on it are wrong here, and each was tried. Narrowing the
predictions to the panel makes the traded universe decide the portfolio without
entering the backtest identity - the caller hashes its specification before this
module sees the run, so a reduced preview would be served the full-universe
result. Refusing the run stops a preview that is legitimately configured this
way: the CI `us_firm_characteristics` fixture holds a 5-symbol panel against
20-symbol predictions, and refusing took four notebooks down. What closes it is
the caller declaring the universe it trades, before it hashes, which is a
notebook change.
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


def test_a_panel_that_cannot_price_the_predictions_says_so(monkeypatch) -> None:
    """The defect: the panel was reduced and the sweep ran over all four names."""
    with pytest.warns(UserWarning) as record:
        captured = _run(monkeypatch, _prices(["A", "C"]))
    message = str(record[0].message)
    assert "2 of which it cannot price" in message
    # Names the knob that does reduce this stage, so the reader is not left to guess.
    assert "TOP_N_PREDICTIONS" in message
    # And says it without changing the run: the identity the caller already hashed
    # describes a sweep over all four names, and that is what it gets.
    assert captured["predictions"]["symbol"].to_list() == ["A", "B", "C", "D"]


def test_a_covering_panel_is_silent(monkeypatch) -> None:
    """The control: nothing is reported when the panel carries every predicted name."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        captured = _run(monkeypatch, _prices(["A", "B", "C", "D", "E"]))
    assert captured["predictions"]["symbol"].to_list() == ["A", "B", "C", "D"]
    assert sorted(captured["weights"]["symbol"].to_list()) == ["A", "B"]


def test_a_precomputed_allocation_is_reported_on_too(monkeypatch) -> None:
    """The Ch19 risk sweep reads y_true from the same predictions frame."""
    weights = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.25, 0.25, 0.25, 0.25],
        }
    )
    with pytest.warns(UserWarning, match="cannot price"):
        _run(monkeypatch, _prices(["A", "C"]), precomputed_weights=weights)


def test_an_empty_panel_says_nothing() -> None:
    """No panel is not a reduction to zero; the engine's own guards cover it."""
    import warnings as _warnings

    from case_studies.utils.backtest_runner import warn_if_the_panel_does_not_bound_the_universe

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        warn_if_the_panel_does_not_bound_the_universe(
            _predictions(), pl.DataFrame(), case_study="demo", label="fwd_ret_1m"
        )


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
    monkeypatch.setattr(br, "declared_rebalance_step", lambda *_: None)
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


def test_the_notebooks_blanket_warning_filter_does_not_hide_it(capsys) -> None:
    """Eleven backtest notebooks ignore warnings at import; the reader still sees this.

    `us_firm_characteristics/11_backtest.py:62` and ten siblings call
    `warnings.filterwarnings("ignore")` before the first backtest runs, so a
    diagnostic that only warns reaches nobody who reads the executed notebook.
    This asserts the message survives that filter, which is the condition the
    notebook actually runs under - `pytest.warns` in the tests above installs its
    own filter and cannot see the difference.
    """
    import warnings as _warnings

    from case_studies.utils.backtest_runner import warn_if_the_panel_does_not_bound_the_universe

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore")
        warn_if_the_panel_does_not_bound_the_universe(
            _predictions(), _prices(["A", "B"]), case_study="filtered", label="fwd_ret_1m"
        )
    assert "cannot price" in capsys.readouterr().out


def test_a_sweep_over_one_prediction_set_says_it_once(capsys) -> None:
    """A twelve-scheme sweep calls `run_backtest` twelve times over one panel.

    Each call sees the same panel and the same predictions, so twelve copies of
    one diagnostic would bury the cell output it is printed into. The case study
    name here is distinct from every other test in this file, so what is counted
    is this loop and not a report some earlier test already made.
    """
    import warnings as _warnings

    from case_studies.utils.backtest_runner import warn_if_the_panel_does_not_bound_the_universe

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore")
        for _ in range(12):
            warn_if_the_panel_does_not_bound_the_universe(
                _predictions(), _prices(["A", "B"]), case_study="swept", label="fwd_ret_1m"
            )
    assert capsys.readouterr().out.count("cannot price") == 1
