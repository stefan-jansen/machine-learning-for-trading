"""The chapter-16 ETF baseline helper must reproduce the notebook that derives it.

`16_strategy_simulation/_etf_baseline.py` exists so the economic-diagnostic notebooks can
score the same strategy `01_backtest_first_principles` builds without restating it. That is
only useful if it produces the same account, and before 2026-08-20 it did not: it filled at
the close rather than the next open, let cash go negative, and rebalanced only when the
target moved by more than one percent, which left it 0.09 of Sharpe away from the notebook
whose numbers its docstring claimed to match.

This test re-derives the notebook's simulation from its own inputs and asserts the helper
agrees, so the claim cannot go stale again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "16_strategy_simulation"
sys.path.insert(0, str(CHAPTER))

pytestmark = pytest.mark.skipif(
    not (CHAPTER / "_etf_baseline.py").exists(), reason="chapter 16 not shipped in this checkout"
)


def _reference_equity(panel, weights, fees: float) -> np.ndarray:
    """The simulator as `01_backtest_first_principles` writes it, in NumPy."""
    opens = panel.opens.to_numpy()
    closes = panel.prices.to_numpy()
    targets = weights.to_numpy()

    from _etf_baseline import INITIAL_CASH, fill_dates

    schedule = panel.prices.index.isin(fill_dates(panel.prices.index))
    holdings = np.zeros(closes.shape[1])
    cash = INITIAL_CASH
    equity = np.zeros(len(closes))

    for i in range(len(closes)):
        if schedule[i]:
            open_values = holdings * opens[i]
            target_values = targets[i] * (cash + open_values.sum())
            sells = np.maximum(open_values - target_values, 0.0)
            holdings -= sells / opens[i]
            cash += sells.sum() * (1 - fees)
            requested = np.maximum(target_values - holdings * opens[i], 0.0)
            required = requested.sum() * (1 + fees)
            scale = min(1.0, cash / required) if required > 0 else 0.0
            buys = requested * scale
            holdings += buys / opens[i]
            cash -= buys.sum() * (1 + fees)
        assert cash >= -1e-8, f"cash went negative at bar {i}"
        cash = max(cash, 0.0)
        equity[i] = cash + float(holdings @ closes[i])
    return equity


@pytest.fixture(scope="module")
def baseline():
    from _etf_baseline import load_panel, momentum_weights

    panel = load_panel()
    return panel, momentum_weights(panel)


@pytest.mark.parametrize("fees", [0.0, 0.0005, 0.01])
def test_helper_matches_the_notebook_simulator(baseline, fees):
    from _etf_baseline import simulate

    panel, weights = baseline
    np.testing.assert_allclose(
        simulate(panel, weights, fees=fees).equity.to_numpy(),
        _reference_equity(panel, weights, fees),
        rtol=0,
        atol=1e-8,
    )


def test_weights_are_executable_and_fully_invested(baseline):
    _, weights = baseline
    assert not weights.isna().any().any()
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, rtol=0, atol=1e-12)


def test_the_account_never_borrows(baseline):
    from _etf_baseline import simulate

    panel, weights = baseline
    result = simulate(panel, weights, fees=0.01)
    assert result.equity.min() > 0
    assert (result.holdings_value.sum(axis=1) <= result.equity + 1e-6).all()


def test_metrics_use_the_library_convention_not_a_growth_rate(baseline):
    """Sharpe must put the mean periodic return in the numerator, not the growth rate.

    The helper used to compute `cagr / vol`, which is the substitution
    `01_backtest_first_principles` explicitly warns against: the arithmetic mean exceeds the
    geometric one by roughly half the variance, so the two differ by a material amount on
    any volatile series and only the first is a Sharpe ratio.
    """
    from _etf_baseline import metrics, simulate
    from ml4t.diagnostic.metrics import sharpe_ratio

    panel, weights = baseline
    result = simulate(panel, weights)
    reported = metrics(result)

    assert np.isclose(
        reported["sharpe"], sharpe_ratio(result.returns.to_numpy(), periods_per_year=252)
    )
    assert not np.isclose(reported["sharpe"], reported["cagr"] / reported["vol"], atol=1e-3)


def test_the_return_series_covers_every_session(baseline):
    """One return per session, including the first, which `pct_change` alone cannot produce."""
    from _etf_baseline import simulate

    panel, weights = baseline
    result = simulate(panel, weights)
    assert len(result.returns) == len(panel.prices)
    assert result.returns.notna().all()
