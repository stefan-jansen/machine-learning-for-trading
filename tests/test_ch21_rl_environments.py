"""Unit tests for the Chapter 21 RL environment modules.

These modules shipped to readers as broken imports (they were never committed),
and review of the restored files found four defects. Each is pinned here.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "21_rl_execution_hedging"))

from crypto_execution_env import CryptoExecutionEnv  # noqa: E402
from market_making_env import (  # noqa: E402
    MarketMakingDynamics,
    generate_garch_market_data,
)

DYNAMICS = MarketMakingDynamics(
    garch_omega=1e-6,
    garch_alpha=0.3,
    garch_beta=0.6,
    unconditional_vol=0.01,
    skew_levels=(-1.0, 0.0, 1.0),
    spread_multipliers=(0.5, 1.0, 2.0),
)


def market_data(n: int = 200, volume: float = 1.0) -> pl.DataFrame:
    base = dt.datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "timestamp": [base + dt.timedelta(hours=i) for i in range(n)],
            "open": np.linspace(100.0, 110.0, n),
            "observed_volume": np.full(n, volume),
            "avg_volume_24h": np.full(n, volume),
            "volatility_24h": np.full(n, 0.02),
            "premium_index_close": np.zeros(n),
            "hour": np.arange(n) % 24,
            "hours_to_funding": np.arange(n) % 8,
        }
    )


def test_volatility_series_is_conditional_not_realized() -> None:
    """The volatility the agent observes at t must not embed the shock that
    moves the price to t+1. Appending the post-update GARCH variance did."""
    prices, vols, _ = generate_garch_market_data(500, np.random.default_rng(0), DYNAMICS)
    # The first entry is the seeded conditional vol, before any shock is drawn.
    assert np.isclose(vols[0], DYNAMICS.unconditional_vol)
    # Reproduce the recursion independently and check every entry is the
    # variance *before* that step's return is folded in.
    rng = np.random.default_rng(0)
    variance = DYNAMICS.unconditional_vol**2
    for i in range(len(vols)):
        assert np.isclose(vols[i], np.sqrt(variance)), f"vol[{i}] is not the conditional vol"
        shock = np.clip(rng.standard_normal(), -5, 5)
        return_t = np.clip(np.sqrt(variance) * shock, -0.1, 0.1)
        variance = np.clip(
            DYNAMICS.garch_omega
            + DYNAMICS.garch_alpha * return_t**2
            + DYNAMICS.garch_beta * variance,
            1e-10,
            0.01,
        )
        rng.uniform(-1, 1)  # imbalance draw, keeps the stream aligned
    assert len(prices) == len(vols) + 1


def test_impact_is_square_root_plus_linear() -> None:
    """The notebook documents a square-root-plus-linear model; the code used
    participation_rate ** 2 for the linear term."""
    env = CryptoExecutionEnv(market_data(), total_shares=10.0, horizon=24, seed=0)
    market = env._get_market_state(env.start_idx)
    shares = 0.5
    price, _ = env._trade_metrics(market, shares)
    participation = shares / (market.volume + 1e-8)
    expected = env.impact_coefficient * (np.sqrt(participation) + participation)
    assert np.isclose(price, market.price * (1 - expected))


def test_history_describes_the_executed_trade() -> None:
    """reference_shares / max_trade_shares must be the pre-trade values, not
    recomputed after inventory was reduced."""
    env = CryptoExecutionEnv(market_data(), total_shares=50.0, horizon=24, seed=1)
    env.reset(seed=1)
    env.step(np.array([1.0], dtype=np.float32))
    row = env.execution_history[0]
    assert row["max_trade_shares"] >= row["shares_sold"], (
        "reported cap is below the size actually executed"
    )
    assert row["reference_shares"] > 0


def test_forced_liquidation_is_reachable_under_thin_liquidity() -> None:
    """The final step used to bypass the participation cap and sell everything,
    making the forced-liquidation branch (and the notebook's diagnostic at
    04_crypto_execution_rl.py:291) unreachable."""
    env = CryptoExecutionEnv(
        market_data(volume=1.0), total_shares=500.0, horizon=24, max_participation_rate=0.10, seed=3
    )
    env.reset(seed=3)
    for _ in range(30):
        _, _, terminated, _, _ = env.step(np.array([1.0], dtype=np.float32))
        if terminated:
            break
    assert any(h.get("forced_liquidation", False) for h in env.execution_history)


def test_ample_liquidity_completes_without_forcing() -> None:
    env = CryptoExecutionEnv(
        market_data(volume=1.0), total_shares=1.0, horizon=24, max_participation_rate=0.10, seed=3
    )
    env.reset(seed=3)
    for _ in range(30):
        _, _, terminated, _, _ = env.step(np.array([1.0], dtype=np.float32))
        if terminated:
            break
    assert not any(h.get("forced_liquidation", False) for h in env.execution_history)


def test_reset_window_covers_the_last_valid_start() -> None:
    """The upper bound excluded the final two otherwise-valid windows."""
    env = CryptoExecutionEnv(market_data(n=24), total_shares=10.0, horizon=24, seed=0)
    assert env.start_idx == 0


def test_dataset_shorter_than_horizon_raises_clearly() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        CryptoExecutionEnv(market_data(n=10), total_shares=10.0, horizon=24, seed=0)
