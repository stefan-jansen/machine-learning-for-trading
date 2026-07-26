"""Unit tests for the Chapter 21 RL environment modules.

These modules shipped to readers as broken imports (they were never committed),
and review of the restored files found six defects. Each is pinned here.
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


def test_history_records_one_row_per_bar_even_when_liquidation_is_forced() -> None:
    """A forced liquidation happens against the final bar, not in an extra hour.

    It used to be appended as a second row carrying ``step == horizon``, so every
    consumer had to collapse the pair. Two did not: the notebook's "last step
    share" column read the involuntary remainder alone, and the trajectory figure
    plotted PPO over 25 hours of a 24-hour horizon while the baselines spanned
    24. The bar is now one row whose ``shares_sold`` is the whole bar and whose
    ``forced_shares`` isolates the involuntary part."""
    env = CryptoExecutionEnv(
        market_data(volume=1.0), total_shares=500.0, horizon=24, max_participation_rate=0.10, seed=3
    )
    env.reset(seed=3)
    for _ in range(30):
        _, _, terminated, _, _ = env.step(np.array([1.0], dtype=np.float32))
        if terminated:
            break

    steps = [int(h["step"]) for h in env.execution_history]
    assert steps == sorted(set(steps)), "a bar is recorded more than once"
    assert max(steps) <= env.horizon - 1, "history runs past the final bar"

    final_bar = env.execution_history[-1]
    assert final_bar["forced_liquidation"] is True
    assert 0 < float(final_bar["forced_shares"]) < float(final_bar["shares_sold"]), (
        "the bar must hold both the policy leg and the forced remainder"
    )
    assert float(final_bar["remaining"]) == 0.0
    assert sum(float(h["shares_sold"]) for h in env.execution_history) == pytest.approx(
        env.total_shares
    )
    assert env.remaining_shares == 0.0
    # Every row carries the same keys, so the aggregate evaluation table and the
    # parquet artifact built from it have one schema rather than nulls on the
    # rows the forced leg used to omit.
    assert {frozenset(h) for h in env.execution_history} == {frozenset(final_bar)}


def test_forced_remainder_is_not_charged_inventory_risk() -> None:
    """Risk prices exposure carried past the bar; a forced remainder is sold into
    that same bar, so it carries none. Charging it made the terminal reward depend
    on how the final order split between the policy leg and the forced leg."""
    rewards, forced, bar_volume = {}, {}, {}
    for final_action in (0.0, 0.25, 0.5, 0.75, 1.0):
        env = CryptoExecutionEnv(
            market_data(volume=50.0),
            total_shares=100.0,
            horizon=4,
            impact_coefficient=0.05,
            max_participation_rate=1.0,
            risk_aversion=1e-2,
            seed=0,
        )
        env.reset(seed=0)
        terminated, total_reward = False, 0.0
        while not terminated:
            action = final_action if env.step_idx == env.horizon - 1 else 0.5
            _, reward, terminated, _, _ = env.step(np.array([action], dtype=np.float32))
            total_reward += reward
        final_bar = env.execution_history[-1]
        assert sum(h["shares_sold"] for h in env.execution_history) == pytest.approx(100.0)
        assert final_bar["risk_penalty"] == 0.0, "the liquidated remainder was charged risk"
        rewards[final_action] = total_reward
        forced[final_action] = final_bar["forced_shares"]
        bar_volume[final_action] = final_bar["shares_sold"]

    # The final bar clears the same quantity in every case; only the voluntary /
    # forced split differs. That is the invariance the reward must respect.
    assert max(forced.values()) > 0 and min(forced.values()) == 0.0, (
        f"the split did not vary, so this asserts nothing: {forced}"
    )
    assert np.allclose(list(bar_volume.values()), next(iter(bar_volume.values())))
    values = list(rewards.values())
    assert np.allclose(values, values[0]), (
        f"splitting the terminal order changes total reward: {rewards}"
    )


def test_terminal_split_is_not_cheaper_than_one_trade() -> None:
    """Square-root impact is concave, so charging the forced-liquidation leg its
    own participation rate would make a low final action buy artificially good
    execution. Both legs are charged on the combined participation instead."""
    costs = []
    for final_action in (0.0, 0.25, 0.5, 0.75, 1.0):
        env = CryptoExecutionEnv(
            market_data(volume=50.0),
            total_shares=100.0,
            horizon=4,
            impact_coefficient=0.05,
            max_participation_rate=1.0,
            seed=0,
        )
        env.reset(seed=0)
        terminated = False
        while not terminated:
            action = final_action if env.step_idx == env.horizon - 1 else 0.5
            _, _, terminated, _, _ = env.step(np.array([action], dtype=np.float32))
        assert np.isclose(sum(h["shares_sold"] for h in env.execution_history), 100.0)
        costs.append(env.total_cost)
    assert np.allclose(costs, costs[0]), f"splitting the terminal order changes total cost: {costs}"


def test_reset_window_covers_the_last_valid_start() -> None:
    """The upper bound excluded the final two otherwise-valid windows."""
    env = CryptoExecutionEnv(market_data(n=24), total_shares=10.0, horizon=24, seed=0)
    assert env.start_idx == 0


def test_dataset_shorter_than_horizon_raises_clearly() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        CryptoExecutionEnv(market_data(n=10), total_shares=10.0, horizon=24, seed=0)
