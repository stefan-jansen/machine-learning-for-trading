# rl_environments.py - Shared RL environment classes for Chapter 21
"""
Shared Gymnasium environments for execution and hedging notebooks.

Provides:
- MarketState: Dataclass for market microstructure state
- ExecutionEnv: Optimal execution environment (Almgren-Chriss style)
- save_figure: Helper for saving Plotly figures to Ch21 figures directory

These are shared utilities for Chapter 21 RL notebooks. Import as:
    from rl_environments import ExecutionEnv, MarketState, save_figure
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
from gymnasium import spaces
from rl_calibration import ExecutionEnvParams


@dataclass
class MarketState:
    """Current market microstructure state."""

    mid_price: float
    spread: float
    depth: float  # Available liquidity
    volatility: float
    regime: int  # 0=normal, 1=stressed


def save_figure(
    fig: go.Figure,
    filename: str,
    width: int = 1200,
    height: int = 800,
    figures_dir: Path | None = None,
) -> None:
    fig.show()


class ExecutionEnv(gym.Env):
    """
    Optimal execution environment for liquidating a position.

    The agent must sell `total_shares` within `horizon` time steps,
    minimizing implementation shortfall while managing market impact.

    State: [inventory_ratio, time_ratio, spread, depth, volatility, regime]
    Action: Continuous [0, 1] -> pace multiplier around a reference schedule
    Reward: Negative implementation shortfall plus inventory-risk penalty

    Parameters
    ----------
    cal_params : ExecutionEnvParams, optional
        Calibrated parameters from real market data. If None, uses defaults.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        total_shares: int = 10_000,
        horizon: int = 60,
        initial_price: float = 100.0,
        cal_params: ExecutionEnvParams | None = None,
        risk_aversion: float = 0.0,
        schedule_penalty: float = 0.0,
        pace_min_multiplier: float = 0.5,
        pace_max_multiplier: float = 1.5,
        max_participation_rate: float = 0.35,
        seed: int | None = None,
    ):
        super().__init__()

        self.total_shares = total_shares
        self.horizon = horizon
        self.initial_price = initial_price
        self.risk_aversion = risk_aversion
        self.schedule_penalty = schedule_penalty
        self.pace_min_multiplier = pace_min_multiplier
        self.pace_max_multiplier = pace_max_multiplier
        self.max_participation_rate = max_participation_rate
        self.rng = np.random.default_rng(seed)

        # Use calibrated or default parameters
        if cal_params is not None:
            self.permanent_impact = cal_params.permanent_impact
            self.temporary_impact = cal_params.temporary_impact
            self.spread_normal = cal_params.spread_normal
            self.spread_stressed = cal_params.spread_stressed
            self.depth_normal = cal_params.depth_normal
            self.depth_stressed = cal_params.depth_stressed
            # GARCH parameters
            self.garch_alpha = cal_params.garch.alpha
            self.garch_beta = cal_params.garch.beta
            self.garch_omega = cal_params.garch.omega
            self.uncond_vol = cal_params.garch.unconditional_vol
            # Regime transition (use matrix diagonal for stay probabilities)
            self.p_stay_normal = cal_params.regimes.transition_matrix[0, 0]
            self.p_stay_stressed = cal_params.regimes.transition_matrix[1, 1]
        else:
            # Defaults (still realistic, just not calibrated)
            self.permanent_impact = 0.01
            self.temporary_impact = 0.001
            self.spread_normal = 0.005
            self.spread_stressed = 0.012
            self.depth_normal = 1000
            self.depth_stressed = 300
            self.garch_alpha = 0.1
            self.garch_beta = 0.85
            self.garch_omega = 0.00001
            self.uncond_vol = 0.02
            self.p_stay_normal = 0.98
            self.p_stay_stressed = 0.95

        # State: [inventory_ratio, time_ratio, spread, depth, vol, regime]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

        # Action: normalized pace multiplier around a reference schedule
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.reset()

    def _generate_market_path(self) -> list[MarketState]:
        """Generate market microstructure path with GARCH volatility and regime switching.

        Uses calibrated parameters for realistic simulation dynamics.
        """
        states = []
        price = self.initial_price

        # Initialize GARCH variance at unconditional level
        variance = self.uncond_vol**2
        regime = 0  # Start in normal regime

        for t in range(self.horizon):
            # Regime switching (calibrated transition probabilities)
            if regime == 0:
                regime = 0 if self.rng.random() < self.p_stay_normal else 1
            else:
                regime = 1 if self.rng.random() < self.p_stay_stressed else 0

            # GARCH(1,1) volatility update
            shock = self.rng.standard_normal()
            return_t = np.sqrt(variance) * shock
            variance = (
                self.garch_omega + self.garch_alpha * return_t**2 + self.garch_beta * variance
            )
            variance = max(variance, 1e-10)  # Floor for stability
            volatility = np.sqrt(variance)

            # Regime-dependent spreads and depth (calibrated)
            if regime == 0:  # Normal
                base_spread = self.spread_normal
                base_depth = self.depth_normal
            else:  # Stressed
                base_spread = self.spread_stressed
                base_depth = self.depth_stressed

            # Add noise around calibrated values
            spread = base_spread * (1 + 0.2 * self.rng.standard_normal())
            spread = max(spread, 0.0001)  # Floor
            depth = base_depth * np.exp(0.3 * self.rng.standard_normal())
            depth = max(depth, 100)  # Floor

            # The unaffected price process is a martingale: regimes change
            # liquidity and volatility, not expected return.
            price *= 1 + return_t
            price = max(price, 0.01)

            states.append(
                MarketState(
                    mid_price=price,
                    spread=spread,
                    depth=depth,
                    volatility=volatility,
                    regime=regime,
                )
            )

        return states

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.market_path = self._generate_market_path()
        self.step_idx = 0
        self.remaining_shares = self.total_shares
        self.arrival_price = self.market_path[0].mid_price
        self.total_cost = 0.0
        self.execution_history = []

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        market = self.market_path[self.step_idx]

        inventory_ratio = self.remaining_shares / self.total_shares
        time_ratio = (self.horizon - self.step_idx) / self.horizon

        return np.array(
            [
                inventory_ratio,
                time_ratio,
                market.spread * 100,  # Scale for learning
                market.depth / 1000,  # Normalize
                market.volatility * 100,
                float(market.regime),
            ],
            dtype=np.float32,
        )

    def _coerce_action_fraction(self, action: np.ndarray | float) -> float:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_action = float(action_array[0]) if action_array.size else 0.0
        return float(np.clip(raw_action, 0.0, 1.0))

    def _remaining_steps(self) -> int:
        return max(self.horizon - self.step_idx, 1)

    def reference_trade_size(self) -> float:
        if self.step_idx >= self.horizon - 1:
            return float(self.remaining_shares)
        return float(self.remaining_shares / self._remaining_steps())

    def max_trade_size(self, market: MarketState) -> float:
        if self.step_idx >= self.horizon - 1:
            return float(self.remaining_shares)
        schedule_cap = self.pace_max_multiplier * self.reference_trade_size()
        liquidity_cap = self.max_participation_rate * market.depth
        return float(min(self.remaining_shares, max(1.0, min(schedule_cap, liquidity_cap))))

    def action_to_target_shares(
        self, action: np.ndarray | float, market: MarketState | None = None
    ) -> int:
        current_market = self.market_path[self.step_idx] if market is None else market
        if self.step_idx >= self.horizon - 1:
            # The horizon step clears the book at any price: the order must complete.
            return int(self.remaining_shares)
        action_frac = self._coerce_action_fraction(action)
        multiplier = self.pace_min_multiplier + action_frac * (
            self.pace_max_multiplier - self.pace_min_multiplier
        )
        desired_shares = multiplier * self.reference_trade_size()
        capped_shares = min(desired_shares, self.max_trade_size(current_market))
        return int(min(self.remaining_shares, max(1.0, round(capped_shares))))

    def target_shares_to_action(self, target_shares: float) -> np.ndarray:
        if self.step_idx >= self.horizon - 1:
            return np.array([1.0], dtype=np.float32)
        reference = max(self.reference_trade_size(), 1e-8)
        multiplier = target_shares / reference
        normalized = (multiplier - self.pace_min_multiplier) / (
            self.pace_max_multiplier - self.pace_min_multiplier
        )
        return np.array([np.clip(normalized, 0.0, 1.0)], dtype=np.float32)

    def _trade_metrics(
        self, market: MarketState, shares_to_sell: int
    ) -> tuple[float, float, float]:
        participation_rate = shares_to_sell / max(market.depth, 1.0)
        temp_impact = self.temporary_impact * (participation_rate + participation_rate**2)
        perm_impact = self.permanent_impact * (shares_to_sell / max(self.total_shares, 1))
        execution_price = market.mid_price * (1 - market.spread / 2 - temp_impact)
        shortfall = (self.arrival_price - execution_price) * shares_to_sell
        return execution_price, shortfall, perm_impact

    def _inventory_risk_penalty(self, market: MarketState, remaining_shares: int) -> float:
        sigma_price = market.mid_price * market.volatility
        inventory_ratio = remaining_shares / max(self.total_shares, 1)
        return float(self.risk_aversion * sigma_price**2 * inventory_ratio**2 * self.total_shares)

    def _schedule_penalty(self, shares_to_sell: int, reference_shares: float) -> float:
        if self.schedule_penalty <= 0:
            return 0.0
        deviation_ratio = (shares_to_sell - reference_shares) / max(reference_shares, 1.0)
        notional = self.arrival_price * self.total_shares
        return float(self.schedule_penalty * notional * deviation_ratio**2)

    def step(self, action: np.ndarray | float):
        market = self.market_path[self.step_idx]
        reference_shares = self.reference_trade_size()

        # The action controls pace around a reference schedule rather than
        # allowing immediate liquidation of all remaining inventory.
        shares_to_sell = self.action_to_target_shares(action, market)

        execution_price, shortfall, perm_impact = self._trade_metrics(market, shares_to_sell)

        # Update state
        self.remaining_shares -= shares_to_sell
        self.total_cost += shortfall

        # Apply permanent impact to future prices
        for future_state in self.market_path[self.step_idx + 1 :]:
            future_state.mid_price *= 1 - perm_impact

        self.execution_history.append(
            {
                "step": self.step_idx,
                "shares_sold": shares_to_sell,
                "execution_price": execution_price,
                "shortfall": shortfall,
                "remaining": self.remaining_shares,
                "regime": market.regime,
                "depth": market.depth,
                "reference_shares": self.reference_trade_size(),
                "max_trade_shares": self.max_trade_size(market),
                "risk_penalty": 0.0,
            }
        )

        self.step_idx += 1

        # Terminal condition
        terminated = self.step_idx >= self.horizon or self.remaining_shares <= 0
        truncated = False

        risk_penalty = (
            0.0
            if self.remaining_shares <= 0
            else self._inventory_risk_penalty(market, self.remaining_shares)
        )
        schedule_penalty = self._schedule_penalty(shares_to_sell, reference_shares)
        self.execution_history[-1]["risk_penalty"] = risk_penalty
        self.execution_history[-1]["schedule_penalty"] = schedule_penalty

        reward = -(shortfall + risk_penalty + schedule_penalty) / max(self.total_shares, 1)

        assert not (terminated and self.remaining_shares > 0), (
            "the horizon step sells the whole remainder, so a terminated episode "
            "holds no inventory - a nonzero remainder means the pacing logic changed"
        )

        # Return terminal observation if episode is done
        if terminated:
            # Clamp step_idx for final observation
            self.step_idx = min(self.step_idx, self.horizon - 1)
        obs = self._get_obs()
        info = {
            "shares_sold": shares_to_sell,
            "remaining_shares": self.remaining_shares,
            "step_shortfall": shortfall,
            "total_shortfall": self.total_cost,
            "risk_penalty": risk_penalty,
            "schedule_penalty": schedule_penalty,
            "regime": market.regime,
        }

        return obs, reward, terminated, truncated, info
