"""Point-in-time crypto execution environment for Chapter 21."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces


@dataclass
class CryptoMarketState:
    """Current crypto market microstructure state."""

    timestamp: np.datetime64
    price: float
    volume: float
    avg_volume: float
    volatility: float
    premium_index: float
    hour: int
    hours_to_funding: int


class CryptoExecutionEnv(gym.Env):
    """
    Optimal execution environment for crypto perpetual futures.

    Uses real market data to simulate execution of a large order,
    modeling market impact based on actual volume patterns.

    State: [inventory_ratio, time_ratio, volatility, premium_index,
            volume_ratio, hour_of_day, hours_to_funding]
    Action: Box([0, 1]) -> pace multiplier around a reference schedule
    Reward: Negative cost-risk objective in basis points
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        market_data: pl.DataFrame,
        symbol: str = "BTCUSDT",
        total_shares: float = 100.0,
        horizon: int = 24,
        impact_coefficient: float = 0.001,
        risk_aversion: float = 0.0,
        schedule_penalty: float = 0.0,
        pace_min_multiplier: float = 0.5,
        pace_max_multiplier: float = 1.5,
        max_participation_rate: float = 0.10,
        seed: int | None = None,
    ):
        super().__init__()

        self.symbol = symbol
        self.total_shares = total_shares
        self.horizon = horizon
        self.impact_coefficient = impact_coefficient
        self.risk_aversion = risk_aversion
        self.schedule_penalty = schedule_penalty
        self.pace_min_multiplier = pace_min_multiplier
        self.pace_max_multiplier = pace_max_multiplier
        self.max_participation_rate = max_participation_rate

        # Extract data for this symbol
        symbol_data = market_data.filter(pl.col("symbol") == symbol).sort("timestamp")
        self.market_data = symbol_data.to_numpy()
        self.timestamps = symbol_data["timestamp"].to_numpy()
        self.prices = symbol_data["open"].to_numpy()
        self.volumes = symbol_data["observed_volume"].to_numpy()
        self.avg_volumes = symbol_data["avg_volume_24h"].to_numpy()
        self.volatilities = symbol_data["volatility_24h"].to_numpy()
        self.premium_indices = symbol_data["premium_index_close"].to_numpy()
        self.hours = symbol_data["hour"].to_numpy()
        self.hours_to_funding = symbol_data["hours_to_funding"].to_numpy()

        self.n_samples = len(self.prices)
        self.rng = np.random.default_rng(seed)

        # State: 7 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reset()

    def _get_market_state(self, idx: int) -> CryptoMarketState:
        """Get market state at given index."""
        return CryptoMarketState(
            timestamp=self.timestamps[idx],
            price=self.prices[idx],
            volume=self.volumes[idx],
            avg_volume=self.avg_volumes[idx],
            volatility=self.volatilities[idx],
            premium_index=self.premium_indices[idx],
            hour=self.hours[idx],
            hours_to_funding=self.hours_to_funding[idx],
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Start at a random point with enough data for the full horizon. The
        # upper bound is exclusive, so n_samples - horizon + 1 keeps the last
        # two otherwise-valid windows in play.
        if self.n_samples < self.horizon:
            raise ValueError(
                f"{self.symbol}: {self.n_samples} rows is fewer than the "
                f"{self.horizon}-step execution horizon"
            )
        self.start_idx = self.rng.integers(0, self.n_samples - self.horizon + 1)
        self.step_idx = 0

        self.remaining_shares = self.total_shares
        self.arrival_price = self.prices[self.start_idx]
        self.total_cost = 0.0
        self.execution_history = []

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        market = self._get_market_state(self.start_idx + self.step_idx)

        inventory_ratio = self.remaining_shares / self.total_shares
        time_ratio = (self.horizon - self.step_idx) / self.horizon

        # Volume ratio (current vs average)
        volume_ratio = market.volume / (market.avg_volume + 1e-8)

        # Normalize premium index (typically in range -0.01 to 0.01)
        premium_normalized = market.premium_index * 100

        # Volatility (typically 0.01-0.05)
        vol_normalized = market.volatility * 100 if not np.isnan(market.volatility) else 2.0

        return np.array(
            [
                inventory_ratio,
                time_ratio,
                vol_normalized,
                premium_normalized,
                min(volume_ratio, 5.0),  # Cap outliers
                market.hour / 24.0,  # Normalize hour
                market.hours_to_funding / 8.0,  # Normalize to funding window
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

    def max_trade_size(self, market: CryptoMarketState) -> float:
        """Largest executable size, capped by both schedule and liquidity.

        The cap binds on the final step too. Exempting it let the agent unwind
        any residual in one trade regardless of available volume, which both
        understated impact and made the forced-liquidation branch in ``step``
        unreachable -- so the notebook's forced-liquidation diagnostic could
        only ever report zero.
        """
        schedule_cap = self.pace_max_multiplier * self.reference_trade_size()
        liquidity_cap = self.max_participation_rate * market.volume
        return float(min(self.remaining_shares, max(1e-8, min(schedule_cap, liquidity_cap))))

    def action_to_target_shares(
        self, action: np.ndarray | float, market: CryptoMarketState | None = None
    ) -> float:
        current_market = (
            self._get_market_state(self.start_idx + self.step_idx) if market is None else market
        )
        action_frac = self._coerce_action_fraction(action)
        multiplier = self.pace_min_multiplier + action_frac * (
            self.pace_max_multiplier - self.pace_min_multiplier
        )
        desired_shares = multiplier * self.reference_trade_size()
        return float(
            min(self.remaining_shares, self.max_trade_size(current_market), desired_shares)
        )

    def target_shares_to_action(self, target_shares: float) -> np.ndarray:
        """Inverse of :meth:`action_to_target_shares`, kept consistent with it."""
        reference = max(self.reference_trade_size(), 1e-8)
        multiplier = target_shares / reference
        normalized = (multiplier - self.pace_min_multiplier) / (
            self.pace_max_multiplier - self.pace_min_multiplier
        )
        return np.array([np.clip(normalized, 0.0, 1.0)], dtype=np.float32)

    def _trade_metrics(
        self,
        market: CryptoMarketState,
        shares_to_sell: float,
        concurrent_shares: float = 0.0,
    ) -> tuple[float, float]:
        """Execution price and shortfall for one trade.

        ``concurrent_shares`` are shares executed against the same bar by a
        second trade. Impact is charged on the *combined* participation, so
        splitting an order across two trades at one timestamp costs exactly
        what executing it in one trade costs. Without this, concave
        square-root impact would make splitting artificially cheap.
        """
        participation_rate = (shares_to_sell + concurrent_shares) / (market.volume + 1e-8)
        # Square-root-plus-linear, matching the model the notebook documents.
        # The linear term was written as participation_rate**2, which made
        # impact grow quadratically and overstated forced-liquidation cost.
        market_impact = self.impact_coefficient * (
            np.sqrt(max(participation_rate, 0.0)) + participation_rate
        )
        execution_price = market.price * (1 - market_impact)
        shortfall = (self.arrival_price - execution_price) * shares_to_sell
        return execution_price, shortfall

    def _inventory_risk_penalty(self, market: CryptoMarketState, remaining_shares: float) -> float:
        sigma_price = market.price * (market.volatility if not np.isnan(market.volatility) else 0.0)
        inventory_ratio = remaining_shares / max(self.total_shares, 1e-8)
        return float(self.risk_aversion * sigma_price**2 * inventory_ratio**2 * self.total_shares)

    def _schedule_penalty(self, shares_to_sell: float, reference_shares: float) -> float:
        if self.schedule_penalty <= 0:
            return 0.0
        deviation_ratio = (shares_to_sell - reference_shares) / max(reference_shares, 1e-8)
        notional = self.arrival_price * self.total_shares
        return float(self.schedule_penalty * notional * deviation_ratio**2)

    def step(self, action: np.ndarray | float):
        market = self._get_market_state(self.start_idx + self.step_idx)
        # Capture both schedule quantities before inventory is reduced, so the
        # history row describes the trade that was actually executed. Computing
        # them after the fact could even report a cap below `shares_sold`.
        reference_shares = self.reference_trade_size()
        max_trade_shares = self.max_trade_size(market)
        shares_to_sell = self.action_to_target_shares(action, market)

        # Any residual left after the final step is liquidated against this same
        # bar, so it is known before the trade is priced and both legs are
        # charged on their combined participation.
        residual_tolerance = 1e-9 * max(self.total_shares, 1.0)
        forced_shares = (
            max(self.remaining_shares - shares_to_sell, 0.0)
            if self.step_idx >= self.horizon - 1
            else 0.0
        )
        if forced_shares <= residual_tolerance:
            forced_shares = 0.0

        execution_price, shortfall = self._trade_metrics(market, shares_to_sell, forced_shares)
        # Both legs are charged on the same combined participation, so they clear
        # at one price: the bar has a single execution price regardless of how
        # the order is split between the policy and the forced remainder.
        if forced_shares > 0:
            _, forced_shortfall = self._trade_metrics(market, forced_shares, shares_to_sell)
        else:
            forced_shortfall = 0.0

        # Update state
        self.remaining_shares -= shares_to_sell
        if self.remaining_shares <= residual_tolerance:
            self.remaining_shares = 0.0
        self.total_cost += shortfall + forced_shortfall

        # Inventory risk prices the exposure actually carried past this bar.
        # A forced remainder is liquidated against this same bar, so it is never
        # carried and must not be charged: otherwise the terminal reward depends
        # on how the final order splits between the policy leg and the forced
        # leg even though exposure and execution cost are identical either way.
        carried_shares = max(self.remaining_shares - forced_shares, 0.0)
        risk_penalty = (
            0.0 if carried_shares <= 0 else self._inventory_risk_penalty(market, carried_shares)
        )
        schedule_penalty = self._schedule_penalty(shares_to_sell, reference_shares)

        if forced_shares > 0:
            self.remaining_shares = 0.0

        # One history row per bar. The forced leg used to be a second row on the
        # final bar, which every consumer then had to remember to collapse -- the
        # trajectory figure plotted it as a fictitious extra hour and the
        # last-bar volume column read the involuntary remainder alone. Recording
        # the bar once, with the forced quantity as its own field, removes the
        # trap instead of patching each reader.
        self.execution_history.append(
            {
                "step": self.step_idx,
                "timestamp": str(market.timestamp),
                "price": market.price,
                "shares_sold": shares_to_sell + forced_shares,
                "forced_shares": forced_shares,
                "forced_liquidation": forced_shares > 0,
                "execution_price": execution_price,
                "shortfall": shortfall + forced_shortfall,
                "remaining": self.remaining_shares,
                "volume": market.volume,
                "premium_index": market.premium_index,
                "reference_shares": reference_shares,
                "max_trade_shares": max_trade_shares,
                "risk_penalty": risk_penalty,
                "schedule_penalty": schedule_penalty,
                "hour": market.hour,
                "hours_to_funding": market.hours_to_funding,
            }
        )

        self.step_idx += 1

        reward = (
            -(shortfall + forced_shortfall + risk_penalty + schedule_penalty)
            / (self.arrival_price * self.total_shares)
            * 10_000
        )

        # Terminal condition
        terminated = self.step_idx >= self.horizon or self.remaining_shares <= 0
        truncated = False

        # Return final observation
        if terminated:
            self.step_idx = min(self.step_idx, self.horizon - 1)

        obs = self._get_obs()
        info = {
            "shares_sold": shares_to_sell + forced_shares,
            "forced_shares": forced_shares,
            "remaining_shares": self.remaining_shares,
            "step_shortfall": shortfall + forced_shortfall,
            "total_shortfall": self.total_cost,
            "risk_penalty": risk_penalty,
            "schedule_penalty": schedule_penalty,
            "premium_index": market.premium_index,
            "volume": market.volume,
        }

        return obs, reward, terminated, truncated, info
