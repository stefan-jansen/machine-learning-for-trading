"""Inventory-aware market-making environment for Chapter 21."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class MarketMakingDynamics:
    """Calibrated dynamics and discrete quote-action grid."""

    garch_omega: float
    garch_alpha: float
    garch_beta: float
    unconditional_vol: float
    skew_levels: tuple[float, ...]
    spread_multipliers: tuple[float, ...]


def generate_garch_market_data(
    n_steps: int, rng: np.random.Generator, dynamics: MarketMakingDynamics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic market data with calibrated GARCH(1,1) volatility."""
    prices = [100.0]
    volatilities = []
    imbalances = []
    variance = max(dynamics.unconditional_vol**2, 1e-8)
    imbalance = 0

    for _ in range(n_steps):
        # Record the conditional volatility of the return about to be drawn, so
        # that volatilities[t] pairs with the move from prices[t] to
        # prices[t+1]. Appending the post-update variance instead paired
        # prices[t] with a variance computed from return_t itself, letting the
        # agent see the size of the move it was quoting into.
        volatilities.append(np.sqrt(variance))

        shock = np.clip(rng.standard_normal(), -5, 5)
        return_t = np.clip(np.sqrt(variance) * shock, -0.1, 0.1)
        variance = np.clip(
            dynamics.garch_omega
            + dynamics.garch_alpha * return_t**2
            + dynamics.garch_beta * variance,
            1e-10,
            0.01,
        )

        imbalance = np.clip(0.9 * imbalance + 0.1 * rng.uniform(-1, 1), -1, 1)
        new_price = np.clip(
            prices[-1] * (1 + 0.0001 * imbalance + return_t), prices[-1] * 0.5, prices[-1] * 2.0
        )

        prices.append(new_price)
        imbalances.append(imbalance)

    prices = np.array(prices, dtype=np.float32)
    if not np.all(np.isfinite(prices)):
        raise ValueError("Generated prices contain NaN or Inf")
    return prices, np.array(volatilities, dtype=np.float32), np.array(imbalances, dtype=np.float32)


def fill_probability(
    distance: float,
    imbalance_factor: float,
    base_spread: float,
    arrival_rate: float = 0.6,
    sensitivity: float = 4.0,
) -> float:
    """Probability of a limit order being filled given its distance from mid."""
    scaled = max(distance / max(base_spread, 1e-6), 0.0)
    intensity = max(float(arrival_rate * np.exp(-sensitivity * scaled) * imbalance_factor), 0.0)
    return float(1.0 - np.exp(-intensity))


def decode_action(action: int, dynamics: MarketMakingDynamics) -> tuple[float, float]:
    """Map a discrete action index to (skew level, spread multiplier)."""
    skew_idx, spread_idx = divmod(int(action), len(dynamics.spread_multipliers))
    return float(dynamics.skew_levels[skew_idx]), float(dynamics.spread_multipliers[spread_idx])


def compute_quotes(price, vol, skew_level, spread_mult, inventory, inventory_limit, base_spread):
    """Reservation-price quoting with inventory skew; returns quote geometry."""
    inv_norm = inventory / max(inventory_limit, 1)
    reservation_price = price + (-inv_norm * vol * price)
    quote_center = reservation_price + skew_level * 0.25 * vol * price
    half_spread = 0.5 * base_spread * price * (1 + 5.0 * vol) * spread_mult
    bid_quote = max(quote_center - half_spread, 0.01)
    ask_quote = max(quote_center + half_spread, bid_quote + 0.01)
    return reservation_price, quote_center, bid_quote, ask_quote, half_spread


def build_mm_obs(
    prices, vols, imbalances, idx, inventory, inventory_limit, episode_length, half_spread
):
    """Build the 6D market-making observation, scaled and clipped."""
    if idx > 0:
        price_change = np.clip((prices[idx] - prices[idx - 1]) / prices[idx - 1], -0.1, 0.1)
    else:
        price_change = 0.0
    vol = np.clip(vols[min(idx, len(vols) - 1)], 0, 0.1)
    imbalance = imbalances[min(idx, len(imbalances) - 1)]
    time_ratio = (episode_length - idx) / episode_length
    spread_bps = 2 * half_spread / max(prices[idx], 1e-6) * 10_000
    return np.array(
        [
            np.clip(inventory / inventory_limit, -1.0, 1.0),
            np.clip(price_change * 10, -1.0, 1.0),
            np.clip(vol * 100, 0, 10.0),
            np.clip(imbalance, -1.0, 1.0),
            np.clip(time_ratio, 0.0, 1.0),
            np.clip(spread_bps / 10.0, 0.0, 10.0),
        ],
        dtype=np.float32,
    )


def simulate_fills(
    rng, inventory, inventory_limit, bid_quote, ask_quote, price, imbalance, base_spread
):
    """Draw bid/ask fills from the distance-based fill-probability model."""
    bid_distance = max((price - bid_quote) / max(price, 1e-6), 0.0)
    ask_distance = max((ask_quote - price) / max(price, 1e-6), 0.0)
    bid_prob = fill_probability(
        bid_distance, np.clip(1.0 - 0.35 * imbalance, 0.2, 2.0), base_spread
    )
    ask_prob = fill_probability(
        ask_distance, np.clip(1.0 + 0.35 * imbalance, 0.2, 2.0), base_spread
    )
    bid_filled = inventory < inventory_limit and rng.random() < bid_prob
    ask_filled = inventory > -inventory_limit and rng.random() < ask_prob
    return bid_filled, ask_filled


def terminal_liquidation(cash, inventory, next_price, base_spread):
    """Liquidate residual inventory at a half-spread cost; returns wealth + cost."""
    liquidation_cost = abs(inventory) * next_price * base_spread / 2
    liquidated_wealth = cash + inventory * next_price - liquidation_cost
    return liquidated_wealth, liquidation_cost


class MarketMakingEnv(gym.Env):
    """Inventory-aware market making environment (Discrete 3 skew x 3 spread)."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        episode_length=500,
        inventory_limit=100,
        lambda_inventory=0.001,
        base_spread=0.001,
        dynamics: MarketMakingDynamics | None = None,
        seed=None,
    ):
        super().__init__()
        self.episode_length = episode_length
        self.inventory_limit = inventory_limit
        self.lambda_inventory = lambda_inventory
        self.base_spread = base_spread
        if dynamics is None:
            raise ValueError("dynamics must provide calibrated GARCH and action-grid parameters")
        self.dynamics = dynamics
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(
            len(dynamics.skew_levels) * len(dynamics.spread_multipliers)
        )
        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.prices, self.volatilities, self.imbalances = generate_garch_market_data(
            self.episode_length, self.rng, self.dynamics
        )
        self.step_idx = 0
        self.inventory = 0
        self.cash = 0.0
        self.wealth = 0.0
        self.n_trades = 0
        self.terminal_inventory = 0
        self.current_half_spread = self.base_spread * self.prices[0] / 2
        self.current_quote_offset = 0.0
        self.history = []
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return build_mm_obs(
            self.prices,
            self.volatilities,
            self.imbalances,
            self.step_idx,
            self.inventory,
            self.inventory_limit,
            self.episode_length,
            self.current_half_spread,
        )

    def step(self, action: int):
        price = self.prices[self.step_idx]
        vol = self.volatilities[self.step_idx]
        imbalance = self.imbalances[self.step_idx]
        next_price = self.prices[min(self.step_idx + 1, self.episode_length)]
        skew_level, spread_mult = decode_action(action, self.dynamics)
        reservation_price, quote_center, bid_quote, ask_quote, half_spread = compute_quotes(
            price,
            vol,
            skew_level,
            spread_mult,
            self.inventory,
            self.inventory_limit,
            self.base_spread,
        )
        self.current_half_spread = half_spread
        self.current_quote_offset = quote_center - price

        wealth_before = self.cash + self.inventory * price
        # The quotes above were computed from the inventory held *before* this
        # bar's fills, so that is the position they respond to. The row's
        # `inventory` is the post-fill position -- the realized path, one fill
        # later -- which is a different series and the wrong x-axis for the
        # quote-skew figure.
        quote_inventory = self.inventory
        bid_filled, ask_filled = simulate_fills(
            self.rng,
            self.inventory,
            self.inventory_limit,
            bid_quote,
            ask_quote,
            price,
            imbalance,
            self.base_spread,
        )
        if bid_filled:
            self.inventory += 1
            self.cash -= bid_quote
            self.n_trades += 1
        if ask_filled:
            self.inventory -= 1
            self.cash += ask_quote
            self.n_trades += 1

        marked_wealth = self.cash + self.inventory * next_price
        inventory_penalty = (
            self.lambda_inventory
            * (self.inventory / max(self.inventory_limit, 1)) ** 2
            * next_price
        )
        reward = np.clip(marked_wealth - wealth_before - inventory_penalty, -100.0, 100.0)
        self.wealth = marked_wealth

        self.history.append(
            {
                "step": self.step_idx,
                "inventory": self.inventory,
                "quote_inventory": quote_inventory,
                "wealth": self.wealth,
                "reward": reward,
                "trades": self.n_trades,
                "mid_price": price,
                "reservation_price": reservation_price,
                "quote_center": quote_center,
                "bid_quote": bid_quote,
                "ask_quote": ask_quote,
                "spread_bps": 2 * half_spread / max(price, 1e-6) * 10_000,
                "quote_offset_bps": (quote_center - price) / max(price, 1e-6) * 10_000,
                "bid_filled": bid_filled,
                "ask_filled": ask_filled,
            }
        )

        self.step_idx += 1
        terminated = self.step_idx >= self.episode_length
        if terminated:
            remaining_inventory = self.inventory
            liquidated_wealth, liquidation_cost = terminal_liquidation(
                self.cash, remaining_inventory, next_price, self.base_spread
            )
            reward += liquidated_wealth - self.wealth
            self.terminal_inventory = remaining_inventory
            self.cash = liquidated_wealth
            self.inventory = 0
            self.wealth = liquidated_wealth
            self.history[-1]["wealth"] = self.wealth
            # The row's reward has to be the reward the agent was actually
            # given, liquidation included. Leaving the pre-liquidation value
            # here while updating `wealth` made the final row disagree with
            # itself and with the returned transition.
            self.history[-1]["reward"] = reward
            # `inventory` stays the post-fill position, which on this row is the
            # position carried into liquidation; post-liquidation inventory is
            # zero by construction and would erase that. `quote_inventory` is
            # what pairs with `quote_offset_bps`, and it is untouched here.
            self.history[-1]["terminal_inventory"] = remaining_inventory
            self.history[-1]["liquidation_cost"] = liquidation_cost

        info = {
            "wealth": self.wealth,
            "inventory": self.inventory,
            "terminal_inventory": self.terminal_inventory,
            "n_trades": self.n_trades,
        }
        return self._obs(), reward, terminated, False, info
