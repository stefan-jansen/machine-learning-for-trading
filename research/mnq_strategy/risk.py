"""Risk management for the MNQ objective strategy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Transaction cost model for MNQ futures.

    Attributes:
        commission_per_contract: Commission per contract per side (USD).
        slippage_points: Slippage in points per side.
    """

    commission_per_contract: float = 1.50
    slippage_points: float = 0.50

    def __post_init__(self) -> None:
        if self.commission_per_contract < 0:
            raise ValueError("commission_per_contract must be nonnegative")
        if self.slippage_points < 0:
            raise ValueError("slippage_points must be nonnegative")


@dataclass(frozen=True)
class RiskDecision:
    """Position sizing decision with risk breakdown.

    Attributes:
        accepted: Whether the position is accepted.
        contracts: Number of contracts (0 if rejected).
        gross_risk: Stop loss risk in USD (stop_points * 2 * contracts).
        estimated_costs: Round-trip commission + slippage in USD.
        total_risk: Gross risk + estimated costs.
        reason: Explanation if rejected, empty if accepted.
    """

    accepted: bool
    contracts: int
    gross_risk: float
    estimated_costs: float
    total_risk: float
    reason: str


MNQ_POINT_VALUE = 2.0  # USD per point per contract
MAX_TOTAL_RISK = 250.0  # USD
MIN_CONTRACTS = 4
MAX_CONTRACTS = 10


def calculate_position_size(
    stop_points: float,
    requested_contracts: int,
    costs: CostModel,
) -> RiskDecision:
    """Calculate position size with risk and cost validation.

    Args:
        stop_points: Stop loss distance in points (must be > 0).
        requested_contracts: Requested number of contracts [4, 10].
        costs: Transaction cost model.

    Returns:
        RiskDecision with accepted flag, contract count, and risk breakdown.

    Raises:
        ValueError: If stop_points <= 0, costs negative, or requested_contracts not in [4, 10].
    """
    if stop_points <= 0:
        raise ValueError("stop_points must be > 0")
    if (
        not isinstance(requested_contracts, int)
        or requested_contracts < MIN_CONTRACTS
        or requested_contracts > MAX_CONTRACTS
    ):
        raise ValueError(
            f"requested_contracts must be an integer in [{MIN_CONTRACTS}, {MAX_CONTRACTS}]"
        )
    if costs.commission_per_contract < 0 or costs.slippage_points < 0:
        raise ValueError("costs must be nonnegative")

    # Gross risk = stop_points * point_value * contracts
    gross_risk = stop_points * MNQ_POINT_VALUE * requested_contracts

    # Round-trip costs:
    # commission_per_contract * contracts * 2 (entry + exit)
    # + slippage_points * 2 * contracts * point_value (entry + exit)
    commission_cost = costs.commission_per_contract * requested_contracts * 2
    slippage_cost = costs.slippage_points * 2 * requested_contracts * MNQ_POINT_VALUE
    estimated_costs = commission_cost + slippage_cost

    total_risk = gross_risk + estimated_costs

    if total_risk >= MAX_TOTAL_RISK:
        return RiskDecision(
            accepted=False,
            contracts=0,
            gross_risk=gross_risk,
            estimated_costs=estimated_costs,
            total_risk=total_risk,
            reason=f"total risk {total_risk:.2f} exceeds maximum {MAX_TOTAL_RISK}",
        )

    return RiskDecision(
        accepted=True,
        contracts=requested_contracts,
        gross_risk=gross_risk,
        estimated_costs=estimated_costs,
        total_risk=total_risk,
        reason="",
    )


class DailyRiskGuard:
    """Daily risk guard for account protection.

    Tracks realized daily PnL and consecutive losses.
    Stops new entries when daily loss >= 400 USD or consecutive losses >= 2.
    """

    def __init__(self, max_daily_loss: float = 400.0, max_consecutive_losses: int = 2) -> None:
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self._daily_pnl = 0.0
        self._consecutive_losses = 0

    def can_trade(self) -> bool:
        """Return True if new entries are allowed."""
        return (
            self._daily_pnl > -self.max_daily_loss
            and self._consecutive_losses < self.max_consecutive_losses
        )

    def record_trade(self, pnl: float) -> None:
        """Record a realized trade PnL.

        Args:
            pnl: Realized profit/loss in USD (negative for loss).
        """
        self._daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset_day(self) -> None:
        """Reset daily state for a new trading day."""
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
