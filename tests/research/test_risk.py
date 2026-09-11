"""Tests for MNQ risk management: position sizing, costs, and account guards."""

import math

import pytest

from research.mnq_strategy.risk import (
    CostModel,
    DailyRiskGuard,
    RiskDecision,
    calculate_position_size,
)


def test_rejects_when_minimum_four_contracts_exceed_250_dollars():
    decision = calculate_position_size(
        stop_points=40,
        requested_contracts=10,
        costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
    )
    assert decision.accepted is False
    assert "250" in decision.reason


def test_caps_contracts_between_four_and_ten():
    decision = calculate_position_size(
        stop_points=5,
        requested_contracts=10,
        costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
    )
    assert 4 <= decision.contracts <= 10


def test_daily_guard_stops_after_two_losses_or_400_dollars():
    guard = DailyRiskGuard()
    guard.record_trade(-200)
    guard.record_trade(-200)
    assert guard.can_trade() is False


def test_daily_guard_stops_after_400_dollar_loss():
    guard = DailyRiskGuard()
    guard.record_trade(-400)
    assert guard.can_trade() is False


def test_daily_loss_stop_latches_until_reset_even_after_a_win():
    guard = DailyRiskGuard()
    guard.record_trade(-400)
    assert guard.can_trade() is False

    guard.record_trade(500)
    assert guard.can_trade() is False

    guard.reset_day()
    assert guard.can_trade() is True


def test_daily_guard_resets_on_win_or_zero():
    guard = DailyRiskGuard()
    guard.record_trade(-200)
    guard.record_trade(100)
    assert guard.can_trade() is True


def test_non_loss_resets_consecutive_loss_streak():
    guard = DailyRiskGuard()
    guard.record_trade(-100)
    guard.record_trade(-100)
    assert guard.can_trade() is False

    guard.record_trade(0)
    assert guard.can_trade() is True


def test_daily_guard_reset_day_restores_state():
    guard = DailyRiskGuard()
    guard.record_trade(-200)
    guard.record_trade(-200)
    assert guard.can_trade() is False
    guard.reset_day()
    assert guard.can_trade() is True


def test_validate_stop_points_positive():
    with pytest.raises(ValueError, match="stop_points"):
        calculate_position_size(
            stop_points=0,
            requested_contracts=5,
            costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
        )


@pytest.mark.parametrize("stop_points", [math.nan, math.inf, -math.inf])
def test_validate_stop_points_finite(stop_points):
    with pytest.raises(ValueError, match="stop_points"):
        calculate_position_size(
            stop_points=stop_points,
            requested_contracts=5,
            costs=CostModel(),
        )


def test_validate_costs_nonnegative():
    with pytest.raises(ValueError, match="commission|cost"):
        calculate_position_size(
            stop_points=10,
            requested_contracts=5,
            costs=CostModel(commission_per_contract=-1.0, slippage_points=0.50),
        )

    with pytest.raises(ValueError, match="slippage|cost"):
        calculate_position_size(
            stop_points=10,
            requested_contracts=5,
            costs=CostModel(commission_per_contract=1.50, slippage_points=-0.50),
        )


@pytest.mark.parametrize("field", ["commission_per_contract", "slippage_points"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_cost_model_rejects_non_finite_values(field, value):
    values = {"commission_per_contract": 1.50, "slippage_points": 0.50}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        CostModel(**values)


def test_validate_requested_contracts_in_range():
    with pytest.raises(ValueError, match="contract"):
        calculate_position_size(
            stop_points=10,
            requested_contracts=3,
            costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
        )

    with pytest.raises(ValueError, match="contract"):
        calculate_position_size(
            stop_points=10,
            requested_contracts=11,
            costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
        )


def test_risk_decision_fields():
    decision = calculate_position_size(
        stop_points=10,
        requested_contracts=5,
        costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
    )
    assert hasattr(decision, "accepted")
    assert hasattr(decision, "contracts")
    assert hasattr(decision, "gross_risk")
    assert hasattr(decision, "estimated_costs")
    assert hasattr(decision, "total_risk")
    assert hasattr(decision, "reason")


def test_calculate_gross_risk():
    decision = calculate_position_size(
        stop_points=10,
        requested_contracts=5,
        costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
    )
    # MNQ: 2 USD/point/contract, gross_risk = stop_points * 2 * contracts
    expected_gross = 10 * 2 * 5  # 100
    assert decision.gross_risk == expected_gross


def test_calculate_estimated_costs():
    # commission_per_contract * contracts * 2 + slippage_points * 2 * contracts * 2
    decision = calculate_position_size(
        stop_points=10,
        requested_contracts=5,
        costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
    )
    expected_costs = 1.50 * 5 * 2 + 0.50 * 2 * 5 * 2  # 15 + 10 = 25
    assert decision.estimated_costs == expected_costs


def test_total_risk_is_gross_plus_costs():
    decision = calculate_position_size(
        stop_points=10,
        requested_contracts=5,
        costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
    )
    assert decision.total_risk == decision.gross_risk + decision.estimated_costs


def test_accepted_when_total_risk_under_250():
    decision = calculate_position_size(
        stop_points=5,
        requested_contracts=5,
        costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
    )
    # gross = 5*2*5=50, costs=1.50*5*2 + 0.50*2*5*2 = 15+10=25, total=75 < 250
    assert decision.accepted is True
    assert decision.contracts == 5


def test_downsizes_to_largest_affordable_contract_count():
    # Ten contracts total exactly 250 USD; nine contracts total 225 USD.
    decision = calculate_position_size(
        stop_points=10,
        requested_contracts=10,
        costs=CostModel(commission_per_contract=1.50, slippage_points=0.50),
    )
    assert decision.accepted is True
    assert decision.contracts == 9
    assert decision.gross_risk == 180
    assert decision.estimated_costs == 45
    assert decision.total_risk == 225


def test_rejects_when_minimum_four_contracts_equal_risk_ceiling():
    decision = calculate_position_size(
        stop_points=31.25,
        requested_contracts=10,
        costs=CostModel(commission_per_contract=0, slippage_points=0),
    )
    assert decision.accepted is False
    assert decision.contracts == 0
    assert decision.total_risk == 250
    assert "250" in decision.reason
    assert "exceeds" not in decision.reason.lower()


@pytest.mark.parametrize("field", ["max_daily_loss", "max_consecutive_losses"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_daily_guard_rejects_non_finite_thresholds(field, value):
    with pytest.raises(ValueError, match=field):
        DailyRiskGuard(**{field: value})


def test_default_cost_model_values():
    costs = CostModel()
    # Defaults should be recorded, not silently assumed
    assert hasattr(costs, "commission_per_contract")
    assert hasattr(costs, "slippage_points")
