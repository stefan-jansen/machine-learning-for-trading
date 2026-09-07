"""A risk overlay that never fired is distinguishable from one never installed.

ml4t/agent-workspace#1051. The registry carried `num_trades` and the performance
metrics and nothing else, so a reader comparing an overlay against the strategy
it was laid on could conclude "the control acted" from a difference and nothing
at all from a match. `rules/notebook-standards.md` C17 records the failure that
hides behind that: 56 registered `crypto_perps_funding/16_risk_management`
results whose Sharpe, drawdown and trade count matched the unprotected book in
every digit, because the configuration declared its controls in a shape the
engine does not read and nothing was installed.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from case_studies.utils.backtest_runner import (
    _apply_vectorized_risk,
    _build_position_rules,
    _build_risk_manager,
)

STOP_LOSS = {"position_rules": [{"name": "stop_loss_5pct", "type": "stop_loss", "threshold": 0.05}]}
DRAWDOWN = {"portfolio_limits": [{"name": "dd_20", "type": "max_drawdown", "threshold": 0.20}]}


def _position_state(unrealized_return: float, high_water_mark: float | None = None):
    """A long position at a given unrealized return, as the broker would present it."""
    from ml4t.backtest.risk.types import PositionState

    entry = 100.0
    price = entry * (1.0 + unrealized_return)
    return PositionState(
        asset="A",
        side="long",
        entry_price=entry,
        current_price=price,
        quantity=10.0,
        initial_quantity=10.0,
        unrealized_pnl=(price - entry) * 10.0,
        unrealized_return=unrealized_return,
        bars_held=3,
        high_water_mark=entry if high_water_mark is None else high_water_mark,
        low_water_mark=price,
        bar_open=price,
        bar_high=price,
        bar_low=price,
        entry_time=datetime(2024, 1, 1),
        current_time=datetime(2024, 1, 5),
    )


def test_no_declared_control_registers_null_everywhere() -> None:
    """A NULL says "no overlay here", never "not measured"."""
    from case_studies.utils.backtest_runner import RiskTriggerLog

    metrics = RiskTriggerLog().as_metrics()
    assert metrics["risk_triggers"] is None
    assert metrics["risk_triggers_stop_loss"] is None
    assert metrics["risk_triggers_max_drawdown"] is None


def test_a_control_that_never_fired_registers_zero() -> None:
    from case_studies.utils.backtest_runner import RiskTriggerLog

    log = RiskTriggerLog()
    rule = _build_position_rules(STOP_LOSS, log)
    rule.evaluate(_position_state(+0.02))  # comfortably above the stop
    metrics = log.as_metrics()
    assert metrics["risk_triggers"] == 0.0
    assert metrics["risk_triggers_stop_loss"] == 0.0
    # The control that was not declared stays NULL alongside it.
    assert metrics["risk_triggers_trailing_stop"] is None


def test_a_control_that_fired_registers_the_count() -> None:
    from case_studies.utils.backtest_runner import RiskTriggerLog

    log = RiskTriggerLog()
    rule = _build_position_rules(STOP_LOSS, log)
    rule.evaluate(_position_state(+0.02))
    rule.evaluate(_position_state(-0.09))
    rule.evaluate(_position_state(-0.12))
    metrics = log.as_metrics()
    assert metrics["risk_triggers"] == 2.0
    assert metrics["risk_triggers_stop_loss"] == 2.0


def test_the_wrapped_rule_returns_what_the_real_rule_returned() -> None:
    """Counting must not change the exit the broker acts on."""
    from ml4t.backtest.risk import ActionType, StopLoss

    from case_studies.utils.backtest_runner import RiskTriggerLog

    log = RiskTriggerLog()
    counted = _build_position_rules(STOP_LOSS, log)
    bare = StopLoss(pct=0.05)
    for unrealized in (0.02, -0.09):
        state = _position_state(unrealized)
        expected, observed = bare.evaluate(state), counted.evaluate(state)
        assert observed.action == expected.action
        assert observed.fill_price == expected.fill_price
    assert counted.evaluate(_position_state(-0.09)).action == ActionType.EXIT_FULL


def test_a_portfolio_limit_counts_the_episode_not_the_bars() -> None:
    """RiskManager re-checks every bar, so a halted book breaches on each one."""
    from case_studies.utils.backtest_runner import RiskTriggerLog

    log = RiskTriggerLog()
    manager = _build_risk_manager(DRAWDOWN, 100_000.0, log)
    manager.update(equity=100_000.0, positions={}, timestamp=datetime(2024, 1, 1))
    for day in range(2, 8):
        manager.update(equity=50_000.0, positions={}, timestamp=datetime(2024, 1, day))
    assert log.as_metrics()["risk_triggers_max_drawdown"] == 1.0


def test_an_unknown_control_type_is_refused_rather_than_ignored() -> None:
    """C17's shape: a key the engine does not read, inside the identity hash."""
    with pytest.raises(ValueError, match="unknown position rule type"):
        _build_position_rules({"position_rules": [{"name": "tp", "type": "take_profit"}]})
    with pytest.raises(ValueError, match="unknown portfolio limit type"):
        _build_risk_manager({"portfolio_limits": [{"name": "var", "type": "var_95"}]}, 1.0)


def _port_ret(returns: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1 + i) for i in range(len(returns))],
            "net_ret": returns,
        }
    )


def test_the_vectorized_drawdown_breaker_reports_whether_it_fired() -> None:
    from case_studies.utils.backtest_runner import RiskTriggerLog

    fired = RiskTriggerLog()
    _apply_vectorized_risk(_port_ret([0.05, -0.30, 0.02, 0.02]), DRAWDOWN, fired)
    assert fired.as_metrics()["risk_triggers_max_drawdown"] == 1.0

    quiet = RiskTriggerLog()
    _apply_vectorized_risk(_port_ret([0.01, -0.02, 0.01, 0.01]), DRAWDOWN, quiet)
    assert quiet.as_metrics()["risk_triggers_max_drawdown"] == 0.0


def test_the_vectorized_path_refuses_a_position_rule_it_cannot_install() -> None:
    """Registering it would name a result for a stop that never ran."""
    from case_studies.utils.backtest_runner import RiskTriggerLog

    with pytest.raises(ValueError, match="cannot be applied on the vectorized path"):
        _apply_vectorized_risk(_port_ret([0.01, -0.02]), STOP_LOSS, RiskTriggerLog())


def test_a_trailing_stop_counts_the_retracement_and_not_the_hold() -> None:
    """The rule acts only when price retraces from its high water mark."""
    from ml4t.backtest.risk import ActionType

    from case_studies.utils.backtest_runner import RiskTriggerLog

    log = RiskTriggerLog()
    rule = _build_position_rules(
        {"position_rules": [{"name": "trail_5pct", "type": "trailing_stop", "threshold": 0.05}]},
        log,
    )
    # Rising into a new high water mark: nothing to exit.
    for gain in np.linspace(0.0, 0.03, 4):
        assert rule.evaluate(_position_state(float(gain))).action == ActionType.HOLD
    assert log.as_metrics()["risk_triggers_trailing_stop"] == 0.0

    # Back to the entry price from a high of 110, a 9% retrace on a 5% trail.
    assert rule.evaluate(_position_state(0.0, high_water_mark=110.0)).action == (
        ActionType.EXIT_FULL
    )
    assert log.as_metrics()["risk_triggers_trailing_stop"] == 1.0
