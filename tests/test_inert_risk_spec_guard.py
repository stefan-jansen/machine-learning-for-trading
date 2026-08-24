"""The inert-risk-spec guard has to run before the cached result is served.

`_build_position_rules` reads `risk_spec["position_rules"]` and installs nothing when the key
is absent, so a control passed in the flat shape `setup.yaml` declares runs the unprotected
book. The block still reaches `strategy.risk`, so the specification hashes differently and
registers as a distinct result carrying the control's name - which is how fifty-six
`crypto_perps_funding` overlay rows came to report the unprotected book under fourteen
different control names, matching their baseline in Sharpe, drawdown and trade count.

`_reject_inert_risk_spec` refuses that. Where it runs decides whether it works: `run_backtest`
returns a cached result before touching the engine when the artifacts already exist, so a guard
placed after that branch validates only the first run and lets every later read of the same
rows through. Those fifty-six rows are still on disk, so this is the route that matters.

Ordering is otherwise pinned by a comment, and moving the call back below the branch restores
the defect silently. These tests fail if it moves.
"""

from __future__ import annotations

import pytest

from case_studies.utils.backtest_runner import _reject_inert_risk_spec, run_backtest

FLAT_CONTROL = {"name": "stop_loss_3pct", "type": "stop_loss", "threshold": 0.03}


@pytest.mark.parametrize(
    "risk_spec",
    [
        FLAT_CONTROL,
        {"name": "time_exit_10", "type": "time_exit", "bars": 10},
        {"name": "empty_rules", "position_rules": []},
    ],
)
def test_a_risk_block_that_installs_nothing_is_refused(risk_spec: dict) -> None:
    with pytest.raises(ValueError, match="no overlay the engine can apply"):
        _reject_inert_risk_spec(risk_spec)


@pytest.mark.parametrize(
    "risk_spec",
    [
        {},
        {"name": "stop_loss_3pct", "position_rules": [{"type": "stop_loss", "threshold": 0.03}]},
        {"portfolio_limits": [{"type": "max_drawdown", "threshold": 0.2}]},
    ],
)
def test_a_risk_block_the_engine_can_apply_is_accepted(risk_spec: dict) -> None:
    _reject_inert_risk_spec(risk_spec)


def test_the_guard_runs_before_anything_else_in_run_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule-less block must be refused before any other work happens.

    Called with no prices and no predictions, so every step `run_backtest` takes before the
    guard fails on its own - `ensure_backtest_spec` reads prices, `normalize_prediction_columns`
    reads predictions, and the skip-if-complete branch is monkeypatched to raise. Only the guard
    running first produces the ValueError. Any other exception means it moved, and the assertion
    below names which one arrived instead so the failure is legible.
    """
    import case_studies.utils.registry as registry

    def _reached(*args, **kwargs):  # pragma: no cover - only runs if the guard moved
        raise AssertionError("reached the skip-if-complete branch")

    monkeypatch.setattr(registry, "backtest_run_status", _reached)

    try:
        run_backtest(
            "crypto_perps_funding",
            "0" * 12,
            {"strategy": {"risk": FLAT_CONTROL}},
            prices=None,
            predictions=None,
            label="fwd_ret_8h",
            register=True,
        )
    except ValueError as exc:
        assert "no overlay the engine can apply" in str(exc)
        return
    except BaseException as exc:  # noqa: BLE001 - the point is to report what arrived
        raise AssertionError(
            "run_backtest did work before validating the risk block: expected the inert-spec "
            f"ValueError, got {type(exc).__name__}: {exc}. _reject_inert_risk_spec must be the "
            "first thing run_backtest does, or a rule-less overlay is canonicalized, and - when "
            "its artifacts exist - served from cache without ever being checked."
        ) from exc
    raise AssertionError("run_backtest accepted a risk block that installs no overlay")
