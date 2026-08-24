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


@pytest.mark.parametrize(
    "shape,strategy_spec",
    [
        ("nested under strategy", {"strategy": {"risk": FLAT_CONTROL}}),
        ("at the top level", {"risk": FLAT_CONTROL}),
        ("nested, empty rule list", {"strategy": {"risk": {"name": "x", "position_rules": []}}}),
        ("top level, empty rule list", {"risk": {"name": "x", "position_rules": []}}),
    ],
)
def test_run_backtest_refuses_a_control_it_cannot_apply_in_any_shape(
    shape: str, strategy_spec: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever shape it arrives in, and before any other work, or the book runs unprotected.

    `run_backtest` takes the risk block nested under ``strategy`` or at the top level, which
    ``backtest_presets.py:368-369`` projects into ``strategy.risk`` during canonicalization. The
    top-level form is the one ``setup.yaml`` declares controls in, so a guard reading only the
    nested key is blind to its own case - measured, after the guard was moved to the front of
    ``run_backtest`` and narrowed to ``strategy_spec["strategy"]["risk"]`` in the same edit.

    Everything before the guard is arranged to fail on its own: no prices for
    ``ensure_backtest_spec``, no predictions for ``normalize_prediction_columns``, and
    ``backtest_run_status`` monkeypatched so reaching the cached-result branch raises. Only the
    guard running first, on the shape under test, produces the ValueError.
    """
    import case_studies.utils.registry as registry

    def _reached(*args, **kwargs):  # pragma: no cover - only runs if the guard misses
        raise AssertionError("reached the skip-if-complete branch")

    monkeypatch.setattr(registry, "backtest_run_status", _reached)

    try:
        run_backtest(
            "crypto_perps_funding",
            "0" * 12,
            strategy_spec,
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
            f"a risk block {shape} that installs no overlay was not refused: expected the "
            f"inert-spec ValueError, got {type(exc).__name__}: {exc}. The spec was canonicalized "
            "and would register under the control's name while running the unprotected book."
        ) from exc
    raise AssertionError(f"run_backtest accepted a risk block {shape} that installs no overlay")


def test_a_valid_control_is_not_refused_by_run_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not stand in front of a control the engine can actually apply."""
    import case_studies.utils.registry as registry

    monkeypatch.setattr(registry, "backtest_run_status", lambda *a, **k: None)
    spec = {
        "strategy": {
            "risk": {"name": "sl", "position_rules": [{"type": "stop_loss", "threshold": 0.03}]}
        }
    }
    with pytest.raises(BaseException) as caught:  # noqa: PT011 - it fails later, on missing data
        run_backtest(
            "crypto_perps_funding",
            "0" * 12,
            spec,
            prices=None,
            predictions=None,
            label="fwd_ret_8h",
            register=True,
        )
    assert "no overlay the engine can apply" not in str(caught.value)
