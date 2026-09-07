"""An account that loses its capital stops there (ml4t/agent-workspace#920).

Measured on the `us_firm_characteristics` registry, 2026-09-07: 46 registered
backtests hold a period return below -100%, so their equity compounds through
zero and `(1 + r)` inverts the sign of every later period. The worst of them,
`e7708f4f376a`, reports sharpe 1.547 and cagr 0.687 against a total return of
-202.3. The series below is that run's shape.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from case_studies.utils.backtest_runner import compute_portfolio_metrics

# 109 monthly periods, one of them -103.42%, which is what e7708f4f376a held.
_RUINED = np.array([0.03] * 40 + [-1.0342] + [0.03] * 68)
_SOLVENT = np.array([0.03] * 40 + [-0.30] + [0.03] * 68)


def _metrics(returns: np.ndarray) -> dict:
    return compute_portfolio_metrics(returns, periods_per_year=12, uncertainty=False)


def test_a_bankrupt_path_reports_no_sharpe() -> None:
    """The defect: a path whose equity crossed zero was still ranked."""
    out = _metrics(_RUINED)
    assert out["ruin"] == 1.0
    assert out["sharpe"] is None
    assert out["sortino"] is None
    assert out["calmar"] is None


def test_a_bankrupt_path_reports_the_loss_it_actually_took() -> None:
    """Total loss of capital, not the sign-flipped product of the raw series."""
    out = _metrics(_RUINED)
    assert out["total_return"] == -1.0
    assert out["max_drawdown"] == -1.0
    assert out["cagr"] == -1.0
    assert out["ruin_period"] == 40.0


def test_a_solvent_path_is_untouched() -> None:
    """The control: the same shape with a survivable loss keeps every metric."""
    out = _metrics(_SOLVENT)
    assert out["ruin"] == 0.0
    assert out["ruin_period"] is None
    assert isinstance(out["sharpe"], float)
    assert out["total_return"] > 0.0


def test_the_stop_floors_equity_at_zero_and_holds_it_there() -> None:
    from case_studies.utils.backtest_runner import apply_ruin_stop

    stopped, index = apply_ruin_stop(_RUINED)
    assert index == 40
    assert stopped[39] == 0.03  # periods before ruin are untouched
    assert stopped[40] == -1.0  # the engine models no creditor
    assert (stopped[41:] == 0.0).all()  # no capital, no positions, no returns
    assert float(np.cumprod(1.0 + stopped)[-1]) == 0.0


def test_the_stop_is_idempotent() -> None:
    from case_studies.utils.backtest_runner import apply_ruin_stop

    once, first = apply_ruin_stop(_RUINED)
    twice, second = apply_ruin_stop(once)
    assert first == second == 40
    assert (once == twice).all()


def test_a_solvent_series_passes_through_the_stop_unchanged() -> None:
    from case_studies.utils.backtest_runner import apply_ruin_stop

    stopped, index = apply_ruin_stop(_SOLVENT)
    assert index is None
    assert (stopped == _SOLVENT).all()


def test_the_registered_return_path_stops_where_the_account_does() -> None:
    from case_studies.utils.backtest_runner import stop_returns_at_ruin

    frame = pl.DataFrame(
        {
            "timestamp": pl.date_range(
                pl.date(2009, 1, 1), pl.date(2009, 1, 1) + pl.duration(days=108), eager=True
            ),
            "daily_return": _RUINED,
        }
    )
    stopped, index = stop_returns_at_ruin(frame)
    assert index == 40
    assert stopped.height == frame.height
    assert stopped["timestamp"].to_list() == frame["timestamp"].to_list()
    assert stopped["daily_return"].to_list()[40:] == [-1.0] + [0.0] * 68


def test_a_loss_that_the_book_survives_is_not_ruin() -> None:
    """A period return below -100% is not by itself ruin: prior gains absorb it."""
    from case_studies.utils.backtest_runner import apply_ruin_stop

    # Equity reaches 4.0 before a -120% period, which leaves it at -0.8: still ruin.
    through_zero, index = apply_ruin_stop(np.array([1.0, 1.0, -1.2, 0.1]))
    assert index == 2
    # A -60% period from equity 4.0 leaves 1.6, which is a drawdown and not ruin.
    survived, no_index = apply_ruin_stop(np.array([1.0, 1.0, -0.6, 0.1]))
    assert no_index is None
    assert (survived == np.array([1.0, 1.0, -0.6, 0.1])).all()
