"""An account that loses its capital stops there (ml4t/agent-workspace#920).

Measured on the `us_firm_characteristics` registry, 2026-09-07: 46 registered
backtests hold a period return below -100%, so their equity compounds through
zero and `(1 + r)` inverts the sign of every later period. The worst of them,
`e7708f4f376a`, reports sharpe 1.547 and cagr 0.687 against a total return of
-202.3. The series below is that run's shape.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

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
    assert math.isnan(out["sharpe"])
    assert math.isnan(out["sortino"])
    assert math.isnan(out["calmar"])


def test_the_unrankable_value_survives_the_round_trip_a_reader_makes() -> None:
    """NaN, so that SQLite stores NULL and a notebook's `f"{x:.3f}"` still prints.

    A None would reach `12_portfolio_management`'s progress line as a TypeError,
    which its `except Exception` counts as a failed execution - a run that
    registered successfully reported as a failure.
    """
    sharpe = _metrics(_RUINED)["sharpe"]
    assert f"{sharpe:.3f}" == "nan"
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE m (sharpe REAL)")
    db.execute("INSERT INTO m VALUES (?)", (sharpe,))
    db.execute("INSERT INTO m VALUES (2.0)")
    assert db.execute("SELECT COUNT(*) FROM m WHERE sharpe IS NOT NULL").fetchone()[0] == 1
    assert db.execute("SELECT sharpe FROM m ORDER BY sharpe DESC").fetchall() == [(2.0,), (None,)]


def test_a_path_that_ruins_in_its_first_period_still_reports_the_loss() -> None:
    """The diagnostic library divides by a running maximum of zero and returns NaN.

    `_safe` turned that into a drawdown of 0.0 beside `ruin=1`, which reads as a
    bankruptcy that cost nothing.
    """
    out = compute_portfolio_metrics(np.array([-1.2, 0.1]), periods_per_year=12, uncertainty=False)
    assert out["ruin"] == 1.0
    assert out["max_drawdown"] == -1.0
    assert out["total_return"] == -1.0
    assert out["cagr"] == -1.0


def test_a_single_period_wipeout_is_not_a_flat_series() -> None:
    """The short branch reported zeros for every metric, ruin included."""
    out = compute_portfolio_metrics(np.array([-1.2]), periods_per_year=12, uncertainty=False)
    assert out["ruin"] == 1.0
    assert out["total_return"] == -1.0
    assert out["max_drawdown"] == -1.0
    assert math.isnan(out["sharpe"])


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
    assert not math.isnan(out["sharpe"])
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


def test_a_common_support_ranking_puts_a_bankrupt_candidate_last() -> None:
    """A null Sharpe must not sort to the top of a descending polars sort.

    SQL puts NULLs last on `ORDER BY ... DESC`; polars puts them first unless
    told otherwise, so the ranking that reads these metrics has to say so.
    """
    from case_studies.utils.strategy_analysis import rank_returns_on_common_support

    timestamps = pl.datetime_range(
        pl.datetime(2009, 1, 1), pl.datetime(2009, 1, 1) + pl.duration(days=108), "1d", eager=True
    )
    ranking = rank_returns_on_common_support(
        {
            "ruined": pl.DataFrame({"timestamp": timestamps, "daily_return": _RUINED}),
            "solvent": pl.DataFrame({"timestamp": timestamps, "daily_return": _SOLVENT}),
        },
        periods_per_year=12,
    )
    assert ranking["backtest_hash"].to_list() == ["solvent", "ruined"]
    assert ranking["sharpe"][1] is None


def test_a_comparison_window_that_misses_the_ruin_does_not_make_it_rankable() -> None:
    """The intersection can end before the period that wiped the account out.

    A bankrupt book restricted to the months before it went bankrupt is not a
    rankable strategy, and it would otherwise outrank a solvent candidate with a
    negative Sharpe on the strength of the window the comparison happened to pick.
    """
    from case_studies.utils.strategy_analysis import rank_returns_on_common_support

    early = pl.datetime_range(
        pl.datetime(2009, 1, 1), pl.datetime(2009, 1, 1) + pl.duration(days=29), "1d", eager=True
    )
    full = pl.datetime_range(
        pl.datetime(2009, 1, 1), pl.datetime(2009, 1, 1) + pl.duration(days=108), "1d", eager=True
    )
    ranking = rank_returns_on_common_support(
        {
            # Ruins at period 40, well past the 30-day intersection below.
            "ruined": pl.DataFrame({"timestamp": full, "daily_return": _RUINED}),
            "losing": pl.DataFrame({"timestamp": early, "daily_return": [-0.01] * 30}),
        },
        periods_per_year=12,
    )
    assert ranking["backtest_hash"].to_list() == ["losing", "ruined"]
    assert ranking["sharpe"][1] is None


def test_the_plumbing_test_refuses_a_bankrupt_random_run(monkeypatch) -> None:
    """No Sharpe to compare against the tolerance, so it says why."""
    from types import SimpleNamespace

    import case_studies.utils.backtest_runner as br

    spec = {
        "version": 2,
        "strategy": {"rebalance": {"mode": "vectorized"}},
        "backtest_config": {},
    }
    predictions = pl.DataFrame(
        {
            "timestamp": [pl.datetime(2024, 1, 1)] * 2,
            "symbol": ["A", "B"],
            "y_score": [0.8, 0.2],
            "y_true": [0.1, -0.1],
        }
    )
    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kwargs: args[2])
    monkeypatch.setattr(
        br,
        "run_backtest",
        lambda *a, **k: SimpleNamespace(
            metrics={"sharpe": None, "ruin": 1.0, "ruin_period": 40.0, "n_periods": 109}
        ),
    )

    with pytest.raises(ValueError, match="went bankrupt at period 40.0 of 109"):
        br.run_plumbing_test(
            "demo", pl.DataFrame(), spec, predictions=predictions, label="fwd_ret_1m", seed=7
        )


def test_a_bankrupt_run_served_from_the_cache_still_formats(monkeypatch, tmp_path) -> None:
    """The skip-if-complete branch reads metrics back out of SQLite, where NaN is NULL.

    So a cached bankrupt run handed `None` to the same `f"{sharpe:.3f}"` lines the
    NaN exists to protect, and failed where a freshly computed one printed.
    """
    import case_studies.utils.backtest_runner as br
    import case_studies.utils.conformal as conformal
    import case_studies.utils.registry.store as store

    case_dir = tmp_path / "cs"
    run_log = case_dir / "run_log"
    backtest_dir = run_log / "backtest" / "cachedhash"
    backtest_dir.mkdir(parents=True)
    pl.DataFrame({"timestamp": [datetime(2009, 1, 1)], "daily_return": [0.0]}).write_parquet(
        backtest_dir / "daily_returns.parquet"
    )
    with sqlite3.connect(run_log / "registry.db") as db:
        db.execute(
            "CREATE TABLE backtest_metrics (backtest_hash TEXT PRIMARY KEY, computed_at TEXT, "
            "sharpe REAL, total_return REAL, ruin REAL)"
        )
        # What the engine wrote: NaN in, NULL out.
        db.execute(
            "INSERT INTO backtest_metrics VALUES ('cachedhash', 'now', ?, -1.0, 1.0)",
            (float("nan"),),
        )
    assert (
        sqlite3.connect(run_log / "registry.db")
        .execute("SELECT sharpe FROM backtest_metrics")
        .fetchone()[0]
        is None
    )

    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kw: args[2])
    monkeypatch.setattr(conformal, "ensure_conformal_calibration_identity", lambda spec: spec)
    monkeypatch.setattr(br, "substitute_continuous_return_for_classification", lambda p, *_: p)
    monkeypatch.setattr(store, "_case_dir", lambda _cs: case_dir)
    monkeypatch.setattr(
        br,
        "_refuse_an_allocation_that_produced_no_target",
        lambda *a, **k: None,
        raising=False,
    )

    from case_studies.utils.registry import completeness

    monkeypatch.setattr(
        completeness,
        "backtest_run_status",
        lambda *a, **k: SimpleNamespace(
            complete=True, backtest_hash="cachedhash", summary=lambda: "cached"
        ),
    )
    import case_studies.utils.registry as registry

    monkeypatch.setattr(
        registry,
        "backtest_run_status",
        lambda *a, **k: SimpleNamespace(
            complete=True, backtest_hash="cachedhash", summary=lambda: "cached"
        ),
    )
    monkeypatch.setattr(registry, "backtest_dir", lambda _cs, h: backtest_dir)

    spec = {
        "version": 2,
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 1, "long_short": False},
            "rebalance": {"mode": "vectorized", "cadence": "daily", "step": 1},
        },
        "backtest_config": {"cash": {"initial": 1.0}, "account": {}},
    }
    result = br.run_backtest(
        "us_firm_characteristics",
        "pred1",
        spec,
        prices=pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "symbol": ["A"], "close": [1.0]}),
        predictions=pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["A"],
                "y_score": [1.0],
                "y_true": [0.1],
            }
        ),
        register=True,
    )
    assert result.metrics["ruin"] == 1.0
    assert math.isnan(result.metrics["sharpe"])
    # The line `12_portfolio_management.py:245` runs on every sweep result.
    assert f"Sharpe={result.metrics.get('sharpe', 0):.3f}" == "Sharpe=nan"
