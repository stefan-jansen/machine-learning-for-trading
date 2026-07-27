"""Per-fold backtest metrics must annualize at the return-series frequency.

`daily_returns` carries one row per session no matter how often the strategy
rebalances. Annualizing it at the *rebalance cadence* under-states Sharpe,
Sortino and volatility by sqrt(true_ppy / cadence_ppy), which for a
monthly-rebalanced ETF strategy on daily returns is sqrt(252/12) = 4.583. That
is how the ETF fold Sharpes came to sit in [-0.040, +0.618] while the
full-window Sharpe of the same return series was +1.3256. CAGR carries the
factor in a compounding exponent, so it moves too but not by that ratio, and
Calmar moves with CAGR. total_return and max_drawdown do not move.

The authority is `evaluation.periods_per_year` in each case study's setup.yaml.
It is the only source that distinguishes a genuinely monthly series
(`us_firm_characteristics`, 12) from a monthly-rebalanced strategy marked to
market daily (`etfs`, 252). A calendar lookup gets the first one wrong and a
cadence lookup gets the second one wrong.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from case_studies.utils.uncertainty import periods_per_year_from_setup

CASE_STUDIES = [
    "cme_futures",
    "crypto_perps_funding",
    "etfs",
    "fx_pairs",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "sp500_options",
    "us_equities_panel",
    "us_firm_characteristics",
]


@pytest.mark.parametrize("case_study", CASE_STUDIES)
def test_every_case_study_declares_periods_per_year(case_study: str) -> None:
    """The fix depends on this being declared everywhere, so pin it."""
    assert periods_per_year_from_setup(case_study) > 0


def test_daily_marked_case_studies_are_not_annualized_at_rebalance_cadence() -> None:
    """`etfs` rebalances monthly but marks to market daily → 252, not 12.

    This is the exact confusion the bug was. If someone re-derives the
    annualization factor from `rebalance.cadence`, this fails.
    """
    assert periods_per_year_from_setup("etfs") == 252
    assert periods_per_year_from_setup("crypto_perps_funding") == 365
    # The one case study whose return grid genuinely is monthly.
    assert periods_per_year_from_setup("us_firm_characteristics") == 12


def _sharpe(returns: np.ndarray, ppy: int) -> float:
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(ppy))


def test_fold_metrics_use_setup_value_when_caller_passes_zero(monkeypatch) -> None:
    """`periods_per_year=0` must resolve via setup.yaml, not the calendar.

    Before the fix the zero path fell straight through to an exchange-calendar
    lookup, which returns ~252 for every equity venue and so disagreed with
    `us_firm_characteristics`'s declared 12. Fold boundaries are supplied here
    rather than read from a modeling dataset, so the test pins the resolution
    branch without needing data on disk.

    Discriminating: against the pre-fix implementation the computed Sharpe is
    the 252 one, and the first assertion fails.
    """
    import case_studies.utils.cv_window as cv_window
    from case_studies.utils.registry.metrics import compute_backtest_fold_metrics

    monkeypatch.setattr(
        cv_window,
        "fold_boundaries",
        lambda cs, label: [{"fold": 0, "val_start": "2020-01-31", "val_end": "2021-12-31"}],
    )

    ts = pl.date_range(date(2020, 1, 31), date(2021, 12, 31), interval="1mo", eager=True)
    rng = np.random.default_rng(7)
    rets = rng.normal(0.004, 0.03, len(ts))
    frame = pl.DataFrame({"timestamp": ts, "daily_return": rets})

    out = compute_backtest_fold_metrics(
        frame, "us_firm_characteristics", label="fwd_ret_1m", periods_per_year=0
    )

    assert set(out) == {0}
    assert out[0]["sharpe"] == pytest.approx(_sharpe(rets, 12), rel=1e-6)
    # The value the exchange-calendar fallback would have produced.
    assert out[0]["sharpe"] != pytest.approx(_sharpe(rets, 252), rel=1e-6)


def test_fold_metrics_reconcile_the_setup_value_against_a_thinned_grid(monkeypatch) -> None:
    """The omitted-argument path is the backfill path, and it sees thinned grids.

    `BacktestExplorer.backfill_fold_metrics` reads a stored `daily_returns.parquet`
    and calls `compute_backtest_fold_metrics` without a factor. Those artifacts
    include the ones the vectorized path wrote, which are thinned to rebalance
    dates. Taking the declared 252 there annualizes a roughly 74-row year at the
    daily rate, so the backfilled fold metrics would contradict the overall
    metrics stored beside them.

    Discriminating: reading setup.yaml unconditionally, which is what this path
    did before the reconciliation, gives the 252 value the last assertion rejects.
    """
    import case_studies.utils.cv_window as cv_window
    from case_studies.utils.registry.metrics import compute_backtest_fold_metrics

    monkeypatch.setattr(
        cv_window,
        "fold_boundaries",
        lambda cs, label: [{"fold": 0, "val_start": "2014-01-01", "val_end": "2021-12-31"}],
    )

    ts = pl.date_range(
        date(2014, 1, 1), date(2021, 12, 31), interval="1d", eager=True
    ).gather_every(5)
    rng = np.random.default_rng(13)
    rets = rng.normal(0.001, 0.02, len(ts))
    frame = pl.DataFrame({"timestamp": ts, "daily_return": rets})

    out = compute_backtest_fold_metrics(frame, "sp500_options", label="ret_to_expiry")

    observed = int(len(ts) / ((ts[-1] - ts[0]).days / 365.25))
    assert observed < 120, "expected the thinned grid to be materially coarser than daily"
    assert out[0]["sharpe"] == pytest.approx(_sharpe(rets, observed), rel=1e-6)
    assert out[0]["sharpe"] != pytest.approx(_sharpe(rets, 252), rel=1e-6)


def _vectorized(case_study: str, label: str, cadence: str, ts) -> tuple[dict, np.ndarray]:
    from case_studies.utils.backtest_runner import _run_vectorized

    rng = np.random.default_rng(11)
    y_true = rng.normal(0.002, 0.02, len(ts))
    frame = {"timestamp": ts, "symbol": ["AAA"] * len(ts)}
    out = _run_vectorized(
        weights=pl.DataFrame({**frame, "weight": [1.0] * len(ts)}),
        predictions=pl.DataFrame({**frame, "y_true": y_true, "y_pred": y_true}),
        prices=pl.DataFrame({**frame, "close": 100.0}),
        cost_spec={"model": "percentage", "commission_bps": 0.0, "slippage_bps": 0.0},
        cadence=cadence,
        label=label,
        case_study=case_study,
        initial_cash=1e6,
        prediction_hash=None,
    )
    return out, out["daily_returns"]["daily_return"].to_numpy()


def test_vectorized_path_uses_declared_frequency_when_the_grid_matches_it() -> None:
    """`us_firm_characteristics` is the case study production sends down this path.

    Its rebalance_step is 1 and its predictions are monthly, so the grid this
    function writes is monthly and the declared 12 describes it.
    """
    ts = pl.date_range(date(2014, 1, 31), date(2021, 12, 31), interval="1mo", eager=True)
    out, realized = _vectorized("us_firm_characteristics", "fwd_ret_1m", "monthly_month_end", ts)

    assert out["metrics"]["sharpe"] == pytest.approx(_sharpe(realized, 12), rel=1e-6)


def test_vectorized_path_defers_to_the_grid_it_wrote_when_thinning_coarsens_it() -> None:
    """The declared value describes the registered grid, not always this one.

    This path thins predictions to rebalance dates first. `sp500_options` has
    rebalance_step 5, so a daily prediction grid leaves roughly 74 rows a year
    against a declared 252. Annualizing at 252 there would inflate Sharpe by
    sqrt(252/74). Discriminating both ways: the pre-fix code used the cadence
    table's 252 for "daily", and reading setup.yaml unconditionally gives 252
    as well.
    """
    ts = pl.date_range(date(2014, 1, 1), date(2021, 12, 31), interval="1d", eager=True)
    out, realized = _vectorized("sp500_options", "ret_to_expiry", "daily", ts)

    n = len(realized)
    span_years = (ts[-1] - ts[0]).days / 365.25
    observed = n / span_years
    assert observed < 120, "expected the thinned grid to be materially coarser than daily"
    assert out["metrics"]["sharpe"] == pytest.approx(_sharpe(realized, int(observed)), rel=1e-6)
    assert out["metrics"]["sharpe"] != pytest.approx(_sharpe(realized, 252), rel=1e-6)


def test_overall_and_fold_metrics_resolve_the_same_factor() -> None:
    """One backtest must not annualize its overall and per-fold metrics apart.

    `run_backtest` and `_run_vectorized` both call `resolve_periods_per_year` on
    the same return frame, so a thinned grid that pushes the overall metrics off
    the declared value pushes the fold metrics with it.
    """
    from case_studies.utils.backtest_runner import resolve_periods_per_year

    daily = pl.DataFrame(
        {
            "timestamp": pl.date_range(date(2014, 1, 1), date(2021, 12, 31), "1d", eager=True),
            "daily_return": 0.001,
        }
    )
    assert resolve_periods_per_year("etfs", daily) == 252

    # A grid thinned well below the declared 252 must not be annualized at 252.
    thinned = daily.gather_every(5)
    assert resolve_periods_per_year("sp500_options", thinned) < 120

    # A genuinely monthly series against a declared 12 stays on the round value.
    monthly = pl.DataFrame(
        {
            "timestamp": pl.date_range(date(2014, 1, 31), date(2021, 12, 31), "1mo", eager=True),
            "daily_return": 0.001,
        }
    )
    assert resolve_periods_per_year("us_firm_characteristics", monthly) == 12


def test_a_long_internal_gap_does_not_look_like_a_coarser_grid() -> None:
    """A hole in a daily series is not evidence that the series is not daily.

    Rows-over-elapsed-span cannot tell a suspended market from a thinned grid,
    and reading a gapped daily series as coarse would annualize it below its own
    convention. The rate is measured over typical intervals instead, so the hole
    leaves the count and the span together.

    Discriminating: the naive density is asserted to be far enough below 252
    that it would have crossed the reconciliation threshold and overridden the
    declared value.
    """
    from case_studies.utils.backtest_runner import (
        observed_periods_per_year,
        reconcile_periods_per_year,
    )

    sessions = pl.date_range(date(2016, 1, 1), date(2021, 12, 31), "1d", eager=True).filter(
        pl.date_range(date(2016, 1, 1), date(2021, 12, 31), "1d", eager=True).dt.weekday() <= 5
    )
    # A market closed for three and a half of the six years.
    gapped = sessions.filter((sessions < date(2017, 7, 1)) | (sessions >= date(2021, 1, 1)))
    frame = pl.DataFrame({"timestamp": gapped, "daily_return": 0.001})

    naive_density = len(gapped) / ((gapped[-1] - gapped[0]).days / 365.25)
    assert naive_density < 126, "test is only meaningful if the naive measure would have fired"

    assert observed_periods_per_year(frame) == pytest.approx(252, rel=0.05)
    assert reconcile_periods_per_year(252, frame, case_study="etfs") == 252

    # The thinned case still has to be caught, or the guard is useless.
    thinned = pl.DataFrame({"timestamp": sessions.gather_every(5), "daily_return": 0.001})
    assert reconcile_periods_per_year(252, thinned, case_study="sp500_options") < 120


def test_an_intraday_grid_is_not_annualized_above_its_declared_convention() -> None:
    """A finer grid than declared is a stale setup.yaml, not a licence to inflate.

    Dropping overnight gaps makes a 15-minute NYSE series look continuous, so
    measuring its rate gives roughly 35,000 a year against the 252 declared for
    `nasdaq100_microstructure`. Deferring to that would multiply Sharpe by about
    12. The override only fires when the grid is coarser than declared, which is
    the one thing thinning can do to it.
    """
    from case_studies.utils.backtest_runner import (
        observed_periods_per_year,
        reconcile_periods_per_year,
    )

    session_bars = []
    for day in pl.date_range(date(2021, 1, 4), date(2021, 12, 31), "1d", eager=True):
        if day.weekday() > 5:
            continue
        session_bars += [
            datetime(day.year, day.month, day.day, 9, 30) + timedelta(minutes=15 * i)
            for i in range(26)
        ]
    frame = pl.DataFrame({"timestamp": session_bars, "daily_return": 0.0001})

    assert observed_periods_per_year(frame) > 20_000, "overnight gaps are dropped by design"
    assert reconcile_periods_per_year(252, frame, case_study="nasdaq100_microstructure") == 252


def test_engine_path_annualizes_like_the_rest_of_the_pipeline() -> None:
    """The engine path read the exchange calendar, which no case study declares.

    `calendar_periods_per_year` returns 252 for every equity venue, so an
    engine-backed `us_firm_characteristics` run annualized its overall metrics
    at 252 while its per-fold metrics used the declared 12 - a factor of 4.58 on
    Sharpe between two numbers stored side by side. Discriminating: the calendar
    value is asserted directly, so taking it again fails the second assertion.
    """
    from case_studies.utils.backtest_runner import (
        calendar_periods_per_year,
        overall_periods_per_year,
    )

    monthly = pl.DataFrame(
        {
            "timestamp": pl.date_range(date(2014, 1, 31), date(2021, 12, 31), "1mo", eager=True),
            "daily_return": 0.001,
        }
    )

    assert calendar_periods_per_year("NYSE") == 252
    assert overall_periods_per_year("us_firm_characteristics", "NYSE", monthly) == 12
    # No case study named → nothing declares a value, so the calendar stands.
    assert overall_periods_per_year(None, "NYSE", monthly) == 252


def test_cadence_table_is_gone() -> None:
    """The removed table was the defect's source; keep it removed."""
    import case_studies.utils.backtest_runner as br

    assert not hasattr(br, "_PERIODS_PER_YEAR")
