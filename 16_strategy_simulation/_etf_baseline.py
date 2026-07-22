"""Shared ETF baseline construction for the §16.4–§16.6 teaching notebooks.

The NB01 (§16.4) "auditable non-ML baseline" defines the working example used by the
§16.6 economic-diagnostic notebooks (cost sensitivity, baseline comparison, regime
slicing). This module exposes the underlying ingredients — universe, signal, weighting
rules, simulator, metrics — so each diagnostic notebook can stay focused on its
diagnostic rather than re-deriving the baseline from scratch.

The simulator and metrics here intentionally match NB01 numerically; calling
`run_baseline()` produces the same Sharpe (0.65) and MaxDD (-31.9%) as Table 16.4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from data import load_etfs, load_macro

ETF_UNIVERSE: list[str] = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "GLD", "VNQ", "DBC"]
DEFAULT_START: str = "2010-01-01"
DEFAULT_END: str = "2024-01-01"
DEFAULT_LOOKBACK: int = 126  # 6 months of trading days
DEFAULT_TOP_N: int = 3
DEFAULT_REGIME_THRESHOLD: float = 0.005  # 10y-2y slope (decimal)
DEFAULT_FEES: float = 0.0005  # 5 bp per leg
INITIAL_CASH: float = 100_000.0


@dataclass
class Panel:
    """Aligned daily close prices and 10y-2y yield-curve slope for the ETF universe."""

    prices: pd.DataFrame
    yc_slope: pd.Series

    @property
    def regime_risk_on(self) -> pd.Series:
        """Risk-on indicator at threshold 0.5%; aligned to prices.index."""
        return (self.yc_slope > DEFAULT_REGIME_THRESHOLD).astype(int)


def load_panel(start: str = DEFAULT_START, end: str = DEFAULT_END) -> Panel:
    """Load close prices for the 10-ETF universe plus aligned yield-curve slope."""
    etf_pl = load_etfs().filter(
        pl.col("symbol").is_in(ETF_UNIVERSE)
        & (pl.col("timestamp") >= pl.lit(start).str.to_date())
        & (pl.col("timestamp") <= pl.lit(end).str.to_date())
    )
    prices = (
        etf_pl.pivot(on="symbol", index="timestamp", values="close")
        .sort("timestamp")
        .to_pandas()
        .set_index("timestamp")
    )
    available = [s for s in ETF_UNIVERSE if s in prices.columns]
    prices = prices[available].ffill()

    macro = load_macro()
    if "YIELD_CURVE_SLOPE" in macro.columns:
        yc_pl = macro.select(
            [pl.col("timestamp"), (pl.col("YIELD_CURVE_SLOPE") / 100).alias("slope")]
        ).drop_nulls()
    else:
        yc_pl = macro.select(
            [pl.col("timestamp"), ((pl.col("dgs10") - pl.col("dgs2")) / 100).alias("slope")]
        ).drop_nulls()
    yc = yc_pl.to_pandas().set_index("timestamp")["slope"]
    yc_aligned = yc.reindex(prices.index, method="ffill")
    return Panel(prices=prices, yc_slope=yc_aligned)


def momentum_score(prices: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    """6-month cumulative return divided by 6-month annualized realized vol."""
    cum_ret = prices.pct_change(lookback)
    vol = prices.pct_change().rolling(lookback).std() * np.sqrt(252)
    return cum_ret / vol


def monthly_rebalance_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Last trading day of each calendar month in `prices`."""
    periods = prices.index.to_period("M")
    # The "last trading day in each month" is the index entry whose month
    # period is not followed by the same period — avoids the pandas≥2.2
    # FutureWarning on groupby(...).apply(scalar).
    is_last = ~periods.duplicated(keep="last")
    return prices.index[is_last]


def momentum_weights(
    panel: Panel,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    top_n: int = DEFAULT_TOP_N,
    regime_threshold: float = DEFAULT_REGIME_THRESHOLD,
    defensive: Sequence[tuple[str, float]] = (("AGG", 0.60), ("TLT", 0.40)),
) -> pd.DataFrame:
    """NB01 §16.4 baseline: top-N risk-adjusted momentum, defensive on flat curve."""
    prices = panel.prices
    regime = (panel.yc_slope > regime_threshold).astype(int)
    rank = momentum_score(prices, lookback).rank(axis=1, ascending=False)

    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    rebalance_dates = monthly_rebalance_dates(prices)
    first_valid = rank.dropna(how="all").index[0]

    weights.iloc[0, :] = 0.0
    for sym, w in defensive:
        weights.iloc[0, weights.columns.get_loc(sym)] = w

    for d in rebalance_dates:
        weights.loc[d, :] = 0.0
        if d < first_valid or d not in rank.index or regime.loc[d] == 0:
            for sym, w in defensive:
                weights.loc[d, sym] = w
            continue
        ranks = rank.loc[d]
        top = ranks[ranks <= top_n].index.tolist()
        if top:
            w = 1.0 / len(top)
            for sym in top:
                weights.loc[d, sym] = w
    weights = weights.ffill()
    # Shift target weights forward by one bar so the month-end-close signal
    # *executes* at the next trading day's open — the first trading day of the
    # next month for these month-end rebalance dates (§16.2 close-to-next-open
    # execution; same-bar would be lookahead).
    weights = weights.shift(1)
    weights.iloc[0, :] = 0.0
    for sym, w in defensive:
        weights.iloc[0, weights.columns.get_loc(sym)] = w
    return weights.ffill()


def equal_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """1/N across the full universe, monthly rebalance — uses all available ETFs."""
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    n = prices.shape[1]
    rebalance_dates = monthly_rebalance_dates(prices)
    weights.iloc[0, :] = 1.0 / n
    for d in rebalance_dates:
        weights.loc[d, :] = 1.0 / n
    return weights.ffill()


def inverse_vol_weights(prices: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    """Inverse trailing-volatility weights across the full universe, monthly rebalance.

    Like ``momentum_weights``, the signal is computed at each month-end close
    and *executes* at the next trading day's open (one-bar shift) — the first
    trading day of the next month for these month-end rebalance dates.
    """
    vol = prices.pct_change().rolling(lookback).std() * np.sqrt(252)
    inv_vol = 1.0 / vol.replace(0, np.nan)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    rebalance_dates = monthly_rebalance_dates(prices)
    first_valid = inv_vol.dropna(how="all").index[0]
    weights.iloc[0, :] = 1.0 / prices.shape[1]
    for d in rebalance_dates:
        if d < first_valid:
            weights.loc[d, :] = 1.0 / prices.shape[1]
            continue
        row = inv_vol.loc[d]
        s = row.sum()
        weights.loc[d, :] = (row / s).fillna(0.0) if s > 0 else 1.0 / prices.shape[1]
    weights = weights.ffill().shift(1)
    weights.iloc[0, :] = 1.0 / prices.shape[1]
    return weights.ffill()


def static_60_40(prices: pd.DataFrame, equity: str = "SPY", bond: str = "AGG") -> pd.DataFrame:
    """Static 60% equity / 40% bond (no rebalance drift adjustment beyond simulator)."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[equity] = 0.60
    weights[bond] = 0.40
    return weights


@dataclass
class SimResult:
    equity: pd.Series
    returns: pd.Series
    trades_dollar: pd.Series  # gross dollar volume traded each rebalance day
    holdings_value: pd.DataFrame  # per-asset dollar value at each bar (post-trade)


def simulate(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    initial_cash: float = INITIAL_CASH,
    fees: float = DEFAULT_FEES,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> SimResult:
    """Bar-by-bar simulator with per-leg proportional costs.

    Mirrors NB01 §16.4: cost = |trade_dollar| * fees per leg. Rebalance dates
    default to the days when target weights change by >1% (matches NB01); pass
    explicit dates (e.g. monthly grid from `monthly_rebalance_dates`) to force
    rebalancing for allocators whose targets are constant by construction.
    """
    n = len(prices)
    pv = np.zeros(n)
    trades_dollar = np.zeros(n)
    holdings_value = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    holdings = pd.Series(0.0, index=prices.columns)
    cash = initial_cash

    if rebalance_dates is None:
        weight_change = weights.diff().abs().sum(axis=1)
        rebalance_dates = weight_change[weight_change > 0.01].index
    if prices.index[0] not in rebalance_dates:
        rebalance_dates = rebalance_dates.insert(0, prices.index[0])

    for i, d in enumerate(prices.index):
        cp = prices.loc[d]
        hv = (holdings * cp).sum()
        total = cash + hv
        pv[i] = total

        if d in rebalance_dates:
            target_value = weights.loc[d] * total
            current_value = holdings * cp
            trade_dollar = target_value - current_value
            for asset in prices.columns:
                t = trade_dollar[asset]
                if abs(t) > 1.0:
                    cash -= abs(t) * fees
                    holdings[asset] += t / cp[asset]
                    cash -= t
                    trades_dollar[i] += abs(t)
        holdings_value.iloc[i, :] = (holdings * cp).values

    equity = pd.Series(pv, index=prices.index)
    return SimResult(
        equity=equity,
        returns=equity.pct_change().dropna(),
        trades_dollar=pd.Series(trades_dollar, index=prices.index),
        holdings_value=holdings_value,
    )


def metrics(equity: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Standard metric set: total return, CAGR, vol, Sharpe, Sortino, MaxDD, Calmar."""
    r = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((1 + total) ** (periods_per_year / len(r)) - 1) if len(r) else 0.0
    vol = float(r.std() * np.sqrt(periods_per_year))
    sharpe = cagr / vol if vol > 0 else 0.0
    downside = r[r < 0]
    dvol = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) else 0.0
    sortino = cagr / dvol if dvol > 0 else 0.0
    cum = equity / equity.iloc[0]
    rmax = cum.expanding().max()
    mdd = float(((cum - rmax) / rmax).min())
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
    }


def annualized_turnover(result: SimResult, periods_per_year: int = 252) -> float:
    """Annualized one-way turnover = sum(|trade_$|) / mean(equity) / years."""
    years = len(result.equity) / periods_per_year
    if years <= 0 or result.equity.mean() <= 0:
        return 0.0
    return float(result.trades_dollar.sum() / result.equity.mean() / years)


def break_even_cost_bp(result_zero_cost: SimResult, periods_per_year: int = 252) -> float:
    """Per-leg cost (in bp) at which gross return == 0.

    Approximation: BE = (annualized gross return) / (annualized turnover) * 10000.
    Uses the zero-cost simulation result.
    """
    m = metrics(result_zero_cost.equity, periods_per_year)
    turn = annualized_turnover(result_zero_cost, periods_per_year)
    if turn <= 0:
        return float("inf")
    return float(m["cagr"] / turn * 10_000)


def run_baseline(
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    fees: float = DEFAULT_FEES,
) -> tuple[Panel, pd.DataFrame, SimResult]:
    """Convenience: load panel, build NB01 momentum weights, simulate. Returns all three."""
    panel = load_panel(start, end)
    weights = momentum_weights(panel)
    result = simulate(panel.prices, weights, fees=fees)
    return panel, weights, result
