"""The ETF momentum baseline, shared by the chapter's economic diagnostics.

`01_backtest_first_principles` builds a fixed-rule ETF momentum strategy from scratch and
explains every step. The diagnostic notebooks that follow need the same strategy but should
not restate it, so this module holds the pieces: the universe, the signal, the weighting
rule, the simulator and the metric set.

The simulator here is the one `01_backtest_first_principles` derives, ported rather than
reimplemented. Orders are sized at the next session's open, sells are filled before buys so
their proceeds are available to spend, purchases are scaled to the cash on hand, and equity
is marked at the close. Running the baseline through it reproduces that notebook's numbers,
which is asserted by `tests/test_etf_baseline_parity.py` rather than claimed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from ml4t.diagnostic.metrics import sharpe_ratio, sortino_ratio

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
    """Aligned daily open and close prices plus the 10y-2y slope, for the ETF universe."""

    prices: pd.DataFrame
    opens: pd.DataFrame
    yc_slope: pd.Series

    @property
    def regime_risk_on(self) -> pd.Series:
        """Risk-on indicator at threshold 0.5%; aligned to prices.index."""
        return (self.yc_slope > DEFAULT_REGIME_THRESHOLD).astype(int)


def load_panel(start: str = DEFAULT_START, end: str = DEFAULT_END) -> Panel:
    """Load open and close prices for the ETF universe plus the aligned yield-curve slope.

    Both price surfaces are needed: closes carry the signal and mark the account, opens are
    where orders fill. Loading them from one frame keeps them on a common date index.
    """
    etf_pl = load_etfs().filter(
        pl.col("symbol").is_in(ETF_UNIVERSE)
        & (pl.col("timestamp") >= pl.lit(start).str.to_date())
        & (pl.col("timestamp") <= pl.lit(end).str.to_date())
    )

    def _pivot(field: str) -> pd.DataFrame:
        return (
            etf_pl.pivot(on="symbol", index="timestamp", values=field)
            .sort("timestamp")
            .to_pandas()
            .set_index("timestamp")
        )

    prices = _pivot("close")
    opens = _pivot("open")
    available = [s for s in ETF_UNIVERSE if s in prices.columns]
    prices = prices[available].ffill()
    opens = opens[available].ffill()
    common = prices.dropna().index.intersection(opens.dropna().index)
    prices, opens = prices.loc[common], opens.loc[common]

    macro = load_macro()
    yc_pl = macro.select(
        [pl.col("timestamp"), (pl.col("YIELD_CURVE_SLOPE") / 100).alias("slope")]
    ).drop_nulls()
    yc = yc_pl.to_pandas().set_index("timestamp")["slope"]
    yc_aligned = yc.reindex(prices.index, method="ffill")
    return Panel(prices=prices, opens=opens, yc_slope=yc_aligned)


def momentum_score(prices: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    """6-month cumulative return divided by 6-month annualized realized vol."""
    cum_ret = prices.pct_change(lookback)
    vol = prices.pct_change().rolling(lookback).std() * np.sqrt(252)
    return cum_ret / vol


def monthly_rebalance_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Last trading day of each calendar month in `prices`, where the signal is formed."""
    return prices.index[~prices.index.to_period("M").duplicated(keep="last")]


def fill_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sessions on which a month-end signal is filled: the day after each month end.

    Plus the first session of the sample, when the account is bought from cash. This is the
    rebalance schedule `01_backtest_first_principles` uses, and it fires whether or not the
    target moved: holdings drift with prices between rebalances, so restoring a constant
    target still trades.
    """
    month_end = ~index.to_period("M").duplicated(keep="last")
    return index[np.r_[True, month_end[:-1]]]


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
    """Equal weight across the whole universe, rebalanced monthly."""
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def inverse_vol_weights(prices: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    """Inverse trailing-volatility weights across the full universe, monthly rebalance.

    Like ``momentum_weights``, the signal is read at each month-end close and the returned
    weights are shifted one session so they are executable at the next open.
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
    """A constant stock-bond split, restored to target on each rebalance session."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[equity] = 0.60
    weights[bond] = 0.40
    return weights


@dataclass
class SimResult:
    equity: pd.Series
    returns: pd.Series
    trades_dollar: pd.Series  # dollars traded on each session, both legs summed
    holdings_value: pd.DataFrame  # per-asset dollar value at each bar (post-trade)


def simulate(
    panel: Panel,
    weights: pd.DataFrame,
    *,
    initial_cash: float = INITIAL_CASH,
    fees: float = DEFAULT_FEES,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> SimResult:
    """Fill already-executable target weights at the open, mark the account at the close.

    This is the simulator `01_backtest_first_principles` derives, and the event order is the
    part that matters. On a rebalance session the book is valued at that session's opening
    prices; sells are filled first, their proceeds net of the fee become available cash, and
    purchases are then scaled down if the cash they need exceeds the cash on hand. Equity is
    marked at the close.

    `weights` must already be executable: row *t* is the target the account is moved to at
    session *t*'s open, which means it was formed no later than session *t-1*'s close. Every
    weight builder in this module returns weights in that form.

    Rebalances default to `fill_dates`, the session after each month end plus the first
    session of the sample. Passing an explicit index overrides that.
    """
    closes, opens = panel.prices, panel.opens
    index = closes.index
    if rebalance_dates is None:
        rebalance_dates = fill_dates(index)
    scheduled = set(rebalance_dates)

    equity = np.zeros(len(index))
    traded = np.zeros(len(index))
    holdings_value = pd.DataFrame(0.0, index=index, columns=closes.columns)
    holdings = pd.Series(0.0, index=closes.columns)
    cash = initial_cash

    for i, day in enumerate(index):
        if day in scheduled:
            open_px = opens.loc[day]
            open_values = holdings * open_px
            target_values = weights.loc[day] * (cash + open_values.sum())

            sells = (open_values - target_values).clip(lower=0.0)
            holdings -= sells / open_px
            cash += sells.sum() * (1 - fees)

            requested = (target_values - holdings * open_px).clip(lower=0.0)
            required = requested.sum() * (1 + fees)
            scale = min(1.0, cash / required) if required > 0 else 0.0
            buys = requested * scale
            holdings += buys / open_px
            cash -= buys.sum() * (1 + fees)
            traded[i] = float(sells.sum() + buys.sum())

        if cash < -1e-8:
            raise RuntimeError(f"cash went negative on {day:%Y-%m-%d}: {cash}")
        cash = max(cash, 0.0)
        close_values = holdings * closes.loc[day]
        holdings_value.iloc[i, :] = close_values.to_numpy()
        equity[i] = cash + float(close_values.sum())

    equity_series = pd.Series(equity, index=index)
    # The first session earns a return too: the account was bought at that session's open and
    # is marked at its close. `pct_change` cannot see it, so it is seeded from the starting
    # cash. Dropping it would leave the return series one observation short of the account's.
    returns = equity_series.pct_change()
    returns.iloc[0] = equity_series.iloc[0] / initial_cash - 1.0
    return SimResult(
        equity=equity_series,
        returns=returns,
        trades_dollar=pd.Series(traded, index=index),
        holdings_value=holdings_value,
    )


def metrics(result: SimResult, periods_per_year: int = 252) -> dict[str, float]:
    """Growth, risk and risk-adjusted metrics for one simulated account.

    Sharpe and Sortino come from `ml4t.diagnostic`, which is also what
    `01_backtest_first_principles` calls, so the three notebooks that report this strategy
    report the same statistic. Both put the mean periodic excess return in the numerator.
    The compound growth rate is reported beside them and is deliberately not substituted
    into either: the arithmetic mean exceeds the geometric one by roughly half the variance,
    so a Sharpe built from a growth rate is a different and smaller number wearing the same
    name.
    """
    returns = result.returns
    equity = result.equity
    total = float(np.prod(1.0 + returns.to_numpy()) - 1.0)
    cagr = float((1 + total) ** (periods_per_year / len(returns)) - 1) if len(returns) else 0.0
    vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(sharpe_ratio(returns.to_numpy(), periods_per_year=periods_per_year))
    sortino = float(sortino_ratio(returns.to_numpy(), periods_per_year=periods_per_year))
    cum = np.cumprod(1.0 + returns.to_numpy())
    mdd = float(np.min(cum / np.maximum.accumulate(np.r_[1.0, cum])[1:] - 1.0))
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
    """Dollars traded per dollar of capital per year, counting both legs of a rebalance."""
    years = len(result.equity) / periods_per_year
    if years <= 0 or result.equity.mean() <= 0:
        return 0.0
    return float(result.trades_dollar.sum() / result.equity.mean() / years)


def break_even_cost_bp(result_zero_cost: SimResult, periods_per_year: int = 252) -> float:
    """Per-leg cost, in basis points, that would consume the whole gross growth rate.

    Growth rate divided by turnover. It ignores compounding, so it sits above the cost at
    which a re-simulated strategy actually reaches zero; `14_cost_sensitivity` shows both.
    """
    m = metrics(result_zero_cost, periods_per_year)
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
    result = simulate(panel, weights, fees=fees)
    return panel, weights, result
