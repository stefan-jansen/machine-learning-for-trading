# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Market Impact and Liquidity in Backtests
#
# **Chapter 21: Reinforcement Learning for Execution and Hedging**
#
# ## Purpose
#
# A backtest that charges a fixed number of basis points per trade is charging
# for the spread and calling it the cost of trading. The larger part of the cost
# of a real order is what the order itself does to the price, and that depends
# not on the order's size in dollars but on its size relative to the volume
# available to absorb it. The same order is a rounding error in one name and a
# day's trading in another.
#
# This notebook makes the difference measurable rather than arguable. One
# momentum strategy, one dollar book, run on real daily bars for five stocks
# chosen across the liquidity range, under four strengths of a square-root
# impact model. Then the same experiment across two cohorts of the whole
# universe, to see how often the impact charge is the difference between a
# profitable backtest and an unprofitable one.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Compute a name's participation rate - the size of an order against the
#   volume that trades in a day - and explain why it, rather than the order's
#   dollar value, determines what the order costs.
# - Apply the square-root impact law in a backtest, and say which of its inputs
#   have to be estimated before the evaluation period to keep the result honest.
# - Select names for an experiment on evidence from a formation window only, and
#   report what happened to the ones that later turned out to be unusable
#   without letting that report change the selection.
# - Measure how much of a strategy's paper return is left after a realistic
#   impact charge, separately for a liquid cohort and a thin one.
#
# ## Book reference
#
# Sections 21.4, *Application I: Optimal Trade Execution*, and 21.8, *The
# Simulation-to-Reality Gap*.
#
# ## Prerequisites
#
# - `02_optimal_execution_ppo`, for implementation shortfall and the impact
#   models an execution schedule is trying to manage.
# - Daily US equity bars, read through `data.load_us_equities`.
# - `ml4t.backtest`, for the engine, the broker and the impact models.

# %%
"""Market Impact and Liquidity in Backtests - how the same order erodes returns differently across the liquidity spectrum."""

import warnings
from datetime import datetime
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display
from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy
from ml4t.backtest.broker import Broker
from ml4t.backtest.execution.impact import NoImpact, SquareRootImpact
from plotly.subplots import make_subplots

import utils  # noqa: F401  - sets the Plotly renderer so figures carry a static PNG
from data import load_us_equities
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# polars emits a repeating FutureWarning from its own deprecations here and it
# says nothing about this run. Convergence, overflow and invalid-value warnings
# stay visible: they report conditions the results depend on.
warnings.filterwarnings("ignore", category=FutureWarning, module="polars")

# %% tags=["parameters"]
START_DATE = "2010-01-01"
END_DATE = "2016-12-31"
FORMATION_END_DATE = "2012-12-31"  # last day any selection or calibration may read
EVALUATION_START_DATE = "2013-01-01"  # first day of every reported backtest
LOOKBACK = 21  # sessions in the momentum signal, about one trading month
BOOK_USD = 5_000_000  # dollar size of the order, held fixed across every name
IMPACT_COEFFICIENTS = [0.0, 0.1, 0.3, 0.6]  # strengths of the square-root impact model
GROSS_MIN = 0.3  # formation gross return a name must clear to enter the pool
COHORT_SAMPLE = 60  # names drawn from each liquidity cohort
MIN_FORMATION_OBS = 700  # sessions a name must have in the formation window
LIQUID_MAX_PARTICIPATION = 0.10  # order below this share of daily volume is the liquid cohort
THIN_MIN_PARTICIPATION = 1.0  # order above this share of daily volume is the thin cohort
SEED = 42

# %%
set_global_seeds(SEED)
rng = np.random.default_rng(SEED)

# %% [markdown]
# ### What the settings decide
#
# `BOOK_USD` is held fixed on purpose. Sizing each order to the name it trades
# would remove the effect being measured: the whole point is that one order,
# unchanged, is a different proposition depending on where it is sent.
#
# The two dates do the other half of the work. Everything that decides which
# names enter the experiment, and everything the impact model needs calibrating,
# is read from bars up to `FORMATION_END_DATE`. Every number reported comes from
# bars on or after `EVALUATION_START_DATE`. Without that separation a name would
# be chosen partly because of the returns it is about to be scored on.

# %%
display(
    Markdown(f"""
- **Order**: {BOOK_USD:,.0f} USD, the same in every name.
- **Strategy**: long while {LOOKBACK}-session momentum is positive, flat otherwise.
- **Formation window**: {START_DATE} to {FORMATION_END_DATE}, used for selection and for
  each name's liquidity and volatility. A name needs {MIN_FORMATION_OBS} sessions in it.
- **Evaluation window**: {EVALUATION_START_DATE} to {END_DATE}. Every reported return comes
  from here.
- **Impact strengths**: coefficients {", ".join(f"{c:g}" for c in IMPACT_COEFFICIENTS)}, where
  zero switches the impact model off and each larger value is a stronger square-root
  charge. Commission and slippage are charged at every level, including zero, so the
  zero-impact run is a backtest with ordinary trading costs and no impact rather than a
  costless one.
- **Cohorts**: {COHORT_SAMPLE} names drawn from those whose order is under
  {LIQUID_MAX_PARTICIPATION:.0%} of daily volume, and {COHORT_SAMPLE} from those whose order
  is over {THIN_MIN_PARTICIPATION:.0%} of it.
""")
)

# %% [markdown]
# ## Load Real US Equity Data
#
# We load daily bars for the broad US equity universe and use the
# split- and dividend-**adjusted** OHLCV series, so corporate actions do not
# create artificial jumps in the momentum signal. Dollar volume
# (price $\times$ volume) is invariant to splits and serves as our liquidity
# measure throughout.


# %%
def load_adjusted_equities(start: str, end: str) -> pl.DataFrame:
    """Load daily US equities and return adjusted OHLCV under canonical column names."""
    raw = load_us_equities(start_date=start, end_date=end).sort(["symbol", "timestamp"])
    return raw.select(
        "symbol",
        "timestamp",
        pl.col("adj_open").alias("open"),
        pl.col("adj_high").alias("high"),
        pl.col("adj_low").alias("low"),
        pl.col("adj_close").alias("close"),
        pl.col("adj_volume").alias("volume"),
    ).drop_nulls()


# %%
prices = load_adjusted_equities(START_DATE, END_DATE)
formation_prices = prices.filter(pl.col("timestamp") <= pl.lit(FORMATION_END_DATE).str.to_date())
evaluation_prices = prices.filter(
    pl.col("timestamp") >= pl.lit(EVALUATION_START_DATE).str.to_date()
)
assert formation_prices["timestamp"].max() < evaluation_prices["timestamp"].min()
print(
    f"Loaded {prices.height:,} daily bars for {prices['symbol'].n_unique():,} symbols "
    f"({prices['timestamp'].min()} to {prices['timestamp'].max()})"
)

# %% [markdown]
# ## Measure Liquidity Ex Ante
#
# To avoid look-ahead in stock selection and impact calibration, we estimate
# each name's liquidity and volatility only through `FORMATION_END_DATE`.
# Reported backtests begin at `EVALUATION_START_DATE`.


# %%
def formation_statistics(prices: pl.DataFrame) -> pl.DataFrame:
    """Liquidity and volatility estimated only from the formation window."""
    return (
        prices.with_columns(
            (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
            pl.col("close").pct_change().over("symbol").alias("daily_return"),
        )
        .group_by("symbol")
        .agg(
            pl.col("dollar_volume").median().alias("adv_usd"),
            pl.col("daily_return").std().alias("ex_ante_volatility"),
            pl.len().alias("formation_obs"),
        )
        .filter(pl.col("formation_obs") >= MIN_FORMATION_OBS)
    )


# %%
formation_stats = formation_statistics(formation_prices)
liquidity = formation_stats
traded = liquidity.filter(pl.col("adv_usd") > 0)
print(f"Symbols with a full formation window: {liquidity.height:,}")
print(
    f"Median daily dollar volume runs from ${traded['adv_usd'].min():,.0f} to "
    f"${traded['adv_usd'].max():,.0f}"
)
print(
    f"{liquidity.height - traded.height} of them trade on fewer than half their formation "
    "days, so their median dollar volume is zero"
)

# %% [markdown]
# ## Scope to a Strategy That Works on Paper
#
# The point of the demonstration is what impact does to a strategy that looked
# profitable before the evaluation period. We screen on the formation window,
# then measure erosion only on later observations. No evaluation outcome enters
# candidate selection.


# %%
def gross_momentum_return(prices: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """Vectorized gross return of a long-when-positive-momentum signal, per symbol."""
    scored = prices.with_columns(
        pl.col("close").pct_change().over("symbol").alias("ret"),
        (pl.col("close") / pl.col("close").shift(lookback).over("symbol") - 1).alias("momentum"),
    ).with_columns(
        (
            pl.when(pl.col("momentum") > 0).then(1.0).otherwise(0.0).shift(1).over("symbol")
            * pl.col("ret")
        ).alias("strategy_ret")
    )
    return scored.group_by("symbol").agg(
        ((1 + pl.col("strategy_ret").fill_null(0)).product() - 1).alias("gross_return"),
        pl.col("ret").count().alias("n_obs"),
    )


# %%
formation_gross = gross_momentum_return(formation_prices, LOOKBACK).filter(
    pl.col("n_obs") >= MIN_FORMATION_OBS
)
evaluation_coverage = evaluation_prices.group_by("symbol").agg(pl.len().alias("evaluation_obs"))
pool = (
    formation_gross.filter(pl.col("gross_return") > GROSS_MIN)
    .join(formation_stats, on="symbol")
    .sort("adv_usd")
)
coverage_audit = (
    pool.select("symbol")
    .join(evaluation_coverage, on="symbol", how="left")
    .with_columns(pl.col("evaluation_obs").fill_null(0))
)
evaluable_names = coverage_audit.filter(pl.col("evaluation_obs") >= LOOKBACK + 2).height
print(f"Gross-profitable names selected from formation data: {pool.height:,}")
print(
    f"Later evaluation availability: {evaluable_names:,}/{pool.height:,} have at least "
    f"{LOOKBACK + 2} observations; availability does not alter selection"
)

# %% [markdown]
# ## Select a Representative Liquidity Spectrum
#
# From the gross-profitable pool we pick one representative name at each of five
# liquidity tiers, defined by percentiles of ex-ante dollar volume. Within each
# tier we take the **median** gross-return name, so each tier is represented by
# a typical member of it rather than by its most flattering one.


# %%
def select_spectrum(pool: pl.DataFrame) -> pl.DataFrame:
    """Pick the median-gross name in each of five ex-ante liquidity percentile buckets."""
    tiers = [("mega", 0.99), ("large", 0.90), ("mid", 0.65), ("small", 0.30), ("micro", 0.08)]
    adv = pool["adv_usd"]
    rows = []
    for label, q in tiers:
        lo, hi = adv.quantile(max(0.0, q - 0.04)), adv.quantile(min(1.0, q + 0.04))
        bucket = pool.filter((pl.col("adv_usd") >= lo) & (pl.col("adv_usd") <= hi)).sort(
            "gross_return"
        )
        if bucket.height:
            pick = bucket[bucket.height // 2]
            rows.append(
                {
                    "tier": label,
                    "symbol": pick["symbol"][0],
                    "adv_usd": pick["adv_usd"][0],
                    "ex_ante_volatility": pick["ex_ante_volatility"][0],
                }
            )
    return pl.DataFrame(rows)


# %%
spectrum = select_spectrum(pool)
assert spectrum.height == 5, (
    f"select_spectrum produced {spectrum.height} tiers; expected 5 "
    "(mega/large/mid/small/micro) - a degenerate pool would silently drop a row."
)
spectrum = (
    spectrum.join(evaluation_coverage, on="symbol", how="left")
    .with_columns(
        pl.col("evaluation_obs").fill_null(0),
        (BOOK_USD / pl.col("adv_usd")).alias("order_participation"),
    )
    .sort("order_participation")
)
assert (spectrum["evaluation_obs"] >= LOOKBACK + 2).all(), (
    "A formation-selected spectrum name lacks enough later observations for evaluation"
)
spectrum

# %% [markdown]
# ## The Square-Root Impact Model
#
# The temporary price impact of an order follows the Almgren-Chriss square-root
# law, which scales with the square root of the participation rate - the order
# size relative to average daily volume:
#
# $$\text{impact} = c \cdot \sigma \cdot \sqrt{\frac{q}{\text{ADV}}} \cdot p$$
#
# where $c$ is an impact coefficient, $\sigma$ the daily volatility, $q$ the
# order quantity, $\text{ADV}$ the bar's traded volume, and $p$ the price.
# Impact rises with participation, so the same dollar order costs far more in a
# thin stock than a liquid one.

# %% [markdown]
# ## Define the Momentum Strategy
#
# A long-only momentum strategy: hold the stock fully while `LOOKBACK`-day
# momentum is positive, otherwise stay in cash. The same strategy and the same
# dollar book are applied to every stock - what varies across the spectrum is
# liquidity and its formation-period volatility (the impact model uses each
# stock's ex-ante $\sigma$ as well as its ADV).


# %%
class MomentumStrategy(Strategy):
    """Long-only momentum: fully invested when LOOKBACK-day momentum is positive, else flat."""

    def __init__(self, symbol: str, lookback: int = 21, min_trade_notional: float = 1_000.0):
        self.symbol = symbol
        self.lookback = lookback
        self.min_trade_notional = min_trade_notional
        self.price_history: list[float] = []

    def on_start(self, broker: Any) -> None:
        self.price_history = []

    def on_data(self, timestamp: datetime, data: dict, context: dict, broker: Any) -> None:
        if self.symbol not in data:
            return

        price = data[self.symbol]["close"]
        self.price_history.append(price)
        if len(self.price_history) < self.lookback + 1:
            return

        momentum = (self.price_history[-1] / self.price_history[-self.lookback - 1]) - 1

        current_pos = broker.get_position(self.symbol)
        current_qty = current_pos.quantity if current_pos else 0
        account_value = broker.get_account_value()
        target_qty = (account_value / price) if (price > 0 and momentum > 0) else 0

        order_qty = target_qty - current_qty
        if abs(order_qty) * price > self.min_trade_notional:
            broker.submit_order(self.symbol, order_qty)


# %% [markdown]
# ## Single-Stock Backtest Runner
#
# Runs the momentum strategy on one stock under a given impact model. The
# square-root model uses volatility fixed before the evaluation period.


# %%
def run_backtest(
    stock_data: pl.DataFrame,
    symbol: str,
    impact_coef: float,
    book_usd: float,
    ex_ante_volatility: float,
) -> float:
    """Run the momentum backtest on one stock under a square-root impact coefficient."""
    impact_model = (
        NoImpact()
        if impact_coef == 0
        else SquareRootImpact(coefficient=impact_coef, volatility=ex_ante_volatility)
    )

    feed = DataFeed(prices_df=stock_data)
    engine_config = BacktestConfig(
        initial_cash=book_usd,
        commission_rate=0.001,  # 10 bps
        slippage_rate=0.0005,  # 5 bps
        execution_mode=ExecutionMode.NEXT_BAR,
    )
    engine = Engine(feed=feed, strategy=MomentumStrategy(symbol, LOOKBACK), config=engine_config)
    engine.broker = Broker.from_config(engine_config, market_impact_model=impact_model)
    return float(engine.run().equity.total_return)


# %% [markdown]
# ## Run the Spectrum Under Each Impact Level
#
# For each stock in the liquidity spectrum, run the strategy under no, low,
# medium, and high impact. Failures are collected and re-raised rather than
# silently treated as zero return.


# %%
def run_spectrum(prices: pl.DataFrame, spectrum: pl.DataFrame, book_usd: float) -> pl.DataFrame:
    """Backtest every (stock, impact-coefficient) pair in the liquidity spectrum."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for tier_row in spectrum.iter_rows(named=True):
        symbol = tier_row["symbol"]
        stock_data = prices.filter(pl.col("symbol") == symbol).sort("timestamp")
        for impact_coef in IMPACT_COEFFICIENTS:
            try:
                ret = run_backtest(
                    stock_data,
                    symbol,
                    impact_coef,
                    book_usd,
                    tier_row["ex_ante_volatility"],
                )
            except Exception as exc:  # noqa: BLE001 - collected and re-raised below
                errors.append(f"{symbol} (coef {impact_coef}): {exc}")
                continue
            rows.append(
                {
                    "tier": tier_row["tier"],
                    "symbol": symbol,
                    "adv_usd": tier_row["adv_usd"],
                    "order_participation": book_usd / tier_row["adv_usd"],
                    "ex_ante_volatility": tier_row["ex_ante_volatility"],
                    "impact_coef": impact_coef,
                    "total_return": ret,
                }
            )
    if errors:
        raise RuntimeError("Spectrum backtest failed:\n" + "\n".join(errors))
    return pl.DataFrame(rows)


# %%
spectrum_results = run_spectrum(evaluation_prices, spectrum, BOOK_USD)
spectrum_results

# %% [markdown]
# ## Returns and Erosion Across the Spectrum
#
# Pivot the spectrum into net return by tier and impact level, and compute the
# erosion at the strongest impact assumption - the no-impact return minus the
# high-impact return. The erosion is pure impact cost: the only thing that
# changes between the two runs is the impact model.


# %%
def spectrum_summary(spectrum_results: pl.DataFrame) -> pl.DataFrame:
    """Net return by tier and impact level, plus high-impact erosion vs the no-impact baseline."""
    max_coef = max(IMPACT_COEFFICIENTS)
    wide = spectrum_results.pivot(
        values="total_return", index=["tier", "symbol", "order_participation"], on="impact_coef"
    ).sort("order_participation")
    return wide.with_columns((pl.col("0.0") - pl.col(str(max_coef))).alias("erosion_high"))


# %%
summary = spectrum_summary(spectrum_results)
summary

# %% [markdown]
# ## How often is impact the difference between a profit and a loss?
#
# Five names are an illustration. The claim they are meant to support is about
# the whole universe, so the same experiment runs over two cohorts drawn from
# the formation pool: one where the order is a small share of a day's volume,
# one where it is more than a day's volume. Each name is run twice, once with no
# impact and once at the strongest coefficient, and the question asked of each
# pair is whether a profit without the impact charge is still a profit once the
# charge is applied. Both runs pay the same commission and slippage.


# %%
def flip_rate(
    prices: pl.DataFrame, candidates: pl.DataFrame, book_usd: float, n_sample: int
) -> dict:
    """Winner flips after formation-only sampling, with later data attrition reported."""
    take = min(n_sample, candidates.height)
    idx = sorted(rng.choice(candidates.height, size=take, replace=False).tolist())
    sample = candidates[idx]
    flips, n_valid, n_evaluable = 0, 0, 0
    returns: list[dict[str, Any]] = []
    high_coef = max(IMPACT_COEFFICIENTS)
    for row in sample.iter_rows(named=True):
        stock_data = prices.filter(pl.col("symbol") == row["symbol"]).sort("timestamp")
        if stock_data.height < LOOKBACK + 2:
            continue
        n_evaluable += 1
        no_impact = run_backtest(
            stock_data,
            row["symbol"],
            0.0,
            book_usd,
            row["ex_ante_volatility"],
        )
        high_impact = run_backtest(
            stock_data,
            row["symbol"],
            high_coef,
            book_usd,
            row["ex_ante_volatility"],
        )
        returns.append(
            {"symbol": row["symbol"], "no_impact": no_impact, "high_impact": high_impact}
        )
        if no_impact > 0:
            n_valid += 1
            flips += int(high_impact < 0)
    return {
        "sampled": take,
        "evaluable": n_evaluable,
        "attrition": take - n_evaluable,
        "n": n_valid,
        "flips": flips,
        "flip_rate": flips / n_valid if n_valid else float("nan"),
        "returns": pl.DataFrame(returns),
    }


# %%
pool_part = pool.with_columns((BOOK_USD / pl.col("adv_usd")).alias("order_participation"))
liquid_group = pool_part.filter(pl.col("order_participation") < LIQUID_MAX_PARTICIPATION)
thin_group = pool_part.filter(pl.col("order_participation") > THIN_MIN_PARTICIPATION)
assert not liquid_group.is_empty(), "liquid_group is empty after participation filter"
assert not thin_group.is_empty(), "thin_group is empty after participation filter"

liquid_flip = flip_rate(evaluation_prices, liquid_group, BOOK_USD, COHORT_SAMPLE)
thin_flip = flip_rate(evaluation_prices, thin_group, BOOK_USD, COHORT_SAMPLE)
display(
    Markdown(f"""
| Cohort | Sampled | Profitable without impact | Turned negative under impact | Share |
|---|---|---|---|---|
| Order under {LIQUID_MAX_PARTICIPATION:.0%} of daily volume | {liquid_flip["sampled"]} | {liquid_flip["n"]} | {liquid_flip["flips"]} | {liquid_flip["flip_rate"]:.0%} |
| Order over {THIN_MIN_PARTICIPATION:.0%} of daily volume | {thin_flip["sampled"]} | {thin_flip["n"]} | {thin_flip["flips"]} | {thin_flip["flip_rate"]:.0%} |

Names dropped for having too little data after {EVALUATION_START_DATE}:
{liquid_flip["attrition"]} of {liquid_flip["sampled"]} in the liquid cohort and
{thin_flip["attrition"]} of {thin_flip["sampled"]} in the thin one. That attrition is reported
after selection and does not feed back into it.
""")
)

# %% [markdown]
# ### The whole shift, not only the sign changes
#
# Counting sign changes throws away most of what the charge did. Each point
# below is one sampled name: its return with the impact model switched off on
# the horizontal axis, its return under the strongest charge on the vertical.
# Both axes already include commission and slippage, so the vertical distance
# below the diagonal is impact alone. A point on the diagonal was untouched by
# the charge, and the lower-right quadrant holds the names that go from a profit
# to a loss.

# %%
fig = go.Figure()
for label, result, color in [
    (f"Order under {LIQUID_MAX_PARTICIPATION:.0%} of daily volume", liquid_flip, COLORS["blue"]),
    (f"Order over {THIN_MIN_PARTICIPATION:.0%} of daily volume", thin_flip, COLORS["copper"]),
]:
    fig.add_trace(
        go.Scatter(
            x=(result["returns"]["no_impact"] * 100).to_list(),
            y=(result["returns"]["high_impact"] * 100).to_list(),
            mode="markers",
            name=label,
            marker=dict(color=color, size=7, opacity=0.75),
        )
    )
axis_span = [-100, 200]
fig.add_trace(
    go.Scatter(
        x=axis_span,
        y=axis_span,
        mode="lines",
        line=dict(color=COLORS["neutral"], width=1, dash="dash"),
        name="no cost",
        hoverinfo="skip",
    )
)
fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1, dash="dot"))
fig.add_vline(x=0, line=dict(color=COLORS["neutral"], width=1, dash="dot"))
fig.update_layout(
    title="Return with and without the impact charge, one point per sampled name",
    xaxis_title="Total return without impact (%)",
    yaxis_title="Total return under the strongest impact charge (%)",
    height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.show()

# %% [markdown]
# ## Visualize Impact Across the Liquidity Spectrum

# %% [markdown]
# ### Net Return vs Liquidity
#
# Left panel: net return for each stock against its order's participation rate
# (log scale), one line per impact level. Right panel: high-impact erosion
# against participation. The cross-section need not be monotonic because each
# name has a different trading path and turnover, even under one impact model.


# %%
IMPACT_STYLES = {
    0.0: dict(color=COLORS["blue"], dash="solid", symbol="circle", name="No Impact"),
    0.1: dict(color=COLORS["slate"], dash="dash", symbol="square", name="Low Impact"),
    0.3: dict(color=COLORS["amber"], dash="dot", symbol="diamond", name="Medium Impact"),
    0.6: dict(color=COLORS["negative"], dash="dashdot", symbol="x", name="High Impact"),
}


# %%
def add_impact_return_traces(fig: go.Figure, spectrum_results: pl.DataFrame) -> None:
    """Add net-return traces ordered by participation rate."""
    for impact_coef in sorted(spectrum_results["impact_coef"].unique().to_list()):
        subset = spectrum_results.filter(pl.col("impact_coef") == impact_coef).sort(
            "order_participation"
        )
        style = IMPACT_STYLES[impact_coef]
        fig.add_trace(
            go.Scatter(
                x=(subset["order_participation"] * 100).to_list(),
                y=(subset["total_return"] * 100).to_list(),
                mode="lines+markers",
                name=style["name"],
                line=dict(color=style["color"], dash=style["dash"], width=2),
                marker=dict(color=style["color"], symbol=style["symbol"], size=8),
                hovertemplate="Order: %{x:.1f}% of ADV<br>Net return: %{y:.1f}%",
            ),
            row=1,
            col=1,
        )


# %%
def add_erosion_trace(fig: go.Figure, summary: pl.DataFrame) -> None:
    """Add the high-impact erosion trace to the right panel."""
    fig.add_trace(
        go.Scatter(
            x=(summary["order_participation"] * 100).to_list(),
            y=(summary["erosion_high"] * 100).to_list(),
            mode="lines+markers",
            line=dict(color=COLORS["negative"], width=2),
            marker=dict(color=COLORS["negative"], symbol="circle", size=8),
            showlegend=False,
            hovertemplate="Order: %{x:.1f}% of ADV<br>Erosion: %{y:.1f} pp",
        ),
        row=1,
        col=2,
    )


# %% [markdown]
# Both horizontal axes are logarithmic. The five names are chosen from
# percentiles of daily volume, so their participation rates differ by orders of
# magnitude and a linear axis would collapse the liquid end onto the origin.


# %%
def plot_impact_spectrum(spectrum_results: pl.DataFrame, summary: pl.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Net Return vs Order Participation",
            "Impact Erosion vs Order Participation",
        ),
        horizontal_spacing=0.13,
    )
    add_impact_return_traces(fig, spectrum_results)
    add_erosion_trace(fig, summary)
    fig.update_xaxes(title_text="Order as % of Daily Volume (log)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Order as % of Daily Volume (log)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Net Total Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="High-Impact Erosion (pp)", row=1, col=2)
    fig.add_hline(
        y=0,
        line=dict(color=COLORS["neutral"], width=1, dash="dot"),
        row=1,
        col=1,
    )
    fig.update_layout(
        title="Net return and impact erosion against order participation",
        height=420,
        width=1000,
    )
    return fig


# %%
fig = plot_impact_spectrum(spectrum_results, summary)
fig.show()

# %% [markdown]
# ## Key takeaways

# %%
most_liquid = summary.sort("order_participation").row(0, named=True)
thinnest = summary.sort("order_participation").row(-1, named=True)
high_coefficient = str(max(IMPACT_COEFFICIENTS))

display(
    Markdown(f"""
**A dollar order has no size until you say what it is trading.** The same
{BOOK_USD:,.0f} USD order is {most_liquid["order_participation"]:.1%} of a day's volume in
`{most_liquid["symbol"]}` and {thinnest["order_participation"]:.0%} of it in
`{thinnest["symbol"]}`. Run through the same strategy over the same period, the strongest
impact charge takes {most_liquid["erosion_high"] * 100:.1f} percentage points off the first
name's total return and {thinnest["erosion_high"] * 100:.1f} off the second. The strategy and
the order are identical; what differs is the volume available to absorb the order, along with
each name's own volatility and price path, which the impact model also reads.

**The charge decides the sign, not only the magnitude.** Of the {liquid_flip["n"]}
liquid-cohort names profitable with the impact model switched off, {liquid_flip["flips"]} are
unprofitable with it on ({liquid_flip["flip_rate"]:.0%}); of the {thin_flip["n"]} thin-cohort
names, {thin_flip["flips"]} are ({thin_flip["flip_rate"]:.0%}). A backtest that omits impact is
therefore not uniformly optimistic: how much it overstates depends on where the order is sent.
The two cohorts differ in more than participation - volatility, turnover and realised path vary
with them - so this measures the gap between the cohorts as constructed rather than isolating
liquidity as its cause.

**Calibrate the cost model before the period it is applied to.** Each name's liquidity and
volatility, and the screen that put it in the pool at all, come from bars ending
{FORMATION_END_DATE}; every return reported comes from bars starting
{EVALUATION_START_DATE}. Calibrating the impact model on the evaluation window would make the
charge depend on the volume that turned out to be there, which is the one thing a trader
placing the order does not know.

**This is what motivates an execution policy rather than a schedule.** The charge here is
applied to a strategy that trades a whole book in one order because its signal changed. A
policy that spreads the order over the session, and adjusts as the day's volume arrives, is
paying a smaller participation rate for the same position - which is the problem
`02_optimal_execution_ppo` and `04_crypto_execution_rl` take up.

### Known limitations

- The impact model is a square-root law with a coefficient set by hand at four levels, not
  estimated from executions. The comparison across those levels is a sensitivity analysis, and
  none of the four is a claim about what this order would actually have cost.
- The strategy holds one name at a time with the whole book, which is what makes the
  participation rate large enough to see. A diversified book of the same size would spread the
  same dollars over many names and face a different problem.
- Volatility in the impact model is a single formation-window number per name, so the charge
  does not rise in the periods when the market was actually harder to trade.
- The cohorts are drawn once at a fixed seed from names that cleared a formation screen. They
  describe those two groups of {COHORT_SAMPLE} names, not the universe.

**Next**: chapter 22 leaves market microstructure for the text side of the research process,
and builds a retrieval pipeline over filings.
""")
)
