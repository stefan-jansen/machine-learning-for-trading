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
# # Corporate Actions: Adjusting for Splits and Dividends
#
# **Docker image**: `ml4t`
#
# **Purpose**: Demonstrate why unadjusted price series mis-represent returns
# across stock splits and cash dividends, derive the industry-standard backward
# adjustment, and validate the `ml4t.data.adjustments.apply_corporate_actions`
# implementation against the pre-adjusted Quandl WIKI series.
#
# **Learning objectives**:
#
# - Identify how splits and dividends break raw price continuity.
# - Apply the backward-adjustment formula for splits and dividends.
# - Use `apply_corporate_actions` to produce an adjusted OHLCV panel.
# - Validate adjusted prices against a trusted reference series.
# - Pick the right price representation for a given strategy horizon.
#
# **Book reference**: §2.3, "A due diligence framework for data sourcing", and §2.2,
# "The asset-class market data landscape".
#
# **Prerequisites**: Wiki Prices US-equities parquet on disk
# (`load_us_equities` resolves it via `ML4T_DATA_PATH`); `ml4t-data` library
# installed.

# %%
"""Corporate Actions — Adjusting for splits and dividends in historical price series."""

import inspect
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import Markdown, display
from ml4t.data.adjustments import apply_corporate_actions

from data import load_us_equities
from utils.paths import get_chapter_dir
from utils.style import COLORS, show_with_alt

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %% [markdown]
# ## 1. Why Corporate Actions Matter
#
# Corporate actions break the continuity of price series. Without proper
# adjustments:
#
# - **Stock splits** appear as massive price drops (e.g., a 2:1 split looks like
#   a $-50\%$ return).
# - **Dividends** cause ex-date price drops that distort return calculations.
# - **ML features** computed on unadjusted prices are therefore systematically
#   wrong.
#
# ### Types of corporate actions
#
# | Type | Description | Effect on price | Effect on shares |
# |------|-------------|-----------------|------------------|
# | Stock split        | e.g., 2:1 split           | Halved          | Doubled    |
# | Reverse split      | e.g., 1:4 split           | Quadrupled      | Quartered  |
# | Cash dividend      | Payment to shareholders   | Drops by amount | Unchanged  |
# | Stock dividend     | Additional shares issued  | Drops pro rata  | Increased  |

# %% [markdown]
# ## 2. Load the Wiki Prices Panel
#
# The Quandl WIKI dataset is well-suited for studying corporate actions because
# it ships raw prices alongside the original split and dividend events plus a
# pre-calculated adjusted series — so the adjusted column doubles as a
# reference to validate against.

# %%
raw_wiki = load_us_equities()
print(f"Wiki Prices loaded: {len(raw_wiki):,} rows")

# %% [markdown]
# Apple is a clean illustrative example: a long history carrying both kinds of action, with
# every split large enough to be unmistakable in the raw price.

# %%
dividend_col = "ex_dividend" if "ex_dividend" in raw_wiki.columns else "ex-dividend"
aapl = raw_wiki.filter(pl.col("symbol") == "AAPL").sort("timestamp")

n_splits = aapl.filter(pl.col("split_ratio") != 1.0).height
n_dividends = aapl.filter(pl.col(dividend_col) > 0).height
print(f"AAPL records: {len(aapl):,}  ({aapl['timestamp'].min()} to {aapl['timestamp'].max()})")
print(f"Corporate actions on record: {n_splits} splits, {n_dividends} cash dividends")

# %% [markdown]
# Stock splits in the AAPL history. `close` is the price as it traded that day; `adj_close`
# is that same price carried back through every split and dividend that came after it. The
# series is anchored at the final observation, where the two are equal, and gets smaller the
# further back you look.

# %%
splits = aapl.filter(pl.col("split_ratio") != 1.0)
splits.select(["timestamp", "close", "split_ratio", "adj_close"])

# %% [markdown]
# The first split's row is worth walking through, because it shows what the adjusted column
# is made of. Divide that day's traded close by the product of every split that came after
# it and you land near the adjusted value; the gap that remains is the dividend stream.

# %%
first_split = splits.row(0, named=True)
later_splits = splits.filter(pl.col("timestamp") > first_split["timestamp"])["split_ratio"]
split_product = float(later_splits.product())
split_only = first_split["close"] / split_product

display(
    Markdown(
        f"The {first_split['timestamp']:%B %Y} close of "
        f"**${first_split['close']:.2f}** divided by the "
        f"**{' x '.join(f'{r:.0f}' for r in later_splits)} = {split_product:.0f}** of the "
        f"{len(later_splits)} later splits is **${split_only:.2f}**. The adjusted close on "
        f"that row is **${first_split['adj_close']:.2f}**, and the difference is every "
        f"dividend Apple has paid since."
    )
)

# %% [markdown]
# First ten of 54 cash dividends. The dividend is in the dollars of its own
# ex-date, so it is not comparable to `adj_close` on the same row.

# %%
dividends = aapl.filter(pl.col(dividend_col) > 0)
dividends.select(["timestamp", "close", dividend_col, "adj_close"]).head(10)

# %% [markdown]
# ## 3. Three Ways to Represent Prices
#
# 1. **Raw (unadjusted)** — exactly as traded on the exchange. Best for
#    order-execution simulation; returns are distorted at action dates.
# 2. **Split-adjusted** — adjusts for stock splits only. Reasonable for
#    short-horizon trading where dividends are negligible.
# 3. **Total-return (split + dividend) adjusted** — the standard for ML
#    features, factor research, and long-horizon backtests.
#
# The choice of representation determines whether returns are economically
# meaningful or dominated by accounting artifacts.

# %% [markdown]
# ## 4. Raw vs Adjusted Cumulative Return
#
# Compute cumulative returns from both raw and adjusted close prices and plot
# them on a log scale; each split shows up as a discontinuity in the raw
# series.

# %%
dates_arr = aapl["timestamp"].to_numpy()
raw_close = aapl["close"].to_numpy()
adj_close = aapl["adj_close"].to_numpy()
split_dates = splits["timestamp"].to_numpy()
split_ratios = splits["split_ratio"].to_numpy()

raw_returns = np.diff(raw_close) / raw_close[:-1]
adj_returns = np.diff(adj_close) / adj_close[:-1]
raw_cumret = np.cumprod(1 + raw_returns)
adj_cumret = np.cumprod(1 + adj_returns)

# %%
fig, ax = plt.subplots(figsize=(12, 6), layout="tight")
ax.plot(
    dates_arr[1:], adj_cumret, label="Adjusted prices (correct)", color=COLORS["blue"], linewidth=2
)
ax.plot(
    dates_arr[1:], raw_cumret, label="Raw prices (wrong)", color=COLORS["negative"], linewidth=2
)

for d, ratio in zip(split_dates, split_ratios, strict=False):
    ax.axvline(d, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    idx = np.searchsorted(dates_arr[1:], d)
    if idx < len(adj_cumret):
        ax.annotate(
            f"{int(ratio)}:1 split",
            xy=(d, adj_cumret[idx]),
            xytext=(10, 20),
            textcoords="offset points",
            fontsize=9,
            color="gray",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
        )

ax.annotate(
    f"Raw: {raw_cumret[-1]:.0f}x\nAdjusted: {adj_cumret[-1]:.0f}x\n({adj_cumret[-1] / raw_cumret[-1]:.0f}x difference)",
    xy=(dates_arr[-1], (raw_cumret[-1] + adj_cumret[-1]) / 2),
    xytext=(-120, 0),
    textcoords="offset points",
    fontsize=11,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    ha="right",
)
ax.annotate(
    "Each split looks like a crash\nin raw prices, causing\ncumulative returns to diverge",
    xy=(split_dates[2], raw_cumret[np.searchsorted(dates_arr[1:], split_dates[2])]),
    xytext=(50, -30),
    textcoords="offset points",
    fontsize=9,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
    arrowprops=dict(arrowstyle="->", color="gray"),
)

ax.set_ylabel("Cumulative return (start = 1)")
ax.set_xlabel("Date")
ax.set_title("Every split reads as a crash in the raw series, and the gap never closes")
ax.legend(loc="upper left", fontsize=10)
ax.set_yscale("log")
ax.set_ylim(0.5, adj_cumret[-1] * 1.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}x" if x >= 1 else f"{x:.1f}x"))
ax.grid(True, alpha=0.3)
show_with_alt(
    fig,
    "Two cumulative return curves on a log scale from 1980 to 2018. The adjusted curve rises "
    "steadily to several hundred times its starting value; the raw curve steps down sharply "
    "at each of the four marked split dates and ends far below it.",
)

# %% [markdown]
# ### Figure 2.3 inputs
#
# The print version of Figure 2.3 is drawn in the book repository from the two frames written
# below, so the book build renders the curves this notebook computed rather than recomputing
# the adjustment.

# %%
ARTIFACTS_DIR = get_chapter_dir(2) / "output" / "book_figure_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
pl.DataFrame(
    {
        "timestamp": dates_arr[1:],
        "adj_cumret": adj_cumret,
        "raw_cumret": raw_cumret,
    }
).write_parquet(ARTIFACTS_DIR / "figure_2_3_corporate_actions_curves.parquet")
pl.DataFrame(
    {
        "split_date": split_dates,
        "split_ratio": split_ratios,
    }
).write_parquet(ARTIFACTS_DIR / "figure_2_3_corporate_actions_splits.parquet")
print(f"Wrote {len(adj_cumret):,} dates and {len(split_dates)} splits for the book figure")

# %% [markdown]
# The size of the divergence is the point. What the raw series is missing is the whole
# dividend stream plus the share-count effect of every split, and over a long holding period
# that is not a correction at the margin - it is most of the return.

# %%
print(f"Final cumulative return — raw prices:      {raw_cumret[-1]:.1f}x")
print(f"Final cumulative return — adjusted prices: {adj_cumret[-1]:.1f}x")
print(f"Ratio: {adj_cumret[-1] / raw_cumret[-1]:.0f}x difference")
print(
    f"Raw prices understate the cumulative return by {(adj_cumret[-1] / raw_cumret[-1] - 1) * 100:.0f}%"
)

# %% [markdown]
# ## 5. The `apply_corporate_actions` API
#
# The `apply_corporate_actions` function in `ml4t.data.adjustments` implements
# the backward-adjustment methodology shared by Quandl and most major data
# vendors. The function signature documents the convention it expects.

# %%
print(f"apply_corporate_actions{inspect.signature(apply_corporate_actions)}")

# %% [markdown]
# Apply the adjustment to AAPL. The function expects a `date` column and
# returns the input frame plus `adj_*` columns; we round-trip through the
# canonical `timestamp` schema.

# %%
adjusted = apply_corporate_actions(
    aapl.rename({"timestamp": "date"}),
    split_col="split_ratio",
    dividend_col=dividend_col,
    price_cols=["open", "high", "low", "close"],
    volume_col="volume",
).rename({"date": "timestamp"})

print("Adjusted columns:", [c for c in adjusted.columns if c.startswith("adj_")])

# %% [markdown]
# Side-by-side comparison at the IPO, three split dates, and the dataset end.
# `our_adj_close` comes from this notebook's invocation of
# `apply_corporate_actions`; `quandl_adj_close` is the pre-computed adjusted
# series shipped with the dataset.

# %%
example_dates = [
    date(1980, 12, 12),
    date(1987, 6, 16),
    date(2000, 6, 21),
    date(2014, 6, 9),
    date(2018, 3, 27),
]

comparison = (
    adjusted.filter(pl.col("timestamp").is_in(example_dates))
    .select(
        pl.col("timestamp"),
        pl.col("close").alias("raw_close"),
        pl.col("adj_close").alias("our_adj_close"),
        pl.col("split_ratio"),
    )
    .join(
        aapl.select(
            pl.col("timestamp"),
            pl.col("adj_close").alias("quandl_adj_close"),
        ),
        on="timestamp",
    )
    .sort("timestamp")
)

comparison

# %% [markdown]
# ## 6. Validation Against Quandl's Reference
#
# Compare every row of the locally-computed adjusted close to Quandl's
# pre-calculated value. With ~9,400 iterative backward updates, small
# floating-point error accumulates; the published convention tolerates a
# relative difference of about $5 \times 10^{-4}$.

# %%
our_adj = adjusted["adj_close"].to_numpy()
quandl_adj = aapl["adj_close"].to_numpy()
dates = adjusted["timestamp"].to_numpy()

absolute_diff = np.abs(our_adj - quandl_adj)
relative_diff = absolute_diff / quandl_adj * 100
tolerance_pct = 0.05  # 0.05 %

print(f"Total comparisons:        {len(our_adj):,}")
print(f"Max absolute difference:  ${absolute_diff.max():.6f}")
print(f"Max relative difference:  {relative_diff.max():.4f}%")
print(f"Mean relative difference: {relative_diff.mean():.6f}%")
print(f"Median relative difference: {np.median(relative_diff):.6f}%")
print()
if np.allclose(our_adj, quandl_adj, rtol=tolerance_pct / 100):
    print(f"VALIDATION PASSED — within {tolerance_pct:.4f}% relative tolerance.")
else:
    n_out = int((relative_diff > tolerance_pct).sum())
    print(f"VALIDATION FAILED — {n_out} rows outside {tolerance_pct:.4f}% tolerance.")

# %% [markdown]
# Visual check — the two series overlay on the log-scale price chart, and the
# rolling relative difference stays below the tolerance line.

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout="tight")

ax1 = axes[0]
ax1.plot(dates, our_adj, label="Our adjustment", alpha=0.85)
ax1.plot(dates, quandl_adj, label="Quandl adjustment", alpha=0.85, linestyle="--")
ax1.set_ylabel("Adjusted price ($)")
ax1.set_title("The two adjustments agree everywhere the eye can separate them")
ax1.set_yscale("log")
ax1.legend()

ax2 = axes[1]
ax2.plot(dates, relative_diff, color=COLORS["blue"], linewidth=1.2, alpha=0.9)
ax2.axhline(
    tolerance_pct,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Tolerance ({tolerance_pct:.4f}%)",
)
ax2.set_ylabel("Relative difference (%)")
ax2.set_xlabel("Date")
ax2.set_title("What is left is floating-point drift, not a difference in method")
ax2.set_ylim(0, max(relative_diff.max() * 1.1, tolerance_pct * 2))
ax2.legend()

show_with_alt(
    fig,
    "Above, two adjusted price series on a log scale that lie on top of each other over the "
    "whole history. Below, their relative difference in percent, which stays well under the "
    "dashed tolerance line across every year.",
)

# %% [markdown]
# ## 7. The Adjustment Formulas
#
# The industry-standard method is **backward adjustment**: start at the most
# recent date with `factor = 1` and walk backwards in time, multiplying the
# factor at each corporate action so future prices remain unchanged and
# historical prices are scaled down.
#
# **Split adjustment.** On the day before a stock split with ratio $R$, all
# prior prices must be divided by $R$:
#
# $$\text{factor}_{t-1} = \text{factor}_{t} \times \frac{1}{R_t}.$$
#
# **Dividend adjustment.** On ex-dividend date $t$, with closing price
# $P_{t-1}$ on the day before and dividend amount $D_t$, the multiplier for
# pre-ex-date prices is
#
# $$m_t = \frac{P_{t-1} - D_t}{P_{t-1}} = 1 - \frac{D_t}{P_{t-1}}, \qquad
#   \text{factor}_{t-1} = \text{factor}_{t} \times m_t.$$
#
# **Combined.** For any date $d$ the adjusted price is
#
# $$P^{\text{adj}}_d = P^{\text{raw}}_d \cdot \text{factor}_d.$$
#
# This backward-looking rule guarantees that returns computed on the adjusted
# series equal the total-return of holding the stock with dividends reinvested.

# %% [markdown]
# ### Numeric demonstration — split


# %%
def demonstrate_split_adjustment(
    price_before: float = 270.0, shares_before: int = 100, ratio: float = 3.0
) -> None:
    """Print a worked example of backward split adjustment."""
    value_before = price_before * shares_before
    price_after = price_before / ratio
    shares_after = shares_before * ratio
    value_after = price_after * shares_after

    print(
        f"Before {int(ratio)}:1 split — price ${price_before:.2f}, shares {shares_before:d}, value ${value_before:,.2f}"
    )
    print(
        f"After  {int(ratio)}:1 split — price ${price_after:.2f}, shares {int(shares_after):d}, value ${value_after:,.2f}"
    )
    print(
        f"Backward-adjusted historical price: ${price_before:.2f} / {ratio:.0f} = ${price_before / ratio:.2f}"
    )
    print(f"Adjusted series is now continuous: ${price_before / ratio:.2f} -> ${price_after:.2f}")


# %%
demonstrate_split_adjustment()

# %% [markdown]
# ### Numeric demonstration — dividend


# %%
def demonstrate_dividend_adjustment(price_before_ex: float = 100.0, dividend: float = 5.0) -> None:
    """Print a worked example of backward dividend adjustment."""
    price_on_ex = price_before_ex - dividend
    raw_return = (price_on_ex - price_before_ex) / price_before_ex
    multiplier = price_on_ex / price_before_ex
    adjusted_before = price_before_ex * multiplier
    adj_return = (price_on_ex - adjusted_before) / adjusted_before

    print(
        f"Day before ex: ${price_before_ex:.2f}    Dividend: ${dividend:.2f}    Ex-date: ${price_on_ex:.2f}"
    )
    print(f"Raw return on ex-date:      {raw_return:.1%}")
    print(f"Backward multiplier: {price_on_ex:.2f} / {price_before_ex:.2f} = {multiplier:.4f}")
    print(
        f"Adjusted pre-ex price: ${price_before_ex:.2f} × {multiplier:.4f} = ${adjusted_before:.2f}"
    )
    print(f"Adjusted return on ex-date: {adj_return:.1%}")


# %%
demonstrate_dividend_adjustment()

# %% [markdown]
# The convention treats the dividend as if it were reinvested in the stock at
# the close on the day before the ex-date. In real markets the ex-date price
# move is not exactly equal to the dividend; the adjustment is a *convention*
# for building continuous total-return series, not a description of price
# behavior.

# %% [markdown]
# ## 8. Practical Implications for ML Trading
#
# | Use case                      | Price type                | Reason |
# |-------------------------------|---------------------------|--------|
# | Feature engineering           | Total-return adjusted     | Returns must be economically meaningful |
# | Long-horizon backtest         | Total-return adjusted     | Captures dividend stream and split discontinuities |
# | Order-execution simulation    | Raw                       | Matches actual exchange prices |
# | Intraday trading (< 1 day)    | Split-adjusted             | Dividends negligible at high frequency |
# | Factor research               | Total-return adjusted     | Standard in academic literature |

# %% [markdown]
# ### Worked example — dividend impact on momentum
#
# A five-ETF rotational momentum sleeve illustrates how price-only momentum
# can mis-rank the cross-section when dividend yields differ materially. The
# numbers below are a stylized one-period example, not measured returns.

# %%
universe = pl.DataFrame(
    {
        "symbol": ["SPY", "QQQ", "TLT", "GLD", "EFA"],
        "price_return": [0.10, 0.15, -0.02, 0.08, 0.07],
        "dividend_yield": [0.015, 0.005, 0.025, 0.00, 0.025],
    }
).with_columns(total_return=pl.col("price_return") + pl.col("dividend_yield"))

ranked = universe.with_columns(
    rank_price=pl.col("price_return").rank(descending=True, method="ordinal"),
    rank_total=pl.col("total_return").rank(descending=True, method="ordinal"),
).with_columns(
    # Cast to a signed dtype before subtracting: ranks are u32, so a drop in
    # rank (e.g. GLD 3 -> 4) would underflow to a huge number otherwise.
    rank_change=pl.col("rank_price").cast(pl.Int32) - pl.col("rank_total").cast(pl.Int32)
)

ranked.select(
    [
        "symbol",
        "price_return",
        "dividend_yield",
        "total_return",
        "rank_price",
        "rank_total",
        "rank_change",
    ]
)

# %% [markdown]
# Across the five ETFs the dividend stream changes the ranking: EFA moves up
# one place when dividends are included and GLD moves down one place. TLT
# flips from a negative price return to a small positive total return but
# stays last. Price-only momentum signals would miss those re-orderings,
# which compound across many decisions over a multi-year horizon.

# %% [markdown]
# ## 9. Data-Source Conventions
#
# Different vendors use different adjustment conventions. The conventions the
# reader is most likely to encounter are summarised below; always validate
# against a known split or dividend date before assuming what a vendor's
# `close` column actually contains.
#
# | Provider           | "Close" column        | "Adj Close" column     |
# |--------------------|-----------------------|------------------------|
# | Quandl WIKI        | Truly unadjusted       | Split + dividend adjusted |
# | Yahoo Finance      | May be split-adjusted  | Split + dividend adjusted |
# | Bloomberg          | Unadjusted             | Configurable             |
# | Binance (crypto)   | Unadjusted             | n/a (no splits)          |
#
# Yahoo Finance behaviour in particular varies by interface (web UI,
# `yfinance` library, REST API). The safe procedure when integrating a new
# source is to pull a known split date — for AAPL, 2014-06-09 (7:1) is the
# canonical test case — and check whether the pre-split price has been
# divided by the split ratio.

# %% [markdown]
# ## Key takeaways
#
# - **A corporate action is not a return, and a raw price series cannot tell the difference.**
#   Compute returns from a raw close and every split enters as a loss of the split ratio.
#   Over a long holding period those artifacts dominate whatever the strategy was measuring.
# - **Backward adjustment anchors at the present.** The factor starts at one on the most
#   recent date and walks backwards, so today's price is untouched and history is scaled to
#   match it. That is why an adjusted price from decades ago is not a price anyone paid, and
#   why a vendor restating history changes every adjusted value before it.
# - **Validate an adjustment against a reference before trusting it.** This panel ships both
#   the raw prices and a pre-computed adjusted series, so the local implementation can be
#   checked row by row rather than assumed. Where no reference ships, one known split date is
#   enough to tell you which convention a vendor's `close` column follows.
# - **A tolerance is a claim about accumulated error, not about correctness.** The recursion
#   here runs once per trading day, so the two series drift apart by floating-point noise that
#   grows with the length of the history. Set the tolerance from that mechanism and say so;
#   a check that passes at any tolerance you happen to pick is not a check.
# - **Match the price representation to the decision.** Returns and features come from the
#   total-return series; a fill price comes from the raw one. The distinction stops mattering
#   only at horizons short enough for the dividend to be negligible, which is a judgement about
#   the holding period rather than about the data.
#
# **Known limitations.** The adjustment is a convention rather than a description of behaviour:
# it treats the dividend as reinvested at the previous close, and real ex-date moves differ from
# the dividend amount. Only one symbol is validated here, and the check does not hold panel-wide -
# `15_survivorship_bias_detection` finds where it fails. The momentum illustration is a stylized
# one-period example built from literals, not a measured result.
#
# **Next**: `03_etfs_eda` profiles the ETF universe used by the rotational strategy.
# **Book reference**: §2.3 and §2.2.
