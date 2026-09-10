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
# # Information-Driven Bars: Formulas and Parameter Study
#
# **Chapter 3: Market Microstructure**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Two-part validation of imbalance-bar construction: (1) verify the AFML
# tick-imbalance formula matches the `ml4t.engineer.bars` library exactly on
# DataBento NVDA trades, and (2) sweep $\alpha$ and the target $E[T]$ across
# three families (alpha-based EWMA, fixed threshold, rolling window) to expose
# the parameter-instability issue that §3.4 warns about.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - State the tick-imbalance threshold formula
#   $\theta = \sum b_t$, $E[\theta_T] = E[T] \cdot |2P[b=1] - 1|$
#   and the volume-imbalance variant.
# - Compare an adaptive threshold, a fixed one and a rolling-window one on the same
#   trade stream, and recognise the two ways the adaptive scheme fails: a threshold
#   that runs away upward, and one that falls until every trade cuts a bar.
# - Diagnose which of those is happening from the bar count and the average bar size,
#   before looking at any downstream statistic.
# - Choose imbalance-bar parameters that yield well-behaved
#   Jarque-Bera / variance-ratio diagnostics.
#
# ## Book reference
#
# Section §3.4, *The Art of Sampling* — information-driven-bars subsection.
#
# ## Prerequisites
#
# - DataBento XNAS-ITCH MBO parquets at
#   `data/equities/market/microstructure/market_by_order/NVDA/`.

# %%
"""Information-Driven Bars: Formulas and Parameter Study — verifying AFML imbalance bar formulas and exploring parameter sensitivity."""

import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy import stats

# Import loader for MBO data
from data import load_mbo_data
from utils.style import show_plotly_with_alt

# Polars display configuration

# %% tags=["parameters"]
MAX_DAYS = 3

# %%
# Get file paths from the canonical loader (handles legacy/new path resolution)
data_files = load_mbo_data(symbols=["NVDA"], list_files=True)
DATABENTO_DIR = data_files[0].parent if data_files else None

# %% [markdown]
# ### Load Trade Data
# Extract and filter trade records from multi-day MBO parquet files.


# %%
def load_trades(data_dir: Path, max_days: int = 10) -> tuple[pl.DataFrame | None, list[str]]:
    """Load trade data from multiple days."""
    data_files = sorted(data_dir.glob("*.parquet"))[:max_days]
    if not data_files:
        return None, []

    all_trades = []
    dates = []

    for file_path in data_files:
        # Derive the date from the trailing 8-digit token so both file layouts
        # work: Download Center `xnas-itch-YYYYMMDD.mbo.dbn.parquet` and API
        # (`mbo_download.py`) `YYYYMMDD.parquet`.
        date_str = re.search(r"(\d{8})", file_path.stem).group(1)
        trade_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        dates.append(trade_date)

        df = pl.read_parquet(file_path)
        # Prefer exchange event time; fall back across the two file layouts.
        ts_col = next(c for c in ("timestamp", "ts_event", "ts_recv") if c in df.columns)
        df = df.select([ts_col, "action", "side", "price", "size"])
        df = df.with_columns(pl.col(ts_col).cast(pl.Datetime("ns")).alias("timestamp"))
        df = df.filter(pl.col("action") == "T")
        # Regular trading hours (09:30-16:00 America/New_York). Convert the UTC
        # instant to exchange-local time so the window is correct in both EDT and
        # EST rather than admitting an hour of pre-market (as a fixed UTC window does).
        _et = (
            pl.col("timestamp").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
        )
        df = df.filter(
            ((_et.dt.hour() > 9) | ((_et.dt.hour() == 9) & (_et.dt.minute() >= 30)))
            & (_et.dt.hour() < 16)
        )
        # Aggressor side for Trade (T) records: DataBento sets `side` to the trade
        # aggressor — B = buy-initiated (+1), A = sell-initiated (-1). (The
        # resting-order interpretation applies to Fill `F` records, not `T`.)
        df = df.with_columns(
            pl.when(pl.col("side") == "B")
            .then(1)
            .when(pl.col("side") == "A")
            .then(-1)
            .otherwise(0)
            .alias("side_num")
        )
        df = df.select(
            [
                "timestamp",
                pl.col("price"),
                pl.col("size").alias("volume"),
                pl.col("side_num").alias("side"),
            ]
        ).sort("timestamp")
        all_trades.append(df)

    return pl.concat(all_trades), dates


# %%
# Load data
trades, dates = load_trades(DATABENTO_DIR, max_days=MAX_DAYS)

if trades is None or len(trades) == 0:
    raise FileNotFoundError(
        "Missing DataBento MBO trade data for NVDA. "
        "Expected parquet files under data/equities/market_by_order/NVDA."
    )

trades = trades.filter(pl.col("side") != 0)
print(f"Loaded {len(trades):,} trades from {len(dates)} days")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Buy fraction: {(trades['side'] > 0).mean():.2%}")

# %% [markdown]
# ## 2. Formula Verification: Manual vs Library
#
# We implement tick imbalance bars manually to verify the library is correct.


# %%
def calculate_tick_imbalance_bars_manual(
    sides: np.ndarray,
    expected_t: float = 1000.0,
    alpha: float = 0.1,
    min_bars_warmup: int = 10,
) -> tuple[list[int], list[dict]]:
    """
    Manual AFML tick imbalance bars.

    θ = Σ b_t (cumulative signed ticks)
    E[θ_T] = E[T] × |2P[b=1] - 1|
    """
    n = len(sides)

    # Initialize from warmup (matches library)
    warmup_size = min(1000, n)
    p_buy = float(np.mean(sides[:warmup_size] > 0))

    bar_indices = []
    bar_info = []

    cumulative_theta = 0.0
    bar_tick_count = 0
    bar_buy_count = 0
    n_bars = 0

    for i in range(n):
        side = sides[i]
        is_buy = side > 0

        cumulative_theta += side
        bar_tick_count += 1
        if is_buy:
            bar_buy_count += 1

        # AFML threshold
        threshold = expected_t * abs(2 * p_buy - 1)

        if abs(cumulative_theta) >= threshold:
            bar_indices.append(i)
            bar_info.append(
                {
                    "bar": n_bars,
                    "ticks": bar_tick_count,
                    "theta": cumulative_theta,
                    "threshold": threshold,
                    "E[T]": expected_t,
                    "P[b=1]": p_buy,
                }
            )

            n_bars += 1

            # Update EWMA after warmup
            if n_bars > min_bars_warmup:
                expected_t = alpha * bar_tick_count + (1 - alpha) * expected_t
                bar_p_buy = bar_buy_count / bar_tick_count
                p_buy = alpha * bar_p_buy + (1 - alpha) * p_buy

            # Reset
            cumulative_theta = 0.0
            bar_tick_count = 0
            bar_buy_count = 0

    return bar_indices, bar_info


# %% [markdown]
# The verification below runs the sampler by hand on the trade signs, with a slow decay
# and a long warm-up, so that the threshold sequence can be inspected step by step
# rather than only through the bars it produced.

# %%
sides_arr = trades["side"].to_numpy()
VERIFY_ET = 1000
VERIFY_ALPHA = 0.001
VERIFY_WARMUP = 100

manual_indices, manual_info = calculate_tick_imbalance_bars_manual(
    sides_arr, expected_t=VERIFY_ET, alpha=VERIFY_ALPHA, min_bars_warmup=VERIFY_WARMUP
)
print(f"Manual TIB calculation: {len(manual_indices)} bars")

# %%
# Compare with library
from ml4t.engineer.bars import TickImbalanceBarSampler

sampler = TickImbalanceBarSampler(
    expected_ticks_per_bar=VERIFY_ET,
    alpha=VERIFY_ALPHA,
    min_bars_warmup=VERIFY_WARMUP,
)
library_bars = sampler.sample(trades)
print(f"Library TIB calculation: {len(library_bars)} bars")

# Verify match
if len(manual_indices) == len(library_bars):
    manual_thresholds = [d["threshold"] for d in manual_info]
    library_thresholds = library_bars["expected_imbalance"].to_list()
    max_diff = max(
        abs(m - lib) for m, lib in zip(manual_thresholds, library_thresholds, strict=False)
    )
    print(f"Max threshold difference: {max_diff:.6f}")
    print("[OK] Manual and library match!")
else:
    print("[FAIL] Bar counts differ")

# Show first few bars
print("\nFirst 5 bars:")
pl.DataFrame(manual_info[:5])

# %% [markdown]
# ## 3. Parameter Study Using Library
#
# Now we use the faster library implementation to study how E[T] affects properties.
#
# **Statistical Metrics Explained**:
# - **Jarque-Bera (JB)**: Tests normality. Lower = more normal (JB=0 is perfectly normal).
#   High JB indicates fat tails/skewness.
# - **Autocorrelation(1)**: Correlation of returns with 1-bar-lagged returns.
#   Should be ~0 for efficient markets.
# - **Variance Ratio(5)**: Var(5-bar returns) / (5 × Var(1-bar returns)).
#   Should be ~1 for random walk. >1 = momentum, <1 = mean reversion.

# %%
from ml4t.engineer.bars import ImbalanceBarSampler, TickImbalanceBarSampler


def compute_stats(bars: pl.DataFrame) -> dict:
    """Compute statistical properties of bar returns."""
    if len(bars) < 30:
        return {
            "n_bars": len(bars),
            "jarque_bera": np.nan,
            "autocorr_1": np.nan,
            "variance_ratio_5": np.nan,
        }

    returns = bars["close"].pct_change().drop_nulls().to_numpy()
    returns = returns[np.isfinite(returns)]
    if len(returns) < 10:
        return {
            "n_bars": len(bars),
            "jarque_bera": np.nan,
            "autocorr_1": np.nan,
            "variance_ratio_5": np.nan,
        }

    jb, _ = stats.jarque_bera(returns)
    ac = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else np.nan

    if len(returns) > 5:
        var_1 = np.var(returns)
        summed = np.array([np.sum(returns[i : i + 5]) for i in range(len(returns) - 4)])
        var_5 = np.var(summed) / 5
        vr = var_5 / var_1 if var_1 > 0 else np.nan
    else:
        vr = np.nan

    return {"n_bars": len(bars), "jarque_bera": jb, "autocorr_1": ac, "variance_ratio_5": vr}


# %% [markdown]
# The sweep below runs each sampler over a grid of target bar sizes. The tick and volume
# grids differ because their thresholds are in different units - trades against shares -
# and both bracket a bar size producing a few hundred to a few thousand bars from this
# session, which is the range where the downstream diagnostics have enough bars to be
# meaningful and few enough that each holds real information.
#
# The decay rate is fixed at the slow end across the whole sweep, so what varies is the
# target and not the sampler's stability.

# %%
TIB_ET = [500, 700, 1000, 1500, 2000, 3000]
VIB_ET = [2000, 5000, 10000, 20000, 50000]
ALPHA = 0.001  # slow decay; the feedback loop above is what this damps
WARMUP = 100  # Longer warmup

print("=" * 70)
print("TICK IMBALANCE BARS (TIBs) - alpha=0.001, warmup=100")
print("=" * 70)
tib_results = []
for et in TIB_ET:
    bars = TickImbalanceBarSampler(
        expected_ticks_per_bar=et, alpha=ALPHA, min_bars_warmup=WARMUP
    ).sample(trades)
    s = compute_stats(bars)
    s["expected_t"] = et
    tib_results.append(s)
    print(
        f"E[T]={et:>6}: {s['n_bars']:>5} bars, JB={s['jarque_bera']:>8.1f}, "
        f"AC(1)={s['autocorr_1']:>6.3f}, VR(5)={s['variance_ratio_5']:>5.2f}"
    )

print("\n" + "=" * 70)
print("VOLUME IMBALANCE BARS (VIBs) - alpha=0.001, warmup=100")
print("=" * 70)
vib_results = []
for et in VIB_ET:
    bars = ImbalanceBarSampler(
        expected_ticks_per_bar=et, alpha=ALPHA, min_bars_warmup=WARMUP
    ).sample(trades)
    s = compute_stats(bars)
    s["expected_t"] = et
    vib_results.append(s)
    print(
        f"E[T]={et:>6}: {s['n_bars']:>5} bars, JB={s['jarque_bera']:>8.1f}, "
        f"AC(1)={s['autocorr_1']:>6.3f}, VR(5)={s['variance_ratio_5']:>5.2f}"
    )

# %% [markdown]
# ## 4. Visualize Results

# %%
tib_df = pl.DataFrame(tib_results)
vib_df = pl.DataFrame(vib_results)

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Bar Count vs E[T]",
        "Jarque-Bera vs Bar Count",
        "Autocorrelation(1) vs Bar Count",
        "Variance Ratio(5) vs Bar Count",
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.12,
)

tib_color, vib_color = "#1e3a5f", "#c74b16"

# Panel 1: bar count vs E[T]
for name, color, df in [("TIB", tib_color, tib_df), ("VIB", vib_color, vib_df)]:
    fig.add_trace(
        go.Scatter(
            x=df["expected_t"].to_list(),
            y=df["n_bars"].to_list(),
            mode="lines+markers",
            name=name,
            line=dict(color=color),
        ),
        row=1,
        col=1,
    )
fig.update_xaxes(type="log", title_text="E[T]", row=1, col=1)
fig.update_yaxes(type="log", title_text="Number of Bars", row=1, col=1)

# Panel 2: Jarque-Bera vs bar count
for name, color, df in [("TIB", tib_color, tib_df), ("VIB", vib_color, vib_df)]:
    fig.add_trace(
        go.Scatter(
            x=df["n_bars"].to_list(),
            y=df["jarque_bera"].to_list(),
            mode="markers",
            marker=dict(color=color, size=10),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
fig.update_xaxes(type="log", title_text="Number of Bars", row=1, col=2)
fig.update_yaxes(type="log", title_text="Jarque-Bera", row=1, col=2)

# Panel 3: autocorrelation(1) vs bar count
for name, color, df in [("TIB", tib_color, tib_df), ("VIB", vib_color, vib_df)]:
    fig.add_trace(
        go.Scatter(
            x=df["n_bars"].to_list(),
            y=df["autocorr_1"].to_list(),
            mode="markers",
            marker=dict(color=color, size=10),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
fig.update_xaxes(type="log", title_text="Number of Bars", row=2, col=1)
fig.update_yaxes(title_text="Autocorrelation(1)", row=2, col=1)

# Panel 4: variance ratio(5) vs bar count
for name, color, df in [("TIB", tib_color, tib_df), ("VIB", vib_color, vib_df)]:
    fig.add_trace(
        go.Scatter(
            x=df["n_bars"].to_list(),
            y=df["variance_ratio_5"].to_list(),
            mode="markers",
            marker=dict(color=color, size=10),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=2)
fig.update_xaxes(type="log", title_text="Number of Bars", row=2, col=2)
fig.update_yaxes(title_text="Variance Ratio(5)", row=2, col=2)

fig.update_layout(
    title="Bar-count and return diagnostics for tick and volume imbalance bars",
    height=650,
    legend=dict(x=0.5, y=1.02, xanchor="center", orientation="h"),
)
show_plotly_with_alt(
    fig,
    "Four panels in a two-by-two grid comparing tick imbalance bars against volume imbalance bars across a grid of target bar sizes. Each panel plots one diagnostic against the number of bars produced, on a logarithmic horizontal axis, with one series per bar type.",
)

# %% [markdown]
# ## 5. Key Takeaways
#
# | Property | Tick imbalance bars | Volume imbalance bars |
# |----------|---------------------|-----------------------|
# | Accumulates | Trade signs, plus or minus one | Signed volume |
# | Threshold units | Trades | Shares |
# | Bars at the same target size | Many more | Far fewer |
#
# The threshold scales differ by orders of magnitude because they are in different
# units, so a value calibrated for one is meaningless for the other.
#
# ### Why the adaptive threshold can run away
#
# The threshold is set from a running estimate of two things: the expected bar size and
# the expected imbalance. Both are estimated from the bars the sampler has already cut,
# which is what makes the scheme self-referential.
#
# When order flow is persistently one-sided - which real data usually is - the estimated
# imbalance rises, the threshold rises with it, the next bar takes longer to fill, and
# the estimate rises again. The same loop runs in the other direction: an estimate that
# falls produces shorter bars, which lower the estimate further, until every trade cuts
# a bar. Both are the same feedback and the decay rate is what governs it.
#
# A slower decay and a longer warm-up damp the loop, at the cost of a sampler that is
# less adaptive. The comparison below runs three decay rates so the two failure
# directions and the working case can be seen against each other.
#
# ### Choosing a target bar size
#
# There is no optimal target. Choose it against:
# - Desired bar frequency (more bars = better normality, but more noise)
# - Trading horizon (intraday needs more bars than swing)
# - Signal strength vs statistical properties tradeoff

# %% [markdown]
# ## 6. Comparing Three Approaches: α-Based, Fixed, and Window-Based
#
# The ml4t-engineer library provides three different implementations:
#
# 1. **α-Based (AFML)**: Exponential decay for E[T] and P[b=1] - requires careful α tuning
# 2. **Fixed Threshold**: No adaptation - simplest and most predictable
# 3. **Window-Based**: Rolling window adaptation - bounded drift
#
# Let's compare them on the same data.

# %%
from ml4t.engineer.bars import (
    FixedTickImbalanceBarSampler,  # Fixed threshold
    TickImbalanceBarSampler,  # α-based
    WindowTickImbalanceBarSampler,  # Window-based
)

# Test parameters - match the handoff comparison
COMPARE_ET = 1000
COMPARE_THRESHOLD = 100  # For fixed


# Track how E[T] drifts for each method
def measure_et_drift(bars: pl.DataFrame) -> float:
    """Measure E[T] drift (last / first expected_t)."""
    if "expected_t" not in bars.columns or len(bars) == 0:
        return 1.0  # No drift for fixed or empty bars
    first = bars["expected_t"][0]
    last = bars["expected_t"][-1]
    return last / first if first > 0 else 1.0


# %%
print("=" * 70)
print("COMPARING THREE TICK IMBALANCE BAR APPROACHES")
print("=" * 70)

comparison = []

# 1. α-based with different alphas
for alpha in [0.001, 0.01, 0.1]:
    bars = TickImbalanceBarSampler(
        expected_ticks_per_bar=COMPARE_ET,
        alpha=alpha,
        min_bars_warmup=100,
    ).sample(trades)
    drift = measure_et_drift(bars)
    bar_stats = compute_stats(bars)
    comparison.append(
        {
            "method": f"α={alpha}",
            "n_bars": len(bars),
            "avg_ticks": len(trades) / len(bars) if len(bars) > 0 else 0,
            "et_drift": drift,
            "jb": bar_stats["jarque_bera"],
            "ac1": bar_stats["autocorr_1"],
        }
    )
    print(
        f"α-based α={alpha}: {len(bars):>4} bars, "
        f"avg_ticks={len(trades) / len(bars) if len(bars) > 0 else 0:>7.0f}, E[T] drift={drift:.2f}x"
    )

# %%
# 2. Fixed threshold
for thresh in [50, 100, 200]:
    bars = FixedTickImbalanceBarSampler(threshold=thresh).sample(trades)
    bar_stats = compute_stats(bars)
    comparison.append(
        {
            "method": f"fixed={thresh}",
            "n_bars": len(bars),
            "avg_ticks": len(trades) / len(bars) if len(bars) > 0 else 0,
            "et_drift": 1.0,
            "jb": bar_stats["jarque_bera"],
            "ac1": bar_stats["autocorr_1"],
        }
    )
    print(
        f"Fixed thresh={thresh}: {len(bars):>4} bars, "
        f"avg_ticks={len(trades) / len(bars) if len(bars) > 0 else 0:>7.0f}, E[T] drift=N/A"
    )

# %%
# 3. Window-based with different tick windows
for tick_win in [2000, 5000, 10000]:
    bars = WindowTickImbalanceBarSampler(
        initial_expected_t=COMPARE_ET,
        bar_window=10,
        tick_window=tick_win,
    ).sample(trades)
    drift = measure_et_drift(bars)
    bar_stats = compute_stats(bars)
    comparison.append(
        {
            "method": f"window={tick_win}",
            "n_bars": len(bars),
            "avg_ticks": len(trades) / len(bars) if len(bars) > 0 else 0,
            "et_drift": drift,
            "jb": bar_stats["jarque_bera"],
            "ac1": bar_stats["autocorr_1"],
        }
    )
    print(
        f"Window tick_win={tick_win}: {len(bars):>4} bars, "
        f"avg_ticks={len(trades) / len(bars) if len(bars) > 0 else 0:>7.0f}, E[T] drift={drift:.2f}x"
    )

# %%
# Summary table
compare_df = pl.DataFrame(comparison)
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
compare_df = compare_df.with_columns(
    [
        pl.col("avg_ticks").round(0).cast(pl.Int64),
        pl.col("et_drift").round(2),
        pl.col("jb").round(1),
        pl.col("ac1").round(3),
    ]
)
print(compare_df)

# %% [markdown]
# ## 7. Recommendations
#
# | Use case | Sampler | Why |
# |----------|---------|-----|
# | Production | `FixedTickImbalanceBarSampler` | The threshold cannot drift, so bar size is predictable |
# | Research | `TickImbalanceBarSampler` with a slow decay | Follows the textbook scheme while damping the feedback loop |
#
# The table above is how to read that recommendation. Two columns settle it before any
# downstream statistic is looked at: how many bars a sampler cut, and how many trades
# went into the average one. A sampler cutting hundreds of thousands of bars at one
# trade each has a threshold that fell to nothing; one cutting a few hundred bars of
# thousands of trades each has a threshold that ran away. Both are the feedback loop
# described above, and neither shows up as an error.
#
# Compare each sampler's average bar against the target it was given. A ratio near one
# is the scheme working; a ratio in the tens is drift, whichever direction it ran.

# %%
print("\n" + "=" * 70)
print("NOTEBOOK SUMMARY")
print("=" * 70)
print("\nTIBs:")
print(tib_df.select(["expected_t", "n_bars", "jarque_bera", "autocorr_1", "variance_ratio_5"]))
print("\nVIBs:")
print(vib_df.select(["expected_t", "n_bars", "jarque_bera", "autocorr_1", "variance_ratio_5"]))
print("\nThree-Approach Comparison:")
print(compare_df)
print("\nNotebook completed.")
