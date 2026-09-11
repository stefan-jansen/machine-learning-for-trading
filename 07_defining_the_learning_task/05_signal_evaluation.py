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
# # Signal Evaluation: IC, Quantiles, and Spreads
#
# **Chapter 7: Defining the Learning Task**
# **Section Reference**: 7.3 - Feature and Label Evaluation as Triage
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# This notebook demonstrates **single-factor evaluation** using Information Coefficient
# (IC) analysis, quantile returns, and spread metrics. We answer: "Is this factor
# predictive in the cross-section, and what horizon does it live on?"
#
# ## Learning Objectives
#
# 1. Compute cross-sectional IC and understand its time series properties
# 2. Interpret IC, ICIR, and HAC-adjusted significance
# 3. Analyze quantile returns, spread, and the monotonicity of the quantile ladder
# 4. Understand horizon comparison with proper overlap warnings
# 5. Measure turnover and signal half-life
#
# ## Data Policy
#
# All examples use **real ETF data** from the case study store.
# NO synthetic data is used in this notebook.
#
# ## Prerequisites
#
# - `02_preprocessing_pipeline` - for split-aware preprocessing concepts that
#   underpin fold-aware IC evaluation under **Fold-Aware Evaluation**.
# - `03_label_methods` - supplies the forward-return labels used as `y_true`.
# - Familiarity with rank correlations (Spearman) and walk-forward CV.

# %%
"""Signal Evaluation - IC analysis, quintile spreads, and classification diagnostics for alpha signals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.diagnostic.evaluation.binary_metrics import (
    binary_classification_report,
    wilson_score_interval,
)
from ml4t.diagnostic.metrics import cross_sectional_ic, pooled_ic
from ml4t.diagnostic.signal import analyze_signal
from plotly.subplots import make_subplots
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import (  # importing utils.style activates the ml4t Plotly template
    COLORS,
    show_plotly_with_alt,
)

# %% tags=["parameters"]
SEED = 42
OUTPUT_DIR = Path("07_defining_the_learning_task/output")
START_DATE = "2006-01-01"
MAX_SYMBOLS = 0
N_PERMUTATIONS = 1000
DECAY_HORIZONS = (1, 2, 3, 5, 7, 10, 15, 21, 42)
N_SPLITS = 8

# %%
set_global_seeds(SEED)


# %% [markdown]
# ## Data Contract
#
# Signal analysis requires two DataFrames with specific schemas:
#
# **Factor Panel**:
# - `timestamp`: Decision date
# - `symbol`: Asset identifier
# - `factor` or signal column(s): Factor value(s)
#
# **Prices Panel**:
# - `timestamp`: Same as factor panel
# - `symbol`: Same as factor panel
# - `close` or `price`: Closing price
#
# The `ml4t-diagnostic` library aligns these using ASOF joins to compute
# forward returns at each decision point.

# %%
# Load real ETF data
etfs = load_etfs()
print(f"ETF universe: {etfs['symbol'].n_unique()} symbols, {len(etfs):,} rows")
print(f"Date range: {etfs['timestamp'].min()} to {etfs['timestamp'].max()}")

# %% [markdown]
# ## Preparing Factor and Price Panels
#
# We compute a simple momentum factor (21-day return) and prepare the data
# in the format required by `analyze_signal()`.

# %% [markdown]
# The factor below is a 21-day return, used here as a teaching example. Production factors
# come from the Chapter 8 feature pipelines.

# %%
if START_DATE != "2006-01-01":
    etfs = etfs.filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
if MAX_SYMBOLS > 0:
    keep = sorted(etfs["symbol"].unique().to_list())[:MAX_SYMBOLS]
    etfs = etfs.filter(pl.col("symbol").is_in(keep))

# Compute 21-day momentum per asset
factor_df = (
    etfs.sort(["symbol", "timestamp"])
    .with_columns(
        [(pl.col("close") / pl.col("close").shift(21).over("symbol") - 1).alias("factor")]
    )
    .filter(pl.col("factor").is_not_null())
    .select(["timestamp", "symbol", "factor"])
)

# Price panel
prices_df = etfs.select(["timestamp", "symbol", "close"]).rename({"close": "price"})

print(f"Factor panel: {factor_df.shape}")
print(f"Price panel: {prices_df.shape}")
print("Factor summary:")
display(factor_df.select("factor").describe())

# %%
# Forward returns, reused by the fold-aware and binary sections below.
eval_df = (
    factor_df.join(prices_df, on=["timestamp", "symbol"], how="inner")
    .sort(["symbol", "timestamp"])
    .with_columns(
        fwd_21d=(pl.col("price").shift(-21).over("symbol") / pl.col("price") - 1),
    )
    .filter(pl.col("fwd_21d").is_not_null())
)
print(f"\nEvaluation panel: {eval_df.shape} (factor + 21D forward returns)")

# %% [markdown]
# ### Correctness Screens
#
# Before evaluating predictive power, verify that the factor is usable under the
# stated protocol. The chapter's *Correctness screens* section prescribes four checks;
# we demonstrate coverage
# and staleness here. Timing/lag consistency and mask alignment become critical
# with fundamental or third-party data (Chapters 8-10) but are trivially satisfied
# for a price-derived momentum signal.

# %% [markdown]
# **Coverage** is the fraction of (date, asset) pairs carrying a non-null factor value,
# reported overall and per date.

# %%
all_pairs = prices_df.select("timestamp", "symbol").unique()
factor_pairs = factor_df.select("timestamp", "symbol").unique()
coverage = len(factor_pairs) / len(all_pairs)

# Per-date coverage (assets with factor / total assets)
daily_coverage = (
    all_pairs.join(factor_df, on=["timestamp", "symbol"], how="left")
    .group_by("timestamp")
    .agg(
        total=pl.len(),
        has_factor=pl.col("factor").is_not_null().sum(),
    )
    .with_columns(coverage=(pl.col("has_factor") / pl.col("total")))
    .sort("timestamp")
)

print("=== Correctness Screen: Coverage ===\n")
print(f"Overall coverage: {coverage:.1%}")
print(
    f"Per-date coverage - min: {daily_coverage['coverage'].min():.1%}, "
    f"median: {daily_coverage['coverage'].median():.1%}, "
    f"max: {daily_coverage['coverage'].max():.1%}"
)
print("\nNote: Coverage < 100% is expected - momentum requires 21 days of history,")
print("so new listings lack factor values during their first 21 trading days.")

# %% [markdown]
# **Staleness** asks whether the factor updates as often as its definition implies. A
# price-derived momentum signal should take a new value every session, so a change rate
# materially below one points at a data gap rather than at the signal.

# %%
staleness = (
    factor_df.sort(["symbol", "timestamp"])
    .with_columns(
        factor_change=(pl.col("factor") != pl.col("factor").shift(1).over("symbol")).cast(pl.Int32)
    )
    .group_by("symbol")
    .agg(
        n_obs=pl.len(),
        n_changes=pl.col("factor_change").sum(),
    )
    .with_columns(change_rate=(pl.col("n_changes") / pl.col("n_obs")))
)

median_change_rate = staleness["change_rate"].median()
min_change_rate = staleness["change_rate"].min()

print("\n=== Correctness Screen: Staleness ===\n")
print(f"Median daily change rate: {median_change_rate:.1%}")
print(f"Min change rate (worst asset): {min_change_rate:.1%}")
if median_change_rate > 0.9:
    print("[PASS] Factor updates daily as expected for a price-derived signal.")
else:
    print("[WARNING] Some assets show stale factor values - investigate data gaps.")

# %% [markdown]
# ## Information Coefficient (IC) Analysis
#
# IC measures the **cross-sectional** rank correlation between signals and forward returns:
#
# $$IC_t = \text{Spearman}(\text{signal}_{t}, \text{return}_{t \to t+h})$$
#
# Where the correlation is computed across assets at each time $t$.

# %%
# Run signal analysis
PERIODS = (1, 5, 21)  # Forward return horizons (days)
QUANTILES = 5  # Quintile analysis

result = analyze_signal(
    factor_df,
    prices_df,
    periods=PERIODS,
    quantiles=QUANTILES,
    ic_method="spearman",  # Rank correlation (robust to outliers)
    date_col="timestamp",
    asset_col="symbol",
)

print("Signal analysis complete")
print(f"Assets: {result.n_assets}, Dates: {result.n_dates}")

# %%
# Information Coefficient by Horizon
ic_rows = []
for period in PERIODS:
    period_key = f"{period}D"
    ic_mean = result.ic.get(period_key, float("nan"))
    icir = result.ic_ir.get(period_key, float("nan"))
    t_stat = result.ic_t_stat.get(period_key, float("nan"))
    p_value = result.ic_p_value.get(period_key, float("nan"))
    sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
    ic_rows.append(
        {
            "horizon": period_key,
            "mean_ic": round(ic_mean, 4),
            "icir": round(icir, 3),
            "t_stat": round(t_stat, 2),
            "p_value": round(p_value, 4),
            "sig": sig,
        }
    )

ic_summary = pl.DataFrame(ic_rows)
ic_summary

# %% [markdown]
# ### Two IC Conventions: Pooled vs Cross-Sectional
#
# The library exposes both, and the distinction matters for ranking strategies:
#
# - **`pooled_ic`** - one global Spearman correlation across **all (date, asset)**
#   observations.  Conflates *which days were good* with *which assets ranked
#   correctly within a day*; sensitive to time-series mean shifts in returns.
# - **`cross_sectional_ic`** - Spearman per date, then mean across dates.  Measures
#   only the daily ranking skill that a long-short strategy can monetise, and exposes
#   IC IR / t-stat / p-value on the per-date series.
#
# Chapter 14 standardises on the cross-sectional convention. The two can disagree
# materially on the same data - pooled may inflate or deflate magnitude depending
# on the regime structure of returns.

# %%
# Build a (date, symbol, y_pred, y_true) frame from eval_df at horizon 21D
ic_frame = eval_df.select(
    [
        pl.col("timestamp").alias("date"),
        pl.col("symbol"),
        pl.col("factor").alias("y_pred"),
        pl.col("fwd_21d").alias("y_true"),
    ]
).drop_nulls()

ic_pooled = pooled_ic(ic_frame["y_pred"], ic_frame["y_true"], method="spearman")
ic_xs = cross_sectional_ic(
    ic_frame,
    ic_frame,
    pred_col="y_pred",
    ret_col="y_true",
    date_col="date",
    entity_col="symbol",
    method="spearman",
    min_obs=5,
)

print(f"pooled_ic           : {ic_pooled:.4f}  (one global Spearman)")
print(
    f"cross_sectional_ic  : {ic_xs['ic_mean']:.4f}  "
    f"(mean of {ic_xs['n_periods']} daily Spearmans; "
    f"t={ic_xs['ic_t']:.2f}, p={ic_xs['p_value']:.4f})"
)

# %% [markdown]
# ### Interpreting an IC magnitude
#
# The right anchor for interpreting a mean IC is not the headline value but the standard
# error of that mean, which is set by the number of periods $T$ in the daily-IC series and
# by the dispersion of that series:
#
# $$\text{SE}(\bar{\text{IC}}) \approx \frac{\sigma_{\text{IC}}}{\sqrt{T}}$$
#
# A single point estimate therefore carries very different evidence depending on
# $\sigma_{\text{IC}}$ and $T$. The cell below works the confidence interval for one
# fixed $\bar{\text{IC}}$ under three sample-size and dispersion combinations, so the
# arithmetic can be checked and re-run rather than read.

# %% tags=["results"]
IC_SCENARIO_MEAN = 0.02  # one headline IC, held fixed across the three scenarios
IC_SCENARIOS = (
    ("ten years, tight daily IC", 2_500, 0.05),
    ("ten years, wide daily IC", 2_500, 0.30),
    ("one year, wide daily IC", 250, 0.30),
)
Z_95 = 1.96

print(f"Mean IC held at {IC_SCENARIO_MEAN} in every row; only T and sigma move.\n")
print(f"{'scenario':<28}{'T':>7}{'sigma':>8}{'SE':>9}{'95% CI':>20}")
print("-" * 72)
for label, T, sigma in IC_SCENARIOS:
    se = sigma / np.sqrt(T)
    lo, hi = IC_SCENARIO_MEAN - Z_95 * se, IC_SCENARIO_MEAN + Z_95 * se
    print(f"{label:<28}{T:>7,}{sigma:>8.2f}{se:>9.4f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}")

# %% [markdown]
# The same headline number is comfortably above zero in the first row, above zero but
# uninformative about the signal's tail behaviour in the second, and indistinguishable
# from zero in the third. Reporting a daily-mean IC therefore requires the CI (or the
# $t$-statistic) alongside, and ideally the dispersion of the daily series as well.
#
# The **ICIR** $= \bar{\text{IC}} / \sigma_{\text{IC}}$ is the signal-level analog of an
# information ratio: $t \approx \text{ICIR} \times \sqrt{T}$ for serially uncorrelated
# daily IC, and a HAC-adjusted $t$ for the realistic correlated case.
#
# The bands printed next are typical magnitudes from the equity-factor literature on
# multi-year daily-rebalanced cross-sectional studies. They are starting points for the SE
# calculation above, not standalone readings, and they live in a declared table so the
# code below can score against the same numbers the prose refers to.

# %% tags=["results"]
IC_BANDS = (
    (
        0.02,
        "at or below the daily-IC noise floor on multi-year samples; the SE alone often spans it",
    ),
    (0.04, "detectable on 5-to-10-year samples at the dispersions seen in published studies"),
    (
        0.06,
        "comparable to documented equity-factor effects such as momentum or short-term reversal",
    ),
    (
        0.10,
        "above most factor-zoo benchmarks; the question is whether it survives expanding-window evaluation",
    ),
    (
        float("inf"),
        "outside the published academic range; the prior is leakage or label corruption until ruled out",
    ),
)


def ic_band(ic: float) -> str:
    """Return the literature band an absolute mean IC falls into."""
    for upper, description in IC_BANDS:
        if abs(ic) < upper:
            return description
    return IC_BANDS[-1][1]


print("Reference bands for |mean IC| (equity-factor literature):")
lower = 0.0
for upper, description in IC_BANDS:
    edge = "and above" if upper == float("inf") else f"to {upper:.2f}"
    print(f"  {lower:.2f} {edge:<12} {description}")
    lower = upper

print(f"\nThis factor's 21-day mean IC: {result.ic.get('21D', float('nan')):.4f}")
print(f"  falls in: {ic_band(result.ic.get('21D', float('nan')))}")

# %% [markdown]
# The daily IC series is plotted at low opacity behind two rolling means. At this sample
# size the raw series is a solid band and carries no readable structure on its own; what a
# reader can act on is whether the smoothed level drifts away from zero, and the two
# windows show whether an apparent drift is still there under a longer average.

# %%
fig = make_subplots(
    rows=1, cols=2, subplot_titles=["Daily IC, 21-day horizon", "Distribution of daily IC"]
)

# Get 21D IC series
ic_21d = result.ic_series.get("21D", [])
if ic_21d:
    # Raw series as context only: 5,000 overlapping daily values plot as a solid band.
    fig.add_trace(
        go.Scatter(
            y=ic_21d,
            mode="lines",
            name="Daily IC",
            line=dict(color=COLORS["neutral"], width=0.4),
            opacity=0.25,
        ),
        row=1,
        col=1,
    )

    ic_series = pl.Series(ic_21d)
    for window, color, width in ((63, COLORS["amber"], 1.5), (252, COLORS["blue"], 2.0)):
        fig.add_trace(
            go.Scatter(
                y=ic_series.rolling_mean(window_size=window).to_list(),
                mode="lines",
                name=f"{window}-day mean",
                line=dict(color=color, width=width),
            ),
            row=1,
            col=1,
        )

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], row=1, col=1)

    # Distribution
    fig.add_trace(
        go.Histogram(x=ic_21d, nbinsx=30, name="IC Distribution", marker_color=COLORS["blue"]),
        row=1,
        col=2,
    )

    mean_ic = np.mean(ic_21d)
    fig.add_vline(x=mean_ic, line_dash="dash", line_color=COLORS["negative"], row=1, col=2)

fig.update_xaxes(title_text="Trading day (index, chronological)", row=1, col=1)
fig.update_yaxes(title_text="IC (Spearman)", row=1, col=1)
fig.update_xaxes(title_text="IC", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_layout(
    height=350, showlegend=True, title_text="Daily cross-sectional IC of the momentum factor"
)
show_plotly_with_alt(
    fig,
    alt=(
        "Two panels. The left panel plots the daily cross-sectional IC against a "
        "chronological index of roughly five thousand trading days. The raw series is a "
        "faint grey band filling the range from about minus 0.9 to plus 0.9 with no "
        "visible trend, and two rolling means are drawn over it. The shorter amber mean "
        "swings within roughly a fifth of a correlation unit either side of zero; the "
        "longer navy mean is flatter still and stays inside about half that. Neither "
        "settles at a level away from zero anywhere in the sample. The right panel is a "
        "histogram of the same "
        "daily values: a broad, roughly symmetric bell centred on zero and running out to "
        "about plus and minus 0.8, with a dashed red line marking the mean sitting "
        "essentially on the zero gridline."
    ),
)

# %% [markdown]
# ### Publication Figure Artifact
#
# The book IC time-series figure reads a compact NumPy artifact so formatting
# changes do not reload the ETF panel or recompute daily cross-sectional ICs.

# %% [markdown]
# The pairs below are sorted on the native timestamp rather than on its string form.
# Lexicographic sorting of stringified dates is correct only for zero-padded ISO output
# and would silently reorder the series under any other rendering.

# %%
ic_pairs: list[tuple[object, float]] = []
for date_df in eval_df.partition_by("timestamp"):
    if len(date_df) < 20:
        continue
    corr, _ = spearmanr(date_df["factor"].to_numpy(), date_df["fwd_21d"].to_numpy())
    if not np.isnan(corr):
        ic_pairs.append((date_df["timestamp"][0], float(corr)))

ic_pairs.sort(key=lambda t: t[0])
ic_dates_for_figure = [str(ts) for ts, _ in ic_pairs]
ic_values_arr = np.array([v for _, v in ic_pairs])

rolling_window = 63
rolling_ic_for_figure = np.full_like(ic_values_arr, np.nan)
for i in range(rolling_window, len(ic_values_arr)):
    rolling_ic_for_figure[i] = np.mean(ic_values_arr[i - rolling_window : i])

min_train = int(len(ic_dates_for_figure) * 0.3)
test_size = 252
fold_boundaries_for_figure = ic_dates_for_figure[min_train::test_size]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure_7_3_artifact = OUTPUT_DIR / "figure_7_3_ic_time_series_with_folds.npz"
np.savez(
    figure_7_3_artifact,
    ic_dates=np.array(ic_dates_for_figure),
    ic_values=ic_values_arr,
    rolling_ic=rolling_ic_for_figure,
    fold_boundaries=np.array(fold_boundaries_for_figure),
)
print(f"Wrote publication figure artifact: {figure_7_3_artifact}")

# %% [markdown]
# ## Quantile Analysis
#
# Examine returns by signal quantile to assess **monotonicity** (do higher signal
# values lead to higher returns?) and **spread** (what's the return difference
# between top and bottom quantiles?).

# %%
# Quantile returns
print("\n=== Mean Returns by Quantile ===\n")

for period in PERIODS:
    period_key = f"{period}D"
    quantile_rets = result.quantile_returns.get(period_key, {})

    if quantile_rets:
        print(f"{period_key} Forward Returns:")
        for quantile in sorted(quantile_rets.keys()):
            ret = quantile_rets[quantile]
            bar = "█" * int(abs(ret) * 500)  # Simple text bar
            sign = "+" if ret >= 0 else ""
            print(f"  Q{quantile}: {sign}{ret:.4%} {bar}")
        print()

# %%
# Visualize quantile returns
fig = make_subplots(
    rows=1,
    cols=len(PERIODS),
    subplot_titles=[f"{p}D Forward Returns" for p in PERIODS],
    horizontal_spacing=0.12,
)

for i, period in enumerate(PERIODS, 1):
    period_key = f"{period}D"
    period_returns = result.quantile_returns.get(period_key, {})

    if period_returns:
        quantiles = [f"Q{q}" for q in sorted(period_returns.keys())]
        returns = [period_returns[q] for q in sorted(period_returns.keys())]

        fig.add_trace(
            go.Bar(
                x=quantiles,
                y=returns,
                name=period_key,
                showlegend=False,
                marker_color=[COLORS["amber"] if v < 0 else COLORS["blue"] for v in returns],
            ),
            row=1,
            col=i,
        )

for i in range(1, len(PERIODS) + 1):
    fig.update_xaxes(title_text="Quantile", row=1, col=i)
    fig.update_yaxes(tickformat=".2%", row=1, col=i)
# Label the shared y-axis only on the first panel to avoid the title overlapping
# the neighbouring panel's bars; each panel keeps its own (per-horizon) scale.
fig.update_yaxes(title_text="Mean forward return", row=1, col=1)
fig.update_layout(title="Mean forward return by signal quantile, three horizons", height=350)
show_plotly_with_alt(
    fig,
    alt=(
        "Three bar panels, one per forward-return horizon, each showing the mean forward "
        "return of the five signal quantiles from Q1 to Q5. Every bar in every panel is "
        "positive, so all five quantiles earned money over the sample. The ordering does "
        "not follow the signal: in the one-day panel Q1 is the tallest bar and Q4 the "
        "shortest; in the five-day panel the bars fall from Q1 to Q4 and tick up at Q5; in "
        "the twenty-one-day panel the five bars are nearly level with Q5 the lowest. "
        "Each panel carries its own vertical scale, so the heights are comparable within a "
        "panel and not across panels."
    ),
)

# %% [markdown]
# The bars do not rise with the quantile in any of the three panels, and in the shortest
# horizon they fall - the bottom quantile of the signal earns the most. A long-short book
# built the obvious way round would be short the better-performing leg. The spread and
# monotonicity numbers below put figures on that, and the sign is the part to read first.

# %%
# Spread and monotonicity analysis
spread_rows = []
for period in PERIODS:
    period_key = f"{period}D"
    spread = result.spread.get(period_key, float("nan"))
    t_stat = result.spread_t_stat.get(period_key, float("nan"))
    mono = result.monotonicity.get(period_key, float("nan"))
    spread_rows.append(
        {
            "horizon": period_key,
            "spread_pct": round(spread * 100, 4),
            "t_stat": round(t_stat, 2),
            "monotonicity_rho": round(mono, 3),
        }
    )

spread_summary = pl.DataFrame(spread_rows)
spread_summary

# %% [markdown]
# ### Monotonicity Interpretation
#
# `analyze_signal` reports monotonicity as the **Spearman rank correlation between the
# quantile ranks and their mean returns**, so it runs from $-1$ to $+1$: $+1$ is a
# perfectly increasing ladder of quantile returns, $-1$ a perfectly decreasing one, and
# $0$ no rank association at all. It is a correlation, not a percentage, and is reported
# as such below.
#
# That is not the same statistic as the one the fold section computes further down, which
# is the *fraction of adjacent quantile steps that move upward* and runs from $0$ to $1$.
# The two answer the same question differently and do not share a scale; where both appear
# they are named apart.
#
# The bands below score the **strength** of the ordering, so they read the magnitude. The
# sign is a separate fact and the more important one here: a strongly ordered signal
# pointing the wrong way is not a weak signal, it is an inverted one, and the two call for
# opposite actions. The bands are kept in a declared table so the reading the prose gives
# is the one the code applies.

# %% tags=["results"]
MONOTONICITY_BANDS = (
    (0.60, "weak ordering; consider a non-linear model"),
    (0.80, "moderate ordering; may work as a long-short book"),
    (1.01, "strong, consistent ordering"),
)


def monotonicity_band(value: float) -> str:
    """Return the strength convention the magnitude of a monotonicity rho falls into."""
    for upper, description in MONOTONICITY_BANDS:
        if abs(value) < upper:
            return description
    return MONOTONICITY_BANDS[-1][1]


lower = 0.0
print("Reference bands for |rho| (ordering strength; the sign is read separately):")
for upper, description in MONOTONICITY_BANDS:
    print(f"  {lower:.2f} to {min(upper, 1.0):.2f}  {description}")
    lower = upper

print()
for period in PERIODS:
    mono = result.monotonicity.get(f"{period}D", float("nan"))
    direction = "as signalled" if mono > 0 else "inverted" if mono < 0 else "flat"
    print(f"  {period}D: {mono:+.2f}  {direction}, {monotonicity_band(mono)}")

# %% [markdown]
# ## Horizon Comparison
#
# Compare IC across different forward return horizons to identify the optimal
# holding period for the signal.

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=["Mean IC by horizon", "ICIR by horizon"])

horizons = list(PERIODS)
ics = [result.ic.get(f"{h}D", float("nan")) for h in horizons]
icirs = [result.ic_ir.get(f"{h}D", float("nan")) for h in horizons]

# IC plot (colorblind-safe blue/amber)
fig.add_trace(
    go.Bar(
        x=[f"{h}D" for h in horizons],
        y=ics,
        name="Mean IC",
        marker_color=[COLORS["blue"] if v > 0 else COLORS["amber"] for v in ics],
    ),
    row=1,
    col=1,
)

# ICIR plot
icir_colors = [
    COLORS["blue"] if v > 0.5 else COLORS["slate"] if v > 0 else COLORS["amber"] for v in icirs
]
fig.add_trace(
    go.Bar(x=[f"{h}D" for h in horizons], y=icirs, name="ICIR", marker_color=icir_colors),
    row=1,
    col=2,
)

fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"], row=1, col=2)

fig.update_xaxes(title_text="Horizon", row=1, col=1)
fig.update_yaxes(title_text="Mean IC", row=1, col=1)
fig.update_xaxes(title_text="Horizon", row=1, col=2)
fig.update_yaxes(title_text="ICIR", row=1, col=2)
fig.update_layout(
    height=350, showlegend=False, title_text="Mean IC and ICIR across the three evaluated horizons"
)
show_plotly_with_alt(
    fig,
    alt=(
        "Two bar panels over the same three horizons, one day, five days and twenty-one "
        "days. The left panel plots mean IC: the one-day and five-day bars hang below "
        "zero in amber, the five-day the longer of the two, and the twenty-one-day bar "
        "rises just above zero in navy. The whole vertical range spans less than one "
        "hundredth of a correlation unit. The right panel plots ICIR on an axis whose top "
        "is a dashed reference line; all three bars are so close to zero that they read as "
        "a flat line along the baseline, nowhere near that reference."
    ),
)

# %% [markdown]
# Read the two panels together. The left panel's bars change sign across horizons, which
# invites a story about reversal giving way to momentum; the right panel says the whole
# left panel is inside the noise, because an ICIR of essentially zero means the daily IC
# series has a standard deviation orders of magnitude larger than its mean. The sign
# pattern is not a finding at these magnitudes.

# %% [markdown]
# ### Overlapping Returns Warning
#
# When comparing IC across horizons, be aware that **longer horizons have overlapping
# forward returns**, which introduces autocorrelation in the IC series.
#
# For example, 21-day forward returns on consecutive days share 20 days of overlap.
# This means:
#
# 1. **IC series are autocorrelated** at longer horizons
# 2. **Standard errors are understated** without HAC adjustment
# 3. **Comparing IC across horizons requires caution** - higher IC at longer
#    horizons may reflect overlap, not better predictability
#
# **Best practice**: Use HAC-adjusted t-statistics (as provided by `analyze_signal`)
# and be skeptical of IC that increases monotonically with horizon.

# %% [markdown]
# ### IC Decay Analysis
#
# IC decay determines the optimal rebalancing frequency: a signal whose IC halves within a
# week should not be held for a month. We compute IC on a finer horizon grid to estimate
# the signal's useful life.
#
# The half-peak marker below is drawn only for a crossing that comes **after** the peak.
# Searching from the shortest horizon instead puts the marker on the first grid point
# whenever the profile rises, which is the opposite of decay and reads to a reader as
# decay. Where IC never falls back within the grid - the peak being the longest horizon on
# it - no marker is drawn at all.

# %%
# Compute IC across a finer horizon grid (single call - batches all horizons)
decay_horizons = DECAY_HORIZONS

decay_result = analyze_signal(
    factor_df,
    prices_df,
    periods=decay_horizons,
    quantiles=3,
    ic_method="spearman",
    date_col="timestamp",
    asset_col="symbol",
)
decay_ics = [decay_result.ic.get(f"{h}D", float("nan")) for h in decay_horizons]

# %%
# Plot IC decay curve
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=decay_horizons,
        y=decay_ics,
        mode="lines+markers",
        name="Mean IC",
        line=dict(width=2),
    )
)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])

# The crossing search starts after the peak; see the markdown above this figure.
peak_idx = int(np.argmax(decay_ics))
peak_ic = decay_ics[peak_idx]
half_ic = peak_ic / 2
crossing = None
if peak_ic > 0:
    for i in range(peak_idx + 1, len(decay_ics)):
        if decay_ics[i] < half_ic:
            crossing = decay_horizons[i]
            break
if crossing is not None:
    fig.add_vline(
        x=crossing,
        line_dash="dot",
        line_color=COLORS["negative"],
        annotation_text=f"IC below half its peak by day {crossing}",
    )

fig.update_layout(
    title="Mean IC against forward-return horizon",
    xaxis_title="Forward return horizon (days)",
    yaxis_title="Mean IC (Spearman)",
    height=350,
)
show_plotly_with_alt(
    fig,
    alt=(
        "A line with markers showing mean IC against forward-return horizon over a grid "
        "running from one to forty-two days. The whole vertical range spans less than "
        "one hundredth of a correlation unit. The line starts negative at the shortest "
        "horizons, reaches its lowest point around five days, climbs through the dashed "
        "zero line between ten and fifteen days, dips slightly at twenty-one and ends at "
        "its highest point at forty-two days. There is no peak followed by decay: the "
        "profile is still rising where the grid stops."
    ),
)

# %% [markdown]
# **Interpretation**: The horizon-IC curve is the tool for choosing a rebalancing
# frequency - for a cleanly decaying signal you rebalance near where IC peaks and stop
# before it fades. This factor does *not* show that decay. Every horizon IC printed above
# sits within a few thousandths of zero, none is statistically distinct from zero, and the
# profile is still rising at the longest horizon on the grid rather than falling away from
# a peak. The half-peak marker is drawn only when a crossing exists *after* the peak, so
# on this profile there is nothing to mark; a marker placed at the first horizon would
# have described decay running backwards. The honest read is a factor with no exploitable
# cross-sectional horizon structure on this universe.

# %% [markdown]
# ## Turnover Analysis
#
# High turnover erodes returns through transaction costs. A signal with high IC
# but excessive turnover may not be profitable after costs.

# %%
# Turnover metrics
print("\n=== Turnover Analysis ===\n")

if result.turnover:
    print("Mean Turnover by Period:")
    for period_key, turnover in result.turnover.items():
        print(f"  {period_key}: {turnover:.1%}")
else:
    print("Turnover: Not computed")

if result.autocorrelation:
    print(
        f"\nSignal Autocorrelation (lag 1-5): {[f'{ac:.3f}' for ac in result.autocorrelation[:5]]}"
    )
else:
    print("\nAutocorrelation: Not computed")

if result.half_life:
    print(f"\nSignal Half-Life: {result.half_life:.1f} periods")

    # Interpretation
    if result.half_life < 5:
        print("  Interpretation: Fast decay - requires frequent rebalancing")
    elif result.half_life < 20:
        print("  Interpretation: Moderate decay - weekly rebalancing appropriate")
    else:
        print("  Interpretation: Slow decay - monthly rebalancing sufficient")

# %% [markdown]
# ### Turnover and Costs
#
# A simple cost-adjusted IC (Grinold approximation):
#
# $$IC_{net} \approx IC - \frac{c \times \text{turnover}}{E[r]}$$
#
# Where $c$ is round-trip transaction cost and $E[r]$ is expected return. For most equity
# strategies, turning the whole book over once a month or more erodes alpha materially.

# %% [markdown]
# ### Break-Even Cost Analysis
#
# A feasibility check asks: **could this signal survive transaction costs?**
#
# We compare the expected spread (top-bottom quantile return) to the cost of
# achieving that spread. If round-trip costs exceed the expected edge, the
# signal is not tradeable at the given rebalancing frequency.

# %%
# Break-even cost analysis
print("\n=== Break-Even Cost Analysis ===\n")

# Get the 21-day spread and turnover
spread_21d = result.spread.get("21D", float("nan"))
turnover_21d = result.turnover.get("21D", 0.5) if result.turnover else 0.5  # default 50%

# Define cost assumptions (conservative for US equities)
# These should match the trading setup's cost model
COST_ASSUMPTIONS = {
    "spread_bps": 10,  # Half-spread in basis points
    "commission_bps": 5,  # Commission per side
    "market_impact_bps": 10,  # Expected market impact
}

round_trip_cost = (
    2 * COST_ASSUMPTIONS["spread_bps"]
    + 2 * COST_ASSUMPTIONS["commission_bps"]
    + 2 * COST_ASSUMPTIONS["market_impact_bps"]
) / 10000  # Convert to decimal

print("Cost Assumptions (per leg):")
print(f"  Spread:        {COST_ASSUMPTIONS['spread_bps']} bps")
print(f"  Commission:    {COST_ASSUMPTIONS['commission_bps']} bps")
print(f"  Market impact: {COST_ASSUMPTIONS['market_impact_bps']} bps")
print(f"  Round-trip:    {round_trip_cost * 10000:.0f} bps ({round_trip_cost:.2%})")

# %%
# Compute cost-adjusted spread
expected_turnover_per_period = turnover_21d * 2  # Both legs
cost_drag = round_trip_cost * expected_turnover_per_period

cost_adjusted_spread = spread_21d - cost_drag

print("\n21D Signal Analysis:")
print(f"  Raw spread:            {spread_21d:.2%}")
print(f"  Expected turnover:     {expected_turnover_per_period:.0%}")
print(f"  Cost drag per period:  {cost_drag:.2%}")
print(f"  Cost-adjusted spread:  {cost_adjusted_spread:.2%}")

# Break-even calculation
if spread_21d > 0:
    break_even_cost = spread_21d / expected_turnover_per_period
    print(f"  Break-even cost:       {break_even_cost * 10000:.0f} bps")

    if cost_adjusted_spread > 0:
        print("\n[PASS] Signal survives cost assumptions")
    else:
        print("\n[WARNING] Signal does not survive cost assumptions at this turnover")
        print("   Consider: longer horizon, lower turnover, or reduced position sizing")
else:
    print("\n[WARNING] Negative spread - signal direction may be inverted")

# %% [markdown]
# ### Feasibility Guidelines
#
# Three checks for signal feasibility, from the chapter's *Preliminary feasibility
# checks* section:
#
# 1. **Turnover proxies**: Measure entry/exit rates in the top-k set (see above)
# 2. **Break-even cost checks**: Compare spread to conservative cost estimates
# 3. **Capacity warnings**: Recompute IC by liquidity bucket (deferred to Ch8)
#
# **Note**: Liquidity-bucket analysis requires actual market microstructure data
# (average volume, bid-ask spreads) which is not available for this placeholder
# feature. Chapter 8 demonstrates this check with real case study data.

# %% [markdown]
# ## Fold-Aware Evaluation
#
# **Critical**: The IC computed above pools all dates into a single statistic. However,
# real trading strategies are evaluated on **out-of-sample** data using walk-forward
# validation. This section demonstrates fold-aware IC computation.
#
# ### Why Folds Matter
#
# Global IC can be misleading because:
# 1. **Regime dependence**: IC may be high in some periods and zero in others
# 2. **Lookahead contamination**: Parameters tuned on full data leak future information
# 3. **Overfitting detection**: Consistent IC across folds suggests robust signal
#
# The text emphasizes computing IC **per fold** and reporting the distribution of
# fold-level statistics, not just their pooled mean.

# %% [markdown]
# The splits below use an expanding window: train on everything up to the split point,
# test on the period that follows.


# %%
def create_walk_forward_splits(
    dates: list, n_splits: int = 5, min_train_pct: float = 0.2, test_periods: int = 63
) -> list[tuple[list, list]]:
    """Create walk-forward cross-validation splits.

    Args:
        dates: Sorted unique dates
        n_splits: Number of test folds
        min_train_pct: Minimum training data as fraction of total
        test_periods: Number of periods per test fold

    Returns:
        List of (train_dates, test_dates) tuples
    """
    n_dates = len(dates)
    min_train = int(n_dates * min_train_pct)

    splits = []
    for i in range(n_splits):
        # Test window
        test_start = min_train + i * test_periods
        test_end = min(test_start + test_periods, n_dates)

        if test_start >= n_dates:
            break

        train_dates = dates[:test_start]
        test_dates = dates[test_start:test_end]

        if len(test_dates) > 0:
            splits.append((train_dates, test_dates))

    return splits


# %%
# Create splits for our data
unique_dates = sorted(factor_df["timestamp"].unique().to_list())
n_splits = N_SPLITS
test_periods = 63  # ~3 months per fold

splits = create_walk_forward_splits(
    unique_dates, n_splits=n_splits, min_train_pct=0.3, test_periods=test_periods
)

print(f"Created {len(splits)} walk-forward splits")
print("\nSplit structure:")
for i, (train_dates, test_dates) in enumerate(splits):
    print(f"  Fold {i + 1}: Train {len(train_dates)} days, Test {len(test_dates)} days")
    print(f"           Train: {train_dates[0]} to {train_dates[-1]}")
    print(f"           Test:  {test_dates[0]} to {test_dates[-1]}")

# %% [markdown]
# ### Compute Per-Fold IC
#
# For each fold, we compute IC on the **test period only**. This mimics how the
# signal would perform in live trading, where we only see future returns after
# making predictions.

# %%
# Slice eval_df per fold; the forward returns were computed once above.
fold_results = []
evaluated_dates: list = []  # every date inside a test window - needed for a like-for-like IC

for fold_idx, (train_dates, test_dates) in enumerate(splits):
    test_data = eval_df.filter(pl.col("timestamp").is_in(test_dates))

    if len(test_data) < 100:
        continue

    # Cross-sectional IC per date, then average
    ic_per_date = []
    for date_df in test_data.partition_by("timestamp"):
        if len(date_df) < 10:  # Need enough assets for meaningful correlation
            continue
        corr, _ = spearmanr(date_df["factor"].to_numpy(), date_df["fwd_21d"].to_numpy())
        if not np.isnan(corr):
            ic_per_date.append(corr)

    if not ic_per_date:
        continue

    fold_ic = np.mean(ic_per_date)

    # Quantile spread, and the adjacent-step ordering fraction (NOT the Spearman rho
    # analyze_signal reports; see "Monotonicity Interpretation" above).
    test_with_q = test_data.with_columns(
        quantile=pl.col("factor")
        .rank()
        .over("timestamp")
        .qcut(5, labels=[str(i) for i in range(1, 6)])
        .over("timestamp")
    )
    q_rets = test_with_q.group_by("quantile").agg(pl.col("fwd_21d").mean()).sort("quantile")
    q_vals = q_rets["fwd_21d"].to_list()
    fold_spread = q_vals[-1] - q_vals[0] if len(q_vals) >= 2 else float("nan")
    # Monotonicity: fraction of consecutive quantiles in correct order
    if len(q_vals) >= 2:
        diffs = [q_vals[i + 1] - q_vals[i] for i in range(len(q_vals) - 1)]
        fold_step_frac = sum(1 for d in diffs if d > 0) / len(diffs)
    else:
        fold_step_frac = float("nan")

    evaluated_dates.extend(test_dates)
    fold_results.append(
        {
            "fold": fold_idx + 1,
            "test_start": str(test_dates[0]),
            "test_end": str(test_dates[-1]),
            "n_obs": len(test_data),
            "ic": fold_ic,
            "spread": fold_spread,
            "up_step_fraction": fold_step_frac,
        }
    )

# %%
fold_df = pl.DataFrame(fold_results)
print("\n=== Per-Fold IC Results (21D horizon) ===\n")
print(fold_df)

# %%
# Summarize fold-level statistics
fold_ic_mean = fold_df["ic"].mean()
fold_ic_std = fold_df["ic"].std()
fold_ic_min = fold_df["ic"].min()
fold_ic_max = fold_df["ic"].max()
pct_positive = (fold_df["ic"] > 0).mean() * 100

print("\n=== Fold-Level Summary ===")
print(f"Mean IC:       {fold_ic_mean:.4f}")
print(f"Std IC:        {fold_ic_std:.4f}")
print(f"IC Range:      [{fold_ic_min:.4f}, {fold_ic_max:.4f}]")
print(f"% Folds > 0:   {pct_positive:.0f}%")
print(f"Fold ICIR:     {fold_ic_mean / fold_ic_std:.3f}" if fold_ic_std > 0 else "N/A")

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["IC by test fold", "Distribution of fold IC"],
    horizontal_spacing=0.15,
)

# Fold test-start dates as x-axis labels, for temporal context
fold_labels = [r["test_start"][:7] for r in fold_results]  # YYYY-MM format

# Colorblind-safe: blue for positive, amber for negative
fig.add_trace(
    go.Bar(
        x=fold_labels,
        y=[r["ic"] for r in fold_results],
        marker_color=[COLORS["blue"] if r["ic"] > 0 else COLORS["amber"] for r in fold_results],
        name="Fold IC",
    ),
    row=1,
    col=1,
)

# Add global mean line
fig.add_hline(
    y=fold_ic_mean,
    line_dash="dash",
    line_color=COLORS["blue"],
    line_width=1.5,
    annotation_text=f"Mean={fold_ic_mean:.3f}",
    row=1,
    col=1,
)
fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=1)

# Histogram of IC values
fig.add_trace(
    go.Histogram(x=[r["ic"] for r in fold_results], nbinsx=10, marker_color=COLORS["blue"]),
    row=1,
    col=2,
)
fig.add_vline(x=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=2)
fig.add_vline(x=fold_ic_mean, line_dash="dash", line_color=COLORS["blue"], row=1, col=2)

fig.update_xaxes(title_text="Test Fold Start", row=1, col=1)
fig.update_yaxes(title_text="IC (Spearman)", row=1, col=1)
fig.update_xaxes(title_text="IC", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=2)

fig.update_layout(
    height=400,
    showlegend=False,
    font=dict(size=12),
    title_text="Out-of-sample IC across the walk-forward test folds",
)
show_plotly_with_alt(
    fig,
    alt=(
        "Two panels covering the eight walk-forward test folds, whose start dates run "
        "from early 2012 to late 2013. The left panel is a bar per fold: seven bars stand "
        "above zero in navy and one amber bar hangs below it, the negative fold being much "
        "the shortest of the eight. A dashed line marks the fold mean, well above zero, "
        "and a dotted line marks zero itself. The right panel is a histogram of those "
        "eight values, sparse by construction, with most of the mass just above zero and a "
        "single isolated bar far to the right."
    ),
)

# %% [markdown]
# ### Interpretation: Full-Sample vs Fold-Level IC
#
# The obvious move is to read the full-sample IC against the fold-level mean and
# call the difference an aggregation effect - the folds are out of sample, so a
# gap looks like the honest shrinkage of an optimistic full-sample number.
#
# That reading is only available if both statistics cover the **same dates**.
# Ours do not. The full-sample IC averages every date in the panel. The fold-level
# mean averages only dates inside a test window, and with the `min_train_pct` and fold
# count set above those windows are a few hundred consecutive dates near the front of the
# sample - the printout below gives the span and the share. The two numbers describe
# different periods, so their
# difference cannot be attributed to aggregation - or to leakage, which the folds
# structurally cannot produce, since each fold's IC only ever touches its own test
# dates.
#
# The way out is a third number: the same cross-sectional IC, restricted to
# exactly the dates the folds evaluated. It splits the gap into two parts we can
# name separately:
#
# | Comparison | Isolates |
# |------------|----------|
# | Fold-date IC vs full-sample IC | **Period** - is this window unrepresentative? |
# | Fold-level mean vs fold-date IC | **Aggregation** - does averaging folds change anything? |

# %%
# The like-for-like statistic: same per-date Spearman, restricted to fold dates
fold_window = eval_df.filter(pl.col("timestamp").is_in(evaluated_dates))
fold_window_ics = []
for date_df in fold_window.partition_by("timestamp"):
    if len(date_df) < 10:
        continue
    corr, _ = spearmanr(date_df["factor"].to_numpy(), date_df["fwd_21d"].to_numpy())
    if not np.isnan(corr):
        fold_window_ics.append(corr)
fold_window_ic = float(np.mean(fold_window_ics))

full_sample_ic = result.ic.get("21D", float("nan"))
n_all_dates = eval_df["timestamp"].n_unique()
n_fold_dates = len(fold_window_ics)

print("\n=== Like-for-Like IC Comparison (21D) ===")
print(f"Full-sample cross-sectional IC ({n_all_dates:,} dates): {full_sample_ic:>8.4f}")
print(f"Same statistic, fold dates only ({n_fold_dates:,} dates): {fold_window_ic:>8.4f}")
print(f"Fold-level mean IC              ({n_fold_dates:,} dates): {fold_ic_mean:>8.4f}")
print("\n--- Decomposition of the full-sample vs fold-level gap ---")
print(
    f"Period effect      (fold-date IC - full-sample IC): {fold_window_ic - full_sample_ic:>8.4f}"
)
print(f"Aggregation effect (fold mean    - fold-date IC):   {fold_ic_mean - fold_window_ic:>8.4f}")

fold_span = f"{min(evaluated_dates)} to {max(evaluated_dates)}"
coverage_pct = 100 * n_fold_dates / n_all_dates
print(f"\nFolds evaluate {fold_span} - {coverage_pct:.0f}% of the panel's dates.")
if abs(fold_window_ic - full_sample_ic) > abs(fold_ic_mean - fold_window_ic):
    print("[READ] The gap is a period effect: this window is not the average window.")
    print("       It says nothing about overfitting, and nothing about leakage.")
else:
    print("[READ] The gap survives on identical dates, so it is an aggregation effect.")

# %% [markdown]
# ### Within-Time Permutation Test
#
# The chapter recommends a **within-time permutation test** as a
# null-distribution benchmark: break the feature-label pairing while preserving the
# structure of the data, then ask how often chance alone reproduces the observed IC.
# Two details decide whether the answer means anything.
#
# **The null must cover the same dates as the observed statistic.** A null built
# from every date in the panel is a distribution of a mean over ~5,000 dates; our
# observed statistic is a mean over ~500. The mean of a larger sample is mechanically
# less dispersed, so such a null is too narrow by roughly $\sqrt{5000/500} \approx 3$
# and would reject almost anything. We therefore permute only the fold dates.
#
# **The null must preserve temporal dependence.** Shuffling assets independently on
# each date implies the per-date ICs are independent, so the null mean's standard
# error shrinks like $\sigma/\sqrt{n_{\text{dates}}}$. Our labels are 21-day forward
# returns sampled daily: consecutive dates share 20 of 21 days of return, so both the
# returns and the per-date ICs are strongly autocorrelated, and the true standard
# error is much larger. Independent within-date shuffling would understate it by
# roughly an order of magnitude.
#
# The repair is a **block permutation**. We draw one asset relabeling per block of 21
# sessions - the label horizon - and hold it fixed across the block. Within a block
# each asset's return path stays intact, so the autocorrelation the observed IC
# inherits also lives in the null; across blocks the relabelings are independent.
# The null then answers the right question: *given how persistent this data is, how
# often does a random assignment rank this well over this window?*

# %%
# Block permutation: one asset relabeling per BLOCK_SESSIONS, restricted to fold dates
rng = np.random.default_rng(SEED)
n_permutations = N_PERMUTATIONS
BLOCK_SESSIONS = 21  # label horizon - blocks must be at least as long as the overlap

perm_df = fold_window.sort(["timestamp", "symbol"])
dates_arr = perm_df["timestamp"].to_numpy()
factors_arr = perm_df["factor"].to_numpy()
returns_arr = perm_df["fwd_21d"].to_numpy()
symbol_codes = perm_df["symbol"].cast(pl.Categorical).to_physical().to_numpy()
n_symbols = int(symbol_codes.max()) + 1
unique_dates_perm = np.unique(dates_arr)

# Group rows by date; within a date, rows are already in symbol order
date_groups = [np.where(dates_arr == d)[0] for d in unique_dates_perm]
date_groups = [idx for idx in date_groups if len(idx) >= 10]

# Ranks are invariant to relabeling, so pre-compute both sides once
factor_ranks_by_group = [rankdata(factors_arr[idx]) for idx in date_groups]
return_ranks_by_group = [rankdata(returns_arr[idx]) for idx in date_groups]
symbols_by_group = [symbol_codes[idx] for idx in date_groups]

# %%
permuted_ics = []
for _ in range(n_permutations):
    ic_per_date = []
    block_keys = None
    for i, f_ranks in enumerate(factor_ranks_by_group):
        # New relabeling only when a block boundary is crossed
        if i % BLOCK_SESSIONS == 0:
            block_keys = rng.permutation(n_symbols)
        # Same key vector across the block => same asset->asset mapping across the block
        order = np.argsort(block_keys[symbols_by_group[i]], kind="stable")
        r_ranks = return_ranks_by_group[i][order]
        corr = np.corrcoef(f_ranks, r_ranks)[0, 1]
        if not np.isnan(corr):
            ic_per_date.append(corr)
    if ic_per_date:
        permuted_ics.append(np.mean(ic_per_date))

permuted_ics = np.array(permuted_ics)
# Observed statistic and null are now the same estimator on the same dates
observed_ic_mean = fold_window_ic

# (r + 1) / (B + 1): a permutation p-value can never be exactly zero - the observed
# assignment is itself one of the arrangements under the null
n_at_least = int(np.sum(permuted_ics >= observed_ic_mean))
p_value_perm = (n_at_least + 1) / (len(permuted_ics) + 1)

print("=== Within-Time Block Permutation Test ===\n")
print(
    f"Dates: {len(factor_ranks_by_group):,} (fold windows only, block = {BLOCK_SESSIONS} sessions)"
)
print(f"Observed mean IC: {observed_ic_mean:.4f}")
print(f"Permutation null - mean: {permuted_ics.mean():.4f}, std: {permuted_ics.std():.4f}")
print(f"Permutation p-value (one-sided): {p_value_perm:.4f}")
print(f"Permutations: {n_permutations} (finest resolvable p = {1 / (n_permutations + 1):.4f})")

# %%
# Visualize permutation distribution vs observed IC
fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=permuted_ics,
        nbinsx=30,
        name="Permuted IC",
        marker_color=COLORS["slate"],
        opacity=0.8,
    )
)
fig.add_vline(
    x=observed_ic_mean,
    line_dash="solid",
    line_color=COLORS["negative"],
    line_width=2,
    annotation_text=f"Observed IC={observed_ic_mean:.4f}",
)
fig.update_layout(
    title="Block-permutation null for mean IC, against the observed value",
    xaxis_title="Mean IC (block-permuted, fold dates only)",
    yaxis_title="Count",
    height=350,
    showlegend=False,
)
show_plotly_with_alt(
    fig,
    alt=(
        "A histogram of the block-permutation null for mean IC over the fold dates, drawn "
        "in slate. It is a symmetric bell centred on zero, running out to roughly plus and "
        "minus 0.045. A solid red vertical line marks the observed mean IC, standing well "
        "clear of the right-hand edge of the null with no permuted value anywhere near it."
    ),
)

# %% [markdown]
# The claim that the block null is the honest one is checkable: rerun the same permutation
# with a block of a single session, which is exactly the independent within-date shuffle,
# and compare the two null widths on identical dates.

# %% tags=["results"]
naive_ics = []
for _ in range(n_permutations):
    ic_per_date = []
    for i, f_ranks in enumerate(factor_ranks_by_group):
        keys = rng.permutation(n_symbols)  # a fresh relabeling every date
        order = np.argsort(keys[symbols_by_group[i]], kind="stable")
        corr = np.corrcoef(f_ranks, return_ranks_by_group[i][order])[0, 1]
        if not np.isnan(corr):
            ic_per_date.append(corr)
    if ic_per_date:
        naive_ics.append(np.mean(ic_per_date))
naive_ics = np.array(naive_ics)

naive_p = (int(np.sum(naive_ics >= observed_ic_mean)) + 1) / (len(naive_ics) + 1)
print("=== Null width: independent within-date shuffle vs block permutation ===\n")
print(f"  independent shuffle  std {naive_ics.std():.4f}   one-sided p {naive_p:.4f}")
print(
    f"  block of {BLOCK_SESSIONS} sessions  std {permuted_ics.std():.4f}   one-sided p {p_value_perm:.4f}"
)
print(f"\n  ratio of null widths: {permuted_ics.std() / naive_ics.std():.1f}x, on identical dates")

# %% [markdown]
# **Interpretation**: over the fold window, the observed IC sits outside every block
# permutation, so the factor ranked ETFs better than chance *in the years those folds
# cover*. That is the only claim the test supports, and it is worth being precise about
# why.
#
# Read it against the full-sample IC printed under **Information Coefficient (IC)
# Analysis**, which is indistinguishable from zero with a HAC $t$ well under one. These
# are not opposing readings of one quantity - they are two different windows, as the
# decomposition above showed: the entire full-sample-versus-fold gap was a period effect,
# and the aggregation term rounded to nothing. The factor worked in the folds' two years
# and does nothing over the full panel. The folds say something about those years, not
# about the signal.
#
# Two lessons generalize past this factor:
#
# - **A window that flatters the signal is the default outcome, not a surprise.** The
#   `min_train_pct` and fold count set above evaluate only the front of the panel and
#   never score its last decade - the coverage share is printed with the decomposition. A
#   fold scheme that leaves most of the sample unevaluated cannot support a claim about
#   the sample.
# - **A null must be as dependent as the data.** The comparison printed just above puts a
#   number on it: shuffling assets independently each date gives a null several times
#   narrower than blocking at the label horizon, on identical dates. Both nulls reject
#   here, and at this permutation count both p-values sit on the floor, so the p-value is
#   not what separates them - the width is. A null understated by that factor is the
#   difference between rejecting and not rejecting for any factor whose IC lands near the
#   boundary rather than far outside it, which is where most candidate factors land.
#
# The rest of the evidence points the other way, and the quantile analysis already showed
# it: the full-sample quintile spread is **negative** at every horizon, and monotonicity is
# negative at every horizon too - strongly so at the short ones, where the ordering is
# nearly perfect with its sign reversed. The bottom quintile out-earns the top.
# Cross-sectional 21-day ETF momentum is a **reversal** signal over this panel. A single
# favorable window does not overturn that; it illustrates how easily a fold scheme can
# hide it.

# %% [markdown]
# ## Factor Scorecard Output
#
# Export a structured summary for downstream use, including both global and
# fold-level statistics.

# %%
# Build factor scorecard with fold-level stats
scorecard = {
    "factor_name": "momentum_21d",
    "n_assets": result.n_assets,
    "n_dates": result.n_dates,
    "horizons": {},
}

for period in PERIODS:
    period_key = f"{period}D"
    scorecard["horizons"][period_key] = {
        "ic_mean": round(result.ic.get(period_key, float("nan")), 4),
        "ic_std": round(np.std(result.ic_series.get(period_key, [])), 4)
        if result.ic_series.get(period_key)
        else None,
        "icir": round(result.ic_ir.get(period_key, float("nan")), 3),
        "t_stat": round(result.ic_t_stat.get(period_key, float("nan")), 2),
        "p_value": round(result.ic_p_value.get(period_key, float("nan")), 4),
        "spread": round(result.spread.get(period_key, float("nan")), 4),
        "monotonicity": round(result.monotonicity.get(period_key, float("nan")), 3),
    }

# Add fold-level statistics (21D only)
if fold_results:
    scorecard["fold_evaluation"] = {
        "n_folds": len(fold_results),
        "ic_mean": round(fold_ic_mean, 4),
        "ic_std": round(fold_ic_std, 4),
        "ic_min": round(fold_ic_min, 4),
        "ic_max": round(fold_ic_max, 4),
        "pct_positive_folds": round(pct_positive, 1),
        "fold_icir": round(fold_ic_mean / fold_ic_std, 3) if fold_ic_std > 0 else None,
    }

if result.turnover:
    scorecard["turnover"] = {k: round(v, 3) for k, v in result.turnover.items()}
if result.half_life:
    scorecard["half_life_periods"] = round(result.half_life, 1)

# Display scorecard
print("\n=== Factor Scorecard ===\n")
print(json.dumps(scorecard, indent=2))

# %% [markdown]
# ## Binary Label Evaluation
#
# When labels are binary (e.g., "positive return" vs "negative return"), we evaluate
# using classification metrics rather than IC. The feature acts as a **score** that
# separates positives from negatives.
#
# **Key metrics:**
# - **ROC AUC**: Area under ROC curve (threshold-free ranking metric)
# - **PR AUC**: Area under Precision-Recall curve (better for imbalanced data)
# - **Confusion matrix**: TP, FP, TN, FN at a chosen threshold

# %%
# Positive = return > 0, negative = return <= 0.
binary_df = eval_df.with_columns(
    pl.when(pl.col("fwd_21d") > 0).then(1).otherwise(0).alias("binary_label")
)

# Use factor as score (higher = predict positive)
y_true = binary_df["binary_label"].to_numpy()
y_score = binary_df["factor"].to_numpy()

# Handle NaN values; sklearn accepts the native int32/float64 dtypes.
mask = ~(np.isnan(y_true) | np.isnan(y_score))
y_true = y_true[mask]
y_score = y_score[mask]

print(f"Binary evaluation: {len(y_true):,} samples")
print(f"Class balance: {y_true.mean():.1%} positive")

# %% [markdown]
# The ROC and precision-recall sweeps below are built by hand rather than taken from
# sklearn's curve helpers, for an environment reason recorded in the code comment: the
# library path fails under the pin this project carries. The manual sweep is deterministic
# and agrees with the library to well inside plotting resolution.

# %%
roc_auc = roc_auc_score(y_true, y_score)

# sklearn's _binary_clf_curve raises IndexError on state-dependent runs of this array
# under the sklearn pin econml imposes; the failure mode is uncharacterized.
_order = np.argsort(-y_score, kind="mergesort")
# int64 promotion guards np.cumsum from int32 overflow at 466k samples.
_yt_sorted = y_true[_order].astype(np.int64)
_score_sorted = y_score[_order]
_n_pos = int(_yt_sorted.sum())
_n_neg = len(_yt_sorted) - _n_pos
_tps_cum = np.cumsum(_yt_sorted)
_fps_cum = np.cumsum(1 - _yt_sorted)
fpr = np.concatenate([[0.0], _fps_cum / _n_neg])
tpr = np.concatenate([[0.0], _tps_cum / _n_pos])
# Thresholds aligned with fpr/tpr via the same _order index; no re-sort.
# Ties are not collapsed (unlike sklearn) - at 466k unique-or-near-unique
# float scores the AUC difference is sub-1e-6.
roc_thresholds = np.concatenate([[np.inf], _score_sorted])

# Precision-Recall (manual sweep omits sklearn's trailing (precision=1, recall=0)
# sentinel; the trapezoidal-AUC difference at 466k points is sub-1e-6)
_pred_pos = np.arange(1, len(_yt_sorted) + 1)
precision = _tps_cum / _pred_pos
recall = _tps_cum / _n_pos
pr_auc = auc(recall, precision)

print("\nThreshold-Free Metrics:")
print(f"  ROC AUC: {roc_auc:.3f}")
print(f"  PR AUC:  {pr_auc:.3f}")

# Interpretation
if roc_auc > 0.55:
    print("  Interpretation: AUC above 0.55 - score ranks positives above negatives")
elif roc_auc > 0.52:
    print(
        "  Interpretation: AUC in (0.52, 0.55] - small ranking signal; tradeability not evaluated here"
    )
else:
    print("  Interpretation: AUC at or near 0.5 - score does not separate the two classes")

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC curve", "Precision-recall curve"])

# ROC curve
fig.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC (AUC={roc_auc:.3f})",
        line=dict(color=COLORS["blue"], width=2),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash", color=COLORS["neutral"]),
    ),
    row=1,
    col=1,
)

# PR curve
fig.add_trace(
    go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"PR (AUC={pr_auc:.3f})",
        line=dict(color=COLORS["blue"], width=2),
    ),
    row=1,
    col=2,
)
baseline_precision = y_true.mean()
fig.add_hline(
    y=baseline_precision,
    line_dash="dash",
    line_color=COLORS["neutral"],
    row=1,
    col=2,
    annotation_text=f"Prevalence={baseline_precision:.1%}",
    annotation_position="bottom right",
)

fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
fig.update_xaxes(title_text="Recall", row=1, col=2)
fig.update_yaxes(title_text="Precision", row=1, col=2)

fig.update_layout(
    height=400,
    showlegend=True,
    font=dict(size=12),
    title_text="Threshold-free separation of positive from negative forward returns",
)
show_plotly_with_alt(
    fig,
    alt=(
        "Two panels. The left panel plots the ROC curve against the dashed random-guess "
        "diagonal; the two are indistinguishable, the solid curve tracing the diagonal "
        "from corner to corner, and the legend reports its area. The right panel plots "
        "precision against recall: the curve is flat across the whole recall range and "
        "sits on the dashed prevalence line, which the annotation labels with the base "
        "rate. Neither panel shows the score separating the classes."
    ),
)

# %%
# Confusion matrix at median threshold
threshold = np.median(y_score)
y_pred = (y_score >= threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix (threshold = median):")
print(f"  True Positives:  {tp:,}")
print(f"  False Positives: {fp:,}")
print(f"  True Negatives:  {tn:,}")
print(f"  False Negatives: {fn:,}")

precision_at_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_at_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (
    2 * precision_at_thresh * recall_at_thresh / (precision_at_thresh + recall_at_thresh)
    if (precision_at_thresh + recall_at_thresh) > 0
    else 0
)

print(f"\nAt threshold = {threshold:.4f}:")
print(f"  Precision: {precision_at_thresh:.1%}")
print(f"  Recall:    {recall_at_thresh:.1%}")
print(f"  F1 Score:  {f1:.3f}")

# %% [markdown]
# ### Library Binary Metrics
#
# Point estimates of precision/recall are noisy. `ml4t-diagnostic` provides
# `binary_classification_report()` with Wilson confidence intervals and
# statistical tests in a single call.

# %%
# Create Polars Series for library (expects 0/1 integer series)
signals_pl = pl.Series("signal", y_pred)
labels_pl = pl.Series("label", y_true)

report = binary_classification_report(signals_pl, labels_pl, confidence=0.95)

print("=== ml4t-diagnostic Binary Classification Report ===\n")
print(
    f"Precision:  {report.precision:.3f}  CI: [{report.precision_ci[0]:.3f}, {report.precision_ci[1]:.3f}]"
)
print(
    f"Recall:     {report.recall:.3f}  CI: [{report.recall_ci[0]:.3f}, {report.recall_ci[1]:.3f}]"
)
print(f"F1 Score:   {report.f1_score:.3f}")
print(f"Lift:       {report.lift:.2f}x")
print(f"Coverage:   {report.coverage:.1%}")
print(f"\nBinomial p-value: {report.binomial_pvalue:.4f}")
print(f"Z-test (vs base rate): z={report.z_test_stat:.2f}, p={report.z_test_pvalue:.4f}")

# %%
# Wilson score intervals for specific metrics
prec_ci = wilson_score_interval(tp, tp + fp, confidence=0.95)
recall_ci = wilson_score_interval(tp, tp + fn, confidence=0.95)

print("\nWilson Score Intervals (95%):")
print(f"  Precision: [{prec_ci[0]:.3f}, {prec_ci[1]:.3f}]")
print(f"  Recall:    [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]")
print("\nWilson intervals are more accurate than normal approximation CIs,")
print("especially when proportions are near 0 or 1.")

# %% [markdown]
# ### Binary Evaluation Guidelines
#
# | Metric | What It Measures | Trading Interpretation |
# |--------|------------------|------------------------|
# | **ROC AUC** | Ranking quality | Against the no-skill diagonal, with a confidence interval |
# | **PR AUC** | Precision at various recalls | Use when positives are rare |
# | **Precision** | % of predicted positives correct | Matters for trade entry |
# | **Recall** | % of actual positives found | Matters for opportunity cost |
#
# **Threshold selection** depends on the trading objective:
# - High precision, low recall: fewer trades, higher win rate
# - High recall, low precision: capture more opportunities, lower win rate
# - The optimal threshold depends on costs and capacity constraints

# %% [markdown]
# ### Fold-Aware Binary Evaluation
#
# Just as with IC, we should compute ROC AUC per fold to assess out-of-sample
# classification performance.

# %%
# Compute per-fold ROC AUC
fold_auc_results = []

for fold_idx, (train_dates, test_dates) in enumerate(splits):
    # Filter to test period
    test_binary = binary_df.filter(pl.col("timestamp").is_in(test_dates))

    if len(test_binary) < 50:
        continue

    # Extract arrays
    y_true_fold = test_binary["binary_label"].to_numpy()
    y_score_fold = test_binary["factor"].to_numpy()

    # Handle NaN
    mask = ~(np.isnan(y_true_fold) | np.isnan(y_score_fold))
    y_true_fold = y_true_fold[mask]
    y_score_fold = y_score_fold[mask]

    if len(np.unique(y_true_fold)) < 2:  # Need both classes
        continue

    try:
        fold_auc = roc_auc_score(y_true_fold, y_score_fold)
        fold_auc_results.append(
            {
                "fold": fold_idx + 1,
                "n_samples": len(y_true_fold),
                "base_rate": y_true_fold.mean(),
                "roc_auc": fold_auc,
            }
        )
    except ValueError:
        pass  # Skip if ROC cannot be computed

if fold_auc_results:
    fold_auc_df = pl.DataFrame(fold_auc_results)
    print("Per-Fold ROC AUC")
    display(fold_auc_df)

    # Summary
    auc_mean = np.mean([r["roc_auc"] for r in fold_auc_results])
    auc_std = np.std([r["roc_auc"] for r in fold_auc_results])
    print(f"Fold-level AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"Global AUC:     {roc_auc:.3f}")

# %% [markdown]
# ## Summary
#
# ### Key Metrics for Signal Evaluation
#
# | Metric | What It Measures | How to read it |
# |--------|------------------|----------------|
# | **IC** | Cross-sectional predictability | Against its own standard error; the literature bands are printed under *Interpreting an IC magnitude* |
# | **ICIR** | Risk-adjusted IC (mean/std) | Together with IC; a near-zero ICIR says the mean is inside the daily dispersion |
# | **Fold ICIR** | IC stability across test folds | Alongside which dates the folds actually cover |
# | **Spread** | Top-bottom quantile difference | Sign first, then size against costs |
# | **Monotonicity** | Spearman rho of quantile rank against mean return | Magnitude against the bands printed under *Monotonicity Interpretation*; sign first |
# | **Turnover** | Signal stability | Against the break-even cost computed above |
# | **Half-life** | Signal decay rate | Only where the horizon profile actually decays |
#
# ### Fold-Aware Evaluation (Critical)
#
# Always compute metrics **per fold** using walk-forward validation:
#
# 1. Create expanding-window or rolling-window splits.
# 2. Compute IC (or AUC) on each test fold.
# 3. Report the **distribution** of fold-level metrics, not just pooled values.
# 4. Before reading a gap between the pooled and fold-level figures, recompute the pooled
#    statistic **on the fold dates only**. This notebook's own decomposition showed the
#    whole gap was a period effect: the folds covered a favourable stretch near the front
#    of the panel, and nothing about aggregation or overfitting was involved. A gap
#    between two statistics computed over different dates is not evidence of either.
#
# ### API Reference
#
# ```python
# from ml4t.diagnostic.signal import analyze_signal
#
# result = analyze_signal(
#     factor_df,                # timestamp, symbol, factor
#     prices_df,                # timestamp, symbol, price
#     periods=(1, 5, 21),       # Forward return horizons
#     quantiles=5,              # Number of quantiles
#     ic_method="spearman",     # Rank correlation
# )
#
# # Access results
# result.ic              # Mean IC by period
# result.ic_ir           # ICIR by period
# result.quantile_returns # Returns by quantile
# result.spread          # Top-bottom spread
# result.monotonicity    # Spearman rho of quantile rank vs mean return, in [-1, 1]
# result.turnover        # Signal turnover
# result.summary()       # Human-readable summary
# ```
#
# ### Next Notebooks
#
# - [`06_ic_inference`](06_ic_inference.ipynb) - HAC adjustment and block bootstrap for IC inference
# - [`07_multiple_testing`](07_multiple_testing.ipynb) - FDR control when evaluating many factors
