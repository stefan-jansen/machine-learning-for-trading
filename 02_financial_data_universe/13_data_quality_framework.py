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
# # Data Quality Framework: Validation, Anomalies, and Remediation
#
# **Docker image**: `ml4t`
#
# ## Purpose
# Demonstrate the four pillars of a production data-quality pipeline applied to US equities
# daily OHLCV: structural validation (OHLC invariants, nulls, duplicates), anomaly detection
# (return outliers, volume spikes, price staleness), distribution drift (PSI), and ingestion
# hygiene (gaps, duplicates, corporate-action detection). All checks come from the
# `ml4t.data.validation` and `ml4t.data.anomaly` modules.
#
# ## Learning Objectives
# - Run `OHLCVValidator` and read its issue report, including on data with injected faults.
# - See why MAD, Z-score and IQR flag different numbers of events at the same threshold.
# - Compute the Population Stability Index, and see what its binning discards.
# - Score a corporate-action detector against the split and dividend columns of the same file.
#
# ## Book reference
# Chapter 2, §2.3 (data quality framework). Downstream chapters that consume the cleaned panel:
# `14_point_in_time_validation` (bitemporal hygiene), `15_survivorship_bias_detection`,
# `17_complete_pipeline`.
#
# ## Prerequisites
# - Quandl/Wiki US equities parquet materialized under `ML4T_DATA_PATH` (the legacy dataset;
#   ends 2018-03-27).
# - Loader `data.load_us_equities`.
# - Library packages `ml4t.data.validation` and `ml4t.data.anomaly`.

# %%
"""Data Quality Framework: validation, anomaly detection, and remediation."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import structlog
from ml4t.data.anomaly import (
    AnomalyManager,
    PriceStalenessDetector,
    ReturnOutlierDetector,
    VolumeSpikeDetector,
)
from ml4t.data.anomaly.config import (
    AnomalyConfig,
    PriceStalenessConfig,
    ReturnOutlierConfig,
    VolumeSpikeConfig,
)
from ml4t.data.validation import OHLCVValidator

from data import load_us_equities
from utils.paths import get_output_dir
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# ### Quieting the library, not the interpreter
#
# The anomaly and validation libraries log a line per symbol per detector through structlog,
# which bypasses stdlib logging, so the cells below would print the library's debug stream
# rather than their own results. Filtering structlog at WARNING fixes that and leaves
# Python's own warnings alone. A blanket `warnings.filterwarnings("ignore")` would also do
# it, and would hide the warnings this notebook needs to see: a plotting call that drops its
# alt text warns, and under Papermill that warning is the only sign it happened.

# %%
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _to_date(value: object) -> object:
    """Normalize a polars timestamp scalar to a datetime.date for printing."""
    return value.date() if hasattr(value, "date") else value


# %% [markdown]
# ### Declared parameters
#
# Every threshold each detector uses is declared here rather than written into the call that
# uses it. Two of them are worth reading before the sections that consume them.
#
# `OUTLIER_THRESHOLD` is passed unchanged to three detectors that use it as a multiplier on
# three different scale estimates, so the same number produces three different cutoffs in
# return space. Section 2 measures those cutoffs rather than leaving the comparison implicit.
#
# `SYMBOLS` holds five large-cap tickers as the legacy Wiki/Quandl vintage writes them, which
# means FB rather than META.
#
# `CORPORATE_ACTION_THRESHOLD` is the overnight return above which a move is treated as a
# candidate corporate action. The Wiki/Quandl panel carries the actual split ratios and
# ex-dividend amounts in its own columns, so Section 4 scores the threshold against them
# instead of eyeballing the flagged dates.

# %% tags=["parameters"]
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "FB"]
DEMO_SYMBOL = "AAPL"

# Structural validation.
MAX_RETURN_THRESHOLD = 0.5  # flag a daily move larger than this
STALENESS_DAYS_VALIDATOR = 5  # flag this many identical prices in a row

# Anomaly detection.
OUTLIER_THRESHOLD = 3.0  # multiplier on each method's scale estimate
OUTLIER_MIN_SAMPLES = 20
MAD_NORMAL_SCALE = 0.6745  # MAD of a standard normal; rescales MAD to a sigma-like unit
VOLUME_WINDOW = 20
VOLUME_THRESHOLD = 3.0
STALENESS_DAYS_DETECTOR = 3

# Drift.
PSI_BINS = 10
PSI_MODERATE = 0.1  # below this, no material change
PSI_SIGNIFICANT = 0.25  # above this, act

# Hygiene.
MAX_GAP_DAYS = 5  # a longer gap than any holiday weekend produces
CORPORATE_ACTION_THRESHOLD = 0.25  # overnight return treated as a candidate action

# %%
OUTPUT_DIR = get_output_dir(2, "quality")

# %% [markdown]
# ## Load Sample Data
#
# The pre-2018 Wiki/Quandl US equities panel. The validation and anomaly detectors operate on
# any DataFrame carrying `timestamp / open / high / low / close / volume`.

# %%
wiki_df = load_us_equities()
print(
    f"US equities loaded: {len(wiki_df):,} rows; "
    f"{wiki_df['timestamp'].min()} to {wiki_df['timestamp'].max()}"
)

datasets = {
    symbol: (
        wiki_df.lazy()
        .filter(pl.col("symbol") == symbol)
        .select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        .collect()
    )
    for symbol in SYMBOLS
}
missing = [s for s, df in datasets.items() if df.is_empty()]
if missing:
    raise RuntimeError(f"Symbols missing from Wiki/Quandl dataset: {missing}")

per_symbol_rows = pl.DataFrame(
    {
        "symbol": list(datasets),
        "rows": [len(df) for df in datasets.values()],
        "first": [_to_date(df["timestamp"].min()) for df in datasets.values()],
        "last": [_to_date(df["timestamp"].max()) for df in datasets.values()],
    }
)
per_symbol_rows

# %% [markdown]
# ---
#
# ## Part 1: OHLC Invariant Validation
#
# **OHLC invariants** are relationships that hold by the definition of the four fields:
#
# | Invariant          | Why it holds        |
# |--------------------|---------------------|
# | High >= Low        | by definition       |
# | High >= Open, Close| high is the maximum |
# | Low <= Open, Close | low is the minimum  |
# | Prices > 0         | no negative prices  |
# | Volume >= 0        | no negative volume  |
#
# A violation is a provider error, transmission corruption, or an adjustment applied to some
# fields and not others.

# %%
validator = OHLCVValidator(
    check_nulls=True,
    check_price_consistency=True,
    negative_price_policy="forbid",
    check_negative_volume=True,
    check_duplicate_timestamps=True,
    check_chronological_order=True,
    check_price_staleness=True,
    check_extreme_returns=True,
    max_return_threshold=MAX_RETURN_THRESHOLD,
    staleness_threshold=STALENESS_DAYS_VALIDATOR,
)

validation_summary = pl.DataFrame(
    [
        {
            "symbol": sym,
            "passed": (r := validator.validate(df)).passed,
            "issues": len(r.issues),
            "critical": r.critical_count,
            "errors": r.error_count,
        }
        for sym, df in datasets.items()
    ]
)
validation_summary

# %% [markdown]
# ### A check that passes has not been tested
#
# Every symbol passes, which is the outcome a clean panel should produce and also the outcome
# a broken validator would produce. The two are told apart by giving it data known to be bad:
# three faults are injected into the demo symbol below, one of each kind the validator claims
# to catch, and the report is checked for all three.

# %%
clean_df = datasets[DEMO_SYMBOL]
highs = clean_df["high"].to_numpy().copy()
lows = clean_df["low"].to_numpy().copy()
volumes = clean_df["volume"].to_numpy().copy()

FAULT_ROWS = slice(10, 13)
FAULT_VOLUME_ROW = 20
FAULT_NULL_ROW = 30

highs[FAULT_ROWS] = lows[FAULT_ROWS] - 1.0
volumes[FAULT_VOLUME_ROW] = -1000
# A genuine null, not NumPy NaN: polars stores np.nan as a float NaN that the null check
# ignores, so only a real null exercises OHLCVValidator's check_nulls.
close_col = pl.Series("close", clean_df["close"].to_numpy()).scatter(FAULT_NULL_ROW, None)

dirty_df = pl.DataFrame(
    {
        "timestamp": clean_df["timestamp"],
        "symbol": clean_df["symbol"],
        "open": clean_df["open"],
        "high": highs,
        "low": lows,
        "close": close_col,
        "volume": volumes,
    }
)

dirty_result = validator.validate(dirty_df)
print(f"Validation passed: {dirty_result.passed}")
print(f"Critical: {dirty_result.critical_count}, Errors: {dirty_result.error_count}")

_checks_fired = {issue.check for issue in dirty_result.issues}
for fault, check in [
    ("high below low", "price_consistency"),
    ("negative volume", "negative_volume"),
    ("null close", "null_values"),
]:
    print(f"  {fault:16s} -> {check:18s} {'fired' if check in _checks_fired else 'NOT FIRED'}")

dirty_issues = pl.DataFrame(
    [
        {
            "severity": issue.severity.name,
            "check": issue.check,
            "rows": issue.row_count or 0,
            "message": issue.message,
        }
        for issue in dirty_result.issues
    ]
)
dirty_issues

# %% [markdown]
# ---
#
# ## Part 2: Anomaly Detection
#
# Validation asks whether the data is structurally possible. Anomaly detection asks whether
# structurally valid data is unusual:
#
# | Detector               | What it finds                  | Method                |
# |------------------------|--------------------------------|-----------------------|
# | ReturnOutlierDetector  | flash crashes, splits, pumps   | MAD, Z-score, IQR     |
# | VolumeSpikeDetector    | unusual trading activity       | rolling Z-score       |
# | PriceStalenessDetector | data gaps, illiquid securities | consecutive-unchanged |

# %% [markdown]
# ### The same threshold is three different cutoffs
#
# All three return-outlier methods take a threshold and compare a scaled deviation against it.
# They do not scale by the same thing:
#
# - **MAD** flags a return when $|r - \mathrm{median}|$ exceeds $\tau \cdot \mathrm{MAD} /
#   c$, where MAD is the median absolute deviation from the median and $c$, declared as
#   `MAD_NORMAL_SCALE`, is the MAD of a standard normal, which puts the result on the same
#   footing as a standard deviation.
# - **Z-score** flags it when $|r - \mathrm{mean}|$ exceeds $\tau \cdot \sigma$.
# - **IQR** flags it when $r$ falls outside $[Q_1 - \tau \cdot \mathrm{IQR},\ Q_3 + \tau \cdot
#   \mathrm{IQR}]$.
#
# Running all three at one threshold therefore compares three different cutoffs, not three
# ways of applying one. The next cell prints the cutoffs the shared threshold implies in
# return space, so the counts that follow can be read against the boundary that produced them.

# %%
sample_df = datasets[DEMO_SYMBOL]
_returns = sample_df.select(pl.col("close").pct_change().alias("r"))["r"].drop_nulls()

_median = _returns.median()
_mad = (_returns - _median).abs().median()
_mean, _sd = _returns.mean(), _returns.std()
_q1, _q3 = _returns.quantile(0.25), _returns.quantile(0.75)
_iqr = _q3 - _q1

cutoffs = pl.DataFrame(
    {
        "method": ["mad", "zscore", "iqr"],
        "scale_estimate": [_mad, _sd, _iqr],
        "lower_cutoff": [
            _median - OUTLIER_THRESHOLD * _mad / MAD_NORMAL_SCALE,
            _mean - OUTLIER_THRESHOLD * _sd,
            _q1 - OUTLIER_THRESHOLD * _iqr,
        ],
        "upper_cutoff": [
            _median + OUTLIER_THRESHOLD * _mad / MAD_NORMAL_SCALE,
            _mean + OUTLIER_THRESHOLD * _sd,
            _q3 + OUTLIER_THRESHOLD * _iqr,
        ],
    }
)
print(f"{DEMO_SYMBOL} daily returns, cutoffs implied by threshold {OUTLIER_THRESHOLD}:")
cutoffs

# %%
methods = ["mad", "zscore", "iqr"]

method_summary = pl.DataFrame(
    [
        {
            "method": m,
            "anomalies": len(
                ReturnOutlierDetector(
                    config=ReturnOutlierConfig(
                        method=m,
                        threshold=OUTLIER_THRESHOLD,
                        min_samples=OUTLIER_MIN_SAMPLES,
                    )
                ).detect(sample_df, symbol=DEMO_SYMBOL)
            ),
        }
        for m in methods
    ]
).join(cutoffs, on="method")
method_summary

# %% [markdown]
# The counts follow the cutoffs: the method with the narrowest band flags the most, and the
# ordering of the counts is the ordering of the bands.
#
# The reason MAD's band is narrowest is worth stating carefully, because the intuitive version
# has it backwards. MAD is a robust scale estimate, meaning the extreme returns barely move it:
# it is the median of the absolute deviations, and a handful of enormous days cannot shift a
# median. The standard deviation has no such protection and is inflated by those same days. So
# on a heavy-tailed return series MAD comes out much smaller than sigma, the cutoff built from
# it is tighter, and more returns fall outside. MAD flags more because it *ignores* the tails
# when measuring scale, not because it is sensitive to them.

# %%
fig = go.Figure(
    go.Bar(
        x=[m.upper() for m in method_summary["method"].to_list()],
        y=method_summary["anomalies"].to_list(),
        marker_color=COLORS["slate"],
        text=method_summary["anomalies"].to_list(),
        textposition="outside",
    )
)
fig.update_layout(
    title=f"{DEMO_SYMBOL} return outliers flagged, by method",
    xaxis_title="Method",
    yaxis_title="Events flagged",
    height=400,
)
show_plotly_with_alt(
    fig,
    "A bar chart of three bars labelled MAD, ZSCORE and IQR, each showing how many return "
    "outliers that method flagged. The MAD bar is several times taller than the other two, "
    "which are of similar height to each other, with the IQR bar the shortest.",
)

# %% [markdown]
# ### What the detector cannot tell you
#
# The largest flagged moves in this series are not data faults. They are stock splits and an
# earnings crash, in a panel that stores prices unadjusted. A return-outlier detector locates
# large moves and has no way to attribute them, so the table below is a work queue rather than
# a defect list, and Section 4 shows what it takes to resolve one.

# %%
mad_anomalies = ReturnOutlierDetector(
    config=ReturnOutlierConfig(
        method="mad", threshold=OUTLIER_THRESHOLD, min_samples=OUTLIER_MIN_SAMPLES
    )
).detect(sample_df, symbol=DEMO_SYMBOL)

top_mad = pl.DataFrame(
    [
        {
            "date": _to_date(a.timestamp),
            "return_pct": float(a.value),  # value is already a percentage
        }
        for a in sorted(mad_anomalies, key=lambda x: abs(x.value), reverse=True)[:5]
    ]
)
top_mad

# %%
returns_df = sample_df.with_columns((pl.col("close").pct_change() * 100).alias("return_pct"))
anomaly_dates = {_to_date(a.timestamp) for a in mad_anomalies}
flagged = returns_df.with_columns(
    pl.col("timestamp")
    .map_elements(lambda t: _to_date(t) in anomaly_dates, return_dtype=pl.Boolean)
    .alias("flagged")
)
flagged_pts = flagged.filter(pl.col("flagged"))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=returns_df["timestamp"].to_list(),
        y=returns_df["return_pct"].to_list(),
        mode="lines",
        line=dict(color=COLORS["slate"], width=0.6),
        name="Daily return (%)",
    )
)
fig.add_trace(
    go.Scatter(
        x=flagged_pts["timestamp"].to_list(),
        y=flagged_pts["return_pct"].to_list(),
        mode="markers",
        marker=dict(color=COLORS["copper"], size=5),
        name="MAD-flagged",
    )
)
fig.update_layout(
    title=f"{DEMO_SYMBOL} daily returns, with MAD-flagged days marked",
    xaxis_title="Date",
    yaxis_title="Return (%)",
    height=420,
)
show_plotly_with_alt(
    fig,
    "A line chart of daily percentage returns across nearly four decades, with flagged days "
    "marked as separate points. The line is a dense band a few percent wide with occasional "
    "long excursions, one of them reaching far below the rest of the series. The marked "
    "points sit on the outer edges of the band throughout, and are denser in the earlier and "
    "more volatile stretches than in the later years.",
)

# %% [markdown]
# ### VolumeSpikeDetector
#
# Flags rolling-window volume Z-scores above a threshold.

# %%
volume_anomalies = VolumeSpikeDetector(
    config=VolumeSpikeConfig(
        window=VOLUME_WINDOW,
        threshold=VOLUME_THRESHOLD,
        min_volume=0,
        min_samples=OUTLIER_MIN_SAMPLES,
    )
).detect(sample_df, symbol=DEMO_SYMBOL)

top_volume = pl.DataFrame(
    [
        {
            "date": _to_date(a.timestamp),
            "volume": int(a.value),
            "ratio_vs_avg": (
                a.value / a.metadata["average_volume"] if a.metadata.get("average_volume") else None
            ),
        }
        for a in sorted(volume_anomalies, key=lambda x: x.value, reverse=True)[:5]
    ]
)
print(f"{DEMO_SYMBOL} volume spikes: {len(volume_anomalies)}")
top_volume

# %% [markdown]
# ### PriceStalenessDetector
#
# Flags runs of consecutive identical prices, which is what a feed outage and an untraded
# security both look like from inside the file.

# %%
stale_anomalies = PriceStalenessDetector(
    config=PriceStalenessConfig(max_unchanged_days=STALENESS_DAYS_DETECTOR, check_close_only=False)
).detect(sample_df, symbol=DEMO_SYMBOL)
print(
    f"{DEMO_SYMBOL} stale-price runs longer than {STALENESS_DAYS_DETECTOR} days: "
    f"{len(stale_anomalies)}"
)

# %% [markdown]
# ### AnomalyManager: running all three across the universe
#
# `AnomalyManager` orchestrates the three detectors and reports per symbol.

# %%
anomaly_config = AnomalyConfig(
    enabled=True,
    report_severity_threshold="warning",
    return_outliers=ReturnOutlierConfig(method="mad", threshold=OUTLIER_THRESHOLD),
    volume_spikes=VolumeSpikeConfig(window=VOLUME_WINDOW, threshold=VOLUME_THRESHOLD),
    price_staleness=PriceStalenessConfig(max_unchanged_days=STALENESS_DAYS_VALIDATOR),
)
manager = AnomalyManager(config=anomaly_config)
reports = manager.analyze_batch(datasets)

batch_summary = pl.DataFrame(
    [
        {
            "symbol": sym,
            "rows": len(datasets[sym]),
            "total_anomalies": len(rep.anomalies),
            "critical": len(rep.get_critical_anomalies()),
        }
        for sym, rep in reports.items()
    ]
).with_columns((pl.col("total_anomalies") / pl.col("rows")).alias("share_of_rows"))
batch_summary

# %% [markdown]
# ---
#
# ## Part 3: Population Stability Index (PSI)
#
# PSI measures distribution drift: whether recent data follows the same distribution as a
# historical baseline. Bin the baseline into quantiles, count how much of the current series
# lands in each of those bins, and sum
#
# $$\mathrm{PSI} = \sum_i (p_i^{\text{current}} - p_i^{\text{baseline}})
# \log \frac{p_i^{\text{current}}}{p_i^{\text{baseline}}}$$
#
# The conventional reading thresholds are declared in the parameters cell and printed below
# rather than written into this paragraph, so the code and the description cannot drift apart.

# %%
print("PSI reading thresholds, as declared:")
print(f"  below {PSI_MODERATE:<12} no material change")
print(f"  {f'{PSI_MODERATE} to {PSI_SIGNIFICANT}':<18} moderate shift, investigate")
print(f"  above {PSI_SIGNIFICANT:<12} significant shift, act")

# %% [markdown]
# ### The binning decides what the measure can see
#
# The bin edges come from the baseline's quantiles, so the outermost edges are the baseline's
# own minimum and maximum. A current observation beyond either edge falls outside every bin,
# and a histogram silently drops it.
#
# That is a problem specific to the thing PSI exists to detect. An observation more extreme
# than anything in the baseline is the strongest possible evidence of drift, and closed edges
# discard exactly those. The implementation below opens the outer bins to infinity and reports
# how many observations that recovers, so the cost of the closed version is visible rather
# than assumed to be small.


# %%
def calculate_psi(
    baseline: pl.Series,
    current: pl.Series,
    n_bins: int = PSI_BINS,
    epsilon: float = 1e-6,
    open_outer_bins: bool = True,
) -> tuple[float, pl.DataFrame]:
    """Population Stability Index between a baseline and a current distribution.

    With ``open_outer_bins`` the first and last bins extend to -inf and +inf, so every current
    observation is counted. With it off the outer edges are the baseline's own extremes and
    anything beyond them is dropped, which is the behaviour a plain histogram gives you.
    """
    baseline_clean = baseline.drop_nulls()
    current_clean = current.drop_nulls()

    percentiles = [i / n_bins for i in range(n_bins + 1)]
    bin_edges = [baseline_clean.quantile(p) for p in percentiles]

    unique_edges = [bin_edges[0]]
    for edge in bin_edges[1:]:
        if edge <= unique_edges[-1]:
            edge = unique_edges[-1] + epsilon
        unique_edges.append(edge)
    if open_outer_bins:
        unique_edges = [-np.inf, *unique_edges[1:-1], np.inf]

    baseline_counts = np.histogram(baseline_clean.to_numpy(), bins=unique_edges)[0]
    current_counts = np.histogram(current_clean.to_numpy(), bins=unique_edges)[0]
    baseline_pct = np.maximum(baseline_counts / len(baseline_clean), epsilon)
    current_pct = np.maximum(current_counts / len(current_clean), epsilon)

    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    breakdown = pl.DataFrame(
        {
            "bin": list(range(1, n_bins + 1)),
            "baseline_pct": baseline_pct.round(4),
            "current_pct": current_pct.round(4),
            "psi_contribution": psi_values.round(4),
            "baseline_counted": baseline_counts,
            "current_counted": current_counts,
        }
    )
    return float(np.sum(psi_values)), breakdown


# %%
df = datasets[DEMO_SYMBOL].with_columns((pl.col("close").pct_change() * 100).alias("return_pct"))
midpoint = len(df) // 2
baseline_returns = df["return_pct"][:midpoint]
current_returns = df["return_pct"][midpoint:]

psi_value, psi_breakdown = calculate_psi(baseline_returns, current_returns)
psi_closed, closed_breakdown = calculate_psi(
    baseline_returns, current_returns, open_outer_bins=False
)

_current_n = current_returns.drop_nulls().len()
_dropped = _current_n - int(closed_breakdown["current_counted"].sum())
psi_severity = (
    "no material change"
    if psi_value < PSI_MODERATE
    else "moderate shift"
    if psi_value < PSI_SIGNIFICANT
    else "significant shift"
)
print(
    f"{DEMO_SYMBOL} daily-return PSI: {psi_value:.4f} ({psi_severity})\n"
    f"Baseline: {_to_date(df['timestamp'][0])} to {_to_date(df['timestamp'][midpoint])}\n"
    f"Current:  {_to_date(df['timestamp'][midpoint])} to {_to_date(df['timestamp'][-1])}"
)
print(
    f"\nWith the outer bins closed at the baseline's own range: PSI {psi_closed:.4f}, "
    f"and {_dropped} of {_current_n:,} current observations counted in no bin at all."
)
print(
    "  Those observations lie outside everything the baseline contained, which is the case "
    "the measure is for."
)
psi_breakdown

# %% [markdown]
# ### The baseline bins are not equal, and the largest one drives the result
#
# Quantile bins are supposed to hold an equal share of the baseline each, and the table above
# shows two that do not. Quantile edges cannot split a point mass: if one value occurs often
# enough to straddle an edge, every copy of it lands on one side and the two neighbouring
# bins come out lopsided. Daily returns have exactly such a point mass at zero, which the next
# cell measures on both halves.
#
# This is not bookkeeping. The per-bin contributions show where the statistic comes from.

# %%
_zero_baseline = (baseline_returns.drop_nulls() == 0).sum()
_zero_current = (current_returns.drop_nulls() == 0).sum()
_n_baseline = baseline_returns.drop_nulls().len()
_n_current = current_returns.drop_nulls().len()
print(
    f"Returns of exactly zero: baseline {_zero_baseline:,} of {_n_baseline:,} "
    f"({100 * _zero_baseline / _n_baseline:.2f}%), current {_zero_current:,} of "
    f"{_n_current:,} ({100 * _zero_current / _n_current:.2f}%)"
)

_top = psi_breakdown.sort("psi_contribution", descending=True).row(0, named=True)
print(
    f"Largest single contribution: bin {_top['bin']}, "
    f"{_top['psi_contribution']:.4f} of the total {psi_value:.4f} "
    f"({100 * _top['psi_contribution'] / psi_value:.0f}%)"
)
print(
    f"  its baseline share is {psi_breakdown['baseline_pct'][_top['bin'] - 1]:.1%}, "
    f"not the {1 / PSI_BINS:.0%} an equal split would give"
)

# %% [markdown]
# The zero-return tie group is large in the first half and nearly gone in the second, and the
# bin next to it supplies the majority of the whole statistic. Both facts have the same cause,
# and it is not a change in how far the stock moves: US equities quoted in fractions until
# 2001, so a small move often rounded to no move at all, and after decimalization it did not.
#
# PSI is not wrong here. The distribution genuinely changed. What is wrong is the obvious
# reading of the number, which is that returns grew calmer or wilder. The outer bins do lose
# mass, which is that reading's evidence, and they contribute a small part of the total. The
# statistic is dominated by a tick-size change, and nothing in the headline value says so.
#
# The general form: **a drift score locates a change and cannot attribute one.** Reading the
# per-bin contributions is what turns it into a question worth asking.

# %%
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=psi_breakdown["bin"].to_list(),
        y=(psi_breakdown["baseline_pct"] * 100).to_list(),
        name="Baseline %",
        marker_color=COLORS["slate"],
    )
)
fig.add_trace(
    go.Bar(
        x=psi_breakdown["bin"].to_list(),
        y=(psi_breakdown["current_pct"] * 100).to_list(),
        name="Current %",
        marker_color=COLORS["copper"],
    )
)
fig.add_hline(y=100 / PSI_BINS, line_dash="dash", line_color=COLORS["amber"], line_width=1)
fig.update_layout(
    title=f"{DEMO_SYMBOL} return-decile membership, baseline against current",
    xaxis_title="Baseline decile bin",
    yaxis_title="Share of observations (%)",
    barmode="group",
    height=420,
)
show_plotly_with_alt(
    fig,
    "A grouped bar chart with ten baseline decile bins on the horizontal axis and two bars "
    "per bin, one for the baseline share and one for the current share, against a dashed "
    "reference line at ten percent. Most baseline bars sit on the reference line, but the "
    "fifth is well below it and the sixth well above. The current bars fall below the line in "
    "the outermost bins on both sides and rise above it through the middle, most sharply in "
    "the fifth bin, where the current bar is more than twice the baseline one.",
)

# %% [markdown]
# ---
#
# ## Part 4: Data Hygiene
#
# Gap detection, deduplication, and corporate-action signalling.

# %% [markdown]
# ### Gap Detection
#
# A weekend puts three calendar days between consecutive rows, a holiday next to one puts four,
# and a holiday on both sides of a weekend puts five. The threshold has to sit above whatever
# the calendar produces, so the distribution of step sizes is printed first and the threshold
# is read against it. What matters is whether the distribution has a break: if the ordinary
# cases stop at one value and the next occupied value is well beyond it, a threshold in
# between separates calendar from incident. If they shade into each other, no threshold does.


# %%
def detect_gaps(df: pl.DataFrame, max_gap_days: int = MAX_GAP_DAYS) -> pl.DataFrame:
    """Return rows whose gap from the previous timestamp exceeds `max_gap_days`."""
    return (
        df.sort("timestamp")
        .with_columns(
            pl.col("timestamp").diff().dt.total_days().alias("days_since_prev"),
            pl.col("timestamp").shift(1).alias("prev_timestamp"),
        )
        .filter(pl.col("days_since_prev") > max_gap_days)
        .select(["prev_timestamp", "timestamp", "days_since_prev"])
    )


# %%
gap_sizes = (
    pl.concat(
        [
            df.sort("timestamp")
            .with_columns(pl.col("timestamp").diff().dt.total_days().alias("days"))
            .select("days")
            .drop_nulls()
            for df in datasets.values()
        ]
    )
    .group_by("days")
    .len()
    .sort("days")
)
print("Calendar days between consecutive rows, pooled across the universe:")
print(gap_sizes)

gap_rows = []
for symbol, df in datasets.items():
    for row in detect_gaps(df).iter_rows(named=True):
        gap_rows.append(
            {
                "symbol": symbol,
                "prev_date": _to_date(row["prev_timestamp"]),
                "next_date": _to_date(row["timestamp"]),
                "days": int(row["days_since_prev"]),
            }
        )
gap_table = (
    pl.DataFrame(gap_rows)
    if gap_rows
    else pl.DataFrame({"symbol": [], "prev_date": [], "next_date": [], "days": []})
)
print(f"\nGaps longer than {MAX_GAP_DAYS} days: {gap_table.height}")
gap_table

# %% [markdown]
# One date, and every symbol that was listed at the time carries it: the NYSE closure that
# followed 11 September 2001. The two symbols without it had not listed yet, which the
# coverage table at the top of the notebook shows. A gap that appears in every symbol trading
# at the time is a market event; a gap in one symbol while its peers keep quoting is the one
# worth investigating.

# %% [markdown]
# ### Deduplication
#
# Duplicates appear when a provider re-sends an overlapping date range in an incremental
# update. Whether to keep the first or the last copy is a question about whether the original
# feed or the correction is more trustworthy, and it is not answerable from the row count.

# %%
sample = datasets[DEMO_SYMBOL].head(100)
df_with_dups = pl.concat([sample, sample.head(10)]).sort("timestamp")
print(
    f"Original: {len(df_with_dups)} rows; keep first: "
    f"{df_with_dups.unique(subset=['timestamp'], keep='first').shape[0]} rows; "
    f"keep last: {df_with_dups.unique(subset=['timestamp'], keep='last').shape[0]} rows"
)

# %% [markdown]
# ### Corporate-Action Detection, Scored Against the File's Own Answer
#
# Splits and large special distributions produce overnight returns far outside the ordinary
# range, and a threshold on the overnight return will find them. It will also find every large
# genuine move, and from the return alone the two are indistinguishable.
#
# This panel does not require guessing. It carries `split_ratio` and `ex-dividend` columns
# alongside the prices, so the threshold's output can be scored: every date carrying a split
# ratio other than one is a corporate action, and the flags that match none of them are what
# the threshold costs.


# %%
def detect_corporate_actions(
    df: pl.DataFrame, threshold: float = CORPORATE_ACTION_THRESHOLD
) -> pl.DataFrame:
    """Flag overnight returns whose magnitude exceeds `threshold`."""
    return (
        df.with_columns(pl.col("close").shift(1).alias("prev_close"))
        .with_columns(((pl.col("open") / pl.col("prev_close")) - 1).alias("overnight_return"))
        .filter(pl.col("overnight_return").abs() > threshold)
        .select(["timestamp", "prev_close", "open", "overnight_return"])
    )


# %%
truth = (
    wiki_df.lazy()
    .filter(pl.col("symbol").is_in(SYMBOLS) & (pl.col("split_ratio") != 1.0))
    .select("symbol", "timestamp", "split_ratio")
    .collect()
)

events_rows = []
for symbol, df in datasets.items():
    for row in detect_corporate_actions(df).iter_rows(named=True):
        events_rows.append(
            {
                "symbol": symbol,
                "timestamp": row["timestamp"],
                "date": _to_date(row["timestamp"]),
                "prev_close": round(row["prev_close"], 2),
                "open": round(row["open"], 2),
                "overnight_pct": round(row["overnight_return"] * 100, 1),
            }
        )
events_table = pl.DataFrame(events_rows)

scored = events_table.join(truth, on=["symbol", "timestamp"], how="left")
_caught = truth.join(
    events_table.select("symbol", "timestamp").with_columns(pl.lit(True).alias("flagged")),
    on=["symbol", "timestamp"],
    how="left",
)
_recall_num = _caught.filter(pl.col("flagged").is_not_null()).height
_not_splits = scored.filter(pl.col("split_ratio").is_null())

print(f"Recorded splits in this universe: {truth.height}")
print(f"  flagged by the {CORPORATE_ACTION_THRESHOLD:.0%} overnight threshold: {_recall_num}")
print(f"Flags raised in total: {events_table.height}")
print(
    f"  of which match no recorded split: {_not_splits.height} "
    f"({100 * _not_splits.height / events_table.height:.0f}%)"
)
_not_splits.select("symbol", "date", "prev_close", "open", "overnight_pct").sort("date")

# %% [markdown]
# The threshold misses no split in this universe, and one flag in three matches no split at
# all. One of those is still a corporate action recorded elsewhere in the same file: the GOOGL
# Class C distribution of April 2014 appears in the `ex-dividend` column rather than as a
# split ratio, which is a reminder that "corporate action" is several columns, not one. The
# rest are ordinary large moves, an earnings crash among them.
#
# So the detector's output is a candidate list whose false-positive rate is knowable here only
# because the file carries the answer. On a feed that does not, the same threshold produces
# the same list with no way to score it, and the difference between the two situations is the
# corporate-action feed, not the detection code.

# %%
fig = go.Figure()
for label, frame, color in [
    ("Matches a recorded split", scored.filter(pl.col("split_ratio").is_not_null()), "slate"),
    ("No recorded split", _not_splits, "copper"),
]:
    fig.add_trace(
        go.Scatter(
            x=frame["date"].to_list(),
            y=frame["overnight_pct"].to_list(),
            mode="markers",
            marker=dict(color=COLORS[color], size=8),
            text=frame["symbol"].to_list(),
            hovertemplate="%{text}<br>%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
            name=label,
        )
    )
for sign in (1, -1):
    fig.add_hline(
        y=sign * CORPORATE_ACTION_THRESHOLD * 100,
        line_dash="dash",
        line_color=COLORS["amber"],
        line_width=1,
    )
fig.update_layout(
    title="Flagged overnight returns, split-matched and not",
    xaxis_title="Date",
    yaxis_title="Overnight return (%)",
    height=420,
)
show_plotly_with_alt(
    fig,
    "A scatter plot of flagged overnight returns against date, in two colours for whether the "
    "date matches a recorded split. Dashed reference lines mark the plus and minus threshold. "
    "Most points sit near minus fifty percent and are split-matched, with one far lower. The "
    "unmatched points are spread on both sides, mostly just outside the threshold lines.",
)

# %% [markdown]
# ---
#
# ## Part 5: Production Quality Pipeline
#
# Validation, anomaly detection and deduplication wired together, with quarantine routing for
# critical failures.


# %%
def quality_check_pipeline(
    df: pl.DataFrame,
    symbol: str,
    quarantine_dir: Path,
    anomaly_cfg: AnomalyConfig | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Run validation, then anomaly detection, then dedup; quarantine on critical issues."""
    results: dict = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "input_rows": len(df),
        "actions": [],
    }

    validation = OHLCVValidator().validate(df)
    results["validation_issues"] = len(validation.issues)
    results["actions"].append(
        f"{'PASS' if validation.passed else 'FAIL'} Validation: {len(validation.issues)} issues"
    )

    if not validation.passed:
        critical = [i for i in validation.issues if i.severity.value == "critical"]
        if critical:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            path = quarantine_dir / f"{symbol}_{datetime.now():%Y%m%d_%H%M%S}.parquet"
            df.write_parquet(path)
            results["actions"].append(f"QUARANTINED: {path.name}")

    mgr = AnomalyManager(config=anomaly_cfg) if anomaly_cfg else AnomalyManager()
    report = mgr.analyze(df, symbol)
    results["anomaly_count"] = len(report.anomalies)
    results["actions"].append(f"Anomalies: {len(report.anomalies)}")

    duplicates = len(df) - df["timestamp"].n_unique()
    if duplicates > 0:
        df = df.unique(subset=["timestamp"], keep="last")
        results["actions"].append(f"Removed {duplicates} duplicates")

    results["output_rows"] = len(df)
    results["status"] = (
        "PASS" if validation.passed and not report.get_critical_anomalies() else "REVIEW"
    )
    return df, results


# %%
quarantine_dir = OUTPUT_DIR / "quarantine"
pipeline_rows = []
for symbol, df in datasets.items():
    _, result = quality_check_pipeline(df, symbol, quarantine_dir, anomaly_config)
    pipeline_rows.append(
        {
            "symbol": symbol,
            "status": result["status"],
            "input_rows": result["input_rows"],
            "output_rows": result["output_rows"],
            "validation_issues": result["validation_issues"],
            "anomalies": result["anomaly_count"],
        }
    )
pipeline_summary = pl.DataFrame(pipeline_rows)
pipeline_summary

# %% [markdown]
# ---
#
# ## Key Takeaways
#
# 1. **A validator that passes has told you nothing until it has been shown a fault.** Every
#    symbol here passes, and so would a validator with its checks switched off. Injecting one
#    fault of each kind and confirming which check fires is what separates the two, and it is
#    three lines of setup.
#
# 2. **One threshold across three methods is three different cutoffs.** MAD, Z-score and IQR
#    each multiply the threshold by their own scale estimate, so the numbers they flag are not
#    comparable until those cutoffs are printed in the units of the data. They are printed
#    above, and the ordering of the flag counts is the ordering of the cutoffs.
#
# 3. **MAD flags more because it ignores the tails, not because it sees them.** Its scale
#    estimate is a median of deviations, which extreme days cannot move, so on a heavy-tailed
#    series it comes out well below the standard deviation and the band built from it is
#    correspondingly narrow. The intuitive explanation runs the other way and gets the
#    mechanism backwards.
#
# 4. **PSI's binning discards the observations PSI exists to find.** Bin edges taken from
#    baseline quantiles end at the baseline's own extremes, so a current observation more
#    extreme than anything seen before falls in no bin and is dropped by the histogram. The
#    count dropped is printed above. Opening the outer bins costs nothing and removes a blind
#    spot aimed squarely at the strongest evidence of drift.
#
# 5. **An anomaly detector produces a work queue, and cannot resolve one.** The overnight-return
#    threshold misses none of this universe's recorded splits, and better than a third of its
#    flags match no split at all: one corporate action recorded in a different column of the
#    same file, and several ordinary large moves. Scoring it was possible only because this file carries
#    the split and dividend columns; on a feed that does not, the same list arrives with no way
#    to grade it.
#
# 6. **An unadjusted panel makes correct detectors fire on real events.** Either adjust
#    upstream or give the pipeline a corporate-action source to consult before it quarantines
#    anything, because the alternative is quarantining the days the market actually moved.
#
# **Next**: `14_point_in_time_validation` adds the temporal dimension, bitemporal queries, on
# top of the structural checks shown here.
