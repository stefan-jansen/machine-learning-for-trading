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
# Demonstrate the four pillars of a production data-quality pipeline applied to
# US equities daily OHLCV: structural validation (OHLC invariants, nulls,
# duplicates), anomaly detection (return outliers, volume spikes, price
# staleness), distribution drift (PSI), and ingestion hygiene (gaps,
# duplicates, corporate-action detection). All checks come from the
# `ml4t.data.validation` and `ml4t.data.anomaly` modules.
#
# ## Learning Objectives
# - Run `OHLCVValidator` and read its issue report.
# - Compare MAD / Z-score / IQR thresholds for return-outlier detection.
# - Compute Population Stability Index for drift monitoring and read its bins.
# - Detect ingestion gaps, duplicates, and likely corporate actions, and wire
#   the pieces together into a quarantine-aware pipeline.
#
# ## Book reference
# Chapter 2, §2.3 (data quality framework). Downstream chapters that consume
# the cleaned panel: `14_point_in_time_validation` (bitemporal hygiene),
# `15_survivorship_bias_detection`, `17_complete_pipeline`.
#
# ## Prerequisites
# - Quandl/Wiki US equities parquet materialized under `ML4T_DATA_PATH`
#   (the legacy dataset; ends 2018-03-27).
# - Loader `data.load_us_equities`.
# - Library packages `ml4t.data.validation` and `ml4t.data.anomaly`.

# %%
"""Data Quality Framework — Validation, anomaly detection, and remediation."""

import warnings

warnings.filterwarnings("ignore")

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
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

# The library emits per-symbol DEBUG logs that drown the cell output;
# silence to WARNING so the notebook prints only call results.
logging.getLogger("ml4t.data.anomaly").setLevel(logging.WARNING)


def _to_date(value: object) -> object:
    """Normalize a polars timestamp scalar to a datetime.date for printing."""
    return value.date() if hasattr(value, "date") else value


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %%
OUTPUT_DIR = get_output_dir(2, "quality")

# Five large-cap symbols from the legacy Wiki/Quandl dataset (1962–2018).
# FB (not META) is the canonical Facebook ticker in this vintage.
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "FB"]

# %% [markdown]
# ## Load Sample Data
#
# Use the pre-2018 Wiki/Quandl US equities panel. The validation and anomaly
# detectors operate on any DataFrame with `timestamp / open / high / low /
# close / volume`.

# %%
wiki_df = load_us_equities()
print(
    f"US equities loaded: {len(wiki_df):,} rows; "
    f"{wiki_df['timestamp'].min()} → {wiki_df['timestamp'].max()}"
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
    {"symbol": list(datasets), "rows": [len(df) for df in datasets.values()]}
)
per_symbol_rows

# %% [markdown]
# ---
#
# ## Part 1: OHLC Invariant Validation
#
# **OHLC Invariants** are mathematical relationships that MUST hold for valid
# market data:
#
# | Invariant       | Meaning              |
# |-----------------|----------------------|
# | High ≥ Low      | by definition        |
# | High ≥ Open, Close | high is the maximum |
# | Low ≤ Open, Close  | low is the minimum  |
# | Prices > 0      | no negative prices   |
# | Volume ≥ 0      | no negative volume   |
#
# Violations indicate provider errors, transmission corruption, or incorrect
# adjustments.

# %%
validator = OHLCVValidator(
    check_nulls=True,
    check_price_consistency=True,
    check_negative_prices=True,
    check_negative_volume=True,
    check_duplicate_timestamps=True,
    check_chronological_order=True,
    check_price_staleness=True,
    check_extreme_returns=True,
    max_return_threshold=0.5,  # flag |return| > 50%
    staleness_threshold=5,  # flag 5+ identical-price days
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
# ### Validation on Dirty Data
#
# Inject three known faults into AAPL and re-run the validator to see what it
# catches.

# %%
clean_df = datasets["AAPL"]
highs = clean_df["high"].to_numpy().copy()
lows = clean_df["low"].to_numpy().copy()
volumes = clean_df["volume"].to_numpy().copy()
closes = clean_df["close"].to_numpy().copy()

# Fault 1: high < low at rows 10–12
highs[10:13] = lows[10:13] - 1.0
# Fault 2: negative volume at row 20
volumes[20] = -1000
# Fault 3: null close at row 30
closes[30] = np.nan

dirty_df = pl.DataFrame(
    {
        "timestamp": clean_df["timestamp"],
        "symbol": clean_df["symbol"],
        "open": clean_df["open"],
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }
)

dirty_result = validator.validate(dirty_df)
print(f"Validation passed: {dirty_result.passed}")
print(f"Critical: {dirty_result.critical_count}, Errors: {dirty_result.error_count}")

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
# Validation checks data correctness; anomaly detection finds unusual patterns
# in otherwise structurally-valid data:
#
# | Detector              | What it finds                | Method                 |
# |-----------------------|-------------------------------|------------------------|
# | ReturnOutlierDetector | flash crashes, splits, pumps  | MAD, Z-score, IQR      |
# | VolumeSpikeDetector   | unusual trading activity      | rolling Z-score        |
# | PriceStalenessDetector| data gaps, illiquid securities| consecutive-unchanged  |

# %% [markdown]
# ### ReturnOutlierDetector
#
# Flags returns whose magnitude exceeds a threshold under three statistics:
# **MAD** (median absolute deviation, robust to extreme tails), **Z-score**
# (Gaussian, threshold-sensitive to fat tails), and **IQR** (interquartile
# range, distribution-free).

# %%
sample_df = datasets["AAPL"]
methods = ["mad", "zscore", "iqr"]

method_summary = pl.DataFrame(
    [
        {
            "method": m,
            "anomalies": len(
                ReturnOutlierDetector(
                    config=ReturnOutlierConfig(method=m, threshold=3.0, min_samples=20)
                ).detect(sample_df, symbol="AAPL")
            ),
        }
        for m in methods
    ]
)
method_summary

# %% [markdown]
# MAD flags the most events because it is more sensitive in the tails of a
# heavy-tailed return distribution. The biggest "anomalies" in the AAPL series
# below are *real* events: stock splits and an earnings-driven crash, not
# data-quality issues. The detector cannot distinguish — that is the operator's
# job downstream.

# %%
mad_anomalies = ReturnOutlierDetector(
    config=ReturnOutlierConfig(method="mad", threshold=3.0, min_samples=20)
).detect(sample_df, symbol="AAPL")

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

# %% [markdown]
# ### VolumeSpikeDetector
#
# Flags rolling-window volume Z-scores above a threshold.

# %%
volume_anomalies = VolumeSpikeDetector(
    config=VolumeSpikeConfig(window=20, threshold=3.0, min_volume=0, min_samples=20)
).detect(sample_df, symbol="AAPL")

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
print(f"Found {len(volume_anomalies)} AAPL volume spikes")
top_volume

# %% [markdown]
# ### PriceStalenessDetector
#
# Flags runs of consecutive identical prices — a strong signal of feed
# outages or illiquid securities.

# %%
stale_anomalies = PriceStalenessDetector(
    config=PriceStalenessConfig(max_unchanged_days=3, check_close_only=False)
).detect(sample_df, symbol="AAPL")
print(f"AAPL stale-price periods (≥4 consecutive days): {len(stale_anomalies)}")

# %% [markdown]
# ### AnomalyManager: Production Pipeline
#
# `AnomalyManager` orchestrates the three detectors and provides batch
# analysis across symbols.

# %%
anomaly_config = AnomalyConfig(
    enabled=True,
    report_severity_threshold="warning",
    return_outliers=ReturnOutlierConfig(method="mad", threshold=3.0),
    volume_spikes=VolumeSpikeConfig(window=20, threshold=3.0),
    price_staleness=PriceStalenessConfig(max_unchanged_days=5),
)
manager = AnomalyManager(config=anomaly_config)
reports = manager.analyze_batch(datasets)

batch_summary = pl.DataFrame(
    [
        {
            "symbol": sym,
            "total_anomalies": len(rep.anomalies),
            "critical": len(rep.get_critical_anomalies()),
        }
        for sym, rep in reports.items()
    ]
)
batch_summary

# %% [markdown]
# ---
#
# ## Part 3: Population Stability Index (PSI)
#
# **PSI** measures distribution drift — whether recent data follows the same
# distribution as a historical baseline. Useful for detecting regime changes,
# data-source switches, and market-structure shifts.
#
# Bin the baseline series into deciles, recompute the same bin boundaries on
# the current series, and sum
#
# $$\mathrm{PSI} = \sum_i (p_i^{\text{current}} - p_i^{\text{baseline}})
# \log \frac{p_i^{\text{current}}}{p_i^{\text{baseline}}}$$
#
# | PSI       | Interpretation                  |
# |-----------|---------------------------------|
# | < 0.1     | no significant change           |
# | 0.1–0.25  | moderate shift, investigate     |
# | > 0.25    | significant shift, action needed|


# %%
def calculate_psi(
    baseline: pl.Series, current: pl.Series, n_bins: int = 10, epsilon: float = 1e-6
) -> tuple[float, pl.DataFrame]:
    """Population Stability Index between baseline and current distributions."""
    baseline_clean = baseline.drop_nulls()
    current_clean = current.drop_nulls()

    percentiles = [i * 100 / n_bins for i in range(n_bins + 1)]
    bin_edges = [baseline_clean.quantile(p / 100) for p in percentiles]

    # Ensure unique edges
    unique_edges = [bin_edges[0]]
    for edge in bin_edges[1:]:
        if edge <= unique_edges[-1]:
            edge = unique_edges[-1] + epsilon
        unique_edges.append(edge)

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
        }
    )
    return float(np.sum(psi_values)), breakdown


# %%
df = datasets["AAPL"].with_columns((pl.col("close").pct_change() * 100).alias("return_pct"))
midpoint = len(df) // 2
baseline_returns = df["return_pct"][:midpoint]
current_returns = df["return_pct"][midpoint:]

psi_value, psi_breakdown = calculate_psi(baseline_returns, current_returns)
psi_severity = (
    "no significant change"
    if psi_value < 0.1
    else "moderate shift"
    if psi_value < 0.25
    else "significant shift"
)
print(
    f"AAPL daily-return PSI: {psi_value:.4f} ({psi_severity})\n"
    f"Baseline: {_to_date(df['timestamp'][0])} → {_to_date(df['timestamp'][midpoint])}\n"
    f"Current:  {_to_date(df['timestamp'][midpoint])} → {_to_date(df['timestamp'][-1])}"
)
psi_breakdown

# %% [markdown]
# ---
#
# ## Part 4: Data Hygiene
#
# Beyond validation and anomaly detection, ingestion hygiene covers gap
# detection, deduplication, and corporate-action signaling.

# %% [markdown]
# ### Gap Detection
#
# Holidays produce expected 3–4 calendar-day gaps; provider outages and
# delistings produce longer ones. The 9/11 close (2001-09-11 → 09-17) is the
# only > 5-day gap visible in this universe.


# %%
def detect_gaps(df: pl.DataFrame, max_gap_days: int = 5) -> pl.DataFrame:
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
gap_rows = []
for symbol, df in datasets.items():
    gaps = detect_gaps(df, max_gap_days=5)
    for row in gaps.iter_rows(named=True):
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
gap_table

# %% [markdown]
# ### Deduplication
#
# Duplicates appear when a provider re-sends overlapping date ranges in
# incremental updates. The choice of `keep="first"` vs `"last"` depends on
# whether you trust the original feed or the correction.

# %%
sample = datasets["AAPL"].head(100)
df_with_dups = pl.concat([sample, sample.head(10)]).sort("timestamp")
print(
    f"Original: {len(df_with_dups)} rows; keep first: "
    f"{df_with_dups.unique(subset=['timestamp'], keep='first').shape[0]} rows; "
    f"keep last: {df_with_dups.unique(subset=['timestamp'], keep='last').shape[0]} rows"
)

# %% [markdown]
# ### Corporate-Action Detection
#
# Splits, reverse splits, and large special dividends produce overnight
# returns that trip the >25 % threshold below. The Wiki/Quandl panel here is
# *not* split-adjusted, so AAPL splits (1987-06-16 2:1, 2000-06-21 2:1,
# 2005-02-28 2:1, 2014-06-09 7:1) and the GOOGL Class C distribution
# (2014-04-03) appear as flagged events.


# %%
def detect_corporate_actions(df: pl.DataFrame, threshold: float = 0.25) -> pl.DataFrame:
    """Flag overnight returns whose magnitude exceeds `threshold`."""
    return (
        df.with_columns(pl.col("close").shift(1).alias("prev_close"))
        .with_columns(((pl.col("open") / pl.col("prev_close")) - 1).alias("overnight_return"))
        .filter(pl.col("overnight_return").abs() > threshold)
        .select(["timestamp", "prev_close", "open", "overnight_return"])
    )


# %%
events_rows = []
for symbol, df in datasets.items():
    for row in detect_corporate_actions(df, threshold=0.25).iter_rows(named=True):
        events_rows.append(
            {
                "symbol": symbol,
                "date": _to_date(row["timestamp"]),
                "prev_close": round(row["prev_close"], 2),
                "open": round(row["open"], 2),
                "overnight_pct": round(row["overnight_return"] * 100, 1),
            }
        )
events_table = pl.DataFrame(events_rows)
print(f"Potential corporate-action events across {len(datasets)} symbols: {len(events_table)}")
events_table.head(10)

# %% [markdown]
# ---
#
# ## Part 5: Production Quality Pipeline
#
# Wire validation, anomaly detection, and deduplication together with
# quarantine routing for critical failures.


# %%
def quality_check_pipeline(
    df: pl.DataFrame,
    symbol: str,
    quarantine_dir: Path,
    anomaly_cfg: AnomalyConfig | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Run validation → anomaly detection → dedup; quarantine on critical issues."""
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
# Quality-pipeline profile across five large-cap symbols (AAPL, MSFT, GOOGL,
# NVDA, FB) on the legacy Wiki/Quandl panel (1962-01 → 2018-03).
#
# ### Quantitative Findings
# - **Structural validation**: All five symbols pass the OHLCVValidator with
#   only an `extreme_returns` warning — that warning fires on the same large
#   moves the corporate-action detector flags below.
# - **Synthetic-fault detection**: After injecting `high < low`, negative
#   volume, and a null close into AAPL, the validator returns `passed=False`
#   with 4 errors / 0 critical / 1 warning, identifying every fault.
# - **Return-outlier detector method spread (AAPL, threshold = 3.0)**:
#   MAD = 336 events, Z-score = 83, IQR = 69. MAD is most sensitive in the
#   tails because the AAPL return distribution is heavy-tailed; the top MAD
#   events are the four AAPL stock splits and the 2000-09-29 earnings crash.
# - **Volume spikes**: 180 AAPL events at threshold 3.0; the largest is
#   2014-09-09 (~3.2× the 20-day average), the eve of the iPhone 6 launch.
# - **Price staleness**: 0 events on AAPL — the legacy panel is clean for
#   this symbol.
# - **PSI (AAPL daily returns, halves split)**: 0.118 ⇒ moderate shift.
#   Baseline 1980-12-12 → 1999-07-21, current 1999-07-21 → 2018-03-27 — the
#   regime change between the two halves is detectable but not extreme.
# - **Hygiene**: The only >5-day gap is the 2001-09-10 → 09-17 NYSE closure
#   following 9/11, present in the three symbols listed before that date
#   (AAPL, MSFT, NVDA); GOOGL and FB IPO'd later. Across the universe the
#   corporate-action detector flags 27 events at the 25% threshold,
#   dominated by AAPL/MSFT/NVDA splits and the GOOGL Class C distribution
#   (2014-04-03).
#
# ### Implications for Practitioners
# - **Pre-adjustment matters**: An unadjusted historical panel makes the
#   anomaly detectors fire on real corporate events. Either adjust upstream
#   or maintain a corporate-action whitelist that the pipeline consults
#   before quarantining.
# - **MAD over Z-score**: For heavy-tailed financial returns, MAD's tail
#   sensitivity is a feature; the operator must classify each event rather
#   than rely on a single auto-threshold.
# - **PSI as alarm, not classifier**: A moderate shift between two 19-year
#   halves is unsurprising; PSI is most useful at the rolling-30-day
#   timescale where regime changes manifest faster.
#
# **Next**: `14_point_in_time_validation` adds the temporal dimension
# (bitemporal queries) on top of the structural checks shown here.
