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
# # Drift Detection for Production ML Strategies
# **Docker image**: `ml4t`
#
# **Purpose**: Walk through the `ml4t.diagnostic.evaluation.drift` toolkit (PSI,
# Wasserstein distance, domain classifier, unified `analyze_drift`) on real
# ETF momentum features split across calm and stressed market windows and a
# real crypto perpetuals + funding-premium panel across market regimes, and
# wrap the result in a `ProductionDriftMonitor` with explicit alert thresholds.
#
# **Learning objectives**:
# 1. Compute Population Stability Index (PSI) and read its bin-level breakdown.
# 2. Compare PSI to the Wasserstein distance interpretation.
# 3. Use a domain classifier (logistic regression on stacked reference + test
#    samples) for multivariate drift detection.
# 4. Translate per-method drift scores into alert levels and a retraining
#    recommendation via `ProductionDriftMonitor.should_retrain()`.
#
# **Book reference**: §19.7 (Adaptive Risk Controls) and §19.8 (Kill Switches
# and Governance, esp. PSI thresholds 0.10 / 0.25).
#
# **Prerequisites**: Feature engineering and distribution diagnostics from
# Chapter 8, an `ml4t-diagnostic` install, and the canonical PSI threshold
# rules (PSI < 0.10 stable / 0.10-0.25 monitor / > 0.25 investigate). Note that
# these are rules of thumb; thresholds must be calibrated per feature
# distribution and sample-size regime before production use.
#
# **Data**: Real ETF and crypto feature panels. ETF features (momentum, volatility,
# RSI, volume ratio) come from `load_etfs` and are split by calendar window
# (calm 2017 vs stressed 2020). Crypto features (funding rate, 1h volatility,
# volume ratio, 24h momentum) come from `load_crypto_perps` joined with
# `load_crypto_premium`, sliced by week to expose real regime changes.

# %%
"""Drift Detection for Production ML Strategies — detect feature and concept drift using PSI, Wasserstein distance, and domain classifiers."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

# ml4t-diagnostic imports
from ml4t.diagnostic.evaluation.drift import (
    DomainClassifierResult,
    PSIResult,
    WassersteinResult,
    analyze_drift,
    compute_domain_classifier_drift,
    compute_psi,
    compute_wasserstein_distance,
)

from data import load_crypto_perps, load_crypto_premium, load_etfs
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
N_SAMPLES = 10000
SEED = 42

# %% [markdown]
# ---
#
# ## Part 1: Understanding Drift Types
#
# ### Covariate Drift vs Concept Drift
#
# **Covariate Drift**: P(X) changes but P(Y|X) stays the same
# - Example: Volatility increases but signal-return relationship unchanged
# - Solution: Retrain on recent data
#
# **Concept Drift**: P(Y|X) changes
# - Example: Momentum factor stops working (regime change)
# - Solution: New model architecture or feature set
#
# **Prior Drift**: P(Y) changes
# - Example: Market goes from bull to bear (different return distribution)
# - Solution: Regime-aware models

# %% [markdown]
# We build a real ETF feature panel — one row per (symbol, trading day) carrying
# 20- and 60-day momentum, 20-day return volatility, the 14-day RSI, and a
# 20-day volume ratio — then split it by calendar window. The calm-vs-stress
# contrast (2017 vs 2020) makes the metrics work on a genuine regime shift
# rather than a fabricated one.

# %%
set_global_seeds(SEED)


# %%
def etf_feature_panel(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute per-symbol momentum, volatility, RSI, and volume-ratio features."""

    return (
        prices.sort(["symbol", "timestamp"])
        .with_columns(ret=pl.col("close").pct_change().over("symbol"))
        .with_columns(
            momentum_20=(pl.col("close") / pl.col("close").shift(20) - 1).over("symbol"),
            momentum_60=(pl.col("close") / pl.col("close").shift(60) - 1).over("symbol"),
            volatility_20=pl.col("ret").rolling_std(20).over("symbol"),
            volume_ratio=(pl.col("volume") / pl.col("volume").rolling_mean(20)).over("symbol"),
            _gain=pl.when(pl.col("ret") > 0).then(pl.col("ret")).otherwise(0.0),
            _loss=pl.when(pl.col("ret") < 0).then(-pl.col("ret")).otherwise(0.0),
        )
        .with_columns(
            _avg_gain=pl.col("_gain").rolling_mean(14).over("symbol"),
            _avg_loss=pl.col("_loss").rolling_mean(14).over("symbol"),
        )
        .with_columns(rsi_14=100 - 100 / (1 + pl.col("_avg_gain") / pl.col("_avg_loss")))
        .drop(["_gain", "_loss", "_avg_gain", "_avg_loss"])
    )


# %%
ETF_FEATURES = ["momentum_20", "momentum_60", "volatility_20", "rsi_14", "volume_ratio"]
etf_panel = etf_feature_panel(load_etfs())


def etf_window(start: str, end: str, cols: list[str] | None = None) -> pd.DataFrame:
    """Slice the ETF panel by date window and return a pandas frame of features."""

    cols = cols or ETF_FEATURES
    return (
        etf_panel.filter(
            (pl.col("timestamp") >= pl.lit(start).str.to_date())
            & (pl.col("timestamp") < pl.lit(end).str.to_date())
        )
        .select(cols)
        .drop_nulls()
        .to_pandas()
    )


# Calm 2017 baseline vs stressed 2020 (COVID) drift window
baseline_features = etf_window(
    "2017-01-01", "2018-01-01", ["momentum_20", "volatility_20", "rsi_14", "volume_ratio"]
)
covariate_drift_features = etf_window(
    "2020-01-01", "2021-01-01", ["momentum_20", "volatility_20", "rsi_14", "volume_ratio"]
)

print(f"Baseline (2017): {len(baseline_features):,} symbol-days")
print(f"Drift   (2020): {len(covariate_drift_features):,} symbol-days")
print("\nBaseline period statistics")
baseline_features.describe().round(4)

# %%
print("Drifted period statistics")
covariate_drift_features.describe().round(4)

# %% [markdown]
# Comparing 2017 to 2020 surfaces a real covariate shift: 20-day return volatility roughly
# doubles, momentum widens, and RSI / volume ratio stay close to their 2017 distributions.
# The next sections test whether PSI, Wasserstein distance, and a domain classifier quantify
# this regime change consistently.

# %% [markdown]
# ---
#
# ## Part 2: Population Stability Index (PSI)
#
# PSI is the industry standard for detecting distribution shift:
#
# $$PSI = \sum_{i=1}^{n} (p_i^{\text{actual}} - p_i^{\text{expected}}) \cdot \ln\left(\frac{p_i^{\text{actual}}}{p_i^{\text{expected}}}\right)$$
#
# ### PSI Interpretation
# - **PSI < 0.1**: No significant drift
# - **0.1 ≤ PSI < 0.25**: Moderate drift - investigate
# - **PSI ≥ 0.25**: Significant drift - action required

# %%
# Compute PSI for each feature
features = ["momentum_20", "volatility_20", "rsi_14", "volume_ratio"]
psi_results = {}

for feature in features:
    result = compute_psi(
        reference=baseline_features[feature].values,
        test=covariate_drift_features[feature].values,
        n_bins=10,
    )
    psi_results[feature] = result

    # Interpret result
    if result.psi < 0.1:
        status = "[OK] Stable"
    elif result.psi < 0.25:
        status = "[WARN] Moderate drift"
    else:
        status = "[ALERT] Significant drift"

    print(f"{feature:20} PSI={result.psi:.4f}  {status}")

# %% [markdown]
# PSI flags the real 2017-to-2020 distribution shift in these ETF features. The largest values should
# appear on the features with both location and dispersion changes, which is why the next plot
# drills into the worst offender bin by bin.

# %% [markdown]
# The PSI chart pairs overall drift magnitude with the contribution of each histogram bin. That
# separation matters in practice because it tells us whether drift comes from a broad regime move
# or from a small number of unstable tails.


# %%
def plot_psi_breakdown(result: PSIResult, feature_name: str):
    """Visualize PSI contribution by bin."""

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Distribution Comparison", "PSI Contribution by Bin")
    )

    # Distribution comparison
    bins = list(range(len(result.reference_percents)))

    fig.add_trace(
        go.Bar(
            x=bins,
            y=result.reference_percents,
            name="Reference",
            marker_color="steelblue",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=bins, y=result.test_percents, name="Current", marker_color="coral", opacity=0.7),
        row=1,
        col=1,
    )

    # PSI contribution by bin
    fig.add_trace(
        go.Bar(
            x=bins,
            y=result.bin_psi,
            name="PSI Contribution",
            marker_color=["red" if c > result.psi / len(bins) else "gray" for c in result.bin_psi],
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title=f"{feature_name} - PSI = {result.psi:.4f}", showlegend=True, height=400)

    return fig


# %% [markdown]
# The highest-PSI feature should show a concentrated set of bins driving the alert rather than
# uniform small deviations everywhere. That pattern usually points to a regime shift worth
# investigating instead of routine sampling noise.

# %%
# Visualize feature with highest PSI
worst_feature = max(psi_results, key=lambda x: psi_results[x].psi)
fig = plot_psi_breakdown(psi_results[worst_feature], worst_feature)
fig.show()

# %% [markdown]
# The bar comparison confirms where the distribution moved. In production, this is the point
# where a risk review would check whether the drift is economically benign or likely to impair
# signal calibration.

# %% [markdown]
# ---
#
# ## Part 3: Wasserstein Distance
#
# Also known as Earth Mover's Distance (EMD), Wasserstein distance measures the minimum "work" required to transform one distribution into another:
#
# $$W_p(P, Q) = \left( \inf_{\gamma \in \Gamma(P,Q)} \int ||x-y||^p d\gamma(x,y) \right)^{1/p}$$
#
# ### Advantages over PSI
# - **Continuous**: No binning required
# - **Geometric intuition**: Respects feature ordering
# - **Robust**: Less sensitive to outliers

# %%
# Compute Wasserstein distance for each feature
wasserstein_results = {}

for feature in features:
    result = compute_wasserstein_distance(
        reference=baseline_features[feature].values, test=covariate_drift_features[feature].values
    )
    wasserstein_results[feature] = result

    # Use drifted status from API
    if not result.drifted:
        status = "[OK] No drift"
    else:
        status = "[ALERT] Drift detected"

    print(f"{feature:20} W={result.distance:.4f}  p={result.p_value:.4f}  {status}")

# %% [markdown]
# Wasserstein distance gives the same drift story without histogram bins. That makes it a useful
# cross-check when PSI is sensitive to the chosen discretization or when the feature has a clear
# ordering that we want the metric to respect.

# %% [markdown]
# The next helper compares densities and empirical CDFs side by side. The CDF gap is especially
# helpful for explaining why Wasserstein distance rises when the entire distribution shifts.


# %%
def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted values and their empirical CDF."""

    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, cdf


# %% [markdown]
# This visualization emphasizes shape changes rather than PSI bin assignments. For continuous
# trading features such as volatility, it often gives the cleaner diagnostic view.


# %%
def plot_wasserstein_comparison(
    reference: np.ndarray, current: np.ndarray, result: WassersteinResult, feature_name: str
):
    """Visualize Wasserstein distance with CDFs."""

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Density Comparison", "CDF Comparison (Wasserstein = Area Between)"),
    )

    x_range = np.linspace(
        min(reference.min(), current.min()), max(reference.max(), current.max()), 200
    )

    kde_ref = stats.gaussian_kde(reference)
    kde_cur = stats.gaussian_kde(current)

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde_ref(x_range),
            name="Reference",
            fill="tozeroy",
            fillcolor="rgba(70,130,180,0.3)",
            line=dict(color="steelblue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde_cur(x_range),
            name="Current",
            fill="tozeroy",
            fillcolor="rgba(255,127,80,0.3)",
            line=dict(color="coral"),
        ),
        row=1,
        col=1,
    )

    ref_sorted, ref_cdf = empirical_cdf(reference)
    cur_sorted, cur_cdf = empirical_cdf(current)

    fig.add_trace(
        go.Scatter(x=ref_sorted, y=ref_cdf, name="Ref CDF", line=dict(color="steelblue", width=2)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=cur_sorted, y=cur_cdf, name="Cur CDF", line=dict(color="coral", width=2)),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"{feature_name} - Wasserstein = {result.distance:.4f} (p={result.p_value:.4f})",
        height=400,
        showlegend=True,
    )

    return fig


# %% [markdown]
# Volatility is a useful example because both the center and the spread changed. If the density
# and CDF panels separate cleanly, the Wasserstein signal is explaining a genuine regime move.

# %%
# Visualize
fig = plot_wasserstein_comparison(
    baseline_features["volatility_20"].values,
    covariate_drift_features["volatility_20"].values,
    wasserstein_results["volatility_20"],
    "volatility_20",
)
fig.show()

# %% [markdown]
# The volatility comparison shows why transport distance is intuitive for readers: the current
# sample is not just noisier, it is shifted upward across most of the support.

# %% [markdown]
# ---
#
# ## Part 4: Domain Classifier Drift Detection
#
# Train a classifier to distinguish reference from current data. If it cannot
# separate them above chance, the test does not reject the null of equal joint
# distributions at this sample size — absence of separation at this capacity
# is not proof the distribution has not moved.
#
# ### How It Works
# 1. Label reference data as 0, current data as 1
# 2. Train a classifier (usually gradient boosting)
# 3. Measure AUC:
#    - **AUC ≈ 0.5**: No drift (can't distinguish)
#    - **AUC > 0.7**: Significant drift (easy to distinguish)

# %%
# Domain classifier for multivariate drift
result = compute_domain_classifier_drift(
    reference=baseline_features[features].values,
    test=covariate_drift_features[features].values,
    cv_folds=5,
    random_state=42,
)

print("Domain Classifier Results:")
print(f"  AUC: {result.auc:.4f} ± {result.cv_auc_std:.4f}")
print(f"  Drifted: {result.drifted}")

if result.auc > 0.7:
    print("\n[ALERT] Significant multivariate drift detected (AUC > 0.7)")
elif result.auc > 0.6:
    print("\n[WARN] Moderate multivariate drift (AUC > 0.6)")
else:
    print("\n[OK] No significant drift (AUC near 0.5)")

# Feature importance (which features contribute most to drift?)
print("\nFeature importance for drift:")
# feature_importances is now a polars DataFrame
fi_df = result.feature_importances
for row in fi_df.sort("importance", descending=True).iter_rows(named=True):
    print(f"  {row['feature']:20} {row['importance']:.4f}")

# %% [markdown]
# The domain classifier complements univariate metrics by asking whether the full feature vector
# looks like it came from a new regime. A materially elevated AUC means the joint distribution
# has changed enough that the model can separate old from new observations.

# %% [markdown]
# Feature importance from the classifier tells us where the multivariate signal is coming from.
# That helps convert a generic drift alert into a concrete investigation queue for the risk team.


# %%
def plot_domain_classifier_results(result: DomainClassifierResult, feature_names: list[str]):
    """Visualize domain classifier drift detection."""

    # Feature importance - from polars DataFrame
    fi_df = result.feature_importances.sort("importance", descending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=fi_df["feature"].to_list(),
            x=fi_df["importance"].to_list(),
            orientation="h",
            marker_color="steelblue",
        )
    )

    drift_status = "[ALERT] DRIFT DETECTED" if result.drifted else "[OK] No Drift"
    fig.update_layout(
        title=f"Feature Importance for Drift Detection<br>AUC: {result.auc:.4f} | {drift_status}",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
    )

    return fig


# %% [markdown]
# We expect the top-ranked features here to overlap with the largest PSI and Wasserstein moves,
# but not necessarily match perfectly because the classifier captures interactions too.

# %%
fig = plot_domain_classifier_results(result, features)
fig.show()

# %% [markdown]
# If one feature dominates this chart, the drift response can start with that signal. A more
# diffuse profile suggests a broader market regime shift and usually argues for a wider review.

# %% [markdown]
# ---
#
# ## Part 5: Unified Drift Analysis
#
# The `analyze_drift()` function provides a comprehensive analysis combining all methods.

# %%
# Comprehensive drift analysis
drift_summary = analyze_drift(
    reference=baseline_features,
    test=covariate_drift_features,
    features=features,
    methods=["psi", "wasserstein", "domain_classifier"],
    psi_config={"psi_threshold_red": 0.25},
    domain_classifier_config={"threshold": 0.7},
)

print("=" * 60)
print("COMPREHENSIVE DRIFT ANALYSIS REPORT")
print("=" * 60)
print(f"\nOverall Drift Detected: {'[ALERT] YES' if drift_summary.overall_drifted else '[OK] NO'}")
print(f"Drifted Features: {drift_summary.drifted_features}")
print(f"Methods Used: {drift_summary.methods_used}")

print("\nPer-Feature Results:")
for fr in drift_summary.feature_results:
    status = "[ALERT]" if fr.drifted else "[OK]"
    psi_val = f"{fr.psi_result.psi:.4f}" if fr.psi_result else "N/A"
    print(f"  {fr.feature:20} PSI={psi_val:>8} {status}")

if drift_summary.domain_classifier_result:
    print(f"\nDomain Classifier AUC: {drift_summary.domain_classifier_result.auc:.4f}")
    print(f"Drifted: {drift_summary.domain_classifier_result.drifted}")

# %% [markdown]
# The unified report should agree with the method-specific sections above. Consistency across the
# three diagnostics is what gives confidence that the alert is meaningful rather than metric noise.

# %% [markdown]
# ---
#
# ## Part 6: Case Study - ETF Rotational Momentum
#
# Monitor feature drift across quarters for a daily equity momentum strategy.

# %%
# Slice the ETF feature panel into real calendar quarters spanning the COVID
# regime shift. Q4 2019 is the calm baseline; Q1-Q4 2020 carry the crisis and
# the partial recovery.
QUARTER_WINDOWS = {
    "Q4_2019": ("2019-10-01", "2020-01-01"),
    "Q1_2020": ("2020-01-01", "2020-04-01"),
    "Q2_2020": ("2020-04-01", "2020-07-01"),
    "Q3_2020": ("2020-07-01", "2020-10-01"),
    "Q4_2020": ("2020-10-01", "2021-01-01"),
}

quarterly_data = {q: etf_window(s, e) for q, (s, e) in QUARTER_WINDOWS.items()}
quarters = ["Q1_2020", "Q2_2020", "Q3_2020", "Q4_2020"]

# %%
print("Quarterly datasets (real ETF features):")
for q, df in quarterly_data.items():
    print(f"  {q}: {len(df):,} symbol-days")

# %% [markdown]
# This sequence exposes the monitor to a real calm → crisis → recovery progression.
# Q1 2020 is the regime break (COVID volatility spike), Q2 the height of stress, with Q3 and
# Q4 walking back toward normal. The point is to test whether the monitoring stack catches the
# transition and whether it stays elevated longer than the underlying shift warrants.

# %%
# Track drift over time
etf_features = ["momentum_20", "momentum_60", "volatility_20", "rsi_14", "volume_ratio"]
reference = quarterly_data["Q4_2019"]

drift_tracking = []

for quarter in quarters:
    current = quarterly_data[quarter]

    # Compute drift metrics
    summary = analyze_drift(
        reference=reference,
        test=current,
        features=etf_features,
        methods=["psi", "wasserstein", "domain_classifier"],
    )

    # Extract PSI values from feature results
    psi_values = [fr.psi_result.psi for fr in summary.feature_results if fr.psi_result]
    avg_psi = np.mean(psi_values) if psi_values else 0
    max_psi = max(psi_values) if psi_values else 0

    # Find most drifted feature
    most_drifted = None
    if psi_values:
        max_idx = np.argmax(psi_values)
        most_drifted = summary.feature_results[max_idx].feature

    drift_tracking.append(
        {
            "quarter": quarter,
            "has_drift": summary.overall_drifted,
            "avg_psi": avg_psi,
            "max_psi": max_psi,
            "domain_auc": summary.domain_classifier_result.auc
            if summary.domain_classifier_result
            else 0.5,
            "most_drifted": most_drifted,
        }
    )

# %%
drift_df = pd.DataFrame(drift_tracking)
print("Quarterly Drift Tracking")
drift_df

# %% [markdown]
# The quarterly table is the operational summary a desk would actually review. Rising average PSI,
# higher domain AUC, and repeated alerts together indicate that the model is moving away from its
# original training environment rather than seeing a one-off anomaly.

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Average PSI Over Time",
        "Domain Classifier AUC",
        "Most Drifted Feature",
        "Drift Alert Timeline",
    ),
)

# Average PSI
fig.add_trace(
    go.Scatter(
        x=drift_df["quarter"],
        y=drift_df["avg_psi"],
        mode="lines+markers",
        name="Avg PSI",
        marker=dict(size=10, color="steelblue"),
    ),
    row=1,
    col=1,
)
fig.add_hline(y=0.1, line_dash="dash", line_color="orange", row=1, col=1)
fig.add_hline(y=0.25, line_dash="dash", line_color="red", row=1, col=1)

# Domain AUC
colors = [
    "green" if auc < 0.6 else "orange" if auc < 0.7 else "red" for auc in drift_df["domain_auc"]
]
fig.add_trace(
    go.Bar(x=drift_df["quarter"], y=drift_df["domain_auc"], marker_color=colors, name="Domain AUC"),
    row=1,
    col=2,
)
fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=1, col=2)

# %%
# Most drifted feature
fig.add_trace(
    go.Scatter(
        x=drift_df["quarter"],
        y=drift_df["max_psi"],
        mode="markers+text",
        name="Max PSI",
        text=drift_df["most_drifted"],
        textposition="top center",
        marker=dict(size=15, color="coral"),
    ),
    row=2,
    col=1,
)

# Alert timeline
alert_colors = ["green" if not d else "red" for d in drift_df["has_drift"]]
fig.add_trace(
    go.Scatter(
        x=drift_df["quarter"],
        y=[1] * len(drift_df),
        mode="markers",
        name="Drift Alert",
        marker=dict(size=30, color=alert_colors, symbol="circle"),
    ),
    row=2,
    col=2,
)

# %%
fig.update_layout(
    title="ETF Momentum Strategy - Quarterly Drift Monitoring Dashboard",
    height=600,
    showlegend=False,
)
fig.show()

# %% [markdown]
# The dashboard turns a raw drift table into an escalation timeline. Once the alerts remain red
# across consecutive quarters, retraining or feature review becomes a disciplined response rather
# than an ad hoc judgment call.

# %% [markdown]
# ---
#
# ## Part 7: Case Study - Crypto Funding Rate (Rapid Drift)
#
# Crypto markets exhibit rapid regime changes. Monitor hourly for drift.

# %%
# Build a real hourly crypto perpetuals feature panel: join OHLCV with the
# funding-premium index by (symbol, timestamp), then compute rolling features.


# %%
def crypto_feature_panel() -> pl.DataFrame:
    """Compute crypto perp features from real OHLCV and funding-premium data."""

    perp = load_crypto_perps()
    premium = load_crypto_premium().select(
        "timestamp", "symbol", funding_rate=pl.col("premium_index_close")
    )
    return (
        perp.sort(["symbol", "timestamp"])
        .with_columns(ret_1h=pl.col("close").pct_change().over("symbol"))
        .with_columns(
            volatility_1h=pl.col("ret_1h").rolling_std(24).over("symbol"),
            volume_ratio=(pl.col("volume") / pl.col("volume").rolling_mean(168)).over("symbol"),
            mom_24h=(pl.col("close") / pl.col("close").shift(24) - 1).over("symbol"),
        )
        .join(premium, on=["timestamp", "symbol"], how="left")
        .with_columns(pl.col("funding_rate").forward_fill().over("symbol"))
    )


# %%
crypto_features = ["funding_rate", "volatility_1h", "volume_ratio", "mom_24h"]
crypto_panel = crypto_feature_panel()


def crypto_window(start: str, end: str) -> pd.DataFrame:
    """Slice the crypto panel by datetime window and return a pandas feature frame."""

    return (
        crypto_panel.filter(
            (pl.col("timestamp") >= pl.lit(start).str.to_datetime(time_zone="UTC"))
            & (pl.col("timestamp") < pl.lit(end).str.to_datetime(time_zone="UTC"))
        )
        .select(crypto_features)
        .drop_nulls()
        .to_pandas()
    )


# Baseline = a calm month before the 2022 stress cycle; test windows walk
# through a normal month, a bullish rally, the LUNA/UST collapse, and the
# post-stress stabilization that preceded FTX.
baseline_crypto = crypto_window("2021-01-01", "2021-02-01")
regimes = {
    "Apr 2021 (Normal)": crypto_window("2021-04-01", "2021-05-01"),
    "Oct 2021 (Bull rally)": crypto_window("2021-10-01", "2021-11-01"),
    "May 2022 (LUNA crisis)": crypto_window("2022-05-01", "2022-06-01"),
    "Sep 2022 (Recovery)": crypto_window("2022-09-01", "2022-10-01"),
}

print(f"Baseline (Jan 2021): {len(baseline_crypto):,} symbol-hours")
for label, df in regimes.items():
    print(f"  {label}: {len(df):,} symbol-hours")

# %% [markdown]
# The crypto setup contrasts a stable baseline with a bullish rally, the LUNA collapse, and a
# post-stress stabilization period. That sequence tests whether the monitor reacts quickly to
# abrupt distribution breaks without staying permanently elevated once conditions normalize.

# %%
# Analyze drift across regimes
crypto_drift = []

for period, current in regimes.items():
    summary = analyze_drift(
        reference=baseline_crypto,
        test=current,
        features=crypto_features,
        methods=["psi", "wasserstein", "domain_classifier"],
    )

    # Extract PSI by feature
    psi_by_feature = {
        fr.feature: fr.psi_result.psi for fr in summary.feature_results if fr.psi_result
    }

    crypto_drift.append(
        {
            "period": period,
            "has_drift": summary.overall_drifted,
            "domain_auc": summary.domain_classifier_result.auc
            if summary.domain_classifier_result
            else 0.5,
            **{f"psi_{f}": psi_by_feature.get(f, 0) for f in crypto_features},
        }
    )

crypto_drift_df = pd.DataFrame(crypto_drift)

# %%
# Report drift results per regime
print("Crypto Regime Drift Analysis:")
print("-" * 80)
for _, row in crypto_drift_df.iterrows():
    status = "[ALERT] REGIME CHANGE" if row["has_drift"] else "[OK] STABLE"
    print(f"\n{row['period']} - {status}")
    print(f"  Domain Classifier AUC: {row['domain_auc']:.3f}")
    print("  Feature PSI:")
    for f in crypto_features:
        psi = row[f"psi_{f}"]
        flag = "[ALERT]" if psi > 0.25 else "[WARN]" if psi > 0.1 else "  "
        print(f"    {flag} {f:20} {psi:.4f}")

# %% [markdown]
# The regime report should spike in the bullish and crisis windows, with crisis usually producing
# the strongest joint signal. The recovery week is a useful reminder that drift alerts can also
# subside once the market returns closer to the training distribution.

# %%
# Heatmap of PSI across features and periods
psi_matrix = crypto_drift_df[[f"psi_{f}" for f in crypto_features]].values

fig = go.Figure(
    data=go.Heatmap(
        z=psi_matrix.T,
        x=[r["period"] for r in crypto_drift],
        y=crypto_features,
        colorscale=[
            [0, "green"],
            [0.1 / 1.0, "lightgreen"],
            [0.25 / 1.0, "yellow"],
            [0.5 / 1.0, "orange"],
            [1.0, "red"],
        ],
        colorbar=dict(title="PSI"),
        zmin=0,
        zmax=1.0,
    )
)

fig.update_layout(
    title="Crypto Funding Rate Strategy - Feature Drift Heatmap",
    xaxis_title="Time Period",
    yaxis_title="Feature",
    height=400,
)
fig.show()

# %% [markdown]
# The heatmap makes cross-feature timing easy to read. Instead of asking whether drift exists in
# the abstract, we can see which inputs moved first and which ones became unstable together.

# %% [markdown]
# ---
#
# ## Part 8: Production Monitoring Workflow
#
# Implementing drift detection in a production pipeline.


# %% [markdown]
# The production section packages the earlier metrics into a monitoring object with explicit alert
# thresholds. Separating configuration from execution makes the governance logic visible and easy
# to tune.


# %%
@dataclass
class DriftMonitorConfig:
    """Configuration for production drift monitoring."""

    feature_columns: list[str]
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.25
    wasserstein_warning_threshold: float = 0.3
    wasserstein_critical_threshold: float = 0.5
    domain_auc_threshold: float = 0.7
    min_samples_for_check: int = 100


# %% [markdown]
# This monitor keeps only the essential control loop: compare against a reference window, record
# the most important diagnostics, and decide whether the recent alert pattern warrants retraining.


# %%
class ProductionDriftMonitor:
    """Production-ready drift monitoring system."""

    def __init__(self, reference_df: pd.DataFrame, config: DriftMonitorConfig):
        self.reference_df = reference_df
        self.config = config
        self.drift_history = []

    def check_drift(self, current_df: pd.DataFrame, timestamp: str) -> dict:
        """Check for drift and return alert if detected."""
        if len(current_df) < self.config.min_samples_for_check:
            return {"status": "insufficient_data", "samples": len(current_df)}
        summary = analyze_drift(
            reference=self.reference_df,
            test=current_df,
            features=self.config.feature_columns,
            methods=["psi", "domain_classifier"],
            psi_config={"psi_threshold_red": self.config.psi_critical_threshold},
            domain_classifier_config={"threshold": self.config.domain_auc_threshold},
        )
        psi_by_feature = {
            fr.feature: fr.psi_result.psi for fr in summary.feature_results if fr.psi_result
        }
        max_psi = max(psi_by_feature.values()) if psi_by_feature else 0
        domain_auc = (
            summary.domain_classifier_result.auc if summary.domain_classifier_result else 0.5
        )
        most_drifted = max(psi_by_feature, key=psi_by_feature.get) if psi_by_feature else None
        if (
            max_psi >= self.config.psi_critical_threshold
            or domain_auc >= self.config.domain_auc_threshold
        ):
            alert_level = "CRITICAL"
        elif max_psi >= self.config.psi_warning_threshold:
            alert_level = "WARNING"
        else:
            alert_level = "OK"
        result = {
            "timestamp": timestamp,
            "status": alert_level,
            "max_psi": max_psi,
            "domain_auc": domain_auc,
            "most_drifted_feature": most_drifted,
            "psi_by_feature": psi_by_feature,
        }
        self.drift_history.append(result)
        return result

    def get_dashboard_data(self) -> pd.DataFrame:
        """Get drift history for dashboard visualization."""
        return pd.DataFrame(self.drift_history)

    def should_retrain(self, lookback: int = 5) -> tuple[bool, str]:
        """Determine if model should be retrained based on drift patterns."""
        if len(self.drift_history) < lookback:
            return False, "Insufficient history"
        recent = self.drift_history[-lookback:]
        critical_count = sum(1 for r in recent if r["status"] == "CRITICAL")
        warning_count = sum(1 for r in recent if r["status"] == "WARNING")
        if critical_count >= 2:
            return True, f"Multiple CRITICAL alerts ({critical_count}/{lookback})"
        if warning_count >= 3:
            return True, f"Persistent WARNING alerts ({warning_count}/{lookback})"
        return False, "Drift levels acceptable"


# %%
# Demo usage
config = DriftMonitorConfig(
    feature_columns=crypto_features, psi_warning_threshold=0.1, psi_critical_threshold=0.25
)

monitor = ProductionDriftMonitor(baseline_crypto, config)

# Simulate monitoring over time
print("Production Drift Monitoring Simulation:")
print("=" * 60)

for i, (period, data) in enumerate(regimes.items()):
    result = monitor.check_drift(data, f"2024-01-{(i + 1) * 7:02d}")

    status_emoji = (
        "[ALERT]"
        if result["status"] == "CRITICAL"
        else "[WARN]"
        if result["status"] == "WARNING"
        else "[OK]"
    )
    print(f"\n{status_emoji} {period} ({result['timestamp']})")
    print(f"   Status: {result['status']}")
    print(f"   Max PSI: {result['max_psi']:.4f}")
    print(f"   Domain AUC: {result['domain_auc']:.4f}")
    print(f"   Most Drifted: {result['most_drifted_feature']}")

# Check if retraining needed
should_retrain, reason = monitor.should_retrain(lookback=4)
print(f"\n{'=' * 60}")
print(f"Retraining Recommendation: {'YES [RETRAIN]' if should_retrain else 'NO [OK]'}")
print(f"Reason: {reason}")

# %% [markdown]
# This simulation illustrates the operational decision rule rather than claiming a production-tuned
# threshold. Consecutive critical alerts are treated as evidence that the model environment has
# changed enough to justify retraining or a deeper risk review.

# %% [markdown]
# ---
#
# ## Part 9: Best Practices and Recommendations
#
# ### When to Use Each Method
#
# | Method | Best For | Limitations |
# |--------|----------|-------------|
# | **PSI** | Regulatory compliance, quick checks | Sensitive to binning, ignores ordering |
# | **Wasserstein** | Continuous features, interpretable | Single feature at a time |
# | **Domain Classifier** | Multivariate drift, complex patterns | Needs more data, less interpretable |
#
# ### Recommended Monitoring Strategy
#
# 1. **Daily/Hourly**: Compute PSI for critical features
# 2. **Weekly**: Run domain classifier for multivariate drift
# 3. **Monthly**: Deep-dive analysis of drifted features
# 4. **On Alert**: Investigate root cause before retraining
#
# ### Common Pitfalls
#
# 1. **Overfitting to drift**: Don't retrain too frequently
# 2. **Ignoring concept drift**: Feature drift without performance drop may be OK
# 3. **Small sample sizes**: Drift metrics can be noisy with few samples
# 4. **Not tracking downstream impact**: Drift → Performance degradation is what matters

# %% [markdown]
# ### Drift Detection Quick Reference
#
# **PSI Thresholds**
#
# | PSI Range | Action |
# |-----------|--------|
# | < 0.10 | No action needed |
# | 0.10 - 0.25 | Monitor closely |
# | > 0.25 | Investigate and consider retraining |
#
# **Domain Classifier AUC**
#
# | AUC Range | Interpretation |
# |-----------|----------------|
# | < 0.60 | No significant multivariate drift |
# | 0.60 - 0.70 | Moderate drift, monitor |
# | > 0.70 | Strong drift signal |
#
# **Action Matrix**
#
# | Drift Status | Performance | Action |
# |-------------|-------------|--------|
# | Drift | OK | Monitor, don't retrain |
# | Drift | Down | Retrain immediately |
# | No Drift | Down | Check for concept drift |
# | No Drift | OK | Keep monitoring |

# %% [markdown]
# ## Key Takeaways
#
# 1. **PSI surfaces the features that actually moved.** On the 2017-vs-2020
#    ETF split, volatility_20 (PSI 1.47) and momentum_20 (PSI 0.46) clear the
#    0.25 alert threshold while rsi_14 (0.019) and volume_ratio (0.005) stay
#    stable — RSI is bounded by construction and volume ratios normalize by
#    a rolling baseline. The unified report pairs this with a domain-classifier
#    AUC of 0.86, agreeing that the joint distribution has shifted.
# 2. **The ETF quarterly demo shows a real crisis-then-recovery arc.** With Q4
#    2019 as the calm reference, average PSI runs 0.48 (Q1 2020) → 1.07 (Q2)
#    → 0.45 (Q3) → 0.33 (Q4); max PSI peaks at 2.75 in Q2 (volatility_20) and
#    decays as conditions stabilize. Recovery is partial — Q4 2020 still trips
#    the threshold, which is exactly when a desk reviews whether to retrain.
# 3. **The crypto regime panel exposes funding-rate as the dominant signal.**
#    Against the Jan 2021 baseline, PSI on `funding_rate` runs 0.10 (Apr 2021)
#    → 0.88 (Oct 2021 bull rally) → 9.00 (May 2022 LUNA) → 7.80 (Sep 2022
#    post-stress). volatility_1h spikes in the rally and crisis windows;
#    volume_ratio normalizes against its rolling baseline and stays quiet.
# 4. **Threshold calibration is non-trivial — every crypto window alerts.** The
#    `ProductionDriftMonitor` flags all four periods as `CRITICAL`. Domain AUC
#    reaches 0.81 even for the "Normal" April 2021 window because crypto
#    funding regimes drift continuously and the test windows are not drawn
#    from the same distribution as the January 2021 baseline. In production,
#    the AUC threshold has to be calibrated against the empirical noise floor
#    of two adjacent calm windows; PSI per feature, paired with a downstream
#    performance check, is the more reliable trigger.
#
# **Next**: `08_ml_exit_signals.ipynb` builds an entry/exit two-model
# architecture that consumes the kind of drift signal generated here.
#
# **Book reference**: §19.7 (Adaptive Risk Controls), §19.8 (Kill Switches and
# Governance — PSI thresholds 0.10 / 0.25).
