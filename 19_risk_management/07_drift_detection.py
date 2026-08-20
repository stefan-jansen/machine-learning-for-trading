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
# # Feature Drift Detection for Production ML Strategies
# **Docker image**: `ml4t-gpu` (CUDA required; execution fails closed without it)
#
# **Purpose**: Walk through the `ml4t.diagnostic.evaluation.drift` toolkit (PSI,
# Wasserstein distance, domain classifier, unified `analyze_drift`) on real
# ETF momentum features split across calm and stressed market windows and a
# real crypto perpetuals + premium-index panel across market regimes, and
# wrap the result in a `ProductionDriftMonitor` with explicit alert thresholds.
#
# **Learning objectives**:
# 1. Compute Population Stability Index (PSI) and read its bin-level breakdown.
# 2. Compare PSI to the Wasserstein distance interpretation.
# 3. Use a GPU-trained domain classifier on stacked reference + test samples
#    for multivariate drift detection.
# 4. Translate per-method drift scores into alert levels and a retraining
#    recommendation via `ProductionDriftMonitor.should_retrain()`.
#
# **Book reference**: Chapter 19, Sections 19.7 and 19.8.
#
# **Prerequisites**: Feature engineering and distribution diagnostics from Chapter 8, and an
# `ml4t-diagnostic` install.
#
# The alert thresholds every section below compares against are declared in the parameters cell
# and printed there with what each band means. They are the conventional rules of thumb and
# nothing more: the right boundary depends on the feature's own distribution and on how many
# observations each check has, and a threshold carried over untested will either alert constantly
# or never.
#
# **Data**: Real ETF and crypto feature panels. ETF features (momentum, volatility,
# RSI, volume ratio) come from `load_etfs` and are split by calendar window
# (calm 2017 vs stressed 2020). Crypto features (premium index, 1h volatility,
# volume ratio, 24h momentum) come from `load_crypto_perps` joined with
# `load_crypto_premium`, compared in selected month-long windows over hourly rows.

# %%
"""Detect feature drift using PSI, Wasserstein distance, and domain classifiers."""

import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import torch
from IPython.display import Markdown, display

# ml4t-diagnostic imports
from ml4t.diagnostic.evaluation.drift import (
    DomainClassifierResult,
    DriftSummaryResult,
    PSIResult,
    WassersteinResult,
    analyze_drift,
    compute_psi,
    compute_wasserstein_distance,
)
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from data import load_crypto_perps, load_crypto_premium, load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI
SEED = 42
PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25
DOMAIN_AUC_WARNING_THRESHOLD = 0.60
DOMAIN_AUC_CRITICAL_THRESHOLD = 0.70
DOMAIN_CV_TIME_BLOCKS = 20

# %% [markdown]
# What each setting decides:
#
# - `PSI_WARNING_THRESHOLD` and `PSI_CRITICAL_THRESHOLD` are the population stability index values
#   at which a single feature's distribution counts as having moved enough to watch, and enough to
#   act on.
# - `DOMAIN_AUC_WARNING_THRESHOLD` and `DOMAIN_AUC_CRITICAL_THRESHOLD` are the same two bands for
#   the multivariate check. Half is the no-drift point, since a classifier that cannot separate the
#   samples scores exactly that.
# - `DOMAIN_CV_TIME_BLOCKS` is how many contiguous time blocks the domain classifier's
#   cross-validation splits into. Splitting by block rather than at random keeps neighbouring days
#   out of opposite folds, which would otherwise let the classifier separate the samples by
#   memorizing a period rather than by detecting drift.
#
# All four thresholds are the conventional rules of thumb. Calibrate them against a period where
# you know whether drift occurred before trusting them to page anyone.

# %%
print(
    f"PSI       stable below {PSI_WARNING_THRESHOLD}, "
    f"warning to {PSI_CRITICAL_THRESHOLD}, critical above\n"
    f"Domain AUC no separation below {DOMAIN_AUC_WARNING_THRESHOLD}, "
    f"warning to {DOMAIN_AUC_CRITICAL_THRESHOLD}, critical above\n"
    f"Domain CV  {DOMAIN_CV_TIME_BLOCKS} contiguous time blocks"
)

# %% [markdown]
# Reader-facing labels and policy colors live outside the Papermill parameters cell because they
# are presentation constants, not experiment controls.

# %%
FEATURE_LABELS = {
    "momentum_20": "20-day momentum",
    "momentum_60": "60-day momentum",
    "volatility_20": "20-day volatility",
    "rsi_14": "14-day RSI",
    "volume_ratio": "Volume ratio",
    "premium_index_close": "Premium index",
    "volatility_1h": "1-hour volatility",
    "mom_24h": "24-hour momentum",
}

ALERT_RANK = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
ALERT_COLORS = {
    "OK": COLORS["positive"],
    "WARNING": COLORS["amber"],
    "CRITICAL": COLORS["negative"],
}

# %% [markdown]
# A single left-inclusive policy helper keeps equality behavior identical for PSI and domain AUC.


# %%
def policy_alert_level(
    value: float,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    """Map a score to the shared left-inclusive alert policy."""
    if warning_threshold >= critical_threshold:
        raise ValueError("The warning threshold must be below the critical threshold")
    if value >= critical_threshold:
        return "CRITICAL"
    if value >= warning_threshold:
        return "WARNING"
    return "OK"


# %% [markdown]
# The production monitor escalates to whichever of PSI or domain AUC has the more severe state.


# %%
def combined_alert_level(
    psi_value: float,
    domain_auc: float,
    psi_warning_threshold: float = PSI_WARNING_THRESHOLD,
    psi_critical_threshold: float = PSI_CRITICAL_THRESHOLD,
    domain_auc_warning_threshold: float = DOMAIN_AUC_WARNING_THRESHOLD,
    domain_auc_critical_threshold: float = DOMAIN_AUC_CRITICAL_THRESHOLD,
) -> str:
    """Return the more severe PSI or domain-AUC alert level."""
    psi_level = policy_alert_level(
        psi_value,
        psi_warning_threshold,
        psi_critical_threshold,
    )
    auc_level = policy_alert_level(
        domain_auc,
        domain_auc_warning_threshold,
        domain_auc_critical_threshold,
    )
    return max((psi_level, auc_level), key=ALERT_RANK.__getitem__)


# %% [markdown]
# Retraining uses counts anywhere in a declared lookback, not a consecutive-run rule.


# %%
def alert_count_condition(
    statuses: list[str],
    critical_required: int = 2,
    warning_required: int = 3,
) -> tuple[bool, str]:
    """Evaluate any-N alert counts within the supplied lookback statuses."""
    lookback = len(statuses)
    critical_count = statuses.count("CRITICAL")
    warning_count = statuses.count("WARNING")
    if critical_count >= critical_required:
        return (
            True,
            f"CRITICAL count in last {lookback} checks: {critical_count} "
            f"(requires {critical_required})",
        )
    if warning_count >= warning_required:
        return (
            True,
            f"WARNING count in last {lookback} checks: {warning_count} "
            f"(requires {warning_required})",
        )
    return False, f"Alert counts in last {lookback} checks are below the retraining thresholds"


# %% [markdown]
# Publication execution is GPU-only. This fail-closed check runs before data loading so a CPU
# environment cannot produce a certifiable-looking partial artifact.

# %%
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this notebook; CPU fallback is disabled")
GPU_DEVICE_INDEX = torch.cuda.current_device()
GPU_DEVICE_NAME = torch.cuda.get_device_name(GPU_DEVICE_INDEX)
print(
    "GPU_DEVICE_PROOF "
    f"cuda_available=True device_index={GPU_DEVICE_INDEX} "
    f"device_name={GPU_DEVICE_NAME!r} lightgbm_device='cuda' cpu_fallback=False"
)

# %% [markdown]
# ---
#
# ## Part 1: Drift Types and Scope
#
# ### Covariate Drift vs Concept Drift
#
# **Covariate Drift**: P(X) changes but P(Y|X) stays the same
# - Example: Volatility increases but signal-return relationship unchanged
# - Response: Check model performance, then consider recalibration or retraining
#
# **Concept Drift**: P(Y|X) changes
# - Example: Momentum factor stops working (regime change)
# - Solution: New model architecture or feature set
#
# **Prior Drift**: P(Y) changes
# - Example: Market goes from bull to bear (different return distribution)
# - Solution: Regime-aware models
#
# This notebook measures **covariate drift only**. Concept drift requires labels or realized
# strategy outcomes, neither of which is used here. A feature-distribution alert is therefore a
# prompt to investigate, not evidence that the signal-return relationship has broken.

# %% [markdown]
# We build a real ETF feature panel - one row per (symbol, trading day) carrying
# 20- and 60-day momentum, 20-day return volatility, the 14-day RSI, and a
# 20-day volume ratio - then split it by calendar window. The calm-vs-stress
# contrast (2017 vs 2020) makes the metrics work on a genuine regime shift
# rather than a fabricated one.

# %%
set_global_seeds(SEED)

# %% [markdown]
# Relative-time blocks keep all symbols at the same within-window timestamp together during
# validation, preventing row-level leakage across domain-classifier folds.


# %%
def relative_time_blocks(frame: pd.DataFrame) -> np.ndarray:
    """Assign rows to complete relative-time blocks."""
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    ranks = timestamps.rank(method="dense").to_numpy(dtype=np.int64) - 1
    n_timestamps = int(ranks.max()) + 1
    if n_timestamps < DOMAIN_CV_TIME_BLOCKS:
        raise ValueError("Too few distinct timestamps for blocked domain-classifier CV")
    return np.minimum(
        ranks * DOMAIN_CV_TIME_BLOCKS // n_timestamps,
        DOMAIN_CV_TIME_BLOCKS - 1,
    )


# %% [markdown]
# The CUDA fit helper performs the same five blocked folds and final all-row fit. It verifies the
# retained LightGBM backend before returning any result.


# %%
def fit_cuda_domain_model(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, lgb.LGBMClassifier, list[float], int]:
    """Fit blocked validation folds and the final CUDA model."""
    X = pd.concat([reference[features], current[features]], ignore_index=True)
    y = np.concatenate(
        [np.zeros(len(reference), dtype=np.int8), np.ones(len(current), dtype=np.int8)]
    )
    groups = np.concatenate([relative_time_blocks(reference), relative_time_blocks(current)])
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=SEED,
        device_type="cuda",
        n_jobs=1,
        verbose=-1,
    )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=False)
    cv_scores = []
    for train_index, validation_index in splitter.split(X, y, groups):
        fold_model = clone(model).fit(X.iloc[train_index], y[train_index])
        fold_device = fold_model.booster_.params.get(
            "device_type", fold_model.booster_.params.get("device")
        )
        if fold_device != "cuda":
            raise RuntimeError("A domain-classifier fold did not use the CUDA backend")
        predictions = fold_model.predict_proba(X.iloc[validation_index])[:, 1]
        cv_scores.append(roc_auc_score(y[validation_index], predictions))
    model.fit(X, y)
    actual_device = model.booster_.params.get("device_type", model.booster_.params.get("device"))
    if actual_device != "cuda":
        raise RuntimeError("LightGBM did not retain the required CUDA device selection")
    print(
        "LIGHTGBM_DEVICE_PROOF "
        f"backend={actual_device!r} cv_folds={splitter.n_splits} "
        "n_jobs=1 cpu_fallback=False"
    )
    return X, model, cv_scores, splitter.n_splits


# %% [markdown]
# AUC interpretation uses the same warning and critical boundaries as the production monitor.


# %%
def domain_auc_interpretation(
    cv_auc: float,
    warning_threshold: float,
    critical_threshold: float,
) -> tuple[str, str]:
    """Return the policy state and reader-facing AUC interpretation."""
    alert_level = policy_alert_level(cv_auc, warning_threshold, critical_threshold)
    if alert_level == "CRITICAL":
        interpretation = (
            "Critical distribution-separation alert from out-of-fold AUC "
            f"({cv_auc:.4f} >= {critical_threshold:.4f})."
        )
    elif alert_level == "WARNING":
        interpretation = (
            "Warning distribution-separation alert from out-of-fold AUC "
            f"({warning_threshold:.4f} <= {cv_auc:.4f} < {critical_threshold:.4f})."
        )
    else:
        interpretation = (
            f"No policy-level separation detected from out-of-fold AUC "
            f"({cv_auc:.4f} < {warning_threshold:.4f})."
        )
    return alert_level, interpretation


# %% [markdown]
# Metadata records the blocked-CV design and fail-closed CUDA environment alongside the result.


# %%
def domain_result_metadata(
    alert_level: str,
    warning_threshold: float,
    critical_threshold: float,
    cv_folds: int,
) -> dict:
    """Build the domain-classifier audit metadata."""
    return {
        "auc_basis": "cross-validation mean",
        "alert_level": alert_level,
        "warning_threshold": warning_threshold,
        "critical_threshold": critical_threshold,
        "cv_folds": cv_folds,
        "cv_scheme": "stratified relative-time blocks",
        "cv_time_blocks": DOMAIN_CV_TIME_BLOCKS,
        "device": "cuda",
        "device_name": GPU_DEVICE_NAME,
        "cpu_fallback": False,
        "random_state": SEED,
    }


# %% [markdown]
# The result builder preserves the toolkit schema while using held-out AUC for policy decisions and
# the final all-row fit only for feature ranking.


# %%
def build_domain_classifier_result(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    model: lgb.LGBMClassifier,
    cv_scores: list[float],
    started: float,
    warning_threshold: float,
    critical_threshold: float,
    cv_folds: int,
) -> DomainClassifierResult:
    """Assemble the audited domain-classifier result."""
    cv_auc = float(np.mean(cv_scores))
    importance = pl.DataFrame({"feature": features, "importance": model.feature_importances_}).sort(
        "importance", descending=True
    )
    importance = importance.with_columns(pl.arange(1, len(importance) + 1).alias("rank"))
    alert_level, interpretation = domain_auc_interpretation(
        cv_auc, warning_threshold, critical_threshold
    )
    return DomainClassifierResult(
        auc=cv_auc,
        drifted=alert_level != "OK",
        feature_importances=importance,
        threshold=warning_threshold,
        n_reference=len(reference),
        n_test=len(current),
        n_features=len(features),
        model_type="lightgbm-cuda",
        cv_auc_mean=cv_auc,
        cv_auc_std=float(np.std(cv_scores)),
        interpretation=interpretation,
        computation_time=time.perf_counter() - started,
        metadata=domain_result_metadata(
            alert_level, warning_threshold, critical_threshold, cv_folds
        ),
    )


# %% [markdown]
# The public adapter validates the feature schema, runs the unchanged CUDA computation, and returns
# a complete `DomainClassifierResult` without a CPU fallback.


# %%
def gpu_domain_classifier(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    warning_threshold: float = DOMAIN_AUC_WARNING_THRESHOLD,
    critical_threshold: float = DOMAIN_AUC_CRITICAL_THRESHOLD,
) -> DomainClassifierResult:
    """Fit on CUDA and score folds that hold out complete relative-time blocks."""
    required = {"timestamp", *features}
    if not required.issubset(reference.columns) or not required.issubset(current.columns):
        raise ValueError("Both domains must contain timestamp and every requested feature")
    started = time.perf_counter()
    _, model, cv_scores, cv_folds = fit_cuda_domain_model(reference, current, features)
    return build_domain_classifier_result(
        reference,
        current,
        features,
        model,
        cv_scores,
        started,
        warning_threshold,
        critical_threshold,
        cv_folds,
    )


# %% [markdown]
# The summary finalizer attaches the validated CUDA result to the two univariate methods without
# changing the library result schema.


# %%
def finalize_drift_summary(
    summary: DriftSummaryResult,
    domain: DomainClassifierResult,
) -> DriftSummaryResult:
    """Attach the domain result and finalize the unified drift summary."""
    summary.domain_classifier_result = domain
    summary.methods_used.append("domain_classifier")
    summary.multivariate_methods.append("domain_classifier")
    summary.overall_drifted = summary.n_features_drifted > 0 or domain.drifted
    summary.computation_time += domain.computation_time
    summary.interpretation = (
        f"Drift detected in {summary.n_features_drifted}/{summary.n_features} features"
        if summary.overall_drifted
        else f"No drift detected across {summary.n_features} features"
    )
    return summary


# %% [markdown]
# The unified helper requires both univariate methods to complete, applies reference-only PSI
# settings, and adds the blocked out-of-fold CUDA classifier. GPU histogram construction is seeded
# but not bitwise deterministic, so conclusions near either AUC cutoff require investigation.


# %%
def checked_drift_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    psi_warning_threshold: float = PSI_WARNING_THRESHOLD,
    psi_critical_threshold: float = PSI_CRITICAL_THRESHOLD,
    domain_auc_warning_threshold: float = DOMAIN_AUC_WARNING_THRESHOLD,
    domain_auc_critical_threshold: float = DOMAIN_AUC_CRITICAL_THRESHOLD,
) -> DriftSummaryResult:
    """Run all drift methods and fail rather than return a degraded partial result."""

    summary = analyze_drift(
        reference=reference,
        test=current,
        features=features,
        methods=["psi", "wasserstein"],
        consensus_threshold=1.0,
        psi_config={
            "psi_threshold_yellow": psi_warning_threshold,
            "psi_threshold_red": psi_critical_threshold,
        },
        wasserstein_config={"threshold_calibration": False},
    )
    if any(result.n_methods_run != 2 for result in summary.feature_results):
        raise RuntimeError("Drift analysis did not complete all requested methods")
    domain = gpu_domain_classifier(
        reference,
        current,
        features,
        warning_threshold=domain_auc_warning_threshold,
        critical_threshold=domain_auc_critical_threshold,
    )
    return finalize_drift_summary(summary, domain)


# %% [markdown]
# ETF features are computed within symbol after sorting, so every rolling value uses only the
# current and prior observations.


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


# %% [markdown]
# The full feature panel is computed once and then reused across fixed teaching windows.

# %%
ETF_FEATURES = ["momentum_20", "momentum_60", "volatility_20", "rsi_14", "volume_ratio"]
etf_panel = etf_feature_panel(load_etfs())

# %% [markdown]
# Window selection keeps the timestamp needed for blocked validation and converts to pandas only at
# the diagnostic boundary.


# %%
def etf_window(start: str, end: str, cols: list[str] | None = None) -> pd.DataFrame:
    """Slice the ETF panel by date window and return a pandas frame of features."""

    cols = cols or ETF_FEATURES
    return (
        etf_panel.filter(
            (pl.col("timestamp") >= pl.lit(start).str.to_date())
            & (pl.col("timestamp") < pl.lit(end).str.to_date())
        )
        .select("timestamp", *cols)
        .drop_nulls()
        .to_pandas()
    )


# %% [markdown]
# Fixed calm and stressed windows create the first real-data comparison.

# %%
# Calm 2017 baseline vs stressed 2020 (COVID) drift window
baseline_features = etf_window(
    "2017-01-01", "2018-01-01", ["momentum_20", "volatility_20", "rsi_14", "volume_ratio"]
)
covariate_drift_features = etf_window(
    "2020-01-01", "2021-01-01", ["momentum_20", "volatility_20", "rsi_14", "volume_ratio"]
)

print(f"Baseline (2017): {len(baseline_features):,} symbol-days")
print(f"Drift   (2020): {len(covariate_drift_features):,} symbol-days")

# %% [markdown]
# Comparing 2017 to 2020 surfaces a real covariate shift: 20-day return volatility roughly
# doubles, momentum widens, and RSI / volume ratio stay closer to their 2017 distributions.
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
# ### Reading the Index
#
# A PSI near zero says the two samples put the same share of their mass in every bin. It grows as
# mass moves between bins, and because each bin contributes the movement times the log of the
# ratio, a bin that empties or fills contributes far more than one that shifts slightly. The bands
# the notebook alerts on are the parameters printed above.

# %%
# Compute PSI for each feature
features = ["momentum_20", "volatility_20", "rsi_14", "volume_ratio"]
psi_results = {}

for feature in features:
    result = compute_psi(
        reference=baseline_features[feature].values,
        test=covariate_drift_features[feature].values,
        n_bins=10,
        psi_threshold_yellow=PSI_WARNING_THRESHOLD,
        psi_threshold_red=PSI_CRITICAL_THRESHOLD,
    )
    psi_results[feature] = result

    alert_level = policy_alert_level(
        result.psi,
        PSI_WARNING_THRESHOLD,
        PSI_CRITICAL_THRESHOLD,
    )
    status = {
        "OK": "[OK] Stable",
        "WARNING": "[WARN] Warning",
        "CRITICAL": "[ALERT] Critical",
    }[alert_level]

    print(f"{feature:20} PSI={result.psi:.4f}  {status}")

# %% [markdown]
# PSI flags the real 2017-to-2020 distribution shift in these ETF features. The largest values should
# appear on the features with both location and dispersion changes, which is why the next plot
# drills into the worst offender bin by bin.

# %% [markdown]
# The first PSI helper creates matched reference and current bars for the same quantile bins.


# %%
def psi_distribution_bars(result: PSIResult, bins: list[int]) -> tuple[go.Bar, go.Bar]:
    """Build matched reference and current distribution bars."""
    reference_bar = go.Bar(
        x=bins,
        y=result.reference_percents,
        name="Reference",
        marker_color=COLORS["blue"],
        opacity=0.7,
    )
    current_bar = go.Bar(
        x=bins,
        y=result.test_percents,
        name="Current",
        marker_color=COLORS["amber"],
        opacity=0.8,
    )
    return reference_bar, current_bar


# %% [markdown]
# The contribution helper highlights bins whose PSI share exceeds an equal-share benchmark.


# %%
def psi_contribution_bar(result: PSIResult, bins: list[int]) -> go.Bar:
    """Build the per-bin PSI contribution bars."""
    return go.Bar(
        x=bins,
        y=result.bin_psi,
        name="PSI Contribution",
        marker_color=[
            COLORS["negative"] if value > result.psi / len(bins) else COLORS["silver_muted"]
            for value in result.bin_psi
        ],
    )


# %% [markdown]
# The combined chart separates overall drift magnitude from the bins that contribute most to it.


# %%
def plot_psi_breakdown(result: PSIResult, feature_name: str):
    """Visualize PSI contribution by bin."""
    feature_label = FEATURE_LABELS.get(feature_name, feature_name)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Distribution Comparison", "PSI Contribution by Bin")
    )
    bins = list(range(len(result.reference_percents)))
    reference_bar, current_bar = psi_distribution_bars(result, bins)
    fig.add_trace(reference_bar, row=1, col=1)
    fig.add_trace(current_bar, row=1, col=1)
    fig.add_trace(psi_contribution_bar(result, bins), row=1, col=2)
    fig.update_layout(
        title=f"Where {feature_label} moved between the two samples",
        barmode="group",
        showlegend=True,
        height=430,
    )
    fig.update_xaxes(title_text="Reference quantile bin", row=1, col=1)
    fig.update_yaxes(title_text="Share of observations", tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title_text="Reference quantile bin", row=1, col=2)
    fig.update_yaxes(title_text="PSI contribution", rangemode="tozero", row=1, col=2)
    return fig


# %% [markdown]
# The highest-PSI feature should show a concentrated set of bins driving the alert rather than
# uniform small deviations everywhere. That pattern usually points to a regime shift worth
# investigating instead of routine sampling noise.

# %%
# Visualize feature with highest PSI
worst_feature = max(psi_results, key=lambda x: psi_results[x].psi)
fig = plot_psi_breakdown(psi_results[worst_feature], worst_feature)
show_plotly_with_alt(
    fig,
    "Two panels: reference and current distributions as grouped bars per bin, and each bin's contribution to the total index. A small number of bins account for most of the total.",
)

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
        reference=baseline_features[feature].values,
        test=covariate_drift_features[feature].values,
        threshold_calibration=False,
    )
    wasserstein_results[feature] = result
    normalized_distance = result.distance / result.reference_stats["std"]

    # Use drifted status from API
    if not result.drifted:
        status = "[OK] No drift"
    else:
        status = "[ALERT] Drift detected"

    print(f"{feature:20} W={result.distance:.4f}  W/ref_sd={normalized_distance:.3f}  {status}")

# %% [markdown]
# Wasserstein distance gives a scale-aware cross-check without histogram bins. Here its threshold
# is a deliberately descriptive half of the reference standard deviation. A row-wise permutation
# p-value would treat autocorrelated symbol-days as independent and would overstate precision.

# %% [markdown]
# The next helper compares histograms and empirical CDFs side by side. The CDF gap is especially
# helpful for explaining why Wasserstein distance rises when the entire distribution shifts.


# %%
def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted values and their empirical CDF."""

    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, cdf


# %% [markdown]
# Matched histograms use shared bin edges so location and dispersion changes remain comparable.


# %%
def wasserstein_histograms(reference: np.ndarray, current: np.ndarray) -> tuple[go.Histogram, ...]:
    """Build matched probability-density histograms."""
    lower = min(reference.min(), current.min())
    upper = max(reference.max(), current.max())
    bin_size = (upper - lower) / 50
    return (
        go.Histogram(
            x=reference,
            histnorm="probability density",
            xbins=dict(start=lower, end=upper, size=bin_size),
            name="Reference",
            marker_color=COLORS["blue"],
            opacity=0.55,
        ),
        go.Histogram(
            x=current,
            histnorm="probability density",
            xbins=dict(start=lower, end=upper, size=bin_size),
            name="Current",
            marker_color=COLORS["amber"],
            opacity=0.55,
        ),
    )


# %% [markdown]
# Empirical-CDF traces show where the cumulative distributions separate without imposing bins.


# %%
def wasserstein_cdf_traces(reference: np.ndarray, current: np.ndarray) -> tuple[go.Scatter, ...]:
    """Build reference and current empirical-CDF traces."""
    ref_sorted, ref_cdf = empirical_cdf(reference)
    cur_sorted, cur_cdf = empirical_cdf(current)
    return (
        go.Scatter(
            x=ref_sorted,
            y=ref_cdf,
            name="Reference CDF",
            line=dict(color=COLORS["blue"], width=2),
        ),
        go.Scatter(
            x=cur_sorted,
            y=cur_cdf,
            name="Current CDF",
            line=dict(color=COLORS["amber"], width=2),
        ),
    )


# %% [markdown]
# The final comparison pairs shape and cumulative views under the same message-first title.


# %%
def plot_wasserstein_comparison(
    reference: np.ndarray, current: np.ndarray, result: WassersteinResult, feature_name: str
):
    """Visualize Wasserstein distance with histograms and CDFs."""
    feature_label = FEATURE_LABELS.get(feature_name, feature_name)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distribution comparison", "Empirical CDF comparison"),
    )
    for trace in wasserstein_histograms(reference, current):
        fig.add_trace(trace, row=1, col=1)
    for trace in wasserstein_cdf_traces(reference, current):
        fig.add_trace(trace, row=1, col=2)
    normalized_distance = result.distance / result.reference_stats["std"]
    fig.update_layout(
        title=f"How far {feature_label} would have to be moved to match",
        barmode="overlay",
        height=430,
        showlegend=True,
    )
    fig.update_xaxes(title_text=feature_label, row=1, col=1)
    fig.update_yaxes(title_text="Probability density", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text=feature_label, row=1, col=2)
    fig.update_yaxes(title_text="Cumulative probability", range=[0, 1], row=1, col=2)
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
show_plotly_with_alt(
    fig,
    "Two panels: overlaid histograms of the two samples, and their empirical cumulative distributions. The gap between the two curves is the distance the measure reports.",
)

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
# distributions at this sample size - absence of separation at this capacity
# is not proof the distribution has not moved.
#
# ### How It Works
# 1. Label reference data as 0, current data as 1
# 2. Train a classifier (usually gradient boosting)
# 3. Measure out-of-fold AUC. A classifier that cannot tell the two samples apart scores one half,
#    which is what no drift looks like. The further above one half it scores, the more the samples
#    differ in some combination of features - which is what makes this test able to see drift that
#    no single feature shows.

# %%
# Domain classifier for multivariate drift. DataFrames preserve real feature names.
result = gpu_domain_classifier(
    reference=baseline_features,
    current=covariate_drift_features,
    features=features,
)

domain_alert_level = policy_alert_level(
    result.auc,
    DOMAIN_AUC_WARNING_THRESHOLD,
    DOMAIN_AUC_CRITICAL_THRESHOLD,
)
print("Domain Classifier Results:")
print(f"  Cross-validated AUC: {result.auc:.4f} +/- {result.cv_auc_std:.4f}")
print(f"  Policy alert level: {domain_alert_level}")

domain_status = {
    "OK": "[OK] No policy-level separation (AUC < 0.60)",
    "WARNING": "[WARN] Distribution-separation warning (0.60 <= AUC < 0.70)",
    "CRITICAL": "[ALERT] Critical multivariate separation (AUC >= 0.70)",
}[domain_alert_level]
print(f"\n{domain_status}")

# %% [markdown]
# The domain classifier complements univariate metrics by asking whether the full feature vector
# looks like it came from a new regime. The alert uses time-blocked cross-validation AUC; the final
# all-row fit exists only to rank feature importance. This avoids treating training-set fit or
# adjacent symbol-days as evidence of separation on unseen observations.

# %% [markdown]
# Feature importance from the classifier tells us where the multivariate signal is coming from.
# That helps convert a generic drift alert into a concrete investigation queue for the risk team.


# %%
def plot_domain_classifier_results(result: DomainClassifierResult):
    """Visualize domain classifier drift detection."""

    # Feature importance - from polars DataFrame
    fi_df = result.feature_importances.sort("importance").with_columns(
        pl.col("feature").replace_strict(FEATURE_LABELS, default=pl.col("feature")).alias("label")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=fi_df["label"].to_list(),
            x=fi_df["importance"].to_list(),
            orientation="h",
            marker_color=COLORS["blue"],
        )
    )

    top_feature = fi_df.tail(1)["label"][0]
    alert_level = policy_alert_level(
        result.auc,
        DOMAIN_AUC_WARNING_THRESHOLD,
        DOMAIN_AUC_CRITICAL_THRESHOLD,
    )
    fig.update_layout(
        title=(
            "Which features let a classifier tell the two samples apart"
            "<br><sup>Split counts from the cross-validated domain classifier</sup>"
        ),
        xaxis_title="LightGBM split count",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=120),
    )

    return fig


# %% [markdown]
# We expect the top-ranked features here to overlap with the largest PSI and Wasserstein moves,
# but not necessarily match perfectly because the classifier captures interactions too.

# %%
fig = plot_domain_classifier_results(result)
show_plotly_with_alt(
    fig,
    "A horizontal bar chart of split counts per feature from the domain classifier, showing which features it relied on to separate the two samples.",
)

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
drift_summary = checked_drift_analysis(
    reference=baseline_features,
    current=covariate_drift_features,
    features=features,
)

print("=" * 60)
print("COMPREHENSIVE DRIFT ANALYSIS REPORT")
print("=" * 60)
print(f"\nOverall Drift Detected: {'[ALERT] YES' if drift_summary.overall_drifted else '[OK] NO'}")
print(f"Consensus Drifted Features: {drift_summary.drifted_features}")
print(f"Methods Used: {drift_summary.methods_used}")

print("\nPer-Feature Results:")
for fr in drift_summary.feature_results:
    status = "[ALERT]" if fr.drifted else "[OK]"
    psi_val = f"{fr.psi_result.psi:.4f}" if fr.psi_result else "N/A"
    print(f"  {fr.feature:20} PSI={psi_val:>8} {status}")

if drift_summary.domain_classifier_result:
    domain_result = drift_summary.domain_classifier_result
    domain_level = policy_alert_level(
        domain_result.auc,
        DOMAIN_AUC_WARNING_THRESHOLD,
        DOMAIN_AUC_CRITICAL_THRESHOLD,
    )
    print(f"\nDomain Classifier AUC: {domain_result.auc:.4f}")
    print(f"Policy alert level: {domain_level}")

# %% [markdown]
# The unified report requires both univariate methods to flag a feature; the multivariate alert is
# separate. This prevents a dependence-sensitive distance threshold from silently overriding a
# stable PSI result.

# %% [markdown]
# ---
#
# ## Part 6: Case Study - ETF Rotational Momentum
#
# Monitor feature drift across quarters for a daily equity momentum strategy.

# %% [markdown]
# ### A Real Drift Episode, Quarter by Quarter
#
# The last quarter of 2019 is the baseline: an ordinary market. The four quarters of 2020 that
# follow it contain a crash, a rebound and a partial recovery, so the sequence shows drift
# arriving and then partly reversing rather than a single before-and-after comparison.

# %%
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
# This sequence exposes the monitor to a real calm-to-crash-to-rebound progression. Q1 2020
# contains the COVID volatility spike, Q2 the sharp rebound, and Q3-Q4 the partial normalization.
# The point is to test whether the monitoring stack catches the transition and whether it remains
# elevated while the distribution is still unlike the reference quarter.

# %%
# Track drift over time
etf_features = ["momentum_20", "momentum_60", "volatility_20", "rsi_14", "volume_ratio"]
reference = quarterly_data["Q4_2019"]

drift_tracking = []

for quarter in quarters:
    current = quarterly_data[quarter]

    # Compute drift metrics
    summary = checked_drift_analysis(
        reference=reference,
        current=current,
        features=etf_features,
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
drift_df["quarter_label"] = drift_df["quarter"].str.replace("_", " ")
drift_df["feature_label"] = drift_df["most_drifted"].map(FEATURE_LABELS)

# %% [markdown]
# The dashboard is the operational summary a desk would review. Average PSI, out-of-fold domain
# AUC, the leading feature, and the alert sequence show whether the environment is moving away
# from the reference window or merely registering a one-off anomaly.

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Average PSI Over Time",
        "Domain Classifier AUC",
        "Maximum feature PSI",
        "Drift Alert Timeline",
    ),
)

# Average PSI
_ = fig.add_trace(
    go.Scatter(
        x=drift_df["quarter_label"],
        y=drift_df["avg_psi"],
        mode="lines+markers",
        name="Avg PSI",
        marker=dict(size=10, color=COLORS["blue"]),
    ),
    row=1,
    col=1,
)
_ = fig.add_hline(
    y=PSI_WARNING_THRESHOLD, line_dash="dash", line_color=COLORS["amber"], row=1, col=1
)
_ = fig.add_hline(
    y=PSI_CRITICAL_THRESHOLD, line_dash="dash", line_color=COLORS["negative"], row=1, col=1
)

# %% [markdown]
# The adjacent panel applies the same policy boundaries to multivariate domain separation.

# %%
# Domain AUC
colors = [
    ALERT_COLORS[
        policy_alert_level(
            auc,
            DOMAIN_AUC_WARNING_THRESHOLD,
            DOMAIN_AUC_CRITICAL_THRESHOLD,
        )
    ]
    for auc in drift_df["domain_auc"]
]
_ = fig.add_trace(
    go.Bar(
        x=drift_df["quarter_label"],
        y=drift_df["domain_auc"],
        marker_color=colors,
        name="Domain AUC",
    ),
    row=1,
    col=2,
)
_ = fig.add_hline(
    y=DOMAIN_AUC_WARNING_THRESHOLD,
    line_dash="dash",
    line_color=COLORS["amber"],
    row=1,
    col=2,
)
_ = fig.add_hline(
    y=DOMAIN_AUC_CRITICAL_THRESHOLD,
    line_dash="dash",
    line_color=COLORS["negative"],
    row=1,
    col=2,
)

# %%
# Most drifted feature
_ = fig.add_trace(
    go.Scatter(
        x=drift_df["quarter_label"],
        y=drift_df["max_psi"],
        mode="lines+markers",
        name="Max PSI",
        customdata=drift_df["feature_label"],
        hovertemplate=("%{x}<br>Maximum PSI=%{y:.3f}<br>Feature=%{customdata}<extra></extra>"),
        line=dict(color=COLORS["copper"]),
        marker=dict(size=15, color=COLORS["copper"]),
    ),
    row=2,
    col=1,
)

# Alert timeline
alert_colors = [COLORS["positive"] if not d else COLORS["negative"] for d in drift_df["has_drift"]]
_ = fig.add_trace(
    go.Scatter(
        x=drift_df["quarter_label"],
        y=["Drift" if drift else "Stable" for drift in drift_df["has_drift"]],
        mode="markers+text",
        text=["Alert" if drift else "OK" for drift in drift_df["has_drift"]],
        textposition="top center",
        name="Drift Alert",
        marker=dict(size=30, color=alert_colors, symbol="circle"),
    ),
    row=2,
    col=2,
)

# %%
fig.update_layout(
    title="ETF drift peaks during the 2020 rebound and remains elevated at year-end",
    height=650,
    showlegend=False,
)
fig.update_xaxes(title_text="Quarter", row=1, col=1)
fig.update_yaxes(title_text="Mean PSI", rangemode="tozero", row=1, col=1)
fig.update_xaxes(title_text="Quarter", row=1, col=2)
fig.update_yaxes(title_text="Cross-validated AUC", range=[0, 1], row=1, col=2)
fig.update_xaxes(title_text="Quarter", row=2, col=1)
fig.update_yaxes(title_text="Maximum PSI", rangemode="tozero", row=2, col=1)
fig.update_xaxes(title_text="Quarter", row=2, col=2)
fig.update_yaxes(
    title_text="Alert state",
    categoryorder="array",
    categoryarray=["Stable", "Drift"],
    row=2,
    col=2,
)
show_plotly_with_alt(
    fig,
    "A heatmap of PSI by feature and quarter on a logarithmic colour scale, with the crisis quarter far darker than the rest across every feature.",
)

# %% [markdown]
# The dashboard turns a raw drift table into an escalation timeline. Multiple alerts within a
# declared lookback can trigger a risk review, while retraining still requires observed performance
# degradation.

# %% [markdown]
# ---
#
# ## Part 7: Case Study - Crypto Perpetual Premium (Rapid Drift)
#
# Crypto markets exhibit rapid regime changes. Here, each comparison is a selected month-long
# market episode built from hourly feature rows. The example does not simulate an hourly or weekly
# monitoring cadence.

# %% [markdown]
# The feature builder joins hourly OHLCV with the premium index by symbol and timestamp, then
# computes rolling inputs without changing the observation cadence.


# %%
def crypto_feature_panel() -> pl.DataFrame:
    """Compute crypto perp features from real OHLCV and funding-premium data."""

    perp = load_crypto_perps()
    premium = load_crypto_premium().select("timestamp", "symbol", "premium_index_close")
    return (
        perp.sort(["symbol", "timestamp"])
        .with_columns(ret_1h=pl.col("close").pct_change().over("symbol"))
        .with_columns(
            volatility_1h=pl.col("ret_1h").rolling_std(24).over("symbol"),
            volume_ratio=(pl.col("volume") / pl.col("volume").rolling_mean(168)).over("symbol"),
            mom_24h=(pl.col("close") / pl.col("close").shift(24) - 1).over("symbol"),
        )
        .join(premium, on=["timestamp", "symbol"], how="left")
        .sort(["symbol", "timestamp"])
        .with_columns(pl.col("premium_index_close").forward_fill().over("symbol"))
    )


# %%
crypto_features = ["premium_index_close", "volatility_1h", "volume_ratio", "mom_24h"]
crypto_panel = crypto_feature_panel()

# %% [markdown]
# The window helper selects one declared month-long episode and returns only complete feature rows.


# %%
def crypto_window(start: str, end: str) -> pd.DataFrame:
    """Slice the crypto panel by datetime window and return a pandas feature frame."""

    return (
        crypto_panel.filter(
            (pl.col("timestamp") >= pl.lit(start).str.to_datetime(time_zone="UTC"))
            & (pl.col("timestamp") < pl.lit(end).str.to_datetime(time_zone="UTC"))
        )
        .select("timestamp", *crypto_features)
        .drop_nulls()
        .to_pandas()
    )


# %% [markdown]
# ### Drift That Arrives in Days Rather Than Quarters
#
# The ETF episode unfolded over quarters. Crypto gives a faster one: the baseline is an ordinary
# month in early 2021, and the comparison windows walk through a second bull month, a later rally,
# the collapse of a large stablecoin and its aftermath. A monitor calibrated to quarterly review
# would learn about the middle window well after it mattered.

# %%
baseline_crypto = crypto_window("2021-01-01", "2021-02-01")
regimes = {
    "Apr 2021 (Bull market)": crypto_window("2021-04-01", "2021-05-01"),
    "Oct 2021 (Bull rally)": crypto_window("2021-10-01", "2021-11-01"),
    "May 2022 (LUNA crisis)": crypto_window("2022-05-01", "2022-06-01"),
    "Sep 2022 (Post-LUNA)": crypto_window("2022-09-01", "2022-10-01"),
}

print(f"Baseline (Jan 2021): {len(baseline_crypto):,} symbol-hours")
for label, df in regimes.items():
    print(f"  {label}: {len(df):,} symbol-hours")

# %% [markdown]
# The crypto setup compares an early-2021 baseline with later bull-market, crisis, and post-crisis
# windows. The labels describe market episodes; they do not assert that any window is statistically
# stable. The monitor must establish that from the data.

# %%
# Analyze drift across regimes
crypto_drift = []

for period, current in regimes.items():
    summary = checked_drift_analysis(
        reference=baseline_crypto,
        current=current,
        features=crypto_features,
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

# %% [markdown]
# The colour scale is logarithmic. A crisis window produces PSI values orders of magnitude above
# the alert bands, and on a linear scale those few cells would compress every warning-level value
# into an indistinguishable block at the bottom.

# %%
# Heatmap of PSI across features and periods
psi_matrix = crypto_drift_df[[f"psi_{f}" for f in crypto_features]].values
log_psi = np.log1p(psi_matrix)
period_labels = [period.split(" (")[0] for period in crypto_drift_df["period"]]
feature_labels = ["Premium index", "1-hour volatility", "Volume ratio", "24-hour momentum"]

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "heatmap"}, {"type": "bar"}]],
    column_widths=[0.72, 0.28],
    subplot_titles=("Feature PSI by market window", "Multivariate separation"),
)
_ = fig.add_trace(
    go.Heatmap(
        z=log_psi.T,
        x=period_labels,
        y=feature_labels,
        colorscale=[
            [0.0, COLORS["bg_light"]],
            [0.35, COLORS["amber"]],
            [1.0, COLORS["negative"]],
        ],
        customdata=psi_matrix.T,
        hovertemplate="%{y}<br>%{x}<br>PSI=%{customdata:.3f}<extra></extra>",
        colorbar=dict(
            title="PSI",
            tickvals=np.log1p([0, 0.1, 0.25, 1, 5, 10]),
            ticktext=["0", "0.10", "0.25", "1", "5", "10"],
        ),
        zmin=0,
        zmax=np.log1p(10),
    ),
    row=1,
    col=1,
)

# %% [markdown]
# The second panel summarizes multivariate separation for the same four windows.

# %%
_ = fig.add_trace(
    go.Bar(
        x=period_labels,
        y=crypto_drift_df["domain_auc"],
        marker_color=[
            ALERT_COLORS[
                policy_alert_level(
                    value,
                    DOMAIN_AUC_WARNING_THRESHOLD,
                    DOMAIN_AUC_CRITICAL_THRESHOLD,
                )
            ]
            for value in crypto_drift_df["domain_auc"]
        ],
        showlegend=False,
    ),
    row=1,
    col=2,
)
_ = fig.add_hline(
    y=DOMAIN_AUC_WARNING_THRESHOLD,
    line_dash="dash",
    line_color=COLORS["amber"],
    row=1,
    col=2,
)
_ = fig.add_hline(
    y=DOMAIN_AUC_CRITICAL_THRESHOLD,
    line_dash="dash",
    line_color=COLORS["negative"],
    row=1,
    col=2,
)

# %% [markdown]
# Shared layout choices keep the two diagnostics aligned while retaining their distinct scales.

# %%
fig.update_layout(
    title="Premium-index drift dominates the crisis and post-crisis windows",
    height=520,
    margin=dict(l=150, r=40, b=90, t=100),
    showlegend=False,
)
fig.update_xaxes(title_text="Market window", tickangle=0, row=1, col=1)
fig.update_yaxes(title_text="Feature", row=1, col=1)
fig.update_xaxes(title_text="Market window", tickangle=-30, row=1, col=2)
fig.update_yaxes(title_text="Cross-validated AUC", range=[0, 1], row=1, col=2)
show_plotly_with_alt(
    fig,
    "A heatmap of PSI by crypto feature and comparison window on a logarithmic scale, with the collapse window standing out sharply from the bull-market ones.",
)

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
    psi_warning_threshold: float = PSI_WARNING_THRESHOLD
    psi_critical_threshold: float = PSI_CRITICAL_THRESHOLD
    domain_auc_warning_threshold: float = DOMAIN_AUC_WARNING_THRESHOLD
    domain_auc_critical_threshold: float = DOMAIN_AUC_CRITICAL_THRESHOLD
    min_samples_for_check: int = 100


# %% [markdown]
# This monitor keeps only the essential control loop: compare against a reference window, record
# the most important diagnostics, and decide whether the recent alert pattern warrants retraining.


# %%
def summarize_drift_check(
    summary: DriftSummaryResult,
    config: DriftMonitorConfig,
    timestamp: str,
) -> dict:
    """Extract the policy-facing fields from a completed drift analysis."""

    psi_by_feature = {
        result.feature: result.psi_result.psi
        for result in summary.feature_results
        if result.psi_result
    }
    max_psi = max(psi_by_feature.values()) if psi_by_feature else 0
    domain_auc = summary.domain_classifier_result.auc if summary.domain_classifier_result else 0.5
    most_drifted = max(psi_by_feature, key=psi_by_feature.get) if psi_by_feature else None
    alert_level = combined_alert_level(
        max_psi,
        domain_auc,
        psi_warning_threshold=config.psi_warning_threshold,
        psi_critical_threshold=config.psi_critical_threshold,
        domain_auc_warning_threshold=config.domain_auc_warning_threshold,
        domain_auc_critical_threshold=config.domain_auc_critical_threshold,
    )
    return {
        "timestamp": timestamp,
        "status": alert_level,
        "max_psi": max_psi,
        "domain_auc": domain_auc,
        "most_drifted_feature": most_drifted,
        "psi_by_feature": psi_by_feature,
    }


# %% [markdown]
# The stateful monitor owns the reference data and alert history; the helper above owns result
# extraction so the control-loop methods stay compact.


# %%
class ProductionDriftMonitor:
    """Teaching implementation of a drift-monitoring control loop."""

    def __init__(self, reference_df: pd.DataFrame, config: DriftMonitorConfig):
        self.reference_df = reference_df
        self.config = config
        self.drift_history = []

    def check_drift(self, current_df: pd.DataFrame, timestamp: str) -> dict:
        """Check for drift and return alert if detected."""
        if len(current_df) < self.config.min_samples_for_check:
            return {"status": "insufficient_data", "samples": len(current_df)}
        summary = checked_drift_analysis(
            reference=self.reference_df,
            current=current_df,
            features=self.config.feature_columns,
            psi_warning_threshold=self.config.psi_warning_threshold,
            psi_critical_threshold=self.config.psi_critical_threshold,
            domain_auc_warning_threshold=self.config.domain_auc_warning_threshold,
            domain_auc_critical_threshold=self.config.domain_auc_critical_threshold,
        )
        result = summarize_drift_check(summary, self.config, timestamp)
        self.drift_history.append(result)
        return result

    def get_dashboard_data(self) -> pd.DataFrame:
        """Get drift history for dashboard visualization."""
        return pd.DataFrame(self.drift_history)

    def should_retrain(self, performance_degraded: bool, lookback: int = 5) -> tuple[bool, str]:
        """Require performance degradation and enough alerts anywhere in the lookback."""
        if not performance_degraded:
            return False, "No realized-performance degradation supplied"
        if len(self.drift_history) < lookback:
            return False, "Insufficient history"
        recent = self.drift_history[-lookback:]
        return alert_count_condition([record["status"] for record in recent])


# %%
# Demo usage
config = DriftMonitorConfig(
    feature_columns=crypto_features,
    psi_warning_threshold=PSI_WARNING_THRESHOLD,
    psi_critical_threshold=PSI_CRITICAL_THRESHOLD,
    domain_auc_warning_threshold=DOMAIN_AUC_WARNING_THRESHOLD,
    domain_auc_critical_threshold=DOMAIN_AUC_CRITICAL_THRESHOLD,
)

monitor = ProductionDriftMonitor(baseline_crypto, config)

# Apply the monitor to the selected episode windows
print("Production Drift Monitoring Simulation:")
print("=" * 60)

for period, data in regimes.items():
    result = monitor.check_drift(data, period)

    status_emoji = (
        "[ALERT]"
        if result["status"] == "CRITICAL"
        else "[WARN]"
        if result["status"] == "WARNING"
        else "[OK]"
    )
    print(f"\n{status_emoji} {result['timestamp']}")
    print(f"   Status: {result['status']}")
    print(f"   Max PSI: {result['max_psi']:.4f}")
    print(f"   Domain AUC: {result['domain_auc']:.4f}")
    print(f"   Most Drifted: {result['most_drifted_feature']}")

# Check if retraining needed
should_retrain, reason = monitor.should_retrain(performance_degraded=False, lookback=4)
print(f"\n{'=' * 60}")
print(f"Retraining Recommendation: {'YES [RETRAIN]' if should_retrain else 'NO [OK]'}")
print(f"Reason: {reason}")

# %% [markdown]
# This simulation illustrates an operational decision rule rather than a production-tuned
# threshold. Two critical or three warning observations anywhere among the last `lookback` checks
# satisfy the alert-count condition. The selected episode windows are not consecutive calendar
# periods, and retraining still requires labeled performance to have deteriorated.

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
# ### Demonstration Workflow
#
# 1. **Scheduled review**: Compare a current month-long window with the reference window.
# 2. **Hourly rows**: Treat the row frequency as input granularity, not monitoring cadence.
# 3. **Warning or critical alert**: Investigate the affected features and the operating regime.
# 4. **Retraining decision**: Require the any-N-in-lookback alert count and performance degradation.
#
# ### What Goes Wrong
#
# 1. **Retraining on every alert.** Each retrain fits the model to a shorter, more recent window,
#    which raises its variance and guarantees the next alert arrives sooner. A drift alert is a
#    reason to look, not a reason to refit.
# 2. **Treating feature drift as failure.** Inputs moving does not mean the relationship between
#    inputs and outcome moved. A model can be entirely healthy on a distribution it has never
#    seen, and can be broken on one identical to its training data - the second is concept drift
#    and none of these measures detects it.
# 3. **Alerting on samples too small to carry the measure.** PSI computed over a few hundred rows
#    moves substantially between adjacent windows with nothing behind it. The alert threshold and
#    the window size have to be set together.
# 4. **Monitoring the inputs and not the outcome.** Drift matters when it degrades performance.
#    A monitor that never joins the two reports weather.

# %% [markdown]
# ### Drift Detection Quick Reference
#
# The alert bands for both measures are the parameters printed at the top of the notebook.
#
# **Action Matrix**
#
# | Drift Status | Performance | Action |
# |-------------|-------------|--------|
# | Drift | OK | Monitor, don't retrain |
# | Drift | Down | Apply the any-N-in-lookback retraining rule |
# | No Drift | Down | Check for concept drift |
# | No Drift | OK | Keep monitoring |

# %% [markdown]
# ## Key Takeaways

# %% tags=["results"]
etf_psi_rank = sorted(psi_results.items(), key=lambda item: item[1].psi, reverse=True)
peak_quarter = drift_df.loc[drift_df["avg_psi"].idxmax()]
crypto_psi_columns = [f"psi_{feature}" for feature in crypto_features]
crypto_peak_column = crypto_drift_df[crypto_psi_columns].max().idxmax()
crypto_peak_feature = crypto_peak_column.removeprefix("psi_")
crypto_peak_value = crypto_drift_df[crypto_peak_column].max()
critical_alerts = sum(record["status"] == "CRITICAL" for record in monitor.drift_history)
display(
    Markdown(
        f"Across the ETF features, **{etf_psi_rank[0][0]}** carries the largest PSI at "
        f"**{etf_psi_rank[0][1].psi:.2f}**. Mean PSI across the quarterly panel peaks in "
        f"**{peak_quarter['quarter_label']}** at **{peak_quarter['avg_psi']:.2f}**. On the crypto "
        f"panel the "
        f"largest single reading is **{crypto_peak_feature}** at **{crypto_peak_value:.2f}**. The "
        f"production monitor raised **{critical_alerts} critical alerts** over its history."
    )
)

# %% [markdown]
# 1. **Measure drift against a reference window you can defend.** Every number here is a
#    comparison, so it inherits whatever the baseline happened to contain. A baseline drawn from a
#    calm period makes ordinary variation look like drift; one spanning a crisis absorbs the next
#    crisis into the reference and reports nothing.
#
# 2. **Read the bin-level breakdown, not just the index.** PSI is a sum over bins, and the same
#    total arises from a uniform shift and from one bin emptying. Only the second is usually a data
#    problem, and only the breakdown distinguishes them.
#
# 3. **Use a multivariate check as well as per-feature ones.** Features can each stay within their
#    own historical range while their joint distribution moves - a correlation breaking, a
#    combination that never previously occurred. A classifier trained to tell the two samples apart
#    sees that; a per-feature index cannot.
#
# 4. **Split the domain classifier's folds by time block, not at random.** Neighbouring rows are
#    nearly identical, so random folds let the classifier memorize periods and score well without
#    any drift being present. That is the difference between detecting drift and detecting
#    autocorrelation.
#
# 5. **Match the monitoring cadence to how fast the market being monitored moves.** The ETF
#    episode here unfolds over quarters and the crypto one over days. A monitor reviewed monthly
#    would have caught the first and learned about the second long after it mattered.
#
# 6. **Require drift and degradation before retraining.** Drift alone is a reason to investigate.
#    Retraining on drift alone shortens the training window every time and makes the next alert
#    more likely, which is a loop rather than a control.
#
# ### Known limitations
#
# - The alert thresholds are the conventional rules of thumb. Nothing here calibrates them against
#   a period where it is known whether drift occurred, so they say where the convention puts the
#   line and not where this data would.
# - The windows on both panels are chosen by hand around episodes already known to contain drift.
#   That is right for showing what the measures do and wrong for estimating how often they fire in
#   ordinary conditions, which is what a production threshold has to be set against.
# - No model performance is tracked alongside the drift. The retraining rule requires degradation
#   as its second condition and the notebook supplies that flag rather than measuring it, so the
#   loop from drift through performance to a decision is demonstrated, not closed.
# - PSI depends on the binning. The reference quantiles fix it here, and a different bin count
#   would move every value reported.
#
# **Next**: `08_ml_exit_signals` builds the two-model entry-and-exit architecture the exit rules in
# `02_exit_strategies` stood in for.
#
# **Book reference**: Chapter 19, Sections 19.7 and 19.8.
