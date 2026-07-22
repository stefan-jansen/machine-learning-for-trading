# ---
# jupyter:
#   jupytext:
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

# %% [markdown] papermill={"duration": 0.011963, "end_time": "2026-05-15T14:10:06.127024+00:00", "exception": false, "start_time": "2026-05-15T14:10:06.115061+00:00", "status": "completed"}
# # Feature Evaluation: NASDAQ-100 Microstructure
#
# **Chapter 8-9 Consolidation**
#
# Consolidated evaluation of Ch8 financial features (66) and Ch9 temporal features
# (22) against 15-minute forward midprice returns. Produces triage decisions for
# Ch11 modeling.
#
# **Learning Objectives**:
# - Evaluate 88 features jointly using HAC-adjusted IC and BH-FDR correction
# - Screen for coverage and staleness issues (critical for intraday data)
# - Assess quantile monotonicity and cross-feature redundancy
# - Triage features into PROCEED / REVISE / STOP categories
#
# **Output Contract**:
# - `evaluation/triage_ledger.parquet` — one row per feature with triage decision
# - `evaluation/ic_timeseries.parquet` — long-format IC time series
#
# **Cross-References**:
# - **Upstream**: `03_financial_features.py` (66 features), `04_temporal.py` (22 features),
#   [`02_labels`](02_labels.ipynb) (fwd_ret_15m labels)
# - **Downstream**: Ch11 (model training, feature selection)

# %% papermill={"duration": 2.655142, "end_time": "2026-05-15T14:10:08.790098+00:00", "exception": false, "start_time": "2026-05-15T14:10:06.134956+00:00", "status": "completed"}
"""Feature Evaluation - NASDAQ-100 Microstructure

Consolidated evaluation of Ch8 financial features and Ch9 temporal features
against forward return labels. Produces triage decisions for Ch11 modeling.
"""

import warnings

import yaml

warnings.filterwarnings("ignore")

import numpy as np
import plotly.graph_objects as go
import polars as pl
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from utils.data_quality import validate_modeling_inputs
from utils.paths import get_case_study_dir


def _normalize_symbol_column(df: pl.DataFrame) -> pl.DataFrame:
    if "symbol" in df.columns:
        return df
    msg = f"Expected symbol-like column in frame, found columns={df.columns}"
    raise KeyError(msg)


# %% papermill={"duration": 0.004816, "end_time": "2026-05-15T14:10:08.796541+00:00", "exception": false, "start_time": "2026-05-15T14:10:08.791725+00:00", "status": "completed"} tags=["parameters"]
# Production defaults — Papermill overrides for testing
# MAX_SYMBOLS: 0 = all symbols; set to 20 for fast CI
MAX_SYMBOLS = 0

# %% papermill={"duration": 0.005763, "end_time": "2026-05-15T14:10:08.804288+00:00", "exception": false, "start_time": "2026-05-15T14:10:08.798525+00:00", "status": "completed"}
CASE_STUDY_ID = "nasdaq100_microstructure"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
EVAL_DIR = CASE_DIR / "evaluation"
EVAL_WRITE_DIR = EVAL_DIR.resolve(strict=False) if EVAL_DIR.is_symlink() else EVAL_DIR
EVAL_WRITE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
JOIN_COLS = ["timestamp", "symbol"]
DATE_COL = "timestamp"
HAC_MAXLAGS = 26  # One full session (26 bars × 15 min = 6.5 hours) for Newey-West
MIN_PERIODS_DEFAULT = 10  # 101-114 symbols → 10 is conservative
IC_THRESHOLD = 0.003  # Intraday, very low bar

# %% [markdown] papermill={"duration": 0.001775, "end_time": "2026-05-15T14:10:08.808127+00:00", "exception": false, "start_time": "2026-05-15T14:10:08.806352+00:00", "status": "completed"}
# ## 0. Load Artifacts and Build Evaluation Panel
#
# Load Ch8 financial features, Ch9 temporal features, and fwd_ret_15m labels.
# Build a unified evaluation panel by joining all three on `(timestamp, symbol)`.

# %% papermill={"duration": 11.251787, "end_time": "2026-05-15T14:10:20.061358+00:00", "exception": false, "start_time": "2026-05-15T14:10:08.809571+00:00", "status": "completed"}
# Identify feature columns before loading (peek at schema)
features_path = CASE_DIR / "features" / "financial.parquet"
temporal_path = CASE_DIR / "features" / "model_based.parquet"
label_path = CASE_DIR / "labels" / "fwd_ret_15m.parquet"

financial_cols = [
    c for c in pl.read_parquet_schema(features_path) if c not in ("timestamp", "symbol")
]
temporal_cols = [
    c for c in pl.read_parquet_schema(temporal_path) if c not in ("timestamp", "symbol", "fold")
]
all_feature_cols = financial_cols + temporal_cols

# Determine symbol filter (if MAX_SYMBOLS > 0, pick top symbols by count)
symbol_filter = None
if MAX_SYMBOLS > 0:
    counts = (
        _normalize_symbol_column(pl.scan_parquet(features_path).collect()).group_by("symbol").len()
    )
    top_syms = counts.sort("len", descending=True).head(MAX_SYMBOLS)["symbol"].to_list()
    symbol_filter = pl.col("symbol").is_in(top_syms)

# Load with optional filter — avoids materializing full 13.8M rows in test mode
features = pl.scan_parquet(features_path)
temporal = pl.scan_parquet(temporal_path)
label_df = pl.scan_parquet(label_path)

if symbol_filter is not None:
    features = features.filter(symbol_filter)
    temporal = temporal.filter(symbol_filter)
    label_df = label_df.filter(symbol_filter)

features = _normalize_symbol_column(features.collect())
temporal = _normalize_symbol_column(temporal.collect())
# Drop fold column (per-fold features are for Ch11+ training, not evaluation)
if "fold" in temporal.columns:
    temporal = temporal.drop("fold").unique(subset=["timestamp", "symbol"], keep="last")
label_df = _normalize_symbol_column(label_df.collect())

# Auto-detect label column
label_col = [c for c in label_df.columns if c not in ("timestamp", "symbol")][0]

# Load CV config for fold-level analysis
with open(CASE_DIR / "config" / "setup.yaml") as f:
    cv_config = yaml.safe_load(f)["evaluation"]

# %% [markdown] papermill={"duration": 0.00264, "end_time": "2026-05-15T14:10:20.067024+00:00", "exception": false, "start_time": "2026-05-15T14:10:20.064384+00:00", "status": "completed"}
# ## 0.5 Data Quality Gate
#
# Verify upstream artifacts are free of critical defects before evaluation.

# %% papermill={"duration": 2.287479, "end_time": "2026-05-15T14:10:22.356095+00:00", "exception": false, "start_time": "2026-05-15T14:10:20.068616+00:00", "status": "completed"}
validate_modeling_inputs(
    features_df=features,
    label_df=label_df,
    feature_cols=financial_cols + temporal_cols,
    label_col=label_col,
    join_cols=JOIN_COLS,
    asset_col="symbol",
    max_abs_return=1.0,  # 15-minute returns (max observed ~0.60, circuit breakers)
    fail_on_critical=True,
)

# %% papermill={"duration": 5.529039, "end_time": "2026-05-15T14:10:27.887567+00:00", "exception": false, "start_time": "2026-05-15T14:10:22.358528+00:00", "status": "completed"}
# Build unified eval panel, then free source DataFrames
eval_panel = features.join(temporal, on=JOIN_COLS, how="left")
eval_panel = eval_panel.join(label_df, on=JOIN_COLS, how="inner")
del features, temporal, label_df

n_rows = len(eval_panel)
n_symbols = eval_panel["symbol"].n_unique()
n_dates = eval_panel[DATE_COL].n_unique()
MIN_PERIODS = min(MIN_PERIODS_DEFAULT, n_symbols)
print(f"Eval panel: {n_rows:,} rows, {n_symbols} symbols, {n_dates:,} timestamps")
print(
    f"Features: {len(financial_cols)} financial + {len(temporal_cols)} temporal"
    f" = {len(all_feature_cols)} total"
)
print(f"Label: {label_col}")

# %% [markdown] papermill={"duration": 0.011368, "end_time": "2026-05-15T14:10:27.900720+00:00", "exception": false, "start_time": "2026-05-15T14:10:27.889352+00:00", "status": "completed"}
# ## 1. Correctness Screens
#
# Check coverage (fraction non-null) and staleness (fraction unchanged from prior
# bar) for every feature before evaluating predictive power. Intraday data is
# especially prone to stale quotes — a quote-based feature that never updates
# signals stale NBBO.
#
# **Gate**: PASS if coverage >= 70% AND staleness <= 50%.

# %% papermill={"duration": 50.666787, "end_time": "2026-05-15T14:11:18.569933+00:00", "exception": false, "start_time": "2026-05-15T14:10:27.903146+00:00", "status": "completed"}
# Coverage: fraction non-null per feature
coverage = {}
for feat in all_feature_cols:
    n_valid = eval_panel[feat].drop_nulls().len()
    coverage[feat] = n_valid / n_rows if n_rows > 0 else 0.0

# Staleness: fraction of rows where feature value is unchanged from prior bar
# (especially important for intraday microstructure data)
# Vectorized: sort once, compute all shift comparisons in one with_columns
sorted_panel = eval_panel.sort(JOIN_COLS)
stale_exprs = [
    (pl.col(f) == pl.col(f).shift(1).over("symbol")).alias(f"_stale_{f}") for f in all_feature_cols
]
stale_df = sorted_panel.select(stale_exprs)
staleness = {
    f: float(stale_df[f"_stale_{f}"].sum()) / max(n_rows - n_symbols, 1) for f in all_feature_cols
}
del stale_df, sorted_panel

# %% papermill={"duration": 0.006863, "end_time": "2026-05-15T14:11:18.578585+00:00", "exception": false, "start_time": "2026-05-15T14:11:18.571722+00:00", "status": "completed"}
# Apply correctness gate
correctness = {}
for feat in all_feature_cols:
    cov_ok = coverage[feat] >= 0.70
    stale_ok = staleness[feat] <= 0.50
    correctness[feat] = cov_ok and stale_ok

n_pass = sum(correctness.values())
n_fail = len(correctness) - n_pass
print(f"Correctness gate: {n_pass} PASS, {n_fail} FAIL")

if n_fail > 0:
    fail_df = pl.DataFrame(
        {
            "feature": [f for f, ok in correctness.items() if not ok],
            "coverage": [coverage[f] for f, ok in correctness.items() if not ok],
            "staleness": [staleness[f] for f, ok in correctness.items() if not ok],
        }
    )
    print(fail_df)

# %% [markdown] papermill={"duration": 0.001493, "end_time": "2026-05-15T14:11:18.581661+00:00", "exception": false, "start_time": "2026-05-15T14:11:18.580168+00:00", "status": "completed"}
# ## 2. Univariate Association (IC + HAC)
#
# Compute the Information Coefficient (IC) — cross-sectional Spearman rank
# correlation between each feature and `fwd_ret_15m` — at every sampled timestamp.
# HAC standard errors (Newey-West, bandwidth=26) correct for serial correlation in
# the IC time series.
#
# **Sampling**: Every 15th timestamp (~1 per label horizon) to approximate
# independence. With the primary label spanning 15 bars, consecutive IC observations
# at 15-bar spacing are approximately non-overlapping.
#
# **N_eff**: The effective sample size is approximately $N_{\text{timestamps}} / 15$
# (one independent observation per label horizon). With ~136K total timestamps and
# 15-bar sampling, we retain ~9K IC observations — well above the ~100 minimum for
# stable HAC estimates.
#
# **Note**: IC is computed across the full 2020-2021 sample, including the 2021-H2
# holdout period. This provides an upper bound on predictive content. Feature
# triage decisions should be interpreted accordingly — features that appear
# significant here may not survive on pre-holdout data alone.

# %% papermill={"duration": 0.345248, "end_time": "2026-05-15T14:11:18.928453+00:00", "exception": false, "start_time": "2026-05-15T14:11:18.583205+00:00", "status": "completed"}
evaluable_features = [f for f in all_feature_cols if correctness[f]]

# Sample every 15th timestamp to approximate independence (label horizon = 15 bars)
all_timestamps = eval_panel[DATE_COL].unique().sort()
sample_step = 15
sample_ts = all_timestamps.gather_every(sample_step)
eval_sampled = eval_panel.filter(pl.col(DATE_COL).is_in(sample_ts))
print(
    f"Evaluating {len(evaluable_features)} features on {len(sample_ts):,}"
    f" sampled timestamps ({len(eval_sampled):,} rows)"
)

# %% papermill={"duration": 0.264749, "end_time": "2026-05-15T14:11:19.194973+00:00", "exception": false, "start_time": "2026-05-15T14:11:18.930224+00:00", "status": "completed"}
# Detect date-level features (zero cross-sectional variance)
cs_std_df = eval_sampled.group_by(DATE_COL).agg(
    [pl.col(f).std().alias(f) for f in evaluable_features]
)
date_level_features = set()
for feat in evaluable_features:
    mean_std = cs_std_df[feat].drop_nulls().mean()
    if mean_std is not None and mean_std < 1e-10:
        date_level_features.add(feat)

if date_level_features:
    print(f"Date-level features (zero CS variance): {sorted(date_level_features)}")

# %% papermill={"duration": 0.662339, "end_time": "2026-05-15T14:11:19.860584+00:00", "exception": false, "start_time": "2026-05-15T14:11:19.198245+00:00", "status": "completed"}
# Vectorized IC: Polars group_by + pl.corr computes all features × timestamps at once
cs_features = [f for f in evaluable_features if f not in date_level_features]

# Filter timestamps with enough observations
ts_counts = eval_sampled.group_by(DATE_COL).len().filter(pl.col("len") >= MIN_PERIODS)
eval_sub = eval_sampled.filter(pl.col(DATE_COL).is_in(ts_counts[DATE_COL]))

# Compute cross-sectional Spearman IC for all features at every timestamp
ic_wide = (
    eval_sub.group_by(DATE_COL)
    .agg(
        [pl.corr(f, label_col, method="spearman").alias(f) for f in cs_features]
        + [pl.len().alias("n_obs")]
    )
    .sort(DATE_COL)
)

print(f"IC computed: {len(cs_features)} features × {len(ic_wide):,} timestamps")

# %% papermill={"duration": 0.059671, "end_time": "2026-05-15T14:11:19.923603+00:00", "exception": false, "start_time": "2026-05-15T14:11:19.863932+00:00", "status": "completed"}
# Unpivot to per-feature IC series and compute HAC stats
ic_results = {}
ic_timeseries = {}
for feat in cs_features:
    ic_df = ic_wide.select([DATE_COL, pl.col(feat).alias("ic"), "n_obs"]).drop_nulls(subset=["ic"])
    if len(ic_df) < 20:
        continue
    hac_stats = compute_ic_hac_stats(ic_df, ic_col="ic", maxlags=HAC_MAXLAGS)
    ic_results[feat] = hac_stats
    ic_timeseries[feat] = ic_df

print(f"HAC stats for {len(ic_results)} cross-sectional features")
print(f"Skipped {len(date_level_features)} date-level features")

# %% [markdown] papermill={"duration": 0.001682, "end_time": "2026-05-15T14:11:19.927786+00:00", "exception": false, "start_time": "2026-05-15T14:11:19.926104+00:00", "status": "completed"}
# ### Fold-Level Stability
#
# Check sign consistency across CV folds. A feature whose IC flips sign between
# folds is unreliable regardless of its full-sample t-statistic.

# %% papermill={"duration": 0.026864, "end_time": "2026-05-15T14:11:19.956539+00:00", "exception": false, "start_time": "2026-05-15T14:11:19.929675+00:00", "status": "completed"}
# Generate approximate fold boundaries from sampled IC timestamps
all_ic_dates = ic_wide[DATE_COL].to_list()
n_folds = cv_config.get("n_splits", 2)
fold_size = len(all_ic_dates) // n_folds if n_folds > 0 else 0

fold_boundaries = []
if fold_size > 0:
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = min((i + 1) * fold_size - 1, len(all_ic_dates) - 1)
        fold_boundaries.append((all_ic_dates[start_idx], all_ic_dates[end_idx]))

fold_stats = {}
for feat in ic_results:
    fold_ics = []
    ts = ic_timeseries[feat]
    for fold_start, fold_end in fold_boundaries:
        fold_ic = ts.filter((pl.col(DATE_COL) >= fold_start) & (pl.col(DATE_COL) <= fold_end))
        if len(fold_ic) >= 5:
            fold_ics.append(float(fold_ic["ic"].mean()))

    if fold_ics:
        sign_consistency = sum(1 for ic in fold_ics if ic > 0) / len(fold_ics)
        fold_stats[feat] = {
            "n_folds": len(fold_ics),
            "sign_consistency": sign_consistency,
            "worst_fold_ic": min(fold_ics),
            "best_fold_ic": max(fold_ics),
            "median_fold_ic": float(np.median(fold_ics)),
        }

n_consistent = sum(1 for s in fold_stats.values() if s["sign_consistency"] >= 0.6)
print(f"Fold stability: {n_consistent}/{len(fold_stats)} features with sign consistency >= 60%")

# %% [markdown] papermill={"duration": 0.001644, "end_time": "2026-05-15T14:11:19.959880+00:00", "exception": false, "start_time": "2026-05-15T14:11:19.958236+00:00", "status": "completed"}
# ## 3. Multiple Testing (BH-FDR)
#
# Apply Benjamini-Hochberg correction at $\alpha = 0.05$. With 88 features tested,
# the naive 5% threshold expects ~4.4 false positives. FDR controls the expected
# fraction of false discoveries among rejected hypotheses.

# %% papermill={"duration": 0.006369, "end_time": "2026-05-15T14:11:19.967912+00:00", "exception": false, "start_time": "2026-05-15T14:11:19.961543+00:00", "status": "completed"}
feature_names = list(ic_results.keys())
p_values = [ic_results[f]["p_value"] for f in feature_names]

fdr_result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)

# Build evaluation summary
eval_summary = pl.DataFrame(
    {
        "feature": feature_names,
        "source": ["temporal" if f in temporal_cols else "financial" for f in feature_names],
        "ic_mean": [ic_results[f]["mean_ic"] for f in feature_names],
        "hac_se": [ic_results[f]["hac_se"] for f in feature_names],
        "hac_t": [ic_results[f]["t_stat"] for f in feature_names],
        "hac_p": p_values,
        "fdr_p": list(fdr_result["adjusted_p_values"]),
        "fdr_sig": list(fdr_result["rejected"]),
        "naive_t": [ic_results[f]["naive_t_stat"] for f in feature_names],
    },
    schema_overrides={
        "ic_mean": pl.Float64,
        "hac_se": pl.Float64,
        "hac_t": pl.Float64,
        "hac_p": pl.Float64,
        "fdr_p": pl.Float64,
        "fdr_sig": pl.Boolean,
        "naive_t": pl.Float64,
    },
).sort(pl.col("ic_mean").cast(pl.Float64, strict=False).abs(), descending=True)

n_significant_naive = sum(1 for p in p_values if p < 0.05)
n_significant_hac = sum(1 for f in feature_names if abs(ic_results[f]["t_stat"]) > 1.96)
n_significant_fdr = int(fdr_result["n_rejected"])

inflation_hac = n_significant_naive / max(n_significant_hac, 1)
inflation_fdr = n_significant_naive / max(n_significant_fdr, 1)

print(f"Naive significant (p < 0.05): {n_significant_naive}")
print(f"HAC significant (|t| > 1.96): {n_significant_hac}")
print(f"FDR significant (q < 0.05):   {n_significant_fdr}")
print(f"Inflation factor (HAC): {inflation_hac:.2f}x")
print(f"Inflation factor (FDR): {inflation_fdr:.2f}x")

# %% [markdown] papermill={"duration": 0.001782, "end_time": "2026-05-15T14:11:19.971375+00:00", "exception": false, "start_time": "2026-05-15T14:11:19.969593+00:00", "status": "completed"}
# ### IC Bar Chart + HAC Scatter

# %% papermill={"duration": 0.828307, "end_time": "2026-05-15T14:11:20.801398+00:00", "exception": false, "start_time": "2026-05-15T14:11:19.973091+00:00", "status": "completed"}
top_n = min(25, len(eval_summary))
top = eval_summary.head(top_n)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        f"Top {top_n} Features by |IC| (green = FDR-sig)",
        "HAC vs Naive t-statistics",
    ],
    horizontal_spacing=0.12,
)

# Panel 1: IC bar chart
colors = ["#2ecc71" if s else "#95a5a6" for s in top["fdr_sig"].to_list()]
fig.add_trace(
    go.Bar(
        x=top["feature"].to_list(),
        y=top["ic_mean"].to_list(),
        marker_color=colors,
        text=[f"t={t:.1f}" for t in top["hac_t"].to_list()],
        textposition="outside",
        showlegend=False,
    ),
    row=1,
    col=1,
)

# %% papermill={"duration": 0.124541, "end_time": "2026-05-15T14:11:20.928713+00:00", "exception": false, "start_time": "2026-05-15T14:11:20.804172+00:00", "status": "completed"}
# Panel 2: HAC scatter
fig.add_trace(
    go.Scatter(
        x=eval_summary["naive_t"].to_list(),
        y=eval_summary["hac_t"].to_list(),
        mode="markers",
        marker=dict(
            color=["#2ecc71" if s else "#e74c3c" for s in eval_summary["fdr_sig"].to_list()],
            size=7,
        ),
        text=eval_summary["feature"].to_list(),
        showlegend=False,
    ),
    row=1,
    col=2,
)

# 45-degree reference line
max_t = (
    max(
        float(eval_summary["naive_t"].abs().max() or 1.0),
        float(eval_summary["hac_t"].abs().max() or 1.0),
    )
    * 1.1
)
fig.add_trace(
    go.Scatter(
        x=[-max_t, max_t],
        y=[-max_t, max_t],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        showlegend=False,
    ),
    row=1,
    col=2,
)

fig.update_layout(template="plotly_white", height=450, width=1100)
fig.update_xaxes(tickangle=-45, row=1, col=1)
fig.update_xaxes(title_text="Naive t", row=1, col=2)
fig.update_yaxes(title_text="HAC t", row=1, col=2)
fig.show()

# %% [markdown] papermill={"duration": 0.003431, "end_time": "2026-05-15T14:11:20.935767+00:00", "exception": false, "start_time": "2026-05-15T14:11:20.932336+00:00", "status": "completed"}
# ## 4. Shape Diagnostics
#
# Quantile monotonicity: does the mean forward return spread monotonically across
# feature quintiles? A monotone relationship is stronger evidence of a genuine
# signal than a high IC that could reflect outlier-driven correlation.

# %% papermill={"duration": 1.051184, "end_time": "2026-05-15T14:11:21.990307+00:00", "exception": false, "start_time": "2026-05-15T14:11:20.939123+00:00", "status": "completed"}
N_QUANTILES = 5
top_features_for_shape = eval_summary.filter(pl.col("fdr_sig").fill_null(False))[
    "feature"
].to_list()[:15]

if not top_features_for_shape:
    # Fallback: top 10 by absolute IC
    top_features_for_shape = eval_summary.head(10)["feature"].to_list()

monotonicity_scores = {}
quantile_spreads = {}
for feat in top_features_for_shape:
    valid = eval_panel.select([feat, label_col]).drop_nulls()
    if len(valid) < N_QUANTILES * 20:
        continue
    valid = valid.with_columns(
        pl.col(feat)
        .qcut(N_QUANTILES, labels=[f"Q{i + 1}" for i in range(N_QUANTILES)])
        .alias("quantile")
    )
    q_means = valid.group_by("quantile").agg(pl.col(label_col).mean()).sort("quantile")
    means = q_means[label_col].to_list()
    spread = means[-1] - means[0]
    quantile_spreads[feat] = {"q_means": means, "spread": spread}

    mono_corr, _ = spearmanr(range(len(means)), means)  # imported at top
    monotonicity_scores[feat] = float(mono_corr)

n_monotone = sum(1 for s in monotonicity_scores.values() if abs(s) >= 0.8)
print(
    f"Quantile analysis: {n_monotone}/{len(monotonicity_scores)} features with monotonicity >= 0.8"
)

# %% [markdown] papermill={"duration": 0.003441, "end_time": "2026-05-15T14:11:21.997752+00:00", "exception": false, "start_time": "2026-05-15T14:11:21.994311+00:00", "status": "completed"}
# ### Quantile Return Plots

# %% papermill={"duration": 0.078446, "end_time": "2026-05-15T14:11:22.079889+00:00", "exception": false, "start_time": "2026-05-15T14:11:22.001443+00:00", "status": "completed"}
if quantile_spreads:
    n_show = min(6, len(quantile_spreads))
    feats_to_show = list(quantile_spreads.keys())[:n_show]
    ncols = min(3, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=feats_to_show[:n_show])
    for idx, feat in enumerate(feats_to_show):
        r, c = divmod(idx, ncols)
        q_means = quantile_spreads[feat]["q_means"]
        fig.add_trace(
            go.Bar(
                x=[f"Q{i + 1}" for i in range(len(q_means))],
                y=q_means,
                marker_color=["#e74c3c", "#f39c12", "#95a5a6", "#3498db", "#2ecc71"],
                showlegend=False,
            ),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(
        template="plotly_white",
        height=250 * nrows,
        width=900,
        title_text="Quantile Mean Returns (Top Features)",
    )
    fig.show()

# %% [markdown] papermill={"duration": 0.004334, "end_time": "2026-05-15T14:11:22.088826+00:00", "exception": false, "start_time": "2026-05-15T14:11:22.084492+00:00", "status": "completed"}
# ## 5. Redundancy & Feature Families
#
# Identify highly correlated feature pairs and aggregate IC by feature family.
# Correlated features carry redundant information — Ch11 should select one
# representative per cluster.
#
# **Feature families** for microstructure data: quote liquidity, order flow,
# volatility/impact, FINRA, calendar, plus temporal sub-families (HAR, FFT,
# signatures).

# %% papermill={"duration": 0.007182, "end_time": "2026-05-15T14:11:22.100211+00:00", "exception": false, "start_time": "2026-05-15T14:11:22.093029+00:00", "status": "completed"}
# Curated family mapping (consistent with 03_financial_features.py FEATURE_FAMILIES dict)
FINANCIAL_FAMILIES = {
    "rel_spread_close": "A_quote_liquidity",
    "depth_imb": "A_quote_liquidity",
    "quote_rate": "A_quote_liquidity",
    "microprice_dev": "A_quote_liquidity",
    "microprice_dev_5m": "A_quote_liquidity",
    "microprice_dev_15m": "A_quote_liquidity",
    "signed_vol_share": "B_order_flow",
    "tick_imb_share": "B_order_flow",
    "trade_to_mid_rel": "B_order_flow",
    "trades_per_1k_shares": "B_order_flow",
    "cross_locked_share": "B_order_flow",
    "r1m": "C_volatility_impact",
    "rv_5m": "C_volatility_impact",
    "rv_15m": "C_volatility_impact",
    "rv_30m": "C_volatility_impact",
    "rv_ewma_30m": "C_volatility_impact",
    "trade_range": "C_volatility_impact",
    "quote_range": "C_volatility_impact",
    "illiq": "C_volatility_impact",
    "kyle_lambda": "C_volatility_impact",
    "dollar_vol": "C_volatility_impact",
    "finra_share_60m": "D_finra",
    "signed_vol_share_5m": "multi_resolution",
    "signed_vol_share_15m": "multi_resolution",
    "signed_vol_share_60m": "multi_resolution",
    "tick_imb_share_5m": "multi_resolution",
    "tick_imb_share_15m": "multi_resolution",
    "rel_spread_5m": "multi_resolution",
    "rel_spread_15m": "multi_resolution",
    "rel_spread_60m": "multi_resolution",
    "trade_to_mid_5m": "multi_resolution",
}

# %% papermill={"duration": 0.007413, "end_time": "2026-05-15T14:11:22.111836+00:00", "exception": false, "start_time": "2026-05-15T14:11:22.104423+00:00", "status": "completed"}
# Temporal families from 04_temporal.py
TEMPORAL_FAMILIES = {}
for feat in temporal_cols:
    if "har" in feat.lower():
        TEMPORAL_FAMILIES[feat] = "temporal_har"
    elif any(k in feat.lower() for k in ["spectral", "fft", "freq", "period", "entropy"]):
        TEMPORAL_FAMILIES[feat] = "temporal_fft"
    elif "sig" in feat.lower():
        TEMPORAL_FAMILIES[feat] = "temporal_signature"
    else:
        TEMPORAL_FAMILIES[feat] = "temporal_other"

# Merge: curated financial + temporal (calendar/XS features get "calendar" or "other")
families = {}
for feat in all_feature_cols:
    if feat in FINANCIAL_FAMILIES:
        families[feat] = FINANCIAL_FAMILIES[feat]
    elif feat in TEMPORAL_FAMILIES:
        families[feat] = TEMPORAL_FAMILIES[feat]
    elif any(k in feat.lower() for k in ["hour", "minute", "session", "tod_", "dow_"]):
        families[feat] = "calendar"
    else:
        families[feat] = "other"

# %% papermill={"duration": 4.611176, "end_time": "2026-05-15T14:11:26.727307+00:00", "exception": false, "start_time": "2026-05-15T14:11:22.116131+00:00", "status": "completed"}
# Pairwise correlation (sample timestamps for efficiency)
sample_corr_step = max(1, n_dates // 200)
corr_dates = eval_panel[DATE_COL].unique().sort().to_list()[::sample_corr_step]
corr_data = eval_panel.filter(pl.col(DATE_COL).is_in(corr_dates)).select(evaluable_features)
corr_matrix = corr_data.to_pandas().corr(method="spearman")

# Count high-correlation pairs
high_corr_pairs = []
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append(
                (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            )

print(f"Feature pairs with |corr| > 0.7: {len(high_corr_pairs)}")

# %% papermill={"duration": 0.009189, "end_time": "2026-05-15T14:11:26.741105+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.731916+00:00", "status": "completed"}
# Family-level IC summary
family_ic = {}
fdr_sig_features = eval_summary.filter(pl.col("fdr_sig").fill_null(False))["feature"].to_list()

for feat in ic_results:
    fam = families.get(feat, "other")
    family_ic.setdefault(fam, []).append(
        {
            "feature": feat,
            "ic": ic_results[feat]["mean_ic"],
            "fdr_sig": feat in fdr_sig_features,
        }
    )

family_summary = {}
for fam, feats in sorted(family_ic.items()):
    ics = [f["ic"] for f in feats if f["ic"] is not None]
    n_sig = sum(1 for f in feats if f["fdr_sig"])
    family_summary[fam] = {
        "n_features": len(feats),
        "avg_abs_ic": float(np.mean([abs(ic) for ic in ics])) if ics else 0.0,
        "avg_ic": float(np.mean(ics)) if ics else 0.0,
        "n_fdr_sig": n_sig,
    }

if family_summary:
    fam_df = pl.DataFrame([{"family": fam, **stats} for fam, stats in family_summary.items()]).sort(
        "avg_abs_ic", descending=True
    )
    print(fam_df)
else:
    fam_df = pl.DataFrame()
    print("No features passed IC evaluation threshold")

# %% [markdown] papermill={"duration": 0.004353, "end_time": "2026-05-15T14:11:26.749946+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.745593+00:00", "status": "completed"}
# ### Correlation Heatmap

# %% papermill={"duration": 0.11213, "end_time": "2026-05-15T14:11:26.866417+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.754287+00:00", "status": "completed"}
fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
    )
)
fig.update_layout(
    title=f"Feature Correlation Matrix ({len(high_corr_pairs)} pairs above 0.7)",
    template="plotly_white",
    height=700,
    width=800,
)
fig.show()

# %% [markdown] papermill={"duration": 0.005363, "end_time": "2026-05-15T14:11:26.877665+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.872302+00:00", "status": "completed"}
# ## 6. Triage & Handoff
#
# Classify each feature as:
#
# | Decision | Criteria |
# |----------|----------|
# | **PROCEED** | FDR-significant OR (sign consistency > 60% AND abs(IC) > 0.003) |
# | **STOP** | Correctness fail (coverage < 70% OR staleness > 50%) |
# | **REVISE** | Everything else — evaluate in multivariate context in Ch11 |

# %% papermill={"duration": 0.008409, "end_time": "2026-05-15T14:11:26.891366+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.882957+00:00", "status": "completed"}
triage = {}
for feat in all_feature_cols:
    if not correctness[feat]:
        triage[feat] = ("STOP", "correctness_fail")
        continue

    if feat not in ic_results:
        triage[feat] = ("REVISE", "insufficient_data")
        continue

    is_fdr_sig = feat in fdr_sig_features
    sign_con = fold_stats.get(feat, {}).get("sign_consistency", 0)
    abs_ic = abs(ic_results[feat]["mean_ic"])

    if is_fdr_sig:
        triage[feat] = ("PROCEED", "fdr_significant")
    elif sign_con >= 0.60 and abs_ic >= IC_THRESHOLD:
        triage[feat] = ("PROCEED", "stable_and_above_threshold")
    else:
        triage[feat] = ("REVISE", "not_significant_standalone")

# %% papermill={"duration": 0.017541, "end_time": "2026-05-15T14:11:26.914253+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.896712+00:00", "status": "completed"}
# Build triage ledger
ledger_rows = []
for feat in all_feature_cols:
    decision, note = triage[feat]
    row = {
        "feature": feat,
        "family": families.get(feat, "other"),
        "source": "temporal" if feat in temporal_cols else "financial",
        "ic_mean": ic_results.get(feat, {}).get("mean_ic"),
        "hac_t": ic_results.get(feat, {}).get("t_stat"),
        "hac_p": ic_results.get(feat, {}).get("p_value"),
        "fdr_p": None,
        "fdr_sig": False,
        "sign_consistency": fold_stats.get(feat, {}).get("sign_consistency"),
        "worst_fold_ic": fold_stats.get(feat, {}).get("worst_fold_ic"),
        "monotonicity": monotonicity_scores.get(feat),
        "coverage": coverage[feat],
        "staleness": staleness[feat],
        "decision": decision,
        "note": note,
    }
    match = eval_summary.filter(pl.col("feature") == feat)
    if len(match) > 0:
        row["fdr_p"] = float(match["fdr_p"][0])
        row["fdr_sig"] = bool(match["fdr_sig"][0])
    ledger_rows.append(row)

triage_ledger = pl.DataFrame(ledger_rows)

# Save triage ledger
triage_ledger.write_parquet(EVAL_WRITE_DIR / "triage_ledger.parquet")
print(f"\nTriage ledger: {EVAL_DIR / 'triage_ledger.parquet'}")
print(triage_ledger.group_by("decision").len().sort("decision"))

# %% papermill={"duration": 0.022924, "end_time": "2026-05-15T14:11:26.942724+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.919800+00:00", "status": "completed"}
# Save IC time series (long format)
ic_ts_frames = []
for feat, ts in ic_timeseries.items():
    ic_ts_frames.append(ts.with_columns(pl.lit(feat).alias("feature")))

if ic_ts_frames:
    ic_ts_all = pl.concat(ic_ts_frames)
    ic_ts_all.write_parquet(EVAL_WRITE_DIR / "ic_timeseries.parquet")
    print(f"IC time series: {EVAL_DIR / 'ic_timeseries.parquet'}")

# %% papermill={"duration": 0.020548, "end_time": "2026-05-15T14:11:26.969044+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.948496+00:00", "status": "completed"}
# Write results JSON
proceed_features = [f for f, (d, _) in triage.items() if d == "PROCEED"]
revise_features = [f for f, (d, _) in triage.items() if d == "REVISE"]
stop_features = [f for f, (d, _) in triage.items() if d == "STOP"]

sorted_by_ic = sorted(ic_results.items(), key=lambda x: x[1].get("mean_ic") or 0, reverse=True)
best = sorted_by_ic[0] if sorted_by_ic else (None, {})
worst = sorted_by_ic[-1] if sorted_by_ic else (None, {})

# %% papermill={"duration": 0.008608, "end_time": "2026-05-15T14:11:26.983210+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.974602+00:00", "status": "completed"}
print(f"\n{'=' * 60}")
print(f"TRIAGE SUMMARY: {CASE_STUDY_ID}")
print(f"{'=' * 60}")
print(f"  PROCEED: {len(proceed_features)} features")
print(f"  REVISE:  {len(revise_features)} features")
print(f"  STOP:    {len(stop_features)} features")
print("\nPROMOTED (PROCEED) features:")
for f in sorted(proceed_features):
    ic = ic_results[f]["mean_ic"]
    t = ic_results[f]["t_stat"]
    print(f"  {f:40s}  IC={ic:+.4f}  t={t:.2f}  [{families.get(f, '?')}]")

# %% [markdown] papermill={"duration": 0.005528, "end_time": "2026-05-15T14:11:26.994242+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.988714+00:00", "status": "completed"}
# ### Quality Gate Verdict
#
# **Fit for modeling.** The triage promotes features to PROCEED across multiple
# microstructure families -- quote liquidity, order flow, and volatility/impact.
# The 15-minute return horizon is short enough that microstructure features retain
# predictive content, and the 114-stock NASDAQ-100 universe provides adequate
# cross-sectional breadth. Calendar and multi-resolution features complement the
# core LOB signals, though individual IC magnitudes are modest given the
# high-frequency, low signal-to-noise regime.

# %% [markdown] papermill={"duration": 0.006014, "end_time": "2026-05-15T14:11:27.005741+00:00", "exception": false, "start_time": "2026-05-15T14:11:26.999727+00:00", "status": "completed"}
# ## Key Takeaways
#
# ### Consolidated Feature Assessment
#
# 1. **Joint evaluation**: 88 features (66 financial + 22 temporal) evaluated against
#    15-minute midprice returns using cross-sectional Spearman IC with HAC adjustment
# 2. **FDR correction** controls false discoveries when testing 88 features simultaneously
# 3. **Correctness screens** catch coverage gaps and stale features before they
#    contaminate IC results
#
# ### Triage Logic
#
# - **PROCEED**: Features with statistical evidence of predictive content (FDR-significant
#   or sign-consistent with IC above threshold). Passed to Ch11 for model training
# - **REVISE**: Marginal features that may contribute in multivariate context.
#   Ch11 evaluates them via regularization / importance
# - **STOP**: Features failing basic quality checks (low coverage, high staleness)
#
# **Next**: Ch11 uses `evaluation/triage_ledger.parquet` and `evaluation/ic_timeseries.parquet`
# to select features for model training.
