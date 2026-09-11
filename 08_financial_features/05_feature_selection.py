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

# %% [markdown] tags=[]
# # Feature Selection and Deduplication
#
# **Chapter 8: Feature Engineering**
# **Section Reference**: 8.6, Combining Features and Controlling Search
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# A feature engineering pipeline produces many candidates: different lookbacks,
# transforms, and interaction variants. This notebook demonstrates how to reduce
# that set to a focused, production-ready collection using systematic selection
# and deduplication.
#
# ## Learning Objectives
#
# 1. Compute cross-sectional IC and rank features by predictive power
# 2. Apply correlation filtering to remove redundant features
# 3. Cluster near-duplicate features and select representatives
# 4. Use Benjamini–Hochberg FDR to control false discovery across multiple tests
# 5. Assess feature stability via bootstrap IC
# 6. Compare IC-based and ML-based (LightGBM) importance rankings
#
# ## Prerequisites
#
# - Run [`03_financial_features`](../case_studies/etfs/03_financial_features.ipynb)
#   to produce `financial.parquet`
# - Requires `ml4t-diagnostic` and `ml4t-engineer` libraries
#
# ## References
#
# - Harvey, Liu, and Zhu (2016), on multiple testing in factor research
# - Meinshausen and Bühlmann (2010), on stability selection
#
# **Output**: Selected feature list for downstream Chapter 9 use

# %% [markdown] tags=[]
# ## Setup

# %% tags=[]
"""Feature Selection and Deduplication: reduce feature candidates to a focused production set."""

import warnings
from datetime import date

# Imported here, before scikit-learn, so LightGBM's OpenMP runtime loads first; the
# ML-importance step far below defers its own import and that is too late to settle it.
import lightgbm  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.api as sm
import yaml
from ml4t.diagnostic.metrics import pooled_ic
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

# Permutation importance predicts with a bare array while LightGBM always records feature
# names, so each of its predictions warns. Named by message; other sklearn warnings stay.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)

from data import load_etfs
from utils.paths import get_case_study_dir, get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_with_alt

# %% tags=["parameters"]
START_DATE = "2006-01-01"
N_BOOTSTRAP = 50
MAX_SYMBOLS = 0
SEED = 42
# Thresholds the steps below apply. Declared here so the figures, the printed tables and
# the prose all read the same number.
CORR_THRESHOLD = 0.9  # |r| above which two features count as redundant
IC_THRESHOLD = 0.01  # |IC| below which a feature is treated as having no edge
FDR_ALPHA = 0.05  # Benjamini-Hochberg false discovery rate
BOOTSTRAP_SAMPLE_FRAC = 0.8  # rows drawn per bootstrap sample, with replacement
STABILITY_MIN_POSITIVE_PCT = 80.0  # share of bootstrap samples whose IC must be positive

# %% tags=[]
set_global_seeds(SEED)

# %% [markdown] tags=[]
# ## Load Features from ETF Case Study
#
# The ETF case study produced features in `case_studies/etfs/features/`.

# %% tags=[]
CASE_DIR = get_case_study_dir("etfs")
FEATURES_PATH = CASE_DIR / "features" / "financial.parquet"

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Features file not found at {FEATURES_PATH}. "
        "Please run case_studies/etfs/03_financial_features.py first."
    )

features_df = pl.read_parquet(FEATURES_PATH)
prices_df = load_etfs()

# %% [markdown] tags=[]
# Feature selection is a development decision, so it must not see the holdout. The
# boundary comes from the case study's own `setup.yaml` under `evaluation.holdout_start`,
# and the rule it follows is set out in `06_strategy_definition/02_cv_foundations`.
# Everything below reads pre-holdout rows only: the IC ranking, the multiple-testing
# correction, the stability selection and the model importances alike. The forward-return
# labels are computed from the already-filtered prices, so no label reaches across the
# boundary either.

# %% tags=[]
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
HOLDOUT_START = date.fromisoformat(setup["evaluation"]["holdout_start"])

# Apply date filters: development window only ([START_DATE, HOLDOUT_START))
features_df = features_df.filter(
    (pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    & (pl.col("timestamp") < HOLDOUT_START)
)
prices_df = prices_df.filter(
    (pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    & (pl.col("timestamp") < HOLDOUT_START)
)

if MAX_SYMBOLS > 0:
    top_symbols = (
        features_df.group_by("symbol")
        .len()
        .sort("len", descending=True)
        .head(MAX_SYMBOLS)["symbol"]
    )
    features_df = features_df.filter(pl.col("symbol").is_in(top_symbols))
    prices_df = prices_df.filter(pl.col("symbol").is_in(top_symbols))

# Compute forward returns on-demand
labels_df = (
    prices_df.sort(["symbol", "timestamp"])
    .with_columns(
        (pl.col("close").shift(-21).over("symbol") / pl.col("close") - 1).alias("fwd_return_1m")
    )
    .select(["timestamp", "symbol", "fwd_return_1m"])
    .drop_nulls()
)

print(f"Features: {features_df.shape}")
print(f"Labels: {labels_df.shape}")
print(f"Development window: {START_DATE} to {HOLDOUT_START}; the holdout is not read here")

# %% tags=[]
all_feature_cols = [c for c in features_df.columns if c not in ["timestamp", "symbol"]]

# %% [markdown] tags=[]
# Non-finite feature values, such as the 0/0 a short-window Sharpe ratio can produce, are
# replaced with nulls. Left as NaN they pass straight through `drop_nulls`, which removes
# only nulls, and then propagate through `pl.corr` into the panel correlation matrix,
# which both corrupts those features' correlations and misgroups them in the clustering.

# %% tags=[]
features_df = features_df.with_columns(
    [
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in all_feature_cols
    ]
)

print(f"Available features: {len(all_feature_cols)}")
for i, col in enumerate(all_feature_cols, 1):
    print(f"  {i:2d}. {col}")

# %% [markdown] tags=[]
# ## Compute Information Coefficient (IC)
#
# IC measures the Spearman rank correlation between features and forward returns.
# We compute IC **cross-sectionally** (per date, then average). Pooled IC
# conflates time-series drift with cross-sectional predictive power.

# %% tags=[]
# Merge features with forward returns
analysis = features_df.join(
    labels_df.select(["timestamp", "symbol", "fwd_return_1m"]),
    on=["timestamp", "symbol"],
    how="inner",
).drop_nulls(subset=["fwd_return_1m"])

print(f"Analysis dataset: {analysis.shape}")

# %% tags=[]
# %% [markdown] tags=[]
# Cross-sectional IC is computed per date, then sorted by timestamp. The sort matters:
# `group_by` does not preserve order, and the Newey-West t-statistic below regresses each
# feature's daily IC series on a constant with an autocovariance correction, which means
# something only on a chronologically ordered series.

# %% tags=[]
ic_by_date = (
    analysis.group_by("timestamp")
    .agg([pl.corr(col, "fwd_return_1m", method="spearman").alias(col) for col in all_feature_cols])
    .sort("timestamp")
)

# %% [markdown] tags=[]
# The daily IC series is serially correlated, through overlapping information sets and
# slow-moving common factors, so the cell below reports both the i.i.d. t-statistic and a
# Newey-West one from regressing the IC series on a constant. The Newey-West figure is the
# one the multiple-testing correction consumes.
#
# `pl.corr` returns a float NaN rather than a null on any date where a feature is constant
# across symbols, so each daily IC series is filtered on finiteness rather than on nulls. A
# feature whose cross-sectional IC is undefined on most dates, or whose defined ICs have no
# variance, is a date-level series carrying no cross-sectional signal; it is dropped from
# the ranking and from every step that follows.

# %% tags=[]
NW_MAXLAGS = 12
MIN_IC_OBS = 20
MIN_DEFINED_FRAC = 0.5


def finite_daily_ics(col: str) -> np.ndarray | None:
    """Finite daily cross-sectional ICs for a feature, or ``None`` when it has
    no usable cross-sectional variation."""
    ics = ic_by_date[col].to_numpy()
    ics = ics[np.isfinite(ics)]
    if len(ics) < MIN_IC_OBS or len(ics) / ic_by_date.height < MIN_DEFINED_FRAC:
        return None
    if np.std(ics, ddof=1) == 0:
        return None
    return ics


ic_results = {}
excluded_features = []
for col in all_feature_cols:
    daily_ics = finite_daily_ics(col)
    if daily_ics is None:
        excluded_features.append(col)
        continue

    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics, ddof=1)
    t_stat_iid = mean_ic / (std_ic / np.sqrt(len(daily_ics)))
    nw = sm.OLS(daily_ics, np.ones(len(daily_ics))).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_MAXLAGS}
    )
    ic_results[col] = {
        "ic": mean_ic,
        "ic_std": std_ic,
        "t_stat_iid": t_stat_iid,
        "t_stat_NW": float(nw.tvalues[0]),
        "n": len(daily_ics),
    }

ic_df = (
    pl.DataFrame(
        [
            {
                "feature": k,
                "ic": v["ic"],
                "t_stat_iid": v.get("t_stat_iid"),
                "t_stat_NW": v.get("t_stat_NW"),
                "n_obs": v["n"],
            }
            for k, v in ic_results.items()
        ]
    )
    .with_columns(pl.col("ic").abs().alias("ic_abs"))
    .sort("ic_abs", descending=True)
)

if excluded_features:
    print(
        f"Excluded {len(excluded_features)} features with no cross-sectional "
        f"variation (date-level series): {excluded_features}"
    )
print(f"\nFeature IC Rankings (top 15), Newey-West with {NW_MAXLAGS} lags:")
ic_df.head(15)

# %% tags=[]
# IC bar chart
fig, ax = plt.subplots(figsize=(10, 8))
ic_pd = ic_df.to_pandas().sort_values("ic_abs", ascending=True)
colors = [COLORS["positive"] if ic > 0 else COLORS["negative"] for ic in ic_pd["ic"]]
ax.barh(ic_pd["feature"], ic_pd["ic"], color=colors)
ax.axvline(0, color="black", linewidth=0.5)
# Reference line at the |IC| threshold used for the final selection in §6, so
# the ranking chart and the selection step agree (features kept in §6 sit at or
# beyond this line).
ax.axvline(
    IC_THRESHOLD, color="orange", linestyle="--", alpha=0.7, label=f"IC threshold ({IC_THRESHOLD})"
)
ax.axvline(-IC_THRESHOLD, color="orange", linestyle="--", alpha=0.7)
ax.set_xlabel("Information Coefficient (Spearman)")
ax.set_title("Feature IC Ranking")
ax.legend()
show_with_alt(
    fig,
    (
        "A horizontal bar chart ranking the candidate features by their cross-sectional "
        "information coefficient, sorted from the largest positive at the top to the "
        "most negative near the bottom, with feature names down the left edge. Bars "
        "extending right from a solid zero line are green and those extending left are "
        "red. Two dashed orange vertical lines mark the IC threshold either side of "
        "zero, and a legend names them. The longest green bar belongs to the distance "
        "from the fifty-two week low, followed by a normalised true range and a group "
        "of volatility measures. The longest red bars belong to two momentum "
        "acceleration features. Most bars in the lower half fall inside the dashed "
        "lines, meaning the majority of candidates carry an IC smaller than the "
        "threshold in absolute terms."
    ),
)

# %% [markdown] tags=[]
# ## Correlation Filtering
#
# Highly correlated features provide overlapping information. We compute
# correlation on the full panel (all dates by symbols), then remove features whose
# absolute correlation exceeds `CORR_THRESHOLD`, keeping the one with the higher IC in
# each redundant pair. The threshold is declared in the parameters cell, and the printed
# header repeats whatever it is set to.

# %% tags=[]
feature_matrix = features_df.select(all_feature_cols).drop_nulls()
corr_np = feature_matrix.corr().to_numpy()

print(f"Correlation matrix: {corr_np.shape[0]} × {corr_np.shape[1]} features")


# %% [markdown] tags=[]
# ### Remove Redundant Features
# Greedily drop the weaker member of each highly correlated pair.


# %% tags=[]
def filter_correlated_features(
    corr_matrix: np.ndarray,
    feature_names: list[str],
    ic_scores: dict[str, float] | None = None,
    threshold: float = CORR_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Remove highly correlated features, keeping the one with higher IC."""
    removed = set()
    n = len(feature_names)

    for i in range(n):
        if feature_names[i] in removed:
            continue
        for j in range(i + 1, n):
            if feature_names[j] in removed:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                if ic_scores:
                    ic_i = abs(ic_scores.get(feature_names[i], 0))
                    ic_j = abs(ic_scores.get(feature_names[j], 0))
                    to_remove = feature_names[j] if ic_i >= ic_j else feature_names[i]
                else:
                    to_remove = feature_names[j]
                removed.add(to_remove)

    kept = [f for f in feature_names if f not in removed]
    return kept, list(removed)


# %% tags=[]
ic_scores = {row["feature"]: row["ic"] for row in ic_df.to_dicts()}

kept_after_corr, removed_by_corr = filter_correlated_features(
    corr_matrix=corr_np,
    feature_names=all_feature_cols,
    ic_scores=ic_scores,
    threshold=CORR_THRESHOLD,
)

print(f"Correlation Filtering (threshold={CORR_THRESHOLD}):")
print(f"  Before: {len(all_feature_cols)} features")
print(f"  After:  {len(kept_after_corr)} features")
print(f"  Removed: {removed_by_corr}")

# %% [markdown] tags=[]
# ## Clustering and Deduplication
#
# Even after removing pairs above that threshold, many features remain near-duplicates.
# Hierarchical clustering groups similar features so we can pick one
# representative per cluster, which preserves diversity across families while
# removing redundancy within them.
#
# **Linkage choice**: We use **complete linkage** (not Ward) because Ward
# assumes Euclidean distance, which correlation-based distances do not satisfy.
# Complete linkage also avoids the chaining that average linkage produces on
# this panel, where many features share moderate correlations; it yields compact
# clusters whose members are mutually near-duplicate.

# %% [markdown] tags=[]
# Only features carrying a cross-sectional IC are clustered; the date-level series dropped
# above have no cross-sectional correlation structure to group on.

# %% tags=[]
cluster_features = [f for f in kept_after_corr if f in ic_scores]

# Build correlation matrix for the clustered features
surv_idx = [all_feature_cols.index(f) for f in cluster_features]
surv_corr = corr_np[np.ix_(surv_idx, surv_idx)]

# Distance = 1 - |ρ| (NaN correlations treated as uncorrelated → distance 1.0)
dist_matrix = 1 - np.abs(np.nan_to_num(surv_corr, nan=0.0))
np.fill_diagonal(dist_matrix, 0)
dist_matrix = (dist_matrix + dist_matrix.T) / 2
dist_matrix = np.clip(dist_matrix, 0, 2)

dist_condensed = squareform(dist_matrix, checks=False)
link = linkage(dist_condensed, method="complete")

# %% tags=[]
# Clustered heatmap
leaves = leaves_list(link)
reordered_names = [cluster_features[i] for i in leaves]
reordered_corr = surv_corr[np.ix_(leaves, leaves)]

fig, ax = plt.subplots(figsize=(14, 12))
n_feats = len(reordered_names)
sns.heatmap(
    reordered_corr,
    annot=(n_feats <= 25),
    fmt=".2f",
    annot_kws={"size": 6},
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    xticklabels=reordered_names,
    yticklabels=reordered_names,
    cbar_kws={"label": "Correlation"},
)
ax.set_title("Feature Correlation (Clustered, Complete Linkage)")
ax.tick_params(axis="both", labelsize=8)
plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
show_with_alt(
    fig,
    (
        "A large square correlation heatmap of the surviving features, rows and columns "
        "in the same clustered order, with a dark red diagonal where each feature "
        "meets itself and a colour bar running from dark blue at minus one through "
        "white at zero to dark red at plus one. Several red blocks sit along the "
        "diagonal where groups of related features correlate strongly with one "
        "another: a block of longer-horizon Sharpe ratios and return ranks at the top "
        "left, a larger block of short-horizon returns and oscillators through the "
        "middle, and a smaller group of drawdown and volatility measures at the bottom "
        "right. Between the blocks the field is mostly pale, and a few features such "
        "as the Hurst exponent and the choppiness index sit in near-white rows and "
        "columns, correlating little with anything else."
    ),
)

# %% [markdown] tags=[]
# The block structure reveals which features are essentially measuring the
# same thing. Within each block, correlations are high, confirming that one
# representative per cluster captures the shared signal. Between blocks,
# correlations are lower, marking genuine diversification.

# %% tags=[]
# Assign clusters and select representatives by highest |IC|
N_CLUSTERS = 10
clusters = fcluster(link, N_CLUSTERS, criterion="maxclust")

print(f"\n=== Factor Clusters ({N_CLUSTERS} groups) ===\n")
representatives = []

for c in range(1, N_CLUSTERS + 1):
    cluster_factors = [cluster_features[i] for i, clust in enumerate(clusters) if clust == c]
    if not cluster_factors:
        continue
    best = max(cluster_factors, key=lambda f: abs(ic_scores[f]))
    representatives.append(best)

    print(f"Cluster {c}:")
    for f in cluster_factors:
        marker = "  →" if f == best else "   "
        print(f"  {marker} {f}: IC = {ic_scores[f]:.4f}")

print(f"\nRepresentatives: {representatives}")

# %% [markdown] tags=[]
# ## Multiple Testing Correction (BH-FDR)
#
# With many features tested, some appear significant by chance.
# Benjamini–Hochberg FDR controls the expected false discovery rate.
#
# **Inference**: the p-values fed into BH-FDR come from the **Newey-West HAC**
# t-statistic on each feature's daily IC series (matching the table above and
# the headline measure in `06_robustness_sensitivity.py`). The i.i.d. t-stat
# would overstate significance because daily ICs share slow-moving common
# factors and overlapping information sets.

# %% tags=[]
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr

ic_pvalues = []
ic_feature_names = []
for col in all_feature_cols:
    # Reuse the same finiteness/variance guard as the IC ranking so degenerate
    # series do not contribute NaN p-values, which would still inflate BH's
    # denominator and tighten the per-rank threshold for every valid feature.
    daily_ics = finite_daily_ics(col)
    if daily_ics is None:
        continue
    nw = sm.OLS(daily_ics, np.ones(len(daily_ics))).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_MAXLAGS}
    )
    p_val = float(nw.pvalues[0])
    if not np.isfinite(p_val):
        continue
    ic_pvalues.append(p_val)
    ic_feature_names.append(col)

if ic_pvalues:
    bh_result = benjamini_hochberg_fdr(ic_pvalues, alpha=FDR_ALPHA, return_details=True)

    n_significant_raw = sum(p < FDR_ALPHA for p in ic_pvalues)
    n_significant_fdr = sum(bh_result["rejected"])

    print(f"Features tested:                 {len(ic_pvalues)}")
    print(f"Significant at p<{FDR_ALPHA} (raw):     {n_significant_raw}")
    print(f"Significant after BH-FDR:        {n_significant_fdr}")
    print(f"False discoveries prevented:     {n_significant_raw - n_significant_fdr}")

    survivors = [ic_feature_names[i] for i, r in enumerate(bh_result["rejected"]) if r]
    if survivors:
        print("\nFeatures surviving FDR correction:")
        for f in survivors:
            print(f"  - {f}")

# %% [markdown] tags=[]
# ## Selection Pipeline
#
# Applying the steps in sequence: correlation filtering removes obvious
# redundancy, clustering reduces each near-duplicate family to a single
# representative, and an IC threshold keeps the representatives with predictive
# power.

# %% tags=[]
# IC filtering applied to the cluster representatives from §4
kept_after_ic = [f for f in representatives if abs(ic_scores[f]) >= IC_THRESHOLD]

print(f"IC Filtering of representatives (|IC| >= {IC_THRESHOLD}):")
print(f"  Representatives: {len(representatives)} features")
print(f"  After IC filter: {len(kept_after_ic)} features")

# %% tags=[]
# Rank the surviving representatives by |IC| (top-K cap)
TOP_K = 10
final_features = sorted(kept_after_ic, key=lambda f: abs(ic_scores[f]), reverse=True)[:TOP_K]

print(f"\nSelected Features ({len(final_features)}):")
for i, f in enumerate(final_features, 1):
    print(f"  {i:2d}. {f} (IC={ic_scores[f]:.4f})")

# %% [markdown] tags=[]
# ## Stability Selection via Bootstrap IC
#
# Stability selection asks whether a feature's IC keeps its sign under resampling, or
# rests on a few periods. Each bootstrap sample draws rows with replacement and recomputes the
# pooled IC, and the table reports, per feature, the mean IC across samples, its standard
# deviation, their ratio as an information ratio, and the share of samples in which the IC
# came out positive.
#
# That last column is the one with a rule attached. `STABILITY_MIN_POSITIVE_PCT` is
# declared in the parameters cell and applied below, so the notebook prints which features
# clear it rather than describing a cut it never makes. Sign consistency is a weaker claim
# than "ranks highly", and it is the claim this resampling supports.
#
# > **Caveat**: The bootstrap below samples individual rows (date × symbol),
# > pooling across dates. A more rigorous approach bootstraps by *date*
# > (block bootstrap), preserving cross-sectional structure. The pooled
# > version here is a quick filter; production systems should use
# > time-aware resampling.


# %% tags=[]
def bootstrap_ic(
    df: pl.DataFrame,
    feature_cols: list[str],
    return_col: str = "fwd_return_1m",
    n_bootstrap: int = 50,
    sample_frac: float = BOOTSTRAP_SAMPLE_FRAC,
) -> pl.DataFrame:
    """Compute IC across bootstrap samples to assess stability.

    Uses the global numpy seed set in the preamble via ``set_global_seeds(SEED)``.
    """
    n_samples = len(df)
    sample_size = int(n_samples * sample_frac)

    results = {f: [] for f in feature_cols}

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        sample = df[indices.tolist()]
        y = sample[return_col].to_numpy()

        for col in feature_cols:
            x = sample[col].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 30:
                results[col].append(np.nan)
                continue
            ic = pooled_ic(x[mask], y[mask])
            results[col].append(ic)

    stability_data = []
    for col in feature_cols:
        ics = np.array(results[col])
        valid = ics[~np.isnan(ics)]
        if len(valid) == 0:
            continue
        stability_data.append(
            {
                "feature": col,
                "ic_mean": np.mean(valid),
                "ic_std": np.std(valid),
                "ic_ir": np.mean(valid) / (np.std(valid) + 1e-8),
                "positive_pct": np.mean(valid > 0) * 100,
            }
        )

    if not stability_data:
        return pl.DataFrame(
            {"feature": [], "ic_mean": [], "ic_std": [], "ic_ir": [], "positive_pct": []}
        )
    return pl.DataFrame(stability_data).sort("ic_ir", descending=True)


# %% tags=[]
stability = bootstrap_ic(df=analysis, feature_cols=final_features, n_bootstrap=N_BOOTSTRAP)
print(f"Stability Selection ({N_BOOTSTRAP} bootstrap samples):")
print(stability)

# Apply the declared cut rather than leaving it in the prose.
stable_features = stability.filter(pl.col("positive_pct") >= STABILITY_MIN_POSITIVE_PCT)
print()
print(
    f"features whose IC was positive in at least {STABILITY_MIN_POSITIVE_PCT:.0f}% of "
    f"samples: {len(stable_features)} of {len(stability)}"
)
for _row in stable_features.iter_rows(named=True):
    print(f"  {_row['feature']:<32} positive in {_row['positive_pct']:5.1f}% of samples")

# %% tags=[]
fig, ax = plt.subplots(figsize=(10, 6))
stab_pd = stability.to_pandas()
ax.errorbar(
    stab_pd["feature"],
    stab_pd["ic_mean"],
    yerr=stab_pd["ic_std"],
    fmt="o",
    capsize=5,
    capthick=2,
    markersize=8,
)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Feature")
ax.set_ylabel("Mean IC ± Std")
ax.set_title("Feature IC Stability (Bootstrap)")
plt.xticks(rotation=45, ha="right")
show_with_alt(
    fig,
    (
        "An error-bar chart with the eight selected features along the horizontal axis, "
        "their names angled, and mean bootstrap information coefficient on the "
        "vertical axis against a solid line at zero. Each feature is a filled marker "
        "with a short vertical bar for one standard deviation across bootstrap "
        "samples; the bars are small enough that the ordering is unambiguous. The "
        "features are sorted left to right from the highest mean to the lowest. The "
        "leftmost four sit clearly above zero, led by the normalised true range, and "
        "the rightmost four sit below it, ending with a Bollinger percent-b measure."
    ),
)

# %% [markdown] tags=[]
# ## ML-Based Feature Importance
#
# Beyond IC ranking, ML models identify features with non-linear predictive
# power. We fit a quick LightGBM model and compare its feature importance
# with the IC rankings above.

# %% tags=[]
from ml4t.diagnostic.metrics import analyze_ml_importance

ml_data = analysis.select(["timestamp", "symbol"] + final_features + ["fwd_return_1m"]).drop_nulls()
# Fit on a named frame rather than a bare array. LightGBM then records the real feature
# names, permutation importance re-predicts with the same names, and the importances come
# back labelled by feature instead of by column position.
X = ml_data.select(final_features).to_pandas()
y = ml_data["fwd_return_1m"].to_numpy()

if len(X) > 100:
    from lightgbm import LGBMRegressor

    lgbm = LGBMRegressor(n_estimators=100, max_depth=5, verbose=-1, random_state=SEED)
    lgbm.fit(X, y)

    importance_result = analyze_ml_importance(
        model=lgbm,
        X=X,
        y=y,
        feature_names=final_features,
        methods=["mdi", "pfi"],
    )

    print("=== ML Feature Importance (LightGBM) ===\n")
    print(f"Consensus top features: {importance_result['consensus_ranking'][:10]}")
    print(f"Methods run: {importance_result['methods_run']}")
    if importance_result.get("method_agreement"):
        print(f"Method agreement: {importance_result['method_agreement']}")
    print(f"\n{importance_result['interpretation']}")

# %% [markdown] tags=[]
# **Interpretation**: MDI (Mean Decrease in Impurity) measures how much each
# feature reduces prediction error in the tree ensemble. PFI (Permutation
# Feature Importance) measures how much shuffling a feature degrades
# predictions. Features ranking high in both IC and ML importance are the
# strongest candidates for production.

# %% [markdown] tags=[]
# ## Post-Selection Verification

# %% [markdown] tags=[]
# The selection is supposed to leave features that are not near-duplicates of each other.
# Whether it did is a number, not an assumption: the heatmap below shows every pairwise
# correlation among the selected set and the cell prints the largest of them. Read that
# against `CORR_THRESHOLD`, which is the only bar the filtering step actually enforced.
# A maximum well below the threshold means the clustering removed more redundancy than the
# pairwise filter alone would have; a maximum close to it means the surviving set still
# contains a pair the filter was content to keep.

# %% tags=[]
selected_matrix = features_df.select(final_features).drop_nulls()
corr_after = selected_matrix.corr().to_numpy()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_after, dtype=bool), k=1)
sns.heatmap(
    corr_after,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    xticklabels=final_features,
    yticklabels=final_features,
    cbar_kws={"label": "Correlation"},
)
ax.set_title("Selected features: residual correlation")
show_with_alt(
    fig,
    (
        "A lower-triangular correlation heatmap of the eight selected features, each "
        "cell annotated with its correlation to two decimal places and shaded from "
        "blue through white to red by a colour bar spanning minus one to plus one. "
        "The diagonal is dark red at one. Off the diagonal the shading is pale, with "
        "the strongest pair being the two skip-recent momentum features, followed by "
        "each of those against the distance from the fifty-two week low. The "
        "remaining pairs sit near white, and several are mildly negative, including "
        "the normalised true range against each momentum feature."
    ),
)

np.fill_diagonal(corr_after, 0)
max_corr = np.abs(corr_after).max()
_i, _j = np.unravel_index(np.abs(corr_after).argmax(), corr_after.shape)
print(f"Max remaining correlation: {max_corr:.3f} (threshold was {CORR_THRESHOLD})")
print(f"  between {final_features[_i]} and {final_features[_j]}")

# %% [markdown] tags=[]
# ## Selection Summary and Output

# %% tags=[]
print("=" * 60)
print("FEATURE SELECTION REPORT")
print("=" * 60)
print(f"\nInitial Features:           {len(all_feature_cols)}")
print(f"After Correlation Filter:   {len(kept_after_corr)}")
print(f"Cluster Representatives:    {len(representatives)}")
print(f"After IC Filter:            {len(kept_after_ic)}")
print(f"Final Selected:             {len(final_features)}")
print(f"Removal Rate:               {100 * (1 - len(final_features) / len(all_feature_cols)):.1f}%")
print("\n" + "-" * 60)
print("SELECTED FEATURES FOR CHAPTER 9")
print("-" * 60)

for i, f in enumerate(final_features, 1):
    ic = ic_scores[f]
    stab_row = stability.filter(pl.col("feature") == f)
    ic_ir = stab_row["ic_ir"][0] if len(stab_row) > 0 else np.nan
    print(f"{i:2d}. {f:30s} IC={ic:+.4f}  IC_IR={ic_ir:.2f}")

print("=" * 60)

# %% tags=[]
# Save selected features for Chapter 9
OUTPUT_DIR = get_output_dir(8, "feature_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

selected_df = pl.DataFrame(
    {"feature": final_features, "ic": [ic_scores[f] for f in final_features]}
)
selected_df.write_parquet(OUTPUT_DIR / "selected_features.parquet")

filtered_features = features_df.select(["timestamp", "symbol"] + final_features)
filtered_features.write_parquet(OUTPUT_DIR / "features_selected.parquet")

print(f"Saved selected features to {OUTPUT_DIR}")
print(f"  - selected_features.parquet: {len(final_features)} features")
print(f"  - features_selected.parquet: {filtered_features.shape}")

# %% [markdown] tags=[]
# ## Key Takeaways
#
# 1. **Cross-sectional IC** is the correct method for factor evaluation, because
#    pooled IC conflates time-series drift with predictive power
# 2. **Correlation filtering** at `CORR_THRESHOLD` removes obvious redundancy;
#    **clustering** catches subtler near-duplicates within feature families
# 3. **Use average or complete linkage** (not Ward) for correlation distances, because
#    Ward assumes Euclidean geometry
# 4. **BH-FDR with HAC-adjusted p-values** controls false discovery when
#    screening many candidates. The p-values fed into BH-FDR come from the
#    Newey-West t-statistic on each feature's daily IC series, not the
#    i.i.d. t-stat, because daily ICs are serially correlated. Without a
#    multiple-testing correction, a share of null features equal to the chosen level
#    appears significant at that level by chance alone, which is what FDR_ALPHA both
#    sets and corrects for
# 5. **Bootstrap stability** separates features whose IC keeps its sign under resampling
#    from those that depend on a few periods
# 6. Features ranking high in both IC and ML importance are the strongest
#    production candidates
#
# **Next**: `06_robustness_sensitivity`, on parameter sensitivity and
# regime-conditional analysis
