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
# **Section Reference**: 8.6 — Combining Features and Controlling Search
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# A feature engineering pipeline produces many candidates — different lookbacks,
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
# - Harvey, Liu, and Zhu (2016) — Multiple testing in factor research
# - Meinshausen and Bühlmann (2010) — Stability selection
#
# **Output**: Selected feature list for downstream Chapter 9 use

# %% [markdown] tags=[]
# ## Setup

# %% tags=[]
"""Feature Selection and Deduplication — reduce feature candidates to a focused production-ready set."""

import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.api as sm
import yaml
from ml4t.diagnostic.metrics import pooled_ic
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

from data import load_etfs
from utils.paths import get_case_study_dir, get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
START_DATE = "2006-01-01"
N_BOOTSTRAP = 50
MAX_SYMBOLS = 0
SEED = 42

# %% tags=[]
set_global_seeds(SEED)

# %% [markdown] tags=[]
# ## 1. Load Features from ETF Case Study
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

# Holdout boundary: feature selection is a development decision, and the
# sealed holdout (setup.yaml `evaluation.holdout_start`; see the rule in
# 06_strategy_definition/02_cv_foundations) must not inform it. Every step
# below — IC ranking, BH-FDR, stability selection, ML importance — sees only
# pre-holdout rows, and the forward-return labels computed from the filtered
# prices never span into the holdout.
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
print(f"Development window: {START_DATE} to {HOLDOUT_START} (holdout sealed)")

# %% tags=[]
all_feature_cols = [c for c in features_df.columns if c not in ["timestamp", "symbol"]]

# Replace non-finite feature values (e.g. 0/0 in short-window Sharpe ratios)
# with nulls. Left in place they survive drop_nulls (which removes only nulls)
# and propagate NaN through pl.corr and the panel correlation matrix, poisoning
# the affected features' correlations and misgrouping them in the clustering.
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
# ## 2. Compute Information Coefficient (IC)
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
# Compute cross-sectional IC per date
ic_by_date = analysis.group_by("timestamp").agg(
    [pl.corr(col, "fwd_return_1m", method="spearman").alias(col) for col in all_feature_cols]
)

# Summary statistics. The daily IC series is serially correlated (overlapping
# information sets, slow-moving common factors). We report both the i.i.d.
# t-stat and a Newey-West HAC t-stat from regressing the IC time series on a
# constant. HAC is the headline used for the BH-FDR step in §5.
#
# ``pl.corr`` returns a float NaN (not a null) on any date where a feature is
# constant across symbols, so we filter each daily IC series on finiteness
# rather than nulls. A feature whose cross-sectional IC is undefined on most
# dates, or whose defined ICs have zero variance, is a date-level series with no
# cross-sectional signal; we drop it from the ranking and every downstream step.
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
print(f"\nFeature IC Rankings (top 15) — Newey-West with {NW_MAXLAGS} lags:")
ic_df.head(15)

# %% tags=[]
# IC bar chart
fig, ax = plt.subplots(figsize=(10, 8))
ic_pd = ic_df.to_pandas().sort_values("ic_abs", ascending=True)
colors = [COLORS["positive"] if ic > 0 else COLORS["negative"] for ic in ic_pd["ic"]]
ax.barh(ic_pd["feature"], ic_pd["ic"], color=colors)
ax.axvline(0, color="black", linewidth=0.5)
ax.axvline(0.02, color="orange", linestyle="--", alpha=0.7, label="IC threshold (0.02)")
ax.axvline(-0.02, color="orange", linestyle="--", alpha=0.7)
ax.set_xlabel("Information Coefficient (Spearman)")
ax.set_title("Feature IC Ranking")
ax.legend()
plt.show()

# %% [markdown] tags=[]
# ## 3. Correlation Filtering
#
# Highly correlated features provide overlapping information. We compute
# correlation on the full panel (all dates × symbols), then remove features
# with |r| > 0.9 — keeping the one with higher IC in each redundant pair.

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
    threshold: float = 0.9,
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
    threshold=0.9,
)

print("Correlation Filtering (threshold=0.9):")
print(f"  Before: {len(all_feature_cols)} features")
print(f"  After:  {len(kept_after_corr)} features")
print(f"  Removed: {removed_by_corr}")

# %% [markdown] tags=[]
# ## 4. Clustering and Deduplication
#
# Even after removing pairs above 0.9, many features remain near-duplicates.
# Hierarchical clustering groups similar features so we can pick one
# representative per cluster — preserving diversity across families while
# removing redundancy within them.
#
# **Linkage choice**: We use **complete linkage** (not Ward) because Ward
# assumes Euclidean distance, which correlation-based distances do not satisfy.
# Complete linkage also avoids the chaining that average linkage produces on
# this panel, where many features share moderate correlations; it yields compact
# clusters whose members are mutually near-duplicate.

# %% tags=[]
# Cluster only features that carry a cross-sectional IC; the date-level series
# excluded from the ranking have no meaningful correlation structure to group.
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
plt.show()

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
# ## 5. Multiple Testing Correction (BH-FDR)
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
    bh_result = benjamini_hochberg_fdr(ic_pvalues, alpha=0.05, return_details=True)

    n_significant_raw = sum(p < 0.05 for p in ic_pvalues)
    n_significant_fdr = sum(bh_result["rejected"])

    print(f"Features tested:                 {len(ic_pvalues)}")
    print(f"Significant at p<0.05 (raw):     {n_significant_raw}")
    print(f"Significant after BH-FDR:        {n_significant_fdr}")
    print(f"False discoveries prevented:     {n_significant_raw - n_significant_fdr}")

    survivors = [ic_feature_names[i] for i, r in enumerate(bh_result["rejected"]) if r]
    if survivors:
        print("\nFeatures surviving FDR correction:")
        for f in survivors:
            print(f"  - {f}")

# %% [markdown] tags=[]
# ## 6. Selection Pipeline
#
# Applying the steps in sequence: correlation filtering removes obvious
# redundancy, clustering reduces each near-duplicate family to a single
# representative, and an IC threshold keeps the representatives with predictive
# power.

# %% tags=[]
# IC filtering applied to the cluster representatives from §4
IC_THRESHOLD = 0.01
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
# ## 7. Stability Selection via Bootstrap IC
#
# Stability selection tests whether features remain important across bootstrap
# samples. Features that rank highly in >80% of samples are considered stable.
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
    sample_frac: float = 0.8,
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
stability

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
plt.show()

# %% [markdown] tags=[]
# ## 8. ML-Based Feature Importance
#
# Beyond IC ranking, ML models identify features with non-linear predictive
# power. We fit a quick LightGBM model and compare its feature importance
# with the IC rankings above.

# %% tags=[]
from ml4t.diagnostic.metrics import analyze_ml_importance

ml_data = analysis.select(["timestamp", "symbol"] + final_features + ["fwd_return_1m"]).drop_nulls()
X = ml_data.select(final_features).to_numpy()
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
# ## 9. Post-Selection Verification

# %% tags=[]
# Verify low inter-correlation among selected features
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
ax.set_title("Selected Features — Residual Correlation")
plt.show()

np.fill_diagonal(corr_after, 0)
max_corr = np.abs(corr_after).max()
print(f"Max remaining correlation: {max_corr:.3f}")

# %% [markdown] tags=[]
# ## 10. Selection Summary and Output

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
# 1. **Cross-sectional IC** is the correct method for factor evaluation —
#    pooled IC conflates time-series drift with predictive power
# 2. **Correlation filtering** (|r| > 0.9) removes obvious redundancy;
#    **clustering** catches subtler near-duplicates within feature families
# 3. **Use average or complete linkage** (not Ward) for correlation distances —
#    Ward assumes Euclidean geometry
# 4. **BH-FDR with HAC-adjusted p-values** controls false discovery when
#    screening many candidates. The p-values fed into BH-FDR come from the
#    Newey-West t-statistic on each feature's daily IC series, not the
#    i.i.d. t-stat, because daily ICs are serially correlated. Without
#    multiple-testing correction, ~5% of null features appear significant
#    at the 5% level by chance alone
# 5. **Bootstrap stability** separates features with robust IC from those
#    that depend on a few outlier periods
# 6. Features ranking high in both IC and ML importance are the strongest
#    production candidates
#
# **Next**: `06_robustness_sensitivity` — parameter sensitivity and
# regime-conditional analysis
