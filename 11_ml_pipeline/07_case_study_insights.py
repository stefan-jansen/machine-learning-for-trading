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
# # Case Study Insights: Linear Models
#
# **Docker image**: `ml4t`
#
# **Purpose**: assemble the cross-case-study view of the linear family
# (OLS, Ridge, Lasso, ElasticNet). Per-case-study deep dives live in
# `case_studies/{cs}/13_model_analysis.py`; this notebook is the
# comparative view across all nine case studies.
#
# **Learning objectives**
#
# - For each case study, read the highest-IC complete linear configuration's
#   mean daily cross-sectional Spearman IC with HAC 95 % CI on the primary label
# - Compare regularization families (OLS, Ridge, Lasso, ElasticNet) at
#   each case study's primary label
# - Inspect per-fold IC distributions and the ICIR diagnostic for
#   stability
# - Extend the comparison across labels (horizon view) and across the
#   classification ↔ regression metric symmetry
# - Inspect coefficient sign consistency, Lasso sparsity, and the overlap
#   of selected features across case studies
#
# **Book reference**: Section 11.6 - Linear Models Across Nine Case Studies.
#
# **Prerequisites**: each case study's `06_linear.py` pipeline has populated
# `run_log/registry.db` for the linear family. Teaching notebooks NB01-NB06
# cover the underlying techniques.

# %% tags=[]
"""Case Study Insights: Linear - cross-case-study aggregation from the registry."""

import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ml4t.diagnostic dlopens cudart; load torch first so its bundled CUDA
# runtime wins. Same precedence pattern as case_studies/utils/model_analysis.py.
import torch  # noqa: F401
from IPython.display import Markdown, display
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import roc_auc_score

from case_studies.utils.analytics import (
    CASE_STUDY_IDS,
    PRIMARY_LABELS,
    SHORT_NAMES,
)
from case_studies.utils.insight_chapter import (
    plot_cross_cs_forest,
    plot_rolling_daily_ic,
)
from case_studies.utils.model_analysis import (
    load_predictions,
)
from utils.paths import get_case_study_dir
from utils.style import COLORS

# %% tags=["parameters"]
FAMILY = "linear"

# %% [markdown] tags=[]
# ## Complete-result guard
#
# A model result is eligible for comparison only when no fold has an undefined
# IC and its validation-day coverage equals the maximum for that label. The
# local loader excludes predictions with any null fold IC and rejects shorter,
# partially populated results before ranking.


# %% tags=[]
METRICS_QUERY = """
    SELECT t.training_hash, t.family, t.config_name, t.label,
           p.prediction_hash, p.checkpoint_value, p.checkpoint_kind,
           pm.ic_mean, pm.ic_std, pm.ic_mean_daily, pm.ic_se_hac,
           pm.ic_ci_lo, pm.ic_ci_hi, pm.ic_t_hac, pm.ic_p_hac,
           pm.ic_n_days, pm.ic_hac_lag,
           (SELECT COUNT(*) FROM fold_metrics fm
            WHERE fm.prediction_hash = p.prediction_hash AND fm.ic IS NULL) AS n_null_folds
    FROM training_runs t
    JOIN prediction_sets p ON p.training_hash = t.training_hash
    JOIN prediction_metrics pm ON pm.prediction_hash = p.prediction_hash
    WHERE t.family = ? AND p.split = 'validation'
"""
FINITE_METRIC_FIELDS = [
    "ic_mean_daily",
    "ic_se_hac",
    "ic_ci_lo",
    "ic_ci_hi",
    "ic_t_hac",
    "ic_p_hac",
    "ic_n_days",
    "ic_hac_lag",
]


# %% [markdown] tags=[]
# The loader preserves exact training and prediction identities so every later
# statistic and coefficient view can be traced to one registry-selected artifact.


# %% tags=[]
def load_complete_metrics(
    case_study: str,
    *,
    label: str | None = None,
) -> pl.DataFrame:
    """Load nondegenerate linear metrics with maximum coverage per label."""
    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    query = METRICS_QUERY
    params: list[str] = [FAMILY]
    if label is not None:
        query += " AND t.label = ?"
        params.append(label)
    with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(query, params).fetchall()]
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, infer_schema_length=None)
    return (
        df.filter(
            (pl.col("n_null_folds") == 0)
            & pl.all_horizontal(
                pl.col(column).is_not_null() & pl.col(column).is_finite()
                for column in FINITE_METRIC_FIELDS
            )
        )
        .with_columns(_max_n_days=pl.col("ic_n_days").max().over("label"))
        .filter(pl.col("ic_n_days") == pl.col("_max_n_days"))
        .drop("_max_n_days", "n_null_folds")
    )


# %% [markdown] tags=[]
# The primary-label collector applies the same completeness rule independently
# to every case study before selecting the highest mean daily IC.


# %% tags=[]
def collect_complete_rank1(case_studies: list[str]) -> pl.DataFrame:
    """Return the highest-IC complete linear result per primary label."""
    frames = []
    for cs in case_studies:
        label = PRIMARY_LABELS[cs]
        df = load_complete_metrics(cs, label=label)
        if df.is_empty():
            raise RuntimeError(f"{cs}: no complete linear result for {label}")
        best_ic = df["ic_mean_daily"].max()
        winner = df.filter(pl.col("ic_mean_daily") == best_ic)
        if winner.height != 1:
            raise RuntimeError(f"{cs}: primary linear rank one is ambiguous")
        frames.append(
            winner.with_columns(
                case_study=pl.lit(cs),
                short_name=pl.lit(SHORT_NAMES[cs]),
            )
        )
    if len(frames) != len(case_studies):
        raise RuntimeError("primary linear coverage is incomplete")
    return pl.concat(frames, how="diagonal_relaxed")


# %% [markdown] tags=[]
# HAC inference is rebuilt from the exact selected prediction artifact. Daily
# correlations are sorted before the Bartlett Newey-West calculation, avoiding
# the fold-order dependence in several stored registry intervals.


# %% tags=[]
def _daily_ic_values(row: dict, target_column: str | None = None) -> np.ndarray:
    """Load chronological daily IC values against an explicit artifact target."""
    parquet = (
        get_case_study_dir(row["case_study"])
        / "run_log"
        / "predictions"
        / row["prediction_hash"]
        / "predictions.parquet"
    )
    schema = pl.scan_parquet(parquet).collect_schema()
    score = "y_score" if "y_score" in schema else "prediction"
    actual = target_column or ("y_true" if "y_true" in schema else "actual")
    if actual not in schema:
        raise RuntimeError(
            f"{row['case_study']}: prediction artifact lacks target column {actual!r}"
        )
    daily = (
        pl.scan_parquet(parquet)
        .select("timestamp", score, actual)
        .group_by("timestamp")
        .agg(
            pl.corr(score, actual, method="spearman").alias("ic"),
            (pl.col(score).is_finite() & pl.col(actual).is_finite()).sum().alias("n_obs"),
        )
        .filter((pl.col("n_obs") >= 5) & pl.col("ic").is_finite())
        .sort("timestamp")
        .collect()
    )
    return daily["ic"].to_numpy()


# %% tags=[]
def _bartlett_hac(values: np.ndarray, lag: int, case_study: str) -> dict[str, float]:
    """Return mean and Bartlett Newey-West inference for ordered observations."""
    n_days = len(values)
    if n_days < 3:
        raise RuntimeError(f"{case_study}: fewer than three daily IC observations")
    mean_ic = float(values.mean())
    residuals = values - mean_ic
    lag = min(lag, n_days - 1)
    long_run_sum = float(residuals @ residuals)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        long_run_sum += 2.0 * weight * float(residuals[offset:] @ residuals[:-offset])
    variance = long_run_sum / n_days**2 * n_days / (n_days - 1)
    if variance < -1e-15:
        raise RuntimeError(f"{case_study}: negative HAC variance {variance}")
    variance = max(variance, 0.0)
    standard_error = float(np.sqrt(variance))
    if not np.isfinite(standard_error) or standard_error == 0.0:
        raise RuntimeError(f"{case_study}: degenerate HAC standard error")
    t_hac = mean_ic / standard_error
    critical = float(stats.t.ppf(0.975, n_days - 1))
    return {
        "ic_mean_daily": mean_ic,
        "ic_n_days": float(n_days),
        "ic_se_hac": standard_error,
        "ic_ci_lo": mean_ic - critical * standard_error,
        "ic_ci_hi": mean_ic + critical * standard_error,
        "ic_t_hac": t_hac,
        "ic_p_hac": float(2.0 * stats.t.sf(abs(t_hac), n_days - 1)),
        "ic_hac_lag": float(lag),
    }


# %% tags=[]
def chronological_hac(row: dict, target_column: str | None = None) -> dict[str, float]:
    """Recompute daily IC and Bartlett Newey-West inference in time order."""
    values = _daily_ic_values(row, target_column=target_column)
    return _bartlett_hac(values, int(row["ic_hac_lag"]), row["case_study"])


# %% [markdown] tags=[]
# Recomputed inference replaces only the selected rows, preserving their exact
# training and prediction identities for later coefficient and fold diagnostics.


# %% tags=[]
def replace_with_chronological_hac(
    selected: pl.DataFrame, target_column: str | None = None
) -> pl.DataFrame:
    """Bind selected rows to inference recomputed against an explicit target."""
    rows = []
    for row in selected.iter_rows(named=True):
        row.update(chronological_hac(row, target_column=target_column))
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


# %% [markdown] tags=[]
# Every grouped comparison fails closed when two artifacts tie for first place;
# arbitrary row order must never determine the reported winner.


# %% tags=[]
def select_unique_best(frame: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    """Select one unambiguous highest daily-IC row in every requested group."""
    winners = []
    for group in frame.partition_by(groups, maintain_order=True):
        best_ic = group["ic_mean_daily"].max()
        best = group.filter(pl.col("ic_mean_daily") == best_ic)
        if best.height != 1:
            identity = {column: group[0, column] for column in groups}
            raise RuntimeError(f"ambiguous daily-IC rank one for {identity}")
        winners.append(best)
    return pl.concat(winners, how="diagonal_relaxed") if winners else pl.DataFrame()


# %% [markdown] tags=[]
# Fold diagnostics use the exact complete configurations selected above, rather
# than independently ranking a second time.


# %% tags=[]
def collect_selected_fold_ic(selected: pl.DataFrame) -> pl.DataFrame:
    """Load fold metrics for the exact primary-label selections."""
    frames = []
    for row in selected.iter_rows(named=True):
        db_path = get_case_study_dir(row["case_study"]) / "run_log" / "registry.db"
        with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as connection:
            records = connection.execute(
                """
                SELECT fold_id, ic, ic_std, n_entities, rmse, mae
                FROM fold_metrics WHERE prediction_hash = ? ORDER BY fold_id
                """,
                (row["prediction_hash"],),
            ).fetchall()
        folds = pl.DataFrame(
            records,
            schema=["fold_id", "ic", "ic_std", "n_entities", "rmse", "mae"],
            orient="row",
        )
        if folds.is_empty() or folds["ic"].null_count() > 0:
            raise RuntimeError(f"{row['case_study']}: selected result has missing fold IC")
        frames.append(
            folds.with_columns(
                case_study=pl.lit(row["case_study"]),
                short_name=pl.lit(row["short_name"]),
            )
        )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


# %% [markdown] tags=[]
# Multi-label comparisons repeat the complete-coverage selection within every
# case-study and label pair.


# %% tags=[]
def collect_complete_multi_label(
    case_studies: list[str],
    label_resolver,
) -> pl.DataFrame:
    """Return the highest-IC complete result for each requested label."""
    frames = []
    for cs in case_studies:
        for label in label_resolver(cs):
            df = load_complete_metrics(cs, label=label)
            if df.is_empty():
                continue
            winner = select_unique_best(df, ["label"])
            frames.append(
                winner.with_columns(
                    case_study=pl.lit(cs),
                    short_name=pl.lit(SHORT_NAMES[cs]),
                )
            )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


# %% [markdown] tags=[]
# ## 1. Scope and Coverage
#
# The linear family is the baseline against which Ch12 (gradient boosting),
# Ch13 (deep learning for time series), and Ch14 (latent factors) are
# compared. Each case study's primary label is fixed in
# `PRIMARY_LABELS`; the primary metric is mean daily cross-sectional Spearman IC.
# Selection uses registry coverage and IC fields, while HAC inference is recomputed
# from each selected prediction artifact after sorting daily IC chronologically.

# %% tags=[]
coverage_rows = []
for cs in CASE_STUDY_IDS:
    primary = PRIMARY_LABELS[cs]
    df = load_complete_metrics(cs)
    if df.is_empty():
        coverage_rows.append(
            {
                "case_study": SHORT_NAMES[cs],
                "primary_label": primary,
                "n_labels": 0,
                "n_complete_configs_primary": 0,
            }
        )
        continue
    n_labels = df["label"].n_unique()
    n_configs_primary = df.filter(pl.col("label") == primary)["config_name"].n_unique()
    coverage_rows.append(
        {
            "case_study": SHORT_NAMES[cs],
            "primary_label": primary,
            "n_labels": n_labels,
            "n_complete_configs_primary": n_configs_primary,
        }
    )

coverage_df = pl.DataFrame(coverage_rows)
print("Complete linear coverage per case study (primary label and labels trained):")
coverage_df

# %% [markdown] tags=[]
# ## 2. Cross-CS Forest of Highest-IC Linear Configurations
#
# For each case study, the linear configuration with the highest
# mean daily IC on the primary label is plotted with its HAC 95 % CI.
# Filled markers indicate $|t_{HAC}| > 2$ (CI excludes zero); open
# markers indicate the CI overlaps zero.

# %% tags=[]
rank1 = replace_with_chronological_hac(collect_complete_rank1(CASE_STUDY_IDS))
print(
    "Highest-IC complete linear configuration per case study "
    "(primary label, mean daily IC ± HAC 95 % CI):"
)
rank1.select(
    "short_name",
    "label",
    "config_name",
    pl.col("ic_mean_daily").round(4).alias("ic"),
    pl.col("ic_ci_lo").round(4).alias("ci_lo"),
    pl.col("ic_ci_hi").round(4).alias("ci_hi"),
    pl.col("ic_t_hac").round(2).alias("t_hac"),
    pl.col("ic_n_days").cast(pl.Int64).alias("n_days"),
)

# %% tags=[]
fig, ax = plot_cross_cs_forest(
    rank1,
    family=FAMILY,
    title="Most primary-label linear intervals overlap zero",
)
ax.set_xlabel("Mean daily IC (HAC 95 % CI)")
fig.show()

# %% tags=[]
significant = rank1.filter((pl.col("ic_ci_lo") > 0) | (pl.col("ic_ci_hi") < 0))
overlapping = rank1.filter(~((pl.col("ic_ci_lo") > 0) | (pl.col("ic_ci_hi") < 0)))
sig_names = ", ".join(significant.sort("short_name")["short_name"].to_list())
overlap_names = ", ".join(overlapping.sort("short_name")["short_name"].to_list())
display(
    Markdown(
        f"The HAC interval excludes zero for **{significant.height} of {rank1.height}** primary "
        f"labels ({sig_names}). It overlaps zero for {overlap_names}. The ETF result now selects "
        "the full-coverage Ridge fit; shorter or degenerate L1 fits are not eligible for this "
        "comparison. These are validation estimates read from the curated registries, not a new "
        "cross-case-study model selection exercise."
    )
)

# %% [markdown] tags=[]
# ## 3. Regularization Comparison
#
# How much does the choice of regularization family change the IC at
# the primary label? For each case study we take the highest-IC OLS,
# Ridge, Lasso, and ElasticNet configurations and compare their daily-
# mean daily IC with HAC 95 % CI, then trace the Ridge regularization path
# with HAC 95 % CI bands.


# %% tags=[]
def family_from_config(s: pl.Expr) -> pl.Expr:
    return (
        pl.when(s.str.starts_with("ols"))
        .then(pl.lit("ols"))
        .when(s.str.starts_with("ridge"))
        .then(pl.lit("ridge"))
        .when(s.str.starts_with("lasso"))
        .then(pl.lit("lasso"))
        .when(s.str.starts_with("enet"))
        .then(pl.lit("elastic_net"))
        .otherwise(pl.lit("other"))
    )


# %% tags=[]
reg_rows = []
for cs in CASE_STUDY_IDS:
    primary = PRIMARY_LABELS[cs]
    df = load_complete_metrics(cs, label=primary)
    if df.is_empty():
        continue
    df = df.filter(pl.col("ic_mean_daily").is_not_null())
    df = df.with_columns(
        reg_family=family_from_config(pl.col("config_name")),
        short_name=pl.lit(SHORT_NAMES[cs]),
        case_study=pl.lit(cs),
    )
    reg_rows.append(df)

reg_df = pl.concat(reg_rows, how="diagonal_relaxed") if reg_rows else pl.DataFrame()

# Rank-1 per (CS, regularization family)
reg_best = replace_with_chronological_hac(
    select_unique_best(reg_df, ["case_study", "reg_family"])
).select(
    "case_study",
    "short_name",
    "reg_family",
    "config_name",
    "prediction_hash",
    "ic_mean_daily",
    "ic_ci_lo",
    "ic_ci_hi",
    "ic_t_hac",
)

# Wide pivot for table
ic_pivot = reg_best.pivot(on="reg_family", index="short_name", values="ic_mean_daily").sort(
    "short_name"
)
print("Mean daily IC by regularization family (complete primary-label results):")
ic_pivot

# %% [markdown] tags=[]
# ### Figure: Regularization Family Comparison
#
# Missing families remain gaps rather than zeros in the grouped chart.


# %% tags=[]
def regularization_metric(
    frame: pl.DataFrame,
    case_studies: list[str],
    column: str,
) -> np.ndarray:
    """Align one regularization metric to the chart's case-study order."""
    lookup = dict(zip(frame["short_name"].to_list(), frame[column].to_list(), strict=True))
    return np.array([lookup.get(case_study, np.nan) for case_study in case_studies], dtype=float)


# %% tags=[]
families_present = ["ols", "ridge", "lasso", "elastic_net"]
plot_data = reg_best.filter(pl.col("reg_family").is_in(families_present)).sort(
    ["short_name", "reg_family"]
)

fig, ax = plt.subplots(figsize=(11, 5))
cs_order = sorted(plot_data["short_name"].unique().to_list())
x = np.arange(len(cs_order))
width = 0.2
fam_colors = {
    "ols": COLORS["neutral"],
    "ridge": COLORS["blue"],
    "lasso": COLORS["amber"],
    "elastic_net": COLORS["copper"],
}

for i, fam in enumerate(families_present):
    sub = plot_data.filter(pl.col("reg_family") == fam)
    ic = regularization_metric(sub, cs_order, "ic_mean_daily")
    lo = regularization_metric(sub, cs_order, "ic_ci_lo")
    hi = regularization_metric(sub, cs_order, "ic_ci_hi")
    err = np.vstack([ic - lo, hi - ic])
    ax.bar(
        x + (i - 1.5) * width,
        ic,
        width=width,
        yerr=err,
        capsize=2,
        color=fam_colors[fam],
        alpha=0.9,
        label=fam.replace("_", " ").title(),
    )

ax.set_xticks(x)
ax.set_xticklabels(cs_order, rotation=35, ha="right")
ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
ax.set_ylabel("Mean daily IC (HAC 95 % CI)")
ax.set_title("Regularization shifts estimates less than their uncertainty")
ax.legend(frameon=False, fontsize=8, loc="best")
if fig.get_layout_engine() is None:
    fig.tight_layout()
fig.show()

# %% tags=[]
ridge_ols = reg_best.filter(pl.col("reg_family").is_in(["ridge", "ols"]))
ridge_ols_wide = ridge_ols.pivot(
    on="reg_family", index="short_name", values="ic_mean_daily"
).drop_nulls(["ridge", "ols"])
ridge_wins = ridge_ols_wide.filter(pl.col("ridge") >= pl.col("ols")).height
l1_cases = reg_best.filter(pl.col("reg_family").is_in(["lasso", "elastic_net"]))[
    "short_name"
].n_unique()
display(
    Markdown(
        f"Ridge matches or exceeds OLS in **{ridge_wins} of {ridge_ols_wide.height}** comparable "
        f"case studies. Complete Lasso or ElasticNet results are available for {l1_cases} case "
        "studies. Most family-to-family point-estimate shifts remain smaller than the displayed "
        "uncertainty intervals, so regularization choice rarely changes the primary conclusion."
    )
)

# %% [markdown] tags=[]
# ### Ridge regularization path
#
# How does mean daily IC change as Ridge $\alpha$ increases? A flat
# path means the conditioning of the feature matrix is already benign;
# a pronounced peak indicates an optimal shrinkage strength.

# %% tags=[]
ridge_path = replace_with_chronological_hac(
    reg_df.filter(pl.col("reg_family") == "ridge")
    .with_columns(
        alpha=pl.col("config_name").str.extract(r"ridge_a([\d.e+]+)", 1).cast(pl.Float64),
    )
    .filter(pl.col("alpha").is_not_null())
    .sort(["case_study", "alpha"])
)

palette = [
    COLORS["blue"],
    COLORS["copper"],
    COLORS["amber"],
    COLORS["positive"],
    COLORS["negative"],
    COLORS["slate"],
    COLORS["amber_light"],
    COLORS["neutral"],
    COLORS["blue_light"],
]
ridge_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h"]
ridge_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]

# %% tags=[]
fig, ax = plt.subplots(figsize=(10, 5))
cs_list_sorted = sorted(ridge_path["case_study"].unique().to_list())
for i, cs in enumerate(cs_list_sorted):
    sub = ridge_path.filter(pl.col("case_study") == cs).sort("alpha")
    if sub.height < 2:
        continue
    a = sub["alpha"].to_numpy()
    ic = sub["ic_mean_daily"].to_numpy()
    lo = sub["ic_ci_lo"].to_numpy()
    hi = sub["ic_ci_hi"].to_numpy()
    color = palette[i % len(palette)]
    name = SHORT_NAMES.get(cs, cs)
    ax.fill_between(a, lo, hi, color=color, alpha=0.10)
    ax.plot(
        a,
        ic,
        marker=ridge_markers[i],
        linestyle=ridge_styles[i],
        color=color,
        label=name,
        linewidth=1.5,
        markersize=4,
        alpha=0.9,
    )

ax.set_xscale("log")
ax.set_xlabel(r"Ridge $\alpha$ (log scale)")
ax.set_ylabel("Mean daily IC (HAC 95 % CI band)")
ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
ax.set_title("Ridge paths are mostly flat relative to uncertainty")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
fig.show()

# %% [markdown] tags=[]
# The confidence bands put the path curvature in context. An apparent optimum
# matters only when its change is large relative to uncertainty; across these
# panels, most neighboring Ridge settings remain statistically difficult to
# distinguish.

# %% [markdown] tags=[]
# ## 4. Persistence and Stability
#
# A positive mean daily IC averaged over the whole validation period can
# still come from a few good stretches. The primary persistence view is a
# three-month rolling mean of the daily IC, which traces how ranking quality
# evolves over time. Because case studies cover different validation windows,
# the comparison is clipped to the common validation period for ETFs and FX.
# (Figure 11.8 in the chapter.) Per-fold IC and ICIR below are secondary stability
# diagnostics.

# %% tags=[]
fig, _ = plot_rolling_daily_ic(
    ["etfs", "fx_pairs"],
    window=63,
    common_window=True,
    selected_prediction_hashes={
        row["case_study"]: row["prediction_hash"]
        for row in rank1.filter(pl.col("case_study").is_in(["etfs", "fx_pairs"])).iter_rows(
            named=True
        )
    },
    title="ETF and FX rolling IC vary through their shared window",
)
fig.show()

# %% [markdown] tags=[]
# The rolling view tests whether a full-period average is broadly persistent or
# concentrated in a few stretches. Both paths cross zero here, so neither
# full-period average describes every validation regime. Neither path is a
# sealed holdout result.

# %% tags=[]
fold_df = collect_selected_fold_ic(rank1)
fold_summary = (
    fold_df.group_by(["case_study", "short_name"])
    .agg(
        n_folds=pl.col("ic").count(),
        mean=pl.col("ic").mean(),
        median=pl.col("ic").median(),
        std=pl.col("ic").std(),
        pct_positive=(pl.col("ic") > 0).mean(),
    )
    .sort("median", descending=True)
)
print("Per-fold IC summary for the highest-IC linear config (primary label):")
fold_summary

# %% [markdown] tags=[]
# ### Figure: Per-Fold IC Distribution

# %% tags=[]
order = rank1.sort("ic_mean_daily", descending=True)["short_name"].to_list()
present = [c for c in order if c in fold_df["short_name"].unique().to_list()]

fig, ax = plt.subplots(figsize=(11, 4.5))
data = [fold_df.filter(pl.col("short_name") == cs)["ic"].to_numpy() for cs in present]
positions = np.arange(len(present))
ax.boxplot(data, positions=positions, widths=0.55, showfliers=True)
for i, arr in enumerate(data):
    if len(arr):
        ax.scatter(np.full(len(arr), i), arr, alpha=0.5, s=14, color=COLORS["blue"])
ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
ax.set_xticks(positions)
ax.set_xticklabels(present, rotation=30, ha="right")
ax.set_ylabel("Per-fold Spearman IC")
ax.set_title("Selected linear fits vary widely across validation folds")
if fig.get_layout_engine() is None:
    fig.tight_layout()
fig.show()

# %% tags=[]
most_positive = fold_summary.sort("pct_positive", descending=True).row(0, named=True)
least_variable = fold_summary.sort("std").row(0, named=True)
display(
    Markdown(
        f"**{most_positive['short_name']}** has the largest positive-fold share "
        f"({most_positive['pct_positive']:.0%}), while **{least_variable['short_name']}** has the "
        f"smallest fold-level standard deviation ({least_variable['std']:.4f}). Fold counts range "
        f"from {fold_summary['n_folds'].min()} to {fold_summary['n_folds'].max()}, so the boxplots "
        "are stability diagnostics rather than equal-precision estimates."
    )
)

# %% [markdown] tags=[]
# ### ICIR cross-CS bar
#
# The information ratio $\text{ICIR} = |\overline{\text{IC}}| / \sigma(\text{IC})$
# is a stability-adjusted signal-strength summary. With only a handful of folds per
# case study it is sample-starved and should be read alongside the
# mean daily IC and per-fold distribution, not as a standalone score.

# %% tags=[]
icir = fold_summary.with_columns(icir=(pl.col("mean").abs() / pl.col("std")).round(3)).sort(
    "icir", descending=True, nulls_last=True
)

fig, ax = plt.subplots(figsize=(9, 3.6))
y = np.arange(icir.height)
ax.barh(y, icir["icir"].to_numpy(), color=COLORS["blue"], alpha=0.9, height=0.55)
ax.set_yticks(y)
ax.set_yticklabels(icir["short_name"].to_list())
ax.invert_yaxis()
ax.set_xlabel("ICIR (|mean fold IC| / fold standard deviation)")
ax.set_title("Fold stability differs sharply across the selected linear fits")
if fig.get_layout_engine() is None:
    fig.tight_layout()
fig.show()

# %% tags=[]
icir_leader = icir.row(0, named=True)
display(
    Markdown(
        f"**{icir_leader['short_name']}** has the largest fold ICIR "
        f"({icir_leader['icir']:.3f}). With only "
        f"{icir_leader['n_folds']} folds for that estimate, ICIR remains a descriptive "
        "stability diagnostic and does not replace the daily HAC interval."
    )
)

# %% [markdown] tags=[]
# ## 5. Multi-Label Horizon and Metric Symmetry

# %% [markdown] tags=[]
# ### 5a. Linear IC across regression labels per case study
#
# Several case studies trained the linear pipeline on multiple regression
# horizons. For each (case study, label), we take the highest complete
# mean daily IC across linear configurations and compare across horizons -
# this diagnoses whether the cross-sectional signal strengthens or
# weakens with the prediction window.


# %% tags=[]
def regression_labels(cs: str) -> list[str]:
    df = load_complete_metrics(cs)
    if df.is_empty():
        return []
    return [
        lbl
        for lbl in df["label"].unique().to_list()
        if lbl is not None
        and lbl.startswith("fwd_ret_")
        and "spot" not in lbl
        and "_win" not in lbl
        and "risk_adj" not in lbl
    ]


horizon_df = replace_with_chronological_hac(
    collect_complete_multi_label(CASE_STUDY_IDS, regression_labels)
)
print(
    f"Highest-IC complete linear configuration per (case study, regression label): "
    f"{horizon_df.height} rows"
)
horizon_df.select(
    "short_name",
    "label",
    pl.col("ic_mean_daily").round(4).alias("ic"),
    pl.col("ic_ci_lo").round(4).alias("ci_lo"),
    pl.col("ic_ci_hi").round(4).alias("ci_hi"),
    pl.col("ic_t_hac").round(2).alias("t_hac"),
)

# %% tags=[]
HORIZON_DAYS = {
    "fwd_ret_5m": 5 / (6.5 * 60),
    "fwd_ret_15m": 15 / (6.5 * 60),
    "fwd_ret_60m": 60 / (6.5 * 60),
    "fwd_ret_8h": 1.0 / 3,
    "fwd_ret_24h": 1.0,
    "fwd_ret_1d": 1.0,
    "fwd_ret_5d": 5.0,
    "fwd_ret_10d": 10.0,
    "fwd_ret_21d": 21.0,
    "fwd_ret_1m": 21.0,
    "fwd_ret_3m": 63.0,
    "fwd_ret_1m_win": 21.0,
    "fwd_ret_risk_adj_5d": 5.0,
}

plot_horizon = horizon_df.with_columns(
    horizon_days=pl.col("label").replace_strict(HORIZON_DAYS, default=None).cast(pl.Float64),
).filter(pl.col("horizon_days").is_not_null())

multi_cs = (
    plot_horizon.group_by("short_name").len().filter(pl.col("len") >= 2)["short_name"].to_list()
)
plot_horizon = plot_horizon.filter(pl.col("short_name").is_in(multi_cs))

# %% tags=[]
if plot_horizon.height > 0:
    palette = [
        COLORS["blue"],
        COLORS["copper"],
        COLORS["amber"],
        COLORS["positive"],
        COLORS["negative"],
        COLORS["slate"],
        COLORS["amber_light"],
        COLORS["neutral"],
    ]

# %% tags=[]
if plot_horizon.height > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    cs_sorted = sorted(plot_horizon["short_name"].unique().to_list())
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    for idx, cs in enumerate(cs_sorted):
        sub = plot_horizon.filter(pl.col("short_name") == cs).sort("horizon_days")
        if sub.height < 2:
            continue
        x = sub["horizon_days"].to_numpy()
        ic = sub["ic_mean_daily"].to_numpy()
        lo = sub["ic_ci_lo"].to_numpy()
        hi = sub["ic_ci_hi"].to_numpy()
        color = palette[idx % len(palette)]
        ax.fill_between(x, lo, hi, color=color, alpha=0.12)
        ax.plot(
            x,
            ic,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            color=color,
            label=cs,
            linewidth=1.6,
            markersize=6,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Horizon (trading days, log scale)")
    ax.set_ylabel("Mean daily IC (HAC 95 % CI band)")
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
    ax.set_title("Linear ranking strength changes unevenly with horizon")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.show()

# %% tags=[]
display(
    Markdown(
        f"The horizon view covers **{plot_horizon['short_name'].n_unique()}** case studies with "
        "at least two comparable regression labels. Direction and magnitude vary by panel, and "
        "the HAC bands show that many within-panel orderings remain uncertain. Each horizon is a "
        "separate estimand; the chart does not treat a longer horizon as more observations."
    )
)

# %% [markdown] tags=[]
# ### 5b. Classification ↔ regression metric symmetry
#
# A linear model trained on a continuous return label has a regression
# score; the same modelling apparatus can be trained on the binary
# direction label of the same horizon. Across the case studies that
# carry binary direction labels, two questions are symmetric:
#
# - **Direction A** - the classification model's score, evaluated as
#   IC against the *continuous* return, asks whether the directional
#   classifier is also a useful *cross-sectional ranker*. This number
#   is on the registry as `prediction_metrics.ic_mean_daily` for
#   `task_type='classification'` rows.
# - **Direction B** - the regression model's score, evaluated as mean daily AUC
#   against the *binary* direction, asks whether the continuous
#   regression score is also a useful *cross-sectional binary classifier*. This number
#   is computed on the fly here from raw OOF predictions, since the
#   registry stores AUC only for classification training runs.
#
# We restrict to case studies with a binary direction label paired to a
# regression label for the same horizon; ternary direction labels (e.g.
# NASDAQ-100 `fwd_dir_15m`, Crypto `fwd_dir_8h_3c`) need a multi-class
# AUC and are out of scope for this brief subsection.

# %% tags=[]
SYMMETRY_PAIRS: dict[str, list[tuple[str, str]]] = {
    # cs -> [(regression_label, binary_direction_label), ...]
    "crypto_perps_funding": [("fwd_ret_8h", "fwd_dir_8h")],
    "sp500_equity_option_analytics": [
        ("fwd_ret_5d", "fwd_dir_5d"),
        ("fwd_ret_10d", "fwd_dir_10d"),
    ],
    "us_firm_characteristics": [("fwd_ret_1m", "fwd_class_1m")],
}


# %% tags=[]
def _load_binary_label(cs: str, dir_label: str) -> pl.DataFrame:
    """Load one canonical binary direction label surface."""
    p = get_case_study_dir(cs) / "labels" / f"{dir_label}.parquet"
    if not p.exists():
        return pl.DataFrame()
    df = pl.read_parquet(p)
    return df.rename({dir_label: "y_dir"}).select("timestamp", "symbol", "y_dir")


# %% tags=[]
def _aligned_direction_scores(cs: str, reg_label: str, dir_label: str, best: dict) -> pl.DataFrame:
    """Join one exact regression prediction artifact to its binary labels."""
    preds = load_predictions(
        cs,
        family=FAMILY,
        label=reg_label,
        config_name=best["config_name"],
        checkpoint_value=best["checkpoint_value"],
        split="validation",
    )
    if preds.height == 0:
        return pl.DataFrame()
    preds = preds.filter(pl.col("prediction_hash") == best["prediction_hash"])
    if preds.is_empty():
        raise RuntimeError(f"{cs}: selected regression prediction artifact is missing")
    dir_df = _load_binary_label(cs, dir_label)
    if dir_df.is_empty():
        return pl.DataFrame()

    # Align timestamp + symbol dtypes for the join.
    if preds["timestamp"].dtype != dir_df["timestamp"].dtype:
        preds = preds.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
        dir_df = dir_df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    if preds["symbol"].dtype != dir_df["symbol"].dtype:
        preds = preds.with_columns(pl.col("symbol").cast(pl.Utf8))
        dir_df = dir_df.with_columns(pl.col("symbol").cast(pl.Utf8))

    return preds.join(dir_df, on=["timestamp", "symbol"], how="inner").filter(
        pl.col("y_dir").is_in([0, 1]) & pl.col("y_score").is_not_null()
    )


# %% tags=[]
def _mean_daily_auc(merged: pl.DataFrame) -> tuple[float, int] | None:
    """Compute unweighted mean daily AUC on dates containing both classes."""
    if merged.height == 0 or merged["y_dir"].n_unique() < 2:
        return None
    daily_auc = []
    for group in merged.sort("timestamp").partition_by("timestamp", maintain_order=True):
        if group["y_dir"].n_unique() < 2:
            continue
        daily_auc.append(roc_auc_score(group["y_dir"].to_numpy(), group["y_score"].to_numpy()))
    if not daily_auc:
        return None
    return float(np.mean(daily_auc)), len(daily_auc)


# %% tags=[]
def _direction_b_auc(cs: str, reg_label: str, dir_label: str) -> dict | None:
    """Highest-IC complete regression config to mean daily binary AUC."""
    reg_metrics = load_complete_metrics(cs, label=reg_label)
    if reg_metrics.is_empty():
        return None
    best = select_unique_best(reg_metrics, ["label"]).row(0, named=True)
    merged = _aligned_direction_scores(cs, reg_label, dir_label, best)
    auc_result = _mean_daily_auc(merged)
    if auc_result is None:
        return None
    auc, n_days = auc_result
    return {
        "case_study": cs,
        "short_name": SHORT_NAMES[cs],
        "reg_label": reg_label,
        "dir_label": dir_label,
        "reg_config": best["config_name"],
        "reg_ic_mean_daily": best["ic_mean_daily"],
        "reg_score_auc": auc,
        "n": merged.height,
        "n_days": n_days,
    }


# %% tags=[]
sym_rows = []
for cs, pairs in SYMMETRY_PAIRS.items():
    for reg_lbl, dir_lbl in pairs:
        # Direction A: classification model score to IC vs continuous return
        cls_metrics = load_complete_metrics(cs, label=dir_lbl)
        cls_ic = cls_ic_lo = cls_ic_hi = cls_t = None
        cls_config = None
        if not cls_metrics.is_empty():
            cls_metrics = cls_metrics.filter(pl.col("ic_mean_daily").is_not_null())
            if not cls_metrics.is_empty():
                selected_cls = select_unique_best(cls_metrics, ["label"]).with_columns(
                    case_study=pl.lit(cs)
                )
                top = replace_with_chronological_hac(selected_cls, target_column="eval_actual").row(
                    0, named=True
                )
                cls_ic = top["ic_mean_daily"]
                cls_ic_lo = top.get("ic_ci_lo")
                cls_ic_hi = top.get("ic_ci_hi")
                cls_t = top.get("ic_t_hac")
                cls_config = top["config_name"]
        # Direction B: regression model score to AUC vs binary direction
        b = _direction_b_auc(cs, reg_lbl, dir_lbl)
        sym_rows.append(
            {
                "short_name": SHORT_NAMES[cs],
                "reg_label": reg_lbl,
                "dir_label": dir_lbl,
                "cls_config": cls_config,
                "cls_score_ic": cls_ic,
                "cls_score_ic_lo": cls_ic_lo,
                "cls_score_ic_hi": cls_ic_hi,
                "cls_score_ic_t": cls_t,
                "reg_config": (b or {}).get("reg_config"),
                "reg_score_auc": (b or {}).get("reg_score_auc"),
                "n_b": (b or {}).get("n"),
                "n_b_days": (b or {}).get("n_days"),
            }
        )


# %% tags=[]
sym_df = pl.DataFrame(
    sym_rows,
    schema_overrides={
        "cls_score_ic": pl.Float64,
        "cls_score_ic_lo": pl.Float64,
        "cls_score_ic_hi": pl.Float64,
        "cls_score_ic_t": pl.Float64,
        "reg_score_auc": pl.Float64,
        "n_b": pl.Int64,
        "n_b_days": pl.Int64,
    },
)
print("Direction A (classification score to IC) and Direction B (regression score to AUC):")
sym_df.select(
    "short_name",
    "reg_label",
    "dir_label",
    pl.col("cls_score_ic").round(4).alias("A_ic"),
    pl.col("cls_score_ic_lo").round(4).alias("A_lo"),
    pl.col("cls_score_ic_hi").round(4).alias("A_hi"),
    pl.col("cls_score_ic_t").round(2).alias("A_t"),
    pl.col("reg_score_auc").round(4).alias("B_auc"),
    "n_b_days",
)

# %% [markdown] tags=[]
# ### Figure: Direction A and Direction B side by side

# %% tags=[]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
labels_y = [f"{r['short_name']} · {r['dir_label']}" for r in sym_df.iter_rows(named=True)]
y = np.arange(sym_df.height)

# Panel (a): Direction A, classification score IC vs continuous return
ax = axes[0]
ic = sym_df["cls_score_ic"].to_numpy()
lo = sym_df["cls_score_ic_lo"].to_numpy()
hi = sym_df["cls_score_ic_hi"].to_numpy()
ax.errorbar(
    ic,
    y,
    xerr=[ic - lo, hi - ic],
    fmt="o",
    color=COLORS["blue"],
    capsize=3,
    lw=1,
)
ax.axvline(0, color=COLORS["neutral"], lw=0.7, linestyle="--")
ax.set_yticks(y)
ax.set_yticklabels(labels_y)
ax.invert_yaxis()
ax.set_xlabel("Mean daily IC (HAC 95 % CI)")
ax.set_title("(a) Classification scores can rank continuous returns")

# Panel (b): Direction B, regression score AUC vs binary direction
ax = axes[1]
auc = sym_df["reg_score_auc"].to_numpy()
ax.scatter(auc, y, color=COLORS["copper"], s=60, zorder=3)
ax.axvline(0.5, color=COLORS["neutral"], lw=0.7, linestyle="--")
ax.set_yticks(y)
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlabel("Mean daily AUC (regression score)")
ax.set_title("(b) Regression scores stay near chance on direction")
ax.set_xlim(0.47, 0.53)

if fig.get_layout_engine() is None:
    fig.tight_layout()
fig.show()

# %% tags=[]
direction_a_clear = sym_df.filter((pl.col("cls_score_ic_lo") > 0) | (pl.col("cls_score_ic_hi") < 0))
max_auc_gap = float((sym_df["reg_score_auc"] - 0.5).abs().max())
display(
    Markdown(
        f"Direction A's HAC interval excludes zero in **{direction_a_clear.height} of "
        f"{sym_df.height}** comparisons. Direction B's mean daily AUC remains within "
        f"**{max_auc_gap:.3f}** of 0.5 across the same pairs. Cross-sectional rank correlation "
        "therefore does not collapse to binary-direction discrimination. The Ch16 equal-weight "
        "baseline uses ranked predictions, while direction-aware classifiers can still inform "
        "long-short construction."
    )
)

# %% [markdown] tags=[]
# ## 6. Coefficient Analysis
#
# Coefficient diagnostics use the per-fold coefficient parquets persisted
# by the linear training pipeline.


# %% tags=[]
def load_coefficients(cs: str, training_hash: str, config_name: str) -> pl.DataFrame | None:
    """Load coefficients from one exact registry-tracked training run."""
    cs_dir = get_case_study_dir(cs)
    coef_path = cs_dir / "run_log" / "training" / training_hash / "coefficients.parquet"
    if not coef_path.exists():
        return None
    return pl.read_parquet(coef_path).with_columns(pl.lit(config_name).alias("config_name"))


# %% [markdown] tags=[]
# ### 6a. Sign consistency across folds
#
# For each case study's highest-IC linear configuration on the primary
# label, the fraction of folds where each coefficient maintains its
# modal sign is a stability diagnostic. Coefficients with consistency
# $\geq 0.8$ are sign-stable; the per-CS mean across features summarizes
# how identifiable the linear fit is.

# %% tags=[]
sign_rows = []
for row in rank1.iter_rows(named=True):
    cs = row["case_study"]
    cfg = row["config_name"]
    coef_df = load_coefficients(cs, row["training_hash"], cfg)
    if coef_df is None:
        continue
    coefs = coef_df.filter(pl.col("config_name") == cfg)
    if coefs.is_empty():
        continue
    stats = (
        coefs.group_by("feature")
        .agg(
            n_pos=(pl.col("coefficient") > 0).sum(),
            n_neg=(pl.col("coefficient") < 0).sum(),
            n_zero=(pl.col("coefficient").abs() <= 1e-10).sum(),
            n_folds=pl.col("coefficient").count(),
        )
        .with_columns(
            consistency=(pl.max_horizontal("n_pos", "n_neg", "n_zero") / pl.col("n_folds")),
        )
    )
    sign_rows.append(
        {
            "short_name": row["short_name"],
            "config_name": cfg,
            "n_features": stats.height,
            "mean_consistency": float(stats["consistency"].mean()),
            "pct_features_above_80": float((stats["consistency"] >= 0.8).mean() * 100),
        }
    )

sign_df = pl.DataFrame(sign_rows).sort("mean_consistency", descending=True)
print("Coefficient sign consistency for the highest-IC linear configuration (primary label):")
sign_df

# %% tags=[]
if sign_df.height > 0:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    plot_sign = sign_df.sort("mean_consistency", descending=True)
    y = np.arange(plot_sign.height)
    ax.barh(
        y,
        plot_sign["mean_consistency"].to_numpy(),
        color=COLORS["blue"],
        alpha=0.9,
        height=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_sign["short_name"].to_list())
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.8, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
    ax.set_xlabel("Mean fold-level sign consistency")
    ax.set_title("Most selected linear fits preserve coefficient signs across folds")
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.show()

# %% tags=[]
stable_sign_count = sign_df.filter(pl.col("mean_consistency") >= 0.8).height
sign_leader = sign_df.row(0, named=True)
display(
    Markdown(
        f"Mean fold-level sign consistency reaches 0.8 in **{stable_sign_count} of "
        f"{sign_df.height}** selected fits. **{sign_leader['short_name']}** is highest at "
        f"{sign_leader['mean_consistency']:.2f}. Sign stability and predictive ranking are "
        "different diagnostics: a stable coefficient direction does not by itself establish "
        "a nonzero out-of-sample IC."
    )
)

# %% [markdown] tags=[]
# ### 6b. Lasso sparsity
#
# Lasso drives coefficients exactly to zero. The fraction of zero
# coefficients per case study summarizes how aggressively Lasso prunes
# features at the highest-IC $\alpha$ for the Lasso family.

# %% tags=[]
sparsity_rows = []
for cs in CASE_STUDY_IDS:
    label = PRIMARY_LABELS.get(cs)
    if label is None:
        continue
    lasso_metrics = load_complete_metrics(cs, label=label).filter(
        pl.col("config_name").str.starts_with("lasso")
    )
    if lasso_metrics.is_empty():
        continue
    lasso_selected = select_unique_best(lasso_metrics, ["label"]).row(0, named=True)
    lasso_config = lasso_selected["config_name"]
    lasso = load_coefficients(cs, lasso_selected["training_hash"], lasso_config)
    if lasso is None:
        continue
    n_total = lasso.height
    n_zero = lasso.filter(pl.col("coefficient").abs() < 1e-10).height
    always_zero = (
        lasso.group_by("feature")
        .agg(all_zero=(pl.col("coefficient").abs() < 1e-10).all())
        .filter(pl.col("all_zero"))
        .height
    )
    n_features = lasso["feature"].n_unique()
    sparsity_rows.append(
        {
            "short_name": SHORT_NAMES[cs],
            "config_name": lasso_config,
            "n_features": n_features,
            "zero_fraction": round(n_zero / n_total, 3),
            "features_always_zero": always_zero,
            "pct_always_zero": round(always_zero / n_features * 100, 1),
        }
    )


# %% tags=[]
sparsity_df = pl.DataFrame(
    sparsity_rows,
    schema={
        "short_name": pl.Utf8,
        "config_name": pl.Utf8,
        "n_features": pl.Int64,
        "zero_fraction": pl.Float64,
        "features_always_zero": pl.Int64,
        "pct_always_zero": pl.Float64,
    },
)
sparsity_df = sparsity_df.sort("zero_fraction", descending=True)
print("Lasso sparsity per case study (at the highest-IC Lasso $\\alpha$):")
sparsity_df

# %% tags=[]
sparsest = sparsity_df.row(0, named=True)
display(
    Markdown(
        f"Complete highest-IC Lasso coefficient artifacts are available for "
        f"**{sparsity_df.height}** case studies. **{sparsest['short_name']}** is sparsest: "
        f"{sparsest['zero_fraction']:.1%} of coefficient-by-fold cells are zero, while "
        f"{sparsest['pct_always_zero']:.1f}% of features are zero in every fold. This comparison "
        "now uses one selected Lasso configuration per case study rather than pooling all alphas."
    )
)

# %% [markdown] tags=[]
# ### 6c. Top-N feature overlap across case studies
#
# Are the same features driving the linear signal across case studies,
# or does each case study have its own world? For each case study's
# highest-IC linear configuration on the primary label, we rank
# features by mean absolute fold coefficient and take the top 10. The
# pairwise Jaccard overlap of these sets shows how much feature
# selection generalizes across panels.

# %% tags=[]
TOP_N = 10
top_features: dict[str, set[str]] = {}
for row in rank1.iter_rows(named=True):
    cs = row["case_study"]
    cfg = row["config_name"]
    coef_df = load_coefficients(cs, row["training_hash"], cfg)
    if coef_df is None:
        continue
    coefs = coef_df.filter(pl.col("config_name") == cfg)
    if coefs.is_empty():
        continue
    feature_score = (
        coefs.group_by("feature")
        .agg(score=pl.col("coefficient").abs().mean())
        .sort("score", descending=True)
        .head(TOP_N)
    )
    top_features[row["short_name"]] = set(feature_score["feature"].to_list())

print(f"Top-{TOP_N} feature sets collected for {len(top_features)} case studies.")


# %% tags=[]
def jaccard(a: set, b: set) -> float:
    """Return Jaccard overlap for two nonempty feature sets."""
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


cs_names = sorted(top_features.keys())
J = np.full((len(cs_names), len(cs_names)), np.nan)
for i, a in enumerate(cs_names):
    for j, b in enumerate(cs_names):
        if i == j:
            J[i, j] = 1.0
        else:
            J[i, j] = jaccard(top_features[a], top_features[b])

# %% tags=[]
if cs_names:
    fig, ax = plt.subplots(figsize=(7, 6))
    overlap_cmap = LinearSegmentedColormap.from_list(
        "ml4t_overlap", [COLORS["bg_light"], COLORS["blue_light"], COLORS["blue"]]
    )
    im = ax.imshow(J, cmap=overlap_cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cs_names)))
    ax.set_yticks(np.arange(len(cs_names)))
    ax.set_xticklabels(cs_names, rotation=40, ha="right")
    ax.set_yticklabels(cs_names)
    for i in range(len(cs_names)):
        for j in range(len(cs_names)):
            v = J[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v > 0.5 else COLORS["blue"],
                )
    ax.set_title(f"Top-{TOP_N} coefficient features overlap little across panels")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Jaccard overlap")
    fig.show()

# %% tags=[]
off_diagonal = J[~np.eye(len(cs_names), dtype=bool)]
max_overlap = float(np.nanmax(off_diagonal)) if off_diagonal.size else float("nan")
display(
    Markdown(
        f"The largest off-diagonal top-{TOP_N} Jaccard overlap is **{max_overlap:.2f}**. "
        "Each selected fit draws primarily on its panel-specific feature library; recurring "
        "momentum or volatility primitives do not form a universal short list."
    )
)

# %% [markdown] tags=[]
# ## 7. Cross-CS Takeaways
#
# **Next**: Ch12 extends the comparison with gradient boosting and
# tabular deep learning; Ch13 with temporal deep learning.

# %% tags=[]
display(
    Markdown(
        f"- **Primary-label evidence is selective:** {significant.height} of {rank1.height} "
        "complete linear results have HAC intervals that exclude zero.\n"
        f"- **Ridge is a durable baseline:** it matches or exceeds OLS in {ridge_wins} of "
        f"{ridge_ols_wide.height} comparable panels, while most differences remain small relative "
        "to uncertainty.\n"
        f"- **Fold stability needs context:** {icir_leader['short_name']} leads ICIR, but its "
        f"estimate uses only {icir_leader['n_folds']} folds.\n"
        f"- **Ranking and direction are not symmetric:** regression-score mean daily AUC stays "
        f"within {max_auc_gap:.3f} of chance in the tested pairs.\n"
        f"- **Coefficient behavior is panel-specific:** {stable_sign_count} selected fits clear "
        f"0.8 mean sign consistency, yet the largest top-{TOP_N} cross-panel feature overlap is "
        f"only {max_overlap:.2f}."
    )
)
