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
# # Case Study Insights: Latent Factors
#
# **Purpose**: synthesize the latent-factor results that are registered across
# the nine case studies and compare selected latent and supervised models on a
# common validation sample.
#
# **Learning objectives**
#
# - Read estimator coverage and the highest mean daily rank IC by panel
# - Compare PCA, IPCA, CAE, SDF, and SAE without mixing ranking statistics
# - Evaluate selected latent and supervised models with paired per-date IC
# - Measure whether neural latent estimators produce complementary rankings
#
# **Book reference**: Section 14.8 (Cross-Case-Study Synthesis).
#
# **Prerequisites**: the case-study pipelines have populated their
# `run_log/registry.db` files and prediction artifacts. This notebook reads
# those immutable artifacts; it does not train models or write results.

# %%
"""Cross-case-study synthesis of latent-factor validation results."""

# case_studies.utils.model_analysis imports lightgbm at module scope, so it must
# not be reached after ml4t.diagnostic, which brings scikit-learn up: both ship
# an OpenMP runtime and the first loaded wins for the process, which segfaults
# LightGBM on macOS ARM64. This notebook only reads registries and fits nothing,
# so the binding is stated here rather than left to import order further down.
import lightgbm  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Load torch before ml4t.diagnostic so its bundled CUDA runtime wins.
import torch  # noqa: F401
from ml4t.diagnostic.metrics import compute_ic_uncertainty

from case_studies.utils.analytics import CASE_STUDY_IDS, PRIMARY_LABELS, SHORT_NAMES
from case_studies.utils.insight_chapter import collect_rank1_per_cs
from case_studies.utils.model_analysis import load_metrics_from_registry, load_predictions
from utils.reproducibility import set_global_seeds
from utils.style import (
    COLORS,
    FIGSIZE,
    add_message_title,
    zero_line,
)

# %% tags=["parameters"]
FAMILY = "latent_factors"
SUPERVISED_FAMILIES = ("linear", "gbm", "tabular_dl", "deep_learning")
ESTIMATORS = ("pca", "ipca", "cae", "sdf", "sae")
N_BOOT = 1000
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# Registry config names are terse. This display map keeps the figures readable
# while leaving selection keyed to the original registry values.

# %%
ESTIMATOR_NAMES = {
    "pca": "PCA",
    "ipca": "IPCA",
    "cae": "CAE",
    "sdf": "SDF",
    "sae": "SAE",
}
FAMILY_NAMES = {
    "linear": "Linear",
    "gbm": "GBM",
    "tabular_dl": "TabM",
    "deep_learning": "Deep learning",
}

# %% [markdown]
# ## 1. Registry coverage
#
# Coverage is an observed property of the frozen registries, not a claim that
# an absent estimator is unsuitable. A blank cell means that no validation IC
# row is registered for that estimator on the panel's primary label.

# %%
qualifying_case_studies = [
    cs for cs in CASE_STUDY_IDS if not load_metrics_from_registry(cs, families=[FAMILY]).is_empty()
]
print(
    f"Loaded latent-factor results for {len(qualifying_case_studies)} of "
    f"{len(CASE_STUDY_IDS)} case studies."
)

# %%
coverage_rows = []
for case_study in qualifying_case_studies:
    metrics = load_metrics_from_registry(
        case_study,
        label=PRIMARY_LABELS[case_study],
        families=[FAMILY],
    )
    available = set(metrics["config_name"].unique().to_list())
    coverage_rows.append(
        {
            "case_study": SHORT_NAMES[case_study],
            **{estimator: int(estimator in available) for estimator in ESTIMATORS},
        }
    )
coverage = pl.DataFrame(coverage_rows)

# %% [markdown]
# The coverage map shows a deliberately uneven experiment grid. Broad panels
# carry conditional or neural estimators, while the narrow CME panel contains
# only PCA and SDF in the current registry snapshot.

# %%
coverage_values = coverage.select(ESTIMATORS).to_numpy()
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.imshow(coverage_values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ESTIMATORS)), [ESTIMATOR_NAMES[e] for e in ESTIMATORS])
ax.set_yticks(range(coverage.height), coverage["case_study"].to_list())
ax.set_xlabel("Latent-factor estimator")
ax.set_ylabel("Case study")
for row in range(coverage.height):
    for col in range(len(ESTIMATORS)):
        label = "available" if coverage_values[row, col] else "not run"
        color = COLORS["silver"] if coverage_values[row, col] else COLORS["neutral"]
        ax.text(col, row, label, ha="center", va="center", fontsize=8, color=color)
add_message_title(
    ax,
    "Estimator coverage varies with the panel",
    subtitle="Registered validation results at each case study's primary label",
)
fig.show()

# %% [markdown]
# ## 2. Highest mean daily IC by case study
#
# Selection and reporting use the same statistic: Spearman rank IC is computed
# within each decision date and then averaged over dates. The error bars are
# the registry's HAC 95% intervals, with lags that reflect the label horizon.

# %%
latent_winners = collect_rank1_per_cs(qualifying_case_studies, family=FAMILY).with_columns(
    estimator=pl.col("config_name").replace_strict(
        ESTIMATOR_NAMES,
        default=pl.col("config_name").str.to_uppercase(),
    )
)
latent_winners = latent_winners.sort("ic_mean_daily")

# %%
names = latent_winners["short_name"].to_list()
means = latent_winners["ic_mean_daily"].to_numpy()
lower = means - latent_winners["ic_ci_lo"].to_numpy()
upper = latent_winners["ic_ci_hi"].to_numpy() - means
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.errorbar(
    means,
    range(len(names)),
    xerr=np.vstack([lower, upper]),
    fmt="o",
    color=COLORS["blue"],
    ecolor=COLORS["slate"],
    capsize=3,
)
for y, row in enumerate(latent_winners.iter_rows(named=True)):
    ax.annotate(
        f"{row['estimator']}  {row['ic_mean_daily']:+.3f}",
        (row["ic_mean_daily"], y),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=8,
    )
ax.set_yticks(range(len(names)), names)
ax.set_xlabel("Mean daily Spearman IC (HAC 95% interval)")
ax.set_ylabel("Case study")
zero_line(ax, axis="x")
add_message_title(
    ax,
    "The strongest latent estimator depends on the panel",
    subtitle="Highest registered validation IC at the primary label",
)
fig.show()

# %% [markdown]
# SDF leads on the ETF, equity-option, and futures panels. SAE leads on US
# Firms, while IPCA leads on US Equities. The result rejects a universal
# estimator ranking: the objective that works best depends on panel structure
# and the prediction target.

# %% [markdown]
# ## 3. The US Firms objective ladder
#
# US Firms is the only registered primary-label panel with IPCA and all three
# neural objectives. Comparing their best checkpoints isolates the role of the
# training objective while keeping the dataset and target fixed.

# %%
us_firms_metrics = (
    load_metrics_from_registry(
        "us_firm_characteristics",
        label=PRIMARY_LABELS["us_firm_characteristics"],
        families=[FAMILY],
    )
    .filter(pl.col("ic_mean_daily").is_not_null())
    .sort("ic_mean_daily", descending=True)
    .group_by("config_name", maintain_order=True)
    .first()
    .with_columns(
        estimator=pl.col("config_name").replace_strict(
            ESTIMATOR_NAMES,
            default=pl.col("config_name").str.to_uppercase(),
        )
    )
    .sort("ic_mean_daily")
)

# %%
objective_names = us_firms_metrics["estimator"].to_list()
objective_means = us_firms_metrics["ic_mean_daily"].to_numpy()
objective_lo = objective_means - us_firms_metrics["ic_ci_lo"].to_numpy()
objective_hi = us_firms_metrics["ic_ci_hi"].to_numpy() - objective_means
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.errorbar(
    objective_means,
    range(len(objective_names)),
    xerr=np.vstack([objective_lo, objective_hi]),
    fmt="o",
    color=COLORS["amber"],
    ecolor=COLORS["copper"],
    capsize=3,
)
for y, value in enumerate(objective_means):
    ax.annotate(f"{value:+.3f}", (value, y), xytext=(5, 5), textcoords="offset points")
ax.set_yticks(range(len(objective_names)), objective_names)
ax.set_xlabel("Mean daily Spearman IC (HAC 95% interval)")
ax.set_ylabel("Estimator")
zero_line(ax, axis="x")
add_message_title(
    ax,
    "Supervised reconstruction leads the US Firms objective ladder",
    subtitle="Best checkpoint per estimator, monthly primary label",
)
fig.show()

# %% [markdown]
# SAE's joint reconstruction and prediction objective leads this panel. CAE's
# reconstruction-only score points in the opposite direction, and IPCA and SDF
# sit between them. This is a validation comparison, so it diagnoses objective
# alignment rather than estimating final holdout performance.

# %% [markdown]
# ## 4. Latent factors versus supervised models
#
# The registry identifies the highest mean daily IC within each supervised
# family and then across families. For each selected pair, predictions are
# inner-joined on the same timestamp-entity keys. Both model ICs and their
# difference are recomputed within each date on that identical cross-section.
#
# The HAC interval is conditional on validation-based model selection. It is a
# paired stability diagnostic, not an unbiased post-selection hypothesis test.

# %% [markdown]
# A label's suffix determines the minimum dependence horizon. Monthly targets
# use one monthly period; daily targets use their stated number of days.


# %%
def label_horizon(label: str) -> int:
    """Return the target horizon in its observation periods."""
    suffix = label.rsplit("_", maxsplit=1)[-1]
    digits = "".join(character for character in suffix if character.isdigit())
    return max(1, int(digits)) if digits else 1


# %% [markdown]
# CME prediction artifacts use the canonical `product` identifier in the
# supervised pipeline and `symbol` in the older latent artifact. Both identify
# the same futures contract, so the loader normalizes either field to `entity`
# only inside this comparison frame.


# %%
def selected_predictions(case_study: str, row: dict, score_name: str) -> pl.DataFrame:
    """Load one selected validation prediction set with a common entity key."""
    frame = load_predictions(
        case_study,
        family=row["family"],
        label=row["label"],
        config_name=row["config_name"],
        checkpoint_value=row["checkpoint_value"],
        split="validation",
    )
    entity = "product" if "product" in frame.columns else "symbol"
    selected = frame.select(
        pl.col("timestamp").cast(pl.Datetime("us")),
        pl.col(entity).cast(pl.Utf8).alias("entity"),
        pl.col("y_true").alias(f"{score_name}_target"),
        pl.col("y_score").alias(f"{score_name}_score"),
    )
    if selected.height != selected.unique(["timestamp", "entity"]).height:
        raise ValueError(f"Duplicate validation keys for {case_study} {score_name}")
    return selected


# %% [markdown]
# The paired calculation ranks both scores and the common target within each
# date. Pearson correlation of those within-date ranks is Spearman IC. The
# function returns the complete, time-sorted daily series for independent
# uncertainty calculation.


# %%
def paired_daily_ic(latent: pl.DataFrame, supervised: pl.DataFrame) -> pl.DataFrame:
    """Compute paired per-date IC after exact timestamp-entity alignment."""
    joined = latent.join(supervised, on=["timestamp", "entity"], how="inner")
    target_gap = joined.select(
        (pl.col("latent_target") - pl.col("supervised_target")).abs().max()
    ).item()
    if target_gap is None or target_gap > 1e-10:
        raise ValueError(f"Aligned targets disagree: maximum gap {target_gap}")
    ranked = joined.with_columns(
        pl.col("latent_score").rank(method="average").over("timestamp").alias("latent_rank"),
        pl.col("supervised_score")
        .rank(method="average")
        .over("timestamp")
        .alias("supervised_rank"),
        pl.col("latent_target").rank(method="average").over("timestamp").alias("target_rank"),
    )
    return (
        ranked.group_by("timestamp")
        .agg(
            pl.len().alias("n_obs"),
            pl.corr("latent_rank", "target_rank").alias("latent_ic"),
            pl.corr("supervised_rank", "target_rank").alias("supervised_ic"),
        )
        .filter(pl.col("n_obs") >= 5)
        .with_columns((pl.col("latent_ic") - pl.col("supervised_ic")).alias("delta"))
        .drop_nulls(["latent_ic", "supervised_ic", "delta"])
        .sort("timestamp")
    )


# %% [markdown]
# Select the strongest supervised registry row for each panel using
# `ic_mean_daily`, the same column used for the latent winners and all displayed
# rankings.

# %%
supervised_winners = {}
for case_study in qualifying_case_studies:
    metrics = load_metrics_from_registry(
        case_study,
        label=PRIMARY_LABELS[case_study],
        families=list(SUPERVISED_FAMILIES),
    ).filter(pl.col("ic_mean_daily").is_not_null())
    supervised_winners[case_study] = metrics.sort("ic_mean_daily", descending=True).row(
        0, named=True
    )

# %%
comparison_rows = []
paired_series = {}
for latent_row in latent_winners.iter_rows(named=True):
    case_study = latent_row["case_study"]
    supervised_row = supervised_winners[case_study]
    latent = selected_predictions(case_study, latent_row, "latent")
    supervised = selected_predictions(case_study, supervised_row, "supervised")
    daily = paired_daily_ic(latent, supervised)
    paired_series[case_study] = daily
    uncertainty = compute_ic_uncertainty(
        daily.select(pl.col("delta").alias("ic")),
        horizon=label_horizon(latent_row["label"]),
        n_boot=N_BOOT,
        seed=SEED,
    )
    comparison_rows.append(
        {
            "case_study": case_study,
            "short_name": latent_row["short_name"],
            "latent_name": latent_row["estimator"],
            "supervised_name": FAMILY_NAMES[supervised_row["family"]],
            "latent_ic": daily["latent_ic"].mean(),
            "supervised_ic": daily["supervised_ic"].mean(),
            "delta": uncertainty["mean_ic"],
            "delta_lo": uncertainty["ci_hac_lower"],
            "delta_hi": uncertainty["ci_hac_upper"],
            "n_dates": uncertainty["n_days"],
            "n_common": latent.join(supervised, on=["timestamp", "entity"], how="inner").height,
        }
    )
comparison = pl.DataFrame(comparison_rows).sort("delta")

# %% [markdown]
# The left panel compares paired-sample mean IC. The right panel isolates the
# latent-minus-supervised difference with its HAC interval. Point estimates
# left of zero favor the selected supervised model.

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dashboard_2x2"], constrained_layout=False)
fig.subplots_adjust(hspace=0.65, left=0.18, right=0.98, top=0.94, bottom=0.1)
y = np.arange(comparison.height)
for idx, row in enumerate(comparison.iter_rows(named=True)):
    axes[0].plot(
        [row["supervised_ic"], row["latent_ic"]],
        [idx, idx],
        color=COLORS["silver_muted"],
        linewidth=2,
    )
axes[0].scatter(comparison["supervised_ic"], y, color=COLORS["amber"], label="Supervised")
axes[0].scatter(comparison["latent_ic"], y, color=COLORS["blue"], label="Latent")
axes[0].set_yticks(y, comparison["short_name"].to_list())
axes[0].set_xlabel("Mean daily Spearman IC")
axes[0].set_ylabel("Case study")
axes[0].legend(loc="lower right", ncol=2)
zero_line(axes[0], axis="x")
add_message_title(axes[0], "Panel structure determines the stronger model class")

delta = comparison["delta"].to_numpy()
delta_lo = delta - comparison["delta_lo"].to_numpy()
delta_hi = comparison["delta_hi"].to_numpy() - delta
axes[1].errorbar(
    delta,
    y,
    xerr=np.vstack([delta_lo, delta_hi]),
    fmt="o",
    color=COLORS["blue"],
    ecolor=COLORS["slate"],
    capsize=3,
)
for idx, value in enumerate(delta):
    axes[1].annotate(f"{value:+.3f}", (value, idx), xytext=(4, 5), textcoords="offset points")
axes[1].set_yticks(y, comparison["short_name"].to_list())
axes[1].set_xlabel("Latent minus supervised mean daily IC (HAC 95% interval)")
axes[1].set_ylabel("Case study")
zero_line(axes[1], axis="x")
add_message_title(axes[1], "Only paired dates and entities enter each difference")
fig.show()

# %% [markdown]
# Latent models lead on some panels and lag on others. The paired construction
# makes this comparison interpretable: each difference uses the same assets on
# the same dates. The intervals still reflect validation uncertainty after
# winner selection and therefore should not be read as final holdout tests.

# %% [markdown]
# ## 5. Neural estimator agreement on US Firms
#
# Low average rank correlation means two estimators order firms differently,
# which can support ensemble diversification. Correlations are computed within
# month first and then averaged, so months with larger cross-sections do not
# dominate the diagnostic.

# %% [markdown]
# This helper aligns a selected pair on common observations and returns the
# mean of its monthly Spearman correlation series.


# %%
def mean_daily_score_correlation(left: pl.DataFrame, right: pl.DataFrame) -> float:
    """Average per-date Spearman correlation on common prediction rows."""
    joined = left.join(right, on=["timestamp", "entity"], how="inner")
    ranked = joined.with_columns(
        pl.col("left_score").rank(method="average").over("timestamp").alias("left_rank"),
        pl.col("right_score").rank(method="average").over("timestamp").alias("right_rank"),
    )
    daily = (
        ranked.group_by("timestamp")
        .agg(pl.len().alias("n_obs"), pl.corr("left_rank", "right_rank").alias("correlation"))
        .filter(pl.col("n_obs") >= 5)
        .drop_nulls("correlation")
        .sort("timestamp")
    )
    return float(daily["correlation"].mean())


# %%
neural_metrics = (
    load_metrics_from_registry(
        "us_firm_characteristics",
        label=PRIMARY_LABELS["us_firm_characteristics"],
        families=[FAMILY],
    )
    .filter(pl.col("config_name").is_in(["cae", "sdf", "sae"]))
    .filter(pl.col("ic_mean_daily").is_not_null())
    .sort("ic_mean_daily", descending=True)
    .group_by("config_name", maintain_order=True)
    .first()
)
neural_predictions = {}
for row in neural_metrics.iter_rows(named=True):
    frame = selected_predictions("us_firm_characteristics", row, "left")
    neural_predictions[row["config_name"]] = frame.select("timestamp", "entity", "left_score")

# %%
neural_names = sorted(neural_predictions)
agreement = np.eye(len(neural_names))
for i, left_name in enumerate(neural_names):
    for j in range(i + 1, len(neural_names)):
        right_name = neural_names[j]
        left = neural_predictions[left_name]
        right = neural_predictions[right_name].rename({"left_score": "right_score"})
        agreement[i, j] = mean_daily_score_correlation(left, right)
        agreement[j, i] = agreement[i, j]

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
image = ax.imshow(agreement, cmap="RdBu_r", vmin=-1, vmax=1)
display_names = [ESTIMATOR_NAMES[name] for name in neural_names]
ax.set_xticks(range(len(display_names)), display_names)
ax.set_yticks(range(len(display_names)), display_names)
ax.set_xlabel("Neural latent estimator")
ax.set_ylabel("Neural latent estimator")
for row in range(len(display_names)):
    for col in range(len(display_names)):
        ax.text(col, row, f"{agreement[row, col]:+.2f}", ha="center", va="center")
colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
colorbar.set_label("Mean monthly Spearman correlation")
add_message_title(
    ax,
    "Neural objectives produce distinct US Firms rankings",
    subtitle="Per-month correlations, averaged over common validation months",
)
fig.show()

# %% [markdown]
# The off-diagonal correlations are far from one, so the neural objectives do
# not merely repackage the same ranking. This supports testing them as separate
# ensemble inputs, but the eventual portfolio decision belongs in Chapter 20
# and must use its sealed evaluation protocol.

# %% [markdown]
# ## Key takeaways
#
# - Registry coverage is uneven, so missing cells are not performance results.
# - No latent estimator wins across every panel; SDF, SAE, and IPCA each lead
#   somewhere on the primary-label validation results.
# - SAE leads the US Firms objective ladder, while reconstruction-only CAE
#   points in the opposite direction on that panel.
# - Paired per-date comparisons show that neither latent nor supervised models
#   dominate everywhere. These are post-selection validation diagnostics.
# - Neural objectives create distinct monthly firm rankings, making model
#   diversity a testable input to Chapter 20 rather than an assumption.
#
# **Next**: Chapter 15 studies causal effects; Chapter 20 evaluates how these
# predictive signals combine under a sealed portfolio protocol.
