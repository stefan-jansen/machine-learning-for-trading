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
# # Case Study Insights: Deep Learning for Time Series
#
# **Docker image**: `ml4t`
#
# **Purpose**: assemble the cross-case-study view of temporal deep learning
# (LSTM, NLinear, TSMixer, TCN, PatchTST) and contrast it with the linear
# baseline (Ch11), gradient boosting (Ch12), and the tabular DL adapter TabM
# (Ch12). Per-case-study deep dives live in `case_studies/{cs}/11_model_analysis.py`;
# this notebook is the comparative view across the eight case studies that carry
# DL pipelines.
#
# **Learning objectives**
#
# - For each case study, read the highest-IC DL configuration's average daily
#   Spearman IC with HAC 95 % CI on the primary label
# - Trace the architecture × case-study coverage map and the per-architecture
#   IC at the primary label
# - Inspect per-fold IC distributions, checkpoint dynamics, and conformal
#   coverage at the 90 % nominal level
# - Compare full-coverage DL and tabular daily-IC point estimates without
#   treating fold summaries as an uncertainty estimator
# - Place the DL family inside the architectural-class taxonomy (recurrent,
#   MLP-style, convolutional, attention)
#
# **Book reference**: Section 13.7 (Practitioner Framework) and Section 13.9
# (Cross-Case-Study Synthesis).
#
# **Prerequisites**: each case study's per-architecture training notebooks
# (`dl_lstm.py`, `dl_nlinear.py`, `dl_tsmixer.py`, `dl_tcn.py`, `dl_patchtst.py`)
# have populated `run_log/registry.db` for the `deep_learning` family. The
# linear, GBM, and TabM baselines come from Ch11-Ch12 pipelines.

# %%
"""Case Study Insights: Deep Learning cross-case-study registry aggregation."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ml4t.diagnostic dlopens cudart; load torch first so its bundled CUDA
# runtime wins. Same precedence pattern as case_studies/utils/model_analysis.py.
import torch  # noqa: F401
from IPython.display import Markdown, display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# %%
# Every comparison below ranks only configurations that covered the same folds
# and the same number of days, so a shorter evaluation window cannot win.
from case_studies.utils.analytics import (
    CASE_STUDY_IDS,
    DATASET_META,
    PRIMARY_LABELS,
    SHORT_NAMES,
)
from case_studies.utils.insight_chapter import (
    RegistrySelectionError,
    collect_checkpoint_fold_trajectories,
    collect_fold_ic_per_cs,
    collect_grid_per_cs,
    collect_multi_label_per_cs,
    collect_rank1_per_cs,
    compare_ic_on_shared_timestamps,
    conformal_coverage_for_selected_prediction,
    plot_cross_cs_forest,
    plot_multi_label_horizon,
    plot_per_fold_violin,
)
from case_studies.utils.model_analysis import (
    load_daily_metrics_series,
    load_metrics_from_registry,
)
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_diverging, ml4t_palette

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
FAMILY = "deep_learning"
TABULAR_BASELINES = ("linear", "gbm", "tabular_dl")
CONFORMAL_LEVEL = 0.90

# %%
set_global_seeds(SEED)


# %% [markdown]
# **Architecture helper**: maps registry `config_name` values to the display
# names used across every downstream table and figure so the four LSTM
# variants collapse to a single `LSTM` row and the per-architecture grids
# stay consistent.


# %%
def architecture(config_name: str) -> str:
    """Display name for a DL config_name (LSTM variants collapse to 'LSTM')."""
    if config_name.startswith("lstm"):
        return "LSTM"
    return {"nlinear": "NLinear", "tsmixer": "TSMixer", "tcn": "TCN", "patchtst": "PatchTST"}.get(
        config_name,
        config_name.upper(),
    )


# %% [markdown]
# Architectural classes support the aggregate comparison, while the complete
# grid loader keeps missing architectures explicit.

# %%
ARCH_CLASS = {
    "LSTM": "recurrent",
    "NLinear": "MLP-style",
    "TSMixer": "MLP-style",
    "TCN": "convolutional",
    "PatchTST": "attention",
}

dl_grid = collect_grid_per_cs(
    CASE_STUDY_IDS,
    FAMILY,
)


# %% [markdown]
# ## 1. Scope and Coverage
#
# The DL grid varies across the case studies - not every architecture trains
# on every panel, and US Firm Characteristics (monthly cross-section) carries
# no DL family at all. The headline metric is the average daily Spearman IC
# with HAC 95 % confidence interval on each case study's primary label
# (`prediction_metrics.ic_mean_daily`, `ic_ci_lo`, `ic_ci_hi`, `ic_t_hac`).
# The linear (Ch11), GBM (Ch12), and TabM (`tabular_dl`, Ch12) baselines are
# loaded so the §5 delta is computable.

# %%
coverage_rows = []
for cs in CASE_STUDY_IDS:
    primary = PRIMARY_LABELS[cs]
    dl = dl_grid.filter(pl.col("case_study") == cs)
    if dl.is_empty():
        coverage_rows.append(
            {
                "case_study": SHORT_NAMES[cs],
                "primary_label": primary,
                "dl_archs": "-",
                "n_dl_configs": 0,
            }
        )
        continue
    archs = sorted({architecture(c) for c in dl["config_name"].unique().to_list()})
    coverage_rows.append(
        {
            "case_study": SHORT_NAMES[cs],
            "primary_label": primary,
            "dl_archs": ", ".join(archs),
            "n_dl_configs": dl["config_name"].n_unique(),
        }
    )

coverage_df = pl.DataFrame(coverage_rows)
print("DL architecture coverage per case study (primary label):")
coverage_df

# %%
# Architecture × CS availability matrix at the primary label
all_archs_sorted = ["LSTM", "NLinear", "TSMixer", "TCN", "PatchTST"]
avail_rows = []
for cs in CASE_STUDY_IDS:
    dl = dl_grid.filter(pl.col("case_study") == cs)
    if dl.is_empty():
        for arch in all_archs_sorted:
            avail_rows.append({"short_name": SHORT_NAMES[cs], "architecture": arch, "ic": None})
        continue
    by_arch = (
        dl.with_columns(
            architecture=pl.col("config_name").map_elements(architecture, return_dtype=pl.Utf8)
        )
        .group_by("architecture")
        .agg(pl.col("ic_mean_daily").max().alias("ic"))
    )
    arch_to_ic = dict(by_arch.iter_rows())
    for arch in all_archs_sorted:
        avail_rows.append(
            {
                "short_name": SHORT_NAMES[cs],
                "architecture": arch,
                "ic": arch_to_ic.get(arch),
            }
        )

avail_df = pl.DataFrame(avail_rows)
avail_pivot = avail_df.pivot(index="short_name", on="architecture", values="ic").sort("short_name")
print(
    "Highest-IC DL configuration per (case study × architecture) at the primary label "
    "(blank = architecture not trained on this case study):"
)
avail_pivot.select(["short_name", *all_archs_sorted])

# %% [markdown]
# Architecture coverage is computed from full-day, exact-fold candidates. A
# missing family stays missing rather than being filled with a shorter-span or
# stale candidate.

# %%
architecture_coverage = (
    dl_grid.with_columns(
        architecture=pl.col("config_name").map_elements(architecture, return_dtype=pl.Utf8)
    )
    .group_by("architecture")
    .agg(n_case_studies=pl.col("case_study").n_unique())
    .sort("n_case_studies", descending=True)
)
missing_dl = [
    SHORT_NAMES[cs] for cs in CASE_STUDY_IDS if cs not in set(dl_grid["case_study"].to_list())
]
display(
    Markdown(
        "**Computed coverage.** "
        f"{dl_grid['case_study'].n_unique()} of {len(CASE_STUDY_IDS)} case studies have a "
        f"complete DL candidate; missing: {', '.join(missing_dl) or 'none'}."
    )
)
architecture_coverage

# %% [markdown]
# ## 2. Cross-CS Forest of Highest-IC DL Configurations
#
# For each DL-covered case study, the architecture and configuration with
# the highest average daily IC on the primary label is plotted with its HAC
# 95 % CI. Filled markers indicate $|t_{HAC}| > 2$ (CI excludes zero); open
# markers indicate the CI overlaps zero.

# %%
dl_rank1 = collect_rank1_per_cs(
    CASE_STUDY_IDS,
    family=FAMILY,
)
dl_rank1_display = dl_rank1.with_columns(
    architecture=pl.col("config_name").map_elements(architecture, return_dtype=pl.Utf8),
)
print("Highest-IC DL configuration per case study (primary label, average daily IC ± HAC 95 % CI):")
dl_rank1_display.select(
    "short_name",
    "label",
    "architecture",
    "config_name",
    pl.col("ic_mean_daily").round(4).alias("ic"),
    pl.col("ic_ci_lo").round(4).alias("ci_lo"),
    pl.col("ic_ci_hi").round(4).alias("ci_hi"),
    pl.col("ic_t_hac").round(2).alias("t_hac"),
    pl.col("ic_n_days").cast(pl.Int64).alias("n_days"),
)

# %%
fig, forest_ax = plot_cross_cs_forest(
    dl_rank1,
    family=FAMILY,
    title="Highest-IC DL per case study (primary label, average daily IC ± HAC 95 % CI)",
)
forest_ax.set_xlabel("Average daily IC (HAC 95 % CI)")
fig.show()

# %%
n_dl_total = dl_rank1.height
n_sig = int((dl_rank1["ic_t_hac"].abs() > 2.0).sum())
print(f"DL CI excludes zero ({{|t_HAC|>2}}) on {n_sig} of {n_dl_total} DL-covered case studies.")

# %%
dl_clear_names = dl_rank1.filter(pl.col("ic_t_hac").abs() > 2)["short_name"].to_list()
dl_overlap_names = dl_rank1.filter(pl.col("ic_t_hac").abs() <= 2)["short_name"].to_list()
display(
    Markdown(
        f"**Computed DL inference.** The HAC interval excludes zero for {len(dl_clear_names)} "
        f"of {dl_rank1.height} selected rows ({', '.join(dl_clear_names) or 'none'}) and "
        f"overlaps zero for {', '.join(dl_overlap_names) or 'none'}."
    )
)

# %% [markdown]
# ## 3. Within-Family Comparison
#
# Three subsections trace the architecture grid: the highest IC by architecture
# inside each case study (3a), the aggregate count of which architecture
# achieves the highest IC (3b), and the per-checkpoint IC trajectory of the
# highest-IC configuration on each case study (3c).

# %% [markdown]
# ### 3a. Architecture × case-study heatmap
#
# Within each case study, the highest IC achieved by each architecture is
# shown as a heatmap cell. Cells are blank where the architecture was not
# trained on that case study - coverage gaps remain visible.

# %%
arch_cols = all_archs_sorted
cs_labels = avail_pivot["short_name"].to_list()
matrix_values = avail_pivot.select(arch_cols).to_numpy()

fig, ax = plt.subplots(figsize=(7.5, 4.5))
finite_vals = matrix_values[np.isfinite(matrix_values.astype(float))]
vmax = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 0.05
masked = np.ma.array(matrix_values.astype(float), mask=~np.isfinite(matrix_values.astype(float)))
cmap = LinearSegmentedColormap.from_list("ml4t_diverging", ml4t_diverging())
cmap.set_bad(color=COLORS["silver_muted"])
im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(np.arange(len(arch_cols)))
ax.set_xticklabels(arch_cols)
ax.set_yticks(np.arange(len(cs_labels)))
ax.set_yticklabels(cs_labels)
for i in range(len(cs_labels)):
    for j in range(len(arch_cols)):
        v = matrix_values[i, j]
        if v is not None and np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=COLORS["silver"] if abs(v) > 0.6 * vmax else COLORS["neutral"],
            )
        else:
            ax.text(j, i, "-", ha="center", va="center", fontsize=8, color=COLORS["silver_muted"])
ax.set_title("Highest-IC DL configuration per (case study × architecture)")
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="Average daily IC")
fig.show()

# %%
winner_text = ", ".join(
    f"{row['short_name']}: {row['architecture']}"
    for row in dl_rank1_display.sort("short_name").iter_rows(named=True)
)
display(
    Markdown(
        f"**Computed architecture leaders.** {winner_text}. Blank heatmap cells remain explicit "
        "coverage gaps and never enter a winner count."
    )
)

# %% [markdown]
# ### 3b. Aggregate: which architecture achieves the highest IC most often
#
# Counting across DL-covered case studies, how often does each architecture
# achieve the highest IC at the primary label? The bar is annotated with the
# mean IC of the case studies where the architecture achieves the highest IC
# (the table above also reports the min/max across those case studies) -
# magnitudes matter as much as counts.

# %%
arch_top_counts = (
    dl_rank1_display.group_by("architecture")
    .agg(
        n_cs_with_highest_ic=pl.col("case_study").len(),
        ic_mean=pl.col("ic_mean_daily").mean(),
        ic_min=pl.col("ic_mean_daily").min(),
        ic_max=pl.col("ic_mean_daily").max(),
    )
    .sort("n_cs_with_highest_ic", descending=True)
)
print("Architecture achieving the highest IC at the primary label (count across DL-covered CSs):")
arch_top_counts

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.bar(
    arch_top_counts["architecture"].to_list(),
    arch_top_counts["n_cs_with_highest_ic"].to_list(),
    color=COLORS["blue"],
)
ax.set_xlabel("Architecture")
ax.set_ylabel("Winning case studies")
ax.set_title("DL architectures achieving the highest IC per case study (count)")
ax.set_ylim(0, float(arch_top_counts["n_cs_with_highest_ic"].max()) + 0.6)
for i, (n, ic) in enumerate(
    zip(
        arch_top_counts["n_cs_with_highest_ic"].to_list(),
        arch_top_counts["ic_mean"].to_list(),
    )
):
    ax.text(i, n + 0.1, f"IC={ic:+.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.show()

# %%
architecture_count_text = ", ".join(
    f"{row['architecture']}: {row['n_cs_with_highest_ic']}"
    for row in arch_top_counts.iter_rows(named=True)
)
display(
    Markdown(
        f"**Computed architecture counts.** {architecture_count_text}. Counts describe the "
        "point-estimate leaders; they do not establish superiority across panels."
    )
)

# %% [markdown]
# ### 3c. Checkpoint dynamics
#
# For each case study's highest-IC DL configuration, the per-checkpoint
# (epoch) IC trajectory is plotted with the IQR band across folds. Peaked
# curves identify clear early-stopping value; flat curves indicate the
# optimization landscape is benign across the budget.

# %%
checkpoint_folds = collect_checkpoint_fold_trajectories(dl_rank1)
ckpt_df = (
    checkpoint_folds.group_by(["short_name", "config_name", "checkpoint_value"])
    .agg(
        ic_median=pl.col("ic").median(),
        ic_q25=pl.col("ic").quantile(0.25),
        ic_q75=pl.col("ic").quantile(0.75),
    )
    .filter(pl.col("checkpoint_value").is_not_null())
    .with_columns(
        architecture=pl.col("config_name").map_elements(architecture, return_dtype=pl.Utf8),
    )
    .sort(["short_name", "checkpoint_value"])
)

# %%
if not ckpt_df.is_empty():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = ml4t_palette(5, categorical=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    line_styles = ["-", "--", "-.", ":"]
    cs_sorted = sorted(ckpt_df["short_name"].unique().to_list())
    for i, cs in enumerate(cs_sorted):
        sub = ckpt_df.filter(pl.col("short_name") == cs).sort("checkpoint_value")
        x = sub["checkpoint_value"].to_numpy()
        med = sub["ic_median"].to_numpy()
        q25 = sub["ic_q25"].to_numpy()
        q75 = sub["ic_q75"].to_numpy()
        color = palette[i % len(palette)]
        arch = sub["architecture"].to_list()[0]
        ax.fill_between(x, q25, q75, color=color, alpha=0.10)
        ax.plot(
            x,
            med,
            color=color,
            marker=markers[i],
            linestyle=line_styles[i % len(line_styles)],
            label=f"{cs} ({arch})",
            linewidth=1.6,
            markersize=4,
            alpha=0.9,
        )
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
    ax.set_xlabel("Training epoch (checkpoint)")
    ax.set_ylabel("Per-fold IC median (IQR band)")
    ax.set_title("Checkpoint dynamics for the highest-IC DL configuration per case study")
    ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.show()
else:
    print("No DL checkpoint data available.")

# %%
trajectory_peaks = (
    ckpt_df.sort("ic_median", descending=True)
    .group_by("short_name")
    .first()
    .select("short_name", "architecture", "checkpoint_value", "ic_median")
    .sort("checkpoint_value")
)
peak_text = ", ".join(
    f"{row['short_name']}: {int(row['checkpoint_value'])}"
    for row in trajectory_peaks.iter_rows(named=True)
)
display(Markdown(f"**Computed checkpoint peaks.** Selected median-IC peaks occur at {peak_text}."))

# %% [markdown]
# ## 4. Stability and Uncertainty
#
# Average daily IC with HAC CI is the headline metric. Per-fold IC is the
# stability diagnostic; conformal coverage is the calibration diagnostic.

# %% [markdown]
# ### 4a. Per-fold IC distribution
#
# For each DL-covered case study, the per-fold IC distribution of the highest-
# IC configuration is shown as a box-plus-scatter, sorted left-to-right by
# headline IC.

# %%
dl_fold = collect_fold_ic_per_cs(dl_rank1)
dl_fold_summary = (
    dl_fold.group_by(["case_study", "short_name"])
    .agg(
        n_folds=pl.col("ic").count(),
        median=pl.col("ic").median(),
        std=pl.col("ic").std(),
        pct_positive=(pl.col("ic") > 0).mean(),
    )
    .sort("median", descending=True)
)
print("Per-fold IC summary for the highest-IC DL configuration (primary label):")
dl_fold_summary

# %%
order = dl_rank1.sort("ic_mean_daily", descending=True)["short_name"].to_list()
fig, _ = plot_per_fold_violin(
    dl_fold,
    order=order,
    title="Per-fold IC distribution for the highest-IC DL configuration (primary label)",
)
fig.show()

# %% [markdown]
# Fold summaries diagnose stability only. Inference remains attached to the
# chronological daily IC series, not to the small collection of fold summaries.

# %%
positive_majority = dl_fold_summary.filter(pl.col("pct_positive") > 0.5)["short_name"].to_list()
display(
    Markdown(
        f"**Computed fold diagnostic.** {len(positive_majority)} of "
        f"{dl_fold_summary.height} selected DL rows have a positive-fold majority: "
        f"{', '.join(positive_majority) or 'none'}."
    )
)

# %% [markdown]
# ### 4b. Cross-fitted OOF fold calibration at the 90 % nominal level
#
# The oldest out-of-fold validation window calibrates a symmetric
# absolute-residual quantile and the return scale. The diagnostic measures
# empirical coverage on the later OOF windows at each nominal level. A
# well-calibrated DL signal sits near the diagonal
# (empirical coverage = nominal level); points below the diagonal indicate
# the prediction interval is too narrow, points above indicate it is too
# wide. Per-CS interval width is reported in calibration-window return
# standard-deviation units. Because each OOF fold can come from a different
# fitted model, this is a retrospective cross-fitted diagnostic, not a
# same-model inductive split-conformal or operational coverage guarantee.

# %%
conformal_rows = []
conformal_unavailable = []
for selected in dl_rank1.iter_rows(named=True):
    cs = selected["case_study"]
    try:
        df = conformal_coverage_for_selected_prediction(selected)
    except RegistrySelectionError as error:
        if "fewer than 30 rows" not in str(error):
            raise
        conformal_unavailable.append(SHORT_NAMES[cs])
        continue
    sub = df.filter(pl.col("nominal_level") == CONFORMAL_LEVEL)
    if sub.is_empty():
        continue
    r = sub.row(0, named=True)
    conformal_rows.append(
        {
            "short_name": SHORT_NAMES[cs],
            "config_name": r["config_name"],
            "architecture": architecture(r["config_name"]),
            "nominal_level": r["nominal_level"],
            "empirical_coverage": r["empirical_coverage"],
            "interval_width_frac_std": r["mean_interval_width_frac_std"],
            "n_test": r["n_test"],
        }
    )

conformal_df = pl.DataFrame(conformal_rows).sort("short_name")
print(
    f"Cross-fitted OOF calibration at the {CONFORMAL_LEVEL:.0%} nominal level; "
    f"insufficient rows: {', '.join(conformal_unavailable) or 'none'}"
)
conformal_df.select(
    "short_name",
    "architecture",
    pl.col("empirical_coverage").round(3).alias("empirical_cov"),
    pl.col("interval_width_frac_std").round(3).alias("width_frac_std"),
    "n_test",
)


# %%
def staggered_offsets(values: np.ndarray, tolerance: float = 0.04) -> list[int]:
    offsets = [4] * len(values)
    last_value, last_offset = -np.inf, 4
    for index in sorted(range(len(values)), key=lambda i: values[i]):
        offsets[index] = last_offset + 12 if values[index] - last_value < tolerance else 4
        last_value, last_offset = values[index], offsets[index]
    return offsets


# %% [markdown]
# The complete calibration chart is assembled in one rendering cell. The
# horizontal reference marks the nominal target, not a performance threshold.


# %%
def plot_conformal_coverage(conformal_df: pl.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    emp = conformal_df["empirical_coverage"].to_numpy()
    width = conformal_df["interval_width_frac_std"].to_numpy()
    names = conformal_df["short_name"].to_list()
    gap = np.abs(emp - CONFORMAL_LEVEL)
    cmap = LinearSegmentedColormap.from_list(
        "ml4t_calibration_gap", [COLORS["silver_muted"], COLORS["amber"]]
    )
    sc = ax.scatter(
        width,
        emp,
        s=90,
        edgecolor=COLORS["silver"],
        c=gap,
        cmap=cmap,
        vmin=0.0,
        vmax=max(0.01, float(gap.max())),
    )
    offsets = staggered_offsets(emp)
    for i, n in enumerate(names):
        ax.annotate(n, (width[i], emp[i]), textcoords="offset points", xytext=(7, offsets[i]))
    ax.axhline(
        CONFORMAL_LEVEL,
        color=COLORS["neutral"],
        linestyle="--",
        label=f"Nominal {CONFORMAL_LEVEL:.0%}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Mean interval width (fraction of calibration-fold return std; log scale)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Cross-fitted OOF calibration exposes scale drift at {CONFORMAL_LEVEL:.0%}")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04, label="Absolute coverage gap")
    return fig


# %%
if not conformal_df.is_empty():
    fig = plot_conformal_coverage(conformal_df)
    fig.show()

# %%
if not conformal_df.is_empty():
    conformal_gap = conformal_df.with_columns(
        gap=(pl.col("empirical_coverage") - pl.col("nominal_level")).abs()
    ).sort("gap", descending=True)
    furthest = conformal_gap.row(0, named=True)
    nearest = conformal_gap.sort("gap").row(0, named=True)
    display(
        Markdown(
            f"**Computed conformal diagnostic.** The nearest panel to nominal is "
            f"{nearest['short_name']} ({nearest['empirical_coverage']:.3f}); the furthest is "
            f"{furthest['short_name']} ({furthest['empirical_coverage']:.3f}). The oldest OOF "
            "fold supplies calibration state and later OOF folds come from different fits, so "
            "this diagnostic is not a same-model conformal or operational coverage guarantee."
        )
    )

# %% [markdown]
# ## 5. DL versus Tabular Baselines
#
# The DL family enters a contested space - for every case study it is
# compared to the highest-IC tabular configuration across linear, GBM, and
# TabM. The selected models are rescored over the exact intersection of their
# evaluation timestamps. The notebook does not infer uncertainty from a small set of
# fold summaries.


# %%
def family_rank1_collect(family: str) -> pl.DataFrame:
    df = collect_rank1_per_cs(
        CASE_STUDY_IDS,
        family=family,
    )
    if df.is_empty():
        return pl.DataFrame()
    return df.select(
        "case_study",
        "short_name",
        "config_name",
        "prediction_hash",
        "training_hash",
        "ic_n_days",
        pl.col("ic_mean_daily").alias(f"{family}_ic"),
    )


# %% [markdown]
# Apply the same complete-coverage selector to each tabular family before
# choosing the strongest point estimate within a case study.

# %%
linear_rank1 = family_rank1_collect("linear")
gbm_rank1 = family_rank1_collect("gbm")
tabm_rank1 = family_rank1_collect("tabular_dl")

# %%
ic_lookup = {
    "linear": linear_rank1,
    "gbm": gbm_rank1,
    "tabular_dl": tabm_rank1,
}


def dl_tabular_delta(cs: str) -> dict | None:
    """Compare the selected DL row with the strongest tabular family row."""
    dl_row = dl_rank1.filter(pl.col("case_study") == cs)
    if dl_row.is_empty():
        return None
    selected_dl = dl_row.row(0, named=True)
    family_rows = {}
    for fam in TABULAR_BASELINES:
        sub = ic_lookup[fam].filter(pl.col("case_study") == cs)
        if not sub.is_empty():
            family_rows[fam] = sub.row(0, named=True)
    if not family_rows:
        return None
    best_fam = max(family_rows, key=lambda family: family_rows[family][f"{family}_ic"])
    best_row = family_rows[best_fam]
    dl_daily = load_daily_metrics_series(cs, selected_dl["prediction_hash"])
    baseline_daily = load_daily_metrics_series(cs, best_row["prediction_hash"])
    if dl_daily.is_empty() or baseline_daily.is_empty():
        return None
    matched = compare_ic_on_shared_timestamps(dl_daily, baseline_daily)
    return {
        "case_study": cs,
        "short_name": SHORT_NAMES[cs],
        "dl_arch": architecture(selected_dl["config_name"]),
        "dl_ic": matched["left_ic"],
        "dl_prediction_hash": selected_dl["prediction_hash"],
        "best_baseline_family": best_fam,
        "best_baseline_ic": matched["right_ic"],
        "baseline_prediction_hash": best_row["prediction_hash"],
        "matched_timestamps": matched["n_timestamps"],
        "delta": matched["left_ic"] - matched["right_ic"],
    }


# %% [markdown]
# The comparison selects complete-coverage rows from each family, then
# computes both point estimates from their shared evaluation timestamps. The
# table does not estimate uncertainty for the difference between models.

# %%
delta_rows = [entry for cs in CASE_STUDY_IDS if (entry := dl_tabular_delta(cs)) is not None]

delta_df = pl.DataFrame(delta_rows).sort("delta", descending=True)
print("DL minus highest-IC full-coverage tabular baseline on shared timestamps (primary label):")
delta_df.select(
    "short_name",
    "dl_arch",
    "best_baseline_family",
    pl.col("dl_ic").round(4).alias("dl"),
    pl.col("best_baseline_ic").round(4).alias("base"),
    pl.col("delta").round(4),
    "matched_timestamps",
    "dl_prediction_hash",
    "baseline_prediction_hash",
)

# %% [markdown]
# ### 5a. DL versus highest-IC tabular baseline - scatter
#
# The scatter plots the highest-IC tabular baseline on the x-axis and the
# highest-IC DL configuration on the y-axis, one point per case study.
# Points above the diagonal indicate DL exceeds the highest-IC tabular
# baseline; points below indicate the tabular baseline tops the DL signal.
# Marker shape identifies the selected tabular family; color shows only the
# direction of the point-estimate difference.

# %%
fam_marker = {"linear": "o", "gbm": "s", "tabular_dl": "D"}


def add_scatter_points(ax: plt.Axes, delta_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xs = delta_df["best_baseline_ic"].to_numpy()
    ys = delta_df["dl_ic"].to_numpy()
    rows = zip(
        xs,
        ys,
        delta_df["short_name"],
        delta_df["dl_arch"],
        delta_df["best_baseline_family"],
        strict=False,
    )
    for x, y, name, arch, family in rows:
        ax.scatter(
            x,
            y,
            s=90,
            marker=fam_marker[family],
            facecolor=COLORS["positive"] if y > x else COLORS["negative"],
            edgecolor=COLORS["blue"],
            linewidth=1.4,
        )
        ax.annotate(f"{name} ({arch})", (x, y), textcoords="offset points", xytext=(8, 4))
    return xs, ys


# %% [markdown]
# A common scale and equality line make the point-estimate comparison
# interpretable without implying paired uncertainty.


# %%
def format_scatter_axes(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    lo = float(min(np.min(xs), np.min(ys))) - 0.005
    hi = float(max(np.max(xs), np.max(ys))) + 0.005
    ax.plot([lo, hi], [lo, hi], color=COLORS["neutral"], linestyle="--")
    ax.axhline(0, color=COLORS["silver_muted"], linewidth=0.5)
    ax.axvline(0, color=COLORS["silver_muted"], linewidth=0.5)
    ax.set_xlabel("Selected tabular baseline IC on shared timestamps")
    ax.set_ylabel("Selected DL IC on shared timestamps")
    ax.set_title("DL vs highest-IC tabular baseline on shared timestamps")


# %% [markdown]
# Shape identifies the selected tabular family; fill identifies only the
# direction of the observed difference.


# %%
def scatter_legend_elements() -> list[Line2D]:
    family_labels = {"linear": "Linear", "gbm": "GBM", "tabular_dl": "TabM"}
    elements = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color=COLORS["silver"],
            markerfacecolor=COLORS["silver_muted"],
            markeredgecolor=COLORS["blue"],
            markersize=9,
            label=f"Highest-IC tabular = {family_labels[family]}",
        )
        for family, marker in fam_marker.items()
    ]
    for color, label in [
        (COLORS["positive"], "DL point estimate > tabular"),
        (COLORS["negative"], "DL point estimate < tabular"),
    ]:
        elements.append(Line2D([0], [0], marker="o", color=color, label=label))
    elements.append(Line2D([0], [0], color=COLORS["neutral"], linestyle="--", label="DL = tabular"))
    return elements


# %% [markdown]
# The figure is created only after all drawing helpers are defined.

# %%
fig, ax = plt.subplots(figsize=(8.5, 6.5))
xs, ys = add_scatter_points(ax, delta_df)
format_scatter_axes(ax, xs, ys)
ax.legend(handles=scatter_legend_elements(), loc="upper left", frameon=False, fontsize=8)
fig.tight_layout()
fig.show()

# %%
n_above = int((ys > xs).sum())
print(
    f"DL > tabular point estimate on {n_above}/{len(ys)} case studies. "
    "No paired-fold inference is attached to this difference."
)

# %% [markdown]
# ### 5b. When does DL help? Diagnostic table
#
# The same delta cross-referenced with panel-level features: data frequency,
# universe size (entities), and the highest-IC tabular family. The rows are
# sorted by the DL-minus-tabular point-estimate delta, largest first.

# %%
diagnostic_rows = []
for r in delta_df.iter_rows(named=True):
    cs = r["case_study"]
    meta = DATASET_META.get(cs, {})
    diagnostic_rows.append(
        {
            "short_name": r["short_name"],
            "frequency": meta.get("frequency"),
            "entities": meta.get("entities"),
            "best_tabular": r["best_baseline_family"],
            "tab_ic": r["best_baseline_ic"],
            "dl_arch": r["dl_arch"],
            "dl_ic": r["dl_ic"],
            "delta": r["delta"],
            "matched_timestamps": r["matched_timestamps"],
        }
    )

diagnostic_df = pl.DataFrame(diagnostic_rows)
print("DL-help diagnostic table (sorted by point-estimate delta, largest first):")
diagnostic_df.select(
    "short_name",
    "frequency",
    "entities",
    "best_tabular",
    "dl_arch",
    pl.col("tab_ic").round(4).alias("tab_ic"),
    pl.col("dl_ic").round(4).alias("dl_ic"),
    pl.col("delta").round(4).alias("delta"),
    "matched_timestamps",
)

# %%
display(
    Markdown(
        f"**Computed baseline comparison.** DL has the higher shared-timestamp point estimate "
        f"in {n_above} of {delta_df.height} comparable case studies. The table does not test "
        "whether these differences are statistically distinguishable."
    )
)

# %% [markdown]
# ## 6. Multi-Label Horizon View
#
# Only case studies with at least two complete registered regression labels
# enter the horizon figure. Single-point series are excluded rather than
# padded.


# %%
def regression_labels(cs: str) -> list[str]:
    df = load_metrics_from_registry(cs, families=[FAMILY])
    if df.is_empty():
        return []
    return [
        lbl
        for lbl in df["label"].unique().to_list()
        if lbl is not None and (lbl.startswith("fwd_ret_") or lbl == "ret_to_expiry")
    ]


# %% [markdown]
# Only complete registered labels enter the horizon census and figure.

# %%
dl_horizon = collect_multi_label_per_cs(
    CASE_STUDY_IDS,
    family=FAMILY,
    labels=regression_labels,
)
print(
    f"DL multi-label horizon coverage: {dl_horizon.height} (CS, label) cells "
    f"across {dl_horizon['case_study'].n_unique() if not dl_horizon.is_empty() else 0} case studies."
)

multi_horizon_cs = (
    dl_horizon.group_by("short_name").len().filter(pl.col("len") >= 2)["short_name"].to_list()
)
if multi_horizon_cs:
    fig, horizon_ax = plot_multi_label_horizon(
        dl_horizon,
        title="Highest-IC DL configuration across regression horizons",
    )
    horizon_ax.set_ylabel("Average daily IC (HAC 95 % CI band)")
    fig.show()
else:
    print(
        "No case study has ≥2 DL labels - horizon comparison is degenerate at the registry "
        "snapshot used here."
    )

# %%
display(
    Markdown(
        f"**Computed horizon coverage.** {len(multi_horizon_cs)} case studies have at least "
        f"two complete DL horizons ({', '.join(multi_horizon_cs) or 'none'}). Cross-panel horizon "
        "claims remain out of scope when the remaining panels have only one trained label."
    )
)

# %% [markdown]
# ## 7. Architectural Patterns
#
# The five architectures map to four classes - recurrent (LSTM), MLP-style
# (NLinear, TSMixer), convolutional (TCN), attention (PatchTST). For each
# class, we read the highest IC achieved on each case study where the class
# has at least one trained configuration.


# %%
def architecture_class_rows(cs: str) -> list[dict]:
    """Return highest-IC complete row for each trained architecture class."""
    df = dl_grid.filter(pl.col("case_study") == cs)
    if df.is_empty():
        return []
    df = (
        df.with_columns(
            architecture=pl.col("config_name").map_elements(architecture, return_dtype=pl.Utf8),
        )
        .with_columns(
            arch_class=pl.col("architecture").replace_strict(ARCH_CLASS, default=None),
        )
        .filter(pl.col("ic_mean_daily").is_not_null() & pl.col("arch_class").is_not_null())
    )
    if df.is_empty():
        return []
    by_class = (
        df.sort("ic_mean_daily", descending=True)
        .group_by("arch_class")
        .first()
        .select("arch_class", "ic_mean_daily", "ic_ci_lo", "ic_ci_hi", "ic_t_hac", "config_name")
    )
    return [{"short_name": SHORT_NAMES[cs], **r} for r in by_class.iter_rows(named=True)]


# %% [markdown]
# Missing architecture classes remain absent rather than being imputed.

# %%
class_rows = [row for cs in CASE_STUDY_IDS for row in architecture_class_rows(cs)]
class_df = pl.DataFrame(class_rows)
print("Highest IC per (case study × architecture class):")
class_df.sort(["short_name", "arch_class"]).select(
    "short_name",
    "arch_class",
    "config_name",
    pl.col("ic_mean_daily").round(4).alias("ic"),
    pl.col("ic_t_hac").round(2).alias("t_hac"),
)

# %%
arch_classes = ["recurrent", "MLP-style", "convolutional", "attention"]
class_colors = {
    "recurrent": COLORS["blue"],
    "MLP-style": COLORS["amber"],
    "convolutional": COLORS["copper"],
    "attention": COLORS["positive"],
}

cs_order_class = sorted(class_df["short_name"].unique().to_list())
x = np.arange(len(cs_order_class))
width = 0.20


def add_architecture_bars(ax: plt.Axes) -> None:
    for i, cls in enumerate(arch_classes):
        sub = class_df.filter(pl.col("arch_class") == cls)
        ic, err_lo, err_hi = [], [], []
        for cs in cs_order_class:
            row = sub.filter(pl.col("short_name") == cs)
            if row.height == 0:
                ic.append(np.nan)
                err_lo.append(0.0)
                err_hi.append(0.0)
            else:
                r = row.row(0, named=True)
                ic.append(r["ic_mean_daily"])
                err_lo.append(r["ic_mean_daily"] - r["ic_ci_lo"])
                err_hi.append(r["ic_ci_hi"] - r["ic_mean_daily"])
        ax.bar(
            x + (i - 1.5) * width,
            np.array(ic, dtype=float),
            width=width,
            yerr=np.vstack([err_lo, err_hi]),
            capsize=2,
            color=class_colors[cls],
            label=cls,
        )


# %% [markdown]
# The HAC intervals belong to each selected daily IC series; they are not
# uncertainty estimates for the difference between architecture classes.

# %%
fig, ax = plt.subplots(figsize=(11, 5))
add_architecture_bars(ax)
ax.set_xticks(x)
ax.set_xticklabels(cs_order_class, rotation=35, ha="right")
ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
ax.set_ylabel("Average daily IC (HAC 95 % CI)")
ax.set_title("Highest IC per architectural class per case study")
ax.legend(frameon=False, fontsize=9, loc="best", ncol=4)
fig.tight_layout()
fig.show()

# %%
class_top_per_cs = (
    class_df.sort("ic_mean_daily", descending=True)
    .group_by("short_name")
    .first()
    .group_by("arch_class")
    .agg(n_cs_with_highest_ic=pl.col("short_name").len())
    .sort("n_cs_with_highest_ic", descending=True)
)
print("Architectural class achieving the highest IC per case study (count):")
class_top_per_cs

# %%
class_count_text = ", ".join(
    f"{row['arch_class']}: {row['n_cs_with_highest_ic']}"
    for row in class_top_per_cs.iter_rows(named=True)
)
display(
    Markdown(
        f"**Computed architecture-class counts.** {class_count_text}. These are "
        "point-estimate counts, not cross-case superiority estimates."
    )
)

# %% [markdown]
# ## Cross-CS Key Takeaways
#
# The synthesis below is computed from the selected prediction hashes and their
# complete fold panels.

# %%
largest_dl_delta = delta_df.row(0, named=True)
display(
    Markdown(
        "**Key takeaways**\n\n"
        f"- {dl_rank1.height} case studies have a complete primary-label DL candidate; "
        f"{len(dl_clear_names)} have HAC intervals that exclude zero.\n"
        f"- DL has the higher full-coverage point estimate in {n_above} of "
        f"{delta_df.height} comparisons with the strongest tabular family.\n"
        f"- The largest DL-minus-tabular point estimate is {largest_dl_delta['short_name']} "
        f"at {largest_dl_delta['delta']:+.4f}; the comparison remains descriptive without a "
        "registered daily paired-difference estimator.\n\n"
        "**Next**: Ch14 adds latent-factor models on panels that satisfy the dimensionality "
        "gate; Ch15 layers causal effects on top of the predictive stack."
    )
)
