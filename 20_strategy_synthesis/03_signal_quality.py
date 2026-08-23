# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Signal Evidence: Is the Chosen Signal Trustworthy?
#
# **Docker image**: `ml4t`
#
# This is NOT a full recap of Ch11-15's model family narrative. It shows
# **only** the signal evidence that matters for downstream trading decisions:
# validation IC, holdout IC, stability, reproducibility flags, and whether
# the chosen signal is trustworthy enough to trade on.
#
# **Scope rule**: Ch11-15 insight notebooks teach what each model family
# reveals. This notebook asks whether the signal each case study selected holds
# up when its validation and holdout evidence are put side by side.
#
# **Learning Objectives**:
# - Assess which case studies have trustworthy signal evidence
# - Compare validation IC with holdout IC (signal persistence)
# - Identify reproducibility and stability concerns
#
# **Book Reference**: Chapter 20, Section 20.2 (Signal Evidence)
#
# **Prerequisites**: Run [`01_aggregate_synthesis`](01_aggregate_synthesis.ipynb) first.

# %%
"""Ch20 Signal Evidence — only the signal evidence that matters for trading decisions."""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from IPython.display import Markdown, display

warnings.filterwarnings("ignore")

from utils.paths import get_chapter_dir
from utils.style import show_with_alt

# %% tags=["parameters"]
MAX_SYMBOLS = 0

# %%
OUTPUT_DIR = get_chapter_dir(20) / "output"

# Display names
DISPLAY_NAMES = {
    "etfs": "ETFs",
    "crypto_perps_funding": "Crypto",
    "nasdaq100_microstructure": "NASDAQ-100",
    "sp500_equity_option_analytics": "S&P 500 Eq+Opt",
    "us_firm_characteristics": "US Firms",
    "fx_pairs": "FX Pairs",
    "cme_futures": "CME Futures",
    "sp500_options": "S&P 500 Options",
    "us_equities_panel": "US Equities",
}

FAMILY_NAMES = {
    "linear": "Linear",
    "gbm": "GBM",
    "deep_learning": "Deep Learning",
    "tabular_dl": "Tabular DL",
    "latent_factors": "Latent Factors",
}

# %% [markdown]
# ## Load Data

# %%
ic_df = pl.read_parquet(OUTPUT_DIR / "ic_comparison.parquet")
variant_df = pl.read_parquet(OUTPUT_DIR / "variant_analysis.parquet")
synthesis = json.load((OUTPUT_DIR / "all_synthesis.json").open())

print(f"IC comparison: {len(ic_df)} entries across {ic_df['case_study'].n_unique()} case studies")
print(
    f"Variant analysis: {len(variant_df)} variants across {variant_df['case_study'].n_unique()} case studies"
)

# %% [markdown]
# ## IC Landscape Heatmap
#
# Which model families produce the strongest signals for which asset classes?
#
# Each cell is the mean information coefficient over every prediction that family
# registered for that case study, on the case study's primary label and excluding
# the holdout split. It is an average over a whole hyperparameter sweep, so it
# measures how the family does typically, not how well its best configuration can
# do. The maximum over the same sweep is the `ic_best` column, used further down;
# the two answer different questions and can rank the families differently.

# %%
# Build IC matrix: case study × model family
ic_matrix = ic_df.pivot(on="family", index="case_study", values="ic_mean")

# Order case studies by average IC
case_order = (
    ic_df.group_by("case_study")
    .agg(avg_ic=pl.col("ic_mean").mean())
    .sort("avg_ic", descending=True)["case_study"]
    .to_list()
)

family_order = ["linear", "gbm", "tabular_dl", "deep_learning", "latent_factors"]
family_labels = [FAMILY_NAMES.get(f, f) for f in family_order]

# Build matrix for heatmap
matrix_data = []
for cs in case_order:
    row = []
    for fam in family_order:
        val = ic_matrix.filter(pl.col("case_study") == cs).select(fam)
        if len(val) > 0 and val[0, 0] is not None:
            row.append(float(val[0, 0]))
        else:
            row.append(np.nan)
    matrix_data.append(row)

matrix_arr = np.array(matrix_data)

fig, ax = plt.subplots(figsize=(10, 7))
# RdYlGn is diverging, so zero has to be the midpoint. Left to autoscale, a
# range that happens to sit above zero paints a weak positive IC red and a
# range below it paints a negative IC green.
ic_extent = float(np.nanmax(np.abs(matrix_arr))) if np.isfinite(matrix_arr).any() else 1.0
im = ax.imshow(matrix_arr, cmap="RdYlGn", aspect="auto", vmin=-ic_extent, vmax=ic_extent)

ax.set_xticks(range(len(family_labels)))
ax.set_xticklabels(family_labels, rotation=45, ha="right")
ax.set_yticks(range(len(case_order)))
ax.set_yticklabels(case_order)

# Annotate cells
for i in range(len(case_order)):
    for j in range(len(family_order)):
        val = matrix_arr[i, j]
        if not np.isnan(val):
            color = "white" if abs(val) > 0.75 * ic_extent else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, ax=ax, label="Mean IC across the family's predictions", shrink=0.8)
ax.set_title("Mean IC by case study and model family")
fig.subplots_adjust(left=0.2, right=0.92, top=0.9, bottom=0.18)
show_with_alt(
    fig,
    "Heatmap of mean information coefficient with case studies as rows and model "
    "families as columns, each cell annotated with its value and coloured on a "
    "red-to-green scale centred on zero. Blank cells are families that case "
    "study did not run.",
)

# %% [markdown]
# ## Signal Sharpe Heatmap
#
# The IC heatmap above measures *prediction quality*. This companion heatmap
# measures *signal quality* — the median Sharpe when predictions from each
# family are converted into positions. The two rankings often disagree:
# a family can lead on average IC but not on median signal Sharpe, or
# vice versa. This is the first concrete view of the IC-to-Sharpe disconnect
# that Section 20.3 explores in detail.

# %%
sharpe_matrix = (
    variant_df.filter(pl.col("sharpe").is_not_null())
    .group_by("case_study", "family")
    .agg(median_sharpe=pl.col("sharpe").median())
    .pivot(on="family", index="case_study", values="median_sharpe")
)

sharpe_data = []
sharpe_rows = []
for cs in case_order:
    sub = sharpe_matrix.filter(pl.col("case_study") == cs)
    if sub.is_empty():
        continue
    row = []
    for fam in family_order:
        if fam in sub.columns:
            v = sub[fam].item()
            row.append(float(v) if v is not None else np.nan)
        else:
            row.append(np.nan)
    sharpe_data.append(row)
    sharpe_rows.append(cs)

sharpe_arr = np.array(sharpe_data)

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(sharpe_arr, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)

ax.set_xticks(range(len(family_labels)))
ax.set_xticklabels(family_labels, rotation=45, ha="right")
ax.set_yticks(range(len(sharpe_rows)))
ax.set_yticklabels(sharpe_rows)

for i in range(len(sharpe_rows)):
    for j in range(len(family_order)):
        val = sharpe_arr[i, j]
        if not np.isnan(val):
            color = "white" if abs(val) > 1.2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, ax=ax, label="Median signal-stage Sharpe", shrink=0.8)
ax.set_title("Median signal-stage Sharpe by case study and model family")
fig.subplots_adjust(left=0.2, right=0.92, top=0.9, bottom=0.18)
show_with_alt(
    fig,
    "Heatmap of the median signal-stage Sharpe with case studies as rows and "
    "model families as columns, annotated per cell and coloured red to green on "
    "a scale from -2 to +2 centred on zero.",
)

# %% [markdown]
# Reading the two heatmaps together separates ranking quality, which is what the
# Sharpe of a long-short book made from the predictions measures, from average
# prediction quality, which is what IC measures. A family can be accurate on
# average and still rank the extremes badly, or the reverse.
#
# The dispersion within a family matters as much as its centre. A family whose
# best configuration sits far above its own median is one where the choice of
# configuration is doing much of the work, and that choice was made on
# validation data.

# %% tags=["results"]
_sp = (
    variant_df.filter(pl.col("sharpe").is_not_null())
    .group_by("family")
    .agg(
        median_sharpe=pl.col("sharpe").median(),
        max_sharpe=pl.col("sharpe").max(),
        n=pl.len(),
    )
    .with_columns(spread=pl.col("max_sharpe") - pl.col("median_sharpe"))
    .sort("spread", descending=True)
)
_w = _sp.row(0, named=True)
_agree = (
    ic_df.filter(pl.col("ic_mean").is_not_null())
    .sort("ic_mean", descending=True)
    .group_by("case_study")
    .first()
    .join(
        variant_df.filter(pl.col("sharpe").is_not_null())
        .group_by("case_study", "family")
        .agg(median_sharpe=pl.col("sharpe").median())
        .sort("median_sharpe", descending=True)
        .group_by("case_study")
        .first()
        .select("case_study", sharpe_leader="family"),
        on="case_study",
        how="inner",
    )
)
_same = _agree.filter(pl.col("family") == pl.col("sharpe_leader")).height
display(
    Markdown(
        f"Widest within-family spread: **{FAMILY_NAMES.get(_w['family'], _w['family'])}**, "
        f"whose best of {_w['n']} configurations reaches Sharpe "
        f"{_w['max_sharpe']:.2f} against a median of {_w['median_sharpe']:.2f}, "
        f"a gap of {_w['spread']:.2f}. Across the {_agree.height} case studies "
        "where both measurements exist, the family with the highest mean IC is "
        f"also the family with the highest median signal Sharpe in {_same} of "
        "them. The two rankings are related but not the same measurement, which "
        "is the disconnect Section 20.3 takes up."
    )
)

# %% [markdown]
# ## Model Family Comparison
#
# Which model family is most consistently effective across asset classes?

# %%
family_stats = (
    ic_df.group_by("family")
    .agg(
        n_studies=pl.col("case_study").n_unique(),
        mean_ic=pl.col("ic_mean").mean(),
        median_ic=pl.col("ic_mean").median(),
        std_ic=pl.col("ic_mean").std(),
        max_ic=pl.col("ic_mean").max(),
        min_ic=pl.col("ic_mean").min(),
        n_positive=(pl.col("ic_mean") > 0).sum(),
    )
    .sort("mean_ic", descending=True)
)

# %% [markdown]
# ### Model Family Summary

# %%
family_stats

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
ic_pandas = ic_df.to_pandas()
ic_pandas["family_label"] = ic_pandas["family"].map(FAMILY_NAMES)
sns.boxplot(
    data=ic_pandas,
    x="family_label",
    y="ic_mean",
    ax=axes[0],
    order=[FAMILY_NAMES[f] for f in family_order if f in ic_pandas["family"].values],
)
axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0].set_xlabel("")
axes[0].set_ylabel("Mean IC")
axes[0].set_title("IC Distribution by Model Family")
axes[0].tick_params(axis="x", rotation=30)

# Bar chart: count of positive ICs
pos_counts = family_stats.with_columns(
    pct_positive=(pl.col("n_positive") / pl.col("n_studies") * 100).round(0),
)
fam_labels = [FAMILY_NAMES.get(f, f) for f in pos_counts["family"].to_list()]
axes[1].barh(fam_labels, pos_counts["pct_positive"].to_list())
axes[1].set_xlabel("% Case Studies with Positive IC")
axes[1].set_title("Model Reliability Across Asset Classes")
axes[1].set_xlim(0, 100)
for i, v in enumerate(pos_counts["pct_positive"].to_list()):
    axes[1].text(v + 1, i, f"{v:.0f}%", va="center")

fig.tight_layout()
show_with_alt(
    fig,
    "Left: box plots of mean IC per model family across case studies, with a "
    "reference line at zero. Right: horizontal bars giving, for each family, the "
    "percentage of the case studies it ran on where its mean IC was positive.",
)

# %% [markdown]
# ## IC by Asset Class Characteristics
#
# Does universe size, frequency, or cost regime predict IC?

# %% [markdown]
# `best_ic` below comes from `ic_comparison.parquet` rather than from the models
# block of `all_synthesis.json`. That block stores the per-family maximum under a
# key named `ic_mean` (agent-workspace #863), so reading it here would put a
# maximum under the same axis label as the mean plotted in the heatmap above. The
# parquet keeps the two apart: `ic_mean` is the family average, `ic_best` the
# family maximum.

# %%
_best_ic = (
    ic_df.filter(pl.col("ic_best").is_not_null())
    .group_by("case_study")
    .agg(best_ic=pl.col("ic_best").max())
)
meta_df = (
    pl.DataFrame(
        [
            {
                "case_study": DISPLAY_NAMES.get(cs, cs),
                "universe_size": data["meta"]["universe_size"],
                "frequency": data["meta"]["frequency"],
                "cost_bps": data["meta"]["cost_bps"],
                "n_families": len(data["pipeline_summary"]["models"]),
            }
            for cs, data in synthesis.items()
        ]
    )
    # An inner join drops any case study with no registered IC rather than
    # plotting it at a fabricated zero.
    .join(_best_ic, on="case_study", how="inner")
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# IC vs universe size
meta_pd = meta_df.to_pandas()
axes[0].scatter(meta_pd["universe_size"], meta_pd["best_ic"], s=80)
for _, row in meta_pd.iterrows():
    axes[0].annotate(
        row["case_study"],
        (row["universe_size"], row["best_ic"]),
        fontsize=7,
        ha="center",
        va="bottom",
    )
axes[0].set_xlabel("Universe Size (instruments)")
axes[0].set_ylabel("Best IC")
axes[0].set_title("Signal Quality vs Universe Size")
axes[0].set_xscale("log")

# IC vs cost
axes[1].scatter(meta_pd["cost_bps"], meta_pd["best_ic"], s=80)
for _, row in meta_pd.iterrows():
    axes[1].annotate(
        row["case_study"], (row["cost_bps"], row["best_ic"]), fontsize=7, ha="center", va="bottom"
    )
axes[1].set_xlabel("Transaction Cost (bps per leg)")
axes[1].set_ylabel("Best IC")
axes[1].set_title("Signal Quality vs Transaction Costs")

fig.tight_layout()
show_with_alt(
    fig,
    "Two scatter panels, each point a case study labelled by name: best IC "
    "against universe size on a log axis, and best IC against the assumed "
    "per-leg transaction cost in basis points.",
)

# %% [markdown]
# ## Key Takeaways
#
# Which family leads where is a property of this registry, not a standing fact
# about the families, so it is computed below rather than written down. The two
# statements that do survive re-running are structural: no family leads
# everywhere, and universe size does not order the case studies by IC.

# %% tags=["results"]
_leaders = (
    ic_df.filter(pl.col("ic_mean").is_not_null())
    .sort("ic_mean", descending=True)
    .group_by("case_study")
    .first()
    .select("case_study", "family", "ic_mean")
    .sort("ic_mean", descending=True)
)
_lead_counts = _leaders.group_by("family").len().sort("len", descending=True).rows(named=True)
_fam = family_stats.row(0, named=True)
_hi, _lo = _leaders.row(0, named=True), _leaders.row(-1, named=True)
display(
    Markdown(
        f"**Highest mean IC per case study** ranges from {_lo['ic_mean']:.4f} "
        f"({_lo['case_study']}, {FAMILY_NAMES.get(_lo['family'], _lo['family'])}) "
        f"to {_hi['ic_mean']:.4f} ({_hi['case_study']}, "
        f"{FAMILY_NAMES.get(_hi['family'], _hi['family'])}).\n\n"
        "**Which family leads, and how often**: "
        + ", ".join(
            f"{FAMILY_NAMES.get(r['family'], r['family'])} {r['len']}" for r in _lead_counts
        )
        + f" of {_leaders.height}. No family leads everywhere.\n\n"
        f"**Best family average**: {FAMILY_NAMES.get(_fam['family'], _fam['family'])}, "
        f"mean IC {_fam['mean_ic']:.4f} over the {_fam['n_studies']} case studies "
        f"it ran on, positive on {_fam['n_positive']} of them."
    )
)

# %% [markdown]
# The IC-against-universe-size panel shows no ordering: the largest universe is
# not the highest-IC case study and the smallest is not the lowest. Nine points
# could not establish such a relationship even if one existed, so the panel is
# there to show that the obvious confound is not driving the spread, not to rule
# it out. What differs between these case studies is the label horizon, the
# feature menu, and how much of the predictable variation the frictions of that
# market consume - which is where §04 goes next.
#
# ## Known Limitations
#
# - Every IC here is measured on validation folds. It is an in-sample-to-the-
#   selection-process number: configurations were chosen by looking at it. The
#   holdout comparison in §01 is where that selection is priced.
# - The per-family mean averages over however many configurations that family
#   happened to sweep, and the counts differ by an order of magnitude between
#   families and case studies. A family that swept 28 configurations and one that
#   swept 150 do not have comparably precise averages, and no interval is
#   attached to either.
# - Families are absent from a case study when it did not run them. A blank cell
#   means not attempted, never attempted and failed, and the two are not
#   distinguished in the heatmaps.
#
# **Next**: [`04_signal_to_strategy`](04_signal_to_strategy.ipynb) explores the
# gap between IC and Sharpe.
