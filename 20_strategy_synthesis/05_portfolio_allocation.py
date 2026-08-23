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
# # Portfolio Allocation Across the Case Studies
#
# **Docker image**: `ml4t`
#
# The previous notebook traced the journey from signals to strategies. Here we
# isolate the allocator layer: the same forecasts are routed through equal
# weight, inverse-vol, MVO, risk parity and score-weighted allocators, and the
# resulting Sharpe spread says how much room the allocator has to add or subtract
# value once the signal is fixed.
#
# A case study can only take part if it has a signal to hold fixed, meaning a
# resolved spine prediction with allocation-stage backtests behind it. How many
# do is a property of the current registry rather than a fixed number, so the
# notebook reports it below instead of stating it here.
#
# **Learning Objectives**:
# - Compare allocator performance across diverse asset classes and frequencies
# - Identify when allocation amplifies signal vs when signal is too weak to matter
# - Examine the spread between best and worst allocators per case study
#
# **Book Reference**: Chapter 20, Section 20.5 (Portfolio Allocation Across the Case Studies)
#
# **Prerequisites**: Run [`01_aggregate_synthesis`](01_aggregate_synthesis.ipynb) first.
# Each case study's registry must contain Ch17 allocation-stage backtests.

# %%
"""Ch20 Portfolio Allocation — spine-pinned allocator comparison across 8 case studies.

The Table 20.6 / Figure 20.7 semantics are: hold the highest-validation-Sharpe
signal carrier fixed per case study, then compare allocators at the allocation
stage only, so the best-minus-worst spread measures what the allocator alone
contributes (risk overlays are a downstream layer covered in §20.7). This
matches the caption "rank-1 signal held fixed" and the spine-pinning logic in
``case_studies.utils.strategy_analysis.build_all_synthesis``.
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import Markdown, display

from case_studies.utils.analytics import (
    CASE_STUDY_IDS,
    SHORT_NAMES,
    extract_allocator,
    load_chapter_backtests,
)
from utils.paths import get_chapter_dir
from utils.style import show_with_alt

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
MAX_CASE_STUDIES = 0  # 0 = all

# %%
CS_LIST = CASE_STUDY_IDS[:MAX_CASE_STUDIES] if MAX_CASE_STUDIES else CASE_STUDY_IDS


def extract_top_k(spec_json: str) -> int:
    """Extract top_k from backtest spec."""
    spec = json.loads(spec_json)
    return spec.get("allocation", {}).get("top_k", 0) or spec.get("signal", {}).get("top_k", 0)


# %% [markdown]
# ## Load Allocation Results from Registry
#
# Ch17 backtests compare different allocation methods (equal weight, inverse
# vol, MVO, risk parity, score-weighted, HRP) while holding the signal
# constant. We load the `stage: "allocation"` runs and restrict to the spine
# prediction_hash per case study so the comparison reads as
# "best within the highest-validation-Sharpe signal carrier".
#
# A case study absent from the carrier file is an error: `01_aggregate_synthesis`
# writes a row for every case study it iterates, so a missing key means the file
# is stale. A case study present with a *null* spine is different: its registry
# holds no backtests to resolve one from. It is named and excluded. The check
# below used to test only that the key existed, so a null passed it and then
# dropped out silently at the filter, the outcome the check exists to prevent.
#
# Universe size, used to size the bubbles much further down, comes from
# `overview.parquet` rather than from a dict typed into this notebook. Every one
# of that dict's nine entries had drifted from the shipped artifact - etfs 64
# against 100, sp500_options 480 against 627, us_equities_panel 311 against 3199.

# %%
# Spine prediction hash per case study — read from the synthesis carrier
# file produced by 01_aggregate_synthesis. backtest_comparison.parquet is
# the canonical Ch20 artifact that records spine_prediction_hash per CS.
_spine_df = pl.read_parquet(get_chapter_dir(20) / "output" / "backtest_comparison.parquet").select(
    "case_study_id", "spine_prediction_hash"
)
SPINE_BY_CS = dict(
    zip(
        _spine_df["case_study_id"].to_list(),
        _spine_df["spine_prediction_hash"].to_list(),
        strict=False,
    )
)

_missing_spine = [cs for cs in CS_LIST if cs not in SPINE_BY_CS]
if _missing_spine:
    msg = (
        f"No spine_prediction_hash in backtest_comparison.parquet for: "
        f"{_missing_spine}. Re-run 01_aggregate_synthesis to refresh the carrier."
    )
    raise RuntimeError(msg)

NO_SPINE = sorted(cs for cs in CS_LIST if SPINE_BY_CS.get(cs) is None)
if NO_SPINE:
    print(f"No resolved spine, excluded from the allocator comparison: {NO_SPINE}")

# This section isolates the allocator layer: the signal is held fixed and only
# the allocation method varies. Load the allocation stage ONLY — a trailing-stop
# or other risk overlay (ch19) is a downstream layer, and including its Sharpe
# would credit the allocator with work the overlay did, contradicting the
# "allocator alone" measurement. The "max across rebalance and top-K variants"
# in Table 20.6 is taken over allocation-stage variants, not over overlays.
combined_raw = load_chapter_backtests(
    "ch17",
    case_studies=CS_LIST,
    metrics=[
        "sharpe",
        "sortino",
        "total_return",
        "max_drawdown",
        "cagr",
        "volatility",
        "num_trades",
        "win_rate",
    ],
)

if combined_raw.is_empty():
    msg = "No allocation-stage backtest results found across case studies"
    raise RuntimeError(msg)

# Pin to spine prediction_hash per case study.
spine_filter = pl.struct(["case_study", "prediction_hash"]).map_elements(
    lambda row: row["prediction_hash"] == SPINE_BY_CS.get(row["case_study"]),
    return_dtype=pl.Boolean,
)
combined = combined_raw.filter(spine_filter)

# Extract allocator method and top_k from spec_json
comparison = (
    combined.with_columns(
        allocator=pl.col("spec_json").map_elements(extract_allocator, return_dtype=pl.Utf8),
        top_k=pl.col("spec_json").map_elements(extract_top_k, return_dtype=pl.Int64),
    )
    .filter(
        pl.col("allocator").is_in(
            [
                "equal_weight",
                "inverse_vol",
                "score_weighted",
                "mvo_ledoit_wolf",
                "risk_parity",
                "hrp",
                "conformal_weighted",
            ]
        )
    )
    .select(
        "case_study",
        pl.col("case_study").replace(SHORT_NAMES).alias("display_name"),
        "label",
        "allocator",
        "top_k",
        "sharpe",
        "sortino",
        "total_return",
        "max_drawdown",
        "cagr",
        "volatility",
        "num_trades",
    )
)

# Collapse to one row per (case_study, allocator) — keep the configuration
# (rebalance / top_k / overlay) that posts the highest Sharpe. Table 20.6
# entries are the "best for that allocator" within the spine carrier.
comparison = comparison.sort("sharpe", descending=True).unique(
    subset=["case_study", "allocator"], keep="first"
)

n_cs = comparison["case_study"].n_unique()
print(
    f"Allocators × case studies (spine-pinned): {len(comparison)} rows across {n_cs} case studies"
)
comparison.head(10)

# %% [markdown]
# ## Highest-Sharpe Allocator by Case Study
#
# Which allocation method achieves the highest Sharpe for each dataset?

# %%
best_per_cs = (
    comparison.sort("sharpe", descending=True)
    .unique(subset=["case_study"], keep="first")
    .select(
        "display_name",
        "label",
        "allocator",
        "top_k",
        "sharpe",
        "sortino",
        "total_return",
        "max_drawdown",
    )
    .sort("sharpe", descending=True)
)
print("Best allocator per case study (sorted by Sharpe):\n")
best_per_cs

# %% [markdown]
# ## Sharpe by Allocator (Bar Chart)
#
# A direct visual comparison of each allocator's best Sharpe across case studies.

# %%
best_data = best_per_cs.to_pandas()

fig, ax = plt.subplots(figsize=(10, 5))
colors_map = {}
color_cycle = plt.cm.Set2.colors
for i, alloc in enumerate(best_data["allocator"].unique()):
    colors_map[alloc] = color_cycle[i % len(color_cycle)]

bars = ax.barh(
    best_data["display_name"],
    best_data["sharpe"],
    color=[colors_map.get(a, "steelblue") for a in best_data["allocator"]],
)
ax.set_xlabel("Sharpe Ratio")
ax.set_title("Best Allocator Sharpe by Case Study")
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

for bar, alloc in zip(bars, best_data["allocator"], strict=False):
    x_pos = bar.get_width()
    ax.text(
        x_pos + 0.05 if x_pos >= 0 else x_pos - 0.05,
        bar.get_y() + bar.get_height() / 2,
        alloc.replace("_", " "),
        va="center",
        ha="left" if x_pos >= 0 else "right",
        fontsize=8,
    )

show_with_alt(
    fig,
    "Bar chart of the highest Sharpe reached by any allocator in each case "
    "study, ordered by that Sharpe, with a reference line at zero.",
)

# %% [markdown]
# **Interpretation**: The bar chart makes the cross-dataset heterogeneity immediately visible.
# Case studies with strong signals show positive Sharpe regardless of allocator choice.
# Case studies with weak signals show negative Sharpe across all allocators — no allocation
# method can rescue a failing signal.

# %% [markdown]
# ## Equal-Weight Baseline vs Best Allocator
#
# How much does allocation choice improve over the simplest baseline?

# %%
ew_sharpes = (
    comparison.filter(pl.col("allocator") == "equal_weight")
    .group_by("case_study")
    .agg(ew_sharpe=pl.col("sharpe").max())
)

best_sharpes = (
    comparison.sort("sharpe", descending=True)
    .unique(subset=["case_study"], keep="first")
    .select("case_study", best_sharpe=pl.col("sharpe"), best_allocator=pl.col("allocator"))
)

improvement = (
    best_sharpes.join(ew_sharpes, on="case_study", how="left")
    .with_columns(
        display_name=pl.col("case_study").replace(SHORT_NAMES),
        sharpe_diff=pl.col("best_sharpe") - pl.col("ew_sharpe"),
    )
    .with_columns(
        # Null, not zero, where there is no equal-weight run to improve on. A
        # zero reads as "the allocator changed nothing", which is a measurement;
        # a missing baseline is the absence of one.
        pct_improvement=pl.when(pl.col("ew_sharpe").abs() > 0.001)
        .then((pl.col("sharpe_diff") / pl.col("ew_sharpe").abs()) * 100)
        .otherwise(None),
    )
    .sort("sharpe_diff", descending=True)
)
print("Sharpe improvement from equal-weight to best allocator:\n")
improvement.select(
    "display_name", "ew_sharpe", "best_sharpe", "best_allocator", "sharpe_diff", "pct_improvement"
)

# %% tags=["results"]
_with_ew = improvement.filter(pl.col("ew_sharpe").is_not_null())
_without = improvement.filter(pl.col("ew_sharpe").is_null())
display(
    Markdown(
        f"{_with_ew.height} of {improvement.height} case studies in this "
        "comparison ran an equal-weight allocator on the spine prediction, so "
        "only those can be measured against the baseline"
        + (
            f". The rest ({', '.join(_without['display_name'].to_list())}) have "
            "a highest-Sharpe allocator but nothing to compare it with, and are "
            "left blank rather than credited with a zero improvement"
            if _without.height
            else ""
        )
        + "."
        + (
            (
                " In every one of them the highest-Sharpe allocator is equal "
                "weight itself, so no alternative improved on the baseline."
                if (_with_ew["sharpe_diff"].abs() < 1e-9).all()
                else f" Where the comparison can be made, the change ranges from "
                f"{_with_ew['sharpe_diff'].min():+.3f} to "
                f"{_with_ew['sharpe_diff'].max():+.3f} Sharpe."
            )
            if _with_ew.height
            else ""
        )
    )
)

# %% [markdown]
# ## Allocator Performance Heatmap
#
# Visualize which allocators appear across case studies and their relative performance.

# %%
# Find the most common allocators
alloc_counts = comparison.group_by("allocator").agg(n=pl.len()).sort("n", descending=True)
common_allocators = alloc_counts.head(5)["allocator"].to_list()

heatmap_data = (
    comparison.filter(pl.col("allocator").is_in(common_allocators))
    .group_by("display_name", "allocator")
    .agg(sharpe=pl.col("sharpe").max())
    .pivot(on="allocator", index="display_name", values="sharpe")
)

cs_names = heatmap_data["display_name"].to_list()
alloc_cols = [c for c in heatmap_data.columns if c != "display_name"]
data_matrix = heatmap_data.select(alloc_cols).to_numpy()
mask = np.isnan(data_matrix)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    np.where(mask, 0, data_matrix),
    cmap="RdYlGn",
    aspect="auto",
    vmin=-1,
    vmax=1,
)

for i in range(len(cs_names)):
    for j in range(len(alloc_cols)):
        val = data_matrix[i, j]
        if not np.isnan(val):
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
        else:
            ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=8)

ax.set_xticks(range(len(alloc_cols)))
ax.set_xticklabels([a.replace("_", " ").title() for a in alloc_cols], rotation=30, ha="right")
ax.set_yticks(range(len(cs_names)))
ax.set_yticklabels(cs_names)
ax.set_title("Allocator Sharpe Ratios Across Case Studies")
fig.colorbar(im, ax=ax, label="Sharpe Ratio", shrink=0.8)
fig.subplots_adjust(left=0.18, right=0.92, top=0.9, bottom=0.2)
show_with_alt(
    fig,
    "Grouped bars comparing the Sharpe of each of the most common allocators "
    "within each case study, so the height differences within a group show how "
    "much the allocator choice moved the result.",
)

# %% [markdown]
# **Interpretation**: The heatmap reveals that no single allocator dominates
# across all case studies. Missing cells (N/A) indicate that not every
# allocator was tested on every dataset — strategy-specific constraints
# (e.g., long-only, no shorting) exclude certain methods. Green cells
# (Sharpe > 0) cluster around a few case studies with strong underlying
# signals, confirming that the upstream prediction quality matters more
# than the allocation method. Red or near-zero cells indicate that even
# the best allocator cannot rescue a weak signal.

# %% [markdown]
# ## Signal Strength vs Allocation Impact
#
# Does allocation choice matter more when the underlying signal is strong?
# We compare the spread between best and worst allocator Sharpe per case study.

# %%
spread_df = (
    comparison.group_by("case_study")
    .agg(
        display_name=pl.col("display_name").first(),
        best_sharpe=pl.col("sharpe").max(),
        worst_sharpe=pl.col("sharpe").min(),
        n_allocators=pl.col("allocator").n_unique(),
    )
    .with_columns(spread=pl.col("best_sharpe") - pl.col("worst_sharpe"))
    .sort("spread", descending=True)
)
print("Allocator Sharpe spread (best - worst) per case study:\n")
spread_df.select("display_name", "best_sharpe", "worst_sharpe", "spread", "n_allocators")

# %% [markdown]
# **Interpretation**: The best-minus-worst spread is the cleanest summary of
# allocator sensitivity. Wide spreads indicate that portfolio construction is
# load-bearing for that dataset, while tight spreads indicate that upstream
# signal quality dominates method choice.

# %% [markdown]
# ## Summary Statistics

# %%
positive = spread_df.filter(pl.col("best_sharpe") > 0).height
total = spread_df.height

print(f"""
=== Cross-Dataset Allocation Summary ===
Case studies loaded:    {n_cs}
Total allocator combos: {len(comparison)}

--- Signal Quality ---""")

for row in spread_df.iter_rows(named=True):
    cs = row["display_name"]
    best = row["best_sharpe"]
    spread = row["spread"]
    print(f"  {cs:20s}  Best SR: {best:+.2f}  Spread: {spread:.3f}")

print("\n--- Key Findings ---")
print(f"- Positive best Sharpe in {positive}/{total} case studies")
if spread_df.height > 0:
    widest = spread_df.row(0, named=True)
    tightest = spread_df.sort("spread").row(0, named=True)
    print(
        f"- Largest allocation sensitivity: {widest['display_name']} (spread={widest['spread']:.3f})"
    )
    print(
        f"- Smallest allocation sensitivity: {tightest['display_name']} "
        f"(spread={tightest['spread']:.3f})"
    )
print("- No single allocator has the highest Sharpe in every case study")

# %% [markdown]
# ## Uplift Breadth: Is Allocation Improvement Broad or Narrow?
#
# A practitioner needs to know: is the allocation improvement driven by
# one lucky combination, or do most allocators improve over the baseline?
# Narrow uplift (only one allocator helps) is less reliable than broad
# uplift (most allocators improve).

# %%
breadth_rows = []
for cs_id in comparison["case_study"].unique().sort().to_list():
    cs_data = comparison.filter(pl.col("case_study") == cs_id)
    ew_row = cs_data.filter(pl.col("allocator") == "equal_weight")
    if ew_row.is_empty():
        continue
    ew_sr = ew_row["sharpe"].max()
    non_ew = cs_data.filter(pl.col("allocator") != "equal_weight")
    if non_ew.is_empty():
        continue
    n_better = non_ew.filter(pl.col("sharpe") > ew_sr).height
    n_total = non_ew.height
    breadth_rows.append(
        {
            "case_study": cs_id,
            "display_name": SHORT_NAMES.get(cs_id, cs_id),
            "ew_sharpe": round(ew_sr, 3),
            "n_allocators_tested": n_total,
            "n_beat_ew": n_better,
            "pct_beat_ew": round(100 * n_better / n_total, 0) if n_total > 0 else 0,
            # "narrow" used to absorb the zero case, labelling "one or two
            # allocators beat the baseline" and "none did" identically.
            "breadth": "none"
            if n_better == 0
            else "broad"
            if n_better / max(n_total, 1) > 0.5
            else "moderate"
            if n_better / max(n_total, 1) > 0.25
            else "narrow",
        }
    )

breadth = (
    pl.DataFrame(breadth_rows).sort("pct_beat_ew", descending=True)
    if breadth_rows
    else pl.DataFrame()
)

# %%
if not breadth.is_empty():
    print("=== Allocation Uplift Breadth ===")
    print(
        breadth.select(
            "display_name",
            "ew_sharpe",
            "n_beat_ew",
            "n_allocators_tested",
            "pct_beat_ew",
            "breadth",
        )
    )

    for _label in ("broad", "moderate", "narrow", "none"):
        _n = breadth.filter(pl.col("breadth") == _label).height
        if _n:
            print(f"\n{_label} uplift: {_n}/{breadth.height}")
    _n_compared = comparison["case_study"].n_unique()
    if breadth.height < _n_compared:
        _dropped = sorted(set(comparison["case_study"].unique()) - set(breadth["case_study"]))
        print(
            f"\nExcluded for having no equal-weight run to compare against "
            f"({len(_dropped)} of {_n_compared}): "
            + ", ".join(SHORT_NAMES.get(c, c) for c in _dropped)
        )
else:
    print("No case study has both an equal-weight run and an alternative to compare it with.")

# %% [markdown]
# **Interpretation**: "Broad" means most non-EW allocators improve over
# equal weight, so the uplift does not depend on which one was chosen. "Narrow"
# means only one or two exceed the baseline, so the improvement is the choice.
# "None" means no alternative beat equal weight at all, which is a different
# result again and was previously reported as "narrow". This matters for Ch20:
# broad uplift
# is more trustworthy as a real improvement; narrow uplift could be
# selection bias.

# %% [markdown]
# ## When Does Allocation Optimization Help?
#
# The previous sections show that allocation uplift varies across datasets.
# Here we ask the structural question: **what predicts whether optimization
# helps?** We hypothesize two factors:
#
# - **Signal strength** (EW baseline Sharpe): When the signal is strong,
#   most allocators produce positive returns — optimization adds little.
# - **Universe size**: In broad universes, allocation determines
#   concentration; in narrow ones, top-$k$ selection dominates.
#
# The scatter below tests this hypothesis.

# %%
# Build structural features for the "when MVO helps" analysis.
# The EW baseline comes from Ch16 (signal stage), not Ch17 allocators.
ch16_raw = load_chapter_backtests("ch16", case_studies=CS_LIST, metrics=["sharpe"])
ch16_ew = (
    ch16_raw.with_columns(
        allocator=pl.col("spec_json").map_elements(extract_allocator, return_dtype=pl.Utf8),
    )
    .filter(pl.col("allocator") == "equal_weight")
    .sort("sharpe", descending=True)
    .unique(subset=["case_study"], keep="first")
    .select("case_study", ew_sharpe=pl.col("sharpe"))
)

UNIVERSE_SIZES = dict(
    pl.read_parquet(get_chapter_dir(20) / "output" / "overview.parquet")
    .select("cs_id", "universe")
    .iter_rows()
)

# %%
# %%
# Best allocator Sharpe per CS from Ch17
best_alloc = (
    comparison.sort("sharpe", descending=True)
    .unique(subset=["case_study"], keep="first")
    .select("case_study", best_sharpe=pl.col("sharpe"))
)

mvo_data = best_alloc.join(ch16_ew, on="case_study", how="inner").with_columns(
    display_name=pl.col("case_study").replace(SHORT_NAMES),
    uplift=pl.col("best_sharpe") - pl.col("ew_sharpe"),
)

mvo_rows = []
for row in mvo_data.iter_rows(named=True):
    mvo_rows.append(
        {
            "case_study": row["case_study"],
            "display_name": row["display_name"],
            "ew_sharpe": float(row["ew_sharpe"]),
            "best_sharpe": float(row["best_sharpe"]),
            "uplift": float(row["uplift"]),
            "universe_size": UNIVERSE_SIZES.get(row["case_study"], 0),
        }
    )

mvo_df = pl.DataFrame(mvo_rows)

# %%
# Scatter plot: signal strength vs allocation uplift
if mvo_df.height >= 3:
    fig, ax = plt.subplots(figsize=(10, 7))

    ew_vals = mvo_df["ew_sharpe"].to_list()
    uplift_vals = mvo_df["uplift"].to_list()
    sizes = [max(30, n / 8) for n in mvo_df["universe_size"].to_list()]
    names = mvo_df["display_name"].to_list()

    # Color by whether uplift is positive
    colors = ["#2ecc71" if u > 0 else "#e74c3c" for u in uplift_vals]

    ax.scatter(ew_vals, uplift_vals, s=sizes, c=colors, alpha=0.7, edgecolors="white", zorder=5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    for name, x, y in zip(names, ew_vals, uplift_vals, strict=False):
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color="#475569",
        )

    ax.set_xlabel("Signal Strength (EW Baseline Sharpe)")
    ax.set_ylabel("Allocation Uplift (Best Allocator − EW Sharpe)")
    ax.set_title("When Does Allocation Optimization Help?")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(
        xlim[1] * 0.95,
        ylim[1] * 0.9,
        "Strong signal\n+ optimization helps",
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#0f7c3a",
        bbox={"facecolor": "white", "edgecolor": "#0f7c3a", "alpha": 0.85, "pad": 3},
    )
    ax.text(
        xlim[0] + (xlim[1] - xlim[0]) * 0.05,
        ylim[0] + (ylim[1] - ylim[0]) * 0.05,
        "Weak signal\n+ optimization hurts",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#a32222",
        bbox={"facecolor": "white", "edgecolor": "#a32222", "alpha": 0.85, "pad": 3},
    )

    from matplotlib.lines import Line2D

    size_legend = [
        Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=5, label="N ≈ 20"),
        Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=10, label="N ≈ 300"),
        Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=15, label="N ≈ 2000+"),
    ]
    ax.legend(
        handles=size_legend,
        title="Universe Size",
        loc="lower right",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
    )

    show_with_alt(
        fig,
        "Scatter of allocation uplift against the equal-weight baseline Sharpe, "
        "one bubble per case study sized by universe size and labelled by name, "
        "with quadrant annotations naming the four possible regimes.",
    )
else:
    print("Insufficient data for MVO diagnostic scatter.")

# %% [markdown]
# %% tags=["results"]
display(
    Markdown(
        f"The scatter carries {mvo_df.height} case studies: only those with both "
        "a Ch16 equal-weight baseline and Ch17 allocation backtests on the spine "
        "prediction qualify. Their baselines run from "
        f"{mvo_df['ew_sharpe'].min():+.2f} to {mvo_df['ew_sharpe'].max():+.2f} "
        f"Sharpe and their uplifts from {mvo_df['uplift'].min():+.2f} to "
        f"{mvo_df['uplift'].max():+.2f}.\n\n"
        "Every point sits in the same region of the plane, so the quadrant "
        "labels describe where a case study could fall rather than where any of "
        "these do. Nothing here separates a weak-signal regime from a "
        "strong-signal one, because no weak-signal case study reached this "
        "comparison. The chart is a template waiting for the registries still "
        "being rebuilt."
    )
)

# %% [markdown]
# The mechanism the chart is meant to test is still worth stating, as a thing to
# check rather than a thing shown: an allocator can only redistribute capital
# across whatever the signal ranked, so when the ranking carries little
# information the allocator is redistributing noise, and a method with more free
# parameters has more ways to fit that noise. Whether that holds here needs
# case studies on both sides of the line, which this registry does not have.

# %% [markdown]
# ## Key Takeaways
#
# The allocator that wins in each case study, and by how much over equal weight,
# are properties of this registry. They are computed below. An earlier version of
# this section typed them out, and named a leading allocator for six case studies
# that have no allocation-stage backtests at all.

# %% tags=["results"]
_wins = best_sharpes.group_by("best_allocator").len().sort("len", descending=True)
_spread_now = spread_df.sort("spread", descending=True)
_wide, _tight = _spread_now.row(0, named=True), _spread_now.row(-1, named=True)
display(
    Markdown(
        "**Which allocator posts the highest Sharpe**: "
        + ", ".join(f"{r['best_allocator']} ({r['len']})" for r in _wins.iter_rows(named=True))
        + f", across {best_sharpes.height} case studies. "
        + (
            "No allocator wins everywhere."
            if _wins.height > 1
            else "One allocator wins in all of them, which at this sample size is "
            "not evidence that it generally does."
        )
        + f"\n\n**Allocator sensitivity** (highest minus lowest Sharpe within a "
        f"case study) runs from {_tight['spread']:.3f} ({_tight['display_name']}, "
        f"{_tight['n_allocators']} allocators) to {_wide['spread']:.3f} "
        f"({_wide['display_name']}, {_wide['n_allocators']} allocators). Where the "
        "spread is small, the choice of allocator is not what determines the "
        "result; where it is large, it is a decision that has to be made on "
        "validation data and carries the selection risk that implies."
    )
)

# %% [markdown]
# What holds independently of the registry:
#
# - **Allocation cannot manufacture a signal.** Every allocator reads the same
#   ranking; if the ranking is uninformative, redistributing capital across it
#   changes the variance of the result and not its expectation.
# - **The spread between allocators is itself the useful number.** A case study
#   where every allocator lands in the same place is one where this decision can
#   be made on grounds other than backtested Sharpe - turnover, capacity,
#   explicability. A case study with a wide spread is one where the decision was
#   made by looking at validation results, and should be treated accordingly.
# - **Equal weight is the baseline worth beating**, because it has no parameters
#   to fit and therefore nothing to overfit. An allocator that does not clear it
#   has bought estimation risk for nothing.
#
# ## Known Limitations
#
# - Only case studies with a resolved spine prediction take part, and the count
#   is printed at the top of the notebook. The rest are mid-rebuild.
# - Every Sharpe here is measured on validation folds. The allocator was chosen
#   by looking at these numbers, so the spread between allocators overstates what
#   the choice is worth out of sample by an amount this notebook does not
#   estimate.
# - Each allocator is represented by its own highest-Sharpe configuration across
#   rebalance and top-k variants, so the comparison is between maxima and each is
#   inflated by however many variants stood behind it.
#
# **Next**: [`06_cost_survival`](06_cost_survival.ipynb) translates the same
# allocator choices into cost-adjusted performance.
