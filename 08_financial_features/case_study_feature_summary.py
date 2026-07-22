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
# # Cross-Case-Study Feature Evaluation Summary
#
# **Chapter 8: Feature Engineering**
# **Section Reference**: 8.6 - Combining Features and Controlling Search
# **Docker image**: `ml4t`
#
# ## Purpose
#
# This notebook is the cross-case-study inventory and presentation layer: it
# aggregates engineered features and the best registry IC per case study across
# all 9 asset classes. It surfaces:
# - **Feature counts and families**: how large each case study's feature space is
# - **Best IC per case study** (from the model registry): how predictive the
#   strongest family is, by asset class
# - **Cross-asset patterns**: which feature families generalize vs which are asset-specific
#
# The HAC-adjusted significance and BH-FDR survival counts themselves are computed
# upstream, in each case study's `13_model_analysis.py`; this notebook reads and
# presents their results rather than recomputing them.
#
# ## Learning Objectives
#
# 1. Compare feature predictability across diverse asset classes
# 2. Read off each case study's best registry IC and feature-space size
# 3. Identify feature families that generalize vs those that are asset-specific
# 4. Understand how universe size (breadth) interacts with IC magnitude
#
# ## Prerequisites
#
# - Case study feature notebooks must have produced `data/features/financial.parquet`
# - If feature data is missing for some case studies, they show as "no features"

# %%
"""Cross-case-study feature evaluation summary."""

import warnings

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

from utils.paths import get_case_study_dir
from utils.style import COLORS  # importing utils.style registers the ml4t Plotly template

# %% tags=["parameters"]
# Scale parameters (Papermill overrides for testing; readers see production values)
START_DATE = None  # use full dataset

# %% [markdown]
# ## 1. Load Feature Data
#
# Scan all case study `data/features/` directories for the `financial.parquet`
# produced by the feature engineering notebooks. We introspect schemas to count
# features and compare across case studies.

# %%
CASE_STUDIES = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]

DISPLAY_NAMES = {
    "etfs": "ETFs",
    "crypto_perps_funding": "Crypto Perps",
    "nasdaq100_microstructure": "NASDAQ-100",
    "sp500_equity_option_analytics": "S&P 500 Eq+Opt",
    "us_firm_characteristics": "US Firm Chars",
    "fx_pairs": "FX Pairs",
    "cme_futures": "CME Futures",
    "sp500_options": "S&P 500 Options",
    "us_equities_panel": "US Equities",
}

# Columns that are identifiers, not features
_ID_COLS = {"timestamp", "symbol", "product", "stock_id", "instrument_id", "date", "asset"}


def load_feature_info(case_study_id: str) -> dict | None:
    """Load feature summary by introspecting financial.parquet schema."""
    case_dir = get_case_study_dir(case_study_id)
    # Case studies materialize features under <case_dir>/features/ (the
    # naming-conventions doc lists this under data/features/, but the
    # current case-study layout writes directly under features/).
    features_path = case_dir / "features" / "financial.parquet"
    if not features_path.exists():
        return None

    schema = pl.scan_parquet(features_path).collect_schema()
    feature_names = [c for c in schema.names() if c not in _ID_COLS]
    n_features = len(feature_names)

    # Group features into families by prefix (e.g. "mom_", "vol_", "carry_")
    family_counts: dict[str, int] = {}
    for name in feature_names:
        parts = name.split("_")
        family = parts[0] if len(parts) > 1 else "other"
        family_counts[family] = family_counts.get(family, 0) + 1

    return {
        "n_features": n_features,
        "feature_names": feature_names,
        "family_counts": family_counts,
    }


# %%
# Load all feature info
all_results: dict[str, dict] = {}
evaluated: dict[str, dict] = {}
awaiting: list[str] = []

for cs in CASE_STUDIES:
    result = load_feature_info(cs)
    if result is None:
        awaiting.append(cs)
        continue
    all_results[cs] = result
    evaluated[cs] = result

print(f"Case studies with features: {len(evaluated)}/{len(CASE_STUDIES)}")
if evaluated:
    print(f"  Available: {', '.join(DISPLAY_NAMES[cs] for cs in evaluated)}")
if awaiting:
    print(f"  No features: {', '.join(DISPLAY_NAMES.get(cs, cs) for cs in awaiting)}")

# %% [markdown]
# ## 2. Feature Inventory Summary
#
# How many features and feature families does each case study engineer, and what
# are its largest families? (The multiple-testing survival counts are produced
# upstream in each case study's `13_model_analysis.py`; here we inventory the
# feature space.)

# %%
if evaluated:
    summary_rows = []
    for cs, result in evaluated.items():
        summary_rows.append(
            {
                "case_study": DISPLAY_NAMES[cs],
                "n_features": result["n_features"],
                "n_families": len(result["family_counts"]),
                "top_families": ", ".join(
                    f"{k}({v})"
                    for k, v in sorted(result["family_counts"].items(), key=lambda x: -x[1])[:5]
                ),
            }
        )

    summary_df = pl.DataFrame(summary_rows)
    display(summary_df)
else:
    print("No feature data available yet. Run case study feature notebooks first.")

# %% [markdown]
# ## 3. Feature Count Comparison
#
# How does feature set size vary across case studies? More features provide
# a richer signal space but also increase the multiple testing burden.

# %%
if evaluated:
    cs_names = [DISPLAY_NAMES[cs] for cs in evaluated]
    n_features = [evaluated[cs]["n_features"] for cs in evaluated]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cs_names,
            y=n_features,
            marker_color=COLORS["blue"],
            text=[str(n) for n in n_features],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Each case study engineers 39-66 financial features",
        xaxis_title="Case study",
        yaxis_title="Number of features",
        height=450,
    )
    fig.show()
else:
    print("No feature data available.")

# %% [markdown]
# ## 4. Feature Family Distribution
#
# We group features into *families* by their name prefix (the token before the
# first underscore: `mom_21` and `mom_63` both count as `mom`). This is a coarse,
# mechanical lens - across nine very different markets it yields on the order of a
# hundred prefixes, but that overstates the number of distinct ideas. Two effects
# inflate it: each asset class contributes genuinely specialized measures that
# appear nowhere else (option `skew`/`term`/`iv`, microstructure `kyle`/`depth`,
# crypto `funding`/`premium`, futures term-structure), and inconsistent naming
# splits a few shared concepts (`bb` vs `bollinger` for the same Bollinger %B;
# `r12`/`r36`/`past` all momentum windows). Read the y-axis as prefixes, not as a
# taxonomy.
#
# What the figure *is* good for is the cross-asset pattern. A heatmap is the right
# lens for the prefixes that *recur* - momentum, volatility, and returns show up
# almost everywhere - so we keep those (used in two or more case studies, broadest
# on top) in the heatmap, and collapse the long tail of single-market prefixes
# into a companion bar counting how many specialized measures each asset class
# adds.

# %%
if evaluated:
    cs_list = list(evaluated)

    # Collect all family names across case studies
    all_families: set[str] = set()
    for cs in cs_list:
        all_families.update(evaluated[cs]["family_counts"].keys())

    if all_families:
        # Breadth = number of case studies each family appears in; total count is
        # the tiebreak so the heavy hitters float to the top of the heatmap.
        breadth = {
            fam: sum(1 for cs in cs_list if evaluated[cs]["family_counts"].get(fam, 0) > 0)
            for fam in all_families
        }
        total_count = {
            fam: sum(evaluated[cs]["family_counts"].get(fam, 0) for cs in cs_list)
            for fam in all_families
        }
        # Ascending sort: Plotly stacks the first y entry at the bottom, so the
        # broadest / heaviest families end up on top.
        recurring = sorted(
            (f for f in all_families if breadth[f] >= 2),
            key=lambda f: (breadth[f], total_count[f], f),
        )
        singletons = [f for f in all_families if breadth[f] == 1]

        # One asset-specific family belongs to exactly one case study; count them.
        singleton_counts = [
            sum(1 for f in singletons if evaluated[cs]["family_counts"].get(f, 0) > 0)
            for cs in cs_list
        ]

        heatmap_data = [
            [(evaluated[cs]["family_counts"].get(fam, 0) or float("nan")) for cs in cs_list]
            for fam in recurring
        ]

        # One catch-all "other" bucket can spike far above the real families and
        # flatten the palette; winsorize the color at the 90th percentile so the
        # gradient stays readable. Printed counts remain exact.
        _flat = [v for row in heatmap_data for v in row if not np.isnan(v)]
        zmax_cap = float(np.nanpercentile(_flat, 90)) if _flat else None

        cs_labels = [DISPLAY_NAMES[cs] for cs in cs_list]
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.82, 0.18],
            vertical_spacing=0.09,
            subplot_titles=(
                f"{len(recurring)} prefixes recur across two or more asset classes",
                "Each asset class also adds its own specialized measures",
            ),
        )
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=cs_labels,
                y=recurring,
                colorscale=[[0.0, COLORS["silver_muted"]], [1.0, COLORS["blue"]]],
                zmax=zmax_cap,
                zmin=0,
                text=[
                    [f"{int(v)}" if not np.isnan(v) else "" for v in row] for row in heatmap_data
                ],
                texttemplate="%{text}",
                textfont={"size": 9},
                colorbar={"title": "Features", "len": 0.82, "y": 1.0, "yanchor": "top"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=cs_labels,
                y=singleton_counts,
                marker_color=COLORS["amber"],
                text=singleton_counts,
                textposition="outside",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Feature prefix", row=1, col=1)
        fig.update_xaxes(title_text="Case study", row=2, col=1)
        fig.update_yaxes(title_text="Specialized", row=2, col=1)
        # Establish a clear title hierarchy: prominent claim title on top, the two
        # panel labels smaller and in body color (the template renders subplot
        # titles larger/blue, which reads backwards against the main title).
        fig.update_annotations(font={"size": 13, "color": COLORS["slate"]})
        fig.update_layout(
            title={
                "text": "A few families recur across asset classes; most are asset-specific",
                "font": {"size": 19},
            },
            height=max(500, len(recurring) * 26 + 280),
            width=max(700, len(cs_list) * 90 + 260),
        )
        fig.show()
    else:
        print("No family-level data available.")
else:
    print("No feature data available.")

# %% [markdown]
# ## 5. Representative Features Across Case Studies
#
# A sample of each case study's feature space — the first few feature names in
# schema order — to illustrate the engineered inputs. This is an inventory view,
# not an IC ranking (per-feature IC is computed in each case study's evaluation
# notebook).

# %%
if evaluated:
    top_features_all = []
    for cs in evaluated:
        # Show first 5 feature names per case study
        for feat_name in evaluated[cs]["feature_names"][:5]:
            top_features_all.append(
                {
                    "case_study": DISPLAY_NAMES[cs],
                    "feature": feat_name,
                }
            )

    if top_features_all:
        top_df = pl.DataFrame(top_features_all)
        display(top_df)
else:
    print("No feature data available.")

# %% [markdown]
# ## 6. Breadth vs IC: The Fundamental Law Perspective
#
# The Fundamental Law of Active Management says:
#
# $$IR \approx IC \times \sqrt{BR}$$
#
# where $BR$ is the number of independent bets (roughly the universe size).
# A case study with IC = 0.01 and 3,000 stocks achieves IR = 0.55,
# while IC = 0.03 with 20 pairs gives IR = 0.13. Breadth matters enormously.

# %%
from case_studies.utils.analytics import DATASET_META, load_best_ic_per_family

if evaluated:
    # Load best IC per family from registry to combine with universe metadata
    best_ic_df = load_best_ic_per_family()

    if not best_ic_df.is_empty():
        # Get best IC per case study (across all families)
        best_per_cs = (
            best_ic_df.sort("ic_mean", descending=True, nulls_last=True)
            .group_by("case_study")
            .first()
            .select("case_study", "ic_mean")
        )

        breadth_data = []
        for row in best_per_cs.iter_rows(named=True):
            cs = row["case_study"]
            meta = DATASET_META.get(cs, {})
            n_entities = meta.get("entities", 0)
            ic = abs(row["ic_mean"]) if row["ic_mean"] is not None else 0.0
            if n_entities == 0 or ic == 0.0:
                continue
            ir_estimate = ic * np.sqrt(n_entities)
            breadth_data.append(
                {
                    "case_study": DISPLAY_NAMES.get(cs, cs),
                    "universe_size": n_entities,
                    "best_abs_ic": round(ic, 4),
                    "estimated_ir": round(ir_estimate, 2),
                }
            )

        if breadth_data:
            breadth_df = pl.DataFrame(breadth_data).sort("estimated_ir", descending=True)
            display(breadth_df)
        else:
            print("No IC data available from registry.")
    else:
        print("No model IC data in registry yet.")
else:
    print("No feature data available.")

# %%
# Visualize breadth vs IC
if evaluated and "breadth_data" in dir() and breadth_data:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[brow["universe_size"] for brow in breadth_data],
            y=[brow["best_abs_ic"] for brow in breadth_data],
            mode="markers+text",
            text=[brow["case_study"] for brow in breadth_data],
            textposition="top center",
            marker=dict(
                size=[brow["estimated_ir"] * 20 + 5 for brow in breadth_data],
                color=COLORS["blue"],
                opacity=0.75,
                line=dict(width=1, color=COLORS["slate"]),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Breadth lifts risk-adjusted signal even where per-bet IC is small",
        xaxis_title="Universe size (number of instruments, log scale)",
        yaxis_title="Best model |IC|",
        xaxis_type="log",
        height=450,
    )
    fig.show()

# %% [markdown]
# ## What the Panels Above Show
#
# The notebook aggregates whatever is present in each case study's
# `data/features/financial.parquet` and the model registry. The substantive
# findings — which feature families have predictive content for which label
# and horizon, how many features survive HAC + BH-FDR, and how breadth
# interacts with IC magnitude — are produced by the per-case-study evaluation
# notebooks (`13_model_analysis.py` in each case study). This summary
# notebook is a cross-case-study inventory and presentation layer; it does
# not itself compute IC or run multiple-testing correction.
#
# **Next**: See `09_model_based_features/case_study_temporal_summary` for the
# temporal/model-based feature companion view.
# **Book**: Chapter 8.6 discusses combining features and controlling the
# search space to avoid data mining.
