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
# aggregates engineered features and the strongest registry IC per case study across
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
# - A case study whose `features/` directory holds no panel yet shows as "awaiting".
#   A checkout that is not wired to a case study at all cannot tell that apart from
#   an empty one, so it refuses rather than reporting a range over what it could see.

# %%
"""Cross-case-study feature evaluation summary."""

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from plotly.subplots import make_subplots

from utils.paths import display_path, get_case_study_dir
from utils.style import (  # importing utils.style registers the ml4t Plotly template
    COLORS,
    show_plotly_with_alt,
)

# %% tags=["parameters"]
# Scale parameters (Papermill overrides for testing; readers see production values)
START_DATE = None  # use full dataset

# %% [markdown]
# ## Load Feature Data
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


def resolve_feature_panel(case_study_id: str) -> tuple[str, Path]:
    """Locate a case study's engineered feature panel and say what state it is in.

    Case studies materialize features under ``<case_dir>/features/`` (the
    naming-conventions doc lists this under ``data/features/``, but the current
    case-study layout writes directly under ``features/``). A worktree wires that
    directory in by symlink, so its absence and an empty one are different facts:

    ``readable``     the panel is on disk and can be scanned.
    ``awaiting``     ``features/`` is there and holds no ``financial.parquet`` yet,
                     so this case study has published no feature panel.
    ``unreachable``  ``features/`` itself is absent, so this checkout is not wired
                     to the case study and cannot report on it either way.
    """
    case_dir = get_case_study_dir(case_study_id)
    features_dir = case_dir / "features"
    panel_path = features_dir / "financial.parquet"
    if panel_path.exists():
        return "readable", panel_path
    if features_dir.exists():
        return "awaiting", panel_path
    return "unreachable", panel_path


def refuse_a_partial_view(unreachable: list[tuple[str, Path]], n_case_studies: int) -> None:
    """Raise when a case study is invisible to this checkout rather than empty.

    ML4T_OUTPUT_DIR redirects every case study to a scratch root, which pytest sets and
    which legitimately holds nothing. Without it the case studies resolve to the canonical
    store, every one of them is wired in, and a missing ``features/`` is this checkout's
    limitation rather than the case study's state. Every number this notebook reports - the
    count, the range, and which family prefixes look universal - is taken over whichever
    case studies were visible, so publishing them under a partial view states the wrong
    denominator without saying so.
    """
    if not unreachable or os.environ.get("ML4T_OUTPUT_DIR"):
        return
    missing = "\n".join(
        f"  {DISPLAY_NAMES.get(cs, cs)}: no features/ at {display_path(path.parent)}"
        for cs, path in unreachable
    )
    raise RuntimeError(
        f"{len(unreachable)} of {n_case_studies} case studies are not wired into this "
        f"checkout, so their feature panels cannot be read and cannot be distinguished "
        f"from case studies that engineered none:\n{missing}\n"
        "Re-run from a checkout that links every case study's features/ directory."
    )


def load_feature_info(features_path: Path) -> dict:
    """Load feature summary by introspecting financial.parquet schema."""
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
unreachable: list[tuple[str, Path]] = []

for cs in CASE_STUDIES:
    state, panel_path = resolve_feature_panel(cs)
    if state == "readable":
        result = load_feature_info(panel_path)
        all_results[cs] = result
        evaluated[cs] = result
    elif state == "awaiting":
        awaiting.append(cs)
    else:
        unreachable.append((cs, panel_path))

refuse_a_partial_view(unreachable, len(CASE_STUDIES))

print(f"Case studies with features: {len(evaluated)}/{len(CASE_STUDIES)}")
if evaluated:
    print(f"  Available: {', '.join(DISPLAY_NAMES[cs] for cs in evaluated)}")
if awaiting:
    print(f"  Awaiting features: {', '.join(DISPLAY_NAMES.get(cs, cs) for cs in awaiting)}")
if unreachable:
    print(
        "  Not visible in this checkout: "
        f"{', '.join(DISPLAY_NAMES.get(cs, cs) for cs, _ in unreachable)}"
    )

# %% [markdown]
# ## Feature Inventory Summary
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
# ## Feature Count Comparison
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
        title="Engineered financial features per case study",
        xaxis_title="Case study",
        yaxis_title="Number of features",
        height=450,
    )
    print(
        f"feature counts range from {min(n_features)} to {max(n_features)} "
        f"across {len(n_features)} of {len(CASE_STUDIES)} case studies"
    )
    show_plotly_with_alt(
        fig,
        (
            "A bar chart of the number of engineered financial features in each case "
            "study, with the case studies along the horizontal axis and the count on the "
            "vertical axis. Each bar is annotated with its exact count above it. The bars "
            "are of broadly similar height: the tallest belongs to the futures case study "
            "and the shortest to the crypto perpetuals one, with the rest clustered "
            "between them. No case study has a feature space dramatically larger or "
            "smaller than the others."
        ),
    )
else:
    print("No feature data available.")

# %% [markdown]
# ## Feature Family Distribution
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
                "Feature prefixes appearing in only one case study",
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
                "text": "Feature families by case study, ordered by how many share each",
                "font": {"size": 19},
            },
            height=max(500, len(recurring) * 26 + 280),
            width=max(700, len(cs_list) * 90 + 260),
        )
        show_plotly_with_alt(
            fig,
            (
                "Two stacked panels sharing a case-study axis along the bottom. The upper panel "
                "is a heatmap of feature prefix against case study, one row per prefix, with "
                "each filled cell annotated by how many features that case study has under "
                "that prefix and shaded darker for larger counts; empty cells are blank rather "
                "than zero. Rows are ordered so the prefixes shared by the most case studies "
                "sit at the top: volatility, return, momentum, Sharpe and a residual catch "
                "prefix appear across several columns, while implied-volatility and "
                "variance-premium prefixes appear only in the two options-bearing case "
                "studies, and several prefixes near the bottom appear in one or two columns "
                "only. The lower panel is a bar chart in amber counting the prefixes that "
                "appear in one case study alone, annotated with its count; each of the "
                "seven has some, with the equity ETF and futures studies carrying the most."
            ),
        )
    else:
        print("No family-level data available.")
else:
    print("No feature data available.")

# %% [markdown]
# ## Representative Features Across Case Studies
#
# A sample of each case study's feature space, the first few feature names in schema
# order, to illustrate the engineered inputs. This is an inventory view,
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
# ## Breadth vs IC: The Fundamental Law Perspective
#
# The Fundamental Law of Active Management says:
#
# $$IR \approx IC \times \sqrt{BR}$$
#
# where $BR$ is the number of independent bets, roughly the universe size. The cell below
# works the law through two contrasting cases so the arithmetic is visible rather than
# asserted: a wide universe with a small IC against a narrow one with an IC three times
# larger. Read which of the two reaches the higher information ratio, and by how much.

# %%
# The two contrasting cases the text describes, worked rather than quoted.
LAW_EXAMPLES = [
    ("wide universe, small IC", 0.01, 3000),
    ("narrow universe, larger IC", 0.03, 20),
]
print("IR = IC * sqrt(BR)")
for _label, _ic, _br in LAW_EXAMPLES:
    print(f"  {_label:<28} IC {_ic:.2f}, BR {_br:>5,}  ->  IR {_ic * np.sqrt(_br):.2f}")

# %%
from case_studies.utils.analytics import DATASET_META, load_best_ic_per_family
from case_studies.utils.paired_metrics import _retired_prediction_hashes

if evaluated:
    # `exclude_prediction_hashes`, which this call omitted. `load_best_ic_per_family`'s own
    # docstring says retirement is the usual reason to pass it and that "a retired generation
    # is exactly the kind of row that holds high coverage" - the coverage bar it ranks inside
    # is a maximum over the population, so a superseded generation that scored every decision
    # day clears the bar and then wins on IC. Measured 2026-09-18, three of the nine case
    # studies were topped by a retired prediction set: cme_futures (0.0443 against the live
    # 0.0430), fx_pairs (0.0150 against 0.0149) and us_equities_panel, where it also changed
    # the configuration reported, gbm/leaves_63_huber at 0.0343 against gbm/leaves_63_mae at
    # 0.0311. The estimated information ratios below are computed from these, so all three
    # were wrong by the same amount.
    #
    # Retirement is expanded along (training run, checkpoint) rather than taken as the
    # recorded hashes, because a prediction identity carries its split and the retirement is
    # recorded on the validation population; `_retired_prediction_hashes` is the helper
    # `populate_paired_metrics` and `20_strategy_synthesis/01_aggregate_synthesis` both use,
    # so the three agree by construction rather than by inspection.
    _retired = frozenset().union(*(_retired_prediction_hashes(cs) for cs in CASE_STUDIES))
    print(f"Excluding {len(_retired):,} retired prediction identities from the IC comparison")
    best_ic_df = load_best_ic_per_family(exclude_prediction_hashes=_retired)

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
    # sizemode="area" is what makes the area proportional to the value. Plotly's default
    # reads `size` as a diameter, so passing the ratio there makes the AREA scale with its
    # square and overstates the spread between case studies by that power.
    MAX_MARKER_DIAMETER = 60
    _irs = [brow["estimated_ir"] for brow in breadth_data]
    _sizeref = 2.0 * max(_irs) / (MAX_MARKER_DIAMETER**2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[brow["universe_size"] for brow in breadth_data],
            y=[brow["best_abs_ic"] for brow in breadth_data],
            mode="markers+text",
            text=[brow["case_study"] for brow in breadth_data],
            textposition="top center",
            marker=dict(
                size=_irs,
                sizemode="area",
                sizeref=_sizeref,
                sizemin=4,
                color=COLORS["blue"],
                opacity=0.75,
                line=dict(width=1, color=COLORS["slate"]),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Best model |IC| against universe size, marker area by estimated IR",
        xaxis_title="Universe size (number of instruments, log scale)",
        yaxis_title="Best model |IC|",
        xaxis_type="log",
        height=450,
    )
    show_plotly_with_alt(
        fig,
        (
            "A scatter plot of a case study's best model absolute information coefficient "
            "against the size of its instrument universe, with the horizontal axis on a "
            "logarithmic scale from about twenty instruments to several thousand. Each "
            "case study is one labelled marker whose area encodes its estimated "
            "information ratio, so a larger circle means a higher ratio. The markers do "
            "not line up along any trend: the highest coefficient belongs to a case study "
            "with about a hundred instruments, and two of the largest universes sit at "
            "middling and low coefficients. Marker area tells a different story from "
            "vertical position, with much the largest circles at the right-hand end of the "
            "axis, where the universes are widest."
        ),
    )

# %% [markdown]
# Read the two encodings in that figure separately, because they say different things.
#
# Vertical position is the per-bet information coefficient, and it does not rise with
# universe size. The highest coefficient in the panel comes from a case study of about a
# hundred instruments, and two of the widest universes sit well below it. Nothing here
# supports "more instruments, better signal".
#
# Marker area is the estimated information ratio, and the largest circles are at the
# right-hand end of the axis. That is the law at the top of this section doing its work:
# an information ratio is a coefficient multiplied by the square root of breadth, so a
# universe a hundred times wider turns a coefficient ten times smaller into the same
# ratio. The wide-universe case studies reach competitive ratios on individually weaker
# signals.
#
# Two cautions about reading this as a result. The estimated ratio is the law applied to
# the plotted coefficient rather than a measured backtest quantity, so the relationship
# between area and position is partly arithmetic rather than empirical. And breadth in the
# law means *independent* bets; a universe of three thousand equities that move together
# supplies far fewer than three thousand, which is the correction the law is most often
# used without.

# %% [markdown]
# ## What the Panels Above Show
#
# The notebook aggregates whatever is present in each case study's
# `data/features/financial.parquet` and the model registry. The substantive
# findings, meaning which feature families have predictive content for which label
# and horizon, how many features survive HAC + BH-FDR, and how breadth
# and how breadth interacts with IC magnitude, are produced by the per-case-study evaluation
# notebooks (`13_model_analysis.py` in each case study). This summary
# notebook is a cross-case-study inventory and presentation layer; it does
# not itself compute IC or run multiple-testing correction.
#
# **Next**: See `09_model_based_features/case_study_temporal_summary` for the
# temporal/model-based feature companion view.
# **Book**: Chapter 8.6 discusses combining features and controlling the
# search space to avoid data mining.
