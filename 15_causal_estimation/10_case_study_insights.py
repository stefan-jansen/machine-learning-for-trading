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
# # Case Study Insights: Causal Estimation
#
# **Docker image**: `ml4t`
#
# **Purpose**: compare the orthogonalized treatment effects registered by
# the nine case studies without treating unlike effect units as a ranking.
#
# **Learning objectives**
#
# - Read DML effects together with HAC uncertainty and block-permutation refutation
# - Compare naive OLS coefficients with orthogonalized estimates
# - Separate multiple labels from genuinely different forecast horizons
# - Interpret causal evidence as an identification diagnostic, not a trading signal
#
# **Book reference**: Sections 15.6 and 15.7.
#
# **Prerequisites**: each case study has an intact `run_log/registry.db`, and at least one of
# them has registered a causal run. A registry whose `causal_runs` table is empty is reported
# as a case study whose causal stage has not run rather than failing the notebook; every
# registry empty stops it, because there is nothing left to compare. The notebook reads those
# registries without modifying them.

# %%
"""Case Study Insights: Causal Estimation from registered DML results."""

import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import Markdown, display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from case_studies.utils.analytics import (
    CASE_STUDY_IDS,
    DATASET_META,
    PRIMARY_LABELS,
    SHORT_NAMES,
    registry_path,
)
from utils.style import COLORS, add_message_title, show_with_alt

# %% tags=["parameters"]
SIG_T = 1.96

# %% [markdown]
# ## 1. Registry contract
#
# Each case study contributes exactly one row per registered label. The loader opens
# SQLite in immutable read-only mode, checks integrity, and rejects duplicate labels.
# The execution evidence pins the nine registry hashes used for publication.

# %%
TREATMENT_LABELS = {
    "etfs": "Momentum (skip-recent 6m/1m)",
    "crypto_perps_funding": "Premium z-score (14d)",
    "cme_futures": "Carry (%)",
    "fx_pairs": "FX momentum (skip-recent)",
    "us_equities_panel": "12-1 momentum",
    "us_firm_characteristics": "12-2 momentum",
    "sp500_equity_option_analytics": "IV-RV spread",
    "nasdaq100_microstructure": "Signed volume share",
    "sp500_options": "Variance risk premium (21d)",
}

CASE_ORDER = {case_study: rank for rank, case_study in enumerate(CASE_STUDY_IDS)}

# %% [markdown]
# The loader returns the current causal row per label from one registry. A refit does not
# overwrite the row it replaces: it writes a new row whose `supersedes_hash` names the old
# one, so the registry keeps the history and the query has to exclude every superseded hash
# to get one row per label. A missing registry, a failed integrity check and a label still
# carrying two live rows all stop the notebook, because each of those is a broken registry.
# A registry that is intact and simply holds no causal row is a different thing: that case
# study has not run its causal stage yet, and the loader returns an empty frame so the
# section below can name it rather than the notebook failing on it.
#
# Supersession is one of three conditions the case-study code uses to decide what a reader
# resolves: a current row also has to carry the current identity version and the execution
# tier asked for. This notebook reads the registries as files and implements the supersession
# condition alone, so it would admit a row written under an older identity version or at a
# preview tier. On the registries as they stand the two rules select the same rows.
# `current_causal_identities` in `case_studies/utils/registry/store.py` is the authority, and
# a reader who needs the full rule should call it rather than copy the query below.


# %%
def _load_causal_runs(case_study: str) -> pl.DataFrame:
    """Load one immutable causal row per label from a case-study registry."""
    db_path = registry_path(case_study).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Missing registry for {case_study}: {db_path}")

    uri = f"file:{db_path}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Registry integrity failed for {case_study}: {integrity}")
        rows = connection.execute(
            "SELECT causal_hash, label, treatment, confounders_json, n_folds, "
            "embargo, n_obs, dml_effect, dml_se_hac, p_value_hac, naive_effect, "
            "confounding_bias_pct, refutation_p, created_at FROM causal_runs "
            "WHERE causal_hash NOT IN ("
            "  SELECT supersedes_hash FROM causal_runs WHERE supersedes_hash IS NOT NULL"
            ") ORDER BY label"
        ).fetchall()

    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame([dict(row) for row in rows], infer_schema_length=None)
    duplicates = frame.group_by("label").len().filter(pl.col("len") != 1)
    if not duplicates.is_empty():
        raise RuntimeError(
            f"Ambiguous causal labels for {case_study}, two rows neither of which is "
            f"superseded: {duplicates}"
        )
    return frame


# %% [markdown]
# Enrichment adds display metadata and derives the HAC statistic, confidence interval,
# confounder count, and refutation classification directly from each registered row.


# %%
def _enrich_causal(frame: pl.DataFrame, case_study: str) -> pl.DataFrame:
    """Add display metadata and derived uncertainty fields."""
    return frame.with_columns(
        case_study=pl.lit(case_study),
        case_order=pl.lit(CASE_ORDER[case_study]),
        short_name=pl.lit(SHORT_NAMES[case_study]),
        frequency=pl.lit(DATASET_META[case_study]["frequency"]),
        primary_label=pl.lit(PRIMARY_LABELS[case_study]),
        treatment_label=pl.lit(TREATMENT_LABELS[case_study]),
        n_confounders=pl.col("confounders_json").map_elements(
            lambda value: len(json.loads(value or "[]")), return_dtype=pl.Int64
        ),
        t_hac=pl.when(pl.col("dml_se_hac") > 0)
        .then(pl.col("dml_effect") / pl.col("dml_se_hac"))
        .otherwise(None),
        ci_lo=pl.col("dml_effect") - SIG_T * pl.col("dml_se_hac"),
        ci_hi=pl.col("dml_effect") + SIG_T * pl.col("dml_se_hac"),
        refutation_class=pl.col("refutation_p").map_elements(
            lambda value: None if value is None else ("Passes" if value < 0.05 else "Fails"),
            return_dtype=pl.Utf8,
        ),
    )


# %% [markdown]
# Loading every registry at once makes coverage explicit. A case study that has registered a
# causal row must have exactly one row at its primary label, and the count every chart and
# sentence below divides by is the number of case studies that loaded - never the number
# that exist. The two are printed side by side so a partial chart cannot be read as a
# complete one.

# %%
all_frames = []
missing_causal = []
for case_study in CASE_STUDY_IDS:
    raw_frame = _load_causal_runs(case_study)
    if raw_frame.is_empty():
        missing_causal.append(case_study)
        continue
    case_frame = _enrich_causal(raw_frame, case_study)
    primary_count = case_frame.filter(pl.col("label") == PRIMARY_LABELS[case_study]).height
    if primary_count != 1:
        raise RuntimeError(
            f"Expected one primary row for {case_study}/{PRIMARY_LABELS[case_study]}, "
            f"found {primary_count}"
        )
    all_frames.append(case_frame)

if not all_frames:
    raise RuntimeError("No case study has registered a causal run; nothing to compare")

all_causal = pl.concat(all_frames, how="diagonal_relaxed")
primary_df = all_causal.filter(pl.col("label") == pl.col("primary_label")).sort("case_order")
n_expected = primary_df.height
print(f"Causal coverage: {n_expected} of {len(CASE_STUDY_IDS)} case studies")
if missing_causal:
    print(
        "No causal row registered yet, so absent from every chart and count below: "
        + ", ".join(SHORT_NAMES[case_study] for case_study in missing_causal)
    )

# %%
coverage_df = primary_df.select(
    "short_name",
    "frequency",
    "label",
    "causal_hash",
    "treatment",
    "n_confounders",
    "n_obs",
).rename({"label": "primary_label"})
coverage_df

# %% [markdown]
# The table is a provenance surface as well as a coverage check: it shows the selected
# causal hash, treatment, sample size, and primary label for every panel.

# %% [markdown]
# ## 2. Primary-label effects and HAC uncertainty
#
# Effect units differ by panel, so the table reports effects and confidence intervals in
# native units while the chart compares their dimensionless HAC statistics. Filled
# markers identify intervals that exclude zero at the conventional two-sided threshold.

# %%
forest_df = primary_df.sort("t_hac")
forest_y = np.arange(forest_df.height)
forest_effect = forest_df["dml_effect"].to_numpy()
forest_t = forest_df["t_hac"].to_numpy()
forest_significant = np.abs(forest_t) > SIG_T
n_sig = int(forest_significant.sum())
n_pos_sig = int(((forest_effect > 0) & forest_significant).sum())
n_neg_sig = int(((forest_effect < 0) & forest_significant).sum())

# %%
forest_df.select(
    "short_name",
    "label",
    "dml_effect",
    "ci_lo",
    "ci_hi",
    "t_hac",
    "p_value_hac",
    "refutation_p",
)

# %%
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.hlines(
    forest_y,
    0,
    forest_t,
    color=COLORS["neutral"],
    linewidth=1.0,
)
ax.scatter(
    forest_t[forest_significant],
    forest_y[forest_significant],
    s=65,
    color=COLORS["blue"],
    label=f"|t| > {SIG_T:g}",
    zorder=3,
)
ax.scatter(
    forest_t[~forest_significant],
    forest_y[~forest_significant],
    s=65,
    facecolor=COLORS["bg_light"],
    edgecolor=COLORS["blue"],
    label=f"|t| <= {SIG_T:g}",
    zorder=3,
)
ax.axvline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
ax.axvline(SIG_T, color=COLORS["amber"], linewidth=0.8, linestyle=":")
ax.axvline(-SIG_T, color=COLORS["amber"], linewidth=0.8, linestyle=":")
ax.set_yticks(forest_y, forest_df["short_name"].to_list())
ax.set_xlabel("HAC t-statistic (dimensionless)")
add_message_title(ax, "HAC t-statistic of each case study's primary effect")
ax.legend(frameon=False, loc="best")
show_with_alt(
    fig,
    "Forest plot with one row per case study, largest t-statistic at the top, plotting the "
    "Driscoll-Kraay t-statistic of that panel's primary DML effect as a stem running from "
    "zero to a marker. A dashed vertical line marks zero and two dotted lines mark the plus "
    "and minus significance threshold; a marker is filled or open according to whether the "
    "t-statistic clears it, and a legend says which is which.",
)

# %%
sig_names = forest_df.filter(pl.col("t_hac").abs() > SIG_T)["short_name"].to_list()
display(
    Markdown(
        f"The HAC interval excludes zero for **{n_sig} of {n_expected}** panels "
        f"({n_pos_sig} positive and {n_neg_sig} negative): "
        f"**{', '.join(sig_names) if sig_names else 'none'}**. The other panels remain "
        "compatible with a zero orthogonalized effect at this threshold."
    )
)

# %% [markdown]
# ## 3. Naive OLS versus orthogonalized effects
#
# Confounding bias is
#
# $$
# \text{Bias\%}=
# \frac{\hat{\theta}_{\text{naive}}-\hat{\theta}_{\text{DML}}}
# {|\hat{\theta}_{\text{DML}}|}\times 100.
# $$
#
# Its sign is directional, not a magnitude ranking. A coefficient sign reversal is
# reported separately because it changes the qualitative interpretation.

# %%
bias_df = primary_df.filter(pl.col("confounding_bias_pct").is_not_null()).sort(
    pl.col("confounding_bias_pct").abs()
)
bias_values = bias_df["confounding_bias_pct"].to_numpy()
bias_y = np.arange(bias_df.height)
median_abs_bias = float(bias_df["confounding_bias_pct"].abs().median())
n_large_bias = int((bias_df["confounding_bias_pct"].abs() > 50).sum())
max_bias_row = bias_df.sort(pl.col("confounding_bias_pct").abs(), descending=True).row(
    0, named=True
)
reversal_df = primary_df.filter(
    (pl.col("naive_effect") * pl.col("dml_effect") < 0) & pl.col("naive_effect").is_not_null()
)

# %%
fig, ax = plt.subplots(figsize=(9.5, 4.8))
bias_colors = [COLORS["copper"] if value > 0 else COLORS["blue"] for value in bias_values]
ax.barh(bias_y, bias_values, color=bias_colors, alpha=0.85, height=0.6)
ax.axvline(0, color=COLORS["neutral"], linewidth=0.8)
for boundary in (-50, 50):
    ax.axvline(boundary, color=COLORS["neutral"], linewidth=0.7, linestyle=":")
ax.set_yticks(bias_y, bias_df["short_name"].to_list())
ax.set_xlabel("Signed confounding bias (%)")
add_message_title(
    ax,
    "Signed confounding bias of the naive estimate on each primary-label panel",
    subtitle="Positive means the naive estimate is the larger one",
)
ax.legend(
    handles=[
        Patch(color=COLORS["copper"], label="Naive estimate is greater"),
        Patch(color=COLORS["blue"], label="Naive estimate is smaller"),
    ],
    frameon=False,
    loc="best",
)
show_with_alt(
    fig,
    "Horizontal bar chart with one bar per case study, ordered by absolute size, giving the "
    "signed confounding bias of the naive estimate as a percentage of the adjusted one. A "
    "solid line marks zero and "
    "dotted lines mark plus and minus fifty percent; bars are coloured by whether the naive "
    "estimate is the larger or the smaller of the two, and a legend says which.",
)

# %%
reversal_df.select(
    "short_name",
    "treatment_label",
    pl.col("naive_effect").alias("naive"),
    pl.col("dml_effect").alias("dml"),
    "t_hac",
    "refutation_p",
).sort("short_name")

# %%
reversal_names = reversal_df["short_name"].to_list()
display(
    Markdown(
        f"The median absolute bias is **{median_abs_bias:.1f} percent** and "
        f"**{n_large_bias} of {n_expected}** panels exceed half. The largest absolute bias "
        f"is **{abs(max_bias_row['confounding_bias_pct']):.1f}%** on "
        f"**{max_bias_row['short_name']}**. Naive and DML signs disagree on "
        f"**{reversal_df.height}** panels: **{', '.join(reversal_names) or 'none'}**."
    )
)

# %% [markdown]
# ## 4. Parametric and permutation evidence
#
# The HAC interval and the block-permutation refutation ask different questions. The
# cross-tabulation keeps their two decisions separate rather than collapsing them into one
# pass or fail.
#
# **What the refutation column inherits.** `refutation_p` is read from each registry rather
# than computed here, so it is only as good as the run that wrote it, and the provenance
# stamp says when these rows were read.
#
# A block permutation that compares raw effects is biased toward "passed" by arithmetic.
# Permuting the treatment also frees it from the controls, so the placebo estimator divides
# by a much larger residual variance and its effects come out smaller whether or not there
# is anything to find. `04_dml_crypto_regime` prints how much of its treatment's variance
# its controls leave - under a tenth - and moving its own comparison to
# t-statistics took its permutation p from the floor to the middle of the null.
# ml4t/agent-workspace#1120 carries the same correction into `case_studies/utils/causal.py`,
# the shared runner these registries are written by, and the rows below were refit under the
# corrected runner. That correction has not landed in this checkout: the runner here still
# collects each placebo's effect and compares those, so regenerating a registry from this
# tree reproduces the column as it read before the refit rather than the column below.
#
# Comparing t-statistics cancels that one-directional bias and does not make the test
# calibrated. Measured on twelve synthetic panels with the true effect fixed at exactly
# zero, highly persistent AR(1) confounders that strongly predict the treatment, and forty
# placebo draws each: at the conventional five percent level the raw-effect comparison
# rejects the permutation null on eleven of the twelve and the t-statistic comparison on
# five, and eleven of the twelve observed t-statistics are negative against a true effect
# of zero. A rejection is what the column below records as `Passes`, so on those panels the
# column reads `Passes` for an effect that is exactly zero. The measurement is on
# ml4t/agent-workspace#1120.
#
# The other label carries no more weight. `Fails` says only that the observed statistic was
# not distinguishable from the permutation distribution, which is as easily a short sample
# as an unbiased estimate. Either way the column reports the distance between an estimate
# and its own permutation null, and a biased estimate sits far from that null too.

# %%
HAC_SIG = "HAC clears"
HAC_NOT = "HAC overlaps zero"
TIER_ORDER = ["Passes", "Fails"]
ref_df = primary_df.filter(pl.col("refutation_class").is_not_null()).with_columns(
    hac_class=pl.when(pl.col("p_value_hac") < 0.05).then(pl.lit(HAC_SIG)).otherwise(pl.lit(HAC_NOT))
)
cross_tab = (
    ref_df.group_by(["hac_class", "refutation_class"])
    .agg(
        panels=pl.col("short_name").str.join(", "),
        n=pl.len(),
        ref_p_min=pl.col("refutation_p").min(),
        ref_p_max=pl.col("refutation_p").max(),
    )
    .sort(["hac_class", "refutation_class"])
)
cross_tab

# %%
sig_order = [HAC_SIG, HAC_NOT]
counts = {
    (hac, ref): ref_df.filter(
        (pl.col("hac_class") == hac) & (pl.col("refutation_class") == ref)
    ).height
    for hac in sig_order
    for ref in TIER_ORDER
}
panels = {
    (hac, ref): ref_df.filter((pl.col("hac_class") == hac) & (pl.col("refutation_class") == ref))[
        "short_name"
    ].to_list()
    for hac in sig_order
    for ref in TIER_ORDER
}
matrix = np.array([[counts[(hac, ref)] for ref in TIER_ORDER] for hac in sig_order])

# %%
heatmap = LinearSegmentedColormap.from_list(
    "ml4t_count", [COLORS["silver"], COLORS["amber_light"], COLORS["blue"]]
)
fig, ax = plt.subplots(figsize=(8.5, 4.5))
image = ax.imshow(matrix, cmap=heatmap, aspect="auto", vmin=0)
for row_index, hac in enumerate(sig_order):
    for column_index, refutation in enumerate(TIER_ORDER):
        count = matrix[row_index, column_index]
        names = panels[(hac, refutation)]
        label = f"{count}\n" + "\n".join(names) if names else "0"
        color = COLORS["silver"] if count == matrix.max() and count > 0 else COLORS["blue"]
        ax.text(column_index, row_index, label, ha="center", va="center", color=color)
ax.set_xticks(range(len(TIER_ORDER)), [f"Refutation {value.lower()}" for value in TIER_ORDER])
ax.set_yticks(range(len(sig_order)), sig_order)
add_message_title(ax, "Panels by HAC significance against refutation outcome")
fig.colorbar(image, ax=ax, label="Panels")
show_with_alt(
    fig,
    "Heatmap crossing HAC significance on the vertical axis with the refutation outcome on "
    "the horizontal one. Each cell is shaded by how many panels fall in it and prints that "
    "count above the names of the panels themselves, so a cell holding none reads zero.",
)

# %%
both_clear = panels[(HAC_SIG, "Passes")]
hac_only = panels[(HAC_SIG, "Fails")]
refutation_only = panels[(HAC_NOT, "Passes")]
neither = panels[(HAC_NOT, "Fails")]
display(
    Markdown(
        f"Both tracks clear their threshold for **{len(both_clear)}** panels "
        f"(**{', '.join(both_clear) or 'none'}**). HAC alone clears for "
        f"**{', '.join(hac_only) or 'none'}**; refutation alone clears for "
        f"**{', '.join(refutation_only) or 'none'}**; neither clears for "
        f"**{', '.join(neither) or 'none'}**."
    )
)

# %% [markdown]
# A refutation pass means the observed effect is unusual under the registered
# block-permutation null. It does not remove the unconfoundedness assumption, and would not
# for SP500 Options, where treatment and outcome both depend on the implied-volatility
# surface.

# %% [markdown]
# ## 5. Multiple labels versus multiple horizons
#
# Two labels can share a forecast horizon, so label count is not horizon count. The
# mapping below converts every registered label to trading-day units and fails on an
# unknown label rather than silently dropping it from the plot.

# %%
HORIZON_DAYS: dict[str, float] = {
    "fwd_ret_5m": 5 / (6.5 * 60),
    "fwd_ret_15m": 15 / (6.5 * 60),
    "fwd_ret_60m": 60 / (6.5 * 60),
    "fwd_ret_8h": 1 / 3,
    "fwd_ret_24h": 1.0,
    "fwd_ret_1d": 1.0,
    "fwd_ret_5d": 5.0,
    "fwd_ret_risk_adj_5d": 5.0,
    "fwd_ret_dh_5d": 5.0,
    "fwd_ret_10d": 10.0,
    "fwd_ret_dh_10d": 10.0,
    "fwd_ret_21d": 21.0,
    "fwd_ret_1m": 21.0,
    "fwd_ret_1m_win": 21.0,
    "ret_to_expiry": 21.0,
}

# %%
horizon_df = all_causal.with_columns(
    horizon_days=pl.col("label").replace_strict(HORIZON_DAYS, default=None).cast(pl.Float64)
)
unknown_labels = horizon_df.filter(pl.col("horizon_days").is_null())["label"].unique().to_list()
if unknown_labels:
    raise RuntimeError(f"Unmapped causal labels: {unknown_labels}")

horizon_coverage = (
    horizon_df.group_by(["case_order", "short_name"])
    .agg(n_labels=pl.len(), n_horizons=pl.col("horizon_days").n_unique())
    .sort("case_order")
)
multi_horizon_names = horizon_coverage.filter(pl.col("n_horizons") >= 2)["short_name"].to_list()
same_horizon_multi = horizon_coverage.filter(
    (pl.col("n_labels") >= 2) & (pl.col("n_horizons") == 1)
)["short_name"].to_list()
horizon_coverage

# %%
plot_horizon = horizon_df.filter(pl.col("short_name").is_in(multi_horizon_names))
n_panels = len(multi_horizon_names)
n_columns = 2
n_rows = int(np.ceil(n_panels / n_columns))
if n_panels == 0:
    # Which case studies load is a property of the registries rather than a constant, so
    # every loaded one registering a single horizon is a reachable state, and
    # plt.subplots(0, 2) raises on it.
    print("No loaded case study registers more than one horizon; nothing to plot here.")
else:
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(10, 3.5 * n_rows), squeeze=False)
    for index, short_name in enumerate(multi_horizon_names):
        ax = axes.flat[index]
        panel = plot_horizon.filter(pl.col("short_name") == short_name).sort("horizon_days")
        x = panel["horizon_days"].to_numpy()
        effect = panel["dml_effect"].to_numpy()
        lo = panel["ci_lo"].to_numpy()
        hi = panel["ci_hi"].to_numpy()
        ax.fill_between(x, lo, hi, color=COLORS["amber"], alpha=0.2)
        ax.plot(x, effect, marker="o", color=COLORS["blue"], linewidth=1.5)
        ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(short_name)
        ax.set_xlabel("Horizon (trading days, log scale)")
        ax.set_ylabel("DML effect (panel units)")
    for index in range(n_panels, n_rows * n_columns):
        axes.flat[index].set_visible(False)
    # No explicit y: constrained layout places the suptitle, and y=1.01 pushed it into the
    # per-panel titles, which the render showed running through "CME Futures".
    fig.suptitle("DML effect against label horizon, for panels with more than one")
    show_with_alt(
        fig,
        "A grid of small panels, one per case study that registers more than one label "
        "horizon. Each plots the DML effect against the horizon in trading days on a "
        "logarithmic axis, as a line with a marker at every horizon, inside a shaded band "
        "for the confidence interval, with a dashed horizontal line at zero.",
    )

# %%
horizon_df.sort(["case_order", "horizon_days", "label"]).select(
    "short_name",
    "label",
    "horizon_days",
    "dml_effect",
    "t_hac",
    "refutation_p",
)

# %%
display(
    Markdown(
        f"**{n_panels}** panels contain multiple distinct horizons: "
        f"**{', '.join(multi_horizon_names) or 'none'}**. Multiple labels share one "
        f"horizon on **{len(same_horizon_multi)}** panels: "
        f"**{', '.join(same_horizon_multi) or 'none'}**. Keeping those categories separate "
        "prevents a risk-adjusted label from being misread as a longer forecast horizon."
    )
)

# %% [markdown]
# ## 6. Takeaways
#
# The summary below is generated from the same frames used by the figures, so a registry
# update changes the displayed interpretation together with the numbers.

# %%
takeaway_text = f"""
- **Coverage is explicit.** The notebook loaded one primary row for **{n_expected}** of the
  **{len(CASE_STUDY_IDS)}** case studies and rejected ambiguous labels. Every count below
  divides by the first number. Absent, because no causal row is registered for them yet:
  **{", ".join(SHORT_NAMES[case_study] for case_study in missing_causal) or "none"}**.
- **HAC evidence is selective.** **{n_sig} of {n_expected}** primary effects have
  Driscoll-Kraay intervals that exclude zero: **{", ".join(sig_names) or "none"}**.
- **Orthogonalization is material.** Median absolute confounding bias is
  **{median_abs_bias:.1f}%**; naive and DML signs differ on **{reversal_df.height}** panels.
- **The two uncertainty tracks are complementary.** HAC and refutation both clear their
  threshold on **{len(both_clear)}** panels: **{", ".join(both_clear) or "none"}**.
- **Labels are not horizons.** **{n_panels}** panels have multiple distinct horizons,
  while **{len(same_horizon_multi)}** have multiple labels at one horizon.

These DML estimates diagnose identification assumptions. They do not enter the backtest
stack as predictions. Chapter 16 evaluates trading rules; Chapter 20 may use causal
evidence only after the registry-consumer hold is released.

**Book**: Sections 15.6 and 15.7.
"""
display(Markdown(takeaway_text))
