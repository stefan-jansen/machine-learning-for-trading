# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # S&P 500 Equity Option Analytics: Feature Evaluation
#
# Univariate screening of the implied-volatility, realized-volatility, momentum
# and quality features against the forward equity return declared at
# `setup.yaml::labels.primary`, plus the GARCH estimators fitted upstream.
# Produces the per-feature triage decisions the modeling notebooks record.
#
# This is the **authoritative holdout-clean feature evaluation** that
# `03_financial_features.py` and `04_model_based_features.py` defer to: every
# statistic below is computed on the pre-2021 development window only, with the
# 2021 holdout sealed out (see "Seal the holdout window" below). The 2021 holdout
# is spent once, on the single selected model in Ch11 onward.
#
# **Learning Objectives**:
# - Evaluate IV-derived features for standalone predictive power on equity returns
# - Apply HAC adjustment for overlapping 5-day return autocorrelation
# - Control false discovery rate with Benjamini-Hochberg correction
# - Report weak standalone IC plainly rather than overstating marginal signal
#
# **Book Reference**: §7.3 (univariate feature-label evaluation) and §7.4 (search
# accounting and multiple testing). §8.6 is the secondary reference for search
# control; §8.5 is about aggregating cross-cutting feature types and is not this
# stage's chapter home.
#
# **What it reads**: `features/financial.parquet`, `features/model_based.parquet`,
# `labels/fwd_ret_5d.parquet`, and the evaluation block of `config/setup.yaml`.
#
# **What it writes**: `evaluation/triage_ledger.parquet`, read by
# `20_strategy_synthesis/02_feature_evaluation.py`, which puts the nine case
# studies' screens side by side; and `evaluation/ic_timeseries.parquet`, plotted
# below by the IC-through-time figure. Nothing else reads either artifact - the
# model notebooks train on the full feature matrix and let regularization select,
# so a STOP here is a recorded judgment about a feature, not a filter applied on
# the reader's behalf.
#
# **Prerequisites**: `03_financial_features.py` and `04_model_based_features.py`
# must have run.

# %%
"""S&P 500 Equity Option Analytics: Feature Evaluation."""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import yaml
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats, compute_ic_uncertainty
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from scipy.stats import t as student_t

import utils.style as style
from utils.cv_splits import load_evaluation_config
from utils.paths import display_path, get_case_study_dir

# Register the ML4T Plotly template (colorway, fonts, gridlines) as the default
# and expose the book palette so every figure sources color from utils.style.
style.apply_ml4t_style()
COLORS = style.COLORS
GRAY_FILLS = style.GRAY_FILLS


def _normalize_asset_column(df: pl.DataFrame) -> pl.DataFrame:
    if "symbol" in df.columns:
        return df
    msg = f"Expected asset-like column in frame, found columns={df.columns}"
    raise KeyError(msg)


# %% tags=["parameters"]
MAX_SYMBOLS = 0

# %%
CASE_STUDY_ID = "sp500_equity_option_analytics"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
EVAL_DIR = CASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

JOIN_COLS = ["timestamp", "symbol"]
DATE_COL = "timestamp"

# Every bound below is read from `config/setup.yaml`; nothing that decides which
# rows are eligible or where a threshold sits is typed into this notebook.
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
PRIMARY_LABEL = setup["labels"]["primary"]
PRIMARY_LABEL_FILE = f"{PRIMARY_LABEL}.parquet"
# `5D` -> 5 sessions. The horizon sets both the embargo at the seal and the
# Newey-West bandwidth, which is why it is read once and spent twice.
LABEL_HORIZON_SESSIONS = int(str(setup["labels"]["horizons"][PRIMARY_LABEL]).rstrip("Dd"))
HAC_MAXLAGS = LABEL_HORIZON_SESSIONS

MIN_PERIODS = 20  # 415-634 symbols
# Screening thresholds, declared here and named in the prose rather than retyped.
COVERAGE_MIN = 0.70
STALENESS_MAX = 0.50
FDR_ALPHA = 0.05
SIGN_CONSISTENCY_MIN = 0.60
REDUNDANCY_CUT = 0.70
IC_THRESHOLD = 0.005  # Weekly horizon
# Metadata columns carried in the feature parquets that are NOT predictive features
# (mirrors the exclusion in utils.modeling so evaluation and modeling agree).
NON_FEATURE_COLS = {"fold", "is_holdout"}

# %% [markdown]
# ## 0. Load Artifacts & Build Evaluation Panel
#
# Load features (45 IV/equity), temporal (3 GARCH per symbol), and labels.
# Both features and temporal have a symbol column, so we join on [date, symbol].
# The `fold` metadata column in the temporal parquet is not a feature and is
# dropped here (`NON_FEATURE_COLS`), matching how `utils.modeling` builds the
# modeling dataset.

# %%
features = _normalize_asset_column(pl.read_parquet(CASE_DIR / "features" / "financial.parquet"))
temporal = _normalize_asset_column(pl.read_parquet(CASE_DIR / "features" / "model_based.parquet"))

label_df = pl.read_parquet(CASE_DIR / "labels" / PRIMARY_LABEL_FILE)
label_col = [c for c in label_df.columns if c not in ("timestamp", "symbol", "timestamp")][0]

cv_config = load_evaluation_config(CASE_STUDY_ID)

print(f"Features: {features.shape}")
print(f"Temporal: {temporal.shape}")
print(f"Labels: {label_df.shape}, column: {label_col}")

# %%
financial_cols = [c for c in features.columns if c not in JOIN_COLS and c not in NON_FEATURE_COLS]
temporal_cols = [c for c in temporal.columns if c not in JOIN_COLS and c not in NON_FEATURE_COLS]

# Join: features + temporal (both have symbol) + labels
eval_panel = features.join(temporal, on=JOIN_COLS, how="left")
eval_panel = eval_panel.join(label_df, on=JOIN_COLS, how="inner")

all_feature_cols = financial_cols + temporal_cols

if MAX_SYMBOLS > 0:
    top = eval_panel.group_by("symbol").len().sort("len", descending=True).head(MAX_SYMBOLS)
    eval_panel = eval_panel.filter(pl.col("symbol").is_in(top["symbol"]))

# %% [markdown]
# ### Seal the holdout window
#
# 03 and 04 defer to this notebook as *the authoritative holdout-clean feature
# evaluation*. To honor that, every statistic below (coverage/staleness, IC, HAC,
# FDR, fold stability, triage) is computed on the **development window only** --
# sessions strictly before the 2021 holdout, with the final
# `LABEL_HORIZON_SESSIONS` dev sessions embargoed so no 5-day forward label
# reaches across the seal. The 2021 holdout is never read here; it is spent once,
# on the single selected model in Ch11 onward.

# %%
from datetime import date as _date

holdout_start = str(cv_config["holdout_start"])[:10]
holdout_start_date = _date.fromisoformat(holdout_start)

dev_sessions = eval_panel.filter(pl.col(DATE_COL) < holdout_start_date)[DATE_COL].unique().sort()
if dev_sessions.len() > LABEL_HORIZON_SESSIONS:
    embargo_cutoff = dev_sessions[-(LABEL_HORIZON_SESSIONS + 1)]
    eval_panel = eval_panel.filter(pl.col(DATE_COL) <= embargo_cutoff)
else:
    eval_panel = eval_panel.filter(pl.col(DATE_COL) < holdout_start_date)

print(
    f"Holdout sealed at {holdout_start}: evaluating "
    f"{eval_panel[DATE_COL].min()} -> {eval_panel[DATE_COL].max()} "
    f"({eval_panel[DATE_COL].n_unique():,} dev sessions, "
    f"{LABEL_HORIZON_SESSIONS}-session label embargo)"
)

n_rows = len(eval_panel)
n_symbols = eval_panel["symbol"].n_unique()
n_dates = eval_panel[DATE_COL].n_unique()
print(f"\nEval panel: {n_rows:,} rows, {n_symbols} symbols, {n_dates:,} dates")
print(
    f"Features: {len(financial_cols)} financial + {len(temporal_cols)} temporal"
    f" = {len(all_feature_cols)} total"
)
print(f"Label: {label_col}")

# %% [markdown]
# ## Artifact Quality Gate
#
# Verify upstream artifacts are free of critical defects before evaluation. This is
# the artifact gate - infinities, negative prices, extreme returns - and it asks a
# different question from the per-feature coverage and staleness screens below.

# %%
from utils.data_quality import validate_modeling_inputs

validate_modeling_inputs(
    features_df=eval_panel,
    label_df=eval_panel,
    feature_cols=all_feature_cols,
    label_col=label_col,
    join_cols=JOIN_COLS,
    asset_col="symbol",
    max_abs_return=10.0,  # 5-day equity returns (max observed ~6.83, small caps)
    fail_on_critical=True,
)

# %% [markdown]
# ## 1. Correctness Screens
#
# Check coverage and staleness before evaluating predictive power. These are two of
# the four correctness questions book §7.3 lists; timing and lag consistency, and
# mask alignment, are settled upstream where each family's lag is declared.
#
# IV features have reduced coverage for less-liquid names, and the one-session IV
# lag makes every IV feature carry a repeated value on any session the surface did
# not update. Both are properties of the data rather than errors, which is exactly
# why the screen has to be a gate rather than a judgement call: a feature clears
# when its coverage reaches `COVERAGE_MIN` and its staleness stays at or below
# `STALENESS_MAX`, and the figure below shows where every candidate sits against
# those two bounds rather than printing only the ones that failed.

# %%
coverage = {}
staleness = {}

for feat in all_feature_cols:
    col = eval_panel[feat]
    coverage[feat] = col.drop_nulls().len() / n_rows

    unchanged = (
        eval_panel.sort(JOIN_COLS)
        .select((pl.col(feat) == pl.col(feat).shift(1).over("symbol")).alias("same"))["same"]
        .sum()
    )
    staleness[feat] = float(unchanged) / max(n_rows - n_symbols, 1)

correctness = {
    feat: coverage[feat] >= COVERAGE_MIN and staleness[feat] <= STALENESS_MAX
    for feat in all_feature_cols
}
n_pass = sum(correctness.values())
n_fail = len(correctness) - n_pass
print(f"Gate: coverage >= {COVERAGE_MIN:.0%}, staleness <= {STALENESS_MAX:.0%}")
print(f"Correctness gate: {n_pass} PASS, {n_fail} FAIL")

if n_fail > 0:
    fail_df = pl.DataFrame(
        {
            "feature": [f for f, ok in correctness.items() if not ok],
            "coverage": [round(coverage[f], 3) for f, ok in correctness.items() if not ok],
            "staleness": [round(staleness[f], 3) for f, ok in correctness.items() if not ok],
        }
    )
    print(fail_df)

# %%
gate_x = [staleness[f] for f in all_feature_cols]
gate_y = [coverage[f] for f in all_feature_cols]
fig = go.Figure()
for cleared, color, name in (
    (True, COLORS["blue"], "Cleared both bounds"),
    (False, COLORS["copper"], "Failed at least one"),
):
    members = [f for f in all_feature_cols if correctness[f] is cleared]
    if not members:
        continue
    _ = fig.add_trace(
        go.Scatter(
            x=[staleness[f] for f in members],
            y=[coverage[f] for f in members],
            mode="markers",
            marker={"color": color, "size": 9, "opacity": 0.85},
            text=members,
            name=name,
        )
    )
_ = fig.add_hline(
    y=COVERAGE_MIN,
    line={"color": COLORS["neutral"], "width": 1, "dash": "dash"},
    annotation_text="coverage bound",
)
_ = fig.add_vline(
    x=STALENESS_MAX,
    line={"color": COLORS["neutral"], "width": 1, "dash": "dash"},
    annotation_text="staleness bound",
)
fig.update_layout(
    title="Coverage and staleness each rule out candidates, and no feature fails both",
    xaxis_title="Fraction of rows unchanged from the prior session",
    yaxis_title="Fraction of rows non-null",
    height=520,
    width=1000,
    legend={"orientation": "h", "y": -0.18},
)
fig.show()
only_staleness = sum(
    1
    for f in all_feature_cols
    if not correctness[f] and coverage[f] >= COVERAGE_MIN and staleness[f] > STALENESS_MAX
)
only_coverage = sum(
    1
    for f in all_feature_cols
    if not correctness[f] and coverage[f] < COVERAGE_MIN and staleness[f] <= STALENESS_MAX
)
print(f"Failed on staleness alone: {only_staleness}")
print(f"Failed on coverage alone:  {only_coverage}")
print(f"Failed on both:            {n_fail - only_staleness - only_coverage}")

# %% [markdown]
# ## 2. Univariate Association (IC + HAC)
#
# Cross-sectional Spearman IC at every session, averaged, with a Newey-West
# correction over `HAC_MAXLAGS` lags - the label's own horizon, because consecutive
# forward returns of that horizon overlap and their ICs are serially dependent by
# construction. The series is sorted on the time axis before it reaches the
# estimator. Date-level features (identical across symbols) produce zero
# cross-sectional IC by construction and are flagged separately.

# %%
evaluable_features = [f for f in all_feature_cols if correctness[f]]

# Detect date-level features (zero cross-sectional variance)
cs_std_df = eval_panel.group_by(DATE_COL).agg(
    [pl.col(f).std().alias(f) for f in evaluable_features]
)
date_level_features = set()
for feat in evaluable_features:
    mean_std = cs_std_df[feat].drop_nulls().mean()
    if mean_std is not None and mean_std < 1e-10:
        date_level_features.add(feat)

if date_level_features:
    print(f"Date-level features (zero CS variance): {sorted(date_level_features)}")

# %%
# Batch IC computation: one pass over dates, all features at once
cs_features = [f for f in evaluable_features if f not in date_level_features]
cols_needed = [DATE_COL] + cs_features + [label_col]
eval_sub = eval_panel.select(cols_needed).drop_nulls(subset=[label_col])

dates_list = eval_sub[DATE_COL].unique().sort().to_list()
n_total = len(dates_list)

ic_series_data = {feat: [] for feat in cs_features}

for i, dt in enumerate(dates_list):
    cross_section = eval_sub.filter(pl.col(DATE_COL) == dt)
    n_obs = len(cross_section)
    if n_obs < MIN_PERIODS:
        continue

    label_arr = cross_section[label_col].to_numpy()
    label_valid = ~np.isnan(label_arr)

    for feat in cs_features:
        feat_arr = cross_section[feat].to_numpy()
        valid_mask = label_valid & ~np.isnan(feat_arr)
        n_valid = int(valid_mask.sum())
        if n_valid >= MIN_PERIODS:
            ic_val, _ = spearmanr(feat_arr[valid_mask], label_arr[valid_mask])
            if not np.isnan(ic_val):
                ic_series_data[feat].append((dt, float(ic_val), n_valid))

    if (i + 1) % 200 == 0:
        print(f"  IC progress: {i + 1}/{n_total} dates")

print(f"  IC progress: {n_total}/{n_total} dates (done)")

# %%
# Convert to DataFrames and compute HAC stats
ic_results = {}
ic_timeseries = {}
for feat in cs_features:
    data = ic_series_data[feat]
    if len(data) < 20:
        continue
    dates_f, ics_f, nobs_f = zip(*data, strict=False)
    ic_df = pl.DataFrame({DATE_COL: list(dates_f), "ic": list(ics_f), "n_obs": list(nobs_f)})
    hac_stats = compute_ic_hac_stats(ic_df, ic_col="ic", maxlags=HAC_MAXLAGS)
    ic_results[feat] = hac_stats
    ic_timeseries[feat] = ic_df

print(f"IC computed for {len(ic_results)} cross-sectional features")
print(f"Skipped {len(date_level_features)} date-level features")

# %% [markdown]
# ### The IC Series Itself
#
# The mean IC is a scalar summary of a series, and the two patterns that decide
# whether a weak association is usable are visible only in the series: an IC carried
# by one episode, and an IC that changes sign. The left panel draws the daily IC of
# the leading feature under a rolling mean; the right panel puts three ways of
# bounding the mean on one axis for the leading features - the naive interval that
# assumes independent sessions, the Newey-West interval that does not, and the
# block-bootstrap bounds that assume neither a variance formula nor a distribution.
# With a five-session label the sessions overlap heavily, so the gap between the
# grey band and the navy bar is the cost of pretending otherwise.
#
# This is the figure that makes `evaluation/ic_timeseries.parquet` read by the
# notebook that writes it.

# %%
IC_ROLLING_WINDOW = 63  # one quarter of sessions
BOOT_BOUNDS = ("ci_boot_lower", "ci_boot_upper")

leaders = sorted(ic_results, key=lambda name: abs(ic_results[name]["mean_ic"]), reverse=True)[:8]
ic_uncertainty = {
    feature: compute_ic_uncertainty(
        ic_timeseries[feature], horizon=LABEL_HORIZON_SESSIONS, ic_col="ic"
    )
    for feature in leaders
}
leader = leaders[0] if leaders else None
print(f"Leading feature by absolute mean IC: {leader}")


# %%
def interval_arms(features: list[str], lower: str, upper: str) -> dict:
    """Asymmetric Plotly error bars from a pair of interval bounds."""
    return {
        "type": "data",
        "symmetric": False,
        "array": [
            ic_uncertainty[name][upper] - ic_uncertainty[name]["mean_ic"] for name in features
        ],
        "arrayminus": [
            ic_uncertainty[name]["mean_ic"] - ic_uncertainty[name][lower] for name in features
        ],
    }


# %%
if leader:
    leader_series = (
        ic_timeseries[leader]
        .sort(DATE_COL)
        .with_columns(pl.col("ic").rolling_mean(IC_ROLLING_WINDOW).alias("rolling"))
    )
    interval_features = list(reversed(leaders))
    interval_means = [ic_uncertainty[name]["mean_ic"] for name in interval_features]
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        subplot_titles=(
            "Daily IC of the leading feature, under its rolling mean",
            "Mean IC against three ways of bounding it",
        ),
        horizontal_spacing=0.18,
    )
    _ = fig.add_trace(
        go.Scatter(
            x=leader_series[DATE_COL],
            y=leader_series["ic"],
            mode="lines",
            line={"color": COLORS["neutral"], "width": 0.6},
            opacity=0.45,
            name="Daily IC",
        ),
        row=1,
        col=1,
    )
    _ = fig.add_trace(
        go.Scatter(
            x=leader_series[DATE_COL],
            y=leader_series["rolling"],
            mode="lines",
            line={"color": COLORS["blue"], "width": 2},
            name="Rolling mean over one quarter",
        ),
        row=1,
        col=1,
    )
    _ = fig.add_hline(
        y=0, line={"color": COLORS["neutral"], "width": 0.8, "dash": "dash"}, row=1, col=1
    )
    _ = fig.add_trace(
        go.Scatter(
            x=interval_means,
            y=interval_features,
            mode="markers",
            marker={"color": COLORS["neutral"], "size": 1, "opacity": 0.0},
            error_x=interval_arms(interval_features, "ci_naive_lower", "ci_naive_upper")
            | {"color": GRAY_FILLS["tertiary"], "thickness": 11, "width": 0},
            name="Naive interval",
        ),
        row=1,
        col=2,
    )
    _ = fig.add_trace(
        go.Scatter(
            x=interval_means,
            y=interval_features,
            mode="markers",
            marker={"color": COLORS["blue"], "size": 9},
            error_x=interval_arms(interval_features, "ci_hac_lower", "ci_hac_upper")
            | {"color": COLORS["blue"], "thickness": 1.5},
            name="Newey-West interval",
        ),
        row=1,
        col=2,
    )
    _ = fig.add_trace(
        go.Scatter(
            x=[ic_uncertainty[name][bound] for name in interval_features for bound in BOOT_BOUNDS],
            y=[name for name in interval_features for _ in BOOT_BOUNDS],
            mode="markers",
            marker={"color": COLORS["copper"], "size": 8, "symbol": "line-ns-open"},
            name="Block-bootstrap bounds",
        ),
        row=1,
        col=2,
    )
    _ = fig.add_vline(
        x=0, line={"color": COLORS["neutral"], "width": 0.8, "dash": "dash"}, row=1, col=2
    )
    fig.update_layout(
        title="Pricing the overlap widens the intervals; all but one then cover zero",
        height=560,
        width=1150,
        margin={"l": 60, "r": 210},
        legend={"orientation": "h", "y": -0.2},
    )
    fig.update_yaxes(title_text="Cross-sectional Spearman IC", row=1, col=1)
    fig.update_xaxes(title_text="Development session", row=1, col=1)
    fig.update_xaxes(title_text="Mean IC, 95% intervals", row=1, col=2)
    fig.show()

# %% [markdown]
# ### Fold-Level Stability
#
# Sign consistency across the walk-forward folds. A feature whose IC flips sign
# between folds is unreliable regardless of its full-sample t-statistic.
#
# **Where the boundaries come from, and what that costs.** `load_evaluation_config`
# returns the declared walk-forward geometry - `n_splits`, `train_size`, `val_size`,
# the holdout bounds - but **not** a materialized `splits` array, so the branch
# below falls through to an equal partition of the development sessions. Those are
# not the fold boundaries the modeling notebooks train against: an equal split of
# the session index ignores `train_size` and `val_size` entirely. It moves
# `sign_consistency` and `worst_fold_ic` in the ledger, so it is tracked in the
# stage-05 notebook standard rather than changed here. The fallback is announced
# below rather than taken silently, which is the part that was wrong: the comment
# at this site used to claim the declared boundaries were being used.
#
# With `n_splits` folds, sign consistency takes only the values zero, one half and
# one, so a quartile across the folds is not a quantity worth reporting. The figure
# shows the fold means themselves.
#
# The figure also exposes a limit of the rule the ledger applies. Sign consistency
# is scored as the share of folds with a *positive* IC, not the share sharing the
# feature's own direction, so a feature whose folds agree on a negative sign scores
# zero and can never clear the stability arm. Every negative-IC feature below is in
# that position, which is why the count printed above is smaller than the figure
# would suggest.

# %%
cv_splits = cv_config.get("splits", [])
if cv_splits:
    from datetime import date as dt_date

    fold_boundaries = [
        (
            dt_date.fromisoformat(str(s["val_start"])[:10]),
            dt_date.fromisoformat(str(s["val_end"])[:10]),
        )
        for s in cv_splits
    ]
else:
    print(
        "cv_config declares no materialized splits; falling back to an equal "
        "partition of the development sessions"
    )
    all_dates = eval_panel[DATE_COL].unique().sort().to_list()
    n_folds = cv_config.get("n_splits", 2)
    fold_size = len(all_dates) // n_folds
    fold_boundaries = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = min((i + 1) * fold_size - 1, len(all_dates) - 1)
        fold_boundaries.append((all_dates[start_idx], all_dates[end_idx]))

fold_stats = {}
for feat in ic_results:
    fold_ics = []
    ts = ic_timeseries[feat]
    for fold_start, fold_end in fold_boundaries:
        fold_ic = ts.filter((pl.col(DATE_COL) >= fold_start) & (pl.col(DATE_COL) <= fold_end))
        if len(fold_ic) >= 5:
            fold_ics.append(float(fold_ic["ic"].mean()))

    if fold_ics:
        sign_consistency = sum(1 for ic in fold_ics if ic > 0) / len(fold_ics)
        fold_stats[feat] = {
            "n_folds": len(fold_ics),
            "sign_consistency": sign_consistency,
            "worst_fold_ic": min(fold_ics),
            "best_fold_ic": max(fold_ics),
            "median_fold_ic": float(np.median(fold_ics)),
            "fold_ics": fold_ics,
        }

n_consistent = sum(
    1 for stats in fold_stats.values() if stats["sign_consistency"] >= SIGN_CONSISTENCY_MIN
)
print(f"Fold stability computed for {len(fold_stats)} features")
print(f"Sign consistency at or above {SIGN_CONSISTENCY_MIN:.0%}: {n_consistent}/{len(fold_stats)}")

# %%
stability_features = [name for name in reversed(leaders) if name in fold_stats]
if stability_features:
    fig = go.Figure()
    _ = fig.add_trace(
        go.Scatter(
            x=[value for name in stability_features for value in fold_stats[name]["fold_ics"]],
            y=[name for name in stability_features for _ in fold_stats[name]["fold_ics"]],
            mode="markers",
            marker={"color": COLORS["neutral"], "size": 9, "opacity": 0.6},
            name="Fold mean IC",
        )
    )
    _ = fig.add_trace(
        go.Scatter(
            x=[fold_stats[name]["median_fold_ic"] for name in stability_features],
            y=stability_features,
            mode="markers",
            marker={"color": COLORS["blue"], "size": 13, "symbol": "diamond"},
            name="Median fold",
        )
    )
    _ = fig.add_trace(
        go.Scatter(
            x=[fold_stats[name]["worst_fold_ic"] for name in stability_features],
            y=stability_features,
            mode="markers",
            marker={
                "color": COLORS["negative"],
                "size": 12,
                "symbol": "x-thin",
                "line": {"width": 2, "color": COLORS["negative"]},
            },
            name="Lowest fold",
        )
    )
    _ = fig.add_vline(x=0, line={"color": COLORS["neutral"], "width": 0.8, "dash": "dash"})
    fig.update_layout(
        title="Both folds share the sign for every one of the leading features",
        xaxis_title="Mean cross-sectional IC within the fold",
        height=500,
        width=1000,
        margin={"l": 220},
        legend={"orientation": "h", "y": -0.18},
    )
    fig.show()

# %% [markdown]
# ## 3. Search Accounting and Multiple Testing
#
# **The searched set.** A p-value means nothing without the set of tests it came
# out of, so the set is declared before the adjustment is applied. It is the
# implied-volatility, realized-volatility, variance-risk-premium, momentum and
# quality features `03_financial_features` generates, plus the GARCH estimators
# `04_model_based_features` fits, restricted to those clearing the correctness
# screen above. Generation is blind to the label: nothing here was added after
# seeing an IC. Only the primary label is screened; the other declared horizons are
# not tested, so they do not enter the count.
#
# **The adjustment.** Benjamini-Hochberg at `FDR_ALPHA` across that set. Testing
# each candidate at the same nominal level would let the expected count of false
# positives grow with the size of the set; BH controls the expected share of false
# discoveries among the features it rejects instead. The expected false-positive
# count under the unadjusted rule is printed with the three significance counts.
#
# Few or no features clearing the adjustment is a plausible reading of this case
# study rather than a failure of the screen. Implied volatility predicts future
# realized volatility; it is not a directional signal on the underlying equity, and
# whatever it contributes is expected to appear in a multivariate fit.

# %%
feature_names = list(ic_results.keys())
p_values = [ic_results[f]["p_value"] for f in feature_names]

fdr_result = benjamini_hochberg_fdr(p_values, alpha=FDR_ALPHA, return_details=True)

eval_summary = pl.DataFrame(
    {
        "feature": feature_names,
        "source": ["temporal" if f in temporal_cols else "financial" for f in feature_names],
        "ic_mean": [ic_results[f]["mean_ic"] for f in feature_names],
        "hac_se": [ic_results[f]["hac_se"] for f in feature_names],
        "hac_t": [ic_results[f]["t_stat"] for f in feature_names],
        "hac_p": p_values,
        "fdr_p": [float(p) for p in fdr_result["adjusted_p_values"]],
        "fdr_sig": [bool(r) for r in fdr_result["rejected"]],
        "naive_t": [ic_results[f]["naive_t_stat"] for f in feature_names],
    },
    schema_overrides={
        "ic_mean": pl.Float64,
        "hac_se": pl.Float64,
        "hac_t": pl.Float64,
        "hac_p": pl.Float64,
        "fdr_p": pl.Float64,
        "fdr_sig": pl.Boolean,
        "naive_t": pl.Float64,
    },
).sort(pl.col("ic_mean").cast(pl.Float64, strict=False).abs(), descending=True)

# The naive tier has to actually be naive. `compute_ic_hac_stats` returns the
# Newey-West p-value and the naive t-statistic; reading the first as if it were the
# unadjusted one makes the inflation ratio a comparison of a quantity with itself.
naive_p_values = [
    float(2 * student_t.sf(abs(ic_results[f]["naive_t_stat"]), df=ic_results[f]["n_periods"] - 1))
    for f in feature_names
]
n_searched = len(feature_names)
expected_false_positives = FDR_ALPHA * n_searched

n_significant_naive = sum(1 for p in naive_p_values if p < FDR_ALPHA)
n_significant_hac = sum(1 for p in p_values if p < FDR_ALPHA)
n_significant_fdr = int(fdr_result["n_rejected"])

inflation_hac = n_significant_naive / max(n_significant_hac, 1)
inflation_fdr = n_significant_naive / max(n_significant_fdr, 1)

print(f"Searched set: {n_searched} features with a computable IC on {label_col}")
print(f"Expected false positives at the unadjusted level: {expected_false_positives:.1f}")
print(f"Naive significant:            {n_significant_naive}")
print(f"Newey-West significant:       {n_significant_hac}")
print(f"BH-FDR significant:           {n_significant_fdr}")
print(f"Inflation factor (Newey-West): {inflation_hac:.2f}x")
print(f"Inflation factor (BH-FDR):     {inflation_fdr:.2f}x")

# %% [markdown]
# ### Ranking, with the Inference Adjustment Visible
#
# The left panel ranks the leading features by absolute mean IC, coloured by whether
# each cleared the false-discovery adjustment. The right panel plots the Newey-West
# t-statistic against its unadjusted twin for every feature in the searched set: a
# point pulled toward zero off the diagonal is a feature whose apparent significance
# came from the overlap between consecutive five-session returns rather than from
# the size of its IC. Bars run horizontally; at this many feature names a rotated
# vertical axis is not legible.

# %%
RANKED_FEATURES = 25
top = eval_summary.head(min(RANKED_FEATURES, len(eval_summary))).reverse()

fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.52, 0.48],
    subplot_titles=[
        "Mean IC of the leading features",
        "Newey-West against unadjusted t-statistics",
    ],
    horizontal_spacing=0.2,
)
for cleared, color, name in (
    (True, COLORS["blue"], "Cleared BH-FDR"),
    (False, COLORS["amber"], "Did not clear"),
):
    arm = top.filter(pl.col("fdr_sig").fill_null(False) == cleared)
    if not len(arm):
        continue
    _ = fig.add_trace(
        go.Bar(
            x=arm["ic_mean"].to_list(),
            y=arm["feature"].to_list(),
            orientation="h",
            marker_color=color,
            name=name,
        ),
        row=1,
        col=1,
    )
_ = fig.add_vline(
    x=0, line={"color": COLORS["neutral"], "width": 0.8, "dash": "dash"}, row=1, col=1
)

max_t = (
    max(
        eval_summary["naive_t"].cast(pl.Float64, strict=False).abs().max() or 1.0,
        eval_summary["hac_t"].cast(pl.Float64, strict=False).abs().max() or 1.0,
    )
    * 1.1
)
_ = fig.add_trace(
    go.Scatter(
        x=[-max_t, max_t],
        y=[-max_t, max_t],
        mode="lines",
        line={"dash": "dash", "color": COLORS["neutral"], "width": 1},
        showlegend=False,
    ),
    row=1,
    col=2,
)
for cleared, color, name in (
    (True, COLORS["blue"], "Cleared BH-FDR"),
    (False, COLORS["amber"], "Did not clear"),
):
    arm = eval_summary.filter(pl.col("fdr_sig").fill_null(False) == cleared)
    if not len(arm):
        continue
    _ = fig.add_trace(
        go.Scatter(
            x=arm["naive_t"].to_list(),
            y=arm["hac_t"].to_list(),
            mode="markers",
            marker={"color": color, "size": 8, "opacity": 0.85},
            text=arm["feature"].to_list(),
            name=name,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
fig.update_layout(
    title="Overlapping five-session returns pull the t-statistics toward zero",
    height=620,
    width=1150,
    margin={"l": 210},
    legend={"orientation": "h", "y": -0.14},
)
# Two traces split the ranking by outcome, so the y order has to be restated or the
# categories fall in trace order and the ranking the panel exists to show is lost.
fig.update_yaxes(categoryorder="array", categoryarray=top["feature"].to_list(), row=1, col=1)
fig.update_xaxes(title_text="Mean cross-sectional Spearman IC", row=1, col=1)
fig.update_xaxes(title_text="Unadjusted t", row=1, col=2)
fig.update_yaxes(title_text="Newey-West t", row=1, col=2)
fig.show()

# %% [markdown]
# ## 4. Shape Diagnostics
#
# Does the mean forward return spread monotonically across feature quintiles? A
# monotone profile is compatible with the ranking-based mapping the strategy
# applies; a high IC with a flat or U-shaped profile is not, and usually means the
# correlation is outlier-driven.
#
# **How the bins are assigned, and what that costs.** The edges come from the
# pooled development sample rather than from within each session, so a bar mixes
# cross-sectional position with time-series level and every other session's
# distribution helps set this session's edges. Book §7.3 specifies the within-date
# construction, which is what the companion IC already uses. The divergence moves
# the `monotonicity` column of the ledger and is tracked in the stage-05 notebook
# standard rather than changed here. Read the profiles below as pooled shape, not
# as a within-session sort.

# %%
N_QUANTILES = 5
top_features_for_shape = eval_summary.filter(pl.col("fdr_sig").fill_null(False))[
    "feature"
].to_list()[:15]
if not top_features_for_shape:
    top_features_for_shape = eval_summary.head(10)["feature"].to_list()

monotonicity_scores = {}
quantile_spreads = {}

for feat in top_features_for_shape:
    valid = eval_panel.select([feat, label_col]).drop_nulls()
    if len(valid) < N_QUANTILES * 20:
        continue

    valid = valid.with_columns(
        pl.col(feat)
        .qcut(N_QUANTILES, labels=[f"Q{i + 1}" for i in range(N_QUANTILES)])
        .alias("quantile")
    )
    q_means = valid.group_by("quantile").agg(pl.col(label_col).mean()).sort("quantile")
    means = q_means[label_col].to_list()
    spread = means[-1] - means[0]
    quantile_spreads[feat] = {"q_means": means, "spread": spread}

    mono_corr, _ = spearmanr(range(len(means)), means)
    monotonicity_scores[feat] = float(mono_corr)

print(f"Shape analysis for {len(quantile_spreads)} features")

# %%
if quantile_spreads:
    n_show = min(6, len(quantile_spreads))
    feats_to_show = list(quantile_spreads.keys())[:n_show]
    n_rows_fig = (n_show + 2) // 3
    fig = make_subplots(
        rows=n_rows_fig,
        cols=3,
        subplot_titles=[
            f"{feat} (monotonicity {monotonicity_scores[feat]:+.2f})" for feat in feats_to_show
        ],
        vertical_spacing=0.22,
    )
    for idx, feat in enumerate(feats_to_show):
        r, c = divmod(idx, 3)
        q_means = quantile_spreads[feat]["q_means"]
        _ = fig.add_trace(
            go.Bar(
                x=[f"Q{i + 1}" for i in range(len(q_means))],
                y=q_means,
                marker_color=[
                    COLORS["negative"],
                    COLORS["copper"],
                    GRAY_FILLS["muted"],
                    COLORS["slate"],
                    COLORS["positive"],
                ],
                showlegend=False,
                text=[f"{m:.4f}" for m in q_means],
                textposition="outside",
            ),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(
        title="No leading feature spreads returns monotonically across its quintiles",
        height=270 * n_rows_fig + 90,
        width=980,
    )
    fig.update_yaxes(title_text="Mean forward return", col=1)
    fig.show()
    print("Quintile spread, Q5 minus Q1, per feature:")
    for feat in feats_to_show:
        print(
            f"  {feat:28s} spread={quantile_spreads[feat]['spread']:+.5f}"
            f"  monotonicity={monotonicity_scores[feat]:+.2f}"
        )

# %% [markdown]
# ## 5. Redundancy & Feature Families


# %%
def assign_feature_family(col: str) -> str:
    """Map feature name to family - canonical prefix-based logic matching 03_financial_features.py."""
    if col.startswith(("iv_30", "iv_7", "iv_90", "d_iv", "iv_mom")):
        return "iv_level_dynamics"
    elif col.startswith(("skew", "term", "iv_30_put", "iv_30_call")):
        return "skew_term"
    elif col.startswith(("rv_", "ivrv", "vrp", "gk_vol", "vol_of_vol", "realized_skew")):
        return "vrp"
    elif col.endswith("_rank"):
        return "cross_sectional_rank"
    elif col.startswith("mom_"):
        return "momentum"
    elif col.startswith(("qc_", "spread_")):
        return "quality"
    elif col.startswith("garch_"):
        return "temporal_garch"
    return "other"


families = {feat: assign_feature_family(feat) for feat in all_feature_cols}

for feat in temporal_cols:
    families[feat] = assign_feature_family(feat)

# %%
# Pairwise correlation
sample_step = max(1, n_dates // 200)
sample_dates = eval_panel[DATE_COL].unique().sort().to_list()[::sample_step]
corr_data = (
    eval_panel.filter(pl.col(DATE_COL).is_in(sample_dates)).select(evaluable_features).to_pandas()
)
corr_matrix = corr_data.corr(method="spearman")

high_corr_pairs = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if abs(corr_matrix.iloc[i, j]) > REDUNDANCY_CUT:
            high_corr_pairs.append((cols[i], cols[j], float(corr_matrix.iloc[i, j])))

print(f"Redundancy cut: |rank correlation| > {REDUNDANCY_CUT}")
print(f"Feature pairs above the cut: {len(high_corr_pairs)}")

# %%
# Family-level IC summary
family_ic = {}
fdr_sig_set = set(eval_summary.filter(pl.col("fdr_sig").fill_null(False))["feature"].to_list())

for feat in ic_results:
    fam = families.get(feat, "other")
    family_ic.setdefault(fam, []).append(
        {
            "feature": feat,
            "ic": ic_results[feat]["mean_ic"],
            "fdr_sig": feat in fdr_sig_set,
        }
    )

family_summary = {}
for fam, feats in sorted(family_ic.items()):
    ics = [f["ic"] for f in feats if f["ic"] is not None]
    n_sig = sum(1 for f in feats if f["fdr_sig"])
    family_summary[fam] = {
        "n_features": len(feats),
        "avg_abs_ic": float(np.mean([abs(ic) for ic in ics])) if ics else 0.0,
        "avg_ic": float(np.mean(ics)) if ics else 0.0,
        "n_fdr_sig": n_sig,
    }

if family_summary:
    fam_df = pl.DataFrame([{"family": fam, **stats} for fam, stats in family_summary.items()]).sort(
        "avg_abs_ic", descending=True
    )
    print(fam_df)
else:
    fam_df = pl.DataFrame()
    print("No features passed IC evaluation threshold")

# %% [markdown]
# ### The Strongest Redundant Pairs
#
# Ranked pairs rather than the full matrix: the question the section asks is which
# two candidates are the same evidence entered twice, and a full heatmap answers it
# only if the reader can still read the tick labels. Each bar names both members
# and the family each belongs to, so a pair inside one family and a pair spanning
# two are distinguishable - the second kind is what a family-by-family selection
# would miss.

# %%
RANKED_PAIRS = 20
if high_corr_pairs:
    ranked = sorted(high_corr_pairs, key=lambda pair: abs(pair[2]), reverse=True)[:RANKED_PAIRS]
    pair_labels = [
        f"{left} ({families.get(left, 'other')})  <->  {right} ({families.get(right, 'other')})"
        for left, right, _ in ranked
    ]
    pair_values = [value for _, _, value in ranked]
    cross_family = sum(
        families.get(left, "other") != families.get(right, "other") for left, right, _ in ranked
    )
    fig = go.Figure(
        go.Bar(
            x=list(reversed(pair_values)),
            y=list(reversed(pair_labels)),
            orientation="h",
            marker_color=[
                COLORS["blue"] if value > 0 else COLORS["copper"] for value in reversed(pair_values)
            ],
            text=[f"{value:+.2f}" for value in reversed(pair_values)],
            textposition="inside",
        )
    )
    fig.update_layout(
        title="Redundancy sits in the volatility and momentum columns, not the IV surface",
        xaxis_title="Spearman rank correlation on sampled development sessions",
        height=640,
        width=1150,
        margin={"l": 480},
        showlegend=False,
    )
    fig.show()
    print(f"Of the {len(ranked)} strongest pairs, {cross_family} span two families")

# %% [markdown]
# ## 6. Triage & Handoff
#
# | Decision | Criteria |
# |----------|----------|
# The book's Table 7.2 rule, with every bound taken from the configuration cell.
# The promotion rule is a **disjunction**, so a feature reaches PROCEED on either
# arm and the promoted count can exceed the count that cleared the adjustment.
# Which arm fired is recorded in the ledger's `note` column.
#
# | Decision | Criteria |
# |----------|----------|
# | **PROCEED** | cleared BH-FDR, **or** sign consistency at least `SIGN_CONSISTENCY_MIN` and absolute mean IC at least `IC_THRESHOLD` |
# | **STOP** | failed the correctness screen: coverage below `COVERAGE_MIN`, or staleness above `STALENESS_MAX` |
# | **REVISE** | everything else, including the features carrying no cross-sectional variation |
#
# `monotonicity` is written to the ledger as evidence and is **not** read by the
# rule above.

# %%
triage = {}
for feat in all_feature_cols:
    if not correctness[feat]:
        triage[feat] = ("STOP", "correctness_fail")
        continue

    if feat in date_level_features:
        triage[feat] = ("REVISE", "date_level_feature")
        continue

    if feat not in ic_results:
        triage[feat] = ("REVISE", "insufficient_data")
        continue

    is_fdr_sig = feat in fdr_sig_set
    sign_con = fold_stats.get(feat, {}).get("sign_consistency", 0)
    abs_ic = abs(ic_results[feat]["mean_ic"])

    if is_fdr_sig:
        triage[feat] = ("PROCEED", "fdr_significant")
    elif sign_con >= SIGN_CONSISTENCY_MIN and abs_ic >= IC_THRESHOLD:
        triage[feat] = ("PROCEED", "stable_and_above_threshold")
    else:
        triage[feat] = ("REVISE", "not_significant_standalone")

# %%
# Build triage ledger
ledger_rows = []
for feat in all_feature_cols:
    decision, note = triage[feat]
    row = {
        "feature": feat,
        "family": families.get(feat, "other"),
        "source": "temporal" if feat in temporal_cols else "financial",
        "ic_mean": ic_results.get(feat, {}).get("mean_ic"),
        "hac_t": ic_results.get(feat, {}).get("t_stat"),
        "hac_p": ic_results.get(feat, {}).get("p_value"),
        "fdr_p": None,
        "fdr_sig": False,
        "sign_consistency": fold_stats.get(feat, {}).get("sign_consistency"),
        "worst_fold_ic": fold_stats.get(feat, {}).get("worst_fold_ic"),
        "monotonicity": monotonicity_scores.get(feat),
        "coverage": coverage[feat],
        "staleness": staleness[feat],
        "decision": decision,
        "note": note,
    }
    match = eval_summary.filter(pl.col("feature") == feat)
    if len(match) > 0:
        row["fdr_p"] = float(match["fdr_p"][0])
        row["fdr_sig"] = bool(match["fdr_sig"][0])
    ledger_rows.append(row)

triage_ledger = pl.DataFrame(ledger_rows)
triage_ledger.write_parquet(EVAL_DIR / "triage_ledger.parquet")
print(f"Triage ledger saved: {display_path(EVAL_DIR / 'triage_ledger.parquet')}")
print(triage_ledger.group_by("decision").len().sort("decision"))

# %%
# Save IC time series (long format)
ic_ts_frames = []
for feat, ts in ic_timeseries.items():
    ic_ts_frames.append(ts.with_columns(pl.lit(feat).alias("feature")))

if ic_ts_frames:
    ic_ts_all = pl.concat(ic_ts_frames)
    ic_ts_all.write_parquet(EVAL_DIR / "ic_timeseries.parquet")
    print(f"IC time series saved: {display_path(EVAL_DIR / 'ic_timeseries.parquet')}")

# %%
proceed_features = sorted(f for f, (d, _) in triage.items() if d == "PROCEED")
revise_features = [f for f, (d, _) in triage.items() if d == "REVISE"]
stop_features = [f for f, (d, _) in triage.items() if d == "STOP"]

print(f"TRIAGE SUMMARY: {CASE_STUDY_ID}")
print(f"  PROCEED: {len(proceed_features)} features")
print(f"  REVISE:  {len(revise_features)} features")
print(f"  STOP:    {len(stop_features)} features")
print(triage_ledger.group_by("note").len().sort("len", descending=True))

# %% [markdown]
# ### The Funnel
#
# What was searched, what survived each stage, and what each arm of the promotion
# rule contributed. Drawn rather than printed, because the width of the drop
# between consecutive bars is what a reader is meant to weigh, and separating the
# two arms is what shows whether a rule written as a disjunction behaves like one.

# %%
arm_counts = {
    note: len([f for f, (d, n) in triage.items() if d == "PROCEED" and n == note])
    for note in ("fdr_significant", "stable_and_above_threshold")
}
funnel_stages = [
    ("Candidates generated upstream", len(all_feature_cols), COLORS["neutral"]),
    ("Cleared the correctness screen", n_pass, COLORS["slate"]),
    ("Computable cross-sectional IC", n_searched, COLORS["copper"]),
    ("Cleared BH-FDR", n_significant_fdr, COLORS["amber"]),
    ("Promoted on the adjustment arm", arm_counts["fdr_significant"], COLORS["blue"]),
    (
        "Promoted on the stability arm",
        arm_counts["stable_and_above_threshold"],
        COLORS["blue_light"],
    ),
]
fig = go.Figure(
    go.Bar(
        x=[count for _, count, _ in funnel_stages],
        y=[stage for stage, _, _ in funnel_stages],
        orientation="h",
        marker_color=[color for _, _, color in funnel_stages],
        text=[str(count) for _, count, _ in funnel_stages],
        textposition="outside",
    )
)
fig.update_layout(
    title="Every promotion here comes from the stability arm, not the adjustment",
    xaxis_title="Features",
    height=460,
    width=1000,
    margin={"l": 260, "r": 80},
    showlegend=False,
)
fig.update_yaxes(autorange="reversed")
fig.show()
print("Promoted features, with the arm that promoted them:")
for f in proceed_features:
    print(
        f"  {f:28s} IC={ic_results[f]['mean_ic']:+.4f}"
        f"  Newey-West t={ic_results[f]['t_stat']:+.2f}"
        f"  [{families.get(f, '?')}, {triage[f][1]}]"
    )

# %% [markdown]
# ## Key Takeaways
#
# ### What the screen does
#
# 1. **Seal first.** Every statistic above is computed on development sessions
#    only, with the last `LABEL_HORIZON_SESSIONS` of them embargoed so no forward
#    label settles across the boundary. The holdout is spent once, later, on one
#    model.
# 2. **Screen the definition before the signal.** Coverage and staleness are
#    properties a feature has whether or not it predicts anything, and a stale
#    column can carry an IC that is an artifact of the carry-forward.
# 3. **Declare the searched set before the p-value.** The adjustment is
#    interpretable only against the set of tests it came from, and that set is
#    fixed by the upstream generation rules, not by what this notebook found.
# 4. **A weak result is a result.** Implied volatility predicts future realized
#    volatility, not the direction of the underlying. Reporting that plainly is the
#    point of a univariate screen; whatever these features contribute is expected
#    to appear in a multivariate fit, and this notebook does not test that claim.
#
# ### Known limitations of this screen
#
# - The fold boundaries are an equal partition of the development sessions, not the
#   declared walk-forward geometry, because `cv_config` carries no materialized
#   splits. `sign_consistency` and `worst_fold_ic` in the ledger reflect that.
# - Sign consistency counts folds with a positive IC rather than folds sharing the
#   feature's own direction, so a stably negative feature cannot clear the
#   stability arm.
# - Quantile bins are assigned over the pooled sample rather than within each
#   session.
# - The IC is screened on the primary label only. The case study ships a ten-session
#   forward return and a risk-adjusted five-session variant, and neither is
#   evaluated here, so the horizon profile book §7.3 asks for is not drawn.
# - Univariate triage is necessary and not sufficient; nothing here claims
#   tradability.
#
# **Next**: `06_linear.py` tests whether these features add incremental value
# beyond price-only features in a multivariate ridge model.
