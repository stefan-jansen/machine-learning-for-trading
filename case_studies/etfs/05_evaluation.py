# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:percent
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
# # ETFs: which features carry information about the forward return
#
# Chapters 8 and 9 built two feature matrices for this case study. `financial.parquet`
# holds arithmetic on past bars - trailing returns, volatility, oscillators. Neither
# notebook asked whether any of it predicts anything, and `04_model_based_features` wrote
# `model_based.parquet`, whose columns are the output of models fitted inside each
# walk-forward window. This notebook asks the question, one feature at a time.
#
# The measurement is the **information coefficient**, IC for short: on each date, rank the
# ETFs by the feature, rank them by the return they went on to earn over the following
# month, and correlate the two rankings. That gives one number per date and a series over
# the development sample. Everything after it is inference on that series - how much of
# its average is left once the overlap between consecutive monthly returns is allowed for,
# how much is left after allowing for having asked the same question of every feature at
# once, and whether the association holds across the walk-forward windows or came out of a
# single favorable stretch.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Measure how well a feature ranks assets against what they went on to earn, date by
#   date, and read the resulting series rather than only its average.
# - Correct a t-statistic for the serial dependence that overlapping forward returns
#   create, and say how much of the apparent significance that correction removes.
# - Adjust a set of simultaneous tests for the number of tests in it, and report the size
#   of the search beside the result, so a reader can judge what a p-value here is worth.
# - Separate a feature that works in every walk-forward window from one that worked in a
#   single period, using the sign the feature takes in each window.
# - Find the pairs of features that are close to the same measurement, so a family's
#   breadth is not read as independent corroboration.
#
# **Book reference**: Chapter 7, Section 7.3 (Univariate feature-label evaluation) and
# Section 7.4 (Search accounting and multiple testing). Section 8.6 is the secondary
# reference for search control.
#
# **Prerequisites**: `02_labels` has written the forward-return labels, and
# `03_financial_features` and `04_model_based_features` have written
# `features/financial.parquet` and `features/model_based.parquet`.
#
# **What it writes.** `evaluation/triage_ledger.parquet`, one row per candidate feature
# carrying its statistics and a PROCEED, REVISE or STOP decision, and
# `evaluation/ic_timeseries.parquet`, the per-date series those statistics summarize. The
# ledger is read by `20_strategy_synthesis/02_feature_evaluation.py`, which sets the nine
# case studies' decisions side by side. The model notebooks from `06_linear` on train on
# the whole feature matrix: this screen is evidence about the features, not a filter
# applied to them.

# %%
"""Feature evaluation - ETFs case study."""

import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats, compute_ic_uncertainty
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

import utils.style as style
from case_studies.utils.feature_engineering import (
    assign_families,
    families_from_config,
    quantile_profile,
)
from utils.artifact_specs import load_setup_config, resolve_label_buffer
from utils.cv_splits import generate_cv_splits
from utils.data_quality import validate_modeling_inputs
from utils.paths import get_case_study_dir

COLORS = style.COLORS
GRAY_FILLS = style.GRAY_FILLS

# %% tags=["parameters"]
# Production defaults
MAX_SYMBOLS = 0

# %% [markdown]
# ## Settings, and what each one decides
#
# Everything the screens, the triage rule and the figures depend on is bound here: from
# `config/setup.yaml` where the case study declares it, and as a named constant where this
# notebook is the one making the choice. A threshold retyped further down would be a
# second source of truth for a decision already made once.
#
# Four of them come from the configuration. The **primary label** is the forward return
# every statistic below is measured against, and its **horizon** sets two other things:
# the width of the seal that keeps the holdout out of this notebook, and the bandwidth of
# the standard-error correction, because a return measured over a month and sampled every
# day overlaps its neighbours for all but one of those days. The **holdout boundary** is
# the date from which no row may inform anything here. The **walk-forward folds** come
# from the same configuration and are re-derived rather than stored.
#
# The rest are this notebook's own judgements, and each one is a place a reader working on
# their own data would choose differently:
#
# - A date needs a minimum number of ETFs quoted before a rank correlation across them
#   means anything; below that the date contributes nothing to the series.
# - A feature has to be present on most rows to be screened at all, and it has to actually
#   change from date to date - a column that repeats yesterday's value for half the panel
#   is describing the feed rather than the market.
# - The false-discovery level is the share of the promotions that are allowed to be false.
# - The exploration arm's threshold is the smallest average rank correlation worth
#   carrying into a monthly rebalance, and the sign-agreement bar is how much of the
#   walk-forward history has to point the same way for that to count as stable.
# - Two features whose ranks agree above the redundancy cut are treated as one piece of
#   evidence rather than two.

# %%
CASE_STUDY_ID = "etfs"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
EVAL_DIR = CASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

SETUP = load_setup_config(CASE_STUDY_ID)
eval_config = SETUP["evaluation"]

PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, PRIMARY_LABEL, SETUP)
assert LABEL_BUFFER, f"No label buffer configured for {PRIMARY_LABEL}"
HAC_MAXLAGS = int(re.match(r"^(\d+)", LABEL_BUFFER).group(1))
LABEL_HORIZON = HAC_MAXLAGS
HOLDOUT_START = date.fromisoformat(eval_config["holdout_start"])

MIN_CROSS_SECTION_TARGET = 10  # ETFs a date needs before its rank correlation is read
IC_THRESHOLD = 0.01  # smallest average IC the exploration arm will promote on
N_QUANTILES = 5  # buckets the shape diagnostic sorts each date into
MIN_COVERAGE = 0.70  # non-null share the correctness gate requires
MAX_STALENESS = 0.50  # unchanged-from-prior-date share the correctness gate allows
FDR_ALPHA = 0.05  # share of promotions allowed to be false discoveries
NAIVE_T = 1.96  # two-sided normal critical value, for the naive-versus-HAC comparison
MIN_SIGN_CONSISTENCY = 0.60  # fold-sign agreement the exploration arm requires
REDUNDANCY_CUT = 0.7  # |rho| above which two features are one piece of evidence

print(
    f"Label: {PRIMARY_LABEL}, a forward return over {LABEL_HORIZON} trading sessions.\n"
    f"Consecutive labels therefore share {LABEL_HORIZON - 1} of those sessions, so the\n"
    f"  standard-error correction is given {HAC_MAXLAGS} sessions of dependence to allow for,\n"
    f"  and the holdout seal is {LABEL_HORIZON} sessions wide.\n"
    f"Holdout: dates from {HOLDOUT_START} onward inform nothing in this notebook.\n"
    f"Walk-forward design: {eval_config['n_splits']} folds, "
    f"{eval_config['train_size']} of training and {eval_config['val_size']} of validation each."
)

# %% [markdown]
# ## A. The panel this is measured on
#
# Three files come in: the two feature matrices and the forward-return label. They are
# joined on the date and the ETF, which is the pair that identifies a row everywhere in
# this case study.
#
# The model-based matrix needs one more key resolved before it can join. Each of its
# columns is the output of a model - a two-state description of whether the broad market
# is calm or stressed, a filtered version of ten reference price series, and a volatility
# model per ETF - and `04_model_based_features` fits every one of them inside a single
# walk-forward window, on that window's training sessions only, then writes the values it
# infers across the whole window. So each row carries the number of the window whose model
# produced it, and the same date and ETF appear once per window.
#
# A **walk-forward window**, or fold, is a pair of adjacent date ranges: a stretch of
# history the model is fitted on, and the stretch that follows it, which the model has not
# seen. A value dated inside the training range was produced by parameters estimated from
# sessions that run past it, so it is not something a reader could have computed at that
# date. Only the values dated inside the validation range are. This notebook therefore
# keeps each model-based row only inside the validation range of the fold that produced
# it, and drops the extra fold `04` appends past the last one - the fold that trains on
# everything before the holdout begins so that the final notebook has features to score
# the holdout with, and which has no validation range on this side of it.
#
# What is left is one value per date and ETF, and **the evaluation panel is the union of
# those validation ranges**, so the two matrices are screened on the same rows.

# %%
JOIN_COLS = ["timestamp", "symbol"]
DATE_COL = "timestamp"

features = pl.read_parquet(CASE_DIR / "features" / "financial.parquet").with_columns(
    pl.col("timestamp").cast(pl.Date)
)
model_based_artifact = pl.read_parquet(CASE_DIR / "features" / "model_based.parquet").with_columns(
    pl.col("timestamp").cast(pl.Date)
)
label_df = pl.read_parquet(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet").with_columns(
    pl.col("timestamp").cast(pl.Date)
)
label_col = [c for c in label_df.columns if c not in ("timestamp", "symbol")][0]

financial_cols = [c for c in features.columns if c not in JOIN_COLS]
model_based_cols = [c for c in model_based_artifact.columns if c not in (*JOIN_COLS, "fold")]

print(
    f"financial.parquet:   {features.height:,} rows, {len(financial_cols)} features, "
    f"{features[DATE_COL].min()} to {features[DATE_COL].max()}\n"
    f"model_based.parquet: {model_based_artifact.height:,} rows, {len(model_based_cols)} features, "
    f"one row per date, ETF and fold\n"
    f"{PRIMARY_LABEL}.parquet:  {label_df.height:,} rows, label column {label_col!r}"
)

# %% [markdown]
# ### What is in the feature set
#
# `config/setup.yaml` declares the case study's feature register: one row per family,
# saying what the family reads, how far back, and whether it is meant to rank ETFs against
# each other or to describe the environment a ranking is formed in.
# `03_financial_features` builds the matrix from that register, and reading the same
# register here is what keeps a feature in the same family on both sides. The model-based
# columns are not in it - the register covers the price-derived matrix - so they are named
# by the model that produced them.

# %%
FAMILY_REGISTER = families_from_config(SETUP)
families = assign_families(financial_cols, FAMILY_REGISTER)


def model_based_family(column: str) -> str:
    """Name a model-based column by the model that produced it."""
    if column.startswith("regime_"):
        return "market regime (HMM)"
    if column.startswith("ffd_"):
        return "fractional differencing"
    return "conditional volatility (GARCH)"


families |= {c: model_based_family(c) for c in model_based_cols}

all_feature_cols = financial_cols + model_based_cols
source_of = {c: ("financial" if c in financial_cols else "model-based") for c in all_feature_cols}

inventory = (
    pl.DataFrame(
        {
            "family": [families[c] for c in all_feature_cols],
            "source": [source_of[c] for c in all_feature_cols],
            "feature": all_feature_cols,
        }
    )
    .group_by(["source", "family"])
    .agg(pl.len().alias("features"), pl.col("feature").sort().str.join(", ").alias("columns"))
    .sort(["source", "features"], descending=[False, True])
)
display(inventory)

# %% [markdown]
# ### The walk-forward folds
#
# `generate_cv_splits` derives the folds from the dates the label frame carries and the
# design declared in `config/setup.yaml`. It is the call `04_model_based_features` makes
# to decide which fold each of its rows belongs to, and the one `load_modeling_dataset`
# makes for the model notebooks, so a fold number denotes the same pair of date ranges
# wherever it appears.
#
# The folds step backwards from the holdout boundary, so fold 0 is the most recent one.
# Each validation range stops far enough before the boundary that the forward return of
# its last date has finished before the holdout opens.

# %%
splits = generate_cv_splits(
    label_df.select(DATE_COL).unique().sort(DATE_COL),
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
)


def _as_date(value) -> date:
    return pd.Timestamp(value).date()


for split in splits:
    print(
        f"  Fold {split['fold']}: train {_as_date(split['train_start'])} → "
        f"{_as_date(split['train_end'])}, validation {_as_date(split['val_start'])} → "
        f"{_as_date(split['val_end'])}"
    )

# %% [markdown]
# ### Resolving the fold dimension, and sealing the holdout
#
# Two filters build the panel. The first keeps each model-based row only inside the
# validation range of its own fold and drops the appended holdout fold, which leaves one
# value per date and ETF; the assertion below is what proves it, since two overlapping
# ranges would silently give a feature two values on one date.
#
# The second is the seal. The holdout must not inform which features look predictive, and
# the date a decision is taken is the wrong place to cut: a date a week before the
# boundary carries a forward return that finishes inside the holdout, so keeping it would
# put holdout prices into the rank correlation, the multiple-testing adjustment and the
# triage. The cut is therefore on the date the label **finishes**, found by stepping
# forward one horizon along the label's own calendar of trading dates.

# %%
val_windows = {int(s["fold"]): (_as_date(s["val_start"]), _as_date(s["val_end"])) for s in splits}
IN_VALIDATION = pl.any_horizontal(
    [(pl.col(DATE_COL) >= start) & (pl.col(DATE_COL) <= end) for start, end in val_windows.values()]
)
model_based = (
    model_based_artifact.filter(pl.col("fold").is_in(list(val_windows)))
    .filter(
        pl.col("fold").replace_strict({f: s for f, (s, _) in val_windows.items()}, default=None)
        <= pl.col(DATE_COL)
    )
    .filter(
        pl.col(DATE_COL)
        <= pl.col("fold").replace_strict({f: e for f, (_, e) in val_windows.items()}, default=None)
    )
    .drop("fold")
)
assert model_based.select(JOIN_COLS).is_duplicated().sum() == 0, (
    "validation windows overlap; a fitted feature would take two values on one date"
)

# %%
eval_panel = features.join(model_based, on=JOIN_COLS, how="left")
assert eval_panel.height == features.height, "the model-based join changed the panel's row count"
eval_panel = eval_panel.join(label_df, on=JOIN_COLS, how="inner")

last_signal_date = (
    label_df.select(DATE_COL)
    .unique()
    .sort(DATE_COL)
    .with_columns(pl.col(DATE_COL).shift(-LABEL_HORIZON).alias("_label_end"))
    .filter(pl.col("_label_end") < HOLDOUT_START)[DATE_COL]
    .max()
)
n_before_seal = eval_panel.height
eval_panel = eval_panel.filter(pl.col(DATE_COL) <= last_signal_date).filter(IN_VALIDATION)
assert eval_panel[DATE_COL].max() <= last_signal_date

if MAX_SYMBOLS > 0:
    top = eval_panel.group_by("symbol").len().sort("len", descending=True).head(MAX_SYMBOLS)
    eval_panel = eval_panel.filter(pl.col("symbol").is_in(top["symbol"]))

n_rows = eval_panel.height
n_symbols = eval_panel["symbol"].n_unique()
n_dates = eval_panel[DATE_COL].n_unique()

# A date needs enough ETFs quoted before a rank correlation across them says anything.
# The floor tracks the universe actually loaded, so a run on a reduced universe narrows
# the gate with it instead of screening every feature out.
MIN_CROSS_SECTION = min(MIN_CROSS_SECTION_TARGET, n_symbols)

print(
    f"Cut on the date the label finishes, then narrowed to the validation ranges: "
    f"{n_before_seal:,} -> {n_rows:,} rows.\n"
    f"The last date whose forward return finishes before {HOLDOUT_START} is "
    f"{last_signal_date}.\n"
    f"Evaluation panel: {n_symbols} ETFs over {n_dates:,} dates, "
    f"{eval_panel[DATE_COL].min()} to {eval_panel[DATE_COL].max()}, "
    f"{len(all_feature_cols)} candidate features.\n"
    f"A date enters the series once {MIN_CROSS_SECTION} ETFs carry both the feature and the label."
)

# %% [markdown]
# ## B. Is the artifact sound, and is each column usable?
#
# Two different questions, and they read different rows.
#
# The first is about the artifacts as `03` and `04` wrote them: does either contain a
# value no amount of screening can make usable - an infinity, a negative price, a monthly
# return so large it can only be a price-adjustment failure? The model notebooks train on
# every fold's training range, back to the start of the panel, so a broken value outside
# the validation ranges reaches them whether or not it is screened here. This check
# therefore reads the whole span up to the seal, and the model-based matrix as it was
# written, one row per date, ETF and fold. It stops at the seal like everything else, and
# it raises rather than warns, because a broken input makes the rest of the notebook
# meaningless.

# %%
sealed_features = features.filter(pl.col(DATE_COL) <= last_signal_date)
sealed_model_based = model_based_artifact.filter(pl.col(DATE_COL) <= last_signal_date)
sealed_labels = label_df.filter(pl.col(DATE_COL) <= last_signal_date)

validate_modeling_inputs(
    features_df=sealed_features,
    label_df=sealed_labels,
    feature_cols=financial_cols,
    label_col=label_col,
    join_cols=JOIN_COLS,
    asset_col="symbol",
    max_abs_return=1.0,  # a 21-day ETF return above this is a price-adjustment failure
    fail_on_critical=True,
)
validate_modeling_inputs(
    features_df=sealed_model_based,
    label_df=sealed_labels,
    feature_cols=model_based_cols,
    label_col=label_col,
    join_cols=JOIN_COLS,
    asset_col="symbol",
    max_abs_return=1.0,
    fail_on_critical=True,
)

# %% [markdown]
# The second question is per column. Before it can be asked, one group has to be separated
# out: some columns describe the market as a whole rather than one ETF against another -
# the yield-curve level, the filtered reference series, the regime probabilities. They take
# the same value for every ETF on a date, so ranking the cross-section by them produces no
# ordering at all and their cross-sectional rank correlation is undefined rather than zero.
# Nothing in the screen below can decide anything about them, and they are separated here
# rather than judged against a statistic they cannot have.

# %%
cs_std_df = eval_panel.group_by(DATE_COL).agg([pl.col(f).std().alias(f) for f in all_feature_cols])
date_level_features = {
    feat
    for feat in all_feature_cols
    if (mean_std := cs_std_df[feat].drop_nulls().mean()) is not None and mean_std < 1e-10
}
cross_sectional_features = [f for f in all_feature_cols if f not in date_level_features]
print(
    f"{len(date_level_features)} columns take one value across the whole cross-section on "
    f"each date:\n  {', '.join(sorted(date_level_features))}\n"
    f"{len(cross_sectional_features)} columns vary across ETFs and go on to the screen."
)

# %% [markdown]
# For the rest, the question is whether the column can be screened at all on the panel
# above:
#
# - **Coverage** is the share of rows where the feature has a value. A column present on
#   less than `MIN_COVERAGE` of the panel is being ranked on a different, smaller
#   cross-section than the one the strategy would trade.
# - **Staleness** is the share of rows that repeat the same ETF's previous value. Above
#   `MAX_STALENESS` the column changes so rarely that a daily rank correlation is mostly
#   reading the same ordering over and over.
#
# A column failing either gate is recorded STOP in the ledger and takes no further part.
# Both quantities are computed for the market-wide columns too, so the ledger carries them,
# but they decide nothing there: a column that cannot be ranked across ETFs is not made
# usable by changing more often.

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
    feat: coverage[feat] >= MIN_COVERAGE and staleness[feat] <= MAX_STALENESS
    for feat in all_feature_cols
}
failed = [f for f in cross_sectional_features if not correctness[f]]
n_gate_pass = len(cross_sectional_features) - len(failed)
print(
    f"Coverage and staleness: {n_gate_pass} of {len(cross_sectional_features)} "
    f"cross-sectional columns pass, {len(failed)} recorded STOP"
)

if failed:
    display(
        pl.DataFrame(
            {
                "feature": failed,
                "family": [families[f] for f in failed],
                "coverage": [round(coverage[f], 3) for f in failed],
                "staleness": [round(staleness[f], 3) for f in failed],
            }
        )
    )

# %% [markdown]
# ### How much independent evidence the panel really holds
#
# The panel has one row per ETF per date, and neither of its two dimensions contributes a
# row's worth of evidence.
#
# Across ETFs, the observation the test is built on is not a row. Section C scores the
# whole cross-section at one date into a single number, the information coefficient for
# that date, and it is that series the average and its standard error are computed on. The
# ETFs also move together, so forty rows on one date are nothing like forty independent
# readings even before the statistic pools them.
#
# Across dates, the readings overlap: a return measured over $h$ sessions and sampled every
# session shares $h-1$ of them with the next one, so consecutive dates score almost the
# same stretch of prices. Dividing the dates by the horizon counts the blocks that do not
# overlap:
#
# $$N_{\text{eff}} \approx \frac{N_{\text{dates}}}{h}$$
#
# So the evidence is measured in tens of independent blocks, not in tens of thousands of
# rows, and that is worth knowing before reading a p-value computed from it. This is not
# the correction the notebook applies - that is the standard-error adjustment in the next
# section, which handles the overlap properly. It is here so the row count printed above is
# read for what it is, which is not a sample size.

# %%
n_eff_dates = n_dates // LABEL_HORIZON
print(
    f"Panel rows: {n_rows:,} ({n_dates:,} dates x {n_symbols} ETFs)\n"
    f"Information coefficients, one per date: {n_dates:,}\n"
    f"Roughly independent blocks among them: ~{n_eff_dates:,} "
    f"(the {n_dates:,} dates over a {LABEL_HORIZON}-session label), "
    f"a factor of {n_dates / n_eff_dates:.0f} fewer."
)

# %% [markdown]
# ## C. Does the feature rank the cross-section the way the return does?
#
# On each date, the ETFs quoted that day are ranked by the feature and by the return they
# went on to earn, and the two rankings are correlated. That is one information
# coefficient per date. Ranks rather than levels, because what a ranking strategy acts on
# is the order, and because a single outlying return would otherwise set the number.
#
# Averaging that series gives the feature's association with the label. Testing whether
# the average is distinguishable from zero needs more care than a textbook t-test, because
# the series is serially dependent by construction: neighbouring dates score overlapping
# return windows. The Newey-West estimator widens the standard error to allow for that,
# given a bandwidth - how many neighbouring dates to treat as dependent. Overlap of a
# window $h$ sessions long reaches $h-1$ sessions, so the bandwidth is set to the label
# horizon itself, which covers it. Deriving that number from the label rather than typing
# it is what stops the correction and the label from drifting apart.

# %%
evaluable_features = [f for f in all_feature_cols if correctness[f]]

# %% [markdown]
# The series is built in one pass over the dates, computing every feature's rank
# correlation on each date's cross-section. A feature enters a date's series only when
# enough ETFs carry both it and the label on that date, which is what keeps a sparsely
# covered column from contributing a correlation over a handful of pairs at the same
# weight as one measured over the whole universe.

# %%
cs_features = [f for f in evaluable_features if f not in date_level_features]
eval_sub = eval_panel.select([DATE_COL, *cs_features, label_col]).drop_nulls(subset=[label_col])

dates_list = eval_sub[DATE_COL].unique().sort().to_list()
n_total = len(dates_list)
ic_series_data = {feat: [] for feat in cs_features}

for i, dt in enumerate(dates_list):
    cross_section = eval_sub.filter(pl.col(DATE_COL) == dt)
    label_arr = cross_section[label_col].to_numpy()
    label_valid = ~np.isnan(label_arr)

    for feat in cs_features:
        feat_arr = cross_section[feat].to_numpy()
        valid_mask = label_valid & ~np.isnan(feat_arr)
        if int(valid_mask.sum()) >= MIN_CROSS_SECTION:
            ic_val, _ = spearmanr(feat_arr[valid_mask], label_arr[valid_mask])
            if not np.isnan(ic_val):
                ic_series_data[feat].append((dt, float(ic_val), int(valid_mask.sum())))

    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{n_total} dates")

print(f"  {n_total}/{n_total} dates")

# %% [markdown]
# The series is sorted by date before the standard error is computed. The estimator reads
# the values in the order it is given them, so a series in any other order would be
# measuring dependence between dates that are not neighbours.

# %%
MIN_IC_DATES = 20  # dates a feature needs before its series is summarized at all

ic_results = {}
ic_timeseries = {}
for feat in cs_features:
    data = ic_series_data[feat]
    if len(data) < MIN_IC_DATES:
        continue
    dates_f, ics_f, nobs_f = zip(*data, strict=False)
    ic_df = pl.DataFrame({DATE_COL: list(dates_f), "ic": list(ics_f), "n_obs": list(nobs_f)}).sort(
        DATE_COL
    )
    ic_results[feat] = compute_ic_hac_stats(ic_df, ic_col="ic", maxlags=HAC_MAXLAGS)
    ic_timeseries[feat] = ic_df

print(
    f"An IC series was computed for {len(ic_results)} of the {len(all_feature_cols)} candidates. "
    f"Of the rest, {len(date_level_features)} carry no cross-sectional variation, "
    f"{len(failed)} failed the coverage or staleness gate, and "
    f"{len(cs_features) - len(ic_results)} had fewer than {MIN_IC_DATES} usable dates."
)

# %% [markdown]
# ## D. Does it hold across the walk-forward windows, or in one of them?
#
# An average taken over eight years can be produced by eight ordinary years or by one
# extraordinary one, and the two say different things about what a reader should expect
# next. So the series is cut into the folds' own validation ranges - the same ranges
# everything else in this pipeline numbers the same way - and summarized within each: the
# mean IC per fold, its median and spread across folds, the worst fold, and how much of
# the history agrees on a direction.
#
# Agreement is measured against the feature's **own** overall sign, not against a positive
# sign. A feature that is negative in every window is exactly as stable as one that is
# positive in every window, and just as usable: a ranking strategy reads it upside-down
# and is otherwise unaffected. A rule that counted positive folds would score the most
# dependable inverse predictor in the panel at zero.

# %%
MIN_FOLD_DATES = 5  # dates a fold must contribute before its mean IC is read


def per_fold_mean_ics(feat: str) -> list[float]:
    """Mean IC inside each fold's validation window, under the screen's own rule."""
    ts = ic_timeseries[feat]
    out = []
    for start, end in val_windows.values():
        window = ts.filter((pl.col(DATE_COL) >= start) & (pl.col(DATE_COL) <= end))
        if len(window) >= MIN_FOLD_DATES:
            out.append(float(window["ic"].mean()))
    return out


fold_stats = {}
for feat in ic_results:
    fold_ics = per_fold_mean_ics(feat)
    if fold_ics:
        pooled_sign = np.sign(ic_results[feat]["mean_ic"])
        agreeing = sum(1 for ic in fold_ics if np.sign(ic) == pooled_sign and pooled_sign != 0)
        fold_stats[feat] = {
            "n_folds": len(fold_ics),
            "sign_consistency": agreeing / len(fold_ics),
            "worst_fold_ic": min(fold_ics),
            "best_fold_ic": max(fold_ics),
            "median_fold_ic": float(np.median(fold_ics)),
        }

n_consistent = sum(1 for s in fold_stats.values() if s["sign_consistency"] >= MIN_SIGN_CONSISTENCY)
print(
    f"Fold statistics for {len(fold_stats)} features. {n_consistent} of them keep their own "
    f"overall sign in at least {MIN_SIGN_CONSISTENCY:.0%} of the folds."
)

# %% [markdown]
# ## E. What the search costs
#
# Every feature above was tested against the same label at the same level, so some of them
# clear the bar by chance alone: at a five percent level, one test in twenty does. How
# many depends on how many tests were run, which is why a p-value here cannot be read
# without the size of the search that produced it. **The searched set is every feature
# that cleared the correctness gate and produced an IC series**, and its size is printed
# below beside the results.
#
# The Benjamini-Hochberg procedure adjusts for that. Rather than holding down the chance of
# any false promotion at all, it holds down the false share *on average*: run this
# procedure over many such searches and the share of promotions that are false averages no
# more than the level set below. That is the right question when the point is to choose a
# set of features rather than to defend a single claim, and it is a weaker guarantee than
# it first sounds - it says nothing about the particular set promoted here, which can carry
# a higher false share than the level or none at all.
#
# Three counts follow, and the gaps between them are the two prices being paid. Naive
# significance ignores the overlap in the labels. The lag-aware count pays for the
# overlap. The false-discovery count pays for the number of tests on top of that.

# %%
feature_names = list(ic_results.keys())
p_values = [ic_results[f]["p_value"] for f in feature_names]

fdr_result = benjamini_hochberg_fdr(p_values, alpha=FDR_ALPHA, return_details=True)

eval_summary = pl.DataFrame(
    {
        "feature": feature_names,
        "source": [source_of[f] for f in feature_names],
        "ic_mean": [ic_results[f]["mean_ic"] for f in feature_names],
        "hac_se": [ic_results[f]["hac_se"] for f in feature_names],
        "hac_t": [ic_results[f]["t_stat"] for f in feature_names],
        "hac_p": p_values,
        "fdr_p": [float(p) for p in fdr_result["adjusted_p_values"]],
        "fdr_sig": [bool(r) for r in fdr_result["rejected"]],
        "naive_t": [ic_results[f]["naive_t_stat"] for f in feature_names],
    }
).sort(pl.col("ic_mean").cast(pl.Float64, strict=False).abs(), descending=True)

n_significant_naive = sum(1 for f in feature_names if abs(ic_results[f]["naive_t_stat"]) > NAIVE_T)
n_significant_hac = sum(1 for f in feature_names if abs(ic_results[f]["t_stat"]) > NAIVE_T)
n_significant_fdr = int(fdr_result["n_rejected"])


def survivor_share(remaining: int, started: int) -> str:
    """How much of the naive count a correction leaves, or that it leaves none."""
    if remaining == 0:
        return "nothing left"
    return f"{remaining / started:.0%} of the naive count"


print(
    f"Searched set: {len(feature_names)} features, each tested once against {label_col}.\n"
    f"  |t| > {NAIVE_T} treating dates as independent: {n_significant_naive}\n"
    f"  |t| > {NAIVE_T} with the overlap-aware standard error: {n_significant_hac} "
    f"({survivor_share(n_significant_hac, n_significant_naive)})\n"
    f"  Benjamini-Hochberg at q < {FDR_ALPHA}: {n_significant_fdr} "
    f"({survivor_share(n_significant_fdr, n_significant_naive)})"
)

# %% [markdown]
# ### The series behind the average
#
# Every statistic so far is one number standing for a series, and two things that decide
# whether a feature is worth anything are invisible once it has been averaged: an
# association that comes from a single episode and is absent around it, and one that
# changes direction partway through. So the series is drawn for the three features with
# the largest average association, before it is reduced any further. The heavy line is a
# six-month moving average, and the dotted pair marks the overlap-aware interval around the
# full-sample mean. That interval is narrow enough on this axis to be worth stating: the
# swing of the daily series is an order of magnitude wider than any uncertainty about where
# its average sits.

# %%
LEADING_FOR_SERIES = 3
ROLLING_DAYS = 126
series_features = eval_summary.head(LEADING_FOR_SERIES)["feature"].to_list()

fig = make_subplots(
    rows=len(series_features), cols=1, shared_xaxes=True, subplot_titles=series_features
)
for row, feat in enumerate(series_features, start=1):
    series = ic_timeseries[feat]
    bands = compute_ic_uncertainty(series, horizon=LABEL_HORIZON, ic_col="ic")
    dates = series[DATE_COL].to_list()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=series["ic"].to_list(),
            mode="lines",
            line=dict(color=GRAY_FILLS["muted"], width=0.6),
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=series["ic"].rolling_mean(ROLLING_DAYS, min_samples=ROLLING_DAYS).to_list(),
            mode="lines",
            line=dict(color=COLORS["blue"], width=1.4),
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    for value, dash in (
        (bands["mean_ic"], "solid"),
        (bands["ci_hac_lower"], "dot"),
        (bands["ci_hac_upper"], "dot"),
    ):
        fig.add_hline(y=value, line=dict(color=COLORS["amber"], width=1, dash=dash), row=row, col=1)
    fig.add_hline(y=0, line=dict(color=GRAY_FILLS["border"], width=0.8), row=row, col=1)
fig.update_layout(
    template="ml4t",
    height=200 * len(series_features) + 80,
    width=900,
    title_text="The daily IC swings far wider than the mean it averages to",
)
fig.update_yaxes(title_text="Rank IC")
style.show_plotly_with_alt(
    fig,
    "One stacked panel per leading feature, each showing the daily rank IC as a pale noisy "
    "series swinging between about -0.75 and +0.75 and a dark rolling mean that stays within "
    "roughly -0.1 to +0.35. Dashed reference lines mark the full-sample mean, which every "
    "panel's rolling line crosses repeatedly.",
)

# %% [markdown]
# ### Every feature, ranked by how strongly it ranks
#
# The features with the largest average association, with the overlap-aware t-statistic
# printed against each bar so the size of the association and the confidence in it are
# read together. Blue marks a feature that clears false-discovery control; grey is
# everything else, and that is the one colour convention every figure in this section
# uses.

# %%
top_n = min(25, len(eval_summary))
top = eval_summary.head(top_n).sort("ic_mean")

SURVIVES, DOES_NOT = COLORS["blue"], GRAY_FILLS["muted"]

fig = go.Figure(
    go.Bar(
        x=top["ic_mean"].to_list(),
        y=top["feature"].to_list(),
        orientation="h",
        marker_color=[SURVIVES if s else DOES_NOT for s in top["fdr_sig"].to_list()],
        text=[f"t={value:.1f}" for value in top["hac_t"].to_list()],
        textposition="outside",
        showlegend=False,
    )
)
fig.add_vline(x=0, line=dict(color=GRAY_FILLS["border"], width=1))
# Room for the t-statistic label past the end of the longest bar.
ic_span = max(abs(value) for value in top["ic_mean"].to_list()) * 1.35
fig.update_layout(
    template="ml4t",
    height=620,
    width=900,
    title_text="Volatility ranks the cross-section upward and recent strength downward",
    xaxis_title="Mean cross-sectional rank IC, with the overlap-aware t-statistic",
    xaxis_range=[-ic_span, ic_span],
    yaxis_title="Feature",
    margin=dict(l=170),
)
style.show_plotly_with_alt(
    fig,
    "Horizontal bars of mean cross-sectional rank IC per feature, sorted, with each bar "
    "annotated by its overlap-aware t-statistic. The volatility and distance-from-low features "
    "run positive to about +0.08 with t near 2.4 to 3.3, and the momentum, oscillator and "
    "drawdown features run negative to about -0.09 with t between -1.1 and -2.8.",
)

# %% [markdown]
# ### The same association, fold by fold
#
# Each row is one feature and each dot is its average IC inside one fold's validation
# range. The amber diamond is the median of those folds. A feature whose dots sit on one
# side of zero behaved the same way in every period; one whose dots straddle zero did not,
# whatever its overall average says.

# %%
FOLDS_SHOWN = 12
fold_features = [f for f in eval_summary["feature"].to_list() if f in fold_stats][:FOLDS_SHOWN]
fig = go.Figure()
for feat in fold_features:
    per_fold = per_fold_mean_ics(feat)
    first = feat == fold_features[0]
    fig.add_trace(
        go.Scatter(
            x=per_fold,
            y=[feat] * len(per_fold),
            mode="markers",
            marker=dict(color=GRAY_FILLS["muted"], size=7),
            name="one fold",
            legendgroup="fold",
            showlegend=first,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[fold_stats[feat]["median_fold_ic"]],
            y=[feat],
            mode="markers",
            marker=dict(color=COLORS["amber"], size=11, symbol="diamond"),
            name="median fold",
            legendgroup="median",
            showlegend=first,
        )
    )
fig.add_vline(x=0, line=dict(color=GRAY_FILLS["border"], width=1))
fig.update_layout(
    template="ml4t",
    height=520,
    width=900,
    title_text="Every leading feature changes sign in at least one fold",
    xaxis_title="Mean rank IC within the fold's validation range",
    yaxis_title="Feature",
    legend=dict(orientation="h", y=-0.12),
    margin=dict(l=170),
)
style.show_plotly_with_alt(
    fig,
    "One row per leading feature, with a grey dot for each fold's mean rank IC and an amber "
    "diamond at the median fold. Every row has dots on both sides of the zero line, so each "
    "feature changes sign in at least one fold, and the spread within a feature is wider than "
    "the gap between features.",
)

# %% [markdown]
# ### What the overlap correction costs
#
# The same t-statistic computed twice: once treating each date as independent evidence,
# once allowing for the overlap between neighbouring labels. A point on the dashed
# diagonal would mean the overlap cost nothing.

# %%
fig = go.Figure(
    go.Scatter(
        x=eval_summary["naive_t"].to_list(),
        y=eval_summary["hac_t"].to_list(),
        mode="markers",
        marker=dict(
            color=[SURVIVES if s else DOES_NOT for s in eval_summary["fdr_sig"].to_list()],
            size=7,
        ),
        text=eval_summary["feature"].to_list(),
        showlegend=False,
    )
)
max_t = (
    max(
        eval_summary["naive_t"].cast(pl.Float64, strict=False).abs().max() or 1.0,
        eval_summary["hac_t"].cast(pl.Float64, strict=False).abs().max() or 1.0,
    )
    * 1.1
)
fig.add_trace(
    go.Scatter(
        x=[-max_t, max_t],
        y=[-max_t, max_t],
        mode="lines",
        line=dict(dash="dash", color=GRAY_FILLS["border"]),
        showlegend=False,
    )
)
fig.update_layout(
    template="ml4t",
    height=480,
    width=760,
    title_text="Overlapping labels pull every t-statistic toward zero",
    xaxis_title="Naive t",
    yaxis_title="HAC t",
)
style.show_plotly_with_alt(
    fig,
    "A scatter of each feature's HAC t-statistic against its naive one, with a dashed diagonal "
    "marking equality. Every point sits between the diagonal and the horizontal zero line - "
    "naive values spread from about -10 to +11 while the HAC values stay inside roughly -3 to "
    "+3.5 - so correcting for overlap shrinks every t-statistic toward zero.",
)

# %% [markdown]
# Every point sits inside the diagonal, and the ones furthest along it lose the most: a
# t-statistic of ten computed as if each date were fresh evidence is worth about three
# once the overlap is allowed for. The number of simultaneous tests is charged on top of
# that, which is what leaves the false-discovery count where it is.

# %% [markdown]
# ## F. Is the relationship shaped like something a model can use?
#
# A rank correlation says the ordering carries information. It does not say the
# information is spread evenly across the ordering, and that matters for what a model can
# do with it. So on each date the ETFs are sorted by the feature and split into five equal
# buckets, and the return of each bucket is averaged - first within the date, then across
# dates, so that every date counts the same and a busy year does not outvote a quiet one.
#
# The buckets are cut **within each date**. Cutting once over the pooled panel would sort a
# 2016 observation against a 2023 one, so the feature's drift through time would decide
# which bucket a row lands in, and the profile would be answering a different question
# from the rank correlation printed beside it.
#
# Both the mean and the median of each bucket are drawn. The mean is what a book holding
# that bucket would earn; the median describes the typical ETF in it. Where the two are far
# apart, a few large moves are carrying the bucket.

# %% [markdown]
# The panels show the features that clear false-discovery control first, then the largest
# remaining associations, so the diagnostic still shows something when few features clear
# it.

# %%
fdr_shape = eval_summary.filter(pl.col("fdr_sig").fill_null(False))["feature"].to_list()
ranked_shape = eval_summary["feature"].to_list()
top_features_for_shape = (fdr_shape + [f for f in ranked_shape if f not in fdr_shape])[:6]

QUANTILE_LABELS = [f"Q{i + 1}" for i in range(N_QUANTILES)]
MIN_SHAPE_DATES = 20

monotonicity_scores = {}
quantile_spreads = {}

for feat in top_features_for_shape:
    profile = quantile_profile(
        eval_panel,
        feat,
        label_col,
        date_col=DATE_COL,
        n_quantiles=N_QUANTILES,
        min_cross_section=MIN_CROSS_SECTION,
    )
    if profile is None or profile.periods_used < MIN_SHAPE_DATES:
        continue
    quantile_spreads[feat] = {
        "q_means": profile.means,
        "q_medians": profile.medians,
        "spread": profile.spread,
    }
    # The ledger's monotonicity column is the rank correlation between the bucket index
    # and the bucket's mean return: +1 for a profile that rises all the way across, -1
    # for one that falls all the way, and near zero for one that turns in the middle.
    monotonicity_scores[feat] = profile.monotonicity

print(f"Bucket profiles built for {len(quantile_spreads)} features.")

# %%
if quantile_spreads:
    n_show = min(6, len(quantile_spreads))
    feats_to_show = list(quantile_spreads.keys())[:n_show]
    n_rows_fig = (n_show + 2) // 3
    fig = make_subplots(rows=n_rows_fig, cols=3, subplot_titles=feats_to_show, shared_yaxes=True)
    for idx, feat in enumerate(feats_to_show):
        r, c = divmod(idx, 3)
        fig.add_trace(
            go.Bar(
                x=QUANTILE_LABELS,
                y=quantile_spreads[feat]["q_means"],
                marker_color=COLORS["blue"],
                name="mean",
                legendgroup="mean",
                showlegend=idx == 0,
            ),
            row=r + 1,
            col=c + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=QUANTILE_LABELS,
                y=quantile_spreads[feat]["q_medians"],
                mode="markers",
                marker=dict(color=COLORS["amber"], size=9, symbol="diamond"),
                name="median",
                legendgroup="median",
                showlegend=idx == 0,
            ),
            row=r + 1,
            col=c + 1,
        )
    # One y range across the panels, so their heights are comparable. It includes zero
    # and is otherwise taken from the values.
    profile_values = [
        value
        for feat in feats_to_show
        for key in ("q_means", "q_medians")
        for value in quantile_spreads[feat][key]
    ]
    lo, hi = min(0.0, min(profile_values)), max(0.0, max(profile_values))
    pad = 0.15 * (hi - lo)
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(
        template="ml4t",
        height=260 * n_rows_fig + 60,
        width=900,
        title_text="The lowest-returning bucket is the bottom-ranked one in every panel",
        legend=dict(orientation="h", y=-0.08),
    )
    fig.update_yaxes(title_text="Mean forward return", col=1)
    style.show_plotly_with_alt(
        fig,
        "A grid of panels, one per leading feature, each with five bars for quintiles Q1 to Q5 of "
        "mean forward return and an amber diamond for the median. In every panel Q1 is the lowest "
        "bar at roughly a quarter of the others, while Q2 through Q5 sit close together, so the "
        "signal separates the bottom bucket rather than ordering the whole cross-section.",
    )

# %% [markdown]
# A profile that rises or falls all the way across is the shape a linear coefficient can
# carry on its own, which is what the ridge and elastic-net fits in `06_linear` are
# limited to. Where a panel turns at one end instead, the information is there but a
# single coefficient has to average the turn away, and the tree ensembles in `07_gbm` can
# split on it rather than fit through it. Where a median marker sits far from the top of
# its bar, that bucket's average is carried by a few large moves rather than by the ETF in
# the middle of it.

# %% [markdown]
# ## G. Which of these are the same evidence twice?
#
# Nothing above stops two features from measuring the same thing. Volatility at 63 and at
# 126 sessions is one measurement at two speeds; an oscillator and the ratio of price to
# its own moving average are two arrangements of the same recent path. Counting them
# separately makes a family's breadth look like independent corroboration, and it is not.
#
# So the ranks of every screened pair are correlated across a sample of the panel's dates,
# and the pairs above `REDUNDANCY_CUT` are counted and ranked. The sample keeps the
# comparison cheap; correlations this strong do not depend on the last decimal place.

# %%
sample_step = max(1, n_dates // 200)
sample_dates = eval_panel[DATE_COL].unique().sort().to_list()[::sample_step]
corr_matrix = (
    eval_panel.filter(pl.col(DATE_COL).is_in(sample_dates))
    .select(evaluable_features)
    .to_pandas()
    .corr(method="spearman")
)

high_corr_pairs = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if abs(corr_matrix.iloc[i, j]) > REDUNDANCY_CUT:
            high_corr_pairs.append((cols[i], cols[j], float(corr_matrix.iloc[i, j])))

print(
    f"{len(high_corr_pairs)} of the "
    f"{len(evaluable_features) * (len(evaluable_features) - 1) // 2:,} feature pairs agree "
    f"above |rho| = {REDUNDANCY_CUT}, sampled every {sample_step} dates."
)

# %% [markdown]
# ### Where the association sits, by family
#
# The register's families are the groups a reader would think in, so the association is
# summarized over them: how large it is on average within the family, and whether the
# family points one way or cancels out inside itself. A family whose features disagree in
# direction is not a family carrying a signal, it is a label over several different ones.

# %%
fdr_sig_set = set(eval_summary.filter(pl.col("fdr_sig").fill_null(False))["feature"].to_list())

family_summary = (
    pl.DataFrame(
        {
            "family": [families[f] for f in ic_results],
            "ic": [ic_results[f]["mean_ic"] for f in ic_results],
            "fdr_sig": [f in fdr_sig_set for f in ic_results],
        }
    )
    .group_by("family")
    .agg(
        pl.len().alias("n_features"),
        pl.col("ic").abs().mean().alias("avg_abs_ic"),
        pl.col("ic").mean().alias("avg_ic"),
        pl.col("fdr_sig").sum().alias("n_fdr_sig"),
    )
    .sort("avg_abs_ic", descending=True)
)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=family_summary["avg_abs_ic"].to_list()[::-1],
        y=family_summary["family"].to_list()[::-1],
        orientation="h",
        marker_color=GRAY_FILLS["muted"],
        name="average |IC| in the family",
    )
)
fig.add_trace(
    go.Scatter(
        x=family_summary["avg_ic"].to_list()[::-1],
        y=family_summary["family"].to_list()[::-1],
        mode="markers",
        marker=dict(color=COLORS["amber"], size=11, symbol="diamond"),
        name="average IC, signs kept",
    )
)
fig.add_vline(x=0, line=dict(color=GRAY_FILLS["border"], width=1))
fig.update_layout(
    template="ml4t",
    height=460,
    width=900,
    title_text="Only some families point one way; the rest cancel out inside themselves",
    xaxis_title="Mean cross-sectional rank IC across the family's features",
    yaxis_title="Feature family",
    legend=dict(orientation="h", y=-0.18),
    margin=dict(l=200),
)
style.show_plotly_with_alt(
    fig,
    "Horizontal bars of the average absolute rank IC within each feature family, with an amber "
    "diamond for the average signed IC. Conditional volatility and volatility keep their "
    "sign, their diamonds sitting near the end of their bars, and the oscillator and "
    "risk-adjusted-momentum families point as consistently the other way, their diamonds "
    "as far to the left as their bars are long. Only range-and-drawdown and cross-sectional "
    "position have a diamond near zero against a much longer bar, which is what cancelling "
    "inside a family looks like.",
)

# %% [markdown]
# ### The pairs that are one measurement
#
# A full correlation matrix over this many features has unreadable tick labels and is
# mostly empty space, and the reader's question is narrower than a matrix answers: which
# specific pairs are one measurement entered twice. So the strongest pairs are ranked
# instead.

# %%
TOP_PAIRS = 15
ranked_pairs = sorted(high_corr_pairs, key=lambda item: -abs(item[2]))[:TOP_PAIRS]
if ranked_pairs:
    fig = go.Figure(
        go.Bar(
            x=[rho for _, _, rho in ranked_pairs][::-1],
            y=[f"{a} / {b}" for a, b, _ in ranked_pairs][::-1],
            orientation="h",
            marker_color=[
                COLORS["blue"] if rho > 0 else COLORS["copper"] for _, _, rho in ranked_pairs
            ][::-1],
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line=dict(color=GRAY_FILLS["border"], width=1))
    fig.update_layout(
        template="ml4t",
        height=520,
        width=900,
        title_text="The strongest pairs are near-duplicates, not merely related",
        xaxis_title="Pairwise Spearman correlation",
        xaxis_range=[-1, 1],
        yaxis_title="Feature pair",
        margin=dict(l=300),
    )
    style.show_plotly_with_alt(
        fig,
        "Horizontal bars of pairwise Spearman correlation for the most correlated feature pairs, on "
        "an axis spanning -1 to 1. Every bar is positive and close to the right edge, above about "
        "0.9, so the strongest pairs are near-duplicates rather than merely related.",
    )

# %% [markdown]
# The leading pairs are the same measurement at two window lengths, two arrangements of
# the same recent price path, or two filtered series of ETFs that track overlapping
# markets. Nothing downstream has to drop one of each pair: the ridge and elastic-net
# penalties in `06_linear` shrink correlated coefficients together rather than letting one
# of them absorb the pair, and the tree ensembles in `07_gbm` split on whichever member
# happens to be available. What the pair count does change is how this section's own
# results are read, because a family with eight members above the cut is not eight
# independent pieces of evidence.

# %% [markdown]
# ## H. The decision each feature gets
#
# | Decision | When | Arm |
# |----------|------|-----|
# | **PROCEED** | clears Benjamini-Hochberg at `FDR_ALPHA` | confirmation |
# | **PROCEED** | keeps its own sign in at least `MIN_SIGN_CONSISTENCY` of the folds, and its average IC is at least `IC_THRESHOLD` in size | exploration |
# | **STOP** | failed the coverage or staleness gate | - |
# | **REVISE** | everything else | - |
#
# The two promotion rules are an **or**, not an **and**, so the promoted set can be larger
# than the set that clears false-discovery control, and the ledger's `note` column records
# which of the two promoted each feature. Reading the count without that column would
# credit the exploration arm's promotions to the confirmation arm.
#
# The second arm exists because false-discovery control over a search this wide can leave
# nothing at all, and a screen that returns an empty menu has told the reader nothing about
# which features to look at first. It is an exploration filter in the sense of Section 7.4
# rather than a test: `IC_THRESHOLD` is a stated judgement about the smallest association
# worth carrying into a monthly rebalance, not a quantity derived from the data, and a
# feature promoted through it has not been confirmed by anything.
#
# The columns that take one value across the whole cross-section are recorded REVISE. They
# cannot have a cross-sectional IC, so there is nothing here to decide them on, and they
# stay in the feature matrix the model notebooks train on.

# %%
triage = {}
for feat in all_feature_cols:
    if feat in date_level_features:
        triage[feat] = ("REVISE", "date_level_feature")
        continue

    if not correctness[feat]:
        triage[feat] = ("STOP", "correctness_fail")
        continue

    if feat not in ic_results:
        triage[feat] = ("REVISE", "insufficient_data")
        continue

    is_fdr_sig = feat in fdr_sig_set
    sign_con = fold_stats.get(feat, {}).get("sign_consistency", 0)
    abs_ic = abs(ic_results[feat]["mean_ic"])

    if is_fdr_sig:
        triage[feat] = ("PROCEED", "fdr_significant")
    elif sign_con >= MIN_SIGN_CONSISTENCY and abs_ic >= IC_THRESHOLD:
        triage[feat] = ("PROCEED", "stable_and_above_threshold")
    else:
        triage[feat] = ("REVISE", "not_significant_standalone")

# %%
ledger_rows = []
for feat in all_feature_cols:
    decision, note = triage[feat]
    row = {
        "feature": feat,
        "family": families[feat],
        "source": source_of[feat],
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

ic_ts_all = pl.concat(
    [ts.with_columns(pl.lit(feat).alias("feature")) for feat, ts in ic_timeseries.items()]
)
ic_ts_all.write_parquet(EVAL_DIR / "ic_timeseries.parquet")

print(
    f"evaluation/triage_ledger.parquet: {triage_ledger.height} features, one row each.\n"
    f"evaluation/ic_timeseries.parquet: {ic_ts_all.height:,} rows, the per-date series behind them."
)

# %% [markdown]
# ### The funnel, end to end
#
# Every candidate feature ends in one of the three decisions, and the figure below is the
# whole path: how many were offered, how many can be ranked across ETFs at all, how many
# of those cleared the coverage and staleness gates, how many cleared false-discovery
# control, and how many were promoted. The distance between the last two bars is the
# exploration arm.

# %%
proceed_features = sorted(f for f, (d, _) in triage.items() if d == "PROCEED")
revise_features = [f for f, (d, _) in triage.items() if d == "REVISE"]
stop_features = [f for f, (d, _) in triage.items() if d == "STOP"]

funnel_stages = [
    ("Candidate features", len(all_feature_cols)),
    ("Vary across the cross-section", len(cross_sectional_features)),
    ("Cleared coverage and staleness", n_gate_pass),
    ("Cleared false-discovery control", n_significant_fdr),
    ("Promoted (PROCEED)", len(proceed_features)),
]
fig = go.Figure(
    go.Bar(
        x=[n for _, n in funnel_stages][::-1],
        y=[stage for stage, _ in funnel_stages][::-1],
        orientation="h",
        marker_color=[COLORS["blue"] if i == 4 else GRAY_FILLS["muted"] for i in range(5)][::-1],
        text=[str(n) for _, n in funnel_stages][::-1],
        textposition="outside",
        showlegend=False,
    )
)
fig.update_layout(
    template="ml4t",
    height=380,
    width=900,
    title_text="The promoted set is wider than what clears false-discovery control",
    xaxis_title="Features",
    xaxis_range=[0, len(all_feature_cols) * 1.15],
    margin=dict(l=250),
)
style.show_plotly_with_alt(
    fig,
    "A funnel of five rows counting the features surviving each screen, each labelled with its "
    "count. The rows narrow from the candidate set down through the cross-sectional and "
    "coverage screens, and the false-discovery row counts zero, so it draws no bar at all - "
    "yet the promoted row beneath it is longer than the coverage row is short of, which is "
    "the gap the title names.",
)

# %% [markdown] tags=["results"]
# **What the triage decided, and on what.** Every candidate feature carries a decision in
# `evaluation/triage_ledger.parquet` and a `note` recording which rule produced it. **No
# feature on this panel clears false-discovery control**, so the confirmation arm never
# fires and every promotion above is an exploration promotion, with `note` reading
# `stable_and_above_threshold` throughout.
#
# That is a statement about how much evidence a panel of this size holds, not a fault in
# the screen. A monthly forward return sampled daily leaves the rough count of independent
# observations printed in section C, and over a search this wide, false-discovery control
# at this level rejects nothing at that amount of evidence. What the exploration arm keeps
# on the table is the set that holds its own direction across the folds and clears the
# threshold in size. None of it has been confirmed, and a reader who takes the promoted
# count as a count of confirmed features will have read past the `note` column.
#
# This notebook does not pronounce on the case study. A screen of one feature at a time is
# necessary and not sufficient, and whether any of this is tradable is decided by a
# backtest several stages later.

# %% [markdown]
# ## Key takeaways
#
# 1. **Report the size of the search beside the p-value.** The three counts in section E
#    are three answers to "how many of these features carry information", and the gaps
#    between them are what was paid for the overlap in the labels and for having asked
#    the question of every feature at once. A significance claim without the size of the
#    search that produced it cannot be read at all.
#
# 2. **Bind the standard-error bandwidth to the label, not to a number.** Overlapping
#    forward returns make neighbouring dates dependent, and how far that dependence
#    reaches is the label horizon. Passing the horizon rather than typing a lag is what
#    stops the correction and the label from drifting apart when the horizon changes.
#
# 3. **Screen every feature on the rows where every feature exists.** Features fitted
#    inside a fold are usable only inside that fold's validation range. Measured over the
#    whole history their coverage is the share of it the folds reach, and the coverage
#    gate reads a property of the design as a broken column; measured over the union of
#    the validation ranges they are comparable with the price-derived features beside
#    them.
#
# 4. **A statistic a column cannot have says nothing about the column.** Anything
#    describing the market as a whole takes one value across the cross-section, so it has
#    no cross-sectional rank correlation. It is recorded as undecided rather than scored
#    at zero, and it stays in the matrix the model notebooks train on.
#
# 5. **Count evidence, not columns.** Features within a family are strongly correlated,
#    and the ranked pairs say which ones are one measurement entered twice. A family with
#    many members above the redundancy cut is not many independent findings.
#
# **Known limitations.** One feature at a time says nothing about what a set of them does
# together, which is what `06_linear` onward measures. The screen is run against the
# primary label only, so a feature that predicts at a different horizon is not visible
# here. And the exploration arm's threshold is a judgement about what a monthly rebalance
# would need, not a quantity the data determined - a reader working at a different horizon
# or cost level should set their own and say so.
#
# **Next**: `06_linear` fits regularized linear models to the whole feature matrix, and
# `20_strategy_synthesis/02_feature_evaluation` reads this ledger alongside the other
# eight case studies'.
