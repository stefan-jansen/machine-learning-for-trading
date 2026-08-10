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
# # FX Pairs: Feature Evaluation
#
# The two feature stages built a matrix of candidate columns for 20 currency pairs.
# This notebook asks the one question that has to be answered before any of them is fed
# to a model: taken on its own, does a column say anything about where the pair's price
# goes next?
#
# Each column is tested alone, against the next session's return, on the same
# walk-forward folds the feature stages fitted their models on - a walk-forward fold is
# a training window followed by the stretch of later sessions that window did not see.
# Only those later stretches are read here, so no column is judged on data its own
# estimator was fitted on. The last two years of the sample, from 2024 on, are the
# holdout. The files loaded below run through it, and the panel every statistic here is
# computed on stops before it, with a further gap so that no label settling inside the
# holdout is read either. Nothing in this notebook measures anything on those sessions,
# and they stay that way until one finished strategy is measured on them once.
#
# **Learning objectives**
#
# - Check that a column is filled in often enough, and moves often enough, to carry
#   information at all, before spending any inference on it.
# - Measure how well a column's ranking of the 20 pairs on one day agrees with their
#   ranking by the next day's return, and put an interval around the average of those
#   daily agreements that allows for one day's agreement resembling the next day's.
# - Adjust for the number of columns tested at once, so that a handful of small
#   p-values out of sixty-odd is not read as a discovery.
# - Separate a column that ranks the pairs against each other from one that describes
#   the market as a whole, which cannot rank anything and has to be tested differently.
#
# **Book reference**: Chapter 7, Section 7.3 (univariate feature-label evaluation) and
# Section 7.4 (search accounting and multiple testing)
#
# **Prerequisites**: `03_financial_features.py` and `04_model_based_features.py`
#
# **Outputs**
#
# - `evaluation/ic_timeseries.parquet`: the per-session agreement score for every
#   column, by fold. It is written as soon as it is computed and then read back, so the
#   figures below and every average in this notebook are drawn from the stored copy.
# - `evaluation/triage_ledger.parquet`: one decision per column with the evidence
#   behind it. `20_strategy_synthesis/02_feature_evaluation.py` reads this file from
#   every case study and compares the decisions across all nine.

# %%
"""Test each FX feature column on its own against the forward return it is built for."""

import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats, compute_ic_uncertainty
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from case_studies.utils.feature_engineering import quantile_profile
from utils.artifact_specs import resolve_label_buffer
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.data_quality import validate_modeling_inputs
from utils.paths import get_case_study_dir
from utils.style import COLORS, show_plotly_with_alt

warnings.filterwarnings("ignore", category=FutureWarning)

# %% tags=["parameters"]
# Production defaults. Papermill can reduce symbols or folds for a smoke test.
MAX_SYMBOLS = 0
MAX_FOLDS = 0

# %% [markdown]
# ## Configuration
#
# Every label, window and boundary below is read from `config/setup.yaml`, the same file
# the label and feature stages bound their own parameters from. Nothing that file
# declares is retyped here, so a decision changed there changes here too rather than
# quietly disagreeing.
#
# Four numbers are judgments rather than measurements. They are settled once, here, so
# that no comparison further down decides anything on a literal typed beside it. The
# cell below prints each with the value it holds; what each one decides is:
#
# - **Coverage floor.** How often a column has to hold a value, counted from the first
#   session it is ever filled in. Below the floor, whatever the column measures is
#   absent from most of the sample, and any average over it is an average over a
#   self-selected set of days.
# - **Staleness ceiling.** Staleness is the share of a pair's consecutive observations
#   of a column that are identical to the one before, and the ceiling is how much of
#   that is tolerated. A column that mostly repeats yesterday's number cannot
#   distinguish today from yesterday, whether because the underlying barely moves or
#   because the feed behind it stopped updating.
# - **Effect-size floor.** The size of average agreement the exploration route at the
#   end promotes above. It sits an order of magnitude below what an FX ranking strategy
#   would need to earn back the spread it pays, so clearing it means clearing a floor,
#   not passing a test.
# - **Direction agreement.** The share of folds a column has to point the same way in
#   before that same route will promote it.

# %%
CASE_STUDY_ID = "fx_pairs"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
EVAL_DIR = CASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
evaluation_config = load_evaluation_config(CASE_STUDY_ID)

JOIN_COLS = ["timestamp", "symbol"]
DATE_COL = "timestamp"
LABEL_COL = setup["labels"]["primary"]
LABEL_BUFFERS = {LABEL_COL: setup["labels"]["buffer"], **setup["labels"]["variant_buffers"]}
LABEL_HORIZONS = {name: int(buffer.rstrip("Dd")) for name, buffer in LABEL_BUFFERS.items()}
LABEL_HORIZON = LABEL_HORIZONS[LABEL_COL]
HOLDOUT_START = date.fromisoformat(str(evaluation_config["holdout_start"]))
REDUNDANCY_CUT = float(setup["features"]["redundancy_cut"])

N_QUANTILES = 5  # buckets the pairs are split into when the shape is read off
MIN_FOLD_DAYS = 5  # sessions a fold needs before its own average is reported
MIN_SESSIONS = 20  # scored sessions a column needs before it gets an average at all
MARKET_LEVEL_SHARE = 0.90  # share of sessions constant across pairs that marks a market state
N_LEADERS = 8  # columns carried into the series, fold and horizon figures
N_SHAPE_PANELS = 6  # bucket profiles drawn; the score is recorded for every column
COVERAGE_FLOOR = 0.70
STALENESS_CEILING = 0.50
IC_THRESHOLD = 0.005
STABILITY_THRESHOLD = 0.60

print(
    f"Primary label: {LABEL_COL}, the return over the {LABEL_HORIZON} session(s) "
    f"following the decision. It sets the Newey-West bandwidth and the width of the gap "
    f"held open in front of the holdout."
)
for name, horizon in sorted(LABEL_HORIZONS.items(), key=lambda item: item[1]):
    role = "primary" if name == LABEL_COL else "compared against it at the end of part 4"
    print(f"  {name}: {horizon} session(s) forward, {role}")
print(
    f"Holdout: {HOLDOUT_START} to {evaluation_config['holdout_end']}, not read here. "
    f"Evaluation runs on the sessions before it."
)
print(
    f"Two columns whose rankings agree above {REDUNDANCY_CUT:.2f} in absolute Spearman "
    f"correlation are treated as one piece of evidence, not two "
    f"(config/setup.yaml, features.redundancy_cut)."
)
print(
    f"Coverage floor {COVERAGE_FLOOR:.0%}: a column filled in less often than this is not tested."
)
print(
    f"Staleness ceiling {STALENESS_CEILING:.0%}: a column repeating its previous value "
    f"more often than this is not tested."
)
print(
    f"Effect-size floor {IC_THRESHOLD}: the exploration route promotes only above this "
    f"average agreement per session."
)
print(
    f"Direction agreement {STABILITY_THRESHOLD:.0%}: and only when at least this share "
    f"of the {evaluation_config['n_splits']} folds point the same way."
)

# %% [markdown]
# ## 1. Build the Panel Every Statistic Runs On
#
# Three files are joined on pair and session: the price-derived columns, the
# model-derived columns, and the label the columns are tested against.
#
# The model-derived file needs care. Its estimators are refitted once per fold, so it
# carries a row for the same pair and session under every fold whose training window
# reaches that session - the same date appears several times with different numbers
# behind it. Keeping any one of those rows at random would attach a parameter fitted on
# later data to an earlier date. What is kept instead is, for each fold, only the
# sessions in that fold's own validation stretch: the window its estimator was not
# fitted on. The `fold` column is carried through the rest of the notebook so every
# statistic can still be traced back to the fit that produced its inputs.

# %%
financial = pl.read_parquet(CASE_DIR / "features" / "financial.parquet")
model_based = pl.read_parquet(CASE_DIR / "features" / "model_based.parquet")
labels = pl.read_parquet(CASE_DIR / "labels" / f"{LABEL_COL}.parquet")

required_temporal = {*JOIN_COLS, "fold"}
missing_temporal = required_temporal.difference(model_based.columns)
if missing_temporal:
    raise ValueError(f"Model-based artifact lacks fold provenance: {sorted(missing_temporal)}")

# %% [markdown]
# ### Derive the Fold Boundaries, and Close the Gap in Front of the Holdout
#
# `generate_cv_splits` reads the label calendar and the walk-forward window declared in
# `setup.yaml`. That is the same call `04_model_based_features` made when it stamped the
# `fold` column this notebook joins on, so deriving the boundaries again here - rather
# than replaying a stored array of splits - is what keeps a fold id naming the same
# window on both sides of the join.
#
# The `outcome_horizon` argument closes the gap in front of the holdout. A one-session
# label taken on the last development session settles on the first holdout session, so
# a statistic computed on that decision has already read a holdout price. The argument
# drops every validation session whose label would settle on or after the holdout
# opens, counted by position on the pairs' own trading grid rather than in calendar
# days, so a weekend or a market closure cannot shift the boundary.

# %%
LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, LABEL_COL, setup)
UNIQUE_DATES = labels.select(DATE_COL).unique().sort(DATE_COL)


def fold_windows(outcome_horizon: str) -> list[dict]:
    """Walk-forward folds, with the gap in front of the holdout set by one horizon."""
    splits = generate_cv_splits(
        UNIQUE_DATES,
        case_study_id=CASE_STUDY_ID,
        label_buffer=LABEL_BUFFER,
        outcome_horizon=outcome_horizon,
    )
    return splits[:MAX_FOLDS] if MAX_FOLDS > 0 else splits


def validation_rows(splits: list[dict]) -> pl.DataFrame:
    """Each fold's own validation sessions, taken from the fold it was fitted out of.

    A fold id that names one window in the artifact and another here would silently
    pair a fit with the wrong sessions, so an id whose stamped rows fall outside the
    window this configuration validates it over raises rather than returning nothing.
    """
    stamped = set(model_based["fold"].unique().to_list())
    frames = []
    for split in splits:
        fold = int(split["fold"])
        if fold not in stamped:
            continue
        val_start = pd.Timestamp(split["val_start"]).date()
        val_end = pd.Timestamp(split["val_end"]).date()
        rows = model_based.filter(
            (pl.col("fold") == fold)
            & pl.col(DATE_COL).is_between(val_start, val_end, closed="both")
        )
        if not len(rows):
            span = model_based.filter(pl.col("fold") == fold)
            raise ValueError(
                f"Fold {fold} is stamped on rows spanning {span[DATE_COL].min()}.."
                f"{span[DATE_COL].max()} but this configuration validates it over "
                f"{val_start}..{val_end}, so the fold id names a different window on "
                f"each side of the join"
            )
        frames.append(rows)
    if not frames:
        raise ValueError(f"No fold in {sorted(stamped)} appears in the configured splits")
    return pl.concat(frames).sort([DATE_COL, "symbol"])


# %%
splits = fold_windows(LABEL_BUFFER)
validation_temporal = validation_rows(splits)
duplicate_keys = validation_temporal.group_by(JOIN_COLS).len().filter(pl.col("len") > 1)
if len(duplicate_keys):
    raise ValueError("The folds' validation windows overlap on timestamp and symbol")

for fold in sorted(validation_temporal["fold"].unique().to_list()):
    window = validation_temporal.filter(pl.col("fold") == fold)
    print(f"  Fold {fold}: validation {window[DATE_COL].min()}..{window[DATE_COL].max()}")

# %%
eval_panel = (
    validation_temporal.join(financial, on=JOIN_COLS, how="inner")
    .join(labels, on=JOIN_COLS, how="inner")
    .sort([DATE_COL, "symbol"])
)
dropped = len(validation_temporal) - len(eval_panel)
if dropped:
    raise ValueError(
        f"{dropped:,} of {len(validation_temporal):,} validation rows have no matching "
        f"row in features/financial.parquet or labels/{LABEL_COL}.parquet, so the three "
        f"files do not cover the same pairs and sessions"
    )

if MAX_SYMBOLS > 0:
    selected_symbols = sorted(eval_panel["symbol"].unique().to_list())[:MAX_SYMBOLS]
    eval_panel = eval_panel.filter(pl.col("symbol").is_in(selected_symbols))

financial_cols = [column for column in financial.columns if column not in JOIN_COLS]
temporal_cols = [column for column in model_based.columns if column not in {*JOIN_COLS, "fold"}]
all_feature_cols = financial_cols + temporal_cols

# %% [markdown]
# ### Check the Panel Before Measuring Anything On It
#
# The gap in front of the holdout was applied when the folds were derived. The check
# below proves it survived the joins, on the frame every statistic in this notebook is
# actually computed from, which is the only frame worth checking. A duplicated pair and
# session would weight one observation twice, so the key has to stay unique as well.
#
# One number is derived rather than declared: how many pairs a session needs to be
# quoting before its ranking is worth reading. A rank correlation over a handful of
# pairs is mostly noise, so the floor is a quarter of the universe on the panel, five of
# the twenty pairs. Deriving it from what is loaded rather than typing it means a run
# reduced to fewer pairs lowers the floor with it, instead of discarding every session.

# %%
assert eval_panel[DATE_COL].max() < HOLDOUT_START, "Evaluation panel reaches the holdout"
if eval_panel.select(JOIN_COLS).n_unique() != len(eval_panel):
    raise ValueError("Evaluation panel has duplicate timestamp-symbol rows")

n_rows = len(eval_panel)
n_symbols = eval_panel["symbol"].n_unique()
n_dates = eval_panel[DATE_COL].n_unique()
MIN_PERIODS = max(3, n_symbols // 4)
print(
    f"Panel: {n_rows:,} rows over {n_dates:,} sessions and {n_symbols} pairs, "
    f"drawn from the validation stretches of {eval_panel['fold'].n_unique()} folds"
)
print(f"Sessions run {eval_panel[DATE_COL].min()} to {eval_panel[DATE_COL].max()}")
print(
    f"Candidates: {len(financial_cols)} price-derived + {len(temporal_cols)} "
    f"model-derived = {len(all_feature_cols)}"
)
print(f"A session is ranked only when at least {MIN_PERIODS} pairs have both values")

# %% [markdown]
# ### What the Candidates Are
#
# A list of sixty column names says little about what is being tested. Grouping them by
# the economic or modeling idea behind each one shows how concentrated the search
# actually is: several families are one idea measured over four or five trailing
# windows, which is why the correlations in part 6 matter and why the multiplicity
# adjustment in part 4 is applied over the whole set rather than family by family.
#
# The grouping is by name, because the feature stages name a column after the family it
# came from. A column that matched no family would be a column this notebook cannot
# describe, so an unmatched name stops the run rather than landing in a bucket called
# "other".


# %%
def assign_feature_family(feature_name: str) -> str:
    """Map an FX feature column to the economic or modeling idea it comes from."""
    family_map = [
        (["kalman_"], "trend and level"),
        (["hmm_"], "dollar regime"),
        (["arima_"], "return forecast"),
        (["rank_"], "cross-sectional rank"),
        (["zscore", "channel_pos", "mom_skip"], "mean reversion"),
        (["ret_"], "momentum"),
        (["vol_gk", "vol_cc", "vol_ratio", "avg_range"], "volatility"),
        (["sharpe_", "accel_"], "risk-adjusted momentum"),
        (["usd_factor", "usd_corr"], "dollar factor"),
        (["rsi", "bollinger", "price_to_ma"], "oscillator and trend"),
        (["max_dd"], "drawdown"),
    ]
    lowered = feature_name.lower()
    for prefixes, family in family_map:
        if any(prefix in lowered for prefix in prefixes):
            return family
    return "other"


# %%
families = {feature: assign_feature_family(feature) for feature in all_feature_cols}
unfamiliar = sorted(name for name, family in families.items() if family == "other")
if unfamiliar:
    raise ValueError(
        f"{len(unfamiliar)} columns match no feature family, so the artifact carries "
        f"names this notebook cannot describe: {unfamiliar}"
    )

family_table = (
    pl.DataFrame(
        {
            "family": list(families.values()),
            "source": [
                "model-derived" if feature in temporal_cols else "price-derived"
                for feature in families
            ],
            "feature": list(families),
        }
    )
    .group_by(["source", "family"])
    .agg(pl.len().alias("columns"), pl.col("feature").sort().str.join(", ").alias("names"))
    # Two families holding the same number of columns would otherwise be ordered by
    # whatever the grouping happened to emit, which differs between runs, so the table
    # printed here would not be the table a reader re-running the notebook sees.
    .sort(["source", "columns", "family"], descending=[False, True, False])
)
with pl.Config(tbl_rows=family_table.height, tbl_width_chars=260, fmt_str_lengths=200):
    display(family_table)

# %% [markdown]
# ## 2. Two Screens That Come Before Any Inference
#
# Testing a column that is mostly empty, or mostly the same number repeated, produces a
# p-value that means nothing but looks like the others. Both are cheap to detect and
# both are settled before a single correlation is computed.
#
# The first screen is on the panel as a whole and asks whether the numbers in it are
# usable at all: infinities, prices below zero, returns too large to be a return. It
# separates a failure from a flag. A failure stops the notebook. A flag reports a column
# whose magnitude is unusual without being wrong - `kalman_smoothness` is the reciprocal
# of a variance the state-space model estimates, so its scale is unbounded by
# construction and it trips an absolute-magnitude test every run.

# %%
validate_modeling_inputs(
    features_df=eval_panel,
    label_df=eval_panel,
    feature_cols=all_feature_cols,
    label_col=LABEL_COL,
    join_cols=JOIN_COLS,
    asset_col="symbol",
    max_abs_return=0.5,
    fail_on_critical=True,
)

# %% [markdown]
# The second screen is per column, and it is two measurements.
#
# **Coverage** is the share of sessions on which the column holds a value, counted from
# the first session it is ever filled in rather than from the start of the panel, so a
# column with a long trailing window is not punished for the sessions before it could
# possibly have one.
#
# **Staleness** is the share of a pair's consecutive observations that are identical to
# the one before. Read it with the column's construction in mind: a percentile rank
# over twenty pairs takes twenty values, so a pair that stays near the top of a
# persistent ordering repeats its rank for weeks without anything being wrong with the
# feed. The measurement is worth reading for what it can do and not more: it finds a
# column that cannot separate today from yesterday, and it does not tell a dead feed
# apart from a slow-moving one.

# %%
coverage = {}
staleness = {}
for feature in all_feature_cols:
    non_null = eval_panel.filter(pl.col(feature).is_not_null())
    if len(non_null) == 0:
        coverage[feature] = 0.0
    else:
        first_date = non_null[DATE_COL].min()
        eligible = eval_panel.filter(pl.col(DATE_COL) >= first_date)
        coverage[feature] = len(non_null) / len(eligible)

    chronological = eval_panel.select(JOIN_COLS + [feature]).sort(["symbol", DATE_COL])
    unchanged = chronological.select(
        (pl.col(feature) == pl.col(feature).shift(1).over("symbol")).sum()
    ).item()
    comparable = chronological.select(
        pl.col(feature).shift(1).over("symbol").is_not_null().sum()
    ).item()
    staleness[feature] = float(unchanged) / max(comparable, 1)

correctness = {
    feature: coverage[feature] >= COVERAGE_FLOOR and staleness[feature] <= STALENESS_CEILING
    for feature in all_feature_cols
}
failed_features = [feature for feature, passed in correctness.items() if not passed]
print(
    f"{len(correctness) - len(failed_features)} of {len(correctness)} columns are filled "
    f"in on at least {COVERAGE_FLOOR:.0%} of their sessions and repeat on at most "
    f"{STALENESS_CEILING:.0%} of them"
)
if failed_features:
    screened_out = pl.DataFrame(
        {
            "feature": failed_features,
            "family": [families[feature] for feature in failed_features],
            "coverage": [coverage[feature] for feature in failed_features],
            "staleness": [staleness[feature] for feature in failed_features],
        }
    ).sort("feature")
    with pl.Config(tbl_rows=screened_out.height):
        display(screened_out)

# %% [markdown]
# One more group has to be set aside before any ranking is attempted. Some columns
# describe the market rather than a pair - how far the dollar moved today, which
# volatility regime the model believes the market is in - so every pair carries the same
# value on a session. A column that is constant across the pairs cannot rank them, and
# the correlation this notebook computes is undefined on it. These columns are still
# useful: a model can read one alongside a ranking column and learn that the ranking
# behaves differently in a turbulent week. They are separated here so that the test they
# cannot pass is not run on them and reported as a failure.

# %%
date_level_features = []
for feature in all_feature_cols:
    if not correctness[feature]:
        continue
    values_per_date = eval_panel.group_by(DATE_COL).agg(
        pl.col(feature).drop_nulls().n_unique().alias("n_values")
    )
    fraction_constant = float((values_per_date["n_values"] <= 1).mean())
    if fraction_constant > MARKET_LEVEL_SHARE:
        date_level_features.append(feature)

print(
    f"{len(date_level_features)} columns hold one value for the whole market on a "
    f"session and cannot rank the pairs:"
)
for feature in date_level_features:
    print(f"  {feature} ({families[feature]})")

# %% [markdown]
# ## 3. Does the Column's Ranking Agree With Tomorrow's?
#
# The measurement at the centre of this stage is one number per session. On each
# session, rank the pairs by the column, rank them again by the return they went on to
# earn, and take the rank correlation between those two orderings. That correlation is
# the **information coefficient**, and it answers exactly the question a long-short rank
# strategy asks: not whether the column predicts a return, but whether it puts the pairs
# in the right order. A value of 0 is no agreement, 1 is the same ordering, -1 is the
# reverse ordering - and the reverse is as useful as the forward one, since a strategy
# can trade it upside down.
#
# One session gives one such number, so the column produces a series of them and the
# summary is that series' average. The series is what is stored and what everything
# below is computed from; it is never pooled into a single correlation over all
# pair-sessions at once, which would answer a different question, and never built by
# averaging fold averages, which weights a short fold like a long one.
#
# The average needs an interval around it, and the usual formula for one assumes each
# session's number is unrelated to the last. **Newey-West** drops that assumption: it
# estimates how much a session's value resembles nearby sessions and rescales the
# interval by what it finds. The number of neighbours it looks at is set from the label
# horizon, because that is how far two decisions have to be apart before their outcomes
# stop covering any of the same days.


# %%
def compute_cross_sectional_ic(
    df: pl.DataFrame,
    feature: str,
    return_col: str,
    min_periods: int = 5,
) -> pl.DataFrame:
    """One rank correlation per session, in date order, carrying the fold it came from."""
    rows = []
    for group in df.partition_by(DATE_COL, maintain_order=True):
        valid = group.select([feature, return_col]).drop_nulls()
        if len(valid) < min_periods:
            continue
        ic, _ = spearmanr(valid[feature].to_numpy(), valid[return_col].to_numpy())
        if np.isfinite(ic):
            rows.append(
                {
                    DATE_COL: group[DATE_COL][0],
                    "fold": int(group["fold"][0]),
                    "ic": float(ic),
                    "n_obs": len(valid),
                }
            )
    return pl.DataFrame(rows).sort(DATE_COL) if rows else pl.DataFrame()


# %%
evaluable_features = [
    feature
    for feature in all_feature_cols
    if correctness[feature] and feature not in date_level_features
]
computed = []
for feature in evaluable_features:
    series = compute_cross_sectional_ic(
        eval_panel, feature=feature, return_col=LABEL_COL, min_periods=MIN_PERIODS
    )
    if len(series) >= MIN_SESSIONS:
        computed.append(series.with_columns(pl.lit(feature).alias("feature")))

IC_SERIES_SCHEMA = {
    "feature": pl.String,
    DATE_COL: pl.Date,
    "fold": pl.Int64,
    "ic": pl.Float64,
    "n_obs": pl.Int64,
}
# Written before it is used, and read back below, so the figures and the averages come
# from the stored file rather than from a copy in memory that could differ from it.
ic_series_frame = (
    pl.concat(computed).select(*IC_SERIES_SCHEMA).cast(IC_SERIES_SCHEMA)
    if computed
    else pl.DataFrame(schema=IC_SERIES_SCHEMA)
)
ic_series_frame.write_parquet(EVAL_DIR / "ic_timeseries.parquet")

# %% [markdown]
# Everything from here reads the file just written. The averages, the intervals, the
# fold summaries and the figures are all derived from the same stored series, so what a
# reader loads from disk is what produced the decisions at the end.

# %%
stored_ic = pl.read_parquet(EVAL_DIR / "ic_timeseries.parquet")
ic_timeseries = {
    part["feature"][0]: part.drop("feature").sort(DATE_COL)
    for part in stored_ic.partition_by("feature")
}
ic_results = {
    feature: compute_ic_hac_stats(series, ic_col="ic", label_horizon=LABEL_HORIZON)
    for feature, series in ic_timeseries.items()
}

print(
    f"Wrote and read back evaluation/ic_timeseries.parquet: {len(stored_ic):,} scored "
    f"sessions over {len(ic_results)} of {len(evaluable_features)} eligible columns"
)

# %% [markdown]
# ### The Series Behind the Average
#
# A mean IC is a summary of a series, and two things it cannot show are the patterns
# most worth catching: an association that lives in one episode, and one that changes
# sign from fold to fold. The daily series is the primary object at this stage, so it
# is drawn before any scalar derived from it. Three intervals accompany the mean
# because each makes a different assumption about how the daily ICs depend on each
# other: the naive one treats every day as independent, Newey-West rescales it by the
# serial correlation the series actually has, and the block bootstrap resamples
# contiguous stretches rather than days.

# %%
IC_ROLLING_WINDOW = 63
BOOT_BOUNDS = ("ci_boot_lower", "ci_boot_upper")

leaders = sorted(ic_results, key=lambda name: abs(ic_results[name]["mean_ic"]), reverse=True)[
    :N_LEADERS
]
ic_uncertainty = {
    feature: compute_ic_uncertainty(ic_timeseries[feature], horizon=LABEL_HORIZON, ic_col="ic")
    for feature in leaders
}
leader = leaders[0] if leaders else None
if leader:
    leader_series = ic_timeseries[leader].with_columns(
        pl.col("ic").rolling_mean(IC_ROLLING_WINDOW).alias("rolling")
    )
print(f"Leading feature by absolute mean IC: {leader}")

# %%
if leader:
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(
            "Daily IC of the leading feature, under its rolling mean",
            "Mean IC against three ways of bounding it",
        ),
        horizontal_spacing=0.16,
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
            name=f"{IC_ROLLING_WINDOW}-session mean",
        ),
        row=1,
        col=1,
    )
    _ = fig.add_hline(
        y=0, line={"color": COLORS["neutral"], "width": 0.8, "dash": "dash"}, row=1, col=1
    )

# %% [markdown]
# The companion panel puts the three intervals on one axis for the same features, with
# the naive one as the grey band behind. The direction of the adjustment is not fixed:
# positively autocorrelated ICs widen the Newey-West interval and negatively
# autocorrelated ones narrow it. The primary label is a one-session forward return, so
# consecutive ICs score disjoint windows and the correction has little overlap to
# undo - the size of the gap is the thing to read, not its sign. It is the longer
# labels, whose windows do overlap, where the adjustment carries weight, and the
# horizon figure further down is where they are compared.


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
    interval_features = list(reversed(leaders))
    means = [ic_uncertainty[name]["mean_ic"] for name in interval_features]
    _ = fig.add_trace(
        go.Scatter(
            x=means,
            y=interval_features,
            mode="markers",
            marker={"color": COLORS["neutral"], "size": 1, "opacity": 0.0},
            error_x=interval_arms(interval_features, "ci_naive_lower", "ci_naive_upper")
            | {"color": COLORS["silver_muted"], "thickness": 9, "width": 0},
            name="Naive interval",
        ),
        row=1,
        col=2,
    )
    _ = fig.add_trace(
        go.Scatter(
            x=means,
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
        title="A small average IC sits inside a daily series that swings across zero",
        height=560,
        width=1150,
        margin={"l": 60, "r": 200},
        legend={"orientation": "h", "y": -0.18},
    )
    fig.update_yaxes(title_text="Daily Spearman IC", row=1, col=1)
    fig.update_xaxes(
        title_text=f"Validation session; mean rolls over {IC_ROLLING_WINDOW} sessions",
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Mean daily Spearman IC, 95% intervals", row=1, col=2)
    show_plotly_with_alt(
        fig,
        "Two panels. On the left, the daily rank correlation of the leading feature "
        "against time, a dense band swinging across the full range from minus one to "
        "one, with a rolling mean drawn over it that stays close to zero and wanders "
        "only slightly either side. On the right, the mean rank correlation of each "
        "leading column with three intervals drawn around it - naive, Newey-West and "
        "block-bootstrap. The three are close to the same width for every column, and "
        "each interval reaches across zero or ends very near it.",
    )

# %% [markdown]
# ### Did the Folds Agree, or Was It One Window?
#
# An average over eight years can come from eight years of the same weak effect or from
# one year of a strong one, and only the second is a reason to be sceptical. So the same
# average is taken again inside each fold's own validation stretch and the folds are
# compared.
#
# The summary is the share of folds pointing the same way as the column does over the
# whole span - not the share of folds that are positive. A column that ranks the pairs
# in reliably the reverse order is as tradable as one that ranks them the right way
# round, so scoring it on how often it is positive would make a perfectly steady inverse
# ranking unpromotable by construction.

# %%
fold_stats = {}
for feature, full_stats in ic_results.items():
    fold_means = []
    for fold in sorted(ic_timeseries[feature]["fold"].unique().to_list()):
        fold_values = (
            ic_timeseries[feature].filter(pl.col("fold") == fold).sort(DATE_COL)["ic"].to_numpy()
        )
        if len(fold_values) >= MIN_FOLD_DAYS:
            fold_means.append(float(np.mean(fold_values)))
    if not fold_means:
        continue
    direction = 1 if full_stats["mean_ic"] >= 0 else -1
    sign_consistency = sum((value * direction) > 0 for value in fold_means) / len(fold_means)
    fold_stats[feature] = {
        "n_folds": len(fold_means),
        "direction": "positive" if direction > 0 else "negative",
        "sign_consistency": sign_consistency,
        "worst_fold_ic": min(fold_means),
        "best_fold_ic": max(fold_means),
        "median_fold_ic": float(np.median(fold_means)),
        "fold_ics": fold_means,
    }

print(f"Fold-by-fold summary built for {len(fold_stats)} columns")

# %% [markdown]
# Each row below carries every fold's own average, the median of them, and the fold that
# went furthest against the column's overall direction. Reading across a row separates a
# column whose evidence repeated from one whose evidence came from a single window - and
# the worst fold is the one marked, rather than the lowest, because for a column that
# ranks in reverse the lowest fold is its best one.


# %%
def signed_direction(feature: str) -> int:
    """+1 where the feature's overall IC is positive, -1 where it is negative."""
    return 1 if fold_stats[feature]["direction"] == "positive" else -1


# %%
stability_features = [name for name in reversed(leaders) if name in fold_stats]
fig = go.Figure()
_ = fig.add_trace(
    go.Scatter(
        x=[value for name in stability_features for value in fold_stats[name]["fold_ics"]],
        y=[name for name in stability_features for _ in fold_stats[name]["fold_ics"]],
        mode="markers",
        marker={"color": COLORS["neutral"], "size": 8, "opacity": 0.6},
        name="Fold mean",
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
        x=[
            min(fold_stats[name]["fold_ics"], key=lambda value: value * signed_direction(name))
            for name in stability_features
        ],
        y=stability_features,
        mode="markers",
        marker={
            "color": COLORS["negative"],
            "size": 12,
            "symbol": "x-thin",
            "line": {"width": 2, "color": COLORS["negative"]},
        },
        name="Fold furthest against the feature's own direction",
    )
)
_ = fig.add_vline(x=0, line={"color": COLORS["neutral"], "width": 0.8, "dash": "dash"})
fig.update_layout(
    title="Folds disagree on the sign of most leading features",
    xaxis_title="Mean daily Spearman IC within the fold",
    height=520,
    width=1000,
    margin={"l": 200},
    legend={"orientation": "h", "y": -0.16},
)
show_plotly_with_alt(
    fig,
    "One row per leading feature, with a point for each fold's mean rank correlation, a "
    "diamond at the median fold and a cross on the fold furthest against the feature's "
    "own direction. Every row has points on both sides of the zero rule, and on most "
    "rows the spread across folds is several times the distance of the median from "
    "zero.",
)

# %% [markdown]
# ## 4. What Was Searched, and What That Costs
#
# Test sixty-odd columns at the conventional five-percent level against a label none of
# them predicts, and about three will still come back significant, because that is what
# the five-percent level means. A p-value is only readable against the set of tests it
# came out of, so that set is declared first and the adjustment is applied over it.
#
# Nothing here was chosen after an agreement score was seen. The price-derived columns
# come from the window register in `setup.yaml`, which fixes every trailing window
# before any of them is built. The model-derived columns come from the three estimators
# `04_model_based_features` fits - a local linear trend filter on each pair's price, a
# two-state regime model on the dollar, and a return forecast per pair - each of which
# contributes the quantities its own fit produces.
#
# **Benjamini-Hochberg** is the adjustment. Rather than demanding every p-value clear a
# far stricter bar, it sorts them and finds the largest set that can be called
# discoveries while holding the expected share of false ones among them to the same
# five percent. The counts before and after are reported separately, along with the
# count under the naive interval, so the reader can see what each correction costs.

# %%
searched_set = {
    "columns the feature stages generated": len(all_feature_cols),
    "of those, filled in and moving enough to test": sum(correctness.values()),
    "of those, able to rank the pairs and scored": len(ic_results),
    "label horizons the case study declares": len(LABEL_HORIZONS),
}
for description, count in searched_set.items():
    print(f"{description}: {count}")

# %%
feature_names = list(ic_results)
hac_p_values = [
    value if np.isfinite(value := ic_results[feature]["p_value"]) else 1.0
    for feature in feature_names
]
fdr_result = benjamini_hochberg_fdr(hac_p_values, alpha=0.05, return_details=True)

eval_summary = pl.DataFrame(
    {
        "feature": feature_names,
        "source": [
            "model_based" if feature in temporal_cols else "financial" for feature in feature_names
        ],
        "ic_mean": [ic_results[feature]["mean_ic"] for feature in feature_names],
        "naive_t": [ic_results[feature]["naive_t_stat"] for feature in feature_names],
        "hac_se": [ic_results[feature]["hac_se"] for feature in feature_names],
        "hac_t": [ic_results[feature]["t_stat"] for feature in feature_names],
        "hac_p": hac_p_values,
        "fdr_p": list(fdr_result["adjusted_p_values"]),
        "fdr_sig": list(fdr_result["rejected"]),
    },
    # Declared, so that a reduced run with no computable IC still yields a frame the
    # boolean filters below can read rather than an all-null one.
    schema={
        "feature": pl.String,
        "source": pl.String,
        "ic_mean": pl.Float64,
        "naive_t": pl.Float64,
        "hac_se": pl.Float64,
        "hac_t": pl.Float64,
        "hac_p": pl.Float64,
        "fdr_p": pl.Float64,
        "fdr_sig": pl.Boolean,
    },
).sort(pl.col("ic_mean").abs(), descending=True)

n_naive = sum(abs(ic_results[feature]["naive_t_stat"]) > 1.96 for feature in feature_names)
n_hac = sum(hac_p < 0.05 for hac_p in hac_p_values)
n_fdr = int(fdr_result["n_rejected"])
print(f"Columns clearing 5% treating each session as independent: {n_naive}")
print(f"Columns clearing 5% once Newey-West rescales the interval: {n_hac}")
print(f"Columns clearing 5% after Benjamini-Hochberg over the set: {n_fdr}")

leading_row = eval_summary.row(0, named=True) if len(eval_summary) else None
if leading_row:
    print(
        f"Largest agreement in absolute terms: {leading_row['feature']}, "
        f"mean {leading_row['ic_mean']:+.4f} per session, "
        f"Newey-West t {leading_row['hac_t']:+.2f}"
    )

# %% [markdown] tags=["results"]
# Of the 63 columns the feature stages produced, 59 are filled in and moving enough to
# test and 54 can rank the pairs, so 54 were scored over 8 folds. One of them clears the
# five-percent level once Newey-West rescales its interval, and none clears the
# Benjamini-Hochberg adjustment over the set. The largest agreement in absolute terms is
# accel_63_126, at a mean of -0.0178 per session and a Newey-West t of -1.85 - a column
# that ranks the pairs in reverse, weakly, and not distinguishably from zero.

# %%
top_n = min(20, len(eval_summary))
top = eval_summary.head(top_n).sort("ic_mean")

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Ranked mean IC, split by the false-discovery decision",
        "Newey-West leaves most t-statistics near naive estimates",
    ),
    horizontal_spacing=0.18,
)
for cleared, color, name in (
    (True, COLORS["blue"], "Cleared BH-FDR"),
    (False, COLORS["amber"], "Did not clear"),
):
    arm = top.filter(pl.col("fdr_sig") == cleared)
    _ = fig.add_trace(
        go.Bar(
            x=arm["ic_mean"],
            y=arm["feature"],
            orientation="h",
            marker_color=color,
            text=[f"{value:+.3f}" for value in arm["ic_mean"]],
            textposition="inside",
            name=name,
        ),
        row=1,
        col=1,
    )

# %% [markdown]
# The companion panel compares naive and Newey-West inference. Points near the
# diagonal have little serial-correlation adjustment; departures show where it
# matters for the significance screen.

# %%
fig.add_trace(
    go.Scatter(
        x=eval_summary["naive_t"],
        y=eval_summary["hac_t"],
        mode="markers",
        marker={"color": COLORS["blue"], "size": 7, "opacity": 0.75},
        text=eval_summary["feature"],
        showlegend=False,
    ),
    row=1,
    col=2,
)
finite_t = [
    abs(float(value))
    for column in ("naive_t", "hac_t")
    for value in eval_summary[column]
    if np.isfinite(value)
]
if finite_t:
    t_limit = max(finite_t) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[-t_limit, t_limit],
            y=[-t_limit, t_limit],
            mode="lines",
            line={"color": COLORS["neutral"], "dash": "dash"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
fig.update_layout(
    title="No candidate column clears the false-discovery adjustment",
    height=620,
    width=1100,
    margin={"l": 180},
    barmode="relative",
    legend={"orientation": "h", "y": -0.16},
)
fig.update_xaxes(title_text="Mean daily Spearman IC", row=1, col=1)
fig.update_xaxes(title_text="Naive t-statistic", row=1, col=2)
fig.update_yaxes(title_text="Newey-West t-statistic", row=1, col=2)
show_plotly_with_alt(
    fig,
    "Two panels. On the left, horizontal bars of mean rank correlation for the candidate "
    "columns, ranked from the most positive at the top through zero to the most negative "
    "at the bottom, each labelled with its value and all drawn in the single colour the "
    "legend gives to columns that did not clear the false-discovery adjustment. On the "
    "right, the Newey-West t-statistic against the naive one, with the points lying "
    "along the diagonal, so correcting for overlap moves the t-statistics very little.",
)

# %% [markdown]
# ### The Same Columns Against the Longer Labels
#
# Everything above was measured against the next session's return. The case study also
# declares a one-week and a one-month forward return, and how long a column stays
# informative decides how long a position built on it can be held. A column that agrees
# with tomorrow's ranking and nothing beyond it supports a strategy that rebalances
# daily and pays the spread every day for the privilege.
#
# Each label gets its own gap in front of the holdout, sized to its own horizon, so the
# one-month label is evaluated over a slightly shorter stretch than the one-day label.
# Beside the average, the second panel shows it divided by how much it varies from fold
# to fold, which separates a small agreement that repeats from a larger one that came
# out of one window.


# %%
def build_horizon_panel(label_name: str) -> pl.DataFrame:
    """Join the validation rows to one declared label, with that label's own end gap."""
    label_frame = pl.read_parquet(CASE_DIR / "labels" / f"{label_name}.parquet")
    panel = (
        validation_rows(fold_windows(LABEL_BUFFERS[label_name]))
        .join(financial, on=JOIN_COLS, how="inner")
        .join(label_frame, on=JOIN_COLS, how="inner")
        .sort([DATE_COL, "symbol"])
    )
    if MAX_SYMBOLS > 0:
        panel = panel.filter(pl.col("symbol").is_in(selected_symbols))
    return panel


# %%
horizon_rows = []
for label_name, horizon in sorted(LABEL_HORIZONS.items(), key=lambda item: item[1]):
    panel = build_horizon_panel(label_name)
    assert panel[DATE_COL].max() < HOLDOUT_START, f"{label_name} panel reaches the holdout"
    for feature in leaders:
        series = compute_cross_sectional_ic(
            panel, feature=feature, return_col=label_name, min_periods=MIN_PERIODS
        )
        if len(series) < 20:
            continue
        fold_means = [
            float(part["ic"].mean())
            for part in series.partition_by("fold")
            if len(part) >= MIN_FOLD_DAYS
        ]
        dispersion = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else np.nan
        horizon_rows.append(
            {
                "feature": feature,
                "horizon": horizon,
                "ic_mean": float(series["ic"].mean()),
                "icir": float(np.mean(fold_means)) / dispersion if dispersion else np.nan,
            }
        )

horizon_ic = pl.DataFrame(horizon_rows)
print(f"Horizon profile computed for {len(horizon_rows)} feature-horizon pairs")

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Mean IC by forward horizon",
        "Fold-level information ratio by forward horizon",
    ),
    horizontal_spacing=0.12,
)
shown_direction = set()
for feature in leaders if horizon_rows else []:
    profile = horizon_ic.filter(pl.col("feature") == feature).sort("horizon")
    if not len(profile):
        continue
    positive = float(profile["ic_mean"][0]) > 0
    group = "Positive at the primary horizon" if positive else "Negative at the primary horizon"
    style = {
        "color": COLORS["blue"] if positive else COLORS["copper"],
        "width": 2.5 if feature == leader else 1.2,
    }
    for column, column_index in (("ic_mean", 1), ("icir", 2)):
        first = column_index == 1 and group not in shown_direction
        _ = fig.add_trace(
            go.Scatter(
                x=profile["horizon"],
                y=profile[column],
                mode="lines+markers",
                line=style,
                opacity=0.85,
                name=group,
                legendgroup=group,
                showlegend=first,
                hovertext=feature,
            ),
            row=1,
            col=column_index,
        )
    shown_direction.add(group)
_ = fig.add_hline(y=0, line={"color": COLORS["neutral"], "width": 0.8, "dash": "dash"})
fig.update_layout(
    title="The leading columns agree more with the longer horizons than the next day",
    height=520,
    width=1100,
    legend={"orientation": "h", "y": -0.22},
)
horizon_ticks = sorted(LABEL_HORIZONS.values())
for column_index in (1, 2):
    fig.update_xaxes(
        title_text="Forward horizon (sessions)",
        tickmode="array",
        tickvals=horizon_ticks,
        row=1,
        col=column_index,
    )
fig.update_yaxes(title_text="Mean Spearman IC", row=1, col=1)
fig.update_yaxes(title_text="Mean fold IC / dispersion across folds", row=1, col=2)
show_plotly_with_alt(
    fig,
    "Two panels against forward horizon, at one, five and twenty-one sessions, with one "
    "line per leading column coloured by the sign it takes at the primary horizon. On "
    "the left, mean rank correlation: the lines fan out from a tight cluster near zero "
    "at one session, the positive group rising and part of the negative group falling "
    "further, so agreement with the label is larger at the longer horizons. On the "
    "right, the same columns as a fold-level information ratio, where the lines are much "
    "flatter across horizons and stay on the side of zero they started.",
)

# %% [markdown]
# ## 5. Is the Relationship the Shape a Ranking Strategy Needs?
#
# A rank correlation is one number and it hides the shape behind it. A column can agree
# with tomorrow's ordering because the top few pairs do well and the rest are noise, or
# because the relationship reverses in the middle, and both look the same as an average.
# A strategy that goes long the top bucket and short the bottom one needs the mean
# return to climb across the buckets, not just to differ at the ends.
#
# So on each session the pairs are sorted by the column and split into five buckets, and
# the returns they went on to earn are averaged inside each bucket on that session. Those
# session averages are then averaged across sessions, so every session counts once. That
# is what a book rebalanced each session earns, and it is the unit the rank correlation
# above already uses, so the two are comparable rather than merely adjacent.
#
# The bucket edges are cut **within** each session, from that session's twenty values
# alone. The natural instinct is to cut them once over the whole sample, and it is worth
# naming because it fails without any sign of failing: bucket edges taken over the whole
# sample are set by every other session's levels, so on a calm week every pair lands in
# the low buckets and the profile then describes when the market was volatile rather than
# which pairs ranked highest.
#
# A session enters the profile on the same two terms it enters the correlation on: enough
# pairs quoting both the column and the return to be worth ranking, and enough such
# sessions behind the column to average over. Building the two on different sets would
# leave a column whose score and whose shape describe different weeks.
#
# The rank correlation between bucket number and bucket mean summarizes how close the
# profile is to a staircase. It is computed for every scored column and recorded in the
# ledger beside that column's decision, while the panels below draw the few with the
# largest average agreement. The triage rule does not read the score, because a shape
# that is not a staircase is a reason to model a column differently rather than to drop
# it.

# %%
monotonicity_scores = {}
quantile_spreads = {}
for feature in eval_summary["feature"].to_list():
    profile = quantile_profile(
        eval_panel,
        feature,
        LABEL_COL,
        date_col=DATE_COL,
        n_quantiles=N_QUANTILES,
        min_cross_section=MIN_PERIODS,
    )
    if profile is None or profile.periods_used < MIN_SESSIONS:
        continue
    quantile_spreads[feature] = profile.means
    monotonicity_scores[feature] = profile.monotonicity

features_to_show = list(quantile_spreads)[:N_SHAPE_PANELS]
print(
    f"Bucket profile built for {len(quantile_spreads)} of the {len(eval_summary)} scored "
    f"columns; the {len(features_to_show)} with the largest average agreement are drawn:"
)
if features_to_show:
    shape_table = pl.DataFrame(
        {
            "feature": features_to_show,
            "family": [families[name] for name in features_to_show],
            "staircase_score": [monotonicity_scores[name] for name in features_to_show],
        }
    ).sort("feature")
    with pl.Config(tbl_rows=shape_table.height):
        display(shape_table)

# %%
if quantile_spreads:
    figure_rows = (len(features_to_show) + 2) // 3
    fig = make_subplots(rows=figure_rows, cols=3, subplot_titles=features_to_show)
    quantile_colors = [
        COLORS["negative"],
        COLORS["copper"],
        COLORS["neutral"],
        COLORS["amber"],
        COLORS["positive"],
    ]
    for index, feature in enumerate(features_to_show):
        row, column = divmod(index, 3)
        fig.add_trace(
            go.Bar(
                x=[f"Q{quantile + 1}" for quantile in range(N_QUANTILES)],
                y=quantile_spreads[feature],
                marker_color=quantile_colors,
                showlegend=False,
            ),
            row=row + 1,
            col=column + 1,
        )
    fig.update_layout(
        title="Bucketing within the session shows monotone and non-monotone shapes alike",
        height=280 * figure_rows,
        width=1000,
    )
    # One shared y range, so a weak profile cannot be rescaled to look like a strong one.
    span = max(abs(value) for feature in features_to_show for value in quantile_spreads[feature])
    fig.update_yaxes(range=[-1.1 * span, 1.1 * span], tickformat=".0e")
    for row in range(1, figure_rows + 1):
        fig.update_yaxes(title_text="Mean next-day return", row=row, col=1)
    show_plotly_with_alt(
        fig,
        "A small-multiple grid, one panel per leading column, each showing the mean "
        "next-day return of the five within-session buckets from the lowest to the "
        "highest value of that column. All panels share one vertical range. Some rise "
        "or fall monotonically across the buckets, and others turn in the middle or put "
        "their extreme in an interior bucket, so a similar mean correlation is produced "
        "by quite different shapes.",
    )

# %% [markdown]
# ## 6. How Much of This Is the Same Evidence Counted Twice?
#
# The family table at the top showed several ideas measured over four or five trailing
# windows each. Two such columns can agree so closely on how they order the pairs that a
# model gains nothing from having both, and the multiplicity adjustment in part 4 has
# already paid for testing both as if they were separate questions.
#
# The measurement is the rank correlation between every pair of columns, taken over a
# sample of the validation sessions rather than all of them, which is enough for an
# ordering this coarse. Pairs above the cut are shown ranked, which stays readable at
# fifty-odd column names where a full matrix does not.
#
# Columns linked by such a pair form a group, following the links transitively: if A
# agrees with B and B with C, all three are one piece of evidence even where A and C were
# never compared directly. One column in each group stands for it, chosen by the largest
# median fold agreement and, where those tie, the steadiest across folds. Standing for a
# group promotes nothing - that column still has to earn its own decision below, and a
# group whose representative fails contributed nothing.
#
# What to watch for in the group table is a single group swallowing most of the matrix.
# Linking transitively at one threshold chains: a column sitting just above the cut
# against its neighbour, which sits just above the cut against the next, ends up in the
# same group as something it has almost nothing in common with. The `weakest_link` column
# is how far that went - it is the smallest correlation holding the group together, and
# when it sits at the cut while the group holds most of the columns, the group is an
# artefact of chaining rather than a claim that those columns are interchangeable. That
# is why the grouping is reported here and used to drop nothing.


# %%
sample_step = max(1, n_dates // 200)
sample_dates = eval_panel[DATE_COL].unique().sort().to_list()[::sample_step]
correlation_data = eval_panel.filter(pl.col(DATE_COL).is_in(sample_dates)).select(
    evaluable_features
)
correlation_matrix = correlation_data.to_pandas().corr(method="spearman")

high_correlation_pairs = []
for left_index in range(len(correlation_matrix)):
    for right_index in range(left_index + 1, len(correlation_matrix)):
        correlation = float(correlation_matrix.iloc[left_index, right_index])
        if np.isfinite(correlation) and abs(correlation) > REDUNDANCY_CUT:
            high_correlation_pairs.append(
                {
                    "left": str(correlation_matrix.columns[left_index]),
                    "right": str(correlation_matrix.columns[right_index]),
                    "correlation": correlation,
                }
            )

high_correlation_pairs.sort(key=lambda row: abs(row["correlation"]), reverse=True)
print(
    f"{len(high_correlation_pairs)} column pairs agree above "
    f"{REDUNDANCY_CUT:.2f} in absolute rank correlation, out of "
    f"{len(evaluable_features) * (len(evaluable_features) - 1) // 2} pairs compared"
)

# %% [markdown]
# ### Which Column Stands for Its Group
#
# The pairs above are collapsed into groups first, then one column per group is chosen.
# A column with no fold summary - one that was screened out before the folds were
# computed - cannot be compared on the criterion, so it is ranked last within its group
# rather than removed from it.


# %%
def cluster_key(feature: str) -> tuple[float, float]:
    """Rank within a group: largest median fold agreement, then steadiest across folds."""
    stats = fold_stats.get(feature)
    if not stats:
        return (-1.0, 0.0)
    return (abs(stats["median_fold_ic"]), -float(np.std(stats["fold_ics"], ddof=0)))


# %%
component_of: dict[str, str] = {feature: feature for feature in evaluable_features}


def root(feature: str) -> str:
    """Label of the group a column currently sits in, following the links transitively."""
    while component_of[feature] != feature:
        component_of[feature] = component_of[component_of[feature]]
        feature = component_of[feature]
    return feature


for pair in high_correlation_pairs:
    left_root, right_root = root(pair["left"]), root(pair["right"])
    if left_root != right_root:
        component_of[left_root] = right_root

clusters: dict[str, list[str]] = {}
for feature in evaluable_features:
    clusters.setdefault(root(feature), []).append(feature)

representative = {}
for members in clusters.values():
    chosen = max(members, key=cluster_key)
    for member in members:
        representative[member] = chosen

redundant_clusters = {name: members for name, members in clusters.items() if len(members) > 1}
print(
    f"{len(evaluable_features)} scored columns collapse to {len(clusters)} groups, of "
    f"which {len(redundant_clusters)} hold more than one column"
)
group_rows = [
    {
        "stands_for": representative[members[0]],
        "columns_in_group": len(members),
        "median_fold_ic": fold_stats[representative[members[0]]]["median_fold_ic"],
        "fold_spread": float(np.std(fold_stats[representative[members[0]]]["fold_ics"], ddof=0)),
        "weakest_link": min(
            (
                abs(pair["correlation"])
                for pair in high_correlation_pairs
                if representative[pair["left"]] == representative[members[0]]
            ),
            default=float("nan"),
        ),
    }
    for members in redundant_clusters.values()
    if representative[members[0]] in fold_stats
]
if group_rows:
    group_table = pl.DataFrame(group_rows).sort("columns_in_group", descending=True)
    with pl.Config(tbl_rows=group_table.height):
        display(group_table)


# %%
def pair_label(pair: dict) -> str:
    """Name both members, marking whichever of them stands for their group."""
    return " / ".join(
        f"{member}*" if representative[member] == member else member
        for member in (pair["left"], pair["right"])
    )


# %%
if high_correlation_pairs:
    correlation_plot = pl.DataFrame(
        [
            {"pair": pair_label(pair), "correlation": pair["correlation"]}
            for pair in high_correlation_pairs[:20]
        ]
    ).sort("correlation")
    fig = go.Figure(
        go.Bar(
            x=correlation_plot["correlation"],
            y=correlation_plot["pair"],
            orientation="h",
            marker_color=[
                COLORS["blue"] if value > 0 else COLORS["copper"]
                for value in correlation_plot["correlation"]
            ],
            text=[f"{value:+.2f}" for value in correlation_plot["correlation"]],
            textposition="inside",
        )
    )
    fig.update_layout(
        title="Many engineered features carry nearly identical rank information",
        xaxis_title=(
            "Rank correlation on sampled validation sessions; "
            "* marks the column standing for its group"
        ),
        height=650,
        width=1000,
        margin={"l": 320},
    )
    show_plotly_with_alt(
        fig,
        "Horizontal bars of absolute rank correlation for the most correlated pairs of "
        "feature columns, each row labelling the two columns and marking with an "
        "asterisk the one standing for its group. Every bar reaches past 0.95 and the "
        "longest reach 1.00, so these pairs order the cross-section almost "
        "identically.",
    )

# %% [markdown]
# ## 7. One Decision per Column
#
# The output of this stage is not a shortlist. It is a decision per column with the
# evidence attached, in the book's three categories.
#
# - `PROCEED`: worth carrying into multivariate work. It is not a model-selection
#   decision, and it says nothing about whether the column ends up in a trained model.
# - `REVISE`: nothing usable was found by this test, and this test is narrow. A column
#   that describes the market rather than the pairs lands here, and so does one whose
#   effect only appears alongside another column, because a one-at-a-time rank
#   correlation cannot see either.
# - `STOP`: the column failed a screen in part 2. Nothing was measured on it because
#   nothing measured on it would mean anything.
#
# Two routes lead to `PROCEED`, and the ledger records which one a column took. The
# first is a confirmation: the column cleared the false-discovery adjustment over the
# declared search. The second is a search rather than a confirmation, in the sense of
# book Section 7.4: it promotes on agreeing in most folds and on being larger than the
# effect-size floor, and it exists so that a twenty-pair cross-section, where almost
# nothing clears a multiplicity adjustment, does not leave the next stage with nothing
# to model. A column promoted that way has not been confirmed by anything, and the
# `note` column in the ledger is what lets a reader tell the two apart.

# %%
fdr_significant = set(eval_summary.filter(pl.col("fdr_sig"))["feature"].to_list())
triage = {}
for feature in all_feature_cols:
    if not correctness[feature]:
        triage[feature] = ("STOP", "correctness_fail")
    elif feature in date_level_features:
        triage[feature] = ("REVISE", "date_level_conditioner")
    elif feature not in ic_results:
        triage[feature] = ("REVISE", "insufficient_validation_data")
    elif feature in fdr_significant:
        triage[feature] = ("PROCEED", "fdr_significant")
    elif (
        fold_stats.get(feature, {}).get("sign_consistency", 0) >= STABILITY_THRESHOLD
        and abs(ic_results[feature]["mean_ic"]) >= IC_THRESHOLD
    ):
        triage[feature] = ("PROCEED", "stable_and_above_threshold")
    else:
        triage[feature] = ("REVISE", "weak_standalone_association")

# %% [markdown]
# ### Write the Ledger
#
# The ledger keeps every quantity the decision rested on, not just the decision: the
# average agreement and its Newey-West t-statistic, the adjusted p-value, which way the
# column pointed and in how many folds, the shape score, and the two part-2 screens. A
# reader who disagrees with the rule can re-derive a different one from the same file
# without re-running anything.

# %%
ledger_rows = []
for feature in all_feature_cols:
    decision, note = triage[feature]
    summary_match = eval_summary.filter(pl.col("feature") == feature)
    ledger_rows.append(
        {
            "feature": feature,
            "family": families[feature],
            "source": "model_based" if feature in temporal_cols else "financial",
            "ic_mean": ic_results.get(feature, {}).get("mean_ic"),
            "hac_t": ic_results.get(feature, {}).get("t_stat"),
            "hac_p": ic_results.get(feature, {}).get("p_value"),
            "fdr_p": summary_match["fdr_p"][0] if len(summary_match) else None,
            "fdr_sig": bool(summary_match["fdr_sig"][0]) if len(summary_match) else False,
            "fold_direction": fold_stats.get(feature, {}).get("direction"),
            "sign_consistency": fold_stats.get(feature, {}).get("sign_consistency"),
            "worst_fold_ic": fold_stats.get(feature, {}).get("worst_fold_ic"),
            "monotonicity": monotonicity_scores.get(feature),
            "coverage": coverage[feature],
            "staleness": staleness[feature],
            "decision": decision,
            "note": note,
        }
    )

triage_ledger = pl.DataFrame(ledger_rows).sort(["decision", "feature"])
triage_ledger.write_parquet(EVAL_DIR / "triage_ledger.parquet")

print(f"Wrote evaluation/triage_ledger.parquet: one row for each of {len(triage_ledger)} columns")
decision_table = (
    triage_ledger.group_by(["decision", "note"])
    .agg(pl.len().alias("columns"))
    .sort(["decision", "note"])
)
with pl.Config(tbl_rows=decision_table.height):
    display(decision_table)

# %% [markdown] tags=["results"]
# The ledger records 27 PROCEED, 32 REVISE and 4 STOP decisions over the 63 columns.
# Every one of the 27 came through the exploration route, so nothing in this case study
# advances on a confirmed association, and the next stage starts from a set of
# candidates rather than from findings.

# %% [markdown]
# ## Key Takeaways
#
# Testing a column one at a time answers whether it carries information about the label
# on its own. It answers nothing about whether a model built on several of them trades,
# and the two questions come apart often enough that the second is asked separately in
# every case study in the book.
#
# What transfers to a reader's own data is the order the steps run in, and the reason
# each one comes where it does.
#
# 1. Close the gap in front of the holdout before measuring anything, sized to the
#    label's own horizon, so that no diagnostic has read a price the final measurement
#    is supposed to see for the first time.
# 2. Throw out the columns that are mostly empty or mostly unchanged first. They are
#    cheap to detect and they produce p-values that look exactly like the others.
# 3. Compute the agreement one decision time at a time and average the series. Let an
#    estimator that allows for one session resembling the next put the interval around
#    the average, rather than assuming the sessions are independent.
# 4. Ask whether the folds agreed before believing the average, and score agreement
#    against the column's own direction so an inverse ranking is not disqualified for
#    being inverse.
# 5. Declare the search before reading a p-value out of it, and adjust over the whole
#    declared set.
# 6. Record a decision per column with the evidence behind it, rather than a shortlist.
#    A shortlist discards the reasons, and the reasons are what the next stage needs.
#
# Three limits travel with the result. A rank correlation reads only a relationship that
# runs one way throughout, so a column that matters through an interaction with another,
# or only past a threshold, is invisible here and lands in `REVISE` rather than `STOP`.
# The exploration route promotes on steadiness and size rather than on significance, so
# a `PROCEED` that came through it is a candidate and not a finding. And with twenty
# pairs, one session's agreement is estimated from twenty observations, so the daily
# series is noisy by construction and its average needs the whole span to say anything -
# which is why the fold-by-fold view is not optional here.
#
# **Next**: `06_linear.py` fits the first models that read several columns at once. It
# takes the same feature matrix rather than the decisions recorded here: the ledger says
# what a one-at-a-time test found, and it does not gate what a model may use.
