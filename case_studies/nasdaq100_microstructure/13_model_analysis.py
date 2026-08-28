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
# # Model Analysis: NASDAQ-100 Microstructure
#
# This notebook evaluates all predictive models trained on the NASDAQ-100
# microstructure case study and answers a single question: **which learned
# signals are real, stable, and worth taking into a backtest?**
#
# This is the highest-frequency case study in the book. The universe spans
# 114 NASDAQ-100 constituents at 15-minute bar frequency, where the
# prediction target (`fwd_ret_15m`) is the next 15-minute return. The
# fundamental question here is different from daily case studies: at
# intraday horizons, does microstructure information - order flow
# imbalances, spread dynamics, volume patterns - contain any predictive
# content, or is it just noise?
#
# Two things make the evidence here thinner than the row count suggests.
# There are only two walk-forward folds over 2020-2021, so two independent
# windows stand behind every stability statement. And a rank correlation at this
# horizon is small by construction: the expected return per bar is a fraction of
# a basis point, the same order as the bid-ask spread a trade has to cross.
# Whether a correlation of that size is worth anything is a question about costs
# and rebalancing frequency, settled in the backtest and cost notebooks.
#
# **Learning Objectives**:
# - Choose one representative prediction set per model family on a condition that
#   makes the representatives comparable to each other
# - Read a rank correlation together with its interval, and say when two families
#   are not separated by the evidence
# - Check whether a model's prediction intervals hold their stated coverage on a
#   later period, which is a different question from whether it ranks well
# - Keep estimated treatment effects out of a ranking of predictive scores
#
# **Prerequisites**: Model training notebooks Ch11–15 must have run for this
# case study. Linear and GBM results come from the registry; DL results come
# from the training pipeline.
#
# **Book Reference**: This notebook bridges Part III (Models, Ch11–15) and
# Part IV (Strategy Implementation, Ch16–20). The chapter insights notebooks
# in Ch11–15 compare each model family *across* case studies; here we compare
# all families *within* a single dataset.

# %%
"""Compare every declared model family on one registered, complete population."""

import warnings

import numpy as np
import polars as pl
import yaml

from case_studies.research import (
    OfficialPopulation,
    Result,
    Study,
    superseded_members,
)
from case_studies.utils.model_analysis import (
    best_model_per_family_fast,
    fold_performance_matrix,
    load_all_metrics,
    load_fold_metrics_from_registry,
    load_gbm_feature_importance,
    load_predictions,
    prediction_bucket_monotonicity,
    prediction_correlation_matrix,
    regime_conditional_ic,
)
from case_studies.utils.model_viz import (
    plot_bucket_monotonicity,
    plot_correlation_matrix,
    plot_cv_timeline,
    plot_feature_importance_heatmap,
    plot_fold_boxplot,
    plot_fold_heatmap,
    plot_label_horizon_forest,
    plot_learning_curves,
    plot_regime_bars,
)
from case_studies.utils.notebook_contracts import (
    degenerate_prediction_hashes,
    excluded_families,
    filter_active_model_rows,
)
from case_studies.utils.notebook_render import conformal_coverage_diagnostic
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY = "nasdaq100_microstructure"
PRIMARY_LABEL = "fwd_ret_15m"
REGRESSION_LABELS = ["fwd_ret_15m", "fwd_ret_5m", "fwd_ret_60m"]
DATE_COL = "timestamp"
ENTITY_COL = "symbol"
N_BUCKETS = 10
TOP_N_FEATURES = 15
REGIME_WINDOW = 252
# The populations 06_linear and 07_gbm publish. Named rather than hashed: a name resolves to the
# generation in force, so a refit that supersedes its predecessor is picked up without editing
# this notebook, while every superseded snapshot stays readable by hash.
LINEAR_POPULATION = "nasdaq100_microstructure-linear-validation-v1"
GBM_POPULATION = "nasdaq100_microstructure-gbm-validation-v1"

# %% [markdown]
# This notebook reads; it registers nothing. That decides how it opens the registry, and the
# distinction is not cosmetic. Every route through `open_study` ends in `Study.activate()`,
# which rewrites `ML4T_OUTPUT_DIR` for the rest of the process and clears the caches keyed on
# it, so every later `get_case_study_dir` answers for a different directory than the one
# resolved here. On the canonical tier with no workspace that route is `Study.regenerate`,
# which refuses outright unless `features`, `labels` and `run_log` are symlinks - true in a
# maintainer worktree, false in every clean clone. On the preview tier it repoints the notebook
# at `.preview/<case>`, whose registry `activate()` creates *empty*: measured here, the catalog
# went from 30 registered prediction sets to 0 and the comparison below reported on nothing
# while reporting success.
#
# `Study.at` is the read-only form: one root, no activation. `CASE_DIR` is that root, and every
# question this notebook asks - the catalog, the lineage, the populations, the artifacts - is
# answered from it.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY)
study = Study.at(CASE_DIR, case_study=CASE_STUDY, entry_point="13_model_analysis")

with open(CASE_DIR / "config" / "setup.yaml") as f:
    setup = yaml.safe_load(f)

n_splits = setup["evaluation"]["n_splits"]
train_size = setup["evaluation"]["train_size"]
val_size = setup["evaluation"]["val_size"]
holdout_start = setup["evaluation"].get("holdout_start")
n_assets = setup["universe"]["n_assets"]

# Cost structure: nasdaq100 uses friction_floor_bps + spread_assumptions
cost_range = setup["costs"].get(
    "per_leg_cost_bps_range",
    [
        setup["costs"].get("spread_assumptions", {}).get("large_cap_bps", [1, 3])[0],
        setup["costs"].get("friction_floor_bps", 5),
    ],
)

print(f"Case Study: {CASE_STUDY}")
print(f"  Universe: {n_assets} NASDAQ-100 stocks at 15-minute bar frequency")
print(f"  Label: {PRIMARY_LABEL} (next 15-min return)")
print(f"  CV: {n_splits} walk-forward folds, train={train_size}, val={val_size}")
print(f"  Holdout: {holdout_start} onwards")
print(f"  Trading costs: {cost_range[0]}–{cost_range[1]} bps per leg")
if excluded_families(CASE_STUDY):
    print(
        "  Active-model filter: excluding "
        f"{', '.join(sorted(excluded_families(CASE_STUDY)))} pending corrected reruns"
    )

# %% [markdown]
# ## 1. What Is the Prediction Problem?
#
# **Primary target tuple**: `fwd_ret_15m` | regression | IC | 15-minute rebalancing
#
# We predict the next 15-minute return for each NASDAQ-100 constituent, ranking
# them against each other at each decision time. What the model has to produce is
# an ordering across the cross-section at one moment, repeated through the
# trading day.
#
# The feature set is entirely microstructure-based:
# relative spread, depth imbalance, quote arrival rate, microprice
# deviation, signed volume share, tick imbalance, trade-to-mid
# distance, realized volatility at multiple horizons (5m, 15m, 30m),
# Kyle's lambda (price impact), FINRA off-exchange share, and
# time-of-day indicators. Each raw feature also has a cross-sectional
# z-score variant, named with an `xs` suffix, capturing how extreme a stock's
# microstructure state is relative to the rest of the universe at that
# moment.
#
# This is fundamentally different from the daily momentum-based case
# studies. Here the signal, if any, comes from order flow pressure,
# liquidity imbalances, and short-lived mispricings - not from
# multi-week price trends or carry. The universe is homogeneous
# (large-cap US tech stocks), so cross-sectional dispersion is lower
# than in multi-asset or size-diverse universes.
#
# With only 2 walk-forward folds (6 months training, 6 months
# validation each) over 2020–2021, the statistical evidence is
# inherently limited. Any finding must be interpreted with extreme
# caution.

# %% [markdown]
# Every row below is one registered prediction set, and one condition decides which are
# admissible: `complete`. It is applied here rather than at the point each figure is drawn, so
# that no chart can be built from a row the comparison would not accept.
#
# `complete` is more than "the run finished". A run that failed partway still leaves rows in the
# registry, and its score is an average over the folds that finished, which is not the quantity
# every other row reports. The catalog also requires a current identity before it calls a row
# complete (`case_studies/research/catalog.py:308-314`): a row registered before the identity
# scheme carries no digest of the labels and features it was fitted on, so nothing can establish
# what it was trained against. It stays inspectable and cannot enter a comparison.
#
# Execution tier is deliberately not tested. A preview is a reduced computation in a throwaway
# workspace, so whether preview rows are in front of this notebook at all is settled by which
# registry `study` opened, not by a column. Re-asserting it as a filter would reject a preview
# run's own rows unconditionally, which is the defect `8fc28044` fixed on the registry path.
#
# This is the correction that unfroze this notebook. Selection used to run over whatever the
# registry held, filtered only by which families the case study excludes - so the representative
# of a family could be a legacy or partial row, and the table gave no sign of it.

# %%
# Admissibility is decided on the catalog and applied to the metrics frame by a join on
# prediction_hash. The catalog is the authority on whether a row may be compared; the metrics
# loader carries the richer per-row detail the figures below read - ic_mean_daily, the HAC
# interval, the t statistic - which the catalog's metric projection does not expose. Reading
# admissibility off one and detail off the other keeps both, where swapping wholesale to the
# catalog would have narrowed the columns and silently emptied the interval on every forest tile.
# `complete` is about the row; supersession is about the name that published it. A refit
# leaves the retired generation complete and current - the schema version it was written under
# has not moved - so a comparison selecting on the catalog alone puts two generations of the
# same name in one table, and the family representative below can be drawn from either. The
# table would not look wrong; it would just be answering two questions at once.
#
# Both calls take the study, and on a `Study.at` handle that is the same thing as naming the
# directory: `root` and `release_case_root` are both `CASE_DIR`, and `PredictionCatalog.table`
# short-circuits on a read-only study to read that one registry. The sibling calls in
# `14_backtest` are `prediction_rows_at` and `superseded_members_at`, which take the directory
# directly, because that notebook registers backtests and so holds no study at all.
_catalog = study.predictions.table()
_retired = superseded_members(study)
_admissible = pl.col("complete") & ~pl.col("prediction_hash").is_in(list(_retired))
_ok = _catalog.filter(_admissible).select("prediction_hash")
_rejected = _catalog.filter(~_admissible)

# `load_all_metrics` returns a bare frame with no columns when the registry holds no scored
# rows, so every expression below it fails on a missing column instead of on the absence that
# caused it. The distinction is worth one branch: "nothing has been fitted into this registry"
# is a state a reader can act on, and `unable to find column "label"; valid columns: []` is not.
_raw_metrics = load_all_metrics(CASE_STUDY, label=None)
if _raw_metrics.is_empty():
    msg = (
        f"{CASE_DIR} holds no scored prediction sets, so there is nothing to compare. Its "
        f"catalog lists {_catalog.height} registered prediction sets. Run the model notebooks "
        "against this registry first."
    )
    raise RuntimeError(msg)

all_labels_metrics = filter_active_model_rows(
    _raw_metrics.filter(pl.col("label").is_not_null()).join(_ok, on="prediction_hash", how="inner"),
    CASE_STUDY,
)
all_metrics = all_labels_metrics.filter(pl.col("label") == PRIMARY_LABEL)

# Say what was set aside and why, rather than letting the row count speak for itself.
if _rejected.height:
    # Two conditions were tested, so two counts are reported. A superseded row is complete and
    # was correct when it was written; reporting it as incomplete sends a reader looking for a
    # fit that never failed. Supersession is named first because it decides the row on its own:
    # a retired generation is not brought back by completing it. A legacy identity is named
    # separately for the opposite reason - it is the one cause of incompleteness that no re-run
    # of the same code will clear, because the row has to be refitted under the current scheme.
    _superseded = _rejected.filter(pl.col("prediction_hash").is_in(list(_retired)))
    _incomplete = _rejected.filter(~pl.col("prediction_hash").is_in(list(_retired)))
    _legacy = _incomplete.filter(pl.col("identity_status") == "legacy").height
    print(
        f"{_rejected.height} of {_catalog.height} registered prediction sets are inadmissible: "
        f"{_incomplete.height} not complete ({_legacy} of those carry a legacy identity), "
        f"{_superseded.height} superseded by a later generation of their own population"
    )
else:
    print(f"all {_catalog.height} registered prediction sets are admissible")

# The two populations 06_linear and 07_gbm published. This is not a fourth filter - admissibility
# above already holds - it is the check that the rows compared here are the ones those notebooks
# declared before they fitted anything. A family whose cohort has not run yet resolves to no
# population and is reported rather than silently omitted.
_population_members: dict[str, set[str]] = {}
for _family, _name in (("linear", LINEAR_POPULATION), ("gbm", GBM_POPULATION)):
    try:
        _population_members[_family] = set(OfficialPopulation.one(study, name=_name).members)
    except (ValueError, FileNotFoundError) as _exc:
        # Failing to resolve the declaration is not the same as there being nothing to check,
        # and answering both with a printed note turns a broken lineage into an unchecked
        # family: its rows still reach every comparison below, now with nothing saying they
        # were the ones declared. Tolerated only where the family has produced no rows at all.
        _produced = _catalog.filter(pl.col("family") == _family).height
        if _produced:
            msg = (
                f"{_family} has {_produced} registered prediction sets but its declared "
                f"population {_name} does not resolve ({_exc}). Comparing them would report a "
                "family no declaration covers. Republish the population, or name the one in "
                "force."
            )
            raise RuntimeError(msg) from _exc
        print(f"no current official population for {_family} ({_name}): {_exc}")

# A population is declared before anything is fitted; degeneracy is only visible afterwards.
# `load_all_metrics` drops any prediction set with a constant-prediction fold, because its
# pooled IC is computed over the surviving folds only and is not a model result. Those rows are
# declared members that correctly never reach a leaderboard, so the comparison allows for them
# and says how many rather than reporting a correct exclusion as a missing member.
_degenerate = degenerate_prediction_hashes(study.root)
_registered = set(_catalog.get_column("prediction_hash"))

for _family, _members in _population_members.items():
    _have = set(
        all_labels_metrics.filter(pl.col("family") == _family).get_column("prediction_hash")
    )
    _dropped = _members & _degenerate
    _missing, _extra = _members - _degenerate - _have, _have - _members
    # "Never produced" has to be decided from the registry, not from `_have`: that set is the
    # admissible rows, so a cohort that ran and whose every member is incomplete, legacy or
    # superseded reaches here empty too. Reading it as "not run yet" would skip the check below
    # in exactly the case the check exists for, and drop the whole family from the comparison
    # while reporting an absence rather than a rejection.
    if not _members & _registered:
        print(
            f"{_family}: none of the {len(_members)} declared members has been produced yet - "
            "the cohort has not run, so this family is absent from every comparison below"
        )
        continue
    if _missing or _extra:
        raise RuntimeError(
            f"{_family}: admissible rows do not match the declared population - "
            f"{len(_missing)} declared members absent, {len(_extra)} admissible rows undeclared"
        )
    _note = f" ({len(_dropped)} declared members degenerate and excluded)" if _dropped else ""
    print(
        f"{_family}: all {len(_members) - len(_dropped)} comparable declared members present{_note}"
    )

if all_metrics.height == 0:
    raise RuntimeError(f"No metrics found for {CASE_STUDY} / {PRIMARY_LABEL}")

families_present = sorted(all_metrics["family"].unique().to_list())
print(f"Pre-computed metrics: {all_metrics.height} entries across {len(families_present)} families")

for fam in families_present:
    sub = all_metrics.filter(pl.col("family") == fam)
    configs = sub["config_name"].n_unique()
    checkpoints = sub["checkpoint_value"].drop_nulls().n_unique()
    best_ic = sub["ic_mean"].max()
    best_ic_text = f"{best_ic:+.4f}" if best_ic is not None else "n/a"
    print(
        f"  {fam:20s}  {configs:3d} configs  {checkpoints:3d} checkpoints  best IC={best_ic_text}"
    )

# Coverage completeness check
EXPECTED_FAMILIES = {
    "linear",
    "gbm",
    "tabular_dl",
    "deep_learning",
    "latent_factors",
    "causal_dml",
} - excluded_families(CASE_STUDY)
missing = EXPECTED_FAMILIES - set(families_present)
if missing:
    print(
        f"\n[WARNING] COVERAGE: {len(families_present)} of {len(EXPECTED_FAMILIES)} model "
        f"families present. Missing: {', '.join(sorted(missing))}"
    )
    print("  Comparisons below describe only the families that are present.")
else:
    print(f"\n[OK] All {len(EXPECTED_FAMILIES)} expected model families present.")

# %% [markdown]
# ### How much of the period each prediction set was scored on
#
# A family's representative stands in for it in every comparison below, so it has
# to have been scored over the same period as the sets it is compared against. A
# run that failed partway still leaves rows in the registry, and its score is an
# average over the days it managed rather than the days it was asked for - which
# is frequently the higher number, because a shorter window is an easier window.
#
# `ic_n_days` counts the decision days behind a set's score. The restriction is
# applied within each family and label, because families legitimately differ: a
# sequence model cannot score the first observations of a fold, since no window
# ends there yet, so it covers fewer days than a model that scores every row.
# Comparing each family against its own peers is what keeps such a family in the
# comparison at all.
#
# The table below shows where each family stands before that restriction runs.

# %%
coverage_by_family = (
    all_metrics.group_by("family")
    .agg(
        pl.len().alias("sets"),
        pl.col("ic_n_days").min().alias("min_days"),
        pl.col("ic_n_days").max().alias("max_days"),
    )
    .sort("family")
)
print(f"Scored-day coverage per family for {PRIMARY_LABEL}:")
print(coverage_by_family)

# %% [markdown]
# `best_model_per_family_fast` applies the full-coverage restriction itself,
# within each family and label, and raises rather than dropping a group whose
# coverage cannot be established.

# %%
best_per_family = best_model_per_family_fast(all_metrics)

print("\nRepresentative model per family:")
print(best_per_family.select(["family", "config_name", "checkpoint_value", "ic_mean", "ic_std"]))

# %%
# Phase 2a: Load per-fold IC from registry (fast path - no prediction files needed)
fold_metrics = load_fold_metrics_from_registry(CASE_STUDY, label=PRIMARY_LABEL)
fold_metrics = filter_active_model_rows(fold_metrics, CASE_STUDY)
if fold_metrics.height > 0:
    print(f"Fold metrics from registry: {fold_metrics.height} entries")
else:
    print("No fold_metrics in registry - will compute from raw predictions")

# %% [markdown]
# ### Load the representative predictions on a thinned timestamp grid
#
# The diagnostics that follow read the prediction rows themselves rather than
# stored summaries, and on a minute panel that is millions of rows per family.
# They are all cross-sectional: each one scores one decision time across the
# symbols quoting at it, then looks at how those scores behave over time. Keeping
# every `SAMPLE_EVERY_N`-th decision time therefore preserves what each statistic
# measures - whole cross-sections, in order - while reducing what has to be held
# in memory.
#
# What it does change is precision. Every figure and statistic below rests on
# this thinned grid, not on all decision times, so their standard errors are
# wider than the full panel would give. The registered scores above are computed
# on the full grid and are unaffected.

# %%
SAMPLE_EVERY_N = 4  # keep every 4th decision time
# The registered artifact's names, and the names the analysis helpers read.
_RAW_TO_ANALYSIS = {"fold": "fold_id", "prediction": "y_score", "actual": "y_true"}
representative_preds = []

# The representative rows came from the catalog, so each already carries the identity of the
# prediction set it names. Re-deriving that with a hand-written join against registry.db - which
# is what stood here - reached past the interface to ask a question the row had already answered,
# and did it with no admissibility condition of its own. `Result.open` resolves the same identity
# through the study, so a hash that does not resolve fails here instead of silently contributing
# no rows.
for row in best_per_family.filter(pl.col("family") != "causal_dml").iter_rows(named=True):
    family, config = row["family"], row["config_name"]
    prediction = Result.open(study, row["prediction_hash"])
    df = prediction.load()
    # `Result.load` returns the registered artifact, whose columns are `fold`, `prediction` and
    # `actual`. Every helper this notebook feeds the frame to - `fold_performance_matrix`,
    # `select_best_checkpoint`, `cross_sectional_ic` - reads `fold_id`, `y_score` and `y_true`,
    # which is the vocabulary `case_studies/utils/backtest_loaders.py` normalises raw prediction
    # artifacts into (:278). This notebook used to reach the same rows through a loader that had
    # already done that rename; going through the study resolves the identity properly but hands
    # back the raw names, so do the rename here rather than teaching each call site a second
    # vocabulary. Conditional because a family that already registered normalised names must not
    # be renamed twice.
    df = df.rename({old: new for old, new in _RAW_TO_ANALYSIS.items() if old in df.columns})
    # Keep every Nth decision time, preserving the whole cross-section at each one it keeps. This
    # reduces only what the correlation and bucket displays below read; every registered score is
    # computed on the full grid and is unaffected.
    if SAMPLE_EVERY_N > 1 and "timestamp" in df.columns:
        keep_ts = df["timestamp"].unique().sort().gather_every(SAMPLE_EVERY_N)
        df = df.filter(pl.col("timestamp").is_in(keep_ts))
    df = df.with_columns(
        pl.lit(family).alias("family"),
        pl.lit(config).alias("config_name"),
        pl.lit(PRIMARY_LABEL).alias("label"),
        pl.lit(row.get("checkpoint_value")).alias("checkpoint_value"),
    )
    representative_preds.append(df)
    print(f"  {family}/{config}: {df.height:,} predictions (sampled 1/{SAMPLE_EVERY_N})")

if representative_preds:
    best_preds = pl.concat(representative_preds, how="diagonal_relaxed")
    print(f"\nTotal representative predictions: {best_preds.height:,}")
else:
    best_preds = pl.DataFrame()
    print("WARNING: No raw predictions could be loaded")

# %%
# Fold date ranges for timeline
if best_preds.height > 0:
    _date_dtype = best_preds[DATE_COL].dtype
    if _date_dtype == pl.String:
        _date_expr = pl.col(DATE_COL).str.to_datetime(strict=False).cast(pl.Date)
    else:
        _date_expr = pl.col(DATE_COL).cast(pl.Date)
    fold_ranges = (
        best_preds.filter(pl.col("fold_id").is_not_null())
        .with_columns(_date_expr)
        .group_by("fold_id")
        .agg(
            pl.col(DATE_COL).min().alias("val_start"),
            pl.col(DATE_COL).max().alias("val_end"),
        )
        .sort("fold_id")
    )

# %% [markdown]
# ### Figure 1: Cross-Validation Timeline

# %%
if best_preds.height > 0 and fold_ranges.height > 0:
    plot_cv_timeline(fold_ranges, n_splits, holdout_start)

# %% [markdown]
# With only 2 walk-forward folds, each covering 6 months of 15-minute
# bars, the validation evidence is inherently thin. Each fold contains
# roughly $114 \times 26 \times 126 \approx 374{,}000$ predictions
# (114 stocks × 26 bars/day × ~126 trading days), so per-fold sample
# size is large. But the temporal diversity is minimal - both folds
# fall within the 2020–2021 period, which was dominated by COVID
# recovery, meme-stock volatility, and an unprecedented retail trading
# surge. Whether patterns learned here generalize to more normal market
# conditions is an open question that 2 folds cannot answer.

# %% [markdown]
# ## 2. What Was Actually Run?
#
# Before comparing results, map what is comparable. Not every model family was
# trained on every label, and the modelling chapters produce different kinds of
# evidence: predictive forecasts, latent structure, and estimated causal effects.
# Each kind is ranked against its own kind, and the map below records which is
# which.

# %%
# Coverage map: family × label × evidence type
EVIDENCE_TYPE = {
    "linear": "predictive",
    "gbm": "predictive",
    "tabular_dl": "predictive",
    "deep_learning": "predictive",
    "latent_factors": "structural",
    "causal_dml": "causal",
}
FAMILY_CHAPTER = {
    "linear": "Ch11",
    "gbm": "Ch12",
    "tabular_dl": "Ch12",
    "deep_learning": "Ch13",
    "latent_factors": "Ch14",
    "causal_dml": "Ch15",
}

coverage = (
    all_labels_metrics.group_by(["family", "label"])
    .agg(
        pl.col("config_name").n_unique().alias("n_configs"),
        pl.col("ic_mean").max().alias("best_ic"),
    )
    .with_columns(
        chapter=pl.col("family").replace(FAMILY_CHAPTER),
        evidence=pl.col("family").replace(EVIDENCE_TYPE),
    )
    .sort(["family", "label"])
)

print("Coverage Map: Families × Labels")
print(coverage.select(["chapter", "family", "label", "evidence", "n_configs", "best_ic"]))

# %%
# Primary label coverage summary
primary_coverage = coverage.filter(pl.col("label") == PRIMARY_LABEL)
predictive_families = primary_coverage.filter(pl.col("evidence") == "predictive")[
    "family"
].to_list()
structural_families = primary_coverage.filter(pl.col("evidence") == "structural")[
    "family"
].to_list()
causal_families = primary_coverage.filter(pl.col("evidence") == "causal")["family"].to_list()
all_labels = sorted(coverage["label"].unique().to_list())

print(f"\nPrimary label ({PRIMARY_LABEL}):")
print(f"  Predictive families: {predictive_families}")
print(f"  Structural families: {structural_families or 'none'}")
print(f"  Causal families: {causal_families or 'none'}")
print(f"\nAll labels trained: {all_labels}")

# %% [markdown]
# The NASDAQ-100 microstructure case study has five model families on the
# primary label (`fwd_ret_15m`), spanning four of the five modeling
# chapters. Latent factor models (Ch14) were not trained - microstructure
# features at 15-minute frequency do not have the cross-sectional depth
# or temporal structure that factor models require. This is expected:
# latent-factor methods are designed for panels with rich cross-sectional
# variation in fundamentals, not for intraday microstructure snapshots.
#
# Four labels were explored: the primary `fwd_ret_15m`, a directional
# variant (`fwd_dir_15m`), a shorter horizon (`fwd_ret_5m`), and a
# longer horizon (`fwd_ret_60m`). Only linear and GBM were trained on
# alternate labels. All cross-family comparisons use the primary label.

# %% [markdown]
# ## 3. Headline Comparative View
#
# Before comparing model families, we establish a baseline. If the
# simplest possible model - linear regression on 66 microstructure
# features - produces zero or negative IC, the prediction problem
# is fundamentally too noisy at this frequency.

# %%
# Linear baseline
linear_metrics = all_metrics.filter(pl.col("family") == "linear")
if linear_metrics.height > 0:
    for name in ["ols", "ridge_a0.001", "ridge_a0.01", "ridge", "linear"]:
        baseline = linear_metrics.filter(pl.col("config_name") == name)
        if baseline.height > 0:
            ic = baseline["ic_mean"][0]
            std = baseline["ic_std"][0] if baseline["ic_std"][0] is not None else 0
            print(f"Linear baseline ({name}):")
            print(f"  IC mean:  {ic:+.4f}" if ic is not None else "  IC mean:  n/a")
            if std > 0:
                print(f"  IC std:   {std:.4f}")
                if ic is not None:
                    t_stat = ic / (std / np.sqrt(n_splits))
                    print(f"  t-stat:   {t_stat:.1f} (across {n_splits} folds)")
                else:
                    print("  t-stat:   n/a")
            break

# %%
# Full ranking (top 15)
print(f"\nFull ranking ({all_metrics.height} model × checkpoint variants):")
print(
    all_metrics.head(15).select(["family", "config_name", "checkpoint_value", "ic_mean", "ic_std"])
)

# %% [markdown]
# **How to read a rank correlation this small.** A cross-sectional rank
# correlation is the correlation between the predicted ordering of stocks at one
# decision time and their realised ordering. A value near zero means the model
# gets barely more pairwise comparisons right than chance would.
#
# The reason such a value is reported at all rather than dismissed is that the
# ordering is acted on repeatedly. Under the Fundamental Law of Active
# Management, an information ratio scales as the correlation times the square
# root of the number of independent bets, so a correlation that is negligible on
# one bar can accumulate over a year of bars. The condition in that sentence is
# where the difficulty lies: consecutive 15-minute decisions on the same names
# are not independent bets, so the square-root term overstates what is available,
# and the multiplication says nothing about whether the trades survive their
# costs.
#
# Read each family's interval alongside its point estimate. Where two intervals
# overlap, the evidence does not separate those families, whatever the ordering
# of the point estimates.
#
# With two folds, a statistic computed across fold-level scores has almost no
# degrees of freedom. The intervals here use the per-day score as the unit of
# analysis instead, which is why they are the reading to rely on.
#
# Causal DML is not in this ranking. It lives in the `causal_runs` table and
# reports an average treatment effect rather than a correlation.

# %% [markdown]
# ### How much signal does each family extract?
#
# The comparison uses each family's representative and looks at consistency
# across folds as well as the average. With two folds that is a weak check, but
# it catches the case that matters most: a representative whose score changes
# sign between folds has not established a stable relationship at all.

# %%
# Build fold × family IC matrix - prefer registry fast path
if fold_metrics.height > 0:
    # Fast path: use per-fold IC from registry (no raw predictions needed)
    best_keys = best_per_family.select(["family", "config_name", "checkpoint_value"])
    fold_ic_parts = []
    for row in best_keys.iter_rows(named=True):
        mask = (pl.col("family") == row["family"]) & (pl.col("config_name") == row["config_name"])
        cp = row.get("checkpoint_value")
        if cp is not None:
            mask = mask & (pl.col("checkpoint_value") == cp)
        else:
            mask = mask & pl.col("checkpoint_value").is_null()
        part = fold_metrics.filter(mask)
        if part.height > 0:
            model_label = f"{row['family']}/{row['config_name']}"
            ic_col = "ic" if "ic" in part.columns else "ic_mean"
            fold_ic_parts.append(
                part.select(
                    pl.lit(model_label).alias("model_label"),
                    pl.col("fold_id"),
                    pl.col(ic_col).alias("ic_mean"),
                )
            )
    fold_ic = pl.concat(fold_ic_parts) if fold_ic_parts else pl.DataFrame()
else:
    # Slow path: compute from raw predictions
    fold_ic = (
        fold_performance_matrix(best_preds, date_col=DATE_COL)
        if best_preds.height > 0
        else pl.DataFrame()
    )

# %% [markdown]
# ### Figure 2: Fold-by-Model Performance Heatmap

# %%
model_labels, fold_cols, matrix = plot_fold_heatmap(fold_ic)

# %%
# Summary statistics per family
if fold_ic.height > 0:
    family_stats = (
        fold_ic.group_by("model_label")
        .agg(
            pl.col("ic_mean").mean().alias("mean_ic"),
            pl.col("ic_mean").median().alias("median_ic"),
            pl.col("ic_mean").std().alias("std_ic"),
            pl.col("ic_mean").min().alias("worst_fold"),
            pl.col("ic_mean").max().alias("best_fold"),
            (pl.col("ic_mean") > 0).mean().alias("pct_positive"),
            pl.col("ic_mean").count().alias("n_folds"),
        )
        .sort("mean_ic", descending=True)
    )
    print("Family performance summary:")
    print(family_stats)

# %% [markdown]
# **How to read the heatmap.** Each row is a model family, each column a fold,
# and the shading is that family's representative's score on that fold. Three
# things are worth checking in it, in order.
#
# First, sign consistency across folds. A representative that is positive on one
# fold and negative on the other has not shown a stable relationship, whatever
# its average.
#
# Second, the spread within a row against the spread between rows. When the
# variation across folds is as large as the variation across families, the
# ranking between families is being read off noise.
#
# Third, which families are absent. A family with no tile was not trained for
# this case study rather than trained and found wanting, and the two are not the
# same conclusion. The coverage map above records which is which.
#
# With two folds, none of this supports a fold-level stability claim. It is a
# check for the obvious failure - a representative that does not hold its sign -
# rather than a measurement of consistency.

# %% [markdown]
# ## 4. Stability Over Time
#
# With only 2 walk-forward folds, the stability analysis is inherently
# limited. We cannot compute meaningful interquartile ranges or detect
# bimodal distributions. Instead, we focus on whether the two folds
# agree on the family ranking and whether either fold is negative.

# %% [markdown]
# ### Figure 3: Fold Performance Distribution by Model Family

# %%
plot_fold_boxplot(fold_ic)

# %% [markdown]
# With only 2 data points per family, the box plots reduce to line
# segments connecting the two fold ICs. Read against the locked
# registry, all three trained families (`gbm/leaves_7_mae`,
# `linear/ridge_a1000000.0`, `deep_learning/nlinear`) are positive in
# both folds at the primary `fwd_ret_15m` label. Per-fold magnitudes
# differ across families but the **HAC-based per-day CIs in §3 and
# §6** are the more reliable evidence - they pool over many days
# rather than two folds and produce overlapping CIs across all three
# families.
#
# **The 2-fold limitation is severe.** With 8 folds (as in the ETF
# case study), we could distinguish "reliable but modest" from "noisy
# but occasionally brilliant." With 2 folds, fold-level conclusions
# about stability are descriptive only. All downstream judgments rely
# on per-day HAC CIs (§3, §6), conformal calibration (§7), and the
# causal robustness check (§7).

# %% [markdown]
# ## 5. What Are the Models Learning?
#
# Beyond aggregate IC, we examine the *structure* of predictions. Two
# diagnostic questions matter:
#
# 1. **Monotonicity**: do higher predicted scores correspond to higher
#    realized 15-minute returns? A monotonic relationship confirms
#    ranking ability.
# 2. **Diversity**: do different model families produce similar or
#    different rankings? At 15-minute frequency, where signal is weak,
#    ensemble diversity is especially valuable.

# %%
# Compute prediction bucket monotonicity for best model per family
bucket_results = {}
for row in best_per_family.iter_rows(named=True):
    family = row["family"]
    config = row["config_name"]
    checkpoint = row.get("checkpoint_value")

    mask = (pl.col("family") == family) & (pl.col("config_name") == config)
    if checkpoint is not None:
        mask = mask & (pl.col("checkpoint_value") == checkpoint)

    model_preds = best_preds.filter(mask) if best_preds.height > 0 else pl.DataFrame()
    if model_preds.height == 0:
        continue

    buckets = prediction_bucket_monotonicity(model_preds, N_BUCKETS, DATE_COL)
    if buckets.height > 0:
        bucket_results[family] = buckets

# %% [markdown]
# ### Figure 4: Prediction Bucket Monotonicity

# %%
unconditional_mean = float(best_preds["y_true"].mean()) if best_preds.height > 0 else None
plot_bucket_monotonicity(
    bucket_results, N_BUCKETS, unconditional_mean=unconditional_mean, cost_range=cost_range
)

# %% [markdown]
# The monotonicity plot confirms genuine ranking ability, but the
# per-bar economic spreads are sub-bp and below the per-leg cost
# floor (1--5 bps). The bucket-spread bars in the figure show the
# highest-IC config from each trained family (`gbm/leaves_7_mae`,
# `linear/ridge_a1000000.0`, `deep_learning/nlinear`) at the primary
# label; magnitudes are reported by the helper rather than transcribed
# here so the prose does not drift from the figure.
#
# **The edge-to-cost ratios are all below 1.** The per-bar decile
# spread cannot survive a single round-trip at any reasonable cost
# assumption - round-trip cost of 2--10 bps swamps the per-bar
# spread. The annualized perspective offers some hope: spreads
# compound across 26 bars per day and ~252 days per year, and a
# selective strategy (trading only the most extreme signals in the
# most favorable regime) might achieve positive net returns. The
# binding requirement is that you cannot trade every bar - only bars
# where the signal materially exceeds the spread.

# %%
# Pairwise prediction correlations
corr_matrix, corr_labels = (
    prediction_correlation_matrix(best_preds, date_col=DATE_COL, entity_col=ENTITY_COL)
    if best_preds.height > 0
    else (np.array([]), [])
)

# %% [markdown]
# ### Figure 5: Prediction Correlation Across Models

# %%
plot_correlation_matrix(corr_matrix, corr_labels)

# %% [markdown]
# **How to read the correlation matrix.** Each cell is the correlation between
# two families' representative predictions over the same rows. It answers a
# question the scores cannot: whether two models that rank comparably well are
# ranking the same way.
#
# High correlation between families means they have found the same structure,
# and combining them adds little. Low correlation means they are reading
# different things from the same features, and a combination could be steadier
# than either alone even when neither is stronger.
#
# Low correlation is not by itself good news. Two models can disagree because
# each is fitting a different part of the noise, which looks identical in this
# matrix to two models capturing genuinely different structure. What separates
# the cases is whether both hold their ordering out of sample, which is what the
# fold view and the intervals address.
#
# Causal DML is absent here by construction: it produces an effect estimate, not
# a per-row prediction to correlate.

# %% [markdown]
# ### How much does additional model complexity help?
#
# For models that recorded checkpoints, the curve shows how the validation score
# evolves as training proceeds. A curve that rises and then falls locates the
# point past which the extra capacity is fitting noise. Row count is no
# protection here: many rows of a weak, heavily overlapping signal still let a
# flexible model memorise the noise.

# %%
# Learning curves from pre-computed metrics (fast path)
cp_data = all_metrics.filter(pl.col("checkpoint_value").is_not_null())
cp_families = (
    cp_data.group_by("family")
    .agg(pl.col("checkpoint_value").n_unique().alias("n_cp"))
    .filter(pl.col("n_cp") > 1)["family"]
    .to_list()
    if cp_data.height > 0
    else []
)

print(f"Families with checkpoint data: {cp_families}")

# %% [markdown]
# ### Figure 6: Learning Curves

# %%
plot_learning_curves(cp_data, cp_families)

# %% [markdown]
# The learning curves trace IC across training checkpoints for the
# families that emit them. From the locked registry, `gbm` highest IC at
# `fwd_ret_15m` is `leaves_7_mae` (small-leaf, MAE loss) - the
# regularization-against-noise pattern is the binding principle,
# not the capacity-for-interactions story. `deep_learning/nlinear` is an
# architecturally near-linear sequence model, not a deep recurrent
# architecture; its highest IC is a single-config result and does not
# come with a meaningful epoch-vs-IC progression for cross-architecture
# comparison.
#
# Both families' curves show monotone-then-plateau behavior rather than
# overfitting (no declining IC beyond the peak), suggesting the models
# extract genuine but weak patterns rather than memorizing noise.

# %% [markdown]
# ### Which Features Drive the Forecasts?
#
# Feature importance from a single model fit is anecdotal. With only
# 2 walk-forward folds, even "recurring" importance is a low bar.
# We examine which of the 66 microstructure features consistently
# drive predictions, looking for features that appear in the top 5
# in both folds.

# %%
# Try GBM booster-based importance first, fall back to feature-prediction correlation
gbm_importance = load_gbm_feature_importance(CASE_STUDY, label=PRIMARY_LABEL, top_n=TOP_N_FEATURES)

if gbm_importance is None:
    # Fallback: compute feature-prediction IC (correlation between each feature and y_score)
    print("No GBM booster files available. Computing feature-prediction correlation as fallback...")

    features_path = CASE_DIR / "features" / "financial.parquet"
    if features_path.exists() and best_preds.height > 0:
        features_df = pl.read_parquet(features_path)
        feat_cols = [c for c in features_df.columns if c not in [DATE_COL, ENTITY_COL]]

        # Join best linear model predictions with features
        linear_preds = best_preds.filter(pl.col("family") == "linear")
        if linear_preds.height > 0:
            # Align timestamp types (predictions=datetime[ms], features may be date)
            if features_df[DATE_COL].dtype == pl.Date:
                features_df = features_df.with_columns(pl.col(DATE_COL).cast(pl.Datetime("ms")))
            merged = linear_preds.join(features_df, on=[DATE_COL, ENTITY_COL], how="inner")

            # Compute correlation of each feature with y_score per fold
            importance_rows = []
            for fold in sorted(merged["fold_id"].unique().drop_nulls().to_list()):
                fold_data = merged.filter(pl.col("fold_id") == fold)
                for feat in feat_cols:
                    vals = fold_data[[feat, "y_score"]].drop_nulls()
                    if vals.height > 50:
                        from scipy.stats import spearmanr

                        corr, _ = spearmanr(vals[feat].to_numpy(), vals["y_score"].to_numpy())
                        importance_rows.append(
                            {
                                "config_name": "linear",
                                "fold_id": int(fold),
                                "feature": feat,
                                "importance": abs(float(corr)),
                            }
                        )

            if importance_rows:
                gbm_importance = pl.DataFrame(importance_rows)
                # Normalize per fold
                gbm_importance = gbm_importance.with_columns(
                    (
                        pl.col("importance")
                        / pl.col("importance").max().over(["config_name", "fold_id"])
                    ).alias("importance_norm")
                )
                # Filter to top features
                top_features = (
                    gbm_importance.group_by("feature")
                    .agg(pl.col("importance_norm").mean().alias("mean_imp"))
                    .sort("mean_imp", descending=True)
                    .head(TOP_N_FEATURES)["feature"]
                    .to_list()
                )
                gbm_importance = gbm_importance.filter(pl.col("feature").is_in(top_features))
                print(
                    f"Computed feature-prediction correlation for {len(top_features)} features across {merged['fold_id'].n_unique()} folds"
                )

if gbm_importance is not None and gbm_importance.height > 0:
    print(
        f"Feature importance: {gbm_importance['feature'].n_unique()} features × {gbm_importance['fold_id'].n_unique()} folds"
    )
else:
    print("Feature importance data not available.")

# %% [markdown]
# ### Figure 7: Feature Importance Stability Heatmap

# %%
plot_feature_importance_heatmap(gbm_importance, TOP_N_FEATURES)

# %% [markdown]
# The feature importance analysis reveals a surprising result: the
# only persistent features (top-5 in both folds) are **`is_first_30m`**
# and **`is_last_30m`** - time-of-day indicators, not the order flow
# or liquidity features we expected.
#
# This is actually economically meaningful. The first and last 30
# minutes of the trading day are well-known to exhibit different
# microstructure dynamics:
#
# - **Opening**: overnight information is incorporated, spreads are
#   wider, volume is concentrated, and mean-reversion patterns are
#   stronger as the opening auction resolves overnight imbalances
# - **Closing**: portfolio rebalancing flows, index tracking, and
#   MOC orders create predictable patterns in the cross-section
#
# The microstructure features (signed volume share, relative spread,
# microprice deviation) appear in the top 15 but are not persistent
# across both folds - their importance shifts with market conditions.
# This suggests the signal is diffuse: no single microstructure
# feature carries the bulk of the signal, but the time-of-day context conditions which
# features matter. The models may be implicitly learning "at the
# open, order flow imbalance matters; at the close, spread dynamics
# matter" - but with only 2 folds, we cannot confirm this.

# %% [markdown]
# ## 6. Heterogeneity: Labels, Horizons, and Regimes
#
# This section answers two questions: does the signal change across
# prediction horizons? And does model ranking depend on the intraday
# volatility regime? Both matter for strategy design at 15-minute
# frequency.

# %% [markdown]
# ### Multi-Label Comparison
#
# Three regression labels span the horizon: a shorter horizon
# (`fwd_ret_5m`), the primary `fwd_ret_15m`, and a longer horizon
# (`fwd_ret_60m`). The classification variant `fwd_dir_15m` is scored on
# different measures and is excluded from the forest below, which reports each
# family's representative and its interval per label. Tiles marked as having no
# run make coverage gaps visible rather than leaving them as absences.

# %%
multi_rows = []
for lbl in REGRESSION_LABELS:
    lbl_metrics = all_labels_metrics.filter(pl.col("label") == lbl)
    for fam in lbl_metrics["family"].unique().to_list():
        fam_data = lbl_metrics.filter(pl.col("family") == fam)
        rank1 = fam_data.sort("ic_mean_daily", descending=True, nulls_last=True).head(1)
        if rank1.height == 0:
            continue
        r = rank1.row(0, named=True)
        if r.get("ic_mean_daily") is None:
            continue
        multi_rows.append(
            {
                "label": lbl,
                "family": fam,
                "config_name": r["config_name"],
                "ic_mean_daily": r["ic_mean_daily"],
                "ic_ci_lo": r.get("ic_ci_lo"),
                "ic_ci_hi": r.get("ic_ci_hi"),
                "ic_t_hac": r.get("ic_t_hac"),
            }
        )

multi_label_df = pl.DataFrame(multi_rows)
multi_label_df

# %%
plot_label_horizon_forest(
    multi_label_df,
    families=["linear", "gbm", "tabular_dl", "deep_learning", "latent_factors", "causal_dml"],
    labels=REGRESSION_LABELS,
    label_display={
        "fwd_ret_5m": "fwd_ret_5m (intra-bar)",
        "fwd_ret_15m": "fwd_ret_15m (one-bar, primary)",
        "fwd_ret_60m": "fwd_ret_60m (four-bar)",
    },
    title="NASDAQ-100 microstructure - highest IC per family × horizon (HAC 95% CI)",
)

# %% [markdown]
# **How to read the horizon forest.** Each row is one family at one label
# horizon, the marker is its representative's score and the bar is that score's
# confidence interval. Three readings, in order of what they can support.
#
# A bar that crosses zero means the sign of the relationship is not established
# at that horizon. That is a stronger statement than a small point estimate, and
# it is the first thing to check on every row.
#
# Comparing rows *down* a family shows how the horizon changes what is
# detectable. A shorter horizon puts less time between the prediction and the
# outcome, so less unrelated movement accumulates in between, and the
# relationship is usually easier to detect. That says nothing about whether it is
# easier to *trade*: a shorter horizon means more rebalances and more crossings
# of the spread, which is exactly the tension the cost notebook resolves.
#
# Comparing rows *across* families at one horizon is the weakest of the three.
# Where the intervals overlap, the families are not separated, and the ordering
# of their markers should not be reported as a ranking.
#
# A missing tile means the family was not trained at that horizon, which is not
# the same as a family that was trained and scored poorly. The coverage map
# records which families ran; `causal_dml` is absent by construction, since it
# estimates an effect rather than an ordering.

# %% [markdown]
# ### Regime Conditioning
#
# At 15-minute frequency, the natural conditioning variable is
# intraday volatility: during volatile periods, cross-sectional
# dispersion increases and microstructure signals become more
# pronounced. We use rolling cross-sectional return dispersion
# (computed over 252 bars, approximately one trading day) as the
# regime indicator.

# %%
# Compute regime-conditional IC
regime_results = []

for row in best_per_family.iter_rows(named=True):
    family = row["family"]
    config = row["config_name"]
    checkpoint = row.get("checkpoint_value")

    mask = (pl.col("family") == family) & (pl.col("config_name") == config)
    if checkpoint is not None:
        mask = mask & (pl.col("checkpoint_value") == checkpoint)

    model_preds = best_preds.filter(mask) if best_preds.height > 0 else pl.DataFrame()
    if model_preds.height == 0:
        continue

    regime_ic = regime_conditional_ic(model_preds, date_col=DATE_COL)
    if regime_ic.height > 0:
        regime_ic = regime_ic.with_columns(pl.lit(family).alias("family"))
        regime_results.append(regime_ic)

regime_df = pl.concat(regime_results) if regime_results else pl.DataFrame()

# %% [markdown]
# ### Figure 8: Conditional Performance by Volatility Regime

# %%
plot_regime_bars(regime_df)

# %% [markdown]
# Regime sensitivity is critical for intraday microstructure strategies.
# During high-volatility periods (large cross-sectional dispersion),
# microstructure signals are typically stronger: spreads widen, order
# flow imbalances become more informative, and the cross-section
# separates more sharply. During low-volatility periods, all stocks
# move together and microstructure features provide less differentiation.
#
# If the high-vol IC is significantly higher than low-vol IC for all
# families, the strategy has a natural "when to trade" filter: reduce
# positions during calm markets when the signal-to-noise ratio is
# poorest. This regime conditioning could meaningfully improve the
# net edge after costs by avoiding trades where the expected return
# per bar is too low to cover the spread.

# %% [markdown]
# ## 7. Structural and Causal Evidence
#
# Not all model chapters produce comparable predictive scores. Ch14
# (latent factors) extracts structure; Ch15 (causal DML) estimates
# treatment effects. These require separate evidence blocks.

# %% [markdown]
# ### Latent Factors (Ch14)
#
# Latent factor models were **not trained** for the NASDAQ-100
# microstructure case study. At 15-minute frequency, the cross-section
# of 114 stocks lacks the fundamental heterogeneity that factor models
# require - these are all large-cap US tech stocks with highly
# correlated microstructure dynamics. PCA or CAE applied to intraday
# microstructure features would extract market-wide volatility modes,
# not tradeable cross-sectional factors.
#
# This contrasts with broader panels like US Firm Characteristics
# (2,483 stocks) and SP500 Equity+Options (638 entities), where
# latent factors can capture meaningful cross-sectional variation
# in fundamentals.

# %% [markdown]
# ### Causal DML (Ch15)
#
# Causal results live in a dedicated `causal_runs` table (separate
# from `prediction_metrics`). Causal DML estimates an **average
# treatment effect** of a microstructure treatment on next-bar
# returns, with HAC-robust standard errors and a refutation test.
# It does not produce a per-asset cross-sectional score, so it is
# reported as ATE/SE/p_HAC rather than IC.
#
# Treatment: `signed_vol_share` (signed volume share at the bar);
# confounders: `rel_spread_close`, `rv_5m`, `r1m`; embargo = 1 bar.

# %%
import sqlite3 as _sqlite3

with _sqlite3.connect(CASE_DIR / "run_log" / "registry.db") as _con:
    _cur = _con.cursor()
    _rows = _cur.execute(
        "SELECT label, dml_effect, dml_se_hac, p_value_hac, "
        "naive_effect, confounding_bias_pct, refutation_p, n_obs "
        "FROM causal_runs ORDER BY label"
    ).fetchall()
    _cols = [d[0] for d in _cur.description]

causal_df = pl.DataFrame(_rows, schema=_cols, orient="row") if _rows else pl.DataFrame()
print("Causal DML on signed_vol_share:")
print(causal_df)

# %% [markdown]
# **How to read the causal rows.** Each row is an estimated average treatment
# effect: how much the outcome moves per unit change in `signed_vol_share`, after
# adjusting for the confounders. Four columns decide what the row supports.
#
# The naive and adjusted effects together say what the adjustment did. If the
# adjusted effect is smaller, the confounders were carrying part of the raw
# association. If it is larger, they were masking it. Either direction is
# informative; a large gap in either direction is a reason to look at the
# nuisance models before trusting the result.
#
# The Newey-West p-value and the permutation p-value answer different questions
# and frequently disagree, as `12_causal_dml` explains in full. When they do,
# the parametric one is the one to distrust.
#
# **`refutation_p` is not yet a test this estimate can fail, and should not be read as a passing
# check.** The permutation refutation shuffles the treatment within blocks, and the block has to be
# at least as long as the autocorrelation the estimate rests on, or the shuffle destroys exactly
# the structure the placebo is supposed to preserve. The block length is currently taken from the
# label buffer rather than from the treatment's own window, and `signed_vol_share` declares no
# window, so the block is not derived from the quantity it has to bound. A refutation built that
# way passes by construction. `us_firm_characteristics` measured what that looks like on its own
# treatment: p = 1.0000 at z = -13.89, with the placebos fourteen standard deviations above the
# observed effect - a placebo distribution that cannot contain the truth is not evidence about it.
# Read the column as unresolved until the treatment declares its window.
#
# The effect is measured per unit of the treatment, and treatment units are not
# comparable across labels or case studies. An effect that looks microscopic in
# absolute terms may or may not matter once multiplied by the spread of
# `signed_vol_share` actually observed across stocks. The strategy notebooks
# carry out that multiplication and add the costs of acting on it.
#
# These rows are not comparable to the correlations above. They estimate the
# effect of intervening on one feature; a rank correlation measures how well an
# ordering of all features anticipates an ordering of returns.

# %% [markdown]
# ### Calibration: Are Prediction Intervals Honest?
#
# Point IC tells us whether the ranking is correct on average; it says
# nothing about whether the model's *uncertainty* is well calibrated.
# Inductive split-conformal prediction (Vovk et al., 2005; Lei et al.,
# 2018) gives a distribution-free check: using fold-0 absolute residuals
# as a calibration set, the symmetric quantile $\hat{q}_{1-\alpha}$
# defines an interval $[\hat{y} - \hat{q}, \hat{y} + \hat{q}]$ that
# should cover the true label at rate $1-\alpha$ on the remaining folds.
# Empirical coverage materially below the nominal level signals
# overconfident residual scaling - the model is more wrong, more often,
# than its training-time spread suggests. Width is reported as a
# fraction of the actuals' standard deviation so families with different
# return scales are comparable; smaller width at matched coverage means
# tighter, more useful intervals. See Ch12 §12.6 / `11_conformal_gbm`
# for the full conformal toolkit (CQR, ACI). What we report here is the
# minimal residual-calibration diagnostic on the highest-IC config per
# family for the primary label.

# %%
conformal_df = conformal_coverage_diagnostic(CASE_STUDY, label=PRIMARY_LABEL)
conformal_df

# %%
if conformal_df.height > 0:
    pivot = conformal_df.pivot(
        on="nominal_level",
        index=["family", "config_name"],
        values=["empirical_coverage", "mean_interval_width_frac_std"],
    )
    print("Empirical coverage and width (× std of returns) at 80/90/95% nominal:")
    print(pivot)

# %% [markdown]
# **How to read the coverage table.** Two columns matter, and they fail
# independently.
#
# Empirical coverage against nominal is the calibration check. An interval
# advertised at a given confidence level should contain the realised value about
# that often on the later folds. Coverage materially below nominal means the model is
# confident more often than it is right, and any position size derived from its
# interval is too large. Coverage materially above nominal means the intervals
# are wider than they need to be, which is safe but wasteful.
#
# Width per standard deviation says what that coverage cost. An interval that
# reaches nominal coverage only by spanning several standard deviations of the
# return distribution is honest and nearly useless for sizing. Read the two
# together: a model is usable for interval-aware sizing only when it holds its
# coverage at a width that still distinguishes one prediction from another.
#
# A family can rank well and fail this badly. Ranking depends only on the order
# of the predictions; coverage depends on the scale of the residuals carrying
# from the calibration window to a later one. When they diverge, the model needs
# recalibration before any interval-aware sizing, even though its ordering is
# unaffected.

# %% [markdown]
# ## 8. Pre-Backtest Judgment and Handoff
#
# The synthesis below collects, per family, the evidence a backtest depends on:
# the representative's score with its interval, its behaviour across folds, and
# whether its intervals are calibrated. Two folds is little temporal evidence, so
# the summary is a description of what was measured rather than a
# recommendation.

# %%
synthesis_rows = []

for row in best_per_family.iter_rows(named=True):
    family = row["family"]
    config = row["config_name"]
    ic_mean = row["ic_mean"]
    ic_std_val = row.get("ic_std") or 0

    label_key = f"{family}/{config}"
    fam_folds = (
        fold_ic.filter(pl.col("model_label") == label_key) if fold_ic.height > 0 else pl.DataFrame()
    )

    if fam_folds.height > 0:
        fold_ics = fam_folds["ic_mean"].to_numpy()
        median_ic = float(np.median(fold_ics))
        std_ic = float(np.std(fold_ics))
        pct_pos = float((fold_ics > 0).mean())
        worst = float(np.min(fold_ics))
    else:
        median_ic, std_ic, pct_pos, worst = (
            ic_mean,
            ic_std_val,
            (1.0 if ic_mean > 0 else 0.0),
            ic_mean,
        )

    spread = 0.0
    if family in bucket_results:
        b = bucket_results[family]
        if b.height >= 2:
            spread = (
                b.filter(pl.col("bucket") == N_BUCKETS)["mean_return"][0]
                - b.filter(pl.col("bucket") == 1)["mean_return"][0]
            )

    if median_ic > 0 and pct_pos > 0.6 and spread > 0:
        ci_tier = "excludes_zero_or_positive_pct>0.6"
    elif ic_mean > 0:
        ci_tier = "positive_mean_only"
    else:
        ci_tier = "below_credibility"

    synthesis_rows.append(
        {
            "family": family,
            "config": config,
            "ic_mean": round(ic_mean, 4),
            "ic_median": round(median_ic, 4),
            "ic_std": round(std_ic, 4),
            "pct_positive": round(pct_pos, 2),
            "worst_fold": round(worst, 4),
            "spread_bps": round(spread * 10000, 1),
            "ci_tier": ci_tier,
        }
    )

synthesis = pl.DataFrame(synthesis_rows).sort("ic_mean", descending=True)
print("Synthesis Table:")
print(synthesis)

# %% [markdown]
# ### How to read the synthesis table
#
# One row per family, collecting what the sections above measured separately:
# the representative's score with its interval, its per-fold behaviour, its
# decile bucket spread, and its interval calibration. Read across a row rather
# than down a column, because a family is only usable when several of these hold
# at once, and they fail independently.
#
# Where intervals overlap between rows, those families are not separated by this
# evidence, and the ordering of their point estimates should not be reported as a
# ranking. With two folds, the per-fold columns are weak evidence about stability
# and are there to expose a sign change rather than to measure consistency.
#
# The calibration columns are the ones most likely to disqualify a family that
# looks fine on score alone. A model whose intervals do not hold their nominal
# coverage can still be traded on its ordering, but not sized from its intervals
# without recalibration first.
#
# Causal rows are reported separately and are never inserted into this table.
# They estimate the effect of a feature rather than forecast a return.
#
# ### The decay question this table cannot answer
#
# Every score here assumes the prediction made at one decision time is acted on
# at the next. Between those two moments sit signal computation, portfolio
# construction and order routing. If the relationship weakens over that interval,
# the achievable version is smaller than the measured one, and at this horizon
# there is little room for it to shrink before trading costs exceed it. Nothing
# in a validation score measures that gap; a backtest that fills at the intended
# time and one that fills a minute later do.
#
# ### Forecast Representation
#
# For backtesting, predictions are used as:
#
# - **Rank-based selection**: sort by score, take the top and bottom names at
#   each decision time in equal measure, so the position is on the ordering
#   rather than on the market's direction.
# - **Regime filter**: trade only when the cross-section is dispersed enough for
#   the ordering to be worth acting on. When predicted differences between names
#   are smaller than the cost of switching between them, trading on the ordering
#   loses money however correct it is.
# - **Ensemble**: averaging families that disagree reduces the variance of the
#   combined ordering. It adds nothing to the ordering's strength, and it is
#   worth most as protection against one family being miscalibrated rather than
#   as a source of edge.
#
# ### What This Analysis Does Not Tell Us
#
# - **Execution feasibility**: 15-minute rebalancing across 114
#   stocks requires institutional-grade execution infrastructure.
#   The market impact of simultaneously trading the top and bottom
#   deciles may exceed the predicted edge.
# - **Signal decay**: we assumed next-bar execution. Any delay
#   degrades performance geometrically.
# - **Capacity**: even NASDAQ-100 stocks have limited intraday
#   liquidity at the bar level. A \$10M strategy may face meaningful
#   market impact; a \$100M strategy almost certainly would.
# - **Survivorship bias**: the 114-stock universe was selected based
#   on NASDAQ-100 membership, which is backward-looking.
# - **2-fold limitation**: the strongest caveat. 2020--2021 was an
#   extraordinary period (COVID, retail trading boom, meme stocks).
#   Whether these patterns persist in normal markets is unknown.
#
# **Next**: `14_backtest.py` for strategy simulation,
# `15_portfolio_management.py` for position sizing, and
# `18_strategy_analysis.py` for end-to-end results.

# %% [markdown] tags=["results"]
# ### What the comparison found on this run
#
# The representative of each family at full coverage, on the primary label, with
# its HAC interval; the same measure across the label horizons; and the causal
# rows, which are effects rather than correlations and are not comparable to the
# rows above.

# %%
_ci_cols = [c for c in ("ic_ci_lo", "ic_ci_hi") if c in best_per_family.columns]
print(f"Representative per family on {PRIMARY_LABEL} (full coverage only):")
print(best_per_family.select(["family", "config_name", "checkpoint_value", "ic_mean", *_ci_cols]))

# %%
if "ic_n_days" in all_labels_metrics.columns:
    _horizon = (
        all_labels_metrics.filter(pl.col("family").is_in(["linear", "gbm", "deep_learning"]))
        .group_by(["label", "family"])
        .agg(pl.col("ic_mean").max().alias("highest_ic"), pl.len().alias("sets"))
        .sort(["family", "label"])
    )
    print("\nHighest registered score by label and family:")
    print(_horizon)

# %%
if causal_df.height > 0:
    print("\nCausal rows (average treatment effects, not correlations):")
    print(causal_df)

# %% [markdown]
# ## Key Takeaways
#
# 1. **Compare only what was measured over the same period.** A prediction set
#    scored on fewer decision days than its neighbours is not a weaker version of
#    the same number, it is a different number. Restricting representatives to
#    full coverage before any comparison is what makes the ranking mean
#    something, and it is the check most likely to change which configuration
#    appears in front.
#
# 2. **A family's representative carries its checkpoint.** Two checkpoints of one
#    configuration are two models. Collapsing them into a single row by
#    configuration name doubles the folds behind a statistic and reports an
#    average across two different fits.
#
# 3. **Correlation strength and interval width answer different questions.** A
#    point estimate says which ordering was better on this sample; the interval
#    says how much of that ordering the sample can actually support. On two
#    walk-forward folds the intervals are wide enough that overlapping ones
#    should be read as not separated, not as a close ranking.
#
# 4. **Conformal coverage is a property of transport, not of accuracy.** A model
#    can rank well and still produce intervals that do not hold their nominal
#    coverage out of sample, because the residual scale it learned in training
#    does not carry to a later window. The two diagnostics are independent and a
#    model has to pass the one you intend to rely on.
#
# 5. **Causal effects are not predictive scores.** The rows from `causal_runs`
#    estimate what a change in one feature does to the outcome. They share no
#    scale with a rank correlation, they answer a different question, and they
#    never enter a predictive ranking or count toward family coverage.
#
# **Known limitations**: Two walk-forward folds over 2020-2021 is a small number
# of independent windows, and that period included unusual market conditions, so
# fold-level stability statements here are provisional. The row-level diagnostics
# read a thinned decision-time grid, which widens their standard errors relative
# to the registered scores. The universe is point-in-time NASDAQ-100 membership,
# which is a large-cap technology cross-section rather than a broad one.
#
# **Next**: `14_backtest.py` applies these predictions to simulated
# intraday trading.
