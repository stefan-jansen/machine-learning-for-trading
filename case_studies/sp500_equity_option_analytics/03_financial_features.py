# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     kernelspec:
#       display_name: Python 3 (ipykernel)
#       language: python
#       name: python3
# ---

# %% [markdown]
# # S&P 500 Equity + Option Analytics: Feature Engineering
#
# The strategy reads a signal off a name's listed options and holds the share, so the matrix
# built here has to carry both sides of that trade: what the option market charges for a name's
# coming month, and what the share has actually been doing. This notebook builds those families,
# states the window and the delay each one carries, shows that withholding later dates leaves
# every value unchanged, and writes the matrix stage 04 and stage 05 read.
#
# ## Learning objectives
#
# - Build implied-volatility level, dynamics, skew and variance-premium families from a daily
#   surface summary, beside the equity momentum they have to earn their place against
# - Lag an input that is published late, and say which of the two frames the lag is counted in
# - State each family's lookback and information lag before writing the code that computes it,
#   and assert the warmup that follows from it
# - Show that withholding later dates leaves every feature value unchanged, which is what
#   separates a trailing statistic from one fitted over the whole sample
# - Read a matrix before modelling it: scale, cross-sectional spread, redundancy and decay
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Sections 8.1-8.6. Reads the daily share bars and the daily option-surface summary
# through `load_sp500_daily_bars()` and `load_sp500_options_surface()`, whose coverage
# [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) establishes, and `config/setup.yaml`,
# which declares the register, every window, the ranked columns and the holdout boundary. Writes
# `features/financial.parquet` with a `.digest.json` sidecar, read by
# [`04_model_based_features`](04_model_based_features.ipynb), which adds GARCH and regime features
# on top of it, and by [`05_evaluation`](05_evaluation.ipynb), which tests fold by fold whether
# any of it predicts. No screen for predictive content runs here: `05_evaluation` owns it and runs
# it fold-aware.

# %%
"""S&P 500 Equity + Option Analytics: Feature Engineering."""

import warnings
from datetime import date

import numpy as np
import polars as pl
import yaml

from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.feature_engineering import (
    assert_values_agree,
    assign_families,
    families_from_config,
    family_coverage,
    plot_coverage_through_time,
    plot_cross_sectional_dispersion,
    plot_feature_distributions,
    plot_persistence,
    plot_redundancy_clusters,
    plot_timing_contract,
    register_frame,
    warmup_audit,
)
from data import load_sp500_daily_bars, load_sp500_options_surface
from utils.paths import display_path, get_case_study_dir

warnings.filterwarnings("ignore")

CASE_STUDY_ID = "sp500_equity_option_analytics"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
FEATURES_DIR = CASE_DIR / "features"

# %% [markdown]
# Both parameters bound the sample and both are read below. The share bars and the surface
# summary are licensed extracts covering these five years, so the defaults span the data.
# Trimming either end shortens a run at the cost of a shorter history for the trailing windows
# in Section C, and every cross-sectional rank is taken within a date across whatever names are
# quoted on it, so a trimmed run computes a different quantity rather than a smaller one.

# %% tags=["parameters"]
START_DATE = "2017-01-01"
END_DATE = "2021-12-31"

# %% [markdown]
# ## Configuration
#
# The register, every window, the ranked-column mapping, the surface selection and the holdout
# boundary are declared in `config/setup.yaml` and bound here. A window retyped into a cell is a
# second source of truth for a decision the register, the warmup assertion and the timing figure
# all have to agree on, and the two copies drift apart the first time either is edited.
#
# The decision cadence is what the persistence figure is measured against: a feature has to hold
# its ordering for at least one rebalance to be tradable at that cadence, and this strategy
# rebalances weekly on the primary label's horizon.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
FEATURES = setup["features"]
FAMILIES = families_from_config(setup)
W = FEATURES["windows"]
RANKED = FEATURES["ranked"]
SURFACE = FEATURES["surface"]
HOLDOUT_START = date.fromisoformat(setup["evaluation"]["holdout_start"])
IV_LAG = int(setup["decision"]["iv_feature_lag"].split("_")[0])

# The panel key, the entity every trailing window is bounded by, and the partition every
# cross-sectional statistic is taken over.
PANEL_KEY = ["symbol", "timestamp"]
ENTITY = "symbol"
WITHIN_DATE = "timestamp"

print(
    f"{len(FAMILIES)} declared families, {len(RANKED)} ranked columns, IV lagged {IV_LAG} session"
)
print(f"Holdout starts {HOLDOUT_START}; Section D rebuilds the matrix without it")
print(f"Surface buckets {SURFACE['dte_buckets']} at deltas {SURFACE['delta_targets']}")

# %% [markdown]
# ## A. What the thesis says should carry information
#
# The hypothesis is cross-sectional and it is about a disagreement between two markets. The
# option market prices a distribution of outcomes for a name over the coming month; the share
# price says what the equity market pays for it today. The claim is that names whose options are
# priced richly against what their shares go on to do can be ranked against each other, and that
# the ranking pays over the following week.
#
# The **carrier** is the level of implied volatility and what it is priced against. Around it sit
# the shape of the surface, how the level moves, and the spread between what is implied and what
# the share has realized, which is the quantity the hypothesis names directly.
#
# The **conditioning** is realized volatility, which is a state variable rather than a signal:
# it describes the dispersion a ranking has to be earned against and is not expected to rank
# names on its own. Equity momentum is in the matrix for a different reason. It is the signal an
# option-derived feature would have to replace, so carrying it is what makes a later comparison
# mean anything.
#
# The **frame** is the single decision date for every cross-sectional statistic and the security
# for every trailing one. A level of implied volatility is not comparable between a utility and a
# semiconductor, so eight of the columns are carried again as their percentile within the date.
#
# The register is declared in `config/setup.yaml`, one row per family, and it is split by
# **observability**: everything read off the option surface carries a one-session lag and
# everything read off the share price does not, for the reason Section B gives.

# %%
register_frame(FAMILIES).select(
    ["family", "role", "inputs", "lookback (bars)", "lag (bars)", "frame"]
)

# %% [markdown]
# ## B. Inputs and their observability
#
# Each row is one security and one session. Two frames arrive and they are not observable on the
# same schedule.
#
# The **share bars** are split- and dividend-adjusted daily OHLC. A close is final at the close,
# so a statistic of the closes through session $t$ is knowable at $t$ and carries no lag.
#
# The **surface summary** reduces the option chain to one row per name and session, selecting the
# contract closest to each delta target inside each fixed maturity bucket. Those buckets and
# targets are declared in `config/setup.yaml` and printed above; the selection itself runs in
# `data/equities/market/sp500/materialize_options.py`, upstream of this notebook.
#
# **The surface is not knowable at the session it is dated.** End-of-day implied volatility is
# solved and published after the close, so every surface column is shifted by the session
# `setup.yaml::decision.iv_feature_lag` declares before anything is computed from it. The
# dynamics in C.1 are then built on the lagged series rather than lagged after the fact, so a
# z-score never mixes a lagged level with an unlagged history.
#
# **The lag is counted in the surface's own rows, and that is a weaker claim than a session.** A
# name is not quoted on every session, so where a name's previous quoted row is older than the
# previous session the shift reaches further back than one session and hands the model a stale
# level under a fresh name. The forward fill below has the same shape: it carries a level up to
# the number of sessions `features.windows.iv_forward_fill` declares, which is a deliberate
# tolerance for a thin quote and not a claim that the level is current. Both are conservative in
# the direction that matters, because both can only ever use information older than the decision.

# %%
daily = load_sp500_daily_bars(start_date=START_DATE, end_date=END_DATE).sort([ENTITY, "timestamp"])
surface_raw = load_sp500_options_surface(start_date=START_DATE, end_date=END_DATE).sort(
    [ENTITY, "timestamp"]
)
SURFACE_COLS = [c for c in surface_raw.columns if c not in PANEL_KEY]

print(f"{daily.height:,} name-sessions of share bars, {daily[ENTITY].n_unique()} tickers")
print(f"{surface_raw.height:,} surface rows carrying {len(SURFACE_COLS)} columns")
print(f"{daily['timestamp'].min()} to {daily['timestamp'].max()}")


# %%
def lag_surface(surface: pl.DataFrame) -> pl.DataFrame:
    """Shift every surface column by the declared lag, then carry it over a thin quote."""
    lagged = surface.sort([ENTITY, "timestamp"]).with_columns(
        pl.col(c).shift(IV_LAG).over(ENTITY).alias(c) for c in SURFACE_COLS
    )
    return lagged.with_columns(
        pl.col(c).forward_fill(limit=W["iv_forward_fill"]).over(ENTITY).alias(c)
        for c in SURFACE_COLS
    )


# %% [markdown]
# ## C. Feature construction, one subsection per family
#
# ### C.1 Implied volatility level, dynamics, skew and term structure
#
# The level columns arrive from the surface summary and need no construction. What is built here
# is how the level sits against its own recent history: the session-over-session change, the
# change over the two momentum windows, the rolling z-scores, and where the level stands inside
# its own trailing range. Every one of them is a trailing statistic of one security's own series,
# and every window comes from `features.windows`.
#
# A z-score divides by a trailing standard deviation, which approaches zero for a name whose
# implied volatility barely moves. The denominator is floored so that a near-constant series
# returns a bounded number rather than an unbounded one, and the floor is a fixed constant rather
# than a quantity fitted to the sample.

# %%
ZERO_FLOOR = 0.001


def _zscore(column: str, window: int) -> pl.Expr:
    """Trailing z-score of *column* over *window* sessions within one security."""
    mean = pl.col(column).rolling_mean(window).over(ENTITY)
    std = pl.col(column).rolling_std(window).over(ENTITY).clip(lower_bound=ZERO_FLOOR)
    return (pl.col(column) - mean) / std


def surface_dynamics(surface: pl.DataFrame) -> pl.DataFrame:
    """Changes, momentum, z-scores and the trailing percentile of the lagged surface."""
    pct_window = W["iv_percentile"]
    low = pl.col("iv_30_atm").rolling_min(pct_window).over(ENTITY)
    high = pl.col("iv_30_atm").rolling_max(pct_window).over(ENTITY)
    return surface.with_columns(
        *[
            (pl.col(c) - pl.col(c).shift(1).over(ENTITY)).alias(f"d_{c}")
            for c in ("iv_30_atm", "skew_rr_30_25d", "term_ratio_atm")
        ],
        *[
            (pl.col("iv_30_atm") - pl.col("iv_30_atm").shift(w).over(ENTITY)).alias(f"iv_mom_{w}d")
            for w in W["iv_momentum"]
        ],
        *[_zscore("iv_30_atm", w).alias(f"iv_30_atm_z_{w}") for w in W["iv_zscore"]],
        _zscore("skew_rr_30_25d", W["skew_zscore"]).alias(f"skew_rr_z_{W['skew_zscore']}"),
        _zscore("term_ratio_atm", W["term_zscore"]).alias(f"term_ratio_z_{W['term_zscore']}"),
        ((pl.col("iv_30_atm") - low) / (high - low).clip(lower_bound=ZERO_FLOOR)).alias(
            f"iv_30_atm_pct_{pct_window}"
        ),
    )


# %% [markdown]
# ### C.2 Realized volatility
#
# What the share actually did, on four estimators that disagree in ways worth carrying. The
# close-to-close standard deviation is the plain one and is computed at two windows. Garman-Klass
# reads the whole daily range, so it sees an intraday move that opened and closed in the same
# place; its per-session term can go negative on a bar whose open and close straddle the range,
# so the average is floored at zero before the square root rather than after it. The volatility
# of volatility is the dispersion of the short estimator itself, and the realized skew is the
# third moment of standardized returns, which says whether the dispersion came from moves in one
# direction.

# %%
ANNUALIZE = setup["evaluation"]["periods_per_year"] ** 0.5
GK_COEFFICIENT = 2 * np.log(2) - 1


def realized_volatility(bars: pl.DataFrame) -> pl.DataFrame:
    """Close-to-close, range-based and higher-moment volatility from the share bars."""
    short, long_ = W["realized_vol"]
    gk_window, vv_window, skew_window = W["garman_klass"], W["vol_of_vol"], W["realized_skew"]
    df = bars.with_columns(pl.col("close").pct_change().over(ENTITY).alias("_ret")).with_columns(
        (
            0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
            - GK_COEFFICIENT * (pl.col("close") / pl.col("open")).log().pow(2)
        ).alias("_gk_session")
    )
    df = df.with_columns(
        *[
            (pl.col("_ret").rolling_std(w).over(ENTITY) * ANNUALIZE).alias(f"rv_{w}")
            for w in (short, long_)
        ],
        (
            pl.col("_gk_session").rolling_mean(gk_window).over(ENTITY).clip(lower_bound=0.0)
            * setup["evaluation"]["periods_per_year"]
        )
        .sqrt()
        .alias(f"gk_vol_{gk_window}"),
    )
    standardized = pl.col("_ret") / pl.col("_ret").rolling_std(skew_window).over(ENTITY).clip(
        lower_bound=ZERO_FLOOR / 10
    )
    return df.with_columns(
        pl.col(f"rv_{short}").rolling_std(vv_window).over(ENTITY).alias(f"vol_of_vol_{vv_window}"),
        standardized.pow(3)
        .rolling_mean(skew_window)
        .over(ENTITY)
        .alias(f"realized_skew_{skew_window}"),
    ).select(
        [
            *PANEL_KEY,
            f"rv_{short}",
            f"rv_{long_}",
            f"gk_vol_{gk_window}",
            f"vol_of_vol_{vv_window}",
            f"realized_skew_{skew_window}",
        ]
    )


# %% [markdown]
# ### C.3 Equity momentum
#
# The simple return at five horizons, its skip-month form, and its volatility-scaled twin. Skip-
# month momentum runs from the start window to the recent window and divides prices rather than
# subtracting returns, because returns compound and the difference of two window returns is not
# the return over the gap between them (Jegadeesh and Titman, 1993). The risk-adjusted form
# divides the medium-horizon return by the annualized volatility realized over the same span,
# with the denominator floored at the level `features.windows.risk_adjusted_vol_floor` declares
# so that a name that barely moved does not return an unbounded ratio.

# %%
PRICE_FLOOR = 1e-8


def equity_momentum(bars: pl.DataFrame) -> pl.DataFrame:
    """Multi-horizon returns, skip-month momentum and its volatility-scaled form."""
    ra_window = W["risk_adjusted"]
    df = bars.select([*PANEL_KEY, "close"]).with_columns(
        pl.col("close").pct_change().over(ENTITY).alias("_ret")
    )
    df = df.with_columns(
        *[
            (
                pl.col("close")
                / pl.col("close").shift(w).over(ENTITY).clip(lower_bound=PRICE_FLOOR)
                - 1
            ).alias(f"mom_{w}d")
            for w in W["momentum"]
        ],
        (
            pl.col("close").shift(W["skip_recent"]).over(ENTITY)
            / pl.col("close").shift(W["skip_start"]).over(ENTITY).clip(lower_bound=PRICE_FLOOR)
            - 1
        ).alias("mom_skip_recent"),
        (pl.col("_ret").rolling_std(ra_window).over(ENTITY) * ANNUALIZE).alias("_rv_ra"),
    )
    return df.with_columns(
        (
            pl.col(f"mom_{ra_window}d")
            / pl.col("_rv_ra").clip(lower_bound=W["risk_adjusted_vol_floor"])
        ).alias(f"mom_risk_adj_{ra_window}")
    ).select(
        [
            *PANEL_KEY,
            *[f"mom_{w}d" for w in W["momentum"]],
            "mom_skip_recent",
            f"mom_risk_adj_{ra_window}",
        ]
    )


# %% [markdown]
# ### C.4 The variance risk premium
#
# The quantity the hypothesis names: what the option market charges for the coming month against
# what the share has realized over the past one. It is a difference of two volatilities in the
# same units, so it needs no scaling, and its trailing z-score says whether today's spread is
# unusual for this name rather than wide in absolute terms.
#
# It is the one family that reads both frames, which is why it is built after the join rather
# than inside either input's own subsection.
#
# ### C.5 Cross-sectional position
#
# A long-short book acts on relative standing, so the eight columns
# `features.ranked` names are carried again as their percentile within the decision date. The
# percentile is taken over the names quoted on that date, which is the cross-section a decision
# is taken over, and over nothing else: a rank pooled across dates would mix a cross-sectional
# claim with a time-series one.


# %%
def variance_premium(df: pl.DataFrame) -> pl.DataFrame:
    """Implied minus realized volatility, and how unusual that spread is for this name."""
    short = W["realized_vol"][0]
    with_spread = df.with_columns(
        (pl.col("iv_30_atm") - pl.col(f"rv_{short}")).alias("ivrv_spread")
    )
    window = W["vrp_zscore"]
    return with_spread.sort([ENTITY, "timestamp"]).with_columns(
        _zscore("ivrv_spread", window).alias(f"vrp_z_{window}")
    )


def cross_sectional_position(df: pl.DataFrame) -> pl.DataFrame:
    """Percentile within the decision date for each column the register ranks."""
    return df.with_columns(
        (
            pl.col(source).rank().over(WITHIN_DATE) / pl.col(source).count().over(WITHIN_DATE) * 100
        ).alias(name)
        for source, name in RANKED.items()
    )


# %% [markdown]
# The five subsections compose into one function, which is what lets D.3 re-run the whole
# construction on a shorter panel and compare. The surface is lagged before anything reads it,
# and the share-derived families are joined onto it, so a name-session with no quoted surface
# still carries its momentum and its realized volatility.


# %%
CARRIERS = ["iv_30_atm", f"rv_{W['realized_vol'][0]}"]


def assemble(bars: pl.DataFrame, surface: pl.DataFrame) -> pl.DataFrame:
    """Every trailing and contemporaneous family, on the surface's own rows."""
    lagged = surface_dynamics(lag_surface(surface))
    return (
        lagged.join(realized_volatility(bars), on=PANEL_KEY, how="left")
        .join(equity_momentum(bars), on=PANEL_KEY, how="left")
        .pipe(variance_premium)
    )


def build_features(bars: pl.DataFrame, surface: pl.DataFrame) -> pl.DataFrame:
    """The whole matrix, from the two input panels, in dependency order."""
    return (
        assemble(bars, surface)
        # The null policy runs here, after every trailing window has been taken and
        # before anything ranks. Both halves of that order are load-bearing. A trailing
        # window computed on the surviving rows would count survivors rather than
        # sessions, so the window would silently span whatever was removed. A percentile
        # taken before the drop would rank a row against names the matrix does not
        # carry, so the ordering a model reads would not be the ordering it can act on.
        .drop_nulls(subset=CARRIERS)
        .pipe(cross_sectional_position)
        .sort(PANEL_KEY)
    )


built = build_features(daily, surface_raw)
EXCLUDED = {*PANEL_KEY, "_ret", "_gk_session", "_rv_ra", "close"}
feature_cols = sorted(c for c in built.columns if c not in EXCLUDED)
assignment = assign_families(feature_cols, FAMILIES)
print(f"{built.height:,} rows carrying {len(feature_cols)} features in {len(FAMILIES)} families")

# %% [markdown]
# ## D. The timing contract
#
# ### D.1 What each construction reads
#
# Four kinds of operation appear above. A **shift** reads one earlier row of the same security's
# series, and it is what the surface lag is. A **rolling** window - every z-score, every realized
# volatility, every trailing range and every return - ends at its own row and reads a fixed number
# of that security's earlier rows. A **contemporaneous** difference - the variance premium - reads
# two columns of the same row. A **cross-sectional** statistic - the eight percentiles - is taken
# with `.over("timestamp")`, so it reads every name quoted on that date and nothing dated before
# or after it.
#
# None of the four is fitted: no bound, scaler or encoder here has parameters estimated once over
# the sample and applied to every row. The two floors in Section C are fixed constants, which is
# a different thing. D.2 checks the windows; D.3 checks all four at once.
#
# ### D.2 Warmup
#
# A trailing window cannot produce a value until it has enough sessions to fill. The audit checks
# that length rather than describing it: a column carrying a value before its window could have
# filled is reading rows that do not exist, and that is the failure it raises on. The declared
# lengths are the windows themselves, so a column that is late because the surface lag pushed it
# one session further out still passes, and a column that is early does not.
#
# **The two audits count on different frames, and the two inputs force it.** A name trades on
# every session and is quoted on the option surface on some of them, so a window over the share
# price and a window over the surface are not the same length of history even when they carry the
# same number. Counting a realized-volatility window in surface rows would report a warmup that
# had elapsed before the sessions it spans had happened, which is the audit raising on a frame
# rather than on a defect. Each audit runs on the frame its windows were taken over rather than on
# the matrix that ships, because the null policy in Section E removes rows from the middle of a
# series and counting bars after it would understate the history every column consumed.

# %%
warmup_audit(
    assemble(daily, surface_raw),
    {
        f"iv_30_atm_z_{W['iv_zscore'][1]}": W["iv_zscore"][1],
        f"iv_30_atm_pct_{W['iv_percentile']}": W["iv_percentile"],
        f"iv_mom_{W['iv_momentum'][1]}d": W["iv_momentum"][1],
        f"skew_rr_z_{W['skew_zscore']}": W["skew_zscore"],
        f"term_ratio_z_{W['term_zscore']}": W["term_zscore"],
        f"vrp_z_{W['vrp_zscore']}": W["vrp_zscore"],
    },
    entity=ENTITY,
)

# %%
warmup_audit(
    realized_volatility(daily).join(equity_momentum(daily), on=PANEL_KEY),
    {
        f"rv_{W['realized_vol'][1]}": W["realized_vol"][1],
        f"gk_vol_{W['garman_klass']}": W["garman_klass"],
        f"vol_of_vol_{W['vol_of_vol']}": W["vol_of_vol"],
        f"mom_{W['momentum'][-1]}d": W["momentum"][-1],
        "mom_skip_recent": W["skip_start"],
    },
    entity=ENTITY,
)

# %% [markdown]
# ### D.3 Withholding the holdout changes nothing
#
# Trailing, contemporaneous and within-date statistics share a property worth checking directly:
# recomputed on inputs that stop before the holdout, they reproduce the same values on the rows
# the two builds share. A parameter fitted over a whole column does not, because truncating the
# column moves the parameter and with it every row it was applied to. Comparing two builds tests
# every emitted column at once and does not depend on anyone having flagged the transform that
# fits. A value on one side against a null on the other counts as a difference.

# %%
before = pl.col("timestamp") < HOLDOUT_START
seal = assert_values_agree(
    built.filter(before),
    build_features(daily.filter(before), surface_raw.filter(before)),
    columns=feature_cols,
    keys=PANEL_KEY,
)
seal.filter(pl.col("column").is_in([f"iv_30_atm_z_{W['iv_zscore'][1]}", "iv_rank", "vrp_z_63"]))

# %% [markdown]
# ## E. Matrix assembly and coverage
#
# The panel key is `symbol` + `timestamp`. Everything the loaders supplied that is not a feature
# is excluded - the raw OHLC, the daily return the volatility family standardizes, and the two
# intermediates the estimators are assembled from - because a model handed a contemporaneous
# close beside a label derived from the same price series would be reading its own answer.
#
# One null policy is applied once, inside the construction: a row is kept when the carrier and
# the quantity it is priced against are both present, which is the pair the hypothesis is stated
# in. It runs after every trailing window and before every percentile, and Section C says why
# each half of that order is load-bearing. Everything longer than the carrier fills in above the
# boundary, which is what F1 shows. Nothing is imputed beyond the forward fill Section B
# declares, so a family that is thin on a date reads as null rather than as a fabricated level.

# %%
features = built.select([*PANEL_KEY, *feature_cols]).sort(PANEL_KEY)
assert features.select(PANEL_KEY).is_duplicated().sum() == 0, "duplicate panel key"

# %%
coverage = family_coverage(features, assignment, every="1mo")
WARMUP_END = features["timestamp"].unique().sort()[max(f.lookback for f in FAMILIES)]
dropped = surface_raw.height - features.height
print(
    f"{len(feature_cols)} features, {features.height:,} rows, {features[ENTITY].n_unique()} names"
)
print(f"{features['timestamp'].min()} to {features['timestamp'].max()}, warmup ends {WARMUP_END}")
print(f"{dropped:,} rows dropped by the null policy ({dropped / surface_raw.height:.1%})")
floor = coverage.filter(pl.col("timestamp") >= WARMUP_END)
print(
    f"thinnest family-month past warmup {min(floor[c].min() for c in set(assignment.values())):.3f}"
    f", in {min(set(assignment.values()), key=lambda f: floor[f].min())}"
)
register_frame(FAMILIES, feature_cols).select(["family", "columns", "role", "representation"])

# %% [markdown] tags=["results"]
# The matrix carries **45 features** on **473,493 rows** across **626 names**, from **2017-02-01**
# to **2021-12-31**. The null policy dropped **57,010 rows**, **10.7%**, which are the sessions on
# which a name had no quoted surface after the lag and the fill, plus the month of warmup the
# realized-volatility carrier pays at the start of each name's series. Past the warmup boundary at
# **2018-02-01** the thinnest family in any month is **0.574** covered, and it is the skew and term
# structure family: it needs three maturity buckets quoted on the same session, so a name whose far
# bucket goes unquoted loses the whole family for that date rather than one column of it.

# %% [markdown]
# ### F1. Coverage through time
#
# Below the boundary the long-window families are empty by construction - a percentile of a year
# of implied volatility cannot exist until a year of it does - so the axis runs the range the
# data occupies rather than the top sliver a dense matrix would need.

# %%
plot_coverage_through_time(
    coverage,
    warmup_boundary=WARMUP_END,
    title="The long-window families fill in over the first year",
    subtitle="Monthly non-null share per feature family, after the null policy",
    alt=(
        "Line chart of non-null share by feature family by month, on an axis running from "
        "about 0.28 to one. Equity momentum starts lowest and climbs through 2017, reaching "
        "one at the marked warmup boundary at the start of 2018; the cross-sectional ranks "
        "and the implied volatility dynamics climb the same way from about 0.4 and 0.5. "
        "Realized volatility and surface quality lie on one for the whole sample, the implied "
        "volatility level sits near 0.9, the dynamics and the variance premium settle around "
        "0.8, and the skew and term structure family is the lowest and the most ragged "
        "throughout, swinging between roughly 0.57 and 0.75."
    ),
)

# %% [markdown]
# ### F4. The timing contract

# %%
plot_timing_contract(
    FAMILIES,
    bar_unit="trading sessions",
    title="Everything read off the surface waits a session; the share does not",
    subtitle="Register lookback per family; a gap at the right edge is a lag",
    alt=(
        "Horizontal bars, one per feature family, each extending leftward from the decision "
        "line by that family's lookback: 252 sessions for the cross-sectional ranks, the "
        "implied volatility dynamics and equity momentum, 63 for skew and term structure, the "
        "variance premium and realized volatility, and one for the implied volatility level "
        "and the surface quality family. Every option-derived bar stops one session short of "
        "the decision line and the gap is hatched; the two share-derived families, realized "
        "volatility and equity momentum, run flush to it."
    ),
)

# %% [markdown]
# ## F. What the features look like
#
# Four properties decide whether this matrix can be used at all: the scale each feature arrives
# on, whether the cross-section disagrees enough to rank on, how much of the set is one ordering
# under several names, and how long a value lasts. Whether any of it predicts is
# `05_evaluation`'s question, and it is asked there fold by fold rather than here on the whole
# sample.
#
# ### F2. Feature distributions
#
# The carrier family is shown on the scale a reader would judge it: the level, its session-over-
# session change, its two z-scores, where it stands in its own trailing range, and its percentile
# across names. The same quantity looks completely different in level, in z-score and in
# percentile form, which is the point of carrying all three.

# %%
plot_feature_distributions(
    features,
    [
        "iv_30_atm",
        "d_iv_30_atm",
        f"iv_30_atm_z_{W['iv_zscore'][0]}",
        f"iv_30_atm_z_{W['iv_zscore'][1]}",
        f"iv_30_atm_pct_{W['iv_percentile']}",
        "iv_rank",
    ],
    title="The level is skewed where its percentile is flat",
    subtitle="Implied volatility family across all name-sessions, display tails clipped",
    alt=(
        "Six histograms in two rows. The level is a right-skewed hump peaking near 0.25 with a "
        "long upper tail. Its session-over-session change is a narrow spike at zero. The two "
        "z-scores are broad and themselves right-skewed, the 63-session one spanning about "
        "minus two to four and the 252-session one reaching six. The trailing percentile is "
        "right-skewed across its zero-to-one range with a spike at each end, and the "
        "cross-sectional percentile is close to uniform across zero to a hundred."
    ),
)

# %% [markdown]
# ### F3. Cross-sectional dispersion through time
#
# A cross-sectional strategy needs the cross-section to disagree. On a date where the band
# narrows to nothing there is nothing to rank, whatever the average level of implied volatility.
# The band is taken across names within each session and only then averaged over the month:
# pooling a month of name-sessions instead would add the movement of the market's own level from
# session to session to the spread across names on a session, and a ranking model is scored on
# the second alone.

# %%
plot_cross_sectional_dispersion(
    features,
    "ivrv_spread",
    every="1mo",
    title="The spread between implied and realized never closes across names",
    subtitle="Interdecile band of implied minus realized volatility, by month",
    alt=(
        "Shaded band of the 10th to 90th percentile of implied minus realized volatility by "
        "month, with the median drawn through it. The median sits a little above zero for most "
        "of the sample. The band is roughly ten volatility points wide in calm periods, never "
        "narrows to nothing, and opens sharply below zero in the first half of 2020 before "
        "widening again above zero through the second half."
    ),
)

# %% [markdown]
# ### F5. Redundancy structure
#
# Clustering on the distance $1 - |\rho|$ groups features that carry the same ordering, whatever
# the sign. Above the cut two features are close enough that a linear model cannot separate their
# contributions. This states the clusters; choosing one representative from each needs a
# fold-aware criterion and belongs to `05_evaluation`.

# %%
CUT = 0.7
clusters = plot_redundancy_clusters(
    features,
    feature_cols,
    cut=CUT,
    title="The surface points are one ordering under several names",
    subtitle=r"Average linkage on $1 - |\rho_s|$, cut drawn at $|\rho_s| = 0.7$",
    alt=(
        "Dendrogram of every feature in the matrix. The five implied volatility level columns "
        "join each other at very small distances, and the three realized volatility estimators "
        "and their rank join the same block. The 252-session z-score joins the trailing "
        "percentile, the momentum horizons join their own ranks, the variance premium joins "
        "its rank, and the skew and term structure columns form a looser group of their own. "
        "The two surface quality columns attach only near the root, sharing an ordering with "
        "nothing else in the matrix."
    ),
)

# %% [markdown]
# ### F6. Persistence and rank stability
#
# The right-hand panel compares the ordering across consecutive **rebalances**, which
# `config/setup.yaml` declares as a Friday close. The autocorrelation on the left is of the
# feature, not of the return, and it runs to four decision cycles. A feature whose value has
# decayed before the next rebalance cannot support that cadence, however well it predicts on the
# day it is computed. It is estimated per name on pairs of dates exactly one lag apart and
# summarized by the median over names, with a bootstrap interval over names: a correlation pooled
# over every name-date pair would read high whenever names sit at different levels, whether or
# not any one of them persists.

# %%
DECISION_CYCLE = 5
DECISION_DATES = (
    features.group_by(pl.col("timestamp").dt.truncate("1w"))
    .agg(pl.col("timestamp").max().alias("decision"))["decision"]
    .sort()
    .to_list()
)

plot_persistence(
    features,
    [
        "iv_30_atm",
        f"iv_30_atm_z_{W['iv_zscore'][0]}",
        "ivrv_spread",
        f"rv_{W['realized_vol'][0]}",
        f"mom_{W['momentum'][2]}d",
    ],
    entity=ENTITY,
    max_lag=4 * DECISION_CYCLE,
    decision_dates=DECISION_DATES,
    title="The z-score turns over fastest of the five",
    subtitle=f"Median over names to {4 * DECISION_CYCLE} sessions",
    alt=(
        "Two panels. On the left, autocorrelation against lag: the implied volatility level, "
        "the realized volatility and the medium-horizon return all start near one and decay "
        "slowly to roughly 0.45 to 0.55 over twenty sessions, while the z-score and the "
        "implied-minus-realized spread fall much faster and reach zero by the end of the axis. "
        "On the right, the cross-sectional rank correlation between consecutive weekly "
        "rebalances puts the medium-horizon return, the level and the realized volatility "
        "around 0.9, the spread at roughly 0.64, and the z-score lowest at roughly 0.56."
    ),
)

# %% [markdown] tags=["results"]
# Cutting the redundancy tree at $|\rho_s| = 0.7$ leaves **24 clusters** across the **45**
# columns, so nearly half the matrix repeats an ordering another column already carries.

# %%
print(f"{len(set(clusters.values()))} clusters over {len(feature_cols)} features at cut {CUT}")

# %% [markdown]
# ## G. Emit
#
# The parquet is written with a sidecar recording the digest of its values, its row count and key
# columns, and the digest of what it was built from. This stage reads no upstream case-study
# artifact, so the sidecar records the two loaded panels, each restricted to the columns and
# window actually consumed, which is what answers "which data vintage produced these values". The
# digest is computed over content rather than file bytes, so row order and parquet metadata leave
# it alone and any feature value moves it.

# %%
record = write_artifact(
    features,
    FEATURES_DIR / "financial.parquet",
    keys=PANEL_KEY,
    written_by=f"case_studies/{CASE_STUDY_ID}/03_financial_features.py",
    inputs={
        "load_sp500_daily_bars": value_digest(
            daily.select([*PANEL_KEY, "open", "high", "low", "close"])
        ),
        "load_sp500_options_surface": value_digest(surface_raw.select([*PANEL_KEY, *SURFACE_COLS])),
    },
)
print(f"Wrote {display_path(FEATURES_DIR / 'financial.parquet')}, digest {record['digest']}")
print(f"Read by 04_model_based_features.py and 05_evaluation.py, on {PANEL_KEY}")

# %% [markdown]
# ## Key takeaways
#
# - **State the timing contract before writing the feature.** The register fixes each family's
#   lookback and lag in the configuration, and the warmup assertion, the timing figure and the
#   review a reader can run all read those numbers rather than re-deriving them from the code.
# - **Lag a late-published input, and say which frame the lag is counted in.** A shift over a
#   panel's own rows is a shift of one *quoted* row, which is one session only where the panel is
#   dense. Saying so is the difference between a stated tolerance and a silent assumption.
# - **Rank inside the frame a decision is taken over.** A percentile within the decision date is
#   the ordering a long-short book acts on; a rank pooled across dates answers a different
#   question with the same arithmetic.
# - **Test the seal by construction, not by inspection.** Rebuilding the matrix from inputs that
#   stop at the boundary and comparing values catches any transform that fits across the sample,
#   including the ones nobody thought to flag.
# - **Read the matrix before modelling it.** Distribution, dispersion, redundancy and decay each
#   rule out a use: a feature with no cross-sectional spread cannot rank, and one whose ordering
#   decays inside the rebalance cycle cannot be traded at that cadence.
#
# ### Known limitations
#
# - The surface lag and the forward fill are both counted in the surface's own rows, so on a name
#   whose quotes are sparse they reach further back than the session they name. Both err toward
#   older information, never newer, and the quality family is what a downstream stage would screen
#   on.
# - The realized-volatility estimators are computed on close-to-close returns and on the daily
#   range. Neither sees the overnight gap separately, so a name that moves between sessions and
#   sits still inside them reads as calmer than it traded.
# - The cross-sectional percentiles are taken over every name quoted on a date rather than over a
#   liquidity-screened subset, so a thinly quoted name occupies a rank position it may not be
#   tradable at. The screen is applied downstream, where `setup.yaml` declares it.
#
# **Next**: `04_model_based_features.py` adds the features that are themselves model outputs,
# where the rule is estimated from the data rather than written in advance.
