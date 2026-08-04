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
# # ETFs: Feature Engineering
#
# Every model here sees the world through this matrix, so what is computable now bounds what
# can be learned later. Each feature answers one question: at the moment a position is decided,
# which observations are already on the tape, and what does the feature make of them?
#
# ## Learning objectives
#
# - State a feature's timing contract - its lookback and its information lag - before writing
#   the code that computes it
# - Compute trailing and cross-sectional statistics that read no observation dated after the
#   decision timestamp, and declare a lag where an input is not yet available at it
# - Show that withholding later dates leaves every feature value unchanged, which is what
#   separates a trailing statistic from one fitted over the whole sample
# - Read a feature set for scale, dispersion, redundancy and decay before any model sees it
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Sections 8.1-8.6. Reads split- and dividend-adjusted daily bars via `load_etfs()`,
# the Treasury constant-maturity series via `load_macro()`, the tradability gate
# `eligibility.csv` from [`01_feasibility_analysis`](01_feasibility_analysis.ipynb), and
# `config/setup.yaml`. Writes `features/financial.parquet` with a `.digest.json` sidecar, read
# by [`04_model_based_features`](04_model_based_features.ipynb), which builds regime and
# memory-preserving features on top of it, and by
# [`05_evaluation`](05_evaluation.ipynb), which tests fold by fold whether any of it predicts.

# %%
"""ETFs: Feature Engineering."""

import warnings
from datetime import date

import polars as pl
import yaml
from ml4t.engineer.features.momentum import adx, aroon, cci, macd, rsi, stochastic
from ml4t.engineer.features.regime import choppiness_index, hurst_exponent
from ml4t.engineer.features.trend import ema, sma
from ml4t.engineer.features.volatility import natr
from ml4t.engineer.features.volume import obv

from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.feature_engineering import (
    EPS,
    assert_values_agree,
    assign_families,
    clip_within_date,
    cross_sectional_percentile,
    drawdown_block,
    families_from_config,
    family_coverage,
    momentum_volatility_block,
    plot_coverage_through_time,
    plot_cross_sectional_dispersion,
    plot_feature_distributions,
    plot_persistence,
    plot_redundancy_clusters,
    plot_timing_contract,
    register_frame,
    trailing_volume_ratio,
    warmup_audit,
)
from data import load_etfs, load_macro
from utils.artifact_specs import resolve_label_horizon
from utils.paths import display_path, get_case_study_dir

warnings.filterwarnings("ignore")

CASE_DIR = get_case_study_dir("etfs")
FEATURES_DIR = CASE_DIR / "features"

# %% tags=["parameters"]
# Production runs START_DATE as None; CI overrides it to shorten the window. The CI fixture is
# already reduced in breadth, so there is no symbol cap here - the cross-sectional families
# below need a cross-section to rank within.
START_DATE = None

# %% [markdown]
# ## Configuration
#
# Every window, the ranked feature list, the regime threshold, the decision horizon and the
# holdout boundary are declared in `config/setup.yaml` and bound here. A value the notebook
# invents is a second source of truth for a decision the rest of the pipeline reads from one
# place. The horizon fixes how far the persistence figure has to look, because a feature has to
# hold its ordering for at least one decision cycle to be usable at that cadence.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
FAMILIES = families_from_config(setup)
WINDOWS = setup["features"]["windows"]
RANKED = setup["features"]["ranked"]
REGIME_THRESHOLD = setup["features"]["regime_threshold"]
OSC = setup["features"]["oscillators"]
STATE = setup["features"]["state"]
DECISION_CYCLE = int(resolve_label_horizon("etfs", setup["labels"]["primary"], setup).rstrip("Dd"))
HOLDOUT_START = date.fromisoformat(setup["evaluation"]["holdout_start"])

print(f"{len(FAMILIES)} declared families, decision cycle {DECISION_CYCLE} sessions")
print(f"Holdout starts {HOLDOUT_START}; Section D rebuilds the panel without it")

# %% [markdown]
# ## A. What the thesis says should carry information
#
# The hypothesis is cross-sectional and it is about persistence: among liquid ETFs spanning
# equities, bonds, commodities and currencies, the ones that have gained relative to the rest
# continue to over the following month. Three things follow.
#
# The **carrier** is trailing relative performance, so the matrix holds returns and
# risk-adjusted returns at eight horizons rather than one - which horizon the effect lives at
# is an empirical question, not a modelling assumption. The skip-recent construction drops the
# most recent month from the long window, because that month reverses where the twelve before
# it continue.
#
# The **conditioning** is regime: a cross-asset rotation only works while the assets disagree,
# so the matrix also carries volatility, drawdown, trend strength, the equity-bond correlation
# and the shape of the yield curve. These rank nothing against each other; they say which
# environment a ranking is being formed in, which is what the register's `role` column records
# and what no assertion can recover from the values.
#
# The **representation** matters as much as the quantity. A raw return has a distribution that
# drifts with the volatility of the period, so the three carrier features are also carried as
# cross-sectional percentiles, comparable across dates by construction.
#
# The register is declared in `config/setup.yaml`, one row per family: what it reads, how far
# back, and with what delay. A `lag` of zero means the input is on the tape at the decision
# itself, which every price-derived family here is: the decision is taken at the close and
# executes at the next open. The yield curve is the one exception, at one session: a Treasury
# series dated t is treated as available from the close of t+1, which is what
# `config/setup.yaml` declares as the case study's macro policy. That lag is why it is a family
# of its own rather than sharing one with the cross-asset correlation, which reads prices.

# %%
register_frame(FAMILIES).select(
    ["family", "role", "inputs", "lookback (bars)", "lag (bars)", "frame"]
)

# %% [markdown]
# ## B. Inputs and their observability
#
# Daily bars are split- and dividend-adjusted, so a trailing return spans a corporate action
# without a jump. Eligibility is a per-year tradability gate holding the cross-section to ETFs
# that traded at volume that year, rather than letting a fund listed in 2019 appear in a 2011
# ranking. The Treasury series skips market holidays, which is why C.4 joins it backward in time
# rather than on an exact date.

# %%
prices = (
    load_etfs()
    .select(["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    .sort(["symbol", "timestamp"])
)
if START_DATE is not None:
    prices = prices.filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())

eligibility = pl.read_csv(CASE_DIR / "eligibility.csv")
yield_curve = (
    load_macro()
    .select("timestamp", ((pl.col("dgs10") - pl.col("dgs2")) / 100).alias("slope"))
    .drop_nulls()
    .sort("timestamp")
)
print(f"{len(prices):,} bars over {prices['symbol'].n_unique()} ETFs")
print(f"{len(yield_curve):,} sessions of 10y-2y spread")

# %% [markdown]
# ## C. Feature construction, one subsection per family
#
# ### C.1 Momentum, volatility and their differences
#
# The trailing return, volatility and risk-adjusted return block is shared with the other
# panel case studies, because three of them had computed it character for character with five
# different denominator guards between them. What stays here is what is specific to this
# universe: the skip-recent windows, the differences between horizons, and the ratio of a short
# volatility window to a long one.


# %%
def momentum_features(df: pl.DataFrame) -> pl.DataFrame:
    """The shared trailing block, plus the differences this case study builds on it."""
    df = momentum_volatility_block(
        df,
        entity="symbol",
        return_windows=WINDOWS["momentum"],
        volatility_windows=WINDOWS["volatility"],
    )
    held = pl.col("close").shift(WINDOWS["skip_recent"]).over("symbol")
    return df.with_columns(
        (held / pl.col("close").shift(252).over("symbol").clip(lower_bound=EPS) - 1).alias(
            "skip_recent_12_1"
        ),
        (held / pl.col("close").shift(126).over("symbol").clip(lower_bound=EPS) - 1).alias(
            "skip_recent_6_1"
        ),
        (pl.col("ret_21d") - pl.col("ret_63d")).alias("mom_accel_short"),
        (pl.col("ret_63d") - pl.col("ret_126d")).alias("mom_accel_medium"),
        (pl.col("ret_126d") - pl.col("ret_252d")).alias("mom_accel_long"),
        (pl.col("vol_21d") / pl.col("vol_63d").clip(lower_bound=EPS)).alias("vol_ratio_short"),
        (pl.col("vol_63d") / pl.col("vol_126d").clip(lower_bound=EPS)).alias("vol_ratio_medium"),
    )


# %% [markdown]
# ### C.2 Oscillators, trend ratios and range
#
# These come from `ml4t.engineer.features` rather than being written here. The smoothing
# convention inside an oscillator is where two implementations of one name diverge - Wilder's
# recursive average and a simple moving average of the same gains give different numbers - and
# a shared implementation is what keeps `rsi_14` meaning one thing across the nine case studies.


# %%
def oscillator_features(df: pl.DataFrame) -> pl.DataFrame:
    """Bounded oscillators, moving-average ratios, normalized range and regime exponents."""
    df = df.with_columns(
        *[rsi("close", period=p).over("symbol").alias(f"rsi_{p}") for p in OSC["rsi"]],
        macd("close", fast_period=OSC["macd_fast"], slow_period=OSC["macd_slow"])
        .over("symbol")
        .alias("macd_line"),
        adx("high", "low", "close", period=OSC["adx"]).over("symbol").alias(f"adx_{OSC['adx']}"),
        *[
            cci("high", "low", "close", period=p).over("symbol").alias(f"cci_{p}")
            for p in OSC["cci"]
        ],
        stochastic("high", "low", "close", fastk_period=OSC["stochastic"])
        .over("symbol")
        .alias("stoch_k"),
        aroon("high", "low", timeperiod=OSC["aroon"]).over("symbol").alias("_aroon"),
        natr("high", "low", "close", period=OSC["natr"])
        .over("symbol")
        .alias(f"natr_{OSC['natr']}"),
        choppiness_index("high", "low", "close", period=OSC["choppiness"])
        .over("symbol")
        .alias(f"chop_{OSC['choppiness']}"),
        # Rounded, and it is the only column here that is. The Hurst exponent is the slope of a
        # log-log least-squares fit over rescaled-range statistics, so its last bits depend on
        # the order the cumulative sums accumulate in, which differs between a full panel and a
        # truncated one on some BLAS builds even though the window is fixed at `period` and the
        # lag set at `period // 2`. That is float accumulation, not a look-ahead: on this data
        # the two builds agree to 0.0 exactly, while CI trips D.3's 1e-12 tolerance. Six
        # decimals is far below any reading of a persistence exponent and far above the noise,
        # so D.3 tests the feature rather than the platform.
        hurst_exponent("close", period=OSC["hurst"])
        .over("symbol")
        .round(6)
        .alias(f"hurst_{OSC['hurst']}"),
        *[
            (pl.col("close") / sma("close", period=p).over("symbol")).alias(f"sma_ratio_{p}")
            for p in OSC["sma"]
        ],
        (pl.col("close") / ema("close", period=OSC["ema"]).over("symbol")).alias(
            f"ema_ratio_{OSC['ema']}"
        ),
        pl.col("close").rolling_mean(OSC["bollinger"]).over("symbol").alias("_mid"),
        pl.col("close").rolling_std(OSC["bollinger"]).over("symbol").alias("_sd"),
    )
    return df.with_columns(
        (pl.col("_aroon").struct.field("up") - pl.col("_aroon").struct.field("down")).alias(
            "aroon_diff"
        ),
        pl.when(pl.col("_sd") > 0)
        .then((pl.col("close") - (pl.col("_mid") - 2 * pl.col("_sd"))) / (4 * pl.col("_sd")))
        .alias(f"bb_pctb_{OSC['bollinger']}"),
    ).drop(["_aroon", "_mid", "_sd"])


# %% [markdown]
# ### C.3 Drawdown, volume and distance from extremes
#
# `max_dd_63d` is the share by which price currently sits below its highest close of the
# trailing quarter, so it is zero at a new high and negative otherwise - the *current*
# drawdown, not the worst decline inside the window, which is a different statistic. Relative
# volume is clipped at the 1st and 99th percentile of its own date, because an index rebalance
# puts one ETF orders of magnitude above its own average and one such row otherwise sets the
# scale every model sees.


# %%
def drawdown_and_extremes(df: pl.DataFrame) -> pl.DataFrame:
    """Drawdown, on-balance volume and position in the 52-week range."""
    df = drawdown_block(df, entity="symbol", windows=WINDOWS["drawdown"])
    df = df.with_columns(obv("close", "volume").over("symbol").alias("_obv"))
    return df.with_columns(
        (
            (pl.col("_obv") - pl.col("_obv").rolling_mean(STATE["obv_zscore"]).over("symbol"))
            / pl.col("_obv").rolling_std(STATE["obv_zscore"]).over("symbol").clip(lower_bound=EPS)
        ).alias(f"obv_zscore_{STATE['obv_zscore']}d"),
        (pl.col("log_return") > 0)
        .cast(pl.Float64)
        .rolling_mean(STATE["positive_share"])
        .over("symbol")
        .alias(f"pct_positive_{STATE['positive_share']}d"),
        (
            pl.col("close")
            / pl.col("close").rolling_max(STATE["extremes"]).over("symbol").clip(lower_bound=EPS)
        ).alias("dist_52w_high"),
        (
            pl.col("close")
            / pl.col("close").rolling_min(STATE["extremes"]).over("symbol").clip(lower_bound=EPS)
        ).alias("dist_52w_low"),
    ).drop("_obv")


# %% [markdown]
# ### C.4 Cross-asset state, macro state and cross-sectional position
#
# The SPY-TLT correlation and the curve slope are one number per date, shared by every ETF. A
# rolling window reads row order rather than the timestamp column, so the pair is re-sorted after
# the join that builds it, and the curve joins backward in time so a market holiday carries the
# previous session's spread forward and never a later one's. Three carrier features are then also
# carried as their percentile within the date; ranking all eight horizons would add nothing,
# because eight highly correlated returns carry one ordering between them.


# %%
def regime_and_state(df: pl.DataFrame) -> pl.DataFrame:
    """Equity-bond correlation and yield-curve state, broadcast to every row."""
    pair = (
        df.filter(pl.col("symbol") == "SPY")
        .select("timestamp", pl.col("log_return").alias("_spy"))
        .join(
            df.filter(pl.col("symbol") == "TLT").select(
                "timestamp", pl.col("log_return").alias("_tlt")
            ),
            on="timestamp",
            how="inner",
        )
        .sort("timestamp")
        .select(
            "timestamp",
            pl.rolling_corr(pl.col("_spy"), pl.col("_tlt"), window_size=STATE["correlation"]).alias(
                f"corr_spy_tlt_{STATE['correlation']}d"
            ),
        )
    )
    # Stamped with its availability date, not its observation date. `config/setup.yaml`
    # declares the macro policy as `alfred_initial_release_close_lagged`: a value dated t
    # is available for a decision at the close of t+1. The offset is a calendar day rather
    # than a shift down the Treasury series, because the two calendars differ - a shift
    # would mean "the next day the Treasury published", so on Columbus Day, when NYSE
    # trades and FRED does not, it would add a second session of delay.
    curve = yield_curve.select(
        pl.col("timestamp").dt.offset_by("1d"),
        pl.when(pl.col("slope") > REGIME_THRESHOLD).then(1).otherwise(0).alias("regime"),
        pl.col("slope").alias("yield_curve_slope"),
        (
            (pl.col("slope") - pl.col("slope").rolling_mean(STATE["curve_zscore"]))
            / pl.col("slope").rolling_std(STATE["curve_zscore"]).clip(lower_bound=EPS)
        ).alias("yield_curve_zscore"),
    )
    return (
        df.join(pair, on="timestamp", how="left")
        .sort("timestamp")
        .join_asof(curve, on="timestamp", strategy="backward")
    )


def gate_to_eligible(df: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows the annual tradability gate excludes, before anything ranks them."""
    return (
        df.with_columns(pl.col("timestamp").dt.year().alias("_year"))
        .join(
            eligibility.select("symbol", pl.col("eligible_year").alias("_year")),
            on=["symbol", "_year"],
            how="semi",
        )
        .drop("_year")
    )


def per_entity_features(df: pl.DataFrame) -> pl.DataFrame:
    """Everything computed from one ETF's own history, gate not yet applied.

    The relative-volume ratio is here rather than after the gate: its trailing mean
    has to read every bar the ETF traded, or an ETF admitted to the universe this
    year divides by a mean of its first few eligible days, and one that re-enters
    after a gap averages across the gap. Only the clip is a cross-sectional step.
    """
    return (
        df.pipe(momentum_features)
        .pipe(oscillator_features)
        .pipe(drawdown_and_extremes)
        .pipe(regime_and_state)
        .pipe(trailing_volume_ratio, entity="symbol", windows=WINDOWS["volume"])
    )


def clip_and_rank(df: pl.DataFrame) -> pl.DataFrame:
    """The two within-date steps, over the eligible cross-section only."""
    clipped = clip_within_date(
        df, columns=[f"vol_ratio_{w}d" for w in WINDOWS["volume"]], time="timestamp"
    )
    return clipped.with_columns(
        cross_sectional_percentile(col, "timestamp").alias(f"{col}_rank") for col in RANKED
    )


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """The whole construction, as one function Section D can re-run on a shorter panel.

    The eligibility gate sits after everything computed per entity and before the two
    statistics computed within a date. A percentile and a clip are properties of the
    cross-section they are taken over, so an ETF the strategy cannot trade must not be in
    that cross-section: ranking against it moves the number written for every ETF that is.
    """
    return df.pipe(per_entity_features).pipe(gate_to_eligible).pipe(clip_and_rank)


EXCLUDED = {"symbol", "timestamp", "open", "high", "low", "close", "volume", "log_return"}
per_entity = per_entity_features(prices)
built = per_entity.pipe(gate_to_eligible).pipe(clip_and_rank)
feature_cols = [c for c in built.columns if c not in EXCLUDED]
print(f"{len(built):,} eligible bars carrying {len(feature_cols)} features")

# %% [markdown]
# ## D. The timing contract
#
# ### D.1 What each construction reads
#
# Three kinds of operation appear above. A **rolling** window ends at its own row and reads a
# fixed number of bars backward. A **cross-sectional** statistic - the three percentiles, and
# the volume clip - is taken with `.over("timestamp")`, so it reads every ETF on that date and
# no other. An **as-of join** carries the most recent macro value at or before the row's
# timestamp. D.2 and D.3 establish that none of the three reaches forward.
#
# ### D.2 Warmup
#
# A trailing window cannot produce a value until it has enough bars to fill, so every family has
# a leading stretch of nulls as long as its lookback. The audit checks that length rather than
# describing it: a column carrying a value before its window could have filled is reading bars
# that do not exist, and that is the failure it raises on. It runs on the panel before the
# eligibility gate, because the gate drops the early rows the warmup stretch is made of.

# %%
warmup_audit(
    per_entity,
    {
        "ret_252d": 252,
        "skip_recent_12_1": 252,
        "sharpe_252d": 252,
        "vol_252d": 252,
        "dist_52w_high": 252,
        "sma_ratio_200": 200,
        "max_dd_126d": 126,
        "hurst_100": 100,
        "obv_zscore_63d": 63,
    },
    entity="symbol",
)

# %% [markdown]
# ### D.3 Withholding the holdout changes nothing
#
# Trailing and within-date statistics share a property worth checking directly: recomputed on a
# panel that stops before the holdout, they reproduce the same values on the rows the two panels
# share. A parameter fitted over a whole column - a winsorization bound, a scaler, an encoder -
# does not, because truncating the column moves the parameter and with it every row it was
# applied to. Building the panel twice and comparing tests the whole construction at once -
# every emitted column, not a sample - and does not depend on anyone having flagged the
# transform that fits. A value on one side against a null on the other counts as a difference,
# because that is the form of the failure a null-skipping comparison hides.

# %%
seal = assert_values_agree(
    built.filter(pl.col("timestamp") < HOLDOUT_START),
    build_features(prices.filter(pl.col("timestamp") < HOLDOUT_START)),
    columns=feature_cols,
    keys=["timestamp", "symbol"],
)
seal.filter(pl.col("column").is_in(["ret_126d", "ret_126d_rank", "yield_curve_zscore"]))

# %% [markdown]
# ## E. Matrix assembly and coverage
#
# The panel key is `symbol` + `timestamp`. Raw OHLC, volume and the intermediate log return are
# excluded: they are the inputs the features are made of, and a model handed the contemporaneous
# log return beside a label derived from the same prices would be reading its own answer. One
# null policy is applied once - a row is kept when the longest carrier feature has warmed up,
# which is the point past which every family is dense.

# %%
features = (
    built.select(["timestamp", "symbol", *feature_cols])
    .drop_nulls(subset=["sharpe_126d"])
    .sort(["timestamp", "symbol"])
)
assert features.select(["timestamp", "symbol"]).is_duplicated().sum() == 0, "duplicate panel key"
assignment = assign_families(feature_cols, FAMILIES)
register_frame(FAMILIES, feature_cols).select(["family", "columns", "role", "representation"])

# %% [markdown]
# ### F1. Coverage through time
#
# The axis is scaled to the data, not pinned to zero: this matrix is dense everywhere, and on a
# zero-based axis every family would draw as one flat line at the top. What is left to see is
# where the residual percent sits, and it sits in the families with the longest windows - an ETF
# admitted by the eligibility gate partway through a year has not yet filled a 252-session
# lookback.

# %%
plot_coverage_through_time(
    family_coverage(features, assignment, every="1mo"),
    warmup_boundary=features["timestamp"].min(),
    title="No family is ever more than about one percent thin",
    subtitle="Monthly non-null share per feature family, on an axis scaled to the data",
    alt=(
        "Line chart of non-null share by feature family on a y-axis spanning roughly 0.985 to "
        "one. Most families sit exactly at one throughout. The long-window families - momentum, "
        "risk-adjusted momentum, extremes - dip by up to about one percent in scattered months, "
        "and no family is ever materially incomplete."
    ),
)

# %% [markdown]
# ### F4. The timing contract

# %%
plot_timing_contract(
    FAMILIES,
    bar_unit="trading sessions",
    title="Only the yield curve waits for its input to publish",
    subtitle="Register lookback per family; a gap at the right edge is an information lag",
    alt=(
        "Horizontal bars, one per feature family, each extending leftward from the decision "
        "line by that family's lookback, from 63 sessions for volume and the cross-asset "
        "correlation to 252 for momentum and the yield curve. Every bar reaches the decision "
        "line except the yield curve, which stops one session short of it."
    ),
)

# %% [markdown]
# ## F. What the features look like
#
# Four properties decide whether this matrix can be used at all: the scale each feature arrives
# on, whether the cross-section disagrees enough to rank on, how much of the set is one ordering
# under several names, and how long a value lasts. `05_evaluation` is where the matrix is tested
# fold by fold for whether any of it predicts.
#
# ### F2. Feature distributions

# %%
plot_feature_distributions(
    features,
    ["ret_21d", "ret_126d", "ret_252d", "sharpe_21d", "sharpe_126d", "sharpe_252d"],
    title="A longer window widens the return and narrows the ratio",
    subtitle="Trailing return and risk-adjusted return, display tails clipped at 0.5%",
    alt=(
        "Six histograms in two rows. Along the top the trailing returns broaden from a narrow "
        "right-skewed peak spanning about plus or minus 0.2 at 21 sessions to a wide body "
        "reaching 0.75 at 252. Below, the annualized risk-adjusted returns move the other way, "
        "from roughly minus five to ten at 21 sessions down to minus two to three at 252, and "
        "are close to symmetric at every horizon."
    ),
)

# %% [markdown]
# ### F3. Cross-sectional dispersion through time
#
# A cross-sectional strategy needs the cross-section to disagree. On a date where the band
# narrows to nothing there is nothing to rank, whatever the average level of the feature.

# %%
plot_cross_sectional_dispersion(
    features,
    "ret_126d",
    every="1mo",
    title="The gap between leading and lagging ETFs widens in stress",
    subtitle="10th-90th percentile of the six-month trailing return across the eligible universe",
    alt=(
        "Shaded band of the 10th to 90th percentile of six-month trailing return by month, with "
        "the median drawn through it. The band is a few tens of percent wide in calm periods "
        "and widens sharply in 2020 and again in 2022."
    ),
)

# %% [markdown]
# ### F5. Redundancy structure
#
# Clustering on the distance $1 - |\rho|$ groups features that carry the same ordering, whatever
# the sign. Above the cut two features are close enough that a linear model cannot separate
# their contributions. This states the clusters; choosing one representative from each needs a
# fold-aware criterion and belongs to `05_evaluation`.

# %%
clusters = plot_redundancy_clusters(
    features,
    feature_cols,
    cut=0.7,
    title="Adjacent horizons pair off; short and long momentum do not",
    subtitle=r"Average linkage on $1 - |\rho|$, cut drawn at $|\rho| = 0.7$",
    alt=(
        "Dendrogram of every feature in the matrix. Neighbouring horizons join at very small "
        "distances - the five- and ten-day returns with their risk-adjusted twins, the six- and "
        "nine-month returns with the 200-day trend ratio - but the short-horizon block and the "
        "long-horizon block only merge near the root, and the volatility, oscillator and macro "
        "features attach as separate branches."
    ),
)

# %% [markdown]
# ### F6. Persistence and rank stability
#
# The right-hand panel compares the ordering across consecutive **rebalances**, which
# `config/setup.yaml` declares as `monthly_month_end` - a varying number of sessions apart, so
# a fixed lag would correlate dates the strategy never puts side by side.
#
# The autocorrelation on the left is of the feature, not of the return, and it runs past one
# full decision cycle. A feature whose value has decayed inside a cycle cannot support that
# rebalance cadence, however well it predicts on the day it is computed. It is estimated per ETF
# on pairs of dates exactly one lag apart and summarized by the median over ETFs, with a
# bootstrap interval over ETFs: a correlation pooled over every ETF-date pair would read high
# whenever ETFs sit at different levels, whether or not any one of them persists.

# %%
DECISION_DATES = (
    features.group_by(pl.col("timestamp").dt.truncate("1mo"))
    .agg(pl.col("timestamp").max().alias("decision"))["decision"]
    .sort()
    .to_list()
)

plot_persistence(
    features,
    ["ret_21d", "ret_126d", "sharpe_126d", "vol_63d", "rsi_14"],
    entity="symbol",
    max_lag=2 * DECISION_CYCLE,
    decision_dates=DECISION_DATES,
    title="The long-window carriers still hold their ordering a month out",
    subtitle=(
        f"Median over ETFs to {2 * DECISION_CYCLE} sessions; rank correlation across rebalances"
    ),
    alt=(
        "Two panels. On the left, autocorrelation against lag: the six-month return, the "
        "three-month volatility and the six-month risk-adjusted return are all still above "
        "0.7 at 42 sessions, while the one-month return reaches zero at exactly 21 sessions - "
        "the length of its own window - and the 14-day oscillator levels off near 0.2. The "
        "bootstrap ribbon around each curve is only a few hundredths wide. On the right, the "
        "cross-sectional rank correlation between consecutive rebalances separates the features "
        "sharply: about 0.95 for the three-month volatility and 0.85 for the two six-month "
        "carriers, against 0.2 for the oscillator and almost nothing for the one-month return, "
        "whose window is about one rebalance long."
    ),
)

# %% [markdown]
# ## G. Emit
#
# The parquet is written with a sidecar recording the digest of its values, its row count and key
# columns, and the digests of what it was built from. The digest is computed over content rather
# than file bytes, so row order and parquet metadata leave it alone and any feature value moves
# it. That is the property the registry's own hashes lack: a feature-set *name* reaches the
# registry, a feature-set *value* does not, so a corrected feature moves every number downstream
# without changing anything the registry stores.

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
record = write_artifact(
    features,
    FEATURES_DIR / "financial.parquet",
    keys=["symbol", "timestamp"],
    written_by="case_studies/etfs/03_financial_features.py",
    inputs={
        "eligibility.csv": value_digest(eligibility),
        "load_etfs": value_digest(prices),
        "load_macro:dgs10-dgs2": value_digest(yield_curve),
    },
)
print(f"Wrote {display_path(FEATURES_DIR / 'financial.parquet')}")

# %% [markdown] tags=["results"]
# The matrix carries **57 features** on **396,186 rows** across **99 ETFs**, from **2007-01-03**
# to **2025-12-31**, under content digest **a1e90493a7de9d0f**. Cutting the redundancy tree
# leaves **22 clusters**, so well over half the columns repeat an ordering another column
# already carries.

# %%
print(f"{len(feature_cols)} features, {len(features):,} rows, {features['symbol'].n_unique()} ETFs")
print(f"{features['timestamp'].min()} to {features['timestamp'].max()}, digest {record['digest']}")
print(f"{len(set(clusters.values()))} redundancy clusters")

# %% [markdown]
# ## Key takeaways
#
# - **State the timing contract before writing the feature.** The register fixes each family's
#   lookback and lag in the configuration, and the warmup assertion, the timing figure and the
#   review a reader can run all read those numbers rather than re-deriving them from the code.
# - **Test the seal by construction, not by inspection.** Rebuilding the panel with later dates
#   withheld and comparing values catches any transform that fits across the sample, including
#   the ones nobody thought to flag.
# - **Rank inside the date.** A percentile taken within one timestamp is comparable across dates
#   in a way a raw level, whose distribution drifts with the period's volatility, is not.
# - **Read the matrix before modelling it.** Distribution, dispersion, redundancy and decay each
#   rule out a use: a feature with no cross-sectional spread cannot rank, and one whose ordering
#   decays inside the rebalance cycle cannot be traded at that cadence.
#
# ### Known limitations
#
# - The cross-asset regime feature is one pair, SPY against TLT. It describes the equity-bond
#   relationship and says nothing about commodities or currencies, which are a third of the
#   universe.
# - The eligibility gate is annual, so an ETF that lost liquidity in June stays in the
#   cross-section until December.
# - The yield-curve features carry the configured one-session availability lag, but they read
#   the revised Treasury history rather than the initial release. A value revised later is not
#   the value the decision could have seen, whatever its timestamp says.
# - Every feature here is a rule written in advance. `04_model_based_features` adds the features
#   that are themselves model outputs, where the rule is estimated from the data.
