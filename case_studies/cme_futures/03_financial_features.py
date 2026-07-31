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
# # CME Futures: Feature Engineering
#
# A futures panel offers something an equity panel does not: the price of the same commodity for
# delivery at two different dates, quoted side by side. The gap between them is the carry, and it is
# knowable at the moment a position is decided rather than inferred from past returns. This notebook
# builds that quantity and the momentum, volatility and calendar families it is read against, states
# what window and what delay each one carries, and shows that none of them reads a settlement dated
# at or after the decision.
#
# ## Learning objectives
#
# - Derive carry and curvature from a term structure, and see why they need the unadjusted prices
#   when every return in the same notebook needs the adjusted ones
# - State each family's lookback and information lag before writing the code that computes it
# - Rank within the decision date **and the contract position**, so a front-month carry is never
#   compared against a deferred-month one
# - Show that withholding later dates leaves every feature value unchanged, which is what separates
#   a trailing statistic from one fitted over the whole sample
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Sections 8.1-8.6. Reads session-aligned daily settlement bars for three tenors per
# product via `load_cme_futures()`, and `config/setup.yaml`. Writes `features/financial.parquet`
# with a `.digest.json` sidecar, read by
# [`04_model_based_features`](04_model_based_features.ipynb), which builds regime and
# memory-preserving features on top of it, and by [`05_evaluation`](05_evaluation.ipynb), which
# tests fold by fold whether any of it predicts.

# %%
"""CME Futures: Feature Engineering."""

import warnings
from datetime import date

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import yaml
from ml4t.engineer.features.momentum import rsi
from ml4t.engineer.features.regime import variance_ratio
from ml4t.engineer.features.volatility import yang_zhang_volatility
from plotly.subplots import make_subplots

from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.feature_engineering import (
    EPS,
    assert_values_agree,
    assign_families,
    cross_sectional_percentile,
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
    rolling_zscore,
    trailing_return,
    warmup_audit,
)
from data import load_cme_futures
from utils.artifact_specs import resolve_label_horizon
from utils.paths import display_path, get_case_study_dir
from utils.style import COLORS, style_subplot_titles

warnings.filterwarnings("ignore")

CASE_DIR = get_case_study_dir("cme_futures")
FEATURES_DIR = CASE_DIR / "features"

# %% tags=["parameters"]
# Production runs START_DATE as None; CI overrides it to shorten the window. There is no product
# cap: the universe is 30 products by construction and the cross-sectional families below need the
# whole cross-section to rank within.
START_DATE = None

# %% [markdown]
# ## Configuration
#
# Every window, the ranked-column mapping, the composite definitions, the sector map, the decision
# horizon and the holdout boundary are declared in `config/setup.yaml` and bound here. A window
# retyped in the notebook is a second source of truth for a decision the register, the warmup
# assertion and the timing figure all have to agree on. The horizon fixes how far the persistence
# figure has to look, because a feature has to hold its ordering for at least one decision cycle to
# be tradable at that cadence.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
FEATURES = setup["features"]
FAMILIES = families_from_config(setup)
WINDOWS = FEATURES["windows"]
RANKED = FEATURES["ranked"]
LEVEL = FEATURES["thresholds"]
SECTOR = {p: g for g, ps in setup["universe"]["product_groups"].items() for p in ps}
SEASONAL = list(FEATURES["seasonal_sectors"])
PERIODS_PER_YEAR = setup["evaluation"]["periods_per_year"]
DECISION_CYCLE = int(
    resolve_label_horizon("cme_futures", setup["labels"]["primary"], setup).rstrip("Dd")
)
HOLDOUT_START = date.fromisoformat(setup["evaluation"]["holdout_start"])

# The panel key, and the partition every cross-sectional statistic is taken over.
ENTITY = ["product", "position"]
PANEL_KEY = ["product", "position", "timestamp"]
WITHIN_DATE = ["timestamp", "position"]

print(f"{len(FAMILIES)} declared families, decision cycle {DECISION_CYCLE} sessions")
print(f"Holdout starts {HOLDOUT_START}; Section D rebuilds the panel without it")

# %% [markdown]
# ## A. What the thesis says should carry information
#
# The hypothesis is cross-sectional and it is about the shape of the curve: among 30 CME products
# spanning equity index, rates, energy, metals, currencies, grains and livestock, the ones whose
# near contract trades above the deferred one earn the roll that shape implies, and go on earning
# it over the following week. Three things follow.
#
# The **carrier** is carry, and it is the one family here that is not a function of past returns.
# It is quoted directly, so it needs no lookback at all - the smoothing, the z-score and the change
# in carry are there to say whether today's spread is unusual *for this product*, which is a
# different claim from the spread being wide.
#
# The **conditioning** is momentum and volatility. Carry and trend are separate premia in the
# futures literature, so the matrix carries returns at four horizons, their risk-adjusted twins,
# the dispersion those are earned against, and whether the market is trending or reverting. The
# calendar families rank nothing: they say which part of the year a grain or a heating-fuel curve
# is being read in, which is what the register's `role` column records and what no assertion can
# recover from the values.
#
# The **frame** is what makes this case study different from the panel ones. Three tenors trade per
# product-date, and a front-month carry and a deferred-month carry are not the same quantity, so
# every percentile below is taken within a date **and** a contract position. Ranking across
# positions would put ES front month and ES third month in one ordering and call the difference
# information.
#
# The register is declared in `config/setup.yaml`, one row per family. Every lag in it is zero:
# settlement prices are published at the close of the session they are dated, and the decision is
# taken at Friday's close, so no family here waits for an input to be released.

# %%
register_frame(FAMILIES).select(
    ["family", "role", "inputs", "lookback (bars)", "lag (bars)", "frame"]
)

# %% [markdown]
# ## B. Inputs and their observability
#
# Each row is one product, one contract position and one session. Position 0 is the front month,
# 1 the second, 2 the third. Two price series arrive per row and they answer different questions.
# `adj_close` is roll-adjusted, so a return spans a roll without the jump the contract change would
# otherwise put there; `raw_close` is what the exchange settled, so a spread between two tenors
# measures today's curve rather than the roll history baked into each adjusted series.
#
# A settlement at or below zero is not a price a ratio can be taken against. WTI settled negative on
# 2020-04-20, and one such row otherwise propagates through every window that contains it as a
# return of the wrong sign and a volatility of the wrong size. Nulling it once, here, is the single
# data policy the rest of the notebook inherits.

# %%
PRICES = ["adj_open", "adj_high", "adj_low", "adj_close", "raw_close"]
bars = (
    load_cme_futures()
    .rename({"session_date": "timestamp", "tenor": "position"})
    .with_columns(pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(None).alias(c) for c in PRICES)
    .sort([*ENTITY, "timestamp"])
)
if START_DATE is not None:
    bars = bars.filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())

print(f"{len(bars):,} bars over {bars['product'].n_unique()} products at 3 contract positions")
print(f"{bars['timestamp'].min()} to {bars['timestamp'].max()}")

# %% [markdown]
# ## C. Feature construction, one subsection per family
#
# ### C.1 Term structure
#
# Carry is the annualized gap between the front and second settlement, and curvature is the second
# difference across all three tenors - a curve can be in backwardation and still be bent, and the
# two say different things about where the pressure on it sits. Both are one value per
# product-date, shared by the three positions, and both read `raw_close` for the reason B gives.
#
# There is no library call for either. `ml4t.engineer.features` covers statistics of a single price
# series, and a term structure is a relation between contemporaneous series, so this subsection is
# the one place in the notebook where the construction is local rather than imported.


# %%
def term_structure(bars: pl.DataFrame) -> pl.DataFrame:
    """Carry, curvature, and how unusual each is for this product, per product-date."""
    smoothing = WINDOWS["carry_smoothing"]
    curve = (
        bars.filter(pl.col("position") <= 2)
        .pivot(on="position", index=["product", "timestamp"], values="raw_close")
        .rename({"0": "c0", "1": "c1", "2": "c2"})
        .sort(["product", "timestamp"])
        .with_columns(
            ((pl.col("c0") - pl.col("c1")) / pl.col("c0") * 12).alias("carry_pct"),
            ((pl.col("c0") - 2 * pl.col("c1") + pl.col("c2")) / pl.col("c1")).alias(
                "curve_curvature_norm"
            ),
        )
        # The windows below span the sessions on which the curve was quoted, not calendar
        # sessions: a product-date where the second contract did not trade has no carry, and
        # keeping it would null the smoothed level for the three weeks that follow it. The
        # left join in `build_features` puts those dates back with a null carry.
        .drop_nulls("carry_pct")
    )
    band = LEVEL["carry_regime_band"]
    curve = curve.with_columns(
        pl.col("carry_pct").rolling_mean(smoothing).over("product").alias("carry_21d"),
        pl.col("curve_curvature_norm")
        .rolling_mean(smoothing)
        .over("product")
        .alias(f"curvature_{smoothing}d"),
        pl.when(pl.col("carry_pct") > band)
        .then(1)
        .when(pl.col("carry_pct") < -band)
        .then(-1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("carry_regime_num"),
    )
    return curve.with_columns(
        *[
            (pl.col("carry_21d") - pl.col("carry_21d").shift(w).over("product")).alias(
                f"carry_momentum_{w}d"
            )
            for w in WINDOWS["carry_momentum"]
        ],
        *[
            rolling_zscore("carry_21d", w, "product").clip(-5.0, 5.0).alias(f"carry_zscore_{w}d")
            for w in WINDOWS["carry_zscore"]
        ],
    ).drop("c0", "c1", "c2")


# %% [markdown]
# ### C.2 Momentum, risk-adjusted momentum and volatility
#
# The trailing return, volatility and Sharpe block is the shared one, called here with this panel's
# entity key so that a shift means "the previous session for this product at this contract
# position" and never "the previous row in the file". A shared implementation is what keeps
# `sharpe_126d` meaning one thing across the case studies: the mean log return over its own
# dispersion, annualized, rather than a window return divided by a volatility of another window.
#
# What stays local is specific to futures. Skip-month momentum runs from $t-252$ to $t-21$ and
# divides prices rather than subtracting returns, because returns compound. Yang-Zhang is carried
# beside the close-to-close estimator because an overnight gap in a futures contract is a real move
# that a close-to-close estimator cannot see. The variance ratio says whether the recent path
# trended or reverted, which is a statement about the regime rather than the level of risk.


# %%
def price_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Everything computed from one contract's own price history."""
    vol = WINDOWS["volatility"]
    horizons = WINDOWS["momentum"]
    vr = WINDOWS["variance_ratio"]
    df = momentum_volatility_block(
        bars,
        entity=ENTITY,
        price="adj_close",
        return_windows=horizons,
        volatility_windows=vol,
        periods_per_year=PERIODS_PER_YEAR,
    )
    held = pl.col("adj_close").shift(WINDOWS["skip_recent"]).over(ENTITY)
    start = pl.col("adj_close").shift(WINDOWS["skip_start"]).over(ENTITY)
    return df.with_columns(
        trailing_return("adj_close", WINDOWS["short_return"], ENTITY).alias("ret_5d"),
        (held / start.clip(lower_bound=EPS) - 1).alias("skip_month_mom"),
        (pl.col(f"vol_{vol[0]}d") / pl.col(f"vol_{vol[1]}d").clip(lower_bound=EPS))
        .clip(upper_bound=10.0)
        .alias("vol_ratio_short"),
        (pl.col(f"vol_{vol[1]}d") / pl.col(f"vol_{vol[2]}d").clip(lower_bound=EPS))
        .clip(upper_bound=10.0)
        .alias("vol_ratio_medium"),
        yang_zhang_volatility(
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            period=WINDOWS["yang_zhang"],
            annualize=True,
        )
        .over(ENTITY)
        .alias(f"vol_yz_{WINDOWS['yang_zhang']}d"),
        variance_ratio("adj_close", periods=[vr["horizon"]], window=vr["window"])[
            f"vr_{vr['horizon']}"
        ]
        .over(ENTITY)
        .alias(f"vr_{vr['window']}d"),
        *[
            pl.col(f"ret_{h}d").sign().cast(pl.Int32).alias(f"ts_mom_{h}d")
            for h in WINDOWS["trend_sign"]
        ],
    ).with_columns(
        (pl.col(f"ret_{a}d") - pl.col(f"ret_{b}d")).alias(f"mom_accel_{name}")
        for name, a, b in zip(("short", "medium", "long"), horizons[:-1], horizons[1:], strict=True)
    )


# %% [markdown]
# ### C.3 Trend and range
#
# Where a contract sits against its own moving averages and its own 52-week extremes. `rsi_14`
# comes from `ml4t.engineer.features.momentum`, which applies Wilder's recursive smoothing to
# prices; a simple moving average of the same gains and losses is a different oscillator under the
# same name, and a reader carrying a column name between chapters is carrying a claim.
#
# The library calls return NaN where their window has not filled, and polars treats NaN as a value
# rather than as missing: a rank over such a column puts every warmup row at the top of the
# cross-section. They are converted to nulls here, so the audit in D.2 and the percentiles in C.5
# see the same missingness as the rest of the matrix.


# %%
LIBRARY_NAN = ["rsi_14", "vol_yz_21d", "vr_63d"]


def trend_and_range(df: pl.DataFrame) -> pl.DataFrame:
    """Price against its own moving averages, its extremes, and a bounded oscillator."""
    close = pl.col("adj_close")
    return df.with_columns(
        *[
            (close / close.rolling_mean(p).over(ENTITY)).alias(f"ma_ratio_{p}")
            for p in WINDOWS["moving_average"]
        ],
        (close / close.rolling_max(WINDOWS["high"]).over(ENTITY)).alias("dist_from_52w_high"),
        (close / close.rolling_min(WINDOWS["low"]).over(ENTITY)).alias("dist_from_6m_low"),
        rsi("adj_close", period=WINDOWS["rsi"]).over(ENTITY).alias(f"rsi_{WINDOWS['rsi']}"),
    ).with_columns(pl.col(LIBRARY_NAN).fill_nan(None))


# %% [markdown]
# ### C.4 Calendar and season
#
# Month is encoded as a point on a circle rather than as an integer, so December sits next to
# January instead of eleven units away from it. The seasonal flag and the sector map are read from
# the configuration's own product groups, which is where the universe is defined; the roll-week
# flag marks the last calendar week of the month, when CME contracts of the monthly cycle are
# rolling and the front month is thinning out.


# %%
def calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """Where in the year, and in the roll cycle, each session sits."""
    angle = 2 * np.pi * pl.col("timestamp").dt.month().cast(pl.Float64) / 12
    sector = pl.col("product").replace_strict(SECTOR, default="unknown")
    return df.with_columns(
        angle.sin().alias("month_sin"),
        angle.cos().alias("month_cos"),
        (pl.col("timestamp").dt.ordinal_day().cast(pl.Float64) / 365.25).alias("day_of_year_norm"),
        pl.col("timestamp").dt.quarter().cast(pl.Float64).alias("quarter"),
        sector.is_in(SEASONAL).cast(pl.Float64).alias("is_seasonal_sector"),
        (pl.col("timestamp").dt.day() >= LEVEL["roll_proximity_day"])
        .cast(pl.Float64)
        .alias("roll_proximity"),
    )


# %% [markdown]
# ### C.5 Cross-sectional position and composites
#
# A long-short strategy can only act on relative standing, so eleven of the levels above are also
# carried as their percentile within the decision date and contract position. The rank is taken
# over one more than the count, which keeps the top product off the boundary and makes the mapping
# a percentile into $(0, 100)$ rather than a rank divided by its own maximum.
#
# Carry is additionally ranked within its own sector: a gold carry and a natural-gas carry differ
# by more than either differs from its own sector's median, so the unconditional ranking is
# dominated by which sector a product belongs to.
#
# The composites average percentiles already on one scale, and `carry_mom_composite` is where the
# thesis is stated - a product is attractive when the curve and the trend agree. `ls_signal` bands
# that score and is null, not zero, wherever the score is: a flat reading and no reading are
# different things, and a model handed zero for both cannot tell them apart.


# %%
def cross_sectional(df: pl.DataFrame) -> pl.DataFrame:
    """Percentiles within the decision date and the contract position."""
    ranked = df.with_columns(
        cross_sectional_percentile(source, WITHIN_DATE).alias(name)
        for source, name in RANKED.items()
    )
    sector = pl.col("product").replace_strict(SECTOR, default="unknown").alias("_sector")
    return (
        ranked.with_columns(sector)
        .with_columns(
            cross_sectional_percentile("carry_pct", [*WITHIN_DATE, "_sector"]).alias(
                "carry_rank_sector"
            )
        )
        .drop("_sector")
    )


# %%
def composite_features(df: pl.DataFrame) -> pl.DataFrame:
    """Scores built from the percentiles, on the same scale as them."""
    df = df.with_columns(
        (pl.sum_horizontal(cols, ignore_nulls=False) / len(cols)).alias(name)
        for name, cols in FEATURES["composites"].items()
    )
    score = pl.col("carry_mom_composite")
    return df.with_columns(
        ((pl.col("carry_rank") + pl.col("momentum_composite")) / 2).alias("carry_mom_composite")
    ).with_columns(
        (score / (pl.col("vol_rank") + 10)).alias("risk_adj_score"),
        (pl.col("carry_zscore_63d") * pl.col("momentum_composite")).alias("carry_mom_interaction"),
        pl.when(score.is_null())
        .then(None)
        .when(score > LEVEL["signal_long"])
        .then(1)
        .when(score < LEVEL["signal_short"])
        .then(-1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("ls_signal"),
    )


# %% [markdown]
# The five subsections compose into one function, which is what lets D.3 re-run the whole
# construction on a shorter panel and compare. Carry is joined to all three positions before
# anything ranks, because a percentile is a property of the cross-section it is taken over and the
# join decides which products are in it.


# %%
def build_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Carry, price, calendar, percentile and composite families, in dependency order."""
    return (
        bars.pipe(price_features)
        .pipe(trend_and_range)
        .pipe(calendar_features)
        .join(term_structure(bars), on=["product", "timestamp"], how="left")
        .pipe(cross_sectional)
        .pipe(composite_features)
    )


built = build_features(bars)
EXCLUDED = {*bars.columns, "log_return"}
feature_cols = sorted(c for c in built.columns if c not in EXCLUDED)
print(f"{len(built):,} rows carrying {len(feature_cols)} features")

# %% [markdown]
# ## D. The timing contract
#
# ### D.1 What each construction reads
#
# Four kinds of operation appear above. A **rolling** window - every return, volatility, moving
# average, extreme, oscillator and carry z-score - ends at its own row and reads a fixed number of
# sessions backward within one product and contract position. A **shift** reads exactly one earlier
# row of the same series. A **contemporaneous** relation - carry and curvature - reads three tenors
# of the same product on the same date and no other date at all. A **cross-sectional** statistic -
# the twelve percentiles - is taken with `.over(["timestamp", "position"])`, so it reads every
# product quoted at that position on that date and nothing dated before or after it.
#
# None of the four is fitted: no bound, scaler or encoder here has parameters estimated once and
# applied to every row. D.2 checks the windows; D.3 checks all four at once.
#
# ### D.2 Warmup
#
# A trailing window cannot produce a value until it has enough sessions to fill. The audit checks
# that length rather than describing it: a column carrying a value before its window could have
# filled is reading sessions that do not exist, and that is the failure it raises on.
#
# The two audits count on different frames, and the panel key forces it. A return is a statistic of
# one contract, counted in that contract's own sessions. Carry is a statistic of the product's
# curve, counted in product-dates: nine of the 30 products list a deferred contract later than
# their front month, and counting the carry z-score from that contract's first session would report
# a warmup that had elapsed before the contract existed.

# %%
warmup_audit(
    built,
    {
        "ret_252d": 252,
        "skip_month_mom": 252,
        "sharpe_252d": 252,
        "dist_from_52w_high": 252,
        "ma_ratio_200": 200,
        "vol_126d": 126,
        "vr_63d": 63,
        "vol_yz_21d": 21,
        "rsi_14": 14,
    },
    entity=ENTITY,
)

# %%
warmup_audit(
    term_structure(bars),
    {"carry_zscore_126d": 146, "carry_zscore_63d": 83, "carry_momentum_21d": 42, "carry_21d": 21},
    entity="product",
)

# %% [markdown]
# ### D.3 Withholding the holdout changes nothing
#
# Trailing, contemporaneous and within-date statistics share a property worth checking directly:
# recomputed on a panel that stops before the holdout, they reproduce the same values on the rows
# the two panels share. A parameter fitted over a whole column does not, because truncating the
# column moves the parameter and with it every row it was applied to. Comparing two builds tests
# every emitted column at once and does not depend on anyone having flagged the transform that
# fits. A value on one side against a null on the other counts as a difference.

# %%
seal = assert_values_agree(
    built.filter(pl.col("timestamp") < HOLDOUT_START),
    build_features(bars.filter(pl.col("timestamp") < HOLDOUT_START)),
    columns=feature_cols,
    keys=PANEL_KEY,
)
seal.filter(pl.col("column").is_in(["carry_zscore_126d", "mom_rank_252d", "carry_rank_sector"]))

# %% [markdown]
# ## E. Matrix assembly and coverage
#
# The panel key is `product` + `position` + `timestamp`. Everything the loader supplied is excluded
# - the two OHLC sets, volume, the roll multiplier and the session metadata - because a model handed
# a contemporaneous settlement price beside a label derived from the same series would be reading
# its own answer. `log_return` goes with them, as the intermediate the volatility family
# standardizes rather than a feature.
#
# One null policy is applied once: a row is kept when the two shortest-window carriers, the
# one-month return and the one-month volatility, have both warmed up. The longer families fill in
# above that point, which is what F1 shows.

# %%
features = (
    built.select([*PANEL_KEY, *feature_cols])
    .drop_nulls(subset=["ret_21d", "vol_21d"])
    .sort(PANEL_KEY)
)
assert features.select(PANEL_KEY).is_duplicated().sum() == 0, "duplicate panel key"
# The front month, which F3, F6, F7 and F8 read: carry is one value per product-date, and a
# cross-section taken over all three positions is an ordering of 90 contracts rather than the
# 30 a decision is taken over.
front = features.filter(pl.col("position") == 0)
assignment = assign_families(feature_cols, FAMILIES)
register_frame(FAMILIES, feature_cols).select(["family", "columns", "role", "representation"])

# %% [markdown] tags=["results"]
# The matrix carries **62 features** on **310,947 rows** across **30 products** at three contract
# positions, from **2011-02-01** to **2025-12-31**. The null policy dropped **1,912 rows**. Past the
# warmup boundary at **2011-12-21** the thinnest family in any month is **0.906** covered, which is
# momentum: a contract listed part-way through the sample has no one-year return yet.

# %%
# The warmup boundary is the register's own longest lookback counted forward from the panel's
# first session - a declared number, not a date read off the values.
sessions = bars["timestamp"].unique().sort()
WARMUP_END = sessions[max(f.lookback for f in FAMILIES)]
coverage = family_coverage(features, assignment, every="1mo")
floor = coverage.filter(pl.col("timestamp") >= WARMUP_END)
print(
    f"{len(feature_cols)} features, {len(features):,} rows, {features['product'].n_unique()} products"
)
print(f"{features['timestamp'].min()} to {features['timestamp'].max()}, warmup ends {WARMUP_END}")
print(f"{len(built) - len(features):,} rows dropped by the null policy")
print(
    f"thinnest family-month past warmup {min(floor[c].min() for c in set(assignment.values())):.3f}"
)

# %% [markdown]
# ### F1. Coverage through time
#
# Below the boundary the composites are empty by construction - a score built from percentiles of
# a one-year return cannot exist until a one-year return does - so the axis runs the full range
# rather than the top percent it would need for a matrix that is dense throughout.

# %%
plot_coverage_through_time(
    coverage,
    warmup_boundary=WARMUP_END,
    title="Long-window families fill in over the first year",
    subtitle="Monthly non-null share per feature family",
    alt=(
        "Line chart of non-null share by feature family by month, on a y-axis running from "
        "zero to one. The composite and cross-sectional families sit at zero until the end of "
        "2011 and then jump to near one at the marked warmup boundary. Every other family "
        "rises through 2011 and all of them then sit between about 0.9 and one for the rest of "
        "the sample, with momentum the lowest and slightly ragged throughout."
    ),
)

# %% [markdown]
# ### F4. The timing contract

# %%
plot_timing_contract(
    FAMILIES,
    bar_unit="settlement sessions",
    title="Every family reads prices that have already settled",
    subtitle="Register lookback per family; a gap at the right edge is a lag",
    alt=(
        "Horizontal bars, one per feature family, each extending leftward from the decision "
        "line by that family's lookback: 252 sessions for momentum, risk-adjusted momentum, "
        "trend and range, the cross-sectional percentiles and the composites, 146 for term "
        "structure, 126 for volatility and one for the calendar family. Every bar reaches the "
        "decision line, so none of them is drawn with a gap at its right-hand end."
    ),
)

# %% [markdown]
# ## F. What the features look like
#
# Four properties decide whether this matrix can be used at all: the scale each feature arrives on,
# whether the cross-section disagrees enough to rank on, how much of the set is one ordering under
# several names, and how long a value lasts. `05_evaluation` is where the matrix is tested fold by
# fold for whether any of it predicts.
#
# ### F2. Feature distributions
#
# The carry family is shown on the scale a reader would judge it: the raw annualized spread, its
# smoothed level, the change in it, how far it sits from its own history, and its two rankings.
# The same quantity looks completely different in level and in percentile form, which is the point
# of carrying both.

# %%
plot_feature_distributions(
    features,
    [
        "carry_pct",
        "carry_21d",
        "carry_momentum_21d",
        "carry_zscore_63d",
        "carry_rank",
        "carry_rank_sector",
    ],
    title="The carry level is sharply peaked where its percentile is flat",
    subtitle="Carry family across all product-sessions, display tails clipped",
    alt=(
        "Six histograms in two rows. The three level features - the raw annualized carry, its "
        "smoothed version and its one-month change - are all sharply peaked at zero with long "
        "thin tails to either side. The 63-session z-score below them is a broad, roughly "
        "symmetric bell spanning about minus three to three. The two percentiles are close to "
        "uniform across their range, the within-sector one reduced to a few tall spikes "
        "because a sector holds only four to six products to rank within."
    ),
)

# %% [markdown]
# ### F3. Cross-sectional dispersion through time
#
# A cross-sectional strategy needs the cross-section to disagree. On a date where the band narrows
# to nothing there is nothing to rank, whatever the average level of carry. Carry is one value per
# product-date, so this reads the front month alone: including all three positions would repeat
# every product three times and report a wider spread than the one a decision is actually taken
# over.

# %%
plot_cross_sectional_dispersion(
    front,
    "carry_pct",
    every="1mo",
    title="The cross-section of carry never collapses to one view",
    subtitle="Interdecile band of annualized front-month carry, by month",
    alt=(
        "Shaded band of the 10th to 90th percentile of annualized front-month carry by month, "
        "with the median drawn through it. The median stays close to zero throughout. The band "
        "is roughly plus or minus 0.2 wide in calm periods and never narrows to nothing; it "
        "reaches below minus 0.7 in 2020 and widens again through 2021 and 2022."
    ),
)

# %% [markdown]
# ### F5. Redundancy structure
#
# Clustering on the distance $1 - |\rho|$ groups features that carry the same ordering, whatever
# the sign. Above the cut two features are close enough that a linear model cannot separate their
# contributions. This states the clusters; choosing one representative from each needs a fold-aware
# criterion and belongs to `05_evaluation`.

# %%
CUT = 0.7
clusters = plot_redundancy_clusters(
    features,
    feature_cols,
    cut=CUT,
    title="One horizon is one ordering under several names",
    subtitle=r"Average linkage on $1 - |\rho_s|$, cut drawn at $|\rho_s| = 0.7$",
    alt=(
        "Dendrogram of every feature in the matrix. The clusters below the cut are horizon "
        "blocks that mix levels with their percentiles: the six-month return, its Sharpe, its "
        "sign and both of their percentiles join at very small distances, and the same happens "
        "at the one-year horizon and again for the carry level with its rank, its regime "
        "indicator and the long-short signal. The volatility windows form one cluster with the "
        "volatility percentile. The calendar features and the variance ratio attach only near "
        "the root, sharing an ordering with nothing else."
    ),
)

# %% [markdown] tags=["results"]
# Cutting the redundancy tree at $|\rho_s| = 0.7$ leaves **25 clusters** across the **62** columns,
# so well over half the matrix repeats an ordering another column already carries.

# %%
print(f"{len(set(clusters.values()))} clusters over {len(feature_cols)} features at cut {CUT}")

# %% [markdown]
# ### F6. Persistence and rank stability
#
# The right-hand panel compares the ordering across consecutive **rebalances**, which
# `config/setup.yaml` declares as `weekly_friday_close`. The autocorrelation on the left is of the
# feature, not of the return, and it runs to four decision cycles. A feature whose value has
# decayed before the next rebalance cannot support that cadence, however well it predicts on the
# day it is computed. It is estimated per product on pairs of dates exactly one lag apart and
# summarized by the median over products, with a bootstrap interval over products: a correlation
# pooled over every product-date pair would read high whenever products sit at different levels,
# whether or not any one of them persists.
#
# Both panels read the front month alone, for the same reason C.5 partitions every percentile by
# contract position. The right-hand panel ranks within a date, and a ranking taken over all three
# positions at once would be an ordering of 90 contracts - not the one a decision is taken over,
# and not the one the matrix carries.

# %%
DECISION_DATES = (
    features.group_by(pl.col("timestamp").dt.truncate("1w"))
    .agg(pl.col("timestamp").max().alias("decision"))["decision"]
    .sort()
    .to_list()
)

plot_persistence(
    front,
    ["carry_pct", "carry_zscore_63d", "ret_63d", "vol_21d", "rsi_14"],
    entity="product",
    max_lag=4 * DECISION_CYCLE,
    decision_dates=DECISION_DATES,
    title="Carry's ordering is the least stable across rebalances",
    subtitle=f"Front month; median over products to {4 * DECISION_CYCLE} sessions",
    alt=(
        "Two panels. On the left, autocorrelation against lag: all five features start near one "
        "and decay over twenty sessions. The six-month return and the one-month volatility fall "
        "slowest, to roughly 0.6, while the carry z-score falls fastest and reaches zero; the raw "
        "carry drops steeply over the first ten sessions and then flattens near 0.27, and the "
        "oscillator ends near 0.2. The bootstrap ribbon is widest around the raw carry and narrow "
        "elsewhere. On the right, the cross-sectional rank correlation between consecutive weekly "
        "rebalances puts the one-month volatility highest at nearly one and the raw carry lowest "
        "at roughly 0.6, with the oscillator, the six-month return and the carry z-score between."
    ),
)

# %% [markdown]
# ### F7. The carrier through time
#
# Two of the families side by side, for one product from each of four sectors, sampled on the
# decision dates rather than on every session - the strategy never looks at a Wednesday. The carry
# z-score is a distance from a product's own recent history, so it is bounded and pulled back to
# zero; the momentum composite is a standing among the other 29 products, so it can sit at one end
# of the range for years. A signal that oscillates and a signal that trends need different holding
# periods, which is why both are in the matrix.

# %%
KEY_PRODUCTS = {
    "CL": COLORS["blue"],
    "GC": COLORS["amber"],
    "ES": COLORS["positive"],
    "ZC": COLORS["copper"],
}
decisions = front.filter(pl.col("timestamp").is_in(DECISION_DATES)).sort("timestamp")
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Carry z-score against 63 sessions", "Momentum composite percentile"],
    vertical_spacing=0.1,
)
for product, color in KEY_PRODUCTS.items():
    series = decisions.filter(pl.col("product") == product)
    for row, column in ((1, "carry_zscore_63d"), (2, "momentum_composite")):
        fig.add_trace(
            go.Scatter(
                x=series["timestamp"].to_list(),
                y=series[column].to_list(),
                name=product,
                line=dict(width=1, color=color),
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], row=1, col=1)
fig.add_hline(y=50, line_dash="dash", line_color=COLORS["neutral"], row=2, col=1)
fig.update_layout(
    height=600, title_text="The carry z-score oscillates where the momentum percentile trends"
)
style_subplot_titles(fig)
fig.show()

# %% [markdown]
# ### F8. The curve on one date
#
# What the carrier looks like across the universe at a single decision, which is the object the
# strategy actually ranks. The date is the most recent one on which all 30 products quoted both of
# the contracts carry is measured between: a snapshot missing a third of the universe would show a
# cross-section no decision was ever taken over. The regime band declared in the configuration is
# what separates the three colours, and the vertical scale is what a single ranking has to absorb.

# %%
REGIME = {1: "backwardation", 0: "flat", -1: "contango"}
quoted = front.group_by("timestamp").agg(pl.col("carry_pct").is_not_null().sum().alias("n"))
snapshot_date = quoted.filter(pl.col("n") == front["product"].n_unique())["timestamp"].max()
snapshot = front.filter(pl.col("timestamp") == snapshot_date).with_columns(
    pl.col("carry_regime_num").replace_strict(REGIME).alias("Regime")
)
fig = px.bar(
    snapshot.sort("carry_pct").to_pandas(),
    x="product",
    y="carry_pct",
    color="Regime",
    color_discrete_map={
        "backwardation": COLORS["positive"],
        "flat": COLORS["neutral"],
        "contango": COLORS["negative"],
    },
    title="One date's cross-section splits evenly between the two regimes",
    labels={"carry_pct": "Annualized carry", "product": "Product"},
)
fig.update_layout(height=400)
fig.show()

# %% [markdown]
# ## G. Emit
#
# The parquet is written with a sidecar recording the digest of its values, its row count and key
# columns, and the digest of what it was built from. This stage reads no upstream case-study
# artifact, so the sidecar records the loaded settlement panel alone, restricted to the columns and
# window actually consumed - which is what answers "which market-data vintage produced these
# values". The digest is computed over content rather than file bytes, so row order and parquet
# metadata leave it alone and any feature value moves it. That is the property the registry's own
# hashes lack: a feature-set *name* reaches the registry, a feature-set *value* does not.

# %%
record = write_artifact(
    features,
    FEATURES_DIR / "financial.parquet",
    keys=PANEL_KEY,
    written_by="case_studies/cme_futures/03_financial_features.py",
    inputs={"load_cme_futures": value_digest(bars.select([*PANEL_KEY, *PRICES, "volume"]))},
)
print(f"Wrote {display_path(FEATURES_DIR / 'financial.parquet')}, digest {record['digest']}")

# %% [markdown]
# ## Key takeaways
#
# - **State the timing contract before writing the feature.** The register fixes each family's
#   lookback and lag in the configuration, and the warmup assertion, the timing figure and the
#   review a reader can run all read those numbers rather than re-deriving them from the code.
# - **Match the price series to the question.** A spread between two tenors needs the settled
#   prices and a return needs the roll-adjusted ones. Using one series for both measures roll
#   history and calls it curve shape.
# - **Rank inside the frame a decision is taken over.** For a term structure that is the date *and*
#   the contract position; a percentile taken across positions compares two different instruments.
# - **Test the seal by construction, not by inspection.** Rebuilding the panel with later dates
#   withheld and comparing values catches any transform that fits across the sample, including the
#   ones nobody thought to flag.
# - **Read the matrix before modelling it.** Distribution, dispersion, redundancy and decay each
#   rule out a use: a feature with no cross-sectional spread cannot rank, and one whose ordering
#   decays inside the rebalance cycle cannot be traded at that cadence.
#
# ### Known limitations
#
# - Carry is measured between the first two contract positions, which for most of this universe is
#   a one-month gap but for the quarterly financial contracts is three. The annualization treats
#   both as monthly, so the level is not comparable across sectors - which is the reason the
#   within-sector percentile is carried beside the unconditional one.
# - The roll-week flag is a calendar approximation, not a contract's own expiry. Products on
#   quarterly cycles do not roll every month, and the flag marks the last week of every month for
#   all of them.
# - Curvature needs a third tenor, so it is null wherever the panel carries only two.
# - Every feature here is a rule written in advance. `04_model_based_features` adds the features
#   that are themselves model outputs, where the rule is estimated from the data.
