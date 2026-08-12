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
# # US Equities Panel: Model-Based Features
#
# Every feature in [`03_financial_features`](03_financial_features.ipynb) is a function of
# past bars: hand it a row's history and it returns the same value whatever else the panel
# contains. A feature on this page is a function of *parameters estimated from* bars, so the
# estimation window is part of what the feature knows, and a parameter fitted once on the
# whole sample carries the whole sample into every row it touches - including the rows a
# model will later be scored on. That is the hazard this stage exists to show, and the
# discipline that removes it is to refit inside every walk-forward fold on that fold's
# training bars alone.
#
# Three transforms are built that way, each explained where it is used:
#
# 1. **A regime distance.** Recent months of market-wide return are compared against two
#    reference months learned from the training window, and each date is given how far it sits
#    from the nearer of them. Section 2.
# 2. **A fractionally differenced price.** A price level is differenced to a fractional order,
#    which keeps part of what the level knows where a return keeps none of it. The weights
#    follow from the order alone, so nothing here is estimated and the transform is the same in
#    every fold. Section 3.
# 3. **A conditional volatility.** Each of the most liquid stocks in a training window gets a
#    volatility model fitted on that window, run forward over the validation period without
#    being re-estimated; every other stock takes a market-level fit. Section 4.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Tell apart the two date ranges any fitted feature has - the range its parameters were
#   estimated from, and the range it produces values over - and keep the first one inside the
#   training window of the fold it belongs to
# - Draw the fold design as bars on a calendar and read off it whether any estimation window
#   crosses the date the test period begins
# - Chart how a model's fitted parameters move as the training window rolls forward, and use
#   that to decide how often the model is worth re-estimating
# - Measure what a differencing order costs in the memory it discards and buys in the
#   stationarity it gains, rather than adopting the number a library defaults to
# - Score how well a single column ranks stocks against their later returns, using only test
#   rows, correcting the standard error for the persistence of the series and the test's
#   threshold for the number of columns tried
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 9, Sections 9.1 (Stationarity), 9.3 (Volatility), 9.5 (Regimes). Assumes
# [`02_labels`](02_labels.ipynb) and [`03_financial_features`](03_financial_features.ipynb)
# have been run.
#
# Reads the adjusted daily panel through `load_us_equities()`, `config/setup.yaml` for the
# fold design and the holdout boundary, and the primary label file written by
# [`02_labels`](02_labels.ipynb) for the ranking check in Section 7. Writes
# `features/model_based.parquet`, which the model stages join to the stage-03 matrix on
# `(symbol, timestamp, fold)`, alongside a small companion file recording what was written -
# the digest sidecar Section 6 describes.

# %%
"""US Equities Panel: Model-Based Features."""

import warnings
from dataclasses import dataclass
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from numpy.typing import NDArray

warnings.filterwarnings("ignore")

from arch import arch_model
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series
from ml4t.diagnostic.splitters.calendar import TradingCalendar
from ml4t.engineer.features.fdiff import ffdiff, get_ffd_weights
from statsmodels.tsa.stattools import adfuller

from case_studies.utils.artifact_digest import read_digest, value_digest, write_artifact
from case_studies.utils.cv_window import modeling_fold_boundaries
from data import load_us_equities
from utils.artifact_specs import resolve_label_horizon
from utils.paths import display_path, get_case_study_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

CASE_DIR = get_case_study_dir("us_equities_panel")
FEATURES_DIR = CASE_DIR / "features"

# The eligibility screen, carried by 02_labels and 03_financial_features from the same three
# constants on the same columns, so all three stages screen one universe.
MIN_ADV_USD = 1_000_000
MIN_PRICE = 5.0
ADV_WINDOW = 21

# Transform parameters. These define the transforms rather than the strategy, so they are
# declared here; everything that defines the strategy is bound from setup.yaml below.
FFD_D = 0.4  # equity-class default; Section 3 measures what it costs and buys
FFD_D_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
FFD_THRESHOLD = 1e-5

WASSERSTEIN_WINDOW = 21  # one month of sessions per clustered window
WASSERSTEIN_OVERLAP = 5  # consecutive windows share a week, so a shift is seen more than once
N_CLUSTERS = 2  # risk-on vs risk-off

GARCH_TOP_N = 200

FDR_ALPHA = 0.05

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# %% [markdown]
# ### The four values a run can be given
#
# These four are the only ones a caller overrides, so they sit in their own cell where
# Papermill can reach them, and nothing below re-assigns them. What each decides:
#
# - **`START_DATE`** is the first session the price panel is read from. It has to match the
#   date `02_labels` ran from, because Section 1 asserts that the panel this notebook reads
#   digests to the value recorded against the label file Section 7 scores against.
# - **`MAX_FOLDS`** truncates the walk-forward design to its first *n* cross-validation folds
#   and keeps the holdout fold. Zero, the default, keeps all of them. A shortened run still
#   exercises every transform; it just fits each one fewer times.
# - **`XS_MIN_STOCKS`** is the narrowest cross-section a daily return distribution is
#   summarized from. The clustering in Section 2 reads the median of that distribution, and a
#   median over a handful of names is not a market. It belongs here rather than with the
#   transform constants above because it is a property of the panel rather than of the
#   transform: a run over fewer stocks has to lower it or every date is dropped and the
#   clustering has nothing to fit on.
# - **`GARCH_MIN_OBS`** is how many returns a stock must have inside a fold's training window
#   before that fold will fit it a GARCH model. Two years of daily bars is the floor because a
#   maximum-likelihood fit of three parameters on a shorter series returns estimates whose
#   standard errors swamp them. A run over a shorter history has to lower it or no stock
#   qualifies and every fit is skipped.
#
# `SEED` fixes the one random step in the notebook, the initialization of the Wasserstein
# clustering in Section 2.

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
START_DATE = "1990-01-01"
MAX_FOLDS = 0
XS_MIN_STOCKS = 50
GARCH_MIN_OBS = 504
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Configuration
#
# The fold design, the holdout boundary and the primary label come from `config/setup.yaml`.
# The label's horizon is what binds Section 7: an IC series scored on a one-session forward
# return needs its Newey-West lag set from that horizon, and the validation window it may be
# scored over ends one session before the holdout opens rather than on the holdout date.
#
# The horizon is stated in sessions and the buffer in calendar days because that is how the
# splitter takes them. A buffer of one day is the gap the walk-forward design leaves between
# the last training session of a fold and the first session it is scored on, so that the
# outcome of the last training decision is already known when the validation window opens.

# %%
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_HORIZON = int(resolve_label_horizon(CASE_STUDY_ID, PRIMARY_LABEL, SETUP).rstrip("Dd"))
LABEL_BUFFER = SETUP["labels"]["buffer"]
HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
END_DATE = str(SETUP["evaluation"]["holdout_end"])
CALENDAR = SETUP["evaluation"]["calendar"]

print(
    f"Section 7 scores against {PRIMARY_LABEL}, the return over the next {LABEL_HORIZON} "
    f"session(s), so its Newey-West lag is set from {LABEL_HORIZON}."
)
print(
    f"The walk-forward design leaves {LABEL_BUFFER} between a fold's last training session "
    "and the first session it is scored on."
)
print(
    f"Everything from {HOLDOUT_START} to {END_DATE} is held out: no parameter here is "
    "estimated from it, and no number here is measured on it."
)
print(
    f"A stock is eligible on a date when its printed close is above ${MIN_PRICE:.0f} and its "
    f"dollar volume has averaged above ${MIN_ADV_USD:,} over the previous {ADV_WINDOW} "
    "sessions - the screen 02_labels and 03_financial_features apply."
)

# %% [markdown]
# ## Why this panel is given regime and volatility features
#
# The strategy this case study builds ranks stocks cross-sectionally and holds the top names
# against the bottom ones. A ranking like that earns steadily for long stretches and then
# gives several years back in a few weeks, and the weeks it gives them back in are the ones
# where the market turns sharply after a decline - Daniel and Moskowitz (2016) call these
# momentum crashes and show they cluster where volatility is high and the market is
# rebounding. A model that only sees each stock's own price history has no way to tell those
# weeks apart from any other.
#
# So the three transforms fitted below each supply something a per-stock price feature cannot:
#
# - **Where the whole cross-section currently sits.** The Wasserstein clustering in Section 2
#   compares the recent month of market-wide returns against two reference months learned from
#   the training window, and reports how close the match is. Its useful output is the *distance*
#   rather than the state, because a crash happens while the market is between states.
# - **How turbulent each stock is right now.** The GARCH fit in Section 4 gives each stock a
#   conditional volatility that responds to its own recent moves, which is the quantity the
#   crash literature conditions on.
# - **A price level that is still usable as a regressor.** Fractional differencing in Section 3
#   keeps part of what the level of a price knows, which a return has thrown away entirely.
#
# None of the three is a trading rule. They are inputs a model in the later stages can
# condition on, and whether conditioning on them helps is a question for `05_evaluation` and
# the model notebooks, not for this page.

# %% [markdown]
# ## 1. Load the panel and screen it
#
# Two screens run here, and they are the ones
# [`02_labels`](02_labels.ipynb) and [`03_financial_features`](03_financial_features.ipynb)
# already run, rebuilt from the same constants on the same columns so that all three stages
# describe one universe.
#
# **Sessions are numbered first.** The archive carries a small number of stray prints on dates
# the exchange held no market. A date that was never open is not a date a position can be taken
# on, and `get_sessions` identifies them: a date that maps to itself is a session, and a stray
# print maps to a neighbouring one. Dropping them and numbering what is left gives a counter
# whose difference between two rows is a count of sessions rather than a count of rows. Every
# window on this page needs it - a variance recursion, a fractional-difference convolution and
# a rolling turnover average all read their input in order and treat consecutive elements as
# consecutive sessions.
#
# **Then eligibility**, on three conditions: a printed close above \$5, dollar volume
# `close * volume` averaging above \$1M over the previous month, and that month being an
# unbroken run of sessions rather than whatever twenty-one rows the stock happens to have. The
# first two legs read figures the tape carried on the day, so neither depends on a corporate
# action that had not happened yet, and Section B of [`02_labels`](02_labels.ipynb) derives why
# the adjusted close cannot serve for either. The third is what stops a stock returning from a
# halt qualifying on volume it traded before the halt.
#
# **Eligibility is applied only to what is emitted, never to what the transforms read.** On the
# eligible frame a per-stock window would count *eligible* rows, so a stock that falls below a
# threshold for two years and recovers would have its convolution and its variance recursion
# reach straight across the excursion as though those were consecutive sessions. Both run on
# the full session panel; the eligible frame decides only which rows leave this notebook.
#
# The digest of the panel read here has to equal the one [`02_labels`](02_labels.ipynb)
# recorded against the label file this notebook scores against in Section 7; the assertion
# below is what makes the two files comparable rather than merely both present.

# %%
raw_df = load_us_equities(start_date=START_DATE, end_date=END_DATE)

if raw_df.schema["timestamp"] == pl.Datetime:
    raw_df = raw_df.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))

raw_df = raw_df.sort(["symbol", "timestamp"])

MARKET_DATA_DIGEST = value_digest(raw_df, ["symbol", "timestamp", "close", "volume", "adj_close"])
LABEL_INPUT_DIGEST = read_digest(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")["inputs"][
    "market_data"
]
print(f"market_data digest: {MARKET_DATA_DIGEST}")
assert MARKET_DATA_DIGEST == LABEL_INPUT_DIGEST, (
    f"the labels were written against market_data {LABEL_INPUT_DIGEST} and this stage read "
    f"{MARKET_DATA_DIGEST}. Re-run 02_labels before scoring features against its output."
)

# %%
# The session counter, built exactly as 02_labels and 03_financial_features build it.
_dates = raw_df.select("timestamp").unique().sort("timestamp")
_settling_session = pl.Series(
    TradingCalendar(CALENDAR)
    .get_sessions(pd.DatetimeIndex(_dates["timestamp"].to_list(), tz="UTC"))
    .to_numpy()
).cast(pl.Date)
_sessions = (
    _dates.filter(_settling_session == pl.col("timestamp"))
    .with_row_index("session")
    .with_columns(pl.col("session").cast(pl.Int64))
)
_archive_rows = raw_df.height
raw_df = raw_df.join(_sessions, on="timestamp", how="inner").sort(["symbol", "timestamp"])
print(
    f"{_sessions.height:,} of {_dates.height:,} dates in the archive are {CALENDAR} sessions; "
    f"the other {_dates.height - _sessions.height} carry stray prints and take "
    f"{_archive_rows - raw_df.height:,} rows with them"
)

# %%
raw_df = raw_df.with_columns(
    (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("returns"),
    (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
)
raw_df = raw_df.with_columns(
    pl.col("dollar_volume").rolling_mean(ADV_WINDOW).over("symbol").alias("adv_21d"),
    (pl.col("session") - pl.col("session").shift(ADV_WINDOW - 1) == ADV_WINDOW - 1)
    .over("symbol")
    .alias("adv_covered"),
)

ELIGIBLE = pl.col("adv_covered") & (pl.col("close") > MIN_PRICE) & (pl.col("adv_21d") > MIN_ADV_USD)
df = raw_df.filter(ELIGIBLE)

print(
    f"{len(raw_df):,} session rows on {raw_df['symbol'].n_unique():,} symbols, "
    f"{raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}"
)
print(
    f"{len(df):,} of them on {df['symbol'].n_unique():,} symbols pass all three conditions and "
    "are eligible to be emitted"
)

# %% [markdown]
# Those two totals are sums over twenty-eight years, and what every transform below actually
# works with is one day's slice of the panel. The figure is that slice through time: how many
# stocks are eligible on each session, with the two thresholds that read the count drawn across
# it.
#
# It is worth looking at before anything is fitted, because three of the decisions on this page
# are decisions about that count. The clustering in Section 2 takes a median across the slice
# and skips any date holding fewer than `XS_MIN_STOCKS` names, so where that line sits relative
# to the curve says whether the threshold ever binds. The GARCH section fits the most liquid
# `GARCH_TOP_N` names and gives every other stock a market-level value, so the distance from
# that line up to the curve is how much of the panel that feature is a broadcast for. And the
# count rises for most of the sample before turning down, which is why two fold windows of the
# same length in sessions are not comparable in how many stocks they saw.

# %%
_coverage = df.group_by("timestamp").len().sort("timestamp")

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(
    _coverage["timestamp"],
    _coverage["len"],
    color=COLORS["blue"],
    lw=0.8,
    label="eligible on the session",
)
ax.axhline(
    GARCH_TOP_N,
    color=COLORS["copper"],
    ls="--",
    lw=1.0,
    label=f"{GARCH_TOP_N}: given their own volatility fit",
)
ax.axhline(
    XS_MIN_STOCKS,
    color=COLORS["neutral"],
    ls=":",
    lw=1.0,
    label=f"{XS_MIN_STOCKS}: below this a date is not summarized",
)
ax.set_ylim(0, None)
ax.set_xlabel("Date")
ax.set_ylabel("Eligible stocks")
ax.legend(frameon=False, fontsize=7, loc="upper left")
add_message_title(
    ax,
    "The panel these transforms fit on grows for two decades, then turns down",
    subtitle="Stocks passing the price and dollar-volume screen on each session",
)
show_with_alt(
    fig,
    "A single line counts the stocks eligible on each session across the sample. It climbs "
    "for about two decades, drops sharply in the 2008 crisis, recovers to its highest point "
    "and then declines over the final years. Two flat reference lines sit far below it: a "
    "dashed one for the number of stocks needed to fit a volatility model per stock, and a "
    "dotted one for the count below which a date is not summarized. The eligible-stock line "
    "stays above both throughout.",
)

# %% [markdown]
# ## 1b. The fold contract
#
# The folds are resolved here, before any model runs, because every fit below is defined
# relative to them and a transform that resolves its own boundaries afterwards has nothing
# to be checked against.
#
# **They are resolved from the label file, through the same call the model stages use.** A
# walk-forward splitter counts backward from the holdout boundary in rows of whatever frame it
# is handed, and it seals the end of each validation window by the horizon of the label being
# predicted. Both of those are properties of the label file, not of the price panel: a price
# frame that differs from it by even a handful of dates yields windows that carry the same fold
# numbers and cover different spans. Since a fold id is the key this artifact is joined on
# downstream, that disagreement would be invisible and wrong. `modeling_fold_boundaries` reads
# the label file's own date index and its own configured buffer and horizon, and it is what
# `load_modeling_dataset` calls on the other side of the join, so the two agree by construction
# rather than by luck.
#
# **Both ends of a window are inclusive**, which is how `validate_temporal_fold_coverage`
# reads them downstream: `train_end` is the last session a fold's parameters may be estimated
# from, and `test_end` is the last session it emits a value for.
#
# A holdout fold is appended and its features **are** emitted: the transforms here are
# unsupervised, they read prices and never labels, so a parameter set estimated entirely
# before `holdout_start` may be run forward to produce filtered values *for* holdout dates.
# The model stages need those rows to score the holdout once. What may not happen is a fit
# that reads a holdout bar, and the assertion below is what rules that out.
#
# The cross-validation folds each carry the rolling ten-year training window `setup.yaml`
# declares. The holdout fold is given the whole pre-holdout sample instead, because it is
# fitted once and there is no later fold whose comparability a shorter window would preserve.

# %%
holdout_start = date.fromisoformat(HOLDOUT_START)
holdout_end = date.fromisoformat(END_DATE)

# The trading calendar the folds are counted on, taken from the label file itself.
SESSIONS = sorted(
    pl.read_parquet(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")["timestamp"]
    .unique()
    .to_list()
)

splits = modeling_fold_boundaries(CASE_STUDY_ID, PRIMARY_LABEL)
n_cv_folds = len(splits)

folds = [
    {
        "fold": s["fold"],
        "is_holdout": False,
        "train_start": s["train_start"],
        "train_end": s["train_end"],
        "test_start": s["val_start"],
        "test_end": s["val_end"],
    }
    for s in splits
]

folds.append(
    {
        "fold": n_cv_folds,
        "is_holdout": True,
        "train_start": folds[-1]["train_start"],
        "train_end": max(d for d in SESSIONS if d < holdout_start),
        "test_start": holdout_start,
        "test_end": holdout_end,
    }
)

if MAX_FOLDS > 0:
    folds = [f for f in folds if not f["is_holdout"]][:MAX_FOLDS] + [
        f for f in folds if f["is_holdout"]
    ]

print(f"Walk-forward folds ({len(folds)} total, {n_cv_folds} cross-validation + 1 holdout):")
for f in folds:
    tag = " [HOLDOUT]" if f["is_holdout"] else ""
    print(
        f"  Fold {f['fold']}{tag}: fitted on {f['train_start']} to {f['train_end']}, "
        f"emitted through {f['test_start']} to {f['test_end']}"
    )

# The one condition the whole stage rests on, asserted rather than described: a fold whose
# training span crept past the boundary would still produce features and still print a table.
for f in folds:
    assert f["train_end"] < holdout_start, (
        f"fold {f['fold']} fits parameters on bars through {f['train_end']}, which is inside "
        f"the holdout opening {holdout_start}"
    )
    if not f["is_holdout"]:
        assert f["test_end"] < holdout_start, (
            f"cross-validation fold {f['fold']} emits rows through {f['test_end']}, past "
            f"{holdout_start}"
        )
print(f"  every fitting window ends before {holdout_start}, and only fold {n_cv_folds} emits")

# %% [markdown]
# The figure is the fold contract itself. Each row is one fold: the filled bar is the span
# the parameters come from, the open bar is the span they are run forward over, and the
# dashed rule is the date the holdout opens. What the reader should be able to see is that no
# filled bar crosses the rule, and that the one open bar to the right of it is the holdout
# fold's.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
for row, f in enumerate(folds):
    tr0, tr1 = f["train_start"], f["train_end"]
    te0, te1 = f["test_start"], f["test_end"]
    ax.barh(row, (tr1 - tr0).days, left=tr0, height=0.62, color=COLORS["blue"], alpha=0.85)
    ax.barh(
        row,
        (te1 - te0).days,
        left=te0,
        height=0.62,
        facecolor="none",
        edgecolor=COLORS["copper"] if f["is_holdout"] else COLORS["neutral"],
        linewidth=1.2,
    )
ax.axvline(holdout_start, color=COLORS["copper"], ls="--", lw=1.4)
ax.set_yticks(range(len(folds)))
ax.set_yticklabels([f"{f['fold']}{'  H' if f['is_holdout'] else ''}" for f in folds], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Date")
ax.set_ylabel("Fold")
add_message_title(
    ax,
    "No fitting window reaches past the date the holdout opens",
    subtitle="Filled: bars the parameters come from. Outlined: bars they run forward over",
)
show_with_alt(
    fig,
    "One horizontal row per fold, fold zero at the top and the holdout row at the bottom. "
    "Each row is a filled bar for the window the parameters are estimated on, followed by a "
    "short outlined bar for the window they are then run forward over. The pairs step later "
    "in time as the fold number falls. A dashed vertical line marks the date the holdout "
    "opens, and no filled bar crosses it; only the bottom row extends past it.",
)

# %% [markdown]
# ## 2. Wasserstein regime distance
#
# At each date the median return is taken across every eligible stock trading that day. That
# one number per date is how the centre of the whole cross-section moves, and it is the series
# everything in this section reads.
#
# The method compares one recent month of that series against reference months learned from
# the training window, and it needs a way to say how far apart two months are. Two months of
# returns are two collections of twenty-one numbers, and the natural comparison is not
# value-by-value in date order - the same month reordered is the same market - but
# distribution against distribution. The **Wasserstein distance** measures exactly that: sort
# both collections, pair the smallest with the smallest and the largest with the largest, and
# average how far each pair has to move. It answers "how much return would have to be shifted,
# and how far, to turn one month into the other".
#
# With a distance in hand, ordinary k-means applies. **k-means** repeatedly assigns each
# window to its nearest of $k$ reference windows and then recomputes each reference as the
# centre of the windows assigned to it, until the references stop moving. Those references are
# called **centroids**, and with $k=2$ the two the algorithm settles on separate the calm,
# mildly positive months from the falling, turbulent ones. Two states is the coarsest split
# that can express that distinction, and it is the one the momentum-crash literature works in.
#
# The centroids are fitted on each fold's training window and then held fixed while every
# window in that fold, training and validation alike, is scored against them. Every stock
# carries the same value on a date, because the series being clustered is market-wide.


# %%
@dataclass(frozen=True)
class LiftedStream:
    """Overlapping windows of cross-sectional return distributions."""

    segments: FloatArray  # (n_segments, window_len)
    sorted_segments: FloatArray  # Sorted per window
    starts: IntArray  # Start indices
    window_len: int
    step: int


def lift_stream(
    returns: FloatArray,
    window_len: int,
    overlap: int,
) -> LiftedStream:
    """Lift a 1D return stream into overlapping windows."""
    step = window_len - overlap
    windows_view = np.lib.stride_tricks.sliding_window_view(returns, window_shape=window_len)
    windows_view = windows_view[::step]
    segments = np.ascontiguousarray(windows_view, dtype=np.float64)
    sorted_segments = np.sort(segments, axis=1)
    starts = np.arange(0, segments.shape[0] * step, step, dtype=np.int64)

    return LiftedStream(
        segments=segments,
        sorted_segments=sorted_segments,
        starts=starts,
        window_len=window_len,
        step=step,
    )


# %%
def wasserstein_distance_1d(sorted_a: FloatArray, sorted_b: FloatArray, p: float = 1.0) -> float:
    """1D p-Wasserstein distance between equal-weight empirical measures."""
    diff_p = np.abs(sorted_a - sorted_b) ** p
    return float(diff_p.mean() ** (1.0 / p))


def wasserstein_barycenter_1d(sorted_members: FloatArray, p: float = 1.0) -> FloatArray:
    """Wasserstein barycenter: median (p=1) or mean (p=2) of sorted atoms."""
    if p == 1.0:
        return np.median(sorted_members, axis=0).astype(np.float64)
    return sorted_members.mean(axis=0).astype(np.float64)


# %%
def fit_wasserstein_kmeans(
    sorted_segments: FloatArray,
    n_clusters: int = 2,
    max_iter: int = 50,
    n_init: int = 5,
    random_state: int = 42,
) -> tuple[IntArray, FloatArray]:
    """Fit Wasserstein k-means on sorted 1D segments.

    Returns (labels, centroids).
    """
    rng = np.random.default_rng(random_state)
    n_samples = sorted_segments.shape[0]
    best_labels = None
    best_centroids = None
    best_inertia = float("inf")

    for _ in range(n_init):
        # Random initialization
        idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centroids = sorted_segments[idx].copy()

        for _ in range(max_iter):
            # Assignment: compute distance to each centroid
            dists = np.zeros((n_samples, n_clusters))
            for k in range(n_clusters):
                diff = np.abs(sorted_segments - centroids[k][None, :])
                dists[:, k] = diff.mean(axis=1)

            labels = dists.argmin(axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                members = sorted_segments[labels == k]
                if len(members) > 0:
                    new_centroids[k] = wasserstein_barycenter_1d(members, p=1.0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        inertia = sum(dists[i, labels[i]] for i in range(n_samples))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids


# %% [markdown]
# ### The series the clustering reads
#
# One median per date, over the eligible stocks that traded that date. Dates whose
# cross-section is thinner than `XS_MIN_STOCKS` are dropped rather than summarized, because a
# median over a handful of names describes those names and not the market.

# %%
xs_stats = (
    df.filter(pl.col("returns").is_not_null())
    .group_by("timestamp")
    .agg(
        pl.col("returns").median().alias("xs_median_ret"),
        pl.col("returns").count().alias("n_stocks"),
    )
    .sort("timestamp")
    .filter(pl.col("n_stocks") >= XS_MIN_STOCKS)
)

market_ret = xs_stats["xs_median_ret"].to_numpy()
dates = xs_stats["timestamp"].to_list()

print(
    f"{len(xs_stats):,} dates carry a cross-section of at least {XS_MIN_STOCKS} eligible "
    f"stocks and are summarized; the median date carries "
    f"{int(xs_stats['n_stocks'].median()):,}"
)

# %% [markdown]
# ### Fitting the centroids inside each fold
#
# The centroids come from the fold's training window and are then held fixed, so the reference
# each window is scored against was learned from bars that close before the window opens. What
# the loop emits per date is the assigned cluster, the distance to the nearer and the farther
# centroid, their ratio, and how differently the window's best and worst days sit against the
# centroid it matched.
#
# k-means labels are arbitrary - which of the two states the algorithm happens to call zero
# depends on where it started - so after each fit the two are reordered by their mean, and
# state zero is always the lower-return one. Without that step a downstream model would see the
# same market condition under one number in one fold and the other number in the next.


# %%
def assign_regime_features(
    market_ret: FloatArray,
    dates_list: list,
    centroids: FloatArray,
    start_idx: int,
    end_idx: int,
) -> list[dict]:
    """Assign regime labels for dates[start_idx:end_idx] given fitted centroids."""
    results = []
    for t in range(start_idx, end_idx):
        if t < WASSERSTEIN_WINDOW:
            continue

        recent_window = market_ret[t - WASSERSTEIN_WINDOW : t]
        recent_sorted = np.sort(recent_window)

        dists = [
            wasserstein_distance_1d(recent_sorted, centroids[k]) for k in range(len(centroids))
        ]

        cluster = int(np.argmin(dists))
        min_dist, max_dist = min(dists), max(dists)

        # Tail divergence: right-tail vs left-tail distance to nearest centroid
        tail_div = float(
            np.mean(np.abs(recent_sorted[-5:] - centroids[cluster][-5:]))
            - np.mean(np.abs(recent_sorted[:5] - centroids[cluster][:5]))
        )

        results.append(
            {
                "timestamp": dates_list[t],
                "wass_cluster": cluster,
                "wass_dist_min": min_dist,
                "wass_dist_max": max_dist,
                "wass_dist_ratio": min_dist / (max_dist + 1e-10),
                "wass_tail_div": tail_div,
            }
        )
    return results


# %%
step = WASSERSTEIN_WINDOW - WASSERSTEIN_OVERLAP
min_length = WASSERSTEIN_WINDOW + (2 * N_CLUSTERS - 1) * step

wasserstein_all_folds = []
wass_centroid_rows = []  # per-fold centroid summary, read by the stability figure

print(
    f"Clustering {WASSERSTEIN_WINDOW}-session windows overlapping by {WASSERSTEIN_OVERLAP} "
    f"into {N_CLUSTERS} states, once per fold"
)

for fold in folds:
    fold_idx = fold["fold"]
    train_start_date = fold["train_start"]
    train_end_date = fold["train_end"]
    test_end_date = fold["test_end"]

    # Both window ends are inclusive, matching what the fold table states and what the
    # downstream coverage check reads.
    train_indices = [i for i, d in enumerate(dates) if train_start_date <= d <= train_end_date]
    full_indices = [i for i, d in enumerate(dates) if train_start_date <= d <= test_end_date]

    if len(train_indices) < min_length:
        print(
            f"  Fold {fold_idx}: insufficient training data ({len(train_indices)} < {min_length})"
        )
        continue

    # Fit centroids on training data only
    train_ret = market_ret[train_indices[0] : train_indices[-1] + 1]
    lifted = lift_stream(train_ret, WASSERSTEIN_WINDOW, WASSERSTEIN_OVERLAP)
    _, centroids = fit_wasserstein_kmeans(
        lifted.sorted_segments, n_clusters=N_CLUSTERS, random_state=SEED
    )

    # State 0 is the lower-return one in every fold.
    sort_idx = np.argsort([c.mean() for c in centroids])
    centroids = centroids[sort_idx]

    wass_centroid_rows.append(
        {
            "fold": fold_idx,
            "stress_centroid_mean": float(centroids[0].mean()),
            "normal_centroid_mean": float(centroids[-1].mean()),
            "centroid_separation": float(np.abs(centroids[-1] - centroids[0]).mean()),
        }
    )

    # Assign features for the full train+test period
    fold_features = []
    if full_indices:
        fold_features = assign_regime_features(
            market_ret,
            dates,
            centroids,
            start_idx=full_indices[0],
            end_idx=full_indices[-1] + 1,
        )
        for row in fold_features:
            row["fold"] = fold_idx
        wasserstein_all_folds.extend(fold_features)

    tag = " [HOLDOUT]" if fold["is_holdout"] else ""
    print(
        f"  Fold {fold_idx}{tag}: fitted on {len(train_indices):,} dates, assigned "
        f"{len(fold_features):,}"
    )

wass_df = pl.DataFrame(wasserstein_all_folds) if wasserstein_all_folds else pl.DataFrame()
n_wass_folds = wass_df["fold"].n_unique() if "fold" in wass_df.columns else 0
print(f"\n{len(wass_df):,} date-fold assignments across {n_wass_folds} folds")
if len(wass_df) > 0:
    cluster_counts = wass_df.group_by("wass_cluster").len().sort("wass_cluster")
    for row in cluster_counts.iter_rows(named=True):
        state = "lower-return" if row["wass_cluster"] == 0 else "higher-return"
        print(f"  state {row['wass_cluster']} ({state}): {row['len']:,}")

# %% [markdown]
# ### What the clustering inferred, on validation dates
#
# The figure draws the quantity the feature actually carries, over the dates it would be
# used on. Each fold contributes only its validation span, so what is plotted is a chain of
# out-of-sample assignments from sixteen different fits rather than one fit's view of the
# whole sample - which is the object a model downstream receives, and the one an illustrative
# full-sample fit would misrepresent.
#
# The line is the trailing cross-sectional median return the assignment reads; the strip
# below it marks the dates assigned to the low-return centroid. Nothing in the fitting
# procedure required those dates to be the market's stressed ones.
#
# **What the lower panel shows is why the state number is the weaker of the two outputs.**
# State zero is whichever centroid has the lower mean *in that fold's training window*. That
# fixes the arbitrariness of k-means labelling within a fold; it does not make the number mean
# the same thing across folds. A fold trained through the 2008 decline and a fold trained on
# the recovery that followed put their lower-return centroid in quite different places, so
# state zero in one and state zero in the next describe different markets under one number.
# The visible consequence is that the share in state zero drains away over the later folds
# while the line above it goes on doing what it always did. `wass_dist_ratio` does not have
# this problem: it is a distance, it is comparable within the fold that produced it, and
# reading it needs no knowledge of which centroid won.
#
# The assignment gets its own panel and is aggregated to a monthly share rather than drawn as
# a daily strip. Sixteen years of daily flags give each session a fraction of a pixel, isolated
# days vanish, and the reader concludes the state stopped occurring when it did not.

# %%
_val_regime = pl.concat(
    [
        wass_df.filter(
            (pl.col("fold") == f["fold"])
            & (pl.col("timestamp") >= f["test_start"])
            & (pl.col("timestamp") <= f["test_end"])
        )
        for f in folds
        if not f["is_holdout"]
    ]
).sort("timestamp")
_val_ret = xs_stats.join(_val_regime.select("timestamp", "wass_cluster"), on="timestamp").sort(
    "timestamp"
)
_smoothed = _val_ret.select(
    "timestamp",
    pl.col("xs_median_ret").rolling_mean(WASSERSTEIN_WINDOW).alias("trailing"),
    "wass_cluster",
).drop_nulls()

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=FIGSIZE["single"],
    sharex=True,
    height_ratios=[3, 1],
    gridspec_kw={"hspace": 0.22},
)
ax1.plot(_smoothed["timestamp"], _smoothed["trailing"], color=COLORS["blue"], lw=0.8)
ax1.axhline(0, color=COLORS["neutral"], lw=0.7)
ax1.set_ylabel("Trailing median return", fontsize=8)
ax1.locator_params(axis="y", nbins=4)
_monthly = (
    _smoothed.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
    .group_by("month")
    .agg((pl.col("wass_cluster") == 0).mean().alias("share"))
    .sort("month")
)
ax2.fill_between(_monthly["month"], 0, _monthly["share"], color=COLORS["copper"], lw=0, step="mid")
ax2.set_ylim(0, 1)
ax2.set_yticks([0, 1])
ax2.set_ylabel("Share in the\nlower state", fontsize=7)
ax2.set_xlabel("Date")
add_message_title(
    ax1,
    "The lower-return state drains away while the series does not change",
    subtitle="Validation dates only. Below: monthly share assigned to that state",
)
show_with_alt(
    fig,
    "Two stacked panels sharing a date axis over the validation window. The upper panel is a "
    "noisy trailing median return oscillating around zero, with its largest excursions in "
    "2008 and 2009 and no visible change in character before or after. The lower panel is a "
    "bar chart of the monthly share of dates assigned to the lower-return state; the bars "
    "are frequent and often reach the top of the panel in the earlier years, become sparse "
    "after 2010, and stop entirely for the last few years.",
)

_shaded = _smoothed.filter(pl.col("wass_cluster") == 0)
_runs = _smoothed.with_columns(
    (pl.col("wass_cluster").diff().fill_null(1) != 0).cum_sum().alias("run")
)
_run_lengths = _runs.filter(pl.col("wass_cluster") == 0).group_by("run").len()["len"]
print(
    f"validation dates {_smoothed.height:,}, assigned to the lower-return state "
    f"{_shaded.height:,} ({_shaded.height / _smoothed.height:.0%}); mean trailing return "
    f"{_shaded['trailing'].mean():+.5f} in that state against "
    f"{_smoothed.filter(pl.col('wass_cluster') == 1)['trailing'].mean():+.5f} in the other"
)
print(
    f"  {_run_lengths.len():,} runs, median {_run_lengths.median():.0f} sessions and longest "
    f"{_run_lengths.max():,}; first assigned {_shaded['timestamp'].min()}, last "
    f"{_shaded['timestamp'].max()}, and the validation window runs to "
    f"{_smoothed['timestamp'].max()}"
)

# %% [markdown]
# `wass_dist_ratio` is the second thing the clustering yields: the distance to the nearer
# centroid divided by the distance to the farther one. A window sitting squarely inside one
# state drives it toward zero and a window equidistant from both drives it toward one, so the
# feature carries how *certain* the match is rather than which state it picked. That is the
# part a momentum model needs, because momentum crashes fall at the transitions rather than
# inside either state.
#
# The cost of clustering the median and nothing else is worth stating plainly: this reads a
# shift in the centre of the cross-section, and a market that keeps its centre while its tails
# widen looks unchanged to it. Reaching that would mean clustering quantile vectors rather than
# a scalar, which is a different transform and not a tuning of this one.

# %% [markdown]
# ## 3. Fractional differencing
#
# A log price is not stationary and a log return has thrown away everything the level knew.
# Fractional differencing (Hosking 1981; Lopez de Prado 2018) takes the difference to a
# non-integer order $d$, which puts a dial between the two: at $d=0$ the series is the level
# and at $d=1$ it is the first difference, and every value in between trades some memory for
# some stationarity. `FFD_D` is the equity-class default this notebook uses.
#
# **The default is measured here rather than quoted.** The cell below runs the whole grid
# `FFD_D_GRID` on a sample of stocks and reports, for each order, the correlation between
# the differenced series and the original log price - how much of the level's memory
# is retained - against the share of sampled stocks whose augmented Dickey-Fuller test rejects
# a unit root. Those are the two quantities the choice trades off, and neither is knowable
# without running it.
#
# **Nothing here is estimated, so nothing is refitted per fold.** The FFD weights are a
# closed-form function of $d$ and of the truncation threshold, so the transform is identical
# in every fold and carries no estimation window at all. It is computed once and the fold
# column is attached during assembly in Section 5. That makes it the useful contrast for the
# section either side of it: the hazard this stage is about is *estimation*, not
# transformation, and a transform with no parameters has none of it.


# %%
def apply_ffd_per_symbol(
    data: pl.DataFrame, d: float = FFD_D, threshold: float = FFD_THRESHOLD
) -> pl.DataFrame:
    """Apply fractional differencing to log prices per symbol.

    Returns DataFrame with (symbol, date, ffd_log_price, ffd_log_volume).
    """
    results = []
    by_symbol = data.sort(["symbol", "timestamp"]).partition_by("symbol", as_dict=True)

    n_success = 0
    n_fail = 0

    for (sym,) in sorted(by_symbol):
        sym_data = by_symbol[(sym,)]

        if len(sym_data) < 100:
            n_fail += 1
            continue

        log_price = sym_data["adj_close"].log()
        # Floor volume at 1 to avoid log(0) = -inf
        log_vol = sym_data["adj_volume"].clip(lower_bound=1).log()

        try:
            ffd_price = ffdiff(log_price, d=d, threshold=threshold)
            ffd_vol = ffdiff(log_vol, d=d, threshold=threshold)

            sym_result = pl.DataFrame(
                {
                    "symbol": [sym] * len(sym_data),
                    "timestamp": sym_data["timestamp"],
                    "ffd_log_price": ffd_price,
                    "ffd_log_volume": ffd_vol,
                }
            ).drop_nulls()

            if len(sym_result) > 0:
                results.append(sym_result)
                n_success += 1
        except Exception:
            n_fail += 1

    print(f"  FFD: {n_success} symbols succeeded, {n_fail} failed/skipped")
    return pl.concat(results) if results else pl.DataFrame()


# %% [markdown]
# The sweep runs on a sample of stocks - every symbol with a long enough eligible history,
# taken at a fixed stride so the sample is not the alphabet's first few hundred names. The
# augmented Dickey-Fuller test asks whether a series has a unit root, which is the formal
# version of "wanders without returning"; what is reported is the *share* of sampled stocks
# whose test rejects that, because a single stock's test says very little and the question is
# whether the order works across the panel.
#
# **The sweep stops at the holdout boundary, on both counts.** It is a measurement that argues
# for a setting, so it is a development-time decision, and a development-time decision may not
# read a held-out bar. That governs which stocks it samples as much as which bars it reads: a
# sample drawn on history-length over the whole panel would let a stock's post-2016 record
# decide whether it is in the sample at all.

# %%
_ffd_dev = raw_df.filter(pl.col("timestamp") < holdout_start)
_ffd_symbols = (
    df.filter(pl.col("timestamp") < holdout_start)
    .group_by("symbol")
    .len()
    .filter(pl.col("len") >= 2000)
    .sort("symbol")["symbol"]
    .to_list()
)
_ffd_sample = _ffd_symbols[:: max(1, len(_ffd_symbols) // 120)][:120]
_ffd_panel = _ffd_dev.filter(pl.col("symbol").is_in(_ffd_sample)).sort(["symbol", "timestamp"])
_ffd_by_symbol = _ffd_panel.partition_by("symbol", as_dict=True)

grid_rows = []
for d in FFD_D_GRID:
    corrs, rejects = [], []
    for key in sorted(_ffd_by_symbol):
        _lp = _ffd_by_symbol[key]["adj_close"].log().drop_nulls()
        if len(_lp) < 500:
            continue
        _fd = ffdiff(_lp, d=d, threshold=FFD_THRESHOLD)
        _pair = pl.DataFrame({"level": _lp, "ffd": _fd}).drop_nulls()
        if _pair.height < 500 or _pair["ffd"].std() == 0:
            continue
        corrs.append(abs(float(np.corrcoef(_pair["level"], _pair["ffd"])[0, 1])))
        rejects.append(adfuller(_pair["ffd"].to_numpy(), autolag="AIC")[1] < FDR_ALPHA)
    grid_rows.append(
        {
            "d": d,
            "memory": float(np.mean(corrs)),
            "stationary_share": float(np.mean(rejects)),
            "n_symbols": len(corrs),
        }
    )

ffd_grid = pl.DataFrame(grid_rows)
print(
    f"{len(FFD_D_GRID)} differencing orders, each measured on the same "
    f"{ffd_grid['n_symbols'].max()} sampled stocks, on bars before {holdout_start}"
)
display(ffd_grid)

# %% [markdown]
# The two curves cross, and where they cross is the whole argument for a fractional order.
# Memory falls with $d$ and the share of stocks that pass the stationarity test rises with
# it; the first difference sits at the right-hand end, stationary and remembering nothing of
# the level.

# %%
_chosen = ffd_grid.filter(pl.col("d") == FFD_D).row(0, named=True)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(ffd_grid["d"], ffd_grid["memory"], color=COLORS["blue"], marker="o", ms=4, label="memory")
ax.plot(
    ffd_grid["d"],
    ffd_grid["stationary_share"],
    color=COLORS["copper"],
    marker="s",
    ms=4,
    label="share passing ADF",
)
ax.axvline(FFD_D, color=COLORS["neutral"], ls="--", lw=1.2)
ax.set_xlabel("Differencing order $d$")
ax.set_ylabel("Correlation with the log level / share of stocks")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8, loc="center right")
add_message_title(
    ax,
    "A fractional order keeps memory the first difference throws away",
    subtitle="Correlation with the log level, and the share of stocks rejecting a unit root",
)
show_with_alt(
    fig,
    "Two curves against the differencing order on the horizontal axis. One falls "
    "monotonically from one at order zero to near zero at order one, the correlation the "
    "series keeps with its own log level. The other rises from near zero to one and then "
    "runs flat, the share of stocks whose unit root is rejected. A dashed vertical line "
    "marks the order chosen; it stands where the rising curve has just reached its ceiling "
    "and the falling curve still retains a substantial part of its height.",
)

print(
    f"at d={FFD_D}: memory {_chosen['memory']:.3f}, {_chosen['stationary_share']:.1%} of "
    f"{_chosen['n_symbols']} sampled stocks reject a unit root | "
    f"at d={ffd_grid['d'].max()}: memory "
    f"{ffd_grid.filter(pl.col('d') == ffd_grid['d'].max())['memory'][0]:.3f}, "
    f"{ffd_grid.filter(pl.col('d') == ffd_grid['d'].max())['stationary_share'][0]:.1%}"
)
print(
    f"  the weight vector at d={FFD_D} truncates at "
    f"{len(get_ffd_weights(FFD_D, threshold=FFD_THRESHOLD))} lags"
)

# %% [markdown]
# ### Apply the transform to the panel
#
# On the complete price series per stock, for the reason Section 1 states: the weight vector
# reaches back hundreds of sessions, and on the screened frame those would be eligible rows
# rather than sessions.

# %%
print("Computing fractional differencing features...")
ffd_df = apply_ffd_per_symbol(raw_df)
print(f"FFD features: {len(ffd_df):,} rows, {ffd_df['symbol'].n_unique()} symbols")

# %% [markdown]
# ## 4. Per-Fold GARCH Conditional Volatility
#
# This is the section the stage is really about. A GARCH conditional volatility is not a
# function of a stock's past returns alone - it is a function of $(\omega, \alpha, \beta)$,
# and those come from a maximum-likelihood fit over some window. Fit them once over
# everything and every row's volatility knows the whole sample. So per fold:
#
# 1. Rank the eligible stocks of the training window by liquidity and take the top
#    `GARCH_TOP_N`, because a fit on a thin name is unstable and the choice of *which* stocks
#    to fit is itself an estimate that must not read past the boundary.
# 2. Fit GARCH(1,1) by maximum likelihood on that fold's training returns, per symbol.
# 3. Run the variance recursion forward over training and validation with `model.fix()`,
#    which applies the fitted parameters without re-estimating them. This is the distinction
#    between filtered and smoothed inference: the volatility at $t$ is built from returns up
#    to $t$ and parameters from before the training window closed, never from later bars.
#
# Stocks outside the subsample take a market-level GARCH fitted the same way on the
# cross-sectional median return, so every emitted row carries a conditional volatility.
#
# The returns handed to both fits come from the **complete** per-symbol series. A variance
# recursion reads its input in order and treats consecutive elements as consecutive sessions;
# feeding it the eligible rows only would splice the two sides of an ineligible spell
# together and price the jump across it as one day's move.


# %%
def select_garch_subsample(data: pl.DataFrame, top_n: int) -> list[str]:
    """Select top-N most liquid symbols by median ADV."""
    ranking = (
        data.group_by("symbol")
        .agg(
            pl.col("adv_21d").median().alias("median_adv"),
            pl.len().alias("n_obs"),
        )
        .filter(pl.col("n_obs") >= GARCH_MIN_OBS)
        .sort("median_adv", descending=True)
        .head(top_n)
    )
    return ranking["symbol"].to_list()


# %%
def fit_garch_per_fold(
    by_symbol: dict,
    fold: dict,
    symbols: list[str],
) -> tuple[pl.DataFrame, list[dict]]:
    """Fit GARCH(1,1) per symbol for a single fold.

    Fit on training data, use model.fix() for the full train+validation period.
    Returns the feature frame and the fitted parameters, one row per symbol.
    """
    fold_idx = fold["fold"]
    results = []
    params_rows = []
    n_success = 0
    n_fail = 0

    for sym in symbols:
        sym_data = by_symbol.get((sym,))
        if sym_data is None:
            n_fail += 1
            continue
        sym_data = sym_data.filter(pl.col("returns").is_not_null())

        # Training data for parameter estimation
        train_data = sym_data.filter(
            pl.col("timestamp").is_between(fold["train_start"], fold["train_end"])
        )

        if len(train_data) < GARCH_MIN_OBS:
            n_fail += 1
            continue

        train_returns_pct = (train_data["returns"] * 100).to_numpy()

        try:
            # Fit on training data only
            train_model = arch_model(
                train_returns_pct,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="Normal",
            )
            train_result = train_model.fit(disp="off", show_warning=False)
            fitted_params = train_result.params
            params_rows.append(
                {
                    "fold": fold_idx,
                    "symbol": sym,
                    "omega": float(fitted_params["omega"]),
                    "alpha": float(fitted_params["alpha[1]"]),
                    "beta": float(fitted_params["beta[1]"]),
                }
            )

            # Full train+test period for feature extraction
            full_data = sym_data.filter(
                pl.col("timestamp").is_between(fold["train_start"], fold["test_end"])
            )
            full_returns_pct = (full_data["returns"] * 100).to_numpy()

            # Run variance recursion with frozen parameters (no re-estimation)
            full_model = arch_model(
                full_returns_pct,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="Normal",
            )
            fixed_result = full_model.fix(fitted_params)
            cond_vol = fixed_result.conditional_volatility

            # Annualized conditional vol (input is in % daily)
            cond_vol_ann = cond_vol * np.sqrt(252) / 100  # Back to decimal

            sym_result = pl.DataFrame(
                {
                    "symbol": [sym] * len(full_data),
                    "timestamp": full_data["timestamp"].to_list(),
                    "garch_cond_vol": cond_vol_ann,
                    "fold": [fold_idx] * len(full_data),
                }
            ).drop_nulls()

            if len(sym_result) > 0:
                results.append(sym_result)
                n_success += 1
        except Exception:
            n_fail += 1

    tag = " [HOLDOUT]" if fold["is_holdout"] else ""
    print(f"  Fold {fold_idx}{tag}: {n_success} of {len(symbols)} fitted, {n_fail} skipped")
    return (pl.concat(results) if results else pl.DataFrame()), params_rows


# %%
def fit_market_garch_per_fold(
    market_ret: FloatArray,
    dates_list: list,
    fold: dict,
) -> pl.DataFrame:
    """Fit market-level GARCH for one fold, return (timestamp, mkt_garch_vol, fold)."""
    fold_idx = fold["fold"]

    train_indices = [
        i for i, d in enumerate(dates_list) if fold["train_start"] <= d <= fold["train_end"]
    ]
    full_indices = [
        i for i, d in enumerate(dates_list) if fold["train_start"] <= d <= fold["test_end"]
    ]

    if len(train_indices) < GARCH_MIN_OBS:
        return pl.DataFrame()

    train_ret_pct = market_ret[train_indices] * 100
    train_ret_clean = train_ret_pct[~np.isnan(train_ret_pct)]

    if len(train_ret_clean) < GARCH_MIN_OBS:
        return pl.DataFrame()

    try:
        train_model = arch_model(
            train_ret_clean,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal",
        )
        train_result = train_model.fit(disp="off", show_warning=False)
        fitted_params = train_result.params

        # Full period with frozen params
        full_ret_pct = market_ret[full_indices] * 100
        # Remove NaN for model fitting but track indices
        valid_mask = ~np.isnan(full_ret_pct)
        full_ret_clean = full_ret_pct[valid_mask]
        valid_dates = [dates_list[full_indices[i]] for i, v in enumerate(valid_mask) if v]

        full_model = arch_model(
            full_ret_clean,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal",
        )
        fixed_result = full_model.fix(fitted_params)
        mkt_cond_vol = fixed_result.conditional_volatility * np.sqrt(252) / 100

        return pl.DataFrame(
            {
                "timestamp": valid_dates[: len(mkt_cond_vol)],
                "mkt_garch_vol": mkt_cond_vol,
                "fold": [fold_idx] * len(mkt_cond_vol),
            }
        )
    except Exception as e:
        print(f"  Fold {fold_idx} market GARCH failed: {e}")
        return pl.DataFrame()


# %% [markdown]
# The recursion reads the complete per-symbol series, so the panel is partitioned by symbol
# once, outside the fold loop. The eligible frame decides only which stocks are liquid enough
# inside a training window to be worth fitting, and that ranking is redone every fold.

# %%
print(f"Fitting up to {GARCH_TOP_N} stocks per fold, plus one market-level fit:")

_returns_by_symbol = raw_df.select("symbol", "timestamp", "returns").partition_by(
    "symbol", as_dict=True
)

garch_all_folds = []
mkt_garch_all_folds = []
garch_param_rows = []

for fold in folds:
    train_data = df.filter(pl.col("timestamp").is_between(fold["train_start"], fold["train_end"]))
    garch_symbols = select_garch_subsample(train_data, GARCH_TOP_N)

    fold_garch, fold_params = fit_garch_per_fold(_returns_by_symbol, fold, garch_symbols)
    if len(fold_garch) > 0:
        garch_all_folds.append(fold_garch)
    garch_param_rows.extend(fold_params)

    fold_mkt = fit_market_garch_per_fold(market_ret, dates, fold)
    if len(fold_mkt) > 0:
        mkt_garch_all_folds.append(fold_mkt)

garch_df = pl.concat(garch_all_folds) if garch_all_folds else pl.DataFrame()
mkt_garch_df = pl.concat(mkt_garch_all_folds) if mkt_garch_all_folds else pl.DataFrame()
garch_params = pl.DataFrame(garch_param_rows)

if len(garch_df) > 0:
    print(
        f"\nPer-stock conditional volatility: {len(garch_df):,} rows on "
        f"{garch_df['symbol'].n_unique():,} distinct stocks across "
        f"{garch_df['fold'].n_unique()} folds"
    )
if len(mkt_garch_df) > 0:
    print(
        f"Market-level conditional volatility, one value per date per fold: "
        f"{len(mkt_garch_df):,} rows across {mkt_garch_df['fold'].n_unique()} folds"
    )

# %% [markdown]
# ## 4b. Fit stability across folds
#
# Refitting each fold costs a fit per symbol per fold, and the question it raises is whether
# the parameters move enough to be worth it. Two of the three transforms have parameters to
# track: the GARCH persistence $\alpha + \beta$, which says how long a volatility shock takes
# to decay, and the separation between the two Wasserstein centroids, which says how far
# apart the two regimes the clustering found actually are. FFD has none, by construction.
#
# A parameter path that is flat says per-fold refitting bought nothing and one fit would have
# served; a path that swings says the transform is chasing a moving target, and the reader
# should carry that into how much weight the feature deserves. The refit cadence here is one
# fit per fold with nothing updating between refits, and that is what the figure judges.
#
# The two answers differ, which is the point of measuring rather than assuming. The prints
# below give both ranges, and the spread of the GARCH band matters separately from the
# position of its median: a median that repeats while the interquartile band widens says the
# typical stock's volatility dynamics are stable and the tails of the subsample are not.
#
# **Two things have to be held fixed for the comparison to be about the parameters.** The
# holdout fold is excluded, because its training window is the whole pre-holdout sample
# against the cross-validation folds' rolling ten years, so a difference there would be a
# window-length difference. And the liquidity ranking selects a slightly different top
# `GARCH_TOP_N` each fold, so the persistence path is restricted to the stocks every fold
# selected - the count is printed. Without both restrictions the line would move for three
# reasons at once and support no statement about any of them.

# %%
_cv_folds = [f["fold"] for f in folds if not f["is_holdout"]]
_cv_params = garch_params.filter(pl.col("fold").is_in(_cv_folds))
_common_symbols = (
    _cv_params.group_by("symbol")
    .agg(pl.col("fold").n_unique().alias("n_folds"))
    .filter(pl.col("n_folds") == len(_cv_folds))["symbol"]
    .to_list()
)
_persistence = (
    _cv_params.filter(pl.col("symbol").is_in(_common_symbols))
    .with_columns((pl.col("alpha") + pl.col("beta")).alias("persistence"))
    .group_by("fold")
    .agg(
        pl.col("persistence").median().alias("median"),
        pl.col("persistence").quantile(0.25).alias("q25"),
        pl.col("persistence").quantile(0.75).alias("q75"),
    )
    .sort("fold")
)
_centroids = pl.DataFrame(wass_centroid_rows).filter(pl.col("fold").is_in(_cv_folds)).sort("fold")
print(
    f"{len(_common_symbols)} of the {GARCH_TOP_N} fitted stocks were selected by all "
    f"{len(_cv_folds)} cross-validation folds; the persistence path below is those"
)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=FIGSIZE["dual_v"], sharex=True, gridspec_kw={"hspace": 0.18}
)
ax1.plot(_persistence["fold"], _persistence["median"], color=COLORS["blue"], marker="o", ms=3)
ax1.fill_between(
    _persistence["fold"],
    _persistence["q25"],
    _persistence["q75"],
    color=COLORS["blue"],
    alpha=0.18,
)
_lo, _hi = _persistence["q25"].min(), _persistence["q75"].max()
ax1.set_ylim(_lo - 0.08 * (_hi - _lo), _hi + 0.08 * (_hi - _lo))
ax1.set_ylabel(r"GARCH $\alpha+\beta$", fontsize=8)
ax2.plot(
    _centroids["fold"], _centroids["centroid_separation"], color=COLORS["copper"], marker="s", ms=3
)
ax2.set_ylabel("Distance between the two centroids", fontsize=8)
ax2.set_xlabel("Fold")
add_message_title(
    ax1,
    "The volatility fit repeats across folds; the regime fit does not",
    subtitle="Persistence median and interquartile band, and the centroid gap",
)
show_with_alt(
    fig,
    "Two stacked panels against the fold number. The upper panel plots the median GARCH "
    "persistence per fold as a line of markers inside a shaded interquartile band. The "
    "line rises gently over the first third of the folds, holds a slight plateau through "
    "the middle, and eases back by the last, staying inside a narrow span near the top of "
    "the axis throughout; the band around it is tight except at the final folds, where it "
    "opens downwards. The lower panel plots the distance between the two regime centroids. "
    "It declines over the fold sequence, steeply at first and then more gradually, but not "
    "monotonically - it steps up at two folds in the second half before resuming - and "
    "ends at a small fraction of where it started.",
)

# %% [markdown] tags=["results"]
# **How far the fitted parameters move as the window rolls.** The persistence range is over
# one cohort of stocks held fixed across the cross-validation folds; the centroid range is the
# gap between the two clustered states in each fold's own training window.

# %%
print(
    f"GARCH persistence: the per-fold median runs from {_persistence['median'].min():.4f} to "
    f"{_persistence['median'].max():.4f} over {_persistence.height} folds"
)
print(
    f"Distance between the two Wasserstein centroids: "
    f"{_centroids['centroid_separation'].min():.5f} to "
    f"{_centroids['centroid_separation'].max():.5f}"
)

# %% [markdown]
# ## 5. Assemble the panel
#
# One frame per fold, then stacked. Each fold's frame starts from the eligible symbol-dates
# inside that fold's span and the three transforms are joined onto it. The two that were
# fitted per fold - the clustering and the GARCH volatility - are filtered to the matching
# fold before they are joined; fractional differencing has no parameters and therefore no
# fold, so the same values are attached to every fold whose span covers them.
#
# The clustering and the market-level volatility are one value per date, so they broadcast
# across every stock trading that date. The per-stock volatility exists only for the stocks
# that fold fitted, and every other stock takes the market-level value in its place, so no
# emitted row is left without a volatility.

# %%
temporal_frames = []

for fold_info in folds:
    fold_idx = fold_info["fold"]

    fold_skeleton = (
        df.filter(pl.col("timestamp").is_between(fold_info["train_start"], fold_info["test_end"]))
        .select(["symbol", "timestamp"])
        .unique()
        .with_columns(pl.lit(fold_idx).alias("fold"))
    )

    if len(wass_df) > 0:
        wass_fold = wass_df.filter(pl.col("fold") == fold_idx).drop("fold")
        fold_skeleton = fold_skeleton.join(wass_fold, on="timestamp", how="left")

    if len(ffd_df) > 0:
        fold_skeleton = fold_skeleton.join(ffd_df, on=["symbol", "timestamp"], how="left")

    if len(garch_df) > 0:
        garch_fold = garch_df.filter(pl.col("fold") == fold_idx).drop("fold")
        fold_skeleton = fold_skeleton.join(garch_fold, on=["symbol", "timestamp"], how="left")

    if len(mkt_garch_df) > 0:
        mkt_fold = mkt_garch_df.filter(pl.col("fold") == fold_idx).drop("fold")
        fold_skeleton = fold_skeleton.join(mkt_fold, on="timestamp", how="left")

    if "garch_cond_vol" in fold_skeleton.columns and "mkt_garch_vol" in fold_skeleton.columns:
        fold_skeleton = fold_skeleton.with_columns(
            pl.when(pl.col("garch_cond_vol").is_null())
            .then(pl.col("mkt_garch_vol"))
            .otherwise(pl.col("garch_cond_vol"))
            .alias("garch_cond_vol")
        )

    temporal_frames.append(fold_skeleton)

temporal = pl.concat(temporal_frames).sort(["fold", "symbol", "timestamp"])
temporal = temporal.drop_nulls(subset=["symbol", "timestamp"])

temporal_feature_cols = [c for c in temporal.columns if c not in ("symbol", "timestamp", "fold")]
n_temporal_features = len(temporal_feature_cols)

# %% [markdown]
# A missing value has to be a null and not a NaN. `ffdiff` returns a float NaN where the log
# price it is handed is not finite, and Polars does not treat that as missing: `drop_nulls`
# keeps the row, and every summary that reaches it returns NaN rather than skipping it -
# including the one printed immediately below. So the conversion happens here rather than at
# the write. `03_financial_features` converts its oscillators' NaN for the same reason.

# %%
_nan_columns = {
    c: int(temporal[c].is_nan().sum())
    for c in temporal_feature_cols
    if temporal.schema[c] in (pl.Float32, pl.Float64) and temporal[c].is_nan().any()
}
if _nan_columns:
    temporal = temporal.with_columns(pl.col(c).fill_nan(None) for c in _nan_columns)
print(
    f"NaN converted to null in {len(_nan_columns)} of {n_temporal_features} features, "
    f"{sum(_nan_columns.values()):,} values: {sorted(_nan_columns)}"
)

print(
    f"\n{n_temporal_features} features on {len(temporal):,} rows, "
    f"{temporal['symbol'].n_unique():,} stocks, {temporal['fold'].n_unique()} folds"
)

# %% [markdown]
# ### What the assembled panel holds
#
# The four columns below are not taken over the same rows, and the split is the point.
#
# `present` and `missing` count every emitted row, the holdout fold included. They describe
# the file rather than the market: the model stages read that fold's rows to score the
# holdout once, so a column with no value there has no value in something they will read, and
# a count stopping at the boundary would not report it.
#
# `mean` and `std` summarize the values a feature takes, and on holdout dates those are values
# this stage may not read. Both are therefore taken over the cross-validation folds alone.
#
# What they summarize is fold-specific values rather than dates. The folds' training windows
# overlap, so a symbol-date carries one value for every fold whose span covers it and enters
# the average that many times. Dropping the holdout fold removes every holdout-dated row, and
# with it one further copy of the development sample - the copy that fold produced from a
# single fit over the whole of it rather than from a fit per fold.

# %%
_development = temporal.filter(pl.col("fold").is_in(_cv_folds))
display(
    pl.DataFrame(
        [
            {
                "feature": c,
                "present": temporal[c].len() - temporal[c].null_count(),
                "missing": temporal[c].null_count(),
                "mean": _development[c].mean(),
                "std": _development[c].std(),
            }
            for c in temporal_feature_cols
        ]
    )
)

# %% [markdown]
# ## 6. Write the artifact
#
# The panel key is `(symbol, timestamp, fold)`, not `(symbol, timestamp)`: the same
# symbol-date appears once per fold whose span covers it, carrying that fold's fit. A
# downstream join that forgets the fold column would multiply rows silently, so the
# uniqueness of the three-column key is asserted before the write rather than trusted.
#
# Three checks run before the write, and each of them is a claim the file would otherwise make
# silently: that the columns present are exactly the ones the three transforms said they would
# emit, that no missing value slipped through as a NaN, and that the three-column key is
# unique.
#
# Beside the parquet the write also leaves a small companion file, the **digest sidecar**. A
# digest is a hash of the values in a frame, so two files with the same digest hold the same
# numbers. The sidecar's job is to let anything reading this artifact establish what it was
# built from, without re-running anything. It records the digest of the values written here,
# how many rows they occupy, which columns identify a row, which notebook wrote them, and the
# digest of the price panel they were computed from. That last entry is what Section 1
# compared against at the top of this notebook, and it is what lets a model notebook confirm
# that the features and the labels it is joining came from one download.
# [`02_labels`](02_labels.ipynb) and [`03_financial_features`](03_financial_features.ipynb)
# leave the same record beside their own artifacts.

# %%
EMITTED_FEATURES = [
    "wass_cluster",
    "wass_dist_min",
    "wass_dist_max",
    "wass_dist_ratio",
    "wass_tail_div",
    "ffd_log_price",
    "ffd_log_volume",
    "garch_cond_vol",
    "mkt_garch_vol",
]
assert sorted(temporal_feature_cols) == sorted(EMITTED_FEATURES), (
    f"emitted {sorted(temporal_feature_cols)} against declared {sorted(EMITTED_FEATURES)}"
)

# No NaN reaches the artifact. Converted in Section 5, checked here.
_still_nan = [
    c
    for c in temporal_feature_cols
    if temporal.schema[c] in (pl.Float32, pl.Float64) and temporal[c].is_nan().any()
]
assert not _still_nan, f"features reaching the artifact with NaN: {_still_nan}"

_key = ["symbol", "timestamp", "fold"]
assert temporal.select(_key).n_unique() == temporal.height, (
    f"{temporal.height - temporal.select(_key).n_unique()} duplicate rows on {_key}"
)

_eligible_keys = df.select("symbol", "timestamp").unique()
assert temporal.join(_eligible_keys, on=["symbol", "timestamp"], how="anti").height == 0, (
    "the emitted panel carries symbol-dates the eligibility screen removed"
)

for f in folds:
    _outside = temporal.filter(
        (pl.col("fold") == f["fold"])
        & ~pl.col("timestamp").is_between(f["train_start"], f["test_end"])
    )
    assert _outside.height == 0, (
        f"fold {f['fold']} emits {_outside.height} rows outside {f['train_start']}..."
        f"{f['test_end']}"
    )
print(
    f"{temporal.height:,} rows, all inside their own fold's span and all on symbol-dates the "
    "screen kept"
)

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
output_path = FEATURES_DIR / "model_based.parquet"
record = write_artifact(
    temporal,
    output_path,
    keys=_key,
    written_by="04_model_based_features",
    inputs={"market_data": MARKET_DATA_DIGEST},
)
print(f"Saved {n_temporal_features} features to {display_path(output_path)}")
print(f"model_based.parquet: {record['n_rows']:,} rows, digest {record['digest']}")
print(f"  Folds: {sorted(temporal['fold'].unique().to_list())}")

# %% [markdown]
# ## 7. What each column ranks on its own
#
# The **information coefficient** of a column is the rank correlation, across the stocks
# scored on one date, between the value the column gives each stock and the return that stock
# actually went on to earn. One correlation per date gives a series, and the series is what is
# summarized below.
#
# What it measures here is each column **on its own**, which is not the same question as what
# each column adds. A column can rank the cross-section well and still add nothing a model did
# not already have from the stage-03 matrix, or rank it barely at all and still matter once a
# model conditions on it. Answering the second question means fitting with and without these
# columns on the same folds, which needs both matrices and a model, so it belongs to
# [`05_evaluation`](05_evaluation.ipynb). This section neither answers it nor selects anything.
#
# **Validation rows only.** Each fold's training bars are the bars its parameters came from,
# so scoring them measures the fit rather than the feature. The frame below is the union of
# the cross-validation folds' validation spans, each row taken from the fold whose validation
# window covers it, and the holdout fold contributes nothing.
#
# **The boundary is where the label resolves, not where it is observed.** A row observed on
# the last validation session of the last fold resolves `LABEL_HORIZON` sessions later, and if
# that lands on or after `holdout_start` the row has read a held-out outcome. The usable
# boundary is therefore the last session whose forward window closes before the holdout opens,
# which is derived below from the panel's own calendar rather than typed.
#
# **Chronologically ordered.** `cross_sectional_ic_series` sorts the dates it returns.
# Feeding a Newey-West correction a series assembled in partition-scan order computes the
# lag structure over an arbitrary permutation of time, and the resulting standard error is
# not merely wrong but unstable between runs.
#
# **Two corrections, kept apart.** Newey-West prices the IC series' own persistence into each
# feature's t-statistic; Benjamini-Hochberg prices the fact that several features are tested
# at once. Neither substitutes for the other.
#
# **And only the features a cross-sectional statistic can measure.** Most of what this
# notebook emits is market-level: the Wasserstein features describe the panel's centre on a
# date, and `mkt_garch_vol` describes its volatility, so every stock carries the same value
# on that date. An information coefficient is a correlation *across* the cross-section, and a
# column with no cross-sectional variation has no correlation with anything in it - not a
# small one, an undefined one. Scoring such a column here returns a number, and the number is
# an artifact of how the tie is broken rather than a statement about the feature. The cell
# below classifies the emitted columns by counting distinct values within a date and scores
# only those that vary, and the ones it sets aside are set aside by measurement rather than
# by a list somebody kept up to date.
#
# Setting them aside says nothing about their worth. A daily regime state can matter to a
# cross-sectional ranker through what it interacts with - momentum conditioned on the regime
# is a different signal from momentum - and it can matter to a timing overlay. Both are
# questions about a fitted model, so both belong downstream; neither is answerable with the
# statistic this section computes.

# %% [markdown]
# The endpoint of a label is the `LABEL_HORIZON`-th next session in the stock's own series,
# read off the complete price frame so that it is the next session and not the next session on
# which the stock happened to still be eligible - a later date, and one decided by what
# happened after the decision was made.
#
# Two assertions carry the section. The first is that the validation windows tile the
# development period without overlapping, so a symbol-date reaches the scored frame from
# exactly one fold; were they ever to overlap, the correlation helper's self-join would
# quietly multiply the cross-section rather than raise. The second is that no scored row sits
# inside the training window of the fold whose parameters produced its features.

# %%
_label_df = pl.read_parquet(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")
label_col = PRIMARY_LABEL

_label_end = raw_df.select(
    "symbol",
    "timestamp",
    pl.col("timestamp").shift(-LABEL_HORIZON).over("symbol").alias("_label_end"),
)

_val_rows = pl.concat(
    [
        temporal.filter(
            (pl.col("fold") == f["fold"])
            & pl.col("timestamp").is_between(f["test_start"], f["test_end"])
        )
        for f in folds
        if not f["is_holdout"]
    ]
)
assert _val_rows.select("symbol", "timestamp").n_unique() == _val_rows.height, (
    "a symbol-date appears in more than one fold's validation window"
)

eval_df = (
    _val_rows.join(_label_df, on=["symbol", "timestamp"], how="inner")
    .join(_label_end, on=["symbol", "timestamp"], how="left")
    .filter(pl.col("_label_end") < holdout_start)
    .drop("_label_end")
)
assert eval_df["timestamp"].max() < holdout_start, "a scored row resolves inside the holdout"
for f in folds:
    _bled = eval_df.filter(
        (pl.col("fold") == f["fold"])
        & pl.col("timestamp").is_between(f["train_start"], f["train_end"])
    )
    assert _bled.height == 0, f"{_bled.height} scored rows sit inside fold {f['fold']}'s own fit"

print(f"Scoring {len(temporal_feature_cols)} columns against {label_col}")
print(
    f"  {eval_df.height:,} validation rows over {eval_df['timestamp'].n_unique():,} dates, "
    f"the last resolving before {holdout_start}"
)

# %% [markdown]
# The narrowest cross-section a correlation is computed over is half the median date's, rather
# than a fixed count, for the reason `02_labels` gives: a rank correlation over a handful of
# names is mostly noise, and a fixed threshold means something different on a panel of a
# hundred stocks than on one of a thousand. Dates below it contribute no point to the series.

# %%
_min_obs = int(eval_df.group_by("timestamp").len()["len"].median() // 2)
print(f"A date is scored when at least {_min_obs:,} stocks are ranked on it")

# %% [markdown]
# Which columns vary across a cross-section is measured, not declared. A feature whose
# median date carries a single distinct value is market-level and is set aside; the count
# for each is printed, so a column that is *nearly* constant - one value for the stocks a
# model was fitted on and a broadcast for the rest - is visible as the partial thing it is
# rather than passing as cross-sectional.

# %%
_variation = (
    eval_df.group_by("timestamp")
    .agg(pl.col(c).n_unique().alias(c) for c in temporal_feature_cols)
    .select(pl.col(c).median().alias(c) for c in temporal_feature_cols)
    .row(0, named=True)
)
CROSS_SECTIONAL = [c for c in temporal_feature_cols if _variation[c] > 1]
MARKET_LEVEL = [c for c in temporal_feature_cols if _variation[c] <= 1]

display(
    pl.DataFrame(
        [
            {
                "feature": c,
                "distinct_values_on_median_date": int(_variation[c]),
                "scored_below": _variation[c] > 1,
            }
            for c in temporal_feature_cols
        ]
    )
)
assert CROSS_SECTIONAL, "no emitted feature varies across the cross-section"

# %%
ic_rows = []
for feat in CROSS_SECTIONAL:
    _ic = cross_sectional_ic_series(
        eval_df,
        eval_df,
        pred_col=feat,
        ret_col=label_col,
        date_col="timestamp",
        entity_col="symbol",
        method="spearman",
        min_obs=_min_obs,
    ).drop_nulls("ic")
    if _ic.height < 20:
        continue
    stats = compute_ic_hac_stats(_ic, ic_col="ic", label_horizon=LABEL_HORIZON)
    ic_rows.append(
        {
            "feature": feat,
            "n_dates": _ic.height,
            "ic_mean": stats["mean_ic"],
            "hac_se": stats["hac_se"],
            "hac_tstat": stats["t_stat"],
            "p_value": stats["p_value"],
        }
    )

temporal_ic = pl.DataFrame(ic_rows)
assert temporal_ic.height > 0, "no temporal feature carried enough scored dates to compute an IC"

_fdr = benjamini_hochberg_fdr(
    temporal_ic["p_value"].to_list(), alpha=FDR_ALPHA, return_details=True
)
temporal_ic = temporal_ic.with_columns(
    pl.Series("adjusted_p", list(_fdr["adjusted_p_values"])),
    pl.Series("significant_fdr05", list(_fdr["rejected"])),
).sort(pl.col("ic_mean").abs(), descending=True)

# %% [markdown] tags=["results"]
# **What each column ranks on its own, over the validation rows.** The count set aside is the
# market-level columns the statistic cannot reach; of the rest, the first figure is how many
# clear the 5% level on their own t-statistic and the second how many survive the
# Benjamini-Hochberg correction for testing several columns at once.

# %%
print(
    f"{temporal_ic.height} of {len(temporal_feature_cols)} columns scored "
    f"({len(MARKET_LEVEL)} set aside as market-level); "
    f"{int((temporal_ic['p_value'] < FDR_ALPHA).sum())} significant on their own and "
    f"{int(_fdr['n_rejected'])} after Benjamini-Hochberg"
)
display(temporal_ic.select("feature", "n_dates", "ic_mean", "hac_tstat", "significant_fdr05"))

# %% [markdown]
# The bars are signed, because a column the panel ranks one way and a column it ranks the
# other are different signals and a sorted magnitude hides that. The whisker is
# $\pm2$ Newey-West standard errors, and the fill marks whether Benjamini-Hochberg still
# rejects the null for that column. Only the columns that vary across a cross-section appear,
# and the table above names the ones set aside; the comparison against the stage-03 features
# is deferred to [`05_evaluation`](05_evaluation.ipynb), which scores both matrices on one
# frame.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
_order = temporal_ic.sort("ic_mean")
_ypos = np.arange(_order.height)
ax.barh(
    _ypos,
    _order["ic_mean"],
    xerr=2 * _order["hac_se"].to_numpy(),
    color=[
        COLORS["blue"] if s else COLORS["silver_muted"]
        for s in _order["significant_fdr05"].to_list()
    ],
    error_kw=dict(ecolor=COLORS["neutral"], lw=0.9),
    height=0.45,
)
ax.set_yticks(_ypos)
ax.set_yticklabels(_order["feature"].to_list(), fontsize=8)
ax.set_ylim(-0.6, _order.height - 0.4)
ax.axvline(0, color=COLORS["neutral"], lw=0.8)
ax.set_xlabel(f"Mean information coefficient against {label_col}, validation rows")
add_message_title(
    ax,
    "The columns that vary across stocks rank in both directions",
    subtitle="Mean signed IC with two Newey-West standard errors. Filled: survives BH",
)
show_with_alt(
    fig,
    "One horizontal bar per temporal feature, giving its mean information coefficient "
    "against the one-day forward return over validation rows, with a whisker of two "
    "Newey-West standard errors and a vertical line at zero. The two fractionally "
    "differenced columns extend well clear of zero in opposite directions and are filled "
    "solid, marking them as surviving the false-discovery correction. The conditional "
    "volatility column sits just left of zero, is drawn unfilled, and its whisker crosses "
    "the zero line.",
)

# %% [markdown]
# ## Key takeaways
#
# 1. **A fitted feature carries its estimation window, so resolve the folds before anything
#    is fitted.** Every parameter here comes from one fold's training bars, and the assertion
#    after the fold table is what establishes that rather than the prose around it. A
#    notebook that resolves its boundaries after the fit has nothing left to check them
#    against.
# 2. **Derive the folds from the same frame the consumer derives them from.** A fold id is a
#    join key between this artifact and every model notebook downstream, and a walk-forward
#    splitter counts backward from the holdout boundary in rows of whatever frame it is handed
#    and seals each window by the horizon of the label being predicted. Two frames that differ
#    by a handful of dates give two sets of windows that carry the same numbers and cover
#    different spans, and nothing about the join says so. Take the boundaries from the artifact
#    the consumer takes them from, and the question stops arising.
# 3. **Count windows in sessions, and get sessions from the exchange calendar.** An archive
#    carries stray prints on dates no market was held. A rolling average, a difference
#    convolution and a variance recursion all read their input in order and treat consecutive
#    elements as consecutive sessions, so a stray row silently widens every window that spans
#    it. Screening on the calendar first is what makes "twenty-one sessions" mean that.
# 4. **Run inference forward, never backward.** `model.fix()` applies training parameters to
#    later returns without re-estimating them, so a row's conditional volatility is built
#    from returns up to that row. The smoothed alternative - refitting or running a smoother
#    over the whole span - conditions every value on the end of the series, and it fails
#    silently because the output looks the same.
# 5. **Emit the holdout fold, and say why that is allowed.** These transforms read prices and
#    never labels, so a pre-holdout fit may produce filtered values *for* holdout dates and
#    the model stages need them. What is forbidden is a fit that reads a holdout bar.
# 6. **Measure the trade a default encodes.** The differencing order is not searched per fold
#    and does not have to be, but the memory it keeps and the stationarity it buys are
#    measurable in a few lines and were worth measuring rather than quoting.
# 7. **Score on validation rows, in time order, and call the result what it is.** Training
#    rows measure the fit rather than the feature; a per-date IC series in partition order
#    gives a Newey-West standard error computed over a permutation of time; the multiplicity
#    correction is a separate quantity from the autocorrelation one; and a per-feature IC is
#    marginal, so it cannot answer the incremental question however many corrections it
#    carries.
# 8. **Check that the statistic can reach the column before reporting it.** Most of what a
#    market-level transform emits is constant within a date, and a rank correlation across a
#    cross-section of identical values is undefined rather than zero. A correlation helper
#    handed such a column still returns a number, decided by how it breaks the ties, so which
#    columns to score is measured from the data rather than taken from a list, and the ones it
#    cannot reach are named and set aside instead of ranked.
#
# ### Known limitations
#
# - Most of the emitted columns are market-level and carry no cross-sectional information as
#   main effects; the distinct-value count in Section 7 says which. They are emitted because
#   a model can use them through an interaction or a timing overlay, and neither is tested
#   here.
# - The GARCH subsample is the most liquid stocks of each training window. Every other stock
#   carries the market-level conditional volatility, so `garch_cond_vol` varies across a few
#   hundred names and is a broadcast for the rest - its IC is measured on that mostly
#   degenerate column, and the distinct-value count printed above is what says so.
# - Clustering on the cross-sectional median reads a shift in the centre of the panel and is
#   blind to a regime that keeps its centre and widens its tails.
# - The refit cadence is one fit per fold with nothing updating between refits. The fit
#   stability figure says what that costs; choosing a different cadence is not attempted here.
# - Running the per-symbol transforms on the complete series stops a window from counting
#   eligible rows instead of sessions, but the complete series still has holes: a stock that
#   is suspended and resumes has consecutive rows spanning months, and a shift, an FFD
#   convolution and a variance recursion all read them as consecutive sessions.
#   [`02_labels`](02_labels.ipynb) Section D measures how often that happens on the forward
#   side and this notebook does not segment on it, so a feature on the first row after a
#   suspension is built partly from before it.
# - `arch`'s `.fix()` recomputes its internal variance bounds over the series it is handed,
#   which spans training and validation. The clipping envelope therefore depends on the
#   validation period even though the parameters do not.
#
# **Next**: [`05_evaluation`](05_evaluation.ipynb) scores this matrix and the stage-03 one on
# the same frame and decides what carries forward.
