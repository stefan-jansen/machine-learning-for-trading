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

# %% [markdown] tags=[]
# # FX Pairs: Features Built From Fitted Models
#
# **Chapter 9: Time Series Analysis**
#
# Chapter 8's features are arithmetic on past prices: a moving average, a return over
# twenty sessions, a ratio of two of them. This notebook builds a different kind. Each
# feature here is the output of a model whose parameters were themselves estimated from
# price history, so the window those parameters came from is part of what the feature
# knows. Three models are fitted, one per section: a state-space model that splits a
# spot rate into a slowly-changing level and the quoting noise around it, a two-state
# model of when the dollar is calm and when it is turbulent, and a short-memory return
# model whose forecast error becomes a surprise measure.
#
# **Learning Objectives**:
# - Split a currency pair's price into a slowly-moving level and the noise around it,
#   by fitting a model that treats the level as hidden and each observed price as a
#   noisy reading of it, on training sessions alone.
# - Estimate, for each session, how likely the dollar is to be in its turbulent state,
#   from a two-state model that is allowed to read only the sessions up to that day.
# - Turn a one-step-ahead return forecast into a feature by keeping what the forecast
#   missed, so the feature measures surprise rather than direction.
# - Show that a feature carries no look-ahead by re-running the same recursion on a
#   series with its tail deleted and checking that the earlier values do not move.
#
# **Book Reference**: Chapter 9, Sections 9.2 (Kalman), 9.5 (HMM), 9.3 (ARIMA)
#
# **Prerequisites**: FX 4H price bars, which section 1 aggregates to sessions, and
# [`02_labels`](02_labels.ipynb), which writes the label parquet read in section 3 and
# whose date index the folds are derived from.
#
# **Output Contract**:
# - `features/model_based.parquet` -- ten columns, five from the state-space fit, two
#   from the dollar-regime fit and three from the return model
# - Keys: `timestamp`, `symbol`, `fold`; `fold` records which fit produced the row and
#   is not itself a feature
# - Every value is computed from observations up to and including its own session
# - Each fold carries its training and validation sessions, so a downstream model reads
#   the rows belonging to the fold it is training on

# %% tags=[]
"""FX Pairs: Features Built From Fitted Models."""

import logging
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from hmmlearn.hmm import GaussianHMM
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series
from ml4t.diagnostic.splitters.calendar import TradingCalendar
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from threadpoolctl import threadpool_limits

from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.cv_window import assert_variant_folds_are_out_of_sample
from case_studies.utils.temporal import filtered_state_probs, sort_states_by_variance
from data import load_fx_pairs
from utils.artifact_specs import load_setup_config, resolve_label_buffer
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.paths import get_case_study_dir
from utils.style import COLORS, show_plotly_with_alt

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

# %% [markdown] tags=[]
# Everything in the next cell can be overridden without editing the file, which is how a
# reader runs a smaller version of the notebook first. Each one trades runtime for scope:
# how many pairs are fitted, how many walk-forward windows are covered, how hard the
# search for the state-space noise parameters tries, and how many times the dollar-regime
# model is refitted from a different starting point before one of them is kept.
#
# `START_DATE` is the earliest session to load. 2011 is where the OANDA four-hour history
# begins, so it is the whole file rather than a choice about how much of it to use.

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
# 0 means every pair and every fold; a positive value keeps that many of each.
MAX_SYMBOLS = 0
MAX_FOLDS = 0
# Iteration cap for the Nelder-Mead search over the three state-space noise parameters.
KALMAN_MAXITER = 300
START_DATE = "2011-01-01"
# Expectation-maximisation reaches a local optimum, so the fit is repeated from this many
# starting points and the highest-likelihood one is kept.
N_HMM_RESTARTS = 10
HMM_N_STATES = 2  # a calm dollar state and a turbulent one
# A restart is rejected when its final EM step falls by more than this fraction of
# the log-likelihood's own magnitude. Real divergence moves hundreds of nats; the
# noise this has to tolerate is single digits against a likelihood of ~4.3e4.
HMM_STABILITY_REL_TOL = 1e-3

# %% [markdown] tags=[]
# The session calendar is read from `setup.yaml` rather than named here. It is the
# calendar that implements the 5PM rollover, so it decides which session a four-hour
# bar belongs to, and `02_labels` reads the same key. A copy typed here would let this
# notebook aggregate onto a different session grid than the labels were built on, and
# the resulting join would simply lose rows.

# %% tags=[]
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
LABELS_DIR = CASE_DIR / "labels"
FEATURES_DIR = CASE_DIR / "features"

ARIMA_ORDER = (1, 0, 1)
# A Spearman IC over fewer pairs than this is a rank correlation over a handful of
# points; dates below the floor are dropped from the series rather than averaged in.
MIN_PAIRS_PER_DATE = 8

SETUP = load_setup_config(CASE_STUDY_ID)
SESSION_CALENDAR = SETUP["decision"]["session_calendar"]

# Two windows this notebook needs are already decided in `setup.yaml`, and both are read
# from it rather than typed, so a configuration change reaches the models rather than
# leaving them measuring against a window the feature stage no longer uses.
#
# `kalman_trend` is the fitted level less a moving average of the price, and it is the
# middle of the three moving-average windows the feature configuration declares: the
# shortest sits inside the filter's own responsiveness, so the difference would be mostly
# filter noise, and the longest is slower than a fold's validation year. Taking the same
# window `03_financial_features` gives `price_to_ma_63d` also means the two columns
# measure price against one reference rather than two.
KALMAN_TREND_WINDOW = int(sorted(SETUP["features"]["windows"]["moving_average"])[1])
# The dollar-regime model is given the shortest close-to-close volatility window the
# configuration declares - about a trading month, long enough for a stable estimate and
# short enough to move when the market does.
USD_VOL_WINDOW = int(min(SETUP["features"]["windows"]["close_to_close_volatility"]))
USD_VOL_COL = f"usd_vol_{USD_VOL_WINDOW}d"

# %% [markdown] tags=[]
# ## 1. Load the Price History and the Universe


# %% [markdown] tags=[]
# The price file holds four-hour bars. Every model here works on sessions, so the bars are
# first collapsed onto the session calendar named in `setup.yaml` - the one that implements
# the 5PM rollover, and the same one `02_labels` used, so the two agree on which session a
# bar belongs to.

# %% tags=[]
fx_4h = load_fx_pairs(
    frequency="4h",
    start_date=START_DATE,
).select(["symbol", "timestamp", "open", "high", "low", "close", "volume"])

cal = TradingCalendar(SESSION_CALENDAR)
sessions = cal.get_sessions(pd.DatetimeIndex(fx_4h["timestamp"].to_pandas()))
# Retain the original 4H timestamp as `bar_ts` so OHLC sort_by inside agg
# is order-safe (polars group_by does not contractually preserve row order).
fx_4h = (
    fx_4h.rename({"timestamp": "bar_ts"})
    .with_columns(pl.Series("timestamp", sessions.values).cast(pl.Date))
    .drop_nulls("timestamp")
)
prices = (
    fx_4h.group_by(["symbol", "timestamp"])
    .agg(
        pl.col("open").sort_by("bar_ts").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").sort_by("bar_ts").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    )
    .sort(["symbol", "timestamp"])
)

# %% [markdown] tags=[]
# ### Select the Universe
#
# The universe is the one declared in `setup.yaml`. The labels were built for that
# list, so a pair present in the price file but absent from the declared universe
# would enter the USD factor and the cross-sectional IC here while appearing in no
# downstream join.

# %% tags=[]
SYMBOLS = sorted(SETUP["universe"]["symbols"])
assert len(SYMBOLS) == SETUP["universe"]["n_assets"], (
    f"setup.yaml declares {SETUP['universe']['n_assets']} assets, "
    f"universe.symbols lists {len(SYMBOLS)}"
)
_loaded = set(prices["symbol"].unique().to_list())
assert set(SYMBOLS) <= _loaded, f"price file is missing {sorted(set(SYMBOLS) - _loaded)}"
prices = prices.filter(pl.col("symbol").is_in(SYMBOLS))
if MAX_SYMBOLS:
    SYMBOLS = SYMBOLS[:MAX_SYMBOLS]
    prices = prices.filter(pl.col("symbol").is_in(SYMBOLS))
n_symbols = len(SYMBOLS)
dates = prices.filter(pl.col("symbol") == SYMBOLS[0])["timestamp"].sort().to_list()

print(f"Loaded: {n_symbols} pairs, {len(dates)} dates")
print(f"Period: {dates[0]} to {dates[-1]}")

# %% [markdown] tags=[]
# ### What Is In This Universe
#
# A count of pairs is not enough to read the rest of the notebook, because the three
# models treat the pairs differently and the differences run along lines the count hides.
#
# The market divides these quotes into two kinds. A **dollar pair** has the US dollar on
# one side of the quote, so its move is largely a move in the dollar itself; the
# dollar-regime model in section 5 is built from exactly these and no others. A **cross**
# is quoted between two other currencies, and the yen crosses are separated out because
# the yen is quoted in hundredths rather than ten-thousandths, which puts its price on a
# different numeric scale from every other pair in the file.
#
# The table below carries what the later sections depend on: how many sessions each group
# has, so the 252-session minimum training length in section 4 can be checked against it,
# and how far a session's return typically travels, which is the quantity the state-space
# model has to attribute between a moving level and quoting noise. Scale is why the models
# read the logarithm of the price rather than the price: the log return of a yen pair and
# of a euro pair are comparable, their price levels are not.

# %% tags=[]
_group = (
    pl.when(pl.col("symbol").str.contains("USD"))
    .then(pl.lit("Dollar pair"))
    .when(pl.col("symbol").str.contains("JPY"))
    .then(pl.lit("Yen cross"))
    .otherwise(pl.lit("Other cross"))
)
universe_table = (
    prices.with_columns(
        _group.alias("group"),
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("_ret"),
    )
    .group_by("group")
    .agg(
        pl.col("symbol").n_unique().alias("pairs"),
        pl.col("symbol").unique().sort().str.join(", ").alias("which"),
        pl.col("timestamp").min().alias("first_session"),
        pl.col("timestamp").n_unique().alias("sessions"),
        (pl.col("_ret").std() * np.sqrt(252) * 100).round(1).alias("annualised_vol_pct"),
    )
    # Two of the three groups hold the same number of pairs, so sorting on the count
    # alone leaves their order to whatever `group_by` happened to emit, which differs
    # between runs. The name breaks the tie, so a reader re-running this sees the table
    # printed here.
    .sort(["pairs", "group"], descending=[True, False])
)
universe_table

# %% [markdown] tags=[]
# ## 2. Why a Fitted Feature Is Different
#
# A Chapter 8 feature is a function of past prices. A twenty-session return reads twenty
# closes and arithmetic turns them into one number. Move the window and the arithmetic is
# unchanged; the only thing that decides the value is which prices fall inside it.
#
# A feature here is a function of *parameters that were themselves estimated from prices*.
# The state-space model in section 4 does not know how much of a day's move is a lasting
# change in the level until it has been told how noisy the quotes are, and it is told that
# by fitting two variances to a stretch of history. Only then can it produce a value for a
# single session. So the feature at any one date depends on two windows, not one: the
# sessions the recursion has walked through, and the window the parameters were fitted on.
#
# That second window is what makes this stage a hazard the last one was not. If the
# parameters are fitted on the whole sample, then the value the model reports for a
# session in 2016 was shaped by what happened in 2022, and no amount of care in the
# recursion removes it. The feature would look ordinary, the notebook would run clean, and
# a strategy built on it could not have been run at the time. Nothing in the emitted
# numbers reveals this: a leaked fit and an honest one produce columns of the same shape,
# the same range and the same plausibility.
#
# The discipline that removes it has two parts, and the rest of the notebook is those two
# parts applied three times:
#
# 1. **Fit inside a window that ends before the sessions being scored.** Every parameter
#    in this notebook is estimated on one fold's training sessions and then held fixed.
# 2. **Run the model forward, never backward.** A fitted model can be asked two different
#    questions about a past session: what do I believe about it given everything up to it,
#    and what do I believe about it given everything including what came after. The second
#    is the more accurate answer and it is unusable, because at the time the decision was
#    made the later data did not exist. Sections 4, 5 and 6 each take the first, and each
#    ends with an executed check that deleting the tail of the series leaves the earlier
#    values untouched - which is the only way to tell the two apart from the outside.
#
# Because these three models read prices and never read a label, the boundary they must
# respect is the observation date alone: a fit may use any session it could have seen, and
# the holdout is the one stretch it may not. The forward-looking part of the discipline -
# not letting a label's outcome window reach into the holdout - binds section 11, where a
# label enters for the first time.

# %% [markdown] tags=[]
# ## 3. Resolve the Walk-Forward Folds Before Anything Is Fitted
#
# A walk-forward fold is a pair of date ranges: a training range the model may be fitted
# on, and a later validation range it is scored over, separated by a gap wide enough that
# the outcome of the last training decision has already resolved before the first
# validation decision is taken. The boundaries come from `generate_cv_splits` reading the
# label file and the window sizes in `setup.yaml`. This is the same route the downstream
# loader takes, so a `fold` id in this artifact selects the same window there as it does
# here. The `fold` id is only an id: nothing downstream re-checks that the window it
# selects is the one the features were fitted on, so deriving both from the same call is
# what keeps them the same window.
#
# The folds are laid out by stepping backward from the date the holdout opens, so **fold
# 0 is the most recent window and the highest-numbered fold is the oldest**. That is worth
# stating rather than inferring: code that treats a lower fold id as the earlier period
# reads every one of these folds in the wrong order, and the dates below are the check.

# %% tags=[]
all_dates = sorted(prices["timestamp"].unique().to_list())

# The label is the case study's configured primary, not a name typed here: the same
# key picks the label file, the buffer that spaces the folds, and the HAC lag below.
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, PRIMARY_LABEL, SETUP)
assert LABEL_BUFFER, f"No label buffer configured for {PRIMARY_LABEL}"
# Consecutive daily decisions share (h - 1) days of outcome window, which is what the
# Newey-West lag has to cover. Read from the buffer rather than typed, so a case study
# that moves to a longer label cannot leave a stale lag behind.
LABEL_HORIZON_SESSIONS = int(re.match(r"^(\d+)", LABEL_BUFFER).group(1))
# One holdout boundary, resolved once. The rule drawn on the fold figure below and the
# assertion in section 11 have to be the same date, or the figure stops describing the
# check.
HOLDOUT_START = pd.Timestamp(load_evaluation_config(CASE_STUDY_ID)["holdout_start"]).date()
print(
    f"Primary label {PRIMARY_LABEL}, buffer {LABEL_BUFFER} -> HAC lag horizon "
    f"{LABEL_HORIZON_SESSIONS}; holdout opens {HOLDOUT_START}"
)

# %% [markdown] tags=[]
# Each split arrives as four dates. The session counts beside them are how many of this
# notebook's own trading sessions fall inside each window, which is what the minimum
# training length in section 4 is checked against.

# %% tags=[]
label_frame = pl.read_parquet(LABELS_DIR / f"{PRIMARY_LABEL}.parquet")
raw_folds = generate_cv_splits(
    label_frame.select("timestamp").unique().sort("timestamp"),
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
)
folds = []
for split in raw_folds:
    fold = {
        "fold": int(split["fold"]),
        "train_start": pd.Timestamp(split["train_start"]).date(),
        "train_end": pd.Timestamp(split["train_end"]).date(),
        "val_start": pd.Timestamp(split["val_start"]).date(),
        "val_end": pd.Timestamp(split["val_end"]).date(),
    }
    fold["n_train"] = sum(fold["train_start"] <= d <= fold["train_end"] for d in all_dates)
    fold["n_val"] = sum(fold["val_start"] <= d <= fold["val_end"] for d in all_dates)
    folds.append(fold)

if MAX_FOLDS:
    folds = folds[:MAX_FOLDS]

print(f"Built {len(folds)} walk-forward folds:")
for f in folds:
    print(
        f"  Fold {f['fold']}: train {f['train_start']}..{f['train_end']} "
        f"({f['n_train']} days), validation {f['val_start']}..{f['val_end']} "
        f"({f['n_val']} days)"
    )

# %% [markdown] tags=[]
# ### One Artifact, Three Labels
#
# The folds above were cut for the primary label. This case study also configures two
# longer-horizon labels, and the gap between training and validation is sized to the label
# being predicted, so each of the three has boundaries of its own. A downstream model
# trained on a longer label resolves its own folds and then reads this one artifact by
# `fold` id - which is safe only if the sessions this notebook emits for a fold cover the
# sessions that label's version of the fold asks for. Where they do not, the model finds
# no feature value for part of its window and fills the gap with an imputed one.
#
# The checks below are the whole of what makes reading by `fold` id safe, so they are
# executed rather than argued, and there are two of them because coverage is not the only
# way the arrangement can fail.
#
# The first is coverage. A longer gap moves `train_end` earlier and leaves `train_start`
# where it was, so each longer label's fold is contained in the one written here; the
# assertion is what would catch a future label whose gap is shorter than the primary's,
# for which the containment runs the other way and this artifact would be short of rows.
#
# The second is that the values a longer label's model is scored on were not fitted on the
# sessions it is scoring. Every value this notebook writes for fold F was fitted on the
# **primary** label's training window for fold F, and a model reading fold F by id gets
# those values whatever its own boundaries are. Containment does not settle it: a variant
# whose validation window opened inside the primary's training window would be contained
# and would still be scoring on sessions its features had already seen, so the property
# has to be stated directly: the variant's validation opens after the primary's training
# closes. `assert_variant_folds_are_out_of_sample` is that check, shared with the loader
# every downstream model calls, and it compares timestamps rather than dates - on a
# minute-bar case study the two disagree, and a fit closing at 15:22 against a validation
# opening at 15:38 reads as a violation on the calendar day alone.

# %% tags=[]
label_geometries = {}
for _label in [PRIMARY_LABEL, *SETUP["labels"].get("variants", [])]:
    _buffer = resolve_label_buffer(CASE_STUDY_ID, _label, SETUP)
    _path = LABELS_DIR / f"{_label}.parquet"
    if not _buffer or not _path.exists():
        continue
    label_geometries[_label] = {
        int(s["fold"]): (
            pd.Timestamp(s["train_start"]).date(),
            pd.Timestamp(s["train_end"]).date(),
            pd.Timestamp(s["val_start"]).date(),
            pd.Timestamp(s["val_end"]).date(),
        )
        for s in generate_cv_splits(
            pl.read_parquet(_path).select("timestamp").unique().sort("timestamp"),
            case_study_id=CASE_STUDY_ID,
            label_buffer=_buffer,
        )
    }

emitted = {f["fold"]: f for f in folds}
for _label, _geometry in label_geometries.items():
    for _fold_id, (_ts, _te, _vs, _ve) in _geometry.items():
        if _fold_id not in emitted:
            continue
        _own = emitted[_fold_id]
        assert _ts >= _own["train_start"] and _ve <= _own["val_end"], (
            f"{_label} fold {_fold_id} spans {_ts}..{_ve}, outside the "
            f"{_own['train_start']}..{_own['val_end']} this notebook emits for that fold"
        )
    print(
        f"  {_label:<14} buffer {resolve_label_buffer(CASE_STUDY_ID, _label, SETUP):<4} "
        f"fold 0 train ends {_geometry[0][1]}, validates {_geometry[0][2]} to "
        f"{_geometry[0][3]}"
    )
print(
    f"All {len(label_geometries)} configured label geometries are covered by the "
    f"{len(folds)} folds written here."
)

# %% [markdown] tags=[]
# The gaps the second check measures, one row per variant label and fold. Every one is
# positive, and the narrowest is the one to watch: it is the label whose validation opens
# closest to the sessions its features were fitted on.

# %% tags=[]
variant_gaps = pl.DataFrame(assert_variant_folds_are_out_of_sample(CASE_STUDY_ID, PRIMARY_LABEL))
# The gap ties across labels and folds, so the label and fold break it: without them
# the five rows shown are whichever five the tie happened to order first.
display(variant_gaps.sort(["gap", "label", "fold"]).head(5))
print(
    f"Narrowest gap between a variant's validation opening and the {PRIMARY_LABEL} "
    f"training window its features were fitted through: {variant_gaps['gap'].min()}."
)

# %% [markdown] tags=[]
# ### The Fold Contract
#
# The figure draws what the saved artifact will contain: per fold, the window each
# model's parameters are estimated on and the window they are then applied to out of
# sample, with the holdout period shaded. The state-space noise parameters, the
# dollar-regime model's emissions and transitions, and the return model's coefficients
# are all estimated inside the blue bar of their own row and then held frozen while the
# recursion runs forward across the amber one.
#
# `fx_pairs` writes features for the cross-validation folds only, so every bar stops
# to the left of the rule; a downstream stage that needs a holdout vintage builds one
# with `append_holdout_fold_if_needed` (`utils/modeling.py:688`).
#
# The gap printed above separates each training bar from its validation bar. At one
# session against a fifteen-year axis it is narrower than a pixel here, so it is a number
# to read rather than a gap to look for.

# %% tags=[]
fig = go.Figure()
_style = {
    "Parameters estimated here": COLORS["blue"],
    "Applied out of sample here": COLORS["amber"],
}
_seen: set[str] = set()
for f in folds:
    row = f"Fold {f['fold']}"
    for kind, (start, end) in (
        ("Parameters estimated here", (f["train_start"], f["train_end"])),
        ("Applied out of sample here", (f["val_start"], f["val_end"])),
    ):
        fig.add_trace(
            go.Scatter(
                x=[start.isoformat(), end.isoformat()],
                y=[row, row],
                mode="lines",
                line={"width": 14, "color": _style[kind]},
                name=kind,
                legendgroup=kind,
                showlegend=kind not in _seen,
            )
        )
        _seen.add(kind)

# %% [markdown] tags=[]
# The holdout is drawn on the same axis: shaded from the date it opens to the end of the
# price file, with a rule at the boundary itself.

# %% tags=[]
fig.add_vrect(
    x0=HOLDOUT_START.isoformat(),
    x1=max(all_dates).isoformat(),
    fillcolor=COLORS["neutral"],
    opacity=0.10,
    line_width=0,
    layer="below",
)
fig.add_vline(x=HOLDOUT_START.isoformat(), line_dash="dash", line_color=COLORS["negative"])
fig.update_layout(
    title=(
        "No fold's parameters come from the right of its own training bar"
        "<br><sup>Dashed rule is where the holdout opens; the shaded region is held back."
        "<br>No bar crosses it - this notebook writes cross-validation folds only.</sup>"
    ),
    xaxis_title="Session",
    yaxis_title="",
    height=420,
    margin={"l": 90},
)
show_plotly_with_alt(
    fig,
    "One horizontal bar per fold, split into the stretch each fold's models are "
    "estimated on and the later stretch they are applied to out of sample. The bars step "
    "up and to the right, fold zero covering the most recent window and the "
    "highest-numbered fold the oldest. A dashed vertical rule marks where the holdout "
    "opens, with the region beyond it shaded, and every bar ends to the left of it.",
)


# %% [markdown] tags=[]
# ## 4. Where the Price Level Is, and How Fast It Is Moving
#
# The first model treats the price a reader observes as an imperfect reading of something
# that cannot be observed directly. There is a true level, it drifts at some rate, and the
# quote prints somewhere near it. Two sources of movement are therefore competing to
# explain each session: the level genuinely moved, or the quote landed away from a level
# that did not. A **local linear trend** model - a state-space model, meaning one written
# as a hidden state that evolves plus a noisy observation of it - is the standard way to
# separate them.
#
# The hidden state has two components, the level and the slope, and the observation is the
# level plus noise:
#
# **State**: $\mathbf{x}_t = [\text{level}_t, \text{slope}_t]^\top$
#
# **Transition**: $\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t$
#
# **Observation**: $y_t = [1, 0]\mathbf{x}_t + v_t$
#
# How the split is made is decided entirely by the relative sizes of the two noise terms:
# $R$, how far a quote strays from the level, and $Q$, how far the level and its slope
# move on their own. Those are the parameters, and they are what gets estimated on each
# training window by maximum likelihood - the values under which the training prices are
# the most probable thing the model could have produced. Once fitted they are held fixed,
# and the recursion runs forward through validation without re-estimating.
#
# The models read the logarithm of the price rather than the price. A yen pair trades near
# 100 and a euro pair near 1, so a fixed $R$ would mean two different things for the two;
# in logarithms both are on the scale of a return, and level, slope, forecast error and
# uncertainty are comparable across every pair in the universe.


# %% tags=[]
def kalman_local_linear(
    prices_arr: np.ndarray,
    observation_noise: float = 1.0,
    level_noise: float = 0.01,
    slope_noise: float = 0.001,
) -> dict[str, np.ndarray]:
    """Local linear trend Kalman filter.

    Returns dict with level, slope, innovation, uncertainty arrays.
    """
    n = len(prices_arr)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[level_noise, 0.0], [0.0, slope_noise]])
    R = np.array([[observation_noise]])

    x = np.array([prices_arr[0], 0.0])
    P = np.eye(2) * 10.0

    levels = np.zeros(n)
    slopes = np.zeros(n)
    innovations = np.zeros(n)
    uncertainties = np.zeros(n)
    log_lik = 0.0

    for t in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        y = prices_arr[t] - H @ x_pred
        S = H @ P_pred @ H.T + R

        log_lik += -0.5 * (np.log(2 * np.pi * S[0, 0]) + y[0] ** 2 / S[0, 0])

        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred

        levels[t] = x[0]
        slopes[t] = x[1]
        innovations[t] = y[0]
        uncertainties[t] = P[0, 0]

    return {
        "level": levels,
        "slope": slopes,
        "innovation": innovations,
        "uncertainty": uncertainties,
        "log_likelihood": log_lik,
    }


# %% [markdown] tags=[]
# ### Fit the Two Noise Sizes to the Training Window
#
# The recursion above returns the log-likelihood of the prices it was given under the
# noise sizes it was given, so fitting is a search over those three numbers for the
# combination that makes the training prices most probable. Each is a variance and must
# stay positive, so the search runs over their logarithms and exponentiates on the way in;
# that removes the constraint rather than enforcing it.


# %% tags=[]
def neg_log_likelihood(params: np.ndarray, prices_arr: np.ndarray) -> float:
    """Negative log-likelihood for MLE optimization."""
    obs_noise = np.exp(params[0])
    level_noise = np.exp(params[1])
    slope_noise = np.exp(params[2])
    result = kalman_local_linear(prices_arr, obs_noise, level_noise, slope_noise)
    return -result["log_likelihood"]


# %% [markdown] tags=[]
# A search of this kind has to be told where to start, and the starting point decides
# which local optimum it reaches. The variance of the training returns is the natural
# choice: it is already on the scale the three parameters live on, and it is measured on
# the same pair, so a yen pair and a euro pair each begin from their own magnitude rather
# than from a shared constant that would suit one and not the other.


# %% tags=[]
def fit_kalman_mle(train_prices: np.ndarray, maxiter: int = 300) -> tuple[float, float, float]:
    """Estimate Kalman noise parameters via MLE on training data."""
    return_variance = max(float(np.var(np.diff(train_prices))), 1e-10)
    x0 = np.log([return_variance * 0.5, return_variance * 0.1, return_variance * 0.01])
    opt = minimize(
        neg_log_likelihood,
        x0,
        args=(train_prices,),
        method="Nelder-Mead",
        options={"maxiter": maxiter},
    )
    return tuple(np.exp(opt.x))


# %% [markdown] tags=[]
# ### Run It Fold by Fold
#
# For each fold and each pair, the noise sizes are fitted on the training sessions and
# the recursion is then run forward across training and validation together, without
# re-estimating. Running it across both is what makes the validation values usable: a
# recursion carries its state forward, so restarting it at the first validation session
# would throw away everything the model had learned about where the level was.
#
# Five columns come out of it. `kalman_trend` is how far the fitted level sits above or
# below a 63-session moving average of the price, `kalman_slope` is the drift rate the
# model currently believes in, `kalman_slope_zscore` puts that drift on the scale of its
# own training-window spread, `kalman_innovation` is the gap between the observed price
# and what the model expected before seeing it, and `kalman_smoothness` is one over the
# uncertainty the model attaches to its own level estimate.
#
# The sessions falling in the gap between training and validation are walked through so
# the state stays current, and are written to neither split. At session $t$ the update
# has read observations through $t$ and no further.
#
# The path's first session is walked through and not emitted either, for a different
# reason. A recursion has to start somewhere, and it starts at the first observed price
# with a slope of zero, so on that one session the forecast equals the observation: the
# forecast error is identically zero, the slope is the zero it was initialised to, and the
# trend is the price minus itself. Those are not small values, they are the starting
# assumption showing through, and nothing downstream could tell them apart from a session
# on which the price happened to land exactly where the model expected.


# %% tags=[]
def extract_kalman_features(fold: dict, symbol: str) -> tuple[list[dict], dict | None]:
    """Fit one training fold and filter its train-to-validation path.

    Returns ``(rows, params)``. ``params`` carries the MLE noise estimates for this
    fold and symbol so the fit-stability section can draw what was estimated rather
    than what the emitted features happened to average to.
    """
    sym_data = prices.filter(pl.col("symbol") == symbol).sort("timestamp")
    sym_dates = sym_data["timestamp"].to_list()
    sym_log_prices = np.log(sym_data["close"].to_numpy())
    train_mask = [fold["train_start"] <= d <= fold["train_end"] for d in sym_dates]
    val_mask = [fold["val_start"] <= d <= fold["val_end"] for d in sym_dates]
    path_mask = [fold["train_start"] <= d <= fold["val_end"] for d in sym_dates]
    train_prices = sym_log_prices[train_mask]
    path_prices = sym_log_prices[path_mask]
    path_dates = [d for d, include in zip(sym_dates, path_mask, strict=True) if include]
    # i > 0 drops the prior; see the note above.
    emit_idx = [
        i
        for i, d in enumerate(path_dates)
        if i > 0 and (d <= fold["train_end"] or d >= fold["val_start"])
    ]
    if len(train_prices) < 252 or sum(val_mask) < 10:
        return [], None
    opt_params = fit_kalman_mle(train_prices, maxiter=KALMAN_MAXITER)
    filtered = kalman_local_linear(path_prices, *opt_params)
    train_idx = np.array([d <= fold["train_end"] for d in path_dates])
    slope_mean = np.mean(filtered["slope"][train_idx])
    slope_std = np.std(filtered["slope"][train_idx]) + 1e-10
    moving_average = (
        pl.Series(path_prices).rolling_mean(KALMAN_TREND_WINDOW, min_samples=1).to_numpy()
    )
    params = {
        "fold": fold["fold"],
        "symbol": symbol,
        "observation_noise": float(opt_params[0]),
        "level_noise": float(opt_params[1]),
        "slope_noise": float(opt_params[2]),
    }
    rows = [
        {
            "timestamp": path_dates[i],
            "symbol": symbol,
            "fold": fold["fold"],
            "kalman_trend": filtered["level"][i] - moving_average[i],
            "kalman_slope": filtered["slope"][i],
            "kalman_slope_zscore": (filtered["slope"][i] - slope_mean) / slope_std,
            "kalman_innovation": filtered["innovation"][i],
            "kalman_smoothness": 1.0 / (filtered["uncertainty"][i] + 1e-10),
        }
        for i in emit_idx
    ]
    return rows, params


# %% tags=[]
kalman_results = []
kalman_params = []
for fold in folds:
    for symbol in SYMBOLS:
        try:
            rows, params = extract_kalman_features(fold, symbol)
            kalman_results.extend(rows)
            if params is not None:
                kalman_params.append(params)
        except Exception as exc:
            raise RuntimeError(
                f"Kalman MLE failed for fold {fold['fold']}, symbol {symbol}"
            ) from exc
    if fold["fold"] % 2 == 1 or fold["fold"] == folds[-1]["fold"]:
        n_fold = sum(row["fold"] == fold["fold"] for row in kalman_results)
        print(f"  Kalman fold {fold['fold']}: {n_fold:,} features")

kalman_df = pl.DataFrame(kalman_results)
print(f"\nKalman features: {len(kalman_df):,} rows, {n_symbols} pairs x {len(folds)} folds")

# %% [markdown] tags=[]
# **The two checks this section rests on, executed.** Each stops the notebook rather than
# leaving plausible numbers behind.
#
# *Containment.* Every emitted row is dated inside its own fold's training or validation
# window, and none reaches the holdout.
#
# *Forward only.* `kalman_local_linear` is a recursion, so the value it reports for
# session `i` must not move when the observations after `i` are deleted. This is the
# distinction section 2 named as the one that matters and the one that is invisible in the
# emitted numbers: a backward pass would produce a column of the same shape and range.
# Deleting the second half of one pair's history and re-running gives a direct answer.
# The truncation runs on the pre-holdout series, the same boundary every other cell reads
# its data through - a check that reads held-back sessions in order to prove they are held
# back reports on a series no other cell is allowed to see.

# %% tags=[]
for fold in folds:
    rows = kalman_df.filter(pl.col("fold") == fold["fold"])
    if len(rows) == 0:
        continue
    assert rows["timestamp"].min() >= fold["train_start"], (
        f"fold {fold['fold']}: Kalman row before its own train_start"
    )
    assert rows["timestamp"].max() <= fold["val_end"], (
        f"fold {fold['fold']}: Kalman row after its own val_end"
    )
assert kalman_df["timestamp"].max() < HOLDOUT_START, "Kalman emitted a holdout-dated row"

seal_prices = np.log(
    prices.filter((pl.col("symbol") == SYMBOLS[0]) & (pl.col("timestamp") < HOLDOUT_START))
    .sort("timestamp")["close"]
    .to_numpy()
)
cut = len(seal_prices) // 2
full_run = kalman_local_linear(seal_prices)
prefix_run = kalman_local_linear(seal_prices[:cut])
kalman_drift = max(
    float(np.abs(full_run[k][:cut] - prefix_run[k]).max()) for k in ("level", "slope", "innovation")
)
assert kalman_drift < 1e-10, f"Kalman state moved by {kalman_drift:.2e} - not a forward filter"
print(
    f"Level-model checks hold across {len(folds)} folds; last emitted date "
    f"{kalman_df['timestamp'].max()} < holdout start {HOLDOUT_START}; deleting the last "
    f"{len(seal_prices) - cut} observations of {SYMBOLS[0]} moves the first {cut} filtered "
    f"states by {kalman_drift:.2e}"
)

# %% [markdown] tags=[]
# ## 5. When the Dollar Is Calm and When It Is Turbulent
#
# The second model answers a question about the market as a whole rather than about one
# pair. Currency volatility arrives in stretches: months where dollar moves are small and
# orderly, then a period where they are not, then back. A **hidden Markov model** is the
# standard way to describe that. It assumes the market is always in one of a small number
# of unobservable states, that each state produces observations with its own mean and
# variance, and that the state persists from one session to the next with a fixed
# probability. Two states are configured here, and after fitting they are ordered so that
# the one with the larger variance is the turbulent one - a naming rule the fit itself does
# not supply, since the two states come back in an arbitrary order every time.
#
# What is emitted is not which state the market was in but how likely each session is to
# have been in the turbulent one, computed from the sessions up to that day. A probability
# carries the model's uncertainty; a hard label discards it.
#
# The larger-variance state is described as turbulent and nothing more. Variance says how
# far the dollar travelled, not which way, so it does not identify the state where
# investors are retreating from risk - that would need the direction of the move as well,
# and this model is not given it.

# %% [markdown] tags=[]
# The model reads one series: an average dollar return across the seven pairs that have
# the dollar on one side of the quote. Those seven are the dollar pairs from the universe
# table, and the sign has to be fixed before averaging, because `USD_JPY` rising and
# `EUR_USD` rising are opposite moves in the dollar. Both sides are derived from the
# declared universe rather than listed here, so a universe change cannot silently drop a
# leg of the average.

# %% tags=[]
USD_LONG = [s for s in SYMBOLS if s.startswith("USD_")]
USD_SHORT = [s for s in SYMBOLS if s.endswith("_USD")]
print(f"USD factor legs: long {USD_LONG}, short {USD_SHORT}")

daily_rets = prices.with_columns(
    (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret")
).drop_nulls(subset=["ret"])

usd_rets = daily_rets.filter(pl.col("symbol").is_in(USD_LONG + USD_SHORT)).with_columns(
    pl.when(pl.col("symbol").is_in(USD_LONG))
    .then(pl.col("ret"))
    .otherwise(-pl.col("ret"))
    .alias("usd_ret")
)

usd_daily = (
    usd_rets.group_by("timestamp").agg(pl.col("usd_ret").mean().alias("usd_ret")).sort("timestamp")
)

# %% [markdown] tags=[]
# The model is given two numbers per session rather than one: the average dollar return
# and a rolling standard deviation of it over the window bound above. The return alone
# would let the model separate the states only through how far individual sessions
# scatter, and the rolling figure states the recent scale directly, which is the quantity
# the two states differ in.

# %% tags=[]
usd_daily = usd_daily.with_columns(pl.col("usd_ret").rolling_std(USD_VOL_WINDOW).alias(USD_VOL_COL))

print(f"USD factor series: {len(usd_daily):,} dates")

# %% [markdown] tags=[]
# ### Reading the Model Forward
#
# The library's own `predict_proba` answers the question section 2 named as unusable: it
# returns the probability of each state given the *whole* series, later sessions included.
# The probability given only the sessions up to and including the one being scored comes
# from the forward recursion, which `case_studies.utils.temporal.filtered_state_probs`
# implements. It is imported rather than written out here because six notebooks in this
# book need the same recursion, and it reaches one library method that is not part of the
# public interface - a detail worth carrying in one place rather than six.


# %% [markdown] tags=[]
# Expectation-maximisation climbs to whichever optimum is nearest its starting point, so
# the fit is repeated from several starting points and the highest-likelihood result is
# kept. A run whose final step *lowers* the likelihood has not converged, and is discarded
# rather than quietly used.
#
# Fixing the starting points is not by itself enough to make this fit reproducible, and
# the difference matters because the feature file is identified by a digest of its values.
# The initial state means come from a k-means partition of the training sample, and
# k-means sums over that sample in parallel. Floating-point addition is not associative,
# so the sums depend on how the work happened to be divided across processor threads, and
# expectation-maximisation carries that difference forward into the transition matrix and
# into every probability the model reports. A seed fixes which starting points are drawn,
# not how the arithmetic is scheduled. Holding the fit to a single thread fixes the
# schedule too, and it costs seconds here because the series is one column of daily
# figures. Measured over three separate runs of this notebook's fit: with the default
# thread pool the transition matrix came back different every time; held to one thread it
# came back identical every time. The other two models were checked the same way and are
# already reproducible across runs.


# %% tags=[]
def fit_best_hmm(X_train: np.ndarray) -> tuple[GaussianHMM, float, int]:
    """Return the highest-likelihood stable training-only HMM fit."""
    best_ll = -np.inf
    best_model = None
    unstable = 0
    for seed in range(N_HMM_RESTARTS):
        try:
            with threadpool_limits(limits=1):
                model = GaussianHMM(
                    n_components=HMM_N_STATES,
                    covariance_type="full",
                    n_iter=100,
                    random_state=seed,
                    tol=1e-4,
                ).fit(X_train)
            history = list(model.monitor_.history)
            final_delta = history[-1] - history[-2] if len(history) >= 2 else 0.0
            # Relative to the likelihood being stepped on: an absolute nat threshold
            # rejects ordinary floating-point chatter at the optimum, which on a
            # likelihood of this magnitude discards every restart.
            scale = max(abs(history[-2]) if len(history) >= 2 else 1.0, 1.0)
            if final_delta < -HMM_STABILITY_REL_TOL * scale:
                unstable += 1
                continue
            score = model.score(X_train)
            if np.isfinite(score) and score > best_ll:
                best_ll, best_model = score, model
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("No stable HMM fit")
    return best_model, best_ll, unstable


# %% [markdown] tags=[]
# Each fold emits training and validation sessions only; the sessions in the gap between
# them advance the recursion and are written to neither split.
#
# Two columns come out. `hmm_regime_prob_high_vol` is the probability the session was in
# the higher-variance state, and `hmm_regime_transition_5d` is how much that probability
# has moved over the last five sessions, which turns a level into a measure of a regime
# changing. The difference is null rather than zero for the first five sessions of a
# fold's path: there is no session five back to difference against. The panel already
# carries rows on which this difference is genuinely zero because the probability did not
# move, and writing a zero here would make the two indistinguishable.


# %% tags=[]
def extract_hmm_features(fold: dict) -> tuple[list[dict], GaussianHMM, np.ndarray, float, int]:
    """Fit one HMM fold and return its filtered feature rows."""
    train_idx = [
        i for i, d in enumerate(valid_dates) if fold["train_start"] <= d <= fold["train_end"]
    ]
    val_idx = [i for i, d in enumerate(valid_dates) if fold["val_start"] <= d <= fold["val_end"]]
    path_idx = [i for i, d in enumerate(valid_dates) if fold["train_start"] <= d <= fold["val_end"]]
    if len(train_idx) < 252 or len(val_idx) < 10:
        raise ValueError(f"Insufficient HMM data for fold {fold['fold']}")
    model, score, unstable = fit_best_hmm(usd_arr[train_idx])
    order = sort_states_by_variance(model)
    filtered = filtered_state_probs(model, usd_arr[path_idx])
    path_dates = [valid_dates[i] for i in path_idx]
    rows = [
        {
            "timestamp": hmm_date,
            "fold": fold["fold"],
            "hmm_regime_prob_high_vol": float(filtered[i, order[1]]),
            "hmm_regime_transition_5d": (
                float(filtered[i, order[1]] - filtered[i - 5, order[1]]) if i >= 5 else None
            ),
        }
        for i, hmm_date in enumerate(path_dates)
        if hmm_date <= fold["train_end"] or hmm_date >= fold["val_start"]
    ]
    return rows, model, order, score, unstable


# %% [markdown] tags=[]
# ### Why the Model Reads Percent and Not Decimals
#
# The series handed to the model is multiplied by 100, so a dollar move of a few tenths of
# a percent arrives as a number near one rather than as a number near one thousandth. The
# reason has nothing to do with the market and everything to do with two constants inside
# `GaussianHMM`, each of which adds a fixed amount to a state's variance and neither of
# which scales with the data it is given:
#
# - `min_covar` is added to the covariance the fit *starts* from, so it decides where the
#   search begins rather than where it ends. It is not a floor on the fitted value.
# - `covars_prior` is added at every step of the fit, divided by how many observations the
#   state currently holds. It inflates each state's variance estimate by an amount that
#   shrinks as the state takes on more observations.
#
# Both defaults are sized for data of order one. A daily FX return is three orders of
# magnitude smaller than that and its variance five, which puts the variance below either
# constant - so on decimal returns the fit would begin from a covariance that is
# essentially the constant rather than the data, and would return variances visibly
# inflated by the second. Multiplying by
# 100 multiplies the variance by 10,000 and puts it in the range those defaults were
# chosen for.
#
# The cell below measures both effects against the series they act on rather than
# asserting them.

# %% tags=[]
HMM_SCALE = 100.0  # decimal returns -> percent, so the two fixed constants stay small
HMM_MIN_COVAR = GaussianHMM().min_covar  # added to the initial covariance
HMM_COVARS_PRIOR = GaussianHMM().covars_prior  # added at every fitting step

# %% [markdown] tags=[]
# The series is cut at the holdout boundary before anything reads it. Every fold's path
# already ends to the left of that boundary, so this removes no session any model is
# fitted on or run over. What it removes is the holdout's contribution to the variance
# printed below - and that variance is the measurement the whole scaling argument rests
# on, so it has to be measured on the same history the models are allowed to see.

# %% tags=[]
valid_usd = usd_daily.drop_nulls(subset=["usd_ret", USD_VOL_COL]).filter(
    pl.col("timestamp") < HOLDOUT_START
)
valid_dates = valid_usd["timestamp"].to_list()
_native = valid_usd.select(["usd_ret", USD_VOL_COL]).to_numpy()
usd_arr = _native * HMM_SCALE
print(
    f"USD series, cut at {HOLDOUT_START}: {len(valid_dates):,} sessions, "
    f"{valid_dates[0]} to {valid_dates[-1]}"
)

# %% [markdown] tags=[]
# Both comparisons are against the variance of the return column the model actually reads.
# The second constant is divided by the number of observations a state holds, so splitting
# the fitted sample evenly between the two states gives its order of magnitude without
# refitting anything.

# %% tags=[]
native_var = float(_native[:, 0].var())
scaled_var = float(usd_arr[:, 0].var())
obs_per_state = len(_native) / HMM_N_STATES
prior_term = HMM_COVARS_PRIOR / obs_per_state

print(f"USD return variance          native {native_var:.3e}   scaled {scaled_var:.3e}")
print(f"Observations per state, approx.    {obs_per_state:,.0f}")
print("\nAt the start, min_covar added straight to the covariance:")
print(
    f"  min_covar {HMM_MIN_COVAR:.1e} / variance   native {HMM_MIN_COVAR / native_var:9.1f}x"
    f"   scaled {HMM_MIN_COVAR / scaled_var:.4f}x"
)
print("\nAt every step, covars_prior spread over a state's observations:")
print(
    f"  prior term {prior_term:.2e}          native inflates "
    f"{1 + prior_term / native_var:.3f}x   scaled inflates {1 + prior_term / scaled_var:.3f}x"
)

# %% [markdown] tags=[]
# Each fold is fitted on its own training window, and the loop keeps the restart count it
# had to discard.

# %% tags=[]
hmm_results = []
hmm_params = []
unstable_hmm_fits = 0
best_model = None
for fold in folds:
    rows, best_model, order, best_ll, unstable = extract_hmm_features(fold)
    hmm_results.extend(rows)
    unstable_hmm_fits += unstable
    _trans = best_model.transmat_[np.ix_(order, order)]
    hmm_params.append(
        {
            "fold": fold["fold"],
            "persist_low_vol": float(_trans[0, 0]),
            "persist_high_vol": float(_trans[1, 1]),
            "log_likelihood": float(best_ll),
        }
    )
    print(f"  HMM fold {fold['fold']}: {len(rows)} dates (train+validation), max LL={best_ll:.1f}")
print(f"HMM unstable restarts excluded: {unstable_hmm_fits}")


# %% [markdown] tags=[]
# The matrix below is from the last fold the loop fitted. Fold 0 covers the most recent
# validation year, so the last fold is the fit made on the oldest training window, and the
# number printed beside the table names it. Each row is the state the session starts in
# and each column the probability of the next session's state, so the diagonal says how
# often a state persists. A state that persists with probability $p$ lasts $1/(1-p)$
# sessions on average, which is the last column and is easier to read than the probability
# itself.

# %% tags=[]
trans = best_model.transmat_[np.ix_(order, order)]
transition_table = pl.DataFrame(
    {
        "from_state": ["low_vol", "high_vol"],
        "to_low_vol": [trans[0, 0], trans[1, 0]],
        "to_high_vol": [trans[0, 1], trans[1, 1]],
        "expected_sessions": [1.0 / (1.0 - trans[0, 0]), 1.0 / (1.0 - trans[1, 1])],
    }
)
print(f"HMM transition matrix, fold {folds[-1]['fold']}, states ordered by variance:")
transition_table

# %% tags=[]
hmm_df = pl.DataFrame(hmm_results)
print(f"HMM features: {len(hmm_df):,} rows")

# %% [markdown] tags=[]
# **The same two checks, against what this section emits.** Containment first, then the
# truncation test. The forward recursion is written out rather than taken from a library
# call, so the truncation test is the only thing standing between it and the probability
# given the whole series - which would carry every validation session into each
# training-date value.

# %% tags=[]
for fold in folds:
    rows = hmm_df.filter(pl.col("fold") == fold["fold"])
    if len(rows) == 0:
        continue
    assert rows["timestamp"].min() >= fold["train_start"], (
        f"fold {fold['fold']}: HMM row before its own train_start"
    )
    assert rows["timestamp"].max() <= fold["val_end"], (
        f"fold {fold['fold']}: HMM row after its own val_end"
    )
assert hmm_df["timestamp"].max() < HOLDOUT_START, "HMM emitted a holdout-dated row"

seal_train_idx = [
    i for i, d in enumerate(valid_dates) if folds[0]["train_start"] <= d <= folds[0]["train_end"]
]
seal_model, _, _ = fit_best_hmm(usd_arr[seal_train_idx])
seal_obs = usd_arr[seal_train_idx]
cut = len(seal_obs) // 2
hmm_drift = float(
    np.abs(
        filtered_state_probs(seal_model, seal_obs)[:cut]
        - filtered_state_probs(seal_model, seal_obs[:cut])
    ).max()
)
assert hmm_drift < 1e-10, f"filtered probabilities moved by {hmm_drift:.2e} - not filtered"
print(
    f"Regime checks hold across {len(folds)} folds; last emitted date "
    f"{hmm_df['timestamp'].max()} < holdout start {HOLDOUT_START}; deleting the last "
    f"{len(seal_obs) - cut} observations of fold {folds[0]['fold']} moves the first {cut} "
    f"probabilities by {hmm_drift:.2e}"
)

# %% [markdown] tags=[]
# ## 6. What the Return Model Did Not See Coming
#
# The third model is the smallest of the three and its output is the most direct. An
# ARIMA(1,0,1) fitted to a pair's daily returns says that today's return is partly
# predictable from yesterday's return and partly from yesterday's forecast error - one
# autoregressive term and one moving-average term, which is the shortest memory the model
# family offers. Daily currency returns are close to unpredictable, so a fit like this
# explains very little, and that is the point: the feature kept is not the forecast but
# **what the forecast missed**. A large error is a session that moved unlike the recent
# past, which is a different quantity from a large return.
#
# Three columns come out: the forecast itself, the error, and that error divided by the
# spread of the training-window errors so a quiet pair and a volatile one are comparable.
#
# The fitted coefficients must not move when the recursion is extended past the training
# window. `fit.apply(path_rets, refit=False)` is the call that guarantees it: the state
# recursion advances with each new observation while the coefficients stay where the
# training fit left them. That is a claim about a library, so the check below compares the
# two parameter vectors element by element rather than trusting the argument name.


# %% tags=[]
def extract_arima_features(fold: dict, symbol: str) -> list[dict]:
    """Fit one training fold and refilter its train-to-validation return path."""
    sym_data = prices.filter(pl.col("symbol") == symbol).sort("timestamp")
    sym_dates = sym_data["timestamp"].to_list()
    sym_close = sym_data["close"].to_numpy()
    sym_rets = np.diff(sym_close) / sym_close[:-1]
    ret_dates = sym_dates[1:]
    train_mask = [fold["train_start"] <= d <= fold["train_end"] for d in ret_dates]
    val_mask = [fold["val_start"] <= d <= fold["val_end"] for d in ret_dates]
    path_mask = [fold["train_start"] <= d <= fold["val_end"] for d in ret_dates]
    train_rets = sym_rets[train_mask]
    path_rets = sym_rets[path_mask]
    path_dates = [d for d, include in zip(ret_dates, path_mask, strict=True) if include]
    if len(train_rets) < 252 or sum(val_mask) < 10:
        return []
    fit = ARIMA(train_rets, order=ARIMA_ORDER).fit()
    train_resid_std = np.std(fit.resid) + 1e-10
    extended = fit.apply(path_rets, refit=False)
    predicted = extended.predict(start=0, end=len(path_rets) - 1)
    return [
        {
            "timestamp": arima_date,
            "symbol": symbol,
            "fold": fold["fold"],
            "arima_forecast": float(predicted[i]),
            "arima_residual": float(path_rets[i] - predicted[i]),
            "arima_residual_zscore": float((path_rets[i] - predicted[i]) / train_resid_std),
        }
        for i, arima_date in enumerate(path_dates)
        if arima_date <= fold["train_end"] or arima_date >= fold["val_start"]
    ]


# %% tags=[]
arima_results = []
for fold in folds:
    for symbol in SYMBOLS:
        try:
            arima_results.extend(extract_arima_features(fold, symbol))
        except Exception as exc:
            raise RuntimeError(
                f"ARIMA fit failed for fold {fold['fold']}, symbol {symbol}"
            ) from exc
    if fold["fold"] % 2 == 0 or fold["fold"] == folds[-1]["fold"]:
        n_fold = sum(row["fold"] == fold["fold"] for row in arima_results)
        print(f"  ARIMA fold {fold['fold']}: {n_fold:,} features")

arima_df = pl.DataFrame(arima_results)
print(f"\nARIMA features: {len(arima_df):,} rows")

# %% [markdown] tags=[]
# **The same checks again, plus one this section needs on its own.** Containment first.
# Then the truncation test the other two sections run, on the forecasts this one emits.
# Between them sits the claim particular to this model: that `apply(..., refit=False)`
# extends the recursion without re-estimating. Truncation alone would not catch a re-fit,
# because a model re-estimated on the longer series is still a forward pass over it - so
# the two parameter vectors are also compared element by element. Every series here stops
# at the holdout boundary, like every other series in this notebook.

# %% tags=[]
for fold in folds:
    rows = arima_df.filter(pl.col("fold") == fold["fold"])
    if len(rows) == 0:
        continue
    assert rows["timestamp"].min() >= fold["train_start"], (
        f"fold {fold['fold']}: ARIMA row before its own train_start"
    )
    assert rows["timestamp"].max() <= fold["val_end"], (
        f"fold {fold['fold']}: ARIMA row after its own val_end"
    )
assert arima_df["timestamp"].max() < HOLDOUT_START, "ARIMA emitted a holdout-dated row"

seal_data = prices.filter(
    (pl.col("symbol") == SYMBOLS[0]) & (pl.col("timestamp") < HOLDOUT_START)
).sort("timestamp")
seal_close = seal_data["close"].to_numpy()
seal_rets = np.diff(seal_close) / seal_close[:-1]
seal_ret_dates = seal_data["timestamp"].to_list()[1:]
seal_fold = folds[0]
seal_train = seal_rets[
    [seal_fold["train_start"] <= d <= seal_fold["train_end"] for d in seal_ret_dates]
]
seal_path = seal_rets[
    [seal_fold["train_start"] <= d <= seal_fold["val_end"] for d in seal_ret_dates]
]
seal_fit = ARIMA(seal_train, order=ARIMA_ORDER).fit()
seal_applied = seal_fit.apply(seal_path, refit=False)
param_drift = float(np.abs(np.asarray(seal_fit.params) - np.asarray(seal_applied.params)).max())
assert param_drift == 0.0, f"apply() re-estimated: parameters moved by {param_drift:.2e}"

seal_cut = len(seal_path) // 2
full_pred = np.asarray(seal_applied.predict(start=0, end=len(seal_path) - 1))
prefix_pred = np.asarray(
    seal_fit.apply(seal_path[:seal_cut], refit=False).predict(start=0, end=seal_cut - 1)
)
arima_drift = float(np.abs(full_pred[:seal_cut] - prefix_pred).max())
assert arima_drift < 1e-10, f"forecasts moved by {arima_drift:.2e} - not a forward pass"
print(
    f"Return-model checks hold across {len(folds)} folds; last emitted date "
    f"{arima_df['timestamp'].max()} < holdout start {HOLDOUT_START}; extending "
    f"{SYMBOLS[0]} fold {seal_fold['fold']} from {len(seal_train)} to {len(seal_path)} "
    f"observations moves the fitted parameters by {param_drift:.2e}, and deleting the "
    f"last {len(seal_path) - seal_cut} of them moves the first {seal_cut} forecasts by "
    f"{arima_drift:.2e}"
)

# %% [markdown] tags=[]
# ## 7. Do the Fitted Parameters Move as the Window Rolls?
#
# Refitting once per fold is a decision, and this is where it gets checked. The training
# windows roll forward one year at a time and overlap by four of their five years, so the
# fitted parameters should move slowly. Parameters that come back identical fold after fold
# say the refitting bought nothing and one fit would have done; parameters that swing say
# the feature depending on them means something different in each window, which is a
# warning to carry into how it is used.
#
# The left panel is the three noise sizes from the state-space fit, taken as the median
# across pairs so one badly behaved pair does not stand for the fold. The axis is
# logarithmic because the three differ by orders of magnitude by construction: quoting
# noise, level movement and slope movement are not comparable quantities.
#
# **Read the observation-noise line before the others.** In most folds it sits near the
# level noise, which is the split the model exists to make. In the folds the cell below
# names it falls instead, by the factor printed there, to a number that is zero for every
# practical purpose. That is the search running the likelihood off the end of its own
# parameter: with $R$ at zero the model believes each observed price exactly, so the level
# it reports is the price and the uncertainty it attaches to that level goes to zero.
# `kalman_smoothness` is one over that uncertainty, so in those folds it saturates at the
# constant its denominator is floored with, and the range printed below is the spread that
# produces. Within a fold the column still ranks pairs; across folds it is not one scale,
# which is a property to know before pooling folds.
#
# That is a limitation of the fit rather than a break in the fold discipline. Nothing about
# it reaches across a fold boundary: the search that failed read that fold's training
# sessions and no others.
#
# The right panel is how long each of the two dollar states persists. Both self-transition
# probabilities sit close enough to one that drawing them directly would put two traces
# against the top of the axis with the difference between them invisible, so each is drawn
# as the run length it implies, $1/(1-p_{\text{stay}})$ - the quantity a reader would want
# anyway, and the one `hmm_regime_transition_5d` responds to.
#
# The return model is refitted per pair and its two coefficients have no market-level
# counterpart to plot against a fold, so it has no line here.

# %% tags=[]
kalman_param_df = pl.DataFrame(kalman_params)
kalman_param_summary = (
    kalman_param_df.group_by("fold")
    .agg(
        pl.col("observation_noise").median().alias("observation_noise"),
        pl.col("level_noise").median().alias("level_noise"),
        pl.col("slope_noise").median().alias("slope_noise"),
        pl.len().alias("n_pairs"),
    )
    .sort("fold")
)
hmm_param_df = pl.DataFrame(hmm_params).sort("fold")

# %% [markdown] tags=[]
# How far the observation noise moves is a scalar, and an axis spanning twenty orders of
# magnitude is not where a reader should have to estimate one. A fold counts as collapsed
# when its median $R$ falls more than six orders below the largest fold's - far enough
# below ordinary fold-to-fold movement that the two cannot be confused. The last line
# prices what the collapse does to the feature that variance feeds.

# %% tags=[]
_r = kalman_param_summary["observation_noise"]
_zero_threshold = float(_r.max()) * 1e-6
_zeroed = kalman_param_summary.filter(pl.col("observation_noise") < _zero_threshold)
_intact = kalman_param_summary.filter(pl.col("observation_noise") >= _zero_threshold)
print("Observation noise R, median across pairs, by fold:")
print(
    f"  folds fitting normally  {sorted(_intact['fold'].to_list())}: "
    f"{_intact['observation_noise'].min():.2e} to {_intact['observation_noise'].max():.2e}"
)
if len(_zeroed):
    _drop = float(_intact["observation_noise"].min()) / float(_zeroed["observation_noise"].max())
    print(
        f"  folds where it went to zero {sorted(_zeroed['fold'].to_list())}: "
        f"{_zeroed['observation_noise'].min():.2e} to "
        f"{_zeroed['observation_noise'].max():.2e}"
    )
    print(f"  the fall is a factor of {_drop:.1e}, {np.log10(_drop):.1f} orders of magnitude")
print(
    f"Level noise moves by a factor of "
    f"{float(kalman_param_summary['level_noise'].max() / kalman_param_summary['level_noise'].min()):.2f}"
    f" across folds; slope noise by "
    f"{float(kalman_param_summary['slope_noise'].max() / kalman_param_summary['slope_noise'].min()):.1f}"
)
_smoothness = kalman_df.group_by("fold").agg(pl.col("kalman_smoothness").median()).sort("fold")
print(
    f"Median kalman_smoothness by fold spans {_smoothness['kalman_smoothness'].min():.2e} to "
    f"{_smoothness['kalman_smoothness'].max():.2e}"
)

# %% tags=[]
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Fitted noise sizes", "Dollar-state run length"],
    horizontal_spacing=0.12,
)
for column, name, color in (
    ("observation_noise", "Observation noise R", COLORS["blue"]),
    ("level_noise", "Level noise", COLORS["amber"]),
    ("slope_noise", "Slope noise", COLORS["copper"]),
):
    fig.add_trace(
        go.Scatter(
            x=kalman_param_summary["fold"].to_list(),
            y=kalman_param_summary[column].to_list(),
            mode="lines+markers",
            name=name,
            line={"color": color},
        ),
        row=1,
        col=1,
    )
# The two panels share one legend, so the regime traces take colours the noise traces do
# not use; a repeated colour in a shared legend reads as a repeated series.
for column, name, color in (
    ("persist_high_vol", "Turbulent state, run length", COLORS["negative"]),
    ("persist_low_vol", "Calm state, run length", COLORS["positive"]),
):
    fig.add_trace(
        go.Scatter(
            x=hmm_param_df["fold"].to_list(),
            y=(1.0 / (1.0 - hmm_param_df[column])).to_list(),
            mode="lines+markers",
            name=name,
            line={"color": color, "dash": "dot"},
        ),
        row=1,
        col=2,
    )

# %% [markdown] tags=[]
# The two panels carry different units, so each axis is set on its own. The left one is
# labelled in powers of ten rather than in the SI prefixes plotly reaches for by default,
# which would render $10^{-6}$ as "1u" - a unit on an axis whose quantity is a variance and
# has none.

# %% tags=[]
fig.update_yaxes(
    title_text="Variance (log scale)",
    type="log",
    exponentformat="power",
    row=1,
    col=1,
)
fig.update_yaxes(title_text="Expected run length (sessions)", rangemode="tozero", row=1, col=2)
fig.update_xaxes(title_text="Fold (0 = most recent)", row=1, col=1)
fig.update_xaxes(title_text="Fold (0 = most recent)", row=1, col=2)
fig.update_layout(
    title=(
        "Every fitted parameter moves as the window rolls, one of them to zero"
        "<br><sup>Left: median across pairs of the three noise sizes, on a log axis."
        "<br>Right: one market-level fit per fold, read as 1/(1 - p) sessions."
        "<br>Both panels are per-fold parameters, not per-fold feature means.</sup>"
    ),
    height=460,
    margin={"t": 150},
)
show_plotly_with_alt(
    fig,
    "Two panels against fold number, with fold zero the most recent. On the left, the "
    "three fitted state-space noise sizes on a logarithmic axis: the level noise is the "
    "largest and holds a nearly flat line across folds, the slope noise sits many orders "
    "below it and also holds, while the observation noise tracks near the level noise "
    "for most folds and drops away by more than ten orders of magnitude on the few where "
    "the search puts it at zero. On the right, the expected run length of each "
    "dollar-regime state in sessions: the calm state starts far higher, falls steeply "
    "over the first few folds, and by the oldest folds the two states have converged to "
    "a similar length.",
)


# %% [markdown] tags=[]
# ## 8. Bring the Three Sets Together
#
# The three models produce frames of different shapes. The state-space and return models
# are fitted per pair, so their rows are keyed by pair, session and fold. The dollar-regime
# model is fitted once per fold on one market-wide series, so its rows carry no pair at all
# and the same two values attach to every pair on a session. Joining it therefore matches
# many rows to one, and the two per-pair frames match one to one; each join declares which
# it expects, so a shape that changed upstream stops the notebook here rather than
# multiplying rows quietly.
#
# `fold` is carried through as a key. It is not a feature and no model should be trained on
# it - it records which fit produced the row, so a downstream model can take the rows
# belonging to the fold it is working on.

# %% tags=[]
temporal_df = kalman_df.sort(["symbol", "timestamp", "fold"])

if len(hmm_df) > 0:
    temporal_df = temporal_df.join(hmm_df, on=["timestamp", "fold"], how="left", validate="m:1")

if len(arima_df) > 0:
    temporal_df = temporal_df.join(
        arima_df, on=["symbol", "timestamp", "fold"], how="left", validate="1:1"
    )

temporal_df = temporal_df.sort(["symbol", "timestamp", "fold"])

temporal_feature_cols = [c for c in temporal_df.columns if c not in {"timestamp", "symbol", "fold"}]
duplicate_keys = temporal_df.select(
    pl.struct("timestamp", "symbol", "fold").is_duplicated().sum()
).item()
assert duplicate_keys == 0, f"Duplicate temporal keys: {duplicate_keys}"

print(f"\nMerged features: {len(temporal_df):,} rows, {len(temporal_feature_cols)} columns")
print(f"Features: {temporal_feature_cols}")

# %% [markdown] tags=[]
# ### Where the Joins Left a Gap
#
# The duplicate-key check above catches a join that multiplies rows. A second failure runs
# the other way and that check cannot see it: a left join that finds no match writes a null
# and conserves the row count exactly, so the keys stay unique while the column stops being
# a measurement. The row count is the same either way, which is why this takes its own
# check.
#
# One gap is expected here and its shape is known. The dollar-regime model needs a
# 21-session window before it can report a volatility figure, so its series begins later
# than the price file does, and the oldest fold opens before it. Those rows are counted and
# reported rather than dropped, because the state-space and return columns on them are
# present and usable.
#
# What decides whether the gap matters is where it falls. A start-up gap reaches only into
# the opening of a fold's training window, so every validation row must carry every value,
# and that is what the check asserts. A null reaching a validation row would be a join key
# that did not match, wearing the shape of a start-up gap - and it would sit inside exactly
# the rows section 11 measures.

# %% tags=[]
val_windows = pl.DataFrame(
    [{"fold": f["fold"], "val_start": f["val_start"], "val_end": f["val_end"]} for f in folds]
)
validation_rows = temporal_df.join(val_windows, on="fold", how="inner").filter(
    pl.col("timestamp").is_between(pl.col("val_start"), pl.col("val_end"), closed="both")
)
null_census = temporal_df.select(
    [pl.col(c).null_count().alias(c) for c in temporal_feature_cols]
).to_dicts()[0]
val_nulls = validation_rows.select(
    [pl.col(c).null_count().alias(c) for c in temporal_feature_cols]
).to_dicts()[0]

for column, n_null in null_census.items():
    if not n_null:
        continue
    missing = temporal_df.filter(pl.col(column).is_null())
    print(
        f"  {column}: {n_null:,} null of {len(temporal_df):,} "
        f"({n_null / len(temporal_df):.3%}), {missing['timestamp'].n_unique()} sessions, "
        f"folds {sorted(missing['fold'].unique().to_list())}, "
        f"latest {missing['timestamp'].max()}"
    )
assert not any(val_nulls.values()), (
    f"a validation row is missing a feature value: { {c: n for c, n in val_nulls.items() if n} }"
)
print(
    f"Columns fully populated: {sum(1 for n in null_census.values() if not n)} "
    f"of {len(temporal_feature_cols)}; "
    f"nulls in the {len(validation_rows):,} validation rows: 0"
)

# %% [markdown] tags=[]
# ## 9. Write the Artifact
#
# Section 3 said in words that fold 0 is the most recent window and the highest-numbered
# fold the oldest. Downstream that sentence is load-bearing: a reader that takes a lower
# fold id for an earlier period joins every fold against the wrong end of the sample, and
# because the row counts and the schema are unaffected, nothing about the result looks
# wrong. A convention held only in prose is how that happens, so the last thing checked
# before the file is written is the ordering of the file itself, read back off the frame
# rather than off the split list it was built from.

# %% tags=[]
fold_spans = (
    temporal_df.group_by("fold")
    .agg(pl.col("timestamp").min().alias("first"), pl.col("timestamp").max().alias("last"))
    .sort("fold")
)
for _earlier, _later in zip(fold_spans.iter_rows(named=True), fold_spans[1:].iter_rows(named=True)):
    assert _later["last"] < _earlier["last"], (
        f"fold {_later['fold']} ends {_later['last']} and fold {_earlier['fold']} ends "
        f"{_earlier['last']}: the fold ids in this artifact are not ordered newest first, "
        f"so anything reading them positionally selects the wrong window"
    )
print(
    f"Fold ids run newest to oldest: fold {fold_spans['fold'][0]} covers "
    f"{fold_spans['first'][0]} to {fold_spans['last'][0]}, fold {fold_spans['fold'][-1]} "
    f"covers {fold_spans['first'][-1]} to {fold_spans['last'][-1]}."
)

# %% [markdown] tags=[]
# The parquet is written together with a short record beside it, in the same form
# `03_financial_features` writes beside its own matrix. The record holds a digest -- a short
# string computed from the file's contents, such that two files with the same values get
# the same string and any change to a value gets a different one. It is computed over the
# feature values rather than over the raw file bytes, so re-running the notebook and
# producing the same numbers leaves it where it was, while a changed fit moves it.
#
# The record also names what the values were built from, and it names two things rather
# than one: the prices the models were fitted on and run over, and the label file section 3
# cut the folds from. A label file rebuilt on refreshed data moves the fold boundaries and
# therefore moves the features, so recording the price digest alone would leave half of
# where these numbers came from unnamed.

# %% tags=[]
output_path = FEATURES_DIR / "model_based.parquet"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
record = write_artifact(
    temporal_df,
    output_path,
    keys=["timestamp", "symbol", "fold"],
    written_by="case_studies/fx_pairs/04_model_based_features.py",
    inputs={
        "load_fx_pairs:4h": value_digest(prices),
        f"labels:{PRIMARY_LABEL}": value_digest(label_frame),
    },
)
print(f"Saved: {output_path.relative_to(CASE_DIR)}")
print(f"  Shape: {temporal_df.shape}")
print(f"  Digest: {record['digest']}")
# %% [markdown] tags=[]
# ## 10. Take the Validation Rows, and Look at Them
#
# Everything from here on is about rows no model in this notebook was fitted on. Each
# fold's validation window is taken from that fold's own rows and the results stacked, so
# a session appears once, carrying the values a model that had never seen it produced.
# Training rows are a valid input to a downstream fit; they are not evidence about the
# feature, because the feature was shaped by them.
#
# The validation windows are contiguous - fold 7's ends the session before fold 6's begins
# - so stacking them gives a continuous run of sessions with no session counted twice,
# which the check below asserts rather than assumes.

# %% tags=[]
validation_frames = [
    temporal_df.filter(
        (pl.col("fold") == fold["fold"])
        & pl.col("timestamp").is_between(fold["val_start"], fold["val_end"], closed="both")
    )
    for fold in folds
]
eval_features = pl.concat(validation_frames).sort(["timestamp", "symbol"])
eval_duplicates = eval_features.select(
    pl.struct("timestamp", "symbol").is_duplicated().sum()
).item()
assert eval_duplicates == 0, f"Overlapping validation features: {eval_duplicates}"
print(
    f"Validation rows: {len(eval_features):,} across {eval_features['timestamp'].n_unique():,} "
    f"sessions, {eval_features['timestamp'].min()} to {eval_features['timestamp'].max()}"
)

# %% [markdown] tags=[]
# ### What the Regime Model Inferred, and What It Read
#
# Before any of this is scored, it is worth seeing one of the fitted quantities against
# the series it was inferred from. The figure draws the probability of the turbulent state
# on every validation session, with the 21-session dollar volatility the model read on the
# axis beneath it. The two should agree in shape without being the same line: the model is
# not thresholding volatility, it is asking which of two states makes the pair of numbers
# it saw most likely, given where it thought the market was yesterday.
#
# The dotted rules mark where one fold's model hands over to the next. Each stretch between
# them was produced by a separate fit on that fold's own training window, so a jump exactly
# at a rule is the two fits disagreeing rather than the market changing - which is the one
# thing this figure can show and the fit-stability panel cannot.

# %% tags=[]
if "hmm_regime_prob_high_vol" in eval_features.columns:
    hmm_validation = (
        eval_features.select("timestamp", "hmm_regime_prob_high_vol")
        .unique("timestamp", keep="last")
        .sort("timestamp")
        .drop_nulls()
        .join(usd_daily.select("timestamp", USD_VOL_COL), on="timestamp", how="left")
    )
    high_vol_share = float((hmm_validation["hmm_regime_prob_high_vol"] >= 0.5).mean())
    print(
        f"Validation sessions the model puts in the turbulent state with probability 0.5 "
        f"or more: {high_vol_share:.1%} of {len(hmm_validation):,}"
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.07,
    )
    fig.add_trace(
        go.Scatter(
            x=hmm_validation["timestamp"],
            y=hmm_validation["hmm_regime_prob_high_vol"],
            mode="lines",
            line={"color": COLORS["blue"], "width": 1.2},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hmm_validation["timestamp"],
            y=(hmm_validation[USD_VOL_COL] * 100),
            mode="lines",
            line={"color": COLORS["copper"], "width": 1.2},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["amber"], row=1, col=1)
    for fold in folds[:-1]:
        fig.add_vline(
            x=fold["val_start"].isoformat(), line_dash="dot", line_color=COLORS["neutral"]
        )
    fig.update_yaxes(title_text="P(turbulent)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(
        title_text=f"Dollar volatility, {USD_VOL_WINDOW}d (%)",
        rangemode="tozero",
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Validation session", row=2, col=1)
    fig.update_layout(
        title=(
            "The turbulent state switches on and off, tracking the volatility below"
            "<br><sup>Validation sessions only, every fold, one continuous run."
            "<br>Dotted rules are fold handovers: each stretch comes from a model fitted"
            "<br>on that fold's training window alone.</sup>"
        ),
        height=560,
        margin={"t": 130},
    )
    show_plotly_with_alt(
        fig,
        "Two stacked panels sharing a time axis of validation sessions. The upper panel "
        "is the probability the model assigns to the turbulent state, which spends most "
        "of its time pinned at zero or one and switches between them abruptly rather "
        "than drifting. The lower panel is the dollar volatility over the same "
        "sessions, and its sustained rises line up with the stretches the upper panel "
        "holds at one. Dotted vertical rules mark the fold handovers, and the series "
        "runs continuously across them.",
    )
else:
    high_vol_share = float("nan")
    print("Regime figure omitted: the reduced run produces no dollar series.")

# %% [markdown] tags=[]
# ## 11. Does Any of This Rank the Cross-Section?
#
# The screen asks one question of each emitted column: on those validation rows, does its
# rank correlation with the next session's return, measured across pairs on each session
# and then averaged, differ from zero? That statistic - the information coefficient - is
# the standard first test of whether a column can order a cross-section, and it is a
# diagnostic here rather than a decision: nothing is selected or dropped on it.
#
# `05_evaluation` joins these columns with the Chapter 8 financial set and measures what
# they add to it. That comparison needs both feature files and is made there.

# %% tags=[]
label_col = [c for c in label_frame.columns if c not in {"timestamp", "symbol"}][0]
label_endpoints = label_frame.sort(["symbol", "timestamp"]).with_columns(
    pl.col("timestamp").shift(-1).over("symbol").alias("_label_end")
)
eval_df = eval_features.join(label_endpoints, on=["timestamp", "symbol"], how="inner")
assert eval_df["_label_end"].max() < HOLDOUT_START, (
    "A validation decision resolves its label inside the holdout window"
)
print(
    f"Scored: {len(eval_df):,} rows over {eval_df['timestamp'].n_unique():,} sessions "
    f"against {label_col}; the last outcome resolves {eval_df['_label_end'].max()}"
)

# %% [markdown] tags=[]
# The join above pairs each validation row with the return the label file records for it,
# and the check beside it enforces the one boundary that binds a supervised comparison: a
# decision taken on the last validation session has an outcome that resolves a session
# later, and that outcome must fall before the holdout opens. This is the constraint
# section 2 set aside - the three fits read prices and never a label, so they are bound by
# the observation date; the moment a label enters, the date its outcome resolves binds too.
#
# The correlation is measured across pairs on each session, giving one number per session,
# and those are then averaged. Consecutive sessions share part of their outcome window, so
# the series is serially correlated and its ordinary standard error would be too small; a
# Newey-West correction widens it by the amount that overlap implies. That correction reads
# row order as time order, so the series must be sorted - `cross_sectional_ic_series` sorts
# on the date column, which is why it is used rather than a hand-rolled loop.
#
# Two kinds of missing value come back and both have to go. A session with fewer usable
# pairs than the floor returns a **null**: the correlation was not computed. A session on
# which the column takes the same value for every pair returns a **NaN**: the correlation
# is undefined, since there is nothing to rank. For the two dollar-regime columns, which
# are market-wide and therefore identical across pairs, every session is the second kind.
#
# Dropping only the nulls is not a smaller mistake than dropping neither. One NaN reaching
# the next cell becomes a NaN average, a NaN p-value, and then a NaN adjusted p-value for
# **every** column, because the multiple-testing procedure sorts the whole family together.
# The screen would return nothing and report it as nothing found - the same output a
# genuine negative result produces.

# %% tags=[]
temporal_ic = {}
for feature in temporal_feature_cols:
    ic_series = (
        cross_sectional_ic_series(
            eval_df.select("timestamp", "symbol", feature),
            eval_df.select("timestamp", "symbol", label_col),
            pred_col=feature,
            ret_col=label_col,
            date_col="timestamp",
            entity_col="symbol",
            method="spearman",
            min_obs=MIN_PAIRS_PER_DATE,
        )
        .drop_nulls("ic")
        .drop_nans("ic")
    )
    if len(ic_series) >= 20:
        temporal_ic[feature] = compute_ic_hac_stats(
            ic_series, ic_col="ic", label_horizon=LABEL_HORIZON_SESSIONS
        )

# %% [markdown] tags=[]
# Ten columns are tested at once, so a p-value read on its own is misleading: test enough
# columns against noise and one of them clears any fixed threshold by chance. The
# Benjamini-Hochberg procedure
# adjusts for how many tests were run, and what it controls is the share of the rejections
# that are expected to be false rather than the chance of any false rejection at all -
# which is the right trade when the purpose is to decide what is worth carrying forward.
#
# A run over a reduced set of pairs leaves no session above the pair floor and produces an
# empty frame, which the branch below handles rather than failing.

# %% tags=[]
feature_names = list(temporal_ic)
if feature_names:
    p_values = [temporal_ic[feature]["p_value"] for feature in feature_names]
    fdr_result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)
    eval_summary = pl.DataFrame(
        {
            "feature": feature_names,
            "ic_mean": [temporal_ic[f]["mean_ic"] for f in feature_names],
            "hac_se": [temporal_ic[f]["hac_se"] for f in feature_names],
            "naive_tstat": [temporal_ic[f]["naive_t_stat"] for f in feature_names],
            "hac_tstat": [temporal_ic[f]["t_stat"] for f in feature_names],
            "p_value": p_values,
            "adjusted_p": list(fdr_result["adjusted_p_values"]),
            "significant_fdr05": list(fdr_result["rejected"]),
        }
    ).sort(pl.col("ic_mean").abs(), descending=True)
    n_naive_sig = sum(abs(temporal_ic[f]["naive_t_stat"]) > 1.96 for f in feature_names)
    n_hac_sig = sum(p < 0.05 for p in p_values)
    n_fdr_sig = int(fdr_result["n_rejected"])
    print(
        f"Columns with a computable ranking: {len(feature_names)} of {len(temporal_feature_cols)}"
    )
    print(f"  clearing |t| > 1.96 with no overlap correction: {n_naive_sig}")
    print(f"  clearing p < 0.05 once the overlap is corrected for: {n_hac_sig}")
    print(f"  still clearing 0.05 after adjusting for ten tests: {n_fdr_sig}")
else:
    eval_summary = pl.DataFrame(
        schema={
            "feature": pl.String,
            "ic_mean": pl.Float64,
            "hac_se": pl.Float64,
            "naive_tstat": pl.Float64,
            "hac_tstat": pl.Float64,
            "p_value": pl.Float64,
            "adjusted_p": pl.Float64,
            "significant_fdr05": pl.Boolean,
        }
    )
    n_fdr_sig = 0
    print(f"Ranking omitted: no session reaches {MIN_PAIRS_PER_DATE} pairs in this run.")

# %% [markdown] tags=[]
# The screen in full, one row per column, largest in absolute size first - so a strong
# negative correlation sorts above a weaker positive one. Reading across a row:
# the average correlation, the corrected standard error behind it, what the uncorrected
# and corrected t-statistics would each have said, and what still clears the threshold once
# testing ten columns at once is adjusted for. The gap between the two t-statistics is how
# much the
# overlap between consecutive outcome windows was inflating the evidence.

# %% tags=[]
eval_summary

# %% [markdown] tags=[]
# ### What This Screen Can and Cannot Settle
#
# Two of the ten columns cannot appear in the chart at all. The dollar-regime columns take
# the same value for every pair on a session, so there is nothing for a cross-sectional
# correlation to rank; that is a property of a market-wide quantity and not a result about
# it. Such a column is used by conditioning on it - letting a model treat calm and
# turbulent sessions differently - and the test for that is a model, not a correlation.
#
# For the remaining eight, a bar is an estimate and the whiskers are what it is worth. A
# column whose interval spans zero has not shown it can rank pairs on this label over this
# period. That is a statement about this measurement, not a reason to delete the column:
# what a set of features contributes jointly, and what it adds over the Chapter 8 set, is
# measured in `05_evaluation`.

# %% [markdown] tags=[]
# Gold is painted only where the largest absolute estimate fails the adjusted threshold,
# so the
# sentence in the subtitle follows the same branch as the colour and cannot describe a
# chart the run did not draw.

# %% tags=[]
if len(eval_summary):
    plot_summary = eval_summary.sort("ic_mean")
    leader = eval_summary["feature"][0]
    bar_colors = [
        COLORS["positive"]
        if row["significant_fdr05"] and row["ic_mean"] > 0
        else COLORS["negative"]
        if row["significant_fdr05"]
        else COLORS["amber"]
        if row["feature"] == leader
        else COLORS["neutral"]
        for row in plot_summary.to_dicts()
    ]
    leader_significant = bool(eval_summary["significant_fdr05"][0])
    leader_note = (
        f"{leader} is the largest in absolute size and clears the adjusted threshold."
        if leader_significant
        else f"Gold marks {leader}, largest in absolute size; it does not clear the threshold."
    )
    ic_title = (
        "Some columns clear the threshold once ten tests are adjusted for"
        if n_fdr_sig
        else "No column clears the threshold once ten tests are adjusted for"
    ) + (
        f"<br><sup>{len(feature_names)} of {len(temporal_feature_cols)} columns can be"
        " ranked across pairs; the two dollar-regime"
        "<br>columns take one value per session and are not rankable."
        "<br>Whiskers are +/-1.96 overlap-corrected standard errors."
        f"<br>{leader_note}</sup>"
    )

# %% [markdown] tags=[]
# The bars carry no printed values: the table above already gives every estimate to three
# decimals, and repeating them on the chart competes with the whiskers for the same space.
# What the chart adds is the comparison - which columns lean the same way, and how much of
# each estimate is spanned by its own uncertainty.

# %% tags=[]
if len(eval_summary):
    fig = go.Figure(
        go.Bar(
            x=plot_summary["ic_mean"],
            y=plot_summary["feature"],
            orientation="h",
            marker_color=bar_colors,
            # Without the interval a bar three times the width of another reads as
            # three times the evidence.
            error_x={
                "type": "data",
                "array": (1.96 * plot_summary["hac_se"]).to_list(),
                "color": COLORS["neutral"],
                "thickness": 1.2,
                "width": 4,
            },
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
    fig.update_layout(
        title=ic_title,
        xaxis_title="Mean rank correlation with the next session's return",
        yaxis_title="",
        margin={"l": 180, "r": 60, "t": 140},
        height=520,
    )
    show_plotly_with_alt(
        fig,
        "Horizontal bars of the mean rank correlation between each model-derived column "
        "and the next session's return, ordered from the most negative at the bottom to "
        "the most positive at the top, each carrying a whisker of plus and minus 1.96 "
        f"overlap-corrected standard errors. {len(feature_names)} columns are shown. "
        "Every estimate is small against its own whisker, and the whiskers cross the "
        "zero rule the chart draws, so the bars are ordered by size without any of them "
        "standing clear of its own uncertainty.",
    )
else:
    print(f"Chart omitted: no session reaches {MIN_PAIRS_PER_DATE} pairs.")

# %% [markdown] tags=[]
# ## Key Takeaways
#
# The method, in the order a reader would apply it to their own data:
#
# 1. **Resolve the folds before fitting anything.** A model fitted first and assigned to a
#    window afterwards cannot be checked, because the window was chosen knowing the fit.
#    Deriving the boundaries from the same call the downstream consumer makes is what keeps
#    a fold id meaning one window on both sides of the join.
# 2. **Fit inside the training window, then hold the parameters still.** The recursion runs
#    on through validation because it has to carry its state forward; what must not move is
#    the parameters.
# 3. **Take the forward answer, not the more accurate one.** Every library that fits a
#    sequence model will happily report what it believes about a past session given the
#    whole series. That answer is the sharper one and it could not have been had at the time.
# 4. **Check it by truncation.** Delete the tail of the series, re-run, and compare the
#    values that remain. Nothing else distinguishes a forward pass from a backward one, and
#    the two produce columns of identical shape.
# 5. **Make the fit reproducible before trusting a digest.** A fixed seed is not enough
#    where the numerical work is divided across threads.
#
# **Known limitations**
#
# - Each model is re-estimated once per fold and then held fixed for the validation year
#   that follows, so a break occurring inside a validation window is read by parameters
#   estimated before it.
# - The dollar-regime model is fitted on one market-wide series, so its two columns take
#   the same value for every pair on a session and cannot order a cross-section.
# - `kalman_smoothness` inverts the uncertainty the state-space model attaches to its own
#   level estimate, and in a linear Gaussian model that quantity follows a recursion in the
#   noise parameters and the session index alone - it never reads a price. Within a fold the
#   column therefore ranks pairs by the noise their own fit estimated, not by anything that
#   happened in the window being scored.
# - The return model's order is fixed at $(1,0,1)$ for every pair and every fold rather than
#   chosen per pair, so its error measures surprise relative to one assumed dynamic.
# - Section 11 measures each column on its own. What the set contributes jointly, and what
#   it adds over the Chapter 8 features, is `05_evaluation`.

# %% [markdown] tags=["results"]
# **What the artifact holds, and what the validation screen found in it.**

# %% tags=[]
print(f"Feature columns written:  {len(temporal_feature_cols)}")
print(f"Walk-forward folds:       {len(folds)}")
if len(eval_summary):
    top_result = eval_summary.row(0, named=True)
    print(f"Columns rankable across pairs:            {len(feature_names)}")
    print(f"Clearing the adjusted 5% threshold:       {n_fdr_sig}")
    print(
        f"Largest in absolute size:                 {top_result['feature']} "
        f"({top_result['ic_mean']:+.4f}, t {top_result['hac_tstat']:+.2f})"
    )
else:
    print(
        f"Ranking omitted: a per-session rank correlation needs {MIN_PAIRS_PER_DATE} "
        "pairs on a session and no session in this run reaches that."
    )

# %% [markdown] tags=[]
# **Next**: [`05_evaluation.py`](05_evaluation.ipynb) reads this artifact together with
# the Chapter 8 financial features and measures what the two sets contribute together.
