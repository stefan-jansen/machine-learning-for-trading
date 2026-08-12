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
# # NASDAQ-100 microstructure: features a model has to be fitted to produce
#
# **Chapter 9: Model-Based Feature Extraction**
#
# The previous stage built features that are arithmetic on past bars - a spread, a realized
# variance, a share of volume. This one builds features that only exist once a model has been
# estimated: the number a reader gets depends on parameters fitted from data, so the window
# those parameters came from is part of what the feature knows. That is the whole subject of
# this notebook, and Section A is where it is argued.
#
# Three procedures run on the minute panel, in increasing distance from ordinary arithmetic:
#
# | Procedure | What it produces | What is estimated |
# |---|---|---|
# | HAR regression | a forecast of the next few minutes' variance, and the error in the last one | three regression coefficients, refitted every bar |
# | Rolling Fourier transform | how much of the recent activity sits at which frequency | nothing; a fixed transform of the window |
# | Depth-2 path signature | the order in which price, order flow and trade intensity moved | nothing; a fixed transform of the window |
#
# **What you will be able to do after working through it**
#
# - Split a volatility forecast into components measured over different lengths of history, and
#   refit it as time passes so that no coefficient is ever estimated from a bar the forecast is
#   supposed to precede.
# - Turn a rolling window of volume into a small set of numbers describing how repetitive the
#   recent activity has been, and read those numbers back.
# - Summarise a short stretch of price and order flow by which one moved first, in a form a
#   model can use as a column.
# - Write out a feature table whose rows carry the walk-forward window each one belongs to, and
#   check that the table covers every window every configured prediction target will ask for.
# - Measure whether any of it ranks the cross-section, with a standard error that accounts for
#   the dependence between neighbouring observations of a time series.
#
# **What it reads and what it writes**
#
# - Reads the AlgoSeek minute archive through `load_nasdaq100_bars`, and the label files under
#   `labels/` for their timestamps alone - those decide the walk-forward windows.
# - Writes `features/model_based.parquet`, one row per (`timestamp`, `symbol`, `fold`), with a
#   `.digest.json` sidecar beside it recording what was written and what it was built from.
#
# **Prerequisites**: [`02_labels`](02_labels.ipynb) must have run, because its output defines the
# windows here. [`03_financial_features`](03_financial_features.ipynb) need not have run; its
# output is joined against for a coverage count and is never read for a value.
#
# **The same methods, taught one at a time**:
# [`09_har_rough_volatility`](../../09_model_based_features/09_har_rough_volatility.ipynb),
# [`05_spectral_features`](../../09_model_based_features/05_spectral_features.ipynb),
# [`06_path_signatures`](../../09_model_based_features/06_path_signatures.ipynb).

# %%
"""NASDAQ-100 Microstructure: Model-Based Features (Ch9)."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series
from ml4t.diagnostic.splitters.calendar import TradingCalendar

from case_studies.utils.artifact_digest import value_digest, write_artifact
from data import load_nasdaq100_bars
from utils.artifact_specs import resolve_label_buffer, resolve_label_horizon
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.paths import get_case_study_dir
from utils.style import COLORS, show_plotly_with_alt

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
START_DATE = "2020-01-01"
END_DATE = "2021-12-31"
MAX_SYMBOLS = 0

# %% [markdown]
# ## Configuration
#
# Everything the run depends on is bound from `config/setup.yaml` rather than typed here, so a
# change to the case study's declared window or label set reaches this notebook without an edit.
# The four estimation windows below are this notebook's own model specification and are declared
# here for the same reason - once, where they can be read, rather than at each call site.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
FEATURES_DIR = CASE_DIR / "features"
LABELS_DIR = CASE_DIR / "labels"

SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
EVAL_CFG = load_evaluation_config(CASE_STUDY_ID)
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = SETUP["labels"]["buffer"]
CONFIGURED_LABELS = [PRIMARY_LABEL, *SETUP["labels"].get("variants", [])]
UNIVERSE = sorted(SETUP["universe"]["symbols"])
CALENDAR = EVAL_CFG["calendar"]
HOLDOUT_START = pd.Timestamp(EVAL_CFG["holdout_start"])
HOLDOUT_END = pd.Timestamp(EVAL_CFG["holdout_end"])
# The config states the holdout's last date; parsed as a timestamp it is that date at midnight,
# which is before every intraday bar of the session it names. Anything comparing a bar against the
# end of the holdout uses the exclusive bound instead.
HOLDOUT_END_EXCLUSIVE = HOLDOUT_END + pd.Timedelta(days=1)

# The bar is one minute, so a horizon written as a duration converts to a bar count exactly.
BAR = pd.Timedelta(minutes=1)
LABEL_HORIZON_BARS = int(pd.Timedelta(LABEL_BUFFER) // BAR)
IC_SAMPLE_STEP = LABEL_HORIZON_BARS

# The three lengths of history the HAR regression averages squared returns over, the amount of
# history each of its refits reads, and the window each spectrum and each signature path spans.
HAR_COMPONENTS = (5, 15, 60)
HAR_FIT_WINDOW = 120
FFT_WINDOW = 60
SIG_WINDOW = 30

# The exchange's regular session. Bars outside it are not part of any decision.
OPEN_HOUR, OPEN_MINUTE, CLOSE_HOUR = 9, 30, 16

# Every table below is short enough to read whole, and a frame shown as ten rows and an
# ellipsis is a table the reader has to take on trust.
pl.Config.set_tbl_rows(40)

# %%
print(f"Sample: {START_DATE} to {END_DATE}, minute bars on the {CALENDAR} calendar.")
print(
    f"The holdout runs {HOLDOUT_START.date()} to {HOLDOUT_END.date()}. Nothing is fitted on it "
    "and no number printed below is measured over it; features are still written for it, from "
    "windows that end before it, so that a model scored there has inputs."
)
print(
    f"Predictions are made for {PRIMARY_LABEL}, the return over the next {LABEL_BUFFER} "
    f"({LABEL_HORIZON_BARS} bars). The case study also configures "
    f"{', '.join(CONFIGURED_LABELS[1:])}, whose horizons differ, and each of them asks for a "
    "different walk-forward split - which is why Section B resolves one per label."
)
print(
    f"The HAR regression reads {HAR_COMPONENTS[0]}, {HAR_COMPONENTS[1]} and "
    f"{HAR_COMPONENTS[2]} minutes of squared returns and is refitted on the trailing "
    f"{HAR_FIT_WINDOW} bars: long enough for the four coefficients to be identified, short "
    "enough that the fit follows the day rather than the quarter."
)
print(
    f"Each spectrum spans {FFT_WINDOW} bars, which resolves periods up to an hour, and each "
    f"signature path spans {SIG_WINDOW} bars, which is the horizon over which order flow and "
    "price are expected to lead one another."
)
if MAX_SYMBOLS:
    print(f"Universe limited to the {MAX_SYMBOLS} symbols with the most bars.")

# %% [markdown]
# ## The panel this notebook reads
#
# The archive is one row per symbol and minute, carrying the closing quote on each side, the
# traded volume split by where each trade printed against the prevailing quote, and the volume
# reported away from the exchanges. Only the columns the three procedures consume are kept; the
# rest of the archive's sixty-odd columns would triple the memory this notebook holds and
# nothing here reads them.

# %%
READ = [
    "timestamp",
    "symbol",
    "close_bid_price",
    "close_ask_price",
    "volume",
    "finra_volume",
    "total_trades",
    "trade_at_bid",
    "trade_at_bid_mid",
    "trade_at_mid_ask",
    "trade_at_ask",
]

df = load_nasdaq100_bars(
    start_date=START_DATE,
    end_date=str(END_DATE),
    include_microstructure=True,
    symbols=UNIVERSE,
).select(READ)

if MAX_SYMBOLS:
    top_syms = (
        df.group_by("symbol")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(MAX_SYMBOLS)["symbol"]
        .to_list()
    )
    df = df.filter(pl.col("symbol").is_in(top_syms))
    print(f"Restricted to {MAX_SYMBOLS} symbols: {top_syms}")

_hour, _minute = pl.col("timestamp").dt.hour(), pl.col("timestamp").dt.minute()
df = df.filter(
    ((_hour > OPEN_HOUR) | ((_hour == OPEN_HOUR) & (_minute >= OPEN_MINUTE))) & (_hour < CLOSE_HOUR)
).with_columns(pl.col("timestamp").dt.date().alias("session_date"))

# %% [markdown]
# The vendor emits a padded 390-bar grid on every date, including the afternoons the exchange
# closes early. A bar after the close carries the last quote forward and no
# position could have been taken on it, so a feature computed for that minute is a feature for
# a time at which no decision existed. The session's real length comes from the exchange
# calendar, and the padding is dropped before anything is built, so nothing below describes a
# minute at which the exchange was not open.

# %%
_schedule = TradingCalendar(CALENDAR).calendar.schedule(start_date=START_DATE, end_date=END_DATE)
sessions = pl.DataFrame(
    {
        "session_date": [d.date() for d in _schedule.index],
        "session_bars": (
            (_schedule["market_close"] - _schedule["market_open"]).dt.total_seconds() // 60
        ).astype("int32"),
    }
)
SHORT = sessions.filter(pl.col("session_bars") < sessions["session_bars"].max())
_unscheduled = set(df["session_date"].unique()) - set(sessions["session_date"])
assert not _unscheduled, f"{len(_unscheduled)} session dates are not on the {CALENDAR} calendar"

_minute_of_day = (
    pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
    + pl.col("timestamp").dt.minute().cast(pl.Int32)
    - (OPEN_HOUR * 60 + OPEN_MINUTE)
)
_padded = df.height
df = (
    df.join(sessions, on="session_date", how="inner")
    .filter(_minute_of_day < pl.col("session_bars"))
    .drop("session_bars")
    .sort(["symbol", "timestamp"])
)
print(f"{sessions.height} scheduled sessions, {SHORT.height} of them early closes")
print(f"{_padded - df.height:,} padded bars dropped past the scheduled close")

# %% [markdown]
# What is in the panel, before anything is computed from it: how many names, how much history,
# and how much of a session an average name actually trades through.

# %%
sessions_seen = df.group_by("session_date").agg(pl.col("symbol").n_unique().alias("symbols"))
symbol_sessions = df.group_by("session_date", "symbol").agg(pl.len().alias("bars"))
display(
    pl.DataFrame(
        {
            "quantity": [
                "symbols",
                "sessions",
                "minute bars",
                "first bar",
                "last bar",
                "median bars a name trades in a session",
                "median names quoting per session",
            ],
            "value": [
                f"{df['symbol'].n_unique():,}",
                f"{df['session_date'].n_unique():,}",
                f"{df.height:,}",
                str(df["timestamp"].min()),
                str(df["timestamp"].max()),
                f"{symbol_sessions['bars'].median():,.0f}",
                f"{sessions_seen['symbols'].median():,.0f}",
            ],
        }
    )
)

# %% [markdown]
# ### The series the three procedures read
#
# The **mid price** is the average of the two sides of the quote, which is the price neither
# side of the market has paid a spread to reach. Its one-minute log change is the return series
# every volatility quantity below is built from, and it restarts at each session open so that
# the overnight move never enters a one-minute return.
#
# **Signed volume** is the volume that printed on the ask side minus the volume that printed on
# the bid side: positive when buyers were the ones crossing the spread. Divided by the volume it
# is counted over it becomes a share between -1 and 1, comparable across a heavily traded name
# and a thin one. That denominator has to be the traded volume on **both** venues, because the
# trade-location buckets the numerator comes from count every trade in the bar, including the
# ones reported to the FINRA trade reporting facility rather than to an exchange; `volume` alone
# counts the exchange prints and would make the share exceed one whenever off-exchange activity
# was heavy. The assertion below is what keeps that true rather than assumed.
#
# The **trade count** is the third dimension of the signature path in Section C.3, taken from
# the archive unchanged: how many separate trades made up the bar's volume, which distinguishes
# one large print from a hundred small ones.

# %%
TRADED_VOLUME = (pl.col("volume") + pl.col("finra_volume")).clip(lower_bound=1)
signed_vol = (pl.col("trade_at_ask") + pl.col("trade_at_mid_ask")) - (
    pl.col("trade_at_bid") + pl.col("trade_at_bid_mid")
)
mid = (pl.col("close_bid_price") + pl.col("close_ask_price")) / 2

df = df.with_columns(mid_close=mid)
df = df.filter(pl.col("mid_close").is_not_null() & (pl.col("mid_close") > 0))

group_cols = ["symbol", "session_date"]
df = df.with_columns(
    r1m=(pl.col("mid_close").log() - pl.col("mid_close").log().shift(1).over(group_cols)),
    signed_vol=signed_vol,
    signed_vol_share=(signed_vol / TRADED_VOLUME),
    bar_of_day=pl.col("timestamp").rank("ordinal").over(group_cols).cast(pl.Int32) - 1,
)

_worst = df["signed_vol_share"].abs().max()
assert _worst <= 1.0 + 1e-9, (
    f"signed volume share reaches {_worst:.3f}: the denominator does not cover the numerator"
)
print(f"Signed volume share stays inside [-1, 1]; the largest magnitude is {_worst:.3f}.")

# The digest of the panel as consumed, taken here so that it describes exactly the rows the
# three procedures read. It goes into the artifact's sidecar at the end of Section E.
RAW_DIGEST = value_digest(df.select(READ))
print(f"Minute panel digest: {RAW_DIGEST}")

# %% [markdown]
# ## A. Why a fitted feature is different
#
# A financial feature is a function of past bars. Written down, it is arithmetic: take the last
# thirty midpoints, take their standard deviation, that is the number. Two people with the same
# thirty bars get the same answer, and the answer for 10:45 does not change when 10:46 arrives.
#
# A model-based feature is a function of *parameters that were estimated from* bars. The HAR
# forecast for 10:45 is a weighted sum of three realized variances, and the weights came from a
# regression on some stretch of history. Change that stretch and the forecast changes, even
# though the three variances did not. So the feature's information set is not just the window it
# reads - it is that window **plus** every bar the parameters were estimated from.
#
# This is where look-ahead gets into a feature block without anyone writing anything obviously
# wrong. Fit the regression once on the whole sample and the weights carry information from the
# end of the sample into a forecast made at the beginning; every forecast is then partly a
# summary of what happened afterwards. The forecast will look good, and the backtest built on it
# will look better, and neither result was available to anyone at the time.
#
# The discipline that removes it is to make the estimation window part of the feature's
# definition and then keep that window behind the bar being described. Here it is kept behind by
# construction: the HAR is refitted at every bar on the immediately preceding stretch, so its
# weights at 10:45 were estimated from bars ending at 10:44 and the question of which fold the
# fit belonged to does not arise. The Fourier transform and the path signature estimate nothing
# at all - they are fixed transforms of a trailing window, and they are here because a reader
# who has met the hazard on the HAR should see what the same discipline costs when there is no
# parameter to place.
#
# Two consequences run through the rest of the notebook. First, the feature values do not depend
# on the walk-forward split, so the `fold` column written at the end selects rows rather than
# changing any of them - which is not true of a case study whose model is fitted once per fold,
# and Section E says what changes there. Second, nothing protects a *diagnostic* the same way:
# a number printed about the features is as capable of reading the holdout as a fitted parameter
# is. That is why the folds are resolved next, before anything is computed or printed.

# %% [markdown]
# ## B. The fold contract
#
# A walk-forward split cuts the history into a training window and the validation window that
# follows it, with a gap between them wide enough that the outcome of the last training decision
# has already been realized before the validation window opens. The width of that gap is the
# horizon of the thing being predicted, so **each prediction target gets its own split**: a
# five-minute return seals five minutes and a sixty-minute return seals an hour, and the two
# disagree about where the training window ends and where validation runs to.
#
# This case study configures more than one target and their horizons differ, so this notebook
# resolves one split per target rather than one for the notebook. Each is derived from that
# target's own label file, which is the same
# frame `load_modeling_dataset` uses downstream: fold boundaries are positions in a timestamp
# index, so deriving them from a different frame - the price panel, say, or a feature frame with
# its warm-up rows removed - moves every boundary by however many timestamps the two indexes
# differ by, and the artifact then answers a question about folds nobody downstream is asking.

# %%
label_splits: dict[str, list[dict]] = {}
label_timeline_digest: dict[str, str] = {}
for label in CONFIGURED_LABELS:
    label_path = LABELS_DIR / f"{label}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(f"{label} is configured but not built - run 02_labels.py first.")
    label_ts = pl.scan_parquet(label_path).select("timestamp").unique().collect()
    label_timeline_digest[label] = value_digest(label_ts)
    label_splits[label] = generate_cv_splits(
        label_ts,
        case_study_id=CASE_STUDY_ID,
        label_buffer=resolve_label_buffer(CASE_STUDY_ID, label, SETUP),
        outcome_horizon=resolve_label_horizon(CASE_STUDY_ID, label, SETUP),
        date_col="timestamp",
    )

splits = label_splits[PRIMARY_LABEL]
N_FOLDS = len(splits)
assert all(len(s) == N_FOLDS for s in label_splits.values()), (
    "the configured labels do not agree on how many folds there are"
)

display(
    pl.DataFrame(
        [
            {
                "label": label,
                "seals": resolve_label_buffer(CASE_STUDY_ID, label, SETUP),
                "fold": s["fold"],
                "train_start": s["train_start"],
                "train_end": s["train_end"],
                "val_start": s["val_start"],
                "val_end": s["val_end"],
            }
            for label, label_split in label_splits.items()
            for s in label_split
        ]
    ).sort(["fold", "label"])
)

# %% [markdown]
# The next cell executes the contract rather than describing it. The first check is that no
# training window runs into the validation window it is scored against. The second is the one
# that binds a supervised quantity: a validation bar at $t$ carries an outcome that resolves at
# $t + h$, so the last validation bar a target may use is $h$ before the holdout opens, not the
# bar before it. Both are checked for every configured target, because a split that holds for
# the fifteen-minute return can fail for the sixty-minute one.

# %%
for label, label_split in label_splits.items():
    seal = pd.Timedelta(resolve_label_buffer(CASE_STUDY_ID, label, SETUP))
    for s in label_split:
        assert pd.Timestamp(s["train_end"]) < pd.Timestamp(s["val_start"]), (
            f"{label} fold {s['fold']}: training window runs into its own validation window"
        )
        assert pd.Timestamp(s["val_end"]) + seal <= HOLDOUT_START, (
            f"{label} fold {s['fold']}: a validation outcome resolves inside the holdout"
        )
print(f"The contract holds for {N_FOLDS} folds on each of {len(label_splits)} targets.")

# %% [markdown]
# A row is emitted under a fold if it falls anywhere in that fold's window, and the window has
# to be wide enough for every target that will read it. So the emitted span for a fold runs from
# the earliest training start to the latest validation end across the targets. The differences
# are minutes, because the horizons are minutes, and minutes are exactly what a coverage check
# downstream is counting.

# %%
fold_window = {
    s["fold"]: (
        min(
            pd.Timestamp(x["train_start"])
            for sp in label_splits.values()
            for x in sp
            if x["fold"] == s["fold"]
        ),
        max(
            pd.Timestamp(x["val_end"])
            for sp in label_splits.values()
            for x in sp
            if x["fold"] == s["fold"]
        ),
    )
    for s in splits
}
for fold, (start, end) in sorted(fold_window.items()):
    print(f"  Fold {fold} emits {start} .. {end}")
print(
    f"  Fold {N_FOLDS} emits every bar from {min(w[0] for w in fold_window.values())} to "
    f"{HOLDOUT_END.date()}, and is trained on everything before the holdout opens."
)


# %%
def validation_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Restrict a frame to the validation windows of the primary target's folds.

    Every diagnostic in this notebook goes through this function. The feature frame carries no
    fold column of its own, so a readout built straight from it spans whatever the frame spans,
    holdout included. The primary target's windows are the right ones here because that is the
    target the readouts in Sections C, D and F are about.
    """
    return pl.concat(
        [
            frame.filter(
                (pl.col("timestamp") >= pd.Timestamp(s["val_start"]))
                & (pl.col("timestamp") <= pd.Timestamp(s["val_end"]))
            ).with_columns(pl.lit(s["fold"], dtype=pl.Int32).alias("fold"))
            for s in splits
        ]
    )


# %% [markdown]
# **Figure F1** draws what the artifact will contain. Each fold is a training span and the
# validation span that follows it; the top row is the extra fold written for the holdout
# period, whose training bars all lie before the holdout opens so that a model scored on the
# holdout has features for it without any of them having been built from it. The point to read
# off the figure is that no bar of any training span lies to the right of the rule.

# %%
spans = [
    (f"Fold {s['fold']}", kind, pd.Timestamp(s[f"{key}_start"]), pd.Timestamp(s[f"{key}_end"]))
    for s in splits
    for kind, key in (("Training bars", "train"), ("Validation bars", "val"))
]
spans += [
    (
        f"Fold {N_FOLDS}",
        "Training bars",
        min(w[0] for w in fold_window.values()),
        HOLDOUT_START,
    ),
    (f"Fold {N_FOLDS}", "Holdout bars", HOLDOUT_START, HOLDOUT_END_EXCLUSIVE),
]
span_colors = {
    "Training bars": COLORS["blue"],
    "Validation bars": COLORS["amber"],
    "Holdout bars": COLORS["recede"],
}

fig = go.Figure()
seen = set()
for row, kind, start, end in spans:
    fig.add_trace(
        go.Scatter(
            x=[start.isoformat(), end.isoformat()],
            y=[row, row],
            mode="lines",
            line={"width": 16, "color": span_colors[kind]},
            name=kind,
            legendgroup=kind,
            showlegend=kind not in seen,
        )
    )
    seen.add(kind)

fig.add_vrect(
    x0=HOLDOUT_START.isoformat(),
    x1=HOLDOUT_END_EXCLUSIVE.isoformat(),
    fillcolor=COLORS["recede"],
    opacity=0.12,
    line_width=0,
    layer="below",
)
fig.add_vline(x=HOLDOUT_START.isoformat(), line_dash="dash", line_color=COLORS["negative"])
fig.update_layout(
    title=(
        "Every fold trains left of the validation span it is scored on"
        "<br><sup>The dashed rule is where the holdout opens and the shaded region is the"
        "<br>holdout itself. The last fold is the one written so that a model scored on the"
        "<br>holdout has features there; its training bars all predate the rule. Spans overlap"
        "<br>between folds because the fold tag selects rows rather than changing values.</sup>"
    ),
    xaxis_title="Session",
    yaxis_title="",
    height=460,
    margin={"l": 90, "t": 140},
)
show_plotly_with_alt(
    fig,
    "Horizontal timeline with one row per fold on a session axis. Every row below the top is a "
    "validation fold: a long dark navy training bar followed by the shorter amber validation bar "
    "it is scored on. Fold 0 is the bottom row and holds the latest of those validation windows, "
    "and each validation row above it holds an earlier one, so their bars overlap. A dashed red "
    "rule marks where the holdout opens and a shaded band to its right is the holdout itself. "
    "The top row is the extra fold written for the holdout and has no validation bar: its "
    "training bar runs from the left edge up to the rule and its light grey holdout bar sits "
    "inside the band, later than every validation window. No training bar of any row crosses "
    "the rule.",
)

# %% [markdown]
# ## C. One section per model: what it infers, and why it cannot see ahead
#
# Each of the three procedures reads a trailing window and writes a value for the bar at the end
# of it. Those windows are counted in bars within a symbol, and a symbol's bars are the
# sessions laid end to end, so a window that is longer than the distance back to the session
# open reaches across the overnight gap. The return series itself does not - `r1m` is null at
# each open and enters the windows below as a zero - but the aggregation is not restarted, so a
# bar early in the session is described partly by yesterday afternoon.
#
# The share of rows this affects is the window length over the session length, which is worth
# measuring rather than asserting: `bar_of_day` is a row's position in its own session, so a row
# with `bar_of_day` below the window is one whose window crosses the gap. A production system
# would bound each window by the session. The approximation is kept here because it makes the cost of the shortcut visible and because it is the cost that
# grows with the window - which is the reason the fit window is the shortest one that identifies
# the regression rather than the longest one available.

# %%
_bod = validation_rows(df.select("timestamp", "bar_of_day"))
display(
    pl.DataFrame(
        {
            "window": ["signature path", "spectrum", "HAR components", "HAR fit window"],
            "bars": [SIG_WINDOW, FFT_WINDOW, HAR_COMPONENTS[-1], HAR_FIT_WINDOW],
            "share of rows reaching across a session gap": [
                f"{_bod.select((pl.col('bar_of_day') < w).mean()).item():.1%}"
                for w in (SIG_WINDOW, FFT_WINDOW, HAR_COMPONENTS[-1], HAR_FIT_WINDOW)
            ],
        }
    )
)
del _bod

# %% [markdown]
# ### C.1 HAR: a variance forecast built from three lengths of history
#
# Realized volatility is persistent, and it is persistent at more than one time scale at once:
# what happened in the last five minutes, the last quarter of an hour and the last hour all say
# something, and they do not say the same thing. The heterogeneous autoregressive model (Corsi,
# 2009) is the simplest way to use all three - a linear regression of the next period's realized
# variance on the realized variance measured over each of those three lengths:
#
# $$RV_{t+1}^{(5)} = c + \beta_5 \, RV_t^{(5)} + \beta_{15} \, RV_t^{(15)} + \beta_{60} \, RV_t^{(60)} + \varepsilon_{t+1}$$
#
# Corsi's original components are a day, a week and a month, on the argument that different
# participants look at different lengths of history. On a minute grid the same argument gives
# minutes, quarter hours and hours, which is what the three components here are.
#
# Two features come out. The **forecast** is the fitted right-hand side, the model's statement
# about the variance of the next few minutes. The **residual** is what the last such statement
# got wrong - realized variance minus the forecast made for it - and it is the more interesting
# of the two, because a large positive residual is variance that arrived without the recent past
# implying it: a news arrival, a liquidity event, something the persistence did not contain.
#
# Realized variance at horizon $w$ is the average squared one-minute return over the $w$ bars
# **before** $t$, so the value at $t$ never includes the bar at $t$.


# %%
def build_har_features_intraday(
    r1m: np.ndarray, components: tuple[int, int, int] = HAR_COMPONENTS
) -> dict[str, np.ndarray]:
    """Realized variance at each of the three HAR horizons.

    Every window ends at ``t`` exclusive, so the value at ``t`` is a function of bars strictly
    before ``t``.
    """
    n = len(r1m)
    r2 = r1m**2
    window_5, window_15, window_60 = components

    rv_5 = np.full(n, np.nan)
    rv_15 = np.full(n, np.nan)
    rv_60 = np.full(n, np.nan)

    for t in range(window_60, n):
        rv_5[t] = np.mean(r2[t - window_5 : t])
        rv_15[t] = np.mean(r2[t - window_15 : t])
        rv_60[t] = np.mean(r2[t - window_60 : t])

    return {"rv_5m": rv_5, "rv_15m": rv_15, "rv_60m": rv_60}


# %% [markdown]
# The regression is refitted at every bar on the immediately preceding stretch of history. That
# is the discipline Section A described, applied at the finest cadence available: the
# coefficients used to describe bar $t$ come from a regression whose last observation is bar
# $t-1$, so there is no window in which a parameter and the bar it describes share information.
# A refit that reads fewer than twenty usable observations is skipped rather than fitted on
# whatever survived, and the bar keeps a null.


# %%
def fit_har_rolling(
    rv_5: np.ndarray,
    rv_15: np.ndarray,
    rv_60: np.ndarray,
    fit_window: int = HAR_FIT_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit the HAR at every bar on ``[t - fit_window, t)`` and forecast one step ahead.

    Returns the forecast of the short-horizon realized variance, the error in the forecast made
    for the current bar, and the rolling coefficients.
    """
    n = len(rv_5)
    har_forecast = np.full(n, np.nan)
    har_residual = np.full(n, np.nan)
    har_betas = np.full((n, 4), np.nan)

    for t in range(fit_window + 1, n):
        start = t - fit_window
        y_train = rv_5[start + 1 : t + 1]
        X_train = np.column_stack(
            [
                np.ones(fit_window),
                rv_5[start:t],
                rv_15[start:t],
                rv_60[start:t],
            ]
        )

        valid_mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
        if valid_mask.sum() < 20:
            continue

        y_fit = y_train[valid_mask]
        X_fit = X_train[valid_mask]

        try:
            beta = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        har_betas[t] = beta

        x_t = np.array([1.0, rv_5[t], rv_15[t], rv_60[t]])
        if np.all(np.isfinite(x_t)):
            har_forecast[t] = x_t @ beta
            if np.isfinite(rv_5[t]):
                har_residual[t] = (
                    rv_5[t] - har_forecast[t - 1] if np.isfinite(har_forecast[t - 1]) else np.nan
                )

    return har_forecast, har_residual, har_betas


# %% [markdown]
# One symbol at a time: build the three regressors, roll the fit across them, and keep the
# coefficients as well as the features. The coefficients are thinned to one per symbol-session
# before they leave the function, because Section D asks how the fit moves over months and
# twenty million rows of it would answer that no better than fifty thousand.


# %%
def compute_har_per_symbol(
    symbol_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """HAR features and rolling coefficients for one symbol's sessions."""
    r1m = symbol_df["r1m"].to_numpy().copy()
    r1m = np.nan_to_num(r1m, nan=0.0)

    har_regs = build_har_features_intraday(r1m)
    har_forecast, har_residual, har_betas = fit_har_rolling(
        har_regs["rv_5m"],
        har_regs["rv_15m"],
        har_regs["rv_60m"],
        fit_window=HAR_FIT_WINDOW,
    )

    features = pl.DataFrame(
        {
            "timestamp": symbol_df["timestamp"],
            "symbol": symbol_df["symbol"],
            "har_rv5_pred": har_forecast,
            "har_residual": har_residual,
        }
    )

    betas = (
        pl.DataFrame(
            {
                "timestamp": symbol_df["timestamp"],
                "symbol": symbol_df["symbol"],
                "session_date": symbol_df["session_date"],
                "bar_of_day": symbol_df["bar_of_day"],
                "beta_5": har_betas[:, 1],
                "beta_15": har_betas[:, 2],
                "beta_60": har_betas[:, 3],
            }
        )
        .filter(pl.col("bar_of_day") == pl.col("bar_of_day").max().over("session_date"))
        .drop("bar_of_day")
        .with_columns(pl.col("^beta_.*$").fill_nan(None))
        .drop_nulls(["beta_5", "beta_15", "beta_60"])
    )

    return features, betas


# %%
symbols = df["symbol"].unique().sort().to_list()
har_results = []
beta_results = []

for i, sym in enumerate(symbols):
    sym_df = df.filter(pl.col("symbol") == sym).sort("timestamp")
    result, betas = compute_har_per_symbol(sym_df)
    har_results.append(result)
    beta_results.append(betas)
    if (i + 1) % 20 == 0 or (i + 1) == len(symbols):
        print(f"  HAR: {i + 1}/{len(symbols)} symbols processed")

har_df = pl.concat(har_results)
har_beta_df = pl.concat(beta_results)
del har_results, beta_results

for c in ["har_rv5_pred", "har_residual"]:
    har_df = har_df.with_columns(pl.col(c).fill_nan(None))

print(
    f"HAR: {har_df['har_rv5_pred'].drop_nulls().len():,} forecasts on {har_df.height:,} bars, "
    f"from {har_beta_df.height:,} retained fits."
)

# %% [markdown]
# There is no single representative set of coefficients to quote: the model is refitted every
# bar, so the object is the distribution of those fits. The medians below say which of the three
# horizons the fit leans on, and their sum says how much of a variance shock the model expects
# to still be there next period. They are taken over validation rows, like every other readout
# here.

# %%
display(
    validation_rows(har_beta_df).select(
        pl.col("beta_5", "beta_15", "beta_60").median().round(4).name.suffix("_median"),
        (pl.col("beta_5") + pl.col("beta_15") + pl.col("beta_60"))
        .median()
        .round(4)
        .alias("persistence_median"),
        pl.len().alias("fits"),
    )
)

# %% [markdown]
# **The forecast is an unconstrained linear extrapolation, and it shows.** The HAR is a
# regression on a variance with nothing in it that keeps a prediction non-negative. When a
# symbol's realized variance jumps well outside the range the trailing window was fitted on -
# a single-name event, an earnings gap, a halt - the fit extrapolates and the forecast can land
# below zero. The distribution below is reported as quantiles rather than as a mean and a
# standard deviation, because those two are set by a handful of such rows and describe nothing
# a reader can use. Modelling the logarithm of the variance is the standard remedy; leaving the
# forecast unconstrained here is what makes the failure mode visible instead of quietly clipped.

# %%
_fc = validation_rows(har_df.select("timestamp", "har_rv5_pred"))["har_rv5_pred"].drop_nulls()
display(
    pl.DataFrame(
        {
            "statistic": [
                "rows",
                "1st percentile",
                "median",
                "99th percentile",
                "minimum",
                "share below zero",
            ],
            "value": [
                f"{len(_fc):,}",
                f"{_fc.quantile(0.01):.3e}",
                f"{_fc.median():.3e}",
                f"{_fc.quantile(0.99):.3e}",
                f"{_fc.min():.3e}",
                f"{(_fc < 0).mean():.2%}",
            ],
        }
    )
)
del _fc

# %% [markdown]
# **Figure F2** shows what the fitted model actually inferred. The forecast is drawn against the
# realized variance it was forecasting, on validation rows only, with the boundaries between
# consecutive validation windows marked. Both series are the cross-sectional median across symbols
# within a session, because one symbol's minute-level realized variance is far too noisy to read
# over a year.
#
# The realized series is not fetched from anywhere: it is reconstructed from what this notebook
# emits, since `har_residual[t] = rv_5[t] - har_forecast[t-1]` rearranges to
# `rv_5[t] = har_forecast[t-1] + har_residual[t]`. Reading the two columns back that way is also
# a check that they mean what the docstring says they mean.

# %%
har_view = (
    validation_rows(har_df.select("timestamp", "symbol", "har_rv5_pred", "har_residual"))
    .sort(["symbol", "timestamp"])
    .with_columns(realized=pl.col("har_rv5_pred").shift(1).over("symbol") + pl.col("har_residual"))
    .with_columns(session=pl.col("timestamp").dt.date())
    .group_by("session")
    .agg(
        pl.col("har_rv5_pred").median().alias("forecast"),
        pl.col("realized").median().alias("realized"),
    )
    .sort("session")
)
print(f"Validation sessions drawn: {len(har_view):,}")

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=har_view["session"],
        y=har_view["realized"],
        mode="lines",
        name="Realized 5-bar variance",
        line={"color": COLORS["blue"], "width": 1.5},
    )
)
fig.add_trace(
    go.Scatter(
        x=har_view["session"],
        y=har_view["forecast"],
        mode="lines",
        name="HAR forecast",
        line={"color": COLORS["amber"], "width": 2},
    )
)
for s in sorted(splits, key=lambda s: pd.Timestamp(s["val_start"]))[1:]:
    fig.add_vline(
        x=pd.Timestamp(s["val_start"]).isoformat(),
        line_dash="dot",
        line_color=COLORS["neutral"],
    )
fig.update_layout(
    title=(
        "The HAR forecast tracks realized variance closely and overshoots its peaks"
        "<br><sup>Cross-sectional median across symbols per session, validation rows only."
        "<br>Each dotted rule is a boundary between consecutive validation windows. Both series"
        "<br>are means of squared one-minute log returns.</sup>"
    ),
    xaxis_title="Session",
    yaxis_title="Mean squared 1-minute log return",
    height=440,
    margin={"t": 120},
)
show_plotly_with_alt(
    fig,
    "Two lines over validation sessions on a shared axis of mean squared one-minute log return. "
    "A thin dark navy line is realized 5-bar variance, spiky, with occasional tall isolated "
    "peaks. A thicker amber line is the HAR forecast, tracking the same path closely and rising "
    "above the navy line at every peak. A single dotted vertical rule near the middle is the "
    "boundary between the two consecutive validation windows.",
)

# %% [markdown]
# ### C.2 A rolling spectrum of volume and of variance
#
# Intraday activity is repetitive. Volume is heavy at the open, thins through the middle of the
# day and picks up into the close, and that shape repeats every session; volatility clusters at
# its own frequencies. A Fourier transform of a trailing window is a way of asking how much of
# the recent activity sits at which repetition rate, and it produces conditioning features - not
# a prediction of direction, but a description of what kind of hour this is.
#
# Four numbers come out of each window. **Spectral energy** is how much variation there is in
# total once the level is removed. The **dominant period** is the repetition length carrying the
# most of it, in bars. **Spectral entropy** is how evenly the variation is spread across
# frequencies: low when one rhythm dominates, high when the window is closer to noise. The
# **low-frequency ratio** is the share sitting at periods longer than twenty bars, which
# separates a slow drift in activity from minute-to-minute churn.
#
# Each window ends at $t$ exclusive, so nothing at or after the bar being described enters its
# own spectrum.


# %%
def rolling_fft_features(
    signal: np.ndarray,
    window: int = FFT_WINDOW,
) -> dict[str, np.ndarray]:
    """Four descriptions of the power spectrum of each trailing window of *signal*."""
    n = len(signal)
    spectral_energy = np.full(n, np.nan)
    dominant_period = np.full(n, np.nan)
    spectral_entropy = np.full(n, np.nan)
    low_freq_ratio = np.full(n, np.nan)

    for t in range(window, n):
        segment = signal[t - window : t]

        if np.all(np.isnan(segment)) or np.nanstd(segment) < 1e-12:
            continue

        seg_clean = np.nan_to_num(segment, nan=0.0)
        seg_clean = seg_clean - seg_clean.mean()

        fft_vals = np.fft.rfft(seg_clean)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(window)

        total_power = np.sum(power[1:])
        if total_power <= 0:
            continue

        spectral_energy[t] = total_power

        dom_idx = np.argmax(power[1:]) + 1
        if freqs[dom_idx] > 0:
            dominant_period[t] = 1.0 / freqs[dom_idx]

        p_norm = power[1:] / total_power
        p_norm = p_norm[p_norm > 0]
        spectral_entropy[t] = -np.sum(p_norm * np.log(p_norm))

        low_mask = freqs[1:] < (1.0 / 20.0)
        if low_mask.any():
            low_freq_ratio[t] = np.sum(power[1:][low_mask]) / total_power

    return {
        "spectral_energy": spectral_energy,
        "dominant_period": dominant_period,
        "spectral_entropy": spectral_entropy,
        "low_freq_ratio": low_freq_ratio,
    }


# %% [markdown]
# The transform is run twice per symbol, on two different signals. Volume answers how structured
# the recent activity pattern was; squared returns answer the same question about volatility.
# Volume is passed through a logarithm first, because raw share counts span several orders of
# magnitude within a session and a spectrum of them is dominated by the largest few bars.


# %%
def compute_fft_per_symbol(
    symbol_df: pl.DataFrame,
    window: int = FFT_WINDOW,
) -> pl.DataFrame:
    """Spectral descriptions of trailing volume and squared-return windows for one symbol."""
    vol_raw = symbol_df["volume"].to_numpy().astype(float)
    vol_signal = np.log1p(np.clip(vol_raw, 0, None))

    r1m = symbol_df["r1m"].to_numpy().copy()
    r2_signal = np.nan_to_num(r1m, nan=0.0) ** 2

    vol_fft = rolling_fft_features(vol_signal, window=window)
    r2_fft = rolling_fft_features(r2_signal, window=window)

    return pl.DataFrame(
        {
            "timestamp": symbol_df["timestamp"],
            "symbol": symbol_df["symbol"],
            "vol_spectral_energy": vol_fft["spectral_energy"],
            "vol_dominant_period": vol_fft["dominant_period"],
            "vol_spectral_entropy": vol_fft["spectral_entropy"],
            "vol_low_freq_ratio": vol_fft["low_freq_ratio"],
            "rv_spectral_energy": r2_fft["spectral_energy"],
            "rv_dominant_period": r2_fft["dominant_period"],
            "rv_spectral_entropy": r2_fft["spectral_entropy"],
            "rv_low_freq_ratio": r2_fft["low_freq_ratio"],
        }
    )


# %%
fft_results = []

for i, sym in enumerate(symbols):
    sym_df = df.filter(pl.col("symbol") == sym).sort("timestamp")
    fft_results.append(compute_fft_per_symbol(sym_df, window=FFT_WINDOW))
    if (i + 1) % 20 == 0 or (i + 1) == len(symbols):
        print(f"  FFT: {i + 1}/{len(symbols)} symbols processed")

fft_df = pl.concat(fft_results)
del fft_results

fft_feature_cols = [c for c in fft_df.columns if c not in ["timestamp", "symbol"]]
for c in fft_feature_cols:
    fft_df = fft_df.with_columns(pl.col(c).fill_nan(None))

print(f"FFT: {fft_df['vol_spectral_energy'].drop_nulls().len():,} of {fft_df.height:,} bars.")

# %% [markdown]
# ### C.3 Path signatures: which moved first, price or flow
#
# The microstructure question this case study is built around is whether order flow leads price
# or price leads order flow. The distinction matters: flow arriving before a move is what an
# informed trade looks like, and a move arriving before flow is what liquidity chasing a price
# looks like, and they imply opposite things about whether the next minute continues.
#
# A **path signature** is a way of summarising a multi-dimensional path so that the order in
# which its dimensions moved is still readable off the summary. Take the three series - price,
# signed volume share, trade count - over a trailing window and treat them as one path through
# three dimensions. The signature is a sequence of iterated integrals of that path. Truncated at
# depth two it is $d + d^2$ numbers for a $d$-dimensional path: $d$ net displacements, one per
# dimension, and $d^2$ cross terms.
#
# The cross terms are the point. The term $S^{i,j}$ accumulates movement in dimension $j$
# weighted by how far dimension $i$ has already travelled, so it is large when $i$ moved first.
# `sig2_svs_ret` large means flow moved before price; `sig2_ret_svs` large means price moved
# first. Their difference is the asymmetry the question is about, and neither a correlation nor
# a lagged regression puts it in one number this way.
#
# Depth two has a closed form, so no library is needed.


# %%
def compute_depth2_signature(path: np.ndarray) -> np.ndarray:
    """The depth-2 truncated signature of a path of shape ``(T, d)``.

    Returns ``d`` net displacements followed by the ``d * d`` iterated integrals
    ``S^{i,j} = int int_{s<t} dX^i_s dX^j_t``, flattened row-major.

    The path is piecewise linear between samples, so a pair of increments contributes to
    ``S^{i,j}`` in two ways: whole earlier segments, ``dX^i_s dX^j_t`` for ``s < t``, and the
    half of each segment that lies below its own diagonal, ``0.5 dX^i_t dX^j_t``. Dropping the
    second is the difference between the signature and a strictly-lagged double sum, and it is
    visible in the diagonal: the identity below fails without it, and ``S^{i,i}`` goes negative
    whenever the increments partly cancel.
    """
    T, d = path.shape
    increments = np.diff(path, axis=0)

    sig1 = path[-1] - path[0]

    sig2 = np.zeros((d, d))
    cumsum = np.zeros(d)
    for t in range(len(increments)):
        sig2 += np.outer(cumsum, increments[t]) + 0.5 * np.outer(increments[t], increments[t])
        cumsum += increments[t]

    return np.concatenate([sig1, sig2.ravel()])


# %% [markdown]
# Two identities hold for the depth-2 signature of any path, whatever the path is, so they are
# what says the implementation computes a signature rather than something that resembles one.
# The diagonal is fixed by the net displacement alone, $S^{i,i} = \frac{1}{2}(\Delta X^i)^2$,
# which also makes it non-negative; and the shuffle relation
# $S^{i,j} + S^{j,i} = \Delta X^i \Delta X^j$ says the symmetric part carries no information
# beyond depth one, which is why the *antisymmetric* part is the feature worth reading.

# %%
_rng = np.random.default_rng(0)
for _trial in range(20):
    _p = np.cumsum(_rng.standard_normal((30, 3)), axis=0)
    _sig = compute_depth2_signature(_p)
    _s1, _s2 = _sig[:3], _sig[3:].reshape(3, 3)
    assert np.allclose(np.diag(_s2), 0.5 * _s1**2), (
        "depth-2 diagonal is not half the squared net move"
    )
    assert np.allclose(_s2 + _s2.T, np.outer(_s1, _s1)), "depth-2 shuffle identity fails"
print("Depth-2 signature identities hold on 20 random 3-dimensional paths.")


# %%
def _window_normalize(x: np.ndarray) -> np.ndarray:
    """Centre and scale a window by its own mean and standard deviation."""
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


# %% [markdown]
# The three dimensions arrive on wildly different scales - a log return near $10^{-4}$, a share
# between minus one and one, a trade count in the hundreds - and a signature of the raw path
# would be a description of those scales rather than of the path's shape. Each window is
# therefore standardised by its **own** mean and standard deviation before the path is built.
# That is what keeps the signature free of any quantity computed over the whole sample: the
# scale each window is put on comes from the window, never from a constant estimated across the
# symbol's history, which would be exactly the leak Section A is about.


# %%
def compute_signatures_per_symbol(
    symbol_df: pl.DataFrame,
    window: int = SIG_WINDOW,
) -> pl.DataFrame:
    """Rolling depth-2 signatures of the (return, signed volume share, trades) path."""
    r1m = symbol_df["r1m"].to_numpy().copy()
    svs = symbol_df["signed_vol_share"].to_numpy().copy()
    trades = symbol_df["total_trades"].to_numpy().astype(float).copy()

    r1m = np.nan_to_num(r1m, nan=0.0)
    svs = np.nan_to_num(svs, nan=0.0)
    trades = np.nan_to_num(trades, nan=0.0)

    n = len(r1m)
    d = 3
    n_features = d + d * d

    sig_features = np.full((n, n_features), np.nan)

    for t in range(window, n):
        seg_r = np.cumsum(_window_normalize(r1m[t - window : t]))
        seg_svs = np.cumsum(_window_normalize(svs[t - window : t]))
        seg_trades = np.cumsum(_window_normalize(trades[t - window : t]))

        path = np.column_stack([seg_r, seg_svs, seg_trades])

        if np.all(np.abs(np.diff(path, axis=0)) < 1e-12):
            continue

        sig_features[t] = compute_depth2_signature(path)

    dims = ["ret", "svs", "trd"]
    col_names = [f"sig1_{name}" for name in dims]
    col_names += [f"sig2_{name_i}_{name_j}" for name_i in dims for name_j in dims]

    result = {
        "timestamp": symbol_df["timestamp"],
        "symbol": symbol_df["symbol"],
    }
    for k, col_name in enumerate(col_names):
        result[col_name] = sig_features[:, k]

    return pl.DataFrame(result)


# %%
sig_results = []

for i, sym in enumerate(symbols):
    sym_df = df.filter(pl.col("symbol") == sym).sort("timestamp")
    sig_results.append(compute_signatures_per_symbol(sym_df, window=SIG_WINDOW))
    if (i + 1) % 20 == 0 or (i + 1) == len(symbols):
        print(f"  Signatures: {i + 1}/{len(symbols)} symbols processed")

sig_df = pl.concat(sig_results)
del sig_results

sig_feature_cols = [c for c in sig_df.columns if c not in ["timestamp", "symbol"]]
for c in sig_feature_cols:
    sig_df = sig_df.with_columns(pl.col(c).fill_nan(None))

print(f"Signatures: {sig_df['sig1_ret'].drop_nulls().len():,} of {sig_df.height:,} bars.")

# %% [markdown]
# ### C.4 Withholding the future changes nothing
#
# Every claim made so far about these three procedures is a claim that a value at $t$ is a
# function of bars before $t$. A notebook cannot establish that by agreeing with itself, so the
# check below computes the features a second time from a different input: the same code on a
# panel that stops at the holdout boundary. If any window, any fit or any normalisation reached
# forward, truncating the panel would move the values on the rows the two runs share.
#
# It runs on three symbols rather than the whole universe because the property is a property of
# the code, not of the sample, and three symbols is enough for a difference to appear. Exact
# equality is the bar - not a tolerance - because these are the same arithmetic on the same
# bars.

# %%
_check_symbols = symbols[:3]
_full = (
    har_df.join(fft_df, on=["timestamp", "symbol"], how="inner")
    .join(sig_df, on=["timestamp", "symbol"], how="inner")
    .filter(pl.col("symbol").is_in(_check_symbols) & (pl.col("timestamp") < HOLDOUT_START))
    .sort(["symbol", "timestamp"])
)
_truncated_parts = []
for sym in _check_symbols:
    _sym = df.filter((pl.col("symbol") == sym) & (pl.col("timestamp") < HOLDOUT_START)).sort(
        "timestamp"
    )
    _h, _ = compute_har_per_symbol(_sym)
    _truncated_parts.append(
        _h.join(compute_fft_per_symbol(_sym, window=FFT_WINDOW), on=["timestamp", "symbol"]).join(
            compute_signatures_per_symbol(_sym, window=SIG_WINDOW), on=["timestamp", "symbol"]
        )
    )
_truncated = (
    pl.concat(_truncated_parts)
    .with_columns(
        [pl.col(c).fill_nan(None) for c in _full.columns if c not in ("timestamp", "symbol")]
    )
    .sort(["symbol", "timestamp"])
    .select(_full.columns)
)
assert _truncated.equals(_full), (
    "a feature moved when the panel was truncated at the holdout boundary: something reads ahead"
)
print(
    f"{_full.height:,} rows on {len(_check_symbols)} symbols recomputed from a panel ending "
    f"{HOLDOUT_START.date()}; every value identical."
)
del _full, _truncated, _truncated_parts, df

# %% [markdown]
# ## D. Fit stability across folds
#
# Only one of the three procedures has parameters, and it is refitted every bar, so the question
# "do the parameters move as the window rolls" is a question about a distribution rather than
# about three numbers per fold. **Figure F3** shows that distribution, one box per component per
# fold, on validation rows.
#
# What to look for: a component whose coefficient sits away from zero in every fold is carrying
# the forecast, and a component centred on zero is telling you that the horizon it represents
# adds nothing at this frequency - which is a statement about the *sampling frequency*, not
# about the HAR, and would come out differently on daily bars. If the boxes moved sharply
# between folds, that would be an argument for refitting more often than the folds do; here the
# refit is already per bar, so what the figure decides is whether the three horizons are worth
# keeping as separate columns at all.
#
# The rolling regression is unconstrained on a variance, so a small number of fits run to three
# figures and the distribution has tails far beyond anything readable. The whiskers are drawn at
# the 5th and 95th percentiles for that reason, and the subtitle says so.

# %%
beta_long = validation_rows(
    har_beta_df.select("timestamp", "symbol", "beta_5", "beta_15", "beta_60")
).unpivot(
    index=["fold"],
    on=["beta_5", "beta_15", "beta_60"],
    variable_name="component",
    value_name="coefficient",
)

beta_summary = (
    beta_long.group_by(["fold", "component"])
    .agg(
        pl.col("coefficient").quantile(0.05).alias("p05"),
        pl.col("coefficient").quantile(0.25).alias("q25"),
        pl.col("coefficient").median().alias("median"),
        pl.col("coefficient").quantile(0.75).alias("q75"),
        pl.col("coefficient").quantile(0.95).alias("p95"),
        pl.len().alias("fits"),
    )
    .sort(["component", "fold"])
)
print(
    f"{beta_summary['fits'].sum():,} retained fits across "
    f"{beta_summary['fold'].n_unique()} folds, summarised in the figure below."
)

# %%
fig = go.Figure()
_component_colors = {
    "beta_5": COLORS["amber"],
    "beta_15": COLORS["copper"],
    "beta_60": COLORS["slate"],
}
for component, color in _component_colors.items():
    part = beta_summary.filter(pl.col("component") == component).sort("fold")
    fig.add_trace(
        go.Box(
            x=part["fold"].cast(pl.Utf8),
            lowerfence=part["p05"],
            q1=part["q25"],
            median=part["median"],
            q3=part["q75"],
            upperfence=part["p95"],
            name=component,
            marker_color=color,
        )
    )
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
fig.update_layout(
    title=(
        "The short-horizon component carries the volatility fit in every fold"
        "<br><sup>Distribution of the rolling regression coefficients, one fit retained per"
        "<br>symbol-session, on validation rows. Boxes span the quartiles and whiskers the"
        "<br>5th to 95th percentiles; the unconstrained fit has tails well beyond them.</sup>"
    ),
    xaxis_title="Fold",
    yaxis_title="Rolling regression coefficient",
    boxmode="group",
    height=440,
    margin={"t": 120},
)
show_plotly_with_alt(
    fig,
    "Grouped box plots with one group of three boxes per fold along the horizontal axis, "
    "coloured amber, copper and slate for the 5-, 15- and 60-minute components. A dashed rule "
    "marks zero. In every fold the amber 5-minute box sits highest and above the rule, while the "
    "copper and slate boxes for the two longer horizons sit lower and straddle it. Boxes span "
    "the quartiles and the whiskers the 5th to 95th percentiles.",
)

# %% [markdown] tags=["results"]
# ### What the coefficient distributions say
#
# The figure above reports, per fold, where the middle half of the fits for each component
# sits. Read the three medians against each other rather than as levels: what decides
# whether the three-horizon decomposition earns its columns is whether the two longer horizons
# are distinguishable from zero once the shortest one is in the regression, and whether that
# answer is the same in both folds. Read the persistence figure - the three medians added
# together - as how much of a variance shock the model expects to survive into the next period;
# a value near one is a near-random-walk variance and a value well below one is a variance the
# model expects to decay within minutes.

# %% [markdown]
# ## E. Combine and emit
#
# The three feature sets are joined on the panel key, the warm-up rows that no procedure could
# produce a value for are dropped, and the result is tagged with the folds resolved in Section
# B and written once.

# %%
temporal_df = har_df.join(fft_df, on=["timestamp", "symbol"], how="inner")
temporal_df = temporal_df.join(sig_df, on=["timestamp", "symbol"], how="inner")
del har_df, fft_df, sig_df

meta_cols = ["timestamp", "symbol"]
temporal_feature_cols = [c for c in temporal_df.columns if c not in meta_cols]

for col in temporal_feature_cols:
    temporal_df = temporal_df.with_columns(pl.col(col).fill_nan(None))

warmup_cols = ["har_rv5_pred", "vol_spectral_energy", "sig1_ret"]
temporal_clean = temporal_df.drop_nulls(subset=warmup_cols)
print(
    f"{len(temporal_feature_cols)} features on {temporal_clean.height:,} rows; "
    f"{temporal_df.height - temporal_clean.height:,} warm-up rows dropped."
)
del temporal_df

for col in temporal_feature_cols:
    temporal_clean = temporal_clean.with_columns(
        pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)
    )

# %% [markdown]
# What is in the feature block, measured on validation rows so that nothing here is a
# description of the holdout. The spread of each column is worth a look before it goes
# downstream: the signature terms are on a common scale by construction, the spectral energies
# are not, and a model that is sensitive to scale will need the standardisation that the
# modelling stage applies rather than this one.

# %%
_n = len(temporal_feature_cols)
_stats = (
    validation_rows(temporal_clean)
    .select(
        [pl.col(c).count().alias(f"n_{c}") for c in temporal_feature_cols]
        + [pl.col(c).median().alias(f"med_{c}") for c in temporal_feature_cols]
        + [
            (pl.col(c).quantile(0.95) - pl.col(c).quantile(0.05)).alias(f"spr_{c}")
            for c in temporal_feature_cols
        ]
    )
    .row(0)
)
display(
    pl.DataFrame(
        {
            "feature": temporal_feature_cols,
            "rows with a value": [f"{v:,}" for v in _stats[:_n]],
            "median": [f"{v:.3g}" for v in _stats[_n : 2 * _n]],
            "width of the 5th to 95th percentile range": [f"{v:.3g}" for v in _stats[2 * _n :]],
        }
    )
)
del _stats

# %% [markdown]
# The features are keyed the same way the labels and the Chapter 8 feature block are, so the
# three join without a rename. The intersection below is what a model is actually handed, and it
# is the number to carry forward: where it falls short of this notebook's own row count, the
# shortfall is rows one of the other two does not carry.

# %%
labels_path = LABELS_DIR / f"{PRIMARY_LABEL}.parquet"
features_path = FEATURES_DIR / "financial.parquet"

labels_keys = pl.scan_parquet(labels_path).select("timestamp", "symbol").collect()
temporal_keys = temporal_clean.select("timestamp", "symbol")
join_rows = {"labels": labels_keys.height, "this notebook": temporal_keys.height}
joined = temporal_keys.join(labels_keys, on=["timestamp", "symbol"], how="inner")
if features_path.exists():
    features_keys = pl.scan_parquet(features_path).select("timestamp", "symbol").collect()
    join_rows["financial features"] = features_keys.height
    joined = joined.join(features_keys, on=["timestamp", "symbol"], how="inner")
    del features_keys
join_rows["all of them"] = joined.height
display(
    pl.DataFrame(
        {
            "frame": list(join_rows),
            "rows": [f"{v:,}" for v in join_rows.values()],
        }
    )
)
del labels_keys, temporal_keys, joined

# %% [markdown]
# ### Tag each row with the walk-forward window it falls in
#
# Every row is written once per fold whose window contains it, and once more under the extra
# fold covering the holdout. A row therefore appears several times, under different fold tags,
# with identical feature values - which is correct here and worth being explicit about: these
# procedures estimate nothing per fold, so the tag says which model may read the row, not what
# the row contains. A case study whose model is fitted once per fold cannot do this, because
# there the same bar genuinely has a different value under each fold's parameters.
#
# The fold numbering follows what `load_modeling_dataset` expects: the walk-forward folds keep
# the numbers `generate_cv_splits` gave them, and the holdout fold takes the next number after
# them.

# %%
fold_frames = []
for fold, (start, end) in sorted(fold_window.items()):
    fold_df = temporal_clean.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
    ).with_columns(pl.lit(fold, dtype=pl.Int32).alias("fold"))
    fold_frames.append(fold_df)
    print(f"  Fold {fold}: {fold_df.height:,} rows")

earliest_train_start = min(w[0] for w in fold_window.values())
holdout_df = temporal_clean.filter(
    (pl.col("timestamp") >= earliest_train_start) & (pl.col("timestamp") < HOLDOUT_END_EXCLUSIVE)
).with_columns(pl.lit(N_FOLDS, dtype=pl.Int32).alias("fold"))
assert holdout_df.filter(pl.col("timestamp").dt.date() == HOLDOUT_END.date()).height > 0, (
    f"the holdout fold does not reach {HOLDOUT_END.date()}, the last date it is configured to cover"
)
fold_frames.append(holdout_df)
print(f"  Fold {N_FOLDS} (holdout): {holdout_df.height:,} rows")

temporal_with_folds = pl.concat(fold_frames)
del fold_frames, holdout_df
print(f"{temporal_with_folds.height:,} rows across {temporal_with_folds['fold'].n_unique()} folds")

# %% [markdown]
# The check that the fold tags mean what the downstream reader will take them to mean. For every
# configured target and every fold, take the decision times that target asks about inside the
# training and the validation window, and ask two separate questions of them. **Is it in the
# panel at all** - a bar with no quote produced no features and never will, and the count is
# reported rather than asserted on. **Was it tagged with this fold** - and that one has to be
# every single one, because a decision time the target asks for and this artifact answers under
# the wrong fold is precisely the defect that shows up three stages downstream as a coverage
# failure with no visible cause.

# %%
panel_ts = set(temporal_clean["timestamp"].unique().to_list())
emitted_ts = {
    fold: set(temporal_with_folds.filter(pl.col("fold") == fold)["timestamp"].unique().to_list())
    for fold in range(N_FOLDS)
}
coverage_rows = []
for label, label_split in label_splits.items():
    label_ts = (
        pl.scan_parquet(LABELS_DIR / f"{label}.parquet")
        .select("timestamp")
        .unique()
        .collect()["timestamp"]
        .sort()
    )
    for s in label_split:
        for window, start_key, end_key in (
            ("train", "train_start", "train_end"),
            ("validation", "val_start", "val_end"),
        ):
            start, end = pd.Timestamp(s[start_key]), pd.Timestamp(s[end_key])
            asked = label_ts.filter((label_ts >= start) & (label_ts <= end)).to_list()
            available = [t for t in asked if t in panel_ts]
            tagged = sum(t in emitted_ts[s["fold"]] for t in available)
            coverage_rows.append(
                {
                    "label": label,
                    "fold": s["fold"],
                    "window": window,
                    "decision times asked for": len(asked),
                    "in the panel": len(available),
                    "tagged with this fold": tagged,
                }
            )
coverage = pl.DataFrame(coverage_rows)
display(coverage.sort(["label", "fold", "window"]))
assert coverage.filter(pl.col("tagged with this fold") < pl.col("in the panel")).is_empty(), (
    "a configured target has a fold window this artifact does not fully cover"
)
print(
    f"Every fold window of all {len(label_splits)} configured targets is covered by the fold it "
    "is tagged under."
)
del emitted_ts, panel_ts

# %% [markdown]
# The artifact is written with a sidecar recording the digest of its values, its row count and
# its key columns, and the digests of what it was built from. The digest is over content rather
# than file bytes, so row order and parquet metadata leave it alone and any feature value moves
# it. Two things go in `inputs` here. The minute panel is the obvious one: every feature value
# came out of it. The second is the set of decision times each configured target carries - not
# its label values, none of which enters a feature, but its timeline, because that is what
# decided the fold boundaries. Move a target's decision times and the `fold` column this
# artifact ships moves with them.

# %%
record = write_artifact(
    temporal_with_folds,
    FEATURES_DIR / "model_based.parquet",
    keys=["timestamp", "symbol", "fold"],
    written_by="case_studies/nasdaq100_microstructure/04_model_based_features.py",
    inputs={
        "load_nasdaq100_bars": RAW_DIGEST,
        **{f"timeline/{label}": digest for label, digest in label_timeline_digest.items()},
    },
)
print(f"Wrote features/model_based.parquet, digest {record['digest']}, {record['n_rows']:,} rows")

# %%
_written = pl.scan_parquet(FEATURES_DIR / "model_based.parquet")
assert _written.select(pl.len()).collect().item() == temporal_with_folds.height
assert (
    temporal_with_folds.select(pl.struct("timestamp", "symbol", "fold").n_unique()).item()
    == temporal_with_folds.height
), "duplicate (timestamp, symbol, fold) key in the artifact"
assert set(temporal_with_folds["fold"].unique().to_list()) == set(range(N_FOLDS + 1))
print(f"Artifact reconciled: {temporal_with_folds.height:,} rows, {N_FOLDS + 1} folds")

# %% [markdown]
# ## F. Incremental evaluation: does a temporal feature rank the cross-section on its own?
#
# The question this stage has to answer before its output goes anywhere is whether the block
# adds anything. Answering it fully means comparing against the Chapter 8 features on the same
# rows, and that comparison is run in [`05_evaluation`](05_evaluation.ipynb), which loads both
# blocks. What is measured here is the half that needs no second block: whether each temporal
# feature ranks the cross-section against the outcome at all.
#
# The measurement is the information coefficient - the cross-sectional Spearman rank correlation
# between the feature and the realized return, computed at each decision time and then averaged
# over the series. **It selects nothing.** No feature is dropped on the strength of it and no
# decision downstream reads it; it is a screen that says which columns are worth looking at
# twice.
#
# Three things make the number mean what it says.
#
# 1. **Validation rows only.** The features carry no fold column of their own, so the frame goes
#    through `validation_rows` and the holdout is never scored. The assertion below is what
#    enforces it.
# 2. **The series is in date order.** A Newey-West correction treats row order as time order and
#    does not sort, while grouping a Polars frame returns groups in whatever order the operation
#    produced. A correction computed over a permutation of time reports no autocorrelation where
#    there is plenty. `cross_sectional_ic_series` sorts its dates, which is why the series comes
#    from it rather than from a loop.
# 3. **The observations do not overlap.** Timestamps are thinned to one per label horizon, so
#    consecutive observations are correlations of returns over disjoint windows. That removes
#    the mechanical floor overlapping windows would put under the bandwidth. Non-overlapping is
#    not the same as independent, so the bandwidth is still left to the Newey-West rule of thumb
#    rather than pinned, and the lag it chose is reported.

# %%
labels_primary = pl.read_parquet(labels_path)
eval_df = validation_rows(temporal_clean).join(
    labels_primary, on=["timestamp", "symbol"], how="inner"
)
del labels_primary
assert eval_df["timestamp"].max() < HOLDOUT_START, "the IC evaluation reached into the holdout"

sample_ts = eval_df["timestamp"].unique().sort().gather_every(IC_SAMPLE_STEP)
eval_sample = eval_df.join(sample_ts.to_frame("timestamp"), on="timestamp", how="semi")
print(
    f"{eval_df.height:,} labelled validation rows spanning {eval_df['timestamp'].min()} to "
    f"{eval_df['timestamp'].max()}, thinned to {len(sample_ts):,} decision times "
    f"({eval_sample.height:,} rows)."
)
del eval_df

# %% [markdown]
# A cross-section is only worth ranking if there is something to rank, so a decision time with
# fewer than ten names carrying both the feature and the outcome is skipped, and a feature whose
# series comes back shorter than twenty observations is not tested at all.

# %%
n_symbols = eval_sample["symbol"].n_unique()
min_cs_size = min(10, n_symbols)

ic_data = {}
for feat in temporal_feature_cols:
    frame = eval_sample.select("timestamp", "symbol", feat, PRIMARY_LABEL).drop_nulls()
    if frame.is_empty():
        continue
    ic_by_ts = cross_sectional_ic_series(
        frame,
        frame,
        pred_col=feat,
        ret_col=PRIMARY_LABEL,
        date_col="timestamp",
        entity_col="symbol",
        method="spearman",
        min_obs=min_cs_size,
    )
    if len(ic_by_ts) >= 20:
        ic_data[feat] = ic_by_ts

print(f"IC series computed for {len(ic_data)}/{len(temporal_feature_cols)} features")

# %% [markdown]
# ### A standard error that allows for dependence, and a correction for testing many columns
#
# Two adjustments stand between an average IC and a claim about it. Newey-West widens the
# standard error to account for the correlation between neighbouring observations of the series.
# Benjamini-Hochberg then controls the share of false discoveries among however many features
# are declared significant, which matters because testing this many columns at the five percent
# level would be expected to turn up a significant result or two from noise alone.

# %%
IC_LABEL_HORIZON = max(1, -(-LABEL_HORIZON_BARS // IC_SAMPLE_STEP))

hac_rows = []
for feat, ic_df in ic_data.items():
    stats = compute_ic_hac_stats(ic_df, ic_col="ic", label_horizon=IC_LABEL_HORIZON)
    stats["feature"] = feat
    hac_rows.append(stats)

if hac_rows:
    hac_df = pl.DataFrame(hac_rows)
    fdr_result = benjamini_hochberg_fdr(
        hac_df["p_value"].to_list(), alpha=0.05, return_details=True
    )
    hac_df = hac_df.with_columns(fdr_significant=pl.Series(fdr_result["rejected"].tolist()))
    hac_df = hac_df.sort(pl.col("mean_ic").abs(), descending=True)
else:
    hac_df = pl.DataFrame(
        schema={
            "feature": pl.Utf8,
            "mean_ic": pl.Float64,
            "hac_se": pl.Float64,
            "t_stat": pl.Float64,
            "p_value": pl.Float64,
            "naive_t_stat": pl.Float64,
            "fdr_significant": pl.Boolean,
        }
    )

# %%
n_tested = len(hac_df)
n_naive_sig = (
    int(hac_df.filter(pl.col("naive_t_stat").abs() > 1.96).shape[0]) if n_tested > 0 else 0
)
n_fdr_sig = int(hac_df.filter(pl.col("fdr_significant")).shape[0]) if n_tested > 0 else 0
if n_tested > 0:
    _hac_mean = hac_df["t_stat"].abs().mean()
    inflation = (
        round(float(hac_df["naive_t_stat"].abs().mean() / _hac_mean), 2)
        if _hac_mean and _hac_mean > 0
        else 1.0
    )
    _lag_lo = int(hac_df["effective_lags"].min())
    _lag_hi = int(hac_df["effective_lags"].max())
    hac_lags = f"{_lag_lo}" if _lag_lo == _lag_hi else f"{_lag_lo}-{_lag_hi}"
else:
    inflation = 1.0
    hac_lags = "none"

print(f"Features tested: {n_tested}")
print(f"Significant before any correction (|t| > 1.96): {n_naive_sig}")
print(f"Retained by Benjamini-Hochberg at 5%: {n_fdr_sig}")
print(f"Newey-West lags chosen by the automatic rule: {hac_lags}")
print(f"Ratio of uncorrected to corrected |t|: {inflation:.1f}x")

# %% [markdown]
# **Figure F4** draws the screen. Each bar is a feature's mean validation IC with its
# Newey-West interval; a bar is coloured when Benjamini-Hochberg retains the feature and left
# neutral when it does not. The interval is what stops the ordering from being read as a
# result: the features are sorted by point estimate, and most of those estimates cannot be told
# apart from zero.

# %%
if n_tested > 0:
    plot_ic = hac_df.sort("mean_ic")
    bar_colors = [
        COLORS["positive"]
        if row["fdr_significant"] and row["mean_ic"] > 0
        else COLORS["negative"]
        if row["fdr_significant"]
        else COLORS["neutral"]
        for row in plot_ic.to_dicts()
    ]
    ic_title = (
        "A few features rank the cross-section; the rest cannot be told from zero"
        if n_fdr_sig
        else "No temporal feature ranks the cross-section on its own"
    ) + (
        "<br><sup>Mean cross-sectional Spearman IC against the primary label on validation"
        "<br>rows, with Newey-West 95% intervals. Coloured bars are the features retained by"
        "<br>Benjamini-Hochberg at 5% across those tested.</sup>"
    )
    fig = go.Figure(
        go.Bar(
            x=plot_ic["mean_ic"],
            y=plot_ic["feature"],
            orientation="h",
            marker_color=bar_colors,
            error_x={
                "type": "data",
                "array": (1.96 * plot_ic["hac_se"]).to_list(),
                "color": COLORS["slate"],
                "thickness": 1,
            },
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
    fig.update_layout(
        title=ic_title,
        xaxis_title="Mean cross-sectional Spearman IC (validation folds)",
        yaxis_title="Feature",
        margin={"l": 180, "t": 120},
        height=580,
    )
    show_plotly_with_alt(
        fig,
        "Horizontal bar chart with one bar per temporal feature, feature names down the left and "
        "mean cross-sectional Spearman IC across the bottom, sorted from the most positive at "
        "the top to the most negative at the bottom. Every bar is short, within one hundredth of "
        "zero, and each carries a Newey-West 95% interval whisker. A dashed rule marks zero; most "
        "whiskers cross it and those bars are left dark slate, while the two features "
        "Benjamini-Hochberg retains are coloured, green for the positive IC at the top and red "
        "for the negative one at the bottom.",
    )
else:
    print("Validation IC chart omitted: too few symbols per timestamp to rank a cross-section.")

# %% [markdown] tags=["results"]
# ### What the validation screen found
#
# The count retained by Benjamini-Hochberg is printed above and drawn in F4. Read it against
# two other printed numbers rather than on its own.
#
# The first is the lag the correction ran at. Thinning to one decision per label horizon removed
# the overlap that would otherwise have forced a wide bandwidth, but non-overlapping returns are
# not independent returns, so the bandwidth was left to the automatic rule. A ratio of
# uncorrected to corrected $t$ near one then means something: the correction looked out over
# that many lags and found little left to widen the standard error by. Pinning the lag to one
# would have produced the same ratio while establishing nothing.
#
# The second is the number significant before any correction. The gap between that count and the
# count Benjamini-Hochberg retains is the price of having tested every column at once,
# and it is the reason a screen like this is reported as a count with a correction attached
# rather than as a ranked list.
#
# The reading to be careful about is a ratio near one on returns that **do** overlap. That is
# what a per-date series assembled by grouping produces, because the series reaches the
# correction in arbitrary order and reports no autocorrelation where there is plenty. It is a
# tell for a broken pipeline rather than for a clean one, and telling the two cases apart is
# what the thinning and the sorted series are for.
#
# **This screen selects nothing.** Whether this block adds anything over the Chapter 8 features
# is decided in `05_evaluation`, on the same validation rows.

# %% [markdown]
# ## Key takeaways
#
# **The estimation window is part of the feature.** A feature computed from fitted parameters
# knows everything those parameters were estimated from. Keeping that window behind the bar the
# feature describes is the whole discipline of this stage, and the cheapest way to keep it is to
# refit at the cadence of the data rather than once per fold - which also removes the question
# of which fold a value belongs to.
#
# **Prove causality against a second computation, not against yourself.** A notebook whose
# features read forward runs clean and agrees with itself. Recomputing on a panel that stops
# early and requiring exact agreement on the shared rows is a check that cannot pass by
# accident, and it costs three symbols' worth of runtime.
#
# **Resolve the walk-forward windows before anything is computed or printed.** These features
# carry no fold column of their own, so any readout built from the feature frame spans whatever
# the frame spans - the holdout included. Resolving folds first and routing every readout
# through one validation-only slice removes the whole class of accident, and it is a class that
# no test catches, because a diagnostic that reads the holdout emits nothing.
#
# **One target's walk-forward split is not another's.** The gap a split seals is the horizon of
# the thing being predicted, so a case study with several horizons has several splits. A feature
# artifact built for one of them silently under-covers the others, and the way that surfaces
# downstream is a coverage failure in a notebook three stages away rather than an error here.
# Derive the splits from the same frame the consumer derives them from, and check the coverage
# where the artifact is written.
#
# **A per-date IC series has to be in date order.** Newey-West treats row order as time order
# and does not sort; a Polars `group_by` returns groups in arbitrary order. Together they report
# a standard error computed over a permutation of time, and it looks like good news.
#
# ### Known limitations
#
# 1. **The variance forecast is unconstrained.** It is a linear regression on a variance with
#    nothing holding it above zero, so a shock outside the trailing window's range extrapolates
#    to a negative forecast. The share is small and measured in Section C.1, but the column's
#    mean and standard deviation are set by those rows and are not a description of the feature.
#    Modelling log-variance is the standard remedy.
# 2. **The aggregation windows cross session boundaries.** The one-minute returns are bounded by
#    the session, but the trailing windows the three procedures read are not, so a bar early in
#    a session is described partly by the previous afternoon. The affected share is measured at
#    the top of Section C and grows with the window.
# 3. **The signature is truncated at depth two.** Depth three would carry the order of three
#    dimensions rather than two, at $d^3$ further terms per path; whether that is worth
#    twenty-seven more columns on a three-dimensional path is a question this notebook does not
#    settle.
#
# **Next**: [`05_evaluation`](05_evaluation.ipynb) puts this block beside the Chapter 8 features
# on the same validation rows and screens both together.
