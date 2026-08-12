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
# # NASDAQ-100 Microstructure: Label Engineering
#
# Every model in this case study predicts the label defined here, so an error in it is
# silent where it is made and reaches every metric and every backtest after it. This
# notebook fixes the execution convention, proves each labelled row has a complete
# forward window inside one scheduled trading session, measures how much independent
# information those rows carry, establishes the floor a feature has to clear, and writes the
# label files the evaluation and modelling stages read.
#
# ## Learning objectives
#
# - Write an intraday forward return as an execution convention - which bar the position
#   opens on and which one it closes on - rather than as a row shift
# - Measure the bar grid the horizon is counted on, and convert a declared duration into
#   bars against it instead of assuming the two agree
# - Assert, rather than describe, that every labelled window is complete inside one session
# - Price the overlap in a per-bar label, both as decay and as an effective row count
# - Establish the floor a feature has to clear, under a standard error that prices in that
#   overlap
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 7, Section 7.2. Reads AlgoSeek NASDAQ-100 minute bars with NBBO quotes through
# `load_nasdaq100_bars()`, whose coverage
# [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) establishes, the NYSE trading
# calendar, and `config/setup.yaml`, which declares the universe, the label set, the
# horizons and the holdout boundary. Writes `labels/fwd_ret_5m.parquet`,
# `labels/fwd_ret_15m.parquet`, `labels/fwd_ret_60m.parquet` and
# `labels/fwd_dir_15m.parquet`, each with a `.digest.json` sidecar beside it.
# [`05_evaluation`](05_evaluation.ipynb) joins these files to the feature panel, and the
# modelling notebooks from `06_linear` on load the primary label through
# `utils.modeling.load_modeling_dataset`. `03_financial_features.py` reads none of them:
# it builds features from the bars and leaves the join to `05_evaluation`.

# %%
"""NASDAQ-100 Microstructure: Label Engineering."""

import warnings
from datetime import date, datetime, time, timedelta

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series
from ml4t.diagnostic.splitters.calendar import TradingCalendar
from ml4t.engineer.labeling import fixed_time_horizon_labels

from case_studies.utils.artifact_digest import value_digest, write_artifact
from case_studies.utils.label_diagnostics import effective_sample_size, panel_autocorrelation
from data import load_nasdaq100_bars
from utils.artifact_specs import resolve_label_horizon
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")

CASE_STUDY_ID = "nasdaq100_microstructure"
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
LABELS_DIR = CASE_DIR / "labels"

# %% [markdown]
# All three parameters are read below. `MAX_SYMBOLS` keeps a seed-deterministic subset of
# the universe and the two dates trim the history. Any of them shortens a run at the cost
# of a thinner panel: the rank correlation in Section G needs a wide cross-section on each
# decision minute, and the boundary profile in Section D needs whole sessions.

# %% tags=["parameters"]
MAX_SYMBOLS = 0
START_DATE = "2020-01-01"
END_DATE = "2021-12-31"

# %% [markdown]
# ## Configuration
#
# Everything that defines a label is declared in `config/setup.yaml` and bound here. A
# horizon or a boundary typed into a cell is a second copy of a value the rest of the
# pipeline reads from the file, and the two drift apart the first time either is edited.
#
# `resolve_label_horizon` returns each label's outcome horizon as a duration - `15min`
# rather than a number of rows - which is the form the rest of the notebook keeps it in.
# The flat band of the direction label is the friction floor the cost model declares:
# a move smaller than it does not pay for its own spread.

# %%
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())


def declared_horizon(label: str) -> timedelta:
    """The outcome horizon `setup.yaml` declares for *label*, as a duration."""
    spec = resolve_label_horizon(CASE_STUDY_ID, label, setup)
    return timedelta(minutes=int(spec.removesuffix("min")))


# %%
PRIMARY_LABEL = setup["labels"]["primary"]
LABEL_NAMES = [PRIMARY_LABEL, *setup["labels"].get("variants", [])]
RETURN_LABELS = [n for n in LABEL_NAMES if n.startswith("fwd_ret")]
DIRECTION_LABEL = next(n for n in LABEL_NAMES if n.startswith("fwd_dir"))
HORIZONS = {name: declared_horizon(name) for name in LABEL_NAMES}
PRIMARY_HORIZON = HORIZONS[PRIMARY_LABEL]
FLAT_BAND = setup["costs"]["friction_floor_bps"] / 10_000
CALENDAR = setup["evaluation"]["calendar"]
HOLDOUT_START = date.fromisoformat(setup["evaluation"]["holdout_start"])
HOLDOUT_TS = datetime.combine(HOLDOUT_START, time())
UNIVERSE = sorted(setup["universe"]["symbols"])
GROUP_COLS = ["symbol", "session_date"]
PALETTE = dict(zip(RETURN_LABELS, (COLORS["blue"], COLORS["amber"], COLORS["copper"])))

print(f"Labels {LABEL_NAMES}, primary {PRIMARY_LABEL}, flat band {FLAT_BAND:.2%}")
print(f"Universe: the {len(UNIVERSE)} names `setup.yaml` declares")
print(f"Sessions come from the {CALENDAR} calendar; holdout opens {HOLDOUT_START}")
print("Each label is sealed on its own endpoint, not on the bar it was observed from")

# %% [markdown]
# ## A. The learning task
#
# The hypothesis is that the order flow of the last few minutes says something about where
# a NASDAQ-100 name trades over the next few, and that whatever it says is small enough to
# be eaten by the cost of acting on it. The label therefore has to be a return an intraday
# trader could actually have captured, measured between two prices they could have traded
# at, and it has to be honest about the delay between seeing a signal and getting filled.
#
# The decision cadence comes from `setup.yaml`: a bar closes, that close is the last thing
# observed, and the position goes on at the next bar. The primary horizon is the middle of
# three - long enough that the move can exceed the spread, short enough to stay inside the
# session and inside the regime the microstructure features describe. The fast and slow
# variants ask the same question of a horizon a third as long and one four times longer,
# which is a question about how quickly the information decays relative to what it costs
# to trade on it. A direction variant discretises the primary label so the same hypothesis
# can be posed as a classification task.

# %% [markdown]
# ## B. Preparation before the label
#
# Three things have to be true of the price series before a forward window means anything.
#
# **The price has to measure where the market is, not which side happened to trade.** Trade
# prices alternate between bid and ask as buyers and sellers arrive, so a return taken
# between two of them carries a bounce that has nothing to do with information (Hasbrouck,
# 2007). The midpoint of the closing NBBO quote removes that bounce. It is not itself a
# price anyone transacts at - a marketable order crosses at the bid or the ask - so what is
# built from it is a midprice return, and the cost of crossing is charged against it
# separately in Section E, out of the half-spread of the same quote.
#
# **The window has to sit inside one session, and the session is the one the exchange
# scheduled.** An overnight gap is not an intraday move, so `session_date` joins `symbol` in
# the entity key and no label crosses either. Regular hours only: the pre-market and
# after-hours books are thin enough that their quotes describe a different market.
#
# Where the session ends cannot be read off the clock, because the vendor emits the same
# padded grid on every date. The half-sessions printed below close early and still carry
# bars out to the usual hour, quoting a price carried forward from before the close. Two
# things go wrong if those bars are kept, and only the first is visible: they get labels of
# their own, and - the one that survives any filter applied further down the pipeline - the
# genuine bars in the final `horizon` minutes before the close take their **exit** price
# from a quote that postdates it. No trade could have been closed at that price, so the
# return is not one anyone could have earned. The bound therefore comes from the exchange
# calendar and is applied before any label is built.
#
# **No eligibility filter runs before the label.** Once rows are dropped from inside a
# series a shift counts survivors rather than bars, and the window silently spans whatever
# was removed - which is the failure Section D exists to catch. The cost-feasible universe
# `setup.yaml` declares is applied at backtest time, not here.
#
# Only four columns are read out of the sixty the microstructure schema carries. The label
# needs the two quote sides and the keys, and projecting at the scan keeps a full-universe
# run inside a couple of gigabytes instead of the twenty-eight the whole schema costs.

# %%
_exchange = TradingCalendar(CALENDAR).calendar
_schedule = _exchange.schedule(start_date=START_DATE, end_date=END_DATE).apply(
    lambda col: col.dt.tz_convert(_exchange.tz).dt.tz_localize(None)
)
sessions = pl.DataFrame(
    {
        "session_date": [stamp.date() for stamp in _schedule.index],
        "session_open": _schedule["market_open"].to_list(),
        "session_close": _schedule["market_close"].to_list(),
    }
)
_scheduled = sessions["session_close"] - sessions["session_open"]
N_EARLY = int((_scheduled < _scheduled.max()).sum())
_lengths = ", ".join(sorted({str(length) for length in _scheduled}))
print(f"{sessions.height} {CALENDAR} sessions of length {_lengths}, {N_EARLY} closing early")

# %%
_clock = pl.col("timestamp").dt.time()
_widest = (sessions["session_open"].dt.time().min(), sessions["session_close"].dt.time().max())
_bid, _ask = pl.col("close_bid_price"), pl.col("close_ask_price")

padded = (
    load_nasdaq100_bars(
        start_date=START_DATE,
        end_date=END_DATE,
        include_microstructure=True,
        max_symbols=MAX_SYMBOLS,
        symbols=UNIVERSE,
        lazy=True,
    )
    .select(["timestamp", "symbol", "close_bid_price", "close_ask_price"])
    .filter((_clock >= _widest[0]) & (_clock < _widest[1]))
    .with_columns(
        ((_bid + _ask) / 2).alias("mid_close"),
        ((_ask - _bid) / (_bid + _ask)).alias("half_spread"),
        pl.col("timestamp").dt.date().alias("session_date"),
    )
    .collect()
)
bars = (
    padded.join(sessions, on="session_date", how="inner")
    .filter(pl.col("timestamp").is_between(pl.col("session_open"), pl.col("session_close"), "left"))
    .drop(["session_open", "session_close"])
)
quoted = bars.filter(pl.col("mid_close") > 0).sort([*GROUP_COLS, "timestamp"])

# %%
print(
    f"{padded.height:,} bars inside the widest scheduled window, {padded['symbol'].n_unique()} symbols"
)
print(f"{padded.height - bars.height:,} dropped past the scheduled close on {N_EARLY} early closes")
print(f"{bars.height - quoted.height:,} dropped for a missing or non-positive quote midpoint")
print(f"{quoted.height:,} quoted bars over {quoted['session_date'].n_unique():,} sessions")

# %% [markdown]
# The horizon is declared in minutes and applied to a frame of bars, so the spacing of
# those bars is what converts one into the other. A 15-row shift is a 15-minute return only
# on a one-minute grid, and the loader returns the raw partition without promising one. The
# spacing is therefore measured, the grid is required to be uniform inside a session, and
# every horizon is required to be a whole number of bars - at least two of them, so that the
# entry bar and the exit bar are different bars.
#
# Uniformity is a weaker property than it sounds and does not subsume the bound above: a
# padded grid is exactly uniform, so this assertion passes on an early close whether or not
# the padding was removed. The schedule is what removes it; this only checks the spacing.

# %%
_gap = pl.col("timestamp") - pl.col("timestamp").shift(1).over(GROUP_COLS)
spacing = quoted.select(_gap.drop_nulls().unique().alias("gap"))["gap"].to_list()
assert len(spacing) == 1, f"the intraday grid is not uniform: spacings {sorted(spacing)}"
BAR = spacing[0]
HORIZON_BARS = {name: horizon // BAR for name, horizon in HORIZONS.items()}
for name, horizon in HORIZONS.items():
    assert horizon % BAR == timedelta(0), f"{name}: {horizon} is not a whole number of {BAR} bars"
    assert HORIZON_BARS[name] >= 2, (
        f"{name}: a {horizon} horizon is {HORIZON_BARS[name]} bar on a {BAR} grid, so the "
        f"entry bar and the exit bar are the same bar and the label cannot be formed"
    )

print(f"Bar spacing {BAR}, uniform within every session; horizons in bars {HORIZON_BARS}")

# %% [markdown]
# ## C. Label construction
#
# One execution convention, written once and applied at all three horizons:
#
# $$r^{(H)}_{s,t} = \frac{M_{s,t+H}}{M_{s,t+B}} - 1$$
#
# where $M$ is symbol $s$'s quote midpoint, $B$ is one bar and $H$ is the declared horizon.
# The decision is taken on the bar closing at $t$, so the earliest bar that can be acted on
# is the one after it, and the position is held until $H$ after the decision. Both ends are
# quote midpoints observed inside the session, so what the formula measures is how far the
# market moved over the holding period; a trade capturing that move pays half the spread at
# each end, one full spread over the round trip, which Section E prices separately.
#
# All four labels are computed on the **minute** grid the data arrives on, and what differs
# between them is the horizon: five, fifteen and sixty minutes, plus a direction label cut
# from the fifteen-minute return. $B$ is therefore one minute, and a horizon of $H$ minutes
# is a shift of $H$ rows only where the minute grid is complete, which is what the section
# above measures and Section D asserts. Chapter 16 rebalances this case study on a coarser
# schedule than the one the labels are built on; reconciling the two is that chapter's
# problem and no claim is made here about how the two line up.
#
# The entry price is the next bar's midpoint, which the uniform grid asserted above makes
# exactly one bar of wall-clock time later; the last bar of a session has no next bar, so it
# carries no entry and no label. The exit is resolved by **time**, not by counting rows: the
# library looks for a bar at exactly $H - B$ past the entry, and a bar that is missing
# resolves to nothing and nulls the label instead of letting a shift reach past the hole and
# return a longer window under a shorter name. Materialising the entry price as its own
# column is what lets the library express this convention - it divides by the price at $t$,
# so shifting the series forward by one bar turns "enter one bar late" into "start here".

# %%
priced = quoted.with_columns(
    pl.col("mid_close").shift(-1).over(GROUP_COLS).alias("entry_mid")
).drop_nulls("entry_mid")

# %%
for name in RETURN_LABELS:
    held = f"{(HORIZONS[name] - BAR) // timedelta(minutes=1)}m"
    priced = fixed_time_horizon_labels(
        priced,
        horizon=held,
        method="returns",
        price_col="entry_mid",
        group_col=GROUP_COLS,
        timestamp_col="timestamp",
        tolerance="0s",
    ).rename({f"label_return_{held}": name})

print(f"Constructed {', '.join(RETURN_LABELS)} on {priced.height:,} bars with a tradable entry")

# %% [markdown]
# The direction label is the primary return discretised into a band around zero: a move
# smaller than the friction floor is called flat because it does not pay for its own
# spread. The library's binary method cannot express it - that method splits at zero and
# reports whether the price merely changed - so the band stays notebook-local.
#
# It is built by arithmetic rather than by a `when`/`otherwise` chain, because arithmetic
# propagates nulls and that chain does not: a Polars comparison against a null is not true,
# so an unguarded `otherwise` fires on every row with no forward window and files it as
# "down". Here the outside-the-band test is null wherever the return is, and multiplying by
# the sign carries that null through.

# %%
_outside = (pl.col(PRIMARY_LABEL).abs() > FLAT_BAND).cast(pl.Int8)
_direction = (_outside * pl.col(PRIMARY_LABEL).sign()).cast(pl.Int8)
_position = pl.int_range(pl.len()).over(GROUP_COLS)

labels_df = (
    quoted.join(
        priced.select(["symbol", "timestamp", *RETURN_LABELS]),
        on=["symbol", "timestamp"],
        how="left",
    )
    .with_columns(_direction.alias(DIRECTION_LABEL))
    .with_columns(
        (pl.len().over(GROUP_COLS) - 1 - _position).alias("from_end"),
        _position.alias("bar_in_session"),
        pl.col("timestamp")
        .shift(-HORIZON_BARS[PRIMARY_LABEL])
        .over(GROUP_COLS)
        .alias("_label_end"),
    )
)

# %%
MARKET_DATA_DIGEST = value_digest(quoted, ["symbol", "timestamp", "mid_close"])
print(f"market_data digest: {MARKET_DATA_DIGEST}")

# %% [markdown]
# ## D. Window validity
#
# A shift always returns something; the question is whether what it returns is the quantity
# the label claims. Each property below fails silently and leaves plausible numbers behind,
# so each is asserted rather than described.
#
# The second is what separates this construction from the row shift it replaces. Every
# labelled window is required to span exactly its declared horizon in wall-clock time - not a
# bar count that happens to agree - so a grid that was ever coarser or gappier than it looks
# would raise here rather than ship a longer return under a shorter name.
#
# The third catches a short label masked by a longer one's null set. Each session has to be
# short by exactly its own horizon and no more, so `fwd_ret_5m` carries ten more rows per
# session than `fwd_ret_15m`; an equal count means one label was gated by the other's nulls.

# %%
for name, h_bars in ((n, HORIZON_BARS[n]) for n in RETURN_LABELS):
    span = pl.col("timestamp").shift(-h_bars).over(GROUP_COLS) - pl.col("timestamp")
    checked = labels_df.with_columns(span.alias("_span"))
    tail = checked.filter(pl.col("from_end") < h_bars)
    labelled = checked.drop_nulls(name)
    # 1. An incomplete forward window is null, never a value.
    assert tail[name].null_count() == tail.height, name
    # 2. Every labelled window spans exactly the declared horizon in wall-clock time.
    assert labelled.filter(pl.col("_span") != HORIZONS[name]).height == 0, name
    # 3. Each session labels its first n - h bars, so no label crosses a session boundary and
    #    none is gated by another label's null set.
    counted = checked.group_by(GROUP_COLS).agg(
        (pl.len() - pl.col(name).is_not_null().sum() - h_bars).alias("excess")
    )
    assert counted.filter(pl.col("excess") != 0).height == 0, name
    print(f"{name}: {labelled.height:,} labelled, every window exactly {HORIZONS[name]}")

# %%
# 4. No discrete label is derived from a null return.
_unlabelled = labels_df.filter(pl.col(PRIMARY_LABEL).is_null())[DIRECTION_LABEL]
assert _unlabelled.null_count() == _unlabelled.len()
print(f"{DIRECTION_LABEL}: null on all {_unlabelled.len():,} bars where {PRIMARY_LABEL} is null")

# %% [markdown]
# Position zero below is the last bar of each session. Each label's non-null rate has to
# fall to zero over exactly the last `horizon` positions and sit flat beyond them, and the
# three curves have to step down at three different places. A scalar count of valid rows
# shows neither failure this catches: a tail fabricated instead of nulled, and a short
# label carried on a longer one's null set, which draws the 5-minute curve exactly on top
# of the 15-minute one.

# %%
profile = (
    labels_df.filter(pl.col("from_end") <= max(HORIZON_BARS.values()) + 2)
    .group_by("from_end")
    .agg([pl.col(name).is_not_null().mean().alias(name) for name in RETURN_LABELS])
    .sort("from_end")
)

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for name, colour in PALETTE.items():
    ax.plot(profile["from_end"], profile[name], ds="steps-mid", lw=1.8, c=colour, label=name)
    ax.axvline(HORIZON_BARS[name] - 0.5, color=colour, linestyle=":", lw=1)
ax.set_xlabel("Bars from the end of the session")
ax.set_ylabel("Share of bars carrying a label")
sub = "Dotted lines mark each horizon; curves lying on top of each other mean one masked another"
add_message_title(ax, "Each horizon nulls its own tail of the session and no other", sub)
ax.legend(loc="center right", frameon=False)
show_with_alt(fig, "Non-null label rate by bar position from the end of each trading session.")

# %% [markdown]
# ## E. Distribution and base rate
#
# What scale is the label, and does it mean the same thing across the panel and across
# time? Everything from here through Section G is computed on the development window only,
# sealed on the label's **endpoint** rather than on the bar it was observed from: a
# decision taken shortly before the holdout still resolves inside it, so a filter on the
# observation time looks sealed and is not. The label files keep every row, because the
# seal governs what this notebook looks at rather than what it writes.
#
# Each label is sealed on its own endpoint, because they do not resolve together: the
# 60-minute window opened from a given bar is still running three quarters of an hour after
# the 15-minute one opened from that same bar has closed, so one boundary applied to all
# three would leave the slowest label reaching furthest into the holdout. The symbol-session is carried as one `entity` key,
# because it is the entity no label may cross and Section F counts overlap within it.


# %%
_entity = (pl.col("symbol") + "|" + pl.col("session_date").cast(pl.Utf8)).alias("entity")
dev = {
    name: labels_df.with_columns(
        pl.col("timestamp").shift(-HORIZON_BARS[name]).over(GROUP_COLS).alias("_label_end"),
        _entity,
    )
    .filter(pl.col("_label_end") < HOLDOUT_TS)
    .drop_nulls(name)
    .select(["timestamp", "symbol", "half_spread", "entity", "bar_in_session", name])
    for name in LABEL_NAMES
}
for name in LABEL_NAMES:
    print(f"{name}: {dev[name].height:,} development rows through {dev[name]['timestamp'].max()}")

# %% [markdown]
# All three horizons go on one axis with identical bins and a logarithmic count axis. The
# claim is about shape rather than width: a longer horizon accumulates more variance, so
# if the three are the same process observed over different spans the bodies should widen
# in proportion to the square root of the horizon while the tails stay heavy throughout.
# The axis is symmetric and narrower than any label's range, so rows outside it are counted
# below rather than drawn.

# %%
bins = np.linspace(-0.02, 0.02, 161)
primary_std = dev[PRIMARY_LABEL][PRIMARY_LABEL].std()
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for name, colour in PALETTE.items():
    series = dev[name][name]
    tag = f"{name}, std {series.std():.5f}"
    ax.hist(series.to_numpy(), bins=bins, histtype="step", lw=1.8, color=colour, label=tag)
ax.set_yscale("log")
ax.set_xlabel("Forward midprice return from the entry bar")
ax.set_ylabel("Bars per bin, log scale")
sub = "Identical bins on the development window; rows beyond the axis are counted below"
add_message_title(ax, "Each horizon widens the label while its tails stay far from normal", sub)
ax.legend(loc="lower center", frameon=False)
show_with_alt(fig, "Histograms of the three labels on identical bins, log count axis.")

# %%
for name in RETURN_LABELS:
    series = dev[name][name]
    out = series.filter((series < bins[0]) | (series > bins[-1])).len()
    root_h = np.sqrt(HORIZONS[name] / PRIMARY_HORIZON)
    print(
        f"{name}: std {series.std():.6f}, kurtosis {series.kurtosis():.1f}, "
        f"{series.std() / primary_std:.2f}x the primary label against {root_h:.2f} under "
        f"square-root-of-horizon scaling, {out:,} beyond the axis"
    )

# %% [markdown]
# The label is what a trade earns before costs, and the spread is most of what it pays.
# Both are in the same units, so they belong on the same axis: the curves are the
# distribution of the absolute move at each horizon, and the vertical line is the spread a
# **round trip** crosses - twice the half-spread, because the position is opened and closed
# - on the same bars.
#
# The comparison is not the case study's answer, it is the reason the case study is worth
# running. A move larger than the spread is necessary for the horizon to be tradable and
# nowhere near sufficient: what a strategy earns is the move times how often it gets the
# direction right, and Section G measures how little of that is on offer here.

# %%
round_trip = 2 * dev[PRIMARY_LABEL]["half_spread"].median() * 10_000
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for name, colour in PALETTE.items():
    moves = (dev[name][name].abs() * 10_000).sort().to_numpy()
    ax.plot(moves, np.arange(1, len(moves) + 1) / len(moves), lw=1.8, color=colour, label=name)
ax.axvline(round_trip, color=COLORS["neutral"], ls="--", lw=1.2, label="median round trip")
ax.set_xscale("log")
ax.set_xlim(0.1, 1000)  # below a tenth of a bp the move is a stale quote, not a move
ax.set_xlabel("Absolute forward move, basis points, log scale")
ax.set_ylabel("Share of bars at or below")
sub = "Absolute label against twice the median half-spread, development window"
add_message_title(ax, "The spread costs the same at every horizon while the move grows", sub)
ax.legend(loc="upper left", frameon=False)
show_with_alt(fig, "Empirical CDF of absolute moves per horizon against the median round trip.")

# %%
for name in RETURN_LABELS:
    frame, moved = dev[name], dev[name][name].abs()
    clears = (moved > 2 * frame["half_spread"]).mean()
    print(
        f"{name}: median absolute move {moved.median() * 10_000:.2f}bps against a "
        f"{2 * frame['half_spread'].median() * 10_000:.2f}bps round trip, {clears:.1%} clears it"
    )

# %% [markdown]
# Chapter 7.2 asks for the base rate to be tracked through time. The direction label is
# where that question has an answer: its three classes are cut at a fixed band, so their
# proportions are free to move, and a classifier trained on one regime and scored in
# another is only comparable if they do not move much. The flat class is the interesting
# one - it is the share of the session where nothing happens that is worth paying for.

# %%
monthly = (
    dev[DIRECTION_LABEL]
    .with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
    .group_by("month")
    .agg([(pl.col(DIRECTION_LABEL) == k).mean().alias(str(k)) for k in (-1, 0, 1)])
    .sort("month")
)

# %%
classes = {"0": "flat", "1": "up", "-1": "down"}
shades = (COLORS["neutral"], COLORS["blue"], COLORS["copper"])
fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
for (key, tag), colour in zip(classes.items(), shades):
    ax.plot(monthly["month"], monthly[key], lw=1.8, color=colour, label=tag)
ax.set_ylim(0, 1)
ax.set_xticks(monthly["month"].to_list()[::4])
ax.set_xlabel("Month")
ax.set_ylabel("Share of labelled bars")
sub = f"Class shares of {DIRECTION_LABEL} by month, development window"
add_message_title(ax, "Up and down stay balanced; the flat share does not hold still", sub)
ax.legend(loc="upper right", frameon=False)
show_with_alt(
    fig, "Monthly class shares of the ternary direction label across the development window."
)

for key, tag in classes.items():
    lo, hi = monthly[key].min(), monthly[key].max()
    share = (dev[DIRECTION_LABEL][DIRECTION_LABEL] == int(key)).mean()
    print(f"{tag}: {share:.3f} of labelled bars, ranging {lo:.3f} to {hi:.3f} across months")

# %% [markdown] tags=["results"]
# On the development window the primary label has a standard deviation of 0.004139, against
# 0.002315 for the 5-minute label and 0.007887 for the 60-minute one - 0.56x and 1.91x the
# primary, against the 0.58x and 2.00x square-root-of-horizon scaling implies, so the shorter
# horizon scales as that rule predicts and the longer one falls a little short of it. None of
# the three is remotely normal: kurtosis runs from 85.4 at 60 minutes to 1135.3 at 5, so the
# shorter the horizon the more of its variance sits in rare bars.
#
# Against cost, the median absolute move is 8.77bps at 5 minutes, 16.39bps at 15 and 32.84bps
# at 60, while the median round trip is 4.94, 4.99 and 5.22bps on the same bars - the move
# roughly doubles with each step up in horizon and the spread does not move at all. The share
# of bars whose move clears that round trip climbs from 65.5% to 79.5% to 89.0%. Cut at the
# 5bps friction floor, the direction label splits 0.415 up, 0.405 down and 0.180 flat, and
# while up and down hold between 0.372-0.471 and 0.360-0.468 across months, the flat share
# runs from 0.061 to 0.268.

# %% [markdown]
# ## F. Overlap and effective sample size
#
# Sampling a multi-bar label at every bar makes consecutive rows share most of their
# forward window, so the row count overstates the evidence. Two measurements answer that in
# different units: how fast the overlap decays, and what the rows are worth once it is
# priced in. Both are counted on the bar position within the session, which is the grid the
# horizon was converted into, and both treat the symbol-session as the entity, because a
# window cannot be concurrent with one on the other side of an overnight gap.
#
# What a label consumes is return intervals, and it consumes one fewer than its horizon in
# bars: entering a bar after the decision and leaving at the horizon spans the moves between
# those two prices, not the move into the entry. Consecutive rows share all but one of those
# intervals, so the decay reads as a straight line falling by one interval per lag. Two rows
# `lag` bars apart share `H - 1 - lag` of them, which runs out one bar short of the horizon:
# the last lag that still shares an interval is `H - 2`, and the first that shares none is
# `H - 1`. That is where the dotted lines sit, and marking the horizon instead would put them
# a lag past the thing the section is about.
# The longest label does not stop at zero when it gets there but keeps going negative, and
# that is a property of the session rather than of the label - a one-hour window is a fifth of
# a trading day, so each session holds few independent windows, and subtracting the session's
# own mean from so few of them forces what is left to correlate negatively at long lags.

# %%
max_lag = max(HORIZON_BARS[name] for name in RETURN_LABELS) + 4
acf = {
    name: panel_autocorrelation(
        dev[name], name, max_lag=max_lag, bar_col="bar_in_session", entity_col="entity"
    )
    for name in RETURN_LABELS
}

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
lags = np.arange(1, max_lag + 1)
for name, colour in PALETTE.items():
    ax.plot(lags, acf[name], lw=1.8, color=colour, label=name)
    ax.axvline(HORIZON_BARS[name] - 1, color=colour, linestyle=":", lw=1.2)
ax.axhline(0, color=COLORS["neutral"], lw=0.8)
ax.set_xlabel("Lag in bars")
ax.set_ylabel("Panel autocorrelation")
sub = "Dotted lines mark the first lag sharing no interval; pooled across symbol-sessions"
add_message_title(ax, "Overlap decays linearly with lag at every horizon", sub)
ax.legend(loc="upper right", frameon=False)
show_with_alt(fig, "Panel autocorrelation of each forward-return label against lag in bars.")

# %%
for name in RETURN_LABELS:
    spans = HORIZON_BARS[name] - 1
    n_rows, n_eff = effective_sample_size(
        dev[name], horizon=spans, bar_col="bar_in_session", entity_col="entity"
    )
    print(
        f"{name}: N={n_rows:,}, N_eff={n_eff:,.0f}, ratio {n_eff / n_rows:.4f} against "
        f"{1 / spans:.4f} for {spans} intervals overlapping fully; autocorrelation "
        f"{acf[name][0]:.3f} at lag one, {acf[name][spans - 2]:.3f} at lag {spans - 1} "
        f"where one interval is still shared, and {acf[name][spans - 1]:.3f} at lag {spans} "
        f"where none is"
    )

# %% [markdown] tags=["results"]
# The primary label's 14,474,850 development rows carry 1,069,852 effective observations, a
# ratio of 0.0739 against the 0.0714 that fourteen fully overlapping intervals imply; the
# 5-minute label's 14,861,830 rows carry 3,744,481 at 0.2520 against 0.2500, and the
# 60-minute label's 12,733,440 carry 253,863 at 0.0199 against 0.0169. Each sits above its
# reference because a session end closes an overlap early, and the longest label sits
# furthest above it because the session ends most often relative to its window. Fourteen
# million rows are worth about a million: the row count overstates the evidence by roughly
# the number of intervals each label spans.
#
# The primary label's autocorrelation falls from 0.923 at lag one to 0.032 at lag thirteen,
# the last lag sharing an interval with it, and to -0.040 at fourteen, the first sharing
# none; the fast label runs 0.748 to 0.233 at three and -0.020 at four. The 60-minute label
# falls from 0.977 to -0.198 at fifty-eight and -0.218 at fifty-nine, and it crosses zero
# near lag fifty rather than at its own boundary - each session holds only a handful of
# hour-long windows, and centring so few of them on their own mean drives what is left
# negative. The purge gap a fold needs is set by the forward window itself, not by any of
# these counts.

# %% [markdown]
# ## G. Baseline floor
#
# One signal against the primary label on the sealed development window, with no feature
# engineering: the trailing return over the same span as the label looks forward. If the
# order flow of the last fifteen minutes carries information about the next fifteen, the
# simplest form it can take is that the move continues or that it reverses, and Chapter 8's
# features have to beat whatever that is worth before they have earned their place.
#
# The information coefficient is the cross-sectional rank correlation across the symbols
# priced at each decision minute, averaged over minutes, which is the quantity a ranking
# model is scored on. The minimum cross-section is half the median rather than a bare
# count, so it means the same thing on a universe of another size.
#
# The standard error is heteroskedasticity- and autocorrelation-consistent (HAC): it widens
# the error bar by however much neighbouring observations repeat each other, instead of
# assuming they are independent draws. That matters here because the primary label spans
# fourteen one-minute return intervals and consecutive decision minutes share thirteen of
# them, so a naive statistic would count each minute as fresh evidence when the outcomes are
# almost the same outcome. The printed naive statistic is there to be compared against the
# adjusted one; the gap between them is the size of the mistake.

# %%
_h = HORIZON_BARS[PRIMARY_LABEL]
_trailing = pl.col("mid_close") / pl.col("mid_close").shift(_h).over(GROUP_COLS) - 1
baseline = (
    labels_df.with_columns(
        _trailing.alias("trailing_return"),
        pl.col("timestamp").shift(-_h).over(GROUP_COLS).alias("_label_end"),
    )
    .filter(pl.col("_label_end") < HOLDOUT_TS)
    .drop_nulls([PRIMARY_LABEL, "trailing_return"])
)
min_obs = int(baseline.group_by("timestamp").len()["len"].median() // 2)

ic = cross_sectional_ic_series(
    baseline,
    baseline,
    pred_col="trailing_return",
    ret_col=PRIMARY_LABEL,
    date_col="timestamp",
    entity_col="symbol",
    min_obs=min_obs,
).sort("timestamp")  # HAC autocovariances are meaningless over a permutation of time
stats = compute_ic_hac_stats(ic, ic_col="ic", label_horizon=_h)

# %%
print(f"Baseline: trailing {PRIMARY_HORIZON} return against {PRIMARY_LABEL}")
print(f"  {baseline.height:,} rows, minimum cross-section {min_obs} symbols")
print(
    f"  decision minutes scored {stats['n_periods']:,}, mean IC {stats['mean_ic']:.5f}, "
    f"HAC t {stats['t_stat']:.2f} on {stats['effective_lags']} Bartlett lags, "
    f"naive t {stats['naive_t_stat']:.2f}, p {stats['p_value']:.3g}"
)

# %% [markdown] tags=["results"]
# The trailing 15-minute return earns a mean information coefficient of -0.00781 against the
# primary label, over 135,360 scored decision minutes drawn from 13,894,380 rows on a
# cross-section of at least 51 symbols. The sign is negative, so on this universe the recent
# move tends to give part of itself back rather than continue.
#
# The Newey-West rule picks 19 Bartlett lags and returns a t-statistic of -5.91 against a
# naive -16.47, so pricing in the overlap cuts the apparent evidence by nearly two thirds -
# and what is left is still far from zero, at p 3.5e-09. That is the floor: a feature that
# ranks the cross-section no better than the last quarter-hour of price has added nothing.
# It is a floor on ranking, not on profit - a coefficient of this size is small next to the
# round trip Section E priced, which is the tension the rest of the case study works through.

# %% [markdown]
# ## H. Artifacts and the audit record
#
# Each label is written with a digest sidecar beside it, recording the content digest of
# the values written, the row count, the key columns, the notebook that wrote it, and the
# digest of the price data it was built from. That last field is what ties a label to its
# data vintage: without it a re-run against a refreshed download is indistinguishable from
# a re-run against this one.
#
# Each file is written from its own null set, so the three horizons carry three different
# row counts. The folds that train models are derived per label by
# `case_studies/utils/cv_window.py` from `config/setup.yaml` and the timeline of the label
# parquet written here, so which rows land in these files is what sets where the fold
# boundaries fall.

# %%
readers = {PRIMARY_LABEL: "05_evaluation.py, and every modelling notebook from 06_linear on"}
for name in LABEL_NAMES:
    record = write_artifact(
        labels_df.select(["timestamp", "symbol", name]).drop_nulls(),
        LABELS_DIR / f"{name}.parquet",
        keys=["timestamp", "symbol"],
        written_by="02_labels",
        inputs={"market_data": MARKET_DATA_DIGEST},
    )
    print(f"{name}.parquet: {record['n_rows']:,} rows, digest {record['digest']}")

# %% [markdown]
# The record Chapter 7.2 requires to close a label definition, one row per label, built
# from the values computed above rather than written by hand.

# %%
print("\nLabel audit record")
for name in LABEL_NAMES:
    horizon, h_bars = HORIZONS[name], HORIZON_BARS[name]
    series = dev[name][name]
    scale = (
        f"mean {series.mean():+.6f}, std {series.std():.6f}"
        if name in RETURN_LABELS
        else ", ".join(f"{k}: {(series == k).mean():.3f}" for k in (-1, 0, 1))
    )
    print(
        f"\n{name}\n  anchor       quote midpoint one bar after the decision bar"
        f"\n  horizon      {horizon}, which is {h_bars} bars on this grid"
        f"\n  resolution   fixed at t+{horizon}; the closing NBBO quote breaks the within-bar tie"
        f"\n  overlap      {h_bars - 2} of its {h_bars - 1} return intervals, with the next row"
        f"\n  base rate    {scale}"
        f"\n  consumed by  {readers.get(name, '05_evaluation.py, as a declared variant')}"
    )

# %% [markdown]
# ## Key takeaways
#
# 1. **Write the label as an execution convention, then resolve the exit by time.** Naming
#    the two moments the position is opened and closed - one bar after the decision, and
#    again at the horizon - fixes what the number means; finding the exit by timestamp
#    rather than by counting rows is what keeps it meaning that when a bar is missing.
# 2. **Measure the bar grid before converting a declared horizon into bars.** A 15-row shift
#    is a 15-minute return only on a one-minute grid, and a loader that returns a raw
#    partition promises no such thing. Measure the spacing, require the horizon to be a
#    whole number of bars, and the conversion stops being an assumption.
# 3. **Take the end of the session from the exchange, not from the data.** A vendor that pads
#    every date to the same grid puts bars after an early close, and they are uniformly
#    spaced, so no grid check finds them. The damage is not confined to those bars: the last
#    `horizon` genuine bars before the close take their exit price from one of them. Bound
#    the session by the published schedule before the forward window is built.
# 4. **Write every label from its own null set.** Dropping rows on the primary label before
#    saving the others silently truncates the shorter horizons at the session close, and the
#    row counts look plausible because the file is still large.
# 5. **Seal a diagnostic on the label's endpoint.** A decision taken before the holdout whose
#    outcome resolves inside it is a holdout row, so the usable boundary is the boundary
#    minus the horizon, counted within the session.
# 6. **A row count overstates the evidence when forward windows overlap.** The effective
#    count says by how much, and the HAC standard error is what stops that overlap from
#    inflating a t-statistic - here by a factor of nearly three.
#
# **Known limitations.** The midprice is not a fill: a marketable order crosses the spread,
# and the round trip charted in Section E is the quoted spread rather than a measured
# execution cost - it carries no commission, no market impact and no queue position, so it
# is a floor on what trading costs. It doubles the half-spread quoted at the decision bar,
# which is one full spread, rather than adding the two half-spreads actually crossed at the
# entry and the exit, so it prices the round trip at the moment the decision is taken and
# not at the two moments it is filled. The universe is the list `setup.yaml` declares, and
# the archive behind it is point-in-time: a name carries bars only for the sessions it was a
# constituent, so AAL and WLTW stop in May 2020 where they left the index and the
# cross-section is narrower at the end of the sample than at its start. What the declared
# list fixes is membership over the whole sample, not presence in every session. The flat
# band is a single constant across every
# symbol and every regime, where the spread it stands for is neither. The baseline is one
# signal at one horizon.
#
# **Next**: `03_financial_features.py` builds the order-flow, liquidity and volatility
# features from the same bars; `05_evaluation.py` joins them to these labels and measures
# each feature against them.
