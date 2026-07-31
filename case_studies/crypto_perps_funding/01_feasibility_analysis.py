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
# # Crypto Perpetuals Funding: Feasibility Analysis
#
# `config/setup.yaml` declares a cross-sectional long-short strategy on Binance perpetual
# futures: which contracts trade, that decisions land on the eight-hour funding schedule, what
# crossing costs, and how the sample is split. This notebook asks whether the data supports
# those declarations, and fits nothing.
#
# ## Learning objectives
#
# - Count an unbalanced panel at the timestamp the strategy acts on, and read clearance of the
#   fee off an exceedance curve
# - Measure how long the premium a decision reads describes the same contract, and turn that
#   persistence into the number of independent decisions the sample holds
#
# ## Book reference
#
# Chapter 6, Sections 6.2-6.6. Reads Binance perpetual bars and `config/setup.yaml`, never writes.

# %%
"""Crypto Perpetuals Funding Case Study - Feasibility Analysis."""

import re
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml

from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from data import load_crypto_perps
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title, zero_line

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "crypto_perps_funding"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
MAX_SYMBOLS = 0

# %% [markdown]
# ## Configuration
#
# Every knob is read from `setup.yaml`, and Sections B and D compute on the development window
# alone, so nothing the holdout contains can shape a choice made here.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
HOLDOUT_TS = pl.lit(HOLDOUT_START).str.to_datetime().dt.replace_time_zone("UTC")
PRIMARY_LABEL, LABEL_BUFFER = SETUP["labels"]["primary"], SETUP["labels"]["buffer"]
DECLARED = set(SETUP["universe"]["symbols"])
BREADTH_FLOOR = 2 * max(SETUP["backtest"]["sweep"]["top_k_grid"][PRIMARY_LABEL])
BAR_HOURS = int(SETUP["decision"]["cadence"].split("_")[0])
HORIZONS = sorted(
    int(re.search(r"(\d+)h$", name).group(1))
    for name in (PRIMARY_LABEL, *SETUP["labels"]["variants"])
    if name.startswith("fwd_ret")
)
MAKER_RT, TAKER_RT = (2 * SETUP["costs"]["fee_schedule"][k] for k in ("maker_bps", "taker_bps"))
PREMIUM = "premium_index_close"
ACF_LAGS = 21 * 24 // BAR_HOURS  # three weeks of funding periods

print(
    f"Development {START_DATE} to {HOLDOUT_START} | sealed holdout to {HOLDOUT_END}\n"
    f"{len(DECLARED)} contracts, floor {BREADTH_FLOOR} | horizons {HORIZONS} hours"
)

# %% [markdown]
# ## A. Orientation
#
# A perpetual future has no expiry, so the venue keeps it near the index it tracks by making one
# side pay the other every eight hours. The premium index is the running measure that payment is
# computed from: positive while the perpetual trades above that index, negative while it trades
# below. The payment is a cash flow between longs and shorts rather than a cost, and this strategy
# does not collect it - it ranks contracts on their premium and holds the price move that follows.
# Three questions decide whether that is worth building: are the declared contracts quoting when
# the funding timestamp arrives, is a typical move large next to the fee that captures it, and
# does the sample hold enough independent decisions for a walk-forward.

# %% [markdown]
# ## B. Universe and cost feasibility
#
# ### B.1 Load and verify the declared universe
#
# The loader aggregates raw hourly bars onto the funding grid, so one row is one contract at one
# settlement and `close` is the last price before payment is exchanged. The panel is unbalanced:
# a contract's first row is its listing and nothing is backfilled behind it.

# %%
bars = load_crypto_perps(
    frequency=f"{BAR_HOURS}h", start_date=START_DATE, end_date=END_DATE, max_symbols=MAX_SYMBOLS
).sort(["symbol", "timestamp"])
research = bars.filter(pl.col("timestamp") < HOLDOUT_TS)

loaded = set(research["symbol"].unique().to_list())
assert not loaded - DECLARED, f"in the data, undeclared in setup.yaml: {sorted(loaded - DECLARED)}"
print(
    f"{len(loaded)} of {len(DECLARED)} declared contracts, {len(research):,} funding bars, "
    f"{research['timestamp'].min().date()} to {research['timestamp'].max().date()}"
)

# %% [markdown]
# ### B.2 Breadth at every funding timestamp
#
# One count of the universe hides what a cross-sectional book has to answer: how many contracts are
# quoting at the timestamp it rebalances on. A sleeve of k names per side needs twice k contracts
# there, and the widest sleeve `backtest.sweep.top_k_grid` declares sets the floor to clear.

# %%
breadth = research.group_by("timestamp").agg(n=pl.col("symbol").n_unique()).sort("timestamp")

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(breadth["timestamp"], breadth["n"], color=COLORS["blue"], lw=1.0)
ax.axhline(BREADTH_FLOOR, color=COLORS["copper"], ls="--", lw=1.5, label="widest declared sleeve")
ax.set_ylim(0, BREADTH_FLOOR + 2)
ax.set_yticks(range(0, BREADTH_FLOOR + 3, 5))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Contracts quoting at the funding timestamp")
ax.legend(frameon=False, fontsize=8, loc="lower right")
add_message_title(
    ax,
    "The panel never carries both legs of the widest sleeve the grid declares",
    subtitle="Perpetuals with a settled funding bar at each timestamp, against twice the largest top-k",
)
plt.show()

# %% [markdown]
# ### B.3 What a move is worth against the fee
#
# `setup.yaml::costs.fee_schedule` charges a flat fee per trade in two tiers rather than a
# per-contract spread, so cost here is a level the venue publishes rather than something this data
# measures. Both round trips are drawn against the absolute moves at every horizon the labels
# declare, on one log axis, so the fraction clearing either tier is read off the curve.

# %%
moves = bars.with_columns(
    pl.when(pl.col("timestamp") + pl.duration(hours=h) < HOLDOUT_TS)
    .then((pl.col("close").shift(-h // BAR_HOURS).over("symbol") / pl.col("close") - 1).abs() * 1e4)
    .alias(f"h{h}")
    for h in HORIZONS
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for h, color in zip(HORIZONS, (COLORS["blue"], COLORS["amber"]), strict=True):
    magnitude, fraction = exceedance_curve(moves[f"h{h}"].drop_nulls().to_numpy())
    ax.plot(magnitude, fraction, color=color, lw=1.6, label=f"{h}-hour move")
ax.axvline(MAKER_RT, color=COLORS["neutral"], ls=":", lw=1.5, label="maker round trip")
ax.axvline(TAKER_RT, color=COLORS["copper"], ls="--", lw=1.5, label="taker round trip")
ax.set_xscale("log")
ax.set_xlim(left=1)
ax.set_xlabel("Absolute move (bps, log scale)")
ax.set_ylabel("Fraction of moves at least this large")
ax.legend(frameon=False, fontsize=8, loc="lower left")
add_message_title(
    ax,
    "Moves at both horizons clear the round trip that captures them",
    subtitle="Exceedance of absolute perpetual returns against the declared maker and taker round trips",
)
plt.show()

# %% [markdown]
# ### B.4 How long the premium describes the contract
#
# Ranking contracts every funding period is only worth the turnover if what the premium says at one
# settlement still describes the same contract at the next. That is an autocorrelation, computed
# inside each contract: stacking the panel and correlating it measures where two contracts meet.

# %%
acf = panel_acf(research, entity_col="symbol", value_col=PREMIUM, max_lags=ACF_LAGS)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.fill_between(acf["lag"], acf["acf_p10"], acf["acf_p90"], color=COLORS["blue"], alpha=0.15)
ax.bar(acf["lag"], acf["acf"], color=COLORS["blue"], width=0.7)
ax.axhspan(-acf["band"][0], acf["band"][0], color=COLORS["copper"], alpha=0.35)
ax.set_xlabel("Lag (funding periods)")
ax.set_ylabel("Autocorrelation of the premium index")
add_message_title(
    ax,
    "Premium persistence outlasts every horizon the labels declare",
    subtitle="Mean within-contract autocorrelation, shaded 10th-90th percentile across contracts",
)
plt.show()

# %% [markdown]
# Persistence describes one contract through time. A cross-sectional book also needs the contracts
# to disagree with each other at a single timestamp, since ranking a panel whose members carry the
# same premium sorts noise. The band is what the ranking works with; the line is the level it nets.

# %%
BANDS = (("lo", 0.1), ("mid", 0.5), ("hi", 0.9))
spread = (
    research.group_by(pl.col("timestamp").dt.truncate("1d").alias("day"))
    .agg(pl.col(PREMIUM).quantile(q).mul(1e4).alias(name) for name, q in BANDS)
    .sort("day")
    .drop_nulls()
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.fill_between(spread["day"], spread["lo"], spread["hi"], color=COLORS["blue"], alpha=0.25)
ax.plot(spread["day"], spread["mid"], color=COLORS["blue"], lw=0.8)
zero_line(ax)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Premium index (bps of the tracked index)")
add_message_title(
    ax,
    "The premium moves as a common level more than contracts disperse",
    subtitle="Daily cross-sectional median of the premium index, banded 10th to 90th percentile",
)
plt.show()

# %% [markdown]
# ### B.5 Move scale against cost
#
# The ratio divides the median absolute move at the primary horizon by the taker round trip,
# the tier a contract outside the majors pays. It says the market moves further than the fee,
# not that the move is forecastable: the numerator is an unsigned realized magnitude.

# %%
primary = moves[f"h{HORIZONS[0]}"].drop_nulls()
print(
    f"Round trip {MAKER_RT} bps at maker and {TAKER_RT} bps at taker | median {HORIZONS[0]}-hour "
    f"move {primary.median():.0f} bps, ratio {primary.median() / TAKER_RT:.0f}x, share above the "
    f"taker round trip {(primary > TAKER_RT).mean():.3f}"
)

# %% [markdown] tags=["results"]
# The maker round trip costs 4 bps and the taker round trip 8 bps. The median absolute 8-hour
# move is 138 bps, 17 times the taker round trip, and 0.963 of moves are larger than it. The
# numerator is an unsigned magnitude, so this bounds the room a forecast has to work in and
# says nothing about whether one exists.

# %% [markdown]
# ## C. Design decisions
#
# ### C.1 Cadence
#
# `setup.yaml::decision.cadence` rebalances on the funding grid and executes at the funding
# timestamp. That is an information schedule rather than a hyperparameter to sweep: a new premium
# observation exists only when a funding period settles, so a decision taken between two
# settlements reads the same premium twice and pays the fee twice. B.4 supports holding through at
# least one such period, since the premium is autocorrelated far past the longest declared horizon.
#
# ### C.2 Kill conditions
#
# Four falsifiable checkpoints send the strategy back to the drawing board, each tested where its
# evidence exists rather than here: a model-driven gross return the fee erases, in Chapter 16; a
# premium that stops predicting before the next funding timestamp, in Chapter 7 through the
# information coefficient rather than the autocorrelation measured above; a venue change to the
# funding formula, cap or interval, which would leave the training distribution describing a
# product that no longer exists; and an equal-weight long-short cross-section reaching a higher
# Sharpe and a shallower drawdown than the strategy on every validation fold, in Chapter 17.
#
# ### C.3 Mapping class
#
# `setup.yaml::mapping.class` ranks contracts on the premium and holds both legs. A perpetual is
# symmetrically tradable from either side, so a long-only restriction would discard half the
# cross-section the ranking produces and would leave the common level in the position - and B.4
# shows that level is most of what the premium does. Sizing is declared as equal weight or risk
# parity because Chapter 16 fixes equal weight as the baseline and Chapter 17 sweeps alternatives.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 Effective sample size
#
# What evaluation spends is independent decisions, not rows. The autocorrelation from B.4 turns one
# into the other: summing the initial positive sequence of the mean within-contract curve gives the
# integrated autocorrelation time, the funding periods one independent observation is worth. That
# sequence has not turned negative by the last lag drawn, so what follows is a ceiling, not an
# estimate.

# %%
curve = acf["acf"].to_numpy()
odd, even = curve[1::2], curve[2::2]
pair_sums = odd[: len(even)] + even
negative = np.flatnonzero(pair_sums <= 0)
tau = 1 + 2 * pair_sums[: negative[0] if negative.size else len(pair_sums)].sum()
raw_decisions = SETUP["evaluation"]["periods_per_year"] * 24 // BAR_HOURS
print(
    f"Funding timestamps {len(breadth):,} | contracts per timestamp {breadth['n'].mean():.1f} | "
    f"integrated autocorrelation {tau:.0f} funding periods, so at most {raw_decisions / tau:.0f} "
    f"independent decisions a year against {raw_decisions:,} raw"
)

# %% [markdown]
# ### D.2 Fold demonstration
#
# `generate_cv_splits` derives the folds from `setup.yaml::evaluation` alone. Between each training
# and validation block sits a purge gap the width of the label horizon, so a label computed at the
# end of training cannot resolve inside validation. The figure draws those boundaries rather than
# recomputing them, so it and the folds cannot disagree. The util returns folds newest first and
# later stages index the same list, so the reordering stays local here.

# %%
splits = sorted(
    generate_cv_splits(
        research, case_study_id=CASE_STUDY_ID, label_buffer=LABEL_BUFFER, date_col="timestamp"
    ),
    key=lambda split: split["train_start"],
)
for position, split in enumerate(splits):
    split["fold"] = position
last_val = max(split["val_end"] for split in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
assert last_val < pd.Timestamp(HOLDOUT_START, tz="UTC"), "a fold reaches into the holdout"

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
add_message_title(
    ax,
    "Folds roll forward and stop short of the sealed holdout",
    subtitle="Blocks as generate_cv_splits returns them; the purge between them is one funding period",
)
plt.show()

# %% [markdown]
# ## E. Derived artifacts. Nothing: `setup.yaml::universe.symbols` fixes the contract list and the
# loader carries no row before a listing, so no downstream file reads an eligibility table here.

# %% [markdown]
# ## F. Findings vs `setup.yaml`
#
# One row per knob: the evidence that motivates it, and what would change it.
#
# | Knob | Evidence | Revise it when |
# |---|---|---|
# | `universe.symbols` | B.2 breadth at each funding timestamp | breadth falls below the contracts a sleeve needs on both legs |
# | `backtest.sweep.top_k_grid` | B.2 breadth against twice the largest top-k | the panel cannot fill both legs at the widest sleeve |
# | `decision.cadence` | B.3 exceedance, B.4 persistence | moves stop clearing the round trip, or the premium decays inside one funding period |
# | `costs.fee_schedule` | B.3 the two declared round trips | the venue changes a tier, or a contract moves between them |
# | `evaluation.n_splits` | D.1 independent decisions, D.2 boundaries | the folds no longer fit the development window |

# %%
print(
    f"universe.n_assets {SETUP['universe']['n_assets']}, breadth {breadth['n'].min()} to "
    f"{breadth['n'].max()}, under the floor of {BREADTH_FLOOR} on "
    f"{breadth.filter(pl.col('n') < BREADTH_FLOOR).height:,} of {len(breadth):,} timestamps\n"
    f"decision.cadence {SETUP['decision']['cadence']} | labels.primary {PRIMARY_LABEL}\n"
    f"evaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, last "
    f"validation ends {last_val.date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# The declared universe holds 19 contracts. Breadth at a funding timestamp runs from 2 to 19 and
# stays under the floor of 20 that the widest declared sleeve needs on all 4,383 of them, so the
# top-10 grid entry cannot fill both legs anywhere in the development window. Two folds are
# generated, the last validation ending 2023-12-31. An integrated autocorrelation of 39 funding
# periods leaves at most 28 independent decisions a year against 1,095 raw.

# %% [markdown]
# ## Key takeaways
#
# 1. **Count the panel at the timestamp the strategy acts on.** A panel whose members enter at
#    their listing dates has a breadth history, and the sleeve the sweep declares has to fit
#    inside it on every one of those timestamps.
# 2. **Compute a panel autocorrelation inside each entity.** Stacking the contracts and
#    correlating the result measures where one contract's series meets the next.
# 3. **Turn persistence into a decision count before trusting a sample size.** Where the initial
#    positive sequence runs past the lags you drew, what you have is a ceiling on independence.
# 4. **Separate the common level from the cross-sectional spread.** A ranking reads only what is
#    left once the level both legs cancel is taken out, and here the level is the larger part.
#
# ### Known limitations
#
# - The contract list is fixed and was drawn knowing which perpetuals stayed listed, so it is not a
#   point-in-time universe and carries selection and delisting bias.
# - Cost is the published fee alone; slippage and the spread on entry need a notional and enter at
#   the cost stage.
# - Funding is a cash flow this strategy does not collect. The labels and the backtest measure the
#   price move net of fees, and the premium enters only as a feature.
#
# **Next**: labels at the declared horizons, built on this development window.
