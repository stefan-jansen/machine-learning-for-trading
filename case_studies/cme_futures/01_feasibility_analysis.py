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
# # CME Futures: Feasibility Analysis
#
# `config/setup.yaml` declares a cross-sectional futures strategy: which products trade,
# how often positions change, what a round trip costs, and how the sample is split for
# evaluation. This notebook asks whether the data supports it, and fits nothing.
#
# ## Learning objectives
#
# - Count the universe on the session the strategy acts on, against the breadth a
#   two-sided book needs, and price a round trip from the contract's own tick
# - Read off an exceedance curve what fraction of moves clears its own product's cost
# - Measure how long the term-structure slope a carry strategy reads stays put
#
# ## Book reference
#
# Chapter 6, Sections 6.2-6.6. Reads CME settlements via `load_cme_futures()` and
# `config/setup.yaml`, the strategy declaration this notebook never writes.

# %%
"""CME Futures Case Study - Feasibility Analysis."""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from data import load_cme_futures
from utils.config import REPO_ROOT
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "cme_futures"
START_DATE = "2011-01-01"
END_DATE = "2025-12-31"

# %% [markdown]
# ## Configuration
#
# Every knob is read from `setup.yaml`. Section B computes on the development window alone,
# so nothing the holdout contains can shape a choice made here.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = SETUP["labels"]["buffer"]
DECLARED_PRODUCTS = sorted(p for g in SETUP["universe"]["product_groups"].values() for p in g)
BREADTH_FLOOR = 2 * max(SETUP["backtest"]["sweep"]["top_k_grid"][PRIMARY_LABEL])
SPREAD_TICKS = SETUP["costs"]["spread_ticks"]["liquid"]
LABELS = [PRIMARY_LABEL, *SETUP["labels"]["variants"]]
HORIZONS = sorted(int(re.search(r"(\d+)d$", name).group(1)) for name in LABELS)

print(f"Development {START_DATE} to {HOLDOUT_START} | sealed holdout to {HOLDOUT_END}")
print(f"{len(DECLARED_PRODUCTS)} products, floor {BREADTH_FLOOR} | horizons {HORIZONS} sessions")

# %% [markdown]
# ## A. Orientation
#
# CME futures give leveraged exposure across equity indices, rates, energy, metals,
# currencies, agriculture and livestock through one order type and one clearing house. A
# front-month position earns the spot move plus the roll yield the term structure implies,
# so ranking products against each other trades a difference in carry as much as one in
# direction. Three questions decide whether that is worth building here: does the declared
# universe exist on every decision date, is a typical move large next to what it costs to
# capture, and does the sample carry enough decision dates for a walk-forward evaluation
# that never reads the holdout? Section C records which `setup.yaml` value each motivates.

# %% [markdown]
# ## B. Universe and cost feasibility
#
# ### B.1 Load and verify the declared universe
#
# The loader returns one row per product, tenor and session; `tenor == 0` is the front
# month. Two price columns matter and are not interchangeable: `raw_close` is the printed
# settlement, while `adj_close` is back-adjusted, so its differences are price moves
# rather than roll gaps.

# %%
futures = load_cme_futures(start_date=START_DATE, end_date=END_DATE)
front = (
    futures.filter(pl.col("tenor") == 0)
    .select(["product", "session_date", "raw_close", "adj_close"])
    .sort(["product", "session_date"])
)
research = front.filter(pl.col("session_date") < pl.lit(HOLDOUT_START).str.to_date())

missing = sorted(set(DECLARED_PRODUCTS) - set(research["product"].unique().to_list()))
assert not missing, f"declared in setup.yaml but absent from the data: {missing}"
print(
    f"{research['product'].n_unique()} products, {len(research):,} settlements, "
    f"{research['session_date'].min()} to {research['session_date'].max()}"
)

# %% [markdown]
# ### B.2 Breadth at every decision date
#
# One count of the universe hides the question a cross-sectional strategy has to answer,
# which is whether the products are there *on the session it acts on*. Counting anywhere
# in the week hides exactly the dates that matter, because the week's final session is
# sometimes a holiday session on which most of the universe does not settle. The line is
# what the widest sweep holds across both legs.

# %%
decisions = research.group_by(pl.col("session_date").dt.truncate("1w")).agg(
    pl.col("session_date").max().alias("decision_date")
)
breadth = (
    research.join(decisions, left_on="session_date", right_on="decision_date")
    .group_by("session_date")
    .agg(pl.col("product").n_unique().alias("n_products"))
    .sort("session_date")
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(breadth["session_date"], breadth["n_products"], color=COLORS["blue"], linewidth=1.2)
ax.axhline(BREADTH_FLOOR, color=COLORS["copper"], ls="--", lw=1.5, label="both-leg position floor")
ax.set_ylim(0, len(DECLARED_PRODUCTS) + 2)
ax.set_ylabel("Products quoting at the weekly decision")
ax.legend(frameon=False, fontsize=8, loc="lower right")
add_message_title(
    ax,
    "Holiday sessions are the only decision dates a two-sided book cannot fill",
    subtitle="Products settling on the week's final session, the date the strategy acts on",
)
plt.show()

# %% [markdown]
# ### B.3 What a round trip costs, and what a move is worth
#
# `setup.yaml::costs` prices a trade as a commission plus a spread quoted in ticks, and the
# tick is a contract specification: `futures_specs.yaml` carries the exchange value per
# product. Do not infer it from the data - the smallest settlement increment is a half-tick
# on several contracts, which halves the cost you charge yourself. A one-tick spread costs
# half a tick per leg, so a round trip is one tick, and the typical price converts it to
# basis points. The commission needs a notional and enters at the cost stage.

# %%
products = yaml.safe_load((REPO_ROOT / "data/futures/market/futures_specs.yaml").read_text())
ticks = pl.DataFrame(
    {
        "product": list(products["products"]),
        "tick": [p["tick_size"] for p in products["products"].values()],
    }
)
cost = (
    research.group_by("product")
    .agg(pl.col("raw_close").median().alias("price"))
    .join(ticks, "product")
    .with_columns((SPREAD_TICKS * pl.col("tick") / pl.col("price") * 1e4).alias("round_trip_bps"))
    .sort("round_trip_bps")
)
COST_BPS = float(cost["round_trip_bps"].median())

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.bar(cost["product"], cost["round_trip_bps"], color=COLORS["blue"], width=0.7)
ax.axhline(COST_BPS, color=COLORS["copper"], ls="--", lw=1.5, label="universe median")
ax.set_ylabel("Round-trip spread (bps)")
ax.tick_params(axis="x", labelsize=6, rotation=90)
ax.legend(frameon=False, fontsize=8)
add_message_title(
    ax,
    "The same universe spans an order of magnitude in what a round trip costs",
    subtitle="One tick as a fraction of the median settlement price, per product",
)
plt.show()

# %% [markdown]
# Against that cost sits the move a position is trying to capture: the fraction of absolute
# returns at or above each magnitude, per horizon, against the universe-median spread.

# %%
returns = research.with_columns(
    pl.col("adj_close").pct_change(h).abs().over("product").alias(f"h{h}") for h in HORIZONS
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for h, color in zip(HORIZONS, (COLORS["blue"], COLORS["amber"]), strict=True):
    magnitude, fraction = exceedance_curve(returns[f"h{h}"].drop_nulls().to_numpy())
    ax.plot(magnitude * 10_000, fraction, color=color, lw=1.6, label=f"{h}-session move")
ax.axvline(COST_BPS, color=COLORS["copper"], ls="--", lw=1.5, label="round-trip spread")
ax.set_xscale("log")
ax.set_xlim(0.4, 3_000)
ax.set_xlabel("Absolute return (bps, log scale)")
ax.set_ylabel("Fraction of moves at least this large")
ax.legend(frameon=False, fontsize=8, loc="lower left")
add_message_title(
    ax,
    "Almost every move at either horizon is larger than the spread it crosses",
    subtitle="Exceedance of absolute front-month returns, development window",
)
plt.show()

# %% [markdown]
# ### B.4 How long the carrier stays put
#
# Rebalancing weekly is only worth the turnover if what the data says at one decision date
# still says something at the next. This strategy reads carry, visible in the raw prices as
# the slope between the front contract and the one behind it: a positive slope means a long
# position rolls into a more expensive contract and gives back part of the spot move. How
# long that reading lasts is an autocorrelation, computed inside each product; stacking
# thirty products and correlating the result measures their joins instead.

# %%
sealed = pl.col("session_date") < pl.lit(HOLDOUT_START).str.to_date()
term = (
    futures.filter((pl.col("tenor") <= 1) & sealed)
    .pivot(on="tenor", index=["product", "session_date"], values="raw_close")
    .rename({"0": "near", "1": "deferred"})
    .drop_nulls()
    .sort(["product", "session_date"])
    .with_columns(((pl.col("deferred") - pl.col("near")) / pl.col("near")).alias("slope"))
)
acf = panel_acf(term, entity_col="product", value_col="slope", max_lags=max(HORIZONS))

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.fill_between(acf["lag"], acf["acf_p10"], acf["acf_p90"], color=COLORS["blue"], alpha=0.15)
ax.bar(acf["lag"], acf["acf"], color=COLORS["blue"], width=0.6)
ax.axhspan(-acf["band"][0], acf["band"][0], color=COLORS["copper"], alpha=0.3)
ax.set_xlabel("Lag (sessions)")
ax.set_ylabel("Autocorrelation of the term-structure slope")
add_message_title(
    ax,
    "Carry moves slowly enough that a weekly rebalance still acts on it",
    subtitle="Mean within-product autocorrelation, shaded 10th-90th percentile across products",
)
plt.show()

# %% [markdown]
# ### B.5 Move scale against cost
#
# The ratio below divides the median absolute move at the primary horizon by the median
# round-trip spread; the clearance share compares each move with *its own* product's
# spread, because the two ends of the universe differ by an order of magnitude. Both run
# on realised moves rather than forecasts, so neither says the sign is predictable.

# %%
move_bps = pl.col(f"h{HORIZONS[0]}") * 1e4
priced = returns.drop_nulls(f"h{HORIZONS[0]}").join(
    cost.select("product", "round_trip_bps"), "product"
)
median_move_bps, clears_cost = priced.select(
    move_bps.median().alias("mid"), (move_bps > pl.col("round_trip_bps")).mean().alias("share")
).row(0)
print(
    f"Round-trip spread {cost['round_trip_bps'].min():.2f} to "
    f"{cost['round_trip_bps'].max():.2f} bps, median {COST_BPS:.2f} bps | median "
    f"{HORIZONS[0]}-session move {median_move_bps:.1f} bps, ratio "
    f"{median_move_bps / COST_BPS:.0f}x, clears its own product's spread {clears_cost:.3f}"
)

# %% [markdown] tags=["results"]
# The median round-trip spread across the thirty products is 1.13 bps, from 0.43 bps on
# the two-year note to 5.80 bps on corn. The median absolute five-session move is 121.2
# bps, and 0.992 of moves exceed the spread their own product would charge to cross.

# %% [markdown]
# ## C. Design decisions
#
# ### C.1 Cadence
#
# `setup.yaml::decision.cadence` rebalances at the Friday settlement and executes at the
# Monday open. B.3 supports that from the cost side: moves at both horizons sit far above
# the spread, so the signal sets the cadence rather than cost.
#
# ### C.2 Kill conditions
#
# Three thresholds send the strategy back to the drawing board, tested where the evidence
# exists rather than here: a combined carry-and-momentum information coefficient below its
# floor across folds; long-backwardation carry turning profitable over the full sample,
# which would make an inverted-carry reading a regime artifact; and round-trip cost above
# the median absolute move for most of the universe.
#
# ### C.3 Mapping class
#
# `setup.yaml::mapping.class` ranks products by carry or momentum and holds both legs.
# Shorting a future carries no borrow, so the short leg costs what the long leg costs.
# Sizing is equal-risk rather than equal-notional because volatilities across the universe
# differ by an order of magnitude, and notional weighting would hand the book to energy
# and grains. A quintile of thirty holds six, which is the real limit here.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 Effective sample size
#
# What evaluation spends is decision dates, not rows: one a week here.

# %%
print(
    f"Sessions {research['session_date'].n_unique():,} | decision dates {len(decisions):,} "
    f"| products per decision {breadth['n_products'].mean():.0f}"
)

# %% [markdown]
# ### D.2 Fold demonstration
#
# `generate_cv_splits` derives the folds from `setup.yaml::evaluation` alone, so the
# boundaries are the declared design rather than a copy of it. Between each training and
# validation block sits a purge gap the width of the label horizon, which stops a label
# computed inside training from resolving inside validation. The figure draws those
# boundaries rather than recomputing them, so it and the folds cannot disagree.

# %%
splits = generate_cv_splits(
    research.select("session_date").rename({"session_date": "timestamp"}),
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
    date_col="timestamp",
)
last_val = max(s["val_end"] for s in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
assert last_val < np.datetime64(HOLDOUT_START), "a fold reaches into the holdout"

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
add_message_title(
    ax,
    "Folds roll back from the sealed holdout and stop short of it",
    subtitle="Training, purge and validation blocks exactly as generate_cv_splits returns them",
)
plt.show()

# %% [markdown]
# ## E. Derived artifacts
#
# Nothing: exchange listings fix the universe, so no later stage reads an eligibility file.

# %% [markdown]
# ## F. Findings vs `setup.yaml`
#
# One row per knob: the evidence that motivates it, and what would change it.
#
# | Knob | Evidence | Revise it when |
# |---|---|---|
# | `universe.n_products` | B.2 breadth per decision date | breadth falls below the position count the sweep asks for on either leg |
# | `decision.cadence` | B.3 exceedance, B.4 persistence | moves stop clearing the spread, or the slope decays inside one rebalancing interval |
# | `costs.spread_ticks` | B.3 tick per product from `futures_specs.yaml` | the exchange changes a tick, or the desk pays more than one tick to cross |
# | `evaluation.n_splits` | D.1 decision dates, D.2 boundaries | the folds no longer fit the development window |

# %%
print(
    f"universe.n_products {SETUP['universe']['n_products']}, breadth per decision date "
    f"{breadth['n_products'].min()} to {breadth['n_products'].max()}, under the floor on "
    f"{breadth.filter(pl.col('n_products') < BREADTH_FLOOR).height} of {len(breadth)} dates\n"
    f"decision.cadence {SETUP['decision']['cadence']} | labels.primary {PRIMARY_LABEL}\n"
    f"evaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, "
    f"last validation ends {last_val.date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# Breadth is 30 products on nearly every decision date and 8 at its worst, a Good Friday;
# the floor of 20 binds on 7 of 678 dates, every one of them a holiday session. Five folds
# are generated from the declared design, the last validation block ending 2023-12-21.

# %% [markdown]
# ## Key takeaways
#
# 1. **Count the universe on the session the strategy acts on.** Anywhere-in-the-week
#    hides the holiday dates where a two-sided book cannot be filled.
# 2. **Take the tick from the contract specification, not from the prices.** Several
#    contracts settle on half-ticks, so the observed grid understates what you pay.
# 3. **Compare moves to cost with an exceedance curve, not a histogram**, and against each
#    product's own spread where the universe spans an order of magnitude in cost.
# 4. **Autocorrelation on a panel is computed inside each entity**, and a fold figure is
#    drawn from the boundaries it reports, or it can contradict them.
#
# ### Known limitations
#
# - Cost here is spread only; commission and roll slippage enter at the cost stage.
# - A persistent carry reading is not a profitable one; whether the slope predicts the
#   next move's sign is what the modelling stages test.
#
# **Next**: labels at the declared horizons, built on this development window.
