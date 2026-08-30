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
# # Crypto perpetuals: how much friction the surviving configuration absorbs
#
# Every backtest so far charged one cost schedule, the one `config/setup.yaml` declares. This
# notebook holds the configuration that survived [`14_portfolio_management`](14_portfolio_management.ipynb)
# completely fixed - same model, same checkpoint, same entry rule, same allocator - and varies
# only what it costs to trade.
#
# **This stage selects nothing.** The three stages before and after it narrow a field: the
# baseline runs everything, allocation runs the top ten configurations, the risk overlay runs the
# top one. Cost sensitivity runs the top one too, but it is not choosing between candidates - it
# is asking a question about the one already chosen, and the answer is a curve rather than a
# ranking. Nothing downstream reads a ranking from here.
#
# The reason a curve is the right output is that a single cost assumption is a guess. The declared
# schedule is one point on it, and a result that holds only at that point is a result about the
# guess rather than about the strategy. What the curve shows is how far the
# assumption can move before the conclusion does.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read a cost level in basis points as a charge on traded notional, and say why turnover, not
#   the rate, is what converts it into a change in Sharpe.
# - Distinguish a per-trade execution charge, which scales with the rate, from a funding cash
#   flow, which is a property of holding the position and does not.
# - Read a cost-decay curve for a breakeven rather than reading a single point estimate.
# - Say why a uniform grid is a sensitivity axis and not a faithful reproduction of the declared
#   fee structure.
#
# **Book reference**: Chapter 18 (Transaction Costs).
#
# **Prerequisites**: [`14_portfolio_management`](14_portfolio_management.ipynb) has frozen a
# candidate set per label.
#
# **What it writes**: one `stage='cost_sensitivity'` backtest per label and cost level. No
# candidate set, because nothing here is selected from.

# %%
"""Sweep the declared cost grid across the surviving crypto perpetuals configuration."""

import sqlite3
from contextlib import closing

import plotly.graph_objects as go
import polars as pl

from case_studies.crypto_perps_funding.research_workflow import (
    ALL_LABELS,
    selected_allocation_result,
)
from case_studies.research import open_study, run_backtests
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.sweep_config import get_cost_grid_bps
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
LABELS: list[str] = []
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
POPULATION_SUFFIX = "v1"

# %%
study = open_study(
    "crypto_perps_funding", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)
labels = list(LABELS) if LABELS else list(ALL_LABELS)
# Where this run's own results are written and read back from: the released case directory on a
# canonical run, the isolated preview directory otherwise. `study.root` is the released one in
# both tiers, so a preview that reads it is reading somebody else's registry.
STORAGE_ROOT = study.storage_root(study.execution_tier)
# A canonical run reads the funnel's frozen sets and publishes its own; a preview run against a
# private workspace reads and writes only what it produced there.
CANONICAL_RUN = EXECUTION_TIER == "canonical" and not WORKSPACE
case_config = get_backtest_config("crypto_perps_funding")

# %% [markdown]
# ## 1. The configuration under test
#
# `14_portfolio_management` froze one candidate set per label holding the baseline and the
# allocation results together, admitting only results that traded every validation fold. The
# configuration this notebook varies costs for is the highest validation Sharpe in that set.
#
# Reading it back through the frozen set rather than re-querying the registry matters, because
# the set is immutable and the query is not. A registry grows: a later run that adds one result
# would change what a fresh "best allocation result" query returns, and the cost curve would then
# describe a different configuration from the one the previous stage chose. `CandidateSet.one`
# resolves a name to exactly one identity or raises.

# %%
chosen_by_label = {
    label: selected_allocation_result(study, label=label, canonical=CANONICAL_RUN)
    for label in labels
}

# %% [markdown]
# One row per label. `stage` says whether the surviving configuration came from the baseline or
# from the allocation grid - equal weight remains eligible, and a label where it wins is a label
# where no allocator beat it.

# %% tags=["results"]
backtests = study.backtests.table(include_preview=not CANONICAL_RUN)
selected = backtests.filter(
    pl.col("backtest_hash").is_in([result.hash for result in chosen_by_label.values()])
).select(
    "label",
    "stage",
    "family",
    "config_name",
    "signal_method",
    pl.col("allocation_method").fill_null("equal_weight").alias("allocator"),
    "sharpe",
    "avg_turnover",
    "prediction_hash",
)
if selected.height != len(chosen_by_label):
    raise RuntimeError("a selected result is absent from the backtest catalog")
selected.sort("label")

# %% [markdown]
# ## 2. The grid, and what a level on it means
#
# `config/setup.yaml` declares the levels under `backtest.sweep.cost_grid_bps`. A level is the
# **round-trip charge on traded notional, in basis points**, split half to commission and half to
# slippage. That split is the convention every case study's cost sweep uses, so a curve here reads
# against a curve elsewhere.
#
# **The grid is a sensitivity axis, not the declared fee structure.** The production schedule for
# these contracts is a 4 bp commission and a 1 bp slippage allowance, which the case study takes
# from the exchange's taker tier. Its total is 5 bps, and 5 bps is a cell on the grid - but the
# cell splits that total evenly, so it is the same amount of friction distributed differently, not
# the registered production point re-run. Slippage moves the fill price and commission is charged
# on the notional that results, so the two are not interchangeable to the last dollar. Read the
# curve as a response to a uniform cost level.
#
# The upper end of the grid is deliberately past anything these venues charge. A perpetual future
# on a major exchange does not cost 50 bps to trade. The point of running it is that the curve
# between the plausible levels and the implausible ones is where a strategy reveals whether it has
# any margin at all.

# %%
cost_grid = get_cost_grid_bps("crypto_perps_funding")
if not cost_grid:
    raise RuntimeError("crypto_perps_funding declares no backtest.sweep.cost_grid_bps")
declared_total = case_config.commission_bps + case_config.slippage_bps
print(
    f"{len(cost_grid)} declared levels: {', '.join(f'{level:g}' for level in cost_grid)} bps.\n"
    f"Production schedule: {case_config.commission_bps:g} bps commission + "
    f"{case_config.slippage_bps:g} bps slippage = {declared_total:g} bps total"
    + (" (a level on the grid)" if declared_total in cost_grid else " (not a level on the grid)")
)

# %% [markdown]
# ## 3. Running the sweep
#
# For each label, that result's own strategy is taken from its registered specification and
# re-run once per cost level. Nothing in the strategy is rebuilt here: the entry rule and the
# allocator are the fields the previous two stages resolved, read back rather than reconstructed,
# so a change to how they are configured cannot silently produce a cost curve for something else.
#
# Prices are loaded with the warmup its allocator needs, which is the same window the
# allocation stage gave it. A moment-based allocator that saw less history here would weight
# differently, and the difference would be attributed to the cost level.

# %%
cost_runs = []
for label in labels:
    chosen = chosen_by_label[label]
    strategy = chosen.spec()["strategy"]
    allocation = strategy.get("allocation")
    warmup = strategy_warmup_periods({"allocation": allocation} if allocation else {})
    prices = load_backtest_prices_for(
        "crypto_perps_funding", label, split="validation", warmup_periods=warmup
    )
    predictions = study.predictions.table(include_preview=not CANONICAL_RUN).filter(
        pl.col("prediction_hash") == chosen.spec()["backtest_config"]["metadata"]["prediction_hash"]
    )
    if predictions.height != 1:
        raise RuntimeError(f"{label}: the selected prediction set is not uniquely resolvable")
    for level in cost_grid:
        execution = run_backtests(
            study,
            predictions=predictions,
            signal=strategy["signal"],
            allocation=allocation,
            costs={
                "model": "percentage",
                "commission_bps": level / 2,
                "slippage_bps": level / 2,
            },
            prices=prices,
            chapter="ch18",
            population_name=(
                f"crypto-cost-{label}-{level:g}bps-{POPULATION_SUFFIX}" if CANONICAL_RUN else None
            ),
        )
        cost_runs.extend(result.hash for result in execution.results)
        print(
            f"{label} @ {level:g} bps: {len(execution.results)} backtests registered\n"
            f"  this execution: {execution.n_computed} computed, "
            f"{execution.n_reused} served from the registry"
        )

# %% [markdown]
# ## 4. What came out
#
# Read back from the registry. `cost_bps` is recovered from each registered specification rather
# than carried over from the loop, so the table describes what was run and not what was intended.


# %%
def funding_metrics(study_root) -> pl.DataFrame:
    """Settled funding per registered backtest.

    The prediction and backtest catalogs project the metrics the pipeline shares across
    case studies, and funding is not one of them - it exists only where the instrument
    settles it. Reading it here keeps the column available without widening a shared
    catalog for one case study's economics.
    """
    with closing(
        sqlite3.connect(f"file:{study_root / 'run_log' / 'registry.db'}?mode=ro", uri=True)
    ) as db:
        rows = db.execute(
            "SELECT backtest_hash, funding_pnl, funding_events, funding_settlements "
            "FROM backtest_metrics"
        ).fetchall()
    return pl.DataFrame(
        rows,
        schema=["backtest_hash", "funding_pnl", "funding_events", "funding_settlements"],
        orient="row",
    )


# %%
commission_rate = pl.col("spec_json").str.json_path_match("$.backtest_config.commission.rate")
slippage_rate = pl.col("spec_json").str.json_path_match("$.backtest_config.slippage.rate")
# Named by hash, and not read back as "every cost_sensitivity row for these labels". The
# registry keeps every generation, so a run whose selected configuration changed - because the
# candidate set it came from was superseded - leaves the previous configuration's cost cells in
# place, and the row-count check below then fails on a valid re-run while the curve it did draw
# would have mixed two configurations.
curve = (
    study.backtests.table(include_preview=not CANONICAL_RUN)
    .filter(pl.col("backtest_hash").is_in(cost_runs))
    .with_columns(
        ((commission_rate.cast(pl.Float64) + slippage_rate.cast(pl.Float64)) * 10_000)
        .round(6)
        .alias("cost_bps")
    )
    .join(funding_metrics(STORAGE_ROOT), on="backtest_hash", how="left")
    .sort("label", "cost_bps")
)
if curve.filter(pl.col("funding_pnl").is_null()).height:
    raise RuntimeError("a registered cost cell has no settled funding recorded")
if curve.filter(~pl.col("complete")).height:
    raise RuntimeError("the cost sweep registered an incomplete result")
expected = len(labels) * len(cost_grid)
if curve.height != expected:
    raise RuntimeError(f"expected {expected} cost cells, the registry holds {curve.height}")

# %% [markdown]
# One row per label and cost level. `total_commission` and `total_slippage` are the dollars the
# engine charged; `funding_pnl` is the funding settled over the same period, which is what the
# cost level does not touch.

# %% tags=["results"]
curve.select(
    "label",
    "cost_bps",
    "sharpe",
    "total_return",
    "num_trades",
    "avg_turnover",
    "total_commission",
    "total_slippage",
    "funding_pnl",
)

# %% [markdown]
# ### The decay, and where it crosses zero
#
# The breakeven is the cost level at which the curve crosses zero Sharpe, found by interpolating
# between the two levels it crosses between rather than reported as the nearest grid point. A
# label whose curve never crosses has no breakeven to report, and that is stated rather than
# filled in: a strategy already below zero at no cost at all does not become viable at a lower
# cost, and one still above zero at 50 bps has more margin than the grid can measure.


# %%
def breakeven_bps(panel: pl.DataFrame) -> float | None:
    """Cost level where validation Sharpe crosses zero, linearly between bracketing levels."""
    rows = panel.sort("cost_bps").select("cost_bps", "sharpe").rows()
    for (low_cost, low_sharpe), (high_cost, high_sharpe) in zip(rows, rows[1:], strict=False):
        if (low_sharpe > 0) != (high_sharpe > 0):
            span = low_sharpe - high_sharpe
            if span == 0:
                return low_cost
            return low_cost + (high_cost - low_cost) * low_sharpe / span
    return None


# %%
breakevens = {label: breakeven_bps(curve.filter(pl.col("label") == label)) for label in labels}
for label in labels:
    panel = curve.filter(pl.col("label") == label).sort("cost_bps")
    crossing = breakevens[label]
    at_zero = panel.item(0, "sharpe")
    if crossing is not None:
        print(f"{label}: Sharpe {at_zero:.2f} at no cost, crossing zero near {crossing:.1f} bps")
    elif at_zero > 0:
        print(
            f"{label}: Sharpe {at_zero:.2f} at no cost, still above zero at {max(cost_grid):g} bps"
        )
    else:
        print(f"{label}: Sharpe {at_zero:.2f} at no cost, below zero across the whole grid")

# %% [markdown]
# One line per label. The vertical marker is the declared production total; the horizontal line
# is zero Sharpe. Where a line is already under the horizontal one at the left edge, the cost
# level is not what is wrong with that configuration.

# %%
fig = go.Figure()
palette = [COLORS["blue"], COLORS["amber"], COLORS["copper"], COLORS["slate"]]
for index, label in enumerate(labels):
    panel = curve.filter(pl.col("label") == label).sort("cost_bps")
    fig.add_trace(
        go.Scatter(
            x=panel.get_column("cost_bps").to_list(),
            y=panel.get_column("sharpe").to_list(),
            mode="lines+markers",
            name=label,
            line={"color": palette[index % len(palette)]},
        )
    )
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
fig.add_vline(
    x=declared_total,
    line_width=1,
    line_dash="dot",
    line_color=COLORS["neutral"],
    annotation_text=f"declared {declared_total:g} bps",
    annotation_position="top",
)
fig.update_layout(
    title={
        "text": "Validation Sharpe against round-trip cost"
        "<br><sup>One line per label, each holding its surviving configuration fixed</sup>",
        "x": 0.02,
        "xanchor": "left",
    },
    xaxis_title="Round-trip cost on traded notional (bps)",
    yaxis_title="Annualized validation Sharpe",
    height=520,
    width=1000,
)
show_plotly_with_alt(
    fig,
    "Line chart of annualized validation Sharpe against round-trip trading cost in basis points, "
    "one line per label. A dashed horizontal line marks zero Sharpe and a dotted vertical line "
    "marks the declared production cost total. Every line slopes downward as cost rises, and the "
    "spacing between the lines at the left edge is larger than the amount any of them falls "
    "across the whole grid.",
)

# %% [markdown]
# ### What the cost level moves and what it does not
#
# Two things are charged against this book, and only one of them is on the horizontal axis above.
# Commission and slippage are paid per trade, so they scale with the rate and with how much of the
# book turns over. Funding settles on the position that is held at each 8-hourly timestamp, so it
# is a cost of carrying the position rather than of establishing it, and the cost level does not
# reach it at all.
#
# That is why the funding column below is flat across the grid while the execution columns are
# not, and it is the reason a cost sweep on perpetual futures answers a narrower question than it
# does on an equity book: the friction a perpetuals strategy pays is only partly execution.

# %% tags=["results"]
curve.group_by("label").agg(
    levels=pl.len(),
    sharpe_at_zero=pl.col("sharpe").filter(pl.col("cost_bps") == 0).first(),
    sharpe_at_max=pl.col("sharpe").filter(pl.col("cost_bps") == max(cost_grid)).first(),
    execution_cost_range=(
        (pl.col("total_commission") + pl.col("total_slippage")).max()
        - (pl.col("total_commission") + pl.col("total_slippage")).min()
    ),
    funding_range=pl.col("funding_pnl").max() - pl.col("funding_pnl").min(),
    median_turnover=pl.col("avg_turnover").median(),
).sort("label")

# %% [markdown]
# ## 5. What to notice
#
# **Turnover is the multiplier.** A cost level is a rate, and a rate charges nothing until
# something trades. Two configurations at the same cost level lose different amounts of Sharpe,
# and the difference is how much of the book each one replaces at every rebalance. This is the
# link back to the previous stage: an allocator that spreads capital more evenly turns the book
# over more, so a sizing choice made on a Sharpe measured at one cost level is partly a bet on
# that cost level.
#
# **A breakeven is a property of the curve, not of a point.** The declared schedule is one
# assumption among the ones a reader might hold. Reporting the Sharpe at that assumption and
# stopping tells nobody whether the result would survive an execution desk that does slightly
# worse. The distance between the declared level and the crossing is the margin, and a strategy
# with no margin is one whose published result depends on the cost model being exactly right.
#
# **Funding does not appear on this axis.** Every level here re-runs the same funding settlement,
# because funding is charged on the position at each 8-hourly timestamp and has nothing to do with
# the trading rate. A perpetuals strategy therefore has two independent friction terms, and this
# notebook varies one. A strategy could be robust to execution cost and still be defeated by the
# funding it pays to hold the book.
#
# **Known limitations.** The grid is uniform across contracts, and real spreads on these venues
# are not - the majors clear far tighter than the alts, so a uniform level over-charges the liquid
# part of the book and under-charges the rest. The sweep also varies cost with the rebalance
# cadence fixed, and cadence is the other side of the same trade: trading less often pays less
# friction and reacts to the model more slowly. Neither is varied here, and neither is free.
#
# **Next**: [`16_risk_management`](16_risk_management.ipynb) returns to the candidate set this
# notebook read from and asks whether a position-level or portfolio-level control improves the
# configuration that survived, at the declared cost schedule.
