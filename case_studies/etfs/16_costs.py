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
# # ETF costs: where the edge runs out
#
# Every Sharpe ratio so far was earned at one cost level - the one `config/setup.yaml` declares.
# That single number hides the question a reader actually needs answered, which is not "what did it
# make" but **"how much friction would it take to make nothing"**. A strategy whose Sharpe is
# unchanged at four times the declared cost is a different proposition from one that breaks even
# just above it, even when the two report the same number today.
#
# This notebook walks a cost grid over the leading allocation-stage combinations and registers a
# backtest at each level, so the decay curve is a set of registered results rather than an
# extrapolation from one point.
#
# **Two cost regimes are swept, and they are not two views of one thing.** The case study's
# declared model is per-share commission plus a half-spread in cents, which is how an ETF actually
# trades: the cost of a share does not scale with its price. The basis-point grid is the convention
# most of the literature uses and most other case studies here declare, and it is run alongside as
# a comparator. The two disagree by construction - a one-cent half-spread is about one basis point
# on a $500 fund and about five on a $20 one - so the panels are read for their slopes and their
# breakevens, not point against point.
#
# **Learning objectives**
#
# - Read a cost decay curve and locate the level at which a strategy stops paying.
# - Say why a per-share cost model and a basis-point cost model give different answers on the same
#   universe, and which one this universe is priced under.
# - Say what a uniform cost sweep does and does not tell you about a strategy priced under a
#   per-asset spread map.
#
# **Book reference**: Chapter 18, Sections 18.2 to 18.5.
#
# **Prerequisites**: [`15_portfolio_management`](15_portfolio_management.ipynb), whose allocation
# stage supplies the combinations this sweep re-prices.
#
# **What it writes**: one row in `backtest_runs` per combination and cost level, at
# `stage='cost_sensitivity'`, under both regimes.
# [`17_risk_management`](17_risk_management.ipynb) takes the same allocation-stage combinations and
# adds risk overlays to them.

# %%
"""Re-price the leading ETF allocation combinations across two cost-model grids."""

import json
import time
import warnings

import plotly.graph_objects as go
import polars as pl
import yaml
from plotly.subplots import make_subplots

from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import (
    clone_backtest_spec,
    ensure_backtest_spec,
    set_backtest_costs_bps,
    set_backtest_costs_per_share,
    strategy_view,
)
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.registry import (
    read_predictions,
    resolve_best_backtest_runs,
)
from case_studies.utils.sweep_config import (
    get_cost_grid_bps,
    get_cost_grid_half_spread_usd,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir
from utils.style import COLORS, ml4t_palette, show_plotly_with_alt

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"
LABEL = ""
MAX_SYMBOLS = 0
TOP_N_COMBOS: int | None = None

# %% [markdown]
# ## 1. What is being re-priced, and under what
#
# The combinations are the allocation-stage leaders by Sharpe. They are re-priced rather than
# re-selected: the prediction, the concentration and the allocator are held exactly as registered,
# and the only thing that moves is the cost model. That is what makes the curve below a statement
# about cost sensitivity and not about which strategy happens to do best under friction.
#
# The per-share commission is read from `setup.yaml` with no default. Elsewhere in the fleet that
# key is exploratory and a missing value can fall back; here it is the headline regime, so a
# missing key has to fail loudly rather than quietly re-price the whole sweep at somebody's
# default.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "cost_sensitivity")
if not LABEL:
    LABEL = bt_config.primary_label

COST_GRID_BPS = get_cost_grid_bps(CASE_STUDY_ID)
COST_GRID_HALF_SPREAD_USD = get_cost_grid_half_spread_usd(CASE_STUDY_ID)
PER_SHARE_COMMISSION = float(
    yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())["costs"]["per_share"]
)

print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}")
print(f"Per-share commission: ${PER_SHARE_COMMISSION}/share")
print(f"Basis-point grid:     {COST_GRID_BPS}")
print(f"Half-spread grid (¢): {[round(v * 100, 2) for v in COST_GRID_HALF_SPREAD_USD]}")

# %% tags=["results"]
top_combos = resolve_best_backtest_runs(
    CASE_STUDY_ID, LABEL, split="validation", stage="allocation", top_n=TOP_N_COMBOS
)
if top_combos.is_empty():
    raise RuntimeError(
        "no allocation-stage backtests are registered, so there is nothing to re-price; "
        "run 15_portfolio_management first"
    )
for row in top_combos.iter_rows(named=True):
    allocator = strategy_view(json.loads(row["spec_json"])).get("allocation", {}).get("method")
    print(
        f"  Sharpe={row['sharpe']:+.3f}  {row['family']}/{row['config_name']}  "
        f"alloc={allocator}  backtest={row['backtest_hash'][:8]}"
    )

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique()} tradeable funds")

# %% [markdown]
# ## 2. Sweeping both grids
#
# **The sweep applies one flat rate to every fund, which the production backtests do not.** The
# signal, allocation and risk stages price each fund from the tiered per-asset half-spread map in
# `setup.yaml` - a half cent on the largest funds, a cent on the sector funds, two cents by
# default. What follows is therefore one-axis sensitivity to a universe-wide cost level, not a
# re-pricing of the live cost structure. Read the slope and the crossing point; do not read a
# single point as what the strategy would have earned.


# %%
def sweep_costs(regime: str, grid, apply_costs) -> tuple[int, list[dict]]:
    """Re-price every leading combination at every level of one cost grid.

    Returns the number of backtests registered and the failures, each with the exception that
    caused it. A count on its own cannot distinguish a sweep that lost one level from one that
    lost every level of one regime, and the two mean very different things about the curve.
    """
    registered, failures = 0, []
    started = time.time()
    total = len(top_combos) * len(grid)

    for combo_row in top_combos.iter_rows(named=True):
        pred_hash = combo_row["prediction_hash"]
        base_spec = ensure_backtest_spec(
            CASE_STUDY_ID,
            bt_config,
            json.loads(combo_row["spec_json"]),
            prices=prices,
            prediction_hash=pred_hash,
            initial_cash=bt_config.initial_cash,
        )
        allocator = strategy_view(base_spec).get("allocation", {}).get("method", "equal_weight")
        predictions = read_predictions(CASE_STUDY_ID, pred_hash)

        for level in grid:
            spec = apply_costs(clone_backtest_spec(base_spec), level)
            spec["chapter"] = "ch18"
            try:
                result = run_backtest(
                    CASE_STUDY_ID,
                    pred_hash,
                    spec,
                    prices=prices,
                    predictions=predictions,
                    label=LABEL,
                    register=True,
                    initial_cash=bt_config.initial_cash,
                    calendar=bt_config.calendar,
                )
            except Exception as error:
                failures.append(
                    {
                        "regime": regime,
                        "allocator": allocator,
                        "level": level,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            registered += 1
            print(
                f"  [{registered}/{total}] {regime} {allocator} @ {level}: "
                f"Sharpe={result.metrics.get('sharpe', 0):+.3f}"
            )

    print(
        f"{regime} sweep: {registered} registered, {len(failures)} failed, "
        f"{time.time() - started:.0f}s"
    )
    return registered, failures


bps_done, bps_failures = sweep_costs(
    "bps",
    COST_GRID_BPS,
    lambda spec, level: set_backtest_costs_bps(
        spec, commission_bps=level / 2, slippage_bps=level / 2
    ),
)
ps_done, ps_failures = sweep_costs(
    "per-share",
    COST_GRID_HALF_SPREAD_USD,
    lambda spec, level: set_backtest_costs_per_share(
        spec, per_share=PER_SHARE_COMMISSION, default_half_spread_usd=level
    ),
)

# %%
failures = bps_failures + ps_failures
if failures:
    failure_frame = pl.DataFrame(failures)
    print(f"{failure_frame.height} backtests raised. Distinct causes:")
    print(failure_frame.group_by("regime", "error").len().sort("len", descending=True))
else:
    print("no backtest raised in either regime")

# %% [markdown]
# ## 3. The decay curves
#
# Read back from the registry rather than from the sweep, so a resumed run and a fresh one show the
# same thing. Each row's cost level and allocator come out of its own registered specification,
# which is what the backtest hash was taken over.

# %%
COST_REGIMES = {"percentage": "bps per leg", "per_share": "cents of half-spread per share"}


def load_cost_curve(commission_model: str) -> pl.DataFrame:
    """Read the registered cost-sensitivity rows for one commission model."""
    rows = resolve_best_backtest_runs(
        CASE_STUDY_ID, LABEL, split="validation", stage="cost_sensitivity", top_n=100000
    )
    if rows.is_empty():
        return pl.DataFrame()
    parsed = []
    for row in rows.iter_rows(named=True):
        spec = json.loads(row["spec_json"])
        config = spec.get("backtest_config", {})
        commission, slippage = config.get("commission", {}), config.get("slippage", {})
        if commission.get("model") != commission_model:
            continue
        level = (
            round((commission.get("rate", 0.0) + slippage.get("rate", 0.0)) * 10_000.0, 4)
            if commission_model == "percentage"
            else round(slippage.get("spread", 0.0) * 100.0, 4)
        )
        parsed.append(
            {
                "level": level,
                "sharpe": row["sharpe"],
                "allocator": strategy_view(spec)
                .get("allocation", {})
                .get("method", "equal_weight"),
            }
        )
    if not parsed:
        return pl.DataFrame()
    return (
        pl.DataFrame(parsed)
        .group_by("allocator", "level")
        .agg(pl.col("sharpe").mean())
        .sort("allocator", "level")
    )


bps_curve = load_cost_curve("percentage")
ps_curve = load_cost_curve("per_share")
if bps_curve.is_empty() and ps_curve.is_empty():
    raise RuntimeError("the cost-sensitivity stage registered no readable rows")
print(f"basis-point regime: {bps_curve.height} allocator-level points")
print(f"per-share regime:   {ps_curve.height} allocator-level points")

# %% [markdown]
# ### Where each regime crosses zero
#
# The breakeven is the cost level at which the mean Sharpe crosses zero, interpolated between the
# two grid points that bracket it. A curve that never crosses within the grid has no breakeven to
# report, and saying so is the answer: the grid did not reach far enough to find one.


# %% tags=["results"]
def breakeven(curve: pl.DataFrame) -> float | None:
    """The cost level where the mean Sharpe first crosses zero, or None if it never does."""
    if curve.is_empty():
        return None
    mean_curve = curve.group_by("level").agg(pl.col("sharpe").mean()).sort("level")
    levels = mean_curve["level"].to_list()
    sharpes = mean_curve["sharpe"].to_list()
    for (low, low_sharpe), (high, high_sharpe) in zip(
        zip(levels, sharpes, strict=True), zip(levels[1:], sharpes[1:], strict=True), strict=False
    ):
        if low_sharpe >= 0 > high_sharpe:
            return low + (high - low) * low_sharpe / (low_sharpe - high_sharpe)
    return None


for name, curve, unit in [
    ("basis points per leg", bps_curve, "bps"),
    ("cents of half-spread", ps_curve, "¢"),
]:
    crossing = breakeven(curve)
    if curve.is_empty():
        print(f"{name}: no registered rows")
    elif crossing is None:
        top = curve["level"].max()
        print(f"{name}: mean Sharpe does not cross zero anywhere up to {top}{unit}")
    else:
        print(f"{name}: mean Sharpe crosses zero at {crossing:.2f}{unit}")

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("Basis points per leg (comparator)", "Cents of half-spread (declared regime)"),
)
allocators = sorted(
    set(bps_curve["allocator"].to_list() if not bps_curve.is_empty() else [])
    | set(ps_curve["allocator"].to_list() if not ps_curve.is_empty() else [])
)
palette = ml4t_palette(max(len(allocators), 1), categorical=True)
for column, curve in ((1, bps_curve), (2, ps_curve)):
    if curve.is_empty():
        continue
    for index, allocator in enumerate(allocators):
        subset = curve.filter(pl.col("allocator") == allocator).sort("level")
        if subset.is_empty():
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["level"].to_list(),
                y=subset["sharpe"].to_list(),
                mode="lines+markers",
                name=allocator,
                legendgroup=allocator,
                showlegend=column == 1,
                line=dict(color=palette[index % len(palette)]),
            ),
            row=1,
            col=column,
        )
    fig.add_hline(
        y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=1, col=column
    )
fig.update_xaxes(title_text="Total cost, bps per leg", row=1, col=1)
fig.update_xaxes(title_text="Half-spread, cents per share", row=1, col=2)
fig.update_yaxes(title_text="Net Sharpe ratio", row=1, col=1)
fig.update_layout(
    title="Two cost models disagree about the same strategy",
    height=460,
    width=980,
    margin=dict(t=110),
)
_all_sharpes = pl.concat(
    [curve["sharpe"] for curve in (bps_curve, ps_curve) if not curve.is_empty()]
)
show_plotly_with_alt(
    fig,
    "Two side-by-side line charts of net Sharpe ratio against transaction cost, one line per "
    "allocator, sharing a y-axis, each with a dashed line at zero. The left panel's x-axis is "
    "basis points per leg and the right panel's is cents of half-spread per share. Counted from "
    f"the frames: {bps_curve.height} points on the left and {ps_curve.height} on the right, across "
    f"{len(allocators)} allocators, net Sharpe from {_all_sharpes.min():+.2f} to "
    f"{_all_sharpes.max():+.2f}.",
)

# %% [markdown]
# ## 4. What to notice
#
# **The rebalancing cadence is what buys cost tolerance, and it is a design choice rather than a
# result.** A position held for a month pays its round trip once and earns a month of return
# against it; the same position held for a day pays it about twenty times over the same month. That
# is why the breakeven printed above is where it is, and it is the single most consequential thing
# the label horizon decided back in [`02_labels`](02_labels.ipynb).
#
# **The two regimes are two different questions, and the declared one is the per-share panel.** A
# basis-point cost says friction scales with the value traded. A per-share cost says it scales with
# the number of shares, which is what an ETF spread actually does - so the same half-spread is
# cheap on a high-priced fund and expensive on a low-priced one, and a universe holding both is
# mis-priced by either model applied uniformly. The gap between the panels is the size of that
# convention's effect on this universe.
#
# **A breakeven is not a safety margin.** It is where the mean Sharpe reaches zero on a curve
# fitted to validation folds, under a uniform cost, with no market impact and no capacity limit.
# The distance between the declared cost and that crossing is worth knowing and is not a promise.
#
# **Known limitations.** The sweep applies one flat rate to every fund while the production stages
# price each from the tiered map, so this is sensitivity to a level rather than a re-pricing.
# Nothing here models impact, so a larger book than the declared initial cash would face costs this
# curve does not contain. The combinations re-priced are the allocation-stage leaders, so the curve
# describes how the strategies that already did well degrade, not how the whole population does.
# And every point is measured on validation folds; the holdout is not consulted.

# %% [markdown]
# **Next**: [`17_risk_management`](17_risk_management.ipynb) adds position and portfolio risk rules
# to the same combinations and asks whether they improve the drawdown without spending the Sharpe.
