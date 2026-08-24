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
# # Friction Survival: Where the Edge Dies
#
# **Docker image**: `ml4t`
#
# This notebook cross-cuts the case studies by **failure mode**: gross-to-net
# Sharpe degradation, breakeven cost thresholds, cadence-frequency vulnerability,
# and cost-model realism caveats. Ch18 establishes the cost taxonomy and the
# per-asset-class machinery; this notebook reads the resulting Ch18 cost-sweep
# backtests directly out of each case study's registry and asks which
# strategies survive the friction of real trading.
#
# **Learning Objectives**:
# - Quantify each strategy's sensitivity to per-leg friction on a common scale
# - Identify breakeven cost thresholds per rebalance cadence
# - Recognize when a generic basis-point sweep is the wrong cost model
#   (single-name options, intraday equities)
#
# **Book Reference**: Chapter 20, Section 20.6 (Trading Realism)
#
# **Prerequisites**: Run [`01_aggregate_synthesis`](01_aggregate_synthesis.ipynb) first.
# Each case study's registry must contain Ch18 `cost_sensitivity`-stage backtests.

# %%
"""Ch20 Friction Survival — cross-case-study cost-sweep analysis from registry."""

import json
import warnings

import matplotlib.pyplot as plt
import polars as pl
from IPython.display import Markdown, display
from matplotlib.patches import Patch

from case_studies.utils.analytics import (
    CASE_STUDY_IDS,
    SHORT_NAMES,
    load_carrier_cost_curves,
)
from utils.paths import get_chapter_dir
from utils.style import show_with_alt

warnings.filterwarnings("ignore")
pl.Config.set_tbl_rows(20)

# %% tags=["parameters"]
MAX_CASE_STUDIES = 0  # 0 = all

# %%
CS_LIST = CASE_STUDY_IDS[:MAX_CASE_STUDIES] if MAX_CASE_STUDIES else CASE_STUDY_IDS
DEFERRED_V31_CASE_STUDIES = {"nasdaq100_microstructure"}
ACTIVE_CS_LIST = [cs for cs in CS_LIST if cs not in DEFERRED_V31_CASE_STUDIES]

# %% [markdown]
# ## Load Cost Sweep Results from Registry
#
# Ch18 backtests vary commission + slippage across a grid of cost levels
# while holding the signal and allocation constant. We read the sweep for
# each case study's **release carrier** -- the declared configuration across
# the signal, allocation, and risk-overlay stages --
# so the breakeven measured here is the cost survival of the strategy the
# chapter actually deploys, not of whichever allocator happened to be best
# at zero cost. NASDAQ-100 is excluded from the v3.0 cross-case cost surface:
# its bounded active scope has no corrected carrier cost grid, so that broad
# regeneration is deferred to v3.1 rather than mixed with historical timing.

# %%
costs_df = load_carrier_cost_curves(ACTIVE_CS_LIST)
print("Deferred to v3.1: NASDAQ-100 timing-corrected broad carrier cost grid")

if costs_df.is_empty():
    msg = "No Ch18 cost-sensitivity backtests found for any deployed carrier"
    raise RuntimeError(msg)

n_cs = costs_df["case_study"].n_unique()
print(f"Loaded {len(costs_df)} carrier cost-sweep entries across {n_cs} case studies")
costs_df.head(5)

# %% [markdown]
# Assumed per-leg cost and rebalance cadence both come from each case study's
# `setup.yaml`, read back through the Ch20 artifacts that `01_aggregate_synthesis`
# writes, rather than being typed into this notebook.
#
# `setup.yaml` records a specific cadence such as `monthly_month_end` or
# `daily_ny_close`. The charts group by period, taken from the leading word, and
# anything unrecognised is grouped as unspecified rather than given a default.
# The turnover multipliers attached to each period say how often a book of that
# cadence is assumed to turn over relative to a daily one. They are an
# assumption, not turnover measured from the backtests.
#
# ## Gross-to-Net Sharpe Degradation
#
# For each case study we compare the zero-cost (gross) Sharpe with the Sharpe at
# the cost that case study actually assumes, which comes from its `setup.yaml`
# by way of `overview.parquet`. The assumed cost rarely falls on a grid point, so
# the net Sharpe is interpolated linearly between the two grid points that
# bracket it, and the bracketing points are reported alongside.
#

# %%
gross_df = costs_df.filter(pl.col("cost_bps") == 0)
net_df = costs_df.filter(pl.col("cost_bps") > 0)

# One allocator per case study (the carrier's); this selects it.
best_alloc = (
    gross_df.sort("sharpe", descending=True)
    .unique(subset=["case_study"], keep="first")
    .select("case_study", "allocator")
)

# %%
_overview = pl.read_parquet(get_chapter_dir(20) / "output" / "overview.parquet")
ASSUMED_COST_BPS = dict(_overview.select("cs_id", "cost_bps").iter_rows())
_synthesis = json.loads((get_chapter_dir(20) / "output" / "all_synthesis.json").read_text())
CADENCE_BY_CS = {
    cs: (data["meta"].get("cadence") or "unspecified") for cs, data in _synthesis.items()
}

CADENCE_PERIODS = ("15min", "hourly", "8_hour", "daily", "weekly", "monthly")
TURNOVER_MULTIPLIER = {
    "15min": 26.0,
    "hourly": 6.5,
    "8_hour": 3.0,
    "daily": 1.0,
    "weekly": 1.0 / 5,
    "monthly": 1.0 / 21,
}


def _cadence_period(cadence: str) -> str:
    for period in CADENCE_PERIODS:
        if cadence.startswith(period):
            return period
    return "unspecified"


def _sharpe_at(curve: pl.DataFrame, cost_bps: float) -> tuple[float, float, float]:
    """Sharpe at `cost_bps`, linearly interpolated on the sweep grid.

    Returns the interpolated Sharpe and the two grid costs it sits between. A
    cost beyond either end of the grid is clamped to that end, which is reported
    by the bracket coming back equal.
    """
    grid = curve["cost_bps"].to_list()
    vals = curve["sharpe"].to_list()
    if cost_bps <= grid[0]:
        return vals[0], grid[0], grid[0]
    if cost_bps >= grid[-1]:
        return vals[-1], grid[-1], grid[-1]
    for lo, hi, v_lo, v_hi in zip(grid, grid[1:], vals, vals[1:], strict=False):
        if lo <= cost_bps <= hi:
            w = 0.0 if hi == lo else (cost_bps - lo) / (hi - lo)
            return v_lo + w * (v_hi - v_lo), lo, hi
    return vals[-1], grid[-1], grid[-1]


def _breakeven(curve: pl.DataFrame) -> tuple[float, bool]:
    """Cost at which Sharpe crosses zero, and whether that crossing was observed.

    Interpolates between the last positive grid point and the first negative one.
    A curve still positive at the top of the grid is censored: the breakeven is
    somewhere above the ceiling and the ceiling is not it.
    """
    grid = curve["cost_bps"].to_list()
    vals = curve["sharpe"].to_list()
    for lo, hi, v_lo, v_hi in zip(grid, grid[1:], vals, vals[1:], strict=False):
        if v_lo > 0 >= v_hi:
            w = v_lo / (v_lo - v_hi)
            return lo + w * (hi - lo), True
    return (grid[-1], False) if vals[-1] > 0 else (0.0, True)


summary_rows = []
for row in best_alloc.iter_rows(named=True):
    cs = row["case_study"]
    alloc = row["allocator"]
    curve = costs_df.filter((pl.col("case_study") == cs) & (pl.col("allocator") == alloc)).sort(
        "cost_bps"
    )
    if curve.height < 2 or curve["cost_bps"].min() > 0:
        continue

    gross_sharpe = curve["sharpe"][0]
    assumed_cost = ASSUMED_COST_BPS.get(cs)
    if assumed_cost is None:
        msg = f"No cost_bps for {cs} in overview.parquet; re-run 01_aggregate_synthesis"
        raise RuntimeError(msg)
    net_sharpe, bracket_lo, bracket_hi = _sharpe_at(curve, assumed_cost)
    breakeven_bps, breakeven_observed = _breakeven(curve)

    summary_rows.append(
        {
            "case_study": cs,
            "display_name": SHORT_NAMES.get(cs, cs),
            "cadence": CADENCE_BY_CS.get(cs, "unspecified"),
            "cadence_period": _cadence_period(CADENCE_BY_CS.get(cs, "unspecified")),
            "allocator": alloc,
            "gross_sharpe": round(gross_sharpe, 3),
            "net_sharpe": round(net_sharpe, 3),
            "sharpe_drag": round(gross_sharpe - net_sharpe, 3),
            "drag_pct": round(100 * (gross_sharpe - net_sharpe) / gross_sharpe, 1)
            if gross_sharpe != 0
            else 0.0,
            "assumed_cost_bps": assumed_cost,
            "grid_bracket": f"{bracket_lo:g}-{bracket_hi:g}",
            "breakeven_bps": round(breakeven_bps, 1),
            "breakeven_observed": breakeven_observed,
            "survives": net_sharpe > 0,
        }
    )

# %%
summary = pl.DataFrame(summary_rows).sort("drag_pct", descending=True)
print("=== Gross-to-Net Sharpe Degradation ===")
summary.select(
    "display_name",
    "cadence",
    "gross_sharpe",
    "assumed_cost_bps",
    "grid_bracket",
    "net_sharpe",
    "sharpe_drag",
    "drag_pct",
    "breakeven_bps",
    "breakeven_observed",
    "survives",
)

# %% tags=["results"]
_dead = summary.filter(~pl.col("survives"))
_censored = summary.filter(~pl.col("breakeven_observed"))
display(
    Markdown(
        f"{summary.height} case studies have a carrier cost sweep. "
        + (
            f"**{', '.join(_dead['display_name'].to_list())}** "
            f"{'has' if _dead.height == 1 else 'have'} a negative Sharpe at the "
            "cost the case study assumes, so the strategy does not survive its "
            "own cost model. "
            if _dead.height
            else "All of them keep a positive Sharpe at the cost they assume. "
        )
        + f"Cost consumes between {summary['drag_pct'].min():.1f} and "
        f"{summary['drag_pct'].max():.1f} percent of gross Sharpe.\n\n"
        + (
            f"For {', '.join(_censored['display_name'].to_list())} the Sharpe is "
            f"still positive at the top of the swept grid, so the breakeven "
            "column is a lower bound rather than a measurement: it is at least "
            "that, and the grid does not say how much more."
            if _censored.height
            else "Every breakeven was observed inside the swept grid."
        )
    )
)

# %% [markdown]
# ## Cost Drag Visualization
#
# The horizontal bar chart shows Sharpe drag (gross minus net) for each
# case study, ordered by severity. Higher-frequency strategies typically
# suffer more because they accumulate turnover costs faster.

# %%
fig, ax = plt.subplots(figsize=(10, 6))

colors = [
    "#d62728" if drag > 50 else "#ff7f0e" if drag > 20 else "#2ca02c"
    for drag in summary["drag_pct"]
]

bars = ax.barh(
    range(len(summary)),
    summary["drag_pct"].to_list(),
    color=colors,
    edgecolor="none",
    height=0.6,
)
ax.set_yticks(range(len(summary)))
ax.set_yticklabels(summary["display_name"].to_list())
ax.set_xlabel("Sharpe Drag (%)")
ax.set_title("Cost Impact: Gross-to-Net Sharpe Degradation")
ax.invert_yaxis()

for bar, row in zip(bars, summary.iter_rows(named=True), strict=False):
    ax.annotate(
        f"BE: {'' if row['breakeven_observed'] else '>'}{row['breakeven_bps']:g} bps",
        xy=(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2),
        va="center",
        fontsize=9,
        color="gray",
    )
# Headroom so the breakeven annotation on the widest bar (FX) is not clipped.
ax.set_xlim(right=max(summary["drag_pct"]) * 1.28)

fig.tight_layout()
show_with_alt(
    fig,
    "Horizontal bars giving the percentage of gross Sharpe consumed by each case "
    "study's assumed cost, ordered by severity, each annotated with the cost at "
    "which that strategy breaks even.",
)

# %% [markdown]
# ## Breakeven Cost Thresholds by Frequency
#
# Breakeven cost is the maximum per-leg cost (in bps) at which the
# deployed carrier still produces a positive Sharpe ratio. It is the cost
# budget that the signal supports before becoming unprofitable.

# %%
freq_order = list(CADENCE_PERIODS)
freq_colors = {
    "15min": "#d62728",
    "hourly": "#e8833a",
    "8_hour": "#ff7f0e",
    "daily": "#1f77b4",
    "weekly": "#5aa469",
    "monthly": "#2ca02c",
}

fig, ax = plt.subplots(figsize=(10, 5))

for i, row in enumerate(summary.sort("breakeven_bps").iter_rows(named=True)):
    color = freq_colors.get(row["cadence_period"], "gray")
    ax.barh(i, row["breakeven_bps"], color=color, height=0.6, edgecolor="none")

ax.set_yticks(range(len(summary)))
sorted_names = summary.sort("breakeven_bps")["display_name"].to_list()
ax.set_yticklabels(sorted_names)
ax.set_xlabel("Breakeven Cost (bps per leg)")
ax.set_title("Breakeven Cost Thresholds — Higher Is More Robust")

legend_handles = [Patch(facecolor=freq_colors[f], label=f) for f in freq_order if f in freq_colors]
ax.legend(handles=legend_handles, loc="lower right", title="Cadence")

fig.tight_layout()
show_with_alt(
    fig,
    "Horizontal bars of the breakeven per-leg cost for each case study, ordered "
    "from lowest to highest and coloured by rebalance cadence.",
)

# %% [markdown]
# ## Cost Drag Curves
#
# For each case study, plot Sharpe ratio as a function of per-leg
# cost. This reveals the "cost cliff" — the point where a profitable
# strategy becomes unprofitable.

# %%
best_alloc_map = dict(
    zip(best_alloc["case_study"].to_list(), best_alloc["allocator"].to_list(), strict=False)
)

fig, ax = plt.subplots(figsize=(12, 7))

for cs_id in CS_LIST:
    alloc = best_alloc_map.get(cs_id)
    if alloc is None:
        continue

    cs_data = costs_df.filter(
        (pl.col("case_study") == cs_id) & (pl.col("allocator") == alloc)
    ).sort("cost_bps")

    if cs_data.is_empty():
        continue

    ax.plot(
        cs_data["cost_bps"].to_list(),
        cs_data["sharpe"].to_list(),
        marker="o",
        markersize=4,
        label=SHORT_NAMES.get(cs_id, cs_id),
    )

ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
ax.set_xlabel("Per-Leg Cost (bps)")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Cost sensitivity: Sharpe against per-leg cost")
ax.legend(loc="upper right", fontsize=9, ncol=2)

# Mark each case study's assumed cost so the curve can be read at the point that
# matters rather than across the whole grid.
for cs_id in best_alloc_map:
    _c = ASSUMED_COST_BPS.get(cs_id)
    if _c is not None:
        ax.axvline(_c, color="gray", alpha=0.25, linewidth=0.8, linestyle=":")

fig.tight_layout()
show_with_alt(
    fig,
    "Line chart of Sharpe against per-leg cost in basis points, one line per "
    "case study over the swept grid, with a reference line at zero Sharpe and "
    "faint vertical lines marking each case study's assumed cost.",
)

# %% [markdown]
# ## Cost Survival Classification
#
# Each case study is classified by cost resilience: the ratio of its breakeven to
# the per-leg cost it is assumed to pay. A higher ratio means more headroom once
# realistic frictions are imposed. A ratio below one means the breakeven sits
# under the assumed cost, so the strategy is already losing money at its own
# assumption, and it is classified apart from a thin but positive margin.
#
# The assumed cost is the one already in `summary`, read from each case study's
# own setup rather than declared again here, so this table and the degradation
# table above cannot disagree about what a case study is assumed to pay.

# %%
survival = summary.with_columns(
    cost_margin_bps=(pl.col("breakeven_bps") - pl.col("assumed_cost_bps")),
    cost_margin_ratio=(pl.col("breakeven_bps") / pl.col("assumed_cost_bps").clip(lower_bound=1)),
).with_columns(
    resilience=pl.when(pl.col("cost_margin_ratio") < 1)
    .then(pl.lit("does not survive"))
    .when(pl.col("cost_margin_ratio") >= 10)
    .then(pl.lit("very robust"))
    .when(pl.col("cost_margin_ratio") >= 3)
    .then(pl.lit("robust"))
    .when(pl.col("cost_margin_ratio") >= 1.5)
    .then(pl.lit("marginal"))
    .otherwise(pl.lit("fragile")),
)

print("=== Cost Survival Classification ===")
survival.select(
    "display_name",
    "cadence",
    "assumed_cost_bps",
    "net_sharpe",
    "breakeven_bps",
    "breakeven_observed",
    "cost_margin_ratio",
    "resilience",
)

# %% tags=["results"]
_res = survival.group_by("resilience").agg(cs=pl.col("display_name")).sort("resilience")
display(
    Markdown(
        "; ".join(
            f"**{r['resilience']}**: {', '.join(sorted(r['cs']))}"
            for r in _res.iter_rows(named=True)
        )
        + ". A ratio is only as good as the breakeven behind it, and where the "
        "sweep never crossed zero the breakeven is the grid ceiling rather than "
        "a crossing, so the ratio for those is a lower bound too."
    )
)

# %% [markdown]
# ## S&P 500 Options: Spread Realism Caveat
#
# The S&P 500 Options case study was validated using executable-label
# backtesting, pricing straddle entries and exits at actual bid/ask quotes rather
# than at an assumed bps cost. That case study has no carrier cost sweep, so it
# does not appear in any table above; the figures below are quoted from its own
# evaluation and are not computed here.
#
# - **Median round-trip spread**: 1091 bps of premium (10.9%)
# - **Best executable Sharpe**: −1.05 (across 5 predictions × 8 schemes)
# - **Three-label decomposition** (best GBM, `leaves_15_mae`, `ew_top5`):
#   mid-unhedged Sharpe = +2.70, mid-DH Sharpe = +0.43, executable Sharpe = −1.50
# - **Spread-adjusted ranking** (optimizing signal + spread jointly) improves
#   Sharpe from −1.50 to −0.30, but stays negative
#
# The ML signal is real (IC = 0.068), but the 15.4 pp average spread impact
# per trade overwhelms the per-period signal. A generic bps cost sweep
# misrepresents this case study because the cost is predominantly the
# bid-ask spread, not commission. The teaching point is that strategy
# design must jointly optimize for signal quality and execution costs:
# single-stock option spreads are the binding constraint, not model
# quality.

# %% [markdown]
# ## Cadence–Frequency–Cost Regime
#
# The same IC translates to very different tradability depending on
# rebalance cadence. A 15-minute strategy accumulates ~25× more turnover
# per day than a daily strategy, and ~500× more than a monthly one.
# This creates distinct cost regimes:

# %%
if not summary.is_empty():
    regime = summary.with_columns(
        turnover_mult=pl.col("cadence_period").replace_strict(
            TURNOVER_MULTIPLIER,
            default=1.0,
            return_dtype=pl.Float64,
        ),
    )

# %%
if not summary.is_empty():
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Turnover-mult on x (varies 0.05→26×); breakeven on y. Both log so the
    # high-frequency cluster (NQ100/Crypto) and the monthly cluster
    # separate cleanly instead of stacking on a constant-x degenerate column.
    assumed_floor = max(float(summary["assumed_cost_bps"].min()), 0.5)

    # Monthly carriers share x (turnover ≈ 0.05) and pair up on y: ETFs and
    # US Firms at 50, CME and SP500 Eq+Opt at 30. Fan their labels vertically
    # so the two pairs stay legible despite the superimposed markers.
    label_offsets = {
        "NQ100": (10, 4),
        "Crypto": (10, 4),
        "FX": (10, 4),
        "US Equities": (10, 4),
        "ETFs": (10, 16),
        "US Firms": (10, 2),
        "SP500 Eq+Opt": (10, -2),
        "CME Futures": (10, -16),
        "SP500 Options": (10, 4),
    }

    for row in regime.iter_rows(named=True):
        color = freq_colors.get(row["cadence_period"], "gray")
        size = max(60, min(360, row["turnover_mult"] ** 0.5 * 120))
        ax.scatter(
            row["turnover_mult"],
            max(row["breakeven_bps"], 0.5),
            s=size,
            c=color,
            edgecolors="white",
            linewidth=1.2,
            zorder=5,
        )
        dx, dy = label_offsets.get(row["display_name"], (8, 8))
        ax.annotate(
            row["display_name"],
            (row["turnover_mult"], max(row["breakeven_bps"], 0.5)),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            zorder=6,
        )

    ax.axhline(
        assumed_floor,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        zorder=3,
        label=f"Survival floor ({assumed_floor:.0f} bps assumed cost)",
    )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(0.03, 60)
    ax.set_ylim(-0.5, 600)

    ax.set_xlabel("Turnover multiplier vs daily (log)")
    ax.set_ylabel("Breakeven cost — bps per leg (symlog)")
    ax.set_title("Cost Regimes: Higher-Frequency Strategies Face Steeper Cliffs")

    legend_handles = [
        Patch(facecolor=freq_colors[f], label=f) for f in freq_order if f in freq_colors
    ]
    ax.legend(
        handles=legend_handles + [ax.get_lines()[0]],
        loc="upper right",
        title="Cadence",
        framealpha=0.9,
    )

    fig.tight_layout()
    show_with_alt(
        fig,
        "Log-log scatter of breakeven cost against assumed relative turnover, one "
        "marker per case study coloured by cadence, with a horizontal line at the "
        "lowest assumed cost in the panel.",
    )

# %% [markdown]
# %% tags=["results"]
_reg = regime.sort("turnover_mult", descending=True)
display(
    Markdown(
        "Marker x-position is assumed per-day turnover relative to a daily "
        "strategy, y is the cost at which net Sharpe crosses zero. The turnover "
        "multipliers are an assumption written into this notebook, not a "
        "measurement from the backtests: they say how often a book of a given "
        "cadence is expected to turn over, and the chart uses them to place the "
        "case studies rather than to test them.\n\n"
        + "; ".join(
            f"**{r['display_name']}** ({r['cadence']}), breakeven "
            f"{'' if r['breakeven_observed'] else 'at least '}"
            f"{r['breakeven_bps']:g} bps against an assumed "
            f"{r['assumed_cost_bps']:g}"
            for r in _reg.iter_rows(named=True)
        )
        + ".\n\nThe cadences present here span a narrow part of the range the "
        "chart is drawn for. The high-frequency corner is empty: NASDAQ-100 is "
        "excluded pending a corrected carrier cost grid, and the other "
        "sub-daily case studies have no cost sweep. Nothing here tests whether "
        "turnover or signal strength sets the breakeven, because the case "
        "studies that would separate them are the ones missing."
    )
)

# %% [markdown]
# ## Key Takeaways
#
# - **A cost sweep is only informative at the cost the strategy assumes.** The
#   gross Sharpe and the Sharpe at the top of the grid are both easy to read off
#   and neither is the number that decides whether the strategy is tradable. The
#   assumed cost comes from the case study's own setup, and the tables above
#   report the net Sharpe there.
# - **Breakeven and assumed cost have to be compared, not reported side by
#   side.** The ratio between them is the headroom, and a ratio below one means
#   the strategy is already under water at its own assumption. The computed
#   classification above says which case studies are where.
# - **A breakeven above the top of the swept grid is not a breakeven.** Where the
#   curve is still positive at the ceiling, the honest statement is that the
#   crossing is somewhere above it, and the tables mark those rows rather than
#   printing the ceiling as though it had been measured.
# - **A basis-point grid does not model every cost structure.** Where the
#   dominant cost is a wide bid-ask spread rather than a proportional fee, a bps
#   sweep understates it, and the answer is an executable backtest against
#   quotes. The S&P 500 Options section above is the worked case.
#
# ## Known Limitations
#
# - Only case studies with a carrier cost sweep appear. NASDAQ-100 is excluded by
#   `DEFERRED_V31_CASE_STUDIES` pending a corrected cost grid, and the rest have
#   no sweep because their registries are being rebuilt. The loaded count is
#   printed at the top.
# - The sweep applies one proportional per-leg cost to every trade. Real costs
#   vary with size, with the instrument, and with the state of the book, and the
#   spread realism section is where that assumption is checked rather than
#   assumed.
# - Net Sharpe at the assumed cost is interpolated between grid points; the
#   bracketing points are in the table so the interpolation can be checked.
# - The turnover multipliers used to place case studies on the cadence chart are
#   stated assumptions about how often each cadence trades, not turnover measured
#   from the backtests.
# - Every Sharpe here is a validation-fold number for a configuration chosen on
#   validation data, so the cost headroom inherits that selection.
#
# **Next**: [`07_regime_risk`](07_regime_risk.ipynb) examines regime
# robustness and risk overlays.
