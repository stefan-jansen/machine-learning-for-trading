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
# # Anatomy of Engine Divergence
#
# **Docker image**: `ml4t`
#
# Backtesting frameworks make implicit execution decisions — fill ordering,
# share rounding, cash management — that produce different P&L from identical
# signals. This notebook makes those differences *visible* using ml4t-backtest's
# configurable `BacktestConfig`, then compares against VectorBT.
#
# **Learning Objectives**:
# - See how fill ordering (exit-first vs FIFO) changes P&L on identical signals
# - Understand why integer vs fractional shares creates trade count divergence
# - Quantify the impact of commission headroom on capital efficiency
# - Learn that execution mechanics are design choices, not implementation bugs
#
# **Book Reference**: Chapter 16, Section 16.3 (Backtesting Engine Comparison)
#
# **Prerequisites**: Complete [`06_linear`](../case_studies/etfs/06_linear.ipynb) (prediction artifacts).

# %%
"""Anatomy of Engine Divergence — how execution mechanics change backtest results."""

import copy
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.config import (
    CommissionType,
    ExecutionMode,
    FillOrdering,
    RebalanceMode,
    ShareType,
    SlippageType,
)
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from validation.adapters.ml4t_adapter import (
    PrecomputedWeightStrategy,
    _convert_result,
    _filter_late_assets,
)
from validation.adapters.vectorbt_adapter import run_vectorbt
from validation.weights import load_case_study_data

from utils.style import COLORS

# %% tags=["parameters"]
CASE_STUDY = "etfs"
LABEL = None  # Auto-detect from available predictions
INITIAL_CASH = 1_000_000

# %% [markdown]
# ## 1. Load Shared Data
#
# We reuse pre-computed target weights from the ETF linear model pipeline.
# The same weights feed every engine configuration, isolating
# execution-mechanics from signal-generation differences.

# %%
prices, signals, weights, meta = load_case_study_data(CASE_STUDY, LABEL)
cost_rate = meta["cost_bps"] / 10_000

if weights.empty:
    raise RuntimeError(
        "No validation weights were produced for this case study. "
        "Run the prerequisite prediction pipeline before executing this notebook."
    )

print(f"Case Study: {meta['case_study']}")
print(f"Label: {meta['label']}, Model: {meta['model']}")
print(f"Universe: {meta['n_assets']} assets, {meta['n_signal_dates']} signal dates")
print(f"Cost: {meta['cost_bps']} bps ({cost_rate:.4f})")
print(f"Period: {weights.index[0].date()} to {weights.index[-1].date()}")

# %% [markdown]
# ### Prepare Engine Infrastructure
#
# Filter late-starting assets once and define a helper that runs the engine
# with any `BacktestConfig`, keeping boilerplate out of the comparison sections.

# %%
# Filter late-starting assets once (e.g., XLC started 2018)
prices_f, weights_f = _filter_late_assets(prices, weights)


# %% [markdown]
# `run_with_config` wires a `BacktestConfig` through the precomputed-weights
# strategy and returns a standardized result; downstream cells call it with
# different configs to isolate one execution mechanic at a time.


# %%
def run_with_config(config, weight_scale=1.0):
    """Run ml4t-backtest with explicit BacktestConfig and return BacktestResult."""
    allow_frac = config.share_type == ShareType.FRACTIONAL
    rebalance_cfg = RebalanceConfig(
        allow_fractional=allow_frac,
        min_trade_value=1.0 if allow_frac else 0.0,
        min_weight_change=0.0001 if allow_frac else 0.0,
    )

    weight_dict: dict[datetime, dict[str, float]] = {}
    for ts in weights_f.index:
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        row = weights_f.loc[ts]
        w = {c: float(row[c]) * weight_scale for c in weights_f.columns if row[c] > 0}
        if w:
            weight_dict[ts_dt] = w

    executor = TargetWeightExecutor(config=rebalance_cfg)
    strategy = PrecomputedWeightStrategy(weight_dict, executor)
    feed = DataFeed(prices_df=prices_f)
    engine = Engine.from_config(feed, strategy, config)
    result = engine.run()
    return _convert_result(result, config.preset_name or "custom", config.initial_cash)


# %% [markdown]
# `metrics_row` extracts the comparable fields from a `BacktestResult` for the
# DataFrame summaries that appear after each comparison.


# %%
def metrics_row(r):
    """Extract key metrics from a BacktestResult."""
    return {
        "final_value": round(r.final_value),
        "return_pct": round(r.total_return * 100, 2),
        "sharpe": round(r.metrics.get("sharpe", 0), 3),
        "max_dd_pct": round(abs(r.metrics.get("max_drawdown", 0)) * 100, 2),
        "trades": r.n_trades,
        "commission": round(r.metrics.get("total_commission", 0)),
    }


# %% [markdown]
# ## 2. Baseline — Default Configuration
#
# The ml4t-backtest "default" preset: next-bar open fills, fractional shares,
# exit-first ordering, percentage commission. We disable slippage so
# differences between variants come purely from execution mechanics.


# %%
def make_baseline():
    """Create baseline config with our cost assumptions."""
    config = BacktestConfig.from_preset("default")
    config.initial_cash = INITIAL_CASH
    config.commission_rate = cost_rate
    config.slippage_type = SlippageType.NONE
    config.slippage_rate = 0.0
    return config


baseline_config = make_baseline()
r_baseline = run_with_config(baseline_config)

print("Baseline (default preset):")
print(f"  Final Value:  ${r_baseline.final_value:,.0f}")
print(f"  Total Return: {r_baseline.total_return:.2%}")
print(f"  Sharpe:       {r_baseline.metrics.get('sharpe', 0):.3f}")
print(f"  Max Drawdown: {r_baseline.metrics.get('max_drawdown', 0):.2%}")
print(f"  Trades:       {r_baseline.n_trades:,}")

# %% [markdown]
# ## 3. Fill Ordering: EXIT_FIRST vs FIFO
#
# When a rebalance requires both sells and buys, the engine must decide
# which orders to process first:
#
# - **EXIT_FIRST**: Sells execute before buys, freeing cash for purchases.
#   The buy-side has more capital available, potentially filling larger orders.
# - **FIFO**: Orders execute in submission order. If early buys consume cash,
#   later buys may be rejected for insufficient funds.
#
# Same signals, same shares, same timing — only the processing sequence differs.

# %%
fifo_config = make_baseline()
fifo_config.fill_ordering = FillOrdering.FIFO

r_fifo = run_with_config(fifo_config)

# Compare
table = pl.DataFrame(
    [
        {"config": "EXIT_FIRST (baseline)", **metrics_row(r_baseline)},
        {"config": "FIFO", **metrics_row(r_fifo)},
    ]
)
table

# %%
eq_base = r_baseline.equity_curve
eq_fifo = r_fifo.equity_curve
common = eq_base.index.intersection(eq_fifo.index)

diff_pct = (eq_base.loc[common] - eq_fifo.loc[common]) / eq_base.loc[common] * 100

fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1], sharex=True)

axes[0].plot(eq_base.loc[common].index, eq_base.loc[common].values, label="EXIT_FIRST", linewidth=1)
axes[0].plot(
    eq_fifo.loc[common].index, eq_fifo.loc[common].values, label="FIFO", linewidth=1, linestyle="--"
)
axes[0].set_ylabel("Portfolio Value ($)")
axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
axes[0].set_title("Fill Ordering: EXIT_FIRST vs FIFO")

axes[1].fill_between(common, diff_pct.values, alpha=0.5, color=COLORS["blue"])
axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].set_ylabel("Difference (%)")
axes[1].set_xlabel("Date")

fig.tight_layout()
fig.show()

val_diff = r_baseline.final_value - r_fifo.final_value
print(f"\nFinal value difference: ${val_diff:,.0f} ({val_diff / r_baseline.final_value:.3%})")
print(
    f"Trade count: {r_baseline.n_trades:,} vs {r_fifo.n_trades:,} "
    f"(delta: {r_baseline.n_trades - r_fifo.n_trades:,})"
)

# %% [markdown]
# **Finding**: EXIT_FIRST and FIFO produce identical results here — zero
# difference in final value or trade count. With fractional shares and
# next-bar execution, all orders fill atomically regardless of processing
# sequence: the engine computes target positions from a pre-trade snapshot,
# and every order executes at the next bar's open with sufficient cash
# (because sells and buys net out against the full portfolio).
#
# Fill ordering matters when **integer shares** constrain capital allocation:
# an early buy may consume cash needed for a later buy, causing rejection.
# EXIT_FIRST avoids this by freeing sell proceeds first. We will see this
# effect surface in the Combined Preset comparison below, where the
# backtrader preset combines integer shares with FIFO ordering.

# %% [markdown]
# ## 4. Share Type: Fractional vs Integer
#
# Fractional shares allow exact target-weight allocation. Integer shares
# introduce two effects:
#
# 1. **Rounding drag**: Each position is rounded down, leaving a small cash
#    residual that compounds over time.
# 2. **Skipped trades**: When the target delta rounds to zero shares,
#    no order is submitted — the portfolio drifts from its target.
#
# For a \$1M portfolio holding 20 ETFs, each position averages \$50K.
# A \$200 ETF requires ~250 shares — rounding error is ~0.4% per position.

# %%
integer_config = make_baseline()
integer_config.share_type = ShareType.INTEGER

r_integer = run_with_config(integer_config)

table = pl.DataFrame(
    [
        {"config": "Fractional (baseline)", **metrics_row(r_baseline)},
        {"config": "Integer", **metrics_row(r_integer)},
    ]
)
table

# %%
eq_int = r_integer.equity_curve
common_int = eq_base.index.intersection(eq_int.index)

diff_int = (eq_base.loc[common_int] - eq_int.loc[common_int]) / eq_base.loc[common_int] * 100

fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1], sharex=True)

axes[0].plot(
    eq_base.loc[common_int].index, eq_base.loc[common_int].values, label="Fractional", linewidth=1
)
axes[0].plot(
    eq_int.loc[common_int].index,
    eq_int.loc[common_int].values,
    label="Integer",
    linewidth=1,
    linestyle="--",
)
axes[0].set_ylabel("Portfolio Value ($)")
axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
axes[0].set_title("Share Type: Fractional vs Integer")

axes[1].fill_between(common_int, diff_int.values, alpha=0.5, color=COLORS["amber"])
axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].set_ylabel("Difference (%)")
axes[1].set_xlabel("Date")

fig.tight_layout()
fig.show()

val_diff_int = r_baseline.final_value - r_integer.final_value
print(
    f"\nFinal value difference: ${val_diff_int:,.0f} ({val_diff_int / r_baseline.final_value:.3%})"
)
print(
    f"Trade count: {r_baseline.n_trades:,} vs {r_integer.n_trades:,} "
    f"(delta: {r_baseline.n_trades - r_integer.n_trades:,})"
)

# %% [markdown]
# **Reading the table.** The integer-share row shows two effects relative to
# the fractional baseline: a final-value gap (positive or negative depending
# on the regime — see the printed numbers above) and a trade-count delta.
#
# The under-investment direction is the easy one: integer rounding always
# rounds positions *down*, leaving a small cash residual that compounds as
# drag. The trade-count effect is less intuitive: with fractional shares, a
# tiny weight change (e.g., 0.01%) translates to a precise dollar adjustment
# that the executor can skip below `min_weight_change`. With integer shares,
# the same small delta rounds to 1 share — a larger discrete adjustment that
# clears the threshold and generates a trade. Rounding can therefore amplify
# small signals into executable orders. Whether the gap shows up as zero,
# small, or material depends on portfolio size relative to share prices; on
# this run the displayed table is the source of truth.

# %% [markdown]
# ## 5. Commission Headroom
#
# Some frameworks (notably Backtrader) reject orders when the portfolio
# lacks sufficient cash to cover both the order and its commission.
# The workaround: scale target weights by 0.998, holding back 0.2% of
# capital as a buffer. This prevents rejection but creates systematic
# under-investment.

# %%
HEADROOM = 0.998

r_headroom = run_with_config(make_baseline(), weight_scale=HEADROOM)

table = pl.DataFrame(
    [
        {"config": "Full allocation (baseline)", **metrics_row(r_baseline)},
        {"config": f"With headroom ({HEADROOM})", **metrics_row(r_headroom)},
    ]
)
table

# %%
eq_hd = r_headroom.equity_curve
common_hd = eq_base.index.intersection(eq_hd.index)

diff_hd_bps = (eq_base.loc[common_hd] - eq_hd.loc[common_hd]) / eq_base.loc[common_hd] * 10_000

fig, ax = plt.subplots(figsize=(12, 3))
ax.fill_between(common_hd, diff_hd_bps.values, alpha=0.5, color=COLORS["copper"])
ax.axhline(0, color="k", linewidth=0.5)
ax.set_ylabel("Equity-curve gap (bps)")
ax.set_xlabel("Date")
ax.set_title(f"Commission Headroom: 100% vs {HEADROOM:.1%} Allocation")
fig.tight_layout()
fig.show()

val_diff_hd = r_baseline.final_value - r_headroom.final_value
print(f"\nFinal value difference: ${val_diff_hd:,.0f} ({val_diff_hd / r_baseline.final_value:.3%})")

# %% [markdown]
# **Reading the table.** The headroom buffer creates systematic
# under-investment: at each rebalance the portfolio leaves 0.2% in cash, and
# the uninvested cash misses the subsequent period's return. The compounding
# effect can be modest in absolute dollars but is *non-linear* in horizon —
# print the final-value gap from the comparison above to see the realized
# drag on this run. The headroom is necessary for frameworks like Backtrader
# that reject orders when cash is insufficient to cover the trade plus
# commission, but it has a measurable cost.

# %% [markdown]
# ## 6. Combined Preset Comparison
#
# Each backtesting framework embeds different combinations of these choices
# as defaults. The ml4t-backtest presets capture three philosophies:
#
# | Preset | Fill Timing | Shares | Fill Order | Headroom |
# |--------|-------------|--------|------------|----------|
# | `default` | Next-bar open | Fractional | EXIT_FIRST | None |
# | `backtrader` | Next-bar open | Integer | FIFO | 0.998 |
# | `vectorbt` | Same-bar close | Fractional | EXIT_FIRST | None |
#
# We also run actual VectorBT for ground truth.

# %%
# Backtrader preset
bt_config = BacktestConfig.from_preset("backtrader")
bt_config.initial_cash = INITIAL_CASH
bt_config.commission_rate = cost_rate
bt_config.slippage_type = SlippageType.NONE
bt_config.slippage_rate = 0.0

r_bt = run_with_config(bt_config, weight_scale=HEADROOM)

# VectorBT preset (ml4t engine matching VBT behavior)
vbt_config = BacktestConfig.from_preset("vectorbt")
vbt_config.initial_cash = INITIAL_CASH
vbt_config.commission_type = CommissionType.PERCENTAGE
vbt_config.commission_rate = cost_rate
vbt_config.slippage_type = SlippageType.NONE
vbt_config.slippage_rate = 0.0

r_vbt_preset = run_with_config(vbt_config)

# %%
# Actual VectorBT engine
r_vbt = run_vectorbt(prices, weights, initial_cash=INITIAL_CASH, commission_rate=cost_rate)
all_results = [r_baseline, r_bt, r_vbt_preset, r_vbt]
all_labels = ["default", "backtrader", "vectorbt (ml4t)", "VectorBT (actual)"]
styles = [
    {"linewidth": 1.5},
    {"linewidth": 1.5, "linestyle": "--"},
    {"linewidth": 1, "linestyle": "-."},
    {"linewidth": 1, "linestyle": ":"},
]

# %%
table = pl.DataFrame(
    [{"config": label, **metrics_row(r)} for r, label in zip(all_results, all_labels, strict=False)]
)
table

# %%
fig, ax = plt.subplots(figsize=(12, 5))
for r, label, style in zip(all_results, all_labels, styles, strict=False):
    ax.plot(r.equity_curve.index, r.equity_curve.values, label=label, **style)

ax.set_ylabel("Portfolio Value ($)")
ax.set_xlabel("Date")
ax.set_title("Preset Comparison: Same Signal, Different Execution Mechanics")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
fig.tight_layout()
fig.show()

# %% [markdown]
# The four equity curves share the same shape (same signal!) but diverge
# in level. The backtrader preset and the actual VectorBT engine both track
# the default closely (within ~0.3%). The `vectorbt (ml4t)` preset diverges
# most — about 4.8% below the default over the 8-year run — driven by its
# one change from the default, same-bar-close fill timing, the lever §5
# showed to be the most consequential.

# %% [markdown]
# ## 7. Divergence Decomposition
#
# How much does each execution decision contribute to the total divergence
# between the "default" and "backtrader" presets? We decompose the gap
# by measuring each single-variable effect against the baseline.

# %%
# Compute individual effects as percentage of baseline final value
baseline_val = r_baseline.final_value
effects = {
    "Fill ordering\n(EXIT_FIRST → FIFO)": (r_baseline.final_value - r_fifo.final_value)
    / baseline_val
    * 100,
    "Share rounding\n(fractional → integer)": (r_baseline.final_value - r_integer.final_value)
    / baseline_val
    * 100,
    "Commission headroom\n(100% → 99.8%)": (r_baseline.final_value - r_headroom.final_value)
    / baseline_val
    * 100,
}

# Combined: default vs backtrader preset
combined = (r_baseline.final_value - r_bt.final_value) / baseline_val * 100

labels = list(effects.keys()) + ["Combined\n(default → backtrader)"]
values = list(effects.values()) + [combined]
colors = [COLORS["blue"]] * 3 + [COLORS["slate"]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)

# Value labels on bars
for bar, val in zip(bars, values, strict=False):
    x = bar.get_width()
    ax.text(
        x + 0.05 * max(abs(v) for v in values),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}%",
        va="center",
        fontweight="bold",
    )

ax.set_xlabel("Impact on Final Value (%)")
ax.set_title("Divergence Decomposition: Which Execution Decision Matters Most?")
ax.axvline(0, color="k", linewidth=0.5)
fig.tight_layout()
fig.show()

# %%
sum_individual = sum(effects.values())
interaction = combined - sum_individual

print(f"Individual effects sum: {sum_individual:+.2f}%")
print(f"Combined effect:        {combined:+.2f}%")
print(f"Interaction term:       {interaction:+.2f}%")
print("\nThe interaction term shows these effects don't compose linearly.")
print("Integer shares change how fill ordering binds (fewer fractional buys")
print("to reject), so the combined effect differs from the sum of parts.")

# %% [markdown]
# ## 8. Rebalance Mode: Snapshot vs Incremental vs Hybrid
#
# When rebalancing a portfolio of 20+ assets, the engine must decide: **does
# each order see the updated portfolio value from prior fills, or do all
# orders use a frozen pre-trade snapshot?**
#
# This is not an implementation detail — it reflects a real-world choice about
# how brokers process multi-asset rebalances:
#
# | Mode | Target Value | Fills | Real-World Analog |
# |------|-------------|-------|-------------------|
# | **Snapshot** | Frozen (computed once) | Batched (all at once) | Backtrader, Zipline |
# | **Incremental** | Recomputed after each fill | Sequential | Conservative broker |
# | **Hybrid** | Frozen for targets | Sequential (cash checked live) | VectorBT |
#
# ml4t-backtest's `BacktestConfig.rebalance_mode` makes this choice explicit.

# %% [markdown]
# `run_rebalance_mode` runs the same baseline with a single rebalance-mode
# variable swapped, holding share type, fill ordering, and execution mode fixed.


# %%
def run_rebalance_mode(mode: RebalanceMode):
    """Run backtest with a specific rebalance mode, keeping everything else identical.

    Uses same-bar execution so INCREMENTAL/HYBRID fills take effect immediately
    within each rebalance — with next-bar execution, all orders are deferred
    to the next bar regardless of mode.
    """
    config = make_baseline()
    config.share_type = ShareType.INTEGER
    config.fill_ordering = FillOrdering.FIFO
    config.execution_mode = ExecutionMode.SAME_BAR
    config.rebalance_mode = mode

    rebal_cfg = RebalanceConfig(
        allow_fractional=False,
        min_trade_value=0.0,
        min_weight_change=0.0,
        rebalance_mode=mode,
    )

    weight_dict: dict = {}
    for ts in weights_f.index:
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        row = weights_f.loc[ts]
        w = {c: float(row[c]) * HEADROOM for c in sorted(weights_f.columns) if row[c] > 0}
        if w:
            weight_dict[ts_dt] = w

    executor = TargetWeightExecutor(config=rebal_cfg)
    strategy = PrecomputedWeightStrategy(weight_dict, executor)
    feed = DataFeed(prices_df=prices_f)
    engine = Engine.from_config(feed, strategy, copy.deepcopy(config))
    result = engine.run()
    return _convert_result(result, mode.value, config.initial_cash)


r_snapshot = run_rebalance_mode(RebalanceMode.SNAPSHOT)
r_incremental = run_rebalance_mode(RebalanceMode.INCREMENTAL)
r_hybrid = run_rebalance_mode(RebalanceMode.HYBRID)

# %%
table = pl.DataFrame(
    [
        {"mode": "Snapshot (Backtrader-style)", **metrics_row(r_snapshot)},
        {"mode": "Incremental (live value)", **metrics_row(r_incremental)},
        {"mode": "Hybrid (VectorBT-style)", **metrics_row(r_hybrid)},
    ]
)
table

# %%
eq_snap = r_snapshot.equity_curve
eq_incr = r_incremental.equity_curve
eq_hybr = r_hybrid.equity_curve
common_rb = eq_snap.index.intersection(eq_incr.index).intersection(eq_hybr.index)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1], sharex=True)

axes[0].plot(common_rb, eq_snap.loc[common_rb].values, label="Snapshot", linewidth=1)
axes[0].plot(
    common_rb, eq_incr.loc[common_rb].values, label="Incremental", linewidth=1, linestyle="--"
)
axes[0].plot(common_rb, eq_hybr.loc[common_rb].values, label="Hybrid", linewidth=1, linestyle=":")
axes[0].set_ylabel("Portfolio Value ($)")
axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
axes[0].set_title("Rebalance Mode: When Does the Broker Snapshot Portfolio Value?")

# Show incremental vs snapshot divergence
diff_is = (eq_incr.loc[common_rb] - eq_snap.loc[common_rb]) / eq_snap.loc[common_rb] * 100
axes[1].fill_between(
    common_rb, diff_is.values, alpha=0.5, color=COLORS["copper"], label="Incremental − Snapshot"
)
axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].set_ylabel("Difference (%)")
axes[1].set_xlabel("Date")
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.show()

# %%
pl.DataFrame(
    [
        {"mode": name, "final_value": int(r.final_value), "trades": int(r.n_trades)}
        for name, r in [
            ("Snapshot", r_snapshot),
            ("Incremental", r_incremental),
            ("Hybrid", r_hybrid),
        ]
    ]
)

# %% [markdown]
# **Reading the table.** All three rebalance modes produce strongly negative
# results under this configuration. The dominant effect is **same-bar
# execution**, not the rebalance mode itself: the ETF ridge model predicts
# *next-day* returns, but same-bar fills execute at today's close, so the
# strategy systematically trades *after* the predicted move has already
# occurred. The rebalance-mode differences between snapshot, hybrid, and
# incremental are an order of magnitude smaller; the printed final-value
# row above quantifies each mode on this run.
#
# **Takeaway**: Execution timing (same-bar vs next-bar) swamps rebalance
# mode as a source of P&L divergence. Choose rebalance mode to match your
# broker's settlement mechanics, but verify that fill timing aligns with
# your signal's prediction horizon — a one-bar misalignment can flip
# a profitable strategy into a consistent loser.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Execution mechanics are design choices, not bugs.** Fill ordering, share
#    rounding, cash management, and rebalance mode are deliberate engineering
#    decisions that each framework makes differently.
#
# 2. **ml4t-backtest makes these explicit** via `BacktestConfig` — readers
#    choose their execution assumptions rather than inheriting hidden defaults.
#
# 3. **Fill timing dominates all other execution choices.** Switching from
#    next-bar to same-bar execution turned an 80% gain into a 90% loss — a
#    signal/fill misalignment that dwarfs every other knob. Always verify
#    that fill timing matches your signal's prediction horizon.
#
# 4. **Share rounding alone is negligible here.** The Integer and Fractional
#    configs are identical on this run ($0 difference, 0.000%, same trade
#    count). Rounding becomes consequential only in combination with
#    next-bar-open timing and FIFO ordering — the default→backtrader gap
#    (~0.3%) reflects that interaction, not rounding in isolation.
#
# 5. **Commission headroom compounds.** A 0.2% buffer costs ~0.17% of final
#    value over 8 years through compounding cash drag at each rebalance.
#
# 6. **Fill ordering is neutral with fractional shares.** EXIT_FIRST and FIFO
#    produce identical results when fractional shares eliminate cash constraints.
#    The ordering becomes consequential only with integer shares.
#
# 7. **Always document your execution assumptions.** When publishing backtest
#    results, specify: fill timing, share type, fill ordering, rebalance mode,
#    and cost model. Without this, results are not reproducible.
#
# **Next**: the external-engine parity notebooks make the same execution knobs
# concrete on the canonical benchmark:
# [`15_lean_engine_parity`](15_lean_engine_parity.ipynb),
# [`17_backtrader_zipline_engine_parity`](17_backtrader_zipline_engine_parity.ipynb),
# and [`18_vectorbt_engine_parity`](18_vectorbt_engine_parity.ipynb).
#
# **Book**: Section 16.3 covers configurable parity and framework validation.
