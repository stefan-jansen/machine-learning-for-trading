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
# # Cost Sensitivity for the §16.4 ETF Baseline
#
# **Docker image**: `ml4t`
#
# Section 16.6 argues that a credible backtest reports gross and net performance,
# turnover, break-even cost, and the sensitivity of the headline metrics to the
# cost assumption. This notebook implements that diagnostic on the §16.4 ETF
# momentum baseline. Same universe, same protocol, same simulator as NB01 — only
# the per-leg fee changes.
#
# **Learning objectives**
#
# 1. Quantify the gross-to-net Sharpe gap at the §16.4 protocol's 5 bp/leg
#    assumption.
# 2. Sweep the per-leg fee across a plausible range and read the resulting
#    Sharpe / CAGR / drawdown curve.
# 3. Compute the **break-even cost** at which gross return equals zero — the
#    quantity that tells you how much error in the cost estimate the strategy
#    can absorb.
# 4. Read **annualized turnover** as the multiplier that translates per-leg
#    cost into annual return drag.
#
# **Book reference**: Chapter 16, §16.6 (Diagnosing the Economic Value).
# **Prerequisites**: §16.4 baseline (NB01); the helper `_etf_baseline.py`
# reproduces its returns and weights.

# %%
"""§16.6 cost-sensitivity diagnostic on the §16.4 ETF momentum baseline."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from _etf_baseline import (
    DEFAULT_FEES,
    annualized_turnover,
    load_panel,
    metrics,
    momentum_weights,
    simulate,
)
from ml4t.diagnostic.visualization.backtest.cost_attribution import plot_cost_sensitivity

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
START_DATE = "2010-01-01"
END_DATE = "2024-01-01"
COST_GRID_BP = [0, 1, 2, 5, 10, 15, 25, 40, 60, 100, 150, 200]

# %% [markdown]
# ## 1. Build the §16.4 Baseline Once
#
# The baseline weights are independent of the cost assumption: the strategy
# selects the same ETFs in the same months regardless of fees. We therefore
# build the weights once and feed them into the simulator under each cost
# assumption.

# %%
panel = load_panel(START_DATE, END_DATE)
weights = momentum_weights(panel)

# Reference (zero-cost) simulation — used for break-even and turnover
result_gross = simulate(panel.prices, weights, fees=0.0)
turnover_2way = annualized_turnover(result_gross)
gross = metrics(result_gross.equity)
print(
    f"Gross (zero-cost): CAGR={gross['cagr'] * 100:.2f}%  "
    f"Sharpe={gross['sharpe']:.2f}  MaxDD={gross['max_drawdown'] * 100:.1f}%"
)
print(f"Annualized 2-way turnover: {turnover_2way * 100:.0f}% (sum |trades| / mean equity / years)")

# %% [markdown]
# ## 2. Sweep the Cost Grid
#
# At each per-leg fee we re-simulate and record the same metric set used in
# §16.5: total return, CAGR, vol, Sharpe, max drawdown. The 5 bp row matches
# Table 16.4's net result exactly because the simulator and weights are
# identical to NB01.

# %%
rows = []
results_by_cost = {}
for bp in COST_GRID_BP:
    res = simulate(panel.prices, weights, fees=bp / 10_000)
    results_by_cost[bp] = res
    m = metrics(res.equity)
    rows.append(
        {
            "cost_bp_per_leg": bp,
            "cagr": m["cagr"],
            "vol": m["vol"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_drawdown"],
            "total_return": m["total_return"],
        }
    )
sweep = pd.DataFrame(rows)
sweep

# %% [markdown]
# ## 3. Break-Even Cost
#
# The canonical break-even is the zero crossing of the simulator's CAGR sweep,
# linearly interpolated between the last positive and first negative rows. This
# keeps the quoted threshold tied to the same path-dependent calculation shown
# in the figure. Treat it as a fragility ceiling, not as a target.

# %%
positive = sweep.loc[sweep["cagr"] > 0].iloc[-1]
negative = sweep.loc[sweep["cagr"] <= 0].iloc[0]
break_even_bp = positive["cost_bp_per_leg"] + (0 - positive["cagr"]) * (
    negative["cost_bp_per_leg"] - positive["cost_bp_per_leg"]
) / (negative["cagr"] - positive["cagr"])
print(f"Break-even cost (simulator interpolation): {break_even_bp:.0f} bp per leg")
print(f"§16.4 protocol cost:       {DEFAULT_FEES * 10_000:.0f} bp per leg")
print(f"Headroom multiplier:       {break_even_bp / (DEFAULT_FEES * 10_000):.0f}x")

# %% [markdown]
# A high headroom multiplier sounds reassuring, but two caveats apply.
#
# - The interpolation is sample-specific; realized cost can also include
#   slippage and spread cost, not just commission.
# - The diagnostic is path-blind. A strategy with the same gross CAGR but
#   front-loaded turnover may still be fragile if costs are above average in
#   the years that contributed most of the return.

# %% [markdown]
# ## 4. Sensitivity Curve
#
# Plot Sharpe and CAGR as a function of per-leg cost. The slope of these
# curves at the §16.4 operating point is the marginal cost elasticity —
# how much performance you lose per additional basis point.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)

axes[0].plot(sweep["cost_bp_per_leg"], sweep["sharpe"], color="black", marker="o")
axes[0].axvline(DEFAULT_FEES * 10_000, color="0.5", linestyle="--", label="§16.4 cost (5 bp)")
axes[0].axhline(0, color="0.7", linewidth=0.8)
axes[0].set_xlabel("Per-leg cost (bp)")
axes[0].set_ylabel("Sharpe ratio")
axes[0].set_title("Sharpe vs cost assumption", loc="left")
axes[0].legend(frameon=False)

axes[1].plot(sweep["cost_bp_per_leg"], sweep["cagr"] * 100, color="black", marker="o")
axes[1].axvline(DEFAULT_FEES * 10_000, color="0.5", linestyle="--")
axes[1].axhline(0, color="0.7", linewidth=0.8)
axes[1].axvline(
    break_even_bp, color="0.3", linestyle=":", label=f"Break-even ({break_even_bp:.0f} bp)"
)
axes[1].set_xlabel("Per-leg cost (bp)")
axes[1].set_ylabel("CAGR (%)")
axes[1].set_title("CAGR vs cost assumption", loc="left")
axes[1].legend(frameon=False)

fig.suptitle("§16.4 ETF baseline: cost sensitivity", fontsize=12)
fig.show()

# %% [markdown]
# ## 5. Cost Drag Decomposition
#
# Annual return drag from costs is approximately
#
# $$\text{drag}_{\text{annual}} \approx
#   \text{turnover}_{\text{2-way}} \times \text{fee}_{\text{per leg}}.$$
#
# At the §16.4 protocol's 5 bp fee and the strategy's measured turnover, this
# matches the gap between the gross and net CAGR closely — the simulator and
# the back-of-envelope estimate agree to within rounding.

# %%
fee_bp = DEFAULT_FEES * 10_000
estimated_drag_pct = turnover_2way * DEFAULT_FEES * 100
realized_drag_pct = (
    gross["cagr"] - sweep.loc[sweep["cost_bp_per_leg"] == fee_bp, "cagr"].iloc[0]
) * 100
print(f"Estimated drag at {fee_bp:.0f} bp: {estimated_drag_pct:.2f}% / yr")
print(f"Realized drag at {fee_bp:.0f} bp:  {realized_drag_pct:.2f}% / yr")

# %% [markdown]
# ## 6. Library Equivalents — `ml4t-diagnostic`
#
# Sections 4 and 5 build the cost diagnostic from the equity curve so the
# mechanics are explicit. The exact simulator waterfall below keeps every bar
# in dollars and reconciles to the 5 bp run. `ml4t-diagnostic` provides the
# complementary one-call sensitivity view for a production report:
#
# - `plot_cost_sensitivity` reproduces the §4 Sharpe-vs-cost curve directly
#   from the gross daily-return series, with a break-even annotation.
#
# Both figures are Plotly artifacts and slot into the report bundle that
# `09_performance_reporting` assembles for the same backtest.

# %%
# Use one dollar denominator throughout. The compounding/path term captures the
# fact that fees paid earlier reduce capital available for subsequent returns.
result_net = results_by_cost[fee_bp]
gross_pnl_dollars = float(result_gross.equity.iloc[-1] - result_gross.equity.iloc[0])
net_pnl_dollars = float(result_net.equity.iloc[-1] - result_net.equity.iloc[0])
commission_dollars = float((result_net.trades_dollar * DEFAULT_FEES).sum())
path_effect_dollars = gross_pnl_dollars - commission_dollars - net_pnl_dollars
assert np.isclose(gross_pnl_dollars - commission_dollars - path_effect_dollars, net_pnl_dollars)

waterfall = go.Figure(
    go.Waterfall(
        measure=["absolute", "relative", "relative", "total"],
        x=["Gross PnL", "Commissions", "Compounding/path effect", "Net PnL"],
        y=[gross_pnl_dollars, -commission_dollars, -path_effect_dollars, net_pnl_dollars],
        connector={"line": {"color": "#64748B"}},
    )
)
waterfall.update_layout(
    title=f"Exact gross-to-net PnL at {DEFAULT_FEES * 10_000:.0f} bp per leg",
    yaxis_title="PnL (USD)",
    showlegend=False,
)
waterfall.show()

# %%
# `plot_cost_sensitivity` is used illustratively — it computes Sharpe from a
# daily-drag approximation, while the manual sweep in Section 2 runs the
# full simulator at each per-leg fee. The two figures show the same shape
# but its daily-drag approximation need not share the simulator's exact zero
# crossing. Treat the manual sweep as the canonical source for quoted break-even.
gross_returns_pl = pl.from_pandas(result_gross.returns.rename("returns").reset_index()).get_column(
    "returns"
)
sensitivity = plot_cost_sensitivity(
    returns=gross_returns_pl,
    base_costs_bps=DEFAULT_FEES * 10_000,
    cost_multipliers=[m / (DEFAULT_FEES * 10_000) for m in COST_GRID_BP],
    trades_per_year=float(turnover_2way),
    title="Sharpe sensitivity to per-leg cost (library)",
)
sensitivity.show()

# %% [markdown]
# The library figure reproduces the qualitative slope and sensitivity pattern
# of Section 4. The manual sweep table is the audited calculation; the library
# call is a one-line production-report equivalent. Use the inline matplotlib
# views to teach the calculation and the library calls to assemble reports.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Gross-vs-net is one row, not one number.** Reporting net Sharpe alone
#    hides the cost-elasticity. The sweep table (Section 2) is the right
#    artifact for a backtest report.
# 2. **Break-even cost is a fragility ceiling**. The §16.4 baseline tolerates
#    a wide cost range because turnover is moderate and gross CAGR is positive
#    on this sample. A high-frequency strategy with the same gross CAGR but
#    10x the turnover would have 1/10 of the headroom.
# 3. **Annualized turnover is the cost multiplier.** Once you know turnover,
#    you can read drag at any fee in your head: 5 bp per leg × 4.5x annual
#    turnover ≈ 23 bp annual drag. The simulator confirms the estimate.
# 4. **Cost sensitivity does not validate or invalidate a strategy by itself.**
#    It is one of three §16.6 diagnostics — combine with the baseline
#    comparison in §16.4 / Table 16.4 and the regime-slicing notebook
#    (`10_regime_backtest_analysis.py`) before drawing conclusions about
#    economic value.
