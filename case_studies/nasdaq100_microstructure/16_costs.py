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
# # NASDAQ-100 Microstructure: Costs
#
# **Chapter 18 — Transaction Costs and Execution**
#
# This is the primary cost-analysis notebook for the NASDAQ-100 case study.
# Transaction costs at 15-minute cadence eliminate the signal on the full
# universe: ~7 bps per bar × 26 rebalances per day produces massive annual drag.
# Two levers recover a tradeable strategy, and this notebook quantifies both:
#
# 1. **Screen the universe for cost feasibility.** Restricting to the
#    cheapest-to-trade names (the cost-feasible universe from the feasibility
#    analysis) raises the same slot design from deeply negative to positive and
#    cuts turnover several-fold — the screen the case study trades on.
# 2. **Slow the cadence.** Cost dominance is **cadence-dependent**: dropping
#    rebalance frequency to hourly or 4-hourly amortizes the per-trade cost so
#    the signal becomes viable at institutional execution levels.
#
# The notebook has three parts:
# - **Sections 1–3**: Standard bps cost grid on full-universe allocation combos,
#   tracing the Sharpe-vs-cost decay curve.
# - **Section 4**: Full universe vs the cost-feasible screen — the first lever,
#   read off existing registry rows for the featured slot design.
# - **Section 5**: Cadence × per-share cost sweep — the second lever and the
#   publication finding. Uses a per-share cost model ($/share, not bps), more
#   realistic for equities, swept across rebalance frequencies.
#
# **Learning Objectives:**
# 1. Run a cost grid sweep on full-universe combos to find breakeven
# 2. Quantify how the cost-feasibility screen recovers Sharpe and cuts turnover
# 3. Sweep cadence × per-share cost to find the viable implementation regime
#
# **Book Reference:** Chapter 18, Sections 18.2–18.5
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""NASDAQ-100 Microstructure: Costs."""

import json
import sqlite3
import time
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    warmup_periods_for,
)
from case_studies.utils.backtest_presets import (
    build_backtest_spec,
    clone_backtest_spec,
    ensure_backtest_spec,
    set_backtest_costs_bps,
    strategy_view,
)
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.notebook_contracts import excluded_families
from case_studies.utils.registry import read_predictions, resolve_best_backtest_runs
from case_studies.utils.sweep_config import (
    get_cadence_sweep,
    get_cost_grid_bps,
    get_cost_grid_half_spread_usd,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
LABEL = ""
MAX_SYMBOLS = 0
TOP_N_COMBOS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "cost_sensitivity")
if not LABEL:
    LABEL = bt_config.primary_label

print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}")

COST_GRID_BPS = get_cost_grid_bps(CASE_STUDY_ID)

if excluded_families(CASE_STUDY_ID):
    print(
        "Active-model filter: excluding "
        f"{', '.join(sorted(excluded_families(CASE_STUDY_ID)))} pending corrected reruns"
    )

# %% [markdown]
# ## 1. Load Top Combos from Allocation Stage
#
# We load the least-negative full-universe allocation-stage backtests. These are
# every-bar combos and are already loss-making (Ch17); the cost grid below
# traces how the Sharpe-vs-cost curve behaves around them before the two
# recovery levers — the screen and the cadence — are applied.

# %%
top_combos = resolve_best_backtest_runs(
    CASE_STUDY_ID, LABEL, split="validation", stage="allocation", top_n=TOP_N_COMBOS
)

if top_combos.is_empty():
    print("No allocation-stage results found. Run the portfolio management notebook first.")
else:
    for row in top_combos.iter_rows(named=True):
        spec = ensure_backtest_spec(
            CASE_STUDY_ID,
            bt_config,
            json.loads(row["spec_json"]),
            prices=load_backtest_prices_for(
                CASE_STUDY_ID,
                LABEL,
                split="validation",
                warmup_periods=warmup_periods_for(CASE_STUDY_ID),
                max_symbols=MAX_SYMBOLS,
            ),
            prediction_hash=row["prediction_hash"],
            initial_cash=bt_config.initial_cash,
        )
        alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
        print(f"  Sharpe={row['sharpe']:.3f}  alloc={alloc}  bt_hash={row['backtest_hash'][:8]}")

# %%
prices = load_backtest_prices_for(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    warmup_periods=warmup_periods_for(CASE_STUDY_ID),
    max_symbols=MAX_SYMBOLS,
)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique()} assets")

# %% [markdown]
# ## 2. Cost Grid Sweep
#
# For each top combo, re-run the backtest at different total cost levels
# (commission + slippage combined). The grid spans from near-zero to levels
# that exceed the signal entirely, tracing the full decay curve.
#
# At 15-minute cadence with ~26 bars per trading day, even 1 bps per leg
# compounds to significant annual drag. The breakeven cost level for this
# case study is expected to be very low — in the range of 1–3 bps total —
# making it viable only for market-makers or prop desks with institutional
# execution quality, or for strategies that extend the hold period to 4–8 bars
# to amortize the per-trade cost.

# %%
n_total = len(top_combos) * len(COST_GRID_BPS) if not top_combos.is_empty() else 0
n_done = 0
t0 = time.time()

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
    alloc_method = strategy_view(base_spec).get("allocation", {}).get("method", "equal_weight")
    predictions = read_predictions(CASE_STUDY_ID, pred_hash)

    for cost_bps in COST_GRID_BPS:
        n_done += 1
        spec = set_backtest_costs_bps(
            clone_backtest_spec(base_spec),
            commission_bps=cost_bps / 2,
            slippage_bps=cost_bps / 2,
        )
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
            if cost_bps % 10 == 0:
                print(
                    f"  [{n_done}/{n_total}] {alloc_method} @ {cost_bps}bps: "
                    f"Sharpe={result.metrics.get('sharpe', 0):.3f}"
                )
        except Exception as e:
            print(f"  [{n_done}/{n_total}] {alloc_method} @ {cost_bps}bps: FAILED — {e}")

# %%
elapsed = time.time() - t0
print(f"Cost sweep complete: {n_done} backtests in {elapsed:.0f}s")

# %% [markdown]
# ## 3. Cost Sensitivity Analysis
#
# This section is **read-only** — queries the registry for cost-sensitivity
# results and computes breakeven levels.
#
# The Sharpe-versus-cost curve for intraday strategies typically falls steeply
# from the near-zero-cost benchmark. For NASDAQ-100 15-minute, the expected
# pattern is: positive Sharpe at 0–2 bps, break-even around 3–5 bps, negative
# at any cost level resembling realistic retail execution. The flat portion of
# the curve (if it exists) defines the practical cost budget.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %%
cost_df = explorer.cost_sensitivity()

if not cost_df.is_empty():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for alloc in cost_df["allocator"].unique().sort().to_list():
        subset = cost_df.filter(pl.col("allocator") == alloc).sort("cost_bps")
        ax.plot(subset["cost_bps"].to_list(), subset["sharpe"].to_list(), marker="o", label=alloc)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Total Cost (bps per leg)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Sharpe Decay Under Transaction Costs")
    ax.legend()
    fig.tight_layout()
    fig.show()
else:
    print("No cost sensitivity data in registry")

# %% [markdown]
# ## 4. Full Universe vs the Cost-Feasible Screen
#
# The first cost lever is *which names to trade*. The feasibility analysis
# (Ch16 §B.3) flagged that the half-spread varies widely across the 114-name
# panel; the expensive tail pays cost the intraday edge cannot cover. The
# **cost-feasible universe** keeps the cheapest-to-trade names (frozen per
# split) and drops that tail.
#
# This section reads the *same* featured slot design (10 slots, 0.90 entry,
# hold-only exit) on both universes directly from the registry — no new
# backtests — and compares Sharpe and trade count. The screen is not a
# parameter tweak: it changes which names the slot mechanism can hold, and with
# it both the turnover and the sign of the net Sharpe.

# %%
conn = sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db"))
screen_compare = pl.read_database(
    """
    SELECT
        COALESCE(json_extract(br.spec_json, '$.strategy.signal.universe_filter'),
                 'full')                                                 AS universe,
        COUNT(*)                                                          AS n_configs,
        ROUND(AVG(bm.sharpe), 3)                                          AS avg_sharpe,
        ROUND(MIN(bm.sharpe), 3)                                          AS min_sharpe,
        ROUND(MAX(bm.sharpe), 3)                                          AS max_sharpe,
        ROUND(AVG(bm.num_trades), 0)                                      AS avg_trades
    FROM backtest_runs br
    JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash
    JOIN prediction_sets ps ON br.prediction_hash = ps.prediction_hash
    JOIN training_runs tr ON tr.training_hash = ps.training_hash
    WHERE br.stage = 'signal' AND ps.split = 'validation'
      AND json_extract(br.spec_json, '$.strategy.signal.method') = 'slot_persistent_signal_exit'
      AND json_extract(br.spec_json, '$.strategy.signal.max_slots') = 10
      AND json_extract(br.spec_json, '$.strategy.signal.long_q') = 0.9
      AND tr.family = 'gbm'
    GROUP BY universe
    ORDER BY universe DESC
    """,
    connection=conn,
    schema_overrides={"avg_trades": pl.Float64},
)
conn.close()
print(screen_compare)

# %% [markdown]
# ### Reading the Screen's Effect
#
# Same slot design, same model family, validation window — the only difference
# is the tradeable universe. On the full 114-name panel the design averages a
# negative Sharpe and churns several thousand trades; on the cost-feasible
# universe it averages positive and trades roughly an order of magnitude less.
# The expensive tail was both the turnover source and the cost sink. Screening
# for cost feasibility is the upstream move that the per-share cadence sweep
# (Section 5) then builds on.

# %%
if not screen_compare.is_empty() and screen_compare.height == 2:
    full_row = screen_compare.filter(pl.col("universe") == "full")
    screened_row = screen_compare.filter(pl.col("universe") == "cost_feasible")
    if not full_row.is_empty() and not screened_row.is_empty():
        d_sharpe = screened_row["avg_sharpe"][0] - full_row["avg_sharpe"][0]
        trade_ratio = full_row["avg_trades"][0] / max(screened_row["avg_trades"][0], 1)
        print(
            f"Screen lifts avg Sharpe by {d_sharpe:+.2f} "
            f"({full_row['avg_sharpe'][0]:+.2f} → {screened_row['avg_sharpe'][0]:+.2f}) "
            f"and cuts turnover {trade_ratio:.1f}x "
            f"({full_row['avg_trades'][0]:.0f} → {screened_row['avg_trades'][0]:.0f} trades)."
        )

# %% [markdown]
# ## 5. Cadence × Per-Share Cost Analysis
#
# The bps sweep above fixes the rebalancing cadence at 15 minutes. But the
# cost-to-edge ratio depends on *how often* we trade, not just *how much* each
# trade costs. At 15-minute cadence the strategy rebalances 26 times per day;
# at hourly cadence only 6–7 times. Holding longer amortizes the fixed per-trade
# cost over a larger expected return per period.
#
# This section sweeps **cadence × per-share spread** — the central exhibit
# for this case study. We use a **per-share cost model** rather than bps,
# because for equities the execution cost is a dollar amount per share (half
# the bid-ask spread plus commission), not a percentage of notional:
#
# - 0.5¢/share ≈ sub-1 bps for a $100 stock (institutional DMA)
# - 1¢/share ≈ 1 bps (good agency execution)
# - 2¢/share ≈ 2 bps (typical NASDAQ-100 effective spread)
# - 5¢/share ≈ 5 bps (retail-quality execution)
#
# **Axis semantics — total per-share cost, simplified from production
# preset.** The grid value `cost_ps` on this heatmap represents *total*
# per-share round-trip cost; we split it half/half across the engine's
# `commission.per_share` and `slippage.fixed` knobs to walk a single
# dollar-per-share axis. Production (Ch16/17 signal & allocation) uses
# `set_backtest_costs_per_share` with a fixed IBKR Pro Tiered commission
# (`$0.0035/share`) plus a measured per-asset half-spread map from
# `liquidity_profile.parquet` — i.e. commission and spread are different
# knobs with different scales. This section's spec is a deliberate
# simplification so the cadence interaction can be read off a single cost
# axis; it is not the production cost shape.

# %%
from case_studies.utils.backtest_runner import normalize_prediction_columns
from case_studies.utils.registry import read_predictions

# Top engine signal-stage prediction by Sharpe
db_path = CASE_DIR / "run_log" / "registry.db"
conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
cur.execute("""
SELECT br.prediction_hash, tr.family, tr.config_name, bm.sharpe
FROM backtest_runs br
JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash
JOIN prediction_sets ps ON br.prediction_hash = ps.prediction_hash
JOIN training_runs tr ON ps.training_hash = tr.training_hash
WHERE br.stage = 'signal'
AND json_extract(br.spec_json, '$.strategy.rebalance.mode') = 'engine'
AND tr.family != 'deep_learning'
ORDER BY bm.sharpe DESC
LIMIT 1
""")
_row = cur.fetchone()
conn.close()

if _row is None:
    print("No signal-stage engine backtest found in registry. Skipping cadence sweep.")
    best_pred_hash = None
else:
    best_pred_hash = _row[0]
    best_source = f"{_row[1]}/{_row[2]}"
    print(f"Cadence sweep prediction: {best_source} (engine Sharpe={_row[3]:.3f})")

if best_pred_hash is not None:
    predictions_raw = normalize_prediction_columns(read_predictions(CASE_STUDY_ID, best_pred_hash))

    # Thin minute-level predictions to 15m for default cadence
    predictions_15m = predictions_raw.filter(
        (pl.col("timestamp").dt.minute() % 15 == 0) & (pl.col("timestamp").dt.second() == 0)
    )
    # Keep minute-level for asof alignment to coarser cadences
    predictions_minute = predictions_raw
    print(f"  Predictions: {len(predictions_raw):,} (minute), {len(predictions_15m):,} (15m)")
else:
    predictions_raw = predictions_15m = predictions_minute = None

# %% [markdown]
# ### Aligning predictions to target bar frequency
#
# The predictions are at 15-minute resolution. When rebalancing at hourly
# cadence, we take the **last available prediction** at or before each price
# bar timestamp via an asof join. This is realistic: the portfolio manager
# uses the most recent signal when the rebalance fires.

# %%
# Alternative cadences for the cadence × cost heatmap, driven by setup.yaml.
# The notebook never declares its own cadence list — single source of truth is
# ``backtest.sweep.cadence_sweep``. Labels and frequency tokens are derived
# from the cadence names so they always stay in sync.
CADENCES = get_cadence_sweep(CASE_STUDY_ID)
_CADENCE_TO_FREQ = {
    "15_minute": "15m",
    "30_minute": "30m",
    "1_hour": "1h",
    "2_hour": "2h",
    "4_hour": "4h",
    "daily_close": "1d",
}
_unknown_cadences = [c for c in CADENCES if c not in _CADENCE_TO_FREQ]
if _unknown_cadences:
    raise ValueError(
        f"cadence_sweep contains unknown cadence(s) {_unknown_cadences!r}; "
        f"valid tokens: {sorted(_CADENCE_TO_FREQ)}"
    )
CADENCE_LABELS = {c: _CADENCE_TO_FREQ[c] for c in CADENCES}
FREQ_MAP = dict(CADENCE_LABELS)

# Per-share cost grid: half-spread + commission in dollars per share.
# Single source of truth is ``backtest.sweep.cost_grid_half_spread_usd`` in
# setup.yaml; labels are derived from the grid so they always match.
COST_PER_SHARE_GRID = get_cost_grid_half_spread_usd(CASE_STUDY_ID)
COST_LABELS = [f"{v * 100:g}¢" for v in COST_PER_SHARE_GRID]

cadence_results = []


def align_predictions_to_bars(preds: pl.DataFrame, bar_timestamps: pl.Series) -> pl.DataFrame:
    """Align 15-min predictions to coarser bar timestamps via asof join."""
    # For each symbol, find the last prediction at or before each bar timestamp
    bar_df = pl.DataFrame({"timestamp": bar_timestamps}).unique().sort("timestamp")
    symbols = preds["symbol"].unique().sort().to_list()

    aligned = []
    for sym in symbols:
        sym_preds = preds.filter(pl.col("symbol") == sym).sort("timestamp")
        sym_bars = bar_df.with_columns(pl.lit(sym).alias("symbol"))
        joined = sym_bars.join_asof(
            sym_preds.drop("symbol"),
            on="timestamp",
            strategy="backward",
        )
        aligned.append(joined.drop_nulls("y_score"))

    return pl.concat(aligned) if aligned else pl.DataFrame()


# %% [markdown]
# ### Run one cadence × cost backtest
#
# Helper that builds the per-share cost spec and runs a single cadence backtest.
# Results are appended to `cadence_results` for the heatmap below.


# %%
def run_cadence_cost_backtest(
    cadence, cadence_label, cost_ps, cadence_prices, aligned_preds, state
):
    """Run one cadence × cost backtest and record results."""
    state["n_done"] += 1
    n_done = state["n_done"]

    spec = build_backtest_spec(
        CASE_STUDY_ID,
        bt_config,
        prices=cadence_prices,
        prediction_hash=best_pred_hash,
        initial_cash=bt_config.initial_cash,
        chapter="ch18",
        signal={"method": "equal_weight_top_k", "top_k": 20, "long_short": bt_config.long_short},
    )
    spec["strategy"]["rebalance"]["cadence"] = cadence
    spec["backtest_config"]["metadata"]["cadence"] = cadence

    if cost_ps > 0:
        spec["backtest_config"]["commission"]["model"] = "per_share"
        spec["backtest_config"]["commission"]["per_share"] = cost_ps / 2
        spec["backtest_config"]["commission"]["rate"] = 0.0
        spec["backtest_config"]["slippage"]["model"] = "fixed"
        spec["backtest_config"]["slippage"]["fixed"] = cost_ps / 2
        spec["backtest_config"]["slippage"]["rate"] = 0.0
    else:
        set_backtest_costs_bps(spec, commission_bps=0.0, slippage_bps=0.0)

    spec["cadence_sweep"] = True

    try:
        result = run_backtest(
            CASE_STUDY_ID,
            best_pred_hash,
            spec,
            prices=cadence_prices,
            predictions=aligned_preds,
            label=LABEL,
            register=True,
            initial_cash=bt_config.initial_cash,
            calendar=bt_config.calendar,
        )
        sharpe = result.metrics.get("sharpe", 0)
        n_trades = result.metrics.get("num_trades", 0)

        cadence_results.append(
            {
                "cadence": cadence_label,
                "cost_per_share": cost_ps,
                "cost_label": COST_LABELS[COST_PER_SHARE_GRID.index(cost_ps)],
                "sharpe": sharpe,
                "num_trades": n_trades,
                "cagr": result.metrics.get("cagr", 0),
                "max_drawdown": result.metrics.get("max_drawdown", 0),
            }
        )
        print(
            f"  [{n_done}/{state['n_total']}] {cadence_label} @ {cost_ps * 100:.1f}¢/sh: "
            f"Sharpe={sharpe:.3f}, trades={n_trades:,}"
        )
    except Exception as e:
        print(
            f"  [{n_done}/{state['n_total']}] {cadence_label} @ {cost_ps * 100:.1f}¢/sh: FAILED — {e}"
        )


# %%
sweep_state = {
    "n_total": len(CADENCES) * len(COST_PER_SHARE_GRID) if best_pred_hash else 0,
    "n_done": 0,
}
t0 = time.time()

for cadence in CADENCES if best_pred_hash else []:
    freq = FREQ_MAP[cadence]
    cadence_label = CADENCE_LABELS[cadence]

    cadence_prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        LABEL,
        split="validation",
        frequency=freq,
        max_symbols=MAX_SYMBOLS,
    )
    bar_ts = cadence_prices["timestamp"].unique().sort()

    aligned_preds = (
        predictions_15m if freq == "15m" else align_predictions_to_bars(predictions_minute, bar_ts)
    )
    if aligned_preds.is_empty():
        print(f"  {cadence_label}: no aligned predictions — skipping")
        continue

    print(
        f"\n--- {cadence_label} cadence: {len(bar_ts)} bars, {len(aligned_preds)} aligned predictions ---"
    )
    for cost_ps in COST_PER_SHARE_GRID:
        run_cadence_cost_backtest(
            cadence, cadence_label, cost_ps, cadence_prices, aligned_preds, sweep_state
        )

# %%
elapsed_cadence = time.time() - t0
print(f"Cadence sweep: {sweep_state['n_done']} backtests in {elapsed_cadence:.0f}s")

# %% [markdown]
# ### Cadence × Cost Heatmap
#
# This is the central finding: the same signal that is worthless at 15-minute
# cadence becomes viable at hourly cadence with institutional-quality execution
# ($\leq$ 2¢/share effective spread). The table shows Sharpe ratio at each
# cadence × cost combination.

# %%
import matplotlib.pyplot as plt
import numpy as np

cadence_df = pl.DataFrame(cadence_results) if cadence_results else pl.DataFrame()

if not cadence_df.is_empty():
    pivot = cadence_df.pivot(on="cost_label", index="cadence", values="sharpe")
    cadence_order = ["15m", "30m", "1h", "4h"]
    cadences_present = [c for c in cadence_order if c in pivot["cadence"].to_list()]
    costs_present = [c for c in COST_LABELS if c in pivot.columns]

    matrix = np.zeros((len(cadences_present), len(costs_present)))
    for i, cad in enumerate(cadences_present):
        row = pivot.filter(pl.col("cadence") == cad)
        for j, cost_col in enumerate(costs_present):
            if cost_col in row.columns:
                val = row[cost_col][0]
                matrix[i, j] = val if val is not None else np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(costs_present)))
    ax.set_xticklabels(costs_present)
    ax.set_yticks(range(len(cadences_present)))
    ax.set_yticklabels(cadences_present)
    ax.set_xlabel("Effective Spread (per share)")
    ax.set_ylabel("Rebalancing Cadence")
    ax.set_title("Sharpe Ratio: Cadence × Per-Share Cost")

    for i in range(len(cadences_present)):
        for j in range(len(costs_present)):
            val = matrix[i, j]
            color = "white" if abs(val) > 1.0 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, label="Sharpe Ratio")
    fig.show()
else:
    print("No cadence sweep results")

# %%
if not cadence_df.is_empty():
    print("=== Cadence × Cost Summary ===")
    print(
        cadence_df.sort("cadence", "cost_per_share").select(
            "cadence", "cost_label", "sharpe", "num_trades"
        )
    )

# %% [markdown]
# ### Trade Count by Cadence
#
# Reducing the rebalancing cadence cuts trade counts dramatically, which is
# the mechanism behind the Sharpe improvement: fewer trades means less
# cumulative cost drag. The trade-off is signal decay — the 15-minute
# prediction becomes stale at longer horizons. The sweet spot for this
# dataset is hourly cadence where the signal retains enough edge to cover
# 1–2¢/share execution costs.

# %%
if not cadence_df.is_empty():
    zero_cost = cadence_df.filter(pl.col("cost_per_share") == 0.0)
    if not zero_cost.is_empty():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        cadences = zero_cost["cadence"].to_list()
        trades = zero_cost["num_trades"].to_list()
        sharpes = zero_cost["sharpe"].to_list()

        axes[0].barh(cadences, trades)
        axes[0].set_xlabel("Number of Trades")
        axes[0].set_title("Trade Count by Cadence (Zero Cost)")

        axes[1].barh(cadences, sharpes)
        axes[1].axvline(0, color="gray", linestyle="--")
        axes[1].set_xlabel("Sharpe Ratio")
        axes[1].set_title("Gross Sharpe by Cadence")

        fig.tight_layout()
        fig.show()

# %% [markdown]
# ## Key Takeaways
#
# 1. **The cost-feasibility screen is the upstream lever**: the same featured
#    slot design averages a negative Sharpe on the full 114-name universe and a
#    positive one on the cost-feasible subset, while trading roughly an order of
#    magnitude less (Section 4). Removing the expensive-spread tail removes both
#    the turnover source and the cost sink — this is the screen the carrier
#    trades on.
# 2. **Cadence is the second lever**: at the default 15-minute cadence the
#    full-universe every-bar baseline churns thousands of trades; coarser
#    cadences (30m–4h) amortize per-trade cost over a longer hold. The summary
#    table above gives exact counts per cadence.
# 3. **Per-share costs at coarser cadences**: The heatmap shows the per-share
#    cost levels at which Sharpe stays above zero at hourly and 4-hour
#    cadences. Institutional execution quality (sub-2¢/share) sits in that
#    band; retail-quality execution does not.
# 4. **4-hour cadence is the most cost-tolerant of the four cadences tested**.
#    The tradeoff is fewer rebalances per day, discarding some intraday signal.
# 5. **30-minute cadence is the cost-sensitive boundary**: Sharpe degrades
#    sharply at higher per-share costs.
# 6. **The cadence-cost interaction is the teaching surface**: Intraday signal
#    presence does not imply intraday execution viability. Strategy design must
#    jointly optimize rebalance frequency and execution infrastructure. The
#    Ch20 strategy-analysis notebook re-runs this sensitivity in bps units on
#    the top-Sharpe prediction surface and finds CI lower bound below zero across
#    the entire grid for all three labels — point-estimate viability, not
#    credibility-resolved positive edge.
# 7. **Benchmark reality check**: This dollar-neutral strategy does not capture
#    market direction. The strategy's value is uncorrelated returns, not
#    absolute performance — but that case must be made explicitly, not assumed.
#
# **Next**: The risk management notebook (Ch19) tests risk overlays on the top combos.
