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
# # NASDAQ-100 Microstructure: Backtest & Signal Evaluation
#
# **Chapter 16 — Strategy Simulation**
#
# This notebook translates Ch11–15 model outputs into backtested strategies for
# the NASDAQ-100 microstructure case study: 15-minute bars across 114 stocks.
# The backtest runs through the **ml4t-backtest engine** using 15-minute OHLCV
# bars constructed from AlgoSeek TAQ trade prices — the same data that supports
# position-level risk controls, realistic execution simulation, and proper cost
# accounting in downstream chapters (Ch17–19).
#
# 1. **Plumbing test** — verify the engine pipeline produces no spurious alpha
# 2. **Parametric sweep** — test all (prediction × signal method) combinations
# 3. **Statistical analysis** — DSR, family comparison, IC-to-Sharpe relationship
#
# Sections 1–2 generate new backtest results (write to registry). Section 3
# is read-only — it queries the registry via `BacktestExplorer` and can be
# re-run independently without re-running the sweep.
#
# **Book Reference:** Chapter 16, Sections 16.4–16.8
#
# **Prerequisites:** Completed model training (Ch11–15) for this case study.

# %%
"""Ch16 Backtest & Signal Evaluation — NASDAQ-100 Microstructure case study."""

import sqlite3
import time
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import build_backtest_spec, serializable_backtest_spec
from case_studies.utils.backtest_runner import (
    normalize_prediction_columns,
    run_backtest,
    run_plumbing_test,
)
from case_studies.utils.notebook_contracts import excluded_families
from case_studies.utils.registry import (
    backtest_hash_from_parts,
    load_existing_backtest_hashes,
    load_prediction_index,
    read_predictions,
)
from case_studies.utils.sweep_config import (
    get_entry_schemes_for,
    get_top_k_values_for,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
LABEL = ""
SPLIT = "validation"
TOP_K = 0  # 0 = use smallest top_k from setup.yaml backtest.sweep.top_k_grid
MAX_SYMBOLS = 0
FORCE_REBACKTEST = False  # Set True to re-backtest even if a complete backtest_hash exists
TOP_N_PREDICTIONS = None

# %% [markdown]
# ## 1. Setup & Plumbing Test
#
# Before running the parametric sweep, we verify the engine backtest pipeline
# is sound. A random signal should not produce spurious positive Sharpe. Under
# quote-aware 15-minute execution with dominant costs, random turnover can
# legitimately produce a negative Sharpe, so the failure condition here is
# positive alpha that survives costs rather than simple distance from zero.
# At 15-minute cadence with 114 stocks, even small pipeline artifacts can compound
# rapidly — the plumbing test is not optional.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_PREDICTIONS is None:
    TOP_N_PREDICTIONS = get_top_n_predictions(CASE_STUDY_ID, "signal")

if not LABEL:
    LABEL = bt_config.primary_label

print(
    f"""=== Protocol Term Sheet ===
  Case study:    {CASE_STUDY_ID}
  Label:         {LABEL}
  Calendar:      {bt_config.calendar}
  Cadence:       {bt_config.cadence}
  Commission:    {bt_config.commission_bps:.1f} bps
  Slippage:      {bt_config.slippage_bps:.1f} bps
  Total cost:    {bt_config.commission_bps + bt_config.slippage_bps:.1f} bps/leg
  Long/short:    {bt_config.long_short}
""",
    flush=True,
)
if excluded_families(CASE_STUDY_ID):
    print(
        "Active-model filter: excluding "
        f"{', '.join(sorted(excluded_families(CASE_STUDY_ID)))} pending corrected reruns",
        flush=True,
    )

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)
n_assets = prices["symbol"].n_unique()
if TOP_K == 0:
    _feasible_top_k = get_top_k_values_for(CASE_STUDY_ID, LABEL, n_assets)
    if not _feasible_top_k:
        raise ValueError(
            f"top_k_grid for {LABEL!r} in {CASE_STUDY_ID} has no value < "
            f"n_assets={n_assets}; declare a feasible k in setup.yaml"
        )
    TOP_K = _feasible_top_k[0]
print(f"Prices: {len(prices):,} rows, {n_assets} assets; plumbing-test TOP_K={TOP_K}", flush=True)

# %%
strategy_spec = build_backtest_spec(
    CASE_STUDY_ID,
    bt_config,
    prices=prices,
    prediction_hash="plumbing_test",
    initial_cash=bt_config.initial_cash,
    chapter="ch16",
    signal={
        "method": "score_weighted_top_k",
        "top_k": TOP_K,
        "long_short": bt_config.long_short,
    },
)

try:
    random_sharpe = run_plumbing_test(
        CASE_STUDY_ID,
        prices,
        strategy_spec,
        top_k=TOP_K,
        initial_cash=bt_config.initial_cash,
        calendar=bt_config.calendar,
    )

    status = "FAIL" if random_sharpe > 1.5 else "PASS"
    print(f"Random signal Sharpe: {random_sharpe:.3f}  [{status}]", flush=True)

    if random_sharpe > 1.5:
        print("WARNING: Random signal produces positive Sharpe — investigate pipeline", flush=True)
    elif random_sharpe < -1.5:
        print(
            "NOTE: Strongly negative random Sharpe reflects turnover drag under quote-aware costs",
            flush=True,
        )
except ValueError as e:
    if "zero variance" in str(e).lower():
        print(f"Plumbing test skipped: {e} (too few assets for meaningful test)", flush=True)
        random_sharpe = 0.0
    else:
        raise

# %% [markdown]
# ## 2. The Full-Universe Sweep (Act 1: Cost-Defeat)
#
# We begin with the naive approach the feasibility analysis warned against:
# rank across **all 114 names** and trade the signal directly, with no
# cost-feasibility screen. This is the full-universe sweep — every (prediction
# × entry scheme) combination through the **same `run_backtest()` function** as
# a single backtest. It establishes the baseline the rest of the chapter has to
# beat: at 15-minute cadence, ranking over the full universe pays the expensive
# tail on every rebalance, and the edge is consumed by cost.
#
# A secondary question this sweep answers is whether the small IC advantage of
# classification predictions (GBM multiclass, IC ~+0.008) translates to
# higher Sharpe than regression predictions (IC ~+0.007). Under the engine's
# next-bar-open execution with per-trade costs, the answer reverses what
# simpler analyses suggested: smooth regression predictions produce fewer
# position changes and less cost drag than discrete classification scores.

# %%
pred_index = load_prediction_index(
    CASE_STUDY_ID,
    label=LABEL,
    split=SPLIT,
)
if not pred_index.is_empty():
    # Exclude causal_dml (not a trading signal) and the synthetic ensemble
    # forecast — the ensemble is a cost-feasible-universe construct introduced
    # in Section 4 (Act 2), not part of the full-universe baseline sweep.
    pred_index = pred_index.filter(~pl.col("family").is_in(["causal_dml", "ensemble"]))

if pred_index.is_empty():
    msg = f"No predictions found for {CASE_STUDY_ID}/{LABEL}/{SPLIT}"
    raise RuntimeError(msg)

if TOP_N_PREDICTIONS > 0:
    pred_index = pred_index.head(TOP_N_PREDICTIONS)

n_predictions = len(pred_index)
print(f"Predictions to sweep: {n_predictions}", flush=True)
ic_min, ic_max = pred_index["ic_mean"].min(), pred_index["ic_mean"].max()
print(
    f"  IC range: {ic_min:.4f} — {ic_max:.4f}"
    if ic_min is not None
    else "  IC range: not yet computed",
    flush=True,
)

# %%
entry_schemes = get_entry_schemes_for(
    CASE_STUDY_ID, LABEL, n_assets, long_short=bt_config.long_short
)
n_schemes = len(entry_schemes)

print(f"\nEntry schemes ({n_schemes}):", flush=True)
for es in entry_schemes:
    print(f"  {es['name']}: {es['method']} (top_k={es.get('top_k', '-')})", flush=True)

total_backtests = n_predictions * n_schemes
print(
    f"\nTotal grid: {n_predictions} predictions × {n_schemes} schemes = {total_backtests} backtests",
    flush=True,
)

# %%
results = []
t0 = time.time()
failed = 0
completed = 0
skipped = 0
existing_hashes = load_existing_backtest_hashes(CASE_STUDY_ID, stage="signal")
print(f"Existing signal-stage hashes in registry: {len(existing_hashes):,}", flush=True)

for i, pred_row in enumerate(pred_index.iter_rows(named=True)):
    pred_hash = pred_row["prediction_hash"]
    source = pred_row["source"]
    ic_mean = pred_row["ic_mean"]

    pending_schemes = []

    for j, scheme in enumerate(entry_schemes):
        idx = i * n_schemes + j + 1

        signal = {
            "method": scheme["method"],
            "top_k": scheme.get("top_k", 20),
            "long_short": bt_config.long_short,
        }
        signal.update({k: v for k, v in scheme.items() if k not in ("name", "method")})
        spec = build_backtest_spec(
            CASE_STUDY_ID,
            bt_config,
            prices=prices,
            prediction_hash=pred_hash,
            initial_cash=bt_config.initial_cash,
            chapter="ch16",
            signal=signal,
        )
        backtest_hash = backtest_hash_from_parts(pred_hash, serializable_backtest_spec(spec))

        if backtest_hash in existing_hashes:
            skipped += 1
            if idx % 20 == 0 or idx == total_backtests:
                elapsed = time.time() - t0
                rate = idx / elapsed if elapsed > 0 else 0
                print(
                    f"  [{idx}/{total_backtests}] {elapsed:.0f}s ({rate:.1f} bt/s) | "
                    f"completed: {completed} skipped: {skipped} failed: {failed}",
                    flush=True,
                )
            continue
        pending_schemes.append((idx, scheme, spec))

    if not pending_schemes:
        continue

    predictions = normalize_prediction_columns(read_predictions(CASE_STUDY_ID, pred_hash))

    for idx, scheme, spec in pending_schemes:
        try:
            result = run_backtest(
                CASE_STUDY_ID,
                pred_hash,
                spec,
                prices=prices,
                predictions=predictions,
                label=LABEL,
                register=True,
                force_rebacktest=FORCE_REBACKTEST,
                initial_cash=bt_config.initial_cash,
                calendar=bt_config.calendar,
            )

            results.append(
                {
                    "prediction_hash": pred_hash,
                    "source": source,
                    "ic_mean": ic_mean,
                    "family": pred_row["family"],
                    "config_name": pred_row["config_name"],
                    "signal_method": scheme["name"],
                    "backtest_hash": result.backtest_hash,
                    "sharpe": result.metrics["sharpe"],
                    "total_return": result.metrics["total_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "cagr": result.metrics.get("cagr", 0.0),
                    "volatility": result.metrics.get("volatility", 0.0),
                    "num_trades": result.metrics.get("num_trades", 0),
                }
            )
            completed += 1
            if result.backtest_hash:
                existing_hashes.add(result.backtest_hash)
        except Exception as e:
            failed += 1
            results.append(
                {
                    "prediction_hash": pred_hash,
                    "source": source,
                    "ic_mean": ic_mean,
                    "family": pred_row["family"],
                    "config_name": pred_row["config_name"],
                    "signal_method": scheme["name"],
                    "backtest_hash": None,
                    "sharpe": None,
                    "total_return": None,
                    "max_drawdown": None,
                    "cagr": None,
                    "volatility": None,
                    "num_trades": None,
                }
            )

        if idx % 20 == 0 or idx == total_backtests:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            print(
                f"  [{idx}/{total_backtests}] {elapsed:.0f}s ({rate:.1f} bt/s) | "
                f"completed: {completed} skipped: {skipped} failed: {failed}",
                flush=True,
            )

elapsed = time.time() - t0
print(
    f"\nSweep complete: completed={completed}, skipped={skipped}, failed={failed} "
    f"in {elapsed:.0f}s",
    flush=True,
)

# %% [markdown]
# ## 3. Full-Universe Signal Evaluation (Act 1)
#
# This section is **read-only** — it queries the registry via `BacktestExplorer`
# and analyzes the full-universe sweep just run. Every table and figure here is
# **scoped to the full universe** (no cost-feasibility screen); the cost-feasible
# carrier is Section 4.
#
# Two selection methods are in the sweep: the naive **equal-weight top-k**
# baseline (re-rank and rebalance every bar) and the turnover-controlled **slot
# mechanism** (introduced in Section 4). The contrast between them is Act 1 of
# the cost story — turnover, not signal quality, sets the validation Sharpe at
# 15-minute cadence.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)
print(repr(explorer))

# Scope Act 1 to the full universe. The cost-feasible carrier rows
# (universe_filter == "cost_feasible") are analyzed in Section 4.
all_signal = explorer.best(stage="signal", top_n=99999)
full_signal = all_signal.filter(pl.col("universe_filter") == "full")
print(
    f"Full-universe signal backtests: {full_signal.height:,} "
    f"(of {all_signal.height:,} total signal-stage rows)"
)

# %% [markdown]
# ### The Naive Every-Bar Baseline Is Cost-Defeated
#
# The equal-weight top-k method re-ranks and rebalances every 15-minute bar.
# On the full universe it pays tens of thousands of round trips over the
# validation window and is **catastrophically** cost-defeated — a median Sharpe
# far below zero. The slot mechanism cuts turnover by roughly an order of
# magnitude; on the full universe its validation Sharpe is a coin-flip (median
# slightly negative, a fifth of configurations positive), and the best slot
# configurations reach the top of the sweep. Those full-universe slot winners do
# **not** survive out of sample — that is what the cost-feasibility screen in
# Section 4 and the holdout test in Ch18/Ch20 address.

# %%
method_split = (
    full_signal.group_by("signal_method")
    .agg(
        n=pl.len(),
        pos_frac=(pl.col("sharpe") > 0).mean().round(3),
        sharpe_max=pl.col("sharpe").max().round(2),
        sharpe_median=pl.col("sharpe").median().round(2),
    )
    .sort("n", descending=True)
)
print(method_split)

# %% [markdown]
# ### Top Full-Universe Strategies
#
# The strongest validation configurations are slot-mechanism runs reaching
# Sharpe ~+2. They are kept here for the out-of-sample test in Ch20, where the
# full-universe slot winners collapse — the in-sample ranking does not hold up.

# %%
top = full_signal.sort("sharpe", descending=True).head(10)
print(top.select("source", "signal_method", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ### Model Family Comparison (Full Universe)
#
# Which ML model families produce the most robust trading signals for 15-minute
# intraday data? The best *model* by IC may not produce the best *strategy* by
# Sharpe. GBM classification predictions (IC ~+0.008) lose ground under engine
# execution because discrete scores cause excessive position flipping; deep
# learning and gbm regression, with smoother outputs, fare better through
# trade-count efficiency.

# %%
families = (
    full_signal.group_by("family")
    .agg(
        n=pl.len(),
        sharpe_max=pl.col("sharpe").max(),
        sharpe_median=pl.col("sharpe").median(),
    )
    .sort("sharpe_max", descending=True)
)
print(families)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sharpe distribution histogram (full universe)
if not full_signal.is_empty():
    axes[0].hist(full_signal["sharpe"].to_numpy(), bins=30, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Sharpe Ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Full-Universe Sweep: Turnover Drives the Sharpe Distribution")

    # IC vs Sharpe
    axes[1].scatter(
        full_signal["ic_mean"].fill_null(0).to_numpy(),
        full_signal["sharpe"].to_numpy(),
        alpha=0.4,
        s=20,
    )
    axes[1].set_xlabel("Prediction IC (mean)")
    axes[1].set_ylabel("Backtest Sharpe")
    axes[1].set_title("IC → Sharpe: Better Prediction = Better Trading?")

fig.tight_layout()
fig.show()

# %% [markdown]
# ### Sharpe vs Trade Count Diagnostic
#
# At 15-minute cadence, the relationship between Sharpe and trade count
# reveals the cost dominance problem. Strategies with high trade counts
# (active rebalancing) have deeply negative Sharpe because cost drag scales
# linearly with the number of trades. The only configurations with positive
# Sharpe are those with very few trades — either degenerate strategies
# (near-constant predictions from DL models) or extreme concentration
# (top_k=5 with infrequent position changes).
#
# This diagnostic motivates the cadence × cost analysis in Ch18: reducing
# the rebalance frequency is the structural solution to cost dominance.

# %%
# Query Sharpe and num_trades for full-universe signal-stage backtests
# (universe_filter absent/null == full universe; the cost-feasible carrier rows
# are excluded so the cost-dominance pattern is read on the naive baseline).

db_path = CASE_DIR / "run_log" / "registry.db"
conn = sqlite3.connect(str(db_path))

trade_df = pl.read_database(
    """
    SELECT
        br.backtest_hash,
        bm.sharpe,
        bm.num_trades
    FROM backtest_runs br
    JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash
    WHERE br.stage = 'signal'
      AND json_extract(br.spec_json, '$.strategy.signal.universe_filter') IS NULL
    """,
    connection=conn,
    schema_overrides={"num_trades": pl.Float64, "sharpe": pl.Float64},
)
conn.close()

trade_df = trade_df.drop_nulls("num_trades")
print(f"Signal-stage backtests with trade data: {len(trade_df)}")
print(f"Trade count range: {trade_df['num_trades'].min():.0f} — {trade_df['num_trades'].max():.0f}")
print(f"Median trades: {trade_df['num_trades'].median():.0f}")

# %%
if not trade_df.is_empty():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: Sharpe vs trades
    axes[0].scatter(
        trade_df["num_trades"].to_numpy(),
        trade_df["sharpe"].to_numpy(),
        alpha=0.15,
        s=10,
    )
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Number of Trades")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].set_title("Sharpe vs Trade Count (Signal Stage)")

    # Highlight: positive Sharpe only
    positive = trade_df.filter(pl.col("sharpe") > 0)
    if not positive.is_empty():
        axes[0].scatter(
            positive["num_trades"].to_numpy(),
            positive["sharpe"].to_numpy(),
            color="green",
            alpha=0.6,
            s=20,
            label=f"Sharpe > 0 ({len(positive)})",
        )
        axes[0].legend()

    # Histogram: trade count distribution
    axes[1].hist(trade_df["num_trades"].to_numpy(), bins=50, edgecolor="white")
    axes[1].set_xlabel("Number of Trades")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Trade Counts")

    fig.tight_layout()
    fig.show()

    # Summary: positive-Sharpe strategies are low-trade
    if not positive.is_empty():
        print(
            f"\nPositive-Sharpe backtests: {len(positive)} / {len(trade_df)} "
            f"({len(positive) / len(trade_df):.1%})"
        )
        print(f"  Median trades (positive Sharpe): {positive['num_trades'].median():.0f}")
        print(f"  Median trades (all): {trade_df['num_trades'].median():.0f}")
        print(
            f"  → Positive Sharpe strategies trade "
            f"{trade_df['num_trades'].median() / max(positive['num_trades'].median(), 1):.1f}x less"
        )

# %% [markdown]
# The scatter confirms the pattern: positive Sharpe concentrates at the
# low-trade-count end. These are strategies where the model produces smooth
# predictions that trigger few position changes — essentially trading less
# frequently within the 15-minute bar structure. This is the motivation for
# the explicit cadence sweep in Ch18: rather than relying on model smoothness
# as a proxy for reduced trading, we directly control the rebalance frequency.

# %% [markdown]
# ## 4. The Cost-Feasible Carrier (Act 2)
#
# Act 1 established that ranking across all 114 names and rebalancing every bar
# is cost-defeated. Act 2 applies the **cost-feasibility screen** from the
# feasibility analysis (restrict to the cost-feasible universe — the
# cheapest-to-trade names, frozen per split) and replaces every-bar rebalancing
# with a turnover-controlled **slot mechanism**: a fixed number of concurrent
# positions (slots), each entered when its signal clears a rolling per-symbol
# percentile gate and held to a maximum horizon, displaced only by fresher
# signals. The book then trades on composition changes rather than re-ranking
# every bar. See `case_studies/utils/slot_strategy.py` for the mechanism.
#
# The featured design — 10 slots, 8-hour maximum hold, 0.90 entry percentile,
# hold-only exit, long-only — was selected on the full-universe grid and the
# validation robustness analysis (longer holds beat the 1–2h horizon on
# out-of-sample stability; hold-only beats added take-profit / signal-exit
# rules). These carrier backtests are registered on
# `universe_filter='cost_feasible'`; this section queries them directly.

# %%
conn = sqlite3.connect(str(db_path))
carrier = pl.read_database(
    """
    SELECT
        tr.family,
        tr.config_name,
        json_extract(br.spec_json, '$.strategy.signal.method')    AS method,
        json_extract(br.spec_json, '$.strategy.signal.max_slots')  AS slots,
        json_extract(br.spec_json, '$.strategy.signal.long_q')     AS entry_q,
        json_extract(br.spec_json, '$.strategy.signal.top_k')      AS top_k,
        bm.sharpe,
        bm.num_trades
    FROM backtest_runs br
    JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash
    JOIN prediction_sets ps ON br.prediction_hash = ps.prediction_hash
    JOIN training_runs tr ON tr.training_hash = ps.training_hash
    WHERE br.stage = 'signal' AND ps.split = 'validation'
      AND json_extract(br.spec_json, '$.strategy.signal.universe_filter') = 'cost_feasible'
    ORDER BY bm.sharpe DESC
    """,
    connection=conn,
    schema_overrides={"sharpe": pl.Float64, "num_trades": pl.Float64},
)
conn.close()
print(carrier)

# %% [markdown]
# ### The Slot Mechanism Clears the Cost Barrier
#
# The featured slot design on the cost-feasible universe trades ~900–1,000 times
# over the validation window — an order of magnitude fewer than the full-universe
# every-bar sweep — and turns positive. The equal-weight top-k baseline, run on
# the *same* screened universe, stays cost-defeated (only the widest, top-20 book
# clears zero): the screen alone is not enough, the turnover control is what
# converts the signal into a tradeable strategy.

# %%
slot_design = carrier.filter(
    (pl.col("method") == "slot_persistent_signal_exit")
    & (pl.col("slots") == 10)
    & (pl.col("entry_q") == 0.9)
)
gbm_slots = slot_design.filter(pl.col("family") == "gbm")
ensemble_slot = slot_design.filter(pl.col("family") == "ensemble")
eqw = carrier.filter(pl.col("method") == "equal_weight_top_k").sort("top_k")

print("Featured slot design (10 / 8h / 0.90 / hold-only), single gbm models:")
if not gbm_slots.is_empty():
    print(
        f"  models: {gbm_slots.height}  "
        f"Sharpe range {gbm_slots['sharpe'].min():+.2f} .. {gbm_slots['sharpe'].max():+.2f}  "
        f"mean {gbm_slots['sharpe'].mean():+.2f}"
    )
else:
    print("  models: 0 (carrier not present in this registry)")
if not ensemble_slot.is_empty():
    print(
        f"  ENSEMBLE (mean forecast of the {gbm_slots.height} gbm): "
        f"Sharpe {ensemble_slot['sharpe'][0]:+.3f}  "
        f"trades {ensemble_slot['num_trades'][0]:.0f}"
    )
print("\nEqual-weight top-k on the same screened universe (Act-1 echo):")
for r in eqw.iter_rows(named=True):
    print(f"  top_{int(r['top_k']):<2}: Sharpe {r['sharpe']:+.3f}  trades {r['num_trades']:.0f}")

# %% [markdown]
# ### Ensemble as a Stability Tool, Not a Sharpe Booster
#
# Across the gbm family the single-model validation Sharpes span a wide band and
# the ranking is dominated by estimation noise: the best model in validation is
# not reliably the best out of sample (this scramble is shown explicitly in the
# Ch20 synthesis). The mean-forecast **ensemble** does not top that band — it
# sits inside it — but it removes the need to bet on which single model will
# generalize. Its value is robustness to model-selection noise, **not** a Sharpe
# boost. This is the same conclusion the cross-case synthesis reaches in Ch20
# §20.4: ensembling is a stability tool, not a universal Sharpe booster.
#
# The contrast that makes the point is the in-sample star: the highest validation
# Sharpe in the whole screened set is a thin 5-slot linear configuration
# (`ridge`, ~+2.1) — and it is exactly the configuration that collapses out of
# sample (Ch20). Chasing the validation maximum is the mistake; the ensemble on
# the robust design is what holds up.

# %%
schematic = carrier.filter(
    (pl.col("family") == "linear") & (pl.col("method") == "slot_persistent_signal_exit")
).sort("sharpe", descending=True)
if not schematic.is_empty():
    print("In-sample star (validation max), collapses out of sample — see Ch20:")
    for r in schematic.iter_rows(named=True):
        print(f"  linear {int(r['slots'])}-slot: validation Sharpe {r['sharpe']:+.3f}")

# %% [markdown]
# ### Deflated Sharpe on the Cost-Feasible Carrier
#
# The selection-bias question — after K configurations were tried, does the
# leader have skill? — is answered on the cost-feasible carrier cohort
# (`cohort_metrics`, written by the uncertainty backfill). Effective trials are
# small here because the configuration search (which slot count, hold, entry,
# exit) was conducted upstream on the full universe; the screened registry
# carries the chosen design across models plus contrasts, not the full grid.
# The DSR therefore reflects selection over the model family at the fixed design,
# not the full config search — a caveat the synthesis chapter makes explicit.

# %%
conn = sqlite3.connect(str(db_path))
dsr_cohorts = pl.read_database(
    """
    SELECT cohort_type, family, k_variants,
           n_trials_effective_er, dsr_er, dsr_er_pvalue, leader_sharpe
    FROM cohort_metrics
    ORDER BY cohort_type, family
    """,
    connection=conn,
)
conn.close()
print(dsr_cohorts)

# %% [markdown]
# ## Key Takeaways
#
# 1. The plumbing test confirms the engine backtest pipeline produces no
#    spurious alpha — necessary validation before interpreting sweep results.
# 2. **Act 1 — turnover sets the Sharpe.** The naive every-bar equal-weight
#    baseline on the full universe is catastrophically cost-defeated (median
#    Sharpe far below zero, tens of thousands of trades). The slot mechanism cuts
#    turnover ~10× and reaches the top of the validation sweep — but on the full
#    universe its Sharpe is a coin-flip and the winners do not survive out of
#    sample. This is the baseline the cost-feasibility screen has to beat.
# 3. The IC-to-Sharpe scatter is not monotone — a higher IC does not guarantee
#    a higher Sharpe. Label choice, portfolio construction, and especially
#    trade frequency all mediate the relationship.
# 4. The Sharpe-vs-trade-count diagnostic reveals cost dominance: configurations
#    with thousands of trades per validation window pay ruinous turnover at
#    15-minute cadence — the structural problem the slot mechanism addresses.
# 5. **Act 2 — the cost-feasible carrier turns positive.** Restricting to the
#    cost-feasible universe and replacing every-bar rebalancing with the
#    turnover-controlled slot mechanism produces a positive validation Sharpe at
#    ~900–1,000 trades. Equal-weight top-k on the *same* screened universe stays
#    cost-defeated, so the turnover control — not the screen alone — does the
#    work.
# 6. The mean-forecast ensemble is a **stability tool, not a Sharpe booster**: it
#    sits inside the single-model band but removes the model-selection lottery.
#    The validation-max thin linear configuration is the overfitting trap that
#    collapses out of sample (Ch20).
#
# **Next:** The allocation notebook (Ch17) carries the cost-feasible carrier
# through portfolio sizing; the cost notebook (Ch18) quantifies the
# full-vs-screened difference directly.
