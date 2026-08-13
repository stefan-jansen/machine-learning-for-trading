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
# # ETF Rotation: Backtest & Signal Evaluation
#
# **Chapter 16 — Strategy Simulation**
#
# Across the six model families trained on this universe, prediction-stage IC and
# signal-stage Sharpe rank configurations differently: the family with the highest
# rank correlation against forward returns is not the family that produces the
# highest Sharpe under top-k rotation. This notebook converts every registered
# prediction into a backtest across the full prediction × entry-scheme grid and
# quantifies that IC–Sharpe relationship for the 100-ETF cross-section at monthly
# cadence.
#
# **Purpose:** Convert ETF rotation model predictions into backtest results across all
# (prediction × entry scheme) combinations and quantify the IC–Sharpe relationship for
# 100 cross-asset ETFs at monthly rebalancing frequency.
#
# **Learning Objectives:**
# - Verify the backtest engine produces no spurious alpha on random signals before
#   committing to the full sweep
# - Run the parametric sweep over all predictions and entry schemes, registering each
#   result for downstream analysis
# - Interpret the IC–Sharpe scatter for ETFs to understand when prediction accuracy
#   predicts trading profitability
# - Apply DSR to identify which signal-stage Sharpes survive multiple-testing correction
#
# **Book Reference:** Chapter 16, Sections 16.4–16.8
#
# **Prerequisites:** Completed model training (Ch11–15) for this case study. Predictions
# for all model families must be registered in `registry.db`.

# %%
"""Ch16 Backtest & Signal Evaluation — ETF rotation case study."""

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
CASE_STUDY_ID = "etfs"
LABEL = ""
SPLIT = "validation"
TOP_K = 0  # 0 = use smallest top_k from setup.yaml backtest.sweep.top_k_grid
MAX_SYMBOLS = 0
FORCE_REBACKTEST = False  # Set True to re-backtest even if a complete backtest_hash exists
TOP_N_PREDICTIONS = None

# %% [markdown]
# ## 1. Setup & Plumbing Test
#
# Before running the parametric sweep, we verify the backtest pipeline itself
# is sound. A random signal should produce Sharpe $\approx 0$. If it doesn't,
# the pipeline has a bug that would contaminate all downstream results.
#
# For the ETF universe — 100 ETFs, monthly calendar, long-only — any systematic
# alpha in a random signal indicates look-ahead, data misalignment, or a cost
# model that is not being applied correctly.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_PREDICTIONS is None:
    TOP_N_PREDICTIONS = get_top_n_predictions(CASE_STUDY_ID, "signal")

if not LABEL:
    LABEL = bt_config.primary_label

print(f"""=== Protocol Term Sheet ===
  Case study:    {CASE_STUDY_ID}
  Label:         {LABEL}
  Calendar:      {bt_config.calendar}
  Cadence:       {bt_config.cadence}
  Initial cash:  {bt_config.initial_cash:,.0f}
  Share type:    {bt_config.share_type}
  Commission:    {bt_config.commission_bps:.1f} bps
  Slippage:      {bt_config.slippage_bps:.1f} bps
  Total cost:    {bt_config.commission_bps + bt_config.slippage_bps:.1f} bps/leg
  Long/short:    {bt_config.long_short}
""")

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
print(f"Prices: {len(prices):,} rows, {n_assets} assets; plumbing-test TOP_K={TOP_K}")

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
    status = "PASS" if abs(random_sharpe) < 1.5 else "FAIL"
    print(f"Random signal Sharpe: {random_sharpe:.3f}  [{status}]")
    if abs(random_sharpe) >= 1.5:
        print("WARNING: Random signal produces non-trivial Sharpe — investigate pipeline")
except ValueError as e:
    if "zero variance" in str(e).lower():
        print(f"Plumbing test skipped: {e} (too few assets for meaningful test)")
        random_sharpe = 0.0
    else:
        raise

# %% [markdown]
# ## 2. Parametric Sweep
#
# With 100 ETFs spanning equity, fixed income, commodity, currency, and alternative
# categories, the sweep tests whether cross-asset momentum survives the full
# implementation pipeline. Each (prediction × entry scheme) combination runs
# the identical `run_backtest()` call — the sweep is orchestration, not a
# separate code path.

# %%
pred_index = load_prediction_index(
    CASE_STUDY_ID,
    label=LABEL,
    split=SPLIT,
)

if pred_index.is_empty():
    msg = f"No predictions found for {CASE_STUDY_ID}/{LABEL}/{SPLIT}"
    raise RuntimeError(msg)

if TOP_N_PREDICTIONS > 0:
    pred_index = pred_index.head(TOP_N_PREDICTIONS)

n_predictions = len(pred_index)
print(f"Predictions to sweep: {n_predictions}")
ic_min, ic_max = pred_index["ic_mean"].min(), pred_index["ic_mean"].max()
if ic_min is not None:
    print(f"  IC range: {ic_min:.4f} — {ic_max:.4f}")
else:
    print("  IC range: not yet computed")

# %%
entry_schemes = get_entry_schemes_for(
    CASE_STUDY_ID, LABEL, n_assets, long_short=bt_config.long_short
)
n_schemes = len(entry_schemes)

print(f"\nEntry schemes ({n_schemes}):")
for es in entry_schemes:
    print(f"  {es['name']}: {es['method']} (top_k={es.get('top_k', '-')})")

total_backtests = n_predictions * n_schemes
print(
    f"\nTotal grid: {n_predictions} predictions × {n_schemes} schemes = {total_backtests} backtests"
)

# %%
results = []
t0 = time.time()
failed = 0
skipped = 0
existing_hashes = load_existing_backtest_hashes(CASE_STUDY_ID, stage="signal")
print(f"Existing signal-stage hashes in registry: {len(existing_hashes):,}")

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
            continue
        pending_schemes.append((scheme, spec))

    if not pending_schemes:
        continue

    predictions = normalize_prediction_columns(read_predictions(CASE_STUDY_ID, pred_hash))

    for scheme, spec in pending_schemes:
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
                f"  [{idx}/{total_backtests}] {elapsed:.0f}s ({rate:.1f} bt/s) | failed: {failed}"
            )

elapsed = time.time() - t0
print(
    f"\nSweep complete: {len(results)} backtests in {elapsed:.0f}s ({failed} failed, {skipped} skipped)"
)

# %% [markdown]
# ## 3. Signal Evaluation
#
# This section is **read-only** — it queries the registry via `BacktestExplorer`
# and does not depend on the sweep having just run. Re-running this section
# after adding predictions only requires new sweep results to be registered; the
# analysis code is unchanged.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)
print(repr(explorer))

# %% [markdown]
# ### Top Strategies
#
# The top signal-stage backtests reveal which model family translates its
# prediction quality into the highest top-k Sharpe, and how the IC–Sharpe
# relationship plays out across families.

# %%
top = explorer.best(stage="signal", top_n=10)
print(top.select("source", "signal_method", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ### Model Family Comparison
#
# The family comparison asks whether the prediction-stage IC ranking carries
# through to signal-stage Sharpe: does higher mean IC within a family reliably
# predict higher mean Sharpe, or do structural models extract portfolio-relevant
# rankings from the 100-ETF cross-section that prediction-focused models miss?

# %%
families = explorer.compare_families(stage="signal")
print(families)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sharpe distribution histogram
all_signal = explorer.best(stage="signal", top_n=9999)
if not all_signal.is_empty():
    axes[0].hist(all_signal["sharpe"].to_numpy(), bins=30, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Sharpe Ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Sweep Sharpes")

    # IC vs Sharpe
    axes[1].scatter(
        all_signal["ic_mean"].fill_null(0).to_numpy(),
        all_signal["sharpe"].to_numpy(),
        alpha=0.4,
        s=20,
    )
    axes[1].set_xlabel("Prediction IC (mean)")
    axes[1].set_ylabel("Backtest Sharpe")
    axes[1].set_title("IC → Sharpe: Better Prediction = Better Trading?")

fig.tight_layout()
fig.show()

# %% [markdown]
# **IC–Sharpe disconnect.** The scatter confirms what model analysis anticipated: the
# relationship between prediction IC and backtest Sharpe is positive but weak. The
# signal-stage Sharpe leader is not the IC leader; configurations with comparable
# rank correlation produce materially different top-k Sharpes. The scatter shows why
# IC alone is insufficient as a strategy selection criterion: for a top-k selection
# rule over 100 ETFs, the *distribution* of predicted scores across the cross-section
# matters as much as their rank correlation with realized returns.
#
# A model that concentrates signal in a subset of assets — even with modest mean IC —
# can outperform a more uniformly accurate model if the concentrated signal aligns with
# the strongest momentum assets. Portfolio construction (which assets enter the top-k
# and how they are sized) mediates the IC-to-Sharpe translation more than prediction
# accuracy alone.

# %% [markdown]
# ### Deflated Sharpe Ratio
#
# The parametric sweep tests many configurations, which inflates the apparent quality
# of the best result through selection. The DSR corrects the observed Sharpe for the
# number of strategies tested and the non-normality of returns.
#
# $$DSR = \Phi\left[\frac{(\hat{SR} - SR^*) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \hat{SR}^2}}\right]$$
#
# For the ETF universe, monthly rebalancing produces fewer return observations than
# daily strategies, which widens DSR confidence intervals. A signal-stage Sharpe near
# +0.6 over a multi-year validation window should survive the correction, but the
# adjustment is informative about how much of the observed performance is attributable
# to search.

# %%
from case_studies.utils.backtest_loaders import print_stage_dsr_summary

print_stage_dsr_summary(explorer, top_n=20, head=10)

# %% [markdown]
# ### Sharpe Progression Preview
#
# Tracking how Sharpe evolves from signal → allocation → costs → risk for the best
# prediction reveals where value is added and where it is destroyed. For ETFs,
# the expectation is that allocation and risk overlays make small adjustments to a
# signal-driven baseline — the monthly cadence limits the damage that portfolio
# construction choices can do to a sound prediction.

# %%
if not top.is_empty():
    best_pred = top["prediction_hash"][0]
    prog = explorer.progression(best_pred)
    if not prog.is_empty():
        print(f"\nSharpe progression for best prediction ({top['source'][0]}):")
        print(prog.select("stage", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ## Key Takeaways
#
# The ETF backtest confirms the IC–Sharpe disconnect identified in model analysis:
# the family with the highest prediction-stage rank correlation against forward
# returns is not the family that produces the highest top-k Sharpe. Prediction
# accuracy as measured by rank correlation does not fully predict trading
# performance for top-k strategies — the *distribution* of predicted scores across
# the cross-section, not just their average correlation with outcomes, shapes what
# enters the portfolio.
#
# Monthly rebalancing at 100-ETF scale produces a clean sweep: few enough rebalances
# to keep costs negligible, enough assets to build a diversified top-k selection.
# DSR and PBO diagnostics on this sweep test whether the signal-stage Sharpe lead
# survives selection across the configurations explored.
#
# The top configurations from this sweep feed directly into the allocation and cost
# analysis notebooks.
#
# **Next:** The allocation notebook (Ch17) tests how portfolio sizing methods
# — equal-weight, inverse-vol, HRP, MVO — interact with the top signal-stage
# predictions across the concentration grid.
