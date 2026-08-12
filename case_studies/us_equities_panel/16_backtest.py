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
# # US Equities Panel: Backtest & Signal Evaluation
#
# **Chapter 16 — Strategy Simulation**
#
# The US Equities Panel is the broadest universe in the book: 3,200 stocks,
# daily frequency, and 16-fold cross-validation. The top-Sharpe lineage on the
# 1-day primary label is GBM `leaves_63_huber` with validation Sharpe 1.685
# [1.18, 2.14] (PSR p = 3.1e-10). The 2016-Q1 to 2018-Q1 holdout reads
# Sharpe 0.681 with paired-bootstrap diff vs validation of -4.15 [-7.69,
# -1.69], p = 0.007 — the deterioration is statistically resolved on the
# negative side under index-paired resampling. This case study is therefore
# the book's principal example of strong validation evidence paired with
# negative holdout closure, and a place to look hard at fold-count
# sufficiency, regime coverage, and the cost-sensitivity envelope before
# any deployment claim.
#
# This notebook runs the full Ch16 pipeline:
#
# 1. **Plumbing test** — verify the backtest engine produces no spurious alpha
# 2. **Parametric sweep** — test all (prediction × signal method) combinations
# 3. **Statistical analysis** — DSR, family comparison, cost sensitivity preview
#
# Sections 1–2 generate new backtest results (write to registry). Section 3
# is read-only — it queries the registry via `BacktestExplorer` and can be
# re-run independently without re-running the sweep.
#
# **Book Reference:** Chapter 16, Sections 16.4–16.8
#
# **Prerequisites:** Completed model training (Ch11–15) for this case study.

# %%
"""Ch16 Backtest & Signal Evaluation — US Equities Panel case study."""

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
CASE_STUDY_ID = "us_equities_panel"
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
# With 3,200 assets the plumbing test is especially important: large universes
# make it easier for structural artifacts (look-ahead, rebalancing timing, cost
# accounting) to create spurious non-zero Sharpe even under random signals.

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
# Sweep all (prediction × entry scheme) combinations using the **same
# `run_backtest()` function** as a single backtest — the sweep is pure
# orchestration, not a separate implementation.
#
# For this case study, GBM predictions are expected to dominate the top of the
# ranking. The sweep confirms whether that IC advantage translates uniformly to
# Sharpe across all signal methods, or whether there is sensitivity to how
# predictions are converted to positions.

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
# and does not depend on the sweep having just run. You can re-run this section
# at any time to analyze existing results.
#
# Key questions for this case study: Does the highest-IC GBM prediction
# (pooled IC 0.023 [0.020, 0.026] on the 1-day primary label) translate to
# proportionally higher Sharpe at the signal stage, and does the broader
# universe produce more stable Sharpe distributions than narrower case
# studies?

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)
print(repr(explorer))

# %% [markdown]
# ### Top Strategies
#
# Best backtests at the signal stage, ranked by Sharpe ratio. For the US
# equities panel, GBM configurations occupy the top positions; the
# highest-Sharpe entry is `leaves_63_huber` × top-K equal-weight at validation Sharpe 1.685
# [1.18, 2.14]. That figure reflects the 1-day rebalancing cadence — a
# significant portion of the gross alpha is turnover-driven, and the
# cost-sensitivity sweep in Ch18 maps how much of it survives realistic
# frictions.

# %%
top = explorer.best(stage="signal", top_n=10)
print(top.select("source", "signal_method", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ### Model Family Comparison
#
# For the US equities panel, GBM holds the top-Sharpe lineage on the 1-day
# primary label; IPCA at the longer horizon is the strongest characteristic-
# conditioned latent-factor reading on this panel and worth comparing here.
# The 1-day GBM signal is the basis for the validation Sharpe of 1.685
# [1.18, 2.14]. The family comparison shows how the predictive IC ranking
# carries through to the signal-stage Sharpe ranking once each family's
# top configuration is mapped through the same backtest spec.

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
# ### Deflated Sharpe Ratio
#
# The DSR corrects observed Sharpe ratios for the number of strategies tested.
# A strategy that looks good after testing hundreds of configurations may simply
# be the best of many noise realizations.
#
# $$DSR = \Phi\left[\frac{(\hat{SR} - SR^*) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \hat{SR}^2}}\right]$$
#
# This is particularly relevant for the US equities panel: 16 CV folds, full
# GBM family sweep, and multiple signal methods produce a large strategy count.
# The top-Sharpe lineage's raw validation Sharpe of 1.685 looks compelling on
# its own; the DSR test asks whether it remains significant after
# accounting for all configurations tested. PBO across the IS/OOS
# combinations is 0.27 in the locked registry — moderate IS/OOS rank
# stability — and DSR / expected-max-Sharpe / k_variants are NULL for this
# case study, so a formal selection-adjusted track-record deflation is not
# available to print here.

# %%
from case_studies.utils.backtest_loaders import print_stage_dsr_summary

print_stage_dsr_summary(explorer, top_n=20, head=10)

# %% [markdown]
# ### Sharpe Progression Preview
#
# For the best prediction, show how Sharpe changes across pipeline stages
# (if allocation/cost/risk stages have been run).
#
# For the US equities panel, the progression from signal to cost stage is
# expected to be steep: daily rebalancing implies high turnover, so even
# modest transaction costs significantly erode the gross Sharpe. The gap
# between signal Sharpe and cost-adjusted Sharpe is the central risk for
# this case study — visible here before the full cost sweep in Ch18.

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
# 1. GBM `leaves_63_huber` is the top-Sharpe lineage on the 1-day primary
#    label: pooled IC 0.023 [0.020, 0.026] translates to validation Sharpe
#    1.685 [1.18, 2.14] on the top-K equal-weight signal-stage backtest.
#    Other families occupy the lower ranks under the same signal-method
#    spec, which is what we want before reading the strategy-stage outputs.
# 2. The validation Sharpe sits well above zero on a 16-fold daily panel
#    (PSR p = 3.1e-10), but the supporting cost-sensitivity sweep in Ch18
#    shows that the daily rebalance cadence carries the gross result; the
#    edge-to-cost ratio against the 1.2x kill-condition floor is recorded
#    as `evidence_partial` in the spine narrative_facts.
# 3. PBO of 0.27 indicates moderate IS/OOS rank stability; DSR /
#    expected-max-Sharpe / k_variants are NULL in the locked registry for
#    this case study, so the selection-adjusted track-record deflation
#    cannot be reported numerically here.
# 4. IC and signal-stage Sharpe ranking line up across families — the
#    high-IC family also produces the high-Sharpe signal-stage backtest,
#    so this is a case where gross IC is a useful predictor of gross
#    Sharpe before frictions enter.
# 5. The progression preview sets up Ch18: the cost-sensitivity envelope
#    runs from a zero-cost gross Sharpe of 3.98 to materially negative at
#    the high-cost end of the post-decimalization grid; the slope is what
#    drives the deployment question, not the gross Sharpe alone.
#
# **Next:** The allocation notebook tests how portfolio sizing methods
# (equal-weight, inverse-vol, HRP, MVO) interact with the best GBM signals,
# and whether concentration adjustments can partially offset cost exposure.
