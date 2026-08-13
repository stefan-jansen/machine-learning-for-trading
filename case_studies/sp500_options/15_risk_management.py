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
# # S&P 500 Options: Risk Management
#
# **Chapter 19 - Risk Management**
#
# The S&P 500 options setup intentionally configures no generic risk-overlay
# sweep. Its overlapping hold-to-expiry cohorts, daily delta hedges, premium
# costs, and cash settlement are part of the strategy engine itself. Generic
# stop-loss and portfolio-halt grids would change the declared instrument and
# can create zero-variance artifacts, so Chapter 19 treats them as governance
# controls rather than validation hyperparameters.
#
# This notebook verifies that boundary against the completed allocation surface.
# It writes no registry rows when both configured risk-control lists are empty,
# then confirms that no risk-overlay result is available for selection. The
# strategy remains governed by the fixed carrier and the HTM cost cascade.
#
# **Book Reference:** Chapter 19, Sections 19.3–19.6
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""S&P 500 Options: Risk: Engine-Level Risk Rules."""

import json
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.sp500_options.backtest_contract import (
    assert_accepted_deep_baselines,
    assert_complete_allocation_surface,
    assert_complete_baseline_surface,
)
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import strategy_view
from case_studies.utils.registry import resolve_best_backtest_runs, resolve_best_predictions
from case_studies.utils.sweep_config import (
    get_allocators,
    get_checkpoints_per_config,
    get_portfolio_risk_controls,
    get_position_risk_controls,
    get_top_k_values_for,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_options"
LABEL = ""
MAX_SYMBOLS = 0
MAX_RISK_VARIANTS = 0  # 0 = all; >0 limits position + portfolio controls each
TOP_N_COMBOS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
assert_accepted_deep_baselines(CASE_DIR / "run_log" / "registry.db")
assert_complete_baseline_surface(CASE_DIR / "run_log" / "registry.db")
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "risk_overlay")
if not LABEL:
    LABEL = bt_config.primary_label

from case_studies.utils.backtest_loaders import VECTORIZED_CASE_STUDIES

IS_VECTORIZED = CASE_STUDY_ID in VECTORIZED_CASE_STUDIES
MODE_LABEL = "vectorized" if IS_VECTORIZED else "engine"
print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}, mode: {MODE_LABEL}")

# %% [markdown]
# ## 1. Load Top Combos from Allocation Stage
#
# The highest-ranked allocation result is loaded only to verify upstream
# completion and lineage. It is not promoted through a risk sweep because no
# risk variants are configured for this case study.

# %%
top_combos = resolve_best_backtest_runs(
    CASE_STUDY_ID, LABEL, split="validation", stage="allocation", top_n=TOP_N_COMBOS
)

if top_combos.is_empty():
    msg = "No allocation-stage results found. Run the portfolio management notebook first."
    raise RuntimeError(msg)

for row in top_combos.iter_rows(named=True):
    spec = json.loads(row["spec_json"])
    alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
    print(f"  Sharpe={row['sharpe']:.3f}  alloc={alloc}  bt_hash={row['backtest_hash'][:8]}")

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)
_allocation_predictions = resolve_best_predictions(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    stage="signal",
    top_n=get_top_n_predictions(CASE_STUDY_ID, "allocation"),
    checkpoints_per_config=get_checkpoints_per_config(CASE_STUDY_ID),
    universe_filter="liquid",
)
_allocation_methods = {
    allocation["method"]
    for allocation in get_allocators(CASE_STUDY_ID)
    if allocation["method"] != "equal_weight"
}
assert_complete_allocation_surface(
    CASE_DIR / "run_log" / "registry.db",
    prediction_hashes=set(_allocation_predictions["prediction_hash"].to_list()),
    top_ks=tuple(get_top_k_values_for(CASE_STUDY_ID, LABEL, prices["symbol"].n_unique())),
    allocators=_allocation_methods,
)

# %% [markdown]
# ### Configured Risk Boundary
#
# Empty position and portfolio grids are the expected production contract. A
# future non-empty grid requires an explicit methodology decision because it
# changes the HTM strategy rather than merely reporting its risk.

# %%
position_controls = get_position_risk_controls(CASE_STUDY_ID)
portfolio_controls = get_portfolio_risk_controls(CASE_STUDY_ID)
if MAX_RISK_VARIANTS > 0:
    position_controls = position_controls[:MAX_RISK_VARIANTS]
    portfolio_controls = portfolio_controls[:MAX_RISK_VARIANTS]
    print(f"Risk variants limited to {MAX_RISK_VARIANTS} each")

if position_controls or portfolio_controls:
    raise RuntimeError(
        "S&P 500 Options declares no generic risk-overlay sweep; "
        "review methodology before adding risk variants"
    )
print("No generic risk variants configured - registry remains read-only in this notebook.")

# %% [markdown]
# ## 2. Risk Overlay Sweep
#
# With both grids empty, there are no eligible inputs and therefore no backtest
# or registry write. The assertion keeps that behavior visible and fail-closed.

# %%
n_done = 0
assert not position_controls and not portfolio_controls
print("Risk sweep complete: 0 backtests; registry remains read-only.")

# %% [markdown]
# ## 3. Risk Impact Analysis
#
# This section is **read-only** and confirms the expected absence of risk-overlay
# rows. The allocation and premium-denominated HTM cost results remain the final
# validation surface. No hypothetical overlay statistic is interpreted.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %%
risk_df = explorer.risk_impact()

if not risk_df.is_empty():
    # Best by risk type
    for risk_type in risk_df["risk_type"].unique().sort().to_list():
        subset = risk_df.filter(pl.col("risk_type") == risk_type).sort("sharpe", descending=True)
        best = subset.head(1)
        print(f"  Best {risk_type}: {best['risk_name'][0]} → Sharpe={best['sharpe'][0]:.3f}")

    print(f"\nAll risk overlays ({len(risk_df)}):")
    print(
        risk_df.select("risk_name", "risk_type", "sharpe", "max_drawdown", "sharpe_delta")
        .sort("sharpe", descending=True)
        .head(15)
    )
else:
    print("No risk overlay data in registry")

# %% [markdown]
# ## Key Takeaways
#
# 1. No generic position or portfolio risk variants are configured, so this
#    notebook makes zero registry writes by design.
# 2. The HTM engine already models the strategy's economic controls: overlapping
#    cohort sizing, daily delta hedging, entry costs, and cash settlement.
# 3. Stop-losses and portfolio halts belong to the governance layer for this
#    case study. Sweeping them as validation hyperparameters would change the
#    declared strategy and expand the selection surface.
# 4. The downstream analysis therefore preserves the validation-only carrier
#    and reports the absence of a risk-overlay stage explicitly.
#
# **Next:** Ch20 synthesis aggregates the cross-stage rank-1 from this
# case study into the cross-case-study comparison; the contribution is
# the worked example of switching to a premium-denominated cost
# framework as the cost mitigation itself.
