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
# # S&P 500 Equity+Options: Risk Controls
#
# This notebook applies predeclared position-level risk rules to the best
# eligible allocation lineage. It asks whether stop losses, trailing stops, or
# time exits improve validation Sharpe and drawdown relative to the same
# no-overlay baseline. The 2021 holdout remains sealed.
#
# **Learning objectives**
#
# 1. Carry one full-coverage allocation winner into a controlled risk sweep.
# 2. Compare fixed stop-loss, trailing-stop, and time-exit rules.
# 3. Evaluate each overlay with a paired block-bootstrap Sharpe-difference
#    interval against its exact no-overlay baseline.
# 4. Separate validation tuning from evidence of out-of-sample efficacy.
#
# **Book reference:** Chapter 19, Sections 19.3-19.6.
#
# **Prerequisites:** `15_portfolio_management` and the cost diagnostics in
# `16_costs`. Signals form after Friday's close and execute at the next
# available open, normally Monday. The current-constituent universe retains
# survivorship bias. Results describe this retrospective roster during the
# development window, not the historical index-membership process or a
# prospective S&P 500 population.

# %%
"""S&P 500 Equity+Options: position-level risk-control sweep."""

import json
import sqlite3
import time
import warnings

import matplotlib.pyplot as plt
import polars as pl

warnings.filterwarnings("ignore")

# %% [markdown]
# Shared helpers keep strategy construction, execution, registry access, and
# paired uncertainty consistent with the preceding pipeline stages.

# %%
from case_studies.utils.backtest_loaders import (
    VECTORIZED_CASE_STUDIES,
    get_backtest_config,
    load_backtest_prices_for,
    warmup_periods_for,
)
from case_studies.utils.backtest_presets import (
    clone_backtest_spec,
    ensure_backtest_spec,
    strategy_view,
)
from case_studies.utils.backtest_runner import precompute_weights, run_backtest
from case_studies.utils.registry import (
    backtest_hash_from_parts,
    model_source,
    read_predictions,
    resolve_best_backtest_runs,
)
from case_studies.utils.sweep_config import (
    get_allocators,
    get_position_risk_controls,
    get_top_n_predictions,
)
from case_studies.utils.uncertainty import (
    compute_paired_uncertainty,
    load_daily_returns_with_timestamp,
    periods_per_year_from_setup,
)
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
LABEL = ""
MAX_SYMBOLS = 0
MAX_RISK_VARIANTS = 0
TOP_N_COMBOS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
REGISTRY_DB = CASE_DIR / "run_log" / "registry.db"
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "risk_overlay")
if not LABEL:
    LABEL = bt_config.primary_label
if CASE_STUDY_ID in VECTORIZED_CASE_STUDIES:
    raise RuntimeError("This notebook requires engine-level position rules")

print(
    f"Case study: {CASE_STUDY_ID}; label: {LABEL}; selected lineages: {TOP_N_COMBOS}; mode: engine"
)

# %% [markdown]
# ## 1. Advance the best eligible strategy carrier
#
# Selection compares the equal-weight baseline with active alternative
# allocators using validation Sharpe and maximum prediction coverage.
# Historical rows from removed allocators cannot enter the corrected risk stage.

# %%
active_allocators = {item["method"] for item in get_allocators(CASE_STUDY_ID)}
baseline_pool = resolve_best_backtest_runs(
    CASE_STUDY_ID, LABEL, split="validation", stage="signal", top_n=9999
)
allocation_pool = resolve_best_backtest_runs(
    CASE_STUDY_ID, LABEL, split="validation", stage="allocation", top_n=9999
)
candidate_pool = pl.concat([baseline_pool, allocation_pool], how="diagonal_relaxed").unique(
    "backtest_hash"
)
candidate_hashes = candidate_pool["prediction_hash"].unique().to_list()
if not candidate_hashes:
    raise RuntimeError("No full-coverage baseline or allocation candidates found")

# %% [markdown]
# Resolve model labels from prediction provenance, then filter the strategy
# rows against the active allocator configuration.

# %%
with sqlite3.connect(REGISTRY_DB) as db:
    source_rows = db.execute(
        f"""
        SELECT p.prediction_hash, t.family, t.config_name
        FROM prediction_sets p
        JOIN training_runs t ON p.training_hash = t.training_hash
        WHERE p.prediction_hash IN ({",".join("?" for _ in candidate_hashes)})
        """,
        candidate_hashes,
    ).fetchall()
source_by_hash = {
    prediction_hash: model_source(family, config_name)
    for prediction_hash, family, config_name in source_rows
}

eligible_rows = []
for row in candidate_pool.iter_rows(named=True):
    strategy = strategy_view(json.loads(row["spec_json"]))
    allocator = strategy.get("allocation", {}).get("method", "equal_weight")
    if allocator == "equal_weight" or allocator in active_allocators:
        eligible_rows.append(
            {
                **row,
                "source": source_by_hash[row["prediction_hash"]],
                "allocator": allocator,
                "top_k": strategy.get("signal", {}).get("top_k"),
            }
        )

if len(eligible_rows) < TOP_N_COMBOS:
    raise RuntimeError(
        f"Expected {TOP_N_COMBOS} eligible strategy lineages, found {len(eligible_rows)}"
    )
top_combos = pl.DataFrame(eligible_rows).sort("sharpe", descending=True).head(TOP_N_COMBOS)
winner = top_combos.row(0, named=True)
print(
    f"Selected {winner['source']} with {winner['allocator']} allocation, "
    f"top-{winner['top_k']}, validation Sharpe {winner['sharpe']:.3f}"
)

# %% [markdown]
# The corrected carrier is NLinear with score weighting and ten stocks. The
# baseline remains fixed while the risk rule changes, so every comparison is a
# paired strategy perturbation rather than a new model-selection contest.

# %%
prices = load_backtest_prices_for(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    warmup_periods=warmup_periods_for(CASE_STUDY_ID),
    max_symbols=MAX_SYMBOLS,
)
print(
    f"Price support: {len(prices):,} rows across {prices['symbol'].n_unique()} historical symbols"
)

# %% [markdown]
# ## 2. Build the risk-control surface
#
# The sweep uses only the 14 rules declared in `setup.yaml`. A prior version
# derived additional MAE thresholds from the full validation price panel and
# evaluated them on that same panel. Those tuned-on-validation rules are not
# eligible for corrected v3.1 selection.

# %%
position_controls = get_position_risk_controls(CASE_STUDY_ID)
if MAX_RISK_VARIANTS > 0:
    position_controls = position_controls[:MAX_RISK_VARIANTS]
if not position_controls:
    raise RuntimeError("No position-level risk controls configured")

base_specs = []
for combo in top_combos.iter_rows(named=True):
    prediction_hash = combo["prediction_hash"]
    base_spec = ensure_backtest_spec(
        CASE_STUDY_ID,
        bt_config,
        json.loads(combo["spec_json"]),
        prices=prices,
        prediction_hash=prediction_hash,
        initial_cash=bt_config.initial_cash,
    )
    base_specs.append((combo, prediction_hash, base_spec))

# %% [markdown]
# Each declared rule changes only the position-risk block of its carrier's
# strategy specification.

# %%
plans = []
for combo, prediction_hash, base_spec in base_specs:
    for control in position_controls:
        spec = clone_backtest_spec(base_spec)
        spec["chapter"] = "ch19"
        rule = {"type": control["type"]}
        if control["type"] == "time_exit":
            rule["bars"] = control["bars"]
        else:
            rule["threshold"] = control["threshold"]
        spec["strategy"]["risk"] = {
            "name": control["name"],
            "position_rules": [rule],
        }
        plans.append(
            {
                "risk_name": control["name"],
                "risk_type": control["type"],
                "source": combo["source"],
                "allocator": combo["allocator"],
                "prediction_hash": prediction_hash,
                "baseline_hash": combo["backtest_hash"],
                "spec": spec,
                "backtest_hash": backtest_hash_from_parts(prediction_hash, spec),
            }
        )

# %%
with sqlite3.connect(REGISTRY_DB) as db:
    existing_hashes = {row[0] for row in db.execute("SELECT backtest_hash FROM backtest_runs")}
cached = sum(plan["backtest_hash"] in existing_hashes for plan in plans)
print(f"Planned {len(plans)} risk backtests; {cached} already complete")

# %% [markdown]
# Rules execute inside the event-driven engine on one precomputed allocation
# path. Missing rows fail the production run; cached hashes are reused.
#
# A shared weight path isolates the position rule as the only difference among
# a carrier's risk variants.


# %%
def execute_risk_plans(combo: dict, combo_plans: list[dict]) -> list[str]:
    combo_failures = []
    predictions = read_predictions(CASE_STUDY_ID, combo["prediction_hash"])
    weights = precompute_weights(
        predictions,
        combo_plans[0]["spec"],
        prices,
        label=LABEL,
        case_study=CASE_STUDY_ID,
        prediction_hash=combo["prediction_hash"],
    )
    for index, plan in enumerate(combo_plans, start=1):
        try:
            result = run_backtest(
                CASE_STUDY_ID,
                combo["prediction_hash"],
                plan["spec"],
                prices=prices,
                predictions=predictions,
                label=LABEL,
                register=True,
                initial_cash=bt_config.initial_cash,
                calendar=bt_config.calendar,
                precomputed_weights=weights,
            )
            existing_hashes.add(plan["backtest_hash"])
            print(
                f"[{index}/{len(combo_plans)}] {plan['risk_name']}: "
                f"Sharpe={result.metrics['sharpe']:.3f}; "
                f"max drawdown={result.metrics['max_drawdown']:.1%}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            combo_failures.append(f"{plan['backtest_hash']} {plan['risk_name']}: {exc}")
    return combo_failures


# %%
failures = []
started = time.monotonic()
for combo in top_combos.iter_rows(named=True):
    combo_plans = [
        plan
        for plan in plans
        if plan["prediction_hash"] == combo["prediction_hash"]
        and plan["backtest_hash"] not in existing_hashes
    ]
    if combo_plans:
        failures.extend(execute_risk_plans(combo, combo_plans))

# %%
if failures:
    raise RuntimeError("Risk-sweep failures:\n" + "\n".join(failures))
print(f"Risk surface complete in {(time.monotonic() - started):.1f}s")

# %% [markdown]
# ## 3. Measure paired overlay effects
#
# The registry query is keyed by the hashes planned above. Each overlay's
# return series is aligned by timestamp with its exact allocation baseline,
# then the same stationary-bootstrap indices are applied to both paths.

# %%
plan_meta = pl.DataFrame(
    [
        {
            "backtest_hash": plan["backtest_hash"],
            "baseline_hash": plan["baseline_hash"],
            "risk_name": plan["risk_name"],
            "risk_type": plan["risk_type"],
            "source": plan["source"],
            "allocator": plan["allocator"],
        }
        for plan in plans
    ]
)
placeholders = ",".join("?" for _ in plans)
with sqlite3.connect(REGISTRY_DB) as db:
    metrics = pl.read_database(
        f"""
        SELECT b.backtest_hash, b.stage, bm.sharpe, bm.max_drawdown, bm.num_trades
        FROM backtest_runs b
        JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
        WHERE b.backtest_hash IN ({placeholders})
        """,
        connection=db,
        execute_options={"parameters": [plan["backtest_hash"] for plan in plans]},
    )

risk_results = plan_meta.join(metrics, on="backtest_hash", how="inner")
if len(risk_results) != len(plans):
    raise RuntimeError(f"Expected {len(plans)} risk rows, found {len(risk_results)}")
if risk_results.filter(pl.col("stage") != "risk_overlay").height:
    raise RuntimeError("A planned risk hash was registered under the wrong stage")

# %% [markdown]
# A paired comparison aligns each overlay with its exact no-overlay carrier,
# drops their shared inactive prefix, and resamples both paths together.


# %%
def paired_overlay_metrics(row: dict) -> dict:
    baseline = load_daily_returns_with_timestamp(CASE_STUDY_ID, row["baseline_hash"])
    challenger = load_daily_returns_with_timestamp(CASE_STUDY_ID, row["backtest_hash"])
    if baseline is None or challenger is None:
        raise RuntimeError(f"Missing daily returns for {row['risk_name']}")
    aligned = (
        baseline.rename({"ret": "baseline_ret"})
        .join(challenger.rename({"ret": "challenger_ret"}), on="timestamp", how="inner")
        .sort("timestamp")
    )
    nonzero = aligned.with_row_index().filter(
        (pl.col("baseline_ret").abs() > 1e-15) | (pl.col("challenger_ret").abs() > 1e-15)
    )
    if nonzero.is_empty():
        raise RuntimeError(f"Degenerate return pair for {row['risk_name']}")
    aligned = aligned.slice(nonzero["index"].min())
    paired = compute_paired_uncertainty(
        aligned["challenger_ret"],
        aligned["baseline_ret"],
        periods_per_year=periods_per_year_from_setup(CASE_STUDY_ID),
        case_study=CASE_STUDY_ID,
        label=LABEL,
        n_boot=2000,
        seed=42,
    )
    if not paired:
        raise RuntimeError(f"Paired uncertainty failed for {row['risk_name']}")
    return {
        "backtest_hash": row["backtest_hash"],
        "sharpe_diff": paired["sharpe_diff"],
        "sharpe_diff_ci_lo": paired["sharpe_diff_ci95_lo"],
        "sharpe_diff_ci_hi": paired["sharpe_diff_ci95_hi"],
        "prob_overlay_wins": paired["prob_challenger_wins"],
    }


# %%
paired_rows = [paired_overlay_metrics(row) for row in risk_results.iter_rows(named=True)]

risk_results = risk_results.join(pl.DataFrame(paired_rows), on="backtest_hash").sort(
    "sharpe", descending=True
)

# %% [markdown]
# Fixed trailing stops dominate this validation comparison. The five-percent
# trailing stop raises Sharpe from 1.186 to 2.088 and reduces maximum drawdown
# magnitude from 32.9% to 12.2%. Its paired Sharpe improvement is 0.907 with a
# 95% interval of [0.067, 1.685]. It is the only fixed overlay whose paired
# interval excludes zero.

# %%
plot_risk = risk_results.sort("sharpe_diff")
fig_delta, ax_delta = plt.subplots(figsize=FIGSIZE["single_tall"], constrained_layout=True)

points = plot_risk["sharpe_diff"].to_list()
lo = plot_risk["sharpe_diff_ci_lo"].to_list()
hi = plot_risk["sharpe_diff_ci_hi"].to_list()
colors = [COLORS["positive"] if point > 0 else COLORS["neutral"] for point in points]
y = list(range(len(plot_risk)))
ax_delta.barh(y, points, color=colors, alpha=0.82)
ax_delta.errorbar(
    points,
    y,
    xerr=[
        [point - lower for point, lower in zip(points, lo, strict=True)],
        [upper - point for point, upper in zip(points, hi, strict=True)],
    ],
    fmt="none",
    ecolor=COLORS["slate"],
    capsize=2,
    linewidth=1,
)
ax_delta.axvline(0, color=COLORS["neutral"], linewidth=1, linestyle="--")
ax_delta.set_yticks(y, plot_risk["risk_name"].to_list(), fontsize=8)
ax_delta.set_xlabel("Paired annualized Sharpe difference vs no overlay")
add_message_title(
    ax_delta,
    "Trailing stops lead the paired validation comparison",
    "Bars: point difference; whiskers: 95% stationary-block bootstrap",
)
fig_delta.show()

# %%
fig_tradeoff, ax_tradeoff = plt.subplots(figsize=FIGSIZE["single"], constrained_layout=True)
type_colors = {
    "stop_loss": COLORS["amber"],
    "trailing_stop": COLORS["blue"],
    "time_exit": COLORS["copper"],
}
for risk_type, color in type_colors.items():
    subset = risk_results.filter(pl.col("risk_type") == risk_type)
    ax_tradeoff.scatter(
        -subset["max_drawdown"],
        subset["sharpe"],
        s=52,
        color=color,
        label=risk_type.replace("_", " ").title(),
    )
winner = risk_results.row(0, named=True)
ax_tradeoff.annotate(
    winner["risk_name"],
    (-winner["max_drawdown"], winner["sharpe"]),
    xytext=(7, -12),
    textcoords="offset points",
    fontsize=8,
    va="top",
)
ax_tradeoff.set_xlabel("Maximum drawdown magnitude")
ax_tradeoff.set_ylabel("Annualized validation Sharpe")
ax_tradeoff.legend(frameon=False)
add_message_title(
    ax_tradeoff,
    "The five-percent trailing stop offers the strongest trade-off",
    "Higher is better; farther left means a shallower drawdown",
)

fig_tradeoff.show()

# %% [markdown]
# ## Key takeaways
#
# 1. Risk selection is validation-only and carries the eligible NLinear,
#    score-weighted, top-ten lineage forward without consulting the holdout.
# 2. The five-percent trailing stop raises validation Sharpe from 1.186 to
#    2.088 and reduces maximum drawdown magnitude from 32.9% to 12.2%.
# 3. Its paired Sharpe improvement is 0.907 with a 95% interval of [0.067,
#    1.685]. The other 13 fixed overlays have intervals that include zero.
# 4. Full-validation MAE-calibrated thresholds are excluded from corrected
#    v3.1 because their thresholds were learned from the same validation paths
#    used to score them.
# 5. Fourteen predeclared overlays still constitute a selection cohort. The
#    holdout can be used once on the final carrier, not to choose among them.
#
# **Next:** `18_strategy_analysis` locks the corrected validation carrier and
# compares it with the preserved historical holdout without reusing the sealed
# window on the new lineage. See Chapter 20 for the strategy synthesis framework.
