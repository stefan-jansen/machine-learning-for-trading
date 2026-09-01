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
# # S&P 500 Equity+Options: Portfolio Allocation
#
# This notebook advances the highest-ranked primary-label model configurations
# from the equal-weight baseline and tests the point-in-time allocation methods
# `setup.yaml` declares. It asks whether portfolio sizing improves validation
# performance without changing the prediction model or consulting the holdout.
#
# **Learning objectives**
#
# 1. Apply the top-ten, one-checkpoint-per-configuration selection funnel.
# 2. Compare score weighting, conformal weighting, inverse volatility, risk
#    parity, MVO and HRP on the same validation predictions.
# 3. Measure how the number of selected stocks changes allocation performance.
# 4. Separate a validation improvement from evidence of out-of-sample efficacy.
#
# **Book reference:** Chapter 17, Sections 17.2-17.8.
#
# **Prerequisites:** `14_backtest` and its registry-backed equal-weight
# baselines. Signals form after Friday's close and execute at the next available
# open, normally Monday. Every result here is validation data. The
# current-constituent universe retains survivorship bias, so results describe
# this retrospective roster rather than historical S&P 500 membership or a
# prospective index population.

# %%
"""S&P 500 Equity+Options: portfolio allocation sweep."""

import sqlite3
import time
import warnings

import matplotlib.pyplot as plt
import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import OfficialPopulation, Study, open_study, population_supersedes
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    warmup_periods_for,
)
from case_studies.utils.backtest_presets import build_backtest_spec
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.conformal import ensure_conformal_calibration_identity
from case_studies.utils.notebook_contracts import prediction_members_in_force
from case_studies.utils.registry import (
    backtest_hash_from_parts,
    read_predictions,
    resolve_best_predictions,
)
from case_studies.utils.sweep_config import (
    get_allocators,
    get_checkpoints_per_config,
    get_top_k_values_for,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
LABEL = ""
MAX_SYMBOLS = 0
SKIP_EXPENSIVE_ALLOC = False
TOP_N_PREDICTIONS = None

# %% [markdown]
# ### What is asked for, and what it resolves to
#
# The parameters above are the request; the values this notebook runs on are resolved here under
# different names, so a resolved value can never overwrite the request that produced it. An
# injected parameter wins; otherwise the case study's own declaration does.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
TOP_N = (
    TOP_N_PREDICTIONS
    if TOP_N_PREDICTIONS is not None
    else get_top_n_predictions(CASE_STUDY_ID, "allocation")
)
CHECKPOINTS_PER_CONFIG = get_checkpoints_per_config(CASE_STUDY_ID)
ALLOCATION_LABEL = LABEL or bt_config.primary_label

print(
    f"Case study: {CASE_STUDY_ID}; label: {ALLOCATION_LABEL}; "
    f"top configs: {TOP_N}; checkpoints/config: {CHECKPOINTS_PER_CONFIG}"
)

# `Study.at` is the read-only form: one root, no activation. These notebooks only read the
# populations - their backtests reach the registry by their own paths - and every other way in
# ends in `activate()`, which rewrites `ML4T_OUTPUT_DIR` process-wide. `open_study` with the
# canonical tier routes to `Study.regenerate`, which refuses unless `features`, `labels` and
# `run_log` are symlinks: true in a maintainer worktree, false in every clean clone and CI run.
# `CASE_DIR` is already the directory this notebook resolved, including under a preview, so
# asking it directly answers for the registry the rest of the notebook reads.
_study = Study.at(CASE_DIR, case_study=CASE_STUDY_ID, entry_point="15_portfolio_management")
_members, _population_notes = prediction_members_in_force(_study)
for _note in _population_notes:
    print(_note)
CURRENT_MEMBERS = _members

# %% [markdown]
# ## 1. Advance the leading baselines
#
# Selection uses validation Sharpe and counts distinct `(family, config_name)`
# pairs. Each configuration enters through its best full-coverage checkpoint,
# so a long checkpoint grid cannot crowd other model families out of the
# allocation round.

# %%
top_preds = resolve_best_predictions(
    CASE_STUDY_ID,
    ALLOCATION_LABEL,
    split="validation",
    stage="signal",
    top_n=TOP_N,
    checkpoints_per_config=CHECKPOINTS_PER_CONFIG,
    prediction_hashes=CURRENT_MEMBERS,
)
if len(top_preds) != TOP_N:
    raise RuntimeError(f"Expected {TOP_N} advancing configurations, found {len(top_preds)}")

selected_hashes = top_preds["prediction_hash"].to_list()
top_preds.select("source", "prediction_hash", "sharpe")

# %% [markdown]
# The table above is authoritative: each row is one current configuration's
# best full-coverage checkpoint. These are inputs to the allocation comparison,
# not conclusions from it.

# %%
prices = load_backtest_prices_for(
    CASE_STUDY_ID,
    ALLOCATION_LABEL,
    split="validation",
    warmup_periods=warmup_periods_for(CASE_STUDY_ID),
    max_symbols=MAX_SYMBOLS,
)
n_assets = prices["symbol"].n_unique()
print(f"Price support: {len(prices):,} rows across {n_assets} historical symbols")

# %% [markdown]
# ## 2. Sweep alternative allocators
#
# Equal weight is the baseline established in `14_backtest`, so it is not an
# allocation-stage method. The sweep combines every declared concentration level with every
# declared alternative allocator, both read from `setup.yaml` rather than counted here, so the
# grid follows the declaration. Existing hashes are reused; only missing combinations run.

# %%
TOP_K_VALUES = get_top_k_values_for(CASE_STUDY_ID, ALLOCATION_LABEL, n_assets)
ALLOC_CONFIGS = get_allocators(CASE_STUDY_ID)
if SKIP_EXPENSIVE_ALLOC:
    ALLOC_CONFIGS = [
        alloc for alloc in ALLOC_CONFIGS if alloc["method"] not in {"mvo_ledoit_wolf", "hrp"}
    ]

allocation_methods = [alloc["method"] for alloc in ALLOC_CONFIGS]
if "equal_weight" in allocation_methods:
    raise RuntimeError("Equal weight is the baseline and cannot enter the allocation sweep")

print(f"TOP_K grid: {TOP_K_VALUES}; alternative allocators: {allocation_methods}")

# %% [markdown]
# Each planned row carries the same prediction set and execution convention as
# its parent baseline. Only the allocation method and concentration change.

# %%
planned = []
for pred_row in top_preds.iter_rows(named=True):
    for top_k in TOP_K_VALUES:
        for alloc in ALLOC_CONFIGS:
            spec = build_backtest_spec(
                CASE_STUDY_ID,
                bt_config,
                prices=prices,
                prediction_hash=pred_row["prediction_hash"],
                initial_cash=bt_config.initial_cash,
                chapter="ch17",
                signal={
                    "method": "equal_weight_top_k",
                    "top_k": top_k,
                    "long_short": bt_config.long_short,
                },
                allocation={**alloc, "top_k": top_k, "long_short": bt_config.long_short},
                label=ALLOCATION_LABEL,
            )
            # run_backtest resolves the conformal calibration identity into the spec
            # before registering, so hash the resolved spec or the cache never hits.
            spec = ensure_conformal_calibration_identity(spec)
            planned.append(
                {
                    "prediction_hash": pred_row["prediction_hash"],
                    "source": pred_row["source"],
                    "top_k": top_k,
                    "allocator": alloc["method"],
                    "spec": spec,
                    "backtest_hash": backtest_hash_from_parts(pred_row["prediction_hash"], spec),
                }
            )

# %% [markdown]
# A production run fails if any planned backtest fails. The notebook does not
# silently drop expensive allocators based on elapsed time.

# %%
with sqlite3.connect(CASE_DIR / "run_log" / "registry.db") as db:
    existing_hashes = {row[0] for row in db.execute("SELECT backtest_hash FROM backtest_runs")}

n_cached = sum(row["backtest_hash"] in existing_hashes for row in planned)
failures = []
started = time.monotonic()
print(f"Planned {len(planned)} allocation backtests; {n_cached} already complete")

for index, row in enumerate(planned, start=1):
    if row["backtest_hash"] in existing_hashes:
        continue
    try:
        result = run_backtest(
            CASE_STUDY_ID,
            row["prediction_hash"],
            row["spec"],
            prices=prices,
            predictions=read_predictions(CASE_STUDY_ID, row["prediction_hash"]),
            label=ALLOCATION_LABEL,
            register=True,
            initial_cash=bt_config.initial_cash,
            calendar=bt_config.calendar,
        )
        existing_hashes.add(row["backtest_hash"])
        print(
            f"[{index}/{len(planned)}] {row['source']} k={row['top_k']} "
            f"{row['allocator']}: Sharpe={result.metrics['sharpe']:.3f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{row['backtest_hash']} {row['source']} {row['allocator']}: {exc}")

if failures:
    raise RuntimeError("Allocation sweep failures:\n" + "\n".join(failures))
print(f"Allocation surface complete in {(time.monotonic() - started):.1f}s")

# %% [markdown]
# ### Record the grid that was run
#
# `planned` is every backtest this sweep intends to register, identified before any of them
# executed, and the loop above raises rather than dropping one. Publishing that list as an
# official population is what lets the freeze in `16_risk_management` tell an interrupted sweep
# from a finished one - which no reading of the registered rows can do, because an interruption
# leaves rows that look exactly like a smaller finished grid whether they are counted as rows,
# as model configurations, or as stages present. An interrupted run reaches neither this cell
# nor the population, which is the answer the freeze needs.
#
# Writing it activates the study, so it happens here rather than earlier: activation rewrites
# `ML4T_OUTPUT_DIR` process-wide, and the sweep above must resolve its artifacts through the
# same directory it read its prices and predictions from. A reader's clean clone has no
# writable registry and reports that instead of failing; it has no field to freeze either.

# %%
ALLOCATION_POPULATION = f"{CASE_STUDY_ID}-allocation-{ALLOCATION_LABEL}-v1"
# The generation this run retires, per population name. A plan that has grown - a new
# configuration advancing, a widened top-k grid - is a changed population under a live name and
# has to say which one it replaces; the refusal prints the current hash. Absent for a name this
# registry has never held, which is every clean clone and every first run of a label.
SUPERSEDES_ALLOCATION_POPULATIONS: dict[str, str] = {}

try:
    _writable = open_study(CASE_STUDY_ID, entry_point="15_portfolio_management")
except PermissionError as exc:
    print(f"Not recording the allocation plan here: {exc}")
else:
    if _writable.root != CASE_DIR:
        raise RuntimeError(
            f"15 ran its sweep against {CASE_DIR} but opened a study rooted at {_writable.root}. "
            "Recording the plan there would describe a registry this run did not write."
        )
    _plan = OfficialPopulation.create(
        _writable,
        name=ALLOCATION_POPULATION,
        member_kind="backtest",
        members=[row["backtest_hash"] for row in planned],
        supersedes=population_supersedes(
            _writable,
            name=ALLOCATION_POPULATION,
            declared=SUPERSEDES_ALLOCATION_POPULATIONS.get(ALLOCATION_POPULATION),
        ),
    )
    print(f"Allocation plan {ALLOCATION_POPULATION}: {_plan.hash}, {len(planned)} backtests")

# %% [markdown]
# ## 3. Compare the active allocation surface
#
# The analysis is restricted to the primary label, maximum-coverage prediction
# sets, and the ten configurations advanced above. Accumulated rows from other
# labels or earlier funnels cannot enter these summaries.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)
alloc_comparison = explorer.compare_allocators(
    label=ALLOCATION_LABEL,
    prediction_hashes=selected_hashes,
).filter(pl.col("allocator").is_in(allocation_methods))
alloc_comparison

# %% [markdown]
# The table pairs each allocator's mean Sharpe across its prediction-by-concentration
# combinations with its single strongest one. Read the pair, not either column alone:
# an allocator can lead on the mean while another owns the peak, and a mean over a
# handful of combinations moves on one of them.

# %%
plot_alloc = alloc_comparison.sort("avg_sharpe")
fig, ax = plt.subplots(figsize=FIGSIZE["single"], constrained_layout=True)
y = range(len(plot_alloc))
_ALLOCATOR_NAMES = {
    "inverse_vol": "Inverse volatility",
    "mvo_ledoit_wolf": "MVO (Ledoit-Wolf)",
    "hrp": "HRP",
}


def allocator_label(method: str) -> str:
    """Chart label for an allocator declared in `setup.yaml`."""
    return _ALLOCATOR_NAMES.get(method, method.replace("_", " ").capitalize())


labels = [allocator_label(name) for name in plot_alloc["allocator"].to_list()]
ax.barh(y, plot_alloc["avg_sharpe"], color=COLORS["blue"], alpha=0.82, label="Mean")
ax.scatter(plot_alloc["best_sharpe"], y, color=COLORS["amber"], s=48, zorder=3, label="Best")
ax.set_yticks(list(y), labels)
ax.set_xlabel("Annualized validation Sharpe")
ax.legend(frameon=False, loc="lower right")
add_message_title(
    ax,
    "Mean and peak Sharpe for each declared allocator",
    "Bars: mean across primary-label combinations; points: strongest single one",
)
fig.show()

# %% [markdown]
# ## 4. Inspect the leading allocation
#
# The table keeps the highest-Sharpe rows visible, and the figure after it shows whether that
# result depends on one concentration choice. **Read the two together.** A row that leads at one
# `top_k` and disappears at the next is a concentration artefact rather than an allocator that
# suits this signal, and the sweep runs every level precisely so that is visible rather than
# assumed.

# %%
top_rows = explorer.best(
    stage="allocation",
    top_n=10,
    label=ALLOCATION_LABEL,
    prediction_hashes=selected_hashes,
)
winner = explorer.inspect(top_rows["backtest_hash"][0])
winner_strategy = winner.spec["strategy"]
winner_allocator = winner_strategy["allocation"]["method"]
winner_top_k = winner_strategy["signal"]["top_k"]
baseline_sharpe = top_preds.filter(pl.col("prediction_hash") == winner.prediction_hash)["sharpe"][0]
allocation_delta = winner.metrics["sharpe"] - baseline_sharpe

print(
    f"Selected allocation: {winner.source}; allocator={winner_allocator}; "
    f"top_k={winner_top_k}; validation Sharpe={winner.metrics['sharpe']:.3f}"
)
print(f"Equal-weight baseline={baseline_sharpe:.3f}; allocation delta={allocation_delta:+.3f}")
top_rows.select("source", "prediction_hash", "sharpe", "cagr", "max_drawdown")

# %% [markdown]
# The point estimate is conditional on selecting this row from the full allocation sweep. An
# ordinary interval for this one return path would omit that search, so it is not reported as
# uncertainty about the selected allocation.

# %% [markdown]
# The curve below asks whether the allocation result depends on how many names are
# held. A Sharpe that falls as the basket widens is what dilution of the
# cross-sectional ranking looks like; one that is flat says the allocator, not the
# concentration, is doing the work.

# %%
winner_curve = explorer.concentration_curve(winner.prediction_hash).filter(
    pl.col("allocator").is_in(allocation_methods)
)
palette = [
    COLORS["blue"],
    COLORS["amber"],
    COLORS["positive"],
    COLORS["copper"],
    COLORS["slate"],
    COLORS["neutral"],
][: len(allocation_methods)]

fig, ax = plt.subplots(figsize=FIGSIZE["single"], constrained_layout=True)
for color, method in zip(palette, allocation_methods, strict=True):
    curve = winner_curve.filter(pl.col("allocator") == method).sort("top_k")
    ax.plot(
        curve["top_k"],
        curve["sharpe"],
        marker="o",
        linewidth=1.8,
        color=color,
        label=allocator_label(method),
    )
ax.axhline(baseline_sharpe, color=COLORS["neutral"], linestyle="--", linewidth=1.2)
ax.set_xticks(TOP_K_VALUES)
ax.set_xlabel("Selected stocks per rebalance")
ax.set_ylabel("Annualized validation Sharpe")
ax.legend(
    frameon=False,
    ncol=3,
    fontsize=8,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
)
add_message_title(
    ax,
    "How each allocator's Sharpe moves as the basket widens",
    f"Dashed line: equal-weight baseline Sharpe {baseline_sharpe:.3f}",
)
fig.show()

# %% [markdown]
# ## Key takeaways
#
# 1. **Equal weight is the baseline, not a competitor.** It is the signal stage's own weighting,
#    so it is excluded from the allocator menu and the notebook raises if it reappears there.
#    Every Sharpe here is read against it rather than ranked alongside it.
#
# 2. **An allocator is judged across concentration levels, not at one.** The method averages and
#    the per-`top_k` figure answer different questions: which allocator suits this signal on
#    average, and whether its leading row persists when the number of names held changes.
#
# 3. **The gain from allocation is measured against that lineage's own baseline.** The
#    difference between a lineage's equal-weight Sharpe and its highest allocated Sharpe is
#    what allocation contributed. That comparison holds the predictions fixed, which is what
#    isolates the allocator's effect from the signal's.
#
# 4. **The declared allocators read three different things.** Score weighting reads the
#    point prediction, `conformal_weighted` reads the width of its interval, and inverse
#    volatility, risk parity, MVO and HRP weight by a moment of returns. Where the conformal
#    intervals under-cover out of time - which `13_model_analysis` measures - only the
#    interval-width allocator inherits that miscalibration, so read its result beside that
#    coverage rather than on its own.
#
# 5. **These are selection-stage results on a current-constituent universe**, so they carry
#    survivorship bias and establish no out-of-sample edge. The holdout is untouched here.
#
# **Next:** [`16_risk_management`](16_risk_management.ipynb) applies friction to the leading validation lineage. See
# Chapter 18 for the transaction-cost framework.
