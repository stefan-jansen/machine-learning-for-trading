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
# # US Equities Panel — Strategy Analysis
#
# This notebook converts the case-study backtest registry for the US
# equities panel into a per-case-study strategy assessment. Every metric
# is reported with its block-bootstrap 95% confidence interval, every
# stage transition and validation→holdout comparison goes through
# `backtest_paired_metrics`, and the strategy tear sheet is generated
# from the diagnostic library with the equal-weight US-equities universe
# wired in as benchmark. Cross-case-study comparison is reserved for
# Chapter 20.
#
# **Learning objectives**
#
# - Read uncertainty-aware backtest metrics (Sharpe ± CI, PSR) on a
#   broad-panel long-short strategy from the registry rather than
#   transcribing point estimates.
# - Trace the rank-1 lineage through pipeline stages with paired-bootstrap
#   stage-transition deltas.
# - Use the equal-weight US-equities universe as a benchmark for both
#   validation and holdout windows.
# - Quantify holdout decay with a paired CI rather than point-difference
#   arithmetic.
# - Decompose performance into FF5+MOM factor exposure and a residual
#   alpha, with a placebo-portfolio reference.
#
# **Book reference**: Chapter 20, §20.1 (the §9 handoff feeds Ch20's
# cross-case-study aggregation).
#
# **Prerequisites**: case-study pipeline through `19_risk_management`;
# the locked registry (`case_studies/us_equities_panel/run_log/registry.db`).
#
# **Scope**: registry-read only — no training, no re-backtesting, no
# registry writes. The `backtest_paired_metrics` table was populated by
# `20_strategy_synthesis/01_aggregate_synthesis.py`.

# %%
"""US Equities Panel — Strategy Analysis."""

import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import torch  # ml4t.diagnostic dlopens cudart; torch must import first
import yaml

warnings.filterwarnings("ignore")

from ml4t.backtest.result import enrich_trades_with_signals
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.factor import FactorData
from ml4t.diagnostic.integration import (
    BacktestReportMetadata,
    generate_tearsheet_from_run_artifacts,
    profile_from_run_artifacts,
)
from ml4t.diagnostic.visualization.backtest.ml_plots import (
    plot_ic_time_series,
    plot_prediction_trade_alignment,
    plot_quintile_returns,
)
from ml4t.diagnostic.visualization.backtest.tail_risk import plot_tail_risk_analysis
from ml4t.diagnostic.visualization.backtest.tearsheet import (
    generate_backtest_tearsheet,
)
from ml4t.diagnostic.visualization.portfolio.drawdown_plots import (
    plot_drawdown_underwater,
)
from ml4t.diagnostic.visualization.portfolio.returns_plots import (
    plot_annual_returns_bar,
    plot_cumulative_returns,
    plot_monthly_returns_heatmap,
    plot_returns_distribution,
)
from ml4t.diagnostic.visualization.portfolio.risk_plots import plot_rolling_sharpe

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.benchmark import load_benchmark_metrics, load_benchmark_returns
from case_studies.utils.external_benchmarks import (
    align_to_strategy as align_external_benchmark,
)
from case_studies.utils.external_benchmarks import (
    compute_benchmark_diagnostics,
    compute_subperiod_diagnostics,
    load_spy_returns,
)
from case_studies.utils.factor_attribution import (
    compute_bootstrap_ci,
    compute_rolling_exposures,
    format_attribution_summary,
    load_factor_data,
    plot_attribution_waterfall,
    plot_rolling_exposures,
    run_factor_regression,
    run_placebo_benchmark,
)
from case_studies.utils.registry import (
    load_backtest_fold_metrics,
    load_backtest_metrics,
    load_paired_metrics,
)
from case_studies.utils.strategy_analysis import (
    ci_status,
    compute_operating_profile,
    fmt_gate,
    gate1_validation_sharpe_geq_zero,
    gate2_holdout_diff_not_excludes_zero_negatively,
    gate_passes,
    plot_concentration_curve,
    plot_equity_drawdown,
    plot_sharpe_waterfall,
    write_strategy_assessment,
)
from utils.paths import get_case_study_dir, get_output_dir

# %% tags=["parameters"]
MAX_SYMBOLS = 0

# %%
CASE_STUDY = "us_equities_panel"
PRIMARY_LABEL = "fwd_ret_1d"
PERIODS_PER_YEAR = 252  # NYSE calendar, daily bars
CASE_DIR = get_case_study_dir(CASE_STUDY)
OUTPUT_DIR = get_output_dir(20, CASE_STUDY)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CASE_DIR / "config" / "setup.yaml") as f:
    setup = yaml.safe_load(f)

explorer = BacktestExplorer(CASE_STUDY)
print(explorer)


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    return "—" if val is None else format(val, fmt)


def _fmt_ci(point: float | None, lo: float | None, hi: float | None, fmt: str = ".3f") -> str:
    if point is None:
        return "—"
    p = format(point, fmt)
    if lo is None or hi is None:
        return f"{p} [—, —]"
    return f"{p} [{format(lo, fmt)}, {format(hi, fmt)}]"


# %% [markdown]
# ## §1 Handoff from model analysis
#
# The strategy phase inherits a single rank-1 model from
# `15_model_analysis.py` §8. That model's daily-pooled IC and HAC-adjusted
# 95% CI define the upstream prior on backtest stability. With ~3,200
# stocks, the fundamental law of active management amplifies even modest
# per-stock IC into a tradable Sharpe through breadth — but the same
# breadth inflates the rank-1 in any large search, so the handoff prepares
# us for a wide validation-CI even when the IC reads positive.

# %%
# Rank-1 selection pools validation backtests across the holdout-eligible
# stages (signal / allocation / risk_overlay) and dedupes by prediction_hash,
# keeping the highest-Sharpe strategy_spec per trained model. Mirrors
# `20_strategy_synthesis/holdout.py::HOLDOUT_SELECTION_STAGES`.
_HOLDOUT_STAGES = ("signal", "allocation", "risk_overlay")
top_signal = (
    pl.concat(
        [explorer.best(stage=s, top_n=2000) for s in _HOLDOUT_STAGES],
        how="diagonal_relaxed",
    )
    .filter(pl.col("family") != "benchmark")
    .sort("sharpe", descending=True)
    .unique(subset=["prediction_hash"], keep="first", maintain_order=True)
    .head(1)
)
TOP_HASH = top_signal.row(0, named=True)["backtest_hash"]
TOP_PHASH = top_signal.row(0, named=True)["prediction_hash"]
RANK1_FAMILY = top_signal.row(0, named=True)["family"]
RANK1_CONFIG = top_signal.row(0, named=True)["config_name"]
RANK1_LABEL = top_signal.row(0, named=True).get("label", PRIMARY_LABEL)

_db = CASE_DIR / "run_log" / "registry.db"
with sqlite3.connect(str(_db)) as _con:
    _row = _con.execute(
        "SELECT ic_mean_daily, ic_ci_lo, ic_ci_hi, ic_t_hac, ic_p_hac, "
        "ic_n_days, ic_hac_lag, ic_pct_positive "
        "FROM prediction_metrics WHERE prediction_hash = ?",
        (TOP_PHASH,),
    ).fetchone()

ic_mean, ic_lo, ic_hi, ic_t, ic_p, ic_ndays, ic_lag, ic_pct = _row

print(f"Rank-1: family={RANK1_FAMILY}, config={RANK1_CONFIG}, training_label={RANK1_LABEL}")
print(f"        prediction_hash={TOP_PHASH}, backtest_hash={TOP_HASH}")
print(
    f"        (PRIMARY_LABEL in setup.yaml is {PRIMARY_LABEL}; IC printed below "
    f"uses {PRIMARY_LABEL}-grid HAC for the daily-pooled column, EW benchmark "
    f"uses {PRIMARY_LABEL}.)"
)
print()
print("Daily-pooled IC (validation):")
print(f"  IC = {_fmt_ci(ic_mean, ic_lo, ic_hi, '.4f')}  (HAC, lag={int(ic_lag)})")
print(f"  t_HAC = {ic_t:.3f}, p_HAC = {ic_p:.4f}")
print(f"  n_days = {int(ic_ndays)}, pct_positive = {ic_pct:.1%}")
print(f"  IC CI status: {ci_status(ic_lo, ic_hi)}")

# %% [markdown]
# The rank-1 prediction's IC sits at a small positive point estimate with
# the HAC-adjusted CI excluding zero on the positive side — modest, but
# statistically resolved at this sample size (≈4,000 daily cross-sections).
# Per the fundamental law, an IC of this magnitude across ~3,200 names
# implies a substantial Sharpe ceiling under decorrelated bets, which §3
# tests directly. Pct-positive close to 50% reflects the daily-noise
# structure of single-day forward returns: cross-sections oscillate
# day-to-day, but the rank-correlation accumulates.
#
# **Kill conditions** are encoded in `setup.yaml`:
#
# - `ic_floor: 0.01` — cross-sectional IC threshold;
# - `edge_to_cost_floor: 1.2` — net Sharpe / cost ratio threshold;
# - `micro_cap_concentration: 0.5` — flagged for manual review (no
#   programmatic ADV-quintile attribution implemented in this CS).
#
# The notebook evaluates these gates in §9 by reading the §3 / §5 / §6
# numbers directly. Outcomes are reported as evidence statements rather
# than verdict labels.

# %% [markdown]
# ## §2 Search context, family comparison, and lineage waterfall
#
# The signal stage of the locked sweep evaluated thousands of validation
# backtests across five model families and three label horizons (1d, 5d,
# 21d). The rank-1 row emerges as one realization of that search; what
# matters next is its context — where the rank-1 sits in the family-level
# distribution and how its performance evolves through the pipeline stages.

# %%
ctx = explorer.search_context("signal")
search_table = pl.DataFrame(
    [
        {"metric": "Total signal backtests", "value": f"{ctx['total']:,}"},
        {"metric": "Mean Sharpe", "value": f"{ctx['mean_sharpe']:.3f}"},
        {"metric": "Median Sharpe", "value": f"{ctx['median_sharpe']:.3f}"},
        {"metric": "P90 Sharpe", "value": f"{ctx['p90_sharpe']:.3f}"},
        {"metric": "% positive Sharpe", "value": f"{ctx['pct_positive']:.1f}%"},
        {"metric": "Top-by-Sharpe in this sweep", "value": f"{ctx['champion_sharpe']:.3f}"},
        {"metric": "Top-by-Sharpe percentile", "value": f"{ctx['champion_percentile']:.1f}%"},
    ]
)
print("Signal-stage search context:")
print(search_table)

# %%
# Family comparison reads sharpe_ci95 from backtest_metrics directly so the
# forest plot can render proper error bars rather than median-only points.
with sqlite3.connect(str(_db)) as _con:
    _famdf = pl.DataFrame(
        _con.execute(
            """
            SELECT
                t.family,
                bm.sharpe,
                bm.sharpe_ci95_lo,
                bm.sharpe_ci95_hi
            FROM backtest_metrics bm
            JOIN backtest_runs b ON bm.backtest_hash = b.backtest_hash
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t  ON p.training_hash = t.training_hash
            WHERE b.stage = 'signal'
              AND p.split = 'validation'
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
            """
        ).fetchall(),
        schema=["family", "sharpe", "sharpe_ci95_lo", "sharpe_ci95_hi"],
        orient="row",
    )

family_summary = (
    _famdf.group_by("family")
    .agg(
        n=pl.len(),
        sharpe_median=pl.col("sharpe").median(),
        sharpe_q25=pl.col("sharpe").quantile(0.25),
        sharpe_q75=pl.col("sharpe").quantile(0.75),
        sharpe_max=pl.col("sharpe").max(),
        pct_positive=((pl.col("sharpe") > 0).sum() / pl.len() * 100),
    )
    .sort("sharpe_median", descending=True)
)
print("Family-level signal-stage Sharpe summary:")
print(family_summary)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
fams = family_summary["family"].to_list()
y = np.arange(len(fams))
medians = family_summary["sharpe_median"].to_numpy()
q25 = family_summary["sharpe_q25"].to_numpy()
q75 = family_summary["sharpe_q75"].to_numpy()
maxima = family_summary["sharpe_max"].to_numpy()
ax.errorbar(
    medians,
    y,
    xerr=[medians - q25, q75 - medians],
    fmt="o",
    color="#1565C0",
    ecolor="#5B9BD5",
    elinewidth=2.0,
    capsize=4,
    label="median ±IQR",
)
ax.scatter(maxima, y, marker="x", color="#C62828", s=60, label="max", zorder=5)
ax.axvline(0, color="#9E9E9E", linewidth=0.8, linestyle="--")
ax.set_yticks(y)
ax.set_yticklabels(fams)
ax.set_xlabel("Validation Sharpe")
ax.set_title("Family-level signal Sharpe — IQR + max")
ax.invert_yaxis()
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.show()

# %% [markdown]
# Family-level distributions separate cleanly: GBM dominates with a
# positive median Sharpe and the highest top-by-family value, linear
# follows at a positive but lower median, while latent_factors,
# deep_learning, and tabular_dl medians cluster near or below zero.
# The maximum-by-family points show the rank-1 selection is led by GBM
# but with clear non-trivial outliers in linear; a search-cost adjustment
# is therefore informative even though GBM is the leader.

# %%
# Lineage with CI bars.
lineage = explorer.champion_lineage(TOP_PHASH)
ci_lo: dict[str, float] = {}
ci_hi: dict[str, float] = {}
for stage_name, info in lineage.items():
    bm = load_backtest_metrics(CASE_STUDY, backtest_hash=info["backtest_hash"])
    if not bm.is_empty():
        row = bm.row(0, named=True)
        ci_lo[stage_name] = row.get("sharpe_ci95_lo")
        ci_hi[stage_name] = row.get("sharpe_ci95_hi")

print("Lineage stages present for rank-1 prediction:")
for s, info in lineage.items():
    lo, hi = ci_lo.get(s), ci_hi.get(s)
    print(f"  {s}: hash={info['backtest_hash']}, Sharpe={_fmt_ci(info['sharpe'], lo, hi)}")

# %%
fig = plot_sharpe_waterfall(lineage, ci_lo=ci_lo, ci_hi=ci_hi)
fig.show()

# %%
# Stage-transition deltas via load_paired_metrics — never recompute paired
# metrics inline. We pull each present stage's challenger row.
stage_pairs = []
for chash, prev_kind in [
    (lineage.get("allocation", {}).get("backtest_hash"), "signal_leader"),
    (lineage.get("cost_sensitivity", {}).get("backtest_hash"), "allocation_leader"),
    (lineage.get("risk_overlay", {}).get("backtest_hash"), "cost_sensitivity_leader"),
]:
    if chash is None:
        continue
    p = load_paired_metrics(
        CASE_STUDY,
        challenger_hash=chash,
        benchmark_kind=prev_kind,
    )
    if p.is_empty():
        continue
    r = p.row(0, named=True)
    stage_pairs.append(
        {
            "transition": prev_kind.replace("_leader", "→") + chash[:8],
            "sharpe_diff": r["sharpe_diff"],
            "ci95_lo": r["sharpe_diff_ci95_lo"],
            "ci95_hi": r["sharpe_diff_ci95_hi"],
            "p_value": r["p_value"],
            "ci_status": ci_status(r["sharpe_diff_ci95_lo"], r["sharpe_diff_ci95_hi"]),
        }
    )

print("Stage-transition paired Sharpe deltas (challenger vs leader of previous stage):")
if stage_pairs:
    print(pl.DataFrame(stage_pairs))
else:
    print("  No stage-transition rows in backtest_paired_metrics for this CS.")

# %% [markdown]
# The lineage waterfall reads the rank-1 prediction's progression through
# the four pipeline stages with bootstrap CIs on each Sharpe. Allocation
# moves slightly *down* from the signal-stage rank-1 (top-K equal-weight
# vs score-weighted top-K), and the paired CI for the transition straddles
# zero — score-weighting is not statistically resolved as an improvement
# on this prediction set. The cost_sensitivity stage records a no-cost
# ablation as a separate row whose Sharpe sits well above the
# allocation-stage point estimate; the *paired* delta against the
# allocation leader excludes zero positively, but this is mechanically
# expected (zero costs lift any cross-sectional strategy with non-trivial
# turnover) and reflects the gross-return ceiling rather than a credit
# claim. The risk_overlay stage gives back nearly all of that ablation
# gain — its paired delta vs the cost_sensitivity leader excludes zero
# *negatively*, so the time-exit risk overlay is materially costly. Net
# of stage transitions, the rank-1 lineage's risk-overlay Sharpe sits
# slightly below the unadorned signal-stage Sharpe.

# %%
conc_df = explorer.concentration_curve(TOP_PHASH)
if not conc_df.is_empty():
    fig = plot_concentration_curve(conc_df)
    fig.show()
    best_per_k = conc_df.sort("sharpe", descending=True).group_by("top_k").first().sort("top_k")
    print("Allocation: best Sharpe by top_k:")
    print(best_per_k.select("top_k", "allocator", "sharpe", "max_drawdown"))
else:
    print("No concentration data — allocation stage absent for this prediction.")

# %% [markdown]
# Concentration on a ~3,200-stock universe is a different question than
# on a small panel: top_k = 20 (the rank-1 setting) leaves the long and
# short legs each holding 0.6% of the universe, which on cap-weighted
# breadth is heavily concentrated in micro-caps if no liquidity filter is
# applied. The setup.yaml `micro_cap_concentration` kill condition (>50%
# of alpha in the bottom ADV quintile) flags this risk but is not
# evaluated programmatically in the locked pipeline — §9 records this as
# a partial-evidence kill gate. Under the breadth-amplification logic of
# §1, larger top_k diversifies away the per-name idiosyncratic noise and
# reduces variance; the rank-1's 20-name setting trades breadth for
# magnitude.

# %% [markdown]
# ## §3 Headline performance with uncertainty
#
# The rank-1 specification is the validation-window backtest associated
# with the highest signal-stage Sharpe. Every metric is reported with
# its block-bootstrap 95% CI from `backtest_metrics`; the equity overlay
# shows the cumulative trajectory against the equal-weight US-equities
# universe.

# %%
full = load_backtest_metrics(CASE_STUDY, backtest_hash=TOP_HASH).row(0, named=True)
# When the spine's rank-1 (max-sharpe sibling) is not the cohort leader,
# load_backtest_metrics returns NULL selection-bias fields. Fall back to
# the family cohort row at (allocation, rank-1 label, rank-1 family) for
# DSR_ER/PBO/min_trl — these are properties of the cohort, not of the
# individual sibling.
if full.get("dsr") is None:
    # Find the spine rank-1's stage (signal | allocation | risk_overlay).
    with sqlite3.connect(str(_db)) as _con:
        _stage_for_cohort = _con.execute(
            "SELECT stage FROM backtest_runs WHERE backtest_hash = ?",
            (TOP_HASH,),
        ).fetchone()[0]
    with sqlite3.connect(str(_db)) as _con:
        _cohort = _con.execute(
            """
            SELECT dsr_er, dsr_er_pvalue, expected_max_sharpe_er,
                   min_trl_periods_er, pbo, pbo_n_combinations, pbo_n_folds,
                   k_variants, leader_hash, leader_sharpe
            FROM cohort_metrics
            WHERE cohort_type='family' AND stage=? AND label=? AND family=?
            """,
            (_stage_for_cohort, RANK1_LABEL, RANK1_FAMILY),
        ).fetchone()
    if _cohort is not None:
        full = dict(full)  # mutate
        (
            full["dsr"],
            full["dsr_pvalue"],
            full["expected_max_sharpe"],
            full["min_trl_periods"],
            full["pbo"],
            full["pbo_n_combinations"],
            full["pbo_n_folds"],
            full["k_variants"],
            _cohort_leader_hash,
            _cohort_leader_sharpe,
        ) = _cohort
        print(
            f"  cohort fallback: family cohort `{_stage_for_cohort}/"
            f"{RANK1_LABEL}/{RANK1_FAMILY}` "
            f"leader_hash={_cohort_leader_hash} leader_sharpe="
            f"{_cohort_leader_sharpe:.3f} (spine rank-1 sibling "
            f"differs by {full['sharpe'] - _cohort_leader_sharpe:+.3f})."
        )

spec_block = {
    "case_study": CASE_STUDY,
    "family": RANK1_FAMILY,
    "config_name": RANK1_CONFIG,
    "label": PRIMARY_LABEL,
    "signal_method": lineage["signal"].get("signal_method"),
    "top_k": lineage["signal"].get("top_k"),
    "allocation": lineage.get("allocation", {}).get("allocator"),
    "rebalance_step_days": setup["labels"]["rebalance_step"][PRIMARY_LABEL],
    "cost_assumption": "no costs at signal stage; sensitivity in §5",
    "risk_overlay": lineage.get("risk_overlay", {}).get("risk_name"),
    "validation_window_periods": int(full["n_periods"]) if full["n_periods"] is not None else None,
    "num_trades": int(full["num_trades"]) if full["num_trades"] is not None else None,
    "avg_turnover": full["avg_turnover"],
    "bootstrap_block_length": int(full["bootstrap_block_length"])
    if full["bootstrap_block_length"] is not None
    else None,
    "bootstrap_n": int(full["bootstrap_n"]) if full["bootstrap_n"] is not None else None,
}
print("Rank-1 specification (signal stage, validation window):")
for k, v in spec_block.items():
    print(f"  {k}: {v}")

_block = int(full["bootstrap_block_length"])
_rstep = setup["labels"]["rebalance_step"][PRIMARY_LABEL]
print(
    f"  audit: bootstrap_block_length={_block} day(s); "
    f"rebalance_step={_rstep} day(s) — both encode the autocorrelation "
    "horizon of the daily-rebalance fwd_ret_1d target."
)

# %%
sharpe_status = ci_status(full["sharpe_ci95_lo"], full["sharpe_ci95_hi"])
sortino_status = ci_status(full["sortino_ci95_lo"], full["sortino_ci95_hi"])
ann_status = ci_status(full["ann_return_ci95_lo"], full["ann_return_ci95_hi"])
mdd_status = ci_status(full["max_dd_ci95_lo"], full["max_dd_ci95_hi"])
calmar_status = ci_status(full["calmar_ci95_lo"], full["calmar_ci95_hi"])


def _hrow(
    metric: str,
    point: float | None,
    lo: float | None,
    hi: float | None,
    status: str,
    fmt: str = ".4f",
) -> dict:
    return {
        "metric": metric,
        "point": _fmt(point, fmt),
        "ci95_lo": _fmt(lo, fmt),
        "ci95_hi": _fmt(hi, fmt),
        "status": status,
    }


headline = pl.DataFrame(
    [
        _hrow(
            "Sharpe", full["sharpe"], full["sharpe_ci95_lo"], full["sharpe_ci95_hi"], sharpe_status
        ),
        _hrow(
            "Sortino",
            full["sortino"],
            full["sortino_ci95_lo"],
            full["sortino_ci95_hi"],
            sortino_status,
        ),
        _hrow(
            "Annualized return",
            full["cagr"],
            full["ann_return_ci95_lo"],
            full["ann_return_ci95_hi"],
            ann_status,
        ),
        _hrow(
            "Max drawdown",
            full["max_drawdown"],
            full["max_dd_ci95_lo"],
            full["max_dd_ci95_hi"],
            mdd_status,
        ),
        _hrow(
            "Calmar", full["calmar"], full["calmar_ci95_lo"], full["calmar_ci95_hi"], calmar_status
        ),
        {
            "metric": "PSR p-value (H0: SR≤0)",
            "point": _fmt(full["psr_pvalue"], ".2e"),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "DSR (selection-adjusted)",
            "point": _fmt(full["dsr"]),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "DSR p-value",
            "point": _fmt(full["dsr_pvalue"]),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "PBO",
            "point": _fmt(full["pbo"], ".3f"),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
    ]
)
print("Rank-1 headline metrics with 95% CIs:")
print(headline)
print()
print(
    "Note: DSR, expected-max-Sharpe, and MinTRL are surfaced from "
    "the rank-1's family `cohort_metrics` row (effective-rank "
    "K_eff; ER ≈ MP within 0.001 here). The selection_bias "
    "narrative_facts block records both ER and MP variants plus "
    "PBO. §9 records the selection-adjusted DSR for Ch20."
)

# %%
# Forest plot: rank-1 metrics with CI bars + reference lines
ew_val = load_benchmark_metrics(CASE_STUDY, PRIMARY_LABEL, period="validation")
ew_overall = load_benchmark_metrics(CASE_STUDY, PRIMARY_LABEL, period="overall")

forest_metrics = [
    ("Sharpe", full["sharpe"], full["sharpe_ci95_lo"], full["sharpe_ci95_hi"]),
    ("Sortino", full["sortino"], full["sortino_ci95_lo"], full["sortino_ci95_hi"]),
    ("Calmar", full["calmar"], full["calmar_ci95_lo"], full["calmar_ci95_hi"]),
    ("Ann. return", full["cagr"], full["ann_return_ci95_lo"], full["ann_return_ci95_hi"]),
]

fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(forest_metrics))
points = np.array([m[1] for m in forest_metrics])
los = np.array([m[2] for m in forest_metrics])
his = np.array([m[3] for m in forest_metrics])
# Clip to zero: bootstrap-resampled CI bounds can occasionally cross the
# realized point estimate (visible on Calmar where the realized
# point is below the lower 2.5% quantile of resampled draws); errorbar
# requires non-negative half-widths.
xerr_lo = np.maximum(points - los, 0.0)
xerr_hi = np.maximum(his - points, 0.0)
ax.errorbar(
    points,
    y,
    xerr=[xerr_lo, xerr_hi],
    fmt="o",
    color="#1565C0",
    ecolor="#5B9BD5",
    elinewidth=2.0,
    capsize=4,
    markersize=7,
)
ax.axvline(0, color="#9E9E9E", linestyle="--", linewidth=0.8)
ax.axvline(
    ew_val["sharpe"],
    color="#43A047",
    linestyle=":",
    linewidth=1.0,
    label=f"EW validation Sharpe ({ew_val['sharpe']:.2f})",
)
_alloc_sharpe = lineage.get("allocation", {}).get("sharpe")
if _alloc_sharpe is not None:
    ax.axvline(
        _alloc_sharpe,
        color="#E53935",
        linestyle=":",
        linewidth=1.0,
        label=f"Allocation rank-1 Sharpe ({_alloc_sharpe:.2f})",
    )
ax.set_yticks(y)
ax.set_yticklabels([m[0] for m in forest_metrics])
ax.invert_yaxis()
ax.set_xlabel("Value")
ax.set_title("Rank-1 headline metrics with 95% CIs")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.show()

# %%
# Equity-curve overlay vs validation EW benchmark
strat_returns_path = CASE_DIR / "run_log" / "backtest" / TOP_HASH / "daily_returns.parquet"
strat_df = (
    pl.read_parquet(strat_returns_path)
    .sort("timestamp")
    .with_columns(pl.col("timestamp").cast(pl.Date).alias("ts"))
    .select(pl.col("ts"), pl.col("daily_return").alias("strategy"))
)

bench_val = (
    load_benchmark_returns(CASE_STUDY, PRIMARY_LABEL, period="validation")
    .with_columns(pl.col("timestamp").cast(pl.Date).alias("ts"))
    .select(pl.col("ts"), pl.col("ew_return").alias("benchmark"))
)

aligned = strat_df.join(bench_val, on="ts", how="inner").sort("ts")
print(
    f"Validation overlay window: {aligned['ts'].min()} → {aligned['ts'].max()}, n={aligned.height}"
)

cum_strat = np.cumprod(1 + aligned["strategy"].to_numpy()) - 1
cum_bench = np.cumprod(1 + aligned["benchmark"].to_numpy()) - 1
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(aligned["ts"], cum_strat, color="#1565C0", linewidth=1.2, label="Rank-1 strategy")
ax.plot(aligned["ts"], cum_bench, color="#43A047", linewidth=1.2, label="EW US-equities universe")
ax.axhline(0, color="#9E9E9E", linewidth=0.6, linestyle="--")
ax.set_ylabel("Cumulative return")
ax.set_title("Validation cumulative return: rank-1 strategy vs EW universe")
ax.legend(loc="best", frameon=False)
fig.tight_layout()
fig.show()

# %% [markdown]
# The rank-1 Sharpe of 2.03 has a 95% CI of [1.46, 2.55] — the lower
# bound stays well above zero, so the validation window's Sharpe is not
# consistent with no edge under block-bootstrap resampling. The PSR
# p-value of ≈2 × 10⁻¹⁵ rejects H₀ (Sharpe ≤ 0) decisively under the
# point estimate's normal-approximation. The validation EW universe
# carried Sharpe 0.92 over the same window, so the rank-1 strategy
# outperforms the universe meaningfully on the point estimate; the §6
# paired test resolves whether that lead survives the holdout. The
# max-drawdown point of −44% sits inside its bootstrap CI [−56%, −29%]
# — the realized worst episode is consistent with the resampled
# distribution at this resolution. Cohort-level selection-adjusted
# metrics (`cohort_metrics`, ER-based) populate for this lineage's
# family cohort (`risk_overlay/fwd_ret_5d/gbm`, K_variants = 20 risk
# overlays on the score-weighted top_k=20 allocation): DSR_ER 0.106
# with p ≈ 0 (CI excludes zero strongly), MP-effective trials ≈ 2,
# PBO 0.0 across 12,870 CSCV combinations × 16 folds. The headline DSR
# exposed via `load_backtest_metrics` is the ER value; the MP value
# (0.112, within 0.006 of ER) is recorded alongside in
# `notebook_summaries.yaml::narrative_facts.selection_bias` for
# transparency.

# %% [markdown]
# ### Second benchmark: SPY
#
# The EW universe is the inside-the-strategy benchmark — what an
# untimed allocator across the same names would have earned. A reader
# also wants to know how the strategy reads against a passive index
# product. SPY is an investable proxy for the broad US equity market;
# it is not the strategy's universe (mid- and small-cap names that
# survive the price/ADV filter are not in the S&P 500), but it is the
# default external comparator. The SPY series begins 2006-01-04, so
# diagnostics here are computed on the SPY-available subset of the
# validation window.

# %%
spy_df = load_spy_returns(start=aligned["ts"].min(), end=aligned["ts"].max())
spy_aligned = align_external_benchmark(
    aligned.select("ts", "strategy"),
    spy_df,
    timestamp_col="ts",
)
spy_diag = compute_benchmark_diagnostics(
    spy_aligned["strategy"].to_numpy(),
    spy_aligned["benchmark_return"].to_numpy(),
    PERIODS_PER_YEAR,
)
ew_diag = compute_benchmark_diagnostics(
    aligned["strategy"].to_numpy(),
    aligned["benchmark"].to_numpy(),
    PERIODS_PER_YEAR,
)
benchmark_table = pl.DataFrame(
    [
        {
            "benchmark": "EW universe (validation)",
            "n_periods": ew_diag["n"],
            "info_ratio": _fmt(ew_diag["info_ratio"], ".3f"),
            "beta": _fmt(ew_diag["beta"], ".3f"),
            "correlation": _fmt(ew_diag["correlation"], ".3f"),
            "tracking_error": _fmt(ew_diag["tracking_error"], ".3f"),
        },
        {
            "benchmark": "SPY (2006-2015 subset)",
            "n_periods": spy_diag["n"],
            "info_ratio": _fmt(spy_diag["info_ratio"], ".3f"),
            "beta": _fmt(spy_diag["beta"], ".3f"),
            "correlation": _fmt(spy_diag["correlation"], ".3f"),
            "tracking_error": _fmt(spy_diag["tracking_error"], ".3f"),
        },
    ]
)
print("Strategy diagnostics vs benchmarks:")
print(benchmark_table)

# %% [markdown]
# ### Sub-period decomposition (5-year buckets)
#
# Pooled validation Sharpe averages over 16 calendar years; the
# realized number can be anchored by a small number of years rather
# than evenly distributed. The 5y buckets below decompose val + ho
# into four windows so non-stationarity is visible to the reader. We
# resolve the holdout backtest hash by matching `training_hash` against
# the validation rank-1 — the same convention §6 uses for the paired
# closure.

# %%
import datetime as _dt

with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as _con_sp:
    _ho_hash_sp = _con_sp.execute(
        """
        SELECT b.backtest_hash
        FROM backtest_runs b
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
        WHERE p.split = 'holdout'
          AND p.training_hash = (
              SELECT training_hash FROM prediction_sets WHERE prediction_hash = ?
          )
        ORDER BY bm.sharpe DESC LIMIT 1
        """,
        (top_signal.row(0, named=True)["prediction_hash"],),
    ).fetchone()
HO_HASH_SP = _ho_hash_sp[0] if _ho_hash_sp else None

ho_strat_path = (
    CASE_DIR / "run_log" / "backtest" / HO_HASH_SP / "daily_returns.parquet" if HO_HASH_SP else None
)
ho_panel = (
    pl.read_parquet(ho_strat_path)
    .sort("timestamp")
    .with_columns(pl.col("timestamp").cast(pl.Date).alias("ts"))
    .select("ts", pl.col("daily_return").alias("strategy"))
    if ho_strat_path is not None and ho_strat_path.exists()
    else pl.DataFrame()
)
bench_ho = (
    load_benchmark_returns(CASE_STUDY, PRIMARY_LABEL, period="holdout")
    .with_columns(pl.col("timestamp").cast(pl.Date).alias("ts"))
    .select("ts", pl.col("ew_return").alias("benchmark"))
)
ho_aligned = (
    ho_panel.join(bench_ho, on="ts", how="inner").sort("ts")
    if ho_panel.height > 0
    else pl.DataFrame()
)

val_buckets = [
    ("2000-2004 (val)", _dt.date(2000, 1, 1), _dt.date(2004, 12, 31)),
    ("2005-2009 (val)", _dt.date(2005, 1, 1), _dt.date(2009, 12, 31)),
    ("2010-2014 (val)", _dt.date(2010, 1, 1), _dt.date(2014, 12, 31)),
    ("2015      (val)", _dt.date(2015, 1, 1), _dt.date(2015, 12, 31)),
]
val_table = compute_subperiod_diagnostics(
    aligned,
    val_buckets,
    periods_per_year=PERIODS_PER_YEAR,
)
ho_table = (
    compute_subperiod_diagnostics(
        ho_aligned,
        [("2016-2018 (ho)", _dt.date(2016, 1, 1), _dt.date(2018, 12, 31))],
        periods_per_year=PERIODS_PER_YEAR,
    )
    if ho_aligned.height > 0
    else pl.DataFrame()
)
subperiod_table = pl.concat([val_table, ho_table]) if ho_table.height > 0 else val_table
print("Sub-period decomposition (validation 5y buckets + holdout):")
print(subperiod_table)

# %% [markdown]
# ## §4 Risk and drawdown analysis
#
# Risk metrics use the validation-window strategy returns paired against
# the validation EW benchmark. The drawdown panel surfaces the worst
# episode and recovery; rolling Sharpe and rolling beta locate when the
# strategy decoupled from the universe. For a long-short broad-panel
# equity strategy, drawdowns are typically driven by factor crowding,
# momentum reversals, or short-squeeze events.

# %%
strat_arr = aligned["strategy"].to_numpy()
bench_arr = aligned["benchmark"].to_numpy()
ts_arr = aligned["ts"].to_list()

pa = PortfolioAnalysis(
    returns=strat_arr,
    benchmark=bench_arr,
    dates=ts_arr,
    periods_per_year=PERIODS_PER_YEAR,
)

dd = pa.compute_drawdown_analysis()
print("Drawdown analysis (validation window):")
print(dd)

# %%
fig = plot_equity_drawdown(strat_returns_path)
fig.show()

# %%
roll = pa.compute_rolling_metrics(windows=[126], metrics=["sharpe", "beta"])
print(
    "Rolling-window keys (126-day):",
    {k: type(v).__name__ for k, v in roll.items()} if isinstance(roll, dict) else roll,
)

tail_table = pl.DataFrame(
    [
        {"metric": "Volatility (ann.)", "value": f"{full['volatility']:.4f}"},
        {"metric": "VaR 95% (daily)", "value": f"{full['var_95']:.4f}"},
        {"metric": "CVaR 95% (daily)", "value": f"{full['cvar_95']:.4f}"},
        {"metric": "Tail ratio", "value": f"{full['tail_ratio']:.3f}"},
        {"metric": "Skewness", "value": f"{full['skewness']:.3f}"},
        {"metric": "Kurtosis", "value": f"{full['kurtosis']:.3f}"},
    ]
)
print()
print("Tail risk profile (validation):")
print(tail_table)

# %%
fold_df = load_backtest_fold_metrics(CASE_STUDY, backtest_hash=TOP_HASH)
print(f"Per-fold breakdown ({fold_df.height} folds):")
print(fold_df.select("fold_id", "sharpe", "max_drawdown", "n_days"))
print()
print(f"Fold Sharpe range: [{fold_df['sharpe'].min():.3f}, {fold_df['sharpe'].max():.3f}]")
print(f"Fold Sharpe std:   {fold_df['sharpe'].std():.3f}")
print(f"Folds positive: {(fold_df['sharpe'] > 0).sum()}/{fold_df.height}")

# %% [markdown]
# Per-fold Sharpes show pronounced regime dependence — a high-Sharpe
# tail in the later folds (the 2010s drift period) and several deeply
# negative folds in the early walk-forward windows (around the 2008
# crisis and the 2011 Eurozone episode). The fold-Sharpe std is large
# enough that the single-realized validation Sharpe of 2.03 is a
# weighted average over heterogeneous regimes, not a stable across-fold
# edge. The realized max drawdown of −44% sits inside the resampled
# CI's interval — the worst episode is consistent with the bootstrap
# distribution rather than a tail outlier. §6's holdout test asks
# whether the high-Sharpe regime persists.

# %% [markdown]
# ## §4b Inline diagnostic panels
#
# §8 generates the full tear sheet as a standalone HTML artifact (the
# Plotly Dash design renders best as a multi-tab app). For inline review
# we lift its core panels directly into the notebook using the same
# diagnostic library: returns/risk views off the validation
# `PortfolioAnalysis`, and ML views off a `BacktestProfile` constructed
# from the rank-1 lineage's on-disk predictions and trades. The
# prediction-trade alignment view requires explicit dtype normalization
# of the predictions timestamp before enrichment (the on-disk parquet is
# millisecond resolution while trades are microsecond — the library's
# auto-enrichment hits the dtype mismatch).

# %%
_inline_backtest_dir = CASE_DIR / "run_log" / "backtest" / TOP_HASH
_inline_pred_hash = top_signal.row(0, named=True)["prediction_hash"]
profile_inline = profile_from_run_artifacts(
    _inline_backtest_dir,
    predictions_path=CASE_DIR
    / "run_log"
    / "predictions"
    / _inline_pred_hash
    / "predictions.parquet",
    calendar=setup["evaluation"]["calendar"],
)
_preds_us = profile_inline.predictions_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
_asset_col = "symbol" if "symbol" in _preds_us.columns else "asset"
profile_inline._prediction_trades_cache = enrich_trades_with_signals(
    trades_df=profile_inline.trades_df,
    signals_df=_preds_us,
    timestamp_col="timestamp",
    asset_col=_asset_col,
)
print(
    f"Profile built: predictions={profile_inline.has_predictions}, "
    f"enriched_trades={profile_inline._prediction_trades_cache.height:,}"
)

# %% [markdown]
# **Cumulative return.** Validation-window equity vs. EW universe.

# %%
plot_cumulative_returns(pa, benchmark_label="EW universe").show()

# %% [markdown]
# **Annual returns.** Calendar-year strategy vs. benchmark bars.

# %%
plot_annual_returns_bar(pa, benchmark_label="EW universe").show()

# %% [markdown]
# **Monthly returns heatmap.** Year×month return surface; banded years
# locate concentrated episodes.

# %%
plot_monthly_returns_heatmap(pa).show()

# %% [markdown]
# **Returns distribution.** Daily return histogram with a normal overlay
# for visual deviation; complements §4's tail-ratio table.

# %%
plot_returns_distribution(pa).show()

# %% [markdown]
# **Drawdown underwater.** The library's underwater curve over the same
# strategy series; reads against the §4 max-drawdown number.

# %%
plot_drawdown_underwater(pa).show()

# %% [markdown]
# **Rolling Sharpe.** 126- and 252-day rolling Sharpe locate when the
# realized validation Sharpe is paid out vs. quiet stretches.

# %%
plot_rolling_sharpe(pa, windows=[126, 252]).show()

# %% [markdown]
# **Tail risk panel.** VaR/CVaR at 95/99% with the empirical tail.

# %%
plot_tail_risk_analysis(strat_arr).show()

# %% [markdown]
# **IC time series.** Daily cross-sectional rank-IC of the rank-1 model;
# the panel exposes ic-stationarity beyond the §3 daily-pooled headline.

# %%
plot_ic_time_series(profile_inline).show()

# %% [markdown]
# **Decile returns.** Forward-return spread across prediction deciles —
# the breadth signal the FLOAM relies on.

# %%
plot_quintile_returns(profile_inline, n_quantiles=10).show()

# %% [markdown]
# **Prediction–trade alignment.** Distribution of `entry_prediction_value`
# split by trade direction; verifies the backtester acted on the
# model's score (and not on a stale or shuffled feed).

# %%
plot_prediction_trade_alignment(profile_inline).show()

# %% [markdown]
# ## §5 Friction budget & cost sensitivity
#
# Broad-panel US equities span two cost regimes: pre-decimalization (before
# 2001-01-29; tick size 1/16, 15–30 bps/leg) and post-decimalization
# (penny tick, 5–15 bps/leg). The cost_sensitivity stage walked an 11-point
# bps grid (0 to 50 bps total per-leg cost) on the top allocation combos;
# a parallel exploratory sweep walked a 6-point half-spread grid (0 to 10¢
# per share at $0.0035 IBKR Pro commission) so the curve can be read in
# nominal-dollar terms as well. The two regimes are panelled side by side:
# the bps panel is the headline (its era-aware bands map directly to
# `setup.yaml.costs.era_dependent`), the per-share panel is presented for
# regime comparison since basis points and dollars-per-share use different
# axes and disagree most for low-priced and pre-decimalization names — both
# heavily represented in this universe. The 50 bps/yr borrow cost on the
# short leg adds an annualized headwind that does not appear on either
# axis but is folded into the §9 operating profile.

# %%
with sqlite3.connect(str(_db)) as _con:
    cost_df = pl.DataFrame(
        _con.execute(
            """
            SELECT
                b.spec_json,
                bm.sharpe,
                bm.sharpe_ci95_lo,
                bm.sharpe_ci95_hi,
                bm.max_drawdown
            FROM backtest_runs b
            JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            JOIN prediction_sets p   ON b.prediction_hash = p.prediction_hash
            WHERE b.stage = 'cost_sensitivity'
              AND p.split = 'validation'
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
            """
        ).fetchall(),
        schema=["spec_json", "sharpe", "sharpe_ci95_lo", "sharpe_ci95_hi", "max_drawdown"],
        orient="row",
    )


def _regime_and_cost(spec_str: str) -> tuple[str, float]:
    spec = json.loads(spec_str)
    bc = spec.get("backtest_config", {})
    comm = bc.get("commission", {}) or {}
    slip = bc.get("slippage", {}) or {}
    model = comm.get("model", "percentage")
    if model == "percentage":
        return "bps", float((comm.get("rate", 0) + slip.get("rate", 0)) * 10000)
    if model == "per_share":
        return "ps", float(slip.get("spread", 0.0))
    return "other", 0.0


_regimes = [_regime_and_cost(s) for s in cost_df["spec_json"].to_list()]
cost_df = cost_df.with_columns(
    regime=pl.Series([r for r, _ in _regimes]),
    cost_value=pl.Series([c for _, c in _regimes]),
)


def _curve(df: pl.DataFrame, regime: str) -> pl.DataFrame:
    return (
        df.filter(pl.col("regime") == regime)
        .group_by("cost_value")
        .agg(
            sharpe_max=pl.col("sharpe").max(),
            sharpe_min=pl.col("sharpe").min(),
            sharpe_median=pl.col("sharpe").median(),
            sharpe_ci_lo=pl.col("sharpe_ci95_lo").min(),
            sharpe_ci_hi=pl.col("sharpe_ci95_hi").max(),
            n=pl.len(),
        )
        .sort("cost_value")
    )


curve_bps = _curve(cost_df, "bps")
curve_ps = _curve(cost_df, "ps")
print("Bps regime cost curve (validation):")
print(curve_bps)
print()
print("Per-share regime cost curve (validation):")
print(curve_ps)

# %%
post_lo, post_hi = setup["costs"]["era_dependent"]["post_decimalization"]["per_leg_cost_bps_range"]
pre_lo, pre_hi = setup["costs"]["era_dependent"]["pre_decimalization"]["per_leg_cost_bps_range"]

fig, (ax_bps, ax_ps) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)


def _plot_curve(ax, curve, x_label, x_unit_xform=lambda v: v, accent_bands=()):
    if curve.is_empty():
        ax.text(0.5, 0.5, "No rows", ha="center", va="center", transform=ax.transAxes)
        return
    xs = [x_unit_xform(v) for v in curve["cost_value"].to_list()]
    ax.fill_between(
        xs,
        curve["sharpe_ci_lo"].to_numpy(),
        curve["sharpe_ci_hi"].to_numpy(),
        alpha=0.18,
        color="#5B9BD5",
        label="best–worst CI envelope across configs",
    )
    ax.plot(
        xs, curve["sharpe_median"].to_numpy(), color="#1565C0", linewidth=1.4, label="median Sharpe"
    )
    ax.plot(
        xs,
        curve["sharpe_max"].to_numpy(),
        color="#43A047",
        linewidth=1.0,
        linestyle="--",
        label="best-config Sharpe",
    )
    ax.axhline(0, color="#9E9E9E", linewidth=0.8, linestyle="--")
    for lo, hi, color, lbl in accent_bands:
        ax.axvspan(lo, hi, color=color, alpha=0.10, label=lbl)
    ax.set_xlabel(x_label)
    ax.legend(loc="best", fontsize=7, frameon=False)


_plot_curve(
    ax_bps,
    curve_bps,
    "Per-leg cost (bps)",
    accent_bands=[
        (post_lo, post_hi, "#43A047", f"post-decimalization ({post_lo}–{post_hi} bps/leg)"),
        (pre_lo, pre_hi, "#FB8C00", f"pre-decimalization ({pre_lo}–{pre_hi} bps/leg)"),
    ],
)
ax_bps.set_ylabel("Sharpe (validation)")
ax_bps.set_title("Bps regime (headline)")
_plot_curve(
    ax_ps,
    curve_ps,
    "Half-spread (¢/share, +$0.0035/sh commission)",
    x_unit_xform=lambda v: v * 100,
)
ax_ps.set_title("Per-share + spread regime (exploratory)")
fig.suptitle("Cost sensitivity — us_equities_panel (validation, top allocation combos)")
fig.tight_layout()
fig.show()


# %%
def _breakeven(curve: pl.DataFrame, label: str, units: str) -> None:
    if curve.is_empty():
        print(f"  [{label}] no rows in registry")
        return
    crossing = curve.filter(pl.col("sharpe_ci_lo") > 0)
    if crossing.is_empty():
        print(f"  [{label}] Sharpe CI lower bound never exceeds 0 across the grid")
    else:
        print(
            f"  [{label}] Sharpe CI lower bound stays > 0 up to: "
            f"{crossing['cost_value'].max():.4f} {units}"
        )


print("Breakeven by regime:")
_breakeven(curve_bps, "bps", "bps/leg")
_breakeven(curve_ps, "ps", "$/share half-spread")
print()
print("Realistic friction (per setup.yaml):")
print(f"  Pre-decimalization (before 2001-01-29): {pre_lo}–{pre_hi} bps/leg")
print(f"  Post-decimalization (after 2001-01-29): {post_lo}–{post_hi} bps/leg")
print(f"  Borrow cost on short leg: {setup['costs']['borrow_cost_note']}")
print("See Chapter 18 for the transaction-cost framework.")

# %% [markdown]
# Sharpe degrades roughly monotonically with cost. The best-config
# Sharpe stays above zero across the full grid swept, but the
# best–worst envelope shows substantial across-config dispersion at
# every cost level — the binding constraint is signal stability, not
# cost survival. At post-decimalization friction (5–15 bps/leg), the
# best-config validation Sharpe still excludes zero on the lower bound;
# at pre-decimalization friction (15–30 bps/leg), the same configs sit
# below the realistic-cost band but the median config drops sharply.
# The 50 bps/yr borrow cost on the short leg subtracts ≈0.5 percentage
# points of annual return per unit gross exposure, which on the
# rank-1's volatility of ≈29% translates into a Sharpe headwind of
# ≈0.02 — small relative to the cross-config dispersion. Chapter 18
# develops the transaction-cost framework.

# %% [markdown]
# ## §6 Holdout closure with paired bootstrap
#
# Two paired tests anchor the holdout read: (i) the holdout rank-1 versus
# the validation rank-1 ("did Sharpe carry?"), and (ii) the holdout
# rank-1 versus the holdout-window equal-weight benchmark ("did the
# strategy beat the universe in the holdout?"). Both numbers come from
# `backtest_paired_metrics` — never from val − holdout point arithmetic.

# %%
with sqlite3.connect(str(_db)) as _con:
    HO_HASH = _con.execute(
        """
        SELECT b.backtest_hash
        FROM backtest_runs b
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
        WHERE p.split = 'holdout'
          AND p.training_hash = (
              SELECT training_hash FROM prediction_sets WHERE prediction_hash = ?
          )
        ORDER BY bm.sharpe DESC
        LIMIT 1
        """,
        (TOP_PHASH,),
    ).fetchone()[0]

print(f"Validation rank-1 hash: {TOP_HASH}")
print(f"Holdout rank-1 hash:    {HO_HASH}")

ho_full = load_backtest_metrics(CASE_STUDY, backtest_hash=HO_HASH).row(0, named=True)
val_full = full

# %%
val_ho_pair = load_paired_metrics(
    CASE_STUDY,
    challenger_hash=HO_HASH,
    benchmark_kind="val_rank1_self",
)
if val_ho_pair.is_empty():
    print(
        "[WARN] Missing val_rank1_self pair for us_equities_panel — populator "
        "skipped (holdout has no trades or insufficient overlap with val "
        "lineage). Continuing with NaN val→holdout decay."
    )
    vh = {
        "sharpe_diff": float("nan"),
        "sharpe_diff_ci95_lo": float("nan"),
        "sharpe_diff_ci95_hi": float("nan"),
        "ret_diff": float("nan"),
        "ret_diff_ci95_lo": float("nan"),
        "ret_diff_ci95_hi": float("nan"),
        "max_dd_diff": float("nan"),
        "max_dd_diff_ci95_lo": float("nan"),
        "max_dd_diff_ci95_hi": float("nan"),
        "p_value": float("nan"),
        "prob_challenger_wins": float("nan"),
        "info_ratio": float("nan"),
        "info_ratio_ci95_lo": float("nan"),
        "info_ratio_ci95_hi": float("nan"),
    }
else:
    vh = val_ho_pair.row(0, named=True)


def _diff_row(
    label: str,
    v: float | None,
    h: float | None,
    diff: float | None,
    lo: float | None,
    hi: float | None,
    p: float | None,
) -> dict:
    return {
        "metric": label,
        "validation": _fmt(v, ".4f"),
        "holdout": _fmt(h, ".4f"),
        "diff (h−v)": _fmt(diff, ".4f") if diff is not None else "—",
        "diff CI95": (f"[{_fmt(lo, '.4f')}, {_fmt(hi, '.4f')}]" if lo is not None else "—"),
        "p-value": _fmt(p, ".4f") if p is not None else "—",
    }


val_ho_table = pl.DataFrame(
    [
        _diff_row(
            "Sharpe",
            val_full["sharpe"],
            ho_full["sharpe"],
            vh["sharpe_diff"],
            vh["sharpe_diff_ci95_lo"],
            vh["sharpe_diff_ci95_hi"],
            vh["p_value"],
        ),
        _diff_row(
            "Annualized return",
            val_full["cagr"],
            ho_full["cagr"],
            vh["ret_diff"],
            vh["ret_diff_ci95_lo"],
            vh["ret_diff_ci95_hi"],
            None,
        ),
        _diff_row(
            "Max drawdown",
            val_full["max_drawdown"],
            ho_full["max_drawdown"],
            vh["max_dd_diff"],
            vh["max_dd_diff_ci95_lo"],
            vh["max_dd_diff_ci95_hi"],
            None,
        ),
        _diff_row(
            "Information ratio",
            None,
            None,
            vh["info_ratio"],
            vh["info_ratio_ci95_lo"],
            vh["info_ratio_ci95_hi"],
            None,
        ),
    ]
)
print("val → holdout paired-bootstrap decay (rank-1 self):")
print(val_ho_table)
print(f"prob_challenger_wins (holdout): {vh['prob_challenger_wins']:.3f}")
print(f"CI status (Sharpe diff): {ci_status(vh['sharpe_diff_ci95_lo'], vh['sharpe_diff_ci95_hi'])}")
print()
print(
    "Note: validation and holdout windows are disjoint by design — the "
    "populator pairs the bootstrapped Sharpe series by row index after a "
    "min-length truncation. The CI is interpreted as bootstrap-resampled "
    "Sharpe difference under index-paired draws, not as a calendar-aligned "
    "overlap."
)

# %%
ho_vs_ew = load_paired_metrics(
    CASE_STUDY,
    challenger_hash=HO_HASH,
    benchmark_kind="equal_weight_holdout_side_artifact",
)
if ho_vs_ew.is_empty():
    # Populator skipped this pair because the holdout backtest had zero
    # trades (or all-zero returns), so the paired bootstrap on
    # `strategy − EW_holdout` is undefined. Surface a placeholder so the
    # rest of the section renders with clear "no trades" context.
    print(
        "[WARN] Missing equal_weight_holdout_side_artifact pair for "
        "us_equities_panel — holdout has no trades or all-zero returns; "
        "paired bootstrap not computable. Continuing with NaN diffs."
    )
    he = {
        "sharpe_diff": float("nan"),
        "sharpe_diff_ci95_lo": float("nan"),
        "sharpe_diff_ci95_hi": float("nan"),
        "ret_diff": float("nan"),
        "ret_diff_ci95_lo": float("nan"),
        "ret_diff_ci95_hi": float("nan"),
        "max_dd_diff": float("nan"),
        "max_dd_diff_ci95_lo": float("nan"),
        "max_dd_diff_ci95_hi": float("nan"),
        "p_value": float("nan"),
        "prob_challenger_wins": float("nan"),
        "info_ratio": float("nan"),
        "info_ratio_ci95_lo": float("nan"),
        "info_ratio_ci95_hi": float("nan"),
    }
else:
    he = ho_vs_ew.row(0, named=True)

val_vs_ew = load_paired_metrics(
    CASE_STUDY,
    challenger_hash=TOP_HASH,
    benchmark_kind="equal_weight_side_artifact",
)
ve = val_vs_ew.row(0, named=True) if not val_vs_ew.is_empty() else None

ew_ho = load_benchmark_metrics(CASE_STUDY, PRIMARY_LABEL, period="holdout")
print("Strategy vs equal-weight universe (paired bootstrap):")
print(
    f"  validation: strategy Sharpe {val_full['sharpe']:.3f} vs EW "
    f"{ew_val['sharpe']:.3f}; "
    f"diff = "
    + (
        _fmt_ci(ve["sharpe_diff"], ve["sharpe_diff_ci95_lo"], ve["sharpe_diff_ci95_hi"])
        if ve
        else "—"
    )
    + (f"  p={ve['p_value']:.4f}  " if ve else "  ")
    + (f"({ci_status(ve['sharpe_diff_ci95_lo'], ve['sharpe_diff_ci95_hi'])})" if ve else "")
)
print(
    f"  holdout:    strategy Sharpe {ho_full['sharpe']:.3f} vs EW "
    f"{ew_ho['sharpe']:.3f}; "
    f"diff = "
    f"{_fmt_ci(he['sharpe_diff'], he['sharpe_diff_ci95_lo'], he['sharpe_diff_ci95_hi'])}"
    f"  p={he['p_value']:.4f}  "
    f"({ci_status(he['sharpe_diff_ci95_lo'], he['sharpe_diff_ci95_hi'])})"
)
print(
    f"  holdout info_ratio (strategy vs EW): "
    f"{_fmt_ci(he['info_ratio'], he['info_ratio_ci95_lo'], he['info_ratio_ci95_hi'])}"
)

# %% [markdown]
# **Decay reading (val_rank1_self pair).** Holdout Sharpe of −0.49 sits
# well below the validation Sharpe of +2.03; the index-paired diff CI
# excludes zero on the negative side with a small p-value, so the
# decay is statistically resolved as material. The validation rank-1
# was **not** a faithful representation of the strategy's edge in the
# holdout regime; either the validation window over-represents favorable
# factor environments, or the rank-1 selection was lifted by selection
# bias from the cross-stage sweep, or both. The decay enters Ch20's
# aggregator as `excludes_zero_strong`.
#
# **Strategy vs EW benchmark on holdout.** The holdout EW Sharpe of
# +1.82 is unusually high (the 2016-Q1–2018-Q1 window includes a
# strong cap-weighted bull run); the rank-1 strategy's holdout Sharpe
# of −0.49 sits well below it. The paired diff CI of [−4.52, −0.07]
# excludes zero negatively at the 5% level (p ≈ 0.04), so the strategy
# did materially under-perform EW on the holdout. In the *validation*
# window the same paired test gives a diff Sharpe of +1.11
# [+0.49, +1.72] (excludes zero positively, p ≈ 0.001), so the
# strategy did beat EW in the training regime — the holdout regime
# both erased that edge in point-estimate terms and pushed the paired
# CI to the negative side.

# %% [markdown]
# ## §7 Benchmark-aware diagnostics
#
# Layer 1 reports the universal alpha/beta/IR profile via
# `PortfolioAnalysis` against the equal-weight US-equities universe.
# Layer 2 — equity factor attribution (FF5+MOM) — is **applicable**:
# the strategy trades a long-short cross-section of US equities, so
# Fama-French factors are the natural risk model. We regress raw daily
# returns (not excess) since the dollar-neutral construction cancels the
# risk-free rate.

# %%
metrics = pa.compute_summary_stats()
attr_df = pl.DataFrame(
    [
        {"metric": "alpha (annualized)", "value": _fmt(getattr(metrics, "alpha", None))},
        {"metric": "beta", "value": _fmt(getattr(metrics, "beta", None), ".3f")},
        {"metric": "information ratio", "value": _fmt(getattr(metrics, "information_ratio", None))},
        {"metric": "tracking error", "value": _fmt(getattr(metrics, "tracking_error", None))},
        {"metric": "up capture", "value": _fmt(getattr(metrics, "up_capture", None))},
        {"metric": "down capture", "value": _fmt(getattr(metrics, "down_capture", None))},
    ]
)
print("Layer 1: rank-1 vs validation EW universe (PortfolioAnalysis):")
print(attr_df)

# %%
# Placebo regression: residual α and HAC t-stat from regressing on EW alone.
X = sm.add_constant(bench_arr)
ols = sm.OLS(strat_arr, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
alpha_daily_ew = ols.params[0]
alpha_t_ew = ols.tvalues[0]
alpha_p_ew = ols.pvalues[0]
beta_to_ew = ols.params[1]
print()
print("Placebo regression (strategy on EW universe, HAC, maxlags=5):")
print(
    f"  α (daily) = {alpha_daily_ew:.6f}, α (annualized) = {alpha_daily_ew * PERIODS_PER_YEAR:.4f}"
)
print(f"  α t-stat = {alpha_t_ew:.3f}, p = {alpha_p_ew:.4f}")
print(f"  β to EW  = {beta_to_ew:.3f}")
print(f"  CI status (α): {'excludes_zero_strong' if alpha_p_ew < 0.05 else 'straddles_zero'}")

# %% [markdown]
# **Layer 2: FF5+MOM factor attribution.**

# %%
strategy_rets = pd.Series(
    aligned["strategy"].to_numpy(),
    index=pd.to_datetime(aligned["ts"].to_list()),
    name="strategy",
)
factor_start = str(strategy_rets.index.min().date())
factor_end = str(strategy_rets.index.max().date())
factors = load_factor_data(start=factor_start, end=factor_end)

reg = run_factor_regression(
    strategy_rets, factors, model="ff5_mom", hac_lags=5, dollar_neutral=True
)

print("=== FF5+MOM regression (validation, HAC, dollar_neutral=True) ===")
print(f"  Observations:    {reg['n_obs']}")
print(f"  Strategy Sharpe: {reg['strategy_sharpe']:+.3f}")
print(f"  Residual Sharpe: {reg['residual_sharpe']:+.3f}")
print(f"  R²:              {reg['r_squared']:.3f}")
print(
    f"  Alpha (ann.):    {reg['alpha_annualized']:+.1%} "
    f"(t={reg['alpha_t_stat']:.2f}, p={reg['alpha_p_value']:.3f})"
)
print()
print("Factor betas (HAC):")
for factor, beta in reg["betas"].items():
    t = reg["t_stats"][factor]
    sig = "*" if reg["p_values"][factor] < 0.05 else ""
    print(f"  {factor:8s}: {beta:+.4f}  (t={t:+.2f}){sig}")

# %%
rolling = compute_rolling_exposures(
    strategy_rets, factors, model="ff5_mom", window=63, dollar_neutral=True
)
fig_roll = plot_rolling_exposures(
    rolling, title="US Equities Panel: rolling factor exposures (63-day)"
)
fig_roll.show()

# %% [markdown]
# Rolling exposures locate when the strategy's factor loadings shifted —
# stable across-fold loadings indicate a static implicit selection rule
# (e.g., persistent SMB tilt), while large swings indicate a regime-
# adaptive selector. For a long-short ranker on cross-sectional features,
# the most informative reads are the SMB and MOM exposures: the SMB
# loading tells whether the model concentrates in the small-cap segment
# (where features are noisier but selection signal is stronger) and the
# MOM loading tells whether the model has reverse-engineered a momentum
# signal from the financial features.

# %%
# Placebo: random dollar-neutral US-equities portfolios over the same window.
from data import load_us_equities

eq_data = load_us_equities(start_date=factor_start, end_date=factor_end)
eq_daily = (
    eq_data.sort("symbol", "timestamp")
    .with_columns(ret=pl.col("close").pct_change().over("symbol"))
    .drop_nulls("ret")
    .with_columns(date=pl.col("timestamp").dt.date())
    .group_by("symbol", "date")
    .agg(daily_ret=pl.col("ret").last())
    .sort("symbol", "date")
)
eq_wide = (
    eq_daily.pivot(on="symbol", index="date", values="daily_ret").to_pandas().set_index("date")
)
eq_wide.index = pd.to_datetime(eq_wide.index)

placebo = run_placebo_benchmark(
    eq_wide,
    factors,
    n_sims=300,
    top_k=20,
    model="ff5_mom",
    dollar_neutral=True,
    seed=42,
)
if placebo["n_sims"] > 0:
    print(f"=== Placebo benchmark ({placebo['n_sims']} random L/S top-20 portfolios) ===")
    print(
        f"  Random Mkt-RF β: {placebo['Mkt-RF_mean']:+.3f} ± {placebo['Mkt-RF_std']:.3f}  "
        f"[90% CI: {placebo['Mkt-RF_p5']:+.3f}, {placebo['Mkt-RF_p95']:+.3f}]"
    )
    print(
        f"  Random SMB β:    {placebo['SMB_mean']:+.3f} ± {placebo['SMB_std']:.3f}  "
        f"[90% CI: {placebo['SMB_p5']:+.3f}, {placebo['SMB_p95']:+.3f}]"
    )
    print(
        f"  Random HML β:    {placebo['HML_mean']:+.3f} ± {placebo['HML_std']:.3f}  "
        f"[90% CI: {placebo['HML_p5']:+.3f}, {placebo['HML_p95']:+.3f}]"
    )
    print(f"  Random α (ann.): {placebo['alpha_ann_mean']:+.2%} ± {placebo['alpha_ann_std']:.2%}")
    print(f"  Random R²:       {placebo['r_squared_mean']:.3f}")
    print()
    for factor in ["Mkt-RF", "SMB", "HML"]:
        strat_beta = reg["betas"].get(factor, 0)
        p5 = placebo[f"{factor}_p5"]
        p95 = placebo[f"{factor}_p95"]
        inside = "within" if p5 <= strat_beta <= p95 else "OUTSIDE"
        print(
            f"  Strategy {factor}: {strat_beta:+.3f} -- {inside} random 90% CI "
            f"[{p5:+.3f}, {p95:+.3f}]"
        )

# %%
# Block-bootstrap CIs on alpha and factor betas
boot = compute_bootstrap_ci(
    strategy_rets,
    factors,
    model="ff5_mom",
    n_boot=500,
    block_size=20,
    dollar_neutral=True,
    seed=42,
)
if boot.get("n_boot", 0) > 0:
    print(f"=== Bootstrap CIs (n={boot['n_boot']}, block=20 days) ===")
    print(f"  α (annualized): [{boot['alpha_ann_lo']:+.2%}, {boot['alpha_ann_hi']:+.2%}]")
    for factor in reg["factor_columns"]:
        key_lo, key_hi = f"{factor}_lo", f"{factor}_hi"
        if key_lo in boot:
            print(f"  {factor:8s}: [{boot[key_lo]:+.4f}, {boot[key_hi]:+.4f}]")

# %%
fig_attr = plot_attribution_waterfall(reg, title="US Equities Panel: factor attribution")
fig_attr.show()

attr_summary = format_attribution_summary(reg, boot)

# %% [markdown]
# **Factor attribution interpretation.** The R² and residual Sharpe
# locate how much of the strategy's validation Sharpe is explained by
# standard factor exposure versus residual selection skill. Significant
# loadings on SMB or MOM would indicate the GBM has implicitly learned
# a small-cap or trend-following lens through the cross-sectional
# features; an alpha t-stat that survives HAC adjustment and a
# residual Sharpe well above zero would point to genuine selection
# beyond factor replication. The placebo benchmark sets the
# universe-driven baseline: random dollar-neutral 20-name long-short
# portfolios from the same panel typically produce near-zero factor
# loadings (the long and short legs cancel), so any *systematic*
# factor exposure in the strategy is attributable to the model's
# stock selection, not to universe composition.

# %% [markdown]
# ## §8 Strategy tear sheet
#
# The diagnostic library renders the rank-1 lineage's full tear sheet
# from on-disk artifacts; we wire the validation-window equal-weight
# universe in as the benchmark series and pass FF5 factor data so the
# tear sheet's factor-attribution panel is populated. The tear sheet
# HTML is written under `OUTPUT_DIR` (gitignored — readers regenerate
# locally). Per the per-CS deviations (Appendix), this case study uses
# `template="full"` rather than `risk_manager`.

# %%
backtest_dir = CASE_DIR / "run_log" / "backtest" / TOP_HASH
ho_dir = CASE_DIR / "run_log" / "backtest" / HO_HASH
ho_has_trades = (ho_dir / "trades.parquet").exists()
print(f"Validation backtest_dir: {backtest_dir} (trades.parquet present)")
print(f"Holdout backtest_dir:    {ho_dir} (trades.parquet present: {ho_has_trades})")
print(
    "Tear sheet sourced from validation backtest; the strategy-analysis convention "
    "renders tear sheets from the validation lineage."
)

bench_aligned = (
    bench_val.rename({"benchmark": "ew_return"})
    .with_columns(pl.col("ts").cast(pl.Datetime("ms")).alias("timestamp"))
    .select("timestamp", "ew_return")
)
bench_series = bench_aligned["ew_return"].to_numpy()

# Build FactorData for tear-sheet factor panel.
from data import load_ff_factors

ff5 = load_ff_factors(dataset="ff5", frequency="daily")
factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
factor_data = FactorData(
    returns=ff5.select(["timestamp"] + factor_names),
    rf_rate=ff5.get_column("RF"),
    factor_names=factor_names,
    source="fama-french",
    frequency="daily",
)

meta = BacktestReportMetadata(
    title="US Equities Panel — Rank-1 Lineage",
    strategy_name=f"{RANK1_FAMILY}/{RANK1_CONFIG} — {PRIMARY_LABEL}",
    universe="~3,200 US equities (NYSE/NASDAQ/AMEX, ADV>$1M, price>$5)",
    benchmark_name="US-equities equal-weight universe (validation window)",
    evaluation_window=f"{aligned['ts'].min()} to {aligned['ts'].max()}",
    calendar=setup["evaluation"]["calendar"],
)

tear_path = OUTPUT_DIR / f"{CASE_STUDY}_tearsheet.html"
profile = profile_from_run_artifacts(
    backtest_dir,
    calendar=setup["evaluation"]["calendar"],
)
html = generate_backtest_tearsheet(
    profile=profile,
    template="full",
    benchmark_returns=bench_series,
    benchmark_name="EW universe (validation)",
    factor_data=factor_data,
    report_metadata=meta,
    output_path=str(tear_path),
)
print(f"Tear sheet written to: {tear_path}")
print(f"HTML size: {len(html):,} bytes")

# %% [markdown]
# ## §9 Pre-Ch20 judgment & handoff
#
# This section is the explicit hand-off point to Chapter 20. Numbers
# below stay strictly inside us_equities_panel — cross-case-study
# comparison is Ch20's lane.

# %%
op_profile = compute_operating_profile(lineage, setup)
op_profile = op_profile.with_columns(
    pl.when(pl.col("property") == "Trading cadence")
    .then(pl.lit(setup["decision"]["cadence"]))
    .otherwise(pl.col("value"))
    .alias("value")
)
print("Operating profile (us_equities_panel, validation window):")
print(op_profile)
print()
print(
    f"Sharpe: {_fmt_ci(val_full['sharpe'], val_full['sharpe_ci95_lo'], val_full['sharpe_ci95_hi'])}"
)
print(f"Info ratio (vs EW val): {_fmt_ci(getattr(metrics, 'information_ratio', None), None, None)}")
print(f"Max drawdown: {val_full['max_drawdown']:.3f}")
print(
    f"Holdout Sharpe: {ho_full['sharpe']:.3f}; "
    f"holdout-vs-validation diff = "
    f"{_fmt_ci(vh['sharpe_diff'], vh['sharpe_diff_ci95_lo'], vh['sharpe_diff_ci95_hi'])}"
)

# %%
# Setup-defined kill conditions, evaluated as evidence statements (no verdict labels).
ic_floor = setup["kill_conditions"]["ic_floor"]
edge_to_cost_floor = setup["kill_conditions"]["edge_to_cost_floor"]
micro_cap_threshold = setup["kill_conditions"]["micro_cap_concentration"]

ic_passes = (ic_lo or 0) >= ic_floor
ic_evidence = (
    f"daily-pooled IC = {ic_mean:.4f} [CI {ic_lo:.4f}, {ic_hi:.4f}], "
    f"floor = {ic_floor}; CI lower {'≥' if ic_passes else '<'} floor."
)

# Edge-to-cost: net Sharpe at ~10 bps/leg vs the post-decimalization band midpoint.
# Computed against the bps regime (the headline) — the per-share regime is
# exploratory and not used for kill-condition evaluation.
mid_post = (post_lo + post_hi) / 2
near_post_mid = curve_bps.filter(
    (pl.col("cost_value") >= mid_post - 1) & (pl.col("cost_value") <= mid_post + 1)
)
if near_post_mid.is_empty():
    near_post_mid = curve_bps.sort((pl.col("cost_value") - mid_post).abs()).head(1)
nearest_cost_row = near_post_mid.row(0, named=True)
sharpe_at_mid = nearest_cost_row["sharpe_max"]
sharpe_zero_cost = (
    curve_bps.filter(pl.col("cost_value") == 0).row(0, named=True)["sharpe_max"]
    if not curve_bps.filter(pl.col("cost_value") == 0).is_empty()
    else None
)
edge_to_cost_passes = (
    sharpe_at_mid is not None
    and sharpe_zero_cost is not None
    and sharpe_at_mid >= edge_to_cost_floor * 0
    and sharpe_zero_cost > 0
    and (sharpe_at_mid / max(sharpe_zero_cost - sharpe_at_mid, 1e-9)) >= edge_to_cost_floor
)
edge_to_cost_evidence = (
    f"best-config Sharpe at {nearest_cost_row['cost_value']:.0f} bps/leg = "
    f"{sharpe_at_mid:.3f}; gross (0 bps) = "
    f"{(sharpe_zero_cost or 0):.3f}; floor = {edge_to_cost_floor}× ratio."
)

micro_cap_evidence = (
    f"threshold = {micro_cap_threshold:.0%} of alpha in bottom ADV quintile; "
    "no programmatic ADV-quintile attribution implemented in the locked "
    "pipeline — partial evidence (manual review)."
)

print("Setup-defined kill conditions (evidence-only; no verdict labels):")
print(f"  [{'evidence_passes' if ic_passes else 'evidence_below_floor'}] ic_floor: {ic_evidence}")
print(
    f"  [{'evidence_passes' if edge_to_cost_passes else 'evidence_partial'}] "
    f"edge_to_cost_floor: {edge_to_cost_evidence}"
)
print(f"  [evidence_partial] micro_cap_concentration: {micro_cap_evidence}")

# Universal gates
gate_a_status = gate1_validation_sharpe_geq_zero(val_full["sharpe_ci95_lo"])
gate_b_ci_status = ci_status(he["sharpe_diff_ci95_lo"], he["sharpe_diff_ci95_hi"])
gate_b_status = gate2_holdout_diff_not_excludes_zero_negatively(gate_b_ci_status, he["sharpe_diff"])
print()
print("Universal gates:")
print(
    f"  [{fmt_gate(gate_a_status)}] "
    f"validation Sharpe CI lower bound ≥ 0: "
    f"{_fmt(val_full['sharpe_ci95_lo'], '.3f')}"
)
print(
    f"  [{fmt_gate(gate_b_status)}] "
    f"holdout strategy-vs-EW CI does not exclude zero negatively: "
    f"{_fmt_ci(he['sharpe_diff'], he['sharpe_diff_ci95_lo'], he['sharpe_diff_ci95_hi'])} "
    f"({gate_b_ci_status})"
)

# %% [markdown]
# **What this analysis does not say.** The us_equities_panel holdout
# window covers 2016-Q1 through 2018-Q1, a single ~2-year regime in
# which equity-cap-weighted returns were unusually strong. The wide
# val→ho diff CI reflects the holdout-vs-validation regime gap rather
# than a long-run uncertainty about the strategy's edge. `setup.yaml`
# names `fwd_ret_1d` as the primary label, but the cross-stage val
# rank-1 by Sharpe in the registry sits on the `fwd_ret_5d` variant
# horizon (gbm/leaves_31_huber, risk_overlay stage, score_weighted
# top_k=20 + time_exit_40); the strategy P&L is daily-marked-to-market
# either way, and the IC/EW comparisons use the daily fwd_ret_1d EW
# benchmark. Daily rebalance with a 20-name top-K cross-section
# concentrates exposure in a small slice of the ~3,200-stock universe
# and depends on the liquidity filter (ADV>$1M, price>$5) holding
# throughout the holdout; the kill condition for >50% of alpha in the
# bottom ADV quintile is defined but not programmatically evaluated in
# the sweep. Cohort-adjusted DSR_ER for the family cohort
# (risk_overlay/fwd_ret_5d/gbm, K_eff ≈ 2, K_variants = 20) reads
# 0.106 with p ≈ 0 — selection adjustment over the position-level
# overlay grid leaves the cohort's leader Sharpe materially positive
# after MP/ER deflation. PBO 0.0 across 12,870 CSCV combinations × 16
# folds. Borrow availability (50 bps/yr on the short leg,
# easy-to-borrow large caps only) is folded into the operating profile
# but not into the bps-per-leg cost grid.

# %% [markdown]
# **Forward pointer to Ch20.** This case study contributes the
# broad-panel / long-short / daily-rebalance / equity-class datapoint
# to Ch20 nb01's rank-1-Sharpe + holdout-decay aggregation; the §6
# decay magnitude (val→ho diff Sharpe −2.52 [−3.80, −1.12], p ≈ 0,
# excludes_zero_strong on the negative side) and the FF5+MOM
# residual-Sharpe read in §7 feed Ch20 nb02's signal-quality layer.

# %%
search_ctx = ctx
assessment = {
    "case_study": CASE_STUDY,
    "primary_label": PRIMARY_LABEL,
    "spine_version": "v1",
    "rank1": {
        "family": RANK1_FAMILY,
        "config_name": RANK1_CONFIG,
        "label": PRIMARY_LABEL,
        "prediction_hash": TOP_PHASH,
        "validation_backtest_hash": TOP_HASH,
        "holdout_backtest_hash": HO_HASH,
    },
    "headline_performance": {
        "sharpe": {
            "point": val_full["sharpe"],
            "ci95_lo": val_full["sharpe_ci95_lo"],
            "ci95_hi": val_full["sharpe_ci95_hi"],
            "ci_status": sharpe_status,
        },
        "sortino": {
            "point": val_full["sortino"],
            "ci95_lo": val_full["sortino_ci95_lo"],
            "ci95_hi": val_full["sortino_ci95_hi"],
        },
        "ann_return": {
            "point": val_full["cagr"],
            "ci95_lo": val_full["ann_return_ci95_lo"],
            "ci95_hi": val_full["ann_return_ci95_hi"],
        },
        "max_drawdown": {
            "point": val_full["max_drawdown"],
            "ci95_lo": val_full["max_dd_ci95_lo"],
            "ci95_hi": val_full["max_dd_ci95_hi"],
        },
        "psr_pvalue": val_full["psr_pvalue"],
        "bootstrap_block_length": int(val_full["bootstrap_block_length"]),
        "bootstrap_n": int(val_full["bootstrap_n"]),
    },
    "selection_bias": {
        "status": "partial",
        "k_variants": (int(val_full["k_variants"]) if val_full["k_variants"] is not None else None),
        "dsr": val_full["dsr"],
        "dsr_pvalue": val_full["dsr_pvalue"],
        "expected_max_sharpe": val_full["expected_max_sharpe"],
        "pbo": val_full["pbo"],
        "pbo_n_combinations": (
            int(val_full["pbo_n_combinations"])
            if val_full["pbo_n_combinations"] is not None
            else None
        ),
        "min_trl_periods": (
            None
            if val_full["min_trl_periods"] in (None, float("inf"))
            else val_full["min_trl_periods"]
        ),
        "reason_partial": (
            "DSR / expected_max_sharpe / MinTRL / k_variants / "
            "reality_check_* not populated in the locked registry for this CS."
        ),
    },
    "benchmark_relative": {
        "benchmark_name": "equal_weight_universe",
        "benchmark_validation_sharpe": ew_val["sharpe"],
        "benchmark_holdout_sharpe": ew_ho["sharpe"],
        "alpha_annualized_placebo": float(alpha_daily_ew * PERIODS_PER_YEAR),
        "alpha_t_hac_placebo": float(alpha_t_ew),
        "beta_to_ew": float(beta_to_ew),
        "factor_attribution": {
            "method": "ff5_mom_hac",
            "n_obs": int(reg["n_obs"]),
            "alpha_annualized": float(reg["alpha_annualized"]),
            "alpha_t_stat": float(reg["alpha_t_stat"]),
            "alpha_p_value": float(reg["alpha_p_value"]),
            "r_squared": float(reg["r_squared"]),
            "residual_sharpe": float(reg["residual_sharpe"]),
            "betas": {k: float(v) for k, v in reg["betas"].items()},
            "classification": attr_summary["classification"],
        },
        "paired_strategy_vs_benchmark_validation": {
            "sharpe_diff": ve["sharpe_diff"] if ve else None,
            "sharpe_diff_ci95_lo": ve["sharpe_diff_ci95_lo"] if ve else None,
            "sharpe_diff_ci95_hi": ve["sharpe_diff_ci95_hi"] if ve else None,
            "p_value": ve["p_value"] if ve else None,
            "ci_status": (
                ci_status(ve["sharpe_diff_ci95_lo"], ve["sharpe_diff_ci95_hi"]) if ve else None
            ),
        },
        "paired_strategy_vs_benchmark_holdout": {
            "sharpe_diff": he["sharpe_diff"],
            "sharpe_diff_ci95_lo": he["sharpe_diff_ci95_lo"],
            "sharpe_diff_ci95_hi": he["sharpe_diff_ci95_hi"],
            "p_value": he["p_value"],
            "info_ratio": he["info_ratio"],
            "info_ratio_ci95_lo": he["info_ratio_ci95_lo"],
            "info_ratio_ci95_hi": he["info_ratio_ci95_hi"],
            "ci_status": ci_status(he["sharpe_diff_ci95_lo"], he["sharpe_diff_ci95_hi"]),
        },
    },
    "holdout_decay": {
        "val_hash": TOP_HASH,
        "holdout_hash": HO_HASH,
        "sharpe_diff": vh["sharpe_diff"],
        "sharpe_diff_ci95_lo": vh["sharpe_diff_ci95_lo"],
        "sharpe_diff_ci95_hi": vh["sharpe_diff_ci95_hi"],
        "sharpe_diff_p_value": vh["p_value"],
        "info_ratio": vh["info_ratio"],
        "decay_classification": ci_status(vh["sharpe_diff_ci95_lo"], vh["sharpe_diff_ci95_hi"]),
    },
    "search_context": {
        "total_signal_backtests": int(search_ctx["total"]),
        "median_sharpe": search_ctx["median_sharpe"],
        "p90_sharpe": search_ctx["p90_sharpe"],
        "pct_positive": search_ctx["pct_positive"],
    },
    "kill_conditions": {
        "ic_floor": {
            "threshold": ic_floor,
            "evidence": ic_evidence,
            "ci_status_passes": bool(ic_passes),
        },
        "edge_to_cost_floor": {
            "threshold": edge_to_cost_floor,
            "evidence": edge_to_cost_evidence,
            "ci_status_passes": bool(edge_to_cost_passes),
        },
        "micro_cap_concentration": {
            "threshold": micro_cap_threshold,
            "evidence": micro_cap_evidence,
            "ci_status_passes": None,
        },
        "universal_validation_sharpe_ci_lower_bound_geq_zero": gate_passes(gate_a_status),
        "universal_holdout_vs_ew_ci_does_not_exclude_zero_negatively": gate_passes(gate_b_status),
    },
    "ch20_handoff": {
        "contributes_to": [
            "Ch20 nb01 — cross-CS rank-1 Sharpe and holdout-decay aggregation",
            "Ch20 nb02 — FF5+MOM residual-Sharpe and IC-vs-Sharpe datapoint",
        ],
        "asset_class_label": "us_equities_long_short",
        "rebalance_step_days": setup["labels"]["rebalance_step"][PRIMARY_LABEL],
    },
}
assessment_path = write_strategy_assessment(CASE_STUDY, assessment)
print(f"strategy_assessment.json written to: {assessment_path}")
