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
# # NASDAQ-100 Microstructure — Strategy Analysis
#
# This notebook converts the locked May 2026 backtest registry for the
# NASDAQ-100 15-minute microstructure case study into a per-case-study
# strategy assessment. Every metric is reported with its block-bootstrap
# 95% confidence interval, every comparison goes through
# `backtest_paired_metrics`, and the holdout closure uses paired rather
# than point-difference reasoning. Cross-case-study comparison is
# reserved for Chapter 20.
#
# **Learning objectives**
#
# - Read uncertainty-aware backtest metrics (Sharpe ± CI, PSR, DSR) from
#   the registry rather than transcribing point estimates.
# - Trace a rank-1 lineage through pipeline stages with paired-bootstrap
#   stage-transition deltas.
# - Use the equal-weight universe benchmark (the 100-stock NASDAQ-100
#   panel) for both validation and holdout diagnostics.
# - Read a case study where the rank-1 strategy Sharpe excludes zero on
#   the *negative* side — the strategy-analysis notebook's CI continuum represents that
#   outcome the same way it represents a credible positive edge.
#
# **Book reference**: Chapter 20, §20.1 (the §9 handoff feeds Ch20's
# cross-case-study aggregation).
#
# **Prerequisites**: case-study pipeline through `14_backtest`; the
# locked registry (`case_studies/nasdaq100_microstructure/run_log/registry.db`).
#
# **Scope**: registry-read only — no training, no re-backtesting, no
# registry writes. The `backtest_paired_metrics` table was populated by
# `20_strategy_synthesis/01_aggregate_synthesis.py`.

# %%
"""NASDAQ-100 Microstructure — Strategy Analysis."""

import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch  # ml4t.diagnostic loads cudart; torch must import first
import yaml

warnings.filterwarnings("ignore")

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.integration import (
    BacktestReportMetadata,
    generate_tearsheet_from_run_artifacts,
)

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.benchmark import load_benchmark_metrics, load_benchmark_returns
from case_studies.utils.factor_attribution import (
    compute_bootstrap_ci,
    compute_rolling_exposures,
    format_attribution_summary,
    load_factor_data,
    plot_attribution_waterfall,
    plot_rolling_exposures,
    run_factor_regression,
)
from case_studies.utils.notebook_contracts import excluded_families
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
CASE_STUDY = "nasdaq100_microstructure"
# Strategy-analysis convention: use the cross-stage rank-1 backtest's
# actual label for the lineage analysis. PRIMARY_LABEL is derived from
# the rank-1 selection below; the hardcoded default mirrors
# setup.yaml::labels.primary.
PRIMARY_LABEL = "fwd_ret_15m"
PERIODS_PER_YEAR = 252  # strategy returns are aggregated to daily before Sharpe
CASE_DIR = get_case_study_dir(CASE_STUDY)
OUTPUT_DIR = get_output_dir(20, CASE_STUDY)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CASE_DIR / "config" / "setup.yaml") as f:
    setup = yaml.safe_load(f)

explorer = BacktestExplorer(CASE_STUDY)
print(explorer)
if excluded_families(CASE_STUDY):
    print(
        "Active-model filter: excluding "
        f"{', '.join(sorted(excluded_families(CASE_STUDY)))} pending corrected reruns"
    )


def _fmt_ci(point: float | None, lo: float | None, hi: float | None, fmt: str = ".3f") -> str:
    if point is None:
        return "—"
    p = format(point, fmt)
    if lo is None or hi is None:
        return f"{p} [—, —]"
    return f"{p} [{format(lo, fmt)}, {format(hi, fmt)}]"


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    return "—" if val is None else format(val, fmt)


# %% [markdown]
# ## §1 Handoff from model analysis
#
# The strategy phase inherits the cross-stage rank-1 lineage selected
# across (signal + allocation + risk_overlay) on validation Sharpe per
# the `HOLDOUT_SELECTION_STAGES` convention. The upstream prior is the
# rank-1 prediction's daily-pooled IC with its HAC 95% CI from
# `prediction_metrics`. For NASDAQ-100 microstructure the rank-1
# prediction's IC point estimate is faintly positive but the HAC 95%
# CI straddles zero on the lower side (`t_HAC ≈ 1.5`, `p ≈ 0.13`);
# the rank-1 strategy Sharpe is deeply negative on validation with a
# CI whose upper bound just barely clips positive. Every cell in the
# locked sweep is loss-making — the rank-1 is the least-bad
# combination, a position-level `time_exit_20` overlay on a
# `score_weighted` allocation against a long-horizon linear signal.
# The kill-gate read in §9 traces how this val-side reading degrades
# in the 2021-H2 holdout window.

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
RANK1_LABEL = top_signal.row(0, named=True)["label"]
# The rank-1 backtest's label can differ from setup.yaml's primary
# (cross-stage selection across signal/allocation/risk_overlay can
# pick up a variant label's rank-1 if its overlay-rescued Sharpe is
# higher). Use the rank-1's actual label for all downstream loaders.
PRIMARY_LABEL = RANK1_LABEL

_db = CASE_DIR / "run_log" / "registry.db"
with sqlite3.connect(str(_db)) as _con:
    _row = _con.execute(
        "SELECT ic_mean_daily, ic_ci_lo, ic_ci_hi, ic_t_hac, ic_p_hac, ic_n_days, "
        "ic_hac_lag, ic_pct_positive "
        "FROM prediction_metrics WHERE prediction_hash = ?",
        (TOP_PHASH,),
    ).fetchone()

ic_mean, ic_lo, ic_hi, ic_t, ic_p, ic_ndays, ic_lag, ic_pct = _row

print(f"Rank-1: family={RANK1_FAMILY}, config={RANK1_CONFIG}, label={PRIMARY_LABEL}")
print(f"        prediction_hash={TOP_PHASH}, backtest_hash={TOP_HASH}")
print()
print("Daily-pooled IC (validation):")
print(f"  IC = {_fmt_ci(ic_mean, ic_lo, ic_hi, '.4f')}  (HAC, lag={int(ic_lag)})")
print(f"  t_HAC = {ic_t:.3f}, p_HAC = {ic_p:.3f}")
print(f"  n_days = {int(ic_ndays)}, pct_positive = {ic_pct:.1%}")
print(f"  CI status: {ci_status(ic_lo, ic_hi)}")

# %% [markdown]
# Daily-pooled IC at the rank-1 prediction set is faintly positive
# but the HAC 95% CI straddles zero on the lower side (cells above
# report the live values from `prediction_metrics`; `t_HAC ≈ 1.5`,
# `p ≈ 0.13`). The strategy Sharpe in §3 is deeply negative and the
# CI's upper bound barely clips positive — PSR does not reject H₀:
# Sharpe ≤ 0. The §6 holdout-closure paired bootstrap is the canonical
# out-of-sample read.
#
# **Kill conditions** are not encoded in `setup.yaml` for this case
# study. The notebook evaluates two universal gates in §9: (i) the
# validation Sharpe CI lower bound ≥ 0, and (ii) the holdout
# strategy-vs-EW paired CI does not exclude zero on the negative side.
# Both are reported as pass / partial / fail without verdict labels.

# %% [markdown]
# ## §2 Search context, family comparison, and lineage waterfall
#
# The signal stage of the locked sweep produced thousands of validation
# backtests across three model families and four label horizons (5m, 15m,
# 60m, plus a 15m direction classification). The rank-1 row emerges as
# one realization of that search; what matters next is its context —
# where the rank-1 sits in the family-level distribution and how its
# performance evolves through the pipeline stages.

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
# Family comparison reads sharpe_ci95 from backtest_metrics directly so
# the forest plot can render proper error bars rather than median-only
# points.
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
            JOIN backtest_runs b      ON bm.backtest_hash = b.backtest_hash
            JOIN prediction_sets p    ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t      ON p.training_hash   = t.training_hash
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
ax.set_title("Family-level Signal Sharpe — IQR + max")
ax.invert_yaxis()
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.show()

# %% [markdown]
# At the signal stage every backtest in the locked sweep lands in
# negative-Sharpe territory under the engine's 5 bps commission + 2 bps
# slippage cost model. Family medians cluster between roughly −13 and
# −18; the maxima within each family are also negative. The
# friction-dominance reading: at 26 rebalances per day a few-bps cost
# compounds to several hundred percent annualized cost drag against a
# per-bar expected return of order 1 bp. The cross-stage rank-1
# selection in §3 still comes from the risk_overlay stage, but it is
# the least-bad cell — a `time_exit_20` overlay on a `score_weighted`
# allocation against a `linear/ridge_a1e6` `fwd_ret_60m` signal. The
# overlay's role is to cap how long any single position stays on the
# book, modestly slowing the cost bleed rather than rescuing the
# strategy.

# %%
# Lineage with CI bars. champion_lineage gives stage hashes; we re-pull
# Sharpe CIs from backtest_metrics for each stage to render error bars.
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

# Cost_sensitivity and risk_overlay stages exist in the registry but
# may be registered against different prediction sets — not the
# rank-1's. Stage transitions that are unavailable for this
# prediction are reported as "stage not run" rather than recomputed
# inline.
missing_stages = [s for s in ("cost_sensitivity", "risk_overlay") if s not in lineage]
if missing_stages:
    print()
    print(f"Stage transitions not run for this prediction: {missing_stages}")

# %%
fig = plot_sharpe_waterfall(lineage, ci_lo=ci_lo, ci_hi=ci_hi)
fig.show()

# %%
# Stage-transition deltas via load_paired_metrics — never recompute
# paired metrics inline. The only present transition for this prediction
# is signal → allocation, registered as benchmark_kind=signal_leader
# with the allocation-stage backtest as challenger.
ALLOC_HASH = lineage.get("allocation", {}).get("backtest_hash")
if ALLOC_HASH is not None:
    pair = load_paired_metrics(
        CASE_STUDY,
        challenger_hash=ALLOC_HASH,
        benchmark_kind="signal_leader",
    )
    if not pair.is_empty():
        r = pair.row(0, named=True)
        print("Stage transition: allocation challenger vs signal leader")
        print(
            f"  sharpe_diff = "
            f"{_fmt_ci(r['sharpe_diff'], r['sharpe_diff_ci95_lo'], r['sharpe_diff_ci95_hi'])}"
        )
        print(f"  p_value = {r['p_value']:.3f}")
        print(f"  prob_challenger_wins = {r['prob_challenger_wins']:.3f}")
        print(f"  CI status: {ci_status(r['sharpe_diff_ci95_lo'], r['sharpe_diff_ci95_hi'])}")

# %% [markdown]
# Live values for the stage transitions printed above are read from
# `backtest_paired_metrics`. On this CS, every stage delta is small in
# magnitude — the position-level overlays do not produce a stage-delta
# CI that excludes zero on the positive side. The signal economics
# remain the binding constraint; allocator choice and overlay choice
# modulate the magnitude of loss rather than producing a credibility-
# resolved positive transition.

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
# The concentration curve maps how the allocation step's Sharpe responds
# to portfolio breadth. With 100 NASDAQ-100 constituents, narrow
# concentration (top_k = 5) maximizes per-stock exposure and turnover,
# while top_k = 50 is half-universe and dilutes any cross-sectional
# spread. The best Sharpe by top_k surfaces where the rebalance
# cadence × allocator combination spends turnover most efficiently. No
# top_k value is positive enough to clear the validation EW universe
# Sharpe of 0.35.

# %% [markdown]
# ## §3 Headline performance with uncertainty
#
# The rank-1 specification is the validation-window backtest associated
# with the highest signal-stage Sharpe. Every metric is reported with
# its block-bootstrap 95% CI from `backtest_metrics`; the equity overlay
# shows the cumulative trajectory against the equal-weight NASDAQ-100
# universe benchmark.

# %%
full = load_backtest_metrics(CASE_STUDY, backtest_hash=TOP_HASH).row(0, named=True)
full = dict(full)  # mutate-friendly

# Per-row DSR / expected-max-Sharpe / MinTRL / PBO live in `cohort_metrics`
# (selection-bias migration). The spine's rank-1 may sit at a stage whose
# cohort_metrics row is absent (cross-stage timestamp alignment limitation
# documented in `memory/UNCERTAINTY_ARCHITECTURE.md`). Cascade through
# (rank-1 stage → allocation → signal) and resolve the first non-empty
# family cohort row at (label, family). ER is the maintainer-recommended
# default; we also surface RAW and MP for transparency.
_db = CASE_DIR / "run_log" / "registry.db"
_cohort_leader_hash = None
_cohort_leader_sharpe = None
_cohort_stage = None
with sqlite3.connect(str(_db)) as _con:
    _stage_for_rank1 = _con.execute(
        "SELECT stage FROM backtest_runs WHERE backtest_hash = ?",
        (TOP_HASH,),
    ).fetchone()[0]
for _try_stage in dict.fromkeys((_stage_for_rank1, "allocation", "signal")):
    with sqlite3.connect(str(_db)) as _con:
        _cohort = _con.execute(
            """
            SELECT dsr_er, dsr_er_pvalue, expected_max_sharpe_er,
                   min_trl_periods_er, dsr_mp, dsr_mp_pvalue,
                   expected_max_sharpe_mp, min_trl_periods_mp,
                   dsr_raw, dsr_raw_pvalue,
                   expected_max_sharpe_raw, min_trl_periods_raw,
                   n_trials_effective_mp, n_trials_effective_er,
                   pbo, pbo_n_combinations, pbo_n_folds,
                   k_variants, leader_hash, leader_sharpe
            FROM cohort_metrics
            WHERE cohort_type='family' AND stage=? AND label=? AND family=?
              AND leader_sharpe IS NOT NULL
            """,
            (_try_stage, RANK1_LABEL, RANK1_FAMILY),
        ).fetchone()
    if _cohort is not None:
        (
            full["dsr_er"],
            full["dsr_er_pvalue"],
            full["expected_max_sharpe_er"],
            full["min_trl_periods_er"],
            full["dsr_mp"],
            full["dsr_mp_pvalue"],
            full["expected_max_sharpe_mp"],
            full["min_trl_periods_mp"],
            full["dsr_raw"],
            full["dsr_raw_pvalue"],
            full["expected_max_sharpe_raw"],
            full["min_trl_periods_raw"],
            full["n_trials_effective_mp"],
            full["n_trials_effective_er"],
            full["pbo"],
            full["pbo_n_combinations"],
            full["pbo_n_folds"],
            full["k_variants"],
            _cohort_leader_hash,
            _cohort_leader_sharpe,
        ) = _cohort
        _cohort_stage = _try_stage
        # Mirror DSR_ER onto the legacy `dsr` / `dsr_pvalue` / `expected_max_sharpe`
        # fields so downstream consumers that still read those keys see the
        # ER-default values.
        full["dsr"] = full["dsr_er"]
        full["dsr_pvalue"] = full["dsr_er_pvalue"]
        full["expected_max_sharpe"] = full["expected_max_sharpe_er"]
        full["min_trl_periods"] = full["min_trl_periods_er"]
        break

spec_block = {
    "case_study": CASE_STUDY,
    "family": RANK1_FAMILY,
    "config_name": RANK1_CONFIG,
    "label": RANK1_LABEL,
    "signal_method": lineage["signal"].get("signal_method"),
    "top_k": lineage["signal"].get("top_k"),
    "allocation": lineage.get("allocation", {}).get("allocator"),
    "risk_overlay": lineage.get("risk_overlay", {}).get("risk_name"),
    "rebalance_step_bars": setup["labels"]["rebalance_step"][RANK1_LABEL],
    "cost_assumption": "engine cost model: 5 bps commission + 2 bps slippage at signal stage; per_share_plus_spread sensitivity in §5",
    "validation_window_periods": int(full["n_periods"]) if full["n_periods"] is not None else None,
    "num_trades": int(full["num_trades"]) if full["num_trades"] is not None else None,
    "avg_turnover": full["avg_turnover"],
    "bootstrap_block_length": int(full["bootstrap_block_length"])
    if full["bootstrap_block_length"] is not None
    else None,
    "bootstrap_n": int(full["bootstrap_n"]) if full["bootstrap_n"] is not None else None,
}
print("Rank-1 specification (cross-stage selection, validation window):")
for k, v in spec_block.items():
    print(f"  {k}: {v}")

# Audit: bootstrap_block_length encodes the autocorrelation horizon of
# the Sharpe target. rebalance_step is the per-label trade frequency
# in 15-minute bars. The two are independent — block length comes from
# the daily-MTM Sharpe series; rebalance_step comes from setup.yaml.
_block = int(full["bootstrap_block_length"])
_rstep = setup["labels"]["rebalance_step"][RANK1_LABEL]
print(
    f"  audit: bootstrap_block_length={_block} day(s); "
    f"rebalance_step={_rstep} bar(s) ({_rstep * 15} minutes)"
)

# Cohort cascade audit: surface where DSR comes from when the rank-1
# stage's cohort row is missing.
if _cohort_stage and _cohort_stage != _stage_for_rank1:
    print(
        f"  cohort cascade: rank-1 stage `{_stage_for_rank1}` has no "
        f"cohort_metrics row (cross-stage timestamp-alignment limit); "
        f"DSR/EMS/MinTRL/PBO sourced from family cohort `{_cohort_stage}/"
        f"{RANK1_LABEL}/{RANK1_FAMILY}` (k={int(full['k_variants'])}, "
        f"leader_hash={_cohort_leader_hash}, leader_sharpe="
        f"{_cohort_leader_sharpe:+.3f}; spine rank-1 sibling differs by "
        f"{full['sharpe'] - _cohort_leader_sharpe:+.3f})."
    )
elif _cohort_stage:
    print(
        f"  cohort: family cohort `{_cohort_stage}/{RANK1_LABEL}/"
        f"{RANK1_FAMILY}` (k={int(full['k_variants'])}, leader_hash="
        f"{_cohort_leader_hash}); spine rank-1 sibling differs by "
        f"{full['sharpe'] - _cohort_leader_sharpe:+.3f}."
    )


# %%
def _row(metric: str, point: str, lo: str, hi: str, status: str) -> dict:
    return {"metric": metric, "point": point, "ci95_lo": lo, "ci95_hi": hi, "status": status}


sharpe_status = ci_status(full["sharpe_ci95_lo"], full["sharpe_ci95_hi"])
sortino_status = ci_status(full["sortino_ci95_lo"], full["sortino_ci95_hi"])
ann_status = ci_status(full["ann_return_ci95_lo"], full["ann_return_ci95_hi"])
mdd_status = ci_status(full["max_dd_ci95_lo"], full["max_dd_ci95_hi"])
calmar_status = ci_status(full["calmar_ci95_lo"], full["calmar_ci95_hi"])

headline = pl.DataFrame(
    [
        _row(
            "Sharpe",
            _fmt(full["sharpe"]),
            _fmt(full["sharpe_ci95_lo"]),
            _fmt(full["sharpe_ci95_hi"]),
            sharpe_status,
        ),
        _row(
            "Sortino",
            _fmt(full["sortino"]),
            _fmt(full["sortino_ci95_lo"]),
            _fmt(full["sortino_ci95_hi"]),
            sortino_status,
        ),
        _row(
            "Annualized return",
            _fmt(full["cagr"]),
            _fmt(full["ann_return_ci95_lo"]),
            _fmt(full["ann_return_ci95_hi"]),
            ann_status,
        ),
        _row(
            "Max drawdown",
            _fmt(full["max_drawdown"]),
            _fmt(full["max_dd_ci95_lo"]),
            _fmt(full["max_dd_ci95_hi"]),
            mdd_status,
        ),
        _row(
            "Calmar",
            _fmt(full["calmar"]),
            _fmt(full["calmar_ci95_lo"]),
            _fmt(full["calmar_ci95_hi"]),
            calmar_status,
        ),
        {
            "metric": "PSR p-value (H0: SR≤0)",
            "point": _fmt(full["psr_pvalue"]),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "DSR_ER (selection-adjusted)",
            "point": _fmt(full.get("dsr_er")),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "DSR_ER p-value",
            "point": _fmt(full.get("dsr_er_pvalue")),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "DSR_MP",
            "point": _fmt(full.get("dsr_mp")),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "DSR_MP p-value",
            "point": _fmt(full.get("dsr_mp_pvalue")),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "DSR_RAW (naive raw-K)",
            "point": _fmt(full.get("dsr_raw")),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "DSR_RAW p-value",
            "point": _fmt(full.get("dsr_raw_pvalue")),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": (
                f"Expected max Sharpe (k={int(full['k_variants'])})"
                if full.get("k_variants") is not None
                else "Expected max Sharpe"
            ),
            "point": _fmt(full.get("expected_max_sharpe_er")),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
        {
            "metric": "PBO (CSCV)",
            "point": _fmt(full.get("pbo"), ".3f"),
            "ci95_lo": "—",
            "ci95_hi": "—",
            "status": "n/a",
        },
    ]
)
print("Rank-1 headline metrics with 95% CIs:")
print(headline)

# %%
# Forest plot: rank-1 metrics with CI bars + reference lines
ew_val = load_benchmark_metrics(CASE_STUDY, PRIMARY_LABEL, period="validation")
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
ax.errorbar(
    points,
    y,
    xerr=[points - los, his - points],
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
ax.set_title("Rank-1 Headline Metrics with 95% CIs")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.show()

# %%
# Equity-curve overlay vs validation EW benchmark.
# Strategy returns are stored at daily frequency; the benchmark series
# is at 15-minute frequency and is aggregated to daily before alignment.
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
    .group_by("ts")
    .agg(((1 + pl.col("ew_return")).product() - 1).alias("benchmark"))
    .sort("ts")
)

aligned = strat_df.join(bench_val, on="ts", how="inner").sort("ts")
print(
    f"Validation overlay window: {aligned['ts'].min()} → {aligned['ts'].max()}, n={aligned.height}"
)

cum_strat = np.cumprod(1 + aligned["strategy"].to_numpy()) - 1
cum_bench = np.cumprod(1 + aligned["benchmark"].to_numpy()) - 1
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(aligned["ts"], cum_strat, color="#1565C0", linewidth=1.2, label="Rank-1 strategy")
ax.plot(aligned["ts"], cum_bench, color="#43A047", linewidth=1.2, label="EW universe")
ax.axhline(0, color="#9E9E9E", linewidth=0.6, linestyle="--")
ax.set_ylabel("Cumulative return")
ax.set_title("Validation-window cumulative return: rank-1 strategy vs EW universe")
ax.legend(loc="best", frameon=False)
fig.tight_layout()
fig.show()

# %% [markdown]
# The headline rank-1 lineage is the cross-stage maximum-Sharpe
# backtest from the (signal + allocation + risk_overlay) pool — read
# the live values printed above for the current carrier, validation
# Sharpe + CI, and benchmark-relative paired-bootstrap statistics.
# Every cell in the post-purge sweep is loss-making; the rank-1 is
# the least-bad combination — a position-level `time_exit` overlay
# on a `score_weighted` allocation against a long-horizon linear
# signal. The overlay caps how long any single position stays on the
# book; it does not produce a credibility-resolved positive Sharpe
# (CI lower bounds remain below zero across the validation window).
# The §6 holdout closure on the same lineage decays further (see §6).
#
# Selection-bias adjustment: the rank-1's stage cohort
# `family/risk_overlay/fwd_ret_60m/linear` is present in
# `cohort_metrics` (k=20 variants); no cascade fallback is needed. The
# DSR triad reads `DSR_ER ≈ -0.25`, `DSR_MP ≈ -0.23`, and `DSR_RAW`
# (naive raw-K counting) `≈ -0.52`, all at `p ≈ 1.0` — selection-bias
# deflation cannot resolve any cell in this stage cohort above zero.
# Expected-max-Sharpe under ER lands around `1.0`, comfortably above
# the observed `-1.985`, and `MinTRL` is infinite (the library returns
# `inf` when the expected maximum exceeds the observed Sharpe). PBO
# under CSCV is `0.0` with only `2` walk-forward folds — the
# combinatorial space is too small for PBO to be load-bearing on this
# CS. The §6 paired holdout-decay bootstrap is the canonical
# out-of-sample resolution.

# %% [markdown]
# ## §4 Risk and drawdown analysis
#
# Risk metrics use the validation-window strategy returns paired against
# the validation EW benchmark. The drawdown panel surfaces the worst
# episode and recovery; rolling Sharpe and rolling beta locate when the
# strategy decoupled from the universe. For an intraday cross-sectional
# strategy on the NASDAQ-100, a deep persistent drawdown is the
# signature of cost compounding rather than an episodic regime shift.

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
# Rolling Sharpe + rolling beta
roll = pa.compute_rolling_metrics(windows=[126], metrics=["sharpe", "beta"])
print("Rolling-window keys:")
print({k: type(v).__name__ for k, v in roll.items()} if isinstance(roll, dict) else roll)

# Tail-risk read straight from registry (already computed and stored)
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
print("Tail risk profile:")
print(tail_table)

# %%
fold_df = load_backtest_fold_metrics(CASE_STUDY, backtest_hash=TOP_HASH)
print(f"Per-fold breakdown ({fold_df.height} folds):")
print(fold_df.select("fold_id", "sharpe", "max_drawdown", "n_days"))
print()
print(f"Fold Sharpe range: [{fold_df['sharpe'].min():.3f}, {fold_df['sharpe'].max():.3f}]")
print(f"Fold Sharpe std:   {fold_df['sharpe'].std():.3f}")

# %% [markdown]
# Per-fold Sharpes are both negative across the two walk-forward
# blocks (live values printed above), confirming that the rank-1's
# loss profile is not a single-fold artifact. The tail-risk read
# carries a moderately-low tail ratio with elevated kurtosis and
# positive skew — the realized P&L distribution is dominated by many
# small adverse-selection losses interrupted by occasional large
# upside bars, the signature of a strategy whose costs accrue on
# every rebalance while wins are concentrated. The block-bootstrap
# max-drawdown CI (live values above) is consistent with resampling
# that cannot rule out severe capital loss. The drawdown is monotone
# — there is no recovery, only continued decline — which is what
# cost-dominant strategies look like when run at a cadence whose
# friction exceeds the per-bar edge.

# %% [markdown]
# ## §5 Friction budget & cost sensitivity
#
# NASDAQ-100 microstructure runs at 15-minute cadence — the highest
# turnover regime in the book — so cost sensitivity is the section
# where this case study earns its place. The locked sweep populated two
# stage-level sensitivity surfaces against the rank-1 prediction
# family: a **cost grid** (commission + slippage walked from 0 to 50
# bps per leg) and a **risk-overlay grid** (trailing-stop, stop-loss
# and time-exit families, ~20 overlays per label). Together they
# answer two distinct questions: (i) how much the strategy can pay
# in friction before the per-bar edge runs out, and (ii) whether a
# position-level overlay (capping per-position holding time or loss)
# can preserve edge by *trading less* on bad positions. The two grids
# are independent — overlays were applied at zero-cost engine
# assumptions in their stage, so an overlay's Sharpe must still be
# read net of the realistic-friction penalty established below.
#
# Realistic NQ100 costs from `setup.yaml`: large-cap effective spreads
# of 1–3 bps, mid-cap 3–8 bps, with a 5 bps friction floor;
# institutional commission of 0.1¢ per share.

# %% [markdown]
# ### §5.1 Cost sensitivity stratified by label horizon

# %%
with sqlite3.connect(str(_db)) as _con:
    cost_df = pl.DataFrame(
        _con.execute(
            """
            SELECT
                b.spec_json,
                t.family,
                t.config_name,
                t.label,
                bm.sharpe,
                bm.sharpe_ci95_lo,
                bm.sharpe_ci95_hi,
                bm.max_drawdown,
                bm.cagr,
                bm.num_trades,
                bm.avg_turnover
            FROM backtest_runs b
            JOIN backtest_metrics bm ON bm.backtest_hash  = b.backtest_hash
            JOIN prediction_sets p   ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t     ON p.training_hash   = t.training_hash
            WHERE b.stage = 'cost_sensitivity'
              AND p.split = 'validation'
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
            """
        ).fetchall(),
        schema=[
            "spec_json",
            "family",
            "config_name",
            "label",
            "sharpe",
            "sharpe_ci95_lo",
            "sharpe_ci95_hi",
            "max_drawdown",
            "cagr",
            "num_trades",
            "avg_turnover",
        ],
        orient="row",
    )


def _cost_bps(spec_str: str) -> float:
    """Cost per leg in bps from the locked spec.

    `backtest_config.commission.rate` and `backtest_config.slippage.rate`
    are decimal fractions; sum × 10000 gives total per-leg cost in bps.
    """
    spec = json.loads(spec_str)
    bc = spec.get("backtest_config", {})
    comm = bc.get("commission", {}) or {}
    slip = bc.get("slippage", {}) or {}
    return float((comm.get("rate", 0) + slip.get("rate", 0)) * 10000)


cost_df = cost_df.with_columns(
    pl.col("spec_json").map_elements(_cost_bps, return_dtype=pl.Float64).alias("cost_bps")
)

# Aggregate per (label, cost_bps): max-config Sharpe + envelope
cost_curve_per_label = (
    cost_df.group_by(["label", "cost_bps"])
    .agg(
        n=pl.len(),
        sharpe_max=pl.col("sharpe").max(),
        sharpe_min=pl.col("sharpe").min(),
        sharpe_median=pl.col("sharpe").median(),
        # CI lower bound *of the best-Sharpe row* per (label, cost):
        # this is the credibility band on the headline cell.
        best_ci_lo=pl.col("sharpe_ci95_lo").top_k_by("sharpe", 1).first(),
        best_ci_hi=pl.col("sharpe_ci95_hi").top_k_by("sharpe", 1).first(),
        best_n_trades=pl.col("num_trades").top_k_by("sharpe", 1).first(),
        best_avg_turnover=pl.col("avg_turnover").top_k_by("sharpe", 1).first(),
        best_family=pl.col("family").top_k_by("sharpe", 1).first(),
        best_config=pl.col("config_name").top_k_by("sharpe", 1).first(),
    )
    .sort(["label", "cost_bps"])
)
print("Per-label cost grid (validation, best Sharpe per (label, cost) cell):")
with pl.Config(tbl_rows=33):
    print(
        cost_curve_per_label.select(
            [
                "label",
                "cost_bps",
                "sharpe_max",
                "best_ci_lo",
                "best_ci_hi",
                "sharpe_median",
                "best_n_trades",
                "best_family",
                "best_config",
            ]
        )
    )

# %%
# Per-label cost curve: one line per label horizon
fig, ax = plt.subplots(figsize=(9.5, 4.5))
label_order = ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"]
label_colors = {
    "fwd_ret_5m": "#C62828",
    "fwd_ret_15m": "#FB8C00",
    "fwd_ret_60m": "#1565C0",
}
for lbl in label_order:
    sub = cost_curve_per_label.filter(pl.col("label") == lbl).sort("cost_bps")
    if sub.is_empty():
        continue
    xs = sub["cost_bps"].to_numpy()
    ax.plot(
        xs,
        sub["sharpe_max"].to_numpy(),
        color=label_colors[lbl],
        linewidth=1.6,
        marker="o",
        markersize=4,
        label=f"{lbl} (best config)",
    )
    ax.fill_between(
        xs,
        sub["best_ci_lo"].to_numpy(),
        sub["best_ci_hi"].to_numpy(),
        color=label_colors[lbl],
        alpha=0.10,
    )

ax.axhline(0, color="#9E9E9E", linewidth=0.8, linestyle="--")
ax.axhline(
    ew_val["sharpe"],
    color="#43A047",
    linewidth=0.9,
    linestyle=":",
    label=f"EW validation Sharpe ({ew_val['sharpe']:.2f})",
)
ax.axvspan(1.0, 3.0, color="#43A047", alpha=0.08, label="large-cap spread (1–3 bps)")
ax.axvspan(3.0, 8.0, color="#FB8C00", alpha=0.08, label="mid-cap spread (3–8 bps)")
ax.axvline(
    5.0, color="#C62828", linewidth=0.8, linestyle=":", label="protocol friction floor (5 bps)"
)
ax.set_xlabel("Per-leg cost (bps)")
ax.set_ylabel("Sharpe (validation, best config per cost level)")
ax.set_title("Cost sensitivity by label horizon — best config per (label, cost)")
ax.legend(loc="best", fontsize=8, frameon=False)
fig.tight_layout()
fig.show()

# %%
# Per-label trade-count contrast at zero cost — the cadence × turnover
# evidence. Higher cadence -> higher num_trades -> higher friction
# elasticity.
turnover_table = (
    cost_df.filter(pl.col("cost_bps") == 0)
    .group_by("label")
    .agg(
        n_configs=pl.len(),
        max_sharpe=pl.col("sharpe").max(),
        median_sharpe=pl.col("sharpe").median(),
        max_num_trades=pl.col("num_trades").max(),
        min_num_trades=pl.col("num_trades").min(),
        median_avg_turnover=pl.col("avg_turnover").median(),
    )
    .sort("median_avg_turnover", descending=True)
)
print("Trade count and turnover at zero cost by label horizon:")
print(turnover_table)


# %%
# Per-label breakeven costs: (a) point-estimate Sharpe still > 0,
# (b) Sharpe CI lower bound still > 0 (the credibility-aware breakeven).
def _breakevens(curve: pl.DataFrame) -> dict:
    point_be = curve.filter(pl.col("sharpe_max") > 0)["cost_bps"].max()
    ci_be = curve.filter(pl.col("best_ci_lo") > 0)["cost_bps"].max()
    return {
        "point_estimate_max_bps": point_be,
        "ci_aware_max_bps": ci_be,
    }


print("Breakeven analysis (per-label):")
for lbl in label_order:
    sub = cost_curve_per_label.filter(pl.col("label") == lbl)
    be = _breakevens(sub)
    print(
        f"  {lbl}: point-estimate Sharpe>0 up to {be['point_estimate_max_bps']} bps; "
        f"CI lower bound>0 up to {be['ci_aware_max_bps']} bps"
    )

print()
print("Realistic NQ100 friction (per setup.yaml):")
print(f"  Cost model: {setup['costs'].get('model')}")
print(f"  Per-share commission: ${setup['costs'].get('per_share')}/share")
print(f"  Minimum per trade: ${setup['costs'].get('minimum', 0.0)}")
print(f"  Half-spread source: {setup['costs'].get('asset_spreads_source', 'n/a')}")
print("See Chapter 18 for transaction-cost framework details.")

# %% [markdown]
# The cadence × cost interaction is the case study's central finding.
# The cost sweep populates one prediction lineage per label across
# eleven cost levels (live row counts print above). The stylized
# reading by label horizon:
#
# - **fwd_ret_5m**: every cell is deeply negative even at zero cost.
#   The 5-minute strategy is already losing on raw signal at this
#   cadence; cost levels do not change the verdict. The cumulative
#   integral of small adverse-selection events at 5-minute bars is
#   the binding constraint, not commission.
# - **fwd_ret_15m**: no cost-grid cell clears zero on point estimate.
#   The 15-minute horizon does not produce a point-estimate-positive
#   configuration even at zero cost.
# - **fwd_ret_60m**: the longest horizon in the locked grid carries
#   the least-negative cost trajectory. linear/ridge configs survive
#   to roughly `-0.8 Sharpe` at zero cost and decay monotonically
#   with friction; no cell clears zero on point estimate or CI.
#
# CI lower bounds sit below zero across the entire grid for every
# label — the cost sweep does not surface a credibility-resolved
# positive cell. The dispersion across horizons mirrors the per-bar
# edge × per-bar friction trade-off: longer horizons reduce trade
# count enough that friction matters less, but the per-bar signal
# at the NQ100 spread band of 1–3 bps for large caps is still too
# small to clear zero by itself. The §5.2 risk-overlay sweep modulates
# the magnitude of the loss via position-level exits; it does not turn
# the strategy positive.

# %% [markdown]
# ### §5.2 Risk-overlay sensitivity sweep
#
# The risk_overlay stage applies position-level overlays per label on
# top of the base allocation: **trailing stop** (exit a position when
# its price retraces a threshold from its high-water mark since entry),
# **stop-loss** (exit when unrealized loss exceeds a threshold), and
# **time exit** (close a position after a fixed number of bars). Each
# rule acts on individual positions; the rest of the book continues
# trading. Portfolio-level kill switches (max_drawdown / daily_loss)
# are NOT swept here — their permanent-halt semantics produced
# zero-std Sharpe artifacts in earlier passes, so they are reserved
# for Ch19 §19.8 governance demos instead of model selection.
#
# At 15-minute cadence every overlay trades cost-induced churn for a
# tighter risk profile. Tight thresholds (1–5%) trigger constantly and
# stack large cost drag; loose thresholds (≥15%) and long time exits
# (20–40 bars) cluster near the top of the sweep distribution because
# they let the (already loss-making) configuration trade more freely.

# %%
with sqlite3.connect(str(_db)) as _con:
    risk_df = pl.DataFrame(
        _con.execute(
            """
            SELECT
                b.spec_json,
                t.label,
                bm.sharpe,
                bm.sharpe_ci95_lo,
                bm.sharpe_ci95_hi,
                bm.max_drawdown,
                bm.cagr,
                bm.num_trades,
                bm.avg_turnover,
                bm.volatility
            FROM backtest_runs b
            JOIN backtest_metrics bm ON bm.backtest_hash  = b.backtest_hash
            JOIN prediction_sets p   ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t     ON p.training_hash   = t.training_hash
            WHERE b.stage = 'risk_overlay'
              AND p.split = 'validation'
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
            """
        ).fetchall(),
        schema=[
            "spec_json",
            "label",
            "sharpe",
            "sharpe_ci95_lo",
            "sharpe_ci95_hi",
            "max_drawdown",
            "cagr",
            "num_trades",
            "avg_turnover",
            "volatility",
        ],
        orient="row",
    )


def _risk_name(spec_str: str) -> str:
    s = json.loads(spec_str).get("strategy", {}) or {}
    r = s.get("risk_overlay", s.get("risk", {})) or {}
    return r.get("name", r.get("type", "-"))


def _risk_family(spec_str: str) -> str:
    """Group the position-level overlays into families for visualization."""
    name = _risk_name(spec_str)
    if name.startswith("trailing_"):
        return "trailing_stop"
    if name.startswith("stop_loss"):
        return "stop_loss"
    if name.startswith("time_exit"):
        return "time_exit"
    return "other"


risk_df = risk_df.with_columns(
    pl.col("spec_json").map_elements(_risk_name, return_dtype=pl.String).alias("risk_name"),
    pl.col("spec_json").map_elements(_risk_family, return_dtype=pl.String).alias("risk_family"),
)

# Per-label / per-family overlay summary
risk_summary = (
    risk_df.group_by(["label", "risk_family"])
    .agg(
        n=pl.len(),
        best_sharpe=pl.col("sharpe").max(),
        best_overlay=pl.col("risk_name").top_k_by("sharpe", 1).first(),
        best_ci_lo=pl.col("sharpe_ci95_lo").top_k_by("sharpe", 1).first(),
        best_ci_hi=pl.col("sharpe_ci95_hi").top_k_by("sharpe", 1).first(),
        best_maxdd=pl.col("max_drawdown").top_k_by("sharpe", 1).first(),
        best_n_trades=pl.col("num_trades").top_k_by("sharpe", 1).first(),
        median_sharpe=pl.col("sharpe").median(),
    )
    .sort(["label", "best_sharpe"], descending=[False, True])
)
print("Risk-overlay sensitivity by (label, family):")
with pl.Config(tbl_rows=20, fmt_str_lengths=40):
    print(risk_summary)

# %%
# Top-3 overlays per label — the headline rescue table
top_per_label = (
    risk_df.sort("sharpe", descending=True)
    .group_by("label")
    .head(3)
    .sort(["label", "sharpe"], descending=[False, True])
)
print("Top-3 risk overlays per label horizon:")
with pl.Config(tbl_rows=12, fmt_str_lengths=40):
    print(
        top_per_label.select(
            [
                "label",
                "risk_name",
                "risk_family",
                "sharpe",
                "sharpe_ci95_lo",
                "sharpe_ci95_hi",
                "max_drawdown",
                "num_trades",
                "avg_turnover",
            ]
        )
    )

# %%
# Forest plot: best overlay per (label, family) with CI bars
families_order = ["trailing_stop", "stop_loss", "time_exit"]
fig, ax = plt.subplots(figsize=(10, 5))
y_pos = []
y_labels = []
y_idx = 0
family_colors = {
    "trailing_stop": "#FB8C00",
    "stop_loss": "#1565C0",
    "time_exit": "#C62828",
}
for lbl in label_order:
    for fam in families_order:
        sub = risk_summary.filter((pl.col("label") == lbl) & (pl.col("risk_family") == fam))
        if sub.is_empty():
            continue
        row = sub.row(0, named=True)
        ax.errorbar(
            row["best_sharpe"],
            y_idx,
            xerr=[
                [row["best_sharpe"] - row["best_ci_lo"]],
                [row["best_ci_hi"] - row["best_sharpe"]],
            ],
            fmt="o",
            color=family_colors.get(fam, "#666"),
            ecolor=family_colors.get(fam, "#666"),
            elinewidth=1.6,
            capsize=4,
            markersize=7,
        )
        y_pos.append(y_idx)
        y_labels.append(f"{lbl} · {fam}\n({row['best_overlay']})")
        y_idx += 1

ax.axvline(0, color="#9E9E9E", linestyle="--", linewidth=0.8)
ax.axvline(
    ew_val["sharpe"],
    color="#43A047",
    linestyle=":",
    linewidth=1.0,
    label=f"EW validation Sharpe ({ew_val['sharpe']:.2f})",
)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Validation Sharpe")
ax.set_title("Risk-overlay sensitivity — best overlay per (label, family) with 95% CIs")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.show()

# %%
# Compare overlay rescue vs no-overlay baseline per label
no_overlay_baseline = pl.read_database(
    """
        SELECT t.label, MAX(bm.sharpe) AS best_signal_sharpe
        FROM backtest_runs b
        JOIN backtest_metrics bm ON bm.backtest_hash  = b.backtest_hash
        JOIN prediction_sets p   ON b.prediction_hash = p.prediction_hash
        JOIN training_runs t     ON p.training_hash   = t.training_hash
        WHERE b.stage = 'signal' AND p.split = 'validation'
          AND bm.sharpe IS NOT NULL AND (bm.num_trades IS NULL OR bm.num_trades > 0)
        GROUP BY t.label
        """,
    sqlite3.connect(str(_db)),
).filter(pl.col("label").is_in(label_order))

best_overlay_per_label = (
    risk_df.sort("sharpe", descending=True)
    .group_by("label")
    .head(1)
    .select(
        [
            "label",
            "sharpe",
            "sharpe_ci95_lo",
            "sharpe_ci95_hi",
            "risk_name",
            "max_drawdown",
            "num_trades",
        ]
    )
    .rename({"sharpe": "best_overlay_sharpe"})
)
rescue_table = no_overlay_baseline.join(best_overlay_per_label, on="label", how="inner").sort(
    "label"
)
print("Risk-overlay rescue analysis (no-overlay best vs best-overlay):")
print(
    rescue_table.with_columns(rescue=pl.col("best_overlay_sharpe") - pl.col("best_signal_sharpe"))
)

# %% [markdown]
# The best-overlay-per-label cells printed above carry the headline
# story: every cell is loss-making, and the cross-stage rank-1 is the
# least-bad combination — a `linear/ridge_a1e6 fwd_ret_60m` signal
# under `score_weighted top_k=5` with a `time_exit_20` overlay. The
# overlay does not rescue the strategy; it makes a loss-making
# configuration slightly less loss-making by capping how long any
# single position stays on the book.
#
# Family ordering matters less than threshold magnitude. Longer time
# exits (20–40 bars) and looser stop/trailing thresholds (≥15%)
# cluster at the top of the distribution — they minimize the
# cost-induced churn that dominates at 15-minute cadence. Tight
# thresholds (1–5%) compound the friction trap. The §6 holdout closure
# on the same lineage decays further (see §6); the signal does not
# generalize and Universal kill gate 2 fails (§9).
#
# Caveat: the risk_overlay stage runs with the engine default cost
# model (5 bps commission + 2 bps slippage per leg). The realistic
# NQ100 spread band of 1–3 bps for large caps and 3–8 bps for mid caps
# would shift every cell, but is unlikely to push the rank-1 into
# credibility-resolved positive territory given the val-window
# distribution.

# %% [markdown]
# ## §6 Holdout closure with paired bootstrap
#
# The deployed holdout carrier is the ensemble (`gbm_reg12_mean`) fallback,
# not the validation rank-1. The validation rank-1 (a long-horizon linear
# signal) retrained to degenerate, near-constant holdout predictions, so
# `generate_holdout` advanced to the next non-degenerate trained model. Two
# paired tests anchor the holdout read: (i) the carrier versus its own
# validation backtest ("did Sharpe hold?"), and (ii) the carrier versus the
# holdout-window equal-weight benchmark ("did the strategy beat random in
# the holdout?"). Numbers come from `backtest_paired_metrics` — never from
# val_sharpe minus holdout_sharpe arithmetic.

# %%
# Identify the canonical holdout backtest. Per the one-holdout-per-CS rule
# there is exactly one canonical (non-conformal) holdout backtest for the
# case study; the conformal_weighted sweep, when present, adds a separate
# transparency sibling that is excluded here by its allocation method.
#
# The holdout's training lineage can differ from the validation rank-1
# (TOP_PHASH): generate_holdout falls back from the rank-1 to a rank-K
# trained model when the rank-1's holdout retrain produces degenerate
# (constant) predictions. For NASDAQ-100 the validation rank-1 is a
# long-horizon linear signal whose holdout retrain was degenerate, so the
# deployed carrier is the ensemble (gbm_reg12_mean) fallback. We therefore
# resolve the canonical holdout directly rather than constraining it to
# TOP_PHASH's lineage, and anchor §6's paired tests on the carrier's own
# validation lineage below.
with sqlite3.connect(str(_db)) as _con:
    _ho_row = _con.execute(
        """
        SELECT b.backtest_hash, p.training_hash
        FROM backtest_runs b
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        WHERE b.stage IN ('signal','allocation','risk_overlay','holdout')
          AND p.split = 'holdout'
          AND COALESCE(
              json_extract(b.spec_json, '$.strategy.allocation.method'), ''
          ) != 'conformal_weighted'
        ORDER BY b.created_at ASC
        LIMIT 1
        """,
    ).fetchone()
if _ho_row is None:
    raise RuntimeError(
        f"No canonical holdout backtest registered for {CASE_STUDY}; cannot anchor §6."
    )
HO_HASH, _HO_TRAINING = _ho_row

# Conformal sibling holdout (if registered) — recorded for transparency.
# Keyed off the canonical holdout's own lineage, not the validation rank-1.
with sqlite3.connect(str(_db)) as _con:
    _ho_conformal_row = _con.execute(
        """
        SELECT b.backtest_hash, bm.sharpe
        FROM backtest_runs b
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
        WHERE b.stage IN ('signal','allocation','risk_overlay','holdout')
          AND p.split = 'holdout'
          AND bm.sharpe IS NOT NULL
          AND json_extract(b.spec_json, '$.strategy.allocation.method') = 'conformal_weighted'
          AND p.training_hash = ?
        ORDER BY b.created_at ASC
        LIMIT 1
        """,
        (_HO_TRAINING,),
    ).fetchone()
HO_HASH_CONFORMAL = _ho_conformal_row[0] if _ho_conformal_row else None
HO_SHARPE_CONFORMAL = _ho_conformal_row[1] if _ho_conformal_row else None

print(f"Validation rank-1 hash:        {TOP_HASH}")
print(f"Holdout (canonical) hash:      {HO_HASH}")
if HO_HASH_CONFORMAL:
    print(
        f"Holdout (conformal sibling):   {HO_HASH_CONFORMAL} "
        f"(Sharpe {HO_SHARPE_CONFORMAL:+.3f}; transparency variant of "
        "canonical baseline, not used for §6 closure)"
    )

ho_full = load_backtest_metrics(CASE_STUDY, backtest_hash=HO_HASH).row(0, named=True)
# Anchor the val→holdout decay on the carrier's OWN validation lineage so the
# displayed validation column matches the paired-bootstrap diff below. When the
# holdout carrier shares the validation rank-1's lineage this equals `full`;
# when it is a degeneracy fallback (NASDAQ-100), it is the fallback model's
# validation backtest.
with sqlite3.connect(str(_db)) as _con:
    _val_self_row = _con.execute(
        "SELECT benchmark_hash FROM backtest_paired_metrics "
        "WHERE challenger_hash = ? AND benchmark_kind = 'val_rank1_self'",
        (HO_HASH,),
    ).fetchone()
if _val_self_row and _val_self_row[0]:
    val_full = load_backtest_metrics(CASE_STUDY, backtest_hash=_val_self_row[0]).row(0, named=True)
else:
    val_full = full

# %%
# Paired val→ho decay table from load_paired_metrics
val_ho_pair = load_paired_metrics(
    CASE_STUDY,
    challenger_hash=HO_HASH,
    benchmark_kind="val_rank1_self",
)
if val_ho_pair.is_empty():
    print(
        "[WARN] Missing val_rank1_self pair for nasdaq100_microstructure — populator "
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
    v: float,
    h: float,
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
        "diff CI95": f"[{_fmt(lo, '.4f')}, {_fmt(hi, '.4f')}]" if lo is not None else "—",
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
print(f"prob_challenger_wins: {vh['prob_challenger_wins']:.3f}")
print(f"CI status (Sharpe diff): {ci_status(vh['sharpe_diff_ci95_lo'], vh['sharpe_diff_ci95_hi'])}")
print()
print(
    "Note: validation and holdout windows are disjoint (validation 2020-07 "
    "→ 2021-06; holdout 2021-07 → 2021-12), so the populator pairs "
    "the bootstrapped Sharpe series by row index after a min-length "
    "truncation. The CI is interpreted as bootstrap-resampled Sharpe "
    "difference under index-paired draws, not as a calendar-aligned overlap."
)

# %%
# Paired strategy-vs-EW-benchmark on the holdout window
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
        "nasdaq100_microstructure — holdout has no trades or all-zero returns; "
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

ew_ho = load_benchmark_metrics(CASE_STUDY, PRIMARY_LABEL, period="holdout")
print("Holdout strategy vs holdout EW universe:")
print(f"  strategy Sharpe:  {ho_full['sharpe']:.3f}")
print(f"  EW Sharpe:        {ew_ho['sharpe']:.3f}")
print(
    "  diff Sharpe: "
    f"{_fmt_ci(he['sharpe_diff'], he['sharpe_diff_ci95_lo'], he['sharpe_diff_ci95_hi'])}"
)
print(f"  p_value:                {he['p_value']:.4f}")
print(f"  prob_challenger_wins:   {he['prob_challenger_wins']:.3f}")
print(
    f"  info_ratio (strategy vs EW): "
    f"{_fmt_ci(he['info_ratio'], he['info_ratio_ci95_lo'], he['info_ratio_ci95_hi'])}"
)
print(f"  CI status: {ci_status(he['sharpe_diff_ci95_lo'], he['sharpe_diff_ci95_hi'])}")

# %% [markdown]
# **Decay reading (val_rank1_self pair):** holdout Sharpe is more
# negative than validation Sharpe; the paired diff CI excludes zero on
# the negative side (live numbers from the cell above). The val→holdout
# decay is statistically resolved as negative under bootstrap-paired
# draws. The interpretation is unchanged from the validation reading:
# at 15-minute cadence the residual cross-sectional signal is dominated
# by friction, and the position-level overlay slows but does not
# reverse the bleed.
#
# **Strategy vs EW benchmark on holdout:** the rank-1 strategy
# underperforms EW NQ100 on the holdout with a diff-Sharpe CI on the
# negative side. The universal kill gate 2 (holdout strategy-vs-EW CI
# does not exclude zero negatively) FAILs on this lineage; gate 1
# (val Sharpe CI lower bound ≥ 0) also fails. Both pair rows enter
# Ch20 as `excludes_zero_strong` decay classifications on the negative
# side. The CS reads as a methodology study — what 15-minute equity
# microstructure looks like under honest accounting — not as a
# deployable strategy.

# %% [markdown]
# ## §7 Benchmark-aware diagnostics
#
# Layer 1 reports the universal alpha/beta/IR profile via
# `PortfolioAnalysis` against the equal-weight NASDAQ-100 universe.
# Layer 2 — equity factor attribution (FF5+MOM with HAC standard errors)
# — applies because this is an equity-class case study; for a
# dollar-neutral intraday strategy on the NQ100, factor attribution
# answers whether residual α survives standard equity-style risk
# decomposition.

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
print("Layer 1: rank-1 vs validation EW NASDAQ-100 universe (PortfolioAnalysis):")
print(attr_df)

# %%
# Placebo regression: residual α and HAC t-stat from regressing on EW alone.
import statsmodels.api as sm

X = sm.add_constant(bench_arr)
ols = sm.OLS(strat_arr, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
alpha_daily = ols.params[0]
alpha_t = ols.tvalues[0]
alpha_p = ols.pvalues[0]
beta_ew = ols.params[1]
beta_t = ols.tvalues[1]
print()
print("Placebo regression (HAC, maxlags=5):")
print(f"  α (daily) = {alpha_daily:.6f}, α (annualized) = {alpha_daily * PERIODS_PER_YEAR:.4f}")
print(f"  α t-stat = {alpha_t:.3f}, p = {alpha_p:.3f}")
print(f"  β        = {beta_ew:.3f}, β t-stat = {beta_t:.3f}")
print(f"  CI status (α): {'excludes_zero_strong' if alpha_p < 0.05 else 'straddles_zero'}")

# %%
# Layer 2: FF5+MOM factor attribution with HAC standard errors.
# Strategy returns are dollar-neutral (long top-K minus short bottom-K),
# so we regress raw daily returns rather than excess returns.
strategy_rets = pd.Series(
    strat_arr,
    index=pd.to_datetime(aligned["ts"].to_list()),
    name="strategy",
)
factor_start = str(strategy_rets.index.min().date())
factor_end = str(strategy_rets.index.max().date())
factor_data = load_factor_data(start=factor_start, end=factor_end)
reg = run_factor_regression(
    strategy_rets, factor_data, model="ff5_mom", hac_lags=5, dollar_neutral=True
)

print()
print("Layer 2: FF5+MOM regression (HAC, dollar-neutral)")
print(f"  Observations:    {reg['n_obs']}")
print(f"  Strategy Sharpe: {reg['strategy_sharpe']:+.3f}")
print(f"  Residual Sharpe: {reg['residual_sharpe']:+.3f}")
print(f"  R²:              {reg['r_squared']:.3f}")
print(
    f"  Alpha (ann.):    {reg['alpha_annualized']:+.4f} "
    f"(t={reg['alpha_t_stat']:+.2f}, p={reg['alpha_p_value']:.3f})"
)
print()
print("Factor betas (significant at p<0.05 marked *):")
for factor, beta in reg["betas"].items():
    sig = "*" if reg["p_values"][factor] < 0.05 else ""
    print(f"  {factor:8s}: {beta:+.4f}  (t={reg['t_stats'][factor]:+.2f}){sig}")

# %%
# Block bootstrap CIs for alpha and factor betas
boot = compute_bootstrap_ci(
    strategy_rets,
    factor_data,
    model="ff5_mom",
    n_boot=1000,
    block_size=20,
    dollar_neutral=True,
    seed=42,
)
if boot.get("n_boot", 0) > 0:
    print(f"Bootstrap CIs (n={boot['n_boot']}, block=20 days):")
    print(f"  Alpha (ann.): [{boot['alpha_ann_lo']:+.4f}, {boot['alpha_ann_hi']:+.4f}]")
    for factor in reg["factor_columns"]:
        lo_key, hi_key = f"{factor}_lo", f"{factor}_hi"
        if lo_key in boot:
            print(f"  {factor:8s}: [{boot[lo_key]:+.4f}, {boot[hi_key]:+.4f}]")

# %%
# Rolling factor exposures (63-day window ≈ 3 months)
rolling = compute_rolling_exposures(
    strategy_rets, factor_data, model="ff5_mom", window=63, dollar_neutral=True
)
fig_roll = plot_rolling_exposures(
    rolling, title="NQ100 Strategy: Rolling Factor Exposures (63-day, FF5+MOM)"
)
fig_roll.show()

# %%
# Attribution waterfall
fig_attr = plot_attribution_waterfall(reg, title="NQ100 Strategy: Factor Attribution")
fig_attr.show()

attr_summary = format_attribution_summary(reg, boot)

# %% [markdown]
# Layer-1 Layer-2 results read consistently with §3: the rank-1
# strategy's α is deeply negative on both the placebo (EW-only)
# regression and the FF5+MOM regression, and the residual Sharpe stays
# negative after factor decomposition. The strategy did not learn a
# factor proxy — there is nothing for the factor model to absorb. The
# FF5+MOM betas surface whether the long-minus-short construction
# produced inadvertent factor tilts; in this case the loadings are
# small relative to the magnitude of the negative α, and any apparent
# tilt is dominated by friction. The block bootstrap CI on α widens
# the parametric HAC interval, but the lower bound remains far below
# zero — there is no inference under which this strategy carried
# positive risk-adjusted alpha during the validation window.

# %% [markdown]
# ## §8 Strategy tear sheet
#
# The diagnostic library renders the rank-1 lineage's full tear sheet
# directly from the on-disk artifacts; the validation-window
# equal-weight NASDAQ-100 benchmark is wired in as the comparison
# series. The tear sheet HTML is written under `OUTPUT_DIR` and is
# gitignored — readers regenerate it locally.

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

# generate_tearsheet_from_run_artifacts expects a 1-D series of benchmark
# returns, not a DataFrame; pass the daily-aggregated EW series.
bench_series = aligned["benchmark"].to_numpy()

meta = BacktestReportMetadata(
    title="NASDAQ-100 Microstructure — Rank-1 Lineage",
    strategy_name=f"{RANK1_FAMILY}/{RANK1_CONFIG} — {PRIMARY_LABEL}",
    universe="100 NASDAQ-100 constituents at 15-minute cadence",
    benchmark_name="NQ100 equal-weight universe (validation window)",
    evaluation_window=f"{aligned['ts'].min()} to {aligned['ts'].max()}",
    calendar=setup["evaluation"]["calendar"],
)

tear_path = OUTPUT_DIR / f"{CASE_STUDY}_tearsheet.html"
html = generate_tearsheet_from_run_artifacts(
    backtest_dir,
    template="risk_manager",
    benchmark=bench_series,
    benchmark_name="EW NQ100 universe (validation)",
    calendar=setup["evaluation"]["calendar"],
    report_metadata=meta,
    output_path=str(tear_path),
)
print(f"Tear sheet written to: {tear_path}")
print(f"HTML size: {len(html):,} bytes")

# %% [markdown]
# ## §9 Pre-Ch20 judgment & handoff
#
# This section is the explicit hand-off point to Chapter 20. Numbers
# below stay strictly inside nasdaq100_microstructure — cross-case-study
# comparison is Ch20's lane.

# %%
op_profile = compute_operating_profile(lineage, setup)
# nasdaq100_microstructure setup uses `decision.bar_frequency`
# (15_minute) as the rebalance cadence; the helper inspects
# evaluation_protocol.rebalance_frequency, which is absent. Override.
op_profile = op_profile.with_columns(
    pl.when(pl.col("property") == "Trading cadence")
    .then(pl.lit(setup["decision"]["bar_frequency"]))
    .otherwise(pl.col("value"))
    .alias("value")
)
print("Operating profile (nasdaq100_microstructure, validation window):")
print(op_profile)
print()
print(
    f"Sharpe: {_fmt_ci(val_full['sharpe'], val_full['sharpe_ci95_lo'], val_full['sharpe_ci95_hi'])}"
)
print(f"Info ratio (vs EW val): {_fmt_ci(getattr(metrics, 'information_ratio', None), None, None)}")
print(f"Max drawdown: {val_full['max_drawdown']:.3f}")

# %%
# Kill-condition assessment (universal gates — see §1)
gate1_status = gate1_validation_sharpe_geq_zero(val_full["sharpe_ci95_lo"])
_gate1_phrase = {
    "pass": "≥ 0 (passes)",
    "fail": "< 0 (fails)",
    "no_data": "CI unavailable",
}[gate1_status]
gate1_evidence = (
    f"Sharpe CI lower bound = {_fmt(val_full['sharpe_ci95_lo'], '.3f')} ({_gate1_phrase})"
)
gate2_ci_status = ci_status(he["sharpe_diff_ci95_lo"], he["sharpe_diff_ci95_hi"])
gate2_status = gate2_holdout_diff_not_excludes_zero_negatively(gate2_ci_status, he["sharpe_diff"])
gate2_evidence = (
    f"Holdout strategy vs EW diff-Sharpe = "
    f"{_fmt_ci(he['sharpe_diff'], he['sharpe_diff_ci95_lo'], he['sharpe_diff_ci95_hi'])} "
    f"({gate2_ci_status})"
)

print("Kill-condition gates:")
print(f"  [{fmt_gate(gate1_status)}] Validation Sharpe CI lower bound ≥ 0:")
print(f"      {gate1_evidence}")
print(f"  [{fmt_gate(gate2_status)}] Holdout strategy CI does not exclude zero negatively:")
print(f"      {gate2_evidence}")

# %% [markdown]
# **What §3–§6 say.** The cross-stage rank-1 lineage is the live
# carrier printed in §3 above — a position-level overlay on a
# `score_weighted` allocation of a long-horizon linear signal. Every
# cost-grid cell is negative across the 11-level bps surface for the
# rank-1 prediction. Every risk-overlay cell is negative; the rank-1
# is the least-bad combination (longest time-exit on the cleanest
# signal horizon). Validation Sharpe CI lower bound stays below zero
# (kill gate 1 FAILs); holdout decays further and underperforms EW
# NQ100 by a margin whose CI excludes zero on the negative side
# (kill gate 2 FAILs). Selection-bias-adjusted DSR cannot resolve
# any cell above zero.
#
# **The case study's reading.** Intraday NQ100 microstructure under
# honest accounting (per-share + half-spread cost dispatch at the
# realistic NQ100 spread band of 1–3 bps for large caps and 3–8 bps
# for mid caps, integer share sizing, daily MTM despite intraday
# signal cadence) does not produce a deployable long-short
# cross-sectional equity strategy at 15-minute cadence. Position-
# level overlays modulate the magnitude of the loss but do not
# reverse it. This is a methodology study, not a candidate strategy
# — it demonstrates what friction-dominance looks like at intraday
# cadence under honest accounting.
#
# **What this analysis does not say.** Engine costs are 5 bps
# commission + 2 bps slippage; the realistic NQ100 spread band is
# 1–3 bps large-cap, 3–8 bps mid-cap, and the per-share
# `$0.0035 + half-spread` model in `setup.yaml` is the production
# cost dispatch. The holdout window of 128 trading days is short
# relative to the bootstrap block length, so the diff-Sharpe CI
# widths reflect both the disjoint window structure and the
# regime-dependent strategy reading. No sector-cap or
# position-limit overlay is applied; a production version would
# constrain per-sector exposure given the NQ100's technology
# concentration. The conformal-weighted holdout sibling (if
# registered) is recorded as a transparency variant against the
# canonical baseline holdout; it does not replace the rank-1
# holdout closure analyzed above.

# %% [markdown]
# **Forward pointer to Ch20.** This case study contributes the
# intraday-cadence friction-dominance datapoint to Ch20 nb01's
# cross-CS rank-1-Sharpe + holdout-decay aggregation. The §6 decay
# magnitude (validation rank-1 negative; holdout decays further;
# both pair rows excludes_zero_strong on the negative side) and the
# strategy-vs-holdout-EW read (excludes_zero_strong on the negative
# side) feed Ch20 nb04's cost-survival comparison. The §5 cadence ×
# cost surface (3 labels × 11 cost levels; no cell positive at
# credibility-resolved level) and the §5.2 risk-overlay sweep
# (position-level overlays only; no portfolio kill switch swept;
# rank-1 cell still negative on point estimate) feed Ch20 nb04's
# cost-survival comparison and nb05's regime risk layer.

# %%
# Strategy-assessment JSON: extend the existing schema with strategy-analysis notebook fields.
search_ctx = ctx
assessment = {
    "case_study": CASE_STUDY,
    "primary_label": PRIMARY_LABEL,
    "spine_version": "v1",
    "rank1": {
        "family": RANK1_FAMILY,
        "config_name": RANK1_CONFIG,
        "label": RANK1_LABEL,
        "prediction_hash": TOP_PHASH,
        "validation_backtest_hash": TOP_HASH,
        "validation_backtest_hash_cohort_leader": _cohort_leader_hash,
        "holdout_backtest_hash": HO_HASH,
        "holdout_backtest_hash_conformal_sibling": HO_HASH_CONFORMAL,
        "holdout_sharpe_conformal_sibling": (
            float(HO_SHARPE_CONFORMAL) if HO_SHARPE_CONFORMAL is not None else None
        ),
    },
    "cohort_leader_sibling": (
        {
            "cohort_stage": _cohort_stage,
            "cohort_leader_hash": _cohort_leader_hash,
            "cohort_leader_sharpe": (
                float(_cohort_leader_sharpe) if _cohort_leader_sharpe is not None else None
            ),
            "spine_rank1_sharpe": val_full["sharpe"],
            "divergence_reason": (
                "cross_stage_alignment_library_limit: rank-1 stage cohort row absent; "
                f"cascade resolved to family/{_cohort_stage}/{RANK1_LABEL}/{RANK1_FAMILY}"
                if (_cohort_stage and _cohort_stage != _stage_for_rank1)
                else "rank1_is_cohort_stage_match"
            ),
        }
        if _cohort_leader_hash is not None
        else None
    ),
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
        "architecture": "cohort_metrics",
        "cohort_cascade_stage": _cohort_stage,
        "cohort_cascade_fallback_used": (
            _cohort_stage != _stage_for_rank1 if _cohort_stage else None
        ),
        "cohort_leader_hash": _cohort_leader_hash,
        "cohort_leader_sharpe": (
            float(_cohort_leader_sharpe) if _cohort_leader_sharpe is not None else None
        ),
        "k_variants": int(val_full["k_variants"]) if val_full["k_variants"] is not None else None,
        "n_trials_effective_mp": (
            float(val_full.get("n_trials_effective_mp"))
            if val_full.get("n_trials_effective_mp") is not None
            else None
        ),
        "n_trials_effective_er": (
            float(val_full.get("n_trials_effective_er"))
            if val_full.get("n_trials_effective_er") is not None
            else None
        ),
        "dsr_er": val_full.get("dsr_er"),
        "dsr_er_pvalue": val_full.get("dsr_er_pvalue"),
        "dsr_mp": val_full.get("dsr_mp"),
        "dsr_mp_pvalue": val_full.get("dsr_mp_pvalue"),
        "dsr_raw": val_full.get("dsr_raw"),
        "dsr_raw_pvalue": val_full.get("dsr_raw_pvalue"),
        "expected_max_sharpe_er": val_full.get("expected_max_sharpe_er"),
        "expected_max_sharpe_mp": val_full.get("expected_max_sharpe_mp"),
        "expected_max_sharpe_raw": val_full.get("expected_max_sharpe_raw"),
        "min_trl_periods_er": (
            None
            if val_full.get("min_trl_periods_er") in (None, float("inf"))
            else val_full.get("min_trl_periods_er")
        ),
        "pbo": val_full.get("pbo"),
        "pbo_n_combinations": val_full.get("pbo_n_combinations"),
        "pbo_n_folds": val_full.get("pbo_n_folds"),
    },
    "benchmark_relative": {
        "benchmark_name": "equal_weight_universe",
        "benchmark_validation_sharpe": ew_val["sharpe"],
        "benchmark_holdout_sharpe": ew_ho["sharpe"],
        "alpha_annualized_placebo": float(alpha_daily * PERIODS_PER_YEAR),
        "alpha_t_hac": float(alpha_t),
        "beta_to_ew": float(beta_ew),
        "factor_attribution": {
            "status": "applicable",
            "model": "ff5_mom_hac",
            "n_obs": int(reg["n_obs"]),
            "alpha_annualized": float(reg["alpha_annualized"]),
            "alpha_t_hac": float(reg["alpha_t_stat"]),
            "alpha_p_value": float(reg["alpha_p_value"]),
            "residual_sharpe": float(reg["residual_sharpe"]),
            "r_squared": float(reg["r_squared"]),
            "betas": {f: float(b) for f, b in reg["betas"].items()},
            "classification": attr_summary.get("classification"),
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
        "vs_ew_holdout": {
            "sharpe_diff": he["sharpe_diff"],
            "sharpe_diff_ci95_lo": he["sharpe_diff_ci95_lo"],
            "sharpe_diff_ci95_hi": he["sharpe_diff_ci95_hi"],
            "p_value": he["p_value"],
            "ci_status": ci_status(he["sharpe_diff_ci95_lo"], he["sharpe_diff_ci95_hi"]),
        },
    },
    "search_context": {
        "total_signal_backtests": int(search_ctx["total"]),
        "median_sharpe": search_ctx["median_sharpe"],
        "p90_sharpe": search_ctx["p90_sharpe"],
        "pct_positive": search_ctx["pct_positive"],
    },
    "cost_sensitivity": {
        "grid_shape": {
            "labels": label_order,
            "n_cost_levels": int(cost_curve_per_label["cost_bps"].n_unique()),
            "configs_per_cell": 3,
            "total_rows": int(cost_df.height),
        },
        "per_label": [
            {
                "label": lbl,
                "best_config_at_zero_cost": {
                    "family": cost_curve_per_label.filter(
                        (pl.col("label") == lbl) & (pl.col("cost_bps") == 0)
                    ).row(0, named=True)["best_family"],
                    "config_name": cost_curve_per_label.filter(
                        (pl.col("label") == lbl) & (pl.col("cost_bps") == 0)
                    ).row(0, named=True)["best_config"],
                    "sharpe": float(
                        cost_curve_per_label.filter(
                            (pl.col("label") == lbl) & (pl.col("cost_bps") == 0)
                        ).row(0, named=True)["sharpe_max"]
                    ),
                    "ci_lo": float(
                        cost_curve_per_label.filter(
                            (pl.col("label") == lbl) & (pl.col("cost_bps") == 0)
                        ).row(0, named=True)["best_ci_lo"]
                    ),
                    "ci_hi": float(
                        cost_curve_per_label.filter(
                            (pl.col("label") == lbl) & (pl.col("cost_bps") == 0)
                        ).row(0, named=True)["best_ci_hi"]
                    ),
                    "num_trades": float(
                        cost_curve_per_label.filter(
                            (pl.col("label") == lbl) & (pl.col("cost_bps") == 0)
                        ).row(0, named=True)["best_n_trades"]
                    ),
                    "avg_turnover": float(
                        cost_curve_per_label.filter(
                            (pl.col("label") == lbl) & (pl.col("cost_bps") == 0)
                        ).row(0, named=True)["best_avg_turnover"]
                    ),
                },
                "breakeven": {
                    "point_estimate_max_bps": (
                        None
                        if cost_curve_per_label.filter(
                            (pl.col("label") == lbl) & (pl.col("sharpe_max") > 0)
                        ).is_empty()
                        else float(
                            cost_curve_per_label.filter(
                                (pl.col("label") == lbl) & (pl.col("sharpe_max") > 0)
                            )["cost_bps"].max()
                        )
                    ),
                    "ci_aware_max_bps": (
                        None
                        if cost_curve_per_label.filter(
                            (pl.col("label") == lbl) & (pl.col("best_ci_lo") > 0)
                        ).is_empty()
                        else float(
                            cost_curve_per_label.filter(
                                (pl.col("label") == lbl) & (pl.col("best_ci_lo") > 0)
                            )["cost_bps"].max()
                        )
                    ),
                },
            }
            for lbl in label_order
            if not cost_curve_per_label.filter(
                (pl.col("label") == lbl) & (pl.col("cost_bps") == 0)
            ).is_empty()
        ],
        "realistic_friction_setup_yaml": dict(setup["costs"]),
    },
    "risk_overlay_sensitivity": {
        "grid_shape": {
            "labels": label_order,
            "n_overlays_per_label": 81,
            "total_rows": int(risk_df.height),
        },
        "best_per_label": [
            {
                "label": row["label"],
                "no_overlay_baseline_sharpe": float(row["best_signal_sharpe"]),
                "best_overlay_name": row["risk_name"],
                "best_overlay_sharpe": float(row["best_overlay_sharpe"]),
                "ci_lo": float(row["sharpe_ci95_lo"]),
                "ci_hi": float(row["sharpe_ci95_hi"]),
                "ci_status": ci_status(row["sharpe_ci95_lo"], row["sharpe_ci95_hi"]),
                "max_drawdown": float(row["max_drawdown"]),
                "num_trades": float(row["num_trades"]),
                "rescue_sharpe_units": float(
                    row["best_overlay_sharpe"] - row["best_signal_sharpe"]
                ),
            }
            for row in rescue_table.to_dicts()
        ],
        "best_overlay_family_per_label": [
            {
                "label": row["label"],
                "risk_family": row["risk_family"],
                "best_overlay": row["best_overlay"],
                "best_sharpe": float(row["best_sharpe"]),
                "ci_lo": float(row["best_ci_lo"]),
                "ci_hi": float(row["best_ci_hi"]),
            }
            for row in risk_summary.to_dicts()
        ],
    },
    "kill_gates": {
        "validation_sharpe_ci_lower_bound_geq_zero": gate_passes(gate1_status),
        "holdout_vs_ew_ci_does_not_exclude_zero_negatively": gate_passes(gate2_status),
    },
    "ch20_handoff": {
        "contributes_to": [
            "Ch20 nb01 — cross-CS rank-1 Sharpe and holdout-decay aggregation",
            "Ch20 nb04 — cost-survival comparison (cadence × cost grid + overlay rescue)",
            "Ch20 nb05 — regime risk layer (position-level overlays modulate magnitude; do not reverse loss)",
        ],
        "asset_class_label": "long_short_equity_intraday",
        "rebalance_step_bars": setup["labels"]["rebalance_step"][PRIMARY_LABEL],
    },
}
assessment_path = write_strategy_assessment(CASE_STUDY, assessment)
print(f"strategy_assessment.json written to: {assessment_path}")
