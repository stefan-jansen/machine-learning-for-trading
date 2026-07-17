"""Strategy analysis figure helpers and assessment writer.

Companion to ``BacktestExplorer`` — produces the figures and structured
artifacts for each case study's ``strategy_analysis.py`` notebook.

Usage::

    from case_studies.utils.strategy_analysis import (
        plot_ic_vs_sharpe,
        plot_sharpe_waterfall,
        plot_concentration_curve,
        plot_cost_decay,
        plot_equity_drawdown,
        load_holdout_metrics,
        write_strategy_assessment,
        load_strategy_assessment,
    )
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from case_studies.utils.notebook_contracts import degenerate_prediction_sql

# ---------------------------------------------------------------------------
# Canonical rank-1 resolution (LABEL_RESTRICTIONS-aware)
# ---------------------------------------------------------------------------
#
# Per-CS whitelist of labels eligible to anchor the registered strategy. The
# only entry today is sp500_options, restricted to ret_to_expiry because the
# four legacy diagnostic variants (fwd_ret_5d, fwd_ret_10d, fwd_ret_dh_5d,
# fwd_ret_dh_10d) were dropped from the sweep + registry 2026-05-17 — they
# went through the vectorized backtest path which treats their 5d/10d
# forward returns as daily returns, inflating Sharpes (e.g. fwd_ret_10d
# allocation Sharpe ~6.5) to non-credible levels. ret_to_expiry runs through
# the HTM daily-MTM cohort path and is the only label with an honest cost
# model for this CS. Mirrors the canonical definition in
# 20_strategy_synthesis/holdout.py::LABEL_RESTRICTIONS — keep these in sync.
LABEL_RESTRICTIONS: dict[str, frozenset[str]] = {
    "sp500_options": frozenset({"ret_to_expiry"}),
}


# Per-CS canonical universe pin: case_study -> strategy.signal.universe_filter
# value eligible to anchor the registered rank-1. sp500_options trades only the
# liquid (bottom-quintile half-spread) subset — the full-universe round-trip
# option spread consumes the variance-risk-premium edge, so full-universe rows
# are excluded from rank-1 selection (the full universe is retained only for the
# Ch18 htm_cost_cascade comparison, never as the deployed carrier). Without this
# pin, full-universe allocation backtests registered by the standard sweep
# (e.g. the 2026-05-31 L1-grid rollout) leak into rank-1 by raw Sharpe and
# orphan the liquid-lineage holdout. Mirrored in 20_strategy_synthesis/holdout.py
# (select_best_models) — keep in sync.
UNIVERSE_RESTRICTIONS: dict[str, str] = {
    "sp500_options": "liquid",
}


# Per-CS carrier pin: case_study -> validation backtest_hash (prefix) to deploy as
# the canonical carrier when the cross-stage val rank-1 is statistically tied with a
# more diversified / more precisely-estimated configuration. This is a documented
# a-priori (validation-time) tie-break, NOT a holdout-based selection.
#
# us_firm_characteristics: validation Sharpe ties at ~2.75 between
#   A = leaves_7_mae / score_weighted (cross-stage rank-1, 2.7589) and
#   B = default_huber / equal_weight  (signal-stage rank-1, 2.7542).
# Block-bootstrap Sharpe CIs (backtest_metrics): B [2.33, 3.37] width 1.04 vs
# A [2.10, 3.57] width 1.46 — B is estimated ~29% more precisely (lower vol, 50
# equal-weight names vs 10 score-weighted). B is also far more diversified (holdout
# MaxDD -8.6% vs -34%). Both criteria are validation-time, so B is pinned as the
# deployed carrier. The pinned row is default_huber/equal_weight t50 at the
# allocation stage; regenerate by re-querying the validation allocation rank-1 with
# config_name='default_huber' AND allocation.method='equal_weight' if the sweep is
# rebuilt. Keep in sync with 20_strategy_synthesis/01_aggregate_synthesis.py, which
# imports this dict to pin the §20.5 / Figure 20.7 allocator-comparison spine.
CARRIER_PINS: dict[str, str] = {
    "us_firm_characteristics": "e676e1989e1f",
}


def select_holdout_self_backtest(
    case_study: str,
    val_backtest_hash: str,
) -> str | None:
    """Return the holdout backtest_hash whose strategy spec exactly matches
    the given validation rank-1 backtest's strategy spec.

    This is the canonical ``val_rank1_self`` lineage anchor for the §6
    holdout closure: the holdout backtest produced by replaying the val
    rank-1 strategy on the holdout prediction set. Matching by strategy
    spec (rather than by max-Sharpe over candidates sharing the
    ``training_hash``) keeps the lookup robust against experimental
    side-channel allocators — most importantly ``conformal_weighted`` —
    that may share the holdout pred set but diverge from val rank-1's
    allocator method. Without this guard, an allocator variant whose
    holdout Sharpe happens to exceed the canonical lineage's silently
    displaces the §6 anchor and the ``backtest_paired_metrics``
    ``val_rank1_self`` pair (written against the canonical lineage's
    holdout hash) goes unfound by the reader.

    Returns ``None`` when no matching holdout backtest exists.
    """
    import sqlite3

    from utils.paths import get_case_study_dir

    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    with sqlite3.connect(str(db_path)) as db:
        row = db.execute(
            "SELECT prediction_hash, spec_json FROM backtest_runs WHERE backtest_hash = ?",
            (val_backtest_hash,),
        ).fetchone()
        if row is None:
            return None
        val_pred_hash, val_spec_json = row
        val_strategy = json.loads(val_spec_json).get("strategy", {})

        train_row = db.execute(
            "SELECT training_hash FROM prediction_sets WHERE prediction_hash = ?",
            (val_pred_hash,),
        ).fetchone()
        if train_row is None:
            return None
        training_hash = train_row[0]

        candidates = db.execute(
            """
            SELECT b.backtest_hash, b.spec_json
            FROM backtest_runs b
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            LEFT JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            WHERE p.training_hash = ? AND p.split = 'holdout'
            ORDER BY bm.sharpe DESC NULLS LAST
            """,
            (training_hash,),
        ).fetchall()

    for bh, spec_json in candidates:
        candidate_strategy = json.loads(spec_json).get("strategy", {})
        if candidate_strategy == val_strategy:
            return bh
    return None


def resolve_canonical_rank1_lineage(case_study: str) -> dict[str, Any]:
    """Resolve the canonical val rank-1 + matching holdout for a case study.

    Cross-stage val rank-1 = max(sharpe) over stage IN (signal, allocation,
    risk_overlay) on split='validation', with LABEL_RESTRICTIONS applied for
    case studies that have one. Holdout match is by training_hash on the
    rank-1's prediction set. Use this in every strategy_analysis notebook
    rather than hardcoding hashes — hardcoded hashes go stale every time the
    sweep is rebuilt, and queries that forget LABEL_RESTRICTIONS surface the
    diagnostic-variant rows (sp500_options' fwd_ret_10d Sharpe ≈ 9.7) as
    bogus rank-1 candidates.

    Returns a dict with keys ``val_backtest_hash``, ``val_prediction_hash``,
    ``val_stage``, ``val_sharpe``, ``training_hash``, ``family``,
    ``config_name``, ``label``, ``holdout_backtest_hash``,
    ``holdout_prediction_hash``, ``holdout_sharpe`` (holdout fields are
    None when no matching holdout row exists yet).
    """
    import sqlite3

    from utils.paths import get_case_study_dir

    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    label_filter = LABEL_RESTRICTIONS.get(case_study)
    universe_pin = UNIVERSE_RESTRICTIONS.get(case_study)
    carrier_pin = CARRIER_PINS.get(case_study)

    base_select = """
        SELECT b.backtest_hash, b.prediction_hash, b.stage,
               t.training_hash, t.family, t.config_name, t.label,
               bm.sharpe
        FROM backtest_runs b
        JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
        JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
        JOIN training_runs t ON t.training_hash = p.training_hash
    """

    if carrier_pin:
        # Documented a-priori carrier pin: resolve directly to the pinned
        # validation backtest rather than the max-Sharpe cross-stage rank-1.
        # See CARRIER_PINS for the rationale (statistical tie broken on CI
        # width + diversification at validation time).
        val_sql = base_select + (
            " WHERE b.backtest_hash LIKE ?"
            " AND p.split = 'validation'"
            " AND bm.sharpe IS NOT NULL"
            + degenerate_prediction_sql("p.prediction_hash")
            + " ORDER BY bm.sharpe DESC LIMIT 1"
        )
        params: tuple = (carrier_pin + "%",)
    else:
        val_sql = base_select + (
            " WHERE b.stage IN ('signal','allocation','risk_overlay','holdout')"
            " AND p.split = 'validation'"
            " AND bm.sharpe IS NOT NULL"
            " AND t.family != 'benchmark'" + degenerate_prediction_sql("p.prediction_hash")
        )
        params = ()
        if label_filter:
            placeholders = ",".join("?" for _ in label_filter)
            val_sql += f" AND t.label IN ({placeholders})"
            params = tuple(label_filter)
        if universe_pin:
            val_sql += " AND json_extract(b.spec_json, '$.strategy.signal.universe_filter') = ?"
            params = params + (universe_pin,)
        # Tie-break: among rows with identical Sharpe (e.g. the signal-stage
        # equal-weight selection and its economically identical equal_weight
        # allocation-stage re-run, which share a prediction), prefer the
        # signal-only spec (no allocation block). That is the spec the holdout
        # is replayed from, so the canonical lineage stays poolable with its
        # holdout. Final ``backtest_hash`` key makes the order deterministic.
        val_sql += (
            " ORDER BY bm.sharpe DESC,"
            " (json_extract(b.spec_json, '$.strategy.allocation') IS NULL) DESC,"
            " b.backtest_hash ASC LIMIT 1"
        )

    db = sqlite3.connect(str(db_path))
    try:
        val = db.execute(val_sql, params).fetchone()
        if val is None:
            raise RuntimeError(
                f"No validation rank-1 candidate for {case_study} (label_filter={label_filter})"
            )
        (val_bh, val_ph, val_stage, train_h, family, config_name, label, val_sharpe) = val

    finally:
        db.close()

    # Match holdout by strategy spec to the val rank-1 backtest, so an
    # experimental side-channel allocator (e.g., conformal_weighted) on
    # the same holdout pred set does not displace the canonical lineage.
    ho_bh = select_holdout_self_backtest(case_study, val_bh)
    ho_ph: str | None = None
    ho_sharpe: float | None = None
    if ho_bh is not None:
        db = sqlite3.connect(str(db_path))
        try:
            ho_row = db.execute(
                """
                SELECT b.prediction_hash, bm.sharpe
                FROM backtest_runs b
                LEFT JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
                WHERE b.backtest_hash = ?
                """,
                (ho_bh,),
            ).fetchone()
        finally:
            db.close()
        if ho_row is not None:
            ho_ph, ho_sharpe = ho_row

    return {
        "val_backtest_hash": val_bh,
        "val_prediction_hash": val_ph,
        "val_stage": val_stage,
        "val_sharpe": val_sharpe,
        "training_hash": train_h,
        "family": family,
        "config_name": config_name,
        "label": label,
        "holdout_backtest_hash": ho_bh,
        "holdout_prediction_hash": ho_ph,
        "holdout_sharpe": ho_sharpe,
    }


# ---------------------------------------------------------------------------
# Spine CI / kill-gate helpers (tri-state contract)
# ---------------------------------------------------------------------------

CIStatus = Literal["excludes_zero_strong", "straddles_zero", "no_data"]
GateStatus = Literal["pass", "fail", "no_data"]


def _missing(x: float | None) -> bool:
    """True when a bound/estimate carries no information.

    NaN reaches these gates whenever a paired bootstrap could not be computed
    (no pair row in ``backtest_paired_metrics``). It must be treated exactly
    like ``None``: every comparison against NaN is False, so an unguarded NaN
    silently falls through to ``pass``.
    """
    return x is None or not np.isfinite(x)


def ci_status(lo: float | None, hi: float | None) -> CIStatus:
    """Three-tier CI continuum used uniformly across spine §3 / §6 / §7.

    `no_data` is reserved for missing CI bounds (upstream bootstrap not run
    or registry NULLs); it is *not* a low-credibility classification.
    """
    if _missing(lo) or _missing(hi):
        return "no_data"
    if lo > 0 or hi < 0:
        return "excludes_zero_strong"
    return "straddles_zero"


def gate1_validation_sharpe_geq_zero(sharpe_ci_lo: float | None) -> GateStatus:
    """Kill gate 1: validation full-period Sharpe CI lower bound ≥ 0.

    Returns ``no_data`` when the CI lower bound is missing.
    """
    if _missing(sharpe_ci_lo):
        return "no_data"
    return "pass" if sharpe_ci_lo >= 0 else "fail"


def gate2_holdout_diff_not_excludes_zero_negatively(
    diff_ci_status: CIStatus, sharpe_diff: float | None
) -> GateStatus:
    """Kill gate 2: holdout strategy-vs-EW Sharpe-diff CI does not exclude
    zero on the negative side.

    Pass: diff CI does not strongly exclude zero, OR strongly excludes zero
    on the positive side. Fail: diff CI strongly excludes zero AND the
    point estimate is negative. ``no_data`` when the diff CI status is
    ``no_data`` or ``sharpe_diff`` is missing.
    """
    if diff_ci_status == "no_data" or _missing(sharpe_diff):
        return "no_data"
    if diff_ci_status == "excludes_zero_strong" and sharpe_diff < 0:
        return "fail"
    return "pass"


def fmt_gate(status: GateStatus) -> str:
    """Display label for a gate status in printed kill-gate summaries."""
    return {"pass": "PASS", "fail": "FAIL", "no_data": "NO DATA"}[status]


def gate_passes(status: GateStatus) -> bool | None:
    """JSON-serializable view: True for pass, False for fail, None for
    no_data. Replaces ``bool(gate_pass)`` in ``strategy_assessment.json``
    so missing-CI cases are not silently coerced to True.
    """
    return {"pass": True, "fail": False, "no_data": None}[status]


# ---------------------------------------------------------------------------
# Holdout metrics loader
# ---------------------------------------------------------------------------


def load_holdout_metrics(case_study: str) -> dict[str, Any]:
    """Load holdout prediction + backtest metrics from the registry.

    Returns dict with keys: available, holdout_sharpe, holdout_ic,
    holdout_cagr, holdout_maxdd, family, config_name, label.
    All values are None if no holdout data exists.
    """
    import sqlite3

    from utils.paths import get_case_study_dir

    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    result: dict[str, Any] = {
        "available": False,
        "holdout_sharpe": None,
        "holdout_ic": None,
        "holdout_cagr": None,
        "holdout_maxdd": None,
        "family": None,
        "config_name": None,
        "label": None,
    }
    if not db_path.exists():
        return result

    db = sqlite3.connect(str(db_path))
    try:
        row = db.execute(
            """
            SELECT tr.family, tr.config_name, tr.label,
                   pm.ic_mean,
                   bm.sharpe, bm.cagr, bm.max_drawdown
            FROM prediction_sets ps
            JOIN training_runs tr ON ps.training_hash = tr.training_hash
            LEFT JOIN prediction_metrics pm
                ON ps.prediction_hash = pm.prediction_hash
            LEFT JOIN backtest_runs br
                ON ps.prediction_hash = br.prediction_hash AND br.stage = 'signal'
            LEFT JOIN backtest_metrics bm
                ON br.backtest_hash = bm.backtest_hash
            WHERE ps.split = 'holdout'
            ORDER BY bm.sharpe DESC NULLS LAST, pm.ic_mean DESC NULLS LAST
            LIMIT 1
            """,
        ).fetchone()
        if row:
            holdout_sharpe, holdout_cagr, holdout_maxdd = row[4], row[5], row[6]
            available = (
                holdout_sharpe is not None
                and holdout_cagr is not None
                and holdout_maxdd is not None
            )
            result.update(
                available=available,
                family=row[0],
                config_name=row[1],
                label=row[2],
                holdout_ic=row[3],
                holdout_sharpe=holdout_sharpe,
                holdout_cagr=holdout_cagr,
                holdout_maxdd=holdout_maxdd,
            )
    finally:
        db.close()
    return result


# ---------------------------------------------------------------------------
# Figure 1: IC vs Signal-Stage Sharpe
# ---------------------------------------------------------------------------


def plot_ic_vs_sharpe(
    explorer,
    *,
    highlight_sources: list[str] | None = None,
    ew_sharpe: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """IC vs signal-stage Sharpe scatter with annotations.

    Parameters
    ----------
    explorer : BacktestExplorer
    highlight_sources : list[str], optional
        Model sources to highlight (e.g. model_analysis recommendations).
    ew_sharpe : float, optional
        Equal-weight benchmark Sharpe (drawn as horizontal line).
    ax : plt.Axes, optional

    Returns
    -------
    plt.Figure
    """
    # Load all signal-stage backtests
    all_bt = explorer.best(stage="signal", top_n=9999)
    if all_bt.is_empty():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No signal backtests", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    ic = all_bt["ic_mean"].to_numpy()
    sharpe = all_bt["sharpe"].to_numpy()
    sources = all_bt["source"].to_list()
    families = all_bt["family"].to_list()

    # Base scatter (all points, light gray)
    ax.scatter(ic, sharpe, c="lightgray", s=20, alpha=0.5, zorder=1, label="_all")

    # Highlight recommended models
    if highlight_sources:
        mask = np.array([s in highlight_sources for s in sources])
        if mask.any():
            # Color by family
            family_colors = _family_color_map()
            highlighted_families = [families[i] for i in range(len(families)) if mask[i]]
            colors = [family_colors.get(f, "#333333") for f in highlighted_families]
            ax.scatter(
                ic[mask],
                sharpe[mask],
                c=colors,
                s=60,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            # Add family legend
            seen = set()
            for f in highlighted_families:
                if f not in seen:
                    ax.scatter([], [], c=family_colors.get(f, "#333333"), s=60, label=f)
                    seen.add(f)

    # Annotate top 3
    top_idx = np.argsort(sharpe)[-3:]
    for idx in top_idx:
        label = sources[idx].split("/")[-1]
        ax.annotate(
            label,
            (ic[idx], sharpe[idx]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            alpha=0.8,
        )

    # EW benchmark line
    if ew_sharpe is not None:
        ax.axhline(
            ew_sharpe,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"EW baseline ({ew_sharpe:.2f})",
        )

    ax.set_xlabel("Information Coefficient (IC)")
    ax.set_ylabel("Signal-Stage Sharpe")
    ax.set_title("Signal Quality vs Strategy Performance")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Sharpe Progression Waterfall (Locked Lineage)
# ---------------------------------------------------------------------------


def plot_sharpe_waterfall(
    lineage: dict[str, dict],
    *,
    ax: plt.Axes | None = None,
    ci_lo: dict[str, float] | None = None,
    ci_hi: dict[str, float] | None = None,
) -> plt.Figure:
    """Locked lineage waterfall: signal -> allocation -> cost -> risk.

    Parameters
    ----------
    lineage : dict
        From ``BacktestExplorer.champion_lineage()``.
    ax : plt.Axes, optional
    ci_lo, ci_hi : dict, optional
        Block-bootstrap 95% CI bounds keyed by stage name. When supplied,
        plotted as asymmetric error bars on each stage's bar.

    Returns
    -------
    plt.Figure
    """
    stage_order = ["signal", "allocation", "cost_sensitivity", "risk_overlay"]
    stage_labels = {
        "signal": "Signal",
        "allocation": "Allocation",
        "cost_sensitivity": "Costs",
        "risk_overlay": "Risk Overlay",
    }

    stages: list[str] = []
    stage_keys: list[str] = []
    sharpes: list[float] = []
    annotations: list[str] = []

    for s in stage_order:
        if s not in lineage:
            continue
        info = lineage[s]
        stages.append(stage_labels[s])
        stage_keys.append(s)
        sharpes.append(info["sharpe"])

        if s == "signal":
            method = info.get("signal_method", "")
            top_k = info.get("top_k", "")
            annotations.append(f"{method}\nk={top_k}" if top_k else method)
        elif s == "allocation":
            annotations.append(info.get("allocator", ""))
        elif s == "cost_sensitivity":
            cost = info.get("cost_bps", "?")
            annotations.append(f"{cost} bps")
        elif s == "risk_overlay":
            annotations.append(info.get("risk_name", ""))

    if not stages:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No lineage data", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    x = np.arange(len(stages))
    colors: list[str] = []
    for i in range(len(sharpes)):
        if i == 0:
            colors.append("#2196F3")
        elif sharpes[i] >= sharpes[i - 1]:
            colors.append("#4CAF50")
        else:
            colors.append("#F44336")

    bars = ax.bar(x, sharpes, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Track which CIs actually bracket the point estimate so the value
    # labels below anchor on the upper bar edge instead of a stale ``ci_hi``
    # that sits below the bar top.
    ci_brackets_point: set[int] = set()
    skipped_ci_stages: list[str] = []
    if ci_lo is not None and ci_hi is not None:
        err_lo = []
        err_hi = []
        valid_idx = []
        valid_centers = []
        for i, k in enumerate(stage_keys):
            lo = ci_lo.get(k)
            hi = ci_hi.get(k)
            if lo is None or hi is None:
                continue
            # Robustness: stale CIs from earlier engine runs may not
            # bracket the current point estimate. Skip those instead of
            # raising in matplotlib, but log the staleness so the
            # data-quality issue surfaces in notebook output rather than
            # only showing up as an absent error bar.
            if lo > sharpes[i] or hi < sharpes[i]:
                skipped_ci_stages.append(k)
                continue
            ci_brackets_point.add(i)
            valid_idx.append(i)
            err_lo.append(sharpes[i] - lo)
            err_hi.append(hi - sharpes[i])
            valid_centers.append(sharpes[i])
        if valid_idx:
            ax.errorbar(
                np.array(valid_idx),
                np.array(valid_centers),
                yerr=np.array([err_lo, err_hi]),
                fmt="none",
                ecolor="#333333",
                elinewidth=1.2,
                capsize=4,
                zorder=4,
            )
        if skipped_ci_stages:
            import warnings

            warnings.warn(
                "plot_sharpe_waterfall: dropped CIs not bracketing the "
                f"point estimate for stages={skipped_ci_stages}; rerun "
                "uncertainty backfill to refresh.",
                stacklevel=2,
            )

    # value labels — always above the upper edge so they don't overlap a CI bar
    for i, (bar, val) in enumerate(zip(bars, sharpes, strict=False)):
        # Only use ci_hi as the anchor when the CI actually brackets the
        # point estimate (see ci_brackets_point above); otherwise the
        # stale ``ci_hi`` can sit below ``val`` and pull the label inside
        # the bar.
        if i in ci_brackets_point:
            top = ci_hi[stage_keys[i]]
        else:
            top = max(val, 0)
        offset = max(abs(top) * 0.04, 0.02)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top + offset,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    for i in range(1, len(sharpes)):
        delta = sharpes[i] - sharpes[i - 1]
        color = "#4CAF50" if delta >= 0 else "#F44336"
        sign = "+" if delta >= 0 else ""
        anchor = max(sharpes[i], sharpes[i - 1])
        offset = max(abs(anchor) * 0.12, 0.08)
        ax.annotate(
            f"{sign}{delta:.3f}",
            xy=(i - 0.5, anchor + offset),
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    for i, ann in enumerate(annotations):
        if ann:
            ax.text(
                i,
                -0.03,
                ann,
                ha="center",
                va="top",
                fontsize=8,
                color="gray",
                transform=ax.get_xaxis_transform(),
            )

    ax.axhline(0, color="#9E9E9E", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Lineage: Sharpe Through Pipeline Stages")

    # Symmetric padding accommodates both positive and negative regimes plus
    # any error bars that extend beyond the bar tops.
    if ci_lo and ci_hi:
        lo_extents = [ci_lo[k] for k in stage_keys if k in ci_lo and ci_lo[k] is not None]
        hi_extents = [ci_hi[k] for k in stage_keys if k in ci_hi and ci_hi[k] is not None]
        all_lo = list(sharpes) + lo_extents
        all_hi = list(sharpes) + hi_extents
    else:
        all_lo = list(sharpes)
        all_hi = list(sharpes)
    lo_lim = min(all_lo + [0])
    hi_lim = max(all_hi + [0])
    span = hi_lim - lo_lim
    pad = max(span * 0.18, 0.15)
    ax.set_ylim(lo_lim - pad, hi_lim + pad)

    return fig


# ---------------------------------------------------------------------------
# Figure 3: Concentration Curve (top_k analysis)
# ---------------------------------------------------------------------------


def plot_concentration_curve(
    conc_df: pl.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Sharpe vs top_k for concentration analysis.

    Parameters
    ----------
    conc_df : pl.DataFrame
        From ``BacktestExplorer.concentration_curve()``.
    ax : plt.Axes, optional

    Returns
    -------
    plt.Figure
    """
    if conc_df.is_empty():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No concentration data", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # Best allocator per top_k
    best_per_k = conc_df.sort("sharpe", descending=True).group_by("top_k").first().sort("top_k")

    top_k = best_per_k["top_k"].to_numpy()
    sharpe = best_per_k["sharpe"].to_numpy()
    max_dd = best_per_k["max_drawdown"].to_numpy()
    allocators = best_per_k["allocator"].to_list()

    # Sharpe curve
    ax.plot(top_k, sharpe, "o-", color="#2196F3", linewidth=2, markersize=8, label="Sharpe")

    # Annotate best allocator at each point
    for k, s, a in zip(top_k, sharpe, allocators, strict=False):
        ax.annotate(
            a.replace("_", " "),
            (k, s),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
            alpha=0.7,
        )

    # Mark optimal
    best_idx = np.argmax(sharpe)
    ax.scatter(
        [top_k[best_idx]],
        [sharpe[best_idx]],
        s=150,
        c="#FF9800",
        zorder=5,
        edgecolors="black",
        linewidths=1,
        label=f"Optimal k={top_k[best_idx]}",
    )

    # Secondary axis for max drawdown
    ax2 = ax.twinx()
    ax2.plot(top_k, max_dd, "s--", color="#F44336", alpha=0.6, markersize=6, label="Max DD")
    ax2.set_ylabel("Max Drawdown", color="#F44336")
    ax2.tick_params(axis="y", labelcolor="#F44336")

    ax.set_xlabel("Top K (Portfolio Concentration)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Concentration Analysis: Sharpe vs Portfolio Size")
    ax.legend(loc="upper left", frameon=False)
    ax2.legend(loc="upper right", frameon=False)

    return fig


# ---------------------------------------------------------------------------
# Figure 4: Cost Decay Curve
# ---------------------------------------------------------------------------


def plot_cost_decay(
    explorer,
    *,
    protocol_cost_bps: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Net Sharpe vs total cost with breakeven annotation.

    Parameters
    ----------
    explorer : BacktestExplorer
    protocol_cost_bps : float, optional
        The assumed cost from setup.yaml.
    ax : plt.Axes, optional

    Returns
    -------
    plt.Figure
    """
    costs_df = explorer.cost_sensitivity()
    if costs_df.is_empty():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No cost sensitivity data", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # Best Sharpe per cost level
    best_per_cost = (
        costs_df.sort("sharpe", descending=True).group_by("cost_bps").first().sort("cost_bps")
    )

    cost_bps = best_per_cost["cost_bps"].to_numpy()
    sharpe = best_per_cost["sharpe"].to_numpy()

    ax.plot(cost_bps, sharpe, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.fill_between(cost_bps, sharpe, alpha=0.1, color="#2196F3")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")

    # Estimate breakeven via interpolation
    if sharpe[0] > 0 and sharpe[-1] < 0:
        from scipy.interpolate import interp1d

        f = interp1d(sharpe, cost_bps)
        breakeven = float(f(0))
        ax.axvline(
            breakeven,
            color="#F44336",
            linestyle="--",
            alpha=0.7,
            label=f"Breakeven: {breakeven:.0f} bps",
        )
    elif sharpe[-1] >= 0:
        breakeven = float(cost_bps[-1])
        ax.annotate(
            f"Still positive at {breakeven:.0f} bps",
            xy=(breakeven, sharpe[-1]),
            fontsize=9,
            color="#4CAF50",
        )
    else:
        breakeven = None

    # Protocol cost annotation
    if protocol_cost_bps is not None:
        ax.axvline(
            protocol_cost_bps,
            color="#4CAF50",
            linestyle=":",
            alpha=0.7,
            label=f"Protocol: {protocol_cost_bps:.0f} bps",
        )

        if breakeven is not None and protocol_cost_bps > 0:
            headroom = breakeven / protocol_cost_bps
            ax.annotate(
                f"Headroom: {headroom:.1f}×",
                xy=(protocol_cost_bps, sharpe[0] * 0.9),
                fontsize=10,
                fontweight="bold",
                color="#4CAF50",
            )

    ax.set_xlabel("Total Cost (bps per leg)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Cost Sensitivity: Strategy Viability Under Friction")
    ax.legend(loc="upper right", frameon=False)

    return fig


# ---------------------------------------------------------------------------
# Figure 5: 2-Panel Equity / Drawdown
# ---------------------------------------------------------------------------


def plot_equity_drawdown(
    daily_returns_path: Path,
    *,
    comparison_path: Path | None = None,
    labels: tuple[str, str] = ("Strategy", "Comparison"),
    ax: tuple[plt.Axes, plt.Axes] | None = None,
) -> plt.Figure:
    """2-panel figure: cumulative return (top) + drawdown (bottom).

    Parameters
    ----------
    daily_returns_path : Path
        Parquet file with ``timestamp`` and ``daily_return`` columns.
    comparison_path : Path, optional
        Second return series for overlay (e.g. pre-cost vs post-cost).
    labels : tuple[str, str]
        Labels for primary and comparison series.
    ax : tuple[plt.Axes, plt.Axes], optional

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
    else:
        ax_eq, ax_dd = ax
        fig = ax_eq.figure

    def _load_and_compute(path: Path):
        df = pl.read_parquet(path).sort("timestamp")
        dates = df["timestamp"].to_numpy()
        rets = df["daily_return"].to_numpy()
        cum = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        return dates, cum, dd

    dates, cum, dd = _load_and_compute(daily_returns_path)

    ax_eq.plot(dates, cum, color="#2196F3", linewidth=1.5, label=labels[0])
    ax_dd.fill_between(dates, dd, 0, color="#F44336", alpha=0.3)
    ax_dd.plot(dates, dd, color="#F44336", linewidth=0.8, label=labels[0])

    if comparison_path is not None and comparison_path.exists():
        dates2, cum2, dd2 = _load_and_compute(comparison_path)
        ax_eq.plot(dates2, cum2, color="#FF9800", linewidth=1.2, alpha=0.7, label=labels[1])
        ax_dd.plot(dates2, dd2, color="#FF9800", linewidth=0.8, alpha=0.7, label=labels[1])

    # Annotate worst drawdown
    worst_idx = np.argmin(dd)
    ax_dd.annotate(
        f"Max DD: {dd[worst_idx]:.1%}",
        xy=(dates[worst_idx], dd[worst_idx]),
        textcoords="offset points",
        xytext=(20, -10),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#F44336"),
        color="#F44336",
    )

    ax_eq.set_ylabel("Cumulative Return")
    ax_eq.set_title("Equity Curve and Drawdown Profile")
    ax_eq.legend(loc="upper left", frameon=False)

    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Holdout Comparison (Paired Dumbbell)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Assessment writer / reader
# ---------------------------------------------------------------------------


def write_strategy_assessment(case_study: str, assessment: dict) -> Path:
    """Write strategy_assessment.json to case study results directory.

    Parameters
    ----------
    case_study : str
        Case study ID.
    assessment : dict
        Assessment dictionary with first-pass pipeline outcome.

    Returns
    -------
    Path
        Path to written file.
    """
    from utils.paths import get_case_study_dir

    results_dir = get_case_study_dir(case_study) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    assessment["generated_at"] = datetime.now(tz=UTC).isoformat()

    path = results_dir / "strategy_assessment.json"
    path.write_text(json.dumps(assessment, indent=2, default=str))
    return path


def load_strategy_assessment(
    case_study: str,
    *,
    verify_against_registry: bool = True,
) -> dict[str, Any]:
    """Load strategy_assessment.json for a case study.

    The assessment JSON is a cached aggregate; the registry is the SSoT
    (registry only, never JSONs). When
    ``verify_against_registry`` is True (default), this function checks that
    the assessment's ``champion`` still exists as a training run in the
    registry and emits a stale-data warning if not. The function returns the
    JSON either way; callers must decide how to treat a stale assessment.

    Parameters
    ----------
    case_study : str
    verify_against_registry : bool, default True
        When True, log a warning if the assessment's champion config no
        longer exists in ``training_runs`` (typical cause: training sweep
        rerun produced new hashes, assessment JSON not refreshed).

    Returns
    -------
    dict
        Assessment dictionary, or empty dict if not found.
    """
    import sqlite3
    import warnings

    from utils.paths import get_case_study_dir

    path = get_case_study_dir(case_study) / "results" / "strategy_assessment.json"
    if not path.exists():
        return {}
    assessment = json.loads(path.read_text())

    if verify_against_registry and assessment:
        champion_source = assessment.get("champion", {}).get("source", "")
        primary_label = assessment.get("primary_label", "")
        if champion_source and "/" in champion_source and primary_label:
            family, config_name = champion_source.split("/", 1)
            db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
            if db_path.exists():
                con = sqlite3.connect(str(db_path))
                n = con.execute(
                    "SELECT COUNT(*) FROM training_runs "
                    "WHERE family = ? AND config_name = ? AND label = ?",
                    (family, config_name, primary_label),
                ).fetchone()[0]
                con.close()
                if n == 0:
                    warnings.warn(
                        f"strategy_assessment.json for '{case_study}' is STALE: "
                        f"champion {champion_source}/{primary_label} is not in the "
                        f"registry. Regenerate by running "
                        f"case_studies/{case_study}/*_strategy_analysis.py.",
                        stacklevel=2,
                    )
    return assessment


def load_all_assessments(
    case_studies: list[str] | None = None,
) -> dict[str, dict]:
    """Load strategy assessments for all case studies.

    Convenience for Ch20 aggregation.

    Returns
    -------
    dict[str, dict]
        Keyed by case study ID.
    """
    if case_studies is None:
        case_studies = [
            "etfs",
            "crypto_perps_funding",
            "nasdaq100_microstructure",
            "sp500_equity_option_analytics",
            "us_firm_characteristics",
            "fx_pairs",
            "cme_futures",
            "sp500_options",
            "us_equities_panel",
        ]

    results = {}
    for cs in case_studies:
        v = load_strategy_assessment(cs)
        if v:
            results[cs] = v
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_cost_bps(setup: dict) -> float:
    """Per-leg cost in bps from a case-study setup.yaml.

    Precedence:
    1. ``costs.per_leg_cost_bps_range`` — average of the declared range.
    2. ``costs.fee_schedule`` + ``costs.cost_tiers`` — tier-weighted average
       of taker/maker fees (tiered structures e.g. crypto).
    3. ``costs.fee_schedule`` with only taker_bps/maker_bps — simple average.
    4. Fallback ``10.0`` — explicit last resort.

    setup.yaml is authoritative. The fallback (10.0) is hit only when the
    case study does not declare any cost structure; flag such a case study
    as a setup.yaml gap rather than silently assuming 10 bps.

    Note on crypto (precedence 3 today): the `cost_tiers` block that
    formerly produced a tier-weighted ~3.47 bps was removed in commit
    `2b3bff1a` (setup.yaml reader-cleanup pass) — the majors/alts
    breakdown lives in the inline YAML comment now, not as machine-
    readable data. The simple (taker+maker)/2 = 3.0 bps headline is
    intentional under the post-cleanup config; if a future revision
    wants to recover the tier-weighted average it must reintroduce
    `cost_tiers` to setup.yaml.
    """
    costs = setup.get("costs", {}) or {}
    cost_range = costs.get("per_leg_cost_bps_range")
    if cost_range:
        return sum(cost_range) / len(cost_range)

    fee_schedule = costs.get("fee_schedule", {}) or {}
    cost_tiers = costs.get("cost_tiers", {}) or {}
    if cost_tiers:
        weighted_sum = 0.0
        total_symbols = 0
        for tier in cost_tiers.values():
            tier_fee = tier.get("fee_bps")
            tier_symbols = tier.get("symbols") or []
            if tier_fee is None or not tier_symbols:
                continue
            weighted_sum += tier_fee * len(tier_symbols)
            total_symbols += len(tier_symbols)
        if total_symbols:
            return weighted_sum / total_symbols

    taker = fee_schedule.get("taker_bps")
    maker = fee_schedule.get("maker_bps")
    if taker is not None and maker is not None:
        return (taker + maker) / 2
    if taker is not None:
        return taker
    if maker is not None:
        return maker

    return 10.0


def compute_search_risk_table(explorer) -> pl.DataFrame:
    """Build search-risk summary table for display.

    Parameters
    ----------
    explorer : BacktestExplorer

    Returns
    -------
    pl.DataFrame
        Single-column table for display.
    """
    ctx = explorer.search_context("signal")
    if not ctx:
        return pl.DataFrame()

    dsr = explorer.deflated_sharpe(stage="signal", top_n=1)
    dsr_pval = None
    dsr_sig = None
    if not dsr.is_empty() and "dsr_pvalue" in dsr.columns:
        row = dsr.row(0, named=True)
        dsr_pval = row.get("dsr_pvalue")
        dsr_sig = row.get("significant")

    rows = [
        {"metric": "Total signal backtests", "value": f"{ctx['total']:,}"},
        {"metric": "Champion Sharpe", "value": f"{ctx['champion_sharpe']:.3f}"},
        {"metric": "Median Sharpe", "value": f"{ctx['median_sharpe']:.3f}"},
        {"metric": "90th percentile Sharpe", "value": f"{ctx['p90_sharpe']:.3f}"},
        {"metric": "Champion percentile", "value": f"{ctx['champion_percentile']:.1f}%"},
        {"metric": "% positive Sharpe", "value": f"{ctx['pct_positive']:.1f}%"},
    ]
    if dsr_pval is not None:
        rows.append({"metric": "DSR p-value", "value": f"{dsr_pval:.4f}"})
        rows.append({"metric": "DSR significant", "value": "Yes" if dsr_sig else "No"})

    return pl.DataFrame(rows)


def compute_operating_profile(lineage: dict, setup: dict) -> pl.DataFrame:
    """Build operating profile table for deployment memo.

    Parameters
    ----------
    lineage : dict
        From ``champion_lineage()``.
    setup : dict
        Loaded setup.yaml.

    Returns
    -------
    pl.DataFrame
    """
    # Extract from lineage and setup
    cadence = setup.get("evaluation_protocol", {}).get("rebalance_frequency", "monthly")
    top_k = None
    allocator = None
    worst_dd = None

    if "allocation" in lineage:
        top_k = lineage["allocation"].get("top_k")
        allocator = lineage["allocation"].get("allocator")

    # Find worst drawdown across all stages
    for stage_data in lineage.values():
        dd = stage_data.get("max_drawdown")
        if dd is not None and (worst_dd is None or dd < worst_dd):
            worst_dd = dd

    cost_model = setup.get("cost_model", {})
    cost_bps = cost_model.get("per_leg_cost_bps", None)

    rows = [
        {"property": "Trading cadence", "value": cadence},
        {"property": "Portfolio concentration (top_k)", "value": str(top_k) if top_k else "—"},
        {"property": "Allocator", "value": (allocator or "—").replace("_", " ")},
        {"property": "Cost assumption", "value": f"{cost_bps} bps/leg" if cost_bps else "—"},
        {"property": "Worst drawdown", "value": f"{worst_dd:.1%}" if worst_dd else "—"},
    ]

    return pl.DataFrame(rows)


def classify_holdout_degradation(
    val_sharpe: float | None,
    hold_sharpe: float | None,
) -> str:
    """Classify holdout degradation type.

    Returns one of: proportional, signal_lost, sign_flip,
    degenerate, evidence_gap.
    """
    if val_sharpe is None or hold_sharpe is None:
        return "evidence_gap"
    if hold_sharpe < -0.1:
        return "sign_flip"
    if abs(hold_sharpe) < 0.05:
        return "signal_lost"
    if val_sharpe > 0 and hold_sharpe > 0:
        ratio = hold_sharpe / val_sharpe
        if ratio > 0.5:
            return "proportional"
        return "signal_lost"
    return "degenerate"


def build_all_synthesis(
    case_studies: list[str],
    explorers: dict,
    configs: dict[str, dict],
    ic_df: pl.DataFrame,
    bt_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    assessments: dict[str, dict],
    display_names: dict[str, str],
    asset_class_map: dict[str, str],
    freq_map: dict[str, str],
    pin_cost_risk_to_spine: frozenset[str] = frozenset(),
    allow_missing_spine: bool = False,
) -> dict[str, dict]:
    """Build per-case-study synthesis dict for all_synthesis.json.

    Queries registry and setup.yaml for each case study. Returns a dict
    keyed by case_study_id with meta, pipeline_summary, strategy_assessment,
    selection_flow, and variant_analysis.

    ``pin_cost_risk_to_spine`` lists case studies whose cost_sensitivity and
    risk_overlay numbers must be scoped to the spine (carrier) prediction
    rather than pooled across the whole registry. nasdaq belongs here: its
    cost-feasible ensemble carrier carries the headline cost/risk numbers,
    while the full-universe sweep rows are the Ch18/Ch19 cost-defeat
    demonstration and must not leak into the cross-case comparison.

    ``allow_missing_spine`` is a test-only relaxation: when True, a pinned
    case study whose spine cannot be resolved (its carrier is registered
    out-of-band and absent from an isolated test registry) is reported with
    cost/risk marked not-applicable instead of raising. Production callers
    leave this False so a genuinely missing carrier still fails loudly.
    """
    import contextlib

    from utils.paths import get_case_study_dir

    synthesis_dict = {}

    for cs in case_studies:
        explorer = explorers.get(cs)
        if explorer is None:
            continue
        setup = configs.get(cs, {})
        case_dir = get_case_study_dir(cs)
        display = display_names.get(cs, cs)

        # --- meta ---
        universe = setup.get("universe", {})
        n_assets = universe.get("n_assets", 0) or len(universe.get("symbols", []))
        cost_bps = compute_cost_bps(setup)
        labels_cfg = setup.get("labels", {})

        # Get date range from labels data if available
        date_start, date_end = "", ""
        for labels_subdir in ["labels", "data/labels"]:
            labels_dir = case_dir / labels_subdir
            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.parquet"))
                if label_files:
                    try:
                        lf = pl.scan_parquet(label_files[0])
                        cols = lf.collect_schema().names()
                        ts_col = (
                            "timestamp"
                            if "timestamp" in cols
                            else "date"
                            if "date" in cols
                            else None
                        )
                        if ts_col:
                            ts_df = lf.select(ts_col).collect()
                            if not ts_df.is_empty():
                                date_start = str(ts_df[ts_col].min())[:10]
                                date_end = str(ts_df[ts_col].max())[:10]
                    except Exception:
                        pass
                if date_start:
                    break

        meta = {
            "case_study_id": cs,
            "asset_class": asset_class_map.get(cs, "unknown"),
            "frequency": freq_map.get(cs, "daily"),
            "universe_size": n_assets,
            "date_range": [date_start, date_end],
            "primary_label": labels_cfg.get("primary", ""),
            "cadence": setup.get("decision", {}).get("cadence", ""),
            "cost_bps": cost_bps,
            "calendar": setup.get("decision", {}).get("calendar", ""),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # --- models: best IC per family ---
        models_dict = {}
        cs_ic = ic_df.filter(pl.col("case_study") == display)
        if not cs_ic.is_empty():
            for row in cs_ic.iter_rows(named=True):
                models_dict[row["family"]] = {
                    "best_model": row.get("source", row["family"]),
                    "ic_mean": round(row["ic_best"], 4) if row["ic_best"] is not None else None,
                    "ic_mean_daily": (
                        round(row["ic_best_daily"], 4)
                        if row.get("ic_best_daily") is not None
                        else None
                    ),
                    "ic_std": None,
                    "n_folds": row.get("n_predictions", 0),
                }

        # --- backtest: signal-stage champion ---
        cs_bt = bt_df.filter(pl.col("case_study") == display)
        backtest_dict: dict[str, Any] = {}
        if not cs_bt.is_empty():
            r = cs_bt.row(0, named=True)
            backtest_dict = {
                "selection_stage": "signal",
                "best_source": r.get("best_source", ""),
                "spine_prediction_hash": r.get("spine_prediction_hash"),
                "ml_sharpe": round(r["signal_sharpe"], 4)
                if r["signal_sharpe"] is not None
                else None,
                "ew_sharpe": None,
                "ml_beats_ew": None,
                "max_dd": None,
                "total_return": None,
                "positive_sharpe": r["signal_sharpe"] is not None and r["signal_sharpe"] > 0,
            }

            # Add holdout fields
            cs_ho = (
                holdout_df.filter(pl.col("cs_id") == cs)
                if not holdout_df.is_empty()
                else pl.DataFrame()
            )
            if not cs_ho.is_empty():
                ho = cs_ho.row(0, named=True)
                backtest_dict.update(
                    {
                        "holdout_available": True,
                        "holdout_best_source": f"{ho.get('family', '')}/{ho.get('config', '')}",
                        "holdout_ml_sharpe": round(ho["holdout_sharpe"], 4)
                        if ho["holdout_sharpe"] is not None
                        else None,
                        "holdout_positive_sharpe": ho["holdout_sharpe"] is not None
                        and ho["holdout_sharpe"] > 0,
                    }
                )
            else:
                backtest_dict.update(
                    {
                        "holdout_available": False,
                        "holdout_best_source": None,
                        "holdout_ml_sharpe": None,
                        "holdout_positive_sharpe": None,
                    }
                )

        # --- allocation ---
        # Restrict to the spine rank-1 prediction_hash when bt_df carries
        # it. Without that pin the allocator MAX-per-method aggregation
        # pools across every prediction in the registry, so Figure 20.7
        # can read off a different prediction than Ch20 prose Tables 20.5–20.7.
        cs_bt_row = bt_df.filter(pl.col("case_study") == display)
        spine_pred = None
        if not cs_bt_row.is_empty() and "spine_prediction_hash" in cs_bt_row.columns:
            spine_pred = cs_bt_row["spine_prediction_hash"][0]
        # Allocation stage ONLY. Figure 20.14 / Table 20.6 isolate the allocator
        # layer with the signal held fixed; a risk overlay (ch19) is a downstream
        # layer covered in §20.7, and folding its Sharpe in here would credit the
        # allocator with work the overlay did (and double-count it against §20.7).
        # This matches the "allocation-stage Sharpe" caption and the spine-pinned
        # allocation-only computation in 05_portfolio_allocation.
        alloc_comp = explorer.compare_allocators(
            prediction_hash=spine_pred,
            stages=("allocation",),
        )
        alloc_dict: dict[str, Any] = {
            "best_allocator": "",
            "best_sharpe": None,
            "allocator_comparison": {},
        }
        if not alloc_comp.is_empty():
            # compare_allocators sorts by avg_sharpe; the heatmap and prose report
            # the allocator with the highest best_sharpe, so re-rank explicitly.
            _top = alloc_comp.sort("best_sharpe", descending=True).head(1)
            alloc_dict["best_allocator"] = _top["allocator"][0]
            alloc_dict["best_sharpe"] = round(float(_top["best_sharpe"][0]), 4)
            for row in alloc_comp.iter_rows(named=True):
                alloc_dict["allocator_comparison"][row["allocator"]] = round(
                    float(row["best_sharpe"]), 4
                )

        # --- costs ---
        # A pinned CS MUST carry cost/risk on its spine prediction; falling back
        # to None here would pool full-universe rows — the exact cost-defeat-demo
        # leak the pin prevents. Fail loudly rather than leak silently.
        skip_cost_risk = False
        if cs in pin_cost_risk_to_spine and spine_pred is None:
            if not allow_missing_spine:
                raise ValueError(
                    f"{cs!r} is pinned to the spine prediction for cost/risk, but no "
                    f"spine_prediction_hash resolved (empty backtest row or missing "
                    f"column); refusing to silently pool full-universe cost/risk rows."
                )
            # Test-mode escape hatch: the pinned carrier is registered out-of-band
            # (e.g. nasdaq's cost-feasible ensemble), so an isolated test registry
            # has no carrier rows to resolve a spine from. Mark cost/risk
            # not-applicable rather than pooling full-universe rows — the same leak
            # the hard raise prevents in production (where allow_missing_spine=False).
            skip_cost_risk = True
        cost_risk_pred = spine_pred if cs in pin_cost_risk_to_spine else None
        cost_df = (
            pl.DataFrame()
            if skip_cost_risk
            else explorer.cost_sensitivity(prediction_hash=cost_risk_pred)
        )
        costs_dict: dict[str, Any] = {
            "actual_bps": cost_bps,
            "breakeven_bps": None,
            "survives_costs": None,
            "gross_sharpe_at_zero": None,
            "net_sharpe_at_actual": None,
            "capacity_usd_10pct": None,
        }
        if not cost_df.is_empty():
            # Zero-cost envelope: best achievable Sharpe before any cost is
            # charged, the gross side of the cost waterfall paired with
            # ``net_sharpe_at_actual`` (same cost sweep, same scoping). Both are
            # max-over-config envelopes, so gross >= net by construction (a
            # higher cost can only lower each config's Sharpe).
            zero_rows = cost_df.filter(pl.col("cost_bps") == 0)
            if not zero_rows.is_empty():
                costs_dict["gross_sharpe_at_zero"] = round(float(zero_rows["sharpe"].max()), 4)

            available = sorted(cost_df["cost_bps"].unique().to_list())
            match_bps = None
            for lvl in available:
                if lvl >= cost_bps:
                    match_bps = lvl
                    break
            if match_bps is None and available:
                match_bps = available[-1]

            if match_bps is not None:
                matched = cost_df.filter(pl.col("cost_bps") == match_bps)
                if not matched.is_empty():
                    net_sr = float(matched["sharpe"].max())
                    costs_dict["net_sharpe_at_actual"] = round(net_sr, 4)
                    costs_dict["survives_costs"] = net_sr > 0

            best_per_cost = (
                cost_df.group_by("cost_bps").agg(sharpe=pl.col("sharpe").max()).sort("cost_bps")
            )
            for row in best_per_cost.iter_rows(named=True):
                if row["sharpe"] is not None and row["sharpe"] <= 0:
                    costs_dict["breakeven_bps"] = row["cost_bps"]
                    break
            else:
                if not best_per_cost.is_empty():
                    costs_dict["breakeven_bps"] = float(best_per_cost["cost_bps"].max()) + 10

        # --- risk ---
        risk_df = (
            pl.DataFrame()
            if skip_cost_risk
            else explorer.risk_impact(prediction_hash=cost_risk_pred)
        )
        risk_dict: dict[str, Any] = {
            "best_overlay": "none",
            "baseline_sharpe": 0,
            "baseline_max_dd": 0,
            "managed_sharpe": None,
            "managed_max_dd": None,
            "overlay_sharpe_delta": None,
            "worst_drawdown_pct": 0,
            "var_95": 0,
            "cvar_95": 0,
            "overlay_count": 0,
        }
        if not risk_df.is_empty():
            if "baseline_sharpe" in risk_df.columns:
                bs = risk_df["baseline_sharpe"].drop_nulls()
                if len(bs) > 0:
                    risk_dict["baseline_sharpe"] = round(float(bs[0]), 4)

            best_risk = risk_df.sort("sharpe", descending=True).head(1)
            risk_dict["best_overlay"] = best_risk["risk_name"][0]
            risk_dict["managed_sharpe"] = round(float(best_risk["sharpe"][0]), 4)
            risk_dict["managed_max_dd"] = round(float(best_risk["max_drawdown"][0] or 0), 4)
            risk_dict["overlay_sharpe_delta"] = round(
                risk_dict["managed_sharpe"] - risk_dict["baseline_sharpe"], 4
            )
            risk_dict["overlay_count"] = len(risk_df)

        # --- labels (from setup.yaml) ---
        labels_dict = {
            "primary": labels_cfg.get("primary", ""),
            "variants": labels_cfg.get("variants", []),
            "n_obs": 0,
            "mean": 0,
            "std": 0,
            "hit_rate": 0,
        }

        # --- features (count from features directory) ---
        n_financial = 0
        n_temporal = 0
        for feat_subdir in ["features", "data/features"]:
            feat_dir = case_dir / feat_subdir
            if feat_dir.exists():
                fin_path = feat_dir / "financial.parquet"
                if fin_path.exists():
                    with contextlib.suppress(Exception):
                        n_financial = len(pl.read_parquet_schema(fin_path)) - 2
                temp_path = feat_dir / "model_based.parquet"
                if temp_path.exists():
                    with contextlib.suppress(Exception):
                        n_temporal = len(pl.read_parquet_schema(temp_path)) - 2
                if n_financial > 0:
                    break

        features_dict = {
            "financial": n_financial,
            "temporal": n_temporal,
            "total": n_financial + n_temporal,
            "passed_eval": n_financial + n_temporal,
            "top_3_by_ic": [],
        }

        # --- selection_flow ---
        best_model_source = backtest_dict.get("best_source", "")
        selection_flow = {
            "validation_selected_label": labels_cfg.get("primary", ""),
            "selection_origin": None,
            "selected_model_id": best_model_source.split("/")[-1] if best_model_source else "",
        }

        # --- strategy assessment ---
        cs_assessment = assessments.get(cs, {})

        # --- assemble ---
        synthesis_dict[cs] = {
            "meta": meta,
            "pipeline_summary": {
                "labels": labels_dict,
                "features": features_dict,
                "models": models_dict,
                "backtest": backtest_dict,
                "allocation": alloc_dict,
                "costs": costs_dict,
                "risk": risk_dict,
            },
            "strategy_assessment": cs_assessment if cs_assessment else None,
            "selection_flow": selection_flow,
            "variant_analysis": {},
            "signal_sweep": {},
            "next_steps": [],
            "key_findings": [],
        }

    return synthesis_dict


def _family_color_map() -> dict[str, str]:
    """Consistent color map for model families."""
    return {
        "linear": "#4CAF50",
        "gbm": "#FF9800",
        "tabular_dl": "#2196F3",
        "deep_learning": "#9C27B0",
        "latent_factors": "#E91E63",
        "causal_dml": "#795548",
    }
