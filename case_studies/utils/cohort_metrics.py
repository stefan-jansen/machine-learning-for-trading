"""Compute-and-register the per-case-study ``cohort_metrics`` selection-bias table.

This is the in-repo home of the cohort selection-bias producer that the
strategy-analysis notebook (``NN_strategy_analysis.py``) runs so a reader who
never touches Chapter 20 still lands a populated ``cohort_metrics`` table. It
was migrated verbatim from ``agents/scripts/backfill_cohort_metrics.py`` (the
post-engine-rerun backfill); the only change is that the per-cohort write goes
through :func:`case_studies.utils.registry.register_cohort_metrics` instead of
an inline upsert.

For each case study the table carries, per cohort:

* Raw-K, MP-K, ER-K DSR (with p-values, expected_max_sharpe, min_trl_periods)
* Effective trial counts (n_trials_effective_mp / _er)
* Rademacher-Adjusted Sharpe leader bound + complexity
* White's Reality Check vs ``STAGE_BASELINE``
* PBO via CSCV on per-fold Sharpe matrix
* Leader Sharpe / Sortino / MinTRL

Three cohort granularities are computed: per-family, per-(stage, label), and
per-label. See ``~/ml4t/agents/memory/UNCERTAINTY_ARCHITECTURE.md`` for the
design rationale.

The ``universe_filter`` argument scopes cohorts to backtests whose
``signal.universe_filter`` equals the given value — load-bearing for
``nasdaq100_microstructure``, whose curated ``causal_top50`` carrier must not be
diluted by thousands of dead full-universe rows in the selection-bias K.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
import polars as pl

from case_studies.utils.notebook_contracts import (
    degenerate_prediction_sql,
    full_coverage_prediction_sql,
)
from case_studies.utils.registry import register_cohort_metrics
from case_studies.utils.registry.store import _case_dir, _utc_now
from case_studies.utils.uncertainty import (
    STAGE_BASELINE,
    compute_cohort_metrics,
    load_daily_returns_with_timestamp,
    periods_per_year_from_setup,
)

logger = logging.getLogger("cohort_metrics")

# Cohort trial counts must exclude prediction sets with any constant-prediction
# (NULL-IC) fold — degenerate L1/EN configs are not valid trials. Single source
# of truth for the clause lives in notebook_contracts.
_DEGENERATE_CLAUSE = degenerate_prediction_sql("ps.prediction_hash")

# Optional gated scope: when set, the three cohort listers additionally require
# the backtest's signal-stage universe_filter to equal this value.
_UNIVERSE_FILTER_CLAUSE = "json_extract(br.spec_json, '$.strategy.signal.universe_filter') = ?"


def _registry_db(cs: str) -> Path:
    return _case_dir(cs) / "run_log" / "registry.db"


def _equal_weight_benchmark_hash(db: sqlite3.Connection, label: str) -> str | None:
    """Find the equal-weight benchmark backtest_hash for this label.

    Heuristic: stage='signal', family='benchmark', allocation method=equal_weight,
    matching label. Falls back to first signal-stage equal-weight row.
    """
    rows = db.execute(
        """
        SELECT br.backtest_hash
        FROM backtest_runs br
        JOIN prediction_sets ps ON br.prediction_hash=ps.prediction_hash
        JOIN training_runs tr ON tr.training_hash=ps.training_hash
        WHERE br.stage = 'signal' AND tr.label = ?
          AND (tr.family = 'benchmark' OR tr.family = 'equal_weight')
        LIMIT 1
        """,
        (label,),
    ).fetchall()
    if rows:
        return rows[0][0]
    return None


def _stage_leader_hash(db: sqlite3.Connection, stage: str, label: str) -> str | None:
    """Find the highest-Sharpe backtest at a given stage+label."""
    row = db.execute(
        f"""
        SELECT br.backtest_hash
        FROM backtest_runs br
        JOIN backtest_metrics bm ON br.backtest_hash=bm.backtest_hash
        JOIN prediction_sets ps ON br.prediction_hash=ps.prediction_hash
        JOIN training_runs tr ON tr.training_hash=ps.training_hash
        JOIN prediction_metrics pm ON pm.prediction_hash=ps.prediction_hash
        WHERE br.stage = ? AND tr.label = ? AND bm.sharpe IS NOT NULL
          {full_coverage_prediction_sql("ps", "tr", "pm")}
        ORDER BY bm.sharpe DESC LIMIT 1
        """,
        (stage, label),
    ).fetchone()
    return row[0] if row else None


def _resolve_baseline_hash(db: sqlite3.Connection, stage: str, label: str) -> str | None:
    kind = STAGE_BASELINE.get(stage, "equal_weight")
    if kind == "equal_weight":
        return _equal_weight_benchmark_hash(db, label)
    if kind == "signal_leader":
        return _stage_leader_hash(db, "signal", label)
    if kind == "cost_sensitivity_leader":
        return _stage_leader_hash(db, "cost_sensitivity", label)
    return None


def _per_fold_sharpe_by_variant(db: sqlite3.Connection, hashes: list[str]) -> dict[str, np.ndarray]:
    """For each variant, return the per-fold Sharpe array (NaN for missing folds).

    Variants are dropped if no fold rows are recorded. Folds present for *any*
    variant become the canonical fold index; missing entries are NaN.
    """
    if not hashes:
        return {}
    rows = db.execute(
        f"""
        SELECT backtest_hash, fold_id, sharpe
        FROM backtest_fold_metrics
        WHERE backtest_hash IN ({",".join("?" * len(hashes))})
              AND sharpe IS NOT NULL
        """,
        hashes,
    ).fetchall()
    by_variant: dict[str, dict[int, float]] = {}
    for h, fid, sh in rows:
        by_variant.setdefault(h, {})[int(fid)] = float(sh)
    if not by_variant:
        return {}
    all_folds = sorted({fid for fmap in by_variant.values() for fid in fmap})
    out = {}
    for h, fmap in by_variant.items():
        out[h] = np.array([fmap.get(f, np.nan) for f in all_folds], dtype=np.float64)
    return out


def _list_family_cohorts(
    db: sqlite3.Connection, universe_filter: str | None = None
) -> list[tuple[str, str, str, list[str]]]:
    """Return [(stage, label, family, [hash, ...])] for each per-family cohort.

    Restricted to ``split='validation'`` rows: cohort selection-bias is a
    measurement on the validation surface, not on the holdout. Including
    holdout backtests pollutes the inner-join alignment for case studies
    whose holdout window is disjoint from the validation window.
    """
    extra = f" AND {_UNIVERSE_FILTER_CLAUSE}" if universe_filter else ""
    params = (universe_filter,) if universe_filter else ()
    rows = db.execute(
        f"""
        SELECT br.stage, tr.label, tr.family, br.backtest_hash
        FROM backtest_runs br
        JOIN prediction_sets ps ON br.prediction_hash=ps.prediction_hash
        JOIN training_runs tr ON tr.training_hash=ps.training_hash
        JOIN prediction_metrics pm ON pm.prediction_hash=ps.prediction_hash
        WHERE ps.split = 'validation'{extra}{_DEGENERATE_CLAUSE}
          {full_coverage_prediction_sql("ps", "tr", "pm")}
        """,
        params,
    ).fetchall()
    cohorts: dict[tuple[str, str, str], list[str]] = {}
    for stage, label, family, h in rows:
        cohorts.setdefault((stage or "signal", label, family), []).append(h)
    return [(s, l, f, hs) for (s, l, f), hs in cohorts.items() if len(hs) >= 2]


def _list_stagelabel_cohorts(
    db: sqlite3.Connection, universe_filter: str | None = None
) -> list[tuple[str, str, list[str]]]:
    """Per ``_list_family_cohorts``: validation-split only."""
    extra = f" AND {_UNIVERSE_FILTER_CLAUSE}" if universe_filter else ""
    params = (universe_filter,) if universe_filter else ()
    rows = db.execute(
        f"""
        SELECT br.stage, tr.label, br.backtest_hash
        FROM backtest_runs br
        JOIN prediction_sets ps ON br.prediction_hash=ps.prediction_hash
        JOIN training_runs tr ON tr.training_hash=ps.training_hash
        JOIN prediction_metrics pm ON pm.prediction_hash=ps.prediction_hash
        WHERE ps.split = 'validation'{extra}{_DEGENERATE_CLAUSE}
          {full_coverage_prediction_sql("ps", "tr", "pm")}
        """,
        params,
    ).fetchall()
    cohorts: dict[tuple[str, str], list[str]] = {}
    for stage, label, h in rows:
        cohorts.setdefault((stage or "signal", label), []).append(h)
    return [(s, l, hs) for (s, l), hs in cohorts.items() if len(hs) >= 2]


def _list_label_cohorts(
    db: sqlite3.Connection, universe_filter: str | None = None
) -> list[tuple[str, list[str]]]:
    # Exclude cost_sensitivity: those rows are perturbation analyses on a
    # fixed strategy spec, not alternative strategies. They must not inflate
    # the selection-bias K — see 20_strategy_synthesis/holdout.py
    # ``HOLDOUT_SELECTION_STAGES`` for the matching rank-1 rule.
    # Restricted to validation split per ``_list_family_cohorts``.
    extra = f" AND {_UNIVERSE_FILTER_CLAUSE}" if universe_filter else ""
    params = (universe_filter,) if universe_filter else ()
    rows = db.execute(
        f"""
        SELECT tr.label, br.backtest_hash
        FROM backtest_runs br
        JOIN prediction_sets ps ON br.prediction_hash=ps.prediction_hash
        JOIN training_runs tr ON tr.training_hash=ps.training_hash
        JOIN prediction_metrics pm ON pm.prediction_hash=ps.prediction_hash
        WHERE COALESCE(br.stage, 'signal') != 'cost_sensitivity'
          AND ps.split = 'validation'{extra}{_DEGENERATE_CLAUSE}
          {full_coverage_prediction_sql("ps", "tr", "pm")}
        """,
        params,
    ).fetchall()
    cohorts: dict[str, list[str]] = {}
    for label, h in rows:
        cohorts.setdefault(label, []).append(h)
    return [(l, hs) for l, hs in cohorts.items() if len(hs) >= 2]


def process_cohort(
    cs: str,
    db: sqlite3.Connection,
    *,
    cohort_type: str,
    stage: str | None,
    label: str,
    family: str | None,
    member_hashes: list[str],
    ppy: float,
) -> tuple[dict | None, str]:
    """Compute one cohort. Returns ``(cohort_row_or_None, message)``.

    On success the first element is a dict ``{cohort_type, stage, label,
    family, metrics}`` ready for :func:`register_cohort_metrics`; ``None`` on a
    skip (fewer than two variants with returns, empty compute result, etc.).
    """
    returns_by_hash: dict[str, pl.DataFrame] = {}
    for h in member_hashes:
        frame = load_daily_returns_with_timestamp(cs, h)
        if frame is not None and frame.height >= 4:
            returns_by_hash[h] = frame
    if len(returns_by_hash) < 2:
        return None, f"only {len(returns_by_hash)} variant(s) with returns"

    # Baseline lookup for Reality Check
    baseline_stage = stage or ("signal" if cohort_type == "label" else None)
    baseline_hash = None
    if baseline_stage:
        baseline_hash = _resolve_baseline_hash(db, baseline_stage, label)
    baseline_frame = (
        load_daily_returns_with_timestamp(cs, baseline_hash)
        if baseline_hash and baseline_hash in returns_by_hash
        else load_daily_returns_with_timestamp(cs, baseline_hash)
        if baseline_hash
        else None
    )
    # If baseline is in the cohort, drop it from challengers to avoid self-reference
    challengers = (
        {h: f for h, f in returns_by_hash.items() if h != baseline_hash}
        if baseline_hash
        else returns_by_hash
    )
    if len(challengers) < 2:
        challengers = returns_by_hash  # baseline drop would collapse cohort

    # Per-fold Sharpe for PBO (only meaningful for family cohorts).
    # Dict keys ARE backtest_hash — compute_cohort_metrics requires this
    # for the leader_hash FK contract.
    fold_returns_by_hash: dict[str, np.ndarray] | None = None
    if cohort_type == "family":
        fold_returns_by_hash = _per_fold_sharpe_by_variant(db, list(challengers.keys()))
        if len(fold_returns_by_hash) < 2:
            fold_returns_by_hash = None

    baseline_arr_for_rc = None
    if baseline_frame is not None:
        baseline_arr_for_rc = baseline_frame.select(pl.col("ret")).to_numpy().ravel()

    metrics = compute_cohort_metrics(
        challengers,
        periods_per_year=ppy,
        baseline_returns=baseline_arr_for_rc,
        fold_returns_by_hash=fold_returns_by_hash,
    )
    if not metrics:
        return None, "compute_cohort_metrics returned empty"

    metrics["computed_at"] = _utc_now()
    leader_hash_short = str(metrics.get("leader_hash", ""))[:8]
    k_total = metrics.get("k_variants", "")
    k_mp = metrics.get("n_trials_effective_mp", "-")
    k_er = metrics.get("n_trials_effective_er", "-")
    dsr_er = metrics.get("dsr_er", "-")
    cohort_row = {
        "cohort_type": cohort_type,
        "stage": stage,
        "label": label,
        "family": family,
        "metrics": metrics,
    }
    return cohort_row, (
        f"leader={leader_hash_short} K={k_total} K_eff_mp={k_mp} K_eff_er={k_er} dsr_er={dsr_er}"
    )


def compute_and_register(
    cs: str,
    *,
    universe_filter: str | None = None,
    verbose: bool = True,
    case_dir: Path | None = None,
) -> dict[str, int]:
    """Compute all cohorts for ``cs`` and persist them to ``cohort_metrics``.

    Runs the three cohort granularities (family, stagelabel, label) over the
    case-study registry, then writes every computed cohort in a single call to
    :func:`register_cohort_metrics` (which also prunes cohort rows whose leader
    no longer maps to a ``backtest_runs`` row).

    Returns a per-granularity count dict, e.g.
    ``{"family": 6, "stagelabel": 4, "label": 2, "errors": 0,
    "dangling_pruned": 0}``.
    """
    db_path = (case_dir / "run_log" / "registry.db") if case_dir else _registry_db(cs)
    if not db_path.exists():
        logger.warning("no registry for %s", cs)
        return {"family": 0, "stagelabel": 0, "label": 0, "errors": 0, "dangling_pruned": 0}

    counts = {"family": 0, "stagelabel": 0, "label": 0, "errors": 0}
    ppy = periods_per_year_from_setup(cs)
    cohort_rows: list[dict] = []

    db = sqlite3.connect(str(db_path), timeout=120.0)
    db.execute("PRAGMA busy_timeout = 60000;")
    try:
        if verbose:
            print(f"[cohort_metrics] {cs} (ppy={ppy})", flush=True)
        for cohort_type, lister, builder in (
            (
                "family",
                _list_family_cohorts,
                lambda r: dict(stage=r[0], label=r[1], family=r[2], hashes=r[3]),
            ),
            (
                "stagelabel",
                _list_stagelabel_cohorts,
                lambda r: dict(stage=r[0], label=r[1], family=None, hashes=r[2]),
            ),
            (
                "label",
                _list_label_cohorts,
                lambda r: dict(stage=None, label=r[0], family=None, hashes=r[1]),
            ),
        ):
            cohorts = lister(db, universe_filter)
            if verbose:
                print(f"  {cohort_type}: {len(cohorts)} cohorts", flush=True)
            if universe_filter is not None and not cohorts:
                logger.warning(
                    "%s: universe_filter=%r yielded 0 %s cohorts "
                    "(filter likely does not apply to this case study)",
                    cs,
                    universe_filter,
                    cohort_type,
                )
            for i, raw in enumerate(cohorts, 1):
                c = builder(raw)
                t0 = time.time()
                try:
                    row, msg = process_cohort(
                        cs,
                        db,
                        cohort_type=cohort_type,
                        stage=c["stage"],
                        label=c["label"],
                        family=c["family"],
                        member_hashes=c["hashes"],
                        ppy=ppy,
                    )
                except Exception as exc:  # noqa: BLE001 — one cohort must not abort the rest
                    if verbose:
                        print(f"    [{i}/{len(cohorts)}] {cohort_type} ERROR: {exc}", flush=True)
                    counts["errors"] += 1
                    continue
                if row is not None:
                    cohort_rows.append(row)
                    counts[cohort_type] += 1
                    if verbose:
                        print(
                            f"    [{i}/{len(cohorts)}] {cohort_type} "
                            f"stage={c['stage']} label={c['label']} family={c['family']} "
                            f"{msg} ({time.time() - t0:.1f}s)",
                            flush=True,
                        )
                elif verbose:
                    print(
                        f"    [{i}/{len(cohorts)}] {cohort_type} "
                        f"stage={c['stage']} label={c['label']} family={c['family']} "
                        f"SKIP: {msg}",
                        flush=True,
                    )
    finally:
        db.close()

    n_pruned = register_cohort_metrics(cs, cohort_rows, replace_all=True, case_dir=case_dir)
    counts["dangling_pruned"] = n_pruned
    if verbose:
        print(
            f"[cohort_metrics] {cs}: family={counts['family']} "
            f"stagelabel={counts['stagelabel']} label={counts['label']} "
            f"errors={counts['errors']} dangling_pruned={n_pruned}",
            flush=True,
        )
    return counts
