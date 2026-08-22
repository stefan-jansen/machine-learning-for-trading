"""Compute-and-register the per-case-study ``backtest_paired_metrics`` table.

In-repo home of the paired-bootstrap producer that the strategy-analysis
notebook (``NN_strategy_analysis.py``) runs so a reader who never touches
Chapter 20 still lands a populated ``backtest_paired_metrics`` table. It renders
§2 (stage-transition waterfall), §6 (holdout decay + holdout-vs-benchmark) and
§7 (benchmark-aware diagnostics) without inline bootstrap recomputation.

The logic is extracted verbatim from the two producer loops in
``20_strategy_synthesis/01_aggregate_synthesis.py``; the only change is that the
per-case-study selection config (label restriction, rung pin, carrier pin,
frequency) is passed in as arguments rather than read from Ch20 module globals,
so a single case study can run standalone. The Chapter-20 aggregate can later
call this function per case study and become a pure reader.

Six pair types are produced per case study:

1. signal rank-1 (overall) ↔ equal-weight (overall)
2. cross-stage rank-1 holdout ↔ equal-weight (holdout window)
3. holdout rank-1 ↔ validation rank-1 of the same lineage (disjoint windows)
4. allocation rank-1 ↔ signal rank-1 (same window, stage transition)
5. cost-sensitivity rank-1 ↔ allocation rank-1 (same window)
6. risk-overlay rank-1 ↔ cost-sensitivity rank-1 (same window)

All pairs use the paired stationary block bootstrap
(``compute_paired_uncertainty``); pair #3 uses independent per-window draws
(``compute_independent_diff_uncertainty``) since the windows are disjoint.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl

from case_studies.utils.analytics import DISPLAY_NAMES
from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.benchmark import load_benchmark_returns
from case_studies.utils.registry.registration import register_paired_metrics
from case_studies.utils.uncertainty import (
    SIGNAL_BASELINE_BY_CASE_STUDY,
    compute_independent_diff_uncertainty,
    compute_paired_uncertainty,
)
from utils.paths import get_case_study_dir

# Cross-stage rank-1 pooling stages — mirrors holdout.py::HOLDOUT_SELECTION_STAGES.
_PAIRED_STAGES = ("signal", "allocation", "risk_overlay")

# Frequency → periods-per-year. Frequencies absent from the map (e.g. "15min")
# fall through to 252, matching the Ch20 producer's ``.get(..., 252)``.
_PPY_BY_FREQ = {"daily": 252, "weekly": 52, "monthly": 12, "8h": 1095}


def _min_paired_n(ppy: int) -> int:
    """Minimum series length for paired-bootstrap stability, frequency-aware.

    The ~21 floor was written for daily cadences (about a month of obs).
    Monthly case studies (e.g. ``us_firm_characteristics``) have ~12 holdout
    observations by design, and ``compute_paired_uncertainty`` runs cleanly
    on n=12. Scale the floor with ``ppy`` so monthly/weekly CSs aren't
    blocked by a daily-tuned guard.
    """
    if ppy <= 12:  # monthly
        return 6
    if ppy <= 52:  # weekly
        return 12
    return 21  # daily / 8h / intraday


def _joint_coerce(c_arr, b_arr):
    """Filter NaN/non-finite jointly across paired series and trim leading
    rows where *either* is zero. Matches ``_coerce_returns`` semantics but
    preserves index alignment so ``compute_paired_uncertainty``'s
    equal-length precondition survives — the upstream helper trims leading
    zeros independently per series, which can desynchronize a paired
    bootstrap if one side has more leading inactive bars than the other.
    """
    c = np.asarray(c_arr, dtype=np.float64)
    b = np.asarray(b_arr, dtype=np.float64)
    finite = np.isfinite(c) & np.isfinite(b)
    c, b = c[finite], b[finite]
    if c.size == 0:
        return c, b
    nonzero = np.flatnonzero((c != 0.0) & (b != 0.0))
    if nonzero.size == 0:
        return c[:0], b[:0]
    start = int(nonzero[0])
    return c[start:], b[start:]


def _apply_rung_restriction(df: pl.DataFrame, rung: dict | None) -> pl.DataFrame:
    """Filter ``df`` to the case study's pinned rung, if one is configured.

    Returns the input untouched if ``rung`` is None. The helper relies on
    ``BacktestExplorer.best()`` always emitting both ``universe_filter`` and
    ``exit_at_max_days`` columns; if a future schema regression drops them, the
    polars ``filter`` raises column-not-found rather than silently letting the
    rank-1 selection drift back to the cross-rung max.
    """
    if rung is None or df.is_empty():
        return df
    return df.filter(rung["predicate"])


def _apply_carrier_pin(df: pl.DataFrame, predicate: pl.Expr | None) -> pl.DataFrame:
    """Restrict candidate rows to the pinned model carrier, if configured.

    Applied alongside ``_apply_rung_restriction`` at every cross-stage
    carrier-selection site (signal / cross-stage spine / holdout-pairing walk).
    """
    if predicate is None or df.is_empty() or "config_name" not in df.columns:
        return df
    return df.filter(predicate)


def _benchmark_returns_from_artifact(
    cs: str, label: str, period: str = "overall"
) -> tuple[str, pl.DataFrame, str] | None:
    """Resolve the side-artifact equal-weight benchmark for ``(cs, label)``.

    The benchmark is the daily-MTM EW reference series persisted by
    ``scripts/compute_vectorized_ew_benchmark.py`` at
    ``case_studies/{cs}/benchmark/{label}.parquet``. Single, well-defined
    methodology per (cs, label). ``period`` selects the window slice
    (``"overall"`` or ``"holdout"``). Classification-label fallback to the
    matching ``fwd_ret_*`` artifact applies in both periods.

    Returns ``(synthetic_hash, returns_df, resolved_label)`` or ``None`` when
    the artifact is missing. ``synthetic_hash`` is a deterministic identifier
    safe as the PK column in ``backtest_paired_metrics`` (no FK on
    ``benchmark_hash``).
    """
    df = load_benchmark_returns(cs, label, period=period)
    bench_label = label
    if df.is_empty() or "ew_return" not in df.columns:
        fallback = None
        for prefix in ("fwd_class_", "fwd_dir_", "fwd_tb_", "fwd_carry_"):
            if label.startswith(prefix):
                fallback = "fwd_ret_" + label[len(prefix) :]
                break
        if fallback is None:
            return None
        df = load_benchmark_returns(cs, fallback, period=period)
        if df.is_empty() or "ew_return" not in df.columns:
            return None
        bench_label = fallback
    suffix = "" if period == "overall" else f":{period}"
    bench_hash = f"side_ew:{cs}:{bench_label}{suffix}"
    return (
        bench_hash,
        df.select(
            pl.col("timestamp").cast(pl.Date).alias("timestamp"),
            pl.col("ew_return").cast(pl.Float64).alias("ret"),
        ),
        bench_label,
    )


def _aligned_returns(cs: str, h: str) -> pl.DataFrame | None:
    """Load and normalize a backtest's daily returns; columns ``[timestamp, ret]``."""
    parquet = get_case_study_dir(cs) / "run_log" / "backtest" / h / "daily_returns.parquet"
    if not parquet.exists():
        return None
    df = pl.read_parquet(parquet)
    ret_col = next(
        (c for c in ("daily_return", "ret", "return", "value") if c in df.columns),
        df.columns[-1],
    )
    ts_col = next(
        (c for c in ("timestamp", "date", "datetime") if c in df.columns),
        df.columns[0],
    )
    return df.select(
        pl.col(ts_col).cast(pl.Date).alias("timestamp"),
        pl.col(ret_col).cast(pl.Float64).alias("ret"),
    )


def _full_strategy_spec_from_backtest(db: sqlite3.Connection, bt_hash: str) -> dict | None:
    """Pull the full strategy spec dict (signal + allocation + risk) from
    ``bt_hash``'s spec_json. Returns None if the row is missing or signal has
    no ``method`` field.

    The carrier of a backtest is the tuple (signal, allocation, risk). Pinning
    the val→holdout pair on this full spec keeps the comparison apples-to-
    apples; pinning on signal alone allows MAX(sharpe) to surface a holdout
    row with a different allocation or risk overlay than the validation rank-1.
    """
    row = db.execute(
        "SELECT spec_json FROM backtest_runs WHERE backtest_hash = ?",
        (bt_hash,),
    ).fetchone()
    if not row:
        return None
    strat = json.loads(row[0]).get("strategy", {})
    sig = strat.get("signal", {})
    if not sig.get("method"):
        return None
    alloc = strat.get("allocation") or {}
    risk = strat.get("risk") or {}
    return {
        "signal": {
            "method": sig.get("method"),
            "top_k": sig.get("top_k"),
            "percentile": sig.get("percentile"),
        },
        "allocation": {
            "method": alloc.get("method"),
            "top_k": alloc.get("top_k"),
            "long_short": alloc.get("long_short"),
        },
        "risk": {
            "name": risk.get("name"),
        },
    }


def _full_strategy_clauses(spec: dict | None) -> tuple[list[str], list[object]]:
    """Build SQL WHERE clauses + params pinning a backtest row to the full
    strategy spec (signal + allocation + risk). Empty list when spec is None.

    Pinning on the full spec ensures ``MAX(sharpe)`` over candidate holdout
    backtests cannot surface a different allocator or risk overlay than the
    validation carrier — the val→holdout pair stays apples-to-apples on the
    full pipeline configuration, not just the signal.
    """
    if not spec:
        return [], []
    clauses: list[str] = []
    params: list[object] = []

    sig = spec.get("signal") or {}
    method = sig.get("method")
    if method is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.signal.method') IS NULL")
    else:
        clauses.append("json_extract(b.spec_json, '$.strategy.signal.method') = ?")
        params.append(method)
    top_k = sig.get("top_k")
    if top_k is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.signal.top_k') IS NULL")
    else:
        clauses.append("CAST(json_extract(b.spec_json, '$.strategy.signal.top_k') AS INTEGER) = ?")
        params.append(int(top_k))
    pct = sig.get("percentile")
    if pct is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.signal.percentile') IS NULL")
    else:
        clauses.append(
            "CAST(json_extract(b.spec_json, '$.strategy.signal.percentile') AS REAL) = ?"
        )
        params.append(float(pct))

    alloc = spec.get("allocation") or {}
    am = alloc.get("method")
    if am is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.allocation.method') IS NULL")
    else:
        clauses.append("json_extract(b.spec_json, '$.strategy.allocation.method') = ?")
        params.append(am)
    ak = alloc.get("top_k")
    if ak is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.allocation.top_k') IS NULL")
    else:
        clauses.append(
            "CAST(json_extract(b.spec_json, '$.strategy.allocation.top_k') AS INTEGER) = ?"
        )
        params.append(int(ak))
    als = alloc.get("long_short")
    if als is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.allocation.long_short') IS NULL")
    else:
        clauses.append(
            "CAST(json_extract(b.spec_json, '$.strategy.allocation.long_short') AS INTEGER) = ?"
        )
        params.append(int(bool(als)))

    risk = spec.get("risk") or {}
    risk_name = risk.get("name")
    if risk_name is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.risk.name') IS NULL")
    else:
        clauses.append("json_extract(b.spec_json, '$.strategy.risk.name') = ?")
        params.append(risk_name)

    return clauses, params


def _val_rank1_full_spec(
    cs: str,
    explorer: BacktestExplorer,
    *,
    label_restriction: frozenset[str] | None,
    rung: dict | None,
    carrier_pin_predicate: pl.Expr | None,
) -> dict | None:
    """Return the val rank-1 *full strategy* spec for ``cs`` — the
    highest-Sharpe validation backtest across (signal, allocation,
    risk_overlay) stages — walking candidates by val Sharpe descending until
    one with a matching holdout backtest at the SAME full spec is found.

    Returns None when no val candidate up to rank ~200 has a matching holdout
    under the case study's label / rung restrictions.
    """
    cand = pl.concat(
        [explorer.best(stage=s, top_n=2000) for s in ("signal", "allocation", "risk_overlay")],
        how="diagonal_relaxed",
    )
    if cand.is_empty() or "backtest_hash" not in cand.columns:
        return None
    if "family" in cand.columns:
        cand = cand.filter(pl.col("family") != "benchmark")
    if label_restriction and "label" in cand.columns:
        cand = cand.filter(pl.col("label").is_in(list(label_restriction)))
    cand = _apply_rung_restriction(cand, rung)
    cand = _apply_carrier_pin(cand, carrier_pin_predicate)
    if cand.is_empty():
        return None
    # Do NOT dedup by prediction_hash here — the walk needs every registered
    # (signal, allocation, risk_overlay) tuple so a same-prediction lower-sharpe
    # variant can serve as the apples-to-apples carrier.
    cand = cand.sort("sharpe", descending=True)

    case_dir = get_case_study_dir(cs)
    db_path = case_dir / "run_log" / "registry.db"
    db = sqlite3.connect(str(db_path))
    try:
        for i in range(min(cand.height, 200)):
            bt_hash = cand["backtest_hash"][i]
            spec = _full_strategy_spec_from_backtest(db, bt_hash)
            if spec is None:
                continue
            spec_clauses, spec_params = _full_strategy_clauses(spec)
            ho_clauses = ["p.split = 'holdout'"] + spec_clauses
            ho_params: list[object] = list(spec_params)
            if label_restriction:
                placeholders = ",".join("?" for _ in label_restriction)
                ho_clauses.append(f"t.label IN ({placeholders})")
                ho_params.extend(sorted(label_restriction))
            if rung is not None:
                ho_clauses.append(
                    "COALESCE(json_extract(b.spec_json, '$.strategy.signal.universe_filter'), 'full') = ?"
                )
                ho_params.append(rung["universe_filter"])
                if rung["exit_at_max_days"] is None:
                    ho_clauses.append(
                        "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') IS NULL"
                    )
                else:
                    ho_clauses.append(
                        "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') = ?"
                    )
                    ho_params.append(rung["exit_at_max_days"])
            row = db.execute(
                f"""
                SELECT 1 FROM prediction_sets p
                JOIN training_runs t ON p.training_hash = t.training_hash
                JOIN backtest_runs b ON p.prediction_hash = b.prediction_hash
                                     AND b.stage IN ('signal','allocation','risk_overlay','holdout')
                JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
                WHERE {" AND ".join(ho_clauses)}
                  AND bm.sharpe IS NOT NULL
                LIMIT 1
                """,
                ho_params,
            ).fetchone()
            if row:
                return spec
    finally:
        db.close()
    return None


def _holdout_lineage_for(
    cs: str,
    leader_label: str,
    strategy_spec: dict | None = None,
    *,
    label_restriction: frozenset[str] | None,
    rung: dict | None,
    prefer_prediction_hash: str | None = None,
) -> dict | None:
    """Return ``{backtest_hash, prediction_hash, family, config_name, label}``
    for the highest-Sharpe holdout backtest registered in this case study,
    honoring per-CS cluster restrictions but **not** the leader's label.

    ``leader_label`` is intentionally unused in the SQL — kept for call-site
    symmetry with ``_val_backtest_for_lineage``. The holdout's *own* label is
    returned so callers can pair it against matching benchmarks (the label may
    differ from the validation rank-1 when ``generate_holdout``'s degeneracy
    fallback accepts a candidate on a different label).

    When ``strategy_spec`` is provided, the holdout pick is restricted to
    backtests with the same full (signal, allocation, risk) tuple as val's
    rank-1 carrier, so the val→holdout comparison stays apples-to-apples.
    """
    case_dir = get_case_study_dir(cs)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return None

    clauses = ["p.split = 'holdout'"]
    params: list[object] = []
    if label_restriction:
        placeholders = ",".join("?" for _ in label_restriction)
        clauses.append(f"t.label IN ({placeholders})")
        params.extend(label_restriction)
    if rung is not None:
        clauses.append(
            "COALESCE(json_extract(b.spec_json, '$.strategy.signal.universe_filter'), 'full') = ?"
        )
        params.append(rung["universe_filter"])
        if rung["exit_at_max_days"] is None:
            clauses.append(
                "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') IS NULL"
            )
        else:
            clauses.append("json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') = ?")
            params.append(rung["exit_at_max_days"])
    spec_clauses, spec_params = _full_strategy_clauses(strategy_spec)
    clauses.extend(spec_clauses)
    params.extend(spec_params)
    where_sql = " AND ".join(clauses)

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        # Same-lineage preference: given the validation rank-1's own prediction
        # set, prefer a holdout sharing its trained model AND its checkpoint.
        # Both are read from that one row here rather than accepted as separate
        # arguments, because a caller that passes the training hash and forgets
        # the checkpoint reintroduces the defect while looking correct.
        #
        # The checkpoint has to be pinned because a trained model registers one
        # prediction set per declared checkpoint and they share a strategy spec,
        # so the training hash alone leaves one indistinguishable candidate per
        # checkpoint. This branch writes the hash the ``val_rank1_self`` pair is
        # stored under and ``select_holdout_self_backtest`` reads it back, so the
        # two must agree. Ordering by ``backtest_hash`` rather than by Sharpe
        # keeps the choice off holdout performance either way.
        if prefer_prediction_hash is not None:
            carrier = db.execute(
                """
                SELECT training_hash, checkpoint_value, checkpoint_kind
                FROM prediction_sets WHERE prediction_hash = ?
                """,
                (prefer_prediction_hash,),
            ).fetchone()
            if carrier is not None:
                row = db.execute(
                    f"""
                    SELECT t.family, t.config_name, t.label,
                           p.prediction_hash, b.backtest_hash
                    FROM prediction_sets p
                    JOIN training_runs t ON p.training_hash = t.training_hash
                    JOIN backtest_runs b ON p.prediction_hash = b.prediction_hash
                                         AND b.stage IN
                                             ('signal','allocation','risk_overlay','holdout')
                    JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
                    WHERE {where_sql}
                      AND p.training_hash = ?
                      AND p.checkpoint_value IS ?
                      AND p.checkpoint_kind IS ?
                    ORDER BY b.backtest_hash
                    LIMIT 1
                    """,
                    params + [carrier[0], carrier[1], carrier[2]],
                ).fetchone()
                if row:
                    return dict(row)

        row = db.execute(
            f"""
            SELECT t.family, t.config_name, t.label,
                   p.prediction_hash, b.backtest_hash
            FROM prediction_sets p
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN backtest_runs b ON p.prediction_hash = b.prediction_hash
                                 AND b.stage IN ('signal','allocation','risk_overlay','holdout')
            JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
            WHERE {where_sql}
            ORDER BY bm.sharpe DESC NULLS LAST
            LIMIT 1
            """,
            params,
        ).fetchone()
    finally:
        db.close()
    return dict(row) if row else None


def _val_backtest_for_lineage(cs: str, family: str, config_name: str, label: str) -> str | None:
    """Return the highest-Sharpe validation signal-stage backtest_hash for the
    given (family, config_name, label) lineage, or None if absent.

    Used by ``val_rank1_self`` pair construction so the comparison stays
    *within* a lineage when the holdout retrain came from a fallback candidate.
    """
    case_dir = get_case_study_dir(cs)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return None
    db = sqlite3.connect(str(db_path))
    try:
        row = db.execute(
            """
            SELECT b.backtest_hash
            FROM prediction_sets p
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN backtest_runs b ON p.prediction_hash = b.prediction_hash
                                 AND b.stage = 'signal'
            JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
            WHERE p.split = 'validation'
              AND t.family = ?
              AND t.config_name = ?
              AND t.label = ?
            ORDER BY bm.sharpe DESC NULLS LAST
            LIMIT 1
            """,
            (family, config_name, label),
        ).fetchone()
    finally:
        db.close()
    return row[0] if row else None


def _populate_pair(
    cs,
    challenger_hash,
    benchmark_hash,
    benchmark_kind,
    challenger_returns,
    benchmark_returns,
    ppy,
    label,
    *,
    disjoint_windows: bool = False,
    benchmark_label: str | None = None,
    write_case_dir: Path | None = None,
):
    """Compute and register one paired-metric row. Idempotent UPSERT.

    With ``disjoint_windows=True`` (val→holdout decay), each side is
    bootstrapped independently over its full window and the difference
    distribution is built from independent draws. Otherwise, the streams are
    inner-joined on timestamp and a paired stationary bootstrap runs on the
    aligned diff series.
    """
    min_n = _min_paired_n(ppy)
    if disjoint_windows:
        c_arr = challenger_returns.sort("timestamp")["ret"].to_numpy()
        b_arr = benchmark_returns.sort("timestamp")["ret"].to_numpy()
        finite_c = np.isfinite(c_arr)
        finite_b = np.isfinite(b_arr)
        c_arr, b_arr = c_arr[finite_c], b_arr[finite_b]
        if c_arr.size < min_n or b_arr.size < min_n:
            return {
                "cs": cs,
                "kind": benchmark_kind,
                "label": label,
                "benchmark_label": benchmark_label if benchmark_label is not None else label,
                "skip": f"insufficient_disjoint:n_c={c_arr.size},n_b={b_arr.size}",
            }
        n_overlap = min(c_arr.size, b_arr.size)
        paired = compute_independent_diff_uncertainty(
            c_arr,
            b_arr,
            periods_per_year=ppy,
            case_study=cs,
            label=label,
            n_boot=2000,
            seed=42,
        )
    else:
        aligned = challenger_returns.join(
            benchmark_returns, on="timestamp", how="inner", suffix="_b"
        )
        if aligned.height < min_n:
            return {
                "cs": cs,
                "kind": benchmark_kind,
                "label": label,
                "benchmark_label": benchmark_label if benchmark_label is not None else label,
                "skip": f"insufficient_overlap:n={aligned.height}",
            }
        c_arr = aligned["ret"].to_numpy()
        b_arr = aligned["ret_b"].to_numpy()
        c_arr, b_arr = _joint_coerce(c_arr, b_arr)
        n_overlap = c_arr.size
        if n_overlap < min_n:
            return {
                "cs": cs,
                "kind": benchmark_kind,
                "label": label,
                "benchmark_label": benchmark_label if benchmark_label is not None else label,
                "skip": f"insufficient_after_coerce:n={n_overlap}",
            }
        paired = compute_paired_uncertainty(
            c_arr,
            b_arr,
            periods_per_year=ppy,
            case_study=cs,
            label=label,
            n_boot=2000,
            seed=42,
        )

    if not paired:
        return {
            "cs": cs,
            "kind": benchmark_kind,
            "label": label,
            "benchmark_label": benchmark_label if benchmark_label is not None else label,
            "skip": "uncertainty_empty",
        }
    register_paired_metrics(
        cs,
        challenger_hash,
        benchmark_hash,
        paired,
        benchmark_kind=benchmark_kind,
        periods_per_year=ppy,
        case_dir=write_case_dir,
    )
    # disjoint path: paired carries n_c/n_b (post-coerce per-side sizes); use
    # min so n_overlap reflects what the bootstrap actually used. paired path:
    # no n_c/n_b, n_overlap already the post-_joint_coerce length.
    n_actual = n_overlap
    n_c = paired.get("n_c")
    n_b = paired.get("n_b")
    if n_c is not None and n_b is not None:
        n_actual = int(min(float(n_c), float(n_b)))
    return {
        "cs": cs,
        "kind": benchmark_kind,
        "label": label,
        "benchmark_label": benchmark_label if benchmark_label is not None else label,
        "n_overlap": n_actual,
        "sharpe_diff": paired.get("sharpe_diff"),
        "sharpe_diff_ci_lo": paired.get("sharpe_diff_ci95_lo"),
        "sharpe_diff_ci_hi": paired.get("sharpe_diff_ci95_hi"),
        "info_ratio": paired.get("info_ratio"),
        "p_value": paired.get("p_value"),
    }


def populate_paired_metrics(
    cs: str,
    explorer: BacktestExplorer | None = None,
    *,
    label_restriction: frozenset[str] | None = None,
    rung: dict | None = None,
    carrier_pin_predicate: pl.Expr | None = None,
    freq: str = "daily",
    verbose: bool = True,
    write_case_dir: Path | None = None,
) -> list[dict]:
    """Compute all six paired-bootstrap pair types for ``cs`` and register them.

    Extracted from the two producer loops in
    ``20_strategy_synthesis/01_aggregate_synthesis.py``, specialized to a single
    case study. The per-CS selection config that Ch20 reads from module globals
    is passed in:

    * ``label_restriction`` — ``holdout.LABEL_RESTRICTIONS.get(cs)`` (e.g.
      sp500_options → ``frozenset({'ret_to_expiry'})``); None for most CSs.
    * ``rung`` — ``{"predicate", "universe_filter", "exit_at_max_days"}`` for
      the rung-pinned CSs (sp500_options, nasdaq100_microstructure); None else.
    * ``carrier_pin_predicate`` — polars expr for the carrier-pinned CS
      (us_firm_characteristics → ``config_name == 'default_huber'``); None else.
    * ``freq`` — cadence key for periods-per-year (``FREQ_MAP[cs]``).

    ``cme_futures`` uses all defaults (no restriction, daily). Returns the list
    of per-pair summary dicts (mirrors the ``paired_rows`` + ``extra_paired_rows``
    the Ch20 producer builds); each pair is also written to
    ``backtest_paired_metrics`` via ``register_paired_metrics``.

    ``write_case_dir`` redirects the registry *write* to an alternate case dir
    (reads still come from the live tree) — used by the verification harness to
    write into a temp registry copy non-destructively.
    """
    if explorer is None:
        explorer = BacktestExplorer(cs)
    ppy = _PPY_BY_FREQ.get(freq, 252)
    rows: list[dict] = []

    # -- Pair #1: signal rank-1 (overall) ↔ equal-weight (overall) -----------
    cand = pl.concat(
        [explorer.best(stage=s, top_n=2000) for s in _PAIRED_STAGES],
        how="diagonal_relaxed",
    )
    skip_pair1 = False
    if cand.is_empty() or "backtest_hash" not in cand.columns:
        skip_pair1 = True
    if not skip_pair1:
        if "family" in cand.columns:
            cand = cand.filter(pl.col("family") != "benchmark")
        if label_restriction and "label" in cand.columns:
            cand = cand.filter(pl.col("label").is_in(list(label_restriction)))
        # NB: pair #1 (Ch20 Loop A) applies ONLY the rung restriction — no
        # carrier pin — unlike pairs #2-6 (Loop B), which apply both. Preserve
        # that asymmetry so carrier-pinned CSs (us_firm_characteristics) match.
        cand = _apply_rung_restriction(cand, rung)
        if cand.is_empty():
            skip_pair1 = True
    if not skip_pair1:
        cand1 = cand.sort("sharpe", descending=True).unique(
            subset=["prediction_hash"], keep="first", maintain_order=True
        )
        leader_hash = cand1["backtest_hash"][0]
        leader_label = cand1["label"][0] if "label" in cand1.columns else None
        if leader_label:
            bench_resolution = _benchmark_returns_from_artifact(cs, leader_label)
            chal = _aligned_returns(cs, leader_hash)
            if bench_resolution and chal is not None:
                benchmark_hash, base, resolved_bench_label = bench_resolution
                min_n = _min_paired_n(ppy)
                aligned = chal.join(base, on="timestamp", how="inner", suffix="_b")
                if aligned.height >= min_n:
                    c_arr, b_arr = _joint_coerce(
                        aligned["ret"].to_numpy(), aligned["ret_b"].to_numpy()
                    )
                    if c_arr.size >= min_n:
                        paired = compute_paired_uncertainty(
                            c_arr,
                            b_arr,
                            periods_per_year=ppy,
                            case_study=cs,
                            label=leader_label,
                            n_boot=2000,
                            seed=42,
                        )
                        if paired:
                            benchmark_kind = (
                                f"{SIGNAL_BASELINE_BY_CASE_STUDY.get(cs, 'equal_weight')}"
                                "_side_artifact"
                            )
                            register_paired_metrics(
                                cs,
                                leader_hash,
                                benchmark_hash,
                                paired,
                                benchmark_kind=benchmark_kind,
                                periods_per_year=ppy,
                                case_dir=write_case_dir,
                            )
                            rows.append(
                                {
                                    "case_study": DISPLAY_NAMES.get(cs, cs),
                                    "kind": benchmark_kind,
                                    "label": leader_label,
                                    "benchmark_label": resolved_bench_label,
                                    "sharpe_diff": paired.get("sharpe_diff"),
                                    "sharpe_diff_ci_lo": paired.get("sharpe_diff_ci95_lo"),
                                    "sharpe_diff_ci_hi": paired.get("sharpe_diff_ci95_hi"),
                                    "info_ratio": paired.get("info_ratio"),
                                    "p_value": paired.get("p_value"),
                                    "prob_wins": paired.get("prob_challenger_wins"),
                                }
                            )

    # -- Pairs #2-6: holdout + stage transitions -----------------------------
    cand = pl.concat(
        [explorer.best(stage=s, top_n=2000) for s in _PAIRED_STAGES],
        how="diagonal_relaxed",
    )
    if cand.is_empty() or "backtest_hash" not in cand.columns:
        _report(cs, rows, verbose)
        return rows
    if "family" in cand.columns:
        cand = cand.filter(pl.col("family") != "benchmark")
    if label_restriction and "label" in cand.columns:
        cand = cand.filter(pl.col("label").is_in(list(label_restriction)))
    cand = _apply_rung_restriction(cand, rung)
    cand = _apply_carrier_pin(cand, carrier_pin_predicate)
    if cand.is_empty():
        _report(cs, rows, verbose)
        return rows
    cand = cand.sort("sharpe", descending=True).unique(
        subset=["prediction_hash"], keep="first", maintain_order=True
    )

    leader = cand.row(0, named=True)
    leader_hash = leader["backtest_hash"]
    leader_phash = leader["prediction_hash"]
    leader_label = leader.get("label")
    if not leader_label:
        _report(cs, rows, verbose)
        return rows

    chal_full = _aligned_returns(cs, leader_hash)
    if chal_full is None:
        _report(cs, rows, verbose)
        return rows

    # Pair #2: cross-stage rank-1 holdout ↔ equal-weight (holdout window)
    val_spec = _val_rank1_full_spec(
        cs,
        explorer,
        label_restriction=label_restriction,
        rung=rung,
        carrier_pin_predicate=carrier_pin_predicate,
    )
    # The carrier's training hash and checkpoint are both resolved inside
    # ``_holdout_lineage_for`` from this one prediction hash, so the same-lineage
    # preference cannot be half-applied by a caller.
    ho_lineage = _holdout_lineage_for(
        cs,
        leader_label,
        strategy_spec=val_spec,
        label_restriction=label_restriction,
        rung=rung,
        prefer_prediction_hash=leader_phash,
    )
    ho_hash = ho_lineage["backtest_hash"] if ho_lineage else None
    ho_label = ho_lineage["label"] if ho_lineage else leader_label
    chal_ho = _aligned_returns(cs, ho_hash) if ho_hash else None
    bench_ho_resolved = _benchmark_returns_from_artifact(cs, ho_label, period="holdout")

    baseline_kind = SIGNAL_BASELINE_BY_CASE_STUDY.get(cs, "equal_weight")
    if bench_ho_resolved is not None and chal_ho is not None:
        bench_ho_hash, bench_ho_norm, bench_ho_label = bench_ho_resolved
        rows.append(
            _populate_pair(
                cs,
                ho_hash,
                bench_ho_hash,
                f"{baseline_kind}_holdout_side_artifact",
                chal_ho,
                bench_ho_norm,
                ppy,
                ho_label,
                benchmark_label=bench_ho_label,
                write_case_dir=write_case_dir,
            )
        )
    else:
        skip_parts: list[str] = []
        if chal_ho is None:
            skip_parts.append("no_holdout_challenger_returns")
        if bench_ho_resolved is None:
            skip_parts.append("no_holdout_benchmark_artifact")
        rows.append(
            {
                "cs": cs,
                "kind": f"{baseline_kind}_holdout_side_artifact",
                "label": ho_label,
                "benchmark_label": None,
                "skip": "+".join(skip_parts),
            }
        )

    # Pair #3: holdout rank-1 ↔ validation backtest of the SAME lineage
    if chal_ho is not None and ho_lineage is not None:
        ho_family = ho_lineage["family"]
        ho_config = ho_lineage["config_name"]
        same_lineage = (
            ho_family == leader["family"]
            and ho_config == leader["config_name"]
            and ho_label == leader_label
        )
        if same_lineage:
            val_self_hash = leader_hash
            val_self_returns = chal_full
        else:
            val_self_hash = _val_backtest_for_lineage(cs, ho_family, ho_config, ho_label)
            val_self_returns = _aligned_returns(cs, val_self_hash) if val_self_hash else None
        if val_self_hash is not None and val_self_returns is not None:
            rows.append(
                _populate_pair(
                    cs,
                    ho_hash,
                    val_self_hash,
                    "val_rank1_self",
                    chal_ho,
                    val_self_returns,
                    ppy,
                    ho_label,
                    disjoint_windows=True,
                    write_case_dir=write_case_dir,
                )
            )

    # Pairs #4-6: stage transitions on the validation rank-1 lineage
    lineage = explorer.champion_lineage(leader_phash)
    for prev_stage, this_stage, kind in [
        ("signal", "allocation", "signal_leader"),
        ("allocation", "cost_sensitivity", "allocation_leader"),
        ("cost_sensitivity", "risk_overlay", "cost_sensitivity_leader"),
    ]:
        prev_entry = lineage.get(prev_stage)
        this_entry = lineage.get(this_stage)
        if not prev_entry or not this_entry:
            continue
        prev_hash = prev_entry["backtest_hash"]
        this_hash = this_entry["backtest_hash"]
        prev_returns = _aligned_returns(cs, prev_hash)
        this_returns = _aligned_returns(cs, this_hash)
        if prev_returns is None or this_returns is None:
            continue
        rows.append(
            _populate_pair(
                cs,
                this_hash,
                prev_hash,
                kind,
                this_returns,
                prev_returns,
                ppy,
                leader_label,
                write_case_dir=write_case_dir,
            )
        )

    _report(cs, rows, verbose)
    return rows


def _report(cs: str, rows: list[dict], verbose: bool) -> None:
    if not verbose:
        return
    written = [r for r in rows if "skip" not in r]
    skipped = [r for r in rows if "skip" in r]
    print(
        f"[paired_metrics] {cs}: {len(written)} pairs written, {len(skipped)} skipped", flush=True
    )
    for r in written:
        print(f"    + {r.get('kind')}: sharpe_diff={r.get('sharpe_diff')}", flush=True)
    for r in skipped:
        print(f"    - {r.get('kind')}: {r.get('skip')}", flush=True)
