"""Cross-dataset analytics for case study insights notebooks.

Replaces per-notebook JSON-loading patterns with direct registry queries.
Each function returns a Polars DataFrame ready for analysis and visualization.

Usage::

    from case_studies.utils.analytics import (
        CASE_STUDY_IDS, PRIMARY_LABELS, SHORT_NAMES, DATASET_META,
        load_model_ic, load_best_ic_per_family, load_chapter_backtests,
    )
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl

from case_studies.utils.backtest_presets import cost_view, strategy_view
from case_studies.utils.notebook_contracts import (
    degenerate_prediction_sql,
    full_coverage_prediction_sql,
)
from utils.paths import REPO_ROOT

CASE_STUDY_META = {
    "etfs": {"display_name": "ETFs", "chapter_track": "Ch6 to Ch21"},
    "crypto_perps_funding": {
        "display_name": "Crypto Perps Funding",
        "chapter_track": "Ch6 to Ch12",
    },
    "nasdaq100_microstructure": {
        "display_name": "NASDAQ-100 Microstructure",
        "chapter_track": "Ch6 to Ch12",
    },
    "sp500_equity_option_analytics": {
        "display_name": "S&P 500 Equity+Options",
        "chapter_track": "Ch6 to Ch21",
    },
    "us_firm_characteristics": {
        "display_name": "US Firm Characteristics",
        "chapter_track": "Ch6 to Ch14",
    },
    "fx_pairs": {"display_name": "FX Pairs", "chapter_track": "Ch6 to Ch17"},
    "cme_futures": {"display_name": "CME Futures", "chapter_track": "Ch6 to Ch17"},
    "sp500_options": {
        "display_name": "S&P 500 Options",
        "chapter_track": "Ch6 to Ch21",
    },
    "us_equities_panel": {
        "display_name": "US Equities Panel",
        "chapter_track": "Ch6 to Ch14",
    },
}

DISPLAY_NAMES = {k: v["display_name"] for k, v in CASE_STUDY_META.items()}
CHAPTER_TRACKS = {k: v["chapter_track"] for k, v in CASE_STUDY_META.items()}

# ---------------------------------------------------------------------------
# Canonical metadata (single source — replaces duplicated dicts in notebooks)
# ---------------------------------------------------------------------------

CASE_STUDY_IDS = list(CASE_STUDY_META.keys())

PRIMARY_LABELS = {
    "etfs": "fwd_ret_21d",
    "crypto_perps_funding": "fwd_ret_8h",
    "nasdaq100_microstructure": "fwd_ret_15m",
    "sp500_equity_option_analytics": "fwd_ret_5d",
    "us_firm_characteristics": "fwd_ret_1m",
    "fx_pairs": "fwd_ret_1d",
    "cme_futures": "fwd_ret_5d",
    "sp500_options": "ret_to_expiry",
    "us_equities_panel": "fwd_ret_1d",
}

SHORT_NAMES = {
    "etfs": "ETFs",
    "crypto_perps_funding": "Crypto",
    "nasdaq100_microstructure": "NQ100",
    "sp500_equity_option_analytics": "SP500 Eq+Opt",
    "us_firm_characteristics": "US Firms",
    "fx_pairs": "FX",
    "cme_futures": "CME Futures",
    "sp500_options": "SP500 Options",
    "us_equities_panel": "US Equities",
}

DATASET_META = {
    "etfs": {"frequency": "Daily", "entities": 99, "horizon": "21d"},
    "crypto_perps_funding": {"frequency": "8-hourly", "entities": 21, "horizon": "8h"},
    "nasdaq100_microstructure": {"frequency": "15-min", "entities": 114, "horizon": "15m"},
    "sp500_equity_option_analytics": {"frequency": "Daily", "entities": 638, "horizon": "5d"},
    "us_firm_characteristics": {"frequency": "Monthly", "entities": 2483, "horizon": "1m"},
    "fx_pairs": {"frequency": "Daily", "entities": 20, "horizon": "1d"},
    "cme_futures": {"frequency": "Daily", "entities": 30, "horizon": "5d"},
    "sp500_options": {"frequency": "Daily", "entities": 612, "horizon": "dh-10d"},
    "us_equities_panel": {"frequency": "Daily", "entities": 3199, "horizon": "1d"},
}

CADENCE_MAP = {
    "etfs": "monthly",
    "crypto_perps_funding": "8-hourly",
    "nasdaq100_microstructure": "15-min",
    "cme_futures": "monthly",
    "fx_pairs": "daily",
    "sp500_equity_option_analytics": "monthly",
    "sp500_options": "monthly",
    "us_equities_panel": "daily",
    "us_firm_characteristics": "monthly",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cs_dir(case_study: str | None = None) -> Path:
    """Resolve case study root, respecting ML4T_OUTPUT_DIR for test isolation."""
    import os

    output_dir = os.environ.get("ML4T_OUTPUT_DIR")
    if output_dir:
        base = Path(output_dir)
        if case_study and (base / case_study / "run_log" / "registry.db").exists():
            return base
    return REPO_ROOT / "case_studies"


def registry_path(case_study: str) -> Path:
    return _cs_dir(case_study) / case_study / "run_log" / "registry.db"


def _has_column(db_path: Path, table: str, column: str) -> bool:
    """Whether ``table`` in this registry carries ``column``.

    The metrics tables are widened as new metrics appear, so which columns a registry has
    depends on when it was last written and which task types it holds. Naming an absent column
    in a SELECT is a hard error, not a null.
    """
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            return any(row[1] == column for row in db.execute(f"PRAGMA table_info({table})"))
    except sqlite3.Error:
        return False


def _query(db_path: Path, sql: str, params: tuple = ()) -> pl.DataFrame:
    """Execute SQL on a registry and return a Polars DataFrame."""
    if not db_path.exists():
        return pl.DataFrame()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame([dict(r) for r in rows], infer_schema_length=None)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Model IC queries
# ---------------------------------------------------------------------------


def load_model_ic(
    families: list[str] | str | None = None,
    *,
    split: str = "validation",
    case_studies: list[str] | None = None,
    require_full_coverage: bool = True,
) -> pl.DataFrame:
    """Load IC metrics across case studies for specified model families.

    Returns a DataFrame with columns:
        case_study, family, config_name, label, split,
        checkpoint_value, ic_mean, ic_std

    A model family can hold prediction sets scored over different numbers of
    decision days, because a run that failed partway still registers rows for the
    folds it completed. Their scores are averages over different periods, so
    comparing them measures the periods as much as the models, and the shorter
    window usually scores higher. Rows are therefore restricted to the maximum
    ``ic_n_days`` within each ``(split, family, label)`` before anything is
    returned.

    Ordering is fully determined: ``ic_mean`` descending, then
    ``prediction_hash``, so a caller taking the first row of a tie gets the same
    row on every machine.

    **This is a diagnostic query, not a selection.** Ranking configurations by IC
    selects nothing anywhere in the pipeline; a configuration is chosen on
    validation backtest Sharpe. See ``reference/CASE_STUDY_PIPELINE.md``.

    Parameters
    ----------
    families : str or list of str, optional
        Filter to specific families (e.g. "linear", ["gbm", "deep_learning"]).
        None returns all families.
    split : str
        Prediction split to filter by ("validation" or "holdout").
    case_studies : list of str, optional
        Case studies to query. None = all.
    require_full_coverage : bool
        Keep only the maximum-coverage rows per (split, family, label). Set False
        for an inventory of everything scored, which cannot be compared across
        rows.
    """
    if isinstance(families, str):
        families = [families]
    cs_list = case_studies or CASE_STUDY_IDS

    frames = []
    for cs_id in cs_list:
        db_path = registry_path(cs_id)
        if not db_path.exists():
            continue

        family_clause = ""
        params: list = []
        if families:
            placeholders = ",".join("?" * len(families))
            family_clause = f"AND t.family IN ({placeholders})"
            params.extend(families)

        params.append(split)

        # Registries predating the daily-uncertainty backfill have no
        # ``ic_mean_daily`` column at all, so probe before referencing it.
        with sqlite3.connect(str(db_path)) as probe_con:
            pm_cols = {row[1] for row in probe_con.execute("PRAGMA table_info(prediction_metrics)")}
        ic_expr = (
            "COALESCE(pm.ic_mean_daily, pm.ic_mean)" if "ic_mean_daily" in pm_cols else "pm.ic_mean"
        )
        # A registry predating that backfill has no coverage column, and one
        # caught mid-backfill has the column with nothing in it. In both cases
        # the rows cannot be shown to span the same decision days, and the
        # clause below would return nothing at all rather than everything
        # unguarded. Probe for a usable value, not just the column, and report
        # per row which of the two happened: that beats failing outright on a
        # legacy registry and beats letting a caller assume a guard that never
        # ran.
        coverage_usable = False
        if "ic_n_days" in pm_cols:
            with sqlite3.connect(str(db_path)) as probe_con:
                coverage_usable = (
                    probe_con.execute(
                        "SELECT 1 FROM prediction_metrics WHERE ic_n_days IS NOT NULL LIMIT 1"
                    ).fetchone()
                    is not None
                )
        coverage_enforced = require_full_coverage and coverage_usable
        coverage_clause = full_coverage_prediction_sql("p", "t", "pm") if coverage_enforced else ""

        sql = f"""
            SELECT
                t.family,
                t.config_name,
                t.label,
                p.split,
                p.checkpoint_value,
                p.prediction_hash,
                -- Daily-pooled IC is the statistic the HAC interval, `13_model_analysis`,
                -- and the book's Table 14.3 all report; `ic_mean` is the legacy fold-mean
                -- kept only as a fallback for rows predating the daily backfill. Ranking on
                -- the legacy column while quoting the daily interval mixes two statistics.
                {ic_expr} AS ic_mean,
                pm.ic_std
            FROM training_runs t
            JOIN prediction_sets p ON t.training_hash = p.training_hash
            JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
            WHERE 1=1 {family_clause}
              AND p.split = ?
              {degenerate_prediction_sql("p.prediction_hash")}
              {coverage_clause}
            ORDER BY ic_mean DESC NULLS LAST, p.prediction_hash ASC
        """
        df = _query(db_path, sql, tuple(params))
        if len(df) > 0:
            # Cast columns to expected types (handles Null-type columns from empty aggregates)
            col_types = {"checkpoint_value": pl.Int64, "ic_mean": pl.Float64, "ic_std": pl.Float64}
            casts = [
                pl.col(c).cast(t, strict=False) for c, t in col_types.items() if c in df.columns
            ]
            if casts:
                df = df.with_columns(casts)
            frames.append(
                df.with_columns(
                    pl.lit(cs_id).alias("case_study"),
                    pl.lit(coverage_enforced).alias("coverage_enforced"),
                )
            )

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def load_classification_metrics(
    families: list[str] | str | None = None,
    *,
    split: str = "validation",
    case_studies: list[str] | None = None,
) -> pl.DataFrame:
    """Load classification metrics (AUC-ROC, accuracy, etc.) from registries.

    ``auc`` is the cross-sectional AUC - computed within each date and then averaged, the same
    shape as IC - and it is what the rows are ordered by. ``auc_roc`` is carried alongside it
    unchanged: that one pools every ``(entity, date)`` row in a fold into a single ROC, so it
    also rewards a model for the base rate moving through time. The two agree to 0.0002 in
    ``us_firm_characteristics``, whose monthly cross-section is close to balanced, and part
    company in ``sp500_equity_option_analytics`` at 0.5308 pooled against 0.5063
    cross-sectional. Ranking on the pooled figure was ranking partly on the calendar.

    Returns a DataFrame with columns:
        case_study, family, config_name, label, split,
        ic_mean, auc, auc_roc, accuracy, balanced_accuracy, log_loss, brier_score, auc_pr
    """
    if isinstance(families, str):
        families = [families]
    cs_list = case_studies or CASE_STUDY_IDS

    frames = []
    for cs_id in cs_list:
        db_path = registry_path(cs_id)
        if not db_path.exists():
            continue

        family_clause = ""
        params: list = []
        if families:
            placeholders = ",".join("?" * len(families))
            family_clause = f"AND t.family IN ({placeholders})"
            params.extend(families)

        params.append(split)

        # A registry written before the cross-sectional block existed has no such column, and
        # naming it in the SELECT is an error rather than a null. Fall back to the pooled value
        # under the same output name so the caller reads one column either way.
        has_daily = _has_column(db_path, "prediction_metrics", "auc_mean_daily")
        auc_select = "pm.auc_mean_daily AS auc" if has_daily else "pm.auc_roc AS auc"
        auc_order = "pm.auc_mean_daily" if has_daily else "pm.auc_roc"

        sql = f"""
            SELECT
                t.family,
                t.config_name,
                t.label,
                p.split,
                pm.ic_mean,
                {auc_select},
                pm.auc_roc,
                pm.accuracy,
                pm.balanced_accuracy,
                pm.log_loss,
                pm.brier_score,
                pm.auc_pr,
                pm.task_type
            FROM training_runs t
            JOIN prediction_sets p ON t.training_hash = p.training_hash
            JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
            WHERE 1=1 {family_clause}
              AND p.split = ?
              AND pm.task_type = 'classification'
            ORDER BY {auc_order} DESC NULLS LAST
        """
        df = _query(db_path, sql, tuple(params))
        if len(df) > 0:
            frames.append(df.with_columns(pl.lit(cs_id).alias("case_study")))

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def load_best_ic_per_family(
    families: list[str] | None = None,
    *,
    split: str = "validation",
    case_studies: list[str] | None = None,
    use_primary_label: bool = True,
) -> pl.DataFrame:
    """Highest-IC row per family per case study (one row per family-case_study pair).

    Restricted to maximum-coverage prediction sets by ``load_model_ic``, so the
    rows compared against each other were scored over the same decision days.
    Exact ties resolve on ``prediction_hash``, so the same registry returns the
    same row everywhere.

    **A diagnostic view, not a selection.** IC ranks nothing in the pipeline; a
    configuration is chosen on validation backtest Sharpe. Callers that display
    a family's strongest score are using it correctly; a caller choosing what to
    trade or fit from this result is not.

    Returns: case_study, display_name, family, config_name, label, ic_mean
    """
    all_ic = load_model_ic(families, split=split, case_studies=case_studies)
    if all_ic.is_empty():
        return pl.DataFrame()

    if use_primary_label:
        label_rows = [
            {"case_study": cs, "primary_label": lbl} for cs, lbl in PRIMARY_LABELS.items()
        ]
        label_df = pl.DataFrame(label_rows)
        all_ic = all_ic.join(label_df, on="case_study").filter(
            pl.col("label") == pl.col("primary_label")
        )

    tie_break = ["prediction_hash"] if "prediction_hash" in all_ic.columns else []
    best = (
        all_ic.sort(
            ["ic_mean", *tie_break],
            descending=[True, *[False] * len(tie_break)],
            nulls_last=True,
        )
        .group_by(["case_study", "family"])
        .first()
        .select("case_study", "family", "config_name", "label", "ic_mean")
    )

    # Add display names
    name_rows = [{"case_study": k, "display_name": v} for k, v in SHORT_NAMES.items()]
    name_df = pl.DataFrame(name_rows)
    best = best.join(name_df, on="case_study", how="left")

    return best.sort(["case_study", "family"])


# ---------------------------------------------------------------------------
# Backtest queries
# ---------------------------------------------------------------------------


_CHAPTER_TO_STAGE = {
    "ch16": "signal",
    "ch17": "allocation",
    "ch18": "cost_sensitivity",
    "ch19": "risk_overlay",
}


def load_chapter_backtests(
    chapter: str,
    *,
    stage: str | None = None,
    case_studies: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pl.DataFrame:
    """Load backtest results for a pipeline stage across case studies.

    Returns a DataFrame with columns:
        case_study, display_name, backtest_hash, prediction_hash,
        family, config_name, label, spec_json,
        plus one column per metric (sharpe, sortino, etc.)

    Parameters
    ----------
    chapter : str
        Legacy chapter tag (e.g. "ch16"). Mapped to stage automatically.
    stage : str, optional
        Pipeline stage to filter by. Overrides chapter mapping.
    case_studies : list of str, optional
        Case studies to query. None = all.
    metrics : list of str, optional
        Metrics to pivot. None = all available.
    """
    cs_list = case_studies or CASE_STUDY_IDS
    resolved_stage = stage or _CHAPTER_TO_STAGE.get(chapter, chapter)

    frames = []
    for cs_id in cs_list:
        db_path = registry_path(cs_id)
        if not db_path.exists():
            continue

        # Prefer stage column; fall back to spec_json LIKE for un-migrated DBs
        sql = """
            SELECT
                b.backtest_hash,
                b.prediction_hash,
                b.spec_json,
                t.family,
                t.config_name,
                t.label,
                bm.*
            FROM backtest_runs b
            JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            WHERE b.stage = ?
        """
        df = _query(db_path, sql, (resolved_stage,))
        if len(df) > 0:
            frames.append(df.with_columns(pl.lit(cs_id).alias("case_study")))

    if not frames:
        return pl.DataFrame()

    result = pl.concat(frames, how="diagonal_relaxed")

    # Filter to requested metrics (select only those columns)
    if metrics:
        # Keep meta columns + requested metric columns
        meta_cols = [
            "case_study",
            "backtest_hash",
            "prediction_hash",
            "spec_json",
            "family",
            "config_name",
            "label",
        ]
        keep = [c for c in meta_cols if c in result.columns]
        keep += [c for c in metrics if c in result.columns]
        result = result.select(keep)

    # Drop internal columns from bm.* join
    drop_cols = [c for c in ["computed_at"] if c in result.columns]
    if drop_cols:
        result = result.drop(drop_cols)

    # Add display names
    name_rows = [{"case_study": k, "display_name": v} for k, v in SHORT_NAMES.items()]
    name_df = pl.DataFrame(name_rows)
    result = result.join(name_df, on="case_study", how="left")

    return result


# ---------------------------------------------------------------------------
# Spec parsing helpers
# ---------------------------------------------------------------------------


def parse_backtest_spec(spec_json: str) -> dict:
    """Parse a backtest spec_json string into a dict."""
    return json.loads(spec_json)


def extract_cost_bps(spec_json: str) -> float:
    """Extract total cost (commission + slippage) in bps from a backtest spec."""
    spec = parse_backtest_spec(spec_json)
    costs = cost_view(spec)
    return float(costs.get("commission_bps", 0.0)) + float(costs.get("slippage_bps", 0.0))


def extract_allocator(spec_json: str) -> str:
    """Extract allocation method from a backtest spec."""
    spec = parse_backtest_spec(spec_json)
    allocation = strategy_view(spec).get("allocation") or {}
    return allocation.get("method", "unknown")


# nasdaq100_microstructure deploys the cost-feasible *ensemble* carrier
# (developed in 20.4). resolve_canonical_rank1_lineage returns the
# non-deployed full-universe val-max gbm instead, so the ensemble's training
# hash and validation backtest are pinned here. See agents
# UNCERTAINTY_ARCHITECTURE / Ch20 audit.
_NASDAQ_ENSEMBLE_TRAINING_HASH = "a9f04b886b9a"
_NASDAQ_ENSEMBLE_VAL_HASH = "4e939dee0a5f"


def _strategy_signature(spec_json: str) -> str:
    """Identity of a backtest's signal + allocation config, ignoring cost.

    The cost sweep holds the strategy fixed and varies only commission +
    slippage, so every cost-grid run for one carrier shares this signature.
    A single training hash can host more than one carrier (e.g. a signal-only
    eq-weight series alongside an allocator series); matching the signature
    selects the deployed one rather than a sibling.
    """
    sv = strategy_view(parse_backtest_spec(spec_json))
    return json.dumps([sv.get("signal"), sv.get("allocation")], sort_keys=True)


def load_carrier_cost_curves(case_studies: list[str] | None = None) -> pl.DataFrame:
    """Cost-sensitivity curves for each case study's *deployed carrier*.

    The carrier is the highest-validation-Sharpe configuration across the
    signal, allocation, and risk-overlay stages, resolved via
    ``resolve_canonical_rank1_lineage``. The cost sweep (Ch18) holds that
    carrier's signal and allocation fixed while varying commission +
    slippage, so the breakeven implied here is the carrier's own cost
    survival — not that of whichever allocator happened to be best at zero
    cost (which need not be the deployed strategy). Rows are matched to the
    carrier's exact strategy signature, so a sibling series sharing the
    training hash (e.g. a signal-only eq-weight run) is excluded.

    Returns tidy rows ``[case_study, display_name, cadence, label,
    allocator, cost_bps, sharpe, total_return, max_drawdown]`` on the
    validation split, sorted by ``cost_bps`` within each case study.
    """
    # Lazy import avoids a module-load cycle (strategy_analysis is heavier).
    from case_studies.utils.strategy_analysis import resolve_canonical_rank1_lineage

    cs_list = case_studies or CASE_STUDY_IDS
    cost_sql = """
        SELECT b.spec_json, t.label, bm.sharpe, bm.total_return, bm.max_drawdown
        FROM backtest_runs b
        JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        JOIN training_runs t ON p.training_hash = t.training_hash
        WHERE b.stage = 'cost_sensitivity' AND p.split = 'validation'
          AND p.training_hash = ?
    """

    frames = []
    for cs_id in cs_list:
        db_path = registry_path(cs_id)
        if not db_path.exists():
            continue

        # Resolve the deployed carrier's training hash and its strategy spec.
        if cs_id == "nasdaq100_microstructure":
            training_hash = _NASDAQ_ENSEMBLE_TRAINING_HASH
            spec_df = _query(
                db_path,
                "SELECT spec_json FROM backtest_runs WHERE backtest_hash LIKE ?",
                (_NASDAQ_ENSEMBLE_VAL_HASH + "%",),
            )
        else:
            try:
                lin = resolve_canonical_rank1_lineage(cs_id)
            except Exception:
                continue
            training_hash = lin.get("training_hash")
            spec_df = _query(
                db_path,
                "SELECT spec_json FROM backtest_runs WHERE backtest_hash = ?",
                (lin.get("val_backtest_hash"),),
            )
        if not training_hash or spec_df.is_empty():
            continue
        carrier_sig = _strategy_signature(spec_df["spec_json"][0])

        df = _query(db_path, cost_sql, (training_hash,))
        if df.is_empty():
            continue
        df = df.with_columns(
            cost_bps=pl.col("spec_json").map_elements(extract_cost_bps, return_dtype=pl.Float64),
            allocator=pl.col("spec_json").map_elements(extract_allocator, return_dtype=pl.Utf8),
            signature=pl.col("spec_json").map_elements(_strategy_signature, return_dtype=pl.Utf8),
            case_study=pl.lit(cs_id),
        ).filter(pl.col("signature") == carrier_sig)
        if df.is_empty():
            continue
        frames.append(
            df.select(
                "case_study",
                "label",
                "allocator",
                "cost_bps",
                "sharpe",
                "total_return",
                "max_drawdown",
            )
        )

    if not frames:
        return pl.DataFrame()

    result = pl.concat(frames, how="diagonal_relaxed")
    # A signature-matched carrier should be one row per cost level, but guard
    # against any residual duplicate deterministically (keep best Sharpe).
    result = result.sort(["case_study", "cost_bps", "sharpe"]).unique(
        subset=["case_study", "cost_bps"], keep="last", maintain_order=True
    )
    name_df = pl.DataFrame([{"case_study": k, "display_name": v} for k, v in SHORT_NAMES.items()])
    return (
        result.join(name_df, on="case_study", how="left")
        .with_columns(cadence=pl.col("case_study").replace(CADENCE_MAP))
        .sort("case_study", "cost_bps")
    )


# Deprecated private aliases. Thirty notebook cells import these names with their leading
# underscore, which is what made them look private in the first place; each is a
# cross-module interface and is now public. The aliases keep those callers working until
# each notebook is re-executed with the public name, and go when the last one moves.
_registry_path = registry_path
