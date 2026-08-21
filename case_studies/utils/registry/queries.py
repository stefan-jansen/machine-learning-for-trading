"""Query and read APIs for experiment registry data."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import polars as pl

from ..notebook_contracts import (
    canonical_coverage_days,
    degenerate_prediction_sql,
    excluded_family_sql,
    filter_active_model_rows,
    full_coverage_prediction_sql,
)
from .specs import IDENTITY_VERSION, canonical_json
from .store import (
    _backtest_dir,
    _case_dir,
    _infer_stage,
    _open_registry,
    _prediction_dir,
    _registry_db_path,
    _run_log_dir,
    _stage_filter_clause,
    _training_dir,
)

logger = logging.getLogger(__name__)

# SQLite's SQLITE_MAX_VARIABLE_NUMBER defaults to 999 on builds prior to 3.32
# (still common on system pythons and some CI images). Chunk IN-clause
# parameters at half that to leave headroom for other bound parameters in
# the same statement.
_IN_CLAUSE_CHUNK = 500


def _table_exists(case_dir: Path, table: str) -> bool:
    """Probe ``sqlite_master`` for a table. Returns False when the registry
    is missing on disk."""
    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        return False
    db = sqlite3.connect(str(db_path))
    try:
        row = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None
    finally:
        db.close()


def _existing_prediction_hashes(case_dir: Path) -> set[str] | None:
    """Return prediction hashes that have physical parquet files, or None to skip filtering.

    Returns None only when the predictions directory doesn't exist (no filtering).
    Returns empty set when the directory exists but has no valid files (filters everything out).
    """
    pred_base = _run_log_dir(case_dir) / "predictions"
    if not pred_base.exists():
        return None
    return {d.name for d in pred_base.iterdir() if (d / "predictions.parquet").exists()}


def _query_table(
    case_dir: Path,
    query: str,
    params: tuple = (),
):
    """Execute a query and return a Polars DataFrame."""
    import polars as pl

    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        return pl.DataFrame()

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(query, params).fetchall()
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame([dict(r) for r in rows], infer_schema_length=None)
    finally:
        db.close()


def load_training_runs(
    case_study: str,
    *,
    family: str | None = None,
    label: str | None = None,
    case_dir: Path | None = None,
):
    """Load training runs, optionally filtered by family and/or label."""
    if case_dir is None:
        case_dir = _case_dir(case_study)

    query = "SELECT * FROM training_runs"
    conditions = []
    params: list[str] = []
    if family:
        conditions.append("family = ?")
        params.append(family)
    if label:
        conditions.append("label = ?")
        params.append(label)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"

    return _query_table(case_dir, query, tuple(params))


def load_prediction_sets(
    case_study: str,
    *,
    training_hash: str | None = None,
    split: str | None = None,
    current_execution_only: bool = False,
    case_dir: Path | None = None,
):
    """Load prediction sets, optionally filtered to the current training execution."""
    if case_dir is None:
        case_dir = _case_dir(case_study)

    if current_execution_only:
        query = (
            "SELECT p.* FROM prediction_sets p "
            "JOIN training_runs t ON t.training_hash = p.training_hash"
        )
        prefix = "p."
    else:
        query = "SELECT * FROM prediction_sets"
        prefix = ""
    conditions = []
    params: list[str] = []
    if training_hash:
        conditions.append(f"{prefix}training_hash = ?")
        params.append(training_hash)
    if split:
        conditions.append(f"{prefix}split = ?")
        params.append(split)
    if current_execution_only:
        conditions.append("p.created_at >= COALESCE(t.started_at, t.created_at)")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += f" ORDER BY {prefix}created_at DESC"

    return _query_table(case_dir, query, tuple(params))


def load_prediction_metrics(
    case_study: str,
    *,
    prediction_hash: str | None = None,
    case_dir: Path | None = None,
):
    """Load prediction metrics (wide format), optionally filtered."""
    if case_dir is None:
        case_dir = _case_dir(case_study)

    query = "SELECT * FROM prediction_metrics"
    conditions = []
    params: list[str] = []
    if prediction_hash:
        conditions.append("prediction_hash = ?")
        params.append(prediction_hash)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    return _query_table(case_dir, query, tuple(params))


def load_backtest_runs(
    case_study: str,
    *,
    prediction_hash: str | None = None,
    case_dir: Path | None = None,
):
    """Load backtest runs, optionally filtered by prediction_hash."""
    if case_dir is None:
        case_dir = _case_dir(case_study)

    query = "SELECT * FROM backtest_runs"
    params: list[str] = []
    if prediction_hash:
        query += " WHERE prediction_hash = ?"
        params.append(prediction_hash)
    query += " ORDER BY created_at DESC"

    return _query_table(case_dir, query, tuple(params))


def load_backtest_metrics(
    case_study: str,
    *,
    backtest_hash: str | None = None,
    case_dir: Path | None = None,
):
    """Load backtest metrics (wide format), optionally filtered.

    Selection-bias columns (``dsr``, ``dsr_pvalue``, ``expected_max_sharpe``,
    ``min_trl_periods``, ``pbo``, ``reality_check_*``, ``k_variants``,
    ``leader_min_trl``) are overridden with the effective-rank (ER) values
    from the persisted ``cohort_metrics`` table (cohort_type='family',
    leader_hash=backtest_hash) when a matching cohort row exists. Rows that
    are not the leader of any (stage, label, family) cohort retain the
    underlying ``backtest_metrics`` values (typically NULL).
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    query = "SELECT * FROM backtest_metrics"
    conditions = []
    params: list[str] = []
    if backtest_hash:
        conditions.append("backtest_hash = ?")
        params.append(backtest_hash)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    df = _query_table(case_dir, query, tuple(params))
    if df.is_empty():
        return df

    # LEFT JOIN cohort_metrics (cohort_type='family') and override the
    # selection-bias columns with ER values where available.
    hashes = df["backtest_hash"].to_list()
    overrides = {
        "dsr": "_cm_dsr",
        "dsr_pvalue": "_cm_dsr_pvalue",
        "expected_max_sharpe": "_cm_expected_max_sharpe",
        "min_trl_periods": "_cm_min_trl_periods",
        "k_variants": "_cm_k_variants",
        "leader_min_trl": "_cm_leader_min_trl",
        "pbo": "_cm_pbo",
        "pbo_n_combinations": "_cm_pbo_n_combinations",
        "pbo_median_oos_rank": "_cm_pbo_median_oos_rank",
        "pbo_mean_degradation": "_cm_pbo_mean_degradation",
        "pbo_n_folds": "_cm_pbo_n_folds",
        "reality_check_pvalue": "_cm_reality_check_pvalue",
        "reality_check_statistic": "_cm_reality_check_statistic",
        "reality_check_k": "_cm_reality_check_k",
    }

    def _ensure_legacy_columns(out: pl.DataFrame) -> pl.DataFrame:
        """Post-Phase-H: backtest_metrics dropped the legacy selection-
        bias columns. If a consumer indexes the row by these names
        (e.g., ``full["dsr"]``), missing columns raise KeyError. Add
        them as null Float64 so consumers see ``None`` instead.
        """
        missing = [c for c in overrides if c not in out.columns]
        if missing:
            out = out.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
        return out

    # Pre-migration registries (some CSes never got the cohort_metrics
    # backfill) lack the table entirely. Probe sqlite_master so a missing
    # table is distinguishable from a SQL syntax error / disk I/O error
    # downstream — narrower than the previous bare ``OperationalError``
    # catch which swallowed every SQLite failure mode (typos, locks, the
    # IN-limit overflow this function now chunks around).
    if not _table_exists(case_dir, "cohort_metrics"):
        return _ensure_legacy_columns(df)

    # ORDER BY makes the dedupe deterministic when two family cohorts share
    # a leader_hash (warning path below). SQLite row order is otherwise
    # unspecified, so the ER override values that survive `unique(keep=
    # 'first')` would silently differ run-to-run. Carrying stage/label/
    # family through the SELECT (under _cm_ aliases dropped before the
    # join) lets the warning report which cohort row was kept.
    cm_query_template = """
        SELECT leader_hash,
               stage                  AS _cm_stage,
               label                  AS _cm_label,
               family                 AS _cm_family,
               dsr_er                 AS _cm_dsr,
               dsr_er_pvalue          AS _cm_dsr_pvalue,
               expected_max_sharpe_er AS _cm_expected_max_sharpe,
               min_trl_periods_er     AS _cm_min_trl_periods,
               k_variants             AS _cm_k_variants,
               leader_min_trl         AS _cm_leader_min_trl,
               pbo                    AS _cm_pbo,
               pbo_n_combinations     AS _cm_pbo_n_combinations,
               pbo_median_oos_rank    AS _cm_pbo_median_oos_rank,
               pbo_mean_degradation   AS _cm_pbo_mean_degradation,
               pbo_n_folds            AS _cm_pbo_n_folds,
               reality_check_pvalue   AS _cm_reality_check_pvalue,
               reality_check_statistic AS _cm_reality_check_statistic,
               reality_check_k        AS _cm_reality_check_k
          FROM cohort_metrics
         WHERE cohort_type = 'family'
           AND leader_hash IN ({placeholders})
         ORDER BY stage, label, family
    """
    cm_frames: list[pl.DataFrame] = []
    for start in range(0, len(hashes), _IN_CLAUSE_CHUNK):
        chunk = hashes[start : start + _IN_CLAUSE_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        chunk_df = _query_table(
            case_dir, cm_query_template.format(placeholders=placeholders), tuple(chunk)
        )
        if not chunk_df.is_empty():
            cm_frames.append(chunk_df)
    if not cm_frames:
        return _ensure_legacy_columns(df)
    cm_df = pl.concat(cm_frames, how="vertical_relaxed")

    # Guard against duplicate (leader_hash) rows in cohort_metrics. The
    # unique index is on (cohort_type, stage, label, family), so the same
    # backtest_hash can be the leader of two different family cohorts.
    # When that happens, a naive LEFT JOIN fans out — one row of
    # backtest_metrics becomes multiple rows. Detect and warn, keeping
    # the first occurrence so the join still has exactly the original
    # row cardinality.
    if cm_df["leader_hash"].n_unique() < cm_df.height:
        dupe_hashes = (
            cm_df.group_by("leader_hash")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)["leader_hash"]
            .to_list()
        )
        # ORDER BY stage, label, family in the chunk SQL plus this
        # final sort make `unique(keep='first')` deterministic across
        # runs. Without it, SQLite row order is unspecified and the
        # surviving ER values for any duplicate-leader hashes would
        # silently churn.
        cm_df = cm_df.sort(["leader_hash", "_cm_stage", "_cm_label", "_cm_family"])
        cm_df = cm_df.unique(subset=["leader_hash"], keep="first")
        kept = (
            cm_df.filter(pl.col("leader_hash").is_in(dupe_hashes))
            .select(
                pl.col("leader_hash"),
                pl.col("_cm_stage").alias("stage"),
                pl.col("_cm_label").alias("label"),
                pl.col("_cm_family").alias("family"),
            )
            .head(5)
            .to_dicts()
        )
        logger.warning(
            "cohort_metrics has duplicate family-cohort leader_hash rows for %s "
            "(n=%d); keeping first by ORDER BY stage, label, family. "
            "Kept (first 5): %s",
            case_study,
            len(dupe_hashes),
            kept,
        )

    # Drop the stage/label/family helper columns — they exist only to
    # make the dedupe deterministic and surface the kept row in the
    # warning above.
    cm_df = cm_df.drop(["_cm_stage", "_cm_label", "_cm_family"])

    df = df.join(cm_df, left_on="backtest_hash", right_on="leader_hash", how="left")
    exprs = []
    for base, cm_col in overrides.items():
        if cm_col not in df.columns:
            continue
        if base in df.columns:
            # Both present (pre-Phase-H registries): coalesce — cohort
            # wins where the row is a family leader, backtest_metrics
            # passes through otherwise.
            exprs.append(pl.coalesce(pl.col(cm_col), pl.col(base)).alias(base))
        else:
            # Post-Phase-H: backtest_metrics no longer carries the
            # legacy column; alias the cohort_metrics value under the
            # base name so consumers keep working.
            exprs.append(pl.col(cm_col).alias(base))
    if exprs:
        df = df.with_columns(exprs)
    drop_cols = [c for c in overrides.values() if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)
    return _ensure_legacy_columns(df)


def load_backtest_fold_metrics(
    case_study: str,
    *,
    backtest_hash: str | None = None,
    case_dir: Path | None = None,
):
    """Load per-fold backtest metrics from `backtest_fold_metrics`.

    Wide-format fold-level breakdown (Sharpe, Sortino, max DD, ann return,
    Calmar with 95% CIs and bootstrap metadata). Filter by backtest_hash to
    pull a single rank-1 lineage.
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    query = "SELECT * FROM backtest_fold_metrics"
    conditions = []
    params: list[str] = []
    if backtest_hash:
        conditions.append("backtest_hash = ?")
        params.append(backtest_hash)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY fold_id"

    return _query_table(case_dir, query, tuple(params))


# ---------------------------------------------------------------------------
# Cross-case-study queries
# ---------------------------------------------------------------------------


def load_all_training_runs():
    """Load training runs from all case studies."""
    import polars as pl

    from utils.paths import REPO_ROOT

    cs_dir = REPO_ROOT / "case_studies"
    frames = []
    for d in sorted(cs_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        db_path = _registry_db_path(d)
        if not db_path.exists():
            continue
        df = _query_table(d, "SELECT * FROM training_runs")
        if len(df) > 0:
            frames.append(df.with_columns(pl.lit(d.name).alias("case_study")))
    return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()


def load_all_prediction_metrics():
    """Load prediction metrics from all case studies."""
    import polars as pl

    from utils.paths import REPO_ROOT

    cs_dir = REPO_ROOT / "case_studies"
    frames = []
    for d in sorted(cs_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        db_path = _registry_db_path(d)
        if not db_path.exists():
            continue
        df = _query_table(d, "SELECT * FROM prediction_metrics")
        if len(df) > 0:
            frames.append(df.with_columns(pl.lit(d.name).alias("case_study")))
    return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()


# ---------------------------------------------------------------------------
# Backtest-oriented queries
# ---------------------------------------------------------------------------


def _spans_identity_generations(case_dir: Path) -> bool:
    """True when ``training_runs`` holds rows from more than one identity generation.

    A NULL ``identity_version`` is a generation, not an absence: it is what a row written
    before the field existed carries. So is a store with no such column at all. One
    generation, whichever it is, is not something a query can select across.
    """
    try:
        versions = _query_table(case_dir, "SELECT DISTINCT identity_version FROM training_runs", ())
    except sqlite3.OperationalError:
        return False
    return versions.height > 1


def load_prediction_index(
    case_study: str,
    *,
    label: str | None = None,
    split: str | None = None,
    family: str | None = None,
    case_dir: Path | None = None,
):
    """Load a prediction index joining training_runs, prediction_sets, and IC metrics.

    Returns a DataFrame with columns:
        prediction_hash, training_hash, family, config_name, label,
        split, checkpoint_value, source, ic_mean, predictions_path

    The ``source`` column follows the convention ``{family}/{split}/{config_name}``.
    """
    import polars as pl

    if case_dir is None:
        case_dir = _case_dir(case_study)

    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        print(f"\n  No model registry found for '{case_study}'.")
        print("  Run a model notebook first to generate predictions:")
        print(f"    uv run python case_studies/{case_study}/06_linear.py\n")
        return pl.DataFrame()

    query = """
        SELECT
            p.prediction_hash,
            p.training_hash,
            t.family,
            t.config_name,
            t.label,
            p.split,
            p.checkpoint_value,
            m.ic_mean
        FROM prediction_sets p
        JOIN training_runs t ON p.training_hash = t.training_hash
        LEFT JOIN prediction_metrics m
            ON p.prediction_hash = m.prediction_hash
    """
    conditions = []
    params: list[str | int] = []
    exclude_clause, exclude_params = excluded_family_sql(case_study, "t.family", for_backtest=True)
    # A registry may hold pre-rebuild rows beside rebuild-era ones. Selecting across both
    # generations compares models fitted under different identity rules, so a backtest must
    # see only the current generation. Legacy rows predate the field and carry NULL.
    #
    # Applied only when the store actually spans generations. The guard is against selecting
    # ACROSS them, and a store holding one cannot. Filtering unconditionally emptied every
    # registry whose rows predate the field - including CI's seeded fixture, whose
    # training_runs has no identity_version column at all - and took three case-study jobs
    # from green to "No predictions found" in the backtest downstream of the model stage.
    if _spans_identity_generations(case_dir):
        conditions.append("t.identity_version = ?")
        params.append(IDENTITY_VERSION)
    if label:
        conditions.append("t.label = ?")
        params.append(label)
    if split and split != "all":
        conditions.append("p.split = ?")
        params.append(split)
    if family:
        conditions.append("t.family = ?")
        params.append(family)
    if exclude_clause:
        conditions.append(exclude_clause.removeprefix(" AND "))
        params.extend(exclude_params)
    # Exclude prediction sets with any constant-prediction (NULL-IC) fold so they
    # never enter a backtest sweep — see degenerate_prediction_sql().
    conditions.append(degenerate_prediction_sql("p.prediction_hash").removeprefix(" AND "))
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY m.ic_mean DESC NULLS LAST"

    df = _query_table(case_dir, query, tuple(params))
    if df.is_empty():
        return df
    df = filter_active_model_rows(df, case_study)
    if df.is_empty():
        return df

    # Build source string: {family}/{split}/{config_name}
    df = df.with_columns(
        (
            pl.col("family") + pl.lit("/") + pl.col("split") + pl.lit("/") + pl.col("config_name")
        ).alias("source"),
    )

    # Add predictions_path and filter to entries with physical files.
    # Seeded registries may contain production hashes without parquet files;
    # CI model runs produce fresh entries with real files.
    pred_base = _run_log_dir(case_dir) / "predictions"
    df = df.with_columns(
        (
            pl.lit(str(pred_base))
            + pl.lit("/")
            + pl.col("prediction_hash")
            + pl.lit("/predictions.parquet")
        ).alias("predictions_path"),
    )
    existing = _existing_prediction_hashes(case_dir)
    if existing is not None:
        df = df.filter(pl.col("prediction_hash").is_in(list(existing)))

    return df


# ---------------------------------------------------------------------------
# Spec file I/O
# ---------------------------------------------------------------------------


def read_training_spec(
    case_study: str,
    training_hash: str,
    *,
    case_dir: Path | None = None,
) -> dict:
    """Read the spec.json for a training run."""
    if case_dir is None:
        case_dir = _case_dir(case_study)
    path = _training_dir(case_dir, training_hash) / "spec.json"
    if not path.exists():
        # Fall back to config.json for migrated entries
        path = _training_dir(case_dir, training_hash) / "config.json"
        config = json.loads(path.read_text())
        return config.get("spec", config)
    return json.loads(path.read_text())


def read_backtest_spec(
    case_study: str,
    backtest_hash: str,
    *,
    case_dir: Path | None = None,
) -> dict:
    """Read the spec.json for a backtest run."""
    if case_dir is None:
        case_dir = _case_dir(case_study)
    path = _backtest_dir(case_dir, backtest_hash) / "spec.json"
    if not path.exists():
        path = _backtest_dir(case_dir, backtest_hash) / "config.json"
        config = json.loads(path.read_text())
        return config.get("spec", config)
    return json.loads(path.read_text())


def read_predictions(
    case_study: str,
    prediction_hash: str,
    *,
    case_dir: Path | None = None,
):
    """Read predictions.parquet for a prediction set."""
    import polars as pl

    if case_dir is None:
        case_dir = _case_dir(case_study)
    path = _prediction_dir(case_dir, prediction_hash) / "predictions.parquet"
    df = pl.read_parquet(path)
    # Normalize legacy column names to canonical schema
    renames = {}
    if "actual" in df.columns and "y_true" not in df.columns:
        renames["actual"] = "y_true"
    if "prediction" in df.columns and "y_score" not in df.columns:
        renames["prediction"] = "y_score"
    if "fold" in df.columns and "fold_id" not in df.columns:
        renames["fold"] = "fold_id"
    if renames:
        df = df.rename(renames)
    return df


# ---------------------------------------------------------------------------
# Filesystem helpers (public)
# ---------------------------------------------------------------------------


def training_dir(case_study: str, training_hash: str, *, case_dir: Path | None = None) -> Path:
    """Get the filesystem directory for a training run."""
    if case_dir is None:
        case_dir = _case_dir(case_study)
    return _training_dir(case_dir, training_hash)


def prediction_dir(case_study: str, prediction_hash: str, *, case_dir: Path | None = None) -> Path:
    """Get the filesystem directory for a prediction set."""
    if case_dir is None:
        case_dir = _case_dir(case_study)
    return _prediction_dir(case_dir, prediction_hash)


def backtest_dir(case_study: str, backtest_hash: str, *, case_dir: Path | None = None) -> Path:
    """Get the filesystem directory for a backtest run."""
    if case_dir is None:
        case_dir = _case_dir(case_study)
    return _backtest_dir(case_dir, backtest_hash)


def load_existing_backtest_hashes(
    case_study: str,
    *,
    stage: str | None = None,
    case_dir: Path | None = None,
) -> set[str]:
    """Return registered backtest hashes, optionally filtered by stage."""
    if case_dir is None:
        case_dir = _case_dir(case_study)

    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        return set()

    query = "SELECT backtest_hash FROM backtest_runs"
    params: tuple[str, ...] = ()
    if stage is not None:
        query += " WHERE stage = ?"
        params = (stage,)

    db = sqlite3.connect(str(db_path))
    try:
        rows = db.execute(query, params).fetchall()
    finally:
        db.close()

    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# Source string helper
# ---------------------------------------------------------------------------


def model_source(family: str, config_name: str | None) -> str:
    """Canonical source identifier: ``{family}/{config_name}``.

    This is the ONE function that produces source strings.
    Used by registration, backtest loading, and results JSON.
    """
    if config_name:
        return f"{family}/{config_name}"
    return family


# ---------------------------------------------------------------------------
# Cross-level queries
# ---------------------------------------------------------------------------


# Canonical schema for resolve_best_predictions() output. Returning a
# schema-stable empty DataFrame on no-match lets downstream
# `.select("source", ...)` surface "(no matching rows)" rather than a
# cryptic ColumnNotFoundError with "valid columns: []".
_BEST_PREDICTIONS_SCHEMA: dict[str, pl.DataType] = {
    "prediction_hash": pl.Utf8,
    "training_hash": pl.Utf8,
    "family": pl.Utf8,
    "config_name": pl.Utf8,
    "label": pl.Utf8,
    "split": pl.Utf8,
    "checkpoint_value": pl.Float64,
    "sharpe": pl.Float64,
    "source": pl.Utf8,
    "predictions_path": pl.Utf8,
}


def _finalize_predictions_frame(df: pl.DataFrame, case_study: str, case_dir: Path):
    """Shared tail for ``resolve_best_predictions``'s two coverage-window paths.

    Drops excluded families, attaches ``source``/``predictions_path``, and keeps only
    prediction sets with a physical parquet on disk. Returns the frame unchanged (still
    possibly empty) rather than the caller's schema-stable empty sentinel; callers apply
    that sentinel themselves.
    """
    import polars as pl

    if df.is_empty():
        return df
    df = filter_active_model_rows(df, case_study)
    if df.is_empty():
        return df

    df = df.with_columns(
        (pl.col("family") + pl.lit("/") + pl.col("config_name").fill_null(pl.lit("default"))).alias(
            "source"
        ),
    )

    pred_base = _run_log_dir(case_dir) / "predictions"
    df = df.with_columns(
        (
            pl.lit(str(pred_base))
            + pl.lit("/")
            + pl.col("prediction_hash")
            + pl.lit("/predictions.parquet")
        ).alias("predictions_path"),
    )
    existing = _existing_prediction_hashes(case_dir)
    if existing is not None:
        df = df.filter(pl.col("prediction_hash").is_in(list(existing)))

    return df


def _canonical_family_coverage_bar(
    case_study: str,
    label: str,
    split: str,
    case_dir: Path,
) -> dict[str, int]:
    """Max in-window coverage per family, over EVERY prediction set for the label.

    The eligibility bar has to be the family's best coverage, not the best among the
    rows that happen to carry a backtest at the stage being resolved. Taking the max
    after a stage filter lowers the bar whenever the family's full-coverage prediction
    was never backtested at that stage, and a partial-coverage prediction then
    qualifies - which is exactly what ``full_coverage_prediction_sql`` prevents on the
    raw path, where the ``MAX(ic_n_days)`` subquery ranges over all of
    ``(split, family, label)`` with no stage restriction.

    Requires a ``prediction_metrics`` row, matching the raw path's implicit
    requirement (``full_coverage_prediction_sql`` joins ``pm``): a prediction set
    whose metrics were never computed can't set the bar other candidates must meet.
    """
    exclude_clause, exclude_params = excluded_family_sql(case_study, "t.family")
    degenerate_clause = degenerate_prediction_sql("p.prediction_hash")
    query = f"""
        SELECT p.prediction_hash, t.family
        FROM prediction_sets p
        JOIN training_runs t ON p.training_hash = t.training_hash
        JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
        WHERE t.label = ?
          AND p.split = ?
          {exclude_clause}
          {degenerate_clause}
    """
    universe = _query_table(case_dir, query, tuple([label, split] + exclude_params))
    bar: dict[str, int] = {}
    if universe.is_empty():
        return bar
    for prediction_hash, family in universe.select("prediction_hash", "family").iter_rows():
        days = canonical_coverage_days(case_study, label, split, prediction_hash, case_dir)
        if days is None:
            continue
        if days > bar.get(family, -1):
            bar[family] = days
    return bar


def _resolve_best_predictions_canonical(
    case_study: str,
    label: str,
    *,
    split: str | None,
    top_n: int,
    stage: str,
    chapter_filter: str | None,
    universe_filter: str | None,
    case_dir: Path | None,
    checkpoints_per_config: int,
):
    """``resolve_best_predictions(coverage_window="canonical")``.

    Re-ranks using ``canonical_coverage_days`` (in-canonical-window day counts recomputed
    from each prediction's parquet) in place of the SQL-embedded
    ``full_coverage_prediction_sql`` comparison on stored ``ic_n_days``. Requires
    an explicit ``split`` — the canonical window is per ``(case_study, label, split)``.
    """
    import polars as pl

    if not split:
        raise ValueError("coverage_window='canonical' requires an explicit split")

    if case_dir is None:
        case_dir = _case_dir(case_study)
    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        logger.warning("No registry.db found for '%s'", case_study)
        return pl.DataFrame(schema=_BEST_PREDICTIONS_SCHEMA)

    stage_clause, stage_params = _stage_filter_clause(stage, chapter_filter)
    exclude_clause, exclude_params = excluded_family_sql(case_study, "t.family")
    degenerate_clause = degenerate_prediction_sql("p.prediction_hash")

    universe_clause = ""
    params: list[str] = [label] + stage_params + exclude_params + [split]
    if universe_filter:
        universe_clause = "AND json_extract(b.spec_json, '$.strategy.signal.universe_filter') = ?"
        params.append(universe_filter)

    # Same per_prediction shape as the raw path, minus full_coverage_prediction_sql —
    # that comparison is replaced below by canonical_coverage_days, computed in Python.
    query = f"""
        SELECT
            p.prediction_hash,
            p.training_hash,
            p.split,
            p.checkpoint_value,
            t.family,
            t.config_name,
            t.label,
            MAX(bm.sharpe) AS sharpe
        FROM backtest_metrics bm
        JOIN backtest_runs b ON bm.backtest_hash = b.backtest_hash
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        JOIN training_runs t ON p.training_hash = t.training_hash
        JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
        WHERE t.label = ?
          {stage_clause}
          {exclude_clause}
          {degenerate_clause}
          AND p.split = ?
          {universe_clause}
        GROUP BY p.prediction_hash
    """
    per_prediction = _query_table(case_dir, query, tuple(params))
    if per_prediction.is_empty():
        return pl.DataFrame(schema=_BEST_PREDICTIONS_SCHEMA)

    in_window = {
        h: canonical_coverage_days(case_study, label, split, h, case_dir)
        for h in per_prediction["prediction_hash"].unique().to_list()
    }
    bar = _canonical_family_coverage_bar(case_study, label, split, case_dir)
    per_prediction = per_prediction.with_columns(
        pl.col("prediction_hash").replace_strict(in_window, default=None).alias("_in_window_days"),
        pl.col("family").replace_strict(bar, default=None).alias("_family_bar"),
    ).filter(pl.col("_in_window_days").is_not_null())
    if per_prediction.is_empty():
        return pl.DataFrame(schema=_BEST_PREDICTIONS_SCHEMA)
    per_prediction = per_prediction.filter(pl.col("_in_window_days") == pl.col("_family_bar")).drop(
        "_in_window_days", "_family_bar"
    )

    top_cfgs = (
        per_prediction.group_by(["family", "config_name"])
        .agg(pl.col("sharpe").max().alias("_best"))
        .sort("_best", descending=True)
        .head(top_n)
        .select(["family", "config_name"])
    )
    ranked = (
        # `nulls_equal` mirrors the raw path's `pp.config_name IS tc.config_name`
        # (see `_resolve_best_predictions`). `config_name` is nullable, group_by
        # keeps a null group, and a default inner join would then drop every row
        # of it - silently removing that family from the ranking even when it
        # holds the best Sharpe.
        per_prediction.join(top_cfgs, on=["family", "config_name"], how="inner", nulls_equal=True)
        .sort(["sharpe", "checkpoint_value"], descending=[True, True])
        .with_columns(pl.int_range(pl.len()).over(["family", "config_name"]).alias("_rn"))
        .filter(pl.col("_rn") < max(1, int(checkpoints_per_config)))
        .drop("_rn")
        .sort("sharpe", descending=True)
    )
    df = ranked.select(
        "prediction_hash",
        "training_hash",
        "family",
        "config_name",
        "label",
        "split",
        "checkpoint_value",
        "sharpe",
    )
    df = _finalize_predictions_frame(df, case_study, case_dir)
    if df.is_empty():
        return pl.DataFrame(schema=_BEST_PREDICTIONS_SCHEMA)
    return df


def resolve_best_predictions(
    case_study: str,
    label: str,
    *,
    split: str | None,
    top_n: int = 10,
    stage: str = "signal",
    chapter_filter: str | None = None,
    universe_filter: str | None = None,
    case_dir: Path | None = None,
    checkpoints_per_config: int = 1,
    coverage_window: str = "raw",
):
    """Return top-N prediction hashes ranked by backtest Sharpe at a given stage.

    Joins backtest_metrics (Sharpe) -> backtest_runs -> prediction_sets ->
    training_runs. Filters by the ``stage`` column.

    Parameters
    ----------
    case_study : str
        Case study identifier.
    label : str
        Label to filter training runs by.
    split : str or None
        Required. Pass "validation" for chapter pipeline (allocation /
        cost / risk) lineage; pass "holdout" for honest out-of-sample
        evaluation; pass ``None`` only for cross-split audit queries
        where mixing is intentional. Mixing splits silently selects
        holdout predictions over validation, which produces NULL
        allocation/cost/risk rows in chapter prose.
    top_n : int
        Number of top model configs to return. The unit is a distinct
        ``(family, config_name)``, not a prediction: with
        ``checkpoints_per_config > 1`` the frame holds more than ``top_n`` rows.
    stage : str
        Pipeline stage to filter by: "signal", "allocation", etc.
    chapter_filter : str, optional
        Chapter-based filter (e.g. "ch16"). Converted to stage internally.
    case_dir : Path, optional
        Override case study directory.
    checkpoints_per_config : int
        How many checkpoints each advancing config contributes, best first.
    coverage_window : str
        "raw" (default): compares stored ``ic_n_days`` via
        ``full_coverage_prediction_sql`` — unchanged behavior for every existing
        caller. "canonical": recomputes coverage from each prediction set's parquet,
        bounded to ``canonical_window(case_study, label, split)``. Use when a case
        study's prediction sets predate a canonical-window change and raw
        ``ic_n_days`` differs across peers only by out-of-window dates.
        Requires ``split``.

    Returns
    -------
    pl.DataFrame
        Columns: prediction_hash, training_hash, family, config_name,
        source, label, split, sharpe, predictions_path
    """
    import polars as pl

    if coverage_window == "canonical":
        return _resolve_best_predictions_canonical(
            case_study,
            label,
            split=split,
            top_n=top_n,
            stage=stage,
            chapter_filter=chapter_filter,
            universe_filter=universe_filter,
            case_dir=case_dir,
            checkpoints_per_config=checkpoints_per_config,
        )

    if case_dir is None:
        case_dir = _case_dir(case_study)

    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        logger.warning("No registry.db found for '%s'", case_study)
        return pl.DataFrame(schema=_BEST_PREDICTIONS_SCHEMA)

    # Find top-N distinct model configs, each represented by its best
    # ``checkpoints_per_config`` checkpoints.
    #
    # A single model config (e.g., latent_factors/sae) may have many prediction
    # hashes — one per checkpoint epoch. Without dedup, the top-N list can be
    # dominated by checkpoints of a single model, hiding model diversity.
    #
    # Three-stage approach:
    #   1. per_prediction: best Sharpe per prediction_hash
    #   2. top_configs:    rank (family, config_name) by their best Sharpe, take top_n
    #   3. ranked:         within each advancing config, keep its best N checkpoints
    #
    # Prefer stage column; fall back to spec_json LIKE for un-migrated DBs.
    stage_clause, stage_params = _stage_filter_clause(stage, chapter_filter)
    exclude_clause, exclude_params = excluded_family_sql(case_study, "t.family")
    degenerate_clause = degenerate_prediction_sql("p.prediction_hash")

    split_clause = ""
    params: list[str] = [label] + stage_params + exclude_params
    if split:
        split_clause = "AND p.split = ?"
        params.append(split)
    universe_clause = ""
    if universe_filter:
        universe_clause = "AND json_extract(b.spec_json, '$.strategy.signal.universe_filter') = ?"
        params.append(universe_filter)
    params.append(str(top_n))
    params.append(str(max(1, int(checkpoints_per_config))))

    query = f"""
        WITH per_prediction AS (
            -- Best backtest Sharpe per prediction_hash (a prediction may have
            -- been backtested with multiple signal methods)
            SELECT
                p.prediction_hash,
                p.training_hash,
                p.split,
                p.checkpoint_value,
                t.family,
                t.config_name,
                t.label,
                MAX(bm.sharpe) AS sharpe
            FROM backtest_metrics bm
            JOIN backtest_runs b ON bm.backtest_hash = b.backtest_hash
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
            WHERE t.label = ?
              {stage_clause}
              {exclude_clause}
              {degenerate_clause}
              {full_coverage_prediction_sql("p", "t", "pm")}
              {split_clause}
              {universe_clause}
            GROUP BY p.prediction_hash
        ),
        top_configs AS (
            -- The advancing model configs, ranked by their best checkpoint's Sharpe
            SELECT family, config_name
            FROM per_prediction
            GROUP BY family, config_name
            ORDER BY MAX(sharpe) DESC
            LIMIT ?
        ),
        ranked AS (
            -- Rank checkpoints within each advancing config, best first
            SELECT pp.*,
                ROW_NUMBER() OVER (
                    PARTITION BY pp.family, pp.config_name
                    ORDER BY pp.sharpe DESC, pp.checkpoint_value DESC
                ) AS rn
            FROM per_prediction pp
            JOIN top_configs tc
              ON pp.family = tc.family
             AND pp.config_name IS tc.config_name
        )
        SELECT
            prediction_hash,
            training_hash,
            family,
            config_name,
            label,
            split,
            checkpoint_value,
            sharpe
        FROM ranked
        WHERE rn <= CAST(? AS INTEGER)
        ORDER BY sharpe DESC
    """

    df = _query_table(case_dir, query, tuple(params))
    if df.is_empty():
        return pl.DataFrame(schema=_BEST_PREDICTIONS_SCHEMA)
    df = _finalize_predictions_frame(df, case_study, case_dir)
    if df.is_empty():
        return pl.DataFrame(schema=_BEST_PREDICTIONS_SCHEMA)

    return df


def _resolve_best_backtest_runs_canonical(
    case_study: str,
    label: str,
    *,
    split: str | None,
    stage: str | None,
    chapter: str | None,
    top_n: int,
    case_dir: Path | None,
):
    """``resolve_best_backtest_runs(coverage_window="canonical")``.

    Same replacement as ``_resolve_best_predictions_canonical``: drops
    ``full_coverage_prediction_sql`` and re-filters in Python using
    ``canonical_coverage_days``.
    """
    import polars as pl

    if not split:
        raise ValueError("coverage_window='canonical' requires an explicit split")

    if case_dir is None:
        case_dir = _case_dir(case_study)
    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        logger.warning("No registry.db found for '%s'", case_study)
        return pl.DataFrame()

    if stage is None and chapter:
        _chapter_to_stage = {
            "ch16": "signal",
            "ch17": "allocation",
            "ch18": "cost_sensitivity",
            "ch19": "risk_overlay",
        }
        stage = _chapter_to_stage.get(chapter, chapter)

    stage_clause, stage_params = _stage_filter_clause(stage)
    exclude_clause, exclude_params = excluded_family_sql(case_study, "t.family")
    degenerate_clause = degenerate_prediction_sql("p.prediction_hash")

    query = f"""
        SELECT
            b.backtest_hash,
            b.prediction_hash,
            b.spec_json,
            bm.sharpe,
            t.family,
            t.config_name
        FROM backtest_metrics bm
        JOIN backtest_runs b ON bm.backtest_hash = b.backtest_hash
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        JOIN training_runs t ON p.training_hash = t.training_hash
        JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
        WHERE t.label = ?
          {exclude_clause}
          {stage_clause}
          {degenerate_clause}
          AND p.split = ?
    """
    df = _query_table(case_dir, query, tuple([label] + exclude_params + stage_params + [split]))
    if df.is_empty():
        return df

    in_window = {
        h: canonical_coverage_days(case_study, label, split, h, case_dir)
        for h in df["prediction_hash"].unique().to_list()
    }
    bar = _canonical_family_coverage_bar(case_study, label, split, case_dir)
    df = df.with_columns(
        pl.col("prediction_hash").replace_strict(in_window, default=None).alias("_in_window_days"),
        pl.col("family").replace_strict(bar, default=None).alias("_family_bar"),
    ).filter(pl.col("_in_window_days").is_not_null())
    if df.is_empty():
        return df
    df = df.filter(pl.col("_in_window_days") == pl.col("_family_bar")).drop(
        "_in_window_days", "_family_bar"
    )

    df = filter_active_model_rows(df, case_study)
    if df.is_empty():
        return df

    existing = _existing_prediction_hashes(case_dir)
    if existing is not None:
        df = df.filter(pl.col("prediction_hash").is_in(list(existing)))

    df = df.sort("sharpe", descending=True).head(top_n)
    return df.select("backtest_hash", "prediction_hash", "spec_json", "sharpe")


def resolve_best_backtest_runs(
    case_study: str,
    label: str,
    *,
    split: str | None,
    stage: str | None = None,
    chapter: str | None = None,
    top_n: int = 3,
    case_dir: Path | None = None,
    coverage_window: str = "raw",
):
    """Return top-N backtest runs at a given stage, ranked by Sharpe.

    Used by Ch18 and Ch19 to find the best prediction+allocator combos
    from the allocation stage.

    Parameters
    ----------
    case_study : str
        Case study identifier.
    label : str
        Label to filter by.
    split : str or None
        Required. Pass "validation" for chapter pipeline (cost / risk)
        lineage; pass "holdout" for honest out-of-sample evaluation;
        pass ``None`` only for cross-split audit queries where mixing
        is intentional. See ``resolve_best_predictions`` for the bug
        history that motivated making this required.
    stage : str, optional
        Pipeline stage: "signal", "allocation", "cost_sensitivity",
        "risk_overlay".
    chapter : str, optional
        **Deprecated.** Legacy chapter tag (e.g., "ch17").
        Ignored when ``stage`` is provided.
    top_n : int
        Number of top runs to return.
    case_dir : Path, optional
        Override case study directory.
    coverage_window : str
        See ``resolve_best_predictions``. "raw" (default) is unchanged behavior;
        "canonical" requires an explicit ``split``.

    Returns
    -------
    pl.DataFrame
        Columns: backtest_hash, prediction_hash, spec_json, sharpe
    """
    import polars as pl

    if coverage_window == "canonical":
        return _resolve_best_backtest_runs_canonical(
            case_study,
            label,
            split=split,
            stage=stage,
            chapter=chapter,
            top_n=top_n,
            case_dir=case_dir,
        )

    if case_dir is None:
        case_dir = _case_dir(case_study)

    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        logger.warning("No registry.db found for '%s'", case_study)
        return pl.DataFrame()

    # Map chapter to stage
    if stage is None and chapter:
        _chapter_to_stage = {
            "ch16": "signal",
            "ch17": "allocation",
            "ch18": "cost_sensitivity",
            "ch19": "risk_overlay",
        }
        stage = _chapter_to_stage.get(chapter, chapter)

    stage_clause, stage_params = _stage_filter_clause(stage)
    exclude_clause, exclude_params = excluded_family_sql(case_study, "t.family")
    degenerate_clause = degenerate_prediction_sql("p.prediction_hash")

    split_clause = ""
    split_params: list[str] = []
    if split:
        split_clause = "AND p.split = ?"
        split_params = [split]

    query = f"""
        SELECT
            b.backtest_hash,
            b.prediction_hash,
            b.spec_json,
            bm.sharpe,
            t.family,
            t.config_name
        FROM backtest_metrics bm
        JOIN backtest_runs b ON bm.backtest_hash = b.backtest_hash
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        JOIN training_runs t ON p.training_hash = t.training_hash
        JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
        WHERE t.label = ?
          {exclude_clause}
          {stage_clause}
          {degenerate_clause}
          {full_coverage_prediction_sql("p", "t", "pm")}
          {split_clause}
        ORDER BY bm.sharpe DESC
        LIMIT ?
    """
    df = _query_table(
        case_dir,
        query,
        tuple([label] + exclude_params + stage_params + split_params + [str(top_n)]),
    )
    if df.is_empty():
        return df
    df = filter_active_model_rows(df, case_study)
    if df.is_empty():
        return df

    # Filter to entries with physical prediction files
    existing = _existing_prediction_hashes(case_dir)
    if existing is not None:
        df = df.filter(pl.col("prediction_hash").is_in(list(existing)))

    return df.select("backtest_hash", "prediction_hash", "spec_json", "sharpe")


def load_paired_metrics(
    case_study: str,
    *,
    challenger_hash: str | None = None,
    benchmark_kind: str | None = None,
    case_dir: Path | None = None,
):
    """Load ``backtest_paired_metrics`` rows for the given case study.

    Filter by ``challenger_hash`` and/or ``benchmark_kind``; both optional.
    Returns an empty Polars DataFrame if no rows match.
    """
    import polars as pl

    if case_dir is None:
        case_dir = _case_dir(case_study)

    where: list[str] = []
    params: list[str] = []
    if challenger_hash is not None:
        where.append("challenger_hash = ?")
        params.append(challenger_hash)
    if benchmark_kind is not None:
        where.append("benchmark_kind = ?")
        params.append(benchmark_kind)

    sql = "SELECT * FROM backtest_paired_metrics"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY computed_at DESC"

    db_path = _registry_db_path(case_dir)
    if not db_path.exists():
        return pl.DataFrame()
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(sql, tuple(params)).fetchall()
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame([dict(r) for r in rows], infer_schema_length=None)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Migration: backfill stage column
# ---------------------------------------------------------------------------


def backfill_stages(case_study: str | None = None) -> dict[str, int]:
    """Backfill the ``stage`` column for existing backtest_runs rows.

    For each row where ``stage IS NULL``, infer the stage from spec_json:
      - Has "chapter":"ch19" or risk rules -> risk_overlay
      - Has "chapter":"ch18" -> cost_sensitivity
      - Has "chapter":"ch17" or allocation key -> allocation
      - Everything else -> signal

    Parameters
    ----------
    case_study : str, optional
        Single case study to backfill. If None, backfill ALL case studies.

    Returns
    -------
    dict[str, int]
        Mapping of case_study_id -> number of rows updated.
    """
    from utils.paths import get_case_study_dir

    if case_study:
        case_studies = [case_study]
    else:
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

    result = {}
    for cs_id in case_studies:
        try:
            case_dir = get_case_study_dir(cs_id)
        except Exception as exc:
            logger.warning("Could not resolve case study dir for %s: %s", cs_id, exc)
            continue
        db_path = _registry_db_path(case_dir)
        if not db_path.exists():
            continue

        db = _open_registry(case_dir)
        try:
            # Check if stage column exists (the CREATE TABLE IF NOT EXISTS
            # in _open_registry should have added it already)
            cursor = db.execute(
                "SELECT backtest_hash, spec_json, prediction_hash "
                "FROM backtest_runs WHERE stage IS NULL"
            )
            rows = cursor.fetchall()
            updated = 0
            for b_hash, spec_json_str, pred_hash in rows:
                if not spec_json_str:
                    continue
                try:
                    spec = json.loads(spec_json_str)
                except (json.JSONDecodeError, TypeError):
                    continue
                stage = _infer_stage(spec, case_dir=case_dir, prediction_hash=pred_hash)
                db.execute(
                    "UPDATE backtest_runs SET stage = ? WHERE backtest_hash = ?",
                    (stage, b_hash),
                )
                updated += 1
            db.commit()
            result[cs_id] = updated
            if updated:
                logger.info("Backfilled %d stage values for '%s'", updated, cs_id)
        finally:
            db.close()

    return result
