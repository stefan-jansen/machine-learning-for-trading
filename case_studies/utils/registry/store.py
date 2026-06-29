"""Storage, schema, and filesystem helpers for the experiment registry."""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .specs import _validate_spec, canonical_json, training_hash_from_spec

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)
UTC = UTC

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REGISTRY_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS training_runs (
    training_hash     TEXT PRIMARY KEY,
    family            TEXT NOT NULL,
    label             TEXT NOT NULL,
    config_name       TEXT,
    spec_json         TEXT,
    created_at        TEXT NOT NULL,
    git_commit        TEXT,
    entry_point       TEXT,
    started_at        TEXT,
    elapsed_s         REAL
);

CREATE INDEX IF NOT EXISTS idx_training_family_label ON training_runs(family, label);
CREATE INDEX IF NOT EXISTS idx_training_config_name ON training_runs(config_name);

CREATE TABLE IF NOT EXISTS prediction_sets (
    prediction_hash     TEXT PRIMARY KEY,
    training_hash       TEXT NOT NULL REFERENCES training_runs(training_hash),
    checkpoint_value    INTEGER,
    checkpoint_kind     TEXT,
    split               TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pred_training ON prediction_sets(training_hash);
CREATE INDEX IF NOT EXISTS idx_pred_split ON prediction_sets(split);

CREATE TABLE IF NOT EXISTS prediction_metrics (
    prediction_hash  TEXT PRIMARY KEY REFERENCES prediction_sets(prediction_hash),
    computed_at      TEXT NOT NULL,
    ic_mean REAL, ic_std REAL, ic_t REAL, n_folds REAL,
    pct_positive REAL, task_type TEXT,
    accuracy REAL, balanced_accuracy REAL, auc_roc REAL, auc_pr REAL,
    log_loss REAL, brier_score REAL
);

CREATE TABLE IF NOT EXISTS fold_metrics (
    prediction_hash  TEXT NOT NULL REFERENCES prediction_sets(prediction_hash),
    fold_id          INTEGER NOT NULL,
    computed_at      TEXT NOT NULL,
    ic REAL, ic_std REAL, n_entities REAL,
    rmse REAL, mae REAL,
    accuracy REAL, balanced_accuracy REAL, auc_roc REAL, auc_pr REAL,
    log_loss REAL, brier_score REAL,
    "auc_class_-1" REAL, auc_class_0 REAL, auc_class_1 REAL,
    PRIMARY KEY (prediction_hash, fold_id)
);

CREATE INDEX IF NOT EXISTS idx_fold_metrics_pred ON fold_metrics(prediction_hash);

CREATE TABLE IF NOT EXISTS backtest_runs (
    backtest_hash    TEXT PRIMARY KEY,
    prediction_hash  TEXT NOT NULL REFERENCES prediction_sets(prediction_hash),
    spec_json        TEXT,
    stage            TEXT,
    created_at       TEXT NOT NULL,
    git_commit       TEXT,
    started_at       TEXT,
    elapsed_s        REAL
);

CREATE INDEX IF NOT EXISTS idx_backtest_pred ON backtest_runs(prediction_hash);
CREATE INDEX IF NOT EXISTS idx_backtest_stage ON backtest_runs(stage);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    backtest_hash    TEXT PRIMARY KEY REFERENCES backtest_runs(backtest_hash),
    computed_at      TEXT NOT NULL,
    sharpe REAL, sortino REAL, total_return REAL, max_drawdown REAL,
    cagr REAL, volatility REAL, calmar REAL, omega REAL, stability REAL,
    tail_ratio REAL, win_rate REAL, kurtosis REAL, skewness REAL,
    var_95 REAL, cvar_95 REAL, n_periods REAL,
    num_trades REAL, total_commission REAL, total_slippage REAL, avg_turnover REAL
);

CREATE TABLE IF NOT EXISTS backtest_fold_metrics (
    backtest_hash    TEXT NOT NULL REFERENCES backtest_runs(backtest_hash),
    fold_id          INTEGER NOT NULL,
    computed_at      TEXT NOT NULL,
    sharpe REAL, sortino REAL, total_return REAL, max_drawdown REAL,
    cagr REAL, volatility REAL, calmar REAL, omega REAL, stability REAL,
    tail_ratio REAL, win_rate REAL, kurtosis REAL, skewness REAL,
    var_95 REAL, cvar_95 REAL, n_days REAL,
    PRIMARY KEY (backtest_hash, fold_id)
);

CREATE INDEX IF NOT EXISTS idx_bt_fold_metrics ON backtest_fold_metrics(backtest_hash);

CREATE TABLE IF NOT EXISTS causal_runs (
    causal_hash      TEXT PRIMARY KEY,
    label            TEXT NOT NULL,
    treatment        TEXT,
    confounders_json TEXT,
    embargo          INTEGER,
    n_folds          INTEGER,
    n_obs            INTEGER,
    dml_effect       REAL,
    dml_se_hac       REAL,
    p_value_hac      REAL,
    naive_effect     REAL,
    confounding_bias_pct REAL,
    refutation_p     REAL,
    spec_json        TEXT,
    notebook         TEXT,
    started_at       TEXT,
    elapsed_s        REAL,
    git_commit       TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_causal_label ON causal_runs(label);

CREATE TABLE IF NOT EXISTS backtest_paired_metrics (
    challenger_hash       TEXT NOT NULL REFERENCES backtest_runs(backtest_hash),
    benchmark_hash        TEXT NOT NULL,
    benchmark_kind        TEXT,
    periods_per_year      INTEGER,
    bootstrap_block_length INTEGER,
    bootstrap_n           INTEGER,
    sharpe_diff           REAL,
    sharpe_diff_ci95_lo   REAL,
    sharpe_diff_ci95_hi   REAL,
    ret_diff              REAL,
    ret_diff_ci95_lo      REAL,
    ret_diff_ci95_hi      REAL,
    max_dd_diff           REAL,
    max_dd_diff_ci95_lo   REAL,
    max_dd_diff_ci95_hi   REAL,
    info_ratio            REAL,
    info_ratio_ci95_lo    REAL,
    info_ratio_ci95_hi    REAL,
    prob_challenger_wins  REAL,
    p_value               REAL,
    computed_at           TEXT NOT NULL,
    PRIMARY KEY (challenger_hash, benchmark_hash)
);

CREATE INDEX IF NOT EXISTS idx_paired_challenger ON backtest_paired_metrics(challenger_hash);
CREATE INDEX IF NOT EXISTS idx_paired_kind ON backtest_paired_metrics(benchmark_kind);

CREATE TABLE IF NOT EXISTS cohort_metrics (
    cohort_type   TEXT NOT NULL,
    stage         TEXT,
    label         TEXT NOT NULL,
    family        TEXT,
    leader_hash   TEXT NOT NULL REFERENCES backtest_runs(backtest_hash),
    k_variants                  INTEGER NOT NULL,
    periods_per_year            REAL NOT NULL,
    computed_at                 TEXT NOT NULL,
    n_trials_effective_mp       REAL,
    n_trials_effective_er       REAL,
    dsr_raw                     REAL, dsr_raw_pvalue REAL,
    expected_max_sharpe_raw     REAL, min_trl_periods_raw REAL,
    dsr_mp                      REAL, dsr_mp_pvalue  REAL,
    expected_max_sharpe_mp      REAL, min_trl_periods_mp  REAL,
    dsr_er                      REAL, dsr_er_pvalue  REAL,
    expected_max_sharpe_er      REAL, min_trl_periods_er  REAL,
    ras_leader                  REAL,
    ras_complexity              REAL,
    ras_n_strategies            REAL,
    ras_pvalue                  REAL,
    reality_check_pvalue        REAL,
    reality_check_statistic     REAL,
    reality_check_k             REAL,
    pbo                         REAL,
    pbo_n_combinations          REAL,
    pbo_median_oos_rank         REAL,
    pbo_mean_degradation        REAL,
    pbo_n_folds                 REAL,
    leader_sharpe               REAL,
    leader_sortino              REAL,
    leader_min_trl              REAL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cohort_unique
    ON cohort_metrics(cohort_type, COALESCE(stage, ''), label, COALESCE(family, ''));
CREATE INDEX IF NOT EXISTS idx_cohort_leader ON cohort_metrics(leader_hash);
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


VALID_STAGES = {"signal", "allocation", "cost_sensitivity", "risk_overlay"}


def _stage_filter_clause(
    stage: str | None, chapter_filter: str | None = None
) -> tuple[str, list[str]]:
    """Build a SQL WHERE clause fragment for stage filtering.

    Returns (clause, params) using parameterized queries.
    """
    if stage:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage {stage!r}, expected one of {VALID_STAGES}")
        return "AND b.stage = ?", [stage]

    if chapter_filter:
        pattern = f'%"chapter":"{chapter_filter}"%'
        return "AND b.spec_json LIKE ?", [pattern]
    return "AND b.stage = 'signal'", []


def _infer_stage(
    spec: dict,
    *,
    case_dir: Path | None = None,
    prediction_hash: str | None = None,
) -> str:
    """Infer pipeline stage from strategy_spec content.

    When ``case_dir`` and ``prediction_hash`` are both provided, this also
    checks ``prediction_sets.split``: if the prediction is ``split='holdout'``,
    the stage is forced to ``'holdout'`` regardless of the spec content. This
    keeps holdout backtests universally identifiable via ``stage='holdout'``
    even when the rank-1 lineage cascades into an allocation- or
    risk_overlay-stage strategy.
    """
    if case_dir is not None and prediction_hash is not None:
        try:
            db = sqlite3.connect(_registry_db_path(case_dir))
            try:
                row = db.execute(
                    "SELECT split FROM prediction_sets WHERE prediction_hash = ?",
                    (prediction_hash,),
                ).fetchone()
            finally:
                db.close()
            if row is not None and row[0] == "holdout":
                return "holdout"
        except sqlite3.DatabaseError:
            # Registry not initialized yet — fall through to spec inference.
            pass
    strategy = spec.get("strategy", spec)
    risk = strategy.get("risk", {})
    if risk and risk.get("name") != "baseline":
        return "risk_overlay"
    # Cost sensitivity: explicit chapter tag of ch18, or caller should set explicitly
    chapter = spec.get("chapter", "")
    if chapter == "ch18":
        return "cost_sensitivity"
    if "allocation" in strategy:
        alloc = strategy["allocation"]
        if isinstance(alloc, dict) and alloc.get("method", "equal_weight") != "equal_weight":
            return "allocation"
    return "signal"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _case_dir(case_study: str) -> Path:
    """Resolve case study directory, respecting ML4T_OUTPUT_DIR."""
    from utils.paths import get_case_study_dir

    return get_case_study_dir(case_study)


def _run_log_dir(case_dir: Path) -> Path:
    return case_dir / "run_log"


def _registry_db_path(case_dir: Path) -> Path:
    return _run_log_dir(case_dir) / "registry.db"


def _training_dir(case_dir: Path, t_hash: str) -> Path:
    return _run_log_dir(case_dir) / "training" / t_hash


def get_training_dir(case_study: str, spec: dict) -> Path:
    """Pre-compute the training artifact directory for a spec.

    Use this to get the save_dir BEFORE training, so model artifacts
    (boosters, coefficients, learning curves) go directly to the registry.

    Usage::

        spec = build_training_spec("gbm", "default_mse", "fwd_ret_21d", ...)
        train_dir = get_training_dir("etfs", spec)
        result = train_gbm_config(config, folds, save_dir=train_dir, ...)
        register_training_run("etfs", spec=spec)  # spec.json written, boosters already in place

    Returns
    -------
    Path
        ``run_log/training/{hash}/`` for this spec.
    """
    spec = _validate_spec(spec)
    t_hash = training_hash_from_spec(spec)
    case_dir = _case_dir(case_study)
    d = _training_dir(case_dir, t_hash)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prediction_dir(case_dir: Path, p_hash: str) -> Path:
    return _run_log_dir(case_dir) / "predictions" / p_hash


def _backtest_dir(case_dir: Path, b_hash: str) -> Path:
    return _run_log_dir(case_dir) / "backtest" / b_hash


# ---------------------------------------------------------------------------
# DB connection and migration
# ---------------------------------------------------------------------------


def _open_registry(case_dir: Path) -> sqlite3.Connection:
    db_path = _registry_db_path(case_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Tolerate concurrent writers (parallel backfills, notebook + script
    # writing the same registry). 120s SQLite-driver timeout + 60s server-side
    # busy_timeout. Without these, any momentary writer conflict raises
    # "database is locked" instantly.
    db = sqlite3.connect(str(db_path), timeout=120.0)
    db.execute("PRAGMA busy_timeout = 60000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    # Migrate existing DBs before running CREATE TABLE IF NOT EXISTS
    _migrate_registry(db)
    db.executescript(REGISTRY_SCHEMA_SQL)
    return db


def _table_has_column(db: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a table has a specific column."""
    return column in {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_registry(db: sqlite3.Connection) -> None:
    """Apply incremental schema migrations to an existing registry."""
    # Check if backtest_runs table exists at all
    tables = {
        row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "backtest_runs" not in tables:
        return  # Fresh DB — schema script will create everything

    # Migration 1: add stage column to backtest_runs
    cols = {row[1] for row in db.execute("PRAGMA table_info(backtest_runs)").fetchall()}
    if "stage" not in cols:
        db.execute("ALTER TABLE backtest_runs ADD COLUMN stage TEXT")
        db.execute("CREATE INDEX IF NOT EXISTS idx_backtest_stage ON backtest_runs(stage)")

    # Migration 2: add runtime columns to training_runs
    if "training_runs" in tables:
        tr_cols = {row[1] for row in db.execute("PRAGMA table_info(training_runs)").fetchall()}
        if "started_at" not in tr_cols:
            db.execute("ALTER TABLE training_runs ADD COLUMN started_at TEXT")
        if "elapsed_s" not in tr_cols:
            db.execute("ALTER TABLE training_runs ADD COLUMN elapsed_s REAL")

    # Migration 2b: add runtime columns to backtest_runs
    if "started_at" not in cols:
        db.execute("ALTER TABLE backtest_runs ADD COLUMN started_at TEXT")
    if "elapsed_s" not in cols:
        db.execute("ALTER TABLE backtest_runs ADD COLUMN elapsed_s REAL")

    # Migration 3: tall → wide metric tables
    if "prediction_metrics" in tables:
        pm_cols = {row[1] for row in db.execute("PRAGMA table_info(prediction_metrics)").fetchall()}
        if "metric" in pm_cols:
            _migrate_tall_to_wide(db)

    # Migration 4: task_type from numeric (1.0 / 0.0) to string
    # ("classification" / "regression"). The schema is now TEXT but legacy
    # rows still carry the float encoding; consumers that filter
    # ``task_type = 'classification'`` would otherwise miss them.
    if "prediction_metrics" in tables:
        pm_cols = {row[1] for row in db.execute("PRAGMA table_info(prediction_metrics)").fetchall()}
        if "task_type" in pm_cols:
            db.execute(
                "UPDATE prediction_metrics SET task_type = 'classification' "
                "WHERE task_type IN (1, 1.0, '1', '1.0')"
            )
            db.execute(
                "UPDATE prediction_metrics SET task_type = 'regression' "
                "WHERE task_type IN (0, 0.0, '0', '0.0')"
            )
    if "fold_metrics" in tables:
        fm_cols = {row[1] for row in db.execute("PRAGMA table_info(fold_metrics)").fetchall()}
        if "task_type" in fm_cols:
            db.execute(
                "UPDATE fold_metrics SET task_type = 'classification' "
                "WHERE task_type IN (1, 1.0, '1', '1.0')"
            )
            db.execute(
                "UPDATE fold_metrics SET task_type = 'regression' "
                "WHERE task_type IN (0, 0.0, '0', '0.0')"
            )

    db.commit()


# ---------------------------------------------------------------------------
# Migration 3: tall (metric-per-row) → wide (metric-as-column) pivot
# ---------------------------------------------------------------------------

_TALL_TO_WIDE_TABLES = {
    "prediction_metrics": {
        "key_cols": ["prediction_hash"],
        "metrics": [
            "ic_mean",
            "ic_std",
            "ic_t",
            "n_folds",
            "n_obs",
            "n_periods",
            "pct_positive",
            "task_type",
            "accuracy",
            "balanced_accuracy",
            "auc_roc",
            "auc_pr",
            "log_loss",
            "brier_score",
            "dml_effect",
            "dml_se_hac",
            "p_value_hac",
            "naive_effect",
            "confounding_bias_pct",
            "refutation_p",
            "ate",
            "ate_se",
        ],
    },
    "fold_metrics": {
        "key_cols": ["prediction_hash", "fold_id"],
        "metrics": [
            "ic",
            "ic_std",
            "n_periods",
            "n_obs",
            "n_entities",
            "rmse",
            "mae",
            "accuracy",
            "balanced_accuracy",
            "auc_roc",
            "auc_pr",
            "log_loss",
            "brier_score",
            "auc_class_-1",
            "auc_class_0",
            "auc_class_1",
        ],
    },
    "backtest_metrics": {
        "key_cols": ["backtest_hash"],
        "metrics": [
            "sharpe",
            "sortino",
            "total_return",
            "max_drawdown",
            "cagr",
            "volatility",
            "calmar",
            "omega",
            "stability",
            "tail_ratio",
            "win_rate",
            "kurtosis",
            "skewness",
            "var_95",
            "cvar_95",
            "n_periods",
            "num_trades",
            "total_commission",
            "total_slippage",
            "avg_turnover",
        ],
    },
    "backtest_fold_metrics": {
        "key_cols": ["backtest_hash", "fold_id"],
        "metrics": [
            "sharpe",
            "sortino",
            "total_return",
            "max_drawdown",
            "cagr",
            "volatility",
            "calmar",
            "omega",
            "stability",
            "tail_ratio",
            "win_rate",
            "kurtosis",
            "skewness",
            "var_95",
            "cvar_95",
            "n_days",
        ],
    },
}


def _migrate_tall_to_wide(db: sqlite3.Connection) -> None:
    """Pivot all 4 metric tables from tall (metric-per-row) to wide (metric-as-column).

    Detects any metric names in the data that aren't in the predefined list
    and adds them as columns automatically.
    """
    logger.info("Migrating metric tables from tall to wide format...")

    for table, spec in _TALL_TO_WIDE_TABLES.items():
        tall_table = f"_{table}_tall"
        key_cols = spec["key_cols"]
        known_metrics = spec["metrics"]

        # Check if this table still has tall format
        if not _table_has_column(db, table, "metric"):
            continue

        # Discover any metric names in the data not in our predefined list
        existing_metrics = {
            row[0] for row in db.execute(f"SELECT DISTINCT metric FROM {table}").fetchall()
        }
        extra_metrics = sorted(existing_metrics - set(known_metrics))
        all_metrics = known_metrics + extra_metrics

        # Rename old table
        db.execute(f"ALTER TABLE {table} RENAME TO {tall_table}")

        # Drop old user-created indexes (skip autoindexes which can't be dropped)
        for idx_row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (tall_table,),
        ).fetchall():
            if not idx_row[0].startswith("sqlite_autoindex_"):
                db.execute(f"DROP INDEX IF EXISTS {idx_row[0]}")

        # Build the pivot SELECT
        group_cols = ", ".join(key_cols)
        case_expressions = []
        for m in all_metrics:
            case_expressions.append(f"MAX(CASE WHEN metric = '{m}' THEN value END) AS \"{m}\"")

        insert_cols = key_cols + ["computed_at"] + [f'"{m}"' for m in all_metrics]
        insert_cols_str = ", ".join(insert_cols)

        pivot_sql = f"""
            INSERT INTO {table} ({insert_cols_str})
            SELECT {group_cols}, MAX(computed_at) AS computed_at,
                   {", ".join(case_expressions)}
            FROM {tall_table}
            GROUP BY {group_cols}
        """

        # Now create the new wide table via the schema script (already in REGISTRY_SCHEMA_SQL),
        # but we need to add any extra metric columns first
        # The REGISTRY_SCHEMA_SQL will be run AFTER migration by _open_registry,
        # so we create the table here manually with the known columns
        _create_wide_table(db, table, key_cols, all_metrics)

        # Pivot the data
        row_count_before = db.execute(f"SELECT COUNT(*) FROM {tall_table}").fetchone()[0]
        db.execute(pivot_sql)
        row_count_after = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        logger.info(f"  {table}: {row_count_before:,} tall rows → {row_count_after:,} wide rows")

        # Drop the old tall table
        db.execute(f"DROP TABLE {tall_table}")

    logger.info("Migration complete. Run VACUUM to reclaim space.")


def _create_wide_table(
    db: sqlite3.Connection,
    table: str,
    key_cols: list[str],
    metric_cols: list[str],
) -> None:
    """Create a wide-format metric table with the given columns."""
    # Build column definitions
    col_defs = []
    for kc in key_cols:
        if kc.endswith("_id"):
            col_defs.append(f"{kc} INTEGER NOT NULL")
        else:
            col_defs.append(f"{kc} TEXT NOT NULL")
    col_defs.append("computed_at TEXT NOT NULL")
    for m in metric_cols:
        col_defs.append(f'"{m}" REAL')

    pk_cols = ", ".join(key_cols)
    all_col_defs = ",\n    ".join(col_defs)

    sql = f"""CREATE TABLE IF NOT EXISTS {table} (
    {all_col_defs},
    PRIMARY KEY ({pk_cols})
)"""
    db.execute(sql)

    # Add index on the first key column for fold tables
    if len(key_cols) > 1:
        idx_name = f"idx_{table}_{key_cols[0].replace('_hash', '')}"
        db.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({key_cols[0]})")


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _save_parquet(path: Path, frame) -> None:
    """Write a DataFrame to parquet, handling pl.Object columns safely.

    Polars cannot write ``pl.Object``-typed columns to parquet directly
    (the Object dtype is an opaque Python object that parquet has no
    schema for). Any Object columns are converted to ``pl.String`` via
    per-element ``str()`` before writing. This makes the writer
    idempotent for DL learning curves and training logs that sometimes
    contain Object-typed diagnostic columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(frame, "write_parquet"):
        # Polars path
        try:
            import polars as _pl

            obj_cols = [c for c in frame.columns if frame[c].dtype == _pl.Object]
            if obj_cols:
                frame = frame.with_columns(
                    _pl.col(c).map_elements(str, return_dtype=_pl.String) for c in obj_cols
                )
        except ImportError:  # pragma: no cover
            pass
        frame.write_parquet(path)
    else:
        frame.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Incremental fold-level persistence (crash safety)
# ---------------------------------------------------------------------------


def flush_fold_predictions(
    incr_dir: Path,
    config_name: str,
    fold: int,
    checkpoint_preds: dict[int, np.ndarray],
    val_dates: np.ndarray,
    val_entities: np.ndarray | None,
    y_val: np.ndarray,
    date_col: str,
    entity_col: str,
) -> None:
    """Write one fold's checkpoint predictions to parquet for crash safety.

    Shared by deep_learning, tabular_dl, and darts_forecasting runners.
    Handles Object-typed date columns from pandas datetime arrays.
    """
    import numpy as np
    import polars as pl

    dates_series = pl.Series(date_col, val_dates)
    if dates_series.dtype == pl.Object:
        dates_series = dates_series.map_elements(str, return_dtype=pl.String).str.to_datetime(
            strict=False
        )

    frames = []
    for ep, preds in checkpoint_preds.items():
        n = len(preds)
        entities = val_entities if val_entities is not None else np.array(["unknown"] * n)
        df = pl.DataFrame(
            {
                date_col: dates_series,
                entity_col: entities,
                "y_true": y_val.astype(np.float64),
                "y_score": preds.astype(np.float64),
                "fold_id": np.full(n, fold, dtype=np.int32),
                "config": [config_name] * n,
                "epoch": np.full(n, ep, dtype=np.int32),
            }
        )
        frames.append(df)

    if frames:
        _save_parquet(incr_dir / f"{config_name}_fold{fold}.parquet", pl.concat(frames))


def flush_fold_training_log(
    log_dir: Path,
    config_name: str,
    fold: int,
    epoch_rows: list[dict],
) -> None:
    """Write one fold's per-epoch training log to parquet for crash safety."""
    if not epoch_rows:
        return
    import polars as pl

    df = pl.DataFrame(epoch_rows)
    _save_parquet(log_dir / f"{config_name}_fold{fold}.parquet", df)


# ---------------------------------------------------------------------------
# Metric insertion
# ---------------------------------------------------------------------------


def _upsert_wide_metrics(
    db: sqlite3.Connection,
    table: str,
    key_values: dict[str, object],
    metrics: dict[str, float],
    computed_at: str | None = None,
) -> None:
    """Insert or update metric columns in a wide-format metrics table.

    Uses native SQLite UPSERT (``ON CONFLICT(key) DO UPDATE``) so partial
    writes preserve columns that were not provided. Auto-adds any unknown
    metric names as new columns via ``ALTER TABLE``.
    """
    if not metrics:
        return
    if computed_at is None:
        computed_at = _utc_now()

    # Ensure all metric columns exist + record their declared SQLite type so we
    # don't blindly cast strings (e.g. ``task_type='regression'``) to float.
    col_types = {
        row[1]: (row[2] or "").upper()
        for row in db.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for metric_name, metric_value in metrics.items():
        if metric_name not in col_types:
            col_type = "TEXT" if isinstance(metric_value, str) else "REAL"
            db.execute(f'ALTER TABLE {table} ADD COLUMN "{metric_name}" {col_type}')
            col_types[metric_name] = col_type

    def _coerce(name: str, v):
        if v is None:
            return None
        if col_types.get(name, "REAL").startswith("TEXT"):
            return str(v)
        # Strings stored in non-TEXT columns: try float, else pass through.
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    insert_cols = list(key_values.keys()) + ["computed_at"] + list(metrics.keys())
    insert_vals = (
        list(key_values.values())
        + [computed_at]
        + [_coerce(name, v) for name, v in metrics.items()]
    )
    quoted_cols = ", ".join(f'"{c}"' for c in insert_cols)
    placeholders = ", ".join("?" for _ in insert_cols)

    # Conflict target = the primary-key columns the caller supplied
    conflict_cols = ", ".join(f'"{c}"' for c in key_values.keys())
    update_clause = ", ".join(f'"{c}" = excluded."{c}"' for c in ["computed_at", *metrics.keys()])

    db.execute(
        f"INSERT INTO {table} ({quoted_cols}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict_cols}) DO UPDATE SET {update_clause}",
        insert_vals,
    )
