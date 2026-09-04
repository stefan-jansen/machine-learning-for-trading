"""Storage, schema, and filesystem helpers for the experiment registry."""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .specs import (
    IDENTITY_VERSION,
    _validate_spec,
    canonical_json,
    training_hash_from_spec,
)

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
    elapsed_s         REAL,
    runtime_json      TEXT,
    identity_version  INTEGER,
    execution_tier    TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_family_label ON training_runs(family, label);
CREATE INDEX IF NOT EXISTS idx_training_config_name ON training_runs(config_name);

CREATE TABLE IF NOT EXISTS training_identity_migrations (
    target_training_hash TEXT PRIMARY KEY REFERENCES training_runs(training_hash),
    source_training_hash TEXT NOT NULL,
    target_spec_json     TEXT NOT NULL,
    prediction_map_json TEXT NOT NULL,
    proof_json          TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_migration_source
    ON training_identity_migrations(source_training_hash);

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

CREATE TABLE IF NOT EXISTS prediction_coverage (
    prediction_hash     TEXT PRIMARY KEY REFERENCES prediction_sets(prediction_hash),
    expected_key_digest TEXT NOT NULL,
    actual_key_digest   TEXT NOT NULL,
    n_expected          INTEGER NOT NULL,
    n_actual            INTEGER NOT NULL,
    n_duplicates        INTEGER NOT NULL,
    n_missing           INTEGER NOT NULL,
    n_extra             INTEGER NOT NULL,
    n_null              INTEGER NOT NULL,
    n_non_finite        INTEGER NOT NULL,
    n_folds_expected    INTEGER NOT NULL,
    n_folds_actual      INTEGER NOT NULL,
    schema_json          TEXT NOT NULL,
    artifact_digest      TEXT NOT NULL,
    status              TEXT NOT NULL
);

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
    elapsed_s        REAL,
    artifact_digests_json TEXT
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
    refutation_n_successful INTEGER,
    refutation_placebo_json TEXT,
    spec_json        TEXT,
    notebook         TEXT,
    started_at       TEXT,
    elapsed_s        REAL,
    git_commit       TEXT,
    -- The causal identity this run retires, mirroring official_populations. A causal
    -- refit produces a second canonical identity for the same label, and without a
    -- declared chain CausalResult.one sees two and refuses forever - there is no
    -- recency rule to fall back on, and there should not be one in a registry that is
    -- otherwise entirely spec-addressed. Declared by a person through the notebook's
    -- SUPERSEDES_CAUSAL parameter, never inferred from created_at.
    supersedes_hash  TEXT REFERENCES causal_runs(causal_hash),
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
    -- sha256 over the cohort's sorted member backtest hashes. A count cannot say
    -- which variants a stored correction was computed over: swap one retired member
    -- for one live member and k_variants is unchanged, so a reader comparing counts
    -- accepts a correction from a different cohort than the one it asked for.
    member_digest               TEXT,
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

CREATE TABLE IF NOT EXISTS candidate_sets (
    set_hash                 TEXT PRIMARY KEY,
    name                     TEXT NOT NULL,
    member_kind              TEXT NOT NULL,
    comparison_contract_json TEXT NOT NULL,
    created_at               TEXT NOT NULL,
    git_commit               TEXT,
    supersedes_hash          TEXT
);

CREATE TABLE IF NOT EXISTS candidate_set_members (
    set_hash    TEXT NOT NULL REFERENCES candidate_sets(set_hash),
    member_hash TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    PRIMARY KEY (set_hash, ordinal),
    UNIQUE (set_hash, member_hash)
);

-- A candidate set is identified by its members and its comparison contract, so two names for
-- the same comparison resolve to one `candidate_sets` row. The binding therefore cannot live
-- on that row: a union that adds nothing to one of its inputs has the input's identity and its
-- own name, and both names have to resolve. Lineage is per name, because superseding is a
-- statement about which generation of a named comparison is in force.
CREATE TABLE IF NOT EXISTS candidate_set_names (
    name            TEXT NOT NULL,
    set_hash        TEXT NOT NULL REFERENCES candidate_sets(set_hash),
    supersedes_hash TEXT,
    created_at      TEXT NOT NULL,
    git_commit      TEXT,
    PRIMARY KEY (name, set_hash)
);

CREATE INDEX IF NOT EXISTS idx_candidate_set_names_hash ON candidate_set_names(set_hash);


CREATE TABLE IF NOT EXISTS execution_attempts (
    attempt_id          TEXT PRIMARY KEY,
    scientific_identity TEXT NOT NULL,
    status              TEXT NOT NULL,
    diagnostics_json    TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    completed_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_execution_attempt_identity
    ON execution_attempts(scientific_identity, started_at);

CREATE TABLE IF NOT EXISTS candidate_fold_completions (
    training_hash           TEXT NOT NULL REFERENCES training_runs(training_hash),
    candidate_identity      TEXT NOT NULL,
    fold_id                 INTEGER NOT NULL,
    fitted_state_path       TEXT NOT NULL,
    fitted_state_digest     TEXT NOT NULL,
    prediction_shard_path   TEXT NOT NULL,
    prediction_shard_digest TEXT NOT NULL,
    resolved_settings_json  TEXT NOT NULL,
    completed_at            TEXT NOT NULL,
    PRIMARY KEY (training_hash, candidate_identity, fold_id)
);

CREATE TABLE IF NOT EXISTS official_populations (
    population_hash  TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    member_kind      TEXT NOT NULL,
    snapshot_json    TEXT NOT NULL,
    supersedes_hash  TEXT REFERENCES official_populations(population_hash),
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_population_name
    ON official_populations(name, created_at);

CREATE TABLE IF NOT EXISTS official_population_members (
    population_hash TEXT NOT NULL REFERENCES official_populations(population_hash),
    member_hash     TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    PRIMARY KEY (population_hash, ordinal),
    UNIQUE (population_hash, member_hash)
);

CREATE TABLE IF NOT EXISTS overlay_references (
    result_hash TEXT NOT NULL,
    result_kind TEXT NOT NULL,
    source_root TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (result_hash, result_kind)
);

CREATE TABLE IF NOT EXISTS decision_artifacts (
    decision_hash       TEXT PRIMARY KEY,
    decision_kind       TEXT NOT NULL,
    spec_json           TEXT NOT NULL,
    artifact_digest     TEXT NOT NULL,
    canonical           INTEGER NOT NULL,
    created_at          TEXT NOT NULL
);

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
    # The explicit tag is read before the risk block, because it states what the caller is
    # doing while the risk block only says what the strategy contains. Once cost sensitivity
    # runs on the winner of the risk stage - which is the order the backtest sequence now
    # takes, risk before costs - every cost row carries an overlay, and inferring from the
    # overlay first made `cost_sensitivity` unreachable for exactly the runs that are cost
    # sensitivity. Measured on sp500_equity_option_analytics: a 17-point cost surface over a
    # `trailing_5pct` carrier registered all 17 rows as `risk_overlay`.
    #
    # This did not bite while costs and risk were parallel branches off allocation, because a
    # cost run then carried no overlay and fell through to the tag.
    chapter = spec.get("chapter", "")
    if chapter == "ch18":
        return "cost_sensitivity"
    risk = strategy.get("risk", {})
    if risk and risk.get("name") != "baseline":
        return "risk_overlay"
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
    _backfill_candidate_set_names(db)
    _declare_uncertainty_columns(db)
    return db


def _backfill_candidate_set_names(db: sqlite3.Connection) -> None:
    """Give every stored candidate set the name binding its identity row records.

    `candidate_sets` holds one row per set of members, so its `name` column can only record the
    first name a set was written under; a second name for the same members had nowhere to go and
    was dropped. `candidate_set_names` is where a binding lives now, and this carries the
    existing ones across. It runs after the schema script rather than in `_migrate_registry`,
    which runs before the table exists.

    One binding per existing row, carrying that row's lineage, so a migrated registry resolves
    every name it resolved before.

    Probed with a read before writing, and this matters more than it looks. `_open_registry` is
    on every path that touches a registry, so an unconditional `INSERT ... SELECT` took the
    write lock on every open - and with `busy_timeout` at 60s, one contended open blocks for a
    minute rather than proceeding. The probe is a covering read that answers instantly and
    leaves the lock alone once the backfill has run, which is every open after the first.
    """
    pending = db.execute(
        "SELECT EXISTS (SELECT 1 FROM candidate_sets s WHERE NOT EXISTS ("
        "  SELECT 1 FROM candidate_set_names n"
        "  WHERE n.name = s.name AND n.set_hash = s.set_hash))"
    ).fetchone()[0]
    if not pending:
        return
    db.execute(
        "INSERT OR IGNORE INTO candidate_set_names "
        "(name, set_hash, supersedes_hash, created_at, git_commit) "
        "SELECT name, set_hash, supersedes_hash, created_at, git_commit FROM candidate_sets"
    )
    db.commit()


# Metric columns the uncertainty layer produces on every run, which the CREATE TABLE statements
# above do not list. ``_upsert_wide_metrics`` adds an unknown metric column on first write, so
# without this a registry's shape depended on its write history: 22 columns in ``backtest_metrics``
# where no backtest had ever been registered and 37 where one had. Every notebook that reads a
# confidence band then failed with ``no such column: m.sharpe_ci95_lo`` against exactly the
# registries a reset had just created, which is where a rebuild always starts.
#
# Each set is what one producer returns, so a key added there is added here:
#   backtest_metrics, backtest_fold_metrics  <- compute_backtest_uncertainty (utils/uncertainty.py)
#   prediction_metrics                       <- the ic_/auc_ blocks in registry/metrics.py
_BACKTEST_UNCERTAINTY_COLUMNS = (
    "sharpe_se_lo",
    "sharpe_ci95_lo",
    "sharpe_ci95_hi",
    "sortino_ci95_lo",
    "sortino_ci95_hi",
    "ann_return_hac_se",
    "ann_return_ci95_lo",
    "ann_return_ci95_hi",
    "max_dd_ci95_lo",
    "max_dd_ci95_hi",
    "calmar_ci95_lo",
    "calmar_ci95_hi",
    "psr_pvalue",
    "bootstrap_block_length",
    "bootstrap_n",
)

_DECLARED_METRIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "backtest_metrics": _BACKTEST_UNCERTAINTY_COLUMNS,
    # n_periods rides along: the fold table declares n_days, and the metric pass writes both.
    "backtest_fold_metrics": _BACKTEST_UNCERTAINTY_COLUMNS + ("n_periods",),
    "prediction_metrics": tuple(
        f"{metric}_{suffix}"
        for metric in ("ic", "auc")
        for suffix in (
            "mean_daily",
            "std_daily",
            "n_days",
            "se_naive",
            "naive_lo",
            "naive_hi",
            "se_hac",
            "ci_lo",
            "ci_hi",
            "t_hac",
            "p_hac",
            "hac_lag",
            "boot_lo",
            "boot_hi",
            "boot_block",
        )
    )
    # The two producers disagree on one name apiece: an IC is signed, so what is counted is the
    # share of days above zero, while an AUC's null is 0.5.
    + ("ic_pct_positive", "auc_pct_above_null"),
}


def _declare_uncertainty_columns(db: sqlite3.Connection) -> None:
    """Give every metric table its full column set, whether or not anything has been written."""
    for table, columns in _DECLARED_METRIC_COLUMNS.items():
        existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
        for column in columns:
            if column not in existing:
                db.execute(f'ALTER TABLE {table} ADD COLUMN "{column}" REAL')
    db.commit()


def _table_has_column(db: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a table has a specific column."""
    return column in {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}


def current_causal_identities(
    db, *, label: str, tier: str = "canonical", exclude: str | None = None
) -> list[str]:
    """The causal identities a reader would currently resolve for *label*.

    A row is current when its spec carries the current ``identity_version``, its
    ``execution_tier`` is the one asked for, and no other row declares that it
    supersedes it.

    This lives here, and not beside either caller, because it has to be one derivation.
    ``CausalResult.one`` decides what a reader resolves and ``register_causal_run``
    decides what may be written; if those two sets differ, a registration is refused
    for an ambiguity the reader never sees, or permitted into one it cannot resolve.
    The first draft duplicated the logic and the copies disagreed within the hour -
    one counted every SUPPORTED_IDENTITY_VERSION, the other only the current one, so a
    legacy row made the first v3 registration for its label impossible to satisfy.
    The reader's rule is the authority, because it is the one a person hits.
    """
    columns = {row[1] for row in db.execute("PRAGMA table_info(causal_runs)").fetchall()}
    supersedes_column = (
        "supersedes_hash" if "supersedes_hash" in columns else "NULL AS supersedes_hash"
    )
    rows = db.execute(
        f"SELECT causal_hash, spec_json, {supersedes_column} FROM causal_runs "
        "WHERE label = ? ORDER BY causal_hash",
        (label,),
    ).fetchall()
    retired = {row[2] for row in rows if row[2]}
    current = []
    for causal_hash, spec_json, _ in rows:
        spec = json.loads(spec_json or "{}")
        if spec.get("identity_version") != IDENTITY_VERSION:
            continue
        if str(spec.get("execution_tier", tier)) != tier:
            continue
        if causal_hash in retired or causal_hash == exclude:
            continue
        current.append(causal_hash)
    return current


def causal_identities_retired(db, *, label: str) -> set[str]:
    """Every causal hash some other row declares it supersedes."""
    columns = {row[1] for row in db.execute("PRAGMA table_info(causal_runs)").fetchall()}
    if "supersedes_hash" not in columns:
        return set()
    return {
        row[0]
        for row in db.execute(
            "SELECT supersedes_hash FROM causal_runs WHERE label = ? AND supersedes_hash IS NOT NULL",
            (label,),
        ).fetchall()
    }


def _migrate_registry(db: sqlite3.Connection) -> None:
    """Apply incremental schema migrations to an existing registry."""
    # Check if backtest_runs table exists at all
    tables = {
        row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if not tables:
        return  # Fresh DB — schema script will create everything

    # Migration 1: add stage column to backtest_runs
    cols: set[str] = set()
    if "backtest_runs" in tables:
        cols = {row[1] for row in db.execute("PRAGMA table_info(backtest_runs)").fetchall()}
        if "stage" not in cols:
            db.execute("ALTER TABLE backtest_runs ADD COLUMN stage TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS idx_backtest_stage ON backtest_runs(stage)")
            cols.add("stage")

    # Migration 2: add runtime columns to training_runs
    if "training_runs" in tables:
        tr_cols = {row[1] for row in db.execute("PRAGMA table_info(training_runs)").fetchall()}
        training_columns = {
            "config_name": "TEXT",
            "spec_json": "TEXT",
            "git_commit": "TEXT",
            "entry_point": "TEXT",
            "started_at": "TEXT",
            "elapsed_s": "REAL",
            "runtime_json": "TEXT",
            "identity_version": "INTEGER",
            "execution_tier": "TEXT",
        }
        for column, sql_type in training_columns.items():
            if column not in tr_cols:
                db.execute(f"ALTER TABLE training_runs ADD COLUMN {column} {sql_type}")

    if "prediction_sets" in tables:
        prediction_cols = {
            row[1] for row in db.execute("PRAGMA table_info(prediction_sets)").fetchall()
        }
        prediction_columns = {"checkpoint_value": "INTEGER", "checkpoint_kind": "TEXT"}
        for column, sql_type in prediction_columns.items():
            if column not in prediction_cols:
                db.execute(f"ALTER TABLE prediction_sets ADD COLUMN {column} {sql_type}")

    if "cohort_metrics" in tables:
        cohort_cols = {row[1] for row in db.execute("PRAGMA table_info(cohort_metrics)").fetchall()}
        if "member_digest" not in cohort_cols:
            db.execute("ALTER TABLE cohort_metrics ADD COLUMN member_digest TEXT")

    if "prediction_coverage" in tables:
        coverage_cols = {
            row[1] for row in db.execute("PRAGMA table_info(prediction_coverage)").fetchall()
        }
        if "schema_json" not in coverage_cols:
            db.execute("ALTER TABLE prediction_coverage ADD COLUMN schema_json TEXT")
        if "artifact_digest" not in coverage_cols:
            db.execute("ALTER TABLE prediction_coverage ADD COLUMN artifact_digest TEXT")

    # Migration 2b: add runtime columns to backtest_runs
    if "backtest_runs" in tables:
        backtest_columns = {
            "spec_json": "TEXT",
            "stage": "TEXT",
            "git_commit": "TEXT",
            "started_at": "TEXT",
            "elapsed_s": "REAL",
            "artifact_digests_json": "TEXT",
        }
        for column, sql_type in backtest_columns.items():
            if column not in cols:
                db.execute(f"ALTER TABLE backtest_runs ADD COLUMN {column} {sql_type}")

    # The number of successful placebo draws decides whether the refutation could have
    # rejected at all: the plus-one correction floors the p-value at 1 / (n + 1), so at
    # 19 or fewer no data could produce a pass. Without it a reader holding only
    # refutation_p cannot tell an underpowered run from a failed one.
    if "causal_runs" in tables and not _table_has_column(
        db, "causal_runs", "refutation_n_successful"
    ):
        db.execute("ALTER TABLE causal_runs ADD COLUMN refutation_n_successful INTEGER")

    # Additive, and it costs no recompute: the column is outside the causal computation
    # specification, so it moves no causal hash and invalidates no registered row.
    if "causal_runs" in tables and not _table_has_column(db, "causal_runs", "supersedes_hash"):
        db.execute("ALTER TABLE causal_runs ADD COLUMN supersedes_hash TEXT")

    # The candidate-set equivalent of the line above. A candidate set is derived from a
    # registry that moves, so re-running the stage that freezes it produces a second set
    # under the same name; without a declared predecessor the name resolves to two live
    # identities and every reader of it raises.
    if "candidate_sets" in tables and not _table_has_column(
        db, "candidate_sets", "supersedes_hash"
    ):
        db.execute("ALTER TABLE candidate_sets ADD COLUMN supersedes_hash TEXT")

    # The placebo draws behind refutation_p. Only the scalars were stored, so the
    # permutation-distribution figure every causal notebook draws had no source in the
    # registry and rendered empty behind its guard while the prose described it.
    if "causal_runs" in tables and not _table_has_column(
        db, "causal_runs", "refutation_placebo_json"
    ):
        db.execute("ALTER TABLE causal_runs ADD COLUMN refutation_placebo_json TEXT")

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


_PREDICTION_TIME_COLUMNS = ("timestamp", "date", "datetime", "ts")


def _timestamps_as_utc(predictions):
    """Give a naive decision-time column an explicit UTC zone before it is written.

    `gbm`, `linear` and `tabular_dl` write `Datetime(_, 'UTC')`; `deep_learning` reaches
    this through `flush_fold_predictions`, whose dates come from a numpy `datetime64`
    array and are therefore naive. Measured on crypto_perps_funding: 578 artifacts UTC-
    aware and 100 naive, same label, same folds, same 19 symbols, same 2,189 decision
    times, identical instants. A tz-aware value never equals a naive one, so an exact join
    on (timestamp, symbol) between the two families returned nothing, and any code
    assuming one dtype across a case study's artifacts dropped rows instead of failing.

    Naive is read as UTC here, which is what it already meant: every producer derives
    these timestamps from the label artifact's own axis, and the naive values are the same
    instants the aware ones carry. This relabels; it never converts a wall time.

    `value_digest` ignores the zone (it is time-unit sensitive and zone-insensitive), so
    an artifact rewritten through here keeps its digest and no immutable-artifact check
    moves. The time unit is deliberately left alone for the same reason.
    """
    if predictions is None:
        return predictions
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return predictions

    if isinstance(predictions, pl.DataFrame):
        naive = [
            column
            for column in _PREDICTION_TIME_COLUMNS
            if column in predictions.columns
            and isinstance(predictions.schema[column], pl.Datetime)
            and predictions.schema[column].time_zone is None
        ]
        if not naive:
            return predictions
        return predictions.with_columns(
            pl.col(column).dt.replace_time_zone("UTC") for column in naive
        )

    # pandas is handled in place rather than converted. Both the legacy registration branch
    # and the pandas side of the versioned one hand the caller's own frame to the writer,
    # and `pl.from_pandas` on an arbitrary frame is a wider change than this needs. A naive
    # pandas column localizes to UTC the same way; an already-aware one is left alone.
    import pandas as pd

    if not isinstance(predictions, pd.DataFrame):
        return predictions
    naive = [
        column
        for column in _PREDICTION_TIME_COLUMNS
        if column in predictions.columns
        and pd.api.types.is_datetime64_any_dtype(predictions[column])
        and getattr(predictions[column].dtype, "tz", None) is None
    ]
    if not naive:
        return predictions
    localized = predictions.copy()
    for column in naive:
        localized[column] = localized[column].dt.tz_localize("UTC")
    return localized


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
    *,
    eval_actual: np.ndarray | None = None,
    eval_col: str = "eval_actual",
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
        if eval_actual is not None:
            df = df.with_columns(pl.Series(eval_col, eval_actual.astype(np.float64)))
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
    metrics: Mapping[str, object],
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
