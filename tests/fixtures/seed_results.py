"""Generate minimal results fixtures for test output directories.

Downstream notebooks need:
1. results/*.json — legacy format for some downstream comparisons
2. run_log/registry.db — SQLite registry queried by case_study_insights,
   model_analysis, and backtest notebooks via utils.case_study_analytics
3. results/causal_dml.json — Ch15 causal insights
4. results/ch08_features.json, ch09_temporal.json — Ch08/09 summaries

All fixtures use minimal but schema-correct data. Only written if not
already present (real upstream runs take priority).
"""

import hashlib
import json
import sqlite3
import warnings
from pathlib import Path

import yaml

from case_studies.utils.registry.specs import IDENTITY_VERSION
from case_studies.utils.registry.store import _open_registry

REPO_ROOT = Path(__file__).parent.parent.parent
CS_ROOT = REPO_ROOT / "case_studies"

# All model families in the pipeline
FAMILIES = ["linear", "gbm", "tabular_dl", "deep_learning", "latent_factors", "causal_dml"]

# Config names per family (representative). Linear has two so Ch26 can find lasso.
FAMILY_CONFIGS = {
    "linear": ["ridge_a1.0", "lasso_a0.01"],
    "gbm": ["lgb_default_mse"],
    "tabular_dl": ["tabm_s"],
    "deep_learning": ["lstm_64"],
    "latent_factors": ["pca_5"],
    "causal_dml": ["dml_linear"],
}

# Backtest stages (Ch16-19)
BACKTEST_STAGES = ["signal", "allocation", "cost_sensitivity", "risk_overlay"]

# Timestamp for all fixture entries
FIXTURE_TS = "2026-01-01T00:00:00"


def _linear_fixture(label: str) -> dict:
    """Minimal linear results JSON — just enough for downstream `best_model` lookups."""
    return {
        "case_study_id": "fixture",
        "chapter": "ch11",
        "stage": f"linear_{label}",
        "timestamp": "2026-01-01T00:00:00",
        "git_commit": "fixture",
        "notebook": "fixture",
        "summary": {
            "n_folds": 2,
            "n_features": 10,
            "n_rows": 100,
            "primary_label": label,
            "label_column": "y",
            "best_model": "ridge",
            "hpo_method": "grid",
            "models": {
                "ridge": {"ic_mean": 0.01, "ic_std": 0.005, "best_alpha": 1.0},
                "ols": {"ic_mean": 0.008, "ic_std": 0.006},
                "lasso": {"ic_mean": 0.009, "ic_std": 0.005, "best_alpha": 0.01},
            },
        },
    }


def _gbm_fixture(label: str) -> dict:
    """Minimal GBM results JSON — just enough for downstream `val_ic_mean` lookups."""
    return {
        "case_study_id": "fixture",
        "chapter": "ch12",
        "stage": f"gbm_{label}",
        "timestamp": "2026-01-01T00:00:00",
        "git_commit": "fixture",
        "notebook": "fixture",
        "summary": {
            "n_folds": 2,
            "n_features": 10,
            "n_rows": 100,
            "primary_label": label,
            "label_column": "y",
            "device": "cpu",
            "num_boost_round": 5,
            "n_configs": 1,
            "best_config": "default_mse",
            "best_iteration": 5,
            "val_ic_mean": 0.015,
            "grid": {"default_mse": {"best_ic": 0.015, "best_iteration": 5}},
        },
    }


def _tabular_dl_fixture(label: str) -> dict:
    """Minimal TabDL results JSON."""
    return {
        "case_study_id": "fixture",
        "chapter": "ch12",
        "stage": f"tabular_dl_{label}",
        "timestamp": "2026-01-01T00:00:00",
        "git_commit": "fixture",
        "notebook": "fixture",
        "summary": {
            "n_folds": 2,
            "n_features": 10,
            "n_rows": 100,
            "primary_label": label,
            "label_column": "y",
            "val_ic_mean": 0.012,
            "best_config": "tabm_s",
        },
    }


def _make_hash(content: str) -> str:
    """Deterministic 12-char hash for fixture data."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _migrate_long_to_wide(db_path: Path) -> None:
    """Migrate registry.db metric tables from long format (metric/value pairs)
    to wide format (one column per metric).

    Old intermediates used EAV-style tables:
      fold_metrics(prediction_hash, fold_id, metric, value, computed_at)
      backtest_metrics(backtest_hash, metric, value, detail_json, computed_at)

    Production code expects wide tables:
      fold_metrics(prediction_hash, fold_id, computed_at, ic, ic_std, rmse, ...)
      backtest_metrics(backtest_hash, computed_at, sharpe, sortino, ...)
    """
    db = sqlite3.connect(str(db_path))

    # --- Migrate prediction_metrics ---
    pm_cols = {r[1] for r in db.execute("PRAGMA table_info(prediction_metrics)").fetchall()}
    if "metric" in pm_cols and "ic_mean" not in pm_cols:
        rows = db.execute(
            "SELECT prediction_hash, metric, value, computed_at FROM prediction_metrics"
        ).fetchall()
        db.execute("DROP TABLE prediction_metrics")
        db.execute("""
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY REFERENCES prediction_sets(prediction_hash),
                computed_at TEXT NOT NULL,
                ic_mean REAL, ic_std REAL, ic_t REAL, n_folds REAL, n_obs REAL,
                n_periods REAL, pct_positive REAL, task_type REAL,
                accuracy REAL, balanced_accuracy REAL, auc_roc REAL, auc_pr REAL,
                log_loss REAL, brier_score REAL
            )
        """)
        wide = {}
        for pred_hash, metric, value, computed_at in rows:
            if pred_hash not in wide:
                wide[pred_hash] = {"computed_at": computed_at}
            wide[pred_hash][metric] = value

        valid_cols = {
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
        }
        for pred_hash, vals in wide.items():
            cols_present = [c for c in valid_cols if c in vals]
            placeholders = ", ".join(["?"] * (2 + len(cols_present)))
            col_names = ", ".join(["prediction_hash", "computed_at"] + cols_present)
            values = [pred_hash, vals["computed_at"]] + [vals[c] for c in cols_present]
            db.execute(
                f"INSERT OR IGNORE INTO prediction_metrics ({col_names}) VALUES ({placeholders})",
                values,
            )

    # --- Migrate fold_metrics ---
    fm_cols = {r[1] for r in db.execute("PRAGMA table_info(fold_metrics)").fetchall()}
    if "metric" in fm_cols and "ic" not in fm_cols:
        rows = db.execute(
            "SELECT prediction_hash, fold_id, metric, value, computed_at FROM fold_metrics"
        ).fetchall()
        db.execute("DROP TABLE fold_metrics")
        db.execute("""
            CREATE TABLE fold_metrics (
                prediction_hash TEXT NOT NULL REFERENCES prediction_sets(prediction_hash),
                fold_id INTEGER NOT NULL, computed_at TEXT NOT NULL,
                ic REAL, ic_std REAL, n_periods REAL, n_obs REAL, n_entities REAL,
                rmse REAL, mae REAL,
                accuracy REAL, balanced_accuracy REAL, auc_roc REAL, auc_pr REAL,
                log_loss REAL, brier_score REAL,
                PRIMARY KEY (prediction_hash, fold_id)
            )
        """)
        # Pivot long → wide
        wide = {}
        for pred_hash, fold_id, metric, value, computed_at in rows:
            key = (pred_hash, fold_id)
            if key not in wide:
                wide[key] = {"computed_at": computed_at}
            wide[key][metric] = value

        valid_cols = {
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
        }
        for (pred_hash, fold_id), vals in wide.items():
            cols_present = [c for c in valid_cols if c in vals]
            placeholders = ", ".join(["?"] * (3 + len(cols_present)))
            col_names = ", ".join(["prediction_hash", "fold_id", "computed_at"] + cols_present)
            values = [pred_hash, fold_id, vals["computed_at"]] + [vals[c] for c in cols_present]
            db.execute(
                f"INSERT OR IGNORE INTO fold_metrics ({col_names}) VALUES ({placeholders})", values
            )

    # --- Migrate backtest_metrics ---
    bm_cols = {r[1] for r in db.execute("PRAGMA table_info(backtest_metrics)").fetchall()}
    if "metric" in bm_cols and "sharpe" not in bm_cols:
        rows = db.execute(
            "SELECT backtest_hash, metric, value, computed_at FROM backtest_metrics"
        ).fetchall()
        db.execute("DROP TABLE backtest_metrics")
        db.execute("""
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY REFERENCES backtest_runs(backtest_hash),
                computed_at TEXT NOT NULL,
                sharpe REAL, sortino REAL, total_return REAL, max_drawdown REAL,
                cagr REAL, volatility REAL, calmar REAL, omega REAL, stability REAL,
                tail_ratio REAL, win_rate REAL, kurtosis REAL, skewness REAL,
                var_95 REAL, cvar_95 REAL, n_periods REAL,
                num_trades REAL, total_commission REAL, total_slippage REAL, avg_turnover REAL
            )
        """)
        # Pivot long → wide
        wide = {}
        for b_hash, metric, value, computed_at in rows:
            if b_hash not in wide:
                wide[b_hash] = {"computed_at": computed_at}
            wide[b_hash][metric] = value

        valid_cols = {
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
        }
        for b_hash, vals in wide.items():
            cols_present = [c for c in valid_cols if c in vals]
            placeholders = ", ".join(["?"] * (2 + len(cols_present)))
            col_names = ", ".join(["backtest_hash", "computed_at"] + cols_present)
            values = [b_hash, vals["computed_at"]] + [vals[c] for c in cols_present]
            db.execute(
                f"INSERT OR IGNORE INTO backtest_metrics ({col_names}) VALUES ({placeholders})",
                values,
            )

    # --- Migrate backtest_fold_metrics (if long format) ---
    bfm_cols = {r[1] for r in db.execute("PRAGMA table_info(backtest_fold_metrics)").fetchall()}
    if "metric" in bfm_cols and "sharpe" not in bfm_cols:
        rows = db.execute(
            "SELECT backtest_hash, fold_id, metric, value, computed_at FROM backtest_fold_metrics"
        ).fetchall()
        db.execute("DROP TABLE backtest_fold_metrics")
        db.execute("""
            CREATE TABLE backtest_fold_metrics (
                backtest_hash TEXT NOT NULL REFERENCES backtest_runs(backtest_hash),
                fold_id INTEGER NOT NULL, metric TEXT NOT NULL,
                value REAL, computed_at TEXT NOT NULL,
                PRIMARY KEY (backtest_hash, fold_id, metric)
            )
        """)
        for row in rows:
            db.execute("INSERT OR IGNORE INTO backtest_fold_metrics VALUES (?,?,?,?,?)", row)

    db.commit()
    db.close()


def _add_cohort_metrics_table(db_path: Path) -> None:
    """Add cohort_metrics + backtest_paired_metrics to an existing test-data
    registry.

    Mirrors the schemas in case_studies/utils/registry/store.py. Both tables
    start empty; consumers use LEFT JOIN / fetchall→pl.DataFrame, so empty
    is fine for CI. Real backfill comes from scripts/backfill_cohort_metrics.py
    and the paired-metrics populator in 01_aggregate_synthesis.
    """
    db = sqlite3.connect(str(db_path))
    db.executescript("""
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
    """)
    db.commit()
    db.close()


def _upgrade_seeded_registry(cs_dir: Path) -> None:
    """Bring a registry copied out of test-data up to the canonical schema and identity.

    `_open_registry` is the function production uses to open any registry: it migrates
    an old one, then runs REGISTRY_SCHEMA_SQL, then declares the uncertainty columns. So
    the tables and columns a shipped registry lacks are added by the same code that
    would add them in a real run, rather than by a copy of it that can drift.

    What the schema cannot supply is the values. A registry written before
    `identity_version` existed carries NULL there, which the catalog reads as "legacy"
    and therefore never complete, and it has no coverage row at all. Both are backfilled
    here for rows that lack them, and only for those - a row that already records its
    identity is left alone.
    """
    db = _open_registry(cs_dir)
    try:
        db.execute(
            "UPDATE training_runs SET identity_version = ? WHERE identity_version IS NULL",
            (IDENTITY_VERSION,),
        )
        db.execute(
            "UPDATE training_runs SET execution_tier = 'canonical' WHERE execution_tier IS NULL"
        )
        uncovered = db.execute(
            """SELECT p.prediction_hash, COUNT(f.fold_id)
               FROM prediction_sets p
               LEFT JOIN fold_metrics f ON f.prediction_hash = p.prediction_hash
               WHERE p.prediction_hash NOT IN (SELECT prediction_hash FROM prediction_coverage)
               GROUP BY p.prediction_hash"""
        ).fetchall()
        for p_hash, n_folds in uncovered:
            # n_folds_expected must equal the fold_metrics rows actually present: the
            # catalog compares the two, so a constant here would mark a row incomplete
            # for the opposite reason.
            db.execute(
                """INSERT OR IGNORE INTO prediction_coverage
                   (prediction_hash, expected_key_digest, actual_key_digest,
                    n_expected, n_actual, n_duplicates, n_missing, n_extra,
                    n_null, n_non_finite, n_folds_expected, n_folds_actual,
                    schema_json, artifact_digest, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    p_hash,
                    _make_hash(f"keys/{p_hash}"),
                    _make_hash(f"keys/{p_hash}"),
                    100,
                    100,
                    0,
                    0,
                    0,
                    0,
                    0,
                    n_folds,
                    n_folds,
                    json.dumps({"symbol": "String", "timestamp": "Date"}),
                    # Empty, not NULL: prediction_coverage.artifact_digest is NOT NULL, and
                    # under INSERT OR IGNORE a NULL silently drops the whole row rather
                    # than raising - which is how the first version of this wrote zero
                    # coverage rows and still passed a test asserting 23 of them.
                    # `results.py:419` guards with `if recorded_digest:`, so "" reads as
                    # "not recorded, backfill it" until the pass below fills it in.
                    "",
                    "complete",
                ),
            )
        db.commit()
    finally:
        db.close()


def _seed_registry_db(cs_dir: Path, cs_id: str, primary_label: str) -> None:
    """Create a minimal registry.db with entries for all families and stages.

    The schema comes from `_open_registry`, the function production uses, so it cannot
    drift from the canonical one.
    Creates entries that utils.case_study_analytics and utils.model_analysis
    can query without crashing.
    """
    db_path = cs_dir / "run_log" / "registry.db"
    if db_path.exists():
        try:
            _db = sqlite3.connect(str(db_path))
            cols = {r[1] for r in _db.execute("PRAGMA table_info(training_runs)").fetchall()}
            tables = {
                r[0]
                for r in _db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            bm_cols = {r[1] for r in _db.execute("PRAGMA table_info(backtest_metrics)").fetchall()}
            fm_cols = {r[1] for r in _db.execute("PRAGMA table_info(fold_metrics)").fetchall()}
            pm_cols = {
                r[1] for r in _db.execute("PRAGMA table_info(prediction_metrics)").fetchall()
            }
            _db.close()

            # Core schema check: training_runs must have training_hash
            has_core = "training_hash" in cols
            # Wide-format metric check
            bm_wide = "sharpe" in bm_cols
            fm_wide = "ic" in fm_cols
            pm_wide = "ic_mean" in pm_cols
            has_all_tables = (
                "fold_metrics" in tables
                and "backtest_runs" in tables
                and "cohort_metrics" in tables
            )

            if has_core and has_all_tables and bm_wide and fm_wide and pm_wide:
                # NOT a return. "Fully current" here is five hand-picked columns, and
                # every one of the nine registries shipped in test-data satisfies all
                # five while missing thirteen tables and two columns. Returning on that
                # check is why #893's first fix changed nothing: the canonical DDL below
                # is only reached when no registry was copied in, which is not the path
                # conftest takes. Bring the copied registry up to schema instead.
                _upgrade_seeded_registry(cs_dir)
                return

            # Schema present but missing cohort_metrics — add it without
            # rebuilding the entire registry (preserves seeded rows).
            if has_core and "cohort_metrics" not in tables:
                _add_cohort_metrics_table(db_path)
                tables.add("cohort_metrics")
                has_all_tables = (
                    "fold_metrics" in tables
                    and "backtest_runs" in tables
                    and "cohort_metrics" in tables
                )
                if has_core and has_all_tables and bm_wide and fm_wide and pm_wide:
                    return

            if not has_core:
                # Legacy schema (run_id instead of training_hash) — must rebuild
                db_path.unlink()
            else:
                # Core schema OK but metrics need migration
                if not bm_wide or not fm_wide or not pm_wide:
                    _migrate_long_to_wide(db_path)
                # Fall through to seed missing entries
        except Exception:
            db_path.unlink(missing_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = _open_registry(cs_dir)
    # The canonical schema itself, not a restatement of it. This block used to repeat
    # nine CREATE TABLE statements under a docstring promising they matched
    # REGISTRY_SCHEMA_SQL "exactly", with nothing checking that promise. They had
    # drifted by thirteen tables - every research-boundary table, prediction_coverage
    # among them - and by training_runs.identity_version, so a notebook on the research
    # API resolved an empty catalog from a fixture that looked fully populated (#893).

    # IC values per family (realistic ordering: gbm > linear > dl > others)
    ic_values = {
        "linear": 0.018,
        "gbm": 0.025,
        "tabular_dl": 0.022,
        "deep_learning": 0.020,
        "latent_factors": 0.015,
        "causal_dml": 0.012,
    }

    # Insert training runs + prediction sets + metrics per family/config.
    # Also insert for ALL labels (not just primary) so Ch26 notebooks
    # can find specific config+label combos like lasso/fwd_ret_1d.
    best_pred_hash = None
    best_ic = -1.0
    all_labels = [primary_label]

    # Get variant labels from setup.yaml
    setup_path = CS_ROOT / cs_id / "config" / "setup.yaml"
    if setup_path.exists():
        setup = yaml.safe_load(setup_path.read_text())
        variants = setup.get("labels", {}).get("variants", [])
        if isinstance(variants, list):
            for v in variants:
                name = v if isinstance(v, str) else v.get("name", "")
                if name and name not in all_labels:
                    all_labels.append(name)

    for family in FAMILIES:
        config_names = FAMILY_CONFIGS[family]
        for config_name in config_names:
            for label in all_labels:
                t_hash = _make_hash(f"{cs_id}/{family}/{config_name}/{label}")
                p_hash = _make_hash(f"pred/{t_hash}/validation")
                ic = ic_values.get(family, 0.01)

                spec = {"family": family, "config_name": config_name, "label": label}
                db.execute(
                    """INSERT OR IGNORE INTO training_runs
                       (training_hash, family, label, config_name, spec_json, created_at,
                        git_commit, entry_point, identity_version, execution_tier)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        t_hash,
                        family,
                        label,
                        config_name,
                        json.dumps(spec),
                        FIXTURE_TS,
                        "fixture",
                        "fixture",
                        # Without this the catalog resolves every seeded row as "legacy"
                        # and therefore incomplete, so a notebook that filters on the
                        # current identity sees nothing. Reading IDENTITY_VERSION rather
                        # than writing 3 means the fixture follows the next bump instead
                        # of silently going legacy again the day it lands.
                        IDENTITY_VERSION,
                        "canonical",
                    ),
                )
                db.execute(
                    """INSERT OR IGNORE INTO prediction_sets
                       (prediction_hash, training_hash, checkpoint_value, checkpoint_kind, split, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (p_hash, t_hash, 100, "final", "validation", FIXTURE_TS),
                )
                db.execute(
                    """INSERT OR IGNORE INTO prediction_metrics
                       (prediction_hash, computed_at, ic_mean, ic_std, ic_t, n_folds)
                       VALUES (?,?,?,?,?,?)""",
                    (p_hash, FIXTURE_TS, ic, ic * 0.3, ic / (ic * 0.3), 2),
                )
                # Fold metrics (2 folds) — wide format matching production schema
                for fold_id in range(2):
                    fold_ic = ic + (0.002 if fold_id == 0 else -0.002)
                    db.execute(
                        """INSERT OR IGNORE INTO fold_metrics
                           (prediction_hash, fold_id, computed_at, ic, ic_std, n_entities, rmse, mae)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (
                            p_hash,
                            fold_id,
                            FIXTURE_TS,
                            fold_ic,
                            fold_ic * 0.3,
                            5,
                            0.05,
                            0.03,
                        ),
                    )

                # `complete` is a conjunction (research/catalog.py:308-313): current
                # identity, a coverage row saying complete, a prediction_metrics row,
                # fold_metrics matching n_folds_expected, and the artifact on disk. The
                # fixture supplied three of the five, so every row read as partial. The
                # two fold_metrics rows above are what n_folds_expected must equal; the
                # digest is filled by _backfill_all_prediction_parquets, which writes
                # the artifact.
                db.execute(
                    """INSERT OR IGNORE INTO prediction_coverage
                       (prediction_hash, expected_key_digest, actual_key_digest,
                        n_expected, n_actual, n_duplicates, n_missing, n_extra,
                        n_null, n_non_finite, n_folds_expected, n_folds_actual,
                        schema_json, artifact_digest, status)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        p_hash,
                        _make_hash(f"keys/{p_hash}"),
                        _make_hash(f"keys/{p_hash}"),
                        100,
                        100,
                        0,
                        0,
                        0,
                        0,
                        0,
                        2,
                        2,
                        json.dumps({"symbol": "String", "timestamp": "Date"}),
                        # Empty, not NULL: prediction_coverage.artifact_digest is NOT NULL, and
                        # under INSERT OR IGNORE a NULL silently drops the whole row rather
                        # than raising - which is how the first version of this wrote zero
                        # coverage rows and still passed a test asserting 23 of them.
                        # `results.py:419` guards with `if recorded_digest:`, so "" reads as
                        # "not recorded, backfill it" until the pass below fills it in.
                        "",
                        "complete",
                    ),
                )

                if label == primary_label and ic > best_ic:
                    best_ic = ic
                    best_pred_hash = p_hash

    # Insert backtest runs for each stage (using best model's prediction)
    if best_pred_hash:
        sharpe_by_stage = {
            "signal": 0.8,
            "allocation": 0.9,
            "cost_sensitivity": 0.7,
            "risk_overlay": 0.85,
        }
        for stage in BACKTEST_STAGES:
            b_hash = _make_hash(f"bt/{cs_id}/{stage}/{best_pred_hash}")
            spec = {
                "stage": stage,
                "prediction_hash": best_pred_hash,
                "chapter": f"ch{16 + BACKTEST_STAGES.index(stage)}",
                "signal": {"method": "equal_weight_top_k", "top_k": 5},
                "allocation": {"method": "equal_weight"},
                "costs": {"commission_bps": 5, "slippage_bps": 5},
                "execution": {"rebalance": "monthly"},
            }
            db.execute(
                """INSERT OR IGNORE INTO backtest_runs
                   (backtest_hash, prediction_hash, spec_json, stage, created_at, git_commit)
                   VALUES (?,?,?,?,?,?)""",
                (b_hash, best_pred_hash, json.dumps(spec), stage, FIXTURE_TS, "fixture"),
            )
            sharpe = sharpe_by_stage.get(stage, 0.5)
            db.execute(
                """INSERT OR IGNORE INTO backtest_metrics
                   (backtest_hash, computed_at, sharpe, sortino, total_return,
                    max_drawdown, cagr, volatility, calmar, n_periods)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    b_hash,
                    FIXTURE_TS,
                    sharpe,
                    sharpe * 1.2,
                    sharpe * 0.1,
                    -0.15,
                    sharpe * 0.05,
                    0.12,
                    sharpe * 0.33,
                    252,
                ),
            )

    db.commit()
    db.close()

    # Create synthetic prediction parquets for ALL prediction_hashes in the
    # registry (both fixture-generated and sampled-from-production). Uses real
    # symbols from setup.yaml and dates spanning the holdout boundary so backtest
    # notebooks can run (results are garbage but the pipeline completes).
    _backfill_all_prediction_parquets(cs_dir, cs_id)


def _backfill_all_backtest_artifacts(cs_dir: Path) -> None:
    """Generate synthetic daily_returns.parquet for every backtest_runs entry.

    Creates `run_log/backtest/{hash}/daily_returns.parquet` with a small daily
    Float64 return series so notebooks that resolve a backtest hash and read
    its daily-returns artifact (e.g., 17_portfolio_construction/01_portfolio_metrics)
    have a file to load. Values are bounded random noise — CI only needs the
    pipeline to complete, not to reproduce production performance.
    """
    db_path = cs_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return

    try:
        import numpy as np
        import polars as _pl
    except ImportError:
        return

    db = sqlite3.connect(str(db_path))
    try:
        rows = db.execute("SELECT backtest_hash FROM backtest_runs").fetchall()
    except sqlite3.OperationalError:
        rows = []
    db.close()
    if not rows:
        return

    bt_root = cs_dir / "run_log" / "backtest"
    bt_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_days = 1000
    import datetime as _dt

    base = _dt.date(2020, 1, 1)
    day_list = [base + _dt.timedelta(days=i) for i in range(n_days)]

    for (b_hash,) in rows:
        artifact_dir = bt_root / b_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "daily_returns.parquet"
        if path.exists():
            continue
        returns = rng.normal(loc=0.0005, scale=0.012, size=n_days).astype("float64")
        df = _pl.DataFrame({"timestamp": day_list, "daily_return": returns}).with_columns(
            _pl.col("timestamp").cast(_pl.Date)
        )
        df.write_parquet(path)


# The identifier a prediction artifact names its assets with. `symbol` everywhere
# except cme_futures, which uses `product`; the older test-data vintages also carry
# `entity` and `ticker`. Resolved from the frame rather than declared per case study
# so a reference panel can be read whichever convention wrote it.
ENTITY_COLUMN_CANDIDATES = ("symbol", "product", "entity", "ticker")

# Distinct dates a seeded artifact carries. Matches the fabricated grid's budget:
# a reference panel is adopted for its keys, not for its length.
SEEDED_DATE_BUDGET = 60


def _reference_panels(cs_dir: Path, hash_rows: list, survives, entity_col: str, _pl) -> dict:
    """One key/target panel per (split, label), taken from an artifact left in place.

    Which artifact: the panel that the most untouched artifacts in the group already
    share, then the larger panel, then the lowest hash. The choice has to come out
    the same on every regeneration, and seeding onto the panel the group already
    agrees on is what makes the seeded sets joinable with the copied ones rather
    than only with each other.

    Artifacts this function will not rewrite are preferred, so the panel is normally
    one that survives seeding. A group whose only on-disk artifacts WILL be rewritten
    still uses one of them as the panel rather than falling back to the fabricated
    grid: rewriting an artifact onto its own keys changes the scores and not the
    grid, so the group stays joinable and keeps a real one.

    That fallback is not cosmetic. crypto_perps_funding sets ``rewrite_existing``, so
    only its cohort leaders survive - and its ``fwd_ret_8h`` leader has no artifact on
    disk at all. Without the fallback the whole label fell to ``_weekday_grid``, which
    emits about sixty *dates* across the window, roughly eight weekdays apart. An 8H
    label cannot live on that grid, and 13_backtest rejected it in CI with
    "decision intervals [9 days, 10 days, 12 days] do not match horizon 8H" while four
    real artifacts on the correct 8-hour grid sat in the same group unused.

    The dates are subsampled to ``SEEDED_DATE_BUDGET``; they are a subset of the
    reference's own dates, which the registry places inside the split window, so the
    split boundary the fabricated grid enforces still holds.
    """
    groups: dict = {}
    fallback_groups: dict = {}
    for p_hash, split, label in hash_rows:
        if survives(p_hash):
            groups.setdefault((split, label), []).append(p_hash)
        elif (cs_dir / "run_log" / "predictions" / p_hash / "predictions.parquet").is_file():
            fallback_groups.setdefault((split, label), []).append(p_hash)
    fallback_only = set()
    for key, hashes in fallback_groups.items():
        if key not in groups:
            groups[key] = hashes
            fallback_only.add(key)

    panels = {}
    for key, hashes in groups.items():
        by_signature: dict = {}
        for p_hash in sorted(hashes):
            frame = _pl.read_parquet(
                cs_dir / "run_log" / "predictions" / p_hash / "predictions.parquet"
            )
            entity = next((c for c in ENTITY_COLUMN_CANDIDATES if c in frame.columns), None)
            if entity is None or not {"timestamp", "actual"} <= set(frame.columns):
                continue
            # A rewritten artifact is only worth keeping as the panel because it
            # carries a real decision grid. One with a single timestamp carries no
            # grid - it is the placeholder a superseded set leaves behind - so the
            # group is better served by the fabricated one than by inheriting it.
            if key in fallback_only and frame["timestamp"].n_unique() < 2:
                continue
            # Only ever compared for equality, so stringifying the identifiers is
            # enough and keeps a column carrying nulls from raising in sorted().
            signature = (
                frame.height,
                tuple(sorted(map(str, frame[entity].unique().to_list()))),
                tuple(map(str, frame["timestamp"].unique().sort().to_list())),
            )
            by_signature.setdefault(signature, []).append((p_hash, frame, entity))
        if not by_signature:
            continue
        _, entries = min(
            by_signature.items(),
            key=lambda item: (-len(item[1]), -item[0][0], item[1][0][0]),
        )
        _, frame, entity = entries[0]
        panels[key] = _subsampled_panel(frame, entity, entity_col, _pl)
    return panels


def _intraday_split_skeleton(panels: dict, split, _pl):
    """A reference panel from another label of the same split, keys and folds only.

    Returned only when the panels this case study has are evenly spaced below a
    day. On a daily case study the fabricated weekday grid is a reasonable
    stand-in and labels can legitimately differ; on an intraday one it is not a
    stand-in at all, and the decision times are a property of the case study
    rather than of the label.
    """
    from datetime import timedelta

    candidates = []
    for (panel_split, panel_label), panel in sorted(
        panels.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        if panel_split != split or panel_label is None:
            continue
        dates = panel["timestamp"].unique().sort()
        if dates.len() < 2:
            continue
        if dates.diff().drop_nulls().min() < timedelta(days=1):
            candidates.append(panel)
    return candidates[0] if candidates else None


def _subsampled_panel(frame, entity: str, entity_col: str, _pl):
    """A reference artifact reduced to the canonical seeded columns and date budget.

    Keeps the reference's own timestamp and identifier dtypes: a seeded artifact has
    to meet the reference on an exact join, and a cast on either side of that key
    would silently drop every row. The identifier is still renamed to the case
    study's declared ``entity_col`` - cme_futures registers ``product`` while its
    copied artifacts carry ``symbol``, and the notebooks resolve that column per
    frame, so following the reference's name here would change what every seeded
    cme artifact is called to fix a join that already works on values.
    """
    dates = frame["timestamp"].unique().sort()
    # A stride is fine on a daily reference, whose own gaps are already uneven
    # across weekends, and it keeps the seeded panel spanning the whole window.
    # It is wrong on an evenly spaced INTRADAY reference: crypto_perps_funding
    # decides every 8 hours, 1956 times, so a stride of 1956 // 60 hands the
    # seeded sets decisions 10 days 16 hours apart and 13_backtest rejects them
    # against a horizon of 8H. Nothing about the panel is recoverable after that
    # - the spacing IS the thing being checked - so a sub-daily reference keeps a
    # contiguous run at its native spacing instead. Ending at the reference's own
    # last decision is what the stride already aimed at, and the run stays inside
    # the split window because every date in it is the reference's own.
    from datetime import timedelta

    # The SMALLEST gap, not the only gap: a panel already sliced per fold carries a
    # jump between folds, and requiring a single distinct gap would reject exactly
    # the intraday panels this exists to recognise.
    gaps = dates.diff().drop_nulls() if dates.len() > 1 else None
    evenly_spaced_intraday = gaps is not None and gaps.min() < timedelta(days=1)
    if evenly_spaced_intraday:
        # Per fold, not across the panel. A flat tail lands entirely inside the
        # last fold and the seeded set comes back with n_folds=1, which
        # expanding-window conformal calibration refuses outright.
        fold_col = next((c for c in ("fold", "fold_id") if c in frame.columns), None)
        if fold_col is None:
            kept = dates.tail(SEEDED_DATE_BUDGET)
        else:
            folds = frame[fold_col].unique().sort()
            per_fold = max(2, SEEDED_DATE_BUDGET // max(1, folds.len()))
            kept = (
                frame.group_by(fold_col)
                .agg(_pl.col("timestamp").unique().sort().tail(per_fold))
                .explode("timestamp")["timestamp"]
                .unique()
                .sort()
            )
    else:
        step = max(1, dates.len() // SEEDED_DATE_BUDGET)
        kept = dates.gather(range(0, dates.len(), step))
        if kept[-1] != dates[-1]:
            kept = kept.append(dates[-1:])
    panel = frame.filter(_pl.col("timestamp").is_in(kept.implode()))
    if "fold" in panel.columns:
        fold = _pl.col("fold")
    else:
        # Mirrors the fabricated grid: folds partition dates, never rows, or every
        # symbol lands in one fold and per-symbol conformal calibration breaks.
        fold = (
            _pl.col("timestamp")
            .rank("dense")
            .sub(1)
            .floordiv(max(1, kept.len() // 2 + 1))
            .mod(2)
            .alias("fold")
        )
    columns = [
        _pl.col(entity).alias(entity_col),
        _pl.col("timestamp"),
        fold.alias("fold"),
        _pl.col("actual"),
    ]
    # A reference artifact of a classification label carries the continuous
    # evaluation target beside the class one. Dropping it here would leave the
    # seeded sets of that group to invent a replacement for a column the group
    # already has, and two members of one group would then disagree about what the
    # realized return was.
    if "eval_actual" in panel.columns:
        columns.append(_pl.col("eval_actual"))
    return panel.select(columns)


def _normalize_prediction_timestamp_zone(cs_dir: Path) -> None:
    """Put one case study's prediction artifacts in the timezone its labels use.

    `_subsampled_panel` preserves each reference artifact's own dtype, and says
    why: a seeded set has to meet its reference on an exact join, and a cast on
    either side of that key silently drops every row. That is right *within* a
    (split, label) group and says nothing across groups, so when the fixture's own
    artifacts disagree a notebook that reads two labels together fails:

        SchemaError: failed to determine supertype of datetime[us] and
                     datetime[us, UTC]

    crypto_perps_funding ships nine artifacts in three dtypes: four `ms, UTC`,
    four `ms` naive, one `us` naive. Stripping the zone collapses all eight
    validation artifacts onto one identical grid and the holdout artifact onto its
    own, so the three are the same instants written three ways.

    The zone comes from the case study's OWN labels rather than from a majority
    vote or a default, because the labels are what a prediction is ultimately
    joined against - crypto's are `ms, UTC` throughout, and normalizing the
    predictions to naive instead traded one collision for another:

        datatypes of join keys don't match - `timestamp`: datetime[ms] on left
        does not match `timestamp`: datetime[ms, UTC] on right

    Only the zone is touched. The time unit is left alone: the fixture already
    mixes `ms` and `us` across labels and features without trouble, so unifying it
    would rewrite artifacts to fix nothing. Values are untouched either way, so
    the historical-target checks a replay notebook runs against a cohort leader
    are unaffected - which is what makes it safe to rewrite an artifact that
    otherwise survives seeding.

    A case study whose artifacts already agree with its labels is not touched.
    """
    predictions = cs_dir / "run_log" / "predictions"
    if not predictions.is_dir():
        return
    try:
        import polars as _pl
    except ImportError:
        return

    zone = None
    for label_file in sorted((cs_dir / "labels").glob("*.parquet")):
        try:
            dtype = _pl.read_parquet_schema(label_file).get("timestamp")
        except Exception:  # noqa: BLE001 - an unreadable label is not ours to fix here
            continue
        if isinstance(dtype, _pl.Datetime):
            zone = dtype.time_zone
            break
    if zone is None:
        return

    for path in sorted(predictions.glob("*/predictions.parquet")):
        try:
            dtype = _pl.read_parquet_schema(path).get("timestamp")
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(dtype, _pl.Datetime) or dtype.time_zone == zone:
            continue
        frame = _pl.read_parquet(path)
        column = _pl.col("timestamp")
        # A naive column carries the same wall clock the zoned ones do, so it is
        # stamped rather than shifted; a genuinely different zone is converted.
        if dtype.time_zone is None:
            frame = frame.with_columns(column.dt.replace_time_zone(zone))
        else:
            frame = frame.with_columns(column.dt.convert_time_zone(zone))
        frame.write_parquet(path)


def _drop_stale_conformal_widths(cs_dir: Path) -> None:
    """Remove a widths artifact that no longer matches the predictions beside it.

    ``load_conformal_widths`` regenerates a missing ``conformal_widths.parquet``,
    and refuses one that predates the ``calibration_version`` column rather than
    silently mis-reading it. The fixture ships exactly such a legacy file for
    crypto_perps_funding/4d279db5157d, which took 14_portfolio_management down at
    its thirteenth cell.

    A widths file is also stale whenever the predictions next to it were just
    rewritten - the calibration residuals came from scores that no longer exist -
    so both are dropped and left to regenerate on demand. Deleting is the whole
    fix: nothing here has to know how to compute widths.
    """
    predictions = cs_dir / "run_log" / "predictions"
    if not predictions.is_dir():
        return
    try:
        import polars as _pl
    except ImportError:
        return
    for widths in predictions.glob("*/conformal_widths.parquet"):
        try:
            columns = _pl.read_parquet_schema(widths).keys()
        except Exception:  # noqa: BLE001 - an unreadable artifact is a stale one
            widths.unlink(missing_ok=True)
            continue
        if "calibration_version" not in columns:
            widths.unlink(missing_ok=True)


def _record_prediction_artifact_digests(cs_dir: Path) -> None:
    """Record the digest the artifact actually has, once the artifact exists.

    `PredictionResult.complete` verifies `value_digest(...)` of the parquet against
    `prediction_coverage.artifact_digest` (`research/results.py:419-423`), and that check is
    stricter than the catalog's `complete` column - which reads `coverage.status` and stops
    there. So a population can freeze cleanly from the catalog and be rejected by
    `require_complete` one line later, which is what happened here: the fixture wrote a
    fabricated 12-character `_make_hash` value where `value_digest` produces 16, so no
    seeded prediction could ever match and every path reaching the stricter definition saw
    `complete=False`.

    Recording the real digest is better than recording none. `results.py:412-418` treats a
    NULL as "legacy, backfill it" rather than as a conflict, so NULL would also have worked -
    but it makes the fixture silent about its artifacts where it can be truthful about them,
    and a later change that starts requiring the digest would find nothing recorded.

    Runs after `_backfill_all_prediction_parquets`, because the digest cannot be computed
    before the file it describes exists.

    Every row with an artifact is (re)recorded, not only the ones left empty. The seeder
    rewrites artifacts after they are sampled - `_backfill_all_prediction_parquets` writes
    synthetic scores over most of them, `_normalize_prediction_timestamp_zone` rewrites the
    timestamp column of the rest - so a digest carried over from production describes a file
    that no longer exists. While the sampler copied no `prediction_coverage`, every row
    reached here empty and the distinction did not arise; once it copies the table, a stale
    production digest survives and every reader of that prediction reads it as incomplete.
    """
    db_path = cs_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return
    import polars as pl

    from case_studies.utils.artifact_digest import value_digest

    with sqlite3.connect(str(db_path)) as db:
        rows = db.execute(
            "SELECT prediction_hash, artifact_digest FROM prediction_coverage"
        ).fetchall()
        for p_hash, recorded in rows:
            artifact = cs_dir / "run_log" / "predictions" / p_hash / "predictions.parquet"
            if not artifact.is_file():
                continue
            try:
                digest = value_digest(pl.read_parquet(artifact))
            except Exception:
                continue
            if digest == recorded:
                continue
            db.execute(
                "UPDATE prediction_coverage SET artifact_digest = ? WHERE prediction_hash = ?",
                (digest, p_hash),
            )
        db.commit()


def _backfill_all_prediction_parquets(cs_dir: Path, cs_id: str) -> None:
    """Generate synthetic prediction parquets for every hash in the registry.

    Every hash registered under one (split, label) is seeded on one key and target
    panel, so any two of them join. Where the fixture already carries an artifact
    this function leaves alone, that artifact supplies the panel - see
    ``_reference_panels``. Otherwise the panel is built from setup.yaml's symbols
    over the window the registry row declares - the CV validation window for
    ``validation`` rows, the configured holdout for ``holdout`` rows - so a seeded
    artifact can never claim decisions outside the split it is registered under.
    Predictions are random noise either way.

    Crypto artifacts are normalized even when copied intermediates exist because its
    model analysis requires that one common panel - except each label's cohort-leader
    prediction, which a replay notebook pins by hash and checks against real
    historical values; those keep whatever real artifact already exists on disk, and
    are the panel the rest of the label is seeded onto.
    """
    db_path = cs_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return

    try:
        import numpy as np
        import polars as _pl
    except ImportError:
        return

    # Kept separate from the numpy/polars import above: without it every hash
    # falls back to the holdout-relative grid, which is a weaker fixture. Folding
    # it into that try would make an import failure skip the backfill entirely,
    # leaving notebooks with no prediction artifacts at all.
    try:
        from case_studies.utils.cv_window import canonical_window
    except ImportError:
        canonical_window = None

    # Get every prediction hash with the split and label it is registered under,
    # so each artifact can be built inside its own window. The registry predates
    # this join in some case studies, so a missing table or column must degrade
    # to the old hash-only behaviour rather than abort seeding.
    db = sqlite3.connect(str(db_path))
    try:
        hash_rows = db.execute(
            """
            SELECT ps.prediction_hash, ps.split, t.label
            FROM prediction_sets ps
            LEFT JOIN training_runs t ON ps.training_hash = t.training_hash
            """
        ).fetchall()
    except sqlite3.OperationalError:
        hash_rows = [
            (r[0], None, None)
            for r in db.execute("SELECT prediction_hash FROM prediction_sets").fetchall()
        ]
    # How many folds each prediction's own training run declares. A fabricated panel that
    # carries a different number is not a weaker fixture, it is a rejected one:
    # `insight_chapter.load_selected_predictions` reads the declared count off the training
    # spec and refuses an artifact whose fold column does not enumerate exactly
    # `range(n_folds)`. Two was hard-coded here, so every notebook reaching that check saw
    # `expected fold IDs [0..7], observed [0, 1]` for any hash the sampler happened to make
    # selectable - which is how widening the sample took ch13 down on etfs/77ca2284b27b.
    declared_folds: dict[str, int] = {}
    try:
        from case_studies.utils.registry.specs import declared_fold_count

        for _p_hash, _spec_json in db.execute(
            """
            SELECT ps.prediction_hash, t.spec_json
            FROM prediction_sets ps
            JOIN training_runs t ON ps.training_hash = t.training_hash
            """
        ).fetchall():
            if not _spec_json:
                continue
            try:
                count = declared_fold_count(json.loads(_spec_json))
            except (ValueError, TypeError):
                continue
            if count > 0:
                declared_folds[_p_hash] = count
    except (sqlite3.OperationalError, ImportError):
        pass
    # Cohort-leader predictions - the frozen carrier a strategy-analysis/portfolio/
    # cost/risk notebook resolves via cohort_metrics(cohort_type='stagelabel',
    # stage='signal') and pins by hash. Those notebooks check the carrier's real
    # historical target against real raw prices (e.g. a >0.99 correlation gate), which
    # synthetic noise can never satisfy - so these hashes keep whatever real artifact
    # is already on disk instead of being swept into the generic rewrite below.
    # A registry predating the cohort_metrics table has no leaders to exempt, and that
    # is the only condition worth tolerating here. Catching OperationalError outright
    # would also absorb schema drift or a typo in the query, empty the exemption set,
    # and let the rewrite below overwrite every real carrier artifact with noise.
    has_cohort_metrics = (
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cohort_metrics'"
        ).fetchone()
        is not None
    )
    cohort_leader_hashes: set = set()
    if has_cohort_metrics:
        cohort_leader_hashes = {
            r[0]
            for r in db.execute(
                """
                SELECT DISTINCT b.prediction_hash
                FROM cohort_metrics c
                JOIN backtest_runs b ON b.backtest_hash = c.leader_hash
                WHERE c.cohort_type = 'stagelabel' AND c.stage = 'signal'
                """
            ).fetchall()
        }
    db.close()
    if not hash_rows:
        return

    # Read symbols from setup.yaml (fall back to generic)
    setup_path = CS_ROOT / cs_id / "config" / "setup.yaml"
    symbols = ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"]
    holdout_start = "2024-01-01"
    entity_col = "symbol"
    # Labels whose predictions carry a continuous evaluation target beside the class
    # one. `labels.classification_eval_label` in setup.yaml is where production
    # decides this: utils.modeling.load_modeling_dataset reads it for a
    # classification label, and every writer then persists the column as
    # `eval_actual` (registry/store.py). The mapping holds exactly - across all nine
    # canonical registries, every artifact of a label named here carries the column
    # and no other artifact does - so it is also what tells this function which
    # synthetic artifacts need it.
    eval_target_labels: set = set()
    if setup_path.exists():
        setup = yaml.safe_load(setup_path.read_text())
        eval_target_labels = set((setup.get("labels") or {}).get("classification_eval_label") or {})
        universe = setup.get("universe", {})
        assets = universe.get("assets", [])
        if assets:
            symbols = assets[:10]  # Cap at 10 for test speed
        if cs_id == "cme_futures":
            entity_col = "product"
        eval_cfg = setup.get("evaluation", {})
        if eval_cfg.get("holdout_start"):
            holdout_start = eval_cfg["holdout_start"]

    from datetime import date, timedelta

    def _weekday_grid(start: date, end: date) -> list[date]:
        """~60 weekdays spanning ``start`` through ``end``, inclusive."""
        days = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        if not days:
            return days
        step = max(1, len(days) // 60)
        grid = days[::step]
        # Keep the true end date: consumers compare the artifact's last decision
        # against the window's last date, and subsampling can drop it.
        if grid[-1] != days[-1]:
            grid.append(days[-1])
        return grid

    # Fallback grids, used when a hash's window cannot be derived. They are keyed by
    # split rather than universal: a single holdout-relative range spanning six months
    # past the boundary hands a `validation` hash decisions inside the holdout, which
    # is the defect this function exists to prevent. The degrade paths are ordinary
    # (a NULL label, an absent label parquet, an older registry schema), so the
    # contract has to hold on them too.
    ho = date.fromisoformat(holdout_start)
    unknown_split_window = (ho - timedelta(days=730), ho + timedelta(days=180))
    fallback_windows = {
        "validation": (ho - timedelta(days=730), ho - timedelta(days=1)),
        "holdout": (ho, ho + timedelta(days=180)),
    }

    def _window_for(split: str | None, label: str | None) -> tuple[date, date]:
        """The date range a hash registered under ``split`` is allowed to cover.

        A seeded artifact that reaches past its own split's window is not a
        weaker fixture, it is a wrong one: notebooks that enforce the sealed
        validation/holdout boundary read the artifact's last decision date and
        reject the carrier. Falls back to the split's own holdout-relative grid
        when the case study has no derivable window for the label.
        """
        fallback = fallback_windows.get(split or "", unknown_split_window)
        if canonical_window is None or split not in fallback_windows or not label:
            return fallback
        try:
            window = canonical_window(cs_id, label, split=split)
        except Exception:  # noqa: BLE001 — a missing label parquet must not break seeding
            return fallback
        if not window:
            return fallback
        # A window that holds no weekday yields an empty grid and therefore an empty
        # artifact. Degrading to this split's own fallback keeps the boundary; the
        # global one used to be reached here and crossed it.
        return window if _weekday_grid(*window) else fallback

    # The keys a fabricated panel has to meet. Production writes a prediction on the same
    # (entity, timestamp) types its labels carry, and a notebook joins the two; a panel keyed
    # on strings and dates when the labels carry UInt32 and Datetime does not fail here, it
    # fails several stages downstream on a polars join:
    #
    #   symbol: str on left does not match symbol: u32 on right      (us_firm_characteristics)
    #   timestamp: date on left does not match timestamp: datetime[us]  (nasdaq100_microstructure)
    #
    # So both the entity values and the two key dtypes come from the case study's own labels
    # where it has them, and fall back to setup.yaml's symbol list otherwise. Only the fabricated
    # grid needs this: a panel borrowed from a copied artifact already carries production's types.
    entity_dtype = None
    timestamp_dtype = None
    for label_file in sorted((cs_dir / "labels").glob("*.parquet")):
        try:
            schema = _pl.read_parquet_schema(label_file)
        except Exception:  # noqa: BLE001 - an unreadable label is not ours to fix here
            continue
        if entity_col not in schema or "timestamp" not in schema:
            continue
        entity_dtype = schema[entity_col]
        timestamp_dtype = schema["timestamp"]
        try:
            entities = (
                _pl.read_parquet(label_file, columns=[entity_col])[entity_col]
                .unique()
                .sort()
                .head(10)
                .to_list()
            )
        except Exception:  # noqa: BLE001
            entities = []
        if entities:
            symbols = entities
        break

    n_symbols = len(symbols)
    target_rng = np.random.default_rng(42)
    templates: dict[tuple[tuple[date, date], int], tuple[object, int]] = {}

    def _template_for(window: tuple[date, date], n_folds: int = 2):
        """One reusable frame per (window, fold count); scores are added per hash."""
        cached = templates.get((window, n_folds))
        if cached is not None:
            return cached
        dates = _weekday_grid(*window)
        n_dates = len(dates)
        n = n_symbols * n_dates
        rows_symbol = [s for _ in dates for s in symbols]
        rows_date = [d for d in dates for _ in range(n_symbols)]
        # Canonical production schema: prediction / actual / fold (NOT y_score / y_true / fold_id).
        # Notebooks read these columns by name; using non-canonical names here would silently
        # break downstream notebooks that resolve hashes from the registry.
        #
        # Fold assignment must mirror walk-forward CV: every symbol is present in every
        # fold for the dates in that fold's window. Assigning fold by row index (e.g.,
        # i % 2) silently partitions symbols across folds and breaks per-symbol
        # conformal calibration (each symbol ends up in one fold only). Partition by
        # date instead so all symbols share the same fold on each date.
        # Contiguous date blocks, one per fold, so the grid mirrors walk-forward CV and
        # every declared fold id appears. The block is floored and the last one absorbs the
        # remainder: a ceiling would leave the highest folds empty whenever the grid holds
        # only slightly more dates than folds, and an absent fold id fails the same check
        # a wrong fold count does.
        block = max(1, n_dates // max(1, n_folds))
        rows_fold = [
            min(_di // block, n_folds - 1) for _di in range(n_dates) for _ in range(n_symbols)
        ]
        frame = _pl.DataFrame(
            {
                entity_col: _pl.Series(rows_symbol, dtype=entity_dtype)
                if entity_dtype is not None
                else _pl.Series(rows_symbol),
                "timestamp": _pl.Series(rows_date).cast(timestamp_dtype or _pl.Date),
                "fold": rows_fold,
                "actual": target_rng.normal(0, 0.01, n).tolist(),
            }
        )
        # This function drew `actual` itself and knows it is a continuous return, so
        # a classification set on this panel can evaluate against it directly.
        # Carried on every window rather than only the classification ones: it costs
        # one column and keeps the panel a single object, so which labels need it is
        # decided at write time.
        frame = frame.with_columns(_pl.col("actual").alias("eval_actual"))
        templates[(window, n_folds)] = (frame, n)
        return templates[(window, n_folds)]

    rewrite_existing = cs_id == "crypto_perps_funding"
    missing_leaders = sorted(
        p_hash
        for p_hash in cohort_leader_hashes
        if not (cs_dir / "run_log" / "predictions" / p_hash / "predictions.parquet").is_file()
    )
    if missing_leaders:
        # The exemption above can only preserve an artifact that exists. A leader with
        # none still falls through to the synthetic write, and the notebook that pins
        # its hash then fails a >0.99 correlation gate against real prices several
        # stages downstream, nowhere near this function. The gap is reported by hash
        # at regeneration time rather than left to surface there. Not fatal: seven of
        # the nine fixtures are missing at least one leader artifact today, so raising
        # would stop every regeneration on a pre-existing gap this function did not
        # introduce.
        #
        # Do NOT close the gap by copying the production artifact in. A copied fixture
        # is stale the moment anything upstream is regenerated, and every case study is
        # being retrained end to end. What the correlation gate needs is a panel whose
        # entities and timestamps are the ones the fixture's own labels carry, and whose
        # historical target is derived from the same synthetic series the scores come
        # from, so the check is satisfied by construction. That is what the branch below
        # already does wherever a reference panel exists; a leader with no artifact of
        # its own is the case that still falls back to a fabricated grid.
        warnings.warn(
            f"{cs_id}: no predictions.parquet on disk for cohort-leader prediction(s) "
            f"{', '.join(missing_leaders)}; each gets synthetic scores on a fabricated "
            "entity grid that a replay notebook's historical-target check will reject. "
            "Generate the artifact against the fixture's own labels, deriving its "
            "target from the series the scores come from; never copy the production one.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _survives(p_hash: str) -> bool:
        """True when the loop below leaves this artifact exactly as it is."""
        pred_file = cs_dir / "run_log" / "predictions" / p_hash / "predictions.parquet"
        return pred_file.is_file() and (p_hash in cohort_leader_hashes or not rewrite_existing)

    # Two prediction sets of one (split, label) have to be joinable on
    # (timestamp, entity), or a notebook that pairs them measures nothing:
    # 14_latent_factors/09_case_study_insights ranks a latent and a supervised
    # configuration against their common target and read "Aligned targets disagree:
    # maximum gap None" - max() over an empty join - because the latent set was a
    # fabricated weekday grid over SYM0..SYM4 and the supervised set was an artifact
    # copied from production. Placeholder symbols cannot meet real ones on any key,
    # so where the fixture already carries an artifact, the seeded sets take its
    # keys, folds and realized targets and synthesize only the score - preferring
    # one that survives, and otherwise using one that will be rewritten onto its
    # own keys. Only a group with no artifact at all keeps the fabricated grid.
    reference_panels = _reference_panels(cs_dir, hash_rows, _survives, entity_col, _pl)

    for p_hash, split, label in hash_rows:
        pred_dir = cs_dir / "run_log" / "predictions" / p_hash
        pred_file = pred_dir / "predictions.parquet"
        if _survives(p_hash):
            continue
        reference = reference_panels.get((split, label))
        borrowed = False
        if reference is None:
            # A label with no artifact of its own borrows the decision grid of
            # another label in the same split, but only where the panels this
            # case study does have are evenly spaced and sub-daily - which is
            # exactly where the fabricated weekday grid cannot represent the
            # decisions at all. crypto_perps_funding decides every 8 hours for
            # every label, and `fwd_dir_8h` is the direction cut of the same
            # return as `fwd_ret_8h`, so its keys are that label's keys.
            #
            # Only the keys and folds are borrowed. `actual` is redrawn below,
            # because the lending label's realized value is not this label's -
            # a direction label's target is a class, not a return.
            reference = _intraday_split_skeleton(reference_panels, split, _pl)
            borrowed = reference is not None
        if reference is not None:
            template, n = reference, reference.height
        else:
            template, n = _template_for(_window_for(split, label), declared_folds.get(p_hash, 2))
        pred_dir.mkdir(parents=True, exist_ok=True)
        score_seed = int(hashlib.sha256(p_hash.encode()).hexdigest()[:16], 16)
        scores = np.random.default_rng(score_seed).normal(0, 0.01, n).tolist()
        frame = template.with_columns(_pl.Series("prediction", scores))
        if borrowed:
            actual_seed = int(
                hashlib.sha256(f"actual/{split}/{label}".encode()).hexdigest()[:16], 16
            )
            frame = frame.with_columns(
                _pl.Series("actual", np.random.default_rng(actual_seed).normal(0, 0.01, n).tolist())
            )
        if label in eval_target_labels and "eval_actual" not in frame.columns:
            # The panel is a copied artifact that does not carry the column. Its
            # `actual` cannot stand in: on a classification set that is the class
            # label, and a rank correlation against it measures class separation
            # rather than a ranking against returns - the substitution
            # 07_case_study_insights refuses to make. Draw a continuous target
            # instead, per (split, label)
            # so every set of the group agrees on what the realized return was.
            group_seed = int(hashlib.sha256(f"{split}/{label}".encode()).hexdigest()[:16], 16)
            frame = frame.with_columns(
                _pl.Series(
                    "eval_actual",
                    np.random.default_rng(group_seed).normal(0, 0.01, frame.height),
                )
            )
        if label not in eval_target_labels and "eval_actual" in frame.columns:
            # A regression set never carries it in production, and the panel it was
            # seeded onto may be a classification artifact that does.
            frame = frame.drop("eval_actual")
        frame.write_parquet(str(pred_file))


def _seed_causal_json(results_dir: Path, cs_id: str, label: str) -> None:
    """Seed results/causal_dml.json for Ch15 insights."""
    path = results_dir / "causal_dml.json"
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "case_study_id": cs_id,
                "label": label,
                "treatment": "momentum_21d",
                "summary": {
                    "ate": 0.003,
                    "ate_se": 0.001,
                    "refutation_placebo": {"new_effect": 0.0001, "p_value": 0.85},
                    "refutation_subset": {"new_effect": 0.0028, "p_value": 0.02},
                },
            },
            indent=2,
        )
    )


def _seed_feature_json(results_dir: Path, cs_id: str) -> None:
    """Seed results/ch08_features.json for Ch08 feature summary."""
    path = results_dir / "ch08_features.json"
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "case_study_id": cs_id,
                "evaluation": {
                    "n_features": 15,
                    "n_features_tested": 15,
                    "n_significant_fdr05": 8,
                    "inflation_factor": 1.5,
                    "max_pairwise_corr": 0.72,
                    "corr_pairs_above_07": 3,
                    "top_features": ["past_ret_21d", "vol_21d", "rsi_14"],
                    "metrics": {"ic_mean": 0.02, "ic_std": 0.01},
                },
            },
            indent=2,
        )
    )


def _seed_temporal_json(results_dir: Path, cs_id: str) -> None:
    """Seed results/ch09_temporal.json for Ch09 temporal summary."""
    path = results_dir / "ch09_temporal.json"
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "case_study_id": cs_id,
                "incremental_evaluation": {
                    "temporal_models": ["arima", "garch", "kalman"],
                    "ic_contribution": {"arima": 0.005, "garch": 0.003, "kalman": 0.002},
                },
            },
            indent=2,
        )
    )


def _drop_legacy_conformal_widths(cs_dir: Path) -> None:
    """Remove seeded conformal widths that predate the `calibration_version` column.

    `case_studies/utils/conformal.py` refuses a widths artifact without that column - the
    calibration it records cannot be identified, so a sizing decision made from it cannot be
    attributed to a version. It also regenerates the artifact when the file is absent, so
    deleting a legacy one is exactly the "preserve and regenerate" the refusal asks for.

    The fixture data ships some of these: ml4t/third-edition-test-data holds pre-column
    widths written 2026-08-25, and the first notebook to size with `conformal_weighted`
    fails on them rather than on anything it did. Dropping them here keeps that repo and
    this one from having to move together for a column that the code can rebuild.
    """
    import polars as _pl

    widths = cs_dir / "run_log" / "predictions"
    if not widths.is_dir():
        return
    for path in widths.glob("*/conformal_widths.parquet"):
        try:
            columns = _pl.read_parquet(path).columns
        except Exception:  # unreadable is handled downstream as "no prior widths"
            continue
        if "calibration_version" not in columns:
            path.unlink()


def _seed_demo_predictions(cs_dir: Path, cs_id: str, primary_label: str) -> None:
    """Seed demo prediction parquets for live-simulation notebooks (Ch25).

    Ch25 notebooks check CASE_DIR / "models" / "predictions_reg_{horizon}d.parquet"
    first. CASE_DIR = get_case_study_dir() which redirects to ML4T_OUTPUT_DIR in tests.
    We seed a single flat predictions file there so the first check succeeds.
    """
    try:
        import numpy as np
        import polars as _pl
    except ImportError:
        return

    CS_CONFIGS = {
        "cme_futures": {
            "asset_col": "product",
            "assets": ["CL", "NG", "GC", "ES", "ZN", "6E"],
            "horizons": [5, 21],
        },
        "fx_pairs": {
            "asset_col": "symbol",
            "assets": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            "horizons": [1, 5],
        },
        "us_equities_panel": {
            "asset_col": "symbol",
            "assets": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "horizons": [5, 21],
        },
        "etfs": {
            "asset_col": "symbol",
            "assets": ["SPY", "QQQ", "IWM", "EFA", "TLT"],
            "horizons": [21],
        },
    }
    config = CS_CONFIGS.get(cs_id)
    if not config:
        return

    rng = np.random.default_rng(42)
    models_dir = cs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for horizon in config["horizons"]:
        pred_file = models_dir / f"predictions_reg_{horizon}d.parquet"
        if pred_file.exists():
            continue
        n_days = 60
        rows = []
        for i in range(n_days):
            for asset in config["assets"]:
                rows.append(
                    {
                        "timestamp": f"2024-{(i // 22) + 1:02d}-{(i % 22) + 1:02d}",
                        config["asset_col"]: asset,
                        "prediction": float(rng.normal(0, 0.01)),
                    }
                )
        df = _pl.DataFrame(rows).with_columns(_pl.col("timestamp").str.to_date().alias("timestamp"))
        df.write_parquet(str(pred_file))


def _seed_news_features(output_dir: Path) -> None:
    """Seed a minimal news_features.parquet for Ch10/08_text_feature_evaluation.

    The notebook loads from get_output_dir(8, "fnspid") / "news_features.parquet".
    In test mode that becomes {ML4T_OUTPUT_DIR}/ch08_fnspid/news_features.parquet.
    Required columns: symbol, timestamp, fwd_ret_1d, fwd_ret_5d, fwd_ret_20d,
    weighted_surprise, sentiment_mean, sentiment_momentum, coverage_count.
    """
    try:
        import numpy as np
        import polars as _pl
    except ImportError:
        return

    out_dir = output_dir / "ch08_fnspid"
    path = out_dir / "news_features.parquet"
    if path.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM"]
    from datetime import date, timedelta

    start = date(2023, 1, 3)
    dates = [
        start + timedelta(days=i) for i in range(60) if (start + timedelta(days=i)).weekday() < 5
    ]
    n = len(symbols) * len(dates)

    df = _pl.DataFrame(
        {
            "symbol": [s for _ in dates for s in symbols],
            "timestamp": _pl.Series([d for d in dates for _ in symbols]).cast(_pl.Date),
            "fwd_ret_1d": rng.normal(0, 0.01, n).tolist(),
            "fwd_ret_5d": rng.normal(0, 0.02, n).tolist(),
            "fwd_ret_20d": rng.normal(0, 0.04, n).tolist(),
            "weighted_surprise": rng.normal(0, 0.5, n).tolist(),
            "sentiment_mean": rng.normal(0, 0.3, n).tolist(),
            "sentiment_momentum": rng.normal(0, 0.2, n).tolist(),
            "coverage_count": rng.poisson(3, n).tolist(),
        }
    )
    df.write_parquet(str(path))


def _seed_ch16_parity_json() -> None:
    """Seed cached parity JSON artifacts for Ch16 notebooks 15-18.

    These notebooks read from get_chapter_dir(16) / "resources" / "<name>.json".
    That path is NOT redirected by ML4T_OUTPUT_DIR — it's a real code-repo path.
    """
    resources_dir = REPO_ROOT / "16_strategy_simulation" / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    # NB15: lean_parity_results.json
    _write_if_missing(
        resources_dir / "lean_parity_results.json",
        {
            "artifact_source": "fixture",
            "scenario_id": "multi_250_20yr",
            "scenario_label": "250 assets, 20 years daily",
            "data_source": "fixture",
            "cached": True,
            "limitations": ["Fixture data for CI testing"],
            "results": [
                {
                    "framework_id": "ml4t-lean",
                    "label": "ml4t-backtest (LEAN profile)",
                    "num_trades": 428459,
                    "final_value": 1234567.89,
                    "runtime_sec": 12.5,
                    "data_points": 1250000,
                },
                {
                    "framework_id": "lean",
                    "label": "QuantConnect LEAN CLI",
                    "num_trades": 428459,
                    "final_value": 1234566.34,
                    "runtime_sec": 95.3,
                    "data_points": 1250000,
                },
            ],
            "comparison": {
                "trade_gap": 0,
                "trade_gap_pct": 0.0,
                "final_value_gap": 1.55,
                "final_value_gap_pct": 1.255e-06,
                "runtime_speedup": 7.62,
                "remaining_gap_driver": "price_precision",
                "notes": [
                    "next-bar open execution is aligned",
                    "margin-enabled LEAN account semantics are aligned",
                    "decoded fill chronology matches exactly at event identity and 4-decimal price",
                ],
            },
        },
    )

    # NB16 (case_study_lean_parity_results.json), NB17
    # (backtrader_zipline_parity_results.json), and NB18
    # (vectorbt_parity_results.json) are intentionally NOT seeded here.
    # Their artifacts hold genuine engine-parity numbers and are committed under
    # 16_strategy_simulation/resources/ (version-controlled, always present on
    # checkout), so the fabricated CI fallbacks were removed. NB16 is reproducible
    # via ml4t.backtest._validation.case_study_lean; NB17 and NB18 via
    # validation/benchmark_suite.py.


def _write_if_missing(path: Path, data: dict) -> None:
    """Write JSON file only if it doesn't already exist."""
    if path.exists():
        return
    path.write_text(json.dumps(data, indent=2))


def seed_results(output_dir: Path, case_study_ids: list[str]) -> None:
    """Write minimal fixture results into test output directories.

    Creates:
    1. results/*.json — legacy format for downstream comparisons
    2. run_log/registry.db — SQLite registry for case_study_insights + model_analysis
    3. results/causal_dml.json — Ch15 causal insights
    4. results/ch08_features.json, ch09_temporal.json — Ch08/09 summaries

    Only writes files that don't already exist (upstream notebooks may have
    produced real results during the same test session).
    """
    for cs_id in case_study_ids:
        setup_path = CS_ROOT / cs_id / "config" / "setup.yaml"
        if not setup_path.exists():
            continue

        setup = yaml.safe_load(setup_path.read_text())
        primary_label = setup.get("labels", {}).get("primary")
        if not primary_label:
            continue

        # Get all label configs for this case study
        labels = [primary_label]
        variants = setup.get("labels", {}).get("variants", [])
        if isinstance(variants, list):
            labels.extend(v if isinstance(v, str) else v.get("name", "") for v in variants)

        cs_dir = output_dir / cs_id
        results_dir = cs_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        for label in labels:
            if not label:
                continue

            # Linear fixture
            linear_path = results_dir / f"linear_{label}.json"
            if not linear_path.exists():
                linear_path.write_text(json.dumps(_linear_fixture(label), indent=2))

            # GBM fixture
            gbm_path = results_dir / f"gbm_{label}.json"
            if not gbm_path.exists():
                gbm_path.write_text(json.dumps(_gbm_fixture(label), indent=2))

            # TabDL fixture
            tabdl_path = results_dir / f"tabular_dl_{label}.json"
            if not tabdl_path.exists():
                tabdl_path.write_text(json.dumps(_tabular_dl_fixture(label), indent=2))

        # Registry DB — the primary data source for insights + analysis notebooks
        _seed_registry_db(cs_dir, cs_id, primary_label)

        # Backfill prediction parquets for ALL hashes in registry
        # (must run AFTER _seed_registry_db, and also when registry was pre-seeded)
        _backfill_all_prediction_parquets(cs_dir, cs_id)
        _drop_stale_conformal_widths(cs_dir)
        _normalize_prediction_timestamp_zone(cs_dir)
        _record_prediction_artifact_digests(cs_dir)
        _drop_legacy_conformal_widths(cs_dir)

        # Backfill daily_returns.parquet for ALL backtest hashes in registry
        # so downstream notebooks (e.g., 17/01_portfolio_metrics) that resolve a
        # backtest hash and read its daily-returns artifact have a file to load.
        # Must run from outer loop because _seed_registry_db early-returns when
        # the schema is already current, skipping any post-commit work.
        _backfill_all_backtest_artifacts(cs_dir)

        # Ch15 causal insights
        _seed_causal_json(results_dir, cs_id, primary_label)

        # Ch08/09 feature + temporal summaries
        _seed_feature_json(results_dir, cs_id)
        _seed_temporal_json(results_dir, cs_id)

        # Ch25 live-simulation demo predictions
        _seed_demo_predictions(cs_dir, cs_id, primary_label)

    # --- Non-case-study chapter fixtures ---
    _seed_news_features(output_dir)
    _seed_ch16_parity_json()
