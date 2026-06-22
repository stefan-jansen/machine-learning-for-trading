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
from pathlib import Path

import yaml

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


def _seed_registry_db(cs_dir: Path, cs_id: str, primary_label: str) -> None:
    """Create a minimal registry.db with entries for all families and stages.

    Schema matches utils/registry.py REGISTRY_SCHEMA_SQL exactly.
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
                return  # Fully current schema — don't overwrite

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

    db = sqlite3.connect(str(db_path))
    db.executescript("""
        CREATE TABLE IF NOT EXISTS training_runs (
            training_hash TEXT PRIMARY KEY, family TEXT NOT NULL,
            label TEXT NOT NULL, config_name TEXT,
            spec_json TEXT, created_at TEXT NOT NULL,
            git_commit TEXT, entry_point TEXT
        );
        CREATE TABLE IF NOT EXISTS prediction_sets (
            prediction_hash TEXT PRIMARY KEY,
            training_hash TEXT NOT NULL REFERENCES training_runs(training_hash),
            checkpoint_value INTEGER, checkpoint_kind TEXT,
            split TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prediction_metrics (
            prediction_hash TEXT PRIMARY KEY REFERENCES prediction_sets(prediction_hash),
            computed_at TEXT NOT NULL,
            ic_mean REAL, ic_std REAL, ic_t REAL, n_folds REAL, n_obs REAL,
            n_periods REAL, pct_positive REAL, task_type REAL,
            accuracy REAL, balanced_accuracy REAL, auc_roc REAL, auc_pr REAL,
            log_loss REAL, brier_score REAL
        );
        CREATE TABLE IF NOT EXISTS fold_metrics (
            prediction_hash TEXT NOT NULL REFERENCES prediction_sets(prediction_hash),
            fold_id INTEGER NOT NULL,
            computed_at TEXT NOT NULL,
            ic REAL, ic_std REAL, n_periods REAL, n_obs REAL, n_entities REAL,
            rmse REAL, mae REAL,
            accuracy REAL, balanced_accuracy REAL, auc_roc REAL, auc_pr REAL,
            log_loss REAL, brier_score REAL,
            PRIMARY KEY (prediction_hash, fold_id)
        );
        CREATE TABLE IF NOT EXISTS backtest_runs (
            backtest_hash TEXT PRIMARY KEY,
            prediction_hash TEXT NOT NULL REFERENCES prediction_sets(prediction_hash),
            spec_json TEXT, stage TEXT, created_at TEXT NOT NULL, git_commit TEXT
        );
        CREATE TABLE IF NOT EXISTS backtest_metrics (
            backtest_hash TEXT PRIMARY KEY REFERENCES backtest_runs(backtest_hash),
            computed_at TEXT NOT NULL,
            sharpe REAL, sortino REAL, total_return REAL, max_drawdown REAL,
            cagr REAL, volatility REAL, calmar REAL, omega REAL, stability REAL,
            tail_ratio REAL, win_rate REAL, kurtosis REAL, skewness REAL,
            var_95 REAL, cvar_95 REAL, n_periods REAL,
            num_trades REAL, total_commission REAL, total_slippage REAL, avg_turnover REAL
        );
        CREATE TABLE IF NOT EXISTS backtest_fold_metrics (
            backtest_hash TEXT NOT NULL REFERENCES backtest_runs(backtest_hash),
            fold_id INTEGER NOT NULL, metric TEXT NOT NULL,
            value REAL, computed_at TEXT NOT NULL,
            PRIMARY KEY (backtest_hash, fold_id, metric)
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
    """)

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
                       (training_hash, family, label, config_name, spec_json, created_at, git_commit, entry_point)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        t_hash,
                        family,
                        label,
                        config_name,
                        json.dumps(spec),
                        FIXTURE_TS,
                        "fixture",
                        "fixture",
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
                       (prediction_hash, computed_at, ic_mean, ic_std, n_folds, n_obs)
                       VALUES (?,?,?,?,?,?)""",
                    (p_hash, FIXTURE_TS, ic, ic * 0.3, 2, 100),
                )
                # Fold metrics (2 folds) — wide format matching production schema
                for fold_id in range(2):
                    fold_ic = ic + (0.002 if fold_id == 0 else -0.002)
                    db.execute(
                        """INSERT OR IGNORE INTO fold_metrics
                           (prediction_hash, fold_id, computed_at, ic, ic_std, n_periods, n_obs, n_entities, rmse, mae)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (
                            p_hash,
                            fold_id,
                            FIXTURE_TS,
                            fold_ic,
                            fold_ic * 0.3,
                            50,
                            100,
                            5,
                            0.05,
                            0.03,
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


def _backfill_all_prediction_parquets(cs_dir: Path, cs_id: str) -> None:
    """Generate synthetic prediction parquets for every hash in the registry.

    Uses real symbols from setup.yaml and dates spanning 2 years before the
    holdout boundary so backtests have data to work with. Predictions are random
    noise — CI only needs the pipeline to complete, not meaningful results.
    """
    db_path = cs_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return

    try:
        import numpy as np
        import polars as _pl
    except ImportError:
        return

    # Get all prediction hashes from registry
    db = sqlite3.connect(str(db_path))
    hashes = [r[0] for r in db.execute("SELECT prediction_hash FROM prediction_sets").fetchall()]
    db.close()
    if not hashes:
        return

    # Read symbols from setup.yaml (fall back to generic)
    setup_path = CS_ROOT / cs_id / "config" / "setup.yaml"
    symbols = ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"]
    holdout_start = "2024-01-01"
    entity_col = "symbol"
    if setup_path.exists():
        setup = yaml.safe_load(setup_path.read_text())
        universe = setup.get("universe", {})
        assets = universe.get("assets", [])
        if assets:
            symbols = assets[:10]  # Cap at 10 for test speed
        if cs_id == "cme_futures":
            entity_col = "product"
        eval_cfg = setup.get("evaluation", {})
        if eval_cfg.get("holdout_start"):
            holdout_start = eval_cfg["holdout_start"]

    # Generate daily dates: 2 years before holdout through 6 months after
    from datetime import date, timedelta

    ho = date.fromisoformat(holdout_start)
    start = ho - timedelta(days=730)
    end = ho + timedelta(days=180)
    dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Weekdays only
            dates.append(d)
        d += timedelta(days=1)
    # Subsample to ~60 dates for speed
    step = max(1, len(dates) // 60)
    dates = dates[::step]

    rng = np.random.default_rng(42)
    n_symbols = len(symbols)
    n_dates = len(dates)
    n = n_symbols * n_dates

    # Build one template DataFrame, reuse for all hashes
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
    n_folds = 2
    rows_fold = [
        (_di // max(1, n_dates // n_folds + 1)) % n_folds
        for _di in range(n_dates)
        for _ in range(n_symbols)
    ]
    template = _pl.DataFrame(
        {
            entity_col: rows_symbol,
            "timestamp": _pl.Series(rows_date).cast(_pl.Date),
            "fold": rows_fold,
            "prediction": rng.normal(0, 0.01, n).tolist(),
            "actual": rng.normal(0, 0.01, n).tolist(),
        }
    )

    for p_hash in hashes:
        pred_dir = cs_dir / "run_log" / "predictions" / p_hash
        pred_file = pred_dir / "predictions.parquet"
        if pred_file.exists():
            continue
        pred_dir.mkdir(parents=True, exist_ok=True)
        template.write_parquet(str(pred_file))


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
