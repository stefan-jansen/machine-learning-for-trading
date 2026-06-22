"""Tests for case_studies/utils/analytics.py — registry query contracts.

Per memory rule ``feedback_results_json``: "Registry only, never JSONs."
This module is the canonical path Ch6–Ch20 insights notebooks take to pull
IC / AUC / Sharpe numbers into the book. A silent regression in any of
these queries would corrupt every cross-chapter summary table.

The tests pin three layers:

1. **Metadata invariants** — the handful of dicts that declare the 9
   case-study IDs must agree on keys, so a new case study can't be added
   to one dict and forgotten in another.

2. **Path resolution** — ``_cs_dir`` / ``_registry_path`` honor
   ``ML4T_OUTPUT_DIR`` for test isolation.

3. **Query contracts** — against a seeded SQLite registry:
   - ``load_model_ic`` filters by family, split, case_studies list and
     returns the expected rows with a ``case_study`` label column.
   - ``load_classification_metrics`` requires ``task_type = 'classification'``.
   - ``load_best_ic_per_family`` picks max IC per (case_study, family)
     pair and optionally restricts to the primary label.
   - ``load_chapter_backtests("ch16")`` maps to ``stage="signal"`` and
     joins backtest_runs × backtest_metrics × prediction_sets ×
     training_runs.
   - Spec helpers: ``extract_cost_bps`` sums commission + slippage
     from either v1 or v2 backtest specs; ``extract_allocator`` reads
     strategy.allocation.method.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils import analytics

# -----------------------------------------------------------------------------
# Metadata invariants
# -----------------------------------------------------------------------------


def test_case_study_ids_match_metadata_keys() -> None:
    assert list(analytics.CASE_STUDY_META.keys()) == analytics.CASE_STUDY_IDS


@pytest.mark.parametrize(
    "dict_name",
    ["PRIMARY_LABELS", "SHORT_NAMES", "DATASET_META", "CADENCE_MAP", "DISPLAY_NAMES"],
)
def test_metadata_dict_keys_match_case_study_ids(dict_name) -> None:
    """Every metadata dict must enumerate the same 9 case studies — otherwise
    cross-dict joins in load_best_ic_per_family / load_chapter_backtests drop
    rows silently.
    """
    d = getattr(analytics, dict_name)
    assert set(d.keys()) == set(analytics.CASE_STUDY_IDS), (
        f"{dict_name} keys differ: missing {set(analytics.CASE_STUDY_IDS) - set(d.keys())}, "
        f"extra {set(d.keys()) - set(analytics.CASE_STUDY_IDS)}"
    )


def test_primary_labels_are_non_empty_strings() -> None:
    for cs, lbl in analytics.PRIMARY_LABELS.items():
        assert isinstance(lbl, str) and lbl, f"{cs}: empty/non-string primary label"


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------


def test_cs_dir_production_path(monkeypatch) -> None:
    """With no ML4T_OUTPUT_DIR, _cs_dir falls back to REPO_ROOT/case_studies."""
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)
    from utils.paths import REPO_ROOT

    assert analytics._cs_dir() == REPO_ROOT / "case_studies"


def test_cs_dir_redirects_to_output_dir_when_registry_present(tmp_path, monkeypatch) -> None:
    """With ML4T_OUTPUT_DIR set AND a registry.db present under tmp, _cs_dir
    returns the tmp root instead of the production case_studies path.
    """
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "etfs" / "run_log").mkdir(parents=True)
    (tmp_path / "etfs" / "run_log" / "registry.db").touch()
    assert analytics._cs_dir("etfs") == tmp_path


def test_cs_dir_falls_back_when_registry_missing_under_output_dir(tmp_path, monkeypatch) -> None:
    """ML4T_OUTPUT_DIR set but no registry.db under it → fall back to production."""
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    from utils.paths import REPO_ROOT

    assert analytics._cs_dir("etfs") == REPO_ROOT / "case_studies"


def test_registry_path_is_three_levels_deep(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "etfs" / "run_log").mkdir(parents=True)
    (tmp_path / "etfs" / "run_log" / "registry.db").touch()
    p = analytics._registry_path("etfs")
    assert p == tmp_path / "etfs" / "run_log" / "registry.db"


# -----------------------------------------------------------------------------
# _query behavior on empty / missing databases
# -----------------------------------------------------------------------------


def test_query_returns_empty_df_when_path_missing(tmp_path) -> None:
    missing = tmp_path / "nope.db"
    out = analytics._query(missing, "SELECT 1")
    assert isinstance(out, pl.DataFrame)
    assert out.is_empty()


def test_query_returns_empty_df_when_no_rows(tmp_path) -> None:
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()
    out = analytics._query(db, "SELECT * FROM t")
    assert out.is_empty()


def test_query_returns_populated_df(tmp_path) -> None:
    db = tmp_path / "rows.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x INTEGER, y TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()
    out = analytics._query(db, "SELECT * FROM t ORDER BY x")
    assert out.to_dicts() == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]


# -----------------------------------------------------------------------------
# Seeded-registry fixture: builds the schema + minimal rows
# -----------------------------------------------------------------------------


def _create_registry_schema(conn: sqlite3.Connection) -> None:
    """Create the subset of the registry schema the analytics queries touch."""
    conn.executescript(
        """
        CREATE TABLE training_runs (
            training_hash TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            label TEXT NOT NULL,
            config_name TEXT,
            spec_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE prediction_sets (
            prediction_hash TEXT PRIMARY KEY,
            training_hash TEXT NOT NULL,
            checkpoint_value INTEGER,
            checkpoint_kind TEXT,
            split TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE prediction_metrics (
            prediction_hash TEXT PRIMARY KEY,
            computed_at TEXT NOT NULL,
            ic_mean REAL,
            ic_std REAL,
            ic_t REAL,
            n_folds REAL,
            pct_positive REAL,
            task_type TEXT,
            accuracy REAL,
            balanced_accuracy REAL,
            auc_roc REAL,
            auc_pr REAL,
            log_loss REAL,
            brier_score REAL
        );
        CREATE TABLE backtest_runs (
            backtest_hash TEXT PRIMARY KEY,
            prediction_hash TEXT NOT NULL,
            spec_json TEXT,
            stage TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE backtest_metrics (
            backtest_hash TEXT PRIMARY KEY,
            computed_at TEXT NOT NULL,
            sharpe REAL,
            sortino REAL,
            total_return REAL,
            max_drawdown REAL
        );
        CREATE TABLE fold_metrics (
            prediction_hash TEXT NOT NULL,
            fold_id INTEGER NOT NULL,
            computed_at TEXT NOT NULL,
            ic REAL,
            PRIMARY KEY (prediction_hash, fold_id)
        );
        """
    )


def _seed_registry(db_path: Path) -> None:
    """Populate a registry with 4 training runs + predictions + 2 backtests.

    Layout (etfs):
        linear_a / fwd_ret_21d / validation / ic=0.05 / regression
        linear_b / fwd_ret_21d / validation / ic=0.08 / regression  <- best linear
        gbm_a / fwd_ret_21d / validation / ic=0.10 / regression  <- best gbm (primary)
        gbm_a / fwd_ret_21d / holdout / ic=0.03 / regression
        linear_c / fwd_dir_5d / validation / ic=0.04 / classification (task_type='classification')
    Plus 1 ch16 (signal) backtest and 1 ch17 (allocation) backtest on the
    same gbm prediction_hash.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    _create_registry_schema(conn)

    # training_runs
    conn.executemany(
        "INSERT INTO training_runs VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("th_lin_a", "linear", "fwd_ret_21d", "ridge_a100", None, "2024-01-01T00:00:00"),
            ("th_lin_b", "linear", "fwd_ret_21d", "ridge_b100", None, "2024-01-01T00:00:00"),
            ("th_gbm_a", "gbm", "fwd_ret_21d", "default", None, "2024-01-01T00:00:00"),
            ("th_lin_c", "linear", "fwd_dir_5d", "logistic", None, "2024-01-01T00:00:00"),
        ],
    )
    # prediction_sets
    conn.executemany(
        "INSERT INTO prediction_sets VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("ph_lin_a_val", "th_lin_a", 0, "final", "validation", "2024-01-02T00:00:00"),
            ("ph_lin_b_val", "th_lin_b", 0, "final", "validation", "2024-01-02T00:00:00"),
            ("ph_gbm_a_val", "th_gbm_a", 100, "final", "validation", "2024-01-02T00:00:00"),
            ("ph_gbm_a_hol", "th_gbm_a", 100, "final", "holdout", "2024-01-02T00:00:00"),
            ("ph_lin_c_val", "th_lin_c", 0, "final", "validation", "2024-01-02T00:00:00"),
        ],
    )
    # prediction_metrics — regression and classification
    conn.executemany(
        "INSERT INTO prediction_metrics (prediction_hash, computed_at, ic_mean, task_type, "
        "auc_roc, accuracy) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("ph_lin_a_val", "2024-01-03", 0.05, "regression", None, None),
            ("ph_lin_b_val", "2024-01-03", 0.08, "regression", None, None),
            ("ph_gbm_a_val", "2024-01-03", 0.10, "regression", None, None),
            ("ph_gbm_a_hol", "2024-01-03", 0.03, "regression", None, None),
            ("ph_lin_c_val", "2024-01-03", 0.04, "classification", 0.62, 0.55),
        ],
    )
    # backtest_runs — one signal (ch16) + one allocation (ch17) on the gbm prediction
    spec_v2 = {
        "version": 2,
        "strategy": {"allocation": {"method": "inverse_vol"}},
        "backtest_config": {
            "commission": {"rate": 0.0005},  # 5 bps
            "slippage": {"rate": 0.0003},  # 3 bps
        },
    }
    conn.executemany(
        "INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?)",
        [
            ("bh_sig", "ph_gbm_a_val", json.dumps(spec_v2), "signal", "2024-01-04"),
            ("bh_alloc", "ph_gbm_a_val", json.dumps(spec_v2), "allocation", "2024-01-04"),
        ],
    )
    conn.executemany(
        "INSERT INTO backtest_metrics (backtest_hash, computed_at, sharpe, sortino, "
        "total_return, max_drawdown) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("bh_sig", "2024-01-05", 1.2, 1.8, 0.35, -0.10),
            ("bh_alloc", "2024-01-05", 1.5, 2.2, 0.45, -0.08),
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def seeded_registries(tmp_path, monkeypatch) -> Path:
    """Build registries for etfs + crypto_perps_funding under a temp output dir.

    The second case study (crypto) is intentionally empty (only schema) so
    multi-case-study queries have a no-op partition to merge against.
    """
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    _seed_registry(tmp_path / "etfs" / "run_log" / "registry.db")

    # crypto: schema but no rows
    crypto_db = tmp_path / "crypto_perps_funding" / "run_log" / "registry.db"
    crypto_db.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(crypto_db))
    _create_registry_schema(conn)
    conn.commit()
    conn.close()

    return tmp_path


# -----------------------------------------------------------------------------
# load_model_ic
# -----------------------------------------------------------------------------


def test_load_model_ic_returns_all_families_by_default(seeded_registries) -> None:
    df = analytics.load_model_ic(case_studies=["etfs", "crypto_perps_funding"], split="validation")
    # etfs: 3 validation rows (lin_a, lin_b, gbm_a) with regression task_type;
    # lin_c is also validation but the query doesn't filter task_type here.
    assert df.height == 4
    assert set(df["case_study"].unique().to_list()) == {"etfs"}
    assert set(df["family"].unique().to_list()) == {"linear", "gbm"}


def test_load_model_ic_filters_by_family(seeded_registries) -> None:
    df = analytics.load_model_ic(families="gbm", case_studies=["etfs"], split="validation")
    assert df["family"].unique().to_list() == ["gbm"]
    assert df.height == 1


def test_load_model_ic_filters_by_split_holdout(seeded_registries) -> None:
    df = analytics.load_model_ic(case_studies=["etfs"], split="holdout")
    # only gbm_a has a holdout prediction
    assert df.height == 1
    assert df["split"].to_list() == ["holdout"]
    assert df["ic_mean"].to_list() == [0.03]


def test_load_model_ic_returns_empty_when_no_case_study_has_data(seeded_registries) -> None:
    df = analytics.load_model_ic(case_studies=["crypto_perps_funding"], split="validation")
    assert df.is_empty()


def test_load_model_ic_has_case_study_label_column(seeded_registries) -> None:
    df = analytics.load_model_ic(case_studies=["etfs"], split="validation")
    assert "case_study" in df.columns
    assert df["case_study"].unique().to_list() == ["etfs"]


# -----------------------------------------------------------------------------
# load_classification_metrics
# -----------------------------------------------------------------------------


def test_load_classification_metrics_filters_task_type_eq_1(seeded_registries) -> None:
    """Only the linear_c row (task_type='classification') should come back."""
    df = analytics.load_classification_metrics(case_studies=["etfs"], split="validation")
    assert df.height == 1
    assert df["family"].to_list() == ["linear"]
    assert df["auc_roc"].to_list() == [0.62]
    assert df["task_type"].to_list() == ["classification"]


def test_load_classification_metrics_excludes_regression_rows(seeded_registries) -> None:
    """Regression rows (task_type='regression') must not leak into the classification view."""
    df = analytics.load_classification_metrics(case_studies=["etfs"], split="validation")
    # Spec: no rows with null AUC should appear
    assert df.filter(pl.col("auc_roc").is_null()).is_empty()


# -----------------------------------------------------------------------------
# load_best_ic_per_family
# -----------------------------------------------------------------------------


def test_load_best_ic_per_family_picks_max_per_pair(seeded_registries) -> None:
    """Primary label for etfs is fwd_ret_21d. Among linear runs on that label,
    lin_b (IC=0.08) beats lin_a (IC=0.05). gbm has only one run (IC=0.10).
    """
    best = analytics.load_best_ic_per_family(case_studies=["etfs"], split="validation")
    rows = {(r["case_study"], r["family"]): r for r in best.to_dicts()}
    assert rows[("etfs", "linear")]["ic_mean"] == 0.08
    assert rows[("etfs", "linear")]["config_name"] == "ridge_b100"
    assert rows[("etfs", "gbm")]["ic_mean"] == 0.10


def test_load_best_ic_per_family_primary_label_excludes_other_labels(
    seeded_registries,
) -> None:
    """With use_primary_label=True (default), the linear_c run on fwd_dir_5d must
    not appear — only rows with label == primary are kept.
    """
    best = analytics.load_best_ic_per_family(case_studies=["etfs"], split="validation")
    assert all(row["label"] == "fwd_ret_21d" for row in best.to_dicts())


def test_load_best_ic_per_family_use_primary_label_false_includes_all(
    seeded_registries,
) -> None:
    """With use_primary_label=False, the fwd_dir_5d row competes for best linear."""
    best = analytics.load_best_ic_per_family(
        case_studies=["etfs"], split="validation", use_primary_label=False
    )
    # linear: best of {lin_a 0.05, lin_b 0.08, lin_c 0.04} is lin_b
    linear_row = next(r for r in best.to_dicts() if r["family"] == "linear")
    assert linear_row["ic_mean"] == 0.08


def test_load_best_ic_per_family_adds_display_name(seeded_registries) -> None:
    best = analytics.load_best_ic_per_family(case_studies=["etfs"], split="validation")
    assert "display_name" in best.columns
    assert best["display_name"].unique().to_list() == ["ETFs"]


# -----------------------------------------------------------------------------
# load_chapter_backtests
# -----------------------------------------------------------------------------


def test_load_chapter_backtests_ch16_maps_to_signal_stage(seeded_registries) -> None:
    """chapter='ch16' → stage='signal' → returns the bh_sig backtest row."""
    df = analytics.load_chapter_backtests("ch16", case_studies=["etfs"])
    assert df.height == 1
    assert df["backtest_hash"].to_list() == ["bh_sig"]


def test_load_chapter_backtests_explicit_stage_overrides_chapter(seeded_registries) -> None:
    df = analytics.load_chapter_backtests("ch16", stage="allocation", case_studies=["etfs"])
    assert df["backtest_hash"].to_list() == ["bh_alloc"]


def test_load_chapter_backtests_joins_sharpe_and_training_columns(seeded_registries) -> None:
    df = analytics.load_chapter_backtests("ch17", case_studies=["etfs"])
    assert df.height == 1
    row = df.to_dicts()[0]
    assert row["sharpe"] == 1.5
    assert row["family"] == "gbm"
    assert row["config_name"] == "default"


def test_load_chapter_backtests_metrics_filter_selects_columns(seeded_registries) -> None:
    df = analytics.load_chapter_backtests("ch17", case_studies=["etfs"], metrics=["sharpe"])
    # Only meta columns + sharpe; sortino/total_return/max_drawdown excluded
    assert "sharpe" in df.columns
    assert "sortino" not in df.columns
    assert "total_return" not in df.columns


def test_load_chapter_backtests_returns_empty_for_unused_stage(seeded_registries) -> None:
    df = analytics.load_chapter_backtests("ch18", case_studies=["etfs"])
    assert df.is_empty()


# -----------------------------------------------------------------------------
# Spec helpers
# -----------------------------------------------------------------------------


def test_parse_backtest_spec_round_trips_json() -> None:
    spec = {"version": 2, "strategy": {"foo": "bar"}, "backtest_config": {}}
    assert analytics.parse_backtest_spec(json.dumps(spec)) == spec


def test_extract_cost_bps_sums_commission_and_slippage_from_v2_spec() -> None:
    spec_json = json.dumps(
        {
            "version": 2,
            "strategy": {},
            "backtest_config": {
                "commission": {"rate": 0.0005},  # 5 bps
                "slippage": {"rate": 0.0003},  # 3 bps
            },
        }
    )
    assert analytics.extract_cost_bps(spec_json) == pytest.approx(8.0)


def test_extract_cost_bps_handles_v1_spec() -> None:
    """v1 specs store costs in a flat ``costs`` dict; cost_view falls back to it."""
    spec_json = json.dumps({"costs": {"commission_bps": 2.0, "slippage_bps": 1.5}})
    assert analytics.extract_cost_bps(spec_json) == pytest.approx(3.5)


def test_extract_allocator_reads_strategy_allocation_method() -> None:
    spec_json = json.dumps(
        {
            "version": 2,
            "strategy": {"allocation": {"method": "risk_parity"}},
            "backtest_config": {},
        }
    )
    assert analytics.extract_allocator(spec_json) == "risk_parity"


def test_extract_allocator_defaults_to_unknown_when_missing() -> None:
    spec_json = json.dumps({"version": 2, "strategy": {}, "backtest_config": {}})
    assert analytics.extract_allocator(spec_json) == "unknown"
