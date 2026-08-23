"""Coverage-aware selection regressions for backtest and cohort readers."""

from __future__ import annotations

import sqlite3

import polars as pl

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.cohort_metrics import (
    _list_family_cohorts,
    _list_label_cohorts,
    _list_stagelabel_cohorts,
    _stage_leader_hash,
)
from case_studies.utils.registry import resolve_best_backtest_runs, resolve_best_predictions
from case_studies.utils.registry.registration import register_cohort_metrics
from case_studies.utils.registry.store import _open_registry


def _build_registry(case_dir) -> None:
    run_log = case_dir / "run_log"
    run_log.mkdir(parents=True)
    with sqlite3.connect(run_log / "registry.db") as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT,
                config_name TEXT,
                label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY,
                training_hash TEXT,
                split TEXT,
                checkpoint_value REAL
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY,
                ic_mean REAL,
                ic_mean_daily REAL,
                ic_ci_lo REAL,
                ic_ci_hi REAL,
                ic_n_days REAL
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY,
                prediction_hash TEXT,
                spec_json TEXT,
                stage TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY,
                sharpe REAL,
                cagr REAL,
                max_drawdown REAL,
                total_return REAL,
                volatility REAL,
                num_trades REAL
            );
            CREATE TABLE backtest_fold_metrics (
                backtest_hash TEXT,
                fold_id INTEGER,
                sharpe REAL
            );
            """
        )
        rows = [
            ("partial", "gbm", "partial", 2.0, 10.0),
            ("full_a", "gbm", "full_a", 4.0, 1.0),
            ("full_b", "gbm", "full_b", 4.0, 0.5),
            ("tabular", "tabular_dl", "tabular", 2.0, 2.0),
        ]
        for prediction_hash, family, config, n_days, sharpe in rows:
            training_hash = f"train_{prediction_hash}"
            backtest_hash = f"bt_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, ?, ?, 'fwd_ret_5d')",
                (training_hash, family, config),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', 0)",
                (prediction_hash, training_hash),
            )
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, 0.1, 0.1, 0.0, 0.2, ?)",
                (prediction_hash, n_days),
            )
            db.execute(
                """
                INSERT INTO backtest_runs VALUES (
                    ?, ?, '{"allocation":{"method":"score_weighted"}}', 'signal'
                )
                """,
                (backtest_hash, prediction_hash),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.1, 0.2, 0.1, 1)",
                (backtest_hash, sharpe),
            )


def test_backtest_readers_exclude_partial_coverage(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    explorer = BacktestExplorer("test", case_dir=case_dir)

    best = explorer.best(top_n=10)
    assert best["backtest_hash"].to_list() == ["bt_tabular", "bt_full_a", "bt_full_b"]
    assert explorer.search_context()["total"] == 3
    assert explorer.compare_families().filter(family="gbm")["n"].item() == 2
    assert explorer.compare_allocators(stages=("signal",), label="fwd_ret_5d")["n"].item() == 3
    assert (
        explorer.compare_allocators(
            stages=("signal",),
            label="fwd_ret_5d",
            prediction_hashes=["full_a"],
        )["n"].item()
        == 1
    )
    assert explorer.best(top_n=10, label="fwd_ret_5d", prediction_hashes=["full_a"])[
        "prediction_hash"
    ].to_list() == ["full_a"]


def test_downstream_resolution_excludes_partial_coverage(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)

    selected = resolve_best_predictions(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=10,
        case_dir=case_dir,
    )

    assert "partial" not in selected["prediction_hash"].to_list()
    assert selected["prediction_hash"].to_list() == ["tabular", "full_a", "full_b"]

    backtests = resolve_best_backtest_runs(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=10,
        case_dir=case_dir,
    )
    assert backtests["prediction_hash"].to_list() == ["tabular", "full_a", "full_b"]


def test_cohort_members_and_leader_exclude_partial_coverage(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        family = _list_family_cohorts(db)
        stage_label = _list_stagelabel_cohorts(db)
        label = _list_label_cohorts(db)

        assert family == [("signal", "fwd_ret_5d", "gbm", ["bt_full_a", "bt_full_b"])]
        assert stage_label == [
            (
                "signal",
                "fwd_ret_5d",
                ["bt_full_a", "bt_full_b", "bt_tabular"],
            )
        ]
        assert label == [("fwd_ret_5d", ["bt_full_a", "bt_full_b", "bt_tabular"])]
        assert _stage_leader_hash(db, "signal", "fwd_ret_5d") == "bt_tabular"


def test_complete_cohort_snapshot_prunes_obsolete_identities(tmp_path) -> None:
    case_dir = tmp_path / "case"
    db = _open_registry(case_dir)
    db.execute("PRAGMA foreign_keys=OFF")
    db.execute(
        """
        INSERT INTO cohort_metrics (
            cohort_type, stage, label, family, leader_hash,
            k_variants, periods_per_year, computed_at
        ) VALUES ('family', 'signal', 'old_label', 'gbm', 'old_hash', 2, 252, 'old')
        """
    )
    db.commit()
    db.close()

    register_cohort_metrics("test", [], replace_all=True, case_dir=case_dir)

    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM cohort_metrics").fetchone()[0] == 0


def test_fold_metric_backfill_is_restricted_to_requested_label(tmp_path, monkeypatch) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("INSERT INTO training_runs VALUES ('train_alt', 'gbm', 'alt', 'fwd_ret_10d')")
        db.execute("INSERT INTO prediction_sets VALUES ('alt', 'train_alt', 'validation', 0)")
        db.execute("INSERT INTO prediction_metrics VALUES ('alt', 0.1, 0.1, 0, 0.2, 4)")
        db.execute("INSERT INTO backtest_runs VALUES ('bt_alt', 'alt', '{}', 'signal')")
        db.execute("INSERT INTO backtest_metrics VALUES ('bt_alt', 1, 0, 0, 0, 0, 1)")

    returns = pl.DataFrame({"timestamp": ["2020-01-01"], "daily_return": [0.0]})
    for backtest_hash in ("bt_full_a", "bt_alt"):
        artifact_dir = case_dir / "run_log" / "backtest" / backtest_hash
        artifact_dir.mkdir(parents=True)
        returns.write_parquet(artifact_dir / "daily_returns.parquet")

    seen: list[str] = []
    monkeypatch.setattr(
        "case_studies.utils.registry.compute_backtest_fold_metrics",
        lambda *args, **kwargs: {0: {"sharpe": 0.0}},
    )
    monkeypatch.setattr(
        "case_studies.utils.registry.register_backtest_fold_metrics",
        lambda case_study, backtest_hash, fold_metrics: seen.append(backtest_hash),
    )

    count = BacktestExplorer("test", case_dir=case_dir).backfill_fold_metrics(
        stage="signal",
        label="fwd_ret_5d",
    )

    assert count == 1
    assert seen == ["bt_full_a"]


def _reference_artifacts(case_dir, source_root, prediction_hashes: list[str]) -> None:
    """Record the rows a workspace copies and leave the parquets in the source root."""
    for prediction_hash in prediction_hashes:
        pred_dir = source_root / "run_log" / "predictions" / prediction_hash
        pred_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"symbol": ["A"], "prediction": [0.1]}).write_parquet(
            pred_dir / "predictions.parquet"
        )
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(
            "CREATE TABLE overlay_references ("
            "result_hash TEXT, result_kind TEXT, source_root TEXT, created_at TEXT)"
        )
        db.executemany(
            "INSERT INTO overlay_references VALUES (?, 'prediction', ?, '2026-08-23T00:00:00Z')",
            [(value, str(source_root)) for value in prediction_hashes],
        )


def test_selection_reaches_a_prediction_the_run_log_only_references(tmp_path) -> None:
    """A workspace run holds the rows and references the parquet; both must select."""
    case_dir = tmp_path / "workspace"
    _build_registry(case_dir)
    _reference_artifacts(case_dir, tmp_path / "release", ["full_a", "full_b", "tabular"])

    selected = resolve_best_predictions(
        "test", "fwd_ret_5d", split="validation", stage="signal", top_n=10, case_dir=case_dir
    )

    assert sorted(selected["prediction_hash"].to_list()) == ["full_a", "full_b", "tabular"]


def test_selection_survives_a_derived_artifact_landing_beside_a_referenced_prediction(
    tmp_path,
) -> None:
    """The predictions directory existing is not evidence that any prediction is in it.

    A workspace run writes derived per-prediction artifacts - conformal widths among them -
    into a directory it creates beside a prediction it only references. Treating the
    directory's existence as the test made this selection order-dependent: it returned every
    candidate before any such artifact was written and none afterwards, from one registry.
    """
    case_dir = tmp_path / "workspace"
    _build_registry(case_dir)
    _reference_artifacts(case_dir, tmp_path / "release", ["full_a", "full_b", "tabular"])
    derived = case_dir / "run_log" / "predictions" / "full_a"
    derived.mkdir(parents=True)
    pl.DataFrame({"alpha": [0.2], "width": [1.0]}).write_parquet(
        derived / "conformal_widths.parquet"
    )

    selected = resolve_best_predictions(
        "test", "fwd_ret_5d", split="validation", stage="signal", top_n=10, case_dir=case_dir
    )

    assert sorted(selected["prediction_hash"].to_list()) == ["full_a", "full_b", "tabular"]
