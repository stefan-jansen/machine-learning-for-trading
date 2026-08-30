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
    assert explorer.best(top_n=10, label="fwd_ret_5d", prediction_hashes=["partial"])[
        "prediction_hash"
    ].to_list() == ["partial"]
    assert explorer.best(
        top_n=10,
        label="fwd_ret_5d",
        prediction_hashes=["partial", "full_a"],
    )["prediction_hash"].to_list() == ["full_a"]
    assert (
        explorer.compare_allocators(
            stages=("signal",),
            label="fwd_ret_5d",
            prediction_hashes=["partial"],
        )["n"].item()
        == 1
    )
    assert (
        explorer.compare_allocators(
            stages=("signal",),
            label="fwd_ret_5d",
            prediction_hashes=["partial", "full_a"],
        )["n"].item()
        == 1
    )


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


def test_prediction_population_is_filtered_before_checkpoint_ranking(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        for prediction_hash, sharpe, n_days in [
            ("retired", 5.0, 5),
            ("current", 0.75, 4),
        ]:
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, 'gbm', 'shared', 'fwd_ret_5d')",
                (training_hash,),
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
                "INSERT INTO backtest_runs VALUES (?, ?, '{}', 'signal')",
                (f"bt_{prediction_hash}", prediction_hash),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.1, 0.2, 0.1, 1)",
                (f"bt_{prediction_hash}", sharpe),
            )

    selected = resolve_best_predictions(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=10,
        case_dir=case_dir,
        prediction_hashes={"current"},
    )

    assert selected["prediction_hash"].to_list() == ["current"]


def test_backtest_population_is_filtered_before_run_ranking(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)

    selected = resolve_best_backtest_runs(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=1,
        case_dir=case_dir,
        prediction_hashes={"full_b"},
    )

    assert selected["prediction_hash"].to_list() == ["full_b"]


def test_backtest_population_sets_the_raw_coverage_bar(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)

    selected = resolve_best_backtest_runs(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=1,
        case_dir=case_dir,
        prediction_hashes={"partial"},
    )

    assert selected["prediction_hash"].to_list() == ["partial"]

    selected_with_peer = resolve_best_backtest_runs(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=10,
        case_dir=case_dir,
        prediction_hashes={"partial", "full_a"},
    )

    assert selected_with_peer["prediction_hash"].to_list() == ["full_a"]


def test_canonical_backtest_population_is_filtered_before_run_ranking(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    monkeypatch.setattr(
        "case_studies.utils.registry.queries.canonical_coverage_days",
        lambda case_study, label, split, prediction_hash, case_dir: {
            "partial": 2,
            "full_a": 4,
            "full_b": 4,
            "tabular": 2,
        }[prediction_hash],
    )

    selected = resolve_best_backtest_runs(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=1,
        case_dir=case_dir,
        coverage_window="canonical",
        prediction_hashes={"full_b"},
    )

    assert selected["prediction_hash"].to_list() == ["full_b"]


def test_canonical_backtest_population_sets_the_coverage_bar(tmp_path, monkeypatch) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    monkeypatch.setattr(
        "case_studies.utils.registry.queries.canonical_coverage_days",
        lambda case_study, label, split, prediction_hash, case_dir: {
            "partial": 2,
            "full_a": 4,
            "full_b": 4,
            "tabular": 2,
        }[prediction_hash],
    )

    selected = resolve_best_backtest_runs(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=1,
        case_dir=case_dir,
        coverage_window="canonical",
        prediction_hashes={"partial"},
    )

    assert selected["prediction_hash"].to_list() == ["partial"]

    selected_with_peer = resolve_best_backtest_runs(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=10,
        case_dir=case_dir,
        coverage_window="canonical",
        prediction_hashes={"partial", "full_a"},
    )

    assert selected_with_peer["prediction_hash"].to_list() == ["full_a"]


def test_canonical_prediction_population_sets_the_coverage_bar(tmp_path, monkeypatch) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    monkeypatch.setattr(
        "case_studies.utils.registry.queries.canonical_coverage_days",
        lambda case_study, label, split, prediction_hash, case_dir: {
            "partial": 2,
            "full_a": 4,
            "full_b": 4,
            "tabular": 2,
        }[prediction_hash],
    )

    selected = resolve_best_predictions(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=1,
        case_dir=case_dir,
        coverage_window="canonical",
        prediction_hashes={"partial"},
    )

    assert selected["prediction_hash"].to_list() == ["partial"]


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


def test_a_population_whose_only_backtested_run_is_incomplete_refuses(tmp_path) -> None:
    """An empty frame here says "no backtests"; the truth is "no COMPLETE run was backtested".

    etfs measured this under a preview reduction: a scoped population of six predictions and
    one backtest, where the maximum coverage belonged to a prediction nothing had backtested.
    The bar is the population's maximum `ic_n_days`, deliberately - that is what makes a run
    complete, and computing it over only the backtested rows would let an incomplete run set
    its own bar and win. So the filter is right to admit nothing, and wrong to say nothing.
    """
    import pytest

    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        # full_a is the coverage bar for gbm/fwd_ret_5d at 4.0 days. Take away its backtest
        # and the scoped population below has a bar no backtested member can reach.
        db.execute("DELETE FROM backtest_runs WHERE backtest_hash = 'bt_full_a'")
        db.execute("DELETE FROM backtest_metrics WHERE backtest_hash = 'bt_full_a'")
    explorer = BacktestExplorer("test", case_dir=case_dir)

    with pytest.raises(RuntimeError) as raised:
        explorer.best(top_n=10, label="fwd_ret_5d", prediction_hashes=["partial", "full_a"])
    message = str(raised.value)
    assert "coverage bar" in message
    assert "partial" in message, f"the refusal must name the run it rejected, got: {message}"

    # The same population with its complete member backtested still resolves, and a
    # population with no backtests at all is still an ordinary empty answer.
    assert explorer.best(top_n=10, label="fwd_ret_5d", prediction_hashes=["full_b"])[
        "prediction_hash"
    ].to_list() == ["full_b"]
    assert explorer.best(top_n=10, label="fwd_ret_5d", prediction_hashes=["nothing"]).is_empty()


def test_an_empty_result_from_another_filter_is_not_blamed_on_the_coverage_bar(
    tmp_path,
) -> None:
    """The refusal must isolate the coverage clause, or it sends the reader after the wrong thing.

    `best` also drops runs that traded nothing and families the case study excludes. If the
    diagnostic query does not mirror those, a population emptied by one of them is reported
    as a coverage failure.
    """
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("DELETE FROM backtest_runs WHERE backtest_hash = 'bt_full_a'")
        db.execute("DELETE FROM backtest_metrics WHERE backtest_hash = 'bt_full_a'")
        # The one remaining backtested member of the population traded nothing, so `best`
        # drops it for that reason - not because it sits below the bar.
        db.execute("UPDATE backtest_metrics SET num_trades = 0 WHERE backtest_hash = 'bt_partial'")
    explorer = BacktestExplorer("test", case_dir=case_dir)

    assert explorer.best(
        top_n=10, label="fwd_ret_5d", prediction_hashes=["partial", "full_a"]
    ).is_empty()
