"""Selecting predictions from a registry that holds more than one traded universe."""

from __future__ import annotations

import json
import sqlite3

from case_studies.utils.registry import resolve_best_predictions


def _build_registry(case_dir) -> None:
    """Two predictions, both swept cost-feasible, only one also swept on the full universe.

    ``mixed`` is the shape ``nasdaq100_microstructure``'s two-pass sweep produces: pass 1
    scores every admissible prediction on the cost-feasible universe, pass 2 re-runs the
    highest-scoring few on the full universe. ``feasible_only`` never reaches pass 2.
    The full-universe row is the weaker of ``mixed``'s two, so a ranking that ignores the
    universe axis puts ``mixed`` first on a number no full-universe backtest carries.

    ``spelled_full`` carries the literal string ``"full"`` the sp500_options cost cascade
    writes, which is the same universe said the other way.
    """
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
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL, ic_std REAL);
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
        predictions = [
            ("mixed", "cfg_mixed"),
            ("feasible_only", "cfg_feasible"),
            ("spelled_full", "cfg_spelled"),
        ]
        for prediction_hash, config in predictions:
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, 'gbm', ?, 'fwd_ret_15m')",
                (training_hash, config),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', 0)",
                (prediction_hash, training_hash),
            )
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, 0.1, 0.1, 0.0, 0.2, 10.0)",
                (prediction_hash,),
            )

        # (backtest_hash, prediction_hash, universe as the spec carries it, sharpe)
        backtests = [
            ("bt_mixed_feasible", "mixed", "cost_feasible", 3.0),
            ("bt_mixed_full", "mixed", None, 0.5),
            ("bt_feasible_only", "feasible_only", "cost_feasible", 2.0),
            # The sp500_options spelling of the same universe: an explicit "full"
            # rather than an absent key. Scored below `mixed` so a ranking that
            # reads only one of the two spellings is visible in the order.
            ("bt_spelled_full", "spelled_full", "full", 0.25),
        ]
        for backtest_hash, prediction_hash, universe, sharpe in backtests:
            signal: dict[str, object] = {"method": "equal_weight_top_k", "top_k": 5}
            # The absent key, not an explicit null: `14_backtest` writes
            # `universe_filter` only when a filter applies.
            if universe is not None:
                signal["universe_filter"] = universe
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, ?, 'signal')",
                (backtest_hash, prediction_hash, json.dumps({"strategy": {"signal": signal}})),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.1, 0.2, 0.1, 1)",
                (backtest_hash, sharpe),
            )


def _ranking(case_dir, **kwargs) -> list[tuple[str, float]]:
    selected = resolve_best_predictions(
        "test",
        "fwd_ret_15m",
        split="validation",
        stage="signal",
        top_n=10,
        case_dir=case_dir,
        **kwargs,
    )
    return list(zip(selected["prediction_hash"].to_list(), selected["sharpe"].to_list()))


def test_unscoped_ranking_mixes_the_two_universes(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)

    # Documented, not endorsed: this is why a caller sweeping one universe must say so.
    assert _ranking(case_dir) == [
        ("mixed", 3.0),
        ("feasible_only", 2.0),
        ("spelled_full", 0.25),
    ]


def test_named_universe_ranks_on_that_universe_only(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)

    assert _ranking(case_dir, universe_filter="cost_feasible") == [
        ("mixed", 3.0),
        ("feasible_only", 2.0),
    ]


def test_full_universe_selects_the_rows_carrying_no_filter(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)

    # The whole point: `feasible_only` has no full-universe backtest and must not appear,
    # and `mixed` must carry the Sharpe its full-universe row earned rather than its best.
    # `spelled_full` comes along because the absent key and the literal "full" name one
    # universe, which is what separates this from a plain equality on the JSON value.
    assert _ranking(case_dir, universe_filter="full") == [("mixed", 0.5), ("spelled_full", 0.25)]
    assert _ranking(case_dir, universe_filter="none") == _ranking(case_dir, universe_filter="full")
