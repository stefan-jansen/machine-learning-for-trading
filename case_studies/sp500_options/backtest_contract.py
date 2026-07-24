"""Static execution contract for the S&P 500 options baseline sweep."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl

SP500_OPTIONS_EXECUTION_UNIVERSES: tuple[str, str] = ("full", "liquid")

ACCEPTED_DEEP_PRODUCERS: dict[str, tuple[str, str]] = {
    "lstm_h64": ("256627760faa", "159d481af175"),
    "patchtst": ("b19c49228948", "c592ca4defbb"),
}


def validate_accepted_deep_predictions(prediction_index: pl.DataFrame) -> pl.DataFrame:
    """Require the accepted LSTM and PatchTST identities in a full sweep."""
    required = {"family", "config_name", "training_hash", "prediction_hash"}
    missing = required - set(prediction_index.columns)
    if missing:
        raise pl.exceptions.ColumnNotFoundError(
            f"Prediction index cannot validate accepted deep producers; missing {sorted(missing)}"
        )

    observed = {
        row["config_name"]: (row["training_hash"], row["prediction_hash"])
        for row in prediction_index.filter(pl.col("family") == "deep_learning").iter_rows(
            named=True
        )
    }
    if observed != ACCEPTED_DEEP_PRODUCERS:
        raise RuntimeError(
            "Deep-producer identity mismatch: "
            f"expected {ACCEPTED_DEEP_PRODUCERS!r}, observed {observed!r}"
        )
    return prediction_index


def _read_only_registry(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)


def assert_accepted_deep_registry(db_path: Path) -> None:
    """Fail unless the registry carries exactly the accepted deep producer pairs."""
    if not db_path.exists():
        raise FileNotFoundError(f"Registry does not exist: {db_path}")
    with _read_only_registry(db_path) as db:
        rows = db.execute(
            """
            SELECT t.config_name, t.training_hash, p.prediction_hash
            FROM training_runs t
            JOIN prediction_sets p ON p.training_hash = t.training_hash
            WHERE t.family = 'deep_learning' AND p.split = 'validation'
            """
        ).fetchall()
    observed = {
        config: (training_hash, prediction_hash) for config, training_hash, prediction_hash in rows
    }
    if observed != ACCEPTED_DEEP_PRODUCERS:
        raise RuntimeError(
            "Registry deep-producer identity mismatch: "
            f"expected {ACCEPTED_DEEP_PRODUCERS!r}, observed {observed!r}"
        )


def assert_accepted_deep_baselines(db_path: Path) -> None:
    """Require baseline backtests for both accepted deep predictions."""
    assert_accepted_deep_registry(db_path)
    prediction_hashes = [pair[1] for pair in ACCEPTED_DEEP_PRODUCERS.values()]
    placeholders = ",".join("?" for _ in prediction_hashes)
    with _read_only_registry(db_path) as db:
        rows = db.execute(
            f"""
            SELECT prediction_hash, COUNT(*)
            FROM backtest_runs
            WHERE stage = 'signal' AND prediction_hash IN ({placeholders})
            GROUP BY prediction_hash
            """,
            prediction_hashes,
        ).fetchall()
    counts = dict(rows)
    missing = [
        prediction_hash
        for prediction_hash in prediction_hashes
        if counts.get(prediction_hash, 0) == 0
    ]
    if missing:
        raise RuntimeError(
            "Accepted deep predictions have no equal-weight baseline backtests: "
            f"{missing}. Rebuild notebook 12 before resolving a carrier or running notebook 16."
        )


def assert_complete_baseline_surface(
    db_path: Path,
    *,
    expected_predictions: int = 57,
    top_ks: tuple[int, ...] = (5, 10, 20),
) -> None:
    """Require the complete prediction x top-k x universe baseline grid."""
    with _read_only_registry(db_path) as db:
        rows = db.execute(
            """
            SELECT b.backtest_hash, b.prediction_hash, b.spec_json,
                   m.sharpe, m.cagr, m.max_drawdown
            FROM backtest_runs b
            JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
            JOIN training_runs t ON t.training_hash = p.training_hash
            LEFT JOIN backtest_metrics m ON m.backtest_hash = b.backtest_hash
            WHERE b.stage = 'signal' AND p.split = 'validation'
              AND t.label = 'ret_to_expiry'
            """
        ).fetchall()

    expected_universes = set(SP500_OPTIONS_EXECUTION_UNIVERSES)
    prediction_hashes = {row[1] for row in rows}
    expected_keys = {
        (prediction_hash, top_k, universe)
        for prediction_hash in prediction_hashes
        for top_k in top_ks
        for universe in expected_universes
    }
    observed_keys: list[tuple[str, int | None, str | None]] = []
    null_metric_hashes: list[str] = []
    for backtest_hash, prediction_hash, spec_json, sharpe, cagr, max_drawdown in rows:
        spec = json.loads(spec_json)
        signal = spec.get("strategy", {}).get("signal", {})
        observed_keys.append((prediction_hash, signal.get("top_k"), signal.get("universe_filter")))
        if any(value is None for value in (sharpe, cagr, max_drawdown)):
            null_metric_hashes.append(backtest_hash)

    observed_set = set(observed_keys)
    if len(prediction_hashes) != expected_predictions:
        raise RuntimeError(
            f"Baseline prediction count is {len(prediction_hashes)}, expected {expected_predictions}"
        )
    if len(observed_keys) != len(observed_set):
        raise RuntimeError("Baseline surface contains duplicate prediction/top-k/universe rows")
    if observed_set != expected_keys:
        missing = sorted(expected_keys - observed_set)
        extra = sorted(observed_set - expected_keys)
        raise RuntimeError(
            f"Baseline surface is incomplete: missing={missing[:5]}, extra={extra[:5]}"
        )
    if null_metric_hashes:
        raise RuntimeError(f"Baseline surface has null metrics: {null_metric_hashes[:5]}")


def assert_complete_allocation_surface(
    db_path: Path,
    *,
    prediction_hashes: set[str],
    top_ks: tuple[int, ...],
    allocators: set[str],
) -> None:
    """Require exactly the canonical liquid allocation Cartesian product."""
    with _read_only_registry(db_path) as db:
        rows = db.execute(
            """
            SELECT b.backtest_hash, b.prediction_hash, b.spec_json,
                   m.sharpe, m.cagr, m.max_drawdown
            FROM backtest_runs b
            LEFT JOIN backtest_metrics m ON m.backtest_hash = b.backtest_hash
            WHERE b.stage = 'allocation'
            """
        ).fetchall()

    expected = {
        (prediction_hash, top_k, allocator, "liquid")
        for prediction_hash in prediction_hashes
        for top_k in top_ks
        for allocator in allocators
    }
    observed: list[tuple[str, int | None, str | None, str | None]] = []
    null_metric_hashes: list[str] = []
    for backtest_hash, prediction_hash, spec_json, sharpe, cagr, max_drawdown in rows:
        spec = json.loads(spec_json)
        strategy = spec.get("strategy", {})
        signal = strategy.get("signal", {})
        allocation = strategy.get("allocation", {})
        observed.append(
            (
                prediction_hash,
                allocation.get("top_k"),
                allocation.get("method"),
                signal.get("universe_filter"),
            )
        )
        if any(value is None for value in (sharpe, cagr, max_drawdown)):
            null_metric_hashes.append(backtest_hash)

    observed_set = set(observed)
    if len(observed) != len(observed_set):
        raise RuntimeError("Allocation surface contains duplicate semantic rows")
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        extra = sorted(observed_set - expected)
        raise RuntimeError(
            f"Allocation surface does not match the canonical grid: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if null_metric_hashes:
        raise RuntimeError(f"Allocation surface has null metrics: {null_metric_hashes[:5]}")
