"""Regression tests for coherent case-study prediction fixtures."""

import sqlite3
from datetime import date

import polars as pl

from tests.fixtures.seed_results import _backfill_all_prediction_parquets


def test_crypto_prediction_hashes_share_keys_and_targets(tmp_path):
    cs_dir = tmp_path / "crypto_perps_funding"
    run_log = cs_dir / "run_log"
    run_log.mkdir(parents=True)
    with sqlite3.connect(run_log / "registry.db") as connection:
        connection.execute("CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO prediction_sets VALUES (?)", [("hash_a",), ("hash_b",)])

    stale_dir = run_log / "predictions" / "hash_a"
    stale_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["STALE"],
            "timestamp": [date(2000, 1, 1)],
            "fold": [0],
            "prediction": [0.0],
            "actual": [1.0],
        }
    ).write_parquet(stale_dir / "predictions.parquet")

    _backfill_all_prediction_parquets(cs_dir, "crypto_perps_funding")

    frames = [
        pl.read_parquet(run_log / "predictions" / value / "predictions.parquet")
        for value in ["hash_a", "hash_b"]
    ]
    assert (
        frames[0]
        .select("timestamp", "symbol", "actual")
        .equals(frames[1].select("timestamp", "symbol", "actual"))
    )
    assert not frames[0]["prediction"].equals(frames[1]["prediction"])
    assert frames[0].height > 1
