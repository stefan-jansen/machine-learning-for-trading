"""Regression tests for coherent case-study prediction fixtures."""

import sqlite3
from datetime import date

import polars as pl

from tests.fixtures.seed_results import _backfill_all_prediction_parquets


def test_crypto_prediction_hashes_share_keys_and_targets(tmp_path):
    cs_dir = tmp_path / "crypto_perps_funding"
    run_log = cs_dir / "run_log"
    run_log.mkdir(parents=True)
    # Registered with training_hash/split, so the shared-panel invariant is checked
    # on the same window-resolution path production takes. A registry carrying only
    # prediction_hash makes the join raise and exercises the degrade path instead,
    # which is not where the invariant has to hold.
    with sqlite3.connect(run_log / "registry.db") as connection:
        connection.execute(
            "CREATE TABLE prediction_sets "
            "(prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT)"
        )
        connection.execute(
            "CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT)"
        )
        connection.execute("INSERT INTO training_runs VALUES ('train_a', 'funding_next_8h')")
        connection.executemany(
            "INSERT INTO prediction_sets VALUES (?, ?, ?)",
            [("hash_a", "train_a", "validation"), ("hash_b", "train_a", "validation")],
        )

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


def test_seeded_predictions_stay_inside_the_split_they_are_registered_under(tmp_path, monkeypatch):
    """A ``validation`` hash must not be handed decisions from the holdout period.

    The generator once derived every artifact's date grid from ``holdout_start``,
    so rows registered as validation carried timestamps months past the sealed
    boundary. Notebooks that enforce that boundary then rejected the carrier -
    a fixture defect that reads as a pipeline failure.
    """
    from case_studies.utils import cv_window

    windows = {
        "validation": (date(2019, 1, 7), date(2020, 12, 23)),
        "holdout": (date(2021, 1, 1), date(2021, 12, 31)),
    }
    monkeypatch.setattr(
        cv_window, "canonical_window", lambda cs, label, split: windows.get(split), raising=True
    )

    cs_dir = tmp_path / "cme_futures"
    run_log = cs_dir / "run_log"
    run_log.mkdir(parents=True)
    with sqlite3.connect(run_log / "registry.db") as connection:
        connection.execute(
            "CREATE TABLE prediction_sets "
            "(prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT)"
        )
        connection.execute(
            "CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT)"
        )
        connection.executemany(
            "INSERT INTO training_runs VALUES (?, ?)", [("train_v", "fwd_ret_5d")]
        )
        connection.executemany(
            "INSERT INTO prediction_sets VALUES (?, ?, ?)",
            [("hash_val", "train_v", "validation"), ("hash_ho", "train_v", "holdout")],
        )

    _backfill_all_prediction_parquets(cs_dir, "cme_futures")

    for hash_value, split in [("hash_val", "validation"), ("hash_ho", "holdout")]:
        frame = pl.read_parquet(run_log / "predictions" / hash_value / "predictions.parquet")
        low, high = windows[split]
        assert frame["timestamp"].min() >= low, f"{hash_value} starts before its {split} window"
        assert frame["timestamp"].max() <= high, f"{hash_value} runs past its {split} window"


def test_an_underivable_window_still_respects_the_split_boundary(tmp_path):
    """The fallback is where the boundary is easiest to lose and hardest to notice.

    A NULL label, an absent label parquet and an older registry schema all reach it
    in ordinary seeding, so a single holdout-relative range shared by both splits
    would reintroduce the defect silently on every one of those paths. Uses the real
    ``canonical_window``, which cannot resolve a label that does not exist.
    """
    cs_dir = tmp_path / "cme_futures"
    run_log = cs_dir / "run_log"
    run_log.mkdir(parents=True)
    with sqlite3.connect(run_log / "registry.db") as connection:
        connection.execute(
            "CREATE TABLE prediction_sets "
            "(prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT)"
        )
        connection.execute(
            "CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT)"
        )
        connection.execute("INSERT INTO training_runs VALUES ('train_x', NULL)")
        connection.executemany(
            "INSERT INTO prediction_sets VALUES (?, ?, ?)",
            [("hash_val", "train_x", "validation"), ("hash_ho", "train_x", "holdout")],
        )

    _backfill_all_prediction_parquets(cs_dir, "cme_futures")

    # case_studies/cme_futures/config/setup.yaml::evaluation.holdout_start
    holdout_start = date(2024, 1, 1)
    validation = pl.read_parquet(run_log / "predictions" / "hash_val" / "predictions.parquet")
    holdout = pl.read_parquet(run_log / "predictions" / "hash_ho" / "predictions.parquet")

    assert validation["timestamp"].max() < holdout_start
    assert holdout["timestamp"].min() >= holdout_start
