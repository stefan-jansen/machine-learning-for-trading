"""Regression tests for coherent case-study prediction fixtures."""

import sqlite3
from datetime import date

import polars as pl
import pytest

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


def test_a_seeded_hash_joins_the_copied_artifact_it_shares_a_label_with(tmp_path):
    """14/09_case_study_insights pairs a latent and a supervised prediction set of one
    case study on their common (timestamp, entity) keys. Seeding the latent one onto a
    fabricated grid of placeholder symbols while the supervised one is an artifact
    copied from production leaves that join empty, which the notebook reported as
    "Aligned targets disagree: maximum gap None" - max() over no rows.
    """
    cs_dir = tmp_path / "us_equities_panel"
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
        connection.execute("INSERT INTO training_runs VALUES ('train_a', 'fwd_ret_1d')")
        connection.executemany(
            "INSERT INTO prediction_sets VALUES (?, ?, ?)",
            [("hash_copied", "train_a", "validation"), ("hash_seeded", "train_a", "validation")],
        )

    copied_dir = run_log / "predictions" / "hash_copied"
    copied_dir.mkdir(parents=True)
    days = [date(2015, 1, 5 + offset) for offset in range(5)]
    copied = pl.DataFrame(
        {
            "symbol": [s for _ in days for s in ("AAPL", "MSFT")],
            "timestamp": [d for d in days for _ in range(2)],
            "fold": [0] * 10,
            "prediction": [0.01 * i for i in range(10)],
            "actual": [0.002 * i for i in range(10)],
        }
    )
    copied.write_parquet(copied_dir / "predictions.parquet")

    _backfill_all_prediction_parquets(cs_dir, "us_equities_panel")

    seeded = pl.read_parquet(run_log / "predictions" / "hash_seeded" / "predictions.parquet")
    joined = seeded.join(copied, on=["timestamp", "symbol"], how="inner")
    assert joined.height == seeded.height, "the seeded set shares no keys with the copied one"
    gap = joined.select((pl.col("actual") - pl.col("actual_right")).abs().max()).item()
    assert gap == 0.0, "the two sets disagree on the realized target they are scored against"
    assert not seeded["prediction"].equals(copied["prediction"]), (
        "the seeded set copied the scores too, so it is not an independent configuration"
    )


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


# --- Cohort-leader handling -------------------------------------------------
#
# A label's cohort leader is the frozen carrier a strategy-analysis/portfolio/cost/
# risk notebook resolves by hash and checks against real historical values, so it is
# the one prediction the synthetic rewrite must not touch. All three tests use
# crypto_perps_funding because it is the only case study with rewrite_existing set;
# everywhere else an existing artifact is already left alone and the exemption is
# unobservable.

LEADER_PRED = "hash_leader"
LEADER_BT = "bt_leader_signal"

_COHORT_DDL = (
    "CREATE TABLE cohort_metrics (cohort_type TEXT, label TEXT, stage TEXT, leader_hash TEXT)"
)
_COHORT_DDL_NO_STAGE = (
    "CREATE TABLE cohort_metrics (cohort_type TEXT, label TEXT, leader_hash TEXT)"
)


def _crypto_registry(run_log, cohort_ddl):
    """Registry with one leader prediction and one ordinary one."""
    run_log.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(run_log / "registry.db") as connection:
        connection.execute(
            "CREATE TABLE prediction_sets "
            "(prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT)"
        )
        connection.execute(
            "CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT)"
        )
        connection.execute(
            "CREATE TABLE backtest_runs "
            "(backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT)"
        )
        connection.execute("INSERT INTO training_runs VALUES ('train_a', 'funding_next_8h')")
        connection.executemany(
            "INSERT INTO prediction_sets VALUES (?, ?, ?)",
            [(LEADER_PRED, "train_a", "validation"), ("hash_other", "train_a", "validation")],
        )
        connection.execute(
            "INSERT INTO backtest_runs VALUES (?, ?, 'signal')", (LEADER_BT, LEADER_PRED)
        )
        if cohort_ddl:
            connection.execute(cohort_ddl)
            if "stage TEXT" in cohort_ddl:
                connection.execute(
                    "INSERT INTO cohort_metrics VALUES ('stagelabel', 'funding_next_8h', "
                    "'signal', ?)",
                    (LEADER_BT,),
                )
            else:
                connection.execute(
                    "INSERT INTO cohort_metrics VALUES ('stagelabel', 'funding_next_8h', ?)",
                    (LEADER_BT,),
                )


def _write_carrier(run_log, prediction_hash, marker):
    path = run_log / "predictions" / prediction_hash
    path.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "symbol": [marker, marker],
            "timestamp": [date(2020, 1, 1), date(2020, 1, 2)],
            "fold": [0, 0],
            "prediction": [0.5, 0.25],
            "actual": [0.4, 0.2],
        }
    )
    frame.write_parquet(path / "predictions.parquet")
    return frame


def test_the_real_carrier_artifact_survives_the_crypto_rewrite(tmp_path):
    """crypto normalizes every prediction onto one key/target panel. The carrier is
    exempt: a notebook pins it by hash and correlates its target against real raw
    prices, which synthetic noise cannot pass."""
    cs_dir = tmp_path / "crypto_perps_funding"
    run_log = cs_dir / "run_log"
    _crypto_registry(run_log, _COHORT_DDL)
    leader_before = _write_carrier(run_log, LEADER_PRED, "REALCARRIER")
    other_before = _write_carrier(run_log, "hash_other", "STALE")

    _backfill_all_prediction_parquets(cs_dir, "crypto_perps_funding")

    leader_after = pl.read_parquet(run_log / "predictions" / LEADER_PRED / "predictions.parquet")
    other_after = pl.read_parquet(run_log / "predictions" / "hash_other" / "predictions.parquet")
    assert leader_after.equals(leader_before), "the cohort leader's real artifact was overwritten"
    assert not other_after.equals(other_before), (
        "hash_other was left alone, so the rewrite this exemption carves out of did not run "
        "and the assertion above passes for the wrong reason"
    )


def test_a_leader_with_no_artifact_is_named_not_silently_synthesized(tmp_path):
    """Nothing here can reconstruct the artifact, so the gap is reported by hash at
    regeneration time rather than left to surface as a correlation failure several
    notebooks downstream."""
    cs_dir = tmp_path / "crypto_perps_funding"
    run_log = cs_dir / "run_log"
    _crypto_registry(run_log, _COHORT_DDL)
    _write_carrier(run_log, "hash_other", "STALE")

    with pytest.warns(RuntimeWarning, match=LEADER_PRED):
        _backfill_all_prediction_parquets(cs_dir, "crypto_perps_funding")


def test_a_cohort_metrics_table_missing_stage_is_not_swallowed(tmp_path):
    """An emptied leader set is the dangerous outcome, not a missing one: every real
    carrier artifact then goes back through the synthetic rewrite, reported as success.
    """
    cs_dir = tmp_path / "crypto_perps_funding"
    run_log = cs_dir / "run_log"
    _crypto_registry(run_log, _COHORT_DDL_NO_STAGE)
    _write_carrier(run_log, LEADER_PRED, "REALCARRIER")

    with pytest.raises(sqlite3.OperationalError, match="stage"):
        _backfill_all_prediction_parquets(cs_dir, "crypto_perps_funding")


def test_a_registry_without_cohort_metrics_still_seeds(tmp_path):
    """The one condition the existence guard tolerates."""
    cs_dir = tmp_path / "crypto_perps_funding"
    run_log = cs_dir / "run_log"
    _crypto_registry(run_log, None)

    _backfill_all_prediction_parquets(cs_dir, "crypto_perps_funding")

    for prediction_hash in (LEADER_PRED, "hash_other"):
        assert (run_log / "predictions" / prediction_hash / "predictions.parquet").is_file()
