"""A retired generation is the one no current official population still holds.

Refitting under a corrected input artifact produces new prediction identities and
publishes a snapshot that supersedes the previous one. The retired rows stay in
``training_runs`` and ``prediction_sets``, because the record of what was replaced is
evidence; only ``official_populations`` says which generation is in force. A reader
that joins the result tables directly therefore sees two rows where the study has one,
and cannot separate them on the numbers, because a refit that changes only a declared
input leaves every metric bit-identical.
"""

import json
import sqlite3

from case_studies.research.population import retired_prediction_hashes


def _registry(rows):
    """An in-memory registry holding one official_populations row per entry."""
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE official_populations (population_hash TEXT, supersedes_hash TEXT, "
        "member_kind TEXT, snapshot_json TEXT)"
    )
    for population_hash, supersedes, kind, members in rows:
        db.execute(
            "INSERT INTO official_populations VALUES (?, ?, ?, ?)",
            (population_hash, supersedes, kind, json.dumps({"members": list(members)})),
        )
    return db


def test_a_registry_with_no_populations_table_retires_nothing():
    # A clean clone: run_log/ is gitignored, so a reader starts with no table at all.
    db = sqlite3.connect(":memory:")
    assert retired_prediction_hashes(db) == set()


def test_a_single_generation_retires_nothing():
    db = _registry([("gen1", None, "prediction", ["a", "b"])])
    assert retired_prediction_hashes(db) == set()


def test_a_superseded_generations_members_are_retired():
    db = _registry(
        [
            ("gen1", None, "prediction", ["old_a", "old_b"]),
            ("gen2", "gen1", "prediction", ["new_a", "new_b"]),
        ]
    )
    assert retired_prediction_hashes(db) == {"old_a", "old_b"}


def test_a_member_carried_into_the_new_generation_is_not_retired():
    # A refit moves some identities and leaves others alone; the ones the new snapshot
    # still holds are live, whatever the old snapshot said.
    db = _registry(
        [
            ("gen1", None, "prediction", ["kept", "moved_old"]),
            ("gen2", "gen1", "prediction", ["kept", "moved_new"]),
        ]
    )
    assert retired_prediction_hashes(db) == {"moved_old"}


def test_a_hash_in_no_population_is_not_retired():
    # Nothing has declared it superseded, and a registry may hold results a study
    # never published under a name.
    db = _registry([("gen1", None, "prediction", ["published"])])
    assert "unpublished" not in retired_prediction_hashes(db)


def test_a_three_generation_chain_retires_all_but_the_tip():
    db = _registry(
        [
            ("gen1", None, "prediction", ["a1"]),
            ("gen2", "gen1", "prediction", ["a2"]),
            ("gen3", "gen2", "prediction", ["a3"]),
        ]
    )
    assert retired_prediction_hashes(db) == {"a1", "a2"}


def test_backtest_populations_do_not_retire_prediction_hashes():
    # The member kinds share a table and a name space; a superseded backtest snapshot
    # says nothing about a prediction identity.
    db = _registry(
        [
            ("bt1", None, "backtest", ["shared"]),
            ("bt2", "bt1", "backtest", ["other"]),
            ("pred1", None, "prediction", ["shared"]),
        ]
    )
    assert retired_prediction_hashes(db) == set()


def test_two_independent_current_generations_both_stay_live():
    # Two names, each with its own chain: a supersession under one must not retire the
    # other's members.
    db = _registry(
        [
            ("a1", None, "prediction", ["a_old"]),
            ("a2", "a1", "prediction", ["a_new"]),
            ("b1", None, "prediction", ["b_only"]),
        ]
    )
    retired = retired_prediction_hashes(db)
    assert retired == {"a_old"}
    assert "b_only" not in retired
