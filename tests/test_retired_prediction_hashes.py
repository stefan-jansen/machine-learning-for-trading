"""A retired generation is one its own publishing name has moved past.

Refitting under a corrected input artifact produces new prediction identities and
publishes a snapshot that supersedes the previous one. The retired rows stay in
``training_runs`` and ``prediction_sets``, because the record of what was replaced is
evidence; only ``official_populations`` says which generation is in force. A reader
that joins the result tables directly therefore sees two rows where the study has one,
and cannot separate them on the numbers, because a refit that changes only a declared
input leaves every metric bit-identical.

The question is asked per name. The same prediction identity is legitimately listed
under several names - a family's own population, a case-wide one a later stage froze, a
narrowed run's own snapshot - and those snapshots are frozen at different times, so one
of them goes on listing a generation its own publisher has replaced. Deciding retirement
from the union instead keeps that generation rankable, which is the defect measured on
fx_pairs on 2026-10-16: the chapter-12 TabM cell for fx_pairs was the retired fit.
"""

import json
import sqlite3

import pytest

from case_studies.research.population import retired_prediction_hashes

FAMILY = "fx_pairs:fwd_ret_1d+fwd_ret_5d+fwd_ret_21d:tabular_dl"
CASE_WIDE = "fx_pairs:validation-predictions"


def _registry(rows):
    """An in-memory registry holding one official_populations row per entry.

    ``name`` is part of the real schema and part of the question, so a fixture without it
    is a shape the pipeline cannot produce.
    """
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE official_populations (population_hash TEXT, name TEXT NOT NULL, "
        "supersedes_hash TEXT, member_kind TEXT, snapshot_json TEXT)"
    )
    for population_hash, name, supersedes, kind, members in rows:
        db.execute(
            "INSERT INTO official_populations VALUES (?, ?, ?, ?, ?)",
            (population_hash, name, supersedes, kind, json.dumps({"members": list(members)})),
        )
    return db


def test_a_registry_with_no_populations_table_retires_nothing():
    # A clean clone: run_log/ is gitignored, so a reader starts with no table at all.
    db = sqlite3.connect(":memory:")
    assert retired_prediction_hashes(db) == set()


def test_a_single_generation_retires_nothing():
    db = _registry([("gen1", FAMILY, None, "prediction", ["a", "b"])])
    assert retired_prediction_hashes(db) == set()


def test_a_superseded_generations_members_are_retired():
    db = _registry(
        [
            ("gen1", FAMILY, None, "prediction", ["old_a", "old_b"]),
            ("gen2", FAMILY, "gen1", "prediction", ["new_a", "new_b"]),
        ]
    )
    assert retired_prediction_hashes(db) == {"old_a", "old_b"}


def test_a_member_carried_into_the_new_generation_is_not_retired():
    # A refit moves some identities and leaves others alone; the ones the new snapshot
    # still holds are live, whatever the old snapshot said.
    db = _registry(
        [
            ("gen1", FAMILY, None, "prediction", ["kept", "moved_old"]),
            ("gen2", FAMILY, "gen1", "prediction", ["kept", "moved_new"]),
        ]
    )
    assert retired_prediction_hashes(db) == {"moved_old"}


def test_a_hash_in_no_population_is_not_retired():
    # Nothing has declared it superseded, and a registry may hold results a study
    # never published under a name.
    db = _registry([("gen1", FAMILY, None, "prediction", ["published"])])
    assert "unpublished" not in retired_prediction_hashes(db)


def test_a_three_generation_chain_retires_all_but_the_tip():
    db = _registry(
        [
            ("gen1", FAMILY, None, "prediction", ["a1"]),
            ("gen2", FAMILY, "gen1", "prediction", ["a2"]),
            ("gen3", FAMILY, "gen2", "prediction", ["a3"]),
        ]
    )
    assert retired_prediction_hashes(db) == {"a1", "a2"}


def test_backtest_populations_do_not_retire_prediction_hashes():
    # The member kinds share a table and a name space; a superseded backtest snapshot
    # says nothing about a prediction identity.
    db = _registry(
        [
            ("bt1", "fx_pairs:backtests", None, "backtest", ["shared"]),
            ("bt2", "fx_pairs:backtests", "bt1", "backtest", ["other"]),
            ("pred1", FAMILY, None, "prediction", ["shared"]),
        ]
    )
    assert retired_prediction_hashes(db) == set()


def test_two_independent_current_generations_both_stay_live():
    # Two names, each with its own chain: a supersession under one must not retire the
    # other's members.
    db = _registry(
        [
            ("a1", FAMILY, None, "prediction", ["a_old"]),
            ("a2", FAMILY, "a1", "prediction", ["a_new"]),
            ("b1", CASE_WIDE, None, "prediction", ["b_only"]),
        ]
    )
    retired = retired_prediction_hashes(db)
    assert retired == {"a_old"}
    assert "b_only" not in retired


def test_a_live_snapshot_under_another_name_does_not_un_retire_a_refit():
    """The fx_pairs shape, and the one the global form got wrong.

    ``08_tabular_dl`` publishes the family's own population and advances it on a refit.
    ``13_backtest`` publishes a case-wide population over every family's predictions and
    was last run in September, so its live snapshot still lists the superseded TabM
    generation beside the other families' live sets. The refit is what retires those
    members; a snapshot nobody has re-run cannot keep them current.
    """
    db = _registry(
        [
            ("fam1", FAMILY, None, "prediction", ["tabm_old_1", "tabm_old_2"]),
            ("fam2", FAMILY, "fam1", "prediction", ["tabm_new_1", "tabm_new_2"]),
            ("wide1", CASE_WIDE, None, "prediction", ["tabm_old_1", "tabm_old_2", "gbm_live"]),
        ]
    )
    retired = retired_prediction_hashes(db)
    assert retired == {"tabm_old_1", "tabm_old_2"}
    # The other families' members in the same stale snapshot are untouched: no name has
    # moved past them.
    assert "gbm_live" not in retired


def test_a_case_wide_snapshot_that_was_re_run_carries_the_new_generation():
    """The crypto shape, which the same rule has to leave alone.

    There the family notebook advances the case-wide population too, so both names name
    the new generation and only the old one is retired. Nothing about the fix turns a
    re-run downstream stage into a source of retirement.
    """
    db = _registry(
        [
            ("fam1", FAMILY, None, "prediction", ["tabm_old"]),
            ("fam2", FAMILY, "fam1", "prediction", ["tabm_new"]),
            ("wide1", CASE_WIDE, None, "prediction", ["tabm_old", "gbm_live"]),
            ("wide2", CASE_WIDE, "wide1", "prediction", ["tabm_new", "gbm_live"]),
        ]
    )
    assert retired_prediction_hashes(db) == {"tabm_old"}


def test_a_lineage_the_reduction_cannot_read_raises_rather_than_retiring_nothing():
    """An unreadable lineage and an empty one are different answers.

    Catching every ``OperationalError`` read a schema mismatch as "nothing has been
    published", which is the shape every reader here is least able to detect: it ranks
    both generations and reports no error. The missing-table case is answered from
    ``sqlite_master`` instead, so only a real absence returns an empty set.
    """
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE official_populations (population_hash TEXT, supersedes_hash TEXT, "
        "member_kind TEXT, snapshot_json TEXT)"
    )
    db.execute(
        "INSERT INTO official_populations VALUES (?, ?, ?, ?)",
        ("gen1", None, "prediction", json.dumps({"members": ["a"]})),
    )
    with pytest.raises(sqlite3.OperationalError):
        retired_prediction_hashes(db)
