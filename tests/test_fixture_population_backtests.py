"""A fixture that publishes a population with no backtested member says so.

`cs-sp500_equity_option_analytics` went red on a `ZeroDivisionError` fifteen minutes
after its last green run, because a stage-08 regeneration of the fixture created two
populations over 30 fresh prediction sets while every backtest in the fixture referenced
predictions from two weeks earlier. `14_backtest` scopes its baseline ranking to the
members in force, got an empty frame, and divided by its height
(ml4t/agent-workspace#1086).

Public #849 made the notebook refuse and say what is wrong. This is the other half:
the generation that produces the state reports it, so the fixture is not committed with
nobody having been told. It cannot avoid producing it - the model stages declare the
population and the backtest stages are numbers 14 and up, which `--through-stage 8`
never reaches - which is exactly why saying so is what it owes.
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.fixture_registry import unbacktested_populations

SCHEMA = """
CREATE TABLE official_populations (population_hash TEXT PRIMARY KEY, name TEXT);
CREATE TABLE official_population_members (population_hash TEXT, member_hash TEXT, ordinal INTEGER);
CREATE TABLE backtest_runs (backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT);
"""


@pytest.fixture
def case_dir(tmp_path):
    (tmp_path / "run_log").mkdir()
    db = sqlite3.connect(str(tmp_path / "run_log" / "registry.db"))
    db.executescript(SCHEMA)
    db.commit()
    db.close()
    return tmp_path


def _connect(case_dir):
    return sqlite3.connect(str(case_dir / "run_log" / "registry.db"))


def test_a_population_whose_members_have_no_signal_backtest_is_reported(case_dir):
    db = _connect(case_dir)
    db.execute("INSERT INTO official_populations VALUES ('pop','seoa-gbm-preview')")
    db.executemany(
        "INSERT INTO official_population_members VALUES ('pop',?,?)",
        [("p1", 0), ("p2", 1)],
    )
    db.commit()
    db.close()
    assert unbacktested_populations(case_dir) == [
        {"name": "seoa-gbm-preview", "population_hash": "pop", "members": 2}
    ]


def test_a_population_with_one_backtested_member_is_not_reported(case_dir):
    """One is enough: the ranking has a denominator, which is what the read needs."""
    db = _connect(case_dir)
    db.execute("INSERT INTO official_populations VALUES ('pop','seoa-gbm-preview')")
    db.executemany(
        "INSERT INTO official_population_members VALUES ('pop',?,?)", [("p1", 0), ("p2", 1)]
    )
    db.execute("INSERT INTO backtest_runs VALUES ('b','p1','signal')")
    db.commit()
    db.close()
    assert unbacktested_populations(case_dir) == []


def test_a_backtest_at_another_stage_does_not_count(case_dir):
    """`14_backtest` reads `stage='signal'`. An allocation row is not one."""
    db = _connect(case_dir)
    db.execute("INSERT INTO official_populations VALUES ('pop','seoa-gbm-preview')")
    db.execute("INSERT INTO official_population_members VALUES ('pop','p1',0)")
    db.execute("INSERT INTO backtest_runs VALUES ('b','p1','allocation')")
    db.commit()
    db.close()
    assert [p["name"] for p in unbacktested_populations(case_dir)] == ["seoa-gbm-preview"]


def test_a_backtest_against_a_prediction_outside_the_population_does_not_count(case_dir):
    """The exact shape #1086 arrived in: 52 backtests, none of them on a member."""
    db = _connect(case_dir)
    db.execute("INSERT INTO official_populations VALUES ('pop','seoa-gbm-preview')")
    db.execute("INSERT INTO official_population_members VALUES ('pop','p_new',0)")
    db.execute("INSERT INTO backtest_runs VALUES ('b','p_old','signal')")
    db.commit()
    db.close()
    assert [p["members"] for p in unbacktested_populations(case_dir)] == [1]


def test_a_population_with_no_members_is_not_reported(case_dir):
    db = _connect(case_dir)
    db.execute("INSERT INTO official_populations VALUES ('pop','empty')")
    db.commit()
    db.close()
    assert unbacktested_populations(case_dir) == []


def test_a_case_study_with_no_registry_reports_nothing(tmp_path):
    assert unbacktested_populations(tmp_path) == []
