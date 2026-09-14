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


# The fixtures that ship in this state today. Every one of these populations lists prediction
# identities that no backtest in the same registry references, so
# `selectable_validation_candidates` removes the whole ranked field and
# `resolve_canonical_rank1_lineage` refuses - which is what the sixteen
# `no_canonical_selection` skip declarations in `tests/overrides.yaml` name
# (ml4t/agent-workspace#907). Four of the sixteen are etfs' and cme_futures' holdout stages.
#
# Not their strategy-analysis stages, which is worth saying because #907's title implies it:
# `cme_futures/19_strategy_analysis` is not skipped and passes, and `etfs/20_strategy_analysis`
# is held by a `fixture_shortfall` blocker on the complete-run filter, which publication has
# nothing to do with.
#
# The detector above is unit-tested and `generate_intermediates.py` prints for it, but only
# for a population THIS run added: a committed one is subtracted on every run after the one
# that created it, so nothing says the shipped fixture is in the state. That is how eight of
# the nine arrived here, and it is what this ratchet answers. The assertion is exact, so a
# fixture that gains a coherent population retires its own line and a ninth cannot join
# silently.
#
# These identities cannot be repaired by re-sampling production. They were written by a
# `--through-stage 8` regeneration running the model stages at fixture scale, so they exist
# in no production registry - checked 2026-09-13 against etfs, 0 of 12 present. Either the
# backtest stages run into the fixture, or the populations are not committed.
#
# The test below is named in `test-unit-data`'s file list in `.github/workflows/test.yml`,
# for the reason that job's own comment gives about `tests/test_skip_blockers.py`: it
# measures a property of the fixture, and `test-unit` checks out no test data, so there it
# skips and asserts nothing. The rest of this file is synthetic and belongs in the sweep.
# That job also fails on a skip, which is what stops this one going quiet again.
FIXTURES_PUBLISHING_UNBACKTESTED_POPULATIONS = {
    "cme_futures",
    "crypto_perps_funding",
    "etfs",
    "fx_pairs",
    "nasdaq100_microstructure",
    "sp500_options",
    "us_equities_panel",
    "us_firm_characteristics",
}


def test_the_shipped_fixtures_publish_what_they_are_known_to(intermediates_dir):
    if intermediates_dir is None:
        pytest.skip("no test-data intermediates on this checkout")
    unbacked = {
        case_dir.name
        for case_dir in sorted(intermediates_dir.iterdir())
        if (case_dir / "run_log" / "registry.db").is_file() and unbacktested_populations(case_dir)
    }

    assert unbacked == FIXTURES_PUBLISHING_UNBACKTESTED_POPULATIONS
