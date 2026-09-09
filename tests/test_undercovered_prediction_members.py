"""The check `13_model_analysis` makes before it ranks the members in force.

`incompletely_registered_predictions` next door asks whether a member finished. This asks
a different question: of the keys the member's inputs let it score, how many did it
deliver? A family that scores every decision date for half the universe finishes, and ties
the maximum on `ic_n_days`, and is a different experiment from the peer it is ranked
against.

Two things this file pins, both of which shipped broken because nothing exercised them:

- the SQL runs at all. It selected `t.case_study`, which `training_runs` does not have, so
  every call with a member to check raised `sqlite3.OperationalError` and every call with
  nothing to check returned `{}` - the shape that reads as "checked, all fine".
- a member is charged against what its own run declared scoreable, not against the raw
  feature panel. Measured on sp500_equity_option_analytics: 143 of 947 members in force
  carry 65% of the panel's keys and 100% of their own, because a sequence model cannot
  score a window shorter than its lookback. Charging them the panel drops the whole
  deep-learning half of the pool for the window builder working as specified.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.notebook_contracts import undercovered_prediction_members

WHOLE = "aaaa11112222"
SHORT = "bbbb33334444"
UNDECLARED = "cccc55556666"

#: The columns `training_runs` actually has. Named here so a query that invents one fails.
TRAINING_RUNS_COLUMNS = (
    "training_hash TEXT PRIMARY KEY, family TEXT, label TEXT, config_name TEXT, "
    "spec_json TEXT, created_at TEXT, git_commit TEXT, entry_point TEXT, "
    "started_at TEXT, elapsed_s REAL, runtime_json TEXT, identity_version INTEGER, "
    "execution_tier TEXT"
)

SESSIONS = [f"2024-01-{day:02d}" for day in range(1, 21)]
ENTITIES = ["AAA", "BBB"]


def _panel_rows() -> list[tuple[str, str]]:
    return [(entity, session) for entity in ENTITIES for session in SESSIONS]


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    (case_dir / "run_log").mkdir(parents=True)
    (case_dir / "features").mkdir()
    (case_dir / "labels").mkdir()

    panel = pl.DataFrame(
        {
            "symbol": [entity for entity, _ in _panel_rows()],
            "timestamp": [session for _, session in _panel_rows()],
            "feature_a": [1.0] * len(_panel_rows()),
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    panel.write_parquet(case_dir / "features" / "financial.parquet")

    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(f"CREATE TABLE training_runs ({TRAINING_RUNS_COLUMNS})")
        db.execute(
            "CREATE TABLE prediction_sets "
            "(prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT)"
        )
    return case_dir


def _register(
    case_dir: Path,
    member: str,
    *,
    delivered: int,
    declared: int | None,
) -> None:
    """Write one member with ``delivered`` prediction rows and a declared expectation."""
    spec = {"computation": {}}
    if declared is not None:
        spec["computation"]["expected_prediction_keys"] = {"n_rows": declared}
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO training_runs (training_hash, family, label, config_name, spec_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"t-{member}", "deep_learning", "fwd_ret_5d", "lstm_h64", json.dumps(spec)),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?, ?, ?)",
            (member, f"t-{member}", "validation"),
        )

    rows = _panel_rows()[:delivered]
    path = case_dir / "run_log" / "predictions" / member
    path.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": [entity for entity, _ in rows],
            "timestamp": [session for _, session in rows],
            "y_score": [0.1] * len(rows),
            "fold": [0] * len(rows),
        }
    ).with_columns(pl.col("timestamp").str.to_date()).write_parquet(path / "predictions.parquet")


class TestTheQueryRunsAgainstTheRealSchema:
    def test_a_member_with_a_registry_row_is_reachable_at_all(self, case_dir: Path) -> None:
        """The regression for `t.case_study`: this raised OperationalError before.

        Every other test here would also have raised, so the assertion is the call.
        """
        _register(case_dir, WHOLE, delivered=40, declared=40)
        assert undercovered_prediction_members(case_dir, [WHOLE], case_study="x") == {}

    def test_nothing_to_check_is_not_evidence_of_anything(self, case_dir: Path) -> None:
        assert undercovered_prediction_members(case_dir, [], case_study="x") == {}


class TestAMemberIsChargedAgainstItsOwnDeclaredExpectation:
    def test_delivering_every_declared_key_is_whole_however_narrow_the_set(
        self, case_dir: Path
    ) -> None:
        """A sequence family declares fewer keys than the panel and is not short for it."""
        _register(case_dir, WHOLE, delivered=20, declared=20)
        assert undercovered_prediction_members(case_dir, [WHOLE], case_study="x") == {}

    def test_delivering_less_than_it_declared_is_reported(self, case_dir: Path) -> None:
        _register(case_dir, SHORT, delivered=10, declared=40)
        reported = undercovered_prediction_members(case_dir, [SHORT], case_study="x")
        assert set(reported) == {SHORT}

    def test_a_run_that_declared_nothing_falls_back_to_the_panel(self, case_dir: Path) -> None:
        """No expectation recorded, so the panel is the only ceiling available."""
        _register(case_dir, UNDECLARED, delivered=10, declared=None)
        reported = undercovered_prediction_members(case_dir, [UNDECLARED], case_study="x")
        assert set(reported) == {UNDECLARED}
