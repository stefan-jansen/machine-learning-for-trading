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
from types import SimpleNamespace

import polars as pl
import pytest

from case_studies.utils import coverage as coverage_module
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

SESSIONS = [f"2024-{1 + day // 28:02d}-{1 + day % 28:02d}" for day in range(100)]
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
    def test_a_member_with_a_registry_row_reaches_the_coverage_check(
        self, case_dir: Path, monkeypatch
    ) -> None:
        """The regression for `t.case_study`: this raised OperationalError before.

        The assertion is that the row came back out of SQL and was carried to the coverage
        call with the label and split the join supplied. A query naming a column the schema
        lacks never gets that far.
        """
        calls: list[str] = []

        def fake(frame, case_study, label, *, split, **kwargs):
            calls.append(f"{case_study}/{label}/{split}")
            return SimpleNamespace(accountable_coverage=1.0, summary=lambda: "")

        monkeypatch.setattr(coverage_module, "check_prediction_cross_section", fake)
        _register(case_dir, WHOLE, delivered=200, declared=200)
        assert undercovered_prediction_members(case_dir, [WHOLE], case_study="x") == {}
        assert calls == ["x/fwd_ret_5d/validation"]

    def test_nothing_to_check_is_not_evidence_of_anything(self, case_dir: Path) -> None:
        assert undercovered_prediction_members(case_dir, [], case_study="x") == {}


class TestAMemberIsChargedAgainstTheFeaturePanel:
    """Not against its own `expected_prediction_keys`, which is where rows get dropped.

    Measured on sp500_equity_option_analytics/fwd_ret_10d/validation: `deep_learning`
    carries 65.0% of the panel and never scores 262 of 548 symbols, while `linear`, `gbm`
    and `tabular_dl` carry 100.0% and miss one. It delivers every key it declared, because
    the declaration is where the 262 symbols went. `notebook_contracts` carries the whole
    table; these stand in for it with a stand-in coverage report, because the real one
    needs a case study's fold boundaries a tmp registry has no way to supply.

    `check_prediction_cross_section` is imported inside the function to break an import
    cycle, so the stand-in goes on `case_studies.utils.coverage`, where the import reads
    it, rather than on the calling module.
    """

    @staticmethod
    def _standin(monkeypatch, *, coverage: float | None):
        """Replace the coverage call with one that reports `coverage`, or raises."""
        calls: list[str] = []

        def fake(frame, case_study, label, **kwargs):
            calls.append(f"{case_study}/{label}")
            if coverage is None:
                raise coverage_module.CoverageError("no label artifact")
            return SimpleNamespace(
                accountable_coverage=coverage, summary=lambda: f"stand-in at {coverage}"
            )

        monkeypatch.setattr(coverage_module, "check_prediction_cross_section", fake)
        return calls

    def test_a_member_below_the_minimum_is_reported(self, case_dir, monkeypatch) -> None:
        """65% of the panel is the deep_learning case, and it is dropped."""
        calls = self._standin(monkeypatch, coverage=0.65)
        _register(case_dir, SHORT, delivered=100, declared=100)
        reported = undercovered_prediction_members(case_dir, [SHORT], case_study="x")
        assert calls == ["x/fwd_ret_5d"]
        assert reported == {SHORT: "stand-in at 0.65"}

    def test_delivering_everything_it_declared_does_not_excuse_it(
        self, case_dir, monkeypatch
    ) -> None:
        """The member here declared 100 rows and delivered all 100, and is still short.

        This is the assertion that pins the denominator. A rule that let a run's own
        declaration answer the question would pass this member, and would pass every
        deep-learning member of sp500_equity_option_analytics with it.
        """
        self._standin(monkeypatch, coverage=0.65)
        _register(case_dir, SHORT, delivered=100, declared=100)
        assert set(undercovered_prediction_members(case_dir, [SHORT], case_study="x")) == {SHORT}

    def test_a_member_above_the_minimum_is_not_reported(self, case_dir, monkeypatch) -> None:
        self._standin(monkeypatch, coverage=0.99)
        _register(case_dir, WHOLE, delivered=200, declared=200)
        assert undercovered_prediction_members(case_dir, [WHOLE], case_study="x") == {}

    def test_the_threshold_is_the_caller_s_to_set(self, case_dir, monkeypatch) -> None:
        """Without this the reported set could come from anywhere."""
        self._standin(monkeypatch, coverage=0.5)
        _register(case_dir, SHORT, delivered=100, declared=100)
        assert undercovered_prediction_members(case_dir, [SHORT], case_study="x", minimum=0.4) == {}
        assert set(
            undercovered_prediction_members(case_dir, [SHORT], case_study="x", minimum=0.6)
        ) == {SHORT}

    def test_a_member_that_cannot_be_evaluated_is_short_rather_than_passed(
        self, case_dir, monkeypatch
    ) -> None:
        self._standin(monkeypatch, coverage=None)
        _register(case_dir, UNDECLARED, delivered=100, declared=None)
        reported = undercovered_prediction_members(case_dir, [UNDECLARED], case_study="x")
        assert "coverage could not be evaluated" in reported[UNDECLARED]
