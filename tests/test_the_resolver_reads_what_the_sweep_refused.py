"""The sweep and the carrier resolver answer "admissible" from one measurement.

`prediction_members_in_force` charges every member against the feature panel it was
offered and drops the ones that fall short. `selectable_validation_candidates` could not
ask that question: `full_coverage_prediction_sql` counts decision days, and a family that
scores every day for half the universe ties the day count while ranking a narrower
cross-section. So the resolver was the looser of the two rules, and a prediction the sweep
refused to backtest could still carry the case study - measured on
nasdaq100_microstructure, where pinning the label made the carrier a 67.4%-coverage set
with an IC of -0.00125.

Recomputing the check inside the resolver would pay the sweep's startup cost once per
strategy-analysis notebook and leave two implementations agreeing by inspection, which is
the arrangement that produced the divergence. The sweep records what it measured; the
resolver reads it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import strategy_analysis
from case_studies.utils.notebook_contracts import (
    predictions_the_sweep_refused,
    record_prediction_admissibility,
)

CASE_STUDY = "fixture_case_study"
SHORTFALL = "covered 67.4% of the cross-section its feature panels offered"


def _registry(case_dir: Path, rows: list[tuple[str, str, float]]) -> None:
    """Each row is ``(backtest_hash, prediction_hash, sharpe)``, all selectable."""
    db_path = case_dir / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT,
                spec_json TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_kind TEXT, checkpoint_value INTEGER
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY, ic_mean REAL, ic_n_days REAL
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL, ic_std REAL);
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT, spec_json TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY, sharpe REAL, max_drawdown REAL
            );
            """
        )
        for backtest_hash, prediction_hash, sharpe in rows:
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT OR IGNORE INTO training_runs VALUES (?,'gbm',?, 'fwd_ret_5d', NULL)",
                (training_hash, f"config_{prediction_hash}"),
            )
            db.execute(
                "INSERT OR IGNORE INTO prediction_sets VALUES (?,?, 'validation', NULL, NULL)",
                (prediction_hash, training_hash),
            )
            db.execute(
                "INSERT OR IGNORE INTO prediction_metrics VALUES (?, 0.02, 250)",
                (prediction_hash,),
            )
            db.execute(
                "INSERT OR IGNORE INTO fold_metrics VALUES (?, 0.02, 0.3)", (prediction_hash,)
            )
            db.execute(
                "INSERT INTO backtest_runs VALUES (?,?, 'signal', ?)",
                (backtest_hash, prediction_hash, '{"strategy": {"signal": {}}}'),
            )
            db.execute("INSERT INTO backtest_metrics VALUES (?,?, -0.2)", (backtest_hash, sharpe))


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    case_dir = tmp_path / CASE_STUDY
    _registry(case_dir, [("bt_short", "pred_short", 2.4), ("bt_whole", "pred_whole", 1.1)])
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda cs, **_: tmp_path / cs)
    return case_dir


class TestTheRecord:
    def test_a_dropped_member_is_written_with_its_reason(self, case_dir: Path) -> None:
        record_prediction_admissibility(
            case_dir, admitted=["pred_whole"], short={"pred_short": SHORTFALL}
        )

        assert predictions_the_sweep_refused(case_dir) == {"pred_short": SHORTFALL}

    def test_an_admitted_member_is_not_reported_as_refused(self, case_dir: Path) -> None:
        record_prediction_admissibility(case_dir, admitted=["pred_whole"], short={})

        assert predictions_the_sweep_refused(case_dir) == {}

    def test_a_later_sweep_replaces_the_earlier_answer(self, case_dir: Path) -> None:
        """Coverage is a property of the artifact, so a refit that fixes it must clear it."""
        record_prediction_admissibility(case_dir, admitted=[], short={"pred_short": SHORTFALL})
        record_prediction_admissibility(case_dir, admitted=["pred_short"], short={})

        assert predictions_the_sweep_refused(case_dir) == {}

    def test_a_registry_with_no_table_reports_nothing_refused(self, case_dir: Path) -> None:
        """Never "everything is refused": that is a registry swept before the table existed."""
        assert predictions_the_sweep_refused(case_dir) == {}

    def test_a_read_only_registry_reports_rather_than_raising(
        self, case_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader's clone must not fail the sweep it is only recording the result of."""
        db_path = case_dir / "run_log" / "registry.db"
        db_path.chmod(0o444)
        try:
            notes = record_prediction_admissibility(
                case_dir, admitted=["pred_whole"], short={"pred_short": SHORTFALL}
            )
        finally:
            db_path.chmod(0o644)

        assert notes and "was not recorded" in notes[0]


class TestTheResolverReadsIt:
    def test_a_refused_prediction_does_not_carry_the_case_study(self, case_dir: Path) -> None:
        """The nasdaq shape: it wins on Sharpe and the sweep will not backtest it."""
        record_prediction_admissibility(
            case_dir, admitted=["pred_whole"], short={"pred_short": SHORTFALL}
        )

        candidates = strategy_analysis.selectable_validation_candidates(CASE_STUDY)

        assert [row["prediction_hash"] for row in candidates] == ["pred_whole"]

    def test_without_a_record_the_same_registry_ranks_both(self, case_dir: Path) -> None:
        """The control. Nothing measured, so nothing is dropped and the pool is unchanged."""
        candidates = strategy_analysis.selectable_validation_candidates(CASE_STUDY)

        assert [row["prediction_hash"] for row in candidates] == ["pred_short", "pred_whole"]

    def test_a_member_recorded_admitted_still_ranks(self, case_dir: Path) -> None:
        """Only a positive refusal removes anything; admission is not a second gate."""
        record_prediction_admissibility(case_dir, admitted=["pred_short", "pred_whole"], short={})

        candidates = strategy_analysis.selectable_validation_candidates(CASE_STUDY)

        assert [row["prediction_hash"] for row in candidates] == ["pred_short", "pred_whole"]

    def test_refusing_every_candidate_says_so_rather_than_returning_nothing(
        self, case_dir: Path
    ) -> None:
        """An empty pool with no reason reads as "the stage has not run"."""
        record_prediction_admissibility(
            case_dir,
            admitted=[],
            short={"pred_short": SHORTFALL, "pred_whole": SHORTFALL},
        )

        with pytest.raises(
            strategy_analysis.NoSelectableCandidates, match="the sweep measured and dropped"
        ):
            strategy_analysis.selectable_validation_candidates(CASE_STUDY)
