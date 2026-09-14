"""The candidate pool stages 14-18 draw from, and the two steps it takes.

A sweep does not name the families it ranks, so it cannot ask `declared_population_members`
whether declared names resolved - it has to ask which members are in force at all. Resolving
each published name is not enough on its own: a narrowed or preview run freezes its own snapshot
of whatever the catalog held that day and stays in force under that name forever, so the union
over names hands a retired generation back through the frozen name that still lists it. The
subtraction is what removes it, and these pin that both steps are there.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.research import OfficialPopulation, Study
from case_studies.utils.notebook_contracts import prediction_members_in_force
from tests.test_research_workspace import _seed_release


@pytest.fixture
def study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _publish(study: Study, name: str, members, supersedes: str | None = None):
    return OfficialPopulation.create(
        study,
        name=name,
        member_kind="prediction",
        members=sorted(members),
        supersedes=supersedes,
    )


class TestARegistryThatPublishesNothing:
    def test_answers_none_rather_than_an_empty_pool(self, study: Study) -> None:
        """The distinction the caller acts on: nothing to filter by, so do not filter. An empty
        set here would read as "no prediction set is admissible" and empty every sweep in a
        fixture and in a reader's clean clone."""
        members, notes = prediction_members_in_force(study)
        assert members is None
        assert notes and "publishes no official populations" in notes[0]


class TestARegistryThatPublishes:
    def test_unions_what_every_name_publishes(self, study: Study) -> None:
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111", "bbbb2222"])
        _publish(study, "etfs-gbm-validation-v1", ["cccc3333"])
        members, notes = prediction_members_in_force(study)
        assert members == frozenset({"aaaa1111", "bbbb2222", "cccc3333"})
        # The fixture registers no coverage rows, which is now reported rather than passed over
        # in silence: an absent row means completeness is unevidenced, not that the run stopped
        # part way, so the members stay in the pool and the gap is named.
        assert len(notes) == 1
        assert "carry no prediction_coverage row" in notes[0]

    def test_a_superseded_generation_is_not_in_the_pool(self, study: Study) -> None:
        first = _publish(study, "etfs-linear-validation-v1", ["aaaa1111", "bbbb2222"])
        _publish(
            study, "etfs-linear-validation-v1", ["aaaa1111", "cccc3333"], supersedes=first.hash
        )
        members, _ = prediction_members_in_force(study)
        assert members == frozenset({"aaaa1111", "cccc3333"})

    def test_a_frozen_name_cannot_reinstate_a_member_its_own_name_retired(
        self, study: Study
    ) -> None:
        """The reason the subtraction is a second step. The preview name is in force - nothing
        supersedes it - so the union over names returns the retired member through it, and only
        the per-name retirement recorded under the other name removes it again."""
        first = _publish(study, "etfs-linear-validation-v1", ["aaaa1111", "bbbb2222"])
        _publish(
            study, "etfs-linear-validation-v1", ["aaaa1111", "cccc3333"], supersedes=first.hash
        )
        _publish(study, "etfs-linear-preview", ["aaaa1111", "bbbb2222"])
        members, _ = prediction_members_in_force(study)
        assert "bbbb2222" not in members
        assert members == frozenset({"aaaa1111", "cccc3333"})


class TestAPoolWithAnUnfinishedMember:
    """The pool is selected from, so a member scored over fewer folds than it was asked for
    cannot be ranked around - a shorter window is an easier window, and the error runs toward
    the top of the ranking rather than away from it."""

    def _register(self, study: Study, member: str, *, expected: int, scored: int) -> None:
        run_log = study.root / "run_log"
        with sqlite3.connect(run_log / "registry.db") as db:
            # Named columns, not positional, and no CREATE TABLE for prediction_coverage. The
            # seeded study already creates it with the fifteen columns the producer writes, so
            # `CREATE TABLE IF NOT EXISTS` was a no-op declaring a three-column shape that does
            # not exist, and the positional insert under it failed against the real table. Only
            # the three columns this test is about are set; the rest keep their defaults.
            db.execute(
                "INSERT OR REPLACE INTO prediction_coverage ("
                "  prediction_hash, expected_key_digest, actual_key_digest, n_expected,"
                "  n_actual, n_duplicates, n_missing, n_extra, n_null, n_non_finite,"
                "  n_folds_expected, n_folds_actual, schema_json, artifact_digest, status"
                ") VALUES (?, '', '', 0, 0, 0, 0, 0, 0, 0, ?, ?, '{}', '', 'complete')",
                (member, expected, scored),
            )
            db.executemany(
                "INSERT INTO fold_metrics (prediction_hash, fold_id, computed_at, ic) "
                "VALUES (?, ?, '2026-01-01T00:00:00Z', 0.01)",
                [(member, fold) for fold in range(scored)],
            )
        artifact = run_log / "predictions" / member
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "predictions.parquet").write_bytes(b"PAR1")

    def test_refuses_and_names_what_is_short(self, study: Study) -> None:
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111", "bbbb2222"])
        self._register(study, "aaaa1111", expected=5, scored=5)
        self._register(study, "bbbb2222", expected=5, scored=2)
        with pytest.raises(RuntimeError, match="2 of 5 folds scored"):
            prediction_members_in_force(study)

    def test_allows_a_pool_whose_members_all_finished(self, study: Study) -> None:
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111", "bbbb2222"])
        self._register(study, "aaaa1111", expected=5, scored=5)
        self._register(study, "bbbb2222", expected=5, scored=5)
        members, notes = prediction_members_in_force(study)
        assert members == frozenset({"aaaa1111", "bbbb2222"})
        assert notes == []


class TestAMemberThatScoredEveryFoldOnFewerDates:
    """Fold completeness cannot see a checkpoint that collapsed inside its folds.

    A sequence model that settles into predicting nearly the same value for every fund on a date
    gives that date no spread to rank, and the date drops out of the daily IC. The member still
    reports every fold scored, so the fold check above passes it, and it is flattered by the days
    it could not score. Nothing in the pool is measured against the dates the resolved eligibility
    declared, so the publishing notebook is the only place that knows - and narrowing what it
    published is the only thing that takes such a member out of the sweep.
    """

    def test_the_fold_check_passes_it_and_it_stays_in_the_pool(self, study: Study) -> None:
        _publish(study, "etfs-nlinear-validation-v1", ["aaaa1111", "bbbb2222"])
        register = TestAPoolWithAnUnfinishedMember()._register
        register(study, "aaaa1111", expected=5, scored=5)
        register(study, "bbbb2222", expected=5, scored=5)
        members, notes = prediction_members_in_force(study)
        assert members == frozenset({"aaaa1111", "bbbb2222"})
        assert notes == []

    def test_narrowing_the_population_removes_it_from_the_candidate_set(self, study: Study) -> None:
        declared = _publish(study, "etfs-nlinear-validation-v1", ["aaaa1111", "bbbb2222"])
        register = TestAPoolWithAnUnfinishedMember()._register
        register(study, "aaaa1111", expected=5, scored=5)
        register(study, "bbbb2222", expected=5, scored=5)
        # What `10a_dl_nlinear` does once coverage is measurable: republish the full-coverage
        # subset under the same name, naming the declared generation it retires.
        _publish(study, "etfs-nlinear-validation-v1", ["aaaa1111"], supersedes=declared.hash)
        members, _ = prediction_members_in_force(study)
        assert members == frozenset({"aaaa1111"})


class TestScopingToWhatTheCallerCanAct:
    """``candidates`` narrows the scope, not the checks.

    Without it this reads every published member of every label. On
    ``nasdaq100_microstructure/14_backtest`` that was 784 members, 94.3 GB of
    ``predictions.parquet``, 8,608 s before the first backtest - and the run then swept 162,
    because the other 537 belong to labels it can never sweep. The caller already knows which
    is which: it has built and narrowed ``pred_index`` before it asks.

    The refusal is registered here too, because what has to be shown is that it narrows with
    the scope rather than that it is skipped.
    """

    _register = TestAPoolWithAnUnfinishedMember._register

    def test_the_pool_is_the_intersection(self, study: Study) -> None:
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111", "bbbb2222"])
        _publish(study, "etfs-gbm-validation-v1", ["cccc3333"])
        members, _ = prediction_members_in_force(study, candidates=["aaaa1111", "cccc3333"])
        assert members == frozenset({"aaaa1111", "cccc3333"})

    def test_a_candidate_no_population_publishes_is_not_admitted(self, study: Study) -> None:
        """Scoping cannot add. The caller's list narrows the pool and never widens it."""
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111"])
        members, _ = prediction_members_in_force(study, candidates=["aaaa1111", "dddd4444"])
        assert members == frozenset({"aaaa1111"})

    def test_omitting_it_is_exactly_todays_answer(self, study: Study) -> None:
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111", "bbbb2222"])
        _publish(study, "etfs-gbm-validation-v1", ["cccc3333"])
        wide, wide_notes = prediction_members_in_force(study)
        same, same_notes = prediction_members_in_force(
            study, candidates=["aaaa1111", "bbbb2222", "cccc3333"]
        )
        assert wide == same
        assert wide_notes == same_notes

    def test_an_unfinished_member_inside_the_scope_still_refuses(self, study: Study) -> None:
        """The check is asked of every member the caller could select, exactly as before."""
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111", "bbbb2222"])
        self._register(study, "aaaa1111", expected=5, scored=5)
        self._register(study, "bbbb2222", expected=5, scored=2)
        with pytest.raises(RuntimeError, match="2 of 5 folds scored"):
            prediction_members_in_force(study, candidates=["aaaa1111", "bbbb2222"])

    def test_an_unfinished_member_outside_the_scope_does_not(self, study: Study) -> None:
        """The one behaviour change, stated as a test rather than left to be discovered.

        Today an unfinished member under any label refuses every run. That refusal protects a
        selection, and a member that cannot be selected cannot corrupt one - so scoped, it is
        the sweep that actually reads that label which refuses. What is lost is an incidental
        early warning about the registry, not a guarantee about this run.
        """
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111"])
        _publish(study, "etfs-gbm-validation-v1", ["bbbb2222"])
        self._register(study, "aaaa1111", expected=5, scored=5)
        self._register(study, "bbbb2222", expected=5, scored=2)

        with pytest.raises(RuntimeError, match="2 of 5 folds scored"):
            prediction_members_in_force(study)

        members, _ = prediction_members_in_force(study, candidates=["aaaa1111"])
        assert members == frozenset({"aaaa1111"})

    def test_a_member_outside_the_scope_is_not_recorded_as_inadmissible(self, study: Study) -> None:
        """The interaction with `record_prediction_admissibility`, pinned rather than reasoned.

        #1076 writes this measurement into `prediction_admissibility` so the carrier resolver
        reads the answer instead of asking a weaker question, and its contract is that absence
        is not a verdict: only measured members are written, and the resolver drops only what
        is recorded as NOT admitted. Scoping makes more members absent, so this asserts the
        half that keeps that safe - an out-of-scope member must be absent from the table, not
        written with `admitted=0`. Writing it as inadmissible would let one label's sweep
        disqualify another label's carrier.
        """
        _publish(study, "etfs-linear-validation-v1", ["aaaa1111"])
        _publish(study, "etfs-gbm-validation-v1", ["bbbb2222"])
        self._register(study, "aaaa1111", expected=5, scored=5)
        self._register(study, "bbbb2222", expected=5, scored=5)

        prediction_members_in_force(study, candidates=["aaaa1111"])

        with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
            rows = dict(
                db.execute("SELECT prediction_hash, admitted FROM prediction_admissibility")
            )
        assert rows.get("aaaa1111") == 1, "the member in scope should be recorded as admitted"
        assert "bbbb2222" not in rows, (
            "a member this sweep never measured must be absent, not recorded inadmissible"
        )
