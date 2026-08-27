"""When a notebook's declared population hash may be offered to the registry.

A modelling notebook that has published a population and then moves a training identity
must name the snapshot it replaces, or ``OfficialPopulation.create`` refuses the write.
The hash it names is committed source, so one declaration has to be right for the author
refitting, for the author re-running, and for a reader on a clean clone whose ``run_log/``
is gitignored and therefore empty. ``population_supersedes`` is the single decision, and
every case below is one that has already broken a notebook when it was decided by hand.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from case_studies.research import (
    OfficialPopulation,
    current_prediction_members,
    population_supersedes,
    superseded_members,
)
from case_studies.research.workspace import Study
from tests.test_research_workspace import _seed_release

MEMBERS_ONE = ("aaaa11112222", "bbbb11112222")
MEMBERS_TWO = ("cccc33334444", "dddd33334444")


@pytest.fixture
def study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _publish(study: Study, members, *, supersedes: str | None = None) -> OfficialPopulation:
    return OfficialPopulation.create(
        study,
        name="etfs-linear-validation-v1",
        member_kind="prediction",
        members=list(members),
        supersedes=supersedes,
    )


class TestWhatTheHelperDecides:
    def test_an_empty_declaration_is_never_offered(self, study: Study) -> None:
        assert population_supersedes(study, name="etfs-linear-validation-v1", declared="") is None
        assert population_supersedes(study, name="etfs-linear-validation-v1", declared=None) is None

    def test_a_clean_clone_withholds_the_declared_hash(self, study: Study) -> None:
        # Nothing has been published under this name, which is what a reader sees: `run_log/`
        # is gitignored, so the registry has no generation to supersede. Offering the hash
        # here makes `create` refuse a first version that claims to supersede something, and
        # the reader's run dies at the population write before any fit.
        assert (
            population_supersedes(study, name="etfs-linear-validation-v1", declared="feedfacefeed")
            is None
        )

    def test_the_refit_offers_the_hash_of_the_generation_in_force(self, study: Study) -> None:
        # The author holds generation one and declares it in order to publish generation two.
        first = _publish(study, MEMBERS_ONE)
        assert population_supersedes(study, name=first.name, declared=first.hash) == first.hash

    def test_the_re_run_offers_the_hash_the_generation_in_force_superseded(
        self, study: Study
    ) -> None:
        # Generation two is in force and its own `supersedes` is generation one, which is what
        # the notebook still declares. Offering it recomputes the same snapshot, so the
        # notebook resolves to the population it published instead of writing a third.
        first = _publish(study, MEMBERS_ONE)
        second = _publish(study, MEMBERS_TWO, supersedes=first.hash)
        assert second.hash != first.hash
        assert population_supersedes(study, name=second.name, declared=first.hash) == first.hash

    def test_a_hash_that_names_neither_is_withheld(self, study: Study) -> None:
        # `create` then refuses and names the hash it requires, which is a better answer than
        # this helper guessing at a lineage the declaration does not describe.
        _publish(study, MEMBERS_ONE)
        assert (
            population_supersedes(study, name="etfs-linear-validation-v1", declared="0123456789ab")
            is None
        )

    def test_a_narrowed_run_under_its_own_name_is_withheld(self, study: Study) -> None:
        # A caller-chosen POPULATION_NAME has no prior generation, so the declaration that is
        # correct for the canonical name must not be carried onto it.
        first = _publish(study, MEMBERS_ONE)
        assert population_supersedes(study, name="etfs-linear-scratch", declared=first.hash) is None


class TestTheDecisionsItReplaces:
    """Each of these is how the decision was made by hand, and each is wrong somewhere."""

    def test_offering_the_hash_whenever_any_generation_exists_is_not_the_rule(
        self, study: Study
    ) -> None:
        # The second run on a clean clone: run 1 wrote generation one, whose own `supersedes`
        # is None and whose hash is not what the notebook declares. "A generation exists" is
        # true, so that rule offers the declared hash and writes a generation nobody asked
        # for. The helper withholds it.
        _publish(study, MEMBERS_ONE)
        assert (
            population_supersedes(study, name="etfs-linear-validation-v1", declared="feedfacefeed")
            is None
        )

    def test_matching_only_the_supersedes_column_blocks_the_refit(self, study: Study) -> None:
        # Testing `current.supersedes == declared` alone withholds the hash from an author
        # holding generation one who declares it to publish generation two, so the clean-clone
        # reader is fixed by making the publication impossible. The helper offers it.
        first = _publish(study, MEMBERS_ONE)
        assert first.supersedes is None
        assert population_supersedes(study, name=first.name, declared=first.hash) == first.hash


class TestWhatALaterGenerationRetires:
    """`superseded_members` is what a downstream stage must consume instead of the catalog.

    `identity_status` is the schema version a row was written under. It cannot distinguish the
    generation a producer publishes from the one it replaced, so a backtest selecting on it
    sweeps both and succeeds - over twice the population, with the retired half frozen into
    whatever it publishes next.
    """

    def test_nothing_is_retired_before_anything_is_superseded(self, study: Study) -> None:
        assert superseded_members(study) == frozenset()
        _publish(study, MEMBERS_ONE)
        assert superseded_members(study) == frozenset()

    def test_a_registry_with_no_population_table_retires_nothing(self, tmp_path: Path) -> None:
        """A reader's clean clone, and the branch a schema-complete workspace cannot reach.

        `Study.open(workspace=...)` runs `_open_registry`, which creates
        `official_populations` empty, so a study built that way exits through the
        no-generations return and never touches the missing-table handler. Dropping the table
        from both registries is what puts that handler under test.
        """
        fresh = Study.open(
            "etfs", workspace=tmp_path / "fresh", release_root=_seed_release(tmp_path)
        )
        for root in (fresh.root, fresh.release_case_root):
            db_path = root / "run_log" / "registry.db"
            if not db_path.exists():
                continue
            with sqlite3.connect(db_path) as db:
                db.execute("DROP TABLE IF EXISTS official_populations")
                db.execute("DROP TABLE IF EXISTS official_population_members")
        assert superseded_members(fresh) == frozenset()

    def test_a_workspace_reads_the_released_registry_s_lineage(self, tmp_path: Path) -> None:
        """The overlay, which is where reading only `study.root` gets it wrong.

        `PredictionCatalog.table()` offers released rows the workspace registry does not hold,
        and their lineage lives in the released registry - which `Study.open` never copies. A
        workspace gets its own `official_populations`, created schema-complete and empty, so
        reading only `study.root` returns nothing retired and the filter is a no-op. Same
        failure mode as the global form, by a different route.
        """
        release_root = _seed_release(tmp_path)
        published = Study.open("etfs", release_root=release_root, workspace=tmp_path / "author")
        first = _publish(published, MEMBERS_ONE)
        _publish(published, MEMBERS_TWO, supersedes=first.hash)
        # Move what the author published into the release root, which is what a reader sees.
        released_db = published.release_case_root / "run_log" / "registry.db"
        released_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(published.root / "run_log" / "registry.db", released_db)

        reader = Study.open("etfs", release_root=release_root, workspace=tmp_path / "reader")
        assert reader.root != reader.release_case_root
        assert superseded_members(reader) == frozenset(MEMBERS_ONE)

    def test_a_refit_retires_the_generation_it_replaced(self, study: Study) -> None:
        first = _publish(study, MEMBERS_ONE)
        _publish(study, MEMBERS_TWO, supersedes=first.hash)
        assert superseded_members(study) == frozenset(MEMBERS_ONE)

    def test_an_identity_the_refit_still_lists_is_not_retired(self, study: Study) -> None:
        # The case that makes this member-wise rather than population-wise: a refit that moves
        # one of two identities retires one. Retiring the whole predecessor would drop a
        # prediction set that is still exactly what its producer publishes.
        kept, moved = MEMBERS_ONE
        first = _publish(study, (kept, moved))
        _publish(study, (kept, "eeee55556666"), supersedes=first.hash)
        assert superseded_members(study) == frozenset({moved})

    def test_the_whole_chain_is_retired_not_only_the_last_step(self, study: Study) -> None:
        first = _publish(study, MEMBERS_ONE)
        second = _publish(study, MEMBERS_TWO, supersedes=first.hash)
        third = ("eeee55556666", "ffff55556666")
        _publish(study, third, supersedes=second.hash)
        assert superseded_members(study) == frozenset(MEMBERS_ONE) | frozenset(MEMBERS_TWO)

    def test_another_name_s_lineage_is_independent(self, study: Study) -> None:
        # Populations are not named to a convention, so the answer must come from the lineage
        # and not from the name. A narrowed run's own chain retires its own predecessors only.
        first = _publish(study, MEMBERS_ONE)
        _publish(study, MEMBERS_TWO, supersedes=first.hash)
        scratch = OfficialPopulation.create(
            study,
            name="etfs-linear-scratch",
            member_kind="prediction",
            members=["9999aaaabbbb"],
        )
        assert scratch.hash not in superseded_members(study)
        assert superseded_members(study) == frozenset(MEMBERS_ONE)

    def test_a_backtest_lineage_does_not_retire_a_prediction(self, study: Study) -> None:
        # Kinds are separate chains: a re-backtest supersedes backtest identities and must not
        # be read as retiring the prediction sets they were run over.
        first = OfficialPopulation.create(
            study, name="etfs-baselines", member_kind="backtest", members=["1111bbbb2222"]
        )
        OfficialPopulation.create(
            study,
            name="etfs-baselines",
            member_kind="backtest",
            members=["3333bbbb4444"],
            supersedes=first.hash,
        )
        assert superseded_members(study, member_kind="prediction") == frozenset()
        assert superseded_members(study, member_kind="backtest") == frozenset({"1111bbbb2222"})

    def test_a_stale_snapshot_under_another_name_cannot_un_retire_a_generation(
        self, study: Study
    ) -> None:
        """The case that made the global form wrong on real data.

        A narrowed or preview run freezes its own snapshot of whatever the catalog held on
        the day it ran, and that snapshot stays in force under its own name forever. Asking
        "retired by someone, listed by nobody in force" then answers no: the stale snapshot
        lists the retired members, so the refit reads as though it never happened.

        Measured on `fx_pairs` 2026-08-25. Refitting `tabular_dl` retired 72 prediction sets;
        `fx_pairs:preflight-baselines`, frozen the previous day, still listed all 72; the
        global form returned zero retired and the backtest sweep would have run over both
        generations exactly as if the filter were absent. Every unit test passed.
        """
        first = _publish(study, MEMBERS_ONE)
        OfficialPopulation.create(
            study,
            name="fx-preflight-baselines",
            member_kind="prediction",
            members=list(MEMBERS_ONE),
        )
        _publish(study, MEMBERS_TWO, supersedes=first.hash)
        assert superseded_members(study) == frozenset(MEMBERS_ONE)

    def test_a_registry_error_that_is_not_a_missing_table_propagates(self, tmp_path: Path) -> None:
        """The blanket catch this replaced turned every failure into "nothing is retired".

        A lock timeout, an I/O error and a half-migrated schema are not evidence that no
        generation has been superseded, and answering them with an empty set is the silent
        wrong answer the whole module exists to prevent - the sweep runs over both generations
        and reports every member complete. Only a missing file or a missing table means
        nothing was ever written.
        """
        study = Study.open(
            "etfs", workspace=tmp_path / "broken", release_root=_seed_release(tmp_path)
        )
        _publish(study, MEMBERS_ONE)
        with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
            db.execute("ALTER TABLE official_populations RENAME COLUMN supersedes_hash TO gone")
        with pytest.raises(sqlite3.OperationalError, match="supersedes_hash"):
            superseded_members(study)

    def test_a_hash_superseded_in_either_registry_is_superseded(self, tmp_path: Path) -> None:
        """Supersession is monotone, so the union of edges across both roots is the merge.

        The two roots disagree in the ordinary case, not the exotic one. An author copies a
        workspace registry into the release root - what `test_a_workspace_reads_the_released_
        registry_s_lineage` calls what a reader sees - and then refits, which leaves the release
        root holding generation A as an unsuperseded tip while the workspace holds A -> B. That
        is the state right after every release.

        A stale root is not independent evidence that A is still published; it is an older copy
        of the same content-addressed chain. Reading tip-ness per root and keeping any root's
        tip alive lets the lagging root veto every retirement for the names it holds, while
        `PredictionCatalog.table()` goes on overlaying generation A's rows into that workspace -
        so the sweep runs over both generations, which is the failure this function exists to
        stop.
        """
        release_root = _seed_release(tmp_path)
        author = Study.open("etfs", release_root=release_root, workspace=tmp_path / "author")
        first = _publish(author, MEMBERS_ONE)
        released_db = author.release_case_root / "run_log" / "registry.db"
        released_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(author.root / "run_log" / "registry.db", released_db)

        refitter = Study.open("etfs", release_root=release_root, workspace=tmp_path / "refit")
        shutil.copy(released_db, refitter.root / "run_log" / "registry.db")
        _publish(refitter, MEMBERS_TWO, supersedes=first.hash)
        assert superseded_members(refitter) == frozenset(MEMBERS_ONE)


class TestTheSetASelectingStageConsumes:
    """`current_prediction_members` is the union of the tips minus what their names retired.

    Six notebooks in `sp500_equity_option_analytics` built this set inline and each built the
    union alone. On that registry the two forms agreed - measured 2026-08-27, 947 members in
    the union and zero of them retired - so nothing published was wrong and no test could
    have caught it from the data. The cases below are the ones where they diverge.
    """

    def test_a_frozen_snapshot_does_not_return_a_retired_generation_to_the_set(
        self, study: Study
    ) -> None:
        """The union alone is the global form the helper's own docstring rejects.

        `fx-preflight-baselines` is in force under its own name and lists the first
        generation, so unioning the tips puts every retired member back. Retirement is a
        statement by the name that published the member; another name's stale snapshot
        does not answer it.
        """
        first = _publish(study, MEMBERS_ONE)
        OfficialPopulation.create(
            study,
            name="fx-preflight-baselines",
            member_kind="prediction",
            members=list(MEMBERS_ONE),
        )
        _publish(study, MEMBERS_TWO, supersedes=first.hash)

        assert current_prediction_members(study, verify_members=False) == frozenset(MEMBERS_TWO)

    def test_a_backtest_population_contributes_no_prediction_identities(self, study: Study) -> None:
        """`member_kind` is filtered rather than assumed: a backtest hash is not a prediction."""
        _publish(study, MEMBERS_ONE)
        OfficialPopulation.create(
            study,
            name="etfs-linear-validation-backtests-v1",
            member_kind="backtest",
            members=["eeee55556666", "ffff55556666"],
        )

        assert current_prediction_members(study, verify_members=False) == frozenset(MEMBERS_ONE)

    def test_a_workspace_reads_the_released_registry_s_names(self, tmp_path: Path) -> None:
        """Reading names from `study.root` alone publishes a set the catalog then overlays.

        `Study.open` never copies the released `run_log` into a workspace, and the workspace
        gets its own `official_populations` created schema-complete and empty. Names read from
        one root and retirements computed across both is a mismatch in the direction that
        loses live members: the reader below would resolve nothing at all.
        """
        release_root = _seed_release(tmp_path)
        published = Study.open("etfs", release_root=release_root, workspace=tmp_path / "author")
        first = _publish(published, MEMBERS_ONE)
        _publish(published, MEMBERS_TWO, supersedes=first.hash)
        released_db = published.release_case_root / "run_log" / "registry.db"
        released_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(published.root / "run_log" / "registry.db", released_db)

        reader = Study.open("etfs", release_root=release_root, workspace=tmp_path / "reader")
        assert reader.root != reader.release_case_root
        assert current_prediction_members(reader, verify_members=False) == frozenset(MEMBERS_TWO)

    def test_a_registry_with_no_populations_yields_an_empty_set(self, tmp_path: Path) -> None:
        study = Study.open(
            "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
        )
        assert current_prediction_members(study, verify_members=False) == frozenset()
