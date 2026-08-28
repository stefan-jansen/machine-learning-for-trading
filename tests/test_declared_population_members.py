"""The contract `13_model_analysis` uses to decide whether a registry declares populations.

`OfficialPopulation.one` reports two different registry states in the same words - "resolved
to 0 current identities" - and they call for opposite responses. A registry that has published
nothing is the ordinary state of a fixture and of a reader's clean clone; a registry that
publishes names and cannot resolve this one has a broken lineage, and comparing its rows would
report a family no declaration covers.

These exercise the decision itself rather than the resolver underneath it, so removing the
branch fails them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.research import OfficialPopulation, Study
from case_studies.utils.notebook_contracts import declared_population_members
from tests.test_research_workspace import _seed_release

NAMES = {"linear": "etfs-linear-validation-v1"}
MEMBERS = ("aaaa11112222", "bbbb33334444")


@pytest.fixture
def writable(tmp_path: Path) -> Study:
    """Publishing needs a writable study; the contract under test reads through `Study.at`.

    Splitting them is not incidental - it is the arrangement the notebook is in. A notebook
    that registers nothing holds a read-only handle, and the population it must resolve was
    written by a different run through a writable one.
    """
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


@pytest.fixture
def case_dir(writable: Study) -> Path:
    return writable.root


def _study(case_dir: Path) -> Study:
    return Study.at(case_dir, case_study="etfs")


def _publish(writable: Study, members=MEMBERS) -> OfficialPopulation:
    return OfficialPopulation.create(
        writable,
        name=NAMES["linear"],
        member_kind="prediction",
        members=list(members),
        supersedes=None,
    )


class TestARegistryThatDeclaresNothing:
    def test_a_registry_predating_the_mechanism_is_answered_not_asked(self, tmp_path: Path) -> None:
        """The case the branch exists for, and the one that fails without it.

        A registry whose schema has no `official_populations` table makes
        `OfficialPopulation.one` raise `sqlite3.OperationalError`, which is neither error the
        handler catches. Removing the guard therefore does not degrade this to a note - it
        propagates, and this test fails.
        """
        case_dir = tmp_path / "etfs"
        (case_dir / "run_log").mkdir(parents=True)
        with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
            db.execute("CREATE TABLE unrelated (x INTEGER)")

        members, notes = declared_population_members(
            _study(case_dir), case_dir, NAMES, produced={"linear": 6}
        )

        assert members == {}
        assert any("publishes no official populations" in note for note in notes)

    def test_it_says_so_rather_than_reporting_a_missing_declaration(self, case_dir: Path) -> None:
        """The note has to name the weaker claim, not a fault: nothing here is wrong."""
        members, notes = declared_population_members(
            _study(case_dir), case_dir, NAMES, produced={"linear": 6}
        )

        assert members == {}
        assert notes and "catalog admissibility" in notes[0]

    def test_a_directory_with_no_registry_is_the_same_state(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "absent"

        members, notes = declared_population_members(
            _study(case_dir), case_dir, NAMES, produced={"linear": 0}
        )

        assert members == {}
        assert notes


class TestARegistryThatDeclares:
    def test_it_returns_the_members_in_force(self, writable: Study, case_dir: Path) -> None:
        _publish(writable)

        members, notes = declared_population_members(
            _study(case_dir), case_dir, NAMES, produced={"linear": 2}
        )

        assert members == {"linear": set(MEMBERS)}
        assert notes == []

    def test_a_family_with_rows_and_no_resolvable_declaration_refuses(
        self, writable: Study, case_dir: Path
    ) -> None:
        """Published populations, and this family's name is not among them."""
        _publish(writable)

        with pytest.raises(RuntimeError, match="does not resolve"):
            declared_population_members(
                _study(case_dir),
                case_dir,
                {"gbm": "etfs-gbm-validation-v1"},
                produced={"gbm": 4},
            )

    def test_a_family_that_produced_nothing_is_only_a_note(
        self, writable: Study, case_dir: Path
    ) -> None:
        """Nothing has been fitted, so there is nothing yet to be undeclared."""
        _publish(writable)

        members, notes = declared_population_members(
            _study(case_dir),
            case_dir,
            {"gbm": "etfs-gbm-validation-v1"},
            produced={"gbm": 0},
        )

        assert members == {}
        assert any("no current official population for gbm" in note for note in notes)
