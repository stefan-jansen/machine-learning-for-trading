"""The candidate pool stages 14-18 draw from, and the two steps it takes.

A sweep does not name the families it ranks, so it cannot ask `declared_population_members`
whether declared names resolved - it has to ask which members are in force at all. Resolving
each published name is not enough on its own: a narrowed or preview run freezes its own snapshot
of whatever the catalog held that day and stays in force under that name forever, so the union
over names hands a retired generation back through the frozen name that still lists it. The
subtraction is what removes it, and these pin that both steps are there.
"""

from __future__ import annotations

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
        assert notes == []

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
