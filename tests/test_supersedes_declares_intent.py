"""A supersedes declaration says which lineage it extends, not which generation it replaces.

A ``SUPERSEDES_*`` literal is committed source that has to equal a value the registry moves on
every publish, so it goes stale by construction and every repair is a repair of one instance.
Measured over the corpus on 2026-09-11: 48 declared literals, of which 6 named a value ``create``
accepts, 19 named the generation it had already replaced, 8 named a generation no lineage holds,
and 5 could not be placed at all.

The 19 are the interesting number, because nothing reported them.
``scripts/check_supersedes_literals.py`` called them ``live`` on the strength of
``population_supersedes`` offering a hash that equals ``current.supersedes`` - which is true of
an unchanged re-run and false of the run that pays for a fit. The first two tests here are that
run, end to end through the real ``create`` on both lineages, because the claim that the
declaration is a latent fault rather than a certain failure is the whole reason it was left
alone.

``SUPERSEDES_LIVE`` is what the declaration was for: the author states that this run extends the
lineage published under this name, and the predecessor is looked up rather than quoted. The rest
of this file is what must still fail once it exists, because a declaration that cannot refuse
anything is not a declaration.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CandidateSet, Study
from case_studies.research.comparison import candidate_set_supersedes
from case_studies.research.population import (
    SUPERSEDES_LIVE,
    OfficialPopulation,
    population_supersedes,
)
from tests.test_research_workspace import _seed_release

NAME = "the-set"
POPULATION = "cs:the-population"


@pytest.fixture
def study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _training_spec(alpha: float) -> dict:
    return {
        "identity_version": 2,
        "family": "linear",
        "label": "fwd_ret_21d",
        "label_artifact": "label-a",
        "feature_artifacts": {"financial": "features-a"},
        "feature_names": ["momentum", "volatility"],
        "cv": {"folds": [{"fold": 0, "val_start": "2024-01-05"}]},
        "model": {"class": "Ridge", "params": {"alpha": alpha}},
        "numerics": {"seed": 42, "precision": "float64"},
        "execution_tier": "canonical",
        "seed": 42,
    }


def _member(study: Study, alpha: float):
    frame = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": ["2024-01-05", "2024-01-05"],
            "fold_id": [0, 0],
            "y_true": [0.01, -0.02],
            "y_score": [0.02, -0.01],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    training = study.results.register_training(_training_spec(alpha))
    return study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )


class TestTheFailureThatWasReportedAsLive:
    """Two generations on record, a literal naming the first, and a run that moves members."""

    def test_a_population_literal_one_generation_behind_is_refused_at_the_freeze(
        self, study: Study
    ) -> None:
        first = OfficialPopulation.create(
            study, name=POPULATION, member_kind="prediction", members=["m1"]
        )
        second = OfficialPopulation.create(
            study,
            name=POPULATION,
            member_kind="prediction",
            members=["m1", "m2"],
            supersedes=population_supersedes(study, name=POPULATION, declared=first.hash),
        )
        assert second.supersedes == first.hash

        # The committed literal still names generation one, and the resolver still offers it -
        # this is exactly the state 19 of the corpus's declarations were in.
        offered = population_supersedes(study, name=POPULATION, declared=first.hash)
        assert offered == first.hash

        with pytest.raises(ValueError, match=f"must explicitly supersedes {second.hash}"):
            OfficialPopulation.create(
                study,
                name=POPULATION,
                member_kind="prediction",
                members=["m1", "m2", "m3"],
                supersedes=offered,
            )

    def test_a_candidate_set_literal_one_generation_behind_is_refused_at_the_freeze(
        self, study: Study
    ) -> None:
        one, two, three = (_member(study, alpha) for alpha in (1.0, 2.0, 3.0))
        first = CandidateSet.create(study, NAME, [one])
        second = CandidateSet.create(
            study,
            NAME,
            [one, two],
            supersedes=candidate_set_supersedes(study, name=NAME, declared=first.hash),
        )
        assert second.supersedes == first.hash

        offered = candidate_set_supersedes(study, name=NAME, declared=first.hash)
        assert offered == first.hash

        with pytest.raises(ValueError, match=f"must explicitly supersedes {second.hash}"):
            CandidateSet.create(study, NAME, [one, two, three], supersedes=offered)


class TestWhatTheSentinelResolves:
    def test_it_publishes_over_the_tip_on_the_population_lineage(self, study: Study) -> None:
        first = OfficialPopulation.create(
            study, name=POPULATION, member_kind="prediction", members=["m1"]
        )
        second = OfficialPopulation.create(
            study,
            name=POPULATION,
            member_kind="prediction",
            members=["m1", "m2"],
            supersedes=first.hash,
        )

        third = OfficialPopulation.create(
            study,
            name=POPULATION,
            member_kind="prediction",
            members=["m1", "m2", "m3"],
            supersedes=population_supersedes(study, name=POPULATION, declared=SUPERSEDES_LIVE),
        )

        assert third.supersedes == second.hash

    def test_it_publishes_over_the_tip_on_the_candidate_set_lineage(self, study: Study) -> None:
        one, two, three = (_member(study, alpha) for alpha in (1.0, 2.0, 3.0))
        first = CandidateSet.create(study, NAME, [one])
        second = CandidateSet.create(study, NAME, [one, two], supersedes=first.hash)

        third = CandidateSet.create(
            study,
            NAME,
            [one, two, three],
            supersedes=candidate_set_supersedes(study, name=NAME, declared=SUPERSEDES_LIVE),
        )

        assert third.supersedes == second.hash

    def test_an_unchanged_re_run_still_resolves_to_the_published_generation(
        self, study: Study
    ) -> None:
        """The sentinel must not turn a no-op re-run into a new generation."""
        first = OfficialPopulation.create(
            study, name=POPULATION, member_kind="prediction", members=["m1"]
        )
        second = OfficialPopulation.create(
            study,
            name=POPULATION,
            member_kind="prediction",
            members=["m1", "m2"],
            supersedes=first.hash,
        )

        again = OfficialPopulation.create(
            study,
            name=POPULATION,
            member_kind="prediction",
            members=["m1", "m2"],
            supersedes=population_supersedes(study, name=POPULATION, declared=SUPERSEDES_LIVE),
        )

        assert again.hash == second.hash


class TestWhatMustStillFail:
    """The negative selftests. A declaration that refuses nothing has stopped being one."""

    def test_a_clean_clone_is_still_handed_nothing(self, study: Study) -> None:
        """`create` refuses a first generation that claims to replace one, so the reader's run
        must be handed None rather than a hash - which is the whole reason the declaration is
        resolved instead of passed straight through."""
        assert population_supersedes(study, name=POPULATION, declared=SUPERSEDES_LIVE) is None
        assert candidate_set_supersedes(study, name=NAME, declared=SUPERSEDES_LIVE) is None

    def test_a_first_generation_declaring_the_sentinel_is_still_refused(self, study: Study) -> None:
        """Resolution returns None on an unbound name; passing the sentinel past it does not."""
        with pytest.raises(ValueError, match="first population version cannot supersede"):
            OfficialPopulation.create(
                study,
                name=POPULATION,
                member_kind="prediction",
                members=["m1"],
                supersedes=SUPERSEDES_LIVE,
            )

    def test_a_hash_two_generations_back_is_still_withheld(self, study: Study) -> None:
        """The sentinel rescues the one-behind case on purpose. It must not rescue this one:
        a literal this far back names a lineage the author has stopped tracking, and offering
        the tip anyway would publish over a generation they have never seen."""
        first = OfficialPopulation.create(
            study, name=POPULATION, member_kind="prediction", members=["m1"]
        )
        second = OfficialPopulation.create(
            study,
            name=POPULATION,
            member_kind="prediction",
            members=["m1", "m2"],
            supersedes=first.hash,
        )
        OfficialPopulation.create(
            study,
            name=POPULATION,
            member_kind="prediction",
            members=["m1", "m2", "m3"],
            supersedes=second.hash,
        )

        assert population_supersedes(study, name=POPULATION, declared=first.hash) is None

    def test_a_hash_in_no_lineage_is_still_withheld(self, study: Study) -> None:
        OfficialPopulation.create(study, name=POPULATION, member_kind="prediction", members=["m1"])

        assert population_supersedes(study, name=POPULATION, declared="feedfacefeed") is None

    def test_a_lineage_this_run_does_not_name_is_still_refused(self, study: Study) -> None:
        """The interlock the sentinel must not weaken. Declaring one name says nothing about
        another, so a second lineage whose membership moves is refused exactly as before."""
        one, two = (_member(study, alpha) for alpha in (1.0, 2.0))
        CandidateSet.create(study, "declared-set", [one])
        first_other = CandidateSet.create(study, "undeclared-set", [one])

        assert (
            candidate_set_supersedes(study, name="declared-set", declared=SUPERSEDES_LIVE)
            == CandidateSet.one(study, name="declared-set").hash
        )
        with pytest.raises(ValueError, match=f"must explicitly supersedes {first_other.hash}"):
            CandidateSet.create(study, "undeclared-set", [one, two])

    def test_a_narrowed_run_under_its_own_name_is_still_handed_nothing(self, study: Study) -> None:
        """A caller-chosen name has no prior generation, so the sentinel resolves to nothing
        there too - the isolation a narrowed run depends on is not a property of the hash."""
        OfficialPopulation.create(study, name=POPULATION, member_kind="prediction", members=["m1"])

        assert population_supersedes(study, name="cs:scratch", declared=SUPERSEDES_LIVE) is None
