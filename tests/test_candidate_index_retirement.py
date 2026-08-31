"""A sweep candidate set must not contain a generation its publisher has retired.

`superseded_members` answers which identities a name has moved past. Nothing consumed that
answer: `load_prediction_index` builds the etfs backtest sweep from `prediction_sets` joined
to `training_runs`, which is the catalog, and the catalog does not carry lineage. Measured on
the etfs registry 2026-08-27, 11 of the 228 validation prediction sets in that index belong to
retired generations - ten `latent_factors/cae` and one `latent_factors/pca` - and one of them
held a top-ten shortlist slot, so a retired generation displaced a live prediction from
selection. That is a selection defect, not a reporting one: the retired row is backtested,
ranked, and carried into allocation.

`split_retired_members` is the join between the two. It is deliberately frame-level rather
than a second index loader: the notebook still shows which index it swept, and what was
dropped is a frame it can print rather than a count that vanished inside a query.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from case_studies.research import (
    OfficialPopulation,
    split_retired_members,
    split_unpublished_members,
)
from case_studies.research.workspace import Study
from tests.test_research_workspace import _seed_release

GEN_A = ("aaaa11112222", "bbbb11112222")
GEN_B = ("bbbb11112222", "cccc33334444")


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


def _index(*hashes: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prediction_hash": list(hashes),
            "family": ["linear"] * len(hashes),
            "ic_mean": [0.05] * len(hashes),
        }
    )


def test_a_retired_generation_leaves_the_candidate_set(study: Study) -> None:
    first = _publish(study, GEN_A)
    _publish(study, GEN_B, supersedes=first.hash)

    split = split_retired_members(study, _index(*GEN_A, "cccc33334444"))

    assert split.retired["prediction_hash"].to_list() == ["aaaa11112222"]
    assert sorted(split.live["prediction_hash"].to_list()) == ["bbbb11112222", "cccc33334444"]


def test_a_member_the_refit_did_not_move_stays(study: Study) -> None:
    """Retirement is member-wise within a name, so a config that did not move is still live."""
    first = _publish(study, GEN_A)
    _publish(study, GEN_B, supersedes=first.hash)

    split = split_retired_members(study, _index("bbbb11112222"))

    assert split.live.height == 1
    assert split.retired.is_empty()


def test_nothing_published_drops_nothing(study: Study) -> None:
    """The no-op case. A clean clone has no lineage, and a filter that empties the sweep
    there is worse than no filter: the run dies with "no predictions" and nothing says why."""
    split = split_retired_members(study, _index(*GEN_A, "cccc33334444"))

    assert split.retired.is_empty()
    assert split.live.height == 3


def test_one_generation_with_no_supersedes_edge_drops_nothing(study: Study) -> None:
    """A published population is not by itself evidence that anything was retired."""
    _publish(study, GEN_A)

    split = split_retired_members(study, _index(*GEN_A))

    assert split.retired.is_empty()
    assert split.live.height == 2


def test_the_split_is_a_partition(study: Study) -> None:
    """Every input row lands in exactly one side, and the index columns survive both.

    Asserting the counts alone would pass an implementation that dropped a row from both
    sides, which is the shape a mis-specified anti-join takes.
    """
    first = _publish(study, GEN_A)
    _publish(study, GEN_B, supersedes=first.hash)
    index = _index(*GEN_A, "cccc33334444")

    split = split_retired_members(study, index)

    assert split.live.columns == index.columns
    assert split.retired.columns == index.columns
    rejoined = pl.concat([split.live, split.retired]).sort("prediction_hash")
    assert rejoined.equals(index.sort("prediction_hash"))


def test_an_empty_index_stays_empty_and_keeps_its_schema(study: Study) -> None:
    empty = _index().clear()

    split = split_retired_members(study, empty)

    assert split.live.is_empty()
    assert split.retired.is_empty()
    assert split.live.columns == empty.columns


def test_a_missing_identity_column_is_refused(study: Study) -> None:
    with pytest.raises(ValueError, match="prediction_hash"):
        split_retired_members(study, pl.DataFrame({"family": ["linear"]}))


# --- membership answers what exclusion cannot -----------------------------------------
#
# "Not retired" and "published" differ by the identities no population ever listed. Nobody
# retired an experimental fit, so `split_retired_members` admits it and a sweep can rank it into
# a slot and carry it downstream as though the case study published it. Measured on the etfs
# registry 2026-08-30: of 682 validation prediction sets, 134 are superseded and 498 are
# published, leaving 50 that are live by exclusion and listed by no population at all.

UNLISTED = "dddd55556666"


def test_a_prediction_no_population_lists_is_not_a_candidate(study: Study) -> None:
    _publish(study, GEN_A)

    split = split_unpublished_members(study, _index(*GEN_A, UNLISTED))

    assert split.live["prediction_hash"].to_list() == list(GEN_A)
    assert split.retired["prediction_hash"].to_list() == [UNLISTED]


def test_the_exclusion_split_admits_the_row_membership_rejects(study: Study) -> None:
    """The two answers differ, which is the whole reason for the second one.

    Asserted against `split_retired_members` on the same index rather than described, so this
    goes on failing if the exclusion split is ever changed to answer the membership question
    and the distinction quietly disappears.
    """
    _publish(study, GEN_A)
    index = _index(*GEN_A, UNLISTED)

    assert UNLISTED in split_retired_members(study, index).live["prediction_hash"].to_list()
    assert UNLISTED not in split_unpublished_members(study, index).live["prediction_hash"].to_list()


def test_a_retired_generation_is_excluded_by_membership_too(study: Study) -> None:
    first = _publish(study, GEN_A)
    _publish(study, GEN_B, supersedes=first.hash)

    live = split_unpublished_members(study, _index(*GEN_A, *GEN_B)).live["prediction_hash"]

    assert sorted(set(live.to_list())) == sorted(set(GEN_B))


def test_with_no_population_declared_every_candidate_survives(study: Study) -> None:
    """Membership cannot be asked, so the exclusion split stands in rather than refusing all.

    A fixture registry, or a study written before populations existed, declares none. Narrowing
    to an empty membership set there would refuse every candidate and report it as though the
    case study published nothing.
    """
    index = _index(*GEN_A, UNLISTED)

    assert split_unpublished_members(study, index).live.height == index.height
