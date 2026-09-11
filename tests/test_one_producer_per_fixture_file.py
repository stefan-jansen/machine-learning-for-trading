"""Two producers wrote the same four fixture paths, and the last one to run won.

``tests/generate_test_microstructure.py`` builds a small synthetic microstructure day.
Four of the paths it wrote were later re-sourced from production - the ITCH ``A``, ``P``
and ``R`` message files on 2026-05-17, to unblock
``08_financial_features/02_microstructure_features``, and
``futures/market/individual/ES/data.parquet`` on 2026-05-06, for the current futures
schema. Nobody told the generator. From then on it wrote 20, 3 and 3 ITCH rows over
2,500, 2,500 and 5, and 345 synthetic ES bars over 19,361 real ones, and the fixture
held whichever producer had run last.

Nothing caught it because no test compared the two producers. These do: the generator
declares what production owns, ``create_test_data.py`` declares the same four paths, and
the two declarations have to partition the fixture rather than overlap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO))

import generate_test_microstructure as generator  # noqa: E402

from tests.create_test_data import DATASETS  # noqa: E402


def _declared_owners(relative: str) -> list[str]:
    """Names of the datasets in create_test_data.py that claim ``relative``."""
    return [
        dataset.name
        for dataset in DATASETS
        if any(
            relative == owned.as_posix() or relative.startswith(f"{owned.as_posix()}/")
            for owned in dataset.owns
        )
    ]


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> set[str]:
    """Everything the synthetic generator writes, relative to its root."""
    root = tmp_path_factory.mktemp("generated")
    return {path.relative_to(root).as_posix() for path in generator.generate_all(root, quiet=True)}


def test_the_generator_writes_nothing_production_owns(generated: set[str]) -> None:
    """The defect itself: a path both producers write is a path with no owner."""
    contended = sorted(generated & generator.PRODUCTION_SOURCED)
    assert not contended, (
        f"{contended} are written by tests/generate_test_microstructure.py and also "
        "declared as production-sourced. Whichever producer runs last wins, which is "
        "how 19,361 real ES bars became 345 synthetic ones."
    )


@pytest.mark.parametrize("relative", sorted(generator.PRODUCTION_SOURCED))
def test_every_production_sourced_path_has_exactly_one_builder(relative: str) -> None:
    """Declaring a path out of the generator is only half of it.

    Removing it there and declaring it nowhere else leaves a fixture file that no
    builder reproduces, which is the same defect pointed the other way.
    """
    owners = _declared_owners(relative)
    assert len(owners) == 1, (
        f"{relative} is claimed by {owners or 'no dataset'} in create_test_data.py; "
        "it needs exactly one."
    )


def test_no_generated_path_is_also_claimed_in_create_test_data(generated: set[str]) -> None:
    """The other direction: the two producers must partition, not merely not collide."""
    doubly_owned = {rel: owners for rel in sorted(generated) if (owners := _declared_owners(rel))}
    assert not doubly_owned, (
        f"tests/generate_test_microstructure.py writes {sorted(doubly_owned)}, which "
        f"create_test_data.py also claims: {doubly_owned}."
    )


def test_the_guard_is_what_stops_the_write(tmp_path, monkeypatch) -> None:
    """Negative selftest.

    Without it the three tests above would pass just as well on a generator that had
    stopped writing those paths for some unrelated reason - a renamed directory, a
    deleted block - and the declaration would be decorative. Empty the declaration and
    the writes have to come back.
    """
    monkeypatch.setattr(generator, "PRODUCTION_SOURCED", frozenset())
    written = {
        path.relative_to(tmp_path).as_posix()
        for path in generator.generate_all(tmp_path, quiet=True)
    }

    missing = sorted(
        path
        for path in (
            "equities/market/microstructure/nasdaq_itch/messages/A/part-000000.parquet",
            "equities/market/microstructure/nasdaq_itch/messages/P/part-000000.parquet",
            "equities/market/microstructure/nasdaq_itch/messages/R/part-000000.parquet",
            "futures/market/individual/ES/data.parquet",
        )
        if path not in written
    )
    assert not missing, (
        f"With the declaration emptied the generator still does not write {missing}, so "
        "the declaration is not what is holding those paths back and these tests prove "
        "nothing about it."
    )
