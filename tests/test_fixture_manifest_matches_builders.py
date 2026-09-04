"""data/manifest.json describes the fixture set, and nothing was checking that.

`tests/create_test_data.py` writes each dataset's declared budget into the
manifest at build time. Nothing compared the committed manifest against the
builder afterwards, so a builder change that was not followed by a regeneration
left the fixture silently describing data that is not on disk.

It cost a case study a day. The manifest still said `sp500_options` kept 3 "most
liquid" underlyings while the builder declared 30 named ones from #652 - a
different subsampling scheme entirely, never regenerated. Downstream,
`cs-sp500_options` failed five notebooks on main; every one was the stale
fixture, and the failure text pointed at the notebook.

Two halves, and the second is why this file is not one assertion:

- the data-free half compares the manifest's `subsets` against the declarations,
  which is the check that would have fired the day #652 landed;
- the data-backed half compares the declarations against the fixture on disk,
  which is what the first half cannot see. Measured 2026-09-03: the manifest and
  the `nasdaq100_minute_bars` builder agreed with each other on 6 symbols while
  the fixture carried 12, so manifest-equals-builder alone would have passed on a
  fixture neither of them described.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.create_test_data import (  # noqa: E402
    DATASETS,
    SUPERSEDED_SUBSETS,
    UNBUILT_FIXTURES,
    declared_subsets,
)


def _fixture_root() -> Path:
    return Path(os.environ.get("ML4T_DATA_PATH", ""))


def _manifest() -> dict:
    path = _fixture_root() / "manifest.json"
    if not path.is_file():
        pytest.skip(f"no fixture manifest at {path}")
    return json.loads(path.read_text())


# --- the declarations themselves, no data needed -----------------------------


def test_every_fixture_is_declared_exactly_once() -> None:
    """A dataset with a builder and a declaration of having none is a contradiction."""
    registered = {dataset.name for dataset in DATASETS}
    unbuilt = {fixture.name for fixture in UNBUILT_FIXTURES}
    assert not registered & unbuilt
    assert not (registered | unbuilt) & set(SUPERSEDED_SUBSETS)


def test_every_superseded_entry_names_the_dataset_that_owns_it() -> None:
    registered = {dataset.name for dataset in DATASETS}
    for name, owner in SUPERSEDED_SUBSETS.items():
        assert owner in registered, f"{name} is superseded by {owner}, which has no builder"


def test_every_unbuilt_fixture_records_why_it_has_none() -> None:
    """Without the reason the entry is a shrug, which is what it replaced."""
    for fixture in UNBUILT_FIXTURES:
        assert fixture.reason.strip(), f"{fixture.name} declares no reason"
        assert fixture.entities, f"{fixture.name} declares nothing a test can measure"


def test_the_nasdaq_builder_and_its_budget_name_the_same_symbols() -> None:
    """The builder's literals and the budget it writes cannot drift apart."""
    from tests.create_test_data import NASDAQ100_MINUTE_SYMBOLS

    budget = declared_subsets()["nasdaq100_minute_bars"]
    assert budget["symbols"] == list(NASDAQ100_MINUTE_SYMBOLS)


# --- the manifest against the declarations -----------------------------------


def test_the_manifest_subsets_are_exactly_the_declared_ones() -> None:
    """Both directions: an undeclared entry and a missing one are the same defect."""
    assert sorted(_manifest()["subsets"]) == sorted(declared_subsets())


@pytest.mark.parametrize("name", sorted(declared_subsets()))
def test_the_manifest_records_what_the_declaration_says(name: str) -> None:
    assert _manifest()["subsets"][name] == declared_subsets()[name], (
        f"data/manifest.json's {name} entry has drifted from tests/create_test_data.py. "
        "Regenerate the dataset, or rerun with --reconcile-manifest if only the "
        "declaration moved."
    )


def test_no_superseded_entry_came_back() -> None:
    present = set(_manifest()["subsets"]) & set(SUPERSEDED_SUBSETS)
    assert not present, f"{sorted(present)} describe fixtures another dataset already owns"


# --- the declarations against the fixture on disk ----------------------------


def test_every_file_the_manifest_lists_is_on_disk() -> None:
    """133 of 176 entries pointed at the pre-reorganization layout before this ran.

    A manifest listing files that are not there is the same wrong answer as a stale
    budget: it is the only record of what the fixture set is supposed to contain.
    """
    root = _fixture_root()
    missing = [rel for rel in _manifest()["files"] if not (root / rel).is_file()]
    assert not missing, f"{len(missing)} manifest paths are not on disk, e.g. {missing[:5]}"


@pytest.mark.parametrize(
    ("name", "rel", "column", "count"),
    [
        (fixture.name, rel, column, count)
        for fixture in UNBUILT_FIXTURES
        for rel, (column, count) in fixture.entities.items()
    ],
)
def test_an_unbuilt_fixture_carries_the_universe_it_declares(
    name: str, rel: str, column: str, count: int
) -> None:
    """These cannot drift against a builder, so they are measured against the data.

    Every one of them declared a budget the fixture did not satisfy before this -
    "15 most liquid ETFs" over 56 symbols, "8 major/cross FX pairs" over 20, "50
    most liquid US equities" over 56, "8 most liquid CME products" over a 30-product
    daily panel.
    """
    path = _fixture_root() / rel
    if not path.is_file():
        pytest.skip(f"{name}: {rel} is not in this checkout")
    observed = pl.scan_parquet(path).select(pl.col(column).n_unique()).collect().item()
    assert observed == count, (
        f"{name} declares {count} distinct {column} in {rel} and the fixture carries {observed}"
    )
