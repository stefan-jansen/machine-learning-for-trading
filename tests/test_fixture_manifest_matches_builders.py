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
import sys
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.create_test_data import (  # noqa: E402
    CRYPTO_PERPS,
    CRYPTO_PERPS_END,
    DATASETS,
    ETF_UNIVERSE,
    ETF_UNIVERSE_UNADJUSTED,
    FX_4H,
    FX_DAILY,
    SUPERSEDED_SUBSETS,
    declared_subsets,
)


@pytest.fixture
def fixture_root(test_data_dir: Path) -> Path:
    """The test-data checkout, or a skip.

    Resolved through conftest's ``test_data_dir`` rather than by reading
    ML4T_DATA_PATH here: that variable is rewritten during a session by the same
    fixture, and under a full-suite run it resolves to the production root, where
    these declarations are a category error rather than a failure. The manifest is
    what distinguishes the two - production has none.
    """
    if not (test_data_dir / "manifest.json").is_file():
        pytest.skip(f"{test_data_dir} is not a test-data checkout (no manifest.json)")
    return test_data_dir


@pytest.fixture
def manifest(fixture_root: Path) -> dict:
    return json.loads((fixture_root / "manifest.json").read_text())


# --- the declarations themselves, no data needed -----------------------------


def test_no_superseded_entry_names_a_registered_dataset() -> None:
    """A superseded manifest key describes a fixture some other dataset owns.

    A name that is both a `Dataset` and a superseded key would be dropped from the
    manifest and written back into it by the same run.
    """
    assert not {dataset.name for dataset in DATASETS} & set(SUPERSEDED_SUBSETS)


def test_every_superseded_entry_names_the_dataset_that_owns_it() -> None:
    registered = {dataset.name for dataset in DATASETS}
    for name, owner in SUPERSEDED_SUBSETS.items():
        assert owner in registered, f"{name} is superseded by {owner}, which has no builder"


def test_every_reducing_builder_declares_something_measurable() -> None:
    """A builder with no `entities` is checked against nothing on the data side.

    institutional_holdings_13f is the exemption and states it: it subsamples on no
    axis, so there is no count to compare.
    """
    unmeasured = {
        dataset.name
        for dataset in DATASETS
        if not dataset.entities and dataset.budget.get("subsample") != "none"
    }
    assert not unmeasured, f"{sorted(unmeasured)} declare no entity count to check"


# --- the manifest against the declarations -----------------------------------


def test_the_manifest_subsets_are_exactly_the_declared_ones(manifest: dict) -> None:
    """Both directions: an undeclared entry and a missing one are the same defect."""
    assert sorted(manifest["subsets"]) == sorted(declared_subsets())


@pytest.mark.parametrize("name", sorted(declared_subsets()))
def test_the_manifest_records_what_the_declaration_says(name: str, manifest: dict) -> None:
    assert manifest["subsets"][name] == declared_subsets()[name], (
        f"data/manifest.json's {name} entry has drifted from tests/create_test_data.py. "
        "Regenerate the dataset, or rerun with --reconcile-manifest if only the "
        "declaration moved."
    )


def test_no_superseded_entry_came_back(manifest: dict) -> None:
    present = set(manifest["subsets"]) & set(SUPERSEDED_SUBSETS)
    assert not present, f"{sorted(present)} describe fixtures another dataset already owns"


# --- the declarations against the fixture on disk ----------------------------


def test_every_file_the_manifest_lists_is_on_disk(fixture_root: Path, manifest: dict) -> None:
    """133 of 176 entries pointed at the pre-reorganization layout before this ran.

    A manifest listing files that are not there is the same wrong answer as a stale
    budget: it is the only record of what the fixture set is supposed to contain.
    """
    missing = [rel for rel in manifest["files"] if not (fixture_root / rel).is_file()]
    assert not missing, f"{len(missing)} manifest paths are not on disk, e.g. {missing[:5]}"


def _distinct(path: Path, column: str) -> int:
    """Distinct values of ``column``, reading a hive directory the way its loader does.

    The CME hourly bars keep `product` in the path (`product=ES/year=2011/...`) and
    not in the parquet, so a non-hive scan of that tree cannot see the entity axis
    at all. `hive_partitioning=True` is a no-op on a tree whose files carry the
    column inline, which is the nasdaq100 minute-bar layout.
    """
    if not path.is_dir():
        return pl.scan_parquet(path).select(pl.col(column).n_unique()).collect().item()
    return (
        pl.scan_parquet(path / "**" / "*.parquet", hive_partitioning=True)
        .select(pl.col(column).n_unique())
        .collect()
        .item()
    )


def _declared_files(root: Path) -> dict[str, list[str]]:
    """{dataset name: the files on disk under the paths it declares}."""
    declared = {dataset.name: dataset.owns for dataset in DATASETS}
    found = {}
    for name, paths in declared.items():
        files = []
        for declared_path in paths:
            path = root / declared_path
            if path.is_file():
                files.append(path.relative_to(root).as_posix())
            elif path.is_dir():
                files += [f.relative_to(root).as_posix() for f in path.rglob("*") if f.is_file()]
        found[name] = sorted(files)
    return found


@pytest.mark.parametrize("name", sorted(d.name for d in DATASETS))
def test_every_file_a_dataset_owns_is_listed_in_the_manifest(
    name: str, fixture_root: Path, manifest: dict
) -> None:
    """The other direction, and the one a recursive glob makes dangerous.

    load_nasdaq100_bars globs '**/*.parquet' under its hive root, so a part left
    behind by an earlier generation is unioned into the panel rather than replaced.
    Manifest-to-disk membership cannot see that; disk-to-manifest can.
    """
    listed = set(manifest["files"])
    unlisted = [rel for rel in _declared_files(fixture_root)[name] if rel not in listed]
    assert not unlisted, (
        f"{name} carries files the manifest does not list: {unlisted[:5]}. Either they are "
        "left over from an earlier generation, or the manifest needs --reconcile-manifest."
    )


@pytest.mark.parametrize(
    ("name", "rel", "column", "count"),
    [
        (dataset.name, rel, column, count)
        for dataset in DATASETS
        for rel, (column, count) in dataset.entities.items()
    ],
)
def test_a_built_fixture_carries_the_universe_its_builder_declares(
    name: str, rel: str, column: str, count: int, fixture_root: Path
) -> None:
    """The builder's own constants against the data, which is where drift shows.

    Measured 2026-09-03: the manifest and the nasdaq100_minute_bars builder agreed
    with each other on 6 symbols while the fixture carried 12, so comparing the two
    proved nothing and the builder, run as it stood, would have narrowed the fixture
    back and left the loader's glob reading two generations at once.
    """
    path = fixture_root / rel
    if not path.exists():
        pytest.skip(f"{name}: {rel} is not in this checkout")
    observed = _distinct(path, column)
    assert observed == count, (
        f"{name}'s builder declares {count} distinct {column} in {rel} and the fixture "
        f"carries {observed}. Regenerate the dataset, or correct the builder."
    )


# --- what a builder does to the time axis ------------------------------------
#
# An entity count cannot see any of this, and the time axis is the whole reason
# three of these builders are more than a filter. `etfs` casts a Datetime at
# midnight down to a Date; `fx` casts a Date and a naive millisecond Datetime up to
# UTC-aware microseconds, and `case_studies/fx_pairs` joins the two frequencies, so
# one representation is not optional. A cast dropped from a builder leaves the row
# counts and the entity counts exactly as they are.

TIMESTAMP_CONTRACT = [
    (ETF_UNIVERSE, pl.Date),
    (ETF_UNIVERSE_UNADJUSTED, pl.Date),
    (FX_4H, pl.Datetime("us", "UTC")),
    (FX_DAILY, pl.Datetime("us", "UTC")),
]


@pytest.mark.parametrize(
    ("rel", "dtype"), [(rel.as_posix(), dtype) for rel, dtype in TIMESTAMP_CONTRACT]
)
def test_a_built_fixture_stores_its_timestamps_the_way_its_builder_says(
    rel: str, dtype: pl.DataType, fixture_root: Path
) -> None:
    path = fixture_root / rel
    if not path.exists():
        pytest.skip(f"{rel} is not in this checkout")
    observed = pl.scan_parquet(path).collect_schema()["timestamp"]
    assert observed == dtype, (
        f"{rel} stores timestamp as {observed} and its builder writes {dtype}. "
        "A Datetime at midnight compares unequal to the Date it prints as."
    )


def test_the_crypto_bars_stop_where_the_builder_bounds_them(fixture_root: Path) -> None:
    """The bound exists so the bars do not outrun the intermediates built on them.

    Removing it would add rows at the end that no label, feature or fold covers -
    which is what ml4t/agent-workspace#970 was, in the cme_futures fixture.
    """
    path = fixture_root / CRYPTO_PERPS
    if not path.exists():
        pytest.skip(f"{CRYPTO_PERPS} is not in this checkout")
    last = pl.scan_parquet(path).select(pl.col("timestamp").max()).collect().item()
    assert last == CRYPTO_PERPS_END, (
        f"{CRYPTO_PERPS} ends at {last} and the builder bounds it at {CRYPTO_PERPS_END}"
    )


# --- what a builder does to the date range -----------------------------------
#
# None of the checks above can see a truncated fixture: a panel cut at either end
# keeps its entity count, its manifest entry and its timestamp type. Five of these
# builders promise the whole production history for the entities they keep, and
# `crypto` promises a declared bound instead, so both are stated as dates and both
# are read off the fixture.
#
# The bounds are the fixture's, measured 2026-09-04, and a production refresh that
# extends a panel is expected to move them. That is the point: the value here has
# to be updated deliberately, by whoever regenerates the fixture.

DATE_RANGE_CONTRACT = [
    ("etfs/market/etf_universe.parquet", "timestamp", "2006-01-03", "2025-12-31"),
    ("etfs/market/etf_universe_unadjusted.parquet", "timestamp", "2006-01-03", "2025-12-31"),
    ("fx/market/4h.parquet", "timestamp", "2011-01-02 14:00:00+00:00", "2025-12-31 18:00:00+00:00"),
    (
        "fx/market/daily.parquet",
        "timestamp",
        "2011-01-03 00:00:00+00:00",
        "2025-12-31 00:00:00+00:00",
    ),
    (
        "crypto/market/perps_1h.parquet",
        "timestamp",
        "2020-01-01 00:00:00+00:00",
        "2025-12-29 23:00:00+00:00",
    ),
    ("equities/market/us_equities/us_equities.parquet", "date", "1962-01-02", "2018-03-27"),
    (
        "futures/market/continuous/daily/continuous_daily.parquet",
        "session_date",
        "2011-01-03",
        "2025-12-31",
    ),
]


@pytest.mark.parametrize(("rel", "column", "first", "last"), DATE_RANGE_CONTRACT)
def test_a_built_fixture_spans_the_dates_its_builder_promises(
    rel: str, column: str, first: str, last: str, fixture_root: Path
) -> None:
    path = fixture_root / rel
    if not path.exists():
        pytest.skip(f"{rel} is not in this checkout")
    frame = pl.scan_parquet(path)
    observed_first = str(frame.select(pl.col(column).min()).collect().item())
    observed_last = str(frame.select(pl.col(column).max()).collect().item())
    assert (observed_first[: len(first)], observed_last[: len(last)]) == (first, last), (
        f"{rel} spans {observed_first} to {observed_last}, and its builder writes "
        f"{first} to {last}. A truncated panel keeps its entity count and its dtype."
    )
