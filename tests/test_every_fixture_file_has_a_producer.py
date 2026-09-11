"""Most of the fixture had no producer, and no test could see that.

`tests/test_fixture_manifest_matches_builders.py` checks every *declared* dataset
against its builder and against the data on disk.
`tests/test_one_producer_per_fixture_file.py` checks that the two producers do not
write the same path. Neither looks at the fixture root and asks which files no
producer accounts for, so a file could sit in the fixture for a year with nothing
able to rebuild it and every test still pass - which is how 149 of 327 files got
there.

A file with no producer cannot be regenerated when production moves, cannot be
checked against production, and cannot be explained: three of them turned out to be
byte-identical copies of each other at pre-migration paths that nothing reads, and
one holds the whole FinancialPhraseBank corpus under a filename that promises the
unanimous subset.

`UNPRODUCED` is the remaining backlog, and it is a ratchet: a new fixture file with
no producer fails immediately, and a file that gains one has to leave the list. It
only shrinks.

Two producers are declared. `create_test_data.py` derives from production and each
`Dataset` names what it owns; `generate_test_microstructure.py` is synthetic and
`generate_all` returns what it writes. A third, `generate_skip_data.py`, declares
nothing at all - its outputs are in `UNPRODUCED` below, and one of them,
`enrich_adv_columns`, writes back into two files `create_test_data.py` owns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import generate_test_microstructure as generator  # noqa: E402

from tests.create_test_data import DATASETS  # noqa: E402

# Fixture files that no declared producer writes. Grouped by top-level directory,
# which is mechanical; the triage of what each one needs is on the tracker, because
# it is a judgement that goes stale and a comment here would not be re-checked.
UNPRODUCED = frozenset(
    {
        # academic/ - 3
        "academic/firm_characteristics_all.parquet",
        "academic/firm_characteristics_test.parquet",
        "academic/firm_characteristics_train.parquet",
        # alternative/ - 4
        "alternative/institutional/13f_expanded/institutional_holdings.parquet",
        "alternative/institutional/13f_expanded/stock_features.parquet",
        "alternative/text/financial_phrasebank/sentences_allagree.parquet",
        "alternative/text/sp500_10q_mda.parquet",
        # autonomous_agents/ - 3
        "autonomous_agents/operator_artifacts/run_20260504T201005.json",
        "autonomous_agents/operator_artifacts/run_etfs_20260504T223150.json",
        "autonomous_agents/operator_artifacts/run_us_firm_characteristics_20260504T225521.json",
        # crypto/ - 6
        "crypto/onchain/coingecko_ethereum.parquet",
        "crypto/onchain/defillama_tvl_arbitrum.parquet",
        "crypto/onchain/defillama_tvl_bsc.parquet",
        "crypto/onchain/defillama_tvl_ethereum.parquet",
        "crypto/onchain/defillama_tvl_solana.parquet",
        "crypto/onchain/defillama_tvl_total.parquet",
        # equities/ - 9
        "equities/market/microstructure/iex/deep/parsed/path_signatures/data.parquet",
        "equities/market/microstructure/nasdaq100_taq/data.parquet",
        "equities/market/microstructure/nasdaq_itch/messages/enriched/C.parquet",
        "equities/market/microstructure/nasdaq_itch/messages/enriched/E.parquet",
        "equities/market/microstructure/nasdaq_itch/messages/enriched/X.parquet",
        "equities/market/sp500/options_eda/year=2019.parquet",
        "equities/market/sp500/options_eda/year=2020.parquet",
        "equities/market/sp500/sp500.csv",
        "equities/positioning/13f/bulk/2024Q3/institutional_holdings.parquet",
        # factors/ - 38
        "factors/aqr/README.md",
        "factors/aqr/_profile.json",
        "factors/aqr/bab/bab_factors_monthly.parquet",
        "factors/aqr/bab/bab_usa_monthly.parquet",
        "factors/aqr/bab/docs/bab_documentation.pdf",
        "factors/aqr/bab_factors.parquet",
        "factors/aqr/bab_factors_daily.parquet",
        "factors/aqr/century_premia.parquet",
        "factors/aqr/credit_premium.parquet",
        "factors/aqr/esg_frontier.parquet",
        "factors/aqr/hml_devil.parquet",
        "factors/aqr/hml_devil_daily.parquet",
        "factors/aqr/metadata.json",
        "factors/aqr/qmj/docs/qmj_documentation.pdf",
        "factors/aqr/qmj/qmj_factors_monthly.parquet",
        "factors/aqr/qmj/qmj_usa_monthly.parquet",
        "factors/aqr/qmj_6_portfolios.parquet",
        "factors/aqr/qmj_factors.parquet",
        "factors/aqr/qmj_factors_daily.parquet",
        "factors/aqr/tsmom.parquet",
        "factors/aqr/vme/docs/vme_documentation.pdf",
        "factors/aqr/vme/vme_factors_monthly.parquet",
        "factors/aqr/vme/vme_global_monthly.parquet",
        "factors/aqr/vme_factors.parquet",
        "factors/aqr/vme_portfolios.parquet",
        "factors/factor_summary_stats.csv",
        "factors/fama-french/_profile.json",
        "factors/fama-french/bp_me_monthly.parquet",
        "factors/fama-french/ff3_daily.parquet",
        "factors/fama-french/ff3_developed_monthly.parquet",
        "factors/fama-french/ff3_monthly.parquet",
        "factors/fama-french/ff5_daily.parquet",
        "factors/fama-french/ff5_monthly.parquet",
        "factors/fama-french/ind_5_monthly.parquet",
        "factors/fama-french/mom_daily.parquet",
        "factors/fama-french/mom_monthly.parquet",
        "factors/fama-french/port_size_bm_25_monthly.parquet",
        "factors/fama-french/port_size_monthly.parquet",
        # institutional/ - 1
        "institutional/13f/institutional_holdings.parquet",
        # macro/ - 7
        "macro/fred_macro.parquet",
        "macro/fred_macro_dictionary.parquet",
        "macro/fred_macro_initial_release.parquet",
        "macro/fred_macro_initial_release_raw.parquet",
        "macro/fred_macro_metadata.parquet",
        "macro/fred_macro_raw.parquet",
        "macro/fred_macro_raw_dictionary.parquet",
        # prediction_markets/ - 1
        "prediction_markets/polymarket_events.parquet",
        # sec_filings/ - 6
        "sec_filings/sp100/10k/AAPL/2023.parquet",
        "sec_filings/sp100/10k/AAPL/2024.parquet",
        "sec_filings/sp100/10k/GOOG/2023.parquet",
        "sec_filings/sp100/10k/GOOG/2024.parquet",
        "sec_filings/sp100/10k/MSFT/2023.parquet",
        "sec_filings/sp100/10k/MSFT/2024.parquet",
    }
)


def _owned_by_a_dataset(on_disk: set[str]) -> set[str]:
    """The files `create_test_data.py` claims, with directory entries expanded.

    Expanded against what is on disk rather than listed: `Dataset.owns` may name a
    directory, and the point of the check is which real files are covered.
    """
    covered: set[str] = set()
    for dataset in DATASETS:
        for owned in dataset.owns:
            prefix = owned.as_posix()
            covered |= {path for path in on_disk if path == prefix or path.startswith(f"{prefix}/")}
    return covered


@pytest.fixture(scope="module")
def fixture_root(test_data_dir: Path) -> Path:
    """The test-data checkout, or a skip.

    Production has no manifest, and against production these declarations are a
    category error rather than a failure.
    """
    if not (test_data_dir / "manifest.json").is_file():
        pytest.skip(f"{test_data_dir} is not a test-data checkout (no manifest.json)")
    return test_data_dir


@pytest.fixture(scope="module")
def on_disk(fixture_root: Path) -> set[str]:
    """Every fixture data file, relative to the root. The manifest describes, so it
    is not itself a fixture file."""
    return {
        path.relative_to(fixture_root).as_posix()
        for path in fixture_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }


@pytest.fixture(scope="module")
def produced(on_disk: set[str], tmp_path_factory) -> set[str]:
    """Everything a declared producer writes."""
    root = tmp_path_factory.mktemp("generated")
    generated = {
        path.relative_to(root).as_posix() for path in generator.generate_all(root, quiet=True)
    }
    return _owned_by_a_dataset(on_disk) | generated


def test_every_fixture_file_has_a_producer(on_disk: set[str], produced: set[str]) -> None:
    """The check itself. A new fixture file with no builder fails here."""
    orphans = sorted(on_disk - produced - UNPRODUCED)
    assert not orphans, (
        f"{len(orphans)} fixture files are written by no declared producer and are not "
        f"in UNPRODUCED: {orphans}. Declare each in tests/create_test_data.py as a "
        "Dataset, or write it from tests/generate_test_microstructure.py. Adding it to "
        "UNPRODUCED is not the remedy: that list only shrinks."
    )


def test_no_exempt_path_has_gained_a_producer(produced: set[str]) -> None:
    """The ratchet. Declaring a file is only half of retiring it from the backlog."""
    retired = sorted(UNPRODUCED & produced)
    assert not retired, (
        f"{retired} now have a producer and must be removed from UNPRODUCED. Left in "
        "place the list stops measuring the backlog and starts hiding a regression."
    )


def test_no_exempt_path_has_left_the_fixture(on_disk: set[str]) -> None:
    """A deleted file leaves the list too, so it never grants a future file cover."""
    gone = sorted(UNPRODUCED - on_disk)
    assert not gone, f"{gone} are in UNPRODUCED and not in the fixture; remove them from it."


def test_the_producers_are_what_cover_the_rest(on_disk: set[str], produced: set[str]) -> None:
    """Negative selftest.

    The three tests above would pass on a `produced` that resolved to nothing, as
    long as UNPRODUCED happened to list the whole fixture - and would pass just as
    well if `Dataset.owns` expansion silently matched no files, which is the failure
    mode a path-prefix match invites. Both make the check decorative. So: the
    producers must account for the fixture that UNPRODUCED does not, exactly.
    """
    assert produced, "no declared producer resolved to any file on disk"
    uncovered = on_disk - UNPRODUCED
    assert uncovered <= produced
    assert len(uncovered) > len(UNPRODUCED), (
        f"{len(uncovered)} of {len(on_disk)} fixture files have a producer against "
        f"{len(UNPRODUCED)} that do not; the backlog is no longer the minority and "
        "this test is the wrong shape for it."
    )
