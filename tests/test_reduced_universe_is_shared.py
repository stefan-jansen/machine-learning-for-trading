"""Every stage of a reduced case-study run covers the same symbols.

`tests/overrides.yaml` injects `MAX_SYMBOLS: 5` into `nasdaq100_microstructure`
stages 01-05 so the pipeline runs small under CI. The stages have to agree on
which five, or the labels and financial features cover one set while the temporal
features cover another, and a symbol only one side chose joins to null features -
a wrong answer that runs clean rather than a failure.

Measured on this fixture before the rules were unified: `02_labels` and
`03_financial_features` reduced through the loader to a seeded random sample,
{AAPL, AMD, CMCSA, CSCO, SIRI}, while `04_model_based_features` took the five
most-observed, {AAPL, AMD, AMZN, FB, TSLA}. Three of five had no temporal
features at all.

The rules are one rule now (`utils.data_quality.top_entities`), which
`tests/test_data_quality.py` pins without data. What needs the real fixture is the
second half: `04` and `05` still select the most-observed symbols with their own
inline expression and no tie break, so they agree with the shared rule only while
no tie straddles the cut. On this fixture five symbols sit at exactly 136,140 bars
and the sixth at 136,131, so the cut at five is clear - and this test is what says
so out loud, and fails on the day a regenerated fixture moves it.

Needs the test-data checkout, so it runs in test-unit-data.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
CASE_STUDY = "nasdaq100_microstructure"
REDUCED_STAGES = ("02_labels", "03_financial_features", "04_model_based_features", "05_evaluation")


def _injected_max_symbols(stage: str) -> int | None:
    overrides = yaml.safe_load((REPO_ROOT / "tests" / "overrides.yaml").read_text()) or {}
    entry = overrides.get(f"case_studies/{CASE_STUDY}/{stage}") or {}
    value = (entry.get("parameters") or {}).get("MAX_SYMBOLS")
    return int(value) if value else None


@pytest.fixture(scope="module")
def bar_counts(test_data_dir: Path) -> pl.DataFrame:
    """Rows per symbol on the panel stages 02, 03 and 04 all read.

    Resolved through conftest's ``test_data_dir`` rather than by reading
    ML4T_DATA_PATH here: that variable is rewritten during a session by the same
    fixture, and this is a statement about the CI fixture. Against the production
    panel - 123 symbols where the fixture has 12 - the cut lands somewhere else
    entirely and the question is not the one being asked.
    """
    if not (test_data_dir / "manifest.json").is_file():
        pytest.skip(f"{test_data_dir} is not a test-data checkout (no manifest.json)")
    hive = test_data_dir / "equities" / "market" / "nasdaq100" / "minute_bars"
    if not hive.is_dir() or not list(hive.glob("year=*")):
        pytest.skip(f"no nasdaq100 minute bars under {hive}")

    setup = yaml.safe_load(
        (REPO_ROOT / "case_studies" / CASE_STUDY / "config" / "setup.yaml").read_text()
    )
    universe = sorted(setup["universe"]["symbols"])
    return (
        pl.scan_parquet(hive / "**/*.parquet", hive_partitioning=True)
        .filter(pl.col("symbol").is_in(universe))
        .filter(pl.col("date").is_between(pl.date(2020, 1, 1), pl.date(2021, 12, 31)))
        .group_by("symbol")
        .len()
        .collect()
    )


def test_the_overrides_still_reduce_these_stages() -> None:
    """Guards the tests below: with no injection they would assert nothing."""
    injected = {stage: _injected_max_symbols(stage) for stage in REDUCED_STAGES}
    assert all(injected.values()), f"no MAX_SYMBOLS injected for {injected}"


def test_every_reduced_stage_is_given_the_same_size() -> None:
    """Equal counts are not sufficient, but unequal ones are already a split."""
    sizes = {_injected_max_symbols(stage) for stage in REDUCED_STAGES}
    assert len(sizes) == 1, f"stages 02-05 are reduced to different sizes: {sizes}"


def test_the_loader_and_an_untied_top_by_count_select_the_same_symbols(
    bar_counts: pl.DataFrame,
) -> None:
    """02 and 03 reduce through the loader; 04 has its own untied expression."""
    max_symbols = _injected_max_symbols("04_model_based_features")

    shared = sorted(
        bar_counts.sort(["len", "symbol"], descending=[True, False])
        .head(max_symbols)["symbol"]
        .to_list()
    )
    untied = sorted(bar_counts.sort("len", descending=True).head(max_symbols)["symbol"].to_list())
    assert shared == untied, (
        "the shared reduction and the stage-04 expression choose different symbols; "
        "stage 04 has to break its tie on the symbol name"
    )


def test_no_tie_straddles_the_cut(bar_counts: pl.DataFrame) -> None:
    """What makes the untied expression safe, stated as the property it needs."""
    max_symbols = _injected_max_symbols("04_model_based_features")
    ordered = bar_counts.sort(["len", "symbol"], descending=[True, False])
    assert ordered.height > max_symbols, "the fixture carries no more symbols than the cut"

    last_kept = ordered["len"][max_symbols - 1]
    first_dropped = ordered["len"][max_symbols]
    assert last_kept > first_dropped, (
        f"symbols tie across the cut at {max_symbols} ({last_kept} bars on both sides), so an "
        "expression that does not break ties by name can pick either one"
    )
