"""The us_firm_characteristics sweep must have something to sweep, and its reader must
survive finding nothing.

`cs-us_firm_characteristics` was red on `main` from #828 until #1075. Four notebooks
failed, and the reported errors named a polars dtype and two missing upstream results,
none of which was the defect. The sweep in `11_backtest` had been running a
`19 predictions x 0 schemes` grid: `get_entry_schemes_for` drops every declared `k` that
the cross-section cannot realize, the CI universe was five symbols, and every member of
`backtest.sweep.top_k_grid` needs at least ten. The sweep therefore registered nothing,
in zero seconds, reporting zero failures, and each later section read a registry that run
had never written to.

Two things are pinned here, because either one alone leaves the failure reachable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest
import yaml

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.registry.store import _open_registry
from case_studies.utils.sweep_config import get_entry_schemes_for, load_sweep
from tests.pm_helpers import invocations_for

CASE_STUDY = "us_firm_characteristics"
REPO_ROOT = Path(__file__).parent.parent
SWEEPING_NOTEBOOK = f"case_studies/{CASE_STUDY}/11_backtest"


def _overrides() -> dict:
    return yaml.safe_load((REPO_ROOT / "tests" / "overrides.yaml").read_text())


def _declared_labels() -> list[str]:
    return sorted(load_sweep(CASE_STUDY).get("top_k_grid") or {})


# ---------------------------------------------------------------------------
# 1. The configured universe must admit a scheme.
# ---------------------------------------------------------------------------


def test_the_ci_universe_admits_an_entry_scheme_for_every_declared_label():
    """The parameters CI runs the sweep under must leave at least one scheme standing.

    This is the assertion that fails on the pre-#1075 tree: `MAX_SYMBOLS: 5` against a
    `top_k_grid` of [5, 10, 20, 50] yields zero schemes for every label.

    Read per invocation rather than off one `parameters` block. The notebook now runs
    once per declared label, each with its own `LABEL` and its own universe cap, so the
    question this asks - can THIS run realize a concentration - is asked of the pair the
    run is actually given. `tests/test_invocation_fanout.py` is what holds those runs to
    the declared labels; here they are taken as given and each is checked.
    """
    runs = invocations_for(_overrides()[SWEEPING_NOTEBOOK], key=SWEEPING_NOTEBOOK)
    assert _declared_labels(), "no top_k_grid declared; this test is guarding nothing"
    starved = {}
    for run in runs:
        label = run.parameters.get("LABEL") or _declared_labels()[0]
        max_symbols = run.parameters["MAX_SYMBOLS"]
        if not get_entry_schemes_for(CASE_STUDY, label, max_symbols, long_short=True):
            starved[label] = max_symbols
    assert not starved, (
        f"no feasible entry scheme for {sorted(starved)} at the universe cap each run "
        f"declares ({starved}). The sweep would register nothing and every section after "
        f"it would read an empty registry."
    )


def test_a_universe_below_the_smallest_declared_k_is_what_empties_the_grid():
    """The premise of the test above: the filter, not the grid, is what can empty it.

    Without this, raising MAX_SYMBOLS could look like a fix for a grid that was simply
    never declared, and the first test would pass for the wrong reason.
    """
    label = _declared_labels()[0]
    smallest_k = min(int(k) for k in load_sweep(CASE_STUDY)["top_k_grid"][label])
    # Long-short needs two disjoint legs, so the universe must hold 2k names.
    assert get_entry_schemes_for(CASE_STUDY, label, 2 * smallest_k - 1, long_short=True) == []
    assert get_entry_schemes_for(CASE_STUDY, label, 2 * smallest_k, long_short=True) != []


# ---------------------------------------------------------------------------
# 2. Reading a registry that holds nothing must not fail on a dtype.
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_case_dir(tmp_path: Path) -> Path:
    """A real registry, migrated by the real opener, holding no backtest runs."""
    case_dir = tmp_path / CASE_STUDY
    (case_dir / "run_log").mkdir(parents=True)
    conn = _open_registry(case_dir)
    try:
        assert conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] == 0
    finally:
        conn.close()
    return case_dir


def test_the_positional_read_the_notebooks_used_raises_on_an_empty_registry():
    """Why `specs()` exists. polars types an empty column as Null, and json_path_match
    demands String, so the old hand-rolled read raised several cells after the sweep."""
    rows: list[tuple[str, str]] = []
    frame = pl.DataFrame(rows, schema=["backtest_hash", "spec_json"], orient="row")
    assert frame["spec_json"].dtype == pl.Null
    with pytest.raises(Exception, match="dtype"):
        frame.with_columns(pl.col("spec_json").str.json_path_match("$.strategy.signal.top_k"))


def test_specs_returns_string_columns_when_no_run_matches(empty_case_dir: Path):
    specs = BacktestExplorer(CASE_STUDY, case_dir=empty_case_dir).specs("signal")
    assert specs.is_empty()
    assert specs.schema["backtest_hash"] == pl.Utf8
    assert specs.schema["stage"] == pl.Utf8
    assert specs.schema["spec_json"] == pl.Utf8


def test_the_notebooks_json_extraction_survives_an_empty_specs_read(empty_case_dir: Path):
    """The two expressions `11_backtest` and `12_portfolio_management` apply, on nothing."""
    specs = BacktestExplorer(CASE_STUDY, case_dir=empty_case_dir).specs(["signal", "allocation"])
    out = specs.with_columns(
        allocator=pl.col("spec_json").str.json_path_match("$.strategy.allocation.method"),
        names_per_side=pl.col("spec_json")
        .str.json_path_match("$.strategy.signal.top_k")
        .cast(pl.Int64),
    ).drop("spec_json")
    assert out.is_empty()
    assert out.schema["names_per_side"] == pl.Int64


def test_specs_reads_back_the_spec_of_a_registered_run(empty_case_dir: Path):
    """The populated path, so the empty-frame tests cannot pass by returning empty always."""
    conn = sqlite3.connect(str(empty_case_dir / "run_log" / "registry.db"))
    try:
        conn.execute(
            "INSERT INTO backtest_runs "
            "(backtest_hash, prediction_hash, stage, spec_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "abc123",
                "pred1",
                "signal",
                '{"strategy": {"signal": {"top_k": 7}}}',
                "2026-09-07T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    specs = BacktestExplorer(CASE_STUDY, case_dir=empty_case_dir).specs("signal")
    assert specs.height == 1
    assert specs["backtest_hash"][0] == "abc123"
    names = specs.with_columns(
        names_per_side=pl.col("spec_json")
        .str.json_path_match("$.strategy.signal.top_k")
        .cast(pl.Int64)
    )["names_per_side"][0]
    assert names == 7


def test_specs_does_not_return_a_stage_it_was_not_asked_for(empty_case_dir: Path):
    conn = sqlite3.connect(str(empty_case_dir / "run_log" / "registry.db"))
    try:
        conn.executemany(
            "INSERT INTO backtest_runs "
            "(backtest_hash, prediction_hash, stage, spec_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("s1", "p1", "signal", "{}", "2026-09-07T00:00:00Z"),
                ("a1", "p1", "allocation", "{}", "2026-09-07T00:00:00Z"),
                ("h1", "p1", "holdout", "{}", "2026-09-07T00:00:00Z"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    explorer = BacktestExplorer(CASE_STUDY, case_dir=empty_case_dir)
    assert sorted(explorer.specs("signal")["backtest_hash"].to_list()) == ["s1"]
    assert sorted(explorer.specs(["signal", "allocation"])["stage"].to_list()) == [
        "allocation",
        "signal",
    ]
