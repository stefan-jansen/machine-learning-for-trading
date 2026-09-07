"""Regression tests for owner-controlled, fail-closed carrier routing."""

from __future__ import annotations

import ast
import re
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils import carrier_pins, strategy_analysis
from case_studies.utils.carrier_pins import (
    CARRIER_PINS,
    carrier_config_name,
    filter_to_carrier_config,
    prioritize_carrier_hash,
)
from case_studies.utils.cohort_reporting import cohort_metric_attribution, reportable_pbo

# A contract test that reads a real case study's pin is testing that case study's
# configuration, not the contract, and it goes red the day an owner re-pins. These
# tests install their own pin against their own tmp registry instead.
FIXTURE_CASE_STUDY = "fixture_case_study"
FIXTURE_PIN = "abcdef012345"


@pytest.fixture
def pinned_case_study(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(CARRIER_PINS, FIXTURE_CASE_STUDY, FIXTURE_PIN)
    carrier_pins._carrier_config_name.cache_clear()
    yield FIXTURE_CASE_STUDY
    carrier_pins._carrier_config_name.cache_clear()


def _pin_db(path: Path, *, backtest_hash: str = f"{FIXTURE_PIN}_suffix") -> None:
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, config_name TEXT);
            CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, training_hash TEXT);
            CREATE TABLE backtest_runs (backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT);
            INSERT INTO training_runs VALUES ('train_us', 'owner_config');
            INSERT INTO prediction_sets VALUES ('pred_us', 'train_us');
            """
        )
        db.execute("INSERT INTO backtest_runs VALUES (?, 'pred_us')", (backtest_hash,))


def _resolver_db(path: Path, *, backtest_hash: str) -> None:
    """A registry the documented rule can select from, carrying one usable backtest.

    Everything `resolve_canonical_rank1_lineage` reads is present and populated, so
    the only reason it can come back empty is the pin.
    """
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, config_name TEXT, family TEXT, label TEXT,
                spec_json TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
            );
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT, spec_json TEXT
            );
            CREATE TABLE backtest_metrics (backtest_hash TEXT PRIMARY KEY, sharpe REAL);
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY, ic_mean REAL, ic_n_days REAL
            );
            INSERT INTO training_runs VALUES ('train_us', 'owner_config', 'gbm', 'fwd_ret_1m', NULL);
            INSERT INTO prediction_sets VALUES ('pred_us', 'train_us', 'validation');
            INSERT INTO fold_metrics VALUES ('pred_us', 0.02);
            INSERT INTO prediction_metrics VALUES ('pred_us', 0.02, 250);
            """
        )
        db.execute(
            "INSERT INTO backtest_runs VALUES (?, 'pred_us', 'signal', '{\"strategy\": {}}')",
            (backtest_hash,),
        )
        db.execute("INSERT INTO backtest_metrics VALUES (?, 1.5)", (backtest_hash,))


def test_carrier_pins_are_single_sourced_and_well_formed() -> None:
    assert strategy_analysis.CARRIER_PINS is CARRIER_PINS
    # Asserting a pin equals its own literal is green whatever the pin resolves to,
    # which is how `us_firm_characteristics` kept a pin matching zero rows of its
    # rebuilt registry while this file stayed passing. What a pin has to be is a
    # lowercase hex prefix long enough to identify one backtest; whether it still
    # resolves is checked against a registry below.
    # An empty mapping is the documented normal state - a case study with no entry
    # selects by the rule - so its emptiness cannot be asserted against. What a
    # deletion would break is the lookup, and that is checked behaviourally below.
    for case_study, pin in CARRIER_PINS.items():
        assert case_study and case_study.strip() == case_study
        assert re.fullmatch(r"[0-9a-f]{8,}", pin), (
            f"carrier pin for {case_study} is {pin!r}; expected a lowercase hex "
            "backtest-hash prefix of at least eight characters"
        )

    repo = Path(__file__).parents[1]
    for relative in (
        "20_strategy_synthesis/holdout.py",
        "20_strategy_synthesis/01_aggregate_synthesis.py",
    ):
        tree = ast.parse((repo / relative).read_text())
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "CARRIER_PINS"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ]
        assert assignments == []


def test_the_carrier_restriction_has_no_second_implementation() -> None:
    """One rule, and where it is not applied, no machinery pretending to apply it.

    `_apply_carrier_pin` was defined twice with different signatures - a mapping lookup in
    `20_strategy_synthesis/01_aggregate_synthesis.py` and a predicate argument in
    `case_studies/utils/paired_metrics.py` - and both had become unreachable: the mapping
    was permanently empty and no caller anywhere supplied the predicate. Two dead
    implementations of one rule is how the config-name copy came to select
    `us_firm_characteristics`' weakest advanced configuration for a whole registry rebuild
    while every notebook inside that case study reported its best.

    Where an owner does pin a carrier, the live mechanism is
    `carrier_pins.carrier_config_name`, which resolves the pin against the registry and
    raises when it matches nothing - checked behaviourally elsewhere in this file. What is
    checked here is that nothing has grown a second one beside it.

    Read by parsing: `01_aggregate_synthesis.py` is a notebook, and importing one runs it.
    """
    repo = Path(__file__).parents[1]
    for relative in (
        "20_strategy_synthesis/01_aggregate_synthesis.py",
        "case_studies/utils/paired_metrics.py",
    ):
        tree = ast.parse((repo / relative).read_text())
        defined = sorted(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and "carrier_pin" in node.name
        )
        assert defined == [], (
            f"{relative} defines {defined}; the carrier restriction has one implementation, "
            "in case_studies/utils/carrier_pins.py, and a second copy is what drifted"
        )


def test_the_selection_restrictions_are_declared_once() -> None:
    """`holdout.py` must import each selection restriction, not declare its own copy.

    `case_studies/utils/strategy_analysis.py` and `20_strategy_synthesis/holdout.py`
    each used to declare `LABEL_RESTRICTIONS` and `UNIVERSE_RESTRICTIONS`, under a
    "keep these in sync" comment where a mechanism should be. A comment is not a
    mechanism: the same arrangement one directory over - `_CARRIER_PIN_PREDICATES`
    hand-copying a carrier choice under a "keep in sync" note - had been out of sync
    across a whole registry rebuild with nothing failing. A drift check was the earlier
    answer here and it only ever asked whether two values agreed today; there is now one
    value, and this asks that the second declaration has not come back.

    Read by parsing rather than by importing, because `holdout.py`'s module scope reaches
    lightgbm and torch, which this job does not install.
    `tests/test_holdout_selection_is_single_sourced.py` asserts the runtime identity in
    the job that does.
    """
    tree = ast.parse(
        (Path(__file__).parents[1] / "20_strategy_synthesis" / "holdout.py").read_text()
    )
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "case_studies.utils.strategy_analysis"
        for alias in node.names
    }
    for name in ("LABEL_RESTRICTIONS", "UNIVERSE_RESTRICTIONS"):
        declared = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ]
        assert not declared, (
            f"20_strategy_synthesis/holdout.py declares its own {name} again. One "
            "declaration, in case_studies/utils/strategy_analysis.py, imported here - a "
            "second copy is a restriction a case study declares and the holdout selector "
            "does not read."
        )
        assert name in imported, (
            f"20_strategy_synthesis/holdout.py neither declares nor imports {name}, so "
            "whatever it applies to holdout selection is not what the case study declared."
        )


def test_owner_pin_resolves_without_copying_config_name(
    tmp_path: Path, pinned_case_study: str
) -> None:
    db_path = tmp_path / "registry.db"
    _pin_db(db_path)
    assert carrier_config_name(pinned_case_study, db_path) == "owner_config"
    candidates = pl.DataFrame({"config_name": ["other", "owner_config"], "sharpe": [3.0, 2.9]})
    result = filter_to_carrier_config(
        candidates,
        pinned_case_study,
        db_path=db_path,
    )
    assert result["config_name"].to_list() == ["owner_config"]


def test_a_pin_that_matches_no_backtest_is_named_as_the_cause(
    tmp_path: Path, pinned_case_study: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale pin must say so, rather than read as a label-restriction problem.

    The registry below holds a validation backtest that the documented rule would
    select happily; the only thing wrong with it is that the pin names a hash it
    does not carry. Before this check the resolver reported "No validation rank-1
    candidate ... (label_filter=None)", which sends the reader to
    LABEL_RESTRICTIONS - a mapping that had nothing to do with it.
    """
    case_dir = tmp_path / pinned_case_study
    (case_dir / "run_log").mkdir(parents=True)
    db_path = case_dir / "run_log" / "registry.db"
    _resolver_db(db_path, backtest_hash="0123456789ab_not_the_pin")
    monkeypatch.setattr(
        "utils.paths.get_case_study_dir", lambda case_study, **_: tmp_path / case_study
    )

    with pytest.raises(RuntimeError, match=f"Carrier pin {FIXTURE_PIN!r}"):
        strategy_analysis.resolve_canonical_rank1_lineage(pinned_case_study)


def test_the_lookup_reads_the_mapping_rather_than_a_copy(pinned_case_study: str) -> None:
    """What an empty mapping cannot be asserted against, and a deletion would break.

    The mapping is empty whenever no owner has a reason to pin, so its contents say
    nothing about whether the lookup still consults it. Installing an entry and
    reading it back does, and it fails the day `carrier_pin` starts answering from
    somewhere else.
    """
    assert carrier_pins.carrier_pin(pinned_case_study) == FIXTURE_PIN
    assert carrier_pins.carrier_pin("a_case_study_with_no_entry") is None


def test_carrier_application_fails_closed_after_filters_and_on_missing_schema(
    pinned_case_study: str,
) -> None:
    with pytest.raises(ValueError, match="absent after candidate filters"):
        prioritize_carrier_hash(
            pl.DataFrame({"backtest_hash": ["not-the-pin"], "ic_mean": [0.1]}),
            pinned_case_study,
        )
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="backtest_hash"):
        prioritize_carrier_hash(pl.DataFrame({"ic_mean": [0.1]}), pinned_case_study)


def test_an_unpinned_case_study_passes_its_candidates_through(pinned_case_study: str) -> None:
    """The empty-mapping path, which is now every case study's.

    Without an entry there is nothing to move first and nothing to fail closed on,
    so the frame is returned as it came - including the frame that has no
    `backtest_hash` column, which is only an error where a pin has to be applied.
    """
    candidates = pl.DataFrame({"backtest_hash": ["raw_max"], "ic_mean": [0.2]})
    assert prioritize_carrier_hash(candidates, "a_case_study_with_no_entry").equals(candidates)
    bare = pl.DataFrame({"ic_mean": [0.2]})
    assert prioritize_carrier_hash(bare, "a_case_study_with_no_entry").equals(bare)


def test_carrier_row_is_prioritized_only_after_surviving_filters(pinned_case_study: str) -> None:
    candidates = pl.DataFrame(
        {
            "backtest_hash": ["raw_max", f"{FIXTURE_PIN}_suffix"],
            "ic_mean": [0.2, 0.1],
        }
    )
    filtered = candidates.filter(pl.col("ic_mean") >= 0.1)
    result = prioritize_carrier_hash(filtered, pinned_case_study)
    assert result["backtest_hash"].to_list()[0] == f"{FIXTURE_PIN}_suffix"


def test_cohort_metrics_are_attributed_to_their_leader() -> None:
    attribution = cohort_metric_attribution({"leader_hash": "lasso_hash"}, "ridge_hash")
    assert attribution == {
        "leader_hash": "lasso_hash",
        "carrier_hash": "ridge_hash",
        "applies_to_carrier": False,
        "subject": "family cohort leader lasso_hash",
    }


def test_pbo_with_two_combinations_is_not_reportable() -> None:
    assert reportable_pbo(0.5, 2) == {
        "value": None,
        "status": "insufficient combinations (2 < 10)",
        "n_combinations": 2,
    }
