"""A cross-study cost chart has to say which check dropped each absent case study.

`load_carrier_cost_curves` used to answer four different questions with a bare
`continue`, so a case study missing from `20_strategy_synthesis/06_cost_survival`
left no trace of why. Three were missing at once for three different reasons and
establishing which was which took a hand-written registry query per case study.
These tests hold the loader to reporting the reason it already knows, and to
resolving every carrier through the selection rule rather than a pasted hash.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import analytics
from case_studies.utils.analytics import load_carrier_cost_curves

CARRIER_BACKTEST = "carrier_val_backtest"
CARRIER_TRAINING = "carrier_training"


def _spec(method: str, *, commission_bps: float = 0.0) -> str:
    """A canonical backtest spec naming one trading scheme at one cost level."""
    return json.dumps(
        {
            "version": 2,
            "strategy": {"signal": {"method": method, "top_k": 10}, "allocation": None},
            "backtest_config": {
                "commission": {"rate": commission_bps / 10_000.0},
                "slippage": {"rate": 0.0},
            },
        }
    )


def _registry(
    path: Path,
    *,
    carrier_method: str = "slot_persistent_signal_exit",
    sweep_method: str | None = None,
    cost_levels: tuple[float, ...] = (0.0, 5.0, 10.0),
    carrier_backtest_row: bool = True,
) -> None:
    """A registry holding one carrier backtest and an optional cost sweep beside it.

    `sweep_method` None registers no cost sweep at all; a method name registers one
    cost-sensitivity backtest per level running that scheme. Only the columns the
    loader reads are present, because a fixture that mirrors the production schema
    is a second implementation of it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT);
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
            );
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT,
                spec_json TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY, sharpe REAL, total_return REAL,
                max_drawdown REAL
            );
            """
        )
        db.execute("INSERT INTO training_runs VALUES (?, 'fwd_ret_1d')", (CARRIER_TRAINING,))
        db.execute(
            "INSERT INTO prediction_sets VALUES ('carrier_pred', ?, 'validation')",
            (CARRIER_TRAINING,),
        )
        if carrier_backtest_row:
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, 'carrier_pred', 'signal', ?)",
                (CARRIER_BACKTEST, _spec(carrier_method)),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, 1.5, 0.2, -0.1)", (CARRIER_BACKTEST,)
            )
        if sweep_method is not None:
            for level in cost_levels:
                db.execute(
                    "INSERT INTO backtest_runs VALUES (?, 'carrier_pred', 'cost_sensitivity', ?)",
                    (f"cost_{level}", _spec(sweep_method, commission_bps=level)),
                )
                db.execute(
                    "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.2)",
                    (f"cost_{level}", 1.5 - level / 20.0),
                )


@pytest.fixture
def case_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """One fixture case study with its own registry, resolving to a known carrier.

    The resolver is replaced rather than fed, because what is under test is what the
    loader does with the carrier it is handed - not the ranking that picks one.
    """
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    from case_studies.utils import strategy_analysis

    monkeypatch.setattr(
        strategy_analysis,
        "resolve_canonical_rank1_lineage",
        lambda cs, **kwargs: {
            "training_hash": CARRIER_TRAINING,
            "val_backtest_hash": CARRIER_BACKTEST,
            "family": "gbm",
            "config_name": "default",
            "label": "fwd_ret_1d",
        },
    )
    return "fixture_case_study"


def _only_exclusion(case_study: str) -> dict[str, str]:
    result = load_carrier_cost_curves([case_study])
    assert result.curves.is_empty()
    assert result.exclusions.height == 1
    return result.exclusions.row(0, named=True)


def test_an_absent_registry_is_reported_rather_than_skipped(case_study: str) -> None:
    row = _only_exclusion(case_study)
    assert row["reason"] == "no registry"
    # The path it looked at, so a reader can tell a missing registry from one the
    # process resolved somewhere else. `_cs_dir` falls back to the checkout when
    # `ML4T_OUTPUT_DIR` holds no registry for this case study, and that fallback is
    # exactly what a reader needs to see named.
    assert row["detail"].endswith(f"{case_study}/run_log/registry.db does not exist")


def test_a_carrier_that_does_not_resolve_carries_the_failure_into_the_reason(
    case_study: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry(tmp_path / case_study / "run_log" / "registry.db", sweep_method=None)
    from case_studies.utils import strategy_analysis

    def _raise(cs: str, **kwargs: object) -> dict[str, str]:
        raise RuntimeError("the field was ruined")

    monkeypatch.setattr(strategy_analysis, "resolve_canonical_rank1_lineage", _raise)

    row = _only_exclusion(case_study)
    assert row["reason"] == "carrier does not resolve"
    assert "RuntimeError" in row["detail"]
    assert "the field was ruined" in row["detail"]


def test_a_carrier_whose_validation_backtest_is_gone_says_which_hash_was_looked_up(
    case_study: str, tmp_path: Path
) -> None:
    _registry(
        tmp_path / case_study / "run_log" / "registry.db",
        sweep_method=None,
        carrier_backtest_row=False,
    )
    row = _only_exclusion(case_study)
    assert row["reason"] == "carrier has no validation backtest row"
    assert CARRIER_BACKTEST in row["detail"]


def test_a_lineage_with_no_cost_sweep_names_the_carrier_it_looked_for(
    case_study: str, tmp_path: Path
) -> None:
    _registry(tmp_path / case_study / "run_log" / "registry.db", sweep_method=None)
    row = _only_exclusion(case_study)
    assert row["reason"] == "no cost sweep on the carrier's lineage"
    assert CARRIER_TRAINING in row["detail"]
    assert "gbm/default on fwd_ret_1d" in row["detail"]


def test_a_sweep_on_another_strategy_is_distinguished_from_no_sweep_at_all(
    case_study: str, tmp_path: Path
) -> None:
    """The distinction this whole file exists for.

    A carrier whose lineage was swept under a different trading scheme and one whose
    lineage was never swept are both absent from the chart and are not the same
    finding: the first is a sweep pointed at the wrong strategy, the second is a
    sweep that never ran. NASDAQ-100 and ETFs were each one of these and the chapter
    could not tell them apart.
    """
    _registry(
        tmp_path / case_study / "run_log" / "registry.db",
        carrier_method="slot_persistent_signal_exit",
        sweep_method="equal_weight_top_k",
    )
    row = _only_exclusion(case_study)
    assert row["reason"] == "cost sweep is on a different strategy"
    assert "3 cost_sensitivity backtests" in row["detail"]
    assert "slot_persistent_signal_exit" in row["detail"]


def test_a_carrier_with_its_own_sweep_draws_a_curve_and_is_not_excluded(
    case_study: str, tmp_path: Path
) -> None:
    _registry(
        tmp_path / case_study / "run_log" / "registry.db",
        carrier_method="slot_persistent_signal_exit",
        sweep_method="slot_persistent_signal_exit",
    )
    result = load_carrier_cost_curves([case_study])
    assert result.exclusions.is_empty()
    assert result.curves.height == 3
    assert sorted(result.curves["cost_bps"].to_list()) == [0.0, 5.0, 10.0]


def test_every_case_study_asked_for_is_either_drawn_or_explained(
    case_study: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant that makes an absence readable rather than a guess.

    A case study that is in neither frame has been dropped by a check that reports
    nothing, which is the state this loader was in.
    """
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    drawn = "drawn_case_study"
    _registry(
        tmp_path / drawn / "run_log" / "registry.db",
        carrier_method="slot_persistent_signal_exit",
        sweep_method="slot_persistent_signal_exit",
    )
    _registry(
        tmp_path / case_study / "run_log" / "registry.db",
        carrier_method="slot_persistent_signal_exit",
        sweep_method="equal_weight_top_k",
    )
    asked = [case_study, drawn, "case_study_with_no_registry"]
    result = load_carrier_cost_curves(asked)

    accounted = set(result.curves["case_study"].to_list()) | set(
        result.exclusions["case_study"].to_list()
    )
    assert accounted == set(asked)
    assert not (
        set(result.curves["case_study"].to_list()) & set(result.exclusions["case_study"].to_list())
    )


def test_the_loader_resolves_every_carrier_and_names_none_of_them() -> None:
    """No case-study special case and no pasted lineage inside the loader.

    `_NASDAQ_ENSEMBLE_TRAINING_HASH = "a9f04b886b9a"` sat beside this function and
    routed one case study around the resolver. The 2026-09-05 registry reset replaced
    the lineage it named, the literal then matched zero rows, and NASDAQ-100 silently
    left the chart. A hash names a lineage rather than a rule, so the next rebuild
    kills it again; the guard is against the shape, not against that one value.
    """
    source = Path(analytics.__file__).read_text()
    tree = ast.parse(source)
    loader = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_carrier_cost_curves"
    )

    # Module-wide, because the two dead literals were module-level constants the
    # loader referenced by name. Scanning only the function body reads as green while
    # the pin sits ten lines above it.
    hex_literals = sorted(
        {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"[0-9a-f]{8,}", node.value)
        }
    )
    assert hex_literals == [], (
        f"case_studies/utils/analytics.py carries the literals {hex_literals}; a "
        "registry identity belongs to the resolver, which is re-read every run"
    )

    known_case_studies = set(analytics.CASE_STUDY_IDS)
    named = [
        node.value
        for node in ast.walk(loader)
        if isinstance(node, ast.Constant) and node.value in known_case_studies
    ]
    assert named == [], (
        f"load_carrier_cost_curves special-cases {named}; every case study resolves "
        "its carrier through the same rule"
    )


def test_a_load_that_draws_nothing_still_carries_the_reasons(case_study: str) -> None:
    """The all-excluded case, which is what a clean clone with no registries reaches.

    `06_cost_survival` refuses when no curve loads, and the refusal used to carry only that
    nothing was found - the reasons were printed on the line after it and never ran. They are
    the diagnosis, and the loader has already established each one, so they belong in the
    message. `exclusion_lines` is what the notebook raises with.
    """
    loaded = load_carrier_cost_curves([case_study])
    assert loaded.curves.is_empty()
    lines = loaded.exclusion_lines()
    assert len(lines) == 1
    assert lines[0].startswith("  no curve for ")
    assert "no registry" in lines[0]

    # The shape the notebook builds. A refusal that named the failure and dropped the reasons
    # is the defect; this asserts they survive into it.
    refusal = "\n".join(
        ["No Ch18 cost-sensitivity backtests found for any deployed carrier", *lines]
    )
    assert "no registry" in refusal
    assert refusal.count("\n") == 1


def test_exclusion_lines_is_empty_when_every_case_study_drew_a_curve(
    case_study: str, tmp_path: Path
) -> None:
    """Negative control: nothing excluded, nothing to say, and no line invented for it."""
    _registry(
        tmp_path / case_study / "run_log" / "registry.db",
        carrier_method="slot_persistent_signal_exit",
        sweep_method="slot_persistent_signal_exit",
    )
    loaded = load_carrier_cost_curves([case_study])
    assert loaded.curves.height == 3
    assert loaded.exclusion_lines() == []
