"""The carrier is chosen from the generation its producer still publishes.

A superseded backtest is still complete and still ``current`` under its schema version,
so it still ranks: ``identity_status`` says the registry understands the row, not that
the row is the one its case study publishes. That is recorded in the population lineage
instead, which is what ``resolve_canonical_rank1_lineage`` consults here.

Two behaviours are pinned, and the second is the one that would not announce itself.
A retired row must lose to a live row it outranks, and a field in which *every* ranked
row is retired must refuse rather than fall back - because falling back returns a
carrier the case study does not publish, and nothing downstream can tell.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import strategy_analysis

FIXTURE_CASE_STUDY = "fixture_case_study"


def _registry(path: Path, rows: list[tuple[str, str, str, float]]) -> None:
    """A registry the canonical resolver can select from.

    Each row is ``(backtest_hash, prediction_hash, stage, sharpe)``. Every prediction
    hash named gets its own training run and prediction set, so retiring one prediction
    generation does not disturb another's rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, config_name TEXT, family TEXT, label TEXT,
                spec_json TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_value TEXT, checkpoint_kind TEXT
            );
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT, spec_json TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY, sharpe REAL, max_drawdown REAL
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
            """
        )
        for prediction_hash in sorted({prediction for _, prediction, _, _ in rows}):
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, 'owner_config', 'gbm', 'fwd_ret_1m', NULL)",
                (training_hash,),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', NULL, NULL)",
                (prediction_hash, training_hash),
            )
            db.execute("INSERT INTO fold_metrics VALUES (?, 0.02)", (prediction_hash,))
        for backtest_hash, prediction_hash, stage, sharpe in rows:
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, ?, ?)",
                (backtest_hash, prediction_hash, stage, '{"strategy": {"signal": {}}}'),
            )
            db.execute("INSERT INTO backtest_metrics VALUES (?, ?, -0.20)", (backtest_hash, sharpe))


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "utils.paths.get_case_study_dir", lambda case_study, **_: tmp_path / case_study
    )
    return tmp_path / FIXTURE_CASE_STUDY


def _retire(monkeypatch: pytest.MonkeyPatch, *, backtests=(), predictions=()) -> None:
    """Record what the population lineage reports as superseded, per member kind."""

    def _superseded(_case_dir, member_kind="backtest"):
        return frozenset(backtests if member_kind == "backtest" else predictions)

    monkeypatch.setattr("case_studies.research.population.superseded_members_at", _superseded)


def test_a_retired_backtest_loses_to_the_live_row_it_outranks(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank alone would take ``retired_run``; the lineage is what demotes it."""
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("retired_run", "pred_old", "allocation", 2.40),
            ("live_run", "pred_new", "allocation", 1.10),
        ],
    )
    _retire(monkeypatch, backtests={"retired_run"})

    assert (
        strategy_analysis.resolve_canonical_rank1_lineage(FIXTURE_CASE_STUDY)["val_backtest_hash"]
        == "live_run"
    )


def test_a_retired_prediction_demotes_the_backtests_that_read_it(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The side that hides.

    A refit that changes no numbers republishes identical predictions under a new
    identity, so the old and new backtests carry the same Sharpe to the last digit and
    the ORDER BY returns whichever it likes. The tie is deliberate here: with the
    prediction side unfiltered this test picks a winner by luck rather than by lineage.
    """
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("a_retired_run", "pred_old", "allocation", 1.75),
            ("b_live_run", "pred_new", "allocation", 1.75),
        ],
    )
    _retire(monkeypatch, predictions={"pred_old"})

    resolved = strategy_analysis.resolve_canonical_rank1_lineage(FIXTURE_CASE_STUDY)
    assert resolved["val_backtest_hash"] == "b_live_run"
    assert resolved["val_prediction_hash"] == "pred_new"


def test_an_all_retired_field_refuses_rather_than_returning_a_retired_carrier(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode that would not announce itself.

    Every candidate belongs to a rebuilt generation and nothing was re-registered under
    a name still in force. There is no configuration this case study publishes, so any
    answer would be one it does not publish - and a returned hash reads downstream
    exactly like a valid selection. The refusal names how many rows were ranked.
    """
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("retired_alloc", "pred_old", "allocation", 2.40),
            ("retired_signal", "pred_old", "signal", 1.10),
        ],
    )
    _retire(monkeypatch, backtests={"retired_alloc", "retired_signal"})

    with pytest.raises(RuntimeError, match="superseded generation"):
        strategy_analysis.resolve_canonical_rank1_lineage(FIXTURE_CASE_STUDY)


def test_an_empty_lineage_retires_nothing(case_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A case study that has never superseded a population selects on rank alone."""
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("top_run", "pred_only", "allocation", 2.40),
            ("second_run", "pred_only", "signal", 1.10),
        ],
    )
    _retire(monkeypatch)

    assert (
        strategy_analysis.resolve_canonical_rank1_lineage(FIXTURE_CASE_STUDY)["val_backtest_hash"]
        == "top_run"
    )
