"""The carrier every downstream notebook runs, and the solvency check on it.

`resolve_solvent_carrier` exists so cost sensitivity, holdout prediction and holdout
backtest do not each rank the registry for themselves. The tests below fix the two
things that makes it worth having: it returns what the canonical resolver returned,
and it refuses a carrier whose equity reached zero rather than passing it on.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import strategy_analysis

FIXTURE_CASE_STUDY = "fixture_case_study"


def _registry(
    path: Path,
    rows: list[tuple[str, str, float, float | None]],
) -> None:
    """A registry the canonical resolver can select from.

    Each row is ``(backtest_hash, stage, sharpe, max_drawdown)``; a ``None`` drawdown
    is a run whose metrics row exists but records no equity path. Every table the
    resolver joins is present and populated, so the only thing distinguishing the
    rows is what the test sets.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, config_name TEXT, family TEXT, label TEXT
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
            INSERT INTO training_runs VALUES ('train_us', 'owner_config', 'gbm', 'fwd_ret_1m');
            INSERT INTO prediction_sets VALUES ('pred_us', 'train_us', 'validation', NULL, NULL);
            INSERT INTO fold_metrics VALUES ('pred_us', 0.02);
            """
        )
        for backtest_hash, stage, sharpe, max_drawdown in rows:
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, 'pred_us', ?, ?)",
                (backtest_hash, stage, '{"strategy": {"signal": {}}}'),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, ?)",
                (backtest_hash, sharpe, max_drawdown),
            )


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "utils.paths.get_case_study_dir", lambda case_study, **_: tmp_path / case_study
    )
    return tmp_path / FIXTURE_CASE_STUDY


def test_carrier_is_the_canonical_rank1_with_its_spec_and_drawdown(case_dir: Path) -> None:
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("signal_run", "signal", 1.10, -0.31),
            ("alloc_run", "allocation", 2.40, -0.22),
            ("overlay_run", "risk_overlay", 1.75, -0.18),
        ],
    )

    carrier = strategy_analysis.resolve_solvent_carrier(FIXTURE_CASE_STUDY)

    lineage = strategy_analysis.resolve_canonical_rank1_lineage(FIXTURE_CASE_STUDY)
    assert carrier["val_backtest_hash"] == lineage["val_backtest_hash"] == "alloc_run"
    assert carrier["val_stage"] == "allocation"
    assert carrier["max_drawdown"] == pytest.approx(-0.22)
    assert carrier["spec_json"] == '{"strategy": {"signal": {}}}'


def test_a_bankrupt_rank1_raises_rather_than_being_swept(case_dir: Path) -> None:
    """The highest Sharpe here belongs to a run whose equity reached zero.

    Nothing downstream of it means anything, and quietly selecting the solvent run
    behind it would hand the cost sweep a configuration the chapter does not report -
    the divergence this resolver exists to close. So it refuses, and names the row.
    """
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("ruined_run", "allocation", 9.90, -1.42),
            ("solvent_run", "signal", 1.10, -0.31),
        ],
    )

    with pytest.raises(RuntimeError, match="ruined_run.*reached zero equity"):
        strategy_analysis.resolve_solvent_carrier(FIXTURE_CASE_STUDY)

    # The check is what refuses, not the resolution: the same rank-1 comes back when
    # solvency is not asserted, so a caller that opts out gets no silent substitution.
    assert (
        strategy_analysis.resolve_solvent_carrier(FIXTURE_CASE_STUDY, require_solvent=False)[
            "val_backtest_hash"
        ]
        == "ruined_run"
    )


def test_exactly_minus_one_is_ruin_and_just_above_it_is_not(case_dir: Path) -> None:
    """-1.0 is zero equity, not the last solvent point. The boundary is `<=`."""
    _registry(case_dir / "run_log" / "registry.db", [("edge_run", "allocation", 2.0, -1.0)])
    with pytest.raises(RuntimeError, match="reached zero equity"):
        strategy_analysis.resolve_solvent_carrier(FIXTURE_CASE_STUDY)

    other = case_dir.parent / "second_fixture"
    _registry(other / "run_log" / "registry.db", [("edge_run", "allocation", 2.0, -0.9999)])
    assert (
        strategy_analysis.resolve_solvent_carrier("second_fixture")["val_backtest_hash"]
        == "edge_run"
    )


def test_an_unmeasured_drawdown_is_refused_rather_than_read_as_solvent(case_dir: Path) -> None:
    """A NULL drawdown is a run that cannot be shown to have survived.

    `max_drawdown IS NULL` compares False against any threshold, so a solvency test
    written as an inequality alone lets it through as if it had been measured.
    """
    _registry(case_dir / "run_log" / "registry.db", [("unmeasured_run", "allocation", 2.0, None)])
    with pytest.raises(RuntimeError, match="no recorded max_drawdown"):
        strategy_analysis.resolve_solvent_carrier(FIXTURE_CASE_STUDY)
