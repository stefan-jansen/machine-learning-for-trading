"""A rebuild must be able to remove what it no longer produces.

`register_paired_metrics` is an UPSERT keyed on `(challenger_hash, benchmark_hash)`. That is
right for the additive callers, but it means a rebuild under a different selection writes its
new pairs *beside* the previous selection's rows rather than replacing them. The rebuild that
`18_strategy_analysis` triggers when `derived_tables_off_canonical_universe` reports the table
stale would then leave it stale, so the staleness would be reported again on the next run, and
the table would carry two selections' comparisons at once.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils import paired_metrics as pm
from case_studies.utils.paired_metrics import _prune_paired_metrics
from case_studies.utils.registry.store import _open_registry


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    case_dir = tmp_path / "nasdaq100_microstructure"
    (case_dir / "run_log").mkdir(parents=True)
    # The registry's own schema, not a hand-rolled stand-in: `register_paired_metrics`
    # writes every column, so a two-column table would pass this test and fail in use.
    _open_registry(case_dir).close()
    return case_dir


def _pairs(case_dir: Path) -> set[tuple[str, str]]:
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        return set(
            db.execute("SELECT challenger_hash, benchmark_hash FROM backtest_paired_metrics")
        )


def _seed(case_dir: Path, *pairs: tuple[str, str]) -> None:
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.executemany(
            "INSERT INTO backtest_paired_metrics "
            "(challenger_hash, benchmark_hash, computed_at) VALUES (?, ?, '2026-01-01T00:00:00Z')",
            pairs,
        )


def _seed_run(case_dir: Path, backtest_hash: str) -> None:
    """`backtest_paired_metrics.challenger_hash` has a FK; `benchmark_hash` deliberately has none."""
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT OR IGNORE INTO backtest_runs "
            "(backtest_hash, prediction_hash, created_at) VALUES (?, 'pred', ?)",
            (backtest_hash, "2026-01-01T00:00:00Z"),
        )


def test_a_pair_from_an_earlier_selection_is_removed(case_dir: Path) -> None:
    _seed(case_dir, ("full_leader", "side_ew:x"), ("feasible_leader", "side_ew:x"))

    n = _prune_paired_metrics(
        "nasdaq100_microstructure", {("feasible_leader", "side_ew:x")}, case_dir, False
    )

    assert n == 1
    assert _pairs(case_dir) == {("feasible_leader", "side_ew:x")}


def test_the_pairs_this_run_wrote_survive(case_dir: Path) -> None:
    written = {("a", "b"), ("c", "d")}
    _seed(case_dir, *written)

    assert _prune_paired_metrics("nasdaq100_microstructure", written, case_dir, False) == 0
    assert _pairs(case_dir) == written


def test_a_run_that_wrote_nothing_destroys_nothing(case_dir: Path) -> None:
    """Every pair skipping is a failed rebuild, not an empty canonical snapshot.

    A missing benchmark artifact skips every pair. Treating that as "the snapshot is empty"
    would delete the last good rows and leave nothing to fall back on.
    """
    _seed(case_dir, ("a", "b"))

    assert _prune_paired_metrics("nasdaq100_microstructure", set(), case_dir, False) == 0
    assert _pairs(case_dir) == {("a", "b")}


def test_a_case_dir_with_no_registry_is_not_an_error(tmp_path: Path) -> None:
    assert _prune_paired_metrics("nasdaq100_microstructure", {("a", "b")}, tmp_path, False) == 0


class TestTheRebuildEntryPoint:
    """`populate_paired_metrics(replace_all=True)` as `18_strategy_analysis` calls it."""

    @staticmethod
    def _explorer(rows: pl.DataFrame):
        class _Explorer:
            def best(
                self,
                *,
                stage: str,
                top_n: int,
                prediction_hashes: list[str] | None = None,
            ) -> pl.DataFrame:
                if prediction_hashes is None:
                    return rows
                return rows.filter(pl.col("prediction_hash").is_in(prediction_hashes))

            def champion_lineage(self, _prediction_hash: str) -> dict:
                return {}

        return _Explorer()

    @staticmethod
    def _leader_row() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "backtest_hash": ["feasible_leader"],
                "prediction_hash": ["pred"],
                "label": ["fwd_ret_5m"],
                "family": ["gbm"],
                "sharpe": [1.5],
            }
        )

    def _stub_pair_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Let pair #1 succeed and every later pair drop out."""
        returns = pl.DataFrame(
            {
                "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 12, 31), "1d", eager=True),
                "ret": [0.001] * 366,
            }
        )
        monkeypatch.setattr(pm, "_aligned_returns", lambda cs, h: returns)
        monkeypatch.setattr(
            pm,
            "_benchmark_returns_from_artifact",
            lambda cs, label, period="overall": (
                ("side_ew:nasdaq100_microstructure:fwd_ret_5m", returns, label)
                if period == "overall"
                else None
            ),
        )
        monkeypatch.setattr(pm, "compute_paired_uncertainty", lambda *a, **k: {"sharpe_diff": 0.1})
        monkeypatch.setattr(
            pm, "_val_rank1_carrier", lambda *a, **k: {"spec": {}, "prediction_hash": None}
        )
        monkeypatch.setattr(pm, "_holdout_lineage_for", lambda *a, **k: None)

    def test_the_previous_selections_rows_are_gone_after_a_rebuild(
        self, case_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_run(case_dir, "full_leader")
        _seed_run(case_dir, "feasible_leader")
        _seed(case_dir, ("full_leader", "side_ew:nasdaq100_microstructure:fwd_ret_5m"))
        self._stub_pair_one(monkeypatch)

        pm.populate_paired_metrics(
            "nasdaq100_microstructure",
            self._explorer(self._leader_row()),
            replace_all=True,
            verbose=False,
            write_case_dir=case_dir,
        )

        assert _pairs(case_dir) == {
            ("feasible_leader", "side_ew:nasdaq100_microstructure:fwd_ret_5m")
        }

    def test_without_replace_all_the_earlier_rows_stay(
        self, case_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default is additive, and the callers that rely on it must keep it."""
        _seed_run(case_dir, "full_leader")
        _seed_run(case_dir, "feasible_leader")
        _seed(case_dir, ("full_leader", "side_ew:nasdaq100_microstructure:fwd_ret_5m"))
        self._stub_pair_one(monkeypatch)

        pm.populate_paired_metrics(
            "nasdaq100_microstructure",
            self._explorer(self._leader_row()),
            verbose=False,
            write_case_dir=case_dir,
        )

        assert ("full_leader", "side_ew:nasdaq100_microstructure:fwd_ret_5m") in _pairs(case_dir)

    def test_a_run_that_produced_no_candidate_leaves_the_table_alone(self, case_dir: Path) -> None:
        """No candidates is a registry that cannot be rebuilt from, not an empty snapshot."""
        _seed(case_dir, ("full_leader", "side_ew:nasdaq100_microstructure:fwd_ret_5m"))

        pm.populate_paired_metrics(
            "nasdaq100_microstructure",
            self._explorer(pl.DataFrame()),
            replace_all=True,
            verbose=False,
            write_case_dir=case_dir,
        )

        assert _pairs(case_dir) == {("full_leader", "side_ew:nasdaq100_microstructure:fwd_ret_5m")}
