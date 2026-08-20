"""A metric table's shape must not depend on what has been written into it.

``_upsert_wide_metrics`` adds a column the first time it sees a metric name it does not know.
That is what lets a case study register its own aux metrics, and it is also how the schema came
to differ between registries: ``backtest_metrics`` had 22 columns where no backtest had ever been
registered and 37 where one had. Every notebook that reads a confidence band raised

    OperationalError: no such column: m.sharpe_ci95_lo

against exactly the registries a reset had just created - which is where a rebuild starts. The
column set the producers emit is now declared when the registry is opened, and these tests fail
if a producer grows a key the declaration does not cover.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from case_studies.utils.registry.store import _open_registry

METRIC_TABLES = ("backtest_metrics", "backtest_fold_metrics", "prediction_metrics")


@pytest.fixture
def registry(tmp_path):
    db = _open_registry(tmp_path)
    yield db
    db.close()


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}


class TestAnEmptyRegistryHasItsFullShape:
    @pytest.mark.parametrize(
        ("table", "band"),
        [
            ("backtest_metrics", "sharpe_ci95_lo"),
            ("backtest_fold_metrics", "sharpe_ci95_lo"),
            ("prediction_metrics", "ic_ci_lo"),
        ],
    )
    def test_a_band_column_exists_before_anything_is_written(self, registry, table, band):
        assert registry.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert band in _columns(registry, table)

    def test_the_query_that_failed_on_a_reset_registry_runs(self, registry):
        """The shape of the leader query in every NN_backtest notebook, against an empty registry."""
        registry.execute(
            """
            SELECT m.sharpe, m.sharpe_ci95_lo, m.sharpe_ci95_hi, m.psr_pvalue
            FROM backtest_metrics m
            JOIN backtest_runs b ON b.backtest_hash = m.backtest_hash
            WHERE b.stage = 'signal'
            """
        ).fetchall()

    def test_reopening_a_registry_declares_nothing_twice(self, tmp_path):
        first = _open_registry(tmp_path)
        shape = {table: _columns(first, table) for table in METRIC_TABLES}
        first.close()

        second = _open_registry(tmp_path)
        try:
            assert {table: _columns(second, table) for table in METRIC_TABLES} == shape
        finally:
            second.close()


class TestTheDeclarationCoversWhatTheProducersEmit:
    """The declaration is a copy of a producer's keys, so it goes stale silently without this."""

    def test_backtest_uncertainty_keys_are_all_declared(self, registry):
        from case_studies.utils.uncertainty import compute_backtest_uncertainty

        generator = np.random.default_rng(0)
        produced = compute_backtest_uncertainty(
            generator.normal(scale=0.01, size=500), periods_per_year=252, n_boot=50
        )

        assert produced, "the producer returned nothing, so this test would pass vacuously"
        for table in ("backtest_metrics", "backtest_fold_metrics"):
            assert set(produced) <= _columns(registry, table), (
                f"{table} cannot hold {sorted(set(produced) - _columns(registry, table))}; add "
                "them to _DECLARED_METRIC_COLUMNS in case_studies/utils/registry/store.py"
            )

    def test_the_ic_and_auc_blocks_are_all_declared(self, registry):
        """Read the keys out of the source, because producing them needs a fitted prediction set."""
        import ast
        import inspect

        from case_studies.utils.registry import metrics

        emitted = {
            key.value
            for node in ast.walk(ast.parse(inspect.getsource(metrics)))
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value.startswith(("ic_", "auc_"))
        }

        assert len(emitted) > 20, f"only found {len(emitted)} keys; the scan stopped working"
        assert emitted <= _columns(registry, "prediction_metrics"), (
            "prediction_metrics cannot hold "
            f"{sorted(emitted - _columns(registry, 'prediction_metrics'))}"
        )


def test_an_unknown_metric_is_still_added_on_write(registry):
    """The auto-add stays: a case study's own aux metric is not part of any declaration."""
    from case_studies.utils.registry.store import _upsert_wide_metrics

    # What is under test is the ALTER, not referential integrity; building the run -> prediction
    # -> training chain a real insert needs would test the fixtures instead.
    registry.execute("PRAGMA foreign_keys=OFF")
    _upsert_wide_metrics(
        registry, "backtest_metrics", {"backtest_hash": "bt1"}, {"cumulative_hedge_cost": 12.5}
    )

    assert registry.execute(
        "SELECT cumulative_hedge_cost FROM backtest_metrics WHERE backtest_hash = 'bt1'"
    ).fetchone() == (12.5,)
