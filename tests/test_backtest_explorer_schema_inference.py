"""A metric added after a registry was written must not break every query that selects it.

`BacktestExplorer._query` builds a frame from `sqlite3.Row` mappings. Polars infers a schema
from the first `infer_schema_length` rows, 100 by default, so a column that is NULL on every
early row and carries a float later is typed `Null` and then refuses the float. That is the
shape a new metric always has: `ruin` arrived with #834 and is NULL on every backtest recorded
before it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.backtest_explorer import BacktestExplorer


def _registry_with_a_late_metric(path: Path, n_null_rows: int, n_valued_rows: int) -> None:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE backtest_metrics (backtest_hash TEXT, sharpe REAL, ruin REAL)")
    db.executemany(
        "INSERT INTO backtest_metrics VALUES (?, ?, NULL)",
        [(f"h{i:04d}", 0.5) for i in range(n_null_rows)],
    )
    db.executemany(
        "INSERT INTO backtest_metrics VALUES (?, ?, ?)",
        [(f"h{n_null_rows + i:04d}", 0.5, 0.0) for i in range(n_valued_rows)],
    )
    db.commit()
    db.close()


@pytest.fixture
def explorer(tmp_path: Path) -> BacktestExplorer:
    case_dir = tmp_path / "run_log"
    case_dir.mkdir(parents=True)
    _registry_with_a_late_metric(case_dir / "registry.db", n_null_rows=180, n_valued_rows=18)
    instance = BacktestExplorer.__new__(BacktestExplorer)
    instance._db_path = case_dir / "registry.db"
    return instance


def test_a_metric_null_for_more_than_a_schema_window_still_reads(
    explorer: BacktestExplorer,
) -> None:
    """180 NULL `ruin` rows ahead of the first float is what broke `etfs` in production."""
    frame = explorer._query("SELECT * FROM backtest_metrics")

    assert frame.height == 198
    assert frame.schema["ruin"] == pl.Float64
    assert frame.get_column("ruin").null_count() == 180
    assert frame.get_column("ruin").drop_nulls().to_list() == [0.0] * 18


def test_a_column_null_in_every_row_stays_readable(explorer: BacktestExplorer) -> None:
    """The all-NULL case must keep working: a metric no row has yet is not an error."""
    frame = explorer._query("SELECT backtest_hash, ruin FROM backtest_metrics LIMIT 180")

    assert frame.height == 180
    assert frame.get_column("ruin").null_count() == 180


def test_an_empty_result_is_an_empty_frame(explorer: BacktestExplorer) -> None:
    frame = explorer._query("SELECT * FROM backtest_metrics WHERE backtest_hash = 'absent'")

    assert frame.is_empty()
