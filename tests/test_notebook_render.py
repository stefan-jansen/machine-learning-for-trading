"""Regression tests for notebook rendering diagnostics."""

import sqlite3

import numpy as np
import polars as pl
import pytest

from case_studies.utils import notebook_render
from case_studies.utils.conformal import (
    holdout_conformal_embargo_steps,
    walk_forward_conformal_coverage,
)
from case_studies.utils.registry.store import _open_registry


def test_conformal_diagnostic_excludes_partial_and_orders_folds_chronologically(
    tmp_path, monkeypatch
):
    """The row describes the family's full-history IC leader, not a partial-history config."""
    run_log = tmp_path / "run_log"
    pred_dir = run_log / "predictions"
    pred_dir.mkdir(parents=True)
    db_path = run_log / "registry.db"

    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT,
                config_name TEXT,
                label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY,
                training_hash TEXT,
                split TEXT
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY,
                ic_mean_daily REAL,
                ic_n_days REAL
            );
            """
        )
        db.executemany(
            "INSERT INTO training_runs VALUES (?, 'gbm', ?, 'fwd_ret_5d')",
            [("train_partial", "partial"), ("train_full", "full")],
        )
        db.executemany(
            "INSERT INTO prediction_sets VALUES (?, ?, 'validation')",
            [("pred_partial", "train_partial"), ("pred_full", "train_full")],
        )
        db.executemany(
            "INSERT INTO prediction_metrics VALUES (?, ?, ?)",
            [("pred_partial", 0.20, 1.0), ("pred_full", 0.05, 2.0)],
        )

    for prediction_hash in ("pred_partial", "pred_full"):
        path = pred_dir / prediction_hash
        path.mkdir()
        _write_panel(path, {"AAA": [0.1] * 80, "BBB": [0.1] * 80})

    monkeypatch.setattr(notebook_render, "registry_path", lambda _case_study: db_path)
    result = notebook_render.conformal_coverage_diagnostic(
        "test", label="fwd_ret_5d", embargo_steps=1
    )

    assert result["config_name"].unique().to_list() == ["full"]
    assert result["nominal_level"].to_list() == [0.80, 0.90, 0.95]
    # A wider nominal level cannot cover fewer decisions on the same panel.
    coverage = result["empirical_coverage"].to_list()
    assert coverage == sorted(coverage)


def _single_config_registry(tmp_path):
    """A registry with one full-history validation config, ready for predictions."""
    run_log = tmp_path / "run_log"
    pred_dir = run_log / "predictions" / "pred_only"
    pred_dir.mkdir(parents=True)
    db_path = run_log / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY, ic_mean_daily REAL, ic_n_days REAL
            );
            """
        )
        db.execute("INSERT INTO training_runs VALUES ('t', 'gbm', 'only', 'fwd_ret_5d')")
        db.execute("INSERT INTO prediction_sets VALUES ('pred_only', 't', 'validation')")
        db.execute("INSERT INTO prediction_metrics VALUES ('pred_only', 0.1, 2.0)")
    return db_path, pred_dir


def _write_panel(pred_dir, panel: dict[str, list[float]]):
    """One row per (day, symbol), with `y_true - y_score` taken from `panel`.

    Every symbol supplies the same number of residuals, so the day grid is the row index and
    an embargo of h steps is h days back. `fold_id` labels the later half 0 and the earlier
    half 1, so a reader that treats fold ids as chronological gets the panel backwards.
    """
    lengths = {len(values) for values in panel.values()}
    assert len(lengths) == 1, "every symbol supplies the same number of steps"
    steps = lengths.pop()
    days = [f"2020-{1 + step // 28:02d}-{1 + step % 28:02d}" for step in range(steps)]
    # `y_score` ramps so the outcomes have a spread to normalize the width against; the
    # residual is still exactly what `panel` says, because `y_true` is built from it.
    scores = [float(step) / steps for step in range(steps)]
    pl.DataFrame(
        {
            "timestamp": [day for day in days for _ in panel],
            "symbol": [symbol for _ in days for symbol in panel],
            "y_true": [
                scores[step] + panel[symbol][step] for step in range(steps) for symbol in panel
            ],
            "y_score": [scores[step] for step in range(steps) for _ in panel],
            "fold_id": [1 if step < steps // 2 else 0 for step in range(steps) for _ in panel],
        }
    ).write_parquet(pred_dir / "predictions.parquet")


def test_the_diagnostic_reports_the_walk_forward_estimator(tmp_path, monkeypatch):
    """The numbers a notebook prints are the ones the sizing widths produce.

    Asserted against `walk_forward_conformal_coverage` on the same artifact rather than
    against constants, because what this pins is that the two are the same measurement: the
    diagnostic used to run a second estimator - pooled, earliest-fold, unembargoed - and print
    its coverage as the strategy's.
    """
    db_path, pred_dir = _single_config_registry(tmp_path)
    panel = {"CALM": [0.1] * 80, "WILD": [10.0] * 80}
    _write_panel(pred_dir, panel)
    monkeypatch.setattr(notebook_render, "registry_path", lambda _cs: db_path)

    result = notebook_render.conformal_coverage_diagnostic(
        "test", label="fwd_ret_5d", levels=(0.80,), embargo_steps=1
    )
    expected = walk_forward_conformal_coverage(
        pl.read_parquet(pred_dir / "predictions.parquet"), levels=(0.80,), embargo_steps=1
    )

    assert result.height == 1
    assert result.row(0, named=True) == {"family": "gbm", "config_name": "only", **expected[0]}


def test_the_embargo_defaults_to_the_reviewed_horizon_for_the_label(tmp_path, monkeypatch):
    """A caller that names no embargo gets the label's own horizon, not zero.

    The reviewed table is the one `compute_conformal_widths` reads, so a coverage figure and
    the widths it describes cannot drift apart over which horizon they embargoed.
    """
    db_path, pred_dir = _single_config_registry(tmp_path)
    _write_panel(pred_dir, {"CALM": [0.1] * 80, "WILD": [10.0] * 80})
    monkeypatch.setattr(notebook_render, "registry_path", lambda _cs: db_path)

    reviewed = holdout_conformal_embargo_steps("etfs", "fwd_ret_5d")
    assert reviewed == 5
    defaulted = notebook_render.conformal_coverage_diagnostic(
        "etfs", label="fwd_ret_5d", levels=(0.80,)
    )
    explicit = notebook_render.conformal_coverage_diagnostic(
        "etfs", label="fwd_ret_5d", levels=(0.80,), embargo_steps=reviewed
    )
    assert defaulted.equals(explicit)
    assert not defaulted.equals(
        notebook_render.conformal_coverage_diagnostic(
            "etfs", label="fwd_ret_5d", levels=(0.80,), embargo_steps=1
        )
    )


def _empty_registry(tmp_path):
    """A registry carrying the production schema and no rows in any of it.

    Built by the production opener rather than by a hand-written subset or by
    `REGISTRY_SCHEMA_SQL` alone: several columns these readers name - `ic_mean_daily` and its
    HAC interval among them - are added by `_declare_uncertainty_columns` rather than by the
    CREATE TABLE statements. A reader naming one of those against a partial fixture fails with
    OperationalError, which is a fixture defect wearing the costume of the contract under test.
    """
    _open_registry(tmp_path).close()
    return tmp_path / "run_log" / "registry.db"


# Each registry reader, the columns a notebook selects off it, and the arguments to reach it.
# The three are covered together because they fail the same way and a reader meets all three
# in one notebook, before the stage that fills any of them has run.
EMPTY_FRAME_READERS = [
    (
        "selection_adjusted_leader_table",
        {"stage": "signal"},
        ("family", "config_name", "sharpe", "dsr", "pbo", "k_variants"),
    ),
    (
        "holdout_decay_table",
        {"label": "fwd_ret_21d"},
        ("family", "config_name", "val_ic", "ho_ic", "decay_pp"),
    ),
    (
        "conformal_coverage_diagnostic",
        {"label": "fwd_ret_21d"},
        (
            "family",
            "config_name",
            "nominal_level",
            "empirical_coverage",
            "n_test",
            "n_uncalibrated",
        ),
    ),
]


@pytest.mark.parametrize("reader, kwargs, columns", EMPTY_FRAME_READERS, ids=lambda v: str(v)[:40])
def test_a_registry_reader_keeps_its_columns_when_it_has_no_rows(
    tmp_path, monkeypatch, reader, kwargs, columns
):
    """A caller selects columns off these frames, so an empty result has to carry them.

    Every case study's model-analysis notebook reads all three before its backtesting stage
    has been run, because that is the order a reader works through the notebooks in.
    Returning a bare `pl.DataFrame()` made the next line - `.select("family", ...)` - raise
    ColumnNotFoundError, which reads as a defect in the notebook rather than as a stage that
    has not run yet, and is what broke `etfs/13_model_analysis` against a fresh registry.
    """
    db_path = _empty_registry(tmp_path)
    monkeypatch.setattr(notebook_render, "registry_path", lambda _case_study: db_path)

    table = getattr(notebook_render, reader)("etfs", **kwargs)

    assert table.is_empty()
    assert table.select(*columns).is_empty()


@pytest.mark.parametrize("reader, kwargs, columns", EMPTY_FRAME_READERS, ids=lambda v: str(v)[:40])
def test_a_registry_reader_keeps_its_columns_when_there_is_no_registry(
    tmp_path, monkeypatch, reader, kwargs, columns
):
    """The same contract on the other empty path, so the two cannot disagree.

    A function returning columns when the query is empty and no columns when the file is
    absent hands its caller a frame whose shape depends on which kind of nothing it found.
    """
    monkeypatch.setattr(
        notebook_render, "registry_path", lambda _case_study: tmp_path / "absent" / "registry.db"
    )

    table = getattr(notebook_render, reader)("etfs", **kwargs)

    assert table.is_empty()
    assert table.select(*columns).is_empty()
