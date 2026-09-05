"""Regression tests for validation-only temporal alignment in options evaluation."""

from __future__ import annotations

import ast
import os
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from utils.cv_splits import generate_cv_splits

NOTEBOOK = Path("case_studies/sp500_options/05_evaluation.py")
JOIN_COLS = ["timestamp", "symbol"]


def _load_alignment_function():
    """Load the pure alignment functions without executing the notebook."""
    tree = ast.parse(NOTEBOOK.read_text())
    wanted = {
        "_as_date",
        "_validate_temporal_keys",
        "build_validation_temporal_panel",
        "keep_outcomes_resolved_before_holdout",
    }
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace = {
        "DATE_COL": "timestamp",
        "date": date,
        "datetime": datetime,
        "JOIN_COLS": JOIN_COLS,
        "pl": pl,
    }
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, NOTEBOOK, "exec"), namespace)
    return namespace


def _synthetic_splits() -> list[dict]:
    return [
        {
            "fold": 0,
            "train_start": date(2018, 1, 5),
            "train_end": date(2019, 11, 12),
            "val_start": date(2020, 1, 6),
            "val_end": date(2020, 12, 31),
        },
        {
            "fold": 1,
            "train_start": date(2017, 2, 2),
            "train_end": date(2018, 11, 12),
            "val_start": date(2019, 1, 7),
            "val_end": date(2020, 1, 3),
        },
    ]


def test_validation_alignment_keeps_only_dates_inside_a_validation_window() -> None:
    """One value per date and symbol, and only the dates some fold scores.

    `04_model_based_features` fits on a refit schedule, so there is no per-fold estimate to
    pick between and no holdout pass to exclude - what is left to get wrong is letting a
    training date or a holdout date into the panel.
    """
    temporal = pl.DataFrame(
        {
            "timestamp": [
                date(2018, 6, 4),  # inside fold 1's training window, scored by nobody
                date(2019, 6, 3),  # fold 1's validation window
                date(2020, 6, 3),  # fold 0's validation window
                date(2021, 6, 3),  # the holdout
            ],
            "symbol": ["AAA"] * 4,
            "temporal_value": [1.0, 11.0, 20.0, 99.0],
        }
    )

    aligned = _load_alignment_function()["build_validation_temporal_panel"](
        temporal, _synthetic_splits()
    ).sort("timestamp")

    assert aligned["timestamp"].to_list() == [date(2019, 6, 3), date(2020, 6, 3)]
    assert aligned["temporal_value"].to_list() == [11.0, 20.0]
    assert aligned.group_by(JOIN_COLS).len().filter(pl.col("len") > 1).is_empty()


def test_a_date_two_folds_both_score_is_kept_once() -> None:
    """Overlapping validation windows contribute one row, not one row per fold.

    The case studies' own folds do not currently overlap, so the overlap is constructed:
    under the per-fold artifact it produced two rows for one date and symbol, which is why
    this function used to check for multiply assigned keys. Selecting by window makes it a
    union, and the check is no longer needed - this pins that it is genuinely not needed
    rather than merely absent.
    """
    overlapping = [
        {
            "fold": 0,
            "train_start": date(2018, 1, 5),
            "train_end": date(2019, 11, 12),
            "val_start": date(2020, 1, 6),
            "val_end": date(2020, 12, 31),
        },
        {
            "fold": 1,
            "train_start": date(2017, 2, 2),
            "train_end": date(2018, 11, 12),
            "val_start": date(2019, 1, 7),
            "val_end": date(2020, 6, 30),
        },
    ]
    temporal = pl.DataFrame(
        {
            "timestamp": [date(2019, 6, 3), date(2020, 3, 16)],
            "symbol": ["AAA", "AAA"],
            "temporal_value": [7.0, 8.0],
        }
    )

    aligned = _load_alignment_function()["build_validation_temporal_panel"](temporal, overlapping)

    # 2020-03-16 falls inside both windows and appears once.
    assert aligned.height == 2
    assert aligned.group_by(JOIN_COLS).len().filter(pl.col("len") > 1).is_empty()


def test_validation_alignment_fails_closed_on_a_fold_column() -> None:
    """A fold column means the artifact came from the design this change replaced."""
    temporal = pl.DataFrame(
        {
            "timestamp": [date(2019, 6, 3), date(2019, 6, 3), date(2020, 6, 3)],
            "symbol": ["AAA", "AAA", "AAA"],
            "fold": [0, 1, 1],
            "temporal_value": [10.0, 10.5, 21.0],
        }
    )

    with pytest.raises(ValueError, match="fold"):
        _load_alignment_function()["build_validation_temporal_panel"](temporal, _synthetic_splits())


def test_validation_alignment_fails_closed_on_duplicate_date_symbol_keys() -> None:
    temporal = pl.DataFrame(
        {
            "timestamp": [date(2019, 6, 3), date(2019, 6, 3), date(2020, 6, 3)],
            "symbol": ["AAA", "AAA", "AAA"],
            "temporal_value": [10.0, 10.5, 21.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate date-symbol keys"):
        _load_alignment_function()["build_validation_temporal_panel"](temporal, _synthetic_splits())


def test_primary_label_purge_uses_each_rows_actual_expiry() -> None:
    labels = pl.DataFrame(
        {
            "timestamp": [date(2020, 11, 20), date(2020, 12, 10)],
            "symbol": ["AAA", "AAA"],
            "dte_calendar": [35, 35],
            "ret_to_expiry": [0.1, 0.2],
        }
    )

    purge = _load_alignment_function()["keep_outcomes_resolved_before_holdout"]
    selected = purge(labels, date(2021, 1, 1))
    assert selected["timestamp"].to_list() == [date(2020, 11, 20)]
    assert selected["_label_end"].to_list() == [date(2020, 12, 25)]


def test_real_artifact_alignment_is_safe_after_regeneration() -> None:
    """Exercise the alignment contract against the available production artifact."""
    # Features come from the artifact store and the config from the repo, because they
    # stopped living under one root on 2026-08-21 when the store moved out of ~/ml4t/code
    # so that repo could be archived. This test kept the old single root and every path
    # under it has been absent since, so it has skipped on the workstation too - the only
    # place it can run, since the artifact store is in no CI job.
    store = Path(
        os.environ.get("ML4T_ARTIFACT_ROOT", Path.home() / "ml4t" / "artifacts" / "case_studies")
    )
    artifact_root = Path(
        os.environ.get("ML4T_SP500_OPTIONS_ARTIFACT_ROOT", store / "sp500_options")
    )
    financial_path = artifact_root / "features/financial.parquet"
    temporal_path = artifact_root / "features/model_based.parquet"
    # An explicitly-pointed root may carry its own config; otherwise the repo's is the
    # only copy there is.
    setup_path = artifact_root / "config/setup.yaml"
    if not setup_path.exists():
        setup_path = Path("case_studies/sp500_options/config/setup.yaml")
    if not all(path.exists() for path in (financial_path, temporal_path, setup_path)):
        pytest.skip("Full sp500_options artifacts are not available")

    financial = pl.read_parquet(financial_path)
    temporal = pl.read_parquet(temporal_path)
    setup = yaml.safe_load(setup_path.read_text())
    # setup_path rather than case_study_id: the buffer above is read from this file, and
    # case_study_id would take the evaluation section from somewhere else - the repo, or
    # ML4T_OUTPUT_DIR's seeded copy when one is set. The folds and the buffer that shifts
    # them have to come from the same config or the artifact is checked against neither.
    folds = generate_cv_splits(
        financial.select("timestamp"),
        setup_path=setup_path,
        label_buffer=str(setup["labels"]["buffer"]),
        # Read from the same config as the buffer, for the same reason. The splitter defaults
        # this to `sessions`, so leaving it out derives 35 sessions from a 35-calendar-day
        # declaration and checks the artifact against windows nothing produced.
        buffer_unit=str(setup["labels"].get("buffer_unit", "sessions")),
    )

    align = _load_alignment_function()["build_validation_temporal_panel"]

    if "fold" in temporal.columns:
        # The artifact on disk still comes from the per-fold design; `04` has been converted
        # to a refit schedule but not yet re-executed against the production store. Rejecting
        # it is the contract, so that is what is asserted - not skipped, because a silent skip
        # here is how a stale artifact reaches a model.
        with pytest.raises(ValueError, match="fold"):
            align(temporal, folds)
        return

    aligned = align(temporal, folds)

    assert aligned.group_by(JOIN_COLS).len().filter(pl.col("len") > 1).is_empty()
    earliest = min(split["val_start"].date() for split in folds)
    latest = max(split["val_end"].date() for split in folds)
    assert aligned["timestamp"].min() >= earliest
    assert aligned["timestamp"].max() <= latest
    for split in folds:
        window = aligned.filter(
            pl.col("timestamp").is_between(
                split["val_start"].date(), split["val_end"].date(), closed="both"
            )
        )
        assert not window.is_empty()
        assert split["train_end"].date() < window["timestamp"].min()


def test_notebook_reads_the_temporal_panel_through_the_alignment_function() -> None:
    """The panel is selected by validation window, never collapsed by dropping a key.

    `drop("fold").unique` was the collapse this guarded against when the artifact carried a
    fold; the artifact has none now, so the equivalent mistake is `unique` on the join keys
    with no window filter at all, which would silently admit training and holdout dates.
    """
    source = NOTEBOOK.read_text()
    assert 'drop("fold").unique' not in source
    assert "build_validation_temporal_panel(temporal, cv_folds)" in source
    assert f"temporal.unique(subset={JOIN_COLS}" not in source
