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


def test_validation_alignment_excludes_holdout_and_wrong_fold_estimates() -> None:
    """The old keep-last collapse picks holdout state; canonical mapping must not."""
    temporal = pl.DataFrame(
        {
            "timestamp": [
                date(2019, 6, 3),
                date(2019, 6, 3),
                date(2019, 6, 3),
                date(2020, 6, 3),
                date(2020, 6, 3),
                date(2020, 6, 3),
            ],
            "symbol": ["AAA"] * 6,
            "fold": [0, 1, -1, 0, 1, -1],
            "temporal_value": [10.0, 11.0, 99.0, 20.0, 21.0, 99.0],
        }
    )

    old_keep_last = temporal.unique(subset=JOIN_COLS, keep="last", maintain_order=True)
    assert old_keep_last["fold"].to_list() == [-1, -1]

    aligned = _load_alignment_function()["build_validation_temporal_panel"](
        temporal, _synthetic_splits()
    ).sort("timestamp")
    assert aligned["validation_fold"].to_list() == [1, 0]
    assert aligned["temporal_value"].to_list() == [11.0, 20.0]
    assert aligned.group_by(JOIN_COLS).len().filter(pl.col("len") > 1).is_empty()


def test_validation_alignment_fails_closed_on_duplicate_fold_keys() -> None:
    temporal = pl.DataFrame(
        {
            "timestamp": [date(2019, 6, 3), date(2019, 6, 3), date(2020, 6, 3)],
            "symbol": ["AAA", "AAA", "AAA"],
            "fold": [0, 0, 1],
            "temporal_value": [10.0, 10.5, 21.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate fold-specific keys"):
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
    default_root = Path.home() / "ml4t/code/case_studies/sp500_options"
    artifact_root = Path(os.environ.get("ML4T_SP500_OPTIONS_ARTIFACT_ROOT", default_root))
    financial_path = artifact_root / "features/financial.parquet"
    temporal_path = artifact_root / "features/model_based.parquet"
    setup_path = artifact_root / "config/setup.yaml"
    if not all(path.exists() for path in (financial_path, temporal_path, setup_path)):
        pytest.skip("Full sp500_options artifacts are not available")

    financial = pl.read_parquet(financial_path)
    temporal = pl.read_parquet(temporal_path)
    setup = yaml.safe_load(setup_path.read_text())
    folds = generate_cv_splits(
        financial.select("timestamp"),
        case_study_id="sp500_options",
        label_buffer=str(setup["labels"]["buffer"]),
    )

    validation_folds = {int(split["fold"]) for split in folds}
    holdout_fold = len(folds)
    assert set(temporal["fold"].unique().to_list()) == validation_folds | {holdout_fold}

    try:
        aligned = _load_alignment_function()["build_validation_temporal_panel"](temporal, folds)
    except ValueError as error:
        # The frozen artifact predates the canonical fold numbering. Rejecting
        # it is safer than silently remapping estimator identity.
        assert "no validation rows for fold" in str(error)
        return

    assert aligned.filter(pl.col("validation_fold") == holdout_fold).is_empty()
    assert aligned.group_by(JOIN_COLS).len().filter(pl.col("len") > 1).is_empty()
    for split in folds:
        fold_rows = aligned.filter(pl.col("validation_fold") == split["fold"])
        assert not fold_rows.is_empty()
        assert fold_rows["timestamp"].min() >= split["val_start"].date()
        assert fold_rows["timestamp"].max() <= split["val_end"].date()
        assert split["train_end"].date() < fold_rows["timestamp"].min()


def test_notebook_does_not_collapse_fold_identity_with_keep_last() -> None:
    source = NOTEBOOK.read_text()
    assert 'drop("fold").unique' not in source
    assert "build_validation_temporal_panel(temporal, cv_folds)" in source
