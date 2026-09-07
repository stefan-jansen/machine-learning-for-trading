"""A fold-scoped temporal artifact declares the window each of its folds was fitted over.

A producer that writes fold-scoped model-based features declares its rolling folds and appends
the holdout rows as one more fold *without* declaring that fold's geometry. The frame states
which fold ids exist and never states what bounded them, and
`temporal_artifact_fold_boundaries` falls back to `generate_cv_splits`, which returns the
cross-validation folds and nothing else - so the appended fold is invisible to every consumer
and the window its estimator was fitted over is recorded nowhere.

Measured on the production artifacts 2026-09-07: `us_equities_panel` is the one case study
whose `features/model_based.parquet` still carries a fold column, and its rows hold folds 0..16
against a resolved geometry of 0..15. Fold 16 holds 9,978,112 rows spanning 1990-01-30 to
2018-03-27 - the holdout - and nothing states the boundary it was estimated under, so
`require_fold_scoped_temporal_holdout_coverage` accepts it on trust.

Neither half of this costs a regeneration. No stage-04 producer in the repository currently
writes a fold column (all eight pass `fold_column=None`), so the producer-side refusal binds the
next time one does, which is exactly when the appended fold would otherwise arrive undeclared
again; and the consumer-side assertion has something to check the moment a producer declares it.

ml4t/agent-workspace#994.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.research.cv import require_fold_scoped_temporal_holdout_coverage
from case_studies.utils.temporal import write_model_based

ROLLING = [
    {
        "fold": 0,
        "train_start": "2016-01-04",
        "train_end": "2016-06-30",
        "val_start": "2016-07-01",
        "val_end": "2016-12-30",
    },
    {
        "fold": 1,
        "train_start": "2016-01-04",
        "train_end": "2016-12-30",
        "val_start": "2017-01-03",
        "val_end": "2017-06-30",
    },
]
HOLDOUT = {
    "fold": 2,
    "train_start": "2016-01-04",
    "train_end": "2017-06-30",
    "val_start": "2017-07-03",
    "val_end": "2017-12-29",
}

WRITE_KW = dict(
    keys=["timestamp", "symbol"],
    feature_columns=["feature"],
    time_column="timestamp",
    written_by="tests/test_temporal_fold_declaration.py",
)


def _frame(folds: range) -> pl.DataFrame:
    days = [date(2016, 1, 4), date(2016, 7, 1), date(2017, 1, 3), date(2017, 7, 3)]
    return pl.DataFrame(
        {
            "fold": [f for f in folds for _ in days],
            "timestamp": [d for _ in folds for d in days],
            "symbol": ["AAA"] * (len(days) * len(folds)),
            "feature": [float(f) + i for f in folds for i in range(len(days))],
        }
    )


# ---------------------------------------------------------------------------
# The producer states the geometry of every fold it writes
# ---------------------------------------------------------------------------


def test_a_fold_scoped_artifact_declaring_every_fold_is_written(tmp_path: Path) -> None:
    """The control: a producer that declares what it writes is unaffected."""
    record = write_model_based(
        _frame(range(3)),
        tmp_path / "model_based.parquet",
        fold_column="fold",
        metadata={"fold_geometry": [*ROLLING, HOLDOUT]},
        **WRITE_KW,
    )
    assert [f["fold"] for f in record["fold_geometry"]] == [0, 1, 2]


def test_an_appended_fold_with_no_declared_geometry_is_refused(tmp_path: Path) -> None:
    """The defect, exactly: rolling folds declared, the holdout fold appended silently."""
    with pytest.raises(ValueError, match=r"\[2\] would be written with no recorded boundary"):
        write_model_based(
            _frame(range(3)),
            tmp_path / "model_based.parquet",
            fold_column="fold",
            metadata={"fold_geometry": ROLLING},
            **WRITE_KW,
        )
    assert not (tmp_path / "model_based.parquet").exists(), "refused before anything is written"


def test_a_fold_scoped_artifact_declaring_nothing_is_refused(tmp_path: Path) -> None:
    """The state every fold-scoped artifact on disk is in today."""
    with pytest.raises(ValueError, match="must declare the geometry of its folds"):
        write_model_based(
            _frame(range(3)),
            tmp_path / "model_based.parquet",
            fold_column="fold",
            **WRITE_KW,
        )


def test_a_fold_free_artifact_declares_nothing(tmp_path: Path) -> None:
    """All eight stage-04 producers write fold-free artifacts, and none of them is asked."""
    frame = _frame(range(1)).drop("fold").unique(subset=["timestamp", "symbol"])
    record = write_model_based(
        frame, tmp_path / "model_based.parquet", fold_column=None, **WRITE_KW
    )
    assert "fold_geometry" not in record


def test_a_malformed_declaration_fails_at_the_write(tmp_path: Path) -> None:
    """Validated under the rule the consumer reads it back with, not a looser one.

    A geometry that only fails on the read side fails in a notebook hours later, on a machine
    that no longer holds the frame that produced it.
    """
    with pytest.raises(ValueError, match="invalid temporal fold geometry"):
        write_model_based(
            _frame(range(2)),
            tmp_path / "model_based.parquet",
            fold_column="fold",
            metadata={"fold_geometry": [{"fold": 0}, {"fold": 1}]},
            **WRITE_KW,
        )


# ---------------------------------------------------------------------------
# The consumer checks the declaration where there is one
# ---------------------------------------------------------------------------


def _coverage(declared, *, holdout=HOLDOUT):
    frame = _frame(range(3))
    require_fold_scoped_temporal_holdout_coverage(
        holdout,
        frame,
        source_timeline=frame.get_column("timestamp"),
        declared_folds=declared,
    )


def test_a_declared_holdout_fold_fitted_before_its_evaluation_window_passes() -> None:
    """`train_end` 2017-06-30 against an evaluation window opening 2017-07-03."""
    _coverage([*ROLLING, HOLDOUT])


def test_a_declared_holdout_fold_fitted_into_its_evaluation_window_is_refused() -> None:
    """The check the declaration buys: the estimator saw the sessions it is scored on."""
    leaked = {**HOLDOUT, "train_end": "2017-09-29"}
    with pytest.raises(ValueError, match="reaches the holdout evaluation window"):
        _coverage([*ROLLING, leaked])


def test_an_undeclared_holdout_fold_is_still_covered_on_trust() -> None:
    """Where the artifact declares no such fold, coverage is all that can be asked.

    Refusing here instead would refuse every holdout lock on the artifacts that exist today,
    which is the trade #994 rules out: the property is enforced by the producer's own assertion
    and that source is inside the sha256 the artifact is pinned by.
    """
    _coverage(ROLLING)
