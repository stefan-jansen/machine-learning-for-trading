"""Tests for the input lineage that gates training-hash reuse.

``07_gbm`` passes this payload to ``build_training_spec``, so a fingerprint that
does not move when an input artifact does lets a changed feature set reuse an old
training hash and overwrite its prediction artifact in place. Nothing asserted
either half of that before.
"""

from pathlib import Path

import pandas as pd
import pytest

from utils.modeling import build_modeling_input_lineage

BASE = {
    "feature_names": ["ret_1d", "vol_21d"],
    "label_buffer": "5d",
    "task_type": "regression",
    "eval_label_col": None,
    "max_symbols": 0,
    "symbols": None,
}


def _artifacts(tmp_path: Path, payload: bytes = b"features") -> dict[str, Path]:
    features = tmp_path / "features.parquet"
    features.write_bytes(payload)
    labels = tmp_path / "labels.parquet"
    labels.write_bytes(b"labels")
    return {"features": features, "labels": labels}


def _splits(val_end: str = "2020-12-23", tz: str | None = None) -> list[dict]:
    def ts(value: str) -> pd.Timestamp:
        return pd.Timestamp(value, tz=tz) if tz else pd.Timestamp(value)

    return [
        {
            "fold": 0,
            "train_start": ts("2015-01-02"),
            "train_end": ts("2018-12-31"),
            "val_start": ts("2019-01-07"),
            "val_end": ts(val_end),
        }
    ]


def test_the_fingerprint_is_stable_across_identical_loads(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    first = build_modeling_input_lineage(artifacts=artifacts, splits=_splits(), **BASE)
    second = build_modeling_input_lineage(artifacts=artifacts, splits=_splits(), **BASE)

    assert first["fingerprint"] == second["fingerprint"]


def test_a_changed_artifact_moves_the_fingerprint(tmp_path: Path) -> None:
    before = build_modeling_input_lineage(artifacts=_artifacts(tmp_path), splits=_splits(), **BASE)
    after = build_modeling_input_lineage(
        artifacts=_artifacts(tmp_path, b"features v2"), splits=_splits(), **BASE
    )

    assert before["fingerprint"] != after["fingerprint"]


def test_a_changed_cv_window_moves_the_fingerprint(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    before = build_modeling_input_lineage(artifacts=artifacts, splits=_splits(), **BASE)
    after = build_modeling_input_lineage(
        artifacts=artifacts, splits=_splits(val_end="2021-06-30"), **BASE
    )

    assert before["fingerprint"] != after["fingerprint"]


@pytest.mark.parametrize("field,value", [("max_symbols", 25), ("label_buffer", "21d")])
def test_a_changed_reduction_or_buffer_moves_the_fingerprint(
    tmp_path: Path, field: str, value: object
) -> None:
    artifacts = _artifacts(tmp_path)
    before = build_modeling_input_lineage(artifacts=artifacts, splits=_splits(), **BASE)
    after = build_modeling_input_lineage(
        artifacts=artifacts, splits=_splits(), **{**BASE, field: value}
    )

    assert before["fingerprint"] != after["fingerprint"]


def test_the_same_window_fingerprints_the_same_whether_or_not_it_is_tz_aware(
    tmp_path: Path,
) -> None:
    """str() on a Timestamp renders "+00:00" only when the caller's boundaries are
    tz-aware, so the same window read two ways used to fingerprint differently."""
    artifacts = _artifacts(tmp_path)
    naive = build_modeling_input_lineage(artifacts=artifacts, splits=_splits(), **BASE)
    aware = build_modeling_input_lineage(artifacts=artifacts, splits=_splits(tz="UTC"), **BASE)

    assert naive["fingerprint"] == aware["fingerprint"]


def test_the_cached_lineage_is_dropped_when_a_holdout_fold_is_appended(tmp_path: Path) -> None:
    """``input_lineage`` memoizes and ``splits`` is appended to in place, so a
    caller that read it before the append would persist a spec describing a fold
    set that no longer exists."""
    from utils.modeling import ModelingDataset

    artifacts = _artifacts(tmp_path)
    dataset = ModelingDataset(
        dataset=None,
        feature_names=list(BASE["feature_names"]),
        label_col="fwd_ret_5d",
        splits=_splits(),
        label_buffer=BASE["label_buffer"],
        task_type=BASE["task_type"],
        date_col="timestamp",
        entity_cols=["symbol"],
        join_cols=["symbol", "timestamp"],
        lineage_inputs={"artifacts": artifacts, "max_symbols": 0, "symbols": None},
    )
    before = dataset.input_lineage["fingerprint"]

    dataset.splits.append(
        {
            "fold": 1,
            "train_start": pd.Timestamp("2015-01-02"),
            "train_end": pd.Timestamp("2021-01-01"),
            "val_start": pd.Timestamp("2021-01-01"),
            "val_end": pd.Timestamp("2021-12-31"),
        }
    )
    dataset._input_lineage = None  # what append_holdout_fold_if_needed does

    assert dataset.input_lineage["fingerprint"] != before
