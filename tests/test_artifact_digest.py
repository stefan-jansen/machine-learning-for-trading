"""The value digest has to be sensitive to values and blind to everything else.

A digest that changed when rows were reordered would fire on every re-run; one that
did not change when a value moved would be worthless as a lineage record. Both
failure modes are silent, so both are tested.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from case_studies.utils.artifact_digest import (
    digest_sidecar,
    read_digest,
    sidecar_path,
    value_digest,
    write_artifact,
)


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "symbol": ["A", "A", "B", "B"],
            "fwd_ret_21d": [0.01, -0.02, 0.03, 0.04],
        }
    )


def test_digest_is_invariant_to_row_order(frame: pl.DataFrame) -> None:
    shuffled = frame.sort("fwd_ret_21d", descending=True)
    assert value_digest(shuffled) == value_digest(frame)


def test_digest_is_invariant_to_column_order(frame: pl.DataFrame) -> None:
    assert value_digest(frame.select(sorted(frame.columns, reverse=True))) == value_digest(frame)


def test_digest_changes_when_a_single_value_changes(frame: pl.DataFrame) -> None:
    moved = frame.with_columns(
        pl.when(pl.col("timestamp") == 1)
        .then(0.010000001)
        .otherwise(pl.col("fwd_ret_21d"))
        .alias("fwd_ret_21d")
    )
    assert value_digest(moved) != value_digest(frame)


def test_digest_changes_when_a_row_is_dropped(frame: pl.DataFrame) -> None:
    assert value_digest(frame.head(3)) != value_digest(frame)


def test_digest_changes_when_a_column_is_added(frame: pl.DataFrame) -> None:
    assert value_digest(frame.with_columns(extra=pl.lit(1))) != value_digest(frame)


def test_column_subset_is_honoured(frame: pl.DataFrame) -> None:
    keys = ["symbol", "timestamp"]
    relabelled = frame.with_columns(fwd_ret_21d=pl.lit(0.0))
    assert value_digest(relabelled, keys) == value_digest(frame, keys)
    assert value_digest(relabelled) != value_digest(frame)


def test_missing_column_raises(frame: pl.DataFrame) -> None:
    with pytest.raises(KeyError):
        value_digest(frame, ["not_a_column"])


def test_sidecar_records_the_shape_and_the_inputs(frame: pl.DataFrame) -> None:
    record = digest_sidecar(
        frame, keys=["timestamp", "symbol"], written_by="02_labels", inputs={"market_data": "abc"}
    )
    assert record["digest"] == value_digest(frame)
    assert record["n_rows"] == 4
    assert record["columns"] == ["fwd_ret_21d", "symbol", "timestamp"]
    assert record["keys"] == ["timestamp", "symbol"]
    assert record["written_by"] == "02_labels"
    assert record["inputs"] == {"market_data": "abc"}


def test_sidecar_rejects_a_key_the_frame_does_not_have(frame: pl.DataFrame) -> None:
    with pytest.raises(KeyError):
        digest_sidecar(frame, keys=["instrument_id"], written_by="02_labels")


def test_write_artifact_round_trips(frame: pl.DataFrame, tmp_path) -> None:
    path = tmp_path / "labels" / "fwd_ret_21d.parquet"
    record = write_artifact(
        frame, path, keys=["timestamp", "symbol"], written_by="02_labels", inputs={"m": "d"}
    )

    assert path.exists()
    assert sidecar_path(path) == path.parent / "fwd_ret_21d.parquet.digest.json"
    assert read_digest(path) == record
    assert json.loads(sidecar_path(path).read_text())["digest"] == value_digest(
        pl.read_parquet(path)
    )
