"""The digest sidecar gets a read side.

`artifact_digest.py` writes `<artifact>.digest.json` beside every artifact and its
own docstring said the chain stopped there: "It is a record, not propagation.
Nothing downstream reads these sidecars today." The registry records feature-set
names and no digest of feature values, so a model trained on corrected features and
one trained on the leaky version it replaced produce the identical training_hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.artifact_digest import value_digest, write_artifact
from utils.modeling import verify_artifact_sidecars


def _artifact(tmp_path: Path, name: str = "financial", values=(1.0, 2.0, 3.0)) -> Path:
    frame = pl.DataFrame(
        {"timestamp": [1, 2, 3], "symbol": ["A", "B", "C"], "feature": list(values)}
    )
    path = tmp_path / f"{name}.parquet"
    write_artifact(frame, path, keys=["timestamp", "symbol"], written_by="test")
    return path


def test_an_artifact_matching_its_sidecar_verifies(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    assert verify_artifact_sidecars({"financial": path}) == {
        "financial": value_digest(pl.read_parquet(path))
    }


def test_a_changed_feature_value_is_caught(tmp_path: Path) -> None:
    """The case the issue is about: values move, the sidecar does not."""
    path = _artifact(tmp_path)
    pl.DataFrame(
        {"timestamp": [1, 2, 3], "symbol": ["A", "B", "C"], "feature": [9.0, 9.0, 9.0]}
    ).write_parquet(path)

    with pytest.raises(ValueError, match="hashes .*, its sidecar records"):
        verify_artifact_sidecars({"financial": path})


def test_an_artifact_with_no_sidecar_is_refused(tmp_path: Path) -> None:
    """An artifact whose producer recorded nothing cannot be told from a changed one."""
    bare = tmp_path / "model_based.parquet"
    pl.DataFrame({"timestamp": [1], "symbol": ["A"], "f": [1.0]}).write_parquet(bare)

    with pytest.raises(ValueError, match="no digest sidecar"):
        verify_artifact_sidecars({"model_based": bare})


def test_a_sidecar_recording_no_digest_is_refused(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    side = Path(f"{path}.digest.json")
    side.write_text(json.dumps({"n_rows": 3}))

    with pytest.raises(ValueError, match="records no digest"):
        verify_artifact_sidecars({"financial": path})


def test_an_unreadable_sidecar_is_refused(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    Path(f"{path}.digest.json").write_text("{not json")

    with pytest.raises(ValueError, match="unreadable"):
        verify_artifact_sidecars({"financial": path})


def test_a_missing_sidecar_is_skipped_when_presence_is_not_required(tmp_path: Path) -> None:
    """The CI fixtures predate the sidecar, so absence must not fail an ordinary load.

    A skipped artifact is absent from the returned mapping rather than present with a
    null, so a caller carrying these into a training spec records what it verified and
    not what it declined to.
    """
    good = _artifact(tmp_path, "financial")
    bare = tmp_path / "model_based.parquet"
    pl.DataFrame({"timestamp": [1], "symbol": ["A"], "f": [1.0]}).write_parquet(bare)

    verified = verify_artifact_sidecars(
        {"financial": good, "model_based": bare}, require_sidecar=False
    )
    assert verified == {"financial": value_digest(pl.read_parquet(good))}


def test_a_disagreeing_sidecar_is_refused_even_when_presence_is_not_required(
    tmp_path: Path,
) -> None:
    """Absence and disagreement carry different evidence, and only absence is excused.

    This is the case the issue is about, and it is the one that has to fire on an
    ordinary load: the values moved and the record did not move with them.
    """
    path = _artifact(tmp_path)
    pl.DataFrame(
        {"timestamp": [1, 2, 3], "symbol": ["A", "B", "C"], "feature": [9.0, 9.0, 9.0]}
    ).write_parquet(path)

    with pytest.raises(ValueError, match="hashes .*, its sidecar records"):
        verify_artifact_sidecars({"financial": path}, require_sidecar=False)


def test_every_failing_artifact_is_named_not_just_the_first(tmp_path: Path) -> None:
    """A run with two stale inputs should report two, so one pass fixes both."""
    good = _artifact(tmp_path, "financial")
    stale = _artifact(tmp_path, "label")
    pl.DataFrame({"timestamp": [1], "symbol": ["A"], "feature": [0.0]}).write_parquet(stale)
    bare = tmp_path / "model_based.parquet"
    pl.DataFrame({"timestamp": [1], "symbol": ["A"], "f": [1.0]}).write_parquet(bare)

    with pytest.raises(ValueError) as excinfo:
        verify_artifact_sidecars({"financial": good, "label": stale, "model_based": bare})
    message = str(excinfo.value)
    assert "label:" in message and "model_based:" in message and "financial:" not in message
