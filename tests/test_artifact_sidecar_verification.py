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


def _tiny_case_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stale: bool):
    """A case study whose `financial.parquet` sidecar either agrees with it or does not."""
    from datetime import datetime

    import utils.modeling as modeling

    case_dir = tmp_path / "cs"
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "features").mkdir()
    (case_dir / "labels").mkdir()
    (case_dir / "config" / "setup.yaml").write_text("labels:\n  primary: primary\n  buffer: 1D\n")

    days = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 3, 1), "1d", eager=True).dt.date()
    keys = {"timestamp": days, "symbol": ["AAA"] * len(days)}
    write_artifact(
        pl.DataFrame({**keys, "primary": [0.1] * len(days)}),
        case_dir / "labels" / "primary.parquet",
        keys=["timestamp", "symbol"],
        written_by="test",
    )
    features = case_dir / "features" / "financial.parquet"
    write_artifact(
        pl.DataFrame({**keys, "feature": [1.0] * len(days)}),
        features,
        keys=["timestamp", "symbol"],
        written_by="test",
    )
    if stale:
        # The values move and the sidecar beside them does not, which is the defect.
        pl.DataFrame({**keys, "feature": [9.0] * len(days)}).write_parquet(features)

    splits = [
        {
            "fold": 0,
            "train_start": days[0],
            "train_end": days[20],
            "val_start": days[25],
            "val_end": days[-1],
        }
    ]
    monkeypatch.setattr(modeling, "get_case_study_dir", lambda _case_id: case_dir)
    monkeypatch.setattr(modeling, "load_feature_spec", lambda *_args: {})
    monkeypatch.setattr(modeling, "load_label_spec", lambda *_args: {})
    monkeypatch.setattr(
        modeling, "resolve_storage_path", lambda _case_id, _spec, fallback: case_dir / fallback
    )
    monkeypatch.setattr(modeling, "resolve_label_buffer", lambda *_args: "1D")
    monkeypatch.setattr(modeling, "resolve_label_horizon", lambda *_args: "1D")
    monkeypatch.setattr(modeling, "generate_cv_splits", lambda *_a, **_k: splits)
    monkeypatch.setattr(modeling, "make_wf_config", lambda *_args, **_kwargs: None)
    return modeling


def test_the_loader_refuses_a_stale_sidecar_under_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the helper's tests pass with no production call site at all.

    Every other test in this file calls `verify_artifact_sidecars` directly, so
    deleting the call inside `load_modeling_dataset` would leave them all green and
    put the defect back. This one fails if the call goes.
    """
    modeling = _tiny_case_study(tmp_path, monkeypatch, stale=True)

    with pytest.raises(ValueError, match="hashes .*, its sidecar records"):
        modeling.load_modeling_dataset("cs", "primary", verify_input_digests=True)


def test_the_loader_is_silent_about_digests_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifying a content digest re-reads and re-hashes the whole artifact.

    That is why the flag exists rather than being on: the cost is a multi-GB read
    plus a sort of every row hash, per load, and the state it catches is created
    only by regenerating an artifact. `scripts/verify_artifact_sidecars.py` is where
    it is paid once, over everything.
    """
    modeling = _tiny_case_study(tmp_path, monkeypatch, stale=True)

    assert modeling.load_modeling_dataset("cs", "primary").label_col == "primary"


def test_the_loader_accepts_an_artifact_that_matches_its_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modeling = _tiny_case_study(tmp_path, monkeypatch, stale=False)

    assert modeling.load_modeling_dataset("cs", "primary", verify_input_digests=True).label_col == (
        "primary"
    )


def test_the_cheap_check_catches_an_artifact_whose_row_count_moved(tmp_path: Path) -> None:
    """`write_artifact` writes the parquet first, so a crash leaves new data, old record.

    The row count comes out of parquet metadata, so this costs no column read and can
    run on every load.
    """
    path = _artifact(tmp_path)
    pl.DataFrame({"timestamp": [1], "symbol": ["A"], "feature": [1.0]}).write_parquet(path)

    with pytest.raises(ValueError, match="holds 1 rows, its sidecar records 3"):
        verify_artifact_sidecars({"financial": path}, values=False)


def test_the_cheap_check_does_not_see_a_value_change_at_the_same_row_count(
    tmp_path: Path,
) -> None:
    """Stated rather than left implicit: this is what `values=True` is for."""
    path = _artifact(tmp_path)
    pl.DataFrame(
        {"timestamp": [1, 2, 3], "symbol": ["A", "B", "C"], "feature": [9.0, 9.0, 9.0]}
    ).write_parquet(path)

    assert verify_artifact_sidecars({"financial": path}, values=False) == {}
    with pytest.raises(ValueError, match="hashes .*, its sidecar records"):
        verify_artifact_sidecars({"financial": path})


def test_the_loader_catches_a_truncated_artifact_without_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default load is not silent about everything, only about values."""
    modeling = _tiny_case_study(tmp_path, monkeypatch, stale=False)
    features = tmp_path / "cs" / "features" / "financial.parquet"
    pl.read_parquet(features).head(3).write_parquet(features)

    with pytest.raises(ValueError, match="holds 3 rows, its sidecar records"):
        modeling.load_modeling_dataset("cs", "primary")
