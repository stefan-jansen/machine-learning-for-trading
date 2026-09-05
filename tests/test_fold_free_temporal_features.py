"""A fold-free model-based artifact has to report which features it contributed.

`load_modeling_dataset` records `temporal_feature_names` on the per-fold branch and,
before this, recorded nothing on the fold-free one. The columns were joined into the
panel and fitted on regardless, so a stage that produces one value per key - which is
what a walk-forward refit schedule produces - reported no model-based features at all
while its features were in use. `nasdaq100_microstructure` has been on that branch the
whole time.

Every consumer of the list also requires `temporal_by_fold`, which stays None here, so
recording the names changes no fold substitution. What it changes is that a notebook
asking which columns came from stage 04 gets an answer.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.artifact_digest import write_artifact

TEMPORAL_FEATURES = ["garch_cond_vol", "kalman_trend"]


def _case_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, by_fold: bool):
    """A minimal case study whose model-based artifact is fold-keyed or is not."""
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
    write_artifact(
        pl.DataFrame({**keys, "feature": [1.0] * len(days)}),
        case_dir / "features" / "financial.parquet",
        keys=["timestamp", "symbol"],
        written_by="test",
    )

    splits_for_artifact = [
        {
            "fold": fold,
            "train_start": days[0],
            "train_end": days[20],
            "val_start": days[25],
            "val_end": days[-1],
        }
        for fold in (0, 1)
    ]
    temporal = pl.DataFrame({**keys, **{name: [0.5] * len(days) for name in TEMPORAL_FEATURES}})
    temporal_keys = ["timestamp", "symbol"]
    if by_fold:
        temporal = pl.concat([temporal.with_columns(pl.lit(fold).alias("fold")) for fold in (0, 1)])
        temporal_keys = ["fold", *temporal_keys]
        # The fold branch resolves each fold's window through a per-case-study
        # resolver, and this synthetic case study is not one of the registered
        # ones. The names the branch reports do not depend on that resolution.
        import case_studies.utils.cv_window as cv_window

        monkeypatch.setattr(
            cv_window,
            "temporal_artifact_fold_boundaries",
            lambda *_a, **_k: splits_for_artifact,
        )
    write_artifact(
        temporal,
        case_dir / "features" / "model_based.parquet",
        keys=temporal_keys,
        written_by="test",
    )

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


def test_a_fold_free_artifact_names_the_features_it_contributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modeling = _case_study(tmp_path, monkeypatch, by_fold=False)

    mds = modeling.load_modeling_dataset("cs", "primary")

    assert mds.temporal_by_fold is None, (
        "the artifact has no fold column, so there is nothing to substitute"
    )
    assert sorted(mds.temporal_feature_names) == sorted(TEMPORAL_FEATURES)


def test_the_named_features_are_the_ones_in_the_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list is only worth anything if it describes the frame that comes back."""
    modeling = _case_study(tmp_path, monkeypatch, by_fold=False)

    mds = modeling.load_modeling_dataset("cs", "primary")

    assert set(mds.temporal_feature_names) <= set(mds.dataset.columns)
    for name in mds.temporal_feature_names:
        assert mds.dataset[name].null_count() == 0


def test_neither_join_key_is_reported_as_a_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modeling = _case_study(tmp_path, monkeypatch, by_fold=False)

    mds = modeling.load_modeling_dataset("cs", "primary")

    assert "timestamp" not in mds.temporal_feature_names
    assert "symbol" not in mds.temporal_feature_names


def test_a_fold_keyed_artifact_still_reports_the_same_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two branches disagreed about this; they no longer do."""
    modeling = _case_study(tmp_path, monkeypatch, by_fold=True)

    mds = modeling.load_modeling_dataset("cs", "primary")

    assert mds.temporal_by_fold is not None
    assert sorted(mds.temporal_feature_names) == sorted(TEMPORAL_FEATURES)
    assert "fold" not in mds.temporal_feature_names
