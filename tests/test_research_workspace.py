from __future__ import annotations

import hashlib
import os
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CVSpec, LabelDefinition, Study
from utils.artifact_specs import load_setup_config


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _seed_release(tmp_path: Path, *, marker: str = "release") -> Path:
    release = tmp_path / "release"
    case_dir = release / "case_studies" / "etfs"
    (case_dir / "run_log").mkdir(parents=True)
    (case_dir / "run_log" / "registry.db").write_bytes(b"")
    (case_dir / "config").mkdir()
    (case_dir / "config" / "setup.yaml").write_text(
        "\n".join(
            [
                f"marker: {marker}",
                "labels:",
                "  primary: fwd_ret_21d",
                "  buffer: 2D",
                "  horizons: {fwd_ret_21d: 2D}",
                "evaluation:",
                "  n_splits: 2",
                "  train_size: 4D",
                "  val_size: 2D",
                "  holdout_start: '2024-01-11'",
                "  holdout_end: '2024-01-12'",
                "  calendar: crypto",
                "decision:",
                "  cadence: daily_close",
                "  execution_delay: next_bar_open",
                "mapping:",
                "  position_state_space: long_only",
                "costs:",
                "  class: negligible",
            ]
        )
        + "\n"
    )
    shared = release / "case_studies" / "config" / "linear"
    shared.mkdir(parents=True)
    (shared / "ridge.yaml").write_text("family: linear\nparams: {}\n")
    return release


def test_study_create_and_reopen_do_not_change_release_bytes(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    before = _tree_digest(release)

    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    reopened = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    assert study.root == reopened.root == tmp_path / "workspace" / "etfs"
    assert study.manifest == reopened.manifest
    assert _tree_digest(release) == before


def test_release_study_is_read_only(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)

    study = Study.open("etfs", release_root=release)

    assert study.read_only
    with pytest.raises(PermissionError, match="read-only"):
        study.labels.publish(
            LabelDefinition("custom", "regression", "1D"),
            pl.DataFrame(
                {
                    "symbol": ["A"],
                    "timestamp": [pl.date(2024, 1, 1)],
                    "custom": [0.1],
                }
            ),
        )


def test_switching_workspaces_invalidates_root_sensitive_config_cache(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    first = Study.open("etfs", workspace=tmp_path / "first", release_root=release)
    (first.root / "config" / "setup.yaml").write_text(
        (first.root / "config" / "setup.yaml").read_text().replace("release", "first")
    )
    assert load_setup_config("etfs")["marker"] == "first"

    second = Study.open("etfs", workspace=tmp_path / "second", release_root=release)
    (second.root / "config" / "setup.yaml").write_text(
        (second.root / "config" / "setup.yaml").read_text().replace("release", "second")
    )

    assert load_setup_config("etfs")["marker"] == "second"


def test_custom_classification_label_resolves_continuous_target(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    frame = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": ["2024-01-01", "2024-01-01"],
            "up_90p_5d": [0, 1],
            "fwd_ret_5d": [-0.01, 0.03],
        }
    ).with_columns(pl.col("timestamp").str.to_date())

    published = study.labels.publish(
        LabelDefinition(
            name="up_90p_5d",
            task_type="classification",
            horizon="5D",
            continuous_eval_label="fwd_ret_5d",
        ),
        frame,
    )
    reopened = Study.open("etfs", workspace=tmp_path / "workspace", release_root=study.release_root)
    resolved = reopened.labels.get("up_90p_5d")

    assert published.digest == resolved.digest
    assert resolved.definition.continuous_eval_label == "fwd_ret_5d"
    assert resolved.load().get_column("fwd_ret_5d").to_list() == [-0.01, 0.03]


def test_invalid_label_publication_has_no_partial_writes(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    definition = LabelDefinition(
        name="up_90p_5d",
        task_type="classification",
        horizon="5D",
        continuous_eval_label="fwd_ret_5d",
    )
    invalid = pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": ["2024-01-01", "2024-01-01"],
            "up_90p_5d": [0, 1],
        }
    )

    with pytest.raises(ValueError):
        study.labels.publish(definition, invalid)

    assert not (study.root / "labels" / "up_90p_5d.parquet").exists()
    assert not (study.root / "labels" / "up_90p_5d.parquet.digest.json").exists()


def test_label_discovery_rejects_artifact_digest_mismatch(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    frame = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": ["2024-01-01"],
            "custom": [0.1],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    published = study.labels.publish(LabelDefinition("custom", "regression", "1D"), frame)
    frame.with_columns(pl.lit(0.2).alias("custom")).write_parquet(published.path)

    with pytest.raises(ValueError, match="digest"):
        study.labels.get("custom")


def test_cvspec_resolves_stable_exact_boundaries_and_changes_with_protocol() -> None:
    timeline = pl.DataFrame(
        {"timestamp": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 12), eager=True)}
    )
    base = CVSpec.walk_forward(
        training_window="4D",
        validation_window="2D",
        retrain_every="2D",
        folds=range(2),
        horizon="1D",
        holdout_start="2024-01-11",
        holdout_end="2024-01-12",
        calendar=None,
    )

    first = base.resolve(timeline)
    second = base.resolve(timeline)
    reordered = base.with_changes(folds=(1, 0)).resolve(timeline)
    changed = base.with_changes(training_window="3D").resolve(timeline)

    assert first == second == reordered
    assert first.normalized_folds != changed.normalized_folds
    assert first.identity != changed.identity


def test_custom_cv_cannot_relabel_fold_scoped_temporal_features() -> None:
    from case_studies.research.cv import require_fold_scoped_temporal_compatibility

    artifact = [
        {
            "fold": 0,
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "val_start": "2021-01-01",
            "val_end": "2021-03-31",
        },
        {
            "fold": 1,
            "train_start": "2020-04-01",
            "train_end": "2021-03-31",
            "val_start": "2021-04-01",
            "val_end": "2021-06-30",
        },
    ]

    require_fold_scoped_temporal_compatibility([artifact[1]], artifact)
    changed = [{**artifact[1], "train_end": "2021-05-31"}]
    with pytest.raises(ValueError, match="incompatible with fold-scoped temporal features"):
        require_fold_scoped_temporal_compatibility(changed, artifact)
