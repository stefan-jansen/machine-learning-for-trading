from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CVSpec, LabelDefinition, Study
from case_studies.research.contracts import ExecutionTier
from case_studies.utils import linear
from case_studies.utils.registry.store import _open_registry
from utils import modeling
from utils.artifact_specs import load_setup_config
from utils.paths import get_case_study_dir


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


def _seed_regeneration_release(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    release = _seed_release(tmp_path)
    case_dir = release / "case_studies" / "etfs"
    generated_root = release / "generated" / "etfs"
    targets = {name: generated_root / name for name in ("features", "labels", "run_log")}
    generated_root.mkdir(parents=True)
    (case_dir / "run_log").rename(targets["run_log"])
    targets["features"].mkdir()
    targets["labels"].mkdir()
    for name, target in targets.items():
        (case_dir / name).symlink_to(target, target_is_directory=True)

    (case_dir / "config" / "setup.yaml").write_text(
        "\n".join(
            [
                "labels:",
                "  primary: fwd_ret_1d",
                "  buffer: 0D",
                "  horizons: {fwd_ret_1d: 0D}",
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
    (release / "case_studies" / "config" / "linear" / "ridge.yaml").write_text(
        "\n".join(
            [
                "config_name: ridge",
                "family: linear",
                "library: sklearn",
                "model_class: Ridge",
                "params:",
                "  alpha: 1.0",
            ]
        )
        + "\n"
    )
    rows = []
    for date_index, timestamp in enumerate(
        pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 10), eager=True)
    ):
        for symbol_index in range(6):
            x1 = float(symbol_index - 2.5)
            x2 = float(date_index) + x1 / 10
            rows.append(
                {
                    "symbol": f"S{symbol_index}",
                    "timestamp": timestamp,
                    "x1": x1,
                    "x2": x2,
                    "fwd_ret_1d": 0.03 * x1 + 0.01 * x2,
                }
            )
    frame = pl.DataFrame(rows)
    frame.select("symbol", "timestamp", "x1", "x2").write_parquet(
        targets["features"] / "financial.parquet"
    )
    frame.select("symbol", "timestamp", "fwd_ret_1d").write_parquet(
        targets["labels"] / "fwd_ret_1d.parquet"
    )
    _open_registry(case_dir).close()
    return release, targets


def test_study_create_and_reopen_do_not_change_release_bytes(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    before = _tree_digest(release)

    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    reopened = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    assert study.root == reopened.root == tmp_path / "workspace" / "etfs"
    assert study.manifest == reopened.manifest
    assert _tree_digest(release) == before


def test_seeded_output_root_becomes_the_explicit_isolated_preview_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    release = _seed_release(tmp_path)
    release_before = _tree_digest(release)
    isolated = tmp_path / "isolated"
    seeded_case = isolated / "etfs"
    (seeded_case / "config").mkdir(parents=True)
    (seeded_case / "config" / "setup.yaml").write_text("marker: seeded\n")
    _open_registry(seeded_case).close()
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(isolated))

    study = Study.open("etfs", workspace=isolated, release_root=release)
    training = study.results.register_training(
        {
            "identity_version": 2,
            "execution_tier": "preview",
            "family": "linear",
            "label": "fwd_ret_21d",
            "config_name": "ridge",
            "seed": 42,
            "preview_reductions": {"folds": [0]},
        },
        execution_tier="preview",
    )

    preview_case = isolated / ".preview" / "etfs"
    assert study.root == seeded_case
    assert study.manifest["adopted_output_root"] is True
    assert training.root == preview_case
    assert (seeded_case / ".study.json").is_file()
    assert (seeded_case / "run_log" / "registry.db").is_file()
    assert (preview_case / "run_log" / "registry.db").is_file()
    assert (preview_case / "run_log" / "training" / training.hash / "spec.json").is_file()
    assert _tree_digest(release) == release_before
    assert not (tmp_path / "experiments").exists()


@pytest.mark.parametrize("regular_directory", ["features", "labels", "run_log"])
def test_regeneration_rejects_regular_generated_artifact_directories(
    tmp_path: Path, regular_directory: str
) -> None:
    release, _ = _seed_regeneration_release(tmp_path)
    path = release / "case_studies" / "etfs" / regular_directory
    path.unlink()
    path.mkdir()

    with pytest.raises(PermissionError, match=regular_directory):
        Study.regenerate("etfs", release_root=release)


def test_regeneration_writes_through_resolved_directory_symlinks(tmp_path: Path) -> None:
    release, targets = _seed_regeneration_release(tmp_path)
    study = Study.regenerate("etfs", release_root=release)

    assert study.root == release / "case_studies" / "etfs"
    assert study.root == get_case_study_dir("etfs")
    assert all(
        (study.root / name).resolve(strict=True) == target for name, target in targets.items()
    )

    feature_probe = pl.DataFrame(
        {"symbol": ["S0"], "timestamp": ["2024-01-01"], "value": [1.0]}
    ).with_columns(pl.col("timestamp").str.to_date())
    feature_probe.write_parquet(get_case_study_dir("etfs") / "features" / "probe.parquet")
    published = study.labels.publish(
        LabelDefinition("probe", "regression", "1D"),
        feature_probe.rename({"value": "probe"}),
    )
    training = study.results.register_training(
        {
            "identity_version": 2,
            "execution_tier": "canonical",
            "family": "linear",
            "label": "probe",
            "config_name": "ridge",
            "seed": 42,
        }
    )

    assert (targets["features"] / "probe.parquet").is_file()
    assert published.path.resolve().parent == targets["labels"]
    assert (targets["run_log"] / "training" / training.hash / "spec.json").is_file()


def test_regeneration_preview_runs_real_model_without_changing_canonical_registry(
    tmp_path: Path, monkeypatch
) -> None:
    release, targets = _seed_regeneration_release(tmp_path)
    study = Study.regenerate("etfs", release_root=release)
    canonical_registry = targets["run_log"] / "registry.db"
    canonical_bytes = canonical_registry.read_bytes()
    with sqlite3.connect(canonical_registry) as db:
        canonical_training_rows = db.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
    loaded_from = None
    real_loader = modeling.load_modeling_dataset

    def observed_loader(*args, **kwargs):
        nonlocal loaded_from
        loaded_from = get_case_study_dir("etfs", create=False)
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(linear, "load_modeling_dataset", observed_loader)
    run = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        execution_tier=ExecutionTier.PREVIEW,
        preview_reductions={"folds": [0], "max_symbols": 3, "train_sample_frac": 1.0},
    ).run()

    preview_case = release / "case_studies" / ".preview" / "etfs"
    assert loaded_from == preview_case
    assert (preview_case / "config").is_symlink()
    assert (preview_case / "config").resolve(strict=True) == study.root / "config"
    preview_shared_config = release / "case_studies" / ".preview" / "config"
    canonical_shared_config = release / "case_studies" / "config"
    assert preview_shared_config.is_symlink()
    assert preview_shared_config.resolve(strict=True) == canonical_shared_config
    assert (preview_case / "features").is_symlink()
    assert (preview_case / "features").resolve(strict=True) == targets["features"]
    assert (preview_case / "labels").is_symlink()
    assert (preview_case / "labels").resolve(strict=True) == targets["labels"]
    assert run.training.root == preview_case
    assert (
        preview_case / "run_log" / "training" / run.training.hash / "models" / "manifest.json"
    ).is_file()
    assert run.predictions[0].complete
    assert canonical_registry.read_bytes() == canonical_bytes
    with sqlite3.connect(canonical_registry) as db:
        assert (
            db.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            == canonical_training_rows
        )
    assert not (targets["run_log"] / "training").exists()
    assert not (targets["run_log"] / "predictions").exists()
    assert not (targets["run_log"] / "backtest").exists()

    preview_request = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        execution_tier=ExecutionTier.PREVIEW,
        preview_reductions={"folds": [0], "max_symbols": 3, "train_sample_frac": 1.0},
    )
    before_change = preview_request.resolve()
    setup_path = study.root / "config" / "setup.yaml"
    setup_path.write_text(setup_path.read_text().replace("train_size: 4D", "train_size: 3D"))
    after_change = preview_request.resolve()

    assert before_change.spec["computation"]["cv"] != after_change.spec["computation"]["cv"]
    canonical_preset = canonical_shared_config / "linear" / "ridge.yaml"
    canonical_preset.unlink()
    assert not (preview_shared_config / "linear" / "ridge.yaml").exists()


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


def test_label_lookup_reactivates_its_owning_workspace(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    first = Study.open("etfs", workspace=tmp_path / "first", release_root=release)
    frame = pl.DataFrame(
        {"symbol": ["A"], "timestamp": ["2024-01-01"], "custom": [0.1]}
    ).with_columns(pl.col("timestamp").str.to_date())
    published = first.labels.publish(LabelDefinition("custom", "regression", "1D"), frame)
    Study.open("etfs", workspace=tmp_path / "second", release_root=release)

    resolved = first.labels.get("custom")

    assert resolved.path == published.path
    assert get_case_study_dir("etfs") == first.root


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
