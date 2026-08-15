"""Tests for isolated case-study experiment setup."""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "create_experiment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("create_experiment", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_source(tmp_path: Path) -> Path:
    """Build a minimal etfs source tree (registry + features + config)."""
    source = tmp_path / "repo/case_studies/etfs"
    registry = source / "run_log/registry.db"
    registry.parent.mkdir(parents=True)
    registry.write_bytes(b"accepted registry")
    release = source / "run_log/.release"
    release.mkdir()
    (release / "SHA256SUMS").write_text("manifest")
    features = source / "features/features.parquet"
    features.parent.mkdir()
    features.write_bytes(b"features")
    # Per-case-study config (input the reader edits to define the experiment).
    config = source / "config"
    (config / "training").mkdir(parents=True)
    (config / "setup.yaml").write_text("modeling:\n  gbm:\n    device: cpu\n")
    (config / "training/fwd_ret_21d.yaml").write_text("gbm:\n- default_mse\n")
    # Shared model presets resolved at {ML4T_OUTPUT_DIR}/config/{model_type}/.
    shared = tmp_path / "repo/case_studies/config/lgb"
    shared.mkdir(parents=True)
    (shared / "default_mse.yaml").write_text("params: {objective: regression}\n")
    return source


def test_create_experiment_copies_baseline_without_mutating_source(tmp_path: Path) -> None:
    module = _load_module()
    source = _seed_source(tmp_path)
    registry = source / "run_log/registry.db"
    release = source / "run_log/.release"
    registry.chmod(registry.stat().st_mode & ~stat.S_IWUSR)
    release.chmod(release.stat().st_mode & ~stat.S_IWUSR)
    registry.parent.chmod(registry.parent.stat().st_mode & ~stat.S_IWUSR)

    output_root = tmp_path / "experiments/etf-test"
    result = module.create_experiment("etfs", output_root, repo_root=tmp_path / "repo")

    assert result == output_root / "etfs"
    assert (result / "run_log/registry.db").read_bytes() == b"accepted registry"
    assert (result / "run_log/.baseline/SHA256SUMS").read_text() == "manifest"
    assert not (result / "run_log/.release").exists()
    assert (result / "features/features.parquet").read_bytes() == b"features"
    assert (result / "run_log/registry.db").stat().st_mode & stat.S_IWUSR
    assert not registry.stat().st_mode & stat.S_IWUSR
    registry.parent.chmod(registry.parent.stat().st_mode | stat.S_IWUSR)
    release.chmod(release.stat().st_mode | stat.S_IWUSR)


def test_create_experiment_seeds_editable_config_and_shared_presets(tmp_path: Path) -> None:
    """The documented workflow reads config through ML4T_OUTPUT_DIR, so the
    experiment must carry its own per-CS config and the shared preset tree, both
    writable so the reader can edit them (regression for the D7 config crash)."""
    module = _load_module()
    _seed_source(tmp_path)

    output_root = tmp_path / "experiments/etf-test"
    result = module.create_experiment("etfs", output_root, repo_root=tmp_path / "repo")

    # Per-CS config copied under the experiment's case-study dir.
    setup = result / "config/setup.yaml"
    training = result / "config/training/fwd_ret_21d.yaml"
    assert setup.exists() and training.exists()
    # Shared presets seeded at the output root so load_configs() resolves them at
    # {case_dir.parent}/config/{model_type}/.
    preset = output_root / "config/lgb/default_mse.yaml"
    assert preset.exists()
    # All copied config is writable (the reader edits it in place).
    assert setup.stat().st_mode & stat.S_IWUSR
    assert preset.stat().st_mode & stat.S_IWUSR
    # Source config is untouched by the experiment run.
    assert (tmp_path / "repo/case_studies/etfs/config/setup.yaml").exists()


def test_create_experiment_without_baseline_accepts_a_linked_release_run_log(
    tmp_path: Path,
) -> None:
    module = _load_module()
    source = _seed_source(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    linked_run_log = artifact_root / "run_log"
    (source / "run_log").rename(linked_run_log)
    (source / "run_log").symlink_to(linked_run_log, target_is_directory=True)

    output_root = tmp_path / "experiments/etf-test"
    result = module.create_experiment(
        "etfs",
        output_root,
        repo_root=tmp_path / "repo",
        include_release_run_log=False,
    )

    assert (source / "run_log").is_symlink()
    assert (result / "run_log").is_dir()
    assert not (result / "run_log/registry.db").exists()


def test_create_experiment_requires_config_tree(tmp_path: Path) -> None:
    """A case study with artifacts but no config/ cannot form a runnable experiment."""
    module = _load_module()
    source = tmp_path / "repo/case_studies/etfs"
    (source / "run_log").mkdir(parents=True)
    (source / "run_log/registry.db").write_bytes(b"registry")

    try:
        module.create_experiment("etfs", tmp_path / "out", repo_root=tmp_path / "repo")
    except ValueError as exc:
        assert "config" in str(exc)
    else:  # pragma: no cover - guard must raise
        raise AssertionError("expected ValueError for missing config/ tree")
