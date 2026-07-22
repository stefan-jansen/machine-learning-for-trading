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


def test_create_experiment_copies_baseline_without_mutating_source(tmp_path: Path) -> None:
    module = _load_module()
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
