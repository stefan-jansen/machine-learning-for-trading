"""End-to-end tests for atomic case-study artifact installation."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import stat
import tarfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_artifacts.py"
SPEC = importlib.util.spec_from_file_location("download_artifacts", SCRIPT_PATH)
assert SPEC and SPEC.loader
download_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_artifacts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_bundle(tmp_path: Path, *, valid_manifest: bool = True) -> Path:
    run_log = tmp_path / "payload/case_studies/etfs/run_log"
    prediction = run_log / "predictions/abc/predictions.parquet"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"stored predictions")

    with sqlite3.connect(run_log / "registry.db") as connection:
        connection.execute("CREATE TABLE release_probe (value TEXT NOT NULL)")
        connection.execute("INSERT INTO release_probe VALUES ('accepted')")

    release_dir = run_log / ".release"
    release_dir.mkdir()
    registry_hash = _sha256(run_log / "registry.db")
    prediction_hash = _sha256(prediction)
    if not valid_manifest:
        prediction_hash = "0" * 64
    (release_dir / "SHA256SUMS").write_text(
        f"{registry_hash}  ./registry.db\n"
        f"{prediction_hash}  ./predictions/abc/predictions.parquet\n"
    )

    archive = tmp_path / "etfs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "payload/case_studies", arcname="case_studies")
    return archive


def test_corrupt_archive_does_not_touch_existing_run_log(tmp_path: Path) -> None:
    archive = _build_bundle(tmp_path)
    existing = tmp_path / "repo/case_studies/etfs/run_log"
    existing.mkdir(parents=True)
    (existing / "sentinel").write_text("keep")

    with pytest.raises(ValueError, match="archive checksum"):
        download_artifacts.install_artifact_archive(
            archive,
            "etfs",
            expected_sha256="0" * 64,
            repo_root=tmp_path / "repo",
            force=True,
        )

    assert (existing / "sentinel").read_text() == "keep"


def test_corrupt_internal_manifest_does_not_touch_existing_run_log(tmp_path: Path) -> None:
    archive = _build_bundle(tmp_path, valid_manifest=False)
    existing = tmp_path / "repo/case_studies/etfs/run_log"
    existing.mkdir(parents=True)
    (existing / "sentinel").write_text("keep")

    with pytest.raises(ValueError, match="artifact checksum"):
        download_artifacts.install_artifact_archive(
            archive,
            "etfs",
            expected_sha256=_sha256(archive),
            repo_root=tmp_path / "repo",
            force=True,
        )

    assert (existing / "sentinel").read_text() == "keep"


def test_verified_bundle_replaces_baseline_atomically(tmp_path: Path) -> None:
    archive = _build_bundle(tmp_path)
    existing = tmp_path / "repo/case_studies/etfs/run_log"
    existing.mkdir(parents=True)
    (existing / "sentinel").write_text("replace")

    installed = download_artifacts.install_artifact_archive(
        archive,
        "etfs",
        expected_sha256=_sha256(archive),
        repo_root=tmp_path / "repo",
        force=True,
    )

    assert installed == existing
    assert not (existing / "sentinel").exists()
    assert (existing / "predictions/abc/predictions.parquet").read_bytes() == b"stored predictions"
    assert not (existing / "registry.db").stat().st_mode & stat.S_IWUSR
    with sqlite3.connect(existing / "registry.db") as connection:
        assert connection.execute("SELECT value FROM release_probe").fetchone() == ("accepted",)
