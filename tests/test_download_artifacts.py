"""End-to-end tests for atomic case-study artifact installation."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import stat
import tarfile
import urllib.error
from contextlib import closing
from pathlib import Path

import pytest

from utils.paths import registry_readonly_uri

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


def _build_bundle(
    tmp_path: Path,
    *,
    valid_manifest: bool = True,
    wal: bool = False,
    ship_wal: bool = False,
) -> Path:
    """Build one bundle. `wal` writes the registry the way the case studies do."""
    run_log = tmp_path / "payload/case_studies/etfs/run_log"
    prediction = run_log / "predictions/abc/predictions.parquet"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"stored predictions")

    connection = sqlite3.connect(run_log / "registry.db")
    if wal:
        connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE release_probe (value TEXT NOT NULL)")
    connection.execute("INSERT INTO release_probe VALUES ('accepted')")
    connection.commit()
    if not ship_wal:
        # Closing checkpoints the log and removes the sidecars, which is what a bundle must
        # ship: the installed tree is unwritable, so nothing can replay a log afterwards.
        connection.close()

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
    if ship_wal:
        connection.close()
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


def test_a_wal_mode_bundle_is_readable_once_it_is_installed(tmp_path: Path) -> None:
    """The installed tree is unwritable, where a WAL read needs the -shm it cannot create."""
    archive = _build_bundle(tmp_path, wal=True)

    installed = download_artifacts.install_artifact_archive(
        archive,
        "etfs",
        expected_sha256=_sha256(archive),
        repo_root=tmp_path / "repo",
        force=True,
    )

    registry = installed / "registry.db"
    assert not registry.stat().st_mode & stat.S_IWUSR
    with closing(sqlite3.connect(registry_readonly_uri(registry), uri=True)) as connection:
        assert connection.execute("SELECT value FROM release_probe").fetchone() == ("accepted",)


def test_a_bundle_shipping_an_uncheckpointed_log_is_refused(tmp_path: Path) -> None:
    """Installed, it would read as a database with no tables - or not open at all."""
    archive = _build_bundle(tmp_path, wal=True, ship_wal=True)
    existing = tmp_path / "repo/case_studies/etfs/run_log"
    existing.mkdir(parents=True)
    (existing / "sentinel").write_text("keep")

    with pytest.raises(ValueError, match="uncheckpointed"):
        download_artifacts.install_artifact_archive(
            archive,
            "etfs",
            expected_sha256=_sha256(archive),
            repo_root=tmp_path / "repo",
            force=True,
        )

    assert (existing / "sentinel").read_text() == "keep"


def _split(path: Path, size: int) -> list[Path]:
    """Split a file the way the release upload does, into .part00, .part01, ..."""
    data = path.read_bytes()
    parts = []
    for index in range(0, len(data), size):
        part = path.with_name(f"{path.name}.part{index // size:02d}")
        part.write_bytes(data[index : index + size])
        parts.append(part)
    return parts


def test_a_split_bundle_reassembles_into_the_byte_identical_tarball(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle over GitHub's 2 GB asset limit ships in parts, and must survive the trip."""
    archive = _build_bundle(tmp_path)
    expected = archive.read_bytes()
    parts = _split(archive, max(1, archive.stat().st_size // 3))
    assert len(parts) > 2, "the fixture must actually split to test reassembly"

    def fake_download(url: str, dest: Path, desc: str) -> bool:
        source = archive.parent / Path(url).name
        if not source.is_file():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        return True

    monkeypatch.setattr(download_artifacts, "download_file", fake_download)
    monkeypatch.setattr(download_artifacts, "BASE_URL", str(archive.parent))

    combined = tmp_path / "out" / "etfs.tar.gz"
    assert download_artifacts._fetch_parts("etfs", len(parts), combined)
    assert combined.read_bytes() == expected
    assert _sha256(combined) == _sha256(archive)
    # The parts are scratch and must not be left behind next to the tarball.
    assert list(combined.parent.iterdir()) == [combined]


def test_a_missing_part_fails_the_download_and_leaves_no_partial_tarball(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _build_bundle(tmp_path)
    parts = _split(archive, max(1, archive.stat().st_size // 3))
    parts[-1].unlink()

    def fake_download(url: str, dest: Path, desc: str) -> bool:
        source = archive.parent / Path(url).name
        if not source.is_file():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        return True

    monkeypatch.setattr(download_artifacts, "download_file", fake_download)
    monkeypatch.setattr(download_artifacts, "BASE_URL", str(archive.parent))

    combined = tmp_path / "out" / "etfs.tar.gz"
    assert not download_artifacts._fetch_parts("etfs", len(parts), combined)
    assert not combined.exists()
    assert not list(combined.parent.iterdir())


def test_a_held_back_case_study_is_fetched_from_the_release_that_still_has_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not every bundle is rebuilt in every release, and the URL has to follow the bundle."""
    seen: list[str] = []

    def record(url: str, dest: Path, desc: str) -> bool:
        seen.append(url)
        return False

    held = "v3.0.0-artifacts"
    assert held != download_artifacts.RELEASE_TAG
    monkeypatch.setitem(download_artifacts.ARTIFACT_RELEASE, "crypto_perps_funding", held)
    monkeypatch.setattr(download_artifacts, "download_file", record)
    monkeypatch.setattr(download_artifacts, "REPO_ROOT", tmp_path)

    assert not download_artifacts.download_case_study("etfs")
    assert not download_artifacts.download_case_study("crypto_perps_funding")

    assert seen[0].endswith(f"/{download_artifacts.RELEASE_TAG}/etfs.tar.gz")
    assert seen[1].endswith(f"/{held}/crypto_perps_funding.tar.gz")


def test_every_published_checksum_names_a_release_that_serves_it(tmp_path: Path) -> None:
    """A part count or a checksum for a case study nobody can reach is a dead entry."""
    for cs_id in download_artifacts.ARTIFACT_PARTS:
        assert cs_id in download_artifacts.ARTIFACT_SHA256
    for cs_id in download_artifacts.ARTIFACT_RELEASE:
        assert cs_id in download_artifacts.ARTIFACT_SHA256
    for cs_id in download_artifacts.ARTIFACT_SHA256:
        assert cs_id in download_artifacts.CASE_STUDIES
    # v3.1 serves every case study the script offers. Iterating ARTIFACT_RELEASE above proves
    # nothing while it is empty, so state the condition that makes it empty: a case study the
    # script lists with no checksum is one --cs would accept and then refuse to download.
    assert set(download_artifacts.ARTIFACT_SHA256) == set(download_artifacts.CASE_STUDIES)


def test_a_refused_fetch_falls_back_to_gh_against_the_release_that_holds_the_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An anonymous fetch of a private asset is refused, and the retry must not switch release."""
    cs_id = "crypto_perps_funding"
    held = "v3.0.0-artifacts"
    assert held != download_artifacts.RELEASE_TAG
    monkeypatch.setitem(download_artifacts.ARTIFACT_RELEASE, cs_id, held)
    dest = tmp_path / f"{cs_id}.tar.gz"
    calls: list[list[str]] = []

    def refuse(request, *args, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    def fake_run(argv, **kwargs):
        calls.append(argv)
        dest.write_bytes(b"payload")
        return None

    monkeypatch.setattr(download_artifacts, "_get_github_token", lambda: None)
    monkeypatch.setattr(download_artifacts.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(download_artifacts.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(download_artifacts.subprocess, "run", fake_run)

    url = f"{download_artifacts._base_url(cs_id)}/{cs_id}.tar.gz"
    assert download_artifacts.download_file(url, dest, cs_id)

    argv = calls[0]
    assert argv[:3] == ["gh", "release", "download"]
    assert argv[3] == held


def test_the_failure_hint_names_the_releases_that_were_read() -> None:
    held = "v3.0.0-artifacts"
    assert held != download_artifacts.RELEASE_TAG
    one = download_artifacts.release_hint(download_artifacts.RELEASE_TAG)
    both = download_artifacts.release_hint(download_artifacts.RELEASE_TAG, held)
    assert download_artifacts.RELEASE_TAG in both and held in both
    assert held not in one
    assert "artifacts release (" in one and "artifacts releases (" in both
