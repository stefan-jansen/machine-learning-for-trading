#!/usr/bin/env python3
"""Download pre-computed model artifacts from GitHub releases.

These artifacts contain registry databases, model predictions, and backtest
results that allow readers to run strategy notebooks (Ch16-20) and insight
notebooks (Ch11-15) without first training all models.

The artifacts are published as a GitHub release asset and are added as the
case-study chapters roll out; until that release lands, the notebooks still
run end to end from scratch (the artifacts only skip retraining).

Usage:
    python scripts/download_artifacts.py                    # all case studies
    python scripts/download_artifacts.py --cs etfs          # single case study
    python scripts/download_artifacts.py --cs etfs --force  # re-download
    python scripts/download_artifacts.py --list             # show available
"""

import argparse
import hashlib
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent

# GitHub release configuration
GITHUB_REPO = "stefan-jansen/machine-learning-for-trading"
RELEASE_TAG = "v3.1.0-artifacts"
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

# Case studies served from a release other than RELEASE_TAG. v3.1 rebuilt all nine, so this
# is empty; it earns its place because a later release need not rebuild every bundle, and the
# download URL, the gh fallback and the failure hint all have to follow the bundle rather than
# the newest tag. The tests drive it with an injected entry.
ARTIFACT_RELEASE: dict[str, str] = {}

CASE_STUDIES = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]

# Bundles too large for a single GitHub release asset, which is capped at 2 GB.
# They are uploaded as `<cs>.tar.gz.part00`, `.part01`, ... and concatenated back
# into the tarball before anything is verified, so the whole-file checksum in
# ARTIFACT_SHA256 still decides whether the download is good.
ARTIFACT_PARTS: dict[str, int] = {
    "nasdaq100_microstructure": 2,
    "us_equities_panel": 2,
}

ARTIFACT_SHA256 = {
    "cme_futures": "d0d0d762ba10272a2cab45a04d96573cdd0e2d82f03108743e361f75be81517a",
    "crypto_perps_funding": "9b5d5eedfce4dd2a2385c713f47cc5e1fa2e8c89b8145bd83ae956269ae08058",
    "etfs": "46693e83409c15f533fc3b0986eb9cf006426ff06f6d7dea0732ed6a65104ceb",
    "fx_pairs": "132503d469fea1efb1d717f21227694ce4257557d6562d4a8b73a653f870e1c2",
    "nasdaq100_microstructure": "c29f964b6b03b81185ca038f09f679a1138370c2c4ae1b680e83beafdc95fb00",
    "sp500_equity_option_analytics": (
        "af621fa9c5e6f76dd1d74b6290c8d0a949bfb1bfcea98efcca6c158bcac44c40"
    ),
    "sp500_options": "816b3810ae65422e49ca9b11eb840d31efde2d274991bcf41fc4454036b73464",
    "us_equities_panel": "39e83df556edd920a3b9b692e443b27f250eff7ff20b6ca34fda3ee3db4e696f",
    "us_firm_characteristics": "9eb5602da0828c71dfeb4a3bda6ac92a4664caebb01a21158916aef74bd99d1e",
}


def release_hint(*tags: str) -> str:
    """The failure hint has to name the releases the caller was actually reading."""
    unique = sorted(set(tags) or {RELEASE_TAG})
    names = ", ".join(unique)
    noun = "release" if len(unique) == 1 else "releases"
    return (
        f"The pre-computed artifacts {noun} ({names}) may not be published yet.\n"
        "The artifacts are added as the case-study chapters roll out; until then every\n"
        "notebook still runs end to end from scratch - the artifacts only skip retraining.\n"
        f"Check the latest releases at https://github.com/{GITHUB_REPO}/releases"
    )


def _get_github_token() -> str | None:
    """Get GitHub token from env or gh CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _tag_from_url(url: str) -> str:
    """The release a download URL points at. Not every asset comes from the latest one."""
    parts = PurePosixPath(urllib.parse.urlsplit(url).path).parts
    if "download" in parts:
        index = parts.index("download")
        if index + 1 < len(parts):
            return parts[index + 1]
    return RELEASE_TAG


def _download_with_gh(url: str, dest: Path, desc: str) -> bool:
    """Download using gh CLI (handles auth automatically)."""
    asset_name = PurePosixPath(urllib.parse.urlsplit(url).path).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                _tag_from_url(url),
                "--repo",
                GITHUB_REPO,
                "--pattern",
                asset_name,
                "--dir",
                str(dest.parent),
                "--clobber",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        downloaded = dest.parent / asset_name
        if downloaded != dest:
            downloaded.rename(dest)
        print(f"  {desc}: done ({dest.stat().st_size / 1024 / 1024:.0f} MB)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  {desc}: gh download failed ({e})")
        return False


def download_file(url: str, dest: Path, desc: str) -> bool:
    """Download a file with progress reporting. Uses token auth if available."""
    token = _get_github_token()
    try:
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/octet-stream")
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        print(
                            f"\r  {desc}: {mb:.0f}/{total_mb:.0f} MB ({pct}%)",
                            end="",
                            flush=True,
                        )
            print()
        return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404) and shutil.which("gh"):
            return _download_with_gh(url, dest, desc)
        if e.code in (401, 403, 404):
            print(f"\r  {desc}: FAILED ({e.code} {e.reason}) - likely missing authentication")
        else:
            print(f"\r  {desc}: FAILED ({e.code} {e.reason})")
        return False
    except Exception as e:
        print(f"\r  {desc}: FAILED ({e})")
        return False


def _release_tag(cs_id: str) -> str:
    """The release this case study's bundle comes from, which is not always the latest."""
    return ARTIFACT_RELEASE.get(cs_id, RELEASE_TAG)


def _base_url(cs_id: str) -> str:
    """Where this case study's assets live, which is not always the current release."""
    tag = ARTIFACT_RELEASE.get(cs_id)
    if tag is None:
        return BASE_URL
    return f"https://github.com/{GITHUB_REPO}/releases/download/{tag}"


def _fetch_parts(cs_id: str, count: int, dest: Path) -> bool:
    """Download a split bundle and concatenate it back into one tarball."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    try:
        for index in range(count):
            name = f"{cs_id}.tar.gz.part{index:02d}"
            part = dest.parent / name
            url = f"{_base_url(cs_id)}/{name}"
            if not download_file(url, part, f"{cs_id} part {index + 1}/{count}"):
                return False
            parts.append(part)
        with dest.open("wb") as combined:
            for part in parts:
                with part.open("rb") as chunk:
                    shutil.copyfileobj(chunk, combined, 1024 * 1024)
        return True
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_members(tar: tarfile.TarFile, cs_id: str) -> None:
    expected_prefix = ("case_studies", cs_id, "run_log")
    for member in tar.getmembers():
        parts = PurePosixPath(member.name).parts
        is_parent = len(parts) <= len(expected_prefix) and parts == expected_prefix[: len(parts)]
        is_payload = parts[:3] == expected_prefix
        if (not is_parent and not is_payload) or ".." in parts:
            raise ValueError(f"Unexpected archive member: {member.name}")


def _verify_run_log(run_log: Path) -> None:
    manifest = run_log / ".release/SHA256SUMS"
    if not manifest.is_file():
        raise ValueError("Bundle has no internal artifact checksum manifest")

    root = run_log.resolve()
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        artifact = (run_log / relative.strip().lstrip("*")).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError(f"Invalid artifact manifest path: {relative}")
        if _sha256(artifact) != expected:
            raise ValueError(f"artifact checksum mismatch: {relative}")

    registry = run_log / "registry.db"
    # A bundle that ships an uncheckpointed write-ahead log cannot be read correctly once it is
    # installed: the tree is left unwritable, so `mode=ro` cannot create the `-shm` sidecar it
    # needs to replay the log, and `immutable=1` reads the pre-WAL main file instead - which can
    # be missing the tables entirely. Neither the checksums nor the integrity check below would
    # notice, because both describe the files as shipped. Refuse the bundle instead.
    wal = registry.with_name(registry.name + "-wal")
    if wal.is_file() and wal.stat().st_size > 0:
        raise ValueError("Bundle ships an uncheckpointed registry write-ahead log")

    # Nothing is writing this tree - it was just extracted and checksum-verified - so the
    # integrity check reads the main file directly.
    uri = f"file:{registry.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if integrity != ("ok",) or foreign_keys:
        raise ValueError("Installed registry failed SQLite integrity checks")


def _set_tree_writable(root: Path, *, writable: bool) -> None:
    paths = [root, *root.rglob("*")]
    for path in paths:
        mode = path.stat().st_mode
        if writable:
            path.chmod(mode | stat.S_IWUSR)
        else:
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def install_artifact_archive(
    archive: Path,
    cs_id: str,
    *,
    expected_sha256: str,
    repo_root: Path = REPO_ROOT,
    force: bool = False,
) -> Path:
    """Verify and atomically install one case-study run log."""
    if _sha256(archive) != expected_sha256:
        raise ValueError(f"{cs_id} archive checksum mismatch")

    case_dir = repo_root / "case_studies" / cs_id
    target = case_dir / "run_log"
    case_dir.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f"Refusing to replace symlinked run log: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Run-log target is not a directory: {target}")
    if target.exists() and not force:
        return target

    stage_parent = Path(tempfile.mkdtemp(prefix=".run-log-install-", dir=case_dir))
    ready = case_dir / f".run-log-ready-{uuid.uuid4().hex}"
    previous = case_dir / f".run-log-previous-{uuid.uuid4().hex}"
    try:
        with tarfile.open(archive, "r:gz") as tar:
            _validate_archive_members(tar, cs_id)
            tar.extractall(path=stage_parent, filter="data")
        staged = stage_parent / "case_studies" / cs_id / "run_log"
        _verify_run_log(staged)
        staged.rename(ready)
        _set_tree_writable(ready, writable=False)

        if target.exists():
            target.rename(previous)
        try:
            ready.rename(target)
        except Exception:
            if previous.exists() and not target.exists():
                previous.rename(target)
            raise
        if previous.exists():
            _set_tree_writable(previous, writable=True)
            shutil.rmtree(previous, ignore_errors=True)
        return target
    finally:
        if ready.exists():
            _set_tree_writable(ready, writable=True)
            shutil.rmtree(ready)
        shutil.rmtree(stage_parent, ignore_errors=True)


def has_artifacts(cs_id: str) -> bool:
    """Check if a case study already has artifacts."""
    cs_dir = REPO_ROOT / "case_studies" / cs_id / "run_log"
    return (cs_dir / "registry.db").exists() and (cs_dir / ".release/SHA256SUMS").exists()


def download_case_study(cs_id: str, force: bool = False) -> bool:
    """Download and extract artifacts for one case study."""
    if has_artifacts(cs_id) and not force:
        print(f"  {cs_id}: already has artifacts (use --force to re-download)")
        return True

    expected_sha256 = ARTIFACT_SHA256.get(cs_id)
    if expected_sha256 is None:
        print(f"  {cs_id}: artifact bundle is not published in {_release_tag(cs_id)}")
        return False

    tarball_name = f"{cs_id}.tar.gz"
    tmp_path = REPO_ROOT / ".cache" / tarball_name

    parts = ARTIFACT_PARTS.get(cs_id)
    if parts:
        ok = _fetch_parts(cs_id, parts, tmp_path)
    else:
        ok = download_file(f"{_base_url(cs_id)}/{tarball_name}", tmp_path, cs_id)
    if not ok:
        tmp_path.unlink(missing_ok=True)
        return False

    print("  Verifying and installing...", end=" ", flush=True)
    try:
        install_artifact_archive(
            tmp_path,
            cs_id,
            expected_sha256=expected_sha256,
            force=force,
        )
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"FAILED ({error})")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)
    print("done")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-computed model artifacts from GitHub releases"
    )
    parser.add_argument("--cs", "--case-study", help="Single case study ID")
    parser.add_argument("--force", action="store_true", help="Re-download even if artifacts exist")
    parser.add_argument("--list", action="store_true", help="List available case studies")
    args = parser.parse_args()

    if args.list:
        print("Available case studies:")
        for cs in CASE_STUDIES:
            if has_artifacts(cs):
                status = "installed"
            elif cs in ARTIFACT_SHA256:
                status = "available"
            else:
                status = "pending"
            print(f"  {cs:40s} [{status}]")
        return

    cs_list = [args.cs] if args.cs else [cs for cs in CASE_STUDIES if cs in ARTIFACT_SHA256]

    # Validate
    for cs in cs_list:
        if cs not in CASE_STUDIES:
            print(f"Unknown case study: {cs}")
            print(f"Available: {', '.join(CASE_STUDIES)}")
            sys.exit(1)

    print(f"Downloading artifacts for {len(cs_list)} case study(ies)")
    for tag in sorted({_release_tag(cs) for cs in cs_list}):
        print(f"Source: https://github.com/{GITHUB_REPO}/releases/download/{tag}")
    print()

    # Ensure cache dir
    (REPO_ROOT / ".cache").mkdir(exist_ok=True)

    success = 0
    failed: list[str] = []
    for cs_id in cs_list:
        if download_case_study(cs_id, force=args.force):
            success += 1
        else:
            failed.append(cs_id)

    # Clean up cache dir
    cache = REPO_ROOT / ".cache"
    if cache.exists() and not any(cache.iterdir()):
        cache.rmdir()

    print(f"\nDone: {success}/{len(cs_list)} case studies ready.")
    if success < len(cs_list):
        print()
        print(release_hint(*(_release_tag(cs) for cs in failed)))
        sys.exit(1)


if __name__ == "__main__":
    main()
