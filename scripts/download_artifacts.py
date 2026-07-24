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
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent

# GitHub release configuration
GITHUB_REPO = "stefan-jansen/machine-learning-for-trading"
RELEASE_TAG = "v3.0.0-artifacts"
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

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

ARTIFACT_SHA256 = {
    "cme_futures": "ab1c97276cdf74aa95d894cc0b0f3ea909c3ed8356f41f217e008c971e34b4f8",
    "crypto_perps_funding": "517030f3def6d0264984c2b545032741100df4ee3d159e44ef4c348a314c4c7f",
    "etfs": "93d3e24dcb4872b965f355a6931f163e0871a58729c1bd1ea6d99f369f2baf39",
    "fx_pairs": "0555a5b2788576ba9eeb8fd874f5375a651d5094b2952eb42aa01aac4bee38e6",
    "nasdaq100_microstructure": "a688081f30f97b2ab9f7f926e0608734ed4145ecfd1bf189624709d694c2167f",
    "us_equities_panel": "ebf5b0b846da310e2611126b0059e5c51f116e185f82d47248f81cf088f32e27",
    "sp500_equity_option_analytics": (
        "4eba2aadcc1fc5af322f0cfb0a8d4dcfb036391141adff59a0358c2a8401ca49"
    ),
    "sp500_options": "55333f313d3a2180a468e326030aa80d713b271c85d41d55989532f9567fccb2",
    "us_firm_characteristics": ("2ec2087a054e1a075f7baa51546a2453c0d7db6108211e9cfbb80f95d894cffb"),
}


RELEASE_HINT = (
    f"The pre-computed artifacts release ({RELEASE_TAG}) may not be published yet.\n"
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


def _download_with_gh(asset_name: str, dest: Path, desc: str) -> bool:
    """Download using gh CLI (handles auth automatically)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                RELEASE_TAG,
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
            return _download_with_gh(Path(url).name, dest, desc)
        if e.code in (401, 403, 404):
            print(f"\r  {desc}: FAILED ({e.code} {e.reason}) - likely missing authentication")
        else:
            print(f"\r  {desc}: FAILED ({e.code} {e.reason})")
        return False
    except Exception as e:
        print(f"\r  {desc}: FAILED ({e})")
        return False


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
        print(f"  {cs_id}: artifact bundle is not published in {RELEASE_TAG}")
        return False

    tarball_name = f"{cs_id}.tar.gz"
    url = f"{BASE_URL}/{tarball_name}"
    tmp_path = REPO_ROOT / ".cache" / tarball_name

    if not download_file(url, tmp_path, cs_id):
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
    print(f"Source: {BASE_URL}\n")

    # Ensure cache dir
    (REPO_ROOT / ".cache").mkdir(exist_ok=True)

    success = 0
    for cs_id in cs_list:
        if download_case_study(cs_id, force=args.force):
            success += 1

    # Clean up cache dir
    cache = REPO_ROOT / ".cache"
    if cache.exists() and not any(cache.iterdir()):
        cache.rmdir()

    print(f"\nDone: {success}/{len(cs_list)} case studies ready.")
    if success < len(cs_list):
        print()
        print(RELEASE_HINT)
        sys.exit(1)


if __name__ == "__main__":
    main()
