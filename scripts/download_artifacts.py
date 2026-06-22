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
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

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


RELEASE_HINT = (
    f"The pre-computed artifacts release ({RELEASE_TAG}) may not be published yet.\n"
    "The artifacts are added as the case-study chapters roll out; until then every\n"
    "notebook still runs end to end from scratch — the artifacts only skip retraining.\n"
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
            print(f"\r  {desc}: FAILED ({e.code} {e.reason}) — likely missing authentication")
        else:
            print(f"\r  {desc}: FAILED ({e.code} {e.reason})")
        return False
    except Exception as e:
        print(f"\r  {desc}: FAILED ({e})")
        return False


def extract_tarball(tarball: Path, extract_to: Path) -> int:
    """Extract tarball and return number of files extracted."""
    count = 0
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=extract_to, filter="data")
        count = len(tar.getmembers())
    return count


def has_artifacts(cs_id: str) -> bool:
    """Check if a case study already has artifacts."""
    cs_dir = REPO_ROOT / "case_studies" / cs_id / "run_log"
    return (cs_dir / "registry.db").exists()


def download_case_study(cs_id: str, force: bool = False) -> bool:
    """Download and extract artifacts for one case study."""
    if has_artifacts(cs_id) and not force:
        print(f"  {cs_id}: already has artifacts (use --force to re-download)")
        return True

    tarball_name = f"{cs_id}.tar.gz"
    url = f"{BASE_URL}/{tarball_name}"
    tmp_path = REPO_ROOT / ".cache" / tarball_name

    if not download_file(url, tmp_path, cs_id):
        return False

    print("  Extracting...", end=" ", flush=True)
    n = extract_tarball(tmp_path, REPO_ROOT)
    print(f"{n} files")

    # Clean up tarball
    tmp_path.unlink()
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
            status = "installed" if has_artifacts(cs) else "not installed"
            print(f"  {cs:40s} [{status}]")
        return

    cs_list = [args.cs] if args.cs else CASE_STUDIES

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
