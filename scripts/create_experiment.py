#!/usr/bin/env python3
"""Create a writable, isolated case-study experiment from installed artifacts."""

from __future__ import annotations

import argparse
import shutil
import stat
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIRS = ("run_log", "labels", "features", "evaluation", "benchmark")


def _make_writable(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def create_experiment(
    case_study: str,
    output_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Copy available generated state into a new ML4T_OUTPUT_DIR."""
    source = repo_root / "case_studies" / case_study
    source_run_log = source / "run_log"
    if not source.is_dir() or not source_run_log.is_dir() or source_run_log.is_symlink():
        raise ValueError(f"Install the {case_study} artifact bundle before creating an experiment")

    output_root = output_root.resolve()
    target = output_root / case_study
    if target.exists():
        raise FileExistsError(f"Experiment already exists: {target}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{case_study}-staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        for name in GENERATED_DIRS:
            candidate = source / name
            if candidate.is_dir():
                shutil.copytree(candidate, staging / name)

        _make_writable(staging)
        release_metadata = staging / "run_log/.release"
        if release_metadata.exists():
            release_metadata.rename(staging / "run_log/.baseline")
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a writable case-study experiment without changing release artifacts"
    )
    parser.add_argument("--cs", "--case-study", required=True, help="Case-study identifier")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New ML4T_OUTPUT_DIR root for the experiment",
    )
    args = parser.parse_args()

    experiment = create_experiment(args.cs, args.output)
    print(f"Experiment ready: {experiment}")
    print(f"Set ML4T_OUTPUT_DIR={args.output.resolve()} when running case-study notebooks.")


if __name__ == "__main__":
    main()
