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


def _seed_shared_presets(repo_root: Path, output_root: Path) -> None:
    """Copy the shared model presets into the experiment's ML4T_OUTPUT_DIR.

    ``load_configs`` resolves preset files at ``{ML4T_OUTPUT_DIR}/config/{model_type}/``
    (it reads ``case_dir.parent / "config"``), so the shared ``case_studies/config/``
    tree must be present there for a preset lookup to succeed under the experiment
    dir. Mirrors ``tests/conftest.py::seeded_output_dir``. Shared across every case
    study in one output root, so seed it once and leave an existing copy in place.
    """
    src = repo_root / "case_studies" / "config"
    dst = output_root / "config"
    if not src.is_dir() or dst.exists():
        return
    shutil.copytree(src, dst)
    _make_writable(dst)


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
    source_config = source / "config"
    if not source_config.is_dir():
        raise ValueError(f"Case study {case_study} has no config/ tree; cannot create experiment")

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

        # config/ is a version-controlled input, not a generated artifact, but the
        # reader edits it to define the experiment and get_case_study_dir() redirects
        # config reads to ML4T_OUTPUT_DIR, so the experiment must carry its own copy
        # (else notebooks crash on config/setup.yaml before any reader edit is reached).
        shutil.copytree(source_config, staging / "config")

        _make_writable(staging)
        release_metadata = staging / "run_log/.release"
        if release_metadata.exists():
            release_metadata.rename(staging / "run_log/.baseline")
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _seed_shared_presets(repo_root, output_root)
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
    output_root = args.output.resolve()
    print(f"Experiment ready: {experiment}")
    print(f"Edit the config in {experiment / 'config'} to change the experiment,")
    print(f"then set ML4T_OUTPUT_DIR={output_root} when running case-study notebooks.")


if __name__ == "__main__":
    main()
