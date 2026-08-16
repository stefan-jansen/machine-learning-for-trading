#!/usr/bin/env python3
"""Create a writable, isolated case-study experiment from installed artifacts."""

from __future__ import annotations

import argparse
import json
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
    manifest: dict | None = None,
    include_release_run_log: bool = True,
) -> Path:
    """Copy available generated state into a new ML4T_OUTPUT_DIR."""
    source = repo_root / "case_studies" / case_study
    source_run_log = source / "run_log"
    if not source.is_dir() or (
        include_release_run_log and (not source_run_log.is_dir() or source_run_log.is_symlink())
    ):
        raise ValueError(f"Install the {case_study} artifact bundle before creating an experiment")
    source_config = source / "config"
    if not source_config.is_dir():
        raise ValueError(f"Case study {case_study} has no config/ tree; cannot create experiment")

    output_root = output_root.resolve()
    target = output_root / case_study
    if target.exists():
        raise FileExistsError(f"Experiment already exists: {target}")

    # Shared model presets (case_studies/config/) resolve at {ML4T_OUTPUT_DIR}/config/
    # (load_configs reads case_dir.parent / "config"). Seed them once per output root;
    # leave an existing copy in place so a second case-study experiment reuses it.
    presets_src = repo_root / "case_studies" / "config"
    presets_dst = output_root / "config"
    seed_presets = presets_src.is_dir() and not presets_dst.exists()

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{case_study}-staging-{uuid.uuid4().hex}"
    presets_staging = output_root / f".config-staging-{uuid.uuid4().hex}" if seed_presets else None
    try:
        staging.mkdir()
        for name in GENERATED_DIRS:
            if name == "run_log" and not include_release_run_log:
                (staging / name).mkdir()
                continue
            candidate = source / name
            if candidate.is_dir():
                shutil.copytree(candidate, staging / name)

        # config/ is a version-controlled input, not a generated artifact, but the
        # reader edits it to define the experiment and get_case_study_dir() redirects
        # config reads to ML4T_OUTPUT_DIR, so the experiment must carry its own copy
        # (else notebooks crash on config/setup.yaml before any reader edit is reached).
        shutil.copytree(source_config, staging / "config")

        if manifest is not None:
            (staging / ".study.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )

        _make_writable(staging)
        release_metadata = staging / "run_log/.release"
        if release_metadata.exists():
            release_metadata.rename(staging / "run_log/.baseline")

        if presets_staging is not None:
            shutil.copytree(presets_src, presets_staging)
            _make_writable(presets_staging)

        # All expensive copies are done; finalize with atomic renames only. Order:
        # presets first so a completed case-study dir never implies missing presets.
        if presets_staging is not None:
            presets_staging.rename(presets_dst)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if presets_staging is not None:
            shutil.rmtree(presets_staging, ignore_errors=True)
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
    output_root = args.output.resolve()
    print(f"Experiment ready: {experiment}")
    print(f"Edit the config in {experiment / 'config'} to change the experiment,")
    print(f"then set ML4T_OUTPUT_DIR={output_root} when running case-study notebooks.")


if __name__ == "__main__":
    main()
