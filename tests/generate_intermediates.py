#!/usr/bin/env python3
"""Generate pipeline intermediates for the test-data repo.

Runs all 9 case study pipelines through specified stages
via Papermill with test overrides, capturing outputs to the specified directory.

The outputs are committed to ml4t/third-edition-test-data so that downstream
chapters (Ch11+) can read pre-computed labels/features/predictions without
re-running the full pipeline.

Usage:
    cd ~/ml4t/third_edition/code
    ML4T_DATA_PATH=~/ml4t/test-data/data \
    uv run python tests/generate_intermediates.py \
        --output ~/ml4t/test-data/intermediates

    # Run only through features (stages 01-03)
    uv run python tests/generate_intermediates.py \
        --output ~/ml4t/test-data/intermediates \
        --through-stage 3

    # Include DL stages (slow)
    uv run python tests/generate_intermediates.py \
        --output ~/ml4t/test-data/intermediates \
        --through-stage 12 --no-skip-dl
"""

import argparse
import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

try:
    from tests.pm_helpers import get_overrides, run_notebook
except ModuleNotFoundError:
    from pm_helpers import get_overrides, run_notebook

REPO_ROOT = Path(__file__).parent.parent

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

# Stage patterns to skip when --skip-dl is active (DL/latent/causal are heavy)
DL_STAGE_PATTERNS = re.compile(
    r"\d{2}_("
    r"dl_|deep_learning|tabular_dl|latent_factors|pca|ipca|cae|sdf|sae|"
    r"autoencoder|term_structure_pca|causal_dml"
    r")"
)


# ---------------------------------------------------------------------------
# Config seeding — replicate conftest.py seeded_output_dir logic
# ---------------------------------------------------------------------------

# Per-model-type overrides applied to copied preset YAMLs.
# Goal: minimal workload that still exercises the training loop + registry.
_TEST_PRESET_PATCHES: dict[str, dict] = {
    "lgb": {"max_iterations": 2, "checkpoint_interval": 1},
    "lstm": {"n_epochs": 2, "checkpoint_interval": 1},
    "tsmixer": {"n_epochs": 2, "checkpoint_interval": 1},
    "tcn": {"n_epochs": 2, "checkpoint_interval": 1},
    "nlinear": {"n_epochs": 2, "checkpoint_interval": 1},
    "patchtst": {"n_epochs": 2, "checkpoint_interval": 1},
    "tabm": {"n_epochs": 2, "checkpoint_interval": 1},
    "cae": {"n_epochs": 2, "checkpoint_interval": 1},
    "sdf": {"n_epochs": 2, "checkpoint_interval": 1},
    "sae": {"n_epochs": 2, "checkpoint_interval": 1},
    "ipca": {"n_epochs": 2, "checkpoint_interval": 1},
}

_MAX_CONFIGS_PER_FAMILY = 2
_TRIM_FAMILIES = {"linear", "gbm"}


def _patch_presets_for_testing(config_dir: Path) -> None:
    """Patch copied preset YAMLs with reduced-workload values for testing."""
    for model_type, overrides in _TEST_PRESET_PATCHES.items():
        model_dir = config_dir / model_type
        if not model_dir.exists():
            continue
        for preset_path in model_dir.glob("*.yaml"):
            preset = yaml.safe_load(preset_path.read_text())
            if preset is None:
                continue
            preset.update(overrides)
            with open(preset_path, "w") as f:
                yaml.dump(preset, f, default_flow_style=False)


def _trim_label_configs(cs_config_dir: Path) -> None:
    """Trim label config YAMLs to at most _MAX_CONFIGS_PER_FAMILY for sweep families."""
    for label_yaml in cs_config_dir.glob("fwd_*.yaml"):
        data = yaml.safe_load(label_yaml.read_text())
        if data is None or not isinstance(data, dict):
            continue
        trimmed = False
        for family, configs in data.items():
            if (
                family in _TRIM_FAMILIES
                and isinstance(configs, list)
                and len(configs) > _MAX_CONFIGS_PER_FAMILY
            ):
                data[family] = configs[:_MAX_CONFIGS_PER_FAMILY]
                trimmed = True
        if trimmed:
            with open(label_yaml, "w") as f:
                yaml.dump(data, f, default_flow_style=False)


def seed_configs(output_dir: Path) -> None:
    """Copy case study configs and global model presets into output_dir.

    Replicates the logic of conftest.py's seeded_output_dir fixture so that
    notebooks executed via generate_intermediates.py find patched configs.
    """
    cs_root = REPO_ROOT / "case_studies"

    # Copy per-case-study config files (setup.yaml, training menus, backtest presets, etc.)
    for cs_id in CASE_STUDIES:
        src_config_dir = cs_root / cs_id / "config"
        if not src_config_dir.exists():
            continue
        dst_config_dir = output_dir / cs_id / "config"
        if dst_config_dir.exists():
            shutil.rmtree(dst_config_dir)
        shutil.copytree(src_config_dir, dst_config_dir)
        _trim_label_configs(dst_config_dir)

    # Copy global model presets so load_configs() can find them.
    # load_configs() resolves presets at {case_dir.parent}/config/{model_type}/*.yaml
    global_config_src = cs_root / "config"
    global_config_dst = output_dir / "config"
    if global_config_src.exists() and not global_config_dst.exists():
        shutil.copytree(global_config_src, global_config_dst)
        _patch_presets_for_testing(global_config_dst)

    print(f"Seeded configs into {output_dir}")


def discover_stages(cs_dir: Path, through_stage: int, skip_dl: bool) -> list[Path]:
    """Auto-discover pipeline stages in a case study directory.

    Returns sorted list of .py notebook paths up through the specified stage number.
    Skips DL/latent/causal stages when skip_dl is True.
    """
    stages = []
    for notebook in sorted(cs_dir.glob("[0-9][0-9]_*.py")):
        if notebook.name.startswith("_"):
            continue

        stage_num = int(notebook.stem[:2])
        if stage_num > through_stage:
            continue

        if skip_dl and DL_STAGE_PATTERNS.match(notebook.stem):
            continue

        stages.append(notebook)

    return stages


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline intermediates for CI")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for intermediates",
    )
    parser.add_argument(
        "--case-studies",
        nargs="+",
        default=CASE_STUDIES,
        help="Case studies to run (default: all)",
    )
    parser.add_argument(
        "--through-stage",
        type=int,
        default=8,
        help="Run stages up to this number (default: 8 = through GBM for all case studies including sp500_options/08_gbm)",
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        default=True,
        help="Skip DL/latent/causal stages (default: True)",
    )
    parser.add_argument(
        "--no-skip-dl",
        action="store_false",
        dest="skip_dl",
        help="Include DL/latent/causal stages",
    )
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed configs (setup.yaml, label configs, model presets) into output dir
    # so notebooks find patched configs when ML4T_OUTPUT_DIR is set.
    seed_configs(output_dir)

    # Set ML4T_OUTPUT_DIR so all pipeline writes go to our output directory
    os.environ["ML4T_OUTPUT_DIR"] = str(output_dir)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PLOTLY_RENDERER"] = "json"

    results = {}
    total_start = time.time()

    for cs in args.case_studies:
        cs_dir = REPO_ROOT / "case_studies" / cs
        if not cs_dir.exists():
            print(f"\nSKIP {cs}: directory not found")
            continue

        stages = discover_stages(cs_dir, args.through_stage, args.skip_dl)
        if not stages:
            print(f"\nSKIP {cs}: no stages found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Case study: {cs} ({len(stages)} stages)")
        print(f"{'=' * 60}")

        cs_failed = False
        for notebook in stages:
            stage = notebook.stem

            if cs_failed:
                print(f"  {stage}: SKIP (earlier stage failed)")
                results[f"{cs}::{stage}"] = "skipped"
                continue

            rel_path = notebook.relative_to(REPO_ROOT).with_suffix("")
            overrides = get_overrides(str(rel_path))

            # Skip if overrides say so
            if overrides.get("skip"):
                reason = overrides.get("skip_reason", "marked skip")
                print(f"  {stage}: SKIP ({reason})")
                results[f"{cs}::{stage}"] = "skipped"
                # Pipeline stages (01-05) cascade their skip
                stage_num = int(stage[:2])
                if stage_num <= 5:
                    cs_failed = True
                continue

            timeout = overrides.get("timeout", 300)
            parameters = overrides.get("parameters", {})

            print(f"  {stage}: running...", end="", flush=True)
            start = time.time()

            result = run_notebook(
                py_path=notebook,
                parameters=parameters,
                timeout=timeout,
                output_dir=output_dir,
            )

            elapsed = time.time() - start

            if result["status"] == "ok":
                print(f" OK ({elapsed:.0f}s)")
                results[f"{cs}::{stage}"] = "ok"
            else:
                print(f" FAILED ({elapsed:.0f}s)")
                print(f"    Error: {result['error']}")
                results[f"{cs}::{stage}"] = "failed"
                cs_failed = True

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary ({total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")
    ok = sum(1 for v in results.values() if v == "ok")
    failed = sum(1 for v in results.values() if v == "failed")
    skipped = sum(1 for v in results.values() if v == "skipped")
    print(f"  OK: {ok}  Failed: {failed}  Skipped: {skipped}")

    if failed:
        print("\nFailed stages:")
        for k, v in results.items():
            if v == "failed":
                print(f"  - {k}")

    # Show output size
    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"\nOutput: {output_dir} ({total_bytes / 1e6:.1f} MB)")

    # Write metadata for staleness tracking
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "through_stage": args.through_stage,
        "skip_dl": args.skip_dl,
        "results": results,
        "total_seconds": round(total_elapsed),
        "size_mb": round(total_bytes / 1e6, 1),
    }
    metadata_path = output_dir / "_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
