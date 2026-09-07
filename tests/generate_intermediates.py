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

Exit status is 0 only when every stage of every requested case study was
generated. A stage that failed, and a pipeline stage (01-05) that
``tests/overrides.yaml`` marks ``skip`` so that nothing downstream of it is
regenerated, both exit 1: the fixture then holds whatever an earlier run wrote,
which is a stale fixture that looks freshly built. ``--ignore-skips`` runs those
stages anyway - the skips exist to keep the timed CI job inside its budget, and
generation has no budget to protect.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

try:
    from tests.pm_helpers import get_overrides, run_notebook
    from tests.preset_patches import _patch_presets_for_testing, _trim_label_configs
except ModuleNotFoundError:
    from pm_helpers import get_overrides, run_notebook
    from preset_patches import _patch_presets_for_testing, _trim_label_configs

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

# _patch_presets_for_testing (and the _TEST_PRESET_PATCHES it reads) is
# imported from tests/preset_patches.py rather than duplicated here: two
# copies of the same workload-reduction table drift the moment one gets a
# fix the other doesn't (e.g. IPCA's factor_ridge/gamma_ridge
# regularization), silently regenerating fixtures against the stale values.


def seed_configs(output_dir: Path, case_studies: Iterable[str] = CASE_STUDIES) -> None:
    """Copy case study configs and global model presets into output_dir.

    Replicates the logic of conftest.py's seeded_output_dir fixture so that
    notebooks executed via generate_intermediates.py find patched configs.

    Only ``case_studies`` are seeded. Seeding all nine regardless of what the run
    was scoped to rewrote 51 tracked files under eight other case studies from a
    single ``--case-studies cme_futures`` run, so an agent could not regenerate
    its own fixture without touching committed state it does not own.
    """
    cs_root = REPO_ROOT / "case_studies"

    # Copy per-case-study config files (setup.yaml, training menus, backtest presets, etc.)
    for cs_id in case_studies:
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
    #
    # Refreshed on every run, like the per-case-study configs above. Guarding this on
    # `not dst.exists()` meant the copy ran once, on a tree that had no config/ yet, and
    # never again - so a preset added or edited after the first generation never reached
    # the fixture. sp500_options/06_linear failed the 2026-09-06 regeneration on
    # `Preset not found: lasso_f0.5.yaml`, which has been in case_studies/config/lasso/
    # since the initial release: the fixture held only the five `lasso_a*` presets that
    # existed when its config/ was first written.
    #
    # Refreshing is not the ownership problem the per-case-study loop guards against.
    # These presets are shared by all nine case studies rather than owned by the one the
    # run was scoped to, so bringing them into line with source is what a scoped run
    # should do, not a rewrite of state it does not own.
    global_config_src = cs_root / "config"
    global_config_dst = output_dir / "config"
    if global_config_src.exists():
        if global_config_dst.exists():
            shutil.rmtree(global_config_dst)
        shutil.copytree(
            global_config_src,
            global_config_dst,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        _patch_presets_for_testing(global_config_dst)

    print(f"Seeded configs into {output_dir} for: {', '.join(case_studies)}")


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


# Outcomes a stage can end a generation run with. `incomplete` is the one this
# script used to have no name for: the stage did not run and nothing downstream
# of it could, so the fixture on disk is whatever a previous run left there.
# Counting that as a skip is how a default regeneration reported success while
# producing nothing for cme_futures stages 04-08.
OK = "ok"
FAILED = "failed"
SKIPPED = "skipped"
INCOMPLETE = "incomplete"
NOT_RUN = "not_run"

# A generation run has not produced the fixture it claims unless every stage
# either ran or was skipped for a reason that leaves nothing downstream unbuilt.
EARNS_NONZERO_EXIT = (FAILED, INCOMPLETE)


def resolve_case_studies(requested: Iterable[str]) -> list[str]:
    """Return the requested case studies, rejecting any name that is not one.

    A name that matches nothing used to be skipped without entering ``results``,
    so the failure count stayed zero and the run exited 0 - a typo produced a
    green run that generated nothing.
    """
    requested = list(requested)
    unknown = [name for name in requested if name not in CASE_STUDIES]
    if unknown:
        raise ValueError(
            f"unknown case study {', '.join(sorted(unknown))} - "
            f"choose from {', '.join(CASE_STUDIES)}"
        )
    return requested


def exit_code(results: dict[str, str]) -> int:
    """0 only when every stage of every requested case study is accounted for."""
    return 1 if any(v in EARNS_NONZERO_EXIT for v in results.values()) else 0


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
    parser.add_argument(
        "--ignore-skips",
        action="store_true",
        help=(
            "Run stages that overrides.yaml marks skip. Those skips exist to keep the "
            "timed CI job inside its budget; generation has no such budget and its whole "
            "purpose is to produce the artifact that job then consumes."
        ),
    )
    args = parser.parse_args()

    try:
        case_studies = resolve_case_studies(args.case_studies)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed configs (setup.yaml, label configs, model presets) into output dir
    # so notebooks find patched configs when ML4T_OUTPUT_DIR is set. Scoped to the
    # requested case studies: see seed_configs.
    seed_configs(output_dir, case_studies)

    # Set ML4T_OUTPUT_DIR so all pipeline writes go to our output directory
    os.environ["ML4T_OUTPUT_DIR"] = str(output_dir)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PLOTLY_RENDERER"] = "json"

    results = {}
    total_start = time.time()

    for cs in case_studies:
        cs_dir = REPO_ROOT / "case_studies" / cs
        if not cs_dir.exists():
            # resolve_case_studies has already accepted the name, so the directory
            # is missing rather than mistyped: nothing gets generated for it and
            # the run has not done what it was asked to.
            print(f"\nINCOMPLETE {cs}: directory not found")
            results[cs] = INCOMPLETE
            continue

        stages = discover_stages(cs_dir, args.through_stage, args.skip_dl)
        if not stages:
            print(f"\nINCOMPLETE {cs}: no stages found through stage {args.through_stage}")
            results[cs] = INCOMPLETE
            continue

        print(f"\n{'=' * 60}")
        print(f"Case study: {cs} ({len(stages)} stages)")
        print(f"{'=' * 60}")

        cs_failed = False
        for notebook in stages:
            stage = notebook.stem

            if cs_failed:
                print(f"  {stage}: NOT RUN (an earlier stage did not complete)")
                results[f"{cs}::{stage}"] = NOT_RUN
                continue

            rel_path = notebook.relative_to(REPO_ROOT).with_suffix("")
            overrides = get_overrides(str(rel_path))

            # Skip if overrides say so, unless the operator asked for the whole
            # pipeline. overrides.yaml's `skip` is read by the timed CI job and by
            # this generator, and the two want different answers from it: the job
            # is protecting a time budget, and generation is producing the artifact
            # that job consumes.
            if overrides.get("skip") and not args.ignore_skips:
                reason = overrides.get("skip_reason", "marked skip")
                stage_num = int(stage[:2])
                # A pipeline stage (01-05) that does not run leaves every later
                # stage of this case study unbuilt, so the fixture keeps whatever
                # the previous run wrote. That is an incomplete generation, not a
                # skipped one, and the exit code has to say so.
                if stage_num <= 5:
                    print(f"  {stage}: INCOMPLETE, skipped by overrides ({reason})")
                    print("    Nothing downstream of it is regenerated; the fixture keeps")
                    print("    whatever an earlier run left on disk. Re-run with --ignore-skips")
                    print("    to generate it anyway.")
                    results[f"{cs}::{stage}"] = INCOMPLETE
                    cs_failed = True
                else:
                    print(f"  {stage}: SKIP ({reason})")
                    results[f"{cs}::{stage}"] = SKIPPED
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
                research_preview=False,
            )

            elapsed = time.time() - start

            if result["status"] == "ok":
                print(f" OK ({elapsed:.0f}s)")
                results[f"{cs}::{stage}"] = OK
            else:
                print(f" FAILED ({elapsed:.0f}s)")
                print(f"    Error: {result['error']}")
                results[f"{cs}::{stage}"] = FAILED
                cs_failed = True

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary ({total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")
    counts = {
        state: sum(1 for v in results.values() if v == state)
        for state in (OK, FAILED, INCOMPLETE, SKIPPED, NOT_RUN)
    }
    print(
        f"  OK: {counts[OK]}  Failed: {counts[FAILED]}  Incomplete: {counts[INCOMPLETE]}  "
        f"Skipped: {counts[SKIPPED]}  Not run: {counts[NOT_RUN]}"
    )

    for state, heading in ((FAILED, "Failed stages"), (INCOMPLETE, "Incomplete units")):
        if counts[state]:
            print(f"\n{heading}:")
            for k, v in results.items():
                if v == state:
                    print(f"  - {k}")

    # Show output size, and what this run did to it. The fixture is git-stored and
    # unreduced production data reaches it silently: regenerating cme_futures stage 03
    # replaced an 86,110-row, 8-product, 19.5 MB features artifact with a 310,947-row,
    # 30-product, 75.9 MB one whose content digest equalled production's exactly, and
    # nothing in the run said so. Whether that growth is wanted is a decision; it can
    # only be made if the run reports it.
    metadata_path = output_dir / "_metadata.json"
    previous = {}
    if metadata_path.is_file():
        try:
            previous = json.loads(metadata_path.read_text()).get("size_mb_by_case_study", {})
        except (OSError, json.JSONDecodeError):
            previous = {}

    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    sizes = {
        cs: round(
            sum(f.stat().st_size for f in (output_dir / cs).rglob("*") if f.is_file()) / 1e6, 1
        )
        for cs in case_studies
        if (output_dir / cs).is_dir()
    }
    print(f"\nOutput: {output_dir} ({total_bytes / 1e6:.1f} MB)")
    for cs, size in sizes.items():
        before = previous.get(cs)
        if before is None:
            print(f"  {cs}: {size:.1f} MB")
        else:
            change = size - before
            factor = f", {size / before:.1f}x" if before else ""
            print(f"  {cs}: {before:.1f} -> {size:.1f} MB ({change:+.1f} MB{factor})")

    # Write metadata for staleness tracking
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "through_stage": args.through_stage,
        "skip_dl": args.skip_dl,
        "results": results,
        "total_seconds": round(total_elapsed),
        "size_mb": round(total_bytes / 1e6, 1),
        "size_mb_by_case_study": sizes,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {metadata_path}")

    # A failed or skipped stage leaves whatever the previous run wrote in place, so
    # exiting 0 reports success while the fixture set still holds the stale
    # artifact. That is how the sp500_options temporal artifact shipped without a
    # `fold` column: the stage timed out, the wrapper ran under `set -e` and saw
    # nothing.
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
