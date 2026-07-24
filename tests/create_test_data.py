#!/usr/bin/env python3
"""Subsample production data into the CI fixture set in ml4t/third-edition-test-data.

This is step 1 of the two-step regeneration path; step 2 is
``tests/generate_intermediates.py``, which runs the case-study pipelines against
whatever this script produces. ``tests/generate_test_data.sh`` chains both.

Each dataset is declared once, in DATASETS, as a spec that names its source under
``--source``, its destination under ``--output``, and how it is reduced. The
subsample budgets are the ones recorded in ``data/manifest.json`` in the test-data
repo, which is rewritten from DATASETS on every run so the manifest cannot drift
from what was actually generated.

Usage:
    # Everything (from the repo root)
    uv run python tests/create_test_data.py \
        --source ~/ml4t/code/data --output ~/ml4t/test-data/data

    # One dataset, which is the common case when a schema gate goes red
    uv run python tests/create_test_data.py \
        --source ~/ml4t/code/data --output ~/ml4t/test-data/data \
        --dataset firm_characteristics

    # Show what would be written, touching nothing
    uv run python tests/create_test_data.py --source ... --output ... --dry-run

Never set PLOTLY_RENDERER=json here or in any wrapper: notebooks executed with the
JSON renderer emit no image/png, and the resulting figures do not render on GitHub.
This script writes no notebooks, but the shell wrapper used to, which is how that
debt was created.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Dataset:
    """One fixture-generation unit.

    Attributes:
        name: Selector for ``--dataset``.
        description: What the reduction keeps, mirrored into the manifest.
        build: Callable invoked as ``build(source, output)``; returns the list of
            files it wrote. Receives the roots, not per-file paths, because some
            datasets fan out to several outputs.
        subdir: Path under the output root that this dataset owns, relative to it.
            ``--clean`` removes exactly this directory, so it must not be shared
            with another dataset.
        budget: Manifest ``subsets`` entry describing the reduction (e.g.
            ``{"max_entities": 200}``). Recorded verbatim in the manifest.
    """

    name: str
    description: str
    build: Callable[[Path, Path], list[Path]]
    subdir: Path
    budget: dict[str, object] = field(default_factory=dict)


# --- firm characteristics -----------------------------------------------------
#
# The published Chen-Pelger-Zhu archive ships one dense (date, firm, variable)
# tensor per split. The firm axis is positional and persistent *within* a split,
# which is the only place anonymous firm identity survives -- the RetChar.csv in
# the same archive drops it. So the fixture is built from the tensors via the
# canonical converter in data/equities/firm_characteristics/download.py, not from
# the CSV and not from the full-size parquets (which predate the symbol schema:
# they carry `date` and no `symbol`, and the loader rejects them).

FIRM_CHAR_SPLITS = (
    # (split, tensor filename, symbol namespace offset) -- offsets match
    # convert_to_parquet: the archive publishes no cross-split firm mapping, so
    # the namespaces are kept disjoint rather than implying an identity join.
    ("train", "Char_train.npz", 0),
    ("valid", "Char_valid.npz", 1_000_000),
    ("test", "Char_test.npz", 2_000_000),
)
FIRM_CHAR_MAX_ENTITIES = 200


def _firm_char_tensor_dir(source: Path) -> Path:
    """Locate the characteristic tensors under a production data root.

    Two layouts exist in the wild: the one download.py's extractor produces (it
    strips a redundant ``datasets/`` wrapper) and the one a manual ``unzip``
    leaves behind (it does not). Both are read here because the manual path is
    what the archive's own instructions tell readers to do.
    """
    base = source / "equities" / "firm_characteristics" / "dl_asset_pricing"
    for candidate in (base / "char", base / "datasets" / "char"):
        if all((candidate / filename).exists() for _, filename, _ in FIRM_CHAR_SPLITS):
            return candidate
    raise FileNotFoundError(
        f"Characteristic tensors not found under {base} (looked in char/ and "
        "datasets/char/). Fetch them with "
        "data/equities/firm_characteristics/download.py."
    )


def build_firm_characteristics(source: Path, output: Path) -> list[Path]:
    """Write the four canonical firm-characteristics parquets, subsampled by firm.

    Keeps the ``FIRM_CHAR_MAX_ENTITIES`` most-observed firms per split, so each
    retained firm carries its full time series. Subsampling per date instead would
    shred the panel: a firm would appear and vanish across adjacent months, and
    every downstream fold, lag and cross-sectional rank would be computed over a
    different, unstable cross-section.
    """
    from data.equities.firm_characteristics.download import _characteristic_frame

    tensor_dir = _firm_char_tensor_dir(source)
    out_dir = output / "equities" / "firm_characteristics"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    split_frames: list[pl.DataFrame] = []
    for split, filename, offset in FIRM_CHAR_SPLITS:
        frame = _characteristic_frame(tensor_dir / filename, split, offset)
        keep = (
            frame.group_by("symbol")
            .len()
            # Ties broken by symbol so the fixture is reproducible bit for bit.
            .sort(["len", "symbol"], descending=[True, False])
            .head(FIRM_CHAR_MAX_ENTITIES)
            .get_column("symbol")
        )
        subsampled = frame.filter(pl.col("symbol").is_in(keep.implode()))
        path = out_dir / f"firm_characteristics_{split}.parquet"
        subsampled.write_parquet(path)
        split_frames.append(subsampled)
        written.append(path)
        print(
            f"    {split}: {len(subsampled):,} rows, "
            f"{subsampled['symbol'].n_unique()} firms, "
            f"{subsampled['timestamp'].n_unique()} dates "
            f"({path.stat().st_size / 1e6:.1f} MB)"
        )

    # load_firm_characteristics(split="all") reads this file rather than
    # concatenating the splits, so it has to be materialized.
    all_path = out_dir / "firm_characteristics_all.parquet"
    pl.concat(split_frames).write_parquet(all_path)
    written.append(all_path)
    print(
        f"    all: {sum(len(f) for f in split_frames):,} rows ({all_path.stat().st_size / 1e6:.1f} MB)"
    )
    return written


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        name="firm_characteristics",
        description=(
            f"{FIRM_CHAR_MAX_ENTITIES} most-observed anonymous firms per published "
            "split, built from the char/*.npz tensors so symbol identity survives"
        ),
        build=build_firm_characteristics,
        subdir=Path("equities") / "firm_characteristics",
        budget={"max_entities": FIRM_CHAR_MAX_ENTITIES},
    ),
)

DATASETS_BY_NAME = {dataset.name: dataset for dataset in DATASETS}


def write_manifest(output: Path, built: dict[str, list[Path]]) -> Path:
    """Rewrite data/manifest.json from what this run actually produced.

    Only the datasets built in this run are refreshed; entries for datasets not
    selected are carried over, so a single ``--dataset`` run does not blank the
    rest of the manifest.
    """
    manifest_path = output / "manifest.json"
    manifest: dict = {"version": "1", "subsets": {}, "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("subsets", {})
        manifest.setdefault("files", {})

    for name, paths in built.items():
        dataset = DATASETS_BY_NAME[name]
        manifest["subsets"][name] = {**dataset.budget, "description": dataset.description}
        for path in paths:
            rel = path.relative_to(output).as_posix()
            size = path.stat().st_size
            manifest["files"][rel] = {"size_bytes": size, "size_mb": round(size / 1e6, 2)}

    manifest["files"] = dict(sorted(manifest["files"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Production data root to subsample from (e.g. ~/ml4t/code/data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fixture data root to write (the test-data repo's data/ directory)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS_BY_NAME),
        help="Build only this dataset; repeatable. Default: all.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove each selected dataset's output directory before writing it",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the plan, write nothing")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"--source is not a directory: {source}")

    selected = [DATASETS_BY_NAME[name] for name in (args.dataset or sorted(DATASETS_BY_NAME))]

    print("=== ML4T test-data subsampler ===")
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Datasets: {', '.join(d.name for d in selected)}")
    print()

    if args.dry_run:
        for dataset in selected:
            print(f"  [dry-run] {dataset.name}: {dataset.description}")
        return 0

    output.mkdir(parents=True, exist_ok=True)
    built: dict[str, list[Path]] = {}
    for dataset in selected:
        print(f"  {dataset.name}: {dataset.description}")
        if args.clean:
            target = output / dataset.subdir
            if target.exists():
                shutil.rmtree(target)
        built[dataset.name] = dataset.build(source, output)
        print()

    manifest_path = write_manifest(output, built)
    print(f"Manifest: {manifest_path}")
    total = sum(path.stat().st_size for paths in built.values() for path in paths)
    print(f"Wrote {sum(len(p) for p in built.values())} files, {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
