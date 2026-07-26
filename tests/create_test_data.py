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
        owns: Paths under the output root, files or directories, that this dataset
            is solely responsible for. ``--clean`` removes exactly these and
            nothing else, so a dataset that shares a directory with an artifact it
            does not generate must name the individual files rather than the
            directory.
        budget: Manifest ``subsets`` entry describing the reduction (e.g.
            ``{"max_entities": 200}``). Recorded verbatim in the manifest.
    """

    name: str
    description: str
    build: Callable[[Path, Path], list[Path]]
    owns: tuple[Path, ...]
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


# --- institutional holdings (13F) ---------------------------------------------

_13F_DIR = Path("equities") / "positioning" / "13f"

# The canonical columns of each artifact data/equities/positioning/13f_download.py
# emits, as required by their consumers. Validating one artifact is not enough:
# the three are generated together and a stale copy of any of them is a different
# schema, not a smaller fixture. Concretely, 22_rag/07 refuses to run without
# holdings.put_call and holdings.report_date (report_date is the SEC
# period-of-report, distinct from filing_date, and put_call marks the option
# positions the notebook separates from share holdings), and
# 10_text_feature_engineering/02 reads stock_features.issuer_name, which an older
# generation of the producer spelled `issuer`.
#
# Listed per file rather than as one set so the error names the artifact that is
# stale. The co-ownership matrix is deliberately absent from the fixture: the
# notebook computes it from the holdings, and the production .npy is 255MB.
_13F_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "institutional_holdings.parquet": (
        "cik",
        "accession_no",
        "cusip",
        "issuer",
        "shares",
        "value_thousands",
        "put_call",
        "report_date",
        "filing_date",
        "company_name",
    ),
    "institution_stock_edges.parquet": (
        "institution_id",
        "stock_id",
        "institution_name",
        "stock_name",
        "weight_value",
        "weight_shares",
        "report_date",
        "timestamp",
    ),
    "stock_features.parquet": (
        "cusip",
        "issuer_name",
        "n_inst_holders",
        "total_inst_value_usd",
        "timestamp",
    ),
}
_13F_FILES = tuple(_13F_REQUIRED_COLUMNS)


def build_institutional_holdings_13f(source: Path, output: Path) -> list[Path]:
    """Copy the production 13F artifacts into the fixture set verbatim.

    No reduction is applied. Together these are about three megabytes, and the
    notebook filters to a hardcoded CIK list, so any subsample keyed on
    institution risks dropping a CIK the notebook asks for and turning a schema
    fixture into a silent coverage gap.
    """
    written: list[Path] = []
    for filename in _13F_FILES:
        src = source / _13F_DIR / filename
        if not src.exists():
            raise FileNotFoundError(
                f"13F artifact not found at {src}. Fetch it with "
                "data/equities/positioning/13f_download.py."
            )

        names = set(pl.scan_parquet(src).collect_schema().names())
        if missing := sorted(set(_13F_REQUIRED_COLUMNS[filename]) - names):
            raise ValueError(
                f"Source 13F artifact at {src} is missing {missing}. It predates the "
                "canonical downloader; regenerate all three artifacts with "
                "data/equities/positioning/13f_download.py before building the fixture."
            )

        dst = output / _13F_DIR / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
        print(f"    {filename}: {rows:,} rows ({dst.stat().st_size / 1e6:.1f} MB), copied verbatim")
        written.append(dst)
    return written


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        name="firm_characteristics",
        description=(
            f"{FIRM_CHAR_MAX_ENTITIES} most-observed anonymous firms per published "
            "split, built from the char/*.npz tensors so symbol identity survives"
        ),
        build=build_firm_characteristics,
        owns=(Path("equities") / "firm_characteristics",),
        budget={"max_entities": FIRM_CHAR_MAX_ENTITIES},
    ),
    Dataset(
        name="institutional_holdings_13f",
        description=(
            "the whole production 13F artifacts, already small enough to ship "
            "intact and carrying the SEC provenance columns readers filter on"
        ),
        build=build_institutional_holdings_13f,
        # Named individually, not as the 13f/ directory: bulk/ sits beside them and
        # is produced elsewhere, so --clean must not take it.
        owns=tuple(Path("equities") / "positioning" / "13f" / name for name in _13F_FILES),
        budget={"subsample": "none"},
    ),
)

DATASETS_BY_NAME = {dataset.name: dataset for dataset in DATASETS}


def _owned_by(dataset: Dataset, rel: str) -> bool:
    """Whether ``rel`` (manifest key, POSIX-relative) falls under the dataset."""
    return any(
        rel == owned.as_posix() or rel.startswith(f"{owned.as_posix()}/") for owned in dataset.owns
    )


def write_manifest(output: Path, built: dict[str, list[Path]]) -> Path:
    """Rewrite data/manifest.json from what this run actually produced.

    Only the datasets built in this run are refreshed; entries for datasets not
    selected are carried over, so a single ``--dataset`` run does not blank the
    rest of the manifest.

    A rebuilt dataset's old entries are dropped before the new ones are recorded,
    keyed on its ``owns`` paths. Merging instead would leave an artifact that has
    been renamed or discontinued listed forever, and the manifest is the only
    record of what the fixture set is supposed to contain.
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
        manifest["files"] = {
            rel: entry for rel, entry in manifest["files"].items() if not _owned_by(dataset, rel)
        }
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
            for owned in dataset.owns:
                target = output / owned
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
        built[dataset.name] = dataset.build(source, output)
        print()

    manifest_path = write_manifest(output, built)
    print(f"Manifest: {manifest_path}")
    total = sum(path.stat().st_size for paths in built.values() for path in paths)
    print(f"Wrote {sum(len(p) for p in built.values())} files, {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
