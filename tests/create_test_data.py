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

DATASETS covers the fixtures whose absence or staleness has taken a CI job red,
not every fixture the test-data repo carries. The rest - equities bars, crypto,
futures, FX, the nasdaq minute set, options - were produced before this script
existed and have no spec here yet, so this is not a from-empty rebuild of the
fixture repo: it operates on a checkout of ml4t/third-edition-test-data and
replaces the datasets it knows. Reconstructing the remaining specs is tracked in
``agents/issues`` (test-data regeneration coverage).

Usage:
    # Every dataset declared below, over an existing test-data checkout
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
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

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
    # Every column build_features_and_matrix emits, not just the ones a consumer
    # names: 22_rag/07 rebuilds this frame and compares it with
    # assert_frame_equal, which checks the whole schema, so a source missing a
    # derived column passes a partial check here and fails the notebook's parity
    # assertion instead.
    "stock_features.parquet": (
        "cusip",
        "issuer_name",
        "n_inst_holders",
        "total_inst_value_usd",
        "avg_position_size_usd",
        "position_size_std_usd",
        "timestamp",
        "ownership_hhi",
        "inst_coverage_pct",
        "position_cv",
        "inst_value_change_usd",
        "inst_pct_change",
    ),
}
_13F_FILES = tuple(_13F_REQUIRED_COLUMNS)


def _load_13f_producer():
    """Import data/equities/positioning/13f_download.py by path.

    Its filename starts with a digit, so it cannot be reached by an import
    statement.
    """
    path = REPO_ROOT / "data" / "equities" / "positioning" / "13f_download.py"
    spec = importlib.util.spec_from_file_location("ml4t_13f_download", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"13F producer not found at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_derived_13f_matches_holdings(source_dir: Path) -> None:
    """Rebuild edges and stock features from the holdings and require equality.

    A column-name check passes an artifact whose dtypes, column order or values
    are those of an older producer generation. 22_rag/07 rebuilds both frames and
    compares them with ``assert_frame_equal``, which checks all of that, so the
    same comparison belongs here - before the stale copy enters the fixture set,
    rather than after a chapter job goes red against it.

    The rebuild is keyed on the producer's configured institution set, not on
    whichever CIKs the holdings happen to carry. Keyed on the holdings, a file
    missing one of the ten institutions is internally consistent and would pass,
    which is the silent coverage gap this builder exists to prevent: the notebook
    asks for all ten by CIK. Passing the configured set also reproduces the
    producer's own step-back rule, so a quarter that is still filling in cannot
    enter either side of the ownership-change comparison.
    """
    producer = _load_13f_producer()
    required_ciks = [cik for _, cik in producer.INSTITUTIONS]
    holdings = pl.read_parquet(source_dir / "institutional_holdings.parquet")
    if absent := sorted(set(required_ciks) - set(holdings["cik"].unique().to_list())):
        raise ValueError(
            f"Source 13F holdings at {source_dir} cover none of the filings of {absent}, "
            "which data/equities/positioning/13f_download.py requests by CIK. The fixture "
            "would be internally consistent and still miss an institution the notebook asks "
            "for; regenerate all three artifacts."
        )
    features, edges, *_ = producer.build_features_and_matrix(holdings, expected_ciks=required_ciks)
    for filename, rebuilt, keys in (
        ("institution_stock_edges.parquet", edges, ["institution_id", "stock_id"]),
        ("stock_features.parquet", features, ["cusip"]),
    ):
        stored = pl.read_parquet(source_dir / filename)
        try:
            assert_frame_equal(rebuilt.sort(keys), stored.sort(keys), check_row_order=True)
        except AssertionError as exc:
            raise ValueError(
                f"Source 13F artifact at {source_dir / filename} is not what the current "
                f"producer builds from institutional_holdings.parquet: {exc}. Regenerate "
                "all three artifacts with data/equities/positioning/13f_download.py."
            ) from exc


def build_institutional_holdings_13f(source: Path, output: Path) -> list[Path]:
    """Copy the production 13F artifacts into the fixture set verbatim.

    No reduction is applied. Together these are about three megabytes, and the
    notebook filters to a hardcoded CIK list, so any subsample keyed on
    institution risks dropping a CIK the notebook asks for and turning a schema
    fixture into a silent coverage gap.
    """
    source_dir = source / _13F_DIR
    for filename in _13F_FILES:
        src = source_dir / filename
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

    _require_derived_13f_matches_holdings(source_dir)

    written: list[Path] = []
    for filename in _13F_FILES:
        dst = output / _13F_DIR / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / filename, dst)
        rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
        print(f"    {filename}: {rows:,} rows ({dst.stat().st_size / 1e6:.1f} MB), copied verbatim")
        written.append(dst)
    return written


# --- nasdaq100 minute bars ----------------------------------------------------
#
# This fixture is reduced on symbol and on session, never on the clock. The
# previous one kept every session and every symbol but only every fifth minute,
# so CI ran a one-minute dataset at five-minute spacing: every declared horizon
# was five times its nominal length, `fwd_ret_5m` resolved to a single bar, and
# the microstructure chapter analysed bars the exchange never published. A
# horizon is a duration, the notebooks convert it to a bar count by measuring the
# grid, and a fixture that changes the grid changes what every consumer computes.
#
# So whole sessions are kept at native one-minute resolution, including extended
# hours, which 03_market_microstructure profiles by session. They are kept in
# consecutive runs rather than one session in six, because not every window in
# this case study is session-bounded: 04_model_based_features fits HAR on a
# 120-bar rolling window and signatures on 30-bar windows spanning overnight
# gaps, and it documents the resulting contamination in its own prose. One
# session in six would silently turn every one of those overnight gaps into a
# six-session gap. Inside a run the gaps are the production gaps.
#
# The runs are spread evenly over the two years rather than taken as one block,
# because the case study builds two cross-validation folds plus a holdout over
# the full span and validate_temporal_fold_coverage fails when an artifact
# covers none of a fold.
#
# The final calendar quarter is exempt from the stride and kept whole. Four
# Chapter 18 notebooks configure 2021-10-01..2021-12-31 by name and estimate on
# 20-session rolling windows inside it, so a stride that samples that quarter
# leaves them nothing to measure. What each one needs of the quarter, and where
# it says so:
#
#   02_spread_estimation           CS_WINDOW = ROLL_WINDOW = 20 daily rows per
#                                  symbol, and >= 2 symbols surviving both.
#   03_market_impact_calibration   >= 4 symbols with a positive normalized Huber
#                                  coefficient (`len(lambda_cross) < 4` raises).
#                                  Measured over the six fixture symbols: the
#                                  whole quarter gives 5, and every shorter
#                                  window inside it gives exactly 4, because
#                                  MSFT's coefficient is +0.001 over the quarter
#                                  and negative over any part of it.
#   07_ml4t_volume_participation   CALIBRATION_SESSIONS = 20 completed sessions
#                                  for AAPL before `execution_sessions[0]`.
#   11_cost_cliff                  SPREAD_START_DATE..SPREAD_END_DATE, which is
#                                  December 2021 alone.
#
# The binding minimum is 03's, and it is the only one the fixture cannot meet by
# holding part of the quarter. 64 sessions is therefore the smallest sample that
# satisfies all four. The sessions before the cutoff keep exactly the selection
# they had, so nothing the microstructure case study reads changes.

NASDAQ100_MINUTE_DIR = Path("equities") / "market" / "nasdaq100" / "minute_bars"
NASDAQ100_MINUTE_SYMBOLS = ("AAPL", "AMD", "AMZN", "FB", "GOOGL", "MSFT")
# Six consecutive sessions kept per six skipped runs: one week in six, which
# keeps a fifth of the fixture's sessions preceded by their true predecessor.
NASDAQ100_MINUTE_RUN_SESSIONS = 6
NASDAQ100_MINUTE_RUN_STRIDE = 6
# Sessions from this date on are kept whole, whatever the stride would have done
# with them. Chapter 18 names this quarter; see the comment above.
NASDAQ100_MINUTE_DENSE_FROM = date(2021, 10, 1)


def _session_runs(sessions: pl.Series, run: int, stride: int, dense_from: date) -> pl.Series:
    """Every ``stride``-th run of ``run`` sessions, then every session from ``dense_from``."""
    strided = sessions.filter(sessions < dense_from)
    keep: list = []
    for start in range(0, len(strided) - run + 1, run * stride):
        keep.extend(strided[start : start + run].to_list())
    keep.extend(sessions.filter(sessions >= dense_from).to_list())
    return pl.Series("date", keep, dtype=sessions.dtype)


def build_nasdaq100_minute_bars(source: Path, output: Path) -> list[Path]:
    """One week of sessions in six, plus the final quarter whole, at one-minute spacing."""
    source_dir = source / NASDAQ100_MINUTE_DIR
    if not source_dir.exists() or not list(source_dir.glob("year=*")):
        raise FileNotFoundError(
            f"NASDAQ-100 minute bars not found at {source_dir}. Build them with "
            "data/equities/market/algoseek_convert.py."
        )

    # Not hive_partitioning: the production layout is year=/month=, and reading
    # those as columns would add two the fixture's consumers do not expect.
    lf = pl.scan_parquet(source_dir / "**/*.parquet", hive_partitioning=False).filter(
        pl.col("symbol").is_in(NASDAQ100_MINUTE_SYMBOLS)
    )
    sessions = lf.select("date").unique().collect()["date"].sort()
    keep = _session_runs(
        sessions,
        NASDAQ100_MINUTE_RUN_SESSIONS,
        NASDAQ100_MINUTE_RUN_STRIDE,
        NASDAQ100_MINUTE_DENSE_FROM,
    )
    frame = lf.filter(pl.col("date").is_in(keep.implode())).collect().sort("symbol", "timestamp")

    # The reduction is only sound if the grid it leaves is the production grid.
    # Measured the way the notebooks measure it: consecutive spacing within a
    # symbol-session, which must be one minute and nothing else.
    gap = pl.col("timestamp") - pl.col("timestamp").shift(1).over("symbol", "date")
    spacing = frame.select(gap.drop_nulls().unique().alias("gap"))["gap"].to_list()
    if spacing != [timedelta(minutes=1)]:
        raise ValueError(f"fixture grid is not one-minute within a session: {sorted(spacing)}")

    written: list[Path] = []
    for year, part in frame.group_by(pl.col("date").dt.year(), maintain_order=True):
        dst = output / NASDAQ100_MINUTE_DIR / f"year={year[0]}" / "data.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(dst)
        print(
            f"    year={year[0]}: {len(part):,} rows ({dst.stat().st_size / 1e6:.1f} MB), "
            f"{part['date'].n_unique()} sessions, {part['symbol'].n_unique()} symbols"
        )
        written.append(dst)
    return written


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        name="nasdaq100_minute_bars",
        description=(
            f"{len(NASDAQ100_MINUTE_SYMBOLS)} symbols, every "
            f"{NASDAQ100_MINUTE_RUN_STRIDE}th run of "
            f"{NASDAQ100_MINUTE_RUN_SESSIONS} consecutive sessions kept whole at "
            "native one-minute spacing including extended hours, and every session "
            f"from {NASDAQ100_MINUTE_DENSE_FROM.isoformat()} on"
        ),
        build=build_nasdaq100_minute_bars,
        owns=(NASDAQ100_MINUTE_DIR,),
        budget={
            "symbols": list(NASDAQ100_MINUTE_SYMBOLS),
            "session_run": NASDAQ100_MINUTE_RUN_SESSIONS,
            "run_stride": NASDAQ100_MINUTE_RUN_STRIDE,
            "dense_from": NASDAQ100_MINUTE_DENSE_FROM.isoformat(),
            "bar_spacing": "1m (unchanged from production)",
        },
    ),
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


def roots_overlap(source: Path, output: Path) -> str | None:
    """Return why these roots cannot be used together, or None.

    The fixture root and the production root have to be separate trees. With
    ``--clean`` the same path on both sides deletes the production tensors and 13F
    artifacts before the build reads them; without it, the copy would be a file
    onto itself. Nesting either way is the same hazard one level down.
    """
    if source == output:
        return f"--source and --output are the same directory: {source}"
    if output in source.parents:
        return f"--output {output} contains --source {source}"
    if source in output.parents:
        return f"--output {output} is inside --source {source}"
    return None


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
    if overlap := roots_overlap(source, output):
        parser.error(overlap)

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
