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

``tests/test_fixture_manifest_matches_builders.py`` checks each entry of the
manifest against a declaration and against the data on disk. What it cannot check
is a file no declaration mentions. A fixture in that state has no builder, no
recorded budget and nothing comparing it to the datasets it has to join against -
which is how the FNSPID news fixture came to sit entirely past the end of its own
price panel, and how an options panel built from a different universe than its
loader documents survived a year of green CI. ``UNPRODUCED`` in
``tests/test_every_fixture_file_has_a_producer.py`` is what is left of that backlog,
and it only shrinks.

It is also not a from-empty rebuild of the fixture repo: it operates on a checkout
of ml4t/third-edition-test-data and replaces the datasets it is asked for.

``--reconcile-manifest`` rewrites the manifest from those declarations and the
files that are there, building nothing. Use it when a declaration moves without
a regeneration; use a real regeneration when the data has to move.

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
from datetime import UTC, date, datetime, timedelta
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
        entities: ``{path: (column, count)}`` the built fixture has to carry, taken
            from the builder's own constants. Checked against the data on disk, which
            is the only place the drift shows: the manifest and the
            ``nasdaq100_minute_bars`` builder agreed with each other on six symbols
            while the fixture carried twelve, so comparing the two proved nothing.
    """

    name: str
    description: str
    build: Callable[[Path, Path], list[Path]]
    owns: tuple[Path, ...]
    budget: dict[str, object] = field(default_factory=dict)
    entities: dict[str, tuple[str, int]] = field(default_factory=dict)


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
# Twelve, not six. Six cannot carry a long-short cross-sectional sweep: signals.py
# clamps each side to n_assets // 2, so the smallest k in nasdaq100_microstructure's
# top_k_grid - 5 - needs ten distinct names, and at six the fixture could only ever
# trade three a side while CI registered a top_k=5 identity it had not run. The six
# added are the highest-volume nasdaq100 members not already present, so the widened
# cross-section carries dispersion rather than six more of the same regime.
NASDAQ100_MINUTE_SYMBOLS = (
    "AAPL", "AMD", "AMZN", "CMCSA", "CSCO", "FB",
    "GOOGL", "INTC", "MSFT", "QCOM", "SIRI", "TSLA",
)  # fmt: skip
# GitHub rejects any blob over 100 MB, and the widened year=2021 is 142 MB whole. Each
# year is written as this many contiguous blocks of sessions instead. load_nasdaq100_bars
# globs '**/*.parquet' under the hive root, so the split reads back as one panel.
NASDAQ100_MINUTE_PARTS_PER_YEAR = 2
NASDAQ100_MINUTE_MAX_BLOB_BYTES = 100_000_000
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


def _contiguous_session_blocks(sessions: pl.Series, parts: int) -> list[pl.Series]:
    """Split sorted sessions into ``parts`` contiguous, near-equal blocks.

    Contiguous on the date axis so each part is a date range rather than a stripe:
    a reader who opens one file sees a continuous stretch of the panel, and the
    split is reproducible from the session list alone.
    """
    total = len(sessions)
    size = -(-total // parts)  # ceiling, so the last block is the short one
    return [sessions[start : start + size] for start in range(0, total, size)]


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

    # Written to a staging directory and moved into place only once every part is
    # written and under the blob limit. The hive root has to be emptied - the loader
    # globs '**/*.parquet' under it, so a part left over from a previous generation
    # is silently unioned into the panel rather than replaced - and emptying it
    # before the build could leave the fixture with no bars at all, or with half a
    # year, if a part turns out to be too large to push.
    root = output / NASDAQ100_MINUTE_DIR
    staging = root.parent / f"{root.name}.staging"
    if staging.is_dir():
        shutil.rmtree(staging)

    staged: list[Path] = []
    for year, year_frame in frame.group_by(pl.col("date").dt.year(), maintain_order=True):
        sessions_in_year = year_frame["date"].unique().sort()
        blocks = _contiguous_session_blocks(sessions_in_year, NASDAQ100_MINUTE_PARTS_PER_YEAR)
        for index, block in enumerate(blocks):
            part = year_frame.filter(pl.col("date").is_in(block.implode()))
            dst = staging / f"year={year[0]}" / f"data-{index:02d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            part.write_parquet(dst)
            size = dst.stat().st_size
            if size > NASDAQ100_MINUTE_MAX_BLOB_BYTES:
                shutil.rmtree(staging)
                raise ValueError(
                    f"year={year[0]} part {index} is {size / 1e6:.1f} MB, above GitHub's "
                    f"100 MB blob limit. Raise NASDAQ100_MINUTE_PARTS_PER_YEAR above "
                    f"{NASDAQ100_MINUTE_PARTS_PER_YEAR}."
                )
            print(
                f"    year={year[0]} part {index}: {len(part):,} rows ({size / 1e6:.1f} MB), "
                f"{part['date'].n_unique()} sessions, {part['symbol'].n_unique()} symbols"
            )
            staged.append(dst)

    if root.is_dir():
        shutil.rmtree(root)
    staging.rename(root)
    return [root / path.relative_to(staging) for path in staged]


# --- S&P 500 share bars and the option chains built on them -------------------
#
# Four artifacts that only make sense together: the share bars carry the security
# identity every window in sp500_equity_option_analytics is bounded by, the daily
# straddle panel names the contracts sp500_options trades, and the raw chain is
# where those contracts' premiums are read from. Generating them separately is how
# the fixture reached a state where the straddle panel offered 22 underlyings and
# the raw chain could price 8 of them, so `ret_to_expiry` was null for the other
# 14 and no date carried the ten names the stage-04 IC screen needs.
#
# The symbol set is the 23 the fixture already carried plus seven large caps with
# complete coverage. Two floors set the size, and both are measured on the
# cross-section of a single date rather than on the total:
#
#   sp500_options/04_model_based_features   MIN_SYMBOLS_PER_DATE = 10 names quoted
#                                           on a date before it contributes an IC.
#   sp500_equity_option_analytics/01        cross_sectional_persistence's
#                                           min_entities = 20 securities shared by
#                                           a pair of decision dates.
#
# Thirty underlyings leave 26 to 28 on the surface on every date in the sample, so
# both clear with margin rather than by a name or two. The seven added are chosen
# for full coverage, and none of the 23 is dropped, so nothing that names a ticker
# changes.

SP500_DIR = Path("equities") / "market" / "sp500"
SP500_SYMBOLS = (
    "AAL", "AAPL", "AMD", "AMZN", "BA", "BAC", "C", "CHK", "CMCSA", "CSCO",
    "DIS", "F", "FB", "FCX", "FTR", "GE", "GOOG", "GOOGL", "INTC", "JPM",
    "MSFT", "MU", "NFLX", "NVDA", "PFE", "T", "TSLA", "UNH", "WFC", "XOM",
)  # fmt: skip
# The floor 01_feasibility_analysis measures at, asserted here so a future
# reduction cannot take the fixture back under it silently.
SP500_MIN_SURFACE_SYMBOLS_PER_DATE = 20

SP500_BARS = SP500_DIR / "daily_bars.parquet"
SP500_SURFACE = SP500_DIR / "options_surface_daily.parquet"
SP500_STRADDLES = SP500_DIR / "options_straddles_daily.parquet"
SP500_RAW_CHAIN = SP500_DIR / "options_straddles_raw"

# The key `_label_artifacts.py` looks a held contract up by, and the key the raw
# chain is reduced on: a row survives when the daily panel selected its contract.
_CONTRACT_KEYS = ["symbol", "strike", "expiration"]


def build_sp500_options(source: Path, output: Path) -> list[Path]:
    """Write the S&P 500 share bars, IV surface, straddle panel and raw chains.

    Each is the production artifact filtered to ``SP500_SYMBOLS``, except the raw
    chain, which is additionally reduced to the contracts the straddle panel
    selects. That reduction is what keeps the chain affordable: production carries
    every strike in the ATM band from first listing, and only the ones a straddle
    is actually entered on are ever read back - by ``_label_artifacts`` for the
    entry and exit premiums, by ``_straddle_moves`` for the premium path, and by
    the hold-to-maturity backtest for the lifecycle. Everything else is weight.
    """
    out_dir = output / SP500_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = list(SP500_SYMBOLS)
    written: list[Path] = []

    def _subset(rel: Path) -> pl.DataFrame:
        path = source / rel
        if not path.exists():
            raise FileNotFoundError(
                f"{rel} not found under {source}. Build it with "
                "data/equities/market/sp500/materialize_options.py."
            )
        return pl.scan_parquet(path).filter(pl.col("symbol").is_in(keep)).collect()

    bars = _subset(SP500_BARS)
    # sec_id identifies a price series, and every trailing window in the equity
    # option analytics case study is bounded by it. A fixture that gives two
    # tickers one sec_id pools two companies inside a window that is supposed to
    # sit in one; the previous build gave AMZN, GOOG and GOOGL sec_id 0.
    # Checked as "one sec_id, one symbol", not as "no two rows share a (sec_id,
    # timestamp)": two tickers whose histories do not overlap can share an id and
    # collide on neither key while still pooling two companies inside one window.
    # The converse is allowed and occurs - DIS carries two sec_ids here, which is a
    # security change and is what the id exists to express.
    collisions = (
        bars.group_by("sec_id")
        .agg(pl.col("symbol").unique().alias("symbols"))
        .filter(pl.col("symbols").list.len() > 1)
    )
    if collisions.height:
        raise ValueError(f"sec_id does not identify a price series: {collisions.to_dicts()}")
    bars.write_parquet(output / SP500_BARS)
    written.append(output / SP500_BARS)
    print(
        f"    daily_bars: {bars.height:,} rows, {bars['symbol'].n_unique()} symbols, "
        f"{bars['sec_id'].n_unique()} securities"
    )

    surface = _subset(SP500_SURFACE)
    per_date = (
        surface.drop_nulls("iv_30_atm").group_by("date").agg(pl.col("symbol").n_unique().alias("n"))
    )
    if (thin := per_date.filter(pl.col("n") < SP500_MIN_SURFACE_SYMBOLS_PER_DATE)).height:
        raise ValueError(
            f"{thin.height} dates carry fewer than {SP500_MIN_SURFACE_SYMBOLS_PER_DATE} "
            f"quoted surfaces, the floor 01_feasibility_analysis measures at "
            f"(thinnest {thin['n'].min()})"
        )
    surface.write_parquet(output / SP500_SURFACE)
    written.append(output / SP500_SURFACE)
    print(
        f"    options_surface_daily: {surface.height:,} rows, "
        f"{per_date['n'].min()} to {per_date['n'].max()} symbols per date"
    )

    straddles = _subset(SP500_STRADDLES)
    # Both legs of a straddle are one contract pair or it is not a straddle. The
    # build this fixture was carrying predated the pairing fix and left 11% of its
    # rows quoting a put at a different strike or expiration from its call.
    unpaired = straddles.filter(
        (pl.col("strike") != pl.col("put_strike"))
        | (pl.col("expiration") != pl.col("put_expiration"))
    )
    if unpaired.height:
        raise ValueError(
            f"{unpaired.height} of {straddles.height} straddle rows pair legs at a "
            "different strike or expiration"
        )
    straddles.write_parquet(output / SP500_STRADDLES)
    written.append(output / SP500_STRADDLES)
    print(
        f"    options_straddles_daily: {straddles.height:,} rows, "
        f"{straddles['symbol'].n_unique()} underlyings"
    )

    contracts = straddles.select(_CONTRACT_KEYS).unique()
    chain = (
        pl.scan_parquet(source / SP500_RAW_CHAIN / "year=*.parquet", hive_partitioning=False)
        .join(contracts.lazy(), on=_CONTRACT_KEYS, how="semi")
        .collect()
        .sort("date", "symbol", "expiration", "call_put", "strike")
    )
    # Checked per contract and per leg, not per underlying. `_label_artifacts` joins
    # on the exact (symbol, strike, expiration) key and reads a call and a put, so a
    # symbol that keeps one contract in the chain while losing another still yields a
    # null premium on every row of the one it lost - and a symbol-level check passes.
    # The 22-versus-8 split this replaces was only the coarsest form of that.
    uncovered = contracts.join(chain.select(_CONTRACT_KEYS).unique(), on=_CONTRACT_KEYS, how="anti")
    if uncovered.height:
        raise ValueError(
            f"{uncovered.height} of {contracts.height} contracts the straddle panel "
            f"selects have no row in the raw chain, e.g. {uncovered.head(3).to_dicts()}"
        )
    one_legged = (
        chain.group_by(_CONTRACT_KEYS)
        .agg(pl.col("call_put").n_unique().alias("legs"))
        .filter(pl.col("legs") < 2)
    )
    if one_legged.height:
        raise ValueError(
            f"{one_legged.height} selected contracts carry only one leg in the raw "
            f"chain, e.g. {one_legged.head(3).to_dicts()}"
        )

    raw_dir = output / SP500_RAW_CHAIN
    raw_dir.mkdir(parents=True, exist_ok=True)
    for (year,), part in chain.group_by(pl.col("date").dt.year(), maintain_order=True):
        dst = raw_dir / f"year={year}.parquet"
        part.write_parquet(dst)
        written.append(dst)
        print(
            f"    options_straddles_raw year={year}: {len(part):,} rows "
            f"({dst.stat().st_size / 1e6:.1f} MB)"
        )
    return written


# --- ETFs ---------------------------------------------------------------------
#
# The production panel carries 100 tickers; the fixture carries these 56, and the
# rule that chose them was not kept, so the list is the specification. What a
# builder does have to reproduce is the date column: production stores `timestamp`
# as a Datetime at midnight and the fixture stores it as a Date. Casting is the
# whole difference - filtering production to these 56 gives the fixture's 274,037
# rows exactly, and every value in them.

ETF_DIR = Path("etfs") / "market"
ETF_UNIVERSE = ETF_DIR / "etf_universe.parquet"
ETF_UNIVERSE_UNADJUSTED = ETF_DIR / "etf_universe_unadjusted.parquet"
ETF_METADATA = ETF_DIR / "etf_universe_metadata.json"
ETF_SYMBOLS = (
    "AGG",
    "BND",
    "DBC",
    "DIA",
    "DVY",
    "EEM",
    "EFA",
    "EMB",
    "EWA",
    "EWC",
    "EWG",
    "EWH",
    "EWI",
    "EWJ",
    "EWL",
    "EWN",
    "EWP",
    "EWQ",
    "EWT",
    "EWU",
    "EWW",
    "EWZ",
    "FXI",
    "GLD",
    "HYG",
    "IAU",
    "IEF",
    "IEFA",
    "IEMG",
    "IWM",
    "KRE",
    "LQD",
    "MDY",
    "QQQ",
    "SHY",
    "SLV",
    "SMH",
    "SPY",
    "TIP",
    "TLT",
    "USO",
    "VEA",
    "VNQ",
    "VTV",
    "VUG",
    "VWO",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)


def build_etfs(source: Path, output: Path) -> list[Path]:
    """Write the adjusted and unadjusted ETF panels for ``ETF_SYMBOLS``.

    The `timestamp` cast is not cosmetic: `utils.data_quality` and the chapter-2
    notebooks compare this column against `date`-typed calendars, and a Datetime
    at midnight compares unequal to the Date it prints as.

    The metadata sidecar is rewritten rather than copied. Nothing reads it -
    `data/etfs/market/download.py` is the only file in the repository that names
    it, and it writes it - but the one the fixture shipped described 50 tickers in
    a category schema production no longer uses, so it was a false statement about
    a fixture carrying 56.
    """
    keep = list(ETF_SYMBOLS)
    written: list[Path] = []
    (output / ETF_DIR).mkdir(parents=True, exist_ok=True)

    for rel in (ETF_UNIVERSE, ETF_UNIVERSE_UNADJUSTED):
        panel = (
            pl.scan_parquet(source / rel)
            .filter(pl.col("symbol").is_in(keep))
            .with_columns(pl.col("timestamp").cast(pl.Date))
            .collect()
        )
        missing = sorted(set(keep) - set(panel["symbol"].unique().to_list()))
        if missing:
            raise ValueError(f"{rel}: production carries no rows for {missing}")
        panel.write_parquet(output / rel)
        written.append(output / rel)
        print(f"    {rel.name}: {panel.height:,} rows, {panel['symbol'].n_unique()} symbols")

    # Written from the fixture, not edited down from production's copy. Production's
    # carries a `name` that counts a different universe and a `dictionary_file`
    # pointing at an absolute path outside the fixture, and both survive an edit that
    # only narrows `categories`.
    source_categories = json.loads((source / ETF_METADATA).read_text()).get("categories", {})
    categories = {
        category: [ticker for ticker in tickers if ticker in set(keep)]
        for category, tickers in source_categories.items()
    }
    uncategorized = sorted(set(keep) - {t for tickers in categories.values() for t in tickers})
    (output / ETF_METADATA).write_text(
        json.dumps(
            {
                "name": f"ML4T {len(keep)}-ETF CI fixture",
                "version": "1.0",
                "description": (
                    "The ETF universe the test-data repository ships, written by "
                    "tests/create_test_data.py from the production panel."
                ),
                "total_tickers": len(keep),
                "range": {
                    "start": str(panel["timestamp"].min()),
                    "end": str(panel["timestamp"].max()),
                    "frequency": "daily",
                },
                "categories": {c: tickers for c, tickers in categories.items() if tickers},
                "uncategorized": uncategorized,
            },
            indent=2,
        )
        + "\n"
    )
    written.append(output / ETF_METADATA)
    return written


# --- crypto perpetuals --------------------------------------------------------
#
# 10 of the 19 perpetuals production carries, across the hourly bars, the 8-hour
# premium index and the funding rate.
#
# Ten is the smallest width that realizes what the case study declares, not a round
# number. `config/setup.yaml` sweeps `top_k_grid [3, 5, 10]` long-short and
# `quantile_grid [5]`, and `case_studies/utils/sweep_config.py::get_entry_schemes_for`
# drops a top-k whose two disjoint legs do not fit (`2k > tradeable`) or that holds
# the whole cross-section (`k >= tradeable`), and drops a quantile axis below
# `2 * n_quantiles` names. At five names every scheme fell to one of those and the
# sweep had nothing to run - ml4t/agent-workspace#1077, which presented as
# `13_backtest` refusing and four notebooks failing downstream of it. At ten, k=3 and
# k=5 both seat and the quintile axis clears its floor exactly, so the fixture
# exercises the concentrated end, the diversified end and the quantile axis. Six
# would have satisfied the refusal with k=3 alone and left the fixture testing one
# scheme of the three a reader runs.
#
# The ten are the longest-lived of the nineteen; the fifth-newest of them starts
# 2020-08-13, so the panel is close to rectangular over the fixture window and its
# ragged head is short enough not to dominate the first fold.
#
# The bars stop two days before the other two files and before production. That is
# where the fixture was cut, and the bound is declared rather than removed because
# `intermediates/crypto_perps_funding` was generated against these bars: extending
# them without regenerating it puts rows in the input that no label, feature or
# fold covers, which is the shape of ml4t/agent-workspace#970. The premium index and
# the funding rate run to production's own last row, and the tail past the last bar
# joins to nothing.

CRYPTO_DIR = Path("crypto") / "market"
CRYPTO_PERPS = CRYPTO_DIR / "perps_1h.parquet"
CRYPTO_PREMIUM = CRYPTO_DIR / "premium_index_8h.parquet"
CRYPTO_FUNDING = CRYPTO_DIR / "funding_rate.parquet"
CRYPTO_SYMBOLS = (
    "ADAUSDT",
    "ATOMUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "COMPUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "LINKUSDT",
    "MKRUSDT",
    "XRPUSDT",
)
CRYPTO_PERPS_END = datetime(2025, 12, 29, 23, 0, tzinfo=UTC)


def build_crypto(source: Path, output: Path) -> list[Path]:
    """Write the perpetual bars, premium index and funding rate for ``CRYPTO_SYMBOLS``."""
    keep = list(CRYPTO_SYMBOLS)
    written: list[Path] = []
    (output / CRYPTO_DIR).mkdir(parents=True, exist_ok=True)

    for rel, end in (
        (CRYPTO_PERPS, CRYPTO_PERPS_END),
        (CRYPTO_PREMIUM, None),
        (CRYPTO_FUNDING, None),
    ):
        frame = pl.scan_parquet(source / rel).filter(pl.col("symbol").is_in(keep))
        if end is not None:
            frame = frame.filter(pl.col("timestamp") <= end)
        panel = frame.collect()
        missing = sorted(set(keep) - set(panel["symbol"].unique().to_list()))
        if missing:
            raise ValueError(f"{rel}: production carries no rows for {missing}")
        panel.write_parquet(output / rel)
        written.append(output / rel)
        print(
            f"    {rel.name}: {panel.height:,} rows, {panel['symbol'].n_unique()} symbols, "
            f"through {panel['timestamp'].max()}"
        )
    return written


# --- FX -----------------------------------------------------------------------
#
# Not reduced on any axis: the fixture is production's whole 20-pair panel at both
# frequencies, and its row counts equal production's exactly. It pins the panel and
# measures nothing about reduction. The pairs are still named here rather than left
# implicit, because "whatever production has" and "these twenty" are different
# statements and only the second one can fail when production changes.
#
# The timestamps are stored differently: the fixture is UTC-aware microseconds, the
# daily panel in production is a Date and the 4h panel a naive millisecond Datetime.
# `case_studies/fx_pairs` joins the two frequencies, so one representation is not
# optional.

FX_DIR = Path("fx") / "market"
FX_4H = FX_DIR / "4h.parquet"
FX_DAILY = FX_DIR / "daily.parquet"
FX_PAIRS = (
    "AUD_JPY",
    "AUD_NZD",
    "AUD_USD",
    "CAD_JPY",
    "CHF_JPY",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_CHF",
    "EUR_GBP",
    "EUR_JPY",
    "EUR_USD",
    "GBP_AUD",
    "GBP_CHF",
    "GBP_JPY",
    "GBP_USD",
    "NZD_JPY",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
)


def build_fx(source: Path, output: Path) -> list[Path]:
    """Write the 4h and daily FX panels for ``FX_PAIRS`` with UTC-aware timestamps."""
    keep = list(FX_PAIRS)
    written: list[Path] = []
    (output / FX_DIR).mkdir(parents=True, exist_ok=True)

    for rel in (FX_4H, FX_DAILY):
        panel = (
            pl.scan_parquet(source / rel)
            .filter(pl.col("symbol").is_in(keep))
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
            .collect()
        )
        missing = sorted(set(keep) - set(panel["symbol"].unique().to_list()))
        if missing:
            raise ValueError(f"{rel}: production carries no rows for {missing}")
        panel.write_parquet(output / rel)
        written.append(output / rel)
        print(f"    {rel.name}: {panel.height:,} rows, {panel['symbol'].n_unique()} pairs")
    return written


# --- US equities --------------------------------------------------------------
#
# 56 tickers of the 3,199 production carries, over the panel's whole 1962-2018
# range. The rule that chose them was not kept and the list is the specification;
# it is not a liquidity ranking - LBAI, MNTX, OCFC, PEBO and RDNT sit beside AAPL
# and MSFT, so the sample spans the size distribution rather than the top of it.

US_EQUITIES = Path("equities") / "market" / "us_equities" / "us_equities.parquet"
US_EQUITIES_TICKERS = (
    "AAL",
    "AAPL",
    "AIG",
    "AMAT",
    "AMD",
    "BAC",
    "BRCD",
    "BRCM",
    "C",
    "CAR",
    "CHK",
    "CSCO",
    "DAL",
    "DELL",
    "EBAY",
    "EMC",
    "ETFC",
    "F",
    "FB",
    "FCX",
    "FOXA",
    "GE",
    "GM",
    "GOOGL",
    "GRPN",
    "HPE",
    "INTC",
    "JDSU",
    "JNPR",
    "JPM",
    "KMI",
    "LBAI",
    "LVLT",
    "LVS",
    "MNTX",
    "MS",
    "MSFT",
    "MSI",
    "MU",
    "NVDA",
    "OCFC",
    "ORCL",
    "PEBO",
    "PFE",
    "PYPL",
    "QCOM",
    "RDNT",
    "S",
    "SIRI",
    "T",
    "TWTR",
    "TWX",
    "TYC",
    "WFC",
    "YHOO",
    "ZNGA",
)


def build_us_equities(source: Path, output: Path) -> list[Path]:
    """Write the US equities panel reduced to ``US_EQUITIES_TICKERS``."""
    keep = list(US_EQUITIES_TICKERS)
    (output / US_EQUITIES.parent).mkdir(parents=True, exist_ok=True)
    panel = pl.scan_parquet(source / US_EQUITIES).filter(pl.col("ticker").is_in(keep)).collect()
    missing = sorted(set(keep) - set(panel["ticker"].unique().to_list()))
    if missing:
        raise ValueError(f"{US_EQUITIES}: production carries no rows for {missing}")
    panel.write_parquet(output / US_EQUITIES)
    print(
        f"    us_equities: {panel.height:,} rows, {panel['ticker'].n_unique()} tickers, "
        f"{panel['date'].min().date()} to {panel['date'].max().date()}"
    )
    return [output / US_EQUITIES]


# --- FNSPID news --------------------------------------------------------------
#
# The one notebook that reads this dataset, `10_text_feature_engineering/
# 07_news_return_signals`, joins headlines to us_equities forward returns, so the
# sample has to sit inside the price panel's window or the join returns nothing.
# The upper bound is read off the panel `build_us_equities` writes rather than
# hardcoded, because the WIKI/Quandl wall it ends at is a property of that dataset.
#
# The fixture this replaced was 209 synthetic headlines over 2020-2023, entirely
# past that wall: 07 joined 109 feature rows against zero overlapping trading
# dates, wrote an empty `news_features.parquet` and exited 0 - ml4t/agent-workspace#1116.
#
# Production's column names are kept verbatim. The notebook detects its text, date
# and ticker columns by name and its priority order exists to skip
# `Textrank_summary`, so a fixture with tidied column names would not exercise the
# detection CI is there to run. `Author`, `Article` and the four summary columns
# are null in production and stay null here for the same reason.

FNSPID_NEWS = Path("alternative") / "news" / "fnspid" / "fnspid_test.parquet"
FNSPID_SOURCE = Path("alternative") / "news" / "fnspid" / "fnspid_1000k.parquet"

# The lower bound is a cross-section decision, not a calendar one. 07 computes a
# daily rank IC and skips any date carrying fewer than ten names, and bounded only
# above these 28 tickers reach ten on 228 of the 2,581 days they span, over 30,649
# headlines - concentrated in the last two years, as coverage widens. Cutting at
# 2017-01-01 keeps 93 of those 228 dates for 5,599 headlines: 41% of the usable
# cross-sections at 18% of the text. The notebook runs two transformer passes over
# every headline on one kernel thread, so the text volume is what is being bought.
FNSPID_START = date(2017, 1, 1)

# The 28 of US_EQUITIES_TICKERS the production download carries news for inside
# that window. Three separate reasons account for the other 28, and none of them
# is a defect here. The production file is a 1,000,000-row prefix of the
# HuggingFace dataset, physically ordered by ticker and ending mid-P at PG, so the
# 12 that sort past it (PYPL, QCOM, RDNT, S, SIRI, T, TWTR, TWX, TYC, WFC, YHOO,
# ZNGA) were never downloaded. Five more sort before PG and still carry no rows at
# all - DELL, EMC, GE, HPE, MSFT - so the prefix is not the whole story and FNSPID
# simply does not cover them here. The remaining 11 have news outside 2017-01-01
# to the price panel's end: AAPL, AMD, CAR, F, FB, GM, GOOGL, INTC, JDSU, JPM, LVS.
#
# Declared rather than derived so a changed download is a build failure naming the
# tickers it lost, not a quietly smaller fixture.
FNSPID_TICKERS = (
    "AAL",
    "AIG",
    "AMAT",
    "BAC",
    "BRCD",
    "BRCM",
    "C",
    "CHK",
    "CSCO",
    "DAL",
    "EBAY",
    "ETFC",
    "FCX",
    "FOXA",
    "GRPN",
    "JNPR",
    "KMI",
    "LBAI",
    "LVLT",
    "MNTX",
    "MS",
    "MSI",
    "MU",
    "NVDA",
    "OCFC",
    "ORCL",
    "PEBO",
    "PFE",
)


def build_fnspid(source: Path, output: Path) -> list[Path]:
    """Write the FNSPID headlines that fall inside the fixture price panel."""
    keep = list(FNSPID_TICKERS)
    price_end = (
        pl.scan_parquet(source / US_EQUITIES)
        .filter(pl.col("ticker").is_in(list(US_EQUITIES_TICKERS)))
        .select(pl.col("date").max())
        .collect()
        .item()
    )
    (output / FNSPID_NEWS.parent).mkdir(parents=True, exist_ok=True)
    # FNSPID stores the timestamp as "YYYY-MM-DD HH:MM:SS UTC"; only the day is ever
    # read, and slicing avoids parsing a million rows to discard the time.
    day = pl.col("Date").str.slice(0, 10).str.to_date()
    news = (
        pl.scan_parquet(source / FNSPID_SOURCE)
        .filter(pl.col("Stock_symbol").is_in(keep))
        .filter(day.is_between(pl.lit(FNSPID_START), pl.lit(price_end)))
        .sort("Date", "Stock_symbol")
        .collect()
    )
    missing = sorted(set(keep) - set(news["Stock_symbol"].unique().to_list()))
    if missing:
        raise ValueError(f"{FNSPID_SOURCE}: production carries no in-window news for {missing}")
    news.write_parquet(output / FNSPID_NEWS)
    days = news["Date"].str.slice(0, 10)
    print(
        f"    fnspid: {news.height:,} headlines, {news['Stock_symbol'].n_unique()} tickers, "
        f"{days.min()} to {days.max()} (price panel ends {price_end.date()})"
    )
    return [output / FNSPID_NEWS]


# --- CME futures --------------------------------------------------------------
#
# Two halves reduced differently, and deliberately. The daily panel is production's
# whole 30 products: `case_studies/cme_futures/config/setup.yaml` declares thirty
# and sizes `initial_cash` and `top_k_grid` for them, and `05_evaluation` keeps a
# date only where ten products carry the feature, so a reduced panel scores nothing.
# The hourly bars are read by one notebook, `02_financial_data_universe/
# 04_cme_futures_eda`, and only for ES, so eight products is already generous there
# and 22 more would be 300 MB of weight.
#
# Neither half is transformed, so the builder copies rather than rewrites: all 121
# files are byte-identical to production's, and a parquet round-trip through polars
# would change their bytes while changing nothing a reader can see.

CME_CONTINUOUS = Path("futures") / "market" / "continuous"
CME_DAILY = CME_CONTINUOUS / "daily" / "continuous_daily.parquet"
CME_HOURLY = CME_CONTINUOUS / "hourly"
CME_HOURLY_PRODUCTS = ("6E", "CL", "ES", "GC", "NQ", "SI", "ZB", "ZN")


def build_cme_futures(source: Path, output: Path) -> list[Path]:
    """Copy the whole daily panel and the hourly bars for ``CME_HOURLY_PRODUCTS``."""
    written: list[Path] = []

    (output / CME_DAILY).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / CME_DAILY, output / CME_DAILY)
    written.append(output / CME_DAILY)
    daily = pl.scan_parquet(output / CME_DAILY).select("product").collect()
    print(f"    continuous_daily: {daily.height:,} rows, {daily['product'].n_unique()} products")

    for product in CME_HOURLY_PRODUCTS:
        src_dir = source / CME_HOURLY / f"product={product}"
        if not src_dir.is_dir():
            raise FileNotFoundError(f"production carries no hourly bars for {product}: {src_dir}")
        parts = sorted(src_dir.glob("year=*/data.parquet"))
        if not parts:
            raise FileNotFoundError(f"no year partitions under {src_dir}")
        for part in parts:
            dst = output / CME_HOURLY / f"product={product}" / part.parent.name / part.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(part, dst)
            written.append(dst)
        print(f"    hourly product={product}: {len(parts)} year partitions")
    return written


# --- CFTC Commitment of Traders -----------------------------------------------
#
# 1.2 MB for the whole directory and nothing is transformed, so the builder copies:
# every fixture file is byte-identical to production's, and a parquet round-trip
# through polars would change their bytes while changing nothing a reader sees.
#
# It copies whatever products production carries rather than a declared list,
# because `data/futures/loader.py::list_cot_products` answers by enumerating this
# directory: a fixture holding a subset makes that loader return a different
# product list under CI than the one a reader gets. What is declared is the
# minimum - the products `04_fundamental_alternative_data/08_futures_positioning`
# loads by name - so a production directory that has lost one fails here rather
# than shipping a fixture the notebook cannot read.

COT_DIR = Path("futures") / "positioning" / "cot"
COT_REQUIRED_PRODUCTS = ("CL", "ES", "GC")


def build_cot(source: Path, output: Path) -> list[Path]:
    """Copy every per-product COT parquet production carries, verbatim."""
    source_dir = source / COT_DIR
    products = sorted(p.stem for p in source_dir.glob("*.parquet"))
    if missing := sorted(set(COT_REQUIRED_PRODUCTS) - set(products)):
        raise FileNotFoundError(
            f"Production carries no COT reports for {missing} at {source_dir}. "
            "08_futures_positioning loads those products by name. Fetch them with "
            f"data/futures/positioning/cot_download.py --products {','.join(missing)}."
        )

    written: list[Path] = []
    rows = 0
    for product in products:
        destination = output / COT_DIR / f"{product}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / f"{product}.parquet", destination)
        rows += pl.scan_parquet(destination).select(pl.len()).collect().item()
        written.append(destination)
    size = sum(path.stat().st_size for path in written) / 1e6
    print(f"    cot/: {len(written)} products, {rows:,} rows ({size:.1f} MB), copied verbatim")
    return written


# --- SEC filings, XBRL fundamentals and Form 4 --------------------------------
#
# Four fixtures that share a producer -- `data/equities/fundamentals/` and
# `data/equities/positioning/form4_download.py` -- and the loaders in
# `data/equities/loader.py` that read them.
#
# Three are production verbatim. Production's own SEC corpora are already the size
# a fixture wants (the sp100 reference panels are one row per filing with the text
# attached, the XBRL panel covers 20 CIKs), so a reduction would remove rows a
# reader can see for no saving. The fourth, the sp500 10-Q panel, is 477 symbols in
# production and is cut to the 13 the fixture carries.
#
# `equities/fundamentals/xbrl/filing_dates/` is not declared here. It is
# `xbrl_download.py`'s own HTTP cache of accession -> filing_date, written and read
# by that script alone; no loader, notebook or test opens it. It is deleted from
# the fixture rather than declared.

SEC_10K_REFERENCE = (
    Path("equities") / "fundamentals" / "10k" / "sp100" / "reference" / "all_10k_filings.parquet"
)
SEC_8K_REFERENCE = (
    Path("equities") / "fundamentals" / "8k" / "sp100" / "reference" / "all_8k_filings.parquet"
)
SEC_10Q_REFERENCE = (
    Path("equities") / "fundamentals" / "10q" / "sp500" / "reference" / "all_10q_filings.parquet"
)
# The 13 the fixture carries. `load_sp500_10q_mda` takes an optional symbol filter
# and defaults to all of them, so this list is what the fixture's readers see.
SEC_10Q_SYMBOLS = (
    "AAPL",
    "AMZN",
    "GOOG",
    "GOOGL",
    "JNJ",
    "MSFT",
    "NVDA",
    "PG",
    "TSLA",
    "UNH",
    "V",
    "WMT",
    "XOM",
)
XBRL_FUNDAMENTALS = Path("equities") / "fundamentals" / "xbrl" / "fundamentals.parquet"
FORM4_DIR = Path("equities") / "positioning" / "form4"


def _copy_verbatim(source: Path, output: Path, relative: Path, download: str) -> Path:
    """Copy one production file into the fixture, unchanged."""
    src = source / relative
    if not src.exists():
        raise FileNotFoundError(f"{relative} not found at {src}. Fetch it with {download}.")
    dst = output / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def build_sec_filing_references(source: Path, output: Path) -> list[Path]:
    """Copy the sp100 10-K and 8-K panels whole; cut the sp500 10-Q panel to 13 symbols."""
    download = "data/equities/fundamentals/filings_download.py"
    written: list[Path] = []
    for relative, form in ((SEC_10K_REFERENCE, "10-K"), (SEC_8K_REFERENCE, "8-K")):
        dst = _copy_verbatim(source, output, relative, f"{download} --form {form} --universe sp100")
        frame = pl.read_parquet(dst)
        print(
            f"    {relative.name}: {frame.height:,} filings, "
            f"{frame['symbol'].n_unique()} symbols, copied verbatim"
        )
        written.append(dst)

    src = source / SEC_10Q_REFERENCE
    if not src.exists():
        raise FileNotFoundError(
            f"{SEC_10Q_REFERENCE} not found at {src}. Fetch it with "
            f"{download} --form 10-Q --universe sp500."
        )
    frame = pl.read_parquet(src)
    missing = sorted(set(SEC_10Q_SYMBOLS) - set(frame["symbol"].unique().to_list()))
    if missing:
        raise ValueError(
            f"Production's sp500 10-Q panel carries no filings for {missing}. The fixture "
            "declares them, so a narrower panel would ship a fixture short of its own budget."
        )
    reduced = frame.filter(pl.col("symbol").is_in(SEC_10Q_SYMBOLS))
    dst = output / SEC_10Q_REFERENCE
    dst.parent.mkdir(parents=True, exist_ok=True)
    reduced.write_parquet(dst)
    print(
        f"    {SEC_10Q_REFERENCE.name}: {reduced.height:,} of {frame.height:,} filings, "
        f"{reduced['symbol'].n_unique()} of {frame['symbol'].n_unique()} symbols"
    )
    written.append(dst)
    return written


def build_xbrl_fundamentals(source: Path, output: Path) -> list[Path]:
    """Copy the XBRL quarterly fundamentals panel verbatim."""
    dst = _copy_verbatim(
        source, output, XBRL_FUNDAMENTALS, "data/equities/fundamentals/xbrl_download.py"
    )
    frame = pl.read_parquet(dst)
    print(
        f"    fundamentals.parquet: {frame.height:,} rows, {frame['symbol'].n_unique()} symbols, "
        "copied verbatim"
    )
    return [dst]


def build_form4(source: Path, output: Path) -> list[Path]:
    """Copy every Form 4 XML production carries, verbatim.

    `04_fundamental_alternative_data/03_sec_form4_insider_transactions` enumerates
    the ticker directories rather than naming one, so the fixture is whatever
    production holds; it raises if that is nothing.
    """
    source_dir = source / FORM4_DIR
    filings = sorted(source_dir.rglob("*.xml")) if source_dir.is_dir() else []
    if not filings:
        raise FileNotFoundError(
            f"No Form 4 filings under {source_dir}. Fetch them with "
            "data/equities/positioning/form4_download.py --ticker TSLA --count 20."
        )
    written: list[Path] = []
    for filing in filings:
        dst = output / FORM4_DIR / filing.relative_to(source_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filing, dst)
        written.append(dst)
    tickers = sorted({filing.relative_to(source_dir).parts[0] for filing in filings})
    print(f"    form4/: {len(written)} filings for {', '.join(tickers)}, copied verbatim")
    return written


# --- NASDAQ ITCH messages and the ES individual contracts ---------------------
#
# Four fixture files under paths that `tests/generate_test_microstructure.py`
# otherwise fills with synthetic data. They were widened from production on
# 2026-05-06 (ES, for the `timestamp`/`tenor`/`product` schema) and 2026-05-17
# (ITCH `A`, `P`, `R`, to unblock 08_financial_features/02_microstructure_features),
# and the generator was not told, so for four months the two producers wrote
# different content to the same paths and whichever ran last won. The generator now
# names them in its `PRODUCTION_SOURCED` set and refuses them; they are declared here.

ITCH_MESSAGES = Path("equities") / "market" / "microstructure" / "nasdaq_itch" / "messages"
# The five the fixture carries. Widening beyond them costs little, but
# 02_microstructure_features asserts >= 100 AAPL trade rows and reads three of these
# by name, so narrowing is what would break.
ITCH_SYMBOLS = ("AAPL", "GOOGL", "MSFT", "NVDA", "TSLA")
ITCH_ROWS_PER_SYMBOL = 500
# `R` is the stock directory: one row per symbol, so it is taken whole rather than
# capped. `A` and `P` are the message types a notebook reads.
ITCH_MESSAGE_TYPES = ("A", "P", "R")

ES_INDIVIDUAL = Path("futures") / "market" / "individual" / "ES" / "data.parquet"


def build_nasdaq_itch_messages(source: Path, output: Path) -> list[Path]:
    """Take the first ``ITCH_ROWS_PER_SYMBOL`` messages per symbol from production.

    Production stores each message type as 43 parquet parts in timestamp order. The
    reduction reads them in that order, keeps the head per symbol, and re-sorts by
    timestamp, which is what makes the result a contiguous early slice of the
    trading day rather than a sample scattered across it: a limit-order-book reader
    needs the adds that precede an event, and a random sample supplies neither side.

    `R` has one row per symbol and is taken whole.
    """
    written: list[Path] = []
    for message_type in ITCH_MESSAGE_TYPES:
        source_dir = source / ITCH_MESSAGES / message_type
        parts = sorted(source_dir.glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(
                f"No ITCH {message_type} messages at {source_dir}. Parse them with "
                "data/equities/market/microstructure/nasdaq_itch_download.py."
            )
        frame = pl.scan_parquet(parts).filter(pl.col("stock").is_in(ITCH_SYMBOLS)).collect()
        missing = sorted(set(ITCH_SYMBOLS) - set(frame["stock"].unique().to_list()))
        if missing:
            raise ValueError(
                f"Production ITCH {message_type} carries no messages for {missing}. "
                "The fixture's consumers read those symbols by name."
            )
        columns = frame.columns
        if message_type != "R":
            frame = (
                frame.group_by("stock", maintain_order=True)
                .head(ITCH_ROWS_PER_SYMBOL)
                .select(columns)
                .sort("timestamp")
            )
        destination = output / ITCH_MESSAGES / message_type / "part-000000.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(destination)
        print(f"    {message_type}/: {frame.height:,} rows, {frame['stock'].n_unique()} symbols")
        written.append(destination)
    return written


def build_individual_futures_es(source: Path, output: Path) -> list[Path]:
    """Copy the production ES individual-contract bars verbatim.

    290 KB for ten years across every listed contract month. A reduction would have
    to choose which months to keep, and the notebook that reads this file rolls
    between them, so dropping any is dropping the thing it demonstrates.

    CL and NQ sit beside this file and are synthetic, from
    `tests/generate_test_microstructure.py`; production carries neither.
    """
    src = source / ES_INDIVIDUAL
    if not src.exists():
        raise FileNotFoundError(
            f"ES individual contracts not found at {src}. Fetch them with "
            "data/futures/market/cme_download.py."
        )
    dst = output / ES_INDIVIDUAL
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
    print(
        f"    ES/data.parquet: {rows:,} rows ({dst.stat().st_size / 1e6:.1f} MB), copied verbatim"
    )
    return [dst]


# --- published factor returns, FRED macro and on-chain TVL ---------------------
#
# Four sources that are downloaded rather than derived, total about 9 MB against a
# 529 MB fixture, so all four builders copy: every fixture file is byte-identical to
# production's, and a parquet round-trip through polars would change their bytes
# while changing nothing a reader sees.
#
# Each copies whatever production carries rather than a declared list, and declares
# the MINIMUM instead -- the files a loader can name or a notebook opens by path. A
# production directory that has lost one fails here rather than shipping a fixture a
# notebook cannot read, and one that has gained a file ships it rather than silently
# omitting it.
#
# What this replaces is not a smaller builder. It is none: all 51 files were in the
# test-data repo with no producer, so nothing regenerated them from production and
# nothing said what they were meant to contain. The evidence they had drifted is in
# the row counts. Every AQR parquet the fixture carried was 2 to 64 rows SHORT of
# production, which is what an old snapshot looks like, and the FRED panel had drifted
# in both directions at once -- `fred_macro_initial_release` 49 rows AHEAD of
# production and `fred_macro_raw` 27,132 rows behind.

FF_DIR = Path("factors") / "fama-french"
AQR_DIR = Path("factors") / "aqr"
MACRO_DIR = Path("macro")
ONCHAIN_DIR = Path("crypto") / "onchain"

# The six `data/factors/loader.py::load_ff_factors` can name. Its two Literal
# parameters are the whole reachable set, so this is read off the signature rather
# than maintained: dataset in (ff3, ff5, mom) x frequency in (daily, monthly). The
# other four the fixture carries -- bp_me, ind_5, port_size, ff3_developed -- are
# reachable only by opening the path directly and are copied, not required.
FF_REQUIRED = tuple(
    f"{dataset}_{frequency}.parquet"
    for dataset in ("ff3", "ff5", "mom")
    for frequency in ("daily", "monthly")
)

# The four `load_aqr_factors` maps by name, plus the two that 01_process_is_edge/
# factor_regimes and 17_portfolio_construction/05_factor_allocation_evidence open by
# path. The rest are copied and not required.
AQR_REQUIRED = (
    "qmj_factors.parquet",
    "bab_factors.parquet",
    "hml_devil.parquet",
    "vme_factors.parquet",
    "century_premia.parquet",
    "tsmom.parquet",
)

# `fred_macro.parquet` is the aligned panel three teaching notebooks read by path;
# the raw and metadata files are what `data/macro/loader.py` names in its outputs.
#
# `fred_macro_initial_release.parquet` is here because it comes from a DIFFERENT
# download script - `download_alfred.py`, not `download.py` - so a production tree can
# hold every other file here and not this one. `04_fundamental_alternative_data/
# 07_macro_data_alignment` calls `load_macro_initial_release()` unconditionally, and
# without it that notebook raises where the build would have succeeded. The three not
# required - the two dictionaries and `initial_release_raw` - are copied because
# production carries them and read by nothing in this repo.
MACRO_REQUIRED = (
    "fred_macro.parquet",
    "fred_macro_raw.parquet",
    "fred_macro_metadata.parquet",
    "fred_macro_initial_release.parquet",
)

# 04_fundamental_alternative_data reads the per-chain files through an f-string over
# the chain name, so the four chains are required by construction, not by choice.
ONCHAIN_REQUIRED = (
    "defillama_tvl_total.parquet",
    "defillama_tvl_arbitrum.parquet",
    "defillama_tvl_bsc.parquet",
    "defillama_tvl_ethereum.parquet",
    "defillama_tvl_solana.parquet",
    "coingecko_ethereum.parquet",
)


def _copy_flat_directory(
    source: Path,
    output: Path,
    directory: Path,
    required: tuple[str, ...],
    patterns: tuple[str, ...],
    label: str,
    fetch_hint: str,
) -> list[Path]:
    """Copy production's files for one downloaded source, verbatim, and check the minimum.

    Args:
        source: Production data root.
        output: Fixture data root.
        directory: Path under both roots holding this source's files.
        required: File names a loader can name or a notebook opens by path. A
            production directory missing one of these fails rather than shipping a
            fixture that cannot satisfy it.
        patterns: Globs naming what belongs to this source. Deliberately not ``*``:
            the AQR directory also holds a ``source/`` tree of the original 15
            spreadsheets, 95 MB that no loader reads.
        label: Printed name for the build summary.
        fetch_hint: What to run when a required file is missing.

    Returns:
        The fixture paths written.

    Raises:
        FileNotFoundError: If production carries none of this source, or is missing a
            required file.
    """
    source_dir = source / directory
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Production carries no {label} at {source_dir}. {fetch_hint}")

    names = sorted({path.name for pattern in patterns for path in source_dir.glob(pattern)})
    if missing := sorted(set(required) - set(names)):
        raise FileNotFoundError(
            f"Production is missing {label} files {missing} at {source_dir}. {fetch_hint}"
        )

    written: list[Path] = []
    for name in names:
        destination = output / directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / name, destination)
        written.append(destination)
    size = sum(path.stat().st_size for path in written) / 1e6
    print(f"    {label}: {len(written)} files ({size:.1f} MB), copied verbatim")
    return written


def build_fama_french_factors(source: Path, output: Path) -> list[Path]:
    """Copy every Fama-French factor parquet production carries, verbatim."""
    return _copy_flat_directory(
        source,
        output,
        FF_DIR,
        FF_REQUIRED,
        ("*.parquet",),
        "fama-french/",
        "Fetch them with data/factors/ff_download.py.",
    )


def build_aqr_factors(source: Path, output: Path) -> list[Path]:
    """Copy AQR's published factor parquets and their metadata, verbatim.

    Not the ``source/`` tree beside them: those are the 15 original spreadsheets the
    parquets were converted from, 95 MB that no loader opens.
    """
    return _copy_flat_directory(
        source,
        output,
        AQR_DIR,
        AQR_REQUIRED,
        ("*.parquet", "metadata.json"),
        "aqr/",
        "Fetch them with data/factors/aqr_download.py.",
    )


def build_fred_macro(source: Path, output: Path) -> list[Path]:
    """Copy the FRED macro panels production carries, verbatim.

    Only the ``fred_macro*`` files. The production directory also holds the download
    scripts, a loader, a README and two profile JSONs, none of which a notebook reads
    through ``ML4T_DATA_PATH``.
    """
    return _copy_flat_directory(
        source,
        output,
        MACRO_DIR,
        MACRO_REQUIRED,
        ("fred_macro*.parquet",),
        "macro/",
        "Fetch them with data/macro/download.py and data/macro/download_alfred.py.",
    )


def build_crypto_onchain(source: Path, output: Path) -> list[Path]:
    """Copy the DefiLlama TVL and CoinGecko on-chain series, verbatim."""
    return _copy_flat_directory(
        source,
        output,
        ONCHAIN_DIR,
        ONCHAIN_REQUIRED,
        ("*.parquet",),
        "onchain/",
        "Fetch them with data/crypto/onchain/download.py.",
    )


# --- S&P 500 options EDA ------------------------------------------------------
#
# A separate dataset from `sp500_options` above, at `sp500/options_eda/`, read by
# `load_sp500_options_eda`. Four notebooks read it: 02/07 for chain structure, the
# smile and the IV surface, 02/08 for Greeks on AAPL 2020, 02/09 for the
# constant-maturity straddle series on AAPL 2019, and 08/03 for cross-instrument
# features on all eight underlyings.
#
# The budget below is taken from those notebooks' own declared parameters, because
# the fixture this replaces was narrower than every one of them and nothing said
# so. It carried moneyness 0.95-1.05 and nothing past 45 days to maturity, against
# 02/07's declarations:
#
#   CHAIN_BAND = (0.7, 1.3)   the smile's shape       - fixture covered 0.95-1.05
#   SPREAD_BAND = (0.5, 1.5)  "what happens in the wings is its subject"
#   WING_BAND = (0.9, 1.1)    splits near-the-money from away-from-the-money
#   SURFACE_MAX_DAYS = 180    far expirations drawn as their own section
#
# So the smile had no smile, the wing analysis had no wings, "away from the money"
# was empty, and the far-expiration section drew nothing - and the notebook passed,
# every time, because each of those sections renders an empty frame rather than
# raising. The symbol list was wrong in the same silent way: the fixture carried
# AMD, GOOG and INTC and lacked BA, JPM, KO and XOM, so four of the eight symbols
# `08_financial_features/03_structural_cross_instrument_features` names at its line
# 73 came back empty.
#
# What is still reduced, and the two axes are chosen so that no declared parameter
# lands outside the data:
#
#   - Contracts outside CHAIN_BAND. This clips SPREAD_BAND, which reaches to
#     (0.5, 1.5); the section keeps real wings either side of WING_BAND, which is
#     what it needs to have a subject, without the deep wings where quotes are
#     stale and which are most of the row count.
#   - Expirations beyond the nearest few. Both the near-dated chain and the far
#     section are kept, because 02/07 draws them separately and a "nearest N" rule
#     alone starves the far one: the sixth-nearest expiration is never more than
#     120 days out.
#
# Sessions are never reduced. 02/09 builds a daily series by picking one straddle
# per day inside a 25-35 day window, so a session stride would thin the series that
# notebook is about.

SP500_OPTIONS_EDA_DIR = SP500_DIR / "options_eda"
SP500_OPTIONS_EDA_YEARS = (2019, 2020)
SP500_OPTIONS_EDA_SYMBOLS = ("AAPL", "AMZN", "BA", "GOOGL", "JPM", "KO", "MSFT", "XOM")
# 02/07's CHAIN_BAND, SURFACE_MAX_DAYS and WING_BAND, named here so the budget and
# the notebook move together.
SP500_OPTIONS_EDA_BAND = (0.7, 1.3)
SP500_OPTIONS_EDA_FAR_DAYS = 180
SP500_OPTIONS_EDA_WING_BAND = (0.9, 1.1)
SP500_OPTIONS_EDA_NEAR_EXPIRATIONS = 6
SP500_OPTIONS_EDA_FAR_EXPIRATIONS = 2
# 02/09's DTE_WINDOW, and the symbol and year it demonstrates on. Asserted rather
# than assumed: the reduction is keyed on expiration, so it is exactly the kind of
# budget that can starve that notebook's selection while every other consumer still
# looks healthy.
SP500_OPTIONS_EDA_DTE_WINDOW = (25, 35)
SP500_OPTIONS_EDA_DEMO = ("AAPL", 2019)


def build_sp500_options_eda(source: Path, output: Path) -> list[Path]:
    """Reduce the production chains on moneyness and expiration, never on session.

    The ``year`` column production carries is dropped. ``load_sp500_options_eda``
    scans ``year=*.parquet`` with ``hive_partitioning=True``, so the partition key
    comes from the path; a column of the same name in the file is a second source
    for it.
    """
    out_dir = output / SP500_OPTIONS_EDA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = list(SP500_OPTIONS_EDA_SYMBOLS)
    low, high = SP500_OPTIONS_EDA_BAND
    written: list[Path] = []

    for year in SP500_OPTIONS_EDA_YEARS:
        src = source / SP500_OPTIONS_EDA_DIR / f"year={year}.parquet"
        if not src.exists():
            raise FileNotFoundError(
                f"{src} not found. Build it with data/equities/market/sp500/build_options_eda.py."
            )
        full = pl.scan_parquet(src).filter(pl.col("symbol").is_in(keep))
        if absent := sorted(
            set(keep) - set(full.select("symbol").unique().collect()["symbol"].to_list())
        ):
            raise ValueError(
                f"Production's {year} chain carries no rows for {absent}. The fixture "
                "declares all eight underlyings and a chapter-8 notebook asks for them "
                "by name, so a narrower source would reintroduce the gap this builder "
                "exists to close."
            )
        banded = full.filter((pl.col("strike") / pl.col("underlying_price")).is_between(low, high))

        def _nearest(frame: pl.LazyFrame, keep_ranks: int) -> pl.LazyFrame:
            return (
                frame.with_columns(
                    pl.col("expiration").rank("dense").over("symbol", "date").alias("_rank")
                )
                .filter(pl.col("_rank") <= keep_ranks)
                .drop("_rank")
            )

        near = _nearest(
            banded.filter(pl.col("days_to_maturity") <= SP500_OPTIONS_EDA_FAR_DAYS),
            SP500_OPTIONS_EDA_NEAR_EXPIRATIONS,
        )
        far = _nearest(
            banded.filter(pl.col("days_to_maturity") > SP500_OPTIONS_EDA_FAR_DAYS),
            SP500_OPTIONS_EDA_FAR_EXPIRATIONS,
        )
        reduced = (
            pl.concat([near, far])
            .collect()
            .drop("year", strict=False)
            .sort("date", "symbol", "expiration", "call_put", "strike")
        )
        _require_every_declared_band_reaches_data(reduced, year)
        written.append(_write_options_eda_part(reduced, out_dir, year))

    _require_constant_maturity_window(out_dir)
    return written


def _write_options_eda_part(frame: pl.DataFrame, out_dir: Path, year: int) -> Path:
    dst = out_dir / f"year={year}.parquet"
    frame.write_parquet(dst)
    print(
        f"    year={year}: {frame.height:,} rows, {frame['symbol'].n_unique()} symbols, "
        f"{frame['date'].n_unique()} sessions, {frame['expiration'].n_unique()} expirations "
        f"({dst.stat().st_size / 1e6:.1f} MB)"
    )
    return dst


def _require_every_declared_band_reaches_data(frame: pl.DataFrame, year: int) -> None:
    """Each of 02/07's declared bands has to have rows on both of its sides.

    This is the check the previous fixture would have failed. A band that lands
    entirely inside the data, or entirely outside it, renders an empty frame and
    the notebook reports a clean run either way.
    """
    moneyness = frame["strike"] / frame["underlying_price"]
    wing_low, wing_high = SP500_OPTIONS_EDA_WING_BAND
    checks = {
        f"below the {wing_low} wing": int((moneyness < wing_low).sum()),
        f"above the {wing_high} wing": int((moneyness > wing_high).sum()),
        f"beyond {SP500_OPTIONS_EDA_FAR_DAYS} days to maturity": int(
            (frame["days_to_maturity"] > SP500_OPTIONS_EDA_FAR_DAYS).sum()
        ),
    }
    if empty := sorted(name for name, count in checks.items() if not count):
        raise ValueError(
            f"{year}: the reduction leaves nothing {' or '.join(empty)}. "
            "02_financial_data_universe/07_sp500_options_eda declares WING_BAND and "
            "SURFACE_MAX_DAYS to separate those regions, and a section with nothing on "
            "one side of its own boundary renders empty rather than failing."
        )
    print(
        "      declared bands reached: "
        + ", ".join(f"{name} {count:,}" for name, count in checks.items())
    )


def _require_constant_maturity_window(out_dir: Path) -> None:
    """Every session of the 02/09 demo must offer a contract inside its DTE window.

    The notebook builds a daily series by picking, each day, the straddle closest to
    30 days and 50 delta. An expiration cap that leaves some sessions with nothing
    in the 25-35 day window does not fail the notebook - it silently shortens the
    series the whole notebook is about.
    """
    symbol, year = SP500_OPTIONS_EDA_DEMO
    low, high = SP500_OPTIONS_EDA_DTE_WINDOW
    frame = pl.read_parquet(out_dir / f"year={year}.parquet").filter(pl.col("symbol") == symbol)
    in_window = frame.filter(pl.col("days_to_maturity").is_between(low, high))
    sessions = frame["date"].n_unique()
    missing = sessions - in_window["date"].n_unique()
    if missing:
        raise ValueError(
            f"{missing} of {sessions} {symbol} {year} sessions carry no contract with "
            f"{low}-{high} days to maturity, which is the window "
            "02_financial_data_universe/09_options_continuous selects in."
        )
    print(f"    {symbol} {year}: all {sessions} sessions offer a {low}-{high} DTE contract")


# --- 13F bulk holdings --------------------------------------------------------
#
# The full-universe quarterly archive, distinct from the curated ten-institution
# panel above. `04_fundamental_alternative_data/10_institutional_holdings_13f`
# reads one quarter and ranks managers by reported value, so the reduction has to
# keep whole filings and has to keep more managers than the notebook ranks.
#
# 600 is not a new choice: filtering production to the 600 CIKs with the largest
# total reported value reproduces the fixture that was already there, row for row
# (1,327,852 rows, 743 filings, 21,388 securities). The rule had not been written
# down, which is the whole of what this declaration adds.

_13F_BULK_QUARTER = "2024Q3"
_13F_BULK_MANAGERS = 600
_13F_BULK = (
    Path("equities")
    / "positioning"
    / "13f"
    / "bulk"
    / _13F_BULK_QUARTER
    / "institutional_holdings.parquet"
)
# The notebook's declared TOP_N and CO_OWNERSHIP_UNIVERSE. Keeping more managers
# than it ranks is what makes its ranking the true one rather than an artifact of
# the fixture's size.
_13F_BULK_NOTEBOOK_TOP_N = 500


def build_13f_bulk_holdings(source: Path, output: Path) -> list[Path]:
    """Keep every position of the 600 managers reporting the most value."""
    src = source / _13F_BULK
    if not src.exists():
        raise FileNotFoundError(
            f"{_13F_BULK} not found at {src}. Fetch it with "
            f"data/equities/positioning/13f_download.py --mode bulk --quarters {_13F_BULK_QUARTER}."
        )
    full = pl.read_parquet(src)
    ranked = (
        full.group_by("cik")
        .agg(pl.col("value_thousands").sum().alias("reported_value"))
        .sort(["reported_value", "cik"], descending=[True, False])
        .head(_13F_BULK_MANAGERS)
    )
    if ranked.height < _13F_BULK_MANAGERS:
        raise ValueError(
            f"Production's {_13F_BULK_QUARTER} archive carries {ranked.height} managers, "
            f"fewer than the {_13F_BULK_MANAGERS} this fixture declares."
        )
    reduced = full.filter(pl.col("cik").is_in(ranked["cik"].implode()))

    # Two sections of the notebook depend on rows a naive top-N could drop, and
    # both would render as an empty table rather than an error.
    repeat_filers = (
        reduced.group_by("cik")
        .agg(pl.col("accession_no").n_unique().alias("filings"))
        .filter(pl.col("filings") > 1)
    )
    if not repeat_filers.height:
        raise ValueError(
            "No manager in the reduced set filed more than once in the window, so "
            "the 'one filing per manager' section has nothing to demonstrate."
        )
    implied = (
        reduced.filter(pl.col("shares") > 0)
        .with_columns((pl.col("value_thousands") / pl.col("shares")).alias("implied_price"))
        .group_by("accession_no")
        .agg(pl.col("implied_price").median().alias("median_implied_price"))
    )
    unbelievable = implied.filter(~pl.col("median_implied_price").is_between(1.0, 10_000.0))
    if not unbelievable.height:
        raise ValueError(
            "Every filing in the reduced set implies a believable price per share, so "
            "the screening section screens nothing out."
        )

    dst = output / _13F_BULK
    dst.parent.mkdir(parents=True, exist_ok=True)
    reduced.write_parquet(dst)
    print(
        f"    {_13F_BULK_QUARTER}: {reduced.height:,} of {full.height:,} positions, "
        f"{reduced['cik'].n_unique()} of {full['cik'].n_unique():,} managers "
        f"(notebook ranks {_13F_BULK_NOTEBOOK_TOP_N}), "
        f"{reduced['accession_no'].n_unique()} filings, {repeat_filers.height} of them "
        f"repeat filers, {unbelievable.height} failing the implied-price screen"
    )
    return [dst]


# --- Financial Phrasebank -----------------------------------------------------
#
# `sentences_allagree.parquet` is the subset of Malo et al. (2014) that every
# annotator labelled the same way. The fixture held 4,846 rows, which is the whole
# corpus at 50% agreement, under the filename that promises the 2,264 unanimous
# ones. `10_text_feature_engineering/04_bert_finetuning` fine-tunes on this file,
# so CI trained on 2,582 sentences the annotators disagreed about, each carrying
# whatever label the corpus assigned.
#
# `load_financial_phrasebank(agreement=...)` looks like it would have caught that
# and does not: it filters on an `agreement` column, and neither production's file
# nor the fixture's has one, so the argument has never selected anything. What
# makes production correct is that its file already holds the unanimous subset.
# Copying it is therefore the fix, and no reduction applies.

PHRASEBANK_DIR = Path("alternative") / "text" / "financial_phrasebank"
PHRASEBANK = PHRASEBANK_DIR / "sentences_allagree.parquet"
PHRASEBANK_ROWS = 2264
PHRASEBANK_LABELS = 3


def build_financial_phrasebank(source: Path, output: Path) -> list[Path]:
    """Copy the unanimous-agreement subset, verbatim."""
    dst = _copy_verbatim(
        source,
        output,
        PHRASEBANK,
        "the download script in data/alternative/text/README.md",
    )
    frame = pl.read_parquet(dst)
    if frame.height != PHRASEBANK_ROWS:
        raise ValueError(
            f"Production's {PHRASEBANK.name} holds {frame.height:,} rows, not the "
            f"{PHRASEBANK_ROWS:,} unanimous sentences the filename promises. The corpus "
            "at lower agreement levels is a different dataset and belongs in a "
            "differently named file."
        )
    if (labels := frame["label"].n_unique()) != PHRASEBANK_LABELS:
        raise ValueError(
            f"{PHRASEBANK.name} carries {labels} distinct labels, not {PHRASEBANK_LABELS}; "
            "a sentiment fixture missing a class trains and scores on a different task."
        )
    print(f"    {PHRASEBANK.name}: {frame.height:,} sentences, {labels} labels, copied verbatim")
    return [dst]


# --- Polymarket events --------------------------------------------------------
#
# 22 rows in production and three in the fixture, which is smaller than the panel
# `13_polymarket_prediction_markets` groups by category. Copied whole instead: the
# file is 5 KB.

POLYMARKET = Path("prediction_markets") / "polymarket_events.parquet"


def build_polymarket_events(source: Path, output: Path) -> list[Path]:
    """Copy the production Polymarket event panel, verbatim."""
    dst = _copy_verbatim(
        source,
        output,
        POLYMARKET,
        "data/prediction_markets/download.py --provider polymarket",
    )
    frame = pl.read_parquet(dst)
    print(
        f"    {POLYMARKET.name}: {frame.height} events, "
        f"{frame['symbol'].n_unique()} markets, copied verbatim"
    )
    return [dst]


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        name="etfs",
        description=(
            f"{len(ETF_SYMBOLS)} ETFs of the 100 production carries, adjusted and "
            "unadjusted, with the date column stored as a Date"
        ),
        build=build_etfs,
        owns=(ETF_UNIVERSE, ETF_UNIVERSE_UNADJUSTED, ETF_METADATA),
        budget={"symbols": list(ETF_SYMBOLS)},
        entities={
            ETF_UNIVERSE.as_posix(): ("symbol", len(ETF_SYMBOLS)),
            ETF_UNIVERSE_UNADJUSTED.as_posix(): ("symbol", len(ETF_SYMBOLS)),
        },
    ),
    Dataset(
        name="crypto",
        description=(
            f"{len(CRYPTO_SYMBOLS)} perpetual futures of the 19 production carries, "
            "hourly bars with their 8-hour premium index and funding rate"
        ),
        build=build_crypto,
        owns=(CRYPTO_PERPS, CRYPTO_PREMIUM, CRYPTO_FUNDING),
        budget={
            "symbols": list(CRYPTO_SYMBOLS),
            "bars_through": CRYPTO_PERPS_END.isoformat(),
        },
        entities={
            CRYPTO_PERPS.as_posix(): ("symbol", len(CRYPTO_SYMBOLS)),
            CRYPTO_PREMIUM.as_posix(): ("symbol", len(CRYPTO_SYMBOLS)),
            CRYPTO_FUNDING.as_posix(): ("symbol", len(CRYPTO_SYMBOLS)),
        },
    ),
    Dataset(
        name="fx",
        description=(
            f"the whole production FX panel, {len(FX_PAIRS)} major and cross pairs at "
            "4h and daily, with UTC-aware timestamps at both frequencies"
        ),
        build=build_fx,
        owns=(FX_4H, FX_DAILY),
        budget={"pairs": list(FX_PAIRS), "subsample": "none"},
        entities={
            FX_4H.as_posix(): ("symbol", len(FX_PAIRS)),
            FX_DAILY.as_posix(): ("symbol", len(FX_PAIRS)),
        },
    ),
    Dataset(
        name="us_equities",
        description=(
            f"{len(US_EQUITIES_TICKERS)} tickers of the 3,199 production carries, "
            "spanning the size distribution rather than the top of it, full date range"
        ),
        build=build_us_equities,
        owns=(US_EQUITIES,),
        budget={"tickers": list(US_EQUITIES_TICKERS)},
        entities={US_EQUITIES.as_posix(): ("ticker", len(US_EQUITIES_TICKERS))},
    ),
    Dataset(
        name="fnspid_news",
        description=(
            f"the production headlines for {len(FNSPID_TICKERS)} of the "
            f"{len(US_EQUITIES_TICKERS)} us_equities tickers, from {FNSPID_START.isoformat()} to "
            "that panel's last date, so the price join has dates to land on"
        ),
        build=build_fnspid,
        owns=(FNSPID_NEWS,),
        budget={
            "tickers": list(FNSPID_TICKERS),
            "news_from": FNSPID_START.isoformat(),
            "news_through": "the last date in equities/market/us_equities/us_equities.parquet",
        },
        entities={FNSPID_NEWS.as_posix(): ("Stock_symbol", len(FNSPID_TICKERS))},
    ),
    Dataset(
        name="cme_futures",
        description=(
            "the whole 30-product production daily panel, with the hourly bars "
            f"reduced to the {len(CME_HOURLY_PRODUCTS)} products a notebook reads"
        ),
        build=build_cme_futures,
        owns=(CME_DAILY, CME_HOURLY),
        budget={
            "daily": "none - setup.yaml declares 30 products and sizes the backtest for them",
            "hourly_products": list(CME_HOURLY_PRODUCTS),
        },
        entities={
            CME_DAILY.as_posix(): ("product", 30),
            CME_HOURLY.as_posix(): ("product", len(CME_HOURLY_PRODUCTS)),
        },
    ),
    Dataset(
        name="sp500_options",
        description=(
            f"{len(SP500_SYMBOLS)} S&P 500 underlyings across the share bars, the "
            "IV surface and the daily straddle panel, with the raw option chains "
            "reduced to the contracts that panel selects"
        ),
        build=build_sp500_options,
        # Named individually: options_eda/ sits in the same directory and is
        # produced elsewhere, so --clean must not take it.
        owns=(SP500_BARS, SP500_SURFACE, SP500_STRADDLES, SP500_RAW_CHAIN),
        budget={
            "symbols": list(SP500_SYMBOLS),
            "raw_chain": "contracts entered by options_straddles_daily, full lifecycle",
            "min_surface_symbols_per_date": SP500_MIN_SURFACE_SYMBOLS_PER_DATE,
        },
        # options_straddles_daily is deliberately not here: an underlying with no
        # straddle meeting the pairing rule drops out of that panel, so it is a
        # subset of the roster by construction rather than equal to it.
        entities={
            SP500_BARS.as_posix(): ("symbol", len(SP500_SYMBOLS)),
            SP500_SURFACE.as_posix(): ("symbol", len(SP500_SYMBOLS)),
        },
    ),
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
            "parts_per_year": NASDAQ100_MINUTE_PARTS_PER_YEAR,
        },
        entities={
            NASDAQ100_MINUTE_DIR.as_posix(): ("symbol", len(NASDAQ100_MINUTE_SYMBOLS)),
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
        entities={
            f"equities/firm_characteristics/firm_characteristics_{split}.parquet": (
                "symbol",
                FIRM_CHAR_MAX_ENTITIES,
            )
            for split, _, _ in FIRM_CHAR_SPLITS
        },
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
    Dataset(
        name="cot",
        description=(
            "the whole production CFTC Commitment of Traders directory, every "
            "product intact, so list_cot_products() enumerates under CI what it "
            "enumerates for a reader"
        ),
        build=build_cot,
        owns=(COT_DIR,),
        budget={"subsample": "none"},
    ),
    Dataset(
        name="sec_filing_references",
        description=(
            "the whole production sp100 10-K and 8-K reference panels, and the "
            f"sp500 10-Q panel cut to {len(SEC_10Q_SYMBOLS)} symbols"
        ),
        build=build_sec_filing_references,
        # Named individually: each sits in a reference/ directory alongside the
        # per-filing artifacts filings_download.py writes, which --clean must not take.
        owns=(SEC_10K_REFERENCE, SEC_8K_REFERENCE, SEC_10Q_REFERENCE),
        budget={"sp100_10k_8k": "none", "sp500_10q_symbols": list(SEC_10Q_SYMBOLS)},
        entities={SEC_10Q_REFERENCE.as_posix(): ("symbol", len(SEC_10Q_SYMBOLS))},
    ),
    Dataset(
        name="xbrl_fundamentals",
        description=(
            "the whole production XBRL quarterly fundamentals panel, which covers "
            "20 CIKs and is already fixture-sized"
        ),
        build=build_xbrl_fundamentals,
        # Named individually: filing_dates/ sits beside it and is xbrl_download.py's
        # own HTTP cache, not a fixture.
        owns=(XBRL_FUNDAMENTALS,),
        budget={"subsample": "none"},
    ),
    Dataset(
        name="form4",
        description=(
            "every Form 4 XML production carries, so the notebook that enumerates "
            "the ticker directories sees under CI what a reader sees"
        ),
        build=build_form4,
        owns=(FORM4_DIR,),
        budget={"subsample": "none"},
    ),
    Dataset(
        name="nasdaq_itch_messages",
        description=(
            f"the first {ITCH_ROWS_PER_SYMBOL} production ITCH add-order and trade "
            f"messages per symbol for {len(ITCH_SYMBOLS)} symbols, plus their stock "
            "directory rows, as a contiguous slice of the open rather than a sample"
        ),
        build=build_nasdaq_itch_messages,
        # Named individually, not as the messages/ directory: the other seven message
        # types there are synthetic and come from tests/generate_test_microstructure.py,
        # so --clean must not take them.
        owns=tuple(
            ITCH_MESSAGES / message_type / "part-000000.parquet"
            for message_type in ITCH_MESSAGE_TYPES
        ),
        budget={
            "symbols": list(ITCH_SYMBOLS),
            "rows_per_symbol": ITCH_ROWS_PER_SYMBOL,
            "message_types": list(ITCH_MESSAGE_TYPES),
        },
        entities={
            (ITCH_MESSAGES / message_type / "part-000000.parquet").as_posix(): (
                "stock",
                len(ITCH_SYMBOLS),
            )
            for message_type in ITCH_MESSAGE_TYPES
        },
    ),
    Dataset(
        name="individual_futures_es",
        description=(
            "the whole production ES individual-contract panel, small enough to "
            "ship intact and carrying every contract month the roll demonstrates"
        ),
        build=build_individual_futures_es,
        # Named individually: CL and NQ sit in the same directory and are synthetic.
        owns=(ES_INDIVIDUAL,),
        budget={"subsample": "none"},
    ),
    Dataset(
        name="fama_french_factors",
        description=(
            "every Fama-French factor parquet production carries, copied verbatim, "
            f"with the {len(FF_REQUIRED)} load_ff_factors() can name required"
        ),
        build=build_fama_french_factors,
        owns=(FF_DIR,),
        budget={"subsample": "none"},
    ),
    Dataset(
        name="aqr_factors",
        description=(
            "AQR's published factor parquets and their metadata, copied verbatim, "
            "without the source/ spreadsheets they were converted from"
        ),
        build=build_aqr_factors,
        owns=(AQR_DIR,),
        budget={"subsample": "none"},
    ),
    Dataset(
        name="fred_macro",
        description=(
            "the FRED macro panels production carries - aligned, raw, initial-release "
            "and their dictionaries and metadata - copied verbatim"
        ),
        build=build_fred_macro,
        owns=(MACRO_DIR,),
        budget={"subsample": "none"},
    ),
    Dataset(
        name="crypto_onchain",
        description="the DefiLlama TVL and CoinGecko on-chain series, copied verbatim",
        build=build_crypto_onchain,
        owns=(ONCHAIN_DIR,),
        budget={"subsample": "none"},
    ),
    Dataset(
        name="sp500_options_eda",
        description=(
            f"the {len(SP500_OPTIONS_EDA_SYMBOLS)} production underlyings over "
            f"{SP500_OPTIONS_EDA_YEARS[0]}-{SP500_OPTIONS_EDA_YEARS[-1]}, every session, "
            f"moneyness {SP500_OPTIONS_EDA_BAND[0]}-{SP500_OPTIONS_EDA_BAND[1]}, the "
            f"nearest {SP500_OPTIONS_EDA_NEAR_EXPIRATIONS} expirations plus "
            f"{SP500_OPTIONS_EDA_FAR_EXPIRATIONS} beyond "
            f"{SP500_OPTIONS_EDA_FAR_DAYS} days"
        ),
        build=build_sp500_options_eda,
        owns=tuple(
            SP500_OPTIONS_EDA_DIR / f"year={year}.parquet" for year in SP500_OPTIONS_EDA_YEARS
        ),
        budget={
            "symbols": list(SP500_OPTIONS_EDA_SYMBOLS),
            "sessions": "none - every trading day is kept",
            "moneyness_band": list(SP500_OPTIONS_EDA_BAND),
            "near_expirations_per_session": SP500_OPTIONS_EDA_NEAR_EXPIRATIONS,
            "far_expirations_per_session": SP500_OPTIONS_EDA_FAR_EXPIRATIONS,
            "far_threshold_days": SP500_OPTIONS_EDA_FAR_DAYS,
            "dte_window_required": list(SP500_OPTIONS_EDA_DTE_WINDOW),
            "clips": (
                "07_sp500_options_eda's SPREAD_BAND (0.5, 1.5) is clipped to the "
                "moneyness band; the deep wings are stale quotes and most of the rows"
            ),
        },
        entities={
            (SP500_OPTIONS_EDA_DIR / f"year={year}.parquet").as_posix(): (
                "symbol",
                len(SP500_OPTIONS_EDA_SYMBOLS),
            )
            for year in SP500_OPTIONS_EDA_YEARS
        },
    ),
    Dataset(
        name="institutional_holdings_13f_bulk",
        description=(
            f"every position of the {_13F_BULK_MANAGERS} managers reporting the most "
            f"value in the {_13F_BULK_QUARTER} filing window, whole filings only"
        ),
        build=build_13f_bulk_holdings,
        owns=(_13F_BULK,),
        budget={
            "quarter": _13F_BULK_QUARTER,
            "managers": _13F_BULK_MANAGERS,
            "notebook_ranks": _13F_BULK_NOTEBOOK_TOP_N,
            "filings": "whole - a manager keeps every position it reported",
        },
        entities={_13F_BULK.as_posix(): ("cik", _13F_BULK_MANAGERS)},
    ),
    Dataset(
        name="financial_phrasebank",
        description=(
            f"the {PHRASEBANK_ROWS:,} sentences every annotator labelled the same way, "
            "copied verbatim - which is the whole of what the filename promises"
        ),
        build=build_financial_phrasebank,
        owns=(PHRASEBANK,),
        budget={"subsample": "none", "agreement": "100% (unanimous)"},
    ),
    Dataset(
        name="polymarket_events",
        description="the whole production Polymarket event panel, copied verbatim",
        build=build_polymarket_events,
        owns=(POLYMARKET,),
        budget={"subsample": "none"},
    ),
)

DATASETS_BY_NAME = {dataset.name: dataset for dataset in DATASETS}


# Manifest entries that describe a fixture another dataset already owns, and are
# dropped rather than declared. `sp500_daily` describes
# equities/market/sp500/daily_bars.parquet, which build_sp500_options writes;
# `nasdaq100` describes the minute bars build_nasdaq100_minute_bars writes. Both
# still carried their pre-reorganization paths and their pre-widening budgets - 20
# and 3 symbols against the 30 and 12 on disk - so a reader consulting the manifest
# got a wrong answer about a fixture that does have a builder.
SUPERSEDED_SUBSETS = {
    "sp500_daily": "sp500_options",
    "nasdaq100": "nasdaq100_minute_bars",
}


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


def declared_subsets() -> dict[str, dict]:
    """What ``manifest.json``'s ``subsets`` block has to say, from the declarations.

    One entry per registered ``Dataset`` and nothing else. This is the comparison
    that would have fired the day #652 landed: the manifest still said sp500_options
    kept 3 "most liquid" underlyings while the builder declared 30 named ones, and
    nothing noticed until a case study spent a day on five notebooks that were
    failing on the stale fixture rather than on anything in the notebooks.
    """
    return {
        dataset.name: {**dataset.budget, "description": dataset.description} for dataset in DATASETS
    }


def reconcile_manifest(output: Path) -> tuple[Path, dict]:
    """Rewrite ``manifest.json`` so it describes the fixture set that is on disk.

    Two things go stale on their own and nothing was checking either. The ``subsets``
    block drifts from the declarations whenever a builder changes without a
    regeneration. The ``files`` block keeps every path it ever recorded, because
    :func:`write_manifest` only drops entries under the dataset being rebuilt - so a
    reorganization of the tree (``crypto/`` to ``crypto/market/``, and the same for
    equities, etfs, fx and futures) left the manifest listing files that are not
    there, which is the same wrong answer as a stale budget.

    Sizes are re-read for the paths that do exist; a path that does not is dropped.
    Every file under a declared dataset's own paths is then recorded, so the block
    describes the declared fixture set completely rather than whatever the last
    per-dataset run happened to write. Files under no declaration are kept if they
    exist and never added, because what belongs in the fixture set is a declaration
    and not whatever is in the directory.
    """
    manifest_path = output / "manifest.json"
    manifest: dict = {"version": "1", "subsets": {}, "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    manifest["subsets"] = declared_subsets()

    def record(rel: str, path: Path) -> None:
        size = path.stat().st_size
        files[rel] = {"size_bytes": size, "size_mb": round(size / 1e6, 2)}

    files: dict[str, dict] = {}
    for rel in sorted(manifest.get("files", {})):
        path = output / rel
        if path.is_file():
            record(rel, path)
    dropped = sorted(set(manifest.get("files", {})) - set(files))

    declared_paths = [owned for dataset in DATASETS for owned in dataset.owns]
    added = []
    for declared in declared_paths:
        root = output / declared
        found = [root] if root.is_file() else sorted(f for f in root.rglob("*") if f.is_file())
        for path in found:
            rel = path.relative_to(output).as_posix()
            if rel not in files:
                added.append(rel)
            record(rel, path)

    manifest["files"] = dict(sorted(files.items()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, {"kept": len(files), "dropped": dropped, "added": sorted(added)}


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
        help="Production data root to subsample from (e.g. ~/ml4t/code/data). "
        "Required unless --reconcile-manifest, which reads no production data.",
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
    parser.add_argument(
        "--reconcile-manifest",
        action="store_true",
        help=(
            "Rewrite data/manifest.json from the declarations and the files on disk, "
            "building nothing. --source is not read."
        ),
    )
    args = parser.parse_args()

    if args.reconcile_manifest:
        if args.dry_run:
            parser.error("--dry-run and --reconcile-manifest are mutually exclusive")
        output = args.output.expanduser().resolve()
        manifest_path, report = reconcile_manifest(output)
        print(f"Manifest: {manifest_path}")
        print(f"  subsets: {', '.join(sorted(declared_subsets()))}")
        print(
            f"  files: {report['kept']} recorded, {len(report['dropped'])} dropped as "
            f"not on disk, {len(report['added'])} added from a declared path"
        )
        for rel in report["dropped"]:
            print(f"  dropped, not on disk: {rel}")
        return 0

    if args.source is None:
        parser.error("--source is required unless --reconcile-manifest is given")
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
