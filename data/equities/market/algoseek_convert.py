"""Convert the AlgoSeek CSV archives to the parquet layout the loaders read.

AlgoSeek publishes three datasets for this book at https://algoseek.com/ml-for-trading/,
as plain Dropbox links with no signup and no account. Two of them are CSV, and every
loader in ``data/equities/loader.py`` reads zstd parquet under a Hive layout, so a
reader who downloads those two cannot run anything until this script has run.

    nasdaq-100-constituents-taq-ext.zip   5.9 GB    NASDAQ-100 extended minute bars
                                                    505 days, 2020-01-02 to 2021-12-31
    options_daily_greeks_sp500.zip       14.1 GB    S&P 500 daily option chains + Greeks
                                                    1,259 days 2017-2021, 634 symbols

The third, ``symbol=AAPL.zip`` (67 MB, NASDAQ-100 trade-and-quote ticks), is already
parquet in the layout ``load_nasdaq100_taq()`` scans. It needs no conversion, only
unzipping into place. Name the members: Dropbox writes a stray root entry into that
archive, and unzipping without ``"*.parquet"`` warns and exits 2:

    unzip -q "symbol=AAPL.zip" "*.parquet" \\
        -d "$ML4T_DATA_PATH/equities/market/microstructure/trade_and_quotes/symbol=AAPL"

The fourth dataset the book uses, the S&P 500 daily bars, ships inside this repository
at ``data/equities/market/sp500/daily_bars.parquet``; nothing to download or convert.

Run from the repository root, once per archive:

    uv run python data/equities/market/algoseek_convert.py \\
        --dataset nasdaq100-minute-bars --source ~/Downloads/nasdaq-100-constituents-taq-ext.zip

    uv run python data/equities/market/algoseek_convert.py \\
        --dataset sp500-options --source ~/Downloads/options_daily_greeks_sp500.zip

``--source`` takes either the ``.zip`` or a directory you have already extracted.
Extracting first is faster for the options archive, which holds **1,275,314** gzip
members: reading them out of the zip pays the archive's index on every open, and
unpacking that many small files is slow on any filesystem and slower on Windows.

Output, under ``$ML4T_DATA_PATH``:

    equities/market/nasdaq100/minute_bars/year=YYYY/month=MM.parquet
    equities/market/sp500/options/year=YYYY/YYYYMMDD.parquet

The options output is the *raw chain*. It is the input to the three build scripts
that produce what the notebooks actually load, and they run after this one:

    uv run python data/equities/market/sp500/build_options_eda.py
    uv run python data/equities/market/sp500/build_options_straddles_raw.py
    uv run python data/equities/market/sp500/materialize_options.py

Both conversions resume: an output partition that already exists is skipped, so an
interrupted run continues where it stopped. Pass ``--force`` to rebuild.
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import time
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import polars as pl

from utils.downloading import print_section, resolve_data_dir

# --------------------------------------------------------------------------------
# NASDAQ-100 extended minute bars
# --------------------------------------------------------------------------------

# AlgoSeek ships 90 columns; the loaders read 62 of them. The dropped ones are
# either counts whose volume twin is already kept (TradeAtBidCount), FINRA/PRP
# splits the book does not use, or the SecId the canonical schema replaces with
# `symbol`. Left is the CSV header, right is the canonical name.
NASDAQ100_COLUMNS: dict[str, str] = {
    "Date": "date",
    "Ticker": "symbol",
    "TimeBarStart": "time",
    "OpenBarTime": "open_bar_time",
    "OpenBidPrice": "open_bid_price",
    "OpenBidSize": "open_bid_size",
    "OpenAskPrice": "open_ask_price",
    "OpenAskSize": "open_ask_size",
    "FirstTradeTime": "first_trade_time",
    "FirstTradePrice": "first_trade_price",
    "FirstTradeSize": "first_trade_size",
    "HighBidTime": "high_bid_time",
    "HighBidPrice": "high_bid_price",
    "HighBidSize": "high_bid_size",
    "HighAskTime": "high_ask_time",
    "HighAskPrice": "high_ask_price",
    "HighAskSize": "high_ask_size",
    "HighTradeTime": "high_trade_time",
    "HighTradePrice": "high_trade_price",
    "HighTradeSize": "high_trade_size",
    "LowBidTime": "low_bid_time",
    "LowBidPrice": "low_bid_price",
    "LowBidSize": "low_bid_size",
    "LowAskTime": "low_ask_time",
    "LowAskPrice": "low_ask_price",
    "LowAskSize": "low_ask_size",
    "LowTradeTime": "low_trade_time",
    "LowTradePrice": "low_trade_price",
    "LowTradeSize": "low_trade_size",
    "CloseBarTime": "close_bar_time",
    "CloseBidPrice": "close_bid_price",
    "CloseBidSize": "close_bid_size",
    "CloseAskPrice": "close_ask_price",
    "CloseAskSize": "close_ask_size",
    "LastTradeTime": "last_trade_time",
    "LastTradePrice": "last_trade_price",
    "LastTradeSize": "last_trade_size",
    "MinSpread": "min_spread",
    "MaxSpread": "max_spread",
    "CancelSize": "cancel_size",
    "VolumeWeightPrice": "vwap",
    "NBBOQuoteCount": "nbbo_quote_count",
    "TradeAtBid": "trade_at_bid",
    "TradeAtBidMid": "trade_at_bid_mid",
    "TradeAtMid": "trade_at_mid",
    "TradeAtMidAsk": "trade_at_mid_ask",
    "TradeAtAsk": "trade_at_ask",
    "TradeAtCrossOrLocked": "trade_at_cross",
    "Volume": "volume",
    "TotalTrades": "total_trades",
    "FinraVolume": "finra_volume",
    "FinraVolumeWeightPrice": "finra_vwap",
    "UptickVolume": "uptick_volume",
    "DowntickVolume": "downtick_volume",
    "RepeatUptickVolume": "repeat_uptick_volume",
    "RepeatDowntickVolume": "repeat_downtick_volume",
    "UnknownTickVolume": "unknown_tick_volume",
    "TradeToMidVolWeight": "trade_to_mid_vol_weight",
    "TradeToMidVolWeightRelative": "trade_to_mid_vol_weight_rel",
    "TimeWeightBid": "time_weight_bid",
    "TimeWeightAsk": "time_weight_ask",
}

# String columns keep AlgoSeek's nanosecond time-of-day text verbatim ("04:00:57.069651992"),
# which no datetime type round-trips. Everything else is numeric.
_NASDAQ100_STRING = {"Ticker", "TimeBarStart"} | {
    c for c in NASDAQ100_COLUMNS if c.endswith("Time")
}
_NASDAQ100_INT = {
    "OpenBidSize",
    "OpenAskSize",
    "FirstTradeSize",
    "HighBidSize",
    "HighAskSize",
    "HighTradeSize",
    "LowBidSize",
    "LowAskSize",
    "LowTradeSize",
    "CloseBidSize",
    "CloseAskSize",
    "LastTradeSize",
    "CancelSize",
    "NBBOQuoteCount",
    "TradeAtBid",
    "TradeAtBidMid",
    "TradeAtMid",
    "TradeAtMidAsk",
    "TradeAtAsk",
    "TradeAtCrossOrLocked",
    "Volume",
    "TotalTrades",
    "FinraVolume",
    "UptickVolume",
    "DowntickVolume",
    "RepeatUptickVolume",
    "RepeatDowntickVolume",
    "UnknownTickVolume",
}


def _nasdaq100_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {}
    for csv_name in NASDAQ100_COLUMNS:
        if csv_name == "Date":
            schema[csv_name] = pl.Int64
        elif csv_name in _NASDAQ100_STRING:
            schema[csv_name] = pl.String
        elif csv_name in _NASDAQ100_INT:
            schema[csv_name] = pl.Int64
        else:
            schema[csv_name] = pl.Float64
    return schema


def parse_nasdaq100_csv(payload: bytes) -> pl.DataFrame:
    """One symbol-day of extended minute bars, in canonical schema."""
    df = pl.read_csv(
        payload,
        columns=list(NASDAQ100_COLUMNS),
        schema_overrides=_nasdaq100_schema(),
        null_values=[""],
    )
    return (
        df.rename(NASDAQ100_COLUMNS)
        .with_columns(
            pl.col("date").cast(pl.String).str.to_date("%Y%m%d"),
        )
        .with_columns(
            # AlgoSeek's bar label is the minute the bar opens, which is what the
            # loaders filter regular hours on.
            (pl.col("date").cast(pl.String) + pl.lit(" ") + pl.col("time"))
            .str.to_datetime("%Y-%m-%d %H:%M")
            .alias("timestamp"),
        )
        .select(list(NASDAQ100_COLUMNS.values()) + ["timestamp"])
    )


# --------------------------------------------------------------------------------
# S&P 500 daily option chains with Greeks
# --------------------------------------------------------------------------------

# 1:1 onto the option-chain schema the sp500_options case study reads. AlgoSeek's
# three timestamp columns (UnderLastMidTime, LastBidTime, LastAskTime) are dropped:
# nothing downstream reads them and they double the width of the largest dataset.
SP500_OPTIONS_COLUMNS: dict[str, str] = {
    "TradeDate": "date",
    "Ticker": "symbol",
    "CallPut": "call_put",
    "OptionStyle": "option_style",
    "Strike": "strike",
    "Expiration": "expiration",
    "YearsToMaturity": "years_to_maturity",
    "DaysToMaturity": "days_to_maturity",
    "UnderLastMidPrice": "underlying_price",
    "LastBidPrice": "bid",
    "LastAskPrice": "ask",
    "LastMidPrice": "mid_price",
    "MidImpliedVol": "implied_vol",
    "MidTheoPrice": "theo_price",
    "MidDelta": "delta",
    "MidGamma": "gamma",
    "MidTheta": "theta",
    "MidVega": "vega",
    "MidRho": "rho",
    "ImpliedVolConvergence": "iv_convergence",
}

_SP500_OPTIONS_SCHEMA: dict[str, pl.DataType] = {
    "TradeDate": pl.Int64,
    "Ticker": pl.String,
    "CallPut": pl.String,
    "OptionStyle": pl.String,
    "Strike": pl.Float64,
    "Expiration": pl.Int64,
    "YearsToMaturity": pl.Float64,
    "DaysToMaturity": pl.Int32,
    "UnderLastMidPrice": pl.Float64,
    "LastBidPrice": pl.Float64,
    "LastAskPrice": pl.Float64,
    "LastMidPrice": pl.Float64,
    "MidImpliedVol": pl.Float64,
    "MidTheoPrice": pl.Float64,
    "MidDelta": pl.Float64,
    "MidGamma": pl.Float64,
    "MidTheta": pl.Float64,
    "MidVega": pl.Float64,
    "MidRho": pl.Float64,
    "ImpliedVolConvergence": pl.String,
}


def parse_sp500_options_csv(payload: bytes) -> pl.DataFrame:
    """One symbol-day of option chain, in canonical schema."""
    df = pl.read_csv(
        payload,
        columns=list(SP500_OPTIONS_COLUMNS),
        schema_overrides=_SP500_OPTIONS_SCHEMA,
        null_values=[""],
    )
    return (
        df.rename(SP500_OPTIONS_COLUMNS)
        .with_columns(
            pl.col("date").cast(pl.String).str.to_date("%Y%m%d"),
            pl.col("expiration").cast(pl.String).str.to_date("%Y%m%d"),
        )
        # read_csv keeps the file's order, which interleaves the time columns and so
        # puts mid before ask. Select explicitly so the layout matches the delivery.
        .select(list(SP500_OPTIONS_COLUMNS.values()))
    )


# --------------------------------------------------------------------------------
# Reading a day out of either archive shape
# --------------------------------------------------------------------------------

_DAY_ZIP = re.compile(r"(?:^|/)(\d{4})/(\d{8})\.zip$")
_OPTIONS_MEMBER = re.compile(r"(?:^|/)(\d{4})/(\d{8})/([^/]+)\.csv\.gz$")
_NASDAQ_MEMBER = re.compile(r"(?:^|/)(\d{8})/[A-Z0-9]/([^/]+)\.csv$", re.IGNORECASE)


class DaySource:
    """The per-day members of one archive, whether zipped or already extracted.

    ``days()`` yields ``(YYYYMMDD, reader)``, and calling ``reader()`` returns the
    CSV payloads for that day. Reading is deferred so a resumed run pays nothing
    for the days it skips.
    """

    def __init__(self, path: Path, dataset: str) -> None:
        self.path = path
        self.dataset = dataset

    def days(self) -> Iterator[tuple[str, object]]:
        if self.path.is_dir():
            yield from self._days_from_directory()
        elif zipfile.is_zipfile(self.path):
            yield from self._days_from_zip()
        else:
            msg = f"--source is neither a directory nor a zip archive: {self.path}"
            raise SystemExit(msg)

    # -- extracted directory ------------------------------------------------

    def _days_from_directory(self) -> Iterator[tuple[str, object]]:
        # Extracting the NASDAQ-100 archive gives per-day zips, not CSVs — its outer
        # archive is a zip of zips. A reader who unpacks one level and points at the
        # result has done nothing wrong, so read both shapes.
        day_zips = sorted(
            (p for p in self.path.rglob("*.zip") if "__MACOSX" not in p.parts and _day_of(p)),
            key=lambda p: _day_of(p),
        )
        if day_zips:
            for path in day_zips:
                yield _day_of(path), lambda path=path: _read_day_zip(path)
            return

        suffix = "*.csv.gz" if self.dataset == "sp500-options" else "*.csv"
        by_day: dict[str, list[Path]] = defaultdict(list)
        for file in self.path.rglob(suffix):
            # macOS zips carry a __MACOSX shadow tree of resource forks.
            if "__MACOSX" in file.parts:
                continue
            day = _day_of(file)
            if day:
                by_day[day].append(file)
        for day in sorted(by_day):
            files = sorted(by_day[day])
            yield day, lambda files=files: [_read_file(f) for f in files]

    # -- zip archive --------------------------------------------------------

    def _days_from_zip(self) -> Iterator[tuple[str, object]]:
        archive = zipfile.ZipFile(self.path)
        # The options archive AlgoSeek serves today was re-zipped on a macOS machine:
        # it nests everything under one top-level folder and carries explicit directory
        # entries, a __MACOSX shadow tree and .DS_Store files. A directory entry sits in
        # a day's folder and reads back as zero bytes, which read_csv rejects, so match
        # the data members by suffix rather than taking every name under a day.
        suffix = ".csv.gz" if self.dataset == "sp500-options" else ".csv"
        names = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and "__MACOSX" not in info.filename
        ]

        # NASDAQ-100: one inner zip per day. Options: members grouped by day directory.
        inner_zips = [n for n in names if _DAY_ZIP.search(n)]
        if inner_zips:
            for name in sorted(inner_zips, key=lambda n: _DAY_ZIP.search(n).group(2)):
                day = _DAY_ZIP.search(name).group(2)
                yield day, lambda name=name: _read_inner_zip(archive, name)
            return

        by_day: dict[str, list[str]] = defaultdict(list)
        for name in names:
            if not name.endswith(suffix):
                continue
            day = _day_of(Path(name))
            if day:
                by_day[day].append(name)
        for day in sorted(by_day):
            members = sorted(by_day[day])
            yield day, lambda members=members: [_read_member(archive, m) for m in members]


def _day_of(path: Path) -> str | None:
    """The YYYYMMDD a member belongs to, from its own name or a parent directory."""
    for part in reversed(path.parts):
        stem = part.split(".", 1)[0]
        if len(stem) == 8 and stem.isdigit():
            return stem
    return None


def _read_file(path: Path) -> bytes:
    payload = path.read_bytes()
    return gzip.decompress(payload) if path.suffix == ".gz" else payload


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    payload = archive.read(name)
    return gzip.decompress(payload) if name.endswith(".gz") else payload


def _members_of_day_zip(inner: zipfile.ZipFile) -> list[bytes]:
    return [
        inner.read(m)
        for m in sorted(inner.namelist())
        if _NASDAQ_MEMBER.search(m) and "__MACOSX" not in m
    ]


def _read_inner_zip(archive: zipfile.ZipFile, name: str) -> list[bytes]:
    with zipfile.ZipFile(io.BytesIO(archive.read(name))) as inner:
        return _members_of_day_zip(inner)


def _read_day_zip(path: Path) -> list[bytes]:
    with zipfile.ZipFile(path) as inner:
        return _members_of_day_zip(inner)


# --------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def _parse_day(payloads: list[bytes], parse, workers: int) -> pl.DataFrame | None:
    """Parse one day's CSVs. Decompression and CSV parsing both release the GIL,
    so threads are what make a 504-file day tolerable."""
    frames: list[pl.DataFrame] = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = [f for f in pool.map(parse, payloads) if f.height]
    else:
        frames = [f for f in (parse(p) for p in payloads) if f.height]
    if not frames:
        return None
    return pl.concat(frames, how="vertical")


def _write(df: pl.DataFrame, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=9, statistics=True)
    tmp.replace(path)
    return path.stat().st_size / 1024 / 1024


def convert_nasdaq100(source: Path, out_root: Path, workers: int, force: bool) -> int:
    """One parquet per month, matching the layout load_nasdaq100_bars() scans."""
    out_dir = out_root / "equities" / "market" / "nasdaq100" / "minute_bars"
    days = list(DaySource(source, "nasdaq100-minute-bars").days())
    if not days:
        _log(f"No day archives found under {source}")
        return 1

    by_month: dict[tuple[str, str], list[tuple[str, object]]] = defaultdict(list)
    for day, reader in days:
        by_month[(day[:4], day[4:6])].append((day, reader))

    _log(f"{len(days)} days across {len(by_month)} months -> {out_dir}")
    total_rows = 0
    for (year, month), entries in sorted(by_month.items()):
        out_path = out_dir / f"year={year}" / f"month={month}.parquet"
        if out_path.exists() and not force:
            _log(f"{year}-{month}: already converted, skipping")
            continue

        t0 = time.time()
        frames = []
        # One line per day. A month is ~21 days of ~504 symbol files each, and nothing is
        # written until the whole month is concatenated, so without this the longest phase of
        # the run produces no output and creates no directory - reported as the converter
        # "not working" when it was working the whole time.
        for n, (day, reader) in enumerate(entries, start=1):
            df = _parse_day(reader(), parse_nasdaq100_csv, workers)
            if df is not None:
                frames.append(df)
                _log(f"  {day} ({n}/{len(entries)}): {df.height:,} rows")
            else:
                _log(f"  {day} ({n}/{len(entries)}): no rows")
        if not frames:
            _log(f"{year}-{month}: nothing to write")
            continue

        _log(f"{year}-{month}: concatenating and sorting {len(frames)} days")
        month_df = pl.concat(frames, how="vertical").sort("symbol", "timestamp")
        del frames
        size_mb = _write(month_df, out_path)
        total_rows += month_df.height
        _log(
            f"{year}-{month}: {month_df.height:,} rows, {month_df['symbol'].n_unique()} symbols, "
            f"{size_mb:.0f} MB, {time.time() - t0:.0f}s"
        )

    _log(f"NASDAQ-100 minute bars: {total_rows:,} rows written")
    return 0


def convert_sp500_options(source: Path, out_root: Path, workers: int, force: bool) -> int:
    """One parquet per trading day, under the year the build scripts glob."""
    out_dir = out_root / "equities" / "market" / "sp500" / "options"
    days = list(DaySource(source, "sp500-options").days())
    if not days:
        _log(f"No daily option chains found under {source}")
        return 1

    _log(f"{len(days)} days -> {out_dir}")
    total_rows = 0
    converted = 0
    t_start = time.time()
    for index, (day, reader) in enumerate(days, start=1):
        out_path = out_dir / f"year={day[:4]}" / f"{day}.parquet"
        if out_path.exists() and not force:
            continue

        t0 = time.time()
        df = _parse_day(reader(), parse_sp500_options_csv, workers)
        if df is None:
            _log(f"{day}: no rows")
            continue
        df = df.sort("symbol", "expiration", "call_put", "strike")
        size_mb = _write(df, out_path)
        total_rows += df.height
        converted += 1
        rate = index / max(time.time() - t_start, 1e-9)
        remaining = (len(days) - index) / rate
        _log(
            f"{day} ({index}/{len(days)}): {df.height:,} rows, {df['symbol'].n_unique()} symbols, "
            f"{size_mb:.0f} MB, {time.time() - t0:.0f}s, ~{remaining / 60:.0f} min left"
        )

    _log(f"S&P 500 option chains: {converted} days converted, {total_rows:,} rows written")
    if converted:
        _log("Next, from the repository root:")
        _log("  uv run python data/equities/market/sp500/build_options_eda.py")
        _log("  uv run python data/equities/market/sp500/build_options_straddles_raw.py")
        _log("  uv run python data/equities/market/sp500/materialize_options.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert the published AlgoSeek CSV archives to the parquet the loaders read",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nasdaq100-minute-bars", "sp500-options"],
        help="Which archive to convert",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="The downloaded .zip, or a directory you have already extracted it to",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data storage location (default: $ML4T_DATA_PATH or repo/data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Threads used to parse one day's files (default: 8)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild partitions that already exist instead of skipping them",
    )
    args = parser.parse_args()

    source = args.source.expanduser()
    if not source.exists():
        _log(f"--source does not exist: {source}")
        return 1

    out_root = resolve_data_dir(args.data_path)
    print_section(f"ALGOSEEK CONVERSION - {args.dataset}")
    _log(f"Source: {source}")
    _log(f"Output: {out_root}")

    if args.dataset == "nasdaq100-minute-bars":
        return convert_nasdaq100(source, out_root, args.workers, args.force)
    return convert_sp500_options(source, out_root, args.workers, args.force)


if __name__ == "__main__":
    sys.exit(main())
