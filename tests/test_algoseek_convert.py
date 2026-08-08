"""Tests for the AlgoSeek CSV to parquet converter.

AlgoSeek publishes both of its book datasets as CSV; every loader in
``data/equities/loader.py`` reads parquet. ``algoseek_convert.py`` is the only
thing between a reader's download and a working notebook, so what it has to get
right is the *schema contract* — the exact column names, order and dtypes the
loaders project out of the result. A converter that runs but renames one column
fails much later, inside a notebook, with an error naming neither.

The mapping was established against the real archives by converting 2020-03 and
comparing to the delivery AlgoSeek shipped in April: all 62 columns, all dtypes,
and every value in the shipped rows matched. These tests pin that outcome
against fixtures so a later edit cannot quietly break it.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

import data.download_all as da

CONVERT_PY = Path(da.__file__).parent / "equities" / "market" / "algoseek_convert.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("algoseek_convert", CONVERT_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def convert():
    return _load_module()


# --------------------------------------------------------------------------------
# Fixtures shaped like the real archives
# --------------------------------------------------------------------------------

# Columns AlgoSeek ships that the loaders do not read. Interleaved with the kept
# ones in the real file, which is why the converter selects by name.
_DROPPED = [
    "SecId",
    "TotalVolumeWeightPrice",
    "VolumeWeightPriceExcludePRP",
    "TradeAtBidCount",
    "TotalQuoteCount",
    "OddLotTradeCount",
    "RetailTRFBuySize",
]


def _nasdaq100_csv(convert, symbol: str = "AAPL", day: str = "20200313") -> bytes:
    """One symbol-day, with the dropped columns interleaved and blanks for nulls."""
    kept = list(convert.NASDAQ100_COLUMNS)
    header = kept[:5] + _DROPPED[:3] + kept[5:] + _DROPPED[3:]

    def row(minute: str, price: float) -> list[str]:
        values = {name: "" for name in header}
        values.update(
            {
                "Date": day,
                "Ticker": symbol,
                "TimeBarStart": minute,
                "OpenBarTime": f"{minute}:00.000000000",
                "OpenBidPrice": str(price),
                "OpenBidSize": "200",
                "FirstTradePrice": str(price + 0.5),
                "FirstTradeSize": "1785",
                "LastTradePrice": str(price + 1.0),
                "Volume": "7486",
                "TotalTrades": "90",
                "TradeAtBid": "668",
                "TradeAtCrossOrLocked": "0",
                "VolumeWeightPrice": str(price + 0.25),
                "NBBOQuoteCount": "153",
            }
        )
        return [values[name] for name in header]

    lines = [",".join(header), ",".join(row("04:00", 255.1)), ",".join(row("04:01", 259.2))]
    return ("\n".join(lines) + "\n").encode()


def _options_csv(convert, symbol: str = "AAPL", day: str = "20200313") -> bytes:
    """One symbol-day of chain, in AlgoSeek's column order.

    The time columns sit between the price columns in the real file, which is why
    the converter cannot rely on read_csv preserving a usable order.
    """
    header = [
        "TradeDate",
        "Ticker",
        "CallPut",
        "OptionStyle",
        "Strike",
        "Expiration",
        "YearsToMaturity",
        "DaysToMaturity",
        "UnderLastMidPrice",
        "UnderLastMidTime",
        "LastBidPrice",
        "LastBidTime",
        "LastMidPrice",
        "LastAskPrice",
        "LastAskTime",
        "MidImpliedVol",
        "MidTheoPrice",
        "MidDelta",
        "MidGamma",
        "MidTheta",
        "MidVega",
        "MidRho",
        "ImpliedVolConvergence",
    ]
    row = [
        day,
        symbol,
        "C",
        "A",
        "250.0",
        "20200417",
        "0.0959",
        "35",
        "255.5",
        "15:59",
        "12.5",
        "15:59:59.592",
        "13.0",
        "13.5",
        "15:59:59.896",
        "0.55",
        "12.9",
        "0.52",
        "0.01",
        "-0.09",
        "0.28",
        "0.11",
        "Converged",
    ]
    return (",".join(header) + "\n" + ",".join(row) + "\n").encode()


# --------------------------------------------------------------------------------
# The schema contract
# --------------------------------------------------------------------------------

# What load_nasdaq100_bars() projects out of the result, in order. Any drift here
# is a reader whose notebook fails on a missing column.
NASDAQ100_EXPECTED = [
    "date",
    "symbol",
    "time",
    "open_bar_time",
    "open_bid_price",
    "open_bid_size",
    "open_ask_price",
    "open_ask_size",
    "first_trade_time",
    "first_trade_price",
    "first_trade_size",
    "high_bid_time",
    "high_bid_price",
    "high_bid_size",
    "high_ask_time",
    "high_ask_price",
    "high_ask_size",
    "high_trade_time",
    "high_trade_price",
    "high_trade_size",
    "low_bid_time",
    "low_bid_price",
    "low_bid_size",
    "low_ask_time",
    "low_ask_price",
    "low_ask_size",
    "low_trade_time",
    "low_trade_price",
    "low_trade_size",
    "close_bar_time",
    "close_bid_price",
    "close_bid_size",
    "close_ask_price",
    "close_ask_size",
    "last_trade_time",
    "last_trade_price",
    "last_trade_size",
    "min_spread",
    "max_spread",
    "cancel_size",
    "vwap",
    "nbbo_quote_count",
    "trade_at_bid",
    "trade_at_bid_mid",
    "trade_at_mid",
    "trade_at_mid_ask",
    "trade_at_ask",
    "trade_at_cross",
    "volume",
    "total_trades",
    "finra_volume",
    "finra_vwap",
    "uptick_volume",
    "downtick_volume",
    "repeat_uptick_volume",
    "repeat_downtick_volume",
    "unknown_tick_volume",
    "trade_to_mid_vol_weight",
    "trade_to_mid_vol_weight_rel",
    "time_weight_bid",
    "time_weight_ask",
    "timestamp",
]

# The option-chain schema, in the order the sp500_options case study reads.
SP500_OPTIONS_EXPECTED = [
    "date",
    "symbol",
    "call_put",
    "option_style",
    "strike",
    "expiration",
    "years_to_maturity",
    "days_to_maturity",
    "underlying_price",
    "bid",
    "ask",
    "mid_price",
    "implied_vol",
    "theo_price",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "iv_convergence",
]


def test_nasdaq100_schema_matches_what_the_loader_projects(convert):
    df = convert.parse_nasdaq100_csv(_nasdaq100_csv(convert))
    assert df.columns == NASDAQ100_EXPECTED


def test_nasdaq100_dtypes(convert):
    df = convert.parse_nasdaq100_csv(_nasdaq100_csv(convert))
    schema = df.schema
    assert schema["date"] == pl.Date
    assert schema["timestamp"] == pl.Datetime("us")
    assert schema["symbol"] == pl.String
    # Nanosecond time-of-day text no datetime type round-trips.
    assert schema["open_bar_time"] == pl.String
    assert schema["first_trade_price"] == pl.Float64
    assert schema["volume"] == pl.Int64


def test_nasdaq100_timestamp_is_the_bar_open(convert):
    """The loader filters regular hours on this, so it has to be the label minute."""
    df = convert.parse_nasdaq100_csv(_nasdaq100_csv(convert))
    assert df["timestamp"].to_list() == [
        datetime(2020, 3, 13, 4, 0),
        datetime(2020, 3, 13, 4, 1),
    ]
    assert df["date"].to_list() == [date(2020, 3, 13)] * 2


def test_nasdaq100_blank_fields_become_null(convert):
    """AlgoSeek leaves a field empty rather than writing a zero, and the difference
    matters: cancel_size is null when nothing was cancelled, not 0."""
    df = convert.parse_nasdaq100_csv(_nasdaq100_csv(convert))
    assert df["cancel_size"].to_list() == [None, None]
    assert df["finra_vwap"].to_list() == [None, None]


def test_nasdaq100_maps_the_columns_that_are_easy_to_confuse(convert):
    """TradeAtBid vs TradeAtBidCount, Volume vs TotalVolume, VolumeWeightPrice vs
    TotalVolumeWeightPrice: each pair differs by a suffix and by an order of
    magnitude, and picking the wrong one is silent."""
    df = convert.parse_nasdaq100_csv(_nasdaq100_csv(convert))
    row = df.row(0, named=True)
    assert row["trade_at_bid"] == 668
    assert row["volume"] == 7486
    assert row["total_trades"] == 90
    assert row["vwap"] == pytest.approx(255.35)
    assert row["trade_at_cross"] == 0


def test_sp500_options_schema_and_order(convert):
    """read_csv keeps the file's order, which puts mid before ask. The delivery has
    bid, ask, mid_price, and the case study reads positionally in places."""
    df = convert.parse_sp500_options_csv(_options_csv(convert))
    assert df.columns == SP500_OPTIONS_EXPECTED
    row = df.row(0, named=True)
    assert (row["bid"], row["ask"], row["mid_price"]) == (12.5, 13.5, 13.0)


def test_sp500_options_dates_are_parsed(convert):
    df = convert.parse_sp500_options_csv(_options_csv(convert))
    row = df.row(0, named=True)
    assert row["date"] == date(2020, 3, 13)
    assert row["expiration"] == date(2020, 4, 17)
    assert df.schema["days_to_maturity"] == pl.Int32


# --------------------------------------------------------------------------------
# Finding the days in either archive shape
# --------------------------------------------------------------------------------


def test_day_source_reads_the_nested_day_zips(convert, tmp_path):
    """The NASDAQ-100 archive is a zip of per-day zips of per-symbol CSVs."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("20200313/A/AAPL.csv", _nasdaq100_csv(convert))
        z.writestr("20200313/M/MSFT.csv", _nasdaq100_csv(convert, symbol="MSFT"))
    outer = tmp_path / "nasdaq.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("2020/20200313.zip", inner.getvalue())

    days = list(convert.DaySource(outer, "nasdaq100-minute-bars").days())
    assert [d for d, _ in days] == ["20200313"]
    assert len(days[0][1]()) == 2


def test_day_source_reads_an_extracted_options_tree(convert, tmp_path):
    """A reader who unpacks the 1.27M-member archive first still gets a working run."""
    day_dir = tmp_path / "2020" / "20200313"
    day_dir.mkdir(parents=True)
    for symbol in ("AAPL", "MSFT"):
        (day_dir / f"{symbol}.csv.gz").write_bytes(gzip.compress(_options_csv(convert)))
    # macOS zips carry a shadow tree of resource forks that is not data.
    shadow = tmp_path / "__MACOSX" / "2020" / "20200313"
    shadow.mkdir(parents=True)
    (shadow / "AAPL.csv.gz").write_bytes(b"not data")

    days = list(convert.DaySource(tmp_path, "sp500-options").days())
    assert [d for d, _ in days] == ["20200313"]
    payloads = days[0][1]()
    assert len(payloads) == 2
    assert convert.parse_sp500_options_csv(payloads[0]).height == 1


def test_day_source_reads_the_options_archive_as_algoseek_serves_it(convert, tmp_path):
    """The published options zip was re-zipped on macOS, and that broke every reader.

    It nests everything under one top-level folder and carries explicit directory
    entries, a __MACOSX shadow tree and .DS_Store files. The directory entry for a
    day's own folder was collected as a member of that day, read back as zero bytes
    and hit polars NoDataError, so the conversion died on the first day it reached.
    """
    archive = tmp_path / "options_daily_greeks_sp500.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("options_daily_greeks_sp500/", b"")
        z.writestr("options_daily_greeks_sp500/.DS_Store", b"\x00")
        z.writestr("__MACOSX/._options_daily_greeks_sp500", b"\x00")
        z.writestr("options_daily_greeks_sp500/2020/", b"")
        z.writestr("options_daily_greeks_sp500/2020/20200313/", b"")
        for symbol in ("AAPL", "MSFT"):
            z.writestr(
                f"options_daily_greeks_sp500/2020/20200313/{symbol}.csv.gz",
                gzip.compress(_options_csv(convert, symbol=symbol)),
            )
            z.writestr(
                f"__MACOSX/options_daily_greeks_sp500/2020/20200313/._{symbol}.csv.gz",
                b"\x00",
            )
        z.writestr("options_daily_greeks_sp500/2020/20200313/.DS_Store", b"\x00")

    days = list(convert.DaySource(archive, "sp500-options").days())
    assert [d for d, _ in days] == ["20200313"]
    payloads = days[0][1]()
    assert len(payloads) == 2
    assert all(convert.parse_sp500_options_csv(p).height == 1 for p in payloads)

    out = tmp_path / "data"
    assert convert.convert_sp500_options(archive, out, workers=1, force=False) == 0
    written = out / "equities" / "market" / "sp500" / "options" / "year=2020" / "20200313.parquet"
    assert pl.read_parquet(written).height == 2


def test_day_source_reads_an_extracted_nasdaq_tree(convert, tmp_path):
    """Extracting the NASDAQ-100 archive one level gives day *zips*, not CSVs.

    Its outer archive is a zip of zips, so a reader who unpacks it and points
    --source at the result has an ``2020/20200313.zip`` tree. Globbing for CSVs
    finds nothing there and the run used to report an empty source.
    """
    year_dir = tmp_path / "2020"
    year_dir.mkdir()
    for day in ("20200313", "20200316"):
        with zipfile.ZipFile(year_dir / f"{day}.zip", "w") as z:
            z.writestr(f"{day}/A/AAPL.csv", _nasdaq100_csv(convert, day=day))
            z.writestr(f"{day}/M/MSFT.csv", _nasdaq100_csv(convert, symbol="MSFT", day=day))

    days = list(convert.DaySource(tmp_path, "nasdaq100-minute-bars").days())
    assert [d for d, _ in days] == ["20200313", "20200316"]
    payloads = days[0][1]()
    assert len(payloads) == 2
    assert convert.parse_nasdaq100_csv(payloads[0]).height == 2


def test_extracted_nasdaq_tree_converts_end_to_end(convert, tmp_path):
    year_dir = tmp_path / "archive" / "2020"
    year_dir.mkdir(parents=True)
    with zipfile.ZipFile(year_dir / "20200313.zip", "w") as z:
        z.writestr("20200313/A/AAPL.csv", _nasdaq100_csv(convert))

    out = tmp_path / "data"
    assert convert.convert_nasdaq100(tmp_path / "archive", out, workers=1, force=False) == 0
    written = out / "equities" / "market" / "nasdaq100" / "minute_bars" / "year=2020"
    assert (written / "month=03.parquet").is_file()


def test_day_source_rejects_a_source_that_is_neither(convert, tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not an archive")
    with pytest.raises(SystemExit):
        list(convert.DaySource(junk, "sp500-options").days())


def test_the_documented_workflow_works_with_an_external_data_root(convert, tmp_path, monkeypatch):
    """Convert, then build, with ML4T_DATA_PATH outside the repository.

    The converter always honored the data root; the three build scripts resolved
    their input and output from their own directory, so with an external root
    they looked where nothing had been written and left their output where no
    loader reads. This walks conversion followed by a build and asserts both
    landed under the configured root.
    """
    import importlib.util

    source = tmp_path / "chains" / "2020" / "20200313"
    source.mkdir(parents=True)
    for symbol in ("AAPL", "MSFT"):
        (source / f"{symbol}.csv.gz").write_bytes(
            gzip.compress(_options_csv(convert, symbol=symbol))
        )

    data_root = tmp_path / "elsewhere"
    assert convert.convert_sp500_options(tmp_path / "chains", data_root, 1, False) == 0
    raw = data_root / "equities" / "market" / "sp500" / "options" / "year=2020" / "20200313.parquet"
    assert raw.is_file(), "the converter must write under the configured data root"

    build_py = Path(da.__file__).parent / "equities" / "market" / "sp500" / "build_options_eda.py"
    spec = importlib.util.spec_from_file_location("build_options_eda_under_test", build_py)
    build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build
    spec.loader.exec_module(build)

    monkeypatch.setenv("ML4T_DATA_PATH", str(data_root))
    monkeypatch.setattr(sys, "argv", ["build_options_eda.py", "--data-path", str(data_root)])
    build.main()

    built = data_root / "equities" / "market" / "sp500" / "options_eda" / "year=2020.parquet"
    assert built.is_file(), "the build script must read and write under the same root"
    # Both fixture symbols are in EDA_SYMBOLS, so both rows survive the filter.
    assert sorted(pl.read_parquet(built)["symbol"].to_list()) == ["AAPL", "MSFT"]


# --------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------


def test_convert_writes_the_layout_the_loader_scans(convert, tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("20200313/A/AAPL.csv", _nasdaq100_csv(convert))
    archive = tmp_path / "nasdaq.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("2020/20200313.zip", inner.getvalue())

    out = tmp_path / "data"
    assert convert.convert_nasdaq100(archive, out, workers=1, force=False) == 0

    written = out / "equities" / "market" / "nasdaq100" / "minute_bars" / "year=2020"
    assert (written / "month=03.parquet").is_file()
    # Exactly how load_nasdaq100_bars() reads it.
    scanned = pl.scan_parquet(written.parent / "**/*.parquet", hive_partitioning=True).collect()
    assert scanned.height == 2
    assert scanned["year"].unique().to_list() == [2020]


def test_convert_resumes_and_force_rebuilds(convert, tmp_path, capsys):
    day_dir = tmp_path / "2020" / "20200313"
    day_dir.mkdir(parents=True)
    (day_dir / "AAPL.csv.gz").write_bytes(gzip.compress(_options_csv(convert)))
    out = tmp_path / "data"

    convert.convert_sp500_options(tmp_path, out, workers=1, force=False)
    written = out / "equities" / "market" / "sp500" / "options" / "year=2020" / "20200313.parquet"
    assert written.is_file()
    first = written.stat().st_mtime_ns

    convert.convert_sp500_options(tmp_path, out, workers=1, force=False)
    assert written.stat().st_mtime_ns == first, "an existing partition must be skipped"

    convert.convert_sp500_options(tmp_path, out, workers=1, force=True)
    assert written.stat().st_mtime_ns != first, "--force must rebuild it"


def test_convert_reports_an_empty_source(convert, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert convert.convert_sp500_options(empty, tmp_path / "data", workers=1, force=False) == 1
