#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from utils.downloading import (
    create_base_parser,
    load_dotenv,
    load_section,
    print_download_summary,
    print_dry_run_notice,
    print_section,
    resolve_data_dir,
    resolve_storage_path,
)


def undo_splits(panel: pl.DataFrame) -> pl.DataFrame:
    """Restate a split-adjusted panel as the price and share count that reached the tape.

    A split on date ``d`` restates every price strictly before ``d``, so the factor that
    undoes it at ``t`` is the product of every split after ``t``. The split date's own
    price is already post-split and keeps whatever later splits apply to it. Yahoo writes
    ``0`` in the split column where nothing split.

    Turnover is invariant under this: price is multiplied and volume divided by the same
    factor, which is why a split distorts a per-share commission but not a dollar-volume
    screen.

    Args:
        panel: ``timestamp``, ``symbol``, ``close``, ``volume``, ``split``, sorted within
            symbol by timestamp.

    Returns:
        ``timestamp``, ``symbol``, ``close``, ``volume``, with the splits removed.
    """
    ratio = pl.when(pl.col("split") > 0).then(pl.col("split")).otherwise(1.0)
    return (
        panel.with_columns(ratio.alias("_ratio"))
        .with_columns(
            pl.col("_ratio")
            .reverse()
            .cum_prod()
            .reverse()
            .shift(-1, fill_value=1.0)
            .over("symbol")
            .alias("_future_splits")
        )
        .select(
            "timestamp",
            "symbol",
            close=pl.col("close") * pl.col("_future_splits"),
            volume=pl.col("volume") / pl.col("_future_splits"),
        )
        .sort(["symbol", "timestamp"])
    )


def write_unadjusted(manager) -> Path:
    """Download the price a share actually traded at, and the number of shares that traded.

    `etf_universe.parquet` is split- and dividend-adjusted throughout, which is what a
    return needs and what a dollar amount must not use. An adjusted close is the traded
    price divided by every distribution that followed it, so on SPY's first session here
    it reads $87.23 against the $126.70 the fund actually changed hands at. Anything
    denominated in dollars or in shares - a turnover screen, a commission quoted per share
    - is wrong by that factor, and wrong by more the further back it looks.

    Two adjustments have to come off, and Yahoo removes only one of them for you.
    ``auto_adjust=False`` gives a close still carrying every **split**: after IVW's 4:1 in
    2020 its whole earlier history reads a quarter of what it traded at, and after OIH's
    1:20 twenty times. Splits leave turnover alone, because the share count is scaled the
    opposite way and the product is unchanged, but a commission quoted per share is wrong
    by the split factor. So the cumulative future split is multiplied back out here and
    the share count divided by it, which is what the two columns then mean: the price on
    the tape that day, and the shares that changed hands that day.

    Splits are read to the present rather than to the end of the sample, because a split
    restates the history behind it: VUG's 6:1 in April 2026 and PPLT's 10:1 in May 2026
    both moved every price in this window.
    """
    import yfinance as yf

    symbols = sorted(manager.config.get_all_symbols())
    frames: list[pl.DataFrame] = []

    # Read to today, not to the configured end: a split after the sample still restates
    # every price inside it, so the factor is only complete if the window reaches it.
    today = (date.today() + timedelta(days=1)).isoformat()
    sample_end = date.fromisoformat(str(manager.config.end))

    for start in range(0, len(symbols), 50):
        chunk = symbols[start : start + 50]
        raw = yf.download(
            chunk,
            start=manager.config.start,
            end=today,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            continue

        parts: list[pl.DataFrame] = []
        for field, name in (("Close", "close"), ("Volume", "volume"), ("Stock Splits", "split")):
            panel = raw[field]
            if panel.ndim != 2:
                raise TypeError(
                    f"yfinance returned a 1-D {field} for {len(chunk)} symbols; "
                    "expected one column per ticker"
                )
            # yfinance names the column axis "Ticker" and leaves the index unnamed once a
            # field is selected out of the multi-index, so name both here rather than
            # depending on what reset_index() invents.
            tidy = panel.rename_axis(index="timestamp", columns=None).reset_index()
            long = pl.from_pandas(tidy).unpivot(
                index="timestamp", variable_name="symbol", value_name=name
            )
            parts.append(long.with_columns(pl.col("timestamp").cast(pl.Datetime("us"))))

        merged = parts[0]
        for part in parts[1:]:
            merged = merged.join(part, on=["timestamp", "symbol"], how="inner")
        frames.append(merged)
        print(f"  {chunk[0]}..{chunk[-1]}: {len(chunk)} symbols")

    panel = pl.concat(frames, how="vertical").drop_nulls("close").sort(["symbol", "timestamp"])
    unadjusted = undo_splits(panel).filter(
        pl.col("timestamp") <= pl.lit(sample_end).cast(pl.Datetime("us"))
    )

    output_path = manager.config.storage_path / "etf_universe_unadjusted.parquet"
    unadjusted.write_parquet(output_path)
    return output_path


def write_dictionary(manager) -> Path:
    rows: list[dict[str, str]] = []
    for group, info in manager.config.tickers.items():
        description = info.get("description", "")
        for symbol in info.get("symbols", []):
            rows.append({"symbol": symbol, "group": group, "description": description})

    dictionary = pl.DataFrame(rows).sort(["group", "symbol"])
    output_path = manager.config.storage_path / "etf_universe_dictionary.parquet"
    dictionary.write_parquet(output_path)
    return output_path


def main() -> None:
    parser = create_base_parser("Download ETF data from Yahoo Finance")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to ETF config",
    )
    parser.add_argument("--symbol", "-s", type=str, help="Download a single ETF symbol")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Extend the configured end date to today and append new data",
    )
    parser.add_argument(
        "--unadjusted-only",
        action="store_true",
        help=(
            "Refresh only etf_universe_unadjusted.parquet, leaving the adjusted panel "
            "untouched. The traded close is settled history and reproduces exactly; the "
            "adjusted panel moves whenever a fund distributes, so re-downloading it "
            "restates every earlier price."
        ),
    )
    args = parser.parse_args()

    load_dotenv()

    if args.dry_run:
        print_dry_run_notice()

    from ml4t.data.etfs import ETFDataManager

    data_root = resolve_data_dir(args.data_path)
    config_path = args.config.expanduser()  # honor --config ~/... like other helpers
    manager = ETFDataManager.from_config(config_path)
    # ETFDataManager.from_config() absolutizes storage_path against the repo
    # directory, which ignores ML4T_DATA_PATH; passing that absolute path back
    # through resolve_storage_path is a no-op (it preserves absolute paths). Read
    # the raw (relative) config value and re-resolve under the selected data root
    # so ETF data lands alongside every other dataset — $ML4T_DATA_PATH/etfs/market
    # — the same pattern the crypto/fx downloaders use.
    configured_storage = load_section(config_path, "etfs").get("storage_path")
    manager.config.storage_path = resolve_storage_path(data_root, configured_storage, "etfs")

    if args.symbol:
        manager.config.tickers = {
            "adhoc": {"description": "Ad hoc symbol", "symbols": [args.symbol]}
        }

    if args.update:
        manager.config.end = date.today().isoformat()

    output_path = manager.config.storage_path / "etf_universe.parquet"

    print_section("ETF DATA DOWNLOAD (Yahoo Finance)")
    print(f"Config: {args.config}")
    print(f"Output: {output_path}")
    print(f"Date range: {manager.config.start} to {manager.config.end}")
    print(f"Symbols: {len(manager.config.get_all_symbols())}")

    if args.dry_run:
        print_download_summary(
            {
                "symbols": len(manager.config.get_all_symbols()),
                "start_date": manager.config.start,
                "end_date": manager.config.end,
                "output_file": str(output_path),
            },
            dry_run=True,
        )
        return

    manager.config.storage_path.mkdir(parents=True, exist_ok=True)

    if args.unadjusted_only:
        print("Traded (unadjusted) close and volume, for dollar- and share-denominated screens:")
        unadjusted_path = write_unadjusted(manager)
        print(f"Written: {unadjusted_path}")
        return

    if output_path.exists() and not args.force:
        stats = manager.update()
        action = "updated"
    else:
        stats = manager.download_all(force=args.force)
        action = "downloaded"

    print("Traded (unadjusted) close and volume, for dollar- and share-denominated screens:")
    unadjusted_path = write_unadjusted(manager)
    print(f"Written: {unadjusted_path}")

    dictionary_path = write_dictionary(manager)
    metadata_path = manager.config.storage_path / "etf_universe_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["dictionary_file"] = str(dictionary_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    df = pl.read_parquet(output_path)
    print_download_summary(
        {
            "action": action,
            "rows": len(df),
            "symbols": df["symbol"].n_unique(),
            "partitions": len(stats),
            "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            "output_file": str(output_path),
            "dictionary_file": str(dictionary_path),
        }
    )


if __name__ == "__main__":
    main()
