#!/usr/bin/env python3
"""Download individual CME futures contracts from Databento.

    uv run python data/futures/market/databento_individual.py --estimate
    uv run python data/futures/market/databento_individual.py --product ES

`download.py` fetches *continuous* contracts (`ES.v.0` and friends): one
synthetic series per tenor, already rolled. The roll itself cannot be taught
from those, because the roll is the thing they have already done. So
`02_futures_continuous` reconstructs it from the individual contracts, and this
script is what fetches them.

The request is by parent symbol (`ES.FUT`, `stype_in="parent"`), which returns
every listed contract for the product, outrights and calendar spreads alike.
The spreads are deliberately kept: the notebook filters them out on price, and
seeing that filter matter is part of the lesson. It is why `close` can be
negative in this file.

One file per product, at `futures/market/individual/{PRODUCT}/data.parquet`,
which is where `data/futures/loader.py` looks.

Cost: the default ES and CL windows are a few dollars at daily resolution.
Nothing downloads before the estimate is shown and acknowledged; `--estimate`
prints the estimate and stops.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from utils.downloading import (
    databento_acknowledge,
    databento_estimate_only_notice,
    patch_databento_symbology,
    resolve_data_dir,
)

# Databento's `rtype` for each OHLCV schema, recorded in the output so a reader
# can tell the resolution of a file from the file itself.
RTYPE_BY_SCHEMA = {
    "ohlcv-1s": 32,
    "ohlcv-1m": 33,
    "ohlcv-1h": 34,
    "ohlcv-1d": 35,
}


@dataclass
class IndividualConfig:
    """The `individual:` block of config.yaml."""

    dataset: str
    schema: str
    stype_in: str
    output_dir: str
    start: str
    end: str
    products: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, config_path: Path) -> IndividualConfig:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        individual = data.get("individual")
        if not individual:
            msg = f"{config_path} has no `individual:` section"
            raise SystemExit(msg)

        return cls(
            dataset=individual.get("dataset", data["dataset"]),
            schema=individual.get("schema", "ohlcv-1d"),
            stype_in=individual.get("stype_in", "parent"),
            output_dir=individual.get("output_dir", "futures/market/individual"),
            start=individual.get("start", data["default_start"]),
            end=individual.get("end", data["default_end"]),
            products=individual.get("products", {}),
        )

    def product_start(self, product: str) -> str:
        return self.products.get(product, {}).get("start", self.start)

    def product_end(self, product: str) -> str:
        return self.products.get(product, {}).get("end", self.end)


def get_config_path() -> Path:
    return Path(__file__).parent / "config.yaml"


def output_path(data_dir: Path, config: IndividualConfig, product: str) -> Path:
    return data_dir / config.output_dir / product / "data.parquet"


def parent_symbol(product: str) -> str:
    """Databento's parent symbol for a futures product's full contract set."""
    return f"{product}.FUT"


def estimate_cost(client, config: IndividualConfig, products: list[str]) -> float:
    """Sum Databento's own cost estimate over the requested products."""
    total = 0.0
    for product in products:
        try:
            total += client.metadata.get_cost(
                dataset=config.dataset,
                symbols=[parent_symbol(product)],
                schema=config.schema,
                start=config.product_start(product),
                end=config.product_end(product),
                stype_in=config.stype_in,
            )
        except Exception as e:  # noqa: BLE001 - a failed estimate must not read as $0
            print(f"  WARNING: cost estimate failed for {product}: {e}")
            print("  Treating this product's cost as unknown, not as zero.")
            return float("nan")
    return total


def download_product(client, config: IndividualConfig, product: str, path: Path) -> int:
    """Fetch one product's contracts and write the parquet. Returns row count."""
    data = client.timeseries.get_range(
        dataset=config.dataset,
        symbols=[parent_symbol(product)],
        schema=config.schema,
        start=config.product_start(product),
        end=config.product_end(product),
        stype_in=config.stype_in,
    )

    df = pl.from_pandas(data.to_df().reset_index())

    # to_df() names the index ts_event; the loader and the notebooks expect
    # `timestamp`.
    if "ts_event" in df.columns:
        df = df.rename({"ts_event": "timestamp"})

    # `symbol` is per-contract and varies in width; instrument_id is what the
    # notebook groups on. Keep the columns the shipped artifact carries, in its
    # order, so a re-download is comparable to what students already have.
    keep = [
        "timestamp",
        "rtype",
        "publisher_id",
        "instrument_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        msg = f"{product}: Databento response is missing {missing}; got {df.columns}"
        raise SystemExit(msg)

    df = df.select(keep).with_columns(pl.lit(product).alias("product"))
    df = df.sort(["timestamp", "instrument_id"])

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
    return df.height


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--product",
        "-p",
        action="append",
        dest="products",
        help="Product(s) to download (can repeat). Default: all in the config",
    )
    parser.add_argument("--start-date", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse to download if the estimate exceeds this",
    )
    parser.add_argument(
        "--estimate",
        "--estimate-only",
        "-e",
        action="store_true",
        help="Print the cost estimate and stop",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be downloaded"
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Re-download even if the file exists"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override the data directory")
    args = parser.parse_args()

    config_path = args.config or get_config_path()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        return 1

    config = IndividualConfig.load(config_path)
    if args.start_date:
        config.start = args.start_date
    if args.end_date:
        config.end = args.end_date

    products = args.products or list(config.products)
    unknown = [p for p in products if p not in config.products]
    if unknown:
        print(f"ERROR: not in the config's individual products: {unknown}")
        print(f"Available: {list(config.products)}")
        return 1

    data_dir = resolve_data_dir(args.data_dir)

    todo, have = [], []
    for product in products:
        path = output_path(data_dir, config, product)
        (have if path.exists() and not args.force else todo).append(product)

    for product in have:
        print(f"  {product}: already at {output_path(data_dir, config, product)} (--force to redo)")

    if not todo:
        print("\nNothing to do.")
        return 0

    print(f"\nTo download: {', '.join(todo)}")
    for product in todo:
        print(
            f"  {product}  {parent_symbol(product)}  {config.schema}  "
            f"{config.product_start(product)} to {config.product_end(product)}"
        )

    if args.dry_run:
        print("\n--dry-run: stopping before any request.")
        return 0

    if not os.environ.get("DATABENTO_API_KEY"):
        print("\nERROR: DATABENTO_API_KEY is not set. Put it in .env or the environment.")
        return 1

    try:
        import databento as db

        patch_databento_symbology()
        client = db.Historical()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not initialize the Databento client: {e}")
        return 1

    cost = estimate_cost(client, config, todo)
    if cost != cost:  # NaN: at least one estimate failed
        print("\nRefusing to download without a cost estimate.")
        return 1

    if args.estimate:
        databento_estimate_only_notice(cost)
        return 0

    if args.max_cost is not None and cost > args.max_cost:
        print(f"\nEstimated ${cost:.2f} exceeds --max-cost ${args.max_cost:.2f}. Not downloading.")
        print("Raise --max-cost, or narrow it with --product / --start-date / --end-date.")
        return 1

    if not databento_acknowledge(cost, force=args.force):
        print("Cancelled.")
        return 0

    for product in todo:
        path = output_path(data_dir, config, product)
        print(f"\nDownloading {product} ...")
        rows = download_product(client, config, product, path)
        print(f"  {rows:,} rows -> {path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
