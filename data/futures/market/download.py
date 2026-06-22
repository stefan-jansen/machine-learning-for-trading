#!/usr/bin/env python3
"""
Download Futures Data from Databento

Config-driven, idempotent download of CME futures continuous contracts.

Features:
- Reads configuration from config/cme_futures.yaml
- Hive-partitioned storage by product/year for easy incremental updates
- Automatic detection of existing data to avoid re-downloading
- Support for extending data (readers can add years later)

Usage:
    # Estimate cost only (no download)
    python cme_futures.py --estimate

    # Download all products from config (idempotent)
    python cme_futures.py

    # Download specific products
    python cme_futures.py --product ES --product NQ

    # Download extension products (crypto, etc.)
    python cme_futures.py --extension

    # Dry run (show what would be downloaded)
    python cme_futures.py --dry-run

    # Force re-download specific years
    python cme_futures.py --product ES --year 2024 --force

Author: ML4T Third Edition
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FuturesConfig:
    """Futures download configuration from YAML."""

    dataset: str
    schema: str
    roll_type: str
    tenors: list[int]
    default_start: str
    default_end: str
    products: dict[str, dict[str, Any]]
    extension_products: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path) -> FuturesConfig:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(
            dataset=data["dataset"],
            schema=data["schema"],
            roll_type=data["roll_type"],
            tenors=data["tenors"],
            default_start=data["default_start"],
            default_end=data["default_end"],
            products=data.get("products", {}),
            extension_products=data.get("extension_products", {}),
        )

    def get_product_start(self, product: str) -> str:
        """Get start date for a product (from config or default)."""
        if product in self.products:
            return self.products[product].get("start", self.default_start)
        if product in self.extension_products:
            return self.extension_products[product].get("start", self.default_start)
        return self.default_start

    def get_all_products(self, include_extension: bool = False) -> list[str]:
        """Get list of products to download."""
        products = list(self.products.keys())
        if include_extension:
            products.extend(self.extension_products.keys())
        return products


def get_config_path() -> Path:
    """Get path to config file."""
    return Path(__file__).parent / "config.yaml"


# ============================================================================
# Hive Partition Helpers
# ============================================================================


def get_hive_base_path(data_dir: Path) -> Path:
    """Get base path for Hive-partitioned futures data."""
    return data_dir / "futures" / "market" / "continuous" / "hourly"


def get_partition_path(data_dir: Path, product: str, year: int) -> Path:
    """Get path to a specific partition."""
    return get_hive_base_path(data_dir) / f"product={product}" / f"year={year}" / "data.parquet"


def list_existing_years(data_dir: Path, product: str) -> list[int]:
    """List years with existing data for a product."""
    product_dir = get_hive_base_path(data_dir) / f"product={product}"
    if not product_dir.exists():
        return []

    years = []
    for year_dir in product_dir.iterdir():
        if year_dir.is_dir() and year_dir.name.startswith("year="):
            try:
                year = int(year_dir.name.split("=")[1])
                # Verify data file exists
                if (year_dir / "data.parquet").exists():
                    years.append(year)
            except ValueError:
                continue

    return sorted(years)


def get_year_coverage(
    data_dir: Path, product: str, year: int
) -> tuple[datetime | None, datetime | None, int]:
    """
    Get date range and row count for a specific year partition.

    Returns:
        (min_date, max_date, row_count) or (None, None, 0) if not exists
    """
    partition_path = get_partition_path(data_dir, product, year)
    if not partition_path.exists():
        return None, None, 0

    try:
        stats = (
            pl.scan_parquet(partition_path)
            .select(
                pl.col("timestamp").min().alias("min_date"),
                pl.col("timestamp").max().alias("max_date"),
                pl.len().alias("rows"),
            )
            .collect()
        )

        if stats.height == 0:
            return None, None, 0

        return stats["min_date"][0], stats["max_date"][0], stats["rows"][0]
    except Exception:
        return None, None, 0


# ============================================================================
# Coverage Analysis
# ============================================================================


@dataclass
class YearStatus:
    """Status of a single year partition."""

    year: int
    exists: bool
    rows: int
    min_date: datetime | None
    max_date: datetime | None
    complete: bool  # Has data through end of year (or end_date if current year)


@dataclass
class ProductCoverage:
    """Coverage status for a product."""

    product: str
    config_start: str
    config_end: str
    years: dict[int, YearStatus]
    years_to_download: list[int]


def analyze_product_coverage(
    data_dir: Path,
    product: str,
    config: FuturesConfig,
    end_date: str,
) -> ProductCoverage:
    """Analyze coverage for a single product."""
    start_date = config.get_product_start(product)
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    years = {}
    years_to_download = []

    for year in range(start_year, end_year + 1):
        min_date, max_date, rows = get_year_coverage(data_dir, product, year)

        if rows == 0:
            # No data for this year
            years[year] = YearStatus(
                year=year,
                exists=False,
                rows=0,
                min_date=None,
                max_date=None,
                complete=False,
            )
            years_to_download.append(year)
        else:
            # Check if year is complete
            # For past years: data within 5 days of Dec 31 is considered complete
            # (accounts for holidays, weekends, and exchange closures)
            # For current/end year: check against end_date with same tolerance
            if year == end_year:
                year_end = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                year_end = datetime(year, 12, 31)

            # Allow 5 day tolerance for completeness (handles Dec 31 holidays, etc.)
            complete = (
                max_date is not None and max_date.date() >= (year_end - timedelta(days=5)).date()
            )

            years[year] = YearStatus(
                year=year,
                exists=True,
                rows=rows,
                min_date=min_date,
                max_date=max_date,
                complete=complete,
            )

            if not complete:
                years_to_download.append(year)

    return ProductCoverage(
        product=product,
        config_start=start_date,
        config_end=end_date,
        years=years,
        years_to_download=years_to_download,
    )


# ============================================================================
# Cost Estimation
# ============================================================================


def estimate_cost(
    config: FuturesConfig,
    products: list[str],
    years_by_product: dict[str, list[int]],
) -> float:
    """
    Estimate download cost using Databento API.

    Returns:
        Estimated cost in USD
    """
    try:
        import databento as db

        client = db.Historical()
    except Exception as e:
        print(f"Warning: Could not initialize Databento client: {e}")
        return 0.0

    # Build symbols and date ranges
    total_cost = 0.0

    for product, years in years_by_product.items():
        if not years:
            continue

        # Build symbols for this product
        symbols = [f"{product}.{config.roll_type}.{pos}" for pos in config.tenors]

        for year in years:
            start = f"{year}-01-01"
            end = f"{year}-12-31"

            # Adjust start for products with later availability
            product_start = config.get_product_start(product)
            if product_start > start:
                start = product_start

            try:
                cost = client.metadata.get_cost(
                    dataset=config.dataset,
                    symbols=symbols,
                    schema=config.schema,
                    start=start,
                    end=end,
                    stype_in="continuous",
                )
                total_cost += cost
            except Exception as e:
                print(f"Warning: Cost estimation failed for {product} {year}: {e}")

    return total_cost


# ============================================================================
# Download Functions
# ============================================================================


def download_full_product(
    product: str,
    config: FuturesConfig,
    data_dir: Path,
    dry_run: bool = False,
) -> tuple[int, str]:
    """
    Download FULL date range for a product in ONE API call, then partition to Hive format.

    This is ~15x more efficient than year-by-year downloads due to reduced API overhead.

    Returns:
        (rows_downloaded, status_message)
    """
    import databento as db

    start_date = config.get_product_start(product)
    end_date = config.default_end

    if dry_run:
        return 0, f"[DRY RUN] Would download {product}: {start_date} to {end_date}"

    # Build symbols for continuous contracts
    symbols = [f"{product}.{config.roll_type}.{pos}" for pos in config.tenors]

    try:
        client = db.Historical()
        print(f"    Downloading {product} ({start_date} to {end_date})...", flush=True)

        data = client.timeseries.get_range(
            dataset=config.dataset,
            symbols=symbols,
            schema=config.schema,
            start=start_date,
            end=end_date,
            stype_in="continuous",
        )

        # Fix databento 0.72.0 symbology bug, then convert via parquet
        patch_databento_symbology()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        data.to_parquet(tmp_path)
        df_pl = pl.read_parquet(tmp_path)
        tmp_path.unlink()

        if df_pl.height == 0:
            return 0, f"No data returned for {product}"

        # Databento returns ts_event — rename to canonical timestamp
        if "ts_event" in df_pl.columns:
            df_pl = df_pl.rename({"ts_event": "timestamp"})

        # Resolve the symbol column (databento uses 'asset' or 'symbol' depending on path)
        sym_col = (
            "asset" if "asset" in df_pl.columns else "symbol" if "symbol" in df_pl.columns else None
        )

        # Add product column (clean name, e.g., "GF")
        df_pl = df_pl.with_columns(pl.lit(product).alias("product"))

        # Extract tenor from continuous symbol (e.g., GF.v.0 -> 0)
        if sym_col:
            df_pl = df_pl.with_columns(
                pl.col(sym_col)
                .str.extract(rf"\.{config.roll_type}\.(\d+)$", 1)
                .cast(pl.Int8)
                .alias("tenor")
            )

        # Select output columns (canonical schema)
        keep_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rtype",
            "publisher_id",
            "instrument_id",
            "tenor",
            "product",
        ]
        df_pl = df_pl.select([c for c in keep_cols if c in df_pl.columns])

        # Extract year from timestamp and partition to Hive format
        df_pl = df_pl.with_columns(pl.col("timestamp").dt.year().alias("year"))

        total_rows = df_pl.height
        years_written = 0

        # Write to Hive partitions
        for year in df_pl["year"].unique().sort().to_list():
            year_data = df_pl.filter(pl.col("year") == year)
            output_path = get_partition_path(data_dir, product, year)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Merge with existing if present
            if output_path.exists():
                existing = pl.read_parquet(output_path)
                common_cols = [c for c in year_data.columns if c in existing.columns]
                year_data = year_data.select(common_cols)
                existing = existing.select(common_cols)
                combined = pl.concat([existing, year_data])
                combined = combined.unique(subset=["timestamp", "tenor"], keep="last").sort(
                    ["timestamp", "tenor"]
                )
                combined.write_parquet(output_path)
            else:
                year_data.sort(["timestamp", "tenor"]).write_parquet(output_path)

            years_written += 1

        return (
            total_rows,
            f"Downloaded {product}: {total_rows:,} rows -> {years_written} year partitions",
        )

    except Exception as e:
        return 0, f"Error downloading {product}: {e}"


def download_product_efficient(
    product: str,
    config: FuturesConfig,
    data_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Download complete product in ONE API call (efficient).

    Returns:
        dict with stats: downloaded, failed, rows, messages
    """
    rows, msg = download_full_product(product, config, data_dir, dry_run=dry_run)

    stats = {"downloaded": 0, "failed": 0, "rows": 0, "messages": [msg]}

    if rows > 0:
        stats["downloaded"] = 1
        stats["rows"] = rows
    elif "Error" in msg:
        stats["failed"] = 1

    return stats


def download_parallel(
    products_to_download: list[str],
    config: FuturesConfig,
    data_dir: Path,
    workers: int = 4,
    dry_run: bool = False,
) -> dict:
    """
    Download multiple products in parallel using efficient full-range downloads.

    Each product is downloaded in ONE API call (full date range), then partitioned locally.
    This is ~15x faster than year-by-year downloads.

    Args:
        products_to_download: list of product symbols to download
        config: FuturesConfig instance
        data_dir: data directory path
        workers: number of parallel workers
        dry_run: if True, don't actually download

    Returns:
        dict with aggregate stats
    """
    total_stats = {"downloaded": 0, "failed": 0, "rows": 0, "products_done": 0}
    total_products = len(products_to_download)

    if not products_to_download:
        print("No products to download.")
        return total_stats

    print(f"\nDownloading {total_products} products with {workers} parallel workers...")
    print("(Using efficient full-range downloads - 1 API call per product)")
    print()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all product downloads using efficient method
        futures = {
            executor.submit(download_product_efficient, product, config, data_dir, dry_run): product
            for product in products_to_download
        }

        # Process results as they complete
        for future in as_completed(futures):
            product = futures[future]
            try:
                result = future.result()
                total_stats["downloaded"] += result["downloaded"]
                total_stats["failed"] += result["failed"]
                total_stats["rows"] += result["rows"]
                total_stats["products_done"] += 1

                # Print progress
                print(
                    f"[{total_stats['products_done']}/{total_products}] {product}: "
                    f"{result['rows']:,} rows"
                )

                # Print individual messages
                for msg in result["messages"]:
                    print(f"  {msg}")

            except Exception as e:
                print(f"[ERROR] {product}: {e}")
                total_stats["failed"] += 1
                total_stats["products_done"] += 1

    return total_stats


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download futures data from Databento (config-driven)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check coverage and estimate cost
    python cme_futures.py --estimate

    # Download all products from config (idempotent)
    python cme_futures.py

    # Download specific products
    python cme_futures.py --product ES --product NQ

    # Download extension products (crypto, etc.)
    python cme_futures.py --extension

    # Dry run (show what would be downloaded)
    python cme_futures.py --dry-run

    # Force re-download specific years
    python cme_futures.py --product ES --year 2024 --force
        """,
    )

    # Product selection
    parser.add_argument(
        "--product",
        "-p",
        action="append",
        dest="products",
        help="Specific product(s) to download (can repeat). Default: all from config",
    )
    parser.add_argument(
        "--extension",
        "-x",
        action="store_true",
        help="Include extension products (crypto, SOFR, etc.)",
    )

    # Year selection
    parser.add_argument(
        "--year",
        "-y",
        action="append",
        type=int,
        dest="years",
        help="Specific year(s) to download (can repeat). Default: all needed",
    )

    # Date override
    parser.add_argument(
        "--end-date",
        default=None,
        help="Override end date from config (YYYY-MM-DD)",
    )

    # Options
    parser.add_argument(
        "--estimate",
        "--estimate-only",
        "-e",
        action="store_true",
        help="Only estimate costs, don't download",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if data exists",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config/cme_futures.yaml)",
    )
    parser.add_argument(
        "--parallel",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel download workers (default: 1 = sequential)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = args.config or get_config_path()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    config = FuturesConfig.load(config_path)

    # Get data directory
    data_dir = resolve_data_dir(None)

    # Determine products to process
    if args.products:
        products = args.products
    else:
        products = config.get_all_products(include_extension=args.extension)

    # Get end date
    end_date = args.end_date or config.default_end

    print("=" * 70)
    print("DATABENTO FUTURES DOWNLOAD (Config-Driven)")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Data directory: {data_dir}")
    print(f"Dataset: {config.dataset} | Schema: {config.schema}")
    print(f"Roll type: {config.roll_type} | Tenors: {config.tenors}")
    print(
        f"Products: {len(products)} ({', '.join(products[:5])}{'...' if len(products) > 5 else ''})"
    )
    print(f"Date range: (per-product start) to {end_date}")
    print()

    # Analyze coverage for each product
    print("=" * 70)
    print("COVERAGE ANALYSIS")
    print("=" * 70)

    coverage_by_product: dict[str, ProductCoverage] = {}
    years_to_download: dict[str, list[int]] = {}

    for product in products:
        cov = analyze_product_coverage(data_dir, product, config, end_date)
        coverage_by_product[product] = cov

        # Apply year filter if specified
        if args.years:
            years_needed = [y for y in args.years if y in cov.years_to_download or args.force]
        elif args.force:
            # Force all years in range
            start_year = int(config.get_product_start(product)[:4])
            end_year = int(end_date[:4])
            years_needed = list(range(start_year, end_year + 1))
        else:
            years_needed = cov.years_to_download

        years_to_download[product] = years_needed

    # Summary of coverage
    total_years_needed = sum(len(years) for years in years_to_download.values())
    products_needing_update = [p for p, years in years_to_download.items() if years]

    print(f"Products needing updates: {len(products_needing_update)}/{len(products)}")
    print(f"Total year-partitions to download: {total_years_needed}")
    print()

    if products_needing_update:
        print("Products needing data:")
        for product in products_needing_update[:10]:
            years = years_to_download[product]
            cov = coverage_by_product[product]
            existing_years = [y for y, s in cov.years.items() if s.exists]
            print(f"  {product}: need {years} (have: {existing_years or 'none'})")
        if len(products_needing_update) > 10:
            print(f"  ... and {len(products_needing_update) - 10} more")
    else:
        print("All products are up to date!")
        if not args.force:
            print("Use --force to re-download anyway.")
            return 0

    # Cost estimation - ALWAYS estimate before any download
    print()
    print("Estimating download cost...")
    cost = estimate_cost(config, products, years_to_download)

    # If --estimate flag, show cost and exit
    if args.estimate:
        databento_estimate_only_notice(cost)
        return 0

    # If dry run, just show what would be done
    if args.dry_run:
        print(f"\n[DRY RUN] Estimated cost: ${cost:.2f}")
        print("[DRY RUN] Would download the products listed above.")
        return 0

    # Require explicit acknowledgment before paid download
    if not databento_acknowledge(cost, force=args.force):
        print("Download cancelled.")
        return 0

    # Download
    print()
    print("=" * 70)
    print("DOWNLOADING")
    print("=" * 70)

    if args.parallel > 1:
        # Efficient parallel download mode (1 API call per product)
        stats = download_parallel(
            products_needing_update,
            config,
            data_dir,
            workers=args.parallel,
            dry_run=args.dry_run,
        )
    else:
        # Efficient sequential download mode (1 API call per product)
        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "rows": 0}

        for product in products_needing_update:
            result = download_product_efficient(product, config, data_dir, dry_run=args.dry_run)

            if result["rows"] > 0:
                stats["downloaded"] += 1
                stats["rows"] += result["rows"]
            elif result["failed"] > 0:
                stats["failed"] += 1
            else:
                stats["skipped"] += 1

            for msg in result["messages"]:
                print(f"  {msg}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Products downloaded: {stats['downloaded']}")
    print(f"Total rows: {stats['rows']:,}")
    if "skipped" in stats:
        print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")

    return 0


if __name__ == "__main__":
    exit(main())
