#!/usr/bin/env python3
"""Download prediction market data from Kalshi and Polymarket.

Both providers are free (no API key required for read access).

The defaults live in `config.yaml`. The CLI flags below override the
YAML on a per-run basis when you want a wider window or a richer
universe without touching the config file.

Usage:
    python data/prediction_markets/download.py
    python data/prediction_markets/download.py --provider kalshi
    python data/prediction_markets/download.py --provider polymarket
    python data/prediction_markets/download.py --list
    python data/prediction_markets/download.py --dry-run

Coverage levers (all override config.yaml when set):
    --start-date 2024-01-01            broaden / narrow the window
    --end-date   2026-12-31
    --max-markets-per-category 10      Polymarket: more markets per category
    --categories crypto,economics      Polymarket: replace category list
    --search-results-per-query 25      Polymarket: deeper search per query
"""

from __future__ import annotations

import sys
from datetime import UTC
from pathlib import Path

import polars as pl

from utils.downloading import (
    atomic_write_parquet,
    create_base_parser,
    load_dotenv,
    load_section,
    print_download_summary,
    print_dry_run_notice,
    print_section,
    resolve_data_dir,
    resolve_storage_path,
    save_dataset_profile,
)


def _is_safe_market(market: dict, excluded_tags: set[str], excluded_keywords: set[str]) -> bool:
    """Filter out politically sensitive markets."""
    slug = market.get("slug", "").lower()
    question = market.get("question", "").lower()
    tags = {t.lower() for t in market.get("tags", [])}

    if tags & excluded_tags:
        return False
    text = f"{slug} {question}"
    return not any(kw in text for kw in excluded_keywords)


def _parse_kalshi_raw(raw_data: list[dict], ticker: str) -> pl.DataFrame:
    """Parse Kalshi raw API response into canonical OHLCV.

    The Kalshi API returns nested bid/ask OHLC instead of flat OHLCV.
    We extract yes_bid prices as the probability OHLCV (the price a YES
    contract trades at = implied probability).
    """
    from datetime import datetime, timezone

    rows = []
    for entry in raw_data:
        ts = entry.get("end_period_ts")
        if ts is None:
            continue

        # Extract yes_bid OHLC (implied probability of YES outcome)
        yes_bid = entry.get("yes_bid", {})
        if not yes_bid:
            continue

        try:
            row = {
                "timestamp": datetime.fromtimestamp(ts, tz=UTC).date(),
                "symbol": ticker,
                "open": float(yes_bid.get("open_dollars", 0)),
                "high": float(yes_bid.get("high_dollars", 0)),
                "low": float(yes_bid.get("low_dollars", 0)),
                "close": float(yes_bid.get("close_dollars", 0)),
                "volume": float(entry.get("volume_fp", 0)),
            }
            # Skip entries with no meaningful price data
            if row["close"] > 0 or row["open"] > 0:
                rows.append(row)
        except (ValueError, TypeError):
            continue

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Date))


def download_kalshi(
    config: dict,
    storage_path: Path,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
) -> pl.DataFrame | None:
    """Download Kalshi economic event series.

    Uses raw API data with custom parsing because the library's
    _transform_data expects flat OHLCV but Kalshi returns nested bid/ask.
    Returns None if the Kalshi provider is not installed.
    """
    try:
        from ml4t.data.providers.kalshi import KalshiProvider
    except ImportError:
        print("Kalshi provider not installed — skipping")
        return None

    series_list = config.get("series", [])
    if not series_list:
        print("No Kalshi series configured")
        return None

    print(f"\nKalshi series: {len(series_list)}")
    for s in series_list:
        print(f"  {s['ticker']:12s} — {s['description']}")

    if dry_run:
        return None

    provider = KalshiProvider()
    frames: list[pl.DataFrame] = []
    failed: list[str] = []

    print("\nDownloading Kalshi data...\n")
    for series_cfg in series_list:
        series_ticker = series_cfg["ticker"]
        description = series_cfg["description"]
        category = series_cfg.get("category", "unknown")
        print(f"  {series_ticker}...", end=" ", flush=True)

        try:
            markets = provider.list_markets(series_ticker=series_ticker, limit=50)
            if not markets:
                print("NO MARKETS")
                failed.append(series_ticker)
                continue

            series_frames = []
            for market in markets:
                ticker = market.get("ticker", "")
                try:
                    # Use raw data + custom parsing to work around library bug
                    raw = provider._fetch_raw_data(ticker, start_date, end_date, frequency="daily")
                    df = _parse_kalshi_raw(raw, ticker)
                    if not df.is_empty():
                        df = df.with_columns(pl.lit(category).alias("category"))
                        series_frames.append(df)
                except Exception:
                    continue

            if series_frames:
                combined = pl.concat(series_frames, how="diagonal_relaxed")
                frames.append(combined)
                total_rows = sum(len(f) for f in series_frames)
                print(f"OK ({total_rows:,} rows, {len(series_frames)} markets) [{description}]")
            else:
                print("EMPTY")
                failed.append(series_ticker)

        except Exception as exc:
            print(f"ERROR ({exc})")
            failed.append(series_ticker)

    provider.close()

    if not frames:
        print("WARNING: No Kalshi data downloaded")
        return None

    result = pl.concat(frames, how="diagonal_relaxed")

    keep_cols = [
        c
        for c in ["timestamp", "symbol", "open", "high", "low", "close", "volume", "category"]
        if c in result.columns
    ]
    result = result.select(keep_cols).sort(["symbol", "timestamp"])

    print(f"\nKalshi total: {len(result):,} rows, {result['symbol'].n_unique()} markets")
    if failed:
        print(f"Failed series: {', '.join(failed)}")

    return result


def download_polymarket(
    config: dict,
    storage_path: Path,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
) -> pl.DataFrame | None:
    """Download Polymarket event data (non-political only).

    Tries curated slugs first, then searches for markets by category.
    The Polymarket API may return historical markets — we take whatever
    has data in the configured date range.
    Returns None if the Polymarket provider is not installed.
    """
    try:
        from ml4t.data.providers.polymarket import PolymarketProvider
    except ImportError:
        print("Polymarket provider not installed — skipping")
        return None

    curated = config.get("curated_slugs", [])
    excluded_tags = set(config.get("excluded_tags", []))
    excluded_keywords = set(config.get("excluded_keywords", []))
    search_categories = config.get("categories", [])
    search_terms = config.get("search_terms", {})
    max_markets_per_category = int(config.get("max_markets_per_category", 5))
    search_results_per_query = int(config.get("search_results_per_query", 5))

    print(f"\nPolymarket curated events: {len(curated)}")
    for s in curated:
        print(f"  {s['slug'][:40]:40s} — {s['description']}")
    if search_categories:
        print(f"Search categories: {', '.join(search_categories)}")

    if dry_run:
        return None

    provider = PolymarketProvider()
    frames: list[pl.DataFrame] = []
    failed: list[str] = []

    print("\nDownloading Polymarket data...\n")

    # Try curated slugs first
    for event_cfg in curated:
        slug = event_cfg["slug"]
        description = event_cfg["description"]
        category = event_cfg.get("category", "unknown")
        print(f"  {slug[:40]}...", end=" ", flush=True)

        try:
            df = provider.fetch_ohlcv(slug, start_date, end_date, frequency="daily")
            if df.is_empty():
                print("EMPTY")
                failed.append(slug)
                continue

            df = df.with_columns(pl.lit(category).alias("category"))
            frames.append(df)
            print(f"OK ({len(df):,} rows) [{description}]")

        except Exception as exc:
            print(f"NOT FOUND ({exc})")
            failed.append(slug)

    # If curated slugs failed, search by category keywords
    if not frames and search_categories:
        print("\n  Curated slugs unavailable, searching by category...")
        seen_slugs: set[str] = set()
        for category in search_categories:
            queries = search_terms.get(category, [category])
            category_downloaded = 0
            try:
                for query in queries:
                    if category_downloaded >= max_markets_per_category:
                        break
                    markets = provider.search_markets(query)
                    safe = [
                        m for m in markets if _is_safe_market(m, excluded_tags, excluded_keywords)
                    ]
                    for market in safe[:search_results_per_query]:
                        if category_downloaded >= max_markets_per_category:
                            break
                        slug = market.get("slug", market.get("condition_id", ""))
                        if not slug or slug in seen_slugs:
                            continue
                        seen_slugs.add(slug)
                        question = market.get("question", "")[:50]
                        print(f"  [{category}:{query}] {slug[:40]}...", end=" ", flush=True)
                        try:
                            df = provider.fetch_ohlcv(slug, start_date, end_date, frequency="daily")
                            if not df.is_empty():
                                df = df.with_columns(pl.lit(category).alias("category"))
                                frames.append(df)
                                category_downloaded += 1
                                print(f"OK ({len(df):,} rows) [{question}]")
                            else:
                                print("EMPTY")
                        except Exception:
                            print("ERROR")
            except Exception as exc:
                print(f"  Search for '{category}' failed: {exc}")

    provider.close()

    if not frames:
        print("WARNING: No Polymarket data downloaded")
        return None

    result = pl.concat(frames, how="diagonal_relaxed")

    # Normalize to canonical schema
    renames = {}
    if "symbol" not in result.columns:
        for col in ["slug", "condition_id", "ticker"]:
            if col in result.columns:
                renames[col] = "symbol"
                break
    if renames:
        result = result.rename(renames)

    if "timestamp" in result.columns and result["timestamp"].dtype != pl.Date:
        result = result.cast({"timestamp": pl.Date})

    keep_cols = [
        c
        for c in ["timestamp", "symbol", "open", "high", "low", "close", "volume", "category"]
        if c in result.columns
    ]
    result = result.select(keep_cols).sort(["symbol", "timestamp"])

    print(f"\nPolymarket total: {len(result):,} rows, {result['symbol'].n_unique()} markets")
    if failed:
        print(f"Failed slugs: {', '.join(failed)}")

    return result


def build_metadata(kalshi_config: dict, polymarket_config: dict) -> pl.DataFrame:
    """Build metadata DataFrame for all configured markets."""
    rows = []

    for series in kalshi_config.get("series", []):
        rows.append(
            {
                "provider": "kalshi",
                "identifier": series["ticker"],
                "description": series["description"],
                "category": series.get("category", ""),
            }
        )

    for event in polymarket_config.get("curated_slugs", []):
        rows.append(
            {
                "provider": "polymarket",
                "identifier": event["slug"],
                "description": event["description"],
                "category": event.get("category", ""),
            }
        )

    return pl.DataFrame(rows)


def main() -> None:
    parser = create_base_parser("Download prediction market data (Kalshi + Polymarket)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to prediction markets config",
    )
    parser.add_argument(
        "--provider",
        choices=["kalshi", "polymarket"],
        help="Download from a single provider only",
    )
    parser.add_argument("--list", action="store_true", help="List configured markets")
    parser.add_argument(
        "--start-date",
        help="Override config 'start' (YYYY-MM-DD). Applies to both providers.",
    )
    parser.add_argument(
        "--end-date",
        help="Override config 'end' (YYYY-MM-DD). Applies to both providers.",
    )
    parser.add_argument(
        "--max-markets-per-category",
        type=int,
        help="Polymarket: cap on markets downloaded per category (overrides config).",
    )
    parser.add_argument(
        "--categories",
        help=(
            "Polymarket: comma-separated category list (overrides config). "
            "Each category must have an entry under polymarket.search_terms."
        ),
    )
    parser.add_argument(
        "--search-results-per-query",
        type=int,
        help=(
            "Polymarket: number of search results inspected per query before "
            "the per-category cap kicks in (overrides config; default 5)."
        ),
    )
    args = parser.parse_args()

    load_dotenv()

    config = load_section(args.config, "prediction_markets")
    kalshi_config = config.get("kalshi", {})
    polymarket_config = dict(config.get("polymarket", {}))  # copy — we may mutate

    # Apply Polymarket-specific CLI overrides into the config dict so
    # downstream callers see a single resolved view.
    if args.max_markets_per_category is not None:
        polymarket_config["max_markets_per_category"] = args.max_markets_per_category
    if args.search_results_per_query is not None:
        polymarket_config["search_results_per_query"] = args.search_results_per_query
    if args.categories is not None:
        polymarket_config["categories"] = [
            c.strip() for c in args.categories.split(",") if c.strip()
        ]

    if args.list:
        print("Configured Kalshi series:")
        for s in kalshi_config.get("series", []):
            print(f"  {s['ticker']:12s} — {s['description']}")
        print("\nConfigured Polymarket events:")
        for s in polymarket_config.get("curated_slugs", []):
            print(f"  {s['slug']:40s} — {s['description']}")
        return

    if args.dry_run:
        print_dry_run_notice()

    data_root = resolve_data_dir(args.data_path)
    storage_path = resolve_storage_path(data_root, config.get("storage_path"), "prediction_markets")
    outputs = config.get("outputs", {})
    kalshi_path = storage_path / str(outputs.get("kalshi_file", "kalshi_events.parquet"))
    polymarket_path = storage_path / str(
        outputs.get("polymarket_file", "polymarket_events.parquet")
    )
    metadata_path = storage_path / str(
        outputs.get("metadata_file", "prediction_markets_metadata.parquet")
    )
    start_date = args.start_date or str(config.get("start", "2024-01-01"))
    end_date = args.end_date or str(config.get("end", "2025-12-31"))

    print_section("PREDICTION MARKETS DOWNLOAD")
    print(f"Config: {args.config}")
    print(f"Output: {storage_path}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Kalshi series: {len(kalshi_config.get('series', []))}")
    print(f"Polymarket events: {len(polymarket_config.get('curated_slugs', []))}")
    print(
        "Polymarket caps: "
        f"max_per_category={polymarket_config.get('max_markets_per_category', 5)}, "
        f"results_per_query={polymarket_config.get('search_results_per_query', 5)}, "
        f"categories={polymarket_config.get('categories', [])}"
    )

    if args.dry_run:
        print_download_summary(
            {
                "kalshi_series": len(kalshi_config.get("series", [])),
                "polymarket_events": len(polymarket_config.get("curated_slugs", [])),
                "start_date": start_date,
                "end_date": end_date,
                "kalshi_file": str(kalshi_path),
                "polymarket_file": str(polymarket_path),
            },
            dry_run=True,
        )
        return

    storage_path.mkdir(parents=True, exist_ok=True)
    summary = {}

    # Download Kalshi
    if args.provider in (None, "kalshi"):
        print_section("KALSHI (CFTC-regulated, free API)")
        kalshi_df = download_kalshi(kalshi_config, storage_path, start_date, end_date)
        if kalshi_df is not None and not kalshi_df.is_empty():
            atomic_write_parquet(kalshi_df, kalshi_path)
            profile_path = save_dataset_profile(
                kalshi_df,
                kalshi_path,
                source="KalshiProvider",
                timestamp_col="timestamp",
                symbol_col="symbol",
            )
            summary["kalshi_rows"] = len(kalshi_df)
            summary["kalshi_markets"] = kalshi_df["symbol"].n_unique()
            summary["kalshi_file"] = str(kalshi_path)
            summary["kalshi_profile"] = str(profile_path)
        else:
            summary["kalshi"] = "no data"

    # Download Polymarket
    if args.provider in (None, "polymarket"):
        print_section("POLYMARKET (crypto, free API)")
        polymarket_df = download_polymarket(polymarket_config, storage_path, start_date, end_date)
        if polymarket_df is not None and not polymarket_df.is_empty():
            atomic_write_parquet(polymarket_df, polymarket_path)
            profile_path = save_dataset_profile(
                polymarket_df,
                polymarket_path,
                source="PolymarketProvider",
                timestamp_col="timestamp",
                symbol_col="symbol",
            )
            summary["polymarket_rows"] = len(polymarket_df)
            summary["polymarket_markets"] = polymarket_df["symbol"].n_unique()
            summary["polymarket_file"] = str(polymarket_path)
            summary["polymarket_profile"] = str(profile_path)
        else:
            summary["polymarket"] = "no data"

    # Save metadata
    metadata_df = build_metadata(kalshi_config, polymarket_config)
    atomic_write_parquet(metadata_df, metadata_path)
    summary["metadata_file"] = str(metadata_path)

    # Kalshi (CFTC-regulated) and Polymarket restrict access by jurisdiction, so
    # "list markets" can fail at the network layer from some regions even though
    # the endpoints are up. This dataset is optional; note it rather than failing.
    # Fire the note whenever nothing was retrieved from the providers actually
    # attempted (success sets *_rows; a bare "kalshi"/"polymarket" key means that
    # provider was tried and returned no data), covering the single-provider case.
    retrieved_any = "kalshi_rows" in summary or "polymarket_rows" in summary
    no_data_reported = summary.get("kalshi") == "no data" or summary.get("polymarket") == "no data"
    if no_data_reported and not retrieved_any:
        print(
            "\nNote: no prediction-market data was retrieved. Kalshi and Polymarket "
            "geo-restrict API access, so this can happen outside their supported "
            "regions. It is optional and does not affect the core datasets."
        )

    print_download_summary(summary)


if __name__ == "__main__":
    main()
