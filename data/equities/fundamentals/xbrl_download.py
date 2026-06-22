#!/usr/bin/env python3
"""Download SEC XBRL fundamentals from EDGAR.

The SEC XBRL Frames API exposes cross-sectional snapshots of every
filer's reported XBRL concepts for a given calendar period — far cheaper
than per-filing XML parsing when you need bulk fundamentals for factor
engineering.

This downloader fetches a fixed set of balance-sheet, income-statement,
and cash-flow concepts across a CIK/year grid, joins filing dates from
the Submissions API for point-in-time correctness, and writes a single
canonical parquet that Chapter 4 NB 04 (`04_sec_xbrl_fundamentals.py`)
and any downstream factor notebook can consume via the
`load_sec_xbrl_fundamentals()` loader.

Output layout under `$ML4T_DATA_PATH/equities/fundamentals/xbrl/`:

    fundamentals.parquet                 joined panel (all requested
                                         concepts × CIKs × quarters)
    filing_dates/CIK{cik}.parquet        per-CIK accession → filing_date
                                         cache (avoids re-hitting
                                         Submissions API on re-runs)

Schema of ``fundamentals.parquet``:

    symbol                ticker from the CIK→symbol mapping (or empty
                          string for CIKs passed via --ciks that we
                          don't know the ticker for)
    cik                   int
    entity_name           SEC's reported entity name
    fiscal_quarter_end    date  (XBRL "end" field)
    announcement_date     date  (filing date from Submissions API)
    accession             str   (SEC accession number)
    <concept>_lower       one numeric column per fetched concept
                          (e.g. `assets`, `revenues`, `netincomeloss`)

Usage:
    # Default: 20 major US equities, 2022-2024, 11 standard concepts
    python data/equities/fundamentals/xbrl_download.py

    # Wider or narrower year range
    python data/equities/fundamentals/xbrl_download.py --years 2020,2021,2022,2023,2024

    # Restrict to a single company (CIK)
    python data/equities/fundamentals/xbrl_download.py --ciks 320193

    # Concept override (semicolon-free comma list; instant vs duration
    # is auto-detected from the concept list below)
    python data/equities/fundamentals/xbrl_download.py --concepts Assets,Revenues,NetIncomeLoss

    # Refresh cached filing dates (Submissions API)
    python data/equities/fundamentals/xbrl_download.py --force-filing-dates

Rate limits: SEC EDGAR caps at 10 req/sec. We sleep 0.2s between
requests — conservative and matches the rest of the SEC downloaders.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
import requests

from utils.downloading import resolve_data_dir

XBRL_FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
HEADERS = {"User-Agent": "ML4T Book stefan@ml4trading.io"}

REQUEST_TIMEOUT = 30
RATE_LIMIT_S = 0.2

# Default CIK → ticker mapping for the 20 large-cap companies used by
# Chapter 4 NB 04. --ciks overrides this list; unknown CIKs pass
# through with an empty symbol.
DEFAULT_CIK_TO_TICKER: dict[int, str] = {
    320193: "AAPL",
    789019: "MSFT",
    1652044: "GOOGL",
    1018724: "AMZN",
    1326801: "META",
    1045810: "NVDA",
    1318605: "TSLA",
    19617: "JPM",
    200406: "JNJ",
    1403161: "V",
    732717: "COST",
    1800: "ABT",
    2488: "AMD",
    886982: "GS",
    70858: "BAC",
    51143: "IBM",
    34088: "XOM",
    93410: "CVX",
    21344: "KO",
    78003: "PFE",
}

# Instant concepts: balance-sheet items. Period suffix is "I".
DEFAULT_INSTANT_CONCEPTS: list[str] = [
    "Assets",
    "StockholdersEquity",
    "LongTermDebt",
    "CashAndCashEquivalentsAtCarryingValue",
    "Liabilities",
]

# Duration concepts: income-statement + cash-flow items. No suffix.
DEFAULT_DURATION_CONCEPTS: list[str] = [
    "Revenues",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "GrossProfit",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
]

DEFAULT_CONCEPTS: list[str] = DEFAULT_INSTANT_CONCEPTS + DEFAULT_DURATION_CONCEPTS
DEFAULT_YEARS: list[int] = [2022, 2023, 2024]


def _fetch_xbrl_frame(
    taxonomy: str, concept: str, unit: str, period: str, verbose: bool = False
) -> list[dict] | None:
    """Fetch one XBRL frame (one concept × one calendar period)."""
    url = XBRL_FRAMES_URL.format(taxonomy=taxonomy, concept=concept, unit=unit, period=period)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        if verbose:
            print(f"    {concept}/{period}: request error ({e})")
        return None
    if resp.status_code == 404:
        # Not all concepts are reported every quarter (e.g. GrossProfit
        # for banks). 404 is a normal signal, not a failure.
        return []
    if resp.status_code != 200:
        if verbose:
            print(f"    {concept}/{period}: HTTP {resp.status_code}")
        return []
    try:
        return resp.json().get("data", [])
    except ValueError:
        return []


def _fetch_filing_dates(cik: int) -> dict[str, str]:
    """Fetch accession → filing_date map for one CIK via Submissions API."""
    url = SUBMISSIONS_URL.format(cik_padded=str(cik).zfill(10))
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except (requests.RequestException, ValueError):
        return {}
    filings = data.get("filings", {}).get("recent", {})
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    return dict(zip(accessions, dates, strict=False))


def _load_cached_filing_dates(cache_dir: Path, cik: int) -> dict[str, str] | None:
    path = cache_dir / f"CIK{cik}.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    return dict(zip(df["accession"].to_list(), df["filing_date"].to_list(), strict=True))


def _save_cached_filing_dates(cache_dir: Path, cik: int, mapping: dict[str, str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"CIK{cik}.parquet"
    pl.DataFrame(
        {
            "accession": list(mapping.keys()),
            "filing_date": list(mapping.values()),
        }
    ).write_parquet(path)


def _build_fundamentals(
    target_ciks: set[int],
    instant_concepts: list[str],
    duration_concepts: list[str],
    years: list[int],
    verbose: bool = False,
) -> pl.DataFrame:
    """Fetch all requested concept/quarter pairs, filter to target CIKs."""
    # (cik, period_end) → record dict accumulating concepts per (cik, quarter)
    records: dict[tuple[int, str], dict] = {}

    periods = [(y, q) for y in years for q in range(1, 5)]
    total = len(periods)

    for i, (year, quarter) in enumerate(periods, 1):
        print(f"  [{i}/{total}] {year}Q{quarter}…", end="", flush=True)
        items = 0

        instant_period = f"CY{year}Q{quarter}I"
        for concept in instant_concepts:
            data = _fetch_xbrl_frame("us-gaap", concept, "USD", instant_period, verbose)
            time.sleep(RATE_LIMIT_S)
            if not data:
                continue
            for r in data:
                cik = r.get("cik")
                if cik not in target_ciks:
                    continue
                key = (cik, r.get("end"))
                rec = records.setdefault(
                    key,
                    {
                        "cik": cik,
                        "entity_name": r.get("entityName"),
                        "period_end": r.get("end"),
                        "accession": r.get("accn"),
                    },
                )
                rec[concept.lower()] = r.get("val")
                items += 1

        duration_period = f"CY{year}Q{quarter}"
        for concept in duration_concepts:
            data = _fetch_xbrl_frame("us-gaap", concept, "USD", duration_period, verbose)
            time.sleep(RATE_LIMIT_S)
            if not data:
                continue
            for r in data:
                cik = r.get("cik")
                if cik not in target_ciks:
                    continue
                key = (cik, r.get("end"))
                rec = records.setdefault(
                    key,
                    {
                        "cik": cik,
                        "entity_name": r.get("entityName"),
                        "period_end": r.get("end"),
                        "accession": r.get("accn"),
                    },
                )
                rec[concept.lower()] = r.get("val")
                items += 1

        print(f" ({items} items)")

    if not records:
        return pl.DataFrame()

    return pl.DataFrame(list(records.values()))


def _add_filing_dates(
    df: pl.DataFrame,
    cache_dir: Path,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Join announcement dates onto the fundamentals panel."""
    if df.is_empty():
        return df

    ciks = df.select("cik").unique().to_series().to_list()
    mapping: dict[str, str] = {}

    print(f"  Filing dates for {len(ciks)} CIK(s):")
    for i, cik in enumerate(ciks, 1):
        cached = None if force_refresh else _load_cached_filing_dates(cache_dir, cik)
        if cached is not None:
            print(f"    [{i}/{len(ciks)}] CIK {cik}: cached ({len(cached)} filings)")
            mapping.update(cached)
            continue

        dates = _fetch_filing_dates(cik)
        time.sleep(RATE_LIMIT_S)
        print(f"    [{i}/{len(ciks)}] CIK {cik}: fetched ({len(dates)} filings)")
        if dates:
            _save_cached_filing_dates(cache_dir, cik, dates)
            mapping.update(dates)

    if not mapping:
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("filing_date"))

    map_df = pl.DataFrame(
        {
            "accession": list(mapping.keys()),
            "filing_date": list(mapping.values()),
        }
    )
    return df.join(map_df, on="accession", how="left")


def _finalize(df: pl.DataFrame, cik_to_ticker: dict[int, str]) -> pl.DataFrame:
    """Normalize column types + order for the canonical parquet."""
    if df.is_empty():
        return df

    df = df.with_columns(
        pl.col("cik").cast(pl.Int64).replace_strict(cik_to_ticker, default="").alias("symbol"),
        pl.col("period_end").str.to_date().alias("fiscal_quarter_end"),
    )

    if "filing_date" in df.columns:
        df = df.with_columns(
            pl.col("filing_date").str.to_date(strict=False).alias("announcement_date")
        ).drop("filing_date")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Date).alias("announcement_date"))

    leading = [
        "symbol",
        "cik",
        "entity_name",
        "fiscal_quarter_end",
        "announcement_date",
        "accession",
    ]
    concept_cols = [c for c in df.columns if c not in leading and c != "period_end"]
    return df.select(leading + sorted(concept_cols)).sort(["symbol", "fiscal_quarter_end"])


def _split_concepts(concepts: list[str]) -> tuple[list[str], list[str]]:
    """Split a user-provided concept list into (instant, duration) buckets.

    Concepts in DEFAULT_INSTANT_CONCEPTS are instant; everything else is
    duration. This matches XBRL reality for the concepts we ship with.
    Users who want to add a new instant concept can extend
    DEFAULT_INSTANT_CONCEPTS.
    """
    known_instant = set(DEFAULT_INSTANT_CONCEPTS)
    instant = [c for c in concepts if c in known_instant]
    duration = [c for c in concepts if c not in known_instant]
    return instant, duration


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SEC XBRL fundamentals (Frames + Submissions APIs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=",".join(DEFAULT_CONCEPTS),
        help=(
            "Comma-separated us-gaap concept names (default: "
            + ", ".join(DEFAULT_CONCEPTS)
            + "). Instant/duration split auto-detected."
        ),
    )
    parser.add_argument(
        "--years",
        type=str,
        default=",".join(str(y) for y in DEFAULT_YEARS),
        help=f"Comma-separated calendar years (default: {','.join(str(y) for y in DEFAULT_YEARS)})",
    )
    parser.add_argument(
        "--ciks",
        type=str,
        default=None,
        help=(
            "Comma-separated CIKs (default: 20 large-cap US equities). "
            "Unknown CIKs pass through with an empty symbol column."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override output root (default: $ML4T_DATA_PATH)",
    )
    parser.add_argument(
        "--force-filing-dates",
        action="store_true",
        help="Refresh the per-CIK filing-date cache instead of reusing it",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    if not concepts:
        print("ERROR: --concepts produced an empty list")
        return 1
    instant, duration = _split_concepts(concepts)

    try:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    except ValueError:
        print("ERROR: --years must be a comma-separated list of integer years")
        return 1

    if args.ciks:
        try:
            target_ciks = {int(c.strip()) for c in args.ciks.split(",") if c.strip()}
        except ValueError:
            print("ERROR: --ciks must be a comma-separated list of integers")
            return 1
    else:
        target_ciks = set(DEFAULT_CIK_TO_TICKER.keys())

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "equities" / "fundamentals" / "xbrl"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "filing_dates"

    print()
    print(f"Output:         {output_dir}")
    print(f"Years:          {years}")
    print(f"CIKs:           {len(target_ciks)}")
    print(f"Instant:        {len(instant)} concepts")
    print(f"Duration:       {len(duration)} concepts")
    print(f"Filing cache:   {cache_dir}")
    print()

    print("Fetching XBRL frames…")
    raw = _build_fundamentals(target_ciks, instant, duration, years, verbose=args.verbose)
    if raw.is_empty():
        print("No records fetched — nothing to write.")
        return 1

    print()
    print("Joining filing dates…")
    with_dates = _add_filing_dates(raw, cache_dir, force_refresh=args.force_filing_dates)

    panel = _finalize(with_dates, DEFAULT_CIK_TO_TICKER)

    out_path = output_dir / "fundamentals.parquet"
    panel.write_parquet(out_path)

    print()
    print(f"  → {out_path.name}")
    print(f"    rows:     {len(panel):,}")
    print(f"    CIKs:     {panel.select('cik').n_unique()}")
    print(f"    quarters: {panel.select('fiscal_quarter_end').n_unique()}")
    if "announcement_date" in panel.columns:
        have_date = panel.filter(pl.col("announcement_date").is_not_null()).height
        print(f"    filed:    {have_date:,} rows with announcement_date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
