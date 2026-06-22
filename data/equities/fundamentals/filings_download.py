#!/usr/bin/env python3
"""
Download SEC filings (10-Q, 10-K, 8-K) from EDGAR and extract text.

Unified downloader for all SEC filing types used in ML4T. All three forms
write the **same canonical schema**; readers pick the form they need via
``load_sec_filings(form_type=...)``. Each form has its own text extraction:

  - 10-Q: MD&A section (Part I, Item 2) — structured quarterly narrative
  - 10-K: Supplier-related excerpt or mid-document text — annual context
  - 8-K:  Short excerpt (first 8000 chars) — discrete corporate events

Uses edgartools for EDGAR access. SEC/EDGAR data is public domain.

Usage:
    # Download 10-Q MD&A for S&P 500 (~4-5 hours; full production run)
    python filings_download.py --form 10-Q --universe sp500 --years 2017-2021

    # Download 10-K + 8-K for S&P 100
    python filings_download.py --form 10-K --universe sp100 --years 2020-2025
    python filings_download.py --form 8-K --universe sp100 --years 2020-2025

    # Quick test
    python filings_download.py --form 10-K --universe sp100 --sample 5
    python filings_download.py --form 10-Q --universe sp500 --sample 20

Output (form-first layout under $ML4T_DATA_PATH/equities/fundamentals/):
    {form}/{universe}/reference/all_{form}_filings.parquet

Canonical schema (identical across 10-K, 10-Q, 8-K):
    symbol        str    — stock ticker (canonical entity column)
    cik           str    — 10-digit zero-padded SEC CIK
    form          str    — exact form name ("10-K", "10-Q", "8-K", "8-K/A", ...)
    filing_date   Date   — filing date (point-in-time correct)
    period_end    Date?  — fiscal period end (nullable for 8-K)
    accession_no  str    — SEC accession number (XXXXXXXXXX-XX-XXXXXX)
    company_name  str
    year          int    — derived: filing_date.year (kept for convenience)
    text          str    — extracted body
    text_length   int

Loader: ``data.load_sec_filings(form_type, universe, ...)``.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from utils.downloading import atomic_write_parquet, print_section, resolve_data_dir

# ---------------------------------------------------------------------------
# Symbol universes
# ---------------------------------------------------------------------------

SP100_TICKERS = sorted(
    [
        "AAPL",
        "ABBV",
        "ABT",
        "ACN",
        "ADBE",
        "AIG",
        "AMD",
        "AMGN",
        "AMT",
        "AMZN",
        "AVGO",
        "AXP",
        "BA",
        "BAC",
        "BK",
        "BKNG",
        "BLK",
        "BMY",
        "BRK.B",
        "C",
        "CAT",
        "CHTR",
        "CL",
        "CMCSA",
        "COF",
        "COP",
        "COST",
        "CRM",
        "CSCO",
        "CVS",
        "CVX",
        "DE",
        "DHR",
        "DIS",
        "DOW",
        "DUK",
        "EMR",
        "EXC",
        "F",
        "FDX",
        "GD",
        "GE",
        "GILD",
        "GM",
        "GOOG",
        "GOOGL",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KHC",
        "KO",
        "LIN",
        "LLY",
        "LMT",
        "LOW",
        "MA",
        "MCD",
        "MDLZ",
        "MDT",
        "MET",
        "META",
        "MMM",
        "MO",
        "MRK",
        "MS",
        "MSFT",
        "NEE",
        "NFLX",
        "NKE",
        "NVDA",
        "ORCL",
        "PEP",
        "PFE",
        "PG",
        "PM",
        "PYPL",
        "QCOM",
        "RTX",
        "SBUX",
        "SCHW",
        "SO",
        "SPG",
        "T",
        "TGT",
        "TMO",
        "TMUS",
        "TSLA",
        "TXN",
        "UNH",
        "UNP",
        "UPS",
        "USB",
        "V",
        "VZ",
        "WFC",
        "WMT",
        "XOM",
    ]
)


def get_universe_symbols(universe: str, data_path: Path) -> list[str]:
    """Resolve symbol list for a universe identifier."""
    if universe == "sp100":
        return SP100_TICKERS

    if universe == "sp500":
        import polars as pl

        bars_path = data_path / "equities" / "market" / "sp500" / "daily_bars.parquet"
        if bars_path.exists():
            df = pl.read_parquet(bars_path, columns=["symbol"])
            symbols = sorted(df["symbol"].unique().to_list())
            print(f"Loaded {len(symbols)} symbols from AlgoSeek daily bars")
            return symbols
        print("WARNING: AlgoSeek data not found, using S&P 100 as fallback")
        return SP100_TICKERS

    raise ValueError(f"Unknown universe: {universe}. Use sp100 or sp500.")


# ---------------------------------------------------------------------------
# Text extraction (form-specific)
# ---------------------------------------------------------------------------

MDA_START_PATTERNS = [
    r"^\s*ITEM\s*2\.?\s*[-\u2013\u2014]?\s*MANAGEMENT(?:'|\u2019)?\s*S?\s*DISCUSSION",
    r"^\s*Item\s*2\.?\s*[-\u2013\u2014]?\s*Management(?:'|\u2019)?\s*s?\s*Discussion",
]
MDA_END_PATTERNS = [r"^\s*ITEM\s*3\b", r"^\s*Item\s*3\b"]


def clean_text(text: str) -> str:
    """Clean extracted text while preserving paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?im)^\s*table of contents\s*$", "", text)
    text = re.sub(r"(?im)^\s*page\s+\d+\s*$", "", text)
    text = re.sub(r"(?im)^\s*\d+\s*of\s*\d+\s*$", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[_=\-]{3,}", " ", text)
    text = re.sub(r"[\u2022\u25cf\u25e6\u25aa]", " ", text)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_10q_mda(text: str) -> str | None:
    """Extract MD&A section from 10-Q filing text."""
    if not text or len(text) < 500:
        return None
    flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
    start_pos = None
    for pattern in MDA_START_PATTERNS:
        matches = list(re.finditer(pattern, text, flags))
        if matches:
            start_pos = matches[-1].end()
            break
    if start_pos is None:
        return None
    end_pos = len(text)
    for pattern in MDA_END_PATTERNS:
        match = re.search(pattern, text[start_pos:], flags)
        if match:
            end_pos = start_pos + match.start()
            break
    cleaned = clean_text(text[start_pos:end_pos])
    return cleaned if len(cleaned.split()) >= 200 else None


def extract_10k_excerpt(text: str) -> str:
    """Extract supplier-related section from 10-K, or mid-document fallback."""
    supplier_idx = text.lower().find("supplier")
    if supplier_idx > 0:
        start = max(0, supplier_idx - 3000)
        return text[start : start + 12000]
    if len(text) > 35000:
        return text[20000:35000]
    return text[:15000]


def extract_8k_excerpt(text: str) -> str:
    """First 8000 chars of 8-K filing text."""
    return text[:8000] if len(text) > 8000 else text


def get_filing_text(filing) -> str:
    """Get filing text, with HTML fallback."""
    try:
        text = filing.text()
        if text and len(text) > 1000:
            return text
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup

        html = filing.html()
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            return soup.get_text(separator="\n")
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------


def download_filings(
    symbol: str,
    form_type: str,
    years: list[int],
    cache_dir: Path | None = None,
    max_filings: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Download filings for one symbol. Returns list of record dicts."""
    from edgar import Company

    records = []
    try:
        company = Company(symbol)
    except Exception as e:
        if verbose:
            print(f"  {symbol}: Company lookup failed - {e}")
        return records

    try:
        filings = company.get_filings(form=form_type, amendments=False)
    except Exception as e:
        if verbose:
            print(f"  {symbol}: get_filings failed - {e}")
        return records

    count = 0
    for filing in filings:
        year = filing.filing_date.year
        if year not in years:
            continue
        if max_filings > 0 and count >= max_filings:
            break

        # Check per-filing cache
        if cache_dir:
            cache_path = (
                cache_dir / symbol / f"{year}_{filing.accession_no.replace('-', '')[:12]}.parquet"
            )
            if cache_path.exists():
                import polars as pl

                cached = pl.read_parquet(cache_path).to_dicts()
                records.extend(cached)
                count += len(cached)
                continue

        try:
            text = get_filing_text(filing)
            if not text:
                continue

            # Form-specific extraction — all three forms produce the SAME schema:
            #   symbol, cik, form, filing_date, period_end, accession_no,
            #   company_name, year, text, text_length
            if form_type == "10-Q":
                extracted = extract_10q_mda(text)
                if not extracted:
                    continue
            elif form_type == "10-K":
                extracted = extract_10k_excerpt(text)
            elif form_type == "8-K":
                extracted = extract_8k_excerpt(text)
            else:
                continue

            period_end = getattr(filing, "period_of_report", None)
            record = {
                "symbol": symbol,
                "cik": str(filing.cik).zfill(10),
                "form": filing.form,
                "filing_date": str(filing.filing_date),
                "period_end": str(period_end) if period_end else None,
                "accession_no": filing.accession_no,
                "company_name": company.name,
                "year": year,
                "text": extracted,
                "text_length": len(extracted),
            }

            records.append(record)
            count += 1

            # Cache individual filing
            if cache_dir:
                import polars as pl

                cache_path = (
                    cache_dir
                    / symbol
                    / f"{year}_{filing.accession_no.replace('-', '')[:12]}.parquet"
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                pl.DataFrame([record]).write_parquet(cache_path)

            time.sleep(0.15)

        except Exception as e:
            if verbose:
                print(f"  {symbol} ({filing.filing_date}): extraction failed - {e}")
            time.sleep(0.15)

    return records


# ---------------------------------------------------------------------------
# Progress / checkpoint
# ---------------------------------------------------------------------------


def load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "records": []}


def save_progress(path: Path, progress: dict):
    with open(path, "w") as f:
        json.dump(progress, f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download SEC filings from EDGAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--form", required=True, choices=["10-Q", "10-K", "8-K"], help="SEC form type"
    )
    parser.add_argument(
        "--universe",
        default="sp100",
        choices=["sp100", "sp500"],
        help="Symbol universe (default: sp100)",
    )
    parser.add_argument(
        "--years", default="2020-2025", help="Year range, e.g. 2020-2025 (default: 2020-2025)"
    )
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=0, help="Download only N tickers (0 = all)")
    parser.add_argument(
        "--max-filings",
        type=int,
        default=0,
        help="Max filings per symbol (0 = all, the default; cap explicitly if needed)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Disable per-filing cache")
    args = parser.parse_args()

    # Parse year range
    if "-" in args.years:
        start_year, end_year = map(int, args.years.split("-"))
    else:
        start_year = end_year = int(args.years)
    years = list(range(start_year, end_year + 1))

    # Resolve paths (form-first layout: equities/fundamentals/{form}/{universe}/…)
    data_path = resolve_data_dir(args.data_path)
    sec_dir = data_path / "equities" / "fundamentals"
    sec_dir.mkdir(parents=True, exist_ok=True)

    form_slug = args.form.lower().replace("-", "")
    form_dir = sec_dir / form_slug / args.universe
    form_dir.mkdir(parents=True, exist_ok=True)
    # Unified output path — same canonical schema across all three forms.
    reference_dir = form_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    output_file = reference_dir / f"all_{form_slug}_filings.parquet"
    progress_file = sec_dir / f".{args.universe}_{form_slug}_progress.json"
    cache_dir = (sec_dir / "cache" / f"{args.universe}_{form_slug}") if not args.no_cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # 0 = all filings in the year window; matches user expectation of
    # "SP100 download" meaning every SP100 filing, not a per-symbol cap.
    max_filings = args.max_filings

    # Get symbols
    symbols = get_universe_symbols(args.universe, data_path)
    if args.sample > 0:
        symbols = symbols[: args.sample]

    print_section(f"SEC {args.form} DOWNLOAD ({args.universe.upper()})")
    print()
    print(f"Symbols:    {len(symbols)}")
    print(f"Years:      {start_year}-{end_year}")
    print(f"Output:     {output_file}")
    if cache_dir:
        print(f"Cache:      {cache_dir}")
    print()

    if args.dry_run:
        print("[DRY RUN] No files created")
        return

    # Init edgartools
    try:
        from edgar import set_identity

        set_identity("ML4T Book stefan@ml4trading.io")
    except ImportError:
        print("ERROR: edgartools not installed. Run: pip install edgartools")
        sys.exit(1)

    # Resume
    if args.resume:
        progress = load_progress(progress_file)
        completed = set(progress["completed"])
        all_records = progress["records"]
        remaining = [s for s in symbols if s not in completed]
        print(f"Resuming: {len(completed)} done, {len(remaining)} remaining")
    else:
        completed = set()
        all_records = []
        remaining = symbols

    # Download
    t0 = time.time()
    for i, symbol in enumerate(remaining):
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1) * 3600
        print(
            f"[{i + 1}/{len(remaining)}] {symbol:6s} ({len(all_records):,} filings, {rate:.0f} sym/hr)"
        )

        records = download_filings(
            symbol,
            args.form,
            years,
            cache_dir=cache_dir,
            max_filings=max_filings,
            verbose=args.verbose,
        )
        if records:
            all_records.extend(records)
            print(f"  -> {len(records)} filings")

        completed.add(symbol)
        if (i + 1) % 25 == 0:
            save_progress(progress_file, {"completed": list(completed), "records": all_records})

    if not all_records:
        print("No filings extracted")
        sys.exit(1)

    # Save
    import polars as pl

    df = pl.DataFrame(all_records)
    if "filing_date" in df.columns:
        df = df.with_columns(pl.col("filing_date").cast(pl.Utf8).str.to_date())
    if "period_end" in df.columns:
        df = df.with_columns(pl.col("period_end").cast(pl.Utf8).str.to_date(strict=False))
    # Dedup: cache-replay + main-loop append can both write the same filing.
    pre_dedup = df.height
    df = df.unique(subset=["symbol", "accession_no"], keep="first")
    if df.height < pre_dedup:
        print(f"Deduped {pre_dedup - df.height} duplicate (symbol, accession_no) rows")
    df = df.sort(["symbol", "filing_date"])
    atomic_write_parquet(df, output_file)

    if progress_file.exists():
        progress_file.unlink()

    elapsed = time.time() - t0
    print_section("COMPLETE")
    print(f"Filings: {len(df):,}")
    print(f"Symbols: {df['symbol'].n_unique()}")
    print(f"Output:  {output_file} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Time:    {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
