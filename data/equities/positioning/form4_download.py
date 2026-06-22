#!/usr/bin/env python3
"""Download SEC Form 4 insider-transaction filings from EDGAR.

Form 4 reports must be filed within 2 business days of an insider's trade
and are small XML documents containing the issuer, reporting owner, and
one or more transaction rows. They are a first-class alternative dataset
for detecting informed trading activity.

This downloader fetches raw XML filings so Chapter 4 NB 03
(`03_sec_form4_insider_transactions.py`) can demonstrate text-based XML parsing
without an intermediate transformation step.

Uses edgartools for EDGAR access. SEC/EDGAR data is public domain.

Usage:
    # Single ticker, 20 most recent filings
    python data/equities/positioning/form4_download.py --ticker TSLA --count 20

    # Multiple tickers, 10 filings each
    python data/equities/positioning/form4_download.py --ticker TSLA,AAPL,MSFT --count 10

    # All available Form 4 filings for a ticker
    python data/equities/positioning/form4_download.py --ticker TSLA --count 0

Output:
    $ML4T_DATA_PATH/equities/positioning/form4/<TICKER>/<accession>.xml

    Each file contains the primary Form 4 XML document. Re-running the
    script skips filings already on disk.

Requirements:
    - edgartools: pip install edgartools
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from utils.downloading import print_section, resolve_data_dir


def download_form4_for_ticker(
    symbol: str,
    output_dir: Path,
    count: int,
    rate_limit_s: float = 0.2,
    verbose: bool = False,
) -> tuple[int, int]:
    """Fetch Form 4 filings for one ticker.

    Writes raw XML to `output_dir/<TICKER>/<accession>.xml`. Returns a tuple
    of (newly_written, already_on_disk) file counts.
    """
    from edgar import Company

    try:
        company = Company(symbol)
    except Exception as e:
        print(f"  {symbol}: company lookup failed — {e}")
        return (0, 0)

    try:
        filings = company.get_filings(form="4", amendments=False)
    except Exception as e:
        print(f"  {symbol}: get_filings failed — {e}")
        return (0, 0)

    ticker_dir = output_dir / symbol.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    seen = 0

    for filing in filings:
        if count > 0 and seen >= count:
            break
        seen += 1

        accession = filing.accession_no.replace("-", "")
        out_path = ticker_dir / f"{accession}.xml"

        if out_path.exists():
            skipped += 1
            if verbose:
                print(f"    {accession}: already on disk")
            continue

        try:
            content = filing.xml()
        except Exception as e:
            if verbose:
                print(f"    {accession}: filing.xml() failed — {e}")
            continue

        if not content:
            if verbose:
                print(f"    {accession}: empty XML")
            continue

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        out_path.write_text(content)
        written += 1
        if verbose:
            print(f"    wrote {out_path.name} ({len(content):,} chars)")

        time.sleep(rate_limit_s)

    return (written, skipped)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SEC Form 4 insider-transaction filings from EDGAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker(s), comma-separated (e.g., TSLA or TSLA,AAPL,MSFT)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Max filings per ticker (default: 20; 0 = all available)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override ML4T_DATA_PATH (default: environment value)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.2,
        help="Seconds between EDGAR requests (default: 0.2s per SEC policy)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    if not tickers:
        print("ERROR: --ticker produced an empty list")
        return 1

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "equities" / "positioning" / "form4"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section(f"SEC FORM 4 DOWNLOAD — {len(tickers)} ticker(s)")
    print()
    print(f"Tickers:  {', '.join(tickers)}")
    print(f"Count:    {args.count} per ticker" if args.count > 0 else "Count:    all available")
    print(f"Output:   {output_dir}")
    print()

    try:
        from edgar import set_identity

        set_identity("ML4T Book stefan@ml4trading.io")
    except ImportError:
        print("ERROR: edgartools not installed. Run: pip install edgartools")
        return 1

    t0 = time.time()
    total_written = 0
    total_skipped = 0

    for i, symbol in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {symbol}")
        written, skipped = download_form4_for_ticker(
            symbol,
            output_dir,
            args.count,
            rate_limit_s=args.rate_limit,
            verbose=args.verbose,
        )
        total_written += written
        total_skipped += skipped
        print(f"  -> {written} new, {skipped} already on disk")

    elapsed = time.time() - t0
    print_section("COMPLETE")
    print(f"New filings:  {total_written}")
    print(f"Already there: {total_skipped}")
    print(f"Output:       {output_dir}")
    print(f"Time:         {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
