#!/usr/bin/env python3
"""Download 13F institutional holdings from SEC EDGAR.

Two modes share one CLI and produce the same canonical schema:

  --mode per-cik (default)
      Walk the SEC JSON submissions API for a curated list of large
      institutional investors (Berkshire, Bridgewater, Renaissance, Two
      Sigma, DE Shaw, AQR, Citadel, Millennium, Point72, Tiger Global),
      fetch each 13F-HR filing's XML information table, and assemble a
      multi-quarter holdings panel. Used by Ch22 NB 07 and Ch23 graph /
      RAG notebooks.

  --mode bulk
      Download the SEC's pre-assembled quarterly bulk data set (one 80 MB
      zip with all 13F filings in a 3-month window), parse INFOTABLE +
      COVERPAGE + SUBMISSION, and normalize to the same column schema.
      Used by Ch4 NB 05 to demonstrate the bulk data source at full
      universe scale (~5K filers, ~3M holdings per quarter).

Output layout under `$ML4T_DATA_PATH/equities/positioning/13f/`:

    (per-cik)
      institutional_holdings.parquet    raw holdings: cik, accession_no,
                                        issuer, cusip, value_thousands,
                                        shares, filing_date, company_name
      institution_stock_edges.parquet   institution → stock edge list
      stock_features.parquet            stock-level features
      coownership_matrix.npy            stock × stock similarity
      coownership_stocks.txt            row/col CUSIPs

    (bulk, per quarter)
      bulk/<YYYYQN>/institutional_holdings.parquet   canonical schema,
                                                     ~3M rows
      bulk/<YYYYQN>/bulk_13f.zip                     cached raw zip

Usage:
    # per-cik — default
    python data/equities/positioning/13f_download.py
    python data/equities/positioning/13f_download.py --num-filings 8
    python data/equities/positioning/13f_download.py --max-institutions 3

    # bulk — one or more quarters (filing windows, SEC's own labels)
    python data/equities/positioning/13f_download.py --mode bulk --quarters 2024Q3
    python data/equities/positioning/13f_download.py --mode bulk --quarters 2024Q2,2024Q3

Rate-limited to respect SEC's 10 requests/sec policy.
"""

from __future__ import annotations

import argparse
import calendar
import io
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import requests

from utils.downloading import resolve_data_dir

SEC_HEADERS = {"User-Agent": "ML4T Book stefan@ml4t.io"}
RATE_LIMIT_SECONDS = 0.1
QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")

INSTITUTIONS: list[tuple[str, str]] = [
    ("Berkshire Hathaway", "0001067983"),
    ("Bridgewater Associates", "0001350694"),
    ("Renaissance Technologies", "0001037389"),
    ("Two Sigma Investments", "0001450144"),
    ("DE Shaw", "0001009207"),
    ("AQR Capital", "0001167557"),
    ("Citadel Advisors", "0001423053"),
    ("Millennium Management", "0001273087"),
    ("Point72 Asset Management", "0001603466"),
    ("Tiger Global", "0001167483"),
]


def get_recent_13f_filings(cik: str, num_filings: int) -> list[dict]:
    """Fetch recent 13F-HR filing metadata from SEC EDGAR."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching {cik}: {e}")
        return []

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    for i, form in enumerate(recent.get("form", [])):
        if form == "13F-HR" and len(filings) < num_filings:
            filings.append(
                {
                    "cik": cik,
                    "company_name": data.get("name", "Unknown"),
                    "accession_number": recent["accessionNumber"][i],
                    "filing_date": recent["filingDate"][i],
                }
            )
    return filings


def fetch_13f_xml_root(cik: str, accession: str) -> ET.Element | None:
    """Fetch and parse the XML information table for one 13F filing."""
    acc_clean = accession.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/"
    try:
        idx_resp = requests.get(base_url + "index.json", headers=SEC_HEADERS, timeout=10)
        idx_resp.raise_for_status()
        idx_data = idx_resp.json()
    except Exception:
        return None

    xml_files = [
        item["name"]
        for item in idx_data.get("directory", {}).get("item", [])
        if item["name"].endswith(".xml") and item["name"] != "primary_doc.xml"
    ]
    if not xml_files:
        return None

    try:
        resp = requests.get(base_url + xml_files[0], headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        return ET.fromstring(resp.content)
    except Exception:
        return None


def parse_13f_holdings(cik: str, accession: str) -> list[dict]:
    """Parse holdings rows from a 13F information table XML."""
    root = fetch_13f_xml_root(cik, accession)
    if root is None:
        return []

    ns_map = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}
    holdings = []
    for info_table in root.findall(".//ns:infoTable", ns_map):
        name = info_table.findtext("ns:nameOfIssuer", "", ns_map)
        cusip = info_table.findtext("ns:cusip", "", ns_map)
        value_str = info_table.findtext("ns:value", "0", ns_map)
        shares_elem = info_table.find("ns:shrsOrPrnAmt/ns:sshPrnamt", ns_map)
        shares_str = shares_elem.text if shares_elem is not None else "0"
        holdings.append(
            {
                "cik": cik,
                "accession_no": accession,
                "issuer": name.strip(),
                "cusip": cusip.strip(),
                "value_thousands": int(value_str) if value_str.isdigit() else 0,
                "shares": int(shares_str) if shares_str.isdigit() else 0,
            }
        )
    return holdings


def build_features_and_matrix(
    holdings_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, list[str]]:
    """Build stock features, edge list, and stock co-ownership similarity matrix."""
    latest = holdings_df.sort("filing_date", descending=True).group_by(["cik", "cusip"]).first()

    edge_list = latest.select(
        [
            pl.col("cik").alias("institution_id"),
            pl.col("cusip").alias("stock_id"),
            pl.col("company_name").alias("institution_name"),
            pl.col("issuer").alias("stock_name"),
            pl.col("value_thousands").alias("weight_value"),
            pl.col("shares").alias("weight_shares"),
        ]
    )

    stock_features = (
        latest.group_by("cusip")
        .agg(
            pl.col("issuer").first(),
            pl.col("cik").n_unique().alias("n_inst_holders"),
            pl.col("value_thousands").sum().alias("total_inst_value"),
        )
        .with_columns(
            (pl.col("total_inst_value") / pl.col("n_inst_holders")).alias("avg_inst_value"),
        )
    )

    # Co-ownership similarity matrix
    stocks = sorted(latest["cusip"].unique().to_list())
    institutions = latest["cik"].unique().to_list()
    stock_idx = {s: i for i, s in enumerate(stocks)}
    inst_idx = {c: i for i, c in enumerate(institutions)}
    ownership = np.zeros((len(institutions), len(stocks)), dtype=np.float32)
    for row in latest.iter_rows(named=True):
        ownership[inst_idx[row["cik"]], stock_idx[row["cusip"]]] = row["value_thousands"]
    row_sums = ownership.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ownership_norm = ownership / row_sums
    coown = ownership_norm.T @ ownership_norm
    diag = np.sqrt(np.diag(coown))
    diag[diag == 0] = 1
    similarity = coown / np.outer(diag, diag)

    return stock_features, edge_list, similarity, stocks


# --- Bulk mode (SEC quarterly data sets) ---


def _bulk_zip_url(quarter: str) -> str:
    """Map a filing-window label like '2024Q3' to the SEC bulk zip URL.

    SEC labels 13F data sets by filing-date window, not report quarter:
      Q1 = Mar–May, Q2 = Jun–Aug, Q3 = Sep–Nov, Q4 = Dec (year) – Feb (year+1).
    """
    m = QUARTER_RE.match(quarter)
    if not m:
        raise ValueError(f"Invalid quarter label {quarter!r}; expected format YYYYQN (e.g. 2024Q3)")
    year, q = int(m.group(1)), int(m.group(2))
    if q == 1:
        window = f"01mar{year}-31may{year}"
    elif q == 2:
        window = f"01jun{year}-31aug{year}"
    elif q == 3:
        window = f"01sep{year}-30nov{year}"
    else:  # Q4 straddles the year boundary
        feb_last = 29 if calendar.isleap(year + 1) else 28
        window = f"01dec{year}-{feb_last:02d}feb{year + 1}"
    return f"https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{window}_form13f.zip"


def _download_bulk_zip(quarter: str, target: Path) -> Path:
    """Download the bulk 13F zip for one quarter, skipping if cached."""
    if target.exists():
        print(f"  Using cached zip: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return target
    url = _bulk_zip_url(quarter)
    print(f"  Fetching {url}")
    resp = requests.get(url, headers=SEC_HEADERS, timeout=600)
    resp.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)
    print(f"  Downloaded {len(resp.content) / 1e6:.1f} MB → {target}")
    return target


def _read_bulk_tsv(
    archive: zipfile.ZipFile, name: str, overrides: dict | None = None
) -> pl.DataFrame:
    """Read one TSV inside the bulk zip as a Polars DataFrame."""
    with archive.open(name) as f:
        buf = io.BytesIO(f.read())
    return pl.read_csv(
        buf,
        separator="\t",
        infer_schema_length=10_000,
        schema_overrides=overrides or {},
    )


def _normalize_bulk_to_canonical(
    infotable: pl.DataFrame,
    coverpage: pl.DataFrame,
    submission: pl.DataFrame,
) -> pl.DataFrame:
    """Join the three bulk tables into the canonical per-cik schema.

    Output columns: cik, accession_no, issuer, cusip, value_thousands,
    shares, filing_date, company_name.
    """
    # SEC SUBMISSION.FILING_DATE is "DD-MON-YYYY" uppercase (e.g. "31-OCT-2024").
    # Parse to a Date so downstream filter by start_date/end_date works.
    submission = submission.filter(pl.col("SUBMISSIONTYPE") == "13F-HR").select(
        [
            pl.col("ACCESSION_NUMBER"),
            pl.col("CIK").cast(pl.Utf8).str.zfill(10).alias("cik"),
            pl.col("FILING_DATE").str.to_date(format="%d-%b-%Y", strict=False).alias("filing_date"),
        ]
    )
    coverpage = coverpage.select(
        [
            pl.col("ACCESSION_NUMBER"),
            pl.col("FILINGMANAGER_NAME").alias("company_name"),
        ]
    )
    # INFOTABLE is the big table — keep only what the canonical schema needs.
    holdings = infotable.select(
        [
            pl.col("ACCESSION_NUMBER").alias("accession_no"),
            pl.col("NAMEOFISSUER").alias("issuer"),
            pl.col("CUSIP").alias("cusip"),
            pl.col("VALUE").cast(pl.Int64).alias("value_thousands"),
            pl.col("SSHPRNAMT").cast(pl.Int64).alias("shares"),
            pl.col("ACCESSION_NUMBER"),
        ]
    )

    # Inner-joins drop any holdings whose submission type isn't 13F-HR.
    return (
        holdings.join(submission, on="ACCESSION_NUMBER", how="inner")
        .join(coverpage, on="ACCESSION_NUMBER", how="inner")
        .select(
            [
                "cik",
                "accession_no",
                "issuer",
                "cusip",
                "value_thousands",
                "shares",
                "filing_date",
                "company_name",
            ]
        )
    )


def _run_bulk(quarters: list[str], bulk_root: Path) -> int:
    """Download + normalize one or more quarterly bulk sets."""
    for quarter in quarters:
        q_dir = bulk_root / quarter
        zip_path = q_dir / "bulk_13f.zip"
        out_path = q_dir / "institutional_holdings.parquet"

        print(f"\n{quarter}:")
        _download_bulk_zip(quarter, zip_path)

        with zipfile.ZipFile(zip_path) as archive:
            members = set(archive.namelist())
            required = {"INFOTABLE.tsv", "COVERPAGE.tsv", "SUBMISSION.tsv"}
            missing = required - members
            if missing:
                print(f"  ERROR: zip missing required tables: {sorted(missing)}")
                return 1

            infotable = _read_bulk_tsv(
                archive,
                "INFOTABLE.tsv",
                overrides={"OTHERMANAGER": pl.Utf8, "FIGI": pl.Utf8},
            )
            coverpage = _read_bulk_tsv(archive, "COVERPAGE.tsv")
            submission = _read_bulk_tsv(archive, "SUBMISSION.tsv")

        print(
            f"  Parsed {len(infotable):,} holdings  "
            f"{len(coverpage):,} coverpages  "
            f"{len(submission):,} submissions"
        )

        canonical = _normalize_bulk_to_canonical(infotable, coverpage, submission)
        canonical.write_parquet(out_path)

        n_cik = canonical["cik"].n_unique()
        n_issuers = canonical["issuer"].n_unique()
        total_value = canonical["value_thousands"].sum() / 1e12
        print(
            f"  Wrote {out_path.name}  "
            f"({len(canonical):,} rows, {n_cik:,} managers, "
            f"{n_issuers:,} unique issuers, ${total_value:.1f}T total)"
        )
    return 0


# --- Per-CIK mode (curated institutions via JSON submissions API) ---


def _run_per_cik(
    output_dir: Path,
    num_filings: int,
    max_institutions: int,
) -> int:
    institutions = INSTITUTIONS[:max_institutions] if max_institutions else INSTITUTIONS

    print(f"Downloading 13F data to: {output_dir}")
    print(f"Institutions: {len(institutions)}  Filings each: {num_filings}")

    all_filings: list[dict] = []
    for name, cik in institutions:
        filings = get_recent_13f_filings(cik, num_filings)
        all_filings.extend(filings)
        print(f"  {name}: {len(filings)} filings")
        time.sleep(RATE_LIMIT_SECONDS)

    if not all_filings:
        print("No filings retrieved.")
        return 1
    filings_df = pl.DataFrame(all_filings)

    all_holdings: list[dict] = []
    for row in filings_df.iter_rows(named=True):
        holdings = parse_13f_holdings(row["cik"], row["accession_number"])
        for h in holdings:
            h["filing_date"] = row["filing_date"]
            h["company_name"] = row["company_name"]
        all_holdings.extend(holdings)
        print(
            f"  {row['company_name'][:32]:<32} {row['filing_date']}  {len(holdings):>5} positions"
        )
        time.sleep(RATE_LIMIT_SECONDS)

    if not all_holdings:
        print("No holdings parsed.")
        return 1

    holdings_df = pl.DataFrame(all_holdings)
    stock_features, edge_list, coown_matrix, stocks = build_features_and_matrix(holdings_df)

    holdings_path = output_dir / "institutional_holdings.parquet"
    edges_path = output_dir / "institution_stock_edges.parquet"
    features_path = output_dir / "stock_features.parquet"
    matrix_path = output_dir / "coownership_matrix.npy"
    stocks_path = output_dir / "coownership_stocks.txt"

    holdings_df.write_parquet(holdings_path)
    edge_list.write_parquet(edges_path)
    stock_features.write_parquet(features_path)
    np.save(matrix_path, coown_matrix)
    stocks_path.write_text("\n".join(stocks))

    print("")
    print(f"Wrote {holdings_path.name}   ({len(holdings_df):,} rows)")
    print(f"Wrote {edges_path.name}      ({len(edge_list):,} rows)")
    print(f"Wrote {features_path.name}   ({len(stock_features):,} stocks)")
    print(f"Wrote {matrix_path.name}     ({coown_matrix.shape})")
    print(f"Wrote {stocks_path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SEC 13F institutional holdings")
    parser.add_argument(
        "--mode",
        choices=["per-cik", "bulk"],
        default="per-cik",
        help="per-cik: curated institutions via SEC JSON API (default). "
        "bulk: SEC quarterly bulk data sets (~80 MB zip per quarter).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override output root (default: $ML4T_DATA_PATH)",
    )
    # Per-CIK args
    parser.add_argument(
        "--num-filings",
        type=int,
        default=4,
        help="[per-cik] Number of recent 13F-HR filings per institution (default 4)",
    )
    parser.add_argument(
        "--max-institutions",
        type=int,
        default=0,
        help="[per-cik] Limit to first N institutions (0 = all)",
    )
    # Bulk args
    parser.add_argument(
        "--quarters",
        type=str,
        default="",
        help="[bulk] Comma-separated filing-window labels (e.g. '2024Q2,2024Q3'). "
        "SEC labels by filing date: Q1=Mar-May, Q2=Jun-Aug, Q3=Sep-Nov, Q4=Dec-Feb.",
    )
    args = parser.parse_args()

    data_path = resolve_data_dir(args.data_path)
    root = data_path / "equities" / "positioning" / "13f"
    root.mkdir(parents=True, exist_ok=True)

    if args.mode == "bulk":
        if not args.quarters:
            parser.error("--mode bulk requires --quarters (e.g. --quarters 2024Q3)")
        quarters = [q.strip() for q in args.quarters.split(",") if q.strip()]
        # Validate early so a bad label doesn't surface only after a long download.
        for q in quarters:
            _bulk_zip_url(q)
        return _run_bulk(quarters, root / "bulk")

    return _run_per_cik(root, args.num_filings, args.max_institutions)


if __name__ == "__main__":
    raise SystemExit(main())
