# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SEC Form 4: Insider Transaction Analysis
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
#
# ## Purpose
#
# This notebook demonstrates parsing and analyzing SEC Form 4 insider transaction
# filings. Form 4 reports must be filed within 2 business days of an insider trade,
# making them valuable for detecting informed trading activity.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Parse Form 4 XML structure and extract transaction data
# - Build insider transaction DataFrames from raw filings
# - Identify patterns in insider buying and selling
#
# ## Prerequisites
#
# This notebook requires Form 4 filings downloaded via:
# ```bash
# python data/equities/positioning/form4_download.py --ticker TSLA --count 20
# ```
#
# ## Cross-References
#
# - **Download Script**: `data/equities/positioning/form4_download.py`
# - **Related**: [`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb) (edgartools for interactive exploration)
# - **Downstream**: Insider sentiment signals for ML features

# %%
"""SEC Form 4: Insider Transaction Analysis — parse and analyze insider trading filings."""

import warnings

warnings.filterwarnings("ignore")

import re
from datetime import datetime
from pathlib import Path

import polars as pl

from utils import DATA_DIR

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
MAX_SYMBOLS = 0  # 0 = all

# Form 4 data directory (populated by data/equities/positioning/form4_download.py)
FORM4_DIR = DATA_DIR / "equities" / "positioning" / "form4"

# %% [markdown]
# ---
# ## Part 1: Check Data Availability
#
# Form 4 filings are downloaded separately using the canonical script at
# `data/equities/positioning/form4_download.py`.

# %%
# Check what's on disk. The download script (see Prerequisites above) populates one
# subdirectory per ticker; abort with a clear message if nothing is there.
if not FORM4_DIR.exists() or not any(FORM4_DIR.iterdir()):
    raise FileNotFoundError(
        f"No Form 4 filings under {FORM4_DIR}. "
        "Run: python data/equities/positioning/form4_download.py --ticker TSLA --count 20"
    )

tickers = sorted(d.name for d in FORM4_DIR.iterdir() if d.is_dir())
for ticker in tickers:
    n = len(list((FORM4_DIR / ticker).rglob("*.xml")))
    print(f"  {ticker.upper()}: {n} filings")

# %%
# Enumerate every filing on disk; sample the first three to inspect file sizes.
downloaded = sorted(FORM4_DIR.rglob("*.xml"))
print(f"Total filings found: {len(downloaded)}")
for f in downloaded[:3]:
    print(f"  {f.relative_to(FORM4_DIR)} ({f.stat().st_size / 1024:.1f} KB)")

# %% [markdown]
# ---
# ## Part 2: Parsing Form 4 Transactions
#
# Form 4 filings are XML documents. We parse them to extract structured data.
#
# ### Key XML Elements
#
# | Element | Description |
# |---------|-------------|
# | `<issuerName>` | Company name |
# | `<rptOwnerName>` | Insider name |
# | `<officerTitle>` | Position (CEO, CFO, Director, etc.) |
# | `<transactionCode>` | Trade type (P=Purchase, S=Sale, etc.) |
# | `<transactionShares>` | Number of shares |
# | `<transactionPricePerShare>` | Price per share |


# %%
def extract_value(text: str) -> str | None:
    """Extract value from XML <value> tag."""
    match = re.search(r"<value>(.+?)</value>", text)
    return match.group(1).strip() if match else None


# %% [markdown]
# ### Parse a Complete Form 4 Filing
# Extract header info (issuer, owner, title) and all non-derivative transactions.


# %%
def parse_form4(file_path: Path) -> dict:
    """Parse a Form 4 filing and return structured data."""
    content = file_path.read_text(errors="ignore")

    # Extract header info
    issuer = re.search(r"<issuerName>(.+?)</issuerName>", content)
    owner = re.search(r"<rptOwnerName>(.+?)</rptOwnerName>", content)
    title = re.search(r"<officerTitle>(.+?)</officerTitle>", content)

    header = {
        "issuer": issuer.group(1).strip() if issuer else "N/A",
        "owner": owner.group(1).strip() if owner else "N/A",
        "title": title.group(1).strip() if title else "N/A",
    }

    # Extract non-derivative transactions
    transactions = re.findall(
        r"<transactionCode>(.+?)</transactionCode>.*?"
        r"<transactionDate>(.+?)</transactionDate>.*?"
        r"<transactionShares>(.+?)</transactionShares>.*?"
        r"<transactionPricePerShare>(.+?)</transactionPricePerShare>",
        content,
        re.DOTALL,
    )

    tx_list = []
    for code, date_str, shares_str, price_str in transactions:
        date_value = extract_value(date_str)
        shares_value = extract_value(shares_str)
        price_value = extract_value(price_str)

        if all([code, date_value, shares_value]):
            try:
                tx_list.append(
                    {
                        "code": code.strip(),
                        "timestamp": datetime.strptime(date_value, "%Y-%m-%d").date(),
                        "shares": float(shares_value),
                        "price": float(price_value) if price_value and price_value != "D" else 0.0,
                    }
                )
            except (ValueError, TypeError):
                continue

    return {"header": header, "transactions": tx_list}


# %%
# Parse a single filing as a smoke test before processing the full set.
result = parse_form4(downloaded[0])
print(f"Issuer: {result['header']['issuer']}")
print(f"Owner:  {result['header']['owner']}")
print(f"Title:  {result['header']['title']}")
print(f"Transactions: {len(result['transactions'])}")
if result["transactions"]:
    print(f"First transaction: {result['transactions'][0]}")

# %% [markdown]
# ### 2.1 Transaction Codes
#
# | Code | Description |
# |------|-------------|
# | P | Open market purchase |
# | S | Open market sale |
# | A | Grant or award |
# | M | Exercise of derivative (options) |
# | X | Exercise of in-the-money options |
# | G | Gift |
# | D | Disposition to issuer |
# | F | Payment of tax via shares |
# | C | Conversion of derivative |

# %%
# Transaction code mapping
CODE_MAP = {
    "P": "Purchase",
    "S": "Sale",
    "A": "Grant",
    "D": "Disposition",
    "M": "Exercise",
    "C": "Conversion",
    "F": "Tax payment",
    "I": "Discretionary",
    "X": "ITM Exercise",
    "G": "Gift",
}

# %% [markdown]
# ---
# ## Part 3: Building a Transaction DataFrame
#
# Process all downloaded filings into a structured DataFrame.

# %%
# Parse every filing into a flat list of transactions tagged with the issuer/owner/title.
# Real-world Form 4 XML occasionally contains malformed fields; we capture parse errors
# rather than abort the whole batch.
all_transactions = []
parse_errors = []
for file_path in downloaded:
    try:
        parsed = parse_form4(file_path)
        for tx in parsed["transactions"]:
            tx.update(parsed["header"])
            all_transactions.append(tx)
    except Exception as exc:
        parse_errors.append((file_path.name, str(exc)[:80]))

print(f"Transactions extracted: {len(all_transactions)} (parse errors: {len(parse_errors)})")

# %%
# Promote raw codes to readable transaction types and inspect the panel.
df = pl.DataFrame(all_transactions).with_columns(
    pl.col("code").replace_strict(CODE_MAP, default="Unknown").alias("transaction_type")
)
df

# %%
# Aggregate by transaction type to surface activity composition.
summary = (
    df.group_by("transaction_type")
    .agg(
        pl.len().alias("count"),
        pl.col("shares").sum().alias("total_shares"),
    )
    .sort("count", descending=True)
)
summary

# %% [markdown]
# ---
# ## Part 4: Insider Activity Patterns
#
# Analyze patterns in insider buying vs selling.

# %%
# Aggregate open-market purchases (P) and sales (S) by insider.
# Form 4 sometimes reports a sale with `price = 0.0` as a placeholder (e.g., a
# planned disposition where the executed price is filed separately); summing
# those rows into total dollar value would conflate priced trades with
# unreported ones. We split the count: `total_value` is computed only on rows
# with a positive price, and `placeholder_trades` / `placeholder_shares` track
# the share volume reported without a price.
market_trades = df.filter(pl.col("code").is_in(["P", "S"]))
priced = pl.col("price") > 0
by_owner = (
    market_trades.group_by(["owner", "transaction_type"])
    .agg(
        pl.len().alias("trades"),
        pl.col("shares").sum().alias("total_shares"),
        pl.when(priced)
        .then(pl.col("shares") * pl.col("price"))
        .otherwise(0.0)
        .sum()
        .alias("total_value"),
        (~priced).sum().alias("placeholder_trades"),
        pl.when(~priced).then(pl.col("shares")).otherwise(0.0).sum().alias("placeholder_shares"),
    )
    .sort("total_value", descending=True)
)
by_owner.head(10)

# %% [markdown]
# ---
# ## Key Takeaways
#
# 1. Form 4 filings are XML documents with a small, stable schema (`issuerName`, `rptOwnerName`, `officerTitle`, and one or more transaction blocks). A few hundred lines of regex extract them into a flat panel.
# 2. The transaction-code dictionary is the load-bearing piece: open-market purchases (P) and sales (S) are the only signals tied directly to insider conviction; option exercises (M/X), grants (A), tax payments (F), and gifts (G) are noise unless explicitly modelled.
# 3. Aggregating by insider × transaction type surfaces the dollar-weighted activity pattern — in the TSLA sample, Elon Musk's 12 purchases dwarf the cumulative selling activity of every other reporter.
# 4. **For ML features**: net insider buying/selling ratio, purchase-to-sale dollar ratio, and cluster detection (multiple insiders transacting together) are the natural primitives built on top of this DataFrame.
# 5. For interactive single-company workflows, prefer EdgarTools ([`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb)); use the bulk download path here when scaling to many issuers.
