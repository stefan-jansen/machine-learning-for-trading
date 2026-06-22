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
# # EdgarTools: Interactive SEC Filing Analysis
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: Section 4.2 (Entity Resolution and Mapping)
#
# ## Purpose
#
# This notebook demonstrates EdgarTools, a high-level Python library for interactive
# SEC EDGAR analysis. EdgarTools excels at company exploration, financial statement
# extraction, and working with structured filing data like Form 4 and 13F.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Look up companies by ticker, CIK, or search
# - Retrieve and filter SEC filings
# - Extract XBRL-parsed financial statements (income, balance sheet, cash flow)
# - Analyze Form 4 insider transactions
# - Examine 13F institutional holdings
#
# ## Cross-References
#
# - **Related**: [`03_sec_form4_insider_transactions`](03_sec_form4_insider_transactions.ipynb) (Form 4 XML parsing)
# - **Downstream**: [`04_sec_xbrl_fundamentals`](04_sec_xbrl_fundamentals.ipynb) (cross-sectional XBRL via API)
#
# ## When to Use EdgarTools
#
# | Use Case | EdgarTools Fit |
# |----------|----------------|
# | Single company exploration | Excellent |
# | Financial statement extraction | Excellent |
# | Form 4/13F structured data | Excellent |
# | Bulk downloads (1000s of filings) | Use sec-edgar instead |
# | Cross-sectional fundamentals | Use XBRL Frames API |
#
# ## WARNING: Point-in-Time (PIT) Note
#
# This notebook demonstrates **data exploration**, not PIT-correct data pipelines.
# For backtesting or ML training, always use the **filing_date** (when data became
# public), not the **period_end** (accounting date). Filing dates are typically
# 30-90 days after period end.

# %%
"""EdgarTools: Interactive SEC Filing Analysis — explore SEC filings, financial statements, and insider transactions."""

import os
import warnings

warnings.filterwarnings("ignore")


import polars as pl

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
# %%
from edgar import Company, find, get_filings, set_identity

# SEC requires a real User-Agent (name + email) for every request and blocks
# placeholder addresses. Set `EDGAR_IDENTITY` in your environment, e.g.
# `export EDGAR_IDENTITY="Jane Doe jane@example.org"`.
edgar_identity = os.environ.get("EDGAR_IDENTITY")
if not edgar_identity:
    raise RuntimeError(
        "EDGAR_IDENTITY environment variable is not set. The SEC requires a "
        "real User-Agent (name + email) for every EDGAR request and blocks "
        "placeholder addresses. Set it before running this notebook, e.g. "
        '`export EDGAR_IDENTITY="Jane Doe jane@example.org"`.'
    )
set_identity(edgar_identity)

print("EdgarTools loaded.")

# %% [markdown]
# ---
# ## Part 1: Company Lookup
#
# EdgarTools provides multiple ways to find companies:
# - By ticker symbol (most common)
# - By CIK (SEC's Central Index Key)
# - By search query

# %%
# Look up by ticker
apple = Company("AAPL")

print(f"=== {apple.name} ===")
print(f"CIK: {apple.cik}")
print(f"Tickers: {apple.tickers}")
print(f"SIC Code: {apple.sic}")

# %%
# Look up by CIK (with or without zero-padding)
tesla_by_cik = Company("1318605")
print(f"Tesla by CIK: {tesla_by_cik.name}")

# Zero-padded also works
tesla_padded = Company("0001318605")
print(f"Tesla padded: {tesla_padded.name}")

# %%
# Search for companies by name
results = find("Microsoft")
results

# %% [markdown]
# ---
# ## Part 2: Retrieving Filings
#
# The `get_filings()` method returns a filterable collection of filings.
# Filter by form type, date range, or both.

# %%
# Get all filings for Apple
all_filings = apple.get_filings()
print(f"Total Apple filings: {len(all_filings)}")

# %%
# Filter by form type
annual_reports = apple.get_filings(form="10-K")
quarterly_reports = apple.get_filings(form="10-Q")
eightks = apple.get_filings(form="8-K")

print(f"10-K filings: {len(annual_reports)}")
print(f"10-Q filings: {len(quarterly_reports)}")
print(f"8-K filings: {len(eightks)}")

# %%
# Get the latest filing
latest_10k = annual_reports.latest()

print("=== Latest 10-K ===")
print(f"Filing Date: {latest_10k.filing_date}")
print(f"Accession Number: {latest_10k.accession_no}")
print(f"Is XBRL: {latest_10k.is_xbrl}")

# %%
# Multiple form types at once
insider_forms = apple.get_filings(form=["3", "4", "5"])
print(f"Insider filings (Forms 3, 4, 5): {len(insider_forms)}")

# %% [markdown]
# ---
# ## Part 3: Financial Statements from XBRL
#
# EdgarTools parses XBRL data to provide direct access to financial statements.
# This is the key differentiator from other SEC libraries.

# %%
# Get financials from the company (uses latest 10-K)
financials = apple.get_financials()
financials

# %% [markdown]
# ### 3.1 Income Statement

# %%
# Get income statement
income = financials.income_statement()
income

# %%
# Convert to DataFrame for analysis (wide format: one column per period)
income_df = income.to_dataframe()
date_cols = [c for c in income_df.columns if c[0].isdigit()]

print(f"Shape: {income_df.shape}")
print(f"Period columns: {date_cols}")
income_df[["label", *date_cols]].head(15)

# %% [markdown]
# ### 3.2 Balance Sheet

# %%
# Get balance sheet
balance = financials.balance_sheet()
balance

# %%
balance_df = balance.to_dataframe()
print(f"Balance sheet rows: {len(balance_df)}")

# %% [markdown]
# ### 3.3 Cash Flow Statement

# %%
# Get cash flow statement
cashflow = financials.cashflow_statement()
cashflow

# %% [markdown]
# ### 3.4 Working with Statement Data
#
# The DataFrame format allows you to compute ratios and perform analysis.

# %%
# Extract specific line items from income statement
income_df = income.to_dataframe()
date_cols = [c for c in income_df.columns if c[0].isdigit()]

key_items = income_df[income_df["label"].str.contains("Revenue|Net income", case=False, na=False)]
key_items[["label", *date_cols]]

# %% [markdown]
# ---
# ## Part 4: Form 4 Insider Transactions
#
# Form 4 reports insider trading activity. EdgarTools parses the XML
# into structured data.

# %%
# Get Tesla's Form 4 filings
tesla = Company("TSLA")
form4_filings = tesla.get_filings(form="4")
print(f"Tesla Form 4 filings: {len(form4_filings)}")

# %%
# Parse the 10 most recent Form 4 filings into a tidy DataFrame.
# Some Form 4 XML is malformed at the source; capture parse failures rather than abort.
recent_form4s = form4_filings.head(10)

records = []
for filing in recent_form4s:
    try:
        form4 = filing.obj()
        records.append(
            {
                "filing_date": str(filing.filing_date),
                "insider": form4.insider_name or "Unknown",
                "n_purchases": len(form4.common_stock_purchases),
                "n_sales": len(form4.common_stock_sales),
                "parse_error": None,
            }
        )
    except Exception as exc:
        records.append(
            {
                "filing_date": str(filing.filing_date),
                "insider": None,
                "n_purchases": None,
                "n_sales": None,
                "parse_error": str(exc)[:80],
            }
        )

insider_summary = pl.DataFrame(records)
insider_summary


# %%
# Detailed look at the first Form 4 in the window that contains stock sales.
# Wrap parsing in a guard so malformed XML doesn't abort the cell, and handle
# the case where no Form 4 in the recent window contains sales.
def _has_sales(f):
    try:
        return len(f.obj().common_stock_sales) > 0
    except Exception:
        return False


sale_filing = next((f for f in recent_form4s if _has_sales(f)), None)
if sale_filing is None:
    print("No Form 4 with stock sales found in the recent window.")
    form4 = None
else:
    form4 = sale_filing.obj()
    print(f"{form4.insider_name} — {sale_filing.filing_date}")
    print(f"Issuer: {form4.issuer}")
form4.common_stock_sales if form4 is not None else None

# %% [markdown]
# ### 4.1 Form 4 Transaction Codes
#
# | Code | Description |
# |------|-------------|
# | P | Open market purchase |
# | S | Open market sale |
# | A | Grant or award |
# | M | Exercise of options |
# | G | Gift |
# | D | Disposition to issuer |
# | F | Tax payment via shares |

# %% [markdown]
# ---
# ## Part 5: 13F Institutional Holdings
#
# Form 13F-HR reports quarterly holdings for institutional investment managers
# with $100M+ in qualifying securities.

# %%
# Get Berkshire Hathaway's 13F filings
berkshire = Company("BRK-A")
thirteenf_filings = berkshire.get_filings(form="13F-HR")
print(f"Berkshire 13F-HR filings: {len(thirteenf_filings)}")

# %%
# Get latest 13F
latest_13f = thirteenf_filings.latest()
holdings = latest_13f.obj()

print("=== Berkshire Hathaway Holdings ===")
print(f"Report Period: {holdings.report_period}")
print(f"Total Value: ${holdings.total_value:,.0f}")
print(f"Number of Holdings: {holdings.total_holdings}")

# %%
# View holdings DataFrame — top 20 by reported market value.
holdings_df = holdings.holdings
top_holdings = holdings_df.nlargest(20, "Value")[["Issuer", "Class", "Value", "SharesPrnAmount"]]
top_holdings

# %%
# Express each position as a share of total portfolio value.
holdings_pl = pl.from_pandas(holdings_df).with_columns(
    (pl.col("Value") / pl.col("Value").sum() * 100).alias("pct_portfolio")
)
holdings_pl.select(["Issuer", "Value", "pct_portfolio"]).head(10)

# %% [markdown]
# ---
# ## Part 6: Global Filing Search
#
# Search across all SEC filers for specific form types.

# %%
# Get recent 10-K filings across all companies
recent_10ks = get_filings(form="10-K")
print(f"Recent 10-K filings available: {len(recent_10ks)}")

# %%
# Filter by date
from datetime import date, timedelta

one_week_ago = date.today() - timedelta(days=7)
recent = get_filings(form="10-K", filing_date=f"{one_week_ago}:")
print(f"10-K filings in last week: {len(recent)}")

# Show a few
for filing in recent.head(5):
    print(f"{filing.filing_date} | {filing.company[:40]}")

# %% [markdown]
# ---
# ## Part 7: Accessing Filing Content
#
# Beyond structured data, you can access the raw filing content.

# %%
# Get filing content
filing = apple.get_filings(form="10-K").latest()

# Available content methods
print("=== Filing Content Methods ===")
print("filing.text()     - Plain text content")
print("filing.html()     - HTML content")
print("filing.open()     - Open in browser")
print("filing.attachments - List of attachments")

# %%
# Get text content (excerpt)
text_content = filing.text()
print(f"\n=== 10-K Text Excerpt ({len(text_content):,} chars total) ===")
print(text_content[:1000])

# %%
# List attachments
print("\n=== Filing Attachments ===")
for i, attachment in enumerate(filing.attachments):
    if i >= 10:
        break
    print(f"{i + 1}. {attachment.document}")

# %% [markdown]
# ---
# ## Key Takeaways
#
# 1. EdgarTools wraps the SEC EDGAR REST API with a Pythonic surface: companies, filings, parsed XBRL statements, and structured Form 4 / 13F objects.
# 2. The library is well suited to interactive single-company workflows — a 10-K returns three years of income statement, balance sheet, and cash flow as ready-to-analyse DataFrames.
# 3. Form 4 and 13F filings expose `common_stock_purchases`, `common_stock_sales`, and a `holdings` DataFrame that fits naturally into a portfolio-concentration analysis (Berkshire's top three positions — Apple, American Express, Coca-Cola — make up about 51% of reported value in this snapshot, and the top four exceed 60%).
# 4. For Form 4 insider-transaction XML parsing, see [`03_sec_form4_insider_transactions`](03_sec_form4_insider_transactions.ipynb); for cross-sectional XBRL fundamentals, use the Frames API ([`04_sec_xbrl_fundamentals`](04_sec_xbrl_fundamentals.ipynb)).
# 5. **PIT reminder**: anything you build for backtesting must key off `filing_date`, not `period_end`. EdgarTools surfaces both — use the former for the as-of timestamp.
