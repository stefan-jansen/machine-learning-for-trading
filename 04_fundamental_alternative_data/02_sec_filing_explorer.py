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
# **Section Reference**: Section 4.1 (The Point-in-Time Pipeline)
#
# ## Purpose
#
# Every US public company files its financial statements, its insider trades and its
# institutional ownership with the Securities and Exchange Commission, and the SEC publishes
# all of it through a free interface called EDGAR. The raw interface returns XML and HTML
# documents. EdgarTools is a Python library that wraps it and hands back objects: a company,
# a list of its filings, an income statement already parsed out of the filing's XBRL tags.
#
# This notebook works through that surface once, so that later notebooks in the chapter can
# reach for the parts of it they need without re-explaining the library. It looks a company up,
# filters its filings, pulls three financial statements, reads a Form 4 insider trade and a
# Form 13F ownership report, and searches the whole filing index by form type.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Look a company up by its ticker or by its Central Index Key, the identifier the SEC assigns.
# - Filter a company's filings down to one form type and read the most recent of them.
# - Pull an income statement, a balance sheet and a cash flow statement out of a 10-K as tables.
# - Read a Form 4 filing, which reports a trade by a company officer or director, and say which
#   transaction it records.
# - Read a Form 13F filing, which reports a large investment manager's quarterly equity
#   holdings, and measure how concentrated the reported portfolio is.
# - Search the whole EDGAR index for filings of one form type over a date range.
#
# ## Cross-References
#
# - **Related**: [`03_sec_form4_insider_transactions`](03_sec_form4_insider_transactions.ipynb) (Form 4 XML parsing)
# - **Downstream**: [`04_sec_xbrl_fundamentals`](04_sec_xbrl_fundamentals.ipynb) (cross-sectional XBRL via API)
#
# ## Prerequisites
#
# The SEC requires every request to identify its sender through a `User-Agent` header holding a
# name and an email address, and it rejects placeholder addresses. Set `EDGAR_IDENTITY` in your
# environment before running this notebook:
#
# ```bash
# export EDGAR_IDENTITY="Jane Doe jane@example.org"
# ```
#
# ## When to use EdgarTools
#
# | Use Case | EdgarTools Fit |
# |----------|----------------|
# | Single company exploration | Excellent |
# | Financial statement extraction | Excellent |
# | Form 4/13F structured data | Excellent |
# | Bulk downloads of thousands of filings | Use the SEC's bulk archives |
# | Cross-sectional fundamentals | Use the XBRL Frames API |
#
# ## Two timestamps, and which one a backtest may use
#
# Every filing carries two dates. The **period end** is the accounting date the statements
# cover; the **filing date** is the day the document reached EDGAR and became public. A 10-K is
# typically filed between one and three months after the period it reports on, so a model that
# keys a fundamental value to its period end is reading a number nobody could have seen at that
# date. Use the filing date as the as-of timestamp for anything that will be backtested. This
# notebook explores rather than builds a pipeline, so it displays both.

# %%
"""EdgarTools: Interactive SEC Filing Analysis - explore SEC filings, financial statements, and insider transactions."""

import os
import warnings
from datetime import date, timedelta

# EdgarTools 5.x emits a DeprecationWarning from three of its own modules while they are
# imported, one per legacy HTML helper it still re-exports. They are the same three on every
# import and they say nothing about the data, so they are filtered by category and by module.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="edgar")

import plotly.graph_objects as go
import polars as pl
from edgar import Company, find, get_filings, set_identity

# Importing utils.style registers and activates the ML4T Plotly template
# (house palette, fonts, gridlines) so figures inherit the book style.
from utils.style import COLORS

# %% [markdown]
# ### The filers this notebook reads, and why each was chosen
#
# Four companies stand in for four kinds of filing. Apple has filed annual reports since the
# 1980s and tags them cleanly, so its 10-K is the reliable case for the XBRL statements in
# Part 3. Tesla's insiders file Form 4s densely enough that the ten most recent span days
# rather than years, which is what makes a fixed window of ten filings reach an actual trade.
# Berkshire Hathaway's 13F is among the most concentrated of the large managers, so its
# holdings show the concentration a chart is meant to expose rather than a flat bar. Microsoft
# is a name to search for.
#
# The three numeric settings that follow are sizes rather than choices about method: how many
# Form 4 filings to scan, how many issuers to draw, and how far back to search the index.
# Replacing any of the tickers changes what the notebook illustrates; a slow filer will need a
# larger Form 4 window before it reaches a purchase or a sale rather than a share grant.

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI.
DEMO_TICKER = "AAPL"  # its 10-K carries the XBRL statements read in Part 3
INSIDER_TICKER = "TSLA"  # a dense Form 4 stream, so a short window still reaches a trade
MANAGER_TICKER = "BRK-A"  # a 13F concentrated enough for the concentration chart to show it
SEARCH_NAME = "Microsoft"  # the name passed to the filer-index search
N_RECENT_FORM4 = 10  # Form 4 filings scanned for a reported sale
N_TOP_HOLDINGS = 10  # issuers drawn on the concentration chart, out of roughly ninety reported
RECENT_FILING_DAYS = 7  # index-search window; EDGAR publishes on business days only

# %%
edgar_identity = os.environ.get("EDGAR_IDENTITY")
if not edgar_identity:
    raise RuntimeError(
        "EDGAR_IDENTITY environment variable is not set. The SEC requires a "
        "real User-Agent (name + email) for every EDGAR request and blocks "
        "placeholder addresses. Set it before running this notebook, e.g. "
        '`export EDGAR_IDENTITY="Jane Doe jane@example.org"`.'
    )
set_identity(edgar_identity)

# %% [markdown]
# ---
# ## Part 1: Finding a company
#
# EDGAR identifies a filer by its Central Index Key, a number the SEC assigns when the company
# first registers. Tickers change and companies are renamed; the CIK does not, which is why
# every mapping in this chapter ends up keyed on it. `Company` accepts either, and accepts the
# CIK with or without the zero padding EDGAR uses in its own URLs.

# %%
company = Company(DEMO_TICKER)

print(f"Name: {company.name}")
print(f"CIK: {company.cik}")
print(f"Tickers: {company.tickers}")
print(f"SIC code: {company.sic}")

# %% [markdown]
# The Standard Industrial Classification code printed above is the SEC's own industry label for
# the filer. It is coarse and it is assigned by the company rather than by a data vendor, but it
# is the only industry field that arrives free with every filing, and it is a reasonable
# fallback where a licensed sector classification is not available.

# %%
tesla_by_cik = Company("1318605")
tesla_padded = Company("0001318605")

print(f"CIK 1318605 resolves to: {tesla_by_cik.name}")
print(f"Zero-padded form resolves to: {tesla_padded.name}")

# %% [markdown]
# Where the CIK is unknown, `find` searches the filer index by name and returns candidates with
# a match score. It is a convenience for exploration rather than a mapping to build a pipeline
# on; the entity-resolution notebook later in this chapter treats name matching as the problem
# it actually is.

# %%
find(SEARCH_NAME)

# %% [markdown]
# ---
# ## Part 2: Retrieving filings
#
# A filer's whole submission history is available as one collection, which is then narrowed by
# form type. The form type is what separates an annual report from a quarterly one, from a
# material-event disclosure, from an insider trade.

# %%
all_filings = company.get_filings()
annual_reports = company.get_filings(form="10-K")
quarterly_reports = company.get_filings(form="10-Q")
current_reports = company.get_filings(form="8-K")

print(f"All filings: {len(all_filings)}")
print(f"10-K, the annual report: {len(annual_reports)}")
print(f"10-Q, the quarterly report: {len(quarterly_reports)}")
print(f"8-K, disclosure of a material event: {len(current_reports)}")

# %% [markdown]
# `latest()` takes the most recent filing in a collection. The `is_xbrl` flag says whether the
# filing carries machine-readable financial tags, which decides whether the statements can be
# read as tables or have to be parsed out of the document text. The SEC has required XBRL of
# large filers since 2009 and of all filers since 2011.

# %%
latest_10k = annual_reports.latest()

print(f"Filing date: {latest_10k.filing_date}")
print(f"Accession number: {latest_10k.accession_no}")
print(f"Carries XBRL tags: {bool(latest_10k.is_xbrl)}")

# %% [markdown]
# A list of form types selects several at once. Forms 3, 4 and 5 are the insider-reporting
# family: a Form 3 is filed when someone becomes an officer, director or large shareholder, a
# Form 4 within two business days of each subsequent trade, and a Form 5 at year end for
# transactions exempt from the Form 4 deadline.

# %%
insider_forms = company.get_filings(form=["3", "4", "5"])
print(f"Insider filings across forms 3, 4 and 5: {len(insider_forms)}")

# %% [markdown]
# ---
# ## Part 3: Financial statements from XBRL
#
# The financial statements inside a 10-K are tagged with XBRL, a standard that attaches a
# machine-readable name to each reported value. EdgarTools reads those tags and assembles the
# three statements, so a line item can be addressed by its label instead of by its position in
# a table. This is the capability that separates it from a library that only downloads
# documents.

# %%
financials = company.get_financials()
income = financials.income_statement()
income

# %% [markdown]
# ### From the rendered statement to a table
#
# The display above is what the library shows in a terminal or a notebook. `to_dataframe`
# returns the same content as a table with one column per reporting period, which is the form
# any downstream calculation needs. A 10-K carries three years of the income statement and two
# of the balance sheet, so the period columns are the fiscal year ends.

# %%
income_df = income.to_dataframe()
period_cols = [c for c in income_df.columns if c[0].isdigit()]

print(f"Rows: {len(income_df)}")
print(f"Reporting periods: {period_cols}")
income_df[["label", *period_cols]].head(15)

# %% [markdown]
# ### The balance sheet and the cash flow statement
#
# Both are reached the same way and return the same shape. Their row counts differ from the
# income statement's because a balance sheet enumerates every asset and liability category the
# filer reports, and because EdgarTools keeps the subtotal rows that the filer presents.

# %%
balance_df = financials.balance_sheet().to_dataframe()
cashflow_df = financials.cashflow_statement().to_dataframe()

print(f"Balance sheet rows: {len(balance_df)}")
print(f"Cash flow statement rows: {len(cashflow_df)}")
balance_df[["label", *[c for c in balance_df.columns if c[0].isdigit()]]].head(10)

# %% [markdown]
# ### Selecting line items by label
#
# Reading a specific line means matching on the label the filer used, and filers do not agree
# on labels: the top line is "Net sales" here and "Revenue", "Total revenues" or "Net revenues"
# elsewhere. That variation is the reason a cross-sectional fundamentals pipeline works from
# the XBRL tag rather than from the label, which is what the Frames API in
# [`04_sec_xbrl_fundamentals`](04_sec_xbrl_fundamentals.ipynb) does.

# %%
TOP_AND_BOTTOM_LINE = r"^(net sales|net revenues?|total (net )?revenues?|revenues?|net income)$"
income_df[income_df["label"].str.strip().str.match(TOP_AND_BOTTOM_LINE, case=False, na=False)][
    ["label", *period_cols]
]

# %% [markdown]
# ---
# ## Part 4: Form 4 insider transactions
#
# A Form 4 records a trade by an officer, a director or a holder of more than ten percent of a
# company's stock, and it is due within two business days of the trade. That deadline is what
# makes the form interesting: it is one of the few disclosures that arrives close enough to the
# event to carry information about it. EdgarTools parses the XML into an object whose
# `common_stock_purchases` and `common_stock_sales` hold the transactions.

# %%
insider_company = Company(INSIDER_TICKER)
form4_filings = insider_company.get_filings(form="4")
print(f"Form 4 filings for {insider_company.name}: {len(form4_filings)}")

# %% [markdown]
# Some Form 4 XML is malformed at the source, and a filing that fails to parse should be
# reported rather than allowed to stop the scan. The summary below records the failure in a
# column instead, so a reader can see how much of the window was readable.

# %%
recent_form4s = form4_filings.head(N_RECENT_FORM4)

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
    except Exception as exc:  # noqa: BLE001 - a source-malformed filing must not stop the scan
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

# %% [markdown]
# One filing in detail shows what a transaction row holds: the security, the trade date, the
# number of shares, the holding that remains afterwards, the price, and the transaction code.
# The scan below takes the first filing in the window that reports a sale.


# %%
def first_filing_with_sales(filings):
    """The first filing in `filings` whose parsed Form 4 reports a common stock sale."""
    for candidate in filings:
        try:
            if len(candidate.obj().common_stock_sales) > 0:
                return candidate
        except Exception:  # noqa: BLE001 - malformed XML is skipped, not raised
            continue
    return None


sale_filing = first_filing_with_sales(recent_form4s)
if sale_filing is None:
    print(f"No sale reported in the {N_RECENT_FORM4} most recent Form 4 filings.")
    sales = None
else:
    form4 = sale_filing.obj()
    print(f"Insider: {form4.insider_name}")
    print(f"Filed: {sale_filing.filing_date}")
    print(f"Issuer: {form4.issuer.name}")
    sales = form4.common_stock_sales
sales

# %% [markdown]
# ### Reading the transaction code
#
# The `Code` column is the SEC's classification of what the trade was, and it is the field that
# decides whether a Form 4 carries a signal at all. An open-market purchase is a decision the
# insider made with their own money; a grant, an option exercise or a share withholding to
# cover tax is compensation machinery running on a schedule set months earlier.
#
# | Code | Description |
# |------|-------------|
# | P | Open market purchase |
# | S | Open market sale |
# | A | Grant or award |
# | M | Exercise of options |
# | G | Gift |
# | D | Disposition to issuer |
# | F | Shares withheld to pay tax |

# %% [markdown]
# ---
# ## Part 5: Form 13F institutional holdings
#
# An investment manager with more than one hundred million dollars in qualifying US equities
# must report its holdings each quarter on Form 13F. Three limits govern how the report can be
# read. It covers long positions only, so a short book is invisible. It reaches only the
# securities the SEC lists as reportable, which is US-listed equities and exchange-traded
# options on them, and not foreign listings, cash or most debt. And it arrives up to forty-five
# days after the quarter end, so the positions it names may already have been closed. Within
# those limits it is the only public record of what large managers hold.
#
# The options are the part that catches a reader out. A reported row carries a `PutCall` field,
# blank for stock and set to `PUT` or `CALL` for an option position, and the value of an option
# row is the value of the underlying it controls rather than a shareholding. Summing the two
# together produces a portfolio that does not exist, so the count below is worth reading before
# anything is computed from the frame.

# %%
manager = Company(MANAGER_TICKER)
thirteenf_filings = manager.get_filings(form="13F-HR")
latest_13f = thirteenf_filings.latest()
holdings = latest_13f.obj()

reported = pl.from_pandas(holdings.holdings).with_columns(
    pl.col("PutCall").cast(pl.Utf8).fill_null("").str.strip_chars().alias("put_call")
)

print(f"13F-HR filings for {manager.name}: {len(thirteenf_filings)}")
print(f"Report period: {holdings.report_period}")
print(f"Reported value: ${holdings.total_value:,.0f}")
print(f"Positions reported: {holdings.total_holdings}")
print(f"Of which option positions: {(reported['put_call'] != '').sum()}")

# %% [markdown]
# ### Portfolio concentration
#
# Expressing each position as a share of the total reported value says how much of the
# portfolio the largest holdings command, which is the first thing worth knowing about any
# manager's book. Option rows are dropped so that the shares below are shares of a stock
# portfolio, and positions are aggregated by issuer so that two share classes of the same
# company count once.

# %%
concentration = (
    reported.filter(pl.col("put_call") == "")
    .group_by("Issuer")
    .agg(pl.col("Value").sum())
    .with_columns((pl.col("Value") / pl.col("Value").sum() * 100).alias("pct_portfolio"))
    .sort("pct_portfolio", descending=True)
    .head(N_TOP_HOLDINGS)
)
concentration

# %%
pct = concentration["pct_portfolio"].to_list()
issuers = [name.title() for name in concentration["Issuer"].to_list()]

# The three largest positions are drawn in amber; the rest give them context in blue.
bar_colors = [COLORS["amber"] if i < 3 else COLORS["blue"] for i in range(len(issuers))]

fig = go.Figure(
    go.Bar(
        x=pct,
        y=issuers,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{p:.1f}%" for p in pct],
        textposition="outside",
        cliponaxis=False,  # keep the outside label on the largest bar from clipping
    )
)
fig.update_layout(
    title=dict(
        text="A few issuers account for most of the reported 13F value"
        f"<br><sup>{manager.name}, as reported for {holdings.report_period}; "
        f"{N_TOP_HOLDINGS} largest issuers</sup>",
    ),
    xaxis_title="Share of reported 13F value (%)",
    xaxis=dict(range=[0, max(pct) * 1.12]),  # headroom for the outside data labels
    yaxis_title="",
    yaxis=dict(autorange="reversed"),  # largest position at the top
    margin=dict(l=160, r=40, t=70, b=55),  # room for long issuer names
    height=430,
)
fig.show()

# %% [markdown]
# ---
# ## Part 6: Searching the whole filing index
#
# The calls so far started from a company. `get_filings` starts from the index instead and
# returns every filer's submissions of a given form, which is how a screen over new disclosures
# begins. Passing a date with a trailing colon reads as "from this date onward".

# %%
recent_10ks = get_filings(form="10-K")
window_start = date.today() - timedelta(days=RECENT_FILING_DAYS)
recent = get_filings(form="10-K", filing_date=f"{window_start}:")

print(f"10-K filings in the current index: {len(recent_10ks)}")
print(f"10-K filings since {window_start}: {len(recent)}")
for filing in recent.head(5):
    print(f"  {filing.filing_date} | {filing.company[:40]}")

# %% [markdown]
# ---
# ## Part 7: The filing document itself
#
# Underneath the parsed objects, a filing is a set of documents, and text is what the NLP
# notebooks later in this book work from. A filing exposes its content four ways:
#
# | Call | Returns |
# |------|---------|
# | `filing.text()` | The filing as plain text |
# | `filing.html()` | The filing as HTML |
# | `filing.open()` | The filing in a browser |
# | `filing.attachments` | Exhibits, XBRL instance and schema files |
#
# A 10-K runs to hundreds of thousands of characters, most of it boilerplate, which is why
# [`14_text_data_extraction`](14_text_data_extraction.ipynb) extracts named sections rather
# than working with the whole document.

# %%
text_content = latest_10k.text()
print(f"Characters in the filing text: {len(text_content):,}")
print(text_content[:600])

# %%
for i, attachment in enumerate(latest_10k.attachments):
    if i >= 10:
        break
    print(f"{i + 1}. {attachment.document}")

# %% [markdown]
# ---
# ## Key Takeaways
#
# 1. EDGAR is free, complete and keyed on the CIK. A pipeline built on tickers will break on
#    the first rename; one built on the CIK will not.
# 2. EdgarTools returns objects rather than documents, which makes single-company work quick:
#    one call reaches a parsed income statement, a Form 4 transaction table, or a 13F holdings
#    frame. Bulk cross-sectional work is a different problem and wants the Frames API or the
#    SEC's bulk archives.
# 3. Statement line items are addressed by the label the filer chose, and filers do not agree
#    on labels. Anything that has to hold across companies keys on the XBRL tag instead.
# 4. A Form 4 is only as informative as its transaction code, because most of the volume in the
#    form is compensation rather than a decision to buy or sell.
# 5. A 13F reports long positions in reportable US securities with a lag of up to forty-five
#    days. Shorts and foreign listings are absent; exchange-traded options are present and carry
#    the value of the underlying they control, so a concentration figure has to separate them
#    from stock. Read one as a description of what was disclosed, not of what the manager held.
# 6. Both dates on a filing are available, and only the filing date is usable as an as-of
#    timestamp for a backtest.
