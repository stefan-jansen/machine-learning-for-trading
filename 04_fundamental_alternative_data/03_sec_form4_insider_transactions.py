# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
# **Section Reference**: Section 4.1 (The Point-in-Time Pipeline)
#
# ## Purpose
#
# A Form 4 is filed when a company's officer, director or ten-percent shareholder trades its
# stock, and it is due within two business days of the trade. That deadline is short enough
# that the filing arrives while the information is still current, which is what makes the form
# worth parsing at all. The previous notebook read Form 4s through a library that had already
# parsed them. This one works from the XML the SEC publishes, because a feature built on
# insider activity is only as trustworthy as the reader that produced it, and the failure modes
# live in the parsing.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Read a Form 4 XML document and name the elements that carry the issuer, the reporting
#   insider and each individual trade.
# - Extract one row per trade with its transaction code, date, share count, price and
#   direction, keeping every field attached to the trade it came from.
# - Reconcile the rows you extracted against the trades present in the raw files, and account
#   for every one you did not keep.
# - Separate trades that reflect a decision to buy or sell from the compensation events that
#   make up most of the volume, using the transaction code.
# - Aggregate priced trades into a per-insider dollar total without letting an unpriced row
#   silently enter it.
#
# ## Prerequisites
#
# This notebook reads Form 4 filings downloaded by:
# ```bash
# uv run python data/equities/positioning/form4_download.py --ticker TSLA --count 20
# ```
#
# ## Cross-References
#
# - **Download Script**: `data/equities/positioning/form4_download.py`
# - **Related**: [`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb) (edgartools for interactive exploration)

# %%
"""SEC Form 4: Insider Transaction Analysis - parse and analyze insider trading filings."""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from utils import DATA_DIR
from utils.style import COLORS  # importing utils.style activates the ml4t Plotly template

# %% tags=["parameters"]
# One subdirectory per ticker, written by data/equities/positioning/form4_download.py.
FORM4_DIR = DATA_DIR / "equities" / "positioning" / "form4"

# %% [markdown]
# ---
# ## Part 1: What is on disk
#
# The download script writes one XML file per filing, under a directory named for the issuer's
# ticker. Reading the file count and the file sizes first is worth the two lines: a Form 4 is a
# small document, and a file of a few hundred bytes is an error page rather than a filing.

# %%
if not FORM4_DIR.exists() or not any(FORM4_DIR.iterdir()):
    raise FileNotFoundError(
        f"No Form 4 filings under {FORM4_DIR}. "
        "Run: uv run python data/equities/positioning/form4_download.py --ticker TSLA --count 20"
    )

tickers = sorted(d.name for d in FORM4_DIR.iterdir() if d.is_dir())
downloaded = sorted(f for t in tickers for f in (FORM4_DIR / t).rglob("*.xml"))

for ticker in tickers:
    files = list((FORM4_DIR / ticker).rglob("*.xml"))
    sizes = sorted(f.stat().st_size / 1024 for f in files)
    print(f"{ticker.upper()}: {len(files)} filings, {sizes[0]:.1f}-{sizes[-1]:.1f} KB each")

# %% [markdown]
# ---
# ## Part 2: Reading the XML
#
# Each filing holds a `<nonDerivativeTable>` for common stock and a `<derivativeTable>` for
# options and other derivatives, and each table holds one transaction block per trade. Those
# blocks are the reason to use an XML parser rather than a regular expression over the document
# text: a regex has no notion of where one block ends, so it will pair the transaction code
# from one trade with the price from the next, and read straight across the boundary from
# common stock into options. Walking the tree keeps every field anchored to its own trade.
#
# ### The elements this notebook reads
#
# | Element | Description |
# |---------|-------------|
# | `<issuerName>` | The company whose stock was traded |
# | `<reportingOwner>` | One block per insider filing the report |
# | `<rptOwnerName>` | The insider's name |
# | `<officerTitle>` | The insider's position, where they hold one |
# | `<nonDerivativeTransaction>` | One common-stock trade |
# | `<transactionCode>` | What kind of trade it was |
# | `<transactionShares>` | Number of shares |
# | `<transactionPricePerShare>` | Price per share, absent on some transaction types |
# | `<transactionAcquiredDisposedCode>` | A for acquired, D for disposed |
#
# Derivative transactions are read separately below only to be counted, so that the rows this
# notebook keeps can be reconciled against everything the files contain.


# %%
def find_text(element: ET.Element, path: str) -> str | None:
    """Return the stripped text at `path` below `element`, or None if absent or empty."""
    found = element.find(path)
    if found is None or found.text is None:
        return None
    return found.text.strip() or None


# %% [markdown]
# ### One filing, one function
#
# A Form 4 may be filed jointly by more than one insider, in which case the document carries
# several `<reportingOwner>` blocks and reports the trades on behalf of all of them. Reading the first
# name found anywhere in the document would attach a joint filing's trades to whichever owner
# happened to be listed first, so the owners are collected as a list and their count is carried
# on every row.
#
# Two categories of trade are deliberately not returned: derivative transactions, which are a
# different instrument, and any block missing the code, the date or the share count, without
# which the row cannot be interpreted. Both are counted rather than dropped in silence, so that
# the reconciliation in Part 3 can account for them.


# %%
def parse_form4(file_path: Path) -> dict:
    """Return the header, the common-stock trades, and what was left out, for one filing."""
    root = ET.parse(file_path).getroot()

    owners = [find_text(block, ".//rptOwnerName") for block in root.findall(".//reportingOwner")]
    owners = [name for name in owners if name]
    header = {
        "issuer": find_text(root, ".//issuerName"),
        "owner": "; ".join(owners) if owners else None,
        "n_owners": len(owners),
        "title": find_text(root, ".//officerTitle"),
    }

    trades, incomplete = [], 0
    for tx in root.findall(".//nonDerivativeTransaction"):
        code = find_text(tx, "transactionCoding/transactionCode")
        traded_on = find_text(tx, "transactionDate/value")
        shares = find_text(tx, "transactionAmounts/transactionShares/value")
        price = find_text(tx, "transactionAmounts/transactionPricePerShare/value")
        direction = find_text(tx, "transactionAmounts/transactionAcquiredDisposedCode/value")

        if not (code and traded_on and shares):
            incomplete += 1
            continue
        try:
            trades.append(
                {
                    "code": code,
                    "timestamp": datetime.strptime(traded_on, "%Y-%m-%d").date(),
                    "shares": float(shares),
                    # A grant, a gift and a disposition to the issuer are all filed without a
                    # price. None keeps them out of every dollar total by construction; a
                    # zero would multiply into one silently.
                    "price": float(price) if price else None,
                    "direction": direction,
                }
            )
        except ValueError:
            incomplete += 1

    return {
        "header": header,
        "trades": trades,
        "n_derivative": len(root.findall(".//derivativeTransaction")),
        "n_incomplete": incomplete,
    }


# %% [markdown]
# Running it on one filing before the whole set is the cheap way to see whether the element
# paths are right: a wrong path returns `None` rather than raising, so a parser that reads
# nothing looks exactly like a parser that read an empty filing.

# %%
sample = parse_form4(downloaded[0])
print(f"Issuer: {sample['header']['issuer']}")
print(f"Reporting owners: {sample['header']['owner']} ({sample['header']['n_owners']})")
print(f"Title: {sample['header']['title']}")
print(f"Common-stock trades: {len(sample['trades'])}")
print(f"Derivative transactions in the same filing: {sample['n_derivative']}")
if sample["trades"]:
    print(f"First trade: {sample['trades'][0]}")

# %% [markdown]
# ### What the transaction codes mean
#
# The code is the field that decides whether a row carries information. An open-market purchase
# or sale is a decision the insider made with their own money. A grant, an option exercise, a
# share withholding to cover tax, or a gift is compensation machinery running on a schedule
# fixed months earlier, and it says nothing about what the insider thinks the stock is worth.
#
# | Code | Description |
# |------|-------------|
# | P | Open market purchase |
# | S | Open market sale |
# | A | Grant or award |
# | M | Exercise of a derivative security |
# | X | Exercise of an in-the-money option |
# | G | Gift |
# | D | Disposition to the issuer |
# | F | Shares withheld to pay tax |
# | C | Conversion of a derivative security |
# | I | Discretionary transaction |

# %%
CODE_MAP = {
    "P": "Purchase",
    "S": "Sale",
    "A": "Grant",
    "D": "Disposition to issuer",
    "M": "Derivative exercise",
    "C": "Conversion",
    "F": "Tax withholding",
    "I": "Discretionary",
    "X": "Option exercise",
    "G": "Gift",
}
# Only these two are decisions to buy or sell at a market price.
DECISION_CODES = ["P", "S"]

# %% [markdown]
# ---
# ## Part 3: Every filing, and an account of every trade
#
# Parsing the whole set produces one row per common-stock trade. What matters more than the row
# count is that the row count can be explained: a parser that quietly drops a tenth of its
# input produces a panel that looks complete and is not.

# %%
rows, per_file = [], []
for file_path in downloaded:
    parsed = parse_form4(file_path)
    for trade in parsed["trades"]:
        rows.append({**trade, **parsed["header"]})
    per_file.append(
        {
            "file": file_path.name,
            "n_owners": parsed["header"]["n_owners"],
            "n_trades": len(parsed["trades"]),
            "n_derivative": parsed["n_derivative"],
            "n_incomplete": parsed["n_incomplete"],
        }
    )

df = pl.DataFrame(rows).with_columns(
    pl.col("code").replace_strict(CODE_MAP, default="Unknown").alias("transaction_type")
)
accounting = pl.DataFrame(per_file)
print(f"Filings parsed: {len(per_file)}")
print(f"Common-stock trades extracted: {len(df)}")

# %% [markdown]
# ### Reconciling against the raw files
#
# The check that a parser is complete cannot come from the parser. Counting the opening tags in
# the file text uses no part of the code above: it does not know the element paths, the schema
# or which blocks were skipped, so agreement between the two counts is evidence rather than
# self-confirmation. Every trade in the files then falls into exactly one of three buckets:
# extracted, excluded as a derivative, or excluded as incomplete.

# %%
raw_text = [f.read_text(errors="replace") for f in downloaded]
raw_common = sum(text.count("<nonDerivativeTransaction>") for text in raw_text)
raw_derivative = sum(text.count("<derivativeTransaction>") for text in raw_text)

extracted = len(df)
excluded_derivative = int(accounting["n_derivative"].sum())
excluded_incomplete = int(accounting["n_incomplete"].sum())

print(f"Transaction blocks in the raw XML: {raw_common + raw_derivative}")
print(f"  common stock: {raw_common}")
print(f"  derivative:   {raw_derivative}")
print(f"Rows extracted: {extracted}")
print(f"Excluded as derivative: {excluded_derivative}")
print(f"Excluded as incomplete: {excluded_incomplete}")
assert extracted + excluded_incomplete == raw_common, "common-stock blocks are unaccounted for"
assert excluded_derivative == raw_derivative, "derivative blocks are unaccounted for"

# %% [markdown]
# ### The panel
#
# Before anything is computed from it, the panel is worth looking at directly: which issuer it
# covers, which insiders appear, over what dates, and whether any filing reported more than one
# owner.

# %%
print(f"Issuers: {df['issuer'].unique().to_list()}")
print(f"Insiders: {df['owner'].n_unique()}")
print(f"Trade dates: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Filings with more than one reporting owner: {int((accounting['n_owners'] > 1).sum())}")
df.head(10)

# %% [markdown]
# ### Frequency against volume
#
# Counting trades and summing shares answer different questions, and the two panels below
# disagree because most insider volume is not a trading decision. Sorting both by share volume
# and reading across is the point: the code that moves the most stock is not the code that
# appears most often, and a feature built on either count alone inherits whichever mistake it
# made.

# %%
summary = (
    df.group_by("transaction_type")
    .agg(pl.len().alias("n_trades"), pl.col("shares").sum().alias("total_shares"))
    .sort("total_shares", descending=True)
)
summary

# %%
types = summary["transaction_type"].to_list()
# The largest-volume category is drawn in amber, the rest in slate, so the two panels can be
# read against each other by position and by colour.
bar_colors = [COLORS["amber"] if i == 0 else COLORS["slate"] for i in range(len(types))]

fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("Trades reported", "Shares traded"),
    horizontal_spacing=0.08,
)
fig.add_trace(
    go.Bar(y=types, x=summary["n_trades"], orientation="h", marker_color=bar_colors),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(y=types, x=summary["total_shares"], orientation="h", marker_color=bar_colors),
    row=1,
    col=2,
)
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title_text="Trades", row=1, col=1)
fig.update_xaxes(title_text="Shares", row=1, col=2)
fig.update_layout(
    title="The most frequent transaction code is not the one that moves the most stock",
    showlegend=False,
    height=340,
)
fig.show()

# %% [markdown]
# ---
# ## Part 4: Who traded, and for how much
#
# Restricting to the two decision codes and aggregating by insider gives the dollar-weighted
# picture. A dollar total needs a price, and Form 4 does not always file one: a gift carries
# none, and a disposition to the issuer may have its price filed separately. Multiplying shares
# by a missing price is the mistake this cell exists to avoid, so unpriced rows are counted in
# their own columns instead of entering the total.

# %%
priced = pl.col("price").is_not_null()
by_insider = (
    df.filter(pl.col("code").is_in(DECISION_CODES))
    .group_by(["owner", "transaction_type"])
    .agg(
        pl.len().alias("trades"),
        pl.col("shares").sum().alias("total_shares"),
        pl.when(priced)
        .then(pl.col("shares") * pl.col("price"))
        .otherwise(0.0)
        .sum()
        .alias("value"),
        (~priced).sum().alias("unpriced_trades"),
        pl.when(~priced).then(pl.col("shares")).otherwise(0.0).sum().alias("unpriced_shares"),
    )
    .sort("value", descending=True)
)
by_insider

# %% [markdown]
# Drawn as one bar per insider and direction, the distribution shows what a per-insider
# aggregate is for. Insider selling is spread across many people and many small trades, because
# it is how equity compensation is turned into cash; buying, when it happens at all, tends to
# be concentrated in one person and one decision. A feature that nets the two without weighting
# by value will read this as balanced.

# %%
labels = [f"{o} ({t})" for o, t in zip(by_insider["owner"], by_insider["transaction_type"])]
value_colors = [
    COLORS["amber"] if t == "Purchase" else COLORS["slate"] for t in by_insider["transaction_type"]
]

fig = go.Figure(
    go.Bar(
        y=labels,
        x=by_insider["value"] / 1e6,
        orientation="h",
        marker_color=value_colors,
    )
)
fig.update_yaxes(autorange="reversed")
fig.update_layout(
    title="Insider buying concentrates where selling is spread across many reporters",
    xaxis_title="Value of priced open-market trades (USD millions)",
    height=360,
    margin=dict(l=210),
)
fig.show()

# %% [markdown]
# ---
# ## Key Takeaways
#
# 1. A Form 4 is XML with a small, stable schema, and the transaction block is the unit that
#    keeps a trade's code, date, share count and price together. Parse it with an XML parser:
#    a regular expression over the document text has no block boundaries and will interleave
#    fields across trades and across the common-stock and derivative tables.
# 2. A parser is finished when its output can be reconciled against the input. Count the
#    transaction blocks in the raw files by a route that does not use the parser, then account
#    for every block as extracted or excluded for a stated reason. A silent `continue` inside a
#    parse loop is how a panel loses rows without anyone noticing.
# 3. The transaction code decides whether a row means anything. Open-market purchases and sales
#    are decisions; grants, exercises, tax withholding and gifts are compensation, and they
#    dominate the share volume. Any feature built on this panel gates on the code first.
# 4. Missing prices are a normal state of this form, not a data error, and they must be kept
#    out of dollar totals explicitly. Substituting zero produces a total that looks complete
#    and understates itself by however many rows were unpriced.
# 5. A Form 4 may be filed jointly, and the reporting owners are a list rather than a name.
#    Reading the first one found attaches the trades to the wrong insider whenever the document
#    carries more than one.
# 6. For interactive single-company work, [`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb)
#    reaches the same data through a library; the raw path here is what scales to many issuers
#    and what lets the reconciliation above be written at all.
