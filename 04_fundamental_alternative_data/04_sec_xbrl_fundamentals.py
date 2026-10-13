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
# # SEC XBRL Fundamentals
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
#
# **Section Reference**: Section 4.1 (The Point-in-Time Pipeline)
#
# ## Purpose
#
# A fundamental value has two dates. There is the period it describes, and there is the day the
# filing carrying it reached the SEC. Between the two, nobody in the market knew the number. A
# backtest that keys a fundamental to its period end is therefore trading on information it did
# not have, and the error is invisible in the results: the strategy simply looks better than it
# was.
#
# This notebook reads the quarterly panel that `data/equities/fundamentals/xbrl_download.py`
# assembles from the SEC's XBRL Frames API, measures how far apart those two dates actually
# are, and builds the query that respects the gap. Chapter 8 turns the panel into features.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Read a quarterly fundamentals panel that carries both the period a value describes and the
#   date every value in that row was public.
# - Measure how completely each reported concept covers the companies and quarters in a panel,
#   and say why a concept is missing where it is.
# - Measure the delay between a fiscal quarter's end and the date its row became public, and
#   explain why a Frames-sourced panel is not point-in-time safe on its own.
# - Write an as-of query that returns only what was public on a given date, filtering on the
#   filing date and ranking on the period.
# - Show what the same query returns when it filters on the period instead, and count the
#   companies for which the two answers differ.
#
# ## Why Direct API vs edgartools?
#
# The `02_sec_filing_explorer.py` notebook demonstrates using the edgartools library for
# **individual company analysis**: exploring filings, extracting financial statements,
# and parsing 13F holdings. edgartools is excellent for deep dives into specific companies.
#
# This notebook takes a different approach: **bulk data retrieval** using the SEC's
# XBRL Frames API, which provides aggregated data across all filers in a single request.
# This is more efficient for building cross-sectional fundamental datasets.
#
# | Use Case | Best Tool |
# |----------|-----------|
# | Analyze a single company's filings | `02_sec_filing_explorer.py` |
# | Build factor dataset for 20+ stocks | This notebook (XBRL Frames API) |
# | Parse complex filing documents | `02_sec_filing_explorer.py` |
# | Get quarterly aggregates across market | This notebook (XBRL Frames API) |
#
# ## Point-in-Time (PIT) Correctness
#
# **Critical**: Fundamental data for backtesting must reflect only information
# available at each historical date. Using fiscal quarter end dates causes
# look-ahead bias because filings are released 30-60 days later.
#
# The downloader joins **filing dates** from the SEC Submissions API onto the
# XBRL frames so each row carries both `fiscal_quarter_end` (valid time) and
# `announcement_date` (knowledge time):
#
# | Date Type | Usage |
# |-----------|-------|
# | `fiscal_quarter_end` | Period the data describes (e.g., 2024-03-31) |
# | `announcement_date` | When SEC filing was submitted (e.g., 2024-05-02) |
#
# For backtesting, use `announcement_date` as the point when data becomes available.
#
# A row is filled from several Frames calls, and its facts can come from different filings, so
# `announcement_date` is the filing date of the **last** fact the row carries - the date on
# which everything in it was public. Dating the row any earlier would hand a backtest a number
# before it existed.
#
# ## Data Sources
#
# The downloader uses two free, public SEC EDGAR APIs (no vendor subscription):
#
# 1. **XBRL Frames API** — aggregated financial data across all filers:
#    `https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json`
#
#    **CY vs FY Frames**: The API uses **CY** (calendar year) quarters, not fiscal year.
#    This is intentional: CY frames provide cross-sectional snapshots where all companies
#    report the same calendar period, enabling apples-to-apples comparisons.
#    Companies with non-calendar fiscal years (e.g., MSFT ends June 30) have their
#    data mapped to the appropriate CY quarter.
#
# 2. **Submissions API** — per-company filing metadata cached per CIK on first
#    fetch so re-runs do not re-hit the endpoint:
#    `https://data.sec.gov/submissions/CIK{cik}.json`
#
# ## Downloader
#
# ```bash
# # 20 large-cap US equities × 2022-2024 × 11 standard concepts (~2-3 min)
# uv run python data/equities/fundamentals/xbrl_download.py
#
# # Custom year range or CIK list
# uv run python data/equities/fundamentals/xbrl_download.py --years 2020,2021,2022,2023,2024
# uv run python data/equities/fundamentals/xbrl_download.py --ciks 320193
# ```
#
# The loader raises `DataNotFoundError` with the exact command if the parquet
# is missing — no hidden HTTP calls inside the notebook.
#
# ## Cross-References
#
# - **Related**: [`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb) (individual company SEC filings)
# - **Downstream**: `08_financial_features/04_fundamentals_macro_calendar.py` (factor engineering)

# %%
"""SEC XBRL Fundamentals — analyze quarterly fundamentals from the canonical xbrl_download.py output."""

import plotly.graph_objects as go
import polars as pl

from data import load_sec_xbrl_fundamentals

# Importing utils.style registers and activates the ML4T Plotly template
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# The as-of demonstration in Part 4 needs a date to be run for. Any date inside the panel's
# range works. One falling shortly after a quarter end shows the widest gap between a query
# that respects the filing date and one that does not, because that is the moment when the most
# recent quarter has ended and has not yet been reported.

# %% tags=["parameters"]
AS_OF_DATE = "2023-06-30"  # the simulated date the Part 4 queries are run for

# %% [markdown]
# ## 1. Load the fundamentals panel
#
# The downloader ships a default universe of twenty large-cap US equities with eleven standard
# us-gaap concepts. Each row is one company and one fiscal quarter, and carries the period end,
# the filing date, the accession number of the filing it came from, and the concepts that
# company reported for that period.

# %%
fundamentals = load_sec_xbrl_fundamentals()

print(f"Rows:                 {len(fundamentals):,}")
print(f"Companies:            {fundamentals['cik'].n_unique()}")
print(f"Distinct period ends: {fundamentals['fiscal_quarter_end'].n_unique()}")
print(f"Earliest period end:  {fundamentals['fiscal_quarter_end'].min()}")
print(f"Latest period end:    {fundamentals['fiscal_quarter_end'].max()}")
print(f"Columns:              {fundamentals.columns}")

# %% [markdown]
# ### What the panel holds
#
# Two views of the same rows, one for the balance sheet and one for the income and cash-flow
# concepts, are enough to see the shape. `announcement_date` sits beside `fiscal_quarter_end`
# in both, because every question in the rest of this notebook is about the distance between
# them.

# %%
balance_cols = [
    c
    for c in [
        "symbol",
        "entity_name",
        "fiscal_quarter_end",
        "announcement_date",
        "assets",
        "liabilities",
        "stockholdersequity",
    ]
    if c in fundamentals.columns
]
fundamentals.select(balance_cols).head(8)

# %% [markdown]
# `revenues` is sparse, and the reason is a taxonomy change rather than a gap in the data.
# Accounting Standards Codification Topic 606, effective for most filers from 2018, replaced
# the general revenue concept with one that names the source of the revenue, and large filers
# now tag their top line as `RevenueFromContractWithCustomerExcludingAssessedTax`. The
# downloader keeps that in its own column rather than folding it into `revenues`, so a null
# here means "reported under a different concept", not "did not report".

# %%
flow_cols = [
    c
    for c in [
        "symbol",
        "fiscal_quarter_end",
        "announcement_date",
        "revenues",
        "netincomeloss",
        "netcashprovidedbyusedinoperatingactivities",
    ]
    if c in fundamentals.columns
]
fundamentals.select(flow_cols).head(8)

# %% [markdown]
# ## 2. Coverage
#
# Before anything is computed from a panel, it is worth knowing which cells of it are actually
# filled. A concept missing for one company in one quarter is a null a feature will have to
# handle; a concept missing for a whole industry is a signal that the industry tags it
# differently, and no amount of imputation will recover it. The grid below counts the rows
# carrying `assets` for each company and calendar quarter.

# %%
coverage_df = (
    fundamentals.with_columns(
        (
            pl.col("fiscal_quarter_end").dt.year().cast(pl.Utf8)
            + "Q"
            + ((pl.col("fiscal_quarter_end").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        ).alias("quarter")
    )
    .group_by(["symbol", "quarter"])
    .agg(
        pl.col("assets").is_not_null().sum().alias("assets_available"),
    )
)

coverage_pivot = coverage_df.pivot(
    on="quarter", index="symbol", values="assets_available"
).fill_null(0)

quarters = sorted(c for c in coverage_pivot.columns if c != "symbol")
coverage_pivot = coverage_pivot.sort("symbol")
symbols = coverage_pivot["symbol"].to_list()
matrix = coverage_pivot.select(quarters).to_numpy().astype(int)

# %%
fig = go.Figure(
    data=go.Heatmap(
        z=matrix,
        x=quarters,
        y=symbols,
        colorscale=[[0, COLORS["silver"]], [1, COLORS["blue"]]],
        showscale=False,
        text=[[str(v) if v > 0 else "" for v in row] for row in matrix.tolist()],
        texttemplate="%{text}",
        textfont={"size": 10},
    )
)
fig.update_layout(
    title="Total assets are reported in nearly every company-quarter",
    xaxis_title="Calendar quarter",
    yaxis_title="Symbol",
    height=600,
    width=800,
)
show_plotly_with_alt(
    fig,
    "Heatmap of company against calendar quarter, shaded where total assets were reported and annotated with the number of filings in each cell. Almost every cell is filled, with a scattering of blanks and a few cells holding two filings.",
)

# The grid is keyed on calendar quarter, and a filer on a 52/53-week calendar can close two
# fiscal periods inside one of them - AAPL and AMD both do in 2023Q3 - which is where a "2"
# comes from. Those are two distinct periods, not two versions of one, so the cell is more
# densely covered than its neighbours rather than doubly reported.
total_cells = matrix.size
filled_cells = int((matrix > 0).sum())
print(f"Company-quarter cells: {total_cells}")
print(f"Cells reporting total assets: {filled_cells} ({filled_cells / total_cells:.1%})")

# %% [markdown]
# ## 3. How stale a Frames panel is
#
# The gap between a period end and the date the row became public decides how far a fundamental
# has to be lagged before a backtest may use it. For a panel built from original filings that
# gap is the statutory window - forty days for a large filer's 10-Q. This panel is not built
# from original filings, and the measurement below is what that costs.

# %%
filing_lag = (
    fundamentals.filter(pl.col("announcement_date").is_not_null())
    .with_columns(
        (pl.col("announcement_date") - pl.col("fiscal_quarter_end"))
        .dt.total_days()
        .alias("lag_days")
    )
    .select(["symbol", "fiscal_quarter_end", "lag_days"])
)

filing_lag.select("lag_days").describe()

# %% [markdown]
# Not one row is inside the statutory window. The minimum is over a year, where a large
# filer's 10-Q is due within forty days of quarter end.
#
# That is not a filing delay, it is what the Frames API returns. Frames answers with each fact
# as it appears in the *latest* filing reporting the period, and for every quarter here at
# least one concept - normally the income statement - has since been re-reported in a
# comparative filed a year later. The row carries that restated figure, so the row was not
# public until that comparative was filed, and dating it by the last of its filings is the only
# honest answer. Dating it by the earliest instead would publish a restated number under the
# original's date, which is exactly the lookahead Figure 4.1 of the chapter illustrates.
#
# This is the chapter's point about Frames made concrete: it is built for cross-sectional
# snapshots and prototyping, and it is not point-in-time safe on its own. A panel that has to
# answer "what was known on this date" has to be reconstructed from filing-level records, where
# each fact keeps the value and the date of the filing that first reported it. The histogram
# shows the shape of the cost: one late cluster, not a statutory window with a tail.

# %%
lags = filing_lag["lag_days"].to_list()
median_lag = filing_lag["lag_days"].median()

fig = go.Figure(
    data=go.Histogram(
        x=lags,
        xbins={"start": 0, "end": 1200, "size": 25},
        marker_color=COLORS["blue"],
    )
)
fig.add_vline(
    x=median_lag,
    line_dash="dash",
    line_color=COLORS["amber"],
    annotation_text=f"median {median_lag:.0f}d",
    annotation_position="top",
    annotation_font_color=COLORS["amber"],
)
fig.update_layout(
    title="Days from fiscal quarter end to the date the row became public",
    xaxis_title="Filing lag (days between fiscal-quarter end and announcement)",
    yaxis_title="Company-quarters",
    height=420,
    width=800,
)
show_plotly_with_alt(
    fig,
    "Histogram of the number of days between a fiscal quarter end and the date every value in its row was public, with a dashed rule at the median. Nothing falls inside the first year; the mass sits in a cluster a little past one year, with a thinner spread running out beyond three.",
)

# %% [markdown]
# ## 4. Querying two time dimensions at once
#
# A panel carrying both dates is **bitemporal**: each row has a *valid time*, the period the
# fact describes, and a *knowledge time*, the date the fact became public. Here they are
# `fiscal_quarter_end` and `announcement_date`.
#
# The two decide different things, and the whole correctness of an as-of query is in keeping
# them apart. Knowledge time decides **admissibility**: a row may be used only if the market
# already knew it on the date being simulated. Valid time decides **recency**: among the rows
# that pass, the one describing the most recent period is the one to carry forward.
#
# Conflating them fails in both directions. Filtering on valid time admits quarters that had
# ended but had not been filed, which is the classic lookahead. Sorting on knowledge time
# returns the most recently *announced* row rather than the most recent *quarter*, so a
# quarter dated by a late comparative walks the answer backwards in fiscal time - and Part 3
# says that is every row in this panel rather than an edge case. Filter on
# `announcement_date`; rank on `fiscal_quarter_end`.


# %%
def query_fundamentals_as_of(df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
    """Return latest known fundamentals as of a specific date (PIT-correct).

    Admissibility is set by `announcement_date` (what the market knew); recency is set
    by `fiscal_quarter_end` (which period the fact describes).
    """
    query_date = pl.lit(as_of_date).str.to_date()
    return (
        df.filter(pl.col("announcement_date") <= query_date)
        # The panel holds one row per company-quarter, so this sort makes the choice
        # deterministic rather than breaking a tie between versions. A Frames-sourced
        # panel carries only the latest filed value for a period and cannot show what
        # that period said before a restatement; reconstructing the earlier version
        # needs the companyfacts API, which returns every filing of every fact.
        .sort(["symbol", "fiscal_quarter_end", "announcement_date"])
        .group_by("symbol", maintain_order=True)
        .last()
    )


# %% [markdown]
# A row with no `announcement_date` has no knowledge time, so it can never be shown to
# be admissible and the filter drops it. That is the right call, but a silent one: the
# coverage it costs is worth stating out loud before relying on the panel.

# %%
n_null = fundamentals["announcement_date"].is_null().sum()
print(f"Rows with no announcement_date: {n_null} of {fundamentals.height}")
print(f"Share of the panel never admissible to an as-of query: {n_null / fundamentals.height:.1%}")


# %% [markdown]
# ### Demonstration: Correct vs Incorrect Queries
#
# Both queries below run on the rows that carry a knowledge time. Restricting the
# comparison to a single universe is what makes it a clean read on lookahead bias: if
# the correct query dropped the null-`announcement_date` rows and the lookahead one kept
# them, the mismatch count would mix the coverage gap into the answer.

# %%
pit_panel = fundamentals.filter(pl.col("announcement_date").is_not_null())
print(f"As of {AS_OF_DATE}, the latest quarter known to the market for each symbol:")

# %% [markdown]
# **Correct** — filter on `announcement_date <= as_of`:

# %%
known_correct = query_fundamentals_as_of(pit_panel, AS_OF_DATE)
cols = [
    c
    for c in ["symbol", "fiscal_quarter_end", "announcement_date", "netincomeloss"]
    if c in known_correct.columns
]
known_correct.select(cols).head(5)

# %% [markdown]
# **Lookahead-biased** — filter on `fiscal_quarter_end <= as_of`. This includes
# quarters whose filings hadn't been submitted yet on the as-of date:

# %%
query_date = pl.lit(AS_OF_DATE).str.to_date()
known_wrong = (
    pit_panel.filter(pl.col("fiscal_quarter_end") <= query_date)
    .sort(["symbol", "fiscal_quarter_end", "announcement_date"])
    .group_by("symbol", maintain_order=True)
    .last()
)
known_wrong.select(cols).head(5)

# %% [markdown]
# Comparing the two — every row here is a symbol the lookahead-biased approach
# would have used a fresher quarter than was actually available.
#
# The join has to start from the biased answer, not the correct one. A symbol can have
# nothing admissible on this date at all, and an inner join from the correct side would
# drop exactly those symbols - the ones where the bias is largest, because the biased
# query picks a quarter for them that the market could not see any part of. Counting a
# missing correct quarter as a mismatch is what keeps them in the total.

# %%
correct_dates = known_correct.select(["symbol", "fiscal_quarter_end"]).rename(
    {"fiscal_quarter_end": "correct_qtr"}
)
wrong_dates = known_wrong.select(["symbol", "fiscal_quarter_end"]).rename(
    {"fiscal_quarter_end": "wrong_qtr"}
)
mismatches = wrong_dates.join(correct_dates, on="symbol", how="left").filter(
    pl.col("correct_qtr").is_null() | (pl.col("correct_qtr") != pl.col("wrong_qtr"))
)
n_nothing_known = mismatches.filter(pl.col("correct_qtr").is_null()).height
print(
    f"{len(mismatches)} of {wrong_dates.height} symbols where lookahead bias would change "
    f"the chosen quarter"
)
print(f"  of those, {n_nothing_known} had nothing admissible on this date at all")
mismatches

# %% [markdown]
# ## Key Takeaways
#
# 1. The SEC's XBRL Frames API is enough to assemble a cross-sectional fundamentals panel with
#    no vendor subscription. What it costs is that the panel arrives in the SEC's terms rather
#    than a vendor's: calendar quarters, us-gaap concept names, and one row per fact as filed.
# 2. A missing concept is usually a tagging difference rather than an absent disclosure.
#    `revenues` is null for filers who tag their top line under the post-ASC-606 concept, so
#    measure coverage per concept before treating a null as a gap, and read the pattern of the
#    nulls before imputing anything.
# 3. A Frames-sourced panel is not point-in-time safe, and measuring it says so: no row here
#    is dated inside the statutory filing window, because Frames returns each fact as last
#    filed and every quarter has at least one concept re-reported a year later. Frames is for
#    cross-sectional snapshots and prototyping. A panel that must answer what was known on a
#    date has to be reconstructed from filing-level records, where each fact keeps the value
#    and the date of the filing that first reported it.
# 4. An as-of query filters on knowledge time and ranks on valid time. Filtering on valid time
#    admits quarters that had not been filed yet, which is lookahead. Ranking on knowledge time
#    returns the most recently announced row rather than the most recent quarter, so it walks
#    backwards in fiscal time every time a restatement lands. The comparison above counts the
#    companies for which the two answers differ on one date.
# 5. Knowledge time can be missing. A row with no filing date can never be shown to have been
#    public, so an as-of query drops it - silently, unless the panel says how many rows that is.
#    Report the gap rather than discovering it later as unexplained coverage loss.
# 6. The downloader and the loader are the production interface. This notebook is the check
#    that the panel is sound and the template for querying it; the feature engineering happens
#    in `08_financial_features/04_fundamentals_macro_calendar.py`.
