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
# # S&P 100 SEC Filings: Data Overview
#
# **Chapter 23: Knowledge Graphs for Financial AI**
#
# **Docker image**: `ml4t`
#
# This notebook loads pre-downloaded 10-K and 8-K filings for S&P 100 companies
# and previews the data that feeds the knowledge graph construction pipeline.
#
# **Learning Objectives**:
# - Understand the scale and structure of SEC filing data for KG construction
# - Inspect text excerpt quality (supplier mentions, event descriptions)
# - Verify data coverage across companies and years
#
# **Book Reference**: Chapter 23, Section 23.2 (Constructing Financial Knowledge Graphs)
#
# **Data Download**: Filing data is acquired via the unified SEC download script:
# ```bash
# uv run python data/equities/fundamentals/filings_download.py --form 10-K --universe sp100 --years 2020-2025
# uv run python data/equities/fundamentals/filings_download.py --form 8-K --universe sp100 --years 2020-2025
# ```
# See Chapter 4 for details on SEC EDGAR data acquisition.
#
# **Prerequisites**: Run the SEC filing download script above, or use the staged
# parquet artifacts from the data bundle (loaded here via `load_sec_filings`).

# %%
"""Preview S&P 100 SEC filings for the knowledge graph pipeline."""

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap

from data import load_sec_filings
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# %% tags=["parameters"]
MAX_DISPLAY = 10

# %% [markdown]
# ### Input Contract
#
# The staged Parquet files must preserve the required filing schema, unique
# symbol-accession keys, consistent form labels, and exact text-length metadata.


# %%
def validate_filings(filings: pl.DataFrame, expected_form: str) -> None:
    """Fail when a staged filing table violates its reader-facing contract."""
    required = {
        "symbol",
        "cik",
        "form",
        "filing_date",
        "accession_no",
        "company_name",
        "year",
        "text",
        "text_length",
    }
    assert required <= set(filings.columns), f"Missing columns: {required - set(filings.columns)}"
    assert filings.filter(pl.any_horizontal(pl.col(list(required)).is_null())).is_empty()
    assert filings.select(pl.struct(["symbol", "accession_no"]).is_duplicated().sum()).item() == 0
    assert filings["form"].unique().to_list() == [expected_form]
    assert filings.filter(pl.col("year") != pl.col("filing_date").dt.year()).is_empty()
    assert filings.filter(pl.col("text").str.len_chars() != pl.col("text_length")).is_empty()


# %% [markdown]
# ## 10-K Annual Reports
#
# Annual reports contain supplier relationships, risk factors, and business
# descriptions that feed the supply chain knowledge graph.

# %%
filings_10k = load_sec_filings("10-K", universe="sp100")
validate_filings(filings_10k, "10-K")

print(f"10-K filings: {len(filings_10k):,}")
print(f"Companies: {filings_10k['symbol'].n_unique()}")
if "year" in filings_10k.columns:
    print(f"Year range: {filings_10k['year'].min()}-{filings_10k['year'].max()}")
print(f"Total text: {filings_10k['text_length'].sum():,} chars")
print(f"Avg text per filing: {filings_10k['text_length'].mean():,.0f} chars")

# %%
filings_10k.select(pl.exclude("text")).head(MAX_DISPLAY)

# %% [markdown]
# ## 8-K Event Filings
#
# Current reports contain discrete corporate events (M&A, leadership changes,
# material agreements) used for temporal knowledge graph construction.

# %%
filings_8k = load_sec_filings("8-K", universe="sp100")
validate_filings(filings_8k, "8-K")

print(f"8-K filings: {len(filings_8k):,}")
print(f"Companies: {filings_8k['symbol'].n_unique()}")
print(f"Avg text: {filings_8k['text_length'].mean():,.0f} chars")

filings_8k.select(pl.exclude("text")).head(MAX_DISPLAY)

# %% [markdown]
# ## Coverage Matrix
#
# Which companies have 10-K filings for which years?

# %%
if "year" in filings_10k.columns:
    coverage = (
        filings_10k.group_by("year")
        .agg(pl.col("symbol").n_unique().alias("companies"), pl.len().alias("filings"))
        .sort("year")
    )

# %% [markdown]
# ### Filing coverage
#
# One cell per company-year. Two details decide whether the picture is
# readable, and the notebook got both wrong before this pass.
#
# **The columns have to be put in year order.** `pivot(on="year")` returns
# columns in the order the years happen to appear in the frame, which after a
# `unique()` is arbitrary - here it came out 2022, 2024, 2025, 2020, 2023,
# 2021 - while the axis was labelled with a sorted year list. Every column in
# the rendered heatmap carried the wrong year, and the one company with a gap
# appeared to be missing the wrong ones.
#
# **The rows have to be ordered so the gaps are visible.** The panel is nearly
# complete, so a few missing cells among six hundred, in alphabetical order,
# are a scatter of pixels nobody will find. Sorting by coverage puts the
# incomplete companies at the top where the chart can be read.

# %%
if "year" in filings_10k.columns:
    years = sorted(filings_10k["year"].unique().to_list())
    presence_df = (
        filings_10k.select("symbol", "year")
        .unique()
        .with_columns(pl.lit(1).alias("present"))
        .pivot(index="symbol", on="year", values="present")
        .fill_null(0)
        # Column order comes from the pivot, not from the data. Name the years.
        .select("symbol", *[str(year) for year in years])
        .with_columns(pl.sum_horizontal([str(year) for year in years]).alias("years_present"))
        .sort(["years_present", "symbol"])
    )
    symbols = presence_df["symbol"].to_list()
    presence = presence_df.select([str(year) for year in years]).to_numpy()
    missing_company_years = int(presence.size - presence.sum())
    incomplete = presence_df.filter(pl.col("years_present") < len(years))
    print(f"Panel: {len(symbols)} companies x {len(years)} years = {presence.size} cells")
    print(f"Missing company-years: {missing_company_years}")
    print(f"Companies with a gap: {dict(incomplete.select('symbol', 'years_present').iter_rows())}")

    fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"], constrained_layout=True)
    coverage_cmap = ListedColormap([COLORS["silver_muted"], COLORS["blue"]])
    ax.imshow(presence, aspect="auto", cmap=coverage_cmap, interpolation="nearest")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    y_positions = list(range(0, len(symbols), 10))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([symbols[i] for i in y_positions], fontsize=7)
    ax.set_xlabel("Filing year")
    ax.set_ylabel("S&P 100 company, fewest filing years first")
    add_message_title(
        ax,
        "10-K coverage by company and year",
        subtitle="Dark cells are present; companies sorted by how many years they cover",
    )
    show_with_alt(
        fig,
        "A tall two-colour grid, one row per S&P 100 company and one column per filing year "
        "from 2020 to 2025, with dark cells marking a filing present. Almost the entire "
        "panel is dark. The rows are ordered by how many years each company covers, so the "
        "only visible light cells are in the first row at the top, where a single company "
        "is missing every year but the last.",
    )

# %% [markdown]
# A missing cell is a coverage fact and not evidence about its cause. A company
# can be absent because it joined the index late, because it changed its filer
# identity, or because the download missed it, and the panel cannot tell those
# apart. Downstream work should carry the gap rather than impute a filing into
# it.

# %% [markdown]
# ## Text Length Distribution
#
# Compare text excerpt lengths between 10-K and 8-K filings. 10-K filings
# provide longer narrative sections (supplier mentions, risk factors) while
# 8-K filings are shorter event disclosures.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"], constrained_layout=True)
bins = np.linspace(
    0, max(filings_10k["text_length"].max(), filings_8k["text_length"].max()) + 500, 40
)
ax.hist(
    filings_10k["text_length"].to_numpy(),
    bins=bins,
    alpha=0.7,
    label=f"10-K ({len(filings_10k):,} filings)",
    color=COLORS["blue"],
)
ax.hist(
    filings_8k["text_length"].to_numpy(),
    bins=bins,
    alpha=0.7,
    label=f"8-K ({len(filings_8k):,} filings)",
    color=COLORS["amber"],
)
ax.set_xlabel("Text Length (characters)")
ax.set_ylabel("Number of Filings")
modal_10k_length = int(filings_10k["text_length"].mode()[0])
add_message_title(
    ax,
    "Excerpt length by filing form",
    subtitle="Counts of filings per length bin; both forms on one axis",
)
ax.legend()
show_with_alt(
    fig,
    "Two overlaid histograms of extracted text length in characters. The 10-K series is "
    "concentrated in one tall spike well to the right, with a smaller spike beyond it and "
    "almost nothing elsewhere. The 8-K series sits entirely to the left of the 10-K spike, "
    "spread across a range of shorter lengths rather than concentrated at one value.",
)

# %% [markdown]
# The 10-K spikes are the download's fixed extraction windows rather than a
# property of annual reports - `01_sec_filing_pipeline` in chapter 22 takes
# that apart. The 8-K distribution is spread because event disclosures vary in
# length and the 8-K rule keeps the opening rather than a fixed window.
#
# So length is a diagnostic of the extraction, not a measure of how much a
# filing says, and it is not comparable across the two forms.

# %% [markdown]
# ## Text Quality Check
#
# Preview text excerpts to verify supplier mention extraction works.

# %%
sample = filings_10k.filter(pl.col("text_length") > 5000).head(3)
for row in sample.iter_rows(named=True):
    print(f"\n{'=' * 60}")
    print(f"{row['symbol']} ({row.get('year', 'N/A')}): {row['text_length']:,} chars")
    print(row["text"][:500] + "...")

# %% [markdown]
# ## Key Takeaways
#
# 1. 10-K filings provide structured annual narratives - supplier mentions and
#    risk factors are the primary input for supply chain KG construction
# 2. 8-K filings capture discrete events - M&A, leadership, material agreements
#    feed the temporal edge layer
# 3. Coverage is nearly complete, but gaps remain explicit rather than imputed
# 4. Fixed extraction windows make text length a pipeline diagnostic, not a
#    direct measure of filing informativeness
# 5. A pivot returns its columns in whatever order the data supplied them.
#    Labelling those columns from a separately sorted list mislabels every one
#    of them, and a heatmap gives no hint that it happened - name the columns
#    when you select them
#
# **Next**: [`02_supply_chain_kg_construction`](02_supply_chain_kg_construction.ipynb)
# extracts supply-chain relations from these filings with a local LLM.

# %%
completion_record = {
    "filings_10k": filings_10k.height,
    "filings_8k": filings_8k.height,
    "companies_10k": filings_10k["symbol"].n_unique(),
    "companies_8k": filings_8k["symbol"].n_unique(),
    "year_min": filings_10k["year"].min(),
    "year_max": filings_10k["year"].max(),
    "missing_company_years": missing_company_years,
    "modal_10k_text_length": modal_10k_length,
}
print("COMPLETION_RECORD=" + json.dumps(completion_record, sort_keys=True))
