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
from utils.style import COLORS, FIGSIZE, add_message_title

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
# ### Filing Coverage Heatmap
#
# Visualize which companies have filings in each year. Gaps reveal
# missing data that could affect downstream KG completeness.

# %%
if "year" in filings_10k.columns:
    years = sorted(filings_10k["year"].unique().to_list())
    symbols = sorted(filings_10k["symbol"].unique().to_list())
    presence_df = (
        filings_10k.select("symbol", "year")
        .unique()
        .with_columns(pl.lit(1).alias("present"))
        .pivot(index="symbol", on="year", values="present")
        .sort("symbol")
        .fill_null(0)
    )
    presence = presence_df.select(pl.exclude("symbol")).to_numpy()
    missing_company_years = int(presence.size - presence.sum())
    gap_label = "company-year is" if missing_company_years == 1 else "company-years are"

    fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"], constrained_layout=True)
    coverage_cmap = ListedColormap([COLORS["silver_muted"], COLORS["blue"]])
    ax.imshow(presence, aspect="auto", cmap=coverage_cmap, interpolation="nearest")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    y_positions = range(0, len(symbols), 10)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([symbols[i] for i in y_positions], fontsize=7)
    ax.set_xlabel("Filing Year")
    ax.set_ylabel("S&P 100 Company")
    add_message_title(
        ax,
        f"Only {missing_company_years} {gap_label} missing from the 10-K panel",
        subtitle=f"{len(symbols)} symbols across {len(years)} filing years; dark cells are present",
    )
    fig.show()

# %% [markdown]
# The nearly complete panel supports longitudinal description. A missing cell is
# a coverage fact, not evidence about its cause; downstream work should retain
# the gap rather than impute a filing.

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
    f"10-K excerpts cluster at {modal_10k_length:,} characters",
    subtitle="8-K event disclosures have a broader, shorter distribution",
)
ax.legend()
fig.show()

# %% [markdown]
# The modal 10-K length exposes the fixed extraction window. The broader 8-K
# distribution reflects varying event-disclosure length, so document length
# should not be treated as a comparable information-volume measure across forms.

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
# 4. Fixed extraction windows make text length a pipeline diagnostic, not a direct
#    measure of filing informativeness
#
# **Next**: See `02_supply_chain_kg_construction.py` for LLM-based entity extraction.

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
