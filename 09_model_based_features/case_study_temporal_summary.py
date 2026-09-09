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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-Case-Study Temporal Feature Inventory
#
# **Chapter 9 | Section 9.7**
#
# **Docker image**: `ml4t`
#
# The nine case studies each run a model-based feature stage, and each writes one artifact:
# `features/model_based.parquet`. This notebook reads the **schema** of those artifacts, so what
# it can report is what columns exist and what they are called, across nine markets, in one place.
#
# Being precise about that boundary is the point of the notebook. A column name is not a fitted
# model and a column count is not a contribution. Whether any of these features predicts anything
# is measured inside each case study's model stage against the results registry, on held-out
# folds, and nothing here substitutes for that. What an inventory does answer is narrower and
# still worth asking: which markets got which kinds of feature, whether the naming convention the
# chapter established was followed, and where a stage produced nothing at all.
#
# **Learning objectives**
#
# - Read a feature stage's output by its schema rather than by loading it, and know what that
#   restricts you to saying.
# - Compare which families of model-based feature appear across nine markets, and see that the
#   coverage is uneven for reasons specific to each market.
# - Recognize the difference between a naming-convention check and a model inventory, which is
#   what a substring match on column names actually gives you.
#
# **Book reference**
#
# Chapter 9, Section 9.7 (Summary).
#
# **Prerequisites**
#
# Each case study's model-based feature stage must have run and written
# `features/model_based.parquet`. That artifact is produced by a pipeline run and is not in the
# repository, so in a fresh checkout this notebook reports an empty inventory. That is the
# correct output and the tables below say which case studies are missing rather than implying
# the features do not exist.

# %% [markdown]
# ## Setup

# %%
"""Cross-case-study temporal feature inventory, read from artifact schemas."""

import plotly.graph_objects as go
import polars as pl
from IPython.display import display

from utils.paths import get_case_study_dir
from utils.style import COLORS, ml4t_palette, show_plotly_with_alt

# %% tags=["parameters"]
CASE_STUDIES = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]
SAMPLE_NAMES = 8

# %%
DISPLAY_NAMES = {
    "etfs": "ETFs",
    "crypto_perps_funding": "Crypto perpetuals",
    "nasdaq100_microstructure": "Nasdaq-100 microstructure",
    "sp500_equity_option_analytics": "S&P 500 equity and options",
    "us_firm_characteristics": "US firm characteristics",
    "fx_pairs": "FX pairs",
    "cme_futures": "CME futures",
    "sp500_options": "S&P 500 options",
    "us_equities_panel": "US equities panel",
}

IDENTIFIER_COLUMNS = {
    "timestamp",
    "date",
    "symbol",
    "product",
    "asset",
    "stock_id",
    "instrument_id",
}

# The names a model-based feature is expected to carry, one per family the chapter builds. A
# column is attributed to a family when its name contains the family's token.
FAMILY_TOKENS = {
    "garch": "conditional volatility",
    "hmm": "hidden Markov states",
    "regime": "regime indicators",
    "kalman": "filtered state",
    "arima": "autoregressive forecasts",
    "ffd": "fractional differencing",
    "spectral": "frequency decomposition",
    "bayesian": "posterior summaries",
    "har": "multi-horizon volatility",
    "hurst": "roughness",
}

# %% [markdown]
# ## Reading nine schemas
#
# `pl.scan_parquet(...).collect_schema()` reads the file's footer and returns the column names
# and types without reading a row. On artifacts that run to millions of rows across nine markets
# that is the difference between a notebook that opens and one that does not, and for an
# inventory it is all that is needed.
#
# The identifier columns are excluded from every count. A timestamp and a symbol are what a
# feature is indexed by and not features, and leaving them in would credit each case study with
# two features it did not build.


# %%
def inventory_one(case_study: str) -> dict:
    """Column names of one case study's model-based artifact, or a row saying it is absent."""
    artifact = get_case_study_dir(case_study) / "features" / "model_based.parquet"
    if not artifact.exists():
        return {"case_study": case_study, "present": False, "columns": []}

    names = pl.scan_parquet(artifact).collect_schema().names()
    return {
        "case_study": case_study,
        "present": True,
        "columns": [name for name in names if name not in IDENTIFIER_COLUMNS],
    }


inventory = [inventory_one(case_study) for case_study in CASE_STUDIES]
present = [entry for entry in inventory if entry["present"]]
absent = [entry["case_study"] for entry in inventory if not entry["present"]]

print(f"Case studies with a model-based artifact: {len(present)} of {len(CASE_STUDIES)}")
if absent:
    print("Absent: " + ", ".join(DISPLAY_NAMES[case_study] for case_study in absent))
    print("An absent artifact means that stage has not run in this checkout, nothing more.")

# %% [markdown]
# ## What each market built
#
# The table counts columns per case study and attributes each column to a family by the token its
# name contains. Two properties of that attribution decide how the table can be read.
#
# It is a **check on names**, not a detection of models. A case study could fit a hidden Markov
# model and name its output `state_probability`, and this table would not see it. The value of
# running the check anyway is that the convention exists precisely so a downstream reader can
# tell what a column is without opening the notebook that made it, and a missing token is a
# convention that was not followed.
#
# A column can also be counted **more than once**, since a name containing both `regime` and
# `hmm` matches both tokens. The per-family counts therefore do not sum to the column count, and
# the table reports both so the difference is visible rather than reconciled.

# %%
family_rows = []
for entry in present:
    columns = [name.lower() for name in entry["columns"]]
    row = {
        "case study": DISPLAY_NAMES[entry["case_study"]],
        "columns": len(columns),
        "columns matching no family": sum(
            1 for name in columns if not any(token in name for token in FAMILY_TOKENS)
        ),
    }
    for token in FAMILY_TOKENS:
        row[token] = sum(1 for name in columns if token in name)
    family_rows.append(row)

family_rows.sort(key=lambda row: row["columns"], reverse=True)

if family_rows:
    families = pl.DataFrame(family_rows)
    display(families)
    print(
        f"Families present in at least one case study: {sum(1 for token in FAMILY_TOKENS if families[token].sum() > 0)} of {len(FAMILY_TOKENS)}"
    )
else:
    print("No artifact to inventory. Run the case study model-based feature stages first.")

# %% [markdown]
# The column worth reading is the count of columns matching no family token. A high value there is
# either a market whose model-based features are genuinely something else, or a stage that used a
# different word for a family that is present. Which of the two it is cannot be read off this
# table, and the sample of names at the end of the notebook is where a reader resolves it: an
# abbreviation for a family the chapter spelled out is the second case, and an unfamiliar quantity
# is the first.

# %%
if family_rows:
    figure = go.Figure(
        go.Bar(
            x=[row["case study"] for row in family_rows],
            y=[row["columns"] for row in family_rows],
            marker_color=COLORS["blue"],
            text=[str(row["columns"]) for row in family_rows],
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Model-based columns per market, counted from the schema",
        yaxis_title="Columns",
        xaxis_tickangle=-40,
    )
    show_plotly_with_alt(
        figure,
        "A bar chart of the number of model-based feature columns each case study produced, "
        "ordered from most to fewest, with the count printed above each bar.",
    )

# %% [markdown]
# ## The same counts by family
#
# The grouped chart is the naming coverage drawn: how many of the nine markets have a column whose
# name carries each family's token. It is a picture of what was built and how it was named, and it
# supports neither of the two conclusions a reader will reach for. A family appearing in one
# market is not therefore asset-specific, since another market may have built it under a different
# name or not built it for reasons of scope rather than fit. And a family appearing everywhere is
# not therefore general, since nothing here says it predicted anything in any of them.

# %%
if family_rows:
    # ml4t_palette caps at the palette's own length, so it is cycled rather than indexed.
    palette = ml4t_palette(len(family_rows), categorical=True)
    figure = go.Figure()
    for position, row in enumerate(family_rows):
        color = palette[position % len(palette)]
        figure.add_trace(
            go.Bar(
                name=row["case study"],
                x=list(FAMILY_TOKENS),
                y=[row[token] for token in FAMILY_TOKENS],
                marker_color=color,
            )
        )
    figure.update_layout(
        title="Which families of model-based feature reached which market",
        yaxis_title="Columns",
        barmode="group",
        xaxis_tickangle=-40,
    )
    show_plotly_with_alt(
        figure,
        "A grouped bar chart with one group per model family and one bar per case study inside "
        "it, showing how many columns each market contributed to each family. Most groups are "
        "occupied by a few markets rather than all of them, and three families - bayesian, "
        "har and hurst - carry no bars at all.",
    )

# %% [markdown]
# ## A sample of the names themselves
#
# The counts above hide what the columns actually are, and this is where the unmatched ones get
# resolved. Each row pairs a column with the families its name matched, so a row matching none is
# either an abbreviation the token list does not carry or a quantity outside the ten families. The
# sample is the first `SAMPLE_NAMES` columns of each artifact in schema order, which is the order
# the stage wrote them and therefore roughly the order its notebook built them.
#
# Extending `FAMILY_TOKENS` with the abbreviations this table turns up is the obvious next step
# and it is also the thing to be careful about: a token list grown until nothing is unmatched
# stops being a convention check and becomes a record of whatever names happen to exist.

# %%
if present:
    display(
        pl.DataFrame(
            [
                {
                    "case study": DISPLAY_NAMES[entry["case_study"]],
                    "column": name,
                    "families its name matches": ", ".join(
                        token for token in FAMILY_TOKENS if token in name.lower()
                    )
                    or "none",
                }
                for entry in present
                for name in entry["columns"][:SAMPLE_NAMES]
            ]
        )
    )

# %% [markdown]
# ## Takeaways
#
# 1. **A schema read is the right tool for an inventory and the wrong one for a claim.** Reading
#    nine parquet footers costs nothing and tells you what exists. It cannot tell you what any
#    column is worth, and the registry comparisons in each case study's model stage are what can.
# 2. **A substring match on column names is a convention check.** It finds the families that were
#    named for what they are and misses the ones that were not, so a zero in this table is
#    ambiguous between a family that is absent and a family that was named differently. The sample
#    of names is what resolves a given case, and widening the token list until nothing is
#    unmatched would remove the check rather than pass it.
# 3. **Uneven coverage is a fact about what was built and named, not about what suits a market.**
#    A family missing from a market's schema may be absent, differently named, or out of that case
#    study's scope, and a schema cannot separate the three. Deciding whether a technique suits a
#    market takes the case study's own evaluation, not this table.
# 4. **An absent artifact is a statement about this checkout.** These files are produced by
#    pipeline runs and are not committed, so an empty inventory here means the stages have not
#    run, and reading it as an absence of features is the mistake this notebook is arranged to
#    prevent.
#
# **Next**: Chapter 10 applies the same discipline to features built from text, where the fitted
# object is an embedding model and the binding constraint is when it was trained.
