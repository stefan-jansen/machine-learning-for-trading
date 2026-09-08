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
# # Chen-Pelger-Zhu Academic Asset Pricing Dataset
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: Section 4.1 (The Point-in-Time Pipeline)
#
# ## Purpose
#
# Research on machine learning for asset pricing is hard to reproduce because the underlying
# firm data usually sits behind a WRDS subscription. Chen, Pelger and Zhu (2021) removed that
# barrier by publishing the panel they used: fifty years of monthly US equity observations,
# each carrying 46 firm characteristics and the return earned over the following month, with
# company names and CRSP identifiers stripped out. This notebook reads that panel, shows what
# is in it, and measures how strongly each characteristic lines up with next-month returns.
#
# It is the reference dataset for the `us_firm_characteristics` case study, so the shape,
# the split boundaries and the normalization conventions established here are the ones every
# model in that case study inherits.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Load the published panel and report how many stocks it covers in each month of its history.
# - Read the 46 characteristics as eight economic groups, and say which quantity each group measures.
# - Confirm that a cross-sectional rank normalization has been applied, by checking the range
#   each characteristic occupies.
# - Measure the information coefficient - the rank correlation between a characteristic and the
#   return that follows it - one month at a time, and average it over the training period.
# - Correct the significance test on that average for the serial correlation between adjacent
#   months, and say how far the correction moves it.
# - State what the anonymous identifiers in this panel allow and what they rule out.
#
# ## Cross-References
#
# - **Downstream**: Ch11 `case_studies/us_firm_characteristics/05_linear.py`, Ch12 `06_gbm.py`
# - **Related**: Ch14 `case_studies/us_firm_characteristics/08_latent_factors.py`
# - **Synthesis**: `case_studies/us_firm_characteristics/10_model_analysis.py`
#
# ## Prerequisites
#
# Download the dataset before running this notebook:
# ```bash
# python data/equities/firm_characteristics/download.py
# ```
#
# ## Data Source
#
# | Attribute | Value |
# |-----------|-------|
# | **Paper** | "Deep Learning in Asset Pricing" (Chen, Pelger, Zhu, 2021) |
# | **Repository** | https://github.com/jasonzy121/Deep_Learning_Asset_Pricing |
# | **Features** | 46 firm characteristics, rank-normalized each month |
# | **Returns** | Next-month excess returns, left in their original scale |
# | **Period** | January 1967 to December 2016 |
# | **Identifiers** | Anonymous integers; the numbering restarts in each split |
#
# ## Data Construction (from paper Section III.A)
#
# ### Stock Universe
# - Source: All securities on CRSP
# - Only stocks with all 46 characteristics available are included
# - That requirement removes predominantly small-cap stocks with missing data
# - The released panel covers roughly two thousand stocks in a typical month
#
# ### 46 Firm Characteristics
# Characteristics are sourced from:
# 1. Kenneth French Data Library
# 2. Freyberger, Neuhierl, and Weber (2020)
#
# **Categories** (cf. Table A.II in paper; this notebook groups them as):
# - **Valuation**: BEME, E2P, S2P, CF2P, D2P, A2ME, Q
# - **Profitability**: ROA, ROE, PROF, OP, PM, PCM, NI, RNA
# - **Investment**: Investment, NOA, OA, AC, AT, D2A
# - **Momentum / past returns**: r2_1, r12_2, r12_7, r36_13, ST_REV, LT_Rev, Rel2High
# - **Risk**: Beta, MktBeta, IdioVol, Variance, Resid_Var
# - **Liquidity / Size**: LME, LTurnover, Spread, SUV
# - **Leverage**: Lev, OL, FC2Y, C, CF, DPI2A
# - **Other**: ATO, CTO, SGA2S
#
# **Construction**:
# - Yearly variables: Updated end of June (Fama-French convention)
# - Monthly variables: Updated end of month for use in next month
# - All from CRSP/Compustat accounting data or CRSP past returns
#
# ### Macroeconomic companion series
# The paper pairs the firm panel with 178 macroeconomic time series: 124 from the FRED-MD
# database (McCracken and Ng, 2016), 46 cross-sectional medians of the firm characteristics
# themselves, and 8 equity-premium predictors from Welch and Goyal (2007). Conditional models
# such as the stochastic discount factor GAN of Chapter 14 use them as state variables. They
# are distributed separately from the firm panel; assemble them from the paper repository and
# join on `timestamp`.
#
# ### Cross-Sectional Rank Normalization
# Following Kelly, Pruitt, Su (2019) and Kozak, Nagel, Santosh (2020), each characteristic is
# ranked across all stocks within a month and the ranks are mapped onto the interval
# $\left[-\frac{1}{2}, +\frac{1}{2}\right]$. Two consequences matter for everything below.
# Every characteristic then occupies the same range whatever its raw units were, so a
# price-to-book ratio and a turnover rate enter a model on equal footing. And because the
# ranking keeps only the ordering, an extreme raw value becomes an extreme rank rather than an
# extreme number, which bounds the influence any one stock can have on a fitted coefficient.
#
# ## Key Concepts
#
# - **Information coefficient (IC)**: the rank correlation, within one month's cross-section,
#   between a characteristic and the return realized over the month that follows. It is the
#   standard summary of how well a single signal orders stocks.
# - **Excess return**: a return measured net of the risk-free rate. The `ret` column holds the
#   excess return of the following month and is the quantity every model here predicts.
# - **Anonymous identifier**: the `symbol` column is an integer standing in for a company whose
#   name was removed. It links a firm's observations across months within one split.
#
# ---

# %%
"""Chen-Pelger-Zhu Academic Asset Pricing Dataset - explore anonymized firm characteristics for ML benchmarking."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from ml4t.diagnostic.metrics import compute_ic_hac_stats, cross_sectional_ic_series

from data import load_firm_characteristics
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_diverging

# Importing utils.style registers and activates the ML4T Plotly template
# (palette, gridlines, fonts) repo-wide; px/go figures below inherit its colorway.

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI
SEED = 42
# The panel's label is the return of the single month that follows, so consecutive monthly
# ICs are built from returns that do not overlap. The HAC lag selection below is told so.
LABEL_HORIZON_MONTHS = 1
# Two characteristics are treated as redundant when the magnitude of their correlation
# reaches this level; it is the threshold used to list pairs in Section 3, not a modelling
# decision, and lowering it lengthens the list rather than changing any fitted result.
REDUNDANCY_THRESHOLD = 0.5

# %%
set_global_seeds(SEED)

# %% [markdown]
# ---
#
# ## Section 1: What is in the panel
#
# The published file holds one row per stock and month. Alongside the 46 characteristics it
# carries `ret`, the excess return of the following month, `split`, the train/validation/test
# label assigned by the authors, and `symbol`, the anonymous integer identifier.

# %%
df = load_firm_characteristics(split="all")

ID_COLS = ["symbol", "timestamp", "ret", "split"]
feature_cols = [c for c in df.columns if c not in ID_COLS]

print(f"Rows: {len(df):,}")
print(f"Characteristics: {len(feature_cols)}")
print(f"Distinct anonymous identifiers: {df['symbol'].n_unique():,}")

# %% [markdown]
# The authors split the history by date rather than at random, which is the only split that
# respects the arrow of time: a model is fitted on the earliest years, tuned on the middle
# ones, and judged on the most recent. The boundaries below are read from the file rather
# than assumed, because every figure in this notebook marks them.

# %%
split_bounds = (
    df.group_by("split")
    .agg(
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
        pl.len().alias("n_obs"),
        pl.col("symbol").n_unique().alias("n_stocks"),
    )
    .sort("start")
)
split_bounds

# %% [markdown]
# The identifiers are drawn from a fresh numbering in each split, so the same integer in the
# training and test periods refers to different companies. That is what the row counts above
# and the pairwise overlap below establish: a firm can be followed from month to month inside
# one split, and cannot be followed across a boundary.

# %%
symbols_by_split = {
    row["split"]: set(df.filter(pl.col("split") == row["split"])["symbol"].unique().to_list())
    for row in split_bounds.iter_rows(named=True)
}
for left, right in [("train", "valid"), ("valid", "test"), ("train", "test")]:
    shared = len(symbols_by_split[left] & symbols_by_split[right])
    print(f"identifiers shared between {left} and {right}: {shared}")

# %% [markdown]
# ### How wide is the cross-section?
#
# The number of stocks available in a given month decides how much a cross-sectional statistic
# computed on that month can be trusted, and it moves a great deal over fifty years. The
# vertical rules mark the two split boundaries read above.

# %%
monthly_counts = df.group_by("timestamp").len().sort("timestamp")

valid_start = split_bounds.filter(pl.col("split") == "valid")["start"].item()
test_start = split_bounds.filter(pl.col("split") == "test")["start"].item()

fig = px.line(
    monthly_counts.to_pandas(),
    x="timestamp",
    y="len",
    title="Coverage rises through the 1990s, peaks in the mid-2000s, then thins",
    labels={"len": "Stocks in the cross-section", "timestamp": "Month"},
    color_discrete_sequence=[COLORS["blue"]],
)
for boundary, label in [(valid_start, "Validation"), (test_start, "Test")]:
    fig.add_vline(x=boundary, line_dash="dash", line_color=COLORS["neutral"])
    fig.add_annotation(
        x=boundary,
        y=1.0,
        yref="paper",
        yanchor="bottom",
        text=label,
        showarrow=False,
        font=dict(color=COLORS["neutral"]),
    )
fig.show()

# %% [markdown]
# ---
#
# ## Section 2: The characteristic taxonomy
#
# The 46 columns are not 46 independent ideas. They fall into eight groups, each measuring one
# economic quantity in several ways: how cheap a firm is relative to its accounting value, how
# profitable it is, how fast it is growing its asset base, how its price has moved recently,
# how volatile it is, how easily it trades, how much debt it carries, and how efficiently it
# turns assets into sales. Grouping them is what makes the correlation structure in Section 3
# and the information coefficients in Section 4 readable.

# %%
CHARACTERISTIC_CATEGORIES = {
    "Valuation": ["BEME", "E2P", "S2P", "CF2P", "D2P", "A2ME", "Q"],
    "Profitability": ["ROA", "ROE", "PROF", "OP", "PM", "PCM", "NI", "RNA"],
    "Investment": ["Investment", "NOA", "OA", "AC", "AT", "D2A"],
    "Momentum": ["r2_1", "r12_2", "r12_7", "r36_13", "ST_REV", "LT_Rev", "Rel2High"],
    "Risk": ["Beta", "MktBeta", "IdioVol", "Variance", "Resid_Var"],
    "Liquidity/Size": ["LME", "LTurnover", "Spread", "SUV"],
    "Leverage": ["Lev", "OL", "FC2Y", "C", "CF", "DPI2A"],
    "Other": ["ATO", "CTO", "SGA2S"],
}

CATEGORY_OF = {
    feature: category
    for category, features in CHARACTERISTIC_CATEGORIES.items()
    for feature in features
}

assert set(CATEGORY_OF) == set(feature_cols), "every characteristic belongs to exactly one group"

for category, features in CHARACTERISTIC_CATEGORIES.items():
    print(f"{category} ({len(features)}): {', '.join(features)}")

# %% [markdown]
# ### Checking that the ranks were applied
#
# The normalization described in the preamble makes a claim the data can be held to: every
# characteristic, in every month, occupies the same bounded range. The check is the smallest
# and largest value any of the 46 columns reaches anywhere in the panel. Reading it also
# confirms that nothing is missing, which matters because the authors admitted only stocks
# with a complete set of characteristics.

# %%
lowest = min(df.select(feature_cols).min().row(0))
highest = max(df.select(feature_cols).max().row(0))
print(f"Lowest value reached by any characteristic: {lowest:.3f}")
print(f"Highest value reached by any characteristic: {highest:.3f}")
print(f"Characteristics checked: {len(feature_cols)}")
print(f"Missing values among them: {df.select(feature_cols).null_count().to_numpy().sum()}")

# %% [markdown]
# Staying inside the interval is not the same as filling it evenly. A month in which many
# stocks tie on a characteristic would pile them at one value and leave gaps elsewhere, and the
# ranking would then be carrying less information than its range suggests. Every characteristic
# is ranked by the same procedure, so one of them settles the question for all: the histogram
# below counts every stock-month of book-to-market in the panel.

# %%
fig = px.histogram(
    df.select("BEME").to_pandas(),
    x="BEME",
    nbins=100,
    color_discrete_sequence=[COLORS["blue"]],
    title="Ranking spreads a characteristic evenly across the interval",
    labels={"BEME": "BEME (book-to-market, cross-sectionally ranked)", "count": "Stock-months"},
)
fig.show()

# %% [markdown]
# ---
#
# ## Section 3: Correlation structure
#
# Characteristics built from overlapping accounting inputs move together. Two measures of
# volatility computed over different windows, or two valuation ratios sharing a denominator,
# carry much of the same information, and a linear model fitted on both splits the coefficient
# between them arbitrarily. Measuring the redundancy first is what tells you whether the model
# you reach for later needs to handle it.
#
# Because each month is normalized on its own, pooling every stock-month into one correlation
# measures how the characteristics co-vary within a cross-section, which is the relevant
# question here, rather than how their levels drift over the decades.

# %%
corr_matrix = np.corrcoef(df.select(feature_cols).to_numpy(), rowvar=False)

fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix,
        x=feature_cols,
        y=feature_cols,
        colorscale=[
            [0.0, ml4t_diverging()[0]],
            [0.5, ml4t_diverging()[1]],
            [1.0, ml4t_diverging()[2]],
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
    )
)
fig.update_layout(
    title="Characteristics form correlated blocks, so the 46 are far from independent",
    width=1000,
    height=900,
    xaxis_tickangle=-45,
)
fig.show()

# %% [markdown]
# The blocks in the heatmap are easier to act on as a list. The pairs below are the ones whose
# correlation reaches the redundancy threshold declared in the parameters cell, ordered from
# most positive to most negative.

# %%
upper = np.triu_indices_from(corr_matrix, k=1)
redundant_pairs = (
    pl.DataFrame(
        {
            "feature_1": [feature_cols[i] for i in upper[0]],
            "feature_2": [feature_cols[j] for j in upper[1]],
            "correlation": corr_matrix[upper],
        }
    )
    .filter(pl.col("correlation").abs() >= REDUNDANCY_THRESHOLD)
    .sort("correlation", descending=True)
)
print(f"Pairs at or above |{REDUNDANCY_THRESHOLD}|: {len(redundant_pairs)}")
redundant_pairs

# %% [markdown]
# ---
#
# ## Section 4: How each characteristic relates to next-month returns
#
# The information coefficient answers one question per month: if you had ordered every stock in
# the cross-section by this characteristic, how well would that ordering have matched the order
# of the returns that followed? Rank correlation is the right measure because only the ordering
# is meaningful once the characteristic has been ranked, and because a single extreme return
# then moves the statistic by one rank rather than by its full magnitude. Averaging the monthly
# values gives the characteristic's average ordering power, which is the Fama-MacBeth template.
#
# **Everything in this section is computed on the training split alone.** Ranking
# characteristics by how well they predicted returns is a selection decision, and a selection
# decision taken over the validation and test years leaks those years into every model built on
# the ranking, including the ones fitted in Chapters 11 and 12. The panel ships with the split
# labels so that the line can be drawn here.

# %%
train = df.filter(pl.col("split") == "train")
train_returns = train.select("symbol", "timestamp", pl.col("ret").alias("forward_return"))

print(f"Training months: {train['timestamp'].n_unique()}")
print(f"Training rows: {len(train):,}")

# %% [markdown]
# ### The two significance tests, and why they differ
#
# A t-statistic on the average of the monthly ICs divides that average by the standard error of
# the mean, and computing the standard error that way assumes each month is an independent draw.
# The months are not independent: slow-moving common factors keep a characteristic in or out of
# favour for stretches at a time, so a run of positive ICs is followed by more positive ICs more
# often than chance allows. Newey and West's estimator widens the standard error by the amount
# of that serial correlation, using a lag window chosen from the length of the series.
#
# `compute_ic_hac_stats` reports both: the corrected t-statistic and the uncorrected one, so the
# gap between them is visible rather than asserted.

# %%
ic_rows = []
for feature in feature_cols:
    ic_series = cross_sectional_ic_series(
        train.select("symbol", "timestamp", pl.col(feature).alias("prediction")),
        train_returns,
        date_col="timestamp",
        entity_col="symbol",
        method="spearman",
    )
    stats = compute_ic_hac_stats(ic_series, ic_col="ic", label_horizon=LABEL_HORIZON_MONTHS)
    ic_rows.append(
        {
            "feature": feature,
            "category": CATEGORY_OF[feature],
            "IC": stats["mean_ic"],
            "t_stat_HAC": stats["t_stat"],
            "t_stat_naive": stats["naive_t_stat"],
            "hac_lags": stats["effective_lags"],
        }
    )

ic_df = pl.DataFrame(ic_rows).sort("IC", descending=True)
ic_df

# %% [markdown]
# Plotted in the order above, the ICs show two things at once: how small the largest of them is,
# and that the characteristics at the two ends of the range are the ones a reader of the factor
# literature would expect there.

# %%
fig = px.bar(
    ic_df.with_columns(
        pl.when(pl.col("IC") > 0)
        .then(pl.lit("Ranks with the return"))
        .otherwise(pl.lit("Ranks against the return"))
        .alias("Direction")
    ).to_pandas(),
    x="feature",
    y="IC",
    color="Direction",
    color_discrete_map={
        "Ranks with the return": COLORS["blue"],
        "Ranks against the return": COLORS["copper"],
    },
    category_orders={"Direction": ["Ranks with the return", "Ranks against the return"]},
    title="No single characteristic carries a large information coefficient",
    labels={
        "IC": "Mean monthly rank correlation with next-month return",
        "feature": "Characteristic",
    },
)
fig.update_layout(xaxis_tickangle=-45, height=520)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
fig.show()

# %% [markdown]
# The correction for serial correlation is worth seeing directly. Each point below is one
# characteristic, placed by its uncorrected t-statistic and its corrected one. Points on the
# diagonal are unaffected by the correction; points pulled toward the horizontal centre line
# have had their significance reduced by it. The dotted rules sit at the conventional
# two-standard-error threshold, so a point between them fails that test on the corrected
# statistic.

# %%
t_limit = float(np.ceil(max(ic_df["t_stat_naive"].abs().max(), ic_df["t_stat_HAC"].abs().max())))
fig = px.scatter(
    ic_df.to_pandas(),
    x="t_stat_naive",
    y="t_stat_HAC",
    hover_name="feature",
    color_discrete_sequence=[COLORS["blue"]],
    title="Correcting for month-to-month persistence moves t-statistics both ways",
    labels={
        "t_stat_naive": "t-statistic assuming independent months",
        "t_stat_HAC": "t-statistic with Newey-West correction",
    },
)
fig.add_shape(
    type="line",
    x0=-t_limit,
    y0=-t_limit,
    x1=t_limit,
    y1=t_limit,
    line=dict(color=COLORS["neutral"], dash="dash"),
)
for threshold in (-2, 2):
    fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS["neutral"])
fig.update_layout(height=520)
fig.show()

# %% [markdown]
# ---
#
# ## Section 5: The return distribution
#
# The characteristics were ranked; `ret` was not. It stays in its original scale because a
# prediction of it has to be readable as a return, which is what makes the loss function of a
# model fitted on this panel comparable to an economic quantity.
#
# Monthly equity returns have far heavier tails than a normal distribution, and the tails grow
# heavier in the later decades of this sample. The table gives the moments; the figure gives
# the shape.

# %%
return_stats = df.group_by("split").agg(
    pl.col("ret").mean().alias("mean"),
    pl.col("ret").std().alias("std"),
    pl.col("ret").min().alias("min"),
    pl.col("ret").max().alias("max"),
    pl.col("ret").quantile(0.25).alias("q25"),
    pl.col("ret").quantile(0.75).alias("q75"),
)
return_stats.sort("split")

# %% [markdown]
# The histogram below is drawn as a density rather than a count, because the three splits hold
# very different numbers of rows and raw counts would compare their sizes instead of their
# shapes. The horizontal axis is clipped at plus and minus sixty percent: the largest returns in
# the panel run to many multiples of that, and leaving them in the frame compresses every bar
# into a single spike. The `max` column in the table above is where the full range is reported.

# %%
fig = px.histogram(
    df.select("ret", "split").to_pandas(),
    x="ret",
    color="split",
    histnorm="probability density",
    nbins=120,
    range_x=[-0.6, 0.6],
    category_orders={"split": ["train", "valid", "test"]},
    color_discrete_sequence=[COLORS["blue"], COLORS["amber"], COLORS["copper"]],
    title="Next-month returns are more dispersed in the later splits",
    labels={"ret": "Next-month excess return (clipped at +/-60%)", "split": "Split"},
    opacity=0.6,
    barmode="overlay",
)
fig.show()

# %% [markdown]
# ---
#
# ## Section 6: What this panel supports
#
# The anonymization is the constraint that decides which questions this data can answer. Inside
# a split, an identifier follows one firm from month to month, so anything that needs a firm's
# own history - a fixed effect, a per-firm time series, a turnover calculation over a portfolio
# held across months - is available. Across a split boundary the numbering restarts, so a
# position cannot be carried from the training years into the test years, and a strategy cannot
# be simulated end to end over the full fifty years.
#
# What is missing entirely is price. Without a price level there is no market capitalization to
# weight a portfolio by, no spread to charge against a trade, and no way to convert a predicted
# return into a position size in currency. That is why every question this panel answers is a
# question about prediction accuracy, and why the trading questions in this book are answered
# with the ETF, crypto and futures case studies, which ship prices.
#
# Used within those limits, the panel is the cleanest available benchmark for comparing model
# families on identical data: the same 46 inputs, the same target, and the same date-ordered
# split for every model, so a difference in measured accuracy is a difference between the
# models rather than between their datasets.

# %% [markdown]
# ## Key Takeaways
#
# 1. Cross-sectional rank normalization puts every characteristic on the same bounded scale each
#    month, which removes the units and caps the influence of any single extreme observation.
#    It also discards the levels, so nothing computed from this panel can speak to how expensive
#    the market was in a given year.
# 2. Characteristics arrive in correlated blocks, because several of them are built from the
#    same accounting inputs. Fitting an unregularized linear model on all 46 splits coefficients
#    arbitrarily within a block; regularization and tree ensembles are the standard answers, and
#    Chapters 11 and 12 apply both to this panel.
# 3. Single-characteristic information coefficients are small in absolute terms. That is the
#    normal state of a cross-sectional equity signal and it is why the case study fits models on
#    all 46 jointly rather than trading any one of them.
# 4. Averaging monthly ICs and testing the average as if the months were independent overstates
#    significance, because a characteristic stays in favour for stretches. Report the
#    Newey-West-corrected statistic, and check how far it moved before drawing a conclusion from
#    either.
# 5. Evaluating characteristics on the training split alone is not a formality. Any ranking
#    computed over the whole history has already read the test years, and a model built on that
#    ranking inherits the leak.
#
# **Known limitations**: the panel carries no prices and no company names; identifiers are not
# comparable across the three splits; and the requirement that all 46 characteristics be present
# tilts the universe away from small caps, so coverage is not a random sample of CRSP.
#
# **Next**: `case_studies/us_firm_characteristics/` fits linear, tree and neural models to this
# panel.
