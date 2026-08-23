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
# # Crypto perpetuals: what the search actually found
#
# Four notebooks narrowed a field. [`13_backtest`](13_backtest.ipynb) ran every prediction set
# equally weighted, [`14_portfolio_management`](14_portfolio_management.ipynb) sized the survivors
# six ways, [`15_costs`](15_costs.ipynb) asked how much friction the result absorbs, and
# [`16_risk_management`](16_risk_management.ipynb) tried fourteen ways of leaving a position early.
# This notebook makes the one choice the case study exists to make, and then says how much
# confidence that choice supports.
#
# **The second half is the part that matters.** A funnel that ranks several hundred backtests on
# one validation period will always return a highest number. Whether that number is evidence
# depends on how many candidates it was the highest of, how wide its own confidence interval is,
# and how much of it is left after correcting for having been selected. Reporting the selected
# row without those three is the failure this notebook is built to avoid.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Select one configuration across labels, and say why the label is part of the configuration
#   rather than a separate axis.
# - Read a Sharpe ratio as an interval rather than a point, and say what the interval covers.
# - Apply a selection-bias correction and say what the corrected number means when the raw one is
#   already negative.
# - Separate what a perpetual-futures strategy earns from price movement and from funding.
#
# **Book reference**: Chapter 20 (Strategy Synthesis).
#
# **Prerequisites**: [`16_risk_management`](16_risk_management.ipynb) has frozen a candidate set
# per label spanning all three selection stages.
#
# **What it writes**: one candidate set holding the whole selection pool. No backtests, no
# training, no holdout evaluation.

# %%
"""Select and assess one crypto perpetuals configuration from the frozen validation pool."""

import plotly.graph_objects as go
import polars as pl

from case_studies.crypto_perps_funding.research_workflow import ALL_LABELS
from case_studies.research import CandidateSet, Result, open_study
from case_studies.utils.uncertainty import (
    compute_cohort_metrics,
    load_daily_returns_with_timestamp,
    periods_per_year_from_setup,
)
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
LABELS: list[str] = []
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
SUPERSEDES: str = ""

# %%
study = open_study(
    "crypto_perps_funding", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)
labels = list(LABELS) if LABELS else list(ALL_LABELS)
periods_per_year = periods_per_year_from_setup("crypto_perps_funding")

# %% [markdown]
# ## 1. One pool, four labels
#
# Each label has its own frozen set spanning the baseline, allocation and overlay stages. The
# object of selection is **one configuration for the case study**, not one per label, so the four
# sets are compared as a single pool and the label comes from the row that is selected.
#
# The four labels are not interchangeable inputs to one experiment. `fwd_ret_8h` and
# `fwd_ret_24h` are continuous returns over different horizons; `fwd_dir_8h` and `fwd_dir_8h_3c`
# are the 8-hour move coded into two and three classes. They carry different label artifacts and
# a different purge interval, so a candidate set spanning them has to declare the return horizon
# as the axis it spans - which is what `comparable_fields` does below. Everything else about the
# protocol must still match, and the set refuses a member whose split or execution tier disagrees.

# %%
pool_members = []
for label in labels:
    candidates = CandidateSet.one(study, name=f"crypto-final-validation-{label}")
    pool_members.extend(Result.open(study, member) for member in candidates.members)
pool = CandidateSet.create(
    study,
    "crypto-final-selection",
    pool_members,
    comparison_contract={"comparable_fields": ["label_artifact", "cv"]},
    supersedes=SUPERSEDES or None,
)
print(f"{len(pool.members)} candidates across {len(labels)} labels")

# %% [markdown]
# ## 2. The selection
#
# Highest validation Sharpe in the pool, which is the rule the whole funnel has applied at every
# stage. It is applied here across labels for the first time.

# %%
selected = pool.best_validation_sharpe()
catalog = study.backtests.table().filter(pl.col("backtest_hash").is_in(pool.members))
if catalog.height != len(pool.members):
    raise RuntimeError("the backtest catalog does not describe every candidate")

# %% [markdown]
# The selected configuration, and the pool it was selected from. The stage column says how far
# down the funnel the selection came from: a baseline row means neither sizing nor an overlay
# improved on equal weight for that ranking.

# %% tags=["results"]
keyed = catalog.with_columns(
    pl.col("allocation_method").fill_null("equal_weight").alias("allocator"),
    pl.col("risk_method").fill_null("none").alias("overlay"),
)
keyed.filter(pl.col("backtest_hash") == selected.hash).select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "stage",
    "signal_method",
    "allocator",
    "overlay",
    "sharpe",
    "sharpe_ci95_lo",
    "sharpe_ci95_hi",
    "max_drawdown",
    "total_return",
    "num_trades",
)

# %% [markdown]
# What it was selected from, by stage. The spread within each stage is the more useful number
# than any single row: it says how much of the difference between candidates is available to be
# picked up by chance.

# %% tags=["results"]
keyed.group_by("stage").agg(
    candidates=pl.len(),
    labels=pl.col("label").n_unique(),
    best_sharpe=pl.col("sharpe").max(),
    median_sharpe=pl.col("sharpe").median(),
    worst_sharpe=pl.col("sharpe").min(),
    above_zero=(pl.col("sharpe") > 0).sum(),
).sort("stage")

# %% [markdown]
# ## 3. How much confidence the selection supports
#
# Three separate questions, and they have different answers.
#
# **How precise is this one number?** The registry stores a ninety-five percent interval for
# every backtest Sharpe, from a stationary block bootstrap on the daily return series with the
# block length set from the label's rebalance step. That interval describes sampling variation
# in one series and says nothing about the search.
#
# **How much of it is selection?** The selected Sharpe is the maximum over the whole pool, and the
# maximum of many draws is above the truth even when every draw is worthless. The **deflated
# Sharpe ratio** subtracts what the maximum would have been under a null of no skill. Three
# variants are computed: a raw one that treats every candidate as an independent trial, and two
# that discount correlated candidates - which matters here, because a hundred backtests on twenty
# prediction sets are anything but independent.
#
# **How long would it take to know?** `min_trl` is the number of periods the selected strategy
# would need to run for its Sharpe to reach significance at the conventional five percent level.
# Where it exceeds the data available, the honest statement is that the validation period cannot
# settle the question.
#
# The alignment below intersects the candidates on timestamp. That is only safe because both
# earlier stages admitted to their candidate sets on **folds traded** rather than on periods
# observed - a result that sat out a fold would align perfectly here and be scored on a different
# period, which is exactly the comparison the funnel is meant to prevent.

# %%
returns_by_hash = {}
for member in pool.members:
    frame = load_daily_returns_with_timestamp("crypto_perps_funding", member)
    if frame is not None:
        returns_by_hash[member] = frame
cohort = compute_cohort_metrics(returns_by_hash, periods_per_year=periods_per_year)
if not cohort:
    raise RuntimeError("cohort alignment failed across the selection pool")
if cohort["leader_hash"] != selected.hash:
    raise RuntimeError(
        "the cohort leader on common support is not the candidate set's selection; "
        "the pool is not aligned on a single period"
    )

# %% [markdown]
# The selected configuration read three ways. `sharpe` is what it scored; `ras_sharpe` is a lower
# bound that holds across the whole search; the three `dsr_*` rows are the selection-corrected
# Sharpe under different assumptions about how independent the candidates were.

# %% tags=["results"]
pl.DataFrame(
    [
        {
            "metric": name,
            "value": cohort.get(key),
        }
        for name, key in [
            ("candidates (K)", "k_variants"),
            ("selected Sharpe", "leader_sharpe"),
            ("expected max Sharpe under the null", "expected_max_sharpe_raw"),
            ("deflated Sharpe, raw K", "dsr_raw"),
            ("deflated Sharpe, Marchenko-Pastur K", "dsr_mp"),
            ("deflated Sharpe, effective-rank K", "dsr_er"),
            ("Rademacher-adjusted Sharpe", "ras_leader"),
            ("periods needed for significance, effective-rank K", "min_trl_periods_er"),
        ]
        if key in cohort
    ]
)

# %% [markdown]
# ## 4. What the selected strategy did
#
# The cumulative return of the selected configuration across the validation folds, against the
# zero line. Funding is settled inside the engine at each 8-hourly timestamp on the position held
# before that timestamp's fills, so the curve already includes it.

# %%
returns = returns_by_hash[selected.hash].sort("timestamp")
column = next(name for name in returns.columns if name != "timestamp")
equity = returns.with_columns(cumulative=(1.0 + pl.col(column)).cum_prod() - 1.0)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=equity.get_column("timestamp").to_list(),
        y=equity.get_column("cumulative").to_list(),
        mode="lines",
        name="selected configuration",
        line={"color": COLORS["blue"]},
    )
)
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
fig.update_layout(
    title={
        "text": "Cumulative validation return of the selected configuration"
        "<br><sup>Net of the declared cost schedule and of settled funding</sup>",
        "x": 0.02,
        "xanchor": "left",
    },
    xaxis_title="Date",
    yaxis_title="Cumulative return",
    height=480,
    width=1000,
)
show_plotly_with_alt(
    fig,
    "Line chart of the cumulative validation return of the selected configuration over the two "
    "validation folds, with a dashed horizontal line at zero, net of declared costs and settled "
    "funding.",
)

# %% [markdown]
# ### Where the money came from
#
# A perpetual-futures strategy has two sources of return that behave differently, and a total
# return hides which one it used. Price P&L is the contract moving; funding is the periodic
# payment between longs and shorts, settled every eight hours on whatever position is held. A
# strategy whose return is mostly funding is a carry strategy whether or not it was built as one,
# and it will behave completely differently when the funding rate changes sign.

# %% tags=["results"]
record = keyed.filter(pl.col("backtest_hash") == selected.hash)
funding = record.select(
    pl.col("metrics_json").str.json_path_match("$.funding_pnl").cast(pl.Float64).alias("funding"),
    pl.col("metrics_json")
    .str.json_path_match("$.funding_settlements")
    .cast(pl.Float64)
    .alias("settlements"),
    pl.col("total_commission"),
    pl.col("total_slippage"),
    pl.col("total_return"),
)
funding

# %% [markdown]
# ## 5. The holdout is untouched
#
# `config/setup.yaml` reserves 2024 and 2025 as a holdout, and nothing in this case study has
# been fitted, selected or measured on it. The registry holds no holdout prediction set, so there
# is no holdout number to report and none is implied by anything above.
#
# It takes three steps, in this order and only once: refit the selected configuration on
# training plus validation, predict the holdout window, and run that one configuration through
# the same backtest specification. What makes it a holdout is that the selection is
# already fixed - the pool above is frozen and immutable, so the configuration cannot be
# re-chosen after the holdout result is seen. Every step of the funnel exists to make that
# sentence true.

# %%
holdout_predictions = study.predictions.table().filter(pl.col("split") == "holdout")
print(
    f"{holdout_predictions.height} holdout prediction sets in the registry; "
    f"selection pool {pool.hash} is frozen at {len(pool.members)} members"
)

# %% [markdown]
# ## 6. What to notice
#
# **The funnel narrows the search, it does not remove it.** Every stage ran fewer configurations
# than the one before, which is why the pool is in the hundreds rather than the thousands. It is
# still a search, and the correction in section 3 is the price of having run it. A case study
# that reported only the highest Sharpe would be reporting the largest of several hundred draws
# as though it were one measurement.
#
# **A negative selected Sharpe is a result, not a failure of the pipeline.** If the highest
# Sharpe in the pool is below zero, the deflated version is further below and the interval is not
# the interesting part - what the case study has established is that this signal, on these
# contracts, over these folds, does not support a profitable strategy once costs and funding are
# charged. That is a finding about the market and the model, and it is the kind of finding a
# selection procedure that could not return it would be worthless for.
#
# **The interval and the correction answer different questions.** The bootstrap interval widens
# with a shorter series; the deflation grows with more candidates. A long backtest of one strategy
# has a tight interval and no deflation. A short backtest of five hundred has both problems, and
# two validation folds of 8-hourly crypto data is closer to the second.
#
# **Funding is a separate return stream, not a cost line.** It is reported beside commission and
# slippage above because all three reduce or increase the same total, but it is the only one that
# depends on holding rather than trading, and the only one that can be positive.
#
# **Known limitations.** Everything above is measured on two validation folds spanning 2022 and
# 2023, on nineteen perpetual contracts, at one declared cost schedule. The pool contains only
# configurations that traded both folds, so nothing here says how a strategy that trades
# selectively would compare - by construction it could not have been ranked against these. And
# the holdout has not been used, so no statement here has been checked out of sample.
