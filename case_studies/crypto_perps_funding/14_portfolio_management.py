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
# # Crypto perpetuals: six ways to size a position the ranking already chose
#
# [`13_backtest`](13_backtest.ipynb) gave every position the same weight. That was the point
# there - equal weight adds no information, so a difference between two baselines is a difference
# between two rankings. This notebook keeps the rankings and the entry rules exactly where they
# were and changes one thing: how much capital each admitted position gets.
#
# Six alternatives are declared in `config/setup.yaml`, and they read three different kinds of
# input. One reads the model's own score, so a contract the model is more confident about gets
# more capital. Four read the history of returns - each contract's own volatility, or the
# covariance between contracts - so that the positions contribute comparable amounts of risk
# rather than comparable amounts of money. One reads how uncertain the model's prediction has
# been on that contract in the past.
#
# **This stage is narrow on purpose.** It runs on the survivors of the baseline rather than on
# everything, because a sizing method applied to a ranking that lost money equally-weighted is
# not a question anyone needs answered, and because every configuration added to a search makes
# the highest result in it easier to reach by luck. The funnel that decides which baselines advance
# is described below and is not a choice this notebook makes.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Take the top model configurations from a completed baseline stage, counting distinct
#   configurations rather than distinct prediction sets, and say why the two differ.
# - Say what each declared allocator reads, and which of them need a history of returns before
#   they can weight anything.
# - Run a sizing variant so that it differs from its own baseline in one field, and read the
#   paired difference rather than comparing two leaders.
# - Recognise when an allocator has traded a different set of dates from the one it is being
#   compared against, and why that makes the comparison an unpaired one.
#
# **Book reference**: Chapter 17 (Portfolio Construction).
#
# **Prerequisites**: [`13_backtest`](13_backtest.ipynb) has registered a complete
# `stage='signal'` baseline for every declared prediction set.
#
# **What it writes**: one `stage='allocation'` backtest per surviving prediction set, entry rule
# and allocator, and one candidate set per label holding the baseline and the allocation results
# together. [`15_costs`](15_costs.ipynb) and [`16_risk_management`](16_risk_management.ipynb)
# read those.

# %%
"""Run the declared allocator grid on the surviving crypto perpetuals baselines."""

import plotly.graph_objects as go
import polars as pl

from case_studies.crypto_perps_funding.research_workflow import ALL_LABELS
from case_studies.research import Result, open_study, run_backtests
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.registry.queries import resolve_best_predictions
from case_studies.utils.sweep_config import (
    get_allocator_lookback,
    get_allocators,
    get_entry_schemes_for,
    get_top_n_predictions,
)
from utils.artifact_specs import load_setup_config
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
LABELS: list[str] = []
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
POPULATION_SUFFIX = "v1"

# %%
study = open_study(
    "crypto_perps_funding", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)
setup = load_setup_config("crypto_perps_funding")
labels = list(LABELS) if LABELS else list(ALL_LABELS)
n_assets = int(setup["universe"]["n_assets"])

# %% [markdown]
# ## 1. Which baselines advance
#
# The selection funnel is sequential: each stage runs on the survivors of the one before it, and
# the survivors are chosen on **validation backtest Sharpe** - never on information coefficient,
# which measures whether a model ranks contracts correctly and says nothing about what a strategy
# trading that ranking earns after costs and funding.
#
# `config/setup.yaml` sets the width of each stage under `backtest.sweep.top_n_predictions`. The
# baseline ran on everything; allocation runs on the top ten. **Ten what** is the part worth being
# precise about: ten distinct `(family, config_name)` pairs, not ten prediction sets. A boosted
# model contributes one prediction set per checkpoint, so counting prediction sets would let a
# single configuration occupy the whole shortlist with ten readings of itself and crowd out every
# other model in the case study. `checkpoints_per_config` then says how many checkpoints each
# advancing configuration brings with it, and it is one - the checkpoint that scored best at the
# baseline, since the checkpoint is part of the configuration rather than a knob to be re-tuned
# here.
#
# `resolve_best_predictions` is the one implementation of that rule, shared by every case study.

# %%
top_n = get_top_n_predictions("crypto_perps_funding", "allocation")
survivors = pl.concat(
    [
        resolve_best_predictions(
            "crypto_perps_funding",
            label,
            split="validation",
            stage="signal",
            top_n=top_n,
            checkpoints_per_config=1,
            case_dir=study.root,
        )
        for label in labels
    ]
)
if survivors.is_empty():
    raise RuntimeError("no baseline survivors: run 13_backtest before this notebook")

# %% [markdown]
# One row per advancing configuration. `sharpe` is the baseline it advanced on, and it is shown
# so the shortlist can be read against the population it came from rather than in isolation.

# %% tags=["results"]
catalog = study.predictions.table().filter(
    pl.col("prediction_hash").is_in(survivors.get_column("prediction_hash"))
)
if catalog.height != survivors.height:
    raise RuntimeError("a surviving prediction is absent from the prediction catalog")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("a surviving prediction set is incomplete")

survivors.select("label", "family", "config_name", "checkpoint_value", "sharpe").sort(
    "label", "sharpe", descending=[False, True]
)

# %% [markdown]
# ## 2. What each allocator reads
#
# All six take the same set of admitted positions from the entry rule and decide only the size of
# each. They differ in what they consult to do it.
#
# - **`score_weighted`** is the only one that reads the prediction. Weight is proportional to the
#   model's score, so the ordering the ranking produced becomes a spread of position sizes rather
#   than a set of equal ones.
# - **`inverse_vol`** weights each contract by the reciprocal of its own recent return
#   volatility. A contract that moves twice as much gets half the capital, so each position
#   contributes a comparable amount of risk. It looks at each contract on its own and ignores how
#   they move together.
# - **`risk_parity`** equalizes each position's contribution to the volatility of the whole
#   portfolio, which requires the covariance between contracts rather than just the diagonal. On
#   a set of perpetual futures that mostly rise and fall together, the difference from
#   `inverse_vol` is the part correlation accounts for.
# - **`hrp`**, hierarchical risk parity, groups the contracts by how correlated they are, splits
#   capital between the groups, and then splits it again inside each. It never inverts the
#   covariance matrix, which is what makes it usable when the estimate is noisy.
# - **`mvo_ledoit_wolf`** does invert one: mean-variance optimization, on a covariance pulled part
#   of the way towards a simpler structured estimate. That pull is the **shrinkage**, and it is
#   there because a nineteen-by-nineteen covariance estimated from a few hundred observations
#   contains enough noise that optimizing against it directly concentrates the book on whichever
#   pair happens to look least correlated in the sample.
# - **`conformal_weighted`** reads how wrong the model has been. For each contract it builds an
#   interval around the prediction, wide where the model's past errors on that contract were
#   large, and gives less capital to the wider intervals.
#
# The four that read return history share one window, so that a difference between them is the
# method rather than the amount of history each was given.

# %%
allocators = get_allocators("crypto_perps_funding")
lookback = get_allocator_lookback("crypto_perps_funding")
bar_hours = int(setup["features"]["bar_hours"])
print(
    f"Rolling window for the moment-based allocators: {lookback} bars of "
    f"{bar_hours}-hourly prices, about {lookback * bar_hours / 24:.0f} days of history before "
    "the first decision each can weight."
)
pl.DataFrame(
    [
        {
            "allocator": allocator["method"],
            "reads": "prediction score"
            if allocator["method"] == "score_weighted"
            else "prediction error history"
            if allocator["method"] == "conformal_weighted"
            else "return history",
            "warmup_bars": strategy_warmup_periods({"allocation": allocator}),
        }
        for allocator in allocators
    ]
)

# %% [markdown]
# ### An allocator that calibrates cannot trade its calibration window
#
# `conformal_weighted` builds its intervals from the model's residuals on **earlier validation
# folds only**, which is what keeps them out of sample. The earliest fold has no earlier fold, so
# no contract has a width there, and the strategy holds nothing at all until the fold after it
# begins. Every other allocator trades from the first decision.
#
# That is not a detail about one allocator. It means two results can cover the same dates and
# still not be comparable, because one of them was in cash for part of the span. A strategy that
# sits out a losing period and trades only the rest is being measured on a different sample from
# one that traded both, and differencing the two attributes the sample to the sizing.
#
# **The period count does not detect this.** A day on which a strategy holds nothing still
# contributes a return observation, of zero, so a book that is flat for a year reports the same
# number of periods as one that traded every day of it. Section 4 measures which folds each
# result actually traded and pairs only within a matching set.
#
# ## 3. Running the grid
#
# The entry rules are the ones the baseline established as feasible on a nineteen-contract
# universe, unchanged, because changing the sizing and the selection together would measure
# neither. For every surviving prediction set, every feasible entry rule and every allocator,
# `run_backtests` resolves a strategy that differs from its baseline in the `allocation` field
# and in nothing else.
#
# Prices are loaded once per label and warmup. The moment-based allocators need the rolling
# window of prices in front of their first decision and the other two do not, so there are two
# price frames per label rather than one - and they are the frames the boundary would have
# loaded for itself, so passing them in changes nothing but the number of reads.

# %%
warmups = sorted({strategy_warmup_periods({"allocation": item}) for item in allocators})
prices_by_key = {
    (label, warmup): load_backtest_prices_for(
        "crypto_perps_funding", label, split="validation", warmup_periods=warmup
    )
    for label in labels
    for warmup in warmups
}
schemes_by_label = {
    label: get_entry_schemes_for("crypto_perps_funding", label, n_assets=n_assets, long_short=True)
    for label in labels
}

# %%
for label in labels:
    label_rows = catalog.filter(pl.col("label") == label)
    for scheme in schemes_by_label[label]:
        signal = {key: value for key, value in scheme.items() if key != "name"}
        for allocator in allocators:
            warmup = strategy_warmup_periods({"allocation": allocator})
            execution = run_backtests(
                study,
                predictions=label_rows,
                signal=signal,
                allocation=allocator,
                prices=prices_by_key[(label, warmup)],
                chapter="ch17",
                population_name=(
                    f"crypto-allocation-{label}-{scheme['name']}-"
                    f"{allocator['method']}-{POPULATION_SUFFIX}"
                ),
            )
            print(
                f"{label} / {scheme['name']} / {allocator['method']}: "
                f"{len(execution.results)} backtests registered"
            )

# %% [markdown]
# ## 4. What came out
#
# Read back from the registry, one row per allocator. `traded_folds` lists the validation folds
# each result actually held a position in, derived below from the registered return series rather
# than assumed from the allocator's name. A fold's number identifies it and does not order it -
# the walk-forward split numbers folds as it builds them - so the list is written oldest first.

# %%
results = study.backtests.table().filter(
    (pl.col("split") == "validation")
    & pl.col("label").is_in(labels)
    & pl.col("stage").is_in(["signal", "allocation"])
)
if results.filter(~pl.col("complete")).height:
    raise RuntimeError("the backtest catalog contains incomplete members")

entry_rule = (
    pl.col("signal_method")
    + pl.when(pl.col("spec_json").str.json_path_match("$.strategy.signal.top_k").is_not_null())
    .then(pl.lit("_top") + pl.col("spec_json").str.json_path_match("$.strategy.signal.top_k"))
    .otherwise(pl.lit(""))
).alias("entry_rule")
keyed = results.with_columns(
    entry_rule,
    pl.col("allocation_method").fill_null("equal_weight").alias("allocator"),
)

# %% [markdown]
# Each decision carries its fold in the prediction set, and the dates a result held a
# position on are in the return series it registered - a day the strategy was flat contributes a
# return of exactly zero. Intersecting the two says which folds each result traded, which is the
# property that has to match before two results can be differenced. The windows are put in date
# order here because the fold numbers are not in date order.


# %%
def fold_windows(label: str) -> pl.DataFrame:
    """First and last decision date of each validation fold, for one label.

    Every configuration for a label predicts the same keys, which 13_backtest established, so
    one prediction set carries the fold calendar for all of them.
    """
    reference = catalog.filter(pl.col("label") == label).sort("prediction_hash")
    return (
        Result.open(study, reference.item(0, "prediction_hash"))
        .load()
        .group_by("fold")
        .agg(
            fold_start=pl.col("timestamp").min().dt.date(),
            fold_end=pl.col("timestamp").max().dt.date(),
        )
        .sort("fold_start")
    )


# %%
def traded_folds(backtest_hash: str, windows: pl.DataFrame) -> tuple[int, ...]:
    """Which validation folds one registered result actually held a position in."""
    returns = pl.read_parquet(
        study.root / "run_log" / "backtest" / backtest_hash / "daily_returns.parquet"
    )
    column = next(name for name in returns.columns if name != "timestamp")
    active = returns.filter(pl.col(column) != 0).select(pl.col("timestamp").dt.date().alias("day"))
    if active.is_empty():
        return ()
    return tuple(
        int(row["fold"])
        for row in windows.iter_rows(named=True)
        if active.filter(pl.col("day").is_between(row["fold_start"], row["fold_end"])).height
    )


# %%
windows_by_label = {label: fold_windows(label) for label in labels}
keyed = keyed.with_columns(
    pl.Series(
        "traded_folds",
        [
            "+".join(
                str(fold)
                for fold in traded_folds(row["backtest_hash"], windows_by_label[row["label"]])
            )
            for row in keyed.iter_rows(named=True)
        ],
    )
)

# %% tags=["results"]
allocation_grid = (
    keyed.filter(pl.col("stage") == "allocation")
    .group_by("allocator")
    .agg(
        backtests=pl.len(),
        labels=pl.col("label").n_unique(),
        median_sharpe=pl.col("sharpe").median(),
        above_zero=(pl.col("sharpe") > 0).sum(),
        traded_folds=pl.col("traded_folds").unique().sort().str.join(", "),
        median_turnover=pl.col("avg_turnover").median(),
    )
    .sort("allocator")
)
allocation_grid

# %% [markdown]
# ### The paired difference, one field at a time
#
# A leader-to-leader comparison across stages is not evidence that sizing helped: the allocation
# leader and the baseline leader can be different models on different checkpoints, and the gap
# between them then contains the search as well as the sizing. The join below pairs each
# allocation result with the baseline built on the **same prediction set and the same entry
# rule**, so the only field that differs is the allocator, and reports the difference.
#
# A pair is admitted only when both legs traded the same folds. That is what excludes the
# conformal rows, and it would exclude anything else that sat out part of the span for any other
# reason - the test is what the result did, not which allocator produced it.

# %% tags=["results"]
baseline = keyed.filter(pl.col("stage") == "signal").select(
    "prediction_hash",
    "entry_rule",
    pl.col("sharpe").alias("baseline_sharpe"),
    pl.col("traded_folds").alias("baseline_traded_folds"),
)
allocation = keyed.filter(pl.col("stage") == "allocation")
paired = (
    allocation.join(baseline, on=["prediction_hash", "entry_rule"], how="inner")
    .filter(pl.col("traded_folds") == pl.col("baseline_traded_folds"))
    .with_columns((pl.col("sharpe") - pl.col("baseline_sharpe")).alias("sharpe_change"))
)
unpaired = (
    allocation.join(baseline, on=["prediction_hash", "entry_rule"], how="inner")
    .filter(pl.col("traded_folds") != pl.col("baseline_traded_folds"))
    .group_by("allocator")
    .agg(left_unpaired=pl.len(), traded_folds=pl.col("traded_folds").unique().sort().str.join(", "))
    .sort("allocator")
)
print(f"{paired.height} paired comparisons")
print(unpaired)

paired.group_by("allocator").agg(
    pairs=pl.len(),
    median_change=pl.col("sharpe_change").median(),
    improved=(pl.col("sharpe_change") > 0).sum(),
    best_change=pl.col("sharpe_change").max(),
    worst_change=pl.col("sharpe_change").min(),
).sort("allocator")

# %% [markdown]
# One distribution per allocator, over every pair. The zero line is equal weight: a point above
# it is a configuration that sizing improved, and the fraction of each distribution above the
# line is the more informative reading than any single point in it.

# %%
order = sorted(set(paired.get_column("allocator")))
fig = go.Figure()
for allocator in order:
    panel = paired.filter(pl.col("allocator") == allocator)
    fig.add_trace(
        go.Box(
            y=panel.get_column("sharpe_change").to_list(),
            name=allocator,
            marker_color=COLORS["blue"],
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            marker={"size": 3, "opacity": 0.5},
            showlegend=False,
        )
    )
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
fig.update_layout(
    title={
        "text": "Change in validation Sharpe from the equal-weight baseline"
        "<br><sup>One point per prediction set and entry rule; the pair differs only in the "
        "allocator</sup>",
        "x": 0.02,
        "xanchor": "left",
    },
    xaxis_title="Allocator",
    yaxis_title="Sharpe minus its own equal-weight baseline",
    height=520,
    width=1000,
)
show_plotly_with_alt(
    fig,
    "Box plots with every pair overlaid as a point, one box per allocator, of the change in "
    "annualized validation Sharpe against the equal-weight baseline built on the same prediction "
    "set and entry rule. A dashed horizontal line marks zero, meaning no change from equal "
    "weight. Every allocator's distribution straddles that line, and the boxes overlap one "
    "another, so no allocator separates from equal weight or from the others.",
)

# %% [markdown]
# ## 5. The candidate set each label hands on
#
# The next two stages choose from the baseline and the allocation results **together**, which is
# what the funnel prescribes: the question at the risk stage is whether an overlay helps the
# highest-Sharpe configuration found so far, and equal weight is still eligible to be that
# configuration. One set per label holds both stages.

# %%
for label in labels:
    members = study.backtests.freeze(
        results.filter(pl.col("label") == label),
        name=f"crypto-signal-allocation-{label}",
    )
    print(f"{members.name}: {len(members.members)} members")

# %% [markdown]
# ## 6. What to notice
#
# **A sizing rule can only redistribute what the ranking selected.** None of the six changes
# which contracts are held; they change how much of each. So the ceiling on what this stage can
# add is set by the entry rule, and a ranking that selects the wrong contracts cannot be sized
# into a good strategy. That is the reason the funnel puts sizing after selection rather than
# searching the two together.
#
# **The paired difference is the only honest reading of a stage increment.** Every allocation row
# above has a baseline row built on the same prediction set, the same checkpoint, the same entry
# rule, the same costs and the same funding, and the difference between those two numbers is the
# allocator. The difference between this stage's highest Sharpe and the previous stage's highest
# Sharpe is not: those are two different configurations, and most of the gap between them is the
# search that produced them.
#
# **Two allocators that look similar are doing different amounts of estimation.** `inverse_vol`
# needs one number per contract; `risk_parity`, `hrp` and `mvo_ledoit_wolf` need a whole
# covariance matrix, estimated from the same window. The more parameters an allocator estimates
# from a fixed history, the more of its weights are noise, and on nineteen contracts with a few
# hundred observations that is not a small consideration. It is also why the shrinkage in
# `mvo_ledoit_wolf` and the clustering in `hrp` exist at all - both are ways of asking the same
# data for fewer numbers.
#
# **An unpaired row is reported, not quietly dropped.** The conformal results are real results
# and they are registered like the others; what they are not is comparable to a baseline that was
# holding positions while they were still calibrating. Excluding them from the paired frame while
# leaving them in the grid table is the distinction, and the frame printed beside the count names
# which allocator it covers and which folds those rows actually traded.
#
# **The count of observations is not the count of exposure.** A flat day still registers a return
# of zero, so period counts, and any check built on them, agree exactly between a result that
# traded a fold and one that sat it out. This is the kind of difference that a comparison hides
# rather than reports, and finding it needs the return series rather than the summary row.
#
# **Known limitations.** The rolling window is one length for every moment-based allocator, so
# nothing here says whether a different amount of history would suit one of them better - that
# would be another search axis, and adding it would widen the very search the funnel narrows.
# Costs are the flat declared schedule, which sizing interacts with directly, since an allocator
# that spreads capital more evenly turns over more of the book at each rebalance;
# [`15_costs`](15_costs.ipynb) varies that assumption. And every number is measured on the
# validation folds.
#
# **Next**: [`15_costs`](15_costs.ipynb) holds the surviving configuration fixed and varies what
# it costs to trade, which is the one stage in the funnel that selects nothing.
