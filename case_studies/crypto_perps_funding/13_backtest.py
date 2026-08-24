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
# # Crypto perpetuals: the baseline that turns a ranking into a book
#
# The model notebooks produced rankings. At each eight-hour funding timestamp, each configuration
# scores the contracts it can see and orders them. A ranking is not a portfolio, and nothing so
# far says what a reader would have earned holding one.
#
# This notebook builds the crudest portfolio that a ranking supports and runs it. An **entry
# rule** turns the scores at one timestamp into a set of positions - take the five highest-scored
# contracts long and the five lowest short, say - and every position gets the same weight. Equal
# weight is deliberate. It is the one sizing choice that contributes no information of its own, so
# a difference between two configurations here is a difference between their rankings and nothing
# else. [`14_portfolio_management`](14_portfolio_management.ipynb) is where sizing starts to vary.
#
# **Funding is the reason this case study exists, and it is settled inside the run.** A perpetual
# future never expires, so no delivery date forces its price towards spot. The exchange applies a
# **funding rate** instead: every eight hours, whoever is long pays whoever is short an amount
# proportional to the gap between the perpetual and the index, and when the gap is negative the
# payment runs the other way. That is a cash flow the holder receives or pays whatever the price
# does. A position can pay while its price prediction is wrong, and a price-only equity curve is
# therefore not the return on a perpetual position - it is a different quantity. The backtest
# boundary settles the official rate against the position carried into each timestamp, before any
# fill at that same timestamp, and the rates it used are part of what identifies the result.
#
# **Nothing is selected here.** Every declared configuration gets a baseline, the results are
# registered, and the ranking of one against another is read in
# [`17_strategy_analysis`](17_strategy_analysis.ipynb).
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - State, for one strategy, the moment the decision is taken, the moment it is filled, how long
#   the position is held, and when the next decision is allowed, and check that the four agree
#   with the horizon the label was built on.
# - Say why a long-short rule that asks for ten contracts a side cannot be run on a universe of
#   nineteen, and read which members of a declared grid the shared selector dropped.
# - Run every member of a frozen prediction population through one entry rule and have each
#   result registered with the funding settlements that produced it.
# - Recognise an equal-weight backtest as the reference every later sizing, cost and risk variant
#   is measured against, rather than as a candidate in its own right.
#
# **Book reference**: Chapter 16 (Strategy Simulation).
#
# **Prerequisites**: the model notebooks [`06_linear`](06_linear.ipynb) through
# [`10_dl_tcn`](10_dl_tcn.ipynb) have registered their complete validation prediction populations.
#
# **What it writes**: one `stage='signal'` backtest per prediction set and entry rule, in
# `run_log/registry.db`, grouped into one immutable population per entry rule and one candidate
# set per label. [`14_portfolio_management`](14_portfolio_management.ipynb) reads those candidate
# sets.

# %%
"""Run the equal-weight baseline for every declared crypto perpetuals prediction set."""

import json
from datetime import timedelta

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from case_studies.crypto_perps_funding.research_workflow import (
    ALL_LABELS,
    freeze_official_model_population,
)
from case_studies.research import CandidateSet, Result, open_study, run_backtests
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    get_rebalance_step,
    load_backtest_prices_for,
)
from case_studies.utils.coverage import check_prediction_coverage
from case_studies.utils.sweep_config import get_entry_schemes_for
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

# %% [markdown]
# ## 1. The population this notebook backtests
#
# A **population** is a named, immutable list of results, written down before the work that
# produces them starts. The model notebooks each published one; the call below re-derives the
# complete case-wide list from the training menus and records it under one name, so that what
# follows is measured against a declaration rather than against whatever the registry happens to
# contain. If a configuration is declared and missing, or present and incomplete, the check two
# cells down fails here rather than producing a baseline over a silently smaller set.
#
# Freezing is a canonical-run step. A run against a private workspace reads the released
# predictions and adds its own; it does not redeclare the published population.

# %%
if EXECUTION_TIER == "canonical" and not WORKSPACE:
    prediction_population = freeze_official_model_population(study)
    print(
        f"declared population {prediction_population.name}: "
        f"{len(prediction_population.members)} prediction sets"
    )

# %% [markdown]
# The catalog is the registry read as one frame: one row per prediction set, with the
# configuration that produced it, the checkpoint it was written at, and whether every fold and
# every expected row is present. `complete` is the column that matters before a backtest -
# a prediction set missing a fold would produce an equity curve with a hole in it and no error.

# %% tags=["results"]
catalog = study.predictions.table().filter(
    (pl.col("split") == "validation") & pl.col("label").is_in(labels)
)
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("the validation prediction catalog contains incomplete members")
if catalog.get_column("identity_status").n_unique() != 1:
    raise RuntimeError("the validation prediction catalog mixes identity versions")

catalog.group_by("family", "label").agg(
    configurations=pl.col("config_name").n_unique(),
    prediction_sets=pl.len(),
    checkpoints=pl.col("checkpoint_value").n_unique(),
).sort("family", "label")

# %% [markdown]
# Gradient boosting contributes many more prediction sets than configurations because a boosted
# model is scored at ten points along its own training, and each of those checkpoints is a
# separate configuration to be backtested rather than a variant of one. A linear fit has a single
# state and so contributes one prediction set per configuration.
#
# ## 2. The decision clock
#
# Four moments define a trade, and a backtest is only meaningful when they line up with the label
# the model was fitted on. For this case study `config/setup.yaml` declares them in the `decision`
# block:
#
# - **The information cutoff** is the pre-funding snapshot. Features are computed from data
#   observable strictly before the settlement, so nothing the model reads is contemporaneous with
#   the payment it is trading around.
# - **The fill** is at the funding timestamp itself. The engine executes at that bar rather than
#   the next one, which is what makes the position the funding is charged against the position the
#   decision asked for.
# - **The holding period** is the label's own horizon: eight hours for the three eight-hour
#   labels, twenty-four for `fwd_ret_24h`.
# - **The next decision** comes one *rebalance step* later. A step is counted in slots of the
#   eight-hour funding schedule, and it is what keeps holding periods from overlapping: a
#   twenty-four-hour position cannot be re-decided at the next settlement without the second trade
#   sitting inside the first one's window, so `fwd_ret_24h` advances three slots and the
#   eight-hour labels advance one.
#
# The check below reads the decision times out of one prediction set per label and confirms that
# every interval between consecutive decisions inside a fold is exactly the label's horizon. It
# reads one prediction set rather than all of them because completeness has already established
# that every configuration for a label predicts the same keys.


# %%
def reference_predictions(label: str) -> pl.DataFrame:
    """Load one label's first prediction set, as the reference for its decision keys."""
    reference = catalog.filter(pl.col("label") == label).sort("prediction_hash")
    return Result.open(study, reference.item(0, "prediction_hash")).load()


# %%
def decision_timeline(label: str) -> pl.DataFrame:
    """Return the distinct fold and decision timestamps one label was predicted at."""
    return (
        reference_predictions(label).select("fold", "timestamp").unique().sort("fold", "timestamp")
    )


# %%
def holding_periods(timeline: pl.DataFrame, step: int) -> list[timedelta]:
    """Return the distinct gaps between decisions `step` slots apart inside a fold."""
    return (
        timeline.with_columns(pl.col("timestamp").shift(-step).over("fold").alias("exit"))
        .drop_nulls("exit")
        .select((pl.col("exit") - pl.col("timestamp")).alias("held"))
        .get_column("held")
        .unique()
        .to_list()
    )


# %% tags=["results"]
decision = setup["decision"]
intervals = []
for label in labels:
    timeline = decision_timeline(label)
    step = get_rebalance_step("crypto_perps_funding", label)
    horizon = study.labels.get(label).definition.horizon.upper()
    if not horizon.endswith("H") or not horizon.removesuffix("H").isdigit():
        raise RuntimeError(f"unsupported crypto label horizon {horizon!r}")
    held = holding_periods(timeline, step)
    if held != [timedelta(hours=int(horizon.removesuffix("H")))]:
        raise RuntimeError(f"{label} decision intervals {held} do not match horizon {horizon}")
    intervals.append(
        {
            "label": label,
            "information_cutoff": decision["snapshot"],
            "fill": decision["execution_delay"],
            "outcome_horizon": horizon,
            "rebalance_step_slots": step,
            "decision_times": timeline.height,
            "first_decision": timeline.get_column("timestamp").min(),
            "last_decision": timeline.get_column("timestamp").max(),
        }
    )
pl.DataFrame(intervals).sort("label")

# %% [markdown]
# ### Every decision the declaration asks for
#
# The interval check above reads the gaps between consecutive decisions and cannot see a decision
# that is not there. A fold that ends early, or is missing outright, still has correct gaps
# between the decisions it does contain, so the check passes on a prediction set covering half
# the period it claims. Every other guard in the pipeline is relative in the same way -
# completeness compares one configuration's key count against its peers', and a fault upstream of
# the fit moves every peer together.
#
# `check_prediction_coverage` compares against the declaration instead: the fold boundaries in
# `config/setup.yaml`, and the sessions the label artifact actually holds inside them. It asks
# that the folds present are the folds declared, that every declared session carries a row, and
# that the declared folds account for the whole window.

# %% tags=["results"]
coverage = [
    check_prediction_coverage(
        reference_predictions(label),
        "crypto_perps_funding",
        label,
        case_dir=study.root,
    )
    for label in labels
]
pl.DataFrame(
    [
        {
            "label": report.label,
            "declared_folds": report.declared_folds,
            "declared_sessions": report.expected_sessions,
            "observed_sessions": report.observed_sessions,
        }
        for report in coverage
    ]
).sort("label")

# %% [markdown]
# ## 3. Which entry rules the universe can support
#
# `config/setup.yaml` declares two axes for this stage. The **top-k** axis takes the k
# highest-scored contracts long and the k lowest short. The **quantile** axis cuts the
# cross-section into equal-sized groups and trades the extreme two against each other; with five
# groups that is the top fifth long and the bottom fifth short.
#
# Both are long-short, and a long-short book cannot hold the same contract on both sides, so a
# top-k rule needs `2k` distinct contracts quoting at every timestamp it trades. The universe here
# is nineteen perpetual contracts and it is unbalanced - a contract enters the panel when it is
# listed, so early timestamps carry fewer than nineteen. The declared grid asks for k of 3, 5 and
# 10; ten a side needs twenty names and there are nineteen at the very best, so that member is
# not a strategy that performs badly, it is a request the cross-section cannot fill.
#
# `get_entry_schemes_for` applies that arithmetic and returns the feasible members. Reading which
# ones it dropped is worth doing explicitly: a rule silently missing from a sweep looks exactly
# like a rule that was never declared.
#
# **Feasible is not the same as filled at every decision.** The selector asks whether a rule can
# ever be filled, against the nineteen contracts the universe declares. Whether it is filled at
# one particular timestamp is a different question, and the answer varies across the period: the
# allocator computes `min(k, n/2)` per timestamp, so a rule the selector kept still takes fewer
# names than it asked for wherever the cross-section is thin. The figure below is what separates
# the two questions, and neither the feasibility table nor the backtest reports it.

# %%
n_assets = int(setup["universe"]["n_assets"])
declared = setup["backtest"]["sweep"]
schemes_by_label = {}
for label in labels:
    schemes = get_entry_schemes_for(
        "crypto_perps_funding", label, n_assets=n_assets, long_short=True
    )
    if not schemes:
        raise RuntimeError(f"no feasible entry rule remains for {label}")
    schemes_by_label[label] = schemes

feasibility = pl.DataFrame(
    [
        {
            "label": label,
            "axis": "top_k",
            "requested": f"k={k}",
            "contracts_needed": 2 * int(k),
            "runs": any(scheme.get("top_k") == int(k) for scheme in schemes_by_label[label]),
        }
        for label in labels
        for k in declared["top_k_grid"][label]
    ]
    + [
        {
            "label": label,
            "axis": "quantile",
            "requested": f"{q} groups",
            "contracts_needed": 2 * int(q),
            "runs": any(scheme.get("n_quantiles") == int(q) for scheme in schemes_by_label[label]),
        }
        for label in labels
        for q in declared["quantile_grid"][label]
    ]
).sort("label", "axis", "requested")
feasibility

# %% [markdown]
# ### How thin the panel actually gets
#
# The count above is the universe at full listing. What decides whether a rule can be filled on a
# given day is how many contracts were quoting *then*, and that is a series rather than a
# constant. The chart draws it against the two thresholds the declared grid asks for.

# %%
breadth = reference_predictions(labels[0]).group_by("timestamp").len().sort("timestamp")

# %%
fig_breadth = go.Figure(
    go.Scatter(
        x=breadth.get_column("timestamp").to_list(),
        y=breadth.get_column("len").to_list(),
        mode="lines",
        line={"color": COLORS["blue"], "width": 1.5},
        name="Contracts scored",
    )
)
for k, style in ((5, "dot"), (10, "dash")):
    fig_breadth.add_hline(
        y=2 * k,
        line={"color": COLORS["amber"] if k == 10 else COLORS["neutral"], "dash": style},
        annotation_text=f"needed for k={k} a side",
        annotation_position="top left",
    )
fig_breadth.update_layout(
    title={
        "text": "The panel never supports a ten-a-side long-short book"
        "<br><sup>Contracts scored at each eight-hour decision, validation period</sup>",
        "x": 0.02,
        "xanchor": "left",
    },
    xaxis_title="Decision timestamp",
    yaxis_title="Contracts scored",
    showlegend=False,
)
show_plotly_with_alt(
    fig_breadth,
    "Line chart of the number of perpetual contracts scored at each eight-hour decision over the "
    "validation period, with two horizontal reference lines at ten and twenty contracts marking "
    "what a five-a-side and a ten-a-side long-short book need. The series starts at fourteen "
    "contracts in January 2022 and ends at nineteen, and it never touches the twenty line, so "
    "the ten-a-side rule is never fillable. It drops below the ten line in two separate "
    "episodes rather than trending: to five between 2 October and 2 November 2022, and to eight "
    "between 9 April and 10 May 2023, ninety-three decisions each and 186 of 2,189 in total. "
    "Across those the five-a-side rule truncates to whatever the cross-section holds rather "
    "than failing.",
)

# %% [markdown]
# ## 4. Running the grid
#
# `run_backtests` takes the selected catalog rows and one entry rule, resolves each into a
# complete strategy specification, computes the identity that specification implies, and only then
# executes. Resolution is where the case-study specifics enter: the engine configuration from
# `config/backtest/base.yaml`, the fee schedule, the fill timing, the price series, and - for this
# case study alone - the official funding rates joined to the exact symbol-timestamp pairs the
# prices cover. All of it is hashed into the result's identity, so a run whose funding data
# changed is a different result rather than the same one with different numbers.
#
# Prices are loaded once per label and passed in. The boundary would load them itself, and loads
# the same rows either way, but it would do so twice for every configuration.
#
# Each call publishes an immutable population of exactly the backtests it is about to produce, and
# requires every member to exist and be complete afterwards. Re-running the notebook re-derives
# the same identities, finds them registered, and returns the stored results rather than
# re-executing - so the cost of a second run is reading the data.

# %%
config = get_backtest_config("crypto_perps_funding")
print(
    f"Costs: {config.commission_bps:.1f} bps commission and "
    f"{config.slippage_bps:.1f} bps slippage per leg, on {config.initial_cash:,.0f} of capital"
)

# %%
executions = []
for label in labels:
    prices = load_backtest_prices_for(
        "crypto_perps_funding", label, split="validation", warmup_periods=0
    )
    label_rows = catalog.filter(pl.col("label") == label)
    for scheme in schemes_by_label[label]:
        signal = {key: value for key, value in scheme.items() if key != "name"}
        execution = run_backtests(
            study,
            predictions=label_rows,
            signal=signal,
            prices=prices,
            chapter="ch16",
            population_name=f"crypto-signal-{label}-{scheme['name']}-{POPULATION_SUFFIX}",
        )
        executions.append((label, scheme["name"], execution))
        print(f"{label} / {scheme['name']}: {len(execution.results)} backtests registered")

# %% [markdown]
# ### The candidate set each label hands on
#
# A **candidate set** is the population downstream stages are allowed to choose from. Registry
# presence is not membership: a result exists in the registry the moment it is written, and the
# candidate set is the separate statement admitting it to a comparison. One set per label holds
# every baseline for that label, across both entry rules, and
# [`14_portfolio_management`](14_portfolio_management.ipynb) opens it by name rather than being
# handed a list of hashes.

# %%
for label in labels:
    members = [
        result
        for member_label, _, execution in executions
        if member_label == label
        for result in execution.results
    ]
    candidates = CandidateSet.create(study, f"crypto-signal-{label}", members)
    print(f"{candidates.name}: {len(candidates.members)} members")

# %% [markdown]
# ## 5. What came out
#
# One row per label and entry rule, read back from the registry rather than from the objects the
# loop returned. `sharpe` is the annualized ratio of mean daily return to its standard deviation,
# on the crypto calendar of 365 days; the median and the spread across configurations describe the
# population, and the count of configurations above zero says how much of it made money at all.
# `avg_turnover` is the fraction of the book replaced at an average rebalance, which is what the
# commission and slippage columns are charged on.

# %% tags=["results"]
results = study.backtests.table().filter(
    (pl.col("stage") == "signal")
    & (pl.col("split") == "validation")
    & pl.col("label").is_in(labels)
)
if results.filter(~pl.col("complete")).height:
    raise RuntimeError("the signal-stage backtest catalog contains incomplete members")

signal_grid = (
    results.with_columns(
        entry_rule=pl.when(pl.col("signal_method") == "equal_weight_top_k")
        .then(
            pl.lit("top-")
            + pl.col("spec_json").str.json_path_match("$.strategy.signal.top_k")
            + pl.lit(" a side")
        )
        .otherwise(pl.col("signal_method"))
    )
    .group_by("label", "entry_rule")
    .agg(
        backtests=pl.len(),
        median_sharpe=pl.col("sharpe").median(),
        min_sharpe=pl.col("sharpe").min(),
        max_sharpe=pl.col("sharpe").max(),
        above_zero=(pl.col("sharpe") > 0).sum(),
        median_turnover=pl.col("avg_turnover").median(),
        median_trades=pl.col("num_trades").median(),
    )
    .sort("label", "entry_rule")
)
signal_grid

# %% [markdown]
# **`entry_rule` is what was requested, not what every decision traded.** The label is read from
# `strategy.signal.top_k` in the registered specification, so a `top-5 a side` row is named for the
# book it asked for. The allocator takes `min(k, n/2)` at each decision, so how often the name
# overstates the book depends on `k` and not only on the panel:
#
# - **`top-5 a side` narrows at 186 of the decisions** - both sub-ten episodes, because five a
#   side needs ten names and the panel holds five in the first and eight in the second.
# - **`top-3 a side` narrows at 93** - only the five-contract episode. Where the panel holds
#   eight, `min(3, 4)` is 3 and a three-a-side book fills exactly as named.
# - The quantile rule takes a fraction of whatever is quoted, so it has no fixed width to fall
#   short of and is not affected.
#
# The counts are the same for all four labels: every one predicts on the same eight-hour decision
# grid, and `fwd_ret_24h` differs only in how long a position is then held, not in when it is
# opened. `fwd_ret_24h` has 2,187 decisions to the others' 2,189, which is its longer horizon
# retiring the last two of each fold.
#
# This is a caveat on reading the table, not a defect in the results. Within a label every
# configuration met the same cross-section on the same dates, so the comparison between rows holds
# even where the name overstates the book.

# %% [markdown]
# ### The spread the baseline produces
#
# One panel per label, one distribution per entry rule, over every configuration that label
# declared. The zero line is the reference: a point below it is a configuration whose ranking,
# traded equally weighted and charged the declared costs and the funding it actually paid, lost
# money over the validation period.
#
# Read the *width* rather than the extreme. Every configuration in a panel saw the same contracts
# over the same timestamps, so the spread within a panel is what changing the model does at fixed
# sizing, and it is the quantity the later stages have to beat to be worth their extra machinery.
# A panel's highest point is the largest of many draws, and how much of it is the draw rather than
# the model is what [`17_strategy_analysis`](17_strategy_analysis.ipynb) accounts for.

# %%
panel_labels = [label for label in labels if results.filter(pl.col("label") == label).height]
fig_spread = make_subplots(
    rows=len(panel_labels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=panel_labels,
)
rules = sorted(set(results.get_column("signal_method")))
for row, label in enumerate(panel_labels, start=1):
    for rule, color in zip(rules, (COLORS["blue"], COLORS["amber"]), strict=False):
        panel = results.filter((pl.col("label") == label) & (pl.col("signal_method") == rule))
        fig_spread.add_trace(
            go.Box(
                x=panel.get_column("sharpe").to_list(),
                name=rule,
                marker_color=color,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker={"size": 3, "opacity": 0.5},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
    fig_spread.add_vline(
        x=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row, col=1
    )
fig_spread.update_xaxes(title_text="Annualized validation Sharpe", row=len(panel_labels), col=1)
fig_spread.update_layout(
    title="Equal-weight baseline Sharpe by label and entry rule",
    height=260 * len(panel_labels),
    width=1000,
    legend_title="Entry rule",
    margin=dict(t=90),
)
show_plotly_with_alt(
    fig_spread,
    "Box plots with every configuration overlaid as a point, one panel per prediction label and "
    "one box per entry rule within each panel, showing annualized validation Sharpe. Each panel "
    "carries a dashed vertical line at zero. The distributions straddle zero in every panel and "
    "the two entry rules overlap heavily within each label, so neither rule separates from the "
    "other and no label separates from the rest.",
)

# %% [markdown]
# ## 6. What to notice
#
# **An equal-weight baseline is a measuring instrument, not a candidate.** It exists so that the
# stages after it can change exactly one thing and attribute the difference. Sizing changes in
# `14_portfolio_management`, the cost assumption in [`15_costs`](15_costs.ipynb), an exit overlay
# in [`16_risk_management`](16_risk_management.ipynb) - each against the same rankings, the same
# timestamps and the same funding. A comparison that changes the model *and* the sizing measures
# neither.
#
# **The funding settlement is inside the identity, which is what makes the later comparisons
# possible.** Nothing here reconstructs the equity curve afterwards to add funding on top. Had it
# done so, the registered return and the funding-adjusted return would be two different series
# with one hash between them, and every downstream stage would have to be told which one it was
# reading. Because the settlement happens in the engine, the registered return, the turnover and
# the drawdown all describe the same book.
#
# **A grid member the cross-section cannot fill is a declaration problem, not a result.** The
# ten-a-side rule is in `setup.yaml` and is never run, and the count of what was requested against
# what executed is in the notebook for that reason. The alternative - letting the engine take
# whatever names are available and calling it a ten-a-side book - produces a result that is
# reported under a name it does not match.
#
# **Two folds, and a cross-section under twenty.** Each Sharpe above is estimated from two
# validation years on a panel that starts thinner than it ends. The spread within a panel is
# therefore wide for reasons that have nothing to do with the models, and a difference of the
# same size as that spread is not evidence of anything.
#
# **Known limitations.** The baseline charges a flat commission and slippage to every contract,
# while the fee schedule this exchange publishes separates the largest contracts from the rest;
# `15_costs` is where that assumption is varied rather than assumed away. Positions are held for
# exactly the label horizon with no exit condition, which `16_risk_management` relaxes. And every
# number here is measured on the validation folds, which the case study has read many times by the
# time it reaches this notebook.
#
# **Next**: [`14_portfolio_management`](14_portfolio_management.ipynb) keeps the rankings and the
# entry rules fixed and varies how much capital each admitted position gets.
