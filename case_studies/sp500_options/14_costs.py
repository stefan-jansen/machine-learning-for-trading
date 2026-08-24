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
# # S&P 500 Options: Cost Sensitivity
#
# Selling an at-the-money straddle collects a premium of a few percent of the underlying's value,
# and the quoted bid-ask spread on those same contracts is a large fraction of that premium. The
# strategy's result therefore depends less on the model than on what fraction of the quoted spread
# a trader actually pays. This notebook measures that dependence by re-running one strategy per
# model family at each of the declared spread fractions and on both option universes.
#
# Two things make option costs different from the equity convention used elsewhere in the book.
# They are denominated as a share of the option premium rather than in basis points of notional,
# because the premium and the notional differ by more than an order of magnitude. And the
# hold-to-expiry construction pays the spread only on the way in: the position is settled in cash
# at intrinsic value, so there is no exit trade to cross a spread on.
#
# Cost variants are diagnostics. They never join the candidate set a strategy is selected from,
# because varying the cost assumption after the fact and keeping the most favourable answer is a
# way of choosing a result rather than measuring one. Their interpretation is in
# `16_strategy_analysis`.
#
# **Learning objectives**
#
# - Express a trading cost in the unit the instrument is quoted in, and say what the unit implies
#   about which side of the trade pays it.
# - Re-run a fixed strategy across a declared cost grid so that the sensitivity is measured on
#   paired series rather than inferred from a single run.
# - Keep a sensitivity sweep out of the set a selection ranges over.
#
# **Book reference**: Chapter 18
#
# **Prerequisites**: the complete baseline population published by `12_backtest`.

# %%
"""Execute the declared S&P 500 options cost-sensitivity population."""

import plotly.express as px
import polars as pl

from case_studies.research import OfficialPopulation, Result
from case_studies.sp500_options.research_workflow import (
    open_study,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.sweep_config import get_htm_cost_cascade
from utils.style import COLORS, show_plotly_with_alt

CASE_STUDY = "sp500_options"
BASELINE_POPULATION = "sp500-options-baseline-validation-v1"
COST_POPULATION = "sp500-options-cost-sensitivity-validation-v1"

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_BASELINE_HASHES: tuple[str, ...] = ()
PREVIEW_COST_FRACTIONS: tuple[float, ...] = (0.203,)
PREVIEW_UNIVERSES: tuple[str, ...] = ("liquid",)

# %% [markdown]
# ## One strategy per model family
#
# The sweep runs on one representative from each family rather than on the whole baseline
# population, because the question is how the result moves with the cost assumption and not which
# model answers it best. Each family contributes its highest-Sharpe baseline, with the backtest
# identity breaking exact ties, so the choice is reproducible from the population alone.
#
# Keeping one per family rather than a single overall representative is what allows the curves to
# be read against each other: if the sensitivity to costs were driven by the model rather than by
# the instrument, the four curves would separate.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
if EXECUTION_TIER == "canonical":
    population = OfficialPopulation.one(study, name=BASELINE_POPULATION)
    baseline_hashes = population.require_complete()
    baseline = study.backtests.table().filter(pl.col("backtest_hash").is_in(baseline_hashes))
else:
    if not WORKSPACE or not PREVIEW_BASELINE_HASHES:
        raise ValueError("preview execution requires WORKSPACE and PREVIEW_BASELINE_HASHES")
    baseline = study.backtests.table(include_preview=True).filter(
        (pl.col("execution_tier") == "preview")
        & pl.col("backtest_hash").is_in(PREVIEW_BASELINE_HASHES)
    )
    if baseline.height != len(PREVIEW_BASELINE_HASHES):
        raise ValueError("preview baseline selection is missing or ambiguous")
if baseline.is_empty() or baseline.filter(~pl.col("complete")).height:
    raise RuntimeError("cost sensitivity requires complete baseline results")
if baseline.get_column("sharpe").null_count():
    raise RuntimeError("a baseline result carries no Sharpe ratio")

representatives = (
    baseline.sort("sharpe", "backtest_hash", descending=[True, False])
    .group_by("family", maintain_order=True)
    .head(1)
    .sort("family")
)

# %% tags=["results"]
representatives.select(
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "sharpe",
    "backtest_hash",
)

# %% [markdown]
# ## The cost grid
#
# A **spread fraction** is the share of the quoted half-spread the entry trade is assumed to pay.
# At the top of the grid the order crosses the full quote. At the bottom it is filled close to the
# midpoint, which is roughly what measured effective-to-quoted spread ratios imply for algorithmic
# execution in at-the-money equity options; the grid printed below gives both ends and the values
# in between. Every other cost - the per-contract
# option commission, the per-share equity commission, the half-spread paid whenever the delta
# hedge trades the underlying - is held at its configured value, so the fraction is the only field
# that moves along a curve.
#
# The **universe** axis is the second rung of the same question. `full` prices the strategy on the
# whole at-the-money straddle surface; `liquid` restricts it to the fifth of that surface with the
# tightest quoted half-spread on each decision date. The canonical strategy in `12_backtest` is
# pinned to `liquid`; this is where the restriction is priced rather than assumed.
#
# The concentration is fixed at the cascade's own value for every request here, so the curves are
# comparable to each other. They are not paired with the baselines the representatives came from,
# which hold a different number of symbols.

# %%
cascade = get_htm_cost_cascade(CASE_STUDY)
cost_fractions = tuple(float(value) for value in cascade["cost_fractions"])
universes = tuple(str(value) for value in cascade["universes"])
cost_top_k = int(cascade["top_k"])
if EXECUTION_TIER == "preview":
    cost_fractions = PREVIEW_COST_FRACTIONS
    universes = PREVIEW_UNIVERSES
if not cost_fractions or not universes:
    raise ValueError("cost-sensitivity request axes cannot be empty")
print(f"Spread fractions of the quoted half-spread: {list(cost_fractions)}")
print(f"Universes: {list(universes)}; symbols held per decision date: {cost_top_k}")

# %% [markdown]
# ## The requests
#
# One request per representative, universe and spread fraction. Each starts from its
# representative's signal and overrides three fields: the universe restriction, the concentration,
# and the spread fraction. Within one representative and universe only the fraction varies, which
# is what makes a curve a curve.

# %%
request_rows = []
for row in representatives.iter_rows(named=True):
    baseline_result = Result.open(
        study,
        row["backtest_hash"],
        include_preview=EXECUTION_TIER == "preview",
    )
    base_signal = baseline_result.spec()["strategy"]["signal"]
    for universe in universes:
        if universe not in {"full", "liquid"}:
            raise ValueError(f"unsupported option cost universe {universe!r}")
        for fraction in cost_fractions:
            signal = dict(base_signal)
            if universe == "liquid":
                signal["universe_filter"] = "liquid"
            else:
                signal.pop("universe_filter", None)
            signal["top_k"] = cost_top_k
            signal["option_spread_fraction"] = fraction
            request_rows.append(
                {
                    "request_name": f"{row['family']}-{universe}-spread-{fraction:g}",
                    "prediction_hash": row["prediction_hash"],
                    "label": row["label"],
                    "family": row["family"],
                    "universe": universe,
                    "spread_fraction": fraction,
                    "signal": signal,
                    "allocation": None,
                    "risk": None,
                    "costs": None,
                    "chapter": "ch18",
                }
            )
requests = strategy_request_frame(request_rows)
print(
    f"{requests.height} requests: {representatives.height} families x "
    f"{len(universes)} universes x {len(cost_fractions)} fractions"
)

# %% [markdown]
# ## Execute
#
# Each request resolves its own contracts, because the universe restriction changes which symbols
# are eligible and the concentration changes how many are held. The engine validates the paired
# option lifecycle, the cash settlement, the retained hedge and every cost input before publishing.

# %%
execution = run_official_backtest_requests(
    study,
    requests,
    population_name=COST_POPULATION if EXECUTION_TIER == "canonical" else None,
)
catalog = execution.catalog_rows.sort("request_name")
if catalog.height != requests.height or catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("cost execution did not publish every declared request")

# %% [markdown]
# ## What the run produced
#
# Read each line for its slope rather than its level. The slope is how much of the result is a
# claim about execution quality; the gap between the two universes at the same fraction is what
# the liquidity restriction buys. Whether any line clears zero by enough to matter needs the
# interval around each point, which `16_strategy_analysis` reports.

# %%
cost_curve = (
    study.backtests.table(include_preview=EXECUTION_TIER == "preview")
    .select("backtest_hash", "sharpe")
    .join(catalog.select("request_name", "backtest_hash"), on="backtest_hash", how="inner")
    .join(
        requests.select("request_name", "family", "universe", "spread_fraction"),
        on="request_name",
        how="inner",
    )
    .sort("family", "universe", "spread_fraction")
)
if cost_curve.height != catalog.height or cost_curve.get_column("sharpe").null_count():
    raise RuntimeError("the published cost population is missing rows or Sharpe metrics")

# %% tags=["results"]
cost_curve.pivot(
    on="spread_fraction",
    index=["family", "universe"],
    values="sharpe",
).sort("family", "universe")

# %%
cost_figure = px.line(
    cost_curve,
    x="spread_fraction",
    y="sharpe",
    color="family",
    line_dash="universe",
    markers=True,
    hover_data=["backtest_hash"],
)
cost_figure.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
cost_figure.update_layout(
    title="Validation Sharpe against the share of the quoted spread paid on entry",
    height=520,
    width=1000,
    margin=dict(t=70),
    legend_title_text="family / universe",
)
cost_figure.update_xaxes(title_text="Fraction of the quoted option half-spread paid on entry")
cost_figure.update_yaxes(title_text="Validation Sharpe")
show_plotly_with_alt(
    cost_figure,
    "Line chart of validation Sharpe against the fraction of the quoted option half-spread paid "
    "on entry, one line per model family and option universe.",
)

# %% [markdown]
# ## Key takeaways
#
# - A cost is stated in the unit its instrument is quoted in. A basis-point-of-notional convention
#   applied to an option premium misstates the friction by the ratio between premium and notional,
#   which for an at-the-money straddle is large.
# - Holding to expiration removes the exit-side spread entirely, so the entry fill is where
#   execution quality shows up. That is a property of the construction, not of the model.
# - A sensitivity grid is evidence about robustness only while it stays outside the set a strategy
#   is selected from. Once a cost assumption can be chosen after seeing the result, the grid has
#   become a search.
#
# **Known limitations**: the spread fractions are declared assumptions calibrated from published
# execution studies, not fills this strategy achieved, so the curve says what the result would be
# under each assumption and not which assumption holds. The hedge spread and both commissions
# are held fixed, so their contribution is inside every point rather than resolved along an axis.
