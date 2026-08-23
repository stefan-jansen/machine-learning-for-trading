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
# # Cost Taxonomy and Fee Schedules
#
# **Docker image**: `ml4t`
#
# A trading cost is charged in whatever unit the venue happens to use: cents per
# share, dollars per contract, a spread quoted in the currency pair itself, or a
# percentage of notional. This notebook reads the daily bars and the published fee
# schedules for six markets, keeps each quantity in the unit it was measured in,
# and converts only where the conversion has a defensible basis. It then asks how
# much gross return a strategy has to earn before those costs are paid.
#
# A basis point (bp) is one hundredth of a percent, and it is the unit almost every
# cost in this chapter is quoted in: 10 bps of a $1,000,000 order is $1,000.
#
# **Learning Objectives:**
# - Read the daily volume of a market and say whether it can be turned into dollars
#   traded per day, given what the price series in hand actually measures
# - Read a published fee schedule and name the trade details you would still need
#   before that fee becomes a number you can subtract from a return
# - Work out which of commission, spread and price impact is the largest part of the
#   cost of an order, and how that answer changes as the order gets bigger
# - Compute the gross annual return a strategy needs just to pay for its trading,
#   from how often it trades and what one round of trading costs
# - Add the cost of borrowing stock to sell short to that same hurdle
#
# **Book Reference:** Chapter 18, Sections 18.1 and 18.2
#
# **Prerequisites:** Access to all six OHLCV datasets covered here and
# case study `config/setup.yaml` files.

# %%
"""Cost taxonomy and fee schedules for unit-aware scenario analysis."""

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml
from _cost_analysis import breakeven_alpha
from IPython.display import Markdown, display
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from data import (
    load_cme_futures,
    load_crypto_perps,
    load_etfs,
    load_fx_pairs,
    load_sp500_daily_bars,
    load_us_equities,
)
from utils.paths import REPO_ROOT, get_case_study_source_dir
from utils.style import COLORS, add_message_title, ml4t_palette, show_with_alt

# %% tags=["parameters"]
MAX_SYMBOLS = 0  # 0 = all symbols
SOURCE_CHECK_DATE = "2026-07-21"

# %% [markdown]
# ## 1. Liquidity Has Units
#
# Average daily volume (ADV) is the average number of units that change hands in a
# day, and the unit differs by market: shares for a stock, contracts for a future,
# base-asset quantity for a crypto perpetual, and - for the FX feed used here -
# the number of price updates rather than any traded amount. Multiplying volume by
# price gives dollars traded only when the two describe the same transaction. That
# holds for the equity and crypto panels, holds for futures once the contract's
# price multiplier is applied, and fails for the ETF and FX panels for reasons
# given in each section below. Those two are therefore reported in their own units.
#
# `MAX_SYMBOLS` caps how many instruments are read from each panel, taking them in
# alphabetical order so the selection never depends on the data. Leave it at 0 to
# use every instrument; set it lower to shorten a run while working.


# %%
def limit_symbols(df: pl.DataFrame, symbol_col: str) -> pl.DataFrame:
    """Keep the alphabetically first MAX_SYMBOLS instruments, or all of them when it is 0."""
    if MAX_SYMBOLS <= 0:
        return df
    symbols = df.select(symbol_col).unique().sort(symbol_col).head(MAX_SYMBOLS)
    return df.join(symbols, on=symbol_col, how="semi")


# %% [markdown]
# ### Daily Dollar-Turnover Summary
#
# Each input to this helper is already daily and carries a defensible dollar
# turnover column. The output retains the observation window so that a reader
# can see when markets are not contemporaneous.


# %%
def summarize_dollar_turnover(
    df: pl.DataFrame,
    symbol_col: str,
    date_col: str,
    asset_class: str,
) -> pl.DataFrame:
    """Summarize a precomputed daily dollar-turnover column by instrument."""
    return (
        df.group_by(symbol_col)
        .agg(
            adv_usd=pl.col("daily_turnover_usd").mean(),
            coverage_start=pl.col(date_col).min().cast(pl.Date),
            coverage_end=pl.col(date_col).max().cast(pl.Date),
        )
        .rename({symbol_col: "symbol"})
        .with_columns(asset_class=pl.lit(asset_class))
        .select("symbol", "asset_class", "adv_usd", "coverage_start", "coverage_end")
    )


# %% [markdown]
# ### Equity Dollar Turnover
#
# Daily share volume multiplied by the traded close gives approximate daily
# dollar turnover. These two panels retain their own source windows.

# %%
adv_records = []
for asset_class, frame in {
    "US Equities": limit_symbols(load_us_equities(), "symbol"),
    "S&P 500 Equities": limit_symbols(load_sp500_daily_bars(), "symbol"),
}.items():
    daily = frame.with_columns(daily_turnover_usd=pl.col("close") * pl.col("volume"))
    adv_records.append(summarize_dollar_turnover(daily, "symbol", "timestamp", asset_class))

# %% [markdown]
# ### Crypto Daily Quote Turnover
#
# Binance volume is base-asset quantity. The loader returns three funding-aligned
# bars per day, so we sum quote turnover by UTC date before computing ADV.

# %%
crypto = limit_symbols(load_crypto_perps(frequency="8h"), "symbol")
crypto_daily = (
    crypto.with_columns(
        activity_date=pl.col("timestamp").dt.date(),
        bar_turnover_usd=pl.col("close") * pl.col("volume"),
    )
    .group_by("symbol", "activity_date")
    .agg(daily_turnover_usd=pl.col("bar_turnover_usd").sum())
)
adv_records.append(
    summarize_dollar_turnover(
        crypto_daily,
        "symbol",
        "activity_date",
        "Crypto Perpetuals",
    )
)

# %% [markdown]
# ### CME Contract-Notional Turnover
#
# Futures volume counts contracts. The price multiplier is derived from tick
# value divided by tick size, matching the backtest contract specification.

# %%
spec_path = REPO_ROOT / "data" / "futures" / "market" / "futures_specs.yaml"
product_specs = yaml.safe_load(spec_path.read_text())["products"]
multiplier_df = pl.DataFrame(
    [
        {
            "product": product,
            "price_multiplier": float(spec["tick_value"]) / float(spec["tick_size"]),
        }
        for product, spec in product_specs.items()
    ]
)

cme = limit_symbols(load_cme_futures(tenors=[0]), "product").join(
    multiplier_df,
    on="product",
    how="left",
)
missing_multipliers = cme.filter(pl.col("price_multiplier").is_null())["product"].unique().to_list()
if missing_multipliers:
    raise ValueError(f"Missing CME multipliers for: {sorted(missing_multipliers)}")

cme = cme.with_columns(
    daily_turnover_usd=pl.col("volume") * pl.col("raw_close") * pl.col("price_multiplier")
)
adv_records.append(summarize_dollar_turnover(cme, "product", "session_date", "CME Futures"))

all_adv = pl.concat(adv_records)

# %% [markdown]
# ### ETF Share Activity
#
# Yahoo supplies adjusted OHLC together with observed share volume. Without raw
# historical closes or adjustment factors, multiplying these fields would not
# recover contemporaneous traded notional. ETF activity therefore remains in
# its native shares-per-day unit.

# %%
etfs = limit_symbols(load_etfs(), "symbol")
etf_activity = (
    etfs.group_by("symbol")
    .agg(
        mean_daily_shares=pl.col("volume").mean(),
        coverage_start=pl.col("timestamp").min().cast(pl.Date),
        coverage_end=pl.col("timestamp").max().cast(pl.Date),
    )
    .sort("mean_daily_shares", descending=True)
)
etf_activity

# %% [markdown]
# ### FX Tick Activity
#
# OANDA daily volume counts price updates rather than traded currency. It can
# describe activity within this dataset, but it cannot share a USD ADV axis.

# %%
fx = limit_symbols(load_fx_pairs(frequency="daily"), "symbol")
fx_activity = (
    fx.group_by("symbol")
    .agg(
        mean_daily_ticks=pl.col("volume").mean(),
        coverage_start=pl.col("timestamp").min().cast(pl.Date),
        coverage_end=pl.col("timestamp").max().cast(pl.Date),
    )
    .sort("mean_daily_ticks", descending=True)
)
fx_activity

# %% [markdown]
# ### Comparable Dollar-Turnover Summary

# %%
adv_summary = (
    all_adv.group_by("asset_class")
    .agg(
        n_symbols=pl.col("symbol").n_unique(),
        median_adv_usd=pl.col("adv_usd").median(),
        p25_adv_usd=pl.col("adv_usd").quantile(0.25),
        p75_adv_usd=pl.col("adv_usd").quantile(0.75),
        coverage_start=pl.col("coverage_start").min(),
        coverage_end=pl.col("coverage_end").max(),
    )
    .sort("median_adv_usd", descending=True)
)
adv_summary

# %% [markdown]
# ### Dollar-Turnover Distribution

# %%
adv_pd = all_adv.filter(pl.col("adv_usd") > 0).to_pandas()
adv_pd["log_adv_usd"] = np.log10(adv_pd["adv_usd"])

order = (
    all_adv.group_by("asset_class")
    .agg(pl.col("adv_usd").median())
    .sort("adv_usd", descending=True)["asset_class"]
    .to_list()
)

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.violinplot(
    data=adv_pd,
    x="asset_class",
    y="log_adv_usd",
    order=order,
    ax=ax,
    inner="quartile",
    cut=0,
    color=COLORS["blue"],
)
ax.set_ylabel("Average daily dollar turnover (log₁₀ USD)")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=15)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")

for val, label in [(6, "$1M"), (7, "$10M"), (8, "$100M"), (9, "$1B")]:
    ax.axhline(val, color=COLORS["neutral"], linestyle=":", alpha=0.3)
    ax.text(len(order) - 0.5, val + 0.05, label, fontsize=8, alpha=0.5)

add_message_title(
    ax,
    "Dollar turnover differs by market and observation window",
    subtitle="Instrument-level mean daily turnover; distributions shown on a log scale",
    source="ML4T datasets; each market retains its source coverage window",
)
fig.subplots_adjust(bottom=0.24)
show_with_alt(
    fig,
    "Violin plots of average daily dollar turnover per instrument, one violin per market, on a log scale. CME futures sit highest and US equities lowest, with each market spanning two to three orders of magnitude.",
)

# %% tags=["results"]
highest_turnover_market = adv_summary.row(0, named=True)
display(
    Markdown(
        f"**Reading the chart:** {highest_turnover_market['asset_class']} has the highest "
        f"median daily dollar turnover in this sample. The comparison is descriptive, not a "
        f"capacity estimate, because coverage ranges from "
        f"{adv_summary['coverage_start'].min()} to {adv_summary['coverage_end'].max()}. "
        "ETFs remain separate because adjusted prices cannot be paired with observed share "
        "volume for historical notional; FX remains separate because tick volume has no "
        "dollar-notional conversion."
    )
)

# %% [markdown]
# ## 2. Exchange Fee Schedules
#
# Fees use different denominators. Stocks charge per share, futures and options
# per contract, FX through a variable spread, and crypto per unit of notional.
# The table preserves those charging units and separates sourced rates from
# case-study assumptions.

# %% [markdown]
# ### One Row per Published Rate
#
# Each row records the venue, the instrument the rate applies to, the rate as the
# venue publishes it, how firm that figure is, and where it came from. The rate is
# stored as text so that a per-share fee and a per-contract fee cannot be added
# together by accident.


# %%
def fee_evidence_row(
    venue: str,
    instrument: str,
    rate: str,
    status: str,
    source: str,
) -> dict:
    """Create one source-dated fee row without coercing its charging unit."""
    return {
        "venue": venue,
        "instrument": instrument,
        "rate": rate,
        "status": status,
        "source": source,
    }


# %%
fee_source_urls = {
    "IBKR stocks": "https://www.interactivebrokers.com/en/pricing/commissions-stocks.php",
    "IBKR commissions": "https://www.interactivebrokers.com/en/pricing/commissions-home.php?menu=A",
    "NFA assessment FAQ": "https://www.nfa.futures.org/faqs/members/nfa-assessment-fees.html",
    "CME fee finder": "https://www.cmegroup.com/company/clearing-fees/fee-finder.html",
    "OANDA pricing": "https://www.oanda.com/us-en/trading/our-pricing/",
    "crypto setup.yaml": "case_studies/crypto_perps_funding/config/setup.yaml",
}

# %% [markdown]
# The first rows have explicit, dated values in official schedules.

# %%
fee_rows = [
    fee_evidence_row(
        "IBKR", "US stocks, Pro Tiered first tier", "USD 0.0035/share", "official", "IBKR stocks"
    ),
    fee_evidence_row("IBKR", "US stocks, Pro Fixed", "USD 0.005/share", "official", "IBKR stocks"),
    fee_evidence_row(
        "IBKR", "US options, Pro Fixed", "USD 0.65/contract", "official", "IBKR commissions"
    ),
    fee_evidence_row(
        "NFA",
        "Futures assessment",
        "USD 0.01/contract/side",
        "effective 2026-07-01",
        "NFA assessment FAQ",
    ),
]

# %% [markdown]
# Variable schedules retain their native qualification. Binance remains a
# case-study assumption because the public fee page is account-dependent.

# %%
fee_rows.extend(
    [
        fee_evidence_row(
            "CME",
            "Futures exchange fee",
            "Product and account specific",
            "official fee finder",
            "CME fee finder",
        ),
        fee_evidence_row(
            "OANDA",
            "Spot FX spread-only plan",
            "Pair- and market-dependent",
            "official description",
            "OANDA pricing",
        ),
        fee_evidence_row(
            "Binance USDT-M",
            "Perpetual taker example",
            "Account-tier dependent; see case-study configuration",
            "case-study assumption",
            "crypto setup.yaml",
        ),
    ]
)
fee_schedule_df = pl.DataFrame(fee_rows).with_columns(
    source_url=pl.col("source").replace_strict(fee_source_urls),
    source_checked=pl.lit(SOURCE_CHECK_DATE),
)
fee_schedule_df

# %% [markdown]
# **Interpretation:** Knowing the dollar size of an order is not enough to put
# these rates on one axis. A per-share fee needs the share price before it becomes
# a fraction of notional. A per-contract fee needs the futures price or option
# premium and the contract multiplier. The FX plan charges through the spread, so
# it needs the spread that was actually quoted, and the crypto rate depends on the
# account's fee tier. Each of those is a trade detail, and the next section supplies
# them by writing out named scenarios.

# %% [markdown]
# ## 3. Which Cost Component Is Largest
#
# The cost of an order splits into three parts. **Commission** is what the broker
# and the venue charge. **Spread** is the gap between the highest price anyone is
# currently bidding and the lowest anyone is offering, half of which an order that
# executes immediately gives up. **Impact** is
# the price move the order itself causes by consuming the resting liquidity, and
# unlike the first two it grows with the size of the order.
#
# Each scenario below fixes five numbers per market: commission and half-spread in
# basis points, the market's daily volatility `sigma`, its average daily dollar
# turnover `adv_usd`, and `impact_eta`, the coefficient that scales the impact
# model. They are round figures chosen to span the range these markets plausibly
# occupy, not estimates fitted to the data loaded above, and the answer the section
# gives holds only for the numbers written here.

# %%
cost_params = {
    "ETFs": {
        "commission_bps": 1.0,
        "spread_bps": 3.0,
        "impact_eta": 0.05,
        "sigma": 0.015,
        "adv_usd": 5e8,
    },
    "Crypto Perps": {
        "commission_bps": 4.0,
        "spread_bps": 2.0,
        "impact_eta": 0.03,
        "sigma": 0.04,
        "adv_usd": 1e9,
    },
    "CME Futures": {
        "commission_bps": 1.5,
        "spread_bps": 1.5,
        "impact_eta": 0.04,
        "sigma": 0.01,
        "adv_usd": 2e9,
    },
    "FX Pairs": {
        "commission_bps": 0.0,
        "spread_bps": 2.0,
        "impact_eta": 0.02,
        "sigma": 0.005,
        "adv_usd": 5e9,
    },
}

# %% [markdown]
# The equity scenarios extend the same illustrative parameter grid.

# %%
cost_params |= {
    "S&P 500": {
        "commission_bps": 1.0,
        "spread_bps": 2.0,
        "impact_eta": 0.05,
        "sigma": 0.015,
        "adv_usd": 1e8,
    },
    "US Equities": {
        "commission_bps": 1.0,
        "spread_bps": 8.0,
        "impact_eta": 0.10,
        "sigma": 0.025,
        "adv_usd": 5e6,
    },
}

scenario_df = pl.DataFrame(
    [{"asset_class": asset, **params} for asset, params in cost_params.items()]
).sort("asset_class")
scenario_df

# %% [markdown]
# ### Component Calculator
#
# Commission and spread do not depend on order size in this model, so they enter at
# their stated rate. Impact does. The square-root law, which Section 18.4 derives
# and `03_market_impact_calibration` fits to data, says the price move an order
# causes grows with the square root of the fraction of a day's volume it represents:
#
# $$\text{impact} = \eta \, \sigma \sqrt{\frac{Q}{\text{ADV}}}$$
#
# where $Q$ is the order's dollar size, ADV is average daily dollar turnover, and
# $\sigma$ is daily volatility. Multiplying by 10,000 states the result in basis
# points. The square root is what makes the answer to "which component is largest"
# depend on order size: doubling the order less than doubles the impact, but a
# hundredfold larger order still pays ten times the impact per dollar traded.


# %%
def cost_components(size: float, params: dict) -> dict:
    """Compute commission, spread, and square-root impact in basis points."""
    participation = size / params["adv_usd"]
    impact = params["impact_eta"] * params["sigma"] * np.sqrt(participation) * 10_000
    return {
        "Commission": params["commission_bps"],
        "Spread": params["spread_bps"],
        "Impact": impact,
    }


# %%
def dominant_component_label(costs: dict[str, float]) -> str:
    """Name the largest component or all components tied for largest."""
    largest = max(costs.values())
    leaders = [name for name, value in costs.items() if np.isclose(value, largest)]
    return " + ".join(leaders)


# %% [markdown]
# ### Map the Dominant Cost Component by Trade Size

# %%
dominance_rows = []
size_labels = ["$10K", "$100K", "$1M", "$10M", "$100M"]
size_values = [1e4, 1e5, 1e6, 1e7, 1e8]

for asset, params in cost_params.items():
    for size, label in zip(size_values, size_labels, strict=False):
        costs = cost_components(size, params)
        dominant = dominant_component_label(costs)
        total = sum(costs.values())

        dominance_rows.append(
            {
                "asset_class": asset,
                "trade_size": label,
                "trade_size_usd": size,
                "dominant": dominant,
                "total_bps": total,
            }
        )

dominance = pl.DataFrame(dominance_rows)

# %% [markdown]
# ### Dominance Map and Crossover Sizes
#
# Cell color identifies the largest component or an exact tie. The annotation
# reports total one-way cost for context. This design keeps dominance distinct
# from magnitude.

# %%
component_codes = {"Commission": 0, "Spread": 1, "Impact": 2, "Commission + Spread": 3}
dominance = dominance.with_columns(
    component_code=pl.col("dominant").replace_strict(component_codes, return_dtype=pl.Int8)
)
dom_pivot = dominance.pivot(on="trade_size", index="asset_class", values="component_code")
col_order = ["asset_class"] + size_labels
dom_pivot = dom_pivot.select([c for c in col_order if c in dom_pivot.columns])

data = dom_pivot.drop("asset_class").to_numpy()
asset_names = dom_pivot["asset_class"].to_list()
total_lookup = {
    (row["asset_class"], row["trade_size"]): row["total_bps"]
    for row in dominance.iter_rows(named=True)
}

# %% [markdown]
# The chart uses the shared ML4T palette for the three components and a tie class.

# %%
fig, ax = plt.subplots()
dominance_cmap = ListedColormap(
    [COLORS["blue"], COLORS["amber"], COLORS["copper"], COLORS["neutral"]]
)
im = ax.imshow(data, cmap=dominance_cmap, aspect="auto", vmin=-0.5, vmax=3.5)
ax.set_xticks(range(len(size_labels)))
ax.set_xticklabels(size_labels)
ax.set_yticks(range(len(asset_names)))
ax.set_yticklabels(asset_names)
ax.set_xlabel("Trade Size")

for i in range(len(asset_names)):
    for j in range(len(size_labels)):
        total_bps = total_lookup[(asset_names[i], size_labels[j])]
        text_color = "white" if data[i, j] in (0, 2, 3) else COLORS["blue"]
        ax.text(j, i, f"{total_bps:.1f}", ha="center", va="center", color=text_color)

colorbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
colorbar.ax.set_yticklabels(["Commission", "Spread", "Impact", "Commission + Spread"])
add_message_title(
    ax,
    "The largest cost component changes with trade size and scenario",
    subtitle="Color shows the largest component or tie; labels show total one-way cost in bps",
    source="Illustrative parameters displayed above; square-root impact model",
)
show_with_alt(
    fig,
    "A grid of markets by order size, each cell colored by which cost component is largest and labelled with the total one-way cost in basis points. Impact takes over only in the least liquid market at the largest sizes.",
)

# %% [markdown]
# The impact crossover solves for the order size at which modeled impact equals
# the larger fixed component. A crossover outside the displayed range explains
# why some rows retain the same color throughout the chart.

# %%
crossover_rows = []
for asset, params in cost_params.items():
    fixed_component = max(params["commission_bps"], params["spread_bps"])
    impact_scale = params["impact_eta"] * params["sigma"] * 10_000
    crossover_usd = params["adv_usd"] * (fixed_component / impact_scale) ** 2
    crossover_rows.append(
        {
            "asset_class": asset,
            "largest_fixed_component_bps": fixed_component,
            "impact_crossover_usd": crossover_usd,
        }
    )

crossover_df = pl.DataFrame(crossover_rows).sort("impact_crossover_usd")
crossover_df

# %% tags=["results"]
million_dominance = dominance.filter(pl.col("trade_size_usd") == 1_000_000).sort("asset_class")
million_reading = ", ".join(
    f"{row['asset_class']}: {row['dominant'].lower()}"
    for row in million_dominance.iter_rows(named=True)
)
display(
    Markdown(
        f"**Conditional reading:** At the $1 million scenario size, the largest components are "
        f"{million_reading}. These results follow from the displayed assumptions; they are not "
        "empirical claims about every venue or instrument."
    )
)

# %% [markdown]
# ## 4. Case Study Cost Assumptions
#
# Every case study in this book records its cost assumption in its own
# `config/setup.yaml`, and each one states it in the unit its market charges in.
# The section below reads all nine and reports which of them already carry a cost
# in basis points - the unit the rest of the book's backtests subtract - and which
# carry something that needs a price, a multiplier or an observed spread first.

# %%
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

setup_data = {}
for cs_id in CASE_STUDIES:
    setup_path = get_case_study_source_dir(cs_id) / "config" / "setup.yaml"
    if not setup_path.exists():
        continue
    setup_data[cs_id] = yaml.safe_load(setup_path.read_text())

missing_setups = sorted(set(CASE_STUDIES) - set(setup_data))
if missing_setups:
    raise FileNotFoundError(f"Missing case-study setup files: {missing_setups}")
if len(setup_data) != len(CASE_STUDIES):
    raise ValueError("Case-study setup IDs are not unique")

print(f"Loaded cost configurations: {len(setup_data)}/{len(CASE_STUDIES)}")

# %% [markdown]
# ### Reading Each Configuration's Cost Unit
#
# The function below inspects one `costs:` block and reports the unit it is stated
# in, the basis-point value where the block already holds one, and the trade detail
# that is still missing where it does not.


# %%
def cost_schema_row(cs_id: str, setup: dict) -> dict:
    """Describe a case-study cost schema without inventing a common-unit value."""
    costs = setup.get("costs", {})
    if "round_trip_cost_bps" in costs:
        classification = (
            "round-trip bps",
            float(costs["round_trip_cost_bps"]),
            "direct configuration value",
        )
    elif "fee_schedule" in costs and "taker_bps" in costs["fee_schedule"]:
        classification = (
            "bps per side",
            float(costs["fee_schedule"]["taker_bps"]),
            "one-way taker scenario",
        )
    elif "per_share" in costs:
        classification = ("USD per share plus spread", None, "needs price and turnover")
    elif "commission_per_contract" in costs:
        classification = ("USD per contract plus ticks", None, "needs contract details")
    elif "spread_bps" in costs:
        classification = ("pair-specific spread range", None, "needs observed spread")
    elif "per_leg_cost_bps_range" in costs:
        classification = ("per-leg bps range", None, "no selected scenario")
    else:
        classification = ("component-specific", None, "needs trade details")

    native_unit, comparable_value, comparison_status = classification
    components = costs.get("components", [])
    component_names = list(components) if isinstance(components, dict) else components
    return {
        "case_study": cs_id,
        "cost_class": costs.get("class", "unspecified"),
        "components": ", ".join(component_names),
        "native_unit": native_unit,
        "configured_bps": comparable_value,
        "comparison_status": comparison_status,
    }


# %%
cost_schema_df = pl.DataFrame([cost_schema_row(cs_id, setup_data[cs_id]) for cs_id in CASE_STUDIES])
cost_schema_df

# %% [markdown]
# **Interpretation:** The `configured_bps` column is filled only where the
# configuration already states a basis-point figure. Every other row names what is
# missing, and the notebook that supplies it comes later in the chapter:
# `02_spread_estimation` measures the spread the FX and equity rows need, and
# `03_market_impact_calibration` fits the impact coefficient that turns an order
# size into a cost.

# %% [markdown]
# ## 5. Breakeven Alpha Analysis
#
# Breakeven alpha is the gross return needed to offset cost. Define annual
# traded-notional turnover as dollars traded during the year divided by average
# portfolio capital. For one-way cost per dollar traded:
#
# $$\alpha_{\text{breakeven}} =
# \text{annual traded-notional turnover}
# \times \frac{\text{one-way cost in bps}}{10{,}000}$$
#
# Both inputs are read off the grid below rather than measured from a strategy, so
# the answer is a hurdle for a strategy with that turnover and that cost, not an
# estimate for any particular one.

# %%
annual_turnover_grid = [1, 3, 6, 12, 24, 60]
one_way_cost_grid = [1, 3, 5, 10, 20]
breakeven_rows = [
    {
        "annual_traded_notional": turnover,
        "one_way_cost_bps": cost_bps,
        "breakeven_alpha_pct": breakeven_alpha(turnover, cost_bps) * 100,
    }
    for turnover in annual_turnover_grid
    for cost_bps in one_way_cost_grid
]
breakeven_df = pl.DataFrame(breakeven_rows)
breakeven_df

# %% [markdown]
# ### Breakeven Sensitivity Map
#
# Each annotation is computed from the displayed turnover and cost axes.

# %%
be_pivot = breakeven_df.pivot(
    on="one_way_cost_bps",
    index="annual_traded_notional",
    values="breakeven_alpha_pct",
).sort("annual_traded_notional")
be_data = be_pivot.drop("annual_traded_notional").to_numpy()
turnover_labels = be_pivot["annual_traded_notional"].to_list()
cost_labels = [int(column) for column in be_pivot.columns[1:]]

fig, ax = plt.subplots()
cost_cmap = LinearSegmentedColormap.from_list(
    "ml4t_cost",
    [COLORS["silver_muted"], COLORS["amber"], COLORS["copper"]],
)
im = ax.imshow(be_data, cmap=cost_cmap, aspect="auto")
ax.set_xticks(range(len(cost_labels)), labels=cost_labels)
ax.set_yticks(range(len(turnover_labels)), labels=turnover_labels)
ax.set_xlabel("One-way cost per dollar traded (bps)")
ax.set_ylabel("Annual traded notional / average capital (x)")
for row_idx in range(len(turnover_labels)):
    for col_idx in range(len(cost_labels)):
        ax.text(col_idx, row_idx, f"{be_data[row_idx, col_idx]:.2f}%", ha="center", va="center")

fig.colorbar(im, ax=ax, label="Breakeven annual alpha (%)")
add_message_title(
    ax,
    "Turnover multiplies even modest one-way trading costs",
    subtitle="Breakeven gross alpha under explicit turnover and cost scenarios",
    source="Scenario grid computed from the breakeven formula above",
)
show_with_alt(
    fig,
    "A grid of annual turnover by one-way cost, each cell labelled with the gross annual return needed to break even. The requirement rises from a fraction of a percent to double digits across the grid.",
)

# %% tags=["results"]
example_turnover = 12
example_cost_bps = 5
example_breakeven = breakeven_df.filter(
    (pl.col("annual_traded_notional") == example_turnover)
    & (pl.col("one_way_cost_bps") == example_cost_bps)
)["breakeven_alpha_pct"].item()
display(
    Markdown(
        f"**Sensitivity reading:** Trading {example_turnover} times average capital per year at "
        f"{example_cost_bps} bps per dollar requires {example_breakeven:.2f}% annual gross alpha "
        "just to cover transaction costs. For your own strategy, read the turnover off its "
        "realized fills and the cost off a model calibrated to the market it trades."
    )
)

# %% [markdown]
# ## 6. Borrow Costs for Long-Short Strategies
#
# Selling a stock short means borrowing it first, and the lender charges a fee for
# the loan for as long as the position is open. That fee is quoted as an annual rate
# on the value borrowed, so it is a running cost rather than a per-trade one, and it
# applies only to the short half of the book. Rates run from a few tens of basis
# points a year on an easily borrowed large-cap to several percent on a stock that
# is hard to locate, and they move with demand, so the curves below span a range
# rather than estimating a single rate.
#
# Every curve holds the portfolio fixed and varies only the borrow rate. Gross alpha
# on the horizontal axis is the return before any cost is subtracted; the vertical
# axis is what is left after trading cost and borrow. The settings below say what each
# of the fixed quantities decides.

# %%
borrow_rate_scenarios = [40, 100, 200, 500, 1000]
short_allocation = 0.50
annual_traded_notional = 3.6
one_way_trading_cost_bps = 20
trading_cost_drag_bps = annual_traded_notional * one_way_trading_cost_bps
gross_alphas = np.linspace(0, 700, 120)

print(
    f"Short side {short_allocation:.0%} of capital - borrow is charged on this half only\n"
    f"Trades {annual_traded_notional}x capital per year at {one_way_trading_cost_bps} bps "
    f"one way, so trading costs {trading_cost_drag_bps:.0f} bps per year\n"
    f"Borrow rates spanned: {borrow_rate_scenarios} bps per year on the amount borrowed"
)

fig, ax = plt.subplots()
for borrow_bps, color in zip(
    borrow_rate_scenarios,
    ml4t_palette(len(borrow_rate_scenarios), categorical=True),
    strict=True,
):
    net_alpha = gross_alphas - trading_cost_drag_bps - borrow_bps * short_allocation
    ax.plot(gross_alphas, net_alpha, color=color, label=f"Borrow: {borrow_bps} bps/year")

ax.axhline(0, color=COLORS["neutral"], linewidth=0.8)
ax.set_xlabel("Gross Alpha (bps/year)")
ax.set_ylabel("Net Alpha (bps/year)")
ax.legend(loc="upper left")
add_message_title(
    ax,
    "Borrow cost raises the gross-alpha hurdle for long-short portfolios",
    subtitle="Net of trading-cost drag and of borrow charged on the short half of the book",
    source="Borrow-rate scenarios and portfolio assumptions stated above; not a market sample",
)

show_with_alt(
    fig,
    "Five lines of net alpha against gross alpha, one per borrow rate. All are parallel and shifted down; higher borrow rates cross zero at higher gross alpha.",
)

# %% tags=["results"]
example_gross_alpha_bps = 200
low_borrow_bps = borrow_rate_scenarios[0]
high_borrow_bps = borrow_rate_scenarios[2]
low_borrow_net = example_gross_alpha_bps - trading_cost_drag_bps - low_borrow_bps * short_allocation
high_borrow_net = (
    example_gross_alpha_bps - trading_cost_drag_bps - high_borrow_bps * short_allocation
)
display(
    Markdown(
        f"**Scenario reading:** At {example_gross_alpha_bps} bps gross alpha, the modeled net "
        f"alpha is {low_borrow_net:.0f} bps with a {low_borrow_bps} bps borrow rate and "
        f"{high_borrow_net:.0f} bps with a {high_borrow_bps} bps rate. Borrow is charged per "
        "security and per day, so a portfolio-level rate like these is a placeholder until the "
        "actual borrow quotes for the names being shorted are in hand."
    )
)

# %% [markdown]
# ## Key Takeaways
#
# 1. **Find out what a fee's denominator is before converting it.** A fee per share
#    becomes a fraction of notional only once you supply the share price; a fee per
#    contract needs the price and the contract multiplier. Write the trade detail
#    down rather than assuming a typical one, because the assumed value is what the
#    whole cost estimate then rests on.
#
# 2. **Multiply volume by price only when the two describe the same transaction.**
#    Share volume times the traded close is dollars traded. Share volume times a
#    split- and dividend-adjusted close is not, and a volume that counts price
#    updates rather than trades has no dollar equivalent at all. Check what the
#    price series in hand actually is before treating turnover as a number.
#
# 3. **Ask which cost component dominates at the size you intend to trade, not in
#    general.** Commission and spread are flat per dollar; impact grows with order
#    size, so the answer flips somewhere. Solve for the order size at which impact
#    overtakes the larger flat component and check which side of it you are on.
#
# 4. **Turn a per-trade cost into an annual hurdle by multiplying it by turnover.**
#    Trading capital over several times a year multiplies a cost that looked small
#    per trade into a return the strategy has to earn before it earns anything.
#
# 5. **Charge borrow separately from trading cost.** It accrues with time held and
#    only on the short side, so it does not scale with turnover and cannot be folded
#    into a per-trade figure.
#
# ### Known limitations
#
# - The six panels do not cover the same dates. The dollar-turnover comparison
#   therefore mixes observation windows, and a market that looks more active may
#   simply have been observed over a busier period.
# - The fee table records rates as published on one date. Schedules change, and the
#   tier a given account qualifies for changes what it is actually charged.
# - The scenario parameters in Sections 3, 5 and 6 are round figures, not estimates
#   fitted to these datasets. They fix the shape of the answer, not its magnitude.
# - The impact model assumes an order is worked over roughly a day at a constant
#   rate. An order executed faster pays more than the square-root law says, which is
#   what `05_almgren_chriss_optimal_execution` makes explicit.
#
# **Next**: `02_spread_estimation` estimates the spread from daily bars, and
# `03_market_impact_calibration` fits the impact coefficient used above.
#
# **Book**: Chapter 18, Sections 18.1 and 18.2.
