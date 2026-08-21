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
# # Backtrader and Zipline on Current Case-Study Strategies
#
# This notebook reports the Backtrader and Zipline Reloaded rows from the current real-strategy
# audit. It does not infer asset support from ML4T's configurability: a pair is included only when
# the external engine and frozen bundle can express the same native contract.
#
# **Learning objectives**
#
# - Compare Backtrader and Zipline against ML4T on supported real strategies
# - Apply a monetary comparison unit to account values without weakening fill comparison
# - Interpret the measured engine-only runtime boundary
# - Use synthetic stress evidence as a secondary conformance result
#
# **Book reference**: Chapter 16, Section 16.3

# %% [markdown]
# ## Setup

# %%
"""Current Backtrader and Zipline parity evidence."""

import json

import polars as pl
from IPython.display import display

from utils.paths import get_chapter_dir

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides after this cell
ROUND_SECONDS = 3

# %%
AUDIT_PATH = get_chapter_dir(16) / "resources" / "framework_parity_audit.json"
audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
FRAMEWORKS = ["backtrader", "zipline"]
FRAMEWORK_NAMES = {
    key: f"{audit['frameworks'][key]['display_name']} {audit['frameworks'][key]['version']}"
    for key in FRAMEWORKS
}
CASE_NAMES = {
    "etfs": "ETF allocation",
    "cme_futures": "CME futures",
    "crypto_perps_funding": "Crypto perpetual funding",
    "fx_pairs": "FX allocation (USD-quoted pairs)",
}

# %% [markdown]
# ## 1. Required comparisons

# %% tags=["results"]
results = (
    pl.DataFrame(audit["real_strategy_records"])
    .filter(pl.col("framework").is_in(FRAMEWORKS))
    .with_columns(
        pl.col("case_study").replace_strict(CASE_NAMES).alias("strategy"),
        pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"),
    )
    .select(
        "strategy",
        "engine",
        "status",
        "fills",
        "valuations",
        "valuation_timestamps_match",
        "equity_gap",
        "equity_raw_gap",
        "terminal_gap",
        "terminal_raw_gap",
    )
    .sort("strategy", "engine")
)

assert results.height == 4
assert results.filter(pl.col("status") == "pass").height == 4
assert results["valuation_timestamps_match"].all()

display(results)

# %% [markdown]
# Backtrader and Zipline both participate in the ETF comparison. Backtrader also participates in the
# CME and USD-quoted foreign-exchange comparisons. The fill stream is compared at eight-decimal
# precision, while account values must round to the same cent. Zipline has no required CME or spot-FX
# row because the frozen inputs do not map to its native asset models.

# %% [markdown]
# ## 2. Unsupported asset models

# %% tags=["results"]
unsupported = (
    pl.DataFrame(audit["unsupported_records"])
    .filter(pl.col("framework").is_in(FRAMEWORKS))
    .with_columns(
        pl.col("case_study").replace_strict(CASE_NAMES).alias("strategy"),
        pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"),
    )
    .select("strategy", "engine", "reason")
    .sort("strategy", "engine")
)
display(unsupported)

# %% [markdown]
# Neither engine is credited with crypto-perpetual funding support it does not natively provide.
# Unsupported rows are excluded from the pass denominator.

# %% [markdown]
# ## 3. Engine-only timing
#
# Timing is retained only for correctness-passing rows and covers the engine call only.

# %% tags=["results"]
timing = (
    pl.DataFrame(audit["performance_records"])
    .filter(pl.col("framework").is_in(FRAMEWORKS))
    .with_columns(
        pl.col("case_study").replace_strict(CASE_NAMES).alias("strategy"),
        pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"),
        pl.col("framework_median_seconds").round(ROUND_SECONDS).alias("external_seconds"),
        pl.col("ml4t_median_seconds").round(ROUND_SECONDS).alias("ml4t_seconds"),
        pl.col("framework_to_ml4t_ratio").round(2).alias("external_div_ml4t"),
    )
    .select("strategy", "engine", "external_seconds", "ml4t_seconds", "external_div_ml4t")
)

assert timing.height == 4
display(timing)

# %% [markdown]
# The timer excludes data and adapter preparation. The ratios should not be applied to other
# strategies or machines.

# %% [markdown]
# ## 4. Synthetic stress evidence

# %% tags=["results"]
stress = (
    pl.DataFrame(audit["synthetic_stress"]["records"])
    .filter(pl.col("framework").is_in(FRAMEWORKS))
    .with_columns(pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"))
    .select("engine", "intents", "fills", "trades", "terminal_value", "status")
)
display(stress)

# %% [markdown]
# Both synthetic stress rows pass against their matching ML4T profiles. Their terminal values differ
# because the profiles reproduce different framework conventions. The test is pairwise: ML4T versus
# Backtrader and ML4T versus Zipline. The workload tests scale and event conventions, while the
# required rows above determine the real-data comparison.
