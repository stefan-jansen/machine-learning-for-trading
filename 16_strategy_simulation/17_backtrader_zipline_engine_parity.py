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
# - See why exact fills do not by themselves establish full equity-path parity
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
}

# %% [markdown]
# ## 1. Required comparisons

# %%
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
        "terminal_gap",
    )
    .sort("strategy", "engine")
)

assert results.height == 3
assert results.filter(pl.col("status") == "pass").height == 2
assert results["valuation_timestamps_match"].all()

display(results)

# %% [markdown]
# Backtrader and Zipline both match the ETF strategy exactly. Backtrader also reproduces every CME
# fill, but its equity and terminal value differ by `0.00000015`, so that row fails the exact gate.
# Zipline has no required CME row because the frozen continuous-root input lacks a native dated
# contract chain and roll map.

# %% [markdown]
# ## 2. Unsupported asset models

# %%
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
# Timing is retained only for correctness-passing rows. The Backtrader CME failure is therefore not
# timed for publication.

# %%
timing = (
    pl.DataFrame(audit["performance_records"])
    .filter(pl.col("framework").is_in(FRAMEWORKS))
    .with_columns(
        pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"),
        pl.col("framework_median_seconds").round(ROUND_SECONDS).alias("external_seconds"),
        pl.col("ml4t_median_seconds").round(ROUND_SECONDS).alias("ml4t_seconds"),
        pl.col("framework_to_ml4t_ratio").round(2).alias("external_div_ml4t"),
    )
    .select("engine", "external_seconds", "ml4t_seconds", "external_div_ml4t")
)

assert timing.height == 2
display(timing)

# %% [markdown]
# On the ETF strategy, ML4T's median engine call is lower than Backtrader's and Zipline's. The timer
# excludes data and adapter preparation, and these ratios should not be applied to other strategies.

# %% [markdown]
# ## 4. Synthetic stress evidence

# %%
stress = (
    pl.DataFrame(audit["synthetic_stress"]["records"])
    .filter(pl.col("framework").is_in(FRAMEWORKS))
    .with_columns(pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"))
    .select("engine", "intents", "fills", "trades", "terminal_value", "status")
)
display(stress)

# %% [markdown]
# Both synthetic stress rows pass against their matching ML4T profiles. Their terminal values differ
# from each other because the profiles reproduce different framework conventions. The test is pairwise:
# ML4T versus Backtrader and ML4T versus Zipline, not Backtrader versus Zipline. The workload tests
# scale and event conventions, while the ETF and CME rows above determine the real-strategy claim.
