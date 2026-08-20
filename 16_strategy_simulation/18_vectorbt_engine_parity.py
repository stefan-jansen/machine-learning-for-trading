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
# # VectorBT Pro and OSS on Current Case-Study Strategies
#
# This notebook reports the VectorBT Pro and VectorBT OSS rows from the current real-strategy audit.
# The current ETF strategy and all required synthetic scenarios match exactly. The remaining
# real-strategy failure is the CME equity path under VectorBT Pro.
#
# **Learning objectives**
#
# - Compare VectorBT Pro and OSS with ML4T on the supported ETF strategy
# - Distinguish exact fill parity from a small equity-path residual
# - Understand why VectorBT OSS is not used for the CME futures contract
# - Read engine-only runtime evidence without treating it as a universal ranking
#
# **Book reference**: Chapter 16, Section 16.3

# %% [markdown]
# ## Setup

# %%
"""Current VectorBT parity evidence."""

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
FRAMEWORKS = ["vectorbt_pro", "vectorbt_oss"]
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
# Both VectorBT editions reproduce the ETF strategy exactly across 2,466 fills and 1,995 valuations.
# VectorBT Pro reproduces all 3,545 CME fills, but the maximum equity gap is `0.00000010` and the
# terminal gap is `0.00000007`; the row therefore fails the `1e-8` gate.

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
# VectorBT OSS does not provide the native multiplier and margin-account model required by the CME
# bundle. Neither edition is used to emulate crypto-perpetual funding and margin accounting.

# %% [markdown]
# ## 3. Engine-only timing
#
# Only the two correctness-passing ETF rows are timed for publication.

# %%
timing = (
    pl.DataFrame(audit["performance_records"])
    .filter(pl.col("framework").is_in(FRAMEWORKS))
    .with_columns(
        pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"),
        pl.col("framework_median_seconds").round(ROUND_SECONDS).alias("vectorbt_seconds"),
        pl.col("ml4t_median_seconds").round(ROUND_SECONDS).alias("ml4t_seconds"),
        pl.col("framework_to_ml4t_ratio").round(2).alias("vectorbt_div_ml4t"),
    )
    .select("engine", "vectorbt_seconds", "ml4t_seconds", "vectorbt_div_ml4t")
)

assert timing.height == 2
display(timing)

# %% [markdown]
# Both VectorBT editions have lower median engine-call time on the ETF workload. The measured region
# excludes data loading, target construction, adapter preparation, and output extraction. The result
# does not establish the same ratio for other datasets or strategy mechanics.

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
# VectorBT Pro and OSS both pass the 250-asset, 1.26-million-bar synthetic stress comparison against
# their matching ML4T profiles. This establishes scale conformance for the fixed target-order recipe.
# It does not overturn the CME real-strategy failure or create support for the excluded asset models.
