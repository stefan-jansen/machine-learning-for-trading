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
# # Real-Strategy Cross-Framework Audit
#
# This notebook reports the current framework comparison on three production-selected case-study
# strategies: ETF allocation, CME futures, and crypto perpetual futures with funding. Every engine
# in a required pair receives the same content-addressed market data and frozen model-derived
# targets. Unsupported pairs are disclosed instead of being approximated with a different asset or
# accounting model.
#
# The result is narrower than universal framework equivalence. Six of eight supported pairs pass.
# The two CME comparisons reproduce every fill but fail the equity and terminal-value checks at the
# audit's `1e-8` canonical quantum.
#
# **Learning objectives**
#
# - Read a parity result across fills, valuation timestamps, equity, and terminal value
# - Separate supported comparisons from asset models a framework does not provide
# - Interpret engine-only timings without generalizing beyond the measured workload and machine
# - Distinguish real-strategy evidence from synthetic convention and stress tests
#
# **Book reference**: Chapter 16, Section 16.3

# %% [markdown]
# ## Setup

# %%
"""Current real-strategy cross-framework audit."""

import json

import matplotlib.pyplot as plt
import polars as pl
from IPython.display import Markdown, display

from utils.paths import get_chapter_dir

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides after this cell
ROUND_SECONDS = 3

# %%
AUDIT_PATH = get_chapter_dir(16) / "resources" / "framework_parity_audit.json"
audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

assert audit["schema_version"] == 1
assert audit["scope"]["required_pairs"] == 8
assert audit["scope"]["unsupported_pairs"] == 7

FRAMEWORK_NAMES = {
    key: f"{value['display_name']} {value['version']}" for key, value in audit["frameworks"].items()
}
CASE_NAMES = {
    "etfs": "ETF allocation",
    "cme_futures": "CME futures",
    "crypto_perps_funding": "Crypto perpetual funding",
}

display(
    Markdown(
        f"**Evidence date:** {audit['audit_generated_at'][:10]}  \n"
        f"**Library evidence commit:** `{audit['library_commit'][:12]}`"
    )
)

# %% [markdown]
# ## 1. What is compared
#
# Model fitting and target construction happen before either engine runs. The same frozen target
# table is identified by its input-bundle hash on both sides of a comparison. This audit therefore
# tests backtest execution, not whether two modeling pipelines happen to produce similar signals.
#
# A pass requires all of the following:
#
# - the complete sorted fill stream matches on timestamp, asset, side, quantity, price, and commission;
# - the engines expose the same valuation timestamp set;
# - every valuation and the terminal value have zero gap after `1e-8` quantization; and
# - a negative control that changes the first fill price by exactly `1e-8` is detected.
#
# "Exact" does not mean bit-identical floating-point state.

# %%
bundle_table = (
    pl.DataFrame(audit["real_strategy_records"])
    .select("case_study", "input_bundle_sha256")
    .unique()
    .with_columns(
        pl.col("case_study").replace_strict(CASE_NAMES).alias("strategy"),
        pl.col("input_bundle_sha256").str.slice(0, 12).alias("bundle_sha256_prefix"),
    )
    .select("strategy", "bundle_sha256_prefix")
    .sort("strategy")
)
display(bundle_table)

# %% [markdown]
# The bundle hash covers the prepared market data, frozen targets, strategy specification, and any
# contract or funding inputs required by the case study.

# %% [markdown]
# ## 2. Current correctness result

# %%
results = (
    pl.DataFrame(audit["real_strategy_records"])
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
        "negative_control_detected",
    )
    .sort("strategy", "engine")
)

passing = results.filter(pl.col("status") == "pass").height
assert passing == audit["scope"]["passing_pairs"] == 6
assert results["valuation_timestamps_match"].all()
assert results["negative_control_detected"].all()

display(results)

# %%
display(
    Markdown(
        f"**Result:** {passing}/{results.height} supported pairs pass. The broad real-strategy "
        "equivalence gate remains **failed** because both CME rows exceed the exact comparison "
        "quantum on equity and terminal value."
    )
)

# %% [markdown]
# All five ETF engines match ML4T Backtest across thousands of fills and 1,995 valuations. LEAN also
# matches the crypto-perpetual strategy across 8,408 fills and 2,426 hourly valuations, including
# native funding settlements. The CME comparisons have identical fill streams and timestamp
# coverage, but their maximum equity gaps are `0.00000010` for VectorBT Pro and `0.00000015` for
# Backtrader. Those rows fail because the audit does not replace an exact gate with an economic
# tolerance.

# %% [markdown]
# ## 3. Unsupported pairs
#
# A comparison is required only when the external engine and the frozen input can express the asset
# contract without substituting different semantics. For example, the current CME bundle contains
# continuous root series but no dated contract chain or roll map, so it is not a valid LEAN or
# Zipline futures input.

# %%
unsupported = (
    pl.DataFrame(audit["unsupported_records"])
    .with_columns(
        pl.col("case_study").replace_strict(CASE_NAMES).alias("strategy"),
        pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"),
    )
    .select("strategy", "engine", "reason")
    .sort("strategy", "engine")
)
display(unsupported)

# %% [markdown]
# These rows are not failures and do not count as passes. They define where this audit has no valid
# comparison.

# %% [markdown]
# ## 4. Engine-only runtime
#
# Timing is reported only for correctness-passing pairs. Each row uses one warmup and ten measured,
# process-isolated runs. The timed region is the engine call. It excludes data loading, model
# inference, target construction, adapter preparation, output extraction, serialization, and
# reporting.

# %%
performance = (
    pl.DataFrame(audit["performance_records"])
    .with_columns(
        pl.col("case_study").replace_strict(CASE_NAMES).alias("strategy"),
        pl.col("framework").replace_strict(FRAMEWORK_NAMES).alias("engine"),
    )
    .with_columns(
        pl.col("framework_median_seconds").round(ROUND_SECONDS).alias("external_seconds"),
        pl.col("ml4t_median_seconds").round(ROUND_SECONDS).alias("ml4t_seconds"),
        pl.col("framework_to_ml4t_ratio").round(2).alias("external_div_ml4t"),
    )
    .select(
        "strategy",
        "engine",
        "external_seconds",
        "ml4t_seconds",
        "external_div_ml4t",
    )
)
display(performance)

# %%
plot_data = performance.to_pandas()
labels = [f"{row.strategy}\n{row.engine}" for row in plot_data.itertuples()]
y = list(range(len(plot_data)))
height = 0.36

fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
ax.barh(
    [value + height / 2 for value in y], plot_data["external_seconds"], height, label="External"
)
ax.barh([value - height / 2 for value in y], plot_data["ml4t_seconds"], height, label="ML4T")
ax.set_yticks(y, labels)
ax.set_xscale("log")
ax.set_xlabel("Median engine-call seconds (log scale)")
ax.set_title("Measured runtime for correctness-passing pairs")
ax.legend()
ax.grid(axis="x", alpha=0.25)
plt.show()

# %% [markdown]
# VectorBT is faster on the ETF workload. ML4T is faster than Backtrader, Zipline, and LEAN on the
# measured rows. These are dated case-and-machine measurements, not stable framework-wide speed
# rankings.

# %% [markdown]
# ## 5. What the evidence supports
#
# The current evidence supports exact ETF equivalence for all five engines and exact
# crypto-perpetual equivalence for LEAN under the named profiles and frozen inputs. It does not
# support a universal equivalence claim because the CME equity paths still fail, and it says nothing
# about unsupported asset-framework combinations. The separate synthetic scenario and 250-asset
# stress suites test convention coverage and scale after this real-strategy result; they do not
# replace it.
