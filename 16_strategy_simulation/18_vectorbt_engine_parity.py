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
# # Same money, different trades
#
# **Docker image**: `ml4t`
#
# ## Purpose
# The other two parity notebooks in this chapter compare event-driven engines, and both reach
# agreement on the trade count and on the terminal value. This one compares a vectorized engine,
# and the result is split: the two accounts finish on the same dollar to fourteen decimal places
# and they get there through a slightly different number of trades.
#
# That combination is worth a notebook on its own, because it is the case where the two obvious
# summary statistics disagree about whether the engines match. Which of them is the right answer
# depends on what the comparison is for, and the notebook is about making that choice explicitly
# rather than picking whichever number is more flattering.
#
# ## Learning objectives
#
# - Read a parity result where the economic outcome matches and the order decomposition does not,
#   and say which surface matters for a given purpose.
# - Recognise that a lower trade count is not automatically better or worse, and what would have to
#   be measured to know.
# - Check an artifact against the same contract its sibling notebooks check, rather than trusting
#   whatever the file happens to contain.
#
# ## Book reference
# Chapter 16, Section 16.3 (vectorized and event-driven backtesting).
#
# ## Prerequisites
#
# - `07_engine_divergence_anatomy`, for what makes two engines differ.
# - `17_backtrader_zipline_engine_parity`, which reaches agreement on both surfaces and so gives
#   this notebook something to contrast against.

# %% [markdown]
# ## Setup

# %%
"""Canonical VectorBT OSS parity benchmark."""

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import ml4t.backtest as ml4t_backtest_pkg
import polars as pl

from utils.paths import get_chapter_dir, get_output_dir

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides after this cell
RUN_LIVE = False
SCENARIO_ID = "multi_250_20yr"
REAL_DATA_PATH = ""
ARTIFACT_SCENARIO_LABEL = "250 assets, 20 years daily"

# %%
OUTPUT_DIR = get_output_dir(16, "vectorbt_engine_parity")
CACHED_ARTIFACT_PATH = get_chapter_dir(16) / "resources" / "vectorbt_parity_results.json"
LIVE_ARTIFACT_PATH = OUTPUT_DIR / "vectorbt_parity_results_live.json"

# %% [markdown]
# ## 1. What produced the numbers
#
# This notebook uses the same validation source of truth as the other external
# engine notebooks: `ml4t-backtest/validation/benchmark_suite.py`.
#
# The benchmark is the canonical target-share daily ranking strategy on real
# market data:
#
# - 250 US equities
# - 20 years of daily bars
# - top-25 / bottom-25 long-short portfolio
# - `ml4t.backtest[vectorbt_strict]` compared against VectorBT OSS


# %%
def resolve_backtest_repo() -> Path | None:
    """Find a checkout carrying the validation harness, or None if there is not one.

    `ML4T_BACKTEST_REPO` names it explicitly; failing that the installed package's own
    ancestry is searched, which finds it in an editable install. Nothing guesses at a path
    under the home directory: a resolver that knows where a repository lives on the author's
    machine fails silently and differently on everybody else's.
    """
    candidates: list[Path] = []
    override = os.environ.get("ML4T_BACKTEST_REPO")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(Path(ml4t_backtest_pkg.__file__).resolve().parents)
    return next(
        (c for c in candidates if (c / "validation" / "benchmark_suite.py").is_file()), None
    )


# %% [markdown]
# `find_real_data_path` resolves the parquet used by the VectorBT comparison
# benchmark, checking the explicit override then a fixed list of fallback
# locations before giving up.


# %%
def find_real_data_path(explicit_path: str) -> Path | None:
    """Locate the daily equity parquet the benchmark reads, via the notebook parameter or
    the canonical data root. There is no third place to look."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    data_root = os.environ.get("ML4T_DATA_PATH")
    if data_root:
        candidates.append(
            Path(data_root) / "equities" / "market" / "us_equities" / "us_equities.parquet"
        )
    return next((path for path in candidates if path.is_file()), None)


# %%
BACKTEST_REPO = resolve_backtest_repo()
BENCHMARK_SUITE = BACKTEST_REPO / "validation" / "benchmark_suite.py" if BACKTEST_REPO else None
REAL_DATA_FILE = find_real_data_path(REAL_DATA_PATH)

print(f"Validation harness: {'found' if BACKTEST_REPO else 'not found (cached artifact only)'}")
print(f"Benchmark suite:    {BENCHMARK_SUITE.name if BENCHMARK_SUITE else 'not found'}")
print(f"Real data:          {REAL_DATA_FILE.name if REAL_DATA_FILE else 'not found'}")

# %% [markdown]
# ## 2. What a live rerun would need
#
# The cached artifact path should always work. A live rerun requires:
#
# - the benchmark harness
# - the real-data parquet
# - `vectorbt`
#
# Unlike LEAN, VectorBT OSS does not need an external workspace or Docker layer.


# %%
def check_live_prerequisites() -> pl.DataFrame:
    """Return a checklist for the optional live VectorBT rerun."""
    rows = [
        {
            "requirement": "benchmark_suite.py",
            "ready": BENCHMARK_SUITE is not None and BENCHMARK_SUITE.exists(),
            "detail": BENCHMARK_SUITE.name if BENCHMARK_SUITE else "set ML4T_BACKTEST_REPO",
        },
        {
            "requirement": "real daily parquet",
            "ready": REAL_DATA_FILE is not None,
            "detail": REAL_DATA_FILE.name if REAL_DATA_FILE else "set ML4T_DATA_PATH",
        },
        {
            "requirement": "vectorbt",
            "ready": importlib.util.find_spec("vectorbt") is not None,
            "detail": "importable" if importlib.util.find_spec("vectorbt") else "missing",
        },
    ]
    return pl.DataFrame(rows)


prereq_df = check_live_prerequisites()
prereq_df

# %% [markdown]
# ## 3. Load the artifact and check what it can support
#
# The committed artifact records a benchmark run. As in the sibling notebooks, reading a recorded
# result is weaker than reproducing one, and what can still be done is check the file against a
# contract: that it describes the scenario it claims, and that its summary reproduces from its own
# primitives.


# %%
def load_parity_artifact(path: Path) -> dict:
    """Load an artifact and check everything it can be checked on from inside itself.

    The same contract the other parity notebooks in this chapter apply: the file names the
    scenario it claims, carries one row for the engine pair it claims, and every derived
    figure in that row reproduces from the primitives beside it. A summary that was edited,
    or that came from different primitives, fails the last check.
    """
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("scenario_id") != SCENARIO_ID:
        raise ValueError(
            f"artifact scenario is {artifact.get('scenario_id')!r}, expected {SCENARIO_ID!r}"
        )
    if artifact.get("scenario_label") != ARTIFACT_SCENARIO_LABEL:
        raise ValueError(f"artifact scenario label is {artifact.get('scenario_label')!r}")
    if artifact.get("data_source") != "real":
        raise ValueError("this comparison is only meaningful on real market data")
    results = artifact.get("results", [])
    if len(results) != 1 or results[0].get("engine_id") != "vectorbt":
        raise ValueError("artifact must contain exactly one VectorBT row")

    result = results[0]
    for field in ("ml4t_num_trades", "reference_num_trades"):
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in (
        "ml4t_final_value",
        "reference_final_value",
        "ml4t_runtime_sec",
        "reference_runtime_sec",
    ):
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be positive and finite")

    ml4t_trades, reference_trades = result["ml4t_num_trades"], result["reference_num_trades"]
    value_gap = abs(result["ml4t_final_value"] - result["reference_final_value"])
    derived = {
        "trade_gap": ml4t_trades - reference_trades,
        "trade_gap_pct": (ml4t_trades - reference_trades) / reference_trades,
        "final_value_gap_abs": value_gap,
        "final_value_gap_pct": value_gap / abs(result["reference_final_value"]),
        "runtime_speedup": result["reference_runtime_sec"] / result["ml4t_runtime_sec"],
    }
    for field, expected in derived.items():
        if not math.isclose(
            float(result.get(field, math.nan)), expected, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"{field} does not reproduce from the primitives in its own row")
    return artifact


payload = load_parity_artifact(CACHED_ARTIFACT_PATH)
print(f"Artifact:   {CACHED_ARTIFACT_PATH.name}")
print(f"Scenario:   {payload['scenario_label']}")
print("Contents:   one engine pair, identity and derived figures verified")

# %% [markdown]
# ## 4. Reproducing it here instead
#
# When `RUN_LIVE = True`, the notebook reruns the canonical benchmark for:
#
# - `ml4t-vbt-strict`
# - `vbt-oss`
#
# The benchmark calls use `--cache-mode refresh` so the real-data cache is
# regenerated under the current Python/NumPy environment if necessary.


# %%
def run_framework_benchmark(
    framework: str, scenario_id: str, output_path: Path, real_data_path: Path
) -> dict:
    """Run one benchmark-suite framework and return its single result row."""
    cmd = [
        sys.executable,
        str(BENCHMARK_SUITE),
        "--framework",
        framework,
        "--scenario",
        scenario_id,
        "--data-source",
        "real",
        "--real-data-path",
        str(real_data_path),
        "--cache-mode",
        "refresh",
        "--output-json",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(BACKTEST_REPO))
    report = json.loads(output_path.read_text(encoding="utf-8"))
    return report["results"][0]


# %% [markdown]
# `build_live_payload` takes the two live benchmark results, one from ml4t-backtest under its
# strict VectorBT profile and one from VectorBT itself, and derives the trade gap, the value gap
# and the runtime ratio. All three are stored as fractions and displayed in basis points, so the
# artifact and the table cannot drift apart in units.


# %%
def build_result_row(ml4t_result: dict, vbt_result: dict) -> dict:
    """Build the VectorBT parity row from two live benchmark results."""
    reference_trades = max(int(vbt_result["num_trades"]), 1)
    reference_value = max(abs(float(vbt_result["final_value"])), 1.0)
    runtime_speedup = (
        float(vbt_result["runtime_sec"]) / float(ml4t_result["runtime_sec"])
        if ml4t_result["runtime_sec"] > 0
        else None
    )
    final_value_gap = float(abs(ml4t_result["final_value"] - vbt_result["final_value"]))
    return {
        "engine_id": "vectorbt",
        "engine_label": "VectorBT OSS",
        "profile": "vectorbt_strict",
        "status": "production",
        "ml4t_framework_id": "ml4t-vbt-strict",
        "reference_framework_id": "vbt-oss",
        "ml4t_num_trades": int(ml4t_result["num_trades"]),
        "reference_num_trades": int(vbt_result["num_trades"]),
        "trade_gap": int(ml4t_result["num_trades"] - vbt_result["num_trades"]),
        "trade_gap_pct": float(
            (ml4t_result["num_trades"] - vbt_result["num_trades"]) / reference_trades
        ),
        "ml4t_final_value": float(ml4t_result["final_value"]),
        "reference_final_value": float(vbt_result["final_value"]),
        "final_value_gap_abs": final_value_gap,
        "final_value_gap_pct": float(final_value_gap / reference_value),
        "ml4t_runtime_sec": float(ml4t_result["runtime_sec"]),
        "reference_runtime_sec": float(vbt_result["runtime_sec"]),
        "runtime_speedup": runtime_speedup,
    }


# %% [markdown]
# Wrap the parity row with the live-run provenance and scope limitations.


# %%
def build_live_payload(ml4t_result: dict, vbt_result: dict, scenario_id: str) -> dict:
    """Build the notebook payload from two live benchmark results."""
    return {
        "artifact_source": "live benchmark_suite.py rerun",
        "scenario_id": scenario_id,
        "scenario_label": ARTIFACT_SCENARIO_LABEL,
        "data_source": "real",
        "cached": False,
        "limitations": [
            "These results cover the canonical target-share benchmark, not the Chapter 16 case-study adapters.",
            "VectorBT OSS matches terminal value to floating-point noise on the current benchmark surface.",
            "A small residual trade gap remains, so this is economic parity rather than exact order decomposition parity.",
        ],
        "results": [build_result_row(ml4t_result, vbt_result)],
        "sources": [
            "ml4t-backtest/validation/METHODOLOGY.md",
            "third_edition/code/16_strategy_simulation/18_vectorbt_engine_parity.py",
        ],
    }


ready_for_live = bool(prereq_df["ready"].all())
if RUN_LIVE:
    if not ready_for_live or REAL_DATA_FILE is None:
        raise RuntimeError(
            "Live rerun requested with missing prerequisites: "
            + ", ".join(prereq_df.filter(~pl.col("ready"))["requirement"].to_list())
        )
    ml4t_result = run_framework_benchmark(
        "ml4t-vbt-strict", SCENARIO_ID, OUTPUT_DIR / "ml4t_vbt_strict_live.json", REAL_DATA_FILE
    )
    vbt_result = run_framework_benchmark(
        "vbt-oss", SCENARIO_ID, OUTPUT_DIR / "vbt_oss_live.json", REAL_DATA_FILE
    )
    payload = build_live_payload(ml4t_result, vbt_result, SCENARIO_ID)
    LIVE_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved a live artifact: {LIVE_ARTIFACT_PATH.name}")
else:
    print("Reading the committed artifact; no live benchmark was requested.")

# %% [markdown]
# ## 5. The comparison
#
# Three columns are worth reading together, because they do not all point the same way: the trade
# gap is small and not zero, the value gap is zero to the precision a float can express, and the
# speedup is below one, meaning the vectorized engine finished this fixture first.


# %%
comparison_df = (
    pl.DataFrame(payload["results"])
    .with_columns(
        (pl.col("trade_gap_pct") * 10_000).round(3).alias("trade_gap_bps"),
        (pl.col("final_value_gap_pct") * 10_000).round(6).alias("value_gap_bps"),
        pl.col("runtime_speedup").round(2),
    )
    .select(
        "engine_label",
        "profile",
        "status",
        "ml4t_num_trades",
        "reference_num_trades",
        "trade_gap",
        "trade_gap_bps",
        "final_value_gap_abs",
        "value_gap_bps",
        "runtime_speedup",
    )
)
comparison_df

# %% [markdown]
# ## 6. What the residual is and is not
#
# The two accounts end within a fifteenth decimal place of each other, and they took a different
# number of trades to get there. Nothing about that is contradictory. Two engines can hold the same
# positions on the same days and disagree about how many orders it took, if one of them bundles
# adjacent moves in the same instrument and the other does not, or if one emits a zero-quantity
# order the other suppresses.
#
# What the trade-count difference does *not* tell you is which engine is right, and it is worth
# being clear about why. A lower count could mean orders were bundled, which is a reporting
# difference and costs nothing. It could equally mean fills were missed, which is an execution
# difference and costs something. Distinguishing the two needs the fills, and this artifact does
# not carry them - so the honest statement is that the accounts agree economically and their order
# decompositions differ by an amount nobody here has attributed.
#
# The runtime is the other asymmetry, and it points the other way from the event-driven notebooks.
# A vectorized engine on a benchmark built out of dense array operations is playing to its
# strength. That is a statement about this fixture, not about either engine in general.

# %%
row = payload["results"][0]
pl.DataFrame(
    {
        "surface": [
            "terminal value",
            "trade count",
            "runtime",
        ],
        "result": [
            "agrees to float64 precision",
            f"{abs(row['trade_gap'])} of {row['reference_num_trades']:,} trades differ",
            "the vectorized engine is faster on this fixture",
        ],
        "what_it_supports": [
            "The two accounts held the same positions and paid the same costs.",
            "The order decompositions differ. Attributing that needs the fills, which this "
            "artifact does not carry.",
            "Nothing about either engine beyond this benchmark, whose shape favours dense "
            "array operations.",
        ],
    }
)

# %% [markdown]
# ## Key takeaways
#
# 1. **Two summary statistics can disagree about whether engines match.** Terminal value says yes
#    to fourteen decimal places; trade count says no. Neither is wrong, and which one the
#    comparison is *for* has to be decided before either is quoted.
# 2. **A trade-count difference is not self-interpreting.** Fewer trades can mean bundled orders,
#    which costs nothing, or missed fills, which costs something. The artifact records the count
#    and not the fills, so the difference here is measured and not explained.
# 3. **Economic parity is the weaker claim and often the sufficient one.** If the question is
#    whether a strategy's returns can be reproduced, matching the account is what matters. If the
#    question is whether an execution model is faithful, the order decomposition is what matters.
# 4. **Runtime comparisons are about the fixture.** A benchmark built from dense array operations
#    flatters a vectorized engine, and this one does. Reading a speed ratio from it as a general
#    property of either engine is reading the fixture, not the engines.
#
# ### Known limitations
#
# - One scenario, one asset class, one frequency, and a fixture whose shape is unusually
#   favourable to vectorization.
# - The artifact carries conclusions and not fills, so the trade-count residual cannot be
#   attributed here to bundling, to suppression, or to a real difference in what filled.
# - The default path reads a recorded result. Reproducing it needs the validation harness and
#   `vectorbt` installed, which is what `RUN_LIVE` is for.
#
# **Next:** [`15_lean_engine_parity`](15_lean_engine_parity.ipynb) and
# [`17_backtrader_zipline_engine_parity`](17_backtrader_zipline_engine_parity.ipynb) show what
# agreement on both surfaces looks like, which is the contrast this notebook is against.
