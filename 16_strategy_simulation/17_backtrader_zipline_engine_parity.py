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
# # Canonical Engine Parity: Backtrader and Zipline Reloaded
#
# **Docker image**: `ml4t`
#
# Chapter 16 has already shown two complementary parity stories:
#
# - `15_lean_engine_parity`: the canonical external-engine benchmark for LEAN
# - `16_case_study_lean_parity`: the transfer of that calibrated LEAN profile to
#   real chapter case-study artifacts
#
# This notebook returns to the **canonical benchmark** and shows the same
# reader-facing evidence for the other two event-driven engines that matter in
# the historical backtesting ecosystem:
#
# - Backtrader
# - Zipline Reloaded
#
# **Learning Objectives**:
# - Inspect the canonical 250-asset, 20-year parity benchmark for Backtrader and Zipline
# - Compare trade-gap, value-gap, and speedup metrics against `ml4t-backtest`
# - Distinguish float-noise parity (Backtrader) from exact trade-count parity with a small dollar residual (Zipline)
# - Re-run the benchmark locally when the optional comparison frameworks are installed
#
# **Book Reference**: Chapter 16, Section 16.3 (Vectorized and Event-Driven Backtesting)
#
# **Prerequisites**: `07_engine_divergence_anatomy` and `15_lean_engine_parity`

# %% [markdown]
# ## Setup

# %%
"""Canonical Backtrader and Zipline parity benchmark."""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

# Keep engine warnings visible: they are part of the parity evidence.
# %%
import matplotlib.pyplot as plt
import ml4t.backtest as ml4t_backtest_pkg
import polars as pl

from utils import ML4T_DATA_PATH
from utils.paths import get_chapter_dir, get_output_dir
from utils.style import COLORS

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides after this cell
RUN_LIVE = False
SCENARIO_ID = "multi_250_20yr"
REAL_DATA_PATH = ""

HARNESS_SCENARIO_LABEL = "Multi-asset (250×20yr daily)"
EXPECTED_IDENTITIES = {
    "zipline": ("Zipline Reloaded", "zipline_strict", "ml4t-zipline-strict", "zipline"),
    "backtrader": (
        "Backtrader",
        "backtrader_strict",
        "ml4t-backtrader-strict",
        "backtrader",
    ),
}
CERTIFIED_COMMON_PROVENANCE = {
    "harness_commit": "459abd81f2f30dc70cf38a40da7591af3da2d02a",
    "harness_source_sha256": "4a4bd3815e1c1db50e0f5f476c72814e57e111c50d0e427734cf7f62c8e072c6",
    "data_sha256": "bb82ee2eac24521544f78c418f86d9073ae4d54b56570eaf838c6e65c6621b90",
    "ordered_universe_sha256": "49e54af392831a61b8c1a4c81b8caa44721aa29891b9219fb339d53cfd72298d",
}
CERTIFIED_RAW_REPORT_HASHES = {
    "ml4t.backtest[zipline_strict]": "54183791cce15c995dd915f54ec65d95ea6f44dc148db3f960935e72e5969cd5",
    "Zipline": "4e415cf52ffa4772ac9070e7dd49db6dd6b5295ec7de7cff6825d5f7c456a07e",
    "ml4t.backtest[backtrader_strict]": "cc8e781e926fdd60c880c8ad15f6429bca9ddaf3c4f3b37c9a952d53c3e35f1c",
    "Backtrader": "73a05e7996ae05a818e56a8b71b85c6bf2b7e56ae9f8c0fbccbab8fe61ad0940",
}

# %%
OUTPUT_DIR = get_output_dir(16, "backtrader_zipline_engine_parity")
CACHED_ARTIFACT_PATH = get_chapter_dir(16) / "resources" / "backtrader_zipline_parity_results.json"
LIVE_ARTIFACT_PATH = OUTPUT_DIR / "backtrader_zipline_parity_live.json"

# %% [markdown]
# ## 1. Source of Truth
#
# These comparisons use the same source of truth as the LEAN notebook: the
# library validation harness in `ml4t-backtest/validation/benchmark_suite.py`.
# The benchmark is a fixed engine-only target-share fixture on real market
# data. It compares execution semantics rather than estimating an investable
# strategy return:
#
# - a deterministic retrospective cohort of 250 US equities
# - 20 years of daily bars
# - top-25 / bottom-25 cross-sectional ranking portfolio
# - one external engine at a time against the matching `ml4t-backtest` profile
#
# The source parquet lacks the same market-wide session (2017-11-08) for every
# selected asset. The harness fills that interior gap from the prior session;
# its backward-fill step supplies zero values on this fixed cohort.


# %%
def resolve_backtest_repo() -> Path | None:
    """Resolve the sibling ml4t-backtest repository root.

    Returns None when the validation harness is not available (e.g. CI),
    which disables the live rerun path but lets the cached artifact work.
    """
    package_path = Path(ml4t_backtest_pkg.__file__).resolve()
    for parent in package_path.parents:
        if (parent / "validation" / "benchmark_suite.py").exists():
            return parent

    fallback = Path.home() / "ml4t" / "libraries" / "ml4t-backtest"
    if (fallback / "validation" / "benchmark_suite.py").exists():
        return fallback

    return None


# %% [markdown]
# `find_real_data_path` resolves the parquet that backs the real-data run for
# every framework, checking environment variables and the standard repo
# layouts before giving up.


# %%
def find_real_data_path(explicit_path: str) -> Path | None:
    """Locate the real daily equity parquet used by the benchmark harness.

    The benchmark reads the historical US-equities parquet and selects a fixed
    retrospective cohort for an engine-only comparison. It does not claim a
    point-in-time or survivorship-bias-free investment universe.
    """
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    candidates.append(
        Path(ML4T_DATA_PATH) / "equities" / "market" / "us_equities" / "us_equities.parquet"
    )

    for path in candidates:
        if path.exists():
            return path
    return None


# %%
BACKTEST_REPO = resolve_backtest_repo()
BENCHMARK_SUITE = BACKTEST_REPO / "validation" / "benchmark_suite.py" if BACKTEST_REPO else None
REAL_DATA_FILE = find_real_data_path(REAL_DATA_PATH)

print(f"Backtest repo:   {BACKTEST_REPO or 'not found (cached artifact only)'}")
print(f"Benchmark suite: {BENCHMARK_SUITE or 'not found'}")
print(f"Real data:       {REAL_DATA_FILE if REAL_DATA_FILE else 'not found'}")

# %% [markdown]
# ## 2. Runtime Prerequisites
#
# The cached artifact should always work. A live rerun additionally requires:
#
# - the real-data parquet
# - `backtrader`
# - `zipline-reloaded`
#
# The notebook checks only the Python packages because the benchmark harness
# manages the rest of the local orchestration.


# %%
def check_live_prerequisites() -> pl.DataFrame:
    """Return a checklist for the optional live parity rerun."""
    rows = [
        {
            "requirement": "benchmark_suite.py",
            "ready": BENCHMARK_SUITE is not None and BENCHMARK_SUITE.exists(),
            "detail": "available" if BENCHMARK_SUITE else "not found",
        },
        {
            "requirement": "real daily parquet",
            "ready": REAL_DATA_FILE is not None,
            "detail": REAL_DATA_FILE.name if REAL_DATA_FILE else "not found",
        },
        {
            "requirement": "backtrader",
            "ready": importlib.util.find_spec("backtrader") is not None,
            "detail": "importable" if importlib.util.find_spec("backtrader") else "missing",
        },
        {
            "requirement": "zipline-reloaded",
            "ready": importlib.util.find_spec("zipline") is not None,
            "detail": "importable" if importlib.util.find_spec("zipline") else "missing",
        },
    ]
    return pl.DataFrame(rows)


prereq_df = check_live_prerequisites()
prereq_df

# %% [markdown]
# ## 3. Load the Cached Parity Snapshot
#
# The committed artifact below records a cache-off benchmark run. Its
# provenance binds the harness, data file, ordered cohort, raw reports, and
# engine versions:
#
# - **Backtrader** now matches the canonical benchmark at trade count and final
#   value to floating-point noise.
# - **Zipline Reloaded** reaches exact trade-count parity and keeps only a
#   small terminal-value residual on the current benchmark surface.


# %%
def require_finite(value: object, field: str, *, positive: bool = False) -> float:
    """Return a validated finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{field} is outside its valid range")
    return number


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# %%
def validate_pair_result(row: dict) -> None:
    """Fail closed on identity, status, primitives, and derived arithmetic."""
    engine_id = row.get("engine_id")
    if engine_id not in EXPECTED_IDENTITIES:
        raise ValueError("unexpected engine identity")
    expected = EXPECTED_IDENTITIES[engine_id]
    keys = ("engine_label", "profile", "ml4t_framework_id", "reference_framework_id")
    if tuple(row.get(key) for key in keys) != expected or row.get("status") != "done":
        raise ValueError("engine identity, profile, or status is invalid")
    counts = [row.get("ml4t_num_trades"), row.get("reference_num_trades")]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts):
        raise TypeError("trade counts must be non-negative integers")
    ml4t_value = require_finite(row.get("ml4t_final_value"), "ml4t_final_value", positive=True)
    ref_value = require_finite(
        row.get("reference_final_value"), "reference_final_value", positive=True
    )
    ml4t_time = require_finite(row.get("ml4t_runtime_sec"), "ml4t_runtime_sec", positive=True)
    ref_time = require_finite(
        row.get("reference_runtime_sec"), "reference_runtime_sec", positive=True
    )
    expected_metrics = {
        "trade_gap": counts[0] - counts[1],
        "trade_gap_pct": (counts[0] - counts[1]) / max(counts[1], 1),
        "final_value_gap_abs": abs(ml4t_value - ref_value),
        "final_value_gap_pct": abs(ml4t_value - ref_value) / max(abs(ref_value), 1.0),
        "runtime_speedup": ref_time / ml4t_time,
    }
    for field, expected_value in expected_metrics.items():
        actual_value = require_finite(row.get(field), field)
        if not math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{field} is inconsistent with primitive values")


# %%
def validate_provenance(artifact: dict, *, certified_reports: bool) -> None:
    """Validate the cache-off run and its exact data and report lineage."""
    provenance = artifact.get("provenance", {})
    if provenance.get("benchmark_cache_mode") != "off":
        raise ValueError("benchmark artifact must come from a cache-off run")
    for field, expected in CERTIFIED_COMMON_PROVENANCE.items():
        if provenance.get(field) != expected:
            raise ValueError(f"{field} does not match the certified fixture")
    universe = provenance.get("ordered_universe")
    if not isinstance(universe, list) or len(universe) != 250 or len(set(universe)) != 250:
        raise ValueError("ordered 250-asset cohort is required")
    encoded = json.dumps(universe, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != provenance.get("ordered_universe_sha256"):
        raise ValueError("ordered cohort hash does not match")
    selection = provenance.get("selection_proof", {})
    missing_sessions = selection.get("missing_market_sessions", [])
    expected_forward = len(missing_sessions) * 250 * 5
    if (
        selection.get("observed_session_rows_per_asset") != 5039
        or not missing_sessions
        or selection.get("forward_filled_values") != expected_forward
        or selection.get("backward_filled_values") != 0
    ):
        raise ValueError("cohort fill proof is incomplete or permits future filling")
    raw = provenance.get("raw_reports", [])
    if len(raw) != 4 or len({item.get("framework") for item in raw}) != 4:
        raise ValueError("four unique raw benchmark reports are required")
    if any(len(item.get("sha256", "")) != 64 for item in raw):
        raise ValueError("raw benchmark report hashes are required")
    if (
        certified_reports
        and {item["framework"]: item["sha256"] for item in raw} != CERTIFIED_RAW_REPORT_HASHES
    ):
        raise ValueError("raw reports do not match the certified fixture")
    versions = provenance.get("versions", {})
    required_versions = ("python", "ml4t-backtest", "zipline-reloaded", "backtrader")
    if not all(isinstance(versions.get(k), str) and versions[k] for k in required_versions):
        raise ValueError("engine and package versions are required")


# %%
def load_parity_artifact(path: Path) -> dict:
    """Load and validate the committed parity artifact."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("scenario_id") != SCENARIO_ID:
        raise ValueError("artifact scenario does not match requested scenario")
    if artifact.get("scenario_label") != HARNESS_SCENARIO_LABEL:
        raise ValueError("artifact scenario label is not canonical")
    if artifact.get("data_source") != "real" or artifact.get("cached") is not True:
        raise ValueError("committed artifact identity is incomplete")
    results = artifact.get("results", [])
    expected_engines = set(EXPECTED_IDENTITIES)
    if len(results) != 2 or {row.get("engine_id") for row in results} != expected_engines:
        raise ValueError("artifact must contain exactly one row per engine pair")
    validate_provenance(artifact, certified_reports=True)
    for row in results:
        validate_pair_result(row)
    return artifact


payload = load_parity_artifact(CACHED_ARTIFACT_PATH)
payload["artifact_source"], payload["scenario_label"]

# %% [markdown]
# ## 4. Optional Live Rerun
#
# When `RUN_LIVE = True`, the notebook replays the canonical benchmark for two
# framework pairs:
#
# - `ml4t-backtest[zipline_strict]` vs Zipline Reloaded
# - `ml4t-backtest[backtrader_strict]` vs Backtrader
#
# Each framework is executed through the library benchmark harness and then
# merged into the same notebook-friendly payload shape as the cached artifact.

# %%
ENGINE_PAIRS = [
    {
        "engine_id": "zipline",
        "engine_label": "Zipline Reloaded",
        "profile": "zipline_strict",
        "ml4t_framework": "ml4t-zipline-strict",
        "reference_framework": "zipline",
        "status_label": "done",
    },
    {
        "engine_id": "backtrader",
        "engine_label": "Backtrader",
        "profile": "backtrader_strict",
        "ml4t_framework": "ml4t-backtrader-strict",
        "reference_framework": "backtrader",
        "status_label": "done",
    },
]


# %%
def run_framework_benchmark(
    framework: str, scenario_id: str, output_path: Path, real_data_path: Path
) -> tuple[dict, dict]:
    """Run one framework and return its result plus report lineage."""
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
        "off",
        "--output-json",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(BACKTEST_REPO))
    report = json.loads(output_path.read_text(encoding="utf-8"))
    lineage = {
        "file": output_path.name,
        "framework": report["results"][0]["framework"],
        "timestamp": report["meta"]["timestamp"],
        "sha256": sha256_file(output_path),
    }
    return report["results"][0], lineage


# %% [markdown]
# `build_live_payload` assembles the cached-vs-live comparison payload from
# the rows returned by `run_framework_benchmark`, computing trade-gap and
# value-gap metrics per (ml4t, reference-engine) pair.


# %%
def validate_raw_result(result: dict, framework: str) -> None:
    """Validate one row returned by the benchmark harness."""
    if result.get("framework", "").lower() != framework.lower():
        raise ValueError("benchmark framework identity mismatch")
    if result.get("scenario") != HARNESS_SCENARIO_LABEL or result.get("error") is not None:
        raise ValueError("benchmark scenario or completion status mismatch")
    count = result.get("num_trades")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TypeError("num_trades must be a non-negative integer")
    require_finite(result.get("final_value"), "final_value", positive=True)
    require_finite(result.get("runtime_sec"), "runtime_sec", positive=True)


# %%
def build_pair_result(pair: dict, ml4t_result: dict, ref_result: dict) -> dict:
    validate_raw_result(ml4t_result, f"ml4t.backtest[{pair['profile']}]")
    validate_raw_result(ref_result, pair["reference_framework"])
    reference_trades = max(ref_result["num_trades"], 1)
    reference_value = max(abs(float(ref_result["final_value"])), 1.0)
    trade_gap = ml4t_result["num_trades"] - ref_result["num_trades"]
    value_gap = abs(ml4t_result["final_value"] - ref_result["final_value"])
    return {
        "engine_id": pair["engine_id"],
        "engine_label": pair["engine_label"],
        "profile": pair["profile"],
        "status": pair["status_label"],
        "ml4t_framework_id": pair["ml4t_framework"],
        "reference_framework_id": pair["reference_framework"],
        "ml4t_num_trades": ml4t_result["num_trades"],
        "reference_num_trades": ref_result["num_trades"],
        "trade_gap": trade_gap,
        "trade_gap_pct": float(trade_gap / reference_trades),
        "ml4t_final_value": float(ml4t_result["final_value"]),
        "reference_final_value": float(ref_result["final_value"]),
        "final_value_gap_abs": float(value_gap),
        "final_value_gap_pct": float(value_gap / reference_value),
        "ml4t_runtime_sec": float(ml4t_result["runtime_sec"]),
        "reference_runtime_sec": float(ref_result["runtime_sec"]),
        "runtime_speedup": float(ref_result["runtime_sec"] / ml4t_result["runtime_sec"]),
    }


# %% [markdown]
# Assemble validated engine-pair rows into the live comparison payload.


# %%
def build_live_payload(rows: list[dict], reports: list[dict], scenario_id: str) -> dict:
    """Build the notebook payload from live benchmark rows."""
    expected_frameworks = {
        "ml4t.backtest[zipline_strict]",
        "Zipline",
        "ml4t.backtest[backtrader_strict]",
        "Backtrader",
    }
    if len(rows) != 4 or {row.get("framework") for row in rows} != expected_frameworks:
        raise ValueError("live rerun must return exactly four unique frameworks")
    if any(row.get("scenario") != HARNESS_SCENARIO_LABEL for row in rows):
        raise ValueError("live rows must all match the requested scenario")
    if BACKTEST_REPO is None or BENCHMARK_SUITE is None or REAL_DATA_FILE is None:
        raise RuntimeError("live benchmark provenance inputs are unavailable")
    harness_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=BACKTEST_REPO
    ).strip()
    common = {
        "harness_commit": harness_commit,
        "harness_source_sha256": sha256_file(BENCHMARK_SUITE),
        "data_sha256": sha256_file(REAL_DATA_FILE),
        "ordered_universe_sha256": payload["provenance"]["ordered_universe_sha256"],
    }
    if common != CERTIFIED_COMMON_PROVENANCE:
        raise ValueError("live rerun inputs do not match the certified fixture")
    results = []
    for pair in ENGINE_PAIRS:
        ml4t_result = next(
            row for row in rows if row["framework"] == f"ml4t.backtest[{pair['profile']}]"
        )
        ref_result = next(
            row for row in rows if row["framework"].lower() == pair["reference_framework"]
        )
        results.append(build_pair_result(pair, ml4t_result, ref_result))
    return {
        "artifact_source": "live benchmark_suite.py rerun",
        "scenario_id": scenario_id,
        "scenario_label": HARNESS_SCENARIO_LABEL,
        "data_source": "real",
        "cached": False,
        "limitations": [
            "This fixed retrospective cohort is an engine fixture, not an investable universe.",
            "The results cover target-share execution, not case-study weight adapters.",
            "Runtime ratios are environment-specific diagnostics, not portable performance guarantees.",
        ],
        "provenance": {
            **payload["provenance"],
            **common,
            "raw_reports": reports,
            "versions": {
                "python": platform.python_version(),
                "ml4t-backtest": importlib.metadata.version("ml4t-backtest"),
                "zipline-reloaded": importlib.metadata.version("zipline-reloaded"),
                "backtrader": importlib.metadata.version("backtrader"),
            },
        },
        "results": results,
    }


# %%
ready_for_live = bool(prereq_df["ready"].all())
if RUN_LIVE and ready_for_live and REAL_DATA_FILE is not None:
    live_rows = []
    live_reports = []
    for pair in ENGINE_PAIRS:
        for framework in (pair["ml4t_framework"], pair["reference_framework"]):
            result_path = OUTPUT_DIR / f"{framework}_{SCENARIO_ID}.json"
            row, report = run_framework_benchmark(
                framework, SCENARIO_ID, result_path, REAL_DATA_FILE
            )
            live_rows.append(row)
            live_reports.append(report)
    payload = build_live_payload(live_rows, live_reports, SCENARIO_ID)
    validate_provenance(payload, certified_reports=False)
    LIVE_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved live artifact: {LIVE_ARTIFACT_PATH}")
elif RUN_LIVE and not ready_for_live:
    print("Live rerun requested, but prerequisites are incomplete. Using cached artifact instead.")
else:
    print("Using cached parity artifact.")

# %% [markdown]
# ## 5. Build the Comparison Table
#
# The table summarizes the parity surface that matters on the canonical
# benchmark:
#
# - trade-gap percentage
# - final-value-gap percentage
# - runtime speedup of `ml4t-backtest` relative to the external engine

# %%
comparison_df = (
    pl.DataFrame(payload["results"])
    .with_columns(
        (pl.col("trade_gap_pct") * 10_000).round(3).alias("trade_gap_bps"),
        (pl.col("final_value_gap_pct") * 10_000).round(3).alias("value_gap_bps"),
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
# ### Gap summary
#
# Both engines are now operationally complete on the canonical benchmark. The
# difference is where the tiny residual remains: Backtrader is at float-noise
# parity, while Zipline keeps a small terminal-value gap with exact trade count.

# %%
for row in payload["results"]:
    print(f"{row['engine_label']}:")
    print(f"  Profile:             {row['profile']}")
    print(f"  Trade gap:           {row['trade_gap']:,} ({row['trade_gap_pct']:.4%})")
    print(
        f"  Final value gap:     ${row['final_value_gap_abs']:,.0f} ({row['final_value_gap_pct']:.4%})"
    )
    print(f"  Runtime speedup:     {row['runtime_speedup']:.2f}x")
    print(f"  Status:              {row['status']}")

# %% [markdown]
# ## 6. Visual Comparison
#
# The chart below shows the three headline parity quantities for the two
# external engines.

# %%
engine_labels = [row["engine_label"] for row in payload["results"]]
trade_gap_bps = [row["trade_gap_pct"] * 10_000 for row in payload["results"]]
value_gap_bps = [row["final_value_gap_pct"] * 10_000 for row in payload["results"]]
speedup = [row["runtime_speedup"] for row in payload["results"]]

fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
panels = [
    (trade_gap_bps, "Both engines match trade counts", "Trade gap (bps)"),
    (value_gap_bps, "Terminal values agree within 0.16 bps", "Value gap (bps)"),
    (speedup, "ml4t completes the fixture about 7x faster", "Runtime ratio (external / ml4t)"),
]
for ax, (values, title, ylabel) in zip(axes, panels, strict=True):
    ax.bar(engine_labels, values, color=[COLORS["amber"], COLORS["blue"]])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(0.0, color=COLORS["neutral"], linewidth=1.0)
    if all(abs(value) < 1e-6 for value in values):
        ax.text(0.5, 0.5, "Gap ≈ 0 bps", transform=ax.transAxes, ha="center", va="center")

fig.suptitle("A fixed real-data fixture isolates engine parity", y=1.04)
plt.show()

# %% [markdown]
# ## 7. Interpret the Results
#
# The key distinction is not between "vectorized" and "event-driven" in the
# abstract. It is whether the execution semantics are aligned closely enough to
# reproduce the same canonical target-share strategy.

# %%
interpretation_df = pl.DataFrame(
    {
        "engine": ["Zipline Reloaded", "Backtrader"],
        "parity_state": ["Exact trade-count parity", "Float-noise parity"],
        "reading": [
            "The benchmark has exact trade-count parity with a small remaining terminal-value gap.",
            "The benchmark matches at trade count and final value to floating-point noise.",
        ],
        "reader_takeaway": [
            "ml4t-backtest can reproduce Zipline's canonical daily ranking surface with exact trade-count parity.",
            "Backtrader parity is now operationally complete on the same canonical benchmark.",
        ],
    }
)
interpretation_df

# %% [markdown]
# ## Key Takeaways
#
# 1. **A fixed real-data fixture is a useful execution trust anchor.** It uses
#    the same large-scale target-share harness as the library validation suite,
#    but its retrospective cohort is not an investment-performance claim.
#
# 2. **Zipline reaches exact trade-count parity on that benchmark.** The
#    residual dollar gap is small relative to the benchmark scale.
#
# 3. **Backtrader now matches to floating-point noise on the cached canonical
#    rerun.** The old one-trade residual in the stale artifact is gone.
#
# 4. **These comparisons complement, rather than replace, the LEAN notebooks.**
#    LEAN gets both a canonical benchmark notebook and a real case-study
#    transfer notebook because it is the primary modern external engine target.
#
# **Next**: [`16_case_study_lean_parity`](16_case_study_lean_parity.ipynb) shows
# how the calibrated LEAN profile transfers from the canonical benchmark to the
# real Chapter 16 case-study artifacts.

# %% [markdown]
# ---
# *Notebook: 17_backtrader_zipline_engine_parity*
# *ML4T 3rd Edition - Chapter 16: Strategy Simulation*
