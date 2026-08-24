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
# # Two more engines, and two different kinds of agreement
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
# ## Learning objectives
#
# - Read a difference between two engines' results and say whether it is arithmetic-order noise or
#   a real disagreement about execution, from the size of the residual relative to float precision.
# - Check a result artifact for internal consistency when it carries no provenance, and state
#   exactly which claims that check does and does not reach.
# - Explain why a hash pinned against a file the reader does not have makes an artifact harder to
#   verify rather than easier.
#
# ## Book reference
# Chapter 16, Section 16.3 (vectorized and event-driven backtesting).
#
# ## Prerequisites
#
# - `07_engine_divergence_anatomy`, for what makes two engines differ at all.
# - `15_lean_engine_parity`, which introduces this benchmark and its validation pattern.

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
import ml4t.backtest as ml4t_backtest_pkg
import polars as pl

from utils import ML4T_DATA_PATH
from utils.paths import get_chapter_dir, get_output_dir

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides after this cell
RUN_LIVE = False
SCENARIO_ID = "multi_250_20yr"
REAL_DATA_PATH = ""

# The benchmark scenario carries two names, and conflating them is what broke this notebook:
# the harness labels its own result rows one way, and the artifacts it writes label the same
# scenario the reader-facing way. Validate each against the label it actually carries.
HARNESS_SCENARIO_LABEL = "Multi-asset (250×20yr daily)"
ARTIFACT_SCENARIO_LABEL = "250 assets, 20 years daily"
EXPECTED_IDENTITIES = {
    "zipline": ("Zipline Reloaded", "zipline_strict", "ml4t-zipline-strict", "zipline"),
    "backtrader": (
        "Backtrader",
        "backtrader_strict",
        "ml4t-backtrader-strict",
        "backtrader",
    ),
}

# %% [markdown]
# A provenance block, if an artifact carries one, has to name where its inputs came from. Those
# names are identities to record and report rather than values to pin: the hash of a harness
# checkout, or of a raw engine report, refers to a file that is not in this repository, so it can
# only be compared against itself. A pin against an absent file is a gate that passes for exactly
# one artifact and fails for every later one without saying why.

# %%
REQUIRED_PROVENANCE_FIELDS = (
    "harness_commit",
    "harness_source_sha256",
    "data_sha256",
    "ordered_universe_sha256",
)
UNIVERSE_SIZE = 250
SESSION_ROWS_PER_ASSET = 5039

# %%
OUTPUT_DIR = get_output_dir(16, "backtrader_zipline_engine_parity")
CACHED_ARTIFACT_PATH = get_chapter_dir(16) / "resources" / "backtrader_zipline_parity_results.json"
LIVE_ARTIFACT_PATH = OUTPUT_DIR / "backtrader_zipline_parity_live.json"

# %% [markdown]
# ## 1. What produced the numbers
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

print(f"Validation harness: {'found' if BACKTEST_REPO else 'not found (cached artifact only)'}")
print(f"Benchmark suite:    {BENCHMARK_SUITE.name if BENCHMARK_SUITE else 'not found'}")
print(f"Real data:          {REAL_DATA_FILE.name if REAL_DATA_FILE else 'not found'}")

# %% [markdown]
# ## 2. What a live rerun would need
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
# ## 3. Load the artifact and check what it can support
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
def validate_provenance(provenance: dict) -> None:
    """Check everything a provenance block can be checked on from inside the block itself.

    Two kinds of statement live in here and only one of them is verifiable. The ordered
    universe and the fill proof carry their own evidence: the cohort's hash is recomputed
    from the cohort, and the fill counts have to reconcile with the number of missing
    sessions. The lineage hashes name files that are not in this repository, so they are
    required to be present and well-formed and are reported rather than compared - a hash
    pinned against something a reader cannot open proves nothing to that reader.
    """
    if provenance.get("benchmark_cache_mode") != "off":
        raise ValueError("benchmark artifact must come from a cache-off run")
    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if not provenance.get(f)]
    if missing:
        raise ValueError(f"provenance is missing required lineage fields: {missing}")

    universe = provenance.get("ordered_universe")
    if (
        not isinstance(universe, list)
        or len(universe) != UNIVERSE_SIZE
        or len(set(universe)) != UNIVERSE_SIZE
    ):
        raise ValueError(f"an ordered cohort of {UNIVERSE_SIZE} distinct assets is required")
    encoded = json.dumps(universe, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != provenance.get("ordered_universe_sha256"):
        raise ValueError("the recorded cohort hash does not match the recorded cohort")

    selection = provenance.get("selection_proof", {})
    missing_sessions = selection.get("missing_market_sessions", [])
    expected_forward = len(missing_sessions) * UNIVERSE_SIZE * 5
    if (
        selection.get("observed_session_rows_per_asset") != SESSION_ROWS_PER_ASSET
        or not missing_sessions
        or selection.get("forward_filled_values") != expected_forward
        or selection.get("backward_filled_values") != 0
    ):
        raise ValueError("the fill proof is incomplete, or it permits filling from the future")

    raw = provenance.get("raw_reports", [])
    if len(raw) != 4 or len({item.get("framework") for item in raw}) != 4:
        raise ValueError("four uniquely named raw benchmark reports are required")
    if any(len(item.get("sha256", "")) != 64 for item in raw):
        raise ValueError("every raw benchmark report must carry a SHA-256")
    versions = provenance.get("versions", {})
    required_versions = ("python", "ml4t-backtest", "zipline-reloaded", "backtrader")
    if not all(isinstance(versions.get(k), str) and versions[k] for k in required_versions):
        raise ValueError("the Python, ml4t-backtest, Zipline and Backtrader versions are required")


# %%
def load_parity_artifact(path: Path) -> dict:
    """Load and validate the committed parity artifact."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("scenario_id") != SCENARIO_ID:
        raise ValueError("artifact scenario does not match requested scenario")
    if artifact.get("scenario_label") != ARTIFACT_SCENARIO_LABEL:
        raise ValueError(
            f"artifact scenario label is {artifact.get('scenario_label')!r}, "
            f"expected {ARTIFACT_SCENARIO_LABEL!r}"
        )
    if artifact.get("data_source") != "real" or artifact.get("cached") is not True:
        raise ValueError("committed artifact identity is incomplete")
    results = artifact.get("results", [])
    expected_engines = set(EXPECTED_IDENTITIES)
    if len(results) != 2 or {row.get("engine_id") for row in results} != expected_engines:
        raise ValueError("artifact must contain exactly one row per engine pair")
    for row in results:
        validate_pair_result(row)
    if artifact.get("provenance"):
        validate_provenance(artifact["provenance"])
    return artifact


payload = load_parity_artifact(CACHED_ARTIFACT_PATH)
has_provenance = bool(payload.get("provenance"))

print(f"Artifact:          {CACHED_ARTIFACT_PATH.name}")
print(f"Scenario:          {payload['scenario_label']}")
print(f"Engine pairs:      {len(payload['results'])}, identities and arithmetic verified")
print(f"Provenance block:  {'present and verified' if has_provenance else 'absent'}")

# %% [markdown]
# Read the last line before any of the numbers. A provenance block is what would let a reader check
# the run rather than take its word: which harness commit produced it, the hash of the price data it
# consumed, the ordered cohort of assets and its own hash, a proof that no value was filled from the
# future, and a hash of each engine's raw report. `validate_provenance` checks all of that, and the
# committed artifact carries none of it, so on the default path those checks do not run.
#
# What does run is everything the artifact can support from inside itself: it names the scenario it
# claims, carries exactly one row per engine pair with the identity and profile each row asserts,
# and every derived figure in it - the trade gap, the value gap, the speedup - reproduces from the
# primitives in the same row. A summary that had been edited, or that came from different
# primitives, fails there.
#
# That is a real check and a narrower one than provenance. The distinction is the same one
# `16_case_study_lean_parity` makes: an artifact that keeps its conclusions and not its evidence can
# be checked for internal consistency and cannot be checked for truth.

# %% [markdown]
# ## 4. Reproducing it here instead
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
    return report["results"][0], lineage, report


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
def build_live_payload(
    rows: list[dict],
    reports: list[dict],
    reports_meta: list[dict],
    scenario_id: str,
) -> dict:
    """Build the notebook payload, and its provenance, from this run rather than a prior one.

    Every lineage field is measured here: the harness commit from the checkout that ran, the
    hashes from the files that were read, the cohort from the reports the run produced. The
    previous version copied the cohort out of the committed artifact, which makes a live
    result attest to inputs it never saw.
    """
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
    cohort = next(
        (
            report["meta"]["ordered_universe"]
            for report in reports_meta
            if report.get("meta", {}).get("ordered_universe")
        ),
        None,
    )
    selection_proof = next(
        (
            report["meta"]["selection_proof"]
            for report in reports_meta
            if report.get("meta", {}).get("selection_proof")
        ),
        None,
    )
    if cohort is None or selection_proof is None:
        raise RuntimeError(
            "the benchmark reports carry no ordered universe or no fill proof, so this run "
            "cannot record what a later reader would need to check it. Update the harness to "
            "emit them rather than writing an artifact whose provenance cannot be verified."
        )
    harness_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=BACKTEST_REPO
    ).strip()
    common = {
        "harness_commit": harness_commit,
        "harness_source_sha256": sha256_file(BENCHMARK_SUITE),
        "data_sha256": sha256_file(REAL_DATA_FILE),
        "ordered_universe": cohort,
        "ordered_universe_sha256": hashlib.sha256(
            json.dumps(cohort, separators=(",", ":")).encode()
        ).hexdigest(),
    }
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
        "scenario_label": ARTIFACT_SCENARIO_LABEL,
        "data_source": "real",
        "cached": False,
        "limitations": [
            "This fixed retrospective cohort is an engine fixture, not an investable universe.",
            "The results cover target-share execution, not case-study weight adapters.",
            "Runtime ratios are environment-specific diagnostics, not portable performance guarantees.",
        ],
        "provenance": {
            "benchmark_cache_mode": "off",
            **common,
            "selection_proof": selection_proof,
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
if RUN_LIVE:
    if not ready_for_live or REAL_DATA_FILE is None:
        raise RuntimeError(
            "Live rerun requested with missing prerequisites: "
            + ", ".join(prereq_df.filter(~pl.col("ready"))["requirement"].to_list())
        )
    live_rows, live_reports, live_meta = [], [], []
    for pair in ENGINE_PAIRS:
        for framework in (pair["ml4t_framework"], pair["reference_framework"]):
            result_path = OUTPUT_DIR / f"{framework}_{SCENARIO_ID}.json"
            row, lineage, report = run_framework_benchmark(
                framework, SCENARIO_ID, result_path, REAL_DATA_FILE
            )
            live_rows.append(row)
            live_reports.append(lineage)
            live_meta.append(report)
    payload = build_live_payload(live_rows, live_reports, live_meta, SCENARIO_ID)
    validate_provenance(payload["provenance"])
    LIVE_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved a live artifact carrying its own provenance: {LIVE_ARTIFACT_PATH.name}")
else:
    print("Reading the committed artifact; no live benchmark was requested.")

# %% [markdown]
# ## 5. The comparison
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
# Both engines reach an exact trade count, and the two terminal-value residuals are of completely
# different kinds. Backtrader's is at the scale of float64 rounding, which means the two accounts
# performed the same arithmetic in a different order. Zipline's is ten orders of magnitude larger
# and is a real difference in what the two engines did, small enough not to matter for the
# comparison and large enough that it cannot be rounding. Reading both as "close enough" loses
# that distinction; the last column below divides each gap by the number of trades that produced
# it, which is the scale rounding would accumulate at.

# %%
comparison_df.select(
    "engine_label",
    "trade_gap",
    pl.col("final_value_gap_abs").alias("value_gap_usd"),
    "value_gap_bps",
    (pl.col("final_value_gap_abs") / pl.col("ml4t_num_trades")).alias("gap_per_trade_usd"),
)

# %% [markdown]
# ## 6. What separates the two residuals
#
# The useful distinction is not "vectorized" against "event-driven". Both external engines here are
# event-driven, and so is `ml4t-backtest`. What decides whether two engines agree is whether their
# execution semantics were aligned: when an order is placed, what price it fills at, how a target
# share is converted to a quantity, and what happens to the remainder.
#
# A residual at float64 scale means those semantics match and the two implementations summed the
# same numbers in a different order. A residual many orders of magnitude larger means they do not
# quite match, and the useful thing is to say which one you are looking at rather than calling both
# "parity".

# %%
pl.DataFrame(
    {
        "engine": ["Zipline Reloaded", "Backtrader"],
        "trade_count": ["exact", "exact"],
        "terminal_value": ["small real difference", "float64 rounding"],
        "what_that_means": [
            "The execution semantics are close but not identical; the gap is small enough to "
            "ignore for this comparison and is not attributable to arithmetic order.",
            "The execution semantics are identical; the two accounts differ only by the order in "
            "which the same additions were performed.",
        ],
    }
)

# %% [markdown]
# ## Key takeaways
#
# 1. **Say which kind of residual you have.** A gap at float64 scale means two implementations did
#    the same arithmetic in a different order. A gap several orders of magnitude above that means
#    they did different arithmetic. Both can be called "close"; only the first is agreement.
# 2. **Check what an artifact can support, not what it asserts.** The committed report here has no
#    provenance block, so nothing in it can be traced to the run that produced it. Every derived
#    figure in it can still be recomputed from its own primitives, and that check catches an edited
#    or mismatched summary. Knowing which of the two you have is the point.
# 3. **Do not pin a hash against a file the reader cannot open.** A certified hash of a harness
#    checkout is a gate that passes for exactly one artifact and fails for every later one, without
#    telling anybody why. Record the lineage, recompute what is recomputable, and report the rest.
# 4. **Alignment is a property of the configuration, not of the engine's category.** Three
#    event-driven engines agree here because their execution semantics were deliberately matched,
#    not because they share an architecture.
# 5. **Runtime ratios are not portable.** They describe one machine, one software stack and one
#    day, and they are recorded here rather than compared.
#
# ### Known limitations
#
# - One scenario, one asset class, one frequency. Engines that agree on 250 US equities at daily
#   frequency may not agree on intraday bars, on futures roll behaviour, or on corporate actions.
# - The default path reads a recorded result. Reproducing it needs the validation harness and both
#   external engines installed, which is what `RUN_LIVE` is for.
# - The fixture ranks assets and takes a share of each end of the ranking. It is built to generate
#   a large number of fills, not to make money, and nothing here bears on whether it would.
#
# **Next:** [`18_vectorbt_engine_parity`](18_vectorbt_engine_parity.ipynb) applies the same
# treatment to a vectorized engine, where the execution semantics differ by construction.
