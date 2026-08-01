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
# # Case-Study LEAN Parity: Real Book Artifacts
#
# **Docker image**: `ml4t`
#
# The canonical benchmark in `15_lean_engine_parity` proves that
# `ml4t-backtest[lean]` reproduces QuantConnect LEAN on the library's 250-asset,
# 20-year daily validation harness. This notebook inspects a cached case-study
# parity report and provides an optional live path for testing whether that
# calibrated profile still matches LEAN on real case-study weights and prices.
# The cached report does not preserve the original raw fill surfaces, run log,
# or hash manifest, so its default reader path is descriptive rather than an
# independent transfer proof.
#
# **Learning Objectives**:
# - Interpret the metrics recorded by a cached case-study parity report
# - Compare `ml4t-backtest[lean]` directly against live LEAN on the optional live path
# - Distinguish a reported aggregate result from independently preserved execution evidence
# - Identify the raw fills and identities a reproducible parity attestation must retain
#
# **Book Reference**: Chapter 16, Section 16.3 (Vectorized and Event-Driven Backtesting)
#
# **Prerequisites**: `07_engine_divergence_anatomy` and `15_lean_engine_parity`

# %% [markdown]
# ## Setup

# %%
"""Case-study LEAN parity on real book artifacts."""

import json
import shutil
import time
from pathlib import Path

# %%
import matplotlib.pyplot as plt
import ml4t.backtest as ml4t_backtest_pkg
import polars as pl

from case_studies.utils.analytics import SHORT_NAMES
from utils import ML4T_DATA_PATH
from utils.paths import get_chapter_dir, get_output_dir
from utils.style import COLORS

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides after this cell
RUN_LIVE = False
CASE_STUDIES = "etfs,sp500_equity_option_analytics,us_equities_panel"
MAX_CASE_STUDIES = 0  # 0 = all requested

# %%
OUTPUT_DIR = get_output_dir(16, "case_study_lean_parity")
RESULTS_PATH = OUTPUT_DIR / "case_study_lean_parity.csv"
CACHED_ARTIFACT_PATH = get_chapter_dir(16) / "resources" / "case_study_lean_parity_results.json"
LIVE_ARTIFACT_PATH = OUTPUT_DIR / "case_study_lean_parity_live.json"

# %% [markdown]
# ## 1. Resolve the Validation Context
#
# The live rerun path depends on the sibling `ml4t-backtest` repository because
# the heavy LEAN orchestration lives in its internal validation layer, not in
# the public runtime API.


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


BACKTEST_REPO = resolve_backtest_repo()
LEAN_WORKSPACE = BACKTEST_REPO / "validation" / "lean" / "workspace" if BACKTEST_REPO else None
LEAN_CONFIG = LEAN_WORKSPACE / "lean.json" if LEAN_WORKSPACE else None

print(f"Backtest repo:  {BACKTEST_REPO or 'not found (cached artifact only)'}")
print(f"LEAN config:    {LEAN_CONFIG or 'not found'}")
print(f"ML4T_DATA_PATH: {ML4T_DATA_PATH}")

# %% [markdown]
# ### Check live-run readiness
#
# The committed cached report should always work. The live path requires the
# LEAN CLI, Docker, a local LEAN workspace, and the case-study artifact root.


# %%
def check_live_prerequisites() -> pl.DataFrame:
    """Return a checklist for the optional live case-study rerun."""
    rows = [
        {
            "requirement": "ml4t-backtest repo",
            "ready": BACKTEST_REPO is not None and BACKTEST_REPO.exists(),
            "detail": BACKTEST_REPO.as_posix() if BACKTEST_REPO else "not found",
        },
        {
            "requirement": "docker",
            "ready": shutil.which("docker") is not None,
            "detail": shutil.which("docker") or "missing",
        },
        {
            "requirement": "lean or uvx",
            "ready": shutil.which("lean") is not None or shutil.which("uvx") is not None,
            "detail": shutil.which("lean") or shutil.which("uvx") or "missing",
        },
        {
            "requirement": "lean workspace config",
            "ready": LEAN_CONFIG is not None and LEAN_CONFIG.exists(),
            "detail": LEAN_CONFIG.as_posix() if LEAN_CONFIG else "not found",
        },
        {
            "requirement": "case-study data root",
            "ready": Path(ML4T_DATA_PATH).exists(),
            "detail": str(ML4T_DATA_PATH),
        },
    ]
    return pl.DataFrame(rows)


prereq_df = check_live_prerequisites()
prereq_df

# %% [markdown]
# ## 2. Load the Cached Parity Report
#
# The default path reads a committed report summarizing a parity run across
# three daily case studies that the report identifies as comparisons with LEAN:
#
# - `etfs`
# - `sp500_equity_option_analytics`
# - `us_equities_panel`
#
# The `RUN_LIVE = True` path below reproduces the same comparison locally: it
# runs each case study through actual LEAN and through `ml4t-backtest[lean]` and
# rebuilds the artifact.
#
# The report identifies its comparison surface as the **sorted daily fill multiset**
# `(timestamp, asset, side, quantity, 4-decimal price)`. Raw trade-log row
# order can differ when two engines emit same-day fills in different asset
# iteration order. Because the original fills, run log, and hash manifest were
# not preserved with this artifact, the cached path cannot independently verify
# either the reported fill identity or its producer lineage.


# %%
def load_artifact(path: Path) -> dict:
    """Load a cached or live case-study parity artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


payload = load_artifact(CACHED_ARTIFACT_PATH)
{
    "cached_report_artifact_source": payload["artifact_source"],
    "cached_report_comparison_surface": payload["comparison_surface"],
}

# %% [markdown]
# ## 3. Optional Live Rerun
#
# When `RUN_LIVE = True`, the notebook reproduces the comparison locally for each
# case study from its self-contained LEAN workspace project:
#
# - **LEAN** runs through actual QuantConnect LEAN (Docker) on the committed
#   `chapter16_<case_study>` workspace.
# - **`ml4t-backtest[lean]`** replays the identical target-weight strategy on the
#   same prices via `ml4t.backtest._validation.case_study_lean`.
#
# Both engines consume the workspace's own daily price data, so any difference is
# execution semantics rather than inputs. A fresh notebook-local artifact is saved.


# %%
def parse_case_studies(case_studies: str, max_case_studies: int) -> list[str]:
    """Parse a comma-separated case-study parameter."""
    requested = [item.strip() for item in case_studies.split(",") if item.strip()]
    if max_case_studies > 0:
        requested = requested[:max_case_studies]
    return requested


# Each case study maps to a self-contained LEAN workspace project and the
# forward-return horizon its target weights encode.
CASE_STUDY_PROJECTS = {
    "etfs": ("chapter16_etfs", "fwd_ret_21d"),
    "sp500_equity_option_analytics": ("chapter16_sp500_equity_option_analytics", "fwd_ret_5d"),
    "us_equities_panel": ("chapter16_us_equities_panel", "fwd_ret_5d"),
}


# %% [markdown]
# ### Run One Case Study Pair
#
# Run actual LEAN (Docker) and `ml4t-backtest[lean]` on one case study's committed
# workspace, then compare the daily fill multiset and terminal portfolio value via
# the library's `case_study_lean` validation module.


# %%
def _load_case_study_lean():
    """Import the case-study LEAN parity module.

    Prefer the installed `ml4t.backtest` package; if it predates the
    `case_study_lean` module, load it from the sibling ml4t-backtest source (the
    live path already requires that repo). Relative imports resolve against the
    installed `_validation` package, so the module's engine/config siblings are used.
    """
    try:
        from ml4t.backtest._validation import case_study_lean

        return case_study_lean
    except ImportError:
        import importlib.util
        import sys

        src = BACKTEST_REPO / "src" / "ml4t" / "backtest" / "_validation" / "case_study_lean.py"
        if not src.exists():
            raise ModuleNotFoundError(
                "case_study_lean is unavailable: update ml4t-backtest or provide the "
                "sibling repository so the live rerun can import it."
            ) from None
        spec = importlib.util.spec_from_file_location(
            "ml4t.backtest._validation.case_study_lean", src
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


# %% [markdown]
# Execute one LEAN and ml4t-backtest parity pair for a selected case study.


# %%
def run_case_study_pair(case_study: str) -> dict:
    """Run actual LEAN and ml4t[lean] on one case study's committed workspace."""
    csl = _load_case_study_lean()
    compare, lean_side = csl.compare, csl.lean_side
    parse_workspace_params, run_ml4t_lean = csl.parse_workspace_params, csl.run_ml4t_lean
    from ml4t.backtest._validation.lean_runner import (
        make_lean_env,
        resolve_lean_command,
        run_lean_backtest,
    )

    project, label = CASE_STUDY_PROJECTS[case_study]
    project_dir = LEAN_WORKSPACE / project
    data_daily = LEAN_WORKSPACE / "data" / "equity" / "usa" / "daily"

    lean_start = time.perf_counter()
    run_lean_backtest(
        lean_cmd=resolve_lean_command(),
        cwd=LEAN_WORKSPACE,
        project_dir=project_dir,
        lean_config=LEAN_CONFIG,
        output_dir=OUTPUT_DIR / f"lean_{case_study}",
        env=make_lean_env(),
    )
    lean_runtime = time.perf_counter() - lean_start
    lean = lean_side(project_dir)

    ml4t_start = time.perf_counter()
    ml4t = run_ml4t_lean(project_dir, data_daily)
    ml4t_runtime = time.perf_counter() - ml4t_start

    params = parse_workspace_params(project_dir)
    row = compare(lean, ml4t)
    row.update(
        {
            "case_study": case_study,
            "display": SHORT_NAMES.get(case_study, case_study),
            "label": label,
            "cost_bps": round(params["fee"] * 10_000, 4),
            "lean_runtime_sec": lean_runtime,
            "ml4t_runtime_sec": ml4t_runtime,
        }
    )
    return row


# %% [markdown]
# ### Build Live Payload
#
# Assemble individual case-study results into the notebook artifact.


# %%
PARITY_SOURCES = [
    "ml4t-backtest/src/ml4t/backtest/_validation/case_study_lean.py",
    "ml4t-backtest/src/ml4t/backtest/_validation/lean_runner.py",
    "ml4t-backtest/validation/lean/workspace/chapter16_<case_study>/",
    "third_edition/code/16_strategy_simulation/16_case_study_lean_parity.py",
]


# %%
def build_live_payload(rows: list[dict], skipped: list[tuple[str, str]]) -> dict:
    """Build the notebook payload from live case-study reruns."""
    if not rows:
        raise RuntimeError("No live LEAN parity rows were produced.")

    max_abs_gap = max(abs(row["final_value_gap_usd"]) for row in rows)
    return {
        "artifact_source": "live case-study LEAN rerun from notebook",
        "cached": False,
        "comparison_surface": "sorted daily fill multiset (timestamp, asset, side, quantity, 4-decimal price)",
        "limitations": [
            "Raw trade-log row order can differ because same-day fills are emitted in different asset iteration order.",
            "Parity is asserted on the decoded daily fill multiset plus terminal portfolio value, not on callback row order.",
        ],
        "results": rows,
        "summary": {
            "n_case_studies": len(rows),
            "matched_case_studies": sum(1 for row in rows if row["sorted_fill_multiset_match"]),
            "all_fill_multisets_match": all(row["sorted_fill_multiset_match"] for row in rows),
            "max_abs_final_value_gap_usd": max_abs_gap,
        },
        "skipped": [{"case_study": case_study, "reason": reason} for case_study, reason in skipped],
        "sources": PARITY_SOURCES,
    }


selected_case_studies = parse_case_studies(CASE_STUDIES, MAX_CASE_STUDIES)
if RUN_LIVE:
    skipped: list[tuple[str, str]] = []
    live_rows: list[dict] = []
    for case_study in selected_case_studies:
        try:
            live_rows.append(run_case_study_pair(case_study))
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            skipped.append((case_study, str(exc)))

    if live_rows:
        payload = build_live_payload(live_rows, skipped)
        LIVE_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        LIVE_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved live artifact: {LIVE_ARTIFACT_PATH}")
    else:
        print("Live rerun failed for all requested case studies; using cached artifact instead.")

# %% [markdown]
# ## 4. Inspect the Case-Study Results
#
# The table below reports:
#
# - terminal portfolio value in LEAN and `ml4t[lean]`
# - the dollar gap between the two
# - fill counts on both sides
# - whether the sorted fill multiset matches exactly
# - whether the raw row order also matches

# %%
results_df = pl.DataFrame(payload["results"]).with_columns(
    pl.col("case_study")
    .replace({key: value for key, value in SHORT_NAMES.items()})
    .alias("display")
)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.write_csv(RESULTS_PATH)

print(f"Saved results: {RESULTS_PATH}")

results_df.select(
    "display",
    "label",
    "cost_bps",
    "lean_final_value",
    "ml4t_final_value",
    "final_value_gap_usd",
    "lean_fills",
    "ml4t_fills",
    "sorted_fill_multiset_match",
    "raw_row_order_match",
)

# %% [markdown]
# ### Summary
#
# The cached report records exact agreement on the sorted fill multiset for all
# three case studies, with terminal-value differences at float noise. These are
# claims carried by the report, not independently recoverable evidence. A live
# rerun with `RUN_LIVE = True` performs a new comparison locally.

# %%
summary_df = pl.DataFrame([payload["summary"]])
summary_df

# %% [markdown]
# ## 5. Visual Comparison
#
# The first panel visualizes the fill counts recorded by the cached report. The
# second visualizes its recorded terminal-value differences, which are near
# machine precision relative to the reported portfolio values.

# %%
plot_df = results_df.sort("display")
labels = plot_df["display"].to_list()
y = range(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")

axes[0].barh(
    [i + 0.18 for i in y],
    plot_df["lean_fills"].to_list(),
    height=0.35,
    color=COLORS["blue"],
    label="LEAN",
)
axes[0].barh(
    [i - 0.18 for i in y],
    plot_df["ml4t_fills"].to_list(),
    height=0.35,
    color=COLORS["amber"],
    label="ml4t[lean]",
)
axes[0].set(yticks=list(y), yticklabels=labels, title="Fill Counts", xlabel="fills")
axes[0].legend()

_value_gaps = plot_df["final_value_gap_usd"].to_list()
axes[1].barh(labels, _value_gaps, color=COLORS["positive"])
axes[1].set(title="Terminal Value Gap", xlabel="ml4t[lean] - LEAN (USD)")
axes[1].axvline(0, color="black", linewidth=0.8, alpha=0.4)
if all(abs(v) < 1e-6 for v in _value_gaps):
    axes[1].text(
        0.5,
        0.5,
        "Gap = $0 across all scenarios",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["slate"],
        alpha=0.85,
    )

fig.suptitle("Metrics Recorded by the Cached Case-Study Parity Report", y=1.02)
fig.show()

# %% [markdown]
# ## 6. Interpretation
#
# The default reader path visualizes a cached report; it does not establish that
# the LEAN-matching profile transfers beyond the preserved synthetic benchmark.
# Establishing that result requires a live rerun that retains both raw fill
# surfaces, the execution log, environment identity, and hashes of every input
# and output used by the comparison.

# %%
matched = int(payload["summary"]["matched_case_studies"])
total = int(payload["summary"]["n_case_studies"])
max_gap = float(payload["summary"]["max_abs_final_value_gap_usd"])
raw_row_matches = int(results_df["raw_row_order_match"].sum())

print(f"Reported sorted fill multiset matches: {matched}/{total}")
print(f"Reported raw row-order matches:        {raw_row_matches}/{total}")
print(f"Reported max absolute value gap:       ${max_gap:.10f}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **The cached path is a report, not a transfer proof.** It records a comparison between LEAN and `ml4t-backtest[lean]`, but the original raw fills and producer identities are unavailable for independent verification.
#
# 2. **Three fill-surface matches are reported.** The artifact records exact sorted-fill agreement for ETFs, S&P 500 equity-option analytics, and the US equities panel; it does not carry the underlying fill surfaces needed to rederive that statement.
#
# 3. **Terminal-value gaps in the report are at floating-point scale.** The largest recorded absolute gap is about one ten-millionth of a dollar, subject to the same evidence limitation.
#
# 4. **A fresh attestation must preserve both representations.** Raw row order may be a logging detail, but a verifier still needs both raw logs and the normalized fill multiset to establish that conclusion rather than assume it.

# %% [markdown]
# ---
# *Notebook: 16_case_study_lean_parity*
# *ML4T 3rd Edition - Chapter 16: Strategy Simulation*
