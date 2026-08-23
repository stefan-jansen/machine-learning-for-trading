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
# # Does the parity hold on the book's own strategies?
#
# **Docker image**: `ml4t`
#
# ## Purpose
# `15_lean_engine_parity` audits a benchmark showing that `ml4t-backtest`'s LEAN profile
# reproduces QuantConnect LEAN on a synthetic 250-asset fixture. A fixture is built to stress an
# engine; it is not what the book actually trades. The obvious next question is whether the same
# agreement still holds on the case studies' real weights, real prices and real cost rates.
#
# This notebook reads a report that says it does, across three case studies, and is careful about
# what that report can support. The recorded artifact keeps the comparison's conclusions and not
# the fills those conclusions were drawn from, so on the default path a reader is trusting the
# producer rather than checking it. Setting `RUN_LIVE = True` re-runs both engines locally, which
# is a different and stronger kind of evidence.
#
# ## Learning objectives
#
# - Distinguish a recorded conclusion from preserved evidence, and say which one a given artifact
#   gives you.
# - Compare two engines on a fill multiset rather than on a trade log, and explain why row order is
#   the wrong surface.
# - Name what an attestation has to retain for somebody else to check it later.
#
# ## Book reference
# Chapter 16, Section 16.3 (vectorized and event-driven backtesting).
#
# ## Prerequisites
#
# - `07_engine_divergence_anatomy`, for what makes two engines differ.
# - `15_lean_engine_parity`, which audits the synthetic benchmark this notebook asks about.

# %% [markdown]
# ## Setup

# %%
"""Case-study LEAN parity on real book artifacts."""

import json
import os
import shutil
import time
from pathlib import Path

import ml4t.backtest as ml4t_backtest_pkg
import numpy as np
import polars as pl

from case_studies.utils.analytics import SHORT_NAMES
from utils import ML4T_DATA_PATH
from utils.paths import get_chapter_dir, get_output_dir

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides after this cell
RUN_LIVE = False
CASE_STUDIES = "etfs,sp500_equity_option_analytics,us_equities_panel"
MAX_CASE_STUDIES = 0  # 0 = all requested

# %%
EPSILON = float(np.finfo(float).eps)
OUTPUT_DIR = get_output_dir(16, "case_study_lean_parity")
RESULTS_PATH = OUTPUT_DIR / "case_study_lean_parity.csv"
CACHED_ARTIFACT_PATH = get_chapter_dir(16) / "resources" / "case_study_lean_parity_results.json"
LIVE_ARTIFACT_PATH = OUTPUT_DIR / "case_study_lean_parity_live.json"

# %% [markdown]
# ## 1. Find what is available in this environment
#
# The live rerun path depends on the sibling `ml4t-backtest` repository because
# the heavy LEAN orchestration lives in its internal validation layer, not in
# the public runtime API.


# %%
def resolve_backtest_repo() -> Path | None:
    """Find a checkout carrying the validation harness, or None if there is not one.

    The harness lives in the `ml4t-backtest` source repository rather than in the installed
    package, because it orchestrates Docker and the LEAN CLI. `ML4T_BACKTEST_REPO` names it
    explicitly; otherwise the installed package's own ancestry is searched, which finds it in
    an editable install. There is deliberately no fallback to a path under the home
    directory: a resolver that guesses where a repository lives on the author's machine
    fails silently and differently on everybody else's.
    """
    candidates: list[Path] = []
    override = os.environ.get("ML4T_BACKTEST_REPO")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(Path(ml4t_backtest_pkg.__file__).resolve().parents)
    return next(
        (c for c in candidates if (c / "validation" / "benchmark_suite.py").is_file()), None
    )


BACKTEST_REPO = resolve_backtest_repo()
LEAN_WORKSPACE = BACKTEST_REPO / "validation" / "lean" / "workspace" if BACKTEST_REPO else None
LEAN_CONFIG = LEAN_WORKSPACE / "lean.json" if LEAN_WORKSPACE else None

print(f"Validation harness: {'found' if BACKTEST_REPO else 'not found (cached report only)'}")
print(f"LEAN workspace:     {'found' if LEAN_CONFIG and LEAN_CONFIG.exists() else 'not found'}")
print(f"Case-study data:    {'found' if Path(ML4T_DATA_PATH).exists() else 'not found'}")

# %% [markdown]
# ### What a live rerun would need
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
            "detail": "found" if BACKTEST_REPO else "set ML4T_BACKTEST_REPO",
        },
        {
            "requirement": "docker",
            "ready": shutil.which("docker") is not None,
            "detail": "on PATH" if shutil.which("docker") else "missing",
        },
        {
            "requirement": "lean or uvx",
            "ready": shutil.which("lean") is not None or shutil.which("uvx") is not None,
            "detail": Path(shutil.which("lean") or shutil.which("uvx") or "missing").name,
        },
        {
            "requirement": "lean workspace config",
            "ready": LEAN_CONFIG is not None and LEAN_CONFIG.exists(),
            "detail": LEAN_CONFIG.name if LEAN_CONFIG else "not found",
        },
        {
            "requirement": "case-study data root",
            "ready": Path(ML4T_DATA_PATH).exists(),
            "detail": "found" if Path(ML4T_DATA_PATH).exists() else "set ML4T_DATA_PATH",
        },
    ]
    return pl.DataFrame(rows)


prereq_df = check_live_prerequisites()
prereq_df

# %% [markdown]
# ## 2. The report, and what surface it compared
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
# ## 3. Running both engines here instead
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
# ### One case study through both engines
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
# ### Assemble the result
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
    # prereq_df is computed above and was never read. Without a LEAN checkout
    # LEAN_WORKSPACE is None, and run_case_study_pair divides it by a string - a
    # TypeError, which is not in the tuple caught below, so a clean clone setting
    # RUN_LIVE=True died on an operand-type message instead of the prerequisite
    # list this notebook promises. 17_ and 18_ both guard the same path this way.
    if not bool(prereq_df["ready"].all()):
        raise RuntimeError(
            "Live rerun requested with missing prerequisites: "
            + ", ".join(prereq_df.filter(~pl.col("ready"))["requirement"].to_list())
        )
    skipped: list[tuple[str, str]] = []
    live_rows: list[dict] = []
    for case_study in selected_case_studies:
        try:
            live_rows.append(run_case_study_pair(case_study))
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            skipped.append((case_study, str(exc)))

    if skipped:
        raise RuntimeError(
            "Live rerun requested but these case studies did not complete: "
            + "; ".join(f"{case_study} ({reason})" for case_study, reason in skipped)
        )
    payload = build_live_payload(live_rows, skipped)
    LIVE_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved live artifact: {LIVE_ARTIFACT_PATH}")

# %% [markdown]
# ## 4. What the report records
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

print(f"Saved {results_df.height} rows to {RESULTS_PATH.name}")

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
# Read `all_fill_multisets_match` against `n_case_studies`: the summary is only as broad as the
# set that ran. A live rerun raises rather than reporting a subset, so this row cannot quietly
# describe fewer case studies than were requested.

# %%
summary_df = pl.DataFrame([payload["summary"]])
summary_df

# %% [markdown]
# ## 5. What the fill counts and value gaps look like
#
# Neither of these has a shape worth plotting. The fill counts are equal on every row by
# construction if the parity holds, and the value gaps are all within a rounding error of zero.
# A pair of bar charts would show three pairs of identical bars beside three bars of zero length,
# which says less than the numbers do.
#
# What is worth showing is the scale the gaps are small *relative to*, because "the difference is
# tiny" means nothing without it.

# %%
scale_df = results_df.select(
    "display",
    pl.col("lean_fills").alias("fills"),
    pl.col("lean_final_value").round(2).alias("terminal_value"),
    pl.col("final_value_gap_usd").alias("gap_usd"),
    (pl.col("final_value_gap_usd").abs() / pl.col("lean_final_value")).alias("relative_gap"),
    (pl.col("final_value_gap_usd").abs() / pl.col("lean_final_value") / EPSILON).alias(
        "gap_in_float_epsilons"
    ),
)
scale_df

# %% [markdown]
# ## 6. What this establishes
#
# The last column is the one that settles it. A float64 carries about fifteen significant digits,
# and the smallest difference representable next to a number is one epsilon of it. The gaps here
# are a few hundred epsilons on accounts that took thousands to tens of thousands of fills, which
# is what accumulated rounding looks like: each fill contributes an error of order one epsilon, and
# they add up roughly with the number of operations. It is not a disagreement about execution.
#
# The row-order column is the one to read carefully. It is false on every case study, and that is
# expected rather than alarming: two engines that iterate assets in different orders will emit the
# same day's fills in different sequence, and a trade log is a record of emission order. Comparing
# logs row by row would report a difference on every case study while the accounts are identical.
# The multiset - the same fills, sorted - is the surface that answers the question actually being
# asked.
#
# What the default path does not do is prove any of it. The artifact carries the conclusions and
# not the fills, so a reader is trusting its producer. Making this checkable by somebody else needs
# the raw fill surfaces from both engines, the execution log, the environment identity, and hashes
# of every input and output the comparison consumed.

# %%
matched = int(payload["summary"]["matched_case_studies"])
total = int(payload["summary"]["n_case_studies"])
max_gap = float(payload["summary"]["max_abs_final_value_gap_usd"])
raw_row_matches = int(results_df["raw_row_order_match"].sum())

print(f"Reported sorted fill multiset matches: {matched}/{total}")
print(f"Reported raw row-order matches:        {raw_row_matches}/{total}")
print(f"Reported max absolute value gap:       ${max_gap:.10f}")

# %% [markdown]
# ## Key takeaways
#
# 1. **A recorded conclusion is not evidence.** This artifact says the engines agreed and does not
#    contain what they agreed on. That is worth reading and is not worth citing as a proof, and
#    the difference is worth being pedantic about, because an artifact that keeps only its own
#    conclusions cannot be checked by anyone who did not run it.
# 2. **Compare the right surface.** Two engines that iterate assets differently emit the same day's
#    fills in a different order, so a row-by-row log comparison reports a difference that is purely
#    about logging. Sorting the fills into a multiset removes the artifact and leaves the question.
# 3. **A difference is small relative to something.** Terminal-value gaps here are parts per
#    billion of the portfolio, which is floating-point accumulation across tens of thousands of
#    fills. Quoting the dollar figure alone would make it sound like a finding.
# 4. **A partial run is not the requested run.** When a live rerun is asked for and any case study
#    fails, this notebook raises rather than summarizing whatever completed. A summary row that
#    silently describes two of three case studies is worse than no summary.
# 5. **Say what a future attestation must keep.** Both engines' raw fill surfaces, the execution
#    log, the environment identity, and hashes of every input and output. Anything less and the
#    next reader is in the position this notebook's default path is in.
#
# ### Known limitations
#
# - The default path audits a cached report, and the report does not preserve what it summarizes.
# - Three case studies, all daily, all long-only target weights. Nothing here says the profile
#   transfers to intraday data, to short positions, or to instruments these case studies never hold.
# - A live rerun needs Docker, the LEAN CLI, a configured workspace and the case-study data root.
#   That is a high bar, and it is the reason the cached path exists rather than a reason to trust
#   it further than it goes.
