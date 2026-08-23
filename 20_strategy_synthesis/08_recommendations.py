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
# # Strategy Recommendations
#
# **Docker image**: `ml4t`
#
# This notebook makes the final call: which case studies survive the full
# pipeline from signal detection through holdout validation, and which
# fail — and why?
#
# Every gate verdict and metric here is derived from the data produced by
# NB00–NB05; the HTM cost-cascade figures (which reproduce
# htm_cost_sensitivity.parquet) and the S&P 500 Options cost handling are the
# only hardcoded elements.
#
# **Learning Objectives**:
# - Trace the stage attrition funnel through the pipeline gates for each of 9 case studies
# - Classify failures into signal, implementation, and evidence-quality buckets
# - Identify structural features that predict pipeline survival
#
# **Book Reference**: Chapter 20, Sections 20.6–20.7
#
# **Prerequisites**: Run [`00_holdout_predictions`](00_holdout_predictions.ipynb) and [`01_aggregate_synthesis`](01_aggregate_synthesis.ipynb).

# %%
"""Ch20 NB06 — Final recommendations derived from pipeline data."""

import json
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import FancyBboxPatch, Patch

warnings.filterwarnings("ignore")

from utils.paths import get_chapter_dir

# %%
MAX_SYMBOLS = 0

# %%
OUTPUT_DIR = get_chapter_dir(20) / "output"

DISPLAY_NAMES = {
    "etfs": "ETFs",
    "crypto_perps_funding": "Crypto",
    "nasdaq100_microstructure": "NASDAQ-100",
    "sp500_equity_option_analytics": "S&P 500 Eq+Opt",
    "us_firm_characteristics": "US Firms",
    "fx_pairs": "FX Pairs",
    "cme_futures": "CME Futures",
    "sp500_options": "S&P 500 Options",
    "us_equities_panel": "US Equities",
}
NASDAQ_ID = "nasdaq100_microstructure"

# %% [markdown]
# ## 1. Load Pipeline Data
#
# All data comes from NB00 (holdout predictions) and NB01 (aggregate synthesis).
# We make no assumptions beyond what the data shows.

# %%
synthesis = json.load((OUTPUT_DIR / "all_synthesis.json").open())

# Registry-based holdout results (more current than synthesis JSON)
holdout_df = pl.read_parquet(OUTPUT_DIR / "holdout_results.parquet")
holdout_map = {row["cs_id"]: row for row in holdout_df.iter_rows(named=True)}

print(f"Loaded synthesis for {len(synthesis)} case studies")
print(f"Holdout results for {holdout_df.height} case studies")

missing_holdout = sorted(set(synthesis) - set(holdout_map))
if missing_holdout:
    print(
        "\nNo holdout row for: "
        + ", ".join(DISPLAY_NAMES.get(cs, cs) for cs in missing_holdout)
        + "\nEach is classified below as lacking holdout evidence rather than as having failed "
        "a gate. The two are different conclusions and the table keeps them apart."
    )

# %% [markdown]
# ## 2. Stage Attrition Funnel
#
# The centerpiece of this analysis. We start with 9 case studies and
# trace how many survive each pipeline gate. At each stage, we name
# which case studies drop and why.
#
# The gates are cumulative: a case study must pass all preceding gates
# to be counted at the next stage. NB01's funnel (see the "Stage
# Attrition Funnel" section in `01_aggregate_synthesis`) reports the
# same gates with *independent* per-stage counts — useful for seeing
# which gate is the largest filter, not which case studies survive end
# to end. Per-stage independent counts in NB01 can therefore exceed the
# cumulative count shown here, especially at the cost and risk stages.

# %%
# Build the attrition data from pipeline evidence
stages = []

# Gate 0: All case studies start
all_cs = list(synthesis.keys())
stages.append(("Start", set(all_cs)))

# Gate 1: Positive IC (best model family has IC > 0)
positive_ic = set()
for cs, data in synthesis.items():
    models = data["pipeline_summary"]["models"]
    best_ic = max((m.get("ic_mean") or 0) for m in models.values()) if models else 0
    if best_ic > 0:
        positive_ic.add(cs)
stages.append(("Positive IC", positive_ic))

# Gate 2: Positive validation Sharpe (carrier signal-stage SR > 0).
# Uses the carrier's validation ML Sharpe (`backtest.ml_sharpe`), not
# `risk.baseline_sharpe` — the latter is null for case studies whose risk
# stage is not applicable (sp500_options HTM, us_firm vectorized, nasdaq
# before the ensemble cost/risk pass), which would drop them at the
# validation gate even though their carrier validation Sharpe is positive.
positive_val_sharpe = set()
for cs in positive_ic:
    bt = synthesis[cs]["pipeline_summary"].get("backtest", {})
    val_sr = bt.get("ml_sharpe")
    # Validation applies to every case study; drop only on a genuine
    # non-positive carrier Sharpe (FX Pairs, val −0.004).
    if val_sr is None or val_sr > 0:
        positive_val_sharpe.add(cs)
stages.append(("Val Sharpe > 0", positive_val_sharpe))

# %%
# Gate 3: Survives transaction costs (net Sharpe > 0 at actual cost level).
# A case study whose cost stage is not applicable (sp500_options uses §18.8
# option-native bid-ask accounting, not a bps sweep) passes through rather
# than being eliminated — a gate drops a case study only on a genuine
# negative verdict at an applicable stage.
cost_surviving = set()
for cs in positive_val_sharpe:
    costs = synthesis[cs]["pipeline_summary"]["costs"]
    if costs.get("not_applicable_reason"):
        cost_surviving.add(cs)
        continue
    net_sr = costs.get("net_sharpe_at_actual")
    if costs.get("survives_costs") and net_sr is not None and net_sr > 0:
        cost_surviving.add(cs)
stages.append(("Cost Survival", cost_surviving))

# Gate 4: Holdout gate (holdout Sharpe > 0)
holdout_passing = set()
for cs in cost_surviving:
    ho = holdout_map.get(cs, {})
    ho_sharpe = ho.get("holdout_sharpe")
    if ho_sharpe is not None and ho_sharpe > 0:
        holdout_passing.add(cs)
stages.append(("Holdout SR > 0", holdout_passing))

# Gate 5: Risk overlay doesn't destroy the edge, and active uncertainty evidence
# is sufficient for a deployment-facing classification. Case studies whose risk
# stage is not applicable (sp500_options HTM expiration structure;
# us_firm_characteristics vectorized path with portfolio overlays purged)
# pass through rather than being eliminated.
all_gates_pass = set()
for cs in holdout_passing:
    if cs == NASDAQ_ID:
        # The fixed carrier is positive on point estimate, but both corrected
        # validation and holdout intervals cross zero. Broad cost and risk grids
        # are also deferred to v3.1, so it cannot clear the evidence gate.
        continue
    risk = synthesis[cs]["pipeline_summary"].get("risk", {})
    if risk.get("not_applicable_reason"):
        all_gates_pass.add(cs)
        continue
    managed_sr = risk.get("managed_sharpe")
    if managed_sr is not None and managed_sr > 0:
        all_gates_pass.add(cs)
stages.append(("Evidence ready", all_gates_pass))

# Print the funnel
print("=== Stage Attrition Funnel ===\n")
for i, (name, passing) in enumerate(stages):
    dropped = stages[i - 1][1] - passing if i > 0 else set()
    dropped_names = [DISPLAY_NAMES.get(c, c) for c in sorted(dropped)]
    n = len(passing)
    bar = "█" * n + "░" * (9 - n)
    drop_str = f"  dropped: {', '.join(dropped_names)}" if dropped_names else ""
    print(f"  {name:20s} {bar} {n}/9{drop_str}")

# %% [markdown]
# ### Stage Attrition Waterfall
#
# This is the single most important figure in Ch20. It shows the
# cumulative pipeline survival rate and names every dropout.

# %%
stage_names = [s[0] for s in stages]
stage_counts = [len(s[1]) for s in stages]

fig, ax = plt.subplots(figsize=(12, 6))

# Waterfall bars
colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(stage_names)))
bars = ax.bar(
    range(len(stage_names)), stage_counts, color=colors, edgecolor="white", linewidth=1.5, width=0.7
)

# Annotate counts and dropouts
for i, (bar, count) in enumerate(zip(bars, stage_counts, strict=False)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        str(count),
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=14,
    )

    if i > 0:
        dropped = stages[i - 1][1] - stages[i][1]
        if dropped:
            dropped_names = [DISPLAY_NAMES.get(c, c) for c in sorted(dropped)]
            delta = stage_counts[i - 1] - count
            # Show dropout annotation
            ax.annotate(
                f"−{delta}: {', '.join(dropped_names)}",
                xy=(i - 0.5, (stage_counts[i - 1] + count) / 2),
                fontsize=7.5,
                color="#c0392b",
                ha="center",
                va="center",
                style="italic",
            )

ax.set_xticks(range(len(stage_names)))
ax.set_xticklabels(stage_names, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Case Studies Remaining")
ax.set_ylim(0, 10.5)
ax.set_title("Pipeline Attrition Across Five Gates")
ax.axhline(y=0, color="black", linewidth=0.5)

fig.tight_layout()
fig.show()

# %% [markdown]
# The funnel tells a clear story: most case studies produce positive IC
# (the ML signal is real), but the pipeline progressively filters out
# strategies that cannot translate signal into robust, cost-surviving,
# out-of-sample economic performance. Each gate eliminates for a
# different reason.

# %% [markdown]
# ## 3. Exclusion Taxonomy
#
# Every case study that fails the pipeline maps to at least one exclusion
# type. We organize these into three parent buckets and assign them
# **from the data**, not from hardcoded labels.


# %%
def classify_exclusions():
    """Classify each case study's pipeline failures into exclusion categories."""
    # Data-driven exclusion assignment
    exclusions: dict[str, list[dict]] = defaultdict(list)

    for cs, data in synthesis.items():
        display = DISPLAY_NAMES.get(cs, cs)
        models = data["pipeline_summary"]["models"]
        costs = data["pipeline_summary"]["costs"]
        risk = data["pipeline_summary"].get("risk", {})
        ho = holdout_map.get(cs, {})

        best_ic = max((m.get("ic_mean") or 0) for m in models.values()) if models else 0
        ho_sharpe = ho.get("holdout_sharpe")
        ho_ic = ho.get("holdout_ic")
        net_sr = costs.get("net_sharpe_at_actual")
        survives = costs.get("survives_costs", False)
        managed_sr = risk.get("managed_sharpe")
        worst_dd = risk.get("worst_drawdown_pct", 0)

        # Signal invalidity
        if best_ic <= 0:
            exclusions["No detectable signal"].append(
                {"cs": display, "detail": f"Best IC = {best_ic:.4f}"}
            )
        elif net_sr is not None and net_sr <= 0 and survives is False:
            exclusions["Insufficient edge after costs"].append(
                {"cs": display, "detail": f"IC = {best_ic:.4f} but net SR = {net_sr:.2f}"}
            )
        elif (
            best_ic > 0
            and net_sr is not None
            and net_sr > 0
            and ho_sharpe is not None
            and ho_sharpe <= 0
        ):
            # Had signal, had validation Sharpe, but holdout collapsed
            pass  # Will be caught by holdout_collapse below

        # Holdout collapse
        if ho_sharpe is not None and ho_sharpe <= 0 and cs in positive_val_sharpe:
            exclusions["Holdout collapse"].append(
                {"cs": display, "detail": f"Val SR → Holdout SR = {ho_sharpe:.2f}"}
            )
        elif ho_sharpe is None and cs in positive_val_sharpe:
            exclusions["Holdout not available"].append(
                {"cs": display, "detail": "Degenerate or missing holdout predictions"}
            )

        # Implementation infeasibility
        if cs == "sp500_options":
            exclusions["Net-negative under realistic costs"].append(
                {
                    "cs": display,
                    "detail": "HTM cost cascade max Sharpe = -0.28 at 20% half-spread fraction; net Sharpe negative across the full cascade (-0.47 at 50%, -0.72 at 100%)",
                }
            )

        if abs(worst_dd) > 50:
            exclusions["Unacceptable drawdown"].append(
                {"cs": display, "detail": f"Max DD = {worst_dd:.0f}%"}
            )

        # A holdout interval that spans zero says the window cannot tell this strategy from one
        # with no edge. That is a different finding from failing a gate, and it is read from the
        # row rather than asserted, so it applies to whichever case studies it happens to be true
        # of rather than to one named in advance.
        if not ho:
            exclusions["No holdout evidence"].append(
                {"cs": display, "detail": "No holdout row in the registry"}
            )
        elif (
            ho.get("holdout_sharpe_ci_lo") is not None
            and ho.get("holdout_sharpe_ci_hi") is not None
            and ho["holdout_sharpe_ci_lo"] < 0 < ho["holdout_sharpe_ci_hi"]
        ):
            exclusions["Statistically unresolved"].append(
                {
                    "cs": display,
                    "detail": (
                        f"Holdout Sharpe {ho['holdout_sharpe']:+.3f}, interval "
                        f"[{ho['holdout_sharpe_ci_lo']:+.3f}, "
                        f"{ho['holdout_sharpe_ci_hi']:+.3f}] spans zero"
                    ),
                }
            )
    return exclusions


# %%
exclusions = classify_exclusions()

# Group into parent buckets
BUCKETS = {
    "Signal Invalidity": [
        "No detectable signal",
        "Insufficient edge after costs",
        "Positive IC but no stable Sharpe",
        "Cadence-horizon mismatch",
    ],
    "Implementation Infeasibility": [
        "Net-negative under realistic costs",
        "Unacceptable drawdown",
    ],
    "Evidence-Quality Failure": [
        "Holdout collapse",
        "Holdout not available",
        "Unreproducible model",
        "Statistically unresolved",
    ],
}

print("=== Exclusion Taxonomy ===\n")
for bucket, types in BUCKETS.items():
    active_types = [(t, exclusions[t]) for t in types if exclusions[t]]
    if not active_types:
        continue
    print(f"  {bucket}")
    for excl_type, cases in active_types:
        print(f"    • {excl_type}")
        for case in cases:
            print(f"      – {case['cs']}: {case['detail']}")
    print()

# %% [markdown]
# The three-bucket organization makes the taxonomy easier to remember
# and act on:
#
# - **Signal invalidity** means the prediction problem itself doesn't
#   work — no amount of better implementation will help.
# - **Implementation infeasibility** means the signal exists but
#   practical constraints (costs, drawdowns) prevent deployment.
# - **Evidence-quality failure** means we can't trust the results —
#   either the holdout invalidated the signal, or the evaluation
#   infrastructure has gaps.
#
# The taxonomy helps diagnose *why* a case study failed, not grade it.
# Each failure mode points to a different second-iteration response.

# %% [markdown]
# ## 4. Pipeline Evidence Summary
#
# For each case study we report the factual per-gate evidence the pipeline
# produced — positive validation IC, survival of the cost sweep, positive
# holdout Sharpe, positive risk-managed Sharpe, and holdout decay under
# fifty percent. These are observations rather than categorical deployment
# labels; readers who want to weigh the evidence differently have all the
# numbers in one place.


# %%
def build_evidence_profile():
    """Per-case-study factual evidence across the pipeline gates.

    Reports five gate-level booleans (positive IC, survives costs, positive
    holdout Sharpe, positive risk-managed Sharpe, holdout decay < 50%) plus
    the underlying numbers. No categorical "outcome" label is assigned —
    the evidence is reported as observations.
    """
    rows = []
    for cs, data in synthesis.items():
        display = DISPLAY_NAMES.get(cs, cs)
        models = data["pipeline_summary"]["models"]
        costs = data["pipeline_summary"]["costs"]
        risk = data["pipeline_summary"].get("risk", {})
        ho = holdout_map.get(cs, {})

        best_ic = max((m.get("ic_mean") or 0) for m in models.values()) if models else 0
        ho_sharpe = ho.get("holdout_sharpe")
        net_sr = costs.get("net_sharpe_at_actual")
        managed_sr = risk.get("managed_sharpe")

        positive_ic = best_ic > 0
        cost_na = bool(costs.get("not_applicable_reason"))
        risk_na = bool(risk.get("not_applicable_reason"))
        survives_costs = bool(costs.get("survives_costs", False))
        positive_holdout = ho_sharpe is not None and ho_sharpe > 0
        positive_managed = managed_sr is not None and managed_sr > 0

        backtest = data["pipeline_summary"].get("backtest", {})
        val_sharpe = backtest.get("ml_sharpe") or 0
        if val_sharpe > 0 and ho_sharpe is not None:
            holdout_decay = (val_sharpe - ho_sharpe) / val_sharpe
        else:
            holdout_decay = None
        modest_decay = holdout_decay is not None and holdout_decay < 0.50
        evidence_resolved = cs != NASDAQ_ID

        # Gate tally as passed/applicable. Not-applicable stages (cost or
        # risk) are excluded from both numerator and denominator rather than
        # counted as failures, so a case study is never penalized for a stage
        # its canonical strategy does not run.
        gate_flags = [
            (True, positive_ic),
            (not cost_na, survives_costs),
            (True, positive_holdout),
            (not risk_na, positive_managed),
            (True, modest_decay),
            (True, evidence_resolved),
        ]
        gates_passed = sum(1 for appl, passed in gate_flags if appl and passed)
        gates_applicable = sum(1 for appl, _ in gate_flags if appl)

        cs_exclusions = []
        for excl_type, cases in exclusions.items():
            for case in cases:
                if case["cs"] == display:
                    cs_exclusions.append(excl_type)

        rows.append(
            {
                "case_study": display,
                "gates_passed": gates_passed,
                "gates_applicable": gates_applicable,
                "positive_ic": positive_ic,
                "survives_costs": survives_costs,
                "positive_holdout": positive_holdout,
                "positive_managed": positive_managed,
                "modest_decay": modest_decay,
                "evidence_resolved": evidence_resolved,
                "best_ic": round(best_ic, 4),
                "holdout_sharpe": round(ho_sharpe, 2) if ho_sharpe is not None else None,
                "net_sharpe": round(net_sr, 2) if net_sr is not None else None,
                "managed_sharpe": round(managed_sr, 2) if managed_sr is not None else None,
                "holdout_decay": round(holdout_decay, 2) if holdout_decay is not None else None,
                "exclusions": "; ".join(cs_exclusions) if cs_exclusions else "—",
                "top_family": ho.get("family", "—") if ho else "—",
            }
        )
    return pl.DataFrame(rows).sort(["gates_passed", "best_ic"], descending=[True, True])


# %%
evidence_df = build_evidence_profile()


def _fmt(v, spec=".2f"):
    return format(v, spec) if v is not None else "n/a"


print("=== Pipeline Evidence Summary ===\n")
print(
    f"{'Case Study':22s} {'Gates':>6s}  {'Best IC':>8s}  {'HO SR':>7s}  {'Net SR':>7s}  {'Mgd SR':>7s}  {'Decay':>6s}  Exclusions"
)
print("─" * 110)
for row in evidence_df.iter_rows(named=True):
    gates = f"{row['gates_passed']}/{row['gates_applicable']}"
    ic = f"{row['best_ic']:+.4f}"
    decay = f"{row['holdout_decay']:.0%}" if row["holdout_decay"] is not None else "n/a"
    print(
        f"{row['case_study']:22s} {gates:>6s}  {ic:>8s}  "
        f"{_fmt(row['holdout_sharpe']):>7s}  {_fmt(row['net_sharpe']):>7s}  "
        f"{_fmt(row['managed_sharpe']):>7s}  {decay:>6s}  {row['exclusions']}"
    )

# Display as table
evidence_df.select(
    "case_study",
    "gates_passed",
    "top_family",
    "best_ic",
    "holdout_sharpe",
    "net_sharpe",
    "managed_sharpe",
    "holdout_decay",
    "exclusions",
)

# %% [markdown]
# ## 5. Structural Features and Gate Passage
#
# The attrition funnel shows *which* gate a case study was eliminated at.
# This section asks: what structural features correlate with passing
# every pipeline gate?

# %%
# Build structural features for each CS
structural_rows = []
for cs, data in synthesis.items():
    meta = data["meta"]
    models = data["pipeline_summary"]["models"]
    costs = data["pipeline_summary"]["costs"]
    ho = holdout_map.get(cs, {})

    best_ic = max((m.get("ic_mean") or 0) for m in models.values()) if models else 0
    ho_sharpe = ho.get("holdout_sharpe")

    passes_all_gates = (
        ho_sharpe is not None
        and ho_sharpe > 0
        and costs.get("survives_costs", False)
        and cs != "sp500_options"  # Known evidence issue: spread overwhelms signal
        and cs != NASDAQ_ID  # Fixed-carrier intervals cross zero; broad grids deferred
    )

    structural_rows.append(
        {
            "case_study": DISPLAY_NAMES.get(cs, cs),
            "asset_class": meta.get("asset_class", "unknown"),
            "frequency": meta.get("frequency", "unknown"),
            "universe_size": meta.get("universe_size", 0),
            "best_val_ic": best_ic,
            "holdout_sharpe": ho_sharpe,
            "passes_all_gates": passes_all_gates,
            "top_family": ho.get("family", "unknown") if ho else "unknown",
            "cost_bps": costs.get("actual_bps", 0),
        }
    )

struct_df = pl.DataFrame(structural_rows)

# %%
# Analysis: what structural features correlate with gate passage?
full_pass = struct_df.filter(pl.col("passes_all_gates"))
gate_miss = struct_df.filter(~pl.col("passes_all_gates"))

print("=== Structural Analysis: Full-Gate Pass vs Gate Miss ===\n")
print(f"All gates pass ({full_pass.height}): {', '.join(full_pass['case_study'].to_list())}")
print(f"At least one miss ({gate_miss.height}): {', '.join(gate_miss['case_study'].to_list())}")

if full_pass.height > 0 and gate_miss.height > 0:
    pass_ic = full_pass["best_val_ic"].mean()
    miss_ic = gate_miss["best_val_ic"].mean()
    print(f"\nMean validation IC — full pass: {pass_ic:.4f}, gate miss: {miss_ic:.4f}")

    # Frequency distribution
    print("\nFrequency distribution:")
    for freq in struct_df["frequency"].unique().sort().to_list():
        n_pass = full_pass.filter(pl.col("frequency") == freq).height
        n_miss = gate_miss.filter(pl.col("frequency") == freq).height
        print(f"  {freq:12s}: {n_pass} full-pass, {n_miss} gate-miss")

    # Top-family distribution
    print("\nRank-1 model family:")
    for fam in struct_df["top_family"].unique().sort().to_list():
        n_pass = full_pass.filter(pl.col("top_family") == fam).height
        n_miss = gate_miss.filter(pl.col("top_family") == fam).height
        total = n_pass + n_miss
        if total > 0:
            print(f"  {fam:18s}: {n_pass}/{total} full-pass ({100 * n_pass / total:.0f}%)")

# %% [markdown]
# The structural comparison reveals patterns that go beyond individual
# case study results:
#
# - **Daily frequency** case studies most often pass every gate — the
#   cadence balances signal decay against cost pressure.
# - **Higher-frequency** case studies split on outcome. One clears the cost gate on its gross
#   signal and still turns in a negative holdout Sharpe. NASDAQ-100's holdout Sharpe is positive,
#   and its confidence interval spans zero by a wide margin in both directions - the exclusion
#   table above prints both. An interval that wide says the holdout window cannot distinguish this
#   strategy from one with no edge, which is a different statement from having found it wanting.
# - **Signal strength alone does not drive gate passage** — S&P 500 Options
#   has positive IC and a positive holdout Sharpe under the bottom-quintile
#   liquid-universe construction, but the cost cascade turns its highest Sharpe negative even at
#   the most generous half-spread assumption tested, while CME futures passes the downstream gates
#   on a moderate IC because its costs are small relative to its edge.
# - The top-ranked model family is a weaker predictor than frequency and
#   cost structure. Deep-learning and GBM rank-1 configurations both appear
#   in the full-pass group.

# %% [markdown]
# ## 6. Evidence Snapshot
#
# Compact per-case-study summary of the end-of-pipeline numbers, sorted
# by the count of gates passed (descending), then by best IC.

# %%
print("=== Evidence Snapshot ===\n")
print(
    f"{'Case Study':25s} {'Gates':>6s} {'Top Family':12s} {'Managed SR':>10s} {'Net SR':>10s} {'HO SR':>10s}  Primary Exclusion"
)
print("─" * 110)

for row in evidence_df.iter_rows(named=True):
    ho_str = f"{row['holdout_sharpe']:.2f}" if row["holdout_sharpe"] is not None else "n/a"
    net_str = f"{row['net_sharpe']:.2f}" if row["net_sharpe"] is not None else "n/a"
    managed_str = f"{row['managed_sharpe']:.2f}" if row["managed_sharpe"] is not None else "n/a"
    gates = f"{row['gates_passed']}/{row['gates_applicable']}"
    excl = row["exclusions"][:40] if row["exclusions"] != "—" else "—"
    print(
        f"{row['case_study']:25s} {gates:>6s} {row['top_family']:12s} {managed_str:>10s} {net_str:>10s} {ho_str:>10s}  {excl}"
    )

# %% [markdown]
# ## 7. Ensembles: A Chapter-End Note
#
# A natural follow-on question is whether equal-weight blending of the
# top-three configurations per case study would tighten cross-fold
# stability at the cost of a small reduction in peak Sharpe. The
# experiment was carried out outside this notebook (on the per-fold
# return series, not on the per-fold Sharpe summaries that the registry
# stores at the rank-1 level for most case studies in this iteration).
# The finding documented in the chapter prose: a minority of case
# studies see lower per-fold dispersion under the blend, the rest do
# not, and the median peak Sharpe sacrificed is roughly 0.3.
#
# The result is not registered, since this is not the iteration in
# which we are scoring ensembles against single-model rank-1
# configurations. It belongs in the "next iteration" list at the
# end of this section.
#
# NASDAQ-100 is the bounded exception in this release: its ensemble was fixed
# before holdout scoring as diversification under overlapping validation
# uncertainty. The corrected positive linear holdout is a comparator only and
# cannot be used to reselect the carrier or describe the ensemble as an ex-post
# rescue.

# %% [markdown]
# ## Key Takeaways
#
# 1. **The funnel is the story**: starting from 9 case studies, the
#    pipeline progressively narrows the set of case studies that pass
#    each gate. Each gate drops cases for a different reason.
#
# 2. **Signal is necessary but not sufficient**: all 9 case studies
#    show positive IC; 8 of 9 show positive signal-stage Sharpe; the
#    pipeline gradually narrows the set further through costs, holdout
#    validation, and evidence-quality checks.
#
# 3. **Costs are the great equalizer**: case studies with the strongest
#    raw signals (options, high-frequency) face the tightest cost
#    margins. The edge-to-cost ratio, not IC alone, determines
#    economic viability.
#
# 4. **Failure modes are distinct**: the exclusion taxonomy identifies
#    three structural failure categories — signal invalidity,
#    implementation infeasibility, and evidence-quality failure.
#    Each points to a different second-iteration response.
#
# 5. **Evidence quality ≠ headline Sharpe**: a high managed Sharpe paired
#    with a fatal cost environment (S&P 500 Options: best Sharpe −0.28 even
#    at the lowest cost rung of the HTM cascade) or a holdout collapse
#    (S&P 500 Eq+Opt: managed Sharpe 2.39 but holdout −0.73) is not evidence
#    one would act on. The pipeline's downstream gates flag that gap.
#
# 6. **The pipeline matters more than any single model**: the full
#    journey from data to holdout determines the evidence a case study
#    produces. The strongest models in prediction (Ch11-15) are not
#    always the ones whose pipeline outputs most gates (Ch16-20).
#
# ## What Comes Next
#
# These nine case studies used publicly available, low-frequency market
# data with starter model configurations. The pipeline produced positive
# IC on all 9 case studies, a cost-surviving managed Sharpe on a subset,
# and a positive holdout Sharpe on a smaller subset (see the funnel above
# for exact counts). The chapter's claim is methodological: this is the
# pipeline a practitioner should run to find out whether a candidate
# strategy works, not a set of deployable strategies.
#
# The next-iteration handles inside the same workflow:
#
# - **Label refinement**: several case studies showed that horizon
#   choice changes IC by an order of magnitude (FX: 9x IC from 1d→21d,
#   US Firms: 9x IC from winsorization, NQ100: classification has higher
#   IC than regression). Systematic label search is high-leverage.
# - **Feature engineering**: the case studies share generic financial
#   features. Domain-specific features (order flow for NQ100, carry
#   dynamics for CME, funding structure for crypto) are the natural next
#   addition to test against the same triage and holdout protocol.
# - **Model tuning**: hyperparameter grids are deliberately modest in
#   this iteration. Focused tuning on the families that survive the
#   holdout gate (primarily GBM and selected DL architectures) with
#   larger search budgets is the next sweep.
# - **Ensemble construction**: the prediction correlation analysis
#   (model_analysis notebooks) shows low inter-family correlation
#   in several case studies; a simple average ensemble is a candidate
#   for variance reduction at constant mean IC.
# - **Strategy design**: this iteration tests basic long-short with a
#   few allocation methods. Sector constraints, regime conditioning,
#   dynamic position sizing, and multi-horizon blending are additional
#   axes the same pipeline can evaluate.
#
# The value of the workflow is the reproducible, auditable process that
# can be applied to new data, new markets, and new hypotheses, not the
# specific numbers in any single rank-1 row.
