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
# # Aggregate Synthesis
#
# **Docker image**: `ml4t`
#
# This notebook queries all 9 case study registries via `BacktestExplorer`
# and builds cross-dataset comparison DataFrames for the remaining Ch20 notebooks.
#
# **Data source**: `registry.db` per case study (no JSON files needed).
#
# **Learning Objectives**:
# - Query per-case-study backtest registries for signal, allocation, cost, and risk metrics
# - Build cross-dataset comparison tables
# - Export summary DataFrames for downstream notebooks (02–06)
#
# **Book Reference**: Chapter 20, Section 20.1 (The Nine Case Studies)
#
# **Prerequisites**: Case studies must have run Ch16–19 backtests.

# %%
"""Ch20 Aggregate Synthesis — query registries and compare all 9 case studies."""

import json
import sqlite3
import warnings

import polars as pl
import yaml
from IPython.display import Markdown, display

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.benchmark import load_benchmark_returns
from case_studies.utils.strategy_analysis import compute_cost_bps
from utils.paths import REPO_ROOT, get_case_study_dir, get_chapter_dir

# %% tags=["parameters"]
MAX_SYMBOLS = 0
# When non-empty, restricts the cross-CS iteration to the given subset.
# Used by the per-CS pipeline driver to populate `backtest_paired_metrics`
# for a single CS after its holdout has landed, without re-running the
# full 9-CS aggregation.
CASE_STUDIES: list[str] = []
# Test-only: in an isolated test registry, nasdaq's out-of-band cost-feasible
# carrier is absent, so its spine cannot resolve. Production leaves this False
# (a missing carrier fails loudly); the test harness sets it True so cost/risk
# for such a case study are reported not-applicable instead of raising.
ALLOW_MISSING_SPINE = False

# %%
OUTPUT_DIR = get_chapter_dir(20) / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_CASE_STUDIES = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    # FX rank-1 is linear/ridge_a100.0 on fwd_ret_21d (val Sharpe +0.048,
    # holdout +0.194), resolved after the 2026-06-01 DL-lookback fix. The
    # earlier deep_learning/tcn carrier (val +0.108 / holdout -1.59) was an
    # artifact of gappy validation folds (lookback=60 warmup consumed each
    # fold's head); those sets were purged and the clean lineage re-resolved.
    # See backtest_audit.md and project_registry_hash_collisions.
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]
if CASE_STUDIES:
    ALL_CASE_STUDIES = [cs for cs in ALL_CASE_STUDIES if cs in set(CASE_STUDIES)]

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

# %%
ASSET_CLASS_MAP = {
    "etfs": "equity_etf",
    "crypto_perps_funding": "crypto",
    "nasdaq100_microstructure": "equity_micro",
    "sp500_equity_option_analytics": "equity_options",
    "us_firm_characteristics": "equity_firm",
    "fx_pairs": "fx",
    "cme_futures": "futures",
    "sp500_options": "options",
    "us_equities_panel": "equity_panel",
}

FREQ_MAP = {
    "etfs": "daily",
    "crypto_perps_funding": "8h",
    "nasdaq100_microstructure": "15min",
    "sp500_equity_option_analytics": "daily",
    "us_firm_characteristics": "monthly",
    "fx_pairs": "daily",
    "cme_futures": "daily",
    "sp500_options": "daily",
    "us_equities_panel": "daily",
}

# %% [markdown]
# ## Load Registries
#
# Create a `BacktestExplorer` for each case study that has a registry.

# %%
explorers: dict[str, BacktestExplorer] = {}
configs: dict[str, dict] = {}

for cs in ALL_CASE_STUDIES:
    try:
        explorers[cs] = BacktestExplorer(cs)
        setup_path = get_case_study_dir(cs) / "config" / "setup.yaml"
        if setup_path.exists():
            configs[cs] = yaml.safe_load(setup_path.read_text())
        else:
            configs[cs] = {}
        summary = explorers[cs].summary()
        total = sum(summary.values())
        print(f"  [OK] {cs}: {total} backtests ({summary})")
    except FileNotFoundError:
        print(f"  [MISSING] {cs}: no registry.db")

print(f"\nLoaded: {len(explorers)}/{len(ALL_CASE_STUDIES)} case studies")

# %% [markdown]
# ## Rank-1 Cluster Diagnostics
#
# Rather than pre-committing to a single rank-1 configuration per
# case study, we inspect the *cluster* of top configurations on the validation
# split. A signal with genuine predictive structure shows a thick top-of-the-
# distribution: many configurations cluster within a fold-standard-error of
# the rank-1 Sharpe, and the implied pick is insensitive to small perturbations
# in the selection rule. A thin cluster (large gap between rank-1 and rank-N)
# suggests the rank-1 result is closer to a tail draw than a stable optimum.
#
# For each case study we report: the rank-1 Sharpe, the rank-10 Sharpe (if
# 10 configs exist), the spread between them, the mean per-fold Sharpe, and
# the number of folds in which the rank-1 configuration has positive Sharpe.
# These are measurements that feed the downstream narrative.
#
# ### Carrier-selection rule
#
# The validation rank-1 for each case study is the highest-Sharpe validation
# backtest across the three pipeline stages — signal selection, allocation,
# and risk overlay. The deployed holdout carrier is the same full strategy
# spec (signal method, allocation method, risk overlay name) retrained on
# holdout data. When holdout retrain produces no usable backtest at that
# full spec — degenerate predictions, vol-window-vs-history mismatch,
# universe-filter rejection, or other generation failures — the rule falls
# back to the next-highest validation Sharpe with a usable holdout, and so
# on until one succeeds. The `_val_rank1_full_spec` helper implements this
# walk; `query_holdout_rows` and `_holdout_lineage_for` consume its output
# to pin val/holdout pairs to the same full strategy carrier.

# %%
# Label restrictions that align the cluster-diagnostics rank-1 with the
# rank-1 used for Ch20 holdout retrain. sp500_options trains ret_to_expiry
# (HTM, coherent option costs) and four fixed-horizon straddle labels
# (vectorized path with generic bps cost). The Ch20 narrative uses
# ret_to_expiry as the HTM-coherent option-strategy reference; restricting
# cluster-diagnostics to the same label keeps the §20.1 top-cluster numbers
# aligned with the §20.5/§20.6 narrative.
from holdout import LABEL_RESTRICTIONS as _CLUSTER_LABEL_RESTRICTIONS  # noqa: E402

# Rung restrictions that pin the headline rank-1 to a specific execution
# regime. sp500_options is evaluated under the O'Donovan-Yu (2025)
# cost-mitigation cascade:
#   - Rung-1 (naive round-trip): universe_filter="full",  exit_at_max_days=10
#   - Rung-2 (HTM, full):        universe_filter="full",  exit_at_max_days=None
#   - Rung-3 (HTM, liquid q20):  universe_filter="liquid", exit_at_max_days=None
# Rung-2 is the chapter-wide rank-1 for cross-case-study comparisons; Rung-3
# carries the rung-3 cascade per O'Donovan & Yu — full-universe HTM (rung-2)
# is the demoted variant discussed in §20.5 and §18.8; the registered
# strategy is rung-3 HTM+liquid (universe_filter='liquid', the bottom-
# quintile half-spread universe). Both rung-1 (mid-to-mid bps) and rung-2
# (full-universe HTM) carry universe_filter="full", so a `universe_filter`
# filter alone is insufficient — `ORDER BY sharpe DESC LIMIT 1` then
# silently picks whichever rung happens to have the higher Sharpe in
# current data. The pin combines `universe_filter` and `exit_at_max_days`
# so the rank-1 row is deterministic and HTM-coherent. Other case studies
# have no entry here and skip the filter altogether.
_RUNG3_PREDICATE = (pl.col("universe_filter") == "liquid") & pl.col("exit_at_max_days").is_null()

# NASDAQ-100 pin: cost-feasible ensemble, chosen before the holdout was opened
# and matched on those two design attributes, which any registry can satisfy.
_NASDAQ_PREDICATE = (pl.col("universe_filter") == "cost_feasible") & (
    pl.col("family") == "ensemble"
)

_CLUSTER_RUNG_RESTRICTIONS: dict[str, dict[str, object]] = {
    "sp500_options": {
        "predicate": _RUNG3_PREDICATE,
        # Mirrors the polars predicate above for the holdout SQL path and
        # for `progression(...)` calls that still need the scope pin.
        "universe_filter": "liquid",
        "exit_at_max_days": None,
    },
    "nasdaq100_microstructure": {
        "predicate": _NASDAQ_PREDICATE,
        "universe_filter": "cost_feasible",
        "exit_at_max_days": None,
    },
}


def _best_pinned(explorer: "BacktestExplorer", cs: str, stage: str, top_n: int) -> pl.DataFrame:
    """`explorer.best` for a stage, fetching enough rows that a rung-restricted
    cohort survives the post-hoc predicate filter.

    `best()` extracts `universe_filter` from `spec_json` in Python, after the
    SQL `LIMIT top_n`. For nasdaq the pinned cost-feasible carrier sits below
    the full-universe in-sample maxima, so a small `top_n` truncates it before
    `_apply_rung_restriction` runs. Pull all rows for restricted case studies."""
    if cs in _CLUSTER_RUNG_RESTRICTIONS:
        return explorer.best(stage=stage, top_n=1_000_000)
    return explorer.best(stage=stage, top_n=top_n)


def _apply_rung_restriction(df: pl.DataFrame, cs: str) -> pl.DataFrame:
    """Filter `df` to the case study's pinned rung, if one is configured.

    Returns the input untouched if no restriction applies. The helper
    relies on `BacktestExplorer.best()` always emitting both
    `universe_filter` and `exit_at_max_days` columns; if a future
    schema regression drops them, the polars `filter` will raise a
    column-not-found error rather than silently allowing the rank-1
    selection to drift back to the cross-rung max."""
    rung = _CLUSTER_RUNG_RESTRICTIONS.get(cs)
    if rung is None or df.is_empty():
        return df
    return df.filter(rung["predicate"])


# Carrier pin (validation-time a-priori tie-break) — distinct from the rung
# restrictions above. It selects the deployed MODEL carrier when the cross-stage
# val rank-1 is statistically tied with a more diversified / more precisely-
# estimated config. us_firm_characteristics: default_huber (signal-stage rank-1,
# val 2.754, block-bootstrap Sharpe CI [2.33,3.37] width 1.04) is pinned over
# leaves_7_mae (cross-stage rank-1, val 2.759, CI [2.10,3.57] width 1.46) — same
# point estimate, narrower CI, far more diversified deployment (50 equal-weight
# names vs 10 score-weighted; holdout MaxDD -8.6% vs -34%). Pinning on
# config_name (NOT allocator) keeps both equal_weight and score_weighted in the
# §20.5 allocator comparison on the carrier's prediction. Mirrors
# case_studies.utils.strategy_analysis.CARRIER_PINS — keep in sync.
_CARRIER_PIN_PREDICATES: dict[str, pl.Expr] = {
    "us_firm_characteristics": pl.col("config_name") == "default_huber",
}


def _apply_carrier_pin(df: pl.DataFrame, cs: str) -> pl.DataFrame:
    """Restrict candidate rows to the pinned model carrier, if configured.

    Applied alongside `_apply_rung_restriction` at every cross-stage
    carrier-selection site (signal / allocation / cross-stage spine /
    holdout-pairing walk). Deliberately NOT applied to the §20.3 rank-cluster
    diagnostics, which measure the rank-cluster width across the full model
    space and must stay carrier-agnostic.
    """
    pred = _CARRIER_PIN_PREDICATES.get(cs)
    if pred is None or df.is_empty() or "config_name" not in df.columns:
        return df
    return df.filter(pred)


def _progression_for(
    explorer: "BacktestExplorer",
    pred_hash: str,
    cs: str,
) -> pl.DataFrame:
    """Call `progression()` with the case study's rung scope, if any."""
    rung = _CLUSTER_RUNG_RESTRICTIONS.get(cs)
    if rung is None:
        return explorer.progression(pred_hash)
    return explorer.progression(
        pred_hash,
        universe_filter=rung["universe_filter"],
        exit_at_max_days=rung["exit_at_max_days"],
    )


# Stages whose registry numbers should never be reported for a case study,
# either because the strategy makes the stage structurally meaningless (HTM
# short-straddle has no allocator choice, no bps cost sweep) or because the
# legacy registry contains deprecated entries that pre-date the strategy
# redesign. Consumed by both `build_backtest_rows` and the `synthesis_dict`
# sanitizer below so the in-notebook attrition funnel and the JSON artifact
# cannot drift on the same case study.
_STAGES_NOT_APPLICABLE: dict[str, set[str]] = {
    "sp500_options": {"costs", "risk"},
    "us_firm_characteristics": {"risk"},
    "nasdaq100_microstructure": {"allocation", "costs", "risk"},
}

# Per-CS rationale strings for `not_applicable_reason` fields written into
# `synthesis_dict`. Keyed by (cs, stage) so two case studies that skip the
# same stage for different structural reasons render different prose.
_STAGE_NA_REASONS: dict[tuple[str, str], str] = {
    ("sp500_options", "allocation"): ("HTM short-straddle has fixed 1/n_roll cohort weighting"),
    ("sp500_options", "costs"): ("option costs use §18.8 bid-ask accounting, not bps sweep"),
    ("sp500_options", "risk"): ("HTM expiration structure sets risk profile"),
    ("us_firm_characteristics", "risk"): (
        "vectorized-engine path; portfolio overlays purged 2026-05-17"
    ),
    ("nasdaq100_microstructure", "allocation"): (
        "carrier is a signal-stage slot strategy; the slot mechanism is the sizing rule"
    ),
    ("nasdaq100_microstructure", "costs"): (
        "timing-corrected broad carrier cost grid deferred to v3.1"
    ),
    ("nasdaq100_microstructure", "risk"): (
        "timing-corrected broad carrier risk grid deferred to v3.1"
    ),
}


def _stage_applicable(cs: str, stage: str) -> bool:
    """Return False if `cs` has stage `stage` declared not applicable.

    `stage` must be one of the canonical labels used by
    ``_STAGES_NOT_APPLICABLE`` itself (`allocation`, `costs`, `risk`).
    Callers in this notebook always pass canonical literals."""
    return stage not in _STAGES_NOT_APPLICABLE.get(cs, set())


# %%
cluster_rows = []
for cs, explorer in explorers.items():
    top = _best_pinned(explorer, cs, "signal", 200)
    if top.is_empty() or top["sharpe"][0] is None:
        continue
    if "family" in top.columns:
        top = top.filter(pl.col("family") != "benchmark")
    label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
    if label_restriction and "label" in top.columns:
        top = top.filter(pl.col("label").is_in(list(label_restriction)))
    top = _apply_rung_restriction(top, cs)
    if top.is_empty() or top["sharpe"][0] is None:
        continue
    rank1 = top["sharpe"][0]
    rank10 = top["sharpe"][9] if top.height >= 10 else None
    spread = (rank1 - rank10) if rank10 is not None else None
    # Fold-level stability for the rank-1 backtest
    try:
        bt_hash = top["backtest_hash"][0]
        fold_df = explorer.fold_performance(bt_hash)
        if not fold_df.is_empty():
            mean_fold_sh = float(fold_df["sharpe"].mean())
            se_fold_sh = (
                float(fold_df["sharpe"].std(ddof=1) / (fold_df.height**0.5))
                if fold_df.height > 1
                else None
            )
            n_folds_pos = int(fold_df.filter(pl.col("sharpe") > 0).height)
            n_folds = fold_df.height
        else:
            mean_fold_sh, se_fold_sh, n_folds_pos, n_folds = None, None, 0, 0
    except Exception:
        mean_fold_sh, se_fold_sh, n_folds_pos, n_folds = None, None, 0, 0

    cluster_rows.append(
        {
            "case_study": DISPLAY_NAMES.get(cs, cs),
            "cs_id": cs,
            "rank1_sharpe": rank1,
            "rank10_sharpe": rank10,
            "rank1_rank10_spread": spread,
            "mean_per_fold_sharpe": mean_fold_sh,
            "fold_sharpe_se": se_fold_sh,
            "n_folds_pos": n_folds_pos,
            "n_folds": n_folds,
            "n_configs": int(top.height),
        }
    )

cluster_df = pl.DataFrame(cluster_rows)
if not cluster_df.is_empty():
    print("\n=== Rank-1 Cluster Diagnostics (validation) ===")
    print(
        cluster_df.select(
            "case_study",
            "rank1_sharpe",
            "rank10_sharpe",
            "rank1_rank10_spread",
            "mean_per_fold_sharpe",
            "fold_sharpe_se",
            "n_folds_pos",
            "n_folds",
            "n_configs",
        )
    )

# %% [markdown]
# Read the table as a tuple: (rank1 Sharpe, rank10 Sharpe, spread, fold-SE,
# folds-positive). A small rank1→rank10 spread relative to the fold-SE signals
# a thick top-of-distribution. Folds-positive close to the total fold count
# signals temporal stability. Both can be read off per case study without
# collapsing the evidence into a single label.

# %% [markdown]
# ## Overview Table
#
# The nine case studies differ in asset class, rebalancing frequency, universe
# size, and the cost assumption each one carries. The table records those four
# properties so that a later result can be attributed to the setting it was
# measured in rather than to the model that produced it.

# %%
overview_rows = []
for cs, explorer in explorers.items():
    setup = configs.get(cs, {})
    cost_bps = compute_cost_bps(setup)
    universe = setup.get("universe", {})
    # A futures universe is sized in products rather than in assets, so
    # cme_futures declares `n_products` where the others declare `n_assets`.
    # Reading only the latter reported that case study as an empty universe.
    n_assets = (
        universe.get("n_assets")
        or universe.get("n_products")
        or len(universe.get("symbols", []))
        or 0
    )
    primary_label = setup.get("labels", {}).get("primary", "")

    families = explorer.compare_families(stage="signal")

    overview_rows.append(
        {
            "case_study": DISPLAY_NAMES.get(cs, cs),
            "cs_id": cs,
            "asset_class": ASSET_CLASS_MAP.get(cs, "unknown"),
            "frequency": FREQ_MAP.get(cs, "daily"),
            "universe": n_assets,
            "primary_label": primary_label,
            "cost_bps": cost_bps,
            "n_model_families": len(families) if not families.is_empty() else 0,
        }
    )

overview_df = pl.DataFrame(overview_rows)
overview_df.select("case_study", "asset_class", "frequency", "universe", "cost_bps")

# %% [markdown]
# The test bed covers equity ETFs, crypto perpetuals, intraday microstructure,
# equity+options, firm characteristics, FX, futures, pure options, and a broad
# equity panel. Cost assumptions range from 6.5 bps (S&P 500 Eq+Opt) to
# 12.5 bps (US Firms, US Equities), reflecting the diversity of transaction
# cost regimes across asset classes.

# %% [markdown]
# ## Model IC Comparison
#
# Mean IC by model family across case studies, queried from `prediction_metrics`.
# Each cell shows the average IC across all configurations within a family,
# filtered to each case study's primary label.

# %%
ic_rows = []
for cs, explorer in explorers.items():
    case_dir = get_case_study_dir(cs)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        continue

    # Filter by primary label so IC values match book prose
    primary_label = configs.get(cs, {}).get("labels", {}).get("primary", "")

    db = sqlite3.connect(str(db_path))
    # Best IC per family on primary label only
    # Exclude causal_dml: it estimates treatment effects, not predictive IC.
    # NOTE: best_ic and best_ic_daily are independent per-family MAXes — they may
    # come from *different* predictions. This is intentional ("best daily IC in
    # the family"), not "the daily IC of the best-by-fold model".
    query = """
        SELECT t.family, MAX(pm.ic_mean) AS best_ic,
               MAX(pm.ic_mean_daily) AS best_ic_daily,
               AVG(pm.ic_mean) AS mean_ic, COUNT(*) AS n_preds
        FROM training_runs t
        JOIN prediction_sets p ON t.training_hash = p.training_hash
        JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
        WHERE p.split != 'holdout'
          AND pm.ic_mean IS NOT NULL
          AND t.family != 'causal_dml'
    """
    params: tuple = ()
    if primary_label:
        query += "      AND t.label = ?\n"
        params = (primary_label,)
    query += "    GROUP BY t.family"

    rows = db.execute(query, params).fetchall()
    db.close()

    for family, best_ic, best_ic_daily, mean_ic, n_preds in rows:
        ic_rows.append(
            {
                "case_study": DISPLAY_NAMES.get(cs, cs),
                "family": family,
                "ic_mean": mean_ic,
                "ic_best": best_ic,
                "ic_best_daily": best_ic_daily,
                "n_predictions": n_preds,
            }
        )

ic_df = pl.DataFrame(ic_rows)

# %%
if not ic_df.is_empty():
    ic_pivot = ic_df.pivot(on="family", index="case_study", values="ic_mean").sort("case_study")
else:
    ic_pivot = pl.DataFrame()

# %% [markdown]
# ### Mean IC by Model Family

# %%
ic_pivot

# %% [markdown]
# No single model family dominates across all nine case studies. GBM tends
# to produce the best or near-best IC in most datasets (especially futures
# and options), but linear models lead for ETFs. Negative mean ICs (e.g.,
# FX Pairs across all families) flag case studies where prediction is
# genuinely difficult. S&P 500 Options shows the highest raw ICs, but
# single-name option execution costs materially compress the translated
# Sharpe — §20.5 discusses how that compression plays out per variant.

# %% [markdown]
# ## Backtest Comparison
#
# Cross-dataset comparison of pipeline outcomes. Each row picks the best
# result at each stage **independently** — the best signal may come from a
# different model than the best allocation.


# %%
def build_backtest_rows():
    """Build backtest comparison rows from all case study explorers."""
    bt_rows = []
    for cs, explorer in explorers.items():
        summary = explorer.summary()

        # Best signal-stage result — exclude benchmark families (equal_weight,
        # etc.) since §20.4's model comparison is about trained models, not
        # passive baselines. Also apply case-study label and universe-filter
        # restrictions so the Ch20 rank-1 is HTM-coherent for sp500_options
        # and pinned to the Rung-2 full-universe baseline (the mitigated
        # Rung-3 liquid-subset variant is reported separately in §20.5).
        label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
        signal_candidates = _best_pinned(explorer, cs, "signal", 200)
        if not signal_candidates.is_empty() and "family" in signal_candidates.columns:
            signal_candidates = signal_candidates.filter(pl.col("family") != "benchmark")
        if (
            label_restriction
            and "label" in signal_candidates.columns
            and not signal_candidates.is_empty()
        ):
            signal_candidates = signal_candidates.filter(
                pl.col("label").is_in(list(label_restriction))
            )
        signal_candidates = _apply_rung_restriction(signal_candidates, cs)
        signal_candidates = _apply_carrier_pin(signal_candidates, cs)
        best_signal = signal_candidates.head(1)
        signal_sharpe = best_signal["sharpe"][0] if not best_signal.is_empty() else None
        best_source = best_signal["source"][0] if not best_signal.is_empty() else ""

        # Carrier-pred pin for cost/risk. Case studies with a rung restriction
        # (nasdaq cost-feasible ensemble) carry their headline cost/risk on the
        # carrier prediction only; the full-universe sweep rows are the
        # Ch18/Ch19 cost-defeat demonstration and must not pool into the
        # cross-case comparison. Other case studies pass None (no pin) and keep
        # the registry-wide aggregation unchanged.
        carrier_pred = (
            best_signal["prediction_hash"][0]
            if cs in _CLUSTER_RUNG_RESTRICTIONS and not best_signal.is_empty()
            else None
        )

        # Best allocation-stage result (same filters). For case studies that
        # declare the allocation stage not applicable (e.g. sp500_options HTM),
        # the registry numbers come from deprecated runs, so report None to
        # match the synthesis_dict sanitizer below.
        if _stage_applicable(cs, "allocation"):
            alloc_candidates = explorer.best(stage="allocation", top_n=200)
            if not alloc_candidates.is_empty() and "family" in alloc_candidates.columns:
                alloc_candidates = alloc_candidates.filter(pl.col("family") != "benchmark")
            if (
                label_restriction
                and "label" in alloc_candidates.columns
                and not alloc_candidates.is_empty()
            ):
                alloc_candidates = alloc_candidates.filter(
                    pl.col("label").is_in(list(label_restriction))
                )
            alloc_candidates = _apply_rung_restriction(alloc_candidates, cs)
            alloc_candidates = _apply_carrier_pin(alloc_candidates, cs)
            best_alloc = alloc_candidates.head(1)
            alloc_sharpe = best_alloc["sharpe"][0] if not best_alloc.is_empty() else None
            # When a carrier pin is active, the allocator comparison must run on
            # the pinned carrier's prediction so the reported best_allocator NAME
            # matches the pinned alloc_sharpe. Without this, compare_allocators
            # pools across every prediction (e.g. an experimental conformal run on
            # a non-carrier model) and can return an allocator that was never run
            # on the deployed carrier. Non-pinned CSes pass None (unchanged).
            alloc_comp_pred = (
                best_alloc["prediction_hash"][0]
                if cs in _CARRIER_PIN_PREDICATES and not best_alloc.is_empty()
                else None
            )
            alloc_comp = explorer.compare_allocators(prediction_hash=alloc_comp_pred)
            best_allocator = alloc_comp["allocator"][0] if not alloc_comp.is_empty() else ""
        else:
            alloc_sharpe = None
            best_allocator = ""

        # Cost sensitivity (gated by stage policy)
        survives_costs = None
        if _stage_applicable(cs, "costs"):
            cost_df = explorer.cost_sensitivity(prediction_hash=carrier_pred)
            if not cost_df.is_empty():
                zero_cost = cost_df.filter(pl.col("cost_bps") == 0)
                survives_costs = not zero_cost.is_empty() and zero_cost["sharpe"].max() > 0

        # Risk overlay (gated by stage policy)
        best_overlay = ""
        managed_sharpe = None
        if _stage_applicable(cs, "risk"):
            risk_df = explorer.risk_impact(prediction_hash=carrier_pred)
            if not risk_df.is_empty():
                best_risk_row = risk_df.sort("sharpe", descending=True).head(1)
                best_overlay = best_risk_row["risk_name"][0]
                managed_sharpe = best_risk_row["sharpe"][0]

        # Spine rank-1 prediction_hash — cross-stage rank-1 across
        # signal/allocation/risk_overlay, matching the paired-bootstrap
        # cross-stage leader logic below (and Ch20 prose Tables 20.5–20.7).
        # Without this, Figure 20.7 can read off a different prediction than
        # the prose tables.
        cross_stage = pl.concat(
            [_best_pinned(explorer, cs, s, 2000) for s in ("signal", "allocation", "risk_overlay")],
            how="diagonal_relaxed",
        )
        if not cross_stage.is_empty() and "family" in cross_stage.columns:
            cross_stage = cross_stage.filter(pl.col("family") != "benchmark")
        if label_restriction and "label" in cross_stage.columns and not cross_stage.is_empty():
            cross_stage = cross_stage.filter(pl.col("label").is_in(list(label_restriction)))
        cross_stage = _apply_rung_restriction(cross_stage, cs)
        cross_stage = _apply_carrier_pin(cross_stage, cs)
        if not cross_stage.is_empty():
            cross_stage = cross_stage.sort("sharpe", descending=True).unique(
                subset=["prediction_hash"], keep="first", maintain_order=True
            )
        spine_pred_hash = cross_stage["prediction_hash"][0] if not cross_stage.is_empty() else None

        bt_rows.append(
            {
                "case_study": DISPLAY_NAMES.get(cs, cs),
                "case_study_id": cs,
                "spine_prediction_hash": spine_pred_hash,
                "n_signal": summary.get("signal", 0),
                "n_allocation": summary.get("allocation", 0),
                "n_cost": summary.get("cost_sensitivity", 0),
                "n_risk": summary.get("risk_overlay", 0),
                "best_source": best_source,
                "signal_sharpe": signal_sharpe,
                "best_allocator": best_allocator,
                "alloc_sharpe": alloc_sharpe,
                "survives_costs": survives_costs,
                "best_overlay": best_overlay,
                "managed_sharpe": managed_sharpe,
            }
        )
    return bt_rows


# %%
bt_rows = build_backtest_rows()

# %%
bt_df = pl.DataFrame(bt_rows)
print("\nPipeline Comparison:")
print(
    bt_df.select(
        "case_study",
        "signal_sharpe",
        "alloc_sharpe",
        "survives_costs",
        "managed_sharpe",
    )
)

# %% [markdown]
# Eight of nine case studies produce positive signal-stage Sharpe; only
# FX Pairs enters the pipeline negative (-0.00) and stays marginal through
# allocation. US Firms and Crypto Perps post the highest signal-stage Sharpes
# (2.75, 2.09), and US Firms carries that signal forward to a 1.77 holdout.
# ETFs progresses 0.89 → 1.03 → 1.08 → 1.21 across signal → allocation →
# cost_sensitivity → risk_overlay, with each stage adding incremental Sharpe
# for that case study. Of the six case studies whose val rank-1 lands at the
# risk-overlay stage, every managed Sharpe exceeds 1. NASDAQ-100 is excluded
# from that comparison in v3.0 because its timing-corrected broad cost and risk
# grids are deferred to v3.1.

# %% [markdown]
# ## Paired-Bootstrap Comparison vs Equal-Weight Benchmark
#
# Each case study's rank-1 signal-stage backtest (with the same label,
# universe-filter, and rung restrictions used for the cluster diagnostics) is
# compared to its equal-weight benchmark using a **paired stationary block
# bootstrap on daily strategy returns**. Block length is derived from
# ``setup.yaml.labels.{label}.rebalance_step`` (falling back to the optimal
# block size, never below the label horizon). Reported quantities:
#
# - ``sharpe_diff`` with 95 % bootstrap CI
# - ``ret_diff`` (annualized return difference) with 95 % CI
# - ``info_ratio`` of the daily-return difference
# - ``prob_challenger_wins`` — bootstrap fraction in which challenger Sharpe
#   exceeds the benchmark
# - ``p_value`` — two-sided bootstrap p-value for ``sharpe_diff = 0``
#
# Results land in ``backtest_paired_metrics`` (per case study) and roll up
# into the cross-dataset table below. This is the right unit of uncertainty
# for the headline rank-1 claim: not the Sharpe alone, but the Sharpe
# **difference vs the passive baseline that experienced the same market
# conditions**.


# %%
def _benchmark_returns_from_artifact(
    cs: str, label: str, period: str = "overall"
) -> tuple[str, pl.DataFrame, str] | None:
    """Resolve the side-artifact equal-weight benchmark for ``(cs, label)``.

    The benchmark is the daily-MTM EW reference series persisted by
    ``scripts/compute_vectorized_ew_benchmark.py`` at
    ``case_studies/{cs}/benchmark/{label}.parquet``. Single, well-defined
    methodology per (cs, label) — no universe/rung/cadence ambiguity that
    a registry-side ``family='benchmark'`` lookup would have to disambiguate.

    ``period`` selects the time window slice (``"overall"`` or ``"holdout"``)
    applied by ``load_benchmark_returns``. Classification-label fallback to
    the matching ``fwd_ret_*`` artifact applies in both periods.

    Returns ``(synthetic_hash, returns_df, resolved_label)`` where
    ``synthetic_hash`` is a deterministic identifier safe to use as the PK
    column in ``backtest_paired_metrics`` (which has no FK on
    ``benchmark_hash``) and ``resolved_label`` is the actual label whose
    artifact was loaded — equal to ``label`` unless the classification
    fallback fired, in which case it's the matching ``fwd_ret_*`` label.
    Returns ``None`` if the artifact is missing.
    """
    df = load_benchmark_returns(cs, label, period=period)
    bench_label = label
    if df.is_empty() or "ew_return" not in df.columns:
        # Fallback: classification labels (``fwd_class_*``, ``fwd_dir_*``,
        # ``fwd_tb_*``, ``fwd_carry_*``) share the same forecast window as
        # their continuous counterpart (``fwd_ret_*``). The EW universe over
        # the same window is identical regardless of the label being
        # predicted, so map e.g. ``fwd_class_1m`` to ``fwd_ret_1m``.
        fallback = None
        for prefix in ("fwd_class_", "fwd_dir_", "fwd_tb_", "fwd_carry_"):
            if label.startswith(prefix):
                fallback = "fwd_ret_" + label[len(prefix) :]
                break
        if fallback is None:
            return None
        df = load_benchmark_returns(cs, fallback, period=period)
        if df.is_empty() or "ew_return" not in df.columns:
            return None
        bench_label = fallback
    suffix = "" if period == "overall" else f":{period}"
    bench_hash = f"side_ew:{cs}:{bench_label}{suffix}"
    return (
        bench_hash,
        df.select(
            pl.col("timestamp").cast(pl.Date).alias("timestamp"),
            pl.col("ew_return").cast(pl.Float64).alias("ret"),
        ),
        bench_label,
    )


def _aligned_returns(cs: str, h: str) -> pl.DataFrame | None:
    """Load and normalize a backtest's daily returns; columns ``[timestamp, ret]``."""
    parquet = get_case_study_dir(cs) / "run_log" / "backtest" / h / "daily_returns.parquet"
    if not parquet.exists():
        return None
    df = pl.read_parquet(parquet)
    ret_col = next(
        (c for c in ("daily_return", "ret", "return", "value") if c in df.columns),
        df.columns[-1],
    )
    ts_col = next(
        (c for c in ("timestamp", "date", "datetime") if c in df.columns),
        df.columns[0],
    )
    return df.select(
        pl.col(ts_col).cast(pl.Date).alias("timestamp"),
        pl.col(ret_col).cast(pl.Float64).alias("ret"),
    )


# %%
import numpy as np

from case_studies.utils.uncertainty import (
    SIGNAL_BASELINE_BY_CASE_STUDY,
    compute_independent_diff_uncertainty,
    compute_paired_uncertainty,
)


def _min_paired_n(ppy: int) -> int:
    """Minimum series length for paired-bootstrap stability, frequency-aware.

    The ~21 floor was written for daily cadences (about a month of obs).
    Monthly case studies (e.g. ``us_firm_characteristics``) have ~12 holdout
    observations by design, and ``compute_paired_uncertainty`` runs cleanly
    on n=12. Scale the floor with ``ppy`` so monthly/weekly CSs aren't
    blocked by a daily-tuned guard.
    """
    if ppy <= 12:  # monthly
        return 6
    if ppy <= 52:  # weekly
        return 12
    return 21  # daily / 8h / intraday


def _joint_coerce(c_arr, b_arr):
    """Filter NaN/non-finite jointly across paired series and trim leading
    rows where *either* is zero. Matches ``_coerce_returns`` semantics but
    preserves index alignment so ``compute_paired_uncertainty``'s
    equal-length precondition survives — the upstream helper trims leading
    zeros independently per series, which can desynchronize a paired
    bootstrap if one side has more leading inactive bars than the other.
    """
    c = np.asarray(c_arr, dtype=np.float64)
    b = np.asarray(b_arr, dtype=np.float64)
    finite = np.isfinite(c) & np.isfinite(b)
    c, b = c[finite], b[finite]
    if c.size == 0:
        return c, b
    nonzero = np.flatnonzero((c != 0.0) & (b != 0.0))
    if nonzero.size == 0:
        return c[:0], b[:0]
    start = int(nonzero[0])
    return c[start:], b[start:]


# Distinguish skipped CSs from real failures so empty cross-dataset rollups
# aren't indistinguishable from a silent crash.
paired_rows: list[dict] = []
paired_skips: list[dict] = []

for cs, explorer in explorers.items():
    label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
    # Cross-stage rank-1 (signal/allocation/risk_overlay), mirroring
    # holdout.py::HOLDOUT_SELECTION_STAGES. Dedup by prediction_hash so the
    # leader corresponds to a distinct trained model.
    cand = pl.concat(
        [explorer.best(stage=s, top_n=2000) for s in ("signal", "allocation", "risk_overlay")],
        how="diagonal_relaxed",
    )
    if cand.is_empty() or "backtest_hash" not in cand.columns:
        paired_skips.append({"case_study": cs, "reason": "no_signal_stage_candidates"})
        continue
    if "family" in cand.columns:
        cand = cand.filter(pl.col("family") != "benchmark")
    if label_restriction and "label" in cand.columns:
        cand = cand.filter(pl.col("label").is_in(list(label_restriction)))
    cand = _apply_rung_restriction(cand, cs)
    if cand.is_empty():
        paired_skips.append({"case_study": cs, "reason": "no_candidates_after_restriction"})
        continue
    cand = cand.sort("sharpe", descending=True).unique(
        subset=["prediction_hash"], keep="first", maintain_order=True
    )

    leader_hash = cand["backtest_hash"][0]
    leader_label = cand["label"][0] if "label" in cand.columns else None
    if not leader_label:
        paired_skips.append({"case_study": cs, "reason": "no_label_on_leader"})
        continue
    bench_resolution = _benchmark_returns_from_artifact(cs, leader_label)
    if not bench_resolution:
        paired_skips.append(
            {"case_study": cs, "reason": f"no_benchmark_artifact_for_label:{leader_label}"}
        )
        continue
    benchmark_hash, base, resolved_bench_label = bench_resolution

    chal = _aligned_returns(cs, leader_hash)
    if chal is None:
        paired_skips.append({"case_study": cs, "reason": "no_challenger_returns_parquet"})
        continue

    ppy = {"daily": 252, "weekly": 52, "monthly": 12, "8h": 1095}.get(
        FREQ_MAP.get(cs, "daily"), 252
    )
    min_n = _min_paired_n(ppy)
    aligned = chal.join(base, on="timestamp", how="inner", suffix="_b")
    if aligned.height < min_n:
        paired_skips.append(
            {"case_study": cs, "reason": f"insufficient_overlap:n={aligned.height}"}
        )
        continue
    c_arr, b_arr = _joint_coerce(aligned["ret"].to_numpy(), aligned["ret_b"].to_numpy())
    if c_arr.size < min_n:
        paired_skips.append(
            {"case_study": cs, "reason": f"insufficient_after_coerce:n={c_arr.size}"}
        )
        continue
    paired = compute_paired_uncertainty(
        c_arr,
        b_arr,
        periods_per_year=ppy,
        case_study=cs,
        label=leader_label,
        n_boot=2000,
        seed=42,
    )
    if not paired:
        paired_skips.append({"case_study": cs, "reason": "paired_uncertainty_empty"})
        continue

    # Side-artifact benchmark — deterministic across (cs, label), no
    # universe/rung ambiguity, no fallback-by-recency.
    benchmark_kind = f"{SIGNAL_BASELINE_BY_CASE_STUDY.get(cs, 'equal_weight')}_side_artifact"
    paired_rows.append(
        {
            "case_study": DISPLAY_NAMES.get(cs, cs),
            "label": leader_label,
            "benchmark_label": resolved_bench_label,  # may differ from leader_label when classification fallback fired
            "sharpe_diff": paired.get("sharpe_diff"),
            "sharpe_diff_ci_lo": paired.get("sharpe_diff_ci95_lo"),
            "sharpe_diff_ci_hi": paired.get("sharpe_diff_ci95_hi"),
            "ret_diff": paired.get("ret_diff"),
            "info_ratio": paired.get("info_ratio"),
            "p_value": paired.get("p_value"),
            "prob_wins": paired.get("prob_challenger_wins"),
            "block": paired.get("bootstrap_block_length"),
            "n_boot": paired.get("bootstrap_n"),
        }
    )

paired_df = pl.DataFrame(paired_rows)
if not paired_df.is_empty():
    print("\n=== Paired Bootstrap: rank-1 vs equal-weight ===")
    print(
        paired_df.select(
            "case_study",
            "label",
            "benchmark_label",
            "sharpe_diff",
            "sharpe_diff_ci_lo",
            "sharpe_diff_ci_hi",
            "info_ratio",
            "prob_wins",
            "p_value",
        )
    )
else:
    print("\n=== Paired Bootstrap: rank-1 vs equal-weight ===")
    print("No paired-bootstrap rows produced — see skip table below for reasons.")

if paired_skips:
    print("\nSkipped case studies:")
    for s in paired_skips:
        print(f"  - {s['case_study']:<32}  {s['reason']}")

# Loud invariant — a 0/N or all-skipped outcome is now obvious in the
# notebook output instead of buried under "no paired-bootstrap rows."
print(f"\npaired={len(paired_rows)}/{len(explorers)}, skipped={len(paired_skips)}/{len(explorers)}")

# %% [markdown]
# Read each row as: rank-1 challenger annualized Sharpe **minus** equal-weight
# benchmark Sharpe, with a 95 % CI from the paired stationary block bootstrap
# on the daily-return difference; the information ratio summarizes the
# excess-return-to-tracking-error ratio; ``prob_wins`` is the fraction of
# bootstrap resamples in which the challenger beat the benchmark; ``p_value``
# tests ``H0: sharpe_diff = 0``. A confident "the model adds skill over the
# passive baseline" claim requires (i) the CI excludes zero, (ii) ``prob_wins``
# close to 1, and (iii) a small ``p_value``. Cases where the CI straddles
# zero are not failures — they signal that the apparent Sharpe gap is within
# block-bootstrap sampling error and should be reported as such.

# %% [markdown]
# ## Paired metrics — full coverage for strategy-analysis notebook
#
# The block above populates pair type #1 (signal rank-1 vs equal-weight,
# overall window). The strategy-analysis notebook (per-CS strategy notebooks)
# requires five additional pair types per case study to render §2 (stage-
# transition waterfall), §6 (holdout decay + holdout-vs-benchmark) and §7
# (benchmark-aware diagnostics) without inline bootstrap recomputation.
#
# The pair set:
#
# 1. signal rank-1 (overall) ↔ equal-weight (overall) — populated above
# 2. signal rank-1 (holdout) ↔ equal-weight (holdout window)
# 3. holdout rank-1 ↔ validation rank-1 (same lineage decay; min-length
#    truncation since the windows are disjoint)
# 4. allocation rank-1 ↔ signal rank-1 (same window, stage transition)
# 5. cost-sensitivity rank-1 ↔ allocation rank-1 (same window)
# 6. risk-overlay rank-1 ↔ cost-sensitivity rank-1 (same window)
#
# Pair #3 truncates both series to ``min(len(val), len(ho))`` to satisfy
# ``compute_paired_uncertainty``'s equal-length precondition. The CI is
# interpreted as bootstrap resampling Sharpe in each window independently
# and taking the difference; the truncation is preserved in the
# ``benchmark_kind`` value (``val_rank1_self`` always carries the truncation
# caveat). All pairs use the same paired stationary block bootstrap helper.


# %%
def _full_strategy_spec_from_backtest(db: sqlite3.Connection, bt_hash: str) -> dict | None:
    """Pull the full strategy spec dict (signal + allocation + risk) from
    `bt_hash`'s spec_json. Returns None if the row is missing or signal has
    no `method` field.

    The carrier of a backtest is the tuple (signal, allocation, risk). Pinning
    the val→holdout pair on this full spec keeps the comparison apples-to-
    apples; pinning on signal alone allows MAX(sharpe) to surface a holdout
    row with a different allocation (e.g. conformal_weighted) or risk overlay
    than the validation rank-1 carrier.
    """
    row = db.execute(
        "SELECT spec_json FROM backtest_runs WHERE backtest_hash = ?",
        (bt_hash,),
    ).fetchone()
    if not row:
        return None
    strat = json.loads(row[0]).get("strategy", {})
    sig = strat.get("signal", {})
    if not sig.get("method"):
        return None
    alloc = strat.get("allocation") or {}
    risk = strat.get("risk") or {}
    return {
        "signal": {
            "method": sig.get("method"),
            "top_k": sig.get("top_k"),
            "percentile": sig.get("percentile"),
        },
        "allocation": {
            "method": alloc.get("method"),
            "top_k": alloc.get("top_k"),
            "long_short": alloc.get("long_short"),
        },
        "risk": {
            "name": risk.get("name"),
        },
    }


def _val_rank1_full_spec(cs: str) -> dict | None:
    """Return the val rank-1 *full strategy* spec for ``cs`` — the
    highest-Sharpe validation backtest across (signal, allocation,
    risk_overlay) stages — walking candidates by val Sharpe descending until
    one with a matching holdout backtest at the SAME full spec is found.

    Implements the carrier-selection rule documented in §20.1: the deployed
    holdout for each case study is the val rank-1 across all three pipeline
    stages, retrained on holdout data; when retrain produces no usable
    holdout at that full spec (degenerate predictions, vol-window mismatch,
    universe filter rejection, etc.) the walk falls through to the next
    candidate by val Sharpe. The first val candidate with a registered
    holdout backtest at the same (signal, allocation, risk) tuple defines
    the apples-to-apples carrier pair.

    Returns None when no val candidate up to rank ~200 has a matching
    holdout under the case study's label / rung restrictions.
    """
    explorer = explorers.get(cs)
    if explorer is None:
        return None
    cand = pl.concat(
        [explorer.best(stage=s, top_n=2000) for s in ("signal", "allocation", "risk_overlay")],
        how="diagonal_relaxed",
    )
    if cand.is_empty() or "backtest_hash" not in cand.columns:
        return None
    if "family" in cand.columns:
        cand = cand.filter(pl.col("family") != "benchmark")
    label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
    if label_restriction and "label" in cand.columns:
        cand = cand.filter(pl.col("label").is_in(list(label_restriction)))
    cand = _apply_rung_restriction(cand, cs)
    cand = _apply_carrier_pin(cand, cs)
    if cand.is_empty():
        return None
    # Do NOT dedup by prediction_hash here. The walk needs to surface every
    # registered (signal, allocation, risk_overlay) tuple — when the val
    # rank-1 carrier has no matching holdout retrain but a same-prediction
    # lower-sharpe variant (different allocator or risk overlay) does, the
    # dedup would silently jump to a *different* prediction instead of
    # accepting the same-prediction variant as the apples-to-apples carrier.
    cand = cand.sort("sharpe", descending=True)

    case_dir = get_case_study_dir(cs)
    db_path = case_dir / "run_log" / "registry.db"
    rung = _CLUSTER_RUNG_RESTRICTIONS.get(cs)
    db = sqlite3.connect(str(db_path))
    try:
        for i in range(min(cand.height, 200)):
            bt_hash = cand["backtest_hash"][i]
            spec = _full_strategy_spec_from_backtest(db, bt_hash)
            if spec is None:
                continue
            spec_clauses, spec_params = _full_strategy_clauses(spec)
            ho_clauses = ["p.split = 'holdout'"] + spec_clauses
            ho_params: list[object] = list(spec_params)
            if label_restriction:
                placeholders = ",".join("?" for _ in label_restriction)
                ho_clauses.append(f"t.label IN ({placeholders})")
                ho_params.extend(sorted(label_restriction))
            if rung is not None:
                ho_clauses.append(
                    "COALESCE(json_extract(b.spec_json, '$.strategy.signal.universe_filter'), 'full') = ?"
                )
                ho_params.append(rung["universe_filter"])
                if rung["exit_at_max_days"] is None:
                    ho_clauses.append(
                        "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') IS NULL"
                    )
                else:
                    ho_clauses.append(
                        "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') = ?"
                    )
                    ho_params.append(rung["exit_at_max_days"])
            row = db.execute(
                f"""
                SELECT 1 FROM prediction_sets p
                JOIN training_runs t ON p.training_hash = t.training_hash
                JOIN backtest_runs b ON p.prediction_hash = b.prediction_hash
                                     AND b.stage IN ('signal','allocation','risk_overlay','holdout')
                JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
                WHERE {" AND ".join(ho_clauses)}
                  AND bm.sharpe IS NOT NULL
                LIMIT 1
                """,
                ho_params,
            ).fetchone()
            if row:
                return spec
    finally:
        db.close()
    return None


def _full_strategy_clauses(spec: dict | None) -> tuple[list[str], list[object]]:
    """Build SQL WHERE clauses + params that pin a backtest row to the full
    strategy spec (signal + allocation + risk). Empty list returned when
    spec is None (no constraint).

    Pinning on the full spec ensures `MAX(sharpe)` over candidate holdout
    backtests cannot surface a different allocator (e.g. conformal_weighted
    when val rank-1 was score_weighted) or a different risk overlay than
    the validation carrier — the val→holdout pair stays apples-to-apples
    on the full pipeline configuration, not just the signal.
    """
    if not spec:
        return [], []
    clauses: list[str] = []
    params: list[object] = []

    sig = spec.get("signal") or {}
    method = sig.get("method")
    if method is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.signal.method') IS NULL")
    else:
        clauses.append("json_extract(b.spec_json, '$.strategy.signal.method') = ?")
        params.append(method)
    top_k = sig.get("top_k")
    if top_k is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.signal.top_k') IS NULL")
    else:
        clauses.append("CAST(json_extract(b.spec_json, '$.strategy.signal.top_k') AS INTEGER) = ?")
        params.append(int(top_k))
    pct = sig.get("percentile")
    if pct is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.signal.percentile') IS NULL")
    else:
        clauses.append(
            "CAST(json_extract(b.spec_json, '$.strategy.signal.percentile') AS REAL) = ?"
        )
        params.append(float(pct))

    alloc = spec.get("allocation") or {}
    am = alloc.get("method")
    if am is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.allocation.method') IS NULL")
    else:
        clauses.append("json_extract(b.spec_json, '$.strategy.allocation.method') = ?")
        params.append(am)
    ak = alloc.get("top_k")
    if ak is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.allocation.top_k') IS NULL")
    else:
        clauses.append(
            "CAST(json_extract(b.spec_json, '$.strategy.allocation.top_k') AS INTEGER) = ?"
        )
        params.append(int(ak))
    als = alloc.get("long_short")
    if als is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.allocation.long_short') IS NULL")
    else:
        clauses.append(
            "CAST(json_extract(b.spec_json, '$.strategy.allocation.long_short') AS INTEGER) = ?"
        )
        params.append(int(bool(als)))

    risk = spec.get("risk") or {}
    risk_name = risk.get("name")
    if risk_name is None:
        clauses.append("json_extract(b.spec_json, '$.strategy.risk.name') IS NULL")
    else:
        clauses.append("json_extract(b.spec_json, '$.strategy.risk.name') = ?")
        params.append(risk_name)

    return clauses, params


def _holdout_lineage_for(
    cs: str,
    leader_label: str,
    strategy_spec: dict | None = None,
    *,
    prefer_training_hash: str | None = None,
) -> dict | None:
    """Return ``{backtest_hash, prediction_hash, family, config_name, label}``
    for the highest-Sharpe holdout backtest registered in this case study,
    honoring per-CS cluster restrictions but **not** the leader's label.

    Note: ``leader_label`` is intentionally unused in the SQL — kept in the
    signature for call-site symmetry with ``_val_backtest_for_lineage`` and
    so call sites remain self-documenting about which validation leader the
    holdout pairs against.

    Returns the holdout's *own* label so callers can pair it against
    matching benchmarks. The decoupling matters when ``generate_holdout``'s
    degeneracy fallback accepts a candidate on a different label than the
    validation rank-1 (e.g., crypto's cross-stage rank-1 — over signal,
    allocation, and risk_overlay — runs on fwd_ret_24h but the next fall-
    through candidate runs on fwd_ret_8h). A label-restricted query would
    silently miss the holdout and leave val_rank1_self /
    equal_weight_holdout_side_artifact pairs unpopulated.

    The label_restriction (e.g., sp500_options pinned to ret_to_expiry)
    is applied to the holdout query so we don't pick up cross-rung
    holdouts in CSs with HTM-coherent label scoping.

    When ``strategy_spec`` is provided, the holdout pick is restricted to
    backtests with the same full (signal, allocation, risk) tuple as val's
    rank-1 carrier, so the val→holdout comparison stays apples-to-apples
    on the full pipeline configuration, not just the signal block.
    """
    case_dir = get_case_study_dir(cs)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return None

    label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
    rung = _CLUSTER_RUNG_RESTRICTIONS.get(cs)
    clauses = ["p.split = 'holdout'"]
    params: list[object] = []
    if label_restriction:
        placeholders = ",".join("?" for _ in label_restriction)
        clauses.append(f"t.label IN ({placeholders})")
        params.extend(label_restriction)
    if rung is not None:
        clauses.append(
            "COALESCE(json_extract(b.spec_json, '$.strategy.signal.universe_filter'), 'full') = ?"
        )
        params.append(rung["universe_filter"])
        if rung["exit_at_max_days"] is None:
            clauses.append(
                "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') IS NULL"
            )
        else:
            clauses.append("json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') = ?")
            params.append(rung["exit_at_max_days"])
    spec_clauses, spec_params = _full_strategy_clauses(strategy_spec)
    clauses.extend(spec_clauses)
    params.extend(spec_params)
    where_sql = " AND ".join(clauses)

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        # Same-lineage preference: when the caller knows the validation rank-1's
        # training_hash, prefer a holdout that shares it. This pins val→holdout
        # decay to the same trained model rather than a same-spec but
        # different-lineage holdout that happens to have a higher Sharpe.
        if prefer_training_hash is not None:
            row = db.execute(
                f"""
                SELECT t.family, t.config_name, t.label,
                       p.prediction_hash, b.backtest_hash
                FROM prediction_sets p
                JOIN training_runs t ON p.training_hash = t.training_hash
                JOIN backtest_runs b ON p.prediction_hash = b.prediction_hash
                                     AND b.stage IN ('signal','allocation','risk_overlay','holdout')
                JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
                WHERE {where_sql} AND p.training_hash = ?
                ORDER BY bm.sharpe DESC NULLS LAST
                LIMIT 1
                """,
                params + [prefer_training_hash],
            ).fetchone()
            if row:
                return dict(row)

        row = db.execute(
            f"""
            SELECT t.family, t.config_name, t.label,
                   p.prediction_hash, b.backtest_hash
            FROM prediction_sets p
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN backtest_runs b ON p.prediction_hash = b.prediction_hash
                                 AND b.stage IN ('signal','allocation','risk_overlay','holdout')
            JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
            WHERE {where_sql}
            ORDER BY bm.sharpe DESC NULLS LAST
            LIMIT 1
            """,
            params,
        ).fetchone()
    finally:
        db.close()
    return dict(row) if row else None


def _val_backtest_for_lineage(cs: str, family: str, config_name: str, label: str) -> str | None:
    """Return the highest-Sharpe validation signal-stage backtest_hash for
    the given (family, config_name, label) lineage, or None if absent.

    Used by ``val_rank1_self`` pair construction so the comparison stays
    *within* a lineage when the holdout retrain came from a fallback
    candidate (ranks 2+) rather than the validation rank-1.
    """
    case_dir = get_case_study_dir(cs)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return None
    db = sqlite3.connect(str(db_path))
    try:
        row = db.execute(
            """
            SELECT b.backtest_hash
            FROM prediction_sets p
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN backtest_runs b ON p.prediction_hash = b.prediction_hash
                                 AND b.stage = 'signal'
            JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
            WHERE p.split = 'validation'
              AND t.family = ?
              AND t.config_name = ?
              AND t.label = ?
            ORDER BY bm.sharpe DESC NULLS LAST
            LIMIT 1
            """,
            (family, config_name, label),
        ).fetchone()
    finally:
        db.close()
    return row[0] if row else None


# %%
def _populate_pair(
    cs,
    challenger_hash,
    benchmark_hash,
    benchmark_kind,
    challenger_returns,
    benchmark_returns,
    ppy,
    label,
    *,
    disjoint_windows: bool = False,
    benchmark_label: str | None = None,
):
    """Compute one paired-metric row without mutating a case-study registry.

    With ``disjoint_windows=True`` (val→holdout decay), each side is
    bootstrapped independently over its full window and the difference
    distribution is built from independent draws — no spurious head/tail
    truncation. ``info_ratio`` columns will be NaN since there is no
    aligned diff series to ratio.

    Otherwise, the streams are inner-joined on timestamp and a paired
    stationary bootstrap runs on the aligned diff series.
    """
    min_n = _min_paired_n(ppy)
    if disjoint_windows:
        c_arr = challenger_returns.sort("timestamp")["ret"].to_numpy()
        b_arr = benchmark_returns.sort("timestamp")["ret"].to_numpy()
        finite_c = np.isfinite(c_arr)
        finite_b = np.isfinite(b_arr)
        c_arr, b_arr = c_arr[finite_c], b_arr[finite_b]
        if c_arr.size < min_n or b_arr.size < min_n:
            return {
                "cs": cs,
                "kind": benchmark_kind,
                "label": label,
                "benchmark_label": benchmark_label if benchmark_label is not None else label,
                "skip": f"insufficient_disjoint:n_c={c_arr.size},n_b={b_arr.size}",
            }
        n_overlap = min(c_arr.size, b_arr.size)
        paired = compute_independent_diff_uncertainty(
            c_arr,
            b_arr,
            periods_per_year=ppy,
            case_study=cs,
            label=label,
            n_boot=2000,
            seed=42,
        )
    else:
        aligned = challenger_returns.join(
            benchmark_returns, on="timestamp", how="inner", suffix="_b"
        )
        if aligned.height < min_n:
            return {
                "cs": cs,
                "kind": benchmark_kind,
                "label": label,
                "benchmark_label": benchmark_label if benchmark_label is not None else label,
                "skip": f"insufficient_overlap:n={aligned.height}",
            }
        c_arr = aligned["ret"].to_numpy()
        b_arr = aligned["ret_b"].to_numpy()
        c_arr, b_arr = _joint_coerce(c_arr, b_arr)
        n_overlap = c_arr.size
        if n_overlap < min_n:
            return {
                "cs": cs,
                "kind": benchmark_kind,
                "label": label,
                "benchmark_label": benchmark_label if benchmark_label is not None else label,
                "skip": f"insufficient_after_coerce:n={n_overlap}",
            }
        paired = compute_paired_uncertainty(
            c_arr,
            b_arr,
            periods_per_year=ppy,
            case_study=cs,
            label=label,
            n_boot=2000,
            seed=42,
        )

    if not paired:
        return {
            "cs": cs,
            "kind": benchmark_kind,
            "label": label,
            "benchmark_label": benchmark_label if benchmark_label is not None else label,
            "skip": "uncertainty_empty",
        }
    # For the disjoint path, paired carries n_c/n_b (post-coerce per-side
    # sizes); use min(n_c, n_b) so n_overlap reflects what the bootstrap
    # actually used, not the pre-coerce min from the populator. For the
    # paired path, paired has no n_c/n_b and n_overlap is already the
    # post-_joint_coerce length.
    n_actual = n_overlap
    n_c = paired.get("n_c")
    n_b = paired.get("n_b")
    if n_c is not None and n_b is not None:
        n_actual = int(min(float(n_c), float(n_b)))
    return {
        "cs": cs,
        "kind": benchmark_kind,
        "label": label,
        "benchmark_label": benchmark_label if benchmark_label is not None else label,
        "n_overlap": n_actual,
        "sharpe_diff": paired.get("sharpe_diff"),
        "sharpe_diff_ci_lo": paired.get("sharpe_diff_ci95_lo"),
        "sharpe_diff_ci_hi": paired.get("sharpe_diff_ci95_hi"),
        "info_ratio": paired.get("info_ratio"),
        "p_value": paired.get("p_value"),
    }


# %%
extra_paired_rows: list[dict] = []
_PAIRED_STAGES = ("signal", "allocation", "risk_overlay")
for cs, explorer in explorers.items():
    label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
    # Pool validation backtests across the same stages that holdout.py uses
    # for cross-stage rank-1 (`HOLDOUT_SELECTION_STAGES`). When the val
    # rank-1 is an allocation- or risk_overlay-stage strategy, the holdout
    # retrain uses THAT strategy_spec; pulling only signal-stage candidates
    # here surfaces a leader whose signal.method differs from the holdout's,
    # so `_val_rank1_signal_spec` can't find a matching holdout (e.g.,
    # crypto signal-stage rank-1 = quintile_long_short but cross-stage
    # rank-1 = score_weighted/equal_weight_top_k). Carrier-selection rule:
    # val rank-1 is the highest-Sharpe validation backtest across the three
    # stages; see `_val_rank1_full_spec`.
    cand = pl.concat(
        [explorer.best(stage=s, top_n=2000) for s in _PAIRED_STAGES],
        how="diagonal_relaxed",
    )
    if cand.is_empty() or "backtest_hash" not in cand.columns:
        continue
    if "family" in cand.columns:
        cand = cand.filter(pl.col("family") != "benchmark")
    if label_restriction and "label" in cand.columns:
        cand = cand.filter(pl.col("label").is_in(list(label_restriction)))
    cand = _apply_rung_restriction(cand, cs)
    cand = _apply_carrier_pin(cand, cs)
    if cand.is_empty():
        continue
    cand = cand.sort("sharpe", descending=True).unique(
        subset=["prediction_hash"], keep="first", maintain_order=True
    )

    leader = cand.row(0, named=True)
    leader_hash = leader["backtest_hash"]
    leader_phash = leader["prediction_hash"]
    leader_label = leader.get("label")
    if not leader_label:
        continue
    ppy = {"daily": 252, "weekly": 52, "monthly": 12, "8h": 1095}.get(
        FREQ_MAP.get(cs, "daily"), 252
    )

    chal_full = _aligned_returns(cs, leader_hash)
    if chal_full is None:
        continue

    # Pair #2: cross-stage rank-1 holdout backtest ↔ equal-weight (holdout
    # window). Use the holdout's *own* label for benchmark resolution. The
    # holdout lineage may differ from the validation rank-1 in both family
    # AND label when generate_holdout's degeneracy fallback fires (e.g.,
    # crypto's cross-stage rank-1 over signal/allocation/risk_overlay runs
    # on fwd_ret_24h but the next fall-through candidate runs on
    # fwd_ret_8h). Constrain by val rank-1's full (signal, allocation,
    # risk) spec so val→holdout decay isn't measured across different
    # allocators (e.g. score_weighted vs conformal_weighted), different
    # position-sizing parameters, or different risk overlays.
    val_spec = _val_rank1_full_spec(cs)
    # Resolve the val rank-1's training_hash so the holdout pick prefers the
    # same-lineage holdout when one exists (avoids cross-pollination across
    # training_hashes that happen to share signal_spec).
    _val_training_hash: str | None = None
    try:
        _case_db = get_case_study_dir(cs) / "run_log" / "registry.db"
        with sqlite3.connect(str(_case_db)) as _con:
            _row = _con.execute(
                "SELECT training_hash FROM prediction_sets WHERE prediction_hash = ?",
                (leader_phash,),
            ).fetchone()
            if _row:
                _val_training_hash = _row[0]
    except (sqlite3.Error, OSError) as exc:
        # Surface real registry corruption / IO problems instead of silently
        # falling back to cross-lineage selection. Keep the fallback so the
        # populator still produces a row, but emit a one-line warning so
        # downstream noticeably distinguishes "no preferred lineage" from
        # "DB unavailable".
        print(
            f"[warn] {cs}: failed to resolve val_training_hash from "
            f"prediction_hash={leader_phash}: {type(exc).__name__}: {exc}"
        )
        _val_training_hash = None
    ho_lineage = _holdout_lineage_for(
        cs,
        leader_label,
        strategy_spec=val_spec,
        prefer_training_hash=_val_training_hash,
    )
    ho_hash = ho_lineage["backtest_hash"] if ho_lineage else None
    ho_label = ho_lineage["label"] if ho_lineage else leader_label
    chal_ho = _aligned_returns(cs, ho_hash) if ho_hash else None
    bench_ho_resolved = _benchmark_returns_from_artifact(cs, ho_label, period="holdout")

    if bench_ho_resolved is not None and chal_ho is not None:
        bench_ho_hash, bench_ho_norm, bench_ho_label = bench_ho_resolved
        extra_paired_rows.append(
            _populate_pair(
                cs,
                ho_hash,
                bench_ho_hash,
                f"{SIGNAL_BASELINE_BY_CASE_STUDY.get(cs, 'equal_weight')}_holdout_side_artifact",
                chal_ho,
                bench_ho_norm,
                ppy,
                ho_label,
                benchmark_label=bench_ho_label,
            )
        )
    else:
        # Surface the silent skip so downstream summaries don't conflate
        # "no holdout EW pair" with "fallback succeeded but bootstrap empty".
        # Hits us_firm_characteristics fwd_class_1m when the holdout-window
        # EW artifact is absent for both the classification label and its
        # fwd_ret_* fallback.
        # Multi-axis classification: when both inputs are missing, combine
        # the reasons so reviewers see the full gap, not just the first one.
        skip_parts: list[str] = []
        if chal_ho is None:
            skip_parts.append("no_holdout_challenger_returns")
        if bench_ho_resolved is None:
            skip_parts.append("no_holdout_benchmark_artifact")
        extra_paired_rows.append(
            {
                "cs": cs,
                "kind": f"{SIGNAL_BASELINE_BY_CASE_STUDY.get(cs, 'equal_weight')}_holdout_side_artifact",
                "label": ho_label,
                "benchmark_label": None,
                "skip": "+".join(skip_parts),
            }
        )

    # Pair #3: holdout rank-1 ↔ validation backtest of the SAME lineage.
    # Per-CS holdout regen may fall back from val rank-1 to rank-K when the
    # rank-1 retrain produces degenerate predictions (see holdout.py
    # `generate_holdout` fallback loop). When that happens, comparing the
    # holdout against the val rank-1 of a *different* lineage measures
    # cross-lineage difference, not decay. Always pair against the
    # holdout-lineage's own validation backtest so val_rank1_self holds its
    # "same-lineage decay" semantics.
    if chal_ho is not None and ho_lineage is not None:
        ho_family = ho_lineage["family"]
        ho_config = ho_lineage["config_name"]
        same_lineage = (
            ho_family == leader["family"]
            and ho_config == leader["config_name"]
            and ho_label == leader_label
        )
        if same_lineage:
            val_self_hash = leader_hash
            val_self_returns = chal_full
        else:
            val_self_hash = _val_backtest_for_lineage(cs, ho_family, ho_config, ho_label)
            val_self_returns = _aligned_returns(cs, val_self_hash) if val_self_hash else None
        if val_self_hash is not None and val_self_returns is not None:
            extra_paired_rows.append(
                _populate_pair(
                    cs,
                    ho_hash,
                    val_self_hash,
                    "val_rank1_self",
                    chal_ho,
                    val_self_returns,
                    ppy,
                    ho_label,
                    disjoint_windows=True,
                )
            )

    # Pairs #4–6: stage transitions on the validation rank-1 lineage
    lineage = explorer.champion_lineage(leader_phash)
    for prev_stage, this_stage, kind in [
        ("signal", "allocation", "signal_leader"),
        ("allocation", "cost_sensitivity", "allocation_leader"),
        ("cost_sensitivity", "risk_overlay", "cost_sensitivity_leader"),
    ]:
        prev_entry = lineage.get(prev_stage)
        this_entry = lineage.get(this_stage)
        if not prev_entry or not this_entry:
            continue
        prev_hash = prev_entry["backtest_hash"]
        this_hash = this_entry["backtest_hash"]
        prev_returns = _aligned_returns(cs, prev_hash)
        this_returns = _aligned_returns(cs, this_hash)
        if prev_returns is None or this_returns is None:
            continue
        extra_paired_rows.append(
            _populate_pair(
                cs,
                this_hash,
                prev_hash,
                kind,
                this_returns,
                prev_returns,
                ppy,
                leader_label,
            )
        )

# %%
extra_paired_df = pl.DataFrame(extra_paired_rows)
if extra_paired_df.is_empty():
    print("\n=== Extended Paired-Bootstrap Coverage ===")
    print("No extended paired rows produced.")
else:
    print(
        f"\nextra_paired={len(extra_paired_rows)} rows across "
        f"{extra_paired_df['cs'].n_unique()} case studies"
    )

# %% [markdown]
# ### Extended Paired-Bootstrap Coverage

# %%
extra_paired_df

# %% [markdown]
# Summary by `benchmark_kind` shows which extension pair types landed for
# which CSs. ``equal_weight_holdout_side_artifact`` and ``val_rank1_self``
# are universal (modulo holdout availability); stage-transition pairs
# (``signal_leader``, ``allocation_leader``, ``cost_sensitivity_leader``)
# vary by CS pipeline coverage. CSs pinned at the signal stage (e.g.
# ``sp500_options`` Rung-2) will surface zero stage-transition rows.

# %% [markdown]
# ## Sharpe Progression
#
# For each case study, trace how the **best signal's** Sharpe evolves
# through the pipeline stages. This follows a single `prediction_hash`
# through allocation, costs, and risk — case studies that show `null`
# at later stages have not had those stages run for this particular signal.

# %%
prog_rows = []
for cs, explorer in explorers.items():
    # Apply the same family / label / universe-filter scoping as the rank-1
    # cluster diagnostics so the Sharpe progression chart follows the
    # chapter-wide rank-1 signal rather than whichever Sharpe happens to be
    # highest under any execution regime.
    label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
    candidates = explorer.best(stage="signal", top_n=200)
    if not candidates.is_empty() and "family" in candidates.columns:
        candidates = candidates.filter(pl.col("family") != "benchmark")
    if label_restriction and "label" in candidates.columns and not candidates.is_empty():
        candidates = candidates.filter(pl.col("label").is_in(list(label_restriction)))
    candidates = _apply_rung_restriction(candidates, cs)
    best_signal = candidates.head(1)
    if best_signal.is_empty():
        continue
    pred_hash = best_signal["prediction_hash"][0]
    # Pin progression() to the same rung so allocation/cost/risk rows for
    # sp500_options trace the Rung-2 lineage instead of the higher-Sharpe
    # Rung-3 backtests that share the same prediction_hash.
    prog = _progression_for(explorer, pred_hash, cs)
    if prog.is_empty():
        continue
    # Inject the scope-filtered best_signal as the signal row to keep the
    # rank-1 Sharpe consistent with the cluster diagnostics table.
    _bs = best_signal.row(0, named=True)
    prog_rows.append(
        {
            "case_study": DISPLAY_NAMES.get(cs, cs),
            "stage": "signal",
            "sharpe": _bs["sharpe"],
            "cagr": _bs.get("cagr"),
            "max_drawdown": _bs.get("max_drawdown"),
        }
    )
    for row in prog.filter(pl.col("stage") != "signal").iter_rows(named=True):
        prog_rows.append(
            {
                "case_study": DISPLAY_NAMES.get(cs, cs),
                "stage": row["stage"],
                "sharpe": row["sharpe"],
                "cagr": row.get("cagr"),
                "max_drawdown": row.get("max_drawdown"),
            }
        )

prog_df = pl.DataFrame(prog_rows)
if not prog_df.is_empty():
    prog_pivot = prog_df.pivot(on="stage", index="case_study", values="sharpe").sort("case_study")
else:
    prog_pivot = pl.DataFrame()

# %% [markdown]
# ### Sharpe Progression (best prediction per CS)

# %%
prog_pivot

# %% [markdown]
# Most case studies only have complete data through the signal and allocation
# stages for their best prediction hash. Where the full pipeline is available
# (ETFs, CME Futures, S&P 500 Options), allocation generally preserves or
# modestly improves signal-stage Sharpe, while costs and risk overlays
# have mixed effects. The `null` entries indicate that the specific
# prediction hash traced here was not tested at that stage — it does not
# mean the case study lacks those stages entirely.

# %% [markdown]
# ## Rank-1 Lineage
#
# For each case study, trace the *locked* path through the pipeline for
# the rank-1 validation signal: how Sharpe evolves when the same
# prediction set is carried through allocation, cost, and risk stages.
# Locking the prediction makes stage-to-stage deltas attributable — if
# Sharpe moves between allocation and cost, we know the variable is
# costs, not a silently changed upstream signal.
#
# The path is: signal rank-1 → best allocation on that signal →
# cost-tested version → risk-managed version.

# %%
lineage_rows = []
for cs, explorer in explorers.items():
    # Apply case-study filters: exclude passive benchmarks (equal_weight,
    # etc.), restrict sp500_options to ret_to_expiry / HTM dispatch, and
    # pin sp500_options to the Rung-2 full-universe baseline so lineage
    # traces the same signal as the cross-case cluster diagnostics.
    label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
    candidates = explorer.best(stage="signal", top_n=200)
    if not candidates.is_empty() and "family" in candidates.columns:
        candidates = candidates.filter(pl.col("family") != "benchmark")
    if label_restriction and "label" in candidates.columns and not candidates.is_empty():
        candidates = candidates.filter(pl.col("label").is_in(list(label_restriction)))
    candidates = _apply_rung_restriction(candidates, cs)
    if candidates.is_empty():
        continue
    best_signal = candidates.head(1)
    pred_hash = best_signal["prediction_hash"][0]
    signal_source = best_signal["source"][0] if "source" in best_signal.columns else ""

    prog = _progression_for(explorer, pred_hash, cs)
    if prog.is_empty():
        continue

    row = {
        "case_study": DISPLAY_NAMES.get(cs, cs),
        "cs_id": cs,
        "pred_hash": pred_hash,
        "signal_source": signal_source,
    }
    stage_order = ["signal", "allocation", "cost_sensitivity", "risk_overlay"]
    for stage in stage_order:
        if stage == "signal":
            # Use the scope-filtered best_signal directly; progression() does
            # not apply universe_filter, so for sp500_options a prediction
            # that's backtested under both Rung-2 (full) and Rung-3 (liquid)
            # universes would otherwise surface the higher-Sharpe Rung-3 run.
            r = best_signal.row(0, named=True)
            row["signal_sharpe"] = round(r["sharpe"], 3)
            row["signal_max_dd"] = round(r.get("max_drawdown") or 0, 3)
            row["signal_hash"] = r.get("backtest_hash", "")
            continue
        stage_data = prog.filter(pl.col("stage") == stage)
        if not stage_data.is_empty():
            r = stage_data.row(0, named=True)
            row[f"{stage}_sharpe"] = round(r["sharpe"], 3)
            row[f"{stage}_max_dd"] = round(r.get("max_drawdown") or 0, 3)
            row[f"{stage}_hash"] = r.get("backtest_hash", "")
        else:
            row[f"{stage}_sharpe"] = None
            row[f"{stage}_max_dd"] = None
            row[f"{stage}_hash"] = None

    lineage_rows.append(row)

# %%
lineage_df = pl.DataFrame(lineage_rows)
if not lineage_df.is_empty():
    print("\n=== Rank-1 Lineage (locked stage path per CS) ===")
    print(
        lineage_df.select(
            "case_study",
            "signal_source",
            "signal_sharpe",
            "allocation_sharpe",
            "cost_sensitivity_sharpe",
            "risk_overlay_sharpe",
        )
    )

# %% [markdown]
# The lineage table shows how the rank-1 signal's Sharpe moves stage by
# stage. Case studies with blanks in later stages haven't run downstream
# backtests for their specific rank-1 prediction hash — this says nothing
# about whether those stages exist in the pipeline, only that they weren't
# re-run after this hash became the rank-1. §20.3 discusses how to read
# this pattern.

# %% [markdown]
# ## Holdout Integration
#
# Load holdout backtest results from each case study's registry.
# These are the frozen out-of-sample validations generated by
# [`00_holdout_predictions`](00_holdout_predictions.ipynb) and registered in `prediction_sets`
# with `split='holdout'`.


# %%
def _optional_metric(db: sqlite3.Connection, table: str, column: str, alias: str) -> str:
    """Select `table.column` when the registry has it, otherwise a NULL of the same alias.

    Registries written before a metric existed simply lack its column, and a query naming one
    aborts with `no such column` - taking every other case study down with it. Probing the schema
    keeps a stale registry a row of missing values rather than a failed run.
    """
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    prefix = {"backtest_metrics": "bm", "prediction_metrics": "pm"}[table]
    return f"{prefix}.{column} AS {alias}" if column in columns else f"NULL AS {alias}"


def query_holdout_rows():
    """Query holdout backtest results from each case study registry.

    Applies the same label / universe-filter restrictions as the
    cluster-diagnostics rank-1 selection so the reported holdout follows
    the canonical signal. For sp500_options this means the headline
    holdout row is the Rung-2 retrain (full universe, HTM) that matches
    §20.1's −0.361 number; the Rung-3 retrain (liquid subset, +0.455) is
    the §20.5 mitigation story and surfaced separately by the cascade
    section of 20_strategy_analysis rather than as the headline.
    """
    holdout_rows = []
    for cs in ALL_CASE_STUDIES:
        case_dir = get_case_study_dir(cs)
        db_path = case_dir / "run_log" / "registry.db"
        if not db_path.exists():
            continue

        label_restriction = _CLUSTER_LABEL_RESTRICTIONS.get(cs)
        rung = _CLUSTER_RUNG_RESTRICTIONS.get(cs)
        clauses = ["p.split = 'holdout'"]
        params: list[object] = []
        if label_restriction:
            placeholders = ",".join("?" for _ in label_restriction)
            clauses.append(f"t.label IN ({placeholders})")
            params.extend(sorted(label_restriction))
        if rung is not None:
            clauses.append(
                "COALESCE(json_extract(b.spec_json, '$.strategy.signal.universe_filter'), 'full') = ?"
            )
            params.append(rung["universe_filter"])
            if rung["exit_at_max_days"] is None:
                clauses.append(
                    "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') IS NULL"
                )
            else:
                clauses.append(
                    "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') = ?"
                )
                params.append(rung["exit_at_max_days"])
        # Pin the holdout pick to val rank-1's *full* strategy spec (signal
        # + allocation + risk) so the val/holdout comparison reads the same
        # full pipeline on both sides. Without this constraint, MAX(sharpe)
        # over holdout backtests can surface a different allocator (e.g.
        # conformal_weighted when the val carrier was score_weighted), a
        # different top_k, or a different risk overlay than the validation
        # carrier — pairing those two reads as decay but is actually a
        # full-spec mismatch.
        val_spec = _val_rank1_full_spec(cs)
        spec_clauses, spec_params = _full_strategy_clauses(val_spec)
        clauses.extend(spec_clauses)
        params.extend(spec_params)
        where_sql = " AND ".join(clauses)

        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row
        optional = ", ".join(
            [
                _optional_metric(db, "prediction_metrics", "ic_mean_daily", "holdout_ic_daily"),
                _optional_metric(db, "prediction_metrics", "ic_se_hac", "holdout_ic_se_hac"),
                _optional_metric(db, "prediction_metrics", "ic_p_hac", "holdout_ic_p_hac"),
                _optional_metric(db, "prediction_metrics", "ic_ci_lo", "holdout_ic_ci_lo"),
                _optional_metric(db, "prediction_metrics", "ic_ci_hi", "holdout_ic_ci_hi"),
                _optional_metric(db, "backtest_metrics", "sharpe_ci95_lo", "holdout_sharpe_ci_lo"),
                _optional_metric(db, "backtest_metrics", "sharpe_ci95_hi", "holdout_sharpe_ci_hi"),
                _optional_metric(db, "backtest_metrics", "psr_pvalue", "holdout_psr_p"),
            ]
        )
        rows = db.execute(
            f"""
            SELECT t.family, t.config_name, t.label,
                   b.backtest_hash AS holdout_backtest_hash,
                   p.prediction_hash AS holdout_prediction_hash,
                   pm.ic_mean AS holdout_ic,
                   {optional},
                   bm.sharpe AS holdout_sharpe,
                   bm.max_drawdown AS holdout_max_dd,
                   bm.cagr AS holdout_cagr,
                   bm.num_trades AS holdout_num_trades
            FROM prediction_sets p
            JOIN training_runs t ON p.training_hash = t.training_hash
            LEFT JOIN prediction_metrics pm
                ON p.prediction_hash = pm.prediction_hash
            LEFT JOIN backtest_runs b
                ON p.prediction_hash = b.prediction_hash AND b.stage IN ('signal','allocation','risk_overlay','holdout')
            LEFT JOIN backtest_metrics bm
                ON b.backtest_hash = bm.backtest_hash
            WHERE {where_sql}
            ORDER BY bm.sharpe DESC NULLS LAST
            LIMIT 1
            """,
            params,
        ).fetchall()
        db.close()

        for row in rows:
            holdout_rows.append(
                {
                    "case_study": DISPLAY_NAMES.get(cs, cs),
                    "cs_id": cs,
                    "family": row["family"],
                    "config": row["config_name"],
                    "label": row["label"],
                    "holdout_backtest_hash": row["holdout_backtest_hash"],
                    "holdout_prediction_hash": row["holdout_prediction_hash"],
                    "holdout_ic": row["holdout_ic"],
                    "holdout_ic_daily": row["holdout_ic_daily"],
                    "holdout_ic_se_hac": row["holdout_ic_se_hac"],
                    "holdout_ic_p_hac": row["holdout_ic_p_hac"],
                    "holdout_ic_ci_lo": row["holdout_ic_ci_lo"],
                    "holdout_ic_ci_hi": row["holdout_ic_ci_hi"],
                    "holdout_sharpe": row["holdout_sharpe"],
                    "holdout_sharpe_ci_lo": row["holdout_sharpe_ci_lo"],
                    "holdout_sharpe_ci_hi": row["holdout_sharpe_ci_hi"],
                    "holdout_max_dd": row["holdout_max_dd"],
                    "holdout_cagr": row["holdout_cagr"],
                    "holdout_num_trades": row["holdout_num_trades"],
                    "holdout_psr_p": row["holdout_psr_p"],
                }
            )
    return holdout_rows


# %%
holdout_rows = query_holdout_rows()

# %%
holdout_df = pl.DataFrame(holdout_rows)
if not holdout_df.is_empty():
    print(f"\n=== Holdout Results ({len(holdout_df)} entries) ===")
    print(
        holdout_df.select("case_study", "family", "holdout_ic", "holdout_sharpe", "holdout_max_dd")
    )

# %% [markdown]
# Six of nine holdout backtests are positive: US Firms (+1.77), CME Futures
# (+1.11), ETFs (+1.00), sp500_options Rung-3 HTM+liquid (+0.97), NASDAQ-100
# (+0.41), and FX Pairs (+0.19). The three negative holdouts are S&P 500
# Eq+Opt (-0.73), US Equities Panel (-0.49), and Crypto Perps (-0.13). Holdout
# IC is positive on five case studies (US Firms 0.048, CME 0.047, ETFs 0.046,
# S&P Eq+Opt 0.036, NASDAQ-100 0.010) and negative on four (Crypto -0.029,
# sp500_options -0.011, US Equities -0.006, FX -0.002). The IC/Sharpe agreement
# is imperfect — sp500_options and FX Pairs both pair a negative holdout IC
# with a positive holdout Sharpe, while S&P 500 Eq+Opt does the reverse
# (positive IC 0.036, negative Sharpe -0.73), reflecting portfolio
# construction contributing variance independent of out-of-sample ranking
# accuracy.
#
# **Crypto carrier note**: Crypto's deployed carrier is the gbm/leaves_7_huber
# signal model on fwd_ret_24h, carried from the validation rank-1 (signal
# Sharpe 2.09) into the holdout retrain. That retrain posts a holdout Sharpe of
# -0.13 and a holdout IC of -0.029, so Crypto is the one case study whose
# validated edge does not survive out-of-sample — the model ranking inverts on
# the holdout window.

# %% [markdown]
# ## Stage Attrition Funnel
#
# How many case studies survive each stage of the pipeline?
# A "good predictor" has positive IC. A "tradable" strategy has positive
# gross Sharpe. "Cost-surviving" means positive Sharpe at assumed costs.
# "Risk-tolerable" means managed Sharpe is positive. "Holdout-valid"
# means the holdout Sharpe is positive.
#
# Each row is counted *independently* against `bt_df` and `holdout_df` —
# a case study can appear in `cost_surviving` without appearing in
# `tradable_gross`, since the pipeline runs both stages off the rank-1
# trained model. NB08 reports a *cumulative* version of the same funnel
# (each gate is the subset that passed every preceding gate); use NB08
# for the strict survivor count and this section for the per-stage
# attrition that the chapter prose discusses.

# %%
attrition = {
    "good_predictor": 0,  # positive IC
    "tradable_gross": 0,  # positive signal-stage Sharpe
    "cost_surviving": 0,  # positive cost-adjusted Sharpe
    "risk_tolerable": 0,  # positive managed Sharpe
    "holdout_valid": 0,  # positive holdout Sharpe
}

for cs in ALL_CASE_STUDIES:
    display_name = DISPLAY_NAMES.get(cs, cs)

    # IC check
    cs_ic = ic_df.filter(pl.col("case_study") == display_name)
    if not cs_ic.is_empty() and cs_ic["ic_best"].max() > 0:
        attrition["good_predictor"] += 1

    # Signal Sharpe check
    cs_bt = bt_df.filter(pl.col("case_study") == display_name)
    if not cs_bt.is_empty():
        sig_sr = cs_bt["signal_sharpe"][0]
        if sig_sr is not None and sig_sr > 0:
            attrition["tradable_gross"] += 1

        # Cost check (from bt_df survives_costs)
        surv = cs_bt["survives_costs"][0] if cs_bt["survives_costs"][0] is not None else False
        if surv:
            attrition["cost_surviving"] += 1

        # Risk check (from bt_df managed_sharpe)
        mgd = cs_bt["managed_sharpe"][0]
        if mgd is not None and mgd > 0:
            attrition["risk_tolerable"] += 1

    # Holdout check
    cs_ho = (
        holdout_df.filter(pl.col("cs_id") == cs) if not holdout_df.is_empty() else pl.DataFrame()
    )
    if not cs_ho.is_empty():
        ho_sr = cs_ho["holdout_sharpe"][0]
        if ho_sr is not None and ho_sr > 0:
            attrition["holdout_valid"] += 1

print("\n=== Stage Attrition Funnel ===")
total = len(ALL_CASE_STUDIES)
for stage, count in attrition.items():
    bar = "█" * count + "░" * (total - count)
    print(f"  {stage:20s}  {bar}  {count}/{total}")

# %% [markdown]
# The funnel reads top-down with per-stage independent counts: 9 of 9
# case studies produce positive IC, 8 of 9 produce positive signal-stage
# Sharpe, 8 of 9 survive their assumed cost regime, 7 of 9 remain
# risk-tolerable, and 6 of 9 sustain positive Sharpe on holdout. Counting
# how many case studies each independent gate removes, the largest
# single-stage attrition is the holdout step (3 of 9 fail to sustain a
# positive holdout Sharpe), followed by the risk-tolerance gate (2 of 9);
# the gross-Sharpe and cost-survival gates each remove 1. See NB08 for the
# cumulative funnel (gates compounded) and the named drop-outs at each cut.
# The 6-of-9 holdout rate does not account for evidence quality, which the
# next section addresses.

# %% [markdown]
# ## Measurement Quality Disclosures
#
# Rather than a single trust label per case study, we surface the
# measurement characteristics that drive how much a rank-1 number should
# be trusted: how much the per-fold Sharpe moves around, how many folds
# are positive, how wide the spread is to the 10th-ranked configuration,
# and how severe the validation→holdout decay is. These are independent
# axes; a case study can have narrow per-fold dispersion but sharp holdout
# decay (signal is temporally stable in validation but doesn't generalize
# forward), or vice versa. Collapsing these into "high-confidence /
# provisional / unreliable" hides the trade-off the reader needs to see.

# %%
measurement_rows = []
for cs in ALL_CASE_STUDIES:
    display_name = DISPLAY_NAMES.get(cs, cs)
    cluster_row = (
        cluster_df.filter(pl.col("cs_id") == cs) if not cluster_df.is_empty() else pl.DataFrame()
    )
    ho_row = (
        holdout_df.filter(pl.col("cs_id") == cs) if not holdout_df.is_empty() else pl.DataFrame()
    )
    cs_assessment_cluster = cluster_row.to_dicts()[0] if not cluster_row.is_empty() else {}
    cs_assessment_ho = ho_row.to_dicts()[0] if not ho_row.is_empty() else {}

    rank1 = cs_assessment_cluster.get("rank1_sharpe")
    rank10_spread = cs_assessment_cluster.get("rank1_rank10_spread")
    fold_se = cs_assessment_cluster.get("fold_sharpe_se")
    n_folds_pos = cs_assessment_cluster.get("n_folds_pos") or 0
    n_folds = cs_assessment_cluster.get("n_folds") or 0
    ho_sharpe = cs_assessment_ho.get("holdout_sharpe")
    decay = (rank1 - ho_sharpe) if (rank1 is not None and ho_sharpe is not None) else None

    measurement_rows.append(
        {
            "case_study": display_name,
            "cs_id": cs,
            "rank1_val_sharpe": rank1,
            "rank1_rank10_spread": rank10_spread,
            "fold_sharpe_se": fold_se,
            "n_folds_positive": n_folds_pos,
            "n_folds": n_folds,
            "holdout_sharpe": ho_sharpe,
            "validation_holdout_decay": decay,
        }
    )

measurement_df = pl.DataFrame(measurement_rows)
print("\n=== Measurement Quality Disclosures ===")
print(
    measurement_df.select(
        "case_study",
        "rank1_val_sharpe",
        "rank1_rank10_spread",
        "fold_sharpe_se",
        "n_folds_positive",
        "n_folds",
        "holdout_sharpe",
        "validation_holdout_decay",
    )
)

# %% [markdown]
# ## Variant Analysis
#
# All signal-stage model variants across case studies — IC, Sharpe, and
# share of variants with positive Sharpe. Used by NB03 for detailed
# cross-variant analysis.


# %%
def build_variant_rows():
    """Collect all signal-stage model variants across case studies."""
    variant_rows = []
    for cs, explorer in explorers.items():
        case_dir = get_case_study_dir(cs)
        db_path = case_dir / "run_log" / "registry.db"
        if not db_path.exists():
            continue

        primary_label = configs.get(cs, {}).get("labels", {}).get("primary", "")
        db = sqlite3.connect(str(db_path))
        var_query = """
            SELECT t.family || '/' || t.config_name AS source, t.family,
                   MAX(pm.ic_mean) AS ic,
                   MAX(bm_s.sharpe) AS sharpe
            FROM training_runs t
            JOIN prediction_sets p ON t.training_hash = p.training_hash
            LEFT JOIN prediction_metrics pm
                ON p.prediction_hash = pm.prediction_hash
            LEFT JOIN backtest_runs b
                ON p.prediction_hash = b.prediction_hash AND b.stage = 'signal'
            LEFT JOIN backtest_metrics bm_s
                ON b.backtest_hash = bm_s.backtest_hash
            WHERE p.split != 'holdout'
              AND (pm.ic_mean IS NOT NULL OR bm_s.sharpe IS NOT NULL)
              AND t.family != 'causal_dml'
        """
        var_params: tuple = ()
        if primary_label:
            var_query += "      AND t.label = ?\n"
            var_params = (primary_label,)
        var_query += "    GROUP BY t.family, t.config_name"
        rows = db.execute(var_query, var_params).fetchall()
        db.close()

        for source, family, ic_val, sharpe_val in rows:
            variant_rows.append(
                {
                    "case_study": DISPLAY_NAMES.get(cs, cs),
                    "cs_id": cs,
                    "cadence": FREQ_MAP.get(cs, "unknown"),
                    "source": source or "",
                    "family": family or "",
                    "ic": ic_val,
                    "sharpe": sharpe_val,
                    "positive_sharpe": sharpe_val is not None and sharpe_val > 0,
                }
            )
    return variant_rows


# %% [markdown]
# The schema is declared rather than inferred. Polars reads the leading rows to guess a column's
# type, and a registry that has no metrics yet contributes rows whose `ic` and `sharpe` are all
# null - enough of them and the guess comes back as a null column, which then refuses the first
# real float that arrives behind it. Declaring the types makes an empty registry contribute
# missing values instead of breaking the frame.

# %%
variant_rows = build_variant_rows()

# %%
variant_df = pl.DataFrame(
    variant_rows,
    schema={
        "case_study": pl.String,
        "cs_id": pl.String,
        "cadence": pl.String,
        "source": pl.String,
        "family": pl.String,
        "ic": pl.Float64,
        "sharpe": pl.Float64,
        "positive_sharpe": pl.Boolean,
    },
)
print(f"\n=== Variant Analysis: {len(variant_df)} variants ===")
if not variant_df.is_empty():
    print(
        variant_df.group_by("case_study")
        .agg(n=pl.len(), pct_positive=((pl.col("positive_sharpe").sum()) / pl.len() * 100))
        .sort("case_study")
    )

# %% [markdown]
# The `pct_positive` column reveals a stark divide: ETFs, S&P 500 Eq+Opt,
# S&P 500 Options, and US Equities have 96%+ variants with positive
# Sharpe at the signal stage — nearly every model configuration produces
# a profitable signal-stage result. At the other extreme, FX Pairs (11%)
# and NASDAQ-100 (18%) struggle, with most variants producing negative
# Sharpe. This echoes the IC landscape: asset classes with weak ICs
# produce few positive strategies regardless of model choice. The S&P 500
# Options positive-Sharpe rate is measured before execution costs — the
# HTM short-straddle backtest with full option bid-ask and commissions in
# §20.5 shows how that rate collapses once single-name option costs are
# recognized.

# %% [markdown]
# ## Synthesis JSON
#
# Build `all_synthesis.json` — the per-case-study summary consumed by
# notebooks 02–06 and `generate_figures.py`. All values are queried
# from `registry.db` and `setup.yaml`; nothing is hardcoded.

# %% [markdown]
# Pinning a case study to its spine prediction stops cost and risk figures being
# pooled out of full-universe rows when the selected strategy runs on a
# restricted subset, and `build_all_synthesis` raises rather than pool silently.
# That guard is aimed at a pin which fails to match rows that do exist, meaning
# the pin has gone stale. A case study whose registry holds no backtests at all
# has nothing to pool and nothing to be stale against, and pinning it would abort
# the nine-case-study aggregation over one empty registry. Those are left
# unpinned, and their cost and risk entries are reported as not applicable.

# %%
_PINNED_WITH_EVIDENCE = frozenset(
    row["case_study_id"]
    for row in bt_rows
    if row["case_study_id"] in _CLUSTER_RUNG_RESTRICTIONS
    and (row["n_signal"] + row["n_allocation"] + row["n_risk"]) > 0
)

from case_studies.utils.strategy_analysis import build_all_synthesis

synthesis_dict = build_all_synthesis(
    case_studies=ALL_CASE_STUDIES,
    explorers=explorers,
    configs=configs,
    ic_df=ic_df,
    bt_df=bt_df,
    holdout_df=holdout_df,
    assessments={},
    display_names=DISPLAY_NAMES,
    asset_class_map=ASSET_CLASS_MAP,
    freq_map=FREQ_MAP,
    pin_cost_risk_to_spine=_PINNED_WITH_EVIDENCE,
    allow_missing_spine=ALLOW_MISSING_SPINE,
)

# Strip pipeline stages that don't apply to a case study's canonical
# strategy. sp500_options under the HTM short-straddle discipline has no
# generic allocation stage (capital is 1/n_roll equal-weight across
# overlapping cohorts by construction), no generic cost_sensitivity
# stage (the §18.8 cascade uses option-native bid-ask accounting, not a
# bps sweep), and no generic risk_overlay stage (the expiration
# structure of the strategy sets the risk profile). us_firm_characteristics
# skips risk_overlay because the vectorized-engine code path used for the
# monthly long-short panel does not consume position-level overlays, and
# portfolio-level overlays were purged 2026-05-17 (`f3d7fa8f`) after the
# permanent-halt zero-std Sharpe artifact was diagnosed. Rationale strings
# live in `_STAGE_NA_REASONS` so two CSes that skip the same stage for
# different reasons render different prose. `_STAGES_NOT_APPLICABLE` is
# defined once near the top of the notebook and consumed both here (for
# `synthesis_dict` JSON) and inside `build_backtest_rows` (for `bt_df`)
# so the two artifacts cannot drift on the same case study.
for _cs, _stages in _STAGES_NOT_APPLICABLE.items():
    if _cs not in synthesis_dict:
        continue
    _ps = synthesis_dict[_cs].get("pipeline_summary", {})
    if "allocation" in _stages:
        _ps["allocation"] = {
            "best_allocator": None,
            "best_sharpe": None,
            "allocator_comparison": {},
            "not_applicable_reason": _STAGE_NA_REASONS[(_cs, "allocation")],
        }
    if "costs" in _stages:
        _ps["costs"] = {
            "actual_bps": None,
            "breakeven_bps": None,
            "survives_costs": None,
            "gross_sharpe_at_zero": None,
            "net_sharpe_at_actual": None,
            "capacity_usd_10pct": None,
            "not_applicable_reason": _STAGE_NA_REASONS[(_cs, "costs")],
        }
    if "risk" in _stages:
        _ps["risk"] = {
            "best_overlay": None,
            "baseline_sharpe": None,
            "managed_sharpe": None,
            "managed_max_dd": None,
            "overlay_sharpe_delta": None,
            "overlay_count": 0,
            "not_applicable_reason": _STAGE_NA_REASONS[(_cs, "risk")],
        }

print(f"\nBuilt synthesis JSON for {len(synthesis_dict)} case studies")
for cs_id, cs_data in synthesis_dict.items():
    models = cs_data["pipeline_summary"]["models"]
    best_fam = max(models.items(), key=lambda x: x[1].get("ic_mean") or 0) if models else ("", {})
    best_ic = best_fam[1].get("ic_mean", 0) if best_fam[1] else 0
    print(f"  {cs_id}: {len(models)} families, best IC={best_ic}")

# %% [markdown]
# ## Save Aggregated DataFrames
#
# Export for use by downstream notebooks (02–06).

# %%
overview_df.write_parquet(OUTPUT_DIR / "overview.parquet")
if not ic_df.is_empty():
    ic_df.write_parquet(OUTPUT_DIR / "ic_comparison.parquet")
bt_df.write_parquet(OUTPUT_DIR / "backtest_comparison.parquet")
if not prog_df.is_empty():
    prog_df.write_parquet(OUTPUT_DIR / "sharpe_progression.parquet")
if not lineage_df.is_empty():
    lineage_df.write_parquet(OUTPUT_DIR / "lineage.parquet")
if not holdout_df.is_empty():
    holdout_df.write_parquet(OUTPUT_DIR / "holdout_results.parquet")
if not cluster_df.is_empty():
    cluster_df.write_parquet(OUTPUT_DIR / "rank1_cluster_diagnostics.parquet")
measurement_df.write_parquet(OUTPUT_DIR / "measurement_quality.parquet")
if not variant_df.is_empty():
    variant_df.write_parquet(OUTPUT_DIR / "variant_analysis.parquet")

# Save JSON artifacts
(OUTPUT_DIR / "stage_attrition.json").write_text(
    json.dumps({"total": total, "stages": attrition}, indent=2)
)
(OUTPUT_DIR / "all_synthesis.json").write_text(json.dumps(synthesis_dict, indent=2))

# Relative to the repository root: an absolute path is specific to the machine
# that ran the notebook and tells a reader nothing.
print(f"\nSaved aggregated data to {OUTPUT_DIR.relative_to(REPO_ROOT)}")
for f in sorted(OUTPUT_DIR.glob("*.parquet")) + sorted(OUTPUT_DIR.glob("*.json")):
    print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")

# %% [markdown]
# ## What the Empty Cells Mean
#
# Several tables above carry nulls for whole case studies. A null here is not a
# failed strategy, it is an absent measurement: those case studies are being
# rebuilt and their registries hold no backtest rows yet, so there is nothing
# for the aggregation to read. Predictions and their IC survive the rebuild -
# those are in `prediction_metrics`, a separate table - which is why a case study
# can have an IC in the family comparison and nulls everywhere downstream of it.
# The cell below names which ones, rather than leaving the reader to infer it
# from a pattern of blanks.

# %% tags=["results"]
_empty = [
    row["case_study"]
    for row in bt_rows
    if (row["n_signal"] + row["n_allocation"] + row["n_cost"] + row["n_risk"]) == 0
]
display(
    Markdown(
        f"**{len(_empty)} of {len(bt_rows)} case studies have no registered backtests**: "
        + (", ".join(_empty) if _empty else "none")
        + ". Every backtest-derived column is null for these, and the stage "
        "attrition counts above treat them as not reaching the tradable stage."
    )
)

# %% [markdown]
# ## Artifacts Produced
#
# This notebook aggregates registry state across all 9 case studies into
# the comparison tables consumed by notebooks 02–06 and by the Ch20 figure
# generator. All quantitative data is sourced from `registry.db` and
# `setup.yaml` per case study; nothing is hardcoded.
#
# - `overview.parquet`: Case study metadata (asset class, frequency, universe
#   size, cost assumptions, primary label, number of model families).
# - `ic_comparison.parquet`: Rank-1 per-family IC per case study, for the
#   model-family comparison in §20.2 / notebook 02.
# - `backtest_comparison.parquet`: Per-(case-study, stage) Sharpe / CAGR /
#   drawdown for the rank-1 configuration at each pipeline stage.
# - `sharpe_progression.parquet`: Stage-by-stage Sharpe for the rank-1
#   configuration per case study — the funnel that §20.4 describes.
# - `lineage.parquet`: The stage-path from signal → allocation → cost →
#   risk for the rank-1 configuration per case study.
# - `holdout_results.parquet`: Validation-vs-holdout Sharpe for the rank-1
#   configuration, used for the validation→holdout decay analysis in §20.6.
# - `rank1_cluster_diagnostics.parquet`: Rank-1 / rank-10 / spread / fold-SE /
#   folds-positive for each case study — the measurement that lets readers
#   judge how stable each rank-1 number is.
# - `measurement_quality.parquet`: Per-case-study disclosures (fold-SE,
#   rank-1-to-rank-10 spread, folds-positive fraction, holdout decay) —
#   the evidence the reader needs to weigh, unbundled from any single
#   trust label.
# - `variant_analysis.parquet`: All model variants per case study with IC,
#   Sharpe, and positive-Sharpe indicator — supports the variant-space
#   density discussion in §20.3.
# - `stage_attrition.json`: Pipeline funnel counts by stage.
# - `all_synthesis.json`: The combined per-case-study summary consumed by
#   downstream notebooks.
#
# The framing throughout is measurement-first: every number is a registry
# read with known uncertainty (per-fold SE, rank-cluster width), and the
# downstream notebooks interpret those measurements rather than collapse
# them into categorical labels.
#
# **Next**: [`02_feature_evaluation`](02_feature_evaluation.ipynb) for the
# cross-case-study feature evaluation, then [`03_signal_quality`](03_signal_quality.ipynb)
# for the IC landscape and model family comparison.
