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

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.strategy_analysis import compute_cost_bps
from utils.paths import get_case_study_dir, get_chapter_dir

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
# on until one succeeds. The shared paired-metrics API resolves this carrier
# for both the aggregate producer and `query_holdout_rows`.

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

# NASDAQ-100 carrier pin. The cost-feasible **ensemble** slot design was fixed
# before holdout scoring as diversification under validation selection
# uncertainty. Corrected validation Sharpe is +1.348 and sealed-holdout Sharpe
# is +0.411; both confidence intervals straddle zero. Pin the exact corrected
# validation row so the reader path fails closed instead of selecting a
# historical family sibling. The corrected linear holdout is a comparator only
# and is ineligible for reselection.
_NASDAQ_ACTIVE_VAL_HASH = "9d111089aa27"
_NASDAQ_ACTIVE_HOLDOUT_HASH = "eb3da38446fe"
_NASDAQ_PREDICATE = (
    (pl.col("universe_filter") == "cost_feasible")
    & (pl.col("family") == "ensemble")
    & (pl.col("backtest_hash") == _NASDAQ_ACTIVE_VAL_HASH)
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
# Summary showing the diversity of the test bed: nine case studies spanning
# six asset classes, three frequencies, and universe sizes from 19 to 634.

# %%
overview_rows = []
for cs, explorer in explorers.items():
    setup = configs.get(cs, {})
    cost_bps = compute_cost_bps(setup)
    universe = setup.get("universe", {})
    n_assets = universe.get("n_assets", 0) or len(universe.get("symbols", []))
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
# bootstrap on label-aligned strategy returns**. Intraday benchmark decisions
# are compounded by session; sparse benchmarks compound strategy daily P&L to
# each label's explicit outcome endpoint. Block length is derived from
# ``setup.yaml.labels.{label}.rebalance_step`` (falling back to the optimal
# block size, never below the label horizon). Reported quantities:
#
# - ``sharpe_diff`` with 95 % bootstrap CI
# - ``ret_diff`` (annualized return difference) with 95 % CI
# - ``info_ratio`` of the aligned-return difference
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
from case_studies.utils.paired_metrics import (
    populate_paired_metrics_for_studies,
    validation_strategy_sql_filter,
)

all_paired_rows = populate_paired_metrics_for_studies(
    explorers,
    label_restrictions=_CLUSTER_LABEL_RESTRICTIONS,
    rungs=_CLUSTER_RUNG_RESTRICTIONS,
    carrier_pin_predicates=_CARRIER_PIN_PREDICATES,
    frequencies=FREQ_MAP,
)


def _is_overall_side_benchmark(row: dict) -> bool:
    kind = str(row.get("kind", ""))
    return kind.endswith("_side_artifact") and "_holdout_" not in kind


paired_rows = [
    row for row in all_paired_rows if _is_overall_side_benchmark(row) and "skip" not in row
]
paired_case_studies = {str(row["cs"]) for row in paired_rows}
paired_skips = [
    {
        "case_study": cs,
        "reason": "shared paired-metric producer returned no overall side benchmark",
    }
    for cs in explorers
    if cs not in paired_case_studies
]

paired_df = pl.DataFrame(paired_rows)
print("\n=== Paired Bootstrap: rank-1 vs equal-weight ===")
if not paired_df.is_empty():
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
    print("No paired-bootstrap rows produced; see the skip table below.")

if paired_skips:
    print("\nSkipped case studies:")
    for skip in paired_skips:
        print(f"  - {skip['case_study']:<32}  {skip['reason']}")

print(f"\npaired={len(paired_rows)}/{len(explorers)}, skipped={len(paired_skips)}/{len(explorers)}")

# %% [markdown]
# Read each row as: rank-1 challenger annualized Sharpe **minus** equal-weight
# benchmark Sharpe, with a 95 % CI from the paired stationary block bootstrap
# on the label-aligned return difference; the information ratio summarizes the
# excess-return-to-tracking-error ratio; ``prob_wins`` is the fraction of
# bootstrap resamples in which the challenger beat the benchmark; ``p_value``
# tests ``H0: sharpe_diff = 0``. A confident "the model adds skill over the
# passive baseline" claim requires (i) the CI excludes zero, (ii) ``prob_wins``
# close to 1, and (iii) a small ``p_value``. Cases where the CI straddles
# zero are not failures — they signal that the apparent Sharpe gap is within
# block-bootstrap sampling error and should be reported as such.

# %% [markdown]
# ## Paired metrics: full strategy-analysis coverage
#
# The shared paired-metric producer above computes the overall benchmark pair,
# the holdout benchmark pair, same-lineage validation-to-holdout decay, and the
# three stage-transition pairs. It owns benchmark frequency alignment and
# registry writes, so this notebook only displays its returned rows.

# %%
extra_paired_rows = [row for row in all_paired_rows if not _is_overall_side_benchmark(row)]
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
        if cs == "nasdaq100_microstructure":
            clauses.append("b.backtest_hash = ?")
            params.append(_NASDAQ_ACTIVE_HOLDOUT_HASH)
        # Pin the holdout pick to val rank-1's *full* strategy spec (signal
        # + allocation + risk) so the val/holdout comparison reads the same
        # full pipeline on both sides. Without this constraint, MAX(sharpe)
        # over holdout backtests can surface a different allocator (e.g.
        # conformal_weighted when the val carrier was score_weighted), a
        # different top_k, or a different risk overlay than the validation
        # carrier — pairing those two reads as decay but is actually a
        # full-spec mismatch.
        spec_clauses, spec_params = validation_strategy_sql_filter(
            cs,
            explorers[cs],
            label_restriction=label_restriction,
            rung=rung,
            carrier_pin_predicate=_CARRIER_PIN_PREDICATES.get(cs),
        )
        clauses.extend(spec_clauses)
        params.extend(spec_params)
        where_sql = " AND ".join(clauses)

        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row
        rows = db.execute(
            f"""
            SELECT t.family, t.config_name, t.label,
                   b.backtest_hash AS holdout_backtest_hash,
                   p.prediction_hash AS holdout_prediction_hash,
                   pm.ic_mean AS holdout_ic,
                   pm.ic_mean_daily AS holdout_ic_daily,
                   pm.ic_se_hac AS holdout_ic_se_hac,
                   pm.ic_p_hac AS holdout_ic_p_hac,
                   pm.ic_ci_lo AS holdout_ic_ci_lo,
                   pm.ic_ci_hi AS holdout_ic_ci_hi,
                   bm.sharpe AS holdout_sharpe,
                   bm.sharpe_ci95_lo AS holdout_sharpe_ci_lo,
                   bm.sharpe_ci95_hi AS holdout_sharpe_ci_hi,
                   bm.max_drawdown AS holdout_max_dd,
                   bm.cagr AS holdout_cagr,
                   bm.num_trades AS holdout_num_trades,
                   bm.psr_pvalue AS holdout_psr_p
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
    display = DISPLAY_NAMES.get(cs, cs)

    # IC check
    cs_ic = ic_df.filter(pl.col("case_study") == display)
    if not cs_ic.is_empty() and cs_ic["ic_best"].max() > 0:
        attrition["good_predictor"] += 1

    # Signal Sharpe check
    cs_bt = bt_df.filter(pl.col("case_study") == display)
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
    display = DISPLAY_NAMES.get(cs, cs)
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
            "case_study": display,
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


# %%
variant_rows = build_variant_rows()

# %%
variant_df = pl.DataFrame(variant_rows)
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

# %%
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
    pin_cost_risk_to_spine=frozenset(_CLUSTER_RUNG_RESTRICTIONS),
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

print(f"\nSaved aggregated data to {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.glob("*.parquet")) + sorted(OUTPUT_DIR.glob("*.json")):
    print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")

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
