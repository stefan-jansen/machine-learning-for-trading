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
# # NASDAQ-100 Microstructure: Costs
#
# **Chapter 18 — Transaction Costs and Execution**
#
# This is the primary cost-analysis notebook for the NASDAQ-100 case study.
#
# Trading costs enter a strategy once per trade, so what they take out over a
# year depends on how often it trades and on how expensive each name is to trade.
# At a 15-minute rebalancing interval, a strategy crosses the spread many times a
# day on every position it holds, and the expected return over each of those
# intervals is a fraction of a basis point. The two quantities are close enough
# in size that the cost assumption decides the outcome.
#
# The notebook measures two levers against that, separately, so each can be
# attributed:
#
# 1. **Which names to trade.** Restricting to the cheapest-to-trade names
#    changes both what is paid per trade and how much trading the ordering
#    provokes, because the expensive names are also the ones whose ranking moves
#    around most.
# 2. **How often to trade.** Rebalancing less frequently spreads each entry cost
#    over a longer holding period. The signal is unchanged; what changes is how
#    many times it is acted on.
#
# The notebook has three parts:
# - **Sections 1–3**: Standard bps cost grid on full-universe allocation combos,
#   tracing the Sharpe-vs-cost decay curve.
# - **Section 4**: Full universe vs the cost-feasible screen — the first lever,
#   read off existing registry rows for the equal-weight top-k arms, which are
#   the arms both universes carry.
# - **Section 5**: Cadence × per-share cost sweep — the second lever and the
#   publication finding. Uses a per-share cost model ($/share, not bps), more
#   realistic for equities, swept across rebalance frequencies.
#
# **Learning Objectives:**
# 1. Sweep a cost grid over existing backtests to locate the cost level at
#    which a strategy stops being profitable
# 2. Separate the effect of restricting the universe from the effect of
#    rebalancing less often, by varying one at a time
# 3. Read a two-dimensional sweep of rebalancing frequency against per-share
#    cost, and say what execution quality each region of it assumes
#
# **Book Reference:** Chapter 18, Sections 18.2–18.5
#
# **Prerequisites:** [`16_risk_management`](16_risk_management.ipynb), and through it
# [`15_portfolio_management`](15_portfolio_management.ipynb) and [`14_backtest`](14_backtest.ipynb).
# This is the last stage that selects, so it runs after all three and draws from all of
# them.

# %%
"""NASDAQ-100 Microstructure: Costs."""

import json
import sqlite3
import time

import polars as pl

from case_studies.research import open_study
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    warmup_periods_for,
)
from case_studies.utils.backtest_presets import (
    build_backtest_spec,
    clone_backtest_spec,
    ensure_backtest_spec,
    prediction_age_declaration,
    set_backtest_costs_bps,
    strategy_view,
    traded_universe_declaration,
)
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.notebook_contracts import excluded_families
from case_studies.utils.registry import read_predictions, resolve_best_backtest_runs
from case_studies.utils.sweep_config import (
    get_cadence_sweep,
    get_cost_grid_bps,
    get_cost_grid_half_spread_usd,
    get_top_n_predictions,
    get_universe_filters_for,
)
from case_studies.utils.uncertainty import STAGE_SEQUENCE
from utils.paths import get_case_study_dir
from utils.style import show_with_alt

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
LABEL = ""
MAX_SYMBOLS = 0
TOP_N_COMBOS = None
# Both names stay bound here although nothing below reads them: that is what makes the harness
# force preview and supply a workspace - `_declares_tier_and_workspace` in `tests/pm_helpers.py`
# looks for exactly this pair. Without them the canonical branch regenerates in place, which
# needs generated-artifact symlinks a CI checkout does not have.
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %%
# A reduced run is a preview run. Refused on the canonical tier so a narrowed result can
# never land in the registry the book's numbers come from, and so the two can never sit in
# one registry to be ranked against each other: `resolve_best_backtest_runs` takes the top
# Sharpe over every backtest at a stage, and a Sharpe earned over a handful of names would
# outrank one earned over the whole panel. `us_equities_panel` 16 through 19 already refuse
# the parameter this way, and `canonically_refused_parameters` reads the refusal out of the
# source, so the canonical fixture path drops the name rather than handing the notebook
# something its first cell raises on (ml4t/agent-workspace#911).
if EXECUTION_TIER == "canonical" and MAX_SYMBOLS:
    raise ValueError(
        "MAX_SYMBOLS narrows the universe this run trades, which makes it a different "
        "portfolio from the declared one and gives it its own backtest identity "
        "(ml4t/agent-workspace#911). A canonical run trades the declared universe: set "
        "MAX_SYMBOLS=0, or run under EXECUTION_TIER='preview' with a WORKSPACE."
    )

# %% [markdown]
# The study is opened before anything resolves a path or reads the registry. Opening it
# activates a root and rewrites `ML4T_OUTPUT_DIR` process-wide, and every later
# `get_case_study_dir`, prediction read and registry write resolves against that variable. A
# `CASE_DIR` bound before this line points at the released registry while this notebook writes
# to the workspace, and the two never meet: the sweep finds nothing registered and every reader
# scoped to hashes from the other root comes back empty.

# %%
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "cost_sensitivity")
if not LABEL:
    LABEL = bt_config.primary_label

print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}")

COST_GRID_BPS = get_cost_grid_bps(CASE_STUDY_ID)

if excluded_families(CASE_STUDY_ID):
    print(
        "Active-model filter: excluding "
        f"{', '.join(sorted(excluded_families(CASE_STUDY_ID)))} pending corrected reruns"
    )

# %% [markdown]
# ## 1. Load the leading pre-cost runs
#
# We load the least-negative full-universe validation backtests. These are every-bar combos
# and are already loss-making (Ch17); the cost grid below traces how the Sharpe-vs-cost curve
# behaves around them before the two recovery levers — the screen and the cadence — are
# applied.
#
# **The pool is every stage a selected configuration can come from, not just `allocation`.** A risk
# overlay is a strategy in its own right: `16_risk_management` registers it at
# `stage='risk_overlay'` with its own Sharpe, and it is a candidate to carry the case study.
# Pricing only the allocation rows would put a cost curve in the chapter for a strategy the case
# study does not select whenever an overlay outranks its own parent, which is the ordinary case -
# four of the seven completed case studies have a `risk_overlay` as their rank-1 validation
# configuration. The stages come from `STAGE_SEQUENCE` rather than a tuple typed here, so the pool
# cannot drift from the library when a stage is added.
#
# `cost_sensitivity` is the one member excluded, because that is the stage this notebook
# writes: including it would re-price rows that already carry a cost model.
#
# **The pool is also pinned to the canonical universe.** `setup.yaml` declares
# `universe_filter: cost_feasible` and says in the same breath that the full-universe variant
# "is NOT a canonical rank-1 / cohort / DSR candidate". Widening the stages without pinning the
# universe would admit exactly that variant, because the signal stage holds both screened and
# full-universe runs and the full-universe ones are not screened out anywhere else - the cost
# curve would then price a strategy the case study excludes by declaration. The filter is read
# out of each run's own `spec_json`, the same place `derived_tables_off_canonical_universe`
# reads it, so the pool cannot disagree with the sweep that produced the runs.
#
# Section 4 is unaffected and has to be: the full-versus-screened contrast is the one place
# the excluded variant belongs, and it reads `universe_filter` in its own query rather than
# through this pool.
#
# This is also why the notebook is numbered after `16_risk_management` rather than before it.
# Run the other way round, the overlay rows do not exist yet and the pool is `allocation`
# whatever it declares.

# %%
PRE_COST_STAGES = tuple(stage for stage in STAGE_SEQUENCE if stage != "cost_sensitivity")
CANONICAL_UNIVERSE = get_universe_filters_for(CASE_STUDY_ID)[0]


def _on_canonical_universe(frame: pl.DataFrame, stage: str = "upstream") -> pl.DataFrame:
    """Drop runs selected under a universe this case study does not treat as canonical.

    `None` means the case study pins no universe, and then every run qualifies - the filter
    has to be a no-op there rather than dropping everything, because a spec that predates the
    universe axis carries no filter at all.
    """
    if CANONICAL_UNIVERSE is None:
        return frame
    # An empty resolver result carries no columns at all, so reading `spec_json` off it raised
    # `ColumnNotFoundError: "spec_json" not found`, naming a column rather than the absence that
    # produced it. Returning the empty frame is NOT the fix: it made this notebook exit 0 having
    # registered nothing, which is the same absence wearing a success. Measured 2026-09-09 on the
    # smoke chain - 202 signal backtests registered, none carrying `universe_filter`, and this
    # notebook reported no error at all.
    if frame.is_empty():
        raise RuntimeError(
            f"no {stage} backtests are registered for {CASE_STUDY_ID}, so there is nothing "
            "to price. Run 14_backtest through 16_risk_management against this registry first."
        )
    keep = [
        strategy_view(json.loads(spec)).get("signal", {}).get("universe_filter")
        == CANONICAL_UNIVERSE
        for spec in frame["spec_json"]
    ]
    return frame.filter(pl.Series(keep, dtype=pl.Boolean))


def resolve_pre_cost_runs(top_n: int) -> pl.DataFrame:
    """The highest-Sharpe validation runs across every stage a selected configuration may come from.

    Each stage is asked for its whole ranked list and the pool is sorted afterwards, rather
    than taking `top_n` from each and merging them: truncating first lets one stage's leader
    hold a slot that a better run in another stage should have had, and at `top_n=1` that
    drops a whole stage from consideration instead of falling through to the next candidate.
    """
    ranked = [
        frame.with_columns(pl.lit(stage).alias("pool_stage"))
        for stage, frame in (
            (
                stage,
                _on_canonical_universe(
                    resolve_best_backtest_runs(
                        CASE_STUDY_ID, LABEL, split="validation", stage=stage, top_n=1_000_000
                    ),
                    stage,
                ),
            )
            for stage in PRE_COST_STAGES
        )
        if not frame.is_empty()
    ]
    if not ranked:
        return pl.DataFrame()
    return (
        pl.concat(ranked)
        .sort("sharpe", descending=True)
        .unique("backtest_hash", maintain_order=True)
        .head(top_n)
    )


top_combos = resolve_pre_cost_runs(TOP_N_COMBOS)

if top_combos.is_empty():
    print(
        f"No results on the {CANONICAL_UNIVERSE or 'full'} universe at any of "
        f"{', '.join(PRE_COST_STAGES)}. Run 14_backtest through 16_risk_management first."
    )
else:
    for row in top_combos.iter_rows(named=True):
        spec = ensure_backtest_spec(
            CASE_STUDY_ID,
            bt_config,
            json.loads(row["spec_json"]),
            prices=load_backtest_prices_for(
                CASE_STUDY_ID,
                LABEL,
                split="validation",
                warmup_periods=warmup_periods_for(CASE_STUDY_ID),
                max_symbols=MAX_SYMBOLS,
            ),
            prediction_hash=row["prediction_hash"],
            initial_cash=bt_config.initial_cash,
        )
        alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
        # The stage is printed because it is the thing that changed: a `risk_overlay` configuration
        # and its `allocation` parent share a prediction hash, so nothing else in this line
        # distinguishes the overlaid run from the un-overlaid one it was built on.
        print(
            f"  Sharpe={row['sharpe']:.3f}  stage={row['pool_stage']}  alloc={alloc}  "
            f"bt_hash={row['backtest_hash'][:8]}"
        )

# %%
prices = load_backtest_prices_for(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    warmup_periods=warmup_periods_for(CASE_STUDY_ID),
    max_symbols=MAX_SYMBOLS,
)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique()} assets")

# %% [markdown]
# ## 2. Cost Grid Sweep
#
# For each top combo, re-run the backtest at different total cost levels
# (commission + slippage combined). The grid spans from near-zero to levels
# that exceed the signal entirely, tracing the full decay curve.
#
# At 15-minute cadence with ~26 bars per trading day, even 1 bps per leg
# compounds to significant annual drag. The breakeven cost level for this
# case study is expected to be very low — in the range of 1–3 bps total —
# making it viable only for market-makers or prop desks with institutional
# execution quality, or for strategies that extend the hold period to 4–8 bars
# to amortize the per-trade cost.

# %%
n_total = len(top_combos) * len(COST_GRID_BPS) if not top_combos.is_empty() else 0
n_done = 0
t0 = time.time()
# The rows this run registers, so section 3 can plot its own curve rather than everything the
# `cost_sensitivity` stage has ever held. Collected here because the hash is only known after
# the run returns.
swept_hashes: list[str] = []
# Grid points that did not produce a backtest. A cost curve with holes is not a slower
# version of the curve - it is a different one, and the breakeven read off it states a cost
# level the strategy was never tested at. Collected rather than only printed, so the
# rendering below can refuse instead of interpolating across the gap.
failed_points: list[str] = []

for combo_row in top_combos.iter_rows(named=True):
    pred_hash = combo_row["prediction_hash"]
    base_spec = ensure_backtest_spec(
        CASE_STUDY_ID,
        bt_config,
        json.loads(combo_row["spec_json"]),
        prices=prices,
        prediction_hash=pred_hash,
        initial_cash=bt_config.initial_cash,
    )
    alloc_method = strategy_view(base_spec).get("allocation", {}).get("method", "equal_weight")
    predictions = read_predictions(CASE_STUDY_ID, pred_hash)

    for cost_bps in COST_GRID_BPS:
        n_done += 1
        spec = set_backtest_costs_bps(
            clone_backtest_spec(base_spec),
            commission_bps=cost_bps / 2,
            slippage_bps=cost_bps / 2,
        )
        spec["chapter"] = "ch18"
        try:
            result = run_backtest(
                CASE_STUDY_ID,
                pred_hash,
                spec,
                prices=prices,
                predictions=predictions,
                label=LABEL,
                register=True,
                initial_cash=bt_config.initial_cash,
                calendar=bt_config.calendar,
            )
            if result.backtest_hash:
                swept_hashes.append(result.backtest_hash)
            if cost_bps % 10 == 0:
                print(
                    f"  [{n_done}/{n_total}] {alloc_method} @ {cost_bps}bps: "
                    f"Sharpe={result.metrics.get('sharpe', 0):.3f}"
                )
        except Exception as e:
            failed_points.append(f"{alloc_method} @ {cost_bps}bps: {e}")
            print(f"  [{n_done}/{n_total}] {alloc_method} @ {cost_bps}bps: FAILED — {e}")

# %%
elapsed = time.time() - t0
print(f"Cost sweep complete: {n_done} backtests in {elapsed:.0f}s")
if failed_points:
    raise RuntimeError(
        f"{len(failed_points)} of {n_total} cost-grid points did not produce a backtest, so "
        "the decay curve below would be drawn through the gaps and the breakeven read off it "
        "would name a cost level nothing was tested at. Fix the failures and re-run; the "
        "points that succeeded are registered and will be reused.\n  " + "\n  ".join(failed_points)
    )

# %% [markdown]
# ## 3. Cost Sensitivity Analysis
#
# This section is **read-only** — queries the registry for cost-sensitivity
# results and computes breakeven levels.
#
# The Sharpe-versus-cost curve for intraday strategies typically falls steeply
# from the near-zero-cost benchmark. For NASDAQ-100 15-minute, the expected
# pattern is: positive Sharpe at 0–2 bps, break-even around 3–5 bps, negative
# at any cost level resembling realistic retail execution. The flat portion of
# the curve (if it exists) defines the practical cost budget.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %% [markdown]
# **The curve is scoped to the rows this run just registered.** `cost_sensitivity()` unscoped
# returns every row the stage has ever held - configurations selected earlier, superseded
# generations, and
# the full-universe rows section 4 registers on purpose. Plotting those together produces one
# line per allocator drawn through several strategies at once, which is not a Sharpe-versus-cost
# curve for anything. `backtest_explorer.cost_sensitivity`'s own docstring names this case study
# as one that must scope, and it was not scoping.

# %%
if not swept_hashes:
    # Falling back to an unscoped read here would be the worst of both: the curve reappears,
    # mixed across every generation the stage has ever held, precisely on the runs where the
    # sweep above registered nothing and there is no curve to draw. An empty frame says the
    # sweep produced nothing, which is what happened.
    print("The sweep above registered no cost rows, so there is no curve for this run to plot.")
    cost_df = pl.DataFrame()
else:
    cost_df = explorer.cost_sensitivity(backtest_hashes=swept_hashes)

if not cost_df.is_empty():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for alloc in cost_df["allocator"].unique().sort().to_list():
        subset = cost_df.filter(pl.col("allocator") == alloc).sort("cost_bps")
        ax.plot(subset["cost_bps"].to_list(), subset["sharpe"].to_list(), marker="o", label=alloc)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Total Cost (bps per leg)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Sharpe Decay Under Transaction Costs")
    ax.legend()
    # `utils/style` and `matplotlibrc` both set `figure.constrained_layout.use`, so
    # `tight_layout()` warns and fights the layout engine already running; `fig.show()` warns
    # that a non-interactive canvas cannot be shown and publishes no alt text. Measured: two
    # UserWarnings per figure. The six notebooks of this case study already at `done` use
    # `show_with_alt` and neither of the other two calls.
    show_with_alt(
        fig,
        "Line chart of net Sharpe against transaction cost. The horizontal axis is total "
        "cost in basis points per leg and the vertical axis is net Sharpe ratio. One line "
        "with circular markers per allocator, named in the legend, tracing that allocator's "
        "Sharpe as cost rises. Two horizontal reference lines mark zero Sharpe (dashed) and "
        "0.5 (dotted), so the cost at which a line crosses each can be read off.",
    )
else:
    print("No cost sensitivity data in registry")

# %% [markdown]
# ## 4. Full Universe vs the Cost-Feasible Screen
#
# The first lever is which names to trade. Half the bid-ask spread is what a
# trade pays to cross it, and that varies widely across the panel: the same
# strategy pays several times more per trade in the least liquid names than in
# the most liquid ones. The cost-feasible universe keeps the cheapest-to-trade
# names, fixed per split so the screen cannot use information from the window it
# is evaluated on.
#
# This section reads the equal-weight top-k arms on both universes directly from
# the registry, running no new backtests, and compares the outcome alongside the
# trade count. Equal weight and not the featured slot design, because the slot
# design is registered on the screened universe alone and a screen effect needs
# both sides; which arms exist where is set by `baseline_schemes` and
# `reference_schemes` in `config/setup.yaml`.
#
# Reporting outcome and trade count together matters: the screen changes which
# names can be held, so it changes how much trading the ordering provokes as well
# as what each trade costs, and the two effects are not separable from the outcome
# alone.

# %%
# Read on the equal-weight arms, not the slot design, and matched pair by pair.
#
# Which arms exist on which universe is decided by `config/setup.yaml`, not here:
# `baseline_schemes` runs ew_top5/10/20 on `baseline_universe: cost_feasible` for every
# pass-1 prediction, and `reference_schemes` runs the same three on
# `reference_universe: full` for the pass-2 predictions only. The slot design this
# section used to query is registered on the screened universe alone, so that query
# returned one row and the derived line below it was skipped, leaving a one-sided table
# under prose describing two sides.
#
# Matched arm by arm, because the two universes do not cover the same predictions: the
# screened side carries every pass-1 prediction and the full side only the pass-2
# survivors. Averaging each side over its own population would compare 510 backtests
# against 10 and attribute the difference to the screen, when most of it is the
# difference between the two populations.
#
# The match key is the whole signal specification with `universe_filter` removed, not
# `(prediction_hash, top_k)`. Those two fields do not identify an arm: pass 2 registers
# `equal_weight_top_k` on the screened universe at both `long_short: true` and a
# `long_only` variant, the full-universe reference carries the long-short one only, and
# a join on prediction and concentration alone therefore pairs one full row with two
# screened rows. The difference it reports would then mix the universe screen with a
# change of trading direction. Keyed on the sorted items of the signal dict so the match
# does not depend on the order SQLite happens to serialize the object in.
conn = sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db"))
_arms = pl.read_database(
    """
    SELECT
        COALESCE(json_extract(br.spec_json, '$.strategy.signal.universe_filter'),
                 'full')                                                  AS universe,
        br.prediction_hash                                                AS prediction_hash,
        json_extract(br.spec_json, '$.strategy.signal')                   AS signal_json,
        bm.sharpe                                                         AS sharpe,
        bm.num_trades                                                     AS num_trades
    FROM backtest_runs br
    JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash
    JOIN prediction_sets ps ON br.prediction_hash = ps.prediction_hash
    JOIN training_runs tr ON tr.training_hash = ps.training_hash
    WHERE br.stage = 'signal' AND ps.split = 'validation'
      AND json_extract(br.spec_json, '$.strategy.signal.method') = 'equal_weight_top_k'
      AND tr.family = 'gbm'
      AND bm.sharpe IS NOT NULL
    """,
    connection=conn,
    schema_overrides={"sharpe": pl.Float64, "num_trades": pl.Float64},
)
conn.close()


def _arm_key(signal_json: str) -> str:
    """The arm a backtest ran, with the universe taken out of it."""
    signal = json.loads(signal_json)
    signal.pop("universe_filter", None)
    return json.dumps(signal, sort_keys=True)


_arms = _arms.with_columns(
    pl.col("signal_json").map_elements(_arm_key, return_dtype=pl.String).alias("arm"),
)

_full = _arms.filter(pl.col("universe") == "full").select(
    "prediction_hash", "arm", "sharpe", "num_trades"
)
_screened = _arms.filter(pl.col("universe") == "cost_feasible").select(
    "prediction_hash", "arm", "sharpe", "num_trades"
)
# One backtest per (prediction, arm, universe) is what the registry's identity
# guarantees; asserted rather than assumed, because a duplicate would silently weight
# one prediction twice in the averages below.
for _side_name, _side in (("full", _full), ("cost_feasible", _screened)):
    _dupes = _side.group_by("prediction_hash", "arm").len().filter(pl.col("len") > 1)
    if not _dupes.is_empty():
        msg = (
            f"{_dupes.height} (prediction, arm) pairs appear more than once on the "
            f"{_side_name} universe, so the match would not be one to one"
        )
        raise RuntimeError(msg)

_matched = _full.join(_screened, on=["prediction_hash", "arm"], how="inner", suffix="_screened")
screen_compare = (
    pl.concat(
        [
            _matched.select(
                pl.lit("full").alias("universe"),
                pl.col("sharpe"),
                pl.col("num_trades"),
            ),
            _matched.select(
                pl.lit("cost_feasible").alias("universe"),
                pl.col("sharpe_screened").alias("sharpe"),
                pl.col("num_trades_screened").alias("num_trades"),
            ),
        ]
    )
    .group_by("universe")
    .agg(
        n_arms=pl.len(),
        avg_sharpe=pl.col("sharpe").mean().round(3),
        min_sharpe=pl.col("sharpe").min().round(3),
        max_sharpe=pl.col("sharpe").max().round(3),
        avg_trades=pl.col("num_trades").mean().round(0),
    )
    .sort("universe", descending=True)
)
print(f"{_matched.height} (prediction, arm) pairs registered on both universes")
print(screen_compare)

# %%
# The section's claim is a difference between two universes, so one universe is not a
# weaker version of it and the derived line must not be skipped when it finds one. A
# one-row table under prose describing two sides is the failure this refusal exists to
# stop, and it is the state the previous query was in.
_universes = set(screen_compare["universe"].to_list())
if _universes != {"full", "cost_feasible"}:
    msg = (
        "section 4 compares one arm across two universes and the matched set offers "
        f"{sorted(_universes) or 'none'}. It reads stage='signal', split='validation', gbm, "
        "signal.method='equal_weight_top_k', and needs the same (prediction, arm) pair "
        "registered on both the full universe and cost_feasible. `baseline_schemes` and "
        "`reference_schemes` in config/setup.yaml decide that; they currently agree on "
        "ew_top5/10/20, so an empty match means the sweep did not reach pass 2."
    )
    raise RuntimeError(msg)

full_row = screen_compare.filter(pl.col("universe") == "full")
screened_row = screen_compare.filter(pl.col("universe") == "cost_feasible")
d_sharpe = screened_row["avg_sharpe"][0] - full_row["avg_sharpe"][0]
# Screened over full, so the direction reads off the number: above 1 the screen traded
# more. A full-over-screened ratio was the previous form and it reads as a reduction
# whichever way the trade count moved.
trade_ratio = screened_row["avg_trades"][0] / max(full_row["avg_trades"][0], 1)
print(
    f"Screen moves avg Sharpe by {d_sharpe:+.2f} "
    f"({full_row['avg_sharpe'][0]:+.2f} to {screened_row['avg_sharpe'][0]:+.2f}) "
    f"and multiplies trades by {trade_ratio:.2f} "
    f"({full_row['avg_trades'][0]:.0f} to {screened_row['avg_trades'][0]:.0f})."
)

# %% [markdown]
# ### Reading the Screen's Effect
#
# Same arm, same model family, same validation window, and the same predictions on
# both sides. The only difference between the two rows is the tradeable universe, so
# the gap between them is what the screen did and not what the model did.
#
# Read the Sharpe difference and the trade ratio together. The screen changes which
# names can be held, so it changes how much trading the ordering provokes as well as
# what each trade costs, and a Sharpe difference alone cannot separate the two. The
# printed line above states both, computed from the table rather than described here,
# because a described ordering goes stale against the next rebuild while a computed
# one cannot.

# %% [markdown]
# ## 5. Cadence × Per-Share Cost Analysis
#
# The bps sweep above fixes the rebalancing cadence at 15 minutes. But the
# cost-to-edge ratio depends on *how often* we trade, not just *how much* each
# trade costs. At 15-minute cadence the strategy rebalances 26 times per day;
# at hourly cadence only 6–7 times. Holding longer amortizes the fixed per-trade
# cost over a larger expected return per period.
#
# This section sweeps **cadence × per-share spread** — the central exhibit
# for this case study. We use a **per-share cost model** rather than bps,
# because for equities the execution cost is a dollar amount per share (half
# the bid-ask spread plus commission), not a percentage of notional:
#
# The cost axis is expressed in cents per share. For a stock priced around one
# hundred dollars, a cent per share is close to one basis point of notional, so
# the axis spans from direct-market-access execution at the low end, through the
# effective spread a large-cap name typically quotes, to retail-quality
# execution at the high end. Naming the axis in cents rather than basis points
# keeps it comparable across stocks at different prices.
#
# **What the cost axis is.** The grid value `cost_ps` is the total per-share
# round-trip cost, split evenly between the engine's commission and slippage
# settings so the whole cost moves along one axis. The signal and allocation
# notebooks instead set a fixed per-share commission and add a per-asset
# half-spread measured from the liquidity profile, which are separate quantities
# on separate scales. The single axis here is what makes the interaction between
# rebalancing frequency and cost readable in one chart; it is a comparison across
# cadences under a simplified cost shape rather than a reproduction of the cost
# model used to produce the registered results.

# %%
from case_studies.utils.backtest_runner import normalize_prediction_columns
from case_studies.utils.registry import read_predictions

# Top engine signal-stage prediction by Sharpe
db_path = CASE_DIR / "run_log" / "registry.db"
conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
# The universe predicate is the same statement section 1 makes about the selected configuration
# pool, and it has to be made again here: this query picks its own row. Without it the cadence
# exhibit - the publication finding of this notebook - is built on whichever signal row ranks
# highest, which is the full-universe variant `setup.yaml` excludes from canonical candidacy
# whenever it wins.
cur.execute(
    """
SELECT br.prediction_hash, tr.family, tr.config_name, bm.sharpe
FROM backtest_runs br
JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash
JOIN prediction_sets ps ON br.prediction_hash = ps.prediction_hash
JOIN training_runs tr ON ps.training_hash = tr.training_hash
WHERE br.stage = 'signal'
AND json_extract(br.spec_json, '$.strategy.rebalance.mode') = 'engine'
AND tr.family != 'deep_learning'
AND (? IS NULL OR json_extract(br.spec_json, '$.strategy.signal.universe_filter') = ?)
ORDER BY bm.sharpe DESC
LIMIT 1
""",
    (CANONICAL_UNIVERSE, CANONICAL_UNIVERSE),
)
_row = cur.fetchone()
conn.close()

if _row is None:
    print(
        f"No signal-stage engine backtest on the {CANONICAL_UNIVERSE or 'full'} universe. "
        "Skipping cadence sweep."
    )
    best_pred_hash = None
else:
    best_pred_hash = _row[0]
    best_source = f"{_row[1]}/{_row[2]}"
    print(f"Cadence sweep prediction: {best_source} (engine Sharpe={_row[3]:.3f})")

if best_pred_hash is not None:
    predictions_raw = normalize_prediction_columns(read_predictions(CASE_STUDY_ID, best_pred_hash))

    # Thin minute-level predictions to 15m for default cadence
    predictions_15m = predictions_raw.filter(
        (pl.col("timestamp").dt.minute() % 15 == 0) & (pl.col("timestamp").dt.second() == 0)
    )
    # Keep minute-level for asof alignment to coarser cadences
    predictions_minute = predictions_raw
    print(f"  Predictions: {len(predictions_raw):,} (minute), {len(predictions_15m):,} (15m)")
else:
    predictions_raw = predictions_15m = predictions_minute = None

# %% [markdown]
# ### Aligning predictions to target bar frequency
#
# The predictions are minute-level. When rebalancing at hourly cadence, we take the
# **last available prediction** at or before each price bar timestamp via an asof join.
# This is realistic: the portfolio manager uses the most recent signal when the
# rebalance fires.
#
# A backward asof join produces a null only *before* a symbol's series begins, so
# dropping nulls trims the leading edge and nothing else. Everything after a symbol's
# series *ends* reuses its last score for as long as the panel runs, and the coarser the
# cadence the larger a share of the result that is. On this registry's `fwd_dir_15m`
# predictions, eleven of 113 symbols stop reporting mid-sample - ten of them together at
# the 2020-12-18 fold boundary and `UAL` after a single session - and the oldest match
# the join produced was 252 sessions, the whole panel. At 30-minute cadence 19,667 of
# 303,641 aligned rows carried a prediction more than one session old; at four-hour
# cadence, 3,593 of 55,355.
#
# So the age is bounded, and in sessions rather than minutes: an overnight or weekend
# carry is legitimate and is 5,600 minutes wide, while a symbol that has left the
# universe is months of *sessions* stale. Measured on the same predictions, age 0 and
# age 1 cover every symbol that stays; every row beyond that belongs to one of the
# twelve that leave or have a hole. `MAX_PREDICTION_AGE_SESSIONS` is the bound, and the
# rows it removes are counted and printed rather than dropped quietly - an alignment
# that silently discards a symbol is the same absence this bound exists to end.

# %% [markdown]
# The cadences swept come from `backtest.sweep.cadence_sweep` in `setup.yaml`,
# and the frequency tokens are derived from those names, so the sweep and the
# configuration cannot disagree. An unrecognised cadence stops the notebook
# rather than being skipped, because a silently dropped cadence would leave a
# gap in the heatmap that reads as a region with no viable cost.

# %%
CADENCES = get_cadence_sweep(CASE_STUDY_ID)
_CADENCE_TO_FREQ = {
    "15_minute": "15m",
    "30_minute": "30m",
    "1_hour": "1h",
    "2_hour": "2h",
    "4_hour": "4h",
    "daily_close": "1d",
}
_unknown_cadences = [c for c in CADENCES if c not in _CADENCE_TO_FREQ]
if _unknown_cadences:
    raise ValueError(
        f"cadence_sweep contains unknown cadence(s) {_unknown_cadences!r}; "
        f"valid tokens: {sorted(_CADENCE_TO_FREQ)}"
    )
CADENCE_LABELS = {c: _CADENCE_TO_FREQ[c] for c in CADENCES}
FREQ_MAP = dict(CADENCE_LABELS)

# Per-share cost grid: half-spread + commission in dollars per share.
# Single source of truth is ``backtest.sweep.cost_grid_half_spread_usd`` in
# setup.yaml; labels are derived from the grid so they always match.
COST_PER_SHARE_GRID = get_cost_grid_half_spread_usd(CASE_STUDY_ID)
COST_LABELS = [f"{v * 100:g}¢" for v in COST_PER_SHARE_GRID]

cadence_results = []


# One session, so a bar may use the session's own prediction or the one before it -
# which is what an overnight or weekend gap produces - and nothing older.
MAX_PREDICTION_AGE_SESSIONS = 1


def align_predictions_to_bars(
    preds: pl.DataFrame,
    bar_timestamps: pl.Series,
    *,
    max_age_sessions: int = MAX_PREDICTION_AGE_SESSIONS,
    label: str = "",
) -> pl.DataFrame:
    """Align minute predictions to coarser bars, refusing a match older than the bound.

    Sessions come from the prediction panel's own dates, so the bound counts trading days
    rather than calendar time: a Friday prediction matched to a Monday bar is one session
    old, not three days.

    Returns the aligned frame and, when the bound removed rows, the declaration that has to
    reach `backtest_hash` with them. Without that second value the fix would be invisible to
    the registry: `prediction_hash` and the strategy spec are unchanged by a filter on the
    aligned frame, so a bounded run and an unbounded one hash alike and the later one is
    served the earlier one's result - the same shape as ml4t/agent-workspace#911.
    """
    bar_df = pl.DataFrame({"timestamp": bar_timestamps}).unique().sort("timestamp")
    sessions = preds.select(
        pl.col("timestamp").dt.date().unique().sort().alias("session")
    ).with_row_index("session_index")
    symbols = preds["symbol"].unique().sort().to_list()

    aligned = []
    for sym in symbols:
        sym_preds = preds.filter(pl.col("symbol") == sym).sort("timestamp")
        sym_bars = bar_df.with_columns(pl.lit(sym).alias("symbol"))
        joined = sym_bars.join_asof(
            sym_preds.drop("symbol").with_columns(pl.col("timestamp").alias("prediction_ts")),
            on="timestamp",
            strategy="backward",
        )
        aligned.append(joined.drop_nulls("y_score"))

    if not aligned:
        return pl.DataFrame(), None

    matched = pl.concat(aligned)
    dated = (
        matched.with_columns(pl.col("timestamp").dt.date().alias("session"))
        .join(sessions, on="session", how="left")
        .drop("session")
        .with_columns(pl.col("prediction_ts").dt.date().alias("session"))
        .join(sessions, on="session", how="left", suffix="_prediction")
        .drop("session")
        .with_columns(
            (pl.col("session_index") - pl.col("session_index_prediction")).alias("age_sessions")
        )
    )
    fresh = dated.filter(pl.col("age_sessions") <= max_age_sessions)
    stale = dated.filter(pl.col("age_sessions") > max_age_sessions)
    # Printed on every cadence, including the ones that drop nothing. A line only on the bad
    # case is indistinguishable from the function not having run, which is the shape of the
    # defect this bound closes.
    summary = (
        f"  {label or 'alignment'}: match age {dated['age_sessions'].median():.0f} session(s) "
        f"median, {dated['age_sessions'].max()} max, bound {max_age_sessions}"
    )
    declaration = None
    if stale.height:
        names = sorted(stale["symbol"].unique().to_list())
        summary += (
            f" - dropped {stale.height:,} of {dated.height:,} on "
            f"{len(names)} of {dated['symbol'].n_unique()} symbols "
            f"({', '.join(names[:6])}{' ...' if len(names) > 6 else ''})"
        )
        # Declared only when it removed something, so a cadence the bound does not touch
        # produces the spec it produced before this existed and keeps its registered identity.
        declaration = prediction_age_declaration(
            max_age_sessions=max_age_sessions,
            dropped=stale.height,
            kept=fresh.height,
            symbols_dropped=len(names),
        )
    else:
        summary += " - nothing dropped"
    print(summary)
    aligned = fresh.drop(
        "session_index", "session_index_prediction", "age_sessions", "prediction_ts"
    )
    return aligned, declaration


# %% [markdown]
# ### Run one cadence × cost backtest
#
# Helper that builds the per-share cost spec and runs a single cadence backtest.
# Results are appended to `cadence_results` for the heatmap below.


# %%
def run_cadence_cost_backtest(
    cadence, cadence_label, cost_ps, cadence_prices, aligned_preds, state, prediction_age=None
):
    """Run one cadence × cost backtest and record results."""
    state["n_done"] += 1
    n_done = state["n_done"]

    spec = build_backtest_spec(
        CASE_STUDY_ID,
        bt_config,
        prices=cadence_prices,
        prediction_hash=best_pred_hash,
        initial_cash=bt_config.initial_cash,
        chapter="ch18",
        label=LABEL,
        # `MAX_SYMBOLS` reduced `cadence_prices` and, until the run said so in its own
        # specification, that reduction did not reach `backtest_hash`: a reduced run and the
        # full run over the same predictions hashed alike (ml4t/agent-workspace#911). Built
        # from the panel this spec is being built against, which is the one `run_backtest`
        # is handed below. A full run declares nothing and hashes as it did before.
        traded_universe=(traded_universe_declaration(cadence_prices) if MAX_SYMBOLS else None),
        # Travels with the spec for the same reason the universe does: it changes what this
        # run is computed from and `prediction_hash` cannot see it.
        prediction_age=prediction_age,
        # The universe travels with the spec, not just with the query above. A row registered
        # without it reads as full-universe to every later reader - including section 4's
        # full-versus-screened query and `derived_tables_off_canonical_universe` - so the
        # cadence rows would be filed against the comparison they are not part of.
        signal={
            "method": "equal_weight_top_k",
            "top_k": 20,
            "long_short": bt_config.long_short,
            **({} if CANONICAL_UNIVERSE is None else {"universe_filter": CANONICAL_UNIVERSE}),
        },
    )
    spec["strategy"]["rebalance"]["cadence"] = cadence
    spec["backtest_config"]["metadata"]["cadence"] = cadence

    if cost_ps > 0:
        spec["backtest_config"]["commission"]["model"] = "per_share"
        spec["backtest_config"]["commission"]["per_share"] = cost_ps / 2
        spec["backtest_config"]["commission"]["rate"] = 0.0
        spec["backtest_config"]["slippage"]["model"] = "fixed"
        spec["backtest_config"]["slippage"]["fixed"] = cost_ps / 2
        spec["backtest_config"]["slippage"]["rate"] = 0.0
    else:
        set_backtest_costs_bps(spec, commission_bps=0.0, slippage_bps=0.0)

    spec["cadence_sweep"] = True

    try:
        result = run_backtest(
            CASE_STUDY_ID,
            best_pred_hash,
            spec,
            prices=cadence_prices,
            predictions=aligned_preds,
            label=LABEL,
            register=True,
            initial_cash=bt_config.initial_cash,
            calendar=bt_config.calendar,
        )
        sharpe = result.metrics.get("sharpe", 0)
        n_trades = result.metrics.get("num_trades", 0)

        cadence_results.append(
            {
                "cadence": cadence_label,
                "cost_per_share": cost_ps,
                "cost_label": COST_LABELS[COST_PER_SHARE_GRID.index(cost_ps)],
                "sharpe": sharpe,
                "num_trades": n_trades,
                "cagr": result.metrics.get("cagr", 0),
                "max_drawdown": result.metrics.get("max_drawdown", 0),
            }
        )
        print(
            f"  [{n_done}/{state['n_total']}] {cadence_label} @ {cost_ps * 100:.1f}¢/sh: "
            f"Sharpe={sharpe:.3f}, trades={n_trades:,}"
        )
    except Exception as e:
        print(
            f"  [{n_done}/{state['n_total']}] {cadence_label} @ {cost_ps * 100:.1f}¢/sh: FAILED — {e}"
        )


# %%
sweep_state = {
    "n_total": len(CADENCES) * len(COST_PER_SHARE_GRID) if best_pred_hash else 0,
    "n_done": 0,
}
t0 = time.time()

for cadence in CADENCES if best_pred_hash else []:
    freq = FREQ_MAP[cadence]
    cadence_label = CADENCE_LABELS[cadence]

    cadence_prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        LABEL,
        split="validation",
        frequency=freq,
        max_symbols=MAX_SYMBOLS,
    )
    bar_ts = cadence_prices["timestamp"].unique().sort()

    if freq == "15m":
        # The 15-minute cadence is the prediction grid itself, so no as-of match is made and
        # there is no age to bound.
        aligned_preds, prediction_age = predictions_15m, None
    else:
        aligned_preds, prediction_age = align_predictions_to_bars(
            predictions_minute, bar_ts, label=cadence_label
        )
    if aligned_preds.is_empty():
        print(f"  {cadence_label}: no aligned predictions — skipping")
        continue

    print(
        f"\n--- {cadence_label} cadence: {len(bar_ts)} bars, {len(aligned_preds)} aligned predictions ---"
    )
    for cost_ps in COST_PER_SHARE_GRID:
        run_cadence_cost_backtest(
            cadence,
            cadence_label,
            cost_ps,
            cadence_prices,
            aligned_preds,
            sweep_state,
            prediction_age=prediction_age,
        )

# %%
elapsed_cadence = time.time() - t0
print(f"Cadence sweep: {sweep_state['n_done']} backtests in {elapsed_cadence:.0f}s")

# %% [markdown]
# ### Cadence × Cost Heatmap
#
# This is the central finding: the same signal that is worthless at 15-minute
# cadence becomes viable at hourly cadence with institutional-quality execution
# ($\leq$ 2¢/share effective spread). The table shows Sharpe ratio at each
# cadence × cost combination.

# %%
import matplotlib.pyplot as plt
import numpy as np

cadence_df = pl.DataFrame(cadence_results) if cadence_results else pl.DataFrame()

if not cadence_df.is_empty():
    pivot = cadence_df.pivot(on="cost_label", index="cadence", values="sharpe")
    cadence_order = ["15m", "30m", "1h", "4h"]
    cadences_present = [c for c in cadence_order if c in pivot["cadence"].to_list()]
    costs_present = [c for c in COST_LABELS if c in pivot.columns]

    matrix = np.zeros((len(cadences_present), len(costs_present)))
    for i, cad in enumerate(cadences_present):
        row = pivot.filter(pl.col("cadence") == cad)
        for j, cost_col in enumerate(costs_present):
            if cost_col in row.columns:
                val = row[cost_col][0]
                matrix[i, j] = val if val is not None else np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(costs_present)))
    ax.set_xticklabels(costs_present)
    ax.set_yticks(range(len(cadences_present)))
    ax.set_yticklabels(cadences_present)
    ax.set_xlabel("Effective Spread (per share)")
    ax.set_ylabel("Rebalancing Cadence")
    ax.set_title("Sharpe Ratio: Cadence × Per-Share Cost")

    for i in range(len(cadences_present)):
        for j in range(len(costs_present)):
            val = matrix[i, j]
            color = "white" if abs(val) > 1.0 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, label="Sharpe Ratio")
    show_with_alt(
        fig,
        "Heatmap of Sharpe over rebalancing cadence against per-share effective spread. "
        "Rows are cadences and columns are spreads, each cell holding one Sharpe printed to "
        "two decimals over a red-yellow-green colour scale clipped at -2 and +2, so red is "
        "the worst and green the best. A colour bar to the right carries the same scale. The "
        "cell values are the figure's data: the colour repeats them rather than adding "
        "anything a reader would otherwise miss.",
    )
else:
    print("No cadence sweep results")

# %%
if not cadence_df.is_empty():
    print("=== Cadence × Cost Summary ===")
    print(
        cadence_df.sort("cadence", "cost_per_share").select(
            "cadence", "cost_label", "sharpe", "num_trades"
        )
    )

# %% [markdown]
# ### Trade Count by Cadence
#
# Reducing the rebalancing cadence cuts trade counts dramatically, which is
# the mechanism behind the Sharpe improvement: fewer trades means less
# cumulative cost drag. The trade-off is signal decay — the 15-minute
# prediction becomes stale at longer horizons. The sweet spot for this
# dataset is hourly cadence where the signal retains enough edge to cover
# 1–2¢/share execution costs.

# %%
if not cadence_df.is_empty():
    zero_cost = cadence_df.filter(pl.col("cost_per_share") == 0.0)
    if not zero_cost.is_empty():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        cadences = zero_cost["cadence"].to_list()
        trades = zero_cost["num_trades"].to_list()
        sharpes = zero_cost["sharpe"].to_list()

        axes[0].barh(cadences, trades)
        axes[0].set_xlabel("Number of Trades")
        axes[0].set_title("Trade Count by Cadence (Zero Cost)")

        axes[1].barh(cadences, sharpes)
        axes[1].axvline(0, color="gray", linestyle="--")
        axes[1].set_xlabel("Sharpe Ratio")
        axes[1].set_title("Gross Sharpe by Cadence")

        show_with_alt(
            fig,
            "Two horizontal bar charts side by side, sharing one cadence per row. The left "
            "panel gives the number of trades each cadence closes and the right panel its "
            "gross Sharpe, both at zero assumed cost, with a dashed vertical line at zero "
            "Sharpe on the right. Reading a row across both panels is what the pair is for: "
            "it puts the trading a cadence provokes beside the return it earns before any "
            "cost is charged.",
        )

# %% [markdown]
# ## Key Takeaways
#
# 1. **Costs are charged per trade, so trading frequency sets what they take.**
#    A cost that is negligible against a multi-day return is decisive against a
#    fifteen-minute one, and the same strategy can be profitable or not on that
#    difference alone. Any strategy result quoted without its cost assumption is
#    incomplete.
#
# 2. **Restricting the universe and slowing the cadence are different levers.**
#    One changes what each trade costs and which names the ordering can act on;
#    the other changes how many trades there are. They are measured separately
#    here because a result that changed both at once cannot attribute its
#    improvement to either.
#
# 3. **A screen on tradability is not a screen on the signal.** Keeping the
#    cheapest-to-trade names uses no information about returns, and it is fixed
#    per split so it cannot see the window it is evaluated on. That is what
#    keeps it a cost decision rather than a selection.
#
# 4. **The breakeven cost is the number to carry forward.** It states the
#    execution quality a strategy requires rather than the profit it produced
#    under one assumption, and it can be checked against what a given venue and
#    order size actually achieve.
#
# **Known limitations**: The cadence sweep splits its cost evenly between the
# engine's commission and slippage settings so a single dollars-per-share axis
# can be swept. Production costs use a fixed per-share commission plus a
# measured per-asset half-spread, which are different quantities on different
# scales. The sweep is therefore a comparison across cadences under one
# simplified cost shape, not a reproduction of the production cost model.
