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
# # US Firm Characteristics: Risk Overlay Applicability
#
# **Chapter 19 - Risk Management**
#
# A risk overlay is a rule that closes a position on something the position does
# while it is held: a stop-loss when it falls a set distance below entry, a
# trailing stop when it falls that distance below its own high, a time exit after
# a fixed number of bars. Every one of those rules asks what the price did
# *between* the moment the position was opened and the moment it would otherwise
# be closed.
#
# This case study backtests on the vectorized forward-return path. That path
# holds one weight vector per rebalance and multiplies it by the realized
# forward return over the whole month; it never sees a price inside the month.
# The information a stop needs is therefore not merely unused here, it is absent
# from the data structure the backtest runs on. Simulating a stop on it would
# mean inventing an intra-month path and reporting what the invention did.
#
# **The absence comes from the data release, not from the backtest engine**, and that is
# what makes it permanent rather than a limitation someone could fund away. Read the two
# declarations in `config/setup.yaml` together. `universe.identifiers` is
# `anonymous_split_scoped_firm_axis`, and the note beside it records that identifiers
# persist only inside each released tensor block, with no published mapping between
# blocks. The observations themselves are monthly characteristic vectors. So there is no
# ticker to look a daily price up against, and no continuous firm identity to look it up
# along; a within-month price series for these firms cannot be bought, joined or
# reconstructed. Switching this case study to an engine path would produce the same empty
# table with more machinery behind it.
#
# The other case studies in the book differ on exactly this point rather than on the
# quality of their engineering. An engine path is available where the instrument has a
# public identifier and an intraday or daily price history to go with it. Here the release
# deliberately does not publish one, because anonymity is what allowed the characteristics
# to be released at all.
#
# So this notebook establishes a boundary rather than a result. It selects the
# parent run the overlays would have been applied to, states which controls the
# configuration declares, and registers none of them. The registry query in
# section 3 is what confirms that: an empty result there is the outcome, not a
# missing input.
#
# **Learning Objectives:**
# 1. Select the parent run across the baseline and allocation stages
# 2. Decide whether a backtest path can represent a rule before configuring it
# 3. Separate a governance control from a validation variant that competes on Sharpe
#
# **Book Reference:** Chapter 19, Sections 19.3-19.6, 19.8
#
# **Prerequisites:** the Chapter 17 allocation sweep (`12_portfolio_management`),
# whose runs are in `registry.db`.

# %%
"""US Firm Characteristics: Risk: Engine-Level Risk Rules."""

import json
import time
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.backtest_loaders import (
    VECTORIZED_CASE_STUDIES,
    get_backtest_config,
    load_backtest_prices_for,
)
from case_studies.utils.backtest_presets import (
    clone_backtest_spec,
    ensure_backtest_spec,
    strategy_view,
)
from case_studies.utils.backtest_runner import precompute_weights, run_backtest
from case_studies.utils.registry import read_predictions, resolve_best_backtest_runs
from case_studies.utils.sweep_config import (
    calibrate_trailing_stops,
    get_portfolio_risk_controls,
    get_position_risk_controls,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% [markdown]
# `MAX_SYMBOLS` reduces the price panel and nothing else. The vectorized path takes its
# universe and its P&L from the predictions frame and reads the panel only for the
# rebalance calendar, so lowering it does not shrink a backtest here. It stays in the
# cell because the same parameter is what reduces the engine-path case studies, and a
# test harness binds it uniformly across all of them.

# %% tags=["parameters"]
CASE_STUDY_ID = "us_firm_characteristics"
LABEL = ""
MAX_SYMBOLS = 0
# Zero means all controls; a positive value limits position and portfolio
# controls each.
MAX_RISK_VARIANTS = 0
TOP_N_COMBOS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "risk_overlay")
if not LABEL:
    LABEL = bt_config.primary_label

IS_VECTORIZED = CASE_STUDY_ID in VECTORIZED_CASE_STUDIES
MODE_LABEL = "vectorized" if IS_VECTORIZED else "engine"
print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}, mode: {MODE_LABEL}")

# %% [markdown]
# ## 1. The Parent Run
#
# An overlay is applied to something, so the first step is to say what. The
# candidate is drawn from two stages at once: the equal-weight baselines from
# `11_backtest` and the allocator variants from `12_portfolio_management`. Taking
# the higher validation Sharpe of the two rather than always taking the allocator
# keeps the funnel honest in the case where portfolio construction did not improve
# on the equal-weight parent it was given.
#
# The selection runs on validation months alone, and every number below comes from
# them. The holdout period stays for the strategy analysis notebook.


# %%
def _resolve_pre_risk_runs(case_study: str, label: str, *, split: str, top_n: int) -> pl.DataFrame:
    candidates = [
        resolve_best_backtest_runs(
            case_study,
            label,
            split=split,
            stage=stage,
            top_n=top_n,
        )
        for stage in ("signal", "allocation")
    ]
    candidates = [frame for frame in candidates if not frame.is_empty()]
    if not candidates:
        return pl.DataFrame()
    return (
        pl.concat(candidates)
        .sort("sharpe", descending=True)
        .unique("backtest_hash", maintain_order=True)
        .head(top_n)
    )


# %% tags=["results"]
top_combos = _resolve_pre_risk_runs(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    top_n=TOP_N_COMBOS,
)

if top_combos.is_empty():
    msg = "No baseline or allocation results found. Run the upstream notebooks first."
    raise RuntimeError(msg)

for row in top_combos.iter_rows(named=True):
    spec = json.loads(row["spec_json"])
    alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
    print(f"  Sharpe={row['sharpe']:.3f}  alloc={alloc}  bt_hash={row['backtest_hash'][:8]}")

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)

# %% [markdown]
# ### MAE/MFE-Calibrated Trailing Stops
#
# Maximum adverse excursion is the furthest a position moved against the direction it was
# opened in before it was closed; maximum favourable excursion is the furthest it moved in
# that direction. Both are properties of the path a position travelled while it was held,
# and calibrating a stop from them means setting the threshold where it would have avoided
# the losers without cutting the winners short: a stop tighter than the typical winner's
# adverse excursion closes trades that were about to work.
#
# That calibration therefore needs the same thing the stops themselves need, which is a
# price between the open and the close. On the vectorized monthly-outcome path a position
# has an entry weight and a realised month, and no excursion at all - not an unmeasured
# one, an undefined one. So this calibration is skipped and the configured
# position-control catalog is left unexecuted.

# %%
_position_grid = get_position_risk_controls(CASE_STUDY_ID)
if not IS_VECTORIZED and "close" in prices.columns:
    calibrated = calibrate_trailing_stops(prices)
    if calibrated:
        existing_thresholds = {rc.get("threshold", 0) for rc in _position_grid}
        new_calibrated = [c for c in calibrated if c["threshold"] not in existing_thresholds]
        position_controls = _position_grid + new_calibrated
        print(f"MAE/MFE calibration added {len(new_calibrated)} thresholds")
    else:
        position_controls = _position_grid
        print("MAE/MFE calibration returned no results; using standard grid")
else:
    position_controls = _position_grid
    reason = (
        "the backtest path is vectorized"
        if IS_VECTORIZED
        else "the price panel carries no close column"
    )
    print(f"Skipping MAE/MFE calibration: {reason}")

portfolio_controls = get_portfolio_risk_controls(CASE_STUDY_ID)
# Portfolio-limit overlays were purged 2026-05-17; this CS sweeps position-level
# overlays only. Fail loudly if a portfolio overlay is ever re-introduced into
# setup.yaml so it cannot silently re-file overlay backtests against the spine.
assert not portfolio_controls, (
    f"Unexpected portfolio risk controls for {CASE_STUDY_ID}: {portfolio_controls}. "
    "Portfolio-limit overlays were removed; only position-level overlays are swept."
)
if MAX_RISK_VARIANTS > 0:
    position_controls = position_controls[:MAX_RISK_VARIANTS]
    portfolio_controls = portfolio_controls[:MAX_RISK_VARIANTS]
    print(f"Risk variants limited to {MAX_RISK_VARIANTS} each")

# %% [markdown]
# ## 2. Risk Overlay Sweep
#
# On an engine-path case study this loop registers one backtest per position-level
# control. Here the position loop is skipped because the path cannot represent the
# rules, and the portfolio-control list is empty by configuration, so the loop body
# has nothing to register and the count below is zero by construction rather than by
# failure. The two are different outcomes and the counters separate them.
#
# The two lists are empty for different reasons, and only one of them is about this
# backtest path. A position-level control asks what one position did while it was held,
# so it is blocked by the missing intra-month price. A portfolio-level control asks what
# the book looked like at a rebalance: gross exposure, the largest weight any single name
# may carry, the number of names that must be held. Every one of those is answerable from
# the weight vector this path does hold, so the vectorized path is no obstacle to them.
#
# They are absent because `config/setup.yaml` declares none, and that is a position rather
# than an oversight. A gross-exposure limit or a per-name cap is a constraint the desk
# operates under whatever the backtest says. Sweeping it alongside the allocators would
# enter it into a competition ranked on validation Sharpe, and the loosest cap wins that
# competition almost by definition, since a cap only ever removes exposure the strategy
# wanted. The result would be a notebook that recommends relaxing a risk limit on the
# grounds that relaxing it scored higher, which is the failure the separation avoids.
# Where this case study does constrain concentration it does so through `top_k`, which is
# declared in the strategy and swept as part of it.

# %%
n_done = 0
n_failed = 0

# %% [markdown]
# Every run inside the loop below is fed `combo_weights`, and computing those means
# running the parent's allocator again. Where neither control list can produce a run,
# that work has no consumer, so the loop is not entered at all and the weights are
# never computed.

# %%
will_register = bool(portfolio_controls) or (not IS_VECTORIZED and bool(position_controls))
if not will_register:
    print(
        "No control can run on this backtest path, so no allocation weights are "
        "computed and no backtest is registered."
    )

for combo_idx, combo_row in enumerate(top_combos.iter_rows(named=True) if will_register else []):
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

    t0 = time.time()
    combo_weights = precompute_weights(
        predictions, base_spec, prices, label=LABEL, case_study=CASE_STUDY_ID
    )
    print(
        f"  Combo {combo_idx + 1}/{len(top_combos)}: {alloc_method} - "
        f"weights precomputed in {time.time() - t0:.0f}s"
    )

    # Position-level risk rules (engine only)
    if not IS_VECTORIZED:
        for rc in position_controls:
            spec_risk = clone_backtest_spec(base_spec)
            spec_risk["chapter"] = "ch19"
            if rc["type"] == "time_exit":
                spec_risk["strategy"]["risk"] = {
                    "name": rc["name"],
                    "position_rules": [{"type": rc["type"], "bars": rc["bars"]}],
                }
            else:
                spec_risk["strategy"]["risk"] = {
                    "name": rc["name"],
                    "position_rules": [{"type": rc["type"], "threshold": rc["threshold"]}],
                }

            try:
                result = run_backtest(
                    CASE_STUDY_ID,
                    pred_hash,
                    spec_risk,
                    prices=prices,
                    predictions=predictions,
                    label=LABEL,
                    register=True,
                    initial_cash=bt_config.initial_cash,
                    calendar=bt_config.calendar,
                    precomputed_weights=combo_weights,
                )
                n_done += 1
                print(
                    f"    {rc['name']}: Sharpe={result.metrics.get('sharpe', 0):.3f}, "
                    f"MaxDD={result.metrics.get('max_drawdown', 0):.2%}"
                )
            except Exception as e:
                n_failed += 1
                print(f"    {rc['name']}: FAILED - {e}")

    # Portfolio-level risk limits
    for rc in portfolio_controls:
        spec_risk = clone_backtest_spec(base_spec)
        spec_risk["chapter"] = "ch19"
        spec_risk["strategy"]["risk"] = {
            "name": rc["name"],
            "portfolio_limits": [{"type": rc["type"], "threshold": rc["threshold"]}],
        }

        try:
            result = run_backtest(
                CASE_STUDY_ID,
                pred_hash,
                spec_risk,
                prices=prices,
                predictions=predictions,
                label=LABEL,
                register=True,
                initial_cash=bt_config.initial_cash,
                calendar=bt_config.calendar,
                precomputed_weights=combo_weights,
            )
            n_done += 1
            print(
                f"    {rc['name']}: Sharpe={result.metrics.get('sharpe', 0):.3f}, "
                f"MaxDD={result.metrics.get('max_drawdown', 0):.2%}"
            )
        except Exception as e:
            n_failed += 1
            print(f"    {rc['name']}: FAILED - {e}")

print(f"\nRisk sweep complete: {n_done} registered, {n_failed} failed")

# %% [markdown]
# ## 3. What The Registry Holds
#
# This section only reads. It asks the registry for every overlay run filed against
# this case study and, for each, the change in Sharpe against the parent it was
# applied to.
#
# An empty answer here is the point of the notebook rather than a gap in it. A Sharpe
# delta next to each rule would read exactly as one a stop had earned, and on this path
# it could only come from an intra-month price series the data does not contain, so an
# empty table is the honest form of the answer.

# %% [markdown]
# The read is scoped to the prediction the parent run carries. The registry accumulates
# across labels and across earlier funnels, and this section's answer is a count of
# rows, so an unscoped read would turn an overlay row filed under some other selection
# into evidence about this one - which is the single way this notebook's argument could
# be reported as refuted by rows that never tested it.

# %%
explorer = BacktestExplorer(CASE_STUDY_ID)
parent_hash = top_combos["prediction_hash"][0]

# %% tags=["results"]
risk_df = explorer.risk_impact(prediction_hash=parent_hash)

if risk_df.is_empty():
    print("No risk overlay run is filed against the parent run, which is the outcome.")
else:
    print(f"Risk overlays filed against the parent run: {len(risk_df)}")
    with pl.Config(tbl_rows=risk_df.height):
        print(
            risk_df.select("risk_name", "risk_type", "sharpe", "max_drawdown", "sharpe_delta").sort(
                "sharpe_delta", descending=True
            )
        )

# %% [markdown]
# ## Key Takeaways
#
# 1. Whether a rule can be represented is a property of the backtest path, not a
#    setting. A stop needs a price between rebalances; the vectorized forward-return
#    path holds one return per rebalance and has none, so a stop cannot be evaluated
#    on it at any parameter value.
# 2. The configuration still declares the position-level controls, because the same
#    file drives the engine-path case studies where they do run. Declared and
#    applicable are separate questions, and this notebook answers the second.
# 3. Portfolio-level limits are absent on purpose. A gross-exposure or per-name cap
#    is a constraint the desk operates under, not a variant that competes for the
#    highest validation Sharpe, and sweeping it as one invites keeping whichever cap
#    was loosest on the grounds that it scored highest.
# 4. The overlay stage registers nothing and reads no holdout month, so the funnel
#    enters the strategy analysis carrying the parent run from section 1 unchanged.
#
# **Next:** [`14_costs`](14_costs.ipynb), which sweeps the cost grid over the
# configuration this stage's result is one candidate for.
# [`17_strategy_analysis`](17_strategy_analysis.ipynb) confronts the selection this funnel
# performed and is where the results are interpreted.
