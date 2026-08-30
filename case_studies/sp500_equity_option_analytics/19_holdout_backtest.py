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
# # S&P 500 Equity+Options: Holdout Backtest
#
# [`18_holdout_predictions`](18_holdout_predictions.ipynb) refit the selected
# configuration and predicted 2021. This notebook runs the selected *strategy*
# on those predictions - the same allocator, the same concentration, the same
# risk overlay, the same costs - and reports what the portfolio would have
# earned. It is the first number in this case study that was not produced on
# data the selection could see.
#
# **Learning objectives**
#
# 1. Hold a strategy specification fixed across two windows and change only the
#    predictions it consumes.
# 2. Read a holdout result against the validation figure that selected it, and
#    know what the gap between them does and does not establish.
# 3. Recognise which parts of a degradation are attributable to selection and
#    which to the period.
#
# **Book reference:** Chapter 20, Section 20.3.
#
# **Prerequisites:** [`18_holdout_predictions`](18_holdout_predictions.ipynb).
# Signals form after Friday's close and execute at the next available open. The
# current-constituent universe retains survivorship bias, and it does so more
# consequentially here than anywhere else in the case study: a 2021 window
# evaluated on today's membership excludes every company that left the index,
# so this number is optimistic by an amount this case study does not measure.

# %%
"""S&P 500 Equity+Options: run the selected strategy on the holdout predictions."""

import json
import sqlite3
import warnings

import matplotlib.pyplot as plt
import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import CandidateSet, Study
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    warmup_periods_for,
)
from case_studies.utils.backtest_presets import (
    clone_backtest_spec,
    ensure_backtest_spec,
    strategy_view,
)
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.notebook_contracts import prediction_members_in_force
from case_studies.utils.registry import (
    backtest_hash_from_parts,
    model_source,
    read_predictions,
)
from case_studies.utils.uncertainty import load_daily_returns_with_timestamp
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
LABEL = ""
MAX_SYMBOLS = 0

# %% [markdown]
# ### What is asked for, and what it resolves to
#
# The parameters above are the request; the values this notebook runs on are resolved here under
# different names, so a resolved value cannot overwrite the request that produced it. An injected
# parameter wins; otherwise the case study's own declaration does.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
REGISTRY_DB = CASE_DIR / "run_log" / "registry.db"
bt_config = get_backtest_config(CASE_STUDY_ID)
HOLDOUT_LABEL = LABEL or bt_config.primary_label
print(f"Case study: {CASE_STUDY_ID}; label: {HOLDOUT_LABEL}")

# %%
_study = Study.at(CASE_DIR, case_study=CASE_STUDY_ID, entry_point="19_holdout_backtest")
CURRENT_MEMBERS, _population_notes = prediction_members_in_force(_study)
for _note in _population_notes:
    print(_note)

# %% [markdown]
# ## 1. The strategy, read out of the frozen candidate set
#
# [`16_risk_management`](16_risk_management.ipynb) freezes the field it ranked
# over, and both this notebook and
# [`18_holdout_predictions`](18_holdout_predictions.ipynb) read the selection out
# of that set rather than re-deriving it. The set is immutable, so the two
# resolve the same configuration without either being told what it is; handing
# it over in a parameter or a file would add a way for them to disagree.
#
# A holdout run that could be pointed at a strategy of the reader's choosing
# would be a way to search the holdout. This one cannot be.

# %%
CANDIDATE_SET_NAME = f"{CASE_STUDY_ID}:holdout-candidates"
try:
    CANDIDATES = CandidateSet.one(_study, name=CANDIDATE_SET_NAME)
except (ValueError, LookupError) as exc:
    raise RuntimeError(
        f"no candidate set {CANDIDATE_SET_NAME!r} resolves in this registry ({exc}). "
        "16_risk_management freezes it as its last step; run that first."
    ) from exc

SELECTED = CANDIDATES.best_validation_sharpe()
if not SELECTED.complete:
    raise RuntimeError(f"the selected validation backtest {SELECTED.hash} is incomplete")
CARRIER_SPEC = SELECTED.spec()
CARRIER_STRATEGY = strategy_view(CARRIER_SPEC)
carrier_record = SELECTED.registry_record()
CARRIER_PREDICTION_HASH = carrier_record["prediction_hash"]

with sqlite3.connect(REGISTRY_DB) as db:
    _carrier_source = db.execute(
        "SELECT t.family, t.config_name FROM prediction_sets p "
        "JOIN training_runs t USING(training_hash) WHERE p.prediction_hash = ?",
        (CARRIER_PREDICTION_HASH,),
    ).fetchone()
print(
    f"Candidate set {CANDIDATES.hash} with {len(CANDIDATES.members)} members selects "
    f"{model_source(*_carrier_source)} with "
    f"{(CARRIER_STRATEGY.get('allocation') or {}).get('method', 'equal_weight')} allocation, "
    f"top-{(CARRIER_STRATEGY.get('signal') or {}).get('top_k')}, "
    + (
        f"risk overlay {(CARRIER_STRATEGY.get('risk') or {}).get('name')}"
        if (CARRIER_STRATEGY.get("risk") or {}).get("name")
        else "no risk overlay"
    )
)

# %% [markdown]
# ## 2. Which holdout prediction set belongs to this carrier
#
# The holdout fit is a different training run from the validation fit by
# construction - a different interval is a different computation - so the two
# cannot be matched on a training hash, and matching on a family and
# configuration name would accept a fit of the same model over the wrong window.
#
# What identifies the pair is the specification itself. A holdout refit changes
# the cross-validation interval and the two fields the resolver derives per
# fold; every other field, including the feature artifacts and the model
# parameters, is carried across unchanged. So the check below is that difference
# exactly: the two specifications must differ in those fields and agree
# everywhere else.

# %%
REKEYED_FIELDS = ("cv", "expected_prediction_keys", "macro_context")

with sqlite3.connect(REGISTRY_DB) as db:
    holdout_rows = db.execute(
        "SELECT p.prediction_hash, t.training_hash, t.spec_json, "
        "       p.checkpoint_kind, p.checkpoint_value "
        "FROM prediction_sets p JOIN training_runs t USING(training_hash) "
        "WHERE p.split = 'holdout'"
    ).fetchall()
    carrier_training = db.execute(
        "SELECT t.training_hash, t.spec_json, p.checkpoint_kind, p.checkpoint_value "
        "FROM prediction_sets p JOIN training_runs t USING(training_hash) "
        "WHERE p.prediction_hash = ?",
        (CARRIER_PREDICTION_HASH,),
    ).fetchone()

carrier_training_hash, carrier_spec_json, carrier_kind, carrier_value = carrier_training
VALIDATION_SPEC = json.loads(carrier_spec_json)


def carries_the_same_configuration(candidate: dict) -> tuple[bool, str]:
    """Whether ``candidate`` is this carrier's configuration refitted for the holdout."""
    if candidate.get("family") != VALIDATION_SPEC.get("family"):
        return False, "different family"
    if candidate.get("label") != VALIDATION_SPEC.get("label"):
        return False, "different label"
    if candidate.get("config_name") != VALIDATION_SPEC.get("config_name"):
        return False, "different configuration"
    left = {
        key: value for key, value in candidate["computation"].items() if key not in REKEYED_FIELDS
    }
    right = {
        key: value
        for key, value in VALIDATION_SPEC["computation"].items()
        if key not in REKEYED_FIELDS
    }
    if left != right:
        moved = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        return False, f"computation differs beyond the re-keyed fields: {moved}"
    if candidate["computation"].get("cv", {}).get("split") != "holdout":
        return False, "the fit does not declare the holdout split"
    return True, "refit of this carrier over the holdout interval"


matches, rejected = [], []
for prediction_hash, training_hash, spec_json, kind, value in holdout_rows:
    ok, reason = carries_the_same_configuration(json.loads(spec_json))
    (matches if ok else rejected).append((prediction_hash, training_hash, kind, value, reason))

if not matches:
    raise RuntimeError(
        "No holdout prediction set belongs to the selected carrier. Run "
        "18_holdout_predictions, which refits it; "
        + (
            "the registry holds "
            + "; ".join(f"{ph} ({reason})" for ph, _, _, _, reason in rejected)
            if rejected
            else "the registry holds no holdout predictions at all"
        )
    )
if len(matches) > 1:
    raise RuntimeError(
        f"{len(matches)} holdout prediction sets match this carrier: "
        + ", ".join(prediction_hash for prediction_hash, *_ in matches)
    )

HOLDOUT_PREDICTION_HASH, HOLDOUT_TRAINING_HASH, holdout_kind, holdout_value, _ = matches[0]
if (holdout_kind, holdout_value) != (carrier_kind, carrier_value):
    raise RuntimeError(
        f"the holdout prediction is at {holdout_kind}={holdout_value} but the selection was made "
        f"on {carrier_kind}={carrier_value}"
    )
print(
    f"Holdout prediction {HOLDOUT_PREDICTION_HASH} from training {HOLDOUT_TRAINING_HASH}, "
    f"the validation carrier {carrier_training_hash} refitted over the holdout interval "
    f"at {holdout_kind}={holdout_value}"
)

# %% [markdown]
# The two windows the two fits saw, side by side. The gap between the training
# end and the holdout open is one label buffer, which is what stops the last
# training label's five-day outcome from resolving inside the window being
# judged.

# %%
with sqlite3.connect(REGISTRY_DB) as db:
    holdout_spec = json.loads(
        db.execute(
            "SELECT spec_json FROM training_runs WHERE training_hash = ?",
            (HOLDOUT_TRAINING_HASH,),
        ).fetchone()[0]
    )
holdout_fold = holdout_spec["computation"]["cv"]["folds"][0]
validation_folds = VALIDATION_SPEC["computation"]["cv"]["folds"]

pl.DataFrame(
    [
        {
            "fit": "validation",
            "folds": len(validation_folds),
            "train from": min(str(f["train_start"])[:10] for f in validation_folds),
            "train to": max(str(f["train_end"])[:10] for f in validation_folds),
            "evaluated": f"{min(str(f['val_start'])[:10] for f in validation_folds)}"
            f" to {max(str(f['val_end'])[:10] for f in validation_folds)}",
        },
        {
            "fit": "holdout",
            "folds": 1,
            "train from": str(holdout_fold["train_start"])[:10],
            "train to": str(holdout_fold["train_end"])[:10],
            "evaluated": f"{str(holdout_fold['val_start'])[:10]}"
            f" to {str(holdout_fold['val_end'])[:10]}",
        },
    ]
)

# %% [markdown]
# ## 3. Run the selected strategy on the holdout window
#
# The specification is the carrier's own, cloned and re-pointed at the holdout
# predictions. Nothing about the strategy is re-derived here: re-deriving it
# would let a change anywhere upstream alter what the holdout evaluates without
# the change being visible as a different selection.
#
# Prices are loaded for the holdout window with the allocator's warmup prefix.
# The covariance estimator this carrier uses needs history before its first
# rebalance, and without the prefix it would fall back to an imputed warmup on
# exactly the dates the result is read from.

# %%
prices = load_backtest_prices_for(
    CASE_STUDY_ID,
    HOLDOUT_LABEL,
    split="holdout",
    warmup_periods=warmup_periods_for(CASE_STUDY_ID),
    max_symbols=MAX_SYMBOLS,
)
print(
    f"Price support: {len(prices):,} rows across {prices['symbol'].n_unique()} symbols, "
    f"{prices['timestamp'].min()} to {prices['timestamp'].max()}"
)
predictions = read_predictions(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH)
print(
    f"Predictions: {predictions.height:,} rows over {predictions['timestamp'].n_unique()} sessions"
)

# %%
holdout_strategy_spec = clone_backtest_spec(
    ensure_backtest_spec(
        CASE_STUDY_ID,
        bt_config,
        CARRIER_SPEC,
        prices=prices,
        prediction_hash=HOLDOUT_PREDICTION_HASH,
        initial_cash=bt_config.initial_cash,
    )
)
HOLDOUT_BACKTEST_HASH = backtest_hash_from_parts(HOLDOUT_PREDICTION_HASH, holdout_strategy_spec)

# The whole backtest specification, not the strategy block. `strategy_view` returns signal,
# allocation and risk and stops there, so a check built on it would pass a holdout run at
# different commissions, slippage, fill timing or stop behaviour - and holding those fixed is
# most of what makes the holdout number comparable to the validation one. What legitimately
# differs between the two runs is the predictions consumed and the price panel sliced to, so
# only those are projected out.
BACKTEST_VARYING_PATHS = (
    ("_runtime_backtest_config",),
    ("input_identity",),
    ("backtest_config", "metadata", "prediction_hash"),
    ("backtest_config", "metadata", "preset_path"),
)


def _without(value, path: tuple[str, ...]):
    """``value`` with ``path`` removed, leaving every sibling in place."""
    if not isinstance(value, dict) or path[0] not in value:
        return value
    pruned = dict(value)
    if len(path) == 1:
        pruned.pop(path[0])
    else:
        pruned[path[0]] = _without(pruned[path[0]], path[1:])
    return pruned


def _comparable_backtest(spec: dict) -> dict:
    projected = spec
    for path in BACKTEST_VARYING_PATHS:
        projected = _without(projected, path)
    return projected


carried = _comparable_backtest(holdout_strategy_spec)
validation_carried = _comparable_backtest(CARRIER_SPEC)
if carried != validation_carried:
    moved = sorted(
        key
        for key in set(carried) | set(validation_carried)
        if carried.get(key) != validation_carried.get(key)
    )
    raise RuntimeError(f"the holdout strategy differs from the selected one in {moved}")
print(f"Holdout backtest identity {HOLDOUT_BACKTEST_HASH}")

# %%
with sqlite3.connect(REGISTRY_DB) as db:
    already = db.execute(
        "SELECT stage FROM backtest_runs WHERE backtest_hash = ?", (HOLDOUT_BACKTEST_HASH,)
    ).fetchone()
if already is None:
    result = run_backtest(
        CASE_STUDY_ID,
        HOLDOUT_PREDICTION_HASH,
        holdout_strategy_spec,
        prices=prices,
        predictions=predictions,
        label=HOLDOUT_LABEL,
        register=True,
        initial_cash=bt_config.initial_cash,
        calendar=bt_config.calendar,
    )
    print(
        f"Ran: Sharpe={result.metrics['sharpe']:.3f}, "
        f"max drawdown={result.metrics['max_drawdown']:.1%}"
    )
else:
    print(f"Already registered under stage {already[0]!r}; read, not re-run")

with sqlite3.connect(REGISTRY_DB) as db:
    registered_stage = db.execute(
        "SELECT stage FROM backtest_runs WHERE backtest_hash = ?", (HOLDOUT_BACKTEST_HASH,)
    ).fetchone()
if registered_stage is None:
    raise RuntimeError("the holdout backtest did not register")
if registered_stage[0] != "holdout":
    raise RuntimeError(
        f"the holdout backtest registered under stage {registered_stage[0]!r}; the stage is "
        "inferred from the prediction's split, so this row is not reachable as holdout evidence"
    )

# %% [markdown]
# ## 4. What the holdout says
#
# The two rows below are the same strategy on two windows. The validation figure
# is the one the configuration was chosen on and is optimistic by construction:
# it is the maximum of a search, and the maximum of a search is a biased estimate
# of the thing searched over. The holdout figure has no such bias from *this*
# case study's selection.
#
# It carries a different one. The window is a single year, and 2021 was a
# particular year - a broad advance in US equities - so a result this good in a
# year like that is weaker evidence than the same result across a decade. One
# window cannot separate a strategy that works from a strategy that suits the
# window it was tested on, and nothing downstream of here can either.

# %%
with sqlite3.connect(REGISTRY_DB) as db:
    comparison = pl.read_database(
        """
        SELECT b.backtest_hash, b.stage, m.sharpe, m.total_return, m.max_drawdown,
               m.volatility, m.num_trades, m.n_periods,
               m.sharpe_ci95_lo, m.sharpe_ci95_hi
        FROM backtest_runs b JOIN backtest_metrics m USING(backtest_hash)
        WHERE b.backtest_hash IN (?, ?)
        """,
        connection=db,
        execute_options={"parameters": [HOLDOUT_BACKTEST_HASH, SELECTED.hash]},
    )
if comparison.height != 2:
    raise RuntimeError("one of the two backtests has no registered metrics")

comparison = comparison.with_columns(
    pl.when(pl.col("backtest_hash") == HOLDOUT_BACKTEST_HASH)
    .then(pl.lit("holdout (2021)"))
    .otherwise(pl.lit("validation (selected on)"))
    .alias("window")
).sort("window")
comparison.select(
    "window",
    "sharpe",
    "sharpe_ci95_lo",
    "sharpe_ci95_hi",
    "total_return",
    "max_drawdown",
    "volatility",
    "num_trades",
    "n_periods",
)

# %% [markdown]
# The Sharpe intervals are the part to read first, and what they support is
# narrower than it looks.
#
# Each interval describes uncertainty in *its own* estimate. The holdout's says
# how much a single year of this strategy's weekly rebalances pins down its
# Sharpe; where it spans zero, that year does not establish that the strategy
# earns anything at all. The validation interval says the same for the
# development window, and carries the additional problem that its point estimate
# is the maximum of a search and is optimistic by an amount neither interval
# measures.
#
# What neither interval gives is a test of the *difference* between the two
# windows. That would need an interval for the gap itself, and the two are
# measured over disjoint periods on different numbers of observations, so it
# cannot be read off by asking whether one point estimate falls inside the
# other's interval. The cells below therefore report both intervals and the
# arithmetic difference, and stop there. Attributing the difference - to
# selection bias, to a different market, or to both - is not something one
# holdout year can do, and no notebook downstream of this one can either.

# %%
_holdout = comparison.filter(pl.col("window") == "holdout (2021)").row(0, named=True)
_validation = comparison.filter(pl.col("window") == "validation (selected on)").row(0, named=True)
_spans_zero = _holdout["sharpe_ci95_lo"] <= 0.0 <= _holdout["sharpe_ci95_hi"]
print(
    f"Validation Sharpe {_validation['sharpe']:.3f} "
    f"[{_validation['sharpe_ci95_lo']:.3f}, {_validation['sharpe_ci95_hi']:.3f}] "
    f"over {int(_validation['n_periods'])} sessions"
)
print(
    f"Holdout Sharpe {_holdout['sharpe']:.3f} "
    f"[{_holdout['sharpe_ci95_lo']:.3f}, {_holdout['sharpe_ci95_hi']:.3f}] "
    f"over {int(_holdout['n_periods'])} sessions"
)
print(f"Arithmetic difference {_holdout['sharpe'] - _validation['sharpe']:+.3f}, untested")
print(
    "The holdout interval "
    + (
        "spans zero, so this year does not establish that the strategy earns anything"
        if _spans_zero
        else "excludes zero, so this year places its Sharpe away from zero"
    )
)

# %% [markdown]
# The holdout equity path, drawn on its own axis. It is one year at weekly
# rebalances, so the shape carries far less information than the same picture
# over the validation window and is shown to make the drawdown legible rather
# than to support a claim.

# %%
holdout_returns = load_daily_returns_with_timestamp(CASE_STUDY_ID, HOLDOUT_BACKTEST_HASH)
if holdout_returns is None or holdout_returns.is_empty():
    raise RuntimeError("the holdout backtest registered no daily return series")

curve = holdout_returns.sort("timestamp").with_columns(
    ((1.0 + pl.col("ret")).cum_prod()).alias("equity")
)
curve = curve.with_columns((pl.col("equity") / pl.col("equity").cum_max() - 1.0).alias("drawdown"))

fig, (ax_equity, ax_dd) = plt.subplots(
    2,
    1,
    figsize=FIGSIZE["dual_v"],
    sharex=True,
    height_ratios=[2, 1],
    constrained_layout=True,
)
ax_equity.plot(curve["timestamp"], curve["equity"], color=COLORS["blue"], linewidth=1.6)
ax_equity.set_ylabel("Growth of 1")
ax_dd.fill_between(curve["timestamp"], curve["drawdown"], 0.0, color=COLORS["negative"], alpha=0.5)
ax_dd.set_ylabel("Drawdown")
add_message_title(
    ax_equity,
    f"The selected strategy on the 2021 holdout: Sharpe {_holdout['sharpe']:.2f}",
    f"Peak-to-trough {_holdout['max_drawdown']:.1%} over {curve.height} sessions",
)
fig.show()

# %% [markdown]
# ## Key takeaways
#
# 1. The strategy run here is the carrier's own specification, cloned and
#    re-pointed at the holdout predictions. Nothing about it was re-derived, so
#    a change upstream would show up as a different selection rather than as a
#    quietly different holdout.
# 2. The holdout prediction set is matched to the carrier by comparing the two
#    training specifications field by field, because a holdout refit is a
#    different training identity and cannot be matched on a hash.
# 3. A validation figure is the maximum of a search and is optimistic. A single
#    holdout year is unbiased with respect to that search and is noisy, and the
#    registered Sharpe interval is what says how noisy. Neither interval is a
#    test of the difference between the two, which is why none is reported.
# 4. A drop from validation to holdout has at least two sufficient explanations
#    - selection bias and a different market - and one window cannot separate
#    them. Neither can any notebook downstream of this one.
# 5. Survivorship bias bites hardest here. The universe is today's S&P 500
#    membership, so the 2021 window is evaluated on companies known to have
#    stayed in the index.
#
# **Next:** [`20_strategy_analysis`](20_strategy_analysis.ipynb) assembles what
# the case study has established across all of its stages, and states which of
# its claims this window supports.
