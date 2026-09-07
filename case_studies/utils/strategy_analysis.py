"""Strategy analysis figure helpers and assessment writer.

Companion to ``BacktestExplorer`` - produces the figures and structured
artifacts for each case study's ``strategy_analysis.py`` notebook.

Usage::

    from case_studies.utils.strategy_analysis import (
        plot_ic_vs_sharpe,
        plot_sharpe_waterfall,
        plot_concentration_curve,
        plot_cost_decay,
        plot_equity_drawdown,
        load_holdout_metrics,
        write_strategy_assessment,
        load_strategy_assessment,
    )
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from case_studies.utils.carrier_pins import CARRIER_PINS
from case_studies.utils.notebook_contracts import (
    degenerate_prediction_sql,
    full_coverage_prediction_sql,
)
from case_studies.utils.uncertainty import STAGE_SEQUENCE

# ---------------------------------------------------------------------------
# Canonical rank-1 resolution (LABEL_RESTRICTIONS-aware)
# ---------------------------------------------------------------------------
#
# Per-CS whitelist of labels eligible to anchor the registered strategy. The
# only entry today is sp500_options, restricted to ret_to_expiry because the
# four legacy diagnostic variants (fwd_ret_5d, fwd_ret_10d, fwd_ret_dh_5d,
# fwd_ret_dh_10d) were dropped from the sweep + registry 2026-05-17 - they
# went through the vectorized backtest path which treats their 5d/10d
# forward returns as daily returns, inflating Sharpes (e.g. fwd_ret_10d
# allocation Sharpe ~6.5) to non-credible levels. ret_to_expiry runs through
# the HTM daily-MTM cohort path and is the only label with an honest cost
# model for this CS. This is the only definition: ``20_strategy_synthesis/holdout.py``
# imports it from here rather than keeping a copy in sync by comment.
LABEL_RESTRICTIONS: dict[str, frozenset[str]] = {
    "sp500_options": frozenset({"ret_to_expiry"}),
}


# Per-CS canonical universe pin: case_study -> strategy.signal.universe_filter
# value eligible to anchor the registered rank-1. sp500_options trades only the
# liquid (bottom-quintile half-spread) subset - the full-universe round-trip
# option spread consumes the variance-risk-premium edge, so full-universe rows
# are excluded from rank-1 selection (the full universe is retained only for the
# Ch18 htm_cost_cascade comparison, never as the deployed carrier). Without this
# pin, full-universe allocation backtests registered by the standard sweep
# (e.g. the 2026-05-31 L1-grid rollout) leak into rank-1 by raw Sharpe and
# orphan the liquid-lineage holdout. This is the only definition; ``holdout.py``
# imports it, and ``select_best_models`` applies it by going through
# :func:`selectable_validation_candidates` rather than by repeating the filter.
UNIVERSE_RESTRICTIONS: dict[str, str] = {
    "sp500_options": "liquid",
}


# Carrier choices are owner-controlled in ``carrier_pins`` and use validation
# information only. The corrected S&P 500 options carrier is the liquid-universe
# cross-stage rank-1. Two alternative allocator rows tie its Sharpe exactly, so
# the deterministic tie-break preserves the simpler equal-weight baseline spec.


def rank_returns_on_common_support(
    returns_by_hash: dict[str, pl.DataFrame], *, periods_per_year: int
) -> pl.DataFrame:
    """Rank backtests after restricting every return series to exact common support."""
    if not returns_by_hash:
        raise ValueError("No return series supplied for common-support ranking")

    normalized: dict[str, pl.DataFrame] = {}
    common_timestamps: set[Any] | None = None
    for backtest_hash, frame in returns_by_hash.items():
        return_col = next(
            (name for name in ("daily_return", "return", "returns") if name in frame.columns),
            None,
        )
        if "timestamp" not in frame.columns or return_col is None:
            raise ValueError(
                f"{backtest_hash}: expected timestamp plus a return column; got {frame.columns}"
            )
        selected = (
            frame.select("timestamp", pl.col(return_col).alias("daily_return"))
            .with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
            .sort("timestamp")
        )
        if selected["timestamp"].n_unique() != selected.height:
            raise ValueError(f"{backtest_hash}: duplicate timestamps in daily returns")
        normalized[backtest_hash] = selected
        timestamps = set(selected["timestamp"].to_list())
        common_timestamps = (
            timestamps if common_timestamps is None else common_timestamps & timestamps
        )

    if common_timestamps is None or len(common_timestamps) < 2:
        raise ValueError("Backtest candidates have fewer than two common timestamps")

    from case_studies.utils.backtest_runner import compute_portfolio_metrics

    common = sorted(common_timestamps)
    common_frame = pl.DataFrame({"timestamp": common}, schema={"timestamp": pl.Datetime("ns")})
    common_ns = common_frame["timestamp"].cast(pl.Int64).to_list()
    rows: list[dict[str, Any]] = []
    for backtest_hash, frame in normalized.items():
        aligned = frame.join(common_frame, on="timestamp", how="inner").sort("timestamp")
        if aligned["timestamp"].cast(pl.Int64).to_list() != common_ns:
            raise ValueError(f"{backtest_hash}: failed exact common-support alignment")
        metrics = compute_portfolio_metrics(
            aligned["daily_return"].to_numpy(),
            periods_per_year=periods_per_year,
            uncertainty=False,
            trim_leading_zeros=False,
        )
        rows.append(
            {
                "backtest_hash": backtest_hash,
                "sharpe": float(metrics["sharpe"]),
                "n_periods": aligned.height,
                "start": common[0],
                "end": common[-1],
            }
        )
    return pl.DataFrame(rows).sort("sharpe", descending=True)


def rank_backtests_on_common_support(
    case_study: str, backtest_hashes: list[str], *, periods_per_year: int
) -> pl.DataFrame:
    """Load registered returns and rank them on their exact timestamp intersection."""
    from utils.paths import get_case_study_dir

    backtest_root = get_case_study_dir(case_study) / "run_log" / "backtest"
    returns_by_hash: dict[str, pl.DataFrame] = {}
    for backtest_hash in backtest_hashes:
        path = backtest_root / backtest_hash / "daily_returns.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing registered return artifact: {path}")
        returns_by_hash[backtest_hash] = pl.read_parquet(path)
    return rank_returns_on_common_support(returns_by_hash, periods_per_year=periods_per_year)


@dataclass(frozen=True)
class HoldoutSelfBacktest:
    """The outcome of looking for a validation run's holdout replay.

    ``select_holdout_self_backtest`` answers with a hash or with ``None``, and ``None``
    covers four different states of the registry: the validation backtest is not
    registered, its prediction set is not, no holdout prediction set exists for the
    configuration at all, or holdout backtests exist and none replays the validation
    strategy. A strategy-analysis notebook that raises on ``None`` therefore tells its
    reader nothing about which, and the most common of the four - the holdout stage has
    simply not been run yet - is a normal state for anyone working the notebooks in
    order rather than a defect.

    ``reason`` is a sentence for the rendered page. It names the validation run that was
    searched for, so a reader can see the search was well formed and is not being told
    that something went wrong.
    """

    backtest_hash: str | None
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.backtest_hash is not None


HoldoutRefitStatus = Literal["refit", "not_out_of_sample", "unattributable"]


def holdout_refit_status(training_spec_json: str | None) -> HoldoutRefitStatus:
    """What a training run's own specification says about how it was fitted.

    Three answers, not two, and the third is why this exists. Read from the training
    specification rather than from the prediction set's split, because the split says where
    the predictions land and says nothing about what the model saw while fitting - a model
    fitted on the validation folds can publish predictions over the holdout window, and that
    is exactly the mistake the holdout exists to rule out.

    ``refit``
        The run's CV declares the holdout fold. This is a holdout evaluation.
    ``not_out_of_sample``
        The run records a CV split and it is not the holdout. **This is a statement about
        the record, not a finding about the fit.** `20_strategy_synthesis/holdout.py`'s
        `generate_holdout` genuinely refits on a holdout fold and then registers the
        predictions under the *validation* training identity, whose CV says ``validation`` -
        so a row answering this way is either a validation-fitted model published over the
        holdout window or a real refit filed under the wrong identity, and the registry
        cannot tell them apart. Either way it may not be reported as a holdout result, and
        either way only an operator can say which it is.
    ``unattributable``
        The run records no CV split, so nothing can be concluded either way.

    Neither of the last two authorizes a deletion on its own, which is the correction to
    make about this function: a non-holdout CV declaration does not establish that no
    holdout evaluation occurred. What they authorize is a refusal that names the row, which
    is what nothing did before - the notebooks filtered on the two-valued predicate and
    these rows were invisible to the refusal and to everything after it.
    """
    if not training_spec_json:
        return "unattributable"
    try:
        cv = (json.loads(training_spec_json).get("computation") or {}).get("cv") or {}
    except (TypeError, ValueError):
        return "unattributable"
    split = cv.get("split")
    if split is None:
        return "unattributable"
    return "refit" if split == "holdout" else "not_out_of_sample"


def training_run_fitted_for_the_holdout(training_spec_json: str | None) -> bool:
    """True when a training run's own CV declares the holdout fold.

    This is what separates a refit from a validation-fitted model scored on a later
    window. See :func:`holdout_refit_status`, of which this is the two-valued reading: a
    run that cannot be shown to have been refitted answers False, which is the right
    default for a lineage lookup and the wrong one for deciding what to delete.
    """
    return holdout_refit_status(training_spec_json) == "refit"


def registered_holdout_generations(case_dir: str | Path) -> list[dict[str, Any]]:
    """Every holdout prediction set in a registry, with what its specification says.

    One implementation. This was copied into `etfs/18_holdout_predictions`,
    `sp500_options/16_holdout_predictions` and
    `us_firm_characteristics/15_holdout_predictions`, and the copies had already drifted:
    two of them delete through the schema-derived cascade in `registry/maintenance.py` and
    the third listed the child tables by hand, which is the version that aborts on a
    registry holding a `cohort_metrics.leader_hash` row.

    ``status`` is :func:`holdout_refit_status` on the training run behind each set.
    ``checkpoint`` is part of the identity rather than a detail of it: one training run
    publishes one prediction set per declared checkpoint, and moving the selection from one
    checkpoint to another is a different configuration evaluated on the same window.
    Identity on the training hash alone would read that as the same generation and let both
    stand.
    """
    import sqlite3

    db_path = Path(case_dir) / "run_log" / "registry.db"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT p.prediction_hash, p.training_hash, p.checkpoint_kind, p.checkpoint_value,
                   t.config_name, t.spec_json
            FROM prediction_sets p
            JOIN training_runs t ON t.training_hash = p.training_hash
            WHERE p.split = 'holdout'
            ORDER BY p.prediction_hash
            """
        ).fetchall()
    return [
        {
            "prediction_hash": prediction_hash,
            "training_hash": training_hash,
            "checkpoint": (checkpoint_kind, checkpoint_value),
            "config_name": config_name,
            "status": holdout_refit_status(training_spec_json),
            "refitted": holdout_refit_status(training_spec_json) == "refit",
        }
        for (
            prediction_hash,
            training_hash,
            checkpoint_kind,
            checkpoint_value,
            config_name,
            training_spec_json,
        ) in rows
    ]


@dataclass(frozen=True)
class HoldoutGenerationsToRetire:
    """What a holdout run finds already registered against its window, in three buckets.

    Splitting them is the point. The rule that governs each is different, and the code this
    replaces had only the first bucket - it filtered the registered generations on
    ``refitted``, so a row that was NOT a refit was invisible to the refusal and invisible
    to the deletion that follows it. Nothing owned removing one, and the note written when a
    stale `us_firm_characteristics` row was removed by hand on 2026-08-30 said exactly that:
    the next case study to reach the stage would accumulate the same second generation and
    the same silence.
    """

    superseded: tuple[dict[str, Any], ...]
    """Refits of a *different* configuration on the same window.

    A second holdout evaluation, and replacing one is a research decision rather than
    maintenance: deleting the rows does not undo having observed them. Whoever owns the
    sweep decides, which is what the notebooks' ``REPLACE_HOLDOUT`` switch is for.
    """

    not_out_of_sample: tuple[dict[str, Any], ...]
    """Rows whose training run records a CV split that is not the holdout.

    Two different things produce this record and it cannot separate them. One is a
    validation-fitted model publishing over the holdout window, the defect `29f13165`
    fixed, which is not a holdout evaluation at all. The other is a genuine refit filed
    under the validation training identity, which is what
    `20_strategy_synthesis/holdout.py`'s `generate_holdout` does - it builds a holdout fold,
    trains on it, and then registers the predictions against `candidate["training_hash"]`.

    So this bucket is refused rather than deleted. A row in it may not be reported as a
    holdout result, because on the first reading nothing out of sample was measured and on
    the second the identity is wrong; but deleting it automatically would, on the second
    reading, destroy a real evaluation on the strength of a record that is known to be
    unreliable for exactly this distinction.
    """

    unattributable: tuple[dict[str, Any], ...]
    """Rows whose training run records no CV split, so neither can be concluded.

    Refused, and with no way past: a missing specification is not evidence that a run was
    not refitted, and there is nothing recorded to adjudicate from.
    """


def holdout_generations_to_retire(
    case_dir: str | Path,
    *,
    this_generation: tuple[str, tuple[Any, Any]],
) -> HoldoutGenerationsToRetire:
    """Divide what is already registered against the holdout window by what governs it.

    ``this_generation`` is ``(training_hash, (checkpoint_kind, checkpoint_value))`` for the
    run about to be registered; a generation equal to it is this run and is in no bucket.
    """
    superseded, not_out_of_sample, unattributable = [], [], []
    for row in registered_holdout_generations(case_dir):
        if (row["training_hash"], row["checkpoint"]) == this_generation:
            continue
        if row["status"] == "refit":
            superseded.append(row)
        elif row["status"] == "not_out_of_sample":
            not_out_of_sample.append(row)
        else:
            unattributable.append(row)
    return HoldoutGenerationsToRetire(
        superseded=tuple(superseded),
        not_out_of_sample=tuple(not_out_of_sample),
        unattributable=tuple(unattributable),
    )


# What a holdout refit is allowed to change, and nothing else. Everything outside this set
# has to agree with the validation run, because the holdout is defined as *that configuration*
# refitted on a later window - not as another run that happens to share its name.
#
#   cv                        the definition of the refit: split, folds, identity, request
#   expected_prediction_keys  derived from the fold geometry, so it moves with cv
#   input_data_spec           carries the fold splits and a fingerprint over them. Its
#                             `artifacts` are still checked, through the top-level
#                             `feature_artifacts` that duplicates them.
#   runtime_identity          the environment the run happened in
#   source_identity           the notebook and commit that ran it
#   model.effective_params_by_fold   keyed by fold number, which the refit renumbers
#
# This is a denylist rather than an allowlist on purpose: a field added to the specification
# later is compared by default, so the check tightens as the spec grows instead of silently
# ignoring the new field.
_REFIT_MAY_CHANGE = frozenset(
    {"cv", "expected_prediction_keys", "input_data_spec", "runtime_identity", "source_identity"}
)


def _refit_comparable(training_spec_json: str | None) -> dict | None:
    """A training specification reduced to what a refit must preserve."""
    if not training_spec_json:
        return None
    computation = dict(json.loads(training_spec_json).get("computation") or {})
    for key in _REFIT_MAY_CHANGE:
        computation.pop(key, None)
    model = dict(computation.get("model") or {})
    if model:
        model.pop("effective_params_by_fold", None)
        computation["model"] = model
    return computation


def is_refit_of(holdout_spec_json: str | None, validation_spec_json: str | None) -> bool:
    """True when a holdout training run is the validation run's own configuration refitted.

    Family, configuration name, label and checkpoint are what the queries can filter on in
    SQL, and they are not enough to identify a configuration. They are a *name*, and a name is
    reused across generations: refit a study after its features change and the new runs carry
    the same four values as the old ones. Measured on the current registries, fx_pairs has 144
    configuration groups spanning more than one feature-artifact generation and etfs has 10 -
    so on those two case studies the coarse filter alone can return a holdout fitted on
    features the study no longer publishes, and report it as the selected carrier's.

    Comparing the specifications closes that. The feature artifact digests are the field that
    catches the stale generation, but the comparison is deliberately not limited to them: model
    hyperparameters, feature names, the task and the sampling all have to agree too, because a
    holdout that differs in any of them is not a refit of what was selected.

    A run with no recorded specification answers False. It cannot be shown to be a refit, and
    the holdout lineage is not a place to assume.
    """
    holdout = _refit_comparable(holdout_spec_json)
    validation = _refit_comparable(validation_spec_json)
    if holdout is None or validation is None:
        return False
    return holdout == validation


def resolve_holdout_self_backtest(
    case_study: str,
    val_backtest_hash: str,
    *,
    labels: Sequence[str] | None = None,
    admitted: frozenset[str] | None = None,
) -> HoldoutSelfBacktest:
    """Find the holdout replay of a validation run's strategy, or say why there is none.

    A thin wrapper over :func:`_resolve_holdout_self_backtest`: every "not found" answer
    goes through :func:`_refuse_a_selection_disagreement` first, so a caller that selected
    a different carrier than the holdout notebooks ran is told that rather than being told
    the holdout has not been produced.

    ``labels`` and ``admitted`` are the caller's own selection scope, and they are carried
    into that diagnosis rather than dropped. A carrier chosen from a preview's label subset
    or from a frozen candidate set is a correct answer within its scope, and comparing it
    against an unrestricted resolution would reject it for disagreeing with a question it
    was never asked.
    """
    found = _resolve_holdout_self_backtest(case_study, val_backtest_hash)
    if found.backtest_hash is None:
        _refuse_a_selection_disagreement(
            case_study, val_backtest_hash, labels=labels, admitted=admitted
        )
    return found


def _resolve_holdout_self_backtest(
    case_study: str,
    val_backtest_hash: str,
) -> HoldoutSelfBacktest:
    """Find the holdout backtest that replays a validation run's strategy, or say why not.

    The lookup itself. Callers go through :func:`resolve_holdout_self_backtest`, which adds
    the selection-disagreement diagnosis to every "not found" answer.

    This is the canonical ``val_rank1_self`` lineage anchor for the section 6 holdout
    closure: the holdout backtest produced by replaying the validation rank-1 strategy on
    the holdout prediction set. Matching by strategy spec, rather than by taking the
    highest holdout Sharpe among candidates sharing the ``training_hash``, keeps the
    lookup robust against experimental side-channel allocators - ``conformal_weighted``
    most of all - that share the holdout prediction set but diverge from the validation
    rank-1's allocator. Without that guard an allocator variant whose holdout Sharpe
    happens to be higher silently displaces the anchor, and the
    ``backtest_paired_metrics`` ``val_rank1_self`` pair, written against the canonical
    lineage's holdout hash, is then never found.

    The checkpoint is part of the model configuration, so the replay is pinned to the
    validation prediction set's own ``checkpoint_value`` and ``checkpoint_kind`` as well
    as its ``training_hash``. One trained model registers one prediction set per declared
    checkpoint and the strategy spec is identical across them, so ``training_hash`` alone
    leaves several indistinguishable holdout candidates. Resolving those by holdout
    Sharpe - which an earlier ``ORDER BY bm.sharpe DESC`` did - reads the holdout to
    choose among configurations, which ``reference/CASE_STUDY_PIPELINE.md`` section 6
    forbids outright.

    Raises when the pinned lineage is still ambiguous, rather than picking one.
    """
    import sqlite3

    from utils.paths import get_case_study_dir

    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    with sqlite3.connect(str(db_path)) as db:
        row = db.execute(
            "SELECT prediction_hash, spec_json FROM backtest_runs WHERE backtest_hash = ?",
            (val_backtest_hash,),
        ).fetchone()
        if row is None:
            return HoldoutSelfBacktest(
                None,
                f"the selected validation backtest {val_backtest_hash} is not registered in "
                f"{case_study}'s run log, so there is nothing to look for a holdout replay of",
            )
        val_pred_hash, val_spec_json = row
        val_strategy = json.loads(val_spec_json).get("strategy", {})

        train_row = db.execute(
            """
            SELECT training_hash, checkpoint_value, checkpoint_kind
            FROM prediction_sets WHERE prediction_hash = ?
            """,
            (val_pred_hash,),
        ).fetchone()
        if train_row is None:
            return HoldoutSelfBacktest(
                None,
                f"the selected validation backtest {val_backtest_hash} names prediction set "
                f"{val_pred_hash}, which is not registered, so the configuration to replay "
                "cannot be identified",
            )
        training_hash, checkpoint_value, checkpoint_kind = train_row

        val_train_row = db.execute(
            "SELECT family, config_name, label, spec_json FROM training_runs "
            "WHERE training_hash = ?",
            (training_hash,),
        ).fetchone()
        configuration = val_train_row[:3] if val_train_row is not None else None
        val_training_spec_json = val_train_row[3] if val_train_row is not None else None
        if configuration is None:
            return HoldoutSelfBacktest(
                None,
                f"the training run {training_hash} behind validation backtest "
                f"{val_backtest_hash} is not registered, so the configuration to look for a "
                "holdout refit of cannot be named",
            )

        # ``IS`` is SQLite's null-safe equality: a configuration with no
        # checkpoint dimension stores NULL on both sides and must still match,
        # while ``=`` would drop it.
        #
        # The join is on the declared configuration rather than on the validation
        # training hash. A holdout prediction produced correctly carries a NEW
        # training identity - it is the same configuration refitted on the holdout
        # fold, and the identity covers the CV interval - so matching on the
        # validation training hash can only ever find a holdout scored from the
        # validation-fitted model, which is the thing the holdout exists to avoid.
        candidates = db.execute(
            """
            SELECT b.backtest_hash, b.spec_json, t.training_hash, t.spec_json
            FROM backtest_runs b
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON t.training_hash = p.training_hash
            WHERE p.split = 'holdout'
              AND p.checkpoint_value IS ?
              AND p.checkpoint_kind IS ?
              AND t.family = ?
              AND t.config_name = ?
              AND t.label = ?
            ORDER BY b.backtest_hash
            """,
            (checkpoint_value, checkpoint_kind, *configuration),
        ).fetchall()

    checkpoint = f"checkpoint {checkpoint_kind}={checkpoint_value}"
    if not candidates:
        return HoldoutSelfBacktest(
            None,
            f"no holdout backtest is registered for the configuration behind validation run "
            f"{val_backtest_hash} ({configuration[0]}/{configuration[1]} on "
            f"{configuration[2]}, {checkpoint}), so the holdout has not been evaluated for "
            "this case study",
        )

    # Three conditions, and the SQL above can express none of them. The strategy spec has to
    # be the validation run's, so the anchor is a replay of what was selected rather than a
    # neighbouring allocator that shares the holdout prediction. The training run behind it has
    # to have been refitted for the holdout: a model fitted on the validation folds can publish
    # predictions over the holdout window, and accepting one would report the exact thing the
    # holdout exists to rule out. And it has to be a refit of THIS specification rather than of
    # a configuration with the same name - see `is_refit_of`, which is what stops a holdout
    # fitted on a superseded feature generation from being reported as the carrier's.
    matched = sorted(
        {
            bh
            for bh, spec_json, _, training_spec_json in candidates
            if json.loads(spec_json).get("strategy", {}) == val_strategy
            and training_run_fitted_for_the_holdout(training_spec_json)
            and is_refit_of(training_spec_json, val_training_spec_json)
        }
    )
    if not matched:
        return HoldoutSelfBacktest(
            None,
            f"{len(candidates)} holdout backtests are registered for "
            f"{configuration[0]}/{configuration[1]} on {configuration[2]} ({checkpoint}), and "
            "none of them replays that run's strategy from a run refitted for the holdout "
            "under the same specification, so the anchor is not a replay of what was "
            "selected",
        )
    if len(matched) > 1:
        raise ValueError(
            f"holdout replay for {val_backtest_hash} is ambiguous: {matched} are all "
            f"{configuration[0]}/{configuration[1]} on {configuration[2]}, refitted for the "
            f"holdout at {checkpoint}, with one strategy spec"
        )
    return HoldoutSelfBacktest(matched[0])


def _refuse_a_selection_disagreement(
    case_study: str,
    val_backtest_hash: str,
    *,
    labels: Sequence[str] | None = None,
    admitted: frozenset[str] | None = None,
) -> None:
    """Raise when a *different* carrier has the holdout the caller could not find.

    A strategy-analysis notebook that ranks its pool's Sharpe column directly can select a
    different configuration than `resolve_solvent_carrier` does - a Sharpe computed over a
    configuration's own available history is not comparable across configurations that
    priced different spans, so the raw ranking rewards whichever candidate had the most
    forgiving window. Measured on cme_futures (992 signal / 120 allocation / 28 risk_overlay
    backtests, rebuilt 2026-08-30): the raw column answered latent_factors/sdf on
    fwd_ret_21d at 1.274, the resolver gbm/leaves_31_mse on fwd_ret_5d at 1.236 raw and
    1.294 once compared over the 1,270 sessions they all price.

    The failure that followed was silent and read as the wrong thing. `17_holdout_predictions`
    and `18_holdout_backtest` resolve through `resolve_solvent_carrier`, so the notebook then
    asked for the holdout replay of a configuration those notebooks never ran, got None, and
    printed "not produced yet" while the holdout sat in the registry. A reader concluded the
    holdout had not been run.

    Only that exact shape raises: a *registered* hash other than the canonical one was asked
    about, and the canonical carrier has a replay. A case study whose holdout stage genuinely
    has not run reports "not produced yet" as before, which is a normal state for anyone
    working the notebooks in order. An unregistered hash keeps its own answer, which is more
    useful than this one - a hash the run log has never seen is a stale constant rather than
    a carrier chosen from a pool. And a resolver that cannot answer - an empty registry, a
    case study with no rank-1 - leaves the original answer standing rather than turning a
    missing holdout into a resolver error.
    """
    import sqlite3

    from utils.paths import get_case_study_dir

    # Only a hash that is actually registered can be a *selection*. When the caller names a
    # backtest the run log has never seen, `_resolve_holdout_self_backtest` already says
    # exactly that, and it is the more useful answer - a stale hardcoded hash, not a
    # carrier chosen from a pool.
    try:
        db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            registered = db.execute(
                "SELECT 1 FROM backtest_runs WHERE backtest_hash = ?", (val_backtest_hash,)
            ).fetchone()
    except sqlite3.Error:
        return
    if registered is None:
        return

    try:
        canonical = resolve_canonical_rank1_lineage(case_study, admitted=admitted, labels=labels)
    except Exception:  # noqa: BLE001 - this diagnoses; it must never replace the real answer
        return
    canonical_hash = canonical.get("val_backtest_hash")
    if not canonical_hash or canonical_hash == val_backtest_hash:
        return
    if canonical.get("holdout_backtest_hash") is None:
        return
    raise RuntimeError(
        f"{case_study}: no holdout replays validation backtest {val_backtest_hash}, but "
        f"{canonical['holdout_backtest_hash']} replays {canonical_hash} - "
        f"{canonical.get('family')}/{canonical.get('config_name')} on "
        f"{canonical.get('label')}, which is what `resolve_solvent_carrier` selects and what "
        "the holdout notebooks ran. This is a selection disagreement, not a missing holdout: "
        "the caller ranked its own pool and chose a different carrier. Resolve the carrier "
        "through `resolve_solvent_carrier`, which re-ranks candidates on exact common "
        "timestamp support and applies LABEL_RESTRICTIONS, UNIVERSE_RESTRICTIONS and "
        "CARRIER_PINS."
    )


def select_holdout_self_backtest(
    case_study: str,
    val_backtest_hash: str,
    *,
    labels: Sequence[str] | None = None,
    admitted: frozenset[str] | None = None,
) -> str | None:
    """The holdout replay's hash, or ``None`` when there is not exactly one.

    Kept for callers that only need the hash. A notebook that has to tell its reader
    what is missing wants ``resolve_holdout_self_backtest``, which carries the reason.
    """
    return resolve_holdout_self_backtest(
        case_study, val_backtest_hash, labels=labels, admitted=admitted
    ).backtest_hash


class NoSelectableCandidates(RuntimeError):
    """The case study currently publishes no configuration that may be selected.

    Every refusal `selectable_validation_candidates` raises for an empty pool is this
    type, so a caller asking *whether* a selection exists can answer "no" without
    swallowing an unrelated failure. It subclasses ``RuntimeError`` because that is what
    the selector raised before the type existed, and every caller that reports the refusal
    rather than branching on it keeps working unchanged.

    `has_holdout_predictions` is the caller that needs the distinction: it reports whether
    a holdout already covers the current top-N, and an initialised registry with no
    eligible validation backtest is a legitimate "not yet", not an error. It used to catch
    `ValueError`, which is what the pool it built itself raised; routing it through the
    canonical selector changed the type under it, and in
    `20_strategy_synthesis/00_holdout_predictions.py` that call sits outside the
    generation loop's handler, so one un-run case study would have stopped every case
    study after it.
    """


SELECTION_STAGES: tuple[str, ...] = ("signal", "allocation", "risk_overlay")
"""The pool a configuration is selected from.

``reference/CASE_STUDY_PIPELINE.md`` section 5: the selected configuration is the
highest validation backtest Sharpe across these three stages, exactly.
``cost_sensitivity`` is a perturbation analysis rather than an alternative strategy and
is excluded; the ``benchmark`` family is excluded separately, below.
"""


def selectable_validation_candidates(
    case_study: str,
    *,
    admitted: frozenset[str] | None = None,
    labels: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Every validation backtest this case study may select from, best Sharpe first.

    One implementation, because holdout selection had two. ``holdout.select_best_models``
    built its own pool out of ``BacktestExplorer.best`` per stage and this function's
    caller built one in SQL, and the two were held together by a comment. They applied
    different eligibility filters - membership on the prediction side there, a
    retired-set exclusion on both sides here - and different orderings, so they could
    name different configurations for the single holdout use. Measured on ``fx_pairs``
    2026-09-07: ``deep_learning/tcn`` on ``fwd_ret_21d`` has two backtests tied at Sharpe
    0.2639142245820113, and ``select_best_models`` answered ``9402978117e9`` while this
    path answered ``56070f34dff1``. Same model, two strategy specifications, and the
    holdout replays the specification exactly.

    Eligibility, in the order it is applied:

    * the stage pool (:data:`SELECTION_STAGES`) on ``split='validation'``, non-null
      Sharpe, ``family != 'benchmark'``, and no degenerate prediction set;
    * ``LABEL_RESTRICTIONS`` and ``UNIVERSE_RESTRICTIONS``, or a ``CARRIER_PINS`` entry
      where the owner has recorded one, which supersedes the ranking entirely;
    * ``families`` and ``labels``, the caller's own narrowing - ``labels`` replaces the
      declared restriction rather than adding to it, so a preview that narrowed its pool
      resolves inside the pool it narrowed to;
    * **publication**: an identity is selectable only where the population its producer
      publishes still lists it, asked per member kind and on both sides of the join.

    Publication is the membership question, not the exclusion one, and the two are not
    the same set. A prediction that no population ever listed was retired by nobody, so
    an exclusion set admits it and a membership set does not - which is how an
    experimental result its case study never published reaches a ranking.
    :func:`published_members_at` already subtracts what is retired, so membership is
    never the weaker test; where a registry declares no population of a kind it answers
    ``None`` and that kind places no constraint, which is the state of a fixture or of a
    registry written before the mechanism existed.

    ``admitted``, when given, is applied here rather than checked against the winner
    afterwards - see :func:`resolve_canonical_rank1_lineage`, whose common-support
    re-ranking is decided by the whole field and not only by the row that wins.

    Each candidate is a dict with ``backtest_hash``, ``prediction_hash``, ``stage``,
    ``training_hash``, ``family``, ``config_name``, ``label``, ``sharpe`` and
    ``spec_json``. Ordering is Sharpe descending, then the signal-only specification
    ahead of one carrying an allocation block, then ``backtest_hash`` ascending - a total
    order, so two callers ranking the same registry cannot disagree.
    """
    import sqlite3

    from case_studies.research.population import published_members_at
    from utils.paths import get_case_study_dir

    case_dir = get_case_study_dir(case_study)
    db_path = case_dir / "run_log" / "registry.db"
    label_filter = tuple(labels) if labels is not None else LABEL_RESTRICTIONS.get(case_study)
    universe_pin = UNIVERSE_RESTRICTIONS.get(case_study)
    carrier_pin = CARRIER_PINS.get(case_study)
    columns = (
        "backtest_hash",
        "prediction_hash",
        "stage",
        "training_hash",
        "family",
        "config_name",
        "label",
        "sharpe",
        "spec_json",
    )

    base_select = """
        SELECT b.backtest_hash, b.prediction_hash, b.stage,
               t.training_hash, t.family, t.config_name, t.label,
               bm.sharpe, b.spec_json
        FROM backtest_runs b
        JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
        JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
        JOIN training_runs t ON t.training_hash = p.training_hash
        JOIN prediction_metrics pm ON pm.prediction_hash = p.prediction_hash
    """
    # A Sharpe computed over fewer decision dates than its peers describes a different
    # sample, so it is not comparable with theirs and must not be ranked beside them
    # (`reference/CASE_STUDY_PIPELINE.md` section 10). `select_best_models` applied this
    # through `BacktestExplorer.best` and this resolver did not, which is one more way the
    # two pools could differ, and the collapsed path takes the stronger of the two rather
    # than the one that happened to be shorter. Measured across all nine production
    # registries 2026-09-07: it removes 18 of etfs' 2,038 ranked rows, none anywhere else,
    # and moves no rank-1.
    #
    # The explorer's other bar, `num_trades > 0`, is not carried over. It is not part of
    # the selection rule - a flat period is an observation of zero, not an abstention - and
    # it removes nothing from any of the nine.
    #
    # The maximum is taken WITHIN the published population, not across the whole table, and
    # the difference is not cosmetic. The bar keeps rows whose `ic_n_days` equals the
    # maximum for their `(split, family, label)`; a retired prediction scored over a longer
    # window sets that maximum, and every live row for the same family and label then falls
    # below it and is dropped by the query. Filtering publication afterwards cannot put them
    # back - they never came out of the database. `BacktestExplorer.best` passes the
    # population for exactly this reason, and dropping it while collapsing the two selectors
    # would have been a regression on the one case study whose refit changed a fold count.
    published_predictions = published_members_at(case_dir, member_kind="prediction")
    if published_predictions is not None and not published_predictions:
        raise NoSelectableCandidates(
            f"{case_study} declares prediction populations and publishes no prediction "
            "identities, so there is nothing it may select. Re-run the stage that publishes "
            "them rather than ranking over an empty population."
        )
    coverage_params: tuple = ()
    if published_predictions is None:
        coverage_bar = full_coverage_prediction_sql("p", "t", "pm")
    else:
        coverage_bar = full_coverage_prediction_sql(
            "p", "t", "pm", population_subquery="SELECT value FROM json_each(?)"
        )
        coverage_params = (json.dumps(sorted(published_predictions)),)

    if carrier_pin:
        # Documented a-priori carrier pin: resolve directly to the pinned
        # validation backtest rather than the max-Sharpe cross-stage rank-1.
        # The owner-controlled pin is a validation-time choice. Current-lineage
        # carrier decisions are deferred until all model producers finish.
        val_sql = base_select + (
            " WHERE b.backtest_hash LIKE ?"
            " AND p.split = 'validation'"
            " AND bm.sharpe IS NOT NULL"
            + degenerate_prediction_sql("p.prediction_hash")
            + coverage_bar
            + " ORDER BY bm.sharpe DESC, b.backtest_hash ASC"
        )
        params: tuple = (carrier_pin + "%",) + coverage_params
    else:
        stages = ",".join("?" for _ in SELECTION_STAGES)
        val_sql = base_select + (
            f" WHERE b.stage IN ({stages})"
            " AND p.split = 'validation'"
            " AND bm.sharpe IS NOT NULL"
            " AND t.family != 'benchmark'"
            + degenerate_prediction_sql("p.prediction_hash")
            + coverage_bar
        )
        params = tuple(SELECTION_STAGES) + coverage_params
        if label_filter:
            placeholders = ",".join("?" for _ in label_filter)
            val_sql += f" AND t.label IN ({placeholders})"
            params += tuple(label_filter)
        if families:
            placeholders = ",".join("?" for _ in families)
            val_sql += f" AND t.family IN ({placeholders})"
            params += tuple(families)
        if universe_pin:
            val_sql += " AND json_extract(b.spec_json, '$.strategy.signal.universe_filter') = ?"
            params += (universe_pin,)
        # Tie-break: among rows with identical Sharpe (e.g. the equal-weight baseline
        # equal-weight selection and its economically identical equal_weight
        # allocation-stage re-run, which share a prediction), prefer the
        # signal-only spec (no allocation block). That is the spec the holdout
        # is replayed from, so the canonical lineage stays poolable with its
        # holdout. Final ``backtest_hash`` key makes the order deterministic.
        val_sql += (
            " ORDER BY bm.sharpe DESC,"
            " (json_extract(b.spec_json, '$.strategy.allocation') IS NULL) DESC,"
            " b.backtest_hash ASC"
        )

    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute(val_sql, params).fetchall()
    finally:
        db.close()
    candidates = [dict(zip(columns, row, strict=True)) for row in rows]

    # A superseded generation is still complete, still `current` under its schema version,
    # and still ranks. `identity_status` says the registry understands the row; it says
    # nothing about whether the row is the one its producer still publishes, which is
    # recorded in the population lineage instead. Without this the carrier can be resolved
    # from a retired generation - measured on fx_pairs, where a rebuilt allocation stage
    # left the retired conformal-v2 backtest ranking first and every downstream notebook
    # refused it as unreproducible rather than selecting the generation in force.
    #
    # Both sides of the join, because a retired generation reaches the ranking through
    # either. The backtest side is the obvious one. The prediction side is the one that
    # survived unnoticed: a refit that changes no numbers - a relabel, a re-key, a rerun
    # that reproduces its inputs - publishes value-for-value identical predictions under a
    # new identity, so the old and new rows carry the SAME Sharpe to the last digit. On an
    # exact tie the ORDER BY returns whichever row it likes, and "whichever it likes" was
    # observed returning the retired one. Measured on sp500_equity_option_analytics: three
    # candidates at sharpe 1.965796084396144, and the resolver took a training run from a
    # superseded generation, against which a full 17-point cost surface was then registered.
    #
    # That is the shape worth remembering: the tie is produced BY CONSTRUCTION whenever a
    # refit changes nothing, so every lane that has ever superseded a population is exposed,
    # and the defect is invisible wherever no tie exists and silently wrong wherever one
    # does. Which is why it survived.
    #
    # The question is asked per NAME rather than globally, for both kinds. The naive
    # "retired by someone and listed by nobody in force" reads as equivalent and is not:
    # one identity is legitimately listed under several names, and a narrowed or preview
    # run keeps its own frozen snapshot in force forever.
    ranked = len(candidates)
    published_backtests = published_members_at(case_dir, member_kind="backtest")
    if published_backtests is not None and not published_backtests:
        raise NoSelectableCandidates(
            f"{case_study} declares backtest populations and publishes no backtest "
            "identities, so there is nothing it may select. Re-run the stage that publishes "
            "them rather than ranking over an empty population."
        )
    for key, published in (
        ("backtest_hash", published_backtests),
        ("prediction_hash", published_predictions),
    ):
        if published is None:
            continue
        candidates = [row for row in candidates if row[key] in published]

    if admitted is not None:
        admitted_before = len(candidates)
        candidates = [row for row in candidates if row["backtest_hash"] in admitted]
        if admitted_before and not candidates:
            raise NoSelectableCandidates(
                f"None of the {admitted_before} live validation backtests for {case_study} "
                f"is among the {len(admitted)} the frozen candidate set admits. The set and "
                "the registry describe different sweeps; re-freeze the set rather than "
                "selecting outside it."
            )
    if ranked and not candidates:
        raise NoSelectableCandidates(
            f"Every one of the {ranked} ranked validation backtests for {case_study} belongs "
            "to a superseded generation or to none the case study publishes, on the backtest "
            "side or the prediction side. The stages have been rebuilt and nothing was "
            "re-registered under a name still in force, so there is no configuration this "
            "case study currently publishes. Re-run the validation stages rather than "
            "selecting a retired one."
        )
    if not candidates:
        # Name the restriction that emptied the set. A pin is the one whose
        # failure is silent and total: it is a backtest-hash prefix, a hash
        # covers the whole strategy spec, and a rebuilt sweep produces new
        # ones - so a pin entered against an earlier registry matches nothing
        # and this is the first cell of the notebook that touches it. Reporting
        # only the label filter sent the reader to LABEL_RESTRICTIONS, which
        # was not the cause.
        if carrier_pin:
            raise NoSelectableCandidates(
                f"Carrier pin {carrier_pin!r} for {case_study} matches no validation "
                f"backtest in {db_path}. A pin is a backtest-hash prefix and every "
                "hash changes when the sweep is rebuilt, so a pin outlives at most "
                "one rebuild. Re-derive it from the current registry, or remove the "
                "entry from CARRIER_PINS to select by validation Sharpe."
            )
        raise NoSelectableCandidates(
            f"No validation rank-1 candidate for {case_study} (label_filter={label_filter})"
        )

    return _rank_on_common_support_where_a_conformal_candidate_is_present(case_study, candidates)


def _rank_on_common_support_where_a_conformal_candidate_is_present(
    case_study: str, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Re-order the field on exact common timestamp support, when one is needed.

    Every conformal calibration abstains, and the version decides only for how long.
    ``walk_forward_v2`` sits out the earliest fold entirely, because it calibrates from
    whole earlier folds and the earliest has none. ``walk_forward_v3`` calibrates from the
    fold's own elapsed history, which shortens the abstention to a warm-up - three
    decisions on an 8-hourly grid - and does not remove it. A candidate that holds nothing
    for its first N decisions books N returns of exactly zero and is ranked against
    allocators measured over the full span, so the field has to be re-ranked on common
    support either way.

    Reading the version to decide decided two things and was right about neither. As the
    trigger it switched the alignment off for a field of v3 candidates, which still need
    it. As the eligibility test it discarded every v3 conformal candidate from a field that
    also held a v2 one, which removes a live result from the comparison rather than
    aligning it. The version is read by neither: the property that matters is that a
    conformal candidate is present, and it is asked directly.

    This sits inside the shared field rather than in one caller, and that is the point. It
    changes the order, not just the winner: when it lived in ``resolve_canonical_rank1_lineage``
    alone, ``select_best_models`` read the same candidates in stored-Sharpe order, so on a
    case study with a conformal candidate the two agreed on eligibility and could still
    name different configurations - and the degeneracy fallback's rank-2 and rank-3 were
    ordered by a criterion the rank-1 was not.

    Each candidate keeps its registered ``sharpe`` and gains ``comparison_sharpe``, which is
    the Sharpe over the timestamps every candidate prices, or ``None`` where no re-ranking
    was needed. Callers reporting the selection's Sharpe want ``comparison_sharpe`` where it
    is set; callers reporting what the registry stored want ``sharpe``.
    """

    def _is_conformal(row: dict[str, Any]) -> bool:
        strategy = json.loads(row["spec_json"]).get("strategy", {})
        allocation = strategy.get("allocation") or {}
        return allocation.get("method") == "conformal_weighted"

    if not any(_is_conformal(row) for row in candidates):
        return [
            {**row, "comparison_sharpe": None, "comparison_n_periods": None} for row in candidates
        ]

    from case_studies.utils.uncertainty import periods_per_year_from_setup

    common_ranking = rank_backtests_on_common_support(
        case_study,
        [row["backtest_hash"] for row in candidates],
        periods_per_year=int(periods_per_year_from_setup(case_study)),
    )
    rank_rows = {row["backtest_hash"]: row for row in common_ranking.iter_rows(named=True)}
    comparison_n_periods = int(common_ranking["n_periods"][0])
    if any(
        rank_rows[row["backtest_hash"]]["n_periods"] != comparison_n_periods for row in candidates
    ):
        raise RuntimeError("Common-support ranking produced unequal n_periods")
    by_hash = {row["backtest_hash"]: row for row in candidates}
    return [
        {
            **by_hash[backtest_hash],
            "comparison_sharpe": float(rank_rows[backtest_hash]["sharpe"]),
            "comparison_n_periods": comparison_n_periods,
            "comparison_start": rank_rows[backtest_hash]["start"],
            "comparison_end": rank_rows[backtest_hash]["end"],
        }
        for backtest_hash in common_ranking["backtest_hash"].to_list()
    ]


def resolve_canonical_rank1_lineage(
    case_study: str,
    *,
    admitted: frozenset[str] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve the canonical val rank-1 + matching holdout for a case study.

    Cross-stage validation rank-1 is selected over stage IN (signal,
    allocation, risk_overlay) with LABEL_RESTRICTIONS applied where defined.
    When a conformal candidate is present at any calibration version, every
    candidate - conformal or not - is re-ranked on exact common timestamp
    support, because a conformal allocator abstains until it is calibrated and
    books zeros over the abstention. Holdout match is by
    training_hash on the rank-1's prediction set. Use this in every strategy_analysis notebook
    rather than hardcoding hashes - hardcoded hashes go stale every time the
    sweep is rebuilt, and queries that forget LABEL_RESTRICTIONS surface the
    diagnostic-variant rows (sp500_options' fwd_ret_10d Sharpe ≈ 9.7) as
    bogus rank-1 candidates.

    ``admitted``, when given, is the set of backtest hashes a case study has frozen as
    the field this selection may choose from - a ``CandidateSet``'s members. It is applied
    to the candidates BEFORE the ranking, not checked against the winner afterwards,
    because the two are not the same test whenever the conformal branch is taken: the
    common-support ranking restricts every series to the timestamps they all share, so a
    candidate that is never going to win still decides how far the intersection reaches
    and therefore which admitted candidate does. Checking membership after the fact passes
    while the answer has already been changed by a row that was never eligible. Default
    None keeps the whole registry in the field, which is what every case study that
    freezes no set wants.

    Returns a dict with keys ``val_backtest_hash``, ``val_prediction_hash``,
    ``val_stage``, ``val_sharpe``, ``training_hash``, ``family``,
    ``config_name``, ``label``, ``holdout_backtest_hash``,
    ``holdout_prediction_hash``, ``holdout_sharpe`` (holdout fields are
    None when no matching holdout row exists yet).
    """
    import sqlite3

    from utils.paths import get_case_study_dir

    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    # `labels` overrides the module-level restriction rather than adding to it. A preview
    # run narrows its pool with PREVIEW_LABELS and the resolver knew nothing about that, so
    # it could resolve a carrier on a label the pool excludes - the carrier is then not in
    # the pool, and the notebook reports it missing. The default is the declared
    # restriction, which is what every canonical run wants. The pool itself is
    # `selectable_validation_candidates`, which `holdout.select_best_models` also ranks,
    # so the two cannot name different carriers for the single holdout use.
    candidates = selectable_validation_candidates(case_study, admitted=admitted, labels=labels)
    val = candidates[0]
    # The field is already in its final order, common-support re-ranking included, so the
    # rank-1 is read rather than recomputed. `comparison_sharpe` is set exactly where that
    # re-ranking ran, and it is the Sharpe over the timestamps every candidate prices - the
    # only one the candidates can be compared on.
    val_sharpe = (
        float(val["comparison_sharpe"])
        if val["comparison_sharpe"] is not None
        else float(val["sharpe"])
    )
    comparison_n_periods: int | None = val["comparison_n_periods"]
    comparison_start = val.get("comparison_start")
    comparison_end = val.get("comparison_end")

    val_bh = val["backtest_hash"]
    val_ph = val["prediction_hash"]
    val_stage = val["stage"]
    train_h = val["training_hash"]
    family = val["family"]
    config_name = val["config_name"]
    label = val["label"]

    # Match holdout by strategy spec to the val rank-1 backtest, so an
    # experimental side-channel allocator (e.g., conformal_weighted) on
    # the same holdout pred set does not displace the canonical lineage.
    # The raw lookup, not the diagnosing wrapper. `_refuse_a_selection_disagreement` asks
    # this function for the canonical carrier, so going through the wrapper here would
    # re-enter it: every call would resolve the whole ranking again, one level deeper,
    # until the recursion limit - and the diagnosis would swallow the RecursionError after
    # hundreds of registry reads. There is nothing to diagnose on this path anyway: this
    # function IS the canonical selection, so it can never disagree with itself.
    ho_bh = _resolve_holdout_self_backtest(case_study, val_bh).backtest_hash
    ho_ph: str | None = None
    ho_sharpe: float | None = None
    if ho_bh is not None:
        db = sqlite3.connect(str(db_path))
        try:
            ho_row = db.execute(
                """
                SELECT b.prediction_hash, bm.sharpe
                FROM backtest_runs b
                LEFT JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
                WHERE b.backtest_hash = ?
                """,
                (ho_bh,),
            ).fetchone()
        finally:
            db.close()
        if ho_row is not None:
            ho_ph, ho_sharpe = ho_row

    return {
        "val_backtest_hash": val_bh,
        "val_prediction_hash": val_ph,
        "val_stage": val_stage,
        "val_sharpe": val_sharpe,
        "comparison_n_periods": comparison_n_periods,
        "comparison_start": comparison_start,
        "comparison_end": comparison_end,
        "training_hash": train_h,
        "family": family,
        "config_name": config_name,
        "label": label,
        "holdout_backtest_hash": ho_bh,
        "holdout_prediction_hash": ho_ph,
        "holdout_sharpe": ho_sharpe,
    }


def allocation_method_of(case_study: str, backtest_hash: str | None) -> str:
    """The allocation method one backtest declares, read from its own specification.

    A page that prints an allocator's name beside a Sharpe has to take both from one row.
    `BacktestExplorer.compare_allocators` cannot supply the pair: it ranks allocators by
    their MEAN Sharpe across the configurations each was run on, so the allocator it puts
    first is routinely not the one on the single highest-Sharpe row - the sweeps vary
    concentration per prediction and per allocator, and mean and maximum then disagree.
    Measured on the production registries 2026-09-07, on each case study's leading
    allocation-stage row: `etfs` printed `risk_parity` beside a Sharpe of 0.8769 that `hrp`
    produced, and `sp500_equity_option_analytics` printed `mvo_ledoit_wolf` beside 2.0408,
    also `hrp`. The other four agreed.

    Scoping the comparison to that row's own prediction narrows the disagreement and does
    not close it, because mean and maximum still differ within one prediction. Reading the
    specification closes it: there is exactly one allocator on the row.

    Ask this of an allocation-stage row. A row there with no allocation block is the
    equal-weight baseline re-run, and equal weight is an allocation decision
    (``reference/CASE_STUDY_PIPELINE.md`` section 12), so it is named rather than left
    blank. A risk-overlay row can also carry no allocation block, meaning something else
    entirely, and this would misname it.

    The strategy block is read through ``strategy_view`` rather than by indexing
    ``spec["strategy"]``, so this and ``compare_allocators`` cannot come to disagree about
    which specification shapes carry one.
    """
    import sqlite3

    from case_studies.utils.backtest_presets import strategy_view
    from utils.paths import get_case_study_dir

    if not backtest_hash:
        return ""
    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT spec_json FROM backtest_runs WHERE backtest_hash = ?", (backtest_hash,)
        ).fetchone()
    if row is None or not row[0]:
        return ""
    allocation = strategy_view(json.loads(row[0])).get("allocation") or {}
    return allocation.get("method") or "equal_weight"


INSOLVENT_MAX_DRAWDOWN = -1.0
"""Drawdown at or past which a run's equity reached zero.

A long-short book with no margin call keeps compounding through zero, so every
metric a run reports after that point is arithmetic on a balance that no longer
exists - including a Sharpe high enough to top a ranking.
"""


def resolve_solvent_carrier(
    case_study: str,
    *,
    require_solvent: bool = True,
    admitted: frozenset[str] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The configuration downstream notebooks run, with its spec and drawdown.

    Cost sensitivity, holdout prediction and holdout backtest all have to run the
    configuration the case study reports, and that configuration is
    ``resolve_canonical_rank1_lineage``'s validation rank-1. Each notebook ranking
    the registry for itself is a standing divergence class rather than a
    hypothetical one: the canonical resolver re-ranks strict-conformal
    candidates on exact common timestamp support and applies LABEL_RESTRICTIONS,
    UNIVERSE_RESTRICTIONS and CARRIER_PINS, and a plain Sharpe ranking beside it
    does none of those. When the two disagree, the cost curve describes a strategy
    the chapter does not report and the strategy-analysis notebook finds no cost
    rows for the carrier it selected.

    Solvency is checked here rather than inside the canonical resolver because the
    two questions are different: the resolver decides which configuration the case
    study is about, and this decides whether that configuration is one anything can
    be measured on. An insolvent or unmeasured carrier raises. It deliberately does
    not fall through to the runner-up - that would hand downstream notebooks a
    configuration the chapter does not report, which is the divergence this
    function exists to close. Selecting past a bankrupt rank-1 is a decision for
    whoever owns the sweep, taken by fixing the sweep or pinning a carrier.

    ``labels`` narrows the candidate pool the same way ``LABEL_RESTRICTIONS`` does and
    replaces it. A notebook run under preview restricts its own pool with ``PREVIEW_LABELS``
    while this resolver read only the module-level restriction, so it could resolve a
    carrier on a label the pool excludes - and the carrier is then simply not found in the
    pool, which reads as a missing result rather than as two different questions. Pass the
    same restriction to both.

    Returns ``resolve_canonical_rank1_lineage``'s dict with ``spec_json`` and
    ``max_drawdown`` for the validation rank-1 added.
    """
    import sqlite3

    from utils.paths import get_case_study_dir

    lineage = resolve_canonical_rank1_lineage(case_study, admitted=admitted, labels=labels)
    backtest_hash = lineage["val_backtest_hash"]

    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    db = sqlite3.connect(str(db_path))
    try:
        row = db.execute(
            """
            SELECT b.spec_json, bm.max_drawdown
            FROM backtest_runs b
            LEFT JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            WHERE b.backtest_hash = ?
            """,
            (backtest_hash,),
        ).fetchone()
    finally:
        db.close()
    if row is None:
        raise RuntimeError(
            f"Canonical rank-1 {backtest_hash} for {case_study} is not in backtest_runs "
            f"of {db_path}. The resolver and this lookup read the same registry, so this "
            "means the registry changed under the process."
        )
    spec_json, max_drawdown = row

    if require_solvent:
        if max_drawdown is None:
            raise RuntimeError(
                f"Canonical rank-1 {backtest_hash} for {case_study} has no recorded "
                "max_drawdown, so it cannot be shown to have survived. Re-run the backtest "
                "so its metrics are registered, rather than sweeping a run whose equity path "
                "is unknown."
            )
        if max_drawdown <= INSOLVENT_MAX_DRAWDOWN:
            raise RuntimeError(
                f"Canonical rank-1 {backtest_hash} for {case_study} reached zero equity "
                f"(max_drawdown={max_drawdown:.4f}). Its Sharpe of {lineage['val_sharpe']:.3f} "
                "is computed on a balance that no longer exists, so nothing measured "
                "downstream of it means anything. Fix the sweep, or pin a carrier in "
                "CARRIER_PINS, rather than selecting past this row silently."
            )

    _assert_carrier_calibration_is_current(case_study, backtest_hash, spec_json)

    return {**lineage, "spec_json": spec_json, "max_drawdown": max_drawdown}


def _assert_carrier_calibration_is_current(
    case_study: str, backtest_hash: str, spec_json: str | None
) -> None:
    """Refuse a carrier whose conformal calibration the code will not run.

    Selection ranks on Sharpe and knows nothing about calibration versions, so a
    backtest fitted under a retired contract stays selectable after the contract is
    corrected. Nothing downstream can execute it - ``run_backtest`` refuses a
    non-current ``calibration_version`` outright - so the first sign is a whole stage
    failing on a configuration that ranked first. On ``us_firm_characteristics``,
    2026-08-30, all 52 conformal backtests were the retired version and 11 of 11 cost
    levels died; the v3 refit happened to win on merit, so the wrong carrier was never
    actually carried, but nothing would have caught it if it had not.

    This raises rather than selecting past the row. Excluding retired rows inside the
    resolver would change which configuration seven case studies report, silently, and
    that is a decision for whoever owns the sweep - taken by re-running the stage under
    the current calibration, not by a filter nobody sees.
    """
    import json

    from case_studies.utils.conformal import CALIBRATION_VERSION

    if not spec_json:
        return
    try:
        allocation = (json.loads(spec_json).get("strategy") or {}).get("allocation") or {}
    except (TypeError, ValueError):
        return
    recorded = allocation.get("calibration_version")
    if recorded is None or recorded == CALIBRATION_VERSION:
        return
    raise RuntimeError(
        f"Canonical rank-1 {backtest_hash} for {case_study} was fitted under conformal "
        f"calibration {recorded!r}, and the current contract is {CALIBRATION_VERSION!r}. "
        "Nothing downstream can execute it. Re-run the allocation stage so the sweep "
        "holds current-calibration candidates, rather than measuring costs or a holdout "
        "on a configuration that cannot be reproduced."
    )


# ---------------------------------------------------------------------------
# Spine CI / kill-gate helpers (tri-state contract)
# ---------------------------------------------------------------------------

CIStatus = Literal["excludes_zero_strong", "straddles_zero", "no_data"]
GateStatus = Literal["pass", "fail", "no_data"]


def _missing(x: float | None) -> bool:
    """True when a bound/estimate carries no information.

    NaN reaches these gates whenever a paired bootstrap could not be computed
    (no pair row in ``backtest_paired_metrics``). It must be treated exactly
    like ``None``: every comparison against NaN is False, so an unguarded NaN
    silently falls through to ``pass``.
    """
    return x is None or not np.isfinite(x)


def ci_status(lo: float | None, hi: float | None) -> CIStatus:
    """Three-tier CI continuum used uniformly across spine §3 / §6 / §7.

    `no_data` is reserved for missing CI bounds (upstream bootstrap not run
    or registry NULLs); it is *not* a low-credibility classification.
    """
    if _missing(lo) or _missing(hi):
        return "no_data"
    if lo > 0 or hi < 0:
        return "excludes_zero_strong"
    return "straddles_zero"


def gate1_validation_sharpe_geq_zero(sharpe_ci_lo: float | None) -> GateStatus:
    """Kill gate 1: validation full-period Sharpe CI lower bound ≥ 0.

    Returns ``no_data`` when the CI lower bound is missing.
    """
    if _missing(sharpe_ci_lo):
        return "no_data"
    return "pass" if sharpe_ci_lo >= 0 else "fail"


def gate2_holdout_diff_not_excludes_zero_negatively(
    diff_ci_status: CIStatus, sharpe_diff: float | None
) -> GateStatus:
    """Kill gate 2: holdout strategy-vs-EW Sharpe-diff CI does not exclude
    zero on the negative side.

    Pass: diff CI does not strongly exclude zero, OR strongly excludes zero
    on the positive side. Fail: diff CI strongly excludes zero AND the
    point estimate is negative. ``no_data`` when the diff CI status is
    ``no_data`` or ``sharpe_diff`` is missing.
    """
    if diff_ci_status == "no_data" or _missing(sharpe_diff):
        return "no_data"
    if diff_ci_status == "excludes_zero_strong" and sharpe_diff < 0:
        return "fail"
    return "pass"


def fmt_gate(status: GateStatus) -> str:
    """Display label for a gate status in printed kill-gate summaries."""
    return {"pass": "PASS", "fail": "FAIL", "no_data": "NO DATA"}[status]


def gate_passes(status: GateStatus) -> bool | None:
    """JSON-serializable view: True for pass, False for fail, None for
    no_data. Replaces ``bool(gate_pass)`` in ``strategy_assessment.json``
    so missing-CI cases are not silently coerced to True.
    """
    return {"pass": True, "fail": False, "no_data": None}[status]


# ---------------------------------------------------------------------------
# Holdout metrics loader
# ---------------------------------------------------------------------------


def load_holdout_metrics(case_study: str) -> dict[str, Any]:
    """Load holdout prediction + backtest metrics from the registry.

    Returns dict with keys: available, holdout_sharpe, holdout_ic,
    holdout_cagr, holdout_maxdd, family, config_name, label.
    All values are None if no holdout data exists.
    """
    import sqlite3

    from utils.paths import get_case_study_dir

    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    result: dict[str, Any] = {
        "available": False,
        "holdout_sharpe": None,
        "holdout_ic": None,
        "holdout_cagr": None,
        "holdout_maxdd": None,
        "family": None,
        "config_name": None,
        "label": None,
    }
    if not db_path.exists():
        return result

    db = sqlite3.connect(str(db_path))
    try:
        row = db.execute(
            """
            SELECT tr.family, tr.config_name, tr.label,
                   pm.ic_mean,
                   bm.sharpe, bm.cagr, bm.max_drawdown
            FROM prediction_sets ps
            JOIN training_runs tr ON ps.training_hash = tr.training_hash
            LEFT JOIN prediction_metrics pm
                ON ps.prediction_hash = pm.prediction_hash
            LEFT JOIN backtest_runs br
                ON ps.prediction_hash = br.prediction_hash AND br.stage = 'signal'
            LEFT JOIN backtest_metrics bm
                ON br.backtest_hash = bm.backtest_hash
            WHERE ps.split = 'holdout'
            ORDER BY bm.sharpe DESC NULLS LAST, pm.ic_mean DESC NULLS LAST
            LIMIT 1
            """,
        ).fetchone()
        if row:
            holdout_sharpe, holdout_cagr, holdout_maxdd = row[4], row[5], row[6]
            available = (
                holdout_sharpe is not None
                and holdout_cagr is not None
                and holdout_maxdd is not None
            )
            result.update(
                available=available,
                family=row[0],
                config_name=row[1],
                label=row[2],
                holdout_ic=row[3],
                holdout_sharpe=holdout_sharpe,
                holdout_cagr=holdout_cagr,
                holdout_maxdd=holdout_maxdd,
            )
    finally:
        db.close()
    return result


# ---------------------------------------------------------------------------
# Figure 1: IC vs Signal-Stage Sharpe
# ---------------------------------------------------------------------------


def plot_ic_vs_sharpe(
    explorer,
    *,
    highlight_sources: list[str] | None = None,
    ew_sharpe: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """IC vs equal-weight baseline Sharpe scatter with annotations.

    Parameters
    ----------
    explorer : BacktestExplorer
    highlight_sources : list[str], optional
        Model sources to highlight (e.g. model_analysis recommendations).
    ew_sharpe : float, optional
        Equal-weight benchmark Sharpe (drawn as horizontal line).
    ax : plt.Axes, optional

    Returns
    -------
    plt.Figure
    """
    # Load all equal-weight baseline backtests
    all_bt = explorer.best(stage="signal", top_n=9999)
    if all_bt.is_empty():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No signal backtests", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    ic = all_bt["ic_mean"].to_numpy()
    sharpe = all_bt["sharpe"].to_numpy()
    sources = all_bt["source"].to_list()
    families = all_bt["family"].to_list()

    # Base scatter (all points, light gray)
    ax.scatter(ic, sharpe, c="lightgray", s=20, alpha=0.5, zorder=1, label="_all")

    # Highlight recommended models
    if highlight_sources:
        mask = np.array([s in highlight_sources for s in sources])
        if mask.any():
            # Color by family
            family_colors = _family_color_map()
            highlighted_families = [families[i] for i in range(len(families)) if mask[i]]
            colors = [family_colors.get(f, "#333333") for f in highlighted_families]
            ax.scatter(
                ic[mask],
                sharpe[mask],
                c=colors,
                s=60,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            # Add family legend
            seen = set()
            for f in highlighted_families:
                if f not in seen:
                    ax.scatter([], [], c=family_colors.get(f, "#333333"), s=60, label=f)
                    seen.add(f)

    # Annotate top 3
    top_idx = np.argsort(sharpe)[-3:]
    for idx in top_idx:
        label = sources[idx].split("/")[-1]
        ax.annotate(
            label,
            (ic[idx], sharpe[idx]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            alpha=0.8,
        )

    # EW benchmark line
    if ew_sharpe is not None:
        ax.axhline(
            ew_sharpe,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"EW baseline ({ew_sharpe:.2f})",
        )

    ax.set_xlabel("Information Coefficient (IC)")
    ax.set_ylabel("Signal-Stage Sharpe")
    ax.set_title("Signal Quality vs Strategy Performance")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Sharpe Progression Waterfall (Locked Lineage)
# ---------------------------------------------------------------------------


def plot_sharpe_waterfall(
    lineage: dict[str, dict],
    *,
    ax: plt.Axes | None = None,
    ci_lo: dict[str, float] | None = None,
    ci_hi: dict[str, float] | None = None,
) -> plt.Figure:
    """Locked lineage waterfall: signal -> allocation -> cost -> risk.

    Parameters
    ----------
    lineage : dict
        From ``BacktestExplorer.champion_lineage()``.
    ax : plt.Axes, optional
    ci_lo, ci_hi : dict, optional
        Block-bootstrap 95% CI bounds keyed by stage name. When supplied,
        plotted as asymmetric error bars on each stage's bar.

    Returns
    -------
    plt.Figure
    """
    stage_order = list(STAGE_SEQUENCE)
    stage_labels = {
        "signal": "Signal",
        "allocation": "Allocation",
        "cost_sensitivity": "Costs",
        "risk_overlay": "Risk Overlay",
    }

    stages: list[str] = []
    stage_keys: list[str] = []
    sharpes: list[float] = []
    annotations: list[str] = []

    for s in stage_order:
        if s not in lineage:
            continue
        info = lineage[s]
        stages.append(stage_labels[s])
        stage_keys.append(s)
        sharpes.append(info["sharpe"])

        if s == "signal":
            method = info.get("signal_method", "")
            top_k = info.get("top_k", "")
            annotations.append(f"{method}\nk={top_k}" if top_k else method)
        elif s == "allocation":
            annotations.append(info.get("allocator", ""))
        elif s == "cost_sensitivity":
            cost = info.get("cost_bps", "?")
            annotations.append(f"{cost} bps")
        elif s == "risk_overlay":
            annotations.append(info.get("risk_name", ""))

    if not stages:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No lineage data", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    x = np.arange(len(stages))
    colors: list[str] = []
    for i in range(len(sharpes)):
        if i == 0:
            colors.append("#2196F3")
        elif sharpes[i] >= sharpes[i - 1]:
            colors.append("#4CAF50")
        else:
            colors.append("#F44336")

    bars = ax.bar(x, sharpes, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Track which CIs actually bracket the point estimate so the value
    # labels below anchor on the upper bar edge instead of a stale ``ci_hi``
    # that sits below the bar top.
    ci_brackets_point: set[int] = set()
    skipped_ci_stages: list[str] = []
    if ci_lo is not None and ci_hi is not None:
        err_lo = []
        err_hi = []
        valid_idx = []
        valid_centers = []
        for i, k in enumerate(stage_keys):
            lo = ci_lo.get(k)
            hi = ci_hi.get(k)
            if lo is None or hi is None:
                continue
            # Robustness: stale CIs from earlier engine runs may not
            # bracket the current point estimate. Skip those instead of
            # raising in matplotlib, but log the staleness so the
            # data-quality issue surfaces in notebook output rather than
            # only showing up as an absent error bar.
            if lo > sharpes[i] or hi < sharpes[i]:
                skipped_ci_stages.append(k)
                continue
            ci_brackets_point.add(i)
            valid_idx.append(i)
            err_lo.append(sharpes[i] - lo)
            err_hi.append(hi - sharpes[i])
            valid_centers.append(sharpes[i])
        if valid_idx:
            ax.errorbar(
                np.array(valid_idx),
                np.array(valid_centers),
                yerr=np.array([err_lo, err_hi]),
                fmt="none",
                ecolor="#333333",
                elinewidth=1.2,
                capsize=4,
                zorder=4,
            )
        if skipped_ci_stages:
            import warnings

            warnings.warn(
                "plot_sharpe_waterfall: dropped CIs not bracketing the "
                f"point estimate for stages={skipped_ci_stages}; rerun "
                "uncertainty backfill to refresh.",
                stacklevel=2,
            )

    # value labels - always above the upper edge so they don't overlap a CI bar
    for i, (bar, val) in enumerate(zip(bars, sharpes, strict=False)):
        # Only use ci_hi as the anchor when the CI actually brackets the
        # point estimate (see ci_brackets_point above); otherwise the
        # stale ``ci_hi`` can sit below ``val`` and pull the label inside
        # the bar.
        if i in ci_brackets_point:
            top = ci_hi[stage_keys[i]]
        else:
            top = max(val, 0)
        offset = max(abs(top) * 0.04, 0.02)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top + offset,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    for i in range(1, len(sharpes)):
        delta = sharpes[i] - sharpes[i - 1]
        color = "#4CAF50" if delta >= 0 else "#F44336"
        sign = "+" if delta >= 0 else ""
        anchor = max(sharpes[i], sharpes[i - 1])
        offset = max(abs(anchor) * 0.12, 0.08)
        ax.annotate(
            f"{sign}{delta:.3f}",
            xy=(i - 0.5, anchor + offset),
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    for i, ann in enumerate(annotations):
        if ann:
            ax.text(
                i,
                -0.03,
                ann,
                ha="center",
                va="top",
                fontsize=8,
                color="gray",
                transform=ax.get_xaxis_transform(),
            )

    ax.axhline(0, color="#9E9E9E", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Lineage: Sharpe Through Pipeline Stages")

    # Symmetric padding accommodates both positive and negative regimes plus
    # any error bars that extend beyond the bar tops.
    if ci_lo and ci_hi:
        lo_extents = [ci_lo[k] for k in stage_keys if k in ci_lo and ci_lo[k] is not None]
        hi_extents = [ci_hi[k] for k in stage_keys if k in ci_hi and ci_hi[k] is not None]
        all_lo = list(sharpes) + lo_extents
        all_hi = list(sharpes) + hi_extents
    else:
        all_lo = list(sharpes)
        all_hi = list(sharpes)
    lo_lim = min(all_lo + [0])
    hi_lim = max(all_hi + [0])
    span = hi_lim - lo_lim
    pad = max(span * 0.18, 0.15)
    ax.set_ylim(lo_lim - pad, hi_lim + pad)

    return fig


# ---------------------------------------------------------------------------
# Figure 3: Concentration Curve (top_k analysis)
# ---------------------------------------------------------------------------


def plot_concentration_curve(
    conc_df: pl.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Sharpe vs top_k for concentration analysis.

    Parameters
    ----------
    conc_df : pl.DataFrame
        From ``BacktestExplorer.concentration_curve()``.
    ax : plt.Axes, optional

    Returns
    -------
    plt.Figure
    """
    if conc_df.is_empty():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No concentration data", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # Best allocator per top_k
    best_per_k = conc_df.sort("sharpe", descending=True).group_by("top_k").first().sort("top_k")

    top_k = best_per_k["top_k"].to_numpy()
    sharpe = best_per_k["sharpe"].to_numpy()
    max_dd = best_per_k["max_drawdown"].to_numpy()
    allocators = best_per_k["allocator"].to_list()

    # Sharpe curve
    ax.plot(top_k, sharpe, "o-", color="#2196F3", linewidth=2, markersize=8, label="Sharpe")

    # Annotate best allocator at each point
    for k, s, a in zip(top_k, sharpe, allocators, strict=False):
        ax.annotate(
            a.replace("_", " "),
            (k, s),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=7,
            ha="center",
            alpha=0.7,
        )

    # Mark optimal
    best_idx = np.argmax(sharpe)
    ax.scatter(
        [top_k[best_idx]],
        [sharpe[best_idx]],
        s=150,
        c="#FF9800",
        zorder=5,
        edgecolors="black",
        linewidths=1,
        label=f"Optimal k={top_k[best_idx]}",
    )

    # Secondary axis for max drawdown
    ax2 = ax.twinx()
    ax2.plot(top_k, max_dd, "s--", color="#F44336", alpha=0.6, markersize=6, label="Max DD")
    ax2.set_ylabel("Max Drawdown", color="#F44336")
    ax2.tick_params(axis="y", labelcolor="#F44336")

    ax.set_xlabel("Top K (Portfolio Concentration)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Concentration Analysis: Sharpe vs Portfolio Size")
    ax.legend(loc="upper left", frameon=False)
    ax2.legend(loc="upper right", frameon=False)

    return fig


# ---------------------------------------------------------------------------
# Figure 4: Cost Decay Curve
# ---------------------------------------------------------------------------


def plot_cost_decay(
    explorer,
    *,
    protocol_cost_bps: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Net Sharpe vs total cost with breakeven annotation.

    Parameters
    ----------
    explorer : BacktestExplorer
    protocol_cost_bps : float, optional
        The assumed cost from setup.yaml.
    ax : plt.Axes, optional

    Returns
    -------
    plt.Figure
    """
    costs_df = explorer.cost_sensitivity()
    if costs_df.is_empty():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No cost sensitivity data", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # Best Sharpe per cost level
    best_per_cost = (
        costs_df.sort("sharpe", descending=True).group_by("cost_bps").first().sort("cost_bps")
    )

    cost_bps = best_per_cost["cost_bps"].to_numpy()
    sharpe = best_per_cost["sharpe"].to_numpy()

    ax.plot(cost_bps, sharpe, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.fill_between(cost_bps, sharpe, alpha=0.1, color="#2196F3")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")

    # Estimate breakeven via interpolation
    if sharpe[0] > 0 and sharpe[-1] < 0:
        from scipy.interpolate import interp1d

        f = interp1d(sharpe, cost_bps)
        breakeven = float(f(0))
        ax.axvline(
            breakeven,
            color="#F44336",
            linestyle="--",
            alpha=0.7,
            label=f"Breakeven: {breakeven:.0f} bps",
        )
    elif sharpe[-1] >= 0:
        breakeven = float(cost_bps[-1])
        ax.annotate(
            f"Still positive at {breakeven:.0f} bps",
            xy=(breakeven, sharpe[-1]),
            fontsize=9,
            color="#4CAF50",
        )
    else:
        breakeven = None

    # Protocol cost annotation
    if protocol_cost_bps is not None:
        ax.axvline(
            protocol_cost_bps,
            color="#4CAF50",
            linestyle=":",
            alpha=0.7,
            label=f"Protocol: {protocol_cost_bps:.0f} bps",
        )

        if breakeven is not None and protocol_cost_bps > 0:
            headroom = breakeven / protocol_cost_bps
            ax.annotate(
                f"Headroom: {headroom:.1f}×",
                xy=(protocol_cost_bps, sharpe[0] * 0.9),
                fontsize=10,
                fontweight="bold",
                color="#4CAF50",
            )

    ax.set_xlabel("Total Cost (bps per leg)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Cost Sensitivity: Strategy Viability Under Friction")
    ax.legend(loc="upper right", frameon=False)

    return fig


# ---------------------------------------------------------------------------
# Figure 5: 2-Panel Equity / Drawdown
# ---------------------------------------------------------------------------


def plot_equity_drawdown(
    daily_returns_path: Path,
    *,
    comparison_path: Path | None = None,
    labels: tuple[str, str] = ("Strategy", "Comparison"),
    ax: tuple[plt.Axes, plt.Axes] | None = None,
) -> plt.Figure:
    """2-panel figure: cumulative return (top) + drawdown (bottom).

    Parameters
    ----------
    daily_returns_path : Path
        Parquet file with ``timestamp`` and ``daily_return`` columns.
    comparison_path : Path, optional
        Second return series for overlay (e.g. pre-cost vs post-cost).
    labels : tuple[str, str]
        Labels for primary and comparison series.
    ax : tuple[plt.Axes, plt.Axes], optional

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
    else:
        ax_eq, ax_dd = ax
        fig = ax_eq.figure

    def _load_and_compute(path: Path):
        df = pl.read_parquet(path).sort("timestamp")
        dates = df["timestamp"].to_numpy()
        rets = df["daily_return"].to_numpy()
        cum = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        return dates, cum, dd

    dates, cum, dd = _load_and_compute(daily_returns_path)

    ax_eq.plot(dates, cum, color="#2196F3", linewidth=1.5, label=labels[0])
    ax_dd.fill_between(dates, dd, 0, color="#F44336", alpha=0.3)
    ax_dd.plot(dates, dd, color="#F44336", linewidth=0.8, label=labels[0])

    if comparison_path is not None and comparison_path.exists():
        dates2, cum2, dd2 = _load_and_compute(comparison_path)
        ax_eq.plot(dates2, cum2, color="#FF9800", linewidth=1.2, alpha=0.7, label=labels[1])
        ax_dd.plot(dates2, dd2, color="#FF9800", linewidth=0.8, alpha=0.7, label=labels[1])

    # Annotate worst drawdown
    worst_idx = np.argmin(dd)
    ax_dd.annotate(
        f"Max DD: {dd[worst_idx]:.1%}",
        xy=(dates[worst_idx], dd[worst_idx]),
        textcoords="offset points",
        xytext=(20, -10),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#F44336"),
        color="#F44336",
    )

    ax_eq.set_ylabel("Cumulative Return")
    ax_eq.set_title("Equity Curve and Drawdown Profile")
    ax_eq.legend(loc="upper left", frameon=False)

    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Holdout Comparison (Paired Dumbbell)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Assessment writer / reader
# ---------------------------------------------------------------------------


def write_strategy_assessment(case_study: str, assessment: dict) -> Path:
    """Write strategy_assessment.json to case study results directory.

    Parameters
    ----------
    case_study : str
        Case study ID.
    assessment : dict
        Assessment dictionary with first-pass pipeline outcome.

    Returns
    -------
    Path
        Path to written file.
    """
    from utils.paths import get_case_study_dir

    results_dir = get_case_study_dir(case_study) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    assessment["generated_at"] = datetime.now(tz=UTC).isoformat()

    path = results_dir / "strategy_assessment.json"
    path.write_text(json.dumps(assessment, indent=2, default=str))
    return path


def load_strategy_assessment(
    case_study: str,
    *,
    verify_against_registry: bool = True,
) -> dict[str, Any]:
    """Load strategy_assessment.json for a case study.

    The assessment JSON is a cached aggregate; the registry is the SSoT
    (registry only, never JSONs). When
    ``verify_against_registry`` is True (default), this function checks that
    the assessment's ``champion`` still exists as a training run in the
    registry and emits a stale-data warning if not. The function returns the
    JSON either way; callers must decide how to treat a stale assessment.

    Parameters
    ----------
    case_study : str
    verify_against_registry : bool, default True
        When True, log a warning if the assessment's champion config no
        longer exists in ``training_runs`` (typical cause: training sweep
        rerun produced new hashes, assessment JSON not refreshed).

    Returns
    -------
    dict
        Assessment dictionary, or empty dict if not found.
    """
    import sqlite3
    import warnings

    from utils.paths import get_case_study_dir

    path = get_case_study_dir(case_study) / "results" / "strategy_assessment.json"
    if not path.exists():
        return {}
    assessment = json.loads(path.read_text())

    if verify_against_registry and assessment:
        champion_source = assessment.get("champion", {}).get("source", "")
        primary_label = assessment.get("primary_label", "")
        if champion_source and "/" in champion_source and primary_label:
            family, config_name = champion_source.split("/", 1)
            db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
            if db_path.exists():
                con = sqlite3.connect(str(db_path))
                n = con.execute(
                    "SELECT COUNT(*) FROM training_runs "
                    "WHERE family = ? AND config_name = ? AND label = ?",
                    (family, config_name, primary_label),
                ).fetchone()[0]
                con.close()
                if n == 0:
                    warnings.warn(
                        f"strategy_assessment.json for '{case_study}' is STALE: "
                        f"champion {champion_source}/{primary_label} is not in the "
                        f"registry. Regenerate by running "
                        f"case_studies/{case_study}/*_strategy_analysis.py.",
                        stacklevel=2,
                    )
    return assessment


def load_all_assessments(
    case_studies: list[str] | None = None,
) -> dict[str, dict]:
    """Load strategy assessments for all case studies.

    Convenience for Ch20 aggregation.

    Returns
    -------
    dict[str, dict]
        Keyed by case study ID.
    """
    if case_studies is None:
        case_studies = [
            "etfs",
            "crypto_perps_funding",
            "nasdaq100_microstructure",
            "sp500_equity_option_analytics",
            "us_firm_characteristics",
            "fx_pairs",
            "cme_futures",
            "sp500_options",
            "us_equities_panel",
        ]

    results = {}
    for cs in case_studies:
        v = load_strategy_assessment(cs)
        if v:
            results[cs] = v
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_cost_bps(setup: dict) -> float:
    """Per-leg cost in bps from a case-study setup.yaml.

    Precedence:
    1. ``costs.per_leg_cost_bps_range`` - average of the declared range.
    2. ``costs.fee_schedule`` + ``costs.cost_tiers`` - tier-weighted average
       of taker/maker fees (tiered structures e.g. crypto).
    3. ``costs.fee_schedule`` with only taker_bps/maker_bps - simple average.
    4. Fallback ``10.0`` - explicit last resort.

    setup.yaml is authoritative. The fallback (10.0) is hit only when the
    case study does not declare any cost structure; flag such a case study
    as a setup.yaml gap rather than silently assuming 10 bps.

    Note on crypto (precedence 3 today): the `cost_tiers` block that
    formerly produced a tier-weighted ~3.47 bps was removed in commit
    `2b3bff1a` (setup.yaml reader-cleanup pass) - the majors/alts
    breakdown lives in the inline YAML comment now, not as machine-
    readable data. The simple (taker+maker)/2 = 3.0 bps headline is
    intentional under the post-cleanup config; if a future revision
    wants to recover the tier-weighted average it must reintroduce
    `cost_tiers` to setup.yaml.
    """
    costs = setup.get("costs", {}) or {}
    cost_range = costs.get("per_leg_cost_bps_range")
    if cost_range:
        return sum(cost_range) / len(cost_range)

    fee_schedule = costs.get("fee_schedule", {}) or {}
    cost_tiers = costs.get("cost_tiers", {}) or {}
    if cost_tiers:
        weighted_sum = 0.0
        total_symbols = 0
        for tier in cost_tiers.values():
            tier_fee = tier.get("fee_bps")
            tier_symbols = tier.get("symbols") or []
            if tier_fee is None or not tier_symbols:
                continue
            weighted_sum += tier_fee * len(tier_symbols)
            total_symbols += len(tier_symbols)
        if total_symbols:
            return weighted_sum / total_symbols

    taker = fee_schedule.get("taker_bps")
    maker = fee_schedule.get("maker_bps")
    if taker is not None and maker is not None:
        return (taker + maker) / 2
    if taker is not None:
        return taker
    if maker is not None:
        return maker

    return 10.0


def compute_search_risk_table(explorer) -> pl.DataFrame:
    """Build search-risk summary table for display.

    Parameters
    ----------
    explorer : BacktestExplorer

    Returns
    -------
    pl.DataFrame
        Single-column table for display.
    """
    ctx = explorer.search_context("signal")
    if not ctx:
        return pl.DataFrame()

    dsr = explorer.deflated_sharpe(stage="signal", top_n=1)
    dsr_pval = None
    dsr_sig = None
    if not dsr.is_empty() and "dsr_pvalue" in dsr.columns:
        row = dsr.row(0, named=True)
        dsr_pval = row.get("dsr_pvalue")
        dsr_sig = row.get("significant")

    rows = [
        {"metric": "Total signal backtests", "value": f"{ctx['total']:,}"},
        {"metric": "Champion Sharpe", "value": f"{ctx['champion_sharpe']:.3f}"},
        {"metric": "Median Sharpe", "value": f"{ctx['median_sharpe']:.3f}"},
        {"metric": "90th percentile Sharpe", "value": f"{ctx['p90_sharpe']:.3f}"},
        {"metric": "Champion percentile", "value": f"{ctx['champion_percentile']:.1f}%"},
        {"metric": "% positive Sharpe", "value": f"{ctx['pct_positive']:.1f}%"},
    ]
    if dsr_pval is not None:
        rows.append({"metric": "DSR p-value", "value": f"{dsr_pval:.4f}"})
        rows.append({"metric": "DSR significant", "value": "Yes" if dsr_sig else "No"})

    return pl.DataFrame(rows)


def compute_operating_profile(lineage: dict, setup: dict) -> pl.DataFrame:
    """Build operating profile table for deployment memo.

    Parameters
    ----------
    lineage : dict
        From ``champion_lineage()``.
    setup : dict
        Loaded setup.yaml.

    Returns
    -------
    pl.DataFrame
    """
    # Extract from lineage and setup
    cadence = setup.get("evaluation_protocol", {}).get("rebalance_frequency", "monthly")
    top_k = None
    allocator = None
    worst_dd = None

    if "allocation" in lineage:
        top_k = lineage["allocation"].get("top_k")
        allocator = lineage["allocation"].get("allocator")

    # Find worst drawdown across all stages
    for stage_data in lineage.values():
        dd = stage_data.get("max_drawdown")
        if dd is not None and (worst_dd is None or dd < worst_dd):
            worst_dd = dd

    cost_model = setup.get("cost_model", {})
    cost_bps = cost_model.get("per_leg_cost_bps", None)

    rows = [
        {"property": "Trading cadence", "value": cadence},
        {"property": "Portfolio concentration (top_k)", "value": str(top_k) if top_k else "-"},
        {"property": "Allocator", "value": (allocator or "-").replace("_", " ")},
        {"property": "Cost assumption", "value": f"{cost_bps} bps/leg" if cost_bps else "-"},
        {"property": "Worst drawdown", "value": f"{worst_dd:.1%}" if worst_dd else "-"},
    ]

    return pl.DataFrame(rows)


def classify_holdout_degradation(
    val_sharpe: float | None,
    hold_sharpe: float | None,
) -> str:
    """Classify holdout degradation type.

    Returns one of: proportional, signal_lost, sign_flip,
    degenerate, evidence_gap.
    """
    if val_sharpe is None or hold_sharpe is None:
        return "evidence_gap"
    if hold_sharpe < -0.1:
        return "sign_flip"
    if abs(hold_sharpe) < 0.05:
        return "signal_lost"
    if val_sharpe > 0 and hold_sharpe > 0:
        ratio = hold_sharpe / val_sharpe
        if ratio > 0.5:
            return "proportional"
        return "signal_lost"
    return "degenerate"


def build_all_synthesis(
    case_studies: list[str],
    explorers: dict,
    configs: dict[str, dict],
    ic_df: pl.DataFrame,
    bt_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    assessments: dict[str, dict],
    display_names: dict[str, str],
    asset_class_map: dict[str, str],
    freq_map: dict[str, str],
    pin_cost_risk_to_spine: frozenset[str] = frozenset(),
    allow_missing_spine: bool = False,
) -> dict[str, dict]:
    """Build per-case-study synthesis dict for all_synthesis.json.

    Queries registry and setup.yaml for each case study. Returns a dict
    keyed by case_study_id with meta, pipeline_summary, strategy_assessment,
    selection_flow, and variant_analysis.

    ``pin_cost_risk_to_spine`` lists case studies whose cost_sensitivity and
    risk_overlay numbers must be scoped to the spine (carrier) prediction
    rather than pooled across the whole registry. nasdaq belongs here: its
    cost-feasible ensemble carrier carries the headline cost/risk numbers,
    while the full-universe sweep rows are the Ch18/Ch19 cost-defeat
    demonstration and must not leak into the cross-case comparison.

    ``allow_missing_spine`` is a test-only relaxation: when True, a pinned
    case study whose spine cannot be resolved (its carrier is registered
    out-of-band and absent from an isolated test registry) is reported with
    cost/risk marked not-applicable instead of raising. Production callers
    leave this False so a genuinely missing carrier still fails loudly.
    """
    import contextlib

    from utils.paths import get_case_study_dir

    synthesis_dict = {}

    for cs in case_studies:
        explorer = explorers.get(cs)
        if explorer is None:
            continue
        setup = configs.get(cs, {})
        case_dir = get_case_study_dir(cs)
        display = display_names.get(cs, cs)

        # --- meta ---
        universe = setup.get("universe", {})
        n_assets = universe.get("n_assets", 0) or len(universe.get("symbols", []))
        cost_bps = compute_cost_bps(setup)
        labels_cfg = setup.get("labels", {})

        # The date range is the primary label's, and reported as empty where that file is
        # absent. It was previously read from `glob("*.parquet")[0]`, an arbitrary pick from an
        # unsorted directory: any file dropped in there decided the answer, and a variant label
        # with a different horizon reported a range the case study's own results do not span.
        date_start, date_end = "", ""
        primary_label = labels_cfg.get("primary", "")
        if primary_label:
            for labels_subdir in ["labels", "data/labels"]:
                label_file = case_dir / labels_subdir / f"{primary_label}.parquet"
                if not label_file.exists():
                    continue
                try:
                    lf = pl.scan_parquet(label_file)
                    cols = lf.collect_schema().names()
                    ts_col = (
                        "timestamp" if "timestamp" in cols else "date" if "date" in cols else None
                    )
                    if ts_col:
                        ts_df = lf.select(ts_col).collect()
                        if not ts_df.is_empty():
                            date_start = str(ts_df[ts_col].min())[:10]
                            date_end = str(ts_df[ts_col].max())[:10]
                except Exception:
                    pass
                if date_start:
                    break

        meta = {
            "case_study_id": cs,
            "asset_class": asset_class_map.get(cs, "unknown"),
            "frequency": freq_map.get(cs, "daily"),
            "universe_size": n_assets,
            "date_range": [date_start, date_end],
            "primary_label": labels_cfg.get("primary", ""),
            "cadence": setup.get("decision", {}).get("cadence", ""),
            "cost_bps": cost_bps,
            "calendar": setup.get("decision", {}).get("calendar", ""),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # --- models: best IC per family ---
        models_dict = {}
        cs_ic = ic_df.filter(pl.col("case_study") == display)
        if not cs_ic.is_empty():
            for row in cs_ic.iter_rows(named=True):
                models_dict[row["family"]] = {
                    "best_model": row.get("source", row["family"]),
                    "ic_mean": round(row["ic_best"], 4) if row["ic_best"] is not None else None,
                    "ic_mean_daily": (
                        round(row["ic_best_daily"], 4)
                        if row.get("ic_best_daily") is not None
                        else None
                    ),
                    "ic_std": None,
                    "n_folds": row.get("n_predictions", 0),
                }

        # --- backtest: equal-weight baseline champion ---
        cs_bt = bt_df.filter(pl.col("case_study") == display)
        backtest_dict: dict[str, Any] = {}
        if not cs_bt.is_empty():
            r = cs_bt.row(0, named=True)
            backtest_dict = {
                "selection_stage": "signal",
                "best_source": r.get("best_source", ""),
                "spine_prediction_hash": r.get("spine_prediction_hash"),
                "ml_sharpe": round(r["signal_sharpe"], 4)
                if r["signal_sharpe"] is not None
                else None,
                "ew_sharpe": None,
                "ml_beats_ew": None,
                "max_dd": None,
                "total_return": None,
                "positive_sharpe": r["signal_sharpe"] is not None and r["signal_sharpe"] > 0,
            }

            # Add holdout fields
            cs_ho = (
                holdout_df.filter(pl.col("cs_id") == cs)
                if not holdout_df.is_empty()
                else pl.DataFrame()
            )
            if not cs_ho.is_empty():
                ho = cs_ho.row(0, named=True)
                backtest_dict.update(
                    {
                        "holdout_available": True,
                        "holdout_best_source": f"{ho.get('family', '')}/{ho.get('config', '')}",
                        "holdout_ml_sharpe": round(ho["holdout_sharpe"], 4)
                        if ho["holdout_sharpe"] is not None
                        else None,
                        "holdout_positive_sharpe": ho["holdout_sharpe"] is not None
                        and ho["holdout_sharpe"] > 0,
                    }
                )
            else:
                backtest_dict.update(
                    {
                        "holdout_available": False,
                        "holdout_best_source": None,
                        "holdout_ml_sharpe": None,
                        "holdout_positive_sharpe": None,
                    }
                )

        # --- allocation ---
        # Restrict to the spine rank-1 prediction_hash when bt_df carries
        # it. Without that pin the allocator MAX-per-method aggregation
        # pools across every prediction in the registry, so Figure 20.7
        # can read off a different prediction than Ch20 prose Tables 20.5–20.7.
        cs_bt_row = bt_df.filter(pl.col("case_study") == display)
        spine_pred = None
        if not cs_bt_row.is_empty() and "spine_prediction_hash" in cs_bt_row.columns:
            spine_pred = cs_bt_row["spine_prediction_hash"][0]
        # Allocation stage ONLY. Figure 20.14 / Table 20.6 isolate the allocator
        # layer with the signal held fixed; a risk overlay (ch19) is a downstream
        # layer covered in §20.7, and folding its Sharpe in here would credit the
        # allocator with work the overlay did (and double-count it against §20.7).
        # This matches the "allocation-stage Sharpe" caption and the spine-pinned
        # allocation-only computation in 05_portfolio_allocation.
        alloc_comp = explorer.compare_allocators(
            prediction_hash=spine_pred,
            stages=("allocation",),
        )
        alloc_dict: dict[str, Any] = {
            "best_allocator": "",
            "best_sharpe": None,
            "allocator_comparison": {},
        }
        if not alloc_comp.is_empty():
            # compare_allocators sorts by avg_sharpe; the heatmap and prose report
            # the allocator with the highest best_sharpe, so re-rank explicitly.
            _top = alloc_comp.sort("best_sharpe", descending=True).head(1)
            alloc_dict["best_allocator"] = _top["allocator"][0]
            alloc_dict["best_sharpe"] = round(float(_top["best_sharpe"][0]), 4)
            for row in alloc_comp.iter_rows(named=True):
                alloc_dict["allocator_comparison"][row["allocator"]] = round(
                    float(row["best_sharpe"]), 4
                )

        # --- costs ---
        # A pinned CS MUST carry cost/risk on its spine prediction; falling back
        # to None here would pool full-universe rows - the exact cost-defeat-demo
        # leak the pin prevents. Fail loudly rather than leak silently.
        skip_cost_risk = False
        if cs in pin_cost_risk_to_spine and spine_pred is None:
            if not allow_missing_spine:
                raise ValueError(
                    f"{cs!r} is pinned to the spine prediction for cost/risk, but no "
                    f"spine_prediction_hash resolved (empty backtest row or missing "
                    f"column); refusing to silently pool full-universe cost/risk rows."
                )
            # Test-mode escape hatch: the pinned carrier is registered out-of-band
            # (e.g. nasdaq's cost-feasible ensemble), so an isolated test registry
            # has no carrier rows to resolve a spine from. Mark cost/risk
            # not-applicable rather than pooling full-universe rows - the same leak
            # the hard raise prevents in production (where allow_missing_spine=False).
            skip_cost_risk = True
        cost_risk_pred = spine_pred if cs in pin_cost_risk_to_spine else None
        cost_df = (
            pl.DataFrame()
            if skip_cost_risk
            else explorer.cost_sensitivity(prediction_hash=cost_risk_pred)
        )
        costs_dict: dict[str, Any] = {
            "actual_bps": cost_bps,
            "breakeven_bps": None,
            "survives_costs": None,
            "gross_sharpe_at_zero": None,
            "net_sharpe_at_actual": None,
            "capacity_usd_10pct": None,
        }
        if not cost_df.is_empty():
            # Zero-cost envelope: best achievable Sharpe before any cost is
            # charged, the gross side of the cost waterfall paired with
            # ``net_sharpe_at_actual`` (same cost sweep, same scoping). Both are
            # max-over-config envelopes, so gross >= net by construction (a
            # higher cost can only lower each config's Sharpe).
            zero_rows = cost_df.filter(pl.col("cost_bps") == 0)
            if not zero_rows.is_empty():
                costs_dict["gross_sharpe_at_zero"] = round(float(zero_rows["sharpe"].max()), 4)

            available = sorted(cost_df["cost_bps"].unique().to_list())
            match_bps = None
            for lvl in available:
                if lvl >= cost_bps:
                    match_bps = lvl
                    break
            if match_bps is None and available:
                match_bps = available[-1]

            if match_bps is not None:
                matched = cost_df.filter(pl.col("cost_bps") == match_bps)
                if not matched.is_empty():
                    net_sr = float(matched["sharpe"].max())
                    costs_dict["net_sharpe_at_actual"] = round(net_sr, 4)
                    costs_dict["survives_costs"] = net_sr > 0

            best_per_cost = (
                cost_df.group_by("cost_bps").agg(sharpe=pl.col("sharpe").max()).sort("cost_bps")
            )
            for row in best_per_cost.iter_rows(named=True):
                if row["sharpe"] is not None and row["sharpe"] <= 0:
                    costs_dict["breakeven_bps"] = row["cost_bps"]
                    break
            else:
                if not best_per_cost.is_empty():
                    costs_dict["breakeven_bps"] = float(best_per_cost["cost_bps"].max()) + 10

        # --- risk ---
        risk_df = (
            pl.DataFrame()
            if skip_cost_risk
            else explorer.risk_impact(prediction_hash=cost_risk_pred)
        )
        risk_dict: dict[str, Any] = {
            "best_overlay": "none",
            "baseline_sharpe": 0,
            "baseline_max_dd": 0,
            "managed_sharpe": None,
            "managed_max_dd": None,
            "overlay_sharpe_delta": None,
            "worst_drawdown_pct": 0,
            "var_95": 0,
            "cvar_95": 0,
            "overlay_count": 0,
        }
        if not risk_df.is_empty():
            if "baseline_sharpe" in risk_df.columns:
                bs = risk_df["baseline_sharpe"].drop_nulls()
                if len(bs) > 0:
                    risk_dict["baseline_sharpe"] = round(float(bs[0]), 4)

            best_risk = risk_df.sort("sharpe", descending=True).head(1)
            risk_dict["best_overlay"] = best_risk["risk_name"][0]
            risk_dict["managed_sharpe"] = round(float(best_risk["sharpe"][0]), 4)
            risk_dict["managed_max_dd"] = round(float(best_risk["max_drawdown"][0] or 0), 4)
            risk_dict["overlay_sharpe_delta"] = round(
                risk_dict["managed_sharpe"] - risk_dict["baseline_sharpe"], 4
            )
            risk_dict["overlay_count"] = len(risk_df)

        # --- labels (from setup.yaml) ---
        labels_dict = {
            "primary": labels_cfg.get("primary", ""),
            "variants": labels_cfg.get("variants", []),
            "n_obs": 0,
            "mean": 0,
            "std": 0,
            "hit_rate": 0,
        }

        # --- features (count from features directory) ---
        n_financial = 0
        n_temporal = 0
        for feat_subdir in ["features", "data/features"]:
            feat_dir = case_dir / feat_subdir
            if feat_dir.exists():
                fin_path = feat_dir / "financial.parquet"
                if fin_path.exists():
                    with contextlib.suppress(Exception):
                        n_financial = len(pl.read_parquet_schema(fin_path)) - 2
                temp_path = feat_dir / "model_based.parquet"
                if temp_path.exists():
                    with contextlib.suppress(Exception):
                        n_temporal = len(pl.read_parquet_schema(temp_path)) - 2
                if n_financial > 0:
                    break

        features_dict = {
            "financial": n_financial,
            "temporal": n_temporal,
            "total": n_financial + n_temporal,
            "passed_eval": n_financial + n_temporal,
            "top_3_by_ic": [],
        }

        # --- selection_flow ---
        best_model_source = backtest_dict.get("best_source", "")
        selection_flow = {
            "validation_selected_label": labels_cfg.get("primary", ""),
            "selection_origin": None,
            "selected_model_id": best_model_source.split("/")[-1] if best_model_source else "",
        }

        # --- strategy assessment ---
        cs_assessment = assessments.get(cs, {})

        # --- assemble ---
        synthesis_dict[cs] = {
            "meta": meta,
            "pipeline_summary": {
                "labels": labels_dict,
                "features": features_dict,
                "models": models_dict,
                "backtest": backtest_dict,
                "allocation": alloc_dict,
                "costs": costs_dict,
                "risk": risk_dict,
            },
            "strategy_assessment": cs_assessment if cs_assessment else None,
            "selection_flow": selection_flow,
            "variant_analysis": {},
            "signal_sweep": {},
            "next_steps": [],
            "key_findings": [],
        }

    return synthesis_dict


def _family_color_map() -> dict[str, str]:
    """Consistent color map for model families."""
    return {
        "linear": "#4CAF50",
        "gbm": "#FF9800",
        "tabular_dl": "#2196F3",
        "deep_learning": "#9C27B0",
        "latent_factors": "#E91E63",
        "causal_dml": "#795548",
    }
