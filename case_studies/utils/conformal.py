"""Per-prediction conformal widths for position sizing.

Validation widths follow the ``walk_forward_v3`` contract, which is one rule for
every fold: the width used at timestamp ``t`` for an entity is calibrated on every
residual for that entity at or before ``t - h``, where ``h`` is the **sizing calibration
lag** in prediction data steps - :func:`sizing_conformal_lag`, never
:data:`HOLDOUT_CONFORMAL_EMBARGO_STEPS` directly. Residuals from earlier folds and from the
current fold's own elapsed history are both eligible. An entity with fewer than
``min_calibration_n`` eligible residuals receives a pooled width from every entity's
eligible residuals, so allocation never changes the selected basket by silently dropping an
uncalibrated entity.

``h`` is load-bearing in both directions. Writing the label's own horizon as ``k``, a
residual at ``t'`` realizes over ``(t', t'+k]``, so without a lag one with ``t' + k > t``
carries information from after the decision it sizes - which the prior-fold-only rule this
replaces did at every fold boundary, not just at the holdout. With a lag, the earliest fold
no longer has to abstain: it calibrates on its own elapsed history after a warm-up rather
than sitting out entirely, which on a two-fold split is the difference between forfeiting
half the evaluation period and forfeiting a warm-up. ``h`` is ``max(1, k)``: the ``k`` term
keeps an unrealized residual out, and the floor keeps out the residual of the decision
itself, which no horizon makes available.

**The lag is not the holdout embargo, and the two tables answer different questions.**
:data:`HOLDOUT_CONFORMAL_EMBARGO_STEPS` records how far a residual reaches forward, which is
what decides whether a calibration residual leaks across the validation/holdout boundary; it
is zero for a zero-horizon label. The sizing lag asks what was known when the position was
chosen and is therefore at least one step whatever the horizon. They coincide for every
label with a horizon of a step or more, which is why one table served both until a
zero-horizon label was declared. :func:`sizing_conformal_lag` states the argument.

**No coverage guarantee is claimed or consumed.** Split conformal's finite-sample coverage
requires the calibration and test scores to be exchangeable, and return residuals are
heteroskedastic and regime-dependent, so they are not. What the allocator does with these
widths is inverse-uncertainty sizing with a conformal quantile standing in for a
volatility estimate; nothing downstream reads an interval or a coverage level.

Storage: alongside ``predictions.parquet`` in the same prediction-hash directory.
Writes are alpha-aware: a new alpha is appended to any existing
``conformal_widths.parquet`` (rows for the same alpha are replaced), so the
single artifact can carry multiple alphas. The output always uses ``symbol``
as the entity column (matching ``backtest_loaders.load_predictions_for_backtest``'s
normalization), regardless of whether the source predictions.parquet uses
``product`` or ``stock_id``.
"""

from __future__ import annotations

import copy
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from utils.modeling import conformal_quantile
from utils.paths import get_case_study_dir

ID_COLS: tuple[str, ...] = ("symbol", "product")

# Legacy → canonical column rename map. Older prediction parquets (pre-IC
# unification) use {fold, prediction, actual}; newer use {fold_id, y_score, y_true}.
_LEGACY_RENAME: dict[str, str] = {
    "fold": "fold_id",
    "prediction": "y_score",
    "actual": "y_true",
}

DEFAULT_ALPHA: float = 0.20
DEFAULT_MIN_CALIBRATION_N: int = 30
# `walk_forward_v3` has required a lag of at least one step since it was introduced in
# `ee67b3b8`, so no artifact carrying this version was ever calibrated on the residual of the
# decision it sizes and none has to be regenerated. Naming the lag (`sizing_conformal_lag`)
# changed which number a caller passing `label` gets for a zero-horizon label, not what the
# stored contract guarantees. A version bump belongs to a change in the calibration rule
# itself, and it is expensive: it invalidates every stored width and moves the identity of
# every backtest that consumes one, across nine case studies.
CALIBRATION_VERSION: str = "walk_forward_v3"
POOLED_FALLBACK: str = "pooled_prior_oos"

HOLDOUT_CONFORMAL_EMBARGO_STEPS: dict[str, int] = {
    "etfs/fwd_ret_21d": 21,
    "etfs/fwd_ret_5d": 5,
    "cme_futures/fwd_ret_5d": 5,
    "cme_futures/fwd_ret_21d": 21,
    "fx_pairs/fwd_ret_1d": 1,
    "fx_pairs/fwd_ret_5d": 5,
    "fx_pairs/fwd_ret_21d": 21,
    "crypto_perps_funding/fwd_ret_24h": 3,
    "crypto_perps_funding/fwd_ret_8h": 1,
    "crypto_perps_funding/fwd_dir_8h": 1,
    "crypto_perps_funding/fwd_dir_8h_3c": 1,
    # Counted in minutes, because that is this panel's step. The three entries used to be
    # the label horizon divided by a 15-minute decision cadence (1, 4, 1). The prediction
    # panel does not sit on that cadence: measured over the registered prediction sets,
    # the modal gap between adjacent prediction timestamps is 0:01:00 for all four labels
    # (fwd_ret_15m 78,829 of 79,082 gaps; fwd_ret_5m 81,359 of 81,612; fwd_ret_60m 67,444
    # of 67,697; the remainder are overnight and weekend). `compute_conformal_widths`
    # documents `h` as a lag "in prediction data steps" and builds its index from that
    # artifact, so a value of 1 embargoed one MINUTE where the label reaches 16.
    #
    # Each value is `labels.buffer` for the label, which this case study sets to the
    # horizon PLUS ONE BAR: the entry leg is the VWAP of the bar after the decision, so a
    # label at t consumes a quote at t+H+1. Taking the horizon alone would leave the last
    # bar of every calibration residual unresolved at the moment the width is used.
    "nasdaq100_microstructure/fwd_ret_15m": 16,
    "nasdaq100_microstructure/fwd_ret_60m": 61,
    "nasdaq100_microstructure/fwd_ret_5m": 6,
    # Declared in labels.variants with variant_buffers.fwd_dir_15m = 16min, and carried
    # 18 registered training runs while absent from this table entirely.
    "nasdaq100_microstructure/fwd_dir_15m": 16,
    "sp500_equity_option_analytics/fwd_ret_5d": 5,
    "sp500_equity_option_analytics/fwd_ret_risk_adj_5d": 5,
    # The three below joined the table when this case study declared the conformal_weighted
    # allocator, which sizes every declared label rather than the primary alone. Each value is
    # the label's own horizon from labels.horizons in sessions, the panel's step: a ten-session
    # return needs ten sessions of embargo for the calibration residuals to stop overlapping the
    # holdout, and the direction labels reach exactly as far as the returns they are signs of.
    "sp500_equity_option_analytics/fwd_ret_10d": 10,
    "sp500_equity_option_analytics/fwd_dir_5d": 5,
    "sp500_equity_option_analytics/fwd_dir_10d": 10,
    # ret_to_expiry has no fixed horizon: each row is a 30-day ATM straddle held to its
    # own expiry, so the residual at session t resolves at that contract's expiry rather
    # than a constant number of steps later. The bound is labels.buffer = 35D, the widest
    # DTE the selection admits (measured max dte_calendar = 35), and five calendar weeks
    # hold exactly 25 weekdays, which NYSE holidays can only reduce. The prediction panel
    # is one row per session per symbol, so 25 sessions is the tight bound and the
    # measured maximum over the label panel is 25 steps (median 21). Embargoing the
    # median would leave the longest-dated quarter of the calibration residuals still
    # resolving inside the holdout.
    "sp500_options/ret_to_expiry": 25,
    "us_equities_panel/fwd_ret_5d": 5,
    "us_equities_panel/fwd_ret_1d": 1,
    "us_equities_panel/fwd_ret_21d": 21,
    # Zero because the horizon is zero. us_firm_characteristics dates each row by the month
    # the return was earned, and `labels.horizons` in its setup.yaml declares `0D` for all
    # three, so the outcome is already realised at the observation and no residual reaches
    # into the holdout window. The entries were 1, which is not a conservative reading of
    # this table - the table records the label horizon, and a value above it discards the
    # last month of calibration for a leak the label cannot have.
    "us_firm_characteristics/fwd_ret_1m_win": 0,
    "us_firm_characteristics/fwd_ret_1m": 0,
    "us_firm_characteristics/fwd_class_1m": 0,
}


def walk_forward_conformal_coverage(
    predictions: pl.DataFrame,
    *,
    embargo_steps: int,
    levels: tuple[float, ...] = (0.80, 0.90, 0.95),
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
) -> list[dict[str, float | int]]:
    """Realised coverage of the widths that size positions, at each nominal level.

    The estimator is :func:`walk_forward_widths`, the one `conformal_weighted` allocates on:
    per entity with a pooled fallback, calibrated on everything known at ``t - embargo_steps``,
    quantile taken at ``interpolation="higher"``. A decision is covered when its absolute
    residual is inside the half-width the allocator would have sized it with.

    What this replaced measured a different estimator - one pooled quantile over every entity,
    fixed on the earliest fold, no embargo, exact order statistic - and printed it as the
    strategy's coverage. Pooling was the largest of those differences: a pooled quantile over
    contracts with different volatilities is not the quantity that sizes any single position,
    and `compute_conformal_weights` normalizes ``1/width`` within each side at each timestamp,
    so only the cross-sectional dispersion of the per-entity widths reaches the portfolio -
    the axis pooling removes.

    ``embargo_steps`` is the calibration lag in prediction data steps, and it is
    :func:`sizing_conformal_lag`'s answer rather than the holdout table's - the two ask
    different questions and disagree on a zero-horizon label.

    **This is a diagnostic of residual dispersion, not a guarantee.** Split conformal's
    finite-sample coverage requires the calibration and test scores to be exchangeable, and
    return residuals are heteroskedastic and regime-dependent. Nothing in the allocation path
    reads an interval or a coverage level; the width stands in for a volatility estimate.

    Returns one row per level with ``nominal_level``, ``empirical_coverage``,
    ``mean_interval_width_frac_std``, ``n_test`` and ``n_uncalibrated`` - the decisions that
    cleared no warm-up, which are the ones the allocator sizes from a pooled width or not at
    all, and which no coverage figure describes.
    """
    if embargo_steps < 1:
        raise ValueError(
            f"embargo_steps={embargo_steps} would measure a width calibrated on the residual "
            "of the decision it sizes; see sizing_conformal_lag"
        )

    renames = {
        legacy: canonical
        for legacy, canonical in _LEGACY_RENAME.items()
        if legacy in predictions.columns and canonical not in predictions.columns
    }
    if renames:
        predictions = predictions.rename(renames)

    id_col = _detect_id_col(predictions.columns)
    if id_col != "symbol":
        predictions = predictions.rename({id_col: "symbol"})
    id_col = "symbol"

    required = {"timestamp", id_col, "fold_id", "y_true", "y_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction artifact missing {sorted(missing)}")
    predictions = predictions.drop_nulls(required)
    if predictions.is_empty():
        raise ValueError("prediction artifact has no finite predictions")
    for column in ("y_true", "y_score"):
        if not np.isfinite(predictions[column].to_numpy()).all():
            raise ValueError(f"prediction artifact has non-finite {column} values")

    measured = predictions.with_columns(
        abs_resid=(pl.col("y_true") - pl.col("y_score")).abs()
    ).select("timestamp", id_col, "y_true", "abs_resid")

    rows: list[dict[str, float | int]] = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError(f"invalid conformal level: {level}")
        widths = walk_forward_widths(
            predictions,
            id_col=id_col,
            alpha=1.0 - level,
            min_calibration_n=min_calibration_n,
            embargo_steps=embargo_steps,
            context=f"conformal coverage at level {level}",
        )
        covered = measured.join(
            widths.select("timestamp", id_col, "width"), on=["timestamp", id_col], how="left"
        )
        sized = covered.drop_nulls("width")
        if sized.is_empty():
            raise ValueError(f"no decision at level {level} carries a calibrated width")
        # The scale is the spread of the outcomes the widths were measured against, so a family
        # trading a different return magnitude stays comparable. Taken over the sized rows
        # rather than over a calibration window, because the widths are not calibrated on one.
        scale = float(sized["y_true"].std() or 0.0)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("conformal coverage target scale is invalid")
        rows.append(
            {
                "nominal_level": float(level),
                "empirical_coverage": float(
                    (sized["abs_resid"] <= sized["width"] / 2.0).mean() or 0.0
                ),
                "mean_interval_width_frac_std": float(sized["width"].mean()) / scale,
                "n_test": int(sized.height),
                "n_uncalibrated": int(covered.height - sized.height),
            }
        )
    return rows


def holdout_conformal_embargo_steps(case_study: str, label: str) -> int:
    """Return the reviewed label horizon in prediction data steps."""
    key = f"{case_study}/{label}"
    try:
        return HOLDOUT_CONFORMAL_EMBARGO_STEPS[key]
    except KeyError as error:
        raise KeyError(
            f"No conformal holdout embargo is defined for {key}; add a reviewed data-step value"
        ) from error


def sizing_conformal_lag(case_study: str, label: str) -> int:
    """The calibration lag a width used to size a position must carry, in data steps.

    Not the same question :data:`HOLDOUT_CONFORMAL_EMBARGO_STEPS` answers, and reading one off
    the other is what put a decision's own residual into its width.

    The table records how far a residual reaches **forward**: a residual at ``t'`` resolves over
    ``(t', t'+h]``, so at the validation/holdout boundary a residual with ``t' + h > t`` carries
    information from after ``t``. For a zero-horizon label nothing reaches forward and the
    reviewed entry is 0 - `us_firm_characteristics` dates each row by the month the return was
    earned, and ``labels.horizons`` declares ``0D`` for all three of its labels.

    A sizing lag asks what was **known** when the position was chosen, which is a different
    thing. The position that earns month ``t``'s return was selected at the end of ``t-1``, and
    the backtest applies its weights to ``y_true`` at ``t``; the residual at ``t`` is therefore
    not available to size it, whatever the horizon. So the lag is at least one step even where
    the holdout embargo is zero, and equals the horizon everywhere else - the two coincide for
    every label with a horizon of one step or more, which is why one table served both until a
    zero-horizon label was declared.
    """
    return max(1, holdout_conformal_embargo_steps(case_study, label))


def ensure_conformal_calibration_identity(
    strategy_spec: dict[str, Any],
    *,
    holdout_embargo_steps: int | None = None,
) -> dict[str, Any]:
    """Return a spec whose conformal allocation carries its full identity.

    ``holdout_embargo_steps`` belongs here rather than only in the widths artifact, and
    only for a holdout run. The widths are an input to the backtest and the embargo decides
    them, so a spec that omits it gives two different calibrations one identity: change the
    embargo, re-run, and the hash does not move. The registry then refuses to overwrite the
    registered run - which is how the state announces itself, and the announcement is a
    conflict rather than a number, so nothing is silently wrong. But the correct behaviour
    is a different hash for a different calibration, which is what recording it gives.

    It is recorded under ``input_identity``, which is where the specification already
    records the digests of things a backtest consumed rather than declared - the price
    frame, the funding rates. The widths are one of those, and the embargo is what decides
    them. It is deliberately not in the allocation block: that block is what a holdout
    replay is matched to its validation carrier by, and the two run the same strategy, so
    putting it there made every holdout spec differ from its own carrier and
    ``select_holdout_self_backtest`` stopped matching. It is not a ``backtest_config``
    section either - that schema is closed, and adding one is rejected by name.

    Pass it only on the holdout path. The embargo applies at the validation-to-holdout
    boundary and has no meaning within validation, so adding it to a validation spec would
    change every registered conformal hash to record something that did not affect it.
    """
    spec = copy.deepcopy(strategy_spec)
    strategy = spec.get("strategy")
    if not isinstance(strategy, dict):
        return spec
    allocation = strategy.get("allocation")
    if not isinstance(allocation, dict) or allocation.get("method") != "conformal_weighted":
        return spec

    requested = allocation.get("calibration_version")
    if requested not in (None, CALIBRATION_VERSION):
        raise ValueError(
            f"Unsupported conformal calibration_version={requested!r}; "
            f"expected {CALIBRATION_VERSION!r}"
        )
    allocation["calibration_version"] = CALIBRATION_VERSION
    allocation.setdefault("min_calibration_n", DEFAULT_MIN_CALIBRATION_N)
    allocation.setdefault("sparse_fallback", POOLED_FALLBACK)
    if holdout_embargo_steps is not None:
        identity = spec.setdefault("input_identity", {})
        identity["conformal_holdout_embargo_steps"] = int(holdout_embargo_steps)
    return spec


def _detect_id_col(columns: list[str]) -> str:
    for c in ID_COLS:
        if c in columns:
            return c
    raise ValueError(
        f"predictions.parquet has no canonical entity column "
        f"(expected one of {ID_COLS}); found {columns}"
    )


def _predictions_dir(
    case_study: str,
    prediction_hash: str,
    *,
    case_dir: Path | None = None,
) -> Path:
    resolved_case_dir = case_dir or get_case_study_dir(case_study)
    return resolved_case_dir / "run_log" / "predictions" / prediction_hash


def _write_widths(
    path: Path,
    new_widths: pl.DataFrame,
    alpha: float,
    *,
    immutable: bool = False,
) -> None:
    """Persist widths to ``path``, merging by alpha.

    If ``path`` already exists, rows with the same ``alpha`` are dropped and
    replaced by ``new_widths``; rows with other alphas are preserved. This
    keeps a single file able to carry multiple alphas, which matches what
    ``load_conformal_widths`` expects when filtering on ``alpha``.
    """
    merged = new_widths
    if path.exists():
        # Tolerate a partially-written file from a concurrent worker — the
        # parallel sweep can race two workers onto the same prediction_hash
        # when both auto-generate widths via load_conformal_widths(). Treat an
        # unreadable existing file as "no prior widths" and overwrite.
        try:
            existing = pl.read_parquet(path)
        except (pl.exceptions.ComputeError, pl.exceptions.NoDataError, OSError, EOFError):
            if immutable:
                raise ValueError(f"locked conformal artifact is unreadable: {path}") from None
            # A zero-byte or missing-magic-bytes file from a half-finished
            # concurrent write surfaces as NoDataError/OSError, not just
            # ComputeError — treat any unreadable file as "no prior widths".
            existing = None
        if existing is not None:
            # A superseded artifact is REPLACED, not refused. Both refusals below used
            # to be unconditional, and their message told the caller to snapshot the file
            # and delete it by hand. That made a corrected calibration unusable until a
            # person intervened in every lane holding a stale artifact: on 2026-08-30 it
            # took us_firm_characteristics' whole conformal stage down - all 52 backtests,
            # 11 of 11 cost levels, including the canonical rank-1 - and the recovery was
            # eleven manual file moves. Superseding IS the intended outcome of a
            # calibration fix, so the writer performs it.
            #
            # `immutable` keeps the old behaviour and must: a locked artifact is pinned by
            # digest, and silently rewriting one would break the pin it exists to hold.
            legacy = "calibration_version" not in existing.columns
            versions = set() if legacy else set(existing["calibration_version"].unique().to_list())
            superseded = legacy or (versions and versions != {CALIBRATION_VERSION})
            if superseded and immutable:
                raise ValueError(
                    f"locked conformal artifact at {path} holds a superseded calibration "
                    f"{sorted(versions) or 'with no version column'}; it cannot be rewritten "
                    "in place because the lock pins it by digest"
                )
            if superseded:
                existing = None
        if existing is not None:
            # Float equality on alpha is fine here: we write Float64 and read
            # back Float64; both sides round-trip bit-identically through parquet.
            same_alpha = existing.filter(pl.col("alpha") == alpha)
            if immutable and not same_alpha.is_empty():
                from case_studies.utils.artifact_digest import value_digest

                if value_digest(same_alpha) != value_digest(new_widths):
                    raise ValueError(f"locked conformal artifact conflicts with {path}")
                return
            keep = existing.filter(pl.col("alpha") != alpha)
            merged = pl.concat([keep, new_widths], how="diagonal_relaxed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        merged.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _expanding_calibration(frame: pl.DataFrame, *, alpha: float) -> pl.DataFrame:
    """Per grid step: the quantile and count of every residual up to and including it.

    One row per distinct ``step`` in ``frame``, carrying what a caller standing at that
    step already knows. The embargo is applied by the as-of join that reads this table,
    not here, so the same construction serves the per-entity and the pooled pool.
    """
    ordered = frame.sort("step")
    window = max(ordered.height, 1)
    return (
        ordered.with_columns(
            cal_q=pl.col("abs_resid").rolling_quantile(
                quantile=1.0 - alpha,
                window_size=window,
                min_samples=1,
                interpolation="higher",
            ),
            cal_n=pl.int_range(1, pl.len() + 1, dtype=pl.UInt32),
        )
        .group_by("step", maintain_order=True)
        .agg(cal_q=pl.col("cal_q").last(), cal_n=pl.col("cal_n").last())
        .with_columns(pl.col("step").cast(pl.Int64))
    )


def walk_forward_widths(
    preds: pl.DataFrame,
    *,
    id_col: str,
    alpha: float,
    min_calibration_n: int,
    embargo_steps: int,
    context: str,
) -> pl.DataFrame:
    """The width at every (timestamp, entity) the ``walk_forward_v3`` rule can calibrate.

    ``preds`` carries canonical column names and ``id_col`` names its entity column; everything
    that reads a prediction artifact from disk, checks it and renames it belongs to the caller.
    Splitting it out is what lets the coverage a notebook prints be measured on these widths
    rather than on a second estimator built to different rules.

    Returns ``[timestamp, <id_col>, fold_id, width, alpha, calibration_n, calibration_scope,
    calibration_version]``, one row per decision that clears the warm-up.
    """
    preds = preds.filter(
        pl.col("y_true").is_not_null() & pl.col("y_score").is_not_null()
    ).with_columns(abs_resid=(pl.col("y_true") - pl.col("y_score")).abs())
    if preds.is_empty():
        raise ValueError(f"{context}: no residuals to calibrate on")

    # The embargo is expressed in data steps, so it is applied on the position of a
    # timestamp in the prediction set's own grid rather than on the calendar. A gap in
    # the grid is a gap in the data, and stepping over it is what "h steps back" means.
    grid = preds.select("timestamp").unique().sort("timestamp").with_row_index("step")
    preds = preds.join(grid, on="timestamp", how="inner")

    # The entity literal carries the source column's dtype rather than whatever polars infers
    # from the Python value. `join_asof` compares its `by` columns by dtype and raises on a
    # mismatch, and a panel keyed on integer identifiers - permnos on us_firm_characteristics -
    # infers Int32 from an unannotated literal against a UInt32 column.
    entity_dtype = preds.schema[id_col]
    per_entity = pl.concat(
        [
            _expanding_calibration(group, alpha=alpha).with_columns(
                pl.lit(entity, dtype=entity_dtype).alias(id_col)
            )
            for (entity,), group in preds.group_by([id_col], maintain_order=True)
        ]
    )
    pooled = _expanding_calibration(preds, alpha=alpha)

    targets = (
        preds.select("timestamp", id_col, "fold_id", "step")
        .unique()
        .with_columns(known_by=pl.col("step").cast(pl.Int64) - embargo_steps)
        # Sorted within entity as well as globally: join_asof cannot verify sortedness once
        # `by` groups are given, and an unsorted group silently resolves to the wrong row.
        .sort(id_col, "known_by")
    )
    resolved = (
        targets.join_asof(
            per_entity.sort("step").rename(
                {"cal_q": "entity_q", "cal_n": "entity_n", "step": "entity_step"}
            ),
            left_on="known_by",
            right_on="entity_step",
            by=id_col,
            strategy="backward",
        )
        .sort("known_by")
        .join_asof(
            pooled.sort("step").rename(
                {"cal_q": "pooled_q", "cal_n": "pooled_n", "step": "pooled_step"}
            ),
            left_on="known_by",
            right_on="pooled_step",
            strategy="backward",
        )
    )

    enough_own = pl.col("entity_n") >= min_calibration_n
    enough_pooled = pl.col("pooled_n") >= min_calibration_n
    widths = (
        resolved.with_columns(
            width=pl.when(enough_own)
            .then(2.0 * pl.col("entity_q"))
            .when(enough_pooled)
            .then(2.0 * pl.col("pooled_q"))
            .otherwise(None),
            calibration_n=pl.when(enough_own)
            .then(pl.col("entity_n"))
            .otherwise(pl.col("pooled_n")),
            calibration_scope=pl.when(enough_own)
            .then(pl.lit("symbol"))
            .otherwise(pl.lit("pooled")),
        )
        .drop_nulls("width")
        .with_columns(
            alpha=pl.lit(alpha, dtype=pl.Float64),
            calibration_version=pl.lit(CALIBRATION_VERSION),
        )
        .select(
            "timestamp",
            id_col,
            "fold_id",
            "width",
            "alpha",
            "calibration_n",
            "calibration_scope",
            "calibration_version",
        )
        .sort("timestamp", id_col)
    )

    if widths.is_empty():
        raise ValueError(
            f"{context}: no decision clears a warm-up of "
            f"min_calibration_n={min_calibration_n} residuals plus an embargo of "
            f"{embargo_steps} steps"
        )
    return widths


def compute_conformal_widths(
    case_study: str,
    prediction_hash: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
    write: bool = True,
    case_dir: Path | None = None,
    label: str | None = None,
    embargo_steps: int | None = None,
) -> pl.DataFrame:
    """Compute and optionally persist expanding walk-forward conformal widths.

    Returns one row per (timestamp, entity) for which a width could be
    calibrated: columns ``[timestamp, <id_col>, fold_id, width, alpha,
    calibration_n, calibration_scope, calibration_version]``.

    Calibration rule, one rule for every fold: the width used at timestamp
    ``t`` for entity ``s`` is ``2·q_{1-α}`` of ``|y_true − y_score|`` over
    every residual for ``s`` at timestamps at or before ``t − h``, where ``h``
    is the sizing calibration lag in prediction data steps. Residuals from earlier
    folds and from the current fold's own elapsed history are both eligible,
    which is what lets the earliest fold trade after a warm-up instead of
    sitting out entirely.

    ``h`` is the sizing calibration lag. It is the same number
    :func:`compute_holdout_conformal_widths` applies at the validation/holdout boundary for
    every label whose horizon is a step or more - a residual at ``t'`` depends on the return
    realized over ``(t', t'+h]``, so one with ``t' + h > t`` carries information from after
    the decision - and it is one step larger for a zero-horizon label, where nothing reaches
    forward but the residual at ``t`` is still not known when the position for ``t`` is
    chosen. Pass ``embargo_steps`` directly or pass ``label`` to take
    :func:`sizing_conformal_lag`'s answer.

    Entities with fewer than ``min_calibration_n`` eligible residuals of their
    own use a pooled quantile over every entity's eligible residuals, so
    allocation never changes the selected basket by silently dropping an
    uncalibrated entity.

    Writes are alpha-aware (see module docstring): an existing
    ``conformal_widths.parquet`` for the same prediction hash retains rows
    at other alphas; rows at this ``alpha`` are replaced.

    Raises ``ValueError`` when neither ``embargo_steps`` nor a ``label`` with a
    reviewed embargo is given, when ``embargo_steps`` is below one, or when no
    (timestamp, entity) pair clears the warm-up.

    One step is the floor and it is not the same quantity
    :data:`HOLDOUT_CONFORMAL_EMBARGO_STEPS` records - see :func:`sizing_conformal_lag`. Pass
    that function's answer rather than the table's where the two can differ.
    """
    pred_dir = _predictions_dir(case_study, prediction_hash, case_dir=case_dir)
    pred_path = pred_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.parquet not found: {pred_path}")

    if embargo_steps is None:
        if label is None:
            raise ValueError(
                f"{case_study}/{prediction_hash}: conformal calibration needs the label "
                "horizon as an embargo - pass embargo_steps, or label to look up the "
                "reviewed value"
            )
        # The sizing lag, not the holdout table: they differ on a zero-horizon label and this
        # is the estimator that sizes a position.
        embargo_steps = sizing_conformal_lag(case_study, label)
    if embargo_steps < 1:
        raise ValueError(
            f"{case_study}/{prediction_hash}: embargo_steps={embargo_steps} would calibrate "
            "on a residual that is not yet realized at the decision it sizes"
        )

    preds = pl.read_parquet(pred_path)
    legacy_present = {k: v for k, v in _LEGACY_RENAME.items() if k in preds.columns}
    if legacy_present:
        preds = preds.rename(legacy_present)
    src_id_col = _detect_id_col(preds.columns)
    # Canonical: emit widths keyed by "symbol", matching backtest_loaders normalization.
    if src_id_col != "symbol":
        preds = preds.rename({src_id_col: "symbol"})
    id_col = "symbol"

    required = {"timestamp", id_col, "y_true", "y_score", "fold_id"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"{case_study}/{prediction_hash}: predictions.parquet missing "
            f"columns {sorted(missing)}; got {preds.columns}"
        )

    widths = walk_forward_widths(
        preds,
        id_col=id_col,
        alpha=alpha,
        min_calibration_n=min_calibration_n,
        embargo_steps=embargo_steps,
        context=f"{case_study}/{prediction_hash}",
    )

    if write:
        # `pred_dir` honours the `case_dir` override the predictions were read
        # from; recomputing it without that override would write the widths into
        # an unrelated run log.
        _write_widths(pred_dir / "conformal_widths.parquet", widths, alpha)

    return widths


def compute_holdout_conformal_widths(
    case_study: str,
    val_prediction_hash: str,
    holdout_prediction_hash: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
    embargo_steps: int = 0,
    write: bool = True,
    immutable: bool = False,
) -> pl.DataFrame:
    """Pooled per-symbol split-conformal widths for the holdout window.

    Calibration set: all validation residuals for the val prediction set,
    pooled across folds within each symbol. Prediction set: every
    (timestamp, symbol) pair in the holdout predictions parquet.

    Per-symbol pooled q_{1-α}(|y_true - y_score|) is broadcast across the
    holdout window for that symbol. Symbols with fewer than
    ``min_calibration_n`` validation residuals receive the pooled width from
    all embargoed validation residuals.

    ``embargo_steps`` drops the trailing ``embargo_steps`` distinct val
    timestamps from the calibration set. Required when the label has a
    non-zero forward-return horizon ``h``: a residual at val timestamp ``t``
    depends on returns realized over ``(t, t+h]``; if ``t+h`` falls inside
    the holdout window, the residual leaks holdout-period price information
    into the calibration. Set this to the label's horizon expressed in
    data-step units — e.g. ``21`` for ``fwd_ret_21d`` on a daily trading
    calendar; ``3`` for ``fwd_ret_24h`` on 8-hourly crypto data; ``1`` for
    ``fwd_ret_15m`` on 15-minute bars.

    Output schema matches ``compute_conformal_widths``'s val output:
    ``[timestamp, symbol, fold_id, width, alpha, calibration_n]`` with
    ``fold_id = -1`` as a sentinel meaning "holdout, no fold partition", plus
    two provenance columns this function alone can supply.

    ``calibration_source`` records ``val_prediction_hash`` and
    ``calibration_embargo_steps`` records ``embargo_steps``. Both are arguments
    here and neither was written, so the artifact could not say which validation
    prediction calibrated it or under what embargo - and ``fold_id = -1`` plus a
    current ``calibration_version`` are true of *any* validation-calibrated
    widths file. Widths taken from a different model, or from the right model
    under a different embargo, satisfied every marker the holdout guard in
    ``case_studies/research/strategy.py`` could check. Stamping the two makes the
    question answerable from the file rather than from the call that wrote it.
    """
    val_dir = _predictions_dir(case_study, val_prediction_hash)
    val_path = val_dir / "predictions.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"val predictions.parquet not found: {val_path}")

    val_preds = pl.read_parquet(val_path)
    legacy_val = {k: v for k, v in _LEGACY_RENAME.items() if k in val_preds.columns}
    if legacy_val:
        val_preds = val_preds.rename(legacy_val)
    src_id_val = _detect_id_col(val_preds.columns)
    if src_id_val != "symbol":
        val_preds = val_preds.rename({src_id_val: "symbol"})

    required = {"timestamp", "symbol", "y_true", "y_score"}
    missing = required - set(val_preds.columns)
    if missing:
        raise ValueError(
            f"{case_study}/{val_prediction_hash}: val predictions.parquet missing "
            f"columns {sorted(missing)}; got {val_preds.columns}"
        )

    val_preds = val_preds.filter(
        pl.col("y_true").is_not_null() & pl.col("y_score").is_not_null()
    ).with_columns(abs_resid=(pl.col("y_true") - pl.col("y_score")).abs())

    if embargo_steps > 0:
        unique_ts = sorted(val_preds.select("timestamp").unique().to_series().to_list())
        if len(unique_ts) <= embargo_steps:
            raise ValueError(
                f"{case_study}/{val_prediction_hash}: embargo_steps={embargo_steps} "
                f">= n_val_timestamps={len(unique_ts)}; no calibration data left "
                f"after embargo"
            )
        cutoff_ts = unique_ts[-embargo_steps - 1]
        val_preds = val_preds.filter(pl.col("timestamp") <= cutoff_ts)

    per_symbol_widths = (
        val_preds.group_by("symbol")
        .agg(
            q=pl.col("abs_resid").quantile(1.0 - alpha, interpolation="higher"),
            calibration_n=pl.len(),
        )
        .filter(pl.col("calibration_n") >= min_calibration_n)
        .with_columns(
            width=2.0 * pl.col("q"),
            alpha=pl.lit(alpha, dtype=pl.Float64),
            calibration_scope=pl.lit("symbol"),
            calibration_version=pl.lit(CALIBRATION_VERSION),
        )
        .select(
            "symbol",
            "width",
            "alpha",
            "calibration_n",
            "calibration_scope",
            "calibration_version",
        )
    )

    if val_preds.height < min_calibration_n:
        raise ValueError(
            f"{case_study}/{val_prediction_hash}: only {val_preds.height} embargoed "
            f"validation residuals; need at least {min_calibration_n}"
        )

    ho_dir = _predictions_dir(case_study, holdout_prediction_hash)
    ho_path = ho_dir / "predictions.parquet"
    if not ho_path.exists():
        raise FileNotFoundError(f"holdout predictions.parquet not found: {ho_path}")

    ho_preds = pl.read_parquet(ho_path)
    legacy_ho = {k: v for k, v in _LEGACY_RENAME.items() if k in ho_preds.columns}
    if legacy_ho:
        ho_preds = ho_preds.rename(legacy_ho)
    src_id_ho = _detect_id_col(ho_preds.columns)
    if src_id_ho != "symbol":
        ho_preds = ho_preds.rename({src_id_ho: "symbol"})

    ho_required = {"timestamp", "symbol"}
    ho_missing = ho_required - set(ho_preds.columns)
    if ho_missing:
        raise ValueError(
            f"{case_study}/{holdout_prediction_hash}: holdout predictions.parquet "
            f"missing columns {sorted(ho_missing)}; got {ho_preds.columns}"
        )

    ho_keys = ho_preds.select("timestamp", "symbol").unique()
    missing_symbols = (
        ho_keys.select("symbol")
        .unique()
        .join(per_symbol_widths.select("symbol"), on="symbol", how="anti")
    )
    if not missing_symbols.is_empty():
        pooled_q = val_preds["abs_resid"].quantile(1.0 - alpha, interpolation="higher")
        if pooled_q is None:
            raise ValueError(
                f"{case_study}/{val_prediction_hash}: holdout pooled quantile is undefined"
            )
        pooled = missing_symbols.with_columns(
            width=pl.lit(2.0 * float(pooled_q), dtype=pl.Float64),
            alpha=pl.lit(alpha, dtype=pl.Float64),
            calibration_n=pl.lit(val_preds.height, dtype=pl.UInt32),
            calibration_scope=pl.lit("pooled"),
            calibration_version=pl.lit(CALIBRATION_VERSION),
        ).select(per_symbol_widths.columns)
        per_symbol_widths = pl.concat([per_symbol_widths, pooled], how="vertical_relaxed")

    widths = (
        ho_keys.join(per_symbol_widths, on="symbol", how="inner")
        .with_columns(
            fold_id=pl.lit(-1, dtype=pl.Int64),
            calibration_source=pl.lit(val_prediction_hash, dtype=pl.Utf8),
            calibration_embargo_steps=pl.lit(int(embargo_steps), dtype=pl.Int64),
        )
        .select(
            "timestamp",
            "symbol",
            "fold_id",
            "width",
            "alpha",
            "calibration_n",
            "calibration_scope",
            "calibration_version",
            "calibration_source",
            "calibration_embargo_steps",
        )
        .sort("timestamp", "symbol")
    )

    if widths.is_empty():
        raise ValueError(
            f"{case_study}/{holdout_prediction_hash}: pooled-width join with "
            f"holdout predictions produced no rows. Holdout symbol set may not "
            f"overlap with val-calibrated symbols."
        )

    if write:
        out = ho_dir / "conformal_widths.parquet"
        _write_widths(out, widths, alpha, immutable=immutable)

    return widths


def load_conformal_widths(
    case_study: str,
    prediction_hash: str,
    *,
    alpha: float | None = None,
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
    calibration_version: str = CALIBRATION_VERSION,
    label: str | None = None,
    embargo_steps: int | None = None,
) -> pl.DataFrame:
    """Load persisted widths. Filters to a specific alpha if supplied.

    Auto-generates ``conformal_widths.parquet`` via ``compute_conformal_widths``
    when missing so the conformal_weighted allocator works end-to-end inside the
    canonical sweep without a separate widths-bootstrap step. Only the default
    alpha is computed on auto-generation; callers asking for a non-default alpha
    on a fresh prediction set should compute widths up-front.
    """
    path = _predictions_dir(case_study, prediction_hash) / "conformal_widths.parquet"

    def _generate() -> None:
        compute_conformal_widths(
            case_study,
            prediction_hash,
            min_calibration_n=min_calibration_n,
            label=label,
            embargo_steps=embargo_steps,
        )

    if not path.exists():
        _generate()
    df = pl.read_parquet(path)

    # An artifact holding only a superseded calibration is REGENERATED, not refused.
    # Auto-generation used to be conditional on the file being absent, so a lane that had
    # computed widths before a calibration fix could never move past it: the read raised
    # "No widths for calibration_version=..." and the write refused to mix versions, and
    # the only way through was to move the file aside by hand. That is the loop this has
    # been round several times. Regenerating is what the caller wanted in every one of
    # them, and it is safe because the widths are derived from the prediction set, which
    # has not changed - only the rule for calibrating against it has.
    #
    # Only when the CURRENT version was asked for. A caller naming an older version is
    # asking a question about history and gets the honest empty answer.
    stale = (
        "calibration_version" not in df.columns
        or df.filter(pl.col("calibration_version") == calibration_version).is_empty()
    )
    if stale and calibration_version == CALIBRATION_VERSION:
        _generate()
        df = pl.read_parquet(path)

    if "calibration_version" not in df.columns:
        raise ValueError(
            f"Legacy conformal artifact at {path}; preserve and regenerate it before use"
        )
    df = df.filter(pl.col("calibration_version") == calibration_version)
    if df.is_empty():
        raise ValueError(f"No widths for calibration_version={calibration_version!r} in {path}")
    if alpha is not None:
        available = sorted(set(df["alpha"].to_list()))
        df = df.filter(pl.col("alpha") == alpha)
        if df.is_empty():
            raise ValueError(f"No widths at alpha={alpha} in {path}; available alphas: {available}")
    return df


def coverage_summary(case_study: str, prediction_hash: str, *, alpha: float | None = None) -> dict:
    """Per-fold coverage and width-dispersion diagnostics (no side effects)."""
    pred_dir = _predictions_dir(case_study, prediction_hash)
    preds = pl.read_parquet(pred_dir / "predictions.parquet")
    src_id_col = _detect_id_col(preds.columns)
    # Widths file always uses canonical "symbol" (see compute_conformal_widths).
    widths = load_conformal_widths(case_study, prediction_hash, alpha=alpha)
    id_col = "symbol"

    n_total = preds[src_id_col].n_unique()
    folds = sorted(widths["fold_id"].unique().to_list())
    by_fold = []
    for k in folds:
        wk = widths.filter(pl.col("fold_id") == k).select(id_col, "width").unique()
        n_with = wk.height
        w_min = float(wk["width"].min()) if n_with else float("nan")
        w_max = float(wk["width"].max()) if n_with else float("nan")
        by_fold.append(
            {
                "fold_id": k,
                "n_with_width": n_with,
                "n_total": n_total,
                "frac_covered": n_with / n_total if n_total else 0.0,
                "mean_width": float(wk["width"].mean()) if n_with else float("nan"),
                "median_width": float(wk["width"].median()) if n_with else float("nan"),
                "width_p10": float(wk["width"].quantile(0.10)) if n_with else float("nan"),
                "width_p90": float(wk["width"].quantile(0.90)) if n_with else float("nan"),
                "max_min_ratio": (w_max / max(w_min, 1e-12)) if n_with else float("nan"),
            }
        )
    return {
        "case_study": case_study,
        "prediction_hash": prediction_hash,
        "id_col": id_col,
        "n_entities": n_total,
        "n_folds_with_widths": len(folds),
        "by_fold": by_fold,
    }
