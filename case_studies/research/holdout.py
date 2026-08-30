from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import time as dt_time
from typing import Any

import pandas as pd
import polars as pl

_FOLD_FIELDS = ("fold", "train_start", "train_end", "val_start", "val_end")


def _validation_folds(validation_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    computation = validation_spec.get("computation")
    if not isinstance(computation, dict):
        raise ValueError("holdout CV derivation requires a current resolved training specification")
    cv = computation.get("cv")
    if not isinstance(cv, dict):
        raise ValueError("selected training specification has no resolved CV interval")
    folds = cv.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("selected training specification has no resolved validation folds")
    missing = {name for fold in folds for name in _FOLD_FIELDS if name not in fold}
    if missing:
        raise ValueError(f"validation folds are missing boundaries: {sorted(missing)}")
    return [dict(fold) for fold in folds]


def widest_label_buffer(case_study: str, setup: Mapping[str, Any]) -> tuple[str, str]:
    """Return the widest buffer any of a case study's labels declares, and whose it is.

    The holdout fold is one fold. A fold-scoped temporal artifact carries a single set of
    boundaries per fold id, and a fold-fitted feature's ``train_end`` is what that feature
    knows, so the fold's geometry has to be the same whichever label a model is later fitted
    on. That leaves one question: which buffer.

    It is the widest, and the narrow ones are unsafe rather than merely tighter. A fold built
    on a one-day buffer and handed to a twenty-one-day model gives that model training rows
    whose features were fitted on data twenty sessions past its own ``train_end`` - the leak
    the buffer exists to prevent, arriving through the feature instead of the label. The
    widest buffer costs the shorter-horizon models a longer gap than they need, which is the
    conservative direction and, on the case studies measured, twenty sessions out of thousands.

    Labels are read from ``labels.primary`` and ``labels.variants`` and resolved through
    :func:`utils.artifact_specs.resolve_label_buffer`, so a label carrying its own spec
    artifact still wins over the setup block.
    """
    from utils.artifact_specs import resolve_label_buffer
    from utils.cv_splits import normalize_label_buffer

    labels = setup.get("labels") or {}
    names = [str(labels["primary"])] if labels.get("primary") else []
    names += [str(name) for name in (labels.get("variants") or [])]
    if not names:
        raise ValueError(f"{case_study} declares no labels, so no holdout buffer can be derived")

    widest: tuple[pd.Timedelta, str, str] | None = None
    for name in names:
        buffer = resolve_label_buffer(case_study, name, setup)
        if not buffer:
            continue
        span = pd.Timedelta(normalize_label_buffer(buffer))
        if widest is None or span > widest[0]:
            widest = (span, str(buffer), name)
    if widest is None:
        raise ValueError(
            f"{case_study} declares labels {names} and a buffer for none of them, so the gap "
            "sealing holdout training from the holdout window cannot be derived"
        )
    return widest[1], widest[2]


def _boundary_iso(moment: pd.Timestamp) -> str:
    """Render a fold boundary the way the panel it describes carries its dates.

    A midnight boundary is a date, and writing it as `2023-11-29T00:00:00` says the panel
    has a time of day that it does not. Every daily panel stores its dates as `Date`, and
    Polars reads a full ISO datetime into `Date` as null rather than truncating it, so the
    datetime rendering could not be read back by the consumer it was written for. The time
    is kept when there is one - `crypto_perps_funding` and `nasdaq100_microstructure` are
    intraday and their boundaries are not midnight.
    """
    if moment.time() == dt_time(0, 0):
        return moment.date().isoformat()
    return moment.isoformat()


def _on_panel_clock(moment: pd.Timestamp, zone: Any) -> pd.Timestamp:
    """Read a boundary on the clock the panel keeps its own observations on.

    ``evaluation.holdout_start`` is a date, and a date is not a moment until something says
    which clock it is read on. The fold boundaries are moments, taken from the panel, so an
    intraday panel carries them tz-aware; pandas then refuses to compare the two rather than
    assuming a zone, and the derivation raises ``Cannot compare tz-naive and tz-aware
    timestamps`` before it computes anything. Every daily case study escaped it because its
    panel is tz-naive, which is why this surfaced first on ``crypto_perps_funding``.

    Localizing rather than converting is what keeps the declaration meaning what it says: the
    window is declared in the calendar the case study trades on, so 2024-01-01 is midnight on
    that calendar and not midnight UTC shifted into it.
    """
    if zone is None or moment.tzinfo is not None:
        return moment
    return moment.tz_localize(zone)


def build_holdout_cv(
    validation_spec: Mapping[str, Any],
    *,
    case_study: str,
    timeline: Sequence[Any],
    label: str | None = None,
    train_start_floor: Any | None = None,
) -> dict[str, Any]:
    """Derive the one holdout CV interval that retrains the selected validation configuration.

    The holdout window is not a choice. It is ``evaluation.holdout_start`` and
    ``evaluation.holdout_end`` from the case study's own ``setup.yaml``, read here through
    :func:`case_studies.utils.cv_window.canonical_window` so this derivation and the window a
    backtest is sliced to cannot disagree.
    :func:`case_studies.research.models.locked_holdout_split` checks it again at execution.

    The training interval is the whole history available before that window, which is
    ``min(train_start)`` across the validation folds and never one fold's own start: the fold
    list runs newest first, so ``folds[0]["train_start"]`` is the *latest* start in the set and
    would hand the retrain the shortest window it could have had rather than the longest.
    :func:`utils.cv_splits.earliest_train_start` is that read, and this calls it rather than
    repeating it.

    ``train_start_floor`` bounds that below, and exists because "the whole history available"
    is a claim about the FEATURES, not about the calendar. A configuration fitted on fold-scoped
    model-based features has no history before the fold that produced them: stage 04 emits each
    fold over a rolling window, so on sp500_equity_option_analytics the deriver asks for
    2017-01-05..2020-12-16 and the artifact's holdout fold begins 2019-01-02, leaving 495 of 977
    training dates covered. Fitting the other 482 on null columns is not the configuration that
    was ranked - every validation fold saw a fully populated three-year window - so the holdout
    would evaluate an estimator nobody selected. Clamping to the producer's geometry applies the
    same rule correctly rather than contradicting it: take everything available, where available
    is what the features actually span. Families with no fold-scoped features supply no floor and
    are unaffected. ml4t/agent-workspace#977 has the measurement and the rejected alternative.

    Training ends one label buffer before the holdout opens, using the same buffer the
    validation folds were built with. That gap is what stops the last training label's outcome
    window from reaching into the holdout, which would train the holdout model on the period it
    is meant to be judged against. The buffer is required rather than defaulted: a case study
    that declares none has no basis for a gap, and a zero gap here is a leak rather than a
    conservative choice.
    """
    from case_studies.utils.artifact_digest import value_digest
    from case_studies.utils.causal import embargo_from_buffer, observation_step
    from case_studies.utils.cv_window import canonical_window
    from utils.artifact_specs import (
        load_setup_config,
        resolve_label_buffer,
        resolve_label_horizon,
    )
    from utils.cv_splits import earliest_train_start, normalize_label_buffer

    resolved_label = str(label if label is not None else validation_spec.get("label") or "")
    if not resolved_label:
        raise ValueError("holdout CV derivation requires the label the selection was made on")

    window = canonical_window(case_study, resolved_label, split="holdout")
    if window is None:
        raise ValueError(
            f"{case_study} declares no holdout window for {resolved_label!r}; "
            "evaluation.holdout_start and evaluation.holdout_end must both be set in "
            "config/setup.yaml before the holdout can be derived"
        )
    holdout_start, holdout_end = window

    folds = _validation_folds(validation_spec)
    validation_cv = dict(validation_spec["computation"]["cv"])
    observations = sorted({pd.Timestamp(str(value)) for value in timeline})
    if len(observations) < 2:
        raise ValueError("holdout CV derivation needs at least two observations to measure cadence")
    # Every boundary below is compared against these observations, so they are what decides
    # the clock. Resolved once, here, rather than at each comparison: a fold set and a window
    # that reach this function on different clocks have to be reconciled in one place or the
    # reconciliation is a thing to remember at four call sites.
    panel_zone = observations[0].tz
    train_start = _on_panel_clock(earliest_train_start(folds), panel_zone)
    floor_applied = None
    if train_start_floor is not None:
        floor = _on_panel_clock(pd.Timestamp(train_start_floor), panel_zone)
        if floor > train_start:
            # Recorded, not silent: the clamp changes the training interval, so a reader of the
            # spec has to be able to see that the window is the producer's and why.
            floor_applied = _boundary_iso(floor)
            train_start = floor

    # Both resolvers fall back to setup.yaml's own labels block, and return None without it
    # for every case study whose label carries no separate spec artifact - which is all nine
    # for their primary label. Passing the setup is what makes the buffer resolvable at all.
    setup = load_setup_config(case_study)
    # The case study's widest buffer, not the selected label's own. The fold-scoped temporal
    # artifact carries one holdout fold whose features every label's holdout model is fitted
    # on, so its boundary has to be label-independent, and the widest is the only choice that
    # leaks for no label. `widest_label_buffer` carries the argument. The selected label still
    # supplies the horizon check below, which this now satisfies by construction.
    buffer, buffer_label = widest_label_buffer(case_study, setup)
    holdout_open = _on_panel_clock(pd.Timestamp(holdout_start), panel_zone)
    holdout_close = _on_panel_clock(pd.Timestamp(holdout_end), panel_zone)
    # `evaluation.holdout_end` is a DATE, and a date on an intraday panel means the whole of that
    # day. Parsed, it is that date at midnight, and every window filter downstream is
    # `timestamp <= val_end`, so the final session sorts after the boundary and is dropped from
    # the interval the holdout is evaluated over. `utils/modeling.py::_inclusive_end_of` says the
    # same thing with a nanosecond sentinel; this says it with an observation the panel actually
    # holds, which is what `train_end` already is and what makes the fold readable as a pair of
    # settlements rather than one settlement and a fencepost.
    #
    # A daily panel is untouched by construction: its last observation of that date IS midnight,
    # so the widening condition is false and the rendering does not move. That matters because
    # this value is inside the hashed fold, so moving it changes the training identity every
    # holdout refit registers under. ml4t/agent-workspace#986.
    within_close = [value for value in observations if value.date() <= holdout_close.date()]
    if within_close and within_close[-1] > holdout_close:
        holdout_close = within_close[-1]

    # Counted in OBSERVATIONS and stepped back along the panel's own dates, never subtracted as
    # calendar time. `utils/cv_splits.py` already carries this bug's epitaph: "21D" as a
    # pd.Timedelta is ~15 trading days, not 21, so a calendar subtraction leaves the last training
    # label resolving inside the holdout - short, silent, and in the direction that looks fine.
    # `generate_cv_splits` converts D-buffers to trading days for exactly this reason, and the
    # causal resolver counts observations for the same one. This is the third construction of the
    # same seal and it must agree with the other two.
    # Measured against the panel's OWN cadence, not a per-unit default. Without observed_step
    # embargo_from_buffer assumes a daily grid, which reads "1M" as 21 observations - correct on a
    # daily panel and 21x too long on us_firm_characteristics' monthly one, where a month IS one
    # observation. AGENTS.md records the mirror of this: "24H as one period on an eight-hour panel".
    cadence = observation_step(pd.DataFrame({"timestamp": observations}))
    # A month has no fixed length, so it cannot be divided by an observation step and
    # embargo_from_buffer refuses. Its other branch takes periods_per_year, which is COUNTED here
    # off the same timeline rather than assumed: falling through to the per-unit defaults is what
    # turns "1M" into 21 observations on a monthly panel where a month is one.
    span_years = (observations[-1] - observations[0]).days / 365.25
    periods_per_year = max(1, round(len(observations) / span_years)) if span_years > 0 else 1
    try:
        buffer_steps = embargo_from_buffer(buffer, observed_step=cadence)
    except ValueError:
        buffer_steps = embargo_from_buffer(buffer, periods_per_year=periods_per_year)
    if buffer_steps < 1:
        raise ValueError(f"label buffer {buffer!r} leaves no gap before the holdout window")

    # A declared zero horizon - us_firm_characteristics dates each row by the month the return was
    # earned, so "0D" - means the outcome is already realised at the observation and there is
    # nothing for the buffer to cover. embargo_from_buffer divides by the value, so it must not be
    # asked about zero rather than being asked and having its answer discarded.
    horizon = resolve_label_horizon(case_study, resolved_label, setup)
    if horizon and pd.Timedelta(normalize_label_buffer(horizon)) > pd.Timedelta(0):
        try:
            horizon_steps = embargo_from_buffer(horizon, observed_step=cadence)
        except ValueError:
            horizon_steps = embargo_from_buffer(horizon, periods_per_year=periods_per_year)
        if buffer_steps < horizon_steps:
            raise ValueError(
                f"label buffer {buffer!r} is {buffer_steps} observations, shorter than the "
                f"outcome horizon {horizon!r} at {horizon_steps}, so the last training label "
                "resolves inside the holdout window"
            )

    pre_holdout = [value for value in observations if value < holdout_open]
    if len(pre_holdout) <= buffer_steps:
        raise ValueError(
            f"{case_study} has {len(pre_holdout)} observations before the holdout opens, which "
            f"cannot absorb a {buffer_steps}-observation buffer"
        )
    # The buffer is the number of observations that must NOT be trained on, so the last retained
    # one sits one step beyond it. `pre_holdout[-buffer_steps]` is the first excluded observation.
    train_end = pre_holdout[-(buffer_steps + 1)]
    if train_end <= train_start:
        raise ValueError(
            f"{case_study} holdout training interval is empty: history starts "
            f"{train_start.date()} and the buffered boundary is {train_end.date()}"
        )

    fold = {
        "fold": max(int(entry["fold"]) for entry in folds) + 1,
        "train_start": _boundary_iso(train_start),
        "train_end": _boundary_iso(train_end),
        "val_start": _boundary_iso(holdout_open),
        "val_end": _boundary_iso(holdout_close),
    }
    identity = value_digest(pl.DataFrame([fold]))
    if identity == validation_cv.get("identity"):
        raise ValueError("derived holdout CV is identical to the selected validation CV")
    return {
        "folds": [fold],
        "identity": identity,
        "split": "holdout",
        "request": {
            "source": "case_study_holdout",
            "label_buffer": str(buffer),
            "label_buffer_label": buffer_label,
            "label_buffer_steps": buffer_steps,
            "observation_cadence": str(cadence),
            "periods_per_year": periods_per_year,
            "holdout_window": [str(holdout_start), str(holdout_end)],
            # Present only when it moved the boundary, so a spec that needed no clamp hashes
            # exactly as it did before this existed and no registered identity moves.
            **({"train_start_floor": floor_applied} if floor_applied else {}),
        },
    }


def build_holdout_training_spec(
    study: Any,
    validation_spec: Mapping[str, Any],
    *,
    timeline: Sequence[Any],
    case_study: str | None = None,
) -> dict[str, Any]:
    """Re-key one validation training specification onto the derived holdout fold.

    Three steps have to happen together and in this order, and each of them already refuses
    on its own terms: derive the holdout interval from the case study's declared window,
    bound its training start at whatever the family's features actually reach, and recompute
    the fields the resolver derived per validation fold. Doing two of the three produces a
    specification that looks complete and fits the wrong estimator - a manifest describing
    the validation folds, or a training window half of which has no features - so they are
    one call rather than three a caller assembles.

    A holdout fit is a computation and nothing more. How many times a case study runs one is
    the reader's business, not this module's: the holdout notebooks re-run like any other
    stage, and a result that turns out to be wrong is deleted and produced again rather than
    treated as spent.

    Returns a new specification; ``validation_spec`` is not modified.
    """
    holdout_spec = deepcopy(dict(validation_spec))
    holdout_spec["computation"]["cv"] = build_holdout_cv(
        validation_spec,
        case_study=str(case_study if case_study is not None else study.case_study),
        timeline=timeline,
        train_start_floor=_holdout_training_floor(study, validation_spec),
    )
    _rekey_holdout_spec(study, holdout_spec, dict(validation_spec))
    return holdout_spec


# Fields the resolver derives PER FOLD, from the data, during a run. They describe the VALIDATION
# fold set, and `validate_locked_model_run` requires them re-keyed to the HOLDOUT fold:
# `validate_locked_expected_keys` raises "no eligibility manifest" when
# `expected_prediction_keys` is absent, and "eligibility mismatch" when it describes a different
# frame. So neither carrying them forward nor dropping them is correct - both produce a training
# specification that fails at execution, one silently wrong and one loudly.
#
# `_rekey_holdout_spec` below recomputes them, dispatched per family because each family derives
# them by its own rule from its own training rows. `CVSpec` is not the vehicle: it carries
# `holdout_start`/`holdout_end`, but `resolve()` passes them to `generate_cv_splits` as boundaries
# to seal VALIDATION against, so it selects validation folds and cannot emit a holdout fold. That
# is why the fold is derived here.
_FOLD_DERIVED_FIELDS = (
    ("computation", "expected_prediction_keys"),
    ("model", "effective_params_by_fold"),
    ("macro_context", "resolved_fold_digest"),
)


def _holdout_training_floor(study: Any, validation_spec: Mapping[str, Any]) -> Any | None:
    """Ask the family how far back its features actually reach, or None if nothing bounds it.

    Dispatches through ``_family_module`` exactly as ``_rekey_holdout_spec`` does. Absence is the
    answer for most families and is not an error: a configuration whose features are defined over
    the whole panel has no floor, and returning None leaves the derivation exactly as it was.
    Only a family whose features are fold-scoped can answer, because only it knows which artifact
    holds them.
    """
    from .models import _family_module

    hook = getattr(_family_module(validation_spec.get("family")), "holdout_training_floor", None)
    return None if hook is None else hook(study, validation_spec=validation_spec)


def _rekey_holdout_spec(study: Any, spec: dict[str, Any], validation_spec: dict[str, Any]) -> None:
    """Recompute the fold-derived fields for the holdout fold, or refuse with the family named.

    The fields are family-specific - each family derives them with its own rule, from its own
    training rows - so this dispatches through ``_family_module`` exactly as
    ``reconstruct_locked_request`` and ``validate_locked_run`` already do. A family that has not
    implemented the hook still refuses, but it now refuses for itself rather than on behalf of
    every family at once.
    """
    from .models import _family_module

    family = spec.get("family")
    module = _family_module(family)
    hook = getattr(module, "rekey_holdout_spec", None)
    if hook is None:
        raise NotImplementedError(
            f"the {family!r} family cannot yet re-key a validation training spec to the holdout "
            "fold, so no holdout fit can be built for it. Implementing it means recomputing this "
            "family's fold-derived fields against the derived holdout fold - the eligibility "
            "manifest from the dataset, and any parameter the family resolves from a fold's own "
            "training rows - by the same rule that produced the recorded validation values. See "
            "`rekey_holdout_spec` in case_studies/utils/linear.py."
        )
    hook(study, spec, validation_spec=validation_spec)
    _require_holdout_keyed_fields(spec)


def _require_holdout_keyed_fields(spec: dict[str, Any]) -> None:
    """Check the re-keyed fields describe the holdout fold, not the validation folds.

    A hook that returned without recomputing, or that recomputed against the wrong split, leaves
    fields that look present and are wrong, and a training identity registers over them either
    way. So the shape is checked here rather than trusted: exactly one fold, and it is the fold
    the derived holdout CV names.
    """
    computation = spec["computation"]
    cv = computation.get("cv")
    if not isinstance(cv, dict):
        raise ValueError("holdout spec has no resolved CV")
    # Both shapes `locked_holdout_split` accepts: an explicit one-fold list, or the flat form
    # where the single fold's boundaries sit on the CV record itself.
    folds = cv.get("folds")
    if folds is None:
        fold_id = str(int(cv.get("fold", 0)))
    elif isinstance(folds, list) and len(folds) == 1:
        fold_id = str(int(folds[0]["fold"]))
    else:
        raise ValueError("holdout spec must carry exactly one resolved fold")

    manifest = computation.get("expected_prediction_keys")
    if not isinstance(manifest, dict) or manifest.get("n_folds") != 1:
        raise ValueError(f"holdout eligibility manifest was not re-keyed to one fold: {manifest!r}")

    model = computation.get("model")
    if isinstance(model, dict) and "effective_params_by_fold" in model:
        keys = set(model["effective_params_by_fold"])
        if keys != {fold_id}:
            raise ValueError(
                f"holdout parameters are keyed to {sorted(keys)}, not the holdout fold "
                f"{fold_id!r}; they still describe the validation folds"
            )

    task = computation.get("task")
    if isinstance(task, dict):
        imbalance = task.get("imbalance")
        if isinstance(imbalance, dict) and "effective_class_weights_by_fold" in imbalance:
            keys = set(imbalance["effective_class_weights_by_fold"])
            if keys != {fold_id}:
                raise ValueError(
                    f"holdout class weights are keyed to {sorted(keys)}, not the holdout fold "
                    f"{fold_id!r}"
                )
