from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry.specs import canonical_json, canonical_value, compute_hash
from utils.cv_splits import generate_cv_splits


def _normalize_boundary(value: Any) -> str:
    raw = value.isoformat() if hasattr(value, "isoformat") else str(value)
    try:
        boundary = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    if boundary.tzinfo is not None:
        boundary = boundary.astimezone(UTC).replace(tzinfo=None)
    return boundary.isoformat()


def require_fold_scoped_temporal_compatibility(
    requested_folds: list[dict[str, Any]],
    artifact_folds: list[dict[str, Any]],
) -> None:
    """Reject CV geometry that cannot reuse fold-scoped temporal features."""
    fields = ("fold", "train_start", "train_end", "val_start", "val_end")

    def normalize(split: dict[str, Any]) -> dict[str, Any]:
        return {
            field: int(split[field]) if field == "fold" else _normalize_boundary(split[field])
            for field in fields
        }

    source = {int(split["fold"]): normalize(split) for split in artifact_folds}
    incompatible = [
        normalize(split)
        for split in requested_folds
        if source.get(int(split["fold"])) != normalize(split)
    ]
    if incompatible:
        raise ValueError(
            "custom CV is incompatible with fold-scoped temporal features; "
            "use the artifact's original fold boundaries"
        )


def require_fold_scoped_temporal_holdout_coverage(
    requested_fold: dict[str, Any],
    temporal_by_fold: Any,
    *,
    source_timeline: Any,
    date_col: str = "timestamp",
    fold_col: str = "fold",
) -> None:
    """Require an existing temporal fold to cover the holdout's training and evaluation windows.

    ``require_fold_scoped_temporal_compatibility`` asks whether the artifact declares a fold with
    the requested geometry. For a holdout that question has no good answer. The holdout fold is
    derived when the lock is taken, so the artifact - built during stage 04, before any lock -
    never declares it, and the artifact cannot be rebuilt to add it: the lock pins the feature
    file by whole-file sha256, so writing the fold in changes the digest the selection was made
    under. Compatibility therefore refuses every holdout lock on a case study with fold-scoped
    model-based features.

    Coverage is the question that can be answered. The model-based features are joined by
    ``(entity, date)``, so what the holdout run actually needs is not a fold labelled for it but
    rows spanning the dates it will train and evaluate on. This checks exactly that, against the
    fold the derived holdout CV names, and it is strictly the stronger check where both apply:
    a fold with matching boundaries but missing rows passes compatibility and fails here.
    """
    import polars as pl

    from utils.modeling import validate_temporal_fold_coverage

    columns = [fold_col, date_col]
    if isinstance(temporal_by_fold, pl.LazyFrame):
        frame = temporal_by_fold.select(columns).collect()
    elif isinstance(temporal_by_fold, pl.DataFrame):
        frame = temporal_by_fold.select(columns)
    else:
        frame = pl.from_pandas(temporal_by_fold.loc[:, columns])
    fold_id = int(requested_fold["fold"])
    dates = frame.filter(pl.col(fold_col) == fold_id).get_column(date_col)
    if dates.is_empty():
        raise ValueError(f"fold-scoped temporal artifact has no holdout fold {fold_id}")
    dtype = dates.dtype

    def boundary(name: str) -> Any:
        value = requested_fold[name]
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return pl.Series([value]).cast(dtype, strict=False).item()

    train_start, train_end = boundary("train_start"), boundary("train_end")
    val_start, val_end = boundary("val_start"), boundary("val_end")
    if not dates.is_between(train_start, train_end, closed="both").any():
        raise ValueError("fold-scoped temporal holdout has no requested training rows")
    if isinstance(source_timeline, pl.Series):
        source_dates = source_timeline.cast(dtype, strict=False)
    else:
        source_dates = pl.Series(source_timeline).cast(dtype, strict=False)
    expected = source_dates.filter(source_dates.is_between(val_start, val_end, closed="both"))
    if expected.is_empty():
        raise ValueError("source data has no observations in the holdout evaluation window")

    source_frame = pl.DataFrame({date_col: source_dates})
    temporal_frame = frame.rename({fold_col: "fold"}) if fold_col != "fold" else frame
    validate_temporal_fold_coverage(
        source_frame,
        temporal_frame,
        [requested_fold],
        date_col=date_col,
    )

    # The endpoint, specifically: a temporal artifact that stops one session short of the holdout
    # window's last observation would otherwise pass every check above, and the holdout would be
    # evaluated on a shorter period than the one the lock declares.
    evaluation = dates.filter(dates.is_between(val_start, val_end, closed="both"))
    if evaluation.is_empty() or not evaluation.eq(expected.max()).any():
        raise ValueError("fold-scoped temporal holdout does not cover the evaluation endpoint")


@dataclass(frozen=True)
class EligibilityManifest:
    entity_schema: dict[str, Any]
    source_identity: dict[str, Any]
    logic_identity: dict[str, Any]
    n_eligible: int
    sorted_key_digest: str
    folds: tuple[dict[str, Any], ...]
    eligible_keys: Any = field(repr=False, compare=False)

    @classmethod
    def resolve(
        cls,
        keys,
        *,
        entity_columns: tuple[str, ...] = ("symbol",),
        timestamp_column: str = "timestamp",
        fold_column: str = "fold",
        source_identity: dict[str, Any],
        logic_identity: dict[str, Any],
        diagnostics_by_fold: dict[int, dict[str, Any]] | None = None,
    ) -> EligibilityManifest:
        import polars as pl

        frame = keys if isinstance(keys, pl.DataFrame) else pl.from_pandas(keys)
        key_columns = [*entity_columns, timestamp_column, fold_column]
        missing = set(key_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"eligibility keys are missing columns: {sorted(missing)}")
        if not entity_columns:
            raise ValueError("eligibility manifest requires at least one entity column")
        selected = frame.select(key_columns)
        if selected.null_count().row(0) != tuple(0 for _ in key_columns):
            raise ValueError("eligibility keys cannot contain null values")
        if selected.n_unique(key_columns) != selected.height:
            raise ValueError("eligibility keys must be unique")
        selected = selected.sort(key_columns)
        diagnostics = diagnostics_by_fold or {}
        fold_records = []
        for fold_id in sorted(selected.get_column(fold_column).unique().to_list()):
            fold_keys = selected.filter(pl.col(fold_column) == fold_id)
            fold_records.append(
                {
                    "fold": int(fold_id),
                    "n_eligible": fold_keys.height,
                    "sorted_key_digest": value_digest(fold_keys, tuple(key_columns)),
                    "diagnostics": canonical_value(diagnostics.get(int(fold_id), {})),
                }
            )
        return cls(
            entity_schema={
                "entity_columns": list(entity_columns),
                "timestamp": timestamp_column,
                "fold": fold_column,
                "dtypes": {column: str(selected.schema[column]) for column in key_columns},
            },
            source_identity=canonical_value(source_identity),
            logic_identity=canonical_value(logic_identity),
            n_eligible=selected.height,
            sorted_key_digest=value_digest(selected, tuple(key_columns)),
            folds=tuple(fold_records),
            eligible_keys=selected,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_schema": self.entity_schema,
            "source_identity": self.source_identity,
            "logic_identity": self.logic_identity,
            "n_eligible": self.n_eligible,
            "sorted_key_digest": self.sorted_key_digest,
            "folds": list(self.folds),
        }


@dataclass(frozen=True)
class ResolvedCVSpec:
    request: dict[str, Any]
    normalized_folds: tuple[dict[str, Any], ...]
    identity: str
    eligibility: EligibilityManifest | None = None

    def as_dict(self) -> dict[str, Any]:
        resolved = {
            "request": self.request,
            "folds": list(self.normalized_folds),
            "identity": self.identity,
        }
        if self.eligibility is not None:
            resolved["eligibility"] = self.eligibility.as_dict()
        return resolved


@dataclass(frozen=True)
class CVSpec:
    training_window: int | float | str | None
    validation_window: int | float | str
    retrain_every: int | str | None
    folds: tuple[int, ...]
    expanding: bool = False
    horizon: str = "0D"
    gap: str | None = None
    holdout_start: str | None = None
    holdout_end: str | None = None
    calendar: str | None = None
    decision_cadence: str | None = None

    def __post_init__(self) -> None:
        normalized_folds = tuple(sorted({int(fold) for fold in self.folds}))
        if not normalized_folds or normalized_folds[0] < 0:
            raise ValueError("folds must contain at least one non-negative fold id")
        object.__setattr__(self, "folds", normalized_folds)

    @classmethod
    def walk_forward(
        cls,
        *,
        training_window: int | float | str | None,
        validation_window: int | float | str,
        retrain_every: int | str | None = None,
        folds=None,
        expanding: bool = False,
        horizon: str = "0D",
        gap: str | None = None,
        holdout_start: str | None = None,
        holdout_end: str | None = None,
        calendar: str | None = None,
        decision_cadence: str | None = None,
    ) -> CVSpec:
        normalized_folds = tuple(int(fold) for fold in (range(5) if folds is None else folds))
        return cls(
            training_window=training_window,
            validation_window=validation_window,
            retrain_every=retrain_every,
            folds=normalized_folds,
            expanding=expanding,
            horizon=horizon,
            gap=gap,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            calendar=calendar,
            decision_cadence=decision_cadence,
        )

    def with_changes(self, **changes) -> CVSpec:
        return replace(self, **changes)

    def resolve(
        self,
        timeline,
        *,
        date_col: str = "timestamp",
        eligibility: EligibilityManifest | None = None,
    ) -> ResolvedCVSpec:
        step_size = self.retrain_every
        if isinstance(step_size, str):
            if step_size != self.validation_window:
                raise ValueError(
                    "a distinct retrain_every duration must be expressed as an integer "
                    "observation step for the existing splitter"
                )
            step_size = None
        config = {
            "n_splits": max(self.folds) + 1,
            "train_size": self.training_window,
            "val_size": self.validation_window,
            "holdout_start": self.holdout_start,
            "holdout_end": self.holdout_end,
            "calendar": self.calendar,
            "step_size": step_size,
            "expanding": self.expanding,
        }
        generated = generate_cv_splits(
            timeline,
            label_buffer=self.gap or self.horizon,
            outcome_horizon=self.horizon,
            date_col=date_col,
            cv_config=config,
        )
        selected = [split for split in generated if int(split["fold"]) in self.folds]
        if len(selected) != len(self.folds):
            raise ValueError("requested folds were not all produced by the existing CV generator")
        normalized = tuple(
            {
                "fold": int(split["fold"]),
                "train_start": _normalize_boundary(split["train_start"]),
                "train_end": _normalize_boundary(split["train_end"]),
                "val_start": _normalize_boundary(split["val_start"]),
                "val_end": _normalize_boundary(split["val_end"]),
            }
            for split in selected
        )
        request = {
            "training_window": self.training_window,
            "validation_window": self.validation_window,
            "retrain_every": self.retrain_every,
            "folds": list(self.folds),
            "expanding": self.expanding,
            "horizon": self.horizon,
            "gap": self.gap or self.horizon,
            "holdout_start": self.holdout_start,
            "holdout_end": self.holdout_end,
            "calendar": self.calendar,
            "decision_cadence": self.decision_cadence,
        }
        if eligibility is not None:
            eligible_folds = {fold["fold"] for fold in eligibility.folds}
            if eligible_folds != set(self.folds):
                raise ValueError(
                    "eligibility manifest folds do not match the resolved CV request: "
                    f"{sorted(eligible_folds)} != {list(self.folds)}"
                )
        identity_input = {"request": request, "folds": normalized}
        if eligibility is not None:
            identity_input["eligibility"] = eligibility.as_dict()
        identity = compute_hash(canonical_json(identity_input))
        return ResolvedCVSpec(request, normalized, identity, eligibility)
