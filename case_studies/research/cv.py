from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from case_studies.utils.registry.specs import canonical_json, compute_hash
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


@dataclass(frozen=True)
class ResolvedCVSpec:
    request: dict[str, Any]
    normalized_folds: tuple[dict[str, Any], ...]
    identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "folds": list(self.normalized_folds),
            "identity": self.identity,
        }


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

    def resolve(self, timeline, *, date_col: str = "timestamp") -> ResolvedCVSpec:
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
        identity = compute_hash(canonical_json({"request": request, "folds": normalized}))
        return ResolvedCVSpec(request=request, normalized_folds=normalized, identity=identity)
