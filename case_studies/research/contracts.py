from __future__ import annotations

from enum import StrEnum


class ExecutionTier(StrEnum):
    CANONICAL = "canonical"
    PREVIEW = "preview"


class LifecycleState(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    LOCKED = "LOCKED"
    HOLDOUT_EVALUATED = "HOLDOUT_EVALUATED"
