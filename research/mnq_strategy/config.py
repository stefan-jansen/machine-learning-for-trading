"""Serializable, deterministic configuration for the MNQ objective strategy."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .risk import MAX_CONTRACTS, MIN_CONTRACTS, MNQ_POINT_VALUE, CostModel


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable strategy parameters with a stable serialized identity.

    The backtest uses a 10.0-point stop and a 20.0-point target unless a
    caller explicitly supplies different finite distances.
    """

    instrument: str = "MNQ"
    timezone: str = "America/New_York"
    bar_minutes: int = 5
    value_area_fraction: float = 0.40
    min_contracts: int = MIN_CONTRACTS
    max_contracts: int = MAX_CONTRACTS
    max_trade_risk: float = 250.0
    daily_stop: float = 400.0
    max_consecutive_losses: int = 2
    point_value: float = MNQ_POINT_VALUE
    cost_model: CostModel = field(
        default_factory=lambda: CostModel(
            commission_per_contract=1.50,
            slippage_points=0.50,
        )
    )

    rejection_wick_ratio: float = 2.0
    rejection_close_pct: float = 0.30
    momentum_body_range: float = 0.60
    momentum_close_pct: float = 0.80
    momentum_lookback: int = 6
    momentum_multiplier: float = 1.5
    lvn_percentile: float = 0.25

    stop_points: float = 10.0
    target_points: float = 20.0

    session_boundaries: dict[str, str] = field(
        default_factory=lambda: {
            "rth_start": "09:30",
            "rth_end": "16:00",
            "maintenance_start": "16:00",
            "maintenance_end": "18:00",
            "overnight_start": "18:00",
        }
    )

    def __post_init__(self) -> None:
        if not self.instrument:
            raise ValueError("instrument must not be empty")
        if not self.timezone:
            raise ValueError("timezone must not be empty")
        if not isinstance(self.bar_minutes, int) or self.bar_minutes <= 0:
            raise ValueError("bar_minutes must be a positive integer")
        if not 0 < self.value_area_fraction <= 1:
            raise ValueError("value_area_fraction must be greater than 0 and at most 1")
        if (
            not isinstance(self.min_contracts, int)
            or not isinstance(self.max_contracts, int)
            or self.min_contracts < 1
            or self.max_contracts < self.min_contracts
        ):
            raise ValueError("contract limits must be positive and ordered")
        if not isinstance(self.max_consecutive_losses, int) or self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be positive")
        for name in (
            "value_area_fraction",
            "max_trade_risk",
            "daily_stop",
            "point_value",
            "rejection_wick_ratio",
            "rejection_close_pct",
            "momentum_body_range",
            "momentum_close_pct",
            "momentum_multiplier",
            "lvn_percentile",
            "stop_points",
            "target_points",
        ):
            _validate_finite(name, float(getattr(self, name)))
        if self.point_value <= 0:
            raise ValueError("point_value must be positive")
        if self.max_trade_risk <= 0 or self.daily_stop <= 0:
            raise ValueError("risk limits must be positive")
        if self.stop_points <= 0 or self.target_points <= 0:
            raise ValueError("stop_points and target_points must be positive")
        if self.momentum_lookback < 1:
            raise ValueError("momentum_lookback must be positive")
        if not 0 <= self.rejection_close_pct <= 1:
            raise ValueError("rejection_close_pct must be between 0 and 1")
        if not 0 <= self.momentum_body_range <= 1:
            raise ValueError("momentum_body_range must be between 0 and 1")
        if not 0 <= self.momentum_close_pct <= 1:
            raise ValueError("momentum_close_pct must be between 0 and 1")
        if self.rejection_wick_ratio <= 0 or self.momentum_multiplier <= 0:
            raise ValueError("wick and momentum multipliers must be positive")
        if not 0 < self.lvn_percentile <= 1:
            raise ValueError("lvn_percentile must be greater than 0 and at most 1")
        if set(self.session_boundaries) != {
            "rth_start",
            "rth_end",
            "maintenance_start",
            "maintenance_end",
            "overnight_start",
        }:
            raise ValueError("session_boundaries must define the approved session times")
        if any(
            not isinstance(value, str) or len(value) != 5 or value[2] != ":"
            for value in self.session_boundaries.values()
        ):
            raise ValueError("session_boundaries must use HH:MM strings")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible representation of this configuration."""
        serialized = asdict(self)
        serialized["cost_model"] = asdict(self.cost_model)
        serialized["session_boundaries"] = dict(self.session_boundaries)
        return serialized

    @property
    def config_hash(self) -> str:
        """Return the SHA-256 hash of canonical sorted configuration JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
