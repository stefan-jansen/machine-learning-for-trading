"""Serializable, deterministic configuration for the MNQ objective strategy."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import time
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import signals
from .risk import MAX_CONTRACTS, MIN_CONTRACTS, MNQ_POINT_VALUE, CostModel
from .volume_profile import LVN_PERCENTILE


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
        default_factory=lambda: MappingProxyType(
            {
                "rth_start": "09:30",
                "rth_end": "16:00",
                "maintenance_start": "16:00",
                "maintenance_end": "18:00",
                "overnight_start": "18:00",
            }
        )
    )

    def __post_init__(self) -> None:
        if not self.instrument:
            raise ValueError("instrument must not be empty")
        if not self.timezone:
            raise ValueError("timezone must not be empty")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
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
        if not isinstance(self.momentum_lookback, int) or isinstance(self.momentum_lookback, bool):
            raise ValueError("momentum_lookback must be an integer")
        if self.momentum_lookback < 1:
            raise ValueError("momentum_lookback must be positive")
        if self.stop_points >= self.target_points:
            raise ValueError("stop_points must be less than target_points")
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
        parsed_boundaries: dict[str, time] = {}
        for name, value in self.session_boundaries.items():
            if not isinstance(value, str) or len(value) != 5:
                raise ValueError("session_boundaries must use HH:MM strings")
            try:
                parsed_boundaries[name] = time.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"invalid session boundary {name}: expected HH:MM") from exc
            if parsed_boundaries[name].strftime("%H:%M") != value:
                raise ValueError(f"invalid session boundary {name}: expected HH:MM")
        if not (
            parsed_boundaries["rth_start"] < parsed_boundaries["rth_end"]
            and parsed_boundaries["rth_end"] == parsed_boundaries["maintenance_start"]
            and parsed_boundaries["maintenance_start"] < parsed_boundaries["maintenance_end"]
            and parsed_boundaries["maintenance_end"] == parsed_boundaries["overnight_start"]
        ):
            raise ValueError("session boundaries must be ordered and contiguous")
        object.__setattr__(
            self, "session_boundaries", MappingProxyType(dict(self.session_boundaries))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible representation of this configuration."""
        serialized = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "session_boundaries"
        }
        serialized["cost_model"] = asdict(self.cost_model)
        serialized["session_boundaries"] = dict(self.session_boundaries)
        return serialized

    def validate_fixed_contract(self) -> bool:
        """Ensure fixed signal thresholds match the Task 3 implementation."""
        expected = {
            "point_value": MNQ_POINT_VALUE,
            "rejection_wick_ratio": signals.REJECTION_WICK_RATIO,
            "rejection_close_pct": signals.REJECTION_CLOSE_PCT,
            "momentum_body_range": signals.MOMENTUM_BODY_RANGE,
            "momentum_close_pct": signals.MOMENTUM_CLOSE_PCT,
            "momentum_lookback": signals.MOMENTUM_LOOKBACK,
            "momentum_multiplier": signals.MOMENTUM_MULT,
        }
        actual = {name: getattr(self, name) for name in expected}
        actual["lvn_percentile"] = self.lvn_percentile
        expected["lvn_percentile"] = LVN_PERCENTILE
        differences = {
            name: (actual[name], expected[name])
            for name in expected
            if actual[name] != expected[name]
        }
        if differences:
            details = ", ".join(
                f"{name}={actual_value!r} (expected {expected_value!r})"
                for name, (actual_value, expected_value) in differences.items()
            )
            raise ValueError(f"fixed signal contract drift: {details}")
        return True

    @property
    def config_hash(self) -> str:
        """Return the SHA-256 hash of canonical sorted configuration JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
