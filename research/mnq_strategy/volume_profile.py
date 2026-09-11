"""Previous New York RTH volume-profile calculations for MNQ research."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import polars as pl


MNQ_TICK_SIZE = 0.25
LVN_PERCENTILE = 0.20


def _validate_profile_input(frame: pl.DataFrame) -> None:
    missing = [column for column in ("price", "volume") if column not in frame.columns]
    if missing:
        raise ValueError(f"required profile columns missing: {', '.join(missing)}")
    if frame.height == 0:
        raise ValueError("profile input must not be empty")
    for price, volume in frame.select(["price", "volume"]).iter_rows():
        if not isinstance(price, (int, float)) or not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError("price values must be finite and positive")
        if not isinstance(volume, (int, float)) or not math.isfinite(float(volume)) or float(volume) <= 0:
            raise ValueError("volume values must be finite and positive")


def _price_bin(price: float) -> float:
    return round(round(price / MNQ_TICK_SIZE) * MNQ_TICK_SIZE, 2)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _find_lvn_zones(volume_by_price: dict[float, float]) -> list[dict[str, float]]:
    threshold = _percentile(list(volume_by_price.values()), LVN_PERCENTILE)
    low_bins = [price for price, volume in volume_by_price.items() if volume < threshold]
    zones: list[dict[str, float]] = []
    for price in sorted(low_bins):
        if zones and math.isclose(price - zones[-1]["high"], MNQ_TICK_SIZE):
            zones[-1]["high"] = price
        else:
            zones.append({"low": price, "high": price})
    return zones


def build_rth_profile(
    rth_bars: pl.DataFrame, value_area_fraction: float = 0.40
) -> dict[str, Any]:
    """Build a deterministic MNQ price-volume profile from one RTH session."""
    if not isinstance(value_area_fraction, (int, float)) or not math.isfinite(float(value_area_fraction)):
        raise ValueError("value_area_fraction must be finite")
    if not 0 < value_area_fraction <= 1:
        raise ValueError("value_area_fraction must be greater than 0 and at most 1")
    _validate_profile_input(rth_bars)

    volume_by_price: dict[float, float] = {}
    for price, volume in rth_bars.select(["price", "volume"]).iter_rows():
        bin_price = _price_bin(float(price))
        volume_by_price[bin_price] = volume_by_price.get(bin_price, 0.0) + float(volume)

    poc = min(
        volume_by_price,
        key=lambda price: (-volume_by_price[price], price),
    )
    total_volume = sum(volume_by_price.values())
    target_volume = total_volume * float(value_area_fraction)
    ordered_prices = sorted(volume_by_price)
    poc_index = ordered_prices.index(poc)
    included_indices = {poc_index}
    included_volume = volume_by_price[poc]
    while included_volume < target_volume and len(included_indices) < len(ordered_prices):
        boundary_indices = {
            index
            for index in (min(included_indices) - 1, max(included_indices) + 1)
            if 0 <= index < len(ordered_prices)
        }
        selected_index = min(
            boundary_indices,
            key=lambda index: (-volume_by_price[ordered_prices[index]], ordered_prices[index]),
        )
        included_indices.add(selected_index)
        included_volume += volume_by_price[ordered_prices[selected_index]]

    included_prices = [ordered_prices[index] for index in included_indices]

    return {
        "poc": poc,
        "vah": max(included_prices),
        "val": min(included_prices),
        "value_area_fraction": float(value_area_fraction),
        "volume_by_price": dict(sorted(volume_by_price.items())),
        "lvn_zones": _find_lvn_zones(volume_by_price),
    }


def attach_previous_rth_profile(bars: pl.DataFrame) -> pl.DataFrame:
    """Attach the profile from the prior fully closed RTH session without leakage."""
    required = {"session_date", "session_type", "close", "volume", "bar_closed"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"required MNQ contract columns missing: {', '.join(missing)}")

    profiles: dict[date, dict[str, Any]] = {}
    rth = bars.filter((pl.col("session_type") == "rth") & pl.col("bar_closed"))
    for session_date in rth.get_column("session_date").unique().sort().to_list():
        session_bars = rth.filter(pl.col("session_date") == session_date).select(
            pl.col("close").alias("price"), "volume"
        )
        profiles[session_date] = build_rth_profile(session_bars)

    session_dates = sorted(profiles)
    attached: list[dict[str, Any] | None] = []
    for current_date in bars.get_column("session_date").to_list():
        previous_dates = [profile_date for profile_date in session_dates if profile_date < current_date]
        previous_date = max(previous_dates) if previous_dates else None
        profile = profiles.get(previous_date) if previous_date is not None else None
        attached.append(
            None
            if profile is None
            else {
                "previous_profile_date": previous_date,
                "previous_poc": profile["poc"],
                "previous_vah": profile["vah"],
                "previous_val": profile["val"],
                "previous_lvn_zones": profile["lvn_zones"],
            }
        )

    return bars.with_columns(
        pl.Series("previous_profile_date", [item["previous_profile_date"] if item else None for item in attached], dtype=pl.Date),
        pl.Series("previous_poc", [item["previous_poc"] if item else None for item in attached], dtype=pl.Float64),
        pl.Series("previous_vah", [item["previous_vah"] if item else None for item in attached], dtype=pl.Float64),
        pl.Series("previous_val", [item["previous_val"] if item else None for item in attached], dtype=pl.Float64),
        pl.Series("previous_lvn_zones", [item["previous_lvn_zones"] if item else None for item in attached]),
    )
