"""Authoritative carrier pins and registry-backed carrier identity resolution."""

from __future__ import annotations

import sqlite3
from functools import cache
from pathlib import Path

import polars as pl

from utils.paths import get_case_study_dir

# These values belong to their case-study owners. Consumers import this mapping
# rather than copying values or translating them to config-name predicates.
CARRIER_PINS: dict[str, str] = {
    "us_firm_characteristics": "e676e1989e1f",
    "sp500_options": "7f32a34d107b",
}


def carrier_pin(case_study: str) -> str | None:
    """Return the owner-controlled validation backtest-hash prefix, if any."""
    return CARRIER_PINS.get(case_study)


@cache
def _carrier_config_name(case_study: str, db_path_str: str) -> str:
    pin = carrier_pin(case_study)
    if pin is None:
        raise ValueError(f"No carrier pin is configured for {case_study}")

    db_path = Path(db_path_str)
    if not db_path.exists():
        raise FileNotFoundError(f"Carrier registry does not exist: {db_path}")

    with sqlite3.connect(str(db_path)) as db:
        rows = db.execute(
            """
            SELECT DISTINCT t.config_name
            FROM backtest_runs b
            JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
            JOIN training_runs t ON t.training_hash = p.training_hash
            WHERE b.backtest_hash LIKE ?
            """,
            (pin + "%",),
        ).fetchall()
    names = {row[0] for row in rows if row[0] is not None}
    if len(names) != 1:
        raise RuntimeError(
            f"Carrier pin {pin!r} for {case_study} resolved to {sorted(names)!r}; "
            "expected exactly one config_name"
        )
    return names.pop()


def carrier_config_name(case_study: str, db_path: Path | None = None) -> str:
    """Resolve a pin to its model config without duplicating owner choices."""
    path = db_path or (get_case_study_dir(case_study) / "run_log" / "registry.db")
    return _carrier_config_name(case_study, str(path.resolve()))


def filter_to_carrier_config(
    df: pl.DataFrame,
    case_study: str,
    *,
    db_path: Path | None = None,
) -> pl.DataFrame:
    """Filter a candidate frame to its pinned model, failing closed on drift."""
    if carrier_pin(case_study) is None or df.is_empty():
        return df
    if "config_name" not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(
            f"config_name is required to apply carrier pin for {case_study}"
        )
    name = carrier_config_name(case_study, db_path)
    pinned = df.filter(pl.col("config_name") == name)
    if pinned.is_empty():
        raise ValueError(
            f"Carrier config {name!r} for {case_study} is absent after candidate filters"
        )
    return pinned


def prioritize_carrier_hash(df: pl.DataFrame, case_study: str) -> pl.DataFrame:
    """Move the pinned row first while retaining fallbacks, failing closed."""
    pin = carrier_pin(case_study)
    if pin is None or df.is_empty():
        return df
    if "backtest_hash" not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(
            f"backtest_hash is required to apply carrier pin for {case_study}"
        )
    is_pinned = pl.col("backtest_hash").str.starts_with(pin)
    if df.filter(is_pinned).is_empty():
        raise ValueError(f"Carrier pin {pin!r} for {case_study} is absent after candidate filters")
    return pl.concat([df.filter(is_pinned), df.filter(~is_pinned)], how="vertical")
