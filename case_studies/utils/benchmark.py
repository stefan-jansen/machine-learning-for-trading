"""Load period-stratified benchmark metrics for case-study analyses.

The benchmark parquets in ``case_studies/{cs}/benchmark/`` cover both validation
and holdout periods. **Consumers must pull the period-specific block** when
comparing strategy holdout to benchmark holdout, etc. Using the overall
metrics for a period-specific comparison would mix windows.

Generated from the declared case-study universe and canonical label artifact, then tracked with the
repository as a release input. JSON schema:

    {
      "case_study": "etfs",
      "label": "fwd_ret_21d",
      "method": "...",
      "periods_per_year": 252,
      "n_symbols_in_universe": 99,
      "sharpe": ..., "cagr": ..., "vol": ..., "n_periods": ...,
      "ts_min": "...", "ts_max": "...",
      "by_period": {
        "overall":    {"sharpe": ..., "cagr": ..., "vol": ..., "n_periods": ...},
        "validation": {"sharpe": ..., "cagr": ..., "vol": ..., "n_periods": ...},
        "holdout":    {"sharpe": ..., "cagr": ..., "vol": ..., "n_periods": ...},
        "validation_window": ["...", "..."],
        "holdout_window":    ["...", "..."]
      }
    }
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import date as _date
from pathlib import Path
from typing import Literal

import polars as pl
import yaml

from utils.paths import get_case_study_dir

from .artifact_digest import read_digest, value_digest
from .backtest_loaders import thin_to_rebalance_dates
from .cv_window import canonical_window

Period = Literal["overall", "validation", "holdout"]
Window = tuple[_date, _date]


def benchmark_dir(case_study: str) -> Path:
    return get_case_study_dir(case_study) / "benchmark"


def _to_date(v) -> _date:
    """Parse a YYYY-MM-DD-prefixed string/date to a Python ``date``.

    Comparing on ``dt.date()`` is tz-agnostic. It sidesteps the
    naive-vs-tz-aware-Datetime cast hazard entirely (Polars silently treats
    naive sources as UTC under cast, which would shift boundaries on a
    non-UTC tz-aware parquet).
    """
    if isinstance(v, _date):
        return v
    return _date.fromisoformat(str(v)[:10])


def _content_digest(value: object) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def declared_universe(setup: Mapping) -> list[str]:
    """Return the explicit benchmark roster declared by a case-study setup."""
    universe = setup.get("universe") or {}
    roster = universe.get("symbols") or universe.get("assets")
    if roster is None and universe.get("product_groups"):
        roster = [
            product for products in universe["product_groups"].values() for product in products
        ]
    if not roster:
        raise ValueError("Benchmark generation requires an explicit universe roster")
    symbols = sorted(str(symbol) for symbol in roster)
    if len(symbols) != len(set(symbols)):
        raise ValueError("The declared benchmark universe contains duplicate symbols")
    declared_count = universe.get("n_assets")
    if declared_count is not None and int(declared_count) != len(symbols):
        raise ValueError(
            f"universe.n_assets={declared_count} does not match the {len(symbols)} declared symbols"
        )
    return symbols


def _return_metrics(returns: pl.DataFrame, periods_per_year: float) -> dict:
    values = returns["ew_return"].to_numpy()
    n_periods = len(values)
    if n_periods == 0:
        return {
            "sharpe": None,
            "cagr": None,
            "vol": None,
            "n_periods": 0,
            "periods_per_year": periods_per_year,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n_periods > 1 else 0.0
    wealth = float((1.0 + values).prod())
    return {
        "sharpe": mean / std * math.sqrt(periods_per_year) if std > 0 else None,
        "cagr": wealth ** (periods_per_year / n_periods) - 1.0 if wealth > 0 else None,
        "vol": std * math.sqrt(periods_per_year),
        "n_periods": n_periods,
        "periods_per_year": periods_per_year,
    }


def _slice_window(frame: pl.DataFrame, window: Window | None) -> pl.DataFrame:
    if window is None:
        return frame.clear()
    start, end = window
    return frame.filter(pl.col("timestamp").is_between(start, end, closed="both"))


def _intraday_cadence_minutes(cadence: str) -> int | None:
    minute = re.fullmatch(r"(\d+)_minute", cadence)
    if minute:
        return int(minute.group(1))
    hour = re.fullmatch(r"(\d+)_hour(?:_\w+)?", cadence)
    if hour:
        return int(hour.group(1)) * 60
    return None


def _align_decision_timestamps(timestamps: pl.Series, cadence: str) -> pl.Series:
    """Select the declared intraday clock from a finer timestamp grid."""
    minutes = _intraday_cadence_minutes(cadence)
    unique = timestamps.unique().sort()
    if minutes is None or minutes == 1 or unique.is_empty():
        return unique
    return (
        pl.DataFrame({"timestamp": unique})
        .with_columns(pl.col("timestamp").dt.date().alias("_session"))
        .with_columns(pl.col("timestamp").min().over("_session").alias("_session_start"))
        .with_columns(
            (pl.col("timestamp") - pl.col("_session_start"))
            .dt.total_minutes()
            .alias("_elapsed_minutes")
        )
        .filter(pl.col("_elapsed_minutes") % minutes == 0)
        .get_column("timestamp")
    )


def _resolve_decision_cadence(setup: Mapping) -> str:
    decision = setup.get("decision") or {}
    for key in ("entry_cadence", "cadence", "bar_frequency"):
        value = decision.get(key)
        if value:
            return str(value)
    raise KeyError("decision requires entry_cadence, cadence, or bar_frequency")


def _decision_periods_per_year(
    cadence: str,
    rebalance_step: int,
    daily_periods_per_year: int,
) -> float:
    """Annualization frequency after intraday decisions are compounded by session."""
    if cadence == "monthly_month_end":
        schedule_periods = 12.0
    elif cadence in {"weekly", "weekly_friday", "weekly_friday_close"}:
        schedule_periods = 52.0
    elif "funding" in cadence:
        schedule_periods = 3.0 * daily_periods_per_year
    else:
        minutes = _intraday_cadence_minutes(cadence)
        if minutes is not None:
            schedule_periods = 390.0 / minutes * daily_periods_per_year
        else:
            schedule_periods = float(daily_periods_per_year)
    return min(float(daily_periods_per_year), schedule_periods / rebalance_step)


def build_equal_weight_benchmark(
    labels: pl.DataFrame,
    *,
    case_study: str,
    label: str,
    symbols: Sequence[str],
    windows: Mapping[str, Window | None],
    cadence: str,
    rebalance_step: int,
    calendar: str,
    periods_per_year: int,
    label_digest: str,
) -> tuple[pl.DataFrame, dict]:
    """Build scheduled full-universe returns from canonical forward labels."""
    entity_col = "product" if case_study == "cme_futures" else "symbol"
    required = {"timestamp", entity_col, label}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"Label frame is missing required columns: {missing}")
    roster = sorted(str(symbol) for symbol in symbols)
    if not roster or len(roster) != len(set(roster)):
        raise ValueError("Benchmark symbols must be a non-empty unique roster")
    if rebalance_step < 1:
        raise ValueError("rebalance_step must be at least one")
    active_windows = [window for window in windows.values() if window is not None]
    if not active_windows:
        raise ValueError("At least one benchmark window is required")
    start = min(window[0] for window in active_windows)
    end = max(window[1] for window in active_windows)
    scoped = labels.filter(
        pl.col("timestamp").cast(pl.Date).is_between(start, end, closed="both")
        & pl.col(entity_col).cast(pl.String).is_in(roster)
    )
    duplicates = scoped.group_by("timestamp", entity_col).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"Label frame contains duplicate (timestamp, {entity_col}) rows")
    nonfinite = scoped.filter(pl.col(label).is_not_null() & ~pl.col(label).is_finite())
    if nonfinite.height:
        raise ValueError(f"Label frame contains {nonfinite.height} non-finite {label} values")

    decision_grid = _align_decision_timestamps(scoped.get_column("timestamp"), cadence)
    schedule = thin_to_rebalance_dates(
        pl.DataFrame({"timestamp": decision_grid}),
        cadence=cadence,
        step=rebalance_step,
        calendar=calendar,
    ).select("timestamp")
    decision_returns = (
        scoped.join(schedule, on="timestamp", how="semi")
        .drop_nulls(label)
        .group_by("timestamp")
        .agg(pl.col(label).mean().alias("decision_return"))
        .sort("timestamp")
    )
    daily = (
        decision_returns.with_columns(pl.col("timestamp").cast(pl.Date).alias("timestamp"))
        .group_by("timestamp", maintain_order=True)
        .agg(((pl.col("decision_return") + 1.0).product() - 1.0).alias("ew_return"))
        .sort("timestamp")
    )
    if daily.is_empty():
        raise ValueError(
            "No finite benchmark returns remain after roster, window, and schedule filters"
        )

    return_periods_per_year = _decision_periods_per_year(
        cadence,
        rebalance_step,
        periods_per_year,
    )

    validation_window = windows.get("validation")
    holdout_window = windows.get("holdout")
    by_period = {
        "overall": _return_metrics(daily, return_periods_per_year),
        "validation": _return_metrics(
            _slice_window(daily, validation_window), return_periods_per_year
        ),
        "holdout": _return_metrics(_slice_window(daily, holdout_window), return_periods_per_year),
        "validation_window": [str(v) for v in validation_window] if validation_window else None,
        "holdout_window": [str(v) for v in holdout_window] if holdout_window else None,
    }
    observed_symbols = scoped.filter(pl.col(label).is_not_null())[entity_col].n_unique()
    metadata = {
        "case_study": case_study,
        "label": label,
        "method": "scheduled_compounded_cross_sectional_mean_forward_label",
        "periods_per_year": return_periods_per_year,
        "n_symbols_in_universe": len(roster),
        "n_symbols_observed": observed_symbols,
        **by_period["overall"],
        "ts_min": str(daily["timestamp"].min()),
        "ts_max": str(daily["timestamp"].max()),
        "by_period": by_period,
        "configuration": {
            "cadence": cadence,
            "rebalance_step": rebalance_step,
            "calendar": calendar,
            "entity_col": entity_col,
            "daily_periods_per_year": periods_per_year,
        },
        "inputs": {
            "label_digest": label_digest,
            "universe_digest": _content_digest(roster),
        },
        "output_digest": value_digest(daily),
        "written_by": "scripts/generate_case_study_benchmarks.py",
    }
    return daily, metadata


def write_benchmark(
    returns: pl.DataFrame,
    metadata: Mapping,
    *,
    output_dir: Path,
    label: str,
) -> tuple[Path, Path]:
    """Replace a benchmark parquet and metadata after both temporary files are complete."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{label}.parquet"
    json_path = output_dir / f"{label}.json"
    token = uuid.uuid4().hex
    parquet_tmp = output_dir / f".{label}.{token}.parquet.tmp"
    json_tmp = output_dir / f".{label}.{token}.json.tmp"
    try:
        returns.write_parquet(parquet_tmp)
        json_tmp.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n")
        os.replace(parquet_tmp, parquet_path)
        os.replace(json_tmp, json_path)
    finally:
        parquet_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
    return parquet_path, json_path


def generate_benchmark(
    case_study: str,
    label: str,
    *,
    windows: Mapping[str, Window | None] | None = None,
) -> tuple[Path, Path]:
    """Generate one tracked benchmark from its setup and canonical label artifact."""
    case_dir = get_case_study_dir(case_study, create=False)
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    label_path = case_dir / "labels" / f"{label}.parquet"
    digest = read_digest(label_path)["digest"]
    resolved_windows = windows or {
        "validation": canonical_window(case_study, label, split="validation"),
        "holdout": canonical_window(case_study, label, split="holdout"),
    }
    returns, metadata = build_equal_weight_benchmark(
        pl.read_parquet(label_path),
        case_study=case_study,
        label=label,
        symbols=declared_universe(setup),
        windows=resolved_windows,
        cadence=_resolve_decision_cadence(setup),
        rebalance_step=int(setup["labels"]["rebalance_step"][label]),
        calendar=str(setup["evaluation"]["calendar"]),
        periods_per_year=int(setup["evaluation"]["periods_per_year"]),
        label_digest=str(digest),
    )
    return write_benchmark(returns, metadata, output_dir=case_dir / "benchmark", label=label)


def load_benchmark_metrics(
    case_study: str,
    label: str,
    period: Period = "overall",
) -> dict | None:
    """Return the {sharpe, cagr, vol, n_periods} block for the requested period.

    None if the JSON is missing or the requested block is not populated (e.g.
    holdout block when the case study has no holdout window).
    """
    p = benchmark_dir(case_study) / f"{label}.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    bp = meta.get("by_period")
    if bp is None:
        # Legacy file without stratification. Only overall is meaningful.
        if period == "overall":
            return {k: meta[k] for k in ("sharpe", "cagr", "vol", "n_periods") if k in meta}
        return None
    return bp.get(period)


def load_benchmark_periods_per_year(case_study: str, label: str) -> float | None:
    """Return the annualization frequency stored with a benchmark artifact."""
    path = benchmark_dir(case_study) / f"{label}.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text()).get("periods_per_year")
    return float(value) if value is not None else None


def load_benchmark_returns(
    case_study: str,
    label: str,
    period: Period = "overall",
) -> pl.DataFrame:
    """Return the scheduled ``ew_return`` series sliced to the requested period.

    Boundary source of truth is the JSON's ``by_period.{validation,holdout}_window``.
    When the JSON is present,
    its ``by_period`` is authoritative. A missing window means the period was
    not populated by the writer (e.g. ``ho_df.height < 2``), and the consumer
    gets an empty frame rather than silently re-deriving from ``setup.yaml``.
    Falls back to ``setup.yaml.evaluation.{holdout_start, holdout_end}`` only
    when the JSON is absent (legacy unstratified files).
    """
    p = benchmark_dir(case_study) / f"{label}.parquet"
    if not p.exists():
        return pl.DataFrame()
    df = pl.read_parquet(p)
    if period == "overall":
        return df
    if period not in ("validation", "holdout"):
        raise ValueError(
            f"Unknown period {period!r}. Expected one of: 'overall', 'validation', 'holdout'."
        )

    json_p = benchmark_dir(case_study) / f"{label}.json"
    if json_p.exists():
        # JSON authoritative: respect what the writer recorded.
        bp = json.loads(json_p.read_text()).get("by_period", {}) or {}
        window = bp.get(f"{period}_window")
        if not window:
            return pl.DataFrame()
        start = _to_date(window[0])
        end = _to_date(window[1])
        return df.filter(
            (pl.col("timestamp").dt.date() >= start) & (pl.col("timestamp").dt.date() <= end)
        )

    # Legacy fallback: derive boundaries from setup.yaml.
    setup_path = get_case_study_dir(case_study) / "config" / "setup.yaml"
    if not setup_path.exists():
        return df
    setup = yaml.safe_load(setup_path.read_text())
    e = setup.get("evaluation", {})
    hs, he = e.get("holdout_start"), e.get("holdout_end")
    if hs is None or he is None:
        return df if period == "validation" else pl.DataFrame()
    hs_d = _to_date(hs)
    he_d = _to_date(he)
    if period == "validation":
        return df.filter(pl.col("timestamp").dt.date() < hs_d)
    return df.filter(
        (pl.col("timestamp").dt.date() >= hs_d) & (pl.col("timestamp").dt.date() <= he_d)
    )
