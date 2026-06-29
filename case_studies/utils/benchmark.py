"""Load period-stratified benchmark metrics for case-study analyses.

The benchmark parquets in ``case_studies/{cs}/benchmark/`` cover both validation
and holdout periods. **Consumers must pull the period-specific block** when
comparing strategy holdout to benchmark holdout, etc. — using the overall
metrics for a period-specific comparison would mix windows.

Part of the downloaded case-study artifacts. JSON schema:

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
from datetime import date as _date
from pathlib import Path
from typing import Literal

import polars as pl
import yaml

from utils.paths import get_case_study_dir

Period = Literal["overall", "validation", "holdout"]


def benchmark_dir(case_study: str) -> Path:
    return get_case_study_dir(case_study) / "benchmark"


def _to_date(v) -> _date:
    """Parse a YYYY-MM-DD-prefixed string/date to a Python ``date``.

    Comparing on ``dt.date()`` is tz-agnostic — it sidesteps the
    naive-vs-tz-aware-Datetime cast hazard entirely (Polars silently treats
    naive sources as UTC under cast, which would shift boundaries on a
    non-UTC tz-aware parquet).
    """
    if isinstance(v, _date):
        return v
    return _date.fromisoformat(str(v)[:10])


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
        # Legacy file without stratification — only overall is meaningful
        if period == "overall":
            return {k: meta[k] for k in ("sharpe", "cagr", "vol", "n_periods") if k in meta}
        return None
    return bp.get(period)


def load_benchmark_returns(
    case_study: str,
    label: str,
    period: Period = "overall",
) -> pl.DataFrame:
    """Return the daily ``ew_return`` series sliced to the requested period.

    Boundary source of truth is the JSON's ``by_period.{validation,holdout}_window``.
    When the JSON is present,
    its ``by_period`` is authoritative — a missing window means the period was
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
