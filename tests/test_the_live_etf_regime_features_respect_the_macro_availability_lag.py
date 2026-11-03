"""The ETFs deployment loop must not build a feature from a macro value it cannot have.

`25_live_trading/_etfs_features.py` says it mirrors the financial-feature subset of
`case_studies/etfs/03_financial_features.py`. The case study re-stamps every yield-curve
value with the lag `config/setup.yaml` declares before its backward as-of join, so a
session reads the last value published on or before the previous day. The live module
joined on the session's own date, which FRED has not published at that session's close:
look-ahead in the training panel, and the served value differs from the fitted one.

The test states that as availability rather than as the implementation. Features for
session *t* are computed twice - once from the whole curve, once from the curve the loop
can actually see at *t*'s close, truncated to publications on or before *t - 1* - and the
three regime columns must agree. Reading the session's own value makes the two disagree,
because the truncated run falls back to the previous publication.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "25_live_trading"))

from _etfs_features import _add_regime_indicators  # noqa: E402

REGIME_COLUMNS = ["regime", "yield_curve_slope", "yield_curve_zscore"]
START = date(2015, 1, 1)
N_CALENDAR_DAYS = 1200
PROBES = (400, 600, 820)


def _calendar_curve(seed: int = 0) -> pl.DataFrame:
    """One row per calendar day, forward-filled, the way `data/macro/download.py` writes it.

    The values are a random walk with no repeats, so a session that reads its own date
    cannot coincide with one that reads the day before.
    """
    rng = np.random.default_rng(seed)
    days = [START + timedelta(d) for d in range(N_CALENDAR_DAYS)]
    business = [d for d in days if d.weekday() < 5]
    published = pl.DataFrame(
        {
            "timestamp": business,
            "slope": np.cumsum(rng.normal(0, 0.0006, len(business))) + 0.01,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    return (
        pl.DataFrame({"timestamp": days})
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .join_asof(published.sort("timestamp"), on="timestamp", strategy="backward")
        .drop_nulls("slope")
    )


def _sessions(curve: pl.DataFrame) -> pl.DataFrame:
    return curve.filter(pl.col("timestamp").dt.weekday() <= 5).select("timestamp")


@pytest.mark.parametrize("probe", PROBES)
def test_a_session_reads_no_yield_curve_value_published_after_the_previous_day(
    probe: int,
) -> None:
    curve = _calendar_curve()
    sessions = _sessions(curve)
    session = sessions["timestamp"][probe]
    visible_through = session - timedelta(days=1)

    whole = _add_regime_indicators(sessions, curve).filter(pl.col("timestamp") == session)
    visible = _add_regime_indicators(
        sessions.filter(pl.col("timestamp") <= session),
        curve.filter(pl.col("timestamp") <= visible_through),
    ).filter(pl.col("timestamp") == session)

    assert whole.height == 1 and visible.height == 1
    for column in REGIME_COLUMNS:
        assert whole[column][0] is not None, f"{column} is null at the probe session"
        assert whole[column][0] == pytest.approx(visible[column][0], abs=1e-12, nan_ok=False), (
            f"{column} at {session:%Y-%m-%d} needs a value published that same session: "
            f"{whole[column][0]} with the whole curve against {visible[column][0]} "
            f"with what the loop can see at the close"
        )


def test_the_live_module_takes_its_macro_lag_and_threshold_from_the_case_study_config() -> None:
    """Both constants are declared once, in the case study's own setup, or they drift."""
    import _etfs_features as live
    import yaml

    setup = yaml.safe_load((ROOT / "case_studies" / "etfs" / "config" / "setup.yaml").read_text())
    assert setup["features"]["regime_threshold"] == live.REGIME_THRESHOLD
    assert (
        setup["modeling"]["latent_factors"]["macro_context"]["availability_lag_days"]
        == live.MACRO_LAG_DAYS
    )
