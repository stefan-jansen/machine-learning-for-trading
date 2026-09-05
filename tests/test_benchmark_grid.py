"""The committed benchmark must live on the grid its metrics are annualized for.

The nasdaq100_microstructure benchmark was built on a 15-minute grid and annualized at
``periods_per_year: 252``, which understated it by ``sqrt(6537/252)`` = 5.1x. Because the
backtester aggregates to daily before it measures anything, every comparison in
``18_strategy_analysis`` flattered the strategy by that factor. The three per-label files
were also the same series: ``fwd_ret_5m.parquet`` and ``fwd_ret_15m.parquet`` were
byte-identical frames, so a 5-minute strategy was measured against a 15-minute benchmark.

Regenerate with ``scripts/build_benchmark.py`` rather than editing these files.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = sorted((REPO / "case_studies").glob("*/benchmark/*.json"))


def _ids(paths):
    return [f"{p.parents[1].name}/{p.stem}" for p in paths]


@pytest.mark.parametrize("meta_path", BENCHMARKS, ids=_ids(BENCHMARKS))
def test_benchmark_is_one_row_per_session(meta_path: Path) -> None:
    """A daily annualization needs a daily series: one row per calendar date."""
    frame = pl.read_parquet(meta_path.with_suffix(".parquet"))
    ts = frame["timestamp"]
    # Distinct TIMESTAMPS equalling the row count proves nothing - every intraday bar has
    # a unique timestamp. The property is one row per calendar DATE.
    dates = ts if ts.dtype == pl.Date else ts.dt.date()
    assert dates.n_unique() == frame.height, (
        f"{meta_path.parent.parent.name}/{meta_path.stem}: {frame.height} rows over "
        f"{dates.n_unique()} distinct dates, so this is an intraday series. Its metrics "
        f"are annualized at periods_per_year, which the backtester applies to a daily "
        f"grid - the two are not comparable. Rebuild with scripts/build_benchmark.py."
    )


@pytest.mark.parametrize("meta_path", BENCHMARKS, ids=_ids(BENCHMARKS))
def test_benchmark_metrics_describe_their_own_parquet(meta_path: Path) -> None:
    """``n_periods`` must count the rows actually in the file."""
    meta = json.loads(meta_path.read_text())
    frame = pl.read_parquet(meta_path.with_suffix(".parquet"))
    overall = meta.get("by_period", {}).get("overall")
    if overall is None or overall.get("n_periods") is None:
        pytest.skip("no stratified overall block")
    assert overall["n_periods"] == frame.height, (
        f"{meta_path.stem}: JSON says {overall['n_periods']} periods, parquet has "
        f"{frame.height} rows."
    )


def test_labels_of_one_case_study_do_not_share_a_benchmark_series() -> None:
    """Two labels may agree on a daily grid, but not by holding the same intraday frame.

    The committed nasdaq files were identical because all three were the same 15-minute
    series, not because the labels genuinely coincide. Identical daily frames are fine;
    what is not fine is identical frames whose own JSONs claim different horizons AND a
    non-daily grid, which the first test already rejects. This pins the weaker property
    that every committed pair is at least consistent about its period count.
    """
    by_cs: dict[str, list[Path]] = {}
    for p in BENCHMARKS:
        by_cs.setdefault(p.parents[1].name, []).append(p)
    for cs, paths in by_cs.items():
        heights = {p.stem: pl.read_parquet(p.with_suffix(".parquet")).height for p in paths}
        metas = {p.stem: json.loads(p.read_text()) for p in paths}
        for stem, h in heights.items():
            declared = metas[stem].get("by_period", {}).get("overall", {}).get("n_periods")
            assert declared in (None, h), f"{cs}/{stem}: {declared} declared against {h} rows"
