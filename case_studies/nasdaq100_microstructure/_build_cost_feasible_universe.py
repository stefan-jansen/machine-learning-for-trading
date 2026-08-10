"""Provenance: builder for the frozen, per-split cost-feasible universe.

Run from the repo root (``uv run python
case_studies/nasdaq100_microstructure/_build_cost_feasible_universe.py``).

The canonical lists live in ``config/setup.yaml::universe.cost_feasible.
{validation,holdout}`` and are consumed by the backtest pipeline via
``strategy.signal.universe_filter='cost_feasible'`` (see
``case_studies/utils/backtest_runner.py::_apply_cost_feasible_filter``). This
script documents HOW those lists were produced so they can be regenerated.

The universe is a frozen snapshot per split (profiled with no look-ahead),
NOT full-sample and NOT per-rebalance rolling:

  - validation universe := top-50 by round-trip-cost proxy, computed on quote
    bars STRICTLY BEFORE the validation window start (2020-01-01 -> 2020-06-30,
    the first expanding-window training block). No validation-period liquidity
    leaks into the universe pick.
  - holdout universe    := top-50 from the pre-holdout liquidity_profile.parquet,
    which 01_feasibility_analysis.py restricts to < HOLDOUT_START (2021-07-01).
    Causal at the holdout boundary.

Round-trip cost proxy: 2*(per_share/mean_price)*1e4 + 2*median_half_spread_bps.
Prints the val/holdout overlap so stability can be documented.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

from data import load_nasdaq100_bars
from utils.paths import get_case_study_dir

CS = "nasdaq100_microstructure"
CASE_DIR = get_case_study_dir(CS)
SETUP = yaml.safe_load((CASE_DIR / "config/setup.yaml").open())
PER_SHARE_USD = float(SETUP["costs"]["per_share"])
UNIVERSE = sorted(SETUP["universe"]["symbols"])

START_DATE = "2020-01-01"
VALIDATION_START = "2020-06-30"
HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])  # 2021-07-01

TOP_N = 50
OUT = Path(__file__).parent


def build_profile(start: str, end: str) -> pl.DataFrame:
    qb = load_nasdaq100_bars(start_date=start, end_date=end, include_quotes=True, symbols=UNIVERSE)
    qb = (
        qb.with_columns(
            mid=(pl.col("bid_close") + pl.col("ask_close")) / 2,
            raw_spread=pl.col("ask_close") - pl.col("bid_close"),
        )
        .filter(
            pl.col("bid_close").is_not_null()
            & pl.col("ask_close").is_not_null()
            & (pl.col("bid_close") > 0)
            & (pl.col("ask_close") >= pl.col("bid_close"))
        )
        .with_columns(half_spread_bps=(pl.col("raw_spread") / 2 / pl.col("mid") * 1e4))
    )
    return (
        qb.group_by("symbol")
        .agg(
            n_bars=pl.len(),
            median_half_spread_bps=pl.col("half_spread_bps").median(),
            mean_price=pl.col("close").mean(),
        )
        .with_columns(
            rt_cost_bps=2 * (PER_SHARE_USD / pl.col("mean_price")) * 1e4
            + 2 * pl.col("median_half_spread_bps"),
        )
        .sort("rt_cost_bps")
    )


def top50_from_existing() -> list[str]:
    lp = pl.read_parquet(CASE_DIR / "liquidity_profile.parquet").select(
        ["symbol", "median_half_spread_bps", "mean_price"]
    )
    lp = lp.with_columns(
        rt_cost_bps=2 * (PER_SHARE_USD / pl.col("mean_price")) * 1e4
        + 2 * pl.col("median_half_spread_bps")
    )
    return lp.sort("rt_cost_bps").head(TOP_N)["symbol"].to_list()


def main() -> None:
    print(f"per_share=${PER_SHARE_USD}  top_n={TOP_N}", flush=True)
    val_prof = build_profile(START_DATE, VALIDATION_START)
    val_univ = val_prof.head(TOP_N)["symbol"].to_list()
    ho_univ = top50_from_existing()
    val_set, ho_set = set(val_univ), set(ho_univ)
    print(f"OVERLAP: {len(val_set & ho_set)}/{TOP_N} symbols common", flush=True)
    (OUT / "universe_validation.json").write_text(json.dumps(val_univ, indent=2))
    (OUT / "universe_holdout.json").write_text(json.dumps(ho_univ, indent=2))
    print(
        "Wrote universe_{validation,holdout}.json — copy into "
        "config/setup.yaml::universe.cost_feasible",
        flush=True,
    )


if __name__ == "__main__":
    main()
