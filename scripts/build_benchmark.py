#!/usr/bin/env python3
"""Regenerate ``case_studies/<cs>/benchmark/<label>.{parquet,json}``.

These files are the buy-and-hold equal-weight return of the declared universe, and every
strategy in a case study is measured against them through
``case_studies.utils.benchmark.load_benchmark_metrics``. They were committed with no
generator anywhere in the repository (agent-workspace #362), which is how the defects
below survived.

**The benchmark must live on the same grid as the strategy.** The backtester aggregates to
a daily series before it measures anything (``to_daily_returns`` /
``extract_daily_returns_frame``) and ``evaluation.periods_per_year`` annualizes that daily
grid. A benchmark built on the trading cadence and annualized at 252 is not comparable to
it and understates itself by ``sqrt(bars_per_year / periods_per_year)`` - 5.1x on
nasdaq100_microstructure's 15-minute grid, 1.7x on crypto_perps_funding's 8-hour one. So
this compounds to a daily close-to-close return first, which is what the ``method`` field
``daily_cross_sectional_mean_close_pct_change`` has always claimed and what the committed
files did not do.

Usage:
    uv run python scripts/build_benchmark.py --case-study nasdaq100_microstructure
    uv run python scripts/build_benchmark.py --case-study nasdaq100_microstructure --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir

# (loader name in the `data` package, bar frequency to request, loader takes `lazy=`).
# Adding an entry is the whole cost of extending this to a case study; an entry not
# exercised by a regenerated benchmark is not an entry.
_LOADERS: dict[str, tuple[str, str, bool]] = {
    "nasdaq100_microstructure": ("load_nasdaq100_bars", "1m", True),
    "crypto_perps_funding": ("load_crypto_perps", "8h", False),
}

# Case studies whose bars are RAW, so a close-to-close return spans corporate actions.
# AlgoSeek's NASDAQ-100 minute archive carries `last_trade_price` and no adjustment field
# in any of its 63 columns: AAPL closes 499.23 on 2020-08-28 and 129.04 on 2020-08-31, and
# TSLA splits 5:1 three days later, both inside the validation window.
# `load_sp500_daily_bars` publishes a cumulative `adj_factor`, and `close * adj_factor` is
# continuous through the split (AAPL 4044.2 -> 4181.4, the +3.4% it actually traded).
#
# Crypto perpetuals have no corporate actions, so they are absent here by fact rather than
# by omission - and the residual screen below is gated on this mapping for that reason.
_ADJUSTMENT_SOURCE: dict[str, str] = {
    "nasdaq100_microstructure": "load_sp500_daily_bars",
}

# Ratios a residual split shows up as, for symbols the adjustment source does not cover.
# Matching a RATIO rather than thresholding a return is what keeps a genuine -40% session
# in the cross-section: only a move within `_SPLIT_TOLERANCE` of one of these is dropped.
_SPLIT_RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0)
_SPLIT_TOLERANCE = 0.02


def _daily_closes(case_study: str, symbols: list[str], start: str, end: str) -> pl.DataFrame:
    """Last traded close per symbol per session, over ``[start, end]``."""
    if case_study not in _LOADERS:
        raise NotImplementedError(
            f"{case_study} has no loader registered in _LOADERS. Add the loader that "
            f"reads its price panel, then regenerate and commit its benchmark files - do "
            f"not leave the committed ones in place, they have no generator behind them."
        )
    import data as _data

    name, frequency, takes_lazy = _LOADERS[case_study]
    kwargs: dict = dict(frequency=frequency, symbols=symbols, start_date=start, end_date=end)
    if takes_lazy:
        kwargs["lazy"] = True
    frame = getattr(_data, name)(**kwargs)
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    return (
        lf.select(["timestamp", "symbol", "close"])
        .with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by(["symbol", "date"])
        .agg(pl.col("close").sort_by("timestamp").last())
        .collect(engine="streaming")
    )


def _apply_adjustment(case_study: str, closes: pl.DataFrame, symbols: list[str]) -> pl.DataFrame:
    """Attach ``adj_close`` and the ``adj_factor`` that produced it, where one exists."""
    if case_study not in _ADJUSTMENT_SOURCE:
        return closes.with_columns(
            pl.col("close").alias("adj_close"),
            pl.lit(None, dtype=pl.Float64).alias("adj_factor"),
        )

    import data as _data

    factors = (
        getattr(_data, _ADJUSTMENT_SOURCE[case_study])(
            start_date=str(closes["date"].min()),
            end_date=str(closes["date"].max()),
            symbols=symbols,
        )
        .select(
            pl.col("timestamp").cast(pl.Date).alias("date"),
            "symbol",
            pl.col("adj_factor").cast(pl.Float64),
        )
        .unique(subset=["symbol", "date"], keep="first")
    )
    print(
        f"  adjustment: {factors['symbol'].n_unique()} of {closes['symbol'].n_unique()} "
        f"symbols carry an adj_factor; the rest are screened for residual split ratios"
    )
    return closes.join(factors, on=["symbol", "date"], how="left").with_columns(
        pl.when(pl.col("adj_factor").is_not_null())
        .then(pl.col("close") * pl.col("adj_factor"))
        .otherwise(pl.col("close"))
        .alias("adj_close")
    )


def _drop_residual_splits(per_symbol: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Null out returns whose raw price ratio is a split ratio, for unadjusted pairs.

    A pair with a factor at both ends is already continuous and is never screened.
    """
    ratio = pl.col("prev_close") / pl.col("close")
    is_split = pl.lit(False)
    for r in _SPLIT_RATIOS:
        for cand in (r, 1.0 / r):
            is_split = is_split | ((ratio / cand - 1.0).abs() < _SPLIT_TOLERANCE)
    flagged = per_symbol.with_columns((is_split & pl.col("unadjusted_pair")).alias("split"))
    return flagged.filter(~pl.col("split")), int(flagged["split"].sum())


def _equal_weight_daily(closes: pl.DataFrame, screen_splits: bool = True) -> pl.DataFrame:
    """Cross-sectional equal-weight mean of each symbol's daily close-to-close return.

    A symbol contributes on a session only when it has a close on that session AND on the
    one before it, so a listing or a delisting never enters as a return.

    **Both endpoints must be on the same price scale.** Falling back to the raw close
    per row divides an adjusted price by an unadjusted one wherever factor coverage
    starts, ends or has a hole: unchanged closes of 100 with factors [8, null, 8] read as
    -87.5% then +700%, and the residual screen cannot see it because the RAW ratio is
    exactly 1. A pair straddling a coverage boundary is dropped, not approximated.
    """
    covered = pl.col("adj_factor").is_not_null()
    per_symbol = (
        closes.sort(["symbol", "date"])
        .with_columns(
            pl.col("adj_close").shift(1).over("symbol").alias("prev_adj"),
            pl.col("close").shift(1).over("symbol").alias("prev_close"),
            covered.alias("covered"),
            covered.shift(1).over("symbol").alias("prev_covered"),
        )
        .with_columns(
            pl.when(pl.col("covered") & pl.col("prev_covered"))
            .then(pl.col("adj_close") / pl.col("prev_adj") - 1.0)
            .when(~pl.col("covered") & ~pl.col("prev_covered"))
            .then(pl.col("close") / pl.col("prev_close") - 1.0)
            .otherwise(None)
            .alias("ret"),
            (~pl.col("covered") & ~pl.col("prev_covered")).alias("unadjusted_pair"),
        )
        .drop_nulls("ret")
    )
    n_split = 0
    if screen_splits:
        per_symbol, n_split = _drop_residual_splits(per_symbol)
    if n_split:
        print(f"  dropped {n_split} symbol-session(s) whose price ratio is a split ratio")
    return (
        per_symbol.group_by("date")
        .agg(pl.col("ret").mean().alias("ew_return"), pl.len().alias("n_symbols"))
        .sort("date")
    )


def _metrics(returns: np.ndarray, periods_per_year: int) -> dict:
    if returns.size < 2:
        return {"sharpe": None, "cagr": None, "vol": None, "n_periods": int(returns.size)}
    sd = float(np.std(returns, ddof=1))
    cum = float(np.prod(1.0 + returns) - 1.0)
    return {
        "sharpe": round(float(np.mean(returns) / sd * np.sqrt(periods_per_year)), 6)
        if sd > 0
        else None,
        "cagr": round(float((1.0 + cum) ** (periods_per_year / returns.size) - 1.0), 6),
        "vol": round(sd * float(np.sqrt(periods_per_year)), 6),
        "n_periods": int(returns.size),
    }


def _evaluation_start(case_study: str, label_path: Path):
    """First session the strategy is measured over, from the splitter itself.

    ``WalkForwardCV`` builds folds BACKWARD from the holdout boundary using ``n_splits``
    and ``val_size``, so advancing the label's first session by ``train_size`` lands in
    the wrong place - on nasdaq it gives 2020-07-02 against the splitter's 2020-06-30,
    silently dropping two evaluated sessions.
    """
    folds = generate_cv_splits(
        pl.scan_parquet(label_path).select("timestamp").unique().collect(),
        case_study_id=case_study,
        date_col="timestamp",
    )
    first = min(f["val_start"] for f in folds)
    return first.date() if hasattr(first, "date") else first


def build(case_study: str, check: bool) -> int:
    case_dir = get_case_study_dir(case_study)
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    symbols = list(setup["universe"]["symbols"])
    ppy = int(setup["evaluation"]["periods_per_year"])
    holdout_start = pl.Series([str(setup["evaluation"]["holdout_start"])[:10]]).str.to_date()[0]
    holdout_end = pl.Series([str(setup["evaluation"]["holdout_end"])[:10]]).str.to_date()[0]

    labels_dir, bench_dir = case_dir / "labels", case_dir / "benchmark"
    label_files = sorted(labels_dir.glob("*.parquet"))
    if not label_files:
        print(f"{case_study}: no label parquets under {labels_dir}", file=sys.stderr)
        return 1

    spans = {}
    for lf in label_files:
        hi = (
            pl.scan_parquet(lf)
            .select(pl.col("timestamp").dt.date().max().alias("hi"))
            .collect()["hi"][0]
        )
        spans[lf.stem] = (_evaluation_start(case_study, lf), hi)

    lo = min(v[0] for v in spans.values())
    hi = max(v[1] for v in spans.values())
    # A year of run-up so the first evaluated session has a prior close to price against.
    closes = _daily_closes(case_study, symbols, str(lo.replace(year=lo.year - 1)), str(hi))
    closes = _apply_adjustment(case_study, closes, symbols)
    ew_all = _equal_weight_daily(closes, screen_splits=case_study in _ADJUSTMENT_SOURCE)

    failures = 0
    for label, (l_lo, l_hi) in sorted(spans.items()):
        ew = ew_all.filter((pl.col("date") >= l_lo) & (pl.col("date") <= l_hi))
        is_hold = (ew["date"] >= holdout_start) & (ew["date"] <= holdout_end)
        r = ew["ew_return"].to_numpy()
        val_rows, hold_rows = ew.filter(~is_hold), ew.filter(is_hold)
        validation = _metrics(val_rows["ew_return"].to_numpy(), ppy)

        meta = {
            "case_study": case_study,
            "label": label,
            "method": "daily_cross_sectional_mean_close_pct_change",
            "periods_per_year": ppy,
            "n_symbols_declared": len(symbols),
            "n_symbols_in_universe": int(
                closes.filter(
                    (pl.col("date") >= ew["date"].min()) & (pl.col("date") <= ew["date"].max())
                )["symbol"].n_unique()
            ),
            **validation,
            "ts_min": str(ew["date"].min()),
            "ts_max": str(ew["date"].max()),
            "by_period": {
                "overall": _metrics(r, ppy),
                "validation": validation,
                "holdout": _metrics(hold_rows["ew_return"].to_numpy(), ppy),
                "validation_window": [str(val_rows["date"].min()), str(val_rows["date"].max())],
                "holdout_window": [str(hold_rows["date"].min()), str(hold_rows["date"].max())],
            },
        }
        pq, js = bench_dir / f"{label}.parquet", bench_dir / f"{label}.json"
        frame = ew.select(pl.col("date").alias("timestamp"), "ew_return")

        if check:
            # Compare EVERYTHING a consumer reads. Checking only the overall block lets a
            # moved holdout boundary report MATCHES while the validation and holdout
            # blocks load_benchmark_metrics actually returns are stale.
            was = json.loads(js.read_text()) if js.exists() else {}
            diffs = [k for k in meta if k != "by_period" and was.get(k) != meta[k]]
            diffs += [
                f"by_period.{k}"
                for k in meta["by_period"]
                if was.get("by_period", {}).get(k) != meta["by_period"][k]
            ]
            # "parquet" is not a metadata key, so it is kept out of the lookup below
            # entirely rather than appended to `diffs`, where meta[k] would raise
            # KeyError and abort every remaining label.
            parquet_note = None
            if not pq.exists():
                parquet_note = f"  parquet: missing, would write {frame.height} rows"
            elif not pl.read_parquet(pq).equals(frame):
                parquet_note = (
                    f"  parquet: committed {pl.read_parquet(pq).height} rows against {frame.height}"
                )
            print(f"{label}: {'DIFFERS' if diffs or parquet_note else 'MATCHES'} committed")
            for k in diffs:
                failures += 1
                if k.startswith("by_period."):
                    inner = k.split(".", 1)[1]
                    print(
                        f"  {k}: {was.get('by_period', {}).get(inner)} -> {meta['by_period'][inner]}"
                    )
                else:
                    print(f"  {k}: {was.get(k)} -> {meta[k]}")
            if parquet_note:
                failures += 1
                print(parquet_note)
            continue

        bench_dir.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(pq)
        js.write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"{label}: {frame.height} sessions, {meta['n_symbols_in_universe']} symbols, "
            f"overall {meta['by_period']['overall']['sharpe']}, "
            f"validation {validation['sharpe']}, "
            f"holdout {meta['by_period']['holdout']['sharpe']}"
        )
    return 1 if (check and failures) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-study", required=True)
    ap.add_argument("--check", action="store_true", help="compare with the committed files")
    args = ap.parse_args()
    raise SystemExit(build(args.case_study, args.check))
