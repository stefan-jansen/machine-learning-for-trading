#!/usr/bin/env python3
"""Regenerate ``case_studies/<cs>/benchmark/<label>.{parquet,json}``.

These files are the buy-and-hold equal-weight return of the declared universe, and
every strategy in a case study is measured against them through
``case_studies.utils.benchmark.load_benchmark_metrics``. They were committed with no
generator (agent-workspace #362), which is how the three defects below survived.

**The benchmark must live on the same grid as the strategy.** The backtester aggregates
to a daily series before it measures anything - ``to_daily_returns`` /
``extract_daily_returns_frame`` - and ``evaluation.periods_per_year`` annualizes that
daily grid. A benchmark built on the trading cadence and annualized at 252 is not
comparable to it, and understates itself by ``sqrt(bars_per_year / 252)``. On
nasdaq100_microstructure that factor was 5.1: a 15-minute series annualized at 252 stored
0.2957 where the same series correctly annualized reads 1.5063. Every comparison in
``18_strategy_analysis`` was flattering the strategy by that factor.

So this compounds each symbol's intraday returns to a daily close-to-close return first,
then takes the cross-sectional equal-weight mean, which is what the ``method`` field
``daily_cross_sectional_mean_close_pct_change`` has always claimed and what the committed
files did not do.

Two further defects the committed nasdaq files carried, both fixed by regenerating:

- The three per-label files were the same series. ``fwd_ret_5m.parquet`` and
  ``fwd_ret_15m.parquet`` were byte-identical frames on a 15-minute grid; ``fwd_ret_60m``
  was the same series with one extra leading bar. A 5-minute strategy was being compared
  against a 15-minute benchmark. On a daily grid the labels legitimately share a series,
  and they now differ only where their decision times cover different sessions.
- ``n_symbols_in_universe`` recorded 111 against a declared universe of 115, with no
  record of which 111. It is now counted from the symbols that actually contribute.

Usage:
    uv run python scripts/build_benchmark.py --case-study nasdaq100_microstructure
    uv run python scripts/build_benchmark.py --case-study nasdaq100_microstructure --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir

# Each case study reads its prices through its own loader. A name here is a statement that
# the daily close-to-close return of these symbols is what the case study's strategies are
# measured against. Adding one is the whole cost of extending this script to a case study;
# an entry that is not exercised by a regenerated benchmark is not an entry.
# (loader name in the `data` package, bar frequency to request, loader takes `lazy=`)
_LOADERS: dict[str, tuple[str, str, bool]] = {
    "nasdaq100_microstructure": ("load_nasdaq100_bars", "1m", True),
    "crypto_perps_funding": ("load_crypto_perps", "8h", False),
}

# Case studies whose bars are RAW, so a close-to-close return spans corporate actions.
# AlgoSeek's NASDAQ-100 minute archive carries `last_trade_price` and no adjustment
# column at all - AAPL closes 499.23 on 2020-08-28 and 129.04 on 2020-08-31, which a
# naive daily return reads as -74%. Both dates are inside the validation window, and
# TSLA's 5:1 split is three days later. `load_sp500_daily_bars` publishes a cumulative
# `adj_factor` for the names it covers; `close * adj_factor` is continuous across the
# split (AAPL: 4044.2 -> 4181.4, the +3.4% it actually traded).
#
# Crypto perpetuals have no corporate actions, so they are absent here by fact rather
# than by omission.
_ADJUSTMENT_SOURCE: dict[str, str] = {
    "nasdaq100_microstructure": "load_sp500_daily_bars",
}

# Ratios a residual split shows up as, for symbols the adjustment source does not cover.
# Matching a RATIO rather than thresholding a return is what keeps a genuine -40% session
# in the cross-section: only a move within `_SPLIT_TOLERANCE` of one of these is dropped.
_SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 1.5, 2.5, 3.0 / 2)
_SPLIT_TOLERANCE = 0.02


def _daily_closes(case_study: str, symbols: list[str], start: str, end: str) -> pl.DataFrame:
    """Last traded close per symbol per session, over ``[start, end]``."""
    if case_study not in _LOADERS:
        raise NotImplementedError(
            f"{case_study} has no loader registered in _LOADERS. Add the loader that reads "
            f"its price panel, then regenerate and commit its benchmark files - do not "
            f"leave the committed ones in place, they have no generator behind them."
        )
    import data as _data

    name, frequency, takes_lazy = _LOADERS[case_study]
    kwargs = dict(frequency=frequency, symbols=symbols, start_date=start, end_date=end)
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
    """Return ``closes`` with an ``adj_close`` column that is continuous across splits."""
    if case_study not in _ADJUSTMENT_SOURCE:
        return closes.with_columns(pl.col("close").alias("adj_close"))

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
    covered = factors["symbol"].n_unique()
    print(
        f"  adjustment: {covered} of {closes['symbol'].n_unique()} symbols carry an "
        f"adj_factor; the rest are screened for residual split ratios"
    )
    return closes.join(factors, on=["symbol", "date"], how="left").with_columns(
        pl.when(pl.col("adj_factor").is_not_null())
        .then(pl.col("close") * pl.col("adj_factor"))
        .otherwise(pl.col("close"))
        .alias("adj_close")
    )


def _drop_residual_splits(per_symbol: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Null out returns whose price ratio is a split ratio, for unadjusted symbols.

    A symbol with an ``adj_factor`` is already continuous and is never screened. For the
    rest, a ratio within ``_SPLIT_TOLERANCE`` of a known split ratio (either direction) is
    a corporate action rather than a move, so the symbol leaves the cross-section for that
    one session instead of entering it as a -75% return.
    """
    ratio = pl.col("prev_close") / pl.col("close")
    is_split = pl.lit(False)
    for r in _SPLIT_RATIOS:
        for cand in (r, 1.0 / r):
            is_split = is_split | ((ratio / cand - 1.0).abs() < _SPLIT_TOLERANCE)
    flagged = per_symbol.with_columns(
        (is_split & pl.col("adj_factor_missing")).alias("residual_split")
    )
    n = int(flagged["residual_split"].sum())
    return flagged.filter(~pl.col("residual_split")), n


def _equal_weight_daily(closes: pl.DataFrame, screen_splits: bool = True) -> pl.DataFrame:
    """Cross-sectional equal-weight mean of each symbol's daily close-to-close return.

    A symbol contributes on a session only when it has a close on that session AND on the
    one before it, so a listing or a delisting never enters as a return.
    """
    has_factor = "adj_factor" in closes.columns
    per_symbol = (
        closes.sort(["symbol", "date"])
        .with_columns(
            pl.col("adj_close").shift(1).over("symbol").alias("prev_adj"),
            pl.col("close").shift(1).over("symbol").alias("prev_close"),
            (pl.col("adj_factor").is_null() if has_factor else pl.lit(True)).alias(
                "adj_factor_missing"
            ),
        )
        .with_columns((pl.col("adj_close") / pl.col("prev_adj") - 1.0).alias("ret"))
        .drop_nulls("ret")
    )
    n_split = 0
    if screen_splits:
        per_symbol, n_split = _drop_residual_splits(per_symbol)
    if n_split:
        print(f"  dropped {n_split} symbol-session(s) whose price ratio is a split ratio")
    return (
        per_symbol.group_by("date")
        .agg(
            pl.col("ret").mean().alias("ew_return"),
            pl.len().alias("n_symbols"),
        )
        .sort("date")
    )


def _apply_adjustment(case_study: str, closes: pl.DataFrame, symbols: list[str]) -> pl.DataFrame:
    """Return ``closes`` with an ``adj_close`` column that is continuous across splits."""
    if case_study not in _ADJUSTMENT_SOURCE:
        return closes.with_columns(pl.col("close").alias("adj_close"))

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
    covered = factors["symbol"].n_unique()
    print(
        f"  adjustment: {covered} of {closes['symbol'].n_unique()} symbols carry an "
        f"adj_factor; the rest are screened for residual split ratios"
    )
    return closes.join(factors, on=["symbol", "date"], how="left").with_columns(
        pl.when(pl.col("adj_factor").is_not_null())
        .then(pl.col("close") * pl.col("adj_factor"))
        .otherwise(pl.col("close"))
        .alias("adj_close")
    )


def _drop_residual_splits(per_symbol: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Null out returns whose price ratio is a split ratio, for unadjusted symbols.

    A symbol with an ``adj_factor`` is already continuous and is never screened. For the
    rest, a ratio within ``_SPLIT_TOLERANCE`` of a known split ratio (either direction) is
    a corporate action rather than a move, so the symbol leaves the cross-section for that
    one session instead of entering it as a -75% return.
    """
    ratio = pl.col("prev_close") / pl.col("close")
    is_split = pl.lit(False)
    for r in _SPLIT_RATIOS:
        for cand in (r, 1.0 / r):
            is_split = is_split | ((ratio / cand - 1.0).abs() < _SPLIT_TOLERANCE)
    flagged = per_symbol.with_columns(
        (is_split & pl.col("adj_factor_missing")).alias("residual_split")
    )
    n = int(flagged["residual_split"].sum())
    return flagged.filter(~pl.col("residual_split")), n


def _equal_weight_daily(closes: pl.DataFrame, screen_splits: bool = True) -> pl.DataFrame:
    """Cross-sectional equal-weight mean of each symbol's daily close-to-close return.

    A symbol contributes on a session only when it has a close on that session AND on the
    one before it, so a listing or a delisting never enters as a return.
    """
    has_factor = "adj_factor" in closes.columns
    per_symbol = (
        closes.sort(["symbol", "date"])
        .with_columns(
            pl.col("adj_close").shift(1).over("symbol").alias("prev_adj"),
            pl.col("close").shift(1).over("symbol").alias("prev_close"),
            (pl.col("adj_factor").is_null() if has_factor else pl.lit(True)).alias(
                "adj_factor_missing"
            ),
        )
        .with_columns((pl.col("adj_close") / pl.col("prev_adj") - 1.0).alias("ret"))
        .drop_nulls("ret")
    )
    n_split = 0
    if screen_splits:
        per_symbol, n_split = _drop_residual_splits(per_symbol)
    if n_split:
        print(f"  dropped {n_split} symbol-session(s) whose price ratio is a split ratio")
    return (
        per_symbol.group_by("date")
        .agg(
            pl.col("ret").mean().alias("ew_return"),
            pl.len().alias("n_symbols"),
        )
        .sort("date")
    )


def _months(size: str) -> int:
    """``"6M"`` -> 6, ``"2Y"`` -> 24. The declared train/val sizes are calendar spans."""
    m = re.fullmatch(r"(\d+)\s*([MY])", str(size).strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot read a calendar span from evaluation size {size!r}")
    return int(m.group(1)) * (12 if m.group(2).upper() == "Y" else 1)


def _advance_months(d: _date, months: int) -> _date:
    y, m = divmod((d.year * 12 + d.month - 1) + months, 12)
    return _date(
        y,
        m + 1,
        min(
            d.day,
            [
                31,
                29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28,
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ][m],
        ),
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


def build(case_study: str, check: bool) -> int:
    case_dir = get_case_study_dir(case_study)
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    symbols = list(setup["universe"]["symbols"])
    ppy = int(setup["evaluation"]["periods_per_year"])
    holdout_start = _date.fromisoformat(str(setup["evaluation"]["holdout_start"])[:10])
    holdout_end = _date.fromisoformat(str(setup["evaluation"]["holdout_end"])[:10])

    labels_dir, bench_dir = case_dir / "labels", case_dir / "benchmark"
    label_files = sorted(labels_dir.glob("*.parquet"))
    if not label_files:
        print(f"{case_study}: no label parquets under {labels_dir}", file=sys.stderr)
        return 1

    spans = {}
    for lf in label_files:
        ts = (
            pl.scan_parquet(lf)
            .select(pl.col("timestamp").dt.date().alias("d"))
            .select(pl.col("d").min().alias("lo"), pl.col("d").max().alias("hi"))
            .collect()
        )
        spans[lf.stem] = (ts["lo"][0], ts["hi"][0])

    lo = min(v[0] for v in spans.values())
    hi = max(v[1] for v in spans.values())
    # One extra session of run-up so the first in-span date has a return.
    closes = _daily_closes(case_study, symbols, str(lo.replace(year=lo.year - 1)), str(hi))
    closes = _apply_adjustment(case_study, closes, symbols)
    # The residual-split screen belongs only where corporate actions exist. A ratio near
    # 1.5 or 2.0 is a split in an equity panel and an ordinary session in crypto, which
    # has no splits at all: screening crypto dropped 11 genuine moves.
    ew_all = _equal_weight_daily(closes, screen_splits=case_study in _ADJUSTMENT_SOURCE)

    failures = 0
    for label, (l_lo, l_hi) in sorted(spans.items()):
        # The benchmark covers what the strategy is measured over, not the whole panel:
        # the training prefix is never traded, and including it would compare a strategy's
        # evaluation-period Sharpe against a benchmark carrying its training period too.
        #
        # Take the boundary from the splitter rather than deriving it. WalkForwardCV
        # builds folds BACKWARD from the holdout boundary using n_splits and val_size, so
        # advancing the label's first session by train_size lands in the wrong place -
        # on nasdaq it gives 2020-07-02 against the splitter's 2020-06-30, silently
        # dropping two evaluated sessions.
        eval_from = min(
            f["val_start"]
            for f in generate_cv_splits(
                pl.scan_parquet(labels_dir / f"{label}.parquet")
                .select("timestamp")
                .unique()
                .collect(),
                case_study_id=case_study,
                date_col="timestamp",
            )
        )
        eval_from = eval_from.date() if hasattr(eval_from, "date") else eval_from
        ew = ew_all.filter((pl.col("date") >= eval_from) & (pl.col("date") <= l_hi))
        r = ew["ew_return"].to_numpy()
        is_hold = (ew["date"] >= holdout_start) & (ew["date"] <= holdout_end)
        hold = ew.filter(is_hold)["ew_return"].to_numpy()
        val = ew.filter(~is_hold)["ew_return"].to_numpy()

        overall = _metrics(r, ppy)
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
            **_metrics(val, ppy),
            "ts_min": str(ew["date"].min()),
            "ts_max": str(ew["date"].max()),
            "by_period": {
                "overall": overall,
                "validation": _metrics(val, ppy),
                "holdout": _metrics(hold, ppy),
                "validation_window": [
                    str(ew.filter(~is_hold)["date"].min()),
                    str(ew.filter(~is_hold)["date"].max()),
                ],
                "holdout_window": [
                    str(ew.filter(is_hold)["date"].min()),
                    str(ew.filter(is_hold)["date"].max()),
                ],
            },
        }

        pq, js = bench_dir / f"{label}.parquet", bench_dir / f"{label}.json"
        frame = ew.select(pl.col("date").alias("timestamp"), "ew_return")
        if check:
            # Compare EVERYTHING a consumer reads. Checking only the overall block lets a
            # moved holdout boundary report MATCHES while the validation and holdout
            # blocks that load_benchmark_metrics actually returns are stale.
            was = json.loads(js.read_text()) if js.exists() else {}
            diffs = [k for k in meta if k != "by_period" and was.get(k) != meta[k]]
            diffs += [
                f"by_period.{k}"
                for k in meta["by_period"]
                if was.get("by_period", {}).get(k) != meta["by_period"][k]
            ]
            if not pq.exists():
                diffs.append("parquet missing")
            elif not pl.read_parquet(pq).equals(frame):
                diffs.append("parquet")
            print(f"{label}: {'MATCHES' if not diffs else 'DIFFERS'} committed")
            for k in diffs:
                failures += 1
                if k == "parquet":
                    print(
                        f"  parquet: committed {pl.read_parquet(pq).height} rows against {frame.height}"
                    )
                else:
                    was_v = (
                        was.get("by_period", {}).get(k.split(".", 1)[1])
                        if k.startswith("by_period.")
                        else was.get(k)
                    )
                    new_v = (
                        meta["by_period"][k.split(".", 1)[1]]
                        if k.startswith("by_period.")
                        else meta[k]
                    )
                    print(f"  {k}: {was_v} -> {new_v}")
            continue

        bench_dir.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(pq)
        js.write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"{label}: {frame.height} sessions, {meta['n_symbols_in_universe']} symbols, "
            f"overall sharpe {overall['sharpe']}, validation {meta['by_period']['validation']['sharpe']}, "
            f"holdout {meta['by_period']['holdout']['sharpe']}"
        )
    return 1 if (check and failures) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-study", required=True)
    ap.add_argument(
        "--check", action="store_true", help="compare against the committed files, write nothing"
    )
    args = ap.parse_args()
    raise SystemExit(build(args.case_study, args.check))
