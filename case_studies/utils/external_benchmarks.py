"""External benchmark loaders for strategy-analysis notebooks.

Provides SPY (broad-equity ETF) and FF-market (Ken French Mkt-RF + RF)
return series on the canonical schema (`timestamp`, `benchmark_return`).
Both flavors return polars DataFrames; alignment with strategy returns
is the caller's responsibility.

The diagnostic helper `compute_benchmark_diagnostics` returns the
information ratio, beta, correlation, and tracking error of a strategy
return series against a benchmark return series.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import polars as pl

from data import load_etfs, load_ff_factors


def _date_or_none(value: str | _dt.date | _dt.datetime | None) -> _dt.date | None:
    if value is None:
        return None
    if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.datetime):
        return value.date()
    return _dt.date.fromisoformat(str(value)[:10])


def load_spy_returns(
    start: str | _dt.date | None = None,
    end: str | _dt.date | None = None,
) -> pl.DataFrame:
    """SPY daily total returns (close-to-close pct change).

    Returns columns: `timestamp` (Date), `benchmark_return` (Float64).
    """
    df = load_etfs(symbols=["SPY"]).sort("timestamp").select("timestamp", "close")
    df = (
        df.with_columns(
            benchmark_return=pl.col("close").pct_change(),
        )
        .drop("close")
        .drop_nulls("benchmark_return")
    )
    s, e = _date_or_none(start), _date_or_none(end)
    if s is not None:
        df = df.filter(pl.col("timestamp") >= s)
    if e is not None:
        df = df.filter(pl.col("timestamp") <= e)
    return df


def load_ff_market_returns(
    start: str | _dt.date | None = None,
    end: str | _dt.date | None = None,
    frequency: str = "daily",
) -> pl.DataFrame:
    """Fama-French market return (Mkt-RF + RF, i.e. nominal market).

    `frequency` is `"daily"` or `"monthly"`. Returns columns:
    `timestamp` (Date), `benchmark_return` (Float64).
    """
    df = load_ff_factors(dataset="ff5", frequency=frequency).sort("timestamp")
    df = df.with_columns(
        benchmark_return=pl.col("Mkt-RF") + pl.col("RF"),
    ).select("timestamp", "benchmark_return")
    s, e = _date_or_none(start), _date_or_none(end)
    if s is not None:
        df = df.filter(pl.col("timestamp") >= s)
    if e is not None:
        df = df.filter(pl.col("timestamp") <= e)
    return df


def align_to_strategy(
    strategy_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    timestamp_col: str = "ts",
    strategy_col: str = "strategy",
    benchmark_col: str = "benchmark_return",
) -> pl.DataFrame:
    """Inner-join a benchmark series onto strategy returns.

    Both inputs must carry the canonical timestamp column (Date). The
    result has columns `[timestamp_col, strategy_col, benchmark_col]`.
    Caller-side alignment for monthly cadences should resample first.
    """
    bench = benchmark_df.with_columns(
        pl.col("timestamp").cast(pl.Date).alias(timestamp_col)
    ).select(timestamp_col, benchmark_col)
    return strategy_df.join(bench, on=timestamp_col, how="inner").sort(timestamp_col)


def align_to_strategy_monthly(
    strategy_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    timestamp_col: str = "ts",
    strategy_col: str = "strategy",
    benchmark_col: str = "benchmark_return",
) -> pl.DataFrame:
    """Month-anchored inner join (FF monthly is first-of-month; benchmark
    series often end-of-month). Joins on (year, month) key.
    """
    s = strategy_df.with_columns(
        _y=pl.col(timestamp_col).dt.year(),
        _m=pl.col(timestamp_col).dt.month(),
    )
    b = benchmark_df.with_columns(
        _y=pl.col("timestamp").dt.year(),
        _m=pl.col("timestamp").dt.month(),
    ).select("_y", "_m", benchmark_col)
    return s.join(b, on=["_y", "_m"], how="inner").drop("_y", "_m").sort(timestamp_col)


def compute_subperiod_diagnostics(
    df: pl.DataFrame,
    buckets: list[tuple[str, _dt.date | str, _dt.date | str]],
    *,
    timestamp_col: str = "ts",
    strategy_col: str = "strategy",
    benchmark_col: str | None = "benchmark",
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Per-bucket diagnostics for sub-period decomposition tables.

    `df` must carry an aligned `(timestamp, strategy[, benchmark])` triple.
    `buckets` is a list of `(label, start_date, end_date)` triples; date
    bounds are inclusive at both ends. Per-bucket metrics: `n`,
    `cum_return`, `ann_return`, `sharpe`, `max_drawdown`. When
    `benchmark_col` is provided, the table also includes `beta_vs_bm` and
    `info_ratio_vs_bm`.
    """
    rows: list[dict[str, object]] = []
    for label, start, end in buckets:
        s = _date_or_none(start)
        e = _date_or_none(end)
        sub = df.filter((pl.col(timestamp_col) >= s) & (pl.col(timestamp_col) <= e))
        n = sub.height
        if n == 0:
            rows.append({"bucket": label, "n": 0})
            continue
        s_arr = sub[strategy_col].to_numpy()
        cum = float(np.prod(1 + s_arr) - 1)
        if periods_per_year > 0:
            ann = (1 + cum) ** (periods_per_year / max(n, 1)) - 1
        else:
            ann = float("nan")
        std = float(np.std(s_arr, ddof=1)) if n > 1 else 0.0
        sharpe = (
            float(np.mean(s_arr) / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
        )
        equity = np.cumprod(1 + s_arr)
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min(equity / peak - 1))
        row: dict[str, object] = {
            "bucket": label,
            "n": n,
            "cum_return": cum,
            "ann_return": ann,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }
        if benchmark_col is not None and benchmark_col in sub.columns:
            b_arr = sub[benchmark_col].to_numpy()
            diag = compute_benchmark_diagnostics(s_arr, b_arr, periods_per_year)
            row["beta_vs_bm"] = diag["beta"]
            row["info_ratio_vs_bm"] = diag["info_ratio"]
        rows.append(row)
    return pl.DataFrame(rows)


def compute_benchmark_diagnostics(
    strategy: np.ndarray,
    benchmark: np.ndarray,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Information ratio, beta, correlation, tracking error vs benchmark.

    Inputs are aligned period returns; lengths must match. NaNs are
    dropped pairwise. Returns NaN for any metric whose inputs are
    degenerate (all-NaN or zero variance).
    """
    s = np.asarray(strategy, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    if s.shape != b.shape:
        raise ValueError(f"shape mismatch: strategy {s.shape} vs benchmark {b.shape}")
    mask = np.isfinite(s) & np.isfinite(b)
    s, b = s[mask], b[mask]
    n = int(s.size)
    out: dict[str, float | int] = {"n": n}
    if n < 2:
        return {
            **out,
            "info_ratio": float("nan"),
            "beta": float("nan"),
            "correlation": float("nan"),
            "tracking_error": float("nan"),
        }
    excess = s - b
    te_period = float(np.std(excess, ddof=1))
    te_ann = te_period * np.sqrt(periods_per_year)
    if te_period > 0:
        ir = float(np.mean(excess) * periods_per_year / te_ann)
    else:
        ir = float("nan")
    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(s, b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
    if np.std(s, ddof=1) > 0 and np.std(b, ddof=1) > 0:
        corr = float(np.corrcoef(s, b)[0, 1])
    else:
        corr = float("nan")
    return {
        "n": n,
        "info_ratio": ir,
        "beta": beta,
        "correlation": corr,
        "tracking_error": te_ann,
    }
