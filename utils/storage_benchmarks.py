"""Shared utilities for storage benchmarks.

This module provides common functionality used by both local and server benchmarks:
- Synthetic data generation (OHLCV, tick data)
- Timing infrastructure with warm-up and percentiles
- Result validation with forced materialization
- Configuration via YAML or environment variables
- Memory estimation utilities
"""

import gc
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import yaml

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Benchmark scale configuration
# BENCHMARK_SCALE: XS, S, M, L, XL, XXL
BENCHMARK_SCALE = os.environ.get("BENCHMARK_SCALE", "").upper()

# Chapter directory paths
from utils.paths import get_chapter_dir

CHAPTER_DIR = get_chapter_dir(2)
CODE_DIR = CHAPTER_DIR

# Load configuration from YAML if available
CONFIG_PATH = CODE_DIR / "benchmark_config.yaml"


def load_config() -> dict:
    """Load benchmark configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


CONFIG = load_config()

# New scale configurations (from YAML or defaults)
# Format: {scale: {ohlcv: {symbols, rows_per_symbol, total_rows}, tick: {...}}}
SCALE_CONFIGS_NEW = {}
if CONFIG.get("scales"):
    for scale_name, scale_cfg in CONFIG["scales"].items():
        ohlcv = scale_cfg.get("ohlcv", {})
        tick = scale_cfg.get("tick", {})
        SCALE_CONFIGS_NEW[scale_name] = {
            "ohlcv": {
                "symbols": ohlcv.get("symbols", 10),
                "rows_per_symbol": ohlcv.get("rows_per_symbol", 1000),
                "total_rows": ohlcv.get("total_rows", 10000),
            },
            "tick": {
                "symbols": tick.get("symbols", 5),
                "trades": tick.get("trades", 5000),
                "quotes": tick.get("quotes", 25000),
            },
            "target_memory": scale_cfg.get("target_memory", "1MB"),
            "description": scale_cfg.get("description", ""),
        }
else:
    # Default scale configs if YAML not present
    SCALE_CONFIGS_NEW = {
        "XS": {
            "ohlcv": {"symbols": 5, "rows_per_symbol": 200, "total_rows": 1000},
            "tick": {"symbols": 3, "trades": 500, "quotes": 2500},
            "target_memory": "100KB",
            "description": "Unit tests",
        },
        "S": {
            "ohlcv": {"symbols": 10, "rows_per_symbol": 1000, "total_rows": 10000},
            "tick": {"symbols": 5, "trades": 5000, "quotes": 25000},
            "target_memory": "1MB",
            "description": "Quick validation",
        },
        "M": {
            "ohlcv": {"symbols": 50, "rows_per_symbol": 2000, "total_rows": 100000},
            "tick": {"symbols": 10, "trades": 50000, "quotes": 250000},
            "target_memory": "10MB",
            "description": "Development",
        },
        "L": {
            "ohlcv": {"symbols": 100, "rows_per_symbol": 10000, "total_rows": 1000000},
            "tick": {"symbols": 50, "trades": 500000, "quotes": 2500000},
            "target_memory": "100MB",
            "description": "Standard benchmark",
        },
        "XL": {
            "ohlcv": {"symbols": 500, "rows_per_symbol": 20000, "total_rows": 10000000},
            "tick": {"symbols": 100, "trades": 5000000, "quotes": 25000000},
            "target_memory": "1GB",
            "description": "Scale testing",
        },
        "XXL": {
            "ohlcv": {"symbols": 1000, "rows_per_symbol": 100000, "total_rows": 100000000},
            "tick": {"symbols": 500, "trades": 50000000, "quotes": 250000000},
            "target_memory": "10GB",
            "description": "Production-scale",
        },
    }


def get_scale_config(scale: str) -> dict:
    """Get configuration for a scale level.

    Args:
        scale: Scale name (XS, S, M, L, XL, XXL)

    Returns:
        Dict with ohlcv and tick configuration
    """
    if scale not in SCALE_CONFIGS_NEW:
        raise ValueError(f"Unknown scale {scale!r}, expected one of {list(SCALE_CONFIGS_NEW)}")
    return SCALE_CONFIGS_NEW[scale]


# Determine active scale
# BENCHMARK_VERBOSE controls whether to print on import (default: False for clean imports)
BENCHMARK_VERBOSE = os.environ.get("BENCHMARK_VERBOSE", "0") == "1"

if BENCHMARK_SCALE and BENCHMARK_SCALE in SCALE_CONFIGS_NEW:
    ACTIVE_SCALE = BENCHMARK_SCALE
    scale_cfg = get_scale_config(BENCHMARK_SCALE)
    N_SYMBOLS = scale_cfg["ohlcv"]["symbols"]
    N_ROWS_PER_SYMBOL = scale_cfg["ohlcv"]["rows_per_symbol"]
    N_TICKS_TRADES = scale_cfg["tick"]["trades"]
    N_TICKS_QUOTES = scale_cfg["tick"]["quotes"]
    TIMING_RUNS = CONFIG.get("execution", {}).get("iterations", {}).get(ACTIVE_SCALE, 3)
else:
    # Default: S scale for quick iteration
    ACTIVE_SCALE = "S"
    scale_cfg = get_scale_config("S")
    N_SYMBOLS = scale_cfg["ohlcv"]["symbols"]
    N_ROWS_PER_SYMBOL = scale_cfg["ohlcv"]["rows_per_symbol"]
    N_TICKS_TRADES = scale_cfg["tick"]["trades"]
    N_TICKS_QUOTES = scale_cfg["tick"]["quotes"]
    TIMING_RUNS = 3

# Database connection configuration (environment variable overrides)
DB_CONFIG = {
    "clickhouse": {
        "host": os.environ.get("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.environ.get("CLICKHOUSE_PORT", "8123")),
    },
    "questdb": {
        "host": os.environ.get("QUESTDB_HOST", "localhost"),
        "http_port": int(os.environ.get("QUESTDB_HTTP_PORT", "9000")),
        "ilp_port": int(os.environ.get("QUESTDB_ILP_PORT", "9009")),
        "pg_port": int(os.environ.get("QUESTDB_PG_PORT", "8812")),
    },
    "timescaledb": {
        "host": os.environ.get("TIMESCALE_HOST", "localhost"),
        "port": int(os.environ.get("TIMESCALE_PORT", "5437")),
        "user": os.environ.get("TIMESCALE_USER", "postgres"),
        "password": os.environ.get("TIMESCALE_PASSWORD", "benchmark"),
        "database": os.environ.get("TIMESCALE_DB", "postgres"),
    },
    "influxdb": {
        "host": os.environ.get("INFLUXDB_HOST", "localhost"),
        "port": int(os.environ.get("INFLUXDB_PORT", "8086")),
        "org": os.environ.get("INFLUXDB_ORG", "benchmark"),
        "token": os.environ.get("INFLUXDB_TOKEN", "benchmark-token"),
        "bucket": os.environ.get("INFLUXDB_BUCKET", "benchmark"),
    },
    "postgres": {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5436")),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "benchmark"),
        "database": os.environ.get("POSTGRES_DB", "ml4t"),
    },
}

# WAL flush timeout (seconds) - adjustable for slower systems
WAL_FLUSH_TIMEOUT = int(os.environ.get("WAL_FLUSH_TIMEOUT", "3"))

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
# Directory structure:
#   .tmp/                  - Transient data (gitignored), regenerated each run
#   ../output/benchmark/   - Results CSVs for book tables and citation
#   ../figures/            - Book figures (numbered: figure_3_N_slug.png)

# Transient benchmark data (synthetic OHLCV, trades, quotes)
TMP_DIR = CHAPTER_DIR / ".tmp"
TMP_DIR.mkdir(exist_ok=True)
BENCHMARK_DIR = TMP_DIR  # Alias used by benchmark notebooks

# Working charts (transient, not book figures)
CHARTS_DIR = TMP_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Results for book tables (CSV files for citation in prose)
RESULTS_DIR = CHAPTER_DIR / "output" / "benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    operation: str
    time_seconds: float
    size_bytes: int = 0
    rows: int = 0

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.time_seconds if self.time_seconds > 0 else 0

    @property
    def mb_per_second(self) -> float:
        return (self.size_bytes / 1e6) / self.time_seconds if self.time_seconds > 0 else 0


# =============================================================================
# TIMING & VALIDATION
# =============================================================================


def time_operation(func, n_runs: int = TIMING_RUNS, warmup: bool = True) -> tuple[float, Any]:
    """Time a function with warm-up and percentile tracking.

    Args:
        func: Function to time
        n_runs: Number of timing runs (default: TIMING_RUNS)
        warmup: Whether to run once before timing to warm up caches/JIT (default: True)

    Returns:
        (mean_time, result): Mean execution time and last result
            Note: Timing stats (percentiles) are stored in result.timing_stats if available
    """
    # Warm-up run (exclude from timing)
    if warmup:
        from contextlib import suppress

        with suppress(Exception):  # Warm-up failure is non-critical
            func()

    # Timing runs
    times = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Calculate statistics
    times_array = np.array(times)
    mean_time = float(np.mean(times_array))

    # Store timing stats as metadata (if result supports it)
    try:
        if hasattr(result, "__dict__"):
            result.timing_stats = {
                "mean": mean_time,
                "std": float(np.std(times_array)),
                "min": float(np.min(times_array)),
                "max": float(np.max(times_array)),
                "p50": float(np.percentile(times_array, 50)),
                "p95": float(np.percentile(times_array, 95)),
                "p99": float(np.percentile(times_array, 99)),
            }
    except Exception:
        pass  # Not all result types support metadata

    return mean_time, result


# -----------------------------------------------------------------------------
# TIMING POLICY
# -----------------------------------------------------------------------------
# One policy, applied to every format and every engine in the chapter §2.4
# benchmarks. Mixing policies within a single comparison makes the resulting
# chart meaningless, so all call sites go through `time_write` / `time_read`
# rather than passing ad-hoc n_runs/warmup arguments to `time_operation`.
#
#   Writes -> single shot, no warm-up, against freshly created storage, with
#             any durability wait (WAL flush, async ingestion) INSIDE the timed
#             region. Bulk load is a once-per-dataset operation; repeating it
#             would either append duplicate rows on the append-only engines or
#             require a teardown that is not part of the operation being timed.
#
#   Reads   -> mean of TIMING_RUNS runs after one untimed warm-up run.
#             These are WARM-CACHE numbers, on both the OS page cache and each
#             server's own buffer pool, and they are reported as such. A client
#             cannot drop a database server's caches, so a "cold" client-side
#             read would be cold for the file formats and warm for the servers:
#             neither cold nor comparable. Warm-cache timing flatters
#             memory-mapped formats (Feather) and every engine with a buffer
#             pool; the notebooks state this where the results are interpreted.


def time_write(func) -> tuple[float, Any]:
    """Time a bulk write under the chapter's write policy: one cold shot.

    Args:
        func: Write function. Must create its own storage and include any
            durability wait, so the timed region ends when the data is durable
            and queryable.

    Returns:
        (time_seconds, result)
    """
    gc.collect()
    return time_operation(func, n_runs=1, warmup=False)


def time_read(func, n_runs: int | None = None) -> tuple[float, Any]:
    """Time a read/aggregation/join under the chapter's read policy: warm mean.

    Args:
        func: Read function, which must force materialization of its result.
        n_runs: Timing runs after the untimed warm-up (default: TIMING_RUNS).

    Returns:
        (mean_time_seconds, result)
    """
    gc.collect()
    return time_operation(func, n_runs=n_runs or TIMING_RUNS, warmup=True)


def wait_until_rows_visible(
    count_func, expected_rows: int, timeout: float = 60.0, poll_seconds: float = 0.05
) -> int:
    """Block until an asynchronously-ingesting engine reports `expected_rows`.

    QuestDB (ILP + WAL) and InfluxDB acknowledge a write before the rows are
    queryable. Call this INSIDE the timed write so that their durability cost
    lands on the same side of the timed region as the synchronous engines
    (PostgreSQL/TimescaleDB), which pay it inline. Replaces a fixed sleep, which
    both under-counts slow ingestion and over-counts fast ingestion.

    Args:
        count_func: Callable returning the row count currently visible.
        expected_rows: Row count to wait for.
        timeout: Seconds to wait before failing.
        poll_seconds: Delay between polls.

    Returns:
        The visible row count once it reaches `expected_rows`.

    Raises:
        TimeoutError: If the rows never become visible. Never returns a short
            count: a silently-incomplete ingest would produce a fast, wrong
            write time and a wrong read count downstream.
    """
    deadline = time.perf_counter() + timeout
    visible = 0
    while time.perf_counter() < deadline:
        visible = count_func()
        if visible >= expected_rows:
            return visible
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Only {visible:,} of {expected_rows:,} rows visible after {timeout:.0f}s; "
        "ingestion did not complete."
    )


def validate_result(
    result: Any, expected_rows: int, operation: str, tolerance: float = 0.1
) -> None:
    """Validate benchmark result has reasonable row count.

    Args:
        result: Result to validate (DataFrame, list, dict with 'dataset', or None)
        expected_rows: Expected number of rows
        operation: Operation name for error messages
        tolerance: Fraction tolerance (0.1 = 10% deviation acceptable)

    Raises:
        AssertionError: If row count is unreasonable
    """
    if result is None:
        return  # Skip validation for None results (optional databases)

    # Extract row count based on result type
    if hasattr(result, "shape"):  # DataFrame
        actual_rows = result.shape[0]
    elif isinstance(result, dict) and "dataset" in result:  # QuestDB result
        actual_rows = len(result["dataset"])
    elif isinstance(result, list):
        actual_rows = len(result)
    else:
        return  # Unknown type, skip validation

    # Check row count is within tolerance
    min_rows = int(expected_rows * (1 - tolerance))
    max_rows = int(expected_rows * (1 + tolerance))

    if not (min_rows <= actual_rows <= max_rows):
        raise AssertionError(
            f"{operation}: Expected {expected_rows:,} rows (±{tolerance:.0%}), got {actual_rows:,}"
        )


# =============================================================================
# MATERIALIZATION HELPERS
# =============================================================================


def force_materialize_polars(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Force full materialization of a Polars DataFrame by scanning ALL columns.

    Memory-mapped formats (like Feather/Arrow IPC) may return handles without
    loading data. This function forces actual data access by touching every column.

    Args:
        df: Polars DataFrame or LazyFrame

    Returns:
        Materialized DataFrame with data actually loaded into memory
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Force materialization by touching EVERY column (not just first 3 numeric)
    # This ensures all data is actually loaded, not just memory-mapped
    exprs = []
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
            # Numeric: compute sum
            exprs.append(pl.col(col_name).sum().alias(f"{col_name}_sum"))
        elif dtype == pl.String:
            # String: compute length sum (forces read of all string data)
            exprs.append(pl.col(col_name).str.len_bytes().sum().alias(f"{col_name}_len"))
        elif dtype in (pl.Datetime, pl.Date):
            # Datetime: compute min/max (forces read)
            exprs.append(pl.col(col_name).min().alias(f"{col_name}_min"))
        else:
            # Other types: count non-null (forces read)
            exprs.append(pl.col(col_name).count().alias(f"{col_name}_count"))

    if exprs:
        _ = df.select(exprs).to_dict()

    return df


def force_materialize_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Force full materialization of a pandas DataFrame by scanning ALL columns.

    Args:
        df: pandas DataFrame

    Returns:
        Materialized DataFrame with all data accessed
    """
    # Force read of ALL columns, not just first 3 numeric
    result = {}

    # Numeric columns: compute sum
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        result["numeric_sums"] = df[numeric_cols].sum().to_dict()

    # String/object columns: compute total string length
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(object_cols) > 0:
        result["string_lens"] = {col: df[col].astype(str).str.len().sum() for col in object_cols}

    # Datetime columns: compute min
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns
    if len(datetime_cols) > 0:
        result["datetime_mins"] = {col: df[col].min() for col in datetime_cols}

    # Force evaluation
    _ = result

    return df


def read_with_materialization(
    read_func,
    path: Path,
    library: str = "polars",
) -> tuple[float, Any]:
    """Time a read operation with forced materialization.

    Args:
        read_func: Function to read data (e.g., pl.read_parquet, pd.read_csv)
        path: Path to file
        library: "polars" or "pandas"

    Returns:
        (time_seconds, result): Tuple of read time and DataFrame
    """
    gc.collect()

    start = time.perf_counter()
    df = read_func(path)

    # Force materialization
    if library == "polars":
        df = force_materialize_polars(df)
    else:
        df = force_materialize_pandas(df)

    elapsed = time.perf_counter() - start
    return elapsed, df


# =============================================================================
# MEMORY UTILITIES
# =============================================================================


def estimate_memory_mb(df: pl.DataFrame | pd.DataFrame) -> float:
    """Estimate memory usage of a DataFrame in MB.

    Args:
        df: Polars or pandas DataFrame

    Returns:
        Estimated memory in megabytes
    """
    if isinstance(df, pl.DataFrame):
        return df.estimated_size("mb")
    else:
        return df.memory_usage(deep=True).sum() / 1_000_000


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes
    """
    import psutil

    process = psutil.Process()
    return process.memory_info().rss / 1_000_000


def run_with_gc(func):
    """Run function with garbage collection before and after.

    Args:
        func: Function to run

    Returns:
        Function result
    """
    gc.collect()
    result = func()
    gc.collect()
    return result


def save_chart(fig: go.Figure, name: str) -> None:
    """Save chart to HTML file instead of opening browser."""
    path = CHARTS_DIR / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"Chart saved: {path}")


# =============================================================================
# DATA GENERATION
# =============================================================================


# Regular US equity session: 09:30-16:00 ET inclusive of the opening minute,
# i.e. 390 one-minute bars per trading day.
SESSION_OPEN_MINUTE = 9 * 60 + 30
SESSION_BARS = 390
SESSION_START_DATE = "2024-01-02"  # first business day of 2024


def session_minute_grid(n_rows: int, start_date: str = SESSION_START_DATE) -> np.ndarray:
    """Build `n_rows` one-minute bar timestamps laid out over trading sessions.

    Bars run 09:30-16:00 on consecutive business days, so the grid carries the
    overnight and weekend gaps a real minute panel has. This matters for the
    benchmarks in two ways: a calendar-day range query selects a realistic
    fraction of the panel, and resampling minute bars to daily bars is a real
    reduction rather than a near-identity.

    Args:
        n_rows: Number of bars to generate.
        start_date: First session date (rolled forward to a business day).

    Returns:
        numpy datetime64[us] array of length `n_rows`, strictly increasing.
    """
    n_sessions = int(np.ceil(n_rows / SESSION_BARS))
    session_days = np.busday_offset(
        np.datetime64(start_date, "D"), np.arange(n_sessions), roll="forward"
    )
    session_open = session_days.astype("datetime64[m]") + np.timedelta64(SESSION_OPEN_MINUTE, "m")
    bar_offsets = np.arange(SESSION_BARS, dtype="timedelta64[m]")
    grid = (session_open[:, np.newaxis] + bar_offsets[np.newaxis, :]).ravel()
    return grid[:n_rows].astype("datetime64[us]")


def generate_ohlcv_data(
    n_symbols: int = N_SYMBOLS,
    n_rows: int = N_ROWS_PER_SYMBOL,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate realistic synthetic OHLCV panel data (fully vectorized).

    Creates a panel of one-minute bars for `n_symbols` symbols over consecutive
    regular trading sessions (390 bars per session). OHLCV constraints are
    enforced: H >= max(O,C), L <= min(O,C).

    All symbols share one session grid, which is what a synchronized bar panel
    looks like: every symbol prints a bar for the same minute of the same
    session. That shared grid, and the low-cardinality symbol column, are
    genuine properties of bar panels rather than artifacts of the generator, and
    the dictionary/delta encoding they enable is part of what the format
    comparison is measuring.

    This implementation is fully vectorized using numpy arrays, enabling
    generation of 10M+ rows in seconds (required for XL/XXL scales).

    Args:
        n_symbols: Number of unique symbols
        n_rows: Number of one-minute bars per symbol
        seed: Random seed for reproducibility

    Returns:
        Polars DataFrame with columns: timestamp, symbol, open, high, low, close, volume, vwap, num_trades
    """
    np.random.seed(seed)
    total_rows = n_symbols * n_rows

    # One session-aware minute grid, shared by every symbol (see docstring).
    single_symbol_times = session_minute_grid(n_rows)
    timestamps = np.tile(single_symbol_times, n_symbols)

    # Generate symbol array (repeat each symbol n_rows times)
    symbol_names = np.array([f"SYM_{i:03d}" for i in range(n_symbols)])
    symbols = np.repeat(symbol_names, n_rows)

    # Generate base prices per symbol, then broadcast to all rows
    base_prices = 100 + np.random.randn(n_symbols) * 10  # (n_symbols,)

    # Generate returns for all rows at once: (n_symbols, n_rows)
    returns = np.random.randn(n_symbols, n_rows) * 0.001

    # Cumulative sum along rows axis, then flatten
    cumret = np.cumsum(returns, axis=1)  # (n_symbols, n_rows)
    prices = (base_prices[:, np.newaxis] * np.exp(cumret)).flatten()  # (total_rows,)

    # Generate intrabar noise for all rows
    noise = np.abs(np.random.randn(total_rows)) * 0.002

    # Generate OHLC
    opens = prices * (1 - noise * 0.5)
    closes = prices * (1 + noise * 0.5)
    highs_raw = prices * (1 + noise)
    lows_raw = prices * (1 - noise)

    # Enforce OHLC constraints: H >= max(O,C), L <= min(O,C)
    highs = np.maximum(np.maximum(opens, closes), highs_raw)
    lows = np.minimum(np.minimum(opens, closes), lows_raw)

    # Generate volume (lognormal distribution)
    volumes = np.exp(np.random.randn(total_rows) * 0.5 + 10).astype(np.int64)

    # VWAP approximation: typical price (H+L+C)/3
    vwap = (highs + lows + closes) / 3

    # Number of trades (proportional to volume with noise)
    num_trades = (volumes / 1000 + np.random.randint(10, 100, total_rows)).astype(np.int32)

    # Build DataFrame directly from numpy arrays (zero-copy where possible)
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbols,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "vwap": vwap,
            "num_trades": num_trades,
        },
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "vwap": pl.Float64,
            "num_trades": pl.Int32,
        },
    )

    return df


def generate_tick_data(
    n_trades: int = N_TICKS_TRADES,
    n_quotes: int = N_TICKS_QUOTES,
    n_symbols: int = N_SYMBOLS,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate realistic synthetic tick data (fully vectorized).

    Trades are correlated with quotes (occur near bid/ask) for realistic simulation.

    This implementation is fully vectorized using numpy arrays, enabling
    generation of millions of ticks in seconds (required for XL/XXL scales).

    Args:
        n_trades: Total number of trade ticks
        n_quotes: Total number of quote ticks
        n_symbols: Number of unique symbols
        seed: Random seed for reproducibility

    Returns:
        (trades_df, quotes_df): Tuple of Polars DataFrames
    """
    np.random.seed(seed)

    symbol_names = np.array([f"SYM_{i:03d}" for i in range(n_symbols)])
    base_time = np.datetime64("2024-01-01T09:30:00", "us")

    # =========================================================================
    # QUOTES: Fully vectorized generation
    # =========================================================================

    # Assign quotes to symbols (weighted: some symbols more active)
    # Use exponential weights so first symbols get more quotes
    weights = np.exp(-np.arange(n_symbols) * 0.1)
    weights = weights / weights.sum()
    quote_symbol_indices = np.random.choice(n_symbols, size=n_quotes, p=weights)
    quote_symbols = symbol_names[quote_symbol_indices]

    # Generate timestamps (100 microseconds apart)
    quote_timestamps = base_time + (np.arange(n_quotes) * 100).astype("timedelta64[us]")

    # Generate mid prices per symbol using random walk
    # Strategy: for each symbol, generate cumulative random walk, then gather
    quotes_per_symbol = np.bincount(quote_symbol_indices, minlength=n_symbols)
    max_quotes_per_symbol = quotes_per_symbol.max()

    # Generate random walk increments for all symbols at once
    base_mids = 150 + np.random.randn(n_symbols) * 20  # Starting mid price per symbol
    walk_increments = np.random.randn(n_symbols, max_quotes_per_symbol) * 0.01
    walk_paths = base_mids[:, np.newaxis] + np.cumsum(walk_increments, axis=1)

    # For each quote, look up the appropriate price from that symbol's walk
    # Track position within each symbol's sequence
    symbol_counters = np.zeros(n_symbols, dtype=np.int64)
    mid_prices = np.empty(n_quotes, dtype=np.float64)

    # Vectorized lookup using cumulative counts
    for sym_idx in range(n_symbols):
        mask = quote_symbol_indices == sym_idx
        count = mask.sum()
        if count > 0:
            mid_prices[mask] = walk_paths[sym_idx, :count]

    # Generate spreads (0.01-0.05% of mid)
    spread_pct = 0.0001 + np.abs(np.random.randn(n_quotes)) * 0.0002
    spreads = mid_prices * spread_pct
    bids = mid_prices - spreads / 2
    asks = mid_prices + spreads / 2

    # Generate sizes (lognormal)
    bid_sizes = np.exp(np.random.randn(n_quotes) * 0.3 + 5).astype(np.int64)
    ask_sizes = np.exp(np.random.randn(n_quotes) * 0.3 + 5).astype(np.int64)

    quotes_df = pl.DataFrame(
        {
            "timestamp": quote_timestamps,
            "symbol": quote_symbols,
            "bid": bids,
            "ask": asks,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
        },
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "bid": pl.Float64,
            "ask": pl.Float64,
            "bid_size": pl.Int64,
            "ask_size": pl.Int64,
        },
    )

    # =========================================================================
    # TRADES: Fully vectorized generation
    # =========================================================================

    # Assign trades to symbols (same distribution as quotes)
    trade_symbol_indices = np.random.choice(n_symbols, size=n_trades, p=weights)
    trade_symbols = symbol_names[trade_symbol_indices]

    # Generate timestamps (100 microseconds apart, offset by 50us from quotes)
    trade_timestamps = base_time + (np.arange(n_trades) * 100 + 50).astype("timedelta64[us]")

    # Generate sizes
    trade_sizes = np.exp(np.random.randn(n_trades) * 0.3 + 5).astype(np.int64)

    # Build preliminary trades DataFrame
    trades_prelim = pl.DataFrame(
        {
            "timestamp": trade_timestamps,
            "symbol": trade_symbols,
            "price": np.zeros(n_trades),  # Placeholder
            "size": trade_sizes,
        },
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "price": pl.Float64,
            "size": pl.Int64,
        },
    )

    # Use ASOF join to match each trade with most recent quote
    trades_with_quotes = trades_prelim.join_asof(
        quotes_df.sort(["symbol", "timestamp"]),
        on="timestamp",
        by="symbol",
        strategy="backward",
    )

    # Calculate trade prices: 25% at ask, 25% at bid, 50% at mid
    trade_sides = np.random.rand(n_trades)
    bid_arr = trades_with_quotes["bid"].to_numpy()
    ask_arr = trades_with_quotes["ask"].to_numpy()

    trade_prices = np.where(
        trade_sides < 0.25,
        ask_arr,  # Buy at ask (25%)
        np.where(
            trade_sides < 0.5,
            bid_arr,  # Sell at bid (25%)
            (bid_arr + ask_arr) / 2,  # Mid (50%)
        ),
    )

    # Handle NaN prices (trades before first quote)
    trade_prices = np.where(np.isnan(trade_prices), 150.0, trade_prices)

    # Create final trades DataFrame
    trades_df = pl.DataFrame(
        {
            "timestamp": trades_with_quotes["timestamp"],
            "symbol": trades_with_quotes["symbol"],
            "price": trade_prices,
            "size": trades_with_quotes["size"],
        }
    )

    return trades_df, quotes_df


# =============================================================================
# RESULTS STORAGE
# =============================================================================

# Map benchmark_type to clean filename prefixes
BENCHMARK_TYPE_TO_FILENAME = {
    "formats": "file_formats",
    "embedded": "embedded_dbs",
    "pandas_polars": "pandas_polars",
    "servers": "server_dbs",
    # Legacy mappings
    "local": "local",
    "server": "server",
}


def save_benchmark_results(
    results: list[BenchmarkResult], benchmark_type: str, scale: str | None = None
) -> Path:
    """Save benchmark results to CSV for book publication.

    Results are saved to output/benchmark/ with clean filenames for citation in prose.
    Example: file_formats_l.csv, embedded_dbs_xl.csv

    Args:
        results: List of BenchmarkResult objects
        benchmark_type: "formats", "embedded", "pandas_polars", "servers"
        scale: Scale level (S, L, XL, etc.) or None for auto-detect

    Returns:
        Path to saved CSV file
    """
    if scale is None:
        scale = ACTIVE_SCALE

    # Clean filename prefix
    filename_prefix = BENCHMARK_TYPE_TO_FILENAME.get(benchmark_type, benchmark_type)

    # Create results DataFrame
    df = pl.DataFrame(
        [
            {
                "benchmark_type": benchmark_type,
                "scale": scale,
                "technology": r.name,
                "operation": r.operation,
                "time_seconds": r.time_seconds,
                "size_mb": r.size_bytes / 1_000_000 if r.size_bytes else None,
                "rows": r.rows if r.rows else None,
                "rows_per_second": r.rows_per_second if r.rows else None,
                "mb_per_second": r.mb_per_second if r.size_bytes else None,
                "timestamp": pd.Timestamp.now().isoformat(),
                "n_symbols": N_SYMBOLS,
                "n_rows_per_symbol": N_ROWS_PER_SYMBOL,
            }
            for r in results
        ]
    )

    # Save as CSV (for book tables and prose citation)
    csv_path = RESULTS_DIR / f"{filename_prefix}_{scale.lower()}.csv"
    df.write_csv(csv_path)

    print(f"\n📁 Results saved to: {csv_path}")
    return csv_path


# =============================================================================
# PRINT CONFIGURATION (only when BENCHMARK_VERBOSE=1 or running as main)
# =============================================================================


def print_config() -> None:
    """Print current benchmark configuration."""
    print(f"📊 Scale: {ACTIVE_SCALE} ({scale_cfg['target_memory']} target)")
    print(f"   OHLCV: {N_SYMBOLS} symbols × {N_ROWS_PER_SYMBOL:,} rows/symbol")
    print(f"   Ticks: {N_TICKS_TRADES:,} trades, {N_TICKS_QUOTES:,} quotes")
    print(f"   Timing runs: {TIMING_RUNS}")


# Only print on import if explicitly requested
if BENCHMARK_VERBOSE:
    print_config()
