# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # File-Format Storage Benchmark
#
# **Environment**: the locked environment (`uv run`) — see Prerequisites below.
#
# **Purpose**: Compare CSV, Parquet, Feather (Arrow IPC), and HDF5 on the same
# 1 M-row OHLCV panel along three axes — write time, read time (with forced
# materialization), and on-disk size — so the trade-offs in §2.4 are
# reproducible end to end.
#
# **Learning objectives**:
# 1. Generate a deterministic OHLCV benchmark panel at the L scale that
#    chapter §2.4 cites (100 symbols × 10,000 one-minute bars = 1,000,000 rows).
# 2. Time write and read for each format under one stated timing policy.
# 3. Force materialization on Feather / HDF5 reads so memory-mapped or lazy
#    reads don't masquerade as instant.
# 4. Quantify the columnar-projection win (read 2 columns vs 9).
# 5. Render a 3-panel comparison (read time / write time / file size).
#
# **Book reference**: §2.4 — file-based storage benchmarks.
#
# **Prerequisites**: PyTables (the HDF5 backend) ships in the locked
# environment, so `uv run python 02_financial_data_universe/20_storage_benchmark_file.py`
# from the repo root is all this notebook needs — no database services, unlike
# `21_storage_benchmark_database`.
#
# > **Which environment produced the numbers**: the locked environment
# > (`uv sync`, i.e. `uv.lock`), which is what §2.4 reports. The `benchmark`
# > Docker image currently resolves a *newer* pandas than `uv.lock` pins, and
# > pandas' newer string dtype makes PyTables store the low-cardinality
# > `symbol` column about 8 bytes/row wider — enough to move the HDF5 file
# > from ~71 MB to ~79 MB for the identical panel. CSV, Parquet, and Feather
# > are unaffected. Run this notebook under `uv run` to reproduce §2.4.

# %% [markdown]
# ## Setup

# %%
"""File-format storage benchmark — CSV / Parquet / Feather / HDF5 at L scale."""

import gc
import os
import time

# %% tags=["parameters"]
# Production scale follows chapter §2.4 prose ("L scale, ~1 M OHLCV rows,
# 64 MB in-memory in Polars"). Override via Papermill for CI: BENCHMARK_SCALE = "S".
BENCHMARK_SCALE = "L"

# %%
# storage_benchmarks reads BENCHMARK_SCALE at import time, so the env var
# must be set before the import below.
os.environ["BENCHMARK_SCALE"] = BENCHMARK_SCALE

import pandas as pd
import plotly.graph_objects as go
import polars as pl

# PyTables is the HDF5 backend; raise loudly if the image is wrong.
import tables  # noqa: F401
from plotly.subplots import make_subplots

from utils.paths import get_output_dir
from utils.storage_benchmarks import (
    ACTIVE_SCALE,
    BENCHMARK_DIR,
    N_ROWS_PER_SYMBOL,
    N_SYMBOLS,
    BenchmarkResult,
    estimate_memory_mb,
    force_materialize_pandas,
    force_materialize_polars,
    generate_ohlcv_data,
    save_benchmark_results,
    time_read,
    time_write,
    validate_result,
)
from utils.style import COLORS

OUTPUT_DIR = get_output_dir(2, "storage_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Timing Policy
#
# Every number below follows one policy, applied identically to all four
# formats. A comparison that mixes policies across the things it compares is
# not a comparison, so this is stated up front rather than left in the call
# sites:
#
# - **Writes** (`time_write`) — a single shot, no warm-up run. Writing the
#   panel is a once-per-dataset operation, so that is what we time.
# - **Reads** (`time_read`) — the mean of `TIMING_RUNS` runs after one untimed
#   warm-up run. **These are warm-cache numbers.** The file has just been
#   written and re-read repeatedly, so it sits in the OS page cache.
#
# Warm-cache reads are the honest description of the *repeated-access* pattern
# a research loop actually has, but they flatter memory-mapped formats: Feather
# maps pages that are already resident, so its read number below is close to a
# best case. A first read of a cold file off disk narrows the gap to Parquet,
# and on a network or object store the compressed format usually wins outright
# because it moves fewer bytes. Rankings here are for warm, local, repeated
# reads — the caveat that §2.4 attaches to these figures.

# %% [markdown]
# ## 1. Generate the Benchmark Panel
#
# `generate_ohlcv_data` returns a deterministic OHLCV panel: per-symbol random
# walks under a fixed seed so format comparisons aren't muddled by data drift
# across runs. At L scale that's 100 symbols × 10,000 one-minute bars =
# 1 M total rows, laid out over 26 regular trading sessions (390 bars each), so
# the panel carries the overnight and weekend gaps real minute data has. All
# symbols share the session grid, as they do in any synchronized bar panel.

# %%
ohlcv_df = generate_ohlcv_data(n_symbols=N_SYMBOLS, n_rows=N_ROWS_PER_SYMBOL)
ohlcv_pandas = ohlcv_df.to_pandas()

panel_summary = pl.DataFrame(
    {
        "field": [
            "scale",
            "symbols",
            "rows per symbol",
            "total rows",
            "in-memory size (Polars, MB)",
            "in-memory size (pandas, MB)",
        ],
        "value": [
            ACTIVE_SCALE,
            f"{N_SYMBOLS:,}",
            f"{N_ROWS_PER_SYMBOL:,}",
            f"{len(ohlcv_df):,}",
            f"{estimate_memory_mb(ohlcv_df):.2f}",
            f"{estimate_memory_mb(ohlcv_pandas):.2f}",
        ],
    }
)
panel_summary

# %%
total_rows = len(ohlcv_df)
results: list[BenchmarkResult] = []

# %% [markdown]
# ## 2. CSV — Universal Baseline
#
# CSV is the row-oriented baseline: human-readable, no compression, no schema.
# Every other format is judged against it.

# %%
csv_path = BENCHMARK_DIR / f"ohlcv_{ACTIVE_SCALE.lower()}.csv"

write_time, _ = time_write(lambda: ohlcv_df.write_csv(csv_path))
csv_size = csv_path.stat().st_size
results.append(BenchmarkResult("CSV", "write", write_time, csv_size, total_rows))


def read_csv_materialized() -> pl.DataFrame:
    return force_materialize_polars(pl.read_csv(csv_path))


read_time, csv_result = time_read(read_csv_materialized)
validate_result(csv_result, total_rows, "CSV read")
results.append(BenchmarkResult("CSV", "read", read_time, csv_size, total_rows))


def read_csv_columnar() -> pl.DataFrame:
    return force_materialize_polars(pl.read_csv(csv_path, columns=["close", "volume"]))


columnar_time, _ = time_read(read_csv_columnar)
results.append(BenchmarkResult("CSV", "columnar_read", columnar_time, csv_size, total_rows))

# %% [markdown]
# ## 3. Parquet — Compressed Columnar Standard
#
# Parquet's row-group layout, dictionary encoding, and Snappy compression make
# it the default for analytical workloads. Column projection reads only the
# row-group chunks for the requested columns.

# %%
parquet_path = BENCHMARK_DIR / f"ohlcv_{ACTIVE_SCALE.lower()}.parquet"

write_time, _ = time_write(lambda: ohlcv_df.write_parquet(parquet_path))
parquet_size = parquet_path.stat().st_size
results.append(BenchmarkResult("Parquet", "write", write_time, parquet_size, total_rows))


def read_parquet_materialized() -> pl.DataFrame:
    return force_materialize_polars(pl.read_parquet(parquet_path))


read_time, parquet_result = time_read(read_parquet_materialized)
validate_result(parquet_result, total_rows, "Parquet read")
results.append(BenchmarkResult("Parquet", "read", read_time, parquet_size, total_rows))


def read_parquet_columnar() -> pl.DataFrame:
    return force_materialize_polars(pl.read_parquet(parquet_path, columns=["close", "volume"]))


columnar_time, _ = time_read(read_parquet_columnar)
results.append(BenchmarkResult("Parquet", "columnar_read", columnar_time, parquet_size, total_rows))

# %% [markdown]
# ## 4. Feather (Arrow IPC) — Zero-Copy Interchange
#
# Feather opens the file by memory-mapping it; the bare `read_ipc` call returns
# almost instantly because no bytes have been read yet. We capture both the
# raw open time and the time to actually materialize the columns into memory.
# Only the materialized number is comparable across formats.

# %%
feather_path = BENCHMARK_DIR / f"ohlcv_{ACTIVE_SCALE.lower()}.feather"

write_time, _ = time_write(lambda: ohlcv_df.write_ipc(feather_path))
feather_size = feather_path.stat().st_size
results.append(BenchmarkResult("Feather", "write", write_time, feather_size, total_rows))

gc.collect()
start = time.perf_counter()
_ = pl.read_ipc(feather_path)  # raw handle — memory-mapped, not materialized
raw_handle_time = time.perf_counter() - start


def read_feather_materialized() -> pl.DataFrame:
    return force_materialize_polars(pl.read_ipc(feather_path))


read_time, feather_result = time_read(read_feather_materialized)
validate_result(feather_result, total_rows, "Feather read")
results.append(BenchmarkResult("Feather", "read", read_time, feather_size, total_rows))


def read_feather_columnar() -> pl.DataFrame:
    return force_materialize_polars(pl.read_ipc(feather_path, columns=["close", "volume"]))


columnar_time, _ = time_read(read_feather_columnar)
results.append(BenchmarkResult("Feather", "columnar_read", columnar_time, feather_size, total_rows))

# %% [markdown]
# ## 5. HDF5 — Legacy Scientific Container
#
# HDF5 keeps a foothold in research codebases that predate Parquet. The
# `fixed` format used by `pandas.HDFStore` doesn't support column projection,
# so we record the columnar read as the same as the full read for fairness.

# %%
hdf5_path = BENCHMARK_DIR / f"ohlcv_{ACTIVE_SCALE.lower()}.h5"


def write_hdf5() -> None:
    with pd.HDFStore(hdf5_path, mode="w") as store:
        store["ohlcv"] = ohlcv_pandas


write_time, _ = time_write(write_hdf5)
hdf5_size = hdf5_path.stat().st_size
results.append(BenchmarkResult("HDF5", "write", write_time, hdf5_size, total_rows))


def read_hdf5_materialized() -> pd.DataFrame:
    with pd.HDFStore(hdf5_path, mode="r") as store:
        df = store["ohlcv"]
    return force_materialize_pandas(df)


read_time, hdf5_result = time_read(read_hdf5_materialized)
validate_result(hdf5_result, total_rows, "HDF5 read")
results.append(BenchmarkResult("HDF5", "read", read_time, hdf5_size, total_rows))
# Fixed-format HDF5 has no column projection — record the full-read time
# so the comparison plot still has a value for the format.
results.append(BenchmarkResult("HDF5", "columnar_read", read_time, hdf5_size, total_rows))

# %% [markdown]
# ## 6. Results Summary

# %%
results_df = pl.DataFrame(
    [
        {
            "format": r.name,
            "operation": r.operation,
            "time_s": r.time_seconds,
            "size_mb": r.size_bytes / 1e6,
            "throughput_M_rows_s": r.rows_per_second / 1e6,
        }
        for r in results
    ]
)

# %%
write_summary = (
    results_df.filter(pl.col("operation") == "write")
    .select(["format", "time_s", "size_mb", "throughput_M_rows_s"])
    .sort("time_s")
)
write_summary

# %%
read_summary = (
    results_df.filter(pl.col("operation") == "read")
    .select(["format", "time_s", "throughput_M_rows_s"])
    .sort("time_s")
)
read_summary

# %%
columnar_summary = (
    results_df.filter(pl.col("operation") == "columnar_read")
    .select(["format", "time_s", "throughput_M_rows_s"])
    .sort("time_s")
)
columnar_summary

# %%
full_read = results_df.filter(pl.col("operation") == "read").select(["format", "time_s"])
col_read = (
    results_df.filter(pl.col("operation") == "columnar_read")
    .select(["format", "time_s"])
    .rename({"time_s": "columnar_time_s"})
)
projection_speedup = (
    full_read.join(col_read, on="format")
    .with_columns(speedup=pl.col("time_s") / pl.col("columnar_time_s"))
    .select(["format", "time_s", "columnar_time_s", "speedup"])
    .sort("speedup", descending=True)
)
projection_speedup

# %% [markdown]
# Memory-mapped reads still need to be materialized before they're useful;
# the raw `read_ipc` handle time below is excluded from the comparison and
# listed only so the gap to the materialized Feather read is visible.

# %%
print(f"Feather raw handle (memory-mapped, not materialized): {raw_handle_time:.4f} s")

# %% [markdown]
# ## 7. Visualisation

# %%
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Read time (lower is better)",
        "Write time (lower is better)",
        "File size (smaller is better)",
    ],
    horizontal_spacing=0.12,
)

read_data = results_df.filter(pl.col("operation") == "read").sort("time_s")
write_data = results_df.filter(pl.col("operation") == "write").sort("time_s")
size_data = results_df.filter(pl.col("operation") == "write").sort("size_mb")

fig.add_trace(
    go.Bar(
        y=read_data["format"].to_list(),
        x=read_data["time_s"].to_list(),
        orientation="h",
        marker_color=COLORS["blue"],
        text=[f"{t:.3f}s" for t in read_data["time_s"].to_list()],
        textposition="outside",
        cliponaxis=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        y=write_data["format"].to_list(),
        x=write_data["time_s"].to_list(),
        orientation="h",
        marker_color=COLORS["amber"],
        text=[f"{t:.3f}s" for t in write_data["time_s"].to_list()],
        textposition="outside",
        cliponaxis=False,
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Bar(
        y=size_data["format"].to_list(),
        x=size_data["size_mb"].to_list(),
        orientation="h",
        marker_color=COLORS["slate"],
        text=[f"{s:.1f} MB" for s in size_data["size_mb"].to_list()],
        textposition="outside",
        cliponaxis=False,
    ),
    row=1,
    col=3,
)

fig.update_xaxes(title_text="Seconds (log)", row=1, col=1, type="log")
fig.update_xaxes(title_text="Seconds (log)", row=1, col=2, type="log")
fig.update_xaxes(title_text="MB", row=1, col=3)

_scale_word = {"S": "Small", "M": "Medium", "L": "Large"}.get(ACTIVE_SCALE, ACTIVE_SCALE)
fig.update_layout(
    title_text=f"File-Format Comparison ({_scale_word} scale, {total_rows:,} rows)",
    height=400,
    showlegend=False,
    paper_bgcolor=COLORS["bg_light"],
    plot_bgcolor=COLORS["bg_light"],
    # Wider right margin so 'XX.X MB' / 'X.XXs' value labels don't crop.
    margin=dict(l=60, r=80, t=70, b=50),
)

fig.show()

# %% [markdown]
# ## Key Takeaways
#
# - **Memory-mapping is not a read.** The Feather raw handle returns in
#   microseconds because no data has been touched. The materialized read is
#   the apples-to-apples comparison.
# - **Columnar projection multiplies wins.** Reading two columns versus all
#   nine is roughly proportional to the column-count ratio for Parquet and
#   Feather; CSV barely benefits because every row must be parsed in full.
# - **Compression collapses on disk, not in RAM.** Parquet writes the panel
#   at roughly a quarter of the CSV size; the in-memory footprint after
#   read-back is identical because both formats land in Arrow buffers.
# - **HDF5 fixed format is single-shot.** It can read or write the entire
#   panel but offers no column projection.
#
# ### Format Picks
#
# - **Long-term storage, cloud, cross-language**: Parquet.
# - **Local interchange between Python tools**: Feather.
# - **Legacy scientific Python pipelines that need append**: HDF5.
# - **Human inspection or small exports**: CSV.
#
# ### Cross-References
#
# - **Database engines** for the same panel: `21_storage_benchmark_database`.
# - **Daily data lifecycle on Parquet**: `19_incremental_updates`.
# - **Library-level storage primitives**: `18_data_management`.

# %%
save_benchmark_results(results, "formats")
