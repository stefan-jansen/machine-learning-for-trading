"""Sample real registry.db data into test intermediates.

Copies a representative subset from each case study's production registry
into the test-data repo. This gives insight/synthesis/strategy_analysis
notebooks real data to work with in CI.

Sampling strategy:
- Model-side tables (training_runs, prediction_sets, prediction_metrics,
  fold_metrics): copied in full — small enough.
- Backtest tables: top N per (family × stage) by Sharpe, plus ALL holdout
  backtests. Includes corresponding backtest_fold_metrics.

Usage:
    uv run python tests/sample_registry_for_tests.py

Writes to: ~/ml4t/test-data/intermediates/{cs}/run_log/registry.db
"""

import contextlib
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CODE_CS_DIR = REPO_ROOT / "case_studies"

TEST_DATA_ROOT = Path.home() / "ml4t" / "test-data"
INTERMEDIATES_DIR = TEST_DATA_ROOT / "intermediates"

CASE_STUDY_IDS = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]

# Keep top N backtests per (family, stage) by absolute Sharpe
TOP_N_PER_GROUP = 3


def _copy_rows(src, dst, table: str, rows: list) -> int:
    """Insert rows into dst table with proper column quoting."""
    if not rows:
        return 0
    cols = [d[0] for d in src.execute(f"SELECT * FROM {table} LIMIT 1").description]
    quoted = [f'"{c}"' for c in cols]
    ph = ",".join(["?"] * len(cols))
    dst.executemany(f"INSERT OR IGNORE INTO {table} ({','.join(quoted)}) VALUES ({ph})", rows)
    return len(rows)


def sample_registry(cs_id: str) -> dict:
    """Sample from production registry into test intermediates. Returns stats."""
    src_db = CODE_CS_DIR / cs_id / "run_log" / "registry.db"
    if not src_db.exists():
        return {"status": "SKIP", "reason": "no source registry.db"}

    dst_dir = INTERMEDIATES_DIR / cs_id / "run_log"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_db = dst_dir / "registry.db"

    # Remove old DB to start fresh
    dst_db.unlink(missing_ok=True)

    src = sqlite3.connect(str(src_db))
    try:
        dst = sqlite3.connect(str(dst_db))
        try:
            return _populate_sample_db(src, dst, dst_db)
        finally:
            dst.close()
    finally:
        src.close()


def _populate_sample_db(src, dst, dst_db) -> dict:
    stats: dict = {}

    # 1. Copy schema from source (dump CREATE statements)
    schema_sql = []
    for row in src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall():
        schema_sql.append(row[0])
    for sql in schema_sql:
        dst.execute(sql)

    # Also copy indexes
    for row in src.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
    ).fetchall():
        with contextlib.suppress(sqlite3.OperationalError):
            dst.execute(row[0])

    # 2. Copy model-side tables in full
    for table in ["training_runs", "prediction_sets", "prediction_metrics", "fold_metrics"]:
        rows = src.execute(f"SELECT * FROM {table}").fetchall()
        n = _copy_rows(src, dst, table, rows)
        stats[table] = n

    # 3. Sample backtests: top N per (family, stage) by |Sharpe|, plus all holdout
    # First, get sampled backtest hashes
    sampled_bt_hashes = set()

    # 3a. Top N per family × stage (validation backtests)
    top_n_sql = """
        WITH ranked AS (
            SELECT
                b.backtest_hash,
                b.stage,
                t.family,
                bm.sharpe,
                ROW_NUMBER() OVER (
                    PARTITION BY b.stage, t.family
                    ORDER BY ABS(bm.sharpe) DESC
                ) AS rn
            FROM backtest_runs b
            JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            WHERE p.split != 'holdout'
        )
        SELECT backtest_hash FROM ranked WHERE rn <= ?
    """
    for row in src.execute(top_n_sql, (TOP_N_PER_GROUP,)).fetchall():
        sampled_bt_hashes.add(row[0])

    # 3b. ALL holdout backtests
    holdout_sql = """
        SELECT b.backtest_hash
        FROM backtest_runs b
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        WHERE p.split = 'holdout'
    """
    for row in src.execute(holdout_sql).fetchall():
        sampled_bt_hashes.add(row[0])

    stats["backtest_runs_sampled"] = len(sampled_bt_hashes)

    # 3c. Copy sampled backtest data (runs, metrics, fold_metrics)
    if sampled_bt_hashes:
        hash_list = list(sampled_bt_hashes)
        batch_size = 500

        for table in ["backtest_runs", "backtest_metrics", "backtest_fold_metrics"]:
            count = 0
            for i in range(0, len(hash_list), batch_size):
                batch = hash_list[i : i + batch_size]
                placeholders = ",".join(["?"] * len(batch))
                rows = src.execute(
                    f"SELECT * FROM {table} WHERE backtest_hash IN ({placeholders})",
                    batch,
                ).fetchall()
                count += _copy_rows(src, dst, table, rows)
            stats[table] = count

    dst.commit()

    stats["file_size_kb"] = dst_db.stat().st_size // 1024
    stats["status"] = "OK"
    return stats


def main():
    print(f"Sampling registries from {CODE_CS_DIR}")
    print(f"Writing to {INTERMEDIATES_DIR}")
    print(f"Top {TOP_N_PER_GROUP} backtests per (family × stage) + all holdout\n")

    total_size = 0
    for cs_id in CASE_STUDY_IDS:
        print(f"--- {cs_id} ---")
        stats = sample_registry(cs_id)
        if stats["status"] != "OK":
            print(f"  {stats['status']}: {stats.get('reason', '')}")
            continue
        for table in [
            "training_runs",
            "prediction_sets",
            "prediction_metrics",
            "fold_metrics",
            "backtest_runs",
            "backtest_metrics",
            "backtest_fold_metrics",
        ]:
            print(f"  {table:30s} {stats.get(table, 0):>6}")
        print(f"  {'file size (KB)':30s} {stats['file_size_kb']:>6}")
        total_size += stats["file_size_kb"]

    print(f"\nTotal registry size: {total_size} KB ({total_size / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
