"""Sample real registry.db data into test intermediates.

Copies a representative subset from each case study's production registry
into the test-data repo. This gives insight/synthesis/strategy_analysis
notebooks real data to work with in CI.

Sampling strategy:
- Model-side tables (training_runs, prediction_sets, prediction_metrics,
  fold_metrics): copied in full — small enough.
- Backtest tables: top N per (family × stage) by Sharpe, plus ALL holdout
  backtests, plus the COMPLETE cost_sensitivity and risk_overlay set of every
  prediction those two rules retain. Includes corresponding
  backtest_fold_metrics.

The last rule matters: strategy-analysis notebooks plan the full declared grid
for the prediction they select and assert its exact row count, so a partially
sampled downstream set fails a contract production satisfies.

Usage:
    uv run python tests/sample_registry_for_tests.py
    uv run python tests/sample_registry_for_tests.py --output ~/ml4t/test-data/intermediates

Writes to: <--output>/{cs}/run_log/registry.db
"""

import argparse
import contextlib
import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CODE_CS_DIR = REPO_ROOT / "case_studies"

TEST_DATA_ROOT = Path.home() / "ml4t" / "test-data"
DEFAULT_INTERMEDIATES_DIR = TEST_DATA_ROOT / "intermediates"

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

# Prediction hashes a shipped notebook names as a literal, per case study. These
# need more than a registry row: the notebook checks that
# run_log/predictions/<hash>/predictions.parquet exists and reads it. The
# registry sample carries the row, the fixture tree carried no artifact, and
# 26_mlops_governance/02_online_drift_detection failed on whichever of the two
# was missing. Keep a hash here for as long as a notebook pins it.
# The identity is declared here rather than read from the production registry,
# because the notebooks do not resolve a hash on its own: they filter on family,
# label, config_name and split. Comparing the fixture against production would
# pass when production itself no longer satisfies those predicates, or when both
# sides return nothing. Declaring it makes the sampler fail on the day a pinned
# hash stops meaning what the notebook expects.
PINNED_PREDICTION_HASHES = {
    "us_equities_panel": {
        # 26/02_online_drift_detection OLS_PREDICTION_HASH, and 26/03's FIXED_OLS
        "f9e84a32a9f0": ("linear", "fwd_ret_1d", "ols", "validation"),
        # 26/02_online_drift_detection RIDGE_PREDICTION_HASH
        "c0b36ffb8f51": ("linear", "fwd_ret_1d", "ridge_a10000000.0", "validation"),
        # 26/03_safe_model_rollout FIXED_RIDGE_PREDICTION_HASH
        "b381d21ffa4a": ("linear", "fwd_ret_1d", "ridge_a100000.0", "validation"),
    },
}

# Symbols to keep when subsampling a pinned prediction artifact. Production
# predictions run 100+ MB each, so the fixture carries a slice.
#
# This is a ceiling, not a target - the binding constraint is normally the
# fixture's own universe, which the pinned set is intersected against. A notebook
# ranking one stream into long and short books needs more than 2 * its top_k
# symbols *on every evaluated date*, or both legs hold the same assets and every
# spread is zero. 26/03_safe_model_rollout uses TOP_K = 100 in production and is
# overridden to 10 in tests/overrides.yaml for exactly this reason. Lower that
# override if a pinned artifact stops clearing 2 * TOP_K per date.
PINNED_PREDICTION_SYMBOLS = 200

# Where the universe a pinned artifact must live inside comes from, per case
# study, as paths relative to the test-data root. The market-data parquet is the
# one that matters - 26/02_online_drift_detection re-derives its liquid universe
# from ML4T_DATA_PATH, not from the intermediates - and the feature table is
# intersected too because a prediction symbol absent from it has no features to
# join. The symbol column is resolved from each file's schema rather than
# declared: this market-data parquet still carries the legacy `ticker`, and a
# canonical snapshot using `symbol` should regenerate rather than fail.
#
# The intersection is against the *whole* file, not against the notebook's ADV
# window. Reproducing get_liquid_universe here would couple the sampler to one
# notebook's liquidity rule and go stale the moment that rule changed; the
# breadth floor inside the notebook is what catches a universe that survives
# this check and collapses at the join.
PINNED_UNIVERSE_SOURCES = {
    "us_equities_panel": [
        "data/equities/market/us_equities/us_equities.parquet",
        "intermediates/us_equities_panel/features/financial.parquet",
    ],
}

# Precedence matches data/equities/loader.py:69, which takes `ticker` when both
# columns exist. Preferring `symbol` here would build the fixture universe from a
# different column than the notebook loads it from.
SYMBOL_COLUMN_CANDIDATES = ("ticker", "symbol")

# The realized outcome a prediction artifact is scored against. Pinned artifacts
# must agree on it row for row, not just on their keys - see
# preflight_pinned_predictions.
PREDICTION_TARGET_COLUMN = "actual"


def _copy_rows(src, dst, table: str, rows: list) -> int:
    """Insert rows into dst table with proper column quoting."""
    if not rows:
        return 0
    cols = [d[0] for d in src.execute(f"SELECT * FROM {table} LIMIT 1").description]
    quoted = [f'"{c}"' for c in cols]
    ph = ",".join(["?"] * len(cols))
    dst.executemany(f"INSERT OR IGNORE INTO {table} ({','.join(quoted)}) VALUES ({ph})", rows)
    return len(rows)


def rejected_output_root(intermediates_dir: Path) -> str | None:
    """Return why this output root must not be written to, or None.

    Each destination is unlinked before its source is opened, and a production
    registry.db is 43-180 MB and gitignored, so a run pointed at a case-study tree
    destroys the results SSOT with nothing to restore it from.

    Two rules. The root may not be, or sit inside, any directory named
    ``case_studies``: in a worktree ``CODE_CS_DIR`` is the worktree's own tree while
    the canonical registries are the ones in ~/ml4t/code, so path equality alone
    would let a run write over them. And no destination may land on a production
    registry once every symlink along it is followed - the per-agent worktree setup
    symlinks each case study's ``run_log`` to the canonical one precisely so the
    results source of truth is shared, which makes a symlinked destination the
    normal case here rather than an exotic one. Both are checked before anything is
    created or removed.
    """
    resolved = intermediates_dir.resolve()
    if any(part.name == "case_studies" for part in (resolved, *resolved.parents)):
        return (
            f"{resolved} is inside a case_studies tree, where each destination is a "
            "production registry.db that this script unlinks before reading"
        )
    for cs_id in CASE_STUDY_IDS:
        src_db = (CODE_CS_DIR / cs_id / "run_log" / "registry.db").resolve()
        dst_db = (resolved / cs_id / "run_log" / "registry.db").resolve()
        if dst_db == src_db:
            return f"{resolved} resolves onto the source registry at {src_db}"
        if any(part.name == "case_studies" for part in dst_db.parents):
            return (
                f"{resolved}/{cs_id}/run_log resolves into a case_studies tree "
                f"({dst_db}), whose registry this script unlinks before reading"
            )
    return None


def sample_registry(cs_id: str, intermediates_dir: Path = DEFAULT_INTERMEDIATES_DIR) -> dict:
    """Sample from production registry into test intermediates. Returns stats."""
    src_db = CODE_CS_DIR / cs_id / "run_log" / "registry.db"
    if not src_db.exists():
        return {"status": "SKIP", "reason": "no source registry.db"}
    if reason := rejected_output_root(intermediates_dir):
        raise ValueError(
            f"Refusing to sample {cs_id}: {reason}. Point --output at the test-data "
            "repo's intermediates/ directory."
        )

    dst_dir = intermediates_dir / cs_id / "run_log"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_db = dst_dir / "registry.db"

    # Remove old DB to start fresh. The artifact dirs go too: they are keyed by
    # backtest_hash, so a re-sample that changes hashes would otherwise leave the
    # previous generation's directories behind alongside the new ones.
    dst_db.unlink(missing_ok=True)
    shutil.rmtree(dst_dir / "backtest", ignore_errors=True)

    src = sqlite3.connect(str(src_db))
    try:
        dst = sqlite3.connect(str(dst_db))
        try:
            stats = _populate_sample_db(src, dst, dst_db)
        finally:
            dst.close()
    finally:
        src.close()

    sampled = stats.pop("sampled_hashes", set())
    artifacts = _copy_backtest_artifacts(src_db.parent, dst_dir, sampled)
    stats["backtest_artifact_dirs"] = artifacts["copied"]
    stats["backtest_artifacts_missing_dir"] = artifacts["missing_dir"]
    stats["backtest_artifacts_missing_returns"] = artifacts["missing_returns"]
    return stats


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

    # 3c. Seed the FULL backtest grid (every stage) for each label's cohort leader -
    # the frozen carrier a strategy-analysis notebook resolves via
    # cohort_metrics(cohort_type='stagelabel', stage='signal'), same query
    # `_baseline_leaders`-style notebooks use. Top-N-per-(family, stage) does not
    # guarantee this specific prediction survives sampling: it may not dominate its
    # family's |Sharpe| bucket in every stage. A notebook that pins an exact row
    # count for the leader's own allocation/cost/risk grid then fails against an
    # incomplete sample even though production satisfies it - the fixture would be
    # testing its own sampling artifact, not the notebook.
    leader_sql = """
        SELECT DISTINCT b.prediction_hash
        FROM cohort_metrics c
        JOIN backtest_runs b ON b.backtest_hash = c.leader_hash
        WHERE c.cohort_type = 'stagelabel' AND c.stage = 'signal'
    """
    # Registries predating cohort_metrics have no table to read; that is the only
    # condition this step tolerates. Catching OperationalError outright would also
    # swallow a schema drift, a locked database or a typo in leader_sql and report a
    # successful sample built without any cohort leader.
    src_has_cohort_metrics = (
        src.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cohort_metrics'"
        ).fetchone()
        is not None
    )
    leader_prediction_hashes: set = set()
    if src_has_cohort_metrics:
        leader_prediction_hashes = {row[0] for row in src.execute(leader_sql).fetchall()}
    if leader_prediction_hashes:
        ph_list = list(leader_prediction_hashes)
        full_grid_sql = """
            SELECT backtest_hash FROM backtest_runs
            WHERE prediction_hash IN ({placeholders})
        """
        for i in range(0, len(ph_list), 500):
            batch = ph_list[i : i + 500]
            sql = full_grid_sql.format(placeholders=",".join(["?"] * len(batch)))
            for row in src.execute(sql, batch).fetchall():
                sampled_bt_hashes.add(row[0])

    # 3d. Complete the downstream surface of every prediction sampled so far.
    #
    # Top-N-per-(family, stage) slices across predictions, so it can retain a
    # prediction's signal and allocation rows while keeping only a few of its
    # cost_sensitivity and risk_overlay rows. A strategy-analysis notebook plans
    # the FULL declared grid for whichever prediction it selects (all 14 fixed
    # risk controls, the whole cost grid) and asserts the exact row count, so a
    # partial set fails a contract the production registry satisfies - the
    # fixture would be testing its own sampling artifact. Whichever prediction
    # survives sampling must therefore bring its complete downstream surface.
    # Includes 'allocation' alongside cost_sensitivity/risk_overlay: a portfolio-
    # management notebook that pins an exact allocation-grid row count for one
    # prediction needs the same completeness guarantee those two stages already get.
    downstream_sql = """
        SELECT backtest_hash FROM backtest_runs
        WHERE stage IN ('allocation', 'cost_sensitivity', 'risk_overlay')
          AND prediction_hash IN (
              SELECT DISTINCT prediction_hash FROM backtest_runs
              WHERE backtest_hash IN ({placeholders})
          )
    """
    seed_hashes = list(sampled_bt_hashes)
    for i in range(0, len(seed_hashes), 500):
        batch = seed_hashes[i : i + 500]
        sql = downstream_sql.format(placeholders=",".join(["?"] * len(batch)))
        for row in src.execute(sql, batch).fetchall():
            sampled_bt_hashes.add(row[0])

    stats["backtest_runs_sampled"] = len(sampled_bt_hashes)
    stats["sampled_hashes"] = sampled_bt_hashes

    # 3d. Copy sampled backtest data (runs, metrics, fold_metrics)
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

        # 3e. Copy cohort_metrics rows whose leader_hash survived sampling. A
        # strategy-analysis/portfolio/cost/risk notebook resolves its frozen carrier
        # via cohort_metrics(cohort_type='stagelabel'|'label', ...) JOIN backtest_runs
        # ON leader_hash - an empty table here makes that JOIN return nothing and every
        # such notebook raise "no frozen carrier" regardless of how complete the
        # backtest_runs sample is. Filtering by leader_hash membership in the sample
        # is sufficient and correct: a row whose leader was not sampled would fail the
        # same JOIN downstream anyway, so it is dropped exactly like an FK would.
        has_cohort_metrics = (
            src.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cohort_metrics'"
            ).fetchone()
            is not None
        )
        count = 0
        if has_cohort_metrics:
            for i in range(0, len(hash_list), batch_size):
                batch = hash_list[i : i + batch_size]
                placeholders = ",".join(["?"] * len(batch))
                rows = src.execute(
                    f"SELECT * FROM cohort_metrics WHERE leader_hash IN ({placeholders})",
                    batch,
                ).fetchall()
                count += _copy_rows(src, dst, "cohort_metrics", rows)
        stats["cohort_metrics"] = count

    dst.commit()

    stats["file_size_kb"] = dst_db.stat().st_size // 1024
    stats["status"] = "OK"
    return stats


# Per-backtest artifact files a downstream notebook reads by hash. daily_returns is
# what paired and cohort uncertainty are computed from; spec.json is the run's own
# provenance. Anything larger stays out of the fixture.
_BACKTEST_ARTIFACTS = ("daily_returns.parquet", "spec.json")


def _copy_backtest_artifacts(src_run_log: Path, dst_run_log: Path, hashes: set) -> dict:
    """Copy each sampled backtest's artifact dir next to the sampled rows.

    A registry row whose ``run_log/backtest/<hash>/`` is absent is worse than a
    missing row: selection still reaches it and the read fails downstream, far from
    the cause. Nothing in the repo used to place these, so the fixture's rows and its
    artifact dirs were kept in step by hand and drifted apart whenever the registry
    was re-sampled alone.

    Returns copied/missing counts. A hash whose source dir or ``daily_returns.parquet``
    is absent is reported rather than skipped in silence - swallowing it here recreates
    the same read-fails-far-from-the-cause shape one layer up.
    """
    src_bt = src_run_log / "backtest"
    if not src_bt.is_dir():
        return {"copied": 0, "missing_dir": len(hashes), "missing_returns": 0}
    dst_bt = dst_run_log / "backtest"
    copied = 0
    missing_dir = 0
    missing_returns = 0
    for backtest_hash in hashes:
        src_dir = src_bt / backtest_hash
        if not src_dir.is_dir():
            missing_dir += 1
            continue
        dst_dir = dst_bt / backtest_hash
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in _BACKTEST_ARTIFACTS:
            src_file = src_dir / name
            if src_file.is_file():
                shutil.copy2(src_file, dst_dir / name)
            elif name == "daily_returns.parquet":
                missing_returns += 1
        copied += 1
    return {"copied": copied, "missing_dir": missing_dir, "missing_returns": missing_returns}


def _copy_rows_onto_existing_schema(src, dst, table: str, where_col: str, value) -> int:
    """Copy matching rows across schemas that have drifted apart.

    A fixture registry written before a column was added to the production
    schema is still usable, so this projects onto the columns the destination
    actually has rather than failing on the ones it does not.
    """
    dst_cols = [row[1] for row in dst.execute(f"PRAGMA table_info({table})")]
    src_cols = {row[1] for row in src.execute(f"PRAGMA table_info({table})")}
    shared = [c for c in dst_cols if c in src_cols]
    if not shared:
        return 0
    quoted = ",".join(f'"{c}"' for c in shared)
    rows = src.execute(f"SELECT {quoted} FROM {table} WHERE {where_col} = ?", (value,)).fetchall()
    if not rows:
        return 0
    ph = ",".join(["?"] * len(shared))
    before = dst.total_changes
    dst.executemany(f"INSERT OR IGNORE INTO {table} ({quoted}) VALUES ({ph})", rows)
    # Rows the destination already had are ignored, not overwritten, so report
    # what the database actually inserted rather than what was offered to it.
    return dst.total_changes - before


def _pinned_prediction_universe(
    src_pred_dir: Path, hashes: list[str], universe_sources: list[Path]
) -> list[str]:
    """One universe shared by every pinned artifact and by the fixture's own data.

    Two constraints, and missing either one produces a fixture that passes while
    measuring nothing.

    Every pinned artifact must carry the *same* symbols, or a notebook joining two
    prediction streams compares different universes. Reading the universe off
    whichever fixture artifact sorted first did not give that: the non-pinned
    us_equities_panel artifacts carry 8 symbols against the OLS artifact's 50,
    sharing 2.

    And those symbols must be ones the fixture's own data covers. The consuming
    notebooks re-derive a liquid universe from the market-data parquet and
    intersect, so a set chosen only from production predictions survives the
    first check and collapses at the second: truncating the production
    intersection alphabetically gave 50 symbols that met the 56-symbol fixture
    universe in AAL and AAPL alone, which would have left two assets per date and
    made every cross-sectional correlation +/-1 by construction.

    Every source is required. Skipping a missing one would silently fall back to
    the alphabetical truncation that produced exactly that fixture.
    """
    import polars as pl

    common: set[str] | None = None
    for prediction_hash in hashes:
        src = src_pred_dir / prediction_hash / "predictions.parquet"
        if not src.exists():
            continue
        symbols = set(pl.read_parquet(src, columns=["symbol"])["symbol"].unique().to_list())
        common = symbols if common is None else (common & symbols)
    if not common:
        raise ValueError(
            f"Pinned prediction hashes {hashes} share no symbols in {src_pred_dir}; "
            "a fixture built from them would compare different universes."
        )
    for path in universe_sources:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing, and it declares part of the universe the consuming "
                f"notebooks join against. Continuing without it would pick symbols "
                f"alphabetically and leave the notebooks with almost none of them."
            )
        schema = pl.read_parquet_schema(path)
        column = next((c for c in SYMBOL_COLUMN_CANDIDATES if c in schema), None)
        if column is None:
            raise ValueError(
                f"{path} carries none of {SYMBOL_COLUMN_CANDIDATES}, so the universe it "
                f"declares cannot be read."
            )
        covered = set(pl.read_parquet(path, columns=[column])[column].unique().to_list())
        common &= covered
        if not common:
            raise ValueError(
                f"No pinned prediction symbol survives the intersection with {path}; "
                "the notebooks join against it and would be left with nothing."
            )
    return sorted(common)[:PINNED_PREDICTION_SYMBOLS]


def ensure_pinned_predictions(cs_id: str, intermediates_dir: Path) -> dict:
    """Give every pinned hash both a registry row and a materialized artifact.

    Idempotent, and safe to run against a fixture the sampler did not just write.
    Registry rows are additive; the pinned artifacts are rewritten every time,
    because they only mean anything as a set that shares one universe and an
    artifact left over from an earlier universe would silently break that.
    """
    import polars as pl

    pinned = PINNED_PREDICTION_HASHES.get(cs_id, {})
    hashes = list(pinned)
    if not hashes:
        return {"pinned": 0}
    src_db = CODE_CS_DIR / cs_id / "run_log" / "registry.db"
    dst_db = intermediates_dir / cs_id / "run_log" / "registry.db"
    if not src_db.exists() or not dst_db.exists():
        return {"pinned": 0, "reason": "no registry"}

    src_pred_dir = CODE_CS_DIR / cs_id / "run_log" / "predictions"
    dst_pred_dir = intermediates_dir / cs_id / "run_log" / "predictions"
    parent_resolved = dst_pred_dir.parent.resolve()
    if any(part.name == "case_studies" for part in (parent_resolved, *parent_resolved.parents)):
        raise ValueError(
            f"{cs_id}: {dst_pred_dir} sits under {parent_resolved}, inside a case_studies "
            f"tree. Refusing to create prediction directories in production."
        )
    dst_pred_dir.mkdir(parents=True, exist_ok=True)

    # The declared sources sit under the test-data root, of which the fixture
    # intermediates are one subdirectory. Deriving that root by walking up one
    # level only holds when the output is the repo's own intermediates/, so say
    # so rather than silently reading a sibling of some other directory.
    if intermediates_dir.name != "intermediates":
        raise ValueError(
            f"{intermediates_dir} is not named 'intermediates', so the test-data root the "
            f"universe sources are relative to cannot be derived from it. Point --output at "
            f"<test-data-root>/intermediates."
        )
    test_data_root = intermediates_dir.parent
    universe_sources = [test_data_root / rel for rel in PINNED_UNIVERSE_SOURCES.get(cs_id, [])]
    if not universe_sources:
        raise ValueError(
            f"{cs_id} pins prediction hashes but declares no PINNED_UNIVERSE_SOURCES. "
            "Without one the pinned symbols need not be ones the notebooks can join to."
        )
    universe = _pinned_prediction_universe(src_pred_dir, hashes, universe_sources)
    rows_added = 0
    artifacts_written = 0
    min_per_date = len(universe)
    prepared: list = []
    src = sqlite3.connect(str(src_db))
    dst = sqlite3.connect(str(dst_db))
    try:
        for prediction_hash, expected_identity in pinned.items():
            training_hash = src.execute(
                "SELECT training_hash FROM prediction_sets WHERE prediction_hash = ?",
                (prediction_hash,),
            ).fetchone()
            if training_hash is None:
                raise ValueError(
                    f"{cs_id}: pinned prediction hash {prediction_hash} is not in the "
                    f"production registry. A notebook pins a hash nothing produces."
                )
            for table, column, value in (
                ("training_runs", "training_hash", training_hash[0]),
                ("prediction_sets", "prediction_hash", prediction_hash),
                ("prediction_metrics", "prediction_hash", prediction_hash),
            ):
                rows_added += _copy_rows_onto_existing_schema(src, dst, table, column, value)

            # INSERT OR IGNORE leaves a conflicting destination row in place, so
            # asking whether the join returns *something* would pass on a stale
            # row. Check both registries against the declared identity: the
            # destination because that is what the notebook reads, and the source
            # because a production registry that has drifted would otherwise be
            # copied faithfully into a fixture the notebook still rejects.
            identity_sql = (
                "SELECT tr.family, tr.label, tr.config_name, ps.split "
                "FROM training_runs tr JOIN prediction_sets ps "
                "ON tr.training_hash = ps.training_hash WHERE ps.prediction_hash = ?"
            )
            for label, conn in (("production", src), ("fixture", dst)):
                got = conn.execute(identity_sql, (prediction_hash,)).fetchone()
                if got != expected_identity:
                    raise RuntimeError(
                        f"{cs_id}: the {label} registry resolves {prediction_hash} to {got}, "
                        f"not the declared {expected_identity}. The consuming notebook filters "
                        f"on exactly these columns, so it would reject this hash."
                    )

            src_parquet = src_pred_dir / prediction_hash / "predictions.parquet"
            if not src_parquet.exists():
                raise FileNotFoundError(
                    f"{cs_id}: pinned prediction hash {prediction_hash} has a registry row "
                    f"but no artifact at {src_parquet}"
                )
            frame = pl.read_parquet(src_parquet).filter(pl.col("symbol").is_in(universe))
            if frame.is_empty():
                raise ValueError(
                    f"{cs_id}: {prediction_hash} carries no rows for the shared universe"
                )
            dst_parquet = dst_pred_dir / prediction_hash / "predictions.parquet"
            # Already resolved and rejected in the preflight; mkdir is safe here.
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            per_date = frame.group_by("timestamp").agg(pl.col("symbol").n_unique().alias("n"))
            min_per_date = min(min_per_date, int(per_date["n"].min()))
            prepared.append((dst_parquet, frame))

        # Nothing is replaced until every hash has cleared its registry identity
        # and produced a frame. The artifacts are only meaningful as a set that
        # shares one universe, and a raise partway through the loop would have
        # rewritten some of them and left the rest at the previous vintage - a
        # state the database transaction rolling back does not undo.
        staged: list[tuple[Path, Path]] = []
        try:
            for dst_parquet, frame in prepared:
                tmp = dst_parquet.with_suffix(".parquet.tmp")
                # Registered before the write, so a partial file left by a failed
                # write is still cleaned up.
                staged.append((tmp, dst_parquet))
                frame.write_parquet(tmp)
        except BaseException:
            for tmp, _ in staged:
                tmp.unlink(missing_ok=True)
            raise
        for tmp, dst_parquet in staged:
            tmp.replace(dst_parquet)
            artifacts_written += 1
        dst.commit()
    finally:
        dst.close()
        src.close()
    return {
        "pinned": len(hashes),
        "pinned_rows": rows_added,
        "pinned_artifacts": artifacts_written,
        "pinned_symbols": len(universe),
        "pinned_min_symbols_per_date": min_per_date,
    }


def preflight_pinned_predictions(cs_id: str, intermediates_dir: Path) -> None:
    """Decide everything about a case study's pinned hashes before writing anything.

    Raises rather than returning a verdict, so a caller that skips it still hits
    the same failures - just later, once the destination has been mutated.
    """
    pinned = PINNED_PREDICTION_HASHES.get(cs_id, {})
    if not pinned:
        return
    if intermediates_dir.name != "intermediates":
        raise ValueError(
            f"{intermediates_dir} is not named 'intermediates', and {cs_id} pins prediction "
            f"hashes whose universe sources are declared relative to the test-data root. "
            f"Point --output at <test-data-root>/intermediates."
        )
    sources = PINNED_UNIVERSE_SOURCES.get(cs_id)
    if not sources:
        raise ValueError(
            f"{cs_id} pins prediction hashes but declares no PINNED_UNIVERSE_SOURCES. "
            "Without one the pinned symbols need not be ones the notebooks can join to."
        )
    src_db = CODE_CS_DIR / cs_id / "run_log" / "registry.db"
    if not src_db.exists():
        return
    src_pred_dir = CODE_CS_DIR / cs_id / "run_log" / "predictions"
    # Raises on a missing source, an unreadable schema, or an empty intersection.
    universe = _pinned_prediction_universe(
        src_pred_dir, list(pinned), [intermediates_dir.parent / rel for rel in sources]
    )
    identity_sql = (
        "SELECT tr.family, tr.label, tr.config_name, ps.split "
        "FROM training_runs tr JOIN prediction_sets ps "
        "ON tr.training_hash = ps.training_hash WHERE ps.prediction_hash = ?"
    )
    import polars as pl

    dst_pred_dir = intermediates_dir / cs_id / "run_log" / "predictions"
    src = sqlite3.connect(str(src_db))
    keys: list[set] = []
    try:
        for prediction_hash, expected_identity in pinned.items():
            got = src.execute(identity_sql, (prediction_hash,)).fetchone()
            if got != expected_identity:
                raise RuntimeError(
                    f"{cs_id}: the production registry resolves {prediction_hash} to {got}, "
                    f"not the declared {expected_identity}."
                )
            artifact = src_pred_dir / prediction_hash / "predictions.parquet"
            if not artifact.exists():
                raise FileNotFoundError(
                    f"{cs_id}: pinned prediction hash {prediction_hash} has a registry row "
                    f"but no artifact at {artifact}"
                )
            # resolve() works on a path that does not exist yet, so the
            # destination can be rejected without creating anything: a
            # `predictions/` symlinked into the canonical tree would otherwise
            # have the hash directory made there before the guard fired.
            resolved = (dst_pred_dir / prediction_hash).resolve()
            if any(part.name == "case_studies" for part in (resolved, *resolved.parents)):
                raise ValueError(
                    f"{cs_id}: {dst_pred_dir / prediction_hash} resolves to {resolved}, "
                    f"inside a case_studies tree. Refusing to write a production artifact."
                )
            # The outcome column travels with the key on purpose. Comparing keys
            # alone lets two streams carry different `actual` values for the same
            # (symbol, timestamp) - a different label, or the same label from a
            # different vintage - and the consuming notebooks would score model
            # error and realized return against different outcomes while every
            # breadth and key check passed. The identity assertion above pins each
            # hash to a declared label, so today this cannot diverge; it is checked
            # here because the pinned set is data and the next hash added to it
            # need not share the label.
            schema = pl.read_parquet_schema(artifact)
            if PREDICTION_TARGET_COLUMN not in schema:
                raise ValueError(
                    f"{cs_id}: pinned artifact {prediction_hash} has no "
                    f"'{PREDICTION_TARGET_COLUMN}' column, so the outcome the notebooks "
                    f"score against cannot be checked for agreement with the other "
                    f"pinned streams."
                )
            frame = pl.read_parquet(
                artifact, columns=["symbol", "timestamp", PREDICTION_TARGET_COLUMN]
            ).filter(pl.col("symbol").is_in(universe))
            null_targets = frame[PREDICTION_TARGET_COLUMN].null_count()
            if null_targets:
                raise ValueError(
                    f"{cs_id}: pinned artifact {prediction_hash} has {null_targets} null "
                    f"'{PREDICTION_TARGET_COLUMN}' values inside the shared universe. A null "
                    f"outcome drops silently out of every comparison that reaches it."
                )
            keys.append(set(map(tuple, frame.unique().rows())))
    finally:
        src.close()

    # A shared symbol set is not a shared cross-section: two streams can hold
    # different names on the same date and still pass every breadth check, which
    # would leave the drift statistics and the rollout returns scoring different
    # assets on that date.
    if any(k != keys[0] for k in keys[1:]):
        raise ValueError(
            f"{cs_id}: the pinned artifacts do not agree on their "
            f"(symbol, timestamp, {PREDICTION_TARGET_COLUMN}) rows, so the notebooks "
            f"comparing them would score different cross-sections, or the same "
            f"cross-section against different outcomes."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INTERMEDIATES_DIR,
        help=(
            "Fixture intermediates root to write (the test-data repo's "
            "intermediates/ directory). Default: ~/ml4t/test-data/intermediates"
        ),
    )
    args = parser.parse_args()
    intermediates_dir = args.output.expanduser().resolve()

    if reason := rejected_output_root(intermediates_dir):
        parser.error(
            f"--output is not usable: {reason}. Point it at the test-data repo's "
            "intermediates/ directory."
        )

    # sample_registry() unlinks and recreates each destination registry, and
    # us_equities_panel is sampled last, so anything raising during the loop
    # leaves earlier registries current and their prediction artifacts at the
    # previous vintage. Everything decidable from the sources alone is decided
    # here: layout, declarations, schemas, the universe intersection, the
    # production identities, and the artifacts.
    missing = [
        cs_id
        for cs_id in CASE_STUDY_IDS
        if not (CODE_CS_DIR / cs_id / "run_log" / "registry.db").exists()
    ]
    if missing:
        parser.error(
            f"No production registry.db for {', '.join(missing)} under {CODE_CS_DIR}. "
            f"Sampling the rest would refresh those fixtures and leave these at the previous "
            f"vintage, which is the mixed state the run is meant to avoid."
        )
    for cs_id in CASE_STUDY_IDS:
        try:
            preflight_pinned_predictions(cs_id, intermediates_dir)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            parser.error(str(exc))

    print(f"Sampling registries from {CODE_CS_DIR}")
    print(f"Writing to {intermediates_dir}")
    print(f"Top {TOP_N_PER_GROUP} backtests per (family × stage) + all holdout\n")

    total_size = 0
    not_refreshed: list[str] = []
    for cs_id in CASE_STUDY_IDS:
        print(f"--- {cs_id} ---")
        stats = sample_registry(cs_id, intermediates_dir)
        if stats["status"] != "OK":
            print(f"  {stats['status']}: {stats.get('reason', '')}")
            not_refreshed.append(cs_id)
            continue
        stats.update(ensure_pinned_predictions(cs_id, intermediates_dir))
        if stats.get("pinned"):
            # The per-date figure is the one that decides whether a ranked book
            # is degenerate, and it is a global minimum over every date in the
            # artifact - including sparse early years no notebook evaluates. Read
            # it against the window the consuming notebook actually uses.
            print(
                f"  {'pinned predictions':30s} {stats['pinned']:>6} "
                f"({stats['pinned_artifacts']} artifact(s) written, "
                f"{stats['pinned_symbols']} symbols, "
                f"min {stats['pinned_min_symbols_per_date']} per date)"
            )
        for table in [
            "training_runs",
            "prediction_sets",
            "prediction_metrics",
            "fold_metrics",
            "backtest_runs",
            "backtest_metrics",
            "backtest_fold_metrics",
            "cohort_metrics",
        ]:
            print(f"  {table:30s} {stats.get(table, 0):>6}")
        print(f"  {'backtest artifact dirs':30s} {stats.get('backtest_artifact_dirs', 0):>6}")
        print(f"  {'file size (KB)':30s} {stats['file_size_kb']:>6}")
        missing_dir = stats.get("backtest_artifacts_missing_dir", 0)
        missing_returns = stats.get("backtest_artifacts_missing_returns", 0)
        if missing_dir or missing_returns:
            print(
                f"  WARNING: {missing_dir} sampled hashes have no source artifact dir, "
                f"{missing_returns} have no daily_returns.parquet - a notebook that "
                f"selects one of them fails on the read, not on the sample"
            )
        total_size += stats["file_size_kb"]

    print(f"\nTotal registry size: {total_size} KB ({total_size / 1024:.1f} MB)")

    if not_refreshed:
        # A skipped case study leaves whatever registry the destination already
        # held, so exiting 0 here reports a refresh that did not happen and the
        # replay-only notebooks then read the previous vintage.
        print(
            f"\nERROR: {len(not_refreshed)} of {len(CASE_STUDY_IDS)} registries were not "
            f"refreshed: {', '.join(not_refreshed)}. Their production registry.db is missing "
            f"under {CODE_CS_DIR}; the fixture keeps whatever it held before this run."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
