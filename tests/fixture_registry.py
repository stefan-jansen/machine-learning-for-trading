"""Keep a fixture's ``run_log`` describing what the fixture actually ships.

The CI fixture registries were copied whole from production while the artifacts beside
them were subsampled, so they pin files the fixture has never held. This module holds the
operations that put a fixture registry back into agreement with its own tree; the callers
are :mod:`tests.generate_intermediates`, which prunes before it registers, and
``tests/reconcile_fixture_registry.py``, which reconciles a fixture already committed.

Imported by ``generate_intermediates.py``, which runs standalone without pytest, so
nothing here may import from ``conftest`` or from pytest.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

#: The directories under a case study whose parquets a training run pins by whole-file
#: sha256. ``utils.modeling`` builds ``input_data_spec.artifacts`` from exactly these:
#: ``financial`` and ``model_based`` from ``features/``, ``label`` and ``eval_label``
#: from ``labels/``.
PINNED_ARTIFACT_DIRS = ("features", "labels")

#: Every table that hangs off a training run, in the order they must be deleted so no
#: statement removes a row another still needs to find its own. Each entry is
#: ``(table, column, key)`` where *key* names which set of hashes the column is matched
#: against: ``training``, ``prediction`` or ``backtest``.
_CASCADE: tuple[tuple[str, str, str], ...] = (
    ("backtest_paired_metrics", "challenger_hash", "backtest"),
    ("backtest_paired_metrics", "benchmark_hash", "backtest"),
    ("cohort_metrics", "leader_hash", "backtest"),
    ("backtest_fold_metrics", "backtest_hash", "backtest"),
    ("backtest_metrics", "backtest_hash", "backtest"),
    ("backtest_runs", "backtest_hash", "backtest"),
    ("fold_metrics", "prediction_hash", "prediction"),
    ("prediction_metrics", "prediction_hash", "prediction"),
    ("prediction_coverage", "prediction_hash", "prediction"),
    ("official_population_members", "member_hash", "prediction"),
    ("candidate_set_members", "member_hash", "prediction"),
    ("holdout_staging", "holdout_prediction_hash", "prediction"),
    ("holdout_evaluations", "holdout_prediction_hash", "prediction"),
    ("prediction_sets", "prediction_hash", "prediction"),
    ("candidate_fold_completions", "training_hash", "training"),
    ("holdout_staging", "holdout_training_hash", "training"),
    ("holdout_evaluations", "holdout_training_hash", "training"),
    ("training_runs", "training_hash", "training"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shipped_artifact_shas(case_dir: Path) -> dict[str, str]:
    """Every sha256 the fixture ships under this case study, to the file carrying it.

    Keyed by digest rather than by artifact name on purpose. A training run pins
    ``label`` to ``labels/<its own label>.parquet`` and ``financial`` to
    ``features/financial.parquet``, so resolving a pin by name means reproducing that
    mapping here and going stale the day it changes. The question this module asks is
    narrower and name-free: does the fixture hold the bytes this run was fitted on.
    """
    shas: dict[str, str] = {}
    for subdir in PINNED_ARTIFACT_DIRS:
        for parquet in sorted((case_dir / subdir).glob("*.parquet")):
            shas[sha256_file(parquet)] = f"{subdir}/{parquet.name}"
    return shas


def pinned_input_shas(spec_json: str | dict | None) -> dict[str, str]:
    """The ``{artifact name: sha256}`` a training run's spec pins, hex and unprefixed.

    Mirrors ``case_studies/utils/registry/registration.py::_input_artifact_shas``, which
    is what the vintage guard compares against, and tolerates the same spec shapes it
    does: a spec that carries no ``computation.input_data_spec.artifacts`` block pins
    nothing and is never stale.
    """
    spec = spec_json
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except (TypeError, ValueError):
            return {}
    if not isinstance(spec, dict):
        return {}
    computation = spec.get("computation")
    if not isinstance(computation, dict):
        return {}
    input_data_spec = computation.get("input_data_spec")
    if not isinstance(input_data_spec, dict):
        return {}
    artifacts = input_data_spec.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}
    return {
        str(name): str(record["sha256"]).removeprefix("sha256:")
        for name, record in sorted(artifacts.items())
        if isinstance(record, dict) and record.get("sha256")
    }


def stale_training_runs(db: sqlite3.Connection, case_dir: Path) -> dict[str, dict[str, str]]:
    """Training runs fitted on input artifacts this fixture does not ship.

    Returns ``{training_hash: {artifact name: pinned sha}}`` for the pins that are
    absent, so a caller can say which file each row wanted rather than only how many
    rows it dropped.
    """
    shipped = set(shipped_artifact_shas(case_dir))
    stale: dict[str, dict[str, str]] = {}
    for training_hash, spec_json in db.execute(
        "SELECT training_hash, spec_json FROM training_runs"
    ):
        absent = {
            name: sha for name, sha in pinned_input_shas(spec_json).items() if sha not in shipped
        }
        if absent:
            stale[str(training_hash)] = absent
    return stale


def _existing_tables(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _in_batches(values: Iterable[str], size: int = 500):
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def delete_training_runs(db: sqlite3.Connection, training_hashes: set[str]) -> dict[str, int]:
    """Remove these training runs and everything that hangs off them.

    SQLite does not enforce the schema's foreign keys unless the connection asks it to,
    so deleting ``training_runs`` alone leaves prediction sets, metrics and backtests
    pointing at a row that is gone - which reads downstream as a corrupt registry rather
    than a pruned one. The cascade is spelled out in :data:`_CASCADE` and applied leaf
    first.
    """
    if not training_hashes:
        return {}
    tables = _existing_tables(db)
    prediction_hashes: set[str] = set()
    backtest_hashes: set[str] = set()
    for batch in _in_batches(training_hashes):
        marks = ",".join("?" * len(batch))
        prediction_hashes.update(
            str(row[0])
            for row in db.execute(
                f"SELECT prediction_hash FROM prediction_sets WHERE training_hash IN ({marks})",
                batch,
            )
        )
    if "backtest_runs" in tables:
        for batch in _in_batches(prediction_hashes):
            marks = ",".join("?" * len(batch))
            backtest_hashes.update(
                str(row[0])
                for row in db.execute(
                    f"SELECT backtest_hash FROM backtest_runs WHERE prediction_hash IN ({marks})",
                    batch,
                )
            )
    keys = {
        "training": training_hashes,
        "prediction": prediction_hashes,
        "backtest": backtest_hashes,
    }
    deleted: dict[str, int] = {}
    for table, column, key in _CASCADE:
        if table not in tables:
            continue
        hashes = keys[key]
        if not hashes:
            continue
        for batch in _in_batches(hashes):
            marks = ",".join("?" * len(batch))
            cursor = db.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", batch)
            deleted[table] = deleted.get(table, 0) + cursor.rowcount
    db.commit()
    return {table: count for table, count in deleted.items() if count}


def prune_stale_training_runs(case_dir: Path) -> dict:
    """Drop this fixture's training runs that were fitted on artifacts it no longer has.

    Called by the fixture generator immediately before its first registering stage. By
    then stages 01-05 have rewritten ``features/`` and ``labels/``, so a row pinning an
    older vintage is describing a population the fixture no longer holds - and the
    vintage guard in ``register_training_run`` refuses to let the run about to start
    join it (ml4t/agent-workspace#1082). Declaring an artifact supersession is the wrong
    override: it would record that the fixture's reduced file deliberately replaces the
    production one inside one continuing population, and the two were never in the same
    population.

    Returns a summary with the rows dropped per table and one example of what was
    pinned, or ``{}`` when the case study has no registry yet.
    """
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.is_file():
        return {}
    db = sqlite3.connect(str(db_path))
    try:
        stale = stale_training_runs(db, case_dir)
        if not stale:
            return {"training_runs_pruned": 0}
        deleted = delete_training_runs(db, set(stale))
    finally:
        db.close()
    example_hash = sorted(stale)[0]
    return {
        "training_runs_pruned": len(stale),
        "deleted": deleted,
        "example": {"training_hash": example_hash, "absent_pins": stale[example_hash]},
    }


#: The tables a prediction set carries with it. Ordered parent first so a re-insert
#: never writes a child before the row its foreign key names.
_PREDICTION_TABLES: tuple[tuple[str, str], ...] = (
    ("training_runs", "training_hash"),
    ("prediction_sets", "prediction_hash"),
    ("prediction_metrics", "prediction_hash"),
    ("prediction_coverage", "prediction_hash"),
    ("fold_metrics", "prediction_hash"),
)


def _table_columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in db.execute(f"PRAGMA table_info({table})")]


def capture_backed_prediction_rows(db_path: Path, case_dir: Path) -> dict:
    """The registry rows describing prediction artifacts this fixture actually ships.

    A re-sample rewrites the registry from production and leaves ``run_log/predictions/``
    alone, so every artifact a generation wrote loses the row that named it: 6 directories
    on etfs, 10 on fx_pairs, 39 on sp500_equity_option_analytics, all addressable by
    nothing afterwards (ml4t/agent-workspace#1081). Captured before the rewrite and put
    back after, these rows keep the fixture's own artifacts reachable.

    Returns ``{table: (columns, rows)}``, empty when there is no registry yet.
    """
    if not db_path.is_file():
        return {}
    shipped = (
        {entry.name for entry in (case_dir / "run_log" / "predictions").iterdir() if entry.is_dir()}
        if (case_dir / "run_log" / "predictions").is_dir()
        else set()
    )
    if not shipped:
        return {}
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = _existing_tables(db)
        if "prediction_sets" not in tables:
            return {}
        prediction_hashes: set[str] = set()
        training_hashes: set[str] = set()
        for batch in _in_batches(sorted(shipped)):
            marks = ",".join("?" * len(batch))
            for prediction_hash, training_hash in db.execute(
                "SELECT prediction_hash, training_hash FROM prediction_sets "
                f"WHERE prediction_hash IN ({marks})",
                batch,
            ):
                prediction_hashes.add(str(prediction_hash))
                training_hashes.add(str(training_hash))
        if not prediction_hashes:
            return {}
        keys = {"training_hash": training_hashes, "prediction_hash": prediction_hashes}
        captured: dict = {}
        for table, column in _PREDICTION_TABLES:
            if table not in tables:
                continue
            columns = _table_columns(db, table)
            rows: list[tuple] = []
            for batch in _in_batches(sorted(keys[column])):
                marks = ",".join("?" * len(batch))
                rows.extend(db.execute(f"SELECT * FROM {table} WHERE {column} IN ({marks})", batch))
            if rows:
                captured[table] = (columns, rows, column, sorted(keys[column]))
        return captured
    finally:
        db.close()


def restore_backed_prediction_rows(db_path: Path, captured: dict) -> dict[str, int]:
    """Put captured rows back, replacing whatever the rebuilt registry holds for them.

    Replace rather than ignore: where production also carries the hash, its metrics
    describe the production artifact, and the file the fixture ships is the subsample.
    The row that names a shipped artifact has to be the one measured against it.
    """
    if not captured or not db_path.is_file():
        return {}
    db = sqlite3.connect(str(db_path))
    restored: dict[str, int] = {}
    try:
        tables = _existing_tables(db)
        for table, _ in _PREDICTION_TABLES:
            if table not in captured or table not in tables:
                continue
            columns, rows, key_column, keys = captured[table]
            present = set(_table_columns(db, table))
            shared = [name for name in columns if name in present]
            if not shared:
                continue
            for batch in _in_batches(keys):
                marks = ",".join("?" * len(batch))
                db.execute(f"DELETE FROM {table} WHERE {key_column} IN ({marks})", batch)
            index = {name: position for position, name in enumerate(columns)}
            quoted = ",".join(f'"{name}"' for name in shared)
            marks = ",".join("?" * len(shared))
            db.executemany(
                f"INSERT INTO {table} ({quoted}) VALUES ({marks})",
                [tuple(row[index[name]] for name in shared) for row in rows],
            )
            restored[table] = len(rows)
        db.commit()
    finally:
        db.close()
    return restored


#: The column carrying the realized outcome, in both conventions the fixture ships.
TARGET_COLUMNS = ("actual", "y_true")
ENTITY_COLUMNS = ("symbol", "product")


def panel_signature(frame, entity: str) -> tuple:
    """What makes two prediction artifacts the same cross-section.

    Height, entities and timestamps - not the scores, which differ by construction, and
    not the target, which is what a caller then checks agrees. Identifiers are
    stringified because they are only ever compared for equality and a column carrying
    nulls raises in ``sorted()``.
    """
    return (
        frame.height,
        tuple(sorted(map(str, frame[entity].unique().to_list()))),
        tuple(map(str, frame["timestamp"].unique().sort().to_list())),
    )


def choose_reference_panel(by_signature: dict, hash_of=lambda entry: entry[0]) -> tuple:
    """The panel a ``(split, label)`` group is seeded onto: most artifacts, then largest.

    Ties break on the lowest hash so the choice comes out the same on every
    regeneration. ``tests/fixtures/seed_results.py`` seeds its synthetic sets onto this
    panel, which is what makes them joinable with the copied artifacts rather than only
    with each other - so the rule has one implementation and both callers read it.

    *by_signature* maps a :func:`panel_signature` to its entries; *hash_of* reads the
    prediction hash out of one, because the two callers carry an entry as a tuple and as
    a mapping. Returns the winning ``(signature, entries)`` pair.
    """
    return min(
        by_signature.items(),
        key=lambda item: (-len(item[1]), -item[0][0], hash_of(item[1][0])),
    )


def prediction_panels(case_dir: Path) -> dict:
    """Per ``(split, label)``, the distinct cross-sections this fixture ships for it.

    ``us_equities_panel`` ``(validation, fwd_ret_1d)`` ships two: four ``gbm`` artifacts
    over 8 symbols storing the target as ``Float32``, and three ``linear`` ones over 56
    storing it as ``Float64``. They share 5,212 of 16,744 keys and disagree on the
    realized target by 2.6e-08 there, which is 260x the 1e-10 that
    ``14_latent_factors/09_case_study_insights::paired_daily_ic`` rejects a pair at
    (ml4t/agent-workspace#288). Which one a notebook lands on is decided by registry
    metrics, so this returns the panels and what separates them rather than a verdict.

    Returns ``{(split, label): {"panels": [...], "cross_panel_target_gap": float|None}}``
    where each panel carries its hashes, entity count, target dtype and a frame.
    """
    import polars as pl

    db_path = case_dir / "run_log" / "registry.db"
    predictions = case_dir / "run_log" / "predictions"
    if not db_path.is_file() or not predictions.is_dir():
        return {}
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT ps.prediction_hash, ps.split, tr.label, tr.family FROM prediction_sets ps "
            "JOIN training_runs tr ON tr.training_hash = ps.training_hash"
        ).fetchall()
    finally:
        db.close()

    groups: dict = {}
    for prediction_hash, split, label, family in rows:
        parquet = predictions / str(prediction_hash) / "predictions.parquet"
        if parquet.is_file():
            groups.setdefault((split, label), []).append((str(prediction_hash), family, parquet))

    result: dict = {}
    for key, members in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        by_signature: dict = {}
        for prediction_hash, family, parquet in sorted(members):
            frame = pl.read_parquet(parquet)
            entity = next((name for name in ENTITY_COLUMNS if name in frame.columns), None)
            target = next((name for name in TARGET_COLUMNS if name in frame.columns), None)
            if entity is None or target is None or "timestamp" not in frame.columns:
                continue
            keyed = frame.select([entity, "timestamp", target]).rename(
                {entity: "entity", target: "target"}
            )
            by_signature.setdefault(panel_signature(frame, entity), []).append(
                {
                    "prediction_hash": prediction_hash,
                    "family": family,
                    "entities": frame[entity].n_unique(),
                    "target_dtype": str(frame.schema[target]),
                    "frame": keyed,
                }
            )
        if not by_signature:
            continue
        reference_signature, _ = choose_reference_panel(
            by_signature, hash_of=lambda entry: entry["prediction_hash"]
        )
        panels = []
        for signature, entries in by_signature.items():
            panels.append(
                {
                    "hashes": [entry["prediction_hash"] for entry in entries],
                    "families": sorted({entry["family"] for entry in entries}),
                    "entities": entries[0]["entities"],
                    "target_dtypes": sorted({entry["target_dtype"] for entry in entries}),
                    "is_reference": signature == reference_signature,
                    "entries": entries,
                }
            )
        result[key] = {"panels": panels, "cross_panel_target_gap": _cross_panel_gap(panels)}
    return result


def _cross_panel_gap(panels: list) -> float | None:
    """The largest target disagreement between two panels, over the keys they share."""
    import itertools

    import polars as pl

    worst: float | None = None
    for left, right in itertools.combinations(panels, 2):
        a = left["entries"][0]["frame"].with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
        )
        b = right["entries"][0]["frame"].with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
        )
        joined = a.join(b, on=["entity", "timestamp"], how="inner")
        if joined.is_empty():
            continue
        gap = float(
            (joined["target"].cast(pl.Float64) - joined["target_right"].cast(pl.Float64))
            .abs()
            .max()
        )
        worst = gap if worst is None else max(worst, gap)
    return worst


def within_panel_target_gap(panel: dict) -> float:
    """The largest target disagreement between artifacts of one cross-section.

    Zero is the only acceptable value: these artifacts carry the same keys, so a
    non-zero gap means two of them are scored against different outcomes while every
    breadth and key check passes.
    """
    import polars as pl

    entries = panel["entries"]
    worst = 0.0
    for other in entries[1:]:
        joined = entries[0]["frame"].join(other["frame"], on=["entity", "timestamp"], how="inner")
        if joined.is_empty():
            continue
        worst = max(
            worst,
            float(
                (joined["target"].cast(pl.Float64) - joined["target_right"].cast(pl.Float64))
                .abs()
                .max()
            ),
        )
    return worst


def unbacktested_populations(case_dir: Path) -> list[dict]:
    """Populations in this fixture whose members carry no signal backtest.

    A population scopes what `14_backtest` ranks: with one in force the notebook reads
    `explorer.best(stage="signal", prediction_hashes=members)`, and with none it reads
    unscoped. So a fixture that publishes a population and no backtests for its members
    gives that read nothing, and before public #849 the first thing to touch the empty
    frame was a division. That is how it arrived: a stage-08 regeneration created two
    populations over 30 fresh prediction sets while every backtest in the fixture
    referenced predictions from two weeks earlier, and the next CI run divided by zero
    (ml4t/agent-workspace#1086).

    Generation cannot avoid it - the model stages declare a population and the backtest
    stages are numbers 14 and up, which `--through-stage 8` never reaches - so what a
    generation owes is to say it left the fixture in that state rather than to let the
    next CI run discover it.
    """
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.is_file():
        return []
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = _existing_tables(db)
        if not {"official_populations", "official_population_members"} <= tables:
            return []
        rows = db.execute(
            "SELECT p.population_hash, p.name, COUNT(m.member_hash) FROM official_populations p "
            "LEFT JOIN official_population_members m ON m.population_hash = p.population_hash "
            "GROUP BY p.population_hash, p.name"
        ).fetchall()
        unbacked = []
        for population_hash, name, members in rows:
            if not members:
                continue
            backtested = 0
            if "backtest_runs" in tables:
                backtested = db.execute(
                    "SELECT COUNT(DISTINCT m.member_hash) FROM official_population_members m "
                    "JOIN backtest_runs b ON b.prediction_hash = m.member_hash "
                    "WHERE m.population_hash = ? AND b.stage = 'signal'",
                    (population_hash,),
                ).fetchone()[0]
            if not backtested:
                unbacked.append(
                    {"name": name, "population_hash": population_hash, "members": members}
                )
        return unbacked
    finally:
        db.close()
