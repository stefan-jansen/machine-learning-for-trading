from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from tests.pm_helpers import collect_chapter_notebooks, get_overrides

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / ".claude" / "work" / "notebook_testing" / "catalog.sqlite"

CASE_STUDIES = [
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

TRACKER_SCHEMA_COMPLETE_CHAPTERS = {1, 2, 8, 9, 10, 11, 12, 13, 14, 15}
TRACKER_SCHEMA_IN_PROGRESS_CHAPTERS = {3, 4, 5, 6, 7, 16, 17, 18, 19, 20}
TRACKER_SCHEMA_CASE_STUDIES = {
    "etfs": "complete",
    "fx_pairs": "complete",
    "crypto_perps_funding": "in_progress",
    "cme_futures": "pending",
    "nasdaq100_microstructure": "pending",
    "sp500_equity_option_analytics": "pending",
    "sp500_options": "pending",
    "us_equities_panel": "pending",
    "us_firm_characteristics": "pending",
}

CRYPTO_REPRO_NOTE = (
    "Current Binance public downloads no longer reproduce MATICUSDT OHLCV. "
    "Crypto case study requires full refreshed model reruns and explicit old-vs-new "
    "comparison against the dev registry."
)

HEAVY_KEYWORDS = {
    "timegan",
    "tailgan",
    "sigcwgan",
    "diffusion",
    "great",
    "dp_gan",
    "patchtst",
    "transformer",
    "lstm",
    "autoencoder",
    "finbert",
    "bert",
    "ner",
    "xgboost",
    "lightgbm",
    "catboost",
    "rl",
    "deepm",
    "gan",
    "backtest_sweep",
}

GPU_KEYWORDS = {"gpu", "cuda", "torch", "tensorflow", "trainer"}

DATA_HINTS = {
    "etfs": ("etf", "etfs", "spy"),
    "crypto": ("crypto", "perp", "perps", "funding", "premium", "binance"),
    "fx": ("fx", "oanda", "eur_usd"),
    "futures": ("future", "futures", "cme", "databento", "glbx"),
    "us_equities": ("equities", "crsp", "stocks", "ticker", "secedgar"),
    "options": ("option", "options", "greeks", "iv"),
    "nasdaq_itch": ("itch", "nasdaq100", "algoseek", "taq", "lob", "iex"),
    "macro": ("macro", "fred", "yield", "calendar"),
    "text": ("text", "news", "filing", "sentiment", "word2vec"),
    "synthetic": ("synthetic", "simulation", "regime", "scenario"),
}


@dataclass(slots=True)
class NotebookEntry:
    path: str
    notebook_key: str
    notebook_type: str
    chapter: int | None
    case_study_id: str | None
    stage: str | None
    stage_order: int | None
    title: str
    resource_profile: str
    execution_lane: str
    parallel_safe: int
    worker_slots: int
    gpu_required: int
    has_parameters_cell: int
    parameter_source: str
    default_timeout_seconds: int
    inputs_hint: str
    override_json: str


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect_catalog(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS notebooks (
            path TEXT PRIMARY KEY,
            notebook_key TEXT NOT NULL UNIQUE,
            notebook_type TEXT NOT NULL,
            chapter INTEGER,
            case_study_id TEXT,
            stage TEXT,
            stage_order INTEGER,
            title TEXT NOT NULL,
            resource_profile TEXT NOT NULL,
            execution_lane TEXT NOT NULL,
            parallel_safe INTEGER NOT NULL DEFAULT 0,
            worker_slots INTEGER NOT NULL DEFAULT 1,
            gpu_required INTEGER NOT NULL DEFAULT 0,
            has_parameters_cell INTEGER NOT NULL DEFAULT 0,
            parameter_source TEXT NOT NULL DEFAULT 'none',
            default_timeout_seconds INTEGER NOT NULL DEFAULT 300,
            inputs_hint TEXT NOT NULL DEFAULT '',
            override_json TEXT NOT NULL DEFAULT '{}',
            last_inventory_at TEXT,
            last_status TEXT NOT NULL DEFAULT 'pending',
            last_execution_mode TEXT,
            last_runtime_seconds REAL,
            last_peak_memory_mb REAL,
            last_error_type TEXT,
            last_error_message TEXT,
            last_run_at TEXT,
            last_batch_id TEXT,
            notes TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS notebook_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            path TEXT NOT NULL,
            notebook_type TEXT NOT NULL,
            chapter INTEGER,
            case_study_id TEXT,
            stage TEXT,
            status TEXT NOT NULL,
            execution_mode TEXT NOT NULL,
            runtime_seconds REAL,
            peak_memory_mb REAL,
            timeout_seconds INTEGER NOT NULL,
            worker_slots INTEGER NOT NULL,
            output_root TEXT,
            data_root TEXT,
            parameter_source TEXT NOT NULL,
            parameters_json TEXT NOT NULL DEFAULT '{}',
            error_type TEXT,
            error_message TEXT,
            log_path TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            FOREIGN KEY(path) REFERENCES notebooks(path)
        );

        CREATE INDEX IF NOT EXISTS idx_notebooks_type_chapter
            ON notebooks(notebook_type, chapter, case_study_id, stage_order);
        CREATE INDEX IF NOT EXISTS idx_notebooks_status
            ON notebooks(last_status, notebook_type, chapter);
        CREATE INDEX IF NOT EXISTS idx_runs_batch
            ON notebook_runs(batch_id, notebook_type, chapter, case_study_id, stage);

        CREATE TABLE IF NOT EXISTS program_tracker (
            item_key TEXT PRIMARY KEY,
            track TEXT NOT NULL,
            scope_type TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            label TEXT NOT NULL,
            sort_order INTEGER NOT NULL,
            required INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'pending',
            status_source TEXT NOT NULL DEFAULT 'auto',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            notes TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_program_tracker_track
            ON program_tracker(track, scope_type, sort_order, scope_id);
        """
    )
    conn.commit()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _has_parameters_cell(text: str) -> bool:
    tags = ('tags=["parameters"]', "tags=['parameters']", '"parameters"', "'parameters'")
    return any(tag in text for tag in tags)


def _detect_inputs(text: str, rel_path: str) -> str:
    haystack = f"{rel_path.lower()} {text.lower()}"
    found = [name for name, markers in DATA_HINTS.items() if any(m in haystack for m in markers)]
    return ",".join(sorted(found))


def _classify_entry(
    notebook_type: str,
    rel_path: str,
    text: str,
    chapter: int | None,
    stage_order: int | None,
    overrides: dict,
) -> tuple[str, str, int, int]:
    key = rel_path.lower()
    haystack = f"{key} {text.lower()}"

    if notebook_type == "case_study":
        if overrides.get("gpu") or any(k in haystack for k in GPU_KEYWORDS):
            return "pipeline_gpu", "serial_case_study", 0, 4
        if stage_order is not None and stage_order >= 11:
            return "pipeline_model", "serial_case_study", 0, 3
        if stage_order is not None and stage_order >= 6:
            return "pipeline_feature", "serial_case_study", 0, 2
        return "pipeline_setup", "serial_case_study", 0, 1

    if overrides.get("gpu") or any(k in haystack for k in GPU_KEYWORDS):
        return "gpu", "serial_heavy", 0, 4

    if any(k in haystack for k in HEAVY_KEYWORDS):
        return "heavy", "serial_heavy", 0, 3

    if chapter is not None and chapter <= 6 and not overrides.get("parameters"):
        return "light", "parallel_light", 1, 1

    if chapter is not None and chapter <= 10:
        return "medium", "parallel_medium", 1, 2

    if overrides.get("parameters"):
        return "medium", "parallel_medium", 1, 2

    return "heavy", "serial_heavy", 0, 3


def _default_timeout(overrides: dict) -> int:
    return int(overrides.get("timeout", 300))


def _parameter_source(overrides: dict, has_parameters_cell: bool) -> str:
    if overrides.get("parameters"):
        return "papermill"
    if has_parameters_cell:
        return "none"
    return "config"


def _chapter_entries(repo_root: Path) -> list[NotebookEntry]:
    entries: list[NotebookEntry] = []
    for path in collect_chapter_notebooks(repo_root, range(1, 28)):
        rel = path.relative_to(repo_root)
        notebook_key = str(rel.with_suffix("")).replace(os.sep, "/")
        text = _read_text(path)
        overrides = get_overrides(notebook_key)
        chapter = int(rel.parts[0][:2])
        has_parameters_cell = int(_has_parameters_cell(text))
        resource_profile, execution_lane, parallel_safe, worker_slots = _classify_entry(
            "chapter", rel.as_posix(), text, chapter, None, overrides
        )
        entries.append(
            NotebookEntry(
                path=rel.as_posix(),
                notebook_key=notebook_key,
                notebook_type="chapter",
                chapter=chapter,
                case_study_id=None,
                stage=None,
                stage_order=None,
                title=path.stem,
                resource_profile=resource_profile,
                execution_lane=execution_lane,
                parallel_safe=parallel_safe,
                worker_slots=worker_slots,
                gpu_required=int(bool(overrides.get("gpu"))),
                has_parameters_cell=has_parameters_cell,
                parameter_source=_parameter_source(overrides, bool(has_parameters_cell)),
                default_timeout_seconds=_default_timeout(overrides),
                inputs_hint=_detect_inputs(text, rel.as_posix()),
                override_json=json.dumps(overrides, sort_keys=True),
            )
        )
    return entries


def _case_study_entries(repo_root: Path) -> list[NotebookEntry]:
    entries: list[NotebookEntry] = []
    for case_study_id in CASE_STUDIES:
        cs_dir = repo_root / "case_studies" / case_study_id
        if not cs_dir.exists():
            continue
        for path in sorted(cs_dir.glob("[0-9][0-9]_*.py")):
            if path.name.startswith("_"):
                continue
            rel = path.relative_to(repo_root)
            notebook_key = str(rel.with_suffix("")).replace(os.sep, "/")
            text = _read_text(path)
            overrides = get_overrides(notebook_key)
            stage = path.stem
            stage_order = int(stage[:2]) if stage[:2].isdigit() else None
            has_parameters_cell = int(_has_parameters_cell(text))
            resource_profile, execution_lane, parallel_safe, worker_slots = _classify_entry(
                "case_study", rel.as_posix(), text, None, stage_order, overrides
            )
            entries.append(
                NotebookEntry(
                    path=rel.as_posix(),
                    notebook_key=notebook_key,
                    notebook_type="case_study",
                    chapter=None,
                    case_study_id=case_study_id,
                    stage=stage,
                    stage_order=stage_order,
                    title=path.stem,
                    resource_profile=resource_profile,
                    execution_lane=execution_lane,
                    parallel_safe=parallel_safe,
                    worker_slots=worker_slots,
                    gpu_required=int(bool(overrides.get("gpu"))),
                    has_parameters_cell=has_parameters_cell,
                    parameter_source=_parameter_source(overrides, bool(has_parameters_cell)),
                    default_timeout_seconds=_default_timeout(overrides),
                    inputs_hint=_detect_inputs(text, rel.as_posix()),
                    override_json=json.dumps(overrides, sort_keys=True),
                )
            )
    return entries


def build_inventory(repo_root: Path | None = None) -> list[NotebookEntry]:
    root = repo_root or REPO_ROOT
    return _chapter_entries(root) + _case_study_entries(root)


def upsert_inventory(conn: sqlite3.Connection, entries: list[NotebookEntry]) -> None:
    now = utc_now()
    current_paths = [entry.path for entry in entries]
    conn.executemany(
        """
        INSERT INTO notebooks (
            path,
            notebook_key,
            notebook_type,
            chapter,
            case_study_id,
            stage,
            stage_order,
            title,
            resource_profile,
            execution_lane,
            parallel_safe,
            worker_slots,
            gpu_required,
            has_parameters_cell,
            parameter_source,
            default_timeout_seconds,
            inputs_hint,
            override_json,
            last_inventory_at
        ) VALUES (
            :path,
            :notebook_key,
            :notebook_type,
            :chapter,
            :case_study_id,
            :stage,
            :stage_order,
            :title,
            :resource_profile,
            :execution_lane,
            :parallel_safe,
            :worker_slots,
            :gpu_required,
            :has_parameters_cell,
            :parameter_source,
            :default_timeout_seconds,
            :inputs_hint,
            :override_json,
            :last_inventory_at
        )
        ON CONFLICT(path) DO UPDATE SET
            notebook_key=excluded.notebook_key,
            notebook_type=excluded.notebook_type,
            chapter=excluded.chapter,
            case_study_id=excluded.case_study_id,
            stage=excluded.stage,
            stage_order=excluded.stage_order,
            title=excluded.title,
            resource_profile=excluded.resource_profile,
            execution_lane=excluded.execution_lane,
            parallel_safe=excluded.parallel_safe,
            worker_slots=excluded.worker_slots,
            gpu_required=excluded.gpu_required,
            has_parameters_cell=excluded.has_parameters_cell,
            parameter_source=excluded.parameter_source,
            default_timeout_seconds=excluded.default_timeout_seconds,
            inputs_hint=excluded.inputs_hint,
            override_json=excluded.override_json,
            last_inventory_at=excluded.last_inventory_at
        """,
        [
            {
                **asdict(entry),
                "last_inventory_at": now,
            }
            for entry in entries
        ],
    )
    if current_paths:
        marks = ",".join("?" for _ in current_paths)
        stale_paths = [
            row[0]
            for row in conn.execute(
                f"SELECT path FROM notebooks WHERE path NOT IN ({marks})", current_paths
            ).fetchall()
        ]
        if stale_paths:
            stale_marks = ",".join("?" for _ in stale_paths)
            conn.execute(f"DELETE FROM notebook_runs WHERE path IN ({stale_marks})", stale_paths)
        conn.execute(f"DELETE FROM notebooks WHERE path NOT IN ({marks})", current_paths)
    conn.commit()


def refresh_inventory(
    conn: sqlite3.Connection, repo_root: Path | None = None
) -> list[NotebookEntry]:
    entries = build_inventory(repo_root)
    upsert_inventory(conn, entries)
    return entries


def resolve_data_root(repo_root: Path | None = None) -> Path | None:
    root = repo_root or REPO_ROOT
    env_data = os.environ.get("ML4T_DATA_PATH")
    if env_data:
        path = Path(env_data).expanduser()
        if path.exists():
            return path

    env_file = root / ".env"
    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "ML4T_DATA_PATH":
            path = Path(value.strip().strip('"').strip("'")).expanduser()
            if path.exists():
                return path
    return None


def latest_status_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT notebook_type, last_status, COUNT(*) AS n
        FROM notebooks
        GROUP BY notebook_type, last_status
        ORDER BY notebook_type, last_status
        """
    ).fetchall()


def _latest_notebook_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH latest AS (
            SELECT
                nr.*,
                ROW_NUMBER() OVER (PARTITION BY path ORDER BY id DESC) AS rn
            FROM notebook_runs nr
        )
        SELECT
            n.path,
            n.notebook_type,
            n.chapter,
            n.case_study_id,
            n.stage_order,
            COALESCE(l.status, 'pending') AS status,
            l.runtime_seconds,
            l.peak_memory_mb
        FROM notebooks n
        LEFT JOIN latest l
            ON n.path = l.path
           AND l.rn = 1
        """
    ).fetchall()


def _rollup_status(statuses: list[str]) -> str:
    if not statuses:
        return "pending"
    unique = set(statuses)
    if unique == {"ok"}:
        return "complete"
    if unique == {"pending"}:
        return "pending"
    if {"error", "blocked", "skipped"} & unique:
        return "blocked"
    if "ok" in unique and "pending" in unique:
        return "in_progress"
    if "ok" in unique:
        return "in_progress"
    return "pending"


def _functional_chapter_metrics(
    rows: list[sqlite3.Row], chapter: int
) -> tuple[str, dict[str, int]]:
    chapter_rows = [
        row for row in rows if row["notebook_type"] == "chapter" and row["chapter"] == chapter
    ]
    counts: dict[str, int] = {}
    for row in chapter_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return _rollup_status([row["status"] for row in chapter_rows]), {
        "total": len(chapter_rows),
        **counts,
    }


def _functional_case_study_metrics(
    rows: list[sqlite3.Row], case_study_id: str, max_stage: int = 5
) -> tuple[str, dict[str, int]]:
    cs_rows = [
        row
        for row in rows
        if row["notebook_type"] == "case_study"
        and row["case_study_id"] == case_study_id
        and row["stage_order"] is not None
        and row["stage_order"] <= max_stage
    ]
    counts: dict[str, int] = {}
    for row in cs_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return _rollup_status([row["status"] for row in cs_rows]), {
        "total": len(cs_rows),
        **counts,
    }


def _schema_status_for_chapter(chapter: int) -> str:
    if chapter in TRACKER_SCHEMA_COMPLETE_CHAPTERS:
        return "complete"
    if chapter in TRACKER_SCHEMA_IN_PROGRESS_CHAPTERS:
        return "in_progress"
    return "pending"


def _schema_status_for_case_study(case_study_id: str) -> str:
    return TRACKER_SCHEMA_CASE_STUDIES.get(case_study_id, "pending")


def sync_program_tracker(conn: sqlite3.Connection) -> None:
    rows = _latest_notebook_rows(conn)
    now = utc_now()
    tracker_rows: list[dict[str, object]] = [
        {
            "item_key": "foundation:catalog",
            "track": "foundation",
            "scope_type": "foundation",
            "scope_id": "catalog",
            "label": "Notebook catalog and Docker runner",
            "sort_order": 0,
            "required": 1,
            "status": "complete",
            "status_source": "manual",
            "metrics_json": json.dumps({}, sort_keys=True),
            "notes": "SQLite catalog, Docker runner, and isolated-output execution are in place.",
            "updated_at": now,
        }
    ]

    for chapter in range(1, 21):
        functional_status, functional_metrics = _functional_chapter_metrics(rows, chapter)
        tracker_rows.append(
            {
                "item_key": f"chapter:{chapter:02d}:functional",
                "track": "functional",
                "scope_type": "chapter",
                "scope_id": f"{chapter:02d}",
                "label": f"Chapter {chapter:02d} functional correctness",
                "sort_order": chapter,
                "required": 1,
                "status": functional_status,
                "status_source": "auto",
                "metrics_json": json.dumps(functional_metrics, sort_keys=True),
                "notes": "",
                "updated_at": now,
            }
        )
        tracker_rows.append(
            {
                "item_key": f"chapter:{chapter:02d}:schema",
                "track": "schema",
                "scope_type": "chapter",
                "scope_id": f"{chapter:02d}",
                "label": f"Chapter {chapter:02d} canonical schema retrofit",
                "sort_order": chapter,
                "required": 1,
                "status": _schema_status_for_chapter(chapter),
                "status_source": "manual",
                "metrics_json": json.dumps({}, sort_keys=True),
                "notes": "",
                "updated_at": now,
            }
        )
        repro_required = int(chapter >= 11)
        repro_status = "pending" if repro_required else "not_required"
        repro_notes = (
            "Required for model/results notebooks and any book-facing figures or reported results."
            if repro_required
            else "Teaching notebooks default to functional-only unless book-facing outputs require parity."
        )
        tracker_rows.append(
            {
                "item_key": f"chapter:{chapter:02d}:repro",
                "track": "repro",
                "scope_type": "chapter",
                "scope_id": f"{chapter:02d}",
                "label": f"Chapter {chapter:02d} dev-registry reproducibility validation",
                "sort_order": chapter,
                "required": repro_required,
                "status": repro_status,
                "status_source": "manual",
                "metrics_json": json.dumps({}, sort_keys=True),
                "notes": repro_notes,
                "updated_at": now,
            }
        )

    for idx, case_study_id in enumerate(CASE_STUDIES, start=1):
        functional_status, functional_metrics = _functional_case_study_metrics(rows, case_study_id)
        tracker_rows.append(
            {
                "item_key": f"case_study:{case_study_id}:functional_1_5",
                "track": "functional",
                "scope_type": "case_study",
                "scope_id": case_study_id,
                "label": f"{case_study_id} stages 1-5 functional correctness",
                "sort_order": idx,
                "required": 1,
                "status": functional_status,
                "status_source": "auto",
                "metrics_json": json.dumps(functional_metrics, sort_keys=True),
                "notes": "",
                "updated_at": now,
            }
        )
        tracker_rows.append(
            {
                "item_key": f"case_study:{case_study_id}:schema_1_5",
                "track": "schema",
                "scope_type": "case_study",
                "scope_id": case_study_id,
                "label": f"{case_study_id} early-stage canonical schema retrofit",
                "sort_order": idx,
                "required": 1,
                "status": _schema_status_for_case_study(case_study_id),
                "status_source": "manual",
                "metrics_json": json.dumps({}, sort_keys=True),
                "notes": "",
                "updated_at": now,
            }
        )
        tracker_rows.append(
            {
                "item_key": f"case_study:{case_study_id}:repro",
                "track": "repro",
                "scope_type": "case_study",
                "scope_id": case_study_id,
                "label": f"{case_study_id} dev-registry reproducibility validation",
                "sort_order": idx,
                "required": 1,
                "status": "pending",
                "status_source": "manual",
                "metrics_json": json.dumps({}, sort_keys=True),
                "notes": CRYPTO_REPRO_NOTE if case_study_id == "crypto_perps_funding" else "",
                "updated_at": now,
            }
        )

    conn.executemany(
        """
        INSERT INTO program_tracker (
            item_key,
            track,
            scope_type,
            scope_id,
            label,
            sort_order,
            required,
            status,
            status_source,
            metrics_json,
            notes,
            updated_at
        ) VALUES (
            :item_key,
            :track,
            :scope_type,
            :scope_id,
            :label,
            :sort_order,
            :required,
            :status,
            :status_source,
            :metrics_json,
            :notes,
            :updated_at
        )
        ON CONFLICT(item_key) DO UPDATE SET
            track=excluded.track,
            scope_type=excluded.scope_type,
            scope_id=excluded.scope_id,
            label=excluded.label,
            sort_order=excluded.sort_order,
            required=excluded.required,
            status=excluded.status,
            status_source=excluded.status_source,
            metrics_json=excluded.metrics_json,
            notes=excluded.notes,
            updated_at=excluded.updated_at
        """,
        tracker_rows,
    )
    conn.commit()


def tracker_status_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT track, status, COUNT(*) AS n
        FROM program_tracker
        GROUP BY track, status
        ORDER BY track, status
        """
    ).fetchall()
