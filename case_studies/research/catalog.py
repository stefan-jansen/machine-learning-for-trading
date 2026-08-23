from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from case_studies.utils.registry.specs import IDENTITY_VERSION, canonical_json

if TYPE_CHECKING:
    from .comparison import CandidateSet
    from .workspace import Study


CATALOG_VERSION = 1
_METRIC_COLUMNS = (
    "ic_mean",
    "ic_std",
    "ic_t",
    # Validation dates that produced a defined cross-sectional IC. A configuration whose
    # predictions collapse to near-constant on some folds yields no IC on those dates, so its
    # ic_mean is measured over fewer of them and is not comparable to a full-coverage one.
    # Ranking without this column reports the partial-coverage artifact as the leader.
    "ic_n_days",
    "n_folds",
    "pct_positive",
    "accuracy",
    "balanced_accuracy",
    # `auc_roc` pools every (entity, date) row in a fold into one ROC, so it pays a model for
    # the base rate moving through the year as well as for ranking within a date.
    # `auc_mean_daily` is the cross-sectional reading, computed within each date and averaged,
    # which is the same shape as `ic_mean` and the one to compare against it. Both are carried:
    # they agree where the cross-section is balanced and diverge where it is not.
    "auc_roc",
    "auc_mean_daily",
    "auc_n_days",
    "auc_t_hac",
    "auc_pr",
    "log_loss",
    "brier_score",
)
RESERVED_COLUMNS: dict[str, Any] = {
    "catalog_version": pl.Int64,
    "origin": pl.String,
    "identity_status": pl.String,
    "family": pl.String,
    "config_name": pl.String,
    "label": pl.String,
    "task": pl.String,
    "direction_label": pl.String,
    "split": pl.String,
    "checkpoint_kind": pl.String,
    "checkpoint_value": pl.Int64,
    "checkpoint_spec_json": pl.String,
    "cv_identity": pl.String,
    "execution_tier": pl.String,
    "approval": pl.String,
    "complete": pl.Boolean,
    "created_at": pl.String,
    "metrics_computed_at": pl.String,
    "artifact_available": pl.Boolean,
    **{metric: pl.Float64 for metric in _METRIC_COLUMNS},
    "diagnostic_metrics_json": pl.String,
    "provenance_json": pl.String,
    "training_hash": pl.String,
    "prediction_hash": pl.String,
    "spec_json": pl.String,
}

_BACKTEST_METRIC_COLUMNS = (
    "sharpe",
    "sortino",
    "total_return",
    "max_drawdown",
    "cagr",
    "volatility",
    "calmar",
    "omega",
    "stability",
    "tail_ratio",
    "win_rate",
    "kurtosis",
    "skewness",
    "var_95",
    "cvar_95",
    "n_periods",
    "num_trades",
    "total_commission",
    "total_slippage",
    "avg_turnover",
)
BACKTEST_RESERVED_COLUMNS: dict[str, Any] = {
    "catalog_version": pl.Int64,
    "origin": pl.String,
    "identity_status": pl.String,
    "family": pl.String,
    "config_name": pl.String,
    "label": pl.String,
    "split": pl.String,
    "checkpoint_kind": pl.String,
    "checkpoint_value": pl.Int64,
    "stage": pl.String,
    "execution_tier": pl.String,
    "approval": pl.String,
    "completion_state": pl.String,
    "complete": pl.Boolean,
    "created_at": pl.String,
    "metrics_computed_at": pl.String,
    "artifact_available": pl.Boolean,
    "signal_method": pl.String,
    "allocation_method": pl.String,
    "risk_method": pl.String,
    "decision_artifact_hash": pl.String,
    **{metric: pl.Float64 for metric in _BACKTEST_METRIC_COLUMNS},
    "metrics_json": pl.String,
    "training_hash": pl.String,
    "prediction_hash": pl.String,
    "backtest_hash": pl.String,
    "training_spec_json": pl.String,
    "spec_json": pl.String,
}


def _tables(db: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}


def _select(column: str, columns: set[str], alias: str, default: str = "NULL") -> str:
    return (
        f"{alias}.{column} AS {alias}_{column}"
        if column in columns
        else f"{default} AS {alias}_{column}"
    )


def _nested(spec: dict[str, Any], *path: str) -> Any:
    value: Any = spec
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _computation(spec: dict[str, Any]) -> dict[str, Any]:
    value = spec.get("computation")
    return value if isinstance(value, dict) else spec


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}__{key}" if prefix else key, value[key], output)
    else:
        output[prefix] = value


def _open_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return canonical_json({"value": value})[9:-1]


def _registry_rows(root: Path, origin: str) -> list[dict[str, Any]]:
    db_path = root / "run_log" / "registry.db"
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return []
    query = f"file:{db_path}?mode=ro"
    if origin == "released":
        query += "&immutable=1"
    with closing(sqlite3.connect(query, uri=True)) as db:
        tables = _tables(db)
        if not {"training_runs", "prediction_sets"} <= tables:
            return []
        training_columns = _columns(db, "training_runs")
        prediction_columns = _columns(db, "prediction_sets")
        coverage_columns = (
            _columns(db, "prediction_coverage") if "prediction_coverage" in tables else set()
        )
        metric_columns = (
            _columns(db, "prediction_metrics") if "prediction_metrics" in tables else set()
        )
        expressions = [
            _select("training_hash", training_columns, "t"),
            _select("family", training_columns, "t"),
            _select("label", training_columns, "t"),
            _select("config_name", training_columns, "t"),
            _select("spec_json", training_columns, "t", "'{}'"),
            _select("identity_version", training_columns, "t"),
            _select("execution_tier", training_columns, "t"),
            _select("git_commit", training_columns, "t"),
            _select("entry_point", training_columns, "t"),
            _select("started_at", training_columns, "t"),
            _select("elapsed_s", training_columns, "t"),
            _select("runtime_json", training_columns, "t", "'{}'"),
            _select("prediction_hash", prediction_columns, "p"),
            _select("checkpoint_kind", prediction_columns, "p"),
            _select("checkpoint_value", prediction_columns, "p"),
            _select("split", prediction_columns, "p"),
            _select("created_at", prediction_columns, "p"),
            _select("status", coverage_columns, "c"),
            _select("n_folds_expected", coverage_columns, "c"),
            _select("prediction_hash", metric_columns, "m"),
            _select("computed_at", metric_columns, "m"),
            _select("task_type", metric_columns, "m"),
            # Which sibling label an `auc_*` block was scored against. A classification row
            # scores its own label and leaves this null; a regression row has no classes, so
            # the AUC it carries is against a declared direction sibling and is meaningless
            # without knowing which. A regression label with no sibling carries no AUC.
            _select("direction_label", metric_columns, "m"),
            *[_select(metric, metric_columns, "m") for metric in _METRIC_COLUMNS],
        ]
        fold_metric_count = (
            "(SELECT COUNT(*) FROM fold_metrics fm "
            "WHERE fm.prediction_hash = p.prediction_hash) AS fold_metric_count"
            if "fold_metrics" in tables
            else "0 AS fold_metric_count"
        )
        expressions.append(fold_metric_count)
        coverage_join = (
            "LEFT JOIN prediction_coverage c ON c.prediction_hash = p.prediction_hash"
            if coverage_columns
            else "LEFT JOIN (SELECT NULL AS prediction_hash) c ON 0"
        )
        metrics_join = (
            "LEFT JOIN prediction_metrics m ON m.prediction_hash = p.prediction_hash"
            if metric_columns
            else "LEFT JOIN (SELECT NULL AS prediction_hash) m ON 0"
        )
        cursor = db.execute(
            f"SELECT {', '.join(expressions)} FROM prediction_sets p "
            "JOIN training_runs t ON t.training_hash = p.training_hash "
            f"{coverage_join} {metrics_join}"
        )
        columns = [description[0] for description in cursor.description]
        records = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        overlay_roots = (
            {
                row[0]: Path(row[1])
                for row in db.execute(
                    "SELECT result_hash, source_root FROM overlay_references "
                    "WHERE result_kind = 'prediction'"
                ).fetchall()
            }
            if "overlay_references" in tables
            else {}
        )

    rows: list[dict[str, Any]] = []
    for record in records:
        try:
            spec = json.loads(record["t_spec_json"] or "{}")
        except json.JSONDecodeError:
            spec = {}
        computation = _computation(spec)
        model_value = computation.get("model")
        model: dict[str, Any] = dict(model_value) if isinstance(model_value, dict) else {}
        cv_value = computation.get("cv")
        cv: dict[str, Any] = dict(cv_value) if isinstance(cv_value, dict) else {}
        task = computation.get("task")
        if isinstance(task, dict):
            task = task.get("type")
        task = task or record["m_task_type"]
        identity_version = record["t_identity_version"]
        identity_status = (
            "current"
            if identity_version == IDENTITY_VERSION
            else ("legacy-v2" if identity_version == 2 else "legacy")
        )
        artifact_root = overlay_roots.get(record["p_prediction_hash"], root)
        row_origin = "released" if record["p_prediction_hash"] in overlay_roots else origin
        artifact = (
            artifact_root
            / "run_log"
            / "predictions"
            / record["p_prediction_hash"]
            / "predictions.parquet"
        )
        metrics = {
            metric: record[f"m_{metric}"]
            for metric in _METRIC_COLUMNS
            if record[f"m_{metric}"] is not None
        }
        spec_provenance = spec.get("provenance")
        provenance = dict(spec_provenance) if isinstance(spec_provenance, dict) else {}
        provenance.update(
            {
                key: value
                for key, value in {
                    "git_commit": record["t_git_commit"],
                    "entry_point": record["t_entry_point"],
                    "started_at": record["t_started_at"],
                    "elapsed_s": record["t_elapsed_s"],
                    "runtime": json.loads(record["t_runtime_json"] or "{}"),
                }.items()
                if value not in (None, {}, "")
            }
        )
        complete = (
            identity_status == "current"
            and record["c_status"] == "complete"
            and record["m_prediction_hash"] is not None
            and record["fold_metric_count"] == record["c_n_folds_expected"]
            and artifact.is_file()
        )
        row: dict[str, Any] = {
            "catalog_version": CATALOG_VERSION,
            "origin": row_origin,
            "identity_status": identity_status,
            "family": record["t_family"],
            "config_name": record["t_config_name"],
            "label": record["t_label"],
            "task": task,
            "direction_label": record["m_direction_label"],
            "split": record["p_split"],
            "checkpoint_kind": record["p_checkpoint_kind"],
            "checkpoint_value": record["p_checkpoint_value"],
            "checkpoint_spec_json": canonical_json(
                {
                    "kind": record["p_checkpoint_kind"],
                    "value": record["p_checkpoint_value"],
                }
            ),
            "cv_identity": cv.get("identity"),
            "execution_tier": record["t_execution_tier"] or "canonical",
            "approval": "unapproved",
            "complete": complete,
            "created_at": record["p_created_at"],
            "metrics_computed_at": record["m_computed_at"],
            "artifact_available": artifact.is_file(),
            **{metric: record[f"m_{metric}"] for metric in _METRIC_COLUMNS},
            "diagnostic_metrics_json": canonical_json(metrics),
            "provenance_json": canonical_json(provenance),
            "training_hash": record["t_training_hash"],
            "prediction_hash": record["p_prediction_hash"],
            "spec_json": canonical_json(spec),
        }
        open_fields: dict[str, Any] = {}
        _flatten("model", model, open_fields)
        _flatten("preprocessing", computation.get("preprocessing", {}), open_fields)
        _flatten("cv", cv.get("request", {}), open_fields)
        row.update({key: _open_value(value) for key, value in open_fields.items()})
        rows.append(row)
    return rows


def _dtype(values: list[Any]) -> Any:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return pl.String
    types = {type(value) for value in concrete}
    if types <= {bool}:
        return pl.Boolean
    if types <= {int}:
        return pl.Int64
    if types <= {int, float}:
        return pl.Float64
    return pl.String


def _frame(
    rows: list[dict[str, Any]],
    reserved_columns: dict[str, Any] = RESERVED_COLUMNS,
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=reserved_columns)
    columns = list(reserved_columns)
    columns.extend(sorted(set().union(*(row.keys() for row in rows)) - set(columns)))
    schema = dict(reserved_columns)
    for column in columns:
        if column not in schema:
            schema[column] = _dtype([row.get(column) for row in rows])
    normalized = []
    for row in rows:
        values = {}
        for column in columns:
            value = row.get(column)
            if (
                schema[column] == pl.Float64
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                value = float(value)
            elif schema[column] == pl.String and value is not None and not isinstance(value, str):
                value = canonical_json({"value": value})[9:-1]
            values[column] = value
        normalized.append(values)
    return pl.DataFrame(normalized, schema=schema).select(columns)


def _backtest_registry_rows(root: Path, origin: str) -> list[dict[str, Any]]:
    db_path = root / "run_log" / "registry.db"
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return []
    query = f"file:{db_path}?mode=ro"
    if origin == "released":
        query += "&immutable=1"
    with closing(sqlite3.connect(query, uri=True)) as db:
        tables = _tables(db)
        if not {"training_runs", "prediction_sets", "backtest_runs"} <= tables:
            return []
        training_columns = _columns(db, "training_runs")
        prediction_columns = _columns(db, "prediction_sets")
        backtest_columns = _columns(db, "backtest_runs")
        coverage_columns = (
            _columns(db, "prediction_coverage") if "prediction_coverage" in tables else set()
        )
        prediction_metric_columns = (
            _columns(db, "prediction_metrics") if "prediction_metrics" in tables else set()
        )
        backtest_metric_columns = (
            _columns(db, "backtest_metrics") if "backtest_metrics" in tables else set()
        )
        expressions = [
            _select("training_hash", training_columns, "t"),
            _select("family", training_columns, "t"),
            _select("label", training_columns, "t"),
            _select("config_name", training_columns, "t"),
            _select("spec_json", training_columns, "t", "'{}'"),
            _select("identity_version", training_columns, "t"),
            _select("execution_tier", training_columns, "t"),
            _select("prediction_hash", prediction_columns, "p"),
            _select("checkpoint_kind", prediction_columns, "p"),
            _select("checkpoint_value", prediction_columns, "p"),
            _select("split", prediction_columns, "p"),
            _select("backtest_hash", backtest_columns, "b"),
            _select("spec_json", backtest_columns, "b", "'{}'"),
            _select("stage", backtest_columns, "b"),
            _select("created_at", backtest_columns, "b"),
            _select("status", coverage_columns, "c"),
            _select("n_folds_expected", coverage_columns, "c"),
            _select("prediction_hash", prediction_metric_columns, "pm"),
            _select("backtest_hash", backtest_metric_columns, "bm"),
            _select("computed_at", backtest_metric_columns, "bm"),
            *[
                _select(metric, backtest_metric_columns, "bm")
                for metric in _BACKTEST_METRIC_COLUMNS
            ],
        ]
        expressions.append(
            "(SELECT COUNT(*) FROM fold_metrics fm "
            "WHERE fm.prediction_hash = p.prediction_hash) AS prediction_fold_metric_count"
            if "fold_metrics" in tables
            else "0 AS prediction_fold_metric_count"
        )
        coverage_join = (
            "LEFT JOIN prediction_coverage c ON c.prediction_hash = p.prediction_hash"
            if coverage_columns
            else "LEFT JOIN (SELECT NULL AS prediction_hash) c ON 0"
        )
        prediction_metrics_join = (
            "LEFT JOIN prediction_metrics pm ON pm.prediction_hash = p.prediction_hash"
            if prediction_metric_columns
            else "LEFT JOIN (SELECT NULL AS prediction_hash) pm ON 0"
        )
        backtest_metrics_join = (
            "LEFT JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash"
            if backtest_metric_columns
            else "LEFT JOIN (SELECT NULL AS backtest_hash) bm ON 0"
        )
        cursor = db.execute(
            f"SELECT {', '.join(expressions)} FROM backtest_runs b "
            "JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash "
            "JOIN training_runs t ON t.training_hash = p.training_hash "
            f"{coverage_join} {prediction_metrics_join} {backtest_metrics_join}"
        )
        columns = [description[0] for description in cursor.description]
        records = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        prediction_roots = (
            {
                row[0]: Path(row[1])
                for row in db.execute(
                    "SELECT result_hash, source_root FROM overlay_references "
                    "WHERE result_kind = 'prediction'"
                ).fetchall()
            }
            if "overlay_references" in tables
            else {}
        )

    rows: list[dict[str, Any]] = []
    for record in records:
        try:
            training_spec = json.loads(record["t_spec_json"] or "{}")
        except json.JSONDecodeError:
            training_spec = {}
        try:
            backtest_spec = json.loads(record["b_spec_json"] or "{}")
        except json.JSONDecodeError:
            backtest_spec = {}
        computation = _computation(training_spec)
        model_value = computation.get("model")
        model = dict(model_value) if isinstance(model_value, dict) else {}
        cv_value = computation.get("cv")
        cv = dict(cv_value) if isinstance(cv_value, dict) else {}
        strategy_value = backtest_spec.get("strategy")
        strategy = dict(strategy_value) if isinstance(strategy_value, dict) else {}
        identity_version = record["t_identity_version"]
        identity_status = (
            "current"
            if identity_version == IDENTITY_VERSION
            else ("legacy-v2" if identity_version == 2 else "legacy")
        )
        prediction_root = prediction_roots.get(record["p_prediction_hash"], root)
        prediction_artifact = (
            prediction_root
            / "run_log"
            / "predictions"
            / record["p_prediction_hash"]
            / "predictions.parquet"
        )
        returns_artifact = (
            root / "run_log" / "backtest" / record["b_backtest_hash"] / "daily_returns.parquet"
        )
        complete = (
            identity_status == "current"
            and record["c_status"] == "complete"
            and record["pm_prediction_hash"] is not None
            and record["prediction_fold_metric_count"] == record["c_n_folds_expected"]
            and prediction_artifact.is_file()
            and record["bm_backtest_hash"] is not None
            and returns_artifact.is_file()
        )
        metrics = {
            metric: record[f"bm_{metric}"]
            for metric in _BACKTEST_METRIC_COLUMNS
            if record[f"bm_{metric}"] is not None
        }
        signal_value = strategy.get("signal")
        signal: dict[str, Any] = dict(signal_value) if isinstance(signal_value, dict) else {}
        allocation_value = strategy.get("allocation")
        allocation: dict[str, Any] = (
            dict(allocation_value) if isinstance(allocation_value, dict) else {}
        )
        risk_value = strategy.get("risk")
        risk: dict[str, Any] = dict(risk_value) if isinstance(risk_value, dict) else {}
        decision_value = backtest_spec.get("decision_artifact")
        decision: dict[str, Any] = dict(decision_value) if isinstance(decision_value, dict) else {}
        row: dict[str, Any] = {
            "catalog_version": CATALOG_VERSION,
            "origin": origin,
            "identity_status": identity_status,
            "family": record["t_family"],
            "config_name": record["t_config_name"],
            "label": record["t_label"],
            "split": record["p_split"],
            "checkpoint_kind": record["p_checkpoint_kind"],
            "checkpoint_value": record["p_checkpoint_value"],
            "stage": record["b_stage"],
            "execution_tier": record["t_execution_tier"]
            or backtest_spec.get("execution_tier")
            or "canonical",
            "approval": "unapproved",
            "completion_state": "complete" if complete else "partial",
            "complete": complete,
            "created_at": record["b_created_at"],
            "metrics_computed_at": record["bm_computed_at"],
            "artifact_available": returns_artifact.is_file(),
            "signal_method": signal.get("method"),
            "allocation_method": allocation.get("method"),
            "risk_method": risk.get("method"),
            "decision_artifact_hash": decision.get("hash"),
            **{metric: record[f"bm_{metric}"] for metric in _BACKTEST_METRIC_COLUMNS},
            "metrics_json": canonical_json(metrics),
            "training_hash": record["t_training_hash"],
            "prediction_hash": record["p_prediction_hash"],
            "backtest_hash": record["b_backtest_hash"],
            "training_spec_json": canonical_json(training_spec),
            "spec_json": canonical_json(backtest_spec),
        }
        open_fields: dict[str, Any] = {}
        _flatten("model", model, open_fields)
        _flatten("preprocessing", computation.get("preprocessing", {}), open_fields)
        _flatten("cv", cv.get("request", {}), open_fields)
        _flatten("strategy", strategy, open_fields)
        row.update({key: _open_value(value) for key, value in open_fields.items()})
        rows.append(row)
    return rows


def _resolve_authoritative_selection(
    study: Study,
    selection: pl.DataFrame,
    *,
    kind: str,
    canonical: bool = True,
) -> tuple[Any, ...]:
    if not isinstance(selection, pl.DataFrame):
        raise TypeError(f"{kind} selection must be a Polars DataFrame")
    identity_columns = (
        ("training_hash", "prediction_hash")
        if kind == "prediction"
        else ("training_hash", "prediction_hash", "backtest_hash")
    )
    missing = set(identity_columns) - set(selection.columns)
    if missing:
        raise ValueError(
            f"{kind} catalog selection is missing required identity columns: {sorted(missing)}"
        )
    if selection.is_empty():
        raise ValueError(f"{kind} catalog selection is empty")
    result_column = identity_columns[-1]
    if selection.get_column(result_column).n_unique() != selection.height:
        raise ValueError(f"duplicate {kind} identities make the selection ambiguous")
    authoritative = (
        study.predictions.table(include_preview=True)
        if kind == "prediction"
        else study.backtests.table(include_preview=True)
    )
    from .results import Result

    resolved = []
    for selected in selection.select(*identity_columns).iter_rows(named=True):
        match = authoritative.filter(pl.col(result_column) == selected[result_column])
        if match.height != 1:
            raise ValueError(
                f"{kind} identity {selected[result_column]!r} resolved to {match.height} rows"
            )
        row = match.row(0, named=True)
        altered = {
            column: (selected[column], row[column])
            for column in identity_columns
            if selected[column] != row[column]
        }
        if altered:
            details = ", ".join(
                f"{column}={actual!r}, not {supplied!r}"
                for column, (supplied, actual) in altered.items()
            )
            raise ValueError(f"{kind} selection has altered lineage: {details}")
        if canonical and row["execution_tier"] == "preview":
            raise ValueError(f"preview {kind} results cannot enter a canonical candidate set")
        if not row["complete"]:
            raise ValueError(f"{kind} {selected[result_column]} is partial")
        result = Result.open(study, str(selected[result_column]), include_preview=True)
        if result.kind != kind:
            raise ValueError(f"catalog identity {selected[result_column]} is not a {kind}")
        resolved.append(result)
    return tuple(resolved)


class PredictionCatalog:
    def __init__(self, study: Study) -> None:
        self.study = study

    def table(self, *, include_preview: bool = False) -> pl.DataFrame:
        released = _registry_rows(self.study.release_case_root, "released")
        if self.study.read_only:
            return _frame(released).sort("prediction_hash")
        workspace = _registry_rows(self.study.root, "workspace")
        preview: list[dict[str, Any]] = []
        if include_preview and self.study.output_root is not None:
            preview = _registry_rows(
                self.study.output_root / ".preview" / self.study.case_study,
                "workspace",
            )
        seen = {row["prediction_hash"] for row in [*workspace, *preview]}
        overlaid = [
            *workspace,
            *preview,
            *(row for row in released if row["prediction_hash"] not in seen),
        ]
        return _frame(overlaid).sort("prediction_hash")

    def one(self, **filters: Any) -> dict[str, Any]:
        table = self.table()
        for field, value in filters.items():
            if field not in table.columns:
                raise ValueError(f"unknown prediction catalog field {field!r}")
            predicate = pl.col(field).is_null() if value is None else pl.col(field) == value
            table = table.filter(predicate)
        if table.height != 1:
            varying = [
                column
                for column in (
                    "label",
                    "cv_identity",
                    "split",
                    "checkpoint_kind",
                    "checkpoint_value",
                    "training_hash",
                    "prediction_hash",
                )
                if column in table.columns and table.get_column(column).n_unique() > 1
            ]
            raise ValueError(
                f"prediction selection matched {table.height} rows; disambiguate with {varying}"
            )
        return table.row(0, named=True)

    def freeze(
        self,
        selection: pl.DataFrame,
        *,
        name: str,
        comparison_contract: dict[str, Any] | None = None,
    ) -> CandidateSet:
        """Freeze exact authoritative prediction members selected with Polars."""
        from .comparison import CandidateSet

        members = _resolve_authoritative_selection(
            self.study,
            selection,
            kind="prediction",
        )
        return CandidateSet.create(
            self.study,
            name,
            members,
            comparison_contract=comparison_contract,
        )


class BacktestCatalog:
    def __init__(self, study: Study) -> None:
        self.study = study

    def table(self, *, include_preview: bool = False) -> pl.DataFrame:
        released = _backtest_registry_rows(self.study.release_case_root, "released")
        if self.study.read_only:
            return _frame(released, BACKTEST_RESERVED_COLUMNS).sort("backtest_hash")
        workspace = _backtest_registry_rows(self.study.root, "workspace")
        preview: list[dict[str, Any]] = []
        if include_preview and self.study.output_root is not None:
            preview = _backtest_registry_rows(
                self.study.output_root / ".preview" / self.study.case_study,
                "workspace",
            )
        seen = {row["backtest_hash"] for row in [*workspace, *preview]}
        overlaid = [
            *workspace,
            *preview,
            *(row for row in released if row["backtest_hash"] not in seen),
        ]
        return _frame(overlaid, BACKTEST_RESERVED_COLUMNS).sort("backtest_hash")

    def one(self, **filters: Any) -> dict[str, Any]:
        table = self.table()
        for field, value in filters.items():
            if field not in table.columns:
                raise ValueError(f"unknown backtest catalog field {field!r}")
            predicate = pl.col(field).is_null() if value is None else pl.col(field) == value
            table = table.filter(predicate)
        if table.height != 1:
            varying = [
                column
                for column in (
                    "label",
                    "split",
                    "checkpoint_kind",
                    "checkpoint_value",
                    "stage",
                    "signal_method",
                    "allocation_method",
                    "risk_method",
                    "training_hash",
                    "prediction_hash",
                    "backtest_hash",
                )
                if column in table.columns and table.get_column(column).n_unique() > 1
            ]
            raise ValueError(
                f"backtest selection matched {table.height} rows; disambiguate with {varying}"
            )
        return table.row(0, named=True)

    def freeze(
        self,
        selection: pl.DataFrame,
        *,
        name: str,
        comparison_contract: dict[str, Any] | None = None,
    ) -> CandidateSet:
        """Freeze exact authoritative backtest members selected with Polars."""
        from .comparison import CandidateSet

        members = _resolve_authoritative_selection(
            self.study,
            selection,
            kind="backtest",
        )
        return CandidateSet.create(
            self.study,
            name,
            members,
            comparison_contract=comparison_contract,
        )
