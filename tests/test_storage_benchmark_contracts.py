"""Contract tests for comparable database benchmark operations."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "02_financial_data_universe" / "21_storage_benchmark_database.py"


def _function_source(name: str) -> str:
    source = NOTEBOOK.read_text()
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_write_timers_exclude_schema_setup_but_include_ingestion() -> None:
    sqlite = _function_source("write_sqlite")
    duckdb = _function_source("write_duckdb")
    postgres = _function_source("write_postgres")
    timescale = _function_source("write_timescaledb")

    for function in (sqlite, duckdb, postgres, timescale):
        assert "DROP TABLE" not in function
        assert "CREATE TABLE" not in function
        assert "CREATE INDEX" not in function
    assert 'if_exists="append"' in sqlite
    assert "INSERT INTO ohlcv" in duckdb
    assert "execute_values" in postgres
    assert "execute_values" in timescale

    source = NOTEBOOK.read_text()
    assert "CREATE INDEX idx_timestamp ON ohlcv(timestamp)" in source
    assert "CREATE INDEX idx_pg_timestamp ON ohlcv_benchmark(timestamp)" in source
    assert "create_hypertable" in source


def test_kdb_operations_use_timed_transfer_and_persisted_table() -> None:
    write = _function_source("write_pykx")
    range_query = _function_source("pykx_range_query")
    aggregation = _function_source("pykx_ohlcv")

    assert "kx.toq(ohlcv_pandas)" in write
    assert 'q["ohlcv"]' in write
    assert "shutil.rmtree" not in write
    assert "get tbl" in range_query and "KDB_TBL_HANDLE" in range_query
    assert "get tbl" in aggregation and "KDB_TBL_HANDLE" in aggregation


def test_range_queries_prune_at_every_scale_and_detect_over_return() -> None:
    source = NOTEBOOK.read_text()

    assert "RANGE_QUERY_FRACTION = 0.2" in source
    assert "if n_sessions > 1:" in source
    assert "total_rows + 1" in source
    assert "range_expected_rows + 1" in source
    assert "agg_expected_rows + 1" in source
    assert source.count("tolerance=0") >= 3


def test_benchmark_ci_uses_reduced_scale_and_explicit_timeout() -> None:
    overrides = yaml.safe_load((REPO_ROOT / "tests" / "overrides.yaml").read_text())
    config = overrides["02_financial_data_universe/21_storage_benchmark_database"]

    assert config["parameters"]["BENCHMARK_SCALE"] == "S"
    assert config["timeout"] >= 900


def test_compose_does_not_override_pykx_qhome() -> None:
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "QHOME=" not in compose
    assert "${HOME}" not in compose

    parsed = yaml.safe_load(compose)
    for service_name in ("benchmark", "benchmark-full"):
        service = parsed["services"][service_name]
        assert "QLIC=/home/ml4t/.kx" in service["environment"]
        assert any("/home/ml4t/.kx:ro" in volume for volume in service["volumes"])
        assert any("/home/ml4t/.pykx:ro" in volume for volume in service["volumes"])


def test_benchmark_dockerfile_uses_one_uv_binary_and_external_lock_stage() -> None:
    dockerfile = (REPO_ROOT / "envs" / "benchmark" / "Dockerfile").read_text()

    assert dockerfile.count("COPY --from=ghcr.io/astral-sh/uv:latest") == 1
    assert "COPY pyproject.toml uv.lock /lock/" in dockerfile
    assert "cd /lock" in dockerfile
    assert "/build/root" not in dockerfile


def test_optional_database_imports_tolerate_runtime_incompatibility() -> None:
    tree = ast.parse(NOTEBOOK.read_text())
    arctic_try = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(statement, ast.Import)
            and any(alias.name == "arcticdb" for alias in statement.names)
            for statement in node.body
        )
    )
    caught = {
        handler.type.id for handler in arctic_try.handlers if isinstance(handler.type, ast.Name)
    }

    assert {"ImportError", "Exception"} <= caught

    full_manifest = (REPO_ROOT / "envs" / "benchmark" / "pyproject.full.toml").read_text()
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    assert '"protobuf>=5,<7"' in full_manifest
    assert "ARCTICDB_REQUIRED=1" in compose
