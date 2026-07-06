"""Unit tests for the shared download plumbing in ``utils/downloading.py``.

These are fast, deterministic, and network-free. They pin the *contracts* that
every ``data/**/download.py`` script relies on for path resolution and output
placement — the layer where the #361-class bugs lived:

- a downloader must write under the *selected* data root (``--data-path`` /
  ``ML4T_DATA_PATH``), never blindly under ``<repo>/data`` (the read-only /
  wrong-dir bug);
- ``~`` in a ``--config`` / ``--data-path`` must expand (the Copilot review bug);
- ``atomic_write_parquet`` must land a complete file and leave no ``.tmp`` behind.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from utils.downloading import (
    atomic_write_parquet,
    create_base_parser,
    flatten_group_values,
    load_section,
    resolve_data_dir,
    resolve_storage_path,
)

# ---------------------------------------------------------------------------
# resolve_data_dir — precedence + expansion
# ---------------------------------------------------------------------------


def test_resolve_data_dir_cli_arg_wins_over_env(tmp_path, monkeypatch):
    """An explicit ``--data-path`` must win over ``ML4T_DATA_PATH``.

    This is the contract that keeps a downloader writing where the reader
    asked — the mechanism behind the #361 read-only-mount fix.
    """
    cli = tmp_path / "cli_root"
    monkeypatch.setenv("ML4T_DATA_PATH", str(tmp_path / "env_root"))
    resolved = resolve_data_dir(cli)
    assert resolved == cli.resolve()


def test_resolve_data_dir_expands_user(monkeypatch, tmp_path):
    """A ``~``-prefixed CLI path is expanded, not taken literally."""
    monkeypatch.setenv("HOME", str(tmp_path))
    resolved = resolve_data_dir(Path("~/some_data"))
    assert resolved == (tmp_path / "some_data").resolve()
    assert "~" not in str(resolved)


# ---------------------------------------------------------------------------
# resolve_storage_path — the exact mechanism behind the ETF wrong-dir bug
# ---------------------------------------------------------------------------


def test_resolve_storage_path_relative_joins_data_root(tmp_path):
    """A relative configured path is joined under the selected data root."""
    root = tmp_path / "data_root"
    resolved = resolve_storage_path(root, "etfs/market", "etfs")
    assert resolved == root / "etfs/market"


def test_resolve_storage_path_absolute_is_preserved(tmp_path):
    """An absolute configured path is preserved (not re-rooted)."""
    absolute = tmp_path / "elsewhere" / "etfs"
    resolved = resolve_storage_path(tmp_path / "data_root", str(absolute), "etfs")
    assert resolved == absolute


def test_resolve_storage_path_falls_back_when_unconfigured(tmp_path):
    """With no configured path, the fallback is used under the data root."""
    root = tmp_path / "data_root"
    assert resolve_storage_path(root, None, "etfs") == root / "etfs"


def test_resolve_storage_path_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    resolved = resolve_storage_path(tmp_path / "data_root", "~/abs_etfs", "etfs")
    assert resolved == (tmp_path / "abs_etfs")


# ---------------------------------------------------------------------------
# load_section — YAML section reader used by the ETF/crypto/fx downloaders
# ---------------------------------------------------------------------------


def test_load_section_reads_named_section(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("etfs:\n  storage_path: etfs/market\n  start: '2010-01-01'\n")
    section = load_section(cfg, "etfs")
    assert section == {"storage_path": "etfs/market", "start": "2010-01-01"}


def test_load_section_missing_section_returns_empty(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("etfs:\n  storage_path: etfs/market\n")
    assert load_section(cfg, "nonexistent") == {}


def test_load_section_expands_user(tmp_path, monkeypatch):
    """``load_section`` must honour ``~`` so ``--config ~/foo.yaml`` works."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("etfs:\n  storage_path: etfs/market\n")
    section = load_section("~/config.yaml", "etfs")
    assert section == {"storage_path": "etfs/market"}


# ---------------------------------------------------------------------------
# flatten_group_values — grouped symbols/pairs flattening
# ---------------------------------------------------------------------------


def test_flatten_group_values_dedups_and_preserves_order():
    groups = {
        "large_cap": {"symbols": ["SPY", "QQQ", "SPY"]},
        "bonds": {"symbols": ["TLT", "QQQ"]},
        "not_a_dict": ["ignored"],
    }
    assert flatten_group_values(groups, "symbols") == ["SPY", "QQQ", "TLT"]


def test_flatten_group_values_empty():
    assert flatten_group_values({}, "symbols") == []


# ---------------------------------------------------------------------------
# atomic_write_parquet — complete write, no temp residue
# ---------------------------------------------------------------------------


def test_atomic_write_parquet_writes_and_roundtrips(tmp_path):
    df = pl.DataFrame({"symbol": ["AAA", "BBB"], "timestamp": [1, 2]})
    out = tmp_path / "nested" / "dir" / "data.parquet"
    atomic_write_parquet(df, out)

    assert out.exists()  # parent dirs created
    assert pl.read_parquet(out).equals(df)
    # no leftover temp file
    assert not (out.parent / f".{out.name}.tmp").exists()
    assert list(out.parent.glob(".*.tmp")) == []


# ---------------------------------------------------------------------------
# create_base_parser — the standard download CLI flags
# ---------------------------------------------------------------------------


def test_create_base_parser_has_standard_flags(tmp_path):
    parser = create_base_parser("test")
    args = parser.parse_args(["--data-path", str(tmp_path), "--dry-run", "--force", "--verbose"])
    assert args.data_path == tmp_path
    assert args.dry_run is True
    assert args.force is True
    assert args.verbose is True


def test_create_base_parser_defaults(tmp_path):
    parser = create_base_parser("test")
    args = parser.parse_args([])
    assert args.data_path is None
    assert args.dry_run is False
    assert args.force is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
