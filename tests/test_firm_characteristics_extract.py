"""Unit tests for the firm-characteristics archive handling.

``extract_zip`` flattens the published Chen-Pelger-Zhu archives, which wrap
their contents in a single top-level ``data/`` or ``datasets/`` directory and
(when zipped on macOS) ship ``__MACOSX`` resource forks + ``.DS_Store`` files.
A mismatch here silently lands the ``.npz`` files at the wrong depth, which
``verify_files`` then reports as "missing", exactly the bug fixed in the
firm-char end-to-end repair. These synthetic-zip tests lock the flatten rule
without pulling the real ~1.5 GB dataset.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from data.equities.firm_characteristics.download import (
    EXPECTED_FILES,
    _parquet_has_persistent_symbols,
    convert_to_parquet,
    extract_zip,
)


def _make_zip(zip_path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for arcname, content in entries.items():
            zf.writestr(arcname, content)


def test_extract_zip_strips_datasets_wrapper(tmp_path):
    """A ``datasets/`` wrapper is stripped so files land directly under the dir."""
    zip_path = tmp_path / "datasets.zip"
    _make_zip(
        zip_path,
        {
            "datasets/RetChar.csv": b"Date,RET\n20000101,0.01\n",
            "datasets/char/Char_train.npz": b"fake-npz",
            "datasets/macro/macro_test.npz": b"fake-npz",
        },
    )
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()

    assert extract_zip(zip_path, extract_dir) is True
    assert (extract_dir / "RetChar.csv").exists()
    assert (extract_dir / "char" / "Char_train.npz").exists()
    assert (extract_dir / "macro" / "macro_test.npz").exists()
    # wrapper directory itself must NOT survive
    assert not (extract_dir / "datasets").exists()


def test_extract_zip_strips_data_wrapper(tmp_path):
    """The alternate ``data/`` wrapper is stripped identically."""
    zip_path = tmp_path / "data.zip"
    _make_zip(zip_path, {"data/char/Char_valid.npz": b"x"})
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()

    assert extract_zip(zip_path, extract_dir) is True
    assert (extract_dir / "char" / "Char_valid.npz").exists()
    assert not (extract_dir / "data").exists()


def test_extract_zip_skips_macos_cruft(tmp_path):
    """``__MACOSX`` resource forks and ``.DS_Store`` files are dropped."""
    zip_path = tmp_path / "datasets.zip"
    _make_zip(
        zip_path,
        {
            "datasets/RetChar.csv": b"data",
            "datasets/.DS_Store": b"junk",
            "__MACOSX/datasets/._RetChar.csv": b"resource-fork",
        },
    )
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()

    assert extract_zip(zip_path, extract_dir) is True
    assert (extract_dir / "RetChar.csv").exists()
    assert not (extract_dir / ".DS_Store").exists()
    assert not (extract_dir / "__MACOSX").exists()
    # the resource-fork file must not have leaked in under any name
    assert not any(p.name == "._RetChar.csv" for p in extract_dir.rglob("*"))


def test_extract_zip_cleans_temp_dir(tmp_path):
    """The intermediate ``_temp_extract`` scratch dir is removed."""
    zip_path = tmp_path / "datasets.zip"
    _make_zip(zip_path, {"datasets/RetChar.csv": b"data"})
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()

    extract_zip(zip_path, extract_dir)
    assert not (extract_dir / "_temp_extract").exists()


def test_extract_zip_returns_false_on_bad_archive(tmp_path):
    """A corrupt archive fails gracefully (returns False, no raise)."""
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"not a real zip")
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()
    assert extract_zip(bad, extract_dir) is False


def test_expected_files_manifest_is_sane():
    """The published dataset ships 11 files across RetChar/Macro/char/macro/RF."""
    assert len(EXPECTED_FILES) == 11
    assert "RetChar.csv" in EXPECTED_FILES
    assert all(size > 0 for size in EXPECTED_FILES.values())
    # the three split families each contribute a train/valid/test triple
    for prefix in ("char/", "macro/", "RF/"):
        assert sum(1 for k in EXPECTED_FILES if k.startswith(prefix)) == 3


def _write_characteristic_npz(
    path: Path,
    dates: list[int],
    returns: list[list[float]],
) -> None:
    """Write the smallest faithful CPZ split tensor used by converter tests."""
    ret = np.asarray(returns, dtype=np.float64)
    data = np.stack([ret, ret * 10], axis=2)
    np.savez(
        path,
        date=np.asarray(dates),
        variable=np.asarray(["ret", "feature"]),
        data=data,
    )


def test_convert_to_parquet_recovers_persistent_anonymous_symbols(tmp_path, monkeypatch):
    """The fixed tensor axis, not a shifting monthly row number, identifies firms."""
    source = tmp_path / "equities" / "firm_characteristics" / "dl_asset_pricing" / "char"
    source.mkdir(parents=True)
    _write_characteristic_npz(
        source / "Char_train.npz",
        [19670131, 19670228],
        [[0.01, -99.99, 0.03], [-99.99, 0.02, 0.04]],
    )
    _write_characteristic_npz(
        source / "Char_valid.npz",
        [19870130],
        [[0.05, -99.99]],
    )
    _write_characteristic_npz(
        source / "Char_test.npz",
        [19920131, 19920228],
        [[0.06, -99.99, 0.08], [-99.99, 0.07, 0.09]],
    )

    assert convert_to_parquet(tmp_path) is True

    output = tmp_path / "equities" / "firm_characteristics"
    all_df = pl.read_parquet(output / "firm_characteristics_all.parquet")
    assert all_df.columns == ["symbol", "timestamp", "ret", "feature", "split"]
    assert all_df["timestamp"].dtype == pl.Date
    assert all_df["symbol"].dtype == pl.UInt32
    assert all_df.group_by("split").len().sort("split").to_dicts() == [
        {"split": "test", "len": 4},
        {"split": "train", "len": 4},
        {"split": "valid", "len": 1},
    ]

    test_df = pl.read_parquet(output / "firm_characteristics_test.parquet")
    # Firm-axis 2 survives both months while the compact row position changes.
    persistent = test_df.filter(pl.col("ret").is_in([0.08, 0.09]))
    assert persistent["symbol"].n_unique() == 1
    # Split-specific axes are deliberately namespaced because the archive does
    # not publish a mapping between its train, validation, and test tensors.
    symbols_by_split = all_df.group_by("split").agg(pl.col("symbol").min()).sort("split")
    assert symbols_by_split["symbol"].to_list() == [2_000_000, 0, 1_000_000]

    from data.equities import loader

    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)
    valid_df = loader.load_firm_characteristics(split="valid")
    assert valid_df.shape == (1, 5)
    assert valid_df["split"].unique().to_list() == ["valid"]


def test_existing_parquet_without_recovered_identity_is_stale(tmp_path):
    stale = tmp_path / "stale.parquet"
    pl.DataFrame({"date": ["1992-01-31"], "ret": [0.01]}).write_parquet(stale)
    assert _parquet_has_persistent_symbols(stale) is False

    current = tmp_path / "current.parquet"
    pl.DataFrame(
        {
            "symbol": [2_000_000],
            "timestamp": ["1992-01-31"],
            "ret": [0.01],
            "split": ["test"],
        }
    ).write_parquet(current)
    assert _parquet_has_persistent_symbols(current) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
