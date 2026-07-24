"""Tests for the CI fixture subsampler in tests/create_test_data.py.

The subsampler is the only thing standing between a schema change in production
data and a CI fixture set that silently predates it, so the properties asserted
here are the ones whose violation is invisible until a job goes red: that the
canonical schema survives the reduction, that firm identity is preserved rather
than re-derived per date, and that the manifest describes what was written.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests.create_test_data import (
    FIRM_CHAR_SPLITS,
    Dataset,
    _firm_char_tensor_dir,
    build_firm_characteristics,
    write_manifest,
)


def _write_tensor(path: Path, dates: list[int], n_firms: int, seed: int) -> None:
    """Write a Char_*.npz in the published (date, firm, variable) layout.

    -99.99 is the archive's missing marker; the converter drops those cells,
    which is what makes firm observation counts differ.
    """
    rng = np.random.default_rng(seed)
    variables = ["ret", "BEME", "AT"]
    data = rng.normal(size=(len(dates), n_firms, len(variables)))
    # Firm f is missing from the first f dates, so coverage strictly decreases
    # with firm index and "most observed" has a deterministic answer.
    for firm in range(n_firms):
        data[:firm, firm, 0] = -99.99
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        date=np.asarray(dates),
        variable=np.asarray(variables),
        data=data,
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A minimal production data root carrying the three characteristic tensors."""
    char_dir = tmp_path / "equities" / "firm_characteristics" / "dl_asset_pricing" / "char"
    _write_tensor(char_dir / "Char_train.npz", [19670131, 19670228, 19670331], 6, seed=0)
    _write_tensor(char_dir / "Char_valid.npz", [19870130, 19870227], 5, seed=1)
    _write_tensor(char_dir / "Char_test.npz", [19920131, 19920228], 5, seed=2)
    return tmp_path


def test_tensor_dir_accepts_the_manually_unzipped_layout(tmp_path: Path) -> None:
    """A manual unzip leaves datasets/char/; download.py's extractor strips it."""
    nested = (
        tmp_path / "equities" / "firm_characteristics" / "dl_asset_pricing" / "datasets" / "char"
    )
    for _, filename, _ in FIRM_CHAR_SPLITS:
        _write_tensor(nested / filename, [19670131], 2, seed=3)
    assert _firm_char_tensor_dir(tmp_path) == nested


def test_tensor_dir_reports_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="datasets/char/"):
        _firm_char_tensor_dir(tmp_path)


def test_fixture_carries_the_canonical_schema(source: Path, tmp_path: Path) -> None:
    """The loader rejects anything lacking symbol/timestamp/split - the exact gate
    that took ch04, ch11-12 and cs-us_firm_characteristics red."""
    output = tmp_path / "out"
    build_firm_characteristics(source, output)

    for split in ("all", "train", "valid", "test"):
        path = (
            output / "equities" / "firm_characteristics" / f"firm_characteristics_{split}.parquet"
        )
        assert path.exists(), f"{split} fixture missing"
        frame = pl.read_parquet(path)
        assert {"symbol", "timestamp", "split"}.issubset(frame.columns)
        assert frame["timestamp"].dtype == pl.Date
        assert frame["symbol"].dtype == pl.UInt32
        # Legacy column names must not reappear alongside the canonical ones.
        assert "date" not in frame.columns


def test_all_is_the_concatenation_of_the_splits(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    build_firm_characteristics(source, output)
    out_dir = output / "equities" / "firm_characteristics"

    all_frame = pl.read_parquet(out_dir / "firm_characteristics_all.parquet")
    parts = sum(
        len(pl.read_parquet(out_dir / f"firm_characteristics_{split}.parquet"))
        for split, _, _ in FIRM_CHAR_SPLITS
    )
    assert len(all_frame) == parts
    assert set(all_frame["split"].unique()) == {"train", "valid", "test"}


def test_symbol_namespaces_stay_disjoint_across_splits(source: Path, tmp_path: Path) -> None:
    """The archive publishes no cross-split firm mapping, so a symbol that appears
    in two splits would assert an identity the data does not support."""
    output = tmp_path / "out"
    build_firm_characteristics(source, output)
    all_frame = pl.read_parquet(
        output / "equities" / "firm_characteristics" / "firm_characteristics_all.parquet"
    )

    per_split = {
        split: set(all_frame.filter(pl.col("split") == split)["symbol"].to_list())
        for split in ("train", "valid", "test")
    }
    assert not per_split["train"] & per_split["valid"]
    assert not per_split["train"] & per_split["test"]
    assert not per_split["valid"] & per_split["test"]


def test_subsampling_keeps_whole_firm_histories(source: Path, tmp_path: Path, monkeypatch) -> None:
    """Keeping the N most-observed firms must not truncate their time series.

    Subsampling per date instead would leave each retained firm with holes, and
    every downstream lag, fold and cross-sectional rank would then be computed
    over a cross-section that changes shape month to month.
    """
    monkeypatch.setattr("tests.create_test_data.FIRM_CHAR_MAX_ENTITIES", 2)
    output = tmp_path / "out"
    build_firm_characteristics(source, output)

    train = pl.read_parquet(
        output / "equities" / "firm_characteristics" / "firm_characteristics_train.parquet"
    )
    assert train["symbol"].n_unique() == 2
    # Firms 0 and 1 are the most observed by construction (3 and 2 dates).
    per_firm = train.group_by("symbol").len().sort("symbol")
    assert per_firm["len"].to_list() == [3, 2]


def test_subsampling_is_deterministic(source: Path, tmp_path: Path) -> None:
    """Ties are broken by symbol, so a regenerated fixture is byte-comparable."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_firm_characteristics(source, first)
    build_firm_characteristics(source, second)

    for split in ("all", "train", "valid", "test"):
        rel = Path("equities") / "firm_characteristics" / f"firm_characteristics_{split}.parquet"
        assert pl.read_parquet(first / rel).equals(pl.read_parquet(second / rel))


def test_manifest_records_sizes_and_preserves_unselected_entries(tmp_path: Path) -> None:
    """A single --dataset run must not blank the rest of the manifest."""
    output = tmp_path / "out"
    output.mkdir()
    (output / "manifest.json").write_text(
        '{"version": "1", "subsets": {"etfs": {"max_symbols": 15}}, '
        '"files": {"etfs/etf_universe.parquet": {"size_bytes": 1, "size_mb": 0.0}}}'
    )
    target = output / "equities" / "firm_characteristics"
    target.mkdir(parents=True)
    written = target / "firm_characteristics_all.parquet"
    pl.DataFrame({"symbol": [1]}).write_parquet(written)

    dataset = Dataset(
        name="firm_characteristics",
        description="test",
        build=lambda source, output: [],
        owns=(Path("equities") / "firm_characteristics",),
        budget={"max_entities": 200},
    )
    import tests.create_test_data as ctd

    monkeypatched = dict(ctd.DATASETS_BY_NAME, firm_characteristics=dataset)
    original = ctd.DATASETS_BY_NAME
    ctd.DATASETS_BY_NAME = monkeypatched
    try:
        manifest_path = write_manifest(output, {"firm_characteristics": [written]})
    finally:
        ctd.DATASETS_BY_NAME = original

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["subsets"]["etfs"] == {"max_symbols": 15}, "unselected subset was dropped"
    assert manifest["files"]["etfs/etf_universe.parquet"]["size_bytes"] == 1
    entry = manifest["files"]["equities/firm_characteristics/firm_characteristics_all.parquet"]
    assert entry["size_bytes"] == written.stat().st_size
    assert manifest["subsets"]["firm_characteristics"]["max_entities"] == 200
