"""Tests for the CI fixture subsampler in tests/create_test_data.py.

The subsampler is the only thing standing between a schema change in production
data and a CI fixture set that silently predates it, so the properties asserted
here are the ones whose violation is invisible until a job goes red: that the
canonical schema survives the reduction, that firm identity is preserved rather
than re-derived per date, and that the manifest describes what was written.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests.create_test_data import (
    _13F_REQUIRED_COLUMNS,
    FIRM_CHAR_SPLITS,
    Dataset,
    _firm_char_tensor_dir,
    _load_13f_producer,
    build_firm_characteristics,
    build_institutional_holdings_13f,
    roots_overlap,
    write_manifest,
)

# The columns data/equities/positioning/13f_download.py writes, transcribed from
# its select() and agg() calls rather than imported from create_test_data. Reading
# them from _13F_REQUIRED_COLUMNS would make every test below agree with whatever
# that constant happens to say, including an incomplete version of it.
_PRODUCER_13F_SCHEMAS: dict[str, tuple[str, ...]] = {
    "institutional_holdings.parquet": (
        "cik",
        "accession_no",
        "issuer",
        "cusip",
        "value_thousands",
        "shares",
        "put_call",
        "report_date",
        "filing_date",
        "company_name",
    ),
    "institution_stock_edges.parquet": (
        "institution_id",
        "stock_id",
        "institution_name",
        "stock_name",
        "weight_value",
        "weight_shares",
        "report_date",
        "timestamp",
    ),
    "stock_features.parquet": (
        "cusip",
        "issuer_name",
        "n_inst_holders",
        "total_inst_value_usd",
        "avg_position_size_usd",
        "position_size_std_usd",
        "timestamp",
        "ownership_hhi",
        "inst_coverage_pct",
        "position_cv",
        "inst_value_change_usd",
        "inst_pct_change",
    ),
}


def ctd_13f_required_columns() -> dict[str, tuple[str, ...]]:
    """The canonical 13F schemas, so a test frame is built from one source."""
    return _PRODUCER_13F_SCHEMAS


def test_13f_required_columns_cover_the_producer_schemas() -> None:
    """A column the producer emits but the check omits is invisible until ch22 runs.

    22_rag/07 rebuilds ``stock_features`` and compares it with ``assert_frame_equal``,
    which checks the whole schema. A source artifact from an older producer
    generation therefore has to be rejected on any missing column, not only on the
    few a loader reads by name.
    """
    assert {name: set(columns) for name, columns in _13F_REQUIRED_COLUMNS.items()} == {
        name: set(columns) for name, columns in _PRODUCER_13F_SCHEMAS.items()
    }


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

    manifest = json.loads(manifest_path.read_text())
    assert manifest["subsets"]["etfs"] == {"max_symbols": 15}, "unselected subset was dropped"
    assert manifest["files"]["etfs/etf_universe.parquet"]["size_bytes"] == 1
    entry = manifest["files"]["equities/firm_characteristics/firm_characteristics_all.parquet"]
    assert entry["size_bytes"] == written.stat().st_size
    assert manifest["subsets"]["firm_characteristics"]["max_entities"] == 200


def test_manifest_drops_entries_the_rebuild_no_longer_produces(tmp_path: Path) -> None:
    """A renamed or discontinued artifact must not stay listed forever."""
    output = tmp_path / "out"
    target = output / "equities" / "firm_characteristics"
    target.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "version": "1",
                "subsets": {},
                "files": {
                    # Under the dataset's owns path, but not produced this run.
                    "equities/firm_characteristics/firm_characteristics_46.parquet": {
                        "size_bytes": 9,
                        "size_mb": 0.0,
                    },
                    # Outside it — belongs to another dataset, must survive.
                    "equities/positioning/13f/stock_features.parquet": {
                        "size_bytes": 7,
                        "size_mb": 0.0,
                    },
                },
            }
        )
    )
    written = target / "firm_characteristics_all.parquet"
    pl.DataFrame({"symbol": [1]}).write_parquet(written)

    dataset = Dataset(
        name="firm_characteristics",
        description="test",
        build=lambda source, output: [],
        owns=(Path("equities") / "firm_characteristics",),
    )
    import tests.create_test_data as ctd

    original = ctd.DATASETS_BY_NAME
    ctd.DATASETS_BY_NAME = dict(original, firm_characteristics=dataset)
    try:
        manifest_path = write_manifest(output, {"firm_characteristics": [written]})
    finally:
        ctd.DATASETS_BY_NAME = original

    files = json.loads(manifest_path.read_text())["files"]
    assert "equities/firm_characteristics/firm_characteristics_46.parquet" not in files
    assert "equities/firm_characteristics/firm_characteristics_all.parquet" in files
    assert files["equities/positioning/13f/stock_features.parquet"]["size_bytes"] == 7


def test_13f_builder_rejects_a_stale_derived_artifact(tmp_path: Path) -> None:
    """Validating the holdings alone lets a stale edges or features file through.

    ``stock_features`` is the case that reached readers: an older producer spelled
    its issuer column ``issuer``, and 10_text_feature_engineering/02 reads
    ``issuer_name``.
    """
    source = tmp_path / "source"
    holdings_dir = source / "equities" / "positioning" / "13f"
    holdings_dir.mkdir(parents=True)

    for filename, columns in ctd_13f_required_columns().items():
        frame = pl.DataFrame({column: [0] for column in columns})
        if filename == "stock_features.parquet":
            frame = frame.drop("issuer_name").with_columns(pl.lit("ACME").alias("issuer"))
        frame.write_parquet(holdings_dir / filename)

    with pytest.raises(ValueError, match=r"stock_features\.parquet.*\['issuer_name'\]"):
        build_institutional_holdings_13f(source, tmp_path / "out")


def test_13f_builder_rejects_a_source_missing_a_derived_feature(tmp_path: Path) -> None:
    """No consumer loads ``ownership_hhi`` by name, and ch22's parity check still needs it."""
    source = tmp_path / "source"
    holdings_dir = source / "equities" / "positioning" / "13f"
    holdings_dir.mkdir(parents=True)

    for filename, columns in ctd_13f_required_columns().items():
        frame = pl.DataFrame({column: [0] for column in columns})
        if filename == "stock_features.parquet":
            frame = frame.drop("ownership_hhi")
        frame.write_parquet(holdings_dir / filename)

    with pytest.raises(ValueError, match=r"stock_features\.parquet.*\['ownership_hhi'\]"):
        build_institutional_holdings_13f(source, tmp_path / "out")


def _write_13f_source(holdings_dir: Path, drop_cik: str | None = None) -> pl.DataFrame:
    """Write a consistent three-artifact 13F source and return the holdings.

    Every institution the producer requests, filing two post-2023 quarters for
    three issuers: one quarter to build the graph from and a prior one to measure
    ownership change against. The derived artifacts come from the producer itself,
    so the source is consistent by construction and a test that perturbs it is
    testing the check rather than the fixture.

    ``drop_cik`` omits one institution from all three artifacts, which is the case
    a rebuild keyed on the holdings' own CIKs cannot see.
    """
    producer = _load_13f_producer()
    ciks = [cik for _, cik in producer.INSTITUTIONS if cik != drop_cik]
    rows = []
    for report_date, filing_date in (
        (date(2024, 6, 30), date(2024, 8, 14)),
        (date(2024, 9, 30), date(2024, 11, 14)),
    ):
        for cik in ciks:
            for index, (cusip, issuer) in enumerate(
                (("037833100", "APPLE INC"), ("594918104", "MICROSOFT"), ("67066G104", "NVIDIA"))
            ):
                rows.append(
                    {
                        "cik": cik,
                        "accession_no": f"{cik}-{report_date:%Y%m%d}-{index}",
                        "issuer": issuer,
                        "cusip": cusip,
                        "value_thousands": 1_000_000 * (index + 1) + int(cik[-4:]),
                        "shares": 10_000 * (index + 1),
                        "put_call": "",
                        "report_date": report_date,
                        "filing_date": filing_date,
                        "company_name": f"INSTITUTION {cik}",
                    }
                )
    holdings = pl.DataFrame(rows)
    holdings.write_parquet(holdings_dir / "institutional_holdings.parquet")

    features, edges, *_ = producer.build_features_and_matrix(holdings, expected_ciks=ciks)
    edges.write_parquet(holdings_dir / "institution_stock_edges.parquet")
    features.write_parquet(holdings_dir / "stock_features.parquet")
    return holdings


def test_13f_builder_rejects_a_source_missing_a_requested_institution(tmp_path: Path) -> None:
    """A consistent artifact set can still be short one institution.

    22_rag/07 asks for all ten by CIK, so a holdings file that simply lacks one is
    a coverage gap the graph reads as an absent manager rather than as a broken
    fixture. Keyed on the holdings' own CIKs the rebuild would agree with itself
    and pass.
    """
    source = tmp_path / "source"
    holdings_dir = source / "equities" / "positioning" / "13f"
    holdings_dir.mkdir(parents=True)
    missing = _load_13f_producer().INSTITUTIONS[0][1]
    _write_13f_source(holdings_dir, drop_cik=missing)

    with pytest.raises(ValueError, match=rf"cover none of the filings of \['{missing}'\]"):
        build_institutional_holdings_13f(source, tmp_path / "out")


def test_13f_builder_rejects_derived_artifacts_the_holdings_do_not_produce(tmp_path: Path) -> None:
    """A complete column set is not the same artifact as the current producer's.

    22_rag/07 rebuilds edges and features and compares whole frames, so a source
    whose derived files carry stale values passes every name check here and fails
    the notebook instead.
    """
    source = tmp_path / "source"
    holdings_dir = source / "equities" / "positioning" / "13f"
    holdings_dir.mkdir(parents=True)
    _write_13f_source(holdings_dir)

    features_path = holdings_dir / "stock_features.parquet"
    pl.read_parquet(features_path).with_columns(
        pl.col("total_inst_value_usd") * 1000
    ).write_parquet(features_path)

    with pytest.raises(ValueError, match=r"stock_features\.parquet is not what the current"):
        build_institutional_holdings_13f(source, tmp_path / "out")


def test_overlapping_roots_are_rejected(tmp_path: Path) -> None:
    """--clean plus one root would delete the production data it then reads."""
    root = tmp_path / "data"
    nested = root / "fixtures"
    assert roots_overlap(root, root) is not None
    assert roots_overlap(root, nested) is not None
    assert roots_overlap(nested, root) is not None
    assert roots_overlap(root, tmp_path / "elsewhere") is None


def test_13f_builder_copies_a_complete_artifact_set(tmp_path: Path) -> None:
    source = tmp_path / "source"
    holdings_dir = source / "equities" / "positioning" / "13f"
    holdings_dir.mkdir(parents=True)
    _write_13f_source(holdings_dir)

    output = tmp_path / "out"
    written = build_institutional_holdings_13f(source, output)

    assert {path.name for path in written} == set(ctd_13f_required_columns())
    assert all(path.exists() for path in written)
    for path in written:
        assert set(ctd_13f_required_columns()[path.name]).issubset(pl.read_parquet(path).columns)
