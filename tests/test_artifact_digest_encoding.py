"""The parquet encoding identity-defining artifacts are written under, pinned to its bytes.

Registry training identity digests these files' *bytes*: `utils/modeling.py::_sha256_file`
feeds `computation.feature_artifacts` and `computation.input_data_spec.artifacts`, both
inside the hashed `computation` block. Two files holding identical data hash differently
if they were written with a different compression codec or row-group size, so a polars
upgrade that moves a default would fork the `training_hash` of every run reading that
artifact - cached runs stop matching, the sweep refits everything, and no number has moved
to say why.

Repeated writes with the same settings are byte-stable, so this only bites on a settings
change or a library upgrade. That is what these tests catch, on the day it happens rather
than on the day a registry is found holding two identities for one piece of work.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from case_studies.utils.artifact_digest import _PARQUET_WRITE_SETTINGS, value_digest, write_artifact

# Recorded 2026-09-03 against polars 1.41.1. A change here means the encoding moved: check
# whether `_PARQUET_WRITE_SETTINGS` still names what the writer does, and if the bytes
# genuinely changed, every feature and label artifact re-written afterwards takes a new
# `training_hash`. Do not re-record it without deciding that.
PINNED_SHA256 = "29e7559707131249e0856c536f1152d9933e08329e532b2b886eb75ee260185a"


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"] * 100,
            "timestamp": list(range(300)),
            "value": [i / 7 for i in range(300)],
        }
    )


def _written(tmp_path: Path, name: str = "artifact.parquet") -> Path:
    path = tmp_path / name
    write_artifact(_frame(), path, keys=("symbol", "timestamp"), written_by="test")
    return path


def test_the_bytes_a_fixed_frame_produces_are_the_recorded_ones(tmp_path: Path) -> None:
    assert hashlib.sha256(_written(tmp_path).read_bytes()).hexdigest() == PINNED_SHA256


def test_the_settings_are_stated_rather_than_inherited(tmp_path: Path) -> None:
    """Stating polars' own defaults moves nothing today and pins them against a change."""
    library_default = tmp_path / "default.parquet"
    _frame().write_parquet(library_default)

    assert _written(tmp_path).read_bytes() == library_default.read_bytes()


def test_the_encoding_is_what_the_byte_digest_is_sensitive_to(tmp_path: Path) -> None:
    """The failure this pins: same values, different codec, different file digest.

    `value_digest` answers the same for all three, which is why it is the right identity
    for a new identity version - and why the byte digest needs the encoding pinned until
    one exists.
    """
    frame = _frame()
    paths = {}
    for name, kwargs in {
        "default": {},
        "lz4": {"compression": "lz4"},
        "row_groups": {"row_group_size": 50},
    }.items():
        path = tmp_path / f"{name}.parquet"
        frame.write_parquet(path, **kwargs)
        paths[name] = path

    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    assert len(set(digests.values())) == 3

    values = {name: value_digest(pl.read_parquet(path)) for name, path in paths.items()}
    assert len(set(values.values())) == 1


def test_the_settings_map_names_only_parameters_the_writer_accepts() -> None:
    import inspect

    accepted = set(inspect.signature(pl.DataFrame.write_parquet).parameters)
    assert set(_PARQUET_WRITE_SETTINGS) <= accepted


def test_the_chunk_layout_a_frame_arrives_in_does_not_move_the_bytes(tmp_path: Path) -> None:
    """`row_group_size` is left at the library's own default, so this is measured not assumed.

    Setting a concrete value would change what the writer emits today and re-key every
    training run that reads one of these artifacts - the exact harm the pin exists to
    prevent - so the question is whether the default is layout-sensitive. It is not: a
    frame assembled from three chunks writes the same bytes as its rechunked self. Recorded
    here so that if a future polars makes the layout matter, it arrives as a failing test
    rather than as two identities for one piece of work.
    """
    frame = _frame()
    chunked = pl.concat([frame[:100], frame[100:200], frame[200:]], rechunk=False)
    assert chunked.n_chunks() > 1 and frame.rechunk().n_chunks() == 1

    single = tmp_path / "single.parquet"
    multi = tmp_path / "multi.parquet"
    write_artifact(frame.rechunk(), single, keys=("symbol", "timestamp"), written_by="test")
    write_artifact(chunked, multi, keys=("symbol", "timestamp"), written_by="test")

    assert single.read_bytes() == multi.read_bytes()
    assert hashlib.sha256(multi.read_bytes()).hexdigest() == PINNED_SHA256
