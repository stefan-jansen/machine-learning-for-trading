"""A reconstruction and its published set hold the same instants in different dtypes.

`validate_locked_run` compares predictions reconstructed from checkpoint files against the
prediction set that was published from the same fit. Publishing relabels a naive decision
time as UTC and the parquet round-trip carries microseconds; the reconstruction has been
through neither. On 2026-09-06 that difference refused fx_pairs' holdout with "locked
sequence fitted state does not reproduce published predictions" while every value matched.
"""

import datetime as dt

import polars as pl

from case_studies.utils.deep_learning import align_keys_to_published

KEY_COLUMNS = ["symbol", "timestamp", "fold"]

TIMESTAMPS = [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)]


def _published() -> pl.DataFrame:
    """The contract a prediction set carries once it has been written and read back."""
    return pl.DataFrame(
        {
            "symbol": ["AUD_JPY", "AUD_JPY"],
            "timestamp": pl.Series(TIMESTAMPS, dtype=pl.Datetime("us", "UTC")),
            "fold": pl.Series([8, 8], dtype=pl.Int32),
            "prediction": pl.Series([-0.001249, -0.001499], dtype=pl.Float64),
        }
    )


def _reconstructed() -> pl.DataFrame:
    """The same rows straight out of `_reconstruct_pytorch_predictions`: naive nanoseconds."""
    return pl.DataFrame(
        {
            "symbol": ["AUD_JPY", "AUD_JPY"],
            "timestamp": pl.Series(TIMESTAMPS, dtype=pl.Datetime("ns")),
            "fold": pl.Series([8, 8], dtype=pl.Int32),
            "prediction": pl.Series([-0.001249, -0.001499], dtype=pl.Float32),
        }
    )


def test_unaligned_keys_compare_unequal() -> None:
    """Without the alignment the comparison fails, which is the defect being fixed.

    If this ever passes, the two frames no longer differ and the test below proves nothing.
    """
    published, reconstructed = _published(), _reconstructed()
    assert not reconstructed.select(KEY_COLUMNS).equals(published.select(KEY_COLUMNS))


def test_alignment_makes_identical_rows_compare_equal() -> None:
    published, reconstructed = _published(), _reconstructed()
    aligned = align_keys_to_published(reconstructed, published, KEY_COLUMNS)
    assert aligned.select(KEY_COLUMNS).equals(published.select(KEY_COLUMNS))


def test_alignment_relabels_rather_than_converting() -> None:
    """A naive value read as UTC keeps its wall time - the writer's own rule.

    A conversion instead of a relabel would shift every decision time by the host's offset
    and silently move which bar a prediction belongs to.
    """
    aligned = align_keys_to_published(_reconstructed(), _published(), KEY_COLUMNS)
    assert aligned["timestamp"].dt.replace_time_zone(None).to_list() == TIMESTAMPS


def test_alignment_leaves_matching_dtypes_alone() -> None:
    published = _published()
    aligned = align_keys_to_published(published, published, KEY_COLUMNS)
    assert aligned.equals(published)


def test_alignment_does_not_touch_value_columns() -> None:
    """Values are compared with `np.allclose`, which spans f32 and f64 on its own.

    Casting them here would hide a real numeric difference behind a widening.
    """
    aligned = align_keys_to_published(_reconstructed(), _published(), KEY_COLUMNS)
    assert aligned.schema["prediction"] == pl.Float32
