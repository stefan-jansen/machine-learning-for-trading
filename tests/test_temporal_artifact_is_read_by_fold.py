"""The per-fold temporal artifact is selected from, not materialised.

``load_modeling_dataset`` cost 38.05 GB on ``us_equities_panel`` and returned 14.2 GB of live
data. The largest object it held was the per-fold model-based feature artifact - 68,684,394 rows
against the panel's own 9,921,350 - read eagerly and then copied again by ``.to_pandas()``, so
both forms were live through every join. No consumer wants it whole: all of them select one fold.

These tests hold the artifact to being read one fold at a time, and hold the four selection
rules that were merged into :func:`fold_temporal_frame` to agreeing with each other. A reader
who cannot allocate 80 GB is the reason both properties matter.
"""

from __future__ import annotations

import gc

import polars as pl
import pytest

from case_studies.utils.runtime import peak_rss_bytes
from utils.modeling import fold_temporal_frame, temporal_fold_index

N_FOLDS = 8
ROWS_PER_FOLD = 60_000
FEATURES = 12


DUPLICATE_KEY = ("S000", 0)


def _artifact() -> pl.DataFrame:
    """Every fold carries one repeated key, so deduplication has something to do.

    Without it the assertion that ``fold_temporal_frame`` deduplicates holds whether or not it
    does, and nothing checks that ``keep="last"`` is the row that survives.
    """
    rows = N_FOLDS * ROWS_PER_FOLD
    frame = pl.DataFrame(
        {
            "fold": [f for f in range(N_FOLDS) for _ in range(ROWS_PER_FOLD)],
            "symbol": [f"S{i % 500:03d}" for i in range(rows)],
            "timestamp": [i % ROWS_PER_FOLD for i in range(rows)],
            **{f"t{j}": [float(i * j) for i in range(rows)] for j in range(FEATURES)},
        }
    )
    repeats = pl.DataFrame(
        {
            "fold": list(range(N_FOLDS)),
            "symbol": [DUPLICATE_KEY[0]] * N_FOLDS,
            "timestamp": [DUPLICATE_KEY[1]] * N_FOLDS,
            # Distinguishable, and appended last, so "last" is identifiable.
            **{f"t{j}": [-999.0] * N_FOLDS for j in range(FEATURES)},
        }
    )
    return pl.concat([frame, repeats.select(frame.columns)])


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("temporal") / "model_based.parquet"
    _artifact().write_parquet(path)
    return str(path)


class TestOneFoldIsWhatComesBack:
    @pytest.mark.parametrize("fold_id", [0, 3, N_FOLDS - 1])
    def test_only_the_requested_fold_is_returned(self, artifact_path, fold_id):
        frame = fold_temporal_frame(pl.scan_parquet(artifact_path), fold_id)
        assert frame.height == ROWS_PER_FOLD + 1  # the fold's rows plus its repeated key
        assert "fold" not in frame.columns

    def test_a_fold_that_is_not_there_comes_back_empty_rather_than_wrong(self, artifact_path):
        assert fold_temporal_frame(pl.scan_parquet(artifact_path), N_FOLDS + 5).is_empty()

    def test_the_fold_index_does_not_read_the_features(self, artifact_path):
        index = temporal_fold_index(pl.scan_parquet(artifact_path), "timestamp")
        assert set(index.columns) == {"timestamp", "fold"}
        assert sorted(index["fold"].unique().to_list()) == list(range(N_FOLDS))


class TestTheThreeFormsAgree:
    """Four selection rules were merged into one. They must have been saying the same thing."""

    @pytest.fixture(scope="class")
    def forms(self, artifact_path):
        eager = pl.read_parquet(artifact_path)
        return {
            "lazy": pl.scan_parquet(artifact_path),
            "eager": eager,
            "pandas": eager.to_pandas(),
        }

    @pytest.mark.parametrize("form", ["lazy", "eager", "pandas"])
    def test_every_form_selects_the_same_rows(self, forms, form):
        expected = fold_temporal_frame(forms["eager"], 2)
        assert fold_temporal_frame(forms[form], 2).equals(expected)

    @pytest.mark.parametrize("form", ["lazy", "eager", "pandas"])
    def test_every_form_deduplicates_on_the_keys(self, forms, form):
        keys = ["symbol", "timestamp"]
        frame = fold_temporal_frame(forms[form], 1, temporal_keys=keys)
        assert frame.height == ROWS_PER_FOLD, "the repeated key must have been collapsed"
        assert frame.height == frame.unique(subset=keys).height

    @pytest.mark.parametrize("form", ["lazy", "eager", "pandas"])
    def test_the_row_that_survives_deduplication_is_the_last(self, forms, form):
        """``keep="last"`` is the artifact writer's convention, so the later row must win."""
        keys = ["symbol", "timestamp"]
        frame = fold_temporal_frame(forms[form], 1, temporal_keys=keys)
        survivor = frame.filter(
            (pl.col("symbol") == DUPLICATE_KEY[0]) & (pl.col("timestamp") == DUPLICATE_KEY[1])
        )
        assert survivor.height == 1
        assert survivor["t1"].item() == -999.0

    def test_the_join_keys_are_cast_to_the_frame_they_will_join_to(self, artifact_path):
        target = pl.DataFrame({"symbol": ["S001"], "timestamp": pl.Series([1], dtype=pl.Int32)})
        frame = fold_temporal_frame(
            pl.scan_parquet(artifact_path),
            0,
            temporal_keys=["symbol", "timestamp"],
            schema=target.schema,
        )
        assert frame.schema["timestamp"] == pl.Int32
        # An artifact whose key width does not match joins to nothing and presents as missing
        # values rather than as an error, so the cast is what makes the join mean anything.
        assert not target.join(frame, on=["symbol", "timestamp"], how="inner").is_empty()


class TestTheArtifactIsNotMaterialised:
    def test_selecting_a_fold_costs_a_fold_not_the_table(self, artifact_path):
        """The regression guard. Re-adding a ``.to_pandas()`` on the artifact fails here.

        ``peak_rss_bytes`` is a high-water mark that never falls, so the baseline is taken before
        anything in this test has read the artifact whole - reading it first would pay the peak
        up front and leave the assertion unable to fail. The budget is derived from one fold,
        not from the table, for the same reason.
        """
        gc.collect()
        before = peak_rss_bytes()

        one_at_a_time = None
        for fold_id in range(N_FOLDS):
            one_at_a_time = fold_temporal_frame(pl.scan_parquet(artifact_path), fold_id)
        gc.collect()
        after = peak_rss_bytes()
        assert one_at_a_time is not None
        growth = after - before

        # Only now is the whole artifact read, to say what a full materialisation would cost.
        whole = pl.read_parquet(artifact_path)
        artifact_bytes = whole.estimated_size()
        del whole
        budget = artifact_bytes / 2

        assert growth < budget, (
            f"selecting {N_FOLDS} folds one at a time grew peak memory by {growth / 1e6:.0f} MB, "
            f"against a budget of {budget / 1e6:.0f} MB - half the {artifact_bytes / 1e6:.0f} MB "
            "artifact. The fold predicate is no longer reaching the scan."
        )
