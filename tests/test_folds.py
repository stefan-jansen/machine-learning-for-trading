"""Shared fold preparation: what it produces, and that a change to it is declared.

``FOLD_PREPARATION_VERSION`` replaced a SHA-256 of the source file in the training identity, so
that a fix which changes no result does not invalidate every registered result. That trade only
holds if a change which *does* alter a result is caught, which is what the golden digest here is
for: it fails when preparation changes, and its message says to bump the version.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest

from case_studies.utils.folds import (
    FOLD_PREPARATION_VERSION,
    clear_memo,
    fold_cache_key,
    gbm_fold,
    prepare_raw_folds,
    prepare_standardized_folds,
    standardized_fold,
)
from utils.modeling import ModelingDataset

FEATURES = ["alpha", "beta", "gamma"]
SPLITS = [
    {
        "fold": 0,
        "train_start": "2020-01-01",
        "train_end": "2020-06-30",
        "val_start": "2020-07-01",
        "val_end": "2020-09-30",
    },
    {
        "fold": 1,
        "train_start": "2020-01-01",
        "train_end": "2020-09-30",
        "val_start": "2020-10-01",
        "val_end": "2020-12-31",
    },
]


def _dataset(*, missing: bool = False, empty_column: bool = False) -> ModelingDataset:
    """A small deterministic panel: three symbols on every business day of 2020."""
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 12, 31), interval="1d", eager=True
    ).to_list()
    symbols = ["AAA", "BBB", "CCC"]
    rows = len(dates) * len(symbols)
    generator = np.random.default_rng(0)
    frame = pl.DataFrame(
        {
            "timestamp": [date for date in dates for _ in symbols],
            "symbol": symbols * len(dates),
            "alpha": generator.normal(size=rows),
            "beta": generator.normal(size=rows) * 3.0 + 1.0,
            "gamma": generator.normal(size=rows),
            "fwd_ret_5d": generator.normal(size=rows) * 0.01,
        }
    )
    if missing:
        mask = pl.Series(generator.random(rows) < 0.1)
        frame = frame.with_columns(
            pl.when(mask).then(None).otherwise(pl.col("alpha")).alias("alpha")
        )
    if empty_column:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("gamma"))
    return ModelingDataset(
        dataset=frame,
        feature_names=list(FEATURES),
        label_col="fwd_ret_5d",
        date_col="timestamp",
        entity_cols=["symbol"],
        join_cols=["symbol", "timestamp"],
        splits=list(SPLITS),
        label_buffer="5d",
        case_study_id="",
    )


def _digest(folds) -> str:
    digest = hashlib.sha256()
    for fold in folds:
        for array in (fold.X_train, fold.y_train, fold.X_val, fold.y_val):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


@pytest.fixture(autouse=True)
def _clean_memo():
    clear_memo()
    yield
    clear_memo()


class TestWhatPreparationProduces:
    def test_every_declared_fold_is_prepared_with_the_declared_features(self) -> None:
        folds = prepare_raw_folds(_dataset(), SPLITS, use_cache=False)

        assert [fold.fold for fold in folds] == [0, 1]
        assert all(fold.X_train.shape[1] == len(FEATURES) for fold in folds)
        assert all(fold.X_val.shape[1] == len(FEATURES) for fold in folds)

    def test_validation_rows_fall_inside_the_declared_window(self) -> None:
        folds = prepare_raw_folds(_dataset(), SPLITS, use_cache=False)

        for fold, split in zip(folds, SPLITS, strict=True):
            dates = fold.meta["timestamp"].to_numpy()
            assert str(dates.min())[:10] >= split["val_start"]
            assert str(dates.max())[:10] <= split["val_end"]

    def test_a_later_fold_trains_on_more_rows_than_an_earlier_one(self) -> None:
        """Walk-forward means the training window grows; a fold set that does not is not one."""
        folds = prepare_raw_folds(_dataset(), SPLITS, use_cache=False)

        assert folds[1].n_train > folds[0].n_train

    def test_missing_feature_values_survive_preparation(self) -> None:
        """Imputation belongs to the family that cannot take a NaN, not to preparation.

        Gradient boosting routes a missing value down its own branch, and would be given a
        fabricated median if preparation filled it in.
        """
        folds = prepare_raw_folds(_dataset(missing=True), SPLITS, use_cache=False)

        assert np.isnan(folds[0].X_train).any()

    def test_subsampling_reduces_training_rows_and_never_validation_rows(self) -> None:
        full = prepare_raw_folds(_dataset(), SPLITS, use_cache=False)
        clear_memo()
        reduced = prepare_raw_folds(_dataset(), SPLITS, train_sample_frac=0.5, use_cache=False)

        assert reduced[0].n_train < full[0].n_train
        assert reduced[0].n_val == full[0].n_val

    def test_a_declared_fold_with_no_rows_raises(self) -> None:
        beyond = [{**SPLITS[0], "val_start": "2031-01-01", "val_end": "2031-12-31"}]

        with pytest.raises(ValueError, match="fold 0 is empty"):
            prepare_raw_folds(_dataset(), beyond, use_cache=False)


class TestTheFamilyAdapters:
    def test_standardising_centres_and_scales_the_training_rows(self) -> None:
        fold = standardized_fold(prepare_raw_folds(_dataset(), SPLITS, use_cache=False)[0])

        assert np.allclose(fold["X_train"].mean(axis=0), 0.0, atol=1e-12)
        assert np.allclose(fold["X_train"].std(axis=0), 1.0, atol=1e-12)

    def test_validation_rows_are_scaled_by_the_training_statistics(self) -> None:
        """A validation mean of exactly zero would mean validation statistics reached the fit."""
        fold = standardized_fold(prepare_raw_folds(_dataset(), SPLITS, use_cache=False)[0])

        assert not np.allclose(fold["X_val"].mean(axis=0), 0.0, atol=1e-6)

    def test_a_feature_missing_across_the_whole_training_window_raises(self) -> None:
        """It used to be dropped, narrowing the design matrix while the spec claimed otherwise."""
        raw = prepare_raw_folds(_dataset(empty_column=True), SPLITS, use_cache=False)

        with pytest.raises(ValueError, match="entirely.*missing"):
            standardized_fold(raw[0])

    def test_the_gradient_boosting_form_keeps_missing_values_and_native_precision(self) -> None:
        fold = gbm_fold(prepare_raw_folds(_dataset(missing=True), SPLITS, use_cache=False)[0])

        assert fold["X_train"].dtype == np.float32
        assert np.isnan(fold["X_train"]).any()

    def test_both_families_see_the_same_rows_in_the_same_order(self) -> None:
        """The point of sharing preparation: a linear and a GBM result are comparable."""
        raw = prepare_raw_folds(_dataset(), SPLITS, use_cache=False)

        assert np.array_equal(standardized_fold(raw[0])["y_val"], gbm_fold(raw[0])["y_val"])


class TestReuse:
    def test_preparing_the_same_folds_twice_returns_the_identical_arrays(self) -> None:
        dataset = _dataset()
        first = prepare_standardized_folds(dataset, SPLITS, use_cache=False)
        second = prepare_standardized_folds(dataset, SPLITS, use_cache=False)

        assert first[0]["X_train"] is second[0]["X_train"]

    def test_the_cache_key_ignores_everything_preparation_does_not_depend_on(self) -> None:
        """Two callers describing the same fold set differently must share the arrays.

        This used to compare the function with itself on byte-identical arguments, which measures
        determinism and nothing else. What it has to show is that the key survives the ways a
        caller can legitimately differ: a sequence type, a dict order, and a field on a split that
        preparation never reads.
        """
        common = {
            "case_study": "etfs",
            "label_col": "fwd_ret_5d",
            "eval_label_col": None,
            "input_lineage": {"fingerprint": "abc"},
            "seed": 42,
            "train_sample_frac": 1.0,
        }
        baseline = fold_cache_key(feature_names=FEATURES, splits=SPLITS, **common)

        assert fold_cache_key(feature_names=tuple(FEATURES), splits=SPLITS, **common) == baseline

        rearranged = [
            {key: split[key] for key in reversed(list(split))} | {"selected_by": "a model"}
            for split in SPLITS
        ]
        assert fold_cache_key(feature_names=FEATURES, splits=rearranged, **common) == baseline

    def test_the_cache_key_changes_with_the_feature_order(self) -> None:
        """The design matrix is built in this order, so it is an input, not a description."""
        common = {
            "case_study": "etfs",
            "label_col": "fwd_ret_5d",
            "eval_label_col": None,
            "splits": SPLITS,
            "input_lineage": {"fingerprint": "abc"},
            "seed": 42,
            "train_sample_frac": 1.0,
        }

        assert fold_cache_key(feature_names=FEATURES, **common) != fold_cache_key(
            feature_names=list(reversed(FEATURES)), **common
        )

    def test_the_cache_key_changes_with_the_sampling_fraction(self) -> None:
        common = {
            "case_study": "etfs",
            "label_col": "fwd_ret_5d",
            "eval_label_col": None,
            "feature_names": FEATURES,
            "splits": SPLITS,
            "input_lineage": {"fingerprint": "abc"},
            "seed": 42,
        }

        assert fold_cache_key(**common, train_sample_frac=1.0) != fold_cache_key(
            **common, train_sample_frac=0.5
        )

    def test_a_round_trip_through_the_cache_returns_the_same_numbers(self, tmp_path) -> None:
        dataset = _dataset()
        dataset.case_study_id = "fixture"
        dataset._input_lineage = {"fingerprint": "fixture"}
        import os

        os.environ["ML4T_FOLD_CACHE"] = str(tmp_path)
        try:
            written = prepare_raw_folds(dataset, SPLITS)
            digest = _digest(written)
            clear_memo()
            read_back = prepare_raw_folds(dataset, SPLITS)
        finally:
            os.environ.pop("ML4T_FOLD_CACHE", None)

        assert _digest(read_back) == digest


class TestTheDeclaredVersion:
    """The version in the training identity has to mean something.

    If this digest moves, preparation changed and every result fitted under the current
    ``FOLD_PREPARATION_VERSION`` describes a different computation than a rerun would produce.
    Bump the version in ``case_studies/utils/folds.py`` and update the constant below in the same
    commit, so the registry can tell the two apart. ``PRE_PRECISION_KEY`` in
    ``tests/test_feature_storage_dtype.py`` pins a fold cache key that also carries the version,
    so it moves with the same bump and has to be re-pinned in that commit as well.
    """

    GOLDEN_VERSION = 1

    def test_the_declared_version_matches_what_this_file_pins(self) -> None:
        assert FOLD_PREPARATION_VERSION == self.GOLDEN_VERSION, (
            "fold preparation declares a version this test does not pin; update GOLDEN_VERSION, "
            "PINNED_PREPARATION_DIGEST, and PRE_PRECISION_KEY in "
            "tests/test_feature_storage_dtype.py together"
        )

    def test_preparation_produces_the_pinned_numbers(self) -> None:
        digest = _digest(prepare_raw_folds(_dataset(), SPLITS, use_cache=False))

        assert digest == PINNED_PREPARATION_DIGEST, (
            f"fold preparation now produces {digest}, not {PINNED_PREPARATION_DIGEST}. If the "
            "change is intended, bump FOLD_PREPARATION_VERSION in case_studies/utils/folds.py "
            "and update this constant in the same commit - every result fitted under the old "
            "version was fitted on different arrays."
        )


PINNED_PREPARATION_DIGEST = "f85909b147e643cb"


class TestPreparationStreams:
    """The generator must hand out a fold before the next one is built.

    This is the property the peak memory depends on, and it is invisible to every other test
    here: collecting the generator gives byte-identical folds either way. What separates the
    two is *when* each is built, so that is what these assert. Under the arrangement this
    replaced - every consumer calling `prepare_raw_folds` - the whole raw set existed before a
    single fold was transformed, and on us_equities_panel that set was 31.03 GB of a 52.56 GB
    peak.
    """

    def test_the_first_fold_arrives_before_the_second_is_built(self):
        from case_studies.utils import folds as folds_module

        dataset = _dataset()
        before = folds_module._BUILT
        stream = folds_module.iter_raw_folds(dataset, SPLITS, use_cache=False)

        assert before == folds_module._BUILT, "construction started before the first next()"
        first = next(stream)
        assert first.fold == 0
        assert before + 1 == folds_module._BUILT, (
            "the whole set was built to yield one fold - the generator is not streaming"
        )
        second = next(stream)
        assert second.fold == 1
        assert before + 2 == folds_module._BUILT

    def test_streaming_and_collecting_agree_exactly(self):
        dataset = _dataset()
        collected = prepare_raw_folds(dataset, SPLITS, use_cache=False)
        clear_memo()
        from case_studies.utils.folds import iter_raw_folds

        streamed = list(iter_raw_folds(dataset, SPLITS, use_cache=False))

        assert _digest(streamed) == _digest(collected)
        assert [fold.fold for fold in streamed] == [fold.fold for fold in collected]

    def test_the_standardising_consumer_never_holds_the_raw_set(self):
        """One raw fold alive at a time, not `len(splits)` of them."""
        from case_studies.utils import folds as folds_module

        dataset = _dataset()
        alive: list[int] = []
        real = folds_module.standardized_fold

        def counting(raw):
            # `raw` plus the generator's own reference is 2; a materialised list would show the
            # whole set reachable from the caller's frame instead.
            alive.append(len(folds_module._RAW_MEMO))
            return real(raw)

        folds_module.standardized_fold = counting
        try:
            out = folds_module.prepare_standardized_folds(dataset, SPLITS, use_cache=False)
        finally:
            folds_module.standardized_fold = real

        assert len(out) == len(SPLITS)
        assert alive == [0, 0], "the raw set was memoised while the consumer was still reading it"
