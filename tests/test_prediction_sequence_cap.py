"""The prediction-side preview cap: `max_predict_sequences`.

`max_train_sequences` bounds how many windows a fold trains on. Nothing bounded how
many it *scores*: `prepare_fold_sequence_stores` built its validation index with a
hardcoded zero, and `sequence_validation_keys` had no cap to take. For an architecture
whose per-window forward pass is expensive the validation pass then scales with the
panel, which is why `nasdaq100_microstructure`'s patchtst notebook sat at its 600s cell
budget while its three siblings finished in 92-94s.

These tests pin the two halves separately and then together, because the failure mode
that matters is not "too slow" - it is the two halves disagreeing, which publishes a
prediction set that does not match the keys it was registered against.
"""

import numpy as np
import pandas as pd
import pytest

from case_studies.utils.sequence_dataset import (
    materialize_store_metadata,
    prepare_fold_sequence_stores,
    sequence_validation_keys,
)
from tests.test_sequence_dataset import _synthetic_fold_df

LOOKBACK = 20
FOLD = 3


def _fixture():
    df, train_mask, val_mask, val_start_ts, val_end_ts = _synthetic_fold_df()
    split = {
        "fold": FOLD,
        "train_start": pd.Timestamp("2020-01-01"),
        "train_end": pd.Timestamp("2020-12-31"),
        "val_start": val_start_ts,
        "val_end": val_end_ts,
    }
    return df, train_mask, val_mask, val_start_ts, split


def _declared(df, split, cap):
    return sequence_validation_keys(
        df,
        [split],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=LOOKBACK,
        max_predict_sequences=cap,
    )


def _stored(df, train_mask, val_mask, val_start_ts, cap):
    train_store, val_store, _ = prepare_fold_sequence_stores(
        df,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feat0", "feat1"],
        label_col="y",
        date_col="timestamp",
        entity_col="symbol",
        lookback=LOOKBACK,
        val_start=val_start_ts,
        max_predict_sequences=cap,
    )
    _, timestamps, symbols = materialize_store_metadata(val_store)
    keys = {
        (str(symbol), pd.Timestamp(timestamp), FOLD)
        for symbol, timestamp in zip(symbols, timestamps, strict=True)
    }
    return train_store, keys


class TestTheTwoHalvesAgree:
    """The declared keys and the scored windows must be the same set, capped or not."""

    @pytest.mark.parametrize("cap", [0, 12, 60])
    def test_declared_keys_equal_the_validation_store(self, cap):
        df, train_mask, val_mask, val_start_ts, split = _fixture()
        declared = set(_declared(df, split, cap).iter_rows())
        _, stored = _stored(df, train_mask, val_mask, val_start_ts, cap)
        assert declared == stored

    def test_a_cap_applied_to_only_one_half_desynchronises_them(self):
        """The regression this guards: capping keys without capping the store."""
        df, train_mask, val_mask, val_start_ts, split = _fixture()
        declared = set(_declared(df, split, 12).iter_rows())
        _, stored_uncapped = _stored(df, train_mask, val_mask, val_start_ts, 0)
        assert declared != stored_uncapped


class TestTheCapActuallyBinds:
    def test_it_draws_no_more_than_the_cap(self):
        df, _, _, _, split = _fixture()
        assert _declared(df, split, 12).height == 12

    def test_it_draws_strictly_fewer_than_uncapped(self):
        df, _, _, _, split = _fixture()
        assert _declared(df, split, 12).height < _declared(df, split, 0).height

    def test_zero_means_uncapped(self):
        """Canonical runs pass zero, so this is the no-change guarantee."""
        df, _, _, _, split = _fixture()
        without = sequence_validation_keys(
            df,
            [split],
            label_col="y",
            date_col="timestamp",
            entity_col="symbol",
            lookback=LOOKBACK,
        )
        assert set(_declared(df, split, 0).iter_rows()) == set(without.iter_rows())


class TestWhatTheCapMustNotDo:
    def test_every_symbol_survives_the_cap(self):
        """A cap that dropped symbols would change the cross-section, not just its size."""
        df, _, _, _, split = _fixture()
        uncapped = {row[0] for row in _declared(df, split, 0).iter_rows()}
        capped = {row[0] for row in _declared(df, split, 12).iter_rows()}
        assert capped == uncapped
        assert len(capped) == 3

    def test_it_leaves_the_training_store_alone(self):
        """It is a coverage reduction, not a model property: the fit must not move."""
        df, train_mask, val_mask, val_start_ts, _ = _fixture()
        uncapped, _ = _stored(df, train_mask, val_mask, val_start_ts, 0)
        capped, _ = _stored(df, train_mask, val_mask, val_start_ts, 12)
        np.testing.assert_array_equal(capped.end_idx, uncapped.end_idx)
        np.testing.assert_array_equal(capped.symbol_idx, uncapped.symbol_idx)
        np.testing.assert_array_equal(capped.feature_mean, uncapped.feature_mean)

    def test_a_cap_below_the_symbol_count_is_refused(self):
        """Silently dropping symbols to honour the number would be worse than failing."""
        df, _, _, _, split = _fixture()
        with pytest.raises(ValueError, match="cannot preserve full universe coverage"):
            _declared(df, split, 2)
