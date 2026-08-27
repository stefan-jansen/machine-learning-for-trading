"""What a holdout retrain is expected to cover, and how that expectation is compared.

`generate_holdout` registered its predictions without ever stating what they should have
contained, which is how a registry call that was simply malformed came back as two
candidates rejected on their merits, with the single holdout use heading to rank-3. The
two functions here are what replaced that: one declares the coverage before any model
runs, the other makes the declaration comparable to what comes back through pandas.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.registry.completeness import evaluate_prediction_coverage

_SPEC = importlib.util.spec_from_file_location(
    "strategy_synthesis_holdout",
    Path(__file__).resolve().parents[1] / "20_strategy_synthesis" / "holdout.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_HOLDOUT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HOLDOUT
_SPEC.loader.exec_module(_HOLDOUT)

_align_expected_timestamps = _HOLDOUT._align_expected_timestamps
_holdout_expected_keys = _HOLDOUT._holdout_expected_keys

SPLIT = {"fold": 0, "val_start": "2016-01-01", "val_end": "2016-12-31"}


@dataclass
class FakeDataset:
    """The four attributes `_holdout_expected_keys` reads off a modeling dataset."""

    dataset: pl.DataFrame
    date_col: str = "timestamp"
    label_col: str = "fwd_ret_1m"
    entity_cols: tuple[str, ...] = ("symbol",)


def _panel(rows: list[tuple[str, date, float | None]]) -> FakeDataset:
    return FakeDataset(
        pl.DataFrame(
            {
                "symbol": [r[0] for r in rows],
                "timestamp": [r[1] for r in rows],
                "fwd_ret_1m": [r[2] for r in rows],
                "feature": [0.5] * len(rows),
            },
            schema_overrides={"fwd_ret_1m": pl.Float64},
        )
    )


class TestWhatIsDeclared:
    def test_the_window_rows_with_a_finite_label_are_the_expectation(self) -> None:
        mds = _panel(
            [
                ("AAA", date(2015, 12, 31), 0.01),  # before the window
                ("AAA", date(2016, 1, 29), 0.02),
                ("BBB", date(2016, 1, 29), 0.03),
                ("AAA", date(2016, 6, 30), None),  # no label to predict against
                ("BBB", date(2016, 6, 30), float("nan")),
                ("AAA", date(2017, 1, 31), 0.04),  # after the window
            ]
        )
        expected = _holdout_expected_keys(mds, SPLIT)
        assert sorted(expected.rows()) == [
            ("AAA", date(2016, 1, 29), 0),
            ("BBB", date(2016, 1, 29), 0),
        ]

    def test_the_declaration_does_not_come_from_the_predictions(self) -> None:
        # The point of building it from the dataset: a model that answered for one of
        # the two names it was asked about is measured as short, not as complete.
        mds = _panel(
            [
                ("AAA", date(2016, 1, 29), 0.02),
                ("BBB", date(2016, 1, 29), 0.03),
            ]
        )
        expected = _holdout_expected_keys(mds, SPLIT)
        predictions = pl.DataFrame(
            {
                "symbol": ["AAA"],
                "timestamp": [date(2016, 1, 29)],
                "fold_id": [0],
                "y_score": [0.11],
            }
        )
        coverage = evaluate_prediction_coverage(expected, predictions)
        assert not coverage.complete
        assert coverage.n_missing == 1
        assert coverage.n_expected == 2

    def test_a_window_with_nothing_to_predict_raises(self) -> None:
        mds = _panel([("AAA", date(2016, 1, 29), None), ("BBB", date(2015, 1, 29), 0.02)])
        with pytest.raises(ValueError, match="no rows with a finite fwd_ret_1m"):
            _holdout_expected_keys(mds, SPLIT)

    def test_a_panel_holding_one_key_twice_raises(self) -> None:
        # A duplicated key would make every coverage comparison against it meaningless,
        # and `evaluate_prediction_coverage` refuses one anyway - raising here says which
        # side is at fault.
        mds = _panel([("AAA", date(2016, 1, 29), 0.02), ("AAA", date(2016, 1, 29), 0.03)])
        with pytest.raises(ValueError, match="duplicate expected prediction keys"):
            _holdout_expected_keys(mds, SPLIT)


class TestHowItIsCompared:
    """Coverage compares keys as strings, so the two frames must agree on representation."""

    @staticmethod
    def _predictions(days: list[date], dtype: pl.DataType) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": ["AAA"] * len(days),
                "timestamp": pl.Series(days).cast(dtype),
                "fold_id": [0] * len(days),
                "y_score": [0.11] * len(days),
            }
        )

    def test_a_date_and_a_millisecond_timestamp_hold_the_same_day(self) -> None:
        mds = _panel([("AAA", date(2016, 1, 29), 0.02)])
        expected = _holdout_expected_keys(mds, SPLIT)
        assert expected.schema["timestamp"] == pl.Date
        predictions = self._predictions([date(2016, 1, 29)], pl.Datetime("ms"))

        assert not evaluate_prediction_coverage(expected, predictions).complete

        aligned = _align_expected_timestamps(expected, predictions)
        assert aligned.schema["timestamp"] == pl.Datetime("ms")
        assert evaluate_prediction_coverage(aligned, predictions).complete

    def test_aligning_the_representation_does_not_excuse_the_wrong_day(self) -> None:
        mds = _panel([("AAA", date(2016, 1, 29), 0.02)])
        expected = _holdout_expected_keys(mds, SPLIT)
        predictions = self._predictions([date(2016, 1, 28)], pl.Datetime("ms"))

        coverage = evaluate_prediction_coverage(
            _align_expected_timestamps(expected, predictions), predictions
        )
        assert not coverage.complete
        assert coverage.n_missing == 1
        assert coverage.n_extra == 1

    def test_predictions_that_already_agree_are_left_alone(self) -> None:
        mds = _panel([("AAA", date(2016, 1, 29), 0.02)])
        expected = _holdout_expected_keys(mds, SPLIT)
        predictions = self._predictions([date(2016, 1, 29)], pl.Date)
        assert _align_expected_timestamps(expected, predictions) is expected
