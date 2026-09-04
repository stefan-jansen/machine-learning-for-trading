"""A row the temporal artifact does not cover must not reach the estimator.

``model_based.parquet`` carries one fold geometry, written by the primary label. A label whose
window is wider has rows the artifact never covered, and the join at
``case_studies/utils/folds.py`` is a LEFT join, so those rows survive with every temporal feature
null. Nothing downstream removed them: the filter after the join tests the LABEL for null, and
the sequence path calls ``np.nan_to_num(..., nan=0.0)``, which after per-feature normalization
hands the model the feature's mean. So an uncovered row was fitted as an average observation
rather than as missing, and no error, warning or metric showed it.

``validate_temporal_alignment`` already declines to score a leading warm-up prefix - it removes
it from the coverage denominator - but nothing removed it from the fit. These tests hold the
two halves together: what the guard refuses to measure is not fitted.

Measured on ``etfs`` when this was written: 5,244 of 1,893,267 training rows over 8 folds, in
all 64 registered ``fwd_ret_5d`` fits.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from utils.modeling import drop_uncovered_temporal_pl as _drop_uncovered_temporal

KEYS = ["symbol", "timestamp"]


def _dates(n: int, start: int = 1) -> list[dt.date]:
    """A window long enough that a realistic warm-up trim stays inside the guard's excuse.

    The drop is bounded by MAX_TEMPORAL_WARMUP_FRACTION (10%), so a fixture of ten dates cannot
    express "a few uncovered rows at the head" - three of ten is 30% and is refused by design.
    """
    base = dt.date(2020, 1, 1) + dt.timedelta(days=start - 1)
    return [base + dt.timedelta(days=i) for i in range(n)]


def _joined(dates: list[dt.date], symbols: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [s for s in symbols for _ in dates],
            "timestamp": [d for _ in symbols for d in dates],
            "y": [1.0] * (len(dates) * len(symbols)),
        }
    )


def _artifact(dates: list[dt.date], symbols: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    frame = _joined(dates, symbols).drop("y")
    return frame.with_columns(pl.lit(0.5).alias("garch_cond_vol"))


def test_a_leading_prefix_the_artifact_never_covered_is_removed():
    """The etfs shape: the window starts before the artifact's first fold row."""
    window = _dates(100)
    covered = _dates(100)[6:]  # artifact starts 3 sessions late, both symbols
    kept = _drop_uncovered_temporal(
        _joined(window),
        _artifact(covered),
        keys=KEYS,
        entity_col="symbol",
        date_col="timestamp",
        what="fold 0 train",
    )
    assert kept.height == 188, "94 covered dates x 2 symbols"
    assert kept.get_column("timestamp").min() == covered[0]
    # The property the sequence builder depends on: what remains is contiguous per symbol.
    for symbol in ("A", "B"):
        got = kept.filter(pl.col("symbol") == symbol).get_column("timestamp").sort().to_list()
        assert got == covered


def test_a_fully_covered_window_is_returned_unchanged():
    """The fix must not touch a case study whose artifact covers its window."""
    window = _dates(100)
    joined = _joined(window)
    kept = _drop_uncovered_temporal(
        joined,
        _artifact(window),
        keys=KEYS,
        entity_col="symbol",
        date_col="timestamp",
        what="fold 0 train",
    )
    assert kept.height == joined.height
    assert kept.sort(KEYS).equals(joined.sort(KEYS))


def test_a_ragged_left_edge_across_symbols_is_still_a_trim():
    """etfs' fold 0 has six distinct first-dates over its symbols. Per symbol it is contiguous."""
    window = _dates(100)
    artifact = pl.concat([_artifact(_dates(100)[4:], ("A",)), _artifact(_dates(100)[9:], ("B",))])
    kept = _drop_uncovered_temporal(
        _joined(window),
        artifact,
        keys=KEYS,
        entity_col="symbol",
        date_col="timestamp",
        what="fold 0 train",
    )
    assert kept.filter(pl.col("symbol") == "A").height == 96
    assert kept.filter(pl.col("symbol") == "B").height == 91


def test_an_interior_hole_raises_rather_than_silently_reshaping_the_training_set():
    """Dropping k interior rows costs k + lookback sequence windows, not k rows."""
    window = _dates(100)
    covered = [d for i, d in enumerate(window) if i not in (40, 41)]
    with pytest.raises(ValueError, match="interior gaps"):
        _drop_uncovered_temporal(
            _joined(window),
            _artifact(covered),
            keys=KEYS,
            entity_col="symbol",
            date_col="timestamp",
            what="fold 0 train",
        )


def test_an_uncovered_window_says_so_rather_than_returning_nothing():
    window = _dates(100)
    with pytest.raises(ValueError, match="covers none of the window"):
        _drop_uncovered_temporal(
            _joined(window),
            _artifact(_dates(4, start=900)),
            keys=KEYS,
            entity_col="symbol",
            date_col="timestamp",
            what="fold 0 train",
        )


class TestThePandasPathSharedByEveryOtherFamily:
    """`split_frames` is not the only join site.

    GBM, TabM, the shared modeling loops and the sequence builder all reach the artifact through
    `replace_temporal_columns`, never through `split_frames`. Fixing only the polars side would
    have left every one of those fitting uncovered rows - and the sequence families are where it
    costs most, because `sequence_dataset.py:217` turns the row into the feature's mean.
    """

    @staticmethod
    def _dataset(dates, symbols=("A", "B")):
        import pandas as pd

        frame = pd.DataFrame(
            [
                {"symbol": s, "timestamp": d, "garch": -1.0, "regime": -1.0}
                for s in symbols
                for d in dates
            ]
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).astype("datetime64[ms]")
        return frame

    @staticmethod
    def _artifact(dates, symbols=("A", "B"), null_features=False):
        import pandas as pd

        value = None if null_features else 0.5
        frame = pd.DataFrame(
            [
                {"fold": 0, "symbol": s, "timestamp": d, "garch": value, "regime": value}
                for s in symbols
                for d in dates
            ]
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).astype("datetime64[ms]")
        return frame

    def _call(self, dataset, artifact, **kw):
        import numpy as np

        from utils.modeling import replace_temporal_columns

        return replace_temporal_columns(
            dataset,
            np.ones(len(dataset), dtype=bool),
            artifact,
            ["symbol", "timestamp"],
            ["garch", "regime"],
            0,
            date_col="timestamp",
            entity_col="symbol",
            what="fold 0 train",
            **kw,
        )

    def test_the_uncovered_prefix_is_removed_when_asked(self):
        import pandas as pd

        dates = _dates(100)
        kept = self._call(self._dataset(dates), self._artifact(dates[6:]), drop_uncovered=True)
        assert len(kept) == 188
        assert kept["timestamp"].min() == pd.Timestamp(dates[6])

    def test_validation_is_not_trimmed_by_default(self):
        """The default must stay False: a prediction set has to cover its declared sessions."""
        dates = _dates(100)
        rows = self._call(self._dataset(dates), self._artifact(dates[6:]))
        assert len(rows) == 200, "an uncovered validation row is a stop, not a silent drop"

    def test_a_covered_row_whose_features_are_all_null_survives(self):
        """Coverage is membership in the artifact, not a test on the feature values.

        Temporal fitting can skip an entity and leave every feature null on a row the artifact
        genuinely carries. Inferring coverage from the values would delete it - a second defect
        wearing the first one's clothes.
        """
        dates = _dates(6)
        kept = self._call(
            self._dataset(dates),
            self._artifact(dates, null_features=True),
            drop_uncovered=True,
        )
        assert len(kept) == 12, "an all-null but covered row is not an uncovered row"

    def test_an_interior_hole_raises(self):
        dates = _dates(100)
        covered = [d for i, d in enumerate(dates) if i not in (40, 41)]
        with pytest.raises(ValueError, match="interior gaps"):
            self._call(self._dataset(dates), self._artifact(covered), drop_uncovered=True)

    def test_the_bounded_trim_keeps_validation_rows_the_artifact_misses(self):
        """Darts overlays one frame spanning train_start..val_end and splits it afterwards.

        The trim is bounded to `train_end` there, so an uncovered row in the validation half
        survives to raise through the coverage guard instead of being silently removed - the
        same rule the two-mask callers get for free.
        """
        dates = _dates(100)
        covered = dates[4:60]  # missing at BOTH ends of the combined frame
        kept = self._call(
            self._dataset(dates),
            self._artifact(covered),
            drop_uncovered=True,
            drop_uncovered_through=dates[59],
        )
        got = sorted({d.date() for d in kept["timestamp"]})
        assert got == dates[4:], (
            "the leading uncovered rows go, the trailing ones stay for the guard to catch"
        )


def test_a_trim_larger_than_the_guard_excuses_raises_in_the_polars_path():
    """The drop must stop where the coverage guard stops excusing.

    `validate_temporal_alignment` excuses a leading gap up to MAX_TEMPORAL_WARMUP_FRACTION and
    refuses anything larger. This drop removes what that guard declines to score, so a more
    permissive bound would silently shorten a training window the guard would have failed -
    `us_equities_panel/12_dl_weekly.py` supplies a fold opening about five years before its
    artifact fold does.
    """
    window = _dates(100)
    kept_dates = _dates(100)[40:]  # 40% uncovered, far past the 10% excuse
    with pytest.raises(ValueError, match="past the 10% the coverage guard will excuse"):
        _drop_uncovered_temporal(
            _joined(window),
            _artifact(kept_dates),
            keys=KEYS,
            entity_col="symbol",
            date_col="timestamp",
            what="fold 0 train",
        )


def test_a_trim_within_the_excuse_is_still_allowed():
    window = _dates(100)
    kept = _drop_uncovered_temporal(
        _joined(window),
        _artifact(_dates(100)[5:]),  # 5%, inside the excuse
        keys=KEYS,
        entity_col="symbol",
        date_col="timestamp",
        what="fold 0 train",
    )
    assert kept.height == 190


def test_the_overlay_scopes_its_contiguity_check_per_symbol():
    """Artifact coverage is ragged across symbols; the check must not read that as a hole.

    etfs' fold 0 has six distinct artifact first-dates across its 100 symbols. Checked globally,
    one symbol's legitimate leading trim falls strictly inside another symbol's date range and
    reads as an interior hole, so a fold that is fine is refused. This is what passing
    `entity_col` through every `_overlay_fold_temporal_features` call site buys.
    """
    import pandas as pd

    from case_studies.utils.darts_forecasting import _overlay_fold_temporal_features

    dates = pd.to_datetime(_dates(100))
    dataset = pd.DataFrame(
        {
            "timestamp": [*dates, *dates],
            "symbol": ["A"] * len(dates) + ["B"] * len(dates),
            "garch_cond_vol": [-1.0] * (2 * len(dates)),
        }
    )
    # A is covered from index 4, B from index 9 - a trim for each, no hole in either.
    covered = [(s, d) for s, cut in (("A", 4), ("B", 9)) for d in dates[cut:]]
    temporal = pd.DataFrame(
        {
            "timestamp": [d for _, d in covered],
            "symbol": [s for s, _ in covered],
            "fold": [0] * len(covered),
            "garch_cond_vol": [0.5] * len(covered),
        }
    )

    fold = _overlay_fold_temporal_features(
        dataset,
        {
            "fold": 0,
            "train_start": dates[0],
            "train_end": dates[-1],
            "val_end": dates[-1],
        },
        "timestamp",
        temporal,
        ["timestamp", "symbol"],
        ["garch_cond_vol"],
        "symbol",
    )

    assert (fold["symbol"] == "A").sum() == 96
    assert (fold["symbol"] == "B").sum() == 91
