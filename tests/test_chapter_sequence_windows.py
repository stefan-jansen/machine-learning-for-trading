"""A sequence window must end on the row its target is dated.

``create_sequences_multi_asset`` and ``create_patched_sequences_multi_asset`` built
``features[i - lookback : i]`` against ``targets[i]``, so the features of the decision
date itself were never in the window. The labels these notebooks read are forward
returns dated at the decision row, which makes those features known when the
prediction is made: the ridge baseline each notebook scores against flattens the same
window, the tabular families take row ``i`` as their design row, and the Darts
forecasters condition on the series through ``i``. The sequence models were therefore
fitted on one period less history than everything they are compared with, and nothing
in the code or the chapter said so.

These tests pin the alignment on both builders. With the old slice they fail on the
last-row assertion.

The rule is about a design matrix, not about every window in the chapter.
``09_foundation_models.prepare_univariate_contexts`` deliberately ends its context at
``i - 1``: it forecasts the series forward from the decision date, so its forecast path
starts at ``i`` and spans the same days as the label. A window that is fed to a model as
features must include row ``i``; a history that is extrapolated past row ``i`` must not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_dl_sequences():
    spec = importlib.util.spec_from_file_location(
        "ch13_dl_sequences", REPO / "13_dl_time_series" / "dl_sequences.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dl_sequences():
    return _load_dl_sequences()


@pytest.fixture
def panel() -> pl.DataFrame:
    """Two symbols, 30 sessions, features that identify their own row."""
    dates = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 30), interval="1d", eager=True
    )
    frames = []
    for offset, symbol in enumerate(("AAA", "BBB")):
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": [symbol] * len(dates),
                    # f0 is the row's own position, so a window's contents name the rows
                    # it covers and an off-by-one is visible rather than inferred.
                    "f0": [float(i + 1000 * offset) for i in range(len(dates))],
                    "f1": [float(2 * i + 1000 * offset) for i in range(len(dates))],
                    "target": [float(i) / 100.0 for i in range(len(dates))],
                }
            )
        )
    return pl.concat(frames)


def test_window_ends_on_the_target_row(dl_sequences, panel: pl.DataFrame) -> None:
    lookback = 4
    X, y, timestamps, symbols = dl_sequences.create_sequences_multi_asset(
        panel, ["f0", "f1"], "target", lookback
    )
    assert X.shape == (len(X), lookback, 2)

    for window, target, timestamp, symbol in zip(X, y, timestamps, symbols):
        row = panel.filter((pl.col("symbol") == symbol) & (pl.col("timestamp") == timestamp))
        assert row.height == 1
        # The decision row's own features are the last position of its window.
        assert window[-1, 0] == pytest.approx(row["f0"][0])
        assert window[-1, 1] == pytest.approx(row["f1"][0])
        # And the window reaches back exactly lookback - 1 rows before it.
        assert window[0, 0] == pytest.approx(row["f0"][0] - (lookback - 1))
        assert target == pytest.approx(row["target"][0])


def test_patched_window_ends_on_the_target_row(dl_sequences, panel: pl.DataFrame) -> None:
    lookback, patch_size = 4, 2
    X, y, timestamps, symbols = dl_sequences.create_patched_sequences_multi_asset(
        panel, ["f0", "f1"], "target", lookback, patch_size
    )
    assert X.shape == (len(X), lookback // patch_size, patch_size * 2)

    for window, timestamp, symbol in zip(X, timestamps, symbols):
        row = panel.filter((pl.col("symbol") == symbol) & (pl.col("timestamp") == timestamp))
        # The patches are the window reshaped, so the decision row's f0 is the
        # third-from-last element of the last patch: [f0, f1] per position.
        flat = window.reshape(-1)
        assert flat[-2] == pytest.approx(row["f0"][0])
        assert flat[0] == pytest.approx(row["f0"][0] - (lookback - 1))


def test_both_builders_cover_the_same_rows(dl_sequences, panel: pl.DataFrame) -> None:
    lookback = 4
    plain = dl_sequences.create_sequences_multi_asset(panel, ["f0", "f1"], "target", lookback)
    patched = dl_sequences.create_patched_sequences_multi_asset(
        panel, ["f0", "f1"], "target", lookback, 2
    )
    order_plain = np.lexsort((plain[3].astype(str), plain[2]))
    order_patched = np.lexsort((patched[3].astype(str), patched[2]))
    assert np.array_equal(plain[2][order_plain], patched[2][order_patched])
    assert np.array_equal(plain[1][order_plain], patched[1][order_patched])
    assert np.allclose(
        plain[0][order_plain].reshape(len(plain[0]), -1),
        patched[0][order_patched].reshape(len(patched[0]), -1),
    )


def test_the_earliest_window_is_the_one_that_fits(dl_sequences, panel: pl.DataFrame) -> None:
    """The first full window ends on row lookback - 1, not on row lookback.

    Moving the window's end onto the target row freed one row at the start of each
    symbol: rows 0 to lookback - 1 are a complete window dated lookback - 1. Both
    loops kept starting at lookback, so each symbol's oldest usable sample was
    dropped - silently, because the output is still well formed.
    """
    lookback = 4
    for builder, args in (
        (dl_sequences.create_sequences_multi_asset, ()),
        (dl_sequences.create_patched_sequences_multi_asset, (2,)),
    ):
        X, _, timestamps, symbols = builder(panel, ["f0", "f1"], "target", lookback, *args)
        for symbol in ("AAA", "BBB"):
            own = timestamps[symbols == symbol]
            expected = panel.filter(pl.col("symbol") == symbol).sort("timestamp")["timestamp"]
            assert own.min() == expected[lookback - 1]
            assert len(own) == len(expected) - (lookback - 1)
        assert len(X) == 2 * (panel.height // 2 - (lookback - 1))


def test_a_symbol_with_exactly_one_window_is_kept(dl_sequences, panel: pl.DataFrame) -> None:
    """lookback rows are one window, and the length guard used to reject them.

    The guard required lookback + 1 rows, which was right when the window stopped
    before the target row and is one too many now. A symbol with exactly enough
    history for a single sample was dropped from the panel entirely.
    """
    lookback = 4
    short = panel.filter(pl.col("symbol") == "AAA").sort("timestamp").head(lookback)
    for builder, args in (
        (dl_sequences.create_sequences_multi_asset, ()),
        (dl_sequences.create_patched_sequences_multi_asset, (2,)),
    ):
        X, y, timestamps, _ = builder(short, ["f0", "f1"], "target", lookback, *args)
        assert len(X) == 1
        assert timestamps[0] == short["timestamp"][lookback - 1]
        assert y[0] == pytest.approx(short["target"][lookback - 1])
