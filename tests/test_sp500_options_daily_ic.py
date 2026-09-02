"""An empty IC series has two causes and they need different answers.

`cross_sectional_ic_series` writes a null `ic` for a date that carried fewer than `min_obs`
valid observations and for a date that carried enough with every prediction or every return
tied. Dropping the nulls collapses both into an empty frame. Reporting the first when the
second happened sends a reader to widen a panel that is already wide enough, and the real
cause then goes unlooked-at.

Both IC call sites in `90_ic_diagnostic` reach this function. They used to differ: one
raised a stated cause on an empty series and the other handed it straight to
`compute_ic_uncertainty`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from case_studies.sp500_options._ic_diagnostics import daily_ic

MIN_SYMBOLS = 10


def _panel(n_dates: int, n_symbols: int, *, constant_score: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = n_dates * n_symbols
    return pl.DataFrame(
        {
            "timestamp": np.repeat(
                np.array(
                    [np.datetime64("2024-01-01") + np.timedelta64(d, "D") for d in range(n_dates)]
                ),
                n_symbols,
            ),
            "symbol": [f"S{i:02d}" for i in range(n_symbols)] * n_dates,
            "y_score": np.zeros(rows) if constant_score else rng.normal(size=rows),
            "y_true": rng.normal(size=rows),
        }
    )


def _daily_ic(panel: pl.DataFrame) -> pl.DataFrame:
    return daily_ic(
        panel,
        pred_col="y_score",
        ret_col="y_true",
        min_symbols_per_date=MIN_SYMBOLS,
        described_as="the test panel",
    )


def test_a_wide_enough_panel_returns_one_row_per_scored_date() -> None:
    scored = _daily_ic(_panel(n_dates=6, n_symbols=MIN_SYMBOLS + 2))

    assert scored.height == 6
    assert scored["ic"].null_count() == 0


def test_a_narrow_panel_is_reported_as_breadth() -> None:
    with pytest.raises(RuntimeError, match=r"no date in the validation panel carries 10 names"):
        _daily_ic(_panel(n_dates=6, n_symbols=MIN_SYMBOLS - 2))


def test_a_wide_panel_with_a_constant_feature_is_not_reported_as_breadth() -> None:
    """The regression this file exists for.

    Every date here carries twelve names, so breadth is not the problem; the feature is
    constant within each date, so the rank correlation has a zero denominator. The old
    message named the floor and the panel's symbol count, both of which are fine.
    """
    with pytest.raises(RuntimeError) as raised:
        _daily_ic(_panel(n_dates=6, n_symbols=MIN_SYMBOLS + 2, constant_score=True))

    message = str(raised.value)
    assert "breadth is not the problem" in message
    assert "6 of 6 dates carry 10 names" in message
    assert "no date in the validation panel carries" not in message


def test_a_mixed_panel_scores_the_dates_that_qualify() -> None:
    """One narrow date among wide ones is dropped, not raised on."""
    wide = _panel(n_dates=3, n_symbols=MIN_SYMBOLS + 2)
    narrow = _panel(n_dates=1, n_symbols=MIN_SYMBOLS - 2).with_columns(
        pl.col("timestamp") + pl.duration(days=100)
    )

    scored = _daily_ic(pl.concat([wide, narrow]))

    assert scored.height == 3
