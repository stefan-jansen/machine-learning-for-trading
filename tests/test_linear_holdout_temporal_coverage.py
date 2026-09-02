"""What the linear family asks of a fold-scoped temporal artifact at the holdout.

`crypto_perps_funding` and `sp500_options` both carry a linear rank-1 and both refused their
own holdout refit here, with "custom CV is incompatible with fold-scoped temporal features".
The check was asking whether the stage-04 artifact declares a fold with the holdout's geometry.
It never does: the holdout fold is derived after stage 04 ran. The features are joined by
(entity, date), so the question that can be answered - and the one that matters - is whether
the artifact holds rows spanning the dates the run trains and evaluates on.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from case_studies.utils.linear import _require_holdout_temporal_features

TIMESTAMPS = pl.datetime_range(
    pl.datetime(2020, 1, 1), pl.datetime(2024, 12, 31), interval="1mo", eager=True
)

# The artifact's own folds are the VALIDATION folds. There is no fold whose declared boundaries
# are the holdout's - that is the whole point - so the holdout rows are carried under a fold id
# the artifact never declares the geometry of.
HOLDOUT_FOLD_ID = 2

HOLDOUT_SPLIT = {
    "fold": HOLDOUT_FOLD_ID,
    "train_start": "2020-01-01T00:00:00",
    "train_end": "2023-11-01T00:00:00",
    "val_start": "2024-01-01T00:00:00",
    "val_end": "2024-12-01T00:00:00",
}


def _mds(temporal_dates: pl.Series) -> Any:
    """The five fields the check reads, and nothing else."""
    return SimpleNamespace(
        dataset=pl.DataFrame({"timestamp": TIMESTAMPS}),
        date_col="timestamp",
        temporal_by_fold=pl.DataFrame(
            {
                "fold": [HOLDOUT_FOLD_ID] * len(temporal_dates),
                "timestamp": temporal_dates,
            }
        ),
        temporal_keys=("symbol", "timestamp"),
        temporal_feature_names=("kalman_trend", "arima_forecast"),
        # Deliberately declares the validation folds only, and never the holdout's geometry.
        # A compatibility check reads this and refuses; a coverage check does not read it.
        temporal_artifact_splits=[
            {
                "fold": 0,
                "train_start": "2020-01-01T00:00:00",
                "train_end": "2021-12-01T00:00:00",
                "val_start": "2022-01-01T00:00:00",
                "val_end": "2022-12-01T00:00:00",
            },
            {
                "fold": 1,
                "train_start": "2020-01-01T00:00:00",
                "train_end": "2022-12-01T00:00:00",
                "val_start": "2023-01-01T00:00:00",
                "val_end": "2023-12-01T00:00:00",
            },
        ],
    )


def test_a_holdout_fold_the_artifact_never_declares_is_accepted_when_its_rows_cover_it() -> None:
    """The refusal that blocked two lanes: geometry the artifact cannot declare, rows it has."""
    _require_holdout_temporal_features(_mds(TIMESTAMPS), dict(HOLDOUT_SPLIT))


def test_an_artifact_that_stops_before_the_holdout_window_is_refused() -> None:
    """The failure direction: coverage is a real check, not a way past the previous one.

    Rows through the training window and nothing over the evaluation window is exactly the
    state that would fit the holdout model on real features and then score it on nulls.
    """
    short = TIMESTAMPS.filter(TIMESTAMPS.lt(datetime(2023, 12, 1)))
    with pytest.raises(ValueError, match="not covered by the fold-scoped temporal artifact"):
        _require_holdout_temporal_features(_mds(short), dict(HOLDOUT_SPLIT))


def test_the_refusal_does_not_tell_a_reader_to_regenerate_stage_04() -> None:
    """The old message said to generate the holdout fold in stage 04 before locking.

    That is the one action ml4t/agent-workspace#994 concludes is wrong: it moves a digest the
    selection was made under, and buys a declaration this check does not read. A refusal whose
    suggested remedy is a defect is worse than a bare refusal.
    """
    short = TIMESTAMPS.filter(TIMESTAMPS.lt(datetime(2023, 12, 1)))
    with pytest.raises(ValueError) as raised:
        _require_holdout_temporal_features(_mds(short), dict(HOLDOUT_SPLIT))
    assert "do NOT regenerate the artifact" in str(raised.value)


def test_a_case_study_without_fold_scoped_temporal_features_is_untouched() -> None:
    mds = _mds(TIMESTAMPS)
    mds.temporal_by_fold = None
    _require_holdout_temporal_features(mds, dict(HOLDOUT_SPLIT))
