"""A sub-daily label's overlap is counted in the steps of the series it is measured on.

``compute_ic_uncertainty(horizon=...)`` sets the HAC bandwidth and the bootstrap block in
observations of the IC series, and that series is one IC per distinct prediction timestamp.
Two things went wrong there in turn:

* The horizon was parsed off the label name, and ``m`` is the unit letter for both minutes
  and months. ``fwd_ret_1m`` is one month, so ``m`` resolved to 21 trading days;
  ``fwd_ret_15m`` is fifteen minutes and resolved to 315, putting a 314-lag standard error
  and a 315-observation bootstrap block on every nasdaq100 information coefficient.
* The first repair divided the declared buffer by ``decision.bar_frequency``. That is the
  rebalance cadence, not the step of the series: nasdaq100 rebalances every 15 minutes and
  registers predictions on the one-minute grid its features are built on. It would have
  turned bands that were far too wide into bands that were too narrow - a 60-minute label
  overlaps 60 observations of a one-minute series, not 4.

So the divisor is measured from the timestamps themselves. These tests hold it there, and
hold the eight daily, weekly and monthly case studies to the name parse they already had.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from case_studies.utils.registry.metrics import (
    _duration_seconds,
    _horizon_in_observations,
    _infer_horizon_from_label,
    _observation_step_seconds,
    compute_prediction_fold_metrics,
)

MINUTE = datetime.timedelta(minutes=1)
REPO_ROOT = Path(__file__).resolve().parents[1]


def declared_buffer(case_study: str, label: str) -> str:
    """Read the buffer out of the repo's own setup.yaml.

    Not through ``_declared_label_buffer``, which resolves the case-study directory via
    ``get_case_study_dir`` and therefore follows ``ML4T_OUTPUT_DIR``. Every job in
    ``test.yml`` sets that to a scratch directory holding no ``config/``, so going through
    the redirect would make these assertions pass or fail on where the output happens to
    be pointed rather than on what the case study declares.
    """
    setup = yaml.safe_load(
        (REPO_ROOT / "case_studies" / case_study / "config" / "setup.yaml").read_text()
    )
    labels = setup["labels"]
    if labels.get("primary") == label:
        return str(labels["buffer"])
    return str(labels["variant_buffers"][label])


def _grid(step: datetime.timedelta, n: int) -> pl.Series:
    start = datetime.datetime(2020, 1, 2, 14, 30)
    return pl.Series("timestamp", [start + step * i for i in range(n)])


class TestTheHorizonIsCountedInTheSeriesOwnStep:
    @pytest.mark.parametrize(
        ("buffer", "step_minutes", "expected"),
        [
            # nasdaq100 as it actually registers: predictions on the one-minute grid.
            ("5min", 1, 5),
            ("15min", 1, 15),
            ("60min", 1, 60),
            # The same labels were the predictions downsampled to the decision grid.
            ("5min", 15, 1),
            ("15min", 15, 1),
            ("60min", 15, 4),
            # An hourly buffer on a half-hourly series.
            ("1H", 30, 2),
        ],
    )
    def test_the_overlap_follows_the_timestamps(self, buffer, step_minutes, expected):
        dates = _grid(MINUTE * step_minutes, 200)
        assert _horizon_in_observations(buffer, dates) == expected

    def test_a_buffer_shorter_than_one_step_still_overlaps_one_observation(self):
        assert _horizon_in_observations("5min", _grid(MINUTE * 15, 100)) == 1

    def test_the_name_parse_read_fifteen_minutes_as_fifteen_months(self):
        """The defect this replaces, kept so the fallback is never wired back in."""
        assert _infer_horizon_from_label("fwd_ret_15m") == 315
        assert _horizon_in_observations("15min", _grid(MINUTE, 200)) == 15


class TestTheStepIsMeasuredNotAssumed:
    def test_the_modal_gap_wins_over_the_session_break(self):
        """An overnight gap is not a step of the grid, and must not enlarge it."""
        session = [datetime.datetime(2020, 1, 2, 14, 30) + MINUTE * i for i in range(60)]
        session += [datetime.datetime(2020, 1, 3, 14, 30) + MINUTE * i for i in range(60)]
        assert _observation_step_seconds(pl.Series("timestamp", session)) == 60.0

    def test_repeated_timestamps_are_one_observation(self):
        """A cross-section has one row per entity per decision time, not one row."""
        stamps = _grid(MINUTE * 15, 40).to_list()
        panel = pl.Series("timestamp", [stamp for stamp in stamps for _ in range(50)])
        assert _observation_step_seconds(panel) == 900.0

    @pytest.mark.parametrize(
        "dates",
        [None, pl.Series("timestamp", [1, 2, 3]), pl.Series("timestamp", ["a", "b", "c"])],
    )
    def test_something_that_is_not_a_timestamp_column_resolves_nothing(self, dates):
        assert _observation_step_seconds(dates) is None

    def test_a_series_too_short_to_have_a_step_resolves_nothing(self):
        assert _observation_step_seconds(_grid(MINUTE, 2)) is None


class TestEveryDailyAndSlowerLabelKeepsTheNameParse:
    @pytest.mark.parametrize(
        ("case_study", "label"),
        [
            ("etfs", "fwd_ret_21d"),
            ("cme_futures", "fwd_ret_5d"),
            ("fx_pairs", "fwd_ret_1d"),
            ("sp500_options", "ret_to_expiry"),
            ("us_equities_panel", "fwd_ret_21d"),
            ("us_firm_characteristics", "fwd_ret_1m"),
        ],
    )
    def test_a_declared_buffer_the_calendar_decides_resolves_nothing(self, case_study, label):
        """A day, a week and a month have no fixed length, so no division can be made."""
        buffer = declared_buffer(case_study, label)
        assert _horizon_in_observations(buffer, _grid(datetime.timedelta(days=1), 200)) is None

    @pytest.mark.parametrize(
        ("label", "expected"),
        [("fwd_ret_8h", 1), ("fwd_ret_24h", 3), ("fwd_dir_8h", 1)],
    )
    def test_crypto_is_sub_daily_too_and_keeps_the_overlap_it_already_had(self, label, expected):
        """The other sub-daily case study: 8-hour funding periods, so the grid is 8-hourly."""
        buffer = declared_buffer("crypto_perps_funding", label)
        grid = _grid(datetime.timedelta(hours=8), 200)
        assert (
            _horizon_in_observations(buffer, grid) == expected == _infer_horizon_from_label(label)
        )

    @pytest.mark.parametrize("text", ["21D", "1M", "1W", "", None, "expiry"])
    def test_a_calendar_duration_does_not_parse_to_seconds(self, text):
        assert _duration_seconds(text) is None

    @pytest.mark.parametrize(
        ("text", "expected"), [("15min", 900.0), ("60_minute", 3600.0), ("8H", 28800.0)]
    )
    def test_a_sub_daily_duration_does(self, text, expected):
        assert _duration_seconds(text) == expected


def _panel(step: datetime.timedelta, n_steps: int, n_entities: int = 20) -> pl.DataFrame:
    """A synthetic cross-section: every entity scored at every decision time."""
    import numpy as np

    stamps = _grid(step, n_steps).to_list()
    generator = np.random.default_rng(0)
    rows = n_steps * n_entities
    return pl.DataFrame(
        {
            "timestamp": [stamp for stamp in stamps for _ in range(n_entities)],
            "symbol": [f"S{i:02d}" for _ in stamps for i in range(n_entities)],
            "fold_id": [0] * rows,
            "y_true": generator.normal(size=rows),
            "y_score": generator.normal(size=rows),
        }
    )


class TestTheHorizonReachesTheUncertaintyCall:
    """The wiring, measured on what comes back rather than on the shape of the source."""

    def test_a_declared_buffer_sets_the_hac_lag_and_the_bootstrap_block(self):
        panel = _panel(MINUTE, 400)
        declared, _ = compute_prediction_fold_metrics(
            panel, label="fwd_ret_15m", label_buffer="15min"
        )
        assert declared["ic_hac_lag"] == 14
        assert declared["ic_boot_block"] >= 15

    def test_without_the_declaration_the_name_parse_is_what_lands(self):
        panel = _panel(MINUTE, 400)
        parsed, _ = compute_prediction_fold_metrics(panel, label="fwd_ret_15m")
        declared, _ = compute_prediction_fold_metrics(
            panel, label="fwd_ret_15m", label_buffer="15min"
        )
        assert parsed["ic_hac_lag"] > declared["ic_hac_lag"], (
            "the name parse resolves fifteen minutes as fifteen months, so it must "
            "produce the larger bandwidth the declaration corrects"
        )

    def test_the_same_label_on_a_coarser_grid_overlaps_fewer_observations(self):
        fine, _ = compute_prediction_fold_metrics(
            _panel(MINUTE, 400), label="fwd_ret_60m", label_buffer="60min"
        )
        coarse, _ = compute_prediction_fold_metrics(
            _panel(MINUTE * 15, 400), label="fwd_ret_60m", label_buffer="60min"
        )
        assert fine["ic_hac_lag"] == 59
        assert coarse["ic_hac_lag"] < fine["ic_hac_lag"]
