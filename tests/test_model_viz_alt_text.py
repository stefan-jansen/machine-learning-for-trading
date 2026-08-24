"""Every model-analysis figure publishes alt text, and that text is computed.

`case_studies/utils/model_viz.py` renders the model-analysis figures for seven case
studies. It called ``fig.show()`` for every one of them, which emits an ``<img>`` with
no alternative text, so all seven notebooks published figures a screen reader cannot
describe (ml4t/agent-workspace#899).

These tests assert two separate things, and the second is the one that matters:

1. Every helper publishes alt text at all.
2. The alt text is **derived from the data just plotted**, not a fixed sentence. Each
   test changes the input and requires the alt text to change with it. A canned string
   would pass (1) and fail (2), which is the failure mode worth protecting against -
   a wrong figure description is a worse defect than a missing one.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from case_studies.utils import model_viz


@pytest.fixture
def captured(monkeypatch) -> list[str]:
    """Record the alt text of every figure a helper publishes, and close the figure."""
    import matplotlib.pyplot as plt

    alts: list[str] = []

    def _record(fig: object, alt: str) -> None:
        alts.append(alt)
        plt.close(fig)  # type: ignore[arg-type]

    monkeypatch.setattr(model_viz, "show_with_alt", _record)
    return alts


def _fold_ic(scale: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_label": ["linear/a", "linear/a", "gbm/b", "gbm/b"],
            "fold_id": [0, 1, 0, 1],
            "ic_mean": [0.01 * scale, 0.03 * scale, 0.02 * scale, 0.04 * scale],
        }
    )


def test_fold_heatmap_publishes_alt_carrying_the_plotted_range(captured):
    model_viz.plot_fold_heatmap(_fold_ic())

    assert len(captured) == 1
    alt = captured[0]
    assert "Heatmap" in alt
    assert "2 models" in alt and "2 validation folds" in alt
    # The extremes of the matrix actually drawn.
    assert "+0.010" in alt and "+0.040" in alt


def test_fold_heatmap_alt_tracks_the_data(captured):
    model_viz.plot_fold_heatmap(_fold_ic(scale=1.0))
    model_viz.plot_fold_heatmap(_fold_ic(scale=10.0))

    assert len(captured) == 2
    assert captured[0] != captured[1], "alt text is fixed prose, not computed from the data"
    assert "+0.400" in captured[1]


def test_fold_boxplot_publishes_alt(captured):
    model_viz.plot_fold_boxplot(_fold_ic())

    assert len(captured) == 1
    assert "Box plot" in captured[0]
    assert "2 model families" in captured[0]
    assert "2 folds" in captured[0]


def test_correlation_matrix_alt_reports_the_off_diagonal(captured):
    corr = np.array([[1.0, 0.42, 0.10], [0.42, 1.0, 0.80], [0.10, 0.80, 1.0]])
    model_viz.plot_correlation_matrix(corr, ["linear/a", "gbm/b", "dl/c"])

    assert len(captured) == 1
    alt = captured[0]
    assert "3 models" in alt
    assert "3 distinct off-diagonal" in alt
    assert "0.10 to 0.80" in alt
    assert "0.44" in alt  # mean of 0.42, 0.10, 0.80


def test_bucket_monotonicity_alt_names_the_unconditional_mean(captured):
    buckets = {
        "gbm/b": pl.DataFrame({"bucket": [1, 2, 3], "mean_return": [-0.002, 0.0, 0.003]}),
    }
    model_viz.plot_bucket_monotonicity(buckets, n_buckets=3, unconditional_mean=0.0005)

    assert len(captured) == 1
    alt = captured[0]
    assert "3 buckets" in alt
    assert "-0.0020 to +0.0030" in alt
    assert "+0.0005" in alt


def test_bucket_monotonicity_omits_the_mean_clause_when_absent(captured):
    buckets = {"gbm/b": pl.DataFrame({"bucket": [1, 2], "mean_return": [-0.001, 0.001]})}
    model_viz.plot_bucket_monotonicity(buckets, n_buckets=2)

    assert "unconditional mean" not in captured[0]


def test_hac_leaderboard_alt_counts_intervals_excluding_zero(captured):
    metrics = pl.DataFrame(
        {
            "family": ["gbm", "linear", "dl"],
            "config_name": ["a", "b", "c"],
            "ic_mean_daily": [0.05, 0.02, -0.01],
            "ic_ci_lo": [0.03, -0.01, -0.04],
            "ic_ci_hi": [0.07, 0.05, 0.02],
        }
    )
    model_viz.plot_hac_ci_leaderboard(metrics)

    alt = captured[0]
    assert "3 model configurations" in alt
    # Only the first interval (0.03, 0.07) excludes zero.
    assert "1 of the 3 intervals drawn exclude zero" in alt
    assert "-0.0100 to +0.0500" in alt


def test_label_horizon_forest_alt_counts_missing_pairs(captured):
    metrics = pl.DataFrame(
        {
            "family": ["gbm", "linear"],
            "label": ["fwd_ret_1d", "fwd_ret_1d"],
            "ic_mean_daily": [0.04, 0.01],
            "ic_ci_lo": [0.02, -0.02],
            "ic_ci_hi": [0.06, 0.04],
        }
    )
    model_viz.plot_label_horizon_forest(
        metrics, families=["gbm", "linear", "dl"], labels=["fwd_ret_1d"]
    )

    alt = captured[0]
    # Three families x one label = three tiles, but only two have a run.
    assert "2 of the 3 family-label pairs have a run" in alt
    assert "1 of the drawn intervals exclude zero" in alt


def test_rolling_daily_ic_alt_reports_the_window_and_span(captured):
    n = 40
    daily = pl.DataFrame(
        {
            "fold_id": [0] * n,
            "date": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 2, 9), interval="1d", eager=True
            ),
            "ic": np.linspace(-0.05, 0.05, n),
            "n_obs": [100] * n,
        }
    )
    model_viz.plot_rolling_daily_ic(daily, window=10)

    alt = captured[0]
    assert "40 days" in alt
    assert "-0.050 to +0.050" in alt
    assert "10-day rolling mean" in alt


def test_regime_bars_publishes_alt_naming_both_regimes(captured):
    regime = pl.DataFrame(
        {
            "regime": ["low_vol", "high_vol", "low_vol", "high_vol"],
            "family": ["gbm", "gbm", "linear", "linear"],
            "ic_mean": [0.03, -0.01, 0.02, 0.00],
            "ic_std": [0.01, 0.01, 0.01, 0.01],
            "n_periods": [100, 100, 100, 100],
        }
    )
    model_viz.plot_regime_bars(regime)

    alt = captured[0]
    assert "2 model families" in alt
    assert "high vol" in alt and "low vol" in alt
    assert "-0.010 to +0.030" in alt


def test_learning_curves_alt_counts_panels_and_configs(captured):
    cp = pl.DataFrame(
        {
            "family": ["gbm", "gbm", "gbm", "gbm"],
            "config_name": ["a", "a", "b", "b"],
            "checkpoint_value": [10, 20, 10, 20],
            "ic_mean_daily": [0.01, 0.02, 0.00, 0.03],
        }
    )
    model_viz.plot_learning_curves(cp, ["gbm"])

    alt = captured[0]
    assert "1 stacked line chart" in alt or "1 stacked line charts" in alt
    assert "2 configurations" in alt
    assert "+0.000 to +0.030" in alt


def test_cv_timeline_alt_reports_the_span_and_holdout(captured):
    import datetime as dt

    folds = pl.DataFrame(
        {
            "fold_id": [0, 1],
            "val_start": [dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 1)],
            "val_end": [dt.datetime(2024, 2, 29), dt.datetime(2024, 4, 30)],
        }
    )
    model_viz.plot_cv_timeline(folds, n_splits=2, holdout_start="2024-05-01")

    alt = captured[0]
    assert "2 walk-forward validation windows" in alt
    assert "2024-01-01" in alt and "2024-04-30" in alt
    assert "2024-05-01" in alt


def test_cv_timeline_omits_the_holdout_clause_when_absent(captured):
    import datetime as dt

    folds = pl.DataFrame(
        {
            "fold_id": [0],
            "val_start": [dt.datetime(2024, 1, 1)],
            "val_end": [dt.datetime(2024, 2, 29)],
        }
    )
    model_viz.plot_cv_timeline(folds, n_splits=1)

    assert "holdout" not in captured[0]


def test_feature_importance_alt_counts_rows_shown_not_rows_available(captured):
    rows = []
    for feature_i in range(20):
        for fold in range(3):
            rows.append(
                {
                    "feature": f"f{feature_i:02d}",
                    "fold_id": fold,
                    "importance_norm": feature_i / 20.0,
                }
            )
    importance = pl.DataFrame(rows)
    model_viz.plot_feature_importance_heatmap(importance, top_n=5)

    alt = captured[0]
    # 20 features exist; the figure draws the top 5, and the alt must say 5.
    assert "5 features" in alt
    assert "3 folds" in alt
    assert "0.75 to 0.95" in alt


# ---------------------------------------------------------------------------
# Regressions from the first version of the alt-text change. Each of these three
# was a defect the computed description introduced: a count that did not match
# what was drawn, or a reduction over an array the drawing code never reaches.
# ---------------------------------------------------------------------------


def test_rolling_daily_ic_returns_when_every_day_is_undefined(captured):
    """`defined_ic` can empty a frame that passed the height guard above it."""
    n = 12
    daily = pl.DataFrame(
        {
            "fold_id": [0] * n,
            "date": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 1, 12), interval="1d", eager=True
            ),
            "ic": [None] * n,
            "n_obs": [100] * n,
        },
        schema_overrides={"ic": pl.Float64},
    )

    # Reducing over the empty date array raises; the pre-alt-text version rendered
    # an empty figure, so raising here would be a regression in the notebook cell.
    model_viz.plot_rolling_daily_ic(daily, window=5)

    assert captured == []


def test_hac_leaderboard_does_not_count_an_interval_it_did_not_draw(captured):
    """A row with a valid CI and no point estimate is skipped by the drawing loop."""
    metrics = pl.DataFrame(
        {
            "family": ["gbm", "linear"],
            "config_name": ["a", "b"],
            "ic_mean_daily": [0.05, None],
            "ic_ci_lo": [0.03, 0.01],
            "ic_ci_hi": [0.07, 0.09],
        },
        schema_overrides={"ic_mean_daily": pl.Float64},
    )
    model_viz.plot_hac_ci_leaderboard(metrics)

    # Both rows have finite intervals, but only the first is drawn.
    assert "1 of the 1 intervals drawn exclude zero" in captured[0]


def test_learning_curves_counts_only_configs_in_the_plotted_families(captured):
    """`cp_data` may carry families the caller did not ask to plot."""
    cp = pl.DataFrame(
        {
            "family": ["gbm", "gbm", "linear", "linear"],
            "config_name": ["a", "a", "b", "b"],
            "checkpoint_value": [10, 20, 10, 20],
            "ic_mean_daily": [0.01, 0.02, 0.30, 0.40],
        }
    )
    model_viz.plot_learning_curves(cp, ["gbm"])

    alt = captured[0]
    # One panel, one config drawn - `linear` is in the frame and not on the chart.
    assert "1 configurations" in alt
    assert "+0.010 to +0.020" in alt
    assert "0.400" not in alt


def test_no_helper_still_calls_fig_show():
    """The defect was `fig.show()`; a new helper must not reintroduce it."""
    from pathlib import Path

    source = Path(model_viz.__file__).read_text()
    assert "fig.show()" not in source


def test_span_describes_only_finite_values():
    assert model_viz._span([np.nan, 0.5, np.inf, -0.25]) == "-0.250 to +0.500"
    assert model_viz._span([np.nan, np.nan]) == "no finite values"
    assert model_viz._span([0.25]) == "+0.250"
