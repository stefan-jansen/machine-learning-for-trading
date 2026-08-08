"""Tests for the shared stage-03 figure helpers in case_studies/utils/feature_engineering.py.

Pins:
- plot_coverage_through_time: the dense stretch stays visible below the top spine when the
  frame starts at zero, which is what the stage-03 notebooks pass, while the zoom the
  helper exists for still happens on an already-dense frame.
- plot_persistence: the left panel is the wider of the two, and the right panel's x-axis
  label is drawn inside the figure rather than cut off at the edge.

All eight 03_financial_features notebooks draw both, so a regression here is a regression
in eight rendered pages at once.
"""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

import case_studies.utils.feature_engineering as fe


@pytest.fixture
def captured(monkeypatch) -> list[Figure]:
    """Intercept the figure the helper would display, so it can be measured."""
    figures: list[Figure] = []
    monkeypatch.setattr(fe, "show_with_alt", lambda fig, alt: figures.append(fig))
    return figures


def _dates(n: int) -> list[date]:
    return [date(2018, 1, 1) + timedelta(days=i) for i in range(n)]


def _coverage(warmups: dict[str, int], n: int = 400) -> pl.DataFrame:
    frame = {"timestamp": _dates(n)}
    for family, w in warmups.items():
        frame[family] = [0.0] * w + [1.0] * (n - w)
    return pl.DataFrame(frame)


def test_coverage_leaves_the_dense_stretch_visible_below_the_spine(captured) -> None:
    # What the stage-03 notebooks pass: the panel before the null policy, so every family
    # starts at zero. A fixed 1.0005 ceiling drew the flat stretch under the top spine.
    fe.plot_coverage_through_time(
        _coverage({"momentum": 21, "volatility": 63, "flow": 252}),
        title="t",
        alt="a",
    )
    (fig,) = captured
    bottom, top = fig.axes[0].get_ylim()
    headroom = (top - 1.0) / (top - bottom)
    assert headroom > 0.02, f"the stretch at 1.0 sits {headroom:.2%} below the top of the axis"


def test_coverage_still_zooms_when_the_frame_is_already_dense(captured) -> None:
    # The case the helper's scaling exists for: a matrix 99% dense everywhere must not
    # draw as one flat line at the top of a full 0-1 axis.
    n = 400
    frame = pl.DataFrame({"timestamp": _dates(n), "momentum": [0.99] * n, "flow": [0.995] * n})
    fe.plot_coverage_through_time(frame, title="t", alt="a")
    (fig,) = captured
    bottom, top = fig.axes[0].get_ylim()
    assert bottom > 0.98, "an already-dense frame must still be zoomed, not pinned near zero"
    assert top - 1.0 < 0.001, "and must not gain the headroom a full-range frame needs"


def _panel(
    n_entities: int = 8, n_days: int = 120, columns: list[str] | None = None
) -> tuple[pl.DataFrame, list[str], list[date]]:
    rng = np.random.default_rng(0)
    columns = list(columns) if columns else ["mom_21d", "vol_63d"]
    days = _dates(n_days)
    level = {f"E{i}": rng.normal(0, 1, len(columns)) for i in range(n_entities)}
    rows = []
    for day in days:
        for entity, value in level.items():
            level[entity] = 0.97 * value + rng.normal(0, 0.25, len(columns))
            rows.append({"symbol": entity, "timestamp": day, **dict(zip(columns, level[entity]))})
    return pl.DataFrame(rows), columns, [d for d in days if d.day in (1, 15)]


def test_persistence_gives_the_lag_panel_the_greater_width(captured) -> None:
    panel, columns, schedule = _panel()
    fe.plot_persistence(
        panel, columns, entity="symbol", max_lag=10, decision_dates=schedule, title="t", alt="a"
    )
    (fig,) = captured
    left, right = fig.axes[0].get_position().width, fig.axes[1].get_position().width
    assert left > right, "the panel the prose reads off must not be the compressed one"


def test_persistence_draws_its_right_hand_axis_label_inside_the_figure(captured) -> None:
    # Feature names of the length the case studies actually use. They are the right panel's
    # y-tick labels, so they set how little width the x-axis label has left - which is why
    # this was seen in cme_futures and not in a two-short-column sketch.
    panel, columns, schedule = _panel(
        columns=[
            "realised_vol_63d",
            "momentum_since_21d",
            "turnover_zscore_252d",
            "spread_bps_median_21d",
        ]
    )
    fe.plot_persistence(
        panel, columns, entity="symbol", max_lag=10, decision_dates=schedule, title="t", alt="a"
    )
    (fig,) = captured
    fig.canvas.draw()
    label = fig.axes[1].xaxis.get_label()
    extent = label.get_window_extent(fig.canvas.get_renderer())
    assert extent.x1 <= fig.bbox.x1, "the label runs off the right edge and is cut mid-word"
    assert extent.x0 >= fig.bbox.x0
