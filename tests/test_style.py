"""Tests for the shared figure style module.

These pin the one invariant the Plotly template cannot enforce on its own: a
subplot figure's panel labels must read as subordinate to its figure title.
Plotly writes ``subplot_titles`` as annotations with an explicit 16pt, and an
explicit value beats anything the template sets in ``annotationdefaults``, so a
template edit that lowers ``title.font.size`` silently inverts the hierarchy on
every subplot figure in the repo.
"""

from __future__ import annotations

import pytest

plotly = pytest.importorskip("plotly")

import plotly.io as pio  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from utils.style import (  # noqa: E402
    _PLOTLY_SUBPLOT_TITLE_SIZE,
    SUBPLOT_TITLE_SIZE,
    style_subplot_titles,
)


def _template_title_size() -> int:
    return pio.templates["ml4t"].layout.title.font.size


def test_panel_labels_stay_below_the_figure_title():
    """The hierarchy this helper exists to enforce."""
    assert _template_title_size() > SUBPLOT_TITLE_SIZE


def test_style_subplot_titles_restyles_panel_labels():
    fig = make_subplots(rows=2, cols=2, subplot_titles=["A", "B", "C", "D"])

    # Plotly's hardcoded default, which the template cannot override. It is also
    # part of the selector's signature, so this failing means the helper has
    # gone blind rather than merely that a number moved.
    assert {a.font.size for a in fig.layout.annotations} == {_PLOTLY_SUBPLOT_TITLE_SIZE}

    style_subplot_titles(fig)

    assert [a.font.size for a in fig.layout.annotations] == [SUBPLOT_TITLE_SIZE] * 4
    for annotation in fig.layout.annotations:
        assert annotation.font.size < _template_title_size()


def test_style_subplot_titles_leaves_other_annotations_alone():
    """A bare update_annotations() would restyle these too - it must not."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["A", "B"])
    fig.add_annotation(
        text="callout",
        x=0.5,
        y=0.5,
        xref="x",
        yref="y",
        showarrow=True,
        font={"size": 22, "color": "#ff0000"},
    )
    # A paper-referenced, arrowless note - the case a naive xref check would miss.
    fig.add_annotation(
        text="source",
        x=0,
        y=-0.1,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 9},
    )

    style_subplot_titles(fig)

    by_text = {a.text: a for a in fig.layout.annotations}
    assert by_text["A"].font.size == SUBPLOT_TITLE_SIZE
    assert by_text["B"].font.size == SUBPLOT_TITLE_SIZE
    assert by_text["callout"].font.size == 22
    assert by_text["callout"].font.color == "#ff0000"
    assert by_text["source"].font.size == 9


def test_style_subplot_titles_spares_a_note_sharing_the_paper_signature():
    """The near-miss: paper-referenced, arrowless, bottom-anchored, centered.

    Everything a subplot title has except Plotly's 16pt. A selector built on the
    reference frame and anchors alone would restyle this.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=["A", "B"])
    fig.add_annotation(
        text="source",
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font={"size": 9, "color": "#333333"},
    )

    style_subplot_titles(fig)

    by_text = {a.text: a for a in fig.layout.annotations}
    assert by_text["source"].font.size == 9
    assert by_text["source"].font.color == "#333333"
    assert by_text["A"].font.size == SUBPLOT_TITLE_SIZE


def test_style_subplot_titles_applies_once():
    """A restyled label loses the 16pt signature, so a second call does nothing."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["A", "B"])
    style_subplot_titles(fig)
    style_subplot_titles(fig, size=30)
    assert {a.font.size for a in fig.layout.annotations} == {SUBPLOT_TITLE_SIZE}


def test_style_subplot_titles_is_chainable():
    fig = make_subplots(rows=1, cols=2, subplot_titles=["A", "B"])
    assert style_subplot_titles(fig) is fig


def test_style_subplot_titles_accepts_an_explicit_size():
    fig = make_subplots(rows=1, cols=2, subplot_titles=["A", "B"])
    style_subplot_titles(fig, size=11)
    assert {a.font.size for a in fig.layout.annotations} == {11}
