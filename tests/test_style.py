"""Tests for the shared figure style module.

These pin two invariants nothing else can enforce.

A subplot figure's panel labels must read as subordinate to its figure title.
Plotly writes ``subplot_titles`` as annotations with an explicit 16pt, and an
explicit value beats anything the template sets in ``annotationdefaults``, so a
template edit that lowers ``title.font.size`` silently inverts the hierarchy on
every subplot figure in the repo.

And ``COLOR_CYCLER``'s six entries must be six colours a reader can tell apart,
on the page and once the page is reduced to gray. Its sixth was ``slate``, a
second navy, so every six-series chart in the book drew its first and last
series as one line and no check anywhere noticed.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from cycler import cycler

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from utils.style import (  # noqa: E402
    _PLOTLY_SUBPLOT_TITLE_SIZE,
    COLOR_CYCLER,
    COLORS,
    GRAY_CYCLER,
    LINESTYLE_CYCLER,
    MARKER_CYCLER,
    SUBPLOT_TITLE_SIZE,
    add_message_title,
    apply_book_style,
    ml4t_palette,
    style_subplot_titles,
)

# Only the Plotly template tests need Plotly. Guarding the whole module on it, as this
# file used to, would let the palette tests below disappear without a word on any
# machine that happened not to have it.
requires_plotly = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly is not installed"
)


def _template_title_size() -> int:
    import plotly.io as pio

    return pio.templates["ml4t"].layout.title.font.size


def make_subplots(*args, **kwargs):
    from plotly.subplots import make_subplots as _make_subplots

    return _make_subplots(*args, **kwargs)


# =============================================================================
# THE PALETTE: six series, six colours
# =============================================================================


def _lab(hexcolor: str) -> np.ndarray:
    """CIE L*a*b* for *hexcolor*, so distance means what an eye would call distance.

    Distance in RGB does not: ``blue`` and ``slate`` sit further apart in RGB than
    ``amber`` and ``copper``, and it is the first pair a reader cannot separate.
    """
    rgb = np.array([int(hexcolor.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = matrix @ linear / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _distance(a: str, b: str) -> float:
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def _gray(hexcolor: str) -> float:
    """0-255, what a naive grayscale reduction of a printed page gives."""
    rgb = np.array([int(hexcolor.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 255 * float(linear @ np.array([0.2126, 0.7152, 0.0722])) ** (1 / 2.2)


# 20 sits above the 12.9 that `blue`/`slate` scored, which is the pair a reader could
# not separate, and below the 25.7 of `amber`/`copper`, which the palette has always
# shipped and which reads as two colours. It is a floor on the defect, not a re-pick
# of the palette.
SEPARABLE = 20.0


def test_six_series_get_six_colours_a_reader_can_separate():
    assert len(COLOR_CYCLER) == len(set(COLOR_CYCLER)) == 6
    worst = min(
        (_distance(a, b), a, b) for i, a in enumerate(COLOR_CYCLER) for b in COLOR_CYCLER[i + 1 :]
    )
    assert worst[0] >= SEPARABLE, f"{worst[1]} and {worst[2]} are {worst[0]:.1f} apart"


def test_the_sixth_series_is_not_a_second_navy():
    """The defect itself: entry 5 against entry 0, in colour and in gray.

    Both readings are needed. ``slate`` failed on colour at 12.9 while clearing any
    gray bar one might set at 20, and a colour picked to clear the gray gap alone
    could be another navy at a different weight.
    """
    navy, sixth = COLOR_CYCLER[0], COLOR_CYCLER[-1]
    assert _distance(navy, sixth) >= SEPARABLE
    # 26 is the tightest gap GRAY_CYCLER itself ships, so it is this file's own
    # standing answer to "how far apart is far enough in gray".
    assert abs(_gray(navy) - _gray(sixth)) >= 26
    assert sixth != COLORS["slate"]


def test_a_six_series_chart_draws_six_different_colours():
    """The acceptance test, through the path a figure actually takes."""
    apply_book_style("color")
    fig, ax = plt.subplots()
    for i in range(6):
        ax.plot([0, 1], [i, i + 1], label=f"series {i}")
    drawn = [line.get_color() for line in ax.get_lines()]
    plt.close(fig)
    assert len(set(drawn)) == 6
    worst = min(_distance(a, b) for i, a in enumerate(drawn) for b in drawn[i + 1 :])
    assert worst >= SEPARABLE


def _matplotlibrc_cycle() -> list[str]:
    """The colours repo-root `matplotlibrc` hands a bare `plt.subplots()`."""
    text = (Path(__file__).resolve().parents[1] / "matplotlibrc").read_text()
    line = next(
        ln
        for ln in text.splitlines()
        if ln.startswith("axes.prop_cycle") and not ln.startswith("#")
    )
    return ["#" + h.lower() for h in re.findall(r"'([0-9a-fA-F]{6})'", line)]


def test_every_cycle_in_the_repo_is_the_same_cycle():
    """Four palettes ordered by hand in four files, and three of them held a stale one.

    `COLOR_CYCLER` is what `apply_book_style("color")` sets and what the book-figure
    scripts read, and it is the only one the first version of this fix corrected. It is
    also the one almost nothing draws from: `matplotlibrc` is what a bare
    `plt.subplots()` uses, the Plotly template colorway is what a bare `go.Figure()`
    uses, and `ml4t_palette(categorical=True)` is what the figure skill tells a notebook
    to call. All three still carried `slate` - at the fourth, third and third positions
    - so the two-navy collision arrived at four series, three traces and three
    categories rather than at six, in the paths that draw nearly every figure here.
    """
    assert _matplotlibrc_cycle() == [c.lower() for c in COLOR_CYCLER], (
        "matplotlibrc and COLOR_CYCLER disagree, so a bare plt.subplots() draws "
        "something other than the palette"
    )
    assert ml4t_palette(5, categorical=True) == COLOR_CYCLER[:5]


@requires_plotly
def test_the_plotly_colorway_is_the_same_cycle():
    import plotly.io as pio

    assert list(pio.templates["ml4t"].layout.colorway) == COLOR_CYCLER


def test_a_bare_matplotlib_figure_draws_six_different_colours():
    """What a notebook actually gets: no style call, just `matplotlibrc`."""
    with plt.rc_context({"axes.prop_cycle": cycler(color=_matplotlibrc_cycle())}):
        fig, ax = plt.subplots()
        for i in range(6):
            ax.plot([0, 1], [i, i + 1])
        drawn = [line.get_color() for line in ax.get_lines()]
        plt.close(fig)
    assert len(set(drawn)) == 6
    worst = min(_distance(a, b) for i, a in enumerate(drawn) for b in drawn[i + 1 :])
    assert worst >= SEPARABLE


def test_the_degradation_cyclers_still_cover_every_hue():
    """A figure that loses its colour falls back on these, so they cannot run short."""
    for fallback in (GRAY_CYCLER, LINESTYLE_CYCLER, MARKER_CYCLER):
        assert len(fallback) >= len(COLOR_CYCLER)
    assert len(set(GRAY_CYCLER)) == len(GRAY_CYCLER)
    assert len(set(MARKER_CYCLER)) == len(MARKER_CYCLER)


def test_a_long_subtitle_does_not_widen_the_figure():
    """#248's silent half: `savefig.bbox="tight"` pays for whatever the subtitle asks.

    Unwrapped, a subtitle wider than the axes did not clip and did not warn - it
    widened the saved canvas, and the plot came out in the left quarter of an image
    whose remaining three quarters held one line of 9pt gray.
    """
    fig, ax = plt.subplots(figsize=(5.833, 3.0))
    ax.plot([0, 1], [0, 1])
    add_message_title(
        ax,
        "Every family reads a window that closes before the decision",
        subtitle=(
            "Lookback and information lag per declared family, in trading sessions, "
            "over the full sample, gross of costs, on the primary label's rebalance "
            "schedule and its configured five-session purge gap"
        ),
    )
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    subtitle = next(
        child
        for child in ax.get_children()
        if getattr(child, "get_text", None) and "Lookback" in str(child.get_text())
    )
    assert "\n" in subtitle.get_text(), "a subtitle this long has to wrap"
    assert subtitle.get_window_extent(renderer).x1 <= fig.bbox.x1 + 1, (
        "the subtitle runs past the figure, so bbox='tight' widens the canvas to fit it"
    )
    plt.close(fig)


# =============================================================================
# THE PLOTLY TEMPLATE
# =============================================================================


@requires_plotly
def test_panel_labels_stay_below_the_figure_title():
    """The hierarchy this helper exists to enforce."""
    assert _template_title_size() > SUBPLOT_TITLE_SIZE


@requires_plotly
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


@requires_plotly
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


@requires_plotly
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


@requires_plotly
def test_style_subplot_titles_applies_once():
    """A restyled label loses the 16pt signature, so a second call does nothing."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["A", "B"])
    style_subplot_titles(fig)
    style_subplot_titles(fig, size=30)
    assert {a.font.size for a in fig.layout.annotations} == {SUBPLOT_TITLE_SIZE}


@requires_plotly
def test_style_subplot_titles_is_chainable():
    fig = make_subplots(rows=1, cols=2, subplot_titles=["A", "B"])
    assert style_subplot_titles(fig) is fig


@requires_plotly
def test_style_subplot_titles_accepts_an_explicit_size():
    fig = make_subplots(rows=1, cols=2, subplot_titles=["A", "B"])
    style_subplot_titles(fig, size=11)
    assert {a.font.size for a in fig.layout.annotations} == {11}
