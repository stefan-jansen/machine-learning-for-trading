"""ML4T Visualization Style.

Canonical color palette, matplotlib rcParams, Plotly template, and chart
helpers for all book visualizations.

## Automatic Styling (Matplotlib)

ML4T style is applied automatically when running from repo root.
The ``matplotlibrc`` file in the repo root is loaded by matplotlib
before any other config. No imports or function calls needed.

## Explicit Color References

    from utils.style import COLORS
    ax.plot(x, y, color=COLORS['blue'])
    ax.axhline(0, color=COLORS['amber'], linestyle='--')

## Plotly

    import plotly.io as pio
    pio.templates.default = "ml4t"  # Auto-registered on import
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# =============================================================================
# ML4T COLOR PALETTE
# =============================================================================
# Aligned with ml4t.io website identity

COLORS = {
    # Primary blues (core identity)
    "blue": "#0a1628",  # Deep blue - primary emphasis, main data
    "blue_light": "#152238",  # Lighter blue - secondary elements
    "slate": "#1a2d4a",  # Mid-blue - tertiary, gridlines
    # Silver tones (backgrounds, text)
    "silver": "#F8F8F6",  # Light silver - text on dark, highlights
    "silver_muted": "#e8e8e6",  # Muted silver - borders, subtle elements
    # Warm accents (highlights, emphasis)
    "amber": "#D4A84B",  # Warm amber - CTAs, important highlights
    "amber_light": "#E4B85B",  # Lighter amber - hover states
    "copper": "#C87533",  # Copper - secondary accent
    # Semantic (for data meaning)
    "positive": "#10b981",  # Success green - profits, gains
    "negative": "#ef4444",  # Error red - losses (use sparingly!)
    "neutral": "#334155",  # Slate gray - neutral elements
    "recede": "#94a3b8",  # Light slate - structure that must not compete with the data
    # Backgrounds
    "bg_light": "#FAFAF9",  # Warm off-white (light mode)
    "bg_dark": "#0a1628",  # Deep blue (dark mode)
}

# Grayscale equivalents for print
GRAYSCALE = {
    "blue": 0.10,  # ~10% gray (very dark)
    "slate": 0.25,  # ~25% gray
    "amber": 0.65,  # ~65% gray
    "silver": 0.97,  # ~97% gray (nearly white)
}

# Categorical cyclers for `axes.prop_cycle`. The print track pairs GRAY_CYCLER
# with LINESTYLE_CYCLER so a B&W readout stays legible; the color track relies
# on hue alone (no linestyle pairing - see apply_book_style). Color order
# prioritizes perceptual separation for the first 4 entries (most figures use
# <=4 series). GRAY_CYCLER mirrors the GRAY_FILLS weight order (secondary widened
# to #808080 for print contrast) while keeping every entry dark enough to read
# as a line on white.
#
# The sixth entry was `slate`, and slate is a second navy: CIEDE2000 8.1 from
# `blue`, and 48 against 28 once a page is reduced to gray. So every six-series
# chart drew its first and last series as one line, which is what a reader saw in
# the stage-03 coverage figure and in the six products of ch16's futures backtest.
# `recede` is the only entry in the locked palette that parts from navy on both
# readings at once - CIEDE2000 50.1, gray 160 against 28 - and it stays clear of
# the other five in color (nearest is 30.6, to `positive`). It is light, so it
# reads as the least emphatic series; that is the trade a sixth series buys, and
# there is no seventh. Beyond six, pair the hue with a line style (`_cycle` in
# case_studies/utils/feature_engineering.py) rather than extending this list.
COLOR_CYCLER = [
    COLORS["blue"],  # navy   - primary
    COLORS["amber"],  # gold   - secondary
    COLORS["copper"],  # orange - tertiary
    COLORS["positive"],  # green  - fourth
    COLORS["negative"],  # red    - fifth (semantic, use sparingly)
    COLORS["recede"],  # light slate - sixth (only when >=6 series)
]
GRAY_CYCLER = ["#000000", "#808080", "#404040", "#a8a8a8", "#666666", "#c8c8c8"]
LINESTYLE_CYCLER = ["-", "--", ":", "-.", "-", "--"]
MARKER_CYCLER = ["o", "s", "^", "D", "v", "P"]

# =============================================================================
# MATPLOTLIB STYLE CONFIGURATIONS
# =============================================================================

_BASE_STYLE = {
    # Figure
    "figure.dpi": 100,
    "figure.figsize": (10, 6),
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # Axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "axes.labelpad": 8,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    # Ticks
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.pad": 4,
    "ytick.major.pad": 4,
    # Lines
    "lines.linewidth": 2,
    "lines.markersize": 6,
    # Legend
    "legend.frameon": False,
    "legend.fontsize": 10,
    # Font (prefer DM Sans, fallback to system sans)
    "font.family": ["sans-serif"],
    "font.sans-serif": ["DM Sans", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
}

ML4T_LIGHT_STYLE = {
    **_BASE_STYLE,
    "figure.facecolor": COLORS["bg_light"],
    "axes.facecolor": "white",
    "axes.edgecolor": COLORS["silver_muted"],
    "axes.labelcolor": COLORS["neutral"],
    "axes.titlecolor": COLORS["blue"],
    "xtick.color": COLORS["neutral"],
    "ytick.color": COLORS["neutral"],
    "grid.color": COLORS["silver_muted"],
    "text.color": COLORS["neutral"],
}

ML4T_DARK_STYLE = {
    **_BASE_STYLE,
    "figure.facecolor": COLORS["bg_dark"],
    "axes.facecolor": COLORS["blue_light"],
    "axes.edgecolor": COLORS["slate"],
    "axes.labelcolor": COLORS["silver"],
    "axes.titlecolor": COLORS["silver"],
    "xtick.color": COLORS["silver_muted"],
    "ytick.color": COLORS["silver_muted"],
    "grid.color": COLORS["slate"],
    "text.color": COLORS["silver"],
}

# =============================================================================
# STYLE APPLICATION
# =============================================================================


def apply_ml4t_style(mode: Literal["light", "dark"] = "light") -> None:
    """Apply ML4T style to both Matplotlib and Plotly.

    Args:
        mode: 'light' (default) for white backgrounds, 'dark' for blue backgrounds
    """
    if mode == "light":
        plt.rcParams.update(ML4T_LIGHT_STYLE)
    else:
        plt.rcParams.update(ML4T_DARK_STYLE)

    # Apply Plotly template if available
    import contextlib

    with contextlib.suppress(ImportError):
        _register_plotly_template()


# =============================================================================
# PALETTE HELPERS
# =============================================================================


def ml4t_palette(n: int = 5, categorical: bool = False) -> list[str]:
    """Return colors from the ML4T palette.

    Args:
        n: Number of colors to return (max 5)
        categorical: If True, returns distinct colors for categories.
                    If False, returns blue gradient for sequential data.

    Returns:
        List of hex color strings
    """
    if categorical:
        # The cycle's first five, so the one categorical helper a notebook is told to
        # call and the cycle a bare figure draws from cannot disagree. This list had
        # `slate` third, so three categories already drew two navies, and
        # `silver_muted` fifth, which is #e8e8e6 and barely visible on the page at all.
        colors = list(COLOR_CYCLER[:5])
    else:
        colors = [
            COLORS["blue"],
            COLORS["slate"],
            COLORS["blue_light"],
            COLORS["silver_muted"],
            COLORS["silver"],
        ]
    return colors[:n]


def ml4t_diverging() -> list[str]:
    """Return diverging palette (negative to positive).

    Use for data with meaningful zero point (e.g., returns, correlations).

    Returns:
        List of 3 colors: [negative, neutral, positive]
    """
    return [COLORS["negative"], COLORS["silver_muted"], COLORS["positive"]]


# =============================================================================
# CHART HELPERS
# =============================================================================


def annotate_peak(ax: Axes, x: object, y: object, label: str, offset: tuple = (10, 10)) -> None:
    """Annotate a peak/trough with ML4T styling.

    Args:
        ax: matplotlib axes
        x, y: Coordinates of the point
        label: Text label
        offset: (x, y) offset in points
    """
    ax.annotate(
        label,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=9,
        color=COLORS["neutral"],
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["amber"],
            "connectionstyle": "arc3,rad=0.2",
        },
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": COLORS["silver"],
            "edgecolor": COLORS["silver_muted"],
        },
    )


def add_regime_shading(ax: Axes, periods: list[tuple], label: str = "Crisis") -> None:
    """Add regime shading to a time series plot.

    Args:
        ax: matplotlib axes
        periods: List of (start, end) tuples defining regime periods
        label: Label for legend
    """
    for i, (start, end) in enumerate(periods):
        ax.axvspan(
            start,
            end,
            alpha=0.15,
            color=COLORS["amber"],
            label=label if i == 0 else None,
        )


def format_pct_axis(ax: Axes, axis: Literal["x", "y", "both"] = "y") -> None:
    """Format axis as percentage with ML4T styling.

    Args:
        ax: matplotlib axes
        axis: Which axis to format ('x', 'y', or 'both')
    """
    from matplotlib.ticker import PercentFormatter

    formatter = PercentFormatter(xmax=1, decimals=0)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)


SUBTITLE_SIZE = 9


def _wrap_text_to_axes(ax: Axes, text: object, minimum: int = 24) -> int:
    """Break *text* onto as many lines as it takes to fit the axes' width.

    Returns the number of lines added, which the caller pays for in layout.

    The break points come from measuring the drawn text rather than from a
    characters-per-inch estimate: the ratio of the rendered width to the axes' width
    says directly how much of the string fits on one line, and it holds for whatever
    font, size and DPI are actually in force. That ratio is an average over the
    string, so a line that draws a run of unusually wide characters can still come
    out slightly over; the second pass measures the wrapped text and takes it back.
    """
    figure = ax.figure
    if figure is None or figure.canvas is None:
        return 0
    # A renderer, not a draw. `matplotlibrc` turns constrained layout on for every
    # figure in the repo, and drawing here runs that engine before the caller has
    # added its legend or called `tight_layout` - which on a small figure collapses
    # the axes and warns, into the notebook's own output. Measuring needs a renderer
    # and nothing else.
    renderer = getattr(figure.canvas, "get_renderer", None)
    if renderer is None:
        return 0
    renderer = renderer()
    axes_box = ax.get_window_extent(renderer)
    # The axes' width is what the subtitle should sit within; the room between the
    # subtitle's own left edge and the figure's is what it must not exceed, on pain
    # of `bbox="tight"` widening the canvas. Callers run `tight_layout` after this,
    # which moves the axes' left edge right to make room for tick labels, so the
    # second bound is taken with a margin covering that shift.
    limit = min(axes_box.width, (figure.bbox.x1 - axes_box.x0) * 0.92)
    if limit <= 0:
        return 0
    # `get_window_extent` on a multi-line Text returns the bounding box of every
    # line, so its width is the widest line's, which is the one that has to fit.
    for _ in range(2):
        drawn = text.get_window_extent(renderer)
        if drawn.width <= limit:
            break
        longest = max(text.get_text().split("\n"), key=len)
        per_char = drawn.width / max(len(longest), 1)
        # `fill` treats its input as one paragraph and collapses the newlines a
        # previous pass put in, so this re-wraps the subtitle rather than wrapping
        # the wrapping.
        text.set_text(textwrap.fill(text.get_text(), max(minimum, int(limit / per_char))))
    return text.get_text().count("\n")


def add_message_title(
    ax: Axes,
    message: str,
    subtitle: str | None = None,
    source: str | None = None,
) -> None:
    """Left-aligned takeaway title (a claim, not a label), optional subtitle + source note.

    `message` should state the finding ("Momentum decays beyond a 12-month hold"), not
    label the axes. `subtitle` carries the qualifier the title omits (metric, universe,
    frequency, period); `source` is a small bottom-left note. No figure number — the
    publisher captions separately.
    """
    extra_lines = 0
    if subtitle:
        note = ax.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=SUBTITLE_SIZE,
            color=COLORS["neutral"],
        )
        # A subtitle was one line anchored to the axes' left edge and never measured
        # against the axes' width, and every figure here saves with
        # `savefig.bbox="tight"`. So a subtitle wider than the axes did not overflow
        # and did not wrap: it widened the saved canvas to fit itself, and the figure
        # came out with the plot in the left quarter and one line of gray running
        # across the rest - which reads as a broken render rather than a long
        # subtitle. Three of the six figures in us_firm_characteristics/03 hit it on
        # a single pass, and nothing warned. Wrapping here retires the whole class:
        # no caller can trip it, and no caller has to count characters against an
        # axes width it cannot see.
        extra_lines = _wrap_text_to_axes(ax, note)
    ax.set_title(
        message,
        loc="left",
        color=COLORS["blue"],
        fontweight="semibold",
        # The subtitle sits inside the title's pad and is anchored by its bottom, so
        # each wrapped line grows up towards the title. `tight_layout` reserves the
        # pad, so paying for those lines here is what keeps the figure from
        # overlapping them.
        pad=(15 + extra_lines * (SUBTITLE_SIZE + 2)) if subtitle else 8,
    )
    if source:
        ax.figure.text(
            0.01, 0.005, source, ha="left", va="bottom", fontsize=8, color=COLORS["neutral"]
        )


def show_with_alt(fig: object, alt: str) -> None:
    """Render *fig* carrying alt text for screen readers, then close it.

    `plt.show()` emits an `<img>` with no alternative text, which nbconvert warns
    about and a screen reader cannot describe. The alt text is a sentence saying
    what the chart shows, not a repeat of the title.
    """
    from IPython.display import display

    display(fig, metadata={"image/png": {"alt": alt}})
    plt.close(fig)


def show_plotly_with_alt(fig: object, alt: str) -> None:
    """Render a Plotly *fig* carrying alt text for screen readers.

    Neither `fig.show()` nor `display(fig, metadata=...)` attaches it. Plotly returns
    its own metadata from `_repr_mimebundle_`, and IPython uses that in preference to
    a caller's, so the alt is silently dropped and the PNG reaches the page described
    as "No description has been provided for this image". Publishing the bundle
    directly is what carries it through.

    The alt text is a sentence saying what the chart shows, not a repeat of the title.
    """
    from IPython.display import publish_display_data

    bundle = fig._repr_mimebundle_()
    data, metadata = bundle if isinstance(bundle, tuple) else (bundle, {})
    publish_display_data(data=data, metadata={**metadata, "image/png": {"alt": alt}})


def label_line_ends(ax: Axes, xpad_points: int = 4, expand_right: float = 0.10) -> None:
    """Direct-label each labeled line at its last finite point; use instead of a legend.

    Preferred over a legend for <=5 series. Expands the right margin to fit labels.
    Do not also call ``ax.legend()``. Each label inherits its line's color, so the chart
    stays legible in grayscale (position + color, no legend lookup).
    """
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + (x1 - x0) * expand_right)
    for line in ax.get_lines():
        label = line.get_label()
        if not label or label.startswith("_"):
            continue
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            continue
        ax.annotate(
            label,
            xy=(x[m][-1], y[m][-1]),
            xytext=(xpad_points, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            color=line.get_color(),
            fontsize=9,
            annotation_clip=False,
        )


def zero_line(ax: Axes, at: float = 0.0, axis: Literal["x", "y"] = "y") -> None:
    """Thin neutral reference line (zero, benchmark=1.0, mean, threshold) behind the data."""
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(at, color=COLORS["neutral"], linewidth=0.8, linestyle="--", zorder=0.5)


def upper_triangle_mask(corr: object) -> np.ndarray:
    """Boolean mask hiding the redundant upper triangle of a symmetric matrix.

    For correlation/covariance heatmaps: ``sns.heatmap(corr, mask=upper_triangle_mask(corr),
    vmin=-1, vmax=1, center=0, cmap=...)``.
    """
    return np.triu(np.ones_like(np.asarray(corr), dtype=bool), k=1)


# Canonical panel-label size for Plotly subplot figures. Plotly writes
# subplot_titles as annotations carrying an explicit size of 16, and an explicit
# value beats the template's annotationdefaults, so the template cannot lower
# them from here. That leaves a subplot figure with 16pt panel labels above a
# 14pt figure title. style_subplot_titles() is the opt-in helper that restores
# the order: title 14 > panel label 13 > body 11.
SUBPLOT_TITLE_SIZE = 13


# The size Plotly stamps on every subplot-title annotation it creates. Part of
# the signature below, so a change to this default makes the helper match
# nothing rather than match the wrong thing;
# test_style_subplot_titles_restyles_panel_labels fails first if it moves.
_PLOTLY_SUBPLOT_TITLE_SIZE = 16


def _is_subplot_title(annotation: object) -> bool:
    """Match the annotations ``make_subplots`` creates for ``subplot_titles``.

    Plotly stamps all six of these properties on a subplot title and on nothing
    else it creates: arrowless, referenced to the paper in both axes, centered
    horizontally, anchored at the bottom, and sized 16. A caption or source note
    added with ``add_annotation`` normally differs in at least one - most often
    the size, since 16 is not a size hand-written annotations pick.

    An ``add_annotation`` call that happens to set all six identically is
    indistinguishable from a panel label and will be restyled. Nothing in the
    figure records which annotations Plotly authored, so that case cannot be
    separated; it just needs a different size or anchor to opt out.
    """
    return (
        getattr(annotation, "showarrow", None) is False
        and getattr(annotation, "xref", None) == "paper"
        and getattr(annotation, "yref", None) == "paper"
        and getattr(annotation, "xanchor", None) == "center"
        and getattr(annotation, "yanchor", None) == "bottom"
        and getattr(getattr(annotation, "font", None), "size", None) == _PLOTLY_SUBPLOT_TITLE_SIZE
    )


def style_subplot_titles(fig: object, size: int = SUBPLOT_TITLE_SIZE) -> object:
    """Bring ``make_subplots`` panel labels under the figure title.

    Plotly hardcodes 16pt on subplot-title annotations, which outranks both the
    body text and the 14pt figure title the template sets. Call this after
    building a subplot figure::

        fig = make_subplots(rows=2, cols=2, subplot_titles=[...])
        ...
        style_subplot_titles(fig)

    Only the panel labels are restyled - ``add_annotation`` callouts keep their
    own font, so the call is safe at any point in figure construction. It also
    applies once: a restyled label no longer carries Plotly's 16pt and so falls
    out of the selector, which makes a second call at a different ``size`` a
    no-op rather than a second restyle.

    Returns the figure so it can be chained.
    """
    fig.update_annotations(  # type: ignore[attr-defined]
        font={"size": size, "color": COLORS["slate"]},
        selector=_is_subplot_title,
    )
    return fig


# =============================================================================
# PLOTLY TEMPLATE (optional — only used if Plotly is installed)
# =============================================================================


def _register_plotly_template() -> None:
    """Register the ML4T template with Plotly."""
    import plotly.graph_objects as go
    import plotly.io as pio

    template = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="DM Sans, DejaVu Sans, sans-serif",
                size=11,
                color=COLORS["neutral"],
            ),
            paper_bgcolor=COLORS["bg_light"],
            plot_bgcolor="white",
            title=dict(
                font=dict(size=14, color=COLORS["blue"]),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                gridcolor=COLORS["silver_muted"],
                linecolor=COLORS["silver_muted"],
                tickfont=dict(size=10),
                title=dict(font=dict(size=11)),
                showgrid=True,
                gridwidth=0.5,
            ),
            yaxis=dict(
                gridcolor=COLORS["silver_muted"],
                linecolor=COLORS["silver_muted"],
                tickfont=dict(size=10),
                title=dict(font=dict(size=11)),
                showgrid=True,
                gridwidth=0.5,
            ),
            # The list itself, not a copy of it. This carried `slate` third, so a bare
            # `go.Figure` with three traces already drew two of them in navy - the same
            # defect as the Matplotlib cycle and one series sooner. Deriving it removes
            # the possibility of the two drifting apart again.
            colorway=list(COLOR_CYCLER),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLORS["silver_muted"],
                borderwidth=1,
                font=dict(size=10),
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=11,
                font_family="DM Sans, DejaVu Sans, sans-serif",
            ),
        )
    )
    pio.templates["ml4t"] = template
    # Make it the default so every Plotly figure inherits the ML4T font,
    # backgrounds, gridlines, and colorway — the palette applies repo-wide
    # without each notebook having to opt in (matplotlib gets this via
    # matplotlibrc; this is the Plotly equivalent).
    pio.templates.default = "ml4t"


# Auto-register Plotly template on import
_register_plotly_template()
HAS_PLOTLY = True


# =============================================================================
# PUBLICATION (BOOK) STYLE — MIT Press, dual-track (grayscale print + color web)
# =============================================================================
# Used by `~/ml4t/book/<ch>/figures/scripts/generate_figure_*.py`.
# Notebooks may also call `apply_book_style()` to render the same look.
#
# Two tracks, same data, same script:
#   - "print": grayscale-first, semantic fills, varied linestyles. Top-level PNG.
#   - "color": ML4T palette overlay. `color/` subdir.
# The grayscale track is the source of truth: data must be legible without color.

# Semantic grayscale fills — vocabulary mirrors `visualization-style/SKILL.md`.
# Use these by ROLE, not by hex. The print track resolves them to grays;
# the color track resolves them to the ML4T palette.
GRAY_FILLS = {
    "primary": "#000000",  # titles, lead data series, key emphasis
    "secondary": "#808080",  # second series — widened from #404040 for print contrast
    "tertiary": "#c8c8c8",  # third series, supporting elements
    "quaternary": "#e8e8e8",  # fourth series only — keep grayscale separable
    "muted": "#a8a8a8",  # de-emphasized, comparison baselines
    "border": "#666666",  # connectors, axis lines (data side)
    "highlight": "#d9d9d9",  # ~85% white — emphasis band fill
    "container": "#f2f2f2",  # ~95% white — phase container fill
    "foundation": "#b3b3b3",  # ~70% white — foundation layer fill
    "canvas": "#ffffff",  # page background
}

COLOR_FILLS = {
    "primary": COLORS["blue"],  # #0a1628 navy — primary series
    "secondary": COLORS["amber"],  # #D4A84B amber — secondary series
    "tertiary": COLORS["copper"],  # #C87533 copper — tertiary (kept distinct from navy)
    "quaternary": COLORS["slate"],  # #1a2d4a mid-blue — fourth series only
    "muted": COLORS["silver_muted"],  # #e8e8e6
    "border": COLORS["neutral"],  # #334155
    "highlight": COLORS["amber_light"],
    "container": COLORS["bg_light"],
    "foundation": COLORS["silver"],
    "canvas": "#ffffff",
}


# =============================================================================
# CANONICAL FIGURE SIZES (Packt embed width = 5.833")
# =============================================================================
# Width is fixed at 5.833" — the typeset width Packt uses in the manuscript
# template. Heights are picked per layout so panels render at proportions
# that don't dominate page vertical space. Use these in generate scripts;
# do NOT introduce ad-hoc figsize tuples per figure.
PAGE_WIDTH = 5.833  # Packt typeset embed width in inches

FIGSIZE = {
    "single_wide": (PAGE_WIDTH, 2.6),  # short time series, comparisons
    "single": (PAGE_WIDTH, 3.4),  # default single panel (~1.7:1)
    "single_tall": (PAGE_WIDTH, 4.0),  # detail-heavy single panel
    "dual_h": (PAGE_WIDTH, 2.6),  # two side-by-side panels
    "dual_h_tall": (PAGE_WIDTH, 3.2),  # two side-by-side, taller panels
    "dual_v": (PAGE_WIDTH, 5.0),  # two stacked panels
    "triple_h": (PAGE_WIDTH, 2.2),  # three side-by-side panels, short
    "triple_h_tall": (PAGE_WIDTH, 3.0),  # three side-by-side, detail
    "grid_2x2": (PAGE_WIDTH, 4.0),  # 2 rows × 2 cols, simple axes
    "grid_2x3": (PAGE_WIDTH, 3.5),  # 2 rows × 3 cols
    "grid_3x2": (PAGE_WIDTH, 5.5),  # 3 rows × 2 cols (square-ish grid)
    "dashboard_2x2": (PAGE_WIDTH, 5.5),  # 2×2 with date axes / rotated labels
    "dashboard_2x3": (PAGE_WIDTH, 4.5),  # 2×3 with date axes / rotated labels
}


_BOOK_BASE_STYLE = {
    # Kept in sync with matplotlibrc at repo root. The auto-applied
    # matplotlibrc covers all default runs; this dict is the explicit-apply
    # override for book-figure scripts that swap between print and color
    # tracks via ``apply_book_style()``.
    "figure.dpi": 100,
    "figure.figsize": FIGSIZE["single"],
    "figure.facecolor": COLORS["bg_light"],
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.facecolor": COLORS["bg_light"],
    "axes.facecolor": COLORS["bg_light"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.labelsize": 9,
    "axes.labelpad": 4,
    "axes.linewidth": 0.75,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.handlelength": 2.0,
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Source Sans 3", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "image.cmap": "cividis",
}


def _cycler(colors: list[str], linestyles: list[str] | None = None):
    """Build a prop_cycle from colors + optional linestyles. Local import keeps
    the module top-level cheap."""
    from cycler import cycler as cy

    cyc = cy(color=colors)
    if linestyles is not None:
        cyc = cyc + cy(linestyle=linestyles[: len(colors)])
    return cyc


BOOK_PRINT_STYLE = {
    # PRINT track is for the printed book — on white paper, so revert
    # the warm-cream backgrounds back to plain white.
    **_BOOK_BASE_STYLE,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#000000",
    "axes.titlecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.color": "#cccccc",
    "text.color": "#000000",
}

BOOK_COLOR_STYLE = {
    # COLOR track is for web/README/Google Drive — matches the website's
    # warm-cream bg_light surface.
    **_BOOK_BASE_STYLE,
    "axes.edgecolor": COLORS["neutral"],
    "axes.labelcolor": COLORS["neutral"],
    "axes.titlecolor": COLORS["neutral"],
    "xtick.color": COLORS["neutral"],
    "ytick.color": COLORS["neutral"],
    "grid.color": COLORS["silver_muted"],
    "text.color": COLORS["neutral"],
}


def apply_book_style(mode: Literal["print", "color"] = "print") -> None:
    """Set rcParams for a book-figure generation script.

    Call once at script start (or before each render in a dual-track loop).
    Resolves the prop_cycle to grayscale (with linestyle variation) for
    ``print`` and to the ML4T color palette for ``color``.
    """
    style = BOOK_PRINT_STYLE if mode == "print" else BOOK_COLOR_STYLE
    plt.rcParams.update(style)
    if mode == "print":
        plt.rcParams["axes.prop_cycle"] = _cycler(GRAY_CYCLER, LINESTYLE_CYCLER)
    else:
        plt.rcParams["axes.prop_cycle"] = _cycler(COLOR_CYCLER)


def save_dual(
    make_fig,
    output_basename: str,
    output_dir: str | Path,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Render and save both tracks of a publication figure.

    ``make_fig(palette, mode)`` is called twice — once with ``GRAY_FILLS`` /
    ``"print"`` and once with ``COLOR_FILLS`` / ``"color"``. The print PNG
    lands at ``output_dir/{basename}.png`` (top-level grayscale default).
    The color PNG lands at ``output_dir/color/{basename}_color.png``.

    Args:
        make_fig: Callable ``(palette: dict, mode: str) -> matplotlib.Figure``.
            Must build the figure from scratch each call — the caller closes it.
        output_basename: e.g. ``"figure_2_2_survivorship_bias"``. No extension.
        output_dir: chapter ``figures/`` directory.
        dpi: PNG resolution (default 300).

    Returns:
        (print_path, color_path) — both absolute.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    color_dir = output_dir / "color"
    color_dir.mkdir(parents=True, exist_ok=True)

    # Print track first — that's the canonical artifact.
    apply_book_style("print")
    fig = make_fig(GRAY_FILLS, "print")
    print_path = output_dir / f"{output_basename}.png"
    fig.savefig(print_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Color track.
    apply_book_style("color")
    fig = make_fig(COLOR_FILLS, "color")
    color_path = color_dir / f"{output_basename}_color.png"
    fig.savefig(color_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return print_path, color_path


# =============================================================================
# BOOK-SPECIFIC FUNCTIONS (LEGACY)
# =============================================================================

# DEPRECATED: Style is now applied automatically via matplotlibrc in repo root.
ML4T_STYLE = Path(__file__).parent.parent / "matplotlibrc"


def save_figure(
    fig,
    name: str,
    chapter: str | None = None,
    formats: list[str] | None = None,
    dpi: int = 150,
) -> None:
    """Save figure with ML4T conventions.

    Args:
        fig: matplotlib figure object
        name: Base filename (without extension)
        chapter: Optional chapter directory (e.g., "06_alpha_factor_engineering")
        formats: List of formats to save (default: ['png', 'pdf'])
        dpi: Resolution for raster formats (default: 150)
    """
    formats = formats or ["png", "pdf"]

    if chapter:
        repo_root = Path(__file__).parent.parent
        output_dir = repo_root / chapter / "visualizations"
    else:
        output_dir = Path(".")

    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = output_dir / f"{name}.{fmt}"
        fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")


def plot_fidelity_comparison(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    title: str = "Real vs Synthetic Distribution",
    n_samples: int = 1000,
    figsize: tuple = (12, 5),
    flatten_method: str = "mean",
    random_state: int = 42,
) -> plt.Figure:
    """Create standardized fidelity comparison plot using PCA and t-SNE.

    Designed for grayscale compatibility:
    - Real data: dark circles (filled)
    - Synthetic data: amber X markers (open)

    Args:
        real_data: Real sequences. Shape can be:
            - (n_samples, seq_len, n_features): 3D time series
            - (n_samples, n_features): 2D tabular
        synthetic_data: Synthetic sequences, same shape as real_data
        title: Plot title
        n_samples: Number of samples to visualize (subsampled if larger)
        figsize: Figure size (width, height)
        flatten_method: How to flatten 3D data to 2D:
            - "mean": Average across time dimension (default)
            - "last": Use last timestep only
            - "flatten": Concatenate all timesteps (high-dim)
        random_state: Random seed for reproducibility

    Returns:
        matplotlib Figure object
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    np.random.seed(random_state)

    # Handle 3D (time series) vs 2D (tabular) data
    if real_data.ndim == 3:
        if flatten_method == "mean":
            real_flat = real_data.mean(axis=1)
            synth_flat = synthetic_data.mean(axis=1)
        elif flatten_method == "last":
            real_flat = real_data[:, -1, :]
            synth_flat = synthetic_data[:, -1, :]
        elif flatten_method == "flatten":
            real_flat = real_data.reshape(real_data.shape[0], -1)
            synth_flat = synthetic_data.reshape(synthetic_data.shape[0], -1)
        else:
            raise ValueError(f"Unknown flatten_method: {flatten_method}")
    else:
        real_flat = real_data
        synth_flat = synthetic_data

    # Subsample for visualization
    n_viz = min(n_samples, len(real_flat), len(synth_flat))
    idx_real = np.random.choice(len(real_flat), n_viz, replace=False)
    idx_synth = np.random.choice(len(synth_flat), n_viz, replace=False)

    real_sample = real_flat[idx_real]
    synth_sample = synth_flat[idx_synth]

    # PCA - fit on real, transform both
    n_features = real_sample.shape[1] if real_sample.ndim > 1 else 1
    n_pca = min(2, n_features, n_viz)
    pca = PCA(n_components=n_pca)
    pca.fit(real_sample)
    real_pca = pca.transform(real_sample)
    synth_pca = pca.transform(synth_sample)

    # t-SNE - fit jointly for proper comparison
    combined = np.vstack([real_sample, synth_sample])
    perplexity = min(40, max(2, n_viz // 4))
    n_tsne = min(2, n_features)
    tsne = TSNE(
        n_components=n_tsne, perplexity=perplexity, max_iter=1000, random_state=random_state
    )
    combined_tsne = tsne.fit_transform(combined)
    real_tsne = combined_tsne[:n_viz]
    synth_tsne = combined_tsne[n_viz:]

    # Create figure with aligned axes
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Style constants for grayscale compatibility
    real_color = COLORS["blue"]
    synth_color = COLORS["amber"]
    marker_size = 25
    alpha = 0.6

    # PCA plot (handle 1D case when n_features < 2)
    pca_y_real = real_pca[:, 1] if n_pca >= 2 else np.zeros(len(real_pca))
    pca_y_synth = synth_pca[:, 1] if n_pca >= 2 else np.zeros(len(synth_pca))
    axes[0].scatter(
        real_pca[:, 0],
        pca_y_real,
        c=real_color,
        marker="o",
        s=marker_size,
        alpha=alpha,
        label="Real",
        edgecolors="none",
    )
    axes[0].scatter(
        synth_pca[:, 0],
        pca_y_synth,
        c=synth_color,
        marker="x",
        s=marker_size,
        alpha=alpha,
        label="Synthetic",
        linewidths=1.5,
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2" if n_pca >= 2 else "")
    axes[0].set_title("PCA Projection")
    axes[0].legend(loc="upper right", framealpha=0.9)

    # t-SNE plot (handle 1D case when n_features < 2)
    tsne_y_real = real_tsne[:, 1] if n_tsne >= 2 else np.zeros(len(real_tsne))
    tsne_y_synth = synth_tsne[:, 1] if n_tsne >= 2 else np.zeros(len(synth_tsne))
    axes[1].scatter(
        real_tsne[:, 0],
        tsne_y_real,
        c=real_color,
        marker="o",
        s=marker_size,
        alpha=alpha,
        label="Real",
        edgecolors="none",
    )
    axes[1].scatter(
        synth_tsne[:, 0],
        tsne_y_synth,
        c=synth_color,
        marker="x",
        s=marker_size,
        alpha=alpha,
        label="Synthetic",
        linewidths=1.5,
    )
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2" if n_tsne >= 2 else "")
    axes[1].set_title("t-SNE Projection")
    axes[1].legend(loc="upper right", framealpha=0.9)

    fig.suptitle(title, fontsize=14, fontweight="semibold", y=1.02)
    plt.tight_layout()

    return fig


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Palette
    "COLORS",
    "GRAYSCALE",
    # Matplotlib styles
    "ML4T_LIGHT_STYLE",
    "ML4T_DARK_STYLE",
    # Style application
    "apply_ml4t_style",
    # Palette helpers
    "ml4t_palette",
    "ml4t_diverging",
    # Chart helpers
    "annotate_peak",
    "add_regime_shading",
    "format_pct_axis",
    "add_message_title",
    "show_with_alt",
    "label_line_ends",
    "zero_line",
    "upper_triangle_mask",
    "style_subplot_titles",
    "SUBPLOT_TITLE_SIZE",
    # Book-specific
    "ML4T_STYLE",
    "HAS_PLOTLY",
    "save_figure",
    "plot_fidelity_comparison",
]
