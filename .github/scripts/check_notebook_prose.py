#!/usr/bin/env python3
"""A figure's description must describe the figure the cell draws.

A reader who cannot see a figure is handed its alt text instead. When that text names a
chart the cell never draws, that reader is not told less than a sighted one, they are
told something false, and nothing in the repository could see it: ruff, the formatter and
the notebook tests all read code, and a prose rewrite that swaps "heatmap" for "grouped
bars" changes no code at all (ml4t/agent-workspace#1164).

Three shapes are reported:

- alt text naming a chart type the cell does not draw
- bars described in groups where one Matplotlib bar call draws one series
- a horizontal bar's extent called its height

What it does NOT decide is worth stating, because the boundary is deliberate and was
measured. `nasdaq100_microstructure/05_evaluation` called a twenty-bar `go.Bar` chart "a
histogram" whose within-family and cross-family pairs were "drawn separately", and the
cell neither bins nor separates. That shipped, and was found by reading rather than by
this rule, which stays silent on it on purpose: a discrete histogram IS drawn with bar
calls, so a bar call cannot refute the word "histogram", and the narrower claim about
separation is not decidable from the call at all. A rule that reported it would report
every honest discrete histogram in the repository with it. The reading half of the clause
belongs in the per-notebook review checklist; this file is the half a machine can settle.

Only chart-type claims are restricted to alt text. Markdown around a figure names other
figures constantly and without a deictic, so reading it as a claim about the cell beside
it reports a false positive per case study; alt text has one subject by construction. The
grouping and height rules apply to both, because both describe a structure rather than
name a chart, and the one shipped height error was in prose.

**This file is canonical for this rule.** `scripts/check_notebook_prose.py` in the agents
repository carries the same three checks as clause A10 of its larger prose sweep, whose
other rules are not gateable here: measured over this repository at `0c4b8253`, the whole
sweep reports 1,237 violations across 212 notebooks, 603 of them a single rule about
comment blocks inside code cells. Wiring that would red every open PR, which is a
stop-work order rather than a gate. A10 alone reports zero. Both copies are pinned by the
same cases - `--selftest` here, `tests/test_notebook_prose_checks.py` there - so an edit
to either is caught by its own tests; a case added to one side only is not, and that is
the accepted limit of keeping two copies.

Stdlib only, nothing imported or executed. `--selftest` proves each rule fires and does
not fire.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

UNRESOLVED = object()


def markdown_cells(src: str):
    """Yield (line_offset, text, tagged_results) for each jupytext percent markdown cell."""
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("# %% [markdown]"):
            tagged = "results" in lines[i]
            start = i + 2
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("# %%"):
                line = lines[i]
                body.append(line[2:] if line.startswith("# ") else "")
                i += 1
            yield start, "\n".join(body), tagged
        else:
            i += 1


def code_cells(src: str):
    """Yield (line_offset, [lines], tagged_results) for each jupytext percent code cell.

    The YAML front matter above the first `# %%` is not a cell and is skipped.
    """
    lines = src.splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("# %%"):
        i += 1
    while i < len(lines):
        if lines[i].startswith("# %%") and "[markdown]" not in lines[i]:
            tagged = "results" in lines[i]
            start = i + 2
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("# %%"):
                body.append(lines[i])
                i += 1
            yield start, body, tagged
        else:
            i += 1


def dedent_cell(body: list[str]) -> str:
    """A cell's lines as parseable source. A cell never opens mid-block, so this is verbatim."""
    return "\n".join(body)


def _cell_literals(tree: ast.AST) -> dict[str, ast.expr]:
    """Names bound exactly once in this cell to something whose text is knowable.

    A name assigned twice, or assigned anything else anywhere in the cell, is dropped:
    following it would report text that is not what reaches the reader.
    """
    bound: dict[str, ast.expr] = {}
    spoiled: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AugAssign):
            target, value = node.target, None
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if target.id in bound or value is None:
            spoiled.add(target.id)
        else:
            bound[target.id] = value
    return {name: value for name, value in bound.items() if name not in spoiled}


def _literal_text(node: ast.expr, names: dict[str, ast.expr] | None = None) -> str | object:
    """The source-literal text of a string, interpolations blanked out.

    ``UNRESOLVED`` where the text cannot be read off the source. Concatenation,
    ``str.join`` over literal parts, a conditional whose branches are both literal, and a
    name bound once in the same cell all resolve.
    """
    names = names or {}
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else UNRESOLVED
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_text(node.left, names)
        right = _literal_text(node.right, names)
        if left is UNRESOLVED or right is UNRESOLVED:
            return UNRESOLVED
        return f"{left}{right}"
    if isinstance(node, ast.IfExp):
        # Both branches reach a reader, on different runs. Checking their concatenation
        # holds each to the rules without deciding which one runs.
        body = _literal_text(node.body, names)
        orelse = _literal_text(node.orelse, names)
        if body is UNRESOLVED or orelse is UNRESOLVED:
            return UNRESOLVED
        return f"{body}\n{orelse}"
    if isinstance(node, ast.Name) and node.id in names:
        return _literal_text(names[node.id], names)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.List | ast.Tuple)
    ):
        separator = _literal_text(node.func.value, names)
        parts = [_literal_text(part, names) for part in node.args[0].elts]
        if separator is UNRESOLVED or any(part is UNRESOLVED for part in parts):
            return UNRESOLVED
        return str(separator).join(str(part) for part in parts)
    return UNRESOLVED


DRAWING_CALLS: dict[str, frozenset[str]] = {
    "bar": frozenset({"bar", "bar_vertical", "histogram"}),
    "barh": frozenset({"bar", "bar_horizontal", "histogram"}),
    "Bar": frozenset({"bar", "bar_vertical", "bar_horizontal", "histogram"}),
    "hist": frozenset({"histogram", "bar", "bar_vertical"}),
    "hist2d": frozenset({"histogram", "heatmap"}),
    "Histogram": frozenset({"histogram", "bar"}),
    "plot": frozenset({"line"}),
    "step": frozenset({"line"}),
    "stackplot": frozenset({"line", "area"}),
    "line": frozenset({"line"}),
    "fill_between": frozenset({"area", "line"}),
    "fill_betweenx": frozenset({"area", "line"}),
    "scatter": frozenset({"scatter"}),
    "Scatter": frozenset({"line", "scatter"}),
    "Scattergl": frozenset({"line", "scatter"}),
    "errorbar": frozenset({"line", "scatter"}),
    "hexbin": frozenset({"scatter", "heatmap"}),
    "imshow": frozenset({"heatmap"}),
    "matshow": frozenset({"heatmap"}),
    "pcolormesh": frozenset({"heatmap"}),
    "pcolor": frozenset({"heatmap"}),
    "Heatmap": frozenset({"heatmap"}),
    "contour": frozenset({"contour"}),
    "contourf": frozenset({"contour"}),
    "pie": frozenset({"pie"}),
    "Pie": frozenset({"pie"}),
    "boxplot": frozenset({"box"}),
    "Box": frozenset({"box"}),
    "bxp": frozenset({"box"}),
    "violinplot": frozenset({"violin"}),
    "Violin": frozenset({"violin"}),
    # Plotly Express and seaborn draw the same charts under their own names, and `fig.add_*`
    # is the Plotly shortcut for the trace constructors above. A name missing here reads as a
    # cell that draws nothing, which is why `px.histogram` and `sns.histplot` each produced a
    # finding against alt text that was correct.
    "histogram": frozenset({"histogram", "bar"}),
    "histplot": frozenset({"histogram", "bar"}),
    "displot": frozenset({"histogram", "bar"}),
    "countplot": frozenset({"bar"}),
    "barplot": frozenset({"bar"}),
    "lineplot": frozenset({"line"}),
    "kdeplot": frozenset({"line"}),
    "ecdfplot": frozenset({"line"}),
    "stairs": frozenset({"line"}),
    "area": frozenset({"area", "line"}),
    "scatterplot": frozenset({"scatter"}),
    "stripplot": frozenset({"scatter"}),
    "swarmplot": frozenset({"scatter"}),
    "regplot": frozenset({"scatter", "line"}),
    "lmplot": frozenset({"scatter", "line"}),
    "jointplot": frozenset({"scatter"}),
    "eventplot": frozenset({"scatter"}),
    "heatmap": frozenset({"heatmap"}),
    "density_heatmap": frozenset({"heatmap"}),
    "pcolorfast": frozenset({"heatmap"}),
    "boxenplot": frozenset({"box"}),
    "broken_barh": frozenset({"bar", "bar_horizontal"}),
    "add_bar": frozenset({"bar", "bar_vertical", "bar_horizontal"}),
    "add_scatter": frozenset({"line", "scatter"}),
    "add_heatmap": frozenset({"heatmap"}),
    "add_histogram": frozenset({"histogram", "bar"}),
    "add_box": frozenset({"box"}),
    "add_violin": frozenset({"violin"}),
    "add_pie": frozenset({"pie"}),
}
BAR_CALLS = frozenset(
    {
        "bar",
        "barh",
        "Bar",
        "hist",
        "Histogram",
        "histogram",
        "histplot",
        "barplot",
        "countplot",
        "add_bar",
        "add_histogram",
    }
)

# A phrase that names a structure, and the kind the cell must therefore draw. Only phrases
# whose reading is not in doubt: a bare "scatter" is a verb as often as a chart, "curve" is
# in "equity curve", and "bars" is in "error bars" and "colour bar", so each of those is
# either spelled out or excluded below.
LAYOUT_CLAIMS: list[tuple[str, str]] = [
    (r"(?<!error[- ])(?<!colour[- ])(?<!color[- ])(?<!tool)\bbar chart\b", "bar"),
    (r"\bgrouped bars?\b|\bclustered bars?\b|\bgroups? of bars\b", "bar"),
    (r"\bheat ?maps?\b", "heatmap"),
    (r"\bscatter ?plots?\b|\bscatter chart\b|\bscatter of\b", "scatter"),
    (r"\bhistograms?\b", "histogram"),
    (r"\bpie chart\b", "pie"),
    (r"\bbox ?plots?\b", "box"),
    (r"\bviolin plots?\b", "violin"),
    (r"\bline charts?\b|\bline plots?\b", "line"),
    (r"\bstacked areas?\b|\barea chart\b", "area"),
    (r"\bcontour (?:plot|lines)\b", "contour"),
]
LAYOUT_CLAIMS_COMPILED = [(re.compile(p, re.I), kind) for p, kind in LAYOUT_CLAIMS]
# "a vertical bar marks the median" and "one horizontal bar per contract" are not claims about
# a bar chart: the first is `ax.scatter(marker="|")` and the second a thick `go.Scatter` line
# drawn as a Gantt bar, and both are what the reader sees. Orientation alone is therefore not
# checkable - three of the corpus's five orientation findings were glyphs described honestly -
# so only the chart types above are, plus the `barh` height rule, which fires only where a
# `barh` really was drawn.

# A claim that the bars are grouped needs more than one bar series on the axis. One
# `ax.barh` over a frame already reduced to one row per category draws one bar each, and the
# within-group comparison the sentence promises is not on that axis at all.
GROUPING_CLAIM = re.compile(
    r"\bgrouped bars?\b|\bclustered bars?\b|\bgroups? of bars\b|\bwithin (?:a|each) group\b",
    re.I,
)
# Bars drawn by `barh` have a length, not a height. The distinction is not pedantry: it is
# the tell that the sentence was written for a different figure.
HEIGHT_CLAIM = re.compile(r"\bheights?\b", re.I)

# A sentence that points at another figure is not describing this one. "the heatmap below
# reports every allocator per case study" is the correct prose for a `barh` cell, and reading
# it as a claim about that axis is how a checker teaches its own waiver. Claims are therefore
# matched a sentence at a time, and a sentence carrying one of these is not a claim about the
# figure the cell draws.
ELSEWHERE = re.compile(
    r"\b(?:below|above|earlier|later|next|previous|preceding|following|elsewhere)\b", re.I
)
SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+|\n{2,}")


def _claim_sentences(text: str) -> list[str]:
    """The sentences of *text* that are about the figure at hand."""
    return [s for s in SENTENCE_SPLIT.split(text) if s and not ELSEWHERE.search(s)]


def _is_matplotlib_bar(node: ast.Call) -> bool:
    """True where `bar`/`barh` is Matplotlib's, whose name alone fixes the orientation."""
    if any(kw.arg == "orientation" for kw in node.keywords):
        return False
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    while isinstance(receiver, (ast.Attribute, ast.Subscript, ast.Call)):
        receiver = getattr(receiver, "value", None) or getattr(receiver, "func", None)
    return not (isinstance(receiver, ast.Name) and receiver.id in ("px", "go", "plotly", "pl"))


def _drawn_kinds(tree: ast.AST) -> tuple[set[str], set[str], int, bool]:
    """`(kinds drawn, drawing call names, bar-family call count, a bar call sits in a loop)`."""
    kinds: set[str] = set()
    names: set[str] = set()
    bars = 0
    looped = False
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.comprehension))]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in DRAWING_CALLS:
            continue
        given = DRAWING_CALLS[name]
        if name in ("bar", "barh") and not _is_matplotlib_bar(node):
            given = frozenset({"bar", "bar_vertical", "bar_horizontal"})
        kinds |= given
        names.add(name)
        if name in BAR_CALLS:
            bars += 1
            # A bar call that can group on its own is not evidence of a single series:
            # `px.bar(..., color=..., barmode="group")` and `sns.barplot(..., hue=...)` each
            # draw a grouped chart from one call, and `bottom=`/`left=` stacks one. Only a
            # bare Matplotlib `ax.bar`/`ax.barh` says "one series" by being written once.
            # `color=` groups in Plotly Express and seaborn and does not in Matplotlib, where
            # it is a per-bar colour list; `bottom=`/`left=` is how Matplotlib stacks.
            stacked = {kw.arg for kw in node.keywords} & {"bottom", "left"}
            if (
                not _is_matplotlib_bar(node)
                or stacked
                or any(node in ast.walk(loop) for loop in loops)
            ):
                looped = True
    return kinds, names, bars, looped


# A figure is not always built in one cell. `make_subplots` in one cell, `fig.add_trace` in
# the next three, `show_plotly_with_alt` in the last is the shape every case study's stage-05
# evaluation notebook uses, and reading only the alt cell there reports the alt text as
# claiming bars it can see no call for while the bars sit two cells up. So the kinds are
# accumulated backwards over the contiguous code cells up to and including the one that
# constructs a figure.
FIGURE_CONSTRUCTORS = frozenset({"subplots", "subplot_mosaic", "figure", "Figure", "make_subplots"})
MAX_CELLS_BACK = 8


def _constructs_a_figure(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in FIGURE_CONSTRUCTORS:
                return True
    return False


def _alt_strings(tree: ast.AST, literals: dict[str, ast.expr]) -> list[str]:
    """Every alt text the cell hands a reader, from `show_with_alt` and `show_plotly_with_alt`."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in ("show_with_alt", "show_plotly_with_alt"):
            continue
        candidates = list(node.args[1:2])
        candidates += [kw.value for kw in node.keywords if kw.arg == "alt"]
        for candidate in candidates:
            text = _literal_text(candidate, literals)
            if isinstance(text, str):
                out.append(text)
    return out


def check_source(path: Path, src: str) -> list[str]:
    """A description of a figure's layout must match the calls that build it."""
    marks = [
        (offset, body, kind)
        for offset, body, kind in (
            *[(o, b, "code") for o, b, _ in code_cells(src)],
            *[(o, t, "markdown") for o, t, _ in markdown_cells(src)],
        )
    ]
    marks.sort(key=lambda m: m[0])
    problems = []
    for index, (offset, body, kind) in enumerate(marks):
        if kind != "code":
            continue
        try:
            tree = ast.parse(dedent_cell(body))
        except SyntaxError:
            continue
        drawn, called, bars, looped = _drawn_kinds(tree)
        # Widen only from a cell that draws. A cell whose figure comes back from a helper
        # draws nothing this can see, and walking back from there attaches the alt text to
        # whatever the previous figure happened to be.
        if drawn and not _constructs_a_figure(tree):
            back = index - 1
            steps = 0
            while back >= 0 and steps < MAX_CELLS_BACK:
                if marks[back][2] == "code":
                    try:
                        earlier = ast.parse(dedent_cell(marks[back][1]))
                    except SyntaxError:
                        break
                    more_drawn, more_called, more_bars, more_looped = _drawn_kinds(earlier)
                    drawn |= more_drawn
                    called |= more_called
                    bars += more_bars
                    looped = looped or more_looped
                    steps += 1
                    if _constructs_a_figure(earlier):
                        break
                back -= 1
        if not drawn:
            continue
        literals = _cell_literals(tree)
        # The alt string is checked as the cell's own text. The markdown either side is the
        # prose the same figure is read through; both neighbours because "Reading the chart"
        # is written below the figure as often as above it.
        sources = [(offset, "alt text", text) for text in _alt_strings(tree, literals)]
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < len(marks) and marks[neighbour][2] == "markdown":
                sources.append((marks[neighbour][0], "prose", marks[neighbour][1]))
        for line, where, text in sources:
            seen: set[str] = set()
            for sentence in _claim_sentences(text):
                for pattern, required in LAYOUT_CLAIMS_COMPILED:
                    match = pattern.search(sentence)
                    # Only the alt text is held to naming the right chart type. Markdown
                    # around a figure names other figures constantly and without a deictic -
                    # `cme_futures/06_linear` says "The bar chart mixes three estimators"
                    # above a `go.Scatter` cell, meaning the bar chart two sections back -
                    # and a checker that reads those as claims about the cell beside them
                    # reports a false positive per case study. Alt text has no such second
                    # subject: it is the description of one figure, for a reader who cannot
                    # see it.
                    if where != "alt text":
                        continue
                    if not match or required in drawn or required in seen:
                        continue
                    seen.add(required)
                    problems.append(
                        f"{path}:{line}: figure {where} claims {match.group(0)!r}, and the "
                        f"cell draws {', '.join(sorted(called))} - no {required}"
                    )
                if (
                    GROUPING_CLAIM.search(sentence)
                    and "bar" in drawn
                    and bars < 2
                    and not looped
                    and "grouping" not in seen
                ):
                    seen.add("grouping")
                    problems.append(
                        f"{path}:{line}: figure {where} describes bars in groups, and the cell "
                        f"makes one bar call over one series, so there is no within-group "
                        f"comparison"
                    )
                # Only where the figure is bars and nothing else. `factor_zoo_validation`
                # pairs an `errorbar` panel with a `barh` panel and says the markers sit "at
                # slightly offset heights", which is the markers' position on the shared axis
                # and correct.
                if (
                    HEIGHT_CLAIM.search(sentence)
                    and "bar_horizontal" in drawn
                    and "bar_vertical" not in drawn
                    and not (drawn - {"bar", "bar_horizontal", "histogram"})
                    and "height" not in seen
                ):
                    seen.add("height")
                    problems.append(
                        f"{path}:{line}: figure {where} calls a horizontal bar's extent its "
                        f"height; `barh` bars have a length"
                    )
    return problems


def check(path: Path) -> list[str]:
    return check_source(path, path.read_text(encoding="utf-8", errors="replace"))


# The violating halves below are `20_strategy_synthesis/05_portfolio_allocation.py` at
# `7567504a~1`, and the conforming halves the same cells at `7567504a`, where the instances
# ml4t/agent-workspace#1164 measured were corrected by hand. That two-sided pair is what the
# rule was calibrated against: five successive narrowings were tried to kill the false
# positives below, and two of them silently killed detection as well. The cases that must
# report ZERO are therefore the valuable half of this list. Each is a real sentence from this
# repository that an earlier form of the rule reported and should not have, so widening a
# waiver to "fix" a future false positive has to keep all of them at zero to be a fix at all.
HEATMAP_CELL = (
    "# %%\n"
    "im = ax.imshow(matrix, cmap=cmap, aspect='auto')\n"
    "fig.colorbar(im, ax=ax, label='Sharpe Ratio', shrink=0.8)\n"
    "show_with_alt(\n"
    "    fig,\n"
    "    {alt!r},\n"
    ")\n"
)
GROUPED_BARS_ALT = (
    "Grouped bars comparing the Sharpe of each of the most common allocators within each "
    "case study, so the height differences within a group show how much the allocator "
    "choice moved the result."
)
HEATMAP_ALT = (
    "Heatmap with one row per case study and one column per allocator, each cell annotated "
    "with that pair's Sharpe and coloured red through green over the range minus one to one."
)
BARH_CELL = (
    "# %%\n"
    "bars = ax.barh(\n"
    "    best_data['display_name'],\n"
    "    best_data['sharpe'],\n"
    "    color=[colors_map.get(a, 'steelblue') for a in best_data['allocator']],\n"
    ")\n"
    "show_with_alt(fig, 'One horizontal bar per case study.')\n"
)
GROUPED_PROSE = (
    "# %% [markdown]\n"
    "# **Reading the chart**: one group of bars per case study, one bar per allocator, all on\n"
    "# a shared Sharpe axis. The height differences within a group are what the allocator\n"
    "# choice is worth for that case study.\n"
)
DESCRIPTIVE_PROSE = (
    "# %% [markdown]\n"
    "# **Reading the chart**: one horizontal bar per case study, whose length is the Sharpe of\n"
    "# that case study's best allocator. Because each bar is already a maximum over\n"
    "# allocators, the heatmap below is where the within-case comparison is read.\n"
)

# Each case is (label, source, findings expected, phrases every finding set must contain).
# The phrases matter as much as the count: a case expecting two findings is satisfied by two
# grouping hits unless the height message is named, and the two rules fire on the same cell.
CASES: list[tuple[str, str, int, tuple[str, ...]]] = [
    (
        "alt text naming a chart the cell never draws",
        HEATMAP_CELL.format(alt=GROUPED_BARS_ALT),
        1,
        ("claims 'Grouped bars'", "no bar"),
    ),
    (
        "alt text naming the chart the cell does draw",
        HEATMAP_CELL.format(alt=HEATMAP_ALT),
        0,
        (),
    ),
    (
        "bars described in groups, and the height of a horizontal bar, over one `barh` call",
        BARH_CELL + GROUPED_PROSE,
        2,
        ("describes bars in groups", "its height"),
    ),
    (
        "the corrected prose for that same cell reports neither",
        BARH_CELL + DESCRIPTIVE_PROSE,
        0,
        (),
    ),
    (
        "Plotly Express groups within one call, so its grouped-bars alt text is honest",
        "# %%\n"
        "fig = px.bar(weights_long, x='symbol', y='Weight', color='Portfolio', barmode='group')\n"
        "show_with_alt(fig, 'Grouped bars of portfolio weight per ETF for the five "
        "allocations.')\n",
        0,
        (),
    ),
    (
        "a stacked Matplotlib bar is not one series",
        "# %%\n"
        "ax.bar(x, lower, color=COLORS['blue'])\n"
        "ax.bar(x, upper, bottom=lower, color=COLORS['amber'])\n"
        "show_with_alt(fig, 'Grouped bars with one group per sector.')\n",
        0,
        (),
    ),
    (
        "markers really do sit at offset heights where a figure is not only bars",
        "# %%\n"
        "axes[0].errorbar(naive, rows, xerr=naive_se, fmt='o')\n"
        "axes[1].barh(rows, counts, color=COLORS['blue'])\n"
        "show_with_alt(fig, 'The left panel plots each factor twice at slightly offset "
        "heights; the right panel is a horizontal bar chart of the control counts.')\n",
        0,
        (),
    ),
    (
        "a sentence pointing at another figure is not a claim about this one",
        "# %%\n"
        "ax.barh(rows, values)\n"
        "show_with_alt(fig, 'One bar per case study. The heatmap below reports every "
        "allocator per case study.')\n",
        0,
        (),
    ),
    (
        "markdown naming another figure's chart type is not held to this cell",
        "# %% [markdown]\n"
        "# The bar chart mixes three estimators. Tracing IC across the Ridge penalty alone\n"
        "# isolates the effect of shrinkage.\n"
        "# %%\n"
        "fig_alpha.add_trace(go.Scatter(x=alphas, y=ics, mode='lines'))\n"
        "show_with_alt(fig_alpha, 'One line per label over the Ridge penalty.')\n",
        0,
        (),
    ),
    (
        "an error-bar chart is not a bar chart claim",
        "# %%\n"
        "ax.errorbar(features, ic_mean, yerr=ic_std, fmt='o')\n"
        "show_with_alt(fig, 'An error-bar chart with the eight selected features along the "
        "horizontal axis.')\n",
        0,
        (),
    ),
    (
        "bars satisfy a histogram claim, because a discrete histogram is drawn with bars",
        "# %%\n"
        "fig.add_trace(go.Bar(x=counts['bar_count'], y=counts['len']))\n"
        "show_with_alt(fig, 'A histogram of how many hourly bars each daily session "
        "contains.')\n",
        0,
        (),
    ),
    (
        "a figure built across cells is read back to its constructor",
        "# %%\n"
        "fig = make_subplots(rows=1, cols=2)\n"
        "fig.add_trace(go.Bar(x=ic, y=names, orientation='h'), row=1, col=1)\n"
        "# %%\n"
        "fig.add_trace(go.Scatter(x=naive, y=hac, mode='markers'), row=1, col=2)\n"
        "show_with_alt(fig, 'Two panels. On the left, a horizontal bar chart of the average "
        "daily rank IC. On the right, a scatter plot of the two t-statistics.')\n",
        0,
        (),
    ),
    (
        "a cell that draws nothing does not borrow the previous figure's kinds",
        "# %%\n"
        "ax.plot(dates, values)\n"
        "# %%\n"
        "fig = plot_characteristic_coverage(panel)\n"
        "show_with_alt(fig, 'Histogram of the number of characteristics reported per firm.')\n",
        0,
        (),
    ),
]


def selftest() -> int:
    """Every rule must be shown firing AND not firing. A check with only no-hit evidence
    cannot fail, which is evidence of nothing."""
    failures = 0
    for label, src, expected, must_say in CASES:
        got = check_source(Path("<selftest>"), src)
        missing = [phrase for phrase in must_say if not any(phrase in g for g in got)]
        if len(got) != expected or missing:
            failures += 1
            if len(got) != expected:
                print(f"FAIL {label}: expected {expected} finding(s), got {len(got)}")
            for phrase in missing:
                print(f"FAIL {label}: no finding says {phrase!r}")
            for problem in got:
                print(f"      {problem}")
        else:
            print(f"ok   {label}")
    fired = sum(1 for _, _, n, _ in CASES if n)
    clean = len(CASES) - fired
    print(f"\n{len(CASES)} cases: {fired} that must fire, {clean} that must not, {failures} failed")
    return 1 if failures else 0


def notebook_sources(roots: list[Path]) -> list[Path]:
    """Every paired notebook source, and nothing else.

    A notebook source is a `.py` with a committed `.ipynb` beside it. A named file is taken
    as given, so a scratch copy can still be checked directly.

    The `# %%` marker alone is not enough, and the difference is not cosmetic. Ten files
    under `tests/` carry notebook source as string fixtures, three of them alt text written
    to be wrong so a test can catch it, and a marker filter hands those to this checker as
    though they were notebooks - a fixture added later would red this gate for describing a
    figure incorrectly on purpose. Measured over this repository: 495 files match the marker,
    484 have a paired `.ipynb`, and the 11 that do not are ten tests and `sync_notebooks.py`.
    Not one notebook is lost, including the three that carry an `.ipynb` and no jupytext
    header, which is why the pair and not the header is the test.

    Dot-directories are skipped: `.venv` alone is tens of thousands of files and none of
    them is a notebook, and pointed at a checkout that has one, an earlier checker reported
    title violations from arviz, torch and econml.
    """
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part.startswith(".") for part in path.parts):
                continue
            if path.with_suffix(".ipynb").exists():
                out.append(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "paths", nargs="*", type=Path, help="files or directories (default: the repo root)"
    )
    ap.add_argument(
        "--selftest", action="store_true", help="prove the rule fires and does not fire"
    )
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    repo = Path(__file__).resolve().parents[2]
    roots = args.paths or [repo]
    problems: list[str] = []
    for path in notebook_sources(roots):
        try:
            found = check(path)
        except OSError as exc:  # an unreadable file is not a clean one
            print(f"{path}: cannot read: {exc}", file=sys.stderr)
            return 2
        for problem in found:
            try:  # a path under the repo reads better relative; one outside it stays absolute
                problem = problem.replace(str(path), str(path.resolve().relative_to(repo)), 1)
            except ValueError:
                pass
            problems.append(problem)

    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} figure description(s) that do not describe the figure drawn")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
