"""A positional read of an unordered `group_by` is a seed that fixes nothing.

Three defects found on one day in 2026-08 shared a shape: something pinned one
dimension of randomness and left ordering free, and in every case the author had
taken a visible precaution - a seed, a sort - which is why none was caught by
reading.

    hmmlearn GaussianHMM.fit    random_state pinned; scikit-learn's k-means split
                                its work across threads, and float addition is not
                                associative, so the initial means moved between runs
                                that printed identical numbers
    _bootstrap_median_interval  seed=42 pinned; `rng.choice(values, ...)` indexes
                                POSITIONALLY, so a reversed input drew differently
                                and F6's ribbon changed on every re-run
    perturb_clock_and_aggregation  `.sort("session")` pinned; the frame came from an
                                unordered `group_by` and 45 ties on one session broke
                                arbitrarily, so the committed `.out` did not reproduce

That class matters more than any of its instances. REVIEW.md step 5a - re-run your
own evidence against the fixed code, and a `.out` that disagrees with the script
beside it is a REFUTE - is the backbone of verification here, and **a capture that
cannot reproduce itself is indistinguishable from one that was fabricated**. So this
defect does not merely produce wrong numbers; it consumes the mechanism that would
detect wrong numbers. It has already cost real work: one instance's magnitude was
quoted wrongly in the issue *and* in a review sheet, because the script measuring it
was itself unpinned.

All three are fixed. This is the sweep the issue asked for and could not assume:
"Whether there are more" was explicitly unestablished.

## What this detects, and what it deliberately does not

Polars does not guarantee the row order of a `group_by`, so a *positional* read of one
is unpinned. Two shapes, and only one of them is a defect:

    df.sort(k).group_by(g).first()      picks WITHIN each group. The rows were ordered
                                        before the grouping, so the pick is defined.
                                        The order of the groups is still unspecified,
                                        but nothing has read it.

    df.group_by(g).agg(...).head(3)     reads the GROUP order, which no inner sort
                                        fixes. This is the defect.

Sorting between the grouping and the read fixes the second, and so does
`maintain_order=True`. A `select` in between whose every expression is an aggregate
collapses the frame to one row, where there is only one order - an `agg` does not,
because after a `group_by` it produces one row per group.

It is a syntactic check over method chains, so it sees neither a chain split across
statements nor an order dependence that arrives through a variable. It is not the
whole class - `rng.choice` on an unordered frame and a `.fit()` whose library
parallelises its initialiser are the other two arms, and neither is visible here.
What it does cover is the arm that is mechanically decidable, and it covers it with
no false positives on the tree it was written against.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXCLUDE_PARTS = {".venv", "__pycache__", "archive", "_archive", ".git", "node_modules"}

# Picks a row from WITHIN each group: decided by the frame's order before the grouping.
PER_GROUP = frozenset({"first", "last", "head", "tail"})
# Reads the frame positionally: decided by the order of the groups themselves.
WHOLE_FRAME = frozenset({"row", "item", "to_list", "to_numpy", "to_series"})
POSITIONAL = PER_GROUP | WHOLE_FRAME
ORDERING = frozenset({"sort", "sort_by", "top_k", "bottom_k"})
# Expressions that reduce a column to one value, so a `select` built only from them
# leaves a single row. `first`/`last` are here as aggregates, which is a different use
# from the frame methods above.
REDUCERS = frozenset(
    {
        "median", "mean", "sum", "max", "min", "len", "n_unique", "std", "var",
        "quantile", "first", "last", "count", "alias", "col", "cast",
    }
)  # fmt: skip


def _chain(node: ast.AST) -> tuple[list[str], list[ast.Call]]:
    """Method names and call nodes of `a.b().c().d()`, innermost first."""
    calls: list[ast.Call] = []
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        calls.append(node)
        node = node.func.value
    calls.reverse()
    return [c.func.attr for c in calls], calls  # type: ignore[union-attr]


def _collapses_to_one_row(call: ast.Call) -> bool:
    """Whether this link leaves a single row, making a positional read of it safe.

    Only the call's ARGUMENTS are inspected. Walking the node itself descends into its
    receiver, so every link would inherit the names of everything before it and a
    `select` would be judged by the chain that produced its input.
    """
    # `select` only. An `agg` after a `group_by` produces one row PER GROUP, so its
    # aggregates say nothing about how many rows the read will see.
    if call.func.attr != "select":  # type: ignore[union-attr]
        return False
    args = [*call.args, *(k.value for k in call.keywords)]
    names = [
        inner.func.attr
        for arg in args
        for inner in ast.walk(arg)
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
    ]
    return bool(names) and all(name in REDUCERS for name in names)


def unordered_read(call: ast.Call) -> str | None:
    """Why this chain reads an unordered `group_by` positionally, or None."""
    names, calls = _chain(call)
    if not names or names[-1] not in POSITIONAL or "group_by" not in names:
        return None
    group = len(names) - 1 - names[::-1].index("group_by")
    read = len(names) - 1
    if any(kw.arg == "maintain_order" for kw in calls[group].keywords):
        return None
    if names[read] in PER_GROUP and read == group + 1:
        if any(name in ORDERING for name in names[:group]):
            return None
        return f".group_by(...).{names[read]}() over rows nothing ordered"
    if any(name in ORDERING for name in names[group + 1 : read]):
        return None
    if any(_collapses_to_one_row(link) for link in calls[group + 1 : read]):
        return None
    return f".{names[read]}() reads the group order, which no sort has fixed"


def _findings(source: str, label: str = "<source>") -> list[str]:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return []
    # Only the outermost call of a chain is the one whose result is used. Judging an
    # inner link on its own reports `.group_by(g).tail(1).sort(k)` as unordered,
    # because the sort is outside the node being looked at.
    inner = {
        id(node.func.value)
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    lines = source.splitlines()
    out = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or id(node) in inner:
            continue
        if why := unordered_read(node):
            out.append(f"{label}:{node.lineno}: {why} — {lines[node.lineno - 1].strip()[:80]}")
    return out


def _scanned_files() -> list[Path]:
    trees = ["case_studies", "utils"] + [
        p.name for p in REPO_ROOT.iterdir() if p.is_dir() and re.match(r"^\d\d_", p.name)
    ]
    return [
        py
        for tree in trees
        for py in sorted((REPO_ROOT / tree).rglob("*.py"))
        if EXCLUDE_PARTS.isdisjoint(py.parts)
    ]


# -----------------------------------------------------------------------------
# A hit fixture and a no-hit fixture per shape, which is what makes the sweep
# above a measurement rather than an assertion.
# -----------------------------------------------------------------------------

HITS = [
    ("group order read positionally", "top = df.group_by('k').agg(pl.col('v').sum()).head(3)\n"),
    ("row(0) on many groups", "first = df.group_by('k').agg(pl.col('v').sum()).row(0)\n"),
    ("to_list of an unordered grouping", "ks = df.group_by('k').len().to_list()\n"),
    ("per-group pick over unsorted rows", "one = df.group_by('k').first()\n"),
    ("head per group over unsorted rows", "five = df.group_by('k').head(5)\n"),
]

NO_HITS = [
    ("sorted before grouping", "one = df.sort('v').group_by('k').first()\n"),
    ("maintain_order", "top = df.group_by('k', maintain_order=True).agg(pl.len()).head(3)\n"),
    ("sorted after grouping", "top = df.group_by('k').agg(pl.len()).sort('k').head(3)\n"),
    (
        "collapsed to one row",
        "v = df.group_by('k').agg(pl.col('a').n_unique()).select(pl.col('a').median()).row(0)\n",
    ),
    ("an aggregate expression, not a frame method", "e = pl.col('v').sort_by('t').first()\n"),
    (
        "a namespace method that is not a frame read",
        "b = pl.col('s').str.split('_').list.first()\n",
    ),
    ("no grouping at all", "top = df.sort('v').head(3)\n"),
    ("a plain aggregation nobody reads positionally", "g = df.group_by('k').agg(pl.len())\n"),
]


@pytest.mark.parametrize(("label", "source"), HITS, ids=[h[0] for h in HITS])
def test_the_detector_sees_each_unordered_read(label: str, source: str):
    assert _findings(source), f"missed: {label}"


@pytest.mark.parametrize(("label", "source"), NO_HITS, ids=[n[0] for n in NO_HITS])
def test_the_detector_does_not_fire_on_ordered_code(label: str, source: str):
    assert not _findings(source), f"false positive on: {label}"


def test_an_inner_link_is_not_judged_without_its_chain():
    """`.group_by(g).tail(1).sort(k)` is ordered, and the sort is outside the read.

    Every link of a chain is a Call, so a scan that judges each one reports the
    `.tail(1)` here as unordered. Four of the eight candidates in the first pass over
    the tree were this, and taking them at face value would have meant editing correct
    code.
    """
    assert not _findings("snap = panel.group_by('symbol').tail(1).sort('symbol')\n")


# -----------------------------------------------------------------------------
# The sweep
# -----------------------------------------------------------------------------


def test_no_positional_read_of_an_unordered_group_by():
    """The tree is clean, and stays clean.

    There is no pending baseline because the sweep found nothing to carry. The first
    text-based pass reported 97 candidates and the first AST pass 39; both were
    dominated by chains that sort before grouping, which is correct code. Every one of
    the four that survived to adjudication was also correct - a per-group pick after a
    full sort, or a `select` of aggregates that leaves one row - and the rules that
    forgive them are the ones the fixtures above pin.
    """
    findings = [
        line
        for path in _scanned_files()
        for line in _findings(
            path.read_text(encoding="utf-8"), path.relative_to(REPO_ROOT).as_posix()
        )
    ]

    assert not findings, (
        "a positional read of a group_by whose order nothing fixed. Polars does not "
        "guarantee group order, so this is a result that changes between runs on "
        "byte-identical input - and a capture that cannot reproduce itself is "
        "indistinguishable from one that was fabricated. Sort before the read, or pass "
        "maintain_order=True:\n  " + "\n  ".join(findings)
    )


def test_the_sweep_actually_reads_the_corpus():
    """A scan that silently found no files would pass the assertion above."""
    files = _scanned_files()

    assert len(files) > 200, f"only {len(files)} files scanned; the globs are wrong"
    assert any("case_studies" in f.parts for f in files)
