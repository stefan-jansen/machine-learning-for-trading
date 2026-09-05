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

Polars does not guarantee the row order of a `group_by`. Two things can be read out of
one positionally, and they are decided in different places:

    df.group_by(g).agg(...).head(3)   reads the order of the GROUPS. Polars establishes
                                      that nowhere, so no upstream code can have fixed
                                      it and the chain alone settles the question. This
                                      is what the check reports.

    df.group_by(g).tail(1)            picks WITHIN each group, so it is decided by the
                                      order of the frame that was grouped - which is
                                      routinely established somewhere else entirely.
                                      `26_mlops_governance/05_feast_feature_store.py:308`
                                      groups a `panel` whose sort happens three
                                      functions away. Flagging that from the chain would
                                      be a guess, so this arm is out of scope.

Scoping to the first is what makes the check sound rather than heuristic, and a
soundness a static check does not have is a check people learn to override.

Only a sort BETWEEN the grouping and the read counts: `.head(3).sort(k)` takes three
arbitrary groups and then tidies them. `maintain_order=True` counts. So does a `select`
in between whose EVERY expression contains an aggregate, which leaves one row - an
`agg` does not, because after a grouping it produces one row per group, and a bare
`select(pl.col('k'))` reduces nothing at all.

It is syntactic, so it sees neither a chain split across statements nor an order
dependence arriving through a variable. And it is one arm of three: `rng.choice` over a
frame whose order is not established, and a `.fit()` whose library parallelises its
initialiser, are the others, and neither is visible to a check like this.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXCLUDE_PARTS = {".venv", "__pycache__", "archive", "_archive", ".git", "node_modules"}

# Reads that depend on the order of the GROUPS. `head`/`tail`/`first`/`last` applied to
# the RESULT of a grouping are here too: they take the first n groups, not the first n
# rows of each.
GROUP_ORDER_READS = frozenset(
    {"row", "item", "to_list", "to_numpy", "to_series", "head", "tail", "first", "last"}
)
# The same four names applied DIRECTLY to a group_by pick within each group instead,
# which is a different question this check does not answer. See `unordered_read`.
PER_GROUP_PICKS = frozenset({"head", "tail", "first", "last"})
ORDERING = frozenset({"sort", "sort_by", "top_k", "bottom_k"})
# Aggregates: each reduces a column to one value.
AGGREGATES = frozenset(
    {
        "median", "mean", "sum", "max", "min", "len", "n_unique", "std", "var",
        "quantile", "first", "last", "count",
    }
)  # fmt: skip
# Allowed to appear alongside an aggregate without adding rows. On their own they are a
# projection, which keeps every row - so a `select` needs an aggregate in each of its
# expressions, not merely somewhere among them.
WRAPPERS = frozenset({"alias", "col", "cast", "round", "fill_null", "fill_nan"})


def _chain(node: ast.AST) -> tuple[list[str], list[ast.Call]]:
    """Method names and call nodes of `a.b().c().d()`, innermost first."""
    calls: list[ast.Call] = []
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        calls.append(node)
        node = node.func.value
    calls.reverse()
    return [c.func.attr for c in calls], calls  # type: ignore[union-attr]


def _expression_names(node: ast.AST) -> list[str]:
    return [
        inner.func.attr
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
    ]


def _collapses_to_one_row(call: ast.Call) -> bool:
    """Whether this link leaves a single row, making a positional read of it safe.

    `select` only. An `agg` after a `group_by` produces one row PER GROUP, so its
    aggregates say nothing about how many rows the read will see.

    EVERY expression must contain an aggregate, not merely one of them:
    `select(pl.col('a').median(), pl.col('b'))` keeps a row per value of `b`. And
    checking only that the names are "allowed" accepts a plain projection -
    `select(pl.col('k'))` is `col` alone, which reduces nothing.

    Only the call's ARGUMENTS are inspected. Walking the node itself descends into its
    receiver, so every link would inherit the names of everything before it and a
    `select` would be judged by the chain that produced its input.
    """
    if call.func.attr != "select":  # type: ignore[union-attr]
        return False
    args = [*call.args, *(k.value for k in call.keywords)]
    if not args:
        return False
    for arg in args:
        # A generator or list of expressions is one argument holding many.
        parts = arg.elts if isinstance(arg, ast.List | ast.Tuple) else [arg]
        for part in parts:
            names = _expression_names(part)
            if not names or not any(n in AGGREGATES for n in names):
                return False
            if any(n not in AGGREGATES and n not in WRAPPERS for n in names):
                return False
    return True


def unordered_read(call: ast.Call) -> str | None:
    """Why this chain reads the order of a `group_by` that nothing has fixed, or None."""
    names, calls = _chain(call)
    if not names or names[-1] not in GROUP_ORDER_READS or "group_by" not in names[:-1]:
        return None
    group = len(names) - 2 - names[-2::-1].index("group_by")
    read = len(names) - 1
    if any(kw.arg == "maintain_order" for kw in calls[group].keywords):
        return None
    if read == group + 1 and names[read] in PER_GROUP_PICKS:
        # `.group_by(g).tail(1)` picks WITHIN each group, so it is decided by the order
        # of the frame that was grouped - which is routinely established outside this
        # chain. `26_mlops_governance/05_feast_feature_store.py:308` groups a `panel`
        # whose sort happens three functions away, inside `load_model_vintage`. Flagging
        # it here would be a false positive with nothing in the chain to disprove it, so
        # this arm is out of scope rather than guessed at.
        return None
    # Only a sort BETWEEN the grouping and the read counts. Sorting afterwards orders
    # the arbitrary subset that was already taken, which is the miss this check had:
    # `.group_by(k).agg(pl.len()).head(3).sort(k)` returns three arbitrary groups in a
    # tidy order.
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
    # Every link is judged, not only the outermost. A read is decided where it happens:
    # `.group_by(k).agg(pl.len()).head(3).sort(k)` takes three arbitrary groups and then
    # orders that arbitrary subset, and looking only at the outermost `.sort` misses it.
    lines = source.splitlines()
    out = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
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
    (
        "sorted only after the arbitrary subset was taken",
        "top = df.group_by('k').agg(pl.len()).head(3).sort('k')\n",
    ),
    (
        "a projection is not a reduction",
        "top = df.group_by('k').agg(pl.len()).select(pl.col('k')).head(3)\n",
    ),
    (
        "one aggregate does not excuse the other expression",
        "v = df.group_by('k').agg(pl.len()).select(pl.col('a').median(), pl.col('b')).row(0)\n",
    ),
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
    ("a per-group pick, which this check does not judge", "one = df.group_by('k').first()\n"),
    ("a per-group pick then presentation order", "s = df.group_by('k').tail(1).sort('k')\n"),
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


def test_a_sort_after_the_read_does_not_rescue_it():
    """Ordering an arbitrary subset leaves it arbitrary.

    Judging only the outermost call of a chain misses this, because the outermost call
    is the `.sort` and it looks like ordering. The three groups were already chosen.
    """
    findings = _findings("top = df.group_by('k').agg(pl.len()).head(3).sort('k')\n")

    assert findings and "group order" in findings[0]


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
