"""Tail p-values must stay positive and finite for extreme test statistics.

A two-tailed p-value written as ``2 * (1 - dist.cdf(abs(stat)))`` returns exactly
``0.0`` once ``abs(stat)`` passes roughly 8.35, because ``cdf`` rounds to ``1.0``
long before the tail mass underflows: the subtraction cancels every remaining
significant digit. It is already wrong by 60% at 8.3. The survival function
``dist.sf`` evaluates the tail directly and stays accurate to the smallest normal
double::

    |t| = 6.00   2 * (1 - cdf) = 2.138e-09   2 * sf = 2.138e-09
    |t| = 8.30   2 * (1 - cdf) = 2.220e-16   2 * sf = 1.385e-16
    |t| = 8.94   2 * (1 - cdf) = 0           2 * sf = 5.702e-19

A notebook that teaches inference must not print ``p=0`` for a value no
computation produced, and ``p_value_hac`` must not be stored as ``0.0``.

Both halves run in the always-on ``test-unit`` job, which installs scipy,
statsmodels and scikit-learn for the numeric case. The ``importorskip`` calls
keep the static scan usable from a stdlib-only environment; they are not the CI
path, and a skip there would mean the modelling stack went missing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Chapter notebooks, case studies, and shared helpers. Tests are excluded: the
# baseline assertion below deliberately writes the broken form.
SCANNED_GLOBS = ("[0-9][0-9]_*/*.py", "case_studies/**/*.py", "utils/**/*.py")

# The backlog is empty, and the two assertions below are what keep it that way.
#
# It was 11 occurrences across 8 paired teaching notebooks, deferred because fixing
# the `.py` makes the committed `.ipynb` stale and the provenance gate
# (`.github/scripts/notebook_provenance.py`) then requires a production re-run.
# Chapters 16 and 17 cleared their five in their own review passes; this change
# clears the remaining six in Chapters 7, 9 and 15.
#
# The deferral was dischargeable rather than someone else's work: three of those
# four notebooks were already stamped `executor: local-uv`, so this environment IS
# the canonical executor. All four were fixed, re-executed unparameterized from the
# repo root, and re-stamped.
#
# A row goes back in here only to record a defect that cannot be fixed in the same
# change, and it must come with the reason. An empty dict with a zero ceiling means a
# new occurrence anywhere in SCANNED_GLOBS fails immediately.
PENDING: dict[str, int] = {}

# The ceiling on the whole backlog, and the thing that makes "only shrinks" a
# rule a test can enforce rather than a sentence in a comment.
#
# Without it, a genuinely new occurrence in a file that already has a row can be
# absorbed by raising that row's count: `test_no_new_...` reads the count, so the
# edit that admits the defect is the edit that hides it. Only the ceiling makes
# the admission visible, because raising any row without lowering another fails
# here.
#
# Lower it whenever a fix lands. Never raise it.
PENDING_CEILING = 0


def _is_cdf_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cdf"
    )


def _is_literal_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and not isinstance(node.value, bool) and node.value == 1


def _cdf_bound_names(tree: ast.AST) -> set[str]:
    """Names assigned an expression that calls ``.cdf`` anywhere inside it."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value, targets = node.value, [node.target]
        else:
            continue
        if any(_is_cdf_call(n) for n in ast.walk(value)):
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _hits(path: Path, rel: str) -> list[str]:
    """Both spellings of the cancellation, in one file.

    Parsed rather than grepped, for two reasons. Comments and docstrings are
    invisible to ``ast``, so prose naming the pattern - including a comment
    explaining why a nearby line uses ``sf`` - does not trip it. And an
    assignment spread over several lines is one node, so
    ``probability = float(`` with the ``norm.cdf(...)`` on the next line is
    still a CDF binding.

    Known limit: a CDF value reached through a subscript or attribute rather
    than a bare name (``1 - result["psr"]``) is not tracked.
    """
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bound = _cdf_bound_names(tree)
    lines = source.splitlines()

    hits = {
        node.lineno: f"{rel}:{node.lineno}: {lines[node.lineno - 1].strip()}"
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and _is_literal_one(node.left)
        and (
            _is_cdf_call(node.right)
            or (isinstance(node.right, ast.Name) and node.right.id in bound)
        )
    }
    return [hits[line] for line in sorted(hits)]


def _occurrences() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for glob in SCANNED_GLOBS:
        for path in sorted(REPO_ROOT.glob(glob)):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if hits := _hits(path, rel):
                found[rel] = hits
    return found


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("inline", "p = 2 * (1 - stats.t.cdf(abs(t), df=n))\n"),
        ("bound name", "probability = norm.cdf(z)\np_value = 1 - probability\n"),
        (
            "binding wrapped over lines",
            "probability = float(\n    stats.norm.cdf(z_score)\n)\np_value = float(1.0 - probability)\n",
        ),
        (
            "subtraction wrapped over lines",
            "probability = stats.norm.cdf(z)\np_value = (\n    1\n    - probability\n)\n",
        ),
    ],
)
def test_detector_finds_every_layout(tmp_path: Path, label: str, source: str):
    """Every way of writing it, including the ones no regex sees."""
    path = tmp_path / "sample.py"
    path.write_text(source)

    assert _hits(path, "sample.py"), f"missed the {label} layout"


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("sf", "p_value = 2 * stats.t.sf(abs(t), df=n)\n"),
        (
            "a comment naming the pattern",
            "# sf, not 1 - norm.cdf(z), which cancels\np = norm.sf(z)\n",
        ),
        ("a CDF used as a probability", "psr = float(norm.cdf(z))\nis_significant = psr >= 0.95\n"),
        ("an unrelated subtraction", "share = 1 - weight\n"),
    ],
)
def test_detector_does_not_fire_on_correct_code(tmp_path: Path, label: str, source: str):
    path = tmp_path / "sample.py"
    path.write_text(source)

    assert not _hits(path, "sample.py"), f"false positive on {label}"


def test_no_new_tail_probability_is_written_as_one_minus_cdf():
    """Static guard: the cancellation cannot appear in a file that is clean today."""
    violations = [
        hit for rel, hits in _occurrences().items() for hit in hits[PENDING.get(rel, 0) :]
    ]

    assert not violations, "use dist.sf(x), not 1 - dist.cdf(x):\n" + "\n".join(violations)


def test_the_backlog_never_grows():
    """A raised row must be paid for by a lowered one, or the ceiling comes down.

    `test_no_new_...` reads PENDING per file, so raising a row is enough to make a
    new occurrence pass it. This is the assertion that notices.
    """
    total = sum(PENDING.values())

    assert total <= PENDING_CEILING, (
        f"the `1 - cdf` backlog grew to {total} against a ceiling of {PENDING_CEILING}. "
        "A row goes up only when another goes down; a genuinely new occurrence is fixed, "
        "never admitted."
    )
    assert total == PENDING_CEILING, (
        f"the backlog is down to {total}: lower PENDING_CEILING to match, so the ground "
        "regained cannot be given back"
    )


def test_pending_baseline_has_no_stale_rows():
    """A row that no longer matches must be deleted, so the baseline only shrinks."""
    found = _occurrences()
    stale = [
        f"{rel}: baseline expects {count}, found {len(found.get(rel, []))}"
        for rel, count in PENDING.items()
        if len(found.get(rel, [])) != count
    ]

    assert not stale, "PENDING is out of date - fix the file and drop the row:\n" + "\n".join(stale)


def test_scipy_baseline_confirms_the_cancellation():
    """The premise: `1 - cdf` is exactly zero where `sf` is not."""
    stats = pytest.importorskip("scipy.stats")

    assert 2 * (1 - stats.t.cdf(8.94, df=4231)) == 0.0
    assert 2 * stats.t.sf(8.94, df=4231) > 0.0


def test_dml_hac_pvalue_survives_extreme_t():
    """`manual_dml_timeseries` writes `p_value_hac` into the registry.

    The treatment effect here is estimated almost without noise, so the HAC
    t-statistic lands around 20 - past the point where `1 - cdf` cancels, well
    short of where the true tail mass underflows.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("statsmodels")
    pytest.importorskip("sklearn")
    from case_studies.utils.causal import manual_dml_timeseries

    rng = np.random.default_rng(7)
    n = 600
    x = rng.normal(size=(n, 2))
    t = x[:, 0] * 0.5 + rng.normal(scale=0.5, size=n)
    y = x[:, 1] * 0.3 + 0.12 * t + rng.normal(scale=1e-4, size=n)

    result = manual_dml_timeseries(y, t, x, n_folds=5, embargo=5)

    assert abs(result["t_stat_hac"]) > 8.35, "fixture must reach the underflow zone"
    assert result["p_value_hac"] > 0.0, "p-value underflowed to exactly zero"
    assert np.isfinite(result["p_value_hac"])
