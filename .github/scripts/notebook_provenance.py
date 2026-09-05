"""Provenance stamp + sync gate for paired ``.py``/``.ipynb`` notebooks.

A committed ``.ipynb`` should be the *current* ``.py`` executed in a real
environment with production parameters — not a stale render, not a TEST-mode run,
not a run in an environment missing a dependency (e.g. CUDA-LightGBM). This module
stamps that fact into the notebook and provides a gate that rejects violations, so
"edited the ``.py``, ran TEST or the wrong env, committed a stale ``.ipynb``" is
caught mechanically instead of by review.

The stamp lives in ``nb.metadata["ml4t_provenance"]``::

    source_py_blob : git blob hash of the paired .py at execution time
    outputs_digest : digest over the outputs the run left in the notebook
    library_digest : digest over the repository code the paired .py imports
    executed_at    : ISO-8601 timestamp
    executor       : environment label (e.g. "ml4t-gpu", "local-uv")
    production     : bool — True iff overrides preserve the full production surface
    parameters     : the papermill parameter overrides, as declared by the executor
    notes          : optional free text (e.g. "GPU libs: xgboost,lightgbm,catboost")

Where ``parameters`` comes from, and why it is not read out of the notebook
----------------------------------------------------------------------------

``metadata.papermill.parameters`` cannot answer "was *this* run parameterized".
Two mechanisms make it a fossil, both reproducible:

1. Papermill does not clear it. Re-executing with no overrides leaves the previous
   run's dict in place next to a freshly written ``start_time``, so the timestamps
   describe the new run and the parameters describe an older one.
2. ``jupytext --sync`` rebuilds the cell list from the ``.py`` and so removes the
   ``injected-parameters`` cell, but notebook-level ``metadata`` survives
   untouched. The marker that belongs to the run is deleted and the stale one is
   kept.

So the executor states the parameters and the stamp records that statement:
``--production`` for an unparameterized run, or ``--parameters '<json>'``. One of
the two is required; this tool never infers.

That choice is about what the run carried, not about what tier it was. ``production``
is computed from the parameters, not from their absence: a run whose every override is
in ``PRODUCTION_SAFE_PARAMETERS`` stamps as production too. A canonical run that must
carry an override - ``SUPERSEDES_POPULATION`` on a re-run into a changed population -
declares it with ``--parameters`` and is still production. ``--production`` is
shorthand for the empty set, not a claim the tool would otherwise have to weigh.

Where the notebook *does* carry evidence of its own execution — papermill's
``injected-parameters`` cell, which lives in the cell list and is rewritten by
every parameterized run — ``stamp`` cross-checks the declaration against it and
refuses to write a stamp that contradicts it. Stamping also rewrites
``metadata.papermill.parameters`` to the declared set, so the fossil cannot
outlive the stamp and disagree with it later.

What the three digests each answer, because they are easy to confuse
--------------------------------------------------------------------

``source_py_blob`` answers "has the paired ``.py`` changed since somebody stamped
this". For a long time that was the whole gate, and it left two ways for a
superseded result to reach ``main`` with every check green.

``outputs_digest`` answers "are these the outputs that run produced". A notebook
whose ``.py`` is untouched passed the staleness check carrying outputs from any
earlier run. Computed over the parsed structure with every ``execution_count`` and
every figure ``alt`` removed, so re-running to the same values, reformatting the
JSON, folding in a prose edit and correcting alt text do not move it - the last
because ``alt_text_only_drift`` already forgives that edit, on the proven grounds
that re-executing cannot change the image.

The digest is written from whatever is on disk at stamp time, so it cannot by
itself catch a run that never wrote the file - ``nbconvert --execute --inplace``
under ``nohup ... &`` exits 0 and does exactly that, and the resulting stamp agrees
with itself forever. ``unwritten_run`` catches it at the only moment the state is
still visible: the ``.py`` moved and the outputs are byte-identical to what the
previous stamp recorded. ``--allow-unchanged-outputs`` answers it for a notebook
genuinely deterministic enough that the change altered nothing it prints.

``library_digest`` answers "did this run use the code that is here now". The
numbers in a case study are computed in ``case_studies/utils/*.py``, which
``source_py_blob`` never covered: #606 changed ``causal.py`` under all nine case
studies at once and the committed stamps went on claiming runs whose outputs
described code no longer in the tree. It digests the repository files the paired
``.py`` imports transitively, keyed by path as well as content.

Gate (``check``): for every tracked ``.ipynb`` that HAS a stamp,

* ``source_py_blob`` must equal ``git hash-object`` of the current paired ``.py``
  (else the ``.py`` changed since the notebook was executed — STALE),
* ``outputs_digest``, where the stamp records one, must match the notebook's
  current outputs (else the stored results are not the ones the stamped run
  produced — OUTPUTS CHANGED),
* ``library_digest``, where the stamp records one, is compared and *reported*
  rather than failed (LIBRARY DRIFT). Failing it would block at least five
  notebooks across three case studies on the day it lands, and that is a separate
  decision from being able to see the drift at all,
* a stamp carrying neither digest predates them both and is counted, not failed:
  every stamp written before they existed lacks them, and none of those notebooks
  has the defect the fields exist to catch,
* ``production`` must be True (else a TEST-mode run was committed),
* some code cell must show it ran - an output or an execution count (else the stamp
  is over a render nothing produced — HOLLOW), and
* the stamp must not contradict a committed ``injected-parameters`` cell (else the
  notebook was re-executed with overrides after it was stamped).

Two states are legal at commit time, and the second is what keeps the workflow
linear:

1. **Stamped and in sync** — the notebook is its current ``.py``, executed in
   production. The checks above apply.
2. **Cleared** — no stamp and no outputs. It makes no claim about a run, so there
   is nothing to be stale. ``clear`` puts a notebook in this state. Edit the
   ``.py``, ``jupytext --sync``, ``clear``, commit; execute later and the stamp and
   the checks come back. Without this the gate has no legal path for a correction
   to an already-executed notebook, because clearing the stale stamp read as
   DE-STAMPED and keeping it read as STALE.

The state the gate must still reject is the one that looks like (2) but claims (1):
a stamp over a notebook showing no trace of execution — a render rebuilt from the
``.py`` and re-stamped without a run behind it. That is HOLLOW and it fails.

Notebooks WITHOUT a stamp are reported as "unverified" but do not fail unless
``--strict`` is passed. This is deliberate: adoption is gradual — stamp notebooks
as they are re-run through the canonical path, and the gate enforces only where
provenance exists. Flip to ``--strict`` once the backfill is complete.

**Where this gate runs.** In two places, asking two different questions. The pre-commit hook
runs it over the *staged* files, so a notebook another session left dirty does not block an
unrelated commit. CI runs it over the *change*: ``check --since <base>`` on a pull request,
over the notebooks that pull request touches, and on a push to ``main`` against the previous
tip. A stale render only reaches a reader through a merge, and until CI ran this the gate
never covered a merge at all - it fired on every local commit and on nothing that publishes.

Usage::

    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --production
    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --parameters '{"MAX_SYMBOLS": 5}'
    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --production --notes "..."
    uv run python .github/scripts/notebook_provenance.py clear <nb.ipynb>  # commit it unexecuted
    uv run python .github/scripts/notebook_provenance.py check          # gate (stamped-only)
    uv run python .github/scripts/notebook_provenance.py check --strict  # also fail on unverified
    uv run python .github/scripts/notebook_provenance.py check --since origin/main  # this branch only
    uv run python .github/scripts/notebook_provenance.py check --since <tip> --no-merge-base  # a push
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = {"_reference", ".venv", ".git", ".ipynb_checkpoints"}
STAMP_KEY = "ml4t_provenance"
INJECTED_TAG = "injected-parameters"


class _ValidatedByConsumer:
    """An override whose value this gate deliberately does not check.

    The allowlist otherwise pins each name to the one value that preserves the
    production surface, because nothing downstream would catch the other one -
    ``FORCE_RETRAIN=False`` silently reuses a cached fit and no later check notices.

    This is a second kind of allowlist entry, not a second entry. Adding a name with a
    pinned value says "this override is harmless at this value"; marking one with this
    says "this gate is the wrong place to check this override at all". A parameter
    earns the marker only when BOTH hold:

    1. No value of it can reduce the execution surface. It adds a declaration rather
       than removing work, which is what separates it from ``FORCE_RETRAIN=False``.
    2. The code that consumes it rejects a wrong value outright, so nothing is left
       unchecked - merely moved to where the check can actually be made. For an
       identity-bearing value the gate could not make it anyway: the correct one
       depends on registry state at run time, which a commit hook does not have.

    Both conditions, not either. A future exemption citing this precedent has to show
    both, and a name that fails the first is a reduced run wearing a declaration.
    """

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<validated by the consumer>"


VALIDATED_BY_CONSUMER = _ValidatedByConsumer()

PRODUCTION_SAFE_PARAMETERS: dict[str, object] = {
    "FORCE_REBACKTEST": True,
    "FORCE_RETRAIN": True,
    "USE_CACHE": False,
    # Names the population this run supersedes, which research/population.py requires
    # on a re-run into a changed population and which a person sets deliberately. It
    # adds a declaration rather than removing work, so it cannot reduce the run, and
    # population.py raises unless it equals the current population hash exactly.
    "SUPERSEDES_POPULATION": VALIDATED_BY_CONSUMER,
    # The causal identity this run retires. Same shape as SUPERSEDES_POPULATION and
    # for the same reason: it adds a declaration rather than removing work, so it
    # cannot reduce the run, and register_causal_run refuses a hash that is not a
    # current canonical identity for the same label.
    "SUPERSEDES_CAUSAL": VALIDATED_BY_CONSUMER,
    # Replaces a holdout evaluation the window already carries instead of adding a second
    # one. It deletes rows, which is why it is worth saying why it belongs here: it cannot
    # reduce what the run computes - the refit, the registration and every check still
    # happen - and the row it removes is one the notebook has already established is a
    # superseded generation of the same window. Without it a correction that moves every
    # training identity could be computed but never carried through to the holdout, because
    # the only route would be editing the notebook's source for one run and editing it back.
    "REPLACE_HOLDOUT": True,
}


def _coerce_bool(value: object) -> bool | None:
    """Return a boolean for Papermill's bool-like CLI values.

    The same override reaches this module as ``True`` through a YAML parameter file,
    ``"true"`` through ``papermill -p`` (which stringifies everything), and ``1``
    through a JSON declaration. All three name the same run, so ``1`` and ``"1"``
    have to coerce alike — otherwise one form is read as a boolean and the other as
    the string ``"1"``, and comparing the two reports a contradiction that is not
    there.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def production_parameters(parameters: dict[str, object]) -> bool:
    """Whether overrides preserve the full production execution surface.

    Note that this is not "carries no overrides". A canonical run may legitimately be
    parameterized; what makes it production is that every override is one the surface
    survives. An unlisted name is never production, whatever its value.
    """
    for name, value in parameters.items():
        if name not in PRODUCTION_SAFE_PARAMETERS:
            return False
        expected = PRODUCTION_SAFE_PARAMETERS[name]
        if expected is VALIDATED_BY_CONSUMER:
            continue
        if _coerce_bool(value) is not expected:
            return False
    return True


def _normalize_value(value: object) -> tuple[str, object]:
    """Tagged scalar form: ``("bool", …)``, ``("num", …)`` or ``("str", …)``.

    The tag is what keeps ``True`` distinct from the number 1. Python treats them as
    equal — ``bool`` subclasses ``int`` — so an untagged form would let a numeric
    override match a boolean one.

    Numbers compare as ``Decimal`` rather than ``float``, so two integers that differ
    above 2**53 stay different. Floats route through ``str`` first, so the literal
    ``0.1`` matches the string ``"0.1"`` instead of its binary expansion. NaN and the
    infinities compare as lowercased text, because a Decimal NaN does not equal
    itself and every form of a non-finite value has to reach the same normal form.
    """
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float, str)):
        text = str(value).strip()
        try:
            number = Decimal(value if isinstance(value, int) else text)
        except (InvalidOperation, ValueError):
            return ("str", text)  # not a number at all
        if number.is_finite():
            return ("num", number)
        # Decimal's own spelling is the normal form that makes `inf`, `Infinity`
        # and `float("inf")` agree, and NaN compare equal to itself at all.
        return ("str", str(number).lower())
    return ("str", str(value))


def _normalize_parameters(parameters: dict[str, object]) -> dict[str, object]:
    """Comparable form of a parameter set.

    Papermill's ``-p`` stringifies every value, so one override arrives as ``True``
    through a YAML file and ``"true"`` through the CLI. Normalizing lets the two
    compare equal, so a declaration is not reported as contradicting the injected
    cell that recorded the same run.

    Only the parameters in ``PRODUCTION_SAFE_PARAMETERS`` are read as booleans.
    Every other name keeps its type, so ``MAX_SYMBOLS = 1`` still contradicts a
    declared ``{"MAX_SYMBOLS": true}`` while matching ``{"MAX_SYMBOLS": "1"}``.
    """
    out: dict[str, object] = {}
    for name, value in parameters.items():
        if name in PRODUCTION_SAFE_PARAMETERS:
            coerced = _coerce_bool(value)
            if coerced is not None:
                out[name] = ("bool", coerced)
                continue
        out[name] = _normalize_value(value)
    return out


def injected_parameters(nb: dict) -> dict[str, object] | None:
    """Parameters papermill injected into *this* execution, or None if it did not.

    Papermill writes an ``injected-parameters``-tagged cell holding one literal
    assignment per override. Unlike ``metadata.papermill.parameters`` this lives in
    the cell list, so it cannot outlive the run it belongs to: a parameterized run
    rewrites it, and ``jupytext --sync`` drops it with every other cell that is not
    in the paired ``.py``. A cell that is merely left over from an earlier run was
    still executed by the later one — its values were in force either way — so it
    is evidence about the committed outputs, which the metadata block is not.

    Returns None when there is no such cell, which is the ordinary shape of an
    unparameterized run *and* of any notebook synced since. Absence is therefore
    not proof of a production run; that is why the executor must declare.
    """
    cells = [
        c for c in nb.get("cells", []) if INJECTED_TAG in (c.get("metadata", {}).get("tags") or [])
    ]
    if not cells:
        return None
    source = cells[-1].get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    params: dict[str, object] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return params
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            params[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            params[target.id] = _non_literal_value(node.value)
    return params


def _non_literal_value(node: ast.expr) -> object:
    """Value for an injected assignment that ``literal_eval`` will not take.

    Papermill spells a non-finite float as a call, ``float('nan')``, so that form
    has to come back as the number to compare against a declared ``float("nan")``.
    Anything else compares as its own source text.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"float", "int"}
        and len(node.args) == 1
        and not node.keywords
    ):
        constructor = float if node.func.id == "float" else int
        try:
            return constructor(ast.literal_eval(node.args[0]))
        except (ValueError, TypeError):
            pass
    return ast.unparse(node)


def iter_notebooks() -> list[Path]:
    out = []
    for p in REPO_ROOT.rglob("*.ipynb"):
        if SKIP_PARTS & set(p.parts):
            continue
        if p.name.startswith("_executed_") or p.name.startswith("_lock_"):
            continue
        out.append(p)
    return sorted(out)


def _changed_paths(ref: str, merge_base: bool, diff_filter: str | None = None) -> list[str]:
    """Repo-relative paths changed relative to ``ref``, optionally by change type.

    ``-z``, because the gate must not lose a notebook for having a space in its name.
    Plain ``--name-only`` quotes such a path and ``.split()`` then tears it into
    fragments that match no suffix, so the notebook drops out of scope and passes
    unchecked - the one thing a gate must never do.

    ``--no-renames`` for the same reason from the other side: rename detection reports
    only the destination, so moving a ``.py`` would hide the source path whose notebook
    is now stale, and a rename arrives here as a delete plus an add instead.
    """
    spec = f"{ref}...HEAD" if merge_base else f"{ref}..HEAD"
    cmd = ["git", "diff", "--name-only", "-z", "--no-renames"]
    if diff_filter:
        cmd.append(f"--diff-filter={diff_filter}")
    cmd.append(spec)
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return [name for name in out.split("\0") if name]


def notebooks_orphaned_since(ref: str, merge_base: bool = True) -> list[str]:
    """Notebooks this change left with no source: the ``.py`` deleted, the ``.ipynb`` kept.

    This is the strongest form of the staleness the gate exists to catch - a render whose
    source is gone can never be re-derived - and :func:`check_all` cannot see it, because
    its ``paired_py() is None`` branch cannot tell a notebook that was just orphaned from
    one that was never paired. Tracked notebooks are deliberately unpaired, so the
    distinction has to come from the diff rather than from the tree.
    """
    orphaned: list[str] = []
    for name in _changed_paths(ref, merge_base, diff_filter="D"):
        if not name.endswith(".py"):
            continue
        nb = (REPO_ROOT / name).with_suffix(".ipynb")
        if nb.exists() and not (SKIP_PARTS & set(nb.parts)):
            orphaned.append(f"{nb.relative_to(REPO_ROOT)} (deleted source: {name})")
    return sorted(orphaned)


def notebooks_changed_since(ref: str, merge_base: bool = True) -> list[Path]:
    """Notebooks this change is answerable for, relative to ``ref``.

    A change owns a notebook if it edited the notebook OR edited the paired ``.py``,
    since changing the ``.py`` is exactly what makes the rendered notebook stale.
    Notebooks nobody touched are somebody else's.

    ``merge_base`` picks which question is being asked. For a pull request it is "what
    does this branch add on top of the base", so the diff runs from the merge base
    (``ref...HEAD``) and commits that landed on the base meanwhile are not this branch's
    problem. For a push it is "what does the published tree become", so the diff must run
    against the previous tip itself (``ref..HEAD``) - a force-push can revert a notebook
    relative to that tip without the merge base ever seeing it.
    """
    owned: set[Path] = set()
    for name in _changed_paths(ref, merge_base):
        path = REPO_ROOT / name
        if path.suffix == ".ipynb":
            owned.add(path)
        elif path.suffix == ".py":
            nb = path.with_suffix(".ipynb")
            if nb.exists():
                owned.add(nb)
    return sorted(p for p in owned if p.exists() and not (SKIP_PARTS & set(p.parts)))


def paired_py(nb_path: Path) -> Path | None:
    """The .py jupytext-paired to this notebook (same dir + stem). None if absent."""
    cand = nb_path.with_suffix(".py")
    return cand if cand.exists() else None


def git_blob(path: Path) -> str:
    """git blob SHA-1 of the file's current content (working tree)."""
    return subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def outputs_digest(nb: dict) -> str:
    """A digest over the outputs this notebook stores, in cell order.

    ``source_py_blob`` answers "has the ``.py`` changed since somebody stamped
    this", which is not the same question as "did these outputs come from that
    run". A notebook whose ``.py`` is untouched passes the stale check carrying
    outputs from any earlier run - including a superseded one - and that is a real
    path a stale result took to ``main``: ``nbconvert --execute --inplace`` under
    ``nohup ... &`` exits 0 without rewriting the notebook, so the agent stamps a
    file that still holds the previous run's outputs.

    Computed over the parsed structure rather than the file's bytes, so
    reformatting, a re-indent or a different ``json.dumps`` cannot move it. Every
    ``execution_count`` is dropped: the kernel renumbers those on each run and
    ``strip_papermill_cell_metadata`` and friends may clear them, and neither
    changes a single value the notebook reports.

    Markdown cells are excluded for the same reason ``sync-prose`` exists: prose
    carries no output, so folding a prose edit into an executed notebook must not
    invalidate the record of the run.
    """
    payload = [_normalized_outputs(c.get("outputs") or []) for c in _code_cells(nb)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]


# Dropped before hashing. `execution_count` is the kernel's counter, renumbered by
# every run and cleared by the metadata strippers, and it names no value the
# notebook reports. `alt` is figure alt text, which `alt_text_only_drift` already
# forgives in the `.py` on the proven grounds that re-executing cannot change the
# image - and the workflow it forgives requires the matching output metadata to be
# corrected too. Hashing the alt would make that documented correction read as a
# changed result and price four rewritten sentences at a 90-minute re-run, which is
# the cost the exception exists to remove.
VOLATILE_OUTPUT_KEYS = frozenset({"execution_count", "alt"})


def _normalized_outputs(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: _normalized_outputs(v) for k, v in value.items() if k not in VOLATILE_OUTPUT_KEYS
        }
    if isinstance(value, list):
        return [_normalized_outputs(v) for v in value]
    return value


def _module_candidates(module: str, from_dir: Path) -> list[Path]:
    """Where a repo-local ``module`` could live, most specific first.

    Two roots, because the repository has two import mechanisms. Dotted names
    resolve from the repo root (``case_studies.utils.causal``). A bare name also
    resolves from the importing file's own directory, because ``sitecustomize``
    appends every ``NN_*`` chapter directory to ``sys.path`` - which is the only
    reason a chapter's ``async_utils`` is importable at all.
    """
    parts = module.split(".")
    # Both roots for a dotted name too. `17_portfolio_construction/deepm/` is a real
    # package inside a chapter directory, so `import deepm.model` resolves against
    # the chapter, not the repo root - and searching only the root left the code
    # that computes that chapter's allocations out of the digest entirely.
    roots = [from_dir, REPO_ROOT]
    out: list[Path] = []
    for root in roots:
        base = root.joinpath(*parts)
        out += [base.with_suffix(".py"), base / "__init__.py"]
    return out


def _imported_modules(tree: ast.AST, py: Path) -> set[str]:
    """Module names *py* imports, with relative imports resolved to dotted names."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # `from . import x` inside case_studies/utils/ is case_studies.utils.x
                package = py.resolve().parent
                for _ in range(node.level - 1):
                    package = package.parent
                try:
                    prefix = ".".join(package.relative_to(REPO_ROOT).parts)
                except ValueError:
                    continue
                base = f"{prefix}.{node.module}" if node.module else prefix
            else:
                base = node.module or ""
            if not base:
                continue
            names.add(base)
            # `from case_studies.utils import causal` names the module in the alias,
            # not in `node.module`, and that alias is the file that changed.
            names.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
    return names


def repo_local_sources(py: Path) -> list[Path]:
    """Every repository file *py* imports, transitively, sorted repo-relative.

    The numbers in a case-study notebook are computed in ``case_studies/utils/*.py``,
    which ``source_py_blob`` does not cover. #606 changed ``causal.py`` under all
    nine case studies at once, and the committed stamps went on claiming runs whose
    outputs described code no longer in the tree - ``analysis_rows`` 88695 -> 88633
    among them. Nothing in the repository could answer "did this run use the code
    that is here now", and that absence was the finding.

    Best effort by construction: a module reached through ``importlib``, a plugin
    registry or a string name is invisible to ``ast``. It covers the import graph,
    which is where the case-study helpers are.
    """
    seen: set[Path] = set()
    queue = [py.resolve()]
    while queue:
        current = queue.pop()
        if current in seen or not current.is_file():
            continue
        seen.add(current)
        try:
            tree = ast.parse(current.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for module in _imported_modules(tree, current):
            for candidate in _module_candidates(module, current.parent):
                if candidate.is_file():
                    queue.append(candidate.resolve())
                    break
    return sorted(p for p in seen - {py.resolve()} if _within_repo(p))


def _within_repo(path: Path) -> bool:
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    return SKIP_PARTS.isdisjoint(path.parts)


def library_digest(py: Path) -> str:
    """A digest over the repository code *py* imports, transitively.

    Keyed by path as well as content, so moving a module is drift too. An empty
    import graph still has a digest, which keeps "no repo-local imports" distinct
    from "not recorded".
    """
    lines = [f"{p.relative_to(REPO_ROOT).as_posix()}:{git_blob(p)}" for p in repo_local_sources(py)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


ALT_FUNCS = frozenset({"show_with_alt", "show_plotly_with_alt"})

# What `_blank_alts` reports for one alt call. A plain literal is its text; a computed
# alt is its prose segments in source order, because the rendered string is not knowable
# from the source; anything else is unknowable and stays in the AST dump.
_AltText = str | tuple[str, ...] | None


def _alt_call_name(func: ast.expr) -> str | None:
    """The called name for a bare ``f(...)`` or a qualified ``mod.f(...)``, else None.

    Both forms occur in the corpus: ``from utils.style import show_plotly_with_alt``
    gives the bare call, and ``import utils.style as style`` gives the qualified one.
    Matching only ``ast.Name`` left the qualified form outside the exception, so a
    corrected caption there was still read as a stale run.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _percent_cells(src: str) -> list[tuple[str, str, str]]:
    """(marker line, kind, body) per jupytext percent cell.

    Comparing cell structure and raw source, rather than an AST, is deliberate. An AST
    is blind to three things that change a notebook's outputs: a trailing semicolon,
    which suppresses a cell's automatic display; a moved ``# %%``, which changes which
    code shares a cell and therefore which value is the cell's last expression; and a
    changed cell tag, which changes how the cell is treated. All three leave the AST
    identical, so an AST comparison would forgive them.
    """
    cells: list[tuple[str, str, list[str]]] = []
    marker, kind, buf = "", "code", []
    for line in src.splitlines(keepends=True):
        if line.startswith("# %%"):
            cells.append((marker, kind, buf))
            marker = line
            kind = "markdown" if "[markdown]" in line or "[raw]" in line else "code"
            buf = []
            continue
        buf.append(line)
    cells.append((marker, kind, buf))
    return [(m, k, "".join(b)) for m, k, b in cells]


def _alt_literal_spans(arg: ast.expr) -> list[ast.Constant] | None:
    """The string constants of an alt argument, each flagged standalone or in-f-string.

    A plain literal is one constant that is the whole argument. An f-string - including
    an implicit concatenation where any part is one - is an ``ast.JoinedStr`` whose
    ``values`` interleave ``Constant`` prose with ``FormattedValue`` expressions, and
    only the prose is safe to edit without re-executing. Changing
    ``{leader['ic_mean']:+.3f}`` to ``{leader['ic_std']:+.3f}`` changes what the alt
    asserts about the data, so the expression parts stay in the AST dump and moving one
    is stale, exactly as it should be.

    None for anything else - an alt passed as a variable, say - which the caller reports
    as unknowable rather than guessing at.
    """
    if isinstance(arg, ast.Constant):
        return [arg] if isinstance(arg.value, str) else None
    if isinstance(arg, ast.JoinedStr):
        parts: list[ast.Constant] = []
        for value in arg.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value)
            elif not isinstance(value, ast.FormattedValue):
                return None
        return parts
    return None


def _blank_alts(code: str) -> tuple[ast.Module, list[_AltText]] | None:
    """(*code* parsed with every alt's prose neutralised, the alts in source order).

    None if *code* does not parse. The blanking is done on the tree rather than on the
    source: a prose segment inside an f-string and a quoted piece of an implicit
    concatenation are the same kind of AST node in the same argument, and only their
    source text tells them apart. Writing a placeholder over either one textually means
    guessing which, and getting it wrong produces source that does not parse -
    ``show_plotly_with_alt(fig, <alt> f"...")`` - which makes the whole exception
    unavailable for that notebook. Setting the constant's value leaves the question
    unasked, and it is what the caller wanted anyway: the tree is what gets dumped and
    compared.

    Both shapes of alt are blanked, and the difference is in what is reported back. A
    plain literal reports its text, which the caller can require the outputs to carry
    exactly. A computed alt reports its prose segments in order, because its rendered
    text is not knowable from the source - the caller can only require that those
    segments still appear, in order, inside whatever the outputs carry.

    Blanking the computed ones is the point of this change. Writing alt text against
    computed values is the right thing to do: it is what stops a description drifting
    from its figure, and several push reviews have asked for it. Leaving those spans in
    the compared dump priced a wording fix to `case_studies/fx_pairs/06_linear` or
    `cme_futures/07_gbm` at a full re-execution, while the same fix to a plain-literal
    alt next door was accepted as a diff - the opposite of what the exception exists
    for, applied to exactly the notebooks that read their figures off the frame.

    An alt that is neither shape - a variable, say - is reported as ``None`` and left
    untouched in the tree, so any edit to it is caught as ordinary source drift. It is
    reported rather than dropped because dropping it would misalign every later
    position against the carried alts.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    found: list[tuple[tuple[int, int], _AltText]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and _alt_call_name(node.func) in ALT_FUNCS
            and len(node.args) >= 2
        ):
            continue
        arg = node.args[1]
        position = (arg.lineno, arg.col_offset)
        parts = _alt_literal_spans(arg)
        if parts is None:
            found.append((position, None))
            continue
        alt: _AltText = (
            parts[0].value if isinstance(arg, ast.Constant) else tuple(c.value for c in parts)
        )
        for const in parts:
            const.value = "<alt>"
        found.append((position, alt))
    # ast.walk is breadth-first, not source order; the outputs it is compared against
    # are in source order.
    found.sort(key=lambda item: item[0])
    return tree, [alt for _, alt in found]


def _semicolon_flags(code: str, tree: ast.Module) -> tuple[bool, ...]:
    """Whether each top-level statement is terminated by ``;``, in source order.

    An AST cannot see this and it changes the outputs: Jupyter suppresses the automatic
    display of a cell's last expression when it ends with a semicolon, so ``summary``
    and ``summary;`` produce identical ASTs and different notebooks. Only top-level
    statements matter, because only the last of them is auto-displayed.
    """
    lines = code.encode("utf-8").splitlines()
    flags = []
    for stmt in tree.body:
        if stmt.end_lineno is None or stmt.end_col_offset is None:
            flags.append(False)
            continue
        line = lines[stmt.end_lineno - 1] if stmt.end_lineno - 1 < len(lines) else b""
        rest = line[stmt.end_col_offset :].split(b"#", 1)[0].strip()
        flags.append(rest.startswith(b";"))
    return tuple(flags)


_PAPERMILL_MARKER = re.compile(r"\s+papermill=\{.*\}(?=(?:\s+[A-Za-z_][A-Za-z0-9_-]*=)|\r?\n?$)")


def _comparable(
    src: str, *, strip_papermill: bool = False, blank_alts: bool = True
) -> list[tuple] | None:
    """Cells of *src* reduced to what an alt-text edit is allowed to leave alone.

    Per cell: the marker line, the kind, and for a code cell an alt-blanked AST dump
    plus its semicolon flags. The AST dump is taken without attributes so that neither
    a longer alt string nor ``ruff format`` rewrapping a call can move it - both are
    formatting, and formatting cannot change an output.

    What the AST would miss is added back explicitly. The marker line carries the cell
    tags, so retagging is a change. The list of cells carries the ``# %%`` boundaries,
    so moving one is a change - it decides which code shares a cell and therefore which
    value is the cell's last expression. The semicolon flags carry display suppression.

    A markdown cell's body is dropped: it is a comment in the ``.py`` and cannot affect
    outputs, so its text may change freely while its marker and position still count.

    ``blank_alts=False`` keeps the alt literals in the dump, which is what a caller wants
    when it is deciding whether a drift is *prose*. Blanking them is right for
    ``alt_text_only_drift``, which pairs it with a check that the outputs already carry the
    new alt; a caller without that check would read a corrected alt as prose and then
    preserve output metadata still holding the old text.
    """
    out: list[tuple] = []
    for marker, kind, body in _percent_cells(src):
        if strip_papermill:
            marker = _PAPERMILL_MARKER.sub("", marker)
        if kind != "code":
            # Only when the body really is all comments. A percent-format markdown cell
            # holds nothing else, so a non-comment line means the marker does not
            # describe the content - and dropping the body would then hide executable
            # code. Measured: appending `fig;` after the last `# %% [markdown]` marker
            # of a real notebook was forgiven until this compared the body.
            if all(not ln.strip() or ln.lstrip().startswith("#") for ln in body.splitlines()):
                out.append((marker, kind))
            else:
                out.append((marker, kind, body))
            continue
        if blank_alts:
            blanked = _blank_alts(body)
            if blanked is None:
                return None
            tree = blanked[0]
        else:
            try:
                tree = ast.parse(body)
            except SyntaxError:
                return None
        # Semicolon flags read the ORIGINAL body: blanking now only changes a constant's
        # value, so every position in the tree still describes the source it came from.
        out.append((marker, kind, ast.dump(tree), _semicolon_flags(body, tree)))
    return out


def alt_text_only_drift(stamped_blob: str, py: Path, nb: dict) -> bool:
    """Whether source drift is proven unable to change the executed outputs.

    A stamp records the ``.py`` blob that was executed, and any edit to the ``.py``
    moves the blob, so the gate reads a corrected figure description as a notebook
    that needs re-executing. For alt text that is the wrong answer, and expensively
    so: ``show_plotly_with_alt`` publishes ``metadata={..., "image/png": {"alt": alt}}``
    and takes the image itself from ``fig._repr_mimebundle_()``, which never sees the
    alt string. So the alt in a notebook's output metadata is a verbatim copy of the
    source literal, and re-executing to change one cannot produce different outputs.
    ``nasdaq100_microstructure/04`` is 90 minutes and 43 GB to restate four sentences.

    Removal of inline Papermill execution metadata is also output-preserving when
    both paired files now declare ``cell_metadata_filter: tags,-all``. Papermill
    writes timing and status into cell markers; Jupyter does not execute those
    values. The exact filter prevents them from returning on the next sync.

    Both halves have to hold, and neither is a judgement:

    * every cell lines up - same markers, same kinds, same order - and every code cell's
      source is byte-identical once the alt literals are blanked, so nothing that
      executes changed. Only markdown bodies are free, because they are comments in the
      ``.py``, and
    * every alt in the notebook's output metadata already equals the literal in its own
      cell, so the outputs on disk are the ones this source produces. For an alt built
      from an f-string the rendered text is not knowable from the source, so what is
      required instead is that the source's prose still appears in the carried alt, in
      order, with the interpolated values in the gaps. Same bargain, weaker only where
      it has to be.

    Anything else - a changed constant, a reordered call, a trailing semicolon, a moved
    ``# %%``, an alt the outputs do not carry - is stale, which is what the stamp is for.
    """
    old = subprocess.run(
        ["git", "cat-file", "blob", stamped_blob],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if old.returncode != 0:
        return False  # the stamped blob is not in this repo; cannot compare
    new_source = py.read_text(encoding="utf-8")
    filter_value = (nb.get("metadata", {}).get("jupytext", {}) or {}).get("cell_metadata_filter")
    cleaned_papermill = (
        "papermill={" in old.stdout
        and "papermill={" not in new_source
        and "cell_metadata_filter: tags,-all" in new_source
        and filter_value == "tags,-all"
    )
    old_cells = _comparable(old.stdout, strip_papermill=cleaned_papermill)
    new_cells = _comparable(new_source)
    if old_cells is None or new_cells is None or old_cells != new_cells:
        return False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not any(fn in src for fn in ALT_FUNCS):
            continue
        blanked = _blank_alts(src)
        if blanked is None:
            return False
        carried = [
            (out.get("metadata") or {}).get("image/png", {}).get("alt")
            for out in cell.get("outputs", [])
            if "image/png" in out.get("data", {})
        ]
        if len(blanked[1]) != len(carried):
            return False
        for a, c in zip(blanked[1], carried, strict=True):
            if a is None:
                # An alt this cannot read - a variable, say. Its whole span stays in the
                # AST dump the first half compared, so any edit to it is already stale;
                # what is left to require is that the output carries an alt at all, which
                # is what catches an alt call added since the notebook was executed.
                if not c:
                    return False
            elif isinstance(a, tuple):
                if not _carries_prose(a, c):
                    return False
            elif a != c:
                return False
    return True


def _carries_prose(segments: tuple[str, ...], carried: str | None) -> bool:
    """Whether *carried* is a rendering of an f-string with these prose *segments*.

    The rendered text of a computed alt is not knowable from the source, so it cannot
    be compared exactly the way a plain literal is. What can be required is that the
    source's prose still appears in the output, in order, with the interpolated values
    in the gaps - which holds exactly when the notebook's outputs were produced by, or
    corrected to, this source.

    That is the same bargain the plain-literal branch strikes: an alt correction is
    accepted as a diff only when the output metadata carries the correction too.
    Editing the ``.py`` and leaving the executed alt saying the old thing is stale, and
    should be.
    """
    if carried is None:
        return False
    position = 0
    for segment in segments:
        found = carried.find(segment, position)
        if found < 0:
            return False
        position = found + len(segment)
    return True


def contradicts_injected_cell(nb: dict, parameters: dict[str, object]) -> str | None:
    """Why ``parameters`` disagrees with the notebook's injected cell, or None.

    Only meaningful when the notebook still carries an ``injected-parameters``
    cell. No cell means no evidence either way, not agreement.
    """
    injected = injected_parameters(nb)
    if injected is None:
        return None
    declared_n = _normalize_parameters(parameters)
    injected_n = _normalize_parameters(injected)
    if declared_n == injected_n:
        return None
    return (
        f"declared parameters {parameters!r} do not match the injected-parameters "
        f"cell this notebook was executed with, {injected!r}"
    )


def unwritten_run(nb: dict, py: Path) -> str | None:
    """Why this stamp would record a run that never rewrote the notebook, or None.

    The digests are computed from whatever is in the file at stamp time, so a run
    that exited without writing leaves the previous run's outputs to be recorded as
    the new ones - and every later check agrees with itself. That is the failure
    that motivates the outputs digest, and only this refusal catches it, because
    afterwards there is nothing left to disagree.

    The signal is a source change with no corresponding output change: the previous
    stamp says the notebook was executed from a different ``.py``, and the outputs
    are byte-identical to what that older run produced. ``nbconvert --execute
    --inplace`` under ``nohup ... &`` exits 0 and leaves exactly that state.

    Silent when there is no previous stamp to compare against, and when the ``.py``
    has not moved - re-running unchanged source to the same values is a normal
    thing to do and says nothing about whether the file was written.
    """
    previous = nb.get("metadata", {}).get(STAMP_KEY) or {}
    recorded = previous.get("outputs_digest")
    if recorded is None:
        return None
    if previous.get("source_py_blob") == git_blob(py):
        return None
    if recorded != outputs_digest(nb):
        return None
    return (
        "the .py has changed since the last stamp, but the outputs are exactly the "
        "ones the previous run produced. A run that changed the source and changed "
        "no output almost always means the execution never wrote this file - "
        "`nbconvert --execute --inplace` in the background exits 0 and does that. "
        "Re-execute and check the outputs moved. If this notebook really is "
        "deterministic enough that the change altered nothing it prints, pass "
        "--allow-unchanged-outputs to say so."
    )


def stamp_notebook(
    nb_path: Path,
    executor: str,
    notes: str | None = None,
    *,
    parameters: dict[str, object],
    allow_unchanged_outputs: bool = False,
) -> dict:
    """Record how this notebook was executed.

    ``parameters`` is the executor's statement of the overrides the run used —
    ``{}`` for a production run. It is never read back out of the notebook: see the
    module docstring for why ``metadata.papermill.parameters`` cannot answer that
    question. Stamping overwrites that metadata with the declared set, so the
    fossil cannot survive to contradict the stamp.
    """
    py = paired_py(nb_path)
    if py is None:
        raise SystemExit(f"no paired .py for {nb_path.relative_to(REPO_ROOT)} — cannot stamp")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    conflict = contradicts_injected_cell(nb, parameters)
    if conflict:
        raise SystemExit(f"refusing to stamp {nb_path.relative_to(REPO_ROOT)}: {conflict}")
    if not was_executed(nb):
        # The gate catches this at commit time as HOLLOW; catching it here names the
        # step that went wrong while the run is still on screen. The way it happens is
        # a `jupytext --sync` that rebuilds the .ipynb from a newer .py after the run
        # and before the stamp, which discards the outputs; nb-run.sh orders the sync
        # after the stamp for exactly this reason.
        raise SystemExit(
            f"refusing to stamp {nb_path.relative_to(REPO_ROOT)}: no code cell carries an "
            "output or an execution count, so nothing in it was executed. A stamp on this "
            "would claim a run that left no trace. Execute it, or leave it cleared."
        )
    if not allow_unchanged_outputs and (reason := unwritten_run(nb, py)):
        raise SystemExit(f"refusing to stamp {nb_path.relative_to(REPO_ROOT)}: {reason}")
    stamp = {
        "source_py_blob": git_blob(py),
        # What the run produced, and the repository code that produced it. Neither is
        # covered by source_py_blob, and each was a way a superseded result reached
        # main with every check green: outputs from an earlier run under an untouched
        # .py, and outputs computed by a case_studies/utils module that has since
        # moved. See outputs_digest and library_digest.
        "outputs_digest": outputs_digest(nb),
        "library_digest": library_digest(py),
        "executed_at": datetime.now(UTC).isoformat(),
        "executor": executor,
        "production": production_parameters(parameters),
        "parameters": parameters,
    }
    if notes:
        stamp["notes"] = notes
    metadata = nb.setdefault("metadata", {})
    metadata[STAMP_KEY] = stamp
    metadata.setdefault("papermill", {})["parameters"] = parameters
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return stamp


def stamped_at(ref: str = "HEAD") -> set[str]:
    """Repo-relative notebooks whose committed version at *ref* carries a stamp.

    One `git grep` over the tree rather than a read per notebook, because the answer
    is needed for every unstamped file and there are several hundred of them.
    """
    result = subprocess.run(
        ["git", "grep", "-l", f'"{STAMP_KEY}"', ref, "--", "*.ipynb"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    prefix = f"{ref}:"
    return {line[len(prefix) :] for line in result.stdout.splitlines() if line.startswith(prefix)}


def stamp_reference(base_branch: str = "main") -> str:
    """The commit a stamp must not have disappeared since.

    Not ``HEAD``. Comparing the working tree with ``HEAD`` catches a removal that has
    not been committed yet and nothing else: once it is committed both sides are
    unstamped, and the check goes quiet exactly where it is needed, which is CI
    reading a pushed branch. The fork point from the base branch makes the question
    "did this branch drop a stamp", and answers it the same way in the pre-commit
    hook, in CI, and days later.

    Falls back to ``HEAD`` when there is no base branch to fork from - a fresh clone
    with no remote, or work on ``main`` itself, where the fork point IS ``HEAD``.
    """
    for ref in (f"origin/{base_branch}", base_branch):
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", ref],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if merge_base.returncode == 0 and merge_base.stdout.strip():
            return merge_base.stdout.strip()
    return "HEAD"


def _fork_point(ref: str) -> str:
    """Where HEAD forked from *ref*, for the de-stamp comparison under ``--since``.

    ``stamp_reference`` answers this for the default base branch. A pull request names
    its own base, so the same question has to be asked against that ref rather than
    against ``origin/main``, or a PR onto a release branch compares its stamps to a
    commit neither side descends from. Falls back to *ref* itself when the two share no
    history, which is the honest answer for an unrelated ref.
    """
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return merge_base.stdout.strip() if merge_base.returncode == 0 else ref


def was_executed(nb: dict) -> bool:
    """True if any non-empty code cell in *nb* shows evidence of having been run.

    Evidence is an output OR a non-null ``execution_count``. Outputs alone are not
    enough to ask about: a cell that only assigns, writes a file or logs somewhere
    else runs successfully and displays nothing, and a notebook made entirely of
    those would otherwise read as never executed. The counter is written by the
    kernel on every execution and cleared only deliberately, so the two together
    separate "ran and said little" from "did not run".

    The distinction matters twice below: a CLEARED notebook shows no evidence and so
    claims nothing, and a STAMPED one claims a production run that left no evidence
    at all, which no real run does.
    """
    saw_code = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if not "".join(cell.get("source") or []).strip():
            continue
        saw_code = True
        if cell.get("outputs") or cell.get("execution_count") is not None:
            return True
    return not saw_code


def is_cleared(nb_path: Path) -> bool:
    """True if *nb_path* carries neither a provenance stamp nor any output.

    This is the state an edited-but-not-yet-executed notebook is committed in, and
    it is the one state that makes the workflow linear: edit the ``.py``, sync,
    clear the ``.ipynb``, commit. The notebook then asserts nothing about a run, so
    there is nothing for the gate to catch it lying about. Execution restores the
    outputs and the stamp, and the gate resumes checking them.

    It is precisely distinguishable from the render the gate exists to reject. A
    hollow render carries ``production: True`` over an empty output set - it claims
    a run that did not happen. A cleared notebook carries no stamp at all.
    """
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return not nb.get("metadata", {}).get(STAMP_KEY) and not was_executed(nb)


def destamped(ref: str | None = None, only: set[str] | None = None) -> list[str]:
    """Notebooks stamped at *ref* and unstamped now.

    ``only`` restricts the answer to those paths, so the pre-commit gate reports a
    stamp this commit is dropping rather than one another commit dropped earlier.

    The gate says nothing about a notebook with no stamp, which is deliberate - the
    corpus is being stamped as notebooks are re-run, and failing the ones that have
    not been yet would fail hundreds of teaching notebooks that are not the subject.
    The hole that leaves is that DROPPING a stamp turns the check off for that file
    instead of failing it, which is the same shape as a review that passes because
    it never looked. A stamp is only ever added, so a notebook that had one and does
    not is a regression whatever removed it.
    """
    return sorted(
        rel
        for rel in stamped_at(ref or stamp_reference())
        if (only is None or rel in only)
        and (REPO_ROOT / rel).exists()
        and not json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
        .get("metadata", {})
        .get(STAMP_KEY)
        # Dropping the stamp AND the outputs together is the documented way to
        # commit an edited notebook before it is re-executed; only dropping the
        # stamp while outputs stay is the regression this looks for.
        and not is_cleared(REPO_ROOT / rel)
    )


class CheckResult(NamedTuple):
    """The gate's verdicts, one list of repo-relative notebooks each.

    A ``NamedTuple`` rather than a dataclass because callers iterate it - "no
    category may name a notebook outside ``only``" is asserted by looping over the
    result - and unpack it positionally.
    """

    stale: list[str]
    testmode: list[str]
    contradicted: list[str]
    unverified: list[str]
    alt_only: list[str]
    hollow: list[str]
    outputs_changed: list[str]
    library_drift: list[str]
    # Stamped before the two digests existed, so neither can be checked. Reported as
    # a count rather than failed: every stamp predating them lacks both, and failing
    # those would turn every branch red at once for a defect none of them has.
    undigested: list[str]


def check_all(
    strict: bool = False,
    only: set[str] | None = None,
) -> CheckResult:
    """Return the gate's verdicts over the tracked notebooks.

    ``stale``, ``testmode``, ``contradicted``, ``hollow`` and ``outputs_changed``
    fail. ``alt_only``, ``library_drift`` and ``undigested`` are reported so that
    forgiving a drift is never silent: a notebook in one of those lists has
    something that no longer matches its stamp, and the reason it is allowed is
    printed rather than assumed.

    ``library_drift`` is deliberately not a failure yet. On the measurement in
    ml4t/agent-workspace#917 it would immediately block at least five notebooks
    across three case studies, and turning it into a hard failure is a separate
    decision from being able to see it at all.

    ``only`` restricts the scan to the notebooks whose ``.ipynb`` or paired ``.py``
    is in that set of repo-relative paths. The pre-commit gate passes the staged
    files, so a notebook someone else left dirty in the working tree no longer
    blocks an unrelated commit; nothing unstaged can reach main, so the narrower
    scan gives up no protection. CI calls it with no ``only`` and still scans the
    whole tree.
    """
    stale: list[str] = []
    testmode: list[str] = []
    contradicted: list[str] = []
    unverified: list[str] = []
    alt_only: list[str] = []
    hollow: list[str] = []
    outputs_changed: list[str] = []
    library_drift: list[str] = []
    undigested: list[str] = []
    for nb_path in iter_notebooks():
        rel = str(nb_path.relative_to(REPO_ROOT))
        py = paired_py(nb_path)
        if py is None:
            continue  # un-paired notebooks have no .py to drift from
        if only is not None and rel not in only and str(py.relative_to(REPO_ROOT)) not in only:
            continue
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        stamp = nb.get("metadata", {}).get(STAMP_KEY)
        if not stamp:
            unverified.append(rel)
            continue
        stamped_blob = stamp.get("source_py_blob")
        if stamped_blob != git_blob(py):
            if alt_text_only_drift(stamped_blob, py, nb):
                alt_only.append(rel)
            else:
                stale.append(rel)
        if not stamp.get("production", False):
            testmode.append(f"{rel} (params={stamp.get('parameters')})")
        conflict = contradicts_injected_cell(nb, stamp.get("parameters") or {})
        if conflict:
            contradicted.append(f"{rel} ({conflict})")
        if not was_executed(nb):
            hollow.append(rel)
        stamped_outputs = stamp.get("outputs_digest")
        stamped_library = stamp.get("library_digest")
        if stamped_outputs is None and stamped_library is None:
            undigested.append(rel)
        else:
            if stamped_outputs is not None and stamped_outputs != outputs_digest(nb):
                outputs_changed.append(rel)
            if stamped_library is not None and stamped_library != library_digest(py):
                library_drift.append(rel)
    return CheckResult(
        stale,
        testmode,
        contradicted,
        unverified,
        alt_only,
        hollow,
        outputs_changed,
        library_drift,
        undigested,
    )


def code_cells_only(comparable: list[tuple] | None) -> list[tuple] | None:
    """*comparable* with the pure-markdown cells dropped.

    A markdown cell is a comment block in the ``.py``. Adding, deleting, merging or
    retagging one cannot change what any code cell computes: each code cell carries its
    own ``# %%`` marker, so removing a markdown cell between two of them does not join
    them, and markdown produces no outputs of its own to be invalidated.

    `_comparable` keeps a non-code cell's body when that body is not all comments,
    because then the marker does not describe the content and dropping the body would
    hide executable text. Those entries are three-tuples and are kept here for the same
    reason - only the two-tuple form, which is provably nothing but prose, is dropped.
    """
    if comparable is None:
        return None
    return [cell for cell in comparable if cell[1] == "code" or len(cell) != 2]


def drift_is_prose_only(stamped_blob: str, py: Path) -> bool:
    """Whether a stale-reading drift changes no code cell, so ``sync-prose`` resolves it.

    The gate compares whole-file blobs, so any edit to the ``.py`` reads as stale. That is
    the right default - it is one hash and it cannot be argued with - but it makes the
    report say "re-run" to an author who moved a paragraph, and a re-run of
    ``us_equities_panel`` is 52 hours to relocate a heading. This does not forgive the
    drift: the ``.ipynb`` still carries the old prose and still has to be brought forward.
    It only decides which of the two ways of doing that the report should name.

    Same comparison ``sync_prose`` gates itself on, so a notebook this calls prose-only is
    exactly one ``sync-prose`` away from passing, and one it calls executable really does
    need the run.
    """
    old = subprocess.run(
        ["git", "cat-file", "blob", stamped_blob],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if old.returncode != 0:
        return False  # stamped blob is gone; cannot compare, so do not soften the report
    before = code_cells_only(_comparable(old.stdout, blank_alts=False))
    after = code_cells_only(_comparable(py.read_text(encoding="utf-8"), blank_alts=False))
    return before is not None and after is not None and before == after


def _output_counts(nb: dict) -> list[int]:
    """Outputs per code cell, in order. The unit a prose sync must leave untouched."""
    return [len(c.get("outputs", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def sync_prose(nb_path: Path) -> str:
    """Fold a prose-only ``.py`` edit into an executed notebook, without re-running it.

    A stamp records the ``.py`` blob that was executed, so any edit to the ``.py`` reads
    as stale. For a prose edit that is the wrong answer, and it is the reason correcting a
    notebook to a written standard has been treated as unaffordable: consolidating four
    paragraphs into one is a markdown change that cannot move a single number, and pricing
    it at a re-run of `us_equities_panel` prices it at 52 hours.

    Editing the *text* of a markdown cell in place was already forgiven, by
    `alt_text_only_drift`. What was not is changing the *shape* of the prose - deleting a
    cell, merging two, adding one - and that is exactly what bringing a notebook under the
    three-tagged-cells rule requires. So the gate was stricter than the physics.

    This is not a way past the gate; it makes the gate's claim true again. The claim is
    "this .ipynb was produced by this .py", and after a prose edit that is still true of
    every cell that computes anything. So:

    * every code cell must match the stamped source exactly - same marker, same tags, same
      AST, same display suppression, same order. Anything else and this refuses, and the
      notebook needs the re-run it always needed.
    * ``jupytext --update`` writes the new prose into the existing ``.ipynb`` and keeps the
      outputs, rather than rebuilding the file and losing them.
    * the stamp keeps its original ``executed_at`` and ``executor``, because that is when
      the notebook really was executed and by what. Only the blob moves, and a note records
      that the move was prose.
    """
    py = paired_py(nb_path)
    rel = nb_path.relative_to(REPO_ROOT)
    if py is None:
        raise SystemExit(f"no paired .py for {rel}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    stamp = nb.get("metadata", {}).get(STAMP_KEY)
    if not stamp:
        raise SystemExit(
            f"{rel} carries no provenance stamp, so there is no executed state to preserve. Run it."
        )
    stamped_blob = stamp["source_py_blob"]
    old = subprocess.run(
        ["git", "cat-file", "blob", stamped_blob],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if old.returncode != 0:
        raise SystemExit(
            f"{rel} is stamped against blob {stamped_blob[:12]}, which is not in this repo, "
            "so the code cells cannot be compared. Re-run it."
        )
    # Alt literals are NOT blanked here. This command keeps the outputs, so an alt the
    # output metadata does not carry would be stamped as current while rendering the old
    # text. A genuine alt correction is adjudicated by `alt_text_only_drift`, which checks
    # the outputs already carry it, and never reaches this command.
    before = code_cells_only(_comparable(old.stdout, blank_alts=False))
    after = code_cells_only(_comparable(py.read_text(encoding="utf-8"), blank_alts=False))
    if before is None or after is None:
        raise SystemExit(f"{rel}: could not parse one of the two sources - refusing")
    if before == after:
        pass
    else:
        # Name the first difference. "something changed" sends an author back to a full
        # diff of a two-thousand-line file to find out whether it was theirs.
        where = next(
            (i for i, (a, b) in enumerate(zip(before, after)) if a != b),
            min(len(before), len(after)),
        )
        detail = f"code cell {where + 1} differs"
        if len(before) != len(after):
            detail = f"{len(before)} code cells in the executed source, {len(after)} now"
        raise SystemExit(
            f"{rel}: {detail}, so this is not a prose-only edit and the outputs on disk are "
            "not the ones this source produces. Re-run the notebook."
        )

    before_counts = _output_counts(nb)

    result = subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "ipynb", "--update", str(py)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "No module named jupytext" in result.stderr:
            raise SystemExit(
                f"{rel}: jupytext is not installed in {sys.executable}. Run this command through "
                "the repository environment with `uv run python`."
            )
        raise SystemExit(f"{rel}: jupytext --update failed:\n{result.stderr}")

    # Per cell, not "the notebook still has some outputs". `was_executed` is True as soon as
    # ONE code cell carries an output, so it cannot see an update that kept the first cell's
    # and dropped the rest - which is the failure mode that matters, because the notebook
    # would then be re-stamped as a complete execution while most of it renders blank.
    after_counts = _output_counts(json.loads(nb_path.read_text(encoding="utf-8")))
    if after_counts != before_counts:
        lost = [i + 1 for i, (b, a) in enumerate(zip(before_counts, after_counts)) if a < b]
        raise SystemExit(
            f"{rel}: the update changed the outputs, which is the one thing it exists to "
            + (
                f"avoid - code cell(s) {lost} lost theirs. "
                if lost
                else f"avoid - {len(before_counts)} code cells before, {len(after_counts)} now. "
            )
            + "The file has been left as jupytext wrote it; restore it with `git checkout`."
        )
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    stamp = dict(stamp)
    stamp["source_py_blob"] = git_blob(py)
    stamp["notes"] = (
        f"prose synced from the .py at {datetime.now(UTC).isoformat()} without re-executing; "
        f"every code cell is identical to blob {stamped_blob[:12]}"
    )
    nb.setdefault("metadata", {})[STAMP_KEY] = stamp
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return stamp["source_py_blob"]


def _cmd_sync_prose(args: argparse.Namespace) -> int:
    for name in args.notebooks:
        path = Path(name).resolve()
        if path.suffix == ".py":
            path = path.with_suffix(".ipynb")
        blob = sync_prose(path)
        print(f"prose synced {path.relative_to(REPO_ROOT)}: source_py_blob={blob[:12]}")
    return 0


def _cmd_stamp(args: argparse.Namespace) -> int:
    if args.production:
        parameters: dict[str, object] = {}
    else:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--parameters is not valid JSON: {exc}") from exc
        if not isinstance(parameters, dict):
            raise SystemExit("--parameters must be a JSON object of override name to value")
    s = stamp_notebook(
        Path(args.notebook).resolve(),
        args.executor,
        args.notes,
        parameters=parameters,
        allow_unchanged_outputs=args.allow_unchanged_outputs,
    )
    print(
        f"stamped {args.notebook}: source_py_blob={s['source_py_blob'][:12]} "
        f"executor={s['executor']} production={s['production']}"
    )
    return 0


def _cmd_clear(args: argparse.Namespace) -> int:
    """Strip outputs, execution counts and the provenance stamp from a notebook.

    The result is committable: it claims no run, so nothing about it can be stale.
    Use it after editing a ``.py`` and before the notebook is re-executed, so the
    correction lands on ``main`` in one commit instead of waiting on a run.
    """
    cleared = 0
    for name in args.notebooks:
        nb_path = Path(name).resolve()
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        nb.get("metadata", {}).pop(STAMP_KEY, None)
        nb.get("metadata", {}).pop("papermill", None)
        kept = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
                cell.get("metadata", {}).pop("papermill", None)
                cell.get("metadata", {}).pop("execution", None)
                if "injected-parameters" in (cell.get("metadata", {}).get("tags") or []):
                    continue
            kept.append(cell)
        nb["cells"] = kept
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        try:
            shown = nb_path.relative_to(REPO_ROOT)
        except ValueError:
            shown = nb_path
        print(f"cleared {shown}")
        cleared += 1
    if not cleared:
        print("nothing to clear")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    only = {str(Path(p)) for p in args.paths} or None
    orphaned: list[str] = []
    since_ref: str | None = None
    if args.since:
        if only is not None:
            print("check: pass --since or paths, not both", file=sys.stderr)
            return 2
        merge_base = not args.no_merge_base
        # Before the scope check returns early: an orphaned notebook is invisible to
        # check_all by construction, and a change that deletes only a .py leaves nothing
        # in scope to report it.
        orphaned = notebooks_orphaned_since(args.since, merge_base=merge_base)
        scope = notebooks_changed_since(args.since, merge_base=merge_base)
        only = {str(nb.relative_to(REPO_ROOT)) for nb in scope}
        since_ref = _fork_point(args.since) if merge_base else args.since
        if not only and not orphaned:
            print(f"notebook sync OK: no notebook is changed relative to {args.since}")
            return 0
        if scope:
            print(f"checking {len(scope)} notebook(s) changed relative to {args.since}:")
            for nb in scope:
                print(f"  {nb.relative_to(REPO_ROOT)}")
            print()
    result = check_all(strict=args.strict, only=only)
    stale, testmode, contradicted, unverified, alt_only, hollow = result[:6]
    outputs_changed, library_drift, undigested = result[6:]
    lost = destamped(ref=since_ref, only=only)
    fail = bool(
        stale or testmode or contradicted or lost or hollow or orphaned or outputs_changed
    ) or (args.strict and bool(unverified))
    if orphaned:
        print(
            "ORPHANED (the paired .py was deleted but the rendered .ipynb was kept — "
            "restore the source or delete the notebook with it):"
        )
        for r in orphaned:
            print(f"  {r}")
    if hollow:
        print(
            "HOLLOW (carries a provenance stamp over a notebook with no trace of a run — "
            "a run that left nothing behind; clear the notebook or re-execute it):"
        )
        for r in hollow:
            print(f"  {r}")
    if lost:
        print("DE-STAMPED (carried a provenance stamp where this branch forked, and does not now):")
        for r in lost:
            print(f"  {r}")
    if stale:
        prose, executable = [], []
        for r in stale:
            nb_path = REPO_ROOT / r
            py = paired_py(nb_path)
            stamped = (
                json.loads(nb_path.read_text(encoding="utf-8"))
                .get("metadata", {})
                .get(STAMP_KEY, {})
                .get("source_py_blob")
            )
            if py is not None and stamped and drift_is_prose_only(stamped, py):
                prose.append(r)
            else:
                executable.append(r)
        if prose:
            print(
                "STALE, prose only (no code cell moved - fold it in, do NOT re-run:\n"
                "  uv run python .github/scripts/notebook_provenance.py sync-prose <nb.py>):"
            )
            for r in prose:
                print(f"  {r}")
        if executable:
            print(
                "STALE (a code cell changed since the notebook was executed - re-run in the "
                "canonical env):"
            )
            for r in executable:
                print(f"  {r}")
    if testmode:
        print(
            "TEST-MODE (committed a run with papermill parameter overrides — must be production):"
        )
        for r in testmode:
            print(f"  {r}")
    if contradicted:
        print(
            "CONTRADICTED (the stamp disagrees with the injected-parameters cell in the "
            "committed notebook — re-stamp the run that produced these outputs):"
        )
        for r in contradicted:
            print(f"  {r}")
    if outputs_changed:
        print(
            "OUTPUTS CHANGED (the stored outputs are not the ones the stamped run "
            "produced — the notebook was edited, or a run exited without rewriting it; "
            "re-execute and re-stamp, or clear it):"
        )
        for r in outputs_changed:
            print(f"  {r}")
    if alt_only:
        print(
            "ALT-TEXT ONLY (the .py drifted from its stamp only in figure alt text the "
            "outputs already carry, so re-executing could not change them — allowed):"
        )
        for r in alt_only:
            print(f"  {r}")
    if library_drift:
        print(
            "LIBRARY DRIFT (repository code this notebook imports has moved since it "
            "was executed, so its outputs may describe code no longer in the tree — "
            "advisory, re-run when the numbers matter):"
        )
        for r in library_drift:
            print(f"  {r}")
    if unverified:
        verb = "UNVERIFIED (no provenance stamp"
        verb += (
            " — FAILING under --strict):" if args.strict else " — advisory, backfill over time):"
        )
        print(verb)
        for r in unverified:
            print(f"  {r}")
    if not fail:
        print(
            f"notebook sync OK: {len(stale)} stale, {len(testmode)} test-mode, "
            f"{len(contradicted)} contradicted, {len(alt_only)} alt-text-only, "
            f"{len(outputs_changed)} outputs-changed, {len(library_drift)} library-drift "
            f"(advisory), {len(undigested)} stamped before the digests existed, "
            f"{len(unverified)} unverified (advisory)"
        )
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("stamp", help="write a provenance stamp into a notebook")
    sp.add_argument("notebook")
    sp.add_argument("--executor", required=True, help="environment label, e.g. ml4t-gpu / local-uv")
    sp.add_argument("--notes", default=None)
    sp.add_argument(
        "--allow-unchanged-outputs",
        action="store_true",
        help=(
            "stamp even though the .py moved and the outputs did not - for a notebook "
            "genuinely deterministic enough that the change altered nothing it prints"
        ),
    )
    # The executor states the parameters; the tool does not infer them from
    # notebook metadata written by a different process. See the module docstring.
    params = sp.add_mutually_exclusive_group(required=True)
    params.add_argument(
        "--production", action="store_true", help="this run used no parameter overrides"
    )
    params.add_argument(
        "--parameters",
        help="JSON object of the overrides this run used, e.g. '{\"MAX_SYMBOLS\": 5}'",
    )
    sp.set_defaults(func=_cmd_stamp)

    cp = sub.add_parser("check", help="gate: fail on stale or test-mode stamped notebooks")
    cp.add_argument("--strict", action="store_true", help="also fail on unstamped notebooks")
    cp.add_argument(
        "paths",
        nargs="*",
        help="restrict the scan to these files (the pre-commit gate passes the staged ones); "
        "with none given, the whole tree is scanned",
    )
    cp.add_argument(
        "--since",
        default=None,
        metavar="REF",
        help="check only notebooks this change touched relative to REF (e.g. origin/main), "
        "counting a notebook as touched when its paired .py changed. This is the merge "
        "gate: CI passes the PR base here",
    )
    cp.add_argument(
        "--no-merge-base",
        action="store_true",
        help="diff REF..HEAD instead of REF...HEAD — use for a push, where the question is "
        "what the published tree becomes relative to the previous tip, not what a branch "
        "adds on top of a base",
    )
    cp.set_defaults(func=_cmd_check)

    lp = sub.add_parser(
        "clear",
        help="strip outputs and the stamp so an edited notebook can be committed unexecuted",
    )
    lp.add_argument("notebooks", nargs="+")
    lp.set_defaults(func=_cmd_clear)

    yp = sub.add_parser(
        "sync-prose",
        help="fold a prose-only .py edit into the executed .ipynb, keeping its outputs",
        description=(
            "For a change that touches only markdown - rewording, merging, deleting or "
            "adding a markdown cell. Refuses if any code cell moved, so it cannot be used "
            "to commit an untested change; the notebook is not re-executed and the stamp "
            "keeps the time and executor of the run that produced the outputs."
        ),
    )
    yp.add_argument("notebooks", nargs="+", help=".ipynb or .py paths")
    yp.set_defaults(func=_cmd_sync_prose)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
