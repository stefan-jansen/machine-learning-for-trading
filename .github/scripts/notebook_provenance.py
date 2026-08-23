"""Provenance stamp + sync gate for paired ``.py``/``.ipynb`` notebooks.

A committed ``.ipynb`` should be the *current* ``.py`` executed in a real
environment with production parameters — not a stale render, not a TEST-mode run,
not a run in an environment missing a dependency (e.g. CUDA-LightGBM). This module
stamps that fact into the notebook and provides a gate that rejects violations, so
"edited the ``.py``, ran TEST or the wrong env, committed a stale ``.ipynb``" is
caught mechanically instead of by review.

The stamp lives in ``nb.metadata["ml4t_provenance"]``::

    source_py_blob : git blob hash of the paired .py at execution time
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

Gate (``check``): for every tracked ``.ipynb`` that HAS a stamp,

* ``source_py_blob`` must equal ``git hash-object`` of the current paired ``.py``
  (else the ``.py`` changed since the notebook was executed — STALE),
* ``production`` must be True (else a TEST-mode run was committed), and
* the stamp must not contradict a committed ``injected-parameters`` cell (else the
  notebook was re-executed with overrides after it was stamped).

Notebooks WITHOUT a stamp are reported as "unverified" but do not fail unless
``--strict`` is passed. This is deliberate: adoption is gradual — stamp notebooks
as they are re-run through the canonical path, and the gate enforces only where
provenance exists. Flip to ``--strict`` once the backfill is complete.

Usage::

    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --production
    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --parameters '{"MAX_SYMBOLS": 5}'
    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --production --notes "..."
    uv run python .github/scripts/notebook_provenance.py check          # gate (stamped-only)
    uv run python .github/scripts/notebook_provenance.py check --strict  # also fail on unverified
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

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


ALT_FUNCS = frozenset({"show_with_alt", "show_plotly_with_alt"})


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


def _blank_alts(code: str) -> tuple[str, list[str | None]] | None:
    """(*code* with each alt literal replaced by a placeholder, the alts in source order).

    None if *code* does not parse. Works in bytes because ``col_offset`` is a UTF-8
    byte offset, and replaces from the end so earlier spans keep their offsets.

    An alt built with an f-string is reported as ``None`` rather than skipped. Its text
    is not knowable from the source, so it cannot be compared against what the outputs
    carry - but for the same reason it is not blanked either, so it stays in the AST dump
    and any edit to it is caught as ordinary source drift. Dropping such a call from the
    list instead would misalign every later position against the carried alts, and
    omitting it entirely made the counts disagree and failed the whole notebook: eight
    case studies write the leading configuration into their alt with an f-string, so the
    carve-out never applied to the notebooks that read their figures off the frame.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    data = code.encode("utf-8")
    line_start = [0]
    for line in data.splitlines(keepends=True):
        line_start.append(line_start[-1] + len(line))

    found: list[tuple[int, int, str | None]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _alt_call_name(node.func) in ALT_FUNCS
            and len(node.args) >= 2
        ):
            arg = node.args[1]
            if arg.end_lineno is None or arg.end_col_offset is None:
                return None
            literal = (
                arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
            )
            found.append(
                (
                    line_start[arg.lineno - 1] + arg.col_offset,
                    line_start[arg.end_lineno - 1] + arg.end_col_offset,
                    literal,
                )
            )
    # ast.walk is breadth-first, not source order; the outputs it is compared against
    # are in source order.
    found.sort(key=lambda item: item[:2])
    for begin, finish, literal in reversed(found):
        if literal is None:
            continue  # not blanked, so an edit to it stays visible in the AST dump
        data = data[:begin] + b'"<alt>"' + data[finish:]
    return data.decode("utf-8"), [alt for _, _, alt in found]


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


def _comparable(src: str, *, strip_papermill: bool = False) -> list[tuple] | None:
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
        blanked = _blank_alts(body)
        if blanked is None:
            return None
        try:
            tree = ast.parse(blanked[0])
        except SyntaxError:
            return None
        out.append((marker, kind, ast.dump(tree), _semicolon_flags(blanked[0], tree)))
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
      cell, so the outputs on disk are the ones this source produces.

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
                # A computed alt. Its literal parts are in the AST dump the first half
                # compared, so a change to them is already stale; what is left to require
                # is that the output carries an alt at all, which is what catches an alt
                # call added since the notebook was executed.
                if not c:
                    return False
            elif a != c:
                return False
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


def stamp_notebook(
    nb_path: Path,
    executor: str,
    notes: str | None = None,
    *,
    parameters: dict[str, object],
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
    stamp = {
        "source_py_blob": git_blob(py),
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
    )


def check_all(
    strict: bool = False,
    only: set[str] | None = None,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Return (stale, testmode, contradicted, unverified, alt_only) repo-relative rows.

    Only the first four fail. ``alt_only`` is reported so that forgiving a drift is
    never silent: a notebook in that list has a ``.py`` that no longer matches its
    stamp, and the reason it is allowed is printed rather than assumed.

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
    return stale, testmode, contradicted, unverified, alt_only


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
        Path(args.notebook).resolve(), args.executor, args.notes, parameters=parameters
    )
    print(
        f"stamped {args.notebook}: source_py_blob={s['source_py_blob'][:12]} "
        f"executor={s['executor']} production={s['production']}"
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    only = {str(Path(p)) for p in args.paths} or None
    stale, testmode, contradicted, unverified, alt_only = check_all(strict=args.strict, only=only)
    lost = destamped(only=only)
    fail = bool(stale or testmode or contradicted or lost) or (args.strict and bool(unverified))
    if lost:
        print("DE-STAMPED (carried a provenance stamp where this branch forked, and does not now):")
        for r in lost:
            print(f"  {r}")
    if stale:
        print(
            "STALE (paired .py changed since the notebook was executed — re-run in the canonical env):"
        )
        for r in stale:
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
    if alt_only:
        print(
            "ALT-TEXT ONLY (the .py drifted from its stamp only in figure alt text the "
            "outputs already carry, so re-executing could not change them — allowed):"
        )
        for r in alt_only:
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
    cp.set_defaults(func=_cmd_check)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
