"""Papermill-based notebook execution helpers.

Provides:
- run_notebook(): Execute a .py notebook via Papermill with parameter injection
- get_overrides(): Load per-notebook overrides from tests/overrides.yaml
- collect_chapter_notebooks(): Discover notebooks in chapter directories
- get_tier() / current_test_tier(): Test-tier routing (per-commit / weekly / on-demand)
- get_record_mode(): VCR cassette mode (consumed by Step 5)
- unusable_parameters(): parameter names an override declares that the notebook
  cannot receive (checked by tests/test_pm_helpers.py)

NOTE: Notebooks live directly in chapter directories
(e.g., 05_synthetic_data/01_timegan.py), NOT in code/ subdirs.

overrides.yaml schema (per-notebook, all optional):
    timeout: int (seconds, default 300)
    parameters: dict (papermill -p overrides). Papermill injects these values in a
        cell right after the notebook's ``# %% tags=["parameters"]`` cell, so
        every name must be one the notebook reads below that point without first
        overwriting it. Where the notebook assigns the name is not the test: an
        assignment above the injection point is overridden. A name nothing reads
        is an unused variable, and one that is overwritten first is gone; either
        way the test runs at production scale while this file states a reduction.
        ``unusable_parameters`` is the detector and tests/test_pm_helpers.py
        fails the build on what it finds, with no allowlist.
    skip: bool — hard skip in uv-native run (Docker tests ignore)
    skip_reason: str
    requires_import: str | list[str]
    requires_env: str | list[str] — environment variables the notebook cannot run
        without (API keys, SEC identity). Skipped when any is unset or blank,
        executed when they are all present. Prefer this over `skip: true` for a
        missing capability: `skip` is unconditional, so a notebook marked that
        way stays unexecuted even in the workflow that supplies the credential.
    gpu: bool
    long_running: bool
    docker_env: str — informational (e.g., "benchmark")
    kernel_python: str — interpreter to execute with, for a notebook whose
        dependencies live in a venv other than the one running pytest (the
        py312 image keeps tfcausalimpact in /opt/bsts on NumPy 1.x).
    kernel_launcher: str — repo-relative launcher script for that interpreter.
    tier: "per_commit" | "weekly" | "on_demand" — default "per_commit"
        Per-commit runs the Tests workflow on every PR/push.
        Weekly runs the weekly-external scheduled workflow (Step 2).
        On_demand runs only on manual dispatch (GPU-only NBs).
    reruns: int — flaky-retry count (consumed once pytest-rerunfailures lands,
        Step 2). Default 0.
    record_mode: "replay" | "rewrite" — VCR cassette mode (consumed by Step 5).
        Default "replay".
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import yaml

REPO_ROOT = Path(__file__).parent.parent
OVERRIDES_PATH = REPO_ROOT / "tests" / "overrides.yaml"

# Thread-pool caps for the papermill kernel.
#
# scikit-learn, NumPy's BLAS, numexpr and their dependencies each default to one
# thread per core. A notebook kernel therefore opens ~24 threads per pool on this
# box, and several suites run at once, so the pools oversubscribe the machine by
# about an order of magnitude. OpenMP pools spin rather than block while they
# wait, so the cost lands as wall-clock time inside whichever cell happens to be
# running and reads as that cell timing out. Measured on
# 11_ml_pipeline/02_regularization_paths: the ten-fit LASSO alpha grid took 761 s
# for one fold unpinned and 0.90 s pinned, with identical iteration counts, so the
# optimizer did identical work both times. End to end that notebook went from a
# 300 s cell timeout to passing in 41 s.
#
# CI runners have few cores and lose nothing. A notebook that genuinely wants more
# threads asks the library rather than the environment (LightGBM's num_threads and
# torch.set_num_threads both win over these), or a caller raises them via
# extra_env, which is applied after this table.
KERNEL_THREAD_CAP = os.environ.get("ML4T_TEST_THREADS", "1")
KERNEL_THREAD_CAPS = {
    "OMP_NUM_THREADS": KERNEL_THREAD_CAP,
    "OPENBLAS_NUM_THREADS": KERNEL_THREAD_CAP,
    "MKL_NUM_THREADS": KERNEL_THREAD_CAP,
    "NUMEXPR_NUM_THREADS": KERNEL_THREAD_CAP,
}

# Cache loaded overrides
_overrides_cache: dict | None = None

# ---------------------------------------------------------------------------
# Test tier — controls when a notebook runs in CI.
# Per-commit (default): every PR / push triggers the Tests workflow.
# Weekly: only the scheduled weekly-external workflow (Mon 06:00 UTC).
# On-demand: only manual workflow_dispatch (e.g., GPU-only Tier 3).
# ---------------------------------------------------------------------------
TIER_PER_COMMIT = "per_commit"
TIER_WEEKLY = "weekly"
TIER_ON_DEMAND = "on_demand"
VALID_TIERS = frozenset({TIER_PER_COMMIT, TIER_WEEKLY, TIER_ON_DEMAND})

# VCR cassette modes (Step 5: pytest-recording).
RECORD_REPLAY = "replay"
RECORD_REWRITE = "rewrite"
VALID_RECORD_MODES = frozenset({RECORD_REPLAY, RECORD_REWRITE})


def get_tier(overrides: dict) -> str:
    """Return the test tier declared by overrides (default: per_commit)."""
    tier = overrides.get("tier") or TIER_PER_COMMIT
    if tier not in VALID_TIERS:
        raise ValueError(
            f"Invalid tier {tier!r} in overrides — must be one of {sorted(VALID_TIERS)}"
        )
    return tier


def current_test_tier() -> str:
    """Return the tier the current pytest run is targeting.

    Read from ML4T_TEST_TIER env var; defaults to per_commit so existing
    workflows that don't set it keep their current behavior (only NBs
    without a tier key — i.e., tier=per_commit — execute).
    """
    tier = os.environ.get("ML4T_TEST_TIER") or TIER_PER_COMMIT
    if tier not in VALID_TIERS:
        raise ValueError(f"Invalid ML4T_TEST_TIER={tier!r} — must be one of {sorted(VALID_TIERS)}")
    return tier


def missing_required_env(overrides: dict) -> list[str]:
    """Return the ``requires_env`` variables that are unset or blank.

    An empty list means the notebook's credentials are all present and it must
    execute. This is what keeps a credential-gated notebook honest in both
    directions: it does not fail the per-commit run that has no secret, and it
    does not silently stay skipped in the weekly run that does.
    """
    declared = overrides.get("requires_env")
    if not declared:
        return []
    names = [declared] if isinstance(declared, str) else list(declared)
    return [name for name in names if not (os.environ.get(name) or "").strip()]


def get_reruns(overrides: dict) -> int:
    """Return per-notebook flaky retry count (default 0).

    Consumed by Step 2 (pytest-rerunfailures dep + collection hook adds
    @pytest.mark.flaky(reruns=N) when N > 0). Until that lands the value
    is parsed but no retries happen.
    """
    val = overrides.get("reruns", 0)
    if not isinstance(val, int) or val < 0:
        raise ValueError(f"Invalid reruns={val!r} — must be non-negative int")
    return val


def get_record_mode(overrides: dict) -> str:
    """Return VCR cassette mode for the notebook (default: replay)."""
    mode = overrides.get("record_mode") or RECORD_REPLAY
    if mode not in VALID_RECORD_MODES:
        raise ValueError(
            f"Invalid record_mode {mode!r} — must be one of {sorted(VALID_RECORD_MODES)}"
        )
    return mode


# ---------------------------------------------------------------------------
# Papermill parameter validation.
#
# Papermill injects the `parameters` block as a cell placed directly after the
# notebook's own `parameters`-tagged cell, so an injected value only reaches the
# code that uses it when something below that point reads the name and nothing
# overwrites it first. Neither condition is enforced at run time: papermill
# accepts any name, and an override that misses either one leaves the notebook
# running at its production defaults while overrides.yaml records a reduction.
# `unusable_parameters` is the detector; test_pm_helpers.py runs it over the
# whole file, with no allowlist.
# ---------------------------------------------------------------------------

PARAMETERS_CELL_MARKER = 'tags=["parameters"]'
INJECTED_CELL_MARKER = 'tags=["injected-parameters"]'


def _percent_cell_bounds(source: str) -> list[tuple[str, int, int]]:
    """Split jupytext percent source into (header, first_line, last_line) cells.

    Lines are 1-based and inclusive, matching `ast` line numbers.
    """
    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("# %%")]
    bounds = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] - 1 if pos + 1 < len(starts) else len(lines) - 1
        bounds.append((lines[start], start + 1, end + 1))
    return bounds


def _module_level_events(node: ast.AST, events: list[tuple[str, str, int]]) -> None:
    """Append ("read"|"bind", name, line) in source order for module-level code.

    `if`, `for`, `try` and `with` bodies are entered: a rebinding under one of
    them still overwrites the injected value when it runs, and the read that
    precedes it is what tells the two apart.

    Function, lambda and class bodies are not entered. Their code runs when
    called, which is after the rest of the module, so a name they mention is not
    a read that protects an earlier value from a later module-level rebinding.
    The parts of a definition that *do* evaluate where they are written -
    decorators, default arguments, base classes - are collected.
    """
    if isinstance(node, ast.Lambda):
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                _module_level_events(default, events)
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        evaluated: list[ast.AST] = list(node.decorator_list)
        evaluated += [d for d in node.args.defaults if d is not None]
        evaluated += [d for d in node.args.kw_defaults if d is not None]
        for sub in evaluated:
            _module_level_events(sub, evaluated_events := [])
            events.extend(evaluated_events)
        events.append(("bind", node.name, node.lineno))
        return
    if isinstance(node, ast.ClassDef):
        # A class body runs where it is written, not when something calls it, so
        # what it reads is a module-level read. What it assigns is a class
        # attribute, so those bindings do not reach the module - but they do
        # shadow the module name for the rest of the body, and a read after one
        # of them resolves to the attribute rather than to the parameter.
        for sub in [*node.decorator_list, *node.bases, *(kw.value for kw in node.keywords)]:
            _module_level_events(sub, events)
        body: list[tuple[str, str, int]] = []
        for statement in node.body:
            _module_level_events(statement, body)
        shadowed: set[str] = set()
        for kind, bound, line in body:
            if kind == "read" and bound not in shadowed:
                events.append((kind, bound, line))
            elif kind == "bind":
                shadowed.add(bound)
        events.append(("bind", node.name, node.lineno))
        return
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        # A comprehension has its own scope: its loop targets bind nothing in the
        # module, and inside it they shadow the module name rather than read it.
        targets: set[str] = set()
        for gen in node.generators:
            for sub in ast.walk(gen.target):
                if isinstance(sub, ast.Name):
                    targets.add(sub.id)
        for gen in node.generators:
            _module_level_events(gen.iter, events)
        inner: list[tuple[str, str, int]] = []
        for gen in node.generators:
            for condition in gen.ifs:
                _module_level_events(condition, inner)
        for part in ("elt", "key", "value"):
            if (sub := getattr(node, part, None)) is not None:
                _module_level_events(sub, inner)
        events.extend(event for event in inner if event[1] not in targets)
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        # `from x import MAX_SYMBOLS` below the cell overwrites what was injected
        # exactly as an assignment would.
        for alias in node.names:
            events.append(("bind", (alias.asname or alias.name).split(".")[0], node.lineno))
        return
    if isinstance(node, ast.Name):
        kind = "read" if isinstance(node.ctx, ast.Load) else "bind"
        events.append((kind, node.id, node.lineno))
        return
    if isinstance(node, ast.Call):
        # The callee and the arguments are evaluated before the call happens, so
        # the event goes after theirs. A call is where a function body runs, which
        # is the only way a deferred read can happen; sequencing it with the reads
        # and bindings is what lets the three be compared.
        for sub in ast.iter_child_nodes(node):
            _module_level_events(sub, events)
        if isinstance(node.func, ast.Name):
            events.append(("call", node.func.id, node.lineno))
        return
    if isinstance(node, ast.AugAssign):
        # `X += 1` reads X before writing it.
        _module_level_events(node.value, events)
        if isinstance(node.target, ast.Name):
            events.append(("read", node.target.id, node.target.lineno))
        _module_level_events(node.target, events)
        return
    if isinstance(node, ast.AnnAssign) and node.value is None:
        # `X: int` annotates without assigning, so it binds nothing and leaves
        # the injected value in place. Only the annotation itself is evaluated.
        _module_level_events(node.annotation, events)
        return
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
        # The right-hand side is evaluated before the target is bound, and the
        # ordering matters: `X = X + 1` reads X, `X = 1` discards it. Child order
        # puts the target first for `Assign` and `NamedExpr` alike, so neither can
        # be left to the generic walk below.
        _module_level_events(node.value, events)
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            _module_level_events(target, events)
        return
    for child in ast.iter_child_nodes(node):
        _module_level_events(child, events)


#: Statement fields whose contents may be skipped on a given run. A binding under
#: one of these reaches only the code in the same branch; a binding outside them
#: all runs on every path.
_CONDITIONAL_BODIES = {
    ast.If: ("body", "orelse"),
    ast.For: ("body", "orelse"),
    ast.AsyncFor: ("body", "orelse"),
    ast.While: ("body", "orelse"),
    # `body` too: a statement in it may never be reached, so a binding there
    # does not reach a handler that runs because something above it raised.
    ast.Try: ("body", "handlers", "orelse"),
    ast.ExceptHandler: ("body",),
}


def _branch_paths(tree: ast.Module) -> dict[int, tuple]:
    """line -> the chain of conditional branches enclosing it.

    An empty path means the line runs on every path through the module. One path
    being a prefix of another means the first branch encloses the second, which
    is what says a binding can reach a read: same branch or an outer one.
    """
    paths: dict[int, tuple] = {}

    def walk(node: ast.AST, path: tuple) -> None:
        for field, value in ast.iter_fields(node):
            branching = field in _CONDITIONAL_BODIES.get(type(node), ())
            for item in value if isinstance(value, list) else [value]:
                if not isinstance(item, ast.AST):
                    continue
                inner = path + ((id(node), field),) if branching else path
                line = getattr(item, "lineno", None)
                if line is not None and len(inner) >= len(paths.get(line, ())):
                    paths[line] = inner
                walk(item, inner)

    walk(tree, ())
    return paths


def _top_level_bindings(tree: ast.Module) -> list[tuple[str, int]]:
    """(name, line) for bindings made by an unconditional statement of the module.

    Nested statements are excluded, so this is the set of bindings that run on
    every path through the notebook. A binding under `if`/`for`/`try` may or may
    not run, and deciding which needs the branch analysis a config linter has no
    business carrying, so those are left alone.
    """
    bound = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.append((node.name, node.lineno))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.append(((alias.asname or alias.name).split(".")[0], node.lineno))
            continue
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    bound.append((sub.id, sub.lineno))
    return bound


_NESTED_SCOPE = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _own_scope_nodes(body: list[ast.AST]) -> list[ast.AST]:
    """Nodes belonging to one scope, stopping at every nested scope.

    A nested scope's own statements are somebody else's locals, so descending
    into one would let a store in an inner function hide a read in its parent.
    Each is visited on its own turn by the caller's `ast.walk`.
    """
    found: list[ast.AST] = []
    stack = list(body)
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, _NESTED_SCOPE):
            stack.extend(ast.iter_child_nodes(node))
    return found


def _comprehension_free_reads(node: ast.AST) -> set[str]:
    """Names a comprehension reads from the scope around it.

    Its own loop targets are its locals, so they are subtracted; everything else
    it evaluates resolves outward.
    """
    targets = {
        sub.id
        for gen in node.generators
        for sub in ast.walk(gen.target)
        if isinstance(sub, ast.Name)
    }
    parts: list[ast.AST] = [gen.iter for gen in node.generators]
    parts += [condition for gen in node.generators for condition in gen.ifs]
    for attr in ("elt", "key", "value"):
        if (sub := getattr(node, attr, None)) is not None:
            parts.append(sub)
    reads = set()
    for part in parts:
        for sub in ast.walk(part):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                reads.add(sub.id)
    return reads - targets


def _class_body_free_reads(node: ast.ClassDef) -> set[str]:
    """Names a class body reads from the scope around it.

    Its own assignments become class attributes and shadow an outer name for the
    rest of the body, so they are subtracted. Methods are not walked here - they
    are functions, and `_deferred_global_reads` reaches every function in the
    tree on its own.
    """
    seen: set[str] = set()
    bound: set[str] = set()
    for sub in _own_scope_nodes(node.body):
        if isinstance(sub, ast.Name):
            (seen if isinstance(sub.ctx, ast.Load) else bound).add(sub.id)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in sub.names}
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(sub.name)
        elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            seen |= _comprehension_free_reads(sub)
    for outer in [*node.bases, *node.decorator_list]:
        seen |= {
            sub.id
            for sub in ast.walk(outer)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
        }
    return seen - bound


def _scope_reads_and_calls(node: ast.AST) -> tuple[set[str], set[str]]:
    """(free names this function reads, names of things it calls).

    A class body at module level is not the caller's concern: Python runs one
    where it is written, so what it reads is a module-level read and
    `_module_level_events` records it as one. A class body *inside* a function is
    the other case - it runs when the function is called - so its reads are
    collected here.

    A name the body also binds is local to it throughout, by Python's own scoping
    rule, so reading it there says nothing about the module-level variable. Those
    are excluded; a `global` declaration puts one back.
    """
    args = node.args
    local = {
        a.arg
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
        + [x for x in (args.vararg, args.kwarg) if x is not None]
    }
    declared_global: set[str] = set()
    seen: set[str] = set()
    called: set[str] = set()
    for sub in _own_scope_nodes(node.body if isinstance(node.body, list) else [node.body]):
        if isinstance(sub, ast.Global):
            declared_global |= set(sub.names)
        elif isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                called.add(sub.func.id)
        elif isinstance(sub, ast.Name):
            (seen if isinstance(sub.ctx, ast.Load) else local).add(sub.id)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            local |= {(a.asname or a.name).split(".")[0] for a in sub.names}
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local.add(sub.name)
            if isinstance(sub, ast.ClassDef):
                # Runs when this function is called, so what it reads from
                # outside is deferred too. `_own_scope_nodes` stops at the
                # class, so collect them directly.
                seen |= _class_body_free_reads(sub)
        elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Its own scope, but its free variables resolve out through this
            # one. `_own_scope_nodes` stops here, so collect them directly.
            seen |= _comprehension_free_reads(sub)
    return (seen - local) | (seen & declared_global), called


def _deferred_readers(tree: ast.Module) -> tuple[dict[str, set[str]], set[str]]:
    """(name -> the module-level functions that read it, names read opaquely).

    Knowing *which* function reads a parameter is what makes a module-level call
    evidence about that parameter rather than about calls in general. Reads reach
    a caller through the functions it calls, so the map is closed transitively
    over local calls.

    The second set is the escape hatch: a lambda, or a function defined inside
    another, is not something a module-level call can name, so a parameter read
    from one keeps the blanket exemption instead of being attributed.
    """
    top = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    direct: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    opaque: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        reads, called = _scope_reads_and_calls(node)
        if top.get(getattr(node, "name", None)) is node:
            direct[node.name] = reads
            calls[node.name] = called
        else:
            opaque |= reads
    changed = True
    while changed:
        changed = False
        for caller, callees in calls.items():
            for callee in callees & direct.keys():
                if not direct[callee] <= direct[caller]:
                    direct[caller] |= direct[callee]
                    changed = True
    readers: dict[str, set[str]] = {}
    for func, names in direct.items():
        for name in names:
            readers.setdefault(name, set()).add(func)
    return readers, opaque


def unusable_parameters(py_path: Path, names: Iterable[str]) -> dict[str, str]:
    """Map each declared parameter name that cannot reach the notebook to why.

    An empty dict means every name is injectable. Everything turns on where
    papermill puts the value: in a cell immediately after the notebook's
    `parameters`-tagged cell, or — when there is no tagged cell — above the
    imports, ahead of the whole notebook. Anything the notebook assigns before
    that point is overridden and does not matter. Two things after it do:

    - some binding overwrites the injected value before anything reads it, so
      the notebook goes on using its own number;
    - nothing reads the name at all, so the value is injected into a variable
      the notebook has no use for.

    The overwrite test turns on the read rather than on the indentation.
    Reading the injected value and then rebinding the same name is how the
    `12_causal_dml` notebooks apply an override: they push `CV_FOLDS` into the
    config dict, then read it back out. The value survives, so the entry is
    valid, and `if MAX_SYMBOLS == 0: MAX_SYMBOLS = len(universe)` is the same
    shape with the guard as the read. Every binding is checked, not just the
    first, so a legitimate read-and-restore does not license an overwrite
    further down.

    Only bindings that run on every path are counted, and only when no function
    body reads the name. A conditional binding may not run, and a function that
    reads the name may be called before one that does; proving either way needs
    branch and call analysis that would buy this file nothing. Where the analysis
    cannot be sure, it says nothing — a missed dead entry costs a stale line,
    a wrong rejection costs a real reduction.

    Args:
        py_path: the notebook's `.py` source
        names: the parameter names `overrides.yaml` declares for it

    Returns:
        {name: reason} for the names that cannot take effect.
    """
    names = list(names)
    if not names:
        return {}
    if not py_path.exists():
        return dict.fromkeys(names, f"no notebook at {py_path}")

    source = py_path.read_text()
    tree = ast.parse(source, filename=str(py_path))
    cells = _percent_cell_bounds(source)
    tagged = [c for c in cells if PARAMETERS_CELL_MARKER in c[0]]
    # A committed `injected-parameters` cell is a leftover from a papermill run,
    # replaced on the next one. Reading it as notebook code would report the
    # notebook overwriting exactly what papermill is about to inject.
    stale = [(lo, hi) for header, lo, hi in cells if INJECTED_CELL_MARKER in header]

    # Papermill's cell lands right after the tagged one, or at the very top of
    # the notebook when there is none.
    injected_at = tagged[-1][2] if tagged else 0
    where = (
        "the parameters cell"
        if tagged
        else 'the top of the notebook (it has no `# %% tags=["parameters"]` cell)'
    )

    def live(line: int) -> bool:
        return line > injected_at and not any(lo <= line <= hi for lo, hi in stale)

    events: list[tuple[str, str, int]] = []
    _module_level_events(tree, events)
    branch = _branch_paths(tree)
    readers, opaque = _deferred_readers(tree)

    def reaches(bind: tuple[int, int], read: tuple[int, int]) -> bool:
        """Does this binding run before that read, and on the read's path?

        Order comes from the event sequence, not the line: `_module_level_events`
        emits an assignment's right-hand side before its target, so in a multiline
        `X = (\\n    X + 1\\n)` the read is sequenced first even though the target
        sits on the earlier line. The line is only how a branch path is looked up.
        """
        (bind_seq, bind_line), (read_seq, read_line) = bind, read
        if bind_seq >= read_seq:
            return False
        here, there = branch.get(bind_line, ()), branch.get(read_line, ())
        return here == there[: len(here)]

    problems = {}
    for name in names:
        reads = [
            (seq, line)
            for seq, (kind, bound, line) in enumerate(events)
            if kind == "read" and bound == name and live(line)
        ]
        binds = [
            (seq, line)
            for seq, (kind, bound, line) in enumerate(events)
            if kind == "bind" and bound == name and live(line)
        ]
        if name in opaque or name in readers:
            # A function that reads the name is normally enough to leave the
            # entry alone: it may be called before any binding below, and the
            # tree does not say which happens first. The exception is a binding
            # that nothing can run ahead of - unconditional, with no module-level
            # read and no call to a function that reads this name above it. Then
            # every such call happens after it and none can see the injected
            # value. A read from a lambda or a nested function is not attributable
            # to a name a call site can use, so it keeps the blanket exemption.
            blocking = [(seq, line) for seq, line in binds if branch.get(line, ()) == ()]
            reaching = [
                seq
                for seq, (kind, bound, line) in enumerate(events)
                if kind == "call" and bound in readers.get(name, set()) and live(line)
            ]
            first = min((seq for seq, _ in blocking), default=None)
            reachable = (
                name in opaque
                or first is None
                or any(seq < first for seq in [*(r for r, _ in reads), *reaching])
            )
            if reachable:
                continue
            problems[name] = (
                f"the binding on line {min(line for _, line in blocking)} overwrites the "
                f"injected value before anything below {where} can call a function that "
                "reads it"
            )
        elif not reads:
            problems[name] = f"the notebook never reads it below {where}"
        elif all(any(reaches(b, r) for b in binds) for r in reads):
            problems[name] = (
                f"the binding on line {min(line for _, line in binds)} overwrites the "
                f"injected value: every read below {where} is on a path that rebinds "
                "the name first"
            )
    return problems


def get_overrides(notebook_key: str) -> dict:
    """Get parameter overrides for a notebook from tests/overrides.yaml.

    Args:
        notebook_key: Notebook path relative to repo root, no extension.
                      e.g., "05_synthetic_data/02_tailgan_tail_risk"

    Returns:
        Dict with optional keys: timeout, gpu, parameters
    """
    global _overrides_cache
    if _overrides_cache is None:
        if OVERRIDES_PATH.exists():
            with open(OVERRIDES_PATH) as f:
                _overrides_cache = yaml.safe_load(f) or {}
        else:
            _overrides_cache = {}

    return _overrides_cache.get(notebook_key) or {}


def sync_notebook(py_path: Path) -> Path:
    """Sync a .py notebook to a temporary .ipynb via Jupytext.

    Writes to a temp file so the real .ipynb (which may contain pre-executed
    outputs) is never overwritten.

    Args:
        py_path: Path to the .py source file

    Returns:
        Path to a temporary .ipynb file (caller must clean up)
    """
    # Write to a temp file — never touch the real .ipynb
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".ipynb", prefix=f"_pm_{py_path.stem}_")
    os.close(tmp_fd)
    tmp_ipynb = Path(tmp_path_str)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupytext",
            "--to",
            "notebook",
            "--set-kernel",
            "python3",
            "--output",
            str(tmp_ipynb),
            str(py_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        tmp_ipynb.unlink(missing_ok=True)
        raise RuntimeError(f"Jupytext sync failed for {py_path}: {result.stderr}")

    if not tmp_ipynb.exists():
        raise FileNotFoundError(f"Expected temp .ipynb not found after sync: {tmp_ipynb}")

    return tmp_ipynb


class KernelRouting(NamedTuple):
    """A notebook's validated kernel routing, or why it cannot be used.

    `problem` is None when the routing is usable. `python` and `launcher` are the
    validated values the caller should pass to `run_notebook`, so the call site
    cannot read the overrides differently from what was checked.
    """

    problem: str | None
    python: str | None = None
    launcher: Path | None = None


def check_kernel_routing(overrides: dict) -> KernelRouting:
    """Validate a notebook's kernel routing without needing the image it names.

    A notebook whose dependencies live outside the interpreter running pytest
    declares `kernel_python` (and optionally `kernel_launcher`) in overrides.yaml.
    Both halves are written straight into a kernelspec `argv`, so a stale image or
    a mistyped path produces either a silently skipped notebook or an opaque
    kernel-startup error. Callers turn a problem into a test failure; returning it
    rather than raising keeps this unit-testable off a real image.
    """
    kernel_python = overrides.get("kernel_python")
    launcher = overrides.get("kernel_launcher")

    if not kernel_python:
        # run_notebook gates the whole kernelspec on kernel_python, so a launcher
        # on its own is dropped in silence - the same undiagnosed mis-routing this
        # function exists to close, on the other half of the pair.
        if launcher:
            return KernelRouting(
                f"overrides.yaml declares kernel_launcher {launcher} with no kernel_python. "
                f"The launcher is ignored and the notebook runs on the pytest interpreter."
            )
        return KernelRouting(None)

    # os.access(..., X_OK) is also true for a searchable directory, so a path like
    # /opt/bsts/bin would pass and die later at kernel startup. A directory is an
    # error in the override itself, so pointing at the image would misdirect.
    if Path(kernel_python).is_dir():
        return KernelRouting(
            f"overrides.yaml routes this notebook to {kernel_python}, "
            f"which is a directory, not an executable file. Correct the kernel_python entry."
        )

    image = overrides.get("docker_env")
    rebuild = (
        f"Rebuild or repull the {image} image." if image else "Rebuild the image that provides it."
    )

    if not (Path(kernel_python).is_file() and os.access(kernel_python, os.X_OK)):
        return KernelRouting(
            f"overrides.yaml routes this notebook to {kernel_python}, "
            f"which is not an executable file here. {rebuild}"
        )

    launcher_path = REPO_ROOT / launcher if launcher else None
    if launcher_path is not None and not launcher_path.is_file():
        return KernelRouting(
            f"overrides.yaml routes this notebook through the kernel launcher {launcher}, "
            f"which does not exist at {launcher_path}. The kernel would die at startup "
            f"with an unrelated papermill error."
        )
    return KernelRouting(None, kernel_python, launcher_path)


def register_kernelspec(python_exe: str, launcher: Path | None = None) -> tuple[str, Path]:
    """Write a throwaway kernelspec that runs a specific interpreter.

    The py312 image keeps `tfcausalimpact` in an isolated `/opt/bsts` venv because
    it needs NumPy 1.x while the signature stack on `/opt/ml4t` needs 2.x. The
    kernelspec baked into the image points at `/app/envs/py312/bsts_kernel.py`,
    which only resolves under docker-compose; CI checks the repo out elsewhere.
    Generating the spec here keeps the launcher path anchored to REPO_ROOT, so
    the same code works in both.

    Args:
        python_exe: Interpreter to run the kernel with
        launcher: Optional kernel launcher script (defaults to plain ipykernel)

    Returns:
        (kernel_name, jupyter_path) — set JUPYTER_PATH to the second value.
    """
    kernel_name = "ml4t-scoped"
    root = Path(tempfile.mkdtemp(prefix="_pm_kernels_"))
    spec_dir = root / "kernels" / kernel_name
    spec_dir.mkdir(parents=True)

    if launcher is not None:
        argv = [python_exe, str(launcher), "-f", "{connection_file}"]
    else:
        argv = [python_exe, "-m", "ipykernel_launcher", "-f", "{connection_file}"]

    (spec_dir / "kernel.json").write_text(
        json.dumps({"argv": argv, "display_name": kernel_name, "language": "python"})
    )
    return kernel_name, root


def research_preview_parameters(
    py_path: Path,
    parameters: dict | None,
    output_dir: Path | None,
) -> dict:
    """Route migrated Study notebooks to the isolated reduced-run workspace."""
    resolved = dict(parameters or {})
    if output_dir is None:
        return resolved
    source = py_path.read_text(encoding="utf-8")
    parameter_bounds = next(
        (
            (first_line, last_line)
            for header, first_line, last_line in _percent_cell_bounds(source)
            if PARAMETERS_CELL_MARKER in header
        ),
        None,
    )
    if parameter_bounds is None:
        return resolved
    first_line, last_line = parameter_bounds
    tree = ast.parse(source, filename=str(py_path))
    declared = {name for name, line in _top_level_bindings(tree) if first_line <= line <= last_line}
    if {"EXECUTION_TIER", "WORKSPACE"} <= declared:
        resolved["EXECUTION_TIER"] = "preview"
        resolved["WORKSPACE"] = str(output_dir.resolve())
    if "PREVIEW_REDUCTIONS" in declared:
        resolved = _collect_preview_reductions(resolved)
    return resolved


def _collect_preview_reductions(parameters: dict) -> dict:
    """Fold the per-notebook reduction overrides into the single parameter that carries them.

    A model notebook takes its reductions as one ``PREVIEW_REDUCTIONS`` mapping, because a
    preview request has to declare every reduction it applies for the recorded identity to
    describe what was actually fitted. ``overrides.yaml`` still states them one per line, the way
    it does for every other notebook, so the translation happens here rather than in nine entries
    that would then have to be kept agreeing with each other.
    """
    resolved = dict(parameters)
    reductions = dict(resolved.get("PREVIEW_REDUCTIONS") or {})
    max_folds = resolved.pop("MAX_FOLDS", None)
    max_symbols = resolved.pop("MAX_SYMBOLS", None)
    train_sample_frac = resolved.pop("TRAIN_SAMPLE_FRAC", None)
    if max_folds is not None:
        reductions.setdefault("folds", list(range(int(max_folds))))
    if max_symbols is not None:
        reductions.setdefault("max_symbols", int(max_symbols))
    if train_sample_frac is not None:
        reductions.setdefault("train_sample_frac", float(train_sample_frac))
    # A preview run that reduces nothing is a canonical run wearing the wrong tier, and the
    # request builder rejects it. Reducing the universe is the reduction that always applies.
    if not reductions:
        reductions["max_symbols"] = 5
    resolved["PREVIEW_REDUCTIONS"] = reductions
    return resolved


def injected_parameters(
    py_path: Path,
    parameters: dict | None,
    output_dir: Path | None,
    *,
    research_preview: bool,
) -> dict | None:
    """Return what Papermill should inject, given the tier this run declares.

    ``PREVIEW_REDUCTIONS`` only means anything under the preview tier, and the request
    builders reject it outright on a canonical one. An override file declares it once and
    both paths read that file: ``tests/generate_intermediates.py`` passes the same entry
    with ``research_preview=False``, so leaving it in fails at request construction
    instead of reducing anything.
    """
    if research_preview:
        return research_preview_parameters(py_path, parameters, output_dir)
    if parameters and "PREVIEW_REDUCTIONS" in parameters:
        return {k: v for k, v in parameters.items() if k != "PREVIEW_REDUCTIONS"}
    return parameters


def run_notebook(
    py_path: Path,
    parameters: dict | None = None,
    timeout: int = 300,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
    kernel_python: str | None = None,
    kernel_launcher: Path | None = None,
    research_preview: bool = False,
) -> dict:
    """Execute a notebook via Papermill with parameter injection.

    This is the core test helper. It:
    1. Syncs .py -> .ipynb via Jupytext
    2. Executes via Papermill with parameter overrides
    3. Logs per-cell progress to log_path (if provided)
    4. Returns status and error info

    Args:
        py_path: Path to the .py notebook source
        parameters: Dict of parameters to inject (overrides defaults in parameters cell)
        timeout: Per-cell timeout in seconds
        output_dir: Directory for ML4T_OUTPUT_DIR (redirects saves to temp)
        data_dir: Directory for ML4T_DATA_PATH (test data location)
        extra_env: Additional environment variables for notebook execution
        log_path: Path to progress log file (appended to)
        kernel_python: Interpreter to execute with, when the notebook needs an
            environment other than the one running pytest
        kernel_launcher: Optional launcher script for that interpreter
        research_preview: Inject preview tier and workspace for migrated Study notebooks

    Returns:
        Dict with keys: status ("ok" or "error"), error (str if failed),
        duration_s (float), n_cells (int)
    """
    import time

    import papermill as pm

    start = time.time()
    nb_name = py_path.stem
    parameters = injected_parameters(
        py_path, parameters, output_dir, research_preview=research_preview
    )

    def _log(msg: str) -> None:
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                f.flush()

    _log(f"START {nb_name} (timeout={timeout}s)")

    # Sync to a temp .ipynb (never overwrites the real .ipynb)
    tmp_ipynb: Path | None = None
    try:
        tmp_ipynb = sync_notebook(py_path)
    except (RuntimeError, FileNotFoundError) as e:
        _log(f"SYNC_FAIL {nb_name}: {e}")
        return {
            "status": "error",
            "error": f"Jupytext sync failed: {e}",
            "duration_s": time.time() - start,
            "n_cells": 0,
        }

    ipynb_path = tmp_ipynb

    # Executed notebook output path
    executed_path = py_path.parent / f"_executed_{py_path.stem}.ipynb"

    # Setup environment - Papermill's execute_notebook inherits os.environ,
    # so we temporarily set env vars and restore them after execution.
    saved_env = {}
    env_vars = {
        "MPLBACKEND": "Agg",
        "PLOTLY_RENDERER": "json",
        "DISABLE_HPO": "1",
        **KERNEL_THREAD_CAPS,
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env_vars["ML4T_OUTPUT_DIR"] = str(output_dir)
    if data_dir:
        env_vars["ML4T_DATA_PATH"] = str(data_dir)
    if extra_env:
        env_vars.update({key: str(value) for key, value in extra_env.items()})

    # PYTHONPATH includes repo root for utils imports + notebook dir for sibling imports
    existing = os.environ.get("PYTHONPATH", "")
    nb_dir = str(py_path.parent.resolve())
    env_vars["PYTHONPATH"] = (
        f"{REPO_ROOT}:{nb_dir}:{existing}" if existing else f"{REPO_ROOT}:{nb_dir}"
    )

    # Ensure torch's bundled CUDA libraries are found before system ones.
    # The system libcudart.so.12 may be outdated and missing symbols like
    # cudaGetDriverEntryPointByVersion that torch's bundled version provides.
    try:
        import torch

        torch_lib = str(Path(torch.__file__).parent / "lib")
        nvidia_libs = list((Path(torch.__file__).parent.parent / "nvidia").glob("*/lib"))
        cuda_paths = [torch_lib] + [str(p) for p in nvidia_libs]
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        env_vars["LD_LIBRARY_PATH"] = ":".join(cuda_paths + [existing_ld])
    except ImportError:
        pass

    # A notebook whose dependencies live in a separate venv runs on its own
    # kernelspec, written to a temp JUPYTER_PATH so nothing global is touched.
    kernel_name = "python3"
    kernel_root: Path | None = None
    if kernel_python:
        kernel_name, kernel_root = register_kernelspec(kernel_python, kernel_launcher)
        env_vars["JUPYTER_PATH"] = str(kernel_root)

    remove_vars = ["TEST", "QUICK_TEST"]
    if extra_env:
        for key in extra_env:
            if key in remove_vars:
                remove_vars.remove(key)

    # Apply environment changes
    for key, value in env_vars.items():
        saved_env[key] = os.environ.get(key)
        os.environ[key] = value
    for key in remove_vars:
        saved_env[key] = os.environ.pop(key, None)

    # Cell-level progress — always log to /tmp/ml4t-pm-{name}.log for visibility.
    # Since request_save_on_cell_execute=True, the executed notebook is updated
    # after each cell. Monitor it with: watch -n5 'python -c "import json; ..."'
    progress_log = Path(f"/tmp/ml4t-pm-{nb_name}.log")
    n_cells = 0
    try:
        with open(progress_log, "w") as pf:
            pf.write(f"START {nb_name} timeout={timeout}s params={parameters}\n")

        pm.execute_notebook(
            str(ipynb_path),
            str(executed_path),
            parameters=parameters or {},
            cwd=str(cwd or REPO_ROOT),
            kernel_name=kernel_name,
            execution_timeout=timeout,
            request_save_on_cell_execute=True,
            progress_bar=False,
            log_output=True,
        )
        # Count cells in executed notebook
        try:
            import nbformat

            nb = nbformat.read(str(executed_path), as_version=4)
            n_cells = len([c for c in nb.cells if c.cell_type == "code"])
        except Exception:
            pass

        elapsed = time.time() - start
        msg = f"OK    {nb_name} ({elapsed:.1f}s, {n_cells} cells)"
        _log(msg)
        with open(progress_log, "a") as pf:
            pf.write(f"{msg}\n")
        return {"status": "ok", "error": None, "duration_s": elapsed, "n_cells": n_cells}
    except pm.PapermillExecutionError as e:
        elapsed = time.time() - start
        msg = f"FAIL  {nb_name} cell {e.cell_index} ({e.ename}): {e.evalue} ({elapsed:.1f}s)"
        _log(msg)
        with open(progress_log, "a") as pf:
            pf.write(f"{msg}\n")
        return {
            "status": "error",
            "error": f"Cell {e.cell_index} ({e.ename}): {e.evalue}",
            "duration_s": elapsed,
            "n_cells": e.cell_index,
        }
    except Exception as e:
        elapsed = time.time() - start
        _log(f"FAIL  {nb_name}: {e} ({elapsed:.1f}s)")
        return {"status": "error", "error": str(e), "duration_s": elapsed, "n_cells": 0}
    finally:
        # Restore environment
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Clean up temp input notebook
        if tmp_ipynb is not None and tmp_ipynb.exists():
            tmp_ipynb.unlink()
        # Clean up executed notebook
        if executed_path.exists():
            executed_path.unlink()
        # Clean up the temp kernelspec
        if kernel_root is not None:
            shutil.rmtree(kernel_root, ignore_errors=True)


def collect_chapter_notebooks(repo_root: Path, chapter_range: range) -> list[Path]:
    """Collect all teaching notebooks from chapter directories.

    NOTE: Review repo has flat layout — notebooks live directly in
    chapter directories (e.g., 05_synthetic_data/01_timegan.py),
    NOT in code/ subdirectories.

    Args:
        repo_root: Repository root path
        chapter_range: Range of chapter numbers to include

    Returns:
        Sorted list of .py notebook paths
    """
    notebooks = []
    for chapter_dir in sorted(repo_root.glob("[0-9][0-9]_*")):
        if not chapter_dir.is_dir():
            continue

        # Extract chapter number from directory name
        try:
            ch_num = int(chapter_dir.name[:2])
        except ValueError:
            continue

        if ch_num not in chapter_range:
            continue

        # Review repo: notebooks are directly in the chapter directory
        for notebook in sorted(chapter_dir.glob("*.py")):
            # Skip non-notebook files — use startswith to avoid false positives
            # (e.g., "test_" must not match "backtest_")
            if any(
                notebook.name.startswith(prefix)
                for prefix in [
                    "test_",
                    "conftest",
                    "extract_book_figures",
                    "export_figures",
                    "batch_",
                ]
            ) or any(x in notebook.name for x in ["__pycache__", "__init__"]):
                continue

            # Skip archived/draft/reserved directories
            if any(
                x in str(notebook)
                for x in ["_archive", "archived", "drafts", "inventory", "_reserved"]
            ):
                continue

            # Skip helper/utility files (start with _)
            if notebook.name.startswith("_"):
                continue

            if not notebook.name[0].isdigit() and not notebook.with_suffix(".ipynb").exists():
                continue

            notebooks.append(notebook)

    return notebooks
