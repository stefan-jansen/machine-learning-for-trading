#!/usr/bin/env python3
"""An int constant subscripting a container built from a second constant.

`STATE_COUNTS = [2, 3, 4]` fills `fitted_models`, `N_STATES = 2` reads it back as
`fitted_models[N_STATES]`, and nothing anywhere says the two agree. Edit either one -
which the notebooks invite, some of them from inside the papermill `parameters` cell -
and the notebook raises `KeyError` the next time it executes. ruff, the formatter,
`py_compile`, the prose checker and an AST audit all pass, and `tests/overrides.yaml`
moves neither constant, so CI stays green until somebody runs the notebook.

The class is only decidable under a filter, and the filter is the rule: a **module-level
integer** constant used as a **bare positional subscript** into a container this module
builds from constants. Both halves of that filter earn their place. Dropping the integer
half admits string keys into dicts a reader never edits - a different failure with a
different fix. Dropping the "container this module builds" half takes the sweep from 6
hits to 18, and the extra twelve are all `array[idx]` with `idx = 0` into a frame whose
length the data decides: there is no second constant to disagree with.

What clears a site is a guard the interpreter runs: `assert N in CONTAINER`, or
`if N not in CONTAINER: raise`. Deriving one constant from the other does NOT clear it and
is the wrong fix in a papermill notebook: injection happens *after* the tagged cell, so
`HEADLINE = SIZES[1]` computed inside that cell keeps its default when `SIZES` is
overridden, turning a loud `KeyError` into a silent headline drawn from a configuration
that never ran.

Stdlib only, no imports executed. `--selftest` proves each rule fires and does not fire.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

# A jupytext percent notebook is ordinary Python: the cell markers are comments, so the
# whole file parses and cell boundaries are irrelevant to this rule. Module level here
# means the module body, which is where every notebook constant lives.


@dataclass
class Container:
    """A module-level name whose admissible subscripts this module fixes."""

    name: str
    line: int
    keys: frozenset[int]
    kind: str  # "mapping" | "sequence"
    built_from: str | None = None  # the second constant, when built in a loop

    def admits(self, key: int) -> bool:
        if self.kind == "sequence":
            return -len(self.keys) <= key < len(self.keys)
        return key in self.keys

    def describe(self) -> str:
        if self.built_from:
            return f"{self.name} (line {self.line}, filled from {self.built_from})"
        return f"{self.name} (line {self.line})"


@dataclass
class Finding:
    path: Path
    line: int
    const: str
    const_line: int
    container: str
    detail: str
    rule: str = "unchecked-subscript"

    def __str__(self) -> str:
        if self.rule == "derived-parameter":
            return f"{self.path}:{self.line}: {self.detail}"
        return (
            f"{self.path}:{self.line}: {self.container}[{self.const}] - {self.const} is declared "
            f"at line {self.const_line} and nothing checks it against {self.container}. {self.detail}"
        )


@dataclass
class Module:
    int_consts: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (value, line)
    int_seqs: dict[str, tuple[list[int], int]] = field(default_factory=dict)
    row_lists: dict[str, dict[str, tuple[list[int], str]]] = field(default_factory=dict)
    keyed_by: dict[str, tuple[int, str]] = field(default_factory=dict)
    containers: dict[str, Container] = field(default_factory=dict)
    guarded: set[tuple[str, str]] = field(default_factory=set)  # (const, container)


def _int_literal(node: ast.expr) -> int | None:
    """The int a node evaluates to, or None. Booleans are not integers here."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    # `-1` parses as UnaryOp(USub, Constant(1)).
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_literal(node.operand)
        return None if inner is None else -inner
    return None


def _int_elements(node: ast.expr) -> list[int] | None:
    """The int list a display evaluates to, or None if any element is not an int literal."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    out = []
    for element in node.elts:
        value = _int_literal(element)
        if value is None:
            return None
        out.append(value)
    return out


def _single_name(targets: list[ast.expr]) -> str | None:
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def _collect_bindings(body: list[ast.stmt], mod: Module) -> None:
    """Module-level constants and the containers built from them, in source order.

    Order matters: a container filled by a loop is only recognised once the list it loops
    over has been seen, which is the same order the interpreter runs them in.
    """
    for stmt in body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            name = _single_name(targets)
            value = stmt.value
            if name is None or value is None:
                continue
            literal = _int_literal(value)
            if literal is not None:
                mod.int_consts[name] = (literal, stmt.lineno)
                continue
            elements = _int_elements(value)
            if elements:  # an empty display declares nothing; it is a container awaiting a fill
                mod.int_seqs[name] = (elements, stmt.lineno)
                mod.containers[name] = Container(
                    name, stmt.lineno, frozenset(range(len(elements))), "sequence"
                )
                continue
            keyed = _keyed_call(value, mod)
            if keyed is not None:
                keys, source = keyed
                mod.containers[name] = Container(
                    name, stmt.lineno, frozenset(keys), "mapping", built_from=source
                )
                continue
            if isinstance(value, ast.List) and not value.elts:
                mod.row_lists.setdefault(name, {})
                continue
            if _indexed_frame(value, mod) is not None:
                keys, source = _indexed_frame(value, mod)  # type: ignore[misc]
                mod.containers[name] = Container(
                    name, stmt.lineno, frozenset(keys), "mapping", built_from=source
                )
                continue
            if isinstance(value, ast.Dict):
                keys = [_int_literal(k) for k in value.keys if k is not None]
                if keys and all(k is not None for k in keys):
                    mod.containers[name] = Container(
                        name,
                        stmt.lineno,
                        frozenset(keys),
                        "mapping",  # type: ignore[arg-type]
                    )
                elif not value.keys:
                    # An empty dict is a container waiting for a loop to fill it.
                    mod.containers[name] = Container(name, stmt.lineno, frozenset(), "mapping")
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id == "dict" and not value.args and not value.keywords:
                    mod.containers[name] = Container(name, stmt.lineno, frozenset(), "mapping")
        elif isinstance(stmt, ast.For):
            _learn_loop_fill(stmt, mod)
            _learn_row_append(stmt, mod)
        elif isinstance(stmt, ast.FunctionDef):
            _learn_keyed_factory(stmt, mod)


def _learn_keyed_factory(func: ast.FunctionDef, mod: Module) -> None:
    """A function that fills a dict in a loop over one of its parameters and returns it.

    `fit_gmm_grid(x, n_components_list)` does exactly that, so `gmm_grid`'s keys are the
    list handed to it and `gmm_grid[N_REGIMES_SELECTED]` is the same coupling as the
    in-line loop, with the fill one call away. The sweep that filed this class missed that
    instance for precisely this reason, so the indirection is worth resolving.
    """
    returned = {
        stmt.value.id
        for stmt in ast.walk(func)
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name)
    }
    if not returned:
        return
    params = [a.arg for a in func.args.args] + [a.arg for a in func.args.kwonlyargs]
    for node in ast.walk(func):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.iter, ast.Name) or node.iter.id not in params:
            continue
        var = node.target.id
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Assign):
                continue
            for target in inner.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in returned
                    and isinstance(target.slice, ast.Name)
                    and target.slice.id == var
                ):
                    mod.keyed_by[func.name] = (params.index(node.iter.id), node.iter.id)
                    return


def _keyed_call(value: ast.expr, mod: Module) -> tuple[list[int], str] | None:
    """`gmm_grid = fit_gmm_grid(x, N_REGIMES_GRID)` -> the keys of `N_REGIMES_GRID`."""
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return None
    spec = mod.keyed_by.get(value.func.id)
    if spec is None:
        return None
    position, param = spec
    argument: ast.expr | None = None
    for keyword in value.keywords:
        if keyword.arg == param:
            argument = keyword.value
    if argument is None and len(value.args) > position:
        argument = value.args[position]
    if not isinstance(argument, ast.Name) or argument.id not in mod.int_seqs:
        return None
    return mod.int_seqs[argument.id][0], argument.id


def _indexed_frame(value: ast.expr, mod: Module) -> tuple[list[int], str] | None:
    """`pd.DataFrame(rows).set_index("block_size")` where `rows` carried the loop variable.

    The index of the frame is then exactly the constant list the loop ran over, so
    `frame.loc[CONST]` is the same coupling as `dict[CONST]` with a pandas spelling. This
    is the shape the reader-reachable instance takes, so a rule that only matches
    `Name[Name]` reports the two mild cases and misses the severe one.
    """
    column = None
    for node in ast.walk(value):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_index"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            column = node.args[0].value
            break
    if column is None:
        return None
    for node in ast.walk(value):
        if isinstance(node, ast.Name) and node.id in mod.row_lists:
            carried = mod.row_lists[node.id].get(column)
            if carried is not None:
                return carried
    return None


def _learn_row_append(loop: ast.For, mod: Module) -> None:
    """`for n in SIZES: rows.append({"block_size": n, ...})` records which column carries n."""
    if not isinstance(loop.target, ast.Name):
        return
    if not isinstance(loop.iter, ast.Name) or loop.iter.id not in mod.int_seqs:
        return
    var = loop.target.id
    values, _ = mod.int_seqs[loop.iter.id]
    for node in ast.walk(loop):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != "append"
            or not isinstance(node.func.value, ast.Name)
            or node.func.value.id not in mod.row_lists
            or len(node.args) != 1
            or not isinstance(node.args[0], ast.Dict)
        ):
            continue
        payload = node.args[0]
        for key, item in zip(payload.keys, payload.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(item, ast.Name)
                and item.id == var
            ):
                mod.row_lists[node.func.value.id][key.value] = (values, loop.iter.id)


def _learn_loop_fill(loop: ast.For, mod: Module) -> None:
    """`for n in STATE_COUNTS: fitted[n] = ...` gives `fitted` the keys of STATE_COUNTS."""
    if not isinstance(loop.target, ast.Name):
        return
    if not isinstance(loop.iter, ast.Name) or loop.iter.id not in mod.int_seqs:
        return
    var = loop.target.id
    values, _ = mod.int_seqs[loop.iter.id]
    for node in ast.walk(loop):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and isinstance(target.slice, ast.Name)
                and target.slice.id == var
                and target.value.id in mod.containers
            ):
                existing = mod.containers[target.value.id]
                mod.containers[target.value.id] = Container(
                    existing.name,
                    existing.line,
                    existing.keys | frozenset(values),
                    "mapping",
                    built_from=loop.iter.id,
                )


def _collect_guards(tree: ast.Module, mod: Module) -> None:
    """A guard is a statement the interpreter runs that fails loudly on disagreement.

    Two shapes, and only these two: `assert CONST in CONTAINER` and
    `if CONST not in CONTAINER: raise ...`. Membership in the *source* list counts as
    much as membership in the container it fills, because that is the same assertion.
    """
    aliases: dict[str, set[str]] = {}
    for name, container in mod.containers.items():
        aliases.setdefault(name, {name})
        if container.built_from:
            aliases[name].add(container.built_from)

    def record(comparison: ast.Compare, want: type) -> None:
        if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], want):
            return
        left, right = comparison.left, comparison.comparators[0]
        if isinstance(right, ast.Attribute) and right.attr == "index":
            right = right.value  # `CONST in frame.index` guards `frame.loc[CONST]`
        if not isinstance(left, ast.Name) or not isinstance(right, ast.Name):
            return
        for container_name, names in aliases.items():
            if right.id in names:
                mod.guarded.add((left.id, container_name))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            record(node.test, ast.In)
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            raises = any(isinstance(inner, ast.Raise) for inner in ast.walk(node))
            if raises:
                record(node.test, ast.NotIn)


def parameters_cell_span(src: str) -> tuple[int, int] | None:
    """The 1-based line range of the papermill `parameters` cell, if the file has one.

    Cell markers are comments, so this is a scan over lines rather than anything the AST
    can answer. The span decides severity, not whether a site is reported: a derived
    binding inside the injection point is silent, the same binding below it is merely
    redundant.
    """
    lines = src.splitlines()
    start = None
    for number, line in enumerate(lines, start=1):
        if not line.startswith("# %%"):
            continue
        if start is not None:
            return start, number - 1
        if "parameters" in line and "tags" in line:
            start = number
    return (start, len(lines)) if start is not None else None


def _derived_parameters(
    body: list[ast.stmt], mod: Module, span: tuple[int, int] | None
) -> list[ast.stmt]:
    """Module-level `NAME = SEQ[<int literal>]` where SEQ is a constant list this module declares.

    This is the fix a reader reaches for after reading an `unchecked-subscript` finding,
    and in a papermill notebook it is worse than the coupling it replaces. Injection
    happens AFTER the tagged cell, so a headline derived inside that cell keeps the
    default when the list is overridden: the `KeyError` that would have stopped the run
    becomes a number quietly drawn from a configuration that never executed.
    """
    hits = []
    for stmt in body:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if _single_name(targets) is None or stmt.value is None:
            continue
        value = stmt.value
        if not isinstance(value, ast.Subscript) or not isinstance(value.value, ast.Name):
            continue
        if value.value.id not in mod.int_seqs:
            continue
        if _int_literal(value.slice) is None:
            continue
        if span is None or not (span[0] <= stmt.lineno <= span[1]):
            continue
        hits.append(stmt)
    return hits


def check_source(path: Path, src: str) -> list[Finding]:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:  # a file that will not parse is a different problem
        return [Finding(path, exc.lineno or 0, "?", 0, "?", f"does not parse: {exc.msg}")]

    mod = Module()
    _collect_bindings(tree.body, mod)
    _collect_guards(tree, mod)

    findings: list[Finding] = []
    seen: set[tuple[int, str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.slice, ast.Name):
            continue
        if isinstance(node.value, ast.Name):
            holder = node.value.id
        elif (
            isinstance(node.value, ast.Attribute)
            and node.value.attr in {"loc", "at"}
            and isinstance(node.value.value, ast.Name)
        ):
            holder = node.value.value.id
        else:
            continue
        container = mod.containers.get(holder)
        const = mod.int_consts.get(node.slice.id)
        if container is None or const is None:
            continue
        if (node.slice.id, holder) in mod.guarded:
            continue
        value, const_line = const
        key = (node.lineno, node.slice.id, holder)
        if key in seen:
            continue
        seen.add(key)
        if container.admits(value):
            detail = (
                f"They agree today ({node.slice.id} = {value}); an edit to either raises at "
                f"execution. Assert the membership after the injection point."
            )
        else:
            detail = (
                f"They do NOT agree: {node.slice.id} = {value} and "
                f"{container.describe()} admits {sorted(container.keys)}. This raises."
            )
        findings.append(
            Finding(path, node.lineno, node.slice.id, const_line, container.describe(), detail)
        )

    span = parameters_cell_span(src)
    for stmt in _derived_parameters(tree.body, mod, span):
        name = _single_name(stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
        source = stmt.value.value.id  # type: ignore[union-attr]
        findings.append(
            Finding(
                path,
                stmt.lineno,
                name or "?",
                stmt.lineno,
                source,
                f"{name} is derived from {source} inside the papermill parameters cell. Papermill "
                f"injects after that cell, so overriding {source} leaves {name} at its default and "
                f"the notebook reports a result for a configuration it never ran. Declare both and "
                f"assert the membership below the injection point.",
                rule="derived-parameter",
            )
        )
    return findings


def check(path: Path) -> list[Finding]:
    return check_source(path, path.read_text(encoding="utf-8", errors="replace"))


CASES: list[tuple[str, str, int]] = [
    (
        "the shape the rule is for: a dict filled from a list, read back by a constant",
        """
STATE_COUNTS = [2, 3, 4]
fitted = {}
for n in STATE_COUNTS:
    fitted[n] = object()
N_STATES = 2
model = fitted[N_STATES]
""",
        1,
    ),
    (
        "the same shape with the assertion that clears it",
        """
STATE_COUNTS = [2, 3, 4]
fitted = {}
for n in STATE_COUNTS:
    fitted[n] = object()
N_STATES = 2
assert N_STATES in STATE_COUNTS
model = fitted[N_STATES]
""",
        0,
    ),
    (
        "cleared by a raise rather than an assert",
        """
SIZES = [3, 21, 42]
table = {}
for s in SIZES:
    table[s] = s
HEADLINE = 21
if HEADLINE not in table:
    raise ValueError("headline block size is not in the sweep")
row = table[HEADLINE]
""",
        0,
    ),
    (
        "a dict literal keyed by ints, read by a constant",
        """
RET_COL = {1: "fwd_1d", 5: "fwd_5d", 20: "fwd_20d"}
KEY_HORIZON = 1
col = RET_COL[KEY_HORIZON]
""",
        1,
    ),
    (
        "already broken: the constant is not a key",
        """
RET_COL = {1: "fwd_1d", 5: "fwd_5d"}
KEY_HORIZON = 7
col = RET_COL[KEY_HORIZON]
""",
        1,
    ),
    (
        "a string subscript is a different class and is out of scope",
        """
RET_COL = {"a": 1, "b": 2}
KEY = "a"
col = RET_COL[KEY]
""",
        0,
    ),
    (
        "a positional index into a list literal, in range",
        """
BLOCK_SIZES = [3, 21, 42]
PICK = 1
size = BLOCK_SIZES[PICK]
""",
        1,
    ),
    (
        "a positional index out of range: already broken",
        """
BLOCK_SIZES = [3, 21]
PICK = 5
size = BLOCK_SIZES[PICK]
""",
        1,
    ),
    (
        "a negative index a list admits",
        """
BLOCK_SIZES = [3, 21, 42]
PICK = -1
size = BLOCK_SIZES[PICK]
""",
        1,
    ),
    (
        "a literal subscript is not a coupling between two names",
        """
BLOCK_SIZES = [3, 21, 42]
size = BLOCK_SIZES[1]
""",
        0,
    ),
    (
        "a loop variable is not a module-level constant",
        """
RET_COL = {1: "a", 5: "b"}
for h in (1, 5):
    col = RET_COL[h]
""",
        0,
    ),
    (
        "a container this module does not build is out of reach of a static rule",
        """
import json

RET_COL = json.loads("{}")
KEY = 1
col = RET_COL[KEY]
""",
        0,
    ),
    (
        "a bool is not an integer parameter",
        """
TABLE = {0: "a", 1: "b"}
FLAG = True
col = TABLE[FLAG]
""",
        0,
    ),
    (
        "a slice is not a positional subscript",
        """
BLOCK_SIZES = [3, 21, 42]
CUT = 2
head = BLOCK_SIZES[:CUT]
""",
        0,
    ),
    (
        "two sites on the same constant are two findings",
        """
RET_COL = {1: "a", 5: "b"}
KEY = 1
first = RET_COL[KEY]
second = RET_COL[KEY]
""",
        2,
    ),
    (
        "the guard names the container rather than the source list",
        """
SIZES = [3, 21, 42]
table = {}
for s in SIZES:
    table[s] = s
HEADLINE = 21
assert HEADLINE in table
row = table[HEADLINE]
""",
        0,
    ),
    (
        "a dict filled by a helper looping over one of its parameters",
        """
GRID = [2, 3, 4, 5, 6]


def fit_grid(x, counts):
    results = {}
    for n in counts:
        results[n] = x
    return results


models = fit_grid(None, GRID)
SELECTED = 2
chosen = models[SELECTED]
""",
        1,
    ),
    (
        "the same helper, with the membership asserted",
        """
GRID = [2, 3, 4, 5, 6]


def fit_grid(x, counts):
    results = {}
    for n in counts:
        results[n] = x
    return results


models = fit_grid(None, GRID)
SELECTED = 2
assert SELECTED in GRID
chosen = models[SELECTED]
""",
        0,
    ),
    (
        "a helper that does not key on its parameter is out of reach",
        """
GRID = [2, 3, 4, 5, 6]


def fit_grid(x, counts):
    results = {}
    for n in counts:
        results[n * 10] = x
    return results


models = fit_grid(None, GRID)
SELECTED = 2
chosen = models[SELECTED]
""",
        0,
    ),
    (
        "a frame indexed by the loop's constant list, read back with .loc",
        """
import pandas as pd

BLOCK_SIZES = [3, 21, 42]
HEADLINE = 21
rows = []
for block_size in BLOCK_SIZES:
    rows.append({"block_size": block_size, "z": 0.0})
sweep = pd.DataFrame(rows).set_index("block_size")
row = sweep.loc[HEADLINE]
""",
        1,
    ),
    (
        "the same frame with the membership asserted against the index",
        """
import pandas as pd

BLOCK_SIZES = [3, 21, 42]
HEADLINE = 21
rows = []
for block_size in BLOCK_SIZES:
    rows.append({"block_size": block_size, "z": 0.0})
sweep = pd.DataFrame(rows).set_index("block_size")
assert HEADLINE in sweep.index
row = sweep.loc[HEADLINE]
""",
        0,
    ),
    (
        "a frame indexed on a column the loop never bound is out of reach",
        """
import pandas as pd

BLOCK_SIZES = [3, 21, 42]
HEADLINE = 21
rows = []
for block_size in BLOCK_SIZES:
    rows.append({"block_size": block_size, "z": 0.0})
sweep = pd.DataFrame(rows).set_index("z")
row = sweep.loc[HEADLINE]
""",
        0,
    ),
    (
        "the anti-fix: a headline derived inside the parameters cell",
        """
# %% tags=["parameters"]
BLOCK_SIZES = [3, 21, 42]
BLOCK_SIZE_HEADLINE = BLOCK_SIZES[1]

# %%
row = BLOCK_SIZES[1]
""",
        1,
    ),
    (
        "the same derivation below the injection point is not silent and is not reported",
        """
# %% tags=["parameters"]
BLOCK_SIZES = [3, 21, 42]

# %%
BLOCK_SIZE_HEADLINE = BLOCK_SIZES[1]
""",
        0,
    ),
    (
        "a notebook with no parameters cell cannot have a silent override",
        """
BLOCK_SIZES = [3, 21, 42]
BLOCK_SIZE_HEADLINE = BLOCK_SIZES[1]
""",
        0,
    ),
    (
        "both rules fire on one file",
        """
# %% tags=["parameters"]
SIZES = [3, 21, 42]
HEADLINE = SIZES[1]

# %%
TABLE = {3: "a", 21: "b", 42: "c"}
PICK = 21
row = TABLE[PICK]
""",
        2,
    ),
]


def selftest() -> int:
    """Every rule must be shown firing AND not firing. A check with only no-hit evidence
    cannot fail, which is evidence of nothing."""
    failures = 0
    for label, src, expected in CASES:
        got = check_source(Path("<selftest>"), src)
        if len(got) != expected:
            failures += 1
            print(f"FAIL {label}: expected {expected} finding(s), got {len(got)}")
            for finding in got:
                print(f"      {finding}")
        else:
            print(f"ok   {label}")
    fired = sum(1 for _, src, n in CASES if n)
    clean = len(CASES) - fired
    print(f"\n{len(CASES)} cases: {fired} that must fire, {clean} that must not, {failures} failed")
    return 1 if failures else 0


def notebook_sources(roots: list[Path]) -> list[Path]:
    """Every paired notebook source, and nothing else.

    A jupytext header is what makes a `.py` a notebook. Dot-directories are skipped:
    `.venv` alone is tens of thousands of files and none of them is a notebook.
    """
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part.startswith(".") for part in path.parts):
                continue
            head = path.read_text(encoding="utf-8", errors="replace")[:2048]
            if "jupytext:" in head or "# %%" in head:
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
    findings: list[Finding] = []
    for path in notebook_sources(roots):
        findings += check(path)

    for finding in findings:
        try:  # a path under the repo reads better relative; one outside it stays absolute
            finding.path = finding.path.resolve().relative_to(repo)
        except ValueError:
            pass
        print(finding)
    if findings:
        print(
            f"\n{len(findings)} constant subscript(s) with nothing checking the two constants agree"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
