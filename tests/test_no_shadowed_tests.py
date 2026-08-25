"""No test file defines the same test twice.

A second `def test_x` in the same scope silently replaces the first, so the
earlier body never runs and nothing goes red. `us_equities_panel` shipped a
duplicate definition of a passing test in `tests/test_research_models.py`
(#601), where the surviving copy monkeypatched an attribute that no longer
existed and asserted nothing. Neither pytest nor a linter configured for this
repo reports it, and the file's own count of collected tests does not move,
which is why it needs a check of its own.

Scoped to the test corpus: a duplicate helper elsewhere is a style question,
a duplicate test is a test that does not exist.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def _duplicate_definitions(tree: ast.AST) -> list[tuple[str, str, int, int]]:
    """(scope, name, first line, shadowing line) for every redefined test."""
    found: list[tuple[str, str, int, int]] = []
    scopes: list[tuple[str, list[ast.stmt]]] = [("module", tree.body)]  # type: ignore[attr-defined]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            scopes.append((node.name, node.body))
    for scope, body in scopes:
        seen: dict[str, int] = {}
        for stmt in body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not stmt.name.startswith("test"):
                continue
            if stmt.name in seen:
                found.append((scope, stmt.name, seen[stmt.name], stmt.lineno))
            else:
                seen[stmt.name] = stmt.lineno
    return found


def test_no_test_is_defined_twice() -> None:
    offenders: list[str] = []
    files = sorted(TESTS.glob("test_*.py"))
    assert files, f"no test files found under {TESTS}"
    for path in files:
        for scope, name, first, second in _duplicate_definitions(
            ast.parse(path.read_text(encoding="utf-8"))
        ):
            offenders.append(
                f"{path.name}:{second} redefines {name} (first defined at line {first}"
                f"{'' if scope == 'module' else f', in {scope}'}); the first body never runs"
            )
    assert not offenders, "shadowed test definitions:\n  " + "\n  ".join(offenders)
