"""No notebook may silence every warning, and the list of the ones that still do may only shrink.

`rules/notebook-standards.md` clause M2 used to prescribe `warnings.filterwarnings("ignore")`
in every preamble, and 243 files took it up. It was narrowed on 2026-09-08
(ml4t/agent-workspace#1096): suppress the noisy messages by name, filtered by category and
module, and leave visible anything that may report a real problem - a fit that did not
converge, an overflow, a divide-by-zero.

The blanket call cannot simply be deleted everywhere at once. Removing it is a code-cell
change, so the provenance gate marks the notebook stale and it has to be re-executed before
it can be committed - correctly, because the point of the change is that the render starts
showing warnings. That makes this a re-execution programme, and
`.github/ci/blanket-warning-filters.txt` is what keeps it honest in the meantime: a file
carrying the call must be listed, and a listed file must still carry it.

The second half is what makes the list shrink rather than rot. Without it a line stays behind
after its notebook is fixed, and the list stops describing the corpus - the failure that
`.github/ci/unit-test-quarantine.txt` documents from the other direction, where files nobody
had listed ran in no CI job at all.
"""

import ast
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWLIST = REPO_ROOT / ".github/ci/blanket-warning-filters.txt"


def blanket_filter_lines(source: str) -> list[int]:
    """Line numbers of module-level `warnings.filterwarnings("ignore")` calls.

    Module level only, and only the argument-free form. A call that names a category or a
    module is what the standard asks for, and one inside `catch_warnings()` or a function is
    scoped to what it wraps - `tests/test_max_symbols_reduces_a_vectorized_backtest.py` uses
    exactly that to prove a diagnostic survives the blanket filter. Matching the source text
    instead would flag both, and would flag `case_studies/utils/backtest_runner.py`, which
    only quotes the call in a docstring.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr != "filterwarnings":
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "warnings":
            continue
        if call.keywords:
            continue
        if len(call.args) == 1 and getattr(call.args[0], "value", None) == "ignore":
            found.append(node.lineno)
    return found


def tracked_python_files() -> list[pathlib.Path]:
    """Every tracked .py file. `git ls-files` rather than a walk, so `.venv` cannot appear."""
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO_ROOT / f for f in out]


def listed() -> set[str]:
    lines = ALLOWLIST.read_text().splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def carrying() -> set[str]:
    out = set()
    for path in tracked_python_files():
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if blanket_filter_lines(source):
            out.add(path.relative_to(REPO_ROOT).as_posix())
    return out


def test_no_unlisted_file_silences_every_warning() -> None:
    unlisted = sorted(carrying() - listed())
    assert not unlisted, (
        "These files install a blanket warnings.filterwarnings('ignore'), which clause M2 no "
        "longer permits. Filter the noisy messages by name instead, giving a category and a "
        "module, and leave convergence and numerical warnings visible:\n  " + "\n  ".join(unlisted)
    )


def test_no_listed_file_has_already_been_fixed() -> None:
    stale = sorted(listed() - carrying())
    assert not stale, (
        "These files no longer install a blanket filter, so their lines in\n"
        f"{ALLOWLIST.relative_to(REPO_ROOT)} are spent and must be deleted. The list only "
        "means something while it describes the corpus exactly:\n  " + "\n  ".join(stale)
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('import warnings\nwarnings.filterwarnings("ignore")\n', 1),
        ('import warnings\nwarnings.filterwarnings("ignore", category=FutureWarning)\n', 0),
        ('import warnings\nwarnings.filterwarnings("ignore", module="pandas")\n', 0),
        ('import warnings\ndef f():\n    warnings.filterwarnings("ignore")\n', 0),
        ('"""A docstring naming warnings.filterwarnings(\'ignore\')."""\n', 0),
        ('import warnings\nwarnings.filterwarnings("error")\n', 0),
    ],
)
def test_the_detector_reads_the_call_and_not_the_text(source: str, expected: int) -> None:
    assert len(blanket_filter_lines(source)) == expected
