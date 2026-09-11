"""The pre-flight checker reads every ``SUPERSEDES_*`` declaration, and every live lineage.

``scripts/check_supersedes_literals.py`` exists to catch, before a chain is queued, a
declaration that the registry will refuse at write time - after the fit is paid for. It read
two names, ``SUPERSEDES_POPULATION`` and ``SUPERSEDES_CAUSAL``, and only string-valued ones.

Both halves of that were wrong against the committed corpus. There are 24 distinct
``SUPERSEDES_*`` names in ``case_studies/``, the candidate-set ones hold a ``dict`` keyed by
set name, and a dict is not an ``ast.Constant`` - so every candidate-set declaration was
invisible whatever it was called. On 2026-09-11 the checker reported "0 declared literal(s);
0 stale" for ``us_equities_panel`` while ``06_linear.py`` declared two live ones.

The costlier half is the declaration that is ABSENT. ``us_equities_panel/06_linear`` ran with
``SUPERSEDES_SETS = {}``, which gives a literal check nothing to classify, and was refused at
registration after 78 minutes of cold fit and 19.2 h of registered fit across 64 configs:
``a changed candidate set named 'us-equities-fwd-ret-1d-linear-v1' must explicitly supersedes
454f73021f33``. So the check that pays for itself is per live generation, not per literal.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "check_supersedes_literals", REPO / "scripts" / "check_supersedes_literals.py"
)
assert _spec and _spec.loader
checker = importlib.util.module_from_spec(_spec)
sys.modules["check_supersedes_literals"] = checker
_spec.loader.exec_module(checker)


def _registry(tmp_path: Path, rows: list[tuple[str, str, str | None]]) -> Path:
    """A registry holding just the candidate-set lineage table the checker reads."""
    path = tmp_path / "case_studies" / "demo" / "run_log" / "registry.db"
    path.parent.mkdir(parents=True)
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE candidate_set_names (name TEXT NOT NULL, set_hash TEXT NOT NULL, "
        "supersedes_hash TEXT, created_at TEXT NOT NULL, git_commit TEXT, "
        "PRIMARY KEY (name, set_hash))"
    )
    db.executemany(
        "INSERT INTO candidate_set_names VALUES (?, ?, ?, '2026-01-01T00:00:00Z', NULL)", rows
    )
    db.commit()
    db.close()
    return path


FREEZING_NOTEBOOK = """
from case_studies.research.comparison import candidate_set_supersedes

SUPERSEDES_SETS: dict = {}

for label_name in LABELS:
    full_set_name = f"demo-{label_name}-linear-v1"
    candidate_set_supersedes(study, name=full_set_name, declared=SUPERSEDES_SETS.get(full_set_name, ""))
"""


# --- reading the declaration ----------------------------------------------------------


def test_a_dict_valued_declaration_is_read_whatever_it_is_called() -> None:
    """The shape the old reader dropped: not an ast.Constant, and not one of two names."""
    declared = checker._declared_literals(
        'SUPERSEDES_SETS: dict = {"a-v1": "1111"}\n'
        'SUPERSEDES_CANDIDATE_SETS: dict[str, str] = {"b-v1": "2222"}\n'
        'SUPERSEDES_COST_BACKTESTS: str = "3333"\n'
    )
    assert declared["SUPERSEDES_SETS"] == {"a-v1": "1111"}
    assert declared["SUPERSEDES_CANDIDATE_SETS"] == {"b-v1": "2222"}
    assert declared["SUPERSEDES_COST_BACKTESTS"] == "3333"


def test_a_declaration_built_at_runtime_is_skipped_not_guessed() -> None:
    """`literal_eval` fails and the name is reported absent rather than half-read."""
    assert checker._declared_literals("SUPERSEDES_SETS = {NAME: resolve(NAME)}") == {}


def test_a_dict_states_its_own_lineage_names() -> None:
    assert checker._declared_pairs({"a-v1": "1111", "b-v1": ""}) == [("a-v1", "1111")]
    assert checker._declared_pairs("3333") == [(None, "3333")]
    assert checker._declared_pairs("") == []


# --- classifying a declared hash ------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "status"),
    [
        # `candidate_set_supersedes` offers the hash when it is the head or what the head
        # replaced, and withholds it otherwise (research/comparison.py:121).
        ("55d6", "live"),
        ("454f", "live"),
        ("dead", "stale"),
    ],
)
def test_a_candidate_set_literal_is_classified_by_the_resolvers_own_rule(
    tmp_path: Path, declared: str, status: str
) -> None:
    registry = _registry(
        tmp_path, [("demo-linear-v1", "454f", None), ("demo-linear-v1", "55d6", "454f")]
    )
    findings = checker._check_lineage_literal(
        "demo", Path("06_linear.py"), registry, "SUPERSEDES_SETS", [("demo-linear-v1", declared)]
    )
    assert [f.status for f in findings] == [status]
    if status == "stale":
        assert findings[0].remedy == "55d6"


def test_a_dead_hash_is_answered_with_the_head_because_the_dict_names_the_lineage(
    tmp_path: Path,
) -> None:
    """More than the population path can do: the key says where to look when the hash is gone.

    `fx_pairs/14_portfolio_management.py` declares three of these - hashes from before a
    registry rebuild, in no lineage the table holds - and each names a set that does have a
    live head, so the repair is printable instead of being reported as unresolvable.
    """
    registry = _registry(tmp_path, [("demo-linear-v1", "77ab", None)])
    findings = checker._check_lineage_literal(
        "demo", Path("06_linear.py"), registry, "SUPERSEDES_SETS", [("demo-linear-v1", "gone")]
    )
    assert findings[0].status == "stale"
    assert findings[0].remedy == "77ab"


# --- the missing declaration ----------------------------------------------------------


def test_a_live_generation_that_nothing_declares_is_reported(tmp_path: Path) -> None:
    """The half that costs a fit. An empty mapping gives a literal check nothing to read."""
    registry = _registry(tmp_path, [("demo-fwd_ret_5d-linear-v1", "e7b7", None)])
    notebook = tmp_path / "06_linear.py"
    notebook.write_text(FREEZING_NOTEBOOK)

    findings = checker._undeclared_heads("demo", registry, [notebook])

    assert [f.status for f in findings] == ["undeclared"]
    assert findings[0].label == "demo-fwd_ret_5d-linear-v1"
    assert findings[0].remedy == "e7b7"
    # Attributed through the f-string template, which is the only thing that can name a set
    # the dict does not: the missing entry is by definition not a key.
    assert findings[0].notebook == "06_linear.py"


def test_a_name_with_no_generation_is_not_reported(tmp_path: Path) -> None:
    """The boundary. Declaring nothing is CORRECT until a generation exists to supersede.

    `create` refuses a first version that claims to replace one, so a rule keyed on the name
    appearing in source would have failed 06_linear on the run where it was right.
    """
    registry = _registry(tmp_path, [("demo-fwd_ret_1d-linear-v1", "454f", None)])
    notebook = tmp_path / "06_linear.py"
    notebook.write_text(FREEZING_NOTEBOOK)

    reported = {f.label for f in checker._undeclared_heads("demo", registry, [notebook])}
    assert reported == {"demo-fwd_ret_1d-linear-v1"}
    assert "demo-fwd_ret_5d-linear-v1" not in reported


def test_a_declared_generation_is_not_reported(tmp_path: Path) -> None:
    registry = _registry(tmp_path, [("demo-fwd_ret_5d-linear-v1", "e7b7", None)])
    notebook = tmp_path / "06_linear.py"
    notebook.write_text(
        FREEZING_NOTEBOOK.replace(
            "SUPERSEDES_SETS: dict = {}",
            'SUPERSEDES_SETS: dict = {"demo-fwd_ret_5d-linear-v1": "e7b7"}',
        )
    )
    assert checker._undeclared_heads("demo", registry, [notebook]) == []


def test_a_string_declaration_covers_its_lineage_by_hash(tmp_path: Path) -> None:
    """A string names no set, so only its hash can place it - the reason this keys on hashes."""
    registry = _registry(tmp_path, [("demo-fwd_ret_5d-linear-v1", "e7b7", None)])
    notebook = tmp_path / "06_linear.py"
    notebook.write_text(FREEZING_NOTEBOOK + '\nSUPERSEDES_CANDIDATES: str = "e7b7"\n')
    assert checker._undeclared_heads("demo", registry, [notebook]) == []


def test_a_notebook_that_only_reads_a_set_is_not_named_as_its_owner(tmp_path: Path) -> None:
    """A reader is handed whatever generation is in force and supersedes nothing."""
    registry = _registry(tmp_path, [("demo-fwd_ret_5d-linear-v1", "e7b7", None)])
    reader = tmp_path / "19_strategy_analysis.py"
    reader.write_text('x = CandidateSet.one(study, name=f"demo-{label}-linear-v1")\n')

    findings = checker._undeclared_heads("demo", registry, [reader])

    assert [f.status for f in findings] == ["undeclared"]
    assert findings[0].notebook == "-"
    assert "no notebook states this name as a readable template" in findings[0].detail


# --- the degenerate template ----------------------------------------------------------


def test_a_template_with_no_literal_text_is_dropped() -> None:
    """`f"{a}-{b}"` becomes `%-%` and would attribute the whole registry to one notebook."""
    assert checker._freeze_templates('x = f"{a}-{b}"') == set()
    assert checker._freeze_templates('x = f"{a}-linear-v1"') == set()
    assert checker._freeze_templates('x = f"demo-{a}-linear-v1"') == {"demo-%-linear-v1"}
