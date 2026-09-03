"""Skips for notebooks whose inputs the rebuild has not produced yet.

The rebuild regenerates every case-study artifact from scratch. While it is in progress a
notebook downstream of a stage that has not run has nothing to read, and its test fails for a
reason that says nothing about the notebook. Twenty-four of the twenty-nine failures on `main`
are that, and they have made the suite uninformative for a month.

A plain ``skip: true`` in ``overrides.yaml`` would trade one problem for a worse one: a skip
nobody removes is a test that has silently stopped running. So a skip declared here is honoured
only while the registry agrees the input is missing.

    case_studies/fx_pairs/12_model_analysis:
      awaiting_rebuild:
        needs: {family: deep_learning, of: fx_pairs}
        issue: 874

``needs`` names what has to exist: a ``family``, a ``backtest_stage``, the official
``populations`` a notebook resolves by name, or bare ``registry``. The moment the registry holds a complete family, the skip
stops applying and the test runs - so the skip cannot outlive its reason, and nobody has to
remember to delete it. ``test_awaiting_rebuild.py`` fails the build on any entry whose condition
is already satisfied, so a stale declaration is a red test rather than a quiet gap.

The registry is the authority, not a date and not a checklist:
``scripts/case_study_completion.py`` in the agents repo reads the same tables.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

ARTIFACT_ROOT = Path(
    os.environ.get("ML4T_ARTIFACT_ROOT", str(Path.home() / "ml4t" / "artifacts" / "case_studies"))
)


def _registry(case_study: str) -> Path:
    return ARTIFACT_ROOT / case_study / "run_log" / "registry.db"


def family_is_available(case_study: str, family: str, *, label: str | None = None) -> bool:
    """Whether the registry holds at least one predicted configuration for ``family``.

    Predicted, not trained: nasdaq100_microstructure's GBM had fifteen training rows and no
    prediction sets, and a downstream notebook reading that registry finds nothing to read. A
    family that trained and produced nothing is not available.
    """
    db = _registry(case_study)
    if not db.is_file():
        return False
    query = """
        select count(*)
          from training_runs t
          join prediction_sets p on p.training_hash = t.training_hash
         where t.family = ?
    """
    params: list[str] = [family]
    if label:
        query += " and t.label = ?"
        params.append(label)
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return False
    try:
        return bool(con.execute(query, params).fetchone()[0])
    except sqlite3.DatabaseError:
        # A registry the current schema cannot read is not one a notebook can read either.
        return False
    finally:
        con.close()


def backtest_stage_is_available(case_study: str, stage: str) -> bool:
    """Whether the registry holds a backtest run at ``stage`` (signal, allocation, ...)."""
    db = _registry(case_study)
    if not db.is_file():
        return False
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return False
    try:
        return bool(
            con.execute("select count(*) from backtest_runs where stage = ?", (stage,)).fetchone()[
                0
            ]
        )
    except sqlite3.DatabaseError:
        return False
    finally:
        con.close()


def missing_populations(case_study: str, names: list[str]) -> list[str]:
    """Which of ``names`` the registry has no official population for.

    A model-analysis notebook resolves populations by name, and it needs *every* name it
    resolves: three of four present is as unrunnable as none. So the condition is the set,
    not the count, and the reason names what is actually absent rather than saying the
    registry is empty when it is not.

    A registry written before the research boundary has no ``official_populations`` table at
    all, which is not the same as having it and finding nothing, but it answers the same way
    here: either way there is no population for the notebook to resolve.
    """
    db = _registry(case_study)
    if not db.is_file():
        return list(names)
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return list(names)
    try:
        if not con.execute(
            "select 1 from sqlite_master where type = 'table' and name = 'official_populations'"
        ).fetchone():
            return list(names)
        present = {row[0] for row in con.execute("select distinct name from official_populations")}
    except sqlite3.DatabaseError:
        return list(names)
    finally:
        con.close()
    return [name for name in names if name not in present]


def unmet_reason(declaration: dict) -> str | None:
    """The skip reason if the declared input is still missing, else None.

    Returning None is what makes the skip self-expiring: once the rebuild produces the input,
    this stops returning a reason and the test runs again with no edit to any file.
    """
    needs = declaration.get("needs") or {}
    case_study = needs.get("of")
    if not case_study:
        raise ValueError(f"awaiting_rebuild needs.of is required: {declaration!r}")

    if family := needs.get("family"):
        label = needs.get("label")
        if family_is_available(case_study, family, label=label):
            return None
        target = f"{case_study} {family}" + (f" ({label})" if label else "")
        return f"awaiting rebuild: no predicted {target} in the registry"

    if stage := needs.get("backtest_stage"):
        if backtest_stage_is_available(case_study, stage):
            return None
        return f"awaiting rebuild: no {case_study} backtest at stage {stage!r} in the registry"

    if declared := needs.get("populations"):
        names = [declared] if isinstance(declared, str) else list(declared)
        if not names:
            raise ValueError(f"awaiting_rebuild needs.populations is empty: {declaration!r}")
        absent = missing_populations(case_study, names)
        if not absent:
            return None
        return (
            f"awaiting rebuild: {case_study} has no official population named "
            f"{', '.join(repr(name) for name in absent)}"
        )

    if not _registry(case_study).is_file():
        return f"awaiting rebuild: {case_study} has no registry yet"
    return None
