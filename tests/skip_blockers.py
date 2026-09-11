"""The condition a ``skip: true`` in ``overrides.yaml`` is claiming, in checkable form.

A ``skip_reason`` is prose, and prose does not fail. The suite is green whether the reason is
sound, stale, or was never true, so the notebook behind a wrong reason stays out of CI with
nothing to say so. Three of the reasons on this file were wrong when this was written: one
named a fixture that has been present for months, one described a refusal the notebook does
not raise, and one covered a notebook that passes.

So every skip declares the condition its reason rests on, beside the prose:

    15_causal_estimation/10_case_study_insights:
      skip: true
      skip_reason: "..."
      skip_blocker:
        absent_fixture_path: futures/market/contract_definitions.parquet

``blocker_unmet_reason`` returns a reason while the condition still holds and ``None`` once it
has expired. The skip is honoured only while it returns a reason, so a fixture that grows the
file, an image that gains the module, or a registry that gains a selectable candidate retires
the skip with no edit to any file - and ``tests/test_skip_blockers.py`` fails the build on a
declaration that has expired, so the retirement is a red test rather than a quiet gap.

This is ``tests/awaiting_rebuild.py``'s mechanism pointed at a different registry. That one
measures against ``ML4T_ARTIFACT_ROOT``, the maintainer's own artifacts; ``overrides.yaml``
records at ``case_studies/sp500_options/11_model_analysis`` why that is the wrong instrument
for a skip about the CI fixture - a condition over the maintainer's registry reads as satisfied
on any workstation that has run the models while the fixture the job runs against still holds
nothing. Everything here is measured against the fixture: ``ML4T_DATA_PATH`` for files and the
seeded ``ML4T_OUTPUT_DIR`` for registries.

**A condition has to be a property of the FIXTURE, not of the workspace the job is building.**
The blocker is evaluated at the moment the notebook's test runs, and a ``cs-*`` job runs a case
study's stages in order into one workspace, so anything the earlier stages register is there by
the time a late stage is reached. The first version of this declared
``no_canonical_selection`` for the fx_pairs holdout notebooks; their standalone refusal was
measured correctly, but in the job the stages above them register a selectable validation
rank-1, the blocker expired mid-job, the skip lifted, and all three failed on the condition
their ORIGINAL reason had named - a candidate set the fixture does not ship. A named set or
population the fixture never carries is stable in a way "what the registry currently resolves"
is not.

Two kinds cannot be decided from inside the run and say so rather than pretending:
``external`` (a service, credential or device the runner does not have) and
``fixture_shortfall`` (a reduction that cannot carry what the notebook needs). Both are
inspected by ``test_skip_blockers.py`` - ``external`` must be off the per-commit tier, and
``fixture_shortfall`` must carry the date its reason was last confirmed by running the
notebook.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

DECIDABLE_KINDS = (
    "absent_fixture_path",
    "no_canonical_selection",
    "absent_population",
    "absent_candidate_set",
)
UNDECIDABLE_KINDS = ("external", "fixture_shortfall")
KINDS = DECIDABLE_KINDS + UNDECIDABLE_KINDS


class UndecidableHere(Exception):
    """The blocker is real but cannot be evaluated in this process."""


def declared_kind(declaration: dict) -> str:
    """The single kind a declaration names, or raise."""
    named = [kind for kind in KINDS if kind in declaration]
    if len(named) != 1:
        raise ValueError(
            f"skip_blocker must name exactly one of {', '.join(KINDS)}; got {sorted(declaration)}"
        )
    return named[0]


def _fixture_root() -> Path | None:
    from tests.conftest import _resolve_data_path

    return _resolve_data_path()


def _fixture_registry(case_study: str) -> Path | None:
    """The registry the CI job actually reads, not the maintainer's."""
    output_dir = os.environ.get("ML4T_OUTPUT_DIR")
    if not output_dir:
        return None
    db = Path(output_dir) / case_study / "run_log" / "registry.db"
    return db if db.is_file() else None


def _distinct(db: Path, table: str, column: str) -> set[str] | None:
    """``column``'s distinct values, or None when the table is not there to read."""
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    try:
        if not con.execute(
            "select 1 from sqlite_master where type = 'table' and name = ?", (table,)
        ).fetchone():
            return None
        return {str(row[0]) for row in con.execute(f"select distinct {column} from {table}")}
    except sqlite3.DatabaseError:
        # A registry the current schema cannot read is not one a notebook can read either.
        return None
    finally:
        con.close()


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(name) for name in value]


def blocker_unmet_reason(declaration: dict) -> str | None:
    """Why the skip still applies, or ``None`` once its condition has expired.

    Raises ``UndecidableHere`` when the condition needs something this process does not have -
    a fixture root that is not populated, or a seeded registry that has not been built. The
    caller decides whether that means "honour the skip" (a notebook run) or "report nothing"
    (the guard test), and neither is allowed to read it as "the blocker is gone".
    """
    kind = declared_kind(declaration)

    if kind == "absent_fixture_path":
        root = _fixture_root()
        if root is None:
            raise UndecidableHere("no ML4T_DATA_PATH to measure against")
        relative = str(declaration[kind])
        # A glob, because the condition is usually "no file of this shape" rather than "this
        # exact path is missing", and the two differ where it matters: the IEX notebook reads
        # `iex/deep/*.pcap.gz` through `load_iex_hist(get_raw_files=True)`, so a declaration
        # naming a `raw/` directory it never opens would stay unmet after the captures landed.
        if any(char in relative for char in "*?["):
            if not any(root.glob(relative)):
                return f"the CI fixture carries no file matching {relative}"
            return None
        if not (root / relative).exists():
            return f"the CI fixture carries no {relative}"
        return None

    if kind == "no_canonical_selection":
        case_study = declaration[kind]["of"]
        if _fixture_registry(case_study) is None:
            raise UndecidableHere(f"no seeded registry for {case_study}")
        from case_studies.utils import strategy_analysis

        try:
            strategy_analysis.resolve_canonical_rank1_lineage(case_study)
        except strategy_analysis.NoSelectableCandidates as refusal:
            return f"the fixture registry has no selectable {case_study} candidate: {refusal}"
        return None

    if kind == "absent_population":
        spec = declaration[kind]
        case_study = spec["of"]
        db = _fixture_registry(case_study)
        if db is None:
            raise UndecidableHere(f"no seeded registry for {case_study}")
        wanted = _names(spec["names"])
        present = _distinct(db, "official_populations", "name") or set()
        absent = [name for name in wanted if name not in present]
        if absent:
            return (
                f"the fixture registry has no official population named "
                f"{', '.join(repr(name) for name in absent)}"
            )
        return None

    if kind == "absent_candidate_set":
        spec = declaration[kind]
        case_study = spec["of"]
        db = _fixture_registry(case_study)
        if db is None:
            raise UndecidableHere(f"no seeded registry for {case_study}")
        wanted = _names(spec["names"])
        present = _distinct(db, "candidate_sets", "name") or set()
        absent = [name for name in wanted if name not in present]
        if absent:
            return (
                f"the fixture registry ships no candidate set named "
                f"{', '.join(repr(name) for name in absent)}"
            )
        return None

    if kind == "external":
        return f"the runner has no {declaration[kind]}"

    if kind == "fixture_shortfall":
        spec = declaration[kind]
        return f"{spec['note']} (last confirmed by running it on {spec['verified']})"

    raise AssertionError(f"unreachable kind {kind!r}")


def confirmation_age_days(declaration: dict, *, today: date | None = None) -> int | None:
    """How long ago a ``fixture_shortfall`` was last confirmed, or None for other kinds."""
    if declared_kind(declaration) != "fixture_shortfall":
        return None
    verified = declaration["fixture_shortfall"]["verified"]
    if not isinstance(verified, date):
        verified = date.fromisoformat(str(verified))
    return ((today or date.today()) - verified).days


def skip_declarations(overrides: dict) -> dict[str, dict]:
    """Every ``skip: true`` row, keyed the way ``get_overrides`` keys them."""
    return {
        key: value
        for key, value in overrides.items()
        if isinstance(value, dict) and value.get("skip") is True
    }


def per_commit_tier(row: dict) -> bool:
    """Whether this row would run in the per-commit suite if it were not skipped."""
    return str(row.get("tier", "per_commit")) == "per_commit"


def unknown_keys(declaration: dict, kind: str) -> Sequence[str]:
    return sorted(set(declaration) - {kind})


def honoured_skip_reason(overrides: dict) -> str | None:
    """The reason to skip this notebook, or ``None`` when the skip no longer applies.

    A declared blocker that has expired stops the skip applying, so the notebook runs again
    with no edit to any file - the same self-retiring contract ``awaiting_rebuild`` has. When
    the condition cannot be measured here (no fixture root, no seeded registry), the skip is
    honoured: an unmeasurable condition is not evidence that it has gone.
    """
    if not overrides.get("skip"):
        return None
    reason = overrides.get("skip_reason", "marked skip in overrides")
    declaration = overrides.get("skip_blocker")
    if not declaration:
        return reason
    try:
        if blocker_unmet_reason(declaration) is None:
            return None
    except UndecidableHere:
        return reason
    return reason
