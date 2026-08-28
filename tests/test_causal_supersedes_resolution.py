"""A committed causal predecessor hash has to be right for the reader too.

`_enforce_causal_supersedes` refuses to register a second current identity for a label,
and it refuses at write time - after the DML fit and every placebo refit have been paid
for. The only way a notebook can answer that refusal is to declare the predecessor, so
the hash becomes committed source.

And then it is wrong for everyone who is not the author. `run_log/` is gitignored, so a
reader's clone holds no causal rows at all, and naming a predecessor that does not exist
fails at the same place after the same computation. The author's fix becomes the reader's
bug.

Leaving the default empty and passing the mapping at run time does not solve it either:
`run-production-notebook.sh` executes with no parameter overrides, because the provenance
gate requires the committed notebook to be the current source executed clean. A value that
only ever arrives as an override can never be stamped.

`causal_supersedes` resolves the declaration against the registry instead, which is what
makes one committed value right for both. Same rule as `population_supersedes`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from case_studies.research import causal_supersedes
from case_studies.research.workspace import Study
from tests.test_research_workspace import _seed_release

DECLARATION = '{"fwd_ret_1d": "6e17a9b4644c", "fwd_ret_5d": "e9623aa44d9a"}'
# The declaration names both, so every call has to say the notebook fits both -
# `supersedes_for` refuses a hash for a label the run does not cover, before any fit.
FITTED = ["fwd_ret_1d", "fwd_ret_5d"]


@pytest.fixture
def study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _register(study: Study, causal_hash: str, label: str) -> None:
    """Put one current canonical causal identity in the registry, and nothing else.

    Written directly rather than through a fit: what is under test is how the declaration
    resolves against rows that exist, not how they got there. The tier and the identity
    version live in `spec_json`, which is where `current_causal_identities` reads them, so a
    row without a spec is invisible to the very query this is exercising.
    """
    from case_studies.utils.registry.specs import IDENTITY_VERSION
    from case_studies.utils.registry.store import _open_registry

    db = _open_registry(study.root)
    try:
        db.execute(
            "INSERT OR REPLACE INTO causal_runs "
            "(causal_hash, label, spec_json, supersedes_hash, created_at) VALUES (?,?,?,?,?)",
            (
                causal_hash,
                label,
                json.dumps({"identity_version": IDENTITY_VERSION, "execution_tier": "canonical"}),
                None,
                "2026-08-27T00:00:00+00:00",
            ),
        )
        db.commit()
    finally:
        db.close()


class TestWhenTheDeclaredPredecessorIsOffered:
    def test_a_clean_clone_withholds_it(self, study: Study) -> None:
        """The reader's case, and the one that made this necessary.

        No causal rows at all. Offering the hash makes the fit complete and then die at
        registration naming an identity the registry has never held.
        """
        assert causal_supersedes(study, DECLARATION, "fwd_ret_1d", labels=FITTED) is None

    def test_the_author_holding_that_identity_is_offered_it(self, study: Study) -> None:
        _register(study, "6e17a9b4644c", "fwd_ret_1d")
        assert causal_supersedes(study, DECLARATION, "fwd_ret_1d", labels=FITTED) == "6e17a9b4644c"

    def test_a_registry_holding_a_different_identity_withholds_it(self, study: Study) -> None:
        """`_enforce_causal_supersedes` then refuses and names the hash it needs, which is a
        better answer than this notebook retiring something nobody asked it to."""
        _register(study, "ffffffffffff", "fwd_ret_1d")
        assert causal_supersedes(study, DECLARATION, "fwd_ret_1d", labels=FITTED) is None

    def test_it_resolves_per_label(self, study: Study) -> None:
        """One label's predecessor being present says nothing about another's."""
        _register(study, "6e17a9b4644c", "fwd_ret_1d")
        labels = FITTED
        assert causal_supersedes(study, DECLARATION, "fwd_ret_1d", labels=labels) == "6e17a9b4644c"
        assert causal_supersedes(study, DECLARATION, "fwd_ret_5d", labels=labels) is None

    def test_an_empty_declaration_is_never_offered(self, study: Study) -> None:
        _register(study, "6e17a9b4644c", "fwd_ret_1d")
        assert causal_supersedes(study, "", "fwd_ret_1d", labels=FITTED) is None
        assert causal_supersedes(study, None, "fwd_ret_1d", labels=FITTED) is None


class TestWhatItDoesNotChange:
    def test_a_declaration_naming_an_unfitted_label_still_raises(self, study: Study) -> None:
        """Parsing stays with `supersedes_for`, and a typo is still caught before the fit.

        Resolving against the registry must not turn a declaration the notebook cannot
        honour into a silent no-op - that is the failure the parser's raise exists to
        prevent, and it has to keep happening before anything is paid for.
        """
        with pytest.raises(ValueError, match="does not fit"):
            causal_supersedes(study, DECLARATION, "fwd_ret_1d", labels=["fwd_ret_1d", "other"])


class TestWhenTheRegistryReadItselfFails:
    """An operational error is not evidence that the registry holds nothing.

    Answering every ``sqlite3.OperationalError`` with ``None`` reads a lock timeout, an I/O
    error and a half-migrated schema as the clean clone. The notebook then withholds the
    predecessor it does hold, pays for the DML fit and every placebo refit, and dies at
    registration for naming none. Only the missing table means what the clean-clone branch
    claims it means.
    """

    def _registry_without_the_causal_table(self, study: Study) -> None:
        import sqlite3

        db_path = study.root / "run_log" / "registry.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(db_path)
        db.execute("CREATE TABLE IF NOT EXISTS unrelated (x INTEGER)")
        db.commit()
        db.close()

    def test_a_missing_causal_table_is_still_the_clean_clone(self, study: Study) -> None:
        self._registry_without_the_causal_table(study)
        assert causal_supersedes(study, DECLARATION, "fwd_ret_1d", labels=FITTED) is None

    def test_a_lock_timeout_propagates(self, study: Study, monkeypatch) -> None:
        import sqlite3

        from case_studies.research import causal as causal_module

        self._registry_without_the_causal_table(study)

        def locked(*args: object, **kwargs: object) -> set[str]:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(causal_module, "current_causal_identities", locked)
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            causal_supersedes(study, DECLARATION, "fwd_ret_1d", labels=FITTED)

    def test_the_read_waits_as_long_as_every_other_reader_of_this_file(
        self, study: Study, monkeypatch
    ) -> None:
        """120s on the driver and 60s server-side, the pair `_open_registry` sets.

        A registry is written by notebooks, backfills and scripts at once, so SQLite's
        five-second default is short enough that ordinary contention produced the very
        error the branch above was swallowing.
        """
        import sqlite3

        from case_studies.research import causal as causal_module

        self._registry_without_the_causal_table(study)
        seen: dict[str, object] = {}

        def record(db: sqlite3.Connection, **kwargs: object) -> set[str]:
            seen["busy_timeout"] = db.execute("PRAGMA busy_timeout").fetchone()[0]
            return set()

        monkeypatch.setattr(causal_module, "current_causal_identities", record)
        causal_supersedes(study, DECLARATION, "fwd_ret_1d", labels=FITTED)
        assert seen["busy_timeout"] == 60000
