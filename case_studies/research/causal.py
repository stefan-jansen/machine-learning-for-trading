from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from case_studies.utils.causal import classify_refutation
from case_studies.utils.registry.specs import (
    IDENTITY_VERSION,
    SUPPORTED_IDENTITY_VERSIONS,
    training_hash_from_spec,
)
from case_studies.utils.registry.store import current_causal_identities

from .adapters import get_adapter, registered_adapters
from .contracts import ExecutionTier

if TYPE_CHECKING:
    from .workspace import Study


def _has_causal_runs(db: sqlite3.Connection) -> bool:
    """Whether this registry has the table at all.

    A release seeded before any causal run holds a registry with no ``causal_runs`` in it, and
    naming an absent table in a SELECT is an error rather than an empty result. Reaching those
    registries is the point of the fallback below, so the check belongs with it.
    """
    return (
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'causal_runs'"
        ).fetchone()
        is not None
    )


def _registry_paths(study: Study, tier: ExecutionTier) -> list[Path]:
    """Registries a causal lookup at this tier reads, nearest first.

    Canonical includes the release: a workspace opened over one holds an empty ``run_log`` until
    something is written into it, so a released canonical result is only ever found there.
    """
    roots = [study.storage_root(tier)]
    if tier is ExecutionTier.CANONICAL and study.release_case_root != study.root:
        roots.append(study.release_case_root)
    return [path for path in (root / "run_log" / "registry.db" for root in roots) if path.is_file()]


@dataclass(frozen=True)
class CausalResult:
    study: Study
    hash: str
    spec: dict[str, Any]
    metrics: dict[str, Any]
    execution_tier: str

    @classmethod
    def one(
        cls,
        study: Study,
        *,
        label: str,
        execution_tier: str = "canonical",
    ) -> CausalResult:
        """Resolve one causal result by declared label and execution tier."""
        tier = ExecutionTier(execution_tier)

        def _current(db_path: Path) -> list[str]:
            with sqlite3.connect(db_path) as db:
                if not _has_causal_runs(db):
                    return []
                # One derivation, shared with register_causal_run's write-time check. Two
                # copies of this disagreed within an hour of being written, and the shape
                # of that disagreement is a refusal to register for an ambiguity the
                # reader never sees.
                return current_causal_identities(db, label=label, tier=tier.value)

        # Nearest registry holding a result at the requested identity and tier wins outright.
        # Merging them would make a workspace that has re-derived the result under a corrected
        # input read as two current identities against the release's prior one, and refuse; and
        # stopping on any row for the label would let a stale workspace row at an older identity
        # hide a current released one.
        current: list[str] = []
        for db_path in _registry_paths(study, tier):
            current = _current(db_path)
            if current:
                break
        if len(current) != 1:
            hint = ""
            if len(current) > 1:
                hint = (
                    f" ({', '.join(current)}). A refit left more than one live, and none of "
                    "them says which it retires - re-register the newer with SUPERSEDES_CAUSAL "
                    "naming the older."
                )
            raise ValueError(
                f"causal selection for {label!r} resolved to {len(current)} identities{hint}"
            )
        return cls.open(
            study,
            current[0],
            include_preview=tier is ExecutionTier.PREVIEW,
        )

    @classmethod
    def open(
        cls,
        study: Study,
        causal_hash: str,
        *,
        include_preview: bool = False,
    ) -> CausalResult:
        roots = [(study.root, ExecutionTier.CANONICAL.value)]
        # A workspace opened over a release starts with an empty `run_log`, so a canonical
        # artifact registered by the release lives only there. The prediction and backtest
        # catalogs already overlay it; without the same fallback here, a preview run asking for
        # the canonical causal result finds nothing.
        release_root = study.release_case_root
        if release_root != study.root:
            roots.append((release_root, ExecutionTier.CANONICAL.value))
        if include_preview and study.output_root is not None:
            roots.insert(
                0,
                (
                    study.output_root / ".preview" / study.case_study,
                    ExecutionTier.PREVIEW.value,
                ),
            )
        for root, namespace in roots:
            db_path = root / "run_log" / "registry.db"
            if not db_path.is_file():
                continue
            with sqlite3.connect(db_path) as db:
                if not _has_causal_runs(db):
                    continue
                # `refutation_n_successful` arrived with a migration, and this read does
                # not go through the migrating opener - deliberately, because a read that
                # rewrites the schema of a registry it was only asked to look at is a
                # write. Naming the column unconditionally instead raises
                # OperationalError on any registry written before it existed, and the
                # cache probe in `run_resolved_causal_request` catches only KeyError, so
                # the error escapes, the registering write that would have migrated the
                # database never happens, and re-running the notebook fails identically.
                columns = {
                    row[1] for row in db.execute("PRAGMA table_info(causal_runs)").fetchall()
                }
                draws_column = (
                    "refutation_n_successful"
                    if "refutation_n_successful" in columns
                    else "NULL AS refutation_n_successful"
                )
                row = db.execute(
                    "SELECT n_obs, dml_effect, dml_se_hac, p_value_hac, naive_effect, "
                    f"confounding_bias_pct, refutation_p, {draws_column}, spec_json "
                    "FROM causal_runs WHERE causal_hash = ?",
                    (causal_hash,),
                ).fetchone()
            if row is None:
                continue
            spec = json.loads(row[8])
            tier = str(spec.get("execution_tier", namespace))
            return cls(
                study=study,
                hash=causal_hash,
                spec=spec,
                metrics={
                    "n_obs": row[0],
                    "dml_effect": row[1],
                    "dml_se_hac": row[2],
                    "p_value_hac": row[3],
                    "naive_effect": row[4],
                    "confounding_bias_pct": row[5],
                    "refutation_p": row[6],
                    "refutation_n_successful": row[7],
                    # Derived here so every reader gets the same verdict from the same
                    # rule. A p-value alone cannot say whether the draws could have
                    # rejected at all, so a caller that re-applies a bare threshold
                    # publishes "Fails" for runs that were merely underpowered. An
                    # unknown draw count is the same problem one step back: a row written
                    # before the column existed carries NULL there, and classifying it on
                    # the p-value alone reports exactly the verdict this derivation is
                    # here to prevent. No count, no verdict.
                    "refutation_class": (
                        classify_refutation(row[6], row[7])
                        if row[6] is not None and row[7] is not None
                        else None
                    ),
                },
                execution_tier=tier,
            )
        raise KeyError(f"unknown causal result {causal_hash!r}")

    @property
    def complete(self) -> bool:
        """Whether this row holds everything the run it records was asked to produce.

        The refutation is part of that whenever one was asked for. A run whose placebo
        refits mostly failed registers with a null ``refutation_p`` and a null draw
        count, and calling that complete has two consequences, neither visible at the
        row: a reader formats and divides nulls, and - worse - the runner's cache probe
        serves this row to every later run, so the first failure is also the last fit.
        Nothing after it ever recomputes the refutation that went missing.

        ``n_placebo`` is read from the spec rather than inferred from the metrics,
        because "no refutation was asked for" and "one was asked for and did not arrive"
        are different states that look identical in the columns. A configuration
        declaring no placebos is complete without a p-value; one declaring a hundred is
        not.

        The p-value is the test, and ``refutation_n_successful`` deliberately is not.
        That column arrived with a migration and is listed in
        ``MIGRATION_BACKFILLED_COLUMNS``, so a row written before it existed carries NULL
        there whatever its run did. Requiring it would read a schema fact as a missing
        refutation and declare every pre-migration row incomplete, which is the opposite
        error: it would send runs back to refit results that are already on record. A
        reader that needs the count - to derive the p-value's floor, or a verdict - has
        to handle its absence itself, which is what ``refutation_class`` already does.
        """
        refutation = (self.spec.get("computation") or {}).get("refutation") or {}
        requested = int(refutation.get("n_placebo") or 0)
        return (
            self.spec.get("identity_version") in SUPPORTED_IDENTITY_VERSIONS
            and self.metrics.get("n_obs", 0) > 0
            and self.metrics.get("dml_effect") is not None
            and self.metrics.get("dml_se_hac") is not None
            and (requested == 0 or self.metrics.get("refutation_p") is not None)
        )


@dataclass(frozen=True)
class ResolvedCausalRequest:
    study: Study
    method: str
    spec: dict[str, Any]
    _context: Any
    # Alongside the spec and deliberately not in it: the identity is what was fitted,
    # and which earlier identity this run retires is a statement about the registry.
    # Putting it in the spec would change the hash of every run that declares one, so
    # a refit would supersede a row and then not be the row it claimed to be.
    supersedes: str | None = None

    @property
    def identity(self) -> str:
        return training_hash_from_spec(self.spec)

    def run(self) -> CausalResult:
        module = get_adapter("causal", self.method)
        runner = getattr(module, "run_resolved_causal_request", None)
        if runner is None:
            raise NotImplementedError(f"{self.method!r} has no shared causal runner")
        result = runner(self.study, self.spec, self._context, supersedes=self.supersedes)
        if not isinstance(result, CausalResult):
            raise TypeError(
                f"{self.method!r} runner returned {type(result).__name__}, not CausalResult"
            )
        return result


@dataclass(frozen=True)
class CausalRequest:
    study: Study
    method: str
    label: str
    config_name: str
    overrides: dict[str, Any]
    execution_tier: ExecutionTier
    preview_reductions: dict[str, Any]
    supersedes: str | None = None

    @classmethod
    def from_request(cls, study: Study, request: dict[str, Any]) -> CausalRequest:
        supported = {
            "method",
            "label",
            "config_name",
            "overrides",
            "execution_tier",
            "preview_reductions",
            "supersedes",
        }
        unknown = set(request) - supported
        if unknown:
            raise ValueError(f"unsupported causal request fields: {sorted(unknown)}")
        missing = {"method", "label"} - set(request)
        if missing:
            raise ValueError(f"causal request is missing fields: {sorted(missing)}")
        method = str(request["method"])
        available = {binding.name for binding in registered_adapters("causal")}
        if method not in available:
            raise ValueError(f"unsupported causal method {method!r}")
        tier = ExecutionTier(request.get("execution_tier", ExecutionTier.CANONICAL))
        reductions = dict(request.get("preview_reductions") or {})
        if tier is ExecutionTier.PREVIEW and not reductions:
            raise ValueError("preview causal requests must declare every reduction")
        if tier is ExecutionTier.CANONICAL and reductions:
            raise ValueError("canonical causal requests cannot declare preview reductions")
        return cls(
            study=study,
            method=method,
            label=str(request["label"]),
            config_name=str(request.get("config_name", method)),
            overrides=dict(request.get("overrides") or {}),
            execution_tier=tier,
            preview_reductions=reductions,
            supersedes=(str(request["supersedes"]) if request.get("supersedes") else None),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "label": self.label,
            "config_name": self.config_name,
            "overrides": dict(self.overrides),
            "execution_tier": self.execution_tier.value,
            "preview_reductions": dict(self.preview_reductions),
        }
        # `supersedes` is absent on purpose: this dict is what the resolver turns into
        # the spec, and the spec is hashed.

    def resolve(self) -> ResolvedCausalRequest:
        module = get_adapter("causal", self.method)
        resolver = getattr(module, "resolve_causal_request", None)
        if resolver is None:
            raise NotImplementedError(f"{self.method!r} has no shared causal resolver")
        spec, context = resolver(self.study, self.as_dict())
        if spec.get("identity_version") != IDENTITY_VERSION:
            raise ValueError("causal resolver did not produce the current identity version")
        if spec.get("resolved_spec_schema") != "ml4t.resolved-spec/v1":
            raise ValueError("causal resolver did not produce the resolved-spec schema")
        if spec.get("execution_tier") != self.execution_tier.value:
            raise ValueError("causal resolver changed the execution tier")
        return ResolvedCausalRequest(self.study, self.method, spec, context, self.supersedes)

    def run(self) -> CausalResult:
        return self.resolve().run()


def causal_supersedes(
    study: Study,
    declaration: str | None,
    label: str,
    *,
    labels: list[str] | None = None,
    execution_tier: str = "canonical",
) -> str | None:
    """The declared predecessor, offered only where this registry holds it.

    A notebook that has refit under a changed causal identity must name the identity it
    retires, or ``_enforce_causal_supersedes`` refuses the write - after the DML fit and every
    placebo refit have been paid for. So the hash is committed source, and then it is wrong for
    everyone who is not the author: ``run_log/`` is gitignored, a reader's clone holds no causal
    rows at all, and naming a predecessor that does not exist fails at exactly the same place,
    after exactly the same computation. The author's fix becomes the reader's bug.

    Passing the mapping as a run-time parameter instead does not work here, because
    ``run-production-notebook.sh`` executes with no parameter overrides - the provenance gate
    requires the committed notebook to be the current source executed clean - so a value that
    only ever arrives as an override can never be stamped.

    Resolving it against the registry is what makes one committed declaration right for both.
    The hash is offered when this registry holds a current identity for the label, and withheld
    when it does not, which is the clean clone. It is the same rule
    :func:`case_studies.research.population.population_supersedes` applies to a population hash,
    for the same reason.

    Parsing is still :func:`supersedes_for`: this only decides whether the parsed value applies.
    A declaration naming a label the notebook does not fit still raises there, before the fit.
    """
    declared = supersedes_for(declaration, label, labels=labels)
    if not declared:
        return None
    tier = ExecutionTier(execution_tier)
    db_path = study.storage_root(tier) / "run_log" / "registry.db"
    if not db_path.is_file():
        return None
    # The registry's own timeouts, as `population_supersedes` uses: five seconds is short
    # enough that ordinary contention with a concurrent writer raises `database is locked`.
    db = sqlite3.connect(str(db_path), timeout=120.0)
    try:
        db.execute("PRAGMA busy_timeout = 60000")
        try:
            current = current_causal_identities(db, label=label, tier=tier.value)
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error):
                # A lock timeout, an I/O error and a half-migrated schema are not evidence that
                # this registry holds no causal identity. Reading them as a clean clone withholds
                # the predecessor, and the notebook then pays for the DML fit and every placebo
                # refit before registration refuses the write for naming none.
                raise
            # No causal table at all, which is the ordinary state of a reader's clone rather than
            # the exotic one. Same reasoning as `population_supersedes`.
            return None
    finally:
        db.close()
    return declared if declared in current else None


def supersedes_for(
    declaration: str | None, label: str, *, labels: list[str] | None = None
) -> str | None:
    """Read one label's superseded causal identity out of a notebook parameter.

    A refit produces a second canonical identity for the same label and
    ``CausalResult.one`` resolves a label to exactly one, so the run has to say which
    identity it retires. Papermill passes parameters as strings, so the declaration is
    one of three things: empty, meaning nothing is being retired and the fit must leave
    a single current identity on its own; a bare causal hash, for a notebook that fits
    one label; or a JSON object mapping label to hash, for one that fits several.

    A hash declared for a label the notebook does not fit is a typo rather than a
    no-op - the run would proceed, retire nothing, and fail at registration - so it
    raises here, before the fit is paid for.
    """
    text = (declaration or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            mapping = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"SUPERSEDES_CAUSAL is not valid JSON: {text!r}. Give a bare causal hash "
                f'for a single-label notebook, or {{"<label>": "<hash>"}} for several.'
            ) from error
        if not isinstance(mapping, dict):
            raise ValueError(
                f"SUPERSEDES_CAUSAL must be an object mapping label to hash, got {text!r}"
            )
        known = set(labels) if labels is not None else None
        if known is not None:
            unknown = sorted(set(mapping) - known)
            if unknown:
                raise ValueError(
                    f"SUPERSEDES_CAUSAL names {unknown}, which this notebook does not fit. "
                    f"It fits {sorted(known)}."
                )
        value = mapping.get(label)
        return str(value) if value else None
    if labels is not None and len(labels) > 1:
        raise ValueError(
            f"SUPERSEDES_CAUSAL is a bare hash but this notebook fits {sorted(labels)}. "
            f'Use {{"<label>": "<hash>"}} so each label retires its own identity.'
        )
    return text
