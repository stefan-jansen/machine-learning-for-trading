"""Relabel the fold ids a registry was fitted under, without fitting again.

``ml4t-diagnostic`` 0.1.4 emits walk-forward windows chronologically, so fold 0 became the
earliest window where it had been the most recent one. The windows themselves did not move -
the library's own test pins the ``(train_start, train_end, val_start, val_end)`` tuples across
the change - and neither did any fitted value. Only the integer label attached to each window
did. Everything registered before that release therefore carries the old labels, and because
``computation`` is hashed whole, every training identity fitted under them differs from the one
the same configuration resolves to now.

Refitting to recover a relabeling is the expensive mistake ``AGENTS.md`` rule 6 names. This
module does the relabeling instead, and the discipline that makes that safe is reproduction:
every derived value it rewrites, it first recomputes from what is stored and checks against the
stored result. A migration that cannot reproduce the old identity has no business writing a new
one, so each of those checks is a refusal rather than a warning.

Five identity-bearing places carry the fold id, not one:

``computation.cv.folds``
    the ids themselves.
``computation.cv.identity``
    a :func:`value_digest` over the fold rows, so the integer is inside the hashed content.
``computation.expected_prediction_keys.digest``
    a :func:`value_digest` over ``(symbol, timestamp, fold)`` of the eligible validation rows.
    Recomputing it needs that frame, which is why planning reads the prediction artifacts.
``computation.input_data_spec.splits``
    a second copy of the windows with the fold as a string, and a ``fingerprint`` over the
    payload that holds it.
``computation.model.effective_params_by_fold`` and
``computation.task.imbalance.effective_class_weights_by_fold``
    dicts keyed by the fold id.

``case_studies/utils/strategy_analysis.py`` independently lists the first four as what a
holdout refit is allowed to change, for the same reason.

The identity cascade runs past training: a prediction identity derives from its training hash,
a backtest identity from its prediction hash, and an official population's hash from a digest
over its members. So a fold renumber moves every one of those too, and anything holding one of
them and left alone becomes a dangling reference rather than a stale label. The completeness
check at the end is therefore not a list of tables to update - a list is what lets a table added
later be missed - but a scan of every text column in the registry for a surviving reference to
a migrated identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl

from case_studies.utils.artifact_digest import _PARQUET_WRITE_SETTINGS, value_digest

from .specs import (
    backtest_hash_from_parts,
    canonical_json,
    compute_hash,
    prediction_hash_from_parts,
    training_hash_from_spec,
)

# A fold-indexed path component, as every producer writes it: ``fold_3``, ``fold_08``,
# ``fold_3.parquet``. The padding width is captured because it varies between producers
# within one registry - ``models/fold_3.joblib`` beside ``models/<candidate>/fold_08/`` -
# and a rename that normalizes it would break the reader that globbed for the old shape.
_FOLD_COMPONENT = re.compile(r"^fold_(\d+)((?:\.[^.]+)*)$")

# Keys whose integer value *is* a fold id, wherever they appear in a registry-written JSON
# document. Established by scanning every JSON file under one registry's training tree
# rather than by reading the producers: `fold_extras.json` uses `fold_id`, the per-fold
# sidecars and the deep-checkpoint metadata use `fold`.
_FOLD_VALUE_KEYS = frozenset({"fold", "fold_id"})

_SPLIT_FIELDS = ("fold", "train_start", "train_end", "val_start", "val_end")


class FoldRenumberRefusal(Exception):
    """A row, file or table the migration will not guess at."""


class IncompleteTrainingRun(FoldRenumberRefusal):
    """A registered training run holding no results for the migration to carry over.

    Distinguished from a refusal because it is not a defect in the migration's understanding
    of the registry: a run with no registered prediction set has nothing fitted to preserve,
    and leaving its identity on the old numbering leaves a row the next run of that
    configuration will not match and will therefore fit - which is what should happen to a
    run that never finished. It is reported rather than silently skipped.
    """


# ---------------------------------------------------------------------------
# The permutation
# ---------------------------------------------------------------------------


def _boundary(value: Any) -> datetime:
    """Parse a stored window boundary in any of the shapes producers write it in."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise FoldRenumberRefusal(f"unparsable window boundary {value!r}") from exc
    return parsed.replace(tzinfo=None)


def _window(fold: Mapping[str, Any]) -> tuple[datetime, datetime, datetime, datetime]:
    missing = [key for key in _SPLIT_FIELDS[1:] if fold.get(key) is None]
    if missing:
        raise FoldRenumberRefusal(f"fold window is missing {missing}")
    return (
        _boundary(fold["train_start"]),
        _boundary(fold["train_end"]),
        _boundary(fold["val_start"]),
        _boundary(fold["val_end"]),
    )


def derive_fold_permutation(folds: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Map each stored fold id to the id the same window carries under 0.1.4.

    Derived from the row's own stored windows rather than from a config file, so every
    migrated row carries its own proof and the result does not depend on a config that
    has since been rewritten. The new id is the window's position in chronological order,
    which is the contract ``generate_cv_splits`` asserts before it returns.

    Refuses rather than guesses: on a duplicate window, on a stored id set that is not
    ``0..n-1`` (a preview that selected a subset of folds cannot be ranked into contiguous
    ids), and on a set whose ordering by validation start disagrees with its ordering by
    training start, which would mean "chronological" is ambiguous for these windows.
    """
    if len(folds) < 2:
        raise FoldRenumberRefusal("a permutation needs at least two folds")
    by_window: dict[tuple[datetime, ...], int] = {}
    for fold in folds:
        window = _window(fold)
        if window in by_window:
            raise FoldRenumberRefusal(f"two folds share the window {window[0].date()}..")
        by_window[window] = int(fold["fold"])
    stored = sorted(by_window.values())
    if stored != list(range(len(stored))):
        raise FoldRenumberRefusal(f"stored fold ids are not 0..n-1: {stored}")
    by_validation = sorted(by_window, key=lambda w: (w[2], w[3], w[0], w[1]))
    by_training = sorted(by_window, key=lambda w: (w[0], w[1], w[2], w[3]))
    if by_validation != by_training:
        raise FoldRenumberRefusal("window order by validation start disagrees with training start")
    return {by_window[window]: rank for rank, window in enumerate(by_validation)}


#: Fold ids that do not name a walk-forward fold, and so are carried through a renumber
#: unchanged. ``-1`` is written by `case_studies/utils/conformal.py:720` to mean "holdout, no
#: fold partition"; permuting it would turn a statement that a row belongs to no fold into a
#: claim that it belongs to one. Sentinels are honoured in stored data - a parquet column, a
#: fold_extras entry - and not in a specification's `cv.folds`, where one would be a defect
#: rather than a convention.
_NON_FOLD_SENTINELS = frozenset({-1})


def _permute(permutation: Mapping[int, int], fold: Any, *, where: str) -> int:
    if isinstance(fold, bool) or not isinstance(fold, int):
        raise FoldRenumberRefusal(f"{where}: fold id {fold!r} is not an integer")
    if fold not in permutation:
        raise FoldRenumberRefusal(f"{where}: fold id {fold} is outside the permutation")
    return permutation[fold]


def _permute_data(permutation: Mapping[int, int], fold: Any, *, where: str) -> int:
    """Permute a fold id recorded beside a result, letting a declared sentinel through."""
    if isinstance(fold, int) and not isinstance(fold, bool) and fold in _NON_FOLD_SENTINELS:
        return fold
    return _permute(permutation, fold, where=where)


# ---------------------------------------------------------------------------
# Documents and paths
# ---------------------------------------------------------------------------


def remap_json_document(
    node: Any,
    permutation: Mapping[int, int],
    *,
    identity_renames: Mapping[str, str] | None = None,
    where: str = "$",
) -> Any:
    """Rewrite the fold ids in a registry-written JSON document.

    Two shapes carry them and both are rewritten: a ``fold``/``fold_id`` key whose value is
    the integer, and a ``fold_<n>`` key naming a per-fold file. Anything else is copied
    unchanged, so a dict keyed by epoch or by checkpoint value is untouched.

    ``identity_renames`` additionally rewrites any string equal to a migrated identity.
    Artifacts record the training hash inside their own sidecars - the deep-checkpoint
    metadata's ``config_name``, the per-fold summary's ``config`` - and a renamed directory
    holding a sidecar that still names the old identity is a file that disagrees with where
    it lives.

    Specs are *not* remapped through here. Their fold-keyed dicts use the bare id as the key
    (``"0"``, ``"1"``), which no general rule can distinguish from a dict keyed by anything
    else that happens to be small integers; :func:`remap_training_spec` names those places.
    """
    renames = identity_renames or {}
    if isinstance(node, dict):
        remapped: dict[str, Any] = {}
        for key, value in node.items():
            new_key = key
            match = _FOLD_COMPONENT.match(str(key))
            if match is not None:
                new_key = _rename_fold_token(match, permutation, where=f"{where}.{key}")
            if key in _FOLD_VALUE_KEYS and isinstance(value, int) and not isinstance(value, bool):
                new_value: Any = _permute_data(permutation, value, where=f"{where}.{key}")
            else:
                new_value = remap_json_document(
                    value,
                    permutation,
                    identity_renames=renames,
                    where=f"{where}.{key}",
                )
            if new_key in remapped:
                raise FoldRenumberRefusal(f"{where}: remapped key {new_key!r} collides")
            remapped[new_key] = new_value
        return remapped
    if isinstance(node, list):
        return [
            remap_json_document(
                item, permutation, identity_renames=renames, where=f"{where}[{index}]"
            )
            for index, item in enumerate(node)
        ]
    if isinstance(node, str):
        return renames.get(node, node)
    return node


def _rename_fold_token(match: re.Match[str], permutation: Mapping[int, int], *, where: str) -> str:
    digits, suffix = match.group(1), match.group(2)
    new_id = _permute(permutation, int(digits), where=where)
    return f"fold_{new_id:0{len(digits)}d}{suffix}"


def remap_relative_path(
    relative: PurePosixPath,
    permutation: Mapping[int, int],
    identity_renames: Mapping[str, str] | None = None,
) -> PurePosixPath:
    """Rewrite every ``fold_<n>`` component of an artifact path, keeping its zero padding.

    Padding is preserved rather than normalized because it differs between producers inside
    one registry - ``models/fold_3.joblib`` beside ``models/<candidate>/fold_08/`` - and a
    reader that globs for one shape would stop finding the other.
    """
    renames = identity_renames or {}
    parts: list[str] = []
    for part in relative.parts:
        match = _FOLD_COMPONENT.match(part)
        if match is None:
            # An identity names a path in two shapes. A directory is the hash alone -
            # `predictions/<prediction_hash>/`, and inside a training tree the candidate
            # directory `models/<training_hash>/fold_08/`. A snapshot file leads with it
            # instead: `_snapshots/<date>-<name>/<prediction_hash>.conformal_widths.parquet`.
            # Matching only whole components leaves the second shape carrying a retired
            # identity in its name.
            if part in renames:
                parts.append(renames[part])
                continue
            head, dot, rest = part.partition(".")
            parts.append(f"{renames[head]}{dot}{rest}" if head in renames else part)
            continue
        parts.append(_rename_fold_token(match, permutation, where=str(relative)))
    return PurePosixPath(*parts) if parts else relative


# ---------------------------------------------------------------------------
# The training specification
# ---------------------------------------------------------------------------


def _cv_identity(folds: Sequence[Mapping[str, Any]]) -> str:
    return value_digest(pl.DataFrame([dict(fold) for fold in folds]))


def _input_fingerprint(input_data_spec: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in input_data_spec.items() if key != "fingerprint"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expected_keys_digest(frame: pl.DataFrame) -> str:
    return value_digest(frame, ("symbol", "timestamp", "fold"))


def _strip_fold_bearing(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The specification with every place a fold id reaches removed.

    What is left has to be byte-identical between the source and the target. Comparing the
    remainder is what turns "I rewrote the fold ids" into a claim that can fail: a spec that
    differs anywhere else is a different computation and is refused, not migrated.
    """
    reduced = copy.deepcopy(dict(spec))
    computation = reduced.get("computation")
    if not isinstance(computation, dict):
        raise FoldRenumberRefusal("spec has no computation block")
    computation.pop("cv", None)
    computation.pop("expected_prediction_keys", None)
    computation.pop("input_data_spec", None)
    model = computation.get("model")
    if isinstance(model, dict):
        model.pop("effective_params_by_fold", None)
    imbalance = (computation.get("task") or {}).get("imbalance")
    if isinstance(imbalance, dict):
        imbalance.pop("effective_class_weights_by_fold", None)
    return reduced


def _remap_fold_keyed(
    mapping: Mapping[str, Any], permutation: Mapping[int, int], *, where: str
) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in mapping.items():
        try:
            stored = int(key)
        except ValueError as exc:
            raise FoldRenumberRefusal(f"{where}: key {key!r} is not a fold id") from exc
        new_key = str(_permute(permutation, stored, where=where))
        if new_key in remapped:
            raise FoldRenumberRefusal(f"{where}: remapped key {new_key} collides")
        remapped[new_key] = copy.deepcopy(value)
    return dict(sorted(remapped.items(), key=lambda item: int(item[0])))


def remap_training_spec(
    spec: Mapping[str, Any],
    permutation: Mapping[int, int],
    *,
    eligible_keys: pl.DataFrame,
) -> dict[str, Any]:
    """Return *spec* relabeled onto the new fold numbering.

    ``eligible_keys`` is the run's validation prediction frame under the *stored* ids. It is
    required rather than optional because ``expected_prediction_keys.digest`` is taken over
    the fold column, so the new digest cannot be derived from the spec alone - and because
    reproducing the stored digest from that frame is what proves the frame is the one the
    digest was taken over.
    """
    computation = spec.get("computation")
    if not isinstance(computation, dict):
        raise FoldRenumberRefusal("spec has no computation block")

    cv = computation.get("cv")
    if not isinstance(cv, dict) or not isinstance(cv.get("folds"), list):
        raise FoldRenumberRefusal("spec has no computation.cv.folds")
    stored_identity = cv.get("identity")
    if stored_identity is not None and _cv_identity(cv["folds"]) != stored_identity:
        raise FoldRenumberRefusal("computation.cv.identity does not reproduce from its folds")

    expected = computation.get("expected_prediction_keys")
    if not isinstance(expected, dict) or "digest" not in expected:
        raise FoldRenumberRefusal("spec has no computation.expected_prediction_keys.digest")
    if _expected_keys_digest(eligible_keys) != expected["digest"]:
        raise FoldRenumberRefusal(
            "expected_prediction_keys.digest does not reproduce from the prediction frame"
        )

    input_data_spec = computation.get("input_data_spec")
    has_splits = isinstance(input_data_spec, dict) and "splits" in input_data_spec
    if has_splits:
        if _input_fingerprint(input_data_spec) != input_data_spec.get("fingerprint"):
            raise FoldRenumberRefusal("input_data_spec.fingerprint does not reproduce")

    target = copy.deepcopy(dict(spec))
    target_computation = target["computation"]

    folds = [
        {**dict(fold), "fold": _permute(permutation, int(fold["fold"]), where="computation.cv")}
        for fold in cv["folds"]
    ]
    folds.sort(key=lambda fold: fold["fold"])
    target_computation["cv"]["folds"] = folds
    if stored_identity is not None:
        target_computation["cv"]["identity"] = _cv_identity(folds)

    remapped_keys = eligible_keys.with_columns(
        pl.col("fold").replace_strict(dict(permutation), return_dtype=pl.Int64)
    )
    target_computation["expected_prediction_keys"]["digest"] = _expected_keys_digest(remapped_keys)

    if has_splits:
        splits = []
        for split in input_data_spec["splits"]:
            stored_fold = int(split["fold"])
            new_fold = _permute(permutation, stored_fold, where="input_data_spec.splits")
            # The producer writes this copy's fold as a string; keeping the stored type
            # matters because the fingerprint is a canonical dump of the payload.
            splits.append(
                {
                    **dict(split),
                    "fold": str(new_fold) if isinstance(split["fold"], str) else new_fold,
                }
            )
        splits.sort(key=lambda split: int(split["fold"]))
        target_input = target_computation["input_data_spec"]
        target_input["splits"] = splits
        target_input.pop("fingerprint", None)
        target_input["fingerprint"] = _input_fingerprint(target_input)

    model = computation.get("model")
    if isinstance(model, dict) and isinstance(model.get("effective_params_by_fold"), dict):
        target_computation["model"]["effective_params_by_fold"] = _remap_fold_keyed(
            model["effective_params_by_fold"],
            permutation,
            where="computation.model.effective_params_by_fold",
        )
    imbalance = (computation.get("task") or {}).get("imbalance")
    if isinstance(imbalance, dict) and isinstance(
        imbalance.get("effective_class_weights_by_fold"), dict
    ):
        target_computation["task"]["imbalance"]["effective_class_weights_by_fold"] = (
            _remap_fold_keyed(
                imbalance["effective_class_weights_by_fold"],
                permutation,
                where="computation.task.imbalance.effective_class_weights_by_fold",
            )
        )

    if canonical_json(_strip_fold_bearing(spec)) != canonical_json(_strip_fold_bearing(target)):
        raise FoldRenumberRefusal("the relabeled spec differs outside its fold-bearing fields")
    if training_hash_from_spec(target) == training_hash_from_spec(dict(spec)):
        raise FoldRenumberRefusal("the relabeled spec has the identity it started with")
    return target


def remaining_fold_references(document: Any, permutation: Mapping[int, int]) -> list[str]:
    """Paths in *document* that still hold a fold id the permutation would have moved.

    Applied to a migrated spec as the completeness check the field list cannot provide: a
    fold-bearing field added to the specification later is found here rather than silently
    left on the old numbering.
    """
    moved = {stored for stored, new in permutation.items() if stored != new}
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _FOLD_VALUE_KEYS and isinstance(value, int) and value in moved:
                    found.append(f"{path}.{key}")
                if _FOLD_COMPONENT.match(str(key)) is not None:
                    found.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(document, "$")
    return found


# ---------------------------------------------------------------------------
# Identities downstream of the training run
# ---------------------------------------------------------------------------


def remap_backtest_spec(spec: Mapping[str, Any], prediction_map: Mapping[str, str]) -> dict:
    """Rewrite the prediction identity a stored strategy spec records in its metadata.

    ``backtest_config.metadata.prediction_hash`` is inside the hashed strategy spec, so a
    backtest whose prediction identity moved has two places to update and both feed the new
    backtest hash.
    """
    target = copy.deepcopy(dict(spec))
    metadata = (target.get("backtest_config") or {}).get("metadata")
    if isinstance(metadata, dict) and "prediction_hash" in metadata:
        stored = str(metadata["prediction_hash"])
        if stored in prediction_map:
            metadata["prediction_hash"] = prediction_map[stored]
    return target


def remap_population_snapshot(
    snapshot: Mapping[str, Any], member_map: Mapping[str, str]
) -> tuple[dict[str, Any], str]:
    """Return the snapshot with its members relabeled, and the hash it now has."""
    target = copy.deepcopy(dict(snapshot))
    members = target.get("members")
    if not isinstance(members, list):
        raise FoldRenumberRefusal("population snapshot has no member list")
    target["members"] = [member_map.get(str(member), str(member)) for member in members]
    # `supersedes` names the generation this one retires, and it is inside the hashed snapshot.
    # Leaving it while `official_populations.supersedes_hash` moves gives one population two
    # answers about its own lineage, and the hash it is stored under is computed over the
    # stale one.
    superseded = target.get("supersedes")
    if isinstance(superseded, str):
        target["supersedes"] = member_map.get(superseded, superseded)
    return target, compute_hash(canonical_json(target))


def cohort_member_digest(hashes: Iterable[str]) -> str:
    """The digest ``cohort_metrics.member_digest`` holds, over a set of member hashes.

    Duplicated from ``case_studies.utils.uncertainty`` rather than imported: that module
    pulls in the whole uncertainty stack, and a maintenance path that has to run against a
    registry should not depend on it. ``tests/test_fold_renumbering.py`` pins the two
    against each other so the copy cannot drift.
    """
    unique = sorted({str(item) for item in hashes})
    return hashlib.sha256("\n".join(unique).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass
class TrainingRemap:
    """One training run's relabeling, with everything derived from it."""

    source_hash: str
    target_hash: str
    permutation: dict[int, int]
    target_spec: dict[str, Any]
    prediction_map: dict[str, str] = field(default_factory=dict)
    backtest_map: dict[str, str] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "permutation": {str(key): value for key, value in sorted(self.permutation.items())},
            "prediction_map": dict(sorted(self.prediction_map.items())),
            "backtest_map": dict(sorted(self.backtest_map.items())),
        }


@dataclass
class RenumberPlan:
    """What a migration will do, and what it refuses to do."""

    case_study: str
    case_dir: Path
    remaps: list[TrainingRemap] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    population_map: dict[str, str] = field(default_factory=dict)
    cohort_digests: dict[str, str] = field(default_factory=dict)
    unresolved_cohorts: list[str] = field(default_factory=list)

    @property
    def training_map(self) -> dict[str, str]:
        return {remap.source_hash: remap.target_hash for remap in self.remaps}

    @property
    def prediction_map(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for remap in self.remaps:
            merged.update(remap.prediction_map)
        return merged

    @property
    def backtest_map(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for remap in self.remaps:
            merged.update(remap.backtest_map)
        return merged

    def as_json(self) -> dict[str, Any]:
        return {
            "case_study": self.case_study,
            "case_dir": str(self.case_dir),
            "remaps": [remap.as_json() for remap in self.remaps],
            "unchanged_training": sorted(self.unchanged),
            "incomplete_training": sorted(self.incomplete),
            "refusals": list(self.refusals),
            "population_map": dict(sorted(self.population_map.items())),
            "cohort_digests": dict(sorted(self.cohort_digests.items())),
            "unresolved_cohorts": list(self.unresolved_cohorts),
        }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

# Where a fold id is allowed to appear in a stored training specification. The scan below
# reports every place one *does* appear; a path outside this set means the specification has
# grown a fold-bearing field this module does not rewrite, and the run is refused rather than
# migrated onto a partly-renumbered identity. This is the check a list of fields cannot be:
# the list says what to rewrite, the scan says whether the list is still complete.
_KNOWN_SPEC_FOLD_PATHS = frozenset(
    {
        "$.computation.cv.folds[].fold",
        "$.computation.input_data_spec.splits[].fold",
        "$.computation.model.effective_params_by_fold",
        "$.computation.task.imbalance.effective_class_weights_by_fold",
    }
)


def spec_fold_reference_paths(spec: Mapping[str, Any], fold_ids: Sequence[int]) -> set[str]:
    """Every path in *spec* that carries a fold id, with list indices collapsed.

    Three shapes count, and the third is the one a name-based search misses: a ``fold`` or
    ``fold_id`` key, a ``fold_<n>`` key, and a dict whose key set is exactly the run's fold
    ids. ``computation.task.imbalance.effective_class_weights_by_fold`` was found by the
    third rule.
    """
    keyed = {str(fold) for fold in fold_ids}
    found: set[str] = set()

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if keyed and set(node) == keyed:
                found.add(path)
                return
            for key, value in node.items():
                if key in _FOLD_VALUE_KEYS or _FOLD_COMPONENT.match(str(key)) is not None:
                    found.add(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for item in node:
                walk(item, f"{path}[]")

    walk(dict(spec), "$")
    return found


def _eligible_keys(run_log: Path, db: sqlite3.Connection, training_hash: str) -> pl.DataFrame:
    """The ``(symbol, timestamp, fold)`` frame ``expected_prediction_keys`` was taken over."""
    rows = db.execute(
        "SELECT prediction_hash FROM prediction_sets WHERE training_hash = ? AND split = ? "
        "ORDER BY prediction_hash",
        (training_hash, "validation"),
    ).fetchall()
    for (prediction_hash,) in rows:
        path = run_log / "predictions" / str(prediction_hash) / "predictions.parquet"
        if not path.is_file():
            continue
        frame = pl.read_parquet(path)
        if {"symbol", "timestamp", "fold"} <= set(frame.columns):
            return frame.select("symbol", "timestamp", "fold")
    if not rows:
        raise IncompleteTrainingRun("no prediction set is registered against it")
    raise FoldRenumberRefusal(f"{training_hash}: no validation prediction frame on disk")


def plan_fold_renumbering(case_dir: Path) -> RenumberPlan:
    """Work out the whole relabeling without writing anything.

    Every derived value the migration will rewrite is reproduced here first and checked
    against what is stored, so a registry this cannot explain produces refusals rather than
    a plan.
    """
    run_log = (case_dir / "run_log").resolve()
    plan = RenumberPlan(case_study=case_dir.name, case_dir=case_dir)
    db = sqlite3.connect(f"file:{run_log / 'registry.db'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        for row in db.execute(
            "SELECT training_hash, spec_json FROM training_runs ORDER BY training_hash"
        ):
            training_hash = str(row["training_hash"])
            try:
                plan_one = _plan_training_run(run_log, db, training_hash, row["spec_json"])
            except IncompleteTrainingRun as incomplete:
                plan.incomplete.append(f"{training_hash}: {incomplete}")
                continue
            except FoldRenumberRefusal as refusal:
                plan.refusals.append(f"{training_hash}: {refusal}")
                continue
            if plan_one is None:
                plan.unchanged.append(training_hash)
            else:
                plan.remaps.append(plan_one)
        _plan_downstream(db, plan)
    finally:
        db.close()
    return plan


def _plan_training_run(
    run_log: Path, db: sqlite3.Connection, training_hash: str, spec_json: str
) -> TrainingRemap | None:
    spec = json.loads(spec_json)
    if training_hash_from_spec(copy.deepcopy(spec)) != training_hash:
        raise FoldRenumberRefusal("stored training_hash does not reproduce from its spec")
    folds = ((spec.get("computation") or {}).get("cv") or {}).get("folds")
    if not isinstance(folds, list) or not folds:
        raise FoldRenumberRefusal("no computation.cv.folds to renumber")
    fold_ids = [int(fold["fold"]) for fold in folds]
    if len(folds) == 1:
        # A holdout refit carries one derived window whose id sits outside the walk-forward
        # numbering. Nothing about it moves, and inventing a permutation for it would.
        return None
    permutation = derive_fold_permutation(folds)
    if all(stored == new for stored, new in permutation.items()):
        return None
    unknown = spec_fold_reference_paths(spec, fold_ids) - _KNOWN_SPEC_FOLD_PATHS
    if unknown:
        raise FoldRenumberRefusal(f"fold ids in unhandled spec fields: {sorted(unknown)}")

    eligible = _eligible_keys(run_log, db, training_hash)
    target_spec = remap_training_spec(spec, permutation, eligible_keys=eligible)
    target_hash = training_hash_from_spec(copy.deepcopy(target_spec))
    remap = TrainingRemap(training_hash, target_hash, permutation, target_spec)

    for prediction in db.execute(
        "SELECT prediction_hash, checkpoint_kind, checkpoint_value, split "
        "FROM prediction_sets WHERE training_hash = ? ORDER BY prediction_hash",
        (training_hash,),
    ):
        stored = str(prediction["prediction_hash"])
        version = target_spec.get("identity_version")
        parts = (prediction["checkpoint_value"], prediction["split"])
        kwargs = {
            "checkpoint_kind": prediction["checkpoint_kind"],
            "identity_version": version,
        }
        if prediction_hash_from_parts(training_hash, *parts, **kwargs) != stored:
            raise FoldRenumberRefusal(f"prediction {stored} does not reproduce from its parts")
        remap.prediction_map[stored] = prediction_hash_from_parts(target_hash, *parts, **kwargs)

    for backtest in db.execute(
        "SELECT backtest_hash, prediction_hash, spec_json FROM backtest_runs "
        f"WHERE prediction_hash IN ({','.join('?' * len(remap.prediction_map))}) "
        "ORDER BY backtest_hash",
        tuple(remap.prediction_map),
    ):
        stored = str(backtest["backtest_hash"])
        strategy = json.loads(backtest["spec_json"])
        # `backtest_hash_from_parts` reads the version off the strategy spec itself. A stored
        # spec carrying `version: 2` is the strategy schema version, not an identity version,
        # and passing it here would hash the backtest down the versioned path a legacy row was
        # never hashed under.
        version = strategy.get("identity_version")
        if (
            backtest_hash_from_parts(
                str(backtest["prediction_hash"]), strategy, identity_version=version
            )
            != stored
        ):
            raise FoldRenumberRefusal(f"backtest {stored} does not reproduce from its parts")
        target_strategy = remap_backtest_spec(strategy, remap.prediction_map)
        remap.backtest_map[stored] = backtest_hash_from_parts(
            remap.prediction_map[str(backtest["prediction_hash"])],
            target_strategy,
            identity_version=version,
        )
    return remap


def _plan_downstream(db: sqlite3.Connection, plan: RenumberPlan) -> None:
    """Populations and cohort digests, which are taken over sets of migrating identities."""
    prediction_map = plan.prediction_map
    backtest_map = plan.backtest_map

    populations = db.execute(
        "SELECT population_hash, member_kind, snapshot_json FROM official_populations "
        "ORDER BY created_at"
    ).fetchall()
    for population in populations:
        stored = str(population["population_hash"])
        snapshot = json.loads(population["snapshot_json"])
        if compute_hash(canonical_json(snapshot)) != stored:
            plan.refusals.append(f"population {stored}: snapshot does not reproduce its hash")
            continue
        members = prediction_map if population["member_kind"] == "prediction" else backtest_map
        # A snapshot names the generation it retires, so a population whose ancestor moved
        # has to be remapped after that ancestor. The rows are read in creation order, which
        # is the order they were chained in.
        chained = dict(members)
        chained.update(plan.population_map)
        _, target = remap_population_snapshot(snapshot, chained)
        if target != stored:
            plan.population_map[stored] = target

    for cohort in db.execute(
        "SELECT cohort_type, stage, label, family, leader_hash, member_digest FROM cohort_metrics "
        "WHERE member_digest IS NOT NULL"
    ):
        key = (
            f"{cohort['cohort_type']}/{cohort['stage']}/{cohort['label']}/{cohort['family']}"
            f" (leader {cohort['leader_hash']})"
        )
        members = _reproduce_cohort_members(db, cohort)
        if members is None:
            plan.unresolved_cohorts.append(key)
            continue
        plan.cohort_digests[str(cohort["member_digest"])] = cohort_member_digest(
            backtest_map.get(member, member) for member in members
        )


# How a cohort is scoped, tried in order and accepted only where the digest it produces equals
# the one stored beside the row. Nothing here is trusted for being plausible: a scoping that does
# not reproduce the stored digest is not this cohort, and the row goes to `unresolved_cohorts`
# for its stage-20 cell to recompute. `cohort_type` says which axes are held: a `family` cohort
# fixes family and label, `stagelabel` fixes the label at one stage, `label` fixes the label
# alone. `BacktestExplorer.best` then applies coverage and tradeless-backtest rules that the
# registry does not record, which is why the cohorts whose stored size is well below their scope
# cannot be reassembled from it.


def _reproduce_cohort_members(db: sqlite3.Connection, cohort: sqlite3.Row) -> list[str] | None:
    scoped = db.execute(
        "SELECT b.backtest_hash AS backtest_hash, b.stage AS stage, t.family AS family "
        "FROM backtest_runs b "
        "JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash "
        "JOIN training_runs t ON t.training_hash = p.training_hash "
        "WHERE t.label = ?",
        (cohort["label"],),
    ).fetchall()
    if cohort["family"] is not None:
        scoped = [row for row in scoped if row["family"] == cohort["family"]]
    if not scoped:
        return None
    candidates: list[list[str]] = [[row["backtest_hash"] for row in scoped]]
    if cohort["stage"] is not None:
        candidates.append(
            [row["backtest_hash"] for row in scoped if row["stage"] == cohort["stage"]]
        )
    for candidate in candidates:
        if candidate and cohort_member_digest(candidate) == cohort["member_digest"]:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

#: Every column holding an identity that a fold renumber moves, by the map that moves it.
#: The list is what the migration updates; it is not what proves the update complete. That is
#: :func:`surviving_identity_references`, which reads the schema rather than this tuple, so a
#: table added later fails the migration instead of being quietly skipped.
_IDENTITY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("training", "training_runs", "training_hash"),
    ("training", "prediction_sets", "training_hash"),
    ("training", "candidate_fold_completions", "training_hash"),
    ("training", "candidate_fold_completions", "candidate_identity"),
    ("training", "execution_attempts", "scientific_identity"),
    ("prediction", "prediction_sets", "prediction_hash"),
    ("prediction", "prediction_coverage", "prediction_hash"),
    ("prediction", "prediction_metrics", "prediction_hash"),
    ("prediction", "fold_metrics", "prediction_hash"),
    ("prediction", "backtest_runs", "prediction_hash"),
    ("prediction", "official_population_members", "member_hash"),
    ("backtest", "backtest_runs", "backtest_hash"),
    ("backtest", "backtest_metrics", "backtest_hash"),
    ("backtest", "backtest_fold_metrics", "backtest_hash"),
    ("backtest", "backtest_paired_metrics", "challenger_hash"),
    ("backtest", "backtest_paired_metrics", "benchmark_hash"),
    ("backtest", "cohort_metrics", "leader_hash"),
    ("backtest", "official_population_members", "member_hash"),
    ("population", "official_populations", "population_hash"),
    ("population", "official_populations", "supersedes_hash"),
    ("population", "official_population_members", "population_hash"),
)

#: Fold-id columns, and the identity that says which permutation applies to each row.
_FOLD_ID_COLUMNS: tuple[tuple[str, str], ...] = (
    ("candidate_fold_completions", "training_hash"),
    ("fold_metrics", "prediction_hash"),
    ("backtest_fold_metrics", "backtest_hash"),
)

#: Added to a fold id before it is written to its final value. A permutation is a bijection,
#: so writing one row at a time collides with a row that has not moved yet - 0 -> 9 while 9 is
#: still there - and the primary keys on these tables refuse it. The offset is a value no fold
#: id can take, so the intermediate state is unambiguous rather than merely unlikely.
_FOLD_ID_OFFSET = 1_000_000

#: A permutation that moves nothing, for a tree whose owning run is not being renumbered.
_IDENTITY_PERMUTATION: dict[int, int] = {fold: fold for fold in range(1024)}


def _table_columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in db.execute(f"PRAGMA table_info({table})")]


def _tables(db: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    ]


#: Columns that record a retired identity on purpose. `training_identity_migrations` exists to
#: say which identity a migrated result used to have, so its provenance columns naming one is the
#: table working. Its `target_training_hash` is deliberately not exempt: a row whose *target* is a
#: retired identity means an earlier migration's result was migrated again without its record
#: being carried forward, which is a real defect and should fail.
_RETIRED_IDENTITY_IS_EXPECTED: frozenset[tuple[str, str]] = frozenset(
    {
        ("training_identity_migrations", "source_training_hash"),
        ("training_identity_migrations", "prediction_map_json"),
        ("training_identity_migrations", "proof_json"),
    }
)


def surviving_identity_references(
    db: sqlite3.Connection, retired: Iterable[str]
) -> list[tuple[str, str, str]]:
    """Every ``(table, column, value)`` still naming an identity the migration retired.

    Reads the schema rather than a written-down column list, so a table introduced after this
    module was written is covered. This is the check that makes a partial migration fail: a
    row left keyed to a retired identity is a dangling reference, not a stale label, and no
    single query over the tables that *were* migrated would reveal it.
    """
    retired_set = {str(value) for value in retired}
    if not retired_set:
        return []
    found: list[tuple[str, str, str]] = []
    for table in _tables(db):
        for column in _table_columns(db, table):
            if (table, column) in _RETIRED_IDENTITY_IS_EXPECTED:
                continue
            for (value,) in db.execute(
                f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL'
            ):
                if not isinstance(value, str):
                    continue
                if value in retired_set:
                    found.append((table, column, value))
                    continue
                hits = {token for token in re.findall(r"[0-9a-f]{12}", value)} & retired_set
                for hit in sorted(hits):
                    found.append((table, column, hit))
    return found


def _remap_parquet(source: Path, target: Path, permutation: Mapping[int, int]) -> bool:
    """Rewrite *source* into *target* with its fold column relabeled; False if it has none."""
    columns = set(pl.read_parquet_schema(source))
    fold_column = next((name for name in ("fold", "fold_id") if name in columns), None)
    if fold_column is None:
        return False
    frame = pl.read_parquet(source)
    dtype = frame.schema[fold_column]
    stored = frame.get_column(fold_column).drop_nulls().cast(pl.Int64).unique().to_list()
    unknown = sorted(set(stored) - set(permutation) - _NON_FOLD_SENTINELS)
    if unknown:
        raise FoldRenumberRefusal(f"{source}: fold ids {unknown} are outside the permutation")
    mapping = {**{sentinel: sentinel for sentinel in _NON_FOLD_SENTINELS}, **dict(permutation)}
    frame = frame.with_columns(
        # `default=None` carries a null fold through as a null rather than failing on it.
        # A value the permutation does not cover is refused above, so nothing else can reach
        # the default and be silently blanked.
        pl.col(fold_column).cast(pl.Int64).replace_strict(mapping, default=None).cast(dtype)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target, **_PARQUET_WRITE_SETTINGS)
    return True


def _copy_artifact_tree(
    source_dir: Path,
    target_dir: Path,
    permutation: Mapping[int, int],
    *,
    spec_override: Mapping[str, Any] | None = None,
    identity_renames: Mapping[str, str] | None = None,
) -> None:
    """Materialize *source_dir* at *target_dir* under the new fold numbering.

    A file whose content does not change is hard-linked rather than copied, so the fitted
    artifact at the new path is the same file it was before - the same inode, not a duplicate
    that happens to compare equal. Only the documents that name a fold are rewritten.
    """
    renames = identity_renames or {}
    moves_nothing = (
        spec_override is None
        and not renames
        and all(stored == new for stored, new in permutation.items())
    )
    for source in sorted(source_dir.rglob("*")):
        if moves_nothing and source.is_file():
            target = target_dir / source.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
            continue
        if source.is_dir():
            continue
        relative = PurePosixPath(source.relative_to(source_dir).as_posix())
        target = target_dir / remap_relative_path(relative, permutation, renames)
        target.parent.mkdir(parents=True, exist_ok=True)
        if spec_override is not None and relative.as_posix() == "spec.json":
            target.write_text(json.dumps(spec_override, indent=2, sort_keys=True) + "\n")
            continue
        _write_migrated_file(source, target, permutation, renames)


def _under(run_log: Path, stored: str) -> Path:
    """Resolve a registry-recorded path, which is relative to the case study root.

    Recorded as ``run_log/training/...``, and during a migration the tree it has to resolve
    against is the staging directory rather than the live ``run_log`` beside it.
    """
    relative = PurePosixPath(stored)
    if relative.parts and relative.parts[0] == "run_log":
        return run_log.joinpath(*relative.parts[1:])
    return run_log.parent / relative


def _identity_owners(plan: RenumberPlan) -> dict[str, tuple[str, dict[int, int]]]:
    """Every migrating identity, with the new one and the permutation that moved it."""
    owners: dict[str, tuple[str, dict[int, int]]] = {}
    for remap in plan.remaps:
        owners[remap.source_hash] = (remap.target_hash, remap.permutation)
        for stored, new in remap.prediction_map.items():
            owners[stored] = (new, remap.permutation)
        for stored, new in remap.backtest_map.items():
            owners[stored] = (new, remap.permutation)
    return owners


def _permutation_for_path(
    relative: PurePosixPath, owners: Mapping[str, tuple[str, dict[int, int]]]
) -> dict[int, int]:
    """The permutation belonging to whichever identity names this path, else the identity map.

    A file outside the three identity trees says which run it belongs to by carrying that
    run's hash in its path, so the permutation to apply to its contents is read from the path
    rather than passed down.
    """
    for part in relative.parts:
        owner = owners.get(part) or owners.get(part.partition(".")[0])
        if owner is not None:
            return owner[1]
    return _IDENTITY_PERMUTATION


def _write_migrated_file(
    source: Path,
    target: Path,
    permutation: Mapping[int, int],
    renames: Mapping[str, str],
) -> None:
    """Materialize one file at its new path, rewriting it only if it names a fold or a run."""
    if source.suffix == ".parquet" and _remap_parquet(source, target, permutation):
        return
    if source.suffix == ".json":
        try:
            document = json.loads(source.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            document = None
        if document is not None:
            remapped = remap_json_document(document, permutation, identity_renames=renames)
            if remapped != document:
                target.write_text(json.dumps(remapped, indent=1) + "\n")
                return
    os.link(source, target)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rename_identities(db: sqlite3.Connection, kind: str, mapping: Mapping[str, str]) -> None:
    """Rewrite one kind of identity everywhere it is stored, without a transient collision.

    Written in two passes through a reserved prefix. A permutation of identities can map one
    stored value onto another that has not been rewritten yet, and a single-pass update would
    either collide on a primary key or, worse, rewrite the same row twice.
    """
    if not mapping:
        return
    db.execute("DROP TABLE IF EXISTS temp._identity_map")
    db.execute("CREATE TEMP TABLE _identity_map (old TEXT PRIMARY KEY, new TEXT NOT NULL)")
    db.executemany(
        "INSERT INTO temp._identity_map (old, new) VALUES (?, ?)", sorted(mapping.items())
    )
    for column_kind, table, column in _IDENTITY_COLUMNS:
        if column_kind != kind or table not in _tables(db):
            continue
        db.execute(
            f'UPDATE "{table}" SET "{column}" = '
            f"(SELECT 'migrating:' || m.new FROM temp._identity_map m WHERE m.old = \"{column}\") "
            f'WHERE "{column}" IN (SELECT old FROM temp._identity_map)'
        )
        db.execute(
            f'UPDATE "{table}" SET "{column}" = substr("{column}", 11) '
            f"WHERE \"{column}\" LIKE 'migrating:%'"
        )
    db.execute("DROP TABLE temp._identity_map")


def _renumber_fold_ids(db: sqlite3.Connection, plan: RenumberPlan) -> None:
    """Relabel every stored ``fold_id``, in the permutation of the run that produced it."""
    permutation_by_training = {remap.source_hash: remap.permutation for remap in plan.remaps}
    prediction_owner = {
        prediction: remap.source_hash
        for remap in plan.remaps
        for prediction in remap.prediction_map
    }
    backtest_owner = {
        backtest: remap.source_hash for remap in plan.remaps for backtest in remap.backtest_map
    }
    owners = {
        "candidate_fold_completions": permutation_by_training.keys().__contains__,
        "fold_metrics": prediction_owner.__contains__,
        "backtest_fold_metrics": backtest_owner.__contains__,
    }
    resolve = {
        "candidate_fold_completions": lambda key: key,
        "fold_metrics": prediction_owner.get,
        "backtest_fold_metrics": backtest_owner.get,
    }
    for table, key_column in _FOLD_ID_COLUMNS:
        if table not in _tables(db):
            continue
        updates: list[tuple[int, int]] = []
        for rowid, key, fold_id in db.execute(
            f'SELECT rowid, "{key_column}", fold_id FROM "{table}"'
        ):
            if not owners[table](str(key)):
                continue
            permutation = permutation_by_training[resolve[table](str(key))]
            moved = _permute_data(permutation, int(fold_id), where=table)
            # A fold id that maps to itself - a declared sentinel, or a permutation's fixed
            # point - is left alone rather than sent through the offset. The offset is added
            # unconditionally and removed only from values at or above it, so a negative
            # sentinel would come back as its offset image instead of itself.
            if moved != int(fold_id):
                updates.append((moved, int(rowid)))
        if not updates:
            continue
        db.executemany(
            f'UPDATE "{table}" SET fold_id = ? + {_FOLD_ID_OFFSET} WHERE rowid = ?', updates
        )
        db.execute(
            f'UPDATE "{table}" SET fold_id = fold_id - {_FOLD_ID_OFFSET} '
            f"WHERE fold_id >= {_FOLD_ID_OFFSET}"
        )


def _rewrite_documents(db: sqlite3.Connection, plan: RenumberPlan, run_log: Path) -> None:
    """Rewrite the stored JSON and the artifact paths that name a fold or an identity."""
    training_map = plan.training_map
    prediction_map = plan.prediction_map
    permutation_by_target = {remap.target_hash: remap.permutation for remap in plan.remaps}

    db.executemany(
        "UPDATE training_runs SET spec_json = ? WHERE training_hash = ?",
        [(canonical_json(remap.target_spec), remap.target_hash) for remap in plan.remaps],
    )

    backtest_updates = []
    for backtest_hash, spec_json in db.execute(
        "SELECT backtest_hash, spec_json FROM backtest_runs"
    ):
        remapped = remap_backtest_spec(json.loads(spec_json), prediction_map)
        rewritten = json.dumps(remapped, sort_keys=True, separators=(",", ":"))
        if rewritten != spec_json:
            backtest_updates.append((rewritten, str(backtest_hash)))
    db.executemany(
        "UPDATE backtest_runs SET spec_json = ? WHERE backtest_hash = ?", backtest_updates
    )

    population_updates = []
    for population_hash, member_kind, snapshot_json in db.execute(
        "SELECT population_hash, member_kind, snapshot_json FROM official_populations"
    ):
        members = prediction_map if member_kind == "prediction" else plan.backtest_map
        chained = dict(members)
        chained.update(plan.population_map)
        snapshot, _ = remap_population_snapshot(json.loads(snapshot_json), chained)
        population_updates.append((canonical_json(snapshot), str(population_hash)))
    db.executemany(
        "UPDATE official_populations SET snapshot_json = ? WHERE population_hash = ?",
        population_updates,
    )

    completion_updates = []
    source_by_target = {remap.target_hash: remap.source_hash for remap in plan.remaps}
    for rowid, training_hash, fitted, shard, settings_json in db.execute(
        "SELECT rowid, training_hash, fitted_state_path, prediction_shard_path, "
        "resolved_settings_json FROM candidate_fold_completions"
    ):
        target_hash = str(training_hash)
        permutation = permutation_by_target.get(target_hash)
        if permutation is None:
            continue
        source_hash = source_by_target[target_hash]
        renames = {source_hash: target_hash}
        fitted_path = str(remap_relative_path(PurePosixPath(str(fitted)), permutation, renames))
        shard_path = str(remap_relative_path(PurePosixPath(str(shard)), permutation, renames))
        settings = remap_json_document(
            json.loads(settings_json), permutation, identity_renames=renames
        )
        if isinstance(settings, dict) and "training_hash" in settings:
            settings["training_hash"] = training_map.get(
                str(settings["training_hash"]), target_hash
            )
        completion_updates.append(
            (
                fitted_path,
                _sha256_file(_under(run_log, fitted_path)),
                shard_path,
                _sha256_file(_under(run_log, shard_path)),
                canonical_json(settings),
                int(rowid),
            )
        )
    db.executemany(
        "UPDATE candidate_fold_completions SET fitted_state_path = ?, fitted_state_digest = ?, "
        "prediction_shard_path = ?, prediction_shard_digest = ?, resolved_settings_json = ? "
        "WHERE rowid = ?",
        completion_updates,
    )

    db.executemany(
        "UPDATE cohort_metrics SET member_digest = ? WHERE member_digest = ?",
        [(new, old) for old, new in plan.cohort_digests.items()],
    )


def _record_migrations(db: sqlite3.Connection, plan: RenumberPlan) -> None:
    """Leave the audit trail beside the rows, in the table the registry already has for it."""
    if "training_identity_migrations" not in _tables(db):
        return
    from .store import _utc_now

    db.executemany(
        "INSERT OR REPLACE INTO training_identity_migrations (target_training_hash, "
        "source_training_hash, target_spec_json, prediction_map_json, proof_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                remap.target_hash,
                remap.source_hash,
                canonical_json(remap.target_spec),
                canonical_json(remap.prediction_map),
                canonical_json(
                    {
                        "migrated_fields": ["computation.cv.folds"],
                        "migration": "fold-renumbering",
                        "permutation": {
                            str(key): value for key, value in sorted(remap.permutation.items())
                        },
                        "backtest_map": dict(sorted(remap.backtest_map.items())),
                    }
                ),
                _utc_now(),
            )
            for remap in plan.remaps
        ],
    )


def _foreign_key_violations(db: sqlite3.Connection) -> dict[str, int]:
    """Dangling references per ``"child -> parent"``, as ``PRAGMA foreign_key_check`` sees them."""
    counted: dict[str, int] = {}
    for row in db.execute("PRAGMA foreign_key_check"):
        key = f"{row[0]} -> {row[2]}"
        counted[key] = counted.get(key, 0) + 1
    return counted


def _table_counts(db: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in _tables(db)
    }


def _build_staging_tree(run_log: Path, staging: Path, plan: RenumberPlan) -> None:
    staging.mkdir(parents=True)
    moved = {"training", "predictions", "backtest"}
    owner = _identity_owners(plan)
    renames = {stored: new for stored, (new, _) in owner.items()}
    for entry in sorted(run_log.iterdir()):
        if entry.name in moved:
            continue
        # The backups beside the live database are snapshots of the registry as it was, and
        # they stay with the tree this replaces rather than being carried into a migrated one,
        # where they would read as recoverable state under the new numbering.
        if entry.name.startswith("registry.db"):
            continue
        if entry.is_file():
            os.link(entry, staging / entry.name)
            continue
        # Anything else under `run_log` is migrated the same way as the identity trees rather
        # than hard-linked whole. `_snapshots` holds one conformal-width file per prediction,
        # named by the prediction hash and carrying a `fold_id` column, so copying it
        # unchanged would leave a retired identity in a filename and old ids in the data.
        for source in sorted(entry.rglob("*")):
            if not source.is_file():
                continue
            relative = PurePosixPath(source.relative_to(run_log).as_posix())
            permutation = _permutation_for_path(relative, owner)
            target = staging / remap_relative_path(relative, permutation, renames)
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_migrated_file(source, target, permutation, renames)
    # A consistent copy of the registry, taken through SQLite rather than off the filesystem:
    # the live database has a write-ahead log beside it, so copying the file alone can produce
    # a database missing its most recent commits.
    with (
        closing(sqlite3.connect(run_log / "registry.db")) as source_db,
        closing(sqlite3.connect(staging / "registry.db")) as target_db,
    ):
        source_db.backup(target_db)

    training_map = plan.training_map
    prediction_map = plan.prediction_map
    backtest_map = plan.backtest_map
    permutation_by_source = {remap.source_hash: remap.permutation for remap in plan.remaps}
    spec_by_source = {remap.source_hash: remap.target_spec for remap in plan.remaps}

    for kind, mapping in (
        ("training", training_map),
        ("predictions", prediction_map),
        ("backtest", backtest_map),
    ):
        source_root = run_log / kind
        if not source_root.is_dir():
            continue
        (staging / kind).mkdir(parents=True, exist_ok=True)
        for source_dir in sorted(source_root.iterdir()):
            if not source_dir.is_dir():
                continue
            stored = source_dir.name
            target_dir = staging / kind / mapping.get(stored, stored)
            renames: dict[str, str] = {}
            if kind == "training":
                permutation = permutation_by_source.get(stored, _IDENTITY_PERMUTATION)
                override = spec_by_source.get(stored)
                if stored in training_map:
                    renames = {stored: training_map[stored]}
            elif kind == "predictions":
                owner = next(
                    (remap.source_hash for remap in plan.remaps if stored in remap.prediction_map),
                    None,
                )
                permutation = permutation_by_source.get(owner or "", _IDENTITY_PERMUTATION)
                override = None
            else:
                # A backtest's `spec.json` carries `backtest_config.metadata.prediction_hash`,
                # which is both inside the hashed strategy and compared against the row. Copying
                # the tree unchanged leaves the file disagreeing with the registry about which
                # prediction the backtest ran on.
                permutation = _IDENTITY_PERMUTATION
                override = None
                renames = dict(prediction_map)
            _copy_artifact_tree(
                source_dir,
                target_dir,
                permutation,
                spec_override=override,
                identity_renames=renames,
            )


def _rewrite_prediction_coverage(
    db: sqlite3.Connection, plan: RenumberPlan, staging: Path, original: Path
) -> None:
    """Recompute the coverage evidence that describes a rewritten prediction frame.

    `prediction_coverage` holds a digest of the prediction artifact and digests of its key
    columns, and the key columns include the fold id. Relabeling the frame moves all three,
    and `PredictionResult.completeness` compares the artifact against its recorded digest -
    so a migration that moved the identity and left the evidence would leave every migrated
    prediction reporting incomplete, which is how a locked backtest would refuse them.

    Each row is reproduced from the frame it was written over before it is rewritten. A row
    that is not `complete` is refused rather than recomputed: its expected keys are not the
    ones on disk, and nothing here can reconstruct which rows were missing.
    """
    from .completeness import evaluate_prediction_coverage

    db.row_factory = sqlite3.Row
    updates: list[tuple[str, str, str | None, str]] = []
    for stored, migrated in sorted(plan.prediction_map.items()):
        row = db.execute(
            "SELECT * FROM prediction_coverage WHERE prediction_hash = ?", (migrated,)
        ).fetchone()
        if row is None:
            continue
        if row["status"] != "complete":
            raise FoldRenumberRefusal(
                f"prediction {stored} has {row['status']!r} coverage; its expected keys "
                "cannot be reconstructed from the artifact"
            )
        before = pl.read_parquet(original / "predictions" / stored / "predictions.parquet")
        after = pl.read_parquet(staging / "predictions" / migrated / "predictions.parquet")
        recorded = row["artifact_digest"]
        if recorded is not None and value_digest(before) != recorded:
            raise FoldRenumberRefusal(
                f"prediction {stored} does not match its recorded artifact digest"
            )
        was = evaluate_prediction_coverage(before, before)
        if (
            was.expected_key_digest != row["expected_key_digest"]
            or was.actual_key_digest != row["actual_key_digest"]
        ):
            raise FoldRenumberRefusal(f"prediction {stored} coverage digests do not reproduce")
        now = evaluate_prediction_coverage(after, after)
        updates.append(
            (
                now.expected_key_digest,
                now.actual_key_digest,
                value_digest(after) if recorded is not None else None,
                migrated,
            )
        )
    db.executemany(
        "UPDATE prediction_coverage SET expected_key_digest = ?, actual_key_digest = ?, "
        "artifact_digest = ? WHERE prediction_hash = ?",
        updates,
    )


def _rewrite_staged_registry(staging: Path, original: Path, plan: RenumberPlan) -> None:
    with closing(sqlite3.connect(staging / "registry.db")) as db:
        db.execute("PRAGMA foreign_keys = OFF")
        db.execute("BEGIN IMMEDIATE")
        try:
            _renumber_fold_ids(db, plan)
            _rename_identities(db, "training", plan.training_map)
            _rename_identities(db, "prediction", plan.prediction_map)
            _rename_identities(db, "backtest", plan.backtest_map)
            _rename_identities(db, "population", plan.population_map)
            _rewrite_documents(db, plan, staging)
            _rewrite_prediction_coverage(db, plan, staging, original)
            _record_migrations(db, plan)
            db.commit()
        except Exception:
            db.rollback()
            raise


def verify_fold_renumbering(staging: Path, retired: Path, plan: RenumberPlan) -> dict[str, Any]:
    """Check the migrated tree against the one it replaces, and refuse on any disagreement.

    Reproduction rather than self-agreement: the new identities are recomputed from the new
    specifications, the retired ones are required to be gone, and the fitted artifacts are
    checked to be the same files - the same inode - rather than merely present. That last
    check exists because a training run fits on whatever artifact is on disk without
    verifying its identity (issue #987), so nothing downstream would notice a fitted state
    that had been silently replaced by a copy of the wrong fold's.
    """
    report: dict[str, Any] = {}
    # The registry this runs against already carries orphans - `us_firm_characteristics` has
    # six, left by a stale-holdout delete - so requiring zero would refuse a correct migration
    # for a defect it did not cause. What the migration must not do is add one, which is what
    # comparing the counts per (table, parent) says.
    with closing(sqlite3.connect(retired / "registry.db")) as before_db:
        baseline = _foreign_key_violations(before_db)
    with closing(sqlite3.connect(staging / "registry.db")) as db:
        db.row_factory = sqlite3.Row
        after = _foreign_key_violations(db)
        introduced = {
            key: (baseline.get(key, 0), count)
            for key, count in after.items()
            if count > baseline.get(key, 0)
        }
        if introduced:
            raise FoldRenumberRefusal(f"migration introduced dangling references: {introduced}")
        report["foreign_key_violations"] = dict(sorted(after.items()))
        report["foreign_key_violations_before"] = dict(sorted(baseline.items()))
        moved = (plan.training_map, plan.prediction_map, plan.backtest_map, plan.population_map)
        retired_identities = {stored for mapping in moved for stored in mapping} - {
            new for mapping in moved for new in mapping.values()
        }
        surviving = surviving_identity_references(db, retired_identities)
        if surviving:
            raise FoldRenumberRefusal(f"retired identities still referenced: {surviving[:5]}")
        for remap in plan.remaps:
            row = db.execute(
                "SELECT spec_json FROM training_runs WHERE training_hash = ?",
                (remap.target_hash,),
            ).fetchone()
            if row is None:
                raise FoldRenumberRefusal(f"migrated training {remap.target_hash} is absent")
            if training_hash_from_spec(json.loads(row["spec_json"])) != remap.target_hash:
                raise FoldRenumberRefusal(
                    f"migrated training {remap.target_hash} does not reproduce its own hash"
                )
        for unchanged in plan.unchanged:
            if (
                db.execute(
                    "SELECT 1 FROM training_runs WHERE training_hash = ?", (unchanged,)
                ).fetchone()
                is None
            ):
                raise FoldRenumberRefusal(f"unmigrated training {unchanged} disappeared")
        report["table_counts"] = _table_counts(db)

    with closing(sqlite3.connect(retired / "registry.db")) as before_db:
        report["table_counts_before"] = _table_counts(before_db)

    # Every table holds what it held: a relabeling moves identities, it does not add or drop
    # results. The one table that grows is the audit trail, which gains a row per migrated run.
    differing = {
        table: (report["table_counts_before"].get(table), count)
        for table, count in report["table_counts"].items()
        if table != "training_identity_migrations"
        and report["table_counts_before"].get(table) != count
    }
    if differing:
        raise FoldRenumberRefusal(f"row counts moved during migration: {differing}")

    same_file = 0
    for remap in plan.remaps:
        source_dir = retired / "training" / remap.source_hash
        target_dir = staging / "training" / remap.target_hash
        if not target_dir.is_dir():
            raise FoldRenumberRefusal(f"migrated training tree {remap.target_hash} is absent")
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file() or source.suffix in {".json", ".parquet"}:
                continue
            relative = PurePosixPath(source.relative_to(source_dir).as_posix())
            target = target_dir / remap_relative_path(
                relative, remap.permutation, {remap.source_hash: remap.target_hash}
            )
            if not target.is_file():
                raise FoldRenumberRefusal(f"fitted artifact {relative} is absent after migration")
            if target.stat().st_ino != source.stat().st_ino:
                raise FoldRenumberRefusal(
                    f"fitted artifact {relative} is a copy, not the file it was"
                )
            same_file += 1
    report["fitted_artifacts_unmoved"] = same_file

    # The database scan cannot see a filename. `_snapshots` names its files by the prediction
    # hash they belong to, so a tree copied whole would keep a retired identity in a path while
    # every table read clean - which is what this catches.
    stale_paths = [
        str(path.relative_to(staging))
        for path in staging.rglob("*")
        if _path_names_a_retired_identity(path.relative_to(staging), retired_identities)
    ]
    if stale_paths:
        raise FoldRenumberRefusal(f"retired identities still in artifact paths: {stale_paths[:5]}")
    report["artifact_paths_checked"] = True
    return report


def _path_names_a_retired_identity(relative: Path, retired: Iterable[str]) -> bool:
    retired_set = set(retired)
    for part in relative.parts:
        if part in retired_set or part.partition(".")[0] in retired_set:
            return True
    return False


def apply_fold_renumbering(case_dir: Path, plan: RenumberPlan) -> dict[str, Any]:
    """Relabel a registry in place, or leave it exactly as it was.

    The new tree is built beside the old one and verified before either is touched, so a
    failure at any point up to the swap leaves the registry unchanged. The swap itself is two
    renames within one directory, and the tree it replaces is kept rather than deleted:
    everything needed to put the registry back is still on disk when this returns.
    """
    if plan.refusals:
        raise FoldRenumberRefusal(f"plan holds {len(plan.refusals)} refusals; nothing applied")
    if not plan.remaps:
        raise FoldRenumberRefusal("plan renumbers nothing")
    # A worktree's case-study directory reaches the registry through a `run_log` symlink into
    # the canonical artifact store. Swapping the symlink would leave a migrated tree inside the
    # worktree and the canonical store untouched, so the swap happens where the data is and the
    # symlink is left pointing at it.
    run_log = (case_dir / "run_log").resolve()
    store_dir = run_log.parent
    retired = store_dir / "run_log.pre-fold-renumber"
    if retired.exists():
        raise FoldRenumberRefusal(f"{retired} exists; an earlier migration was not cleaned up")
    staging = store_dir / f".run_log.migrating.{uuid.uuid4().hex}"
    try:
        _build_staging_tree(run_log, staging, plan)
        _rewrite_staged_registry(staging, run_log, plan)
        report = verify_fold_renumbering(staging, run_log, plan)
        os.replace(run_log, retired)
        try:
            os.replace(staging, run_log)
        except Exception:
            os.replace(retired, run_log)
            raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report["retired_tree"] = str(retired)
    return report
