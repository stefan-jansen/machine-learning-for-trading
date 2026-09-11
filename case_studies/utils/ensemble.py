"""Mean-forecast ensembles over a family's registered prediction sets.

A case study whose lesson is that selection within a family is noise needs an
object that does not select: one ordering built from every member of the family,
registered as a prediction set of its own so that the backtest stage, the
cohort work and the holdout all read it the same way they read a model.

``nasdaq100_microstructure`` is the case study with that lesson and the one this
module exists for. Its featured carrier is the mean forecast of the regularized
LightGBM configurations, and nothing in the repository produced it: three
notebooks read ``family == 'ensemble'``, no notebook wrote it, and every
registry in the fleet held zero rows of it
(ml4t/agent-workspace#1157).

The member set is resolved from what is registered, never listed by hand, so it
cannot go stale against the configurations a case study actually fitted.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import polars as pl

# LightGBM's own default when a preset declares no `num_leaves`. Written down rather
# than read off the library at run time: the member set is part of what the ensemble
# is, so a library default change must arrive as a disagreement here and not as a
# silently different ensemble carrying the same name.
LIGHTGBM_DEFAULT_NUM_LEAVES = 31

# The canonical prediction schema, which is what `normalize_prediction_columns`
# renames every family's frame into: gbm writes these names already, the sequence
# families write `fold`/`prediction`/`actual`, and an ensemble built from one
# family's names would be unreadable beside the other's.
KEY_COLUMNS = ("fold_id", "symbol", "timestamp")
SCORE_COLUMN = "y_score"
TRUTH_COLUMN = "y_true"


def member_num_leaves(config_name: str) -> int:
    """The `num_leaves` a LightGBM preset trains at, explicit or defaulted."""
    from case_studies.utils.registry.specs import load_preset

    preset = load_preset("gbm", config_name)
    params = preset.get("params") or {}
    declared = params.get("num_leaves")
    return int(declared) if declared is not None else LIGHTGBM_DEFAULT_NUM_LEAVES


def resolve_members(
    case_dir: Path,
    *,
    label: str,
    family: str = "gbm",
    split: str = "validation",
    max_num_leaves: int | None = None,
) -> pl.DataFrame:
    """The prediction set of each qualifying configuration, one row per configuration.

    Takes each configuration's **last** checkpoint. Any other choice - the
    checkpoint with the best validation metric, say - would select on the window
    the ensemble is then measured on, which is the thing the ensemble exists to
    avoid doing.

    Returns columns ``config_name``, ``training_hash``, ``prediction_hash``,
    ``checkpoint_value``, ``num_leaves``, ordered by ``config_name`` so the member
    list is stable across calls and so the registered spec is too.
    """
    db = sqlite3.connect(f"file:{case_dir / 'run_log' / 'registry.db'}?mode=ro", uri=True)
    try:
        rows = db.execute(
            """
            SELECT t.config_name, t.training_hash, p.prediction_hash, p.checkpoint_value
            FROM prediction_sets p
            JOIN training_runs t ON t.training_hash = p.training_hash
            WHERE t.family = ? AND t.label = ? AND p.split = ?
            """,
            (family, label, split),
        ).fetchall()
    finally:
        db.close()
    if not rows:
        msg = (
            f"no {family} prediction sets for {label!r} at split {split!r} in "
            f"{case_dir}; the ensemble has nothing to average"
        )
        raise ValueError(msg)

    frame = pl.DataFrame(
        rows,
        schema={
            "config_name": pl.String,
            "training_hash": pl.String,
            "prediction_hash": pl.String,
            "checkpoint_value": pl.Int64,
        },
        orient="row",
    )
    frame = frame.with_columns(
        pl.col("config_name")
        .map_elements(member_num_leaves, return_dtype=pl.Int64)
        .alias("num_leaves")
    )
    if max_num_leaves is not None:
        frame = frame.filter(pl.col("num_leaves") <= int(max_num_leaves))
        if frame.is_empty():
            msg = (
                f"every {family} configuration for {label!r} trains at more than "
                f"{max_num_leaves} leaves, so the ensemble has no members"
            )
            raise ValueError(msg)
    return (
        frame.sort("config_name", "checkpoint_value")
        .group_by("config_name", maintain_order=True)
        .last()
        .sort("config_name")
    )


def mean_forecast(case_study: str, prediction_hashes: list[str]) -> pl.DataFrame:
    """Average the members' forecasts per key into one prediction frame.

    Every member must cover exactly the same keys. A member covering a subset
    would average over fewer forecasts on the rows it misses, so the result would
    be an ensemble of one width on some rows and another width on others while
    reporting a single member count. A disagreement raises here rather than being
    filled, dropped, or left to a join to decide.

    The members are summed one at a time into a running total rather than stacked
    and grouped. On this case study a member carries five to seven million rows
    and there are twelve of them, so the stacked frame would be the largest
    object in the notebook by an order of magnitude for the sake of an addition.
    Summing positionally is what the key check above licenses: two frames sorted
    by the same keys and carrying the same keys are row-aligned.
    """
    from case_studies.utils.backtest_runner import normalize_prediction_columns
    from case_studies.utils.registry import read_predictions

    if len(prediction_hashes) < 2:
        msg = f"a mean forecast needs at least two members, got {len(prediction_hashes)}"
        raise ValueError(msg)

    total: pl.DataFrame | None = None
    for p_hash in prediction_hashes:
        frame = normalize_prediction_columns(read_predictions(case_study, p_hash))
        missing = [c for c in (*KEY_COLUMNS, SCORE_COLUMN, TRUTH_COLUMN) if c not in frame.columns]
        if missing:
            msg = f"prediction set {p_hash} is missing {missing}; columns: {frame.columns}"
            raise ValueError(msg)
        frame = frame.select(*KEY_COLUMNS, TRUTH_COLUMN, SCORE_COLUMN).sort(KEY_COLUMNS)
        if frame.select(KEY_COLUMNS).is_duplicated().any():
            msg = f"prediction set {p_hash} carries duplicate {list(KEY_COLUMNS)} keys"
            raise ValueError(msg)
        if total is None:
            total = frame
            continue
        if frame.height != total.height or not frame.select(KEY_COLUMNS).equals(
            total.select(KEY_COLUMNS)
        ):
            msg = (
                f"member {p_hash} covers {frame.height} keys against {total.height} for "
                f"{prediction_hashes[0]}; the members of a mean forecast must cover the "
                "same rows"
            )
            raise ValueError(msg)
        total = total.with_columns(
            (pl.col(SCORE_COLUMN) + frame.get_column(SCORE_COLUMN)).alias(SCORE_COLUMN)
        )

    assert total is not None
    return total.with_columns(pl.col(SCORE_COLUMN) / len(prediction_hashes))


def ensemble_training_spec(
    member_spec_json: str,
    *,
    members: pl.DataFrame,
    method: str,
    config_name: str,
    provenance: dict[str, Any],
) -> dict:
    """Build the ensemble's training spec from one member's resolved spec.

    The ensemble is fitted on nothing: it has no model, no epochs and no folds of
    its own. What it does have is the members' inputs, and those are what its
    identity has to be keyed on - the same features, the same fold plan, the same
    label artifact, the same expected keys. Deriving them from a member rather
    than rebuilding them is what guarantees the two agree, and the member key
    check in :func:`mean_forecast` is what guarantees every member agrees with
    that one.

    What replaces the model block is the member list itself, by prediction hash.
    Two ensembles over different members are then different identities, and
    re-running a member's fit moves its hash and so moves the ensemble's, which
    is the behaviour every other family already has.
    """
    member = json.loads(member_spec_json)
    computation = dict(member["computation"])
    for key in ("model", "checkpoint_schedule", "runtime_identity", "source_identity"):
        computation.pop(key, None)
    computation["ensemble"] = {
        "method": method,
        "member_family": "gbm",
        "n_members": int(members.height),
        "members": [
            {
                "config_name": row["config_name"],
                "prediction_hash": row["prediction_hash"],
                "checkpoint_value": row["checkpoint_value"],
                "num_leaves": row["num_leaves"],
            }
            for row in members.sort("config_name").iter_rows(named=True)
        ],
    }
    computation["source_identity"] = {"ensemble_builder": 1}
    return {
        "identity_version": member["identity_version"],
        "resolved_spec_schema": member["resolved_spec_schema"],
        "family": "ensemble",
        "label": member["label"],
        "seed": member["seed"],
        "config_name": config_name,
        "execution_tier": member["execution_tier"],
        "computation": computation,
        "provenance": provenance,
    }


def load_ensemble_declaration(case_study: str) -> dict | None:
    """Return the top-level ``ensemble`` block from a case study's setup.yaml.

    ``None`` when the case study declares none, which is every case study but
    ``nasdaq100_microstructure``. Required keys are checked here rather than at
    each use: a typo in one of them would otherwise resolve to a default and
    build a different ensemble under the declared name.
    """
    from case_studies.utils.sweep_config import _load_setup

    block = (_load_setup(case_study) or {}).get("ensemble")
    if block is None:
        return None
    required = (
        "member_family",
        "max_num_leaves",
        "method",
        "config_name",
        "checkpoint",
        "featured_scheme",
        "universe_filter",
    )
    missing = [k for k in required if block.get(k) is None]
    if missing:
        msg = (
            f"the `ensemble` block in case_studies/{case_study}/config/setup.yaml is "
            f"missing {missing}; every key is part of what the ensemble is, so none of "
            "them has a default"
        )
        raise KeyError(msg)
    if str(block["method"]) != "mean_forecast":
        msg = f"unsupported ensemble method {block['method']!r}; only mean_forecast exists"
        raise ValueError(msg)
    if str(block["checkpoint"]) != "last":
        msg = (
            f"unsupported ensemble checkpoint rule {block['checkpoint']!r}; only `last` "
            "exists, because any other rule selects on the validation window"
        )
        raise ValueError(msg)
    return dict(block)
