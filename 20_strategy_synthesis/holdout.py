"""Read the holdout a case study registered. This module no longer generates one.

Generating holdout predictions belongs to the case studies, which each own a
``NN_holdout_predictions`` / ``NN_holdout_backtest`` pair. Chapter 20 reads results from
the registry and computes further analysis over them; it does not produce case-study
results. The generation half that used to live here - ``generate_holdout``, the
``_train_*`` family behind ``TRAIN_DISPATCH``, ``build_holdout_split`` and
``delete_holdout_predictions`` - was removed with ``00_holdout_predictions``.

Removing it also removes a writer of rows nothing could read as a holdout.
``generate_holdout`` refit on the pre-holdout window and then called
``register_prediction_set`` with the *validation* candidate's ``training_hash``, so what
it published over the holdout window was indistinguishable from a validation-fitted model
scored there. ``_holdout_lineage_for`` in ``01_aggregate_synthesis`` already declined
those rows - it requires a training run actually refitted for the holdout - so nothing
Chapter 20 reports changes. ``tests/test_holdout_generation_retirement.py`` records the
shape and why two of its three cases are refusals rather than deletions.

What is left is the selection and read side:

* ``LABEL_RESTRICTIONS`` and ``UNIVERSE_RESTRICTIONS``, re-exported from
  ``case_studies.utils.strategy_analysis`` so there is one declaration.
  ``01_aggregate_synthesis`` imports the first from here.
* ``HOLDOUT_SELECTION_STAGES``, the stages a rank-1 may be drawn from.
* ``select_best_models`` / ``select_best_model``, the validation ranking.
* ``load_existing_holdout`` and ``has_holdout_predictions``, which read what is there.
* ``_is_degenerate_predictions``, the rule ``01_aggregate_synthesis`` cites.

The ``_train_*`` family took lightgbm out of this module's import closure, measured by
executing it and reading ``sys.modules``. torch is still in it, reached transitively
through ``case_studies.utils.strategy_analysis``, so the tests that load this file stay
behind the same boundary in ``.github/ci/unit-test-quarantine.txt``.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3

import numpy as np
import polars as pl

from case_studies.utils.strategy_analysis import (
    LABEL_RESTRICTIONS,
    SELECTION_STAGES,
    UNIVERSE_RESTRICTIONS,
    NoSelectableCandidates,
    selectable_validation_candidates,
)
from utils.paths import get_case_study_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_from_row(cs_id: str, row: dict) -> dict:
    """Hydrate a :func:`selectable_validation_candidates` row into a retrain candidate.

    The row already carries the identities and the strategy spec the ranking read. What
    the retrain additionally needs is the checkpoint the prediction set sits on - the
    checkpoint is part of the configuration, so a replay that ignores it replays a
    different model - and the training specification the refit is built from.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        pred_row = db.execute(
            "SELECT checkpoint_value, checkpoint_kind FROM prediction_sets "
            "WHERE prediction_hash = ?",
            (row["prediction_hash"],),
        ).fetchone()
        train_row = db.execute(
            "SELECT spec_json FROM training_runs WHERE training_hash = ?",
            (row["training_hash"],),
        ).fetchone()
    finally:
        db.close()

    return {
        "backtest_hash": row["backtest_hash"],
        "prediction_hash": row["prediction_hash"],
        "training_hash": row["training_hash"],
        "checkpoint_value": pred_row["checkpoint_value"],
        "checkpoint_kind": pred_row["checkpoint_kind"],
        "family": row["family"],
        "config_name": row["config_name"],
        "training_spec": json.loads(train_row["spec_json"]),
        "strategy_spec": json.loads(row["spec_json"]),
        # The Sharpe the selection was made on, which is the common-support one wherever the
        # field was re-ranked. Reporting the stored value instead would have the two entry
        # points agree on the configuration and print different numbers for it, and chapter
        # 20 measures holdout decay against this.
        #
        # The fallback to the stored value is for the field that was never re-ranked, where
        # it is the only Sharpe there is. It must not be reached for a ruined candidate,
        # whose `comparison_sharpe` is also None and whose stored Sharpe describes an
        # account that went to zero; `select_best_models` keeps those out of this pool.
        "val_sharpe": (
            row["comparison_sharpe"] if row["comparison_sharpe"] is not None else row["sharpe"]
        ),
        "label": row["label"] or "",
    }


HOLDOUT_SELECTION_STAGES: tuple[str, ...] = SELECTION_STAGES
"""Stages eligible for holdout rank-1 selection.

An alias for :data:`case_studies.utils.strategy_analysis.SELECTION_STAGES`, kept for
callers that import this name. The stage pool is a property of the selection rule
(``reference/CASE_STUDY_PIPELINE.md`` section 5), not of this module, and declaring it
twice is how the two holdout selectors came to disagree in the first place.
"""


def select_best_models(
    cs_id: str,
    *,
    top_n: int = 5,
    min_ic: float | None = None,
    families: list[str] | None = None,
    labels: list[str] | None = None,
) -> list[dict]:
    """The top-N distinct trained models by validation Sharpe, best first.

    The ranking is :func:`selectable_validation_candidates`, which is also what
    ``resolve_canonical_rank1_lineage`` and therefore every case study's own holdout
    notebook ranks. This function used to build its own pool - ``BacktestExplorer.best``
    per stage, concatenated and re-sorted - with its own eligibility filter and its own
    ordering, and the two were kept together by a comment. They disagreed. Measured on
    ``fx_pairs`` 2026-09-07: ``deep_learning/tcn`` on ``fwd_ret_21d`` carries two
    backtests tied at Sharpe 0.2639142245820113, this path answered ``9402978117e9`` and
    the canonical resolver answered ``56070f34dff1``. Two strategy specifications for one
    model, and the holdout replays the specification exactly, so the choice decided which
    strategy spent the case study's single holdout use.

    What is added here and belongs here, because it is about *retraining* rather than
    about which configuration the case study reports:

    * the dedupe by ``prediction_hash``, keeping each model's best-Sharpe backtest. The
      holdout retrain falls back to rank-2 when rank-1's refit produces degenerate
      predictions, and the fallback has to reach a different *trained model* rather than
      a different strategy spec on the same one;
    * ``min_ic``, a caller-supplied floor that no production path passes. IC selects
      nothing (``reference/CASE_STUDY_PIPELINE.md`` section 5); this only ever narrows a
      pool already ordered by Sharpe.

    ``families`` and ``labels`` narrow the pool and are passed straight through;
    ``labels`` replaces ``LABEL_RESTRICTIONS`` rather than adding to it.
    """
    candidates_df = selectable_validation_candidates(cs_id, labels=labels, families=families)

    if min_ic is not None:
        case_dir = get_case_study_dir(cs_id)
        db_path = case_dir / "run_log" / "registry.db"
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            ic_map = {
                prediction_hash: ic_mean
                for prediction_hash, ic_mean in db.execute(
                    "SELECT prediction_hash, ic_mean FROM prediction_metrics "
                    "WHERE ic_mean IS NOT NULL"
                ).fetchall()
            }
        finally:
            db.close()
        viable = [
            row
            for row in candidates_df
            if (ic := ic_map.get(row["prediction_hash"])) is not None and ic > min_ic
        ]
        if viable:
            candidates_df = viable

    seen_phashes: set[str] = set()
    candidates: list[dict] = []
    for row in candidates_df:
        # A candidate the engine stopped at ruin carries no Sharpe over the common support
        # and is not a strategy anything can be retrained into. It stays on the ranked frame
        # so it is visibly compared, and the ranking already orders it below every solvent
        # one - but this pool is the rank-2 and rank-3 the holdout retrain falls back to,
        # and a field with fewer than `top_n` solvent members would otherwise offer one.
        # `_candidate_from_row` would then report its registered Sharpe as `val_sharpe`,
        # which is the bankrupt path comparing as a solvent one.
        if row.get("comparison_ruined"):
            continue
        ph = row["prediction_hash"]
        if ph in seen_phashes:
            continue
        seen_phashes.add(ph)
        candidates.append(_candidate_from_row(cs_id, row))
        if len(candidates) >= top_n:
            break

    if not candidates:
        raise ValueError(
            f"No viable candidates for {cs_id} after dedupe across stages "
            f"{HOLDOUT_SELECTION_STAGES}"
        )
    return candidates


def select_best_model(
    cs_id: str,
    *,
    min_ic: float | None = None,
    families: list[str] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Return the rank-1 equal-weight baseline model. Thin wrapper around select_best_models.

    Kept for backward compatibility with the Ch20 nb00 preview cell and any
    external callers. The holdout pipeline uses ``select_best_models`` to
    enable degeneracy-driven fallback to rank-2 / rank-3.
    """
    return select_best_models(cs_id, top_n=1, min_ic=min_ic, families=families, labels=labels)[0]


def _is_degenerate_predictions(
    predictions: pl.DataFrame, *, std_threshold: float = 1e-6
) -> tuple[bool, str]:
    """Return (is_degenerate, reason). Used by the holdout fallback loop.

    Degenerate cases:
    - ``y_score`` column has near-zero std (regression collapsed to constant)
    - ``y_score`` is all-NaN (training failed silently)
    - fewer than 2 distinct values across the full holdout window

    The threshold defaults to 1e-6 (one part per million of std). Below that,
    cross-sectional ranking ties everything; the resulting backtest is a
    machine-precision artifact rather than a real out-of-sample test.
    """
    s = predictions["y_score"]
    if s.null_count() == s.len():
        return True, "all_null"
    finite = s.drop_nulls()
    if finite.len() < 2:
        return True, f"insufficient_finite_values (n={finite.len()})"
    arr = finite.to_numpy()
    if not np.isfinite(arr).any():
        return True, "all_non_finite"
    pred_std = float(np.nanstd(arr))
    if pred_std < std_threshold:
        return True, f"constant_predictions (std={pred_std:.2e} < {std_threshold:.0e})"
    n_unique = int(pl.Series(arr).n_unique())
    if n_unique < 2:
        return True, f"single_unique_value (n_unique={n_unique})"
    return False, ""


def load_existing_holdout(cs_id: str) -> dict:
    """Load existing holdout results from registry without retraining.

    Returns the result dict shape the retired ``generate_holdout`` returned, which is
    what the chapter's summary tables still read.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return {"cs_id": cs_id, "skipped": True}

    with contextlib.closing(sqlite3.connect(str(db_path))) as db:
        db.row_factory = sqlite3.Row

        pred = db.execute(
            "SELECT ps.prediction_hash, ps.training_hash, ps.checkpoint_value, "
            "ps.checkpoint_kind, pm.ic_mean "
            "FROM prediction_sets ps "
            "LEFT JOIN prediction_metrics pm ON ps.prediction_hash = pm.prediction_hash "
            "WHERE ps.split = 'holdout' "
            "ORDER BY ps.created_at DESC LIMIT 1"
        ).fetchone()

        if not pred:
            return {"cs_id": cs_id, "skipped": True}

        train = db.execute(
            "SELECT family, config_name, spec_json FROM training_runs WHERE training_hash=?",
            (pred["training_hash"],),
        ).fetchone()

        bt = db.execute(
            "SELECT bm.sharpe, bm.cagr, bm.max_drawdown, br.backtest_hash "
            "FROM backtest_runs br "
            "JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash "
            "WHERE br.prediction_hash = ?",
            (pred["prediction_hash"],),
        ).fetchone()

        # Validation Sharpe for the SAME checkpoint that produced this
        # holdout (matched on training_hash + checkpoint_kind + checkpoint_value).
        # Training-only matching can drift to a sibling checkpoint of the
        # holdout's lineage; the strict scope keeps val_sharpe on the same
        # rung as the holdout. Linear/GBM rows have NULL checkpoint dims -
        # build the WHERE clause to honor that.
        ck = pred["checkpoint_kind"]
        cv = pred["checkpoint_value"]
        ckind_clause = "ps.checkpoint_kind IS NULL" if ck is None else "ps.checkpoint_kind = ?"
        cval_clause = "ps.checkpoint_value IS NULL" if cv is None else "ps.checkpoint_value = ?"
        params: list = [pred["training_hash"]]
        if ck is not None:
            params.append(ck)
        if cv is not None:
            params.append(cv)
        val_bt = db.execute(
            f"SELECT bm.sharpe FROM prediction_sets ps "
            f"JOIN backtest_runs br ON ps.prediction_hash = br.prediction_hash "
            f"JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash "
            f"WHERE ps.training_hash = ? AND ps.split = 'validation' "
            f"AND {ckind_clause} AND {cval_clause} "
            f"ORDER BY bm.sharpe DESC LIMIT 1",
            params,
        ).fetchone()
        val_sharpe = val_bt["sharpe"] if val_bt else float("nan")

        spec = json.loads(train["spec_json"]) if train else {}
        label = spec.get("label", "")

        return {
            "cs_id": cs_id,
            "family": train["family"] if train else "",
            "config_name": train["config_name"] if train else "",
            "label": label,
            "val_sharpe": val_sharpe,
            "holdout_ic": pred["ic_mean"] if pred["ic_mean"] is not None else float("nan"),
            "holdout_sharpe": bt["sharpe"] if bt else float("nan"),
            "holdout_cagr": bt["cagr"] if bt else float("nan"),
            "holdout_maxdd": bt["max_drawdown"] if bt else float("nan"),
            "prediction_hash": pred["prediction_hash"],
            "backtest_hash": bt["backtest_hash"] if bt else "",
            "elapsed_s": 0,
        }


def has_holdout_predictions(cs_id: str, *, top_n: int = 5) -> bool:
    """Check if any of the top-N validation candidates has holdout predictions.

    Returns True iff at least one of the top-N validation candidates'
    training_hash values appears in `prediction_sets` with split='holdout'.
    Returns False if a holdout exists but none of the top-N candidates'
    training_hashes match it - that signals the holdout is stale relative
    to the current validation sweep (e.g., the rank-1 reshuffled out of
    the top-N) and should be regenerated.

    The top-N envelope (rather than rank-1 only) accommodates a degeneracy fallback:
    when rank-1's holdout retrain collapses, the accepted holdout's training_hash
    matches a rank-2/3/... candidate. Without the envelope, every subsequent run would
    see "no rank-1 holdout" and stack another holdout backtest in the registry. The
    retired `generate_holdout` was the fallback this was written for; the case studies'
    own holdout notebooks fall back the same way, so the envelope is still required.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return False

    # Both refusals mean the same thing here: nothing is currently selectable, so no
    # holdout can cover the current top-N. `ValueError` is this module's own refusal after
    # the dedupe; `NoSelectableCandidates` is the canonical selector's for an empty pool -
    # an initialised registry with no eligible validation backtest, a population that
    # publishes nothing, a selected configuration pin left over from an earlier sweep. Answering False
    # sends the caller to `generate_holdout`, which asks the same selector again without a
    # guard and reports whichever refusal applies from inside the driver's own handler.
    try:
        candidates = select_best_models(cs_id, top_n=top_n)
    except (ValueError, NoSelectableCandidates):
        return False
    candidate_hashes = [c["training_hash"] for c in candidates]
    if not candidate_hashes:
        return False

    db = sqlite3.connect(str(db_path))
    placeholders = ",".join("?" for _ in candidate_hashes)
    count = db.execute(
        f"SELECT COUNT(*) FROM prediction_sets "
        f"WHERE split = 'holdout' AND training_hash IN ({placeholders})",
        candidate_hashes,
    ).fetchone()[0]
    db.close()
    return count > 0
