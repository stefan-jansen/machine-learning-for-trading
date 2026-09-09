"""The shipped fixture registry describes the artifacts shipped beside it.

The nine registries under ``intermediates/*/run_log/`` were copied whole from production
while the artifacts beside them were subsampled, which produced two states a reader
cannot tell apart from a pipeline failure:

* a ``prediction_metrics`` row measuring a file 32x larger than the ``predictions.parquet``
  it sits beside, so ``16_strategy_simulation/08_signal_method_comparison`` rejects the
  pair it is given (ml4t/agent-workspace#286), and
* a ``run_log/predictions/<hash>/`` no registry row names, left behind when a re-sample
  rewrote the registry after a generation had written the artifact
  (ml4t/agent-workspace#1081).

Both are properties of the committed fixture rather than of any code path, so this module
reads the fixture itself. ``tests/reconcile_fixture_registry.py`` is what repairs a
fixture these fail against.

What is deliberately not asserted: that every registered prediction set ships an
artifact. 97.7% of them do not, and seeding them is 88 MB per panel. Those rows are the
replay surface the chapter notebooks rank over, and a row with no artifact is only a
defect once a reader reaches it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.notebook_contracts import defined_ic
from tests.reconcile_fixture_registry import (
    _resolve,
    _spec_context,
    prediction_dirs,
    shipped_predictions,
)

#: The tolerance `08_signal_method_comparison::validate_prediction_set` uses when it
#: checks a loaded parquet against its registered coverage and mean rank IC. Matched
#: here so a fixture that passes this file is one that notebook accepts.
REL_TOL = 1e-12


def _case_studies(intermediates_dir: Path | None) -> list[Path]:
    if intermediates_dir is None:
        pytest.skip("no test-data intermediates on this checkout")
    return sorted(
        entry
        for entry in intermediates_dir.iterdir()
        if (entry / "run_log" / "registry.db").is_file()
    )


def _registered(case_dir: Path) -> set[str]:
    db = sqlite3.connect(f"file:{case_dir / 'run_log' / 'registry.db'}?mode=ro", uri=True)
    try:
        return {str(row[0]) for row in db.execute("SELECT prediction_hash FROM prediction_sets")}
    finally:
        db.close()


def test_no_prediction_directory_is_unreachable(intermediates_dir):
    """A directory no ``prediction_sets`` row names is addressable by nothing.

    Every reader resolves a prediction by hash out of the registry, so an unnamed
    directory is not a spare artifact - it is bytes in the fixture that no code path can
    ever open, and its presence makes the tree look richer than it is.
    """
    unreachable = {}
    for case_dir in _case_studies(intermediates_dir):
        registered = _registered(case_dir)
        orphans = sorted(set(prediction_dirs(case_dir)) - registered)
        if orphans:
            unreachable[case_dir.name] = orphans
    assert not unreachable, (
        "prediction directories with no registry row: "
        + "; ".join(
            f"{name} {len(hashes)} ({hashes[0]}...)" for name, hashes in unreachable.items()
        )
        + ". Run tests/reconcile_fixture_registry.py --apply."
    )


def _daily_ic(parquet: Path, context: dict) -> tuple[int, float] | None:
    """The IC coverage and mean the shipped artifact itself produces.

    Only the daily block, not the whole metrics row: this is the pair the consuming
    notebook validates, and computing it needs no bootstrap, which keeps the check
    inside a unit job's budget.
    """
    from ml4t.diagnostic.metrics import cross_sectional_ic_series

    frame = pl.read_parquet(parquet)
    columns = frame.columns
    score = _resolve(columns, "score")
    target = _resolve(columns, "target")
    date = _resolve(columns, "date")
    entity = _resolve(columns, "entity")
    if not (score and target and date):
        return None
    # Classification sets score IC against the continuous return the binary label was
    # derived from, exactly as `compute_prediction_fold_metrics` does. Against the binary
    # column the statistic is 2*(AUC-0.5), not a rank correlation, so comparing one to a
    # registered value computed the other way would report a disagreement that is not one.
    if str(context.get("task_type") or "regression") == "classification":
        if "eval_actual" not in columns:
            return None
        target = "eval_actual"
    series = cross_sectional_ic_series(
        frame,
        frame,
        pred_col=score,
        ret_col=target,
        date_col=date,
        entity_col=entity,
        method="spearman",
        min_obs=5,
    )
    defined = defined_ic(series) if isinstance(series, pl.DataFrame) else None
    if defined is None or defined.height == 0:
        return None
    return defined.height, float(defined["ic"].mean())


def test_a_shipped_prediction_artifact_reproduces_its_registered_ic(intermediates_dir):
    """Registered daily-IC coverage and mean come from the parquet shipped beside them.

    This is the check `08_signal_method_comparison` makes on whichever set it selects; a
    fixture that fails here fails that notebook, and the notebook is right.
    """
    disagreements: list[str] = []
    for case_dir in _case_studies(intermediates_dir):
        db = sqlite3.connect(f"file:{case_dir / 'run_log' / 'registry.db'}?mode=ro", uri=True)
        try:
            columns = {row[1] for row in db.execute("PRAGMA table_info(prediction_metrics)")}
            if not {"ic_n_days", "ic_mean_daily"} <= columns:
                continue
            for prediction_hash, parquet in shipped_predictions(case_dir).items():
                row = db.execute(
                    "SELECT tr.spec_json, pm.ic_n_days, pm.ic_mean_daily "
                    "FROM prediction_sets ps "
                    "JOIN training_runs tr ON tr.training_hash = ps.training_hash "
                    "LEFT JOIN prediction_metrics pm ON pm.prediction_hash = ps.prediction_hash "
                    "WHERE ps.prediction_hash = ?",
                    (prediction_hash,),
                ).fetchone()
                if row is None:
                    continue  # covered by the unreachable-directory test above
                spec_json, n_days, mean_daily = row
                measured = _daily_ic(parquet, _spec_context(spec_json))
                if measured is None:
                    continue
                if n_days is None or mean_daily is None:
                    disagreements.append(
                        f"{case_dir.name} {prediction_hash}: no registered daily IC, "
                        f"artifact gives {measured[0]} days"
                    )
                    continue
                same_days = int(n_days) == measured[0]
                same_ic = abs(float(mean_daily) - measured[1]) <= REL_TOL * max(
                    1.0, abs(measured[1])
                )
                if not (same_days and same_ic):
                    disagreements.append(
                        f"{case_dir.name} {prediction_hash}: registered "
                        f"{int(n_days)} days / {float(mean_daily):.6f}, artifact gives "
                        f"{measured[0]} days / {measured[1]:.6f}"
                    )
        finally:
            db.close()
    assert not disagreements, (
        f"{len(disagreements)} shipped prediction artifact(s) disagree with their registered "
        "metrics. Run tests/reconcile_fixture_registry.py --apply.\n  "
        + "\n  ".join(disagreements[:20])
    )
