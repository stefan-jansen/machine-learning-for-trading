"""`prediction_admissibility` records the denominator `prediction_coverage` cannot hold.

`prediction_coverage.status` reads `complete` in all 6,480 rows across all nine registries.
Two mechanisms produce that. Registration raises when coverage is partial and `allow_partial`
defaults to False everywhere in production, so a non-complete evaluation is refused before the
row is written. And `evaluate_prediction_coverage` compares a prediction against
`expected_keys`, which the model family's own adapter builds from its own prepared fold inputs
after its own eligibility filtering - so `complete` means "the model produced what it set out
to produce" and never "the model covered the declared universe".

The declared-universe comparison is measured by `measure_prediction_cross_sections` and now
stored as numbers beside the verdict. On `sp500_equity_option_analytics` all 140 members ruled
inadmissible carry a `prediction_coverage` row reading `complete` with `n_missing = 0`, whose
`n_expected` (126,458) is exactly the narrowed numerator against a declared 248,460.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.coverage import CrossSectionReport
from case_studies.utils.notebook_contracts import (
    incompletely_registered_predictions,
    record_prediction_admissibility,
)

# The narrowed sequence-model member from the issue, and a whole one beside it.
NARROWED = CrossSectionReport(
    case_study="sp500_equity_option_analytics",
    label="fwd_ret_risk_adj_5d",
    split="validation",
    source="deep_learning/lstm_h64",
    expected=248_460,
    delivered=126_458,
    achievable=194_748,
    delivered_achievable=126_458,
    never_scored=("AAA",),
    partially_scored=(),
    entities_declared=549,
    per_fold=(),
)
WHOLE = CrossSectionReport(
    case_study="sp500_equity_option_analytics",
    label="fwd_ret_risk_adj_5d",
    split="validation",
    source="gbm/default_mae",
    expected=248_460,
    delivered=248_460,
    achievable=194_748,
    delivered_achievable=194_748,
    never_scored=(),
    partially_scored=(),
    entities_declared=549,
    per_fold=(),
)


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    root = tmp_path / "case_study"
    (root / "run_log").mkdir(parents=True)
    sqlite3.connect(str(root / "run_log" / "registry.db")).close()
    return root


def _row(case_dir: Path, prediction_hash: str) -> dict:
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT * FROM prediction_admissibility WHERE prediction_hash = ?",
            (prediction_hash,),
        ).fetchone()
    return dict(row)


def test_a_refused_member_stores_both_numerator_and_denominator(case_dir):
    assert (
        record_prediction_admissibility(
            case_dir,
            admitted=[],
            short={"pred_narrowed": NARROWED.summary()},
            measured={"pred_narrowed": NARROWED},
        )
        == []
    )
    row = _row(case_dir, "pred_narrowed")
    assert row["admitted"] == 0
    assert row["n_declared"] == 248_460
    assert row["n_delivered"] == 126_458
    assert row["n_offered"] == 194_748
    assert row["n_delivered_offered"] == 126_458
    assert row["n_entities_declared"] == 549


def test_an_admitted_member_stores_its_numbers_too(case_dir):
    """A reader comparing the two needs the row for the member that passed as well."""
    record_prediction_admissibility(
        case_dir, admitted=["pred_whole"], short={}, measured={"pred_whole": WHOLE}
    )
    row = _row(case_dir, "pred_whole")
    assert row["admitted"] == 1
    assert (row["n_declared"], row["n_delivered"]) == (248_460, 248_460)


def test_an_unmeasured_member_stores_nulls_not_zeros(case_dir):
    """Zero of zero is a measurement; "never measured" is not."""
    record_prediction_admissibility(case_dir, admitted=["pred_unmeasured"], short={}, measured={})
    row = _row(case_dir, "pred_unmeasured")
    assert row["admitted"] == 1
    assert row["n_declared"] is None
    assert row["n_delivered"] is None


def test_the_numbers_survive_a_second_recording(case_dir):
    record_prediction_admissibility(
        case_dir, admitted=["pred_whole"], short={}, measured={"pred_whole": WHOLE}
    )
    record_prediction_admissibility(
        case_dir,
        admitted=[],
        short={"pred_whole": NARROWED.summary()},
        measured={"pred_whole": NARROWED},
    )
    row = _row(case_dir, "pred_whole")
    assert row["admitted"] == 0
    assert row["n_delivered"] == 126_458


def _coverage_registry(case_dir: Path, **columns) -> None:
    db_path = case_dir / "run_log" / "registry.db"
    with sqlite3.connect(str(db_path)) as db:
        db.executescript(
            """
            CREATE TABLE prediction_coverage (
                prediction_hash TEXT PRIMARY KEY, expected_key_digest TEXT,
                actual_key_digest TEXT, n_expected INTEGER, n_actual INTEGER,
                n_duplicates INTEGER, n_missing INTEGER, n_extra INTEGER,
                n_folds_expected INTEGER, status TEXT
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
            """
        )
        db.execute(
            "INSERT INTO prediction_coverage VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "pred",
                columns.get("expected_key_digest", "d"),
                columns.get("actual_key_digest", "d"),
                100,
                100,
                columns.get("n_duplicates", 0),
                columns.get("n_missing", 0),
                columns.get("n_extra", 0),
                1,
                columns.get("status", "complete"),
            ),
        )
        db.execute("INSERT INTO fold_metrics VALUES ('pred', 0.1)")
    (case_dir / "run_log" / "predictions" / "pred").mkdir(parents=True, exist_ok=True)
    (case_dir / "run_log" / "predictions" / "pred" / "predictions.parquet").write_bytes(b"PAR1")


def test_a_row_claiming_complete_while_carrying_a_gap_is_caught(case_dir):
    """The gate reads the counts, not only the verdict the counts were drawn from."""
    _coverage_registry(case_dir, status="complete", n_missing=7)
    short = incompletely_registered_predictions(case_dir, ["pred"])
    assert "missing" in short["pred"]


def test_a_row_whose_key_sets_disagree_is_caught(case_dir):
    _coverage_registry(case_dir, expected_key_digest="a", actual_key_digest="b")
    assert incompletely_registered_predictions(case_dir, ["pred"]) == {
        "pred": "coverage key set differs from the expected one"
    }


def test_a_whole_row_still_passes(case_dir):
    _coverage_registry(case_dir)
    assert incompletely_registered_predictions(case_dir, ["pred"]) == {}
