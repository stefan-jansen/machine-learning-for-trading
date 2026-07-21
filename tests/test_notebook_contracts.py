"""Tests for shared case-study SQL contracts."""

from __future__ import annotations

import sqlite3

from case_studies.utils.notebook_contracts import degenerate_prediction_sql


def test_degenerate_prediction_filter_is_null_safe() -> None:
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE candidates (prediction_hash TEXT);
        CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
        INSERT INTO candidates VALUES ('good'), ('bad');
        INSERT INTO fold_metrics VALUES ('good', 0.10), ('bad', NULL), (NULL, NULL);
        """
    )

    rows = db.execute(
        "SELECT c.prediction_hash FROM candidates c "
        f"WHERE 1=1 {degenerate_prediction_sql('c.prediction_hash')}"
    ).fetchall()
    db.close()

    assert rows == [("good",)]
