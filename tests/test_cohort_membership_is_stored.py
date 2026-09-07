"""A stored cohort correction can be checked against a cohort in hand, without a replay.

`cohort_metrics.member_digest` is a sha256 over the cohort's member hashes, and the members
are the thing it is a digest of. Storing only the digest means verifying a row requires
rebuilding the member list from the registry and hashing it, and that rebuild runs through
every selection rule in force when the row was written: `BacktestExplorer.best` applies
coverage, excluded families and the tradeless-backtest rule under a population scope, and
the Sharpe matrix then drops rows that cannot be aligned. When any of those has moved since,
the digest cannot be reproduced, and a genuine membership disagreement is indistinguishable
from a rule change.

Read-only fleet count on 2026-09-07: 109 `cohort_metrics` rows across five registries, all
109 carrying a `member_digest` and none carrying the members it covers.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.registry import register_cohort_metrics
from case_studies.utils.registry.store import _open_registry
from case_studies.utils.uncertainty import cohort_member_digest

MEMBERS = ["bt_alpha", "bt_beta", "bt_gamma"]


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    case_dir = tmp_path / "unit_cs"
    (case_dir / "run_log").mkdir(parents=True)
    # The registry's own schema, so a column this test reads back has to be one the
    # producer actually writes rather than one the fixture invented.
    _open_registry(case_dir).close()
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.executemany(
            "INSERT INTO backtest_runs (backtest_hash, prediction_hash, created_at) "
            "VALUES (?, 'pred', '2026-01-01T00:00:00Z')",
            [(member,) for member in MEMBERS],
        )
    return case_dir


def _register(case_dir: Path, members: list[str]) -> dict:
    register_cohort_metrics(
        "unit_cs",
        [
            {
                "cohort_type": "family",
                "stage": "signal",
                "label": "fwd_ret_5d",
                "family": "linear",
                "metrics": {
                    "leader_hash": members[0],
                    "k_variants": len(members),
                    "periods_per_year": 252.0,
                    "member_digest": cohort_member_digest(members),
                    "members_json": json.dumps(sorted(members)),
                },
            }
        ],
        case_dir=case_dir,
        prune_dangling=False,
    )
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.row_factory = sqlite3.Row
        return dict(db.execute("SELECT * FROM cohort_metrics").fetchone())


def test_the_members_a_digest_covers_are_stored_beside_it(case_dir: Path) -> None:
    """The stored fact the digest is a digest of."""
    row = _register(case_dir, MEMBERS)

    assert json.loads(row["members_json"]) == sorted(MEMBERS)
    assert row["member_digest"] == cohort_member_digest(json.loads(row["members_json"]))


def test_a_disagreement_names_the_members_that_differ(case_dir: Path) -> None:
    """What a digest comparison cannot do, and the reason the members are worth storing.

    Two cohorts of the same size with one member swapped produce different digests and an
    identical `k_variants`. The digest says they differ; it cannot say whether one member
    moved or the selection rule that assembles the cohort did.
    """
    from case_studies.utils.uncertainty import cohort_membership_diff

    row = _register(case_dir, MEMBERS)
    in_hand = ["bt_alpha", "bt_beta", "bt_delta"]

    assert row["member_digest"] != cohort_member_digest(in_hand)
    missing, extra = cohort_membership_diff(json.loads(row["members_json"]), in_hand)
    assert missing == ["bt_gamma"], "stored, and not in the cohort the reader assembled"
    assert extra == ["bt_delta"], "assembled, and not in the cohort the correction covers"


def test_an_agreeing_cohort_reports_no_difference(case_dir: Path) -> None:
    """Guards the premise: a diff that always found something would pass the test above."""
    from case_studies.utils.uncertainty import cohort_membership_diff

    row = _register(case_dir, MEMBERS)

    assert cohort_membership_diff(json.loads(row["members_json"]), reversed(MEMBERS)) == ([], [])


def test_a_row_written_before_the_members_were_stored_says_so(case_dir: Path) -> None:
    """The legacy row, which is every row in the fleet today.

    It has to be distinguishable from a row whose cohort is empty, because the two call for
    different things: one cannot be verified at all, the other verifies trivially.
    """
    from case_studies.utils.uncertainty import stored_cohort_members

    assert stored_cohort_members(None) is None
    assert stored_cohort_members(json.dumps([])) == []
    assert stored_cohort_members(json.dumps(sorted(MEMBERS))) == sorted(MEMBERS)


def test_the_producer_stores_the_members_it_digests(case_dir: Path) -> None:
    """End to end: what `compute_cohort_metrics` emits is what a reader verifies against.

    Persisting the members from a different list than the digest covers would be worse than
    not persisting them, so the two come from one place.
    """
    from datetime import date, timedelta

    import numpy as np
    import polars as pl

    from case_studies.utils.uncertainty import compute_cohort_metrics

    rng = np.random.default_rng(4)
    start = date(2020, 1, 1)
    cohort = {
        name: pl.DataFrame(
            {
                "timestamp": [start + timedelta(days=i) for i in range(200)],
                "ret": rng.normal(0.0005, 0.01, 200),
            }
        )
        for name in MEMBERS
    }

    out = compute_cohort_metrics(cohort, periods_per_year=252)

    assert json.loads(out["members_json"]) == sorted(MEMBERS)
    assert out["member_digest"] == cohort_member_digest(json.loads(out["members_json"]))
