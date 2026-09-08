"""Two prediction artifacts of one (split, label) are scored against one outcome.

`us_equities_panel` `(validation, fwd_ret_1d)` ships seven artifacts in two clusters:
four `gbm` over 8 symbols storing the target as `Float32`, three `linear` over 56
storing it as `Float64`. Over the 5,212 keys they share they disagree on the realized
target by 2.6e-08 - 260 times the 1e-10 at which
`14_latent_factors/09_case_study_insights::paired_daily_ic` rejects a pair
(ml4t/agent-workspace#288).

Two things have to hold for that to be harmless, and only one of them was ever written
down. Within a cluster the artifacts must agree exactly, or two streams on the same keys
are scored against different outcomes while every breadth and key check passes. And the
artifact a notebook selects by metric rank has to sit on the cluster
`tests/fixtures/seed_results.py` seeded its synthetic sets onto, or the supervised side
and the latent side are comparing different cross-sections.

The second currently holds by four gbm artifacts outranking three linear ones, which is
a property of the metrics rather than of anything declared - #288's own point. It is
checked here so that the day it stops holding arrives as this test rather than as
`ch14-15` red with a message about neither.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.fixture_registry import (
    choose_reference_panel,
    panel_signature,
    prediction_panels,
    within_panel_target_gap,
)


def _case_studies(intermediates_dir: Path | None) -> list[Path]:
    if intermediates_dir is None:
        pytest.skip("no test-data intermediates on this checkout")
    return sorted(
        entry
        for entry in intermediates_dir.iterdir()
        if (entry / "run_log" / "registry.db").is_file()
    )


def test_artifacts_on_one_panel_agree_on_the_realized_target(intermediates_dir):
    """Exactly, not nearly. They carry the same keys, so any gap is two outcomes."""
    disagreements = []
    for case_dir in _case_studies(intermediates_dir):
        for (split, label), group in prediction_panels(case_dir).items():
            for panel in group["panels"]:
                if len(panel["entries"]) < 2:
                    continue
                gap = within_panel_target_gap(panel)
                if gap:
                    disagreements.append(
                        f"{case_dir.name} ({split}, {label}) {panel['families']} "
                        f"{panel['entities']} entities: {gap:.3e} across {panel['hashes']}"
                    )
    assert not disagreements, (
        "artifacts on one cross-section disagree on the target:\n  " + "\n  ".join(disagreements)
    )


def test_the_top_ranked_artifact_sits_on_the_panel_the_seeder_aligns_to(intermediates_dir):
    """Where a group ships two cross-sections, the selected one is the seeded one.

    `09_case_study_insights` picks its supervised side by `ic_mean_daily` rank and joins
    it against the seeded latent panel. `_reference_panels` seeds onto the cluster the
    most artifacts share; if the rank leader is on the other cluster, the two sides are
    joined across a 2.6e-08 target gap and the notebook refuses.
    """
    misaligned = []
    for case_dir in _case_studies(intermediates_dir):
        panels_by_group = prediction_panels(case_dir)
        db = sqlite3.connect(f"file:{case_dir / 'run_log' / 'registry.db'}?mode=ro", uri=True)
        try:
            for (split, label), group in panels_by_group.items():
                if len(group["panels"]) < 2:
                    continue
                ranked = None
                best = None
                for panel in group["panels"]:
                    for entry in panel["entries"]:
                        row = db.execute(
                            "SELECT ic_mean_daily FROM prediction_metrics WHERE prediction_hash = ?",
                            (entry["prediction_hash"],),
                        ).fetchone()
                        if row is None or row[0] is None:
                            continue
                        if best is None or float(row[0]) > best:
                            best = float(row[0])
                            ranked = panel
                if ranked is None or ranked["is_reference"]:
                    continue
                reference = next(p for p in group["panels"] if p["is_reference"])
                misaligned.append(
                    f"{case_dir.name} ({split}, {label}): rank leader is on the "
                    f"{ranked['entities']}-entity {ranked['families']} panel, the seeder aligns to "
                    f"the {reference['entities']}-entity {reference['families']} one; the two "
                    f"disagree on the target by {group['cross_panel_target_gap']:.3e}"
                )
        finally:
            db.close()
    assert not misaligned, (
        "a notebook selecting by metric rank would join across two cross-sections:\n  "
        + "\n  ".join(misaligned)
    )


# --- the mechanism, on a fixture built to break it -------------------------------
#
# Both checks above pass against the committed fixture, so on their own they do not
# show they can fail. These build the two states they exist to catch.

SCHEMA = """
CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, family TEXT, label TEXT);
CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT);
CREATE TABLE prediction_metrics (prediction_hash TEXT PRIMARY KEY, ic_mean_daily REAL);
"""


def _panelled_case(tmp_path, artifacts):
    """A case study shipping *artifacts*: (hash, family, symbols, targets, ic).

    Under an intermediates root of its own, not directly under ``tmp_path``. The
    fixture-wide checks take the root and walk every case study in it, and
    ``tmp_path.parent`` is the session-wide temp directory pytest shares with every
    other test - so passing that root made this test read whatever another test had
    just written, which in CI was a deliberately truncated parquet from
    ``test_fixture_registry_prune.py`` and a `ComputeError` here.
    """
    import datetime

    import polars as pl

    tmp_path = tmp_path / "intermediates" / "case_study"
    (tmp_path / "run_log").mkdir(parents=True)
    db = sqlite3.connect(str(tmp_path / "run_log" / "registry.db"))
    db.executescript(SCHEMA)
    for prediction_hash, family, symbols, targets, ic in artifacts:
        db.execute(
            "INSERT INTO training_runs VALUES (?,?,?)",
            (f"t_{prediction_hash}", family, "fwd_ret_1d"),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?)",
            (prediction_hash, f"t_{prediction_hash}", "validation"),
        )
        db.execute("INSERT INTO prediction_metrics VALUES (?,?)", (prediction_hash, ic))
        directory = tmp_path / "run_log" / "predictions" / prediction_hash
        directory.mkdir(parents=True)
        pl.DataFrame(
            {
                "symbol": symbols,
                "timestamp": [datetime.date(2020, 1, 1 + i) for i in range(len(symbols))],
                "prediction": [0.1 * i for i in range(len(symbols))],
                "actual": targets,
            }
        ).write_parquet(directory / "predictions.parquet")
    db.commit()
    db.close()
    return tmp_path


def test_the_within_panel_check_catches_two_outcomes_on_one_cross_section(tmp_path):
    case_dir = _panelled_case(
        tmp_path,
        [
            ("aaaa", "gbm", ["A", "B"], [1.0, 2.0], 0.3),
            ("bbbb", "gbm", ["A", "B"], [1.0, 2.5], 0.2),
        ],
    )
    panels = prediction_panels(case_dir)[("validation", "fwd_ret_1d")]["panels"]
    assert len(panels) == 1
    assert within_panel_target_gap(panels[0]) == pytest.approx(0.5)


def test_the_alignment_check_catches_a_rank_leader_off_the_seeded_panel(tmp_path):
    """Two artifacts on one panel, one on another, and the outlier ranks first."""
    case_dir = _panelled_case(
        tmp_path,
        [
            ("aaaa", "gbm", ["A", "B"], [1.0, 2.0], 0.1),
            ("bbbb", "gbm", ["A", "B"], [1.0, 2.0], 0.2),
            ("cccc", "linear", ["A", "B", "C"], [1.0, 2.0, 3.0], 0.9),
        ],
    )
    with pytest.raises(AssertionError, match="join across two cross-sections"):
        test_the_top_ranked_artifact_sits_on_the_panel_the_seeder_aligns_to(case_dir.parent)


def test_the_seeder_and_the_check_read_one_panel_rule(tmp_path):
    """`seed_results` must pick the same panel this file calls the reference one.

    Two implementations of "most artifacts, then largest, then lowest hash" would drift,
    and the drift is invisible: the seeded sets simply stop being joinable with the
    copied ones.
    """
    from tests.fixtures import seed_results

    assert seed_results.choose_reference_panel is choose_reference_panel
    assert seed_results.panel_signature is panel_signature
