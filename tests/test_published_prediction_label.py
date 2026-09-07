"""A published prediction frame states which label it was produced under.

The declared validation axis is sized by the label's own outcome horizon, because the last
validation fold ends early enough that a decision's label is observable before the holdout
opens. Two labels on one case study therefore declare different session counts - measured on
`crypto_perps_funding`, `fwd_ret_8h` declares 2,189 validation sessions and `fwd_ret_24h`
2,187 - so checking one label's artifact against another's declaration reports a small,
plausible, entirely spurious gap.

The mistake is invisible in one direction: a shorter declared horizon makes the observed frame
a strict subset of the declaration, which no condition in the coverage gate can tell from a
real gap. `case_studies/utils/coverage.py` has refused a frame whose own `label` column
disagrees with the label it was asked to check against since that was found, and skipped the
cross-check where the column is absent - which was everywhere. These tests pin that the column
is now written, and that writing it costs nothing already registered.

ml4t/agent-workspace#887.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.artifact_digest import published_prediction_digest, value_digest
from case_studies.utils.registry.registration import (
    register_prediction_set,
    register_training_run,
)

CASE_STUDY = "etfs"
LABEL = "fwd_ret_21d"
OTHER_LABEL = "fwd_ret_5d"


def _keys() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "timestamp": [date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 3)],
            "fold": [0, 0, 0, 0],
        }
    )


def _predictions() -> pl.DataFrame:
    return _keys().with_columns(
        pl.Series("prediction", [0.1, -0.2, 0.3, -0.4]),
        pl.Series("actual", [0.01, -0.02, 0.03, -0.04]),
    )


def _training(case_dir: Path, *, label: str = LABEL, config: str = "ridge_default") -> str:
    return register_training_run(
        CASE_STUDY,
        {
            "identity_version": 2,
            "execution_tier": "canonical",
            "family": "linear",
            "label": label,
            "seed": 42,
            "config_name": config,
            "computation": {"model": {"class": "Ridge", "config": config}},
        },
        case_dir=case_dir,
    )


def _publish(case_dir: Path, training_hash: str, **kwargs) -> str:
    return register_prediction_set(
        CASE_STUDY,
        training_hash,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=kwargs.pop("predictions", _predictions()),
        expected_keys=_keys(),
        case_dir=case_dir,
        **kwargs,
    )


def _frame(case_dir: Path, prediction_hash: str) -> pl.DataFrame:
    return pl.read_parquet(
        case_dir / "run_log" / "predictions" / prediction_hash / "predictions.parquet"
    )


def _recorded_digest(case_dir: Path, prediction_hash: str) -> str:
    with closing(sqlite3.connect(case_dir / "run_log" / "registry.db")) as db:
        return db.execute(
            "SELECT artifact_digest FROM prediction_coverage WHERE prediction_hash = ?",
            (prediction_hash,),
        ).fetchone()[0]


def test_a_published_frame_states_its_label(tmp_path: Path) -> None:
    """The whole fix: the artifact says which declaration applies to it."""
    p_hash = _publish(tmp_path, _training(tmp_path), label=LABEL)
    frame = _frame(tmp_path, p_hash)
    assert "label" in frame.columns
    assert frame.get_column("label").unique().to_list() == [LABEL]


def test_the_label_comes_from_the_training_run_when_the_caller_omits_it(tmp_path: Path) -> None:
    """`publish_predictions` takes `label` optionally and most callers do not pass it.

    `training_runs.label` is authoritative and always present, so an omitted argument must
    not mean an unlabelled artifact - that is the state the whole fleet is in.
    """
    p_hash = _publish(tmp_path, _training(tmp_path))
    assert _frame(tmp_path, p_hash).get_column("label").unique().to_list() == [LABEL]


def test_a_frame_whose_label_disagrees_with_the_publication_is_refused(tmp_path: Path) -> None:
    """Two statements about what was fitted, and quietly picking one is the original mistake."""
    mislabelled = _predictions().with_columns(pl.lit(OTHER_LABEL).alias("label"))
    with pytest.raises(ValueError, match="is being published as"):
        _publish(tmp_path, _training(tmp_path), predictions=mislabelled, label=LABEL)


# ---------------------------------------------------------------------------
# ...and it strands nothing, which is why #887 was deferred
# ---------------------------------------------------------------------------


def test_the_recorded_digest_is_the_digest_of_the_frame_without_the_label(
    tmp_path: Path,
) -> None:
    """The column is data about the frame, so it is outside the frame's content identity.

    Measured on the fleet 2026-09-07: all 4,469 `prediction_coverage.artifact_digest` rows
    across the nine registries reproduce under this digest, so writing the column invalidates
    none of them.
    """
    p_hash = _publish(tmp_path, _training(tmp_path), label=LABEL)
    assert _recorded_digest(tmp_path, p_hash) == value_digest(_predictions())
    assert published_prediction_digest(_frame(tmp_path, p_hash)) == value_digest(_predictions())


def test_republishing_a_labelled_artifact_is_not_a_conflict(tmp_path: Path) -> None:
    """The immutability check re-reads the file, so it has to digest it the same way.

    Registering an identity that already exists is idempotent by design - a re-run that
    re-registers rather than skipping must not be told its own artifact changed.
    """
    training_hash = _training(tmp_path)
    first = _publish(tmp_path, training_hash, label=LABEL)
    assert _publish(tmp_path, training_hash, label=LABEL) == first


def test_the_immutability_check_still_catches_a_changed_frame(tmp_path: Path) -> None:
    """The control for the test above: excluding the label must not excuse a real change."""
    training_hash = _training(tmp_path)
    _publish(tmp_path, training_hash, label=LABEL)
    moved = _predictions().with_columns(pl.Series("prediction", [9.0, 9.0, 9.0, 9.0]))
    with pytest.raises(ValueError, match="immutable prediction artifact conflict"):
        _publish(tmp_path, training_hash, predictions=moved, label=LABEL)


def test_the_coverage_guard_now_has_evidence_to_refuse_on(tmp_path: Path) -> None:
    """The reason the column is worth writing, stated as the behaviour it restores.

    `_reject_label_mismatch` was written for this and was a no-op wherever the column was
    absent, which was every published frame in the fleet.
    """
    from case_studies.utils.coverage import CoverageError, _reject_label_mismatch

    published = _frame(tmp_path, _publish(tmp_path, _training(tmp_path), label=LABEL))
    _reject_label_mismatch(
        published, case_study=CASE_STUDY, label=LABEL, split="validation", source="predictions"
    )
    with pytest.raises(CoverageError, match="not 'fwd_ret_5d'"):
        _reject_label_mismatch(
            published,
            case_study=CASE_STUDY,
            label=OTHER_LABEL,
            split="validation",
            source="predictions",
        )
    # Without the column there is nothing to check against, which is the state this closes.
    _reject_label_mismatch(
        published.drop("label"),
        case_study=CASE_STUDY,
        label=OTHER_LABEL,
        split="validation",
        source="predictions",
    )


def test_a_caller_label_the_training_run_contradicts_is_refused(tmp_path: Path) -> None:
    """The parent run is authoritative, so an argument disagreeing with it is a mistake.

    Believing the argument would stamp the artifact with a label its own training run was
    not fitted under, and `_reject_label_mismatch` would then accept the wrong declaration
    and refuse the right one - the mistake the column exists to catch, made one layer up.
    No family runner passes `label` at all; every one of them reaches this through
    `publish_predictions`, which leaves it None.
    """
    with pytest.raises(ValueError, match="is registered against label 'fwd_ret_21d'"):
        _publish(tmp_path, _training(tmp_path), label=OTHER_LABEL)


def test_a_caller_label_agreeing_with_the_training_run_is_fine(tmp_path: Path) -> None:
    """The control: passing the right label is not an error, it is redundant."""
    p_hash = _publish(tmp_path, _training(tmp_path, label=OTHER_LABEL), label=OTHER_LABEL)
    assert _frame(tmp_path, p_hash).get_column("label").unique().to_list() == [OTHER_LABEL]


def test_the_published_artifact_re_registers_as_it_was_read(tmp_path: Path) -> None:
    """A replayed registration hands back the frame it wrote, label column and all.

    `schema_json` is recorded from the frame handed in and compared against the next
    checkpoint's, so recording a schema without the label while writing a parquet that has
    it made the published artifact fail against itself: reading it back and registering it
    again raised "prediction schema differs from an existing checkpoint" on identical
    content. Both are now taken without the column, the same way its digest is.
    """
    training_hash = _training(tmp_path)
    first = _publish(tmp_path, training_hash, label=LABEL)
    published = _frame(tmp_path, first)
    assert "label" in published.columns
    assert _publish(tmp_path, training_hash, predictions=published) == first
    assert _recorded_digest(tmp_path, first) == value_digest(_predictions())


def test_a_read_back_artifact_published_under_another_run_is_still_refused(
    tmp_path: Path,
) -> None:
    """The control: the label travels with the frame and still has to agree."""
    published = _frame(tmp_path, _publish(tmp_path, _training(tmp_path), label=LABEL))
    other = _training(tmp_path, label=OTHER_LABEL, config="ridge_other")
    with pytest.raises(ValueError, match="is being published as"):
        _publish(tmp_path, other, predictions=published)
