"""A coverage digest records the rendering that produced it.

`_canonical_key_column` decides how each key column is rendered before it is digested, and it
has changed once: rows written before it normalized temporal columns cast them straight to
String, so a `Date` rendered `2016-01-29` and a `Datetime("ms")` of the same instant rendered
`2016-01-29 00:00:00.000`. `value_digest` is taken over the rendered frame, so the same key set
digests two ways - and the two are simply unequal, with nothing saying why.

That reaches a consumer. `sp500_options/11_model_analysis` groups predictions by
`expected_key_digest` to decide which checkpoints were scored on identical rows and are
therefore comparable. When #1065 was filed, grouping each registry by the stored digest against
one uniform rendering split one set in sp500_options, one in us_firm_characteristics and two in
cme_futures, and nothing raised: the guard on each group only rejects members that disagree on
`n_expected`, `n_actual` or `n_folds`, and the split halves agree on all three.

Measured again 2026-09-07, after the re-run sweep #1065 anticipated: 4,303 of 4,469 coverage
rows now reproduce under the current rendering and 166 under the legacy one - and the 166 are
all `us_equities_panel`'s, exactly the residue #1065 named. **No registry carries a mix any
more**, so the splits are gone and nothing here refuses a fleet that exists today. What is
fixed is the mechanism: the next rendering change arrives as an error naming its cause.

ml4t/agent-workspace#1065.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.registry.completeness import (
    KEY_DIGEST_RENDERING,
    LEGACY_KEY_DIGEST_RENDERING,
    coverage_key_digest,
    evaluate_prediction_coverage,
    key_digest_rendering,
    key_digest_value,
    require_comparable_key_digests,
)
from case_studies.utils.registry.registration import (
    register_prediction_set,
    register_training_run,
)

CASE_STUDY = "etfs"
LABEL = "fwd_ret_21d"


def _keys(dtype) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "timestamp": [date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 3)],
            "fold": [0, 0, 0, 0],
        }
    ).with_columns(pl.col("timestamp").cast(dtype))


def _predictions(dtype) -> pl.DataFrame:
    return _keys(dtype).with_columns(
        pl.Series("prediction", [0.1, -0.2, 0.3, -0.4]),
        pl.Series("actual", [0.01, -0.02, 0.03, -0.04]),
    )


def test_a_digest_names_the_rendering_that_produced_it() -> None:
    coverage = evaluate_prediction_coverage(_keys(pl.Date), _predictions(pl.Date))
    assert coverage.expected_key_digest.startswith(f"{KEY_DIGEST_RENDERING}:")
    assert key_digest_rendering(coverage.expected_key_digest) == KEY_DIGEST_RENDERING


def test_an_unprefixed_digest_names_no_rendering_on_its_own() -> None:
    """A bare digest says only that it predates the stamp, and every stored row is bare.

    Measured across the fleet 2026-09-07: 4,303 of the 4,469 stored digests reproduce under
    the current rendering and 166 - all `us_equities_panel`'s - under `k1`, and all 4,469 are
    stored without a prefix. Reading bare as `k1` declared a rendering change on the 4,303
    where there was none, which refused the next checkpoint registered against any of them.
    """
    assert key_digest_rendering("0adbebe7d4b67c11") is None


def test_a_bare_digest_is_identified_by_recomputing_it() -> None:
    """The measurement that replaces the assumption: digest the keys both ways and see."""
    keys = _keys(pl.Date)
    current = key_digest_value(coverage_key_digest(keys))
    legacy = key_digest_value(coverage_key_digest(keys, rendering=LEGACY_KEY_DIGEST_RENDERING))
    assert current != legacy, "the two renderings must actually differ on this fixture"
    assert key_digest_rendering(current, expected_keys=keys) == KEY_DIGEST_RENDERING
    assert key_digest_rendering(legacy, expected_keys=keys) == LEGACY_KEY_DIGEST_RENDERING


def test_a_bare_digest_of_other_keys_stays_unidentified() -> None:
    """Neither rendering reproduces it, so it holds different keys - a different question."""
    assert key_digest_rendering("0adbebe7d4b67c11", expected_keys=_keys(pl.Date)) is None


def test_the_same_keys_digest_alike_across_temporal_dtypes() -> None:
    """The normalization the rendering exists for, still doing its job.

    A `Date` and a `Datetime("ms")` of the same instant are one key set, and the whole reason
    `_canonical_key_column` normalizes before casting is that they used to digest differently.
    """
    as_date = evaluate_prediction_coverage(_keys(pl.Date), _predictions(pl.Date))
    as_datetime = evaluate_prediction_coverage(
        _keys(pl.Datetime("ms")), _predictions(pl.Datetime("ms"))
    )
    assert as_date.expected_key_digest == as_datetime.expected_key_digest


class TestComparingAcrossRenderingsIsAnError:
    def test_one_rendering_compares_fine(self) -> None:
        require_comparable_key_digests(["k2:aaaa", "k2:bbbb"], what="a population")

    def test_a_mix_is_refused(self) -> None:
        """The silent inequality this converts into an error."""
        with pytest.raises(ValueError, match=r"spans coverage-key renderings \['k1', 'k2'\]"):
            require_comparable_key_digests(["k2:aaaa", "k1:bbbb"], what="a population")

    def test_a_bare_digest_alone_makes_no_claim(self) -> None:
        """Without the keys there is nothing to measure, and guessing refuses the fleet."""
        require_comparable_key_digests(["k2:aaaa", "0adbebe7d4b67c11"], what="a population")

    def test_a_bare_digest_is_refused_once_the_keys_identify_it(self) -> None:
        """With the keys in hand the same pair is a genuine rendering change, and says so."""
        keys = _keys(pl.Date)
        legacy = key_digest_value(coverage_key_digest(keys, rendering=LEGACY_KEY_DIGEST_RENDERING))
        with pytest.raises(ValueError, match=r"spans coverage-key renderings \['k1', 'k2'\]"):
            require_comparable_key_digests(
                [coverage_key_digest(keys), legacy], what="a population", expected_keys=keys
            )

    def test_a_bare_digest_of_the_current_rendering_compares_fine(self) -> None:
        """The regression: 4,303 fleet rows are this, and none of them is a rendering change."""
        keys = _keys(pl.Date)
        require_comparable_key_digests(
            [coverage_key_digest(keys), key_digest_value(coverage_key_digest(keys))],
            what="a population",
            expected_keys=keys,
        )

    def test_nulls_and_blanks_are_not_a_rendering(self) -> None:
        """A row with no digest is a different complaint, and one this must not raise for."""
        require_comparable_key_digests(["k2:aaaa", None, ""], what="a population")


def test_a_rendering_change_within_a_training_run_is_refused(tmp_path: Path) -> None:
    """Where the split would do the most damage: two checkpoints of one run, one contract.

    A consumer grouping a training run's checkpoints by eligibility would see two contracts
    where there is one, and the dimension check on each group cannot catch it because the two
    halves agree on every dimension.
    """
    training_hash = register_training_run(
        CASE_STUDY,
        {
            "identity_version": 2,
            "execution_tier": "canonical",
            "family": "linear",
            "label": LABEL,
            "seed": 42,
            "config_name": "ridge_default",
            "computation": {"model": {"class": "Ridge"}},
        },
        case_dir=tmp_path,
    )

    def publish(checkpoint: int) -> str:
        return register_prediction_set(
            CASE_STUDY,
            training_hash,
            checkpoint_kind="epoch",
            checkpoint_value=checkpoint,
            split="validation",
            predictions=_predictions(pl.Date),
            expected_keys=_keys(pl.Date),
            case_dir=tmp_path,
        )

    publish(1)
    publish(2)  # a second checkpoint under the same rendering is ordinary

    # Now rewrite the stored digests as the previous rendering of these same keys, which is
    # exactly what a rendering change leaves behind, and register a third checkpoint.
    _rewrite_stored_digests(
        tmp_path,
        training_hash,
        key_digest_value(
            coverage_key_digest(_keys(pl.Date), rendering=LEGACY_KEY_DIGEST_RENDERING)
        ),
    )
    with pytest.raises(ValueError, match="spans coverage-key renderings"):
        publish(3)


def test_a_checkpoint_joins_siblings_written_before_the_stamp(tmp_path: Path) -> None:
    """The regression this guard caused, as the fleet meets it.

    Every one of the 4,469 coverage rows in the nine registries was written before the
    rendering was stamped, so every one is bare - and 4,303 of them under the rendering in
    force today. Reading bare as a changed rendering refused the next checkpoint registered
    against any of them, which is every re-run and every resumed sweep.
    """
    training_hash = register_training_run(
        CASE_STUDY,
        {
            "identity_version": 2,
            "execution_tier": "canonical",
            "family": "linear",
            "label": LABEL,
            "seed": 42,
            "config_name": "ridge_default",
            "computation": {"model": {"class": "Ridge"}},
        },
        case_dir=tmp_path,
    )

    def publish(checkpoint: int) -> str:
        return register_prediction_set(
            CASE_STUDY,
            training_hash,
            checkpoint_kind="epoch",
            checkpoint_value=checkpoint,
            split="validation",
            predictions=_predictions(pl.Date),
            expected_keys=_keys(pl.Date),
            case_dir=tmp_path,
        )

    first = publish(1)
    _rewrite_stored_digests(
        tmp_path, training_hash, key_digest_value(coverage_key_digest(_keys(pl.Date)))
    )
    assert publish(2) != first
    assert publish(1) == first, "and re-registering the same checkpoint is still idempotent"


def _rewrite_stored_digests(case_dir: Path, training_hash: str, digest: str) -> None:
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(case_dir / "run_log" / "registry.db")) as db:
        db.execute(
            "UPDATE prediction_coverage SET expected_key_digest = ? "
            "WHERE prediction_hash IN (SELECT prediction_hash FROM prediction_sets "
            "WHERE training_hash = ?)",
            (digest, training_hash),
        )
        db.commit()
