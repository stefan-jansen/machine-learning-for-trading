"""A fold's index is an input to the fit, so renumbering the windows is a refit.

Six sites derive a fold's random seed from the fold's *index*. Renumbering the folds -
which `ml4t-diagnostic` 0.1.4 did, emitting the windows chronologically so fold 0 became
the earliest rather than the most recent - therefore changes the seed each window is
fitted under for `tabular_dl`, `latent_factors`, `deep_learning`, `darts_forecasting` and
any reduced `gbm` or `linear` run. A relabelling of stored results is not
equivalence-preserving for those families.

None of the six leaves a signal in any artifact. The fold's contribution to the seed is
consumed and discarded, so a renumbered run looks correct everywhere it is written down,
and a registry that finds a matching identity and skips the fit never recomputes the one
thing that would disagree. That is what makes "the identity reproduces" no evidence at all
here.

**This is the guard, not the fix.** Seeding from the window boundaries instead would make a
renumber stop being a computation change and is the right design; it also moves fitted
values for those families in every case study, and that trade was declined on 2026-09-06.
So the derivation is pinned where it is, and a change to it - or to the numbering it reads
- fails here rather than being discovered after a migration has been built. See #1056.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from case_studies.utils import (
    darts_forecasting,
    deep_learning,
    folds,
    gbm,
    tabular_dl,
)
from case_studies.utils.folds import _subsample_index, fold_seed
from case_studies.utils.latent_factors import cv as latent_cv

# The six sites, as the modules that reach the derivation. `folds` carries
# `_subsample_index`, the shared helper `iter_raw_folds` and `training_labels_for_split`
# both call, so a reduced run in ANY family draws a different training subsample - and
# through the second caller a different fitted hyperparameter - when its folds are
# renumbered.
SEEDING_MODULES = (
    tabular_dl,
    latent_cv,
    deep_learning,
    darts_forecasting,
    gbm,
    folds,
)


def test_the_seed_a_fold_is_fitted_under_is_the_base_plus_its_index() -> None:
    """The derivation itself, pinned. Changing it moves every seed in five families."""
    assert [fold_seed(42, fold) for fold in range(6)] == [42, 43, 44, 45, 46, 47]


def test_a_renumbered_window_is_fitted_under_a_different_seed() -> None:
    """The consequence, stated against behaviour rather than arithmetic: the same window
    carrying a different number draws a different random stream, so a migration that
    renumbers folds and keeps the stored predictions serves an artifact a fresh run would
    not reproduce."""
    first = np.random.default_rng(fold_seed(42, 0)).normal(size=8)
    renumbered = np.random.default_rng(fold_seed(42, 1)).normal(size=8)

    assert not np.array_equal(first, renumbered)


def test_a_numpy_fold_index_seeds_exactly_as_a_python_one() -> None:
    """`split["fold"]` arrives as a numpy integer on some paths and a Python int on
    others. Both must land on the same stream, or the derivation would depend on which
    frame library produced the split - and routing the six sites through one definition
    would have moved seeds rather than left them alone."""
    assert fold_seed(42, np.int64(3)) == fold_seed(42, 3) == 45
    assert np.array_equal(
        np.random.default_rng(fold_seed(42, np.int64(3))).normal(size=8),
        np.random.default_rng(45).normal(size=8),
    )


@pytest.mark.parametrize("module", SEEDING_MODULES, ids=lambda m: m.__name__)
def test_every_module_that_seeds_a_fold_derives_it_through_one_definition(module) -> None:
    """The six sites resolve the same object, so the coupling has one address.

    This pins that the definition is shared and single - the drift it catches is a module
    re-inlining `seed + fold` of its own, which is the state this issue found. It does not
    prove a module calls it at its seeding line; the end-to-end test below does that for
    one of the four ungated sites, `_subsample_index` covers the gated one directly, and
    the comment at each site carries the rest.
    """
    assert module.fold_seed is fold_seed


def test_a_reduced_run_resamples_its_training_rows_when_its_fold_is_renumbered() -> None:
    """The gated sites, `gbm.py` and `_subsample_index`, are one mechanism reached through
    this helper. No registered run is reduced today - all 739 that record
    `train_sample_frac` record 1.0 - so this constrains the next case study to turn a
    reduction on rather than anything already stored."""
    kept = _subsample_index(100, fold_id=0, train_sample_frac=0.5, seed=42)
    renumbered = _subsample_index(100, fold_id=1, train_sample_frac=0.5, seed=42)

    assert kept.shape == renumbered.shape == (50,)
    assert not np.array_equal(kept, renumbered)


def test_a_renumbered_fold_reseeds_the_fit_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole path for one ungated site: the fold id in the split reaches the process
    RNG, so the same two windows numbered differently are fitted under different seeds.

    `latent_factors` is the site driven here because its runner takes its splits as an
    argument. `tabular_dl`, `deep_learning` and `darts_forecasting` seed identically at the
    top of their own fold loops.

    The run opens with one seeding of the base at `cv.py:558`, before any fold exists, and
    the per-fold seedings follow it - so the assertion is on the tail. That leading call is
    the thing a renumbering does NOT touch, and it is exactly why a spec records one seed
    per run and no per-fold one: the fold's contribution is never written down.
    """
    from utils.modeling import RANDOM_SEED

    dates = pl.date_range(datetime(2020, 1, 1), datetime(2020, 1, 20), "1d", eager=True)
    dataset = pl.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": f"S{symbol}",
                "value": float(symbol + date_index / 100),
                "return": float(symbol + date_index / 100),
            }
            for date_index, timestamp in enumerate(dates)
            for symbol in range(6)
        ]
    )

    def fake_cae(chars_train, returns_train, chars_val, returns_val, **_kwargs):
        del chars_train, returns_train, returns_val
        physical = chars_val[..., 0]
        return (
            {0: physical - 1.0, 5: physical},
            {"checkpoint_epochs": [0, 5], "converged": True},
        )

    monkeypatch.setitem(latent_cv._MODEL_RUNNERS, "cae", fake_cae)

    def _seeds_for(fold_ids: list[int]) -> list[int]:
        seen: list[int] = []
        monkeypatch.setattr(latent_cv, "seed_everything", seen.append)
        latent_cv.run_latent_factor_cv(
            panel_data=None,
            splits=[
                {
                    "fold": fold_id,
                    "train_start": dates[0],
                    "train_end": dates[14],
                    "val_start": dates[15],
                    "val_end": dates[19],
                }
                for fold_id in fold_ids
            ],
            models=["cae"],
            n_factors=1,
            n_epochs=5,
            model_kwargs={"cae": {"checkpoint_interval": 5}},
            save_dir=None,
            dataset=dataset,
            feature_names=["value"],
            label_col="return",
            device="cpu",
            num_threads=1,
            checkpoint_surface="fitted_state",
            use_cache=False,
        )
        return seen

    two_folds = _seeds_for([0, 1])
    renumbered = _seeds_for([3, 4])

    assert two_folds[0] == RANDOM_SEED, "the run-level seeding comes first and never moves"
    assert two_folds[-2:] == [RANDOM_SEED, RANDOM_SEED + 1]
    assert renumbered[-2:] == [RANDOM_SEED + 3, RANDOM_SEED + 4]


def test_the_numbering_the_seeds_read_is_the_one_the_generator_emits() -> None:
    """The other half of the coupling. `_assert_chronological` refuses a fold set that is
    not numbered oldest first, and `tests/test_cv_splits.py` pins that the generator emits
    `0..n-1` in that order. Here that refusal is stated as what it costs: the id it
    protects is the integer the seed is derived from, so a numbering change is a refit of
    `tabular_dl`, `latent_factors`, `deep_learning`, `darts_forecasting` and every reduced
    run, not a relabelling of stored ones.
    """
    from utils.cv_splits import _assert_chronological

    oldest_first = [
        {"fold": index, "val_start": datetime(2020, 1, 1 + index), "val_end": datetime(2020, 2, 1)}
        for index in range(3)
    ]
    _assert_chronological(oldest_first)
    assert [fold_seed(42, split["fold"]) for split in oldest_first] == [42, 43, 44]

    newest_first = [
        {"fold": index, "val_start": datetime(2020, 1, 3 - index), "val_end": datetime(2020, 2, 1)}
        for index in range(3)
    ]
    with pytest.raises(RuntimeError, match="oldest first"):
        _assert_chronological(newest_first)

    reversed_ids = [{**split, "fold": 2 - split["fold"]} for split in oldest_first]
    with pytest.raises(RuntimeError, match="fold ids"):
        _assert_chronological(reversed_ids)
