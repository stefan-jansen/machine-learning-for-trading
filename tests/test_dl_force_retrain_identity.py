"""Forced retraining must clear the training hash it re-registers under.

``run_dl_cv`` builds each config's training spec once, folding in the caller's
``identity_params``, and registers results under that identity. Its
``force_retrain`` branch used to rebuild the spec on its own and omit
``extra_params``, so it asked ``clear_prediction_sets`` to remove a hash nothing
was ever registered under.

Nothing downstream distinguishes "cleared nothing" from "there was nothing to
clear": ``clear_prediction_sets`` returns a count, and a zero is unremarkable.
A run that invalidates none of its stale predictions and then registers
alongside them therefore looks exactly like a clean retrain.

This ran as a strict ``xfail`` while the fix waited on an identity batch to pay
for the refit. Putting the training device into the run identity is that batch,
so the branch now reuses the spec it already built and this asserts it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import case_studies.utils.registry as registry
from case_studies.utils import deep_learning

IDENTITY_PARAMS = {"feature_names": ["feature", "temporal_feature"]}


def _hash(spec: dict) -> str:
    return str(sorted(spec.items()))


def test_force_retrain_clears_the_hash_it_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict] = []
    cleared: list[str] = []

    def build_spec(family, config_name, label, **kwargs):
        spec = {
            "family": family,
            "config_name": config_name,
            "label": label,
            "n_folds": kwargs.get("n_folds"),
            "n_epochs": kwargs.get("n_epochs"),
            "extra_params": str(kwargs.get("extra_params")),
        }
        built.append(spec)
        return spec

    def clear(_case_study, training_hash, *, split):
        cleared.append(training_hash)
        return {"prediction_sets": 0}

    monkeypatch.setattr(registry, "build_training_spec", build_spec)
    monkeypatch.setattr(registry, "training_hash_from_spec", _hash)
    monkeypatch.setattr(registry, "clear_prediction_sets", clear)

    config = {
        "family": "deep_learning",
        "config_name": "model",
        "n_epochs": 2,
        "params": {"architecture": "lstm", "lookback": 2},
    }
    # Training cannot run on an empty panel; it raises while resolving the fixture case study,
    # which is downstream of the clear. The assertions below check the run actually got that far
    # rather than trusting that an exception means the branch executed.
    with pytest.raises(KeyError):
        deep_learning.run_dl_cv(
            pd.DataFrame(),
            [{"fold": 0}],
            configs=[config],
            n_features=len(IDENTITY_PARAMS["feature_names"]),
            feature_names=list(IDENTITY_PARAMS["feature_names"]),
            label_col="target",
            date_col="timestamp",
            entity_col="symbol",
            device="cpu",
            register=True,
            force_retrain=True,
            case_study="example",
            save_dir=tmp_path,
            identity_params=dict(IDENTITY_PARAMS),
        )

    identity_qualified = [_hash(spec) for spec in built if spec["extra_params"] != "None"]
    assert identity_qualified, (
        "the run never built a spec carrying identity_params, so it stopped before the code "
        f"under test; specs built: {built}"
    )
    assert cleared, (
        "force_retrain never called clear_prediction_sets, so it stopped before the code under "
        f"test; specs built: {built}"
    )
    assert cleared[0] in identity_qualified, (
        "forced retraining cleared a hash the run does not register under, so the stale "
        f"predictions survive it: cleared {cleared[0]!r}, registers under {identity_qualified!r}"
    )
