"""The holdout is evaluated once, and a notebook re-run must not be able to spend it again.

``evaluate_holdout`` is the sequence every case study needs and none of them had: read the
rank-1 validation backtest out of the candidate set, derive the holdout interval, lock, and
execute. The property these tests exist for is the one a caller cannot be trusted to check
for itself - re-running the notebook that publishes the page must read the recorded evaluation
back rather than producing a second one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from case_studies.research.holdout import evaluate_holdout
from case_studies.research.models import prepare_locked_holdout_spec
from tests.test_locked_holdout_execution import _install_fixture_adapter, _locked_study

# The fixture study's own observation grid. build_holdout_cv is pinned below, so the timeline is
# only carried through the driver here; its own derivation is covered separately.
TIMELINE = pd.bdate_range("2023-12-01", "2024-02-01")


def _pin_derivation_to_the_fixture(monkeypatch: pytest.MonkeyPatch, lock) -> None:
    """Supply the interval the fixture study was locked under.

    The derivation itself is covered against real case-study configuration in
    tests/test_holdout_cv_derivation.py. What is under test here is the driver: the selection it
    reads, the lock it creates, and what it refuses to do twice. Pinning the interval keeps those
    two concerns from being able to mask each other.
    """
    from case_studies.research import holdout as module

    locked_cv = lock.record["holdout_training_spec"]["computation"]["cv"]
    monkeypatch.setattr(module, "build_holdout_cv", lambda *a, **k: locked_cv)


def test_it_fits_the_locked_lineage_and_reports_that_it_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    fit_calls = _install_fixture_adapter(monkeypatch, prices)
    _pin_derivation_to_the_fixture(monkeypatch, lock)

    outcome = evaluate_holdout(study, candidate_set_name="locked-selection", timeline=TIMELINE)

    assert outcome.evaluated_now is True
    assert outcome.lock.state == "HOLDOUT_EVALUATED"
    assert outcome.lineage["lock_hash"] == lock.hash
    assert len(fit_calls) == 1


def test_a_second_call_reads_the_recorded_evaluation_and_fits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety property. A re-run of the notebook must not spend the holdout twice.

    ``run_locked_holdout`` raises on a second call, which protects the registry but would make
    every notebook re-run fail. The driver has to absorb that into a read, so the page rebuilds
    with the numbers it published before.
    """
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    fit_calls = _install_fixture_adapter(monkeypatch, prices)
    _pin_derivation_to_the_fixture(monkeypatch, lock)

    first = evaluate_holdout(study, candidate_set_name="locked-selection", timeline=TIMELINE)
    fits_after_first = list(fit_calls)

    second = evaluate_holdout(study, candidate_set_name="locked-selection", timeline=TIMELINE)

    assert first.evaluated_now is True
    assert second.evaluated_now is False
    assert fit_calls == fits_after_first, "the second call refitted the holdout model"
    assert second.lineage == first.lineage
    assert second.lock.state == "HOLDOUT_EVALUATED"


def test_the_recorded_selection_is_the_documented_rule_and_names_the_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices)
    _pin_derivation_to_the_fixture(monkeypatch, lock)

    outcome = evaluate_holdout(study, candidate_set_name="locked-selection", timeline=TIMELINE)

    evidence = outcome.lock.record["selection_evidence"]
    assert evidence["metric"] == "validation_backtest_sharpe"


def test_an_unknown_candidate_set_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices)
    _pin_derivation_to_the_fixture(monkeypatch, lock)

    with pytest.raises(ValueError, match="resolved to 0 identities"):
        evaluate_holdout(study, candidate_set_name="no-such-set", timeline=TIMELINE)


def test_a_spec_carrying_fold_derivations_requires_an_adapter_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock cannot retain a validation eligibility manifest."""
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices)
    _pin_derivation_to_the_fixture(monkeypatch, lock)

    from case_studies.research import models

    monkeypatch.setattr(
        models,
        "get_adapter",
        lambda kind, name: object(),
    )

    spec = {"family": "linear", "computation": {"expected_prediction_keys": {}}}
    with pytest.raises(NotImplementedError, match="cannot re-key"):
        prepare_locked_holdout_spec(
            study,
            spec,
            checkpoint_kind="final",
            checkpoint_value=None,
        )

    clean = {"family": "linear", "computation": {"cv": {}}}
    assert (
        prepare_locked_holdout_spec(
            study,
            clean,
            checkpoint_kind="final",
            checkpoint_value=None,
        )
        == clean
    )


def test_adapter_may_change_only_fold_derived_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, _, _ = _locked_study(tmp_path, monkeypatch)
    from case_studies.research import models

    def prepare(_study, spec, **_checkpoint):
        spec["computation"]["expected_prediction_keys"] = {
            "digest": "holdout",
            "n_rows": 2,
            "n_folds": 1,
        }
        return spec

    monkeypatch.setattr(
        models,
        "get_adapter",
        lambda kind, name: type("Adapter", (), {"prepare_locked_holdout_spec": prepare}),
    )
    spec = {
        "family": "linear",
        "computation": {
            "cv": {"split": "holdout"},
            "expected_prediction_keys": {"digest": "validation", "n_rows": 4, "n_folds": 2},
        },
    }
    prepared = prepare_locked_holdout_spec(
        study,
        spec,
        checkpoint_kind="final",
        checkpoint_value=None,
    )
    assert prepared["computation"]["expected_prediction_keys"]["digest"] == "holdout"

    def remove_manifest(_study, value, **_checkpoint):
        value["computation"].pop("expected_prediction_keys")
        return value

    monkeypatch.setattr(
        models,
        "get_adapter",
        lambda kind, name: type("Adapter", (), {"prepare_locked_holdout_spec": remove_manifest}),
    )
    with pytest.raises(ValueError, match="removed required holdout fold derivations"):
        prepare_locked_holdout_spec(
            study,
            spec,
            checkpoint_kind="final",
            checkpoint_value=None,
        )

    def change_model(_study, value, **_checkpoint):
        value["config_name"] = "different"
        return value

    monkeypatch.setattr(
        models,
        "get_adapter",
        lambda kind, name: type("Adapter", (), {"prepare_locked_holdout_spec": change_model}),
    )
    with pytest.raises(ValueError, match="outside the holdout fold derivations"):
        prepare_locked_holdout_spec(
            study,
            spec,
            checkpoint_kind="final",
            checkpoint_value=None,
        )
