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


def test_a_family_without_a_rekey_hook_refuses_to_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock cannot be revised, so producing one known to fail at execution is worse than none.

    ``expected_prediction_keys`` describes which rows the fit was eligible to predict on, and
    ``validate_locked_expected_keys`` refuses a locked spec whose manifest describes a different
    frame. A holdout spec inheriting the validation folds' manifest therefore fails at execution,
    and one with the field stripped fails there too - so the fields must be recomputed against the
    holdout fold, by the rule that produced them, which is family-specific.

    A family with no hook still refuses. What changed is that it refuses for itself and says what
    implementing it means, rather than refusing on behalf of all five families at once.
    """
    from case_studies.research import holdout as module

    monkeypatch.setattr("case_studies.research.models._family_module", lambda _family: object())
    spec = {"family": "linear", "computation": {"cv": {"folds": [{"fold": 9}]}}}
    with pytest.raises(NotImplementedError, match="cannot yet re-key"):
        module._rekey_holdout_spec(None, spec, {"computation": {}})


def _holdout_shaped_spec(*, manifest_folds: int, param_folds: int) -> dict:
    return {
        "family": "linear",
        "computation": {
            "cv": {
                "folds": [
                    {
                        "fold": 8,
                        "train_start": "a",
                        "train_end": "b",
                        "val_start": "c",
                        "val_end": "d",
                    }
                ]
            },
            "expected_prediction_keys": {"digest": "abc", "n_rows": 1, "n_folds": manifest_folds},
            "model": {
                "effective_params_by_fold": {str(i): {"alpha": 0.1} for i in range(param_folds)}
                if param_folds != 1
                else {"8": {"alpha": 0.1}}
            },
        },
    }


def test_a_hook_that_leaves_the_validation_folds_in_place_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-keyed shape is checked, not trusted.

    A hook that returns without recomputing, or that recomputes against the wrong split, leaves
    fields that look present and are wrong - and the lock is the one artifact in the pipeline that
    cannot be revised. Both fields are checked, because a hook that re-keys the manifest and
    forgets the parameters produces a lock that reconstructs a different model than the one
    selection ranked, and the manifest check alone would pass it.
    """
    from case_studies.research import holdout as module

    class _NoOpFamily:
        @staticmethod
        def rekey_holdout_spec(_study, _spec, *, validation_spec):  # noqa: ARG004
            return None

    monkeypatch.setattr("case_studies.research.models._family_module", lambda _family: _NoOpFamily)

    with pytest.raises(ValueError, match="was not re-keyed to one fold"):
        module._rekey_holdout_spec(None, _holdout_shaped_spec(manifest_folds=8, param_folds=8), {})

    with pytest.raises(ValueError, match="still describe the validation folds"):
        module._rekey_holdout_spec(None, _holdout_shaped_spec(manifest_folds=1, param_folds=8), {})

    module._rekey_holdout_spec(None, _holdout_shaped_spec(manifest_folds=1, param_folds=1), {})
