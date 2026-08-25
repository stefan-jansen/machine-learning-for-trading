"""The holdout is evaluated once, and a notebook re-run must not be able to spend it again.

``evaluate_holdout`` is the sequence every case study needs and none of them had: read the
rank-1 validation backtest out of the candidate set, derive the holdout interval, lock, and
execute. The property these tests exist for is the one a caller cannot be trusted to check
for itself - re-running the notebook that publishes the page must read the recorded evaluation
back rather than producing a second one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from case_studies.research.holdout import evaluate_holdout
from tests.test_locked_holdout_execution import _install_fixture_adapter, _locked_study


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

    outcome = evaluate_holdout(study, candidate_set_name="locked-selection")

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

    first = evaluate_holdout(study, candidate_set_name="locked-selection")
    fits_after_first = list(fit_calls)

    second = evaluate_holdout(study, candidate_set_name="locked-selection")

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

    outcome = evaluate_holdout(study, candidate_set_name="locked-selection")

    evidence = outcome.lock.record["selection_evidence"]
    assert evidence["metric"] == "validation_backtest_sharpe"


def test_an_unknown_candidate_set_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices)
    _pin_derivation_to_the_fixture(monkeypatch, lock)

    with pytest.raises(ValueError, match="resolved to 0 identities"):
        evaluate_holdout(study, candidate_set_name="no-such-set")
