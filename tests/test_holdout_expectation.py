"""What `Strategy` requires before it will execute a holdout backtest.

A holdout prediction says nothing about which configuration was selected. Before this contract
existed, that fact was supplied by a `research_locks` row in the `LOCKED` state, which made an
authorization token a prerequisite for a computation - and made a holdout found to be wrong after
a bug fix impossible to correct, because the lock is by construction the artifact that cannot be
revised.

`HoldoutExpectation` carries the same facts without the token: the caller states what it expects
and `Strategy` compares. The test that matters most here is
`test_a_holdout_without_an_expectation_is_refused`: if a missing expectation is anything other
than a refusal, the guarantee is gone and every other test in this file still passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from case_studies.research.strategy import HoldoutExpectation, Strategy

SELECTED_STRATEGY: dict[str, Any] = {"strategy": {"signal": {"name": "rank"}}, "version": 2}


@dataclass
class _Prediction:
    """The three things `Strategy.__post_init__` reads off a prediction."""

    study: object
    record: dict[str, Any]
    complete: bool = True
    execution_tier: str = "canonical"

    def registry_record(self) -> dict[str, Any]:
        return self.record


def _holdout_prediction(
    study: object,
    *,
    training_hash: str = "b02411b28bc5",
    checkpoint_kind: str = "final",
    checkpoint_value: int | None = None,
) -> _Prediction:
    return _Prediction(
        study=study,
        record={
            "split": "holdout",
            "training_hash": training_hash,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_value": checkpoint_value,
        },
    )


def _expectation(**overrides: Any) -> HoldoutExpectation:
    fields: dict[str, Any] = {
        "training_hash": "b02411b28bc5",
        "checkpoint_kind": "final",
        "checkpoint_value": None,
        "strategy": SELECTED_STRATEGY,
        "validation_prediction_hash": "136bccd19c46",
    }
    fields.update(overrides)
    return HoldoutExpectation(**fields)


def _strategy(prediction: _Prediction, expectation: HoldoutExpectation | None) -> Strategy:
    study = prediction.study
    return Strategy(
        study=study,  # type: ignore[arg-type]
        prediction=prediction,  # type: ignore[arg-type]
        signal={"name": "rank"},
        holdout_expectation=expectation,
    )


def test_a_holdout_without_an_expectation_is_refused() -> None:
    """The test that decides whether this is a contract or a formality.

    A permissive fallback here - accepting any canonical holdout prediction when the caller
    states nothing - would let a holdout produced from a configuration nobody selected be
    executed and reported, which is exactly what the lock was standing in front of. Every other
    test in this file passes under that fallback, so this one carries the guarantee alone.
    """
    study = object()
    with pytest.raises(ValueError, match="requires a holdout_expectation"):
        _strategy(_holdout_prediction(study), None)


def test_a_holdout_whose_training_differs_from_the_expectation_is_refused() -> None:
    study = object()
    prediction = _holdout_prediction(study, training_hash="3cffc6db6c25")
    with pytest.raises(ValueError, match="but the caller expects training b02411b28bc5"):
        _strategy(prediction, _expectation())


def test_a_holdout_at_the_wrong_checkpoint_is_refused() -> None:
    """One training run registers one prediction set per declared checkpoint.

    They share a training hash and a strategy specification, so the checkpoint is the only thing
    that tells them apart, and a caller expecting one must not silently execute another.
    """
    study = object()
    prediction = _holdout_prediction(study, checkpoint_kind="epoch", checkpoint_value=40)
    with pytest.raises(ValueError, match="holdout prediction is training"):
        _strategy(prediction, _expectation())


def test_a_matching_expectation_constructs() -> None:
    study = object()
    strategy = _strategy(_holdout_prediction(study), _expectation())
    assert strategy.holdout_expectation is not None
    assert strategy.holdout_expectation.training_hash == "b02411b28bc5"


def test_a_validation_prediction_needs_no_expectation() -> None:
    """The contract is about holdout execution only.

    Validation runs are ranked against each other and re-run freely, so requiring an expectation
    there would be ceremony on the path that does not need it - and would break every existing
    caller for no guarantee.
    """
    study = object()
    prediction = _Prediction(
        study=study,
        record={
            "split": "validation",
            "training_hash": "72fd9f5cf07e",
            "checkpoint_kind": "final",
            "checkpoint_value": None,
        },
    )
    assert _strategy(prediction, None).holdout_expectation is None


def test_a_preview_holdout_prediction_is_refused_before_the_expectation_is_read() -> None:
    """Tier is checked first: a preview holdout is not a holdout whatever the caller expects."""
    study = object()
    prediction = _holdout_prediction(study)
    prediction.execution_tier = "preview"
    with pytest.raises(ValueError, match="validation or canonical holdout predictions"):
        _strategy(prediction, _expectation())
