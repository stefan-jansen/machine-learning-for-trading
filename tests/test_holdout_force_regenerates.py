"""`generate_holdout(force=True)` deletes the existing holdout, stale or not.

The delete used to be gated on `has_holdout_predictions`, which asks whether a holdout is
tied to one of the *current* validation top-N. Its own docstring says it returns False
when a holdout exists but no top-N candidate matches it - a holdout whose training fell
out of the top-N after a sweep reshuffle. So the delete was skipped exactly when the
existing holdout was most stale, the new one was written beside it, and the case study
carried two holdouts against the one-holdout-per-case-study rule.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "strategy_synthesis_holdout_force",
    Path(__file__).resolve().parents[1] / "20_strategy_synthesis" / "holdout.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_HOLDOUT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HOLDOUT
_SPEC.loader.exec_module(_HOLDOUT)


class _ReachedSelection(Exception):
    """Raised in place of the retrain, which is what this test is not exercising."""


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def _delete(cs_id: str) -> int:
        seen.append("delete")
        return 1

    def _select(*args, **kwargs):
        raise _ReachedSelection

    monkeypatch.setattr(_HOLDOUT, "delete_holdout_predictions", _delete)
    monkeypatch.setattr(_HOLDOUT, "select_best_models", _select)
    monkeypatch.setattr(_HOLDOUT, "load_existing_holdout", lambda cs_id: seen.append("load") or {})
    return seen


def test_force_deletes_a_holdout_no_top_n_candidate_matches(calls, monkeypatch) -> None:
    """The stale case: `has_holdout_predictions` is False and a holdout is still on disk."""
    monkeypatch.setattr(_HOLDOUT, "has_holdout_predictions", lambda cs_id: False)

    with pytest.raises(_ReachedSelection):
        _HOLDOUT.generate_holdout("nasdaq100_microstructure", force=True, verbose=False)

    assert calls == ["delete"]


def test_force_deletes_a_holdout_the_top_n_still_matches(calls, monkeypatch) -> None:
    monkeypatch.setattr(_HOLDOUT, "has_holdout_predictions", lambda cs_id: True)

    with pytest.raises(_ReachedSelection):
        _HOLDOUT.generate_holdout("nasdaq100_microstructure", force=True, verbose=False)

    assert calls == ["delete"]


def test_without_force_a_matching_holdout_is_still_loaded(calls, monkeypatch) -> None:
    monkeypatch.setattr(_HOLDOUT, "has_holdout_predictions", lambda cs_id: True)

    _HOLDOUT.generate_holdout("nasdaq100_microstructure", force=False, verbose=False)

    assert calls == ["load"]
