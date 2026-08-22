"""Guard the awaiting-rebuild skips so they cannot become permanent.

The risk with any skip mechanism is that it quietly becomes the state of the world: a third of
the suite skipping because nobody removed the declarations. These tests make that visible.

A declaration whose condition the registry already satisfies is a failure, not a warning. It
means the input exists, the notebook could run, and the skip is now hiding it.
"""

from __future__ import annotations

import pytest
import yaml

from tests.awaiting_rebuild import unmet_reason
from tests.pm_helpers import OVERRIDES_PATH


def _declarations() -> dict[str, dict]:
    overrides = yaml.safe_load(OVERRIDES_PATH.read_text()) or {}
    return {
        key: value["awaiting_rebuild"]
        for key, value in overrides.items()
        if isinstance(value, dict) and value.get("awaiting_rebuild")
    }


def test_every_declaration_is_well_formed():
    """`needs.of` is required and `needs` must name something checkable."""
    for key, declaration in _declarations().items():
        needs = declaration.get("needs")
        assert isinstance(needs, dict), f"{key}: awaiting_rebuild.needs must be a mapping"
        assert needs.get("of"), f"{key}: awaiting_rebuild.needs.of must name a case study"
        assert needs.keys() & {"family", "backtest_stage", "registry"}, (
            f"{key}: needs must name a family, a backtest_stage, or registry"
        )
        assert declaration.get("issue"), (
            f"{key}: awaiting_rebuild.issue must name the issue tracking the missing input, "
            "so an untracked skip cannot be added"
        )


@pytest.mark.parametrize("key", sorted(_declarations()))
def test_declaration_has_not_outlived_its_reason(key):
    """The declared input is still missing.

    When the rebuild produces it this fails, which is the signal to delete the declaration. The
    test the skip was covering then runs on its own, with no other edit.
    """
    reason = unmet_reason(_declarations()[key])
    assert reason is not None, (
        f"{key}: the input this skip waits for now exists in the registry. Delete the "
        f"awaiting_rebuild block from tests/overrides.yaml; the notebook's own test runs again."
    )


def test_skips_stay_a_minority():
    """A ceiling on how much of the suite may be waiting at once.

    Not a substitute for the per-declaration check above, which is what actually retires them.
    This catches the failure mode where declarations accumulate faster than the rebuild retires
    them, before it reaches a third of the suite.
    """
    overrides = yaml.safe_load(OVERRIDES_PATH.read_text()) or {}
    waiting = len(_declarations())
    assert waiting <= 40, (
        f"{waiting} notebooks are awaiting the rebuild out of {len(overrides)} declared. "
        "The rebuild is meant to retire these, not accumulate them."
    )
