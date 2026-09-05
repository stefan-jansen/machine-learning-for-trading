"""A stage whose inputs exist only in this workspace declares that, and skips rather than fails.

`tests/test_case_studies.py` says each notebook "runs independently against pre-generated
intermediates". For the preview downstream stages that is not true. They select their inputs
by execution tier, `backtest_runs` has no tier of its own - a backtest inherits the tier of
the training run behind its prediction set - and every seeded row is `canonical`. So a
freshly seeded registry holds no preview rows and those stages resolve only what an earlier
notebook registered in the same workspace.

Seeding the preview root is the obvious fix and it was tried and reverted: `tests/conftest.py`
records that the seeded rows land in the same `(split, family, label)` coverage group as the
predictions a preview run actually fits, cannot win it - 60 dates against 447-483 - and so
`full_coverage_prediction_sql` admits neither and `14_backtest` refuses to rank anything.

So the dependency is declared instead. A full in-order run satisfies it and is unaffected;
a focused run, or one distributed across xdist workers with separate workspaces, skips with
the prerequisite named rather than failing several cells in on a refusal that reads like a
broken pipeline.

Tracked as ml4t/agent-workspace#1036.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.pm_helpers import get_overrides  # noqa: E402
from tests.test_case_studies import _required_stages  # noqa: E402


def test_no_declaration_means_no_prerequisite() -> None:
    assert _required_stages({}) == []
    assert _required_stages({"requires_stage": None}) == []


def test_one_name_or_a_list_both_work() -> None:
    """A stage reading two upstream tiers should not need a second key."""
    assert _required_stages({"requires_stage": "13_backtest"}) == ["13_backtest"]
    assert _required_stages({"requires_stage": ["06_linear", "13_backtest"]}) == [
        "06_linear",
        "13_backtest",
    ]


def test_the_declared_preview_chain_is_ordered() -> None:
    """Each declaration names a stage that sorts before it.

    A prerequisite that sorts later can never be satisfied: the parametrization runs in
    numeric order, so the stage tracker would not hold it yet and the notebook would skip on
    every run including the full one - a stage silently never exercised.
    """
    declared = {
        "case_studies/crypto_perps_funding/13_backtest": "06_linear",
        "case_studies/crypto_perps_funding/14_portfolio_management": "13_backtest",
    }
    for key, expected in declared.items():
        overrides = get_overrides(key)
        required = _required_stages(overrides)
        assert required == [expected], f"{key} declares {required}, expected [{expected}]"
        stage = key.rsplit("/", 1)[1]
        for prerequisite in required:
            assert prerequisite < stage, (
                f"{key} requires {prerequisite}, which sorts after it, so the tracker can "
                "never hold it and the stage would skip on every run"
            )
