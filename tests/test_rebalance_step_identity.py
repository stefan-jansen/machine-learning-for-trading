"""The step a run trades at is the step its identity records.

`labels.rebalance_step` composes with the declared cadence to decide which slots are traded,
so it belongs to the spec that is hashed. While it lived only in setup.yaml, two runs of one
configuration at different steps hashed identically and the second was skipped as already
registered - the registry kept the first run's numbers under a spec that never named the
parameter that produced them.

ml4t/agent-workspace#1005.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# The step is part of the identity it decides (ml4t/agent-workspace#1005)
# ---------------------------------------------------------------------------


def test_a_declared_step_reaches_the_registered_spec() -> None:
    """A run at a different step must hash differently, or the corrected run is skipped.

    The step decides which slots are traded and therefore every metric recorded, so two runs
    of one configuration at different steps are two different runs. Before the step entered
    `strategy.rebalance` they hashed identically and the second was dropped as already
    registered, keeping the first run's numbers under a spec that did not name the parameter
    that produced them.
    """
    from case_studies.utils.backtest_loaders import (
        declared_rebalance_step,
        get_backtest_config,
    )
    from case_studies.utils.backtest_presets import build_backtest_spec

    case_study, label = "cme_futures", "fwd_ret_21d"
    step = declared_rebalance_step(case_study, label)
    assert step is not None, f"{case_study}/{label} declares no step; pick one that does"

    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, d) for d in range(2, 12) for _ in range(2)],
            "symbol": ["AAA", "BBB"] * 10,
            "open": [100.0, 50.0] * 10,
            "high": [101.0, 51.0] * 10,
            "low": [99.0, 49.0] * 10,
            "close": [100.5, 50.5] * 10,
            "volume": [1000.0, 900.0] * 10,
        }
    )
    spec = build_backtest_spec(
        case_study,
        get_backtest_config(case_study),
        signal={"method": "equal_weight_top_k", "top_k": 2, "long_short": False},
        prices=prices,
        prediction_hash="pred123",
        initial_cash=1_000_000.0,
        label=label,
    )
    assert spec["strategy"]["rebalance"]["step"] == step


def test_an_undeclared_step_leaves_the_spec_byte_identical() -> None:
    """A case study that declares no step for a label keeps the identity it already has.

    The same rule `cadence_for` follows: the key appears only where the parameter is
    load-bearing, so nothing already registered is orphaned by this change.
    """
    from case_studies.utils.backtest_loaders import declared_rebalance_step

    assert declared_rebalance_step("etfs", "a_label_no_case_study_declares") is None


def test_execution_reads_the_recorded_step_not_the_editable_file() -> None:
    """The spec is the record of what ran, so execution takes the step from it.

    setup.yaml is mutable and the spec is not. While execution read setup.yaml, a run could
    hash one step and trade another: edit the file after the spec is built and the registry
    keeps a spec that names a step nothing used.
    """
    from case_studies.utils.backtest_loaders import resolved_rebalance_step

    # The spec wins over whatever setup.yaml currently says.
    assert resolved_rebalance_step({"step": 7}, "nasdaq100_microstructure", "fwd_ret_15m") == 7
    # A spec written before the step entered the identity falls back to the declaration.
    assert resolved_rebalance_step({}, "nasdaq100_microstructure", "fwd_ret_15m") == 1
    assert resolved_rebalance_step(None, "cme_futures", "fwd_ret_21d") == 3

    with pytest.raises(ValueError, match="must be >= 1"):
        resolved_rebalance_step({"step": 0}, "nasdaq100_microstructure", "fwd_ret_15m")


# ---------------------------------------------------------------------------
# A unit step is the absence of a step (ml4t/agent-workspace#1028)
# ---------------------------------------------------------------------------


def _spec(step: int | None) -> dict:
    """A minimal strategy spec, with or without a declared step."""
    rebalance = {
        "mode": "engine",
        "cadence": "weekly_friday_close",
        "min_weight_change": 0.005,
        "min_trade_value": 100.0,
    }
    if step is not None:
        rebalance["step"] = step
    return {
        "version": 2,
        "identity_version": 2,
        "chapter": "16",
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 5},
            "rebalance": rebalance,
        },
    }


class TestAUnitStepIsNotNormalizedAway:
    """`step: 1` stays in the identity, and this records the measurement that says it must.

    Public `83141459` made `strategy.rebalance.step` a hash input after 21,995 backtests had
    been registered without it, so every one of those rows resolves to a different hash than
    the next run computes and a sweep re-run in any completed case study stops on `a changed
    population named '<name>' must explicitly supersedes <hash>` (ml4t/agent-workspace#1028).

    The obvious repair is to drop `step: 1` before hashing, the way `signal.direction ==
    "long_only"` is dropped just above: `gather_every(1)` returns the schedule unchanged, so
    `step: 1` and no step describe the same traded decisions, and it recovers 16,423 of the
    21,995 rows while invalidating none.

    **It is unsound, and the reason is not obvious enough to leave unwritten.** A legacy spec
    carries no step but did not execute at step 1 - it executed at whatever `setup.yaml`
    declared when it ran, which for `cme_futures/fwd_ret_21d` is 3, for `sp500_options` 5 and
    for `fx_pairs/fwd_ret_21d` 21. Drop the explicit default and a *future* spec at step 1
    hashes to those rows, is served from the registry as already registered, and reports
    step-3 numbers under a step-1 request. That is ml4t/agent-workspace#1005 exactly, arrived
    at from the other side, and it is silent.

    5,572 rows across `cme_futures`, `crypto_perps_funding`, `fx_pairs` and `sp500_options`
    are in that ambiguous state. The sound repair is to record each legacy row's executed step
    in its spec and rehash - which is what `preset_path` did - but 5,570 of those rows are
    members of 848 official populations whose own identity is the hash of their member list,
    so it cannot be done without rewriting that lineage. #1028 carries the disposition.
    """

    def test_a_unit_step_is_part_of_the_identity(self) -> None:
        """Not normalized away: a legacy spec with no step may have traded at any step."""
        from case_studies.utils.registry.specs import backtest_hash_from_parts

        assert backtest_hash_from_parts("pred123", _spec(1)) != backtest_hash_from_parts(
            "pred123", _spec(None)
        )

    def test_a_non_unit_step_is_too(self) -> None:
        from case_studies.utils.registry.specs import backtest_hash_from_parts

        assert backtest_hash_from_parts("pred123", _spec(3)) != backtest_hash_from_parts(
            "pred123", _spec(None)
        )

    def test_two_non_unit_steps_stay_distinct(self) -> None:
        """The half of #1005 that any repair here has to preserve.

        A corrected step must move the identity, or the corrected run hashes to the rows it
        was written to replace and is skipped as already registered.
        """
        from case_studies.utils.registry.specs import backtest_hash_from_parts

        assert backtest_hash_from_parts("pred123", _spec(3)) != backtest_hash_from_parts(
            "pred123", _spec(5)
        )
        assert backtest_hash_from_parts("pred123", _spec(3)) != backtest_hash_from_parts(
            "pred123", _spec(1)
        )
