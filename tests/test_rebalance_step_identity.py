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
