"""Which pair shape each registered paired-metric row is computed under.

``compute_paired_uncertainty`` needs to be told whether a leading flat run on the challenger
is a warmup before its first signal or a position the challenger chose to hold, because the
two are indistinguishable in the returns and the trim differs.

**Every pair `paired_metrics` produces takes the default, independent-series shape**, including
the three stage transitions. That is not because a stage transition cannot be an overlay
relationship - it is because neither thing the overlay rule needs is established here.
``champion_lineage`` takes the best-Sharpe backtest at each stage independently, sharing only a
prediction hash, so adjacent entries are not demonstrably parent and child; and a challenger at
these stages can carry a genuine warmup, as ``conformal_weighted`` does by keeping only
timestamps with prior-only calibration. ``17_risk_management`` and ``18_strategy_analysis`` do
use the overlay shape, because there the overlay is paired with its own no-overlay carrier by
construction rather than by a highest-Sharpe query.

Getting this wrong is a wiring defect rather than a maths one: the numbers are computed
correctly under the wrong rule and the row registers without complaint. The first tests pin the
behaviour at ``_populate_pair``; the last drives ``populate_paired_metrics`` itself, because a
test that passes the flag in by hand cannot see what the producer chooses.

The producer hands the bootstrap an untrimmed pair and reads the sample size back out of it,
rather than trimming first and passing the result on. The default rule is not idempotent -
applied twice it slides the start forward again whenever the earlier starter is exactly zero on
the later starter's first traded session - so a producer that pre-trims silently shortens the
sample and then reports the longer figure it measured. Both are checked below.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from case_studies.utils import paired_metrics
from case_studies.utils.uncertainty import joint_returns


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record every call into the bootstrap, and keep the registry out of it."""
    calls: list[dict] = []
    real = paired_metrics.compute_paired_uncertainty

    def spy(challenger, baseline, **kwargs):
        calls.append(
            {
                "challenger": np.asarray(challenger, dtype=np.float64),
                "baseline": np.asarray(baseline, dtype=np.float64),
                "overlay": kwargs.get("challenger_overlays_baseline", False),
            }
        )
        return real(challenger, baseline, **kwargs)

    monkeypatch.setattr(paired_metrics, "compute_paired_uncertainty", spy)
    monkeypatch.setattr(paired_metrics, "register_paired_metrics", lambda *a, **k: None)
    return calls


def _returns(values: list[float]) -> pl.DataFrame:
    start = date(2020, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(len(values))],
            "ret": values,
        }
    )


def _pair() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(5)
    baseline = rng.normal(0.0005, 0.01, size=40)
    challenger = baseline + rng.normal(0.0, 0.002, size=40)
    challenger[:6] = 0.0
    return _returns(list(challenger)), _returns(list(baseline))


def test_a_benchmark_pair_drops_the_challengers_leading_flat_run(captured: list[dict]) -> None:
    """The default shape: an independent strategy is not charged for its warmup."""
    challenger, baseline = _pair()

    row = paired_metrics._populate_pair(
        "unit_cs",
        "chal",
        "bench",
        "equal_weight",
        challenger,
        baseline,
        252,
        "fwd_ret_5d",
    )

    assert len(captured) == 1
    assert captured[0]["overlay"] is False
    # The bootstrap is handed all 40 aligned sessions and trims once; the six the challenger
    # sat out before its first signal are gone from the sample it reports.
    assert captured[0]["challenger"].size == 40
    assert row["n_overlap"] == 34


def test_a_stage_transition_keeps_the_sessions_the_challenger_sat_out(
    captured: list[dict],
) -> None:
    """A challenger built on top of its baseline is live from the baseline's first session."""
    challenger, baseline = _pair()

    row = paired_metrics._populate_pair(
        "unit_cs",
        "chal",
        "bench",
        "cost_sensitivity_leader",
        challenger,
        baseline,
        252,
        "fwd_ret_5d",
        challenger_overlays_baseline=True,
    )

    assert len(captured) == 1
    assert captured[0]["overlay"] is True
    assert captured[0]["challenger"].size == 40
    assert row["n_overlap"] == 40


def test_the_two_shapes_do_not_agree_on_the_difference(captured: list[dict]) -> None:
    """If they agreed, neither the flag nor these tests would be worth having."""
    challenger, baseline = _pair()
    rows = [
        paired_metrics._populate_pair(
            "unit_cs",
            "chal",
            "bench",
            kind,
            challenger,
            baseline,
            252,
            "fwd_ret_5d",
            challenger_overlays_baseline=overlay,
        )
        for kind, overlay in (("equal_weight", False), ("cost_sensitivity_leader", True))
    ]

    assert all("skip" not in row for row in rows)
    assert rows[0]["sharpe_diff"] != rows[1]["sharpe_diff"]


def test_the_producer_gives_every_stage_transition_the_default_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What `populate_paired_metrics` chooses, not what a caller passes it by hand.

    The three stage transitions carried `challenger_overlays_baseline=True` for one commit, on a
    premise `champion_lineage` does not support. Reverting it changed no test, because every
    test at the time handed the flag to `_populate_pair` directly - so the wiring the revert was
    about had no coverage in either direction. This drives the producer and reads the flag off
    the call.
    """
    seen: list[tuple[str, bool]] = []

    def spy(*args, **kwargs):
        seen.append((args[3], kwargs.get("challenger_overlays_baseline", False)))
        return {"kind": args[3]}

    monkeypatch.setattr(paired_metrics, "_populate_pair", spy)
    monkeypatch.setattr(paired_metrics, "_aligned_returns", lambda cs, h: _returns([0.001] * 60))
    monkeypatch.setattr(paired_metrics, "_val_rank1_full_spec", lambda *a, **k: None)
    monkeypatch.setattr(paired_metrics, "_holdout_lineage_for", lambda *a, **k: None)
    monkeypatch.setattr(paired_metrics, "_benchmark_returns_from_artifact", lambda *a, **k: None)

    class _Explorer:
        def best(self, **kwargs):
            return pl.DataFrame(
                {
                    "backtest_hash": ["bt_leader"],
                    "prediction_hash": ["pred_leader"],
                    "label": ["fwd_ret_5d"],
                    "sharpe": [1.0],
                }
            )

        def champion_lineage(self, prediction_hash):
            return {
                stage: {"backtest_hash": f"bt_{stage}"}
                for stage in ("signal", "allocation", "cost_sensitivity", "risk_overlay")
            }

    paired_metrics.populate_paired_metrics("unit_cs", _Explorer(), verbose=False)

    transitions = {
        kind: overlay
        for kind, overlay in seen
        if kind in {"signal_leader", "allocation_leader", "cost_sensitivity_leader"}
    }
    assert transitions == {
        "signal_leader": False,
        "allocation_leader": False,
        "cost_sensitivity_leader": False,
    }


def test_the_producer_does_not_trim_the_pair_before_handing_it_to_the_bootstrap(
    captured: list[dict],
) -> None:
    """The exact case the default trim is not idempotent on.

    The challenger starts three sessions late, and the baseline - the earlier starter - posts
    an exactly zero return on the challenger's first traded session. One application of the
    default rule starts the sample there, at index 3. A second application sees a baseline that
    is zero at its own index 0 and slides the start to index 4, dropping a live session from
    both series while the producer still reports the sample it measured before the bootstrap
    ran. Passing the untrimmed pair through is what keeps the two numbers the same one.
    """
    rng = np.random.default_rng(11)
    baseline = list(rng.normal(0.0005, 0.01, size=30))
    challenger = list(rng.normal(0.0005, 0.01, size=30))
    challenger[:3] = [0.0, 0.0, 0.0]
    baseline[3] = 0.0

    row = paired_metrics._populate_pair(
        "unit_cs",
        "chal",
        "bench",
        "equal_weight",
        _returns(challenger),
        _returns(baseline),
        252,
        "fwd_ret_5d",
    )

    assert captured[0]["challenger"].size == 30
    assert row["n_overlap"] == 27
    # The figure the producer reports is the sample the bootstrap ran on, not a longer one
    # measured before a second trim shortened it.
    used, _ = joint_returns(captured[0]["challenger"], captured[0]["baseline"])
    assert used.size == row["n_overlap"]
