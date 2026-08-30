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
rather than trimming first and passing the result on, so the size it registers is always the
size the bootstrap saw. Both rules survive a second application today; the point of the
pass-through is that neither has to, and the check below is on what the bootstrap received
rather than on what the producer measured.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from case_studies.utils import paired_metrics
from case_studies.utils.uncertainty import STAGE_SEQUENCE, joint_returns


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


class _Explorer:
    """Minimal `BacktestExplorer` stand-in: one leader, one lineage entry per stage."""

    def __init__(self, stages=STAGE_SEQUENCE):
        self._stages = tuple(stages)

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
        return {stage: {"backtest_hash": f"bt_{stage}"} for stage in self._stages}


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

    # `unit_cs` is synthetic and has no config/setup.yaml, so the annualization factor
    # is stated here. `populate_paired_metrics` reads it from the case study's own
    # declaration when omitted, and this test is about the pair shape, not the scale.
    paired_metrics.populate_paired_metrics(
        "unit_cs", _Explorer(), periods_per_year=252, verbose=False
    )

    # One transition per consecutive pair of stages, and none of them takes the overlay
    # shape. Derived from STAGE_SEQUENCE rather than listed, so reordering the stages
    # cannot make this pass by agreeing with itself.
    expected = {f"{prev}_leader": False for prev in STAGE_SEQUENCE[:-1]}
    assert {kind: overlay for kind, overlay in seen if kind in expected} == expected


def test_the_producer_skips_a_stage_the_case_study_has_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairs are consecutive *present* stages, so a missing middle costs one transition.

    A case study that has not run the risk stage still gets its allocation-to-cost
    comparison. Pairing off a fixed list dropped both transitions instead of bridging.
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

    explorer = _Explorer(("signal", "allocation", "cost_sensitivity"))
    paired_metrics.populate_paired_metrics("unit_cs", explorer, periods_per_year=252, verbose=False)

    kinds = [kind for kind, _ in seen if kind.endswith("_leader")]
    assert "allocation_leader" in kinds
    assert "risk_overlay_leader" not in kinds


def test_the_producer_reports_the_sample_the_bootstrap_ran_on(captured: list[dict]) -> None:
    """The size the producer registers has to be the size the bootstrap saw.

    The producer measures the trim and then hands `compute_paired_uncertainty` the untrimmed
    pair, which trims it once itself. Trimming first and passing the result on would leave the
    two figures free to drift apart the moment either rule stops surviving a second
    application, and the registered `n_overlap` would then name a sample that was never used.
    The pair below is the case where the two rules are closest to parting company: the
    challenger starts three sessions late and the baseline is flat on the session it opens.
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
    assert row["n_overlap"] == 26
    # The figure the producer reports is the sample the bootstrap ran on, not a longer one
    # measured before a second trim shortened it.
    used, _ = joint_returns(captured[0]["challenger"], captured[0]["baseline"])
    assert used.size == row["n_overlap"]
