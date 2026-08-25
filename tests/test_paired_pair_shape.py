"""Which pair shape each registered paired-metric row is computed under.

``compute_paired_uncertainty`` needs to be told whether a leading flat run on the challenger
is a warmup before its first signal or a position the challenger chose to hold, because the
two are indistinguishable in the returns and the trim differs. ``paired_metrics`` produces
both shapes: the equal-weight benchmark pairs are independent series, and the stage
transitions - allocation over signal, cost over allocation, risk overlay over cost - run the
challenger on top of its own baseline.

Getting that wrong is a wiring defect rather than a maths one: the numbers are computed
correctly under the wrong rule, and the row registers without complaint. These tests pin the
wiring at the layer where the choice is made.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from case_studies.utils import paired_metrics


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record every call into the bootstrap, and keep the registry out of it."""
    calls: list[dict] = []
    real = paired_metrics.compute_paired_uncertainty

    def spy(challenger, baseline, **kwargs):
        calls.append(
            {
                "challenger": np.asarray(challenger, dtype=np.float64),
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

    paired_metrics._populate_pair(
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
    assert captured[0]["challenger"].size == 34


def test_a_stage_transition_keeps_the_sessions_the_challenger_sat_out(
    captured: list[dict],
) -> None:
    """A challenger built on top of its baseline is live from the baseline's first session."""
    challenger, baseline = _pair()

    paired_metrics._populate_pair(
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
