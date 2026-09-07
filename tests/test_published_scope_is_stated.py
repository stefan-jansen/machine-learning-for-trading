"""What a published number is computed over has to be stated by its publisher.

``populate_paired_metrics`` and ``compute_and_register`` each write a table that a
strategy-analysis notebook prints as its own result. Three of their keyword arguments
decide which registry rows those numbers cover:

* ``prediction_hashes`` — the population the case study still stands behind,
* ``carrier`` — the lineage the case study reports, rather than a raw-Sharpe re-rank,
* ``replace_all`` — whether the write is a snapshot or an addition to what is there.

Each one used to be optional with the widest possible scope as its default, so omitting
it type-checked, ran, and produced plausible numbers that were wrong exactly when the
registry held something the notebook does not report. The tests below drive the two
producers and assert that a call which states no scope does not run at all.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.utils import cohort_metrics, paired_metrics
from case_studies.utils.uncertainty import STAGE_SEQUENCE


class _Explorer:
    """Minimal ``BacktestExplorer`` stand-in that records the scope it was queried under."""

    def __init__(self) -> None:
        self.scopes: list[object] = []

    def best(self, **kwargs):
        self.scopes.append(kwargs.get("prediction_hashes"))
        return pl.DataFrame(
            {
                "backtest_hash": ["bt_leader"],
                "prediction_hash": ["pred_leader"],
                "label": ["fwd_ret_5d"],
                "sharpe": [1.0],
            }
        )

    def champion_lineage(self, prediction_hash):
        blocks = {
            "signal": {"signal": {"method": "top_k"}},
            "allocation": {"allocation": {"method": "risk_parity"}},
            "risk_overlay": {"risk": {"name": "trailing_5pct"}},
            "cost_sensitivity": {"costs": {"commission_bps": 5}},
        }
        lineage: dict = {}
        carried: dict = {}
        for stage in STAGE_SEQUENCE:
            carried = {**carried, **blocks[stage]}
            lineage[stage] = {"backtest_hash": f"bt_{stage}", "_strategy": dict(carried)}
        return lineage


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the producer off the real registry; this is about the call, not the numbers."""
    monkeypatch.setattr(paired_metrics, "_populate_pair", lambda *a, **k: {"kind": a[3]})
    monkeypatch.setattr(paired_metrics, "_aligned_returns", lambda cs, h: None)
    monkeypatch.setattr(paired_metrics, "_val_rank1_carrier", lambda *a, **k: None)
    monkeypatch.setattr(paired_metrics, "_holdout_lineage_for", lambda *a, **k: None)
    monkeypatch.setattr(paired_metrics, "_benchmark_returns_from_artifact", lambda *a, **k: None)
    monkeypatch.setattr(paired_metrics, "register_paired_metrics", lambda *a, **k: None)
    monkeypatch.setattr(paired_metrics, "_drop_retired_generations", lambda cs, df: df)


def test_the_paired_producer_will_not_run_without_a_stated_scope(offline: None) -> None:
    """Omitting all three has to fail, not fall back to the whole registry.

    Pre-fix this call returned a full list of pair rows: every scope argument defaulted
    to the widest reading, so a notebook that forgot one published a number computed
    over rows it does not report and nothing anywhere said so.
    """
    with pytest.raises(TypeError) as excinfo:
        paired_metrics.populate_paired_metrics(
            "unit_cs", _Explorer(), periods_per_year=252, verbose=False
        )

    message = str(excinfo.value)
    for name in ("prediction_hashes", "carrier", "replace_all"):
        assert name in message, f"{name} is a scope the caller has to state"


def test_the_cohort_producer_will_not_run_without_a_stated_scope() -> None:
    """``compute_and_register`` computes K over its population, so it must be given one.

    Pre-fix this returned a populated count dict computed over every prediction set in
    the registry, retired generations included, and the deflated Sharpe that a notebook
    prints below it was adjusted for trials the case study never reported.
    """
    with pytest.raises(TypeError) as excinfo:
        cohort_metrics.compute_and_register("unit_cs", verbose=False)

    assert "prediction_hashes" in str(excinfo.value)


def test_the_whole_registry_is_reachable_only_by_naming_it(offline: None) -> None:
    """The sentinel is not a rename of the default: it has to reach the query as the wide read.

    Guards the premise of the two tests above. If asking for the whole registry did not
    still produce the whole-registry read, making the argument required would have
    changed what the surviving callers compute rather than only how they say it.
    """
    from case_studies.utils.uncertainty import ENTIRE_REGISTRY, NO_CARRIER

    explorer = _Explorer()
    paired_metrics.populate_paired_metrics(
        "unit_cs",
        explorer,
        periods_per_year=252,
        verbose=False,
        prediction_hashes=ENTIRE_REGISTRY,
        carrier=NO_CARRIER,
        replace_all=False,
    )
    assert explorer.scopes and all(scope is None for scope in explorer.scopes)

    scoped = _Explorer()
    paired_metrics.populate_paired_metrics(
        "unit_cs",
        scoped,
        periods_per_year=252,
        verbose=False,
        prediction_hashes=["pred_leader"],
        carrier=NO_CARRIER,
        replace_all=False,
    )
    assert scoped.scopes and all(scope == ["pred_leader"] for scope in scoped.scopes)


@pytest.mark.parametrize("omitted", ["prediction_hashes", "carrier", "replace_all"])
def test_each_scope_is_required_on_its_own(offline: None, omitted: str) -> None:
    """Naming two of the three does not excuse the third."""
    from case_studies.utils.uncertainty import ENTIRE_REGISTRY, NO_CARRIER

    stated = {
        "prediction_hashes": ENTIRE_REGISTRY,
        "carrier": NO_CARRIER,
        "replace_all": False,
    }
    del stated[omitted]

    with pytest.raises(TypeError, match=omitted):
        paired_metrics.populate_paired_metrics(
            "unit_cs", _Explorer(), periods_per_year=252, verbose=False, **stated
        )
