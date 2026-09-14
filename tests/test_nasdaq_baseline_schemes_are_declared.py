"""`signal_passes.baseline_schemes` must name arms the declared grids produce.

`cs-nasdaq100_microstructure` was red on the merge of #1050. `14_backtest` raised

    backtest.sweep.signal_passes.baseline_schemes names ['ew_top10', 'ew_top20'], which
    get_entry_schemes_for does not produce for fwd_ret_15m

and 15, 16 and 17 then failed for want of the rows 14 never registered. The config was
correct. `get_entry_schemes_for` drops every concentration the cross-section cannot realize,
and a long-short selection needs 2k distinct names: the fixture trades twelve symbols, so
`ew_top5` is realizable there and neither `ew_top10` nor `ew_top20` is. Calling the function
at n_assets=12 with long_short=True reproduces that run's arm list exactly. The names went
missing on a narrow run by construction. The notebook's check read `MAX_ENTRY_SCHEMES`, a
different reduction that the fixture leaves unset, and so reported a typo that was not there.

The notebook now asks `get_entry_schemes_for` what the grids declare, with the feasibility
rule lifted, and compares the names against that. Both halves of the separation are pinned
here: a name must be declared, and a narrow cross-section must still drop names. Without the
second, the first would pass against a function that had stopped filtering at all.
"""

from __future__ import annotations

import pytest

from case_studies.utils.sweep_config import get_entry_schemes_for, load_sweep

CASE_STUDY = "nasdaq100_microstructure"

# What `14_backtest` passes when it asks what the grids declare rather than what a panel can
# trade. It has to stay above every concentration any grid in setup.yaml declares.
NO_FEASIBILITY_LIMIT = 1_000_000


def _labels() -> list[str]:
    return sorted(load_sweep(CASE_STUDY).get("top_k_grid") or {})


def _declared_names(label: str, long_short: bool) -> set[str]:
    return {
        s["name"]
        for s in get_entry_schemes_for(
            CASE_STUDY,
            label,
            NO_FEASIBILITY_LIMIT,
            long_short=long_short,
            ranked_width=NO_FEASIBILITY_LIMIT,
        )
    }


@pytest.mark.parametrize("label", _labels())
@pytest.mark.parametrize("long_short", [False, True])
def test_baseline_and_reference_schemes_are_names_the_grids_produce(
    label: str, long_short: bool
) -> None:
    passes = load_sweep(CASE_STUDY).get("signal_passes") or {}
    if not passes:
        pytest.skip(f"{CASE_STUDY} declares no backtest.sweep.signal_passes")

    declared = _declared_names(label, long_short)
    for key in ("baseline_schemes", "reference_schemes"):
        named = [str(n) for n in (passes.get(key) or [])]
        assert named, f"signal_passes.{key} is empty, so that pass would sweep nothing"
        missing = [n for n in named if n not in declared]
        assert not missing, (
            f"signal_passes.{key} names {missing} for {label}, which the declared "
            f"top_k/percentile/quantile grids do not produce at any cross-section width. "
            f"Names they do produce: {sorted(declared)[:8]}"
        )


def test_a_narrow_cross_section_still_drops_concentrations() -> None:
    """The control. If feasibility stopped filtering, the test above could not fail."""
    label = _labels()[0]
    declared_k = sorted(int(k) for k in load_sweep(CASE_STUDY)["top_k_grid"][label])
    assert len(declared_k) > 1, "needs at least two concentrations to have one to drop"

    # Wide enough for the smallest concentration, narrower than the largest.
    width = declared_k[-1]
    narrow = {
        s["name"]
        for s in get_entry_schemes_for(
            CASE_STUDY, label, width, long_short=False, ranked_width=width
        )
    }
    assert f"ew_top{declared_k[0]}" in narrow
    assert f"ew_top{declared_k[-1]}" not in narrow


def test_the_ci_fixture_width_is_what_made_the_job_red() -> None:
    """The run that failed, replayed at its own width.

    Pins the mechanism rather than the incident: twelve symbols long-short realizes the
    smallest declared concentration and neither of the other two. If the fixture widens,
    this is the test that says so, and the notebook keeps working either way.
    """
    fixture_n_assets = 12  # tests/overrides.yaml, nasdaq100_microstructure/14_backtest
    label = "fwd_ret_15m"
    at_fixture_width = {
        s["name"]
        for s in get_entry_schemes_for(
            CASE_STUDY,
            label,
            fixture_n_assets,
            long_short=True,
            ranked_width=fixture_n_assets,
        )
    }
    declared = _declared_names(label, long_short=True)

    assert "ew_top5" in at_fixture_width
    assert {"ew_top10", "ew_top20"}.isdisjoint(at_fixture_width)
    assert {"ew_top5", "ew_top10", "ew_top20"} <= declared
