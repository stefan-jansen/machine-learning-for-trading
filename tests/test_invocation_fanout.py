"""What `invocations` in tests/overrides.yaml has to be true about.

Three case studies take their label as a papermill parameter and declare more than
one, and until this grammar existed the runner could execute a notebook once. The
chain ran the primary label and the rest were never exercised - while `17_costs` and
`20_strategy_analysis` (`14_costs` and `17_strategy_analysis` in
`us_firm_characteristics`) assert that every declared label carries rankable
validation backtests. A claim tested on one label of five is not tested.

These checks are about the declaration, not about a run: they hold the fan-out to the
labels the case study declares, hold the stages of one chain to the same fan-out, and
hold a `requires_stage` pair to fan-outs that line up. None of them needs a notebook.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.pm_helpers import OVERRIDES_PATH, invocations_for

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERRIDES: dict = yaml.safe_load(OVERRIDES_PATH.read_text()) or {}

FANNED_OUT = sorted(
    key for key, entry in OVERRIDES.items() if isinstance(entry, dict) and entry.get("invocations")
)


def _declared_labels(case_study: str) -> set[str]:
    """The labels this case study sweeps, read from its own setup.yaml."""
    setup = yaml.safe_load(
        (REPO_ROOT / "case_studies" / case_study / "config" / "setup.yaml").read_text()
    )
    return set(((setup.get("backtest") or {}).get("sweep") or {}).get("top_k_grid") or {})


def _case_study(key: str) -> str:
    return key.split("/")[1]


def test_some_notebook_declares_several_invocations() -> None:
    """Without this the three checks below hold vacuously and would not say so."""
    assert FANNED_OUT, "no entry declares `invocations`; the checks below test nothing"


@pytest.mark.parametrize("key", FANNED_OUT)
def test_a_fanned_out_notebook_covers_every_label_its_case_study_declares(key: str) -> None:
    """The point of the fan-out, checked against setup.yaml rather than against itself.

    Adding a label to `backtest.sweep.top_k_grid` and not to this file is how a case
    study goes back to exercising a subset while its aggregators still claim it covers
    everything - silently, because nothing else compares the two.
    """
    ids = {run.id for run in invocations_for(OVERRIDES[key], key=key)}
    assert ids == _declared_labels(_case_study(key))


def test_every_stage_of_one_chain_fans_out_the_same_way() -> None:
    """15 reads what 14 registered for its label; a stage covering fewer breaks the chain."""
    by_case_study: dict[str, dict[str, set[str]]] = {}
    for key in FANNED_OUT:
        ids = {run.id for run in invocations_for(OVERRIDES[key], key=key)}
        by_case_study.setdefault(_case_study(key), {})[key] = ids
    for case_study, stages in by_case_study.items():
        assert len(set(map(frozenset, stages.values()))) == 1, (
            f"{case_study}: stages fan out over different labels: "
            + ", ".join(f"{k} -> {sorted(v)}" for k, v in sorted(stages.items()))
        )


def test_a_required_stage_offers_every_invocation_its_dependant_declares() -> None:
    """`_STAGES_RUN_HERE` is keyed by stage, and this is what makes that sufficient.

    `test_case_studies.py` records that a stage ran without recording which invocation
    did, and `requires_stage` reads it the same way. That is exact only while a
    prerequisite fans out over at least what its dependant does: otherwise a run could
    be told its input is present because some other label's run of that stage passed.
    """
    for key, entry in OVERRIDES.items():
        if not isinstance(entry, dict) or not entry.get("requires_stage"):
            continue
        required = entry["requires_stage"]
        required = [required] if isinstance(required, str) else required
        mine = {run.id for run in invocations_for(entry, key=key)}
        if mine == {None}:
            continue
        for stage in required:
            producer_key = f"{key.rsplit('/', 1)[0]}/{stage}"
            producer = OVERRIDES.get(producer_key) or {}
            theirs = {run.id for run in invocations_for(producer, key=producer_key)}
            assert theirs == {None} or mine <= theirs, (
                f"{key} runs {sorted(mine)} but requires {producer_key}, which runs "
                f"{sorted(theirs)}; the missing runs would be reported present"
            )
