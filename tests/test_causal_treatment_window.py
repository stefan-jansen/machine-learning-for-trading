"""Every configured treatment must resolve to a construction window.

A canonical causal run refuses when `case_studies/utils/causal.py` cannot say how many bars
the treatment's own construction spans, because a placebo that permutes in blocks shorter
than that window destroys the serial dependence the refutation exists to preserve - and the
p-value it produces reads like a refutation without being one.

The refusal shipped ahead of the declarations. Eight of the nine case studies resolved to
None: the feature register can only answer for a family that is suffix-keyed, and `carry_pct`,
`skip_recent_6_1`, `mom_skip_recent`, `r12_2`, `past_ret_12m_skip`, `vrp_21d`, `ivrv_spread`
and `signed_vol_share` are not. Their causal stages would have failed at resolve time, before
any fit, and nothing would have caught it: `tests/pm_helpers.py` forces the preview tier in CI,
and preview only warns.

This reads the configuration and nothing else, so it costs no data and runs everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from case_studies.utils.causal import _treatment_persistence_steps

REPO = Path(__file__).resolve().parent.parent
SETUPS = sorted(REPO.glob("case_studies/*/config/setup.yaml"))


def _configured() -> list[tuple[str, dict]]:
    out = []
    for path in SETUPS:
        setup = yaml.safe_load(path.read_text()) or {}
        if (setup.get("causal") or {}).get("treatment"):
            out.append((path.parent.parent.name, setup))
    return out


CONFIGURED = _configured()


def test_every_case_study_configures_a_causal_treatment() -> None:
    """Nine case studies, nine causal stages - so nine is what the parametrization covers.

    Without this a setup.yaml that lost its `causal` block would silently drop out of the
    check below rather than fail it.
    """
    assert len(CONFIGURED) == 9, [name for name, _ in CONFIGURED]


@pytest.mark.parametrize("case_study,setup", CONFIGURED, ids=[n for n, _ in CONFIGURED])
def test_treatment_window_resolves(case_study: str, setup: dict) -> None:
    treatment = setup["causal"]["treatment"]
    steps = _treatment_persistence_steps(setup, treatment)
    assert steps is not None, (
        f"{case_study} declares treatment {treatment!r} and no construction window for it, so "
        "a canonical causal run refuses at resolve time. Declare `causal.treatment_window` as "
        "the number of bars the treatment's own construction spans, derived from the code that "
        "builds the column - not from the name."
    )
    assert steps >= 1
