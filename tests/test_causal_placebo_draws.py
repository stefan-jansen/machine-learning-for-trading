"""The placebo draws behind ``refutation_p`` must survive into the registry.

Only the scalars were stored - the p-value and the successful draw count - so every
causal notebook's permutation-distribution figure read ``placebo_effects`` off an
in-memory result. That key is populated on the run that fits and absent on every read
afterwards, so the figure rendered empty behind its guard while the prose described the
distribution it was meant to show.

The draws are the evidence for the refutation verdict rather than a diagnostic
byproduct: a p-value cannot say whether the draws could have rejected at all, and the
distribution is what shows a reader the observed effect against what noise produces.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from case_studies.research.causal import CausalResult
from case_studies.utils.causal import _placebo_draws_json
from case_studies.utils.registry.registration import register_causal_run

SPEC = '{"family":"causal_dml","identity_version":3}'
DRAWS = [0.011, -0.004, 0.002, -0.017, 0.009]


def _register(case_dir, *, placebo_json, started_at="first") -> None:
    register_causal_run(
        "test_case",
        "causal_placebo",
        label="fwd_ret_5d",
        treatment="ivrv_spread",
        confounders_json='["rv_20"]',
        embargo=10,
        n_folds=5,
        n_obs=100,
        dml_effect=-0.02,
        dml_se_hac=0.02,
        p_value_hac=0.25,
        naive_effect=-0.02,
        confounding_bias_pct=-0.5,
        refutation_p=0.4,
        refutation_n_successful=len(DRAWS),
        refutation_placebo_json=placebo_json,
        spec_json=SPEC,
        notebook="12_causal_dml",
        started_at=started_at,
        elapsed_s=1.0,
        case_dir=case_dir,
    )


def _study(case_dir):
    return SimpleNamespace(root=case_dir, output_root=None)


def _stored(case_dir):
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        return db.execute(
            "SELECT refutation_placebo_json FROM causal_runs WHERE causal_hash = ?",
            ("causal_placebo",),
        ).fetchone()[0]


def test_the_draws_round_trip_from_the_fit_to_the_reader(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, placebo_json=_placebo_draws_json({"placebo_effects": DRAWS}))

    result = CausalResult.open(_study(case_dir), "causal_placebo")

    assert result.metrics["placebo_effects"] == DRAWS


def test_a_run_registered_before_the_column_existed_reads_as_empty(tmp_path) -> None:
    """An empty list, not None, so a caller needs one check rather than two."""
    case_dir = tmp_path / "test_case"
    _register(case_dir, placebo_json=None)

    result = CausalResult.open(_study(case_dir), "causal_placebo")

    assert result.metrics["placebo_effects"] == []


def test_re_registration_without_draws_does_not_erase_them(tmp_path) -> None:
    """Fill-once. A re-registration that recomputes the draws fills the column; one
    that does not - a metadata correction, a supersedes declaration - must not blank
    evidence the earlier run established."""
    case_dir = tmp_path / "test_case"
    _register(case_dir, placebo_json=_placebo_draws_json({"placebo_effects": DRAWS}))

    _register(case_dir, placebo_json=None, started_at="second")

    assert _stored(case_dir) is not None
    assert CausalResult.open(_study(case_dir), "causal_placebo").metrics["placebo_effects"] == DRAWS


def test_a_fit_that_produced_no_draws_stores_nothing(tmp_path) -> None:
    """`_placebo_draws_json` distinguishes 'no draws' from 'draws that were all zero'
    only by emptiness, which is the right rule: a refutation that ran no successful
    placebo has no distribution to show, and storing `[]` would claim it did."""
    assert _placebo_draws_json({"placebo_effects": []}) is None
    assert _placebo_draws_json({}) is None
    assert _placebo_draws_json({"placebo_effects": [0.0, 0.0]}) == "[0.0, 0.0]"
