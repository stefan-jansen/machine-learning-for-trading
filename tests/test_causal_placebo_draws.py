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

import pytest

from case_studies.research.causal import CausalResult
from case_studies.utils.causal import _placebo_draws_json
from case_studies.utils.registry.registration import register_causal_run

SPEC = '{"family":"causal_dml","identity_version":3}'
DRAWS = [0.011, -0.004, 0.002, -0.017, 0.009]

# The two draw columns, and the same three properties have to hold for each. They are
# separate columns rather than one because they are different quantities on different
# scales: `placebo_effects` is what a reader interprets against the estimate, and
# `placebo_t_stats` is what `refutation_p` is computed on since
# ml4t/agent-workspace#1120. A row can carry the first and not the second - that is
# exactly what every row written before the correction looks like - so the t-statistic
# column needs its own round trip, its own pre-column NULL read, and its own fill-once,
# not the effect column's by association.
# (registry column, `_register` keyword, `CausalResult.metrics` key)
DRAW_COLUMNS = [
    ("refutation_placebo_json", "placebo_json", "placebo_effects"),
    ("refutation_placebo_t_json", "placebo_t_json", "placebo_t_stats"),
]


def _register(case_dir, *, placebo_json=None, placebo_t_json=None, started_at="first") -> None:
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
        refutation_placebo_t_json=placebo_t_json,
        spec_json=SPEC,
        notebook="12_causal_dml",
        started_at=started_at,
        elapsed_s=1.0,
        case_dir=case_dir,
    )


def _study(case_dir):
    # `release_case_root` equals `root` for a study opened over a case directory alone, which is
    # what this fixture is. `CausalResult.open` reads it to overlay a release's `run_log`, and a
    # stub missing it fails there rather than at the behaviour under test.
    return SimpleNamespace(root=case_dir, release_case_root=case_dir, output_root=None)


def _stored(case_dir, column="refutation_placebo_json"):
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        return db.execute(
            f"SELECT {column} FROM causal_runs WHERE causal_hash = ?",  # noqa: S608 - test-local
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


def test_a_tiny_frozen_share_does_not_read_as_zero():
    """The warning has to say how much is frozen, and `.1%` rounded small shares to "0.0%".

    A warning whose text says nothing is wrong is worse than no warning: it trains the
    reader to skip the next one. Measured on nasdaq100_microstructure's 1.2M-row causal
    preview, which froze a share small enough to print as zero under the old format.
    """
    import warnings as _warnings

    from case_studies.utils.causal import _assert_placebo_permutation_possible

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _assert_placebo_permutation_possible(0, 100, 15, 2e-05)
    assert caught, "a nonzero frozen share must warn"
    message = str(caught[0].message)
    assert "0.0% of" not in message, message
    assert "2.00e-05" in message, message


@pytest.mark.parametrize(("column", "keyword", "metric"), DRAW_COLUMNS)
class TestBothDrawColumnsSurviveTheRegistry:
    """The three properties `placebo_effects` had, now asserted for the t-statistic column too.

    `refutation_placebo_t_json` was added with the refutation itself and had none of them:
    the new test file exercised the in-memory refutation dict and stopped at the registry
    boundary, which is the boundary the whole column exists to cross.
    """

    @staticmethod
    def _payload(keyword):
        return {keyword: _placebo_draws_json({"placebo_effects": DRAWS})}

    def test_the_draws_round_trip_from_the_fit_to_the_reader(
        self, tmp_path, column, keyword, metric
    ):
        case_dir = tmp_path / "test_case"
        _register(case_dir, **self._payload(keyword))

        result = CausalResult.open(_study(case_dir), "causal_placebo")

        assert result.metrics[metric] == DRAWS

    def test_a_run_registered_before_the_column_existed_reads_as_empty(
        self, tmp_path, column, keyword, metric
    ):
        case_dir = tmp_path / "test_case"
        _register(case_dir)

        result = CausalResult.open(_study(case_dir), "causal_placebo")

        assert result.metrics[metric] == []

    def test_re_registration_without_draws_does_not_erase_them(
        self, tmp_path, column, keyword, metric
    ):
        case_dir = tmp_path / "test_case"
        _register(case_dir, **self._payload(keyword))

        _register(case_dir, started_at="second")

        assert _stored(case_dir, column) is not None
        assert CausalResult.open(_study(case_dir), "causal_placebo").metrics[metric] == DRAWS


def test_a_row_can_carry_effects_without_t_stats(tmp_path) -> None:
    """Every row written before ml4t/agent-workspace#1120 has this shape, and a reader has
    to be able to tell it apart from a row with no draws at all: the effects are there, the
    t-statistics are not, and `refutation_p` on that row was computed the old way."""
    case_dir = tmp_path / "test_case"
    _register(case_dir, placebo_json=_placebo_draws_json({"placebo_effects": DRAWS}))

    metrics = CausalResult.open(_study(case_dir), "causal_placebo").metrics

    assert metrics["placebo_effects"] == DRAWS
    assert metrics["placebo_t_stats"] == []
