"""A configuration that reproduces another one's result is not a second trial.

A regularisation grid that runs past the point where the penalty stops binding submits
several configurations and produces one result. On `nasdaq100_microstructure`,
`fwd_dir_15m`, L1 logistic at C=10 and C=100 agree to six decimals in log loss and to
the coefficient in sparsity (195 of 198 non-zero), under two solvers independently. The
same shape is in the fleet: 375 of the 15152 registered backtests carrying returns
reproduce another one in the same (stage, label) bit for bit, in runs of up to six -
`ols` through `ridge_a10.0` on sp500_equity_option_analytics and us_firm_characteristics.

The multiple-testing adjustment asks how many chances the selection had to find a high
Sharpe by luck. A configuration that reproduces another one's series supplies no chance,
and counting it inflates K, the expected maximum Sharpe that follows from K, and so
understates every deflated Sharpe below it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from case_studies.utils.uncertainty import compute_cohort_metrics

PERIODS_PER_YEAR = 252
OBSERVATIONS = 400


def _frame(values: np.ndarray) -> pl.DataFrame:
    start = date(2020, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(values.size)],
            "ret": values,
        }
    )


def _cohort(*, with_repeat: bool) -> dict[str, pl.DataFrame]:
    """Five configurations that differ, optionally plus one that reproduces the first.

    The repeat is an exact copy rather than a near-copy, which is what the grid actually
    produces: two configurations that converge to the same fitted model emit the same
    predictions and therefore the same returns to the last bit.
    """
    rng = np.random.default_rng(11)
    series = {f"bt_{i}": rng.normal(0.0006, 0.01, OBSERVATIONS) for i in range(5)}
    cohort = {name: _frame(values) for name, values in series.items()}
    if with_repeat:
        cohort["bt_repeat"] = _frame(series["bt_0"].copy())
    return cohort


def test_a_repeated_result_is_counted_once_among_the_trials() -> None:
    """The cohort holds six configurations and the adjustment faces five trials."""
    out = compute_cohort_metrics(_cohort(with_repeat=True), periods_per_year=PERIODS_PER_YEAR)

    assert out["k_variants_submitted"] == 6, "the configurations the cohort holds"
    assert out["k_variants"] == 5, "two configurations, one result, one trial"


def test_a_repeated_result_does_not_move_the_correction() -> None:
    """Submitting a configuration that changes nothing must change no published number.

    This is the defect: pre-fix the sixth configuration raised K from 5 to 6, which
    raised the expected maximum Sharpe a zero-skill cohort would reach and lowered the
    deflated Sharpe underneath it, on evidence that had not changed at all.
    """
    without = compute_cohort_metrics(_cohort(with_repeat=False), periods_per_year=PERIODS_PER_YEAR)
    with_repeat = compute_cohort_metrics(
        _cohort(with_repeat=True), periods_per_year=PERIODS_PER_YEAR
    )

    assert with_repeat["leader_hash"] == without["leader_hash"]
    for key in ("dsr_raw", "expected_max_sharpe_raw", "min_trl_periods_raw", "dsr_raw_pvalue"):
        assert with_repeat[key] == without[key], key


def test_a_cohort_with_no_repeat_keeps_every_configuration_as_a_trial() -> None:
    """Guards the premise: a collapse that fires on distinct variants would pass the above."""
    out = compute_cohort_metrics(_cohort(with_repeat=False), periods_per_year=PERIODS_PER_YEAR)

    assert out["k_variants_submitted"] == 5
    assert out["k_variants"] == 5


def test_the_repeat_stays_in_the_cohort_it_is_a_member_of() -> None:
    """Collapsing trials is not dropping members.

    `member_digest` covers every aligned member, and `k_variants_submitted` counts them.
    A collapse that removed the repeat from those would make the row unmatchable against
    the cohort a reader assembles from the registry, which is a different defect from the
    one being fixed.
    """
    from case_studies.utils.uncertainty import cohort_member_digest

    cohort = _cohort(with_repeat=True)
    out = compute_cohort_metrics(cohort, periods_per_year=PERIODS_PER_YEAR)

    assert out["member_digest"] == cohort_member_digest(cohort.keys())
    assert out["k_variants_submitted"] == 6
    assert out["ras_n_strategies"] == 5.0, "RAS records the trials it adjusted for"


def test_a_cohort_that_tried_one_thing_reports_no_correction() -> None:
    """Two configurations and one result between them is not a selection.

    Left unguarded the row is worse than absent: `dsr_raw` is computed at K=1, which is
    the undeflated Sharpe, while the MP and ER estimators refuse a single strategy and
    leave NULLs beside it. A reader takes the one populated column for a corrected figure.
    """
    rng = np.random.default_rng(3)
    values = rng.normal(0.0005, 0.01, OBSERVATIONS)
    cohort = {"bt_a": _frame(values), "bt_b": _frame(values.copy())}

    assert compute_cohort_metrics(cohort, periods_per_year=PERIODS_PER_YEAR) == {}
