"""Tests for case_studies/utils/uncertainty.py CSCV partition + PBO smoke.

Covers two pieces P2.5 added:

1. ``_cscv_split_pairs`` — IS/OOS partition shape and balance for
   ``n_folds`` in {2, 3, 4}, including the asymmetric odd-fold case.
2. ``compute_cohort_metrics`` end-to-end with a ``fold_returns_by_hash``
   argument, asserting that ``pbo`` / ``pbo_median_oos_rank`` /
   ``pbo_mean_degradation`` come back populated (i.e. the
   ``compute_pbo`` field-name and partition wiring is intact).
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from math import comb
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.mark.parametrize(
    "n_folds, is_half, oos_half",
    [
        (2, 1, 1),  # balanced
        (3, 1, 2),  # asymmetric — OOS gets the extra fold
        (4, 2, 2),  # balanced
    ],
)
def test_cscv_split_pairs_partition_shape(n_folds: int, is_half: int, oos_half: int) -> None:
    from case_studies.utils.uncertainty import _cscv_split_pairs

    rng = np.random.default_rng(0)
    k_variants = 5
    fold_sharpes = rng.normal(size=(n_folds, k_variants))

    is_perf, oos_perf = _cscv_split_pairs(fold_sharpes)

    expected_n = comb(n_folds, n_folds // 2)
    assert is_perf.shape == (expected_n, k_variants)
    assert oos_perf.shape == (expected_n, k_variants)

    # Every row must be the mean of `is_half` folds (IS) and
    # `oos_half` folds (OOS) of the original matrix — verified by
    # reconstructing the underlying sums.
    for row_is, row_oos in zip(is_perf, oos_perf, strict=True):
        # IS mean × is_half + OOS mean × oos_half == sum of all folds
        total = fold_sharpes.sum(axis=0)
        reconstructed = row_is * is_half + row_oos * oos_half
        np.testing.assert_allclose(reconstructed, total, atol=1e-12)


def test_cscv_split_pairs_single_fold_returns_empty() -> None:
    from case_studies.utils.uncertainty import _cscv_split_pairs

    is_perf, oos_perf = _cscv_split_pairs(np.array([[1.0, 2.0, 3.0]]))
    assert is_perf.shape == (0, 3)
    assert oos_perf.shape == (0, 3)


def test_compute_cohort_metrics_populates_pbo_with_fold_returns() -> None:
    """End-to-end smoke: PBO fields must come back non-null when
    ``fold_returns_by_hash`` is supplied for >=2 variants with >=2 folds.

    The pre-P2.5 code called ``compute_pbo(fs, fs)`` and read the wrong
    PBOResult attribute names — both bugs would surface here as NULLs.
    """
    from case_studies.utils.uncertainty import compute_cohort_metrics

    rng = np.random.default_rng(7)
    n_periods = 252
    timestamps = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2020, 12, 31),
        interval="1d",
        eager=True,
    ).head(n_periods)

    def _make_frame(mu: float) -> pl.DataFrame:
        ret = rng.normal(loc=mu / 252, scale=0.01, size=n_periods)
        return pl.DataFrame({"timestamp": timestamps, "ret": ret})

    # Three "variants" with hash-shaped keys (32 hex chars satisfies any
    # downstream FK convention; here we just need stable dict keys).
    returns_by_hash = {f"{i:032x}": _make_frame(mu=mu) for i, mu in enumerate([0.05, 0.08, 0.12])}

    n_folds = 4
    fold_returns_by_hash = {
        h: rng.normal(loc=0.0, scale=1.0, size=n_folds) for h in returns_by_hash
    }

    out = compute_cohort_metrics(
        returns_by_hash,
        periods_per_year=252,
        fold_returns_by_hash=fold_returns_by_hash,
        rademacher_n_simulations=50,
        rademacher_seed=0,
    )

    assert out, "compute_cohort_metrics returned empty dict — alignment failed"
    assert out["leader_hash"] in returns_by_hash
    assert out["k_variants"] == 3

    # PBO fields must be populated (the bug-surface check).
    assert out["pbo"] is not None
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["pbo_n_combinations"] == float(comb(n_folds, n_folds // 2))
    assert out["pbo_median_oos_rank"] is not None
    assert out["pbo_mean_degradation"] is not None
    assert out["pbo_n_folds"] == float(n_folds)


def test_an_overlay_that_sits_out_the_first_sessions_still_gets_a_paired_bootstrap() -> None:
    """A risk overlay exists to sit out sessions, and that must not cost it the comparison.

    The two series reach the bootstrap pre-aligned on the timestamp, so position i is the same
    session on both sides. Coercing each side on its own breaks that: the leading run of zeros
    is trimmed per series, so a challenger that stays flat while its carrier trades comes out
    shorter and the equal-length precondition refuses the pair. ``17_risk_management`` raised on
    every overlay for this reason. The trim has to be taken once, over both sides, so that what
    is dropped is the prefix where neither side had a position.
    """
    from case_studies.utils.uncertainty import compute_paired_uncertainty

    rng = np.random.default_rng(3)
    baseline = rng.normal(0.0004, 0.01, size=60)
    challenger = baseline + rng.normal(0.0, 0.002, size=60)
    # The overlay is out of the market for the first three sessions the carrier trades.
    challenger[:3] = 0.0

    paired = compute_paired_uncertainty(
        challenger, baseline, n_boot=20, seed=5, challenger_overlays_baseline=True
    )

    assert paired["bootstrap_n"] == 20.0
    assert np.isfinite(paired["sharpe_diff"])
    assert paired["sharpe_diff_ci95_lo"] <= paired["sharpe_diff"] <= paired["sharpe_diff_ci95_hi"]


def test_a_paired_bootstrap_keeps_the_sessions_the_overlay_sat_out() -> None:
    """Only the leading sessions on which NEITHER side traded are dropped.

    A session the carrier traded and the overlay sat out is the largest instance of the effect
    the comparison exists to measure, whether it falls at the start of the sample or in the
    middle. Starting the sample where both sides are non-zero would delete exactly those rows
    and pull the measured difference toward zero in the direction the overlay is being tested
    for. The trim is the joint analogue of "bars before the first signal", so the sample starts
    where anything first held a position.
    """
    from case_studies.utils.uncertainty import joint_returns

    #                       both flat  | carrier only | both trade | overlay sits out
    baseline = np.array([0.0, 0.0, 0.01, 0.02, 0.015, 0.03, -0.01])
    challenger = np.array([0.0, 0.0, 0.00, 0.00, 0.012, 0.03, -0.02])

    c, b = joint_returns(challenger, baseline, challenger_overlays_baseline=True)

    # Two leading sessions go; the two where only the carrier traded stay.
    assert c.size == b.size == 5
    np.testing.assert_allclose(b, baseline[2:])
    np.testing.assert_allclose(c, challenger[2:])


def test_an_overlay_flat_for_the_whole_sample_is_compared_rather_than_refused() -> None:
    """Holding nothing all sample is an answer about the overlay, not an absence of data.

    Under a both-sides-traded start rule this pair has no starting session at all, so the
    bootstrap returns an empty mapping and `17_risk_management` raises. The overlay did make a
    decision on every one of these sessions; its return was zero. The difference is then the
    carrier's own Sharpe, negated, over the sessions the carrier traded.
    """
    from case_studies.utils.uncertainty import _sample_stats, compute_paired_uncertainty

    rng = np.random.default_rng(7)
    baseline = rng.normal(0.0006, 0.01, size=50)
    challenger = np.zeros(50)

    paired = compute_paired_uncertainty(
        challenger, baseline, n_boot=20, seed=2, challenger_overlays_baseline=True
    )

    assert paired
    assert paired["sharpe_diff"] == pytest.approx(-_sample_stats(baseline, 252).sharpe)


def test_a_strategy_is_not_charged_for_the_sessions_before_its_first_signal() -> None:
    """The default pairing drops the challenger's warmup, and it has to.

    A strategy backtest's daily returns begin at the first bar, not at the first signal, so the
    series carries a leading run of zeros while the equal-weight benchmark it is compared
    against is invested from the first joined session. Those rows are pre-sample for the
    strategy rather than a result, and keeping them would deflate its Sharpe against every
    benchmark. This is the mirror of the risk-overlay case, which is why the caller has to say
    which pair it holds rather than one rule serving both.
    """
    from case_studies.utils.uncertainty import compute_paired_uncertainty, joint_returns

    rng = np.random.default_rng(19)
    benchmark = rng.normal(0.0005, 0.008, size=60)
    strategy = rng.normal(0.0009, 0.011, size=60)
    # No signal yet for the first eight sessions the benchmark was invested through.
    strategy[:8] = 0.0

    c, b = joint_returns(strategy, benchmark)

    assert c.size == b.size == 52
    np.testing.assert_allclose(c, strategy[8:])
    np.testing.assert_allclose(b, benchmark[8:])

    # And the bootstrap agrees with the coercion it performs itself.
    paired = compute_paired_uncertainty(strategy, benchmark, n_boot=20, seed=4)
    overlay_reading = compute_paired_uncertainty(
        strategy, benchmark, n_boot=20, seed=4, challenger_overlays_baseline=True
    )
    assert paired["sharpe_diff"] > overlay_reading["sharpe_diff"]


def test_joint_returns_refuses_a_pair_that_did_not_arrive_aligned() -> None:
    """Position i must already be the same session; there is no way to recover it here."""
    from case_studies.utils.uncertainty import joint_returns

    with pytest.raises(ValueError, match="must arrive aligned"):
        joint_returns(np.ones(10), np.ones(9))


def test_a_paired_bootstrap_refuses_two_series_of_different_lengths() -> None:
    """Truncating to the shorter one would silently compare different sessions."""
    from case_studies.utils.uncertainty import compute_paired_uncertainty

    rng = np.random.default_rng(11)
    baseline = rng.normal(0.0004, 0.01, size=40)

    assert compute_paired_uncertainty(baseline[:30], baseline, n_boot=10, seed=1) == {}


def test_bootstrap_uncertainty_uses_seeded_generator() -> None:
    from case_studies.utils.uncertainty import (
        compute_backtest_uncertainty,
        compute_independent_diff_uncertainty,
        compute_paired_uncertainty,
    )

    rng = np.random.default_rng(17)
    baseline = rng.normal(0.0002, 0.01, size=80)
    challenger = baseline + rng.normal(0.0001, 0.002, size=80)

    backtest = compute_backtest_uncertainty(challenger, n_boot=20, seed=41)
    paired = compute_paired_uncertainty(challenger, baseline, n_boot=20, seed=41)
    independent = compute_independent_diff_uncertainty(
        challenger,
        baseline[:60],
        n_boot=20,
        seed=41,
    )

    assert backtest["bootstrap_n"] == 20.0
    assert paired["bootstrap_n"] == 20.0
    assert independent["bootstrap_n"] == 20.0
    assert backtest == compute_backtest_uncertainty(challenger, n_boot=20, seed=41)
    assert paired == compute_paired_uncertainty(challenger, baseline, n_boot=20, seed=41)
    repeated_independent = compute_independent_diff_uncertainty(
        challenger,
        baseline[:60],
        n_boot=20,
        seed=41,
    )
    assert independent.keys() == repeated_independent.keys()
    np.testing.assert_allclose(
        list(independent.values()),
        list(repeated_independent.values()),
        equal_nan=True,
    )


def test_sparse_bootstrap_samples_do_not_emit_correlation_warnings() -> None:
    from case_studies.utils.uncertainty import compute_backtest_uncertainty

    sparse_returns = np.r_[np.zeros(70), np.ones(10) * 0.01]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = compute_backtest_uncertainty(sparse_returns, n_boot=100, seed=0)

    assert result["bootstrap_n"] == 100.0


def test_where_a_benchmark_pair_starts_is_pinned_by_what_it_returns() -> None:
    """One producer, and the start rule stated as a property rather than as an agreement.

    ``20_strategy_synthesis/01_aggregate_synthesis.py`` carried its own copy of the joint
    coercion, ``_joint_coerce``, while ``case_studies/utils/paired_metrics.py`` reached
    ``joint_returns``. Both wrote ``backtest_paired_metrics`` for the same stage
    transitions, so a change to one and not the other made ``sharpe_diff`` for a row depend
    on which producer ran last. The copy is gone and Ch20 calls the shared helper, so there
    is no second implementation left for a test to check the first against.

    What that test was really protecting is the rule itself, and this pins it directly: for
    a strategy against a benchmark the sample keeps only sessions finite on both sides, and
    opens on the first session where both are trading. Asserted on what comes back rather
    than against a second copy of the arithmetic, because a reference implementation in the
    test would be the same duplication one file further away.

    The pairs are built rather than drawn at random, because the session the rule turns on
    is one where the earlier starter posts an exactly zero return on the later starter's
    opening day, and a continuous distribution never produces one.
    """
    from case_studies.utils.uncertainty import joint_returns

    pairs = [
        # The boundary: the baseline is flat on the session the challenger first trades.
        (
            np.array([0.0, 0.0, 0.0, 0.011, 0.018, -0.009, 0.021]),
            np.array([0.01, -0.02, 0.015, 0.0, 0.02, -0.01, 0.03]),
        ),
        # The mirror: the challenger is flat on the session the baseline first trades.
        (
            np.array([0.012, -0.004, 0.0, 0.011, 0.018, -0.009, 0.021]),
            np.array([0.0, 0.0, 0.015, 0.02, 0.02, -0.01, 0.03]),
        ),
        # Both flat on the same opening session.
        (
            np.array([0.0, 0.0, 0.0, 0.011, 0.018]),
            np.array([0.0, 0.0, 0.0, 0.02, -0.01]),
        ),
        # A non-finite value inside the sample, dropped from both sides.
        (
            np.array([0.0, 0.01, np.nan, 0.011, 0.018]),
            np.array([0.02, 0.0, 0.005, 0.02, -0.01]),
        ),
    ]
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 40))
        challenger = rng.normal(0.0, 0.01, n)
        baseline = rng.normal(0.0, 0.01, n)
        challenger[: rng.integers(0, 6)] = 0.0
        baseline[: rng.integers(0, 6)] = 0.0
        challenger[rng.integers(0, n)] = 0.0
        baseline[rng.integers(0, n)] = 0.0
        pairs.append((challenger, baseline))

    for challenger, baseline in pairs:
        kept_c, kept_b = joint_returns(challenger, baseline)

        assert kept_c.size == kept_b.size, "a paired bootstrap refuses a misaligned pair"
        assert np.isfinite(kept_c).all() and np.isfinite(kept_b).all()

        finite = np.isfinite(challenger) & np.isfinite(baseline)
        survivors_c = np.asarray(challenger, dtype=np.float64)[finite]
        survivors_b = np.asarray(baseline, dtype=np.float64)[finite]
        both_trading = (survivors_c != 0.0) & (survivors_b != 0.0)

        if not both_trading.any():
            assert kept_c.size == 0, "no session where both traded is no sample"
            continue

        # The sample is a suffix of the jointly finite sessions, so nothing inside it is
        # reordered or dropped, and it opens on the first session where both were trading.
        assert kept_c.size == survivors_c.size - int(np.flatnonzero(both_trading)[0])
        np.testing.assert_allclose(kept_c, survivors_c[survivors_c.size - kept_c.size :])
        np.testing.assert_allclose(kept_b, survivors_b[survivors_b.size - kept_b.size :])
        assert kept_c[0] != 0.0 and kept_b[0] != 0.0
        assert not (
            (survivors_c[: survivors_c.size - kept_c.size] != 0.0)
            & (survivors_b[: survivors_b.size - kept_b.size] != 0.0)
        ).any(), "no earlier session had both sides trading"


def test_the_ch20_producer_can_say_which_pairing_it_holds() -> None:
    """The copy took no position, and a producer that cannot say gets one pair type wrong.

    Ch20's ``_joint_coerce`` hard-coded the both-sides-traded rule. That is right for a
    strategy against a benchmark, where the challenger's leading zeros are warmup before its
    first signal, and wrong for a risk overlay against its own carrier, where a flat
    challenger is a position it chose to hold and is the effect the comparison exists to
    measure. Trimming those rows pulls the measured difference toward zero in exactly the
    direction the overlay is under test, and nothing in the copy could express the
    difference.

    Every pair this notebook builds today is the first kind. The point is that the second is
    now sayable, so a future overlay pair is a keyword rather than a second copy of the
    coercion with the other rule in it.

    Ch20's real function is lifted out of the notebook source rather than reimplemented,
    because a test that passes the flag to the shared helper by hand cannot see what the
    producer does with it.
    """
    import ast

    from case_studies.utils.uncertainty import compute_paired_uncertainty, joint_returns

    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "20_strategy_synthesis" / "01_aggregate_synthesis.py").read_text()
    tree = ast.parse(source)
    wanted = {"_populate_pair", "_min_paired_n"}
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict = {
        "np": np,
        "pl": pl,
        "joint_returns": joint_returns,
        "compute_paired_uncertainty": compute_paired_uncertainty,
        "compute_independent_diff_uncertainty": lambda *a, **k: {},
    }
    exec(  # noqa: S102 - running the producer itself is the point of the check
        compile(ast.Module(body=definitions, type_ignores=[]), "01_aggregate_synthesis", "exec"),
        namespace,
    )
    populate_pair = namespace["_populate_pair"]

    # The challenger sits flat for its first six sessions while the baseline trades.
    rng = np.random.default_rng(5)
    baseline = rng.normal(0.0005, 0.01, 40)
    challenger = baseline + rng.normal(0.0, 0.002, 40)
    challenger[:6] = 0.0
    frames = [
        pl.DataFrame(
            {
                "timestamp": [date(2020, 1, 1) + timedelta(days=i) for i in range(40)],
                "ret": list(values),
            }
        )
        for values in (challenger, baseline)
    ]

    as_benchmark = populate_pair(
        "unit_cs",
        "chal",
        "bench",
        "equal_weight",
        *frames,
        252,
        "fwd_ret_5d",
        challenger_overlays_baseline=False,
    )
    as_overlay = populate_pair(
        "unit_cs",
        "chal",
        "bench",
        "risk_overlay_leader",
        *frames,
        252,
        "fwd_ret_5d",
        challenger_overlays_baseline=True,
    )

    assert as_benchmark["n_overlap"] == 34, "warmup before the challenger's first signal"
    assert as_overlay["n_overlap"] == 40, "six sessions the overlay chose to sit out"
    assert as_benchmark["sharpe_diff"] != as_overlay["sharpe_diff"], (
        "if the two shapes agreed, neither the keyword nor this test would be worth having"
    )


def test_the_overlay_trim_starts_at_the_earlier_of_the_two_first_sessions() -> None:
    """The overlay rule is the mirror: the sample opens as soon as anything held a position."""
    from case_studies.utils.uncertainty import joint_returns

    baseline = np.array([0.0, 0.0, 0.01, 0.02, 0.0, 0.03])
    challenger = np.array([0.0, 0.012, 0.0, 0.0, 0.0, 0.02])

    c, b = joint_returns(challenger, baseline, challenger_overlays_baseline=True)

    assert c.size == b.size == 5
    np.testing.assert_allclose(c, challenger[1:])
    np.testing.assert_allclose(b, baseline[1:])
