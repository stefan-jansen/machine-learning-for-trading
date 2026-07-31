"""Tail p-values must stay positive and finite for extreme test statistics.

A two-tailed p-value written as ``2 * (1 - dist.cdf(abs(stat)))`` returns exactly
``0.0`` once ``abs(stat)`` passes roughly 8.35, because ``cdf`` rounds to ``1.0``
long before the tail mass underflows: the subtraction cancels every remaining
significant digit. It is already wrong by 60% at 8.3. The survival function
``dist.sf`` evaluates the tail directly and stays accurate to the smallest normal
double::

    |t| = 6.00   2 * (1 - cdf) = 2.138e-09   2 * sf = 2.138e-09
    |t| = 8.30   2 * (1 - cdf) = 2.220e-16   2 * sf = 1.385e-16
    |t| = 8.94   2 * (1 - cdf) = 0           2 * sf = 5.702e-19

A notebook that teaches inference must not print ``p=0`` for a value no
computation produced, and ``p_value_hac`` must not be stored as ``0.0``.

Both halves run in the always-on ``test-unit`` job, which installs scipy,
statsmodels and scikit-learn for the numeric case. The ``importorskip`` calls
keep the static scan usable from a stdlib-only environment; they are not the CI
path, and a skip there would mean the modelling stack went missing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Chapter notebooks, case studies, and shared helpers. Tests are excluded: the
# baseline assertion below deliberately writes the broken form.
SCANNED_GLOBS = ("[0-9][0-9]_*/*.py", "case_studies/**/*.py", "utils/**/*.py")

ONE_MINUS_CDF = re.compile(r"1(\.0)?\s*-\s*[A-Za-z_][\w.]*\.cdf\(")

# PENDING, not approved. Each of these is the same defect, and the one-token fix
# is known. It is not applied here because every one of them sits in a paired
# notebook: editing the `.py` makes the committed `.ipynb` stale, and the
# provenance gate (`.github/scripts/notebook_provenance.py`) then requires a
# production re-run in the canonical environment. That re-run is the owning
# chapter's or case study's work, in its own worktree and its own PR, so the fix
# ships with the run that renders it rather than leaving output the source no
# longer produces.
#
# A worker touching one of these files must apply `sf`, re-run, and delete the
# row. The list only shrinks: it is a baseline so a *new* occurrence fails
# immediately, not permission to add one.
PENDING = {
    "07_defining_the_learning_task/06_ic_inference.py": 2,
    "07_defining_the_learning_task/07_multiple_testing.py": 1,
    "09_model_based_features/02_structural_breaks.py": 2,
    "15_causal_estimation/09_adia_causal_benchmark.py": 1,
    "15_causal_estimation/11_factor_zoo_validation.py": 1,
    "16_strategy_simulation/11_sharpe_ratio_inference.py": 1,
    "17_portfolio_construction/05_factor_allocation_evidence.py": 1,
    "19_risk_management/01_var_cvar.py": 2,
    "case_studies/sp500_equity_option_analytics/03_financial_features.py": 1,
}


def _occurrences() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for glob in SCANNED_GLOBS:
        for path in sorted(REPO_ROOT.glob(glob)):
            rel = path.relative_to(REPO_ROOT).as_posix()
            hits = [
                f"{rel}:{lineno}: {line.strip()}"
                for lineno, line in enumerate(path.read_text().splitlines(), start=1)
                if ONE_MINUS_CDF.search(line)
            ]
            if hits:
                found[rel] = hits
    return found


def test_no_new_tail_probability_is_written_as_one_minus_cdf():
    """Static guard: the cancellation cannot appear in a file that is clean today."""
    violations = [
        hit for rel, hits in _occurrences().items() for hit in hits[PENDING.get(rel, 0) :]
    ]

    assert not violations, "use dist.sf(x), not 1 - dist.cdf(x):\n" + "\n".join(violations)


def test_pending_baseline_has_no_stale_rows():
    """A row that no longer matches must be deleted, so the baseline only shrinks."""
    found = _occurrences()
    stale = [
        f"{rel}: baseline expects {count}, found {len(found.get(rel, []))}"
        for rel, count in PENDING.items()
        if len(found.get(rel, [])) != count
    ]

    assert not stale, "PENDING is out of date - fix the file and drop the row:\n" + "\n".join(stale)


def test_scipy_baseline_confirms_the_cancellation():
    """The premise: `1 - cdf` is exactly zero where `sf` is not."""
    stats = pytest.importorskip("scipy.stats")

    assert 2 * (1 - stats.t.cdf(8.94, df=4231)) == 0.0
    assert 2 * stats.t.sf(8.94, df=4231) > 0.0


def test_dml_hac_pvalue_survives_extreme_t():
    """`manual_dml_timeseries` writes `p_value_hac` into the registry.

    The treatment effect here is estimated almost without noise, so the HAC
    t-statistic lands around 20 - past the point where `1 - cdf` cancels, well
    short of where the true tail mass underflows.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("statsmodels")
    pytest.importorskip("sklearn")
    from case_studies.utils.causal import manual_dml_timeseries

    rng = np.random.default_rng(7)
    n = 600
    x = rng.normal(size=(n, 2))
    t = x[:, 0] * 0.5 + rng.normal(scale=0.5, size=n)
    y = x[:, 1] * 0.3 + 0.12 * t + rng.normal(scale=1e-4, size=n)

    result = manual_dml_timeseries(y, t, x, n_folds=5, embargo=5)

    assert abs(result["t_stat_hac"]) > 8.35, "fixture must reach the underflow zone"
    assert result["p_value_hac"] > 0.0, "p-value underflowed to exactly zero"
    assert np.isfinite(result["p_value_hac"])
