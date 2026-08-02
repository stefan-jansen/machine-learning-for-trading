"""The eligibility screen must sit between the per-symbol and cross-sectional features.

`us_equities_panel/03_financial_features` computes two kinds of feature. Per-symbol
features are shifts and rolling windows `.over("symbol")`, and those count *rows*: run
them on the screened frame and they count eligible rows instead of trading sessions, so
an intermittently eligible stock carries windows spanning the whole excursion. Measured
on production data, `ret_12m_skip` computed the wrong way disagrees with the right way on
14.38% of rows, with a median absolute difference of 0.1411, and 36% of its "252-session"
lookbacks span more than a calendar year (worst case 23.2 years).

Cross-sectional features rank over `timestamp` and must see only the eligible universe.

So the ordering is: per-symbol features on the complete series, then the screen, then the
cross-sectional features. Both tests below need a fixture with an *interior eligibility
gap* - a stock that leaves the universe and comes back. Without one the two orderings are
indistinguishable, which is why this defect read as fine in every rendered figure.
"""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

NOTEBOOK = Path("case_studies/us_equities_panel/03_financial_features.py")

# Applied to the complete series, before the screen.
PER_SYMBOL_STEPS = (
    "compute_momentum_returns",
    "compute_volatility_sharpe",
    "compute_oscillators",
    "compute_trend_distance",
    "compute_rolling_liquidity",
)
# Applied to the screened frame, after it.
CROSS_SECTIONAL_STEPS = (
    "compute_xs_ranks",
    "compute_xs_liquidity_reversion",
    "compute_composites",
)


def _notebook_tree() -> ast.Module:
    return ast.parse(NOTEBOOK.read_text())


def _load_notebook_functions(*wanted_names: str) -> dict[str, object]:
    """Exec the named notebook functions, plus the constants they close over."""
    tree = _notebook_tree()
    wanted = set(wanted_names)
    constants = {"MOMENTUM_HORIZONS", "VOLATILITY_HORIZONS", "MA_HORIZONS"}
    support = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
    ]
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    missing = wanted - {node.name for node in definitions}
    assert not missing, f"{NOTEBOOK} no longer defines {sorted(missing)}"
    module = ast.Module(body=support + definitions, type_ignores=[])
    namespace: dict[str, object] = {"np": np, "pl": pl}
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)  # noqa: S102
    return {name: namespace[name] for name in wanted_names}


@pytest.fixture
def gapped_panel() -> pl.DataFrame:
    """Two stocks over 400 sessions; one leaves the eligible universe and returns.

    `GAPPY` trades under the $5 floor for sessions 120-219 and clears it either side,
    so its eligible history has a 100-session interior gap. `CLEAN` never leaves. The
    price path is a deterministic ramp so a hand-computed expectation is exact.

    Inside the gap `returns` and `dollar_volume` take values sharply unlike the ones
    either side of it. That is what makes the Amihud window discriminating: a rolling
    mean taken over the complete series has to pick the gap values up, and one taken
    over the screened frame reaches past them to rows a hundred sessions earlier.
    `adv_21d` is pinned to a constant well above the $1M floor so that eligibility is
    driven purely by price and the gap has exactly the extent the tests assert.
    """
    start = date(2000, 1, 3)
    rows = []
    for i in range(400):
        stamp = start + timedelta(days=i)
        gapped = 120 <= i < 220
        # CLEAN: 100.0, 100.1, 100.2, ... always well above the floor.
        rows.append(
            {
                "symbol": "CLEAN",
                "timestamp": stamp,
                "close": 100.0 + i * 0.1,
                "adj_close": 100.0 + i * 0.1,
                "adj_high": 100.0 + i * 0.1,
                "adj_low": 100.0 + i * 0.1,
                "volume": 1_000_000.0,
                "returns": 0.01,
                "dollar_volume": 1_000_000_000.0,
                "adv_21d": 1_000_000_000.0,
            }
        )
        # GAPPY: 50 + i/10 normally, but 1.0 (a penny stock) across the gap, where it
        # is also far more volatile and far thinner.
        rows.append(
            {
                "symbol": "GAPPY",
                "timestamp": stamp,
                "close": 1.0 if gapped else 50.0 + i * 0.1,
                "adj_close": 1.0 if gapped else 50.0 + i * 0.1,
                "adj_high": 1.0 if gapped else 50.0 + i * 0.1,
                "adj_low": 1.0 if gapped else 50.0 + i * 0.1,
                "volume": 1_000_000.0,
                "returns": 0.50 if gapped else 0.01,
                "dollar_volume": 2_000_000.0 if gapped else 1_000_000_000.0,
                "adv_21d": 1_000_000_000.0,
            }
        )
    return pl.DataFrame(rows).sort(["symbol", "timestamp"])


ELIGIBLE = (pl.col("close") > 5.0) & (pl.col("adv_21d") > 1_000_000)


def test_fixture_really_has_an_interior_eligibility_gap(gapped_panel: pl.DataFrame) -> None:
    """Guard the guard: a fixture without a gap would make both orderings agree."""
    eligible = gapped_panel.filter(ELIGIBLE)
    per_symbol = eligible.group_by("symbol").agg(pl.len().alias("n"))
    counts = dict(zip(per_symbol["symbol"], per_symbol["n"]))
    assert counts["CLEAN"] == 400
    assert counts["GAPPY"] == 300, "GAPPY must lose exactly the 100-session gap"

    gappy = eligible.filter(pl.col("symbol") == "GAPPY").sort("timestamp")
    spans = gappy["timestamp"].diff().drop_nulls().dt.total_days()
    assert spans.max() == 101, "the gap must be interior, not a truncated head or tail"


def test_per_symbol_features_count_sessions_not_eligible_rows(
    gapped_panel: pl.DataFrame,
) -> None:
    """`ret_12m_skip` must reach back 252 sessions, not 252 surviving rows."""
    (compute_momentum_returns,) = _load_notebook_functions("compute_momentum_returns").values()

    correct = compute_momentum_returns(gapped_panel).filter(ELIGIBLE)
    wrong = compute_momentum_returns(gapped_panel.filter(ELIGIBLE))

    # The last session of GAPPY: index 399, so a true skip-momentum window runs from
    # session 399-252=147 to session 399-21=378. Session 147 is *inside* the gap, and
    # its price is still knowable - the stock traded, it was simply not eligible.
    last = gapped_panel.filter(pl.col("symbol") == "GAPPY")["timestamp"].max()
    price = dict(
        zip(
            gapped_panel.filter(pl.col("symbol") == "GAPPY")["timestamp"],
            gapped_panel.filter(pl.col("symbol") == "GAPPY")["adj_close"],
        )
    )
    stamps = sorted(price)
    expected = price[stamps[378]] / price[stamps[147]] - 1

    got = correct.filter((pl.col("symbol") == "GAPPY") & (pl.col("timestamp") == last))[
        "ret_12m_skip"
    ].item()
    assert got == pytest.approx(expected, rel=1e-12), (
        "computed on the complete series, the 12-1 window must span sessions 147->378"
    )

    # And the screened-frame ordering must give a different, wrong answer - otherwise
    # this test proves nothing about the ordering.
    got_wrong = wrong.filter((pl.col("symbol") == "GAPPY") & (pl.col("timestamp") == last))[
        "ret_12m_skip"
    ].item()
    assert got_wrong != pytest.approx(expected, rel=1e-9), (
        "fixture does not discriminate the two orderings"
    )

    # CLEAN has no gap, so both orderings must agree on it.
    for frame in (correct, wrong):
        clean = frame.filter(pl.col("symbol") == "CLEAN").sort("timestamp")["ret_12m_skip"]
        assert clean.drop_nulls().len() == 400 - 252


def test_rolling_liquidity_counts_sessions_not_eligible_rows(
    gapped_panel: pl.DataFrame,
) -> None:
    """`amihud_illiq` is a 21-session rolling mean and must not close over the gap."""
    (compute_rolling_liquidity,) = _load_notebook_functions("compute_rolling_liquidity").values()

    correct = compute_rolling_liquidity(gapped_panel).filter(ELIGIBLE)
    wrong = compute_rolling_liquidity(gapped_panel.filter(ELIGIBLE))

    gappy = gapped_panel.filter(pl.col("symbol") == "GAPPY").sort("timestamp")
    stamps = list(gappy["timestamp"])
    ratio = [
        abs(r) / (dv + 1) for r, dv in zip(gappy["returns"], gappy["dollar_volume"], strict=True)
    ]
    # Session 220 is GAPPY's first eligible session after the gap. Its 21-session
    # window is sessions 200..220: twenty inside the gap, plus itself.
    first_after_gap = stamps[220]
    expected = sum(ratio[200:221]) / 21

    got = correct.filter((pl.col("symbol") == "GAPPY") & (pl.col("timestamp") == first_after_gap))[
        "amihud_illiq"
    ].item()
    assert got == pytest.approx(expected, rel=1e-12), (
        "on the complete series the Amihud window must cover the 21 sessions before it, "
        "which lie inside the gap"
    )

    # On the screened frame the same row's window reaches back over the gap to the
    # twenty eligible sessions that precede it - sessions 100..119, a hundred sessions
    # earlier - so it must differ, and by a wide margin given the fixture's values.
    eligible_ratio = [ratio[i] for i in range(400) if not (120 <= i < 220)]
    expected_wrong = (sum(eligible_ratio[100:120]) + ratio[220]) / 21
    got_wrong = wrong.filter(
        (pl.col("symbol") == "GAPPY") & (pl.col("timestamp") == first_after_gap)
    )["amihud_illiq"].item()
    assert got_wrong == pytest.approx(expected_wrong, rel=1e-12)

    assert got != pytest.approx(got_wrong, rel=1e-6), (
        "fixture does not discriminate the two orderings for the Amihud window"
    )
    assert got > got_wrong * 100, (
        "the gap sessions are thin and volatile, so the correct window must be far larger"
    )


def test_notebook_applies_the_screen_between_the_two_kinds_of_feature() -> None:
    """The ordering lives in the pipeline cell, so assert it in the source.

    This is the regression guard. The behavioural tests above show the two orderings
    differ; this one pins which order the notebook actually uses.
    """
    tree = _notebook_tree()
    piped: dict[str, tuple[int, str]] = {}  # step -> (lineno, assignment target)
    screen_lineno: int | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        for call in (n for n in ast.walk(node.value) if isinstance(n, ast.Call)):
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "pipe" and call.args:
                arg = call.args[0]
                if isinstance(arg, ast.Name):
                    piped[arg.id] = (arg.lineno, target.id)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "filter"
                and isinstance(func.value, ast.Name)
                and func.value.id == "raw_df"
                and target.id == "df"
            ):
                screen_lineno = node.lineno

    assert screen_lineno is not None, (
        "expected a `df = raw_df.filter(...)` applying the eligibility screen"
    )

    for step in PER_SYMBOL_STEPS:
        assert step in piped, f"{step} is no longer piped in {NOTEBOOK}"
        lineno, target = piped[step]
        assert lineno < screen_lineno, (
            f"{step} is a per-symbol shift/rolling step and must run BEFORE the screen "
            f"(line {lineno} vs screen at {screen_lineno}); on the screened frame its "
            "windows count eligible rows instead of trading sessions"
        )
        assert target == "raw_df", f"{step} must build on the complete series, not {target!r}"

    for step in CROSS_SECTIONAL_STEPS:
        assert step in piped, f"{step} is no longer piped in {NOTEBOOK}"
        lineno, target = piped[step]
        assert lineno > screen_lineno, (
            f"{step} ranks over `timestamp` and must run AFTER the screen "
            f"(line {lineno} vs screen at {screen_lineno}); ranking against ineligible "
            "names is not the ranking the strategy sorts on"
        )
        assert target == "df", f"{step} must build on the screened frame, not {target!r}"
