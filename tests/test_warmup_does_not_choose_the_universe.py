"""A warmup prefix adds history to the universe a run trades; it does not pick it.

``load_backtest_prices_for`` walks ``start_date`` back when the caller declares
``warmup_periods``, so a rolling-vol allocator sees history before the first rebalance.
``max_symbols`` keeps the entities with the most rows. Applying the second to the panel
the first widened lets the warmup decide *which* symbols survive: a name quoted before
the window but thin inside it outranks one quoted only inside it.

Two stages of one chain then reduce to the same count and a different set.
``sp500_equity_option_analytics/14_backtest`` declares its universe off the canonical
window and writes the digest into ``backtest_hash``; ``16_risk_management`` loads the
same label with an allocator warmup and checks the declaration against its own panel.
On 2026-09-09 that read "Declared 21 symbols (digest 029b8ccd0047), panel holds 21
(digest 99ec40e8286d)" and stopped the run, which is the check doing its job - the
portfolio would otherwise have registered under an identity describing a different one.

Only a reduced run can reach this. Production passes ``max_symbols=0``.
"""

from __future__ import annotations

import pytest

CASE_STUDY = "sp500_equity_option_analytics"
LABEL = "fwd_dir_10d"
MAX_SYMBOLS = 21
WARMUP_PERIODS = 126


@pytest.fixture
def panels():
    pl = pytest.importorskip("polars")
    from case_studies.utils.backtest_loaders import load_backtest_prices, load_backtest_prices_for

    try:
        windowed = load_backtest_prices_for(
            CASE_STUDY, LABEL, split="validation", max_symbols=MAX_SYMBOLS
        )
        warmed = load_backtest_prices_for(
            CASE_STUDY,
            LABEL,
            split="validation",
            warmup_periods=WARMUP_PERIODS,
            max_symbols=MAX_SYMBOLS,
        )
        unreduced = load_backtest_prices_for(CASE_STUDY, LABEL, split="validation")
        warmed_unreduced = load_backtest_prices_for(
            CASE_STUDY, LABEL, split="validation", warmup_periods=WARMUP_PERIODS
        )
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(f"no {CASE_STUDY} price panel on this checkout: {exc}")
    del pl
    return windowed, warmed, unreduced, warmed_unreduced


def test_the_reduction_actually_bites_on_this_panel(panels) -> None:
    """Otherwise the test below passes on a panel where every symbol survives."""
    windowed, _, unreduced, _ = panels

    assert windowed["symbol"].n_unique() == MAX_SYMBOLS
    assert unreduced["symbol"].n_unique() > MAX_SYMBOLS, (
        "the panel is no wider than the cap, so nothing is being chosen and a set "
        "mismatch could not arise either way"
    )


def test_the_warmup_read_trades_the_universe_the_windowed_read_declares(panels) -> None:
    from case_studies.utils.backtest_presets import traded_universe_declaration

    windowed, warmed, _, _ = panels

    assert (
        traded_universe_declaration(warmed)["digest"]
        == traded_universe_declaration(windowed)["digest"]
    )


def test_the_cap_does_not_shorten_the_warmup(panels) -> None:
    """The narrowing must not undo what the warmup is for.

    Stated against the *unreduced warmed* read rather than against the windowed one. How far
    back a warmup reaches depends on how much history the panel holds before the window, and
    the CI fixture holds none - its first session is the window's first session, so a strict
    `warmed.min() < windowed.min()` is false there through no defect and the job fails on the
    shape of the fixture. Comparing the two warmed reads asks the question the cap can
    actually get wrong: whether reducing the universe also reduced the history.
    """
    windowed, warmed, _, warmed_unreduced = panels

    assert warmed["timestamp"].min() == warmed_unreduced["timestamp"].min()
    assert warmed["timestamp"].min() <= windowed["timestamp"].min()
    assert warmed["timestamp"].max() == windowed["timestamp"].max()


def test_the_warmup_reaches_back_where_the_panel_has_history(panels) -> None:
    """The other half, and it only holds where there is history to reach.

    Skipped rather than weakened on a panel whose first session is the window's first
    session, and the skip says which panel it measured so an absent assertion is never read
    as a passing one.
    """
    windowed, warmed, _, warmed_unreduced = panels

    if warmed_unreduced["timestamp"].min() >= windowed["timestamp"].min():
        pytest.skip(
            f"{CASE_STUDY} holds no session before {windowed['timestamp'].min()}, so a "
            f"{WARMUP_PERIODS}-period warmup has nothing to reach back to"
        )
    assert warmed["timestamp"].min() < windowed["timestamp"].min()
