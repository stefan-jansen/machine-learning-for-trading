"""Absolute moves in a straddle's own premium, followed by contract and by session."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

CONTRACT = ["symbol", "strike", "expiration"]


def straddle_premium_moves(
    chains: pl.LazyFrame,
    *,
    horizons: Sequence[int],
    entry_window: tuple[int, int],
) -> pl.DataFrame:
    """Absolute relative moves in the straddle premium, one row per entry candidate.

    Two constructions here are easy to get wrong and silent when they are. The daily
    straddle panel re-picks the contract nearest the money every session, so differencing
    it measures the switch to a new strike rather than a move in the premium; the legs are
    therefore paired on ``(symbol, strike, expiration)`` and followed through that one
    contract's life. And a row offset counts quotes rather than sessions, so a contract
    that goes unquoted for a day would contribute a mistimed move; the later observation
    is joined on a session index built from the chains' own trading days instead, and a
    gap yields a null that drops out of the horizon rather than a wrong value.

    Parameters
    ----------
    chains
        Lifecycle-preserving option chains, one row per leg, session and contract, as
        :func:`data.load_sp500_options_straddles_raw` returns them lazily.
    horizons
        Session offsets to measure the move over. Each becomes a column ``h<offset>``.
    entry_window
        Inclusive range of days to maturity at which the strategy would enter, used to
        keep the entry candidates and not every day of every contract's life.

    Returns
    -------
    pl.DataFrame
        Columns ``symbol``, ``strike``, ``expiration``, ``timestamp``, ``premium`` and one
        ``h<offset>`` per horizon holding ``|premium(t + offset) / premium(t) - 1|``, null
        where the contract carries no two-sided quote at that offset.
    """
    # The index counts every session the chains quote on, not every session they pair on:
    # built from the paired frame, a day on which no contract happened to pair would drop
    # out of the count and the offset would silently reach one session too far.
    sessions = chains.select("timestamp").unique().collect().sort("timestamp").with_row_index("s")
    chain = (
        chains.filter((pl.col("ask") > pl.col("bid")) & (pl.col("bid") > 0))
        .group_by([*CONTRACT, "timestamp"])
        .agg(
            pl.col("call_put").n_unique().alias("n_legs"),
            ((pl.col("ask") + pl.col("bid")) / 2).sum().alias("premium"),
            pl.col("days_to_maturity").first().alias("dte"),
        )
        .filter(pl.col("n_legs") == 2)
        .collect()
        .join(sessions, "timestamp")
    )

    moves = chain.filter(pl.col("dte").is_between(*entry_window)).select(
        [*CONTRACT, "timestamp", "s", "premium"]
    )
    for horizon in horizons:
        later = chain.select([*CONTRACT, pl.col("s") - horizon, pl.col("premium").alias("later")])
        moves = moves.join(later, on=[*CONTRACT, "s"], how="left").with_columns(
            (pl.col("later") / pl.col("premium") - 1).abs().alias(f"h{horizon}")
        )
        moves = moves.drop("later")
    return moves.drop("s")
