"""Daily mark-to-market backtest for the hold-to-expiry short-straddle strategy.

The production backtest path for ``ret_to_expiry`` in sp500_options. The
``backtest_runner.run_backtest`` dispatcher routes seven case studies through
``_run_engine`` (the ml4t-backtest ``Engine`` path), one through ``_run_vectorized``
(``us_firm_characteristics``, weights × y_true on rebalance dates), and this
single specialization through ``_run_htm_daily_mtm``. Neither generic path can
express an overlapping-cohort book with daily delta hedging: ``_run_vectorized``
has one P&L per rebalance date per symbol, and ``_run_engine`` has no notion
of an option-chain instrument with a fixed expiry and pre-computed delta path.

Strategy as declared in ``config/setup.yaml``:

- At the final available session of each Friday week, rank S&P 500 constituents
  by ``y_score`` (from the model trained on ``ret_to_expiry``) and pick the
  top-K = 20.
- For each of the 20, sell one ATM straddle (~30-day DTE, strike ≈ spot).
- Every subsequent trading day until expiry, delta-hedge using the underlying
  stock: if the straddle's |net delta| exceeds ``delta_threshold``, trade shares
  to push net delta back toward zero.
- Let the straddle cash-settle at expiry (no market exit → no exit bid-ask).
  A contract whose quote history ends before its expiration date never reaches
  that settlement and is bought back at its last quoted ask instead.
- Transaction costs applied: entry option bid-ask on both legs, per-contract
  option commission on both legs, per-hedge spread and per-share commission on
  every hedge rebalance, and an exit option bid-ask plus commission on the
  contracts that are bought back rather than settled.

Capital model: the entry cadence is weekly Friday but each cohort holds ~30 days,
so at steady state 5 cohorts are open simultaneously. Each cohort is sized at
``1/n_roll`` of capital so the book is fully invested.

Inputs:
- ``labels/contract_returns.parquet`` - one row per (feature_date, symbol, strike,
  expiration) with entry call/put mid, bid, ask and expiration date.
- Raw daily option chain at ``data/equities/market/sp500/options_straddles_raw/year=YYYY.parquet``
  for tracking same-contract quotes, deltas, underlying prices, and settlement
  inputs through expiration. This is the reader-distributed slim dataset, a
  lifecycle-preserving superset of every contract
  the ATM-band candidate filter (DTE 25-35, |delta| 0.35-0.65, converged IV,
  bid >= 0.01, relative spread <= 0.30) ever picks, with every daily observation
  from first listing through expiration. Backtest results are byte-identical
  to a run against the full unsliced AlgoSeek delivery.

Outputs: a daily portfolio-return Series and a metrics dict, returned to
``backtest_runner`` which handles registry writes identically to any other
backtest.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

CASE_STUDY_ID = "sp500_options"
TOP_K_DEFAULT = 20
N_ROLL_DEFAULT = 5  # 5 concurrent cohorts (weekly entry × ~30-day DTE)
DELTA_THRESHOLD_DEFAULT = 0.10  # setup.yaml hedging_protocol.delta_threshold

# Cost defaults from setup.yaml.costs. Per-leg entry spread uses the actual
# bid-ask from contract_returns; these are fallback values only. A forthcoming
# broker-selector API will let readers opt into a specific broker plan when
# they want a realistic per-trade floor; not enforced as a default here.
HEDGE_SPREAD_BPS_DEFAULT = 0.5  # 0.5 bps of hedge-trade notional
EQUITY_COMMISSION_PER_SHARE_DEFAULT = 0.0035  # USD / share, IBKR Pro Tiered top
OPTION_COMMISSION_PER_CONTRACT_DEFAULT = 1.00  # USD / contract incl. exchange/clearing/regulatory
OPTION_CONTRACT_MULTIPLIER_DEFAULT = 100


#: How far a borrowed expiration close may sit from the underlying at a contract's own last
#: marked session before the two are taken to be different quoting regimes. A split moves the
#: underlying by the split ratio while the listed strike does not follow, so two-for-one lands at
#: 0.50 and four-for-one at 0.25 - well outside this - while an ordinary move over the fortnight
#: these gaps typically run stays well inside it. The band is deliberately wide: its job is to
#: separate a corporate action from a market move, not to bound a market move.
SPLIT_GUARD_LOW: float = 0.7
SPLIT_GUARD_HIGH: float = 1.4


def option_accounting_parameters(signal: dict[str, Any]) -> dict[str, Any]:
    """Resolve every identity-bearing input used by the specialized option engine."""
    import yaml

    from utils.paths import get_case_study_dir

    setup_path = get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())
    costs = setup["costs"]["components"]
    exit_at_max_days = signal.get("exit_at_max_days")
    if exit_at_max_days is not None:
        exit_at_max_days = int(exit_at_max_days)
        if exit_at_max_days < 1:
            raise ValueError("exit_at_max_days must be positive")
    n_roll = int(signal.get("n_roll", N_ROLL_DEFAULT))
    if n_roll < 1:
        raise ValueError("n_roll must be positive")
    option_spread_fraction = float(signal.get("option_spread_fraction", 1.0))
    if not 0 <= option_spread_fraction <= 1:
        raise ValueError("option_spread_fraction must be between zero and one")
    delta_threshold = float(setup["hedging_protocol"]["delta_threshold"])
    if delta_threshold < 0:
        raise ValueError("delta_threshold cannot be negative")
    return {
        "schema_version": 1,
        "n_roll": n_roll,
        "portfolio_sizing": "equal_premium_within_cohort_and_fixed_cohort_fraction",
        "delta_hedge": True,
        "delta_threshold": delta_threshold,
        "hedge_spread_bps": float(costs["hedge_spread"]["estimate_bps_of_notional"]),
        "equity_commission_per_share": float(costs["commission"]["equity_per_share"]),
        "option_commission_per_contract": float(costs["commission"]["option_per_contract"]),
        "option_contract_multiplier": OPTION_CONTRACT_MULTIPLIER_DEFAULT,
        "option_spread_fraction": option_spread_fraction,
        "exit_at_max_days": exit_at_max_days,
        "entry_quote": "bid_for_short_call_and_put",
        "daily_mark": "paired_midpoint",
        # Two behaviours, so two names. A position held to expiry settles in cash at intrinsic;
        # one whose contract a corporate action removes from the chain is bought back against
        # the last mark the chain carried. Declaring only the first would put two behaviours
        # under one identity, which is what makes a backtest hash stop meaning anything.
        "settlement": (
            "cash_intrinsic_at_expiration"
            if exit_at_max_days is None
            else "ask_for_buy_to_close_at_holding_limit"
        ),
        "delisted_contract_exit": "ask_for_buy_to_close_on_first_unquoted_session_at_prior_mark",
        "hedge_liquidation": "final_observation_close",
    }


def option_data_paths() -> tuple[Path, Path]:
    """Resolve the canonical option artifact and raw lifecycle roots."""
    from utils import ML4T_DATA_PATH
    from utils.paths import get_case_study_dir

    labels_dir = get_case_study_dir(CASE_STUDY_ID) / "labels"
    raw_options_dir = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "options_straddles_raw"
    if not (labels_dir / "contract_returns.parquet").is_file():
        raise FileNotFoundError("sp500_options contract_returns.parquet is missing")
    if not raw_options_dir.is_dir():
        raise FileNotFoundError(f"raw option lifecycle directory is missing: {raw_options_dir}")
    return labels_dir, raw_options_dir


@lru_cache(maxsize=64)
def _file_sha256(path_string: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_file(path: Path) -> str:
    stat = path.stat()
    return _file_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def option_contract_source_identity(labels_dir: Path) -> str:
    """Return the exact contract-selection artifact identity."""
    contract_returns = labels_dir / "contract_returns.parquet"
    if not contract_returns.is_file():
        raise FileNotFoundError(f"option contract artifact is missing: {contract_returns}")
    return _digest_file(contract_returns)


def option_source_identity(labels_dir: Path, raw_options_dir: Path) -> dict[str, Any]:
    """Bind the specialized engine to exact contract and lifecycle files."""
    raw_files = sorted(raw_options_dir.glob("year=*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"raw option lifecycle directory is empty: {raw_options_dir}")
    return {
        "contract_returns": option_contract_source_identity(labels_dir),
        "raw_lifecycle": {path.name: _digest_file(path) for path in raw_files},
    }


def raw_lifecycle_identity(raw_options_dir: Path) -> str:
    """Digest the raw lifecycle files exactly as a backtest identity records them."""
    from case_studies.utils.registry import canonical_json, compute_hash

    raw_files = sorted(raw_options_dir.glob("year=*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"raw option lifecycle directory is empty: {raw_options_dir}")
    return compute_hash(canonical_json({path.name: _digest_file(path) for path in raw_files}))


_PRIMED_LIFECYCLE: dict[str, pl.DataFrame] = {}


def prime_option_lifecycle(cohorts: pl.DataFrame, raw_options_dir: Path) -> str:
    """Load the lifecycle once for a run of many requests, keyed by the files it was read from.

    A run that backtests hundreds of variants over one decision set should read the option
    chain once. The obvious way to arrange that is to load the frame and hand it to each
    request, and it is the wrong way: the request then has to trust a frame it did not
    produce. No check inside the request closes that. Comparing the frame against a source
    dictionary the same caller supplies checks the label on the box. Comparing it against the
    decision catches an unrelated frame but not a stale one, whose entry quotes are unchanged
    and whose later quotes, deltas and settlement rows are not. The frame's own digest cannot
    go into the identity either, because the identity is fixed when the request is planned,
    before any frame exists.

    So no frame is handed over. The loader reads the raw files, digests exactly those files,
    and stores what it built under that digest. A request looks up its own declared digest and
    gets either the frame those files produce or nothing at all, and nothing a caller passes
    can make it get anything else.
    """
    key = raw_lifecycle_identity(raw_options_dir)
    frame = _load_option_lifecycle(cohorts, raw_options_dir)
    if raw_lifecycle_identity(raw_options_dir) != key:
        raise RuntimeError("the raw option lifecycle files changed while they were being read")
    _PRIMED_LIFECYCLE.clear()
    _PRIMED_LIFECYCLE[key] = frame
    return key


def primed_option_lifecycle(identity: str | None) -> pl.DataFrame | None:
    """Return the primed lifecycle for these exact raw files, or None to load canonically."""
    if identity is None:
        return None
    return _PRIMED_LIFECYCLE.get(identity)


def _quote_is_ordered() -> pl.Expr:
    """An ask at or above its bid, everywhere the quote is used to mark the straddle.

    The expiration session is exempt. The straddle settles there in cash at the intrinsic
    value of its legs, computed from the underlying and the strike, and
    :func:`_load_option_lifecycle` overwrites both mids with that value - so the vendor's
    end-of-session bid and ask on that date are read by nothing. Requiring them to be
    ordered rejects a decision over a quote that cannot reach a single accounted number.

    This is the same exemption ``_defective_lifecycle_contracts`` already makes when it
    counts half-quoted sessions, and it is drawn for the same reason. It was missing here,
    so COST 295 expiring 2020-01-03 - whose call closed its last session bid 0.02 / ask 0.01,
    a crossed quote in one of 20.5 million rows - failed the paired-session count and halted
    the whole backtest at "1 of 1711 selected option decisions have no complete paired
    lifecycle". Every session whose prices the accounting does read is still checked.
    """
    return (pl.col("ask") >= pl.col("bid")) | (pl.col("date") == pl.col("expiration"))


def _price_end_of_session_quotes(chain: pl.DataFrame) -> pl.DataFrame:
    """Read a null bid beside a quoted ask as a bid of zero.

    The chain carries AlgoSeek's end-of-session quote: ``LastBidPrice``, ``LastAskPrice`` and
    ``LastMidPrice``. A null bid beside a positive ask is not a missing observation. It is a
    session that ended with an offer and no bid, which is what happens to an option nobody will
    pay for, and the vendor writes no mid because it has no two-sided quote to take the midpoint
    of. There are 9,140 such rows in the 20.5 million the validation window holds, and they
    cluster in the last sessions before expiry.

    Reading them as missing is not a neutral simplification for this strategy. A leg of a short
    straddle loses its bid exactly when the underlying has moved far enough from the strike to
    leave it worthless, which is the same move that makes the other leg expensive. A screen that
    discards those sessions discards the decisions that lost money, and every performance number
    computed afterwards would be measured on the survivors.

    The bid is therefore zero and the mid is half the ask, which is the same midpoint convention
    applied to the interval the quote actually spans.

    A leg the vendor did not quote at all is left alone. It is tempting to call it worthless
    when it is out of the money, but nothing here establishes that: the chain writes a delta of
    exactly 0.0 on every one of the 1,705 fully unquoted rows in the window, including ones that
    are not remotely worthless, so the delta is a filler rather than a measurement and cannot
    carry the argument. An unquoted leg makes its session one the position cannot be marked at,
    which :func:`_load_option_lifecycle` handles by not remarking the position that day, rather
    than by inventing a price for it. Where no quote ever returns, it books the exit on the
    first unquoted session against the mark the previous one carried.

    ``_source_quoted`` records what the vendor wrote, before this rule rewrites it. Whether a
    session can be marked at all is a fact about the source, and reading it back off the priced
    frame would make it a fact about this function instead.
    """
    unquoted = pl.col("bid").is_null() & pl.col("ask").is_null() & pl.col("mid_price").is_null()
    return chain.with_columns(
        _source_quoted=~unquoted,
        bid=pl.when(pl.col("bid").is_null() & pl.col("ask").is_not_null())
        .then(pl.lit(0.0))
        .otherwise(pl.col("bid")),
        mid_price=pl.when(
            pl.col("mid_price").is_null() & pl.col("bid").is_null() & pl.col("ask").is_not_null()
        )
        .then(pl.col("ask") / 2.0)
        .otherwise(pl.col("mid_price")),
    )


def _defective_lifecycle_contracts(
    contract_returns: pl.DataFrame,
    raw_options_dir: Path,
) -> pl.DataFrame:
    """Return the entry contracts whose paired lifecycle the chain contradicts itself about.

    Reads quotes from entry through expiration, so every input date is later than the decision
    it belongs to. Apply it to a selection to check the data the engine is about to account
    with; applying it to the candidate universe would decide the decision-time universe from
    data that does not exist yet.

    It reports a defect and never a shortfall. A contract the vendor stops quoting is not a
    defect and must not remove anything from a selection: doing so would condition the realized
    portfolio on a corporate action nobody knew about at entry, throw away the P&L the position
    earned before it, and hand the missing weight to the names that happened to survive.
    :func:`_load_option_lifecycle` books a liquidation on the first unquoted session instead,
    against the previous session's mark.
    """
    if contract_returns.is_empty():
        return contract_returns
    candidate_keys = ["feature_date", "symbol", "strike", "expiration", "entry_date"]
    if contract_returns.n_unique(candidate_keys) != contract_returns.height:
        raise ValueError("candidate option contracts are not unique by decision and entry keys")
    candidates = contract_returns.with_row_index("_candidate_id")
    entry_min = candidates["entry_date"].min()
    exp_max = candidates["expiration"].max()
    if not isinstance(entry_min, (date, datetime)) or not isinstance(exp_max, (date, datetime)):
        raise TypeError("candidate option entry and expiration columns must contain dates")

    contracts = candidates.select("symbol", "strike", "expiration").unique()
    symbols = candidates.get_column("symbol").unique().to_list()
    raw_parts = []
    calendars = []
    for year in range(entry_min.year, exp_max.year + 1):
        parquet_path = raw_options_dir / f"year={year}.parquet"
        if not parquet_path.exists():
            continue
        raw = pl.scan_parquet(parquet_path)
        in_window = pl.col("date").is_between(entry_min, exp_max, closed="both")
        raw_parts.append(
            raw.select(
                "date",
                "symbol",
                "strike",
                "expiration",
                "call_put",
                "mid_price",
                "bid",
                "ask",
                "delta",
                "underlying_price",
            )
            .filter(pl.col("symbol").is_in(symbols) & in_window)
            .join(contracts.lazy(), on=["symbol", "strike", "expiration"], how="semi")
            .collect()
        )
        calendars.append(raw.select("date").filter(in_window).unique().collect())
    if not raw_parts or not calendars:
        raise FileNotFoundError(f"No raw option data in {raw_options_dir}")

    raw_lookup = _price_end_of_session_quotes(pl.concat(raw_parts))
    calendar = pl.concat(calendars).unique().sort("date").with_row_index("_session")
    bounds = (
        candidates.join(
            calendar.rename({"date": "entry_date", "_session": "_entry_session"}),
            on="entry_date",
            how="left",
        )
        .join(
            calendar.rename({"date": "expiration", "_session": "_expiry_session"}),
            on="expiration",
            how="left",
        )
        .with_columns(
            (pl.col("_expiry_session") - pl.col("_entry_session") + 1).alias("_expected_dates")
        )
    )

    required_values = ["mid_price", "bid", "ask", "delta", "underlying_price"]
    valid = raw_lookup.filter(
        ~pl.any_horizontal(
            [
                pl.col(column).is_null() | ~pl.col(column).cast(pl.Float64).is_finite()
                for column in required_values
            ]
        )
        & (pl.col("mid_price") >= 0)
        & (pl.col("bid") >= 0)
        & _quote_is_ordered()
        & (pl.col("underlying_price") > 0)
    )
    join_keys = ["date", "symbol", "strike", "expiration"]
    calls = valid.filter(pl.col("call_put") == "C").select(
        *join_keys,
        pl.col("mid_price").alias("_raw_call_mid"),
        pl.col("underlying_price").alias("_call_underlying"),
    )
    puts = valid.filter(pl.col("call_put") == "P").select(
        *join_keys,
        pl.col("mid_price").alias("_raw_put_mid"),
        pl.col("underlying_price").alias("_put_underlying"),
    )
    paired = calls.join(puts, on=join_keys, how="inner").filter(
        (pl.col("_call_underlying") - pl.col("_put_underlying")).abs() <= 1e-10
    )

    contract_join = ["symbol", "strike", "expiration"]
    # How many of a session's two legs the vendor quoted. A session it quoted neither leg on is
    # one the position cannot be marked at, whether the legs are absent from the chain or
    # present with every price null. A session it quoted one leg on is a chain that holds the
    # contract and lost a leg, which is a defect. The distinction is drawn on what the source
    # carried, not on what survives validation, so that the pricing rules above cannot turn a
    # session nobody quoted into a session that looks half quoted.
    quoted_legs = (
        bounds.select("_candidate_id", "entry_date", *contract_join)
        .join(
            raw_lookup.select("date", *contract_join, "_source_quoted"),
            on=contract_join,
            how="left",
        )
        .filter(pl.col("date").is_between(pl.col("entry_date"), pl.col("expiration")))
        .group_by("_candidate_id", "expiration", "date")
        .agg(
            pl.len().alias("_legs_on_date"),
            pl.col("_source_quoted").sum().alias("_quoted_on_date"),
        )
        .group_by("_candidate_id")
        .agg(
            # The expiration session is left out of the one-leg count. The straddle settles
            # there in cash at the intrinsic value of its legs, computed from the underlying and
            # the strike, and the quoted mids are discarded rather than used - so a leg the
            # chain stops carrying on that session is the worthless one, and its absence says
            # nothing about whether the contract is intact.
            ((pl.col("_legs_on_date") == 1) & (pl.col("date") < pl.col("expiration")))
            .sum()
            .alias("_one_leg_dates"),
            (pl.col("_quoted_on_date") == 2).sum().alias("_fully_quoted_dates"),
        )
    )
    observed_pairs = (
        bounds.select(
            "_candidate_id",
            "entry_date",
            "entry_call_mid",
            "entry_put_mid",
            *contract_join,
        )
        .join(paired, on=contract_join, how="left")
        .filter(pl.col("date").is_between(pl.col("entry_date"), pl.col("expiration")))
        .group_by("_candidate_id")
        .agg(
            pl.len().alias("_observed_dates"),
            pl.col("date").max().alias("_last_paired_date"),
            (
                (pl.col("date") == pl.col("entry_date"))
                & ((pl.col("_raw_call_mid") - pl.col("entry_call_mid")).abs() <= 1e-10)
                & ((pl.col("_raw_put_mid") - pl.col("entry_put_mid")).abs() <= 1e-10)
            )
            .any()
            .alias("_entry_matches"),
        )
    )
    judged = (
        bounds.join(quoted_legs, on="_candidate_id", how="left")
        .join(observed_pairs, on="_candidate_id", how="left")
        .filter(pl.col("_expected_dates").is_not_null() & (pl.col("_expected_dates") > 0))
    )
    # Every session the vendor quoted both legs on must yield a valid pair. A session where
    # both legs are quoted but the pair fails validation - a nonfinite price, an ask below its
    # bid, the two legs disagreeing on the underlying - is a defect in the data and raises. So
    # is a session where the chain carries a row for one leg and no row at all for the other:
    # that is a chain that still holds the contract and lost a leg.
    #
    # A leg whose row is PRESENT with no bid and no ask is a different thing, and not a defect.
    # It is a leg nobody would trade that session, which is what happens to the far side of a
    # straddle once the underlying has moved away from the strike - KEYS 67.5 expiring
    # 2019-02-15 held a call at 9.45 and a put with a null bid, a null ask and a delta of zero
    # for one session, and quoted that put at 0.05 on the sessions either side. Marking the
    # straddle needs both legs, so such a session is one the position cannot be marked at, the
    # same as a session the chain quoted neither leg on. Neither rejects anything:
    # `_load_option_lifecycle` leaves the position unmarked where quotes resume, and liquidates
    # it on that session where the name has left the chain for good.
    defective = (
        (pl.col("_one_leg_dates").fill_null(0) > 0)
        | (pl.col("_observed_dates").fill_null(0) != pl.col("_fully_quoted_dates").fill_null(0))
        | ~pl.col("_entry_matches").fill_null(False)
    )
    judged = judged.with_columns(_defective=defective.fill_null(True))
    return (
        candidates.join(
            judged.filter(pl.col("_defective")).select("_candidate_id"),
            on="_candidate_id",
            how="semi",
        )
        .drop("_candidate_id")
        .sort("feature_date", "symbol")
    )


def _select_cohorts(
    predictions: pl.DataFrame,
    contract_returns: pl.DataFrame,
    *,
    method: str = "equal_weight_top_k",
    top_k: int = TOP_K_DEFAULT,
    percentile: float = 90.0,
    raw_options_dir: Path | None = None,
) -> pl.DataFrame:
    """For each weekly decision date, select symbols by the entry method and
    attach strike, expiration, entry quotes, and within-cohort weight.

    Supported methods mirror the vectorized signal dispatcher
    (``case_studies.utils.signals.build_target_weights_from_config``):

    - ``equal_weight_top_k``: top-K by y_score, equal weight within cohort.
    - ``score_weighted_top_k``: top-K by y_score, positive-score-proportional
      weight within cohort (negative scores clipped to 0; all-non-positive
      falls back to equal weight).
    - ``cross_sectional_percentile``: keep names above the p-th percentile of
      y_score per date, equal weight within cohort.

    Returns one row per (cohort, symbol) with a ``weight`` column summing to
    1.0 per ``timestamp`` (cohort).
    """
    from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

    pred = predictions.with_columns(pl.col("timestamp").cast(pl.Date).alias("date"))
    weekly_dates = resolve_rebalance_timestamps(
        pred["date"].unique().sort(),
        "weekly_friday",
    )
    pred = pred.filter(pl.col("date").is_in(weekly_dates.implode()))

    cr = contract_returns.select(
        pl.col("feature_date").alias("timestamp"),
        "symbol",
        "strike",
        "expiration",
        "entry_date",
        "entry_straddle_mid",
        "entry_call_mid",
        "entry_call_bid",
        "entry_call_ask",
        "entry_put_mid",
        "entry_put_bid",
        "entry_put_ask",
    )
    if cr.n_unique(["timestamp", "symbol"]) != cr.height:
        raise ValueError("contract-return rows are not unique by decision timestamp and symbol")
    # What the decision date knows: the chain names a contract for this symbol. `entry_date` is
    # the session after the decision and the entry quotes are read on it, so a missing quote
    # cannot be allowed to decide what was rankable - it would drop the higher-scored name and
    # hand its place to the runner-up with nothing said. Those columns are checked on the
    # selection instead, below.
    decision_time_keys = ("strike", "expiration")
    entry_values = (
        "entry_date",
        "entry_straddle_mid",
        "entry_call_mid",
        "entry_call_bid",
        "entry_call_ask",
        "entry_put_mid",
        "entry_put_bid",
        "entry_put_ask",
    )
    eligible = cr.filter(
        ~pl.any_horizontal([pl.col(column).is_null() for column in decision_time_keys])
    )
    pred = pred.join(
        eligible.select(pl.col("timestamp").alias("date"), "symbol"),
        on=["date", "symbol"],
        how="semi",
    )

    if method == "cross_sectional_percentile":
        q = max(0.0, min(100.0, float(percentile))) / 100.0
        ranked = (
            pred.with_columns(
                _threshold=pl.col("y_score").quantile(q).over("date"),
            )
            .filter(pl.col("y_score") >= pl.col("_threshold"))
            .select([pl.col("date").alias("timestamp"), "symbol", "y_score"])
        )
    elif method in ("equal_weight_top_k", "score_weighted_top_k"):
        ranked = (
            pred.with_columns(
                pl.col("y_score").rank("ordinal", descending=True).over("date").alias("_rank"),
            )
            .filter(pl.col("_rank") <= int(top_k))
            .select([pl.col("date").alias("timestamp"), "symbol", "y_score"])
        )
    else:
        raise ValueError(
            f"Unknown entry method for HTM: {method!r}. Supported: "
            f"equal_weight_top_k, score_weighted_top_k, cross_sectional_percentile."
        )

    # `pred` was already restricted to symbols `cr` carries a contract for, and `cr` holds one
    # row per (timestamp, symbol), so this join keeps every ranked row.
    cohorts = ranked.join(cr, on=["timestamp", "symbol"], how="inner")
    incomplete = cohorts.filter(
        pl.any_horizontal([pl.col(column).is_null() for column in entry_values])
    )
    if not incomplete.is_empty():
        raise ValueError(
            f"{incomplete.height} of {cohorts.height} selected option decisions have no complete "
            f"entry quote on the session after the decision "
            f"(first: {incomplete.select('timestamp', 'symbol').head(5).to_dicts()})"
        )
    if raw_options_dir is not None:
        # Completeness is checked on what was selected, never on what could be selected. The
        # screen reads quotes from entry through expiration, so applying it to the candidate
        # universe would let a data gap that opens after the decision date decide what the model
        # was allowed to rank on that date.
        defective = _defective_lifecycle_contracts(
            cohorts.rename({"timestamp": "feature_date"}), raw_options_dir
        )
        if not defective.is_empty():
            raise ValueError(
                f"{defective.height} of {cohorts.height} selected option decisions have no "
                f"complete paired lifecycle from entry through expiration "
                f"(first: {defective.select('feature_date', 'symbol').head(5).to_dicts()})"
            )
    if method == "score_weighted_top_k":
        cohorts = _score_proportional_weights(cohorts)
    else:
        cohorts = (
            cohorts.with_columns(
                _n=pl.len().over("timestamp"),
            )
            .with_columns(
                weight=1.0 / pl.col("_n").cast(pl.Float64),
            )
            .drop("_n")
        )

    return cohorts


def _load_option_lifecycle(
    cohorts: pl.DataFrame,
    raw_options_dir: Path,
) -> pl.DataFrame:
    """Load paired option quotes, deltas, and explicit cash settlement through expiry."""
    contracts = cohorts.select(["symbol", "strike", "expiration"]).unique()
    entry_min = cohorts["entry_date"].min()
    exp_max = cohorts["expiration"].max()
    if not isinstance(entry_min, (date, datetime)) or not isinstance(exp_max, (date, datetime)):
        raise TypeError("option cohort entry and expiration columns must contain dates")
    years = list(range(entry_min.year, exp_max.year + 1))
    parts = []
    calendars = []
    underlyings = []
    symbols = cohorts.get_column("symbol").unique().to_list()
    for year in years:
        parquet_path = raw_options_dir / f"year={year}.parquet"
        if not parquet_path.exists():
            continue
        raw = pl.scan_parquet(parquet_path)
        selected = (
            raw.select(
                [
                    "date",
                    "symbol",
                    "strike",
                    "expiration",
                    "call_put",
                    "mid_price",
                    "bid",
                    "ask",
                    "delta",
                    "underlying_price",
                ]
            )
            .filter(pl.col("symbol").is_in(symbols))
            .filter(pl.col("date").is_between(entry_min, exp_max, closed="both"))
            .join(contracts.lazy(), on=["symbol", "strike", "expiration"], how="semi")
            .collect()
        )
        calendar = (
            raw.select("date")
            .filter(pl.col("date").is_between(entry_min, exp_max, closed="both"))
            .unique()
            .collect()
        )
        # The underlying close per (date, symbol), read across every contract on the name
        # rather than only the selected ones. A liquidation session is by definition one where
        # the selected contract carries no quote, so its own rows cannot supply the stock price
        # the retained hedge is unwound at - but any other contract on the same underlying
        # still can, and the chain writes the same underlying_price on all of them.
        underlyings.append(
            raw.select(["date", "symbol", "underlying_price"])
            .filter(pl.col("symbol").is_in(symbols))
            .filter(pl.col("date").is_between(entry_min, exp_max, closed="both"))
            .filter(pl.col("underlying_price") > 0)
            .unique(subset=["date", "symbol"])
            .collect()
        )
        parts.append(selected)
        calendars.append(calendar)
    if not parts:
        raise FileNotFoundError(f"No raw option data in {raw_options_dir}")
    raw_lookup = _price_end_of_session_quotes(pl.concat(parts))
    key_columns = ["date", "symbol", "strike", "expiration", "call_put"]
    if raw_lookup.n_unique(key_columns) != raw_lookup.height:
        raise ValueError("raw option lifecycle contains duplicate contract-leg dates")
    # Validate the rows the vendor actually quoted, and only those.
    #
    # `_price_end_of_session_quotes` leaves a fully unquoted leg null on purpose: its bid, ask
    # and mid are all absent because the chain carried no quote, not because a quote went
    # missing. Checking those rows for nulls therefore rejects the very representation the rest
    # of this function is built to handle - an unmarked session, or the trigger for a
    # liquidation - and it rejects it before any of that logic runs. The chain carries 1,705
    # such rows in the validation window, so this is what a canonical run meets, not an edge
    # case. The tests missed it because they delete rows to simulate a vendor stopping, and an
    # absent row and a null row travel different paths from here.
    quoted = raw_lookup.filter(pl.col("_source_quoted"))
    required_values = ["mid_price", "bid", "ask", "delta", "underlying_price"]
    invalid = quoted.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_null() | ~pl.col(column).cast(pl.Float64).is_finite()
                for column in required_values
            ]
        )
        | (pl.col("mid_price") < 0)
        | (pl.col("bid") < 0)
        | ~_quote_is_ordered()
        | (pl.col("underlying_price") <= 0)
    )
    if not invalid.is_empty():
        raise ValueError(f"raw option lifecycle contains {invalid.height} invalid quote rows")
    calls = quoted.filter(pl.col("call_put") == "C").select(
        [
            "date",
            "symbol",
            "strike",
            "expiration",
            pl.col("mid_price").alias("call_mid"),
            pl.col("bid").alias("call_bid"),
            pl.col("ask").alias("call_ask"),
            pl.col("delta").alias("call_delta"),
            "underlying_price",
        ]
    )
    puts = quoted.filter(pl.col("call_put") == "P").select(
        [
            "date",
            "symbol",
            "strike",
            "expiration",
            pl.col("mid_price").alias("put_mid"),
            pl.col("bid").alias("put_bid"),
            pl.col("ask").alias("put_ask"),
            pl.col("delta").alias("put_delta"),
            pl.col("underlying_price").alias("put_underlying_price"),
        ]
    )
    paired = calls.join(puts, on=["date", "symbol", "strike", "expiration"], how="inner")
    if paired.filter(
        (pl.col("underlying_price") - pl.col("put_underlying_price")).abs() > 1e-10
    ).height:
        raise ValueError("call and put rows disagree on the underlying settlement price")

    # Where the position ends, and the two ways it can end are not the same event.
    #
    # It normally ends at expiration, where the straddle settles in cash at the intrinsic value
    # of its legs. Intrinsic needs only the underlying price and the strike, and either leg
    # carries the underlying, so the settlement row is built from whichever leg the chain still
    # quotes rather than from a pair. Expiration is a known date, so marking intrinsic there
    # reads no future - and it is the only place intrinsic is the correct mark.
    #
    # A contract a corporate action adjusts stops being quoted before then, and the adjusted
    # strike the position continues under is not in this chain. The holder cannot know which
    # session was the last one on the day it happens; they discover it on the first session no
    # quote arrives. So the exit is booked on that session, against the mark the previous
    # session carried, crossed to the side the trade executes on. Booking it on the last quoted
    # session instead dates the exit by hindsight, whether it is marked at the mid there (an
    # exit that crosses no spread) or at the ask there (a fill at a price the holder had no
    # reason to take while the contract was still quoting).
    #
    # After that session the proceeds are cash. `_aggregate_portfolio` divides by a fixed
    # `n_roll` rather than by a count of surviving positions, so a contract that stops appearing
    # contributes zero on its own weight for the rest of the holding period, which is what
    # holding cash is. Emitting rows to say so would state it twice, and reweighting the
    # survivors would say something else entirely.
    # A session where the chain carries a row for one leg and no row at all for the other,
    # before expiration, is a defect rather than an end. It looks downstream like a session the
    # chain quoted nothing on - neither yields a paired row - but it is a different claim about
    # the data: a chain that carries nothing has stopped carrying the contract, while a chain
    # that carries one leg's row still holds it and lost the other. Reading the second as a
    # termination would let a half-written session end a position early and book a liquidation
    # against it.
    #
    # The count is over rows the chain carries, not over rows it quoted. A leg present with a
    # null bid and a null ask is a leg nobody would trade that session, not a leg the chain
    # dropped, and the straddle simply goes unmarked there because marking it needs both legs.
    # Counting quoted legs instead halted a 1,923-decision run on KEYS 67.5 expiring
    # 2019-02-15, whose deep out-of-the-money put lost its market for a single session and was
    # quoted at 0.05 on the sessions either side.
    #
    # Expiration is exempt for the reason rule 2 gives: the settlement is intrinsic, so a leg
    # the chain drops there is the worthless one and its absence says nothing about the
    # contract.
    entry_window = cohorts.group_by(["symbol", "strike", "expiration"]).agg(
        pl.col("entry_date").min().alias("_held_from")
    )
    one_legged = (
        raw_lookup.group_by(["date", "symbol", "strike", "expiration"])
        .len(name="_legs")
        .join(entry_window, on=["symbol", "strike", "expiration"], how="inner")
        .filter(
            (pl.col("_legs") == 1)
            & (pl.col("date") >= pl.col("_held_from"))
            & (pl.col("date") < pl.col("expiration"))
        )
    )
    if not one_legged.is_empty():
        raise ValueError(
            f"raw option lifecycle is missing {one_legged.height} contract-leg dates "
            f"(first: {one_legged.select('date', 'symbol').head(5).to_dicts()})"
        )

    quote_columns = [
        "date",
        "symbol",
        "strike",
        "expiration",
        "call_mid",
        "call_bid",
        "call_ask",
        "call_delta",
        "put_mid",
        "put_bid",
        "put_ask",
        "put_delta",
        "underlying_price",
    ]
    marked = (
        paired.filter(pl.col("date") < pl.col("expiration"))
        .select(quote_columns)
        .with_columns(cash_settled=pl.lit(False), liquidated=pl.lit(False))
    )

    contract_keys = ["symbol", "strike", "expiration"]
    underlying_panel = (
        pl.concat(underlyings)
        .unique(subset=["date", "symbol"])
        .filter(pl.col("underlying_price").is_not_null())
        if underlyings
        else pl.DataFrame(
            schema={
                "date": paired.schema["date"],
                "symbol": pl.Utf8,
                "underlying_price": pl.Float64,
            }
        )
    )

    at_expiry = quoted.filter(pl.col("date") == pl.col("expiration"))
    expiry_underlying = at_expiry.group_by(contract_keys).agg(
        pl.col("underlying_price").min().alias("_underlying_low"),
        pl.col("underlying_price").max().alias("_underlying_high"),
    )
    if expiry_underlying.filter(
        (pl.col("_underlying_high") - pl.col("_underlying_low")).abs() > 1e-10
    ).height:
        raise ValueError("option legs disagree on the underlying settlement price at expiration")
    # How a position ends, and why the two endings divide where they do.
    #
    # A straddle held to expiry settles at the intrinsic value of its legs, which is a
    # function of the underlying close and the strike. Neither leg has to be quoted for that
    # number to exist, so the settlement price is taken from whichever contract on the name
    # carries the expiration close - the held one if it is still quoted, any other contract on
    # the same underlying otherwise. The alternative for an unquoted expiration session is the
    # liquidation below, which books the exit at the previous session's option mark: a mark
    # taken before the move the position was held through, at a price nobody could have
    # traded, and it is the worse of the two wherever a close exists.
    #
    # The two endings divide on whether the NAME is still in the chain, not on whether the
    # CONTRACT is still quoted:
    #
    #   - The name closed on the expiration session. The position is held to expiry and
    #     settles at intrinsic. This is the interior-gap reading - a session nobody quoted is
    #     a session the position cannot be marked at, not the end of it - extended to a gap
    #     that runs to expiry. A short straddle whose legs stop being quoted cannot be bought
    #     back, because there is no market to buy it back in. It is held.
    #   - The name is gone by the expiration session. Nothing observable prices the position
    #     there, so rule 3 of the straddle-lifecycle rule liquidates it on the first unquoted
    #     session against the last mark the chain carried. That mark is not executable either,
    #     which is why this branch is narrow: it is the delisting case, where the alternative
    #     is not a worse price but no price at all.
    #
    # Only contracts the chain actually priced are settled. A contract with no marked session
    # has no position to end, and a settlement row for it would put a holding in the lifecycle
    # that the entry check below would then have to reject.
    held = marked.select(contract_keys).unique()
    # Borrowing a sibling's expiration close assumes the strike and that close are quoted in the
    # same regime, and a stock split is exactly when they are not. ISRG split three-for-one on
    # 2021-10-05: the chain reprices the underlying from 970.75 to 330.23 that session, lists the
    # pre-split strikes one last time with no bid, ask or mid, and drops them the next day. A
    # position held at K=1080 then settles against 328.44 for an intrinsic of 751.56, against a
    # last real straddle mark of 110.68 four sessions earlier. It is not a settlement rule
    # choosing badly between two defensible prices - it is |S - K| computed across a corporate
    # action, and 180 of the 322 holdable contracts on this path show it.
    #
    # The contract's own last marked session is what says which regime its strike belongs to.
    # Where the settlement close is not a market move away from the underlying that session, the
    # borrowed close prices a different security and the position falls to the liquidation path
    # below, which marks it at the last price the chain actually carried for THIS contract.
    last_marked_underlying = (
        marked.sort("date")
        .group_by(contract_keys)
        .agg(pl.col("underlying_price").last().alias("_underlying_at_last_mark"))
    )
    borrowed = (
        held.join(expiry_underlying.select(contract_keys), on=contract_keys, how="anti")
        .join(
            underlying_panel.select(
                pl.col("date").alias("expiration"), "symbol", "underlying_price"
            ),
            on=["symbol", "expiration"],
            how="inner",
        )
        .join(last_marked_underlying, on=contract_keys, how="inner")
        .filter(
            (pl.col("_underlying_at_last_mark") > 0)
            & (pl.col("underlying_price") / pl.col("_underlying_at_last_mark")).is_between(
                SPLIT_GUARD_LOW, SPLIT_GUARD_HIGH
            )
        )
        .select(*contract_keys, "underlying_price")
    )
    settlement_underlying = pl.concat(
        [
            expiry_underlying.select(
                *contract_keys, pl.col("_underlying_low").alias("underlying_price")
            ),
            borrowed,
        ]
    )
    settled = (
        settlement_underlying.select(
            pl.col("expiration").alias("date"),
            "symbol",
            "strike",
            "expiration",
            "underlying_price",
        )
        .with_columns(
            call_mid=(pl.col("underlying_price") - pl.col("strike")).clip(lower_bound=0.0),
            put_mid=(pl.col("strike") - pl.col("underlying_price")).clip(lower_bound=0.0),
        )
        .with_columns(
            # Cash settlement is not a trade, so the settlement row carries no executable
            # sides to cross and no delta to hedge. Writing the intrinsic mid into the bid and
            # ask says that directly; leaving the quoted ones there would let a later cost rule
            # find a spread on a session where there is no market to pay one to.
            call_bid=pl.col("call_mid"),
            call_ask=pl.col("call_mid"),
            call_delta=pl.lit(0.0),
            put_bid=pl.col("put_mid"),
            put_ask=pl.col("put_mid"),
            put_delta=pl.lit(0.0),
            cash_settled=pl.lit(True),
            liquidated=pl.lit(False),
        )
        .select(*quote_columns, "cash_settled", "liquidated")
    )

    sessions = (
        pl.concat(calendars)
        .unique()
        .sort("date")
        .with_row_index("_session")
        .select("date", pl.col("_session").cast(pl.Int64))
    )
    last_marked = marked.group_by(contract_keys).agg(pl.col("date").max().alias("_last_marked"))
    liquidating = last_marked.join(
        settled.select(contract_keys), on=contract_keys, how="anti"
    ).join(
        sessions.select(pl.col("date").alias("_last_marked"), "_session"),
        on="_last_marked",
        how="inner",
    )
    liquidation_rows = (
        liquidating.with_columns(_session=pl.col("_session") + 1)
        .join(
            sessions.select(pl.col("date").alias("_liquidation_date"), "_session"),
            on="_session",
            how="inner",
        )
        .filter(pl.col("_liquidation_date") <= pl.col("expiration"))
        .join(
            marked,
            left_on=[*contract_keys, "_last_marked"],
            right_on=[*contract_keys, "date"],
            how="inner",
        )
        .join(
            underlying_panel.select(
                pl.col("date").alias("_liquidation_date"),
                "symbol",
                pl.col("underlying_price").alias("_liquidation_underlying"),
            ),
            on=["_liquidation_date", "symbol"],
            how="left",
        )
        .with_columns(
            date=pl.col("_liquidation_date"),
            # The stock is marked at its own close on this session, not at the previous one's.
            # The option cannot be: rule 3 exits it against the last mark the chain carried,
            # because there is no quote here to trade against. The stock has no such problem -
            # it is still trading, the retained hedge is unwound into it at that day's price,
            # and copying the previous close would silently zero the hedge's P&L on a session
            # the underlying may well have moved. Where the whole name has left the chain there
            # is no close to read, and the previous one is carried forward as a stated
            # assumption rather than a measurement.
            underlying_price=pl.coalesce(
                pl.col("_liquidation_underlying"), pl.col("underlying_price")
            ),
            # The straddle is bought back here, so it holds no delta into the next session.
            call_delta=pl.lit(0.0),
            put_delta=pl.lit(0.0),
            cash_settled=pl.lit(False),
            liquidated=pl.lit(True),
        )
        .select(*quote_columns, "cash_settled", "liquidated")
    )

    # Rule 4 of the straddle-lifecycle rule: a position that can be neither settled nor
    # liquidated is not dropped, it stops the run. Dropping it would report a portfolio that
    # silently excluded a position the strategy held.
    unendable = contracts.join(
        pl.concat([settled.select(contract_keys), liquidation_rows.select(contract_keys)]).unique(),
        on=contract_keys,
        how="anti",
    )
    if not unendable.is_empty():
        raise ValueError(
            f"{unendable.height} selected option contracts can be neither settled at expiration "
            f"nor liquidated against a prior mark "
            f"(first: {unendable.head(5).to_dicts()})"
        )

    lifecycle = (
        pl.concat([marked, settled, liquidation_rows])
        .with_columns(
            instr_mid=pl.col("call_mid") + pl.col("put_mid"),
            instr_delta=pl.col("call_delta") + pl.col("put_delta"),
        )
        .select(
            [
                "date",
                "symbol",
                "strike",
                "expiration",
                "instr_mid",
                "call_mid",
                "call_bid",
                "call_ask",
                "put_mid",
                "put_bid",
                "put_ask",
                "call_delta",
                "put_delta",
                "instr_delta",
                "underlying_price",
                "cash_settled",
                "liquidated",
            ]
        )
    )
    cohort_keys = cohorts.select(
        pl.col("timestamp").alias("cohort_feature_date"),
        "symbol",
        "strike",
        "expiration",
        "entry_date",
        "entry_call_mid",
        "entry_put_mid",
    ).unique()
    # No session-by-session completeness check between entry and the end of the position.
    #
    # It used to require a row on every calendar session in that window, which made a single
    # unquoted session mid-life reject a run that `_select_cohorts` had already accepted - the
    # two disagreed about the same contract, and the lifecycle was the one that raised. A
    # session the vendor quoted neither leg on is not a defect; it is a session the position
    # cannot be marked at. Where quotes resume, the position simply is not remarked that day
    # and the move is recognised when they do. Where they never resume, the session after the
    # last marked one is the liquidation booked above. What still has to hold is that the
    # position is marked on its entry date and that it ends, and both are checked below.
    endpoints = cohort_keys.join(
        lifecycle, on=["symbol", "strike", "expiration"], how="inner"
    ).filter(pl.col("date") == pl.col("entry_date"))
    if endpoints.height != cohort_keys.height:
        raise ValueError("selected option contracts do not all have an entry observation")
    if endpoints.filter(
        ((pl.col("call_mid") - pl.col("entry_call_mid")).abs() > 1e-10)
        | ((pl.col("put_mid") - pl.col("entry_put_mid")).abs() > 1e-10)
    ).height:
        raise ValueError("contract-return entry quotes disagree with the raw option lifecycle")
    expiry_keys = cohort_keys.select("cohort_feature_date", "symbol", "strike", "expiration").join(
        lifecycle.filter(pl.col("cash_settled") | pl.col("liquidated")).select(
            "symbol", "strike", "expiration"
        ),
        on=["symbol", "strike", "expiration"],
        how="semi",
    )
    if expiry_keys.height != cohort_keys.height:
        raise ValueError("selected option contracts do not all have an end-of-position observation")
    return lifecycle.sort(["symbol", "strike", "expiration", "date"])


_load_daily_contract_mids = _load_option_lifecycle


def _compute_cohort_daily_pnl(
    cohorts: pl.DataFrame,
    contract_mids: pl.DataFrame,
    *,
    delta_hedge: bool,
    hedge_spread_bps: float,
    equity_commission_per_share: float,
    option_commission_per_contract: float,
    option_contract_multiplier: int = OPTION_CONTRACT_MULTIPLIER_DEFAULT,
    delta_threshold: float,
    exit_at_max_days: int | None = None,
    option_spread_fraction: float = 1.0,
) -> pl.DataFrame:
    """Compute daily option and retained-hedge P&L for every selected contract."""
    if not 0 <= option_spread_fraction <= 1:
        raise ValueError("option_spread_fraction must be between zero and one")
    if delta_threshold < 0:
        raise ValueError("delta_threshold cannot be negative")
    if option_contract_multiplier < 1:
        raise ValueError("option_contract_multiplier must be positive")
    cohort_keys = cohorts.select(
        [
            pl.col("timestamp").alias("cohort_feature_date"),
            "symbol",
            "strike",
            "expiration",
            "entry_date",
            "entry_straddle_mid",
            "entry_call_mid",
            "entry_call_bid",
            "entry_put_mid",
            "entry_put_bid",
            "weight",
        ]
    )

    daily = (
        cohort_keys.join(contract_mids, on=["symbol", "strike", "expiration"], how="inner")
        .filter(pl.col("date") >= pl.col("entry_date"))
        .filter(pl.col("date") <= pl.col("expiration"))
        .sort(["cohort_feature_date", "symbol", "strike", "expiration", "date"])
    )
    partition = ["cohort_feature_date", "symbol", "strike", "expiration"]
    expected_cohorts = cohort_keys.n_unique(partition)
    if daily.n_unique(partition) != expected_cohorts:
        raise ValueError("option lifecycle silently dropped one or more selected cohorts")
    entry_coverage = daily.group_by(partition).agg(pl.col("date").min().alias("first_date"))
    if (
        entry_coverage.join(cohort_keys.select(*partition, "entry_date"), on=partition, how="left")
        .filter(pl.col("first_date") != pl.col("entry_date"))
        .height
    ):
        raise ValueError("option lifecycle does not begin on every selected entry date")
    if exit_at_max_days is None:
        settlement_coverage = daily.group_by(partition).agg(
            pl.col("date").max().alias("last_date"),
            pl.col("cash_settled").last().alias("cash_settled"),
            pl.col("liquidated").last().alias("liquidated"),
        )
        # Every position ends, and the check is that its last session says how. Expiration for
        # all but the ones a corporate action ended early, a liquidation for those. What is
        # refused is a position that simply stops with neither - a lifecycle that runs out of
        # rows is a position nothing valued, and rule 4 of the straddle-lifecycle rule says a
        # run that cannot be valued honestly does not publish.
        if settlement_coverage.filter(~(pl.col("cash_settled") | pl.col("liquidated"))).height:
            raise ValueError("option lifecycle does not end every selected contract")

    # Round-trip mode: cap the holding window at entry + exit_at_max_days
    # trading days (day 1 = entry, so we keep ranks 1 .. exit_at_max_days+1).
    if exit_at_max_days is not None:
        partition_pre = ["cohort_feature_date", "symbol", "strike", "expiration"]
        daily = daily.with_columns(
            _pre_rank=pl.col("date").rank("ordinal").over(partition_pre),
        )
        daily = daily.filter(pl.col("_pre_rank") <= int(exit_at_max_days) + 1).drop("_pre_rank")

    daily = daily.with_columns(
        dmid=pl.col("instr_mid") - pl.col("instr_mid").shift(1).over(partition),
        dS=pl.col("underlying_price") - pl.col("underlying_price").shift(1).over(partition),
        _day_rank=pl.col("date").rank("ordinal").over(partition),
        _last_rank=pl.len().over(partition),
    )
    daily = daily.with_columns(
        premium_pnl_norm=(-pl.col("dmid") / pl.col("entry_straddle_mid")).fill_null(0.0),
    )
    day_rank = daily.get_column("_day_rank").to_numpy()
    last_rank = daily.get_column("_last_rank").to_numpy()
    deltas = daily.get_column("instr_delta").cast(pl.Float64).to_numpy()
    underlying_moves = daily.get_column("dS").fill_null(0.0).cast(pl.Float64).to_numpy()
    entry_mids = daily.get_column("entry_straddle_mid").cast(pl.Float64).to_numpy()
    underlying_prices = daily.get_column("underlying_price").cast(pl.Float64).to_numpy()
    hedge_positions = np.zeros(daily.height, dtype=np.float64)
    hedge_trades = np.zeros(daily.height, dtype=np.float64)
    hedge_pnl = np.zeros(daily.height, dtype=np.float64)
    hedge_cost = np.zeros(daily.height, dtype=np.float64)
    retained_position = 0.0
    hedge_spread_rate = hedge_spread_bps / 10_000.0
    for index in range(daily.height):
        if day_rank[index] == 1:
            retained_position = 0.0
        hedge_pnl[index] = retained_position * underlying_moves[index] / entry_mids[index]
        if delta_hedge:
            final_observation = day_rank[index] == last_rank[index]
            if final_observation:
                next_position = 0.0
            elif abs(deltas[index] - retained_position) > delta_threshold:
                next_position = deltas[index]
            else:
                next_position = retained_position
            trade = next_position - retained_position
            hedge_trades[index] = trade
            retained_position = next_position
            hedge_cost[index] = (
                abs(trade)
                * (underlying_prices[index] * hedge_spread_rate + equity_commission_per_share)
                / entry_mids[index]
            )
        hedge_positions[index] = retained_position
    daily = daily.with_columns(
        pl.Series("hedge_position", hedge_positions),
        pl.Series("hedge_trade", hedge_trades),
        pl.Series("hedge_pnl_norm", hedge_pnl),
        pl.Series("hedge_cost_norm", hedge_cost),
    )

    daily = daily.with_columns(
        entry_cost_norm=(
            pl.when(pl.col("_day_rank") == 1)
            .then(
                (
                    option_spread_fraction
                    * (
                        (pl.col("entry_call_mid") - pl.col("entry_call_bid"))
                        + (pl.col("entry_put_mid") - pl.col("entry_put_bid"))
                    )
                    + (2.0 * option_commission_per_contract / option_contract_multiplier)
                )
                / pl.col("entry_straddle_mid")
            )
            .otherwise(0.0)
        ).fill_null(0.0),
    )

    # A liquidation pays the spread in every mode, not only the round-trip one. The liquidation
    # row sits on the first session the chain quoted nothing and carries the previous session's
    # quote, so this crosses that mark to the ask a buy-to-close executes at, plus commission -
    # the same price a round-trip exit pays, because it is the same trade. Only the expiration
    # session is free, and only because cash settlement is not a trade.
    if exit_at_max_days is None:
        daily = daily.with_columns(
            exit_cost_norm=(
                pl.when(pl.col("liquidated"))
                .then(
                    (
                        option_spread_fraction
                        * (
                            (pl.col("call_ask") - pl.col("call_mid"))
                            + (pl.col("put_ask") - pl.col("put_mid"))
                        )
                        + (2.0 * option_commission_per_contract / option_contract_multiplier)
                    )
                    / pl.col("entry_straddle_mid")
                )
                .otherwise(0.0)
            ).fill_null(0.0),
        )
    else:
        daily = daily.with_columns(
            exit_cost_norm=(
                # No exit bid-ask on the expiration session: the straddle cash-settles
                # there, so there is no market exit to cross a spread on. Charging one
                # would also read the expiration quote, which _quote_is_ordered exempts
                # from the crossed-quote check precisely because nothing reads it.
                pl.when(
                    (pl.col("_day_rank") == pl.col("_last_rank"))
                    & (pl.col("date") != pl.col("expiration"))
                )
                .then(
                    (
                        option_spread_fraction
                        * (
                            (pl.col("call_ask") - pl.col("call_mid"))
                            + (pl.col("put_ask") - pl.col("put_mid"))
                        )
                        + (2.0 * option_commission_per_contract / option_contract_multiplier)
                    )
                    / pl.col("entry_straddle_mid")
                )
                .otherwise(0.0)
            ).fill_null(0.0),
        )

    return daily.select(
        [
            "cohort_feature_date",
            "symbol",
            "strike",
            "expiration",
            "entry_date",
            "date",
            "weight",
            "cash_settled",
            "liquidated",
            "hedge_position",
            "hedge_trade",
            "premium_pnl_norm",
            "hedge_pnl_norm",
            "hedge_cost_norm",
            "entry_cost_norm",
            "exit_cost_norm",
        ]
    )


def _aggregate_portfolio(
    daily_cohort: pl.DataFrame,
    *,
    n_roll: int,
) -> pl.DataFrame:
    """Daily portfolio return = (1 / n_roll) × sum over open cohorts of
    (weighted cohort return after costs).

    Per-cohort per-day return = sum_i weight_i × (premium_pnl_norm +
    hedge_pnl_norm − hedge_cost_norm − entry_cost_norm)_i, where weight_i is
    the within-cohort weight set by the entry method (equal or
    score-proportional) and weights sum to 1.0 per cohort.

    Portfolio daily return = (1 / n_roll) × sum over open cohorts of
    per-cohort return.
    """
    has_exit_cost = "exit_cost_norm" in daily_cohort.columns
    if not has_exit_cost:
        daily_cohort = daily_cohort.with_columns(exit_cost_norm=pl.lit(0.0))
    daily_cohort = daily_cohort.with_columns(
        total_pnl_norm=(
            pl.col("premium_pnl_norm")
            + pl.col("hedge_pnl_norm")
            - pl.col("hedge_cost_norm")
            - pl.col("entry_cost_norm")
            - pl.col("exit_cost_norm")
        ),
        gross_pnl_norm=(pl.col("premium_pnl_norm") + pl.col("hedge_pnl_norm")),
    )
    cohort_daily = (
        daily_cohort.group_by(["cohort_feature_date", "date"])
        .agg(
            cohort_pnl_norm=(pl.col("total_pnl_norm") * pl.col("weight")).sum(),
            cohort_gross_norm=(pl.col("gross_pnl_norm") * pl.col("weight")).sum(),
            cohort_entry_cost=(pl.col("entry_cost_norm") * pl.col("weight")).sum(),
            cohort_hedge_cost=(pl.col("hedge_cost_norm") * pl.col("weight")).sum(),
            cohort_exit_cost=(pl.col("exit_cost_norm") * pl.col("weight")).sum(),
        )
        .sort(["date", "cohort_feature_date"])
    )
    port = (
        cohort_daily.group_by("date")
        .agg(
            portfolio_ret=(pl.col("cohort_pnl_norm").sum() / n_roll),
            gross_ret=(pl.col("cohort_gross_norm").sum() / n_roll),
            entry_cost_day=(pl.col("cohort_entry_cost").sum() / n_roll),
            hedge_cost_day=(pl.col("cohort_hedge_cost").sum() / n_roll),
            exit_cost_day=(pl.col("cohort_exit_cost").sum() / n_roll),
            n_open=pl.len(),
        )
        .sort("date")
    )
    return port


def _compute_metrics(port: pl.DataFrame) -> dict:
    """Sharpe / CAGR / MaxDD / win-rate from the daily portfolio return series."""
    ret = port["portfolio_ret"].to_numpy()
    if len(ret) < 2:
        return {"sharpe": float("nan")}
    mean = float(ret.mean())
    std = float(ret.std(ddof=1))
    sharpe = mean / std * math.sqrt(252) if std > 0 else float("nan")
    wealth = np.cumprod(1.0 + ret)
    final = float(wealth[-1])
    years = len(ret) / 252.0
    cagr = final ** (1 / years) - 1 if final > 0 and years > 0 else float("nan")
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    maxdd = float(drawdown.min())
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": maxdd,
        "volatility": std * math.sqrt(252),
        "total_return": final - 1.0,
        "n_periods": len(ret),
        "win_rate": float((ret > 0).mean()),
        "mean_daily_return": mean,
    }


def _load_underlying_price_panel(
    raw_options_dir: Path, symbols: list[str], date_min, date_max
) -> pl.DataFrame:
    """Build a deduplicated [date, symbol, close] panel of underlying-stock
    prices over [date_min, date_max] for the given symbols, by reading
    underlying_price out of the raw options chain (where the same
    underlying_price appears for every contract on the same (date, symbol)).
    """
    years = list(range(date_min.year, date_max.year + 1))
    parts = []
    for year in years:
        p = raw_options_dir / f"year={year}.parquet"
        if not p.exists():
            continue
        df = (
            pl.scan_parquet(p)
            .select(["date", "symbol", "underlying_price"])
            .filter(pl.col("symbol").is_in(symbols))
            .filter(pl.col("date") >= date_min)
            .filter(pl.col("date") <= date_max)
            .filter(pl.col("underlying_price") > 0)
            .unique(subset=["date", "symbol"])
            .collect()
        )
        parts.append(df)
    if not parts:
        return pl.DataFrame(schema={"timestamp": pl.Date, "symbol": pl.Utf8, "close": pl.Float64})
    panel = pl.concat(parts).rename({"date": "timestamp", "underlying_price": "close"})
    return panel.sort(["timestamp", "symbol"])


def _score_proportional_weights(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a positive-score-proportional `weight` within each entry date.

    A date whose scores are all non-positive has no proportion to size by, so
    it falls back to equal weight for that date alone. Shared by the entry
    method `score_weighted_top_k` and the allocator `score_weighted`, which
    have to agree on what score-proportional means.
    """
    return (
        frame.with_columns(
            _pos_score=pl.max_horizontal(pl.col("y_score"), pl.lit(0.0)),
        )
        .with_columns(
            _score_sum=pl.col("_pos_score").sum().over("timestamp"),
            _n=pl.len().over("timestamp"),
        )
        .with_columns(
            weight=pl.when(pl.col("_score_sum") > 0)
            .then(pl.col("_pos_score") / pl.col("_score_sum"))
            .otherwise(1.0 / pl.col("_n").cast(pl.Float64)),
        )
        .drop(["_pos_score", "_score_sum", "_n"])
    )


def _apply_cohort_allocator(
    cohorts: pl.DataFrame,
    raw_options_dir: Path,
    allocation_spec: dict,
    *,
    prediction_hash: str | None = None,
    label: str | None = None,
) -> pl.DataFrame:
    """Replace cohort `weight` column with allocator-derived weights.

    For each cohort entry date, treats the cohort's symbols as the asset
    selection and computes per-symbol weights: from the predicted score
    (`score_weighted`), from the width of each conformal prediction interval
    (`conformal_weighted`), or from a rolling window of underlying-stock
    returns (`inverse_vol`, `risk_parity`, `mvo_ledoit_wolf`, `hrp`).

    Returns `cohorts` unchanged only for `equal_weight`, which is the weighting
    `_select_cohorts` has already applied. Raises for `long_short=True`, an
    empty underlying price panel, an empty allocator output, or any
    unrecognized method - never falls through silently to equal-weight under
    an allocator-named run.
    """
    method = allocation_spec.get("method", "equal_weight")
    if method == "equal_weight":
        return cohorts

    if bool(allocation_spec.get("long_short", False)):
        # HTM cohort accounting models holding a basket to expiry - a single
        # cohort's weights must be non-negative and sum to 1. The downstream
        # `.abs()` + renorm would silently strip a short leg and persist the
        # run as `long_short=True` while realizing a long-only series. Refuse.
        raise ValueError(
            "HTM dispatch does not support long_short=True (would silently "
            "strip the short leg in cohort weight normalization). Use the "
            "vectorized path or set long_short=False."
        )

    max_weight = float(allocation_spec.get("max_weight", 0.0))

    # Build predictions df from cohorts (already filtered to top-K per Friday).
    # Sorted, because `unique` returns rows in whatever order the hash pass produced and
    # every allocator below turns that order into numbers: `score_weighted` sums floats
    # within a date, and the covariance allocators take their column order from
    # `preds["symbol"].unique()`. Both are stable to the last bits only if the row order
    # is. Without the sort the same request resolved twice digests differently, which is
    # exactly what the clean-process decision replay refuses.
    preds = (
        cohorts.select(
            pl.col("timestamp"),
            pl.col("symbol"),
            pl.col("y_score"),
        )
        .unique(subset=["timestamp", "symbol"])
        .sort("timestamp", "symbol")
    )

    max_cohort_size = cast(int, preds.group_by("timestamp").len()["len"].max())
    top_k_for_alloc = max(int(allocation_spec.get("top_k", 0)), max_cohort_size)

    if method == "score_weighted":
        # The signal stage may have selected the cohort under any entry method,
        # so the weighting has to be computed here rather than assumed. Reading
        # it off the signal is what let a `score_weighted` run realize equal
        # weight whenever the signal was `equal_weight_top_k`.
        weights = _score_proportional_weights(preds).select("timestamp", "symbol", "weight")
    elif method == "conformal_weighted":
        from case_studies.utils.allocation import compute_conformal_weights
        from case_studies.utils.conformal import (
            CALIBRATION_VERSION,
            DEFAULT_ALPHA,
            DEFAULT_MIN_CALIBRATION_N,
            load_conformal_widths,
        )

        if not prediction_hash:
            raise ValueError(
                "conformal_weighted sizes each position by the width of its conformal "
                "prediction interval, which is calibrated per prediction set; the caller "
                "must pass prediction_hash."
            )
        # The label goes with the request. Where no widths artifact exists yet,
        # `load_conformal_widths` generates one, and the generation needs an embargo - which
        # it looks up from the label when it is not given a number. Omitting the label here
        # made every first conformal request on a prediction set fail with "conformal
        # calibration needs the label horizon as an embargo" instead of calibrating, which is
        # what stopped 13_portfolio_management. The generic runner has always passed it.
        widths = load_conformal_widths(
            CASE_STUDY_ID,
            prediction_hash,
            alpha=float(allocation_spec.get("alpha", DEFAULT_ALPHA)),
            min_calibration_n=int(
                allocation_spec.get("min_calibration_n", DEFAULT_MIN_CALIBRATION_N)
            ),
            calibration_version=str(
                allocation_spec.get("calibration_version", CALIBRATION_VERSION)
            ),
            label=label or None,
        )
        if widths["timestamp"].dtype != preds["timestamp"].dtype:
            widths = widths.cast({"timestamp": preds["timestamp"].dtype})
        # An entry date earlier than the first calibration window has no
        # prior-only width, so it cannot be sized this way. Drop those cohorts
        # rather than filling them with the equal weights this allocator exists
        # to replace; the vectorized path drops the same decision dates.
        calibrated = widths.select("timestamp").unique()
        cohorts = cohorts.join(calibrated, on="timestamp", how="inner")
        preds = preds.join(calibrated, on="timestamp", how="inner")
        if preds.is_empty():
            raise RuntimeError(
                "conformal_weighted: no cohort entry date has prior-only calibrated "
                "widths. Refuse to silently fall back to equal-weight."
            )
        weights = compute_conformal_weights(
            preds,
            widths,
            top_k_for_alloc,
            long_short=False,
            floor_quantile=float(allocation_spec.get("floor_quantile", 0.01)),
        )
    else:
        from case_studies.utils.allocation import (
            compute_hrp_weights,
            compute_inverse_vol_weights,
            compute_mvo_weights,
            compute_risk_parity_weights,
        )

        # Independent defaults preserve the prior numerics: `vol_window=63`
        # (used by inverse_vol/risk_parity/hrp's rolling-stdev window) and
        # `lookback=126` (covariance window for MVO). The panel-backfill formula
        # below uses max(vol_window, lookback) so it covers the larger of the two.
        vol_window = int(allocation_spec.get("vol_window", 63))
        lookback = int(allocation_spec.get("lookback", 126))

        symbols = sorted(preds["symbol"].unique().to_list())
        date_min = preds["timestamp"].min()
        date_max = preds["timestamp"].max()
        # Backfill window for covariance estimation: 2x the longest lookback used.
        if not isinstance(date_min, (date, datetime)) or not isinstance(date_max, (date, datetime)):
            raise TypeError("option allocation timestamps must contain dates")
        # Pull ~1 year prior to earliest cohort to ensure rolling-window coverage.
        backfill_days = max(vol_window, lookback) * 2
        panel_start = date_min - timedelta(days=backfill_days + 30)

        prices = _load_underlying_price_panel(raw_options_dir, symbols, panel_start, date_max)
        if prices.is_empty():
            # Refuse to silently fall through to equal-weight under a non-EW
            # allocator label - that would persist a run record claiming
            # `method=hrp/mvo/etc.` while realizing equal_weight, breaking
            # downstream allocator-comparison analysis.
            raise RuntimeError(
                f"HTM allocator '{method}' requires an underlying price panel; "
                f"none available for {len(symbols)} symbols in [{panel_start}, "
                f"{date_max}]. Refuse to silently fall back to equal-weight."
            )

        # Dtype harmonization: predictions/cohorts use Date for timestamp; allocators
        # do internal pct_change which needs sortable timestamps - Date works.
        if prices["timestamp"].dtype != preds["timestamp"].dtype:
            prices = prices.cast({"timestamp": preds["timestamp"].dtype})

        if method == "inverse_vol":
            weights = compute_inverse_vol_weights(
                preds, prices, top_k_for_alloc, vol_window=vol_window, long_short=False
            )
        elif method == "risk_parity":
            weights = compute_risk_parity_weights(
                preds, prices, top_k_for_alloc, vol_window=vol_window, long_short=False
            )
        elif method in ("mvo", "mvo_ledoit_wolf"):
            weights = compute_mvo_weights(
                preds,
                prices,
                top_k_for_alloc,
                lookback=lookback,
                max_weight=max_weight if max_weight > 0 else 0.15,
                long_short=False,
            )
        elif method == "hrp":
            weights = compute_hrp_weights(
                preds, prices, top_k_for_alloc, vol_window=vol_window, long_short=False
            )
        else:
            raise ValueError(
                f"HTM dispatch: unsupported allocation method '{method}'. "
                "Supported: equal_weight, score_weighted, conformal_weighted, "
                "inverse_vol, risk_parity, mvo, mvo_ledoit_wolf, hrp."
            )

    if weights.is_empty():
        raise RuntimeError(
            f"HTM allocator '{method}' returned no weights for "
            f"{preds['timestamp'].n_unique()} cohort dates. Refuse to silently "
            "fall back to equal-weight."
        )

    # Apply max_weight cap (consistent with vectorized path). Both MVO variants
    # cap internally inside compute_mvo_weights - skip the external cap so we
    # don't double-cap (which would produce different tails vs. equal max_weight).
    if max_weight > 0 and method not in ("mvo", "mvo_ledoit_wolf"):
        from case_studies.utils.allocation import _cap_weights

        weights = _cap_weights(weights, max_weight)

    # Re-normalize to 1.0 per timestamp (allocators may produce signed weights;
    # cohort weights must be non-negative and sum to 1 within a cohort).
    weights = weights.with_columns(pl.col("weight").abs().alias("weight"))
    weights = weights.with_columns(
        (pl.col("weight") / pl.col("weight").sum().over("timestamp")).alias("weight")
    )

    # Replace cohorts.weight column with allocator weights via a left join.
    out = cohorts.drop("weight").join(
        weights.select(["timestamp", "symbol", "weight"]),
        on=["timestamp", "symbol"],
        how="left",
    )
    # Symbols missing from the allocator (insufficient history etc.) get
    # backfilled to equal-weight per cohort to avoid dropping selections.
    out = out.with_columns(
        pl.when(pl.col("weight").is_null())
        .then(1.0 / pl.len().over("timestamp").cast(pl.Float64))
        .otherwise(pl.col("weight"))
        .alias("weight")
    )
    # Final renormalization in case some null fills broke sum-to-1.
    out = out.with_columns(
        (pl.col("weight") / pl.col("weight").sum().over("timestamp")).alias("weight")
    )
    return out


def run_htm_daily_mtm(
    case_study: str,
    predictions: pl.DataFrame,
    labels_dir: Path,
    raw_options_dir: Path,
    *,
    method: str = "equal_weight_top_k",
    top_k: int = TOP_K_DEFAULT,
    percentile: float = 90.0,
    n_roll: int = N_ROLL_DEFAULT,
    delta_hedge: bool = True,
    delta_threshold: float = DELTA_THRESHOLD_DEFAULT,
    hedge_spread_bps: float = HEDGE_SPREAD_BPS_DEFAULT,
    equity_commission_per_share: float = EQUITY_COMMISSION_PER_SHARE_DEFAULT,
    option_commission_per_contract: float = OPTION_COMMISSION_PER_CONTRACT_DEFAULT,
    option_contract_multiplier: int = OPTION_CONTRACT_MULTIPLIER_DEFAULT,
    exit_at_max_days: int | None = None,
    allocation_spec: dict | None = None,
    decisions: pl.DataFrame | None = None,
    option_lifecycle: pl.DataFrame | None = None,
    option_spread_fraction: float = 1.0,
    prediction_hash: str | None = None,
    label: str | None = None,
) -> dict:
    """Run the HTM daily-MTM short-straddle backtest and return a result dict.

    ``method`` selects the entry-and-weighting scheme (mirrors the vectorized
    signal dispatcher): ``equal_weight_top_k`` picks top-K by y_score with
    equal weight; ``score_weighted_top_k`` picks top-K with positive-score-
    proportional weight; ``cross_sectional_percentile`` picks names above the
    p-th y_score percentile with equal weight.

    Returns a dict with:
      - ``daily_returns``: pl.DataFrame [date, portfolio_ret, gross_ret, n_open,
        entry_cost_day, hedge_cost_day]
      - ``metrics``: dict with sharpe, cagr, max_drawdown, volatility, total_return,
        n_periods, win_rate, mean_daily_return
    """
    assert case_study == CASE_STUDY_ID, f"htm_daily_mtm is sp500_options only, got {case_study!r}"
    if n_roll < 1:
        raise ValueError("n_roll must be positive")

    if decisions is None:
        contract_returns = pl.read_parquet(labels_dir / "contract_returns.parquet")
        cohorts = _select_cohorts(
            predictions,
            contract_returns,
            method=method,
            top_k=top_k,
            percentile=percentile,
            raw_options_dir=raw_options_dir,
        )
    else:
        required = {
            "timestamp",
            "symbol",
            "strike",
            "expiration",
            "entry_date",
            "entry_straddle_mid",
            "entry_call_mid",
            "entry_call_bid",
            "entry_call_ask",
            "entry_put_mid",
            "entry_put_bid",
            "entry_put_ask",
            "weight",
        }
        missing = required - set(decisions.columns)
        if missing:
            raise ValueError(f"typed option decisions are missing columns: {sorted(missing)}")
        cohorts = decisions
    if cohorts.is_empty():
        raise ValueError("No cohorts selected from predictions")

    # Apply allocator (Ch17). When `allocation_spec.method` is one of
    # inverse_vol/risk_parity/mvo_ledoit_wolf/hrp, override the per-symbol
    # weight set by `_select_cohorts` (equal or score-weighted) with weights
    # computed from a rolling underlying-stock returns window. Each cohort
    # entry date is a separate allocation problem — we delegate to the
    # standard `case_studies.utils.allocation` helpers, which already
    # implement covariance-shrinkage MVO, hierarchical risk parity, etc.
    if allocation_spec and decisions is None:
        cohorts = _apply_cohort_allocator(
            cohorts,
            raw_options_dir,
            allocation_spec,
            prediction_hash=prediction_hash,
            label=label,
        )

    contract_mids = (
        option_lifecycle
        if option_lifecycle is not None
        else _load_option_lifecycle(cohorts, raw_options_dir)
    )

    daily_cohort = _compute_cohort_daily_pnl(
        cohorts,
        contract_mids,
        delta_hedge=delta_hedge,
        hedge_spread_bps=hedge_spread_bps,
        equity_commission_per_share=equity_commission_per_share,
        option_commission_per_contract=option_commission_per_contract,
        option_contract_multiplier=option_contract_multiplier,
        delta_threshold=delta_threshold,
        exit_at_max_days=exit_at_max_days,
        option_spread_fraction=option_spread_fraction,
    )

    port = _aggregate_portfolio(daily_cohort, n_roll=n_roll)
    metrics = _compute_metrics(port)

    return {
        "daily_returns": port,
        "metrics": metrics,
    }
