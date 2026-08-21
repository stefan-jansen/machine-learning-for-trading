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
- Transaction costs applied: entry option bid-ask on both legs, per-contract
  option commission on both legs (entry only — HTM has no exit), per-hedge
  spread and per-share commission on every hedge rebalance.

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
        "settlement": (
            "cash_intrinsic_at_expiration"
            if exit_at_max_days is None
            else "ask_for_buy_to_close_at_holding_limit"
        ),
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


def _select_cohorts(
    predictions: pl.DataFrame,
    contract_returns: pl.DataFrame,
    *,
    method: str = "equal_weight_top_k",
    top_k: int = TOP_K_DEFAULT,
    percentile: float = 90.0,
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
    invalid_contracts = cr.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_null()
                for column in (
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
            ]
        )
    )
    if not invalid_contracts.is_empty():
        cr = cr.join(
            invalid_contracts.select("timestamp", "symbol"),
            on=["timestamp", "symbol"],
            how="anti",
        )
    cohorts = ranked.join(cr, on=["timestamp", "symbol"], how="inner")
    missing = ranked.join(
        cohorts.select("timestamp", "symbol"),
        on=["timestamp", "symbol"],
        how="anti",
    )
    if not missing.is_empty():
        raise ValueError(
            f"selected option decisions have no complete entry contract: {missing.height} rows"
        )

    if method == "score_weighted_top_k":
        cohorts = (
            cohorts.with_columns(
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
        parts.append(selected)
        calendars.append(calendar)
    if not parts:
        raise FileNotFoundError(f"No raw option data in {raw_options_dir}")
    raw_lookup = pl.concat(parts)
    key_columns = ["date", "symbol", "strike", "expiration", "call_put"]
    if raw_lookup.n_unique(key_columns) != raw_lookup.height:
        raise ValueError("raw option lifecycle contains duplicate contract-leg dates")
    required_values = ["mid_price", "bid", "ask", "delta", "underlying_price"]
    invalid = raw_lookup.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_null() | ~pl.col(column).cast(pl.Float64).is_finite()
                for column in required_values
            ]
        )
        | (pl.col("mid_price") < 0)
        | (pl.col("bid") < 0)
        | (pl.col("ask") < pl.col("bid"))
        | (pl.col("underlying_price") <= 0)
    )
    if not invalid.is_empty():
        raise ValueError(f"raw option lifecycle contains {invalid.height} invalid quote rows")
    calls = raw_lookup.filter(pl.col("call_put") == "C").select(
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
    puts = raw_lookup.filter(pl.col("call_put") == "P").select(
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
    lifecycle = (
        calls.join(puts, on=["date", "symbol", "strike", "expiration"], how="inner")
        .with_columns(
            cash_settled=pl.col("date") == pl.col("expiration"),
            instr_delta=pl.col("call_delta") + pl.col("put_delta"),
        )
        .with_columns(
            call_mid=pl.when(pl.col("cash_settled"))
            .then((pl.col("underlying_price") - pl.col("strike")).clip(lower_bound=0.0))
            .otherwise(pl.col("call_mid")),
            put_mid=pl.when(pl.col("cash_settled"))
            .then((pl.col("strike") - pl.col("underlying_price")).clip(lower_bound=0.0))
            .otherwise(pl.col("put_mid")),
        )
        .with_columns(instr_mid=pl.col("call_mid") + pl.col("put_mid"))
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
                "put_underlying_price",
                "cash_settled",
            ]
        )
    )
    if lifecycle.filter(
        (pl.col("underlying_price") - pl.col("put_underlying_price")).abs() > 1e-10
    ).height:
        raise ValueError("call and put rows disagree on the underlying settlement price")

    cohort_keys = cohorts.select(
        pl.col("timestamp").alias("cohort_feature_date"),
        "symbol",
        "strike",
        "expiration",
        "entry_date",
        "entry_call_mid",
        "entry_put_mid",
    ).unique()
    calendar = pl.concat(calendars).unique()
    expected = (
        cohort_keys.join(calendar, how="cross")
        .filter(pl.col("date").is_between(pl.col("entry_date"), pl.col("expiration")))
        .select(
            "cohort_feature_date",
            "symbol",
            "strike",
            "expiration",
            "entry_date",
            "date",
        )
    )
    observed = (
        cohort_keys.join(lifecycle, on=["symbol", "strike", "expiration"], how="inner")
        .filter(pl.col("date").is_between(pl.col("entry_date"), pl.col("expiration")))
        .select(expected.columns)
    )
    missing = expected.join(observed, on=expected.columns, how="anti")
    if not missing.is_empty():
        raise ValueError(f"selected option contracts are missing {missing.height} lifecycle dates")
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
        lifecycle.filter(pl.col("cash_settled")).select("symbol", "strike", "expiration"),
        on=["symbol", "strike", "expiration"],
        how="semi",
    )
    if expiry_keys.height != cohort_keys.height:
        raise ValueError("selected option contracts do not all have cash-settlement inputs")
    return lifecycle.drop("put_underlying_price").sort(["symbol", "strike", "expiration", "date"])


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
        )
        if settlement_coverage.filter(
            (pl.col("last_date") != pl.col("expiration")) | ~pl.col("cash_settled")
        ).height:
            raise ValueError("option lifecycle does not cash-settle every selected contract")

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

    if exit_at_max_days is not None:
        daily = daily.with_columns(
            exit_cost_norm=(
                pl.when(pl.col("_day_rank") == pl.col("_last_rank"))
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
        daily = daily.with_columns(exit_cost_norm=pl.lit(0.0))

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


def _apply_cohort_allocator(
    cohorts: pl.DataFrame,
    raw_options_dir: Path,
    allocation_spec: dict,
) -> pl.DataFrame:
    """Replace cohort `weight` column with allocator-derived weights.

    For each cohort entry date, treats the cohort's symbols as the asset
    selection and computes per-symbol weights from a rolling window of
    underlying-stock returns using one of the standard allocators.

    Returns `cohorts` unchanged for `equal_weight` / `score_weighted` (no-op
    methods). Raises for `long_short=True`, an empty underlying price panel,
    an empty allocator output, or any unrecognized method — never falls
    through silently to equal-weight under an allocator-named run.
    """
    method = allocation_spec.get("method", "equal_weight")
    if method in ("equal_weight", "score_weighted"):
        return cohorts

    from case_studies.utils.allocation import (
        compute_hrp_weights,
        compute_inverse_vol_weights,
        compute_mvo_weights,
        compute_risk_parity_weights,
    )

    if bool(allocation_spec.get("long_short", False)):
        # HTM cohort accounting models holding a basket to expiry — a single
        # cohort's weights must be non-negative and sum to 1. The downstream
        # `.abs()` + renorm would silently strip a short leg and persist the
        # run as `long_short=True` while realizing a long-only series. Refuse.
        raise ValueError(
            "HTM dispatch does not support long_short=True (would silently "
            "strip the short leg in cohort weight normalization). Use the "
            "vectorized path or set long_short=False."
        )

    # Independent defaults preserve the prior numerics: `vol_window=63`
    # (used by inverse_vol/risk_parity/hrp's rolling-stdev window) and
    # `lookback=126` (covariance window for MVO). The panel-backfill formula
    # below uses max(vol_window, lookback) so it covers the larger of the two.
    vol_window = int(allocation_spec.get("vol_window", 63))
    lookback = int(allocation_spec.get("lookback", 126))
    max_weight = float(allocation_spec.get("max_weight", 0.0))

    # Build predictions df from cohorts (already filtered to top-K per Friday).
    preds = cohorts.select(
        pl.col("timestamp"),
        pl.col("symbol"),
        pl.col("y_score"),
    ).unique(subset=["timestamp", "symbol"])

    symbols = preds["symbol"].unique().to_list()
    date_min = preds["timestamp"].min()
    date_max = preds["timestamp"].max()
    # Backfill window for covariance estimation: 2× the longest lookback used.
    if not isinstance(date_min, (date, datetime)) or not isinstance(date_max, (date, datetime)):
        raise TypeError("option allocation timestamps must contain dates")
    # Pull ~1 year prior to earliest cohort to ensure rolling-window coverage.
    backfill_days = max(vol_window, lookback) * 2
    panel_start = date_min - timedelta(days=backfill_days + 30)

    prices = _load_underlying_price_panel(raw_options_dir, symbols, panel_start, date_max)
    if prices.is_empty():
        # Refuse to silently fall through to equal-weight under a non-EW
        # allocator label — that would persist a run record claiming
        # `method=hrp/mvo/etc.` while realizing equal_weight, breaking
        # downstream allocator-comparison analysis.
        raise RuntimeError(
            f"HTM allocator '{method}' requires an underlying price panel; "
            f"none available for {len(symbols)} symbols in [{panel_start}, "
            f"{date_max}]. Refuse to silently fall back to equal-weight."
        )

    # Dtype harmonization: predictions/cohorts use Date for timestamp; allocators
    # do internal pct_change which needs sortable timestamps — Date works.
    if prices["timestamp"].dtype != preds["timestamp"].dtype:
        prices = prices.cast({"timestamp": preds["timestamp"].dtype})

    max_cohort_size = cast(int, preds.group_by("timestamp").len()["len"].max())
    top_k_for_alloc = max(int(allocation_spec.get("top_k", 0)), max_cohort_size)

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
            "Supported: equal_weight, score_weighted, inverse_vol, risk_parity, "
            "mvo, mvo_ledoit_wolf, hrp."
        )

    if weights.is_empty():
        raise RuntimeError(
            f"HTM allocator '{method}' returned no weights for "
            f"{preds['timestamp'].n_unique()} cohort dates. Refuse to silently "
            "fall back to equal-weight."
        )

    # Apply max_weight cap (consistent with vectorized path). Both MVO variants
    # cap internally inside compute_mvo_weights — skip the external cap so we
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
        cohorts = _apply_cohort_allocator(cohorts, raw_options_dir, allocation_spec)

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
