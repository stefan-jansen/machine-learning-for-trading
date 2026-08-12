"""Build same-contract label artifacts for the S&P 500 options case study."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from data import load_sp500_options_straddles, load_sp500_options_straddles_raw
from utils.paths import get_case_study_dir

HORIZONS = (5, 10)
MAX_HOLDING = max(HORIZONS)
JOIN_KEYS = ["symbol", "strike", "expiration"]


def ensure_label_artifacts(
    *,
    case_study_id: str = "sp500_options",
    max_symbols: int = 0,
    start_date: str | None = None,
    force_rebuild: bool = False,
    save_prices: bool = False,
) -> dict[str, Path]:
    """Ensure same-contract option artifacts exist for label construction."""
    case_dir = get_case_study_dir(case_study_id)
    labels_dir = case_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    contract_returns_path = labels_dir / "contract_returns.parquet"
    hedge_path_path = labels_dir / "hedge_path.parquet"
    prices_path = labels_dir / "prices.parquet"

    required = [contract_returns_path, hedge_path_path]
    # A cached artifact was built over whatever window the run that wrote it saw, so the
    # scope is recorded beside it and the cache is only reused for the same request.
    # Without that, a narrower run returns the full panel and leaves the reduction
    # silently unapplied, and the reduced files a narrower run writes are then accepted
    # by the next default run as if they covered everything.
    scope_path = labels_dir / "contract_returns.scope.json"
    scope = {"max_symbols": max_symbols, "start_date": start_date}
    cached_scope = json.loads(scope_path.read_text()) if scope_path.exists() else None
    if not force_rebuild and cached_scope == scope and all(path.exists() for path in required):
        return {
            "contract_returns": contract_returns_path,
            "hedge_path": hedge_path_path,
            "prices": prices_path,
        }

    straddles = load_sp500_options_straddles()
    if start_date is not None:
        straddles = straddles.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())

    # Horizons are counted in market sessions, so the calendar comes from the whole panel
    # and `max_symbols` thins only the entries it is applied to. Deriving the offsets from
    # a symbol subset instead would drop every session none of those symbols was quoted
    # on, and the exit dates would then be that many sessions further out than declared.
    trading_dates = straddles["timestamp"].unique().sort().to_list()
    entry_rows = straddles
    if max_symbols > 0:
        top_syms = (
            straddles.group_by("symbol")
            .len()
            .sort("len", descending=True)
            .head(max_symbols)["symbol"]
            .to_list()
        )
        entry_rows = straddles.filter(pl.col("symbol").is_in(top_syms))

    # A signal date qualifies once the panel has a session to enter on; how far past
    # that the panel has to run is a property of each horizon, so every offset column
    # runs off the end of the panel as a null rather than shortening the frame. Sizing
    # one frame for the longest horizon instead trims the five-session labels by the
    # ten-session one, and drops signal dates from the hold-to-expiry label, which
    # needs only an entry price and an expiration.
    def _shifted(step: int) -> list[object]:
        return trading_dates[step:] + [None] * step

    offset_data = {"feature_date": trading_dates, "entry_date": _shifted(1)}
    for horizon in HORIZONS:
        offset_data[f"exit_{horizon}d_date"] = _shifted(1 + horizon)
    for day in range(MAX_HOLDING + 1):
        offset_data[f"path_date_{day}"] = _shifted(1 + day)

    date_offsets = pl.DataFrame(offset_data).drop_nulls("entry_date")
    entries = (
        entry_rows.select(["timestamp", "symbol", "strike", "expiration"])
        .join(date_offsets.rename({"feature_date": "timestamp"}), on="timestamp", how="inner")
        .rename({"timestamp": "feature_date"})
    )

    contracts = entries.select("symbol", "strike", "expiration").unique()
    raw_lookup = (
        load_sp500_options_straddles_raw(lazy=True)
        .join(contracts.lazy(), on=JOIN_KEYS, how="semi")
        .filter(pl.col("bid") >= 0.01)
        .select(
            [
                pl.col("timestamp").alias("date"),
                "symbol",
                "strike",
                "expiration",
                "call_put",
                "mid_price",
                "bid",
                "ask",
                "delta",
                "underlying_price",
                "days_to_maturity",
            ]
        )
        .collect()
    )

    raw_calls = raw_lookup.filter(pl.col("call_put") == "C")
    raw_puts = raw_lookup.filter(pl.col("call_put") == "P")
    result = entries

    result = result.join(
        _build_price_lookup(raw_calls, "entry_date", "entry_call"),
        on=["entry_date"] + JOIN_KEYS,
        how="left",
    )
    result = result.join(
        _build_price_lookup(raw_puts, "entry_date", "entry_put"),
        on=["entry_date"] + JOIN_KEYS,
        how="left",
    )

    for horizon in HORIZONS:
        date_col = f"exit_{horizon}d_date"
        result = result.join(
            _build_price_lookup(raw_calls, date_col, f"exit_call_{horizon}d"),
            on=[date_col] + JOIN_KEYS,
            how="left",
        )
        result = result.join(
            _build_price_lookup(raw_puts, date_col, f"exit_put_{horizon}d"),
            on=[date_col] + JOIN_KEYS,
            how="left",
        )

    result = result.with_columns(
        (pl.col("entry_call_mid") + pl.col("entry_put_mid")).alias("entry_straddle_mid"),
    )
    for horizon in HORIZONS:
        exit_straddle = f"exit_straddle_mid_{horizon}d"
        result = result.with_columns(
            (pl.col(f"exit_call_{horizon}d_mid") + pl.col(f"exit_put_{horizon}d_mid")).alias(
                exit_straddle
            ),
            pl.col(f"exit_call_{horizon}d_mid").is_not_null().alias(f"exit_found_{horizon}d"),
        )
        result = result.with_columns(
            (
                (pl.col("entry_straddle_mid") - pl.col(exit_straddle))
                / pl.col("entry_straddle_mid")
            ).alias(f"fwd_ret_{horizon}d"),
        )

    hedge_needs = (
        entries.select(
            ["feature_date", "symbol", "strike", "expiration"]
            + [f"path_date_{d}" for d in range(MAX_HOLDING + 1)]
        )
        .unpivot(
            [f"path_date_{d}" for d in range(MAX_HOLDING + 1)],
            index=["feature_date", "symbol", "strike", "expiration"],
            variable_name="holding_day_str",
            value_name="holding_date",
        )
        .with_columns(
            pl.col("holding_day_str").str.extract(r"(\d+)").cast(pl.Int32).alias("holding_day")
        )
        .drop("holding_day_str")
    )

    hedge_call = raw_calls.select(
        [
            pl.col("date").alias("holding_date"),
            "symbol",
            "strike",
            "expiration",
            pl.col("delta").alias("call_delta"),
        ]
    )
    hedge_put = raw_puts.select(
        [
            pl.col("date").alias("holding_date"),
            "symbol",
            "strike",
            "expiration",
            pl.col("delta").alias("put_delta"),
        ]
    )
    underlying_prices = raw_calls.select(
        [pl.col("date").alias("holding_date"), "symbol", "strike", "expiration", "underlying_price"]
    ).unique(subset=["holding_date", "symbol", "strike", "expiration"])

    hedge_path = (
        hedge_needs.join(hedge_call, on=["holding_date"] + JOIN_KEYS, how="left")
        .join(hedge_put, on=["holding_date"] + JOIN_KEYS, how="left")
        .join(underlying_prices, on=["holding_date"] + JOIN_KEYS, how="left")
        .with_columns((pl.col("call_delta") + pl.col("put_delta")).alias("instr_delta"))
    )

    return_cols = (
        ["feature_date", "symbol", "strike", "expiration", "entry_date"]
        + [f"exit_{horizon}d_date" for horizon in HORIZONS]
        + ["entry_call_mid", "entry_put_mid", "entry_straddle_mid"]
        + ["entry_call_bid", "entry_call_ask", "entry_put_bid", "entry_put_ask"]
    )
    for horizon in HORIZONS:
        return_cols += [
            f"exit_call_{horizon}d_mid",
            f"exit_put_{horizon}d_mid",
            f"exit_straddle_mid_{horizon}d",
            f"fwd_ret_{horizon}d",
            f"exit_found_{horizon}d",
            f"exit_call_{horizon}d_bid",
            f"exit_call_{horizon}d_ask",
            f"exit_put_{horizon}d_bid",
            f"exit_put_{horizon}d_ask",
        ]
    contract_returns = result.select(return_cols)
    contract_returns.write_parquet(contract_returns_path)

    hedge_path_out = hedge_path.select(
        [
            "feature_date",
            "symbol",
            "strike",
            "expiration",
            "holding_day",
            "holding_date",
            "call_delta",
            "put_delta",
            "instr_delta",
            "underlying_price",
        ]
    ).sort(["symbol", "feature_date", "holding_day"])
    hedge_path_out.write_parquet(hedge_path_path)

    scope_path.write_text(json.dumps(scope, sort_keys=True) + "\n")

    if save_prices:
        straddles.write_parquet(prices_path)

    return {
        "contract_returns": contract_returns_path,
        "hedge_path": hedge_path_path,
        "prices": prices_path,
    }


def accrued_hedge_pnl(
    hedge_path: pl.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> pl.DataFrame:
    """Hedge P&L accrued over each horizon, with the number of days it was observed on.

    The hedge is rebalanced at each close, so the P&L on holding day ``d`` is the delta
    set on day ``d-1`` applied to that day's move in the underlying. A day the contract
    was not quoted on contributes nothing and is *counted*: a sum over a path with holes
    is a partial hedge, and the count is what lets the caller null the label rather than
    present it as fully hedged.

    Summed in day order. An unordered parallel sum re-associates the floating point, which
    moves the label's last bit and its content digest from one run to the next.
    """
    cohort = ["symbol", "feature_date"]
    move = pl.col("underlying_price") - pl.col("underlying_price").shift(1).over(cohort)
    daily = hedge_path.sort([*cohort, "holding_day"]).with_columns(
        (pl.col("instr_delta").shift(1).over(cohort) * move).alias("daily_pnl")
    )
    accrued = daily.group_by(cohort).agg(
        *[
            expr
            for horizon in horizons
            for expr in (
                pl.col("daily_pnl")
                .filter(pl.col("holding_day").is_between(1, horizon))
                .sort_by(pl.col("holding_day").filter(pl.col("holding_day").is_between(1, horizon)))
                .sum()
                .alias(f"hedge_pnl_{horizon}d"),
                pl.col("daily_pnl")
                .filter(pl.col("holding_day").is_between(1, horizon))
                .is_not_null()
                .sum()
                .alias(f"hedge_days_{horizon}d"),
            )
        ]
    )
    return accrued.rename({"feature_date": "timestamp"})


def summarize_label_artifacts(
    *,
    case_study_id: str = "sp500_options",
) -> dict[str, int]:
    """Return row counts for persisted same-contract artifacts."""
    case_dir = get_case_study_dir(case_study_id)
    labels_dir = case_dir / "labels"
    summary: dict[str, int] = {}
    for name in ("contract_returns", "hedge_path", "prices"):
        path = labels_dir / f"{name}.parquet"
        if path.exists():
            summary[name] = int(pl.scan_parquet(path).select(pl.len()).collect().item())
    return summary


def _build_price_lookup(raw_leg: pl.DataFrame, date_alias: str, prefix: str) -> pl.DataFrame:
    return raw_leg.select(
        [
            pl.col("date").alias(date_alias),
            "symbol",
            "strike",
            "expiration",
            pl.col("mid_price").alias(f"{prefix}_mid"),
            pl.col("bid").alias(f"{prefix}_bid"),
            pl.col("ask").alias(f"{prefix}_ask"),
        ]
    )
