"""LEAN adapter for case-study parity checks."""

from __future__ import annotations

import csv
import json
import math
import time
from textwrap import dedent

import pandas as pd
import polars as pl

from validation._library_validation import load_validation_helper, resolve_backtest_repo
from validation.result import BacktestResult


def _check_daily_frequency(prices: pl.DataFrame) -> None:
    timestamps = prices.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()
    if len(timestamps) < 2:
        return
    deltas = pd.Series(pd.to_datetime(timestamps)).diff().dropna()
    if deltas.median() < pd.Timedelta(hours=12):
        raise ValueError("LEAN case-study adapter currently supports daily data only.")


def run_lean(
    prices: pl.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float = 1_000_000,
    commission_rate: float = 0.001,
    project_slug: str = "case_study_lean",
) -> BacktestResult:
    """Run LEAN on case-study target weights."""
    _check_daily_frequency(prices)

    lean_runner = load_validation_helper("lean_runner")
    backtest_repo = resolve_backtest_repo()
    lean_workspace = backtest_repo / "validation" / "lean" / "workspace"
    lean_config = lean_workspace / "lean.json"
    if not lean_config.exists():
        raise FileNotFoundError(f"LEAN config not found: {lean_config}")

    lean_cmd = lean_runner.resolve_lean_command()

    asset_names = list(weights.columns)
    asset_to_ticker = lean_runner.build_hashed_ticker_map(project_slug, asset_names)
    ticker_to_asset = {ticker: asset for asset, ticker in asset_to_ticker.items()}

    project_dir = lean_workspace / project_slug
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "backtests").mkdir(parents=True, exist_ok=True)

    config_json = {
        "algorithm-language": "Python",
        "parameters": {},
        "description": "ml4t case-study lean parity",
    }
    (project_dir / "config.json").write_text(json.dumps(config_json, indent=4), encoding="utf-8")
    (project_dir / "ml4t_symbol_map.json").write_text(
        json.dumps({ticker: asset for ticker, asset in ticker_to_asset.items()}, indent=2),
        encoding="utf-8",
    )
    (project_dir / "asset_symbols.csv").write_text(
        "asset,ticker\n"
        + "\n".join(f"{asset},{asset_to_ticker[asset]}" for asset in asset_names)
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "rebalance_dates.csv").write_text(
        "\n".join(pd.to_datetime(weights.index).strftime("%Y-%m-%d")) + "\n",
        encoding="utf-8",
    )

    weights_long = weights.stack().reset_index()
    weights_long.columns = ["timestamp", "asset", "target_weight"]
    weights_long = weights_long[weights_long["target_weight"] != 0.0]
    weights_long["timestamp"] = pd.to_datetime(weights_long["timestamp"]).dt.strftime("%Y-%m-%d")
    weights_long.to_csv(project_dir / "weights.csv", index=False)

    first_date = pd.Timestamp(weights.index[0]).date()
    last_date = pd.Timestamp(weights.index[-1]).date()
    main_code = (
        dedent(
            f"""
        # region imports
        from AlgorithmImports import *
        # endregion

        import csv
        import math
        from pathlib import Path


        class Ml4tPercentFeeModel(FeeModel):
            def __init__(self, rate: float):
                self.rate = rate

            def GetOrderFee(self, parameters: OrderFeeParameters) -> OrderFee:
                order = parameters.Order
                security = parameters.Security
                reference_price = float(security.Open) if float(security.Open) > 0 else float(security.Price)
                fee = abs(float(order.AbsoluteQuantity)) * reference_price * self.rate
                return OrderFee(CashAmount(fee, "USD"))


        class Ml4tCaseStudyParity(QCAlgorithm):
            def initialize(self):
                self.set_start_date({first_date.year}, {first_date.month}, {first_date.day})
                self.set_end_date({last_date.year}, {last_date.month}, {last_date.day})
                self.set_cash({float(initial_cash)})
                self.set_brokerage_model(BrokerageName.DEFAULT, AccountType.MARGIN)
                self._targets = {{}}
                self._rebalance_dates = set()
                base_path = Path(__file__).resolve().parent
                self._equity_path = base_path / "ml4t_daily_equity.csv"
                self._order_events_path = base_path / "ml4t_order_events.csv"
                self._initialize_artifact_files()

                with (base_path / "weights.csv").open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = row["timestamp"]
                        if key not in self._targets:
                            self._targets[key] = {{}}
                        self._targets[key][row["asset"]] = float(row["target_weight"])
                self._rebalance_dates = {{
                    line.strip()
                    for line in (base_path / "rebalance_dates.csv").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }}

                self._asset_order = []
                self._symbols = {{}}
                fee_model = Ml4tPercentFeeModel({float(commission_rate)})
                with (base_path / "asset_symbols.csv").open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        asset = row["asset"].strip()
                        ticker = row["ticker"].strip()
                        if not asset or not ticker:
                            continue
                        security = self.add_equity(ticker, Resolution.DAILY)
                        security.set_leverage(2.0)
                        security.set_fee_model(fee_model)
                        security.set_slippage_model(ConstantSlippageModel(0))
                        self._asset_order.append(asset)
                        self._symbols[asset] = security.symbol
                if self._symbols:
                    self.set_benchmark(self._symbols[self._asset_order[0]])

            def _initialize_artifact_files(self):
                with self._equity_path.open("w", newline="") as f:
                    csv.writer(f).writerow(
                        ["timestamp", "equity", "cash", "total_fees", "holdings_value"]
                    )
                with self._order_events_path.open("w", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            "timestamp",
                            "symbol",
                            "status",
                            "direction",
                            "fill_quantity",
                            "fill_price",
                            "fee",
                            "message",
                            "order_id",
                        ]
                    )

            def _append_csv_row(self, path: Path, row: list[object]) -> None:
                with path.open("a", newline="") as f:
                    csv.writer(f).writerow(row)

            def _target_quantity(self, target_weight: float, price: float) -> int:
                raw_qty = (target_weight * float(self.portfolio.total_portfolio_value)) / price
                if raw_qty >= 0:
                    return int(math.floor(raw_qty + 1e-12))
                return -int(math.floor(abs(raw_qty) + 1e-12))

            def on_data(self, data: Slice):
                key = self.time.strftime("%Y-%m-%d")
                if key not in self._rebalance_dates:
                    return
                targets = self._targets.get(key, {{}})

                bars = data.bars
                for asset in self._asset_order:
                    symbol = self._symbols[asset]
                    if bars is None or symbol not in bars:
                        continue
                    price = float(bars[symbol].close)
                    if price <= 0:
                        continue
                    target_qty = self._target_quantity(targets.get(asset, 0.0), price)
                    current_qty = int(self.portfolio[symbol].quantity)
                    delta = target_qty - current_qty
                    if delta != 0:
                        self.market_order(symbol, delta)

                self._append_csv_row(
                    self._equity_path,
                    [
                        key,
                        float(self.portfolio.total_portfolio_value),
                        float(self.portfolio.cash),
                        float(self.portfolio.total_fees),
                        float(self.portfolio.total_holdings_value),
                    ],
                )

            def on_order_event(self, order_event: OrderEvent):
                fee_amount = 0.0
                if order_event.order_fee and order_event.order_fee.value is not None:
                    fee_amount = float(order_event.order_fee.value.amount)

                message = str(order_event.message or "").replace("\\n", " ").strip()
                symbol = order_event.symbol.value if order_event.symbol is not None else ""
                self._append_csv_row(
                    self._order_events_path,
                    [
                        self.time.strftime("%Y-%m-%d %H:%M:%S"),
                        symbol,
                        str(order_event.status),
                        str(order_event.direction),
                        float(order_event.fill_quantity),
                        float(order_event.fill_price),
                        fee_amount,
                        message,
                        int(order_event.order_id),
                    ],
                )
        """
        ).strip()
        + "\n"
    )
    (project_dir / "main.py").write_text(main_code, encoding="utf-8")

    prices_pd = prices.to_pandas().rename(columns={"symbol": "asset"})
    prices_pd["timestamp"] = pd.to_datetime(prices_pd["timestamp"]).dt.tz_localize(None)
    prices_by_asset = {
        asset_name: prices_pd.loc[prices_pd["asset"] == asset_name]
        .sort_values("timestamp")
        .set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for asset_name in asset_names
    }

    data_root = lean_workspace / "data" / "equity" / "usa"
    manifest_path = data_root / f"{project_slug}_manifest.json"
    cache_hit = lean_runner.export_lean_daily_data(
        data_root=data_root,
        prices_by_asset=prices_by_asset,
        asset_to_ticker=asset_to_ticker,
        manifest_path=manifest_path,
        signature_payload={
            "cache_version": 2,
            "project_slug": project_slug,
            "asset_names": asset_names,
            "asset_to_ticker": asset_to_ticker,
            "timestamp_min": str(prices_pd["timestamp"].min()),
            "timestamp_max": str(prices_pd["timestamp"].max()),
            "row_count": int(len(prices_pd)),
        },
    )

    output_dir = project_dir / "backtests" / f"{project_slug}_{int(time.time())}"
    runtime_sec = lean_runner.run_lean_backtest(
        lean_cmd=lean_cmd,
        cwd=backtest_repo,
        project_dir=project_dir,
        lean_config=lean_config,
        output_dir=output_dir,
        timeout=1800,
        env=lean_runner.make_lean_env(),
    )
    lean_runner.copy_lean_artifacts(
        project_dir,
        output_dir,
        ["ml4t_daily_equity.csv", "ml4t_order_events.csv", "ml4t_symbol_map.json"],
    )

    num_trades, final_value, trades_df, equity_df, order_events_df = (
        lean_runner.load_lean_artifacts(output_dir)
    )
    if equity_df is None:
        raise RuntimeError("LEAN equity artifact missing.")

    equity_series = pd.Series(
        equity_df["equity"].to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(equity_df["timestamp"])),
        name="portfolio_value",
    )
    metrics = {
        "runtime_sec": runtime_sec,
        "final_value": final_value,
        "n_trades": num_trades,
        "cache_hit": cache_hit,
        "order_events": 0 if order_events_df is None else len(order_events_df),
    }

    return BacktestResult(
        framework="lean",
        equity_curve=equity_series,
        trades=trades_df if trades_df is not None else pd.DataFrame(),
        positions=pd.DataFrame(),
        metrics=metrics,
    )
