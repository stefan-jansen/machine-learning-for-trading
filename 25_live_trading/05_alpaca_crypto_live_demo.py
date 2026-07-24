# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Alpaca Crypto Live Trading Demo
#
# **Docker image**: `ml4t`
#
# **Chapter**: 25 - Live Trading Systems
# **Section**: 25.3 (Alpaca Integration - Crypto)
# **Learning Outcome**: LO2 - Deploy crypto strategies with 24/7 market access
#
# **Purpose**: Demonstrate the operational shape of an always-on crypto strategy connected to Alpaca: how the
# 19-perp case study universe maps onto Alpaca's USD spot venue, how a momentum-based z-score signal is
# routed through a broker connection, and where funding-window timing fits in the execution loop.
#
# **Data contract - what is and is not implemented**:
# - The deployed signal is a **momentum z-score proxy**, not the Chapter 6 perp-spot premium index. The
#   production path for the Ch6 premium index would read perp+spot prices from a venue feed and compute
#   `(perp - spot) / spot`, then take a rolling z-score; this notebook stops short of that wiring and uses
#   the close-to-close momentum z-score as a shape-equivalent stand-in.
# - The simulated path (no credentials) uses a minimal mock broker; it is **not** a SafeBroker shadow-mode
#   test. The credential-present live path is the only SafeBroker-mediated execution path in this notebook.
# - The 19-perp universe is the case-study target; Alpaca supports a strict subset as USD spot pairs. The
#   universe-mapping table below makes the gap explicit.
#
# **Learning Objectives**
# - Contrast the operational demands of 24/7 crypto deployment with regular-hours equity deployment.
# - Inspect which case-study perps are tradeable on Alpaca and which are dropped at the venue boundary.
# - Connect funding-window-aware logging to a broker-facing crypto execution loop.
#
# **Prerequisites**:
# - Alpaca account with crypto trading enabled (live path)
# - Environment variables: ALPACA_API_KEY, ALPACA_SECRET_KEY (live path)

# %%
"""Demonstrate a paper-safe crypto deployment loop for an always-on market."""

import asyncio
import logging
import os
import warnings
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from async_utils import run_async
from ml4t.backtest import OrderSide, Strategy

from utils.paths import display_path, get_output_dir
from utils.reproducibility import set_global_seeds

HAS_ALPACA_SDK = False
try:
    import alpaca  # noqa: F401
    from ml4t.live import AlpacaBroker, AlpacaDataFeed, LiveEngine, LiveRiskConfig
    from ml4t.live.safety import SafeBroker

    HAS_ALPACA_SDK = True
except ImportError:
    pass

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("alpaca").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

if HAS_ALPACA_SDK:
    print("[OK] ml4t.live Alpaca components imported")
else:
    print("Alpaca SDK not installed (uv add alpaca-py); running simulation only")

# %% tags=["parameters"]
DEMO_DURATION_SECONDS = 60
MAX_SYMBOLS = 0
SIMULATION_STEPS = 20
LIVE_FEED = 0  # explicit opt-in; default execution is offline and paper-safe
SEED = 42

# %%
# The Alpaca crypto WebSocket loop is incompatible with papermill's nest_asyncio:
# `asyncio.wait_for` cannot reliably cancel the inner streaming task, so a
# headless run hangs past DEMO_DURATION_SECONDS. Detect papermill and fall back
# to the simulated path; interactive Jupyter is unaffected.
if os.environ.get("ML4T_HEADLESS_PAPERMILL") == "1":
    LIVE_FEED = 0

set_global_seeds(SEED)

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
PAPER_TRADING = True

# %% [markdown]
# ## 1. Universe Mapping: Case Study to Alpaca
#
# The crypto_perps_funding case study defines a 19-perp Binance universe (USDT suffix). Alpaca runs a USD
# spot venue and only lists a subset of these. The table below makes the gap explicit so readers can see
# exactly which strategy holdings would route through Alpaca and which would need a different venue.


# %%
# Source: case_studies/crypto_perps_funding/config/setup.yaml :: universe.symbols (n_assets = 19)
CASE_STUDY_PERP_UNIVERSE = [
    "AAVEUSDT",
    "ADAUSDT",
    "APTUSDT",
    "ATOMUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "COMPUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "ETHUSDT",
    "INJUSDT",
    "LINKUSDT",
    "MKRUSDT",
    "NEARUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "UNIUSDT",
    "XRPUSDT",
]

# Alpaca USD spot symbols that map to the case-study perps, checked on 2026-07-22
# against <https://alpaca.markets/support/what-cryptocurrencies-does-alpaca-currently-support>.
# Alpaca's Assets API remains authoritative because venue coverage can change.
PERP_TO_ALPACA_USD = {
    "AAVEUSDT": "AAVE/USD",
    "ADAUSDT": "ADA/USD",
    "AVAXUSDT": "AVAX/USD",
    "BTCUSDT": "BTC/USD",
    "DOGEUSDT": "DOGE/USD",
    "DOTUSDT": "DOT/USD",
    "ETHUSDT": "ETH/USD",
    "LINKUSDT": "LINK/USD",
    "MKRUSDT": "MKR/USD",
    "SOLUSDT": "SOL/USD",
    "UNIUSDT": "UNI/USD",
    "XRPUSDT": "XRP/USD",
}

UNIVERSE_MAPPING = pl.DataFrame(
    [
        {
            "perp_symbol": perp,
            "alpaca_usd_pair": PERP_TO_ALPACA_USD.get(perp),
            "tradeable_on_alpaca": perp in PERP_TO_ALPACA_USD,
        }
        for perp in CASE_STUDY_PERP_UNIVERSE
    ]
)
UNIVERSE_MAPPING

# %%
COVERAGE_FRACTION = UNIVERSE_MAPPING["tradeable_on_alpaca"].mean()
print(
    f"\nCoverage: {len(PERP_TO_ALPACA_USD)}/{len(CASE_STUDY_PERP_UNIVERSE)} perps "
    f"({COVERAGE_FRACTION:.0%}) tradeable on Alpaca USD spot."
)

# The runnable subset for the rest of the notebook (in case-study order)
ALL_CRYPTO_SYMBOLS = [
    PERP_TO_ALPACA_USD[p] for p in CASE_STUDY_PERP_UNIVERSE if p in PERP_TO_ALPACA_USD
]
CRYPTO_SYMBOLS = ALL_CRYPTO_SYMBOLS[:MAX_SYMBOLS] if MAX_SYMBOLS > 0 else ALL_CRYPTO_SYMBOLS.copy()

# %% [markdown]
# ## 2. Crypto Market Characteristics
#
# Crypto markets differ from equities:
#
# | Aspect | Equities | Crypto |
# |--------|----------|--------|
# | Trading Hours | 9:30-16:00 ET | 24/7/365 |
# | Settlement | T+2 | Instant |
# | Minimum Trade | 1 share | Fractional |
# | Volatility | ~1% daily | ~3-5% daily |
# | Funding Rates | N/A | 8-hour intervals |

# %%
print(f"24/7 market access; symbols routed via Alpaca USD spot: {', '.join(CRYPTO_SYMBOLS)}")

# %% [markdown]
# ## 3. Credential Check
#
# The credential check determines which path the notebook runs: live (Alpaca USD spot) or simulated (mock
# broker). It is not a shadow-mode test in the simulated case - that distinction matters because shadow mode
# implies a real broker connection with virtualised order routing, which the simulation path does not have.

# %%
HAS_CREDENTIALS = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY) and HAS_ALPACA_SDK
if HAS_CREDENTIALS:
    print("Alpaca paper credentials are available; live transport still requires LIVE_FEED=1.")
else:
    print("No Alpaca credentials; running mock-broker simulation (not SafeBroker shadow mode)")

# %% [markdown]
# **Finding**: The credential block exposes whether the notebook is connected, simulated, or only partially
# configured before any strategy state is created.
#
# **Trading implication**: Crypto notebooks should make execution mode obvious because overnight and weekend
# operation amplify the cost of confusing a shadow session with a routed broker session.
#
# %% [markdown]
# ## 4. Momentum Z-Score Strategy (Premium-Proxy)
#
# The deployed signal is a **momentum z-score proxy**, not the Chapter 6 premium index. The strategy
# computes the per-bar return, then converts it into a z-score using rolling mean and stdev. This shares the
# *shape* of the premium signal (a centered, dimensionless mean-reversion driver) without requiring a perp
# feed, so it can run on the Alpaca USD spot venue end-to-end.
#
# **Production wiring (not implemented here)**: read perp and spot closes from a venue feed, compute
# `(perp - spot) / spot`, then take the rolling z-score. The strategy structure below would consume the
# resulting series in place of the momentum proxy.


# %%
# Binance funding times in UTC. Comparing requires a tz-aware UTC timestamp;
# `_is_funding_hour` normalises before checking so naive or non-UTC inputs do
# not silently miss funding windows.
FUNDING_HOURS_UTC = [0, 8, 16]


def _is_funding_hour(timestamp: datetime) -> bool:
    """Return True if `timestamp` falls on a Binance funding hour, in UTC.

    Naive timestamps are interpreted as UTC; tz-aware timestamps are converted
    before comparison.
    """
    if timestamp.tzinfo is None:
        ts_utc = timestamp.replace(tzinfo=UTC)
    else:
        ts_utc = timestamp.astimezone(UTC)
    return ts_utc.hour in FUNDING_HOURS_UTC


def _compute_momentum_zscore(prices: list[float], lookback: int) -> float:
    """Premium-proxy: rolling z-score of close-to-close returns."""
    if len(prices) < lookback + 2:
        return 0.0
    baseline_prices = prices[-lookback - 2 : -1]
    baseline_returns = [
        (baseline_prices[i] - baseline_prices[i - 1]) / baseline_prices[i - 1]
        for i in range(1, len(baseline_prices))
    ]
    mean_ret = float(np.mean(baseline_returns))
    std_ret = float(np.std(baseline_returns))
    if std_ret < 1e-8:
        return 0.0
    current_ret = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0.0
    return (current_ret - mean_ret) / std_ret


# %%
# compliance: skip cell_size - cohesive Strategy implementation binds signals to broker state
class CryptoPremiumStrategy(Strategy):
    """Mean-reversion on a momentum z-score proxy, routed through a crypto broker."""

    def __init__(
        self,
        lookback: int = 10,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.25,
        position_size: float = 0.1,
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size

        self.prices: dict[str, list[float]] = {}
        self.signals: list[dict] = []
        self.funding_events: list[dict] = []

    def on_start(self, broker):
        logger.info(
            f"Strategy started: lookback={self.lookback}, entry={self.entry_threshold:.2f}z"
        )
        for symbol in CRYPTO_SYMBOLS:
            self.prices[symbol] = []

    def _route_signal(self, broker, timestamp, symbol, close, zscore, current_qty):
        """Translate a z-score into a long-only spot entry or exit."""
        if zscore < -self.entry_threshold and current_qty <= 0:
            action, side = "BUY", OrderSide.BUY
            qty = self.position_size
            reason = "z-score below -entry_threshold (mean reversion long)"
        elif zscore >= -self.exit_threshold and current_qty > 0:
            action = "CLOSE"
            side = OrderSide.SELL
            qty = current_qty
            reason = "negative z-score reverted toward zero"
        else:
            return
        self.signals.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "action": action,
                "zscore": zscore,
                "price": close,
                "reason": reason,
            }
        )
        logger.info(f"{action} {symbol}: z-score={zscore:.2f}")
        order = broker.submit_order(symbol, qty, side=side)
        if isinstance(order, dict):
            self.signals[-1]["order_status"] = order["status"]

    def on_data(self, timestamp: datetime, data: dict, context: dict, broker):
        for symbol, bar in data.items():
            if symbol not in self.prices:
                self.prices[symbol] = []
            close = bar["close"]
            self.prices[symbol].append(close)

            zscore = _compute_momentum_zscore(self.prices[symbol], self.lookback)
            position = broker.get_position(symbol)
            current_qty = position.quantity if position else 0.0

            if _is_funding_hour(timestamp):
                self.funding_events.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "position": current_qty,
                        "zscore": zscore,
                    }
                )

            self._route_signal(broker, timestamp, symbol, close, zscore, current_qty)

    def on_end(self, broker):
        logger.info(
            f"Strategy ended. Signals: {len(self.signals)}; funding events: {len(self.funding_events)}"
        )


# %% [markdown]
# ## 5. Simulated Path: Flat-Dict Mock Broker
#
# When credentials are missing, the demo runs against a tiny mock broker. The portfolio is a flat dict so
# the simulation state is inspectable without nested dataclasses. This is **not** a SafeBroker shadow-mode
# test because shadow mode requires a real underlying broker connection.


# %%
DEFAULT_REF_PRICES = {
    "AAVE/USD": 280.0,
    "ADA/USD": 0.65,
    "AVAX/USD": 35.0,
    "BTC/USD": 43000.0,
    "DOGE/USD": 0.15,
    "DOT/USD": 7.0,
    "ETH/USD": 2200.0,
    "LINK/USD": 14.0,
    "MKR/USD": 1700.0,
    "SOL/USD": 100.0,
    "UNI/USD": 6.5,
    "XRP/USD": 0.55,
}


class MockCryptoBroker:
    """Minimal sync broker for the no-credential simulation path.

    Long-only: a SELL without sufficient existing inventory is rejected.
    The strategy matches this spot-venue constraint and never opens shorts.
    """

    def __init__(self, initial_cash: float = 10_000.0):
        self.portfolio = {"cash": initial_cash, "positions": {}}
        self.order_log: list[dict] = []
        self.current_prices = dict(DEFAULT_REF_PRICES)
        self.current_timestamp = datetime(2026, 1, 1, tzinfo=UTC)

    def update_market(self, timestamp: datetime, prices: dict[str, float]) -> None:
        """Update the simulated market used for subsequent fills."""
        self.current_timestamp = timestamp
        self.current_prices.update(prices)

    def get_position(self, symbol: str):
        pos = self.portfolio["positions"].get(symbol)
        if pos is None:
            return None

        class _PosView:
            quantity = pos["quantity"]

        return _PosView()

    def submit_order(self, asset: str, quantity: float, side=None, **kwargs) -> dict:
        price = self.current_prices[asset]
        side_name = side.value if hasattr(side, "value") else str(side or "BUY")
        status = "rejected"
        if side is None or side == OrderSide.BUY:
            cost = quantity * price
            if cost <= self.portfolio["cash"]:
                self.portfolio["cash"] -= cost
                pos = self.portfolio["positions"].get(asset)
                if pos is None:
                    self.portfolio["positions"][asset] = {
                        "quantity": quantity,
                        "entry_price": price,
                    }
                else:
                    total_qty = pos["quantity"] + quantity
                    avg = (pos["quantity"] * pos["entry_price"] + quantity * price) / total_qty
                    self.portfolio["positions"][asset] = {"quantity": total_qty, "entry_price": avg}
                status = "filled"
        elif side == OrderSide.SELL:
            pos = self.portfolio["positions"].get(asset)
            if pos is not None and pos["quantity"] >= quantity:
                self.portfolio["cash"] += quantity * price
                remaining = pos["quantity"] - quantity
                if remaining > 0:
                    self.portfolio["positions"][asset] = {
                        "quantity": remaining,
                        "entry_price": pos["entry_price"],
                    }
                else:
                    del self.portfolio["positions"][asset]
                status = "filled"
        order = {
            "order_id": f"SIM-{len(self.order_log) + 1}",
            "timestamp": self.current_timestamp,
            "symbol": asset,
            "side": side_name,
            "quantity": quantity,
            "price": price,
            "status": status,
        }
        self.order_log.append(order)
        return order


# %%
async def run_simulated_crypto_demo() -> dict:
    """Drive the strategy against the mock broker; return a small results dict."""
    print("MOCK-BROKER SIMULATION (no Alpaca connection)")
    strategy = CryptoPremiumStrategy(
        lookback=10,
        entry_threshold=1.5,
        exit_threshold=0.25,
        position_size=0.01,
    )
    mock = MockCryptoBroker()
    strategy.on_start(mock)

    set_global_seeds(SEED)
    base_prices = {sym: DEFAULT_REF_PRICES[sym] for sym in CRYPTO_SYMBOLS}
    # Start at a known funding hour in UTC so the funding-event capture exercises the new tz-aware check.
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    for i in range(SIMULATION_STEPS):
        timestamp = start + timedelta(hours=i)
        data = {}
        for symbol in CRYPTO_SYMBOLS:
            base_prices[symbol] *= 1 + np.random.normal(0.001, 0.03)
            data[symbol] = {
                "open": base_prices[symbol] * 0.998,
                "high": base_prices[symbol] * 1.01,
                "low": base_prices[symbol] * 0.99,
                "close": base_prices[symbol],
                "volume": int(np.random.randint(100, 10000)),
            }
        mock.update_market(timestamp, {symbol: bar["close"] for symbol, bar in data.items()})
        strategy.on_data(timestamp, data, {}, mock)
    strategy.on_end(mock)

    signals_df = pl.DataFrame(strategy.signals) if strategy.signals else pl.DataFrame()
    funding_df = (
        pl.DataFrame(strategy.funding_events) if strategy.funding_events else pl.DataFrame()
    )
    orders_df = pl.DataFrame(mock.order_log) if mock.order_log else pl.DataFrame()
    if len(orders_df):
        assert orders_df.filter(pl.col("status") == "rejected").is_empty()
    return {
        "signals": signals_df,
        "funding_events": funding_df,
        "orders": orders_df,
        "broker": mock,
    }


# %% [markdown]
# ## 6. Live Path: Alpaca Engine Wiring
#
# When credentials are present and `LIVE_FEED=1`, the demo wires `AlpacaDataFeed > SafeBroker > LiveEngine`
# and runs for at most `DEMO_DURATION_SECONDS`. The SafeBroker layer is the only place this notebook talks
# about shadow mode. The simulated path uses the mock broker above and is not a shadow-mode test.


# %%
def create_alpaca_crypto_engine(strategy):
    """Wire AlpacaDataFeed, SafeBroker, and LiveEngine for the live crypto demo."""
    broker = AlpacaBroker(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=PAPER_TRADING)
    risk_state_path = get_output_dir(25, "alpaca_crypto_demo") / "risk_state.json"
    risk_config = LiveRiskConfig(
        shadow_mode=True,
        max_position_value=5_000.0,
        max_order_value=1_000.0,
        max_orders_per_minute=20,
        state_file=str(risk_state_path),
    )
    safe_broker = SafeBroker(broker, risk_config)
    feed = AlpacaDataFeed(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        symbols=CRYPTO_SYMBOLS,
        data_type="bars",
    )
    engine = LiveEngine(strategy=strategy, broker=safe_broker, feed=feed)
    for name in [
        "alpaca",
        "alpaca.data",
        "alpaca.data.live",
        "alpaca.data.live.websocket",
        "alpaca.trading.stream",
        "websockets",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    print(f"Risk State: {display_path(risk_state_path)}")
    return engine, safe_broker, feed, broker


# %%
async def run_live_alpaca_demo() -> dict:
    """Drive the explicitly selected live Alpaca paper path."""
    strategy = CryptoPremiumStrategy(
        lookback=10,
        entry_threshold=1.5,
        exit_threshold=0.25,
        position_size=0.01,
    )
    engine, safe_broker, feed, raw_broker = create_alpaca_crypto_engine(strategy)
    print(
        f"Starting Alpaca crypto engine for {DEMO_DURATION_SECONDS}s; symbols {', '.join(CRYPTO_SYMBOLS)}"
    )
    try:
        await asyncio.wait_for(engine.connect(), timeout=10)
        await asyncio.wait_for(engine.run(), timeout=DEMO_DURATION_SECONDS)
    except TimeoutError:
        print("Demo duration reached")
    finally:
        feed.stop()
        await raw_broker.disconnect()

    signals_df = pl.DataFrame(strategy.signals) if strategy.signals else pl.DataFrame()
    funding_df = (
        pl.DataFrame(strategy.funding_events) if strategy.funding_events else pl.DataFrame()
    )
    return {
        "signals": signals_df,
        "funding_events": funding_df,
        "orders": pl.DataFrame(),
        "broker": safe_broker,
    }


# %%
# Dispatch: explicit LIVE_FEED opt-in gates the live path.
async def crypto_demo_dispatch():
    if not LIVE_FEED:
        print(f"LIVE_FEED={LIVE_FEED}: running the offline mock-broker simulation")
        return await run_simulated_crypto_demo()
    if not HAS_CREDENTIALS:
        raise RuntimeError("LIVE_FEED requires the Alpaca SDK and paper credentials")
    return await run_live_alpaca_demo()


crypto_results = run_async(crypto_demo_dispatch())

print("\nRESULTS")
print(
    f"Signals: {len(crypto_results['signals'])}; "
    f"Funding events: {len(crypto_results['funding_events'])}; "
    f"Orders: {len(crypto_results['orders'])}"
)
crypto_results["signals"]

# %%
crypto_results["funding_events"]

# %%
crypto_results["orders"]

# %% [markdown]
# **Finding**: The results section ties signals, funding events, and portfolio state together in one replay.
# That gives the reader a broker-facing view of how the strategy behaves outside a pure backtest.
#
# **Trading implication**: The timestamps are operational markers only. Alpaca spot positions do not receive
# perpetual-swap funding; a production funding strategy needs perp-venue cash-flow records.
#
# %% [markdown]
# ## 7. 24/7 Trading Considerations
#
# When deploying crypto strategies:
#
# 1. **Infrastructure**:
#    - Use cloud deployment (always-on)
#    - Implement heartbeat monitoring
#    - Handle reconnection gracefully
#
# 2. **Risk Management**:
#    - Higher volatility = tighter stop losses
#    - Consider liquidation risk on leveraged positions
#    - Monitor exchange maintenance windows
#
# 3. **Funding Rates**:
#    - Binance: 00:00, 08:00, 16:00 UTC
#    - Track funding to optimize entry/exit timing
#    - Funding can be significant over time

# %%
print("\n" + "=" * 60)
print("24/7 TRADING CONSIDERATIONS")
print("=" * 60)

print("\n1. Always-On Infrastructure:")
print("   - Cloud VM or container deployment")
print("   - Automatic restart on failure")
print("   - Health check endpoints")

print("\n2. Funding Rate Timing:")
print("   - Binance: 00:00, 08:00, 16:00 UTC")
print("   - Treat these as clock markers on Alpaca spot, not funding cash flows")
print("   - Reconcile actual funding PnL only on the perpetual venue")

print("\n3. Weekend Considerations:")
print("   - Liquidity may be lower")
print("   - Volatility can spike")
print("   - No 'market close' for stops")

# %% [markdown]
# **Finding**: The operational checklist shows that 24/7 trading is an infrastructure problem as much as a
# signal problem.
#
# **Trading implication**: A strategy that looks stable in backtests can still fail operationally if it
# assumes maintenance windows or supervision patterns borrowed from regular-hours equity trading.
#
# %% [markdown]
# ## Key Takeaways
#
# - **Universe mapping is a hard venue constraint, not a soft target.** The current Alpaca list adds ADA to
#   the mapped subset. The output above computes current coverage; unsupported perps need another adapter.
# - **The deployed signal is a momentum z-score proxy, not the Ch6 premium index.** Production wiring would
#   read perp + spot closes from a venue feed and compute `(perp - spot) / spot`; the structure of the
#   strategy is the same, but the input series is the missing piece on the Alpaca USD spot venue.
# - **Funding-window detection must normalise to UTC.** `_is_funding_hour` treats naive timestamps as UTC and
#   converts tz-aware timestamps before comparing against `[0, 8, 16]`; otherwise a non-UTC clock silently
#   skips every funding event.
# - **The simulated path is not shadow mode.** Without credentials the demo uses a flat-dict `MockCryptoBroker`
#   for inspection only; shadow mode requires a real broker connection wrapped in `SafeBroker`, which the
#   credential-present live path provides.
#
# **Next**: see `09_crypto_funding_deployment_loop.py` for the production-style perp deployment loop with
# OKX as the venue.
