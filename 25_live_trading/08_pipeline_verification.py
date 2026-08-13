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
# # Pipeline Verification: Backtest vs Live Parity
#
# **Docker image**: `ml4t`
#
# **Chapter**: 25 - Live Trading Systems
# **Section**: 25.6 (Pipeline Verification)
# **Learning Outcome**: LO4 - Implement pipeline verification tests
#
# This notebook implements the verification methodology from Section 25.6:
# 1. **Feature Parity**: Same inputs → same features
# 2. **Prediction Consistency**: Same features → same predictions
# 3. **Sizing Logic**: Same signals → same order sizes
# 4. **Automated Regression**: CI-friendly test suite
#
# **Key Principle**: If backtest and live pipelines produce identical outputs
# for identical inputs, we have **technical parity**.
#
# **Prerequisites**: Familiarity with Chapter 16 strategy simulation and Section 25.6's parity-testing
# workflow. The notebook assumes the reader wants to verify an implementation, not just inspect a model.

# %%
"""Pipeline Verification: stage-by-stage backtest vs live parity checks."""

import asyncio
import logging
import os
import sys
import tempfile
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy as np
    from async_utils import run_async
    from ml4t.backtest import OrderSide, Strategy
    from ml4t.backtest.types import Order, OrderStatus, OrderType
    from ml4t.live import LiveRiskConfig
    from ml4t.live.safety import SafeBroker, VirtualPortfolio
    from ml4t.live.wrappers import ThreadSafeBrokerWrapper

from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

print("[OK] Components imported")

# %% tags=["parameters"]
N_BARS = 30
SEED = 12345  # Pinned for §25.6 deterministic harness

# %%
set_global_seeds(SEED)

# %% [markdown]
# **Learning Objectives**
# - Build deterministic data that can exercise both backtest and live pipelines.
# - Compare features, predictions, and order logic stage by stage.
# - Turn parity checks into regression tests that fail loudly when behavior diverges.

# %% [markdown]
# **Finding:** Importing the same backtest and live components into one notebook is the first parity test.
# If the infrastructure cannot coexist in a single environment, the chapter's unified-framework claim has
# already broken down before any strategy logic runs.

# %% [markdown]
# ## 1. Define Test Strategy
#
# The strategy is intentionally deterministic so any mismatch between backtest and live outputs is a
# technical bug, not a statistical fluctuation.


# %%
@dataclass
class VerificationResult:
    """Result of a single verification test."""

    test_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str = ""
    skipped: bool = False
    expected_difference: bool = False


# A parity harness must not waive a mismatch merely because it is convenient.
# Any intentional divergence belongs in a separate, explicitly tested contract.
EXPECTED_DIFFERENCE: set[str] = set()


# %% [markdown]
# ### Deterministic Policy Functions
#
# These pure functions isolate the four decisions that both execution paths must reproduce.


# %%
def compute_features(prices: list[float], lookback: int) -> dict[str, float]:
    """Compute the feature vector from the latest complete lookback window."""
    if len(prices) < lookback:
        return {}
    values = np.asarray(prices[-lookback:])
    returns = np.diff(values) / values[:-1]
    return {
        "momentum": (values[-1] / values[0]) - 1,
        "volatility": float(np.std(returns)),
        "mean": float(np.mean(values)),
        "current": float(values[-1]),
    }


# %% [markdown]
# The prediction is a fixed linear rule so a mismatch can only come from implementation drift.


# %%
def compute_prediction(features: dict[str, float]) -> float:
    """Map a complete feature vector to one deterministic score."""
    if not features:
        return 0.0
    return (
        features["momentum"] * 0.5
        - features["volatility"] * 0.3
        + (features["current"] / features["mean"] - 1) * 0.2
    )


# %% [markdown]
# Position state gates entries and exits; HOLD is the only action inside the no-trade band.


# %%
def compute_signal(prediction: float, threshold: float, has_position: bool) -> str:
    """Convert a score and current position state to an order intent."""
    if prediction > threshold and not has_position:
        return "BUY"
    if prediction < -threshold and has_position:
        return "SELL"
    return "HOLD"


# %% [markdown]
# Fixed-fraction sizing keeps the parity exercise focused on identical inputs and state.


# %%
def compute_size(signal: str, price: float, cash: float, position_quantity: int) -> int:
    """Size entries from cash and bound exits by the current long position."""
    if signal == "HOLD":
        return 0
    target = max(0, int(cash * 0.1 / price))
    return target if signal == "BUY" else min(target, max(0, position_quantity))


# %% [markdown]
# The fill record captures the realized order and broker state after submission.


# %%
def fill_record(
    timestamp: datetime,
    symbol: str,
    signal: str,
    size: int,
    price: float,
    order: Order,
    broker: Any,
) -> dict[str, Any]:
    positions = tuple(
        sorted((asset, int(item.quantity)) for asset, item in broker.positions.items())
    )
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "signal": signal,
        "size": size,
        "price": price,
        "order_id": order.order_id,
        "status": order.status.value,
        "filled_quantity": int(order.filled_quantity),
        "filled_price": float(order.filled_price),
        "cash": float(broker.get_cash()),
        "positions": positions,
    }


# %% [markdown]
# The per-symbol step records every intermediate value before it submits an actionable intent.


# %%
def process_symbol(strategy: Any, timestamp: datetime, symbol: str, bar: dict, broker: Any) -> None:
    """Advance one symbol through the complete decision pipeline."""
    prices = strategy.prices.setdefault(symbol, deque(maxlen=strategy.lookback + 5))
    close = bar["close"]
    prices.append(close)
    features = compute_features(list(prices), strategy.lookback)
    strategy.feature_log.append(
        {"timestamp": timestamp, "symbol": symbol, "features": features.copy()}
    )
    if not features:
        return

    prediction = compute_prediction(features)
    strategy.prediction_log.append(
        {"timestamp": timestamp, "symbol": symbol, "prediction": prediction}
    )
    position = broker.get_position(symbol)
    signal = compute_signal(
        prediction, strategy.threshold, bool(position and position.quantity > 0)
    )
    strategy.signal_log.append(
        {"timestamp": timestamp, "symbol": symbol, "signal": signal, "prediction": prediction}
    )
    if signal == "HOLD":
        return

    cash = broker.get_cash()
    position_quantity = int(position.quantity) if position else 0
    size = compute_size(signal, close, cash, position_quantity)
    if size <= 0:
        return
    side = OrderSide.BUY if signal == "BUY" else OrderSide.SELL
    order = broker.submit_order(symbol, size, side=side)
    strategy.order_log.append(fill_record(timestamp, symbol, signal, size, close, order, broker))


# %% [markdown]
# ### Verifiable Strategy
#
# The strategy owns state and delegates each symbol to the observable decision step above.


# %%
class VerifiableStrategy(Strategy):
    """Strategy with stage logs for backtest-live comparison."""

    def __init__(self, lookback: int = 10, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold
        self.prices: dict[str, deque] = {}
        self.feature_log: list[dict] = []
        self.prediction_log: list[dict] = []
        self.signal_log: list[dict] = []
        self.order_log: list[dict] = []

    def on_start(self, broker: Any) -> None:
        logger.info("VerifiableStrategy started")

    def on_data(self, timestamp: datetime, data: dict, context: dict, broker: Any) -> None:
        for symbol, bar in data.items():
            process_symbol(self, timestamp, symbol, bar, broker)

    def on_end(self, broker: Any) -> None:
        logger.info("Strategy ended: %s orders", len(self.order_log))


# %% [markdown]
# ## 2. Create Test Data
#
# Deterministic test data that both backtest and live will process.
#
# Determinism matters here because any randomness would weaken the causal link between a mismatch and the code
# path that produced it. The notebook is trying to isolate technical divergence, not market noise.

# %%
set_global_seeds(SEED)

SYMBOLS = ["SPY", "QQQ"]

# Static per-symbol offsets ensure the test tape is identical across processes,
# independent of PYTHONHASHSEED. Python's built-in hash() is process-randomised
# unless the interpreter is launched with a fixed PYTHONHASHSEED, which the
# notebook cannot control on a reader's machine.
SYMBOL_OFFSETS = {"SPY": 101, "QQQ": 202}


# %%
def generate_test_data() -> list[tuple[datetime, dict]]:
    """Generate deterministic OHLCV bars."""
    data = []
    base_prices = {"SPY": 500.0, "QQQ": 400.0}

    for i in range(N_BARS):
        timestamp = datetime(2025, 1, 1, 10, 0) + timedelta(minutes=i)
        bar_data = {}

        for symbol in SYMBOLS:
            np.random.seed(SEED + i * 100 + SYMBOL_OFFSETS[symbol])
            # A rising-then-falling SPY tape and its QQQ mirror guarantee that
            # the harness exercises BUY and SELL order paths, not only HOLD.
            regime = 1.0 if i < N_BARS // 2 else -1.0
            direction = regime if symbol == "SPY" else -regime
            change = direction * 0.004 + np.random.normal(0, 0.0005)
            base_prices[symbol] *= 1 + change
            price = base_prices[symbol]

            bar_data[symbol] = {
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": 1000000,
            }

        data.append((timestamp, bar_data))

    return data


# %%
TEST_DATA = generate_test_data()
print(f"[OK] Generated {len(TEST_DATA)} test bars for {SYMBOLS}")

# %% [markdown]
# **Finding:** The generated bars above create a shared input tape for both engines. That common tape is what
# lets the notebook blame a mismatch on implementation details rather than on different market states.

# %% [markdown]
# ## 3. Run Backtest Pipeline
#
# The backtest run establishes the reference outputs. Every later comparison asks whether the live-style path
# reproduces the same feature values, predictions, and order intentions on the same synthetic market tape.


# %%
class TestDataFeed:
    """Simple feed that replays test data."""

    def __init__(self, data: list):
        self._data = data
        self._index = 0
        self._running = False

    async def start(self):
        self._running = True
        self._index = 0

    def stop(self):
        self._running = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._running or self._index >= len(self._data):
            raise StopAsyncIteration
        timestamp, bar_data = self._data[self._index]
        self._index += 1
        await asyncio.sleep(0.001)  # Minimal delay
        return timestamp, bar_data, {}

    @property
    def stats(self):
        return {"bars": self._index}


# %% [markdown]
# ### In-Memory Reference Broker
#
# The reference broker fills immediately at the latest synthetic price and updates one virtual portfolio.


# %%
class BacktestBroker:
    """Minimal synchronous broker for the reference replay."""

    order_prefix = "BT"

    def __init__(self):
        self._portfolio = VirtualPortfolio(initial_cash=100_000.0)
        self._prices: dict[str, float] = {}
        self._next_order_id = 1

    @property
    def positions(self):
        return self._portfolio.positions

    def get_position(self, asset):
        return self._portfolio.positions.get(asset)

    def get_cash(self):
        return self._portfolio.cash

    def update_price(self, symbol, price):
        self._prices[symbol] = price

    def submit_order(self, asset, quantity, side=None, **kwargs):
        price = self._prices.get(asset, 100.0)
        order_id = f"{self.order_prefix}-{self._next_order_id:04d}"
        self._next_order_id += 1
        order = Order(
            asset=asset,
            side=side or OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            order_id=order_id,
            status=OrderStatus.FILLED,
            created_at=datetime.now(),
            filled_price=price,
            filled_quantity=quantity,
        )
        self._portfolio.process_fill(order)
        return order


# %% [markdown]
# ### Backtest Driver
#
# The reference driver logs every stage without threading or risk-control wrappers.


# %%
backtest_strategy = VerifiableStrategy(lookback=10, threshold=0.01)


async def run_backtest():
    """Replay the fixed tape through the reference strategy and broker."""

    broker = BacktestBroker()
    backtest_strategy.on_start(broker)

    for timestamp, bar_data in TEST_DATA:
        for symbol, bar in bar_data.items():
            broker.update_price(symbol, bar["close"])
        backtest_strategy.on_data(timestamp, bar_data, {}, broker)

    backtest_strategy.on_end(broker)


# %%
print("BACKTEST PIPELINE")
backtest_executed = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        run_async(run_backtest())
    backtest_executed = True
except RuntimeError as _e:
    if "Timeout" in str(_e) or "task" in str(_e).lower():
        print(f"Async backtest skipped (Papermill environment): {_e}")
    else:
        raise

print("\nBacktest Results:")
print(f"   Features computed: {len(backtest_strategy.feature_log)}")
print(f"   Predictions made: {len(backtest_strategy.prediction_log)}")
print(f"   Signals generated: {len(backtest_strategy.signal_log)}")
print(f"   Orders submitted: {len(backtest_strategy.order_log)}")
if backtest_strategy.order_log:
    print(f"   Final cash: ${backtest_strategy.order_log[-1]['cash']:,.2f}")
    print(f"   Final positions: {backtest_strategy.order_log[-1]['positions']}")

# %% [markdown]
# **Finding:** The backtest run establishes the reference counts for each stage of the pipeline on a fixed
# synthetic tape.
#
# **Trading implication:** Without a baseline run like this, later live-style discrepancies are hard to
# classify because there is no agreed-upon correct output to compare against.
#
# %% [markdown]
# ## 4. Run Live Pipeline (Simulated)
#
# The live pipeline uses simulated infrastructure so the notebook can isolate framework behavior without
# introducing broker or network noise.

# %%
print("\n" + "=" * 60)
print("LIVE PIPELINE (Simulated)")
print("=" * 60)

live_strategy = VerifiableStrategy(lookback=10, threshold=0.01)


def _temporary_state_path() -> str:
    """Return a non-existent temporary path for SafeBroker state."""
    fd, name = tempfile.mkstemp(prefix="nb08_safe_broker_", suffix=".json")
    os.close(fd)
    Path(name).unlink(missing_ok=True)
    return name


# %% [markdown]
# The live-style broker adds the asynchronous surface required by `SafeBroker` while reusing the
# same fill accounting as the reference broker.


# %%
class LiveBroker(BacktestBroker):
    """Asynchronous adapter around the in-memory reference broker."""

    order_prefix = "LIVE"

    def __init__(self):
        super().__init__()
        self._connected = False

    @property
    def execution_capabilities(self):
        return frozenset()

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    async def is_connected_async(self):
        return self._connected

    @property
    def pending_orders(self):
        return []

    async def get_positions_async(self):
        return self.positions

    async def get_account_value_async(self):
        return self._portfolio.cash

    async def get_cash_async(self):
        return self._portfolio.cash

    async def submit_order_async(self, asset, quantity, side=None, **kwargs):
        return self.submit_order(asset, quantity, side=side, **kwargs)

    async def cancel_order_async(self, order_id):
        return False

    async def close_position_async(self, asset):
        return None


# %% [markdown]
# The live driver adds the risk wrapper and synchronous strategy adapter, then replays the same tape.


# %%
async def run_live():
    """Run the strategy through the live-style wrapper on the fixed tape."""
    broker = LiveBroker()
    state_path = Path(_temporary_state_path())
    risk_config = LiveRiskConfig(
        shadow_mode=True,
        max_position_value=100_000.0,
        max_order_value=50_000.0,
        dedup_window_seconds=0.0,
        state_file=str(state_path),
    )
    safe_broker = SafeBroker(broker, risk_config)
    loop = asyncio.get_running_loop()
    wrapped_broker = ThreadSafeBrokerWrapper(safe_broker, loop)
    feed = TestDataFeed(TEST_DATA)
    try:
        await safe_broker.connect()
        live_strategy.on_start(wrapped_broker)
        await feed.start()
        async for timestamp, data, context in feed:
            for symbol, bar in data.items():
                broker.update_price(symbol, bar["close"])
                safe_broker.record_market_snapshot(symbol, bar["close"], timestamp)
            await asyncio.to_thread(live_strategy.on_data, timestamp, data, context, wrapped_broker)
        live_strategy.on_end(wrapped_broker)
    finally:
        feed.stop()
        await safe_broker.disconnect()
        state_path.unlink(missing_ok=True)


# %%
live_executed = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        run_async(run_live())
    live_executed = True
except RuntimeError as _e:
    if "Timeout" in str(_e) or "task" in str(_e).lower():
        print(f"Async live test skipped (Papermill environment): {_e}")
    else:
        raise

print("\nLive Results:")
print(f"   Features computed: {len(live_strategy.feature_log)}")
print(f"   Predictions made: {len(live_strategy.prediction_log)}")
print(f"   Signals generated: {len(live_strategy.signal_log)}")
print(f"   Orders submitted: {len(live_strategy.order_log)}")
if live_strategy.order_log:
    print(f"   Final cash: ${live_strategy.order_log[-1]['cash']:,.2f}")
    print(f"   Final positions: {live_strategy.order_log[-1]['positions']}")

# %% [markdown]
# **Finding:** The live-style replay produces a second, fully instrumented pipeline trace on the same data.
# That converts backtest-versus-live from an intuition into a measurable comparison.
#
# **Trading implication:** Technical parity should be verified with identical inputs and logged stage outputs,
# not inferred from rough similarity in aggregate returns.
#
# %% [markdown]
# ## 5. Verification Tests
#
# Verification tests compare backtest and live outputs at each stage. They are the release gate that decides
# whether parity is preserved or broken.


# %%
TEST_NAMES = (
    "Feature Count Parity",
    "Feature Value Parity",
    "Prediction Parity",
    "Signal Parity",
    "Order Count Parity",
    "Order Fill Parity",
    "Cash Path Parity",
    "Position Path Parity",
    "Order ID Uniqueness",
)


# %% [markdown]
# A skipped path yields explicit failed records, preserving the reason without promoting the candidate.


# %%
def skipped_results() -> list[VerificationResult]:
    """Build the complete failed result set when either replay did not execute."""
    message = "SKIPPED: one replay did not execute, so parity cannot be evaluated"
    return [
        VerificationResult(
            test_name=name,
            passed=False,
            expected="N/A",
            actual="N/A",
            message=message,
            skipped=True,
            expected_difference=name in EXPECTED_DIFFERENCE,
        )
        for name in TEST_NAMES
    ]


# %% [markdown]
# Count parity prevents a zip-based value comparison from hiding missing records.


# %%
def count_result(name: str, reference: list[dict], candidate: list[dict]) -> VerificationResult:
    """Compare complete record counts for one pipeline stage."""
    return VerificationResult(
        test_name=name,
        passed=len(reference) == len(candidate),
        expected=len(reference),
        actual=len(candidate),
        message=f"Backtest: {len(reference)}, Live: {len(candidate)}",
    )


# %% [markdown]
# Record parity checks identity plus values and uses a tolerance only for named floating fields.


# %%
def matching_records(
    reference: list[dict],
    candidate: list[dict],
    fields: tuple[str, ...],
    float_fields: tuple[str, ...] = (),
) -> int:
    """Count ordered records whose selected fields match."""
    matches = 0
    for left, right in zip(reference, candidate, strict=False):
        same = True
        for field in fields:
            if field in float_fields:
                same &= abs(float(left[field]) - float(right[field])) < 1e-10
            else:
                same &= left[field] == right[field]
        matches += int(same)
    return matches


# %% [markdown]
# The sequence result fails empty or unequal logs, so a vacuous comparison can never pass.


# %%
def sequence_result(
    name: str,
    reference: list[dict],
    candidate: list[dict],
    fields: tuple[str, ...],
    float_fields: tuple[str, ...] = (),
) -> VerificationResult:
    """Compare two complete, ordered stage logs."""
    matches = matching_records(reference, candidate, fields, float_fields)
    complete = bool(reference) and len(reference) == len(candidate)
    ratio = matches / len(reference) if complete else 0.0
    return VerificationResult(
        test_name=name,
        passed=ratio == 1.0,
        expected=1.0,
        actual=ratio,
        message=f"{matches}/{len(reference)} reference records match",
    )


# %% [markdown]
# Each path must also emit unique order identifiers so fills remain auditable.


# %%
def unique_order_ids_result(reference: list[dict], candidate: list[dict]) -> VerificationResult:
    """Require nonempty, unique order IDs independently on both execution paths."""
    reference_ids = [record["order_id"] for record in reference]
    candidate_ids = [record["order_id"] for record in candidate]
    passed = bool(reference_ids) and len(reference_ids) == len(set(reference_ids))
    passed &= bool(candidate_ids) and len(candidate_ids) == len(set(candidate_ids))
    return VerificationResult(
        test_name="Order ID Uniqueness",
        passed=passed,
        expected="unique IDs on both paths",
        actual=f"backtest={reference_ids}; live={candidate_ids}",
        message=f"Backtest: {len(set(reference_ids))}/{len(reference_ids)} unique; "
        f"Live: {len(set(candidate_ids))}/{len(candidate_ids)} unique",
    )


# %% [markdown]
# The first result group covers pre-submit features, predictions, and signals.


# %%
def pipeline_results(bt: VerifiableStrategy, live: VerifiableStrategy) -> list[VerificationResult]:
    """Compare the deterministic decision pipeline before order submission."""
    return [
        count_result("Feature Count Parity", bt.feature_log, live.feature_log),
        sequence_result(
            "Feature Value Parity",
            bt.feature_log,
            live.feature_log,
            ("timestamp", "symbol", "features"),
        ),
        sequence_result(
            "Prediction Parity",
            bt.prediction_log,
            live.prediction_log,
            ("timestamp", "symbol", "prediction"),
            ("prediction",),
        ),
        sequence_result(
            "Signal Parity",
            bt.signal_log,
            live.signal_log,
            ("timestamp", "symbol", "signal"),
        ),
    ]


# %% [markdown]
# The second group checks submitted orders, realized fills, and the broker state path.


# %%
def execution_results(bt: VerifiableStrategy, live: VerifiableStrategy) -> list[VerificationResult]:
    """Compare order fills and post-fill broker state."""
    return [
        count_result("Order Count Parity", bt.order_log, live.order_log),
        sequence_result(
            "Order Fill Parity",
            bt.order_log,
            live.order_log,
            (
                "timestamp",
                "symbol",
                "signal",
                "size",
                "price",
                "status",
                "filled_quantity",
                "filled_price",
            ),
            ("price", "filled_price"),
        ),
        sequence_result("Cash Path Parity", bt.order_log, live.order_log, ("cash",), ("cash",)),
        sequence_result("Position Path Parity", bt.order_log, live.order_log, ("positions",)),
        unique_order_ids_result(bt.order_log, live.order_log),
    ]


# %% [markdown]
# The release gate composes both result groups and fails closed when either replay was skipped.


# %%
def run_verification_tests() -> list[VerificationResult]:
    """Run the complete parity contract."""
    if not (backtest_executed and live_executed):
        return skipped_results()
    return pipeline_results(backtest_strategy, live_strategy) + execution_results(
        backtest_strategy, live_strategy
    )


# %% [markdown]
# The human-readable report shows every check, including skipped and intentionally different stages.


# %%
print("\n" + "=" * 60)
print("VERIFICATION RESULTS")
print("=" * 60)
results = run_verification_tests()


# %% [markdown]
# A compact status label keeps the detailed table readable in notebook and CI output.


# %%
def result_status(result: VerificationResult) -> str:
    if result.skipped:
        return "[SKIP]"
    if result.passed:
        return "[OK] PASS"
    if result.expected_difference:
        return "[EXPECTED] DIFF"
    return "[FAIL] FAIL"


# %%
for result in results:
    print(f"\n{result_status(result)}: {result.test_name}")
    print(f"   {result.message}")
    if not result.passed and not result.skipped:
        print(f"   Expected: {result.expected}")
        print(f"   Actual: {result.actual}")

# %% [markdown]
# **Finding:** The verification block compresses multiple parity questions into explicit pass/fail tests.
# That turns deployment readiness into an artifact a CI system can enforce.
#
# **Trading implication:** If parity checks are not automated, teams eventually skip them under time pressure,
# and the backtest-live gap reappears as an operational surprise.
#
# %% [markdown]
# ## 6. Regression Test Output
#
# This summary is formatted for CI integration so the notebook can act like a reproducible deployment gate,
# not just a narrative walkthrough.

# %%
print("\n" + "=" * 60)
print("CI REGRESSION TEST SUMMARY")
print("=" * 60)

# The CI gate counts only tests that ran AND were not in EXPECTED_DIFFERENCE.
# Skipped tests and explicitly contracted differences do not consume a
# pass/fail slot. This harness currently expects no differences.
gate_results = [r for r in results if not r.skipped and not r.expected_difference]
passed = sum(1 for r in gate_results if r.passed)
failed = sum(1 for r in gate_results if not r.passed)
skipped = sum(1 for r in results if r.skipped)
expected_diff = sum(1 for r in results if r.expected_difference)

print(f"\nGate tests passed:  {passed}/{len(gate_results)}")
print(f"Gate tests failed:  {failed}/{len(gate_results)}")
print(f"Expected differences (informational): {expected_diff}")
print(f"Skipped:           {skipped}")

if skipped == len(results):
    print("\n[FAIL] LIVE PIPELINE NOT EXECUTED")
    print("   Parity could not be evaluated; this candidate cannot be promoted")
    exit_code = 1
elif failed == 0 and len(gate_results) > 0:
    print("\n[OK] ALL GATED VERIFICATION TESTS PASSED")
    print("   Technical parity confirmed between backtest and live pipelines")
    exit_code = 0
else:
    print("\n[FAIL] VERIFICATION FAILED")
    print("   Investigate failing tests before deploying live")
    for r in gate_results:
        if not r.passed:
            print(f"   - {r.test_name}: {r.message}")
    exit_code = 1

print(f"\nExit code: {exit_code}")

# %% [markdown]
# **Finding:** The CI-style summary translates notebook results into a machine-readable release gate.
#
# **Trading implication:** Production deployment should promote only code paths that can express parity
# success or failure unambiguously to automated tooling.
#
# %% [markdown]
# ## Pytest Assertion Pattern
#
# A library test suite would assert the same parity conditions directly. The
# pattern below mirrors what `tests/live/test_parity.py` would carry in a
# project that promotes this notebook into a CI harness.


# %%
def test_parity_gate(results: list[VerificationResult]) -> None:
    """Pytest assertion pattern for the parity gate.

    A skipped pipeline blocks promotion because parity was not evaluated.
    EXPECTED_DIFFERENCE results are informational. Only gated results must pass.
    """
    if all(r.skipped for r in results):
        raise AssertionError("Live pipeline did not execute; parity gate is incomplete")

    gated = [r for r in results if not r.skipped and not r.expected_difference]
    failed = [r for r in gated if not r.passed]
    assert not failed, "Parity gate failed: " + ", ".join(r.test_name for r in failed)


if __name__ == "__main__":
    test_parity_gate(results)

# %%
print("\n" + "=" * 60)
print("PIPELINE VERIFICATION COMPLETE")
print("=" * 60)
if skipped == len(results):
    print("Result: [FAIL] live pipeline did not execute")
elif failed == 0:
    print(
        f"Result: [OK] {passed}/{len(gate_results)} gated tests passed; "
        f"{expected_diff} expected differences"
    )
else:
    print(f"Result: [FAIL] {failed}/{len(gate_results)} gated tests failed")

# %% [markdown]
# ## Key Takeaways
#
# - **Four parity stages catch distinct classes of bug.** Feature parity flags
#   data ordering and float ordering issues; prediction parity catches model
#   versioning or preprocessing drift; signal parity catches threshold or
#   position-state bugs; order parity catches sizing and rounding gaps.
# - **A mismatch is a failure until a separate contract proves otherwise.**
#   Both paths consume the same tape, so feature counts, identities, and values
#   must match exactly; the harness does not waive warm-up differences.
# - **Skipped is not passed.** When the live pipeline cannot execute (Papermill
#   async constraint), the harness reports `FAIL` rather than passing parity
#   tests against an empty live log. The candidate remains blocked until both
#   paths execute.
# - **Determinism must still exercise the decision path.** Static per-symbol
#   seed offsets and a fixed two-regime tape reproduce across processes while
#   forcing non-vacuous BUY and SELL comparisons.
#
# **Next:** Extend the same parity harness to real feature pipelines and saved
# model artifacts; rerun whenever broker wrappers, sizing logic, or
# preprocessing code changes.
