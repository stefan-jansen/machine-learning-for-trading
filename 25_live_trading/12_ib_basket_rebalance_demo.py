# ---
# jupyter:
#   jupytext:
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
# # Interactive Brokers Basket Rebalance Demo
#
# **Chapter 25: Live Trading Systems**
# **Section**: 25.2 (Interactive Brokers Integration)
#
# **Docker image**: `ml4t` (requires IB TWS/Gateway running on host port 7497)
#
# This notebook extends `03_ib_paper_trading_demo` from a single-order workflow to a full daily-rebalance
# workflow against a fixed 20-name US large-cap universe. Publication execution is planning-only;
# the opt-in paper path demonstrates three mechanics that a single-order
# demo cannot:
#
# 1. **Reconciliation.** Compare current IB positions against model-target positions and compute the
#    order basket required to close the gap. In production this is the *only* place position drift gets
#    corrected — without reconciliation, every divergence compounds.
# 2. **Basket planning and optional submission.** Build the complete delta basket. Only an explicitly
#    authorized paper run routes it concurrently through `SafeBroker`.
# 3. **Post-fill attribution contract.** The optional paper path compares fills with the last completed
#    close. Publication mode reports planned notional and makes no broker mutation.
#
# **Universe**: 20 fixed US large-cap names (AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, V, JNJ, PG, UNH, HD,
# DIS, MA, BAC, KO, PEP, XOM, CVX, WMT).
#
# **Cross-References**
# - Chapter 25.2: Interactive Brokers integration overview
# - Chapter 25.5: Order lifecycle management (state machine used by `SafeBroker`)
# - Chapter 25.7: Operational readiness (kill switch, risk limits wrapped around the basket)
#
# **Learning Objectives**
# - Diff current broker positions against model targets and express the gap as a concrete order list.
# - Inspect a no-mutation order plan with per-order risk inputs intact.
# - Understand how an explicitly authorized paper run reconciles fills and estimates execution cost.
#
# **Prerequisites**
# - Chapter 25.2 for IB connectivity; Chapter 25.5 for the order state machine.
# - IB TWS or Gateway running with paper-account API access enabled (required — the notebook fails
#   loudly when no session is reachable, never substitutes a mock).

# %% [markdown]
# ## 1. Setup and Configuration
#
# The parameters cell fixes the deployment surface: which host/port to reach, how much notional to deploy
# per rebalance, and how many names the long leg spans. Production overrides flow through Papermill.

# %%
"""IB Basket Rebalance Demo — daily rebalance of a 20-name US large-cap universe via IB paper."""

import asyncio
import hashlib
import json
import logging
import warnings
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import polars as pl
from async_utils import run_async
from ib_async import Stock
from ml4t.backtest import OrderSide, OrderType
from ml4t.live import LiveRiskConfig, SafeBroker
from ml4t.live.brokers.ib import IBBroker

from utils.paths import display_path, get_output_dir
from utils.style import COLORS, add_message_title


def run_demo(awaitable):
    """Run an async demo while suppressing only nest_asyncio's Python 3.14 deprecation."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nest_asyncio")
        return run_async(awaitable)


# %% tags=["parameters"]
IB_HOST = "127.0.0.1"
IB_PORT = 7497
CLIENT_ID = 12
ACCOUNT = ""
UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "V",
    "JNJ",
    "PG",
    "UNH",
    "HD",
    "DIS",
    "MA",
    "BAC",
    "KO",
    "PEP",
    "XOM",
    "CVX",
    "WMT",
]
TOP_K_LONG = 5
TARGET_NOTIONAL_USD = 50_000
MAX_POSITION_USD = 15_000
MAX_ORDER_USD = 12_000
MAX_DAILY_LOSS_USD = 5_000
WARMUP_DAYS = 60
OUTSIDE_RTH = False  # set True to permit extended-hours execution on market orders
# IB market-data type: None leaves TWS at its configured default (use this if
# the paper account has live Level 1 subscriptions). Paper accounts without
# market-data subscriptions reject MARKET orders with "No market data
# available..." — set this to 3 (delayed) to fall back to delayed quotes.
# 1=real-time, 2=frozen, 3=delayed, 4=delayed-frozen.
MARKET_DATA_TYPE: int | None = 3
SUBMIT_PAPER_ORDERS = False  # explicit opt-in only; publication execution is read-only

# %%
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("basket_demo")
logging.getLogger("ib_async").setLevel(logging.WARNING)
logging.getLogger("ml4t.live.brokers.ib").setLevel(logging.WARNING)

print("=" * 60)
print("IB BASKET REBALANCE DEMO")
print("=" * 60)
print(f"Universe: {len(UNIVERSE)} names ({', '.join(UNIVERSE[:5])}, ...)")
print(f"Long leg: top {TOP_K_LONG} by signal")
print(f"Target notional per leg: ${TARGET_NOTIONAL_USD:,}")
print(f"Position cap: ${MAX_POSITION_USD:,}  |  Order cap: ${MAX_ORDER_USD:,}")
print(f"Host/port: {IB_HOST}:{IB_PORT}  client_id={CLIENT_ID}")
print(f"Order submission: {'ENABLED' if SUBMIT_PAPER_ORDERS else 'DISABLED (planning only)'}")

# %% [markdown]
# **Finding:** The configuration banner makes the deployment envelope explicit. Position and order caps are
# risk controls, not guidelines — they are enforced by `SafeBroker`, so exceeding them raises before the
# order reaches IB.

# %% [markdown]
# ## 2. Connect to IB Paper Session
#
# The notebook requires a reachable IB paper session — there is no silent fallback. If TWS or IB Gateway
# is not running on the configured host and port, the cell prints actionable setup instructions and exits.
# Demos that quietly substitute mock state when the real broker is missing are inconsistent with the
# operational discipline this chapter is teaching.

# %%
OUTPUT_DIR = get_output_dir(25, "ib_basket_rebalance_demo")
STATE_FILE = OUTPUT_DIR / "risk_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# The connection gate rejects unreachable services and any managed account that is not an IB paper account.


# %%
async def open_ib_session() -> IBBroker:
    """Connect to the IB paper session. Raises on failure with actionable instructions."""
    broker = IBBroker(
        host=IB_HOST,
        port=IB_PORT,
        client_id=CLIENT_ID,
        account=ACCOUNT or None,
        market_data_type=MARKET_DATA_TYPE,
    )
    try:
        await asyncio.wait_for(broker.connect(), timeout=10.0)
    except (TimeoutError, ConnectionRefusedError, OSError) as exc:
        print()
        print("=" * 60)
        print("ERROR: IB paper session unreachable")
        print("=" * 60)
        print(f"Could not connect to {IB_HOST}:{IB_PORT} (client_id={CLIENT_ID})")
        print(f"Underlying error: {type(exc).__name__}: {exc}")
        print()
        print("Setup checklist:")
        print("  1. Start TWS or IB Gateway and log into a paper account.")
        print("  2. Edit -> Global Configuration -> API -> Settings:")
        print("     - 'Enable ActiveX and Socket Clients' must be checked.")
        print(f"     - Socket port must be {IB_PORT} (TWS paper=7497, Gateway paper=4002).")
        print("     - 127.0.0.1 must be in 'Trusted IPs', or unchecked 'Read-Only API'.")
        print(f"  3. Confirm no other client is using client_id={CLIENT_ID}.")
        print()
        print("Re-run this notebook once TWS is reachable.")
        raise SystemExit(1) from exc
    accounts = [str(account) for account in broker.ib.managedAccounts()]
    if not accounts or any(not account.upper().startswith("DU") for account in accounts):
        await broker.disconnect()
        raise RuntimeError(
            "Refusing non-paper IB session: every managed account must start with 'DU'"
        )
    logger.info("Connected to IB paper session at %s:%s", IB_HOST, IB_PORT)
    return broker


# %%
broker = run_demo(open_ib_session())
print(f"\nVerified IB paper session connected at {IB_HOST}:{IB_PORT}")

# %% [markdown]
# **Finding:** Reaching this line means the paper session is real. Every subsequent order, position
# snapshot, and reconciliation diff reflects the broker's authoritative state, not an in-process
# simulation.

# %% [markdown]
# ## 3. Wrap in SafeBroker and Reconcile Persisted State
#
# `SafeBroker` enforces position and order caps, persists kill-switch state across runs, and — on
# `connect()` — diffs the persisted snapshot from the previous session against the broker's current
# positions and pending orders. A non-clean report means something changed between sessions
# (uncleared after-hours order, manual close, partial fill landed after the last persist), and the
# notebook must stop and let the operator investigate before submitting a basket.

# %%
risk_config = LiveRiskConfig(
    max_position_value=MAX_POSITION_USD,
    max_order_value=MAX_ORDER_USD,
    max_daily_loss=MAX_DAILY_LOSS_USD,
    max_data_staleness_seconds=120,
    shadow_mode=not SUBMIT_PAPER_ORDERS,
    state_file=str(STATE_FILE),
)
safe_broker = SafeBroker(broker, risk_config)
run_demo(safe_broker.connect())

report = safe_broker.reconciliation_report
if report is None or not report["clean"]:
    print()
    print("=" * 60)
    print("ERROR: Startup reconciliation is not clean")
    print("=" * 60)
    print(f"State file: {STATE_FILE}")
    if report is not None:
        for key in (
            "missing_positions",
            "unexpected_positions",
            "quantity_mismatches",
            "missing_pending_orders",
            "unexpected_pending_orders",
        ):
            value = report.get(key)
            if value:
                print(f"  {key}: {value}")
    print()
    print("Investigate the divergence in TWS before re-running the basket loop.")
    print(f"If the divergence is expected (manual flatten, etc.), delete {STATE_FILE} to reset.")
    raise SystemExit(1)

print(
    f"SafeBroker configured with max_position=${MAX_POSITION_USD:,}, "
    f"max_order=${MAX_ORDER_USD:,}, max_daily_loss=${MAX_DAILY_LOSS_USD:,}"
)
print(f"Reconciliation: clean (state file {display_path(STATE_FILE)})")

# %% [markdown]
# **Finding:** `SafeBroker.connect()` is the single place where stale-session damage gets caught.
# A previous run that left orders pending after-hours, a manual flatten in the GUI, or a partial fill
# that landed after the last persist all show up as a non-clean reconciliation report — the notebook
# refuses to launch the basket until the operator resolves the divergence. The position and order
# caps that follow apply uniformly because every leg flows through `safe_broker`.

# %% [markdown]
# ## 4. Fetch Warmup Bars from IB
#
# The rebalance ranks names against a 20-day momentum signal, so the notebook needs roughly three months
# of recent daily history. A live-trading notebook must source warmup bars from the same broker session
# that will execute the orders — using a research-time loader risks ranking names on stale prices and
# computing position sizes against historical levels that no longer reflect the live tape. We pull
# `WARMUP_DAYS` of daily bars per universe symbol via `reqHistoricalDataAsync`, parallelising the 20
# requests through `asyncio.gather`.


# %%
IB_BAR_AUDIT: list[dict] = []


# %% [markdown]
# Raw IB bars are hashed and counted before parsing so the broker-data boundary is reproducible.


# %%
def audit_ib_bars(symbol: str, bars: list) -> dict:
    """Return raw-record identity and duplicate-date diagnostics."""
    canonical = [
        {
            "date": str(bar.date),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume or 0),
        }
        for bar in bars
    ]
    dates = [row["date"] for row in canonical]
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    return {
        "symbol": symbol,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_rows": len(canonical),
        "duplicate_dates": len(dates) - len(set(dates)),
    }


# %% [markdown]
# Each symbol parser preserves identity and converts one raw IB bar to one canonical row.


# %%
async def fetch_one_symbol_bars(ib_app: object, symbol: str, days: int) -> list[dict]:
    """Return daily OHLCV bars for `symbol` from IB. Returns [] when IB has no data for the contract."""
    contract = Stock(symbol, "SMART", "USD")
    qualified = await ib_app.qualifyContractsAsync(contract)
    if not qualified:
        logger.warning("IB could not qualify %s; skipping", symbol)
        return []
    raw_bars = await ib_app.reqHistoricalDataAsync(
        qualified[0],
        endDateTime="",
        durationStr=f"{days} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        # `timeout=0` disables ib_async's internal `asyncio.wait_for` wrapper,
        # which fails under nest_asyncio on Python 3.14. The IB request keeps
        # its own network timeout.
        timeout=0,
    )
    audit = audit_ib_bars(symbol, raw_bars)
    rows = []
    for bar in raw_bars:
        ts = bar.date
        if not isinstance(ts, datetime):
            ts = datetime.combine(ts, datetime.min.time())
        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume or 0),
            }
        )
    audit["parsed_rows"] = len(rows)
    IB_BAR_AUDIT.append(audit)
    return rows


# %% [markdown]
# Fan the per-symbol fetch out across the universe via `asyncio.gather`,
# then concatenate the rows into a single OHLCV panel for downstream use.


# %%
async def fetch_warmup_bars(ib_app: object, universe: list[str], days: int) -> pl.DataFrame:
    """Pull daily bars for every name in `universe` from IB in parallel."""
    per_symbol = await asyncio.gather(*[fetch_one_symbol_bars(ib_app, s, days) for s in universe])
    rows = [r for symbol_rows in per_symbol for r in symbol_rows]
    if not rows:
        raise RuntimeError(
            "IB returned zero bars for the entire universe — check market-data subscriptions"
        )
    return pl.DataFrame(rows).sort(["symbol", "timestamp"])


# %%
bars = run_demo(fetch_warmup_bars(broker.ib, UNIVERSE, WARMUP_DAYS))
ib_audit_frame = pl.DataFrame(IB_BAR_AUDIT)
if len(ib_audit_frame) == 0:
    raise RuntimeError("No IB bar payload reached the parser audit")
assert (ib_audit_frame["raw_rows"] == ib_audit_frame["parsed_rows"]).all()
assert ib_audit_frame["duplicate_dates"].sum() == 0


# %% [markdown]
# The decision panel excludes the current UTC date and reconciles that exclusion against raw counts.


# %%
bars = bars.filter(pl.col("timestamp").dt.date() < datetime.now(UTC).date())
if len(bars) == 0:
    raise RuntimeError("IB returned no completed daily bars before the current UTC date")

completed_counts = bars.group_by("symbol").len().rename({"len": "completed_rows"})
ib_audit_frame = (
    ib_audit_frame.join(completed_counts, on="symbol", how="left")
    .with_columns(pl.col("completed_rows").fill_null(0))
    .with_columns(current_or_future_rows_excluded=pl.col("parsed_rows") - pl.col("completed_rows"))
)
assert (ib_audit_frame["current_or_future_rows_excluded"] >= 0).all()
print(
    ib_audit_frame.select(
        "symbol",
        "payload_sha256",
        "raw_rows",
        "completed_rows",
        "current_or_future_rows_excluded",
    )
)

print(
    f"\nWarmup bars: {len(bars):,} rows over {bars['symbol'].n_unique()} symbols (last close: {bars['timestamp'].max().date()})"
)
latest_close = (
    bars.group_by("symbol")
    .agg(
        pl.col("close").last().alias("last_close"),
        pl.col("timestamp").last().alias("timestamp"),
    )
    .sort("symbol")
)
print(f"Latest close table: {len(latest_close)} symbols")

# %% [markdown]
# ## 5. Signal Computation and Target Basket
#
# The signal is a simple 20-day momentum proxy — adequate for a live-mechanics demonstration, even though
# Chapter 6 establishes that naïve momentum on its own is insufficient as a research-grade factor. The
# target basket longs the top `TOP_K_LONG` by signal and leaves everything else flat.


# %%
def compute_target_basket(bars_frame: pl.DataFrame, top_k: int, notional: float) -> pl.DataFrame:
    """Return a frame with `symbol`, `signal`, `target_qty` for the current rebalance."""
    log_close = pl.col("close").log()
    with_momentum = bars_frame.with_columns(
        momentum_20d=(log_close - log_close.shift(20)).over("symbol"),
        last_close=pl.col("close").last().over("symbol"),
    )
    latest = (
        with_momentum.group_by("symbol", maintain_order=True)
        .agg(
            pl.col("momentum_20d").last().alias("signal"),
            pl.col("last_close").last().alias("last_close"),
        )
        .sort("signal", descending=True)
    )
    longs = latest.head(top_k).with_columns(
        target_notional=pl.lit(notional / top_k),
        target_qty=(pl.lit(notional / top_k) / pl.col("last_close")).round(0).cast(pl.Float64),
    )
    flats = latest.tail(len(latest) - top_k).with_columns(
        target_notional=pl.lit(0.0), target_qty=pl.lit(0.0)
    )
    return pl.concat([longs, flats], how="vertical").sort("signal", descending=True)


target = compute_target_basket(bars, TOP_K_LONG, TARGET_NOTIONAL_USD)
print("\nTarget basket (top 5 by signal):")
target.head(TOP_K_LONG)

# %% [markdown]
# The ranked momentum view shows exactly why each name enters or misses the
# target basket before any broker mutation is possible.

# %%
plot_target = target.sort("signal")
target_symbols = set(target.head(TOP_K_LONG)["symbol"].to_list())
bar_colors = [
    COLORS["positive"] if symbol in target_symbols else COLORS["neutral"]
    for symbol in plot_target["symbol"].to_list()
]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(
    plot_target["symbol"].to_list(),
    plot_target["signal"].to_list(),
    color=bar_colors,
)
ax.axvline(0, linestyle="--", color=COLORS["neutral"], linewidth=1)
ax.set(xlabel="20-day log return (decimal)", ylabel="US equity")
add_message_title(
    ax,
    "The strongest 20-day momentum names form the target basket",
    subtitle="Completed IB daily bars; green identifies the five planned long positions",
)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Finding:** Printing the head of the target basket before any order is sent makes the intended
# portfolio state auditable. In production the same table is logged and stored as the rebalance intent,
# so that post-fill state can be compared against it.

# %% [markdown]
# ## 6. Reconciliation: Current Positions versus Target
#
# Reconciliation is the step that distinguishes a real live-trading loop from a fire-and-forget signal
# pipeline. The `current` frame comes from the broker; the `target` frame comes from the model. The
# `delta` frame is the order basket required to move from one to the other.


# %%
async def fetch_current_positions(active_broker: object, universe: list[str]) -> pl.DataFrame:
    """Return current broker positions as a frame keyed by symbol, with every universe name represented."""
    raw = await active_broker.get_positions_async()
    rows = []
    for sym in universe:
        pos = raw.get(sym)
        rows.append(
            {
                "symbol": sym,
                "current_qty": float(pos.quantity) if pos else 0.0,
                "entry_price": float(pos.entry_price) if pos else 0.0,
            }
        )
    return pl.DataFrame(rows)


current_positions = run_demo(fetch_current_positions(broker, UNIVERSE))

# %% [markdown]
# `reconcile` joins current and target positions and emits a `delta_qty`
# column — the per-symbol order instruction that drives the basket
# submission step.


# %%
def reconcile(current: pl.DataFrame, target_frame: pl.DataFrame) -> pl.DataFrame:
    """Join current and target positions, emit a `delta_qty` column as the order instruction per symbol."""
    joined = current.join(target_frame, on="symbol", how="full", coalesce=True).with_columns(
        pl.col("target_qty").fill_null(0.0),
        pl.col("current_qty").fill_null(0.0),
    )
    return joined.with_columns(
        delta_qty=pl.col("target_qty") - pl.col("current_qty"),
    ).filter(pl.col("delta_qty").abs() > 0)


orders_needed = reconcile(current_positions, target)
print(f"\nReconciliation: {len(orders_needed)} orders needed to reach target")
print(orders_needed.select(["symbol", "current_qty", "target_qty", "delta_qty", "last_close"]))

# %% [markdown]
# **Finding:** The delta table is the audit surface. Each row justifies exactly one order; rows that
# vanish (delta already zero) are implicit parity confirmations. A production run would checkpoint this
# table before any submission, so a mid-rebalance crash can resume from the same intent.

# %% [markdown]
# ## 6b. Cache Reference Prices for SafeBroker
#
# `SafeBroker.submit_order_async` rejects orders when no recent price snapshot is cached for the asset.
# We seed the cache with the latest completed daily close per symbol, using the same
# `reqHistoricalDataAsync` data that fed the signal in section 4. The notebook excludes the current
# UTC date before selecting these observations, so an unfinished daily bar cannot leak into sizing.
#
# A continuous-loop deployment would replace this seed with a streaming `IBDataFeed` driven by
# `LiveEngine`, so every rebalance is preceded by a continuously refreshed quote. For this one-shot
# planning demonstration, the completed close provides a reproducible reference for the safety checks.

# %%
for row in latest_close.iter_rows(named=True):
    safe_broker.record_market_snapshot(
        row["symbol"], float(row["last_close"]), timestamp=row["timestamp"]
    )
print(f"Cached {len(latest_close)} IB-sourced reference prices for SafeBroker staleness guard")

# %% [markdown]
# ## 7. Basket Submission
#
# Submitting the basket through `asyncio.gather` parallelises I/O without changing the per-order
# semantics. Every leg still flows through `SafeBroker`, so caps and the kill switch apply uniformly.


# %%
def dry_run_basket(basket: pl.DataFrame) -> list[dict]:
    """Convert order deltas to auditable intents without broker calls."""
    return [
        {
            "symbol": row["symbol"],
            "side": "buy" if row["delta_qty"] > 0 else "sell",
            "quantity": abs(row["delta_qty"]),
            "delta_qty": row["delta_qty"],
            "order_id": None,
            "last_close": row["last_close"],
            "status": "dry_run",
        }
        for row in basket.to_dicts()
    ]


# %% [markdown]
# One submitted leg carries its close as a risk-check hint; IB still receives a market order.


# %%
async def submit_one_basket_order(active_broker: SafeBroker, row: dict) -> dict:
    """Submit one delta through `SafeBroker` and return its audit record."""
    qty = abs(row["delta_qty"])
    side = OrderSide.BUY if row["delta_qty"] > 0 else OrderSide.SELL
    record = {
        "symbol": row["symbol"],
        "side": side.value,
        "quantity": qty,
        "delta_qty": row["delta_qty"],
        "order_id": None,
        "last_close": row["last_close"],
    }
    try:
        order = await active_broker.submit_order_async(
            asset=row["symbol"],
            quantity=qty,
            side=side,
            order_type=OrderType.MARKET,
            limit_price=float(row["last_close"]),
            outsideRth=OUTSIDE_RTH,
        )
        record.update(status="submitted", order_id=getattr(order, "order_id", None))
    except Exception as exc:  # noqa: BLE001
        record["status"] = f"rejected: {exc}"
    return record


# %% [markdown]
# Basket submission is an explicit switch: publication mode returns intents, while an authorized paper
# run submits all legs concurrently through the risk wrapper.


# %%
async def submit_basket(active_broker: SafeBroker, basket: pl.DataFrame) -> list[dict]:
    """Plan or submit every non-zero rebalance delta."""
    if not SUBMIT_PAPER_ORDERS:
        return dry_run_basket(basket)
    tasks = [submit_one_basket_order(active_broker, row) for row in basket.to_dicts()]
    return await asyncio.gather(*tasks)


# %%
submissions = run_demo(submit_basket(safe_broker, orders_needed))
submissions_frame = pl.DataFrame(submissions) if submissions else pl.DataFrame()
if not SUBMIT_PAPER_ORDERS:
    assert all(record["status"] == "dry_run" for record in submissions)
label = "submitted" if SUBMIT_PAPER_ORDERS else "planned"
print(f"\nBasket {label}: {len(submissions_frame)} orders")
if len(submissions_frame):
    print(submissions_frame)

# %% [markdown]
# **Finding:** The fills frame is the first place an execution problem becomes visible — a row with a
# `rejected` status or a large gap between `fill_price` and `last_close` is a signal the operator must
# investigate before the next rebalance.

# %% [markdown]
# ## 8. Wait for Fills and Reconcile Post-Submission State
#
# `submit_order_async` returns as soon as the order is queued; fills arrive asynchronously via IB
# callbacks. A short sleep lets routing complete, after which re-fetching positions reflects the
# executed basket. A residual delta after this wait indicates an unfilled or rejected order — exactly
# what the operator needs to see before the next rebalance cycle.


# %%
async def wait_for_basket_to_settle(
    active_broker: object, expected_symbols: list[str], timeout_s: float = 15.0
) -> dict:
    """Poll the broker for position updates until every expected symbol is reflected, or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if hasattr(active_broker, "_sync_positions"):
            await active_broker._sync_positions()
        positions = await active_broker.get_positions_async()
        present = {s for s, p in positions.items() if p.quantity != 0}
        missing = [s for s in expected_symbols if s not in present]
        if not missing:
            return positions
        await asyncio.sleep(1.0)
    return await active_broker.get_positions_async()


if SUBMIT_PAPER_ORDERS:
    expected_long = orders_needed.filter(pl.col("delta_qty") > 0)["symbol"].to_list()
    raw_post = run_demo(wait_for_basket_to_settle(broker, expected_long, timeout_s=15.0))
    post_positions = run_demo(fetch_current_positions(broker, UNIVERSE))
    post_delta = reconcile(post_positions, target)
    print(f"\nPost-submission residual deltas: {len(post_delta)}")
    if len(post_delta):
        print(post_delta.select(["symbol", "current_qty", "target_qty", "delta_qty"]))
    else:
        print("All positions reconciled to target.")
else:
    raw_post = run_demo(broker.get_positions_async())
    post_positions = current_positions
    post_delta = orders_needed
    print("\nPost-submission reconciliation skipped: planning mode made no broker mutations.")

# %% [markdown]
# **Finding:** A non-empty post-submission residual is not automatically a bug — it can also mean a
# kill-switch or risk-cap triggered mid-basket and blocked a leg. Distinguishing the two requires the
# per-order status from step 7, which is why both are printed side by side.

# %% [markdown]
# ## 9. Execution Cost and Expected-versus-Realised P\&L
#
# The final check estimates execution cost per leg and aggregates it against the notional deployed. In
# production this number feeds Chapter 26 (MLOps and Governance) — sustained drift between `fill_price`
# and `last_close` is the first signal that venue or router assumptions have changed.


# %%
def execution_summary(submissions: pl.DataFrame, raw_positions: dict) -> dict:
    """Cross-reference submitted orders with post-fill positions and report slippage vs last_close.

    For a new long position, the post-fill `entry_price` is the weighted-average fill price for that
    symbol. We compare it to the `last_close` at submission time to estimate intraday slippage in bps.
    """
    if not len(submissions):
        return {"orders": 0, "notional": 0.0, "slippage_bps": 0.0, "buys": 0, "sells": 0}

    def fill_price_for(symbol: str) -> float | None:
        pos = raw_positions.get(symbol)
        return float(pos.entry_price) if pos and pos.quantity != 0 else None

    enriched = submissions.with_columns(
        fill_price=pl.col("symbol").map_elements(fill_price_for, return_dtype=pl.Float64),
    ).with_columns(
        slippage_bps=(
            (pl.col("fill_price") - pl.col("last_close")) / pl.col("last_close") * 10_000.0
        ).cast(pl.Float64),
        notional=(pl.col("fill_price") * pl.col("quantity")).cast(pl.Float64),
    )
    filled = enriched.filter(pl.col("fill_price").is_not_null())
    total_notional = float(filled["notional"].sum()) if len(filled) else 0.0
    vwap_slip = (
        float((filled["slippage_bps"] * filled["notional"]).sum() / total_notional)
        if total_notional
        else 0.0
    )
    return {
        "orders_submitted": len(submissions),
        "orders_filled": len(filled),
        "notional": total_notional,
        "slippage_bps_vwap": vwap_slip,
        "buys": int((submissions["side"] == "buy").sum()),
        "sells": int((submissions["side"] == "sell").sum()),
    }


# %%
if SUBMIT_PAPER_ORDERS:
    summary = execution_summary(submissions_frame, raw_post)
else:
    summary = {
        "orders_planned": len(submissions_frame),
        "broker_mutations": 0,
        "planned_notional": (
            float((submissions_frame["quantity"] * submissions_frame["last_close"]).sum())
            if len(submissions_frame)
            else 0.0
        ),
    }
print("\nExecution summary:")
for key, value in summary.items():
    if key in {"notional", "planned_notional"}:
        print(f"  {key:18s}: ${value:,.2f}")
    elif key == "slippage_bps_vwap":
        print(f"  {key:18s}: {value:+.2f} bps")
    else:
        print(f"  {key:18s}: {value}")

# %% [markdown]
# ## 10. End-of-Day Flatten via Market-On-Close
#
# A daily basket strategy that opens positions at start of day and re-evaluates
# at the next rebalance leaves overnight exposure unless the operator explicitly
# closes out. The clean primitive for this is a Market-On-Close (MOC) order:
# IB queues it for the closing auction and targets the official session close.
# Exchange cutoffs, halts, and broker rejection still prevent any guarantee.
#
# This cell submits opposing-side MOC orders only for positions opened by this
# notebook run. Existing account positions and orders remain outside its
# mutation boundary. In a continuous
# loop deployment, the same MOC step would be conditional on the next-rebalance
# decision (hold overnight if the signal still ranks the name long, flatten via
# MOC otherwise). The operator must use the current broker and exchange cutoff;
# late orders may be rejected.


# %% [markdown]
# `submit_eod_flatten` issues one MOC order per position opened by this run. The
# MOC primitive routes each leg to the closing auction; the side is
# opposite the current position.


# %%
async def submit_eod_flatten(
    active_broker: SafeBroker, symbols_with_qty: list[tuple[str, float]]
) -> list[dict]:
    """Submit one MOC order per non-zero position to flatten via the closing auction."""

    async def one_flatten(symbol: str, qty: float) -> dict:
        side = OrderSide.SELL if qty > 0 else OrderSide.BUY
        try:
            order = await active_broker.submit_order_async(
                asset=symbol,
                quantity=abs(qty),
                side=side,
                order_type=OrderType.MOC,
            )
            return {
                "symbol": symbol,
                "side": side.value,
                "quantity": abs(qty),
                "order_id": getattr(order, "order_id", None),
                "status": "submitted",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "symbol": symbol,
                "side": side.value,
                "quantity": abs(qty),
                "order_id": None,
                "status": f"rejected: {exc}",
            }

    tasks = [one_flatten(s, q) for s, q in symbols_with_qty]
    return await asyncio.gather(*tasks)


# %% [markdown]
# `confirm_orders_accepted` waits for IB to acknowledge every MOC order
# before disconnect. Without this gate, a fast teardown can race the
# acceptance and queued orders never reach the closing-auction queue.


# %%
async def confirm_orders_accepted(
    active_ib: object, expected_count: int, timeout_s: float = 5.0
) -> list[dict]:
    """Wait until IB acknowledges all expected open orders (status != 'PendingSubmit')."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        trades = active_ib.openTrades()
        statuses = [t.orderStatus.status for t in trades]
        if len(trades) >= expected_count and all(s != "PendingSubmit" for s in statuses):
            break
        await asyncio.sleep(0.5)
    return [
        {
            "symbol": t.contract.symbol,
            "order_type": t.order.orderType,
            "tif": t.order.tif,
            "ib_status": t.orderStatus.status,
        }
        for t in active_ib.openTrades()
    ]


# %%
if SUBMIT_PAPER_ORDERS:
    opened_symbols = {
        row["symbol"]
        for row in submissions
        if row["status"] == "submitted" and row["delta_qty"] > 0
    }
    post_flatten_positions = run_demo(fetch_current_positions(broker, UNIVERSE))
    to_flatten = [
        (row["symbol"], row["current_qty"])
        for row in post_flatten_positions.iter_rows(named=True)
        if row["symbol"] in opened_symbols and row["current_qty"] != 0
    ]
    if to_flatten:
        flatten_records = run_demo(submit_eod_flatten(safe_broker, to_flatten))
        flatten_frame = pl.DataFrame(flatten_records)
        print(f"\nEnd-of-day flatten: submitted {len(flatten_frame)} MOC order(s)")
        print(flatten_frame)

        ack = run_demo(confirm_orders_accepted(broker.ib, expected_count=len(to_flatten)))
        print(f"\nIB acknowledgement ({len(ack)} orders open at IB):")
        for row in ack:
            print(
                f"  {row['symbol']}: orderType={row['order_type']} "
                f"tif={row['tif']} status={row['ib_status']}"
            )
    else:
        print("\nEnd-of-day flatten: this run opened no positions.")
else:
    print("\nEnd-of-day flatten skipped: planning mode created no positions or orders.")

# %% [markdown]
# **Finding:** The MOC primitive turns end-of-day flattening into a single library call
# per leg. Without it, a basket strategy must either submit market orders ahead of the
# 16:00 ET clock or accept overnight exposure. The MOC path is explicit and
# auditable, but exchange acceptance and the broker acknowledgement remain gates.
#
# **Operator note:** TWS's `Global Configuration → API → Settings → Auto-cancel API
# orders on disconnect` defaults to *enabled* on most paper accounts. With that
# setting, IB cancels these MOC orders the moment the notebook disconnects, defeating
# the close-out. An operator who explicitly enables paper submission must reconcile
# that account-level setting before relying on queued DAY orders.

# %% [markdown]
# ## 11. Teardown

# %%
run_demo(broker.disconnect())
print(f"\nIB paper session closed at {datetime.now(UTC).isoformat(timespec='seconds')}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Reconciliation is the loop.** A daily rebalance is not "submit these orders"; it is "diff current
#    against target, then submit only the delta." The diff table is the audit artefact.
# 2. **Basket I/O belongs in `asyncio.gather`, order controls belong in `SafeBroker`.** Parallelising
#    submission is a latency optimisation; enforcing caps is a risk control. Keeping them at separate
#    layers prevents one concern from overriding the other.
# 3. **Post-fill reconciliation is a separate step.** The same reconciliation routine used to *plan*
#    the basket is used to *verify* the basket. Residual deltas name the specific failure modes.
# 4. **Execution cost is a monitoring signal, not a KPI.** When paper submission is enabled, drift in
#    average slippage across successive rebalances is the Chapter 26 monitoring input.
# 5. **Programmatic close-out is part of the loop, not an operator chore.** A daily basket strategy
#    that opens positions at start of day needs an explicit close-out path. `OrderType.MOC`
#    targets each newly opened leg for the closing auction, subject to exchange acceptance
#    and cutoff rules. Planning mode creates no exposure to flatten.
# 6. **Live-trading data sources come from the broker.** Warmup bars and pre-submission quotes both
#    come from IB in this notebook — research-time loaders (which may have a different cutoff date,
#    survivor universe, or vendor) belong in the training pipeline, not in the live execution path.
#    Chapter 25.6's feature-parity discussion applies: when the live and training data sources differ,
#    the signal ranks names the model was never validated on.
#
# **Next:** See `10_safety_risk_demo` for the kill-switch and state-persistence behaviour that `SafeBroker`
# enforces on every leg of this basket.
