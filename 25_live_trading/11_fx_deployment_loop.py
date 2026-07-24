# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FX Pairs Deployment Loop
#
# **Chapter 25: Live Trading Systems**
# **Section**: 25.6 (Pipeline Verification: Ensuring Technical Parity)
#
# **Docker image**: `ml4t` (requires IB TWS/Gateway running on host port 7497 with FX permissions)
#
# This notebook is the chapter's FX deployment-loop demonstration. The
# *Chapter 12* FX-pairs case study trains a daily cross-sectional model on
# 20 FX majors and crosses (`AUD_JPY` through `USD_JPY`); this notebook
# re-enacts the deployment cycle for that strategy through Interactive
# Brokers' paper account, which provides both the live data plane (daily
# FX bars via `reqHistoricalDataAsync` on IDEALPRO) and the execution
# plane (Forex spot orders on IDEALPRO). The cycle has six steps:
#
# 1. **Connect** to a TWS or IB Gateway paper session.
# 2. **Train** a Ridge regressor on the historical 20-pair daily panel
#    using a small inline feature schema (returns, momentum, volatility).
# 3. **Persist** the deployment artefacts under the configured isolated output directory.
# 4. **Predict** the latest cross-section by fetching daily FX bars from
#    IB and scoring all 20 pairs.
# 5. **Plan** the resulting top-K long basket against qualified IB Forex
#    contracts on IDEALPRO, with paper submission available only by explicit opt-in.
# 6. **Persist** the run as a JSON record for monitoring and audit.
#
# **Why one venue?** FX is one of the few asset classes where a single
# broker can serve as both the data plane and the execution plane: IB's
# IDEALPRO desk provides continuous interbank quotes during the FX trading
# week, and a paper account routes through the same data and order flow
# (with the obvious caveat that paper fills do not move real markets).
# This is a contrast to the crypto deployment loop in NB09, which had to
# split data (OKX) from execution (Alpaca paper) because no single retail
# venue offered both.
#
# **Important framing.** This notebook is a deployment-engineering
# rehearsal, not investment advice. Paper trading on IB exercises the
# real broker session under real account credentials; it does not commit
# capital. The notebook is structured to teach the operational mechanics
# of the deployment loop, and is not a recommendation to trade real money
# on this strategy.
#
# **Cross-References**
# - Chapter 12: FX-pairs case study (model training and registry)
# - Chapter 7: Forward-return labels for daily FX
# - Chapter 25.2: Interactive Brokers integration
# - Chapter 25.6: Pipeline verification across venues
# - Chapter 26: Repeated model serving and monitoring
#
# **Learning Objectives**
# - Run a real retrain-and-deploy cycle on the FX-pairs strategy with no
#   mock components.
# - Use one broker (IB paper) as both data plane and execution plane and
#   reason about the symmetry that simplifies relative to NB09.
# - Surface FX-specific deployment frictions: contract qualification and
#   IDEALPRO minimums.
#
# **Prerequisites**
# - **TWS or IB Gateway running locally** with API access enabled and a
#   paper account logged in. Default port 7497 (TWS) or 4002 (Gateway).
# - Historical FX panel under `ML4T_DATA_PATH/fx_pairs` (loaded via
#   `data.load_fx_pairs`).
# - Forward-return label parquet at
#   `case_studies/fx_pairs/labels/fwd_ret_1d.parquet`.

# %%
"""FX Pairs Deployment Loop — train, predict, paper-trade, persist on IB."""

import hashlib
import json
import logging
import pickle
import warnings
from datetime import UTC, datetime, timedelta

import matplotlib.pyplot as plt
import polars as pl
from async_utils import run_async
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data import load_fx_pairs
from utils.config import CASE_STUDIES_DIR
from utils.paths import display_path, get_chapter_dir, get_output_dir
from utils.style import COLORS, add_message_title

try:
    from ib_async import Forex
except ImportError as exc:
    raise RuntimeError(
        "ib_async is required for the IB data and execution planes. "
        "Install with: uv pip install ib_async"
    ) from exc

from ml4t.backtest import OrderSide
from ml4t.live.brokers.ib import IBBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fx_pairs_deployment")
logging.getLogger("ml4t").setLevel(logging.WARNING)


def run_demo(awaitable):
    """Run an async demo while suppressing only nest_asyncio's Python 3.14 deprecation."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nest_asyncio")
        return run_async(awaitable)


# %% tags=["parameters"]
TRAIN_END_DATE = "2024-12-31"
RIDGE_ALPHA = 1.0
TOP_K = 5  # long the top-K predicted-return pairs
BASE_QTY_PER_LEG = 20_000.0  # base-ccy units per leg; IDEALPRO accepts 1k step on majors
IB_HOST = "127.0.0.1"
IB_PORT = 7497  # TWS paper = 7497, Gateway paper = 4002
IB_CLIENT_ID = 11
IB_HISTORICAL_DURATION = "60 D"
SUBMIT_PAPER_ORDERS = False  # explicit opt-in only; publication execution is dry-run

# %% [markdown]
# ## 1. Setup and IB Connection
#
# A single TWS/Gateway session backs both data and execution. The notebook
# fails loudly if the session is unreachable rather than degrading silently
# — a deployment loop that hides its failure modes teaches the wrong lesson.

# %%
CHAPTER_DIR = get_chapter_dir(25)
ARTIFACTS_DIR = get_output_dir(25, "fx_deployment")
RUNS_DIR = ARTIFACTS_DIR / "runs"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

CASE_STUDY_UNIVERSE = [
    "AUD_JPY",
    "AUD_NZD",
    "AUD_USD",
    "CAD_JPY",
    "CHF_JPY",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_CHF",
    "EUR_GBP",
    "EUR_JPY",
    "EUR_USD",
    "GBP_AUD",
    "GBP_CHF",
    "GBP_JPY",
    "GBP_USD",
    "NZD_JPY",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
]

# IB Forex symbology drops the underscore (EUR_USD → EURUSD on IDEALPRO).
IB_SYMBOL = {sym: sym.replace("_", "") for sym in CASE_STUDY_UNIVERSE}


# %% [markdown]
# The live feature schema mirrors the compact daily model trained below.


# %%
FEATURE_COLS = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "vol_5d",
    "vol_21d",
    "vol_ratio",
    "rsi_14",
    "dist_ma_21d",
]


# %% [markdown]
# The startup banner makes the service, data, execution, artifact, and mutation boundaries explicit.


# %%
print("=" * 70)
print("FX PAIRS DEPLOYMENT LOOP")
print("=" * 70)
print(f"Strategy universe: {len(CASE_STUDY_UNIVERSE)} pairs (case-study underscored)")
print(f"  Single venue:    Interactive Brokers paper ({IB_HOST}:{IB_PORT})")
print("  Data plane:      reqHistoricalDataAsync, daily MIDPOINT bars")
print("  Execution plane: Forex spot on IDEALPRO")
print(f"Artifacts dir:     {display_path(ARTIFACTS_DIR)}")
print(f"Order submission:  {'ENABLED' if SUBMIT_PAPER_ORDERS else 'DISABLED (dry-run)'}")

# %%
print("\nConnecting to IB paper session ...")


# %% [markdown]
# The connection gate rejects unreachable services and any managed account that is not an IB paper account.


# %%
async def open_ib() -> IBBroker:
    broker = IBBroker(host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID)
    try:
        await broker.connect()
    except Exception as exc:
        msg = (
            f"Could not connect to IB at {IB_HOST}:{IB_PORT}: {exc}\n"
            "Checklist:\n"
            "  1. Start TWS or IB Gateway and log into a paper account.\n"
            "  2. Configure → API → Settings: enable ActiveX/Socket Clients.\n"
            f"     Socket port must be {IB_PORT} (TWS paper=7497, Gateway paper=4002).\n"
            "  3. Allow 127.0.0.1 in Trusted IPs.\n"
            "  4. Re-run this notebook once TWS is reachable."
        )
        raise RuntimeError(msg) from exc
    accounts = [str(account) for account in broker.ib.managedAccounts()]
    if not accounts or any(not account.upper().startswith("DU") for account in accounts):
        await broker.disconnect()
        raise RuntimeError(
            "Refusing non-paper IB session: every managed account must start with 'DU'"
        )
    print(f"Connected to a verified IB paper session ({len(accounts)} managed account(s)).")
    return broker


# %%
broker = run_demo(open_ib())

# %% [markdown]
# ### Qualify Forex Contracts
#
# Each FX pair must be resolved against IB's contract universe before it
# can be priced or traded. We qualify all 20 case-study pairs once and
# cache the qualified Forex contracts on the broker so subsequent
# `submit_order_async` calls find them.


# %%
async def qualify_forex_contracts():
    """Resolve and cache `Forex(symbol)` contracts for the case-study pairs.

    `IBBroker._contracts` is the cache the broker consults from
    `submit_order_async`; populating it here means the submit path uses
    the qualified Forex contract (IDEALPRO) rather than the default
    `Stock(symbol, "SMART", "USD")`.
    """
    pairs_to_qualify = [Forex(IB_SYMBOL[sym]) for sym in CASE_STUDY_UNIVERSE]
    qualified = await broker.ib.qualifyContractsAsync(*pairs_to_qualify)
    qualified_by_local = {q.localSymbol.replace(".", ""): q for q in qualified if q.conId}
    out = {}
    missing = []
    for sym in CASE_STUDY_UNIVERSE:
        ib_sym = IB_SYMBOL[sym]
        if ib_sym in qualified_by_local:
            contract = qualified_by_local[ib_sym]
            out[sym] = contract
            broker._contracts[ib_sym] = contract
        else:
            missing.append(sym)
    return out, missing


qualified_contracts, missing_contracts = run_demo(qualify_forex_contracts())
print(
    f"\nQualified {len(qualified_contracts)}/{len(CASE_STUDY_UNIVERSE)} Forex contracts "
    f"on IDEALPRO."
)
if missing_contracts:
    print(f"  Missing: {', '.join(missing_contracts)}")

# %% [markdown]
# **Finding.** The qualification table above is the authoritative coverage
# result for this run. Missing contracts remain visible and never enter the
# live cross-section. This explicit gate keeps broker coverage from becoming
# an unstated assumption.

# %% [markdown]
# ## 2. Build Training Panel
#
# Historical daily bars from `load_fx_pairs` are the training source.
# Features and the forward-return label are joined to form the panel
# the Ridge model is fit against.

# %%
print("\n" + "=" * 70)
print("STEP 2: BUILD TRAINING PANEL")
print("=" * 70)

prices = (
    load_fx_pairs()
    .select(["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    .sort(["symbol", "timestamp"])
)
# load_fx_pairs returns intraday bars; collapse to one row per (symbol, date)
# at the latest bar of each session.
prices = (
    prices.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    .group_by(["symbol", "_session_date"], maintain_order=True)
    .agg(
        [
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ]
    )
    .rename({"_session_date": "timestamp"})
    .sort(["symbol", "timestamp"])
)
print(f"Loaded daily prices: {prices.shape}")

# %% [markdown]
# ### Feature Schema
#
# The eight-feature deployment-loop schema is computed from daily OHLCV.
# Window sizes are in trading days. Used for both the training panel
# (Step 2) and the live cross-section (Step 4).


# %%
def daily_rsi_expression(window: int = 14) -> pl.Expr:
    """Return the bounded rolling RSI expression for each symbol."""
    change = pl.col("close") - pl.col("close").shift(1).over("symbol")
    avg_gain = change.clip(lower_bound=0).rolling_mean(window_size=window).over("symbol")
    avg_loss = (-change).clip(lower_bound=0).rolling_mean(window_size=window).over("symbol")
    return (100 * avg_gain / (avg_gain + avg_loss).clip(lower_bound=1e-10)).alias("rsi_14")


# %% [markdown]
# Returns are computed first because volatility and the short-to-long volatility ratio depend on them.


# %%
def compute_features_daily(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute the eight-feature deployment-loop schema on a daily panel."""
    returns = [
        ((pl.col("close") / pl.col("close").shift(h).over("symbol")) - 1).alias(f"ret_{h}d")
        for h in (1, 5, 21)
    ]
    volatility = [
        pl.col("ret_1d").rolling_std(window_size=h).over("symbol").alias(f"vol_{h}d")
        for h in (5, 21)
    ]
    distance = (
        (pl.col("close") / pl.col("close").rolling_mean(window_size=21).over("symbol")) - 1
    ).alias("dist_ma_21d")
    return (
        panel.sort(["symbol", "timestamp"])
        .with_columns(returns)
        .with_columns(volatility + [daily_rsi_expression(), distance])
        .with_columns(
            (pl.col("vol_5d") / pl.col("vol_21d").clip(lower_bound=1e-8)).alias("vol_ratio")
        )
    )


# %%
panel = compute_features_daily(prices)
print(f"Features computed: {panel.shape}")

# %% [markdown]
# ### Join Forward-Return Labels
#
# Pair the eight-feature panel with the 1-day forward-return label parquet
# from the fx_pairs case study to form the supervised training panel.

# %%
labels_path = CASE_STUDIES_DIR / "fx_pairs" / "labels" / "fwd_ret_1d.parquet"
labels = pl.read_parquet(labels_path)
print(f"Labels loaded: {labels.shape}, cols: {labels.columns}")

labels = labels.with_columns(pl.col("timestamp").cast(pl.Date))
train_panel = panel.with_columns(
    pl.col("timestamp").shift(-1).over("symbol").alias("_label_end")
).join(
    labels.select(["symbol", "timestamp", "fwd_ret_1d"]),
    on=["symbol", "timestamp"],
    how="inner",
)
print(f"After label join: {train_panel.shape}")

# %% [markdown]
# ## 3. Train and Persist
#
# A Ridge regression on the eight-feature schema produces a daily
# expected-return forecast per pair.

# %%
TRAIN_CUTOFF_EXCLUSIVE = datetime.fromisoformat(TRAIN_END_DATE).date() + timedelta(days=1)
train_full = train_panel.filter(pl.col("_label_end") < TRAIN_CUTOFF_EXCLUSIVE).drop_nulls(
    subset=FEATURE_COLS + ["fwd_ret_1d"]
)
print(
    f"\nTraining rows with 1D label endpoints before "
    f"{TRAIN_CUTOFF_EXCLUSIVE}: {train_full.shape[0]:,}"
)

X_train = train_full.select(FEATURE_COLS).to_numpy()
y_train = train_full["fwd_ret_1d"].to_numpy()

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

model = Ridge(alpha=RIDGE_ALPHA).fit(X_train_scaled, y_train)
print(f"Ridge fit: alpha={RIDGE_ALPHA}, n={len(y_train):,}")

# %% [markdown]
# Persist the model, scaler, feature schema, and training metadata under
# the isolated output directory. Storing the schema next to the binary
# artefacts lets the inference path validate its inputs before scoring.

# %%
model_path = ARTIFACTS_DIR / "model.pkl"
scaler_path = ARTIFACTS_DIR / "scaler.pkl"
features_path = ARTIFACTS_DIR / "feature_columns.json"
metadata_path = ARTIFACTS_DIR / "training_metadata.json"

with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
with open(features_path, "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
metadata = {
    "case_study": "fx_pairs",
    "label": "fwd_ret_1d",
    "task": "regression",
    "train_end_date": TRAIN_END_DATE,
    "train_rows": int(train_full.shape[0]),
    "num_features": len(FEATURE_COLS),
    "ridge_alpha": RIDGE_ALPHA,
    "trained_at_utc": datetime.now(UTC).isoformat(),
    "training_data_source": "load_fx_pairs (canonical FX panel)",
    "inference_data_source": "IB reqHistoricalDataAsync (IDEALPRO daily MIDPOINT)",
}
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Persisted artefacts to {display_path(ARTIFACTS_DIR)}")

# %% [markdown]
# ## 4. Live Cross-Section from IB
#
# Fetch the most recent daily bars per pair via IB's
# `reqHistoricalDataAsync`, stitch them into a panel, compute the eight
# features, and isolate the latest valid feature row per symbol.

# %%
print("\n" + "=" * 70)
print("STEP 4: LIVE CROSS-SECTION FROM IB")
print("=" * 70)
IB_BAR_AUDIT: list[dict] = []


# %% [markdown]
# Raw IB bars are hashed and counted before parsing so the live record boundary is reproducible.


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
# Each pair parser preserves symbol identity and converts one raw IB bar to one canonical row.


# %%
async def fetch_one_pair(sym, contract):
    bars = await broker.ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr=IB_HISTORICAL_DURATION,
        barSizeSetting="1 day",
        whatToShow="MIDPOINT",
        # FX is not RTH-gated in the equity sense; IB's useRTH=True for FX
        # restricts to the broker's defined trading-hours window (skips the
        # weekend gap and any maintenance pause).
        useRTH=True,
        formatDate=1,
        # timeout=0 disables ib_async's internal asyncio.wait_for, which
        # raises "Timeout should be used inside a task" under nest_asyncio
        # on Python 3.14. IB keeps its own network timeout.
        timeout=0,
    )
    audit = audit_ib_bars(sym, bars)
    if not bars:
        audit["parsed_rows"] = 0
        IB_BAR_AUDIT.append(audit)
        return pl.DataFrame()
    rows = [
        {
            "symbol": sym,
            "timestamp": (b.date.date() if hasattr(b.date, "date") else b.date),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume or 0),
        }
        for b in bars
    ]
    audit["parsed_rows"] = len(rows)
    IB_BAR_AUDIT.append(audit)
    return pl.DataFrame(rows)


# %% [markdown]
# Fan out across the case-study universe in parallel via
# `asyncio.gather`, collecting any per-pair fetch errors so the run record
# can surface them.


# %%
async def fetch_all_pairs():
    """Fetch daily bars for the whole case-study universe in parallel."""
    import asyncio

    syms_with_contracts = [
        (sym, qualified_contracts[sym]) for sym in CASE_STUDY_UNIVERSE if sym in qualified_contracts
    ]
    missing_now = [sym for sym in CASE_STUDY_UNIVERSE if sym not in qualified_contracts]

    async def safe_fetch(sym, contract):
        try:
            df = await fetch_one_pair(sym, contract)
            return sym, df, None
        except Exception as exc:
            return sym, None, repr(exc)[:160]

    fetched = await asyncio.gather(*[safe_fetch(s, c) for s, c in syms_with_contracts])
    out_frames = []
    errors = [(sym, "no qualified contract") for sym in missing_now]
    for sym, df, err in fetched:
        if err is not None:
            errors.append((sym, err))
        elif df is None or len(df) == 0:
            errors.append((sym, "empty bars"))
        else:
            out_frames.append(df)
    if not out_frames:
        return pl.DataFrame(), errors
    return pl.concat(out_frames).sort(["symbol", "timestamp"]), errors


# %%
live_prices, fetch_errors = run_demo(fetch_all_pairs())

if fetch_errors:
    print(f"Fetch errors: {len(fetch_errors)} of {len(CASE_STUDY_UNIVERSE)} pairs")
    for sym, e in fetch_errors[:8]:
        print(f"  {sym}: {e}")

if len(live_prices) == 0:
    raise RuntimeError(
        "IB returned no bars for any pair — cannot continue. "
        f"Check TWS market-data permissions. First 3 errors: {fetch_errors[:3]}"
    )


# %% [markdown]
# Parser conservation must hold before the potentially incomplete current daily bar is excluded.


# %%
ib_audit_frame = pl.DataFrame(IB_BAR_AUDIT)
if len(ib_audit_frame) == 0:
    raise RuntimeError("No IB bar payload reached the parser audit")
assert (ib_audit_frame["raw_rows"] == ib_audit_frame["parsed_rows"]).all()
assert ib_audit_frame["duplicate_dates"].sum() == 0


# %%
# A daily bar labelled with today's date may still be forming. Use only
# sessions strictly before the current UTC date for a conservative,
# reproducible decision boundary.
live_prices = live_prices.filter(pl.col("timestamp") < datetime.now(UTC).date())
if len(live_prices) == 0:
    raise RuntimeError("IB returned no completed daily bars before the current UTC date")

completed_counts = live_prices.group_by("symbol").len().rename({"len": "completed_rows"})
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

n_pairs = live_prices["symbol"].n_unique()
print(f"Live bars: {live_prices.shape} across {n_pairs} pairs")

# %% [markdown]
# Compute the eight features on the live panel and isolate the latest
# valid feature row per symbol — that row is the input to inference.

# %%
live_panel = compute_features_daily(live_prices)
latest_features = (
    live_panel.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in FEATURE_COLS]))
    .group_by("symbol")
    .agg(pl.all().last())
    .sort("symbol")
)
print(f"Latest valid feature rows: {latest_features.shape[0]} of {len(CASE_STUDY_UNIVERSE)} pairs")

# %% [markdown]
# ## 5. Predict and Rank
#
# Score the latest cross-section, rank by predicted return, and select
# the top-K pairs to long. The strategy in this demo is long-only on the
# top-ranked pairs; a production version would also short the bottom-K
# (IB Forex supports shorts), but the demo keeps the long basket to
# emphasize the deployment-mechanics learning objective.

# %%
print("\n" + "=" * 70)
print("STEP 5: PREDICT AND RANK")
print("=" * 70)

X_live = latest_features.select(FEATURE_COLS).to_numpy()
X_live_scaled = scaler.transform(X_live)
y_pred = model.predict(X_live_scaled)

predictions = (
    latest_features.select(["symbol", "timestamp", "close"])
    .with_columns(pl.Series("pred_ret_1d", y_pred))
    .sort("pred_ret_1d", descending=True)
)

# Long-only basket: filter on positive predicted returns before taking the
# top-K. In sustained risk-off regimes the cross-section can be entirely
# negative; trading "the best of the worst" would short volatility for no
# expected return.
top_k = predictions.filter(pl.col("pred_ret_1d") > 0).head(TOP_K)
print(f"\nTop-{TOP_K} predicted-return pairs (long basket):")
print(top_k)
print("\nFull cross-section:")
print(predictions)

# %% [markdown]
# The ranked forecast chart makes the decision boundary visible. Bars above
# zero are eligible for the long basket; negative forecasts remain flat.

# %%
plot_predictions = predictions.sort("pred_ret_1d")
colors = [
    COLORS["positive"] if value > 0 else COLORS["neutral"]
    for value in plot_predictions["pred_ret_1d"].to_list()
]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(
    plot_predictions["symbol"].to_list(),
    plot_predictions["pred_ret_1d"].to_list(),
    color=colors,
)
ax.axvline(0, linestyle="--", color=COLORS["neutral"], linewidth=1)
ax.set(xlabel="Predicted 1-day return (decimal)", ylabel="FX pair")
add_message_title(
    ax,
    "Only positive FX forecasts qualify for the long basket",
    subtitle="Completed IB daily bars; green identifies eligible long positions",
)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Plan IB Paper Orders
#
# Construct the long basket against the qualified Forex contracts. The
# publication run records dry-run intents; explicit opt-in enables IB
# paper market orders. Quantities are fixed at `BASE_QTY_PER_LEG` units
# of the base currency, rounded to a 1k step. A USD-notional basket would need
# explicit per-pair cross-rate conversion (since EUR/USD's quote is in
# USD but EUR/JPY's is in JPY); fixed base-ccy quantities avoid that
# gymnastics for a demo.

# %%
print("\n" + "=" * 70)
print("STEP 6: PAPER EXECUTION (IB)")
print("=" * 70)


def round_to_step(qty: float, step: float = 1000.0) -> float:
    return float(int(qty // step) * step)


# %% [markdown]
# Each top-K row passes through `IBBroker.submit_order_async` with a fixed base-currency quantity. The record carries the
# resolved IB symbol, the reference price, the submission status, and any
# error so the run JSON is self-describing.


# %%
async def submit_fx_intent(row: dict) -> dict:
    """Plan or submit one long FX intent and return an audit record."""
    symbol = row["symbol"]
    ib_symbol = IB_SYMBOL[symbol]
    record = {
        "symbol": symbol,
        "ib_symbol": ib_symbol,
        "pred_ret_1d": row["pred_ret_1d"],
        "ref_price": row["close"],
    }
    qty = round_to_step(BASE_QTY_PER_LEG, step=1000.0)
    if qty <= 0:
        return record | {"status": "qty_too_small"}
    if not SUBMIT_PAPER_ORDERS:
        return record | {"status": "dry_run", "qty": qty}
    try:
        order = await broker.submit_order_async(
            asset=ib_symbol,
            quantity=qty,
            side=OrderSide.BUY,
        )
        order_id = getattr(order, "id", None)
        return record | {
            "status": "submitted",
            "qty": qty,
            "order_id": order_id if order_id is not None else str(order),
        }
    except Exception as exc:
        logger.warning("IB submit failed for %s (%s): %s", symbol, ib_symbol, exc)
        return record | {"status": "submit_failed", "error": repr(exc)[:200]}


# %% [markdown]
# The basket driver preserves a deterministic symbol order for both dry-run and authorized paper execution.


# %%
async def submit_basket(intents: list[dict]) -> list[dict]:
    """Process the selected FX intents in rank order."""
    return [await submit_fx_intent(row) for row in intents]


# %%
exec_results = run_demo(submit_basket(top_k.to_dicts()))
if not SUBMIT_PAPER_ORDERS:
    assert all(record["status"] == "dry_run" for record in exec_results)

status_counts = {}
for r in exec_results:
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

print("\nExecution summary:")
for status, n in sorted(status_counts.items()):
    print(f"  {status:<22} {n}")

print("\nPer-symbol detail:")
for r in exec_results:
    extras = []
    if "qty" in r:
        extras.append(f"qty={r['qty']:.0f}")
    if "order_id" in r:
        extras.append(f"order={str(r['order_id'])[:24]}")
    if "error" in r:
        extras.append(r["error"][:60])
    extra = " ".join(extras)
    print(f"  {r['symbol']:<8} pred={r['pred_ret_1d']:+.5f} → {r['status']:<16} {extra}")

# %% [markdown]
# ## 7. Persist Run JSON
#
# Per-cycle audit record. Stored under the configured isolated output
# directory so live runs accumulate without polluting the working tree.

# %%
print("\n" + "=" * 70)
print("STEP 7: PERSIST RUN")
print("=" * 70)

run_ts = datetime.now(UTC)
run = {
    "run_ts_utc": run_ts.isoformat(),
    "case_study": "fx_pairs",
    "model_metadata": metadata,
    "ib_session": {
        "host": IB_HOST,
        "port": IB_PORT,
        "client_id": IB_CLIENT_ID,
        "qualified_contracts": list(qualified_contracts.keys()),
        "missing_contracts": missing_contracts,
        "fetch_errors": [{"symbol": s, "error": e} for s, e in fetch_errors],
        "bar_parser_audit": ib_audit_frame.to_dicts(),
    },
    "predictions": predictions.to_dicts(),
    "top_k_basket": top_k.to_dicts(),
    "execution": exec_results,
    "summary": {
        "n_pairs_predicted": len(predictions),
        "top_k": TOP_K,
        "status_counts": status_counts,
    },
}
run_path = RUNS_DIR / f"{run_ts.strftime('%Y%m%dT%H%M%SZ')}.json"
with open(run_path, "w") as f:
    json.dump(run, f, indent=2, default=str)
print(f"Run persisted: {display_path(run_path)}")


# %% [markdown]
# Disconnect cleanly so the notebook does not leave an IB session behind.


# %%
async def close_ib():
    try:
        await broker.disconnect()
    except Exception as exc:
        logger.warning("IB disconnect: %s", exc)


run_demo(close_ib())
print("Disconnected from IB.")

# %% [markdown]
# ## Key Takeaways
#
# 1. **One operational venue covers live data and execution for FX.** Unlike crypto, where
#    NB09 splits OKX (data) from Alpaca (execution), IB serves as a
#    single live interface for both planes on the FX-pairs universe.
#    The deployment loop is correspondingly simpler.
# 2. **Contract qualification is its own deployment step.** The Forex
#    contract (IDEALPRO, 6-character localSymbol) is not the same object
#    as the case-study symbol (`AUD_JPY`) and must be resolved through
#    IB before any data fetch or order submission. Caching the qualified
#    contract on the broker is the lightest mechanism for making that
#    resolution explicit and reusable.
# 3. **The deployment artefact is not the research artefact.** The
#    Chapter 12 fx_pairs case study uses a 41-feature financial-feature
#    pipeline; the deployment loop here uses an eight-feature subset
#    that can be computed inline from raw OHLCV. Same data, same labels,
#    different feature surface, different code path.
# 4. **IDEALPRO minimums and base-ccy semantics are real.** Quantities
#    round to a 1k-base-ccy step. The notebook uses a fixed base-ccy
#    quantity per leg rather than a USD notional, because Forex pair
#    semantics (`Forex('EURUSD')` is EUR-quoted-in-USD whereas
#    `Forex('EURJPY')` is EUR-quoted-in-JPY) make a uniform "USD
#    notional" target ambiguous without an extra cross-rate conversion.
# 5. **Order submission is explicit opt-in.** The publication run leaves
#    `SUBMIT_PAPER_ORDERS=False`, records a dry-run basket, and preserves
#    the paper account without broker mutations.
#
# **Next**: see `08_pipeline_verification.ipynb` for systematic parity
# testing across pipeline stages, and `10_safety_risk_demo.ipynb` for the
# SafeBroker controls that would wrap this loop in a production
# deployment.
