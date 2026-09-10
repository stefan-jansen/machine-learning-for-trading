# Chapter 25: Live Trading Systems

The transition from profitable backtest to live execution is where most algorithmic trading projects fail. Not because the strategy lacks edge, but because the production system diverges from the research environment in subtle ways that erode returns. This chapter demonstrates how a unified framework eliminates that divergence by running identical strategy code in backtest, paper, and live modes.

## Learning Objectives

After completing this chapter, you will be able to:

1. Explain why technical divergence between research and production is a primary failure mode, and how a unified framework reduces that risk
2. Design a dual-mode, event-driven trading architecture where deterministic strategy logic runs unchanged across backtest, paper, and live execution
3. Compare broker, exchange, and managed-platform deployment paths in terms of asset coverage, execution quality, and operational burden
4. Model order handling as an explicit state machine supporting partial fills, cancellations, rejections, and idempotent crash recovery
5. Verify technical parity across the full pipeline, from raw data and features to predictions, sizing, and orders
6. Plan a staged live rollout using pre-flight checks, shadow trading, kill switches, and reconciliation procedures

## Chapter Sections

| Section | Title | Core Idea |
|---------|-------|-----------|
| 25.1 | The unified research-to-production framework | Identical strategy code across backtest and live modes eliminates two-pipeline divergence bugs |
| 25.2 | Integrating with Interactive Brokers | IBKR provides multi-asset coverage with TWS/Gateway connection management and state reconciliation |
| 25.3 | Integrating with Alpaca | Lower-friction deployment for US equities, ETFs, and crypto with REST/WebSocket APIs |
| 25.4 | QuantConnect and managed platforms | Trade-offs between self-hosted and managed platforms in speed, flexibility, and IP exposure |
| 25.5 | Order lifecycle management | Live execution is a stateful async process requiring formal state machines and idempotent recovery |
| 25.6 | Ensuring technical parity through pipeline verification | Staged parity testing across data, features, predictions, and orders distinguishes bugs from market changes |
| 25.7 | Operational readiness | Defense-in-depth safety controls bridge the gap between "code works" and "safe to trade with money" |

## Notebooks

### 25.1 The unified research-to-production framework (notebooks 01 and 02)

*Proves that the same strategy class produces identical signals in both backtest and live engines.*

| # | Notebook | What It Teaches |
|---|----------|-----------------|
| 01 | [`01_unified_framework_demo`](01_unified_framework_demo.ipynb) | Runs a simple dual MA crossover strategy through both `ml4t.backtest.Engine` and `ml4t.live.LiveEngine` on the same ETF data, then compares signals to prove 9/9 perfect parity. Demonstrates the zero-code-change deployment claim. |
| 02 | [`02_etfs_deployment_loop`](02_etfs_deployment_loop.ipynb) | The chapter's anchor demonstration of the seven-step deployment cycle: refresh ETF data through `ml4t-data`, recompute the financial-only feature subset, refit a Ridge regressor with the case study's α=10⁶ regularisation, persist the deployment artefacts, predict the live window, replay it through `ml4t.backtest.Engine` for the offline reference tape, and stage the latest top-K basket with a run record for monitoring. |

### 25.2 Integrating with Interactive Brokers (notebooks 03 and 12)

| # | Notebook | What It Teaches |
|---|----------|-----------------|
| 03 | [`03_ib_paper_trading_demo`](03_ib_paper_trading_demo.ipynb) | Connects to IB TWS/Gateway via `IBBroker`, wraps with `SafeBroker` (shadow mode, position/order/daily-loss limits, persisted `RiskState`, startup reconciliation via `safe_broker.connect()`), and runs a momentum strategy. Hard-fails with an operator checklist when TWS is unreachable, with no silent fallback. |
| 12 | [`12_ib_basket_rebalance_demo`](12_ib_basket_rebalance_demo.ipynb) | Extends the single-order IB demo to a full daily-rebalance workflow on a 20-name US large-cap universe: startup reconciliation via `SafeBroker.connect()` against a persisted state file, basket submission through `asyncio.gather` and `SafeBroker`, post-fill state polling, and slippage-vs-last-close execution summary. Requires a live TWS or Gateway paper session, with no silent fallback. |

### 25.3 Integrating with Alpaca (notebooks 04 and 05)

| # | Notebook | What It Teaches |
|---|----------|-----------------|
| 04 | [`04_alpaca_paper_trading_demo`](04_alpaca_paper_trading_demo.ipynb) | Complete Alpaca paper trading workflow: credential verification, SafeBroker wrapping, ETF momentum strategy, and order type demonstrations. Under headless papermill (`ML4T_HEADLESS_PAPERMILL=1`) the notebook auto-switches `LIVE_FEED=0` and runs the simulated path against a flat-dict `MockBroker` that records fill status (`filled` / `rejected` / `unsupported`) from the outcome rather than blindly logging fills. |
| 05 | [`05_alpaca_crypto_live_demo`](05_alpaca_crypto_live_demo.ipynb) | Maps the 19-perp case-study universe (Binance USDT) to Alpaca USD spot, where 11 pairs are tradeable and ADA, APT, ATOM, BNB, COMP, INJ, NEAR and SUI are not, and reframes the strategy as a momentum z-score proxy over the executable subset, with tz-aware UTC funding-hour handling. Same headless-papermill `LIVE_FEED` auto-switch as notebook 04. |

### 25.4 QuantConnect and managed platforms (notebook 06)

| # | Notebook | What It Teaches |
|---|----------|-----------------|
| 06 | [`06_quantconnect_case_study`](06_quantconnect_case_study.ipynb) | Exports 46,466 precomputed ETF predictions (95 symbols over 497 dates, 2024-01-02 to 2025-12-23) to QuantConnect-compatible JSON, demonstrating the prediction-bridge pattern that avoids reimplementing feature engineering in LEAN. |

### 25.5 Order lifecycle management (notebook 07)

| # | Notebook | What It Teaches |
|---|----------|-----------------|
| 07 | [`07_order_state_machine`](07_order_state_machine.ipynb) | Implements the order lifecycle as a finite state machine with 10 states and 19 valid state-event transitions, audit trail logging, and visualization. Demonstrates invalid transition rejection, the PENDING_CANCEL → FILLED race, and weighted-average fill-price calculation. Replace flows are out of scope and live in `ml4t.live.safety`. |

### 25.6 Ensuring technical parity through pipeline verification (notebooks 08, 09 and 11)

| # | Notebook | What It Teaches |
|---|----------|-----------------|
| 08 | [`08_pipeline_verification`](08_pipeline_verification.ipynb) | Runs 5 gated parity tests + 1 expected difference (feature warm-up) on a deterministic synthetic tape, producing a CI-compatible pass/fail summary. Uses static `SYMBOL_OFFSETS` instead of process-randomised hashing so the tape is byte-identical across machines, and emits SKIP semantics when the async live pipeline cannot execute under Papermill rather than silently passing against an empty live log. |
| 09 | [`09_crypto_funding_deployment_loop`](09_crypto_funding_deployment_loop.ipynb) | Demonstrates the full ML-to-live pipeline across two venues: OKX supplies the data plane (bars and 8-hour funding for the available subset of the 19-perp research universe) and Alpaca paper the execution plane on the eleven USD-quoted spot pairs. Trains a LightGBM 3-class direction model on the Chapter 12 panel, then runs one deployment cycle: connect, train, persist, predict the live cross-section, stage orders, and write the run record. |
| 11 | [`11_fx_deployment_loop`](11_fx_deployment_loop.ipynb) | FX deployment loop using IB paper as both the data plane and the execution plane: pulls live FX bars from the same TWS/Gateway session that routes the orders, computes momentum/carry/USD-factor features matching the Ch12 FX schema, ranks pairs and longs the top-K with a positive predicted return, and runs a daily paper-rebalance loop. The single-broker topology contrasts with the split-venue OKX+Alpaca crypto case in §25.6. |

### 25.7 Operational readiness (notebooks 10 and 13)

| # | Notebook | What It Teaches |
|---|----------|-----------------|
| 10 | [`10_safety_risk_demo`](10_safety_risk_demo.ipynb) | Drives six of SafeBroker's risk controls into their failure modes: order size limits, position limits, rate limiting, asset restrictions, the kill switch (which persists across restarts), and shadow mode with VirtualPortfolio carrying weighted-average cost basis. Duplicate-order filtering and price-deviation checks are configured through the same `LiveRiskConfig` but are not demonstrated anywhere in the chapter; daily-loss monitoring is driven into its kill-switch trip in notebook 13. |
| 13 | [`13_runtime_safety_showcase`](13_runtime_safety_showcase.ipynb) | Drives the runtime-safety contract under failure: stale-data rejection via `max_data_staleness_seconds`, automatic kill-switch trip on a simulated daily-loss breach (and latch survival across `SafeBroker` reconstruction), `SafeBroker.connect()` startup reconciliation against a deliberately divergent persisted state file, and `LiveEngine.runtime_status()` health-state transitions (`stopped` → `ok` → `feed_silent`). Closes with an `ml4t-live status` CLI walk-through. No real broker required. |

## Running Notebooks

```bash
# From repo root: production mode
uv run python 25_live_trading/01_unified_framework_demo.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "25_live_trading"

# Headless (no display)
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python 25_live_trading/01_unified_framework_demo.py
```

## Required Environment Variables

Live-broker notebooks read credentials from environment variables (typically
loaded from `.env`):

- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`, required by notebooks 02, 04, 05 and 09 for
  Alpaca paper trading and crypto market data.
- Interactive Brokers TWS or Gateway on `127.0.0.1:7497` (paper), required by
  notebooks 03, 11 and 12. Set the `Read-Only API` flag off and add a loopback trusted
  IP. CLIENT_ID is hardcoded per notebook (03 uses 10, 11 uses 11, 12 uses 12) so the
  three can run back-to-back without socket conflicts.
- OKX REST API (`public` endpoints, no key needed), used by notebook 09 to fetch
  perpetual-swap bars and funding. The SDK ships in the `live` extra
  (`uv sync --extra live`); its PyPI name is `python-okx`, not `okx`.

## Deferred / Environment-Gated Notebooks

| Notebook | Reason | Path |
|----------|--------|------|
| `03_ib_paper_trading_demo` | Requires IB Gateway up AND US-equity RTH (09:30 to 16:00 New York, Monday to Friday). | Run during market hours with TWS reachable on port 7497. |
| `12_ib_basket_rebalance_demo` | Requires IB Gateway up AND a clean state-file reconciliation (delete `~/.ml4t/live_state/basket_demo_*.json` between rehearsal runs). | Same as notebook 03; also reset state-file before re-running. |
| `11_fx_deployment_loop` | Requires IB Gateway up; FX trades 24/5 so RTH is non-binding. | Standard rerun. |
| `04_alpaca_paper_trading_demo` | Under headless papermill (`ML4T_HEADLESS_PAPERMILL=1`) the notebook auto-switches `LIVE_FEED=0` and runs the simulated path, because Alpaca's WebSocket loop is incompatible with `nest_asyncio` and the production timer cannot cancel the inner streaming task. Run interactively in Jupyter to exercise the real WebSocket feed. | Set the env var explicitly: `ML4T_HEADLESS_PAPERMILL=1 papermill 04_alpaca_paper_trading_demo.ipynb out.ipynb`. |
| `05_alpaca_crypto_live_demo` | Same `LIVE_FEED` auto-switch under headless papermill. | Same. |

## Dependencies

- **Upstream**: Chapters 6 to 20 provide case study predictions consumed by the QuantConnect export and the ML strategy demo
- **Downstream**: Chapter 26 (MLOps) builds on the deployment patterns established here

Key libraries:
- `ml4t-backtest`: backtest engine and strategy base class
- `ml4t-live` (>=0.1.0) - live engine, `SafeBroker` with enforced position/order/daily-loss caps, persisted `RiskState`, startup reconciliation, `VirtualPortfolio` for shadow mode, and the `ml4t-live` CLI (`status`, `shadow`)
- `alpaca-py`: Alpaca broker integration
- `ib_async`: Interactive Brokers connection
- `python-okx`: OKX exchange SDK (used by notebook 09)
