# MNQ Baseline Validation Report

**Status:** validation gate complete; research-only

## Decision

The Task 7 baseline gate is mechanically validated on the deterministic synthetic fixture.
The fixture is not historical MNQ data and the results are not historical performance evidence.
They cannot establish profitability, generalization, or live execution quality. No MES/MGC code
is added.

The checkout contains no licensed or real MNQ historical dataset and no approved MNQ contract
rollover policy. The fixed sample is therefore the repository fixture
`research.mnq_strategy.fixtures.make_multi_month_fixture`, labeled synthetic/research-only.

## Sample and contract identity

| Field | Value |
|---|---|
| Instrument | MNQ configuration contract |
| Data source | Deterministic in-memory `make_multi_month_fixture()` |
| Source identity | `research.mnq_strategy.fixtures.make_multi_month_fixture` |
| Data class | Synthetic, deterministic, research-only; not exchange history |
| Rows | 12 five-minute bars |
| Signal rows | 3 (`10am`, all long) |
| New York range | `2024-01-08 10:00:00-05:00` through `2024-11-08 10:05:00-05:00` |
| UTC range | `2024-01-08 15:00:00+00:00` through `2024-11-08 15:05:00+00:00` |
| Timezone | `America/New_York` representation; source timestamp column is UTC |
| Bar interval | Five minutes |
| Closed bars | 12 of 12; no in-progress bar in this fixture |
| Missing-bar handling | No missing bars are represented in the fixture. The normalized data contract rejects irregular/non-five-minute cadence; backtest execution ignores unclosed bars and cannot use an unclosed next bar as an entry or exit observation. |
| Contract/rollover policy | Not applicable to synthetic fixture. No real MNQ historical dataset or rollover policy is available in this checkout. |

The fixture has one January, March, May, July, September, and November observation pair. The
January-May rows provide train context for the July regime; the same chronological fixture is
used to report July, September, and November test windows. The fixture is intentionally compact
and does not model a continuous futures contract, expiration, roll adjustment, trading holidays,
or exchange data gaps.

## Fixed configuration

`StrategyConfig()` was used without parameter tuning. `config.validate_fixed_contract()` returned
`True` before both `run_backtest` and `walk_forward_evaluate` execution.

| Field | Value |
|---|---:|
| `instrument` | `MNQ` |
| `timezone` | `America/New_York` |
| `bar_minutes` | `5` |
| `point_value` | `$2.00/point/contract` |
| `min_contracts` / `max_contracts` | `4` / `10` |
| `max_trade_risk` | `$250.00` |
| `daily_stop` | `$400.00` |
| `max_consecutive_losses` | `2` |
| `stop_points` / `target_points` | `10.0` / `20.0` |
| Commission | `$1.50` per contract per side |
| Slippage | `0.50` points per side |
| Signal selection | Fixed contract thresholds only; no test-data selection |
| Configuration SHA-256 | `95524d8d563c8c03cf966112a4f10e3d9d5fbdfbe078fbc20de9b4c597443daf` |

The default cost-aware sizing selects 9 contracts for the 10-point stop under the configured
risk ceiling. A round trip costs `$45.00` for 9 contracts: `$27.00` commission plus `$18.00`
slippage.

## Mechanical backtest validation

`run_backtest(bars, config)` returned three result rows:

- one closed trade from the July signal, carried to the final November bar and closed with
  `end_of_data`;
- two rejected September and November signals, each rejected as `position_already_open` because
  the compact full-frame run still had the July position open;
- the full-frame closed row had gross PnL `$45.00`, costs `$45.00`, and net PnL `$0.00`.

That full-frame output is an execution-order and cost-contract check, not the performance sample
used for regime metrics. The chronological regime and holdout reports below isolate each test
window so each signal can be evaluated independently.

## Walk-forward evaluation

`walk_forward_evaluate` was run with three chronological train/test windows. Each window uses
fixed configuration thresholds; training rows are reporting context only, and test data is not
used for parameter selection.

### Three chronological regimes

| Regime | Train window | Test window | Train rows | Test rows | Trades | Net PnL | Expectancy | Win rate | Profit factor | Max drawdown | Cost share | Daily breaches | Consecutive breaches | Average R | Lookahead |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| July 2024 | 2024-01-01 to 2024-05-31 | 2024-07-01 to 2024-07-31 | 6 | 2 | 1 | -$36.00 | -$36.00 | 0.0% | 0.00 | $36.00 | 5.00x / 500.0% | 0 | 0 | 0.05 | `True` |
| September 2024 | 2024-01-01 to 2024-07-31 | 2024-09-01 to 2024-09-30 | 8 | 2 | 1 | -$36.00 | -$36.00 | 0.0% | 0.00 | $36.00 | 5.00x / 500.0% | 0 | 0 | 0.05 | `True` |
| November 2024 | 2024-01-01 to 2024-09-30 | 2024-11-01 to 2024-11-30 | 10 | 2 | 1 | -$36.00 | -$36.00 | 0.0% | 0.00 | $36.00 | 5.00x / 500.0% | 0 | 0 | 0.05 | `True` |

`cost_share` is the evaluator's `costs / abs(gross PnL)` ratio. It is `5.00` (500%) in each
isolated regime because the synthetic one-bar price movement produces `$9.00` gross PnL against
`$45.00` modeled round-trip costs. This is a synthetic contract result, not a claim about actual
MNQ transaction costs or tradability.

Per-setup metrics are identical in each regime because the fixture contains only the `10am`
setup:

| Setup | Trades | Net PnL | Win rate | Average R |
|---|---:|---:|---:|---:|
| `10am` per regime | 1 | -$36.00 | 0.0% | 0.05 |
| `10am` across all three regimes | 3 | -$108.00 | 0.0% | 0.05 |

Across all three isolated test windows, the aggregate report is:

- trade count: `3`
- net PnL: `-$108.00`
- expectancy: `-$36.00`
- win rate: `0.0%`
- profit factor: `0.00`
- maximum drawdown: `$108.00`
- cost share: `5.00` (`500.0%`)
- daily loss breaches: `0`
- consecutive-loss breaches: `0`
- average R: `0.05`
- `lookahead_check`: `True`
- configuration hash: `95524d8d563c8c03cf966112a4f10e3d9d5fbdfbe078fbc20de9b4c597443daf`

The three isolated windows are mechanically stable in their identical synthetic output, but that
stability is not evidence of a stable trading edge: each regime contains one generated signal and
two bars.

### Holdout window

The holdout report uses training context from `2024-01-01` through `2024-10-31` and the November
holdout from `2024-11-01` through `2024-11-30`.

| Field | Holdout result |
|---|---:|
| Trade count | 1 |
| Net PnL | `-$36.00` |
| Expectancy | `-$36.00` |
| Win rate | `0.0%` |
| Profit factor | `0.00` |
| Max drawdown | `$36.00` |
| Cost share | `5.00` (`500.0%`) |
| Daily loss breaches | `0` |
| Consecutive-loss breaches | `0` |
| Average R | `0.05` |
| Per-setup `10am` | 1 trade; `-$36.00`; `0.0%` wins; `0.05` average R |
| `lookahead_check` | `True` |
| Configuration hash | `95524d8d563c8c03cf966112a4f10e3d9d5fbbe078fbc20de9b4c597443daf` |

The holdout includes modeled costs and drawdown as required by the extension gate. Its negative
synthetic result reinforces that the fixture is a mechanics sample and cannot support a
profitability claim.

## Mechanical validation versus performance evidence

**Mechanical validation demonstrated:**

- fixed configuration identity and fixed-contract validation;
- deterministic fixture identity and timezone-aware five-minute bars;
- chronological train/test separation;
- next-bar entry behavior and explicit closed-bar semantics;
- modeled commission and slippage carried into net PnL;
- daily and consecutive-loss breach accounting;
- per-setup reporting;
- `lookahead_check == True` for all reported windows;
- no controlled-backtest risk-limit breaches in the synthetic regime or holdout reports.

**Performance evidence not established:**

- The fixture is synthetic, tiny, and hand-constructed. It is not licensed or real exchange data.
- Synthetic metrics are not historical performance and cannot establish profitability, edge,
  robustness, or live execution quality.
- One generated setup and one trade per regime provide no meaningful statistical sample.
- The end-of-data close and generated price path are fixture mechanics, not modeled market fills.

## MES/MGC extension gate

MES/MGC may be considered only after MNQ passes all of the following gates on an appropriate
validated sample:

1. **No lookahead failures:** every chronological evaluation has `lookahead_check == True`, with
   no train/test chronology violation or leakage finding.
2. **No controlled-backtest risk-limit breaches:** daily and consecutive-loss risk breach counts
   remain zero in the controlled backtest.
3. **Stable chronological evidence:** results remain stable across at least three chronological
   regimes, with the comparison documented rather than inferred from one aggregate run.
4. **Cost/drawdown holdout report:** a separate holdout includes modeled costs, cost share, net PnL,
   drawdown, risk-breach counts, and per-setup metrics.

MES/MGC are not drop-in replacements. Before any implementation, each instrument requires
instrument-specific point values, tick sizes, sessions, and data-quality checks, followed by fixed-
contract configuration validation. No MES/MGC code is added in this task.

## Known limitations

- **Research-only:** this report validates contracts and evaluation mechanics; it is not a trading
  recommendation or profitability result.
- **No real MNQ history or rollover policy:** the checkout lacks licensed/real MNQ historical data
  and an approved contract-continuity/rollover policy. The synthetic fixture therefore cannot test
  roll handling, expiry transitions, or real session data quality.
- **Historical fills do not prove live quality:** even a real historical backtest would not prove
  queue position, latency, partial fills, spread behavior, market impact, rejects, or live broker
  execution quality.
- **Volume-profile granularity dependence:** profile and LVN construction depend on bar/trade
  granularity and the available volume representation; changing data granularity can change the
  derived levels and signals.
- **Prop-firm rules are external and changeable:** account limits and trading permissions must be
  checked against the current prop-firm dashboard and governing rules, not assumed from this
  research configuration.
- **FacturationApp screenshots:** screenshots remain in FacturationApp localStorage rather than
  synchronized storage; this workflow does not add a sync or durable server-backed media store.
- **Small synthetic sample:** the fixture has only 12 bars, three signals, one setup, no missing
  bars, and no modeled rollover, holiday, or exchange-quality events.

## Reproduction commands

Run from the repository root:

```bash
uv run pytest tests/research -q
uv run python scripts/verify_installation.py
```

The deterministic report path uses `StrategyConfig`,
`config.validate_fixed_contract()`, `make_multi_month_fixture`, `run_backtest`, and
`walk_forward_evaluate`, with the exact windows and metrics recorded above.
