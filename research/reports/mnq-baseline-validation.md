# MNQ Baseline Validation Report

**Status:** mechanical validation complete; historical MNQ validation blocked; research-only

## Decision

Task 7 mechanical validation is complete on the deterministic synthetic fixture. Historical MNQ
validation is blocked because the fixture is not historical MNQ data and the checkout has no
licensed/real MNQ historical sample or approved contract rollover policy. The results are not
historical performance evidence and cannot establish profitability, generalization, or live
execution quality.

The MES/MGC extension gate is **NOT CLEARED**. It must remain blocked until an appropriate
real/licensed MNQ sample and approved rollover policy are available and the MNQ workflow passes
all stated gate criteria. No MES/MGC code is added.

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
`True` before both `run_backtest` and `walk_forward_evaluate` execution. The full active
configuration identity, including instrument/session/risk/bracket/cost fields, is included in the
configuration hash.

| Field | Value |
|---|---:|
| `instrument` | `MNQ` |
| `timezone` | `America/New_York` |
| `bar_minutes` | `5` |
| `value_area_fraction` | `0.40` |
| `point_value` | `$2.00/point/contract` |
| `min_contracts` / `max_contracts` | `4` / `10` |
| `max_trade_risk` | `$250.00` |
| `daily_stop` | `$400.00` |
| `max_consecutive_losses` | `2` |
| `stop_points` / `target_points` | `10.0` / `20.0` |
| Commission | `$1.50` per contract per side |
| Slippage | `0.50` points per side |
| Session boundaries | RTH `09:30-16:00`; maintenance `16:00-18:00`; overnight starts `18:00` |
| Signal selection | Fixed contract thresholds only; no test-data selection |
| Configuration SHA-256 | `95524d8d563c8c03cf966112a4f10e3d9d5fbdfbe078fbc20de9b4c597443daf` |

The default cost-aware sizing selects 9 contracts for the 10-point stop under the configured
risk ceiling. A round trip costs `$45.00` for 9 contracts: `$27.00` commission plus `$18.00`
slippage. The `$250.00` total-risk rule is strict: a trade is accepted only when total risk is
strictly less than `$250.00`; equality is rejected.

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

### Executable daily guard evidence

`make_daily_guard_fixture()` contains only canonical bars and signal rows. Running it through
`run_backtest(bars, StrategyConfig())` produces two closed stop-loss trades at `-$225.00` net
each (including modeled costs), followed by one rejected signal with the stable reason
`daily_loss_limit`. The fixture's synthetic `net_pnl` column is not used because it no longer
exists; the outcomes are realized by the configured OHLC path.

## Walk-forward evaluation

`walk_forward_evaluate` was run as three separate one-window calls: one July call, one September
call, and one November call. `evaluation.py` rejects overlapping train/test windows when multiple
windows are passed in one call, so the three regimes are not combined into one multi-window call.
A separate one-window `walk_forward_evaluate` call was then used for the November holdout. Each
call uses fixed configuration thresholds; training rows are reporting context only, and test data
is not used for parameter selection.

The expanding train contexts are intentional and explicit: July uses January-May context,
September uses January-July context, and November uses January-September context. This records the
chronological information available before each test segment while preserving the fixed-contract
selection policy; it does not imply that parameters were fitted from the synthetic context.

The reproducible call sequence is:

```python
walk_forward_evaluate(bars, [july_window], config)
walk_forward_evaluate(bars, [september_window], config)
walk_forward_evaluate(bars, [november_window], config)
walk_forward_evaluate(bars, [holdout_window], config)
```

### Three chronological regimes

| Regime | Train window | Test window | Train rows | Test rows | Trades | Net PnL | Expectancy | Win rate | Profit factor | Max drawdown | Cost share | Daily breaches | Consecutive breaches | Average R | Chronology | Signal provenance |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| July 2024 | 2024-01-01 to 2024-05-31 | 2024-07-01 to 2024-07-31 | 6 | 2 | 1 | -$36.00 | -$36.00 | 0.0% | 0.00 | $36.00 | 5.00x / 500.0% | 0 | 0 | 0.05 | `True` | `not_available` |
| September 2024 | 2024-01-01 to 2024-07-31 | 2024-09-01 to 2024-09-30 | 8 | 2 | 1 | -$36.00 | -$36.00 | 0.0% | 0.00 | $36.00 | 5.00x / 500.0% | 0 | 0 | 0.05 | `True` | `not_available` |
| November 2024 | 2024-01-01 to 2024-09-30 | 2024-11-01 to 2024-11-30 | 10 | 2 | 1 | -$36.00 | -$36.00 | 0.0% | 0.00 | $36.00 | 5.00x / 500.0% | 0 | 0 | 0.05 | `True` | `not_available` |

`cost_share` is the evaluator's `costs / abs(gross PnL)` ratio. It is `5.00` (500%) in each
isolated regime because the synthetic one-bar price movement produces `$9.00` gross PnL against
`$45.00` modeled round-trip costs. This is a synthetic contract result, not a claim about actual
MNQ transaction costs or tradability.

`average_r` is gross PnL divided by gross stop risk; modeled costs are excluded from the R
numerator.

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
- `chronology_check`: `True`
- `signal_provenance_check`: `not_available`
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
| `chronology_check` | `True` |
| `signal_provenance_check` | `not_available` |
| Configuration hash | `95524d8d563c8c03cf966112a4f10e3d9d5fbdfbe078fbc20de9b4c597443daf` |

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
- `chronology_check == True` for all reported windows;
- signal-level provenance evidence is not available, so the report does not claim full no-lookahead proof;
- no controlled-backtest risk-limit breaches in the synthetic regime or holdout reports.

**Performance evidence not established:**

- The fixture is synthetic, tiny, and hand-constructed. It is not licensed or real exchange data.
- Synthetic metrics are not historical performance and cannot establish profitability, edge,
  robustness, or live execution quality.
- One generated setup and one trade per regime provide no meaningful statistical sample.
- The end-of-data close and generated price path are fixture mechanics, not modeled market fills.

## MES/MGC extension gate (NOT CLEARED)

The MES/MGC extension gate is **NOT CLEARED** by this synthetic report. It cannot be cleared until
an appropriate real/licensed MNQ sample and approved rollover policy are available and that MNQ
sample passes all of the following gates:

1. **No chronology failures:** every chronological evaluation has `chronology_check == True`, with
   no train/test chronology violation. Signal-level provenance remains a separate, unavailable
   evidence item in this synthetic report.
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
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=120 \
  research/notebooks/mnq_objective_strategy_validation.ipynb \
  --output /tmp/mnq_objective_strategy_validation.executed.ipynb
```

The deterministic report path uses `StrategyConfig`,
`config.validate_fixed_contract()`, `make_multi_month_fixture`, `run_backtest`, and
`walk_forward_evaluate`, with the exact windows and metrics recorded above. The notebook cell is
the executable report-regeneration path; its synthetic status remains blocked for historical
validation and does not clear MES/MGC.
