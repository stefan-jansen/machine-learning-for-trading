# Task 5 Report: Event-Driven MNQ Backtest and Walk-Forward Evaluator

## Status

Implemented and verified locally. The requested Task 5 files are the only
Task 5 source/test files changed. Existing unrelated working-tree changes in
`pyproject.toml`, `uv.lock`, `docs/superpowers/`, and `progress.md` were left
untouched.

## Implementation

- Added `research/mnq_strategy/backtest.py` with chronological Polars input
  validation, fixed-contract validation, next-bar confirmation entry, one
  open position, direction-aware stop/target levels, conservative stop-first
  same-bar resolution, end-of-data exits, risk sizing, `DailyRiskGuard`, and
  machine-readable rejection rows.
- Preserved the raw observed `entry_price` and exposed separate adjusted
  execution fields while accounting for commission/slippage in
  `total_costs` and `net_pnl`.
- Added `research/mnq_strategy/evaluation.py` with immutable
  `WalkForwardWindow`, chronological non-overlap validation, fixed-threshold
  selection policy metadata, look-ahead checks, zero-safe metrics, and
  per-setup reporting.
- Added focused deterministic tests in
  `tests/research/test_backtest.py` and `tests/research/test_evaluation.py`.
- No `__init__.py` export was required; the public interfaces are available
  from their specified modules.

## TDD Evidence

### RED

Command:

```bash
uv run pytest tests/research/test_backtest.py tests/research/test_evaluation.py -q
```

Result: expected collection failure before implementation:

```text
collected 0 items / 2 errors
ModuleNotFoundError: No module named 'research.mnq_strategy.backtest'
ModuleNotFoundError: No module named 'research.mnq_strategy.evaluation'
```

### GREEN

After implementation and focused edge-case additions:

```text
uv run pytest tests/research/test_backtest.py tests/research/test_evaluation.py -q
19 passed in 0.24s
```

## Verification

Final combined verification:

```text
uv run pytest tests/research -q
133 passed in 0.35s

uv run ruff check research/mnq_strategy/backtest.py research/mnq_strategy/evaluation.py tests/research/test_backtest.py tests/research/test_evaluation.py
All checks passed!

uv run ruff format --check research/mnq_strategy/backtest.py research/mnq_strategy/evaluation.py tests/research/test_backtest.py tests/research/test_evaluation.py
4 files already formatted

uv run python -m compileall -q research/mnq_strategy/backtest.py research/mnq_strategy/evaluation.py tests/research/test_backtest.py tests/research/test_evaluation.py
Completed without output or errors.

git diff --check
Completed without output or errors.
```

## Coverage of Requested Cases

- Fixed-contract validation before execution/evaluation.
- Closed confirmation rows only and first eligible next-bar entry.
- Long and short PnL, direction-aware stops/targets, and raw entry auditability.
- Conservative stop-first ordering when both stop and target are touched.
- One-position rejection with `position_already_open`.
- Commission/slippage cost accounting without hiding slippage in raw prices.
- Position-size rejection rows and daily guard latch behavior.
- Chronological walk-forward windows and no-lookahead metadata.
- Trade count, net PnL, expectancy, win rate, profit factor, drawdown, guard
  breach counts, average R, cost share, per-setup data, config hash, and
  selection policy.
- Empty, signal-free, invalid-window, invalid-schema, and no-entry cases.

## Concerns

- `test_results` is intentionally JSON-like (`list[dict]`) in the evaluation
  report, while `run_backtest` returns the typed Polars DataFrame required by
  the interface.
- The evaluator reports train/test row counts and timestamps as metadata; it
  does not tune thresholds, consistent with the fixed-contract ruling.
- The backtest accepts canonical `timestamp_ny` inputs and can also coerce a
  `timestamp` input into the configured timezone, but callers still need to
  provide the signal columns and explicit boolean `bar_closed` contract.

## Commit

The implementation files, focused tests, and this report are committed
together with a concise conventional commit message after the verification
commands above pass.

## Report Path

`/Users/theinnerchild/quant-references/machine-learning-for-trading-mnq-strategy/.superpowers/sdd/2026-09-11-mnq-objective-strategy/task-5-report.md`

## Review Fixes

Applied the five Task 5 review findings without touching the prohibited
project files:

- Normalized the selected `timestamp`/`timestamp_ny` to the configured
  timezone before chronology checks or train/test date masks. Added a UTC
  previous-New-York-date regression test and normalized timestamp metadata.
- Added `session_date` to every result row and reset evaluator daily and
  consecutive-loss counters at each new session date, matching
  `DailyRiskGuard`/backtest behavior.
- Repaired the overlapping fixture's declared entry times to equal each
  signal's next chronological bar while retaining overlapping windows and the
  one-position rejection case.
- Added fixed MNQ `point_value == 2.0` validation to
  `StrategyConfig.validate_fixed_contract()` and a backtest regression test.
- Filtered unclosed bars out of simulation while preserving an explicit
  `no_eligible_entry_bar` rejection for signals whose next bar is unclosed;
  unclosed OHLC rows cannot trigger exits or end-of-data fills.

### Review RED

```bash
uv run pytest tests/research/test_backtest.py tests/research/test_evaluation.py -q
```

```text
25 collected; 6 failed, 19 passed
```

The failures reproduced the fixture timing, timezone masking, session breach,
point-value, and unclosed-bar defects before the fixes.

### Review GREEN and Final Verification

```bash
uv run pytest tests/research/test_backtest.py tests/research/test_evaluation.py -q
25 passed in 0.26s

uv run pytest tests/research -q
139 passed in 0.23s
```

The final focused and complete research suites passed after the fixes. The
review regression additions preserve the raw-versus-adjusted entry/cost
semantics and conservative stop-first ordering.

After adding the multiple-window regression, the final rerun collected 26
focused tests and 140 research tests; the exact final results are recorded
below.

### Exact Fix Verification Commands

```bash
uv run ruff check research/mnq_strategy/backtest.py research/mnq_strategy/evaluation.py research/mnq_strategy/config.py research/mnq_strategy/fixtures.py tests/research/test_backtest.py tests/research/test_evaluation.py
```

```text
All checks passed!
```

```bash
uv run ruff format --check research/mnq_strategy/backtest.py research/mnq_strategy/evaluation.py research/mnq_strategy/config.py research/mnq_strategy/fixtures.py tests/research/test_backtest.py tests/research/test_evaluation.py
```

```text
6 files already formatted
```

```bash
uv run python -m compileall -q research/mnq_strategy/backtest.py research/mnq_strategy/evaluation.py research/mnq_strategy/config.py research/mnq_strategy/fixtures.py tests/research/test_backtest.py tests/research/test_evaluation.py
```

```text
Completed without output or errors.
```

```bash
uv run pytest tests/research/test_backtest.py tests/research/test_evaluation.py -q
```

```text
26 passed in 0.24s
```

```bash
uv run pytest tests/research -q
```

```text
140 passed in 0.21s
```

```bash
git diff --check
```

```text
Completed without output or errors.
```
