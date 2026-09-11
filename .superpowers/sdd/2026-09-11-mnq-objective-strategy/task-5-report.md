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
