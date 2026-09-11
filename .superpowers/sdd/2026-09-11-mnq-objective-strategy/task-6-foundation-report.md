# Task 6 Foundation Report

## Scope

This staged Task 6 execution implements only the configuration and fixture foundation required before Task 5:

- `research/mnq_strategy/config.py`
- `research/mnq_strategy/fixtures.py`
- `tests/research/test_config.py`

The Task 6 notebook and `README.md` command documentation were intentionally deferred until after Task 5, as required. `pyproject.toml` and `uv.lock` were not modified by this work; their existing working-tree changes were left untouched.

## Implementation

### `StrategyConfig`

- Added a frozen dataclass with the approved MNQ defaults:
  - instrument `MNQ`
  - timezone `America/New_York`
  - five-minute bars
  - 40% value area
  - 4-10 contract bounds
  - $250 maximum trade risk
  - $400 daily stop
  - two-loss consecutive-loss guard
  - $2.00 MNQ point value
- Imports and reuses Task 4's `CostModel` and its MNQ risk constants; no duplicate cost model was added.
- Records explicit Task 3 signal thresholds, including rejection, momentum, and LVN percentile values.
- Records explicit finite backtest defaults: `stop_points=10.0` and `target_points=20.0`.
- Records serializable New York session boundaries for RTH, maintenance, and overnight transitions.
- Provides `to_dict()` as a plain JSON-compatible representation.
- Provides a SHA-256 `config_hash` over canonical JSON using sorted keys and compact separators.
- Validates finite and ordered configuration values without network calls, credentials, or data downloads.

### Deterministic fixtures

Added fresh Polars DataFrame factories for:

- confirmed next-bar entry with a closed signal and next-bar open `100.25`
- overlapping eligible entry windows
- chronological multi-month 2024 train/test data with holdout signals
- RTH/maintenance/overnight transitions
- previous-session profile attachment
- midnight rejection
- LVN break/retest
- 10:00 confirmation
- stop/target execution
- costs
- daily guard PnL sequences

The market fixtures use explicit UTC and `America/New_York` timestamp columns, session metadata, OHLCV, `bar_closed`, and signal metadata columns. They are pure in-memory data with no external files.

## TDD Evidence

### RED

After writing `tests/research/test_config.py` first, the exact required command was run:

```bash
uv run pytest tests/research/test_config.py -q
```

Observed failure:

```text
collected 0 items / 1 error
ModuleNotFoundError: No module named 'research.mnq_strategy.config'
```

This was the expected pre-implementation failure because `StrategyConfig` did not yet exist.

### GREEN

After implementing the two modules, the focused test command passed:

```bash
uv run pytest tests/research/test_config.py -q
```

Observed result:

```text
collected 15 items
15 passed in 0.22s
```

## Verification

The complete research test suite passed:

```bash
uv run pytest tests/research -q
```

Observed result:

```text
collected 93 items
93 passed in 0.14s
```

The focused quality checks passed:

```bash
uv run ruff check research/mnq_strategy/config.py research/mnq_strategy/fixtures.py tests/research/test_config.py
uv run ruff format --check research/mnq_strategy/config.py research/mnq_strategy/fixtures.py tests/research/test_config.py
uv run python -m compileall -q research/mnq_strategy/config.py research/mnq_strategy/fixtures.py tests/research/test_config.py
```

Results:

- Ruff check: all checks passed.
- Ruff format check: all three files already formatted.
- Python compile check: completed without errors.
- `git diff --check`: completed without whitespace errors.

## Self-review

- Scope is limited to the requested config, fixtures, and config tests, plus this report.
- Task 4 `CostModel` is imported and reused rather than duplicated.
- The required defaults and all fixed signal thresholds are explicit and serializable.
- Stop and target distances are named configuration values, not backtest literals.
- Configuration hashing is deterministic and sensitive to configuration changes.
- Required fixtures return fresh deterministic frames and preserve chronological ordering.
- The confirmed fixture exposes a closed signal and a matching next bar whose open is `100.25`.
- The overlapping fixture has two windows with a non-empty overlap.
- The multi-month fixture stays within 2024 and contains test-window signals.
- No network calls, credentials, downloads, notebook changes, or README changes were introduced.
- Existing unrelated changes in `pyproject.toml`, `uv.lock`, `.superpowers/sdd/2026-09-11-mnq-objective-strategy/progress.md`, and `docs/superpowers/` were not reverted or staged.

## Deferred work

The notebook smoke workflow and exact README run commands remain intentionally deferred until Task 5 is complete. Task 5 can now consume `StrategyConfig` and the deterministic fixtures for event-driven backtest and walk-forward tests.
