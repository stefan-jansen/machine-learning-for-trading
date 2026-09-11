# Task 6 Cumulative Report

## Scope

This cumulative Task 6 report preserves the foundation evidence from the staged execution and records completion of the deferred notebook and README work.

The Task 6 deliverables are:

- `research/mnq_strategy/config.py`
- `research/mnq_strategy/fixtures.py`
- `tests/research/test_config.py`
- `research/notebooks/mnq_objective_strategy_validation.ipynb`
- focused MNQ workflow documentation in `README.md`

Existing unrelated working-tree changes in `pyproject.toml`, `uv.lock`, `docs/superpowers/`, and `progress.md` were left untouched.

## Foundation implementation evidence

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

## Foundation TDD evidence

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

### Foundation GREEN

After implementing the two modules, the focused test command passed:

```bash
uv run pytest tests/research/test_config.py -q
```

The staged foundation run observed `15 passed in 0.22s`; later review fixes expanded the final focused coverage to 36 tests.

## Foundation review fixes preserved

The following review findings were addressed before the deferred notebook work:

- Parsed every session boundary with `datetime.time.fromisoformat()` and rejected malformed values such as `99:99`.
- Validated the session contract as `09:30 < 16:00 == 16:00 < 18:00 == 18:00`.
- Validated `timezone` through `ZoneInfo`.
- Required `momentum_lookback` to be an integer and required `stop_points < target_points`.
- Replaced the mutable nested boundary mapping with a mapping-compatible immutable representation. `to_dict()` still returns a normal JSON-compatible dictionary, while post-construction mutation fails and cannot change `config_hash`.
- Added `validate_fixed_contract()`, which compares all Task 3 fixed thresholds against `signals.py` constants and the fixed LVN percentile. Changed thresholds raise a clear drift error before Task 5 execution.
- Repaired the cost, stop/target, daily guard, and multi-month fixtures so their entry and signal fields are explicit and deterministic.
- Expanded tests over canonical columns, UTC/New York timestamp zones, supplementary fixtures, invalid configurations, nested immutability, config identity drift, corrected fixture timing, and execution contracts.

## Deferred Task 6 completion

### Validation notebook

Created `research/notebooks/mnq_objective_strategy_validation.ipynb` as a valid nbformat 4.5 notebook with 11 concise cells. It:

- resolves the repository root from the current working directory without requiring a hard-coded machine path;
- imports `StrategyConfig`, `make_confirmed_signal_fixture`, `make_multi_month_fixture`, `run_backtest`, `WalkForwardWindow`, and `walk_forward_evaluate`;
- calls `config.validate_fixed_contract()` before execution;
- labels the default data as deterministic, synthetic, and research-only;
- runs the event-driven backtest and displays the first 20 input signal rows and first 20 backtest signal/trade rows;
- derives and displays a realized net-PnL equity curve and drawdown table;
- plots the curve with existing matplotlib when available and keeps the table as the clear fallback when matplotlib cannot be imported;
- prints the full walk-forward validation report, including `config_hash`, `lookahead_check`, window metadata, metrics, and fixed-threshold selection policy;
- asserts the report hash matches the active configuration and that the lookahead check is true;
- uses no downloads, credentials, broker/API access, or external data files by default.

The notebook intentionally uses the deterministic fixture path rather than a local file loader. A local normalized MNQ file can be substituted only after applying the existing timestamp, closed-bar, signal, cost, and evaluation contracts.

### README documentation

Added one focused `MNQ objective strategy research workflow` section to `README.md`. It documents exact repository-root commands for:

```bash
uv run pytest tests/research/test_config.py -q
uv run pytest tests/research -q
uv run jupyter nbconvert --to notebook --execute \
  research/notebooks/mnq_objective_strategy_validation.ipynb \
  --output-dir /tmp \
  --output mnq_objective_strategy_validation.executed.ipynb
```

The section explicitly states that the default run is synthetic/research-only, requires no live data or credentials, and does not claim profitability or live execution quality.

## Deferred-work verification

### Notebook JSON and schema

```bash
uv run python - <<'PY'
import json
from pathlib import Path
import nbformat
path = Path('research/notebooks/mnq_objective_strategy_validation.ipynb')
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
raw = json.loads(path.read_text(encoding='utf-8'))
assert raw['nbformat'] == 4
assert all('id' in cell for cell in raw['cells'])
print(f'validated {path}: {len(nb.cells)} cells, nbformat {raw["nbformat"]}.{raw["nbformat_minor"]}')
PY
```

Observed result: `validated research/notebooks/mnq_objective_strategy_validation.ipynb: 11 cells, nbformat 4.5`.

### Headless notebook smoke execution

```bash
rm -f /tmp/mnq_objective_strategy_validation.executed.ipynb && \
uv run jupyter nbconvert --to notebook --execute \
  research/notebooks/mnq_objective_strategy_validation.ipynb \
  --output-dir /tmp \
  --output mnq_objective_strategy_validation.executed.ipynb
```

Observed result: `NbConvertApp` converted the notebook and wrote `48469 bytes` to `/tmp/mnq_objective_strategy_validation.executed.ipynb` with exit status 0. The generated executed notebook was not added to the repository.

### Focused tests

```bash
uv run pytest tests/research/test_config.py -q
```

Observed result:

```text
collected 36 items
36 passed in 0.11s
```

### Full research suite

```bash
uv run pytest tests/research -q
```

Observed result:

```text
collected 140 items
140 passed in 0.19s
```

### Lint, formatting, and whitespace

```bash
uv run ruff check research/notebooks/mnq_objective_strategy_validation.ipynb \
  research/mnq_strategy/config.py research/mnq_strategy/fixtures.py \
  research/mnq_strategy/backtest.py research/mnq_strategy/evaluation.py \
  tests/research/test_config.py
uv run ruff format --check research/mnq_strategy/config.py research/mnq_strategy/fixtures.py \
  research/mnq_strategy/backtest.py research/mnq_strategy/evaluation.py \
  tests/research/test_config.py
git diff --check
```

Observed results:

- Ruff check: `All checks passed!`
- Ruff format check: `5 files already formatted`
- `git diff --check`: completed without whitespace errors.

`markdownlint` and `markdownlint-cli2` were not installed in the environment, so no Markdown-specific executable check was available. The README diff was inspected and is limited to the focused MNQ section.

## Scope and concerns

- No prohibited source files were modified for the deferred work: `pyproject.toml`, `uv.lock`, `risk.py`, `backtest.py`, `evaluation.py`, `fixtures.py`, and `config.py` remain outside this change.
- The notebook uses synthetic fixture data only by default. Its PnL, equity curve, and drawdown are contract-smoke outputs, not evidence of a trading edge or profitability.
- The notebook does not introduce dependencies; Polars, IPython, matplotlib, Jupyter, nbconvert, and nbformat are already available in the repository environment.
- The executed notebook output remains outside the repository at `/tmp/mnq_objective_strategy_validation.executed.ipynb`.
- Pre-existing unrelated working-tree changes in `pyproject.toml`, `uv.lock`, `progress.md`, and `docs/superpowers/` were not reverted or staged.

## Report path

`/Users/theinnerchild/quant-references/machine-learning-for-trading-mnq-strategy/.superpowers/sdd/2026-09-11-mnq-objective-strategy/task-6-report.md`

## Task 6 review fix: initial equity baseline

### Finding

The notebook's drawdown table started at the first realized trade instead of at zero equity.
For the default synthetic fixture, the first net result is `-36.0`, so the previous calculation
reported a drawdown of `0.0` because the first negative equity value also became the running peak.
This differed from `evaluation._max_drawdown`, whose initial peak is zero and therefore reports a
drawdown of `36.0` for the same PnL sequence.

### Fix

- Added an explicit zero-PnL, zero-equity baseline row before closed trades in
  `research/notebooks/mnq_objective_strategy_validation.ipynb`.
- Kept the existing cumulative net-PnL calculation and drawdown formula, so the notebook now
  follows the evaluator's initial-peak semantics.
- Added notebook assertions for the initial equity value and the default fixture drawdown:
  `equity_curve["equity"][0] == 0.0` and `equity_curve["drawdown"].max() == 36.0`.
- The executed notebook output shows the baseline row followed by `net_pnl=-36.0`,
  `equity=-36.0`, and `drawdown=36.0`.

### Fix verification

```bash
uv run python - <<'PY'
import json
from pathlib import Path
import nbformat
path = Path('research/notebooks/mnq_objective_strategy_validation.ipynb')
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
raw = json.loads(path.read_text(encoding='utf-8'))
assert raw['nbformat'] == 4
assert all('id' in cell for cell in raw['cells'])
print(f'notebook validation: PASS ({len(nb.cells)} cells, nbformat {raw["nbformat"]}.{raw["nbformat_minor"]})')
PY
```

Observed result: `notebook validation: PASS (11 cells, nbformat 4.5)`.

```bash
rm -f /tmp/mnq_objective_strategy_validation.executed.ipynb && \
uv run jupyter nbconvert --to notebook --execute \
  research/notebooks/mnq_objective_strategy_validation.ipynb \
  --output-dir /tmp \
  --output mnq_objective_strategy_validation.executed.ipynb
```

Observed result: headless conversion passed and wrote `64667 bytes` to
`/tmp/mnq_objective_strategy_validation.executed.ipynb`. The rendered table contained:

```text
null                           0.0     0.0    0.0
2024-01-08 10:10:00 EST      -36.0   -36.0   36.0
```

The executed notebook was not added to the repository.

```bash
uv run pytest tests/research -q
uv run ruff check research/notebooks/mnq_objective_strategy_validation.ipynb
uv run ruff format --check research/notebooks/mnq_objective_strategy_validation.ipynb
uv run python -m compileall -q research/mnq_strategy
git diff --check
```

Observed results:

- `tests/research`: `140 passed in 0.37s`.
- Ruff check: `All checks passed!`.
- Ruff format check: `1 file already formatted`.
- Compileall completed without output or errors.
- `git diff --check` completed without whitespace errors.

The first format check correctly identified the edited notebook as needing formatting; Ruff format
was run once, then the notebook validation, headless execution, lint, format, compile, and diff
checks were rerun successfully.
