# Task 4 Report: Position Sizing, Costs, and Account Guards

## Files Created/Modified

- **Created**: `research/mnq_strategy/risk.py` — Risk management module with `CostModel`, `RiskDecision`, `calculate_position_size`, and `DailyRiskGuard`
- **Created**: `tests/research/test_risk.py` — 16 focused tests covering all requirements
- **Modified**: `research/mnq_strategy/__init__.py` — Exported risk module public API

## Commands & Results

### RED Phase (Tests First)
```bash
uv run pytest tests/research/test_risk.py -q
```
**Result**: 1 error (ModuleNotFoundError: No module named 'research.mnq_strategy.risk') — Expected failure

### GREEN Phase (Implementation)
```bash
uv run pytest tests/research/test_risk.py -q
```
**Result**: 16 passed in 0.06s

### Full Test Suite
```bash
uv run pytest tests/research/ -q
```
**Result**: 60 passed in 0.26s (all research tests pass)

### Lint & Format
```bash
uv run ruff check research/mnq_strategy/risk.py tests/research/test_risk.py research/mnq_strategy/__init__.py
uv run ruff format --check research/mnq_strategy/risk.py tests/research/test_risk.py research/mnq_strategy/__init__.py
```
**Result**: All checks passed, 3 files already formatted

### Commit
```bash
git add research/mnq_strategy/risk.py tests/research/test_risk.py research/mnq_strategy/__init__.py
git commit -m "feat: add position sizing, costs, and account guards for MNQ strategy"
```
**Result**: Commit edefceb

## TDD Evidence

### RED Phase (Before Implementation)
```
tests/research/test_risk.py:3: ImportError while importing test module
ModuleNotFoundError: No module named 'research.mnq_strategy.risk'
```

### GREEN Phase (After Implementation)
```
============================= test session starts ==============================
collected 16 items
tests/research/test_risk.py ................                             [100%]
============================== 16 passed in 0.06s ==============================
```

### Test Coverage (16 tests)
| Test | Purpose |
|------|---------|
| `test_rejects_when_minimum_four_contracts_exceed_250_dollars` | Brief requirement: stop=40, requested=10 rejected with "250" in reason |
| `test_caps_contracts_between_four_and_ten` | Contracts stay in [4, 10] range |
| `test_daily_guard_stops_after_two_losses_or_400_dollars` | Brief requirement: 2 consecutive losses blocks trading |
| `test_daily_guard_stops_after_400_dollar_loss` | Daily loss >= 400 blocks trading |
| `test_daily_guard_resets_on_win_or_zero` | Win or zero resets consecutive loss streak |
| `test_daily_guard_reset_day_restores_state` | `reset_day()` restores initial state |
| `test_validate_stop_points_positive` | stop_points > 0 enforced |
| `test_validate_costs_nonnegative` | commission/slippage >= 0 enforced |
| `test_validate_requested_contracts_in_range` | requested_contracts in [4, 10] enforced |
| `test_risk_decision_fields` | RiskDecision has all required fields |
| `test_calculate_gross_risk` | gross_risk = stop_points * 2 * contracts |
| `test_calculate_estimated_costs` | Round-trip: commission*2 + slippage*2*2*contracts |
| `test_total_risk_is_gross_plus_costs` | total_risk = gross_risk + estimated_costs |
| `test_accepted_when_total_risk_under_250` | Accepted when total_risk < 250 |
| `test_rejected_when_total_risk_exceeds_250` | Rejected when total_risk >= 250 |
| `test_default_cost_model_values` | Default costs recorded, not silently assumed |

## Self-Review

### Requirements Met ✅
- **CostModel dataclass**: `commission_per_contract`, `slippage_points` with validation
- **RiskDecision dataclass**: `accepted`, `contracts`, `gross_risk`, `estimated_costs`, `total_risk`, `reason`
- **MNQ point value**: Exactly 2 USD/point/contract (constant `MNQ_POINT_VALUE = 2.0`)
- **Round-trip costs**: `commission_per_contract * contracts * 2 + slippage_points * 2 * contracts * 2`
- **Gross risk**: `stop_points * 2 * contracts`
- **Total risk**: `gross_risk + estimated_costs`
- **Validation**: stop_points > 0, nonnegative costs, requested_contracts in [4, 10]
- **No silent downsize**: Rejected with reason containing "250" when 4+ contracts exceed ceiling
- **Brief's stop=40/requested=10 case**: Rejected (gross=160, costs=30+40=70, total=230 → wait, let me recalculate: 40*2*10=800 gross, costs=1.50*10*2 + 0.50*2*10*2 = 30+20=50, total=850 → rejected with "250" in reason) ✅
- **DailyRiskGuard**: defaults max_daily_loss=400.0, max_consecutive_losses=2
- **Track realized daily PnL & consecutive losses**: Loss = pnl < 0, win/zero resets streak
- **can_trade()**: False at or beyond either limit
- **reset_day()**: Restores initial state

### Code Quality ✅
- Follows existing Polars/Python conventions in research/mnq_strategy
- Uses frozen dataclasses for immutability
- Type hints throughout
- Clear docstrings
- Constants for magic numbers (MNQ_POINT_VALUE, MAX_TOTAL_RISK, MIN_CONTRACTS, MAX_CONTRACTS)
- Proper error messages with validation context

## Concerns

None. Implementation is minimal, focused, and passes all tests including the exact test cases from the brief.

## Report Path

/Users/theinnerchild/quant-references/machine-learning-for-trading-mnq-strategy/.superpowers/sdd/2026-09-11-mnq-objective-strategy/task-4-report.md

## Review Fixes

Addressed the four review findings without changing `pyproject.toml` or `uv.lock`:

- Added finite-value validation for `CostModel` costs, `stop_points`, and `DailyRiskGuard` thresholds.
- Added a daily-loss latch that remains active until `reset_day()`; non-loss trades still reset only the consecutive-loss streak.
- Updated position sizing to search downward from the requested size and return the largest affordable size in `[4, requested_contracts]`. A request is rejected only when four contracts cannot fit strictly below the 250 USD ceiling.
- Updated the rejection reason to say the minimum position is "not below" the ceiling, including at the exact 250 USD boundary.
- Added focused tests for non-finite inputs, daily-loss latching, safe downsizing, minimum-four rejection, and the boundary reason.

### Review Verification

```bash
uv run pytest tests/research/test_risk.py -q
```

```text
collected 33 items
tests/research/test_risk.py .................................            [100%]
33 passed in 0.14s
```

```bash
uv run pytest tests/research/ -q
```

```text
collected 77 items
tests/research/test_mnq_data_contract.py ................                [ 20%]
tests/research/test_risk.py .................................            [ 63%]
tests/research/test_signals.py ................                          [ 84%]
tests/research/test_volume_profile.py ............                       [100%]
77 passed in 0.22s
```

```bash
uv run ruff check research/mnq_strategy/risk.py tests/research/test_risk.py
uv run ruff format --check research/mnq_strategy/risk.py tests/research/test_risk.py
uv run python -m compileall -q research/mnq_strategy/risk.py tests/research/test_risk.py
```

```text
All checks passed!
2 files already formatted
```

The compile command completed without output or errors.

### Review Commit

The existing Task 4 commit remains unchanged. The review fixes are staged for a separate follow-up commit.

## Review Fix Commit

```bash
git add research/mnq_strategy/risk.py tests/research/test_risk.py .superpowers/sdd/2026-09-11-mnq-objective-strategy/task-4-report.md
git commit -m "fix: harden MNQ risk controls"
```

The follow-up commit appends these review fixes to the existing Task 4 commit.
