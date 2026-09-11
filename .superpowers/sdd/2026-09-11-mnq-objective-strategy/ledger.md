# SDD ledger — plan: docs/superpowers/plans/2026-09-11-mnq-objective-strategy.md

**Base commit:** `9438dec6c711c5f9ac4d95013128da233716c978`
**Workspace:** `/Users/theinnerchild/quant-references/machine-learning-for-trading-mnq-strategy`
**Branch:** `feat/mnq-objective-strategy`

## Plan scan

| Tasks | Shared interface or file | Finding | Ruling |
|---|---|---|---|
| Task 4 → Task 5 | `CostModel`, `RiskDecision` | Backtest must apply Task 4 sizing and account guards. | Task 4 owns these types; Task 5 imports them and applies the guard before entry. |
| Task 5 → Task 6 | `StrategyConfig` | Task 5 tests and implementation require config before Task 6's listed config task. | Execute Task 6's `config.py`, fixtures, and config tests before Task 5; defer Task 6 notebook/README until Task 5 is complete. |
| Task 6 → Task 5 | `make_*_fixture()` helpers | Task 5's required failing tests reference helpers listed under Task 6. | Task 6's config/fixtures subset supplies deterministic helpers before Task 5; Task 5 may add task-local helpers only when a scenario is backtest-specific. |
| Task 5 → Task 7 | Backtest trade rows and metrics | Task 7 needs config hash, costs, drawdown, risk breaches, and per-setup metrics. | Task 5 must expose stable trade/result fields and evaluation metrics for Task 7. |
| Task 6 → Task 7 | Config hash and deterministic fixture | Task 7 needs a fixed sample and reproducible configuration. | Use the deterministic MNQ fixture as the fixed sample, record it explicitly as synthetic, and make no profitability claim. |
| Task 7 | Historical data source | Plan asks for a fixed historical sample but provides no licensed MNQ source or rollover policy. | Use the deterministic synthetic fixture for the validation gate; document the missing production data source and rollover policy as a limitation rather than inventing market history. |
| Task 5 | Entry price versus slippage | The plan's required fixture asserts the next-bar raw open is the entry price, while event ordering also requires configured slippage. | Preserve `entry_price` as the observed next-bar open for auditability and charge configured slippage in `total_costs`; use any separate adjusted fill field only if needed without changing the required output contract. |

## Execution order

1. Task 4 risk controls.
2. Task 6 config and fixtures subset.
3. Task 5 backtest and walk-forward evaluation.
4. Task 6 notebook and README completion.
5. Task 7 validation report and extension gate.

## Completed gates

- Task 4: fix round 1/5 (4 addressed, 0 open; commits `edefceb`..`62a92bd`).
- Task 4: complete (commits `edefceb`..`62a92bd`, review clean).
- Task 6 foundation: implementation commit `eb9d730` recorded; scoped review pending.
- Task 6 foundation: fix rounds 1/5 (6 addressed, 0 open; commits `dbead05`..`9a62f95`).
- Task 6 foundation: complete (commits `eb9d730`..`9a62f95`, review clean).
- Task 5 ruling: call `StrategyConfig.validate_fixed_contract()` before execution and keep signal thresholds fixed; no test-window parameter selection.
- Task 5: implementation commit `412d5b3` recorded; scoped review pending.
- Task 5: fix round 1/5 (5 addressed, 0 open; commit `b59913c`).
- Task 5: complete (commits `412d5b3`..`b59913c`, review clean).
- Task 6: notebook/README commit `bebda9d` recorded; notebook review fix rounds addressed by `bc4a178` and `abdc00d`.
- Task 6: complete (commits `eb9d730`..`abdc00d`, review clean).
- Task 7: implementation commit `abf017d` recorded; scoped review pending.
- Task 7: final whole-branch review completed locally after fixes; verdict **CLEAR**; commit `daad550` finalized progress.
- Task 7: branch closure tagged `mnq/v1-review-cleared` (@ `daad550`).

## Closed

- MNQ v1 branch `feat/mnq-objective-strategy` is marked complete with tag `mnq/v1-review-cleared`.
- MES/MGC extension gate remains **NOT CLEARED** by design (no licensed MNQ historical data or rollover policy in checkout).
- Remote push/PR requires GitHub authentication; see next steps below.

## Next steps for closure

1. Authenticate GitHub CLI (`gh auth login`) or configure Git for `https://github.com/stefan-jansen/machine-learning-for-trading.git`.
2. Push the branch and tag:
   - `git push origin feat/mnq-objective-strategy`
   - `git push origin mnq/v1-review-cleared`
3. Open a PR from `feat/mnq-objective-strategy` into `main`, referencing `mnq-baseline-validation.md` and `task-7-report.md`.
4. After merge, delete the local/remote branch if no longer needed.
