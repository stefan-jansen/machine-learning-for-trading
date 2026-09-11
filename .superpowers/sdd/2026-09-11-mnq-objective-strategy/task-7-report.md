# Task 7 Process Report

## Scope and status

Task 7 completed the repository checks and deterministic synthetic mechanics validation, recorded
the MNQ baseline limitations, defined the MES/MGC extension gate, and documented why historical
MNQ validation remains blocked.

Changed deliverables:

- `README.md`: focused link to the baseline report and extension-gate/real-data limitation note
  inside the existing MNQ workflow section.
- `research/reports/mnq-baseline-validation.md`: complete baseline validation report.
- `.superpowers/sdd/2026-09-11-mnq-objective-strategy/task-7-report.md`: this process report.

No subagents were dispatched. No source module, notebook, test, `pyproject.toml`, `uv.lock`, or
`progress.md` changes were made for Task 7. Existing unrelated working-tree changes were not
reverted.

## Exact required validation

### Research tests

Command:

```bash
uv run pytest tests/research -q
```

Observed output:

```text
============================= test session starts ==============================
platform darwin -- Python 3.14.7, pytest-9.0.3, pluggy-1.6.0
collected 140 items

140 passed in 0.48s
```

Result: **PASS, 140 passed, 0 failed**.

### Installation verification

Command:

```bash
uv run python scripts/verify_installation.py
```

Final required-check result:

```text
PASS  All required checks passed (149/155).
5 optional packages not installed (Docker-only extras).
```

The category counts reported as passing were: Core Data Science `7/7`, Visualization `8/8`,
Machine Learning `24/24`, Deep Learning `8/8`, NLP/Transformers `4/4`, Time Series `16/16`,
Causal Inference `6/6`, Portfolio/Finance `5/5`, Reinforcement Learning `2/2`, Data Sources `7/7`,
Technical Analysis `2/2`, Synthetic Data `1/1`, Utilities `12/12`, Jupyter `9/9`,
Storage/Formats `4/4`, ML4T Libraries `6/6`, ML4T Sub-modules `5/5`, and Repo Packages `10/10`.

The script displayed Runtime `9/10`: CUDA was skipped because this macOS host has no GPU and is
running CPU-only. The script's exit status remained zero because all required checks passed. The
five optional Docker-only failures were `gensim`, `signatory`, `arcticdb`, `clickhouse-connect`,
and `influxdb-client`; they are not required for standard usage. The command also emitted
non-blocking warnings for macOS Torch redirect support, deprecated `websockets.legacy` and
`asyncio.iscoroutinefunction` usage, and unavailable optional `botocore` Bedrock/SageMaker stream
shape loading. These warnings did not change the required-check result.

## Deterministic baseline execution

The baseline used these existing interfaces and no external data:

- `StrategyConfig()`
- `config.validate_fixed_contract()` -> `True`
- `make_multi_month_fixture()` -> 12 deterministic synthetic bars
- `run_backtest()`
- `walk_forward_evaluate()`

The fixed configuration hash was:

```text
95524d8d563c8c03cf966112a4f10e3d9d5fbdfbe078fbc20de9b4c597443daf
```

The fixture spans `2024-01-08 10:00:00-05:00` through `2024-11-08 10:05:00-05:00`, uses
five-minute bars, represents timestamps in `America/New_York` (with the source timestamp column
in UTC), and contains no missing or unclosed bars. Missing/irregular cadence is not silently
filled: the normalized contract rejects irregular non-five-minute input, while the backtest
excludes unclosed observations from entry and exit decisions.

The default cost model is `$1.50` commission per contract per side and `0.50` points slippage per
side. The fixture report records the full execution rows, three isolated chronological regimes
(July, September, and November), one November holdout, trade count, net PnL, expectancy, win
rate, profit factor, maximum drawdown, cost share, daily/consecutive breach counts, average R,
per-setup metrics, and `lookahead_check`.

The report explicitly separates these mechanical checks from performance evidence. The sample is
synthetic/research-only and has no licensed or real MNQ history or rollover policy, so its metrics
are not historical performance and cannot establish profitability. Mechanical validation is
complete; historical MNQ validation is blocked, and the MES/MGC extension gate is **NOT CLEARED**
until an appropriate real/licensed MNQ sample and approved rollover policy are available and pass
the stated gate.

The three chronological regimes were reproduced as three separate one-window
`walk_forward_evaluate` calls because `evaluation.py` rejects overlapping train/test windows in a
single multi-window call. A separate one-window call produced the November holdout. The expanding
train contexts are reported to make the chronological information available before each test
segment explicit; fixed thresholds mean they were not used for parameter selection.

`average_r` is gross PnL divided by gross stop risk; modeled costs are excluded from the R
numerator.

## Extension gate and limitations

The report defines the MES/MGC gate exactly as:

1. no lookahead failures;
2. no risk-limit breaches in the controlled backtest;
3. stable results across at least three chronological regimes; and
4. a cost/drawdown holdout report.

MES/MGC require instrument-specific point values, tick sizes, sessions, data-quality checks, and
fixed-contract configuration validation. No MES/MGC code was added.

The baseline report also records the required limitations: research-only status, historical fills
not proving live quality, volume-profile granularity dependence, current prop-firm dashboard rules,
FacturationApp screenshots remaining in localStorage rather than synchronized storage, and the
absence of real MNQ data and an approved rollover policy.

## Report and repository checks

The report-consistency check is reproducible from the repository root with this exact command. It
verifies the report path, required status/gate/limitation phrases, the active configuration hash,
the three regime labels, the separate-call wording, the holdout label, and the README report link:

```bash
uv run python - <<'PY'
from pathlib import Path

from research.mnq_strategy.config import StrategyConfig

root = Path('.')
readme = (root / 'README.md').read_text(encoding='utf-8')
report = (root / 'research/reports/mnq-baseline-validation.md').read_text(encoding='utf-8')
process = (root / '.superpowers/sdd/2026-09-11-mnq-objective-strategy/task-7-report.md').read_text(encoding='utf-8')
required_report_phrases = [
    'mechanical validation complete',
    'Historical MNQ',
    'validation is blocked',
    'MES/MGC extension gate is **NOT CLEARED**',
    'three separate one-window calls',
    'average_r',
    'appropriate real/licensed MNQ sample',
    'approved rollover policy',
    'July 2024',
    'September 2024',
    'November 2024',
    'Holdout window',
    'lookahead_check',
]
for phrase in required_report_phrases:
    assert phrase in report, phrase
assert 'research/reports/mnq-baseline-validation.md' in readme
assert '**NOT CLEARED**' in readme
assert StrategyConfig().config_hash in report
assert 'separate one-window' in process
print(f'report consistency: PASS ({len(required_report_phrases)} required phrases; hash/link/call checks passed)')
PY
```

No Markdown-specific linter result is claimed.

The following whitespace check was run:

```bash
git diff --check
```

Result: completed without whitespace errors.

Before staging, the intended Task 7 paths were the only Task 7 paths changed:

```text
README.md
research/reports/mnq-baseline-validation.md
.superpowers/sdd/2026-09-11-mnq-objective-strategy/task-7-report.md
```

Pre-existing unrelated changes remained untouched in `progress.md`, `pyproject.toml`, `uv.lock`,
and `docs/superpowers/`.

## Commit scope

Only `README.md` and the two Task 7 report files were staged for the concise conventional commit.
The final status inspection was used to confirm that unrelated existing changes were not staged.

## Final review fix

The final review corrections were applied only to the three permitted files:

- changed the report status and decision from an unqualified completion statement to mechanical
  validation complete, historical MNQ validation blocked, and MES/MGC gate **NOT CLEARED**;
- stated that an appropriate real/licensed MNQ sample and approved rollover policy are
  prerequisites before the MES/MGC gate can clear;
- documented the three separate one-window regime calls, the separate November holdout call, and
  the reason for the expanding train contexts;
- defined `average_r` as gross PnL divided by gross stop risk with modeled costs excluded from the
  R numerator;
- recorded the exact reproducible report-consistency assertion command above;
- aligned the README gate and caveat wording with the blocked historical-validation status.

Post-fix checks:

- `uv run pytest tests/research -q`: `140 passed`, `0 failed`.
- `uv run python scripts/verify_installation.py`: required checks passed, `149/155`; CUDA was
  skipped on the CPU-only macOS host and five optional Docker-only packages remained absent.
- The exact report-consistency command above: `PASS`.
- `git diff --check`: completed without whitespace errors.
