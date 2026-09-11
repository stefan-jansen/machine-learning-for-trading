# Task 7 Process Report

## Scope and status

Task 7 ran the full repository and strategy validation gate, recorded the deterministic MNQ
baseline, defined the MES/MGC extension gate, and documented the known limitations.

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
are not historical performance and cannot establish profitability.

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

Markdown/report consistency checks were run with a local Python assertion script. It verified the
report path, required headings and limitation/gate phrases, the active configuration hash, the
three regime labels, the holdout label, and the README report link. No `markdownlint` or
`markdownlint-cli2` executable was available, so no Markdownlint result is claimed.

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
