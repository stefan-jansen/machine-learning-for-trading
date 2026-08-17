# The Run Log

Every model training run, prediction set, causal-effect estimate, and backtest in
the case studies is recorded in a per-case-study **run log**. The run log is a
fully-deterministic experiment archive: each entry is content-addressed by a hash
of its complete configuration, every artifact it points to is reproducible from
that configuration, and nothing is anonymous or re-derivable only by re-running
the pipeline.

The run log lives at `case_studies/{case_study}/run_log/` and consists of three
things:

1. A SQLite index (`registry.db`) that catalogs every run.
2. Filesystem subdirectories holding the artifacts each run produced
   (`training/`, `predictions/`, `backtest/`).
3. A consistent identity scheme — three layers of SHA-256 hashes — that ties
   the index rows to the artifacts and to one another.

Chapter 6 motivates the run log as the centerpiece of disciplined trading
research. Without an experiment archive, walk-forward results cannot be audited,
hyperparameter searches cannot be compared apples-to-apples, and the chain from
model spec to portfolio P&L cannot be reconstructed. This chapter operationalizes
that idea for the nine case studies.

## What's in this document

- The configuration flow from a case study definition down to a single
  hyperparameter set, and how that flow becomes a hash.
- What goes into an identity, what is deliberately left out, and what follows
  from that when you re-run a notebook or change a configuration.
- The three-level entity model (training run → prediction set → backtest run)
  and the causal-runs side table.
- The filesystem layout under `run_log/`.
- The SQL schema of the eight tables a reader queries directly.
- How to query the run log from a notebook.
- How a model notebook writes to the run log, and how to run configurations of
  your own into a copy of it.
- How precomputed run logs are distributed as release artifacts so readers can
  run downstream analysis without retraining.

## Configuration flow

Configuration flows from the broadest scope (a case study) to the narrowest
(one trained model with specific hyperparameters):

```
config/setup.yaml                       Case study: universe, cadence, costs,
                                        labels, evaluation
    │
    ▼
config/training/{label}.yaml            Training menu: which model configs
                                        to train for this label
    │
    ▼
case_studies/config/{model_type}/...    Preset YAMLs: hyperparameters for
                                        each named config
    │
    ▼
request.resolve()                       Resolver: binds the preset to the label
                                        and feature files on disk, the fold
                                        dates, and the rows the fit must predict
    │
    ▼
SHA-256(canonical_json(identity))[:12]  Content-addressed hash → registry row
```

### Layer 1: case study setup (`config/setup.yaml`)

Each case study has a single `setup.yaml` that defines the trading problem. It is
the source of truth for the universe, decision cadence, execution defaults, cost
model, primary and secondary labels, walk-forward parameters, the Ch16–19 sweep
grid, and (where applicable) the causal estimand. Notebooks read it at runtime
rather than carrying their own copies of these values, and that is the intent
rather than an invariant the code enforces. Known exceptions: several training
stages keep hardcoded constants that override the preset they just loaded; the
shared GBM runner substitutes the seed on CPU; and in `sp500_options`, the
`12_backtest.py` and `13_portfolio_management.py` stages prefilter with a
hardcoded `LIQUID_QUANTILE = 0.20` ahead of the configured sweep value. See
[`docs/running-notebooks.md`](../docs/running-notebooks.md) for the full list and
what wins in each case.

```yaml
# case_studies/etfs/config/setup.yaml (excerpt)
strategy_id: etfs
universe:
  assets: [ACWI, AGG, GLD, QQQ, SPY, ...]
  n_assets: 100
decision:
  cadence: monthly_month_end
  snapshot: close
  execution_delay: next_bar_open
execution:
  initial_cash: 100_000
  share_type: integer
  allocator_lookback: 63
costs:
  class: material
  model: per_share_plus_spread
  per_share: 0.0035
  default_half_spread_usd: 0.02
  spread_convention: half_spread
labels:
  primary: fwd_ret_21d
  variants: [fwd_ret_5d]
  rebalance_step:            # one entry per label - required, never inferred
    fwd_ret_21d: 1
    fwd_ret_5d: 1
backtest:
  sweep:
    top_k_grid: {fwd_ret_21d: [10, 20], fwd_ret_5d: [10, 20]}
    cost_grid_bps: [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    allocators: [{name: score_weighted, method: score_weighted}, ...]
evaluation:
  n_splits: 8
  train_size: 10Y
  val_size: 1Y
causal:
  treatment: skip_recent_6_1
  confounders: [vol_21d, vol_126d, regime, yield_curve_slope]
```

The `costs` block is market-specific: ETFs and NASDAQ minute bars declare a
per-share-plus-spread model, futures declare commission per contract and spread
ticks, crypto declares a maker/taker fee schedule, and the equity panels declare
a per-leg basis-point range. Read the case study's own file rather than assuming
the shape.

### Layer 2: training menus (`config/training/{label}.yaml`)

Each label has a YAML file that lists, by family, the named configs to train.
This is the entry point for adding or removing models from a sweep:

```yaml
# case_studies/etfs/config/training/fwd_ret_21d.yaml
linear:
  - ols
  - ridge_a0.001
  - ridge_a0.01
  # ...
gbm:
  - default_mse
  - default_mae
  - leaves_7_mse
  # ...
deep_learning:
  - nlinear
  - lstm_h64
tabular_dl:
  - tabm_s
  - tabm_m
  - tabm_l
latent_factors:
  - pca
  - ipca
  - cae
  - sdf
  - sae
causal_dml:
  - dml
```

Each entry is a **config name** that resolves to a preset YAML in the shared
preset directory.

### Layer 3: preset YAMLs (`case_studies/config/{model_type}/`)

Preset YAMLs hold the hyperparameters for one named configuration. They live in
a shared directory organized by model type, so a single `ridge_a0.1` preset can
be reused by every case study that lists it in its training menu:

```
case_studies/config/
├── ols/ols.yaml
├── ridge/ridge_a0.001.yaml      # 12 ridge alpha values
├── ridge/ridge_a0.01.yaml
├── lasso/lasso_a0.01.yaml
├── elastic_net/enet_a0.01.yaml
├── logistic/logistic_l2_C1.0.yaml
├── lgb/default_huber.yaml       # 15 LightGBM configs
├── lgb/leaves_15_huber.yaml
├── lstm/lstm_h64.yaml
├── tcn/tcn.yaml
├── tsmixer/tsmixer.yaml
├── nlinear/nlinear.yaml
├── tabm/tabm_s.yaml             # TabM small / medium / large
├── pca/pca.yaml
├── ipca/ipca.yaml
├── cae/cae.yaml
├── sae/sae.yaml
├── sdf/sdf.yaml
└── dml/dml.yaml
```

A preset contains hyperparameters only. Metadata such as `family` and `library`
are derived from the directory name at load time.

```yaml
# case_studies/config/lgb/leaves_15_huber.yaml
checkpoint_interval: 50
max_iterations: 500
params:
  bagging_fraction: 0.8
  bagging_freq: 1
  feature_fraction: 0.7
  lambda_l1: 0.5
  lambda_l2: 5.0
  learning_rate: 0.05
  min_child_samples: 50
  num_leaves: 15
  objective: huber
  seed: 42
```

### Resolving the three layers

The three configuration layers describe a computation; they do not yet determine
one. `ridge_a1.0` names an estimator and a penalty, but says nothing about which
feature columns exist today, where the walk-forward folds fall, or which
symbol-date pairs have both a feature row and a label. **Resolving** a request is
the step that goes and finds all of that:

```python
from case_studies.research import load_model_configs, model_requests, open_study

study = open_study("etfs")
configs = load_model_configs(study, "linear", labels=["fwd_ret_21d"])
requests = model_requests(study, configs)
resolved = tuple(request.resolve() for request in requests)
```

`load_model_configs` walks the menu, loads each preset, and returns the declared
population as a Polars frame. `model_requests` turns each row into a request.
`resolve()` reads the label and feature files, computes the fold boundaries from
the case study's walk-forward parameters, works out the rows the fit is expected
to predict, and substitutes any hyperparameter that is defined relative to the
data - a penalty expressed as a fraction of a fold-specific quantity becomes the
actual number used on each fold. The result is a **resolved specification**: a
complete, self-contained description of one computation, with nothing left to
look up.

Resolving is a read of the inputs and costs seconds. Nothing is fitted and
nothing is written, so a notebook can resolve its whole population and show the
reader what each member will do before any of it runs.

## Identity

The hash of a resolved specification's canonical JSON is the run's identity.
`canonical_json` sorts keys and strips whitespace, so two specifications that
describe the same computation hash identically regardless of how they were
built.

### What the identity is computed from

Everything that can change the numbers:

| In the identity | Why it belongs there |
|---|---|
| Label name and the content digest of the label file | Relabeling the same symbols changes what the model is fitting |
| Feature file digests and the exact list of feature columns | Adding, dropping or rebuilding a feature changes the design matrix |
| Task type and, for classification, the class values | The same features and label can be fitted as regression or as classification |
| Every fold's train and validation start and end date | Two runs on different windows are different experiments |
| Estimator class and the parameters it was constructed with, per fold | A data-dependent penalty resolves to a different number on each fold, and each of those numbers is recorded |
| Preprocessing: the imputer and the scaler, with their settings | Median imputation and mean imputation give different fits |
| Checkpoint schedule | A GBM registering predictions every 50 trees produces a different set of results from one registering only the last |
| A digest of the exact (symbol, timestamp, fold) rows the run must predict, with the row and fold counts | This is the eligibility manifest, and it is what makes coverage part of identity rather than an afterthought |
| Random seed | Two seeds are two experiments |
| What the family's runner computes - a declared version for linear and GBM, a content digest of the source files for the other four - plus the versions of the numerical libraries it calls | Fitting code and library versions change results |
| Whether the run is canonical or a preview, and every reduction a preview applies | A run on a fifth of the symbols is not the same result as a full one |

### What it deliberately leaves out

Which notebook launched the run, the git commit of the repository at the time,
when the row was written, and how long the fit took. These are stored beside the
row as provenance and are visible in the registry, but they are not hashed. Two
runs that differ only in these are the same result, and the run log should say so
rather than accumulate a second copy every time a notebook is re-run from a new
commit.

### What follows from it

**Re-running a notebook is cheap.** Every identity is re-derived from the inputs,
the registry already holds those rows, and the runner returns the stored result
instead of fitting again. An interrupted sweep resumes where it stopped, and the
cost of an interruption is the one configuration that was in flight. This is also
true one level down: a fit that completed folds 0 through 4 before failing
reuses those five folds and fits only the rest.

**Changing a configuration adds a result, it does not replace one.** Edit a
preset, add a feature, move a fold boundary, and the affected runs resolve to new
identities and register as new rows. The old rows stay exactly where they were,
still complete and still queryable, so an experiment accumulates alongside the
published results rather than overwriting them.

**What an edit to a runner costs depends on which runner.** Two schemes are in
use, and they differ in what a change to the fitting code does to the results
already registered.

*Linear and GBM declare a version.* `LINEAR_RUNNER_VERSION`,
`GBM_RUNNER_VERSION`, `FOLD_PREPARATION_VERSION` for the shared fold preparation,
and `PREPROCESSING_ID` / `GBM_PREPROCESSING_ID` for the cast applied to a fold are
what enter the identity. A comment, a log line or a refactoring that moves code
without changing what it computes therefore leaves every registered result valid,
and only bumping the version refits the family. The bump is the decision about
compute, and it is required whenever the change would move the numbers, because a
cache that survived such a change would be lying. What holds the declaration
honest is `tests/test_linear_identity.py`, `tests/test_gbm_identity.py` and
`tests/test_folds.py`: each pins the quantities its version claims to describe and
fails when they move without a bump.

*Tabular deep learning, sequence models, latent factors and causal inference
digest their source.* The SHA-256 of each of a set of files is in the identity, so
editing any file in that set - including a change that alters nothing a model
computes - refits the configurations it covers on the next run. Only causal
digests a single file; the others digest several, and latent factors digests
fourteen, including `utils/modeling.py`. Read the set from the function that
builds it rather than from a list here, which would drift:

| Family | Its digest set |
|---|---|
| Tabular DL | `_tabm_source_identity` in `case_studies/utils/tabular_dl.py` |
| Sequence models | `_sequence_source_identity` in `case_studies/utils/deep_learning.py` |
| Latent factors | `_source_identity` in `case_studies/utils/latent_factors/adapter.py` |
| Causal | `_causal_source_identity` in `case_studies/utils/causal.py` |

Sequence models are the one case where the blast radius is narrower than the
family: the set includes the selected architecture's own module, so editing that
refits only the configurations that select it. Everywhere else, an edit inside the
set refits the whole family, and is a decision about compute as well as about
code. The declared-version scheme replaced this for linear and GBM first because
those two are re-run most often; the remaining four have not been converted.

The two identities downstream of a training run are built from it rather than
from scratch; the next section gives their exact composition.

## The three-level entity model

Every supervised model run flows through three levels, each identified by a
12-character SHA-256 hash:

```
training_run ─────► prediction_set ─────► backtest_run
   spec                predictions          daily_returns
```

| Level             | Identity input                                    | Key artifact                       |
|-------------------|---------------------------------------------------|------------------------------------|
| `training_run`    | `canonical_json(spec)`                            | `training/{hash}/spec.json`        |
| `prediction_set`  | `training_hash + checkpoint + split`              | `predictions/{hash}/predictions.parquet` |
| `backtest_run`    | `prediction_hash + canonical_json(strategy_spec)` | `backtest/{hash}/...`              |

The training-to-prediction relationship is **one-to-many**. A single training
run can produce multiple prediction sets, one per `(checkpoint, split)`
combination:

- **Different splits**: the same trained model is scored on the validation
  walk-forward and on the held-out tail.
- **Different checkpoints**: GBM and DL models register predictions at multiple
  intermediate states (every 50 trees, every 5 epochs) so a single fit produces
  a learning curve.
- **Different recompute passes**: re-evaluating predictions on the same model
  (e.g., to add a new metric) does not require retraining.

The prediction-to-backtest relationship is also one-to-many. A single set of
walk-forward predictions feeds multiple strategy variants — different top-K
selections, allocation methods, cost regimes, or risk overlays — each producing
its own backtest run.

### Hash computation in detail

```python
training_hash   = SHA256(canonical_json(training_identity))[:12]
prediction_hash = SHA256(canonical_json({
    "training_hash": ..., "split": ..., "checkpoint_kind": ...,
    "checkpoint_value": ..., "identity_version": ...,
}))[:12]
backtest_hash   = SHA256(canonical_json({
    "prediction_hash": ..., "strategy": strategy_spec, "identity_version": ...,
}))[:12]
```

`identity_version` is part of every hash, so a change to what an identity is
computed from is itself versioned: results registered under an earlier scheme
stay readable and stay distinguishable from results registered under the current
one, instead of silently colliding with them.

The hash space is 12 hex characters (48 bits, ≈ 2.8 × 10¹⁴ values), well above
the working set in any single case study. Collisions are not a practical concern
at this scale.

## The causal side table

Causal DML estimates do not fit the supervised flow above. They produce a single
treatment-effect estimate per `(case_study, label, treatment, confounders)`
configuration with no per-asset prediction set and no downstream backtest. They
live in their own table — `causal_runs` — keyed by a `causal_hash` derived from
the same content-addressed scheme.

A causal run records the estimated treatment effect, its HAC standard error and
p-value, a naive (uncontrolled) comparison effect, the implied confounding bias
percentage, and a refutation-test p-value. Chapter 15 reads from this table
directly.

## Filesystem layout

```
case_studies/{case_study}/
├── config/
│   ├── setup.yaml                        Case study definition
│   ├── training/{label}.yaml             Training menu per label
│   └── cv/cv_config.json                 Walk-forward parameters
└── run_log/
    ├── registry.db                       SQLite index
    ├── training/{training_hash}/
    │   ├── spec.json                     Canonical training spec (identity)
    │   └── coefficients.parquet | model.pt | ...   Family-specific weights
    ├── predictions/{prediction_hash}/
    │   └── predictions.parquet           timestamp, symbol, fold, y_true, y_score
    └── backtest/{backtest_hash}/
        ├── spec.json                     Canonical strategy spec
        ├── daily_returns.parquet         Net daily strategy returns (SSOT)
        ├── equity.parquet                Cumulative equity
        ├── weights.parquet               Per-asset weights at each rebalance
        ├── trades.parquet                Trade list
        ├── fills.parquet                 Executed fills with cost attribution
        └── portfolio_state.parquet       Snapshots of position / cash / margin
```

`registry.db` is the catalog. The `training/`, `predictions/`, and `backtest/`
subdirectories hold the heavy artifacts each run produced. The catalog is small
(a few megabytes per case study); the artifacts are large (gigabytes for case
studies with high-cardinality intraday or panel data).

The shapes that matter most to readers:

- `predictions.parquet` is in long format — one row per `(timestamp, symbol,
  fold)` — with columns `y_true` (the realized continuous label) and `y_score`
  (the model's prediction). This format is consumed directly by Chapter 16's
  `Engine` to drive a backtest.
- `daily_returns.parquet` is the canonical strategy P&L: one row per trading
  day with the net (post-cost) return. It is the SSOT for every Sharpe, drawdown,
  and equity curve in Chapters 16-20.

## Database schema

The eight tables below are the ones a reader queries: the runs, the predictions
they produced, and the metrics computed from them. `registry.db` holds fourteen
more that the interface maintains on the notebook's behalf - the official
populations, the per-attempt execution ledger that makes an interrupted fit
resumable, the candidate sets and decision artifacts Chapters 16 to 19 write, and
the lock that permits the holdout to be scored once. All nine case studies share
an identical schema, with one intentional exception noted below.

### `training_runs`

One row per supervised training run.

| Column          | Type | Description                                                   |
|-----------------|------|---------------------------------------------------------------|
| `training_hash` | TEXT | 12-char SHA-256 of the spec (primary key)                     |
| `family`        | TEXT | `linear`, `gbm`, `tabular_dl`, `deep_learning`, `latent_factors` |
| `label`         | TEXT | Target variable, e.g. `fwd_ret_21d`                           |
| `config_name`   | TEXT | Preset name, e.g. `leaves_15_huber`                           |
| `spec_json`     | TEXT | Full canonical-JSON spec                                      |
| `created_at`    | TEXT | ISO 8601 UTC timestamp of registration                        |
| `git_commit`    | TEXT | Short commit hash of the producing code                       |
| `entry_point`   | TEXT | Notebook that produced this run                               |
| `started_at`    | TEXT | ISO 8601 UTC timestamp of training start                      |
| `elapsed_s`     | REAL | Wall-clock training time in seconds                           |

### `prediction_sets`

One row per prediction set produced from a training run.

| Column             | Type    | Description                                                |
|--------------------|---------|------------------------------------------------------------|
| `prediction_hash`  | TEXT    | 12-char SHA-256 (primary key)                              |
| `training_hash`    | TEXT    | Foreign key to `training_runs`                             |
| `checkpoint_value` | INTEGER | Trees, epochs, or NULL for `final`                         |
| `checkpoint_kind`  | TEXT    | `tree_limit`, `epoch`, or `final`                          |
| `split`            | TEXT    | `validation` or `holdout`                                  |
| `created_at`       | TEXT    | ISO 8601 UTC timestamp                                     |

### `prediction_metrics`

Headline metrics aggregated across walk-forward folds, one row per prediction set.

| Column               | Type | Description                                                |
|----------------------|------|------------------------------------------------------------|
| `prediction_hash`    | TEXT | Primary key, foreign to `prediction_sets`                  |
| `computed_at`        | TEXT | ISO 8601 UTC timestamp                                     |
| `ic_mean`            | REAL | Mean cross-sectional Spearman IC over the folds with a defined IC |
| `ic_std`             | REAL | Std-dev of fold-level IC (0.0 when fewer than two folds define one) |
| `ic_t`               | REAL | Diagnostic fold-level t, `ic_mean / (ic_std / √n_folds_ic)`. **NULL when undefined** — fewer than two folds with an IC, or no dispersion across them. Read `ic_t_hac` for inference: it is the HAC t on the daily IC series and comes with `ic_ci_lo` / `ic_ci_hi` |
| `n_folds`            | REAL | Number of folds present in the prediction set              |
| `n_folds_ic`         | REAL | Number of those folds that produced a defined IC. Below `n_folds` means partial coverage: some fold scored constant and contributes to no IC statistic |
| `ic_n_days`          | REAL | Number of validation **dates** that produced a defined cross-sectional IC, the same coverage question at daily resolution. Below the maximum across a population means the row's `ic_mean` is averaged over a smaller and self-selected set of dates, and is not comparable to a full-coverage one. Written alongside the daily-IC uncertainty columns (`ic_se_hac`, `ic_ci_lo`, `ic_ci_hi`, `ic_t_hac`) when at least three dates define an IC |
| `pct_positive`       | REAL | Fraction of the folds with a defined IC whose IC > 0       |
| `task_type`          | TEXT | `'regression'` or `'classification'`                       |
| `accuracy`           | REAL | Classification: accuracy at threshold (NULL for regression) |
| `balanced_accuracy`  | REAL | Classification: balanced accuracy                          |
| `auc_roc`            | REAL | Classification: ROC AUC                                    |
| `auc_pr`             | REAL | Classification: precision-recall AUC                       |
| `log_loss`           | REAL | Classification: log loss (NULL when scores aren't probabilities) |
| `brier_score`        | REAL | Classification: Brier score (NULL when scores aren't probabilities) |

IC is computed against the continuous return — `y_true` in
`predictions.parquet` is always the continuous label, even on classification
prediction sets. This makes IC comparable across regression and classification
models trained on the same target horizon. AUC, log loss, and the accuracy
family are computed against the binary or three-class label that the model was
trained on.

### `fold_metrics`

Per-fold breakdown of `prediction_metrics`.

| Column            | Type | Description                                                |
|-------------------|------|------------------------------------------------------------|
| `prediction_hash` | TEXT | Foreign to `prediction_sets`                               |
| `fold_id`         | INTEGER | Fold index (0-based)                                    |
| `computed_at`     | TEXT | ISO 8601 UTC timestamp                                     |
| `ic`              | REAL | Cross-sectional Spearman IC for this fold                  |
| `ic_std`          | REAL | Within-fold IC dispersion across rebalance dates           |
| `n_entities`      | REAL | Distinct entities (symbols) in this fold                   |
| `rmse`, `mae`     | REAL | Regression metrics                                         |
| `accuracy`, `balanced_accuracy`, `auc_roc`, `auc_pr`, `log_loss`, `brier_score` | REAL | Classification metrics |
| `auc_class_-1`, `auc_class_0`, `auc_class_1` | REAL | One-vs-rest AUC for three-class direction labels |

Primary key: `(prediction_hash, fold_id)`.

### `causal_runs`

One row per causal-effect estimate.

| Column                 | Type    | Description                                            |
|------------------------|---------|--------------------------------------------------------|
| `causal_hash`          | TEXT    | 12-char SHA-256 (primary key)                          |
| `label`                | TEXT    | Target variable                                        |
| `treatment`            | TEXT    | Treatment variable name                                |
| `confounders_json`     | TEXT    | Sorted JSON list of confounders                        |
| `embargo`              | INTEGER | Embargo periods between train and evaluation           |
| `n_folds`, `n_obs`     | INTEGER | Cross-fitting folds, total observations                |
| `dml_effect`           | REAL    | Estimated treatment effect (DML)                       |
| `dml_se_hac`           | REAL    | HAC standard error of the DML estimate                 |
| `p_value_hac`          | REAL    | HAC p-value                                            |
| `naive_effect`         | REAL    | Effect from a controls-free regression                 |
| `confounding_bias_pct` | REAL    | Implied confounding bias as a percentage of naive      |
| `refutation_p`         | REAL    | Refutation-test p-value                                |
| `spec_json`            | TEXT    | Canonical-JSON causal spec                             |
| `notebook`             | TEXT    | Notebook that produced this run                        |
| `started_at`           | TEXT    | ISO 8601 UTC timestamp of estimation start             |
| `elapsed_s`            | REAL    | Wall-clock estimation time in seconds                  |
| `git_commit`           | TEXT    | Short commit hash                                      |
| `created_at`           | TEXT    | ISO 8601 UTC timestamp of registration                 |

### `backtest_runs`

One row per `(prediction_set, strategy_spec)` pair.

| Column            | Type | Description                                                |
|-------------------|------|------------------------------------------------------------|
| `backtest_hash`   | TEXT | 12-char SHA-256 (primary key)                              |
| `prediction_hash` | TEXT | Foreign to `prediction_sets`                               |
| `spec_json`       | TEXT | Canonical-JSON strategy spec                               |
| `stage`           | TEXT | `signal`, `allocation`, `cost_sensitivity`, or `risk_overlay` |
| `created_at`      | TEXT | ISO 8601 UTC timestamp                                     |
| `git_commit`      | TEXT | Short commit hash                                          |
| `started_at`      | TEXT | ISO 8601 UTC timestamp of backtest start                   |
| `elapsed_s`       | REAL | Wall-clock backtest time in seconds                        |

The `stage` column tags which book chapter produced the backtest:

- `signal` — Chapter 16: equal-weight top-K from the prediction signal.
- `allocation` — Chapter 17: portfolio-construction overlays
  (mean-variance, hierarchical risk parity, inverse volatility, RL).
- `cost_sensitivity` — Chapter 18: backtests run across a grid of cost
  parameters to map the cost-frontier.
- `risk_overlay` — Chapter 19: backtests with volatility targeting,
  drawdown control, or other risk-management overlays.

### `backtest_metrics` and `backtest_fold_metrics`

Standard backtest performance metrics aggregated across the full sample
(`backtest_metrics`) and broken down per walk-forward fold
(`backtest_fold_metrics`):

`sharpe`, `sortino`, `total_return`, `max_drawdown`, `cagr`, `volatility`,
`calmar`, `omega`, `stability`, `tail_ratio`, `win_rate`, `kurtosis`,
`skewness`, `var_95`, `cvar_95`, `n_periods` (or `n_days` per-fold),
`num_trades`, `total_commission`, `total_slippage`, `avg_turnover`.

The `sp500_options` case study extends `backtest_metrics` with seven options-
specific cohort-accounting columns (`mean_daily_return`, `cumulative_entry_cost`,
`cumulative_exit_cost`, `cumulative_hedge_cost`, `avg_cohorts_open`,
`cohort_days_open`, `n_rebalance_dates`) because its hold-to-maturity short
straddle strategy uses overlapping daily mark-to-market cohorts.

## Querying the run log

The Python API in `case_studies.utils.registry` is the canonical way to read
the run log:

```python
from case_studies.utils.registry import (
    load_training_runs,
    load_prediction_sets,
    load_prediction_metrics,
    read_training_spec,
    read_predictions,
    read_backtest_returns,
)

# All GBM training runs for ETFs
runs = load_training_runs("etfs", family="gbm")

# Headline metrics for one prediction set
metrics = load_prediction_metrics("etfs", prediction_hash="3fd4fec94687")

# Fetch the full spec from disk
spec = read_training_spec("etfs", "3040ebdc3ea4")

# Fetch the predictions parquet directly
predictions = read_predictions("etfs", "3fd4fec94687")

# Fetch the daily returns of a backtest
returns = read_backtest_returns("etfs", backtest_hash="9c1f7d40bea2")
```

The higher-level analytics module wraps common queries:

```python
from case_studies.utils.analytics import (
    load_model_ic,
    load_classification_metrics,
    load_best_ic_per_family,
)

# Validation IC across all linear/GBM/DL/LF runs for a case study
ic_df = load_model_ic(case_studies=["etfs"], split="validation")

# Just the classification rows
cls_df = load_classification_metrics(case_studies=["etfs"], split="validation")
```

## How a model notebook writes to the run log

Every notebook that fits a declared model population uses the same five calls,
whatever the family:

```python
from case_studies.research import (
    load_model_configs, model_requests, open_study,
    resolved_model_plan, run_model_population,
)

study = open_study("etfs")                                    # which copy to write to
configs = load_model_configs(study, "linear", labels=["fwd_ret_21d"])
requests = model_requests(study, configs)
resolved = tuple(request.resolve() for request in requests)   # bind to data, folds, rows
resolved_model_plan(resolved)                                 # inspect before running
execution, population = run_model_population(
    study, resolved, population_name="etfs-linear-validation-v1"
)
```

`run_model_population` fits each resolved request in turn. For one request it
fits every fold, concatenates the fold predictions, writes a `training_runs` row
and the fitted coefficients or weights under `training/{hash}/`, writes a
`prediction_sets` row and `predictions/{hash}/predictions.parquet`, and computes
the metrics that go into `prediction_metrics` and `fold_metrics`. It does this
per configuration, not once at the end, so an interrupted run keeps everything
that had already finished.

`execution.catalog_rows` returns those registry rows as a Polars frame, which is
what the notebook displays. `execution.diagnostics` says, per configuration,
whether it was fitted or served from the registry, and which folds were reused.

### Populations

A **population** is a named, immutable list of prediction identities. The
identities can be computed from the resolved specifications alone, so
`run_model_population` writes the whole list down *before* the first fit, and
checks afterwards that exactly those predictions exist and are complete.

That check is what makes a downstream comparison well defined. Chapter 16
backtests a population, not "whatever predictions happen to be in the registry",
so a sweep that half-finished cannot be silently compared against one that ran to
completion. If a configuration raises, the call raises: nothing is published
under the population name, the configurations that did finish stay registered,
and re-running fits only what is missing.

A population name refers to one set of members forever. Running a different set
of configurations under a name that already exists raises rather than redefining
what the name means.

## Running your own configurations

The installed run log is read-only. To add runs, write to a copy: a workspace
holds its own `registry.db` and its own artifacts, reads the same labels and
features, and cannot touch the published copy.

```python
study = open_study("etfs", workspace="~/ml4t-experiments")
```

From there, four changes, in increasing order of how much they change:

1. **Fit a subset of what is already declared.** Pass
   `config_names=["ols", "ridge_a1.0"]` to `load_model_configs`. A name the
   label's training menu does not declare raises, rather than quietly fitting a
   smaller population than you asked for.
2. **Add a configuration.** Write a preset at
   `case_studies/config/{model_type}/{name}.yaml` holding the hyperparameters,
   and add `{name}` to the family's list in
   `case_studies/{case_study}/config/training/{label}.yaml`. `family` and
   `library` come from the directory, so a file in `config/lgb/` is a LightGBM
   configuration by construction.
3. **Change an existing configuration.** Edit its preset. The resolved
   specification changes, so the result registers under a new hash beside the old
   one.
4. **Fit a different label.** Each label has its own training menu; a label with
   no menu file has no declared population for any family.

Give your run its own `population_name`. Everything downstream reads the
registry rather than the notebook, so predictions produced this way participate
in the same selection and backtesting as the published ones, inside your
workspace.

For a cheap rehearsal, open the study with `execution_tier="preview"` and declare
what makes it cheap - `preview_reductions={"max_symbols": 20,
"train_sample_frac": 0.25}`. A preview writes under `.preview/` in the workspace
and is barred from populations, so a reduced run cannot be mistaken for a full
one.

The command-line route to the same thing, including how to produce the
`features/` and `labels/` a model stage needs, is
[Experimenting Without Changing the Release Baseline](../docs/running-notebooks.md#experimenting-without-changing-the-release-baseline).

## Scale

Across the nine case studies, the run log currently catalogs roughly:

- 880 training runs
- 1,070 prediction sets
- 22 causal-effect estimates
- 15,140 backtest runs (across the four stages)
- 7,400 fold-level prediction-metric rows
- 110,900 fold-level backtest-metric rows

Each case study's `registry.db` is a few megabytes; the on-disk artifacts under
`training/`, `predictions/`, and `backtest/` are gigabytes for high-cardinality
case studies (intraday NASDAQ-100, daily US equity panel) and hundreds of
megabytes for the smaller ones.

## Distribution as release artifacts

Most readers will not retrain every model from scratch. Each release bundle contains the accepted
registry plus every registered training, prediction, and backtest artifact for that case study. This
lets Chapters 11-20 consume the same stored inputs without mixing registry vintages.

After installing the repo, run:

```bash
uv run python scripts/download_artifacts.py
```

The downloader verifies the archive checksum, every internal artifact checksum, SQLite integrity,
and foreign keys before installing `case_studies/{case_study}/run_log/`. It installs the run log as a
read-only baseline. Create a writable experiment before adding runs:

```bash
uv run python scripts/create_experiment.py \
  --cs etfs \
  --output /tmp/ml4t-etf-experiment
```

To download artifacts for a single case study:

```bash
uv run python scripts/download_artifacts.py --cs etfs
```

The full installation and experiment procedures are documented in
[`docs/running-notebooks.md`](../docs/running-notebooks.md).
