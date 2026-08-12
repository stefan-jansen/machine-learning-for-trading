# What This Repository Is, and What It Is Not

This page states what you can do with this code before you invest time in it. Four levels, from
"works today with one command" to "not promised at all". Read it once; it will save you from
planning work the repository cannot support.

If you only read one line: **the case-study results behind the book download, so you can reach them
without training anything.** Chapter notebooks are different - many train their own models on the
spot, and that is the point of them. Everything beyond those two things costs progressively more.

---

## 1. Reproduce directly

No training, no GPU, no paid data.

**Chapter notebooks.** Most of the 27 chapters run on the free datasets:

```bash
uv run python data/download_all.py --free-only
uv run python 01_process_is_edge/factor_regimes.py
```

The free download includes the roughly 1.5 GB firm-characteristics panel; add
`--skip-firm-characteristics` to leave it out and the core ETF, crypto, and factor data comes to
about 70 MB. Some notebooks need a specialized Docker image, and some need a licensed dataset (see
level 3); each one says so in its preamble. See [Docker environments](../envs/README.md) and the
[data guide](../data/README.md).

**Published case-study results.** The artifact release ships the complete run logs behind the book's
tables and figures, so the analysis notebooks read real registered results instead of retraining to
get them. Some of those notebooks still recompute downstream quantities - cost cascades, risk
overlays, portfolio statistics - from the downloaded artifacts, so a figure there can differ from the
printed one even though nothing was retrained. See "One thing to expect when you re-run" below.

```bash
uv run python scripts/download_artifacts.py --cs etfs   # 33 MB
uv run python scripts/download_artifacts.py             # all nine, about 3.1 GB
```

What that gets you, counted from the release itself:

| | Training runs | Prediction sets | Backtests |
|---|---:|---:|---:|
| All nine case studies | 860 | 1,119 | 11,099 |
| `etfs` alone | 53 | 128 | 759 |

Those artifacts are the input to the model-analysis and strategy-analysis stages, which is the
shortest path from a clean clone to a real result:

```bash
uv run python scripts/download_artifacts.py --cs etfs
uv run python case_studies/etfs/13_model_analysis.py
uv run python case_studies/etfs/18_strategy_analysis.py
```

## 2. Modify with reasonable effort

Changing what a case study *does* is a configuration edit, not a code edit. The model notebooks hold
no hyperparameter grids; they read three files:

| File | Controls |
|---|---|
| `case_studies/{cs}/config/setup.yaml` | universe, labels, decision cadence, costs, backtest sweep, GBM device |
| `case_studies/{cs}/config/training/{label}.yaml` | which presets run, per model family |
| `case_studies/config/{model_type}/{preset}.yaml` | the actual hyperparameters |

For gradient boosting and the linear families, adding a preset file and listing its name in the
training menu is enough: the next run trains it and registers it under a new hash that the analysis
notebooks pick up alongside the book's own runs. Do this inside a writable experiment so the
downloaded release stays untouched.

The deep, tabular-deep, latent-factor, and causal stages dispatch more narrowly - they select on
architecture or on fixed model names, and a listed preset they do not recognize is skipped rather
than trained. Check the stage you are targeting before assuming a new preset runs.

The full procedure, including which stages override a preset value with their own constant, is in
[running notebooks](running-notebooks.md#experimenting-without-changing-the-release-baseline). That
override table matters: a few stages ignore parts of the preset you edited, and the list names them.

Reasonable effort here means minutes to hours of compute for a single family on one case study, not
a full pipeline.

## 3. Needs substantial infrastructure

These are supported and documented, but they are not a weekend exercise. Plan for them.

**A full end-to-end retrain of a case study.** The published run logs on the authoring machine total
73 GB across the nine, and two of them dominate:

| Case study | Registered backtests | Run-log size |
|---|---:|---:|
| `nasdaq100_microstructure` | 12,531 | 28 GB |
| `us_equities_panel` | 1,313 | 30 GB |
| `sp500_equity_option_analytics` | 2,931 | 7.7 GB |
| `etfs` | 2,415 | 2.5 GB |
| the remaining five | 525 to 2,536 each | 224 MB to 2.3 GB |

The downloadable bundles are a reduced selection of that, which is why `etfs` is 45 MB installed
rather than 2.5 GB. Reproducing the full sweep means the compute and the disk, over days.

**GPU work.** The deep-learning chapters and the deep and tabular-deep model families in the case
studies are written for a CUDA GPU. Most run on CPU, much more slowly. Some do not run on CPU at
all: the four `crypto_perps_funding` model notebooks (`07_gbm`, `08_tabular_dl`, `09_dl_lstm`,
`10_dl_tcn`) stop with an error when PyTorch cannot see a CUDA device, and the `us_firm_characteristics`
deep notebooks are marked GPU-only. Gradient boosting on GPU is also not bitwise reproducible against
a CPU run. See [installation](installation.md) for the GPU setup.

**AlgoSeek data.** Three case studies read AlgoSeek data: `nasdaq100_microstructure`,
`sp500_options` and `sp500_equity_option_analytics`. All four datasets they need are available with
no account, no API key and no license request. AlgoSeek publishes three of them openly — the
NASDAQ-100 minute bars and the S&P 500 option chains, which one script converts to what the loaders
read, and the NASDAQ-100 TAQ ticks, which only need unzipping. The fourth, the S&P 500 daily bars,
ships inside this repository by AlgoSeek's permission, so nothing is downloaded or configured for
it. See [AlgoSeek datasets](../data/README.md#algoseek-datasets). Everything else in those case
studies rebuilds from raw data.

## 4. Not promised

This is a teaching and demonstration repository that accompanies a book. It is not a research
platform and not a production trading system.

- No strategy here is offered as profitable, and several are shown failing on purpose because the
  failure is the lesson.
- Live trading appears in Chapter 25 as worked integrations against Interactive Brokers, Alpaca, and
  QuantConnect. They are demonstrations of the interfaces, not a deployment you should point at
  capital.
- There is no turnkey path from a notebook to a running strategy, no hosted service, and no support
  commitment.

---

## One thing to expect when you re-run

Some case-study numbers move slightly when you re-run them on your own machine. Library versions,
hardware, and CPU-versus-GPU execution all shift results at the margin, and gradient boosting on a
CUDA GPU is not bitwise deterministic. Point estimates drift; the conclusions in the book do not
depend on the last digit. Where a number in this repository differs from the printed book, the
repository is the current value and the book is the value at the time of writing.

If you find a difference large enough to change a conclusion, that is worth reporting: open an
[issue](https://github.com/stefan-jansen/machine-learning-for-trading/issues).
